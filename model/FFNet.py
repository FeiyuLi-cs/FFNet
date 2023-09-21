import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from model.model_util import LayerNorm2d


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class CA(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(CA, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.ca(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            CA(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Attention(nn.Module):
    def __init__(self, dim, head, H, W, num_res):
        super(Attention, self).__init__()

        layers = [AttBlock(dim, head, H, W) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.layers(x)
        out = self.conv(out)
        return out + x


class AttBlock(nn.Module):
    def __init__(self, dim, head, H, W):
        super(AttBlock, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.FFB = FourierFilterBlock(dim=dim, head=head, H=H, W=W)

        self.norm2 = LayerNorm2d(dim)
        self.CAB = CAB(num_feat=dim)

        # self.norm3 = LayerNorm2d(dim)
        # self.MLP = nn.Sequential(
        #     nn.Conv2d(dim, dim * head, kernel_size=1, padding=0, stride=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim * head, dim, kernel_size=1, padding=0, stride=1)
        # )

    def forward(self, x):
        out = self.norm1(x)
        out = self.FFB(out)
        out = out + x

        out = self.norm2(out)
        out = self.CAB(out)

        # res2 = out
        # out = self.norm3(out)
        # out = self.MLP(out)
        # out = out + res2
        return out


class GlobalFourierBlock(nn.Module):
    def __init__(self, dim, H, W):
        super(GlobalFourierBlock, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, H, W // 2 + 1, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight = nn.Parameter(torch.randn(dim, 64, 33, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = self.complex_weight

        if not weight.shape[1:3] == x.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())
        x = x * weight
        x = torch.fft.irfft2(x, dim=(2, 3), norm='ortho')
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class LocalFourierBlock(nn.Module):
    def __init__(self, dim):
        super(LocalFourierBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x_f = torch.cat([x.real, x.imag], dim=1)
        x_f = self.conv(x_f)
        x_real, x_imag = x_f.chunk(2, dim=1)
        x_f = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x_f, dim=(2, 3), norm='ortho')
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class FourierFilterBlock(nn.Module):
    def __init__(self, dim, head, H, W):
        super(FourierFilterBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim * head, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
        )
        self.GFB = GlobalFourierBlock(dim, H, W)
        self.LFB = LocalFourierBlock(dim)
        self.conv1 = nn.Conv2d(dim * head, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.conv(x)

        x1, x2 = out.chunk(2, dim=1)
        x1 = self.GFB(x1)
        x2 = self.LFB(x2)

        out = torch.cat([x1, x2], dim=1)
        out = self.conv1(out)

        return out


class FFNet(nn.Module):
    def __init__(self, in_channel, img_size):
        super(FFNet, self).__init__()
        dim = 64
        head = 2
        H = img_size
        W = img_size
        self.init_conv = nn.Conv2d(in_channel, dim, kernel_size=3, padding=1, stride=1)

        att_group = [Attention(dim, head, H // 2, W // 2, num_res=6) for _ in range(6)]
        self.body = nn.Sequential(*att_group)
        self.conv_after_body = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.up = Upsample(dim)
        self.out = nn.Conv2d(dim, 3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.init_conv(x)
        res = out
        out = self.body(out)
        out = self.conv_after_body(out)
        out = out + res

        # res
        out = self.conv_before_upsample(out)
        out = self.up(out)
        out = self.out(out)
        return out


if __name__ == '__main__':
    model = FFNet(in_channel=4, img_size=128)
    # print(model)
    # out = model(torch.randn(1, 4, 64, 64))
    # print('out.shape:', out.shape)
    for name, param in model.named_parameters():
        print(name, param.nelement())
    print(sum([param.nelement() for param in model.parameters()]))
    macs, params = get_model_complexity_info(model, (4, 64, 64), verbose=False, print_per_layer_stat=False)
    print(macs, params)
    # 6332055
    # 14395091
