# add _init__.py, so the folder is viewed as package
from model.FFNet import FFNet


def build_net(args):
    if args.model == 'FFNet':
        return FFNet(in_channel=args.in_channels, img_size=args.patch_size)
    if args.model == 'FFNet-DM-B':
        return FFNet(in_channel=args.in_channels, img_size=args.patch_size)


def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total
