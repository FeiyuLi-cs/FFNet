import argparse
import datetime
import shutil
import logging
import time

import torch
from torch.utils.data import DataLoader
from config.jdd_config import BaseArgs
from dataload.load_dataset import LoadJDDDIV2KData
from utils.metrics import PSNR, AverageMeter
from utils.model_util import load_pretrained_models, load_pretrained_optimizer
from torch.utils.tensorboard import SummaryWriter

from model import build_net


def main():
    logging.info('======> Creating dataloader...')
    train_set = LoadJDDDIV2KData(image_path=args.train_path, mode='train',
                                 patch_size=args.patch_size, in_type=args.in_type,
                                 min_noise=args.min_noise, max_noise=args.max_noise)
    val_set = LoadJDDDIV2KData(image_path=args.val_path, mode='valid',
                               patch_size=args.patch_size, in_type=args.in_type,
                               min_noise=args.min_noise, max_noise=args.max_noise)

    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)

    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True)

    # =================
    logging.info('======> Loading the network ...')
    # module = importlib.import_module("model.{}".format(args.model))
    # model = module.NET(args)
    model = build_net(args)
    # =================
    # load checkpoints
    start_epoch = 0

    if args.pretrain:
        model, best_psnr, start_epoch = load_pretrained_models(model, args.pretrain)
        start_epoch += 1

    # if args.n_gpus > 1:
    #     model = nn.DataParallel(model)
    # print(model)
    model = model.to(args.device)
    if args.pretrain is None:
        logging.info(model)

    # =================
    logging.info('======> Loading the Optimizers ...')

    L1_Loss = torch.nn.L1Loss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.min_lr)

    optimizer, scheduler = load_pretrained_optimizer(args.pretrain, optimizer, scheduler, args.device)

    # train + val
    logging.info('---------- Start training -------------\n')
    iters = len(train_loader)
    best_psnr = 0

    for epoch in range(start_epoch, args.max_epochs):
        # train
        losses = AverageMeter()
        l1_losses = AverageMeter()
        model.train()

        start_time = time.time()
        for i, (raw, gt) in enumerate(train_loader):

            raw = raw.to(args.device)
            gt = gt.to(args.device)

            batch_size = gt.size(0)
            output = model(raw)

            l1_loss = L1_Loss(output, gt)
            fft_loss = L1_Loss(torch.fft.rfft2(output), torch.fft.rfft2(gt))
            loss = l1_loss + 0.05 * fft_loss

            # zero parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), batch_size)
            l1_losses.update(l1_loss.item(), batch_size)

            if i % args.print_freq == 0:
                logging.info('Epoch: [{}]/[{}] Iter:[{}]/[{}]\t'
                             'l1_losses {l1_losses.val:.4f} '
                             'losses {losses.val:.4f}'.format(
                    epoch, args.max_epochs, i, iters, l1_losses=l1_losses, losses=losses))

        scheduler.step()
        logging.info('One training time: {:s}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        # loss_last = loss.avg
        writer.add_scalar('losses', losses.avg, epoch)
        writer.add_scalar('l1_losses', l1_losses.avg, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[-1], epoch)

        # val
        if epoch % args.eval_freq == 0 or epoch == args.max_epochs - 1:
            args.epoch = epoch
        cur_psnr = val(val_loader, model)
        is_best = (cur_psnr > best_psnr)
        best_psnr = max(cur_psnr, best_psnr)
        # model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_psnr': best_psnr,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, args=args)
        writer.add_scalar('eval_psnr', cur_psnr, epoch)

        logging.info('Saving the final model.\n')

    logging.info('training  done.\n')


def val(val_loader, model):
    psnrs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (raw, gt) in enumerate(val_loader):
            raw = raw.to(args.device)
            gt = gt.to(args.device)

            output = model(raw)

            # psnr
            mse = (output.clamp(0, 1) - gt).pow(2).mean()
            psnr = PSNR(mse)
            psnrs.update(psnr, gt.size(0))

    logging.info('Valid PSNR:{psnrs.avg: .4f}'.format(psnrs=psnrs))

    return psnrs.avg


def save_checkpoint(state, is_best, args):
    filename = '{}/{}_checkpoint_latest.pth'.format(args.ckpt_dir, args.filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_checkpoint_best.pth'.format(args.ckpt_dir, args.filename))
    if args.epoch % args.epoch_freq == 0:
        shutil.copyfile(filename, '{}/{}_checkpoint_epoch{}.pth'.format(args.ckpt_dir, args.filename, args.epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    writer = SummaryWriter(log_dir=args.exp_dir)
    main()
