import os
import os.path as osp
import sys
import time
import logging
import pathlib
import shutil
import torch
import random
import numpy as np


class BaseArgs:
    def __init__(self, parser):
        parser.add_argument('--task', type=str, default='JDnDm', help='task')
        parser.add_argument('--phase', type=str, default='test',
                            help='phase. Default: test')
        parser.add_argument('--seed', type=int, default=42)

        # datasets args
        parser.add_argument('--dataset', type=str, default='DIV2K',
                            help='path to train list')
        parser.add_argument('--train_path', type=str, default='dataset/DIV2K/train/DIV2K_train_HR_sub/',
                            help='path to train list')
        parser.add_argument('--val_path', type=str, default='dataset/DIV2K/valid/DIV2K_valid_HR_sub/',
                            help='path to val list')
        parser.add_argument('--num_workers', default=14, type=int)
        parser.add_argument('--patch_size', default=128, type=int,
                            help='width and height for a patch (default: 128); '
                                 'if performing joint DM and SR, then use 128.')

        parser.add_argument('--in_channels', default=4, type=int,
                            help='in_channels, RGB')
        parser.add_argument('--gt_channels', default=3, type=int,
                            help='gt_channels, RGB')
        parser.add_argument('--in_type', type=str, default='noisy_rgb',
                            help='the input image type: rgb, noisy_rgb'
                            )

        # train args
        parser.add_argument('--bs_per_gpu', default=32, type=int,
                            help='batch size per GPU (default:32)')
        parser.add_argument('--n_gpus', default=1, type=int,
                            help='number of GPUs (default:1)')
        parser.add_argument('--max_epochs', default=200, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=5e-4, type=float,
                            help='initial learning rate')
        parser.add_argument('--min_lr', default=1e-6, type=float,
                            help='min learning rate')

        # logger parse
        parser.add_argument('--root_dir', type=str, default='logs',
                            help='path for saving experiment files')
        parser.add_argument('--print_freq', default=1000, type=int,
                            help='show loss information every xxx iterations(default: 1000)')
        parser.add_argument('--eval_freq', default=1, type=int,
                            help='perform evaluation every xxx epochs(default: 1)')
        parser.add_argument('--epoch_freq', default=1, type=int,
                            help='save milestone epoch every 500 epochs (default: 1)')

        # model args
        parser.add_argument('--model', default='FFNet', type=str,
                            help='model type (default: tenet)')
        parser.add_argument('--channels', default=64, type=int,
                            help='channels')
        parser.add_argument('--n_blocks', default=6, type=int,
                            help='number of basic blocks')

        # for super-resolution
        parser.add_argument('--min_noise', default=0.0, type=float,
                            help='[min_noise, max_noise]')
        parser.add_argument('--max_noise', default=0.0784, type=float,
                            help='[min_noise, max_noise]')

        # test args
        parser.add_argument('--save_dir', type=str, default='results/',
                            help='path to save the test result')
        parser.add_argument('--test_path', type=str, default='dataset/JDnDm/test/',
                            help='test all datasets')
        parser.add_argument('--pretrain',
                            default=None,
                            type=str, help='path to pretrained model(default: none)')

        # log args

        args = parser.parse_args()

        args.batch_size = args.bs_per_gpu * args.n_gpus

        # file name
        args.filename = '-'.join(['model_' + args.model,
                                  'in_type_' + args.in_type,
                                  'C_' + str(args.channels),
                                  'B_' + str(args.batch_size),
                                  'Patch_' + str(args.patch_size),
                                  'Epoch_' + str(args.max_epochs)])

        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(self.args.seed)  # set seed

        # ===> generate log dir
        if self.args.phase == 'train':
            if self.args.pretrain is None:  # if is train, init
                self.args.filemode = 'w'
                self._generate_exp_directory()
                self._configure_logger()
                self._print_args()  # print args
            else:  # if is resume, reload
                self.args.filemode = 'a'
                self.args.exp_name = os.path.basename(os.path.dirname(os.path.dirname(self.args.pretrain)))
                self.args.exp_dir = os.path.dirname(os.path.dirname(self.args.pretrain))
                self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")
                self._configure_logger()
        elif self.args.phase == 'test':
            self.args.exp_name = os.path.basename(os.path.dirname(os.path.dirname(self.args.pretrain)))
            self.args.exp_dir = os.path.dirname(os.path.dirname(self.args.pretrain))
            self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")
            self.args.save_dir = os.path.join(self.args.exp_dir, self.args.save_dir)
            pathlib.Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)

    def _generate_exp_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """

        self.args.exp_name = '-'.join([self.args.filename, str(int(time.time()))])
        self.args.exp_dir = osp.join(self.args.root_dir, self.args.task, self.args.dataset, self.args.exp_name)
        self.args.ckpt_dir = osp.join(self.args.exp_dir, "checkpoint")
        pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevel))

        log_format = logging.Formatter('%(asctime)s %(message)s')

        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(osp.join(self.args.exp_dir,
                                                    '{}.log'.format(osp.basename(self.args.phase))),
                                           mode=self.args.filemode)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.basicConfig()
        logging.root = logger

        if self.args.pretrain and self.args.phase == 'train':
            logging.info("\n")
            logging.info("\n")
            logging.info("==========       continue train      =============")
        logging.info("save log path to: {}".format(self.args.exp_dir))

    def _print_args(self):
        logging.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            logging.info("{}: {}".format(arg, content))
        logging.info("==========     args END    =============\n")
        # logging.info("\n")
        logging.info('===> Phase is {}.'.format(self.args.phase))

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # set this to False, if being exactly deterministic is in need.
