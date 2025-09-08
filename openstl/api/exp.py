# Copyright (c) CAIRI AI Lab. All rights reserved

import sys
import time
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from openstl.methods import method_maps
from openstl.datasets.base_data import BaseDataModule
from openstl.utils import (get_dataset, measure_throughput, SetupCallback, EpochEndCallback, BestCheckpointCallback)

import argparse
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import seed_everything, Trainer
import pytorch_lightning.callbacks as plc


class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None, strategy='ddp'):
        if args.set_float32_matmul_precision:
            torch.set_float32_matmul_precision('high')
        if args.disable_find_unused_parameters:
            strategy = DDPStrategy(find_unused_parameters=False)

        """Initialize experiments (non-dist as an example)"""
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist

        base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name if not args.ex_name.startswith(args.res_dir) \
            else args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(args.seed)
        self.data = self._get_data(dataloaders)
        self.method = method_maps[self.args.method](steps_per_epoch=len(self.data.train_loader),
            test_mean=self.data.test_mean, test_std=self.data.test_std, save_dir=save_dir, **self.config)
        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        self.trainer = self._init_trainer(self.args, callbacks, strategy)

    def _init_trainer(self, args, callbacks, strategy):
        trainer_config = {
            'devices': args.gpus,  # Use the all GPUs
            'max_epochs': args.epoch,  # Maximum number of epochs to train for
            'max_steps': args.steps,
            "strategy": strategy, # 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_false'
            # "strategy": None,
            'accelerator': 'gpu',  # Use distributed data parallel
            'callbacks': callbacks,
            'check_val_every_n_epoch': args.check_val_every_n_epoch,
            'gradient_clip_val': args.gradient_clip_val if args.gradient_clip_val is not None else 0  # 0 means don't clip
        }
        return Trainer.from_argparse_args(argparse.Namespace(**trainer_config))

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._dist == 0:
            if not self.args.no_display_method_info:
                method_info = self.display_method_info(args)

        setup_callback = SetupCallback(
            prefix = 'train' if (not args.test) else 'test',
            setup_time = time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir = save_dir,
            ckpt_dir = ckpt_dir,
            args = args,
            method_info = method_info,
            argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
        )

        # ckpt_callback = BestCheckpointCallback(
        #     monitor=args.metric_for_bestckpt,
        #     filename='best-{epoch:02d}-{val_loss:.3f}',
        #     mode='min',
        #     save_last=True,
        #     dirpath=ckpt_dir,
        #     verbose=True,
        #     every_n_epochs=args.log_step,
        # )

        best_last_ckpt_callback = ModelCheckpoint(
            monitor=args.metric_for_bestckpt,
            filename='best-{epoch:02d}-{val_loss:.3f}',
            mode='min',
            save_last=True,
            dirpath=ckpt_dir,
            verbose=True,
            save_top_k=1,
            every_n_epochs=1,
        )

        every_n_epoch_ckpt_callback = ModelCheckpoint(
            monitor=args.metric_for_bestckpt,
            filename='{epoch:02d}-{val_loss:.3f}',
            mode='min',
            save_last=False,
            dirpath=ckpt_dir,
            verbose=True,
            save_top_k=-1,
            every_n_epochs=args.save_every_n_epoch,
        )
        
        epochend_callback = EpochEndCallback()

        callbacks = [setup_callback, best_last_ckpt_callback, every_n_epoch_ckpt_callback, epochend_callback]
        if args.sched:
            callbacks.append(plc.LearningRateMonitor(logging_interval=None))
        return callbacks, save_dir

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            train_loader, vali_loader, test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            train_loader, vali_loader, test_loader = dataloaders

        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)

    def train(self):
        self.trainer.fit(self.method, self.data, ckpt_path=None if len(self.args.continue_ckpt) == 0 else self.args.continue_ckpt)

    def test(self):
        if self.args.test == True:
            if self.args.continue_ckpt is not None and self.args.continue_ckpt != '':
                ckpt_path = self.args.continue_ckpt
                ckpt = torch.load(ckpt_path)
                self.method.load_state_dict(ckpt['state_dict'])
            else:
                ckpt = torch.load(osp.join(self.save_dir, 'checkpoints', 'best.ckpt'))
                self.method.load_state_dict(ckpt['state_dict'])
        self.trainer.test(self.method, self.data)
    
    def display_method_info(self, args):
        """Plot the basic infomation of supported methods"""
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            device = torch.device(assign_gpu)
        if args.method in ['fsae']:
            T, D = args.in_shape
            x = torch.randn(1, T, D, device=device)
            x_time = torch.randn(1, T, 5, device=device)
            input_dummy = ('enc&dec', x, x_time)
        elif args.method in ['fsaemspredictor']:
            T, D = args.in_shape[0], (args.d_model if hasattr(args, 'd_model') else args.d_in)
            branches_encoder = [torch.randn(1, T, D, device=device) for _ in range(args.num_branches)]
            pred_time = torch.randn(1, T, 5, device=device)
            input_dummy = (branches_encoder, pred_time)
        else:
            raise ValueError(f'Invalid method name {args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model.to(device), input_dummy)
        flops = flop_count_table(flops)
        if args.fps:
            fps = measure_throughput(self.method.model.to(device), input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line