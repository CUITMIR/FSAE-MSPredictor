from typing import Union, List

import numpy as np
import torch.nn as nn
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from timm.scheduler import CosineLRScheduler, TanhLRScheduler

from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler
from openstl.core import metric


class Base_method(pl.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch" if by_epoch else "step"
                },
            }
        else:
            return {
                "optimizer": optimizer
            }

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        if isinstance(scheduler, (CosineLRScheduler, TanhLRScheduler)):
            scheduler.step(self.current_epoch)
        else:
            scheduler.step()

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)

        eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                             self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list,
                             channel_names=self.channel_names, spatial_norm=self.spatial_norm)

        for k, v in eval_res.items():
            self.log(f'val_{k}', v, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        return {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}

    def test_epoch_end(self, outputs):
        results_all = {}
        for k in outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm)

        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
        return results_all