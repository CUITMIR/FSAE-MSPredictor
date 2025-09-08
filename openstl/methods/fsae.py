import datetime

import torch

from openstl.core import metric
from openstl.methods.base_method import Base_method
from openstl.models.fsae_model import FSAE_model


class FSAE(Base_method):
    def __init__(self, **args):
        super().__init__(**args)

        self.dynamic_coefficient = self.hparams.fft_loss_coefficient * 0.1
        self.alpha = 0.8

    def _build_model(self, **args):
        return FSAE_model(**args)

    def forward(self, batch_x, batch_year, batch_month, batch_day, **kwargs):
        time = torch.stack([
            self.make_mark(year, month, day, self.hparams.in_shape[0])
        for year, month, day in zip(batch_year, batch_month, batch_day)], dim=0).to(batch_x.device)

        return self.model('enc&dec', batch_x, time)

    def make_mark(self, year, month, day, time_steps):
        out = []
        cur = datetime.datetime(year, month, day)
        for i in range(time_steps):
            step = torch.tensor([
                (cur.minute - 1) / 59 - 0.5,
                (cur.hour - 1) / 23 - 0.5,
                cur.weekday() / 6 - 0.5,
                (cur.day - 1) / 30 - 0.5,
                (cur.timetuple().tm_yday - 1) / 365 - 0.5
            ])
            out.append(step)
            cur += datetime.timedelta(minutes=10)
        return torch.stack(out, dim=0)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_year, batch_month, batch_day = batch
        estimated_x, branches_encoder = self(batch_x, batch_year, batch_month, batch_day)

        loss = 0
        if 'mse' in self.hparams.loss_types:
            loss_mse = self.criterion(estimated_x, batch_x)

            loss += loss_mse
            self.log('train_loss_mse', loss_mse, on_step=False, on_epoch=True, prog_bar=True)

        if 'fft2_abs' in self.hparams.loss_types:
            loss_fft2_abs = 0
            branches_encoder = [torch.log(torch.abs(torch.fft.fft2(branch)) + 1) for branch in branches_encoder]
            for i in range(len(branches_encoder)):
                for j in range(i + 1, len(branches_encoder)):
                    loss_fft2_abs -= self.criterion(branches_encoder[i], branches_encoder[j])

            self.log('train_loss_fft_org', loss_fft2_abs, on_step=False, on_epoch=True, prog_bar=True)
            if self.hparams.num_branches == 2:
                if self.current_epoch < 40:
                    if torch.abs(self.dynamic_coefficient * loss_fft2_abs) > 1 * loss_mse:
                        self.dynamic_coefficient *= 0.5
                elif self.current_epoch < 160:
                    if torch.abs(self.dynamic_coefficient * loss_fft2_abs) > 2 * loss_mse:
                        self.dynamic_coefficient *= 0.8
                else:
                    if torch.abs(self.dynamic_coefficient * loss_fft2_abs) > 1 * loss_mse:
                        self.dynamic_coefficient *= 0.2
                loss_fft2_abs = self.dynamic_coefficient * loss_fft2_abs
            elif self.hparams.num_branches == 3:
                if torch.abs(self.dynamic_coefficient * loss_fft2_abs) > 1.5 * loss_mse:
                    self.dynamic_coefficient *= 0.5
                loss_fft2_abs = self.dynamic_coefficient * loss_fft2_abs
            elif self.hparams.num_branches == 4:
                if torch.abs(self.dynamic_coefficient * loss_fft2_abs) > 1.5 * loss_mse:
                    self.dynamic_coefficient *= 0.5
                loss_fft2_abs = self.dynamic_coefficient * loss_fft2_abs
            elif self.hparams.num_branches == 5:
                if torch.abs(self.dynamic_coefficient * loss_fft2_abs) > 1.5 * loss_mse:
                    self.dynamic_coefficient *= 0.5
                loss_fft2_abs = self.dynamic_coefficient * loss_fft2_abs
            else:
                raise NotImplementedError
            loss += loss_fft2_abs
            self.log('train_loss_fft', loss_fft2_abs, on_step=False, on_epoch=True, prog_bar=True)
        elif 'fft_abs' in self.hparams.loss_types:
            loss_fft_abs = 0
            branches_encoder = [torch.log(torch.abs(torch.fft.fft(branch, dim=1)) + 1) for branch in branches_encoder]
            for i in range(len(branches_encoder)):
                for j in range(i + 1, len(branches_encoder)):
                    loss_fft_abs -= self.criterion(branches_encoder[i], branches_encoder[j])

            self.log('train_loss_fft_org', loss_fft_abs, on_step=False, on_epoch=True, prog_bar=True)
            if torch.abs(self.dynamic_coefficient * loss_fft_abs) > 1.5 * loss_mse:
                self.dynamic_coefficient *= 0.5
            loss_fft_abs = self.dynamic_coefficient * loss_fft_abs

            loss += loss_fft_abs
            self.log('train_loss_fft', loss_fft_abs, on_step=False, on_epoch=True, prog_bar=True)
        elif 'rfft_abs' in self.hparams.loss_types:
            loss_fft_abs = 0
            branches_encoder = [torch.log(torch.abs(torch.fft.rfft(branch, dim=1)) + 1) for branch in branches_encoder]
            for i in range(len(branches_encoder)):
                for j in range(i + 1, len(branches_encoder)):
                    loss_fft_abs -= self.criterion(branches_encoder[i], branches_encoder[j])

            self.log('train_loss_fft_org', loss_fft_abs, on_step=False, on_epoch=True, prog_bar=True)
            if torch.abs(self.dynamic_coefficient * loss_fft_abs) > 1.5 * loss_mse:
                self.dynamic_coefficient *= 0.5
            loss_fft_abs = self.dynamic_coefficient * loss_fft_abs

            loss += loss_fft_abs
            self.log('train_loss_fft', loss_fft_abs, on_step=False, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_year, batch_month, batch_day = batch
        estimated_x, branches_encoder = self(batch_x, batch_year, batch_month, batch_day)
        loss = self.criterion(estimated_x, batch_x)

        loss_fft2_abs = 0
        branches_encoder = [torch.log(torch.abs(torch.fft.fft2(branch)) + 1) for branch in branches_encoder]
        for i in range(len(branches_encoder)):
            for j in range(i + 1, len(branches_encoder)):
                loss_fft2_abs -= self.criterion(branches_encoder[i], branches_encoder[j])
        self.log('val_loss_fft_org', loss_fft2_abs, on_step=False, on_epoch=True, prog_bar=True)

        eval_res, _ = metric(estimated_x.cpu().numpy(), batch_x.cpu().numpy(),
                             self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list,
                             channel_names=self.channel_names, spatial_norm=self.spatial_norm)

        for k, v in eval_res.items():
            self.log(f'val_{k}', v, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_year, batch_month, batch_day = batch
        estimated_x, branches_encoder = self(batch_x, batch_year, batch_month, batch_day)
        return {'inputs': batch_x.cpu().numpy(), 'preds': estimated_x.cpu().numpy(), 'trues': batch_x.cpu().numpy()}