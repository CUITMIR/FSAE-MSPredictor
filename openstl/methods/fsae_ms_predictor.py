import datetime

import torch

from openstl.core import metric
from openstl.methods import FSAE
from openstl.methods.base_method import Base_method
from openstl.models.fsae_ms_predictor_model import FFTVAEPredictor_model


class FSAEMSPredictor(Base_method):
    def __init__(self, **args):
        super().__init__(**args)

        self.fsae_model = FSAE.load_from_checkpoint(self.hparams.vae_checkpoint)

    def _build_model(self, **args):
        return FFTVAEPredictor_model(**args)

    def forward(self, batch_x, batch_year, batch_month, batch_day, **kwargs):
        T = self.hparams.in_shape[0]
        time = torch.stack([
            self.make_mark(year, month, day, T * 2)
        for year, month, day in zip(batch_year, batch_month, batch_day)], dim=0).to(batch_x.device)

        branches_encoder = self.fsae_model.model('enc', batch_x, time[:, :T, :])
        if hasattr(self.hparams, 'without_decoder') and self.hparams.without_decoder == True:
            out = self.model(branches_encoder, time[:, T:, :])
        else:
            branches_predictor = self.model(branches_encoder, time[:, T:, :])
            out = self.fsae_model.model('dec', branches_predictor)
        return out

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
        pred_y = self(batch_x, batch_year, batch_month, batch_day)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_year, batch_month, batch_day = batch
        pred_y = self(batch_x, batch_year, batch_month, batch_day)
        loss = self.criterion(pred_y, batch_y)

        eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                             self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list,
                             channel_names=self.channel_names, spatial_norm=self.spatial_norm)

        for k, v in eval_res.items():
            self.log(f'val_{k}', v, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_year, batch_month, batch_day = batch
        pred_y = self(batch_x, batch_year, batch_month, batch_day)
        return {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}