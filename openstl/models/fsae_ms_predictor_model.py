import torch
import torch.nn as nn

from .fsae_model import MambaBlock, TimeFeatureEmbedding

def FFTVAEPredictor_model(predictor_version=3, **kwargs):
    if predictor_version == 3:
        return FFTVAEPredictorV3_model(**kwargs)
    else:
        raise NotImplementedError

class FFTVAEPredictorV3_model(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=1,
            expand=2,
            bidirectional=False,
            mlp_type='mlp_relu',
            mlp_ratio=4.,
            sql_len=-1,
            share_weight=False,
            merge_type='v1',
            predictor_lens=(72, 144),
            num_predictor_layers=(1, 1, 1),
            dropout=0.,
            num_branches=3,

            **kwargs
    ):
        super().__init__()

        self.pred_time_embed = TimeFeatureEmbedding(d_model=d_model, embed_type='timeF', freq='t')
        self.predictors = nn.ModuleList([
            FFTVAEPredictorV3SingleBranch(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                bidirectional=bidirectional,
                mlp_type=mlp_type,
                mlp_ratio=mlp_ratio,
                sql_len=sql_len,
                share_weight=share_weight,
                merge_type=merge_type,
                predictor_lens=predictor_lens,
                num_predictor_layers=num_predictor_layers,
                dropout=dropout,
            ) for _ in range(num_branches)
        ])

    def forward(self, branches_encoder, pred_time):
        """
        branches_encoder[i]: (B T d_model)
        pred_time: (B T 5)
        branches_predictor[i]: (B T d_model)
        """
        embed = self.pred_time_embed(pred_time)
        branches_predictor = []
        for i, predictor in enumerate(self.predictors):
            branches_predictor.append(predictor(branches_encoder[i], embed))

        return branches_predictor

class FFTVAEPredictorV3SingleBranch(nn.Module):
    """
    [144] -> Mamba -> (36) -> [144]+(36)+(72)+(144) -> Mamba -> 144
          -> Mamba -> (72)
          -> Mamba -> (144)
    """
    def __init__(
            self,
            d_model,
            d_state=1,
            expand=2,
            bidirectional=False,
            mlp_type='mlp_relu',
            mlp_ratio=4.,
            sql_len=-1,
            share_weight=False,
            merge_type='v1',
            predictor_lens=(72, 144),
            num_predictor_layers=(1, 1, 1),
            dropout=0.,
    ):
        super().__init__()
        self.predictor_lens = predictor_lens
        self.merge_type = merge_type
        self.share_weight = share_weight

        assert len(predictor_lens) + 1 == len(num_predictor_layers)

        if not share_weight:
            self.predictor_layers = nn.ModuleList([
                nn.Sequential(*[
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        expand=expand,
                        bidirectional=bidirectional,
                        mlp_type=mlp_type,
                        mlp_ratio=mlp_ratio,
                        sql_len=sql_len,
                        dropout=dropout
                    ) for _ in range(num_predictor_layers[i])
                ]) for i in range(len(num_predictor_layers))
            ])
        else:
            self.predictor_layers = nn.ModuleList([
                nn.Sequential(*[
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        expand=expand,
                        bidirectional=bidirectional,
                        mlp_type=mlp_type,
                        mlp_ratio=mlp_ratio,
                        sql_len=sql_len,
                        dropout=dropout
                    ) for _ in range(num_predictor_layers[0])
                ]),
                nn.Sequential(*[
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        expand=expand,
                        bidirectional=bidirectional,
                        mlp_type=mlp_type,
                        mlp_ratio=mlp_ratio,
                        sql_len=sql_len,
                        dropout=dropout
                    ) for _ in range(num_predictor_layers[-1])
                ]),
            ])

    def forward(self, x, pred_time_embed):
        """
        x.shape: (B T d_model)
        pred_time_embed: (B T d_model)
        output shape: (B T d_model)
        """
        T_pred = x.shape[1]
        assert T_pred == self.predictor_lens[-1]

        if not self.share_weight:
            layer_output = []
            for pred_layer, pred_len in zip(self.predictor_layers[:-1], self.predictor_lens):
                step = T_pred // pred_len
                out = torch.cat((x, pred_time_embed[:, ::step, :]), dim=1)  # (B T_pred+pred_len d_model)
                out = pred_layer(out)
                layer_output.append(out[:, -pred_len:])  # (B pred_len d_model)
        else:
            layer_output = []
            pred_layer = self.predictor_layers[0]
            for pred_len in self.predictor_lens:
                step = T_pred // pred_len
                out = torch.cat((x, pred_time_embed[:, ::step, :]), dim=1)  # (B T_pred+pred_len d_model)
                out = pred_layer(out)
                layer_output.append(out[:, -pred_len:])  # (B pred_len d_model)

        if self.merge_type == 'v1':
            layer_output.insert(0, x)
            x = torch.cat(layer_output, dim=1)  # (B T_pred+sum(predictor_lens) d_model)
        elif self.merge_type == 'v2':
            layer_output.insert(0, out[:, :T_pred, :])
            x = torch.cat(layer_output, dim=1)  # (B T_pred+sum(predictor_lens) d_model)
        elif self.merge_type == 'v3':
            layer_output.insert(-1, out[:, :T_pred, :])
            x = torch.cat(layer_output, dim=1)  # (B T_pred+sum(predictor_lens) d_model)
        x = self.predictor_layers[-1](x)
        return x[:, -T_pred:, :]