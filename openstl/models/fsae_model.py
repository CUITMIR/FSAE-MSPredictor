import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

class FSAE_model(nn.Module):
    def __init__(
            self,
            d_in,
            d_model,
            d_state=1,
            expand=2,
            bidirectional=False,
            fft_path_type='none',
            fft_norm='ortho',
            fft_expand=1,
            mlp_type='mlp_relu',
            mlp_ratio=4.,
            sql_len=-1,
            num_encoder_layers=(1, 1),
            num_decoder_layers=(1, 1),
            num_branches=3,
            merge_type='add',
            dropout=0.,

            **kwargs
    ):
        super().__init__()

        self.encoder = FSAEEncoder(
            d_in=d_in,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            bidirectional=bidirectional,
            fft_path_type=fft_path_type,
            fft_norm=fft_norm,
            fft_expand=fft_expand,
            mlp_type=mlp_type,
            mlp_ratio=mlp_ratio,
            sql_len=sql_len,
            num_encoder_layers=num_encoder_layers,
            num_branches=num_branches,
            dropout=dropout,
        )
        self.decoder = FSAEDecoder(
            d_model=d_model,
            d_out=d_in,
            d_state=d_state,
            expand=expand,
            bidirectional=bidirectional,
            fft_path_type=fft_path_type,
            fft_norm=fft_norm,
            fft_expand=fft_expand,
            mlp_type=mlp_type,
            mlp_ratio=mlp_ratio,
            sql_len=sql_len,
            num_decoder_layers=num_decoder_layers,
            num_branches=num_branches,
            merge_type=merge_type,
            dropout=dropout,
        )

    def forward(self, forward_type, *args):
        """
        We use the 'enc&dec' forward_type  in training phase 1 to train the encoder and decoder.
        In training phase 2, we freeze the encoder and decoder, and use the 'enc' and 'dec' forward_type to get the ouput from the encoder and decoder separately.

        The output of FASE is described by the following pseudocode:

        if forward_type == 'enc&dec':
            x, x_time = args

            x.shape: (B T d_in)
            x_time.shape: (B T 5)

            output:
                out_decoder: (B T d_in)
                branches_encoder[i]: (B T d_model)

        elif forward_type == 'enc':
            x, x_time = args

            x.shape: (B T d_in)
            x_time.shape: (B T 5)

            output:
                branches_encoder[i]: (B T d_model)

        elif forward_type == 'dec':
            branches_encoder = args[0]

            branches_encoder[i]: (B T d_model)

            output:
                out_decoder: (B T d_in)
        """
        if forward_type == 'enc&dec':
            x, x_time = args
            branches_encoder = self.encoder(x, x_time)
            out_decoder = self.decoder(branches_encoder)
            return out_decoder, branches_encoder

        elif forward_type == 'enc':
            x, x_time = args
            branches_encoder = self.encoder(x, x_time)

            return branches_encoder

        elif forward_type == 'dec':
            branches_encoder = args[0]
            out_decoder = self.decoder(branches_encoder)
            return out_decoder


class FSAEEncoder(nn.Module):
    def __init__(
            self,
            d_in,
            d_model,
            d_state=1,
            expand=2,
            bidirectional=False,
            fft_path_type='none',
            fft_norm='ortho',
            fft_expand=1,
            mlp_type='mlp_relu',
            mlp_ratio=4.,
            sql_len=-1,
            num_encoder_layers=(1, 1),
            num_branches=3,
            dropout=0.,
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_in, d_model)
        self.mark_embedding = mark_Embedding(d_model, d_model, embed_type='timeF', freq='t')

        self.encoder_stage1 = nn.Sequential(*[
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                bidirectional=bidirectional,
                fft_path_type=fft_path_type,
                fft_norm=fft_norm,
                fft_expand=fft_expand,
                mlp_type=mlp_type,
                mlp_ratio=mlp_ratio,
                sql_len=sql_len,
                dropout=dropout
            ) for _ in range(num_encoder_layers[0])
        ])
        self.encoder_stage2 = nn.ModuleList([
            nn.Sequential(*[
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    bidirectional=bidirectional,
                    fft_path_type=fft_path_type,
                    fft_norm=fft_norm,
                    fft_expand=fft_expand,
                    mlp_type=mlp_type,
                    mlp_ratio=mlp_ratio,
                    sql_len=sql_len,
                    dropout=dropout
                ) for _ in range(num_encoder_layers[1])
            ]) for _ in range(num_branches)
        ])

    def forward(self, x, x_time):
        """
        x.shape: (B T d_in)
        x_time.shape: (B T 5)

        output shape: (B T d_model)
        """
        x = self.in_proj(x)
        x = x + self.mark_embedding(x, x_time)

        x = self.encoder_stage1(x)
        branches_encoder = []
        for layer in self.encoder_stage2:
            branches_encoder.append(layer(x))

        return branches_encoder

class FSAEDecoder(nn.Module):
    def __init__(
            self,
            d_model,
            d_out,
            d_state=1,
            expand=2,
            bidirectional=False,
            fft_path_type='none',
            fft_norm='ortho',
            fft_expand=1,
            mlp_type='mlp_relu',
            mlp_ratio=4.,
            sql_len=-1,
            num_decoder_layers=(1, 1),
            num_branches=3,
            merge_type='add',
            dropout=0.,
    ):
        super().__init__()
        self.merge_type = merge_type

        self.decoder_stage1 = nn.ModuleList([
            nn.Sequential(*[
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    bidirectional=bidirectional,
                    fft_path_type=fft_path_type,
                    fft_norm=fft_norm,
                    fft_expand=fft_expand,
                    mlp_type=mlp_type,
                    mlp_ratio=mlp_ratio,
                    sql_len=sql_len,
                    dropout=dropout
                ) for _ in range(num_decoder_layers[0])
            ]) for _ in range(num_branches)
        ])
        self.decoder_stage2 = nn.Sequential(*[
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                bidirectional=bidirectional,
                fft_path_type=fft_path_type,
                fft_norm=fft_norm,
                fft_expand=fft_expand,
                mlp_type=mlp_type,
                mlp_ratio=mlp_ratio,
                sql_len=sql_len,
                dropout=dropout
            ) for _ in range(num_decoder_layers[1])
        ])
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, branches_encoder):
        branches_decoder = []
        for i, layer in enumerate(self.decoder_stage1):
            branches_decoder.append(layer(branches_encoder[i]))

        if self.merge_type == 'add':
            x = sum(branches_decoder)
        else:
            raise NotImplemented

        x = self.decoder_stage2(x)
        x = self.out_proj(x)
        return x

class MambaBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state,
            expand,
            bidirectional=False,
            fft_path_type='none',
            fft_norm='ortho',
            fft_expand=1,
            mlp_type='mlp_relu',  # ['mlp_relu', 'mlp_gelu', 'gmlp']
            mlp_ratio=4.,
            sql_len=-1,
            dropout=0.
    ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.bidirectional = bidirectional
        self.fft_path_type = fft_path_type
        self.fft_norm = fft_norm
        self.mlp_type = mlp_type

        self.norm1 = nn.LayerNorm(d_model)
        if bidirectional:
            self.forward_path = Mamba(
                d_model=d_model,
                d_state=d_state,
                expand=expand
            )
            self.backward_path = Mamba(
                d_model=d_model,
                d_state=d_state,
                expand=expand
            )
            if fft_path_type in ['rfft']:
                self.fft_forward_path = Mamba(
                    d_model=d_model * 2,
                    d_state=d_state,
                    expand=fft_expand
                )
                self.fft_backward_path = Mamba(
                    d_model=d_model * 2,
                    d_state=d_state,
                    expand=fft_expand
                )
        else:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                expand=expand
            )
            if fft_path_type in ['rfft']:
                self.fft_mamba = Mamba(
                    d_model=d_model * 2,
                    d_state=d_state,
                    expand=fft_expand
                )

        if mlp_ratio > 0.:
            if mlp_type in ['mlp_relu', 'mlp_gelu']:
                self.norm2 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, int(d_model * mlp_ratio)),
                    nn.ReLU(inplace=True) if mlp_type == 'mlp_relu' else nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(d_model * mlp_ratio), d_model),
                )
            elif mlp_type == 'gmlp':
                self.gmlp = GMLP(
                    d_model=d_model,
                    d_ffn=int(d_model * mlp_ratio),
                    seq_len=sql_len,
                )
            else:
                raise NotImplementedError

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        input shape: (B T d_model)
        output shape: (B T d_model)
        """
        if self.bidirectional:
            norm = self.norm1(x)

            path = self.forward_path(norm) + torch.flip(
                self.backward_path(torch.flip(norm, dims=[1]))
                , dims=[1])
            if self.fft_path_type == 'rfft':
                raise NotImplemented

            x = norm + self.dropout(path)
        else:
            norm = self.norm1(x)

            path = self.mamba(norm)
            if self.fft_path_type == 'rfft':
                d_model = x.shape[2]
                fft = torch.fft.rfft(norm, dim=1, norm=self.fft_norm)
                fft = torch.cat((fft.real, fft.imag), dim=2)  # (B T_fft d_model*2), T_fft < T

                fft_mamba = self.fft_mamba(fft)
                fft_mamba = torch.complex(fft_mamba[:, :, :d_model], fft_mamba[:, :, d_model:])
                path += torch.fft.irfft(fft_mamba, dim=1, norm=self.fft_norm)  # (B T d_model)

            x = x + self.dropout(path)

        if self.mlp_ratio > 0.:
            if self.mlp_type in ['mlp_relu', 'mlp_gelu']:
                x = x + self.dropout(self.mlp(self.norm2(x)))
            elif self.mlp_type.startswith('kan_1_G') or self.mlp_type.startswith('kan_2_relu_G') or self.mlp_type.startswith('kan_2_gelu_G'):
                B, T, D = x.shape
                stack = []
                for b in range(B):
                    batch_elem = x[b, :, :] + self.dropout(self.mlp(self.norm2(x[b, :, :])))
                    stack.append(batch_elem)
                torch.stack(stack, dim=0)
            elif self.mlp_type == 'gmlp':
                x = self.gmlp(x)
            else:
                raise NotImplementedError

        return x

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class GMLP(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class mark_Embedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(mark_Embedding, self).__init__()

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.position_embedding(x)
        else:
            x = self.temporal_embedding(x_mark) + self.position_embedding(x)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 6
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 'residual':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()