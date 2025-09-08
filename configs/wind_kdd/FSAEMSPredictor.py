method = 'FSAEMSPredictor'
# model
predictor_version = 3
d_model = 128
d_state = 1
expand = 2
bidirectional = False
mlp_type = 'mlp_relu'
mlp_ratio = 4.
sql_len = -1
share_weight = True
merge_type = 'v1'
predictor_lens = (36, 72, 144)
num_predictor_layers = (2, 2, 2, 1)
dropout = 0.1
num_branches = 4

vae_checkpoint = '~/code/openstl/tools/work_dirs/FSAE_wind_kdd/checkpoints/epoch=39-val_loss=0.002.ckpt'
# training
lr = 5e-4
batch_size = 64
sched = 'none'