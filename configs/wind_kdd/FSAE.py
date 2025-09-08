method = 'FSAE'
# model
d_in = 2
d_model = 128
d_state = 1
expand = 2
bidirectional = True
mlp_ratio = 4.
num_encoder_layers = (1, 2)
num_decoder_layers = (2, 1)
num_branches = 4
merge_type = 'add'
dropout = 0.1
# training
loss_types = ['mse', 'fft2_abs']
fft_loss_coefficient = 6

lr = 5e-4
batch_size = 64
sched = 'none'