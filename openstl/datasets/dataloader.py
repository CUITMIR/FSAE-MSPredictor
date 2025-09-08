# Copyright (c) CAIRI AI Lab. All rights reserved

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    if 'wind' == dataname:
        from .dataloader_wind import load_data
        return load_data(batch_size, val_batch_size, num_workers)
    elif 'wind_kdd' == dataname:
        from .dataloader_wind_kdd import load_data
        return load_data(batch_size, val_batch_size, num_workers)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
