import sys
import os.path
_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_PROJ_ROOT = os.path.realpath(os.path.join(_CURR_DIR, '..'))
sys.path.append(os.path.join(_PROJ_ROOT))  # 添加openstl的包路径

# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    cfg_path = osp.join(_PROJ_ROOT, 'configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else osp.join(_PROJ_ROOT, args.config_file)
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()