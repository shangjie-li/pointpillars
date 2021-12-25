import argparse
import glob
from pathlib import Path
import time

try:
    import open3d
    from utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/pointpillar.yaml',
        help='specify the config for training')
    parser.add_argument('--split', choices=['train', 'val'], default='train')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_config()
    print(cfg)
    
    training_mode = True if args.split == 'train' else False
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=training_mode,
        logger=None,
    )
    
    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        data_dict = dataset[i]
        
        print()
        print('<<< data_dict >>>')
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                print(key, type(val), val.shape)
                print(val)
            else:
                print(key, type(val))
                print(val)
        print()
        
        V.draw_scenes(
            points=data_dict['points'].reshape(-1, 4), ref_boxes=data_dict['gt_boxes'].reshape(-1, 8)[:, :7],
            ref_scores=None, ref_labels=data_dict['gt_boxes'].reshape(-1, 8)[:, 7].astype(np.int)
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)

