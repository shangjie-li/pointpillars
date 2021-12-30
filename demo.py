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
from pointpillar import build_network, load_data_to_gpu
from utils import common_utils


class DemoDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, data_path=None, logger=None, ext='.bin'):
        """
        Args:
            dataset_cfg:
            class_names:
            training:
            data_path:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )
        self.data_path = data_path
        self.ext = ext
        file_list = glob.glob(str(data_path / f'*{self.ext}')) if self.data_path.is_dir() else [self.data_path]
        file_list.sort()
        self.sample_file_list = file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne/000008.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='weights/pointpillar_7728.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of PointPillars-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        data_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    print()
    print('<<< model >>>')
    print(model)
    print()
    
    """
    <<< model >>>
    PointPillar(
      (vfe): PillarVFE(
        (pfn_layers): ModuleList(
          (0): PFNLayer(
            (linear): Linear(in_features=10, out_features=64, bias=False)
            (norm): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          )
        )
      )
      (backbone_3d): None
      (map_to_bev_module): PointPillarScatter()
      (pfe): None
      (backbone_2d): BaseBEVBackbone(
        (blocks): ModuleList(
          (0): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
          )
          (1): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
          (2): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
        )
        (deblocks): ModuleList(
          (0): Sequential(
            (0): ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): Sequential(
            (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): Sequential(
            (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
            (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
      )
      (dense_head): AnchorHeadSingle(
        (cls_loss_func): SigmoidFocalClassificationLoss()
        (reg_loss_func): WeightedSmoothL1Loss()
        (dir_loss_func): WeightedCrossEntropyLoss()
        (conv_cls): Conv2d(384, 18, kernel_size=(1, 1), stride=(1, 1))
        (conv_box): Conv2d(384, 42, kernel_size=(1, 1), stride=(1, 1))
        (conv_dir_cls): Conv2d(384, 12, kernel_size=(1, 1), stride=(1, 1))
      )
      (point_head): None
      (roi_head): None
    )
    """
    
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            
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
            
            """
            <<< data_dict >>>
            points <class 'numpy.ndarray'> (60266, 5)
            [[ 0.    21.554  0.028  0.938  0.34 ]
             [ 0.    21.24   0.094  0.927  0.24 ]
             [ 0.    21.056  0.159  0.921  0.53 ]
             ...
             [ 0.     3.805 -1.433 -1.78   0.37 ]
             [ 0.     3.731 -1.391 -1.741  0.07 ]
             [ 0.     3.841 -1.419 -1.793  0.   ]]
            frame_id <class 'numpy.ndarray'> (1,)
            [0]
            voxels <class 'numpy.ndarray'> (7260, 32, 4)
            [[[ 2.1554e+01  2.8000e-02  9.3800e-01  3.4000e-01]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              ...
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]]
            
             [[ 2.1240e+01  9.4000e-02  9.2700e-01  2.4000e-01]
              [ 2.1148e+01  3.6000e-02  7.9000e-01  2.7000e-01]
              [ 2.1216e+01  1.7000e-02  6.9100e-01  3.6000e-01]
              ...
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]]
            
             [[ 2.1056e+01  1.5900e-01  9.2100e-01  5.3000e-01]
              [ 2.1072e+01  1.0200e-01  7.8800e-01  3.6000e-01]
              [ 2.1098e+01  1.1600e-01  6.8900e-01  3.1000e-01]
              ...
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]]
            
             ...
            
             [[ 3.6790e+00 -1.7550e+00 -1.7850e+00  3.5000e-01]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              ...
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]]
            
             [[ 3.7550e+00 -1.5920e+00 -1.7860e+00  3.0000e-01]
              [ 3.7530e+00 -1.5770e+00 -1.7820e+00  4.2000e-01]
              [ 3.7580e+00 -1.5650e+00 -1.7820e+00  3.5000e-01]
              ...
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]]
            
             [[ 3.8050e+00 -1.4330e+00 -1.7800e+00  3.7000e-01]
              [ 3.7310e+00 -1.3910e+00 -1.7410e+00  7.0000e-02]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              ...
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]
              [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]]]
            voxel_coords <class 'numpy.ndarray'> (7260, 4)
            [[  0   0 248 134]
             [  0   0 248 132]
             [  0   0 248 131]
             ...
             [  0   0 237  22]
             [  0   0 238  23]
             [  0   0 239  23]]
            voxel_num_points <class 'numpy.ndarray'> (7260,)
            [ 1 10 11 ...  1 14  2]
            batch_size <class 'int'>
            1
            """
            
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict) # 0
            
            time_start = time.time()
            pred_dicts, _ = model.forward(data_dict) # 1
            pred_dicts, _ = model.forward(data_dict) # 2
            pred_dicts, _ = model.forward(data_dict) # 3
            pred_dicts, _ = model.forward(data_dict) # 4
            pred_dicts, _ = model.forward(data_dict) # 5
            pred_dicts, _ = model.forward(data_dict) # 6
            pred_dicts, _ = model.forward(data_dict) # 7
            pred_dicts, _ = model.forward(data_dict) # 8
            pred_dicts, _ = model.forward(data_dict) # 9
            pred_dicts, _ = model.forward(data_dict) # 10
            time_end = time.time()
            
            print()
            print('<<< pred_dicts[0] >>>') # It seems that there is only one element in the list of pred_dicts.
            for key, val in pred_dicts[0].items():
                try:
                    print(key, type(val), val.shape)
                    print(val)
                except:
                    print(key, type(val))
                    print(val)
            print()
            
            """
            <<< pred_dicts[0] >>>
            pred_boxes <class 'torch.Tensor'> torch.Size([32, 7])
            tensor([[ 14.7530,  -1.0668,  -0.7949,   3.7316,   1.5734,   1.5017,   5.9684],
                    [  8.1338,   1.2251,  -0.8056,   3.7107,   1.5718,   1.6053,   2.8312],
                    [  6.4539,  -3.8711,  -1.0125,   2.9638,   1.5000,   1.4587,   6.0006],
                    [  4.0341,   2.6423,  -0.8589,   3.5184,   1.6252,   1.6226,   6.0168],
                    [ 33.5379,  -7.0922,  -0.5771,   4.1590,   1.6902,   1.6740,   2.8695],
                    [ 20.2419,  -8.4190,  -0.8768,   2.2857,   1.5067,   1.5971,   5.9195],
                    [ 24.9979, -10.1047,  -0.9413,   3.7697,   1.6151,   1.4467,   5.9671],
                    [ 55.4206, -20.1693,  -0.5863,   4.1936,   1.6783,   1.5897,   2.8009],
                    [ 40.9520,  -9.8718,  -0.5903,   3.7940,   1.5752,   1.5658,   5.9509],
                    [ 28.7372,  -1.6067,  -0.3582,   3.7827,   1.5546,   1.5801,   1.2517],
                    [ 29.8940, -14.0270,  -0.7138,   0.7105,   0.5286,   1.8181,   1.8177],
                    [ 10.5068,   5.3847,  -0.6656,   0.8203,   0.6050,   1.7170,   4.5543],
                    [ 14.7198, -13.9145,  -0.7675,   0.6548,   0.6050,   1.8767,   6.3586],
                    [ 40.5776,  -7.1297,  -0.4536,   0.7717,   0.6421,   1.8219,   6.3071],
                    [ 18.6621,   0.2868,  -0.7225,   0.6963,   0.5903,   1.6171,   3.4939],
                    [ 33.5909, -15.3372,  -0.6708,   1.5792,   0.4420,   1.6632,   5.8578],
                    [ 53.6673, -16.1789,  -0.2170,   0.9555,   0.5120,   1.9663,   4.0730],
                    [ 30.4546,  -3.7337,  -0.3892,   1.6604,   0.5506,   1.7268,   2.8738],
                    [ 37.2168,  -6.0348,  -0.4855,   0.8860,   0.5873,   1.7859,   6.3918],
                    [ 34.0845,  -4.9617,  -0.4192,   0.8911,   0.6893,   1.8796,   6.0675],
                    [ 13.2934,   4.3788,  -0.5723,   1.7745,   0.5844,   1.7321,   5.5894],
                    [  1.5887,   8.8918,  -0.5623,   1.7521,   0.3996,   1.6873,   6.9082],
                    [  1.6363,  10.6976,  -0.4213,   0.5559,   0.5656,   1.6537,   1.1167],
                    [ 10.1203,  -7.5959,  -0.8065,   1.6906,   0.5269,   1.8206,   6.0078],
                    [  1.3104,  -5.3168,  -0.9996,   3.8217,   1.5819,   1.5247,   5.7200],
                    [  1.9891,   6.9479,  -0.6237,   0.7172,   0.6449,   1.8667,   5.1038],
                    [ 37.0710, -16.5266,  -0.6848,   1.4592,   0.5439,   1.6777,   2.5990],
                    [ 18.6999,   1.1810,  -0.4766,   0.7327,   0.6436,   1.8375,   5.8503],
                    [  2.6479,  17.1586,  -0.1585,   0.5904,   0.6348,   1.8937,   3.6890],
                    [  0.9431,  10.5031,  -0.3420,   0.5309,   0.5733,   1.7027,   2.1916],
                    [  5.7515, -12.5565,  -0.7717,   0.5685,   0.5493,   1.6204,   2.1157],
                    [ 45.0186,  -7.5816,  -0.0797,   3.7895,   1.6455,   1.7168,   4.4490]],
                   device='cuda:0')
            pred_scores <class 'torch.Tensor'> torch.Size([32])
            tensor([0.9654, 0.9511, 0.9037, 0.8834, 0.8346, 0.6788, 0.6594, 0.5516, 0.5041,
                    0.4658, 0.3139, 0.3063, 0.2938, 0.2692, 0.2396, 0.2348, 0.2258, 0.2208,
                    0.2194, 0.1883, 0.1608, 0.1559, 0.1516, 0.1449, 0.1427, 0.1358, 0.1239,
                    0.1227, 0.1212, 0.1207, 0.1192, 0.1003], device='cuda:0')
            pred_labels <class 'torch.Tensor'> torch.Size([32])
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3,
                    1, 2, 3, 2, 2, 2, 2, 1], device='cuda:0')
            """

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
                
            print('Time cost per batch: %s' % (round((time_end - time_start) / 10, 3)))

    logger.info('Demo done.')


if __name__ == '__main__':
    print('ok!')
    main()
