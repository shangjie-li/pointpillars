import argparse
import numpy as np
from pathlib import Path
import open3d

from data import cfg, cfg_from_yaml_file
from data.kitti_dataset import KittiDataset
from utils import box_utils


COLORS = (
    (0, 255, 0), # Car
    (255, 0, 0), # Pedestrian
    (0, 255, 255) # Cyclist
)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/pointpillar.yaml',
        help='specify the config for training')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, cfg


def get_o3d_box(box3d, color=[1.0, 0.0, 0.0]):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        box3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    box3d = np.array(box3d)
    assert len(box3d.shape) == 1
    corners3d = box_utils.boxes_to_corners_3d(np.array([box3d]))[0]
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [0, 5], [1, 4], # heading
    ])
    colors = [color for i in range(len(edges))]
    
    lines = open3d.geometry.LineSet()
    lines.points = open3d.utility.Vector3dVector(corners3d)
    lines.lines = open3d.Vector2iVector(edges)
    lines.colors = open3d.utility.Vector3dVector(colors)
    return lines


if __name__ == '__main__':
    args, cfg = parse_config()
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=True,
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
        
        vis = open3d.visualization.Visualizer()
        vis.create_window(window_name='points', width=800, height=600)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        
        points = data_dict['points'].reshape(-1, 4)
        colors = np.array([[1.0, 1.0, 1.0]]).repeat(points.shape[0], axis=0)
        
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(colors)
        vis.add_geometry(pts)
        
        gt_boxes = data_dict['gt_boxes'].reshape(-1, 8)
        for j in range(gt_boxes.shape[0]):
            class_idx = int(gt_boxes[j, 7]) - 1
            color = [c / 255.0 for c in COLORS[class_idx]]
            box = get_o3d_box(gt_boxes[j, :7], color=color)
            vis.add_geometry(box)
        
        vis.run()
        vis.destroy_window()
