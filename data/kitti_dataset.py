import copy
import pickle
from skimage import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils import box_utils, calibration_kitti, common_utils, object3d_kitti


class KittiDataset(torch_data.Dataset):
    def __init__(self, dataset_cfg, class_names, training=True, logger=None,
            include_kitti_infos=True, include_processor=True, data_augmentation=True
        ):
        """
        Args:
            dataset_cfg: dict
            class_names: list
        """
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        
        self.logger = logger
        self.training = training
        if self.training:
            self.split = dataset_cfg.DATA_SPLIT['train']
            self.info_path = dataset_cfg.INFO_PATH['train']
            self.data_augmentation = data_augmentation
        else:
            self.split = dataset_cfg.DATA_SPLIT['test']
            self.info_path = dataset_cfg.INFO_PATH['test']
            self.data_augmentation = False
            
        
        self.root_path = Path(self.dataset_cfg.DATA_PATH) # e.g. Path('data/kitti')
        self.root_split_path = self.root_path / 'training' if self.split in ['train', 'val'] else 'testing'

        split_file = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_file).readlines()] if split_file.exists() else None

        if include_kitti_infos:
            self.include_kitti_infos()
        if include_processor:
            self.include_processor()

    def get_points(self, idx):
        """
        Args:
            idx: str, Sample index
        Returns:
            points: (N, 4), Points of (x, y, z, intensity)
        """
        pts_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert pts_file.exists(), 'File not found: %s' % pts_file
        return np.fromfile(str(pts_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Args:
            idx: str, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists(), 'File not found: %s' % img_file
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        """
        Args:
            idx: str, Sample index
        Returns:
            image_shape: (2), h * w
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists(), 'File not found: %s' % img_file
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        """
        Args:
            idx: str, Sample index
        Returns:
            objects: list, [Object3d, Object3d, ...]
        """
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists(), 'File not found: %s' % label_file
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        """
        Args:
            idx: str, Sample index
        Returns:
            calib: calibration_kitti.Calibration
        """
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists(), 'File not found: %s' % calib_file
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        """
        Args:
            idx: str, Sample index
        Returns:
            plane: (4), [a, b, c, d]
        """
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        assert plane_file.exists(), 'File not found: %s' % plane_file

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(points, img_shape, calib):
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            
            calib = self.get_calib(sample_idx)
            P2_4x4 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            
            calib_info = {'P2': P2_4x4, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare']) # including Pedestrian, Truck, Car, Cyclist, Misc, etc.
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar # (M, 7), [x, y, z, l, w, h, heading] in lidar coordinate system

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_points(sample_idx)
                    calib = self.get_calib(sample_idx)
                    
                    fov_flag = self.get_fov_flag(points, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self):
        import torch

        database_save_path = Path(self.root_path) / 'gt_database'
        db_info_save_path = Path(self.root_path) / 'kitti_dbinfos_train.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        info_path = self.root_path / self.info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_points(sample_idx)
            annos = info['annos']
            names = annos['name']
            bbox = annos['bbox']
            score = annos['score']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                           'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                           'difficulty': difficulty[i], 'bbox': bbox[i], 'score': score[i]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id: (batch_size), str, Sample index
                calib: (batch_size), calibration_kitti.Calibration
                gt_boxes: (batch_size, M_max, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system
                points: (N1 + N2 + ..., 5), Points of (batch_id, x, y, z, intensity)
                voxels: (num_voxels1 + num_voxels2 + ..., max_points_per_voxel, 4), [x, y, z, intensity]
                voxel_coords: (num_voxels1 + num_voxels2 + ..., 4), Location of voxels, [batch_id, zi, yi, xi]
                voxel_num_points: (num_voxels1 + num_voxels2 + ...), Number of points in each voxel
                image_shape: (batch_size, 2), h * w
                batch_size: int
                image: optional, (batch_size, H_max, W_max, 3), RGB Image
            pred_dicts: list, [{pred_boxes: (M, 7), pred_scores: (M), pred_labels: (M)}, {...}, ...]
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def include_kitti_infos(self):
        self.kitti_infos = []
        
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')

        info_path = self.root_path / self.info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        self.kitti_infos.extend(infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))

    def include_processor(self):
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.used_feature_list = self.dataset_cfg.USED_FEATURE_LIST
        self.num_point_features = len(self.used_feature_list)
        
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, self.num_point_features, logger=self.logger
        ) if self.data_augmentation else None
        
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, self.point_cloud_range, self.training, self.num_point_features
        )
        
        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

    def __getitem__(self, index):
        """
        Returns:
            data_dict:
                frame_id: str, Sample index
                calib: calibration_kitti.Calibration
                gt_boxes: (M', 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system
                points: (N', 4), Points of (x, y, z, intensity)
                voxels: (num_voxels, max_points_per_voxel, 4), [x, y, z, intensity]
                voxel_coords: (num_voxels, 3), Location of voxels, [zi, yi, xi]
                voxel_num_points: (num_voxels), Number of points in each voxel
                image_shape: (2), h * w
                image: optional, (H, W, 3), RGB Image
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
        
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare') # exclude class: DontCare
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar'],
                'road_plane': self.get_road_plane(sample_idx)
            })

        if "points" in get_item_list:
            points = self.get_points(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                fov_flag = self.get_fov_flag(points, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "image" in get_item_list:
            input_dict['image'] = self.get_image(sample_idx)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        
        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                frame_id: str, Sample index
                calib: calibration_kitti.Calibration
                gt_names: (M), str
                gt_boxes: (M, 7), [x, y, z, l, w, h, heading] in lidar coordinate system
                road_plane: (4), [a, b, c, d]
                points: (N, 4), Points of (x, y, z, intensity)
                image: optional, (H, W, 3), RGB Image

        Returns:
            data_dict:
                frame_id: str, Sample index
                calib: calibration_kitti.Calibration
                gt_boxes: (M', 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system
                points: (N', 4), Points of (x, y, z, intensity)
                voxels: (num_voxels, max_points_per_voxel, 4), [x, y, z, intensity]
                voxel_coords: (num_voxels, 3), Location of voxels, [zi, yi, xi]
                voxel_num_points: (num_voxels), Number of points in each voxel
                image: optional, (H, W, 3), RGB Image
        """
        
        data_dict = self.data_augmentor.forward(data_dict=data_dict) if self.data_augmentor is not None else data_dict

        if data_dict.get('gt_boxes', None) is not None:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            ) # limit heading to [-pi, pi)
            
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool) # include class: Car, Pedestrian, Cyclist
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32) # Car: 1, Pedestrian: 2, Cyclist: 3
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes # (M', 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        data_dict.pop('road_plane', None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        Args:
            batch_list: list, [data_dict, data_dict, ...]
        
        Returns:
            ret:
                frame_id: (batch_size), str, Sample index
                calib: (batch_size), calibration_kitti.Calibration
                gt_boxes: (batch_size, M_max, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system
                points: (N1 + N2 + ..., 5), Points of (batch_id, x, y, z, intensity)
                voxels: (num_voxels1 + num_voxels2 + ..., max_points_per_voxel, 4), [x, y, z, intensity]
                voxel_coords: (num_voxels1 + num_voxels2 + ..., 4), Location of voxels, [batch_id, zi, yi, xi]
                voxel_num_points: (num_voxels1 + num_voxels2 + ...), Number of points in each voxel
                image_shape: (batch_size, 2), h * w
                batch_size: int
                image: optional, (batch_size, H_max, W_max, 3), RGB Image
        """
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ["images"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        image_pad = np.pad(image, pad_width=(pad_h, pad_w, (0, 0)), mode='constant', constant_values=0)
                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


def create_kitti_infos(dataset_cfg, class_names, workers=4):
    root_dir = (Path(__file__).resolve().parent / '../').resolve() # ~/pointpillars
    save_path = root_dir / dataset_cfg.DATA_PATH # ~/pointpillars/data/kitti

    print('---------------Start to generate data infos---------------')
    
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=True,
        include_kitti_infos=False, include_processor=False)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    train_filename = save_path / dataset_cfg.INFO_PATH['train']
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)
    
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False,
        include_kitti_infos=False, include_processor=False)
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    test_filename = save_path / dataset_cfg.INFO_PATH['test']
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=True,
        include_kitti_infos=False, include_processor=False)
    dataset.create_groundtruth_database()

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        try:
            cfg = EasyDict(yaml.load(open(sys.argv[2]), Loader=yaml.FullLoader)) # YAML 5.1 use this for safety
        except:
            cfg = EasyDict(yaml.load(open(sys.argv[2])))
        create_kitti_infos(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
        )
