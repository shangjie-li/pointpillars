# pointpillars

Implementation of PointPillars in PyTorch for KITTI 3D Object Detetcion

## Acknowledgement
 - This repository references [open-mmlab](https://github.com/open-mmlab/OpenPCDet)'s work.

## Installation
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/pointpillars.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n pcdet.v0.5.0 python=3.6
   conda activate pcdet.v0.5.0
   cd pointpillars
   pip install -r requirements.txt
   ```
 - Install spconv
   ```
   # Try the command below:
   pip install spconv-cu102
   
   # If there is `ERROR: Cannot uninstall 'certifi'.`, try:
   pip install spconv-cu102 --ignore-installed
   ```
 - Compile external modules
   ```
   cd pointpillars
   python setup.py develop
   ```
 - Install visualization tools
   ```
   pip install mayavi
   pip install pyqt5
   
   # If you want import opencv, run:
   pip install opencv-python
   
   # If you want import open3d, run:
   pip install open3d-python
   ```

## KITTI3D Dataset (41.5GB)
 - Download KITTI3D dataset: [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip), [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip), [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) and [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip).
 - Download [road plane](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) for data augmentation.
 - Organize the downloaded files as follows
   ```
   pointpillars
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2 & planes
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   ├── layers
   ├── utils
   ```
 - Generate the ground truth database and data infos by running the following command
   ```
   # This will create gt_database dir and info files in pointpillars/data/kitti.
   cd pointpillars
   python -m data.kitti_dataset create_kitti_infos data/config.yaml
   ```
 - Display the dataset
   ```
   # Display the training dataset with data augmentation
   python dataset_player.py --training --data_augmentation --show_boxes
   
   # Display the testing dataset
   python dataset_player.py --show_boxes
   ```

## Demo
 - Run the demo with a pretrained model (Download [pointpillar_7728.pth](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) and save it in pointpillars/weights.)
   ```
   # Run on the testing dataset
   python demo.py --ckpt=weights/pointpillar_7728.pth
   
   # Run on a single sample from the testing dataset
   python demo.py --ckpt=weights/pointpillar_7728.pth --sample_idx=000008
   ```

## Training
 - Run the command below to train
   ```
   python train.py --batch_size=2
   ```

## Evaluation
 - Run the command below to evaluate
   ```
   python test.py --ckpt=weights/pointpillar_7728.pth
   ```
 - The 3D detection performance on KITTI should be
   | Class                | AP (R11) BEV              | AP (R11) 3D               |
   |:--------------------:|:-------------------------:|:-------------------------:|
   | Car (Iou=0.7)        | 89.6590, 87.1725, 84.3762 | 86.4617, 77.2839, 74.6530 |
   | Pedestrian (Iou=0.5) | 61.6348, 56.2747, 52.6007 | 57.7500, 52.2916, 47.9072 |
   | Cyclist (Iou=0.5)    | 82.2593, 66.1110, 62.5585 | 80.0483, 62.6080, 59.5260 |
    * Report in different difficulties, which are Easy, Moderate and Hard.
