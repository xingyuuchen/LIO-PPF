# LIO-PPF

The official implementation of the paper "[Fast LiDAR-Inertial Odometry via Incremental Plane Pre-Fitting and Skeleton Tracking](https://ieeexplore.ieee.org/abstract/document/10341524/)" (**IROS 2023**)

We introduce **LIO-PPF**, a plane pre-fitting and skeleton tracking technique, that can ease the computation of state-of-the-art LIO systems, e.g. LIO-SAM.
Please refer to this 


In LIO-PPF, we track mainly the *basic skeleton of the 3D scene*, the planes of which are not fitted individually for each LiDAR scan, let alone for each LiDAR point. However, they are updated incrementally as the scene gradually `flows'.

By contrast, LIO-PPF can consume only 36% of the original local map size to achieve up to 4x faster residual computing and 1.92x overall FPS, while maintaining the same level of accuracy.

## Quick Start

```bash
catkin_make
source devel/setup.bash
roslaunch lio_sam run.launch
```

In another terminal:
```bash
rosbag play /path/to/your/bag/file
```

For details about building and running, please refer to [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM).


## FasterLIO with PPF
If you are looking for FasterLIO with PPF, please check out [faster-lio-ppf](https://github.com/xingyuuchen/faster-lio-ppf).


## Citation

If you find our work useful or interesting, please consider citing our paper:
```latex
@inproceedings{chen2023lio,
  title={LIO-PPF: Fast LiDAR-Inertial Odometry via Incremental Plane Pre-Fitting and Skeleton Tracking},
  author={Chen, Xingyu and
        Wu, Peixi and
        Li, Ge and
        Li, Thomas H},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1458--1465},
  year={2023},
  organization={IEEE}
}
```

