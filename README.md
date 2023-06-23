## [IROS 2023] Fast LiDAR-Inertial Odometry via Incremental Plane Pre-Fitting and Skeleton Tracking

[pre-print](https://arxiv.org/pdf/2302.14674.pdf)

We introduce LIO-PPF, a plane pre-fitting and skeleton tracking technique, that can ease the computation of state-of-the-art LIO systems, e.g. LIO-SAM.
Please refer to this [link](https://github.com/TixiaoShan/LIO-SAM) for details about building.


In LIO-PPF, we track mainly the *basic skeleton of the 3D scene*, the planes of which are not fitted individually for each LiDAR scan, let alone for each LiDAR point. However, they are updated incrementally as the scene gradually `flows'.

By contrast, LIO-PPF can consume only 36% of the original local map size to achieve up to 4x faster residual computing and 1.92x overall FPS, while maintaining the same
level of accuracy.
