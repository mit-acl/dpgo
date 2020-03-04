# DPGO

This repository contains implementation of synchronous and asynchronous Distributed Pose Graph Optimization (DPGO).  The algorithms are described in the following papers:

 - Y. Tian, K. Khosoussi, J. P. How. [**Block-Coordinate Descent on the Riemannian Staircase for Certifiably Correct Distributed Rotation and Pose Synchronization**](https://arxiv.org/abs/1911.03721). arXiv preprint [arXiv:1911.03721](https://arxiv.org/abs/1911.03721).
 
 - Y.Tian, A. Koppel, A. S. Bedi, J. P. How.  **Asynchronous and Parallel Distributed Pose Graph Optimization**.

## Building the C++ Library 

Install dependencies.

```
sudo apt-get install build-essential cmake-gui libsuitesparse-dev
```

Inside the C++ directory, execute the following commands.

```
mkdir build
cd build
cmake ../
make
```

The built executables are located in directory build/bin. For a serialized demo of distributed PGO, run:
```
./C++/build/bin/distributed-example 5 data/smallGrid3D.g2o
```

Optionally, install the C++ library via,
```
sudo make install
```
The installation is required for using the ROS wrapper. 

## ROS support

A ROS wrapper for DPGO is provided in https://yuluntian@bitbucket.org/yuluntian/dpgo_ros . 

