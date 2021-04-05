# DPGO


## Introduction
This repository contains implementation of synchronous and asynchronous Distributed Pose Graph Optimization (DPGO).  The algorithms are described in the following papers:

 - Y. Tian, K. Khosoussi, D. M. Rosen, J. P. How. [**Distributed Certifiably Correct Pose-Graph Optimization**](https://arxiv.org/abs/1911.03721). arXiv preprint [arXiv:1911.03721](https://arxiv.org/abs/1911.03721).
 
 - Y.Tian, A. Koppel, A. S. Bedi, J. P. How.  [**Asynchronous and Parallel Distributed Pose Graph Optimization**](https://arxiv.org/abs/2003.03281). IEEE Robotics and Automation Letters (RA-L) 2020. Full paper: [arXiv:2003.03281](https://arxiv.org/abs/2003.03281).

## Dependencies
DPGO uses Eigen3 and the following packages:
```
sudo apt-get install build-essential cmake-gui libsuitesparse-dev
```
#### ROPTLIB
In addition, the code currently use our public fork of the ROPTLIB library (use the `feature/0.7_cmake` branch). Source codes and installation instructions can be found at: https://github.com/yuluntian/ROPTLIB/tree/feature/0.7_cmake

## Building the C++ Library 

Execute the following commands:

```
mkdir build
cd build
cmake ../
make
```

The built executables are located in directory build/bin. For a serialized demo of distributed PGO on one of the benchmark datasets, move to the top-level directory and run:
```
./C++/build/bin/multi-robot-example 5 data/smallGrid3D.g2o
```

Optionally, install the C++ library via,
```
sudo make install
```
The installation is required for using the ROS wrapper. 

## ROS support

A ROS wrapper for DPGO is provided in https://gitlab.com/mit-acl/dpgo/dpgo_ros.git .






