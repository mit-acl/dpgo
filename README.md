# DPGO


## Introduction
This repository contains implementation of synchronous and asynchronous Distributed Pose Graph Optimization (DPGO).  The algorithms are described in the following publications:

 - Y. Tian, K. Khosoussi, D. M. Rosen, J. P. How. [**Distributed Certifiably Correct Pose-Graph Optimization**](https://arxiv.org/abs/1911.03721), in IEEE Transactions on Robotics, 2021.
 
 - Y.Tian, A. Koppel, A. S. Bedi, J. P. How.  [**Asynchronous and Parallel Distributed Pose Graph Optimization**](https://arxiv.org/abs/2003.03281), in IEEE Robotics and Automation Letters, RA-L best paper honorable mention, 2020. 

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

The built executables are located in directory build/bin. For a serialized demo of distributed PGO on one of the benchmark datasets, inside the build directory run:
```
./bin/multi-robot-example 5 ../data/smallGrid3D.g2o
```

Optionally, run the unit tests by,
```
./bin/testDPGO
```

## ROS Support

A ROS wrapper of dpgo is provided: [dpgo_ros](https://github.com/mit-acl/dpgo_ros). If you want to use dpgo with ROS, please follow instructions in the [README](https://github.com/mit-acl/dpgo_ros). 


## References

If you find this library useful, please consider citing our papers:
```
@ARTICLE{Tian2021Distributed,
  author={Tian, Yulun and Khosoussi, Kasra and Rosen, David M. and How, Jonathan P.},
  journal={IEEE Transactions on Robotics}, 
  title={Distributed Certifiably Correct Pose-Graph Optimization}, 
  year={2021},
  volume={},
  number={},
  pages={1-20},
  doi={10.1109/TRO.2021.3072346}}

@ARTICLE{Tian2020Asynchronous,
  author={Tian, Yulun and Koppel, Alec and Bedi, Amrit Singh and How, Jonathan P.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Asynchronous and Parallel Distributed Pose Graph Optimization}, 
  year={2020},
  volume={5},
  number={4},
  pages={5819-5826},
  doi={10.1109/LRA.2020.3010216}}
```




