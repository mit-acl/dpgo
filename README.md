# dpgo


## Introduction
This repository contains implementation of synchronous and asynchronous Distributed Pose Graph Optimization (DPGO).  The algorithms are described in the following publications:

 - Y. Tian, K. Khosoussi, D. M. Rosen, J. P. How. [**Distributed Certifiably Correct Pose-Graph Optimization**](https://arxiv.org/abs/1911.03721), in IEEE Transactions on Robotics, 2021, **honorable mention for 2021 King-Sun Fu Memorial Best Paper Award**.
 
 - Y.Tian, A. Koppel, A. S. Bedi, J. P. How.  [**Asynchronous and Parallel Distributed Pose Graph Optimization**](https://arxiv.org/abs/2003.03281), in IEEE Robotics and Automation Letters, 2020, **honorable mention for 2020 RA-L best paper**. 

## Building the C++ Library 

Install dependencies.

```
sudo apt-get install build-essential cmake-gui libsuitesparse-dev libboost-all-dev libeigen3-dev libgoogle-glog-dev
```

Inside the C++ directory, execute the following commands.

```
mkdir build
cd build
cmake ../
make
```

## Running a minimal example

The built executables are located in directory build/bin. For a minimal demo of distributed PGO on one of the benchmark datasets, inside the build directory run:
```
./bin/multi-robot-example 5 ../data/smallGrid3D.g2o
```

Optionally, run the unit tests by,
```
./bin/testDPGO
```

## More Examples in ROS

A ROS wrapper of dpgo is provided: [dpgo_ros](https://github.com/mit-acl/dpgo_ros). The ROS extension also provides examples for using the complete set of features implemented in dpgo. These include running the asynchronous version, speeding up convergence with Nesterov acceleration, and using robust optimization on real-world datasets to reject outlier measurements. To try out these examples, please checkout the [README](https://github.com/mit-acl/dpgo_ros).

## Usage in multi-robot collaborative SLAM

DPGO is currently used as the distributed back-end in [Kimera-Multi](https://github.com/MIT-SPARK/Kimera-Multi), which is a robust and fully distributed system for multi-robot collaborative SLAM. Check out the [full system](https://github.com/MIT-SPARK/Kimera-Multi) as well as the accompanying [datasets](https://github.com/MIT-SPARK/Kimera-Multi-Data)!

## Citations

If you are using the dpgo library, please cite the following papers. For the basic dpgo library,
```
@ARTICLE{Tian2021Distributed,
  author={Tian, Yulun and Khosoussi, Kasra and Rosen, David M. and How, Jonathan P.},
  journal={IEEE Transactions on Robotics}, 
  title={Distributed Certifiably Correct Pose-Graph Optimization}, 
  year={2021},
  volume={37},
  number={6},
  pages={2137-2156},
  doi={10.1109/TRO.2021.3072346}}
```
In addition, the extension to asynchronous optimization is described in,
```
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
Lastly, the extension to outlier-robust optimization is described in,
```
@ARTICLE{tian22tro_kimeramulti,
  author={Tian, Yulun and Chang, Yun and Herrera Arias, Fernando and Nieto-Granda, Carlos and How, Jonathan P. and Carlone, Luca},
  journal={IEEE Transactions on Robotics}, 
  title={Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems}, 
  year={2022},
  volume={38},
  number={4},
  pages={2022-2038},
  doi={10.1109/TRO.2021.3137751}
}
```

