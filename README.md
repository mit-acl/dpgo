# DPGO


## Introduction
This repository contains implementation of synchronous and asynchronous Distributed Pose Graph Optimization (DPGO).  The algorithms are described in the following papers:

 - Y. Tian, K. Khosoussi, D. M. Rosen, J. P. How. [**Distributed Certifiably Correct Pose-Graph Optimization**](https://arxiv.org/abs/1911.03721). arXiv preprint [arXiv:1911.03721](https://arxiv.org/abs/1911.03721).
 
 - Y.Tian, A. Koppel, A. S. Bedi, J. P. How.  [**Asynchronous and Parallel Distributed Pose Graph Optimization**](https://arxiv.org/abs/2003.03281). IEEE Robotics and Automation Letters (RA-L) 2020. Full paper: [arXiv:2003.03281](https://arxiv.org/abs/2003.03281).

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

Optionally, install the C++ library via,
```
sudo make install
```






