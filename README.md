# DPGO

Distributed (Asynchronous) Pose-Graph Optimization

## MATLAB
TODO.

## C++ 

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

The built executables are located in directory build/bin. For a demo of multithreaded PGO, run:
```
./C++/build/bin/multithread-example data/smallGrid3D.g2o
```
