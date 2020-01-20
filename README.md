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

The built executables are located in directory build/bin. 

For a demo of multithreaded PGO, run:
```
./C++/build/bin/multithread-example data/smallGrid3D.g2o
```

For a demo of distributed PGO (serial version), run:
```
./C++/build/bin/distributed-example 5 data/smallGrid3D.g2o
```
