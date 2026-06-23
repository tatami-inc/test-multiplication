# Performance testing for matrix multiplication

This repository contains performance testing for different matrix multiplication strategies, 
used to inform the implementation of the various `multiply()` overloads in [**tatami_mult**](https://github.com/tatami-inc/tatami_mult).
To build the various tests, use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Check out each subdirectory's `README.md` for specific commentary on each multiplication strategy.
