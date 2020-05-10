Implements Matrix Vector product for row major matrices with [2|4|8] columns, using avx-2 intrinsics.  
Requires Conan package manager.
## Instructions
[Tested on Linux with Clang 10]
```
$ git clone https://github.com/kitegi/matvec-prod.git
$ cd matvec-prod
$ cmake -B build -H. -DCMAKE_BUILD_TYPE=Release -DARCH_NATIVE=ON -DENABLE_IPO=ON
$ cmake --build build --parallel 16
$ ./build/bin/check
$ for f in ./build/bin/f*; do $f; done && python draw_plots.py bench_out/*
```

## Plots
Output on my machine:

![f32-2](https://user-images.githubusercontent.com/40109184/81493920-e1466400-92a4-11ea-900c-50d046c2f69f.png)
![f32-4](https://user-images.githubusercontent.com/40109184/81493924-ec00f900-92a4-11ea-830a-174264128ab1.png)
![f32-8](https://user-images.githubusercontent.com/40109184/81493925-ee635300-92a4-11ea-8f5d-1a07fb002ad5.png)
![f64-2](https://user-images.githubusercontent.com/40109184/81493929-f3280700-92a4-11ea-9f51-c9938daf1c52.png)
![f64-4](https://user-images.githubusercontent.com/40109184/81493934-f58a6100-92a4-11ea-8f17-758e547e288d.png)
![f64-8](https://user-images.githubusercontent.com/40109184/81493938-f8855180-92a4-11ea-84f5-731bceadb300.png)
