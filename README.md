# `[sd]gemm` benchmark

## Introduction

This is a small `[sd]gemm` benchmark based, similar to
[ACES DGEMM](https://www.lanl.gov/projects/crossroads/benchmarks-performance-analysis.php),
implemented in Rust. It supports the following BLAS libraries:

* Accelerate (macOS)
* Intel MKL
* OpenBLAS

## Building

### Build with Accelerate (macOS)

```
$ cargo build --features accelerate --release
```
### Build with Intel MKL

To build the benchmark with Intel MKL statically linked, use:

```
$ cargo build --features intel-mkl --release
```

Intel MKL uses Zen-specific `[sd]gemm`kernels on AMD Zen CPUs.
However, these kernels are slower on many Zen CPUs than the AVX2
kernels. You can build the benchmark to override Intel CPU
detection, so that MKL uses AVX2 kernels on Zen CPUs as well. This
does require dynamic linking, since it is not permitted to modify
MKL binaries. To enable this override, use the `intel-mkl-amd`
feature:

```
$ cargo build --features intel-mkl-amd --release
```

### Build with OpenBLAS

```shell
$ cargo build --features openblas --release
```

Set `OPENBLAS_NUM_THREADS=1` before running.

## Benchmarking

By default, `sgemm` is benchmarked using *256 x 256* matrices, for
*1,000* iterations and *1* thread. The dimensionality (`-d`), number
of iterations (`-i`), and the number of threads (`-t`) can be set
with command-line flags. For example:

```shell
$ gemm-benchmark -d 1024 -i 2000 -t 4
```

Runs the benchmark using *1024 x 1024* matrices, for *1,000* iterations,
and *4* threads. It is also possible to benchmark `dgem,` using the
`--dgemm` option:

```shell
$ gemm-benchmark -d 1024 -i 2000 -t 4 --dgemm
```
