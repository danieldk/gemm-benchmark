#[cfg(feature = "intel-mkl-amd")]
use std::os::raw::c_int;
use std::time::Duration;

use criterion::measurement::{Measurement, WallTime};
use criterion::{BenchmarkGroup, Criterion, Throughput};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, LinalgScalar};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use structopt::StructOpt;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "blis")]
extern crate blis_src;

#[cfg(any(feature = "intel-mkl", feature = "intel-mkl-amd"))]
extern crate intel_mkl_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

#[derive(StructOpt)]
#[structopt(name = "basic")]
struct Opts {
    /// Matrix dimensionality (N x N).
    #[structopt(short, long, default_value = "256")]
    dim: usize,

    /// Double precision (DGEMM).
    #[structopt(long)]
    dgemm: bool,

    /// The number of gemm iterations per thread.
    #[structopt(short, long, default_value = "1000")]
    iterations: usize,

    /// The number of benchmark threads.
    #[structopt(short, long, default_value = "1")]
    threads: usize,
}

fn dim_to_flop(dim: usize) -> usize {
    (dim.pow(3) * 2) + (dim.pow(2) * 2)
}

fn gemm_benchmark<A, M>(mut group: BenchmarkGroup<M>)
where
    A: LinalgScalar + Send + Sync,
    M: Measurement,
{
    let one = A::one();
    let two = one + one;
    let point_five = one / two;

    for dim in [128, 256, 384, 512, 768, 1024, 1152, 1280, 1408, 2048] {
        let matrix_a: Array2<A> = Array2::from_elem((dim, dim), two);
        let matrix_b: Array2<A> = Array2::from_elem((dim, dim), point_five);
        let mut matrix_c = Array2::from_elem((dim, dim), one);

        group.throughput(Throughput::Elements(dim_to_flop(dim) as u64));
        group.bench_function(format!("{}x{}", dim, dim), |b| {
            b.iter(|| general_mat_mul(A::one(), &matrix_a, &matrix_b, A::one(), &mut matrix_c))
        });
    }
}

fn main() {
    let opts: Opts = Opts::from_args();

    rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build_global()
        .unwrap();

    println!("Threads: {}", opts.threads);
    println!("Iterations per thread: {}", opts.iterations);
    println!("Matrix shape: {} x {}", opts.dim, opts.dim);

    let mut criterion = Criterion::default();

    gemm_benchmark::<f32, WallTime>(criterion.benchmark_group("sgemm"));
    gemm_benchmark::<f64, WallTime>(criterion.benchmark_group("dgemm"));
}

#[cfg(feature = "intel-mkl-amd")]
#[allow(dead_code)]
#[no_mangle]
extern "C" fn mkl_serv_intel_cpu_true() -> c_int {
    1
}
