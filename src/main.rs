#[cfg(feature = "intel-mkl-amd")]
use std::os::raw::c_int;
use std::time::Duration;

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

struct BenchmarkStats {
    elapsed: Duration,
    flops: usize,
}

fn gemm_benchmark<A>(dim: usize, iterations: usize, threads: usize) -> BenchmarkStats
where
    A: LinalgScalar + Send + Sync,
{
    let one = A::one();
    let two = one + one;
    let point_five = one / two;

    let matrix_a: Array2<A> = Array2::from_elem((dim, dim), two);
    let matrix_b: Array2<A> = Array2::from_elem((dim, dim), point_five);
    let c_matrices: Vec<_> = std::iter::repeat(Array2::from_elem((dim, dim), one))
        .take(threads)
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();

    c_matrices.into_par_iter().for_each(|mut matrix_c| {
        for _ in 0..iterations {
            general_mat_mul(A::one(), &matrix_a, &matrix_b, A::one(), &mut matrix_c);
        }
    });

    let elapsed = start.elapsed();

    BenchmarkStats {
        elapsed,
        flops: (dim.pow(3) * 2 * iterations * threads) + (dim.pow(2) * 2 * iterations * threads),
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

    let stats = if opts.dgemm {
        gemm_benchmark::<f64>(opts.dim, opts.iterations, opts.threads)
    } else {
        gemm_benchmark::<f32>(opts.dim, opts.iterations, opts.threads)
    };

    println!(
        "GFLOPS/s: {:.2}",
        (stats.flops as f64 / stats.elapsed.as_secs_f64()) / 1000_000_000.
    );
}

#[cfg(feature = "intel-mkl-amd")]
#[allow(dead_code)]
#[no_mangle]
extern "C" fn mkl_serv_intel_cpu_true() -> c_int {
    1
}
