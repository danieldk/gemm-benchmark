use std::io::{stdout, Write};
#[cfg(feature = "intel-mkl-amd")]
use std::os::raw::c_int;
use std::time::Duration;

use anyhow::{bail, Result};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, LinalgScalar};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use structopt::StructOpt;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "blis")]
extern crate blis_src;

#[cfg(any(feature = "intel-mkl", feature = "intel-mkl-amd"))]
extern crate intel_mkl_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

mod stats;
use stats::BenchmarkStats;

#[derive(Clone, Copy)]
enum StoppingCondition {
    TimeElapsed(Duration),
    IteratorExhausted,
}

impl StoppingCondition {
    fn done(&self, duration: Duration, has_next: bool) -> Result<bool> {
        match *self {
            Self::TimeElapsed(max_duration) => {
                if duration >= max_duration && !has_next {
                    bail!("Maximum duration has not passed, but the iterator is exhausted")
                } else {
                    Ok(duration >= max_duration)
                }
            }
            Self::IteratorExhausted => Ok(!has_next),
        }
    }
}

fn exponential_iterations() -> impl Iterator<Item = usize> {
    itertools::unfold(0, |v| {
        Some(if *v == 0 {
            *v = 1;
            1
        } else {
            *v *= 2;
            *v
        })
    })
}

fn triangular(n_samples: usize, duration: Duration, nanos_per_iter: u128) -> (usize, usize) {
    let triangular = n_samples * (n_samples + 1) / 2;
    let d = ((duration.as_nanos() as f64 / nanos_per_iter as f64 / triangular as f64).ceil()
        as usize)
        .max(1);

    (triangular, d)
}

fn triangular_iterations(
    n_samples: usize,
    duration: Duration,
    nanos_per_iter: u128,
) -> impl Iterator<Item = usize> {
    let (_triangular, d) = triangular(n_samples, duration, nanos_per_iter);

    itertools::unfold(0, move |v| {
        *v += 1;
        if *v <= n_samples {
            Some(*v * d)
        } else {
            None
        }
    })
}

#[derive(StructOpt)]
#[structopt(name = "basic")]
struct Opts {
    /// Benchmark time to approximate
    #[structopt(long, default_value = "5000")]
    benchmark_time: u64,

    /// Matrix dimensionality (N x N).
    #[structopt(short, long, default_value = "256")]
    dim: usize,

    /// Double precision (DGEMM).
    #[structopt(long)]
    dgemm: bool,

    /// The number of samples to take.
    #[structopt(short, long, default_value = "100")]
    n_samples: usize,

    /// The number of benchmark threads.
    #[structopt(short, long, default_value = "1")]
    threads: usize,

    /// Warmup time in milliseconds.
    #[structopt(long, default_value = "3000")]
    warmup_time: u64,
}

fn dim_to_flop(dim: usize, n_threads: usize, sample_iterations: usize) -> usize {
    (dim.pow(3) * 2 * sample_iterations * n_threads)
        + (dim.pow(2) * 2 * sample_iterations * n_threads)
}

fn gemm_benchmark<A, I>(
    dim: usize,
    n_threads: usize,
    stop_condition: StoppingCondition,
    n_iterations: I,
) -> Result<BenchmarkStats>
where
    A: LinalgScalar + Send + Sync,
    I: Iterator<Item = usize>,
{
    let one = A::one();
    let two = one + one;
    let point_five = one / two;

    let matrix_a: Array2<A> = Array2::from_elem((dim, dim), two);
    let matrix_b: Array2<A> = Array2::from_elem((dim, dim), point_five);
    let mut c_matrices: Vec<_> = std::iter::repeat(Array2::from_elem((dim, dim), one))
        .take(n_threads)
        .collect::<Vec<_>>();

    let mut total_iters = 0;
    let mut samples = Vec::new();

    let mut n_iterations = n_iterations.peekable();

    let bench_start = std::time::Instant::now();
    while !stop_condition.done(bench_start.elapsed(), n_iterations.peek().is_some())? {
        let start = std::time::Instant::now();

        let sample_iters = n_iterations.next().unwrap();
        total_iters += sample_iters;
        c_matrices.par_iter_mut().for_each(|mut matrix_c| {
            for _ in 0..sample_iters {
                general_mat_mul(A::one(), &matrix_a, &matrix_b, A::one(), &mut matrix_c);
            }
        });

        let elapsed = start.elapsed();
        let flops = dim_to_flop(dim, n_threads, sample_iters) as f64 / elapsed.as_secs_f64();
        samples.push(flops as f64);
    }

    Ok(BenchmarkStats::new(
        samples,
        total_iters,
        bench_start.elapsed(),
    ))
}

fn main() -> Result<()> {
    let opts: Opts = Opts::from_args();

    rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build_global()
        .unwrap();

    println!(
        "Threads: {}, samples: {}, matrix shape: {} x {}",
        opts.threads, opts.n_samples, opts.dim, opts.dim
    );

    print!("Warming up for {}ms... ", opts.warmup_time);
    stdout().lock().flush()?;

    let warmup_samples = if opts.dgemm {
        gemm_benchmark::<f64, _>(
            opts.dim,
            opts.threads,
            StoppingCondition::TimeElapsed(Duration::from_millis(opts.warmup_time)),
            exponential_iterations(),
        )
    } else {
        gemm_benchmark::<f32, _>(
            opts.dim,
            opts.threads,
            StoppingCondition::TimeElapsed(Duration::from_millis(opts.warmup_time)),
            exponential_iterations(),
        )
    }?;

    println!(
        "{:.1}it/s",
        warmup_samples.total_iters() as f64 / warmup_samples.total_time().as_secs_f64()
    );

    let nanos_per_iter =
        warmup_samples.total_time().as_nanos() / warmup_samples.total_iters() as u128;

    let (triangular, d) = triangular(
        opts.n_samples,
        Duration::from_millis(opts.benchmark_time),
        nanos_per_iter,
    );

    print!(
        "Benchmarking {} iterations in {:.2}s ... ",
        triangular * d,
        Duration::from_nanos(triangular as u64 * d as u64 * nanos_per_iter as u64).as_secs_f64()
    );
    stdout().lock().flush()?;

    let mut samples = if opts.dgemm {
        gemm_benchmark::<f64, _>(
            opts.dim,
            opts.threads,
            StoppingCondition::IteratorExhausted,
            triangular_iterations(
                opts.n_samples,
                Duration::from_millis(opts.benchmark_time),
                nanos_per_iter,
            ),
        )
    } else {
        gemm_benchmark::<f32, _>(
            opts.dim,
            opts.threads,
            StoppingCondition::IteratorExhausted,
            triangular_iterations(
                opts.n_samples,
                Duration::from_millis(opts.benchmark_time),
                nanos_per_iter,
            ),
        )
    }?;

    println!("done");

    let mut samples_without_outliers = samples.without_outliers();

    println!(
        "Median: {:.2}, mean: {:.2}, stddev: {:.2} GFLOPS/s (removed {} outliers)",
        samples_without_outliers.median() / 1_000_000_000.,
        samples_without_outliers.mean().unwrap() / 1_000_000_000.,
        samples_without_outliers.std_dev().unwrap() / 1_000_000_000.,
        samples.len() - samples_without_outliers.len()
    );

    Ok(())
}

#[cfg(feature = "intel-mkl-amd")]
#[allow(dead_code)]
#[no_mangle]
extern "C" fn mkl_serv_intel_cpu_true() -> c_int {
    1
}
