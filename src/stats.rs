use std::time::Duration;

use statrs::statistics::{Data, Distribution, OrderStatistics};

pub struct BenchmarkStats {
    samples: Data<Vec<f64>>,
    total_iters: usize,
    total_time: Duration,
}

impl BenchmarkStats {
    pub fn new(samples: impl Into<Vec<f64>>, total_iters: usize, total_time: Duration) -> Self {
        BenchmarkStats {
            samples: Data::new(samples.into()),
            total_iters,
            total_time,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn mean(&self) -> Option<f64> {
        self.samples.mean()
    }

    pub fn median(&mut self) -> f64 {
        self.samples.median()
    }

    pub fn std_dev(&self) -> Option<f64> {
        self.samples.std_dev()
    }

    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    pub fn total_iters(&self) -> usize {
        self.total_iters
    }

    pub fn without_outliers(&mut self) -> Self {
        let lq = self.samples.lower_quartile();
        let uq = self.samples.upper_quartile();
        let iqr = self.samples.interquartile_range();

        // Tukey's method for filtering outliers.
        let no_outliers: Vec<_> = self
            .samples
            .iter()
            .map(ToOwned::to_owned)
            .filter(|&v| v >= lq - 1.5 * iqr && v <= uq + 1.5 * iqr)
            .collect();

        BenchmarkStats {
            samples: Data::new(no_outliers),
            total_iters: self.total_iters,
            total_time: self.total_time,
        }
    }
}
