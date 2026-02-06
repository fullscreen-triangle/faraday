//! Ternary trisection search algorithm.
//!
//! O(log₃N) complexity, providing 37% fewer iterations than binary search.
//!
//! The ternary search divides the search space into three regions at each step,
//! corresponding to the fundamental ternary structure of partition physics.

use std::cmp::Ordering;

/// Result of a ternary search operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TernarySearchResult {
    /// Index of found element (if found)
    pub index: Option<usize>,
    /// Number of iterations performed
    pub iterations: u32,
}

/// Ternary search on a sorted slice.
///
/// Returns the index of the target if found, along with iteration count.
/// Complexity: O(log₃N)
pub fn ternary_search<T: Ord>(slice: &[T], target: &T) -> TernarySearchResult {
    if slice.is_empty() {
        return TernarySearchResult {
            index: None,
            iterations: 0,
        };
    }

    let mut low = 0usize;
    let mut high = slice.len() - 1;
    let mut iterations = 0u32;

    while low <= high {
        iterations += 1;

        // Divide into three regions
        let third = (high - low) / 3;
        let mid1 = low + third;
        let mid2 = if third == 0 { high } else { high - third };

        match target.cmp(&slice[mid1]) {
            Ordering::Equal => {
                return TernarySearchResult {
                    index: Some(mid1),
                    iterations,
                };
            }
            Ordering::Less => {
                if mid1 == 0 {
                    break;
                }
                high = mid1 - 1;
                continue;
            }
            Ordering::Greater => {}
        }

        match target.cmp(&slice[mid2]) {
            Ordering::Equal => {
                return TernarySearchResult {
                    index: Some(mid2),
                    iterations,
                };
            }
            Ordering::Greater => {
                low = mid2 + 1;
            }
            Ordering::Less => {
                low = mid1 + 1;
                if mid2 == 0 {
                    break;
                }
                high = mid2 - 1;
            }
        }

        // Prevent infinite loop
        if low > high || high >= slice.len() {
            break;
        }
    }

    TernarySearchResult {
        index: None,
        iterations,
    }
}

/// Binary search for comparison (returns iteration count).
pub fn binary_search_iterations<T: Ord>(slice: &[T], target: &T) -> TernarySearchResult {
    if slice.is_empty() {
        return TernarySearchResult {
            index: None,
            iterations: 0,
        };
    }

    let mut low = 0usize;
    let mut high = slice.len() - 1;
    let mut iterations = 0u32;

    while low <= high {
        iterations += 1;
        let mid = low + (high - low) / 2;

        match target.cmp(&slice[mid]) {
            Ordering::Equal => {
                return TernarySearchResult {
                    index: Some(mid),
                    iterations,
                };
            }
            Ordering::Less => {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
            Ordering::Greater => {
                low = mid + 1;
            }
        }
    }

    TernarySearchResult {
        index: None,
        iterations,
    }
}

/// Calculate theoretical binary search iterations for size N.
#[inline]
pub fn binary_iterations_theoretical(n: usize) -> u32 {
    if n <= 1 {
        return 0;
    }
    (n as f64).log2().ceil() as u32
}

/// Calculate theoretical ternary search iterations for size N.
#[inline]
pub fn ternary_iterations_theoretical(n: usize) -> u32 {
    if n <= 1 {
        return 0;
    }
    (n as f64).log(3.0).ceil() as u32
}

/// Calculate the theoretical speedup of ternary over binary.
///
/// Speedup = (log₂N - log₃N) / log₂N = 1 - log₂(3)⁻¹ ≈ 0.369 = 37%
pub fn theoretical_speedup(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let binary = binary_iterations_theoretical(n) as f64;
    let ternary = ternary_iterations_theoretical(n) as f64;
    if binary > 0.0 {
        (binary - ternary) / binary * 100.0
    } else {
        0.0
    }
}

/// The asymptotic speedup factor: 1 - 1/log₂(3) ≈ 36.9%
pub const ASYMPTOTIC_SPEEDUP: f64 = 36.907024642854;

/// Spatial localization after i ternary iterations.
///
/// Δr(i) = Δr₀ × 3^(-i)
///
/// This exponential convergence is faster than binary search's 2^(-i).
pub fn spatial_localization(initial_uncertainty: f64, iterations: u32) -> f64 {
    initial_uncertainty * 3.0_f64.powi(-(iterations as i32))
}

/// Compare binary vs ternary localization after same number of iterations.
pub fn localization_comparison(initial_uncertainty: f64, iterations: u32) -> LocalizationComparison {
    let binary = initial_uncertainty * 2.0_f64.powi(-(iterations as i32));
    let ternary = initial_uncertainty * 3.0_f64.powi(-(iterations as i32));
    let improvement = (binary - ternary) / binary * 100.0;

    LocalizationComparison {
        binary_uncertainty: binary,
        ternary_uncertainty: ternary,
        improvement_percent: improvement,
    }
}

/// Comparison of localization between binary and ternary search.
#[derive(Debug, Clone, Copy)]
pub struct LocalizationComparison {
    /// Uncertainty after binary search iterations
    pub binary_uncertainty: f64,
    /// Uncertainty after ternary search iterations
    pub ternary_uncertainty: f64,
    /// Improvement percentage
    pub improvement_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_search_found() {
        let data: Vec<i32> = (0..1000).collect();
        let result = ternary_search(&data, &500);
        assert_eq!(result.index, Some(500));
    }

    #[test]
    fn test_ternary_search_not_found() {
        let data: Vec<i32> = (0..1000).step_by(2).collect();
        let result = ternary_search(&data, &501);
        assert_eq!(result.index, None);
    }

    #[test]
    fn test_binary_search_found() {
        let data: Vec<i32> = (0..1000).collect();
        let result = binary_search_iterations(&data, &500);
        assert_eq!(result.index, Some(500));
    }

    #[test]
    fn test_ternary_fewer_iterations() {
        let data: Vec<i32> = (0..10000).collect();
        let target = 5000;

        let ternary = ternary_search(&data, &target);
        let binary = binary_search_iterations(&data, &target);

        // Ternary should use fewer iterations
        assert!(
            ternary.iterations <= binary.iterations,
            "Ternary {} > Binary {}",
            ternary.iterations,
            binary.iterations
        );
    }

    #[test]
    fn test_theoretical_speedup() {
        // For large N, speedup should approach 37%
        let speedup = theoretical_speedup(1_000_000);
        assert!(
            (speedup - 37.0).abs() < 5.0,
            "Speedup {:.1}% not near 37%",
            speedup
        );
    }

    #[test]
    fn test_spatial_localization() {
        let initial = 1.0;
        let after_10 = spatial_localization(initial, 10);

        // 3^-10 ≈ 1.69e-5
        let expected = 3.0_f64.powi(-10);
        assert!((after_10 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_localization_comparison() {
        let comp = localization_comparison(1.0, 10);

        // Ternary should be more precise
        assert!(comp.ternary_uncertainty < comp.binary_uncertainty);
        assert!(comp.improvement_percent > 0.0);
    }

    #[test]
    fn test_empty_slice() {
        let data: Vec<i32> = vec![];
        let result = ternary_search(&data, &42);
        assert_eq!(result.index, None);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_single_element() {
        let data = vec![42];
        let result = ternary_search(&data, &42);
        assert_eq!(result.index, Some(0));
    }
}
