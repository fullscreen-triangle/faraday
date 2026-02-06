//! S-coordinate system for categorical entropy space.
//!
//! Position in categorical S-entropy space:
//! - S_k: Knowledge entropy - uncertainty in state (which partition)
//! - S_t: Temporal entropy - uncertainty in timing (when)
//! - S_e: Evolution entropy - uncertainty in trajectory (how)
//!
//! These are NOT approximations - they ARE the fundamental coordinates.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{FaradayError, FaradayResult};

/// Position in categorical S-entropy space [0,1]³.
///
/// The S-coordinate represents a point in the bounded categorical phase space.
/// Each component is clamped to the unit interval [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SCoordinate {
    /// Knowledge entropy: uncertainty in state (which partition)
    s_k: f64,
    /// Temporal entropy: uncertainty in timing (when)
    s_t: f64,
    /// Evolution entropy: uncertainty in trajectory (how)
    s_e: f64,
}

impl SCoordinate {
    /// Create a new S-coordinate, clamping values to [0, 1].
    pub fn new(s_k: f64, s_t: f64, s_e: f64) -> Self {
        Self {
            s_k: s_k.clamp(0.0, 1.0),
            s_t: s_t.clamp(0.0, 1.0),
            s_e: s_e.clamp(0.0, 1.0),
        }
    }

    /// Create a new S-coordinate with strict bounds checking.
    pub fn try_new(s_k: f64, s_t: f64, s_e: f64) -> FaradayResult<Self> {
        if !(0.0..=1.0).contains(&s_k) {
            return Err(FaradayError::invalid_s_coordinate("S_k", s_k));
        }
        if !(0.0..=1.0).contains(&s_t) {
            return Err(FaradayError::invalid_s_coordinate("S_t", s_t));
        }
        if !(0.0..=1.0).contains(&s_e) {
            return Err(FaradayError::invalid_s_coordinate("S_e", s_e));
        }
        Ok(Self { s_k, s_t, s_e })
    }

    /// Create S-coordinate from hardware timing.
    ///
    /// The precision-by-difference ΔP = T_ref - t_local determines position.
    /// This is NOT simulation - the hardware timing IS the categorical state.
    pub fn from_hardware_timing(reference_ns: u64, local_ns: u64) -> Self {
        let delta_p = if reference_ns > local_ns {
            reference_ns - local_ns
        } else {
            local_ns - reference_ns
        };

        // Map timing difference to S-coordinates via ternary encoding
        let s_k = (delta_p % 1000) as f64 / 1000.0; // Sub-microsecond: knowledge
        let s_t = ((delta_p / 1000) % 1000) as f64 / 1000.0; // Microsecond: temporal
        let s_e = ((delta_p / 1_000_000) % 1000) as f64 / 1000.0; // Millisecond: evolution

        Self { s_k, s_t, s_e }
    }

    /// Create S-coordinate by sampling current hardware timing.
    pub fn sample_now(reference: Instant) -> Self {
        let elapsed = reference.elapsed();
        let nanos = elapsed.as_nanos() as u64;
        Self::from_hardware_timing(0, nanos)
    }

    /// Knowledge entropy component.
    #[inline]
    pub fn s_k(&self) -> f64 {
        self.s_k
    }

    /// Temporal entropy component.
    #[inline]
    pub fn s_t(&self) -> f64 {
        self.s_t
    }

    /// Evolution entropy component.
    #[inline]
    pub fn s_e(&self) -> f64 {
        self.s_e
    }

    /// Distance from origin in S-space.
    pub fn magnitude(&self) -> f64 {
        (self.s_k.powi(2) + self.s_t.powi(2) + self.s_e.powi(2)).sqrt()
    }

    /// Total categorical entropy (sum of components).
    pub fn total_entropy(&self) -> f64 {
        self.s_k + self.s_t + self.s_e
    }

    /// Convert to array representation.
    pub fn to_array(&self) -> [f64; 3] {
        [self.s_k, self.s_t, self.s_e]
    }

    /// Create from array representation.
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Euclidean distance to another S-coordinate.
    pub fn distance_to(&self, other: &SCoordinate) -> f64 {
        let dk = self.s_k - other.s_k;
        let dt = self.s_t - other.s_t;
        let de = self.s_e - other.s_e;
        (dk.powi(2) + dt.powi(2) + de.powi(2)).sqrt()
    }

    /// Linear interpolation between two S-coordinates.
    pub fn lerp(&self, other: &SCoordinate, t: f64) -> SCoordinate {
        let t = t.clamp(0.0, 1.0);
        SCoordinate::new(
            self.s_k + t * (other.s_k - self.s_k),
            self.s_t + t * (other.s_t - self.s_t),
            self.s_e + t * (other.s_e - self.s_e),
        )
    }

    /// Origin of S-space (0, 0, 0).
    pub const ORIGIN: SCoordinate = SCoordinate {
        s_k: 0.0,
        s_t: 0.0,
        s_e: 0.0,
    };

    /// Maximum entropy point (1, 1, 1).
    pub const MAX_ENTROPY: SCoordinate = SCoordinate {
        s_k: 1.0,
        s_t: 1.0,
        s_e: 1.0,
    };
}

impl Default for SCoordinate {
    fn default() -> Self {
        Self::ORIGIN
    }
}

impl std::ops::Add for SCoordinate {
    type Output = SCoordinate;

    fn add(self, other: SCoordinate) -> SCoordinate {
        SCoordinate::new(
            self.s_k + other.s_k,
            self.s_t + other.s_t,
            self.s_e + other.s_e,
        )
    }
}

impl std::ops::Sub for SCoordinate {
    type Output = SCoordinate;

    fn sub(self, other: SCoordinate) -> SCoordinate {
        SCoordinate::new(
            self.s_k - other.s_k,
            self.s_t - other.s_t,
            self.s_e - other.s_e,
        )
    }
}

impl std::ops::Mul<f64> for SCoordinate {
    type Output = SCoordinate;

    fn mul(self, scalar: f64) -> SCoordinate {
        SCoordinate::new(self.s_k * scalar, self.s_t * scalar, self.s_e * scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamping() {
        let coord = SCoordinate::new(1.5, -0.5, 0.5);
        assert_eq!(coord.s_k(), 1.0);
        assert_eq!(coord.s_t(), 0.0);
        assert_eq!(coord.s_e(), 0.5);
    }

    #[test]
    fn test_strict_bounds() {
        assert!(SCoordinate::try_new(0.5, 0.5, 0.5).is_ok());
        assert!(SCoordinate::try_new(1.5, 0.5, 0.5).is_err());
        assert!(SCoordinate::try_new(0.5, -0.1, 0.5).is_err());
    }

    #[test]
    fn test_magnitude() {
        let coord = SCoordinate::new(0.0, 0.0, 0.0);
        assert_eq!(coord.magnitude(), 0.0);

        let coord = SCoordinate::new(1.0, 0.0, 0.0);
        assert_eq!(coord.magnitude(), 1.0);

        let coord = SCoordinate::new(1.0, 1.0, 1.0);
        assert!((coord.magnitude() - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_total_entropy() {
        let coord = SCoordinate::new(0.2, 0.3, 0.5);
        assert!((coord.total_entropy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let a = SCoordinate::new(0.0, 0.0, 0.0);
        let b = SCoordinate::new(1.0, 0.0, 0.0);
        assert!((a.distance_to(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lerp() {
        let a = SCoordinate::new(0.0, 0.0, 0.0);
        let b = SCoordinate::new(1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.s_k() - 0.5).abs() < 1e-10);
        assert!((mid.s_t() - 0.5).abs() < 1e-10);
        assert!((mid.s_e() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_arithmetic() {
        let a = SCoordinate::new(0.2, 0.3, 0.4);
        let b = SCoordinate::new(0.1, 0.2, 0.3);
        let sum = a + b;
        assert!((sum.s_k() - 0.3).abs() < 1e-10);

        let scaled = a * 0.5;
        assert!((scaled.s_k() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_hardware_timing() {
        // Test that from_hardware_timing produces valid coordinates
        let coord = SCoordinate::from_hardware_timing(1_500_000, 500_000);
        assert!((0.0..=1.0).contains(&coord.s_k()));
        assert!((0.0..=1.0).contains(&coord.s_t()));
        assert!((0.0..=1.0).contains(&coord.s_e()));
    }
}
