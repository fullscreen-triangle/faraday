//! Partition photon representation.
//!
//! A photon IS a cascade of partition completions.
//!
//! NOT: An electromagnetic wave packet
//! IS: The information encoded in sequential partition outcomes
//!
//! The photon's "position" is its current S-coordinate.
//! The photon's "trajectory" is the sequence of partition trits.

use faraday_core::coordinate::SCoordinate;
use faraday_core::constants::{C, HBAR};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Ternary partition outcome for photon propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PartitionOutcome {
    /// No interaction (pass through)
    PassThrough = 0,
    /// Elastic scatter (phase shift)
    ElasticScatter = 1,
    /// Absorption/re-emission (energy exchange)
    AbsorptionReemission = 2,
}

impl PartitionOutcome {
    /// Convert from u8 value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::PassThrough),
            1 => Some(Self::ElasticScatter),
            2 => Some(Self::AbsorptionReemission),
            _ => None,
        }
    }

    /// Convert to u8 value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// A photon IS a cascade of partition completions.
///
/// The photon's trajectory through a medium IS encoded as a sequence
/// of ternary partition outcomes. Each molecular encounter produces
/// one of three outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionPhoton {
    /// Angular frequency [rad/s]
    omega: f64,
    /// Current S-coordinate
    s_coordinate: SCoordinate,
    /// Partition outcomes (trit string)
    trit_string: Vec<PartitionOutcome>,
}

impl PartitionPhoton {
    /// Create a new photon at given angular frequency.
    pub fn new(omega: f64) -> Self {
        Self {
            omega,
            s_coordinate: SCoordinate::new(0.5, 0.5, 0.5),
            trit_string: Vec::new(),
        }
    }

    /// Create photon from wavelength [nm].
    pub fn from_wavelength(wavelength_nm: f64) -> Self {
        let omega = super::wavelength_to_omega(wavelength_nm);
        Self::new(omega)
    }

    /// Angular frequency [rad/s].
    #[inline]
    pub fn omega(&self) -> f64 {
        self.omega
    }

    /// Wavelength in vacuum [m].
    pub fn wavelength(&self) -> f64 {
        2.0 * PI * C / self.omega
    }

    /// Wavelength in vacuum [nm].
    pub fn wavelength_nm(&self) -> f64 {
        self.wavelength() * 1e9
    }

    /// Frequency [Hz].
    pub fn frequency(&self) -> f64 {
        self.omega / (2.0 * PI)
    }

    /// Photon energy E = ℏω [J].
    pub fn energy(&self) -> f64 {
        HBAR * self.omega
    }

    /// Photon energy [eV].
    pub fn energy_ev(&self) -> f64 {
        self.energy() / 1.602176634e-19
    }

    /// Wavenumber k = ω/c [m⁻¹].
    pub fn wavenumber(&self) -> f64 {
        self.omega / C
    }

    /// Current S-coordinate.
    pub fn s_coordinate(&self) -> &SCoordinate {
        &self.s_coordinate
    }

    /// Trit string (partition outcomes).
    pub fn trit_string(&self) -> &[PartitionOutcome] {
        &self.trit_string
    }

    /// Number of partition operations encountered.
    pub fn n_operations(&self) -> usize {
        self.trit_string.len()
    }

    /// Photon encounters a partition operation.
    ///
    /// Each outcome updates the S-coordinate, encoding the trajectory
    /// through partition space.
    pub fn propagate_through_partition(&mut self, outcome: PartitionOutcome) {
        self.trit_string.push(outcome);

        // Update S-coordinates based on outcome
        // Delta decreases with each operation (ternary refinement)
        let delta = 1.0 / 3.0_f64.powi(self.trit_string.len() as i32);

        // Get current coordinates
        let (s_k, s_t, s_e) = (
            self.s_coordinate.s_k(),
            self.s_coordinate.s_t(),
            self.s_coordinate.s_e(),
        );

        // Update based on outcome type
        self.s_coordinate = match outcome {
            PartitionOutcome::PassThrough => SCoordinate::new(s_k + delta, s_t, s_e),
            PartitionOutcome::ElasticScatter => SCoordinate::new(s_k, s_t + delta, s_e),
            PartitionOutcome::AbsorptionReemission => SCoordinate::new(s_k, s_t, s_e + delta),
        };
    }

    /// Information content of trajectory [bits].
    ///
    /// Each ternary decision contributes log₂(3) ≈ 1.585 bits.
    pub fn trajectory_entropy(&self) -> f64 {
        self.trit_string.len() as f64 * 3.0_f64.log2()
    }

    /// Trajectory as numeric trit string.
    pub fn trajectory_as_trits(&self) -> Vec<u8> {
        self.trit_string.iter().map(|o| o.as_u8()).collect()
    }

    /// Count of each outcome type.
    pub fn outcome_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for outcome in &self.trit_string {
            counts[outcome.as_u8() as usize] += 1;
        }
        counts
    }
}

impl Default for PartitionPhoton {
    fn default() -> Self {
        // Default: visible light at 550 nm (green)
        Self::from_wavelength(550.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photon_energy() {
        // Green light at 550 nm should have energy ~2.25 eV
        let photon = PartitionPhoton::from_wavelength(550.0);
        let energy_ev = photon.energy_ev();
        assert!((energy_ev - 2.25).abs() < 0.1);
    }

    #[test]
    fn test_wavelength_roundtrip() {
        let wavelength_nm = 589.0;
        let photon = PartitionPhoton::from_wavelength(wavelength_nm);
        assert!((photon.wavelength_nm() - wavelength_nm).abs() < 0.01);
    }

    #[test]
    fn test_propagation() {
        let mut photon = PartitionPhoton::from_wavelength(550.0);

        photon.propagate_through_partition(PartitionOutcome::PassThrough);
        photon.propagate_through_partition(PartitionOutcome::ElasticScatter);
        photon.propagate_through_partition(PartitionOutcome::AbsorptionReemission);

        assert_eq!(photon.n_operations(), 3);
        assert_eq!(photon.outcome_counts(), [1, 1, 1]);
    }

    #[test]
    fn test_trajectory_entropy() {
        let mut photon = PartitionPhoton::new(1e15);

        for _ in 0..10 {
            photon.propagate_through_partition(PartitionOutcome::PassThrough);
        }

        // 10 ternary decisions = 10 * log2(3) ≈ 15.85 bits
        let entropy = photon.trajectory_entropy();
        assert!((entropy - 10.0 * 3.0_f64.log2()).abs() < 1e-10);
    }

    #[test]
    fn test_outcome_conversion() {
        assert_eq!(PartitionOutcome::from_u8(0), Some(PartitionOutcome::PassThrough));
        assert_eq!(PartitionOutcome::from_u8(1), Some(PartitionOutcome::ElasticScatter));
        assert_eq!(PartitionOutcome::from_u8(2), Some(PartitionOutcome::AbsorptionReemission));
        assert_eq!(PartitionOutcome::from_u8(3), None);
    }
}
