//! Electron oscillator for optical response.
//!
//! An electron oscillator in a molecular potential well is the source of
//! optical response. The electron oscillates at frequency ω₀ and is damped
//! by partition operations at rate γ.
//!
//! The partition framework reveals:
//! - ω₀ comes from the binding potential (partition boundary)
//! - γ comes from partition lag (collision-induced dephasing)

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// An electron oscillator in a molecular potential well.
///
/// This IS the source of optical response. The damping rate γ = 1/τ_c
/// connects optical properties directly to the partition structure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ElectronOscillator {
    /// Natural frequency [rad/s]
    omega_0: f64,
    /// Damping rate [rad/s] = 1/τ_c
    gamma: f64,
    /// Oscillator strength (dimensionless)
    f_osc: f64,
}

impl ElectronOscillator {
    /// Create a new electron oscillator.
    pub fn new(omega_0: f64, gamma: f64, f_osc: f64) -> Self {
        Self {
            omega_0,
            gamma,
            f_osc,
        }
    }

    /// Create oscillator from resonance wavelength and partition lag.
    pub fn from_wavelength_and_tau(wavelength_nm: f64, tau_c: f64, f_osc: f64) -> Self {
        let omega_0 = super::wavelength_to_omega(wavelength_nm);
        let gamma = 1.0 / tau_c;
        Self::new(omega_0, gamma, f_osc)
    }

    /// Natural frequency [rad/s].
    #[inline]
    pub fn omega_0(&self) -> f64 {
        self.omega_0
    }

    /// Damping rate [rad/s].
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Oscillator strength.
    #[inline]
    pub fn f_osc(&self) -> f64 {
        self.f_osc
    }

    /// Oscillation period [s].
    pub fn period(&self) -> f64 {
        2.0 * PI / self.omega_0
    }

    /// Quality factor Q = ω₀/γ.
    pub fn quality_factor(&self) -> f64 {
        if self.gamma > 0.0 {
            self.omega_0 / self.gamma
        } else {
            f64::INFINITY
        }
    }

    /// Time for phase coherence loss [s].
    pub fn dephasing_time(&self) -> f64 {
        if self.gamma > 0.0 {
            1.0 / self.gamma
        } else {
            f64::INFINITY
        }
    }

    /// Resonance wavelength [nm].
    pub fn resonance_wavelength(&self) -> f64 {
        super::omega_to_wavelength(self.omega_0)
    }

    /// Frequency [Hz].
    pub fn frequency(&self) -> f64 {
        self.omega_0 / (2.0 * PI)
    }
}

impl Default for ElectronOscillator {
    fn default() -> Self {
        // Default: UV resonance at 220 nm, typical liquid τ_c ~ 1 ps
        Self::from_wavelength_and_tau(220.0, 1e-12, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_factor() {
        let osc = ElectronOscillator::new(1e15, 1e12, 1.0);
        assert!((osc.quality_factor() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_dephasing_time() {
        let gamma = 1e12;
        let osc = ElectronOscillator::new(1e15, gamma, 1.0);
        assert!((osc.dephasing_time() - 1e-12).abs() < 1e-25);
    }

    #[test]
    fn test_period() {
        let omega_0 = 2.0 * PI * 1e15; // 1 PHz
        let osc = ElectronOscillator::new(omega_0, 1e12, 1.0);
        assert!((osc.period() - 1e-15).abs() < 1e-25);
    }
}
