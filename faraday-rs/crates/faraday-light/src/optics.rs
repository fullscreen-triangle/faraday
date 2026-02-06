//! Optical helper functions and additional optical properties.
//!
//! Provides wavelength conversions, dispersion relations, and
//! absorption cross-section calculations.

use faraday_core::constants::{C, HBAR};
use std::f64::consts::PI;

/// Convert wavelength [nm] to angular frequency [rad/s].
#[inline]
pub fn wavelength_to_omega(wavelength_nm: f64) -> f64 {
    2.0 * PI * C / (wavelength_nm * 1e-9)
}

/// Convert angular frequency [rad/s] to wavelength [nm].
#[inline]
pub fn omega_to_wavelength(omega: f64) -> f64 {
    2.0 * PI * C / omega * 1e9
}

/// Convert frequency [Hz] to wavelength [nm].
#[inline]
pub fn frequency_to_wavelength(frequency: f64) -> f64 {
    C / frequency * 1e9
}

/// Convert wavelength [nm] to frequency [Hz].
#[inline]
pub fn wavelength_to_frequency(wavelength_nm: f64) -> f64 {
    C / (wavelength_nm * 1e-9)
}

/// Convert photon energy [eV] to wavelength [nm].
#[inline]
pub fn energy_ev_to_wavelength(energy_ev: f64) -> f64 {
    // λ = hc/E
    let energy_j = energy_ev * 1.602176634e-19;
    HBAR * 2.0 * PI * C / energy_j * 1e9
}

/// Convert wavelength [nm] to photon energy [eV].
#[inline]
pub fn wavelength_to_energy_ev(wavelength_nm: f64) -> f64 {
    let omega = wavelength_to_omega(wavelength_nm);
    HBAR * omega / 1.602176634e-19
}

/// Wavenumber [cm⁻¹] to wavelength [nm].
#[inline]
pub fn wavenumber_to_wavelength(wavenumber_cm: f64) -> f64 {
    1e7 / wavenumber_cm
}

/// Wavelength [nm] to wavenumber [cm⁻¹].
#[inline]
pub fn wavelength_to_wavenumber(wavelength_nm: f64) -> f64 {
    1e7 / wavelength_nm
}

/// Cauchy dispersion coefficients for common materials.
#[derive(Debug, Clone, Copy)]
pub struct CauchyCoefficients {
    /// A coefficient (n at infinite wavelength)
    pub a: f64,
    /// B coefficient [nm²]
    pub b: f64,
    /// C coefficient [nm⁴] (optional)
    pub c: f64,
}

impl CauchyCoefficients {
    /// Create new Cauchy coefficients.
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    /// Fused silica (SiO₂).
    pub const FUSED_SILICA: Self = Self {
        a: 1.4580,
        b: 3540.0,
        c: 0.0,
    };

    /// BK7 optical glass.
    pub const BK7: Self = Self {
        a: 1.5046,
        b: 4200.0,
        c: 0.0,
    };

    /// Water at 20°C.
    pub const WATER: Self = Self {
        a: 1.3325,
        b: 3119.0,
        c: 0.0,
    };

    /// Refractive index at given wavelength [nm].
    pub fn refractive_index(&self, wavelength_nm: f64) -> f64 {
        let lambda2 = wavelength_nm * wavelength_nm;
        let lambda4 = lambda2 * lambda2;
        self.a + self.b / lambda2 + self.c / lambda4
    }

    /// Group refractive index at given wavelength [nm].
    pub fn group_index(&self, wavelength_nm: f64) -> f64 {
        let n = self.refractive_index(wavelength_nm);
        let dn_dlambda = self.dn_dlambda(wavelength_nm);
        n - wavelength_nm * dn_dlambda
    }

    /// Derivative dn/dλ [nm⁻¹].
    pub fn dn_dlambda(&self, wavelength_nm: f64) -> f64 {
        let lambda3 = wavelength_nm.powi(3);
        let lambda5 = wavelength_nm.powi(5);
        -2.0 * self.b / lambda3 - 4.0 * self.c / lambda5
    }
}

/// Absorption cross-section calculation.
#[derive(Debug, Clone, Copy)]
pub struct AbsorptionCrossSection {
    /// Peak cross-section [m²]
    pub sigma_0: f64,
    /// Center frequency [rad/s]
    pub omega_0: f64,
    /// Line width (FWHM) [rad/s]
    pub gamma: f64,
}

impl AbsorptionCrossSection {
    /// Create new absorption cross-section with Lorentzian profile.
    pub fn lorentzian(sigma_0: f64, omega_0: f64, gamma: f64) -> Self {
        Self {
            sigma_0,
            omega_0,
            gamma,
        }
    }

    /// Cross-section at given frequency [m²].
    ///
    /// Lorentzian profile: σ(ω) = σ₀ × (γ/2)² / [(ω-ω₀)² + (γ/2)²]
    pub fn at_frequency(&self, omega: f64) -> f64 {
        let delta = omega - self.omega_0;
        let half_gamma = self.gamma / 2.0;
        self.sigma_0 * half_gamma.powi(2) / (delta.powi(2) + half_gamma.powi(2))
    }

    /// Cross-section at given wavelength [nm] → [m²].
    pub fn at_wavelength(&self, wavelength_nm: f64) -> f64 {
        let omega = wavelength_to_omega(wavelength_nm);
        self.at_frequency(omega)
    }

    /// Integrated cross-section (area under curve) [m²·rad/s].
    pub fn integrated(&self) -> f64 {
        // For Lorentzian: ∫σdω = π×σ₀×γ/2
        PI * self.sigma_0 * self.gamma / 2.0
    }
}

/// Common spectral lines.
pub mod spectral_lines {
    /// Sodium D1 line wavelength [nm].
    pub const SODIUM_D1: f64 = 589.592;
    /// Sodium D2 line wavelength [nm].
    pub const SODIUM_D2: f64 = 588.995;
    /// Hydrogen alpha line wavelength [nm].
    pub const H_ALPHA: f64 = 656.281;
    /// Hydrogen beta line wavelength [nm].
    pub const H_BETA: f64 = 486.135;
    /// Helium-Neon laser wavelength [nm].
    pub const HENE_LASER: f64 = 632.8;
    /// Nd:YAG laser fundamental wavelength [nm].
    pub const NDYAG: f64 = 1064.0;
    /// Nd:YAG second harmonic wavelength [nm].
    pub const NDYAG_SHG: f64 = 532.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelength_roundtrip() {
        let wavelength = 550.0;
        let omega = wavelength_to_omega(wavelength);
        let recovered = omega_to_wavelength(omega);
        assert!((wavelength - recovered).abs() < 1e-10);
    }

    #[test]
    fn test_energy_wavelength() {
        // Green light at 550 nm should be ~2.25 eV
        let energy = wavelength_to_energy_ev(550.0);
        assert!((energy - 2.25).abs() < 0.1);

        // And convert back
        let wavelength = energy_ev_to_wavelength(energy);
        assert!((wavelength - 550.0).abs() < 0.1);
    }

    #[test]
    fn test_cauchy_dispersion() {
        let n_589 = CauchyCoefficients::FUSED_SILICA.refractive_index(589.0);
        // Fused silica at sodium D line is about 1.458-1.468
        // Cauchy equation is an approximation, so use wider tolerance
        assert!(n_589 > 1.45 && n_589 < 1.48, "n={} out of expected range", n_589);
    }

    #[test]
    fn test_absorption_lorentzian() {
        let omega_0 = 1e15;
        let gamma = 1e12;
        let sigma_0 = 1e-20;

        let cross_section = AbsorptionCrossSection::lorentzian(sigma_0, omega_0, gamma);

        // At center, should equal sigma_0
        let sigma_center = cross_section.at_frequency(omega_0);
        assert!((sigma_center - sigma_0).abs() / sigma_0 < 1e-10);

        // At half-width, should be half
        let sigma_half = cross_section.at_frequency(omega_0 + gamma / 2.0);
        assert!((sigma_half - sigma_0 / 2.0).abs() / sigma_0 < 1e-10);
    }

    #[test]
    fn test_wavenumber_conversion() {
        // 10000 cm⁻¹ = 1000 nm
        let wavelength = wavenumber_to_wavelength(10000.0);
        assert!((wavelength - 1000.0).abs() < 1e-10);
    }
}
