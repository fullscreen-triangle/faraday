//! Physical constants for the Faraday framework.
//!
//! These constants define the fundamental physical quantities used throughout
//! the partition physics framework.

/// Boltzmann constant [J/K]
pub const K_B: f64 = 1.380649e-23;

/// Avogadro's number [mol⁻¹]
pub const N_A: f64 = 6.02214076e23;

/// Planck constant [J·s]
pub const H: f64 = 6.62607015e-34;

/// Reduced Planck constant ℏ = h/(2π) [J·s]
pub const HBAR: f64 = H / (2.0 * std::f64::consts::PI);

/// Speed of light in vacuum [m/s]
pub const C: f64 = 299792458.0;

/// Elementary charge [C]
pub const E_CHARGE: f64 = 1.602176634e-19;

/// Electron mass [kg]
pub const M_ELECTRON: f64 = 9.1093837015e-31;

/// Proton mass [kg]
pub const M_PROTON: f64 = 1.67262192369e-27;

/// Fine structure constant α ≈ 1/137 (dimensionless)
pub const ALPHA: f64 = 7.2973525693e-3;

/// Bohr radius [m]
pub const A_0: f64 = 5.29177210903e-11;

/// Rydberg energy [J]
pub const E_RYDBERG: f64 = 2.1798723611035e-18;

/// Vacuum permittivity ε₀ [F/m]
pub const EPSILON_0: f64 = 8.8541878128e-12;

/// Vacuum permeability μ₀ [H/m]
pub const MU_0: f64 = 1.25663706212e-6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hbar_relation() {
        let computed_hbar = H / (2.0 * std::f64::consts::PI);
        assert!((HBAR - computed_hbar).abs() < 1e-50);
    }

    #[test]
    fn test_speed_of_light() {
        // c² = 1/(ε₀μ₀)
        let c_squared = 1.0 / (EPSILON_0 * MU_0);
        let c_computed = c_squared.sqrt();
        let rel_error = (c_computed - C).abs() / C;
        assert!(rel_error < 1e-6);
    }

    #[test]
    fn test_fine_structure() {
        // α = e²/(4πε₀ℏc)
        let alpha_computed = E_CHARGE.powi(2)
            / (4.0 * std::f64::consts::PI * EPSILON_0 * HBAR * C);
        let rel_error = (alpha_computed - ALPHA).abs() / ALPHA;
        assert!(rel_error < 1e-6);
    }
}
