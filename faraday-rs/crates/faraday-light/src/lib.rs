//! Faraday Light - Light propagation and optics for the partition physics framework.
//!
//! Light IS partition completion propagation.
//!
//! # Key Concepts
//!
//! - Light is NOT electromagnetic waves on a grid
//! - Light IS the sequential completion of partition operations
//! - Refractive index EMERGES from partition lag τ_c
//! - Absorption linewidth γ = 1/τ_c (inverse partition lag)
//!
//! # The Factor of 2
//!
//! The optical-mechanical τ_c ratio of 2.0 arises because:
//! - Mechanical: measures single partition operation (collision)
//! - Optical: resolves TWO electron "commitments" per collision
//!   (approach commitment + separation commitment)
//!
//! # Example
//!
//! ```rust
//! use faraday_core::fluid::{MolecularSpecies, PartitionFluid};
//! use faraday_light::PartitionLight;
//!
//! // Create CCl4 at room temperature
//! let fluid = PartitionFluid::from_species(MolecularSpecies::CCl4, 298.0);
//!
//! // Create light propagation through the fluid
//! let light = PartitionLight::from_fluid(&fluid, 218.0, 4);
//!
//! // Validate the key prediction
//! let validation = light.validate_tau_c_relationship();
//! assert!(validation.validated);
//! assert!((validation.ratio - 2.0).abs() < 0.01);
//! ```

pub mod light;
pub mod optics;
pub mod oscillator;
pub mod photon;

// Re-export commonly used types
pub use light::{PartitionLight, TauCValidation};
pub use optics::{
    energy_ev_to_wavelength, frequency_to_wavelength, omega_to_wavelength,
    wavelength_to_energy_ev, wavelength_to_frequency, wavelength_to_omega,
    wavelength_to_wavenumber, wavenumber_to_wavelength,
    AbsorptionCrossSection, CauchyCoefficients,
};
pub use oscillator::ElectronOscillator;
pub use photon::{PartitionOutcome, PartitionPhoton};
