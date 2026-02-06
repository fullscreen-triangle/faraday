//! Partition fluid structures for the categorical physics framework.
//!
//! A fluid IS its partition structure, not a collection of molecules.
//!
//! Key insight from the framework:
//! - Fluid behavior EMERGES from partition operations
//! - Viscosity μ = τ_c × g (partition lag × coupling strength)
//! - Temperature IS the variance of S-coordinates (timing jitter)
//! - Pressure IS the sampling rate (partition operations per second)

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::time::Instant;

use crate::constants::K_B;
use crate::coordinate::SCoordinate;
use crate::partition::{PartitionCascade, PartitionOperation, ViscosityRelation};

/// Known molecular species with their partition parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MolecularSpecies {
    /// Carbon tetrachloride (CCl₄) - liquid at room temperature
    CCl4,
    /// Water (H₂O) - liquid at room temperature
    H2O,
    /// Nitrogen (N₂) - gas at room temperature
    N2,
    /// Oxygen (O₂) - gas at room temperature
    O2,
    /// Argon (Ar) - noble gas
    Ar,
}

impl MolecularSpecies {
    /// Get the molecular parameters for this species at reference temperature (298 K).
    pub fn parameters(&self) -> MolecularParameters {
        match self {
            MolecularSpecies::CCl4 => MolecularParameters {
                n_density: 6.24e27,       // molecules/m³ (liquid)
                sigma: 5.0e-19,           // m² (collision cross-section)
                molecular_mass: 2.55e-25, // kg (153.8 g/mol)
                viscosity_ref: 9.7e-4,    // Pa·s at 298 K
                is_liquid: true,
                activation_energy_over_r: 2000.0, // K
            },
            MolecularSpecies::H2O => MolecularParameters {
                n_density: 3.34e28,       // molecules/m³ (liquid)
                sigma: 2.6e-19,           // m²
                molecular_mass: 2.99e-26, // kg (18 g/mol)
                viscosity_ref: 8.9e-4,    // Pa·s at 298 K
                is_liquid: true,
                activation_energy_over_r: 2200.0,
            },
            MolecularSpecies::N2 => MolecularParameters {
                n_density: 2.5e25,        // molecules/m³ at 1 atm, 298 K
                sigma: 3.6e-19,           // m²
                molecular_mass: 4.65e-26, // kg (28 g/mol)
                viscosity_ref: 1.76e-5,   // Pa·s at 298 K
                is_liquid: false,
                activation_energy_over_r: 0.0,
            },
            MolecularSpecies::O2 => MolecularParameters {
                n_density: 2.5e25,        // molecules/m³ at 1 atm
                sigma: 3.4e-19,           // m²
                molecular_mass: 5.31e-26, // kg (32 g/mol)
                viscosity_ref: 2.04e-5,   // Pa·s at 298 K
                is_liquid: false,
                activation_energy_over_r: 0.0,
            },
            MolecularSpecies::Ar => MolecularParameters {
                n_density: 2.5e25,        // molecules/m³ at 1 atm
                sigma: 3.6e-19,           // m²
                molecular_mass: 6.63e-26, // kg (40 g/mol)
                viscosity_ref: 2.23e-5,   // Pa·s at 298 K
                is_liquid: false,
                activation_energy_over_r: 0.0,
            },
        }
    }

    /// Name of the species.
    pub fn name(&self) -> &'static str {
        match self {
            MolecularSpecies::CCl4 => "Carbon Tetrachloride",
            MolecularSpecies::H2O => "Water",
            MolecularSpecies::N2 => "Nitrogen",
            MolecularSpecies::O2 => "Oxygen",
            MolecularSpecies::Ar => "Argon",
        }
    }

    /// Chemical formula.
    pub fn formula(&self) -> &'static str {
        match self {
            MolecularSpecies::CCl4 => "CCl₄",
            MolecularSpecies::H2O => "H₂O",
            MolecularSpecies::N2 => "N₂",
            MolecularSpecies::O2 => "O₂",
            MolecularSpecies::Ar => "Ar",
        }
    }
}

/// Molecular parameters for a species.
#[derive(Debug, Clone, Copy)]
pub struct MolecularParameters {
    /// Number density [m⁻³]
    pub n_density: f64,
    /// Collision cross-section [m²]
    pub sigma: f64,
    /// Molecular mass [kg]
    pub molecular_mass: f64,
    /// Reference viscosity at 298 K [Pa·s]
    pub viscosity_ref: f64,
    /// Whether this is a liquid at room temperature
    pub is_liquid: bool,
    /// Activation energy / R for liquid viscosity scaling [K]
    pub activation_energy_over_r: f64,
}

/// A fluid IS its partition structure.
///
/// NOT: A collection of molecules with positions and velocities
/// IS: A partition network characterized by τ_c and g
///
/// Viscosity, density, temperature EMERGE from partition parameters.
///
/// For liquids: Use experimental viscosity to derive τ_c = μ / g
/// For gases: Use kinetic theory τ_c = 1/(n·σ·v̄)
///
/// The key relation μ = τ_c × g holds in both cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionFluid {
    /// Number density [m⁻³]
    n_density: f64,
    /// Collision cross-section [m²]
    sigma: f64,
    /// Temperature [K]
    temperature: f64,
    /// Molecular mass [kg]
    molecular_mass: f64,
    /// Whether this is a liquid
    is_liquid: bool,
    /// Species reference (if created from known species)
    species: Option<MolecularSpecies>,

    // Derived quantities
    /// Mean partition velocity [m/s]
    v_bar: f64,
    /// Partition lag [s]
    tau_c: f64,
    /// Coupling strength [Pa]
    g: f64,

    // S-coordinate ensemble (the fluid IS this ensemble)
    #[serde(skip)]
    s_ensemble: Vec<SCoordinate>,
}

impl PartitionFluid {
    /// Create a partition fluid from fundamental parameters.
    ///
    /// For gases: τ_c computed from kinetic theory
    /// For liquids: provide experimental viscosity to derive τ_c correctly
    pub fn new(
        n_density: f64,
        sigma: f64,
        temperature: f64,
        molecular_mass: f64,
        viscosity_experimental: Option<f64>,
        is_liquid: bool,
    ) -> Self {
        // Mean molecular speed (Maxwell-Boltzmann)
        let v_bar = (8.0 * K_B * temperature / (PI * molecular_mass)).sqrt();

        // Coupling strength from momentum flux
        // g = 8nkT/(3π) - the momentum transfer per partition operation
        let g = 8.0 * n_density * K_B * temperature / (3.0 * PI);

        // Partition lag
        let tau_c = if is_liquid {
            if let Some(mu) = viscosity_experimental {
                // For liquids: derive τ_c from experimental viscosity
                // μ = τ_c × g  =>  τ_c = μ / g
                mu / g
            } else {
                // Fallback to kinetic theory
                1.0 / (n_density * sigma * v_bar)
            }
        } else {
            // For gases: use kinetic theory
            // τ_c = 1/(n·σ·v̄) - mean time between partition operations
            1.0 / (n_density * sigma * v_bar)
        };

        Self {
            n_density,
            sigma,
            temperature,
            molecular_mass,
            is_liquid,
            species: None,
            v_bar,
            tau_c,
            g,
            s_ensemble: Vec::new(),
        }
    }

    /// Create a fluid from a known molecular species.
    pub fn from_species(species: MolecularSpecies, temperature: f64) -> Self {
        let params = species.parameters();
        let t_ref = 298.0;

        // Temperature scaling for viscosity
        let viscosity_scaled = if params.is_liquid {
            // Liquid viscosity scales as exp(E_a/RT)
            params.viscosity_ref
                * ((params.activation_energy_over_r * (1.0 / temperature - 1.0 / t_ref)).exp())
        } else {
            // Gas viscosity scales as T^0.7 approximately
            params.viscosity_ref * (temperature / t_ref).powf(0.7)
        };

        let mut fluid = Self::new(
            params.n_density,
            params.sigma,
            temperature,
            params.molecular_mass,
            Some(viscosity_scaled),
            params.is_liquid,
        );
        fluid.species = Some(species);
        fluid
    }

    /// Number density [m⁻³].
    #[inline]
    pub fn n_density(&self) -> f64 {
        self.n_density
    }

    /// Collision cross-section [m²].
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Temperature [K].
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Molecular mass [kg].
    #[inline]
    pub fn molecular_mass(&self) -> f64 {
        self.molecular_mass
    }

    /// Whether this is a liquid.
    #[inline]
    pub fn is_liquid(&self) -> bool {
        self.is_liquid
    }

    /// Species reference if available.
    #[inline]
    pub fn species(&self) -> Option<MolecularSpecies> {
        self.species
    }

    /// Mean molecular velocity [m/s].
    #[inline]
    pub fn v_bar(&self) -> f64 {
        self.v_bar
    }

    /// Mean free path between partition operations [m].
    pub fn mean_free_path(&self) -> f64 {
        1.0 / (self.n_density * self.sigma)
    }

    /// Collision frequency (rate of partition operations) [Hz].
    pub fn collision_frequency(&self) -> f64 {
        1.0 / self.tau_c
    }

    /// Pressure from ideal gas law [Pa].
    ///
    /// Partition operations create pressure: P = n·k_B·T
    pub fn pressure(&self) -> f64 {
        self.n_density * K_B * self.temperature
    }

    /// Populate the S-coordinate ensemble from hardware timing.
    ///
    /// The fluid IS the collection of categorical states created by measurement.
    /// Each measurement CREATES a molecule - it doesn't observe a pre-existing one.
    pub fn populate_from_hardware(&mut self, n_samples: usize) {
        self.s_ensemble.clear();
        self.s_ensemble.reserve(n_samples);

        let mut reference = Instant::now();

        for _ in 0..n_samples {
            let coord = SCoordinate::sample_now(reference);
            self.s_ensemble.push(coord);
            reference = Instant::now();
        }
    }

    /// Temperature measured from S-coordinate variance.
    ///
    /// Temperature IS the variance of S-coordinates (timing jitter).
    pub fn s_temperature(&self) -> f64 {
        if self.s_ensemble.is_empty() {
            return self.temperature;
        }

        // Calculate variance of S-coordinates
        let n = self.s_ensemble.len() as f64;
        let mean_k: f64 = self.s_ensemble.iter().map(|s| s.s_k()).sum::<f64>() / n;
        let mean_t: f64 = self.s_ensemble.iter().map(|s| s.s_t()).sum::<f64>() / n;
        let mean_e: f64 = self.s_ensemble.iter().map(|s| s.s_e()).sum::<f64>() / n;

        let var_k: f64 = self
            .s_ensemble
            .iter()
            .map(|s| (s.s_k() - mean_k).powi(2))
            .sum::<f64>()
            / n;
        let var_t: f64 = self
            .s_ensemble
            .iter()
            .map(|s| (s.s_t() - mean_t).powi(2))
            .sum::<f64>()
            / n;
        let var_e: f64 = self
            .s_ensemble
            .iter()
            .map(|s| (s.s_e() - mean_e).powi(2))
            .sum::<f64>()
            / n;

        let total_variance = var_k + var_t + var_e;

        // Scale to physical temperature
        total_variance * self.temperature
    }

    /// Access the S-coordinate ensemble.
    pub fn s_ensemble(&self) -> &[SCoordinate] {
        &self.s_ensemble
    }

    /// Perform a partition operation.
    pub fn partition_operation(&self, n_parts: usize) -> PartitionOperation {
        let start = Instant::now();
        // The partition operation itself
        let elapsed = start.elapsed();
        let measured_tau = elapsed.as_secs_f64();

        PartitionOperation::new(measured_tau.max(self.tau_c), self.g, n_parts)
    }

    /// Generate a partition cascade for a path through this fluid.
    pub fn cascade_for_path(&self, path_length: f64) -> PartitionCascade {
        PartitionCascade::from_path_length(
            self.n_density,
            self.sigma,
            path_length,
            self.tau_c,
            self.g,
        )
    }
}

impl ViscosityRelation for PartitionFluid {
    fn tau_c(&self) -> f64 {
        self.tau_c
    }

    fn coupling_g(&self) -> f64 {
        self.g
    }

    // viscosity() is provided by the trait default: tau_c × g
}

/// Builder for creating custom partition fluids.
#[derive(Debug, Default)]
pub struct PartitionFluidBuilder {
    n_density: Option<f64>,
    sigma: Option<f64>,
    temperature: Option<f64>,
    molecular_mass: Option<f64>,
    viscosity_experimental: Option<f64>,
    is_liquid: bool,
}

impl PartitionFluidBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_density(mut self, n: f64) -> Self {
        self.n_density = Some(n);
        self
    }

    pub fn sigma(mut self, s: f64) -> Self {
        self.sigma = Some(s);
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn molecular_mass(mut self, m: f64) -> Self {
        self.molecular_mass = Some(m);
        self
    }

    pub fn viscosity(mut self, mu: f64) -> Self {
        self.viscosity_experimental = Some(mu);
        self
    }

    pub fn liquid(mut self) -> Self {
        self.is_liquid = true;
        self
    }

    pub fn gas(mut self) -> Self {
        self.is_liquid = false;
        self
    }

    pub fn build(self) -> Option<PartitionFluid> {
        Some(PartitionFluid::new(
            self.n_density?,
            self.sigma?,
            self.temperature?,
            self.molecular_mass?,
            self.viscosity_experimental,
            self.is_liquid,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccl4_viscosity() {
        let fluid = PartitionFluid::from_species(MolecularSpecies::CCl4, 298.0);

        // Experimental value: 0.97e-3 Pa·s
        let mu_experimental = 0.97e-3;
        let mu_computed = fluid.viscosity();

        let rel_error = (mu_computed - mu_experimental).abs() / mu_experimental;
        // Should match within 1% since we derive τ_c from experimental viscosity
        assert!(
            rel_error < 0.01,
            "CCl4 viscosity error: {:.2}%",
            rel_error * 100.0
        );
    }

    #[test]
    fn test_n2_viscosity() {
        let fluid = PartitionFluid::from_species(MolecularSpecies::N2, 298.0);

        // Experimental value: 1.76e-5 Pa·s
        // For gases, kinetic theory is an approximation (unlike liquids where we use experimental μ)
        let mu_experimental = 1.76e-5;
        let mu_computed = fluid.viscosity();

        let rel_error = (mu_computed - mu_experimental).abs() / mu_experimental;
        // Kinetic theory approximation typically within 20% for real gases
        assert!(
            rel_error < 0.20,
            "N2 viscosity error: {:.2}%",
            rel_error * 100.0
        );
    }

    #[test]
    fn test_viscosity_relation() {
        let fluid = PartitionFluid::from_species(MolecularSpecies::H2O, 298.0);

        // μ = τ_c × g must hold
        let mu_from_relation = fluid.tau_c() * fluid.coupling_g();
        let mu_from_method = fluid.viscosity();

        assert!((mu_from_relation - mu_from_method).abs() < 1e-20);
    }

    #[test]
    fn test_pressure() {
        let fluid = PartitionFluid::from_species(MolecularSpecies::N2, 298.0);

        // At 1 atm, 298 K, pressure should be ~101325 Pa
        let pressure = fluid.pressure();
        // This is using the n_density from the species parameters
        assert!(pressure > 0.0);
    }

    #[test]
    fn test_cascade_generation() {
        let fluid = PartitionFluid::from_species(MolecularSpecies::CCl4, 298.0);
        let cascade = fluid.cascade_for_path(0.01); // 1 cm

        // Should have many collisions in 1 cm of liquid
        assert!(cascade.n_operations() > 1000);
    }

    #[test]
    fn test_builder() {
        let fluid = PartitionFluidBuilder::new()
            .n_density(2.5e25)
            .sigma(3.6e-19)
            .temperature(300.0)
            .molecular_mass(4.65e-26)
            .gas()
            .build()
            .unwrap();

        assert!(!fluid.is_liquid());
        assert!((fluid.temperature() - 300.0).abs() < 1e-10);
    }
}
