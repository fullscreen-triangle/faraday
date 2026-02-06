//! Faraday Core - Core data structures for the partition physics framework.
//!
//! This crate provides the fundamental types and traits for the Faraday framework:
//!
//! - **Constants**: Physical constants (ℏ, k_B, c, etc.)
//! - **Coordinates**: S-coordinate system for categorical entropy space
//! - **Quantum**: Quantum states and selection rules
//! - **Partition**: Partition operations and cascades
//! - **Fluid**: Partition fluid structures (viscosity μ = τ_c × g)
//!
//! # Key Concepts
//!
//! ## Bounded Phase Space
//! The framework is built on the axiom that phase space is bounded.
//! The S-coordinate (S_k, S_t, S_e) ∈ [0,1]³ represents position in
//! categorical entropy space.
//!
//! ## Viscosity Relation
//! The fundamental viscosity relation μ = τ_c × g connects:
//! - τ_c: partition lag (time in undetermined state)
//! - g: coupling strength (momentum transfer per operation)
//!
//! ## Partition Operations
//! Physical interactions are partition operations that create
//! categorical distinctions. Each collision generates entropy
//! S = k_B × ln(n) where n is the number of partitions.
//!
//! # Example
//!
//! ```rust
//! use faraday_core::{
//!     fluid::{MolecularSpecies, PartitionFluid},
//!     partition::ViscosityRelation,
//! };
//!
//! // Create CCl4 at room temperature
//! let fluid = PartitionFluid::from_species(MolecularSpecies::CCl4, 298.0);
//!
//! // Viscosity emerges from partition structure
//! let viscosity = fluid.viscosity(); // μ = τ_c × g
//!
//! // Generate cascade for 1 cm path
//! let cascade = fluid.cascade_for_path(0.01);
//! println!("Collisions: {}", cascade.n_operations());
//! ```

pub mod constants;
pub mod coordinate;
pub mod error;
pub mod fluid;
pub mod partition;
pub mod quantum;

// Re-export commonly used types
pub use constants::*;
pub use coordinate::SCoordinate;
pub use error::{FaradayError, FaradayResult};
pub use fluid::{MolecularSpecies, PartitionFluid};
pub use partition::{PartitionCascade, PartitionOperation, ViscosityRelation};
pub use quantum::{shell_capacity, shell_states, QuantumState};
