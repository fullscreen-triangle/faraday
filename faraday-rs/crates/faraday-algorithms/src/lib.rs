//! Faraday Algorithms - Algorithms for the partition physics framework.
//!
//! This crate provides core algorithms implementing partition physics concepts:
//!
//! - **Ternary search**: O(log₃N) complexity, 37% faster than binary
//! - **Partition capacity**: C(n) = 2n² from bounded phase space
//! - **Selection rules**: Δl=±1, Δm∈{0,±1}, Δs=0 for allowed transitions
//! - **Commutation relations**: [Ô_cat, Ô_phys] = 0 verification
//!
//! # Example
//!
//! ```rust
//! use faraday_algorithms::{ternary_search, shell_capacity, is_transition_allowed};
//! use faraday_core::quantum::QuantumState;
//!
//! // Ternary search with 37% fewer iterations
//! let data: Vec<i32> = (0..10000).collect();
//! let result = ternary_search(&data, &5000);
//! assert_eq!(result.index, Some(5000));
//!
//! // Shell capacity C(n) = 2n²
//! assert_eq!(shell_capacity(3), 18);
//!
//! // Selection rules: 1s → 2p allowed, 1s → 2s forbidden
//! let ground = QuantumState::GROUND_1S_UP;
//! let excited_p = QuantumState::EXCITED_2P_UP;
//! let excited_s = QuantumState::EXCITED_2S_UP;
//! assert!(is_transition_allowed(&ground, &excited_p));
//! assert!(!is_transition_allowed(&ground, &excited_s));
//! ```

pub mod capacity;
pub mod commutation;
pub mod selection;
pub mod ternary;

// Re-export commonly used functions and types
pub use capacity::{
    cumulative_capacity, electron_configuration, shell_breakdown, shell_capacity,
    subshell_capacity, verify_capacity, verify_capacity_range,
    CapacityVerification, ElectronConfiguration, SubshellInfo,
};

pub use commutation::{
    operators_commute, verify_commutation_theorem, verify_s_coordinate_commutation,
    categorical_normalize, physical_scale,
    CategoricalOperator, CommutationResult, CommutationTheorem, PhysicalOperator,
    SCoordinateCommutation,
};

pub use selection::{
    allowed_transitions_from, analyze_transition, is_transition_allowed, transition_statistics,
    SelectionRules, SelectionViolation, TransitionAnalysis, TransitionStatistics,
};

pub use ternary::{
    binary_iterations_theoretical, binary_search_iterations, localization_comparison,
    spatial_localization, ternary_iterations_theoretical, ternary_search, theoretical_speedup,
    LocalizationComparison, TernarySearchResult, ASYMPTOTIC_SPEEDUP,
};
