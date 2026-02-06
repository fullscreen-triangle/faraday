//! Error types for the Faraday framework.
//!
//! Provides structured error handling using thiserror for all framework operations.

use thiserror::Error;

/// Core errors for the Faraday framework.
#[derive(Error, Debug)]
pub enum FaradayError {
    /// Invalid S-coordinate value (must be in [0, 1])
    #[error("S-coordinate {name} = {value} out of bounds [0, 1]")]
    InvalidSCoordinate { name: &'static str, value: f64 },

    /// Invalid quantum number
    #[error("Invalid quantum number: {message}")]
    InvalidQuantumNumber { message: String },

    /// Invalid partition operation
    #[error("Invalid partition operation: {message}")]
    InvalidPartition { message: String },

    /// Invalid fluid parameters
    #[error("Invalid fluid parameter: {message}")]
    InvalidFluidParameter { message: String },

    /// Numerical computation error
    #[error("Numerical error: {message}")]
    NumericalError { message: String },

    /// Validation failure
    #[error("Validation failed: {message}")]
    ValidationError { message: String },

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type alias for Faraday operations.
pub type FaradayResult<T> = Result<T, FaradayError>;

impl FaradayError {
    /// Create an invalid S-coordinate error.
    pub fn invalid_s_coordinate(name: &'static str, value: f64) -> Self {
        Self::InvalidSCoordinate { name, value }
    }

    /// Create an invalid quantum number error.
    pub fn invalid_quantum_number(message: impl Into<String>) -> Self {
        Self::InvalidQuantumNumber {
            message: message.into(),
        }
    }

    /// Create an invalid partition error.
    pub fn invalid_partition(message: impl Into<String>) -> Self {
        Self::InvalidPartition {
            message: message.into(),
        }
    }

    /// Create an invalid fluid parameter error.
    pub fn invalid_fluid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidFluidParameter {
            message: message.into(),
        }
    }

    /// Create a numerical error.
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Create a validation error.
    pub fn validation_error(message: impl Into<String>) -> Self {
        Self::ValidationError {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FaradayError::invalid_s_coordinate("S_k", 1.5);
        assert!(err.to_string().contains("S_k"));
        assert!(err.to_string().contains("1.5"));
    }

    #[test]
    fn test_quantum_number_error() {
        let err = FaradayError::invalid_quantum_number("l must be < n");
        assert!(err.to_string().contains("l must be < n"));
    }
}
