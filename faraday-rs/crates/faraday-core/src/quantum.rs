//! Quantum state representation for the partition framework.
//!
//! Quantum numbers (n, l, m, s) emerge from bounded phase space geometry
//! as the natural coordinates for electron states in atoms.

use serde::{Deserialize, Serialize};

use crate::constants::{A_0, E_RYDBERG, HBAR};
use crate::error::{FaradayError, FaradayResult};

/// Quantum state defined by quantum numbers (n, l, m, s).
///
/// These quantum numbers emerge from the bounded phase space geometry:
/// - n (principal): determines energy shell, n ≥ 1
/// - l (angular momentum): 0 ≤ l < n
/// - m (magnetic): -l ≤ m ≤ l
/// - s (spin): ±1/2
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantumState {
    /// Principal quantum number (n ≥ 1)
    n: u32,
    /// Angular momentum quantum number (0 ≤ l < n)
    l: u32,
    /// Magnetic quantum number (-l ≤ m ≤ l)
    m: i32,
    /// Spin quantum number (±0.5)
    s: f64,
}

impl QuantumState {
    /// Create a new quantum state with validation.
    pub fn new(n: u32, l: u32, m: i32, s: f64) -> FaradayResult<Self> {
        // Validate quantum numbers
        if n < 1 {
            return Err(FaradayError::invalid_quantum_number(
                "Principal quantum number n must be ≥ 1",
            ));
        }
        if l >= n {
            return Err(FaradayError::invalid_quantum_number(format!(
                "Angular momentum l={} must be < n={}",
                l, n
            )));
        }
        if m.abs() as u32 > l {
            return Err(FaradayError::invalid_quantum_number(format!(
                "Magnetic quantum number |m|={} must be ≤ l={}",
                m.abs(),
                l
            )));
        }
        if (s - 0.5).abs() > 1e-10 && (s + 0.5).abs() > 1e-10 {
            return Err(FaradayError::invalid_quantum_number(format!(
                "Spin s={} must be ±0.5",
                s
            )));
        }

        Ok(Self { n, l, m, s })
    }

    /// Create quantum state without validation (use with caution).
    pub const fn new_unchecked(n: u32, l: u32, m: i32, s: f64) -> Self {
        Self { n, l, m, s }
    }

    /// Principal quantum number.
    #[inline]
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Angular momentum quantum number.
    #[inline]
    pub fn l(&self) -> u32 {
        self.l
    }

    /// Magnetic quantum number.
    #[inline]
    pub fn m(&self) -> i32 {
        self.m
    }

    /// Spin quantum number.
    #[inline]
    pub fn s(&self) -> f64 {
        self.s
    }

    /// Energy of the state in hydrogen-like atom.
    ///
    /// E_n = -E_R / n² where E_R is the Rydberg energy.
    pub fn energy(&self) -> f64 {
        -E_RYDBERG / (self.n as f64).powi(2)
    }

    /// Energy of the state for a given nuclear charge Z.
    pub fn energy_z(&self, z: u32) -> f64 {
        -E_RYDBERG * (z as f64).powi(2) / (self.n as f64).powi(2)
    }

    /// Orbital angular momentum magnitude |L| = ℏ√(l(l+1)).
    pub fn angular_momentum(&self) -> f64 {
        HBAR * ((self.l as f64) * (self.l as f64 + 1.0)).sqrt()
    }

    /// Z-component of angular momentum L_z = mℏ.
    pub fn angular_momentum_z(&self) -> f64 {
        HBAR * self.m as f64
    }

    /// Mean radius of the orbital ⟨r⟩ for hydrogen.
    ///
    /// ⟨r⟩ = a₀ × n² × [3/2 - l(l+1)/(2n²)]
    pub fn mean_radius(&self) -> f64 {
        let n2 = (self.n as f64).powi(2);
        let ll1 = (self.l as f64) * (self.l as f64 + 1.0);
        A_0 * n2 * (1.5 - ll1 / (2.0 * n2))
    }

    /// Shell capacity C(n) = 2n² (from partition capacity theorem).
    pub fn shell_capacity(&self) -> u32 {
        2 * self.n * self.n
    }

    /// Number of states in this shell with same n.
    pub fn degeneracy(&self) -> u32 {
        self.shell_capacity()
    }

    /// Spectroscopic notation (e.g., "1s", "2p", "3d").
    pub fn spectroscopic_notation(&self) -> String {
        let l_letter = match self.l {
            0 => 's',
            1 => 'p',
            2 => 'd',
            3 => 'f',
            4 => 'g',
            5 => 'h',
            _ => '?',
        };
        format!("{}{}", self.n, l_letter)
    }

    /// Check if a transition to another state is allowed by selection rules.
    ///
    /// Selection rules: Δl = ±1, Δm ∈ {0, ±1}, Δs = 0
    pub fn can_transition_to(&self, final_state: &QuantumState) -> bool {
        let delta_l = (final_state.l as i32) - (self.l as i32);
        let delta_m = final_state.m - self.m;
        let delta_s = final_state.s - self.s;

        // Selection rules
        let l_allowed = delta_l.abs() == 1;
        let m_allowed = delta_m.abs() <= 1;
        let s_allowed = delta_s.abs() < 1e-10;

        l_allowed && m_allowed && s_allowed
    }

    /// Photon energy for transition to another state.
    pub fn transition_energy(&self, final_state: &QuantumState) -> f64 {
        (final_state.energy() - self.energy()).abs()
    }

    /// Common ground state: 1s (n=1, l=0, m=0, s=+1/2).
    pub const GROUND_1S_UP: QuantumState = QuantumState::new_unchecked(1, 0, 0, 0.5);

    /// Common ground state: 1s (n=1, l=0, m=0, s=-1/2).
    pub const GROUND_1S_DOWN: QuantumState = QuantumState::new_unchecked(1, 0, 0, -0.5);

    /// Common excited state: 2s.
    pub const EXCITED_2S_UP: QuantumState = QuantumState::new_unchecked(2, 0, 0, 0.5);

    /// Common excited state: 2p (m=0).
    pub const EXCITED_2P_UP: QuantumState = QuantumState::new_unchecked(2, 1, 0, 0.5);
}

impl std::fmt::Display for QuantumState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let spin_str = if self.s > 0.0 { "↑" } else { "↓" };
        write!(
            f,
            "|n={}, l={}, m={}, s={}⟩ = {}{}",
            self.n,
            self.l,
            self.m,
            spin_str,
            self.spectroscopic_notation(),
            spin_str
        )
    }
}

/// Generate all quantum states for a given principal quantum number n.
pub fn shell_states(n: u32) -> Vec<QuantumState> {
    let mut states = Vec::with_capacity((2 * n * n) as usize);

    for l in 0..n {
        for m in -(l as i32)..=(l as i32) {
            for &s in &[0.5, -0.5] {
                // These are guaranteed valid by construction
                states.push(QuantumState::new_unchecked(n, l, m, s));
            }
        }
    }

    states
}

/// Calculate shell capacity C(n) = 2n².
#[inline]
pub fn shell_capacity(n: u32) -> u32 {
    2 * n * n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_state() {
        let state = QuantumState::new(2, 1, 0, 0.5).unwrap();
        assert_eq!(state.n(), 2);
        assert_eq!(state.l(), 1);
        assert_eq!(state.m(), 0);
        assert!((state.s() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_l() {
        assert!(QuantumState::new(2, 2, 0, 0.5).is_err());
    }

    #[test]
    fn test_invalid_m() {
        assert!(QuantumState::new(2, 1, 2, 0.5).is_err());
    }

    #[test]
    fn test_invalid_spin() {
        assert!(QuantumState::new(1, 0, 0, 0.3).is_err());
    }

    #[test]
    fn test_shell_capacity() {
        assert_eq!(shell_capacity(1), 2);
        assert_eq!(shell_capacity(2), 8);
        assert_eq!(shell_capacity(3), 18);
        assert_eq!(shell_capacity(4), 32);
    }

    #[test]
    fn test_shell_states_count() {
        for n in 1..=5 {
            let states = shell_states(n);
            assert_eq!(states.len() as u32, shell_capacity(n));
        }
    }

    #[test]
    fn test_selection_rules() {
        let s1 = QuantumState::GROUND_1S_UP;
        let s2 = QuantumState::EXCITED_2P_UP;

        // 1s → 2p is allowed (Δl = +1)
        assert!(s1.can_transition_to(&s2));

        // 1s → 2s is forbidden (Δl = 0)
        assert!(!s1.can_transition_to(&QuantumState::EXCITED_2S_UP));
    }

    #[test]
    fn test_spectroscopic_notation() {
        assert_eq!(QuantumState::GROUND_1S_UP.spectroscopic_notation(), "1s");
        assert_eq!(QuantumState::EXCITED_2P_UP.spectroscopic_notation(), "2p");
    }

    #[test]
    fn test_energy() {
        let ground = QuantumState::GROUND_1S_UP;
        // Ground state energy should be -13.6 eV = -E_R
        let energy_ev = ground.energy() / 1.602176634e-19;
        assert!((energy_ev + 13.6).abs() < 0.1);
    }
}
