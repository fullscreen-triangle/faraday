//! Selection rules for quantum transitions.
//!
//! The selection rules Δl = ±1, Δm ∈ {0, ±1}, Δs = 0 emerge from
//! the categorical structure of angular momentum.
//!
//! These rules determine which transitions are "allowed" (electric dipole)
//! versus "forbidden" (require higher-order processes).

use faraday_core::quantum::QuantumState;

/// Selection rules for electric dipole transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionRules {
    /// Allowed change in l: ±1
    pub delta_l: i32,
    /// Allowed change in m: 0, ±1
    pub delta_m: i32,
    /// Allowed change in s: 0 (spin conserved)
    pub delta_s_zero: bool,
}

impl SelectionRules {
    /// Standard electric dipole selection rules.
    pub const ELECTRIC_DIPOLE: Self = Self {
        delta_l: 1,   // |Δl| = 1
        delta_m: 1,   // |Δm| ≤ 1
        delta_s_zero: true,
    };
}

/// Check if a transition between two quantum states is allowed.
pub fn is_transition_allowed(initial: &QuantumState, final_state: &QuantumState) -> bool {
    let delta_l = (final_state.l() as i32) - (initial.l() as i32);
    let delta_m = final_state.m() - initial.m();
    let delta_s = final_state.s() - initial.s();

    // Selection rules
    let l_allowed = delta_l.abs() == 1;
    let m_allowed = delta_m.abs() <= 1;
    let s_allowed = delta_s.abs() < 1e-10; // Spin conserved

    l_allowed && m_allowed && s_allowed
}

/// Detailed transition analysis.
pub fn analyze_transition(
    initial: &QuantumState,
    final_state: &QuantumState,
) -> TransitionAnalysis {
    let delta_n = (final_state.n() as i32) - (initial.n() as i32);
    let delta_l = (final_state.l() as i32) - (initial.l() as i32);
    let delta_m = final_state.m() - initial.m();
    let delta_s = final_state.s() - initial.s();

    let l_allowed = delta_l.abs() == 1;
    let m_allowed = delta_m.abs() <= 1;
    let s_allowed = delta_s.abs() < 1e-10;

    let is_allowed = l_allowed && m_allowed && s_allowed;

    let violation = if !is_allowed {
        Some(determine_violation(l_allowed, m_allowed, s_allowed))
    } else {
        None
    };

    TransitionAnalysis {
        delta_n,
        delta_l,
        delta_m,
        delta_s,
        l_allowed,
        m_allowed,
        s_allowed,
        is_allowed,
        violation,
    }
}

/// Determine which selection rule is violated.
fn determine_violation(l_allowed: bool, m_allowed: bool, s_allowed: bool) -> SelectionViolation {
    if !l_allowed {
        SelectionViolation::DeltaL
    } else if !m_allowed {
        SelectionViolation::DeltaM
    } else if !s_allowed {
        SelectionViolation::DeltaS
    } else {
        SelectionViolation::None
    }
}

/// Detailed transition analysis result.
#[derive(Debug, Clone)]
pub struct TransitionAnalysis {
    /// Change in principal quantum number
    pub delta_n: i32,
    /// Change in angular momentum quantum number
    pub delta_l: i32,
    /// Change in magnetic quantum number
    pub delta_m: i32,
    /// Change in spin quantum number
    pub delta_s: f64,
    /// Whether Δl = ±1 is satisfied
    pub l_allowed: bool,
    /// Whether |Δm| ≤ 1 is satisfied
    pub m_allowed: bool,
    /// Whether Δs = 0 is satisfied
    pub s_allowed: bool,
    /// Overall allowed status
    pub is_allowed: bool,
    /// Which rule is violated (if any)
    pub violation: Option<SelectionViolation>,
}

/// Type of selection rule violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionViolation {
    /// No violation
    None,
    /// Δl ≠ ±1
    DeltaL,
    /// |Δm| > 1
    DeltaM,
    /// Δs ≠ 0
    DeltaS,
}

/// Generate all allowed transitions from a given state.
pub fn allowed_transitions_from(state: &QuantumState, n_max: u32) -> Vec<QuantumState> {
    let mut allowed = Vec::new();

    for n in 1..=n_max {
        for l in 0..n {
            // Check Δl = ±1
            let delta_l = (l as i32) - (state.l() as i32);
            if delta_l.abs() != 1 {
                continue;
            }

            for m in -(l as i32)..=(l as i32) {
                // Check |Δm| ≤ 1
                let delta_m = m - state.m();
                if delta_m.abs() > 1 {
                    continue;
                }

                // Spin is conserved
                if let Ok(final_state) = QuantumState::new(n, l, m, state.s()) {
                    allowed.push(final_state);
                }
            }
        }
    }

    allowed
}

/// Count allowed vs forbidden transitions in a shell.
pub fn transition_statistics(n_initial: u32, n_final: u32) -> TransitionStatistics {
    use faraday_core::quantum::shell_states;

    let initial_states = shell_states(n_initial);
    let final_states = shell_states(n_final);

    let total = initial_states.len() * final_states.len();
    let mut allowed_count = 0;
    let mut forbidden_count = 0;

    for initial in &initial_states {
        for final_state in &final_states {
            if is_transition_allowed(initial, final_state) {
                allowed_count += 1;
            } else {
                forbidden_count += 1;
            }
        }
    }

    TransitionStatistics {
        n_initial,
        n_final,
        total_transitions: total,
        allowed: allowed_count,
        forbidden: forbidden_count,
        allowed_fraction: allowed_count as f64 / total as f64,
    }
}

/// Statistics about transitions between shells.
#[derive(Debug, Clone)]
pub struct TransitionStatistics {
    /// Initial shell
    pub n_initial: u32,
    /// Final shell
    pub n_final: u32,
    /// Total possible transitions
    pub total_transitions: usize,
    /// Number of allowed transitions
    pub allowed: usize,
    /// Number of forbidden transitions
    pub forbidden: usize,
    /// Fraction that are allowed
    pub allowed_fraction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1s_to_2p_allowed() {
        // 1s → 2p is the classic allowed transition
        let initial = QuantumState::GROUND_1S_UP;
        let final_state = QuantumState::EXCITED_2P_UP;

        assert!(is_transition_allowed(&initial, &final_state));
    }

    #[test]
    fn test_1s_to_2s_forbidden() {
        // 1s → 2s is forbidden (Δl = 0)
        let initial = QuantumState::GROUND_1S_UP;
        let final_state = QuantumState::EXCITED_2S_UP;

        assert!(!is_transition_allowed(&initial, &final_state));
    }

    #[test]
    fn test_spin_flip_forbidden() {
        // Spin flip is forbidden for electric dipole
        let initial = QuantumState::GROUND_1S_UP;
        let final_state = QuantumState::new(2, 1, 0, -0.5).unwrap();

        assert!(!is_transition_allowed(&initial, &final_state));
    }

    #[test]
    fn test_transition_analysis() {
        let initial = QuantumState::GROUND_1S_UP;
        let final_state = QuantumState::EXCITED_2P_UP;

        let analysis = analyze_transition(&initial, &final_state);

        assert_eq!(analysis.delta_n, 1);
        assert_eq!(analysis.delta_l, 1);
        assert_eq!(analysis.delta_m, 0);
        assert!(analysis.is_allowed);
        assert!(analysis.violation.is_none());
    }

    #[test]
    fn test_forbidden_analysis() {
        let initial = QuantumState::GROUND_1S_UP;
        let final_state = QuantumState::EXCITED_2S_UP;

        let analysis = analyze_transition(&initial, &final_state);

        assert!(!analysis.is_allowed);
        assert_eq!(analysis.violation, Some(SelectionViolation::DeltaL));
    }

    #[test]
    fn test_allowed_transitions_from_1s() {
        let ground = QuantumState::GROUND_1S_UP;
        let allowed = allowed_transitions_from(&ground, 4);

        // From 1s, only p orbitals are allowed (l=0 → l=1)
        for state in &allowed {
            assert_eq!(state.l(), 1, "Non-p state found: {:?}", state);
        }

        // Should have transitions to 2p, 3p, 4p
        assert!(allowed.len() > 0);
    }

    #[test]
    fn test_transition_statistics() {
        let stats = transition_statistics(1, 2);

        // Shell 1 has 2 states, shell 2 has 8 states
        assert_eq!(stats.total_transitions, 2 * 8);

        // Should have some allowed and some forbidden
        assert!(stats.allowed > 0);
        assert!(stats.forbidden > 0);
    }
}
