//! Shell capacity calculation: C(n) = 2n²
//!
//! The partition capacity theorem states that the number of quantum states
//! in shell n is exactly C(n) = 2n².
//!
//! This emerges from the bounded phase space geometry:
//! - n values of l: 0, 1, ..., n-1
//! - (2l+1) values of m for each l
//! - 2 values of spin s
//!
//! Sum: Σ(2l+1) × 2 = 2 × n² = 2n²

use faraday_core::quantum::shell_states;

/// Calculate shell capacity C(n) = 2n².
///
/// This is the number of quantum states in shell n.
#[inline]
pub fn shell_capacity(n: u32) -> u32 {
    2 * n * n
}

/// Calculate cumulative capacity up to and including shell n.
///
/// Total states from shell 1 to n: Σ C(k) = 2 × n(n+1)(2n+1)/6
pub fn cumulative_capacity(n: u32) -> u32 {
    // Σ_{k=1}^n 2k² = 2 × n(n+1)(2n+1)/6 = n(n+1)(2n+1)/3
    let n = n as u64;
    let result = n * (n + 1) * (2 * n + 1) / 3;
    result as u32
}

/// Verify that counting quantum states matches C(n) = 2n².
pub fn verify_capacity(n: u32) -> CapacityVerification {
    let theoretical = shell_capacity(n);
    let states = shell_states(n);
    let actual = states.len() as u32;

    CapacityVerification {
        n,
        theoretical,
        actual,
        matches: theoretical == actual,
    }
}

/// Result of capacity verification.
#[derive(Debug, Clone, Copy)]
pub struct CapacityVerification {
    /// Principal quantum number
    pub n: u32,
    /// Theoretical capacity C(n) = 2n²
    pub theoretical: u32,
    /// Actual count of quantum states
    pub actual: u32,
    /// Whether they match
    pub matches: bool,
}

/// Verify capacity formula for a range of shells.
pub fn verify_capacity_range(n_max: u32) -> Vec<CapacityVerification> {
    (1..=n_max).map(verify_capacity).collect()
}

/// Calculate the subshell capacity for given l.
///
/// Each subshell l contains (2l + 1) × 2 states (accounting for spin).
#[inline]
pub fn subshell_capacity(l: u32) -> u32 {
    2 * (2 * l + 1)
}

/// Get the subshell breakdown for shell n.
pub fn shell_breakdown(n: u32) -> Vec<SubshellInfo> {
    (0..n)
        .map(|l| SubshellInfo {
            l,
            n_m_values: 2 * l + 1,
            n_spin_states: 2,
            total_states: subshell_capacity(l),
        })
        .collect()
}

/// Information about a subshell.
#[derive(Debug, Clone, Copy)]
pub struct SubshellInfo {
    /// Angular momentum quantum number
    pub l: u32,
    /// Number of m values (-l to +l)
    pub n_m_values: u32,
    /// Number of spin states (always 2)
    pub n_spin_states: u32,
    /// Total states in subshell
    pub total_states: u32,
}

/// Calculate which shell a given total state count corresponds to.
///
/// Given cumulative count, find the shell n such that
/// cumulative_capacity(n) >= count.
pub fn shell_from_electron_count(electron_count: u32) -> u32 {
    let mut n = 1;
    while cumulative_capacity(n) < electron_count {
        n += 1;
    }
    n
}

/// Get electron configuration information.
pub fn electron_configuration(z: u32) -> ElectronConfiguration {
    let filled_shells = shell_from_electron_count(z);
    let mut remaining = z;
    let mut shells = Vec::new();

    for n in 1..=filled_shells {
        let capacity = shell_capacity(n);
        let electrons_in_shell = if remaining >= capacity {
            remaining -= capacity;
            capacity
        } else {
            let e = remaining;
            remaining = 0;
            e
        };

        if electrons_in_shell > 0 {
            shells.push((n, electrons_in_shell));
        }

        if remaining == 0 {
            break;
        }
    }

    ElectronConfiguration {
        z,
        shells,
        valence_shell: filled_shells,
    }
}

/// Electron configuration result.
#[derive(Debug, Clone)]
pub struct ElectronConfiguration {
    /// Atomic number
    pub z: u32,
    /// Electrons in each shell (n, count)
    pub shells: Vec<(u32, u32)>,
    /// Valence shell number
    pub valence_shell: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_capacity() {
        assert_eq!(shell_capacity(1), 2);  // 1s²
        assert_eq!(shell_capacity(2), 8);  // 2s² 2p⁶
        assert_eq!(shell_capacity(3), 18); // 3s² 3p⁶ 3d¹⁰
        assert_eq!(shell_capacity(4), 32); // 4s² 4p⁶ 4d¹⁰ 4f¹⁴
        assert_eq!(shell_capacity(5), 50);
    }

    #[test]
    fn test_verify_capacity() {
        for n in 1..=7 {
            let result = verify_capacity(n);
            assert!(
                result.matches,
                "Shell {} capacity mismatch: theoretical={}, actual={}",
                n,
                result.theoretical,
                result.actual
            );
        }
    }

    #[test]
    fn test_cumulative_capacity() {
        // Cumulative: 2, 10, 28, 60, 110, ...
        assert_eq!(cumulative_capacity(1), 2);
        assert_eq!(cumulative_capacity(2), 2 + 8);
        assert_eq!(cumulative_capacity(3), 2 + 8 + 18);
        assert_eq!(cumulative_capacity(4), 2 + 8 + 18 + 32);
    }

    #[test]
    fn test_subshell_capacity() {
        assert_eq!(subshell_capacity(0), 2);  // s
        assert_eq!(subshell_capacity(1), 6);  // p
        assert_eq!(subshell_capacity(2), 10); // d
        assert_eq!(subshell_capacity(3), 14); // f
    }

    #[test]
    fn test_shell_breakdown() {
        let breakdown = shell_breakdown(3);
        assert_eq!(breakdown.len(), 3);
        assert_eq!(breakdown[0].l, 0); // s
        assert_eq!(breakdown[1].l, 1); // p
        assert_eq!(breakdown[2].l, 2); // d

        let total: u32 = breakdown.iter().map(|s| s.total_states).sum();
        assert_eq!(total, shell_capacity(3));
    }

    #[test]
    fn test_shell_from_electron_count() {
        assert_eq!(shell_from_electron_count(1), 1);
        assert_eq!(shell_from_electron_count(2), 1);
        assert_eq!(shell_from_electron_count(3), 2);
        assert_eq!(shell_from_electron_count(10), 2);
        assert_eq!(shell_from_electron_count(11), 3);
    }

    #[test]
    fn test_electron_configuration() {
        // Helium (Z=2): 1s²
        let he = electron_configuration(2);
        assert_eq!(he.z, 2);
        assert_eq!(he.shells, vec![(1, 2)]);

        // Neon (Z=10): 1s² 2s² 2p⁶
        let ne = electron_configuration(10);
        assert_eq!(ne.z, 10);
        assert_eq!(ne.shells, vec![(1, 2), (2, 8)]);
    }
}
