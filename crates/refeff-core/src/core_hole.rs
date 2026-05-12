//! Core-hole labels, quantum numbers, and lifetime widths.
//!
//! This module ports small FEFF common routines used before the heavier
//! scattering solvers: `isedge`/`stdnm` for edge-name recognition and
//! normalization, `setkap` for initial-state angular momentum and relativistic
//! kappa, and `setgam` for Rahkonen-Krause core-hole lifetime widths.

use crate::Real;

/// Error returned by core-hole helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CoreHoleError {
    /// FEFF `setkap` only defines shells in the `1..=30` subset listed here.
    #[error("invalid FEFF hole number {ihole}")]
    InvalidHole { ihole: i32 },
    /// Core-hole width interpolation requires a positive atomic number.
    #[error("atomic number must be positive, got {z}")]
    InvalidAtomicNumber { z: i32 },
}

/// Initial-state angular momentum and relativistic kappa from FEFF `setkap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreHoleQuantumNumbers {
    /// Relativistic kappa quantum number.
    pub kappa: i32,
    /// Initial-state orbital angular momentum.
    pub angular_momentum: i32,
}

/// Return true when `label` is one of FEFF's accepted edge labels or numeric aliases.
#[must_use]
pub fn is_edge_label(label: &str) -> bool {
    edge_index(label).is_some()
}

/// Return FEFF's integer hole index for an edge label or numeric alias.
#[must_use]
pub fn edge_index(label: &str) -> Option<i32> {
    let normalized = label.trim().to_ascii_uppercase();
    EDGE_LABELS
        .iter()
        .position(|edge| *edge == normalized)
        .or_else(|| {
            normalized
                .parse::<usize>()
                .ok()
                .filter(|index| *index < EDGE_LABELS.len())
        })
        .and_then(|index| i32::try_from(index).ok())
}

/// Return FEFF's canonical edge label for an edge label or numeric alias.
///
/// This ports `COMMON/stdnm.f90`: valid numeric aliases such as `"1"` and
/// `"4"` are rewritten to `"K"` and `"L3"`, existing labels are normalized to
/// FEFF's uppercase spelling, and invalid inputs return `None`.
#[must_use]
pub fn standard_edge_label(label: &str) -> Option<&'static str> {
    edge_index(label)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| EDGE_LABELS.get(index).copied())
}

/// Return FEFF `setkap` quantum numbers for an integer hole index.
pub fn core_hole_quantum_numbers(ihole: i32) -> Result<CoreHoleQuantumNumbers, CoreHoleError> {
    let (angular_momentum, kappa) = match ihole {
        1 | 2 | 5 | 10 | 17 | 24 | 29 => (0, -1),
        3 | 6 | 11 | 18 | 25 | 30 => (1, 1),
        4 | 7 | 12 | 19 | 26 => (1, -2),
        8 | 13 | 20 | 27 => (2, 2),
        9 | 14 | 21 | 28 => (2, -3),
        15 | 22 => (3, 3),
        16 | 23 => (3, -4),
        _ => return Err(CoreHoleError::InvalidHole { ihole }),
    };

    Ok(CoreHoleQuantumNumbers {
        kappa,
        angular_momentum,
    })
}

/// Return FEFF `setgam` core-hole lifetime width in eV.
///
/// FEFF returns zero for `ihole <= 0` and uses a fixed `0.1 eV` fallback for
/// O-shell and higher holes (`ihole > 16`) because the Rahkonen-Krause table in
/// `setgam` only covers the first 16 hole indices.
pub fn core_hole_width_ev(z: i32, ihole: i32) -> Result<Real, CoreHoleError> {
    if ihole <= 0 {
        return Ok(0.0);
    }
    if ihole > 16 {
        return Ok(0.1);
    }
    if z <= 0 {
        return Err(CoreHoleError::InvalidAtomicNumber { z });
    }

    let hole_index =
        usize::try_from(ihole - 1).map_err(|_| CoreHoleError::InvalidHole { ihole })?;
    let grid_z = &Z_TABLE[hole_index];
    let grid_gamma = &GAMMA_TABLE[hole_index];
    let z = Real::from(z);
    let idx = grid_z
        .windows(2)
        .position(|window| z < window[1])
        .map_or(grid_z.len() - 2, |index| index);
    let x0 = grid_z[idx];
    let x1 = grid_z[idx + 1];
    let y0 = grid_gamma[idx].log10();
    let y1 = grid_gamma[idx + 1].log10();
    Ok(10.0_f64.powf(y0 + (z - x0) * (y1 - y0) / (x1 - x0)))
}

const EDGE_LABELS: [&str; 41] = [
    "NO", "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5", "N6",
    "N7", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "P1", "P2", "P3", "P4", "P5", "P6",
    "P7", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3",
];

const Z_TABLE: [[Real; 8]; 16] = [
    [0.99, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 95.1],
    [0.99, 18.0, 22.0, 35.0, 50.0, 52.0, 75.0, 95.1],
    [0.99, 17.0, 28.0, 31.0, 45.0, 60.0, 80.0, 95.1],
    [0.99, 17.0, 28.0, 31.0, 45.0, 60.0, 80.0, 95.1],
    [0.99, 20.0, 28.0, 30.0, 36.0, 53.0, 80.0, 95.1],
    [0.99, 20.0, 22.0, 30.0, 40.0, 68.0, 80.0, 95.1],
    [0.99, 20.0, 22.0, 30.0, 40.0, 68.0, 80.0, 95.1],
    [0.99, 36.0, 40.0, 48.0, 58.0, 76.0, 79.0, 95.1],
    [0.99, 36.0, 40.0, 48.0, 58.0, 76.0, 79.0, 95.1],
    [0.99, 30.0, 40.0, 47.0, 50.0, 63.0, 80.0, 95.1],
    [0.99, 40.0, 42.0, 49.0, 54.0, 70.0, 87.0, 95.1],
    [0.99, 40.0, 42.0, 49.0, 54.0, 70.0, 87.0, 95.1],
    [0.99, 40.0, 50.0, 55.0, 60.0, 70.0, 81.0, 95.1],
    [0.99, 40.0, 50.0, 55.0, 60.0, 70.0, 81.0, 95.1],
    [0.99, 71.0, 73.0, 79.0, 86.0, 90.0, 95.0, 100.0],
    [0.99, 71.0, 73.0, 79.0, 86.0, 90.0, 95.0, 100.0],
];

const GAMMA_TABLE: [[Real; 8]; 16] = [
    [0.02, 0.28, 0.75, 4.8, 10.5, 21.0, 60.0, 105.0],
    [0.07, 3.9, 3.8, 7.0, 6.0, 3.7, 8.0, 19.0],
    [0.001, 0.12, 1.4, 0.8, 2.6, 4.1, 6.3, 10.5],
    [0.001, 0.12, 0.55, 0.7, 2.1, 3.5, 5.4, 9.0],
    [0.001, 1.0, 2.9, 2.2, 5.5, 10.0, 22.0, 22.0],
    [0.001, 0.001, 0.5, 2.0, 2.6, 11.0, 15.0, 16.0],
    [0.001, 0.001, 0.5, 2.0, 2.6, 11.0, 10.0, 10.0],
    [0.0006, 0.09, 0.07, 0.48, 1.0, 4.0, 2.7, 4.7],
    [0.0006, 0.09, 0.07, 0.48, 0.87, 2.2, 2.5, 4.3],
    [0.001, 0.001, 6.2, 7.0, 3.2, 12.0, 16.0, 13.0],
    [0.001, 0.001, 1.9, 16.0, 2.7, 13.0, 13.0, 8.0],
    [0.001, 0.001, 1.9, 16.0, 2.7, 13.0, 13.0, 8.0],
    [0.001, 0.001, 0.15, 0.1, 0.8, 8.0, 8.0, 5.0],
    [0.001, 0.001, 0.15, 0.1, 0.8, 8.0, 8.0, 5.0],
    [0.001, 0.001, 0.05, 0.22, 0.1, 0.16, 0.5, 0.9],
    [0.001, 0.001, 0.05, 0.22, 0.1, 0.16, 0.5, 0.9],
];

#[cfg(test)]
mod tests {
    use super::{
        CoreHoleError, CoreHoleQuantumNumbers, core_hole_quantum_numbers, core_hole_width_ev,
        edge_index, is_edge_label, standard_edge_label,
    };

    #[test]
    fn recognizes_edge_labels_and_numeric_aliases() {
        assert!(is_edge_label("K"));
        assert!(is_edge_label("l3"));
        assert!(is_edge_label("4"));
        assert!(is_edge_label("S3"));
        assert_eq!(edge_index("NO"), Some(0));
        assert_eq!(edge_index("40"), Some(40));
        assert_eq!(edge_index("Q1"), None);
        assert_eq!(edge_index("41"), None);
    }

    #[test]
    fn standardizes_edge_labels_like_feff_stdnm() {
        assert_eq!(standard_edge_label("0"), Some("NO"));
        assert_eq!(standard_edge_label("1"), Some("K"));
        assert_eq!(standard_edge_label("4"), Some("L3"));
        assert_eq!(standard_edge_label("10"), Some("N1"));
        assert_eq!(standard_edge_label("s3"), Some("S3"));
        assert_eq!(standard_edge_label("Q1"), None);
        assert_eq!(standard_edge_label("41"), None);
    }

    #[test]
    fn returns_setkap_quantum_numbers() -> Result<(), CoreHoleError> {
        assert_eq!(
            core_hole_quantum_numbers(1)?,
            CoreHoleQuantumNumbers {
                kappa: -1,
                angular_momentum: 0,
            }
        );
        assert_eq!(
            core_hole_quantum_numbers(3)?,
            CoreHoleQuantumNumbers {
                kappa: 1,
                angular_momentum: 1,
            }
        );
        assert_eq!(
            core_hole_quantum_numbers(8)?,
            CoreHoleQuantumNumbers {
                kappa: 2,
                angular_momentum: 2,
            }
        );
        assert_eq!(
            core_hole_quantum_numbers(16)?,
            CoreHoleQuantumNumbers {
                kappa: -4,
                angular_momentum: 3,
            }
        );
        assert_eq!(
            core_hole_quantum_numbers(41),
            Err(CoreHoleError::InvalidHole { ihole: 41 })
        );
        Ok(())
    }

    #[test]
    fn interpolates_setgam_table_values() -> Result<(), CoreHoleError> {
        assert_close(core_hole_width_ev(20, 1)?, 0.75);
        assert_close(core_hole_width_ev(40, 1)?, 4.8);
        assert_close(core_hole_width_ev(28, 3)?, 1.4);
        Ok(())
    }

    #[test]
    fn uses_setgam_fallbacks() -> Result<(), CoreHoleError> {
        assert_close(core_hole_width_ev(29, 0)?, 0.0);
        assert_close(core_hole_width_ev(29, 17)?, 0.1);
        Ok(())
    }

    #[test]
    fn rejects_invalid_atomic_number() {
        assert_eq!(
            core_hole_width_ev(0, 1),
            Err(CoreHoleError::InvalidAtomicNumber { z: 0 })
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
