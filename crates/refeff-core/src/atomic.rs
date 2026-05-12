//! Atomic lookup tables ported from FEFF.
//!
//! This module currently ports `ATOM/nucmass.f90`, the standard atomic-weight
//! table used by FEFF's high-Z nuclear-potential setup. FEFF stores unsuffixed
//! real literals in a double-precision array, so the values are rounded through
//! single precision before use; the Rust table keeps that behavior explicitly.

use thiserror::Error;

use crate::Real;

/// Error returned by FEFF atomic lookup helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AtomicError {
    /// FEFF `nucmass` contains values for `Z = 1..=138`.
    #[error("FEFF nuclear mass table covers atomic numbers 1..=138, got {z}")]
    InvalidAtomicNumber { z: usize },
}

/// Port of FEFF `nucmass`: return the tabulated standard atomic weight.
///
/// The value is returned in atomic mass units. FEFF uses this table when the
/// `HIGHZ` path requests a finite nuclear-radius model for heavy atoms.
pub fn nuclear_mass(atomic_number: usize) -> Result<Real, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_NUCLEAR_MASSES
        .get(atomic_number - 1)
        .map(|&mass| Real::from(mass))
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

#[allow(clippy::excessive_precision)]
const FEFF_NUCLEAR_MASSES: [f32; 138] = [
    1.00794,
    4.002602,
    6.941,
    9.012182,
    10.811,
    12.0107,
    14.0067,
    15.9994,
    18.9984032,
    20.1797,
    22.98976928,
    24.305,
    26.9815386,
    28.0855,
    30.973762,
    32.065,
    35.453,
    39.948,
    39.0983,
    40.078,
    44.955912,
    47.867,
    50.9415,
    51.9961,
    54.938045,
    55.845,
    58.933195,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.64,
    74.9216,
    78.96,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90585,
    91.224,
    92.90638,
    95.96,
    98.0,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.411,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.9054519,
    137.327,
    138.90547,
    140.116,
    140.90765,
    144.242,
    145.0,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93032,
    167.259,
    168.93421,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.59,
    204.3833,
    207.2,
    208.9804,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.03806,
    231.03588,
    238.02891,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
    258.0,
    259.0,
    262.0,
    265.0,
    268.0,
    271.0,
    272.0,
    277.0,
    276.0,
    281.0,
    280.0,
    285.0,
    284.0,
    289.0,
    288.0,
    293.0,
    294.0,
    294.0,
    315.0,
    320.0,
    330.0,
    334.0,
    337.0,
    340.0,
    344.0,
    347.0,
    350.0,
    354.0,
    357.0,
    361.0,
    364.0,
    367.0,
    371.0,
    374.0,
    378.0,
    381.0,
    385.0,
    388.0,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn nuclear_mass_matches_feff_reference() -> Result<(), AtomicError> {
        assert_close(nuclear_mass(1)?, 1.007_940_053_939_819_3);
        assert_close(nuclear_mass(6)?, 12.010_700_225_830_078);
        assert_close(nuclear_mass(29)?, 63.546_001_434_326_17);
        assert_close(nuclear_mass(57)?, 138.905_471_801_757_8);
        assert_close(nuclear_mass(92)?, 238.028_915_405_273_44);
        assert_close(nuclear_mass(118)?, 294.0);
        assert_close(nuclear_mass(121)?, 330.0);
        assert_close(nuclear_mass(138)?, 388.0);
        Ok(())
    }

    #[test]
    fn nuclear_mass_rejects_invalid_atomic_numbers() {
        assert_eq!(
            nuclear_mass(0),
            Err(AtomicError::InvalidAtomicNumber { z: 0 })
        );
        assert_eq!(
            nuclear_mass(139),
            Err(AtomicError::InvalidAtomicNumber { z: 139 })
        );
    }
}
