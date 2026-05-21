use crate::Real;

use super::AtomicError;

/// Port of FEFF `COMMON/pertab.f90::atwtd`: return the periodic-table weight.
///
/// This is the table used by FEFF's Debye and Einstein-model mass calculations.
/// It covers `Z = 1..=139` and intentionally preserves FEFF's single-precision
/// rounding of the literal data statements.
pub fn atomic_weight(atomic_number: usize) -> Result<Real, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_WEIGHTS
        .get(atomic_number - 1)
        .map(|&weight| Real::from(weight))
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

/// Port of FEFF `COMMON/pertab.f90::atsym`: return the element symbol.
///
/// FEFF's table is returned trimmed rather than padded to Fortran
/// `character*3` width. The values, including historical placeholder names
/// and table quirks, match FEFF's data statement.
pub fn atomic_symbol(atomic_number: usize) -> Result<&'static str, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_SYMBOLS
        .get(atomic_number - 1)
        .copied()
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
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

const FEFF_ATOMIC_WEIGHTS: [f32; 139] = [
    1.0079, 4.0026, 6.941, 9.0122, 10.81, 12.01, 14.007, 15.999, 18.998, 20.18, 22.9898, 24.305,
    26.982, 28.086, 30.974, 32.064, 35.453, 39.948, 39.09, 40.08, 44.956, 47.90, 50.942, 52.00,
    54.938, 55.85, 58.93, 58.71, 63.55, 65.38, 69.72, 72.59, 74.922, 78.96, 79.91, 83.80, 85.47,
    87.62, 88.91, 91.22, 92.91, 95.94, 98.91, 101.07, 102.90, 106.40, 107.87, 112.40, 114.82,
    118.69, 121.75, 127.60, 126.90, 131.30, 132.91, 137.34, 138.91, 140.12, 140.91, 144.24, 145.0,
    150.35, 151.96, 157.25, 158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95,
    183.85, 186.2, 190.20, 192.22, 195.09, 196.97, 200.59, 204.37, 207.19, 208.98, 210.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.04, 231.0, 238.03, 237.05, 244.0, 243.0, 247.0, 247.0, 251.0,
    252.0, 257.0, 258.0, 259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 282.0,
    285.0, 286.0, 289.0, 289.0, 293.0, 294.0, 294.0, 315.0, 320.0, 330.0, 334.0, 337.0, 340.0,
    344.0, 347.0, 350.0, 354.0, 357.0, 361.0, 364.0, 367.0, 371.0, 374.0, 378.0, 381.0, 385.0,
    388.0, 392.0,
];

const FEFF_ATOMIC_SYMBOLS: [&str; 139] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Te", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo", "Uue", "Ubn", "Ubu", "Ubb", "Ubt", "Ubq", "Ubp", "Ubh",
    "Ubs", "Ubo", "Ube", "Utn", "Utu", "Utb", "Utt", "Utq", "Utp", "Uth", "Uts", "Uto", "Ute",
];

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
