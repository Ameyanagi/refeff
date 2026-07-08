//! Atomic lookup tables and small ATOM helper kernels ported from FEFF.
//!
//! This module ports `ATOM/nucmass.f90` and `COMMON/pertab.f90`. FEFF stores
//! many unsuffixed real literals in double-precision arrays, so those values
//! are rounded through single precision before use; the Rust tables keep that
//! behavior explicitly. It also includes compact helper routines from
//! `ATOM/aprdev.f90`, `ATOM/cofcon.f90`, `ATOM/dentfa.f90`,
//! `ATOM/fdmocc.f90`, `ATOM/akeato.f90`, `ATOM/muatco.f90`,
//! `ATOM/inmuat.f90`,
//! `ATOM/lagdat.f90`, `ATOM/ortdat.f90`, `ATOM/tabrat.f90`,
//! `ATOM/fpf0.f90`, `ATOM/nucdev.f90`, `ATOM/dsordf.f90`,
//! `ATOM/yzkteg.f90`, `ATOM/yzkrdf.f90`, `ATOM/fdrirk.f90`,
//! `ATOM/wfirdf.f90`,
//! `ATOM/vlda.f90`, `ATOM/potrdf.f90`, `ATOM/intdir.f90`,
//! `ATOM/soldir.f90`,
//! `ATOM/bkmrdf.f90`, `ATOM/potslw.f90`, and
//! `ATOM/s02at.f90`.

use crate::Real;
use crate::angular::wigner_3j;
use crate::constants::{
    BOHR_ANGSTROM as ATOM_FPF0_BOHR_ANGSTROM, HARTREE_EV as ATOM_TABRAT_HARTREE_EV,
};
use crate::exchange::{dirac_hara_exchange_potential, von_barth_hedin_potential};
use crate::grid::FEFF_FERMI_MOMENTUM_FACTOR;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder, Slice,
};

const ATOM_TABRAT_MOMENT_POWERS: [i32; 7] = [6, 4, 2, 1, -1, -2, -3];
const ATOM_TABRAT_LABELS: [&str; 9] = ["s", "p*", "p", "d*", "d", "f*", "f", "g*", "g"];
const ATOM_FPF0_FINE_STRUCTURE: Real = 1.0 / 137.035_989_56;
const ATOM_FPF0_FORM_FACTOR_POINTS: usize = 81;
const ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM: Real = 0.5;
const ATOM_NUCDEV_RADIUS_FACTOR: Real = 2.2677e-05;
const ATOM_INMUAT_WAVEFUNCTION_PRECISION: Real = 1.0e-5;
const ATOM_INMUAT_ENERGY_PRECISION: Real = 5.0e-6;
const ATOM_INMUAT_PRIMARY_RATIO: Real = 100.0;
const ATOM_INMUAT_SECONDARY_RATIO: Real = 10.0;
const ATOM_INMUAT_DEVELOPMENT_ORDER: usize = 10;
const ATOM_INMUAT_ATTEMPT_COUNT: usize = 50;
const ATOM_INMUAT_NUCLEUS_INDEX: usize = 11;
const ATOM_INMUAT_LAGRANGE_CAPACITY: usize = 820;
const ATOM_INMUAT_DEFAULT_RADIAL_COUNT: usize = 251;
const ATOM_INMUAT_ELECTRON_TOLERANCE: Real = 0.001;
const ATOM_INMUAT_DEFAULT_CONVERGENCE: Real = 0.3_f32 as Real;

mod breit;
mod dirac;
mod driver;
mod form_factor;
mod initial;
mod math;
mod nuclear;
mod orbitals;
mod overlap;
mod radial;
mod scf;
mod schmidt;
mod tables;
mod tabulation;
mod total_energy;
mod types;
mod validation;

pub use breit::*;
pub use dirac::*;
pub use driver::*;
pub use form_factor::*;
pub use initial::*;
pub use math::{
    atomic_convergence_mix, atomic_coulomb_coefficients, atomic_direct_coulomb_coefficient,
    atomic_exchange_coulomb_coefficient, atomic_occupation_product,
    atomic_polynomial_product_coefficient, thomas_fermi_density_potential,
};
pub use nuclear::*;
pub use orbitals::*;
pub use overlap::*;
pub use radial::*;
pub use scf::*;
pub use schmidt::*;
pub use tables::*;
pub use tabulation::*;
pub use total_energy::*;
pub use types::*;

use math::*;
use validation::*;

#[cfg(test)]
mod tests;
