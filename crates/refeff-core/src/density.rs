//! FEFF density and LDOS accumulation helpers.
//!
//! This module ports compact numerical routines that update radial valence
//! densities and angular-momentum-resolved density of states after scattering
//! terms have been computed.

use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView5,
};
use num_complex::Complex32;

use crate::fovrg::{FovrgDiracSolverInput, FovrgError};
use crate::grid::{
    CoulombPotentialSlwInput, GridError, LoucksSphericalOverlapInput, NormanRadius,
    NormanRadiusInput, coulomb_potential_slw, norman_radius_from_density,
    sum_loucks_spherical_overlap,
};
use crate::interpolation::InterpolationError;
use crate::quadrature::{QuadratureError, somm2};
use crate::rhorrp::{RhorrpError, RhorrpWavefunctionTables};
use crate::vector::distance_between;
use crate::{Complex, Real};

const OVRLP_DENSITY_POINTS: usize = 251;
const OVRLP_GEOMETRY_CUTOFF: Real = 12.0;
const COULOM_DELTA: Real = 0.05;
const COULOM_LITERAL_DELTA: Real = 0.05_f32 as Real;
const COULOM_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const BROYDN_DELTA: Real = 0.05;
const BROYDN_LITERAL_OFFSET: Real = 8.8_f32 as Real;

mod broyden;
mod coulomb;
mod ldos;
mod overlap;
mod scf;
mod support;
mod types;
mod valence;

pub use broyden::mix_broyden_density;
pub use coulomb::update_coulomb_potential;
pub use ldos::{
    ldos_ff2rho_tables, ldos_fmsdos_trace, ldos_fmsdos_trace_grid,
    ldos_hubbard_magnetic_ff2rho_tables, ldos_hubbard_step1, ldos_rhol_assemble_radial_components,
    ldos_rhol_channel, ldos_rhol_density, ldos_rhol_density_grid, ldos_rhol_exact_radial_tail,
    ldos_rhol_table_driver, ldos_rhol_wavefunction_tables, ldos_spin_ff2rho_tables,
    pot_rholie_density, pot_rholie_density_grid, pot_scf_contour_source_rows,
    pot_scf_energy_density,
};
pub use overlap::overlap_potential_density;
pub use scf::{
    advance_pot_scf_state, finish_pot_scf_outer_iteration, run_pot_scf_iteration,
    update_scf_density_potential,
};
pub use types::*;
pub use valence::{
    accumulate_pot_scf_energy_point, finish_pot_scf_fermi_endpoint, pot_scf_contour_step,
    run_pot_scf_contour, update_valence_density,
};

use support::*;

#[cfg(test)]
mod tests;
