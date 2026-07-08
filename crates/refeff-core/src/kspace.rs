//! Reciprocal-space lattice helpers ported from FEFF BAND/KSPACE routines.
//!
//! The routines here cover the small deterministic helpers used before the
//! heavier KSPACE solvers: Bravais classification from `BAND/ibravais.f90`,
//! high-symmetry K-path segment generation from `BAND/kpath.f90`, point-group
//! operation discovery and closure checks from `KSPACE/pointgroup.f90` and
//! `KSPACE/symmetrycheck.f90`, k-mesh division helpers from `KSPACE/kmesh.f90`,
//! the LDOS Weyl-mesh replacement from `LDOS/reldos.f90` `changeklist`, and
//! the coordinate reductions from `KSPACE/subtract_a.f90` and `change_car.f90`.
//! FEFF exits the process for unsupported lattices; Rust returns typed errors.

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

use crate::{Real, RealMat, Vector3};

const PI2: Real = std::f64::consts::TAU;
const BRAVAIS_PI2: Real = 2.0 * (std::f32::consts::PI as Real);
const BRAVAIS_RIGHT_ANGLE: Real = 1_570_796.0 / 1_000_000.0;
const BRAVAIS_ANGLE_EPSILON: Real = 0.0001;
const REDUCE_NEGATIVE_EPSILON: Real = -1.0e-8;
const LATTICE_VOLUME_EPSILON: Real = Real::EPSILON;
const BASDIV_OFFSET: Real = 1.0e-6;
/// FEFF `KSPACE/m_tetrahedra.f90` `mwrit` chunk size for tetrahedron records.
pub const KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE: usize = 101;
const DIVISI_PRIMES: [i32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const DIVISI_ITERATIONS: usize = 10;

mod basis;
mod mesh;
mod path;
mod structure_factor;
mod support;
mod symmetry;
mod types;
mod weyl;

pub use basis::{
    bravais_lattice, bravais_lattice_index, change_cartesian_basis, kmesh_bravais_basis,
    reciprocal_lattice_vectors, reduce_to_lattice_cell, subtract_lattice_translation,
};
pub use mesh::{
    kmesh_arbitrary_mesh, kmesh_basis_divisions, kmesh_tetrahedron_division,
    kmesh_tetrahedron_records, reduce_kmesh_common_divisor, reduce_kmesh_irreducible_points,
};
pub use path::{band_k_path_mesh, define_k_path};
pub use structure_factor::{
    KSPACE_Q_PAIR_TOLERANCE, kspace_angular_tables, kspace_direct_lattice_setup,
    kspace_direct_lattice_terms, kspace_energy_dependent_terms, kspace_ewald_energy_tables,
    kspace_harmonic_polynomials, kspace_q_pair_groups, kspace_qjltab,
    kspace_reciprocal_lattice_setup, kspace_reciprocal_pair_phases, kspace_strbbdd_lattice_sum,
    kspace_strset_non_rel_from_lattice_sum, kspace_strset_non_relativistic,
    kspace_strset_rel_from_lattice_sum, kspace_strset_relativistic,
};
pub use symmetry::{
    point_group_operations, reciprocal_metric, redefine_lattice_symmetry_operations,
    symmetry_check, transform_lapw_symmetry_operations,
};
pub use types::*;
pub use weyl::ldos_weyl_kmesh;

#[cfg(test)]
mod tests;
