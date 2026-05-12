#![forbid(unsafe_code)]

//! Core numerical types for the FEFF10 Rust port.
//!
//! `ndarray` is the primary storage and view API. Helpers in this crate create
//! arrays with FEFF-friendly Fortran-order layout where the original algorithms
//! and file formats depend on column-major traversal.

use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::Complex64;

pub mod angular;
pub mod bessel;
pub mod convolution;
pub mod core_hole;
pub mod fms;
pub mod grid;
pub mod interpolation;
pub mod phase;
pub mod quadrature;
pub mod roots;
pub mod self_energy;
pub mod sort;
pub mod special;
pub mod state;
pub mod vector;

pub use angular::{
    AngularError, PolarizationTensorMode, SpinOrbitCouplingTables, TransitionBMatrix,
    TransitionBMatrixInput, legendre_normalization, legendre_normalization_table,
    legendre_polynomials, legendre_polynomials_into, polarization_tensor, spherical_harmonics,
    spin_orbit_coupling_tables, transition_b_matrix, wigner_3j, wigner_rotation,
};
pub use bessel::{
    BesselError, SphericalBessel, SphericalBesselValue, SphericalHankel, besjh, besjn, exjlnl,
    spherical_bessel_j_h, spherical_bessel_j_y,
};
pub use convolution::{
    ConvolutionError, conv, conv as lorentz_convolve, conv_in_place, conv1,
    conv1 as lorentz_convolution_segment,
};
pub use core_hole::{
    CoreHoleError, CoreHoleQuantumNumbers, core_hole_quantum_numbers, core_hole_width_ev,
    edge_index, is_edge_label,
};
pub use fms::{
    FmsAtom, FmsError, pair_polar_angles, rehr_albers_polynomials, rehr_albers_z_axis_propagator,
    sort_atoms_by_radius, sort_representative_atoms,
};
pub use grid::{
    GridError, LOUCKS_DELTA, LOUCKS_X_OFFSET, loucks_index_below, loucks_radius, loucks_x,
    radial_index_below, radial_radius, radial_x, wave_number_from_hartree,
};
pub use interpolation::{
    Interpolation, InterpolationError, LintCache, lint, lint_with_cache, locate_below,
    polynomial_interpolate, polynomial_interpolate_complex, terp, terp1, terpc,
};
pub use phase::{
    ComplexAmplitudePhase, PhaseError, complex_atan, complex_atan2_amplitude_phase,
    muffin_tin_phase_amplitude, remove_phase_jump, remove_phase_jumps, remove_phase_jumps_array,
};
pub use quadrature::{QuadratureError, csomm, csomm2, somm, somm2, strap, trap};
pub use roots::{ComplexRoots, RootError, cubic_zeros, depressed_quartic_roots, quadratic_zeros};
pub use self_energy::{SelfEnergyError, SingularityFunction, find_self_energy_singularities};
pub use sort::{
    SortError, qsortd_order_1based, qsorti_compatible_order, qsorti_order_1based, sort_order,
    sort_order_1based,
};
pub use special::{SpecialFunctionError, x_log_x};
pub use state::{
    StateKet, StateKetError, StateKetSet, construct_state_kets, construct_state_kets_with_limit,
};
pub use vector::{
    ReferenceFrameRotation, Vector3, distance_between, normalize_vector, nrixs_qtrig,
    rotate_into_reference_frame, single_precision_distance_between, vector_norm,
};

pub type Real = f64;
pub type Complex = Complex64;

pub type RealVec = Array1<Real>;
pub type RealMat = Array2<Real>;
pub type RealCube = Array3<Real>;
pub type RealArray4 = Array4<Real>;

pub type ComplexVec = Array1<Complex>;
pub type ComplexMat = Array2<Complex>;
pub type ComplexCube = Array3<Complex>;
pub type ComplexArray4 = Array4<Complex>;

pub fn zeros_real_vec(len: usize) -> RealVec {
    Array1::zeros(len)
}

pub fn zeros_complex_vec(len: usize) -> ComplexVec {
    Array1::zeros(len)
}

pub fn zeros_real_mat_fortran(rows: usize, cols: usize) -> RealMat {
    Array2::zeros((rows, cols).f())
}

pub fn zeros_complex_mat_fortran(rows: usize, cols: usize) -> ComplexMat {
    Array2::zeros((rows, cols).f())
}

pub fn zeros_real_cube_fortran(a: usize, b: usize, c: usize) -> RealCube {
    Array3::zeros((a, b, c).f())
}

pub fn zeros_complex_cube_fortran(a: usize, b: usize, c: usize) -> ComplexCube {
    Array3::zeros((a, b, c).f())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeffDimensions {
    pub nph: usize,
    pub ne: usize,
    pub nsp: usize,
    pub lmax: usize,
}

impl FeffDimensions {
    pub fn phase_shape(self) -> (usize, usize, usize, usize) {
        (self.ne, 2 * self.lmax + 1, self.nsp, self.nph + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_fortran_order_matrix_storage() {
        let mat = zeros_real_mat_fortran(3, 2);
        assert_eq!(mat.shape(), &[3, 2]);
        assert_eq!(mat.strides(), &[1, 3]);
    }

    #[test]
    fn computes_phase_shape_with_absorber_potential() {
        let dims = FeffDimensions {
            nph: 2,
            ne: 10,
            nsp: 1,
            lmax: 3,
        };
        assert_eq!(dims.phase_shape(), (10, 7, 1, 3));
    }
}
