use super::{
    FmsAtom, FmsBiCgStabInput, FmsFreePropagatorInput, FmsFreePropagatorMatrixInput,
    FmsFullPotentialLuInput, FmsGravesMorrisInput, FmsHubbardFullScatteringTransformInput,
    FmsHubbardScatteringTransformInput, FmsHubbardTMatrixInput, FmsHubbardTMatrixTableInput,
    FmsHubbardTMatrixTransformInput, FmsIterativeSystemInput, FmsLuInput, FmsRealSpaceEnergyInput,
    FmsRecursionInput, FmsRotationDirection, FmsScatteringInput, FmsScatteringMethod,
    FmsScatteringMethodSelection, FmsSpinFreePropagatorMatrixInput, FmsTMatrixInput,
    FmsTMatrixTableInput, FmsTfqmrInput, FmsYprepClusterInput, MkgtrGreenTraceInput,
    MkgtrJasGreenTraceInput, MkgtrJasQPairMode, MkgtrJasTransition, fms_bicgstab_scattering,
    fms_driver_setup, fms_free_propagator_element, fms_free_propagator_matrix,
    fms_full_potential_lu_scattering, fms_graves_morris_scattering,
    fms_hubbard_back_transform_full_scattering, fms_hubbard_back_transform_scattering,
    fms_hubbard_t_matrix_element, fms_hubbard_t_matrix_table, fms_hubbard_transform_t_matrix,
    fms_iterative_system_matrix, fms_lu_scattering, fms_pair_tables, fms_real_space_energy,
    fms_recursion_scattering, fms_rotation_matrix, fms_scattering, fms_scattering_method_selection,
    fms_spin_free_propagator_matrix, fms_spin_pair_tables, fms_t_matrix_element,
    fms_t_matrix_table, fms_tfqmr_scattering, fms_yprep_cluster, fms_yprep_geometry,
    mkgtr_green_trace, mkgtr_jas_green_trace, pair_polar_angles, sort_atoms_by_radius,
    sort_representative_atoms,
};
use super::{
    FmsDriverSetupInput, FmsError, rehr_albers_polynomials, rehr_albers_z_axis_propagator,
};
use crate::{
    Complex, Real, XsphNrixsTransitionIndicesInput,
    angular::{TransitionBMatrix, legendre_normalization_table, spin_orbit_coupling_tables},
    state::{StateKet, construct_state_kets},
    xsph_nrixs_transition_indices,
};
use ndarray::{
    Array1, Array2, Array3, Array4, Array5, Array6, ArrayView2, ArrayView3, ArrayView4, Axis,
    ShapeBuilder, array,
};
use num_complex::Complex32;
use std::error::Error;

const REFERENCE_LCALC: [bool; 2] = [true, true];
const REFERENCE_POTENTIAL_LMAX: [usize; 1] = [1];

mod cluster;
mod driver;
mod mkgtr;
mod pairs;
mod reciprocal;
mod scattering_dispatch;
mod solvers;
mod t_matrix;
mod xgllm;

fn reference_xgllm_tables() -> Result<(Array4<Complex32>, Array2<Real>), Box<dyn Error>> {
    let clm = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;
    let mut xclm = Array4::zeros((4, 4, 2, 2).f());
    for l in 0..=3 {
        for m in 0..=3 {
            xclm[(m, l, 1, 0)] = clm[(l, m)];
            xclm[(m, l, 0, 1)] = clm[(l, m)];
        }
    }
    Ok((xclm, legendre_normalization_table(3)?))
}

fn reference_phase_shifts() -> Array3<Complex32> {
    let mut phases = Array3::zeros((2, 5, 2).f());
    phases[(0, 4, 1)] = Complex32::new(0.2, 0.05);
    phases[(0, 0, 1)] = Complex32::new(-0.1, 0.03);
    phases[(1, 4, 1)] = Complex32::new(0.15, -0.02);
    phases[(1, 0, 1)] = Complex32::new(0.07, 0.04);
    phases
}

fn reference_gglu_inputs(state_count: usize) -> (Array2<Complex32>, Array2<Complex32>) {
    let mut free_propagator = Array2::zeros((state_count, state_count).f());
    let mut t_matrix = Array2::zeros((2, state_count).f());
    for column in 0..state_count {
        for row in 0..state_count {
            let row_feff = row as f32 + 1.0;
            let column_feff = column as f32 + 1.0;
            if row != column {
                free_propagator[(row, column)] = Complex32::new(
                    0.01 * row_feff - 0.02 * column_feff,
                    0.015 * row_feff + 0.005 * column_feff,
                );
            }
        }
        let column_feff = column as f32 + 1.0;
        t_matrix[(0, column)] = Complex32::new(0.02 * column_feff, -0.01 * column_feff);
        t_matrix[(1, column)] = Complex32::new(-0.005 * column_feff, 0.003 * column_feff);
    }
    (free_propagator, t_matrix)
}

fn reference_scattering_input<'a>(
    method: FmsScatteringMethod,
    states: &'a [StateKet],
    representative_offsets: &'a [Option<usize>],
    free_propagator: ArrayView2<'a, Complex32>,
    t_matrix: ArrayView2<'a, Complex32>,
) -> FmsScatteringInput<'a> {
    FmsScatteringInput {
        method,
        calculate_full_scattering: false,
        states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &REFERENCE_POTENTIAL_LMAX,
        representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator,
        t_matrix,
        calculated_l: &REFERENCE_LCALC,
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    }
}

fn reference_full_potential_t_matrix(state_count: usize) -> Array2<Complex32> {
    let mut t_matrix = Array2::zeros((state_count, state_count).f());
    for column in 0..state_count {
        for row in 0..state_count {
            let row_feff = row as f32 + 1.0;
            let column_feff = column as f32 + 1.0;
            t_matrix[(row, column)] = Complex32::new(
                0.002 * row_feff + 0.001 * column_feff,
                -0.0015 * row_feff + 0.0007 * column_feff,
            );
        }
    }
    t_matrix
}

fn matrix_sum(matrix: ArrayView2<'_, Complex32>) -> Complex32 {
    matrix
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn nonzero_count(matrix: ArrayView2<'_, Complex32>) -> usize {
    matrix
        .iter()
        .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
        .count()
}

fn rotation_sum(matrix: ArrayView3<'_, Complex32>) -> Complex32 {
    matrix
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn rotation_nonzero_count(matrix: ArrayView3<'_, Complex32>) -> usize {
    matrix
        .iter()
        .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
        .count()
}

fn scattering_sum(table: ArrayView3<'_, Complex32>) -> Complex32 {
    table
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn pair_table_sum(table: ArrayView4<'_, Complex32>) -> Complex32 {
    table
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn pair_table_nonzero_count(table: ArrayView4<'_, Complex32>) -> usize {
    table
        .iter()
        .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
        .count()
}

fn rotation_value(
    matrix: &Array3<Complex32>,
    m2: isize,
    m1: isize,
    angular_momentum: usize,
) -> Complex32 {
    let offset = 3_isize;
    matrix[(
        (m2 + offset) as usize,
        (m1 + offset) as usize,
        angular_momentum,
    )]
}

fn copy_rotation_pair(
    rotations: &mut Array6<Complex32>,
    atom2: usize,
    atom1: usize,
    direction: FmsRotationDirection,
    table: &Array3<Complex32>,
) {
    let branch = match direction {
        FmsRotationDirection::Forward => 0,
        FmsRotationDirection::Backward => 1,
    };
    for l in 0..table.shape()[2] {
        for m1 in 0..table.shape()[1] {
            for m2 in 0..table.shape()[0] {
                rotations[(m2, m1, l, branch, atom2, atom1)] = table[(m2, m1, l)];
            }
        }
    }
}

fn sample_mkgtr_transition_matrix(orbital_momenta: [i32; 8]) -> TransitionBMatrix {
    TransitionBMatrix {
        kappa_indices: [0; 8],
        orbital_momenta,
        matrix: Array6::zeros((1, 2, 8, 1, 2, 8).f()),
        l_offset: 0,
    }
}

fn widen_complex32_for_test(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert!(
        (actual - expected).norm() < 1.0e-11,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_complex32_close(actual: Complex32, expected: Complex32) {
    assert!(
        (actual - expected).norm() < 2.0e-4,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_close_f32(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 2.0e-6,
        "actual={actual} expected={expected}"
    );
}

fn assert_close_f64(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual} expected={expected}"
    );
}
