//! SCREEN and CRPA response-matrix helper kernels.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use num_complex::Complex32;
use refeff_linalg::{real_lu_factor, real_lu_solve_vector};

use crate::{Complex, ComplexMat, Real, RealMat, RealVec};

use super::types::*;
use super::validation::{
    validate_active_count, validate_active_matrix_shape, validate_count_at_least, validate_finite,
    validate_finite_complex_input, validate_finite_complex_matrix,
    validate_finite_complex32_matrix, validate_finite_matrix, validate_increasing,
    validate_positive, validate_result_finite, validate_result_finite_complex,
};

/// Port the SCREEN/CRPA contour trapezoid energy step.
///
/// `screensub.f90` and `chi_crpa.f90` integrate each `chi0re(:,:,ie)` slice
/// with endpoint half-steps and centered interior steps:
/// `(em(ie+1) - em(ie-1)) / 2`. The `energy_index` argument is zero-based and
/// maps to FEFF's one-based `ie`.
pub fn screen_energy_integration_delta(
    energies: ArrayView1<'_, Complex>,
    energy_index: usize,
) -> Result<Complex, ScreenError> {
    validate_count_at_least("energies", energies.len(), 2)?;
    if energy_index >= energies.len() {
        return Err(ScreenError::EnergyIndexOutOfRange {
            index: energy_index,
            len: energies.len(),
        });
    }
    for &energy in energies {
        validate_finite_complex_input("energy", energy)?;
    }

    let delta = if energy_index == 0 {
        (energies[1] - energies[0]) / 2.0
    } else if energy_index + 1 == energies.len() {
        (energies[energy_index] - energies[energy_index - 1]) / 2.0
    } else {
        (energies[energy_index + 1] - energies[energy_index - 1]) / 2.0
    };
    validate_result_finite_complex("energy_integration_delta", delta)?;
    Ok(delta)
}

/// Accumulate one SCREEN/CRPA response slice into the contour integral.
///
/// FEFF stores only the active upper triangle during the energy loop:
/// `chi0r(ir1,i) += chi0re(ir1,i) * de` for `ir1 <= i`. This helper preserves
/// that convention and leaves the lower triangle from `accumulated` unchanged;
/// use [`screen_symmetrize_response_upper`] before building the response system.
pub fn screen_integrate_response_step(
    accumulated: ArrayView2<'_, Complex>,
    response_at_energy: ArrayView2<'_, Complex>,
    energy_delta: Complex,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape(
        "accumulated_response",
        accumulated.nrows(),
        accumulated.ncols(),
        active_count,
    )?;
    validate_active_matrix_shape(
        "response_at_energy",
        response_at_energy.nrows(),
        response_at_energy.ncols(),
        active_count,
    )?;
    validate_finite_complex_input("energy_delta", energy_delta)?;

    let mut output = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in 0..active_count {
            let value = accumulated[(row, column)];
            validate_finite_complex_matrix("accumulated_response", row, column, value)?;
            output[(row, column)] = value;
        }
    }
    for row in 0..active_count {
        for column in row..active_count {
            let response = response_at_energy[(row, column)];
            validate_finite_complex_matrix("response_at_energy", row, column, response)?;
            let value = output[(row, column)] + response * energy_delta;
            validate_result_finite_complex("integrated_response", value)?;
            output[(row, column)] = value;
        }
    }
    Ok(output)
}

/// Mirror FEFF's stored upper-triangle response matrix.
///
/// The original SCREEN/CRPA routines fill `chi0r(ir1,i)` only for `ir1 <= i`
/// during energy integration, then copy `chi0r(i,ir1)` into the lower triangle
/// before solving. This is a plain symmetric copy, not a Hermitian conjugate.
pub fn screen_symmetrize_response_upper(
    response_upper: ArrayView2<'_, Complex>,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape(
        "response_upper",
        response_upper.nrows(),
        response_upper.ncols(),
        active_count,
    )?;

    let mut output = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in row..active_count {
            let value = response_upper[(row, column)];
            validate_finite_complex_matrix("response_upper", row, column, value)?;
            output[(row, column)] = value;
            output[(column, row)] = value;
        }
    }
    Ok(output)
}

/// Port the CRPA angular-channel density row.
///
/// `chi_crpa.f90` stores
/// `DIMAG((pr*pn + pr**2*gtrl)*ck*4) * (2*l + 1) / pi` in `den_CRPA(:,ie)`
/// for the selected CRPA angular momentum. The regular and irregular radial
/// solutions are passed as `ndarray` views over FEFF's active radial prefix.
pub fn screen_crpa_orbital_density(
    regular_solution: ArrayView1<'_, Complex>,
    irregular_solution: ArrayView1<'_, Complex>,
    cluster_green: Complex,
    wave_number: Complex,
    angular_momentum: usize,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_scale = (2.0 * angular_momentum as Real + 1.0) / std::f64::consts::PI;
    let mut density = Array1::zeros(active_count);
    for index in 0..active_count {
        let regular = regular_solution[index];
        let irregular = irregular_solution[index];
        validate_finite_complex_input("regular_solution", regular)?;
        validate_finite_complex_input("irregular_solution", irregular)?;
        let response =
            (regular * irregular + regular * regular * cluster_green) * wave_number * 4.0;
        validate_result_finite_complex("crpa_orbital_density_response", response)?;
        let value = response.im * angular_scale;
        validate_result_finite("crpa_orbital_density", value)?;
        density[index] = value;
    }
    Ok(density)
}

/// Build one SCREEN atomic response slice.
///
/// In `screensub.f90`, each angular channel adds an upper-triangle contribution
/// `factor * r(m) * r(n) * pr(m)^2 * pn(n)^2`, where
/// `factor = -((2*l + 1) * (2*ck)^2 * dx^2) / (2*pi^2)`. The returned matrix
/// stores the active upper triangle in Fortran order; lower-triangle entries
/// remain zero until [`screen_symmetrize_response_upper`] is applied after
/// energy integration.
pub fn screen_atomic_response_slice(
    radii: &[Real],
    regular_solution: ArrayView1<'_, Complex>,
    irregular_solution: ArrayView1<'_, Complex>,
    wave_number: Complex,
    dx: Real,
    angular_momentum: usize,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor = doubled_wave
        * doubled_wave
        * (-(angular_weight * dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("atomic_response_prefactor", prefactor)?;

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        let row_radius = radii[row];
        let regular = regular_solution[row];
        validate_positive("radius", row_radius)?;
        validate_finite_complex_input("regular_solution", regular)?;
        for column in row..active_count {
            let column_radius = radii[column];
            let irregular = irregular_solution[column];
            validate_positive("radius", column_radius)?;
            validate_finite_complex_input("irregular_solution", irregular)?;
            let value =
                prefactor * row_radius * column_radius * regular * regular * irregular * irregular;
            validate_result_finite_complex("atomic_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Build one SCREEN FMS cluster response correction slice.
///
/// When the FMS cluster contains more than the absorber, `screensub.f90` adds a
/// `1:jnrm` upper-triangle correction to the atomic response slice:
/// `factor*r(m)*r(n)*(2*gtrl*pr(m)^2*pr(n)*pn(n) + gtrl^2*pr(m)^2*pr(n)^2)`.
/// `fms_count` is FEFF `jnrm`; entries outside that prefix remain zero.
pub fn screen_fms_response_slice(
    input: ScreenFmsResponseSliceInput<'_>,
) -> Result<ComplexMat, ScreenError> {
    let ScreenFmsResponseSliceInput {
        radii,
        regular_solution,
        irregular_solution,
        cluster_green,
        wave_number,
        dx,
        angular_momentum,
        active_count,
        fms_count,
    } = input;

    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_count_at_least("fms_count", fms_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    if fms_count > active_count {
        return Err(ScreenError::ActiveCountOutOfRange {
            active_count: fms_count,
            len: active_count,
        });
    }
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor = doubled_wave
        * doubled_wave
        * (-(angular_weight * dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("fms_response_prefactor", prefactor)?;
    let cluster_green_squared = cluster_green * cluster_green;

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..fms_count {
        let row_radius = radii[row];
        let regular_row = regular_solution[row];
        validate_positive("radius", row_radius)?;
        validate_finite_complex_input("regular_solution", regular_row)?;
        let regular_row_squared = regular_row * regular_row;
        for column in row..fms_count {
            let column_radius = radii[column];
            let regular_column = regular_solution[column];
            let irregular_column = irregular_solution[column];
            validate_positive("radius", column_radius)?;
            validate_finite_complex_input("regular_solution", regular_column)?;
            validate_finite_complex_input("irregular_solution", irregular_column)?;
            let cluster_term =
                2.0 * cluster_green * regular_row_squared * regular_column * irregular_column
                    + cluster_green_squared * regular_row_squared * regular_column * regular_column;
            let value = prefactor * row_radius * column_radius * cluster_term;
            validate_result_finite_complex("fms_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Build one CRPA response slice from `chi_crpa.f90`.
///
/// CRPA stores the same upper-triangle `chi0re(m,n)` workspace as SCREEN, but
/// separates the angular prefactor from the base factor and applies a
/// `sin(...)^4` radial projection to the selected constrained channel. Passing
/// `cluster_green = 0` yields the atomic part. A nonzero `cluster_green` adds
/// the diagonal FMS terms used by the CRPA driver.
pub fn screen_crpa_response_slice(
    input: ScreenCrpaResponseSliceInput<'_>,
) -> Result<ComplexMat, ScreenError> {
    let ScreenCrpaResponseSliceInput {
        radii,
        regular_solution,
        irregular_solution,
        cluster_green,
        wave_number,
        dx,
        angular_momentum,
        crpa_angular_momentum,
        projection_window,
        active_count,
    } = input;

    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;
    let projection_window = projection_window.filter(|_| angular_momentum == crpa_angular_momentum);
    if let Some(window) = projection_window {
        validate_finite("projection_inner_radius", window.inner_radius)?;
        validate_finite("projection_outer_radius", window.outer_radius)?;
        validate_increasing(
            "projection_inner_radius",
            window.inner_radius,
            "projection_outer_radius",
            window.outer_radius,
        )?;
    }

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor =
        doubled_wave * doubled_wave * (-(dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("crpa_response_prefactor", prefactor)?;
    let cluster_green_squared = cluster_green * cluster_green;

    let mut projection_weights = Vec::with_capacity(active_count);
    for &radius in radii.iter().take(active_count) {
        validate_positive("radius", radius)?;
        let weight = match projection_window {
            Some(window) => crpa_response_projection_weight(radius, window)?,
            None => 1.0,
        };
        projection_weights.push(weight);
    }

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        let row_radius = radii[row];
        let regular_row = regular_solution[row];
        validate_finite_complex_input("regular_solution", regular_row)?;
        let row_factor = row_radius * projection_weights[row] * regular_row * regular_row;
        for column in row..active_count {
            let column_radius = radii[column];
            let regular_column = regular_solution[column];
            let irregular_column = irregular_solution[column];
            validate_finite_complex_input("regular_solution", regular_column)?;
            validate_finite_complex_input("irregular_solution", irregular_column)?;
            let response_column = irregular_column * irregular_column
                + 2.0 * cluster_green * regular_column * irregular_column
                + cluster_green_squared * regular_column * regular_column;
            let value = prefactor
                * angular_weight
                * row_factor
                * column_radius
                * projection_weights[column]
                * response_column;
            validate_result_finite_complex("crpa_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Convert an FMS diagonal scattering block into SCREEN/CRPA `gtrl(l,ie)`.
///
/// `screensub.f90` sums `gg(l^2+m,l^2+m,iph)` over the `2*l+1` magnetic
/// substates, widens the single-precision FMS result to double precision, and
/// applies the absorber phase factor `exp(2*i*ph_l)/(2*l+1)`. The CRPA
/// diagonal `gtrl(l,l,ie)` expression reduces to the same formula.
pub fn screen_fms_cluster_green_trace(
    scattering: ArrayView2<'_, Complex32>,
    phase_shift: Complex,
    angular_momentum: usize,
) -> Result<Complex, ScreenError> {
    validate_finite_complex_input("phase_shift", phase_shift)?;
    let start =
        angular_momentum
            .checked_mul(angular_momentum)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "angular_momentum",
            })?;
    let required_order = angular_momentum
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(ScreenError::IndexSizeOverflow {
            name: "angular_momentum",
        })?;
    validate_active_matrix_shape(
        "fms_scattering",
        scattering.nrows(),
        scattering.ncols(),
        required_order,
    )?;

    let mut trace = Complex::new(0.0, 0.0);
    for state_index in start..required_order {
        let value = scattering[(state_index, state_index)];
        validate_finite_complex32_matrix("fms_scattering", state_index, state_index, value)?;
        trace += Complex::new(value.re as Real, value.im as Real);
    }

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let value = trace * (Complex::new(0.0, 2.0) * phase_shift).exp() / angular_weight;
    validate_result_finite_complex("fms_cluster_green_trace", value)?;
    Ok(value)
}

fn crpa_response_projection_weight(
    radius: Real,
    window: ScreenCrpaProjectionWindow,
) -> Result<Real, ScreenError> {
    let clamped = radius.max(window.inner_radius).min(window.outer_radius);
    let scaled = (clamped - window.inner_radius) / (window.outer_radius - window.inner_radius);
    let weight = (scaled * std::f64::consts::FRAC_PI_2).sin().powi(4);
    validate_result_finite("crpa_response_projection_weight", weight)?;
    Ok(weight)
}

/// Port the SCREEN/CRPA response-system matrix setup.
///
/// FEFF builds the real system matrix as `A = I - K * imag(chi0)`, then passes
/// that matrix to LAPACK `dgetrf`/`dgetrs`. The inputs are `ndarray` views so
/// callers can pass full FEFF work arrays and select the active `ilast` prefix.
/// The returned matrix uses Fortran-order storage to preserve the layout that
/// downstream FEFF-compatible linear algebra expects.
pub fn screen_response_system_matrix(
    kernel: ArrayView2<'_, Real>,
    susceptibility: ArrayView2<'_, Complex>,
    active_count: usize,
) -> Result<RealMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape("kernel", kernel.nrows(), kernel.ncols(), active_count)?;
    validate_active_matrix_shape(
        "susceptibility",
        susceptibility.nrows(),
        susceptibility.ncols(),
        active_count,
    )?;

    for row in 0..active_count {
        for column in 0..active_count {
            validate_finite_matrix("kernel", row, column, kernel[(row, column)])?;
            validate_finite_complex_matrix(
                "susceptibility",
                row,
                column,
                susceptibility[(row, column)],
            )?;
        }
    }

    let mut system = Array2::zeros((active_count, active_count).f());
    for index in 0..active_count {
        system[(index, index)] = 1.0;
    }
    for column in 0..active_count {
        for index in 0..active_count {
            let susceptibility_imaginary = susceptibility[(index, column)].im;
            if susceptibility_imaginary == 0.0 {
                continue;
            }
            for row in 0..active_count {
                system[(row, column)] -= kernel[(row, index)] * susceptibility_imaginary;
            }
        }
        for row in 0..active_count {
            validate_result_finite("response_system_matrix", system[(row, column)])?;
        }
    }
    Ok(system)
}

/// Solve FEFF's screened-core-hole response equation.
///
/// This is the matrix-inversion block shared by `SCREEN/screensub.f90` and
/// `CRPA/chi_crpa.f90`: build `A = I - K * imag(chi0)` and solve
/// `A * wscrn = v_ch` with FEFF-compatible real LU factorization. The result is
/// the screened potential vector that FEFF stores back into `wscrn`.
pub fn screen_solve_response_potential(
    kernel: ArrayView2<'_, Real>,
    susceptibility: ArrayView2<'_, Complex>,
    bare_potential: ArrayView1<'_, Real>,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, bare_potential.len())?;
    for &value in bare_potential.iter().take(active_count) {
        validate_finite("bare_potential", value)?;
    }

    let system = screen_response_system_matrix(kernel, susceptibility, active_count)?;
    let rhs = Array1::from_iter(bare_potential.iter().take(active_count).copied());
    let lu = real_lu_factor(system.view())?;
    let solution = real_lu_solve_vector(&lu, rhs.view())?;
    for &value in &solution {
        validate_result_finite("screened_response_potential", value)?;
    }
    Ok(solution)
}
