use super::lambda::checked_i32;
use super::validation::*;
use super::*;
use crate::{
    atomic_symbol, fullspectrum::FEFF_BOHR_ANGSTROM, grid::wave_number_from_hartree,
    phase::remove_phase_jump, quadrature::trap,
};
use ndarray::{Axis, Slice};

/// Plan the ordinary FEFF GENFMT `fmtrxi` scattering-matrix calls for one path.
///
/// This ports the driver block after `sclmz`: the first `f(2,1)` matrix is
/// always built, the last ordinary matrix is built before the intermediate
/// loop when `nleg > 2`, and intermediate matrices use FEFF `ilegp=2:nsc-1`.
pub fn genfmt_scattering_matrix_plan(
    input: GenfmtScatteringMatrixPlanInput,
) -> Result<GenfmtScatteringMatrixPlan, GenfmtError> {
    if input.leg_count < 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "leg_count",
            value: input.leg_count,
        });
    }
    validate_positive_limit("full_lambda_count", input.full_lambda_count)?;
    validate_positive_limit("initial_lambda_count", input.initial_lambda_count)?;
    if input.initial_lambda_count > input.full_lambda_count {
        return Err(GenfmtError::TableAxisTooShort {
            table: "scattering_matrix_plan",
            axis: "full_lambda",
            length: input.full_lambda_count,
            required: input.initial_lambda_count,
        });
    }

    let scattering_count = input.leg_count - 1;
    let mut tasks = Vec::with_capacity(scattering_count);
    tasks.push(GenfmtScatteringMatrixTask {
        role: GenfmtScatteringMatrixRole::First,
        current_leg_index: 1,
        previous_leg_index: 0,
        matrix_slot_index: 0,
        left_lambda_count: input.full_lambda_count,
        right_lambda_count: input.initial_lambda_count,
    });

    if input.leg_count > 2 {
        tasks.push(GenfmtScatteringMatrixTask {
            role: GenfmtScatteringMatrixRole::LastOrdinary,
            current_leg_index: input.leg_count - 1,
            previous_leg_index: input.leg_count - 2,
            matrix_slot_index: input.leg_count - 2,
            left_lambda_count: input.initial_lambda_count,
            right_lambda_count: input.full_lambda_count,
        });
    }

    for previous_leg_1based in 2..scattering_count {
        tasks.push(GenfmtScatteringMatrixTask {
            role: GenfmtScatteringMatrixRole::Intermediate,
            current_leg_index: previous_leg_1based,
            previous_leg_index: previous_leg_1based - 1,
            matrix_slot_index: previous_leg_1based - 1,
            left_lambda_count: input.full_lambda_count,
            right_lambda_count: input.full_lambda_count,
        });
    }

    Ok(GenfmtScatteringMatrixPlan {
        scattering_count,
        tasks,
    })
}

/// Build one FEFF GENFMT `nstar.dat` row for a path.
///
/// This ports the `wnstar` path block in `genfmtsub.f90` and `genfmtjas.f90`:
/// vectors are measured from the absorber (`rat(:,0)`), the secondary
/// polarization is `xivec cross evec` only for nonzero ellipticity, and the
/// path degeneracy is rounded with FEFF `nint` before evaluating `xstar`.
pub fn genfmt_nstar_row(input: GenfmtNStarInput<'_>) -> Result<GenfmtNStarRow, GenfmtError> {
    validate_positive_limit("path_number", input.path_number)?;
    let leg_count = input.positions.shape()[0];
    let coordinate_columns = input.positions.shape()[1];
    if leg_count < 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "leg_count",
            value: leg_count,
        });
    }
    if coordinate_columns != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: coordinate_columns,
        });
    }

    let absorber = path_position(input.positions, leg_count - 1)?;
    let first_leg = vector_between(path_position(input.positions, 0)?, absorber);
    let last_leg = vector_between(path_position(input.positions, leg_count - 2)?, absorber);
    let secondary_polarization = if input.ellipticity != 0.0 {
        cross(input.ellipticity_vector, input.primary_polarization)
    } else {
        [0.0; 3]
    };
    let nstar = xstar(XStarInput {
        primary_polarization: input.primary_polarization,
        secondary_polarization,
        first_leg,
        last_leg,
        degeneracy: input.degeneracy.round(),
        initial_l: input.initial_l,
        ellipticity: input.ellipticity,
    })?;

    Ok(GenfmtNStarRow {
        path_number: input.path_number,
        nstar,
    })
}

/// Build FEFF GENFMT `nstar.dat` rows in path traversal order.
///
/// When `wnstar` is enabled, both GENFMT drivers write the polarization header
/// once and then write one `npath, n*` row for every examined path, regardless
/// of later path retention. Rust keeps that file-level shape here while leaving
/// text rendering to the IO crate.
pub fn genfmt_nstar_rows(input: GenfmtNStarRowsInput<'_>) -> Result<GenfmtNStarRows, GenfmtError> {
    validate_fixed_vector("primary_polarization", input.primary_polarization)?;
    validate_fixed_vector("ellipticity_vector", input.ellipticity_vector)?;
    validate_finite_scalar("ellipticity", input.ellipticity)?;

    let mut rows = Vec::with_capacity(input.path_inputs.len());
    for (path_index, path) in input.path_inputs.iter().enumerate() {
        let path_number = path_index
            .checked_add(1)
            .ok_or(GenfmtError::IntegerOverflow {
                field: "path_number",
                value: path_index,
            })?;
        rows.push(genfmt_nstar_row(GenfmtNStarInput {
            path_number,
            positions: path.positions,
            primary_polarization: input.primary_polarization,
            ellipticity_vector: input.ellipticity_vector,
            degeneracy: path.degeneracy,
            initial_l: input.initial_l,
            ellipticity: input.ellipticity,
        })?);
    }

    Ok(GenfmtNStarRows {
        primary_polarization: input.primary_polarization,
        rows,
    })
}

fn genfmt_nstar_rows_from_ordinary_driver_inputs(
    input: GenfmtNStarDriverInput,
    path_inputs: &[GenfmtOrdinaryPathEvaluationFromDriverSetupInput<'_>],
) -> Result<GenfmtNStarRows, GenfmtError> {
    let path_inputs = path_inputs
        .iter()
        .map(|path| GenfmtNStarPathInput {
            positions: path.positions,
            degeneracy: path.degeneracy,
        })
        .collect::<Vec<_>>();
    genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization: input.primary_polarization,
        ellipticity_vector: input.ellipticity_vector,
        initial_l: input.initial_l,
        ellipticity: input.ellipticity,
        path_inputs: &path_inputs,
    })
}

fn genfmt_nstar_rows_from_jas_driver_inputs(
    input: GenfmtNStarDriverInput,
    path_inputs: &[GenfmtJasPathEvaluationFromDriverSetupInput<'_>],
) -> Result<GenfmtNStarRows, GenfmtError> {
    let path_inputs = path_inputs
        .iter()
        .map(|path| GenfmtNStarPathInput {
            positions: path.positions,
            degeneracy: path.degeneracy,
        })
        .collect::<Vec<_>>();
    genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization: input.primary_polarization,
        ellipticity_vector: input.ellipticity_vector,
        initial_l: input.initial_l,
        ellipticity: input.ellipticity,
        path_inputs: &path_inputs,
    })
}

/// Build FEFF GENFMT's alternating `pmati` path F-matrix product.
///
/// This ports the shared `pmati` product loops used by `genfmtsub.f90` and
/// `genfmtjas.f90`: the first matrix is copied into FEFF's two-slot product
/// work array, then each intermediate matrix is multiplied from the left.
pub fn genfmt_path_matrix_product(
    input: GenfmtPathMatrixProductInput<'_>,
) -> Result<GenfmtPathMatrixProduct, GenfmtError> {
    validate_path_matrix_product_input(input)?;

    let mut product =
        Array2::<Complex>::zeros((input.full_lambda_count, input.initial_lambda_count).f());
    for lambda in 0..input.full_lambda_count {
        for initial_lambda in 0..input.initial_lambda_count {
            product[(lambda, initial_lambda)] = table_complex_entry(
                input.first_scattering,
                "first_scattering",
                lambda,
                initial_lambda,
            )?;
        }
    }

    for intermediate_leg in 0..input.intermediate_scattering.shape()[0] {
        let mut next =
            Array2::<Complex>::zeros((input.full_lambda_count, input.initial_lambda_count).f());
        for lambda in 0..input.full_lambda_count {
            for initial_lambda in 0..input.initial_lambda_count {
                let mut value = Complex::new(0.0, 0.0);
                for inner_lambda in 0..input.full_lambda_count {
                    value += tensor3_complex_entry(
                        input.intermediate_scattering,
                        "intermediate_scattering",
                        intermediate_leg,
                        lambda,
                        inner_lambda,
                    )? * product[(inner_lambda, initial_lambda)];
                }
                next[(lambda, initial_lambda)] = value;
            }
        }
        product = next;
    }

    Ok(GenfmtPathMatrixProduct {
        product_matrix: product,
    })
}

/// Contract FEFF GENFMT path F matrices into the final path trace.
///
/// This ports the `pmati` product loops and the final `ptrac` contraction in
/// `GENFMT/genfmtsub.f90`. The termination matrix is traced against the
/// transposed initial-lambda block of the product.
pub fn genfmt_path_matrix_trace(
    input: GenfmtPathMatrixTraceInput<'_>,
) -> Result<GenfmtPathMatrixTrace, GenfmtError> {
    validate_path_matrix_trace_input(input)?;

    let product = genfmt_path_matrix_product(GenfmtPathMatrixProductInput {
        first_scattering: input.first_scattering,
        intermediate_scattering: input.intermediate_scattering,
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
    })?;
    let trace = genfmt_termination_path_product_trace(
        input.termination_matrix,
        product.product_matrix.view(),
        input.initial_lambda_count,
    )?;

    Ok(GenfmtPathMatrixTrace {
        product_matrix: product.product_matrix,
        trace,
    })
}

fn genfmt_termination_path_product_trace(
    termination_matrix: ArrayView2<'_, Complex>,
    product_matrix: ArrayView2<'_, Complex>,
    initial_lambda_count: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(
        "termination_matrix",
        "lambda",
        termination_matrix.shape()[0],
        initial_lambda_count,
    )?;
    ensure_axis_len(
        "termination_matrix",
        "initial_lambda",
        termination_matrix.shape()[1],
        initial_lambda_count,
    )?;
    ensure_axis_len(
        "path_product",
        "lambda",
        product_matrix.shape()[0],
        initial_lambda_count,
    )?;
    ensure_axis_len(
        "path_product",
        "initial_lambda",
        product_matrix.shape()[1],
        initial_lambda_count,
    )?;

    let mut trace = Complex::new(0.0, 0.0);
    for lambda in 0..initial_lambda_count {
        for initial_lambda in 0..initial_lambda_count {
            trace +=
                table_complex_entry(
                    termination_matrix,
                    "termination_matrix",
                    lambda,
                    initial_lambda,
                )? * table_complex_entry(product_matrix, "path_product", initial_lambda, lambda)?;
        }
    }

    Ok(trace)
}

/// Build FEFF GENFMT scattering matrices and the shared `pmati` product.
///
/// This composes the `fmtrxi` call order and product setup common to
/// `genfmtsub.f90` and `genfmtjas.f90`, stopping before the final ordinary or
/// JAS termination contraction.
pub fn genfmt_scattering_path_product(
    input: GenfmtScatteringPathProductInput<'_>,
) -> Result<GenfmtScatteringPathProduct, GenfmtError> {
    let leg_count = input.path_potential_indices.len();
    if leg_count < 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "path_potential_indices",
            value: leg_count,
        });
    }
    ensure_axis_len("eta_angles", "leg", input.eta_angles.len(), leg_count + 1)?;
    ensure_axis_len(
        "curved_wave_polynomials",
        "leg",
        input.curved_wave_polynomials.shape()[2],
        leg_count,
    )?;
    ensure_axis_len("rotations", "leg", input.rotations.shape()[0], leg_count)?;

    let plan = genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
        leg_count,
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
    })?;
    let mut scattering_slots: Vec<Option<Array2<Complex>>> = vec![None; plan.scattering_count];

    for task in plan.tasks {
        let previous_potential_index =
            genfmt_path_potential_index(input.path_potential_indices, task.previous_leg_index)?;
        ensure_axis_len(
            "angular_limits",
            "potential",
            input.angular_limits.len(),
            previous_potential_index + 1,
        )?;
        ensure_axis_len(
            "phase_shifts",
            "potential",
            input.phase_shifts.shape()[1],
            previous_potential_index + 1,
        )?;

        let angular_limit = input.angular_limits[previous_potential_index];
        let phase_shifts = genfmt_phase_shifts_for_potential(
            input.phase_shifts,
            input.signed_angular_offset,
            previous_potential_index,
            angular_limit,
        )?;
        let first_leg_polynomials = input
            .curved_wave_polynomials
            .index_axis(Axis(2), task.current_leg_index);
        let second_leg_polynomials = input
            .curved_wave_polynomials
            .index_axis(Axis(2), task.previous_leg_index);
        let rotation = input.rotations.index_axis(Axis(0), task.previous_leg_index);
        // `eta_angles` retains FEFF's padded `eta(0:nleg+1)` layout. The
        // task indices are zero-based, while `fmtrxi` reads `eta(ileg)`.
        let eta = finite_vector_value(input.eta_angles, "eta_angles", task.current_leg_index + 1)?;

        let matrix = scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
            m_indices: input.m_indices,
            n_indices: input.n_indices,
            left_lambda_count: task.left_lambda_count,
            right_lambda_count: task.right_lambda_count,
            phase_shifts: phase_shifts.view(),
            angular_limit,
            first_leg_polynomials,
            second_leg_polynomials,
            rotation,
            rotation_magnetic_offset: input.rotation_magnetic_offset,
            xnlm: input.xnlm,
            eta,
        })?;
        scattering_slots[task.matrix_slot_index] = Some(matrix);
    }

    let mut scattering_matrices = Vec::with_capacity(plan.scattering_count);
    for (slot_index, matrix) in scattering_slots.into_iter().enumerate() {
        let Some(matrix) = matrix else {
            return Err(GenfmtError::TableAxisTooShort {
                table: "scattering_matrices",
                axis: "slot",
                length: slot_index,
                required: plan.scattering_count,
            });
        };
        scattering_matrices.push(matrix);
    }

    let first_scattering = &scattering_matrices[0];
    let mut intermediate_scattering = Array3::<Complex>::zeros(
        (
            plan.scattering_count.saturating_sub(1),
            input.full_lambda_count,
            input.full_lambda_count,
        )
            .f(),
    );
    for (slot_index, matrix) in scattering_matrices.iter().enumerate().skip(1) {
        for row in 0..matrix.shape()[0] {
            for column in 0..matrix.shape()[1] {
                intermediate_scattering[(slot_index - 1, row, column)] =
                    table_complex_entry(matrix.view(), "scattering_matrix", row, column)?;
            }
        }
    }

    let matrix_product = genfmt_path_matrix_product(GenfmtPathMatrixProductInput {
        first_scattering: first_scattering.view(),
        intermediate_scattering: intermediate_scattering.view(),
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
    })?;

    Ok(GenfmtScatteringPathProduct {
        scattering_matrices,
        matrix_product,
    })
}

/// Build the ordinary FEFF GENFMT path trace for one energy and spin channel.
///
/// This composes the `genfmtsub.f90` block that calls `fmtrxi` for
/// `f(2,1)`, the optional `f(N,N-1)` and intermediate scattering matrices,
/// calls `mmtrxi` for the terminating matrix, then contracts the product into
/// `ptrac`.
pub fn genfmt_ordinary_path_trace(
    input: GenfmtOrdinaryPathTraceInput<'_>,
) -> Result<GenfmtOrdinaryPathTrace, GenfmtError> {
    let leg_count = input.path_potential_indices.len();
    let scattering_product = genfmt_scattering_path_product(GenfmtScatteringPathProductInput {
        m_indices: input.m_indices,
        n_indices: input.n_indices,
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits,
        phase_shifts: input.phase_shifts,
        signed_angular_offset: input.signed_angular_offset,
        curved_wave_polynomials: input.curved_wave_polynomials,
        rotations: input.rotations,
        rotation_magnetic_offset: input.rotation_magnetic_offset,
        xnlm: input.xnlm,
        eta_angles: input.eta_angles,
    })?;

    let termination_matrix =
        polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
            m_indices: input.m_indices,
            n_indices: input.n_indices,
            lambda_count: input.initial_lambda_count,
            transition_angular_momenta: input.transition_angular_momenta,
            radial_factors: input.radial_factors,
            transition_matrix: input.transition_matrix,
            transition_magnetic_offset: input.transition_magnetic_offset,
            first_leg_polynomials: input.curved_wave_polynomials.index_axis(Axis(2), 0),
            second_leg_polynomials: input
                .curved_wave_polynomials
                .index_axis(Axis(2), leg_count - 1),
            xnlm: input.xnlm,
            // FEFF calls `mmtrxi(..., ileg=1, ...)`.
            eta: finite_vector_value(input.eta_angles, "eta_angles", 1)?,
        })?;

    let product_matrix = scattering_product.matrix_product.product_matrix;
    let trace = genfmt_termination_path_product_trace(
        termination_matrix.view(),
        product_matrix.view(),
        input.initial_lambda_count,
    )?;
    let matrix_trace = GenfmtPathMatrixTrace {
        product_matrix,
        trace,
    };

    Ok(GenfmtOrdinaryPathTrace {
        scattering_matrices: scattering_product.scattering_matrices,
        termination_matrix,
        matrix_trace,
    })
}

/// Build one ordinary FEFF GENFMT energy/spin path contribution.
///
/// This composes the `genfmtsub.f90` energy-loop block from `rho`/`reff`
/// setup through `sclmz`, `fmtrxi`, `mmtrxi`, `cfac`, and the `cchi(ie)`
/// accumulation. The FEFF zero-momentum branch is preserved by returning only
/// geometry when `abs(ck(ie)) <= eps`.
pub fn genfmt_ordinary_path_energy_point(
    input: GenfmtOrdinaryPathEnergyPointInput<'_>,
) -> Result<GenfmtOrdinaryPathEnergyPoint, GenfmtError> {
    let required_energy_count =
        input
            .energy_index
            .checked_add(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "energy_index",
                value: input.energy_index,
            })?;
    ensure_axis_len(
        "phase_shifts",
        "energy",
        input.phase_shifts.shape()[0],
        required_energy_count,
    )?;
    ensure_axis_len(
        "radial_factors",
        "energy",
        input.radial_factors.shape()[0],
        required_energy_count,
    )?;

    let geometry = genfmt_path_geometry(GenfmtPathGeometryInput {
        leg_lengths: input.leg_lengths,
        complex_momentum: input.complex_momentum,
        momentum_zero_epsilon: input.momentum_zero_epsilon,
    })?;
    if !geometry.active {
        return Ok(GenfmtOrdinaryPathEnergyPoint {
            geometry,
            leg_limits: None,
            curved_wave_polynomials: None,
            path_trace: None,
            path_factor: None,
            signal: None,
        });
    }

    let leg_limits = genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits,
        energy_index: input.energy_index,
        max_m_plus_one: input.max_m_plus_one,
        max_n: input.max_n,
    })?;
    let curved_wave_polynomials =
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: geometry.leg_rhos.view(),
            leg_limits: &leg_limits.limits,
            mixed_order_capacity: leg_limits.mixed_order_capacity,
        })?;
    let path_trace = genfmt_ordinary_path_trace(GenfmtOrdinaryPathTraceInput {
        m_indices: input.m_indices,
        n_indices: input.n_indices,
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits.index_axis(Axis(0), input.energy_index),
        phase_shifts: input.phase_shifts.index_axis(Axis(0), input.energy_index),
        signed_angular_offset: input.signed_angular_offset,
        curved_wave_polynomials: curved_wave_polynomials.tables.view(),
        rotations: input.rotations,
        rotation_magnetic_offset: input.rotation_magnetic_offset,
        xnlm: input.xnlm,
        eta_angles: input.eta_angles,
        transition_angular_momenta: input.transition_angular_momenta,
        radial_factors: input.radial_factors.index_axis(Axis(0), input.energy_index),
        transition_matrix: input.transition_matrix,
        transition_magnetic_offset: input.transition_magnetic_offset,
    })?;
    let path_factor = genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
        leg_rhos: geometry.leg_rhos.view(),
        wave_number: input.wave_number,
        effective_path_length: geometry.effective_path_length,
    })?;
    let signal = genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
        accumulated_chi: input.accumulated_chi,
        path_trace: path_trace.matrix_trace.trace,
        path_factor: path_factor.factor,
        spin_channel_count: input.spin_channel_count,
        spin_index: input.spin_index,
    })?;

    Ok(GenfmtOrdinaryPathEnergyPoint {
        geometry,
        leg_limits: Some(leg_limits),
        curved_wave_polynomials: Some(curved_wave_polynomials),
        path_trace: Some(path_trace),
        path_factor: Some(path_factor),
        signal: Some(signal),
    })
}

/// Build an ordinary FEFF GENFMT energy-loop input from checked path setup.
///
/// This ports the path-local driver wiring before the ordinary spin loop:
/// lambda indices, leg lengths, rotations, and eta angles come from
/// `rdpath`/`setlam`/`rot3i`, while spin-resolved phase, momentum, radial, and
/// transition tables remain explicit because they are selected inside the
/// ordinary `genfmtsub.f90` spin loop.
pub fn genfmt_ordinary_path_energy_grid_from_setup<'a>(
    input: GenfmtOrdinaryPathEnergyGridFromSetupInput<'a>,
) -> Result<GenfmtOrdinaryPathEnergyGridInput<'a>, GenfmtError> {
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    validate_positive_limit("complex_momenta", input.complex_momenta.shape()[0])?;
    let energy_count = input.complex_momenta.shape()[0];
    ensure_axis_len(
        "complex_momenta",
        "spin",
        input.complex_momenta.shape()[1],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "wave_numbers",
        "energy",
        input.wave_numbers.len(),
        energy_count,
    )?;
    ensure_axis_len(
        "angular_limits",
        "energy",
        input.angular_limits.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_phase_shifts",
        "energy",
        input.spin_phase_shifts.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_phase_shifts",
        "spin",
        input.spin_phase_shifts.shape()[2],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "spin_radial_factors",
        "energy",
        input.spin_radial_factors.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_radial_factors",
        "spin",
        input.spin_radial_factors.shape()[2],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "transition_matrices",
        "spin",
        input.transition_matrices.shape()[0],
        input.spin_channel_count,
    )?;
    let transition_count = input.transition_angular_momenta.len();
    validate_positive_limit("transition_angular_momenta", transition_count)?;
    ensure_axis_len(
        "spin_radial_factors",
        "transition",
        input.spin_radial_factors.shape()[1],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrices",
        "transition1",
        input.transition_matrices.shape()[2],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrices",
        "transition2",
        input.transition_matrices.shape()[4],
        transition_count,
    )?;

    let full_lambda_count = input.path_setup.lambda.m_indices.len();
    let initial_lambda_count = input.path_setup.lambda.initial_l_prefix_len;
    validate_positive_limit("full_lambda_count", full_lambda_count)?;
    validate_positive_limit("initial_lambda_count", initial_lambda_count)?;
    ensure_axis_len(
        "n_indices",
        "lambda",
        input.path_setup.lambda.n_indices.len(),
        full_lambda_count,
    )?;
    ensure_axis_len(
        "m_indices",
        "initial_lambda",
        full_lambda_count,
        initial_lambda_count,
    )?;

    let leg_count = input.path_setup.angles.leg_lengths.len();
    validate_positive_limit("leg_lengths", leg_count)?;
    ensure_axis_len(
        "path_potential_indices",
        "leg",
        input.path_potential_indices.len(),
        leg_count,
    )?;
    ensure_axis_len(
        "rotations",
        "leg",
        input.path_setup.rotations.rotations.shape()[0],
        leg_count,
    )?;
    validate_positive_limit("xnlm_rows", input.xnlm.shape()[0])?;
    validate_positive_limit("xnlm_columns", input.xnlm.shape()[1])?;

    Ok(GenfmtOrdinaryPathEnergyGridInput {
        m_indices: input.path_setup.lambda.m_indices.view(),
        n_indices: input.path_setup.lambda.n_indices.view(),
        full_lambda_count,
        initial_lambda_count,
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits,
        spin_phase_shifts: input.spin_phase_shifts,
        signed_angular_offset: input.signed_angular_offset,
        leg_lengths: input.path_setup.angles.leg_lengths.view(),
        complex_momenta: input.complex_momenta,
        wave_numbers: input.wave_numbers,
        momentum_zero_epsilon: input.momentum_zero_epsilon,
        max_m_plus_one: input.path_setup.lambda.max_m_plus_one,
        max_n: input.path_setup.lambda.max_n,
        rotations: input.path_setup.rotations.rotations.view(),
        rotation_magnetic_offset: input.path_setup.rotations.rotation_magnetic_offset,
        xnlm: input.xnlm,
        eta_angles: input.path_setup.angles.eta_values.view(),
        transition_angular_momenta: input.transition_angular_momenta,
        spin_radial_factors: input.spin_radial_factors,
        transition_matrices: input.transition_matrices,
        transition_magnetic_offset: input.transition_magnetic_offset,
        spin_channel_count: input.spin_channel_count,
    })
}

/// Build an ordinary FEFF GENFMT energy-loop input from checked driver/path setup.
///
/// This is the driver-level counterpart to
/// [`genfmt_ordinary_path_energy_grid_from_setup`]: path-local tables come from
/// `rdpath`/`setlam`/`rot3i`, while ordinary driver setup supplies the
/// spin-resolved `ck(ie,is)` grid and shared `xk(ie)` values.
pub fn genfmt_ordinary_path_energy_grid_from_driver_setup<'a>(
    input: GenfmtOrdinaryPathEnergyGridFromDriverSetupInput<'a>,
) -> Result<GenfmtOrdinaryPathEnergyGridInput<'a>, GenfmtError> {
    genfmt_ordinary_path_energy_grid_from_setup(GenfmtOrdinaryPathEnergyGridFromSetupInput {
        path_setup: input.path_setup,
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits,
        spin_phase_shifts: input.spin_phase_shifts,
        signed_angular_offset: input.signed_angular_offset,
        complex_momenta: input.driver_setup.spin_momentum_grid.complex_momenta.view(),
        wave_numbers: input.driver_setup.spin_momentum_grid.wave_numbers.view(),
        momentum_zero_epsilon: input.momentum_zero_epsilon,
        xnlm: input.xnlm,
        transition_angular_momenta: input.transition_angular_momenta,
        spin_radial_factors: input.spin_radial_factors,
        transition_matrices: input.transition_matrices,
        transition_magnetic_offset: input.transition_magnetic_offset,
        spin_channel_count: input.driver_setup.spin_channel_count,
    })
}

/// Build ordinary FEFF GENFMT path contributions over spin and energy.
///
/// This composes the `genfmtsub.f90` spin loop and the nested big energy loop:
/// each spin selects its own `ph`, `rkk`, `bmati`, and `ck`, then each active
/// energy evaluates `sclmz`, `fmtrxi`, `mmtrxi`, `cfac`, and accumulates
/// `cchi(ie)`. The first spin is negated in the two-spin branch exactly as in
/// FEFF.
pub fn genfmt_ordinary_path_energy_grid(
    input: GenfmtOrdinaryPathEnergyGridInput<'_>,
) -> Result<GenfmtOrdinaryPathEnergyGrid, GenfmtError> {
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    validate_positive_limit("complex_momenta", input.complex_momenta.shape()[0])?;
    let energy_count = input.complex_momenta.shape()[0];
    ensure_axis_len(
        "complex_momenta",
        "spin",
        input.complex_momenta.shape()[1],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "wave_numbers",
        "energy",
        input.wave_numbers.len(),
        energy_count,
    )?;
    ensure_axis_len(
        "angular_limits",
        "energy",
        input.angular_limits.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_phase_shifts",
        "energy",
        input.spin_phase_shifts.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_phase_shifts",
        "spin",
        input.spin_phase_shifts.shape()[2],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "spin_radial_factors",
        "energy",
        input.spin_radial_factors.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_radial_factors",
        "spin",
        input.spin_radial_factors.shape()[2],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "transition_matrices",
        "spin",
        input.transition_matrices.shape()[0],
        input.spin_channel_count,
    )?;

    let mut active = Array2::<bool>::from_elem((input.spin_channel_count, energy_count).f(), false);
    let mut path_traces = Array2::<Complex>::zeros((input.spin_channel_count, energy_count).f());
    let mut path_factors = Array2::<Complex>::zeros((input.spin_channel_count, energy_count).f());
    let mut contributions = Array2::<Complex>::zeros((input.spin_channel_count, energy_count).f());
    let mut chi = Array1::<Complex>::zeros(energy_count);

    for spin in 0..input.spin_channel_count {
        let phase_shifts = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: input.spin_phase_shifts,
            angular_limits: input.angular_limits,
            signed_angular_offset: input.signed_angular_offset,
            spin_channel_count: input.spin_channel_count,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: spin },
        })?;
        let radial_factors = genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: input.spin_radial_factors,
            spin_channel_count: input.spin_channel_count,
            spin_index: spin,
        })?;
        let transition_matrix = input.transition_matrices.index_axis(Axis(0), spin);

        for energy in 0..energy_count {
            let energy_point =
                genfmt_ordinary_path_energy_point(GenfmtOrdinaryPathEnergyPointInput {
                    m_indices: input.m_indices,
                    n_indices: input.n_indices,
                    full_lambda_count: input.full_lambda_count,
                    initial_lambda_count: input.initial_lambda_count,
                    energy_index: energy,
                    path_potential_indices: input.path_potential_indices,
                    angular_limits: input.angular_limits,
                    phase_shifts: phase_shifts.phase_shifts.view(),
                    signed_angular_offset: input.signed_angular_offset,
                    leg_lengths: input.leg_lengths,
                    complex_momentum: table_complex_entry(
                        input.complex_momenta,
                        "complex_momenta",
                        energy,
                        spin,
                    )?,
                    wave_number: finite_vector_value(input.wave_numbers, "wave_numbers", energy)?,
                    momentum_zero_epsilon: input.momentum_zero_epsilon,
                    max_m_plus_one: input.max_m_plus_one,
                    max_n: input.max_n,
                    rotations: input.rotations,
                    rotation_magnetic_offset: input.rotation_magnetic_offset,
                    xnlm: input.xnlm,
                    eta_angles: input.eta_angles,
                    transition_angular_momenta: input.transition_angular_momenta,
                    radial_factors: radial_factors.radial_factors.view(),
                    transition_matrix,
                    transition_magnetic_offset: input.transition_magnetic_offset,
                    accumulated_chi: chi[energy],
                    spin_channel_count: input.spin_channel_count,
                    spin_index: spin,
                })?;

            active[(spin, energy)] = energy_point.geometry.active;
            if let Some(path_trace) = energy_point.path_trace.as_ref() {
                path_traces[(spin, energy)] = path_trace.matrix_trace.trace;
            }
            if let Some(path_factor) = energy_point.path_factor.as_ref() {
                path_factors[(spin, energy)] = path_factor.factor;
            }
            if let Some(signal) = energy_point.signal {
                contributions[(spin, energy)] = signal.contribution;
                chi[energy] = signal.accumulated_chi;
            }
        }
    }

    Ok(GenfmtOrdinaryPathEnergyGrid {
        active,
        path_traces,
        path_factors,
        signals: GenfmtPathSignals { contributions, chi },
    })
}

/// Contract FEFF GENFMTJAS left/right amplitudes into the path trace.
///
/// This ports the `fmatl`/`fmatr` `ptrac` loop in `GENFMT/genfmtjas.f90`,
/// plus the optional `lgfmatl`/`lgfmatr` decomposition loop that feeds
/// `pgtrl(lg2,lg1,ie)`. The decomposition axes intentionally preserve FEFF's
/// output order: the first index is the left-side decomposition (`lg2`) and
/// the second index is the right-side decomposition (`lg1`).
pub fn genfmt_jas_left_right_path_trace(
    input: GenfmtJasLeftRightPathTraceInput<'_>,
) -> Result<GenfmtJasLeftRightPathTrace, GenfmtError> {
    let (mj_count, q_count, decomposition_count) = validate_jas_left_right_path_trace_input(input)?;

    let mut trace = Complex::new(0.0, 0.0);
    for lambda in 0..input.lambda_count {
        for initial_lambda in 0..input.lambda_count {
            let mut amplitude_sum = Complex::new(0.0, 0.0);
            for q in 0..q_count {
                for mj in 0..mj_count {
                    amplitude_sum += tensor3_complex_entry(
                        input.left_amplitudes,
                        "left_amplitudes",
                        mj,
                        q,
                        lambda,
                    )? * tensor3_complex_entry(
                        input.right_amplitudes,
                        "right_amplitudes",
                        mj,
                        q,
                        initial_lambda,
                    )?;
                }
            }
            trace += amplitude_sum
                * table_complex_entry(input.path_product, "path_product", initial_lambda, lambda)?;
        }
    }

    let decomposed_traces = match (
        input.decomposed_left_amplitudes,
        input.decomposed_right_amplitudes,
        decomposition_count,
    ) {
        (Some(left), Some(right), Some(count)) => {
            let mut traces = Array2::<Complex>::zeros((count, count).f());
            for right_l in 0..count {
                for left_l in 0..count {
                    let mut decomposed_trace = Complex::new(0.0, 0.0);
                    for lambda in 0..input.lambda_count {
                        for initial_lambda in 0..input.lambda_count {
                            let mut amplitude_sum = Complex::new(0.0, 0.0);
                            for q in 0..q_count {
                                for mj in 0..mj_count {
                                    amplitude_sum += tensor4_complex_entry(
                                        right,
                                        "decomposed_right_amplitudes",
                                        mj,
                                        q,
                                        right_l,
                                        lambda,
                                    )? * tensor4_complex_entry(
                                        left,
                                        "decomposed_left_amplitudes",
                                        mj,
                                        q,
                                        left_l,
                                        initial_lambda,
                                    )?;
                                }
                            }
                            decomposed_trace += amplitude_sum
                                * table_complex_entry(
                                    input.path_product,
                                    "path_product",
                                    initial_lambda,
                                    lambda,
                                )?;
                        }
                    }
                    traces[(left_l, right_l)] = decomposed_trace;
                }
            }
            Some(traces)
        }
        _ => None,
    };

    Ok(GenfmtJasLeftRightPathTrace {
        trace,
        decomposed_traces,
    })
}

/// Contract FEFF GENFMTJAS spherical amplitudes into the path trace.
///
/// This ports the spherical-averaging `fmats` branch in `genfmtjas.f90`. When
/// angular decomposition is active, FEFF only fills the diagonal
/// `pgtrl(ll,ll,ie)` channels; off-diagonal entries are returned as zero.
pub fn genfmt_jas_spherical_path_trace(
    input: GenfmtJasSphericalPathTraceInput<'_>,
) -> Result<GenfmtJasSphericalPathTrace, GenfmtError> {
    let (mj_count, _) = validate_jas_spherical_path_trace_input(input)?;

    let mut trace = Complex::new(0.0, 0.0);
    for lambda in 0..input.lambda_count {
        for initial_lambda in 0..input.lambda_count {
            let mut amplitude_sum = Complex::new(0.0, 0.0);
            for spin in 0..2 {
                for mj in 0..mj_count {
                    amplitude_sum += tensor4_complex_entry(
                        input.amplitudes,
                        "amplitudes",
                        mj,
                        spin,
                        initial_lambda,
                        lambda,
                    )?;
                }
            }
            trace += amplitude_sum
                * table_complex_entry(input.path_product, "path_product", initial_lambda, lambda)?;
        }
    }

    let decomposed_traces = if let Some(decomposed) = input.decomposed_amplitudes {
        let count = decomposed.shape()[2];
        let mut traces = Array2::<Complex>::zeros((count, count).f());
        for decomposition_l in 0..count {
            let mut decomposed_trace = Complex::new(0.0, 0.0);
            for lambda in 0..input.lambda_count {
                for initial_lambda in 0..input.lambda_count {
                    let mut amplitude_sum = Complex::new(0.0, 0.0);
                    for spin in 0..2 {
                        for mj in 0..mj_count {
                            amplitude_sum += tensor5_complex_entry(
                                decomposed,
                                "decomposed_amplitudes",
                                mj,
                                spin,
                                decomposition_l,
                                initial_lambda,
                                lambda,
                            )?;
                        }
                    }
                    decomposed_trace += amplitude_sum
                        * table_complex_entry(
                            input.path_product,
                            "path_product",
                            initial_lambda,
                            lambda,
                        )?;
                }
            }
            traces[(decomposition_l, decomposition_l)] = decomposed_trace;
        }
        Some(traces)
    } else {
        None
    };

    Ok(GenfmtJasSphericalPathTrace {
        trace,
        decomposed_traces,
    })
}

/// Build the GENFMTJAS path trace for one energy point.
///
/// This composes the `genfmtjas.f90` branch that chooses either `mmtrxijas`
/// left/right amplitudes or `mmtrxijas0` spherical amplitudes, then contracts
/// those amplitudes against the already-built scattering path product.
pub fn genfmt_jas_path_trace(
    input: GenfmtJasPathTraceInput<'_>,
) -> Result<GenfmtJasPathTrace, GenfmtError> {
    match input {
        GenfmtJasPathTraceInput::LeftRight {
            path_product,
            amplitude_input,
        } => {
            let amplitudes = jas_left_right_amplitude_matrices(amplitude_input)?;
            let trace = genfmt_jas_left_right_path_trace(GenfmtJasLeftRightPathTraceInput {
                path_product,
                left_amplitudes: amplitudes.left_amplitudes.view(),
                right_amplitudes: amplitudes.right_amplitudes.view(),
                lambda_count: amplitude_input.lambda_count,
                decomposed_left_amplitudes: amplitudes
                    .decomposed_left_amplitudes
                    .as_ref()
                    .map(|table| table.view()),
                decomposed_right_amplitudes: amplitudes
                    .decomposed_right_amplitudes
                    .as_ref()
                    .map(|table| table.view()),
            })?;

            Ok(GenfmtJasPathTrace::LeftRight { amplitudes, trace })
        }
        GenfmtJasPathTraceInput::Spherical {
            path_product,
            amplitude_input,
        } => {
            let amplitudes = jas_scattering_amplitude_matrices(amplitude_input)?;
            let trace = genfmt_jas_spherical_path_trace(GenfmtJasSphericalPathTraceInput {
                path_product,
                amplitudes: amplitudes.amplitudes.view(),
                lambda_count: amplitude_input.lambda_count,
                decomposed_amplitudes: amplitudes
                    .decomposed_amplitudes
                    .as_ref()
                    .map(|table| table.view()),
            })?;

            Ok(GenfmtJasPathTrace::Spherical { amplitudes, trace })
        }
    }
}

/// Build one FEFF GENFMTJAS energy-point contribution.
///
/// This composes the `genfmtjas.f90` energy-loop block from `rho`/`reff`
/// setup through `sclmz`, shared `fmtrxi`/`pmati`, the selected JAS
/// termination branch, `cfac`, and total/decomposed `cchi` output. The
/// zero-momentum branch is preserved by returning only geometry when
/// `abs(ck(ie)) <= eps`.
pub fn genfmt_jas_path_energy_point(
    input: GenfmtJasPathEnergyPointInput<'_>,
) -> Result<GenfmtJasPathEnergyPoint, GenfmtError> {
    let required_energy_count =
        input
            .energy_index
            .checked_add(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "energy_index",
                value: input.energy_index,
            })?;
    ensure_axis_len(
        "phase_shifts",
        "energy",
        input.phase_shifts.shape()[0],
        required_energy_count,
    )?;

    let geometry = genfmt_path_geometry(GenfmtPathGeometryInput {
        leg_lengths: input.leg_lengths,
        complex_momentum: input.complex_momentum,
        momentum_zero_epsilon: input.momentum_zero_epsilon,
    })?;
    if !geometry.active {
        return Ok(GenfmtJasPathEnergyPoint {
            geometry,
            leg_limits: None,
            curved_wave_polynomials: None,
            scattering_product: None,
            path_trace: None,
            path_factor: None,
            signal: None,
        });
    }

    let leg_count = input.path_potential_indices.len();
    if leg_count < 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "path_potential_indices",
            value: leg_count,
        });
    }
    ensure_axis_len("eta_angles", "leg", input.eta_angles.len(), leg_count + 1)?;

    let leg_limits = genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits,
        energy_index: input.energy_index,
        max_m_plus_one: input.max_m_plus_one,
        max_n: input.max_n,
    })?;
    let curved_wave_polynomials =
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: geometry.leg_rhos.view(),
            leg_limits: &leg_limits.limits,
            mixed_order_capacity: leg_limits.mixed_order_capacity,
        })?;
    let scattering_product = genfmt_scattering_path_product(GenfmtScatteringPathProductInput {
        m_indices: input.m_indices,
        n_indices: input.n_indices,
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits.index_axis(Axis(0), input.energy_index),
        phase_shifts: input.phase_shifts.index_axis(Axis(0), input.energy_index),
        signed_angular_offset: input.signed_angular_offset,
        curved_wave_polynomials: curved_wave_polynomials.tables.view(),
        rotations: input.rotations,
        rotation_magnetic_offset: input.rotation_magnetic_offset,
        xnlm: input.xnlm,
        eta_angles: input.eta_angles,
    })?;
    let first_leg_polynomials = curved_wave_polynomials.tables.index_axis(Axis(2), 0);
    let second_leg_polynomials = curved_wave_polynomials
        .tables
        .index_axis(Axis(2), leg_count - 1);
    // FEFF termination routines are called with `ileg=1`.
    let eta = finite_vector_value(input.eta_angles, "eta_angles", 1)?;

    let path_trace = match input.branch {
        GenfmtJasPathEnergyBranchInput::LeftRight {
            transition_angular_momenta,
            radial_factors,
            q_weights,
            left_transition_matrix,
            right_transition_matrix,
            initial_j2,
            transition_magnetic_offset,
            max_angular_momentum,
            decomposition_l_max,
        } => genfmt_jas_path_trace(GenfmtJasPathTraceInput::LeftRight {
            path_product: scattering_product.matrix_product.product_matrix.view(),
            amplitude_input: JasLeftRightAmplitudeInput {
                m_indices: input.m_indices,
                n_indices: input.n_indices,
                lambda_count: input.initial_lambda_count,
                transition_angular_momenta,
                radial_factors,
                q_weights,
                left_transition_matrix,
                right_transition_matrix,
                initial_j2,
                transition_magnetic_offset,
                first_leg_polynomials,
                second_leg_polynomials,
                xnlm: input.xnlm,
                eta,
                max_angular_momentum,
                decomposition_l_max,
            },
        })?,
        GenfmtJasPathEnergyBranchInput::Spherical {
            transition_angular_momenta,
            radial_factors,
            q_weights,
            transition_matrix,
            initial_j2,
            transition_magnetic_offset,
            max_angular_momentum,
            decomposition_l_max,
        } => genfmt_jas_path_trace(GenfmtJasPathTraceInput::Spherical {
            path_product: scattering_product.matrix_product.product_matrix.view(),
            amplitude_input: JasScatteringAmplitudeInput {
                m_indices: input.m_indices,
                n_indices: input.n_indices,
                lambda_count: input.initial_lambda_count,
                transition_angular_momenta,
                radial_factors,
                q_weights,
                transition_matrix,
                initial_j2,
                transition_magnetic_offset,
                first_leg_polynomials,
                second_leg_polynomials,
                xnlm: input.xnlm,
                eta,
                max_angular_momentum,
                decomposition_l_max,
            },
        })?,
    };

    let path_factor = genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
        leg_rhos: geometry.leg_rhos.view(),
        wave_number: input.wave_number,
        effective_path_length: geometry.effective_path_length,
    })?;
    let (trace, decomposed_traces) = match &path_trace {
        GenfmtJasPathTrace::LeftRight { trace, .. } => (
            trace.trace,
            trace.decomposed_traces.as_ref().map(|table| table.view()),
        ),
        GenfmtJasPathTrace::Spherical { trace, .. } => (
            trace.trace,
            trace.decomposed_traces.as_ref().map(|table| table.view()),
        ),
    };
    let signal = genfmt_jas_path_signal(GenfmtJasPathSignalInput {
        path_trace: trace,
        path_factor: path_factor.factor,
        decomposed_traces,
    })?;

    Ok(GenfmtJasPathEnergyPoint {
        geometry,
        leg_limits: Some(leg_limits),
        curved_wave_polynomials: Some(curved_wave_polynomials),
        scattering_product: Some(scattering_product),
        path_trace: Some(path_trace),
        path_factor: Some(path_factor),
        signal: Some(signal),
    })
}

/// Build FEFF GENFMTJAS path contributions over the energy grid.
///
/// This ports the `genfmtjas.f90` big energy loop as a composition of the
/// per-energy worker: for each `ie` it builds `rho`, `clmi`, shared `pmati`,
/// the selected JAS termination trace, `cfac`, and the total/decomposed
/// `cchi` storage arrays. Zero-momentum energies stay zero.
pub fn genfmt_jas_path_energy_grid(
    input: GenfmtJasPathEnergyGridInput<'_>,
) -> Result<GenfmtJasPathEnergyGrid, GenfmtError> {
    validate_positive_limit("complex_momenta", input.complex_momenta.len())?;
    let energy_count = input.complex_momenta.len();
    ensure_axis_len(
        "wave_numbers",
        "energy",
        input.wave_numbers.len(),
        energy_count,
    )?;
    ensure_axis_len(
        "phase_shifts",
        "energy",
        input.phase_shifts.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "angular_limits",
        "energy",
        input.angular_limits.shape()[0],
        energy_count,
    )?;
    validate_jas_energy_grid_branch(input.branch, energy_count)?;

    let decomposition_count = jas_energy_grid_decomposition_count(input.branch)?;
    let mut active = Array1::<bool>::from_elem(energy_count, false);
    let mut path_traces = Array1::<Complex>::zeros(energy_count);
    let mut path_factors = Array1::<Complex>::zeros(energy_count);
    let mut decomposed_traces =
        decomposition_count.map(|count| Array3::<Complex>::zeros((count, count, energy_count).f()));

    for energy_index in 0..energy_count {
        let energy_point = genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
            m_indices: input.m_indices,
            n_indices: input.n_indices,
            full_lambda_count: input.full_lambda_count,
            initial_lambda_count: input.initial_lambda_count,
            energy_index,
            path_potential_indices: input.path_potential_indices,
            angular_limits: input.angular_limits,
            phase_shifts: input.phase_shifts,
            signed_angular_offset: input.signed_angular_offset,
            leg_lengths: input.leg_lengths,
            complex_momentum: complex_vector_entry(
                input.complex_momenta,
                "complex_momenta",
                energy_index,
            )?,
            wave_number: finite_vector_value(input.wave_numbers, "wave_numbers", energy_index)?,
            momentum_zero_epsilon: input.momentum_zero_epsilon,
            max_m_plus_one: input.max_m_plus_one,
            max_n: input.max_n,
            rotations: input.rotations,
            rotation_magnetic_offset: input.rotation_magnetic_offset,
            xnlm: input.xnlm,
            eta_angles: input.eta_angles,
            branch: jas_energy_point_branch(input.branch, energy_index)?,
        })?;

        active[energy_index] = energy_point.geometry.active;
        if let Some(path_factor) = energy_point.path_factor.as_ref() {
            path_factors[energy_index] = path_factor.factor;
        }
        if let Some(path_trace) = energy_point.path_trace.as_ref() {
            let (trace, decomposed) = jas_path_trace_components(path_trace);
            path_traces[energy_index] = trace;
            if let Some(output) = decomposed_traces.as_mut() {
                let Some(decomposed) = decomposed else {
                    return Err(GenfmtError::MismatchedJasFinalizationDecomposition);
                };
                ensure_axis_len(
                    "decomposed_traces",
                    "row",
                    decomposed.shape()[0],
                    output.shape()[0],
                )?;
                ensure_axis_len(
                    "decomposed_traces",
                    "column",
                    decomposed.shape()[1],
                    output.shape()[1],
                )?;
                for row in 0..output.shape()[0] {
                    for column in 0..output.shape()[1] {
                        output[(row, column, energy_index)] =
                            table_complex_entry(decomposed, "decomposed_traces", row, column)?;
                    }
                }
            }
        }
    }

    let signals = genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        decomposed_traces: decomposed_traces.as_ref().map(|traces| traces.view()),
    })?;

    Ok(GenfmtJasPathEnergyGrid {
        active,
        path_traces,
        path_factors,
        decomposed_traces,
        signals,
    })
}

fn validate_jas_energy_grid_branch(
    branch: GenfmtJasPathEnergyGridBranchInput<'_>,
    energy_count: usize,
) -> Result<(), GenfmtError> {
    match branch {
        GenfmtJasPathEnergyGridBranchInput::LeftRight { radial_factors, .. }
        | GenfmtJasPathEnergyGridBranchInput::Spherical { radial_factors, .. } => ensure_axis_len(
            "radial_factors",
            "energy",
            radial_factors.shape()[0],
            energy_count,
        ),
    }
}

fn jas_energy_grid_decomposition_count(
    branch: GenfmtJasPathEnergyGridBranchInput<'_>,
) -> Result<Option<usize>, GenfmtError> {
    let decomposition_l_max = match branch {
        GenfmtJasPathEnergyGridBranchInput::LeftRight {
            decomposition_l_max,
            ..
        }
        | GenfmtJasPathEnergyGridBranchInput::Spherical {
            decomposition_l_max,
            ..
        } => decomposition_l_max,
    };
    decomposition_l_max
        .map(|limit| checked_count("decomposition_l_max", limit))
        .transpose()
}

fn jas_energy_point_branch<'a>(
    branch: GenfmtJasPathEnergyGridBranchInput<'a>,
    energy_index: usize,
) -> Result<GenfmtJasPathEnergyBranchInput<'a>, GenfmtError> {
    match branch {
        GenfmtJasPathEnergyGridBranchInput::LeftRight {
            transition_angular_momenta,
            radial_factors,
            q_weights,
            left_transition_matrix,
            right_transition_matrix,
            initial_j2,
            transition_magnetic_offset,
            max_angular_momentum,
            decomposition_l_max,
        } => Ok(GenfmtJasPathEnergyBranchInput::LeftRight {
            transition_angular_momenta,
            radial_factors: radial_factors.index_axis_move(Axis(0), energy_index),
            q_weights,
            left_transition_matrix,
            right_transition_matrix,
            initial_j2,
            transition_magnetic_offset,
            max_angular_momentum,
            decomposition_l_max,
        }),
        GenfmtJasPathEnergyGridBranchInput::Spherical {
            transition_angular_momenta,
            radial_factors,
            q_weights,
            transition_matrix,
            initial_j2,
            transition_magnetic_offset,
            max_angular_momentum,
            decomposition_l_max,
        } => Ok(GenfmtJasPathEnergyBranchInput::Spherical {
            transition_angular_momenta,
            radial_factors: radial_factors.index_axis_move(Axis(0), energy_index),
            q_weights,
            transition_matrix,
            initial_j2,
            transition_magnetic_offset,
            max_angular_momentum,
            decomposition_l_max,
        }),
    }
}

fn jas_path_trace_components(
    path_trace: &GenfmtJasPathTrace,
) -> (Complex, Option<ArrayView2<'_, Complex>>) {
    match path_trace {
        GenfmtJasPathTrace::LeftRight { trace, .. } => (
            trace.trace,
            trace.decomposed_traces.as_ref().map(|table| table.view()),
        ),
        GenfmtJasPathTrace::Spherical { trace, .. } => (
            trace.trace,
            trace.decomposed_traces.as_ref().map(|table| table.view()),
        ),
    }
}

/// Select the active FEFF GENFMT spin-channel count from `ispin`.
///
/// This ports the top-level `nsp=1; if (ispin.eq.1) nsp=nspx` branch in
/// `GENFMT/genfmtsub.f90`. FEFF uses a single active channel for every selector
/// except `ispin == 1`.
pub fn genfmt_spin_channel_count(input: GenfmtSpinChannelCountInput) -> Result<usize, GenfmtError> {
    if input.available_spin_channels == 0 || input.available_spin_channels > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: input.available_spin_channels,
        });
    }

    if input.spin_selector == 1 {
        Ok(input.available_spin_channels)
    } else {
        Ok(1)
    }
}

/// Select the FEFF GENFMTJAS spin channel from `ispin`.
///
/// This ports the `genfmtjas.f90` setup branch `is=1; if (ispin.eq.1) is=nspx`.
/// Unlike ordinary GENFMT, JAS/NRIXS does not average two spin channels in the
/// header setup; it copies one selected spin slot into `eref`, `ph`, and `rkk`.
pub fn genfmt_jas_spin_selection(
    input: GenfmtJasSpinSelectionInput,
) -> Result<GenfmtJasSpinSelection, GenfmtError> {
    if input.available_spin_channels == 0 || input.available_spin_channels > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: input.available_spin_channels,
        });
    }

    let spin_index = if input.spin_selector == 1 {
        input.available_spin_channels - 1
    } else {
        0
    };

    Ok(GenfmtJasSpinSelection { spin_index })
}

/// Prepare FEFF GENFMT `eref(1:ne)` from spin-resolved `eref2`.
///
/// This ports both reference-energy setup branches in `GENFMT/genfmtsub.f90`:
/// the header path uses spin slot 1 for one active channel and averages the
/// first/last active slots for two active channels, while the per-spin path
/// loop copies the requested spin slot.
pub fn genfmt_spin_reference_energies(
    input: GenfmtSpinReferenceEnergyInput<'_>,
) -> Result<GenfmtSpinReferenceEnergies, GenfmtError> {
    validate_positive_limit("energy_count", input.spin_reference_energies.shape()[0])?;
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    ensure_axis_len(
        "spin_reference_energies",
        "spin",
        input.spin_reference_energies.shape()[1],
        input.spin_channel_count,
    )?;

    let energy_count = input.spin_reference_energies.shape()[0];
    let mut reference_energies = Array1::<Complex>::zeros(energy_count);
    match input.mode {
        GenfmtReferenceEnergyMode::Header if input.spin_channel_count == 1 => {
            for energy in 0..energy_count {
                reference_energies[energy] = table_complex_entry(
                    input.spin_reference_energies,
                    "spin_reference_energies",
                    energy,
                    0,
                )?;
            }
        }
        GenfmtReferenceEnergyMode::Header => {
            let last_spin = input.spin_channel_count - 1;
            for energy in 0..energy_count {
                let first = table_complex_entry(
                    input.spin_reference_energies,
                    "spin_reference_energies",
                    energy,
                    0,
                )?;
                let last = table_complex_entry(
                    input.spin_reference_energies,
                    "spin_reference_energies",
                    energy,
                    last_spin,
                )?;
                let average = (first + last) * 0.5;
                validate_finite_complex("spin_reference_energy_average", average)?;
                reference_energies[energy] = average;
            }
        }
        GenfmtReferenceEnergyMode::SpinChannel { spin_index } => {
            if spin_index >= input.spin_channel_count {
                return Err(GenfmtError::InvalidAngularLimit {
                    name: "spin_index",
                    value: spin_index,
                });
            }
            for energy in 0..energy_count {
                reference_energies[energy] = table_complex_entry(
                    input.spin_reference_energies,
                    "spin_reference_energies",
                    energy,
                    spin_index,
                )?;
            }
        }
    }

    Ok(GenfmtSpinReferenceEnergies { reference_energies })
}

/// Prepare FEFF GENFMT `ph(1:ne,-ltot:ltot,0:npot)` from spin-resolved `ph4`.
///
/// This ports the phase-shift copy/average branches in `GENFMT/genfmtsub.f90`.
/// Only entries within each FEFF `lmax(ie,iph)` range are populated; entries
/// outside that active range remain zero in the returned table.
pub fn genfmt_spin_phase_shifts(
    input: GenfmtSpinPhaseShiftInput<'_>,
) -> Result<GenfmtSpinPhaseShifts, GenfmtError> {
    let source_shape = input.spin_phase_shifts.shape();
    let energy_count = source_shape[0];
    let signed_l_count = source_shape[1];
    let potential_count = input.angular_limits.shape()[1];
    validate_positive_limit("energy_count", energy_count)?;
    validate_positive_limit("potential_count", potential_count)?;
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    ensure_axis_len(
        "angular_limits",
        "energy",
        input.angular_limits.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "spin_phase_shifts",
        "spin",
        source_shape[2],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "spin_phase_shifts",
        "potential",
        source_shape[3],
        potential_count,
    )?;

    let mut phase_shifts =
        Array3::<Complex>::zeros((energy_count, signed_l_count, potential_count).f());
    for energy in 0..energy_count {
        for potential in 0..potential_count {
            let angular_limit = input.angular_limits[(energy, potential)];
            if input.signed_angular_offset < angular_limit {
                return Err(GenfmtError::InvalidAngularLimit {
                    name: "signed_angular_offset",
                    value: input.signed_angular_offset,
                });
            }
            let required_signed_l_count = input
                .signed_angular_offset
                .checked_add(angular_limit)
                .and_then(|value| value.checked_add(1))
                .ok_or(GenfmtError::InvalidAngularLimit {
                    name: "angular_limit",
                    value: angular_limit,
                })?;
            ensure_axis_len(
                "spin_phase_shifts",
                "signed_angular_momentum",
                signed_l_count,
                required_signed_l_count,
            )?;

            let lower = -(angular_limit as isize);
            let upper = angular_limit as isize;
            for signed_l in lower..=upper {
                let signed_l_index = (input.signed_angular_offset as isize + signed_l) as usize;
                phase_shifts[(energy, signed_l_index, potential)] =
                    spin_phase_shift_entry(input, energy, signed_l_index, potential)?;
            }
        }
    }

    Ok(GenfmtSpinPhaseShifts { phase_shifts })
}

/// Select FEFF GENFMT central-atom phase shifts for the `feff.bin` header.
///
/// Both GENFMT drivers use `ll=linit+1`, flip the sign when `kinit < 0`, and
/// then write `ph(1:ne,ll,0)`. The Rust helper returns the selected signed
/// channel together with the copied energy vector.
pub fn genfmt_central_phase_shifts(
    input: GenfmtCentralPhaseShiftInput<'_>,
) -> Result<GenfmtCentralPhaseShifts, GenfmtError> {
    let shape = input.phase_shifts.shape();
    let energy_count = shape[0];
    let signed_l_count = shape[1];
    validate_positive_limit("energy_count", energy_count)?;
    ensure_axis_len("phase_shifts", "potential", shape[2], 1)?;
    if input.initial_kappa == 0 {
        return Err(GenfmtError::InvalidInitialKappa { kappa: 0 });
    }

    let header_channel =
        input
            .initial_orbital_l
            .checked_add(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "initial_orbital_l",
                value: input.initial_orbital_l,
            })?;
    let signed_channel = checked_i32("initial_orbital_l", header_channel)?;
    let signed_angular_momentum = if input.initial_kappa < 0 {
        -signed_channel
    } else {
        signed_channel
    };
    let signed_l_index = input.signed_angular_offset as isize + signed_angular_momentum as isize;
    if signed_l_index < 0 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "signed_angular_offset",
            value: input.signed_angular_offset,
        });
    }
    let signed_l_index = signed_l_index as usize;
    ensure_axis_len(
        "phase_shifts",
        "signed_angular_momentum",
        signed_l_count,
        signed_l_index + 1,
    )?;

    let mut phase_shifts = Array1::<Complex>::zeros(energy_count);
    for energy in 0..energy_count {
        phase_shifts[energy] = tensor3_complex_entry(
            input.phase_shifts,
            "phase_shifts",
            energy,
            signed_l_index,
            0,
        )?;
    }

    Ok(GenfmtCentralPhaseShifts {
        signed_angular_momentum,
        phase_shifts,
    })
}

/// Prepare FEFF GENFMT `rkk(1:ne,1:8)` for one active spin channel.
///
/// This ports `rkk(ie,kdif)=rkk2(ie,kdif,is)` from the `genfmtsub.f90` spin
/// loop. GENFMT does not average radial transition factors in the header
/// branch; callers request the active spin slot explicitly.
pub fn genfmt_spin_radial_factors(
    input: GenfmtSpinRadialFactorInput<'_>,
) -> Result<GenfmtSpinRadialFactors, GenfmtError> {
    let shape = input.spin_radial_factors.shape();
    let energy_count = shape[0];
    let transition_count = shape[1];
    validate_positive_limit("energy_count", energy_count)?;
    validate_positive_limit("transition_count", transition_count)?;
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    ensure_axis_len(
        "spin_radial_factors",
        "spin",
        shape[2],
        input.spin_channel_count,
    )?;
    if input.spin_index >= input.spin_channel_count {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "spin_index",
            value: input.spin_index,
        });
    }

    let mut radial_factors = Array2::<Complex>::zeros((energy_count, transition_count).f());
    for energy in 0..energy_count {
        for transition in 0..transition_count {
            radial_factors[(energy, transition)] = tensor3_complex_entry(
                input.spin_radial_factors,
                "spin_radial_factors",
                energy,
                transition,
                input.spin_index,
            )?;
        }
    }

    Ok(GenfmtSpinRadialFactors { radial_factors })
}

/// Prepare FEFF GENFMTJAS `rkk(1:ne,1:nq,1:indmax)` for the selected spin.
///
/// This ports `rkk(ie,iq,kdif)=rkk2(ie,iq,kdif,is)` from the
/// `genfmtjas.f90` setup block.
pub fn genfmt_jas_spin_radial_factors(
    input: GenfmtJasSpinRadialFactorInput<'_>,
) -> Result<GenfmtJasSpinRadialFactors, GenfmtError> {
    let shape = input.spin_radial_factors.shape();
    let energy_count = shape[0];
    let q_count = shape[1];
    let transition_count = shape[2];
    validate_positive_limit("energy_count", energy_count)?;
    validate_positive_limit("q_count", q_count)?;
    validate_positive_limit("transition_count", transition_count)?;
    let required_spin_count =
        input
            .spin_index
            .checked_add(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "spin_index",
                value: input.spin_index,
            })?;
    ensure_axis_len("spin_radial_factors", "spin", shape[3], required_spin_count)?;

    let mut radial_factors =
        Array3::<Complex>::zeros((energy_count, q_count, transition_count).f());
    for energy in 0..energy_count {
        for q_index in 0..q_count {
            for transition in 0..transition_count {
                radial_factors[(energy, q_index, transition)] = tensor4_complex_entry(
                    input.spin_radial_factors,
                    "spin_radial_factors",
                    energy,
                    q_index,
                    transition,
                    input.spin_index,
                )?;
            }
        }
    }

    Ok(GenfmtJasSpinRadialFactors { radial_factors })
}

/// Apply FEFF GENFMTJAS's effective `jinit` setup.
///
/// This ports the NRIXS branch in `GENFMT/regenf.f90`: when `elpty < 0`, FEFF
/// logs the spherical-averaging mode and sets `jinit=jmax` before calling
/// `genfmtjas`. Non-spherical JAS/NRIXS keeps the input `jinit`.
pub fn genfmt_jas_effective_initial_j(
    input: GenfmtJasEffectiveInitialJInput,
) -> Result<GenfmtJasEffectiveInitialJ, GenfmtError> {
    validate_finite_scalar("ellipticity", input.ellipticity)?;
    validate_nonnegative_doubled_j("jinit", input.initial_j2)?;
    validate_nonnegative_doubled_j("jmax", input.final_j2_max)?;

    let promoted_to_final_j2_max = input.ellipticity < 0.0;
    let initial_j2 = if promoted_to_final_j2_max {
        input.final_j2_max
    } else {
        input.initial_j2
    };

    Ok(GenfmtJasEffectiveInitialJ {
        initial_j2,
        promoted_to_final_j2_max,
    })
}

/// Check FEFF GENFMTJAS `phase.bin` transition-count consistency.
///
/// This ports the `if (indmaxt.ne.indmax) stop` guard in `genfmtjas.f90` after
/// transition matrices are prepared. Rust returns a structured error instead
/// of terminating the process.
pub fn genfmt_jas_transition_count(
    input: GenfmtJasTransitionCountInput,
) -> Result<GenfmtJasTransitionCount, GenfmtError> {
    if input.phase_transition_count != input.requested_transition_count {
        return Err(GenfmtError::MismatchedJasTransitionCount {
            phase_transition_count: input.phase_transition_count,
            requested_transition_count: input.requested_transition_count,
        });
    }

    Ok(GenfmtJasTransitionCount {
        transition_count: input.requested_transition_count,
    })
}

/// Prepare checked FEFF GENFMTJAS transition matrices before the energy loop.
///
/// This composes the `regenf.f90` spherical `jinit=jmax` override with the
/// `genfmtjas.f90` transition branch (`mmtrjas` vs. `mmtrjas0`) and the
/// `indmaxt == indmax` consistency check. The selected branch receives the
/// effective `jinit`; the unused branch is left untouched.
pub fn genfmt_jas_transition_setup(
    input: GenfmtJasTransitionSetupInput<'_>,
) -> Result<GenfmtJasTransitionSetup, GenfmtError> {
    validate_finite_scalar("ellipticity", input.ellipticity)?;

    let transition_count = genfmt_jas_transition_count(GenfmtJasTransitionCountInput {
        phase_transition_count: input.phase_transition_count,
        requested_transition_count: input.requested_transition_count,
    })?;

    if input.ellipticity >= 0.0 {
        let effective_initial_j =
            genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
                ellipticity: input.ellipticity,
                initial_j2: input.left_right.initial_j2,
                final_j2_max: input.left_right.final_j2_max,
            })?;
        let mut left_right = input.left_right;
        left_right.initial_j2 = effective_initial_j.initial_j2;
        let matrices = genfmt_jas_transition_matrices(GenfmtJasTransitionMatricesInput {
            ellipticity: input.ellipticity,
            left_right,
            spherical: input.spherical,
        })?;
        Ok(GenfmtJasTransitionSetup {
            effective_initial_j,
            transition_count,
            matrices,
        })
    } else {
        let effective_initial_j =
            genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
                ellipticity: input.ellipticity,
                initial_j2: input.spherical.initial_j2,
                final_j2_max: input.spherical.final_j2_max,
            })?;
        let mut spherical = input.spherical;
        spherical.initial_j2 = effective_initial_j.initial_j2;
        let matrices = genfmt_jas_transition_matrices(GenfmtJasTransitionMatricesInput {
            ellipticity: input.ellipticity,
            left_right: input.left_right,
            spherical,
        })?;
        Ok(GenfmtJasTransitionSetup {
            effective_initial_j,
            transition_count,
            matrices,
        })
    }
}

/// Wire checked FEFF GENFMTJAS transition setup into the path energy loop.
///
/// This is the Rust equivalent of carrying the selected `mmtrjas`/`mmtrjas0`
/// output tensors plus the effective `jinit` into the `genfmtjas.f90` energy
/// loop. The returned views are sliced to the checked `indmax` transition count
/// and the supplied q weights, so callers cannot accidentally mix stale FEFF
/// table capacity with the active transition set.
pub fn genfmt_jas_energy_grid_branch_from_transition_setup<'a>(
    input: GenfmtJasEnergyGridBranchFromTransitionSetupInput<'a>,
) -> Result<GenfmtJasPathEnergyGridBranchInput<'a>, GenfmtError> {
    let transition_count = input.transition_setup.transition_count.transition_count;
    let q_count = input.q_weights.len();
    validate_positive_limit("q_weights", q_count)?;
    validate_positive_limit("radial_factors", input.radial_factors.shape()[0])?;
    ensure_axis_len(
        "transition_angular_momenta",
        "transition",
        input.transition_angular_momenta.len(),
        transition_count,
    )?;
    ensure_axis_len(
        "radial_factors",
        "q",
        input.radial_factors.shape()[1],
        q_count,
    )?;
    ensure_axis_len(
        "radial_factors",
        "transition",
        input.radial_factors.shape()[2],
        transition_count,
    )?;

    let transition_angular_momenta = input
        .transition_angular_momenta
        .slice_axis_move(Axis(0), Slice::from(..transition_count));
    let radial_factors = input
        .radial_factors
        .slice_axis_move(Axis(1), Slice::from(..q_count))
        .slice_axis_move(Axis(2), Slice::from(..transition_count));
    let q_weights = input
        .q_weights
        .slice_axis_move(Axis(0), Slice::from(..q_count));
    let initial_j2 = input.transition_setup.effective_initial_j.initial_j2;

    match &input.transition_setup.matrices {
        GenfmtJasTransitionMatrices::LeftRight(matrices) => {
            ensure_axis_len(
                "left_transition_matrix",
                "q",
                matrices.left_matrix.shape()[2],
                q_count,
            )?;
            ensure_axis_len(
                "left_transition_matrix",
                "transition",
                matrices.left_matrix.shape()[3],
                transition_count,
            )?;
            ensure_axis_len(
                "right_transition_matrix",
                "q",
                matrices.right_matrix.shape()[2],
                q_count,
            )?;
            ensure_axis_len(
                "right_transition_matrix",
                "transition",
                matrices.right_matrix.shape()[3],
                transition_count,
            )?;

            Ok(GenfmtJasPathEnergyGridBranchInput::LeftRight {
                transition_angular_momenta,
                radial_factors,
                q_weights,
                left_transition_matrix: matrices
                    .left_matrix
                    .view()
                    .slice_axis_move(Axis(2), Slice::from(..q_count))
                    .slice_axis_move(Axis(3), Slice::from(..transition_count)),
                right_transition_matrix: matrices
                    .right_matrix
                    .view()
                    .slice_axis_move(Axis(2), Slice::from(..q_count))
                    .slice_axis_move(Axis(3), Slice::from(..transition_count)),
                initial_j2,
                transition_magnetic_offset: input.transition_magnetic_offset,
                max_angular_momentum: input.max_angular_momentum,
                decomposition_l_max: input.decomposition_l_max,
            })
        }
        GenfmtJasTransitionMatrices::Spherical(matrix) => {
            ensure_axis_len(
                "transition_matrix",
                "transition",
                matrix.matrix.shape()[4],
                transition_count,
            )?;

            Ok(GenfmtJasPathEnergyGridBranchInput::Spherical {
                transition_angular_momenta,
                radial_factors,
                q_weights,
                transition_matrix: matrix
                    .matrix
                    .view()
                    .slice_axis_move(Axis(4), Slice::from(..transition_count)),
                initial_j2,
                transition_magnetic_offset: input.transition_magnetic_offset,
                max_angular_momentum: input.max_angular_momentum,
                decomposition_l_max: input.decomposition_l_max,
            })
        }
    }
}

/// Build a FEFF GENFMTJAS energy-loop input from checked setup products.
///
/// This ports the driver wiring between the pre-path setup blocks and the
/// per-path energy loop: path-local lambda/rotation tables come from
/// `rdpath`/`setlam`/`rot3i`, spin-selected phase and momentum tables come from
/// the driver setup, and the JAS termination branch is supplied by the checked
/// transition setup adapter.
pub fn genfmt_jas_path_energy_grid_from_setup<'a>(
    input: GenfmtJasPathEnergyGridFromSetupInput<'a>,
) -> Result<GenfmtJasPathEnergyGridInput<'a>, GenfmtError> {
    let energy_count = input.driver_setup.momentum_grid.complex_momenta.len();
    validate_positive_limit("complex_momenta", energy_count)?;
    ensure_axis_len(
        "wave_numbers",
        "energy",
        input.driver_setup.momentum_grid.wave_numbers.len(),
        energy_count,
    )?;
    ensure_axis_len(
        "phase_shifts",
        "energy",
        input.driver_setup.phase_shifts.phase_shifts.shape()[0],
        energy_count,
    )?;
    ensure_axis_len(
        "angular_limits",
        "energy",
        input.angular_limits.shape()[0],
        energy_count,
    )?;

    let full_lambda_count = input.path_setup.lambda.m_indices.len();
    let initial_lambda_count = input.path_setup.lambda.initial_l_prefix_len;
    validate_positive_limit("full_lambda_count", full_lambda_count)?;
    validate_positive_limit("initial_lambda_count", initial_lambda_count)?;
    ensure_axis_len(
        "n_indices",
        "lambda",
        input.path_setup.lambda.n_indices.len(),
        full_lambda_count,
    )?;
    ensure_axis_len(
        "m_indices",
        "initial_lambda",
        full_lambda_count,
        initial_lambda_count,
    )?;

    let leg_count = input.path_setup.angles.leg_lengths.len();
    validate_positive_limit("leg_lengths", leg_count)?;
    ensure_axis_len(
        "path_potential_indices",
        "leg",
        input.path_potential_indices.len(),
        leg_count,
    )?;
    ensure_axis_len(
        "rotations",
        "leg",
        input.path_setup.rotations.rotations.shape()[0],
        leg_count,
    )?;
    validate_positive_limit("xnlm_rows", input.xnlm.shape()[0])?;
    validate_positive_limit("xnlm_columns", input.xnlm.shape()[1])?;
    validate_jas_energy_grid_branch(input.branch, energy_count)?;

    Ok(GenfmtJasPathEnergyGridInput {
        m_indices: input.path_setup.lambda.m_indices.view(),
        n_indices: input.path_setup.lambda.n_indices.view(),
        full_lambda_count,
        initial_lambda_count,
        path_potential_indices: input.path_potential_indices,
        angular_limits: input.angular_limits,
        phase_shifts: input.driver_setup.phase_shifts.phase_shifts.view(),
        signed_angular_offset: input.signed_angular_offset,
        leg_lengths: input.path_setup.angles.leg_lengths.view(),
        complex_momenta: input.driver_setup.momentum_grid.complex_momenta.view(),
        wave_numbers: input.driver_setup.momentum_grid.wave_numbers.view(),
        momentum_zero_epsilon: input.momentum_zero_epsilon,
        max_m_plus_one: input.path_setup.lambda.max_m_plus_one,
        max_n: input.path_setup.lambda.max_n,
        rotations: input.path_setup.rotations.rotations.view(),
        rotation_magnetic_offset: input.path_setup.rotations.rotation_magnetic_offset,
        xnlm: input.xnlm,
        eta_angles: input.path_setup.angles.eta_values.view(),
        branch: input.branch,
    })
}

/// Build FEFF GENFMT momentum arrays from the complex energy grid.
///
/// This ports the shared `genfmtsub.f90`/`genfmtjas.f90` setup loop:
/// `xk=getxk(dble(em)-edge)`, `ck=sqrt(2*(em-eref))`, `ckmag=abs(ck)`,
/// and `xkr=real(xk)`.
pub fn genfmt_momentum_grid(
    input: GenfmtMomentumGridInput<'_>,
) -> Result<GenfmtMomentumGrid, GenfmtError> {
    validate_positive_limit("energies", input.energies.len())?;
    ensure_axis_len(
        "reference_energies",
        "energy",
        input.reference_energies.len(),
        input.energies.len(),
    )?;
    validate_finite_scalar("edge", input.edge)?;

    let energy_count = input.energies.len();
    let mut wave_numbers = Array1::<Real>::zeros(energy_count);
    let mut complex_momenta = Array1::<Complex>::zeros(energy_count);
    let mut complex_momentum_magnitudes = Array1::<Real>::zeros(energy_count);
    let mut output_wave_numbers = Array1::<Real>::zeros(energy_count);

    for energy_index in 0..energy_count {
        let energy = complex_vector_entry(input.energies, "energies", energy_index)?;
        let reference_energy =
            complex_vector_entry(input.reference_energies, "reference_energies", energy_index)?;

        let relative_energy = energy.re - input.edge;
        if !relative_energy.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field: "relative_energies",
                index: energy_index,
                value: relative_energy,
            });
        }

        let wave_number = wave_number_from_hartree(relative_energy);
        if !wave_number.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field: "wave_numbers",
                index: energy_index,
                value: wave_number,
            });
        }

        let complex_momentum = ((energy - reference_energy) * 2.0).sqrt();
        if !complex_momentum.re.is_finite() || !complex_momentum.im.is_finite() {
            return Err(GenfmtError::NonFiniteTableComplex {
                table: "complex_momenta",
                row: energy_index,
                column: 0,
                real: complex_momentum.re,
                imaginary: complex_momentum.im,
            });
        }

        let complex_momentum_magnitude = complex_momentum.norm();
        if !complex_momentum_magnitude.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field: "complex_momentum_magnitudes",
                index: energy_index,
                value: complex_momentum_magnitude,
            });
        }

        wave_numbers[energy_index] = wave_number;
        complex_momenta[energy_index] = complex_momentum;
        complex_momentum_magnitudes[energy_index] = complex_momentum_magnitude;
        output_wave_numbers[energy_index] = wave_number;
    }

    Ok(GenfmtMomentumGrid {
        wave_numbers,
        complex_momenta,
        complex_momentum_magnitudes,
        output_wave_numbers,
    })
}

/// Prepare FEFF ordinary GENFMT `ck(ie,is)` for every active spin channel.
///
/// The ordinary driver computes a header-averaged momentum grid for
/// `feff.bin`, but the path spin loop uses the spin-selected `eref2(:,is)`
/// values. This helper ports that per-spin setup into a compact
/// `(energy, spin)` table while preserving the shared `xk(ie)` grid.
pub fn genfmt_ordinary_spin_momentum_grid(
    input: GenfmtOrdinarySpinMomentumGridInput<'_>,
) -> Result<GenfmtOrdinarySpinMomentumGrid, GenfmtError> {
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    validate_positive_limit("energies", input.energies.len())?;

    let energy_count = input.energies.len();
    let mut wave_numbers = Array1::<Real>::zeros(energy_count);
    let mut complex_momenta =
        Array2::<Complex>::zeros((energy_count, input.spin_channel_count).f());
    let mut complex_momentum_magnitudes =
        Array2::<Real>::zeros((energy_count, input.spin_channel_count).f());

    for spin in 0..input.spin_channel_count {
        let reference_energies = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: input.spin_reference_energies,
            spin_channel_count: input.spin_channel_count,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: spin },
        })?;
        let grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: input.energies,
            reference_energies: reference_energies.reference_energies.view(),
            edge: input.edge,
        })?;

        if spin == 0 {
            wave_numbers.assign(&grid.wave_numbers);
        }
        for energy in 0..energy_count {
            complex_momenta[(energy, spin)] = grid.complex_momenta[energy];
            complex_momentum_magnitudes[(energy, spin)] = grid.complex_momentum_magnitudes[energy];
        }
    }

    Ok(GenfmtOrdinarySpinMomentumGrid {
        wave_numbers,
        complex_momenta,
        complex_momentum_magnitudes,
    })
}

/// Prepare the shared FEFF GENFMT `feff.bin` header block.
///
/// This ports the header setup before the path loop in `genfmtsub.f90` and
/// `genfmtjas.f90`: write the FEFF version/misc fields, fill blank `potlbl`
/// entries from `atsym(iz)`, then copy `phc`, `ck`, and `xk` arrays.
pub fn genfmt_feff_bin_header(
    input: GenfmtFeffBinHeaderInput<'_>,
) -> Result<GenfmtFeffBinHeader, GenfmtError> {
    let version = input.version.trim_end();
    if version.is_empty() || !version.is_ascii() {
        return Err(GenfmtError::InvalidTextField { field: "version" });
    }
    if input.pad_width <= 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "pad_width",
            value: input.pad_width,
        });
    }
    validate_finite_scalar("average_norman_radius", input.average_norman_radius)?;
    validate_finite_scalar("fermi_level", input.fermi_level)?;
    validate_finite_scalar("edge_energy", input.edge_energy)?;

    let potential_count = input.atomic_numbers.len();
    validate_positive_limit("atomic_numbers", potential_count)?;
    ensure_axis_len(
        "potential_labels",
        "potential",
        input.potential_labels.len(),
        potential_count,
    )?;

    let energy_count = input.central_phase_shifts.len();
    validate_positive_limit("central_phase_shifts", energy_count)?;
    ensure_axis_len(
        "complex_momenta",
        "energy",
        input.complex_momenta.len(),
        energy_count,
    )?;
    ensure_axis_len(
        "wave_numbers",
        "energy",
        input.wave_numbers.len(),
        energy_count,
    )?;

    let mut potentials = Vec::with_capacity(potential_count);
    for index in 0..potential_count {
        potentials.push(GenfmtFeffBinPotential {
            label: genfmt_output_potential_label(
                input.potential_labels[index],
                input.atomic_numbers[index],
                index,
            )?,
            atomic_number: input.atomic_numbers[index],
        });
    }

    let mut central_phase_shifts = Array1::<Complex>::zeros(energy_count);
    let mut complex_momenta = Array1::<Complex>::zeros(energy_count);
    let mut wave_numbers = Array1::<Real>::zeros(energy_count);
    for energy in 0..energy_count {
        central_phase_shifts[energy] =
            complex_vector_entry(input.central_phase_shifts, "central_phase_shifts", energy)?;
        complex_momenta[energy] =
            complex_vector_entry(input.complex_momenta, "complex_momenta", energy)?;
        wave_numbers[energy] = finite_vector_value(input.wave_numbers, "wave_numbers", energy)?;
    }

    Ok(GenfmtFeffBinHeader {
        version: version.to_string(),
        pad_width: input.pad_width,
        core_hole: input.core_hole,
        order: input.order,
        initial_angular_momentum: input.initial_angular_momentum,
        average_norman_radius: input.average_norman_radius,
        fermi_level: input.fermi_level,
        edge_energy: input.edge_energy,
        potentials,
        central_phase_shifts,
        complex_momenta,
        wave_numbers,
    })
}

/// Prepare the common FEFF GENFMT driver state before evaluating paths.
///
/// This composes the setup block shared by `genfmtsub.f90` and `genfmtjas.f90`:
/// apply FEFF's active spin-channel rule, build the header reference-energy
/// and phase-shift tables, select the central phase channel, compute `xk`/`ck`,
/// and assemble the `feff.bin` header payload.
pub fn genfmt_driver_setup(
    input: GenfmtDriverSetupInput<'_>,
) -> Result<GenfmtDriverSetup, GenfmtError> {
    let spin_channel_count = genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
        spin_selector: input.spin_selector,
        available_spin_channels: input.available_spin_channels,
    })?;
    let reference_energies = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
        spin_reference_energies: input.spin_reference_energies,
        spin_channel_count,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;
    let phase_shifts = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: input.spin_phase_shifts,
        angular_limits: input.angular_limits,
        signed_angular_offset: input.signed_angular_offset,
        spin_channel_count,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;
    let central_phase_shifts = genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
        phase_shifts: phase_shifts.phase_shifts.view(),
        signed_angular_offset: input.signed_angular_offset,
        initial_orbital_l: input.initial_orbital_l,
        initial_kappa: input.initial_kappa,
    })?;
    let spin_momentum_grid =
        genfmt_ordinary_spin_momentum_grid(GenfmtOrdinarySpinMomentumGridInput {
            energies: input.energies,
            spin_reference_energies: input.spin_reference_energies,
            edge: input.edge_energy,
            spin_channel_count,
        })?;
    let momentum_grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
        energies: input.energies,
        reference_energies: reference_energies.reference_energies.view(),
        edge: input.edge_energy,
    })?;
    let initial_angular_momentum =
        input
            .initial_orbital_l
            .checked_add(1)
            .ok_or(GenfmtError::IntegerOverflow {
                field: "initial_angular_momentum",
                value: input.initial_orbital_l,
            })?;
    let initial_angular_momentum =
        checked_i32("initial_angular_momentum", initial_angular_momentum)?;
    let header = genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
        version: input.version,
        pad_width: input.pad_width,
        core_hole: input.core_hole,
        order: input.order,
        initial_angular_momentum,
        average_norman_radius: input.average_norman_radius,
        fermi_level: input.fermi_level,
        edge_energy: input.edge_energy,
        potential_labels: input.potential_labels,
        atomic_numbers: input.atomic_numbers,
        central_phase_shifts: central_phase_shifts.phase_shifts.view(),
        complex_momenta: momentum_grid.complex_momenta.view(),
        wave_numbers: momentum_grid.output_wave_numbers.view(),
    })?;

    Ok(GenfmtDriverSetup {
        spin_channel_count,
        reference_energies,
        phase_shifts,
        central_phase_shifts,
        spin_momentum_grid,
        momentum_grid,
        header,
    })
}

/// Prepare the FEFF GENFMTJAS driver state before evaluating paths.
///
/// This composes the `genfmtjas.f90` setup block that copies one active spin
/// channel into `eref`, `ph`, and `rkk`, computes `xk`/`ck`, and assembles the
/// shared `feff.bin` header. The spin-copy rule intentionally differs from
/// ordinary GENFMT's two-spin averaging branch.
pub fn genfmt_jas_driver_setup(
    input: GenfmtJasDriverSetupInput<'_>,
) -> Result<GenfmtJasDriverSetup, GenfmtError> {
    let spin_selection = genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
        spin_selector: input.spin_selector,
        available_spin_channels: input.available_spin_channels,
    })?;
    let reference_energies = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
        spin_reference_energies: input.spin_reference_energies,
        spin_channel_count: input.available_spin_channels,
        mode: GenfmtReferenceEnergyMode::SpinChannel {
            spin_index: spin_selection.spin_index,
        },
    })?;
    let phase_shifts = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: input.spin_phase_shifts,
        angular_limits: input.angular_limits,
        signed_angular_offset: input.signed_angular_offset,
        spin_channel_count: input.available_spin_channels,
        mode: GenfmtReferenceEnergyMode::SpinChannel {
            spin_index: spin_selection.spin_index,
        },
    })?;
    let radial_factors = genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
        spin_radial_factors: input.spin_radial_factors,
        spin_index: spin_selection.spin_index,
    })?;
    let central_phase_shifts = genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
        phase_shifts: phase_shifts.phase_shifts.view(),
        signed_angular_offset: input.signed_angular_offset,
        initial_orbital_l: input.initial_orbital_l,
        initial_kappa: input.initial_kappa,
    })?;
    let momentum_grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
        energies: input.energies,
        reference_energies: reference_energies.reference_energies.view(),
        edge: input.edge_energy,
    })?;
    let initial_angular_momentum =
        input
            .initial_orbital_l
            .checked_add(1)
            .ok_or(GenfmtError::IntegerOverflow {
                field: "initial_angular_momentum",
                value: input.initial_orbital_l,
            })?;
    let initial_angular_momentum =
        checked_i32("initial_angular_momentum", initial_angular_momentum)?;
    let header = genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
        version: input.version,
        pad_width: input.pad_width,
        core_hole: input.core_hole,
        order: input.order,
        initial_angular_momentum,
        average_norman_radius: input.average_norman_radius,
        fermi_level: input.fermi_level,
        edge_energy: input.edge_energy,
        potential_labels: input.potential_labels,
        atomic_numbers: input.atomic_numbers,
        central_phase_shifts: central_phase_shifts.phase_shifts.view(),
        complex_momenta: momentum_grid.complex_momenta.view(),
        wave_numbers: momentum_grid.output_wave_numbers.view(),
    })?;

    Ok(GenfmtJasDriverSetup {
        spin_selection,
        reference_energies,
        phase_shifts,
        radial_factors,
        central_phase_shifts,
        momentum_grid,
        header,
    })
}

/// Select FEFF `sclmz` angular limits for each path leg at one energy.
///
/// This ports the GENFMT driver loop that computes `mnmxp1=mmaxp1+nmax`,
/// wraps `isc0=0` to `nleg`, chooses
/// `lxp1=max(lmax(ie,ipot(isc0))+1,lmax(ie,ipot(isc1))+1)`, and clamps
/// `mnp1=min(lxp1,mnmxp1)` before calling `sclmz`.
pub fn genfmt_curved_wave_leg_limits(
    input: GenfmtCurvedWaveLegLimitsInput<'_>,
) -> Result<GenfmtCurvedWaveLegLimits, GenfmtError> {
    validate_positive_limit("path_potential_indices", input.path_potential_indices.len())?;
    validate_positive_limit("max_m_plus_one", input.max_m_plus_one)?;
    if isize::try_from(input.max_n).is_err() {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "max_n",
            value: input.max_n,
        });
    }
    let required_energy_count =
        input
            .energy_index
            .checked_add(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "energy_index",
                value: input.energy_index,
            })?;
    ensure_axis_len(
        "angular_limits",
        "energy",
        input.angular_limits.shape()[0],
        required_energy_count,
    )?;

    let mixed_order_capacity =
        input
            .max_m_plus_one
            .checked_add(input.max_n)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "mixed_order_capacity",
                value: input.max_m_plus_one,
            })?;

    let mut limits = Vec::with_capacity(input.path_potential_indices.len());
    for leg_index in 0..input.path_potential_indices.len() {
        let previous_index = if leg_index == 0 {
            input.path_potential_indices.len() - 1
        } else {
            leg_index - 1
        };
        let previous_potential_index = input.path_potential_indices[previous_index];
        let current_potential_index = input.path_potential_indices[leg_index];
        let previous_angular_count = genfmt_angular_count_for_potential(
            input.angular_limits,
            input.energy_index,
            previous_potential_index,
        )?;
        let current_angular_count = genfmt_angular_count_for_potential(
            input.angular_limits,
            input.energy_index,
            current_potential_index,
        )?;
        let angular_count = previous_angular_count.max(current_angular_count);
        let mixed_order_count = angular_count.min(mixed_order_capacity);

        limits.push(GenfmtCurvedWaveLegLimit {
            previous_potential_index,
            current_potential_index,
            angular_count,
            mixed_order_count,
        });
    }

    Ok(GenfmtCurvedWaveLegLimits {
        mixed_order_capacity,
        limits,
    })
}

/// Build FEFF `clmi(il,im,ileg)` curved-wave polynomial tables for all legs.
///
/// The GENFMT drivers zero the full `clmi` common block and then call `sclmz`
/// once per path leg with that leg's `lxp1`/`mnp1` limits. This helper keeps
/// the same behavior by returning a zero-filled Fortran-order 3D table and
/// copying each active `sclmz` prefix into the matching leg plane.
pub fn genfmt_curved_wave_polynomial_tables(
    input: GenfmtCurvedWavePolynomialTablesInput<'_>,
) -> Result<GenfmtCurvedWavePolynomialTables, GenfmtError> {
    validate_positive_limit("leg_rhos", input.leg_rhos.len())?;
    validate_positive_limit("leg_limits", input.leg_limits.len())?;
    ensure_axis_len(
        "leg_rhos",
        "leg",
        input.leg_rhos.len(),
        input.leg_limits.len(),
    )?;
    validate_positive_limit("mixed_order_capacity", input.mixed_order_capacity)?;

    let max_angular_count = input
        .leg_limits
        .iter()
        .map(|limit| limit.angular_count)
        .max()
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "leg_limits",
            value: 0,
        })?;
    validate_positive_limit("angular_count", max_angular_count)?;
    let row_count = max_angular_count
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "angular_count",
            value: max_angular_count,
        })?;

    let mut tables = Array3::<Complex>::zeros(
        (
            row_count,
            input.mixed_order_capacity,
            input.leg_limits.len(),
        )
            .f(),
    );
    for (leg_index, limit) in input.leg_limits.iter().enumerate() {
        validate_positive_limit("angular_count", limit.angular_count)?;
        validate_positive_limit("mixed_order_count", limit.mixed_order_count)?;
        if limit.mixed_order_count > input.mixed_order_capacity {
            return Err(GenfmtError::TableAxisTooShort {
                table: "curved_wave_polynomial_tables",
                axis: "mixed_order",
                length: input.mixed_order_capacity,
                required: limit.mixed_order_count,
            });
        }
        if limit.mixed_order_count > limit.angular_count {
            return Err(GenfmtError::TableAxisTooShort {
                table: "curved_wave_polynomial_tables",
                axis: "angular_count",
                length: limit.angular_count,
                required: limit.mixed_order_count,
            });
        }
        let rho = complex_vector_entry(input.leg_rhos, "leg_rhos", leg_index)?;
        let leg_table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: limit.angular_count,
            mmaxp1: limit.mixed_order_count,
            rho,
        })?;

        for angular in 0..leg_table.shape()[0] {
            for mixed_order in 0..leg_table.shape()[1] {
                tables[(angular, mixed_order, leg_index)] = leg_table[(angular, mixed_order)];
            }
        }
    }

    Ok(GenfmtCurvedWavePolynomialTables { tables })
}

/// Prepare FEFF GENFMT path geometry for one energy.
///
/// This ports the driver setup immediately before the scattering-matrix work:
/// `reff=sum(ri)/2`, `rho(ileg)=ck(ie)*ri(ileg)`, and the `abs(ck) <= eps`
/// branch that skips undefined XAFS calculations.
pub fn genfmt_path_geometry(
    input: GenfmtPathGeometryInput<'_>,
) -> Result<GenfmtPathGeometry, GenfmtError> {
    validate_positive_limit("leg_lengths", input.leg_lengths.len())?;
    validate_finite_complex("complex_momentum", input.complex_momentum)?;
    validate_finite_scalar("momentum_zero_epsilon", input.momentum_zero_epsilon)?;
    if input.momentum_zero_epsilon < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "momentum_zero_epsilon",
            value: input.momentum_zero_epsilon,
        });
    }

    let mut leg_rhos = Array1::<Complex>::zeros(input.leg_lengths.len());
    let mut path_length = 0.0;
    for (index, &leg_length) in input.leg_lengths.iter().enumerate() {
        if !leg_length.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field: "leg_lengths",
                index,
                value: leg_length,
            });
        }
        path_length += leg_length;
        let rho = input.complex_momentum * leg_length;
        if !rho.re.is_finite() || !rho.im.is_finite() {
            return Err(GenfmtError::NonFiniteTableComplex {
                table: "leg_rhos",
                row: index,
                column: 0,
                real: rho.re,
                imaginary: rho.im,
            });
        }
        leg_rhos[index] = rho;
    }

    let effective_path_length = path_length / 2.0;
    validate_finite_scalar("effective_path_length", effective_path_length)?;

    Ok(GenfmtPathGeometry {
        leg_rhos,
        effective_path_length,
        active: input.complex_momentum.norm() > input.momentum_zero_epsilon,
    })
}

fn genfmt_effective_half_path_length_bohr(
    leg_lengths: ArrayView1<'_, Real>,
) -> Result<Real, GenfmtError> {
    validate_positive_limit("leg_lengths", leg_lengths.len())?;
    let mut path_length = 0.0;
    for leg_index in 0..leg_lengths.len() {
        let leg_length = finite_vector_value(leg_lengths, "leg_lengths", leg_index)?;
        if leg_length < 0.0 {
            return Err(GenfmtError::NegativeScalar {
                field: "leg_lengths",
                value: leg_length,
            });
        }
        path_length += leg_length;
    }
    let effective_half_path_length = path_length / 2.0;
    validate_finite_scalar("effective_half_path_length", effective_half_path_length)?;
    Ok(effective_half_path_length)
}

/// Accumulate one FEFF GENFMT path contribution into `cchi(ie)`.
///
/// This ports the `cchi(ie)=cchi(ie)+ptrac*cfac` update in
/// `GENFMT/genfmtsub.f90`. In the two-spin branch FEFF flips the first spin
/// channel by negating `cfac` before accumulation; this helper applies the
/// same sign while keeping the caller's path factor unchanged.
pub fn genfmt_path_signal_contribution(
    input: GenfmtPathSignalContributionInput,
) -> Result<GenfmtPathSignalContribution, GenfmtError> {
    validate_finite_complex("accumulated_chi", input.accumulated_chi)?;
    validate_finite_complex("path_trace", input.path_trace)?;
    validate_finite_complex("path_factor", input.path_factor)?;
    if input.spin_channel_count == 0 || input.spin_channel_count > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channel_count",
            value: input.spin_channel_count,
        });
    }
    if input.spin_index >= input.spin_channel_count {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "spin_index",
            value: input.spin_index,
        });
    }

    let spin_sign = if input.spin_channel_count == 2 && input.spin_index == 0 {
        -1.0
    } else {
        1.0
    };
    let contribution = input.path_trace * input.path_factor * spin_sign;
    validate_finite_complex("path_signal_contribution", contribution)?;
    let accumulated_chi = input.accumulated_chi + contribution;
    validate_finite_complex("accumulated_chi", accumulated_chi)?;

    Ok(GenfmtPathSignalContribution {
        contribution,
        accumulated_chi,
    })
}

/// Accumulate ordinary FEFF GENFMT path signals over all energy points.
///
/// This ports the `cchi(1:ne)=0` setup and the spin/energy accumulation in
/// `genfmtsub.f90` after `ptrac` and `cfac` have been prepared. Inactive
/// zero-momentum energies remain zero.
pub fn genfmt_path_signals(
    input: GenfmtPathSignalsInput<'_>,
) -> Result<GenfmtPathSignals, GenfmtError> {
    validate_genfmt_spin_channel_count(input.spin_channel_count)?;
    validate_positive_limit("path_factors", input.path_factors.len())?;
    let energy_count = input.path_factors.len();
    ensure_axis_len(
        "path_traces",
        "spin",
        input.path_traces.shape()[0],
        input.spin_channel_count,
    )?;
    ensure_axis_len(
        "path_traces",
        "energy",
        input.path_traces.shape()[1],
        energy_count,
    )?;
    ensure_axis_len("active", "energy", input.active.len(), energy_count)?;

    let mut contributions = Array2::<Complex>::zeros((input.spin_channel_count, energy_count).f());
    let mut chi = Array1::<Complex>::zeros(energy_count);
    for spin in 0..input.spin_channel_count {
        for energy in 0..energy_count {
            if !input.active[energy] {
                continue;
            }
            let contribution =
                genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
                    accumulated_chi: chi[energy],
                    path_trace: table_complex_entry(
                        input.path_traces,
                        "path_traces",
                        spin,
                        energy,
                    )?,
                    path_factor: complex_vector_entry(input.path_factors, "path_factors", energy)?,
                    spin_channel_count: input.spin_channel_count,
                    spin_index: spin,
                })?;
            contributions[(spin, energy)] = contribution.contribution;
            chi[energy] = contribution.accumulated_chi;
        }
    }

    Ok(GenfmtPathSignals { contributions, chi })
}

/// Finalize one ordinary FEFF GENFMT path after trace and path-factor setup.
///
/// This ports the ordinary `genfmtsub.f90` path branch from the spin/energy
/// `cchi` accumulation through the shared path-importance, retention, and
/// retained-output block. Matrix construction remains outside this helper;
/// callers pass the already-computed `ptrac` and `cfac` arrays.
pub fn genfmt_ordinary_path_finalization(
    input: GenfmtOrdinaryPathFinalizationInput<'_>,
) -> Result<GenfmtOrdinaryPathFinalization, GenfmtError> {
    let signals = genfmt_path_signals(GenfmtPathSignalsInput {
        path_traces: input.path_traces,
        path_factors: input.path_factors,
        active: input.active,
        spin_channel_count: input.spin_channel_count,
    })?;

    let output_decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        chi: signals.chi.view(),
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        effective_half_path_length_bohr: input.effective_half_path_length_bohr,
        potential_indices: input.potential_indices,
        positions: input.positions,
        beta_angles: input.beta_angles,
        eta_angles: input.eta_angles,
        leg_lengths: input.leg_lengths,
        phase_epsilon: input.phase_epsilon,
    })?;

    Ok(GenfmtOrdinaryPathFinalization {
        signals,
        output_decision,
    })
}

/// Finalize one ordinary FEFF GENFMT path from a full spin/energy grid.
///
/// This covers the `genfmtsub.f90` branch after the spin loop has already
/// accumulated `cchi(1:ne)`: use the accumulated grid signals for importance,
/// retention, and retained-output generation without recomputing the spin
/// contributions from a single shared `cfac` vector.
pub fn genfmt_ordinary_path_energy_grid_finalization(
    input: GenfmtOrdinaryPathEnergyGridFinalizationInput<'_>,
) -> Result<GenfmtOrdinaryPathFinalization, GenfmtError> {
    let output_decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        chi: input.energy_grid.signals.chi.view(),
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        effective_half_path_length_bohr: input.effective_half_path_length_bohr,
        potential_indices: input.potential_indices,
        positions: input.positions,
        beta_angles: input.beta_angles,
        eta_angles: input.eta_angles,
        leg_lengths: input.leg_lengths,
        phase_epsilon: input.phase_epsilon,
    })?;

    Ok(GenfmtOrdinaryPathFinalization {
        signals: input.energy_grid.signals.clone(),
        output_decision,
    })
}

/// Evaluate one ordinary FEFF GENFMT path from energy-grid setup through output decision.
///
/// This composes the per-path body of `genfmtsub.f90` after lambda/rotation
/// setup: run the spin/energy loop, compute `reff=sum(ri)/2`, then apply path
/// importance, retention, and retained-output conversion.
pub fn genfmt_ordinary_path_evaluation(
    input: GenfmtOrdinaryPathEvaluationInput<'_>,
) -> Result<GenfmtOrdinaryPathEvaluation, GenfmtError> {
    let energy_grid = genfmt_ordinary_path_energy_grid(input.energy_grid)?;
    let effective_half_path_length_bohr =
        genfmt_effective_half_path_length_bohr(input.energy_grid.leg_lengths)?;
    let finalization = genfmt_ordinary_path_energy_grid_finalization(
        GenfmtOrdinaryPathEnergyGridFinalizationInput {
            path_index: input.path_index,
            print_level: input.print_level,
            curved_wave_criterion_percent: input.curved_wave_criterion_percent,
            energy_grid: &energy_grid,
            momentum_magnitudes: input.momentum_magnitudes,
            edge_start_index: input.edge_start_index,
            active_energy_count: input.active_energy_count,
            degeneracy: input.degeneracy,
            current_normalization: input.current_normalization,
            effective_half_path_length_bohr,
            potential_indices: input.energy_grid.path_potential_indices,
            positions: input.positions,
            beta_angles: input.beta_angles,
            eta_angles: input.energy_grid.eta_angles,
            leg_lengths: input.energy_grid.leg_lengths,
            phase_epsilon: input.phase_epsilon,
        },
    )?;

    Ok(GenfmtOrdinaryPathEvaluation {
        energy_grid,
        finalization,
    })
}

/// Evaluate one ordinary FEFF GENFMT path from checked setup products.
///
/// This composes the ordinary setup adapter with the existing path evaluation
/// worker. Path beta angles and leg lengths are sourced from `GenfmtPathSetup`,
/// matching the `rdpath` output used by `genfmtsub.f90`.
pub fn genfmt_ordinary_path_evaluation_from_setup(
    input: GenfmtOrdinaryPathEvaluationFromSetupInput<'_>,
) -> Result<GenfmtOrdinaryPathEvaluation, GenfmtError> {
    let path_setup = input.energy_grid.path_setup;
    let leg_count = path_setup.angles.leg_lengths.len();
    validate_positive_limit("leg_lengths", leg_count)?;
    ensure_axis_len("positions", "leg", input.positions.shape()[0], leg_count)?;
    if input.positions.shape()[1] != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: input.positions.shape()[1],
        });
    }

    let energy_grid = genfmt_ordinary_path_energy_grid_from_setup(input.energy_grid)?;
    genfmt_ordinary_path_evaluation(GenfmtOrdinaryPathEvaluationInput {
        energy_grid,
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        positions: input.positions,
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: input.phase_epsilon,
    })
}

/// Evaluate one ordinary FEFF GENFMT path from checked driver/path setup.
///
/// This composes the driver-backed ordinary setup adapter with the existing
/// path evaluation worker. Path beta angles and leg lengths are sourced from
/// `GenfmtPathSetup`, and `ckmag(1:ne)` comes from the checked driver setup.
pub fn genfmt_ordinary_path_evaluation_from_driver_setup(
    input: GenfmtOrdinaryPathEvaluationFromDriverSetupInput<'_>,
) -> Result<GenfmtOrdinaryPathEvaluation, GenfmtError> {
    let path_setup = input.energy_grid.path_setup;
    let leg_count = path_setup.angles.leg_lengths.len();
    validate_positive_limit("leg_lengths", leg_count)?;
    ensure_axis_len("positions", "leg", input.positions.shape()[0], leg_count)?;
    if input.positions.shape()[1] != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: input.positions.shape()[1],
        });
    }

    let energy_grid = genfmt_ordinary_path_energy_grid_from_driver_setup(input.energy_grid)?;
    genfmt_ordinary_path_evaluation(GenfmtOrdinaryPathEvaluationInput {
        energy_grid,
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        momentum_magnitudes: input
            .energy_grid
            .driver_setup
            .momentum_grid
            .complex_momentum_magnitudes
            .view(),
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        positions: input.positions,
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: input.phase_epsilon,
    })
}

/// Evaluate ordinary FEFF GENFMT paths in driver order.
///
/// This ports the path-loop `xportx` threading around `genfmtsub.f90`: each
/// path sees the normalization produced by the previous path, starting from
/// FEFF's initial `xportx=-1`, then retained outputs are collected in traversal
/// order.
pub fn genfmt_ordinary_path_sequence(
    input: GenfmtOrdinaryPathSequenceInput<'_>,
) -> Result<GenfmtOrdinaryPathSequence, GenfmtError> {
    validate_finite_scalar("initial_normalization", input.initial_normalization)?;

    let mut current_normalization = input.initial_normalization;
    let mut evaluations = Vec::with_capacity(input.path_inputs.len());
    for path_input in input.path_inputs {
        let mut path_input = *path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_ordinary_path_evaluation(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        evaluations.push(evaluation);
    }

    let path_finalizations = evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &path_finalizations,
    });

    Ok(GenfmtOrdinaryPathSequence {
        evaluations,
        outputs,
    })
}

/// Evaluate setup-based ordinary FEFF GENFMT paths in driver order.
///
/// This mirrors [`genfmt_ordinary_path_sequence`] while using checked path
/// setup inputs for each path. The FEFF `xportx` normalization is threaded
/// through retained and discarded paths in traversal order.
pub fn genfmt_ordinary_path_sequence_from_setup(
    input: GenfmtOrdinaryPathSequenceFromSetupInput<'_>,
) -> Result<GenfmtOrdinaryPathSequence, GenfmtError> {
    validate_finite_scalar("initial_normalization", input.initial_normalization)?;

    let mut current_normalization = input.initial_normalization;
    let mut evaluations = Vec::with_capacity(input.path_inputs.len());
    for path_input in input.path_inputs {
        let mut path_input = *path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_ordinary_path_evaluation_from_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        evaluations.push(evaluation);
    }

    let path_finalizations = evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &path_finalizations,
    });

    Ok(GenfmtOrdinaryPathSequence {
        evaluations,
        outputs,
    })
}

/// Evaluate driver-backed ordinary FEFF GENFMT paths in driver order.
///
/// This mirrors [`genfmt_ordinary_path_sequence`] while sourcing ordinary
/// path-loop momenta from checked driver setup. The FEFF `xportx`
/// normalization is threaded through retained and discarded paths in traversal
/// order.
pub fn genfmt_ordinary_path_sequence_from_driver_setup(
    input: GenfmtOrdinaryPathSequenceFromDriverSetupInput<'_>,
) -> Result<GenfmtOrdinaryPathSequence, GenfmtError> {
    validate_finite_scalar("initial_normalization", input.initial_normalization)?;

    let mut current_normalization = input.initial_normalization;
    let mut evaluations = Vec::with_capacity(input.path_inputs.len());
    for path_input in input.path_inputs {
        let mut path_input = *path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_ordinary_path_evaluation_from_driver_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        evaluations.push(evaluation);
    }

    let path_finalizations = evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &path_finalizations,
    });

    Ok(GenfmtOrdinaryPathSequence {
        evaluations,
        outputs,
    })
}

/// Assemble ordinary GENFMT driver outputs from checked setup and path inputs.
///
/// This composes the post-setup `genfmtsub.f90` responsibilities that are not
/// text I/O: write the prepared header once, walk all path inputs in order,
/// thread `xportx`, collect retained path payloads, and optionally build one
/// `nstar.dat` row per examined path.
pub fn genfmt_ordinary_driver_output(
    input: GenfmtOrdinaryDriverOutputInput<'_>,
) -> Result<GenfmtOrdinaryDriverOutput, GenfmtError> {
    let nstar_rows = input
        .nstar
        .map(|nstar| genfmt_nstar_rows_from_ordinary_driver_inputs(nstar, input.path_inputs))
        .transpose()?;
    let path_sequence = genfmt_ordinary_path_sequence_from_driver_setup(
        GenfmtOrdinaryPathSequenceFromDriverSetupInput {
            path_inputs: input.path_inputs,
            initial_normalization: input.initial_normalization,
        },
    )?;

    Ok(GenfmtOrdinaryDriverOutput {
        header: input.driver_setup.header.clone(),
        path_sequence,
        nstar_rows,
    })
}

/// Collect retained ordinary FEFF GENFMT path outputs in driver order.
///
/// This ports the bookkeeping after each ordinary path finalization: FEFF
/// increments the examined-path count for every path, increments the retained
/// count only for kept paths, and writes retained path payloads in traversal
/// order.
pub fn genfmt_ordinary_path_outputs(
    input: GenfmtOrdinaryPathOutputsInput<'_>,
) -> GenfmtOrdinaryPathOutputs {
    let path_summaries = input
        .path_finalizations
        .iter()
        .map(|path| path.output_decision.summary)
        .collect::<Vec<_>>();
    let retained_paths = input
        .path_finalizations
        .iter()
        .filter_map(|path| path.output_decision.retained_output.clone())
        .collect::<Vec<_>>();
    let final_normalization = input
        .path_finalizations
        .last()
        .map(|path| path.output_decision.importance.normalization);

    GenfmtOrdinaryPathOutputs {
        examined_path_count: input.path_finalizations.len(),
        retained_path_count: retained_paths.len(),
        final_normalization,
        path_summaries,
        retained_paths,
    }
}

/// Apply FEFF GENFMTJAS path factors to total and decomposed traces.
///
/// The JAS driver does not loop over spin channels here. It writes
/// `cchi(ie)=ptrac*cfac`, and when angular decomposition is enabled it writes
/// `pgtrl(lg2,lg1,ie)=ptrac(lg2,lg1)*cfac` while accumulating `lgcchi`.
pub fn genfmt_jas_path_signal(
    input: GenfmtJasPathSignalInput<'_>,
) -> Result<GenfmtJasPathSignal, GenfmtError> {
    validate_finite_complex("path_trace", input.path_trace)?;
    validate_finite_complex("path_factor", input.path_factor)?;

    let chi = input.path_trace * input.path_factor;
    validate_finite_complex("jas_path_signal", chi)?;

    let (decomposed_chi, decomposed_sum) = if let Some(decomposed_traces) = input.decomposed_traces
    {
        validate_positive_limit("decomposition_rows", decomposed_traces.shape()[0])?;
        validate_positive_limit("decomposition_columns", decomposed_traces.shape()[1])?;

        let rows = decomposed_traces.shape()[0];
        let columns = decomposed_traces.shape()[1];
        let mut decomposed_chi = Array2::<Complex>::zeros((rows, columns).f());
        let mut decomposed_sum = Complex::new(0.0, 0.0);
        for row in 0..rows {
            for column in 0..columns {
                let trace =
                    table_complex_entry(decomposed_traces, "decomposed_traces", row, column)?;
                let signal = trace * input.path_factor;
                validate_finite_complex("decomposed_path_signal", signal)?;
                decomposed_chi[(row, column)] = signal;
                decomposed_sum += signal;
            }
        }
        validate_finite_complex("decomposed_path_signal_sum", decomposed_sum)?;
        (Some(decomposed_chi), Some(decomposed_sum))
    } else {
        (None, None)
    };

    Ok(GenfmtJasPathSignal {
        chi,
        decomposed_chi,
        decomposed_sum,
    })
}

/// Build FEFF GENFMTJAS path signals over all energy points.
///
/// This ports the JAS driver's energy-loop storage behavior after `ptrac` and
/// `cfac` have been prepared. Active energies receive `ptrac*cfac`; inactive
/// zero-momentum energies remain zero, including optional decomposition output.
pub fn genfmt_jas_path_signals(
    input: GenfmtJasPathSignalsInput<'_>,
) -> Result<GenfmtJasPathSignals, GenfmtError> {
    validate_positive_limit("path_traces", input.path_traces.len())?;
    let energy_count = input.path_traces.len();
    ensure_axis_len(
        "path_factors",
        "energy",
        input.path_factors.len(),
        energy_count,
    )?;
    ensure_axis_len("active", "energy", input.active.len(), energy_count)?;

    let mut chi = Array1::<Complex>::zeros(energy_count);
    let (mut decomposed_chi, mut decomposed_sums) =
        if let Some(decomposed_traces) = input.decomposed_traces {
            validate_positive_limit("decomposition_rows", decomposed_traces.shape()[0])?;
            validate_positive_limit("decomposition_columns", decomposed_traces.shape()[1])?;
            ensure_axis_len(
                "decomposed_traces",
                "energy",
                decomposed_traces.shape()[2],
                energy_count,
            )?;
            (
                Some(Array3::<Complex>::zeros(
                    (
                        decomposed_traces.shape()[0],
                        decomposed_traces.shape()[1],
                        energy_count,
                    )
                        .f(),
                )),
                Some(Array1::<Complex>::zeros(energy_count)),
            )
        } else {
            (None, None)
        };

    for energy in 0..energy_count {
        if !input.active[energy] {
            continue;
        }

        let decomposed_energy = input
            .decomposed_traces
            .as_ref()
            .map(|traces| traces.index_axis(Axis(2), energy));
        let signal = genfmt_jas_path_signal(GenfmtJasPathSignalInput {
            path_trace: complex_vector_entry(input.path_traces, "path_traces", energy)?,
            path_factor: complex_vector_entry(input.path_factors, "path_factors", energy)?,
            decomposed_traces: decomposed_energy,
        })?;

        chi[energy] = signal.chi;
        if let (Some(output), Some(signal_decomposed)) =
            (decomposed_chi.as_mut(), signal.decomposed_chi.as_ref())
        {
            for row in 0..signal_decomposed.shape()[0] {
                for column in 0..signal_decomposed.shape()[1] {
                    output[(row, column, energy)] = signal_decomposed[(row, column)];
                }
            }
        }
        if let (Some(sums), Some(sum)) = (decomposed_sums.as_mut(), signal.decomposed_sum) {
            sums[energy] = sum;
        }
    }

    Ok(GenfmtJasPathSignals {
        chi,
        decomposed_chi,
        decomposed_sums,
    })
}

/// Finalize one FEFF GENFMTJAS path after trace and path-factor setup.
///
/// This ports the JAS driver branch from `cchi`/`pgtrl` storage through the
/// shared path-importance, retention, and retained-output block. Optional
/// decomposition amplitude/phase output is prepared only when the path is
/// retained, matching the `feffl.bin` write path.
pub fn genfmt_jas_path_finalization(
    input: GenfmtJasPathFinalizationInput<'_>,
) -> Result<GenfmtJasPathFinalization, GenfmtError> {
    let signals = genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
        path_traces: input.path_traces,
        path_factors: input.path_factors,
        active: input.active,
        decomposed_traces: input.decomposed_traces,
    })?;

    let output_decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        chi: signals.chi.view(),
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        effective_half_path_length_bohr: input.effective_half_path_length_bohr,
        potential_indices: input.potential_indices,
        positions: input.positions,
        beta_angles: input.beta_angles,
        eta_angles: input.eta_angles,
        leg_lengths: input.leg_lengths,
        phase_epsilon: input.phase_epsilon,
    })?;

    let decomposed_output = if output_decision.retention.keep {
        match signals.decomposed_chi.as_ref() {
            Some(decomposed_chi) => Some(genfmt_decomposed_chi_amplitude_phase(
                GenfmtDecomposedChiAmplitudePhaseInput {
                    decomposed_chi: decomposed_chi.view(),
                    phase_epsilon: input.phase_epsilon,
                },
            )?),
            None => None,
        }
    } else {
        None
    };

    Ok(GenfmtJasPathFinalization {
        signals,
        output_decision,
        decomposed_output,
    })
}

/// Finalize one FEFF GENFMTJAS path from a full energy grid.
///
/// This composes the post-energy-loop branch of `genfmtjas.f90` using the
/// already accumulated `cchi` and optional `pgtrl` arrays from
/// [`genfmt_jas_path_energy_grid`].
pub fn genfmt_jas_path_energy_grid_finalization(
    input: GenfmtJasPathEnergyGridFinalizationInput<'_>,
) -> Result<GenfmtJasPathFinalization, GenfmtError> {
    let output_decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        chi: input.energy_grid.signals.chi.view(),
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        effective_half_path_length_bohr: input.effective_half_path_length_bohr,
        potential_indices: input.potential_indices,
        positions: input.positions,
        beta_angles: input.beta_angles,
        eta_angles: input.eta_angles,
        leg_lengths: input.leg_lengths,
        phase_epsilon: input.phase_epsilon,
    })?;

    let decomposed_output = if output_decision.retention.keep {
        match input.energy_grid.signals.decomposed_chi.as_ref() {
            Some(decomposed_chi) => Some(genfmt_decomposed_chi_amplitude_phase(
                GenfmtDecomposedChiAmplitudePhaseInput {
                    decomposed_chi: decomposed_chi.view(),
                    phase_epsilon: input.phase_epsilon,
                },
            )?),
            None => None,
        }
    } else {
        None
    };

    Ok(GenfmtJasPathFinalization {
        signals: input.energy_grid.signals.clone(),
        output_decision,
        decomposed_output,
    })
}

/// Evaluate one FEFF GENFMTJAS path from energy-grid setup through output decision.
///
/// This composes the per-path body of `genfmtjas.f90` after lambda/rotation and
/// transition-matrix setup: run the JAS energy loop, compute `reff=sum(ri)/2`,
/// then apply path importance, retention, retained-output conversion, and
/// optional decomposition output.
pub fn genfmt_jas_path_evaluation(
    input: GenfmtJasPathEvaluationInput<'_>,
) -> Result<GenfmtJasPathEvaluation, GenfmtError> {
    let energy_grid = genfmt_jas_path_energy_grid(input.energy_grid)?;
    let effective_half_path_length_bohr =
        genfmt_effective_half_path_length_bohr(input.energy_grid.leg_lengths)?;
    let finalization =
        genfmt_jas_path_energy_grid_finalization(GenfmtJasPathEnergyGridFinalizationInput {
            path_index: input.path_index,
            print_level: input.print_level,
            curved_wave_criterion_percent: input.curved_wave_criterion_percent,
            energy_grid: &energy_grid,
            momentum_magnitudes: input.momentum_magnitudes,
            edge_start_index: input.edge_start_index,
            active_energy_count: input.active_energy_count,
            degeneracy: input.degeneracy,
            current_normalization: input.current_normalization,
            effective_half_path_length_bohr,
            potential_indices: input.energy_grid.path_potential_indices,
            positions: input.positions,
            beta_angles: input.beta_angles,
            eta_angles: input.energy_grid.eta_angles,
            leg_lengths: input.energy_grid.leg_lengths,
            phase_epsilon: input.phase_epsilon,
        })?;

    Ok(GenfmtJasPathEvaluation {
        energy_grid,
        finalization,
    })
}

/// Evaluate one FEFF GENFMTJAS path from checked setup products.
///
/// This composes the driver/path setup adapter with the existing path
/// evaluation worker. Path beta angles and leg lengths are sourced from
/// `GenfmtPathSetup`, matching the `rdpath` output used by `genfmtjas.f90`.
pub fn genfmt_jas_path_evaluation_from_setup(
    input: GenfmtJasPathEvaluationFromSetupInput<'_>,
) -> Result<GenfmtJasPathEvaluation, GenfmtError> {
    let path_setup = input.energy_grid.path_setup;
    let leg_count = path_setup.angles.leg_lengths.len();
    validate_positive_limit("leg_lengths", leg_count)?;
    ensure_axis_len("positions", "leg", input.positions.shape()[0], leg_count)?;
    if input.positions.shape()[1] != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: input.positions.shape()[1],
        });
    }

    let energy_grid = genfmt_jas_path_energy_grid_from_setup(input.energy_grid)?;
    genfmt_jas_path_evaluation(GenfmtJasPathEvaluationInput {
        energy_grid,
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        positions: input.positions,
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: input.phase_epsilon,
    })
}

/// Evaluate one FEFF GENFMTJAS path from checked driver/path setup.
///
/// This composes the setup adapter with the existing path evaluation worker,
/// sourcing `ckmag(1:ne)` from the checked JAS driver setup. This mirrors the
/// `genfmtjas.f90` path loop after `ckmag` has been prepared in the pre-loop
/// momentum setup.
pub fn genfmt_jas_path_evaluation_from_driver_setup(
    input: GenfmtJasPathEvaluationFromDriverSetupInput<'_>,
) -> Result<GenfmtJasPathEvaluation, GenfmtError> {
    let path_setup = input.energy_grid.path_setup;
    let leg_count = path_setup.angles.leg_lengths.len();
    validate_positive_limit("leg_lengths", leg_count)?;
    ensure_axis_len("positions", "leg", input.positions.shape()[0], leg_count)?;
    if input.positions.shape()[1] != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: input.positions.shape()[1],
        });
    }

    let energy_grid = genfmt_jas_path_energy_grid_from_setup(input.energy_grid)?;
    genfmt_jas_path_evaluation(GenfmtJasPathEvaluationInput {
        energy_grid,
        path_index: input.path_index,
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        momentum_magnitudes: input
            .energy_grid
            .driver_setup
            .momentum_grid
            .complex_momentum_magnitudes
            .view(),
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
        positions: input.positions,
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: input.phase_epsilon,
    })
}

/// Evaluate FEFF GENFMTJAS paths in driver order.
///
/// This mirrors the ordinary driver sequence while preserving the JAS
/// decomposition consistency checks in [`genfmt_jas_path_outputs`].
pub fn genfmt_jas_path_sequence(
    input: GenfmtJasPathSequenceInput<'_>,
) -> Result<GenfmtJasPathSequence, GenfmtError> {
    validate_finite_scalar("initial_normalization", input.initial_normalization)?;

    let mut current_normalization = input.initial_normalization;
    let mut evaluations = Vec::with_capacity(input.path_inputs.len());
    for path_input in input.path_inputs {
        let mut path_input = *path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_jas_path_evaluation(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        evaluations.push(evaluation);
    }

    let path_finalizations = evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &path_finalizations,
    })?;

    Ok(GenfmtJasPathSequence {
        evaluations,
        outputs,
    })
}

/// Evaluate setup-based FEFF GENFMTJAS paths in driver order.
///
/// This mirrors [`genfmt_jas_path_sequence`] while using checked driver/path
/// setup inputs for each path. The FEFF `xportx` normalization is threaded
/// through retained and discarded paths in traversal order.
pub fn genfmt_jas_path_sequence_from_setup(
    input: GenfmtJasPathSequenceFromSetupInput<'_>,
) -> Result<GenfmtJasPathSequence, GenfmtError> {
    validate_finite_scalar("initial_normalization", input.initial_normalization)?;

    let mut current_normalization = input.initial_normalization;
    let mut evaluations = Vec::with_capacity(input.path_inputs.len());
    for path_input in input.path_inputs {
        let mut path_input = *path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_jas_path_evaluation_from_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        evaluations.push(evaluation);
    }

    let path_finalizations = evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &path_finalizations,
    })?;

    Ok(GenfmtJasPathSequence {
        evaluations,
        outputs,
    })
}

/// Evaluate driver-backed FEFF GENFMTJAS paths in driver order.
///
/// This mirrors [`genfmt_jas_path_sequence`] while sourcing path-loop momentum
/// magnitudes from checked JAS driver setup. The FEFF `xportx` normalization is
/// threaded through retained and discarded paths in traversal order.
pub fn genfmt_jas_path_sequence_from_driver_setup(
    input: GenfmtJasPathSequenceFromDriverSetupInput<'_>,
) -> Result<GenfmtJasPathSequence, GenfmtError> {
    validate_finite_scalar("initial_normalization", input.initial_normalization)?;

    let mut current_normalization = input.initial_normalization;
    let mut evaluations = Vec::with_capacity(input.path_inputs.len());
    for path_input in input.path_inputs {
        let mut path_input = *path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_jas_path_evaluation_from_driver_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        evaluations.push(evaluation);
    }

    let path_finalizations = evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &path_finalizations,
    })?;

    Ok(GenfmtJasPathSequence {
        evaluations,
        outputs,
    })
}

/// Assemble GENFMTJAS driver outputs from checked setup and path inputs.
///
/// This composes the post-setup `genfmtjas.f90` responsibilities that are not
/// text I/O: write the prepared header once, walk all path inputs in order,
/// thread `xportx`, collect retained path and optional decomposition payloads,
/// and optionally build one `nstar.dat` row per examined path.
pub fn genfmt_jas_driver_output(
    input: GenfmtJasDriverOutputInput<'_>,
) -> Result<GenfmtJasDriverOutput, GenfmtError> {
    let nstar_rows = input
        .nstar
        .map(|nstar| genfmt_nstar_rows_from_jas_driver_inputs(nstar, input.path_inputs))
        .transpose()?;
    let path_sequence =
        genfmt_jas_path_sequence_from_driver_setup(GenfmtJasPathSequenceFromDriverSetupInput {
            path_inputs: input.path_inputs,
            initial_normalization: input.initial_normalization,
        })?;

    Ok(GenfmtJasDriverOutput {
        header: input.driver_setup.header.clone(),
        path_sequence,
        nstar_rows,
    })
}

/// Collect retained FEFF GENFMTJAS path outputs in driver order.
///
/// This mirrors ordinary GENFMT retained-path bookkeeping and also preserves
/// the retained `pgtrl` amplitude/phase payloads for `feffl.bin`. Decomposition
/// output is a run-wide mode in FEFF, so retained paths must either all carry
/// decomposition output or none of them do.
pub fn genfmt_jas_path_outputs(
    input: GenfmtJasPathOutputsInput<'_>,
) -> Result<GenfmtJasPathOutputs, GenfmtError> {
    let path_summaries = input
        .path_finalizations
        .iter()
        .map(|path| path.output_decision.summary)
        .collect::<Vec<_>>();
    let mut retained_paths = Vec::new();
    let mut decomposed_paths = Vec::new();
    let mut saw_decomposition = false;
    let mut saw_missing_decomposition = false;

    for path in input.path_finalizations {
        if let Some(retained) = path.output_decision.retained_output.clone() {
            retained_paths.push(retained);
            match path.decomposed_output.clone() {
                Some(decomposed) => {
                    saw_decomposition = true;
                    decomposed_paths.push(decomposed);
                }
                None => {
                    saw_missing_decomposition = true;
                }
            }
        }
    }

    if saw_decomposition && saw_missing_decomposition {
        return Err(GenfmtError::MismatchedJasFinalizationDecomposition);
    }

    let final_normalization = input
        .path_finalizations
        .last()
        .map(|path| path.output_decision.importance.normalization);
    let decomposed_paths = if saw_decomposition {
        Some(decomposed_paths)
    } else {
        None
    };

    Ok(GenfmtJasPathOutputs {
        examined_path_count: input.path_finalizations.len(),
        retained_path_count: retained_paths.len(),
        final_normalization,
        path_summaries,
        retained_paths,
        decomposed_paths,
    })
}

/// Build FEFF's curved-wave path factor `cfac`.
///
/// The GENFMT drivers compute `srho=sum(rho)`, `prho=product(rho)`, and
/// `cfac=exp(i*(srho-2*xk*reff))/prho` for each energy and path. Rust reports
/// non-finite and zero-product inputs explicitly instead of relying on the
/// Fortran floating-point environment.
pub fn genfmt_curved_wave_path_factor(
    input: GenfmtCurvedWavePathFactorInput<'_>,
) -> Result<GenfmtCurvedWavePathFactor, GenfmtError> {
    validate_positive_limit("leg_rhos", input.leg_rhos.len())?;
    validate_finite_scalar("wave_number", input.wave_number)?;
    validate_finite_scalar("effective_path_length", input.effective_path_length)?;

    let mut rho_sum = Complex::new(0.0, 0.0);
    let mut rho_product = Complex::new(1.0, 0.0);
    for (index, &rho) in input.leg_rhos.iter().enumerate() {
        if !rho.re.is_finite() || !rho.im.is_finite() {
            return Err(GenfmtError::NonFiniteTableComplex {
                table: "leg_rhos",
                row: index,
                column: 0,
                real: rho.re,
                imaginary: rho.im,
            });
        }
        rho_sum += rho;
        rho_product *= rho;
    }

    if rho_product == Complex::new(0.0, 0.0) {
        return Err(GenfmtError::ZeroComplex {
            field: "rho_product",
        });
    }
    validate_finite_complex("rho_product", rho_product)?;

    let phase_argument =
        rho_sum - Complex::new(2.0 * input.wave_number * input.effective_path_length, 0.0);
    let factor = (Complex::new(0.0, 1.0) * phase_argument).exp() / rho_product;
    validate_finite_complex("curved_wave_path_factor", factor)?;

    Ok(GenfmtCurvedWavePathFactor {
        rho_sum,
        rho_product,
        factor,
    })
}

/// Compute FEFF GENFMT path importance from the complex path signal.
///
/// This ports the `ffmag`, `trap`, `xport`, and `crit` block used after the
/// energy loop in both GENFMT drivers. The Rust API makes FEFF's active
/// `ik0..ne1` integration window explicit and returns the updated
/// normalization alongside the path percentage.
pub fn genfmt_path_importance(
    input: GenfmtPathImportanceInput<'_>,
) -> Result<GenfmtPathImportance, GenfmtError> {
    validate_path_importance_input(input)?;

    let mut magnitudes = Array1::<Real>::zeros(input.active_energy_count);
    for energy in 0..input.active_energy_count {
        let chi = complex_vector_entry(input.chi, "chi", energy)?;
        magnitudes[energy] = chi.norm();
    }

    let integration_end = input.active_energy_count;
    let momenta: Vec<Real> = input
        .momentum_magnitudes
        .iter()
        .skip(input.edge_start_index)
        .take(integration_end - input.edge_start_index)
        .copied()
        .collect();
    let values: Vec<Real> = magnitudes
        .iter()
        .skip(input.edge_start_index)
        .take(integration_end - input.edge_start_index)
        .copied()
        .collect();
    let raw_integral = trap(&momenta, &values)?;
    let raw_importance = (input.degeneracy * raw_integral).abs();
    if !raw_importance.is_finite() {
        return Err(GenfmtError::NonFiniteScalar {
            field: "raw_importance",
            value: raw_importance,
        });
    }

    let normalization = if input.current_normalization <= 0.0 {
        raw_importance
    } else {
        input.current_normalization
    };
    if normalization == 0.0 {
        return Err(GenfmtError::ZeroScalar {
            field: "path_importance_normalization",
        });
    }
    let percent = 100.0 * raw_importance / normalization;
    validate_finite_scalar("path_importance_percent", percent)?;

    Ok(GenfmtPathImportance {
        magnitudes,
        raw_importance,
        normalization,
        percent,
    })
}

/// Decide whether FEFF GENFMT writes one path's output data.
///
/// Both GENFMT drivers compute `crit0=2*critcw/3` when `ipr3 <= 0`, then keep
/// a path if `ipr3 >= 1` or `crit >= crit0`. The positive-print-level branch
/// deliberately does not report a threshold because FEFF does not use `crit0`
/// when output is forced.
pub fn genfmt_path_retention(
    input: GenfmtPathRetentionInput,
) -> Result<GenfmtPathRetention, GenfmtError> {
    validate_finite_scalar(
        "curved_wave_criterion_percent",
        input.curved_wave_criterion_percent,
    )?;
    validate_finite_scalar("path_importance_percent", input.path_importance_percent)?;
    if input.curved_wave_criterion_percent < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "curved_wave_criterion_percent",
            value: input.curved_wave_criterion_percent,
        });
    }
    if input.path_importance_percent < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "path_importance_percent",
            value: input.path_importance_percent,
        });
    }

    if input.print_level >= 1 {
        return Ok(GenfmtPathRetention {
            discard_threshold_percent: None,
            keep: true,
        });
    }

    let threshold = 2.0 * input.curved_wave_criterion_percent / 3.0;
    validate_finite_scalar("discard_threshold_percent", threshold)?;

    Ok(GenfmtPathRetention {
        discard_threshold_percent: Some(threshold),
        keep: input.path_importance_percent >= threshold,
    })
}

/// Apply the FEFF GENFMT post-energy path output decision.
///
/// This ports the shared driver branch after `cchi`, `ffmag`, and `crit` are
/// known: update the importance normalization, decide whether the path is
/// retained, and prepare the `feff.bin`/`list.dat` path payload only for
/// retained paths.
pub fn genfmt_path_output_decision(
    input: GenfmtPathOutputDecisionInput<'_>,
) -> Result<GenfmtPathOutputDecision, GenfmtError> {
    validate_positive_limit("path_index", input.path_index)?;
    let leg_count = input.potential_indices.len();
    validate_positive_limit("potential_indices", leg_count)?;
    validate_finite_scalar(
        "effective_half_path_length_bohr",
        input.effective_half_path_length_bohr,
    )?;
    if input.effective_half_path_length_bohr < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "effective_half_path_length_bohr",
            value: input.effective_half_path_length_bohr,
        });
    }

    let importance = genfmt_path_importance(GenfmtPathImportanceInput {
        chi: input.chi,
        momentum_magnitudes: input.momentum_magnitudes,
        edge_start_index: input.edge_start_index,
        active_energy_count: input.active_energy_count,
        degeneracy: input.degeneracy,
        current_normalization: input.current_normalization,
    })?;
    let retention = genfmt_path_retention(GenfmtPathRetentionInput {
        print_level: input.print_level,
        curved_wave_criterion_percent: input.curved_wave_criterion_percent,
        path_importance_percent: importance.percent,
    })?;
    let effective_half_path_length_angstrom =
        input.effective_half_path_length_bohr * FEFF_BOHR_ANGSTROM;
    validate_finite_scalar(
        "effective_half_path_length_angstrom",
        effective_half_path_length_angstrom,
    )?;

    let summary = GenfmtPathOutputSummary {
        path_index: input.path_index,
        retained: retention.keep,
        criterion_percent: importance.percent,
        degeneracy: input.degeneracy,
        leg_count,
        effective_half_path_length_bohr: input.effective_half_path_length_bohr,
        effective_half_path_length_angstrom,
    };

    let retained_output = if retention.keep {
        Some(genfmt_retained_path_output(
            GenfmtRetainedPathOutputInput {
                path_index: input.path_index,
                degeneracy: input.degeneracy,
                criterion_percent: importance.percent,
                effective_half_path_length_bohr: input.effective_half_path_length_bohr,
                potential_indices: input.potential_indices,
                positions: input.positions,
                beta_angles: input.beta_angles,
                eta_angles: input.eta_angles,
                leg_lengths: input.leg_lengths,
                chi: input.chi,
                phase_epsilon: input.phase_epsilon,
            },
        )?)
    } else {
        None
    };

    Ok(GenfmtPathOutputDecision {
        summary,
        importance,
        retention,
        retained_output,
    })
}

/// Convert FEFF GENFMT complex path values into amplitude and phase arrays.
///
/// This ports the `amff`/`phff` block before path output. Phases below
/// `phase_epsilon` are written as zero, and later phases are unwrapped with
/// FEFF's `pijump` branch selection.
pub fn genfmt_chi_amplitude_phase(
    input: GenfmtChiAmplitudePhaseInput<'_>,
) -> Result<GenfmtChiAmplitudePhase, GenfmtError> {
    validate_positive_limit("chi", input.chi.len())?;
    validate_finite_scalar("phase_epsilon", input.phase_epsilon)?;
    if input.phase_epsilon < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "phase_epsilon",
            value: input.phase_epsilon,
        });
    }

    let mut amplitudes = Array1::<Real>::zeros(input.chi.len());
    let mut phases = Array1::<Real>::zeros(input.chi.len());
    let mut previous_phase = 0.0;
    for energy in 0..input.chi.len() {
        let chi = complex_vector_entry(input.chi, "chi", energy)?;
        let amplitude = chi.norm();
        amplitudes[energy] = amplitude;

        let mut phase = if amplitude >= input.phase_epsilon {
            chi.im.atan2(chi.re)
        } else {
            0.0
        };
        if energy > 0 {
            phase = remove_phase_jump(phase, previous_phase)?;
        }
        previous_phase = phase;
        phases[energy] = phase;
    }

    Ok(GenfmtChiAmplitudePhase { amplitudes, phases })
}

/// Prepare the data FEFF writes for one retained GENFMT path.
///
/// This ports the path-output block shared by `genfmtsub.f90` and
/// `genfmtjas.f90`: path metadata is copied in FEFF leg order, `reff` is
/// converted to Angstrom for the text header/list row, the list Debye-Waller
/// column is hard-coded to zero, and `cchi` is converted to `amff`/`phff`.
pub fn genfmt_retained_path_output(
    input: GenfmtRetainedPathOutputInput<'_>,
) -> Result<GenfmtRetainedPathOutput, GenfmtError> {
    validate_positive_limit("path_index", input.path_index)?;
    validate_finite_scalar("degeneracy", input.degeneracy)?;
    validate_finite_scalar("criterion_percent", input.criterion_percent)?;
    validate_finite_scalar(
        "effective_half_path_length_bohr",
        input.effective_half_path_length_bohr,
    )?;
    if input.effective_half_path_length_bohr < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "effective_half_path_length_bohr",
            value: input.effective_half_path_length_bohr,
        });
    }

    let leg_count = input.potential_indices.len();
    validate_positive_limit("potential_indices", leg_count)?;
    ensure_axis_len("positions", "leg", input.positions.shape()[0], leg_count)?;
    if input.positions.shape()[1] != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: input.positions.shape()[1],
        });
    }
    ensure_axis_len("beta_angles", "leg", input.beta_angles.len(), leg_count)?;
    ensure_axis_len("eta_angles", "leg", input.eta_angles.len(), leg_count)?;
    ensure_axis_len("leg_lengths", "leg", input.leg_lengths.len(), leg_count)?;

    let mut positions = Array2::<Real>::zeros((leg_count, 3).f());
    for leg in 0..leg_count {
        for component in 0..3 {
            let value = input.positions[(leg, component)];
            if !value.is_finite() {
                return Err(GenfmtError::NonFinitePathCoordinate {
                    leg_index: leg,
                    component,
                    value,
                });
            }
            positions[(leg, component)] = value;
        }
    }

    let mut beta_angles = Array1::<Real>::zeros(leg_count);
    let mut eta_angles = Array1::<Real>::zeros(leg_count);
    let mut leg_lengths = Array1::<Real>::zeros(leg_count);
    for leg in 0..leg_count {
        beta_angles[leg] = finite_vector_value(input.beta_angles, "beta_angles", leg)?;
        eta_angles[leg] = finite_vector_value(input.eta_angles, "eta_angles", leg)?;
        let leg_length = finite_vector_value(input.leg_lengths, "leg_lengths", leg)?;
        if leg_length < 0.0 {
            return Err(GenfmtError::NegativeScalar {
                field: "leg_lengths",
                value: leg_length,
            });
        }
        leg_lengths[leg] = leg_length;
    }

    let amplitude_phase = genfmt_chi_amplitude_phase(GenfmtChiAmplitudePhaseInput {
        chi: input.chi,
        phase_epsilon: input.phase_epsilon,
    })?;
    let effective_half_path_length_angstrom =
        input.effective_half_path_length_bohr * FEFF_BOHR_ANGSTROM;
    validate_finite_scalar(
        "effective_half_path_length_angstrom",
        effective_half_path_length_angstrom,
    )?;

    Ok(GenfmtRetainedPathOutput {
        path_index: input.path_index,
        degeneracy: input.degeneracy,
        criterion_percent: input.criterion_percent,
        effective_half_path_length_bohr: input.effective_half_path_length_bohr,
        effective_half_path_length_angstrom,
        list_sigma2: 0.0,
        potential_indices: input.potential_indices.to_owned(),
        positions,
        beta_angles,
        eta_angles,
        leg_lengths,
        amplitudes: amplitude_phase.amplitudes,
        phases: amplitude_phase.phases,
    })
}

/// Convert FEFF GENFMTJAS decomposition signals into amplitude/phase tables.
///
/// This ports the `pgtrl(ilm2,ilm1,ie)` output loop in `genfmtjas.f90`. FEFF
/// resets `phffo` for each `(ilm2, ilm1)` pair; this helper preserves that by
/// unwrapping phases independently for every decomposition channel.
pub fn genfmt_decomposed_chi_amplitude_phase(
    input: GenfmtDecomposedChiAmplitudePhaseInput<'_>,
) -> Result<GenfmtDecomposedChiAmplitudePhase, GenfmtError> {
    validate_decomposed_chi_amplitude_phase_input(input)?;

    let shape = input.decomposed_chi.shape();
    let (rows, columns, energies) = (shape[0], shape[1], shape[2]);
    let mut amplitudes = Array3::<Real>::zeros((rows, columns, energies).f());
    let mut phases = Array3::<Real>::zeros((rows, columns, energies).f());

    for row in 0..rows {
        for column in 0..columns {
            let mut previous_phase = 0.0;
            for energy in 0..energies {
                let chi = tensor3_complex_entry(
                    input.decomposed_chi,
                    "decomposed_chi",
                    row,
                    column,
                    energy,
                )?;
                let amplitude = chi.norm();
                amplitudes[(row, column, energy)] = amplitude;

                let mut phase = if amplitude >= input.phase_epsilon {
                    chi.im.atan2(chi.re)
                } else {
                    0.0
                };
                if energy > 0 {
                    phase = remove_phase_jump(phase, previous_phase)?;
                }
                previous_phase = phase;
                phases[(row, column, energy)] = phase;
            }
        }
    }

    Ok(GenfmtDecomposedChiAmplitudePhase { amplitudes, phases })
}

fn validate_path_matrix_product_input(
    input: GenfmtPathMatrixProductInput<'_>,
) -> Result<(), GenfmtError> {
    validate_positive_limit("full_lambda_count", input.full_lambda_count)?;
    validate_positive_limit("initial_lambda_count", input.initial_lambda_count)?;
    if input.initial_lambda_count > input.full_lambda_count {
        return Err(GenfmtError::TableAxisTooShort {
            table: "path_product",
            axis: "full_lambda",
            length: input.full_lambda_count,
            required: input.initial_lambda_count,
        });
    }

    ensure_axis_len(
        "first_scattering",
        "lambda",
        input.first_scattering.shape()[0],
        input.full_lambda_count,
    )?;
    ensure_axis_len(
        "first_scattering",
        "initial_lambda",
        input.first_scattering.shape()[1],
        input.initial_lambda_count,
    )?;
    ensure_axis_len(
        "intermediate_scattering",
        "left_lambda",
        input.intermediate_scattering.shape()[1],
        input.full_lambda_count,
    )?;
    ensure_axis_len(
        "intermediate_scattering",
        "right_lambda",
        input.intermediate_scattering.shape()[2],
        input.full_lambda_count,
    )?;

    Ok(())
}

fn validate_path_matrix_trace_input(
    input: GenfmtPathMatrixTraceInput<'_>,
) -> Result<(), GenfmtError> {
    validate_path_matrix_product_input(GenfmtPathMatrixProductInput {
        first_scattering: input.first_scattering,
        intermediate_scattering: input.intermediate_scattering,
        full_lambda_count: input.full_lambda_count,
        initial_lambda_count: input.initial_lambda_count,
    })?;
    ensure_axis_len(
        "termination_matrix",
        "lambda",
        input.termination_matrix.shape()[0],
        input.initial_lambda_count,
    )?;
    ensure_axis_len(
        "termination_matrix",
        "initial_lambda",
        input.termination_matrix.shape()[1],
        input.initial_lambda_count,
    )?;

    Ok(())
}

fn validate_jas_left_right_path_trace_input(
    input: GenfmtJasLeftRightPathTraceInput<'_>,
) -> Result<(usize, usize, Option<usize>), GenfmtError> {
    validate_positive_limit("lambda_count", input.lambda_count)?;
    ensure_axis_len(
        "path_product",
        "lambda",
        input.path_product.shape()[0],
        input.lambda_count,
    )?;
    ensure_axis_len(
        "path_product",
        "initial_lambda",
        input.path_product.shape()[1],
        input.lambda_count,
    )?;

    let mj_count = input.left_amplitudes.shape()[0];
    let q_count = input.left_amplitudes.shape()[1];
    validate_positive_limit("mj_count", mj_count)?;
    validate_positive_limit("q_count", q_count)?;
    ensure_axis_len(
        "left_amplitudes",
        "lambda",
        input.left_amplitudes.shape()[2],
        input.lambda_count,
    )?;
    ensure_axis_len(
        "right_amplitudes",
        "mj",
        input.right_amplitudes.shape()[0],
        mj_count,
    )?;
    ensure_axis_len(
        "right_amplitudes",
        "q",
        input.right_amplitudes.shape()[1],
        q_count,
    )?;
    ensure_axis_len(
        "right_amplitudes",
        "lambda",
        input.right_amplitudes.shape()[2],
        input.lambda_count,
    )?;

    let decomposition_count = match (
        input.decomposed_left_amplitudes,
        input.decomposed_right_amplitudes,
    ) {
        (Some(left), Some(right)) => {
            ensure_axis_len(
                "decomposed_left_amplitudes",
                "mj",
                left.shape()[0],
                mj_count,
            )?;
            ensure_axis_len("decomposed_left_amplitudes", "q", left.shape()[1], q_count)?;
            ensure_axis_len(
                "decomposed_left_amplitudes",
                "lambda",
                left.shape()[3],
                input.lambda_count,
            )?;
            ensure_axis_len(
                "decomposed_right_amplitudes",
                "mj",
                right.shape()[0],
                mj_count,
            )?;
            ensure_axis_len(
                "decomposed_right_amplitudes",
                "q",
                right.shape()[1],
                q_count,
            )?;
            ensure_axis_len(
                "decomposed_right_amplitudes",
                "lambda",
                right.shape()[3],
                input.lambda_count,
            )?;

            let left_count = left.shape()[2];
            let right_count = right.shape()[2];
            validate_positive_limit("decomposition_count", left_count)?;
            if left_count < right_count {
                return Err(GenfmtError::TableAxisTooShort {
                    table: "decomposed_left_amplitudes",
                    axis: "decomposition",
                    length: left_count,
                    required: right_count,
                });
            }
            if right_count < left_count {
                return Err(GenfmtError::TableAxisTooShort {
                    table: "decomposed_right_amplitudes",
                    axis: "decomposition",
                    length: right_count,
                    required: left_count,
                });
            }
            Some(left_count)
        }
        (None, None) => None,
        _ => return Err(GenfmtError::MismatchedJasDecompositionTables),
    };

    Ok((mj_count, q_count, decomposition_count))
}

fn validate_jas_spherical_path_trace_input(
    input: GenfmtJasSphericalPathTraceInput<'_>,
) -> Result<(usize, Option<usize>), GenfmtError> {
    validate_positive_limit("lambda_count", input.lambda_count)?;
    ensure_axis_len(
        "path_product",
        "lambda",
        input.path_product.shape()[0],
        input.lambda_count,
    )?;
    ensure_axis_len(
        "path_product",
        "initial_lambda",
        input.path_product.shape()[1],
        input.lambda_count,
    )?;

    let mj_count = input.amplitudes.shape()[0];
    validate_positive_limit("mj_count", mj_count)?;
    ensure_axis_len("amplitudes", "spin", input.amplitudes.shape()[1], 2)?;
    ensure_axis_len(
        "amplitudes",
        "lambda2",
        input.amplitudes.shape()[2],
        input.lambda_count,
    )?;
    ensure_axis_len(
        "amplitudes",
        "lambda1",
        input.amplitudes.shape()[3],
        input.lambda_count,
    )?;

    let decomposition_count = if let Some(decomposed) = input.decomposed_amplitudes {
        ensure_axis_len(
            "decomposed_amplitudes",
            "mj",
            decomposed.shape()[0],
            mj_count,
        )?;
        ensure_axis_len("decomposed_amplitudes", "spin", decomposed.shape()[1], 2)?;
        ensure_axis_len(
            "decomposed_amplitudes",
            "lambda2",
            decomposed.shape()[3],
            input.lambda_count,
        )?;
        ensure_axis_len(
            "decomposed_amplitudes",
            "lambda1",
            decomposed.shape()[4],
            input.lambda_count,
        )?;
        let count = decomposed.shape()[2];
        validate_positive_limit("decomposition_count", count)?;
        Some(count)
    } else {
        None
    };

    Ok((mj_count, decomposition_count))
}

fn spin_phase_shift_entry(
    input: GenfmtSpinPhaseShiftInput<'_>,
    energy: usize,
    signed_l_index: usize,
    potential: usize,
) -> Result<Complex, GenfmtError> {
    match input.mode {
        GenfmtReferenceEnergyMode::Header if input.spin_channel_count == 1 => {
            tensor4_complex_entry(
                input.spin_phase_shifts,
                "spin_phase_shifts",
                energy,
                signed_l_index,
                0,
                potential,
            )
        }
        GenfmtReferenceEnergyMode::Header => {
            let last_spin = input.spin_channel_count - 1;
            let first = tensor4_complex_entry(
                input.spin_phase_shifts,
                "spin_phase_shifts",
                energy,
                signed_l_index,
                0,
                potential,
            )?;
            let last = tensor4_complex_entry(
                input.spin_phase_shifts,
                "spin_phase_shifts",
                energy,
                signed_l_index,
                last_spin,
                potential,
            )?;
            let average = (first + last) * 0.5;
            validate_finite_complex("spin_phase_shift_average", average)?;
            Ok(average)
        }
        GenfmtReferenceEnergyMode::SpinChannel { spin_index } => {
            if spin_index >= input.spin_channel_count {
                return Err(GenfmtError::InvalidAngularLimit {
                    name: "spin_index",
                    value: spin_index,
                });
            }
            tensor4_complex_entry(
                input.spin_phase_shifts,
                "spin_phase_shifts",
                energy,
                signed_l_index,
                spin_index,
                potential,
            )
        }
    }
}

fn genfmt_angular_count_for_potential(
    angular_limits: ArrayView2<'_, usize>,
    energy_index: usize,
    potential_index: usize,
) -> Result<usize, GenfmtError> {
    let required_potential_count =
        potential_index
            .checked_add(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "potential_index",
                value: potential_index,
            })?;
    ensure_axis_len(
        "angular_limits",
        "potential",
        angular_limits.shape()[1],
        required_potential_count,
    )?;
    let angular_limit = angular_limits[(energy_index, potential_index)];
    angular_limit
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "angular_limit",
            value: angular_limit,
        })
}

fn path_position(
    positions: ArrayView2<'_, Real>,
    leg_index: usize,
) -> Result<[Real; 3], GenfmtError> {
    let mut position = [0.0; 3];
    for (component, coordinate) in position.iter_mut().enumerate() {
        let value = positions[(leg_index, component)];
        if !value.is_finite() {
            return Err(GenfmtError::NonFinitePathCoordinate {
                leg_index,
                component,
                value,
            });
        }
        *coordinate = value;
    }
    Ok(position)
}

fn vector_between(point: [Real; 3], origin: [Real; 3]) -> [Real; 3] {
    [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ]
}

fn cross(left: [Real; 3], right: [Real; 3]) -> [Real; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn validate_genfmt_spin_channel_count(value: usize) -> Result<(), GenfmtError> {
    if value == 0 || value > 2 {
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channel_count",
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_path_importance_input(input: GenfmtPathImportanceInput<'_>) -> Result<(), GenfmtError> {
    validate_positive_limit("active_energy_count", input.active_energy_count)?;
    validate_finite_scalar("degeneracy", input.degeneracy)?;
    validate_finite_scalar("current_normalization", input.current_normalization)?;
    ensure_axis_len("chi", "energy", input.chi.len(), input.active_energy_count)?;
    ensure_axis_len(
        "momentum_magnitudes",
        "energy",
        input.momentum_magnitudes.len(),
        input.active_energy_count,
    )?;
    if input.edge_start_index >= input.active_energy_count {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "edge_start_index",
            value: input.edge_start_index,
        });
    }
    if input.active_energy_count - input.edge_start_index < 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "active_energy_count",
            value: input.active_energy_count,
        });
    }

    for energy in 0..input.active_energy_count {
        let value = input.momentum_magnitudes[energy];
        if !value.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field: "momentum_magnitudes",
                index: energy,
                value,
            });
        }
    }

    Ok(())
}

fn validate_decomposed_chi_amplitude_phase_input(
    input: GenfmtDecomposedChiAmplitudePhaseInput<'_>,
) -> Result<(), GenfmtError> {
    validate_positive_limit("decomposition_rows", input.decomposed_chi.shape()[0])?;
    validate_positive_limit("decomposition_columns", input.decomposed_chi.shape()[1])?;
    validate_positive_limit("decomposition_energies", input.decomposed_chi.shape()[2])?;
    validate_finite_scalar("phase_epsilon", input.phase_epsilon)?;
    if input.phase_epsilon < 0.0 {
        return Err(GenfmtError::NegativeScalar {
            field: "phase_epsilon",
            value: input.phase_epsilon,
        });
    }
    Ok(())
}

fn ensure_axis_len(
    table: &'static str,
    axis: &'static str,
    length: usize,
    required: usize,
) -> Result<(), GenfmtError> {
    if length >= required {
        Ok(())
    } else {
        Err(GenfmtError::TableAxisTooShort {
            table,
            axis,
            length,
            required,
        })
    }
}

fn genfmt_path_potential_index(
    path_potential_indices: ArrayView1<'_, usize>,
    leg_index: usize,
) -> Result<usize, GenfmtError> {
    ensure_axis_len(
        "path_potential_indices",
        "leg",
        path_potential_indices.len(),
        leg_index + 1,
    )?;
    Ok(path_potential_indices[leg_index])
}

fn genfmt_phase_shifts_for_potential(
    phase_shifts: ArrayView2<'_, Complex>,
    signed_angular_offset: usize,
    potential_index: usize,
    angular_limit: usize,
) -> Result<Array1<Complex>, GenfmtError> {
    if signed_angular_offset < angular_limit {
        return Err(GenfmtError::TableAxisTooShort {
            table: "phase_shifts",
            axis: "negative_l",
            length: signed_angular_offset,
            required: angular_limit,
        });
    }
    let max_row = signed_angular_offset.checked_add(angular_limit).ok_or(
        GenfmtError::InvalidAngularLimit {
            name: "signed_angular_offset",
            value: signed_angular_offset,
        },
    )?;
    ensure_axis_len(
        "phase_shifts",
        "signed_l",
        phase_shifts.shape()[0],
        max_row + 1,
    )?;
    ensure_axis_len(
        "phase_shifts",
        "potential",
        phase_shifts.shape()[1],
        potential_index + 1,
    )?;

    let output_len = checked_double_plus_one("angular_limit", angular_limit)?;
    let mut centered = Array1::<Complex>::zeros(output_len);
    for output_index in 0..output_len {
        let source_row = signed_angular_offset + output_index - angular_limit;
        centered[output_index] =
            table_complex_entry(phase_shifts, "phase_shifts", source_row, potential_index)?;
    }
    Ok(centered)
}

fn table_complex_entry(
    table: ArrayView2<'_, Complex>,
    table_name: &'static str,
    row: usize,
    column: usize,
) -> Result<Complex, GenfmtError> {
    let value = table[(row, column)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table: table_name,
            row,
            column,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex_vector_entry(
    table: ArrayView1<'_, Complex>,
    table_name: &'static str,
    index: usize,
) -> Result<Complex, GenfmtError> {
    let value = table[index];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table: table_name,
            row: index,
            column: 0,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn finite_vector_value(
    table: ArrayView1<'_, Real>,
    field: &'static str,
    index: usize,
) -> Result<Real, GenfmtError> {
    let value = table[index];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteVector {
            field,
            index,
            value,
        })
    }
}

fn validate_fixed_vector(field: &'static str, vector: [Real; 3]) -> Result<(), GenfmtError> {
    for (index, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_nonnegative_doubled_j(field: &'static str, value: i32) -> Result<(), GenfmtError> {
    if value < 0 {
        Err(GenfmtError::InvalidDoubledAngularMomentum { field, value })
    } else {
        Ok(())
    }
}

fn genfmt_output_potential_label(
    label: &str,
    atomic_number: usize,
    index: usize,
) -> Result<String, GenfmtError> {
    let label = label.trim();
    let label = if label.is_empty() {
        atomic_symbol(atomic_number)?
    } else {
        label
    };
    if label.is_empty() || label.len() > 6 || !label.is_ascii() {
        return Err(GenfmtError::InvalidPotentialLabel { index });
    }
    Ok(label.to_string())
}

fn tensor3_complex_entry(
    table: ArrayView3<'_, Complex>,
    table_name: &'static str,
    i0: usize,
    i1: usize,
    i2: usize,
) -> Result<Complex, GenfmtError> {
    let value = table[(i0, i1, i2)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensor3Complex {
            table: table_name,
            i0,
            i1,
            i2,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn tensor4_complex_entry(
    table: ArrayView4<'_, Complex>,
    table_name: &'static str,
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> Result<Complex, GenfmtError> {
    let value = table[(i0, i1, i2, i3)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensorComplex {
            table: table_name,
            i0,
            i1,
            i2,
            i3,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn tensor5_complex_entry(
    table: ArrayView5<'_, Complex>,
    table_name: &'static str,
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
    i4: usize,
) -> Result<Complex, GenfmtError> {
    let value = table[(i0, i1, i2, i3, i4)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensor5Complex {
            table: table_name,
            i0,
            i1,
            i2,
            i3,
            i4,
            real: value.re,
            imaginary: value.im,
        })
    }
}
