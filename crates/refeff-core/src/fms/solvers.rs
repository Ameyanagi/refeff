use super::*;

/// Assemble FEFF's iterative FMS system matrix `1 - T*G0`.
///
/// This is the shared matrix-building branch used by FEFF `ggbi`, `ggrm`, and
/// `ggtf`. It differs from [`fms_lu_scattering`] because the compact
/// single-site T-matrix multiplies `G0` from the left, and it applies FEFF's
/// `toler2` cutoff to individual `G0` elements before adding each contribution.
/// The returned matrix is Fortran-order for LAPACK-compatible downstream use.
pub fn fms_iterative_system_matrix(
    input: FmsIterativeSystemInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    fms_compact_tg_work_matrix(input, Complex32::new(-1.0, 0.0), Complex32::new(1.0, 0.0))
}

fn fms_graves_morris_system_matrix(
    input: FmsIterativeSystemInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    fms_compact_tg_work_matrix(input, Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0))
}

fn fms_compact_tg_work_matrix(
    input: FmsIterativeSystemInput<'_>,
    factor: Complex32,
    diagonal: Complex32,
) -> Result<Array2<Complex32>, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    if !input.zero_tolerance.is_finite() || input.zero_tolerance < 0.0 {
        return Err(FmsError::InvalidTolerance {
            name: "toler2",
            value: input.zero_tolerance,
        });
    }
    ensure_square_table("g0", input.free_propagator, input.states.len())?;
    ensure_axis_len(
        "tmatrx",
        "spin_band",
        input.t_matrix.shape()[0],
        input.spin_channels - 1,
    )?;
    ensure_axis_len(
        "tmatrx",
        "state",
        input.t_matrix.shape()[1],
        input.states.len() - 1,
    )?;

    let mut system_matrix = Array2::zeros((input.states.len(), input.states.len()).f());
    for column in 0..input.states.len() {
        for (row, &state) in input.states.iter().enumerate() {
            ensure_state_spin(state.spin, input.spin_channels)?;
            let diagonal_g0 = input.free_propagator[(row, column)];
            if diagonal_g0.norm() > input.zero_tolerance {
                system_matrix[(row, column)] += factor * input.t_matrix[(0, row)] * diagonal_g0;
            }

            if input.spin_channels == 2
                && let Some(partner) = fms_spin_partner_index(state, row, input.states.len())?
            {
                let spin_flip_g0 = input.free_propagator[(partner, column)];
                if spin_flip_g0.norm() > input.zero_tolerance {
                    system_matrix[(row, column)] +=
                        factor * input.t_matrix[(1, partner)] * spin_flip_g0;
                }
            }
        }
        system_matrix[(column, column)] += diagonal;
    }

    Ok(system_matrix)
}

/// Dispatch FEFF's compact FMS scattering branches.
///
/// This mirrors the final `minv` branch in `fmspack.f90` after setup and
/// matrix assembly are complete. The LU branch ignores iterative tolerances
/// and `lcalc`, while iterative branches return FEFF's reported
/// multiple-scattering order in [`FmsScatteringResult::multiple_scattering_order`].
pub fn fms_scattering(input: FmsScatteringInput<'_>) -> Result<FmsScatteringResult, FmsError> {
    match input.method {
        FmsScatteringMethod::Lu => {
            let result = fms_lu_scattering(FmsLuInput {
                states: input.states,
                calculate_full_scattering: input.calculate_full_scattering,
                spin_channels: input.spin_channels,
                global_lmax: input.global_lmax,
                potential_lmax: input.potential_lmax,
                representative_offsets: input.representative_offsets,
                potential_start: input.potential_start,
                potential_end: input.potential_end,
                free_propagator: input.free_propagator,
                t_matrix: input.t_matrix,
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: result.full_scattering,
                multiple_scattering_order: None,
            })
        }
        FmsScatteringMethod::BiCgStab => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_bicgstab_scattering(FmsBiCgStabInput {
                states: input.states,
                spin_channels: input.spin_channels,
                global_lmax: input.global_lmax,
                potential_lmax: input.potential_lmax,
                representative_offsets: input.representative_offsets,
                potential_start: input.potential_start,
                potential_end: input.potential_end,
                free_propagator: input.free_propagator,
                t_matrix: input.t_matrix,
                calculated_l: input.calculated_l,
                convergence_tolerance: input.convergence_tolerance,
                zero_tolerance: input.zero_tolerance,
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
        FmsScatteringMethod::Recursion => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_recursion_scattering(FmsRecursionInput {
                states: input.states,
                spin_channels: input.spin_channels,
                global_lmax: input.global_lmax,
                potential_lmax: input.potential_lmax,
                representative_offsets: input.representative_offsets,
                potential_start: input.potential_start,
                potential_end: input.potential_end,
                free_propagator: input.free_propagator,
                t_matrix: input.t_matrix,
                calculated_l: input.calculated_l,
                convergence_tolerance: input.convergence_tolerance,
                zero_tolerance: input.zero_tolerance,
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
        FmsScatteringMethod::GravesMorris => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_graves_morris_scattering(FmsGravesMorrisInput {
                states: input.states,
                spin_channels: input.spin_channels,
                global_lmax: input.global_lmax,
                potential_lmax: input.potential_lmax,
                representative_offsets: input.representative_offsets,
                potential_start: input.potential_start,
                potential_end: input.potential_end,
                free_propagator: input.free_propagator,
                t_matrix: input.t_matrix,
                calculated_l: input.calculated_l,
                convergence_tolerance: input.convergence_tolerance,
                zero_tolerance: input.zero_tolerance,
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
        FmsScatteringMethod::Tfqmr => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_tfqmr_scattering(FmsTfqmrInput {
                states: input.states,
                spin_channels: input.spin_channels,
                global_lmax: input.global_lmax,
                potential_lmax: input.potential_lmax,
                representative_offsets: input.representative_offsets,
                potential_start: input.potential_start,
                potential_end: input.potential_end,
                free_propagator: input.free_propagator,
                t_matrix: input.t_matrix,
                calculated_l: input.calculated_l,
                convergence_tolerance: input.convergence_tolerance,
                zero_tolerance: input.zero_tolerance,
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
    }
}

/// Port of FEFF `ggbi`: BiCGStab-style iterative FMS scattering.
///
/// FEFF's `ggbi` solves columns of `(1 - T*G0) * x = e_j` and packs
/// `G0*x` into `gg`. This implementation preserves the FEFF single-precision
/// control flow and compact spin-orbit T-matrix storage, while returning
/// explicit errors for invalid tolerances or zero solver denominators.
pub fn fms_bicgstab_scattering(input: FmsBiCgStabInput<'_>) -> Result<FmsBiCgStabResult, FmsError> {
    let result = fms_iterative_scattering(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        fms_bicgstab_solve,
    )?;

    Ok(FmsBiCgStabResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `ggrm`: recursion-method iterative FMS scattering.
///
/// This branch solves the same `(1 - T*G0) * x = e_j` systems as
/// [`fms_bicgstab_scattering`], but follows FEFF's bi-orthogonal recursion
/// update with a bounded restart loop and explicit breakdown errors.
pub fn fms_recursion_scattering(
    input: FmsRecursionInput<'_>,
) -> Result<FmsRecursionResult, FmsError> {
    let result = fms_iterative_scattering(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        fms_recursion_solve,
    )?;

    Ok(FmsRecursionResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `gggm`: Graves-Morris/Salam iterative FMS scattering.
///
/// Unlike the other iterative branches, FEFF's `gggm` builds the compact
/// `T*G0` work matrix directly and applies the GMS update to recover
/// `(1 - T*G0)^-1 * e_j` before packing `G0*x` into `gg`.
pub fn fms_graves_morris_scattering(
    input: FmsGravesMorrisInput<'_>,
) -> Result<FmsGravesMorrisResult, FmsError> {
    let system_matrix = fms_graves_morris_system_matrix(FmsIterativeSystemInput {
        states: input.states,
        spin_channels: input.spin_channels,
        free_propagator: input.free_propagator,
        t_matrix: input.t_matrix,
        zero_tolerance: input.zero_tolerance,
    })?;
    let result = fms_iterative_scattering_with_system(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        system_matrix,
        fms_graves_morris_solve,
    )?;

    Ok(FmsGravesMorrisResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `ggtf`: TFQMR iterative FMS scattering.
///
/// This branch solves the same `(1 - T*G0) * x = e_j` systems as
/// [`fms_bicgstab_scattering`], but uses FEFF's TFQMR iteration from `ggtf`.
pub fn fms_tfqmr_scattering(input: FmsTfqmrInput<'_>) -> Result<FmsTfqmrResult, FmsError> {
    let result = fms_iterative_scattering(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        fms_tfqmr_solve,
    )?;

    Ok(FmsTfqmrResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `gglu`: solve `(1 - G0*T) * G = G0` and pack `gg`.
///
/// This is the LU branch used by FEFF FMS. It preserves the compact `tmatrx`
/// multiplication, including the spin-orbit off-diagonal band when
/// `spin_channels == 2`, then solves with FEFF-compatible single-precision
/// complex LU factors from `refeff-linalg`'s `faer` backend.
pub fn fms_lu_scattering(input: FmsLuInput<'_>) -> Result<FmsLuResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    ensure_axis_len(
        "states",
        "potential_start",
        input.representative_offsets.len(),
        input.potential_start,
    )?;
    ensure_axis_len(
        "states",
        "potential_end",
        input.representative_offsets.len(),
        input.potential_end,
    )?;
    if input.potential_start > input.potential_end {
        return Err(FmsError::TableIndexOutOfRange {
            table: "potential_range",
            axis: "potential",
            index: input.potential_start,
        });
    }
    ensure_square_table("g0", input.free_propagator, input.states.len())?;
    ensure_axis_len(
        "tmatrx",
        "spin_band",
        input.t_matrix.shape()[0],
        input.spin_channels - 1,
    )?;
    ensure_axis_len(
        "tmatrx",
        "state",
        input.t_matrix.shape()[1],
        input.states.len() - 1,
    )?;

    let system_matrix = fms_lu_system_matrix(
        input.states,
        input.spin_channels,
        input.free_propagator,
        input.t_matrix,
    )?;
    let lu = complex32_faer_lu_factor(system_matrix.view())?;
    let channel_count = input
        .global_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(input.spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "global_lmax",
            value: input.global_lmax,
            lx: input.global_lmax,
        })?;
    let mut scattering = Array3::zeros(
        (
            channel_count,
            channel_count,
            input.representative_offsets.len(),
        )
            .f(),
    );

    for potential in input.potential_start..=input.potential_end {
        let lmax = potential_lmax_for(input.potential_lmax, potential)?.min(input.global_lmax);
        let ipart = lmax
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .and_then(|value| value.checked_mul(input.spin_channels))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lipotx",
                value: lmax,
                lx: input.global_lmax,
            })?;
        let offset = representative_offset(input.representative_offsets, potential)?;
        ensure_axis_len(
            "g0",
            "representative_state",
            input.free_propagator.shape()[1],
            offset,
        )?;
        ensure_axis_len(
            "g0",
            "representative_block",
            input.free_propagator.shape()[1],
            offset + ipart - 1,
        )?;

        let mut rhs = Array2::zeros((input.states.len(), ipart).f());
        for row in 0..input.states.len() {
            for column in 0..ipart {
                rhs[(row, column)] = input.free_propagator[(row, offset + column)];
            }
        }
        let solved = complex32_faer_lu_solve(&lu, rhs.view())?;
        for column in 0..ipart {
            for row in 0..ipart {
                scattering[(row, column, potential)] = solved[(offset + row, column)];
            }
        }
    }

    let full_scattering = if input.calculate_full_scattering {
        Some(complex32_faer_lu_solve(&lu, input.free_propagator)?)
    } else {
        None
    };

    Ok(FmsLuResult {
        system_matrix,
        scattering,
        full_scattering,
    })
}

/// Port of FEFF `gglufullpot`: LU FMS scattering with a full T-matrix.
///
/// FEFF's full-potential branch accepts `tmatrx(state,state)` rather than the
/// compact spin-band table used by [`fms_lu_scattering`]. The assembled work
/// matrix follows the original `gglufullpot` diagonal assignment before the
/// pure-Rust LU solve.
pub fn fms_full_potential_lu_scattering(
    input: FmsFullPotentialLuInput<'_>,
) -> Result<FmsFullPotentialLuResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    ensure_axis_len(
        "states",
        "potential_start",
        input.representative_offsets.len(),
        input.potential_start,
    )?;
    ensure_axis_len(
        "states",
        "potential_end",
        input.representative_offsets.len(),
        input.potential_end,
    )?;
    if input.potential_start > input.potential_end {
        return Err(FmsError::TableIndexOutOfRange {
            table: "potential_range",
            axis: "potential",
            index: input.potential_start,
        });
    }
    ensure_square_table("g0", input.free_propagator, input.states.len())?;
    ensure_square_table("tmatrx", input.t_matrix, input.states.len())?;
    for &state in input.states {
        ensure_state_spin(state.spin, input.spin_channels)?;
    }

    let system_matrix =
        fms_full_potential_lu_system_matrix(input.states, input.free_propagator, input.t_matrix)?;
    let lu = complex32_faer_lu_factor(system_matrix.view())?;
    let channel_count = input
        .global_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(input.spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "global_lmax",
            value: input.global_lmax,
            lx: input.global_lmax,
        })?;
    let mut scattering = Array3::zeros(
        (
            channel_count,
            channel_count,
            input.representative_offsets.len(),
        )
            .f(),
    );

    for potential in input.potential_start..=input.potential_end {
        let lmax = potential_lmax_for(input.potential_lmax, potential)?.min(input.global_lmax);
        let ipart = lmax
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .and_then(|value| value.checked_mul(input.spin_channels))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lipotx",
                value: lmax,
                lx: input.global_lmax,
            })?;
        let offset = representative_offset(input.representative_offsets, potential)?;
        ensure_axis_len(
            "g0",
            "representative_state",
            input.free_propagator.shape()[1],
            offset,
        )?;
        ensure_axis_len(
            "g0",
            "representative_block",
            input.free_propagator.shape()[1],
            offset + ipart - 1,
        )?;

        let mut rhs = Array2::zeros((input.states.len(), ipart).f());
        for row in 0..input.states.len() {
            for column in 0..ipart {
                rhs[(row, column)] = input.free_propagator[(row, offset + column)];
            }
        }
        let solved = complex32_faer_lu_solve(&lu, rhs.view())?;
        for column in 0..ipart {
            for row in 0..ipart {
                scattering[(row, column, potential)] = solved[(offset + row, column)];
            }
        }
    }

    Ok(FmsFullPotentialLuResult {
        system_matrix,
        scattering,
    })
}
