use ndarray::Axis;

use super::*;

/// Accumulate valence LDOS and radial density from FEFF scattering terms.
///
/// This ports `POT/ff2g.f90`. The routine first folds the single-precision
/// scattering trace into the embedded LDOS, then integrates the current and
/// previous energy endpoints. For `energy_index == 1`, FEFF initializes the
/// previous-energy work arrays from the current values; for later energies it
/// preserves the caller-provided previous state.
pub fn update_valence_density(
    input: ValenceDensityUpdateInput<'_>,
) -> Result<ValenceDensityUpdate, DensityError> {
    validate_valence_density_input(input)?;

    let l_count = input.scattering_trace.len();
    let radial_count = input.last_radial_index;
    let potential = input.potential_index;
    let mut embedded_ldos = input.embedded_ldos.to_owned();
    let mut previous_ldos = input.previous_ldos.to_owned();
    let mut embedded_density = input.embedded_density.to_owned();
    let mut previous_density = input.previous_density.to_owned();
    let mut valence_density = input.valence_density.to_owned();
    let mut occupancy_by_l = input.occupancy_by_l.to_owned();
    let mut left_sum = input.left_sum;
    let mut right_sum = input.right_sum;
    let mut total_electron_count = input.total_electron_count;

    for angular in 0..l_count {
        embedded_ldos[(angular, potential)] +=
            widen_complex32(input.scattering_trace[angular]) * input.scattering_ldos[angular];
        if input.energy_index == 1 {
            previous_ldos[(angular, potential)] = embedded_ldos[(angular, potential)];
        }
    }

    let mut left_step = input.current_energy - input.previous_energy;
    let mut right_step = left_step;
    if input.current_floor == 1 {
        right_step -= Complex::new(0.0, 2.0 * input.current_energy.im);
    }
    if input.previous_floor == 1 {
        left_step += Complex::new(0.0, 2.0 * input.previous_energy.im);
    }

    for angular in 0..l_count {
        if includes_angular_channel(angular, input.include_high_l) {
            left_sum += previous_ldos[(angular, potential)] * (2.0 * input.potential_multiplicity);
            right_sum += embedded_ldos[(angular, potential)] * (2.0 * input.potential_multiplicity);
            occupancy_by_l[angular] += (embedded_ldos[(angular, potential)] * right_step
                + previous_ldos[(angular, potential)] * left_step)
                .im;
            total_electron_count += occupancy_by_l[angular] * input.potential_multiplicity;
        }
    }

    for angular in 0..l_count {
        if includes_angular_channel(angular, input.include_high_l) {
            let trace = widen_complex32(input.scattering_trace[angular]);
            for radial in 0..radial_count {
                embedded_density[radial] += trace * input.scattering_density[(radial, angular)];
                if input.energy_index == 1 {
                    previous_density[radial] = embedded_density[radial];
                }
            }
        }
    }

    for radial in 0..radial_count {
        valence_density[radial] +=
            (embedded_density[radial] * right_step + previous_density[radial] * left_step).im;
    }

    Ok(ValenceDensityUpdate {
        embedded_ldos,
        previous_ldos,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
        left_sum,
        right_sum,
        total_electron_count,
    })
}

/// Accumulate one FEFF `POT/scmt.f90` energy point across all potentials.
///
/// SCMT resets `xntot`, `fl`, and `fr` at each energy point, then calls
/// `ff2g` once per potential. This adapter owns that all-potential loop while
/// leaving radial-solver and FMS-trace generation to the caller.
pub fn accumulate_pot_scf_energy_point(
    input: PotScfEnergyPointInput<'_>,
) -> Result<PotScfEnergyPoint, DensityError> {
    validate_pot_scf_energy_point_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let mut embedded_ldos = input.embedded_ldos.to_owned();
    let mut previous_ldos = input.previous_ldos.to_owned();
    let mut embedded_density = input.embedded_density.to_owned();
    let mut previous_density = input.previous_density.to_owned();
    let mut valence_density = input.valence_density.to_owned();
    let mut occupancy_by_l = input.occupancy_by_l.to_owned();
    let mut total_electron_count = 0.0;
    let mut left_sum = Complex::new(0.0, 0.0);
    let mut right_sum = Complex::new(0.0, 0.0);

    for potential in 0..potential_count {
        let update = update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: input.scattering_trace.index_axis(Axis(1), potential),
            potential_index: potential,
            energy_index: input.energy_index,
            last_radial_index: input.last_indices[potential],
            scattering_ldos: input.scattering_ldos.index_axis(Axis(1), potential),
            embedded_ldos: embedded_ldos.view(),
            previous_ldos: previous_ldos.view(),
            scattering_density: input.scattering_density.index_axis(Axis(2), potential),
            embedded_density: embedded_density.index_axis(Axis(1), potential),
            previous_density: previous_density.index_axis(Axis(1), potential),
            valence_density: valence_density.index_axis(Axis(1), potential),
            occupancy_by_l: occupancy_by_l.index_axis(Axis(1), potential),
            current_energy: input.current_energy,
            previous_energy: input.previous_energy,
            potential_multiplicity: input.potential_multiplicities[potential],
            current_floor: input.current_floor,
            previous_floor: input.previous_floor,
            left_sum,
            right_sum,
            total_electron_count,
            include_high_l: input.include_high_l,
        })?;

        embedded_ldos = update.embedded_ldos;
        previous_ldos = update.previous_ldos;
        embedded_density
            .index_axis_mut(Axis(1), potential)
            .assign(&update.embedded_density);
        previous_density
            .index_axis_mut(Axis(1), potential)
            .assign(&update.previous_density);
        valence_density
            .index_axis_mut(Axis(1), potential)
            .assign(&update.valence_density);
        occupancy_by_l
            .index_axis_mut(Axis(1), potential)
            .assign(&update.occupancy_by_l);
        left_sum = update.left_sum;
        right_sum = update.right_sum;
        total_electron_count = update.total_electron_count;
    }

    validate_complex_values(
        "pot_scf_energy_point_embedded_ldos",
        embedded_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_previous_ldos",
        previous_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_embedded_density",
        embedded_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_previous_density",
        previous_density.iter().copied(),
    )?;
    validate_real_table_values(
        "pot_scf_energy_point_valence_density",
        valence_density.view(),
    )?;
    validate_real_table_values("pot_scf_energy_point_occupancy_by_l", occupancy_by_l.view())?;
    validate_real_scalar(
        "pot_scf_energy_point_total_electron_count",
        total_electron_count,
    )?;
    validate_complex_scalar("pot_scf_energy_point_left_sum", left_sum)?;
    validate_complex_scalar("pot_scf_energy_point_right_sum", right_sum)?;

    Ok(PotScfEnergyPoint {
        embedded_ldos,
        previous_ldos,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
        total_electron_count,
        left_sum,
        right_sum,
    })
}

/// Run the source-backed FEFF `POT/scmt.f90` contour loop over supplied work arrays.
///
/// The caller supplies one row of radial-solver and FMS work arrays for each
/// energy point in loop order. This driver owns the FEFF state transitions:
/// `xrhocp`/`yrhocp` copies, per-energy `ff2g` accumulation, `xndif`
/// tracking, contour stepping, and final Fermi end-cap correction when the
/// lowest-floor bracket is found.
pub fn run_pot_scf_contour(
    input: PotScfContourRunInput<'_>,
) -> Result<PotScfContourRun, DensityError> {
    let (point_count, angular_count, radial_count, potential_count) =
        validate_pot_scf_contour_run_input(input)?;

    let mut embedded_ldos = Array2::<Complex>::zeros((angular_count, potential_count));
    let mut previous_ldos = Array2::<Complex>::zeros((angular_count, potential_count));
    let mut embedded_density = Array2::<Complex>::zeros((radial_count, potential_count));
    let mut previous_density = Array2::<Complex>::zeros((radial_count, potential_count));
    let mut valence_density = Array2::<Real>::zeros((radial_count, potential_count));
    let mut occupancy_by_l = Array2::<Real>::zeros((angular_count, potential_count));

    let mut current_energy = input.energy_grid[0];
    let mut previous_energy = Complex::new(current_energy.re, 0.0);
    let mut current_floor = input.floor_count;
    let mut previous_floor = input.floor_count;
    let mut direction = 1;
    let mut can_step_up = false;
    let mut current_electron_delta = 0.0;
    let mut previous_electron_delta = 0.0;
    let mut total_electron_count = 0.0;
    let mut left_sum = Complex::new(0.0, 0.0);
    let mut right_sum = Complex::new(0.0, 0.0);

    for point_index in 0..point_count {
        let energy_index = point_index + 1;
        validate_pot_scf_contour_source_energy(
            point_index,
            input.source_energies[point_index],
            current_energy,
        )?;
        if point_index > 0 {
            previous_ldos.assign(&embedded_ldos);
            previous_density.assign(&embedded_density);
        }

        embedded_ldos.assign(&input.embedded_ldos_source.index_axis(Axis(0), point_index));
        embedded_density.assign(
            &input
                .embedded_density_source
                .index_axis(Axis(0), point_index),
        );

        let point = accumulate_pot_scf_energy_point(PotScfEnergyPointInput {
            energy_index,
            current_energy,
            previous_energy,
            current_floor: contour_floor_as_i32(current_floor)?,
            previous_floor: contour_floor_as_i32(previous_floor)?,
            highest_potential_index: input.highest_potential_index,
            last_indices: input.last_indices,
            potential_multiplicities: input.potential_multiplicities,
            scattering_trace: input.scattering_trace.index_axis(Axis(0), point_index),
            scattering_ldos: input.scattering_ldos.index_axis(Axis(0), point_index),
            embedded_ldos: embedded_ldos.view(),
            previous_ldos: previous_ldos.view(),
            scattering_density: input.scattering_density.index_axis(Axis(0), point_index),
            embedded_density: embedded_density.view(),
            previous_density: previous_density.view(),
            valence_density: valence_density.view(),
            occupancy_by_l: occupancy_by_l.view(),
            include_high_l: input.include_high_l,
        })?;

        embedded_ldos = point.embedded_ldos;
        previous_ldos = point.previous_ldos;
        embedded_density = point.embedded_density;
        previous_density = point.previous_density;
        valence_density = point.valence_density;
        occupancy_by_l = point.occupancy_by_l;
        total_electron_count = point.total_electron_count;
        left_sum = point.left_sum;
        right_sum = point.right_sum;

        if energy_index != 1 {
            previous_electron_delta = current_electron_delta;
        }
        current_electron_delta = total_electron_count - input.electron_count_target;

        let step = pot_scf_contour_step(PotScfContourStepInput {
            first_scmt_call: input.first_scmt_call,
            energy_index,
            active_energy_count: input.active_energy_count,
            floor_count: input.floor_count,
            energy_grid: input.energy_grid,
            steps: input.steps,
            current_energy,
            previous_energy,
            current_floor,
            previous_floor,
            direction,
            can_step_up,
            current_electron_delta,
            previous_electron_delta,
        })?;

        if step.status == PotScfContourStepStatus::Bracketed {
            let endpoint = finish_pot_scf_fermi_endpoint(PotScfFermiEndpointInput {
                current_energy,
                previous_energy,
                current_electron_delta,
                previous_electron_delta,
                left_sum,
                right_sum,
                highest_potential_index: input.highest_potential_index,
                last_indices: input.last_indices,
                embedded_ldos: embedded_ldos.view(),
                previous_ldos: previous_ldos.view(),
                embedded_density: embedded_density.view(),
                previous_density: previous_density.view(),
                valence_density: valence_density.view(),
                occupancy_by_l: occupancy_by_l.view(),
                include_high_l: input.include_high_l,
            })?;
            return Ok(PotScfContourRun {
                status: PotScfContourRunStatus::Bracketed,
                energy_points_used: energy_index,
                current_energy,
                previous_energy,
                current_floor,
                previous_floor,
                direction,
                can_step_up,
                current_electron_delta,
                previous_electron_delta,
                total_electron_count,
                left_sum,
                right_sum,
                fermi_energy: Some(endpoint.fermi_energy),
                interpolation_fraction: Some(endpoint.interpolation_fraction),
                embedded_ldos,
                previous_ldos,
                embedded_density,
                previous_density,
                valence_density: endpoint.valence_density,
                occupancy_by_l: endpoint.occupancy_by_l,
            });
        }

        current_energy = step.current_energy;
        previous_energy = step.previous_energy;
        current_floor = step.current_floor;
        previous_floor = step.previous_floor;
        direction = step.direction;
        can_step_up = step.can_step_up;
    }

    Ok(PotScfContourRun {
        status: PotScfContourRunStatus::NeedsMoreSourcePoints,
        energy_points_used: point_count,
        current_energy,
        previous_energy,
        current_floor,
        previous_floor,
        direction,
        can_step_up,
        current_electron_delta,
        previous_electron_delta,
        total_electron_count,
        left_sum,
        right_sum,
        fermi_energy: None,
        interpolation_fraction: None,
        embedded_ldos,
        previous_ldos,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
    })
}

/// Apply FEFF `POT/scmt.f90` Fermi end-cap corrections after the contour bracket.
///
/// FEFF refines the interpolation fraction `a` between current energy `ee` and
/// previous energy `ep`, then applies the same endpoint integral correction to
/// angular occupations `xnmues` and radial valence density `rhoval`.
pub fn finish_pot_scf_fermi_endpoint(
    input: PotScfFermiEndpointInput<'_>,
) -> Result<PotScfFermiEndpoint, DensityError> {
    validate_pot_scf_fermi_endpoint_input(input)?;

    let interpolation_fraction = pot_scf_fermi_endpoint_fraction(input)?;
    validate_real_scalar(
        "pot_scf_fermi_interpolation_fraction",
        interpolation_fraction,
    )?;
    let fermi_energy = (input.current_energy * (1.0 - interpolation_fraction)
        + input.previous_energy * interpolation_fraction)
        .re;
    validate_real_scalar("pot_scf_fermi_energy", fermi_energy)?;

    let mut valence_density = input.valence_density.to_owned();
    let mut occupancy_by_l = input.occupancy_by_l.to_owned();
    let potential_count = input.highest_potential_index + 1;
    let angular_count = input.embedded_ldos.nrows();

    for potential in 0..potential_count {
        for angular in 0..angular_count {
            if includes_angular_channel(angular, input.include_high_l) {
                let previous = input.previous_ldos[(angular, potential)] * 2.0;
                let current = input.embedded_ldos[(angular, potential)] * 2.0;
                let correction = pot_scf_fermi_endpoint_correction(
                    input.current_energy,
                    input.previous_energy,
                    current,
                    previous,
                    interpolation_fraction,
                )?;
                occupancy_by_l[(angular, potential)] += interpolation_fraction * correction;
                validate_real_scalar(
                    "pot_scf_fermi_occupancy",
                    occupancy_by_l[(angular, potential)],
                )?;
            }
        }

        for radial in 0..input.last_indices[potential] {
            let previous = input.previous_density[(radial, potential)] * 2.0;
            let current = input.embedded_density[(radial, potential)] * 2.0;
            let correction = pot_scf_fermi_endpoint_correction(
                input.current_energy,
                input.previous_energy,
                current,
                previous,
                interpolation_fraction,
            )?;
            valence_density[(radial, potential)] += interpolation_fraction * correction;
            validate_real_scalar(
                "pot_scf_fermi_valence_density",
                valence_density[(radial, potential)],
            )?;
        }
    }

    Ok(PotScfFermiEndpoint {
        fermi_energy,
        interpolation_fraction,
        valence_density,
        occupancy_by_l,
    })
}

fn pot_scf_fermi_endpoint_fraction(
    input: PotScfFermiEndpointInput<'_>,
) -> Result<Real, DensityError> {
    if input.current_electron_delta == 0.0 {
        return Ok(0.0);
    }

    let denominator = input.current_electron_delta - input.previous_electron_delta;
    validate_nonzero_real_scalar("pot_scf_fermi_delta_difference", denominator)?;
    let mut fraction = input.current_electron_delta / denominator;
    validate_real_scalar("pot_scf_fermi_initial_fraction", fraction)?;

    for _ in 0..4 {
        let correction = pot_scf_fermi_endpoint_correction(
            input.current_energy,
            input.previous_energy,
            input.right_sum,
            input.left_sum,
            fraction,
        )?;
        validate_nonzero_real_scalar("pot_scf_fermi_newton_correction", correction)?;
        let residual = input.current_electron_delta + fraction * correction;
        validate_real_scalar("pot_scf_fermi_newton_residual", residual)?;
        fraction -= residual / correction;
        validate_real_scalar("pot_scf_fermi_newton_fraction", fraction)?;
    }

    Ok(fraction)
}

fn pot_scf_fermi_endpoint_correction(
    current_energy: Complex,
    previous_energy: Complex,
    current_value: Complex,
    previous_value: Complex,
    fraction: Real,
) -> Result<Real, DensityError> {
    let interpolated = previous_value * fraction + current_value * (1.0 - fraction);
    let imaginary = Complex::new(0.0, 1.0);
    let correction = ((previous_energy - current_energy) * (current_value + interpolated) / 2.0
        + imaginary * current_energy.im * (current_value - previous_value))
        .im;
    validate_real_scalar("pot_scf_fermi_endpoint_correction", correction)?;
    Ok(correction)
}

/// Advance FEFF `POT/scmt.f90` contour-search state after one energy point.
///
/// This ports the branch that decides whether SCMT should consume another
/// prebuilt `emg` point, move horizontally on the current floor, move up/down
/// between floors, or stop because the lowest-floor Fermi bracket was found.
pub fn pot_scf_contour_step(
    input: PotScfContourStepInput<'_>,
) -> Result<PotScfContourStep, DensityError> {
    validate_pot_scf_contour_step_input(input)?;

    let mut output = PotScfContourStep {
        status: PotScfContourStepStatus::Continue,
        previous_energy: input.current_energy,
        current_energy: input.current_energy,
        current_floor: input.current_floor,
        previous_floor: input.previous_floor,
        direction: input.direction,
        can_step_up: input.can_step_up,
    };

    if (!input.first_scmt_call && input.energy_index < input.active_energy_count)
        || (input.first_scmt_call && input.energy_index < input.floor_count)
    {
        output.current_energy = input.energy_grid[input.energy_index];
        if input.energy_index == input.active_energy_count - 1 {
            output.previous_floor = 2;
            output.current_floor = 1;
        }
        validate_pot_scf_contour_step_output(output)?;
        return Ok(output);
    }

    if input.first_scmt_call && input.energy_index == input.floor_count {
        output.can_step_up = false;
        output.direction = if input.current_electron_delta > 0.0 {
            -1
        } else {
            1
        };
        output.current_energy = input.current_energy
            + Complex::new(
                input.steps[input.current_floor - 1] * output.direction as Real,
                0.0,
            );
        validate_pot_scf_contour_step_output(output)?;
        return Ok(output);
    }

    if !input.first_scmt_call && input.energy_index == input.active_energy_count {
        output.can_step_up = true;
        output.previous_floor = 1;
        output.current_floor = 1;
        output.direction = if input.current_electron_delta < 0.0 {
            1
        } else {
            -1
        };
        output.current_energy = input.current_energy
            + Complex::new(
                input.steps[output.current_floor - 1] * output.direction as Real,
                0.0,
            );
        validate_pot_scf_contour_step_output(output)?;
        return Ok(output);
    }

    if input.previous_floor == 1
        && input.current_floor == 1
        && input.previous_electron_delta * input.current_electron_delta <= 0.0
    {
        output.status = PotScfContourStepStatus::Bracketed;
        output.previous_energy = input.previous_energy;
        output.current_energy = input.current_energy;
        validate_pot_scf_contour_step_output(output)?;
        return Ok(output);
    }

    if input.current_floor == input.previous_floor {
        if input.previous_electron_delta * input.current_electron_delta <= 0.0 {
            output.can_step_up = false;
            output.previous_floor = input.current_floor;
            output.current_floor =
                input
                    .current_floor
                    .checked_sub(1)
                    .ok_or(DensityError::InvalidIndex {
                        name: "pot_scf_contour_floor",
                        index: input.current_floor,
                    })?;
            if output.current_floor == 0 {
                return Err(DensityError::InvalidIndex {
                    name: "pot_scf_contour_floor",
                    index: output.current_floor,
                });
            }
            output.current_energy = Complex::new(
                input.current_energy.re,
                4.0 * input.steps[output.current_floor - 1],
            );
        } else if input.current_electron_delta.abs()
            > 10.0 * (input.current_electron_delta - input.previous_electron_delta).abs()
            && input.can_step_up
        {
            output.previous_floor = input.current_floor;
            if input.current_floor < input.floor_count {
                output.current_floor = input.current_floor + 1;
                output.current_energy = Complex::new(
                    input.current_energy.re,
                    4.0 * input.steps[output.current_floor - 1],
                );
            } else {
                output.current_energy = input.current_energy
                    + Complex::new(
                        input.steps[input.current_floor - 1] * input.direction as Real,
                        0.0,
                    );
            }
        } else {
            output.current_energy = input.current_energy
                + Complex::new(
                    input.steps[input.current_floor - 1] * input.direction as Real,
                    0.0,
                );
        }
    } else {
        output.direction = if input.current_electron_delta < 0.0 {
            1
        } else {
            -1
        };
        output.previous_floor = input.current_floor;
        output.current_energy = input.current_energy
            + Complex::new(
                input.steps[input.current_floor - 1] * output.direction as Real,
                0.0,
            );
    }

    validate_pot_scf_contour_step_output(output)?;
    Ok(output)
}

fn validate_valence_density_input(
    input: ValenceDensityUpdateInput<'_>,
) -> Result<(), DensityError> {
    if input.energy_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "energy_index",
            index: input.energy_index,
        });
    }
    if input.last_radial_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "last_radial_index",
            index: input.last_radial_index,
        });
    }

    let l_count = input.scattering_trace.len();
    ensure_length_match(
        "scattering_trace",
        l_count,
        "scattering_ldos",
        input.scattering_ldos.len(),
    )?;
    ensure_len("occupancy_by_l", input.occupancy_by_l.len(), l_count)?;
    ensure_len(
        "embedded_density",
        input.embedded_density.len(),
        input.last_radial_index,
    )?;
    ensure_len(
        "previous_density",
        input.previous_density.len(),
        input.last_radial_index,
    )?;
    ensure_len(
        "valence_density",
        input.valence_density.len(),
        input.last_radial_index,
    )?;
    ensure_shape(
        "embedded_ldos",
        input.embedded_ldos.shape(),
        l_count,
        input.potential_index + 1,
    )?;
    ensure_shape(
        "previous_ldos",
        input.previous_ldos.shape(),
        l_count,
        input.potential_index + 1,
    )?;
    ensure_shape(
        "scattering_density",
        input.scattering_density.shape(),
        input.last_radial_index,
        l_count,
    )?;

    validate_complex32_values("scattering_trace", input.scattering_trace)?;
    validate_complex_values("scattering_ldos", input.scattering_ldos.iter().copied())?;
    validate_complex_values("embedded_ldos", input.embedded_ldos.iter().copied())?;
    validate_complex_values("previous_ldos", input.previous_ldos.iter().copied())?;
    validate_complex_values(
        "scattering_density",
        input.scattering_density.iter().copied(),
    )?;
    validate_complex_values("embedded_density", input.embedded_density.iter().copied())?;
    validate_complex_values("previous_density", input.previous_density.iter().copied())?;
    validate_real_values("valence_density", input.valence_density)?;
    validate_real_values("occupancy_by_l", input.occupancy_by_l)?;
    validate_complex_scalar("current_energy", input.current_energy)?;
    validate_complex_scalar("previous_energy", input.previous_energy)?;
    validate_complex_scalar("left_sum", input.left_sum)?;
    validate_complex_scalar("right_sum", input.right_sum)?;
    validate_real_scalar("potential_multiplicity", input.potential_multiplicity)?;
    validate_real_scalar("total_electron_count", input.total_electron_count)?;

    Ok(())
}

fn validate_pot_scf_energy_point_input(
    input: PotScfEnergyPointInput<'_>,
) -> Result<(), DensityError> {
    if input.energy_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_energy_point_energy_index",
            index: input.energy_index,
        });
    }
    validate_complex_scalar("pot_scf_energy_point_current_energy", input.current_energy)?;
    validate_complex_scalar(
        "pot_scf_energy_point_previous_energy",
        input.previous_energy,
    )?;
    validate_real_scalar(
        "pot_scf_energy_point_current_floor",
        input.current_floor as Real,
    )?;
    validate_real_scalar(
        "pot_scf_energy_point_previous_floor",
        input.previous_floor as Real,
    )?;

    let potential_count =
        input
            .highest_potential_index
            .checked_add(1)
            .ok_or(DensityError::InvalidIndex {
                name: "pot_scf_energy_point_highest_potential_index",
                index: input.highest_potential_index,
            })?;
    ensure_len(
        "pot_scf_energy_point_last_indices",
        input.last_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "pot_scf_energy_point_potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    validate_real_values(
        "pot_scf_energy_point_potential_multiplicities",
        input.potential_multiplicities,
    )?;

    let angular_count = input.scattering_trace.nrows();
    ensure_len("pot_scf_energy_point_angular_count", angular_count, 1)?;
    let mut radial_count = 0;
    for potential in 0..potential_count {
        let last_index = input.last_indices[potential];
        if last_index == 0 {
            return Err(DensityError::InvalidIndex {
                name: "pot_scf_energy_point_last_index",
                index: last_index,
            });
        }
        radial_count = radial_count.max(last_index);
    }

    ensure_shape(
        "pot_scf_energy_point_scattering_trace",
        input.scattering_trace.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_scattering_ldos",
        input.scattering_ldos.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_embedded_ldos",
        input.embedded_ldos.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_previous_ldos",
        input.previous_ldos.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape3(
        "pot_scf_energy_point_scattering_density",
        input.scattering_density.shape(),
        radial_count,
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_embedded_density",
        input.embedded_density.shape(),
        radial_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_previous_density",
        input.previous_density.shape(),
        radial_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_valence_density",
        input.valence_density.shape(),
        radial_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_energy_point_occupancy_by_l",
        input.occupancy_by_l.shape(),
        angular_count,
        potential_count,
    )?;

    for (index, &value) in input.scattering_trace.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name: "pot_scf_energy_point_scattering_trace",
                index,
                real: value.re as Real,
                imaginary: value.im as Real,
            });
        }
    }
    validate_complex_values(
        "pot_scf_energy_point_scattering_ldos",
        input.scattering_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_embedded_ldos",
        input.embedded_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_previous_ldos",
        input.previous_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_scattering_density",
        input.scattering_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_embedded_density",
        input.embedded_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_energy_point_previous_density",
        input.previous_density.iter().copied(),
    )?;
    validate_real_table_values(
        "pot_scf_energy_point_valence_density",
        input.valence_density,
    )?;
    validate_real_table_values("pot_scf_energy_point_occupancy_by_l", input.occupancy_by_l)?;
    Ok(())
}

fn validate_pot_scf_contour_run_input(
    input: PotScfContourRunInput<'_>,
) -> Result<(usize, usize, usize, usize), DensityError> {
    validate_real_scalar(
        "pot_scf_contour_run_electron_count_target",
        input.electron_count_target,
    )?;

    let point_count = input.source_energies.len();
    ensure_len("pot_scf_contour_run_source_energies", point_count, 1)?;
    validate_complex_values(
        "pot_scf_contour_run_source_energies",
        input.source_energies.iter().copied(),
    )?;

    if input.active_energy_count == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_run_active_energy_count",
            index: input.active_energy_count,
        });
    }
    if input.floor_count == 0 || input.floor_count > i32::MAX as usize {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_run_floor_count",
            index: input.floor_count,
        });
    }
    ensure_len(
        "pot_scf_contour_run_energy_grid",
        input.energy_grid.len(),
        input.active_energy_count,
    )?;
    ensure_len(
        "pot_scf_contour_run_steps",
        input.steps.len(),
        input.floor_count,
    )?;
    validate_complex_values(
        "pot_scf_contour_run_energy_grid",
        input
            .energy_grid
            .iter()
            .take(input.active_energy_count)
            .copied(),
    )?;
    for floor in 0..input.floor_count {
        validate_positive_real_scalar("pot_scf_contour_run_steps", input.steps[floor])?;
    }

    let potential_count =
        input
            .highest_potential_index
            .checked_add(1)
            .ok_or(DensityError::InvalidIndex {
                name: "pot_scf_contour_run_highest_potential_index",
                index: input.highest_potential_index,
            })?;
    ensure_len(
        "pot_scf_contour_run_last_indices",
        input.last_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "pot_scf_contour_run_potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    validate_real_values(
        "pot_scf_contour_run_potential_multiplicities",
        input.potential_multiplicities,
    )?;

    let mut active_radial_count = 0;
    for potential in 0..potential_count {
        let last_index = input.last_indices[potential];
        if last_index == 0 {
            return Err(DensityError::InvalidIndex {
                name: "pot_scf_contour_run_last_index",
                index: last_index,
            });
        }
        active_radial_count = active_radial_count.max(last_index);
    }

    let scattering_trace_shape = input.scattering_trace.shape();
    ensure_len(
        "pot_scf_contour_run_scattering_trace_points",
        scattering_trace_shape[0],
        point_count,
    )?;
    let angular_count = scattering_trace_shape[1];
    ensure_len("pot_scf_contour_run_angular_count", angular_count, 1)?;
    ensure_len(
        "pot_scf_contour_run_scattering_trace_potentials",
        scattering_trace_shape[2],
        potential_count,
    )?;
    validate_complex32_table_values(
        "pot_scf_contour_run_scattering_trace",
        input.scattering_trace.iter().copied(),
    )?;

    validate_contour_shape3(
        "pot_scf_contour_run_scattering_ldos",
        input.scattering_ldos.shape(),
        point_count,
        angular_count,
        potential_count,
    )?;
    validate_contour_shape3(
        "pot_scf_contour_run_embedded_ldos_source",
        input.embedded_ldos_source.shape(),
        point_count,
        angular_count,
        potential_count,
    )?;
    validate_contour_shape4(
        "pot_scf_contour_run_scattering_density",
        input.scattering_density.shape(),
        point_count,
        active_radial_count,
        angular_count,
        potential_count,
    )?;
    validate_contour_shape3(
        "pot_scf_contour_run_embedded_density_source",
        input.embedded_density_source.shape(),
        point_count,
        active_radial_count,
        potential_count,
    )?;
    let radial_count = input.embedded_density_source.shape()[1];
    ensure_len(
        "pot_scf_contour_run_scattering_density_radial",
        input.scattering_density.shape()[1],
        radial_count,
    )?;

    validate_complex_values(
        "pot_scf_contour_run_scattering_ldos",
        input.scattering_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_run_embedded_ldos_source",
        input.embedded_ldos_source.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_run_scattering_density",
        input.scattering_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_run_embedded_density_source",
        input.embedded_density_source.iter().copied(),
    )?;

    Ok((point_count, angular_count, radial_count, potential_count))
}

fn validate_complex32_table_values<I>(name: &'static str, values: I) -> Result<(), DensityError>
where
    I: IntoIterator<Item = Complex32>,
{
    for (index, value) in values.into_iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re as Real,
                imaginary: value.im as Real,
            });
        }
    }
    Ok(())
}

fn validate_pot_scf_contour_source_energy(
    index: usize,
    actual: Complex,
    expected: Complex,
) -> Result<(), DensityError> {
    let tolerance = 1.0e-10_f64.max(expected.norm() * 1.0e-10);
    if (actual.re - expected.re).abs() <= tolerance && (actual.im - expected.im).abs() <= tolerance
    {
        Ok(())
    } else {
        Err(DensityError::ContourEnergyMismatch {
            index,
            expected_real: expected.re,
            expected_imaginary: expected.im,
            actual_real: actual.re,
            actual_imaginary: actual.im,
        })
    }
}

fn validate_contour_shape3(
    name: &'static str,
    shape: &[usize],
    required_points: usize,
    required_rows: usize,
    required_columns: usize,
) -> Result<(), DensityError> {
    ensure_len(name, shape[0], required_points)?;
    ensure_len(name, shape[1], required_rows)?;
    ensure_len(name, shape[2], required_columns)
}

fn validate_contour_shape4(
    name: &'static str,
    shape: &[usize],
    required_points: usize,
    required_rows: usize,
    required_columns: usize,
    required_depth: usize,
) -> Result<(), DensityError> {
    ensure_len(name, shape[0], required_points)?;
    ensure_len(name, shape[1], required_rows)?;
    ensure_len(name, shape[2], required_columns)?;
    ensure_len(name, shape[3], required_depth)
}

fn contour_floor_as_i32(floor: usize) -> Result<i32, DensityError> {
    i32::try_from(floor).map_err(|_| DensityError::InvalidIndex {
        name: "pot_scf_contour_floor",
        index: floor,
    })
}

fn validate_pot_scf_fermi_endpoint_input(
    input: PotScfFermiEndpointInput<'_>,
) -> Result<(), DensityError> {
    validate_complex_scalar("pot_scf_fermi_current_energy", input.current_energy)?;
    validate_complex_scalar("pot_scf_fermi_previous_energy", input.previous_energy)?;
    validate_real_scalar(
        "pot_scf_fermi_current_electron_delta",
        input.current_electron_delta,
    )?;
    validate_real_scalar(
        "pot_scf_fermi_previous_electron_delta",
        input.previous_electron_delta,
    )?;
    validate_complex_scalar("pot_scf_fermi_left_sum", input.left_sum)?;
    validate_complex_scalar("pot_scf_fermi_right_sum", input.right_sum)?;

    let potential_count =
        input
            .highest_potential_index
            .checked_add(1)
            .ok_or(DensityError::InvalidIndex {
                name: "pot_scf_fermi_highest_potential_index",
                index: input.highest_potential_index,
            })?;
    ensure_len(
        "pot_scf_fermi_last_indices",
        input.last_indices.len(),
        potential_count,
    )?;
    let mut radial_count = 0;
    for potential in 0..potential_count {
        let last_index = input.last_indices[potential];
        if last_index == 0 {
            return Err(DensityError::InvalidIndex {
                name: "pot_scf_fermi_last_index",
                index: last_index,
            });
        }
        radial_count = radial_count.max(last_index);
    }

    let angular_count = input.embedded_ldos.nrows();
    ensure_len("pot_scf_fermi_angular_count", angular_count, 1)?;
    ensure_shape(
        "pot_scf_fermi_embedded_ldos",
        input.embedded_ldos.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_fermi_previous_ldos",
        input.previous_ldos.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_fermi_embedded_density",
        input.embedded_density.shape(),
        radial_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_fermi_previous_density",
        input.previous_density.shape(),
        radial_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_fermi_valence_density",
        input.valence_density.shape(),
        radial_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_fermi_occupancy_by_l",
        input.occupancy_by_l.shape(),
        angular_count,
        potential_count,
    )?;

    validate_complex_values(
        "pot_scf_fermi_embedded_ldos",
        input.embedded_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_fermi_previous_ldos",
        input.previous_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_fermi_embedded_density",
        input.embedded_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_fermi_previous_density",
        input.previous_density.iter().copied(),
    )?;
    validate_real_table_values("pot_scf_fermi_valence_density", input.valence_density)?;
    validate_real_table_values("pot_scf_fermi_occupancy_by_l", input.occupancy_by_l)?;
    Ok(())
}

fn validate_pot_scf_contour_step_input(
    input: PotScfContourStepInput<'_>,
) -> Result<(), DensityError> {
    if input.energy_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_energy_index",
            index: input.energy_index,
        });
    }
    if input.active_energy_count == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_active_energy_count",
            index: input.active_energy_count,
        });
    }
    if input.floor_count == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_floor_count",
            index: input.floor_count,
        });
    }
    ensure_len(
        "pot_scf_contour_energy_grid",
        input.energy_grid.len(),
        input.active_energy_count,
    )?;
    ensure_len(
        "pot_scf_contour_steps",
        input.steps.len(),
        input.floor_count,
    )?;
    validate_complex_values(
        "pot_scf_contour_energy_grid",
        input
            .energy_grid
            .iter()
            .take(input.active_energy_count)
            .copied(),
    )?;
    for floor in 0..input.floor_count {
        validate_positive_real_scalar("pot_scf_contour_steps", input.steps[floor])?;
    }
    validate_complex_scalar("pot_scf_contour_current_energy", input.current_energy)?;
    validate_complex_scalar("pot_scf_contour_previous_energy", input.previous_energy)?;
    validate_real_scalar(
        "pot_scf_contour_current_electron_delta",
        input.current_electron_delta,
    )?;
    validate_real_scalar(
        "pot_scf_contour_previous_electron_delta",
        input.previous_electron_delta,
    )?;
    validate_pot_scf_contour_floor(
        "pot_scf_contour_current_floor",
        input.current_floor,
        input.floor_count,
    )?;
    validate_pot_scf_contour_floor(
        "pot_scf_contour_previous_floor",
        input.previous_floor,
        input.floor_count,
    )?;
    validate_pot_scf_contour_direction(input.direction)?;
    Ok(())
}

fn validate_pot_scf_contour_step_output(output: PotScfContourStep) -> Result<(), DensityError> {
    validate_complex_scalar(
        "pot_scf_contour_output_previous_energy",
        output.previous_energy,
    )?;
    validate_complex_scalar(
        "pot_scf_contour_output_current_energy",
        output.current_energy,
    )?;
    if output.current_floor == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_output_current_floor",
            index: output.current_floor,
        });
    }
    if output.previous_floor == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_output_previous_floor",
            index: output.previous_floor,
        });
    }
    validate_pot_scf_contour_direction(output.direction)?;
    Ok(())
}

fn validate_pot_scf_contour_floor(
    name: &'static str,
    floor: usize,
    floor_count: usize,
) -> Result<(), DensityError> {
    if floor == 0 || floor > floor_count {
        Err(DensityError::InvalidIndex { name, index: floor })
    } else {
        Ok(())
    }
}

fn validate_pot_scf_contour_direction(direction: i32) -> Result<(), DensityError> {
    if matches!(direction, -1 | 1) {
        Ok(())
    } else {
        Err(DensityError::InvalidIndex {
            name: "pot_scf_contour_direction",
            index: direction.unsigned_abs() as usize,
        })
    }
}
