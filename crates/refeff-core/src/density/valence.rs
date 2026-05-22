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
