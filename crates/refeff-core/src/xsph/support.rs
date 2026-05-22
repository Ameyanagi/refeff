use super::*;

pub(super) fn validate_active_len(
    name: &'static str,
    actual: usize,
    active_len: usize,
) -> Result<(), XsphError> {
    if active_len == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    if actual < active_len {
        return Err(XsphError::LengthTooShort {
            name,
            required: active_len,
            actual,
        });
    }
    Ok(())
}
pub(super) fn validate_final_lj(
    final_lj: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<(), XsphError> {
    for index in 0..active_len {
        let value = final_lj[index];
        if value < 0 {
            return Err(XsphError::NegativeAngularMomentum {
                name: "final_lj",
                index,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_finite_real(name: &'static str, value: Real) -> Result<(), XsphError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(XsphError::NonFiniteScalar { name, value })
    }
}

pub(super) fn validate_phase_mesh_capacity(capacity: usize) -> Result<(), XsphError> {
    if capacity == 0 {
        Err(XsphError::InvalidPhaseMeshCapacity { capacity })
    } else {
        Ok(())
    }
}

pub(super) fn validate_phase_mesh_step(name: &'static str, value: Real) -> Result<(), XsphError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(XsphError::InvalidPhaseMeshStep { name, value })
    }
}

pub(super) fn validate_phase_mesh_endpoint(
    name: &'static str,
    value: Real,
) -> Result<(), XsphError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(XsphError::InvalidPhaseMeshEndpoint { name, value })
    }
}

pub(super) fn phase_mesh_count(final_offset: i32) -> Result<usize, XsphError> {
    usize::try_from(final_offset)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "phase_mesh_offset",
            value: final_offset,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "phase_mesh_offset",
            value: final_offset,
        })
}
pub(super) fn append_phase_mesh_segment(
    values: &mut Vec<Complex>,
    capacity: usize,
    segment: Array1<Complex>,
) {
    let remaining = capacity.saturating_sub(values.len());
    values.extend(segment.iter().take(remaining).copied());
}

pub(super) fn append_danes_phase_extension(
    values: &mut Vec<Complex>,
    horizontal_count: usize,
    capacity: usize,
) -> Result<usize, XsphError> {
    let extension_limit = capacity.min(150);
    let extension_slots = extension_limit.saturating_sub(values.len());
    if extension_slots <= 1 {
        return Ok(0);
    }

    let extension_count = extension_slots - 1;
    let previous = horizontal_count
        .checked_sub(2)
        .ok_or(XsphError::InvalidPhaseMeshCapacity { capacity })?;
    let min_energy = 2.0 * values[horizontal_count - 1].re - values[previous].re;
    let max_energy = 7.0e4;
    validate_phase_mesh_endpoint("danes_min_energy", min_energy)?;
    let exponent_step = (max_energy / min_energy).ln() / extension_count as Real;
    validate_phase_mesh_step("danes_exponent_step", exponent_step)?;
    values.extend(
        (0..extension_count)
            .map(|index| Complex::new(min_energy * (exponent_step * index as Real).exp(), 2.0e-8)),
    );
    Ok(extension_count)
}

pub(super) fn xsph_thermal_contour_height(temperature: Real) -> Result<(usize, Real), XsphError> {
    validate_phase_mesh_endpoint("thermal_temperature", temperature)?;
    let minimum_height = 0.05;
    let period = 2.0 * std::f64::consts::PI * temperature;
    validate_phase_mesh_endpoint("thermal_period", period)?;
    let pole_count = if period < minimum_height {
        let count = (minimum_height / period).ceil();
        validate_phase_mesh_endpoint("thermal_pole_count", count)?;
        count as usize
    } else {
        1
    };
    let upper_imaginary = pole_count as Real * period;
    validate_phase_mesh_endpoint("thermal_upper_imaginary", upper_imaginary)?;
    Ok((pole_count, upper_imaginary))
}

pub(super) fn xsph_default_thermal_horizontal_grid(
    core_valence_separation: Real,
    upper_imaginary: Real,
) -> Result<Array1<Complex>, XsphError> {
    validate_finite_real("core_valence_separation", core_valence_separation)?;
    validate_phase_mesh_endpoint("thermal_upper_imaginary", upper_imaginary)?;
    let maximum_energy = 5.8;
    let trial_step = upper_imaginary / 4.0;
    validate_phase_mesh_endpoint("thermal_trial_step", trial_step)?;

    let below_count = ((0.0 - core_valence_separation) / trial_step)
        .ceil()
        .min(21.0);
    validate_phase_mesh_endpoint("thermal_below_count", below_count)?;
    let below_count = below_count as usize;
    let energy_step = (0.0 - core_valence_separation) / below_count as Real;
    validate_phase_mesh_endpoint("thermal_energy_step", energy_step)?;

    let horizontal_count = ((maximum_energy - core_valence_separation) / energy_step).ceil();
    validate_phase_mesh_endpoint("thermal_horizontal_count", horizontal_count)?;
    let horizontal_count = horizontal_count as usize;
    Ok(Array1::from_shape_fn(horizontal_count, |index| {
        Complex::new(
            core_valence_separation + (index + 1) as Real * energy_step,
            0.0,
        )
    }))
}

pub(super) fn xsph_thermal_phase_mesh_count(
    horizontal_count: usize,
    pole_count: usize,
) -> Result<usize, XsphError> {
    horizontal_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(10))
        .and_then(|count| count.checked_add(pole_count))
        .and_then(|count| count.checked_add(1))
        .ok_or(XsphError::SizeOutOfRange {
            name: "thermal_phase_mesh_count",
            value: horizontal_count,
        })
}

#[derive(Debug, Clone, Copy)]
struct ResolvedPhaseUserGrid {
    kind: XsphPhaseUserGridKind,
    minimum: Real,
    maximum: Real,
    step: Real,
}

#[derive(Debug, Clone, Copy)]
struct PreviousPhaseUserGrid {
    kind: XsphPhaseUserGridKind,
    maximum: Real,
}

pub(super) fn xsph_user_phase_horizontal_grid(
    records: &[XsphPhaseUserGridRecord<'_>],
    capacity: usize,
) -> Result<Vec<Complex>, XsphError> {
    validate_phase_mesh_capacity(capacity)?;

    let mut user_points = Vec::new();
    let mut regular_records = Vec::new();
    let mut previous = None;

    for (record_index, record) in records.iter().enumerate() {
        match record {
            XsphPhaseUserGridRecord::Regular(record) => {
                let resolved = resolve_phase_user_regular_grid(record_index, *record, previous)?;
                previous = Some(PreviousPhaseUserGrid {
                    kind: resolved.kind,
                    maximum: resolved.maximum,
                });
                regular_records.push(resolved);
            }
            XsphPhaseUserGridRecord::User(points) => {
                if points.is_empty() {
                    return Err(XsphError::EmptyPhaseGridRecords);
                }
                for (point_index, &point) in points.iter().enumerate() {
                    validate_finite_complex("user_grid", point_index, point)?;
                    if user_points.len() < capacity {
                        user_points.push(point / XSPH_HARTREE_EV);
                    }
                }
                let last = points[points.len() - 1].re;
                validate_finite_real("user_grid_maximum", last)?;
                previous = Some(PreviousPhaseUserGrid {
                    kind: XsphPhaseUserGridKind::Energy,
                    maximum: last,
                });
            }
        }
    }

    let mut values = user_points;
    for record in regular_records {
        append_phase_user_regular_grid(&mut values, capacity, record)?;
    }

    Ok(values)
}

pub(super) fn validate_phase_user_grid_records(
    records: &[XsphPhaseUserGridRecord<'_>],
) -> Result<(), XsphError> {
    if records.is_empty() {
        return Err(XsphError::EmptyPhaseGridRecords);
    }
    if records.len() > XSPH_USER_PHASE_GRID_MAX_RECORDS {
        return Err(XsphError::TooManyPhaseGridRecords {
            count: records.len(),
            max: XSPH_USER_PHASE_GRID_MAX_RECORDS,
        });
    }
    Ok(())
}

fn resolve_phase_user_regular_grid(
    _record_index: usize,
    record: XsphPhaseUserRegularGrid,
    previous: Option<PreviousPhaseUserGrid>,
) -> Result<ResolvedPhaseUserGrid, XsphError> {
    validate_finite_real("user_grid_maximum", record.maximum)?;
    validate_phase_mesh_endpoint("user_grid_step", record.step)?;
    let minimum = match record.minimum {
        XsphPhaseUserGridMinimum::Value(value) => {
            validate_finite_real("user_grid_minimum", value)?;
            value
        }
        XsphPhaseUserGridMinimum::Last => {
            let value = previous.map_or(0.0, |previous| {
                phase_user_last_minimum(record.kind, previous, record.step)
            });
            validate_finite_real("user_grid_last_minimum", value)?;
            value
        }
    };
    validate_finite_real("user_grid_minimum", minimum)?;
    Ok(ResolvedPhaseUserGrid {
        kind: record.kind,
        minimum,
        maximum: record.maximum,
        step: record.step,
    })
}

fn phase_user_last_minimum(
    current_kind: XsphPhaseUserGridKind,
    previous: PreviousPhaseUserGrid,
    step: Real,
) -> Real {
    let current_is_k = current_kind == XsphPhaseUserGridKind::WaveNumber;
    let previous_is_k = previous.kind == XsphPhaseUserGridKind::WaveNumber;
    if current_is_k == previous_is_k {
        previous.maximum + step
    } else if current_is_k {
        (2.0 * previous.maximum / XSPH_HARTREE_EV).sqrt() / XSPH_BOHR_ANGSTROM + step
    } else {
        (previous.maximum * XSPH_BOHR_ANGSTROM).powi(2) / 2.0 * XSPH_HARTREE_EV + step
    }
}

fn append_phase_user_regular_grid(
    values: &mut Vec<Complex>,
    capacity: usize,
    record: ResolvedPhaseUserGrid,
) -> Result<(), XsphError> {
    if values.len() >= capacity {
        return Ok(());
    }
    let remaining = capacity - values.len();
    let segment = match record.kind {
        XsphPhaseUserGridKind::Energy => xsph_even_energy_mesh(
            record.minimum / XSPH_HARTREE_EV,
            record.maximum / XSPH_HARTREE_EV,
            record.step / XSPH_HARTREE_EV,
            remaining,
        )?,
        XsphPhaseUserGridKind::WaveNumber => xsph_k_energy_mesh(
            record.minimum * XSPH_BOHR_ANGSTROM,
            record.maximum * XSPH_BOHR_ANGSTROM,
            record.step * XSPH_BOHR_ANGSTROM,
            remaining,
        )?,
        XsphPhaseUserGridKind::Exponential => {
            let minimum = record.minimum / XSPH_HARTREE_EV;
            let maximum = record.maximum / XSPH_HARTREE_EV;
            let step = record.step / XSPH_HARTREE_EV;
            let span = maximum - minimum + 1.0;
            validate_phase_mesh_endpoint("user_exp_grid_span", span)?;
            let exponential = xsph_exponential_energy_mesh(1.0, span, step, remaining)?;
            Array1::from_iter(
                exponential
                    .iter()
                    .map(|energy| Complex::new(energy.re + minimum - 1.0, 0.0)),
            )
        }
    };
    append_phase_mesh_segment(values, capacity, segment);
    Ok(())
}
pub(super) fn validate_indexed_angular_momentum(
    name: &'static str,
    index: usize,
    value: i32,
) -> Result<i32, XsphError> {
    if value < 0 {
        Err(XsphError::NegativeAngularMomentum { name, index, value })
    } else {
        Ok(value)
    }
}

pub(super) fn usize_to_i32(name: &'static str, value: usize) -> Result<i32, XsphError> {
    i32::try_from(value).map_err(|_| XsphError::SizeOutOfRange { name, value })
}
pub(super) fn validate_finite_complex(
    name: &'static str,
    index: usize,
    value: Complex,
) -> Result<(), XsphError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(XsphError::NonFiniteComplex {
            name,
            index,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn doubled_j_from_kappa(name: &'static str, kappa: i32) -> Result<i32, XsphError> {
    let abs_kappa = kappa
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange { name, value: kappa })?;
    abs_kappa
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(XsphError::IntegerOutOfRange { name, value: kappa })
}

pub(super) fn validate_cwig3j_doubled_argument(
    name: &'static str,
    original_value: i32,
    doubled_value: i32,
) -> Result<(), XsphError> {
    if doubled_value <= CWIG3J_MAX_DOUBLED_ARGUMENT {
        Ok(())
    } else {
        Err(XsphError::IntegerOutOfRange {
            name,
            value: original_value,
        })
    }
}

pub(super) fn validate_cwig3j_integer_argument(
    name: &'static str,
    value: i32,
) -> Result<(), XsphError> {
    if value <= CWIG3J_MAX_DOUBLED_ARGUMENT / 2 {
        Ok(())
    } else {
        Err(XsphError::IntegerOutOfRange { name, value })
    }
}
pub(super) fn nint(value: Real) -> i32 {
    value.round() as i32
}
