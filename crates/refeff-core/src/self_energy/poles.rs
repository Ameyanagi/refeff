use super::*;

/// Port of FEFF `MkExc`: fit a loss function with a many-pole model.
///
/// `energy` and `loss` are the two columns of FEFF `loss.dat`, in eV for the
/// standalone `eps2exc` path. `static_dielectric` is FEFF `Eps0`: values in
/// `(-1.5, 0)` select the metallic normalization, values greater than `1`
/// select dielectric scaling, and other values leave the total strength
/// unchanged. The returned poles are the rows FEFF writes to `exc.dat`.
pub fn make_excitation_poles(
    energy: ArrayView1<'_, Real>,
    loss: ArrayView1<'_, Real>,
    static_dielectric: Real,
    requested_poles: usize,
) -> Result<Vec<ExcitationPole>, SelfEnergyError> {
    validate_loss_grid(energy, loss, static_dielectric, requested_poles)?;

    let fine_grid = fine_loss_grid(energy, loss)?;
    let total_inverse_moment = inverse_loss_moment(&fine_grid.energy, &fine_grid.loss);
    ensure_positive_real("MkExc total inverse loss moment", total_inverse_moment)?;
    let target_inverse_moment = total_inverse_moment / requested_poles as Real;

    let mut drafts = Vec::with_capacity(requested_poles);
    let mut segment_start = 0;
    let mut first_moment = 0.0;
    let mut inverse_moment = 0.0;

    for index in 0..fine_grid.energy.len() - 1 {
        inverse_moment += inverse_loss_interval(&fine_grid.energy, &fine_grid.loss, index);
        first_moment += first_loss_interval(&fine_grid.energy, &fine_grid.loss, index);

        if inverse_moment >= target_inverse_moment || index == fine_grid.energy.len() - 2 {
            drafts.push(make_pole_draft(
                &fine_grid.energy,
                segment_start,
                index,
                first_moment,
                inverse_moment,
            )?);
            segment_start = index + 1;
            first_moment = 0.0;
            inverse_moment = 0.0;
        }

        if drafts.len() >= requested_poles {
            break;
        }
    }

    scale_excitation_poles(drafts, static_dielectric)
}

struct FineLossGrid {
    energy: Vec<Real>,
    loss: Vec<Real>,
}

#[derive(Clone, Copy)]
struct ExcitationPoleDraft {
    energy: Real,
    amplitude: Real,
    width: Real,
}

fn validate_loss_grid(
    energy: ArrayView1<'_, Real>,
    loss: ArrayView1<'_, Real>,
    static_dielectric: Real,
    requested_poles: usize,
) -> Result<(), SelfEnergyError> {
    if energy.len() != loss.len() {
        return Err(SelfEnergyError::LossGridLengthMismatch {
            energy_len: energy.len(),
            loss_len: loss.len(),
        });
    }
    if energy.len() < 2 {
        return Err(SelfEnergyError::InsufficientLossGrid { len: energy.len() });
    }
    if requested_poles == 0 {
        return Err(SelfEnergyError::InvalidPoleCount);
    }
    ensure_finite_real("Eps0", static_dielectric)?;

    for (index, (&energy_value, &loss_value)) in energy.iter().zip(loss.iter()).enumerate() {
        ensure_positive_real("loss energy", energy_value)?;
        ensure_nonnegative_real("loss value", loss_value)?;
        if index > 0 {
            let previous = energy[index - 1];
            if energy_value < previous {
                return Err(SelfEnergyError::NonIncreasingLossEnergy {
                    index,
                    previous,
                    current: energy_value,
                });
            }
        }
    }

    let fine_span = energy[energy.len() - 1].min(1000.0) - energy[0];
    ensure_positive_real("MkExc fine-grid span", fine_span)
}

fn fine_loss_grid(
    energy: ArrayView1<'_, Real>,
    loss: ArrayView1<'_, Real>,
) -> Result<FineLossGrid, SelfEnergyError> {
    let energy_values = energy.to_vec();
    let loss_values = loss.to_vec();
    let step = (energy_values[energy_values.len() - 1].min(1000.0) - energy_values[0])
        / MKEXC_FINE_POINTS as Real;
    let first_energy = energy_values[0];
    let first_loss = loss_values[0];

    let fine_energy: Vec<_> = (0..MKEXC_FINE_POINTS)
        .map(|index| step * index as Real)
        .collect();
    let fine_loss = fine_energy
        .iter()
        .map(|&value| {
            if value > first_energy {
                Ok(terp(&energy_values, &loss_values, 1, value)?.value)
            } else {
                Ok(value * first_loss / first_energy)
            }
        })
        .collect::<Result<Vec<_>, SelfEnergyError>>()?;

    Ok(FineLossGrid {
        energy: fine_energy,
        loss: fine_loss,
    })
}

fn make_pole_draft(
    fine_energy: &[Real],
    segment_start: usize,
    index: usize,
    first_moment: Real,
    inverse_moment: Real,
) -> Result<ExcitationPoleDraft, SelfEnergyError> {
    if inverse_moment == 0.0 {
        return Err(SelfEnergyError::ZeroMoment {
            name: "MkExc segment inverse loss moment",
        });
    }
    let radicand = first_moment / inverse_moment;
    if radicand < 0.0 || !radicand.is_finite() {
        return Err(SelfEnergyError::NegativeRadicand {
            name: "MkExc pole energy",
            value: radicand,
        });
    }
    let width = fine_energy[index + 1] - fine_energy[segment_start];
    ensure_positive_real("MkExc pole width", width)?;

    Ok(ExcitationPoleDraft {
        energy: radicand.sqrt(),
        amplitude: 2.0 * inverse_moment / std::f64::consts::PI,
        width,
    })
}

fn scale_excitation_poles(
    drafts: Vec<ExcitationPoleDraft>,
    static_dielectric: Real,
) -> Result<Vec<ExcitationPole>, SelfEnergyError> {
    let total_amplitude = drafts.iter().map(|pole| pole.amplitude).sum::<Real>();
    if total_amplitude == 0.0 {
        return Err(SelfEnergyError::ZeroMoment {
            name: "MkExc total pole amplitude",
        });
    }

    let scale = if static_dielectric > -1.5 && static_dielectric < 0.0 {
        1.0 / total_amplitude
    } else if static_dielectric > 1.0 {
        (1.0 - 1.0 / static_dielectric) / total_amplitude
    } else {
        1.0
    };
    ensure_positive_real("MkExc pole scale", scale)?;
    let scale_root = scale.sqrt();

    drafts
        .into_iter()
        .map(|draft| {
            let energy = draft.energy / scale_root;
            let amplitude = scale * draft.amplitude;
            let loss_height = std::f64::consts::FRAC_PI_2 * amplitude * energy / draft.width;
            ensure_finite_real("MkExc pole energy", energy)?;
            ensure_finite_real("MkExc pole amplitude", amplitude)?;
            ensure_finite_real("MkExc pole height", loss_height)?;
            Ok(ExcitationPole {
                energy,
                width: MKEXC_WIDTH_EV,
                amplitude,
                loss_height,
            })
        })
        .collect()
}

fn inverse_loss_moment(fine_energy: &[Real], fine_loss: &[Real]) -> Real {
    (0..fine_energy.len() - 1)
        .map(|index| inverse_loss_interval(fine_energy, fine_loss, index))
        .sum()
}

fn inverse_loss_interval(fine_energy: &[Real], fine_loss: &[Real], index: usize) -> Real {
    if fine_energy[index] != 0.0 {
        0.5 * (fine_loss[index + 1] / fine_energy[index + 1]
            + fine_loss[index] / fine_energy[index])
            * (fine_energy[index + 1] - fine_energy[index])
    } else {
        fine_loss[index + 1]
    }
}

fn first_loss_interval(fine_energy: &[Real], fine_loss: &[Real], index: usize) -> Real {
    0.5 * (fine_loss[index + 1] * fine_energy[index + 1] + fine_loss[index] * fine_energy[index])
        * (fine_energy[index + 1] - fine_energy[index])
}
