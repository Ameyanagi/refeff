use super::*;

/// Port of FEFF `EELS/writeangulardependence3.f90`.
///
/// FEFF uses this path to write `gos1.txt` and `gos2.txt` for an
/// orientation-averaged EELS calculation. The q-grid defaults and prefactor are
/// intentionally the hardcoded values from the reference routine. File-format
/// rendering is left to the caller; this function returns the q grid and
/// generalized oscillator strength matrix.
pub fn eels_generalized_oscillator_strength(
    input: EelsGosInput<'_>,
) -> Result<EelsGosTable, EelsError> {
    validate_gos_input(input)?;

    let (q_values, q_scale, q_log_step) = eels_gos_q_grid()?;
    let energy_count = input.energy_loss_ev.len();
    let mut strengths = Array2::<Real>::zeros((FEFF_EELS_GOS_Q_COUNT, energy_count).f());
    let gamma = 1.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV;
    let beam_factor = input.incident_energy_ev * (1.0 + gamma)
        / (2.0 * gamma.powi(2))
        / (4.0 * std::f64::consts::PI * FEFF_EELS_GOS_RYDBERG_EV.powi(2))
        * 1000.0;

    for energy_index in 0..energy_count {
        let loss = input.energy_loss_ev[energy_index];
        let prefactor = loss * beam_factor;
        for q_index in 0..FEFF_EELS_GOS_Q_COUNT {
            let q = q_values[q_index];
            let qfac = if input.relativistic {
                (q.powi(2) - (loss / FEFF_HBARC_EV).powi(2)).powi(2)
            } else {
                q.powi(4)
            };
            if !qfac.is_finite() || qfac.abs() <= Real::MIN_POSITIVE {
                return Err(EelsError::SingularQFactor {
                    energy_index,
                    position: q_index,
                });
            }
            strengths[(q_index, energy_index)] =
                q.powi(2) / qfac * input.averaged_spectrum[energy_index] * prefactor;
        }
    }

    validate_finite_matrix("gos_strengths", strengths.view())?;
    Ok(EelsGosTable {
        q_values,
        strengths,
        q_scale,
        q_log_step,
        edge_parameter: FEFF_EELS_GOS_EDGE_PARAMETER,
        energy_start_ev: FEFF_EELS_GOS_ENERGY_START_EV,
        energy_step_ev: FEFF_EELS_GOS_ENERGY_STEP_EV,
    })
}

fn validate_gos_input(input: EelsGosInput<'_>) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", input.incident_energy_ev)?;
    if input.incident_energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy {
            value: input.incident_energy_ev,
        });
    }
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "energy_count",
            value: 0,
        });
    }
    if input.averaged_spectrum.len() != energy_count {
        return Err(EelsError::SpectrumLengthMismatch {
            name: "averaged_spectrum",
            expected: energy_count,
            actual: input.averaged_spectrum.len(),
        });
    }
    for (index, &loss) in input.energy_loss_ev.iter().enumerate() {
        validate_finite("energy_loss_ev", loss)?;
        if loss <= 0.0 || loss >= input.incident_energy_ev {
            return Err(EelsError::InvalidEnergyLoss {
                index,
                value: loss,
                incident_energy_ev: input.incident_energy_ev,
            });
        }
    }
    validate_finite_array("averaged_spectrum", input.averaged_spectrum)?;
    Ok(())
}

fn eels_gos_q_grid() -> Result<(RealVec, Real, Real), EelsError> {
    let q_min = FEFF_EELS_GOS_Q_BASE * (FEFF_EELS_GOS_Q_STEP_SEED.exp() - 1.0) * FEFF_EELS_GOS_A0;
    let q_max = FEFF_EELS_GOS_Q_BASE
        * ((FEFF_EELS_GOS_Q_COUNT as Real * FEFF_EELS_GOS_Q_STEP_SEED).exp() - 1.0)
        * FEFF_EELS_GOS_A0;
    let q_log_step = ((1.0 + q_max) / (1.0 + q_min)).ln() / (FEFF_EELS_GOS_Q_COUNT as Real - 1.0);
    let q_scale = q_min / (FEFF_EELS_GOS_A0 * (q_log_step.exp() - 1.0));
    validate_finite("q_scale", q_scale)?;
    validate_finite("q_log_step", q_log_step)?;

    let q_values = Array1::from_shape_fn(FEFF_EELS_GOS_Q_COUNT, |index| {
        q_scale * (((index + 1) as Real * q_log_step).exp() - 1.0) * FEFF_EELS_GOS_A0
    });
    validate_finite_array("q_values", q_values.view())?;
    Ok((q_values, q_scale, q_log_step))
}
