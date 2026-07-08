use super::*;

/// Build FEFF manual-q EELS-MDFF q grids from `mdff_eels.f90`.
///
/// FEFF's `qinput=1` branch keeps the user supplied q-vectors fixed across
/// all energy losses, stores the classical lengths before correction, and then
/// optionally shortens the z component by `(1 - beta**2)`.
pub fn mdff_manual_q_grid(input: MdffManualQGridInput<'_>) -> Result<MdffQGrid, EelsError> {
    validate_manual_q_grid_input(input)?;

    let (_, q_count) = input.q_vectors.dim();
    let beta = mdff_beta(input.incident_energy_ev);
    let relativistic_factor = if input.relativistic {
        1.0 - beta * beta
    } else {
        1.0
    };
    let mut q_vectors = Array3::<Real>::zeros((input.energy_count, 3, q_count).f());
    let mut classical_q_lengths = Array2::<Real>::zeros((input.energy_count, q_count).f());

    for q_index in 0..q_count {
        let raw = [
            input.q_vectors[(0, q_index)],
            input.q_vectors[(1, q_index)],
            input.q_vectors[(2, q_index)],
        ];
        let classical_length = raw[0].hypot(raw[1]).hypot(raw[2]);
        if !classical_length.is_finite() {
            return Err(EelsError::NonFiniteResult {
                name: "mdff_classical_q_length",
                value: classical_length,
            });
        }

        for energy_index in 0..input.energy_count {
            q_vectors[(energy_index, 0, q_index)] = raw[0];
            q_vectors[(energy_index, 1, q_index)] = raw[1];
            q_vectors[(energy_index, 2, q_index)] = raw[2] * relativistic_factor;
            classical_q_lengths[(energy_index, q_index)] = classical_length;
        }
    }

    validate_finite_tensor("mdff_q_vectors", q_vectors.view())?;
    validate_finite_matrix("mdff_classical_q_lengths", classical_q_lengths.view())?;

    Ok(MdffQGrid {
        q_vectors,
        classical_q_lengths,
    })
}

/// Build FEFF automatic-q EELS-MDFF q grids from `mdff_qmesh.f90`.
///
/// In FEFF's `qinput=2` branch, `mdff_eels.f90` recalculates the q-vectors for
/// each energy loss using the scattered electron energy `ebeam - s(i,1)`, then
/// copies `QV(:,1,:)` into the MDFF spectrum reducer. This helper ports that
/// q-grid construction without choosing the still-undefined beam amplitudes from
/// the legacy standalone MDFF driver.
pub fn mdff_automatic_q_grid(input: MdffAutomaticQGridInput<'_>) -> Result<MdffQGrid, EelsError> {
    validate_automatic_q_grid_input(input)?;

    let energy_count = input.energy_loss_ev.len();
    let q_count = input.theta_x.len();
    let mut q_vectors = Array3::<Real>::zeros((energy_count, 3, q_count).f());
    let mut classical_q_lengths = Array2::<Real>::zeros((energy_count, q_count).f());

    for energy_index in 0..energy_count {
        let loss = input.energy_loss_ev[energy_index];
        let qmesh = eels_qmesh(EelsQMeshInput {
            incident_energy_ev: input.incident_energy_ev,
            scattered_energy_ev: input.incident_energy_ev - loss,
            beam_direction: input.beam_direction,
            theta_x: input.theta_x,
            theta_y: input.theta_y,
            relativistic: input.relativistic,
        })?;

        for q_index in 0..q_count {
            classical_q_lengths[(energy_index, q_index)] = qmesh.classical_q_lengths[q_index];
            for component in 0..3 {
                q_vectors[(energy_index, component, q_index)] =
                    qmesh.q_vectors[(component, q_index)];
            }
        }
    }

    validate_finite_tensor("mdff_q_vectors", q_vectors.view())?;
    validate_finite_matrix("mdff_classical_q_lengths", classical_q_lengths.view())?;

    Ok(MdffQGrid {
        q_vectors,
        classical_q_lengths,
    })
}

/// Port of FEFF `EELSMDFF/mdff_eels.f90` complex spectrum accumulation.
///
/// The input tensor corresponds to FEFF `s(:,2:10)` before the EELS prefactor
/// is applied. `q_vectors` should be the `qve` values used in the summation,
/// while `classical_q_lengths` are the pre-relativistic `QLenVClas` values
/// used in FEFF's q-dependent denominator.
pub fn mdff_spectrum(input: MdffSpectrumInput<'_>) -> Result<MdffSpectrum, EelsError> {
    validate_mdff_spectrum_input(input)?;

    let energy_count = input.energy_loss_ev.len();
    let q_count = input.amplitudes.len();
    let channel_count = q_count
        .checked_mul(q_count)
        .and_then(|count| count.checked_add(1))
        .ok_or(EelsError::MeshSizeOverflow)?;
    let incident_wavelength = electron_wavelength_atomic_units(input.incident_energy_ev)?;
    let beam_factor = (1.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV).powi(2)
        / std::f64::consts::PI
        * FEFF_HBARC_ATOMIC;
    let mut spectrum = Array2::<Complex64>::zeros((energy_count, channel_count).f());
    let mut partials = Array3::<Complex64>::zeros((energy_count, 9, channel_count).f());

    for energy_index in 0..energy_count {
        let loss = input.energy_loss_ev[energy_index];
        let scattered_energy = input.incident_energy_ev - loss;
        let prefactor = incident_wavelength / electron_wavelength_atomic_units(scattered_energy)?
            * beam_factor
            / loss;
        let loss_wave_number_squared = (loss / FEFF_HBARC_EV).powi(2);

        for iq in 0..q_count {
            let qfac = mdff_q_factor(
                input.classical_q_lengths[(energy_index, iq)],
                loss_wave_number_squared,
                input.relativistic,
                energy_index,
                iq,
            )?;
            for iqq in 0..q_count {
                let qqfac = mdff_q_factor(
                    input.classical_q_lengths[(energy_index, iqq)],
                    loss_wave_number_squared,
                    input.relativistic,
                    energy_index,
                    iqq,
                )?;
                let q_pair_factor = 1.0 / (qfac * qqfac);
                let amplitude = input.amplitudes[iq] * input.amplitudes[iqq].conj();
                let channel = 1 + iq * q_count + iqq;

                for row in 0..3 {
                    let q_row = input.q_vectors[(energy_index, row, iq)];
                    for column in 0..3 {
                        let partial = 3 * row + column;
                        let term = amplitude
                            * (q_pair_factor
                                * q_row
                                * input.q_vectors[(energy_index, column, iqq)]
                                * input.transition_tensor[(energy_index, row, column)]
                                * prefactor);
                        spectrum[(energy_index, 0)] += term;
                        spectrum[(energy_index, channel)] += term;
                        partials[(energy_index, partial, 0)] += term;
                        partials[(energy_index, partial, channel)] += term;
                    }
                }
            }
        }
    }

    validate_finite_complex_matrix("mdff_spectrum", spectrum.view())?;
    validate_finite_complex_tensor("mdff_partials", partials.view())?;

    Ok(MdffSpectrum {
        energy_loss_ev: input.energy_loss_ev.to_owned(),
        spectrum,
        partials,
    })
}

fn validate_manual_q_grid_input(input: MdffManualQGridInput<'_>) -> Result<(), EelsError> {
    validate_incident_energy(input.incident_energy_ev)?;
    if input.energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "mdff_energy_count",
            value: 0,
        });
    }

    let (components, q_count) = input.q_vectors.dim();
    if q_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "mdff_q_count",
            value: 0,
        });
    }
    if components != 3 {
        return Err(EelsError::InvalidTableShape {
            name: "mdff_manual_q_vectors",
            rows: components,
            columns: q_count,
            expected_rows: 3,
            expected_columns: q_count,
        });
    }
    validate_finite_matrix("mdff_manual_q_vectors", input.q_vectors)?;
    Ok(())
}

fn validate_automatic_q_grid_input(input: MdffAutomaticQGridInput<'_>) -> Result<(), EelsError> {
    validate_incident_energy(input.incident_energy_ev)?;
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "mdff_energy_count",
            value: 0,
        });
    }
    if input.theta_x.is_empty() {
        return Err(EelsError::InvalidMeshCount {
            name: "mdff_q_count",
            value: 0,
        });
    }
    if input.theta_x.len() != input.theta_y.len() {
        return Err(EelsError::QMeshLengthMismatch {
            theta_x_len: input.theta_x.len(),
            theta_y_len: input.theta_y.len(),
        });
    }
    for (index, &loss) in input.energy_loss_ev.iter().enumerate() {
        validate_finite("mdff_energy_loss_ev", loss)?;
        if loss <= 0.0 || loss >= input.incident_energy_ev {
            return Err(EelsError::InvalidEnergyLoss {
                index,
                value: loss,
                incident_energy_ev: input.incident_energy_ev,
            });
        }
    }
    Ok(())
}

fn validate_mdff_spectrum_input(input: MdffSpectrumInput<'_>) -> Result<(), EelsError> {
    validate_incident_energy(input.incident_energy_ev)?;

    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "mdff_energy_count",
            value: 0,
        });
    }
    let (tensor_energies, tensor_rows, tensor_columns) = input.transition_tensor.dim();
    if (tensor_energies, tensor_rows, tensor_columns) != (energy_count, 3, 3) {
        return Err(EelsError::InvalidSpectrumTensorShape {
            expected_energies: energy_count,
            energies: tensor_energies,
            rows: tensor_rows,
            columns: tensor_columns,
        });
    }

    let (q_energies, q_components, q_count) = input.q_vectors.dim();
    if q_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "mdff_q_count",
            value: 0,
        });
    }
    if (q_energies, q_components) != (energy_count, 3) {
        return Err(EelsError::InvalidMdffQVectorShape {
            expected_energies: energy_count,
            expected_q_count: q_count,
            energies: q_energies,
            components: q_components,
            q_count,
        });
    }
    if input.amplitudes.len() != q_count {
        return Err(EelsError::InvalidMdffAmplitudeLength {
            expected: q_count,
            actual: input.amplitudes.len(),
        });
    }

    let (length_energies, length_q_count) = input.classical_q_lengths.dim();
    if (length_energies, length_q_count) != (energy_count, q_count) {
        return Err(EelsError::InvalidMdffQLengthShape {
            expected_energies: energy_count,
            expected_q_count: q_count,
            energies: length_energies,
            q_count: length_q_count,
        });
    }

    for (index, &loss) in input.energy_loss_ev.iter().enumerate() {
        validate_finite("mdff_energy_loss_ev", loss)?;
        if loss <= 0.0 || loss >= input.incident_energy_ev {
            return Err(EelsError::InvalidEnergyLoss {
                index,
                value: loss,
                incident_energy_ev: input.incident_energy_ev,
            });
        }
    }
    validate_finite_tensor("mdff_transition_tensor", input.transition_tensor)?;
    validate_finite_tensor("mdff_q_vectors", input.q_vectors)?;
    validate_finite_matrix("mdff_classical_q_lengths", input.classical_q_lengths)?;
    for &amplitude in &input.amplitudes {
        validate_finite_complex_input("mdff_amplitude", amplitude)?;
    }
    Ok(())
}

fn validate_incident_energy(value: Real) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", value)?;
    if value <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy { value });
    }
    Ok(())
}

fn mdff_beta(incident_energy_ev: Real) -> Real {
    ((2.0 + incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV)
        / (2.0
            + incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV
            + FEFF_ELECTRON_REST_ENERGY_EV / incident_energy_ev))
        .sqrt()
}

fn mdff_q_factor(
    classical_q_length: Real,
    loss_wave_number_squared: Real,
    relativistic: bool,
    energy_index: usize,
    q_index: usize,
) -> Result<Real, EelsError> {
    let value = if relativistic {
        classical_q_length * classical_q_length - loss_wave_number_squared
    } else {
        classical_q_length * classical_q_length
    };
    if !value.is_finite() || value.abs() <= Real::MIN_POSITIVE {
        return Err(EelsError::SingularQFactor {
            energy_index,
            position: q_index,
        });
    }
    Ok(value)
}

fn validate_finite_complex_input(name: &'static str, value: Complex64) -> Result<(), EelsError> {
    validate_finite(name, value.re)?;
    validate_finite(name, value.im)
}

fn validate_finite_complex_matrix(
    name: &'static str,
    values: ndarray::ArrayView2<'_, Complex64>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite_complex_result(name, value)?;
    }
    Ok(())
}

fn validate_finite_complex_tensor(
    name: &'static str,
    values: ndarray::ArrayView3<'_, Complex64>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite_complex_result(name, value)?;
    }
    Ok(())
}

fn validate_finite_complex_result(name: &'static str, value: Complex64) -> Result<(), EelsError> {
    if !value.re.is_finite() {
        return Err(EelsError::NonFiniteResult {
            name,
            value: value.re,
        });
    }
    if !value.im.is_finite() {
        return Err(EelsError::NonFiniteResult {
            name,
            value: value.im,
        });
    }
    Ok(())
}
