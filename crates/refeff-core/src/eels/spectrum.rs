use super::*;

/// Port of the FEFF `EELS/eels.f90` spectrum accumulation loop.
///
/// The input tensor corresponds to FEFF `s(:,2:10)` after `readsp`: row/column
/// order is Cartesian `xx, xy, ..., zz`. FEFF first applies the beam-energy
/// prefactor to both the tensor spectra and atomic background, then integrates
/// `q_i q_j / qfac` over the angular mesh. This function returns the same
/// total, atomic background, fine-structure, and partial tensor columns without
/// doing any file I/O.
pub fn eels_spectrum(input: EelsSpectrumInput<'_>) -> Result<EelsSpectrum, EelsError> {
    validate_spectrum_input(input)?;
    let integration_mesh = eels_integration_mesh(input.mesh)?;
    let energy_count = input.energy_loss_ev.len();
    let mut total = Array1::<Real>::zeros(energy_count);
    let mut background = Array1::<Real>::zeros(energy_count);
    let mut partials = Array2::<Real>::zeros((energy_count, 9).f());
    let incident_wavelength = electron_wavelength_atomic_units(input.incident_energy_ev)?;
    let beam_factor = (1.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV).powi(2)
        / std::f64::consts::PI
        * FEFF_HBARC_ATOMIC;

    for energy_index in 0..energy_count {
        let loss = input.energy_loss_ev[energy_index];
        let scattered_energy = input.incident_energy_ev - loss;
        let prefactor = incident_wavelength / electron_wavelength_atomic_units(scattered_energy)?
            * beam_factor
            / loss;
        let qmesh = eels_qmesh(EelsQMeshInput {
            incident_energy_ev: input.incident_energy_ev,
            scattered_energy_ev: scattered_energy,
            beam_direction: input.beam_direction,
            theta_x: integration_mesh.theta_x.view(),
            theta_y: integration_mesh.theta_y.view(),
            relativistic: input.relativistic,
        })?;
        let scaled_background = input.atomic_background[energy_index] * prefactor;

        for position in 0..integration_mesh.setup.point_count {
            let classical_len = qmesh.classical_q_lengths[position];
            let qfac = if input.relativistic {
                (classical_len.powi(2) - (loss / FEFF_HBARC_EV).powi(2)).powi(2)
            } else {
                classical_len.powi(4)
            };
            if !qfac.is_finite() || qfac.abs() <= Real::MIN_POSITIVE {
                return Err(EelsError::SingularQFactor {
                    energy_index,
                    position,
                });
            }
            let weight = integration_mesh.weights[position] / qfac;
            for row in 0..3 {
                let q_row = qmesh.q_vectors[(row, position)];
                for column in 0..3 {
                    let partial_index = 3 * row + column;
                    let contribution = weight
                        * q_row
                        * qmesh.q_vectors[(column, position)]
                        * input.transition_tensor[(energy_index, row, column)]
                        * prefactor;
                    total[energy_index] += contribution;
                    partials[(energy_index, partial_index)] += contribution;
                    if row == column {
                        background[energy_index] += weight * q_row * q_row * scaled_background;
                    }
                }
            }
        }
    }

    let fine_structure = &total - &background;
    validate_finite_array("total", total.view())?;
    validate_finite_array("background", background.view())?;
    validate_finite_array("fine_structure", fine_structure.view())?;
    validate_finite_matrix("partials", partials.view())?;

    Ok(EelsSpectrum {
        total,
        background,
        fine_structure,
        partials,
        integration_mesh,
    })
}

/// Port of FEFF `EELS/readsp.f90` after file parsing.
///
/// FEFF reads `xmuNN.dat` or `opconsKKNN.dat` files into `xmufile`, then maps
/// the selected spectrum column into `s(:,2:10)`. This helper performs that
/// polarization-index reduction on already-read source columns: orientation
/// sensitive runs keep the requested tensor components, no-cross runs suppress
/// off-diagonal files when FEFF would set `ipsteplocal = 4`, and averaged runs
/// copy either file 10 or the average of files 1, 5, and 9 onto the diagonal.
pub fn eels_read_spectrum(input: EelsReadSpectrumInput<'_>) -> Result<EelsReadSpectrum, EelsError> {
    let sources = validate_read_spectrum_input(input)?;
    let reference = read_spectrum_source(&sources, input.polarization_min)?;
    validate_read_spectrum_energy_grids(&sources, reference.energy_loss_ev)?;

    let energy_count = reference.energy_loss_ev.len();
    let mut transition_tensor = Array3::<Real>::zeros((energy_count, 3, 3).f());
    let effective_step = if input.orientation_averaged {
        assemble_averaged_read_spectrum(&sources, input, &mut transition_tensor)?
    } else {
        assemble_sensitive_read_spectrum(&sources, input, &mut transition_tensor)?
    };

    validate_finite_tensor("read_spectrum_tensor", transition_tensor.view())?;
    Ok(EelsReadSpectrum {
        energy_loss_ev: reference.energy_loss_ev.to_owned(),
        transition_tensor,
        atomic_background: reference.atomic_background.to_owned(),
        effective_polarization_step: effective_step,
    })
}

fn validate_spectrum_input(input: EelsSpectrumInput<'_>) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", input.incident_energy_ev)?;
    if input.incident_energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy {
            value: input.incident_energy_ev,
        });
    }
    for &value in &input.beam_direction {
        validate_finite("beam_direction", value)?;
    }
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "energy_count",
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
    if input.atomic_background.len() != energy_count {
        return Err(EelsError::SpectrumLengthMismatch {
            name: "atomic_background",
            expected: energy_count,
            actual: input.atomic_background.len(),
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
    validate_finite_tensor("transition_tensor", input.transition_tensor)?;
    validate_finite_array("atomic_background", input.atomic_background)?;
    Ok(())
}

fn validate_read_spectrum_input<'a>(
    input: EelsReadSpectrumInput<'a>,
) -> Result<[Option<EelsReadSpectrumSource<'a>>; 11], EelsError> {
    if input.polarization_step == 0
        || input.polarization_min == 0
        || input.polarization_min > input.polarization_max
        || input.polarization_max > 10
    {
        return Err(EelsError::InvalidPolarizationRange {
            min: input.polarization_min,
            step: input.polarization_step,
            max: input.polarization_max,
        });
    }

    let mut sources = [None; 11];
    for &source in input.sources {
        let index = source.polarization_index;
        if !(1..=10).contains(&index) {
            return Err(EelsError::InvalidPolarizationIndex { value: index });
        }
        if sources[index].is_some() {
            return Err(EelsError::DuplicatePolarizationSource { index });
        }

        let energy_count = source.energy_loss_ev.len();
        if energy_count == 0 {
            return Err(EelsError::InvalidMeshCount {
                name: "energy_count",
                value: 0,
            });
        }
        validate_read_spectrum_len(
            index,
            "selected_spectrum",
            source.selected_spectrum.len(),
            energy_count,
        )?;
        validate_read_spectrum_len(
            index,
            "atomic_background",
            source.atomic_background.len(),
            energy_count,
        )?;
        validate_finite_array("readsp_energy_loss_ev", source.energy_loss_ev)?;
        validate_finite_array("readsp_selected_spectrum", source.selected_spectrum)?;
        validate_finite_array("readsp_atomic_background", source.atomic_background)?;
        sources[index] = Some(source);
    }

    Ok(sources)
}

fn validate_read_spectrum_len(
    polarization_index: usize,
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), EelsError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EelsError::ReadSpectrumLengthMismatch {
            polarization_index,
            name,
            expected,
            actual,
        })
    }
}

fn validate_read_spectrum_energy_grids(
    sources: &[Option<EelsReadSpectrumSource<'_>>; 11],
    reference_energy: ArrayView1<'_, Real>,
) -> Result<(), EelsError> {
    for source in sources.iter().flatten() {
        if source.energy_loss_ev.len() != reference_energy.len() {
            return Err(EelsError::ReadSpectrumLengthMismatch {
                polarization_index: source.polarization_index,
                name: "energy_loss_ev",
                expected: reference_energy.len(),
                actual: source.energy_loss_ev.len(),
            });
        }
        for (row, (&expected, &actual)) in reference_energy
            .iter()
            .zip(source.energy_loss_ev.iter())
            .enumerate()
        {
            if actual != expected {
                return Err(EelsError::ReadSpectrumEnergyMismatch {
                    polarization_index: source.polarization_index,
                    row,
                    expected,
                    actual,
                });
            }
        }
    }
    Ok(())
}

fn read_spectrum_source<'a>(
    sources: &[Option<EelsReadSpectrumSource<'a>>; 11],
    index: usize,
) -> Result<EelsReadSpectrumSource<'a>, EelsError> {
    sources
        .get(index)
        .and_then(|source| *source)
        .ok_or(EelsError::MissingPolarizationSource { index })
}

fn assemble_sensitive_read_spectrum(
    sources: &[Option<EelsReadSpectrumSource<'_>>; 11],
    input: EelsReadSpectrumInput<'_>,
    transition_tensor: &mut Array3<Real>,
) -> Result<usize, EelsError> {
    if input.polarization_min != 1 || input.polarization_max != 9 {
        return Err(EelsError::InvalidPolarizationRange {
            min: input.polarization_min,
            step: input.polarization_step,
            max: input.polarization_max,
        });
    }
    if input.cross_terms && input.polarization_step != 1 {
        return Err(EelsError::InvalidPolarizationRange {
            min: input.polarization_min,
            step: input.polarization_step,
            max: input.polarization_max,
        });
    }

    let effective_step = if input.cross_terms || input.polarization_step != 1 {
        input.polarization_step
    } else {
        4
    };

    for polarization_index in
        (input.polarization_min..=input.polarization_max).step_by(effective_step)
    {
        let source = read_spectrum_source(sources, polarization_index)?;
        copy_read_spectrum_component(transition_tensor, polarization_index, source);
    }
    Ok(effective_step)
}

fn assemble_averaged_read_spectrum(
    sources: &[Option<EelsReadSpectrumSource<'_>>; 11],
    input: EelsReadSpectrumInput<'_>,
    transition_tensor: &mut Array3<Real>,
) -> Result<usize, EelsError> {
    match (input.polarization_min, input.polarization_max) {
        (10, 10) => {
            let source = read_spectrum_source(sources, 10)?;
            copy_read_spectrum_diagonal(transition_tensor, source.selected_spectrum);
        }
        (1, 9) => {
            let x = read_spectrum_source(sources, 1)?;
            let y = read_spectrum_source(sources, 5)?;
            let z = read_spectrum_source(sources, 9)?;
            for energy_index in 0..transition_tensor.dim().0 {
                let averaged = (x.selected_spectrum[energy_index]
                    + y.selected_spectrum[energy_index]
                    + z.selected_spectrum[energy_index])
                    / 3.0;
                for diagonal in 0..3 {
                    transition_tensor[(energy_index, diagonal, diagonal)] = averaged;
                }
            }
        }
        _ => {
            return Err(EelsError::InvalidPolarizationRange {
                min: input.polarization_min,
                step: input.polarization_step,
                max: input.polarization_max,
            });
        }
    }
    Ok(input.polarization_step)
}

fn copy_read_spectrum_component(
    transition_tensor: &mut Array3<Real>,
    polarization_index: usize,
    source: EelsReadSpectrumSource<'_>,
) {
    let component = polarization_index - 1;
    let row = component / 3;
    let column = component % 3;
    for (energy_index, &value) in source.selected_spectrum.iter().enumerate() {
        transition_tensor[(energy_index, row, column)] = value;
    }
}

fn copy_read_spectrum_diagonal(transition_tensor: &mut Array3<Real>, values: ArrayView1<'_, Real>) {
    for (energy_index, &value) in values.iter().enumerate() {
        for diagonal in 0..3 {
            transition_tensor[(energy_index, diagonal, diagonal)] = value;
        }
    }
}
