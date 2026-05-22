use super::*;

/// Port of FEFF `EELS/writeangulardependence1.f90`.
///
/// FEFF removes the angular integration weights from `sdlm` partial spectra and
/// maps each spherical q-vector to a small-angle scattering angle in mrad. This
/// function returns the same nine output columns without doing file I/O.
pub fn eels_angular_dependence(
    input: EelsAngularDependenceInput<'_>,
) -> Result<EelsAngularDependenceTable, EelsError> {
    validate_angular_dependence_input(input)?;

    let position_count = input.weights.len();
    let mut rows =
        Array2::<Real>::zeros((position_count, FEFF_EELS_ANGULAR_DEPENDENCE_COLUMN_COUNT).f());
    for position in 0..position_count {
        let weight = input.weights[position];
        let q = input.q_vectors_spherical[(0, position)];
        let theta_q = input.q_vectors_spherical[(1, position)];
        let denominator = input.incident_wave_number.powi(2) + q.powi(2)
            - 2.0 * input.incident_wave_number * q * theta_q.cos();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(EelsError::SingularScatteringAngle { position });
        }

        let pi_component = input.partial_spectra[(2, position)] / weight;
        let sigma_dipole =
            (input.partial_spectra[(1, position)] + input.partial_spectra[(3, position)]) / weight;
        let sigma = (input.partial_spectra[(1, position)]
            + input.partial_spectra[(3, position)]
            + input.partial_spectra[(0, position)])
            / weight;
        let quadrupole = (4..=8)
            .map(|partial| input.partial_spectra[(partial, position)])
            .sum::<Real>()
            / weight;
        let octupole = input.partial_spectra[(9, position)] / weight;
        let monopole = sigma - sigma_dipole;
        let total_dipole = pi_component + sigma_dipole;
        let total = pi_component + sigma + quadrupole;

        rows[(position, 0)] = q * theta_q.sin() * -1000.0 / denominator.sqrt();
        rows[(position, 1)] = pi_component;
        rows[(position, 2)] = sigma;
        rows[(position, 3)] = total;
        rows[(position, 4)] = sigma_dipole;
        rows[(position, 5)] = total_dipole;
        rows[(position, 6)] = monopole;
        rows[(position, 7)] = quadrupole;
        rows[(position, 8)] = octupole;
    }

    validate_finite_matrix("angular_dependence", rows.view())?;
    Ok(EelsAngularDependenceTable { rows })
}

/// Port of FEFF `EELS/writeangulardependence2.f90`.
///
/// FEFF builds q-vectors once on the original full mesh, then recomputes only
/// the integration weights while sweeping the collection semiangle. The output
/// is the five floating-point columns that FEFF writes to file 59, plus the
/// integer `npos` column as metadata.
pub fn eels_collection_angle_dependence(
    input: EelsCollectionDependenceInput<'_>,
) -> Result<EelsCollectionDependenceTable, EelsError> {
    validate_collection_dependence_input(input)?;

    let magic_index = eels_magic_energy_index(
        input.energy_loss_ev,
        input.sigma_x_spectrum,
        input.magic_energy_ev,
    );
    let magic_energy_loss = input.energy_loss_ev[magic_index];
    if magic_energy_loss <= 0.0 || magic_energy_loss >= input.incident_energy_ev {
        return Err(EelsError::InvalidEnergyLoss {
            index: magic_index,
            value: magic_energy_loss,
            incident_energy_ev: input.incident_energy_ev,
        });
    }

    let collections = eels_collection_sweep(input.mesh)?;
    let original_mesh = eels_angular_mesh(input.mesh)?;
    let qmesh = eels_qmesh(EelsQMeshInput {
        incident_energy_ev: input.incident_energy_ev,
        scattered_energy_ev: input.incident_energy_ev - magic_energy_loss,
        beam_direction: input.beam_direction,
        theta_x: original_mesh.theta_x.view(),
        theta_y: original_mesh.theta_y.view(),
        relativistic: input.relativistic,
    })?;

    let mut rows = Array2::<Real>::zeros(
        (
            collections.len(),
            FEFF_EELS_COLLECTION_DEPENDENCE_COLUMN_COUNT,
        )
            .f(),
    );
    let mut point_counts = Vec::with_capacity(collections.len());
    for (collection_index, &collection_angle) in collections.iter().enumerate() {
        let radial_count = collection_index + 1;
        let (weights, setup) =
            eels_collection_sweep_weights(input.mesh, collection_angle, radial_count)?;
        if setup.point_count > qmesh.classical_q_lengths.len() {
            return Err(EelsError::MeshSizeMismatch {
                expected: setup.point_count,
                actual: qmesh.classical_q_lengths.len(),
            });
        }

        let mut pi_component = 0.0;
        let mut sigma_dipole = 0.0;
        for position in 0..setup.point_count {
            let classical_len = qmesh.classical_q_lengths[position];
            let qfac = if input.relativistic {
                (classical_len.powi(2) - (magic_energy_loss / FEFF_HBARC_EV).powi(2)).powi(2)
            } else {
                classical_len.powi(4)
            };
            if !qfac.is_finite() || qfac.abs() <= Real::MIN_POSITIVE {
                return Err(EelsError::SingularQFactor {
                    energy_index: magic_index,
                    position,
                });
            }
            let weight = weights[position] / qfac;
            let qx = qmesh.q_vectors[(0, position)];
            let qy = qmesh.q_vectors[(1, position)];
            let qz = qmesh.q_vectors[(2, position)];
            pi_component += weight * qz * qz * input.pi_spectrum[magic_index];
            sigma_dipole += weight
                * (qx * qx * input.sigma_x_spectrum[magic_index]
                    + qy * qy * input.sigma_y_spectrum[magic_index]);
        }
        let total = pi_component + sigma_dipole;
        rows[(collection_index, 0)] = collection_angle;
        rows[(collection_index, 1)] = if total.abs() > 0.0 {
            pi_component / total
        } else {
            0.0
        };
        rows[(collection_index, 2)] = pi_component;
        rows[(collection_index, 3)] = sigma_dipole;
        rows[(collection_index, 4)] = total;
        point_counts.push(setup.point_count);
    }

    validate_finite_matrix("collection_dependence", rows.view())?;
    Ok(EelsCollectionDependenceTable {
        rows,
        point_counts: Array1::from_vec(point_counts),
        magic_index,
        magic_energy_loss_ev: magic_energy_loss,
    })
}

fn validate_angular_dependence_input(
    input: EelsAngularDependenceInput<'_>,
) -> Result<(), EelsError> {
    validate_finite("incident_wave_number", input.incident_wave_number)?;
    if input.incident_wave_number <= 0.0 {
        return Err(EelsError::InvalidWaveNumber {
            value: input.incident_wave_number,
        });
    }

    let position_count = input.weights.len();
    if position_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "position_count",
            value: 0,
        });
    }
    let (q_rows, q_columns) = input.q_vectors_spherical.dim();
    if (q_rows, q_columns) != (3, position_count) {
        return Err(EelsError::InvalidTableShape {
            name: "q_vectors_spherical",
            rows: q_rows,
            columns: q_columns,
            expected_rows: 3,
            expected_columns: position_count,
        });
    }
    let (partial_rows, partial_columns) = input.partial_spectra.dim();
    if (partial_rows, partial_columns)
        != (FEFF_EELS_ANGULAR_DEPENDENCE_PARTIAL_COUNT, position_count)
    {
        return Err(EelsError::InvalidTableShape {
            name: "partial_spectra",
            rows: partial_rows,
            columns: partial_columns,
            expected_rows: FEFF_EELS_ANGULAR_DEPENDENCE_PARTIAL_COUNT,
            expected_columns: position_count,
        });
    }

    for (index, &weight) in input.weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EelsError::InvalidWeight {
                index,
                value: weight,
            });
        }
    }
    validate_finite_matrix("q_vectors_spherical", input.q_vectors_spherical)?;
    validate_finite_matrix("partial_spectra", input.partial_spectra)?;
    Ok(())
}

fn validate_collection_dependence_input(
    input: EelsCollectionDependenceInput<'_>,
) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", input.incident_energy_ev)?;
    if input.incident_energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy {
            value: input.incident_energy_ev,
        });
    }
    validate_finite("magic_energy_ev", input.magic_energy_ev)?;
    for &value in &input.beam_direction {
        validate_finite("beam_direction", value)?;
    }
    validate_mesh_inputs(input.mesh)?;
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "energy_count",
            value: 0,
        });
    }
    for (name, spectrum) in [
        ("sigma_x_spectrum", input.sigma_x_spectrum),
        ("sigma_y_spectrum", input.sigma_y_spectrum),
        ("pi_spectrum", input.pi_spectrum),
    ] {
        if spectrum.len() != energy_count {
            return Err(EelsError::SpectrumLengthMismatch {
                name,
                expected: energy_count,
                actual: spectrum.len(),
            });
        }
        validate_finite_array(name, spectrum)?;
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
    Ok(())
}

fn eels_magic_energy_index(
    energy_loss_ev: ArrayView1<'_, Real>,
    reference_spectrum: ArrayView1<'_, Real>,
    magic_energy_ev: Real,
) -> usize {
    let mut origin = -5.0;
    for (index, (&loss, &spectrum)) in energy_loss_ev
        .iter()
        .zip(reference_spectrum.iter())
        .enumerate()
    {
        if spectrum > 1.0e-6 && origin < 0.0 {
            origin = loss;
        }
        if magic_energy_ev > loss - origin && origin >= 0.0 {
            return index;
        }
    }
    energy_loss_ev.len() - 1
}

fn eels_collection_sweep(input: EelsMeshInput) -> Result<RealVec, EelsError> {
    let count = match input.mode {
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional => {
            if input.radial_count <= 1 {
                return Err(EelsError::InvalidMeshCount {
                    name: "radial_count",
                    value: input.radial_count,
                });
            }
            if input.collection_angle <= 0.0 || !input.collection_angle.is_finite() {
                return Err(EelsError::InvalidLogMeshParameter {
                    name: "collection_angle",
                    value: input.collection_angle,
                });
            }
            if input.theta0 <= 0.0 || !input.theta0.is_finite() {
                return Err(EelsError::InvalidLogMeshParameter {
                    name: "theta0",
                    value: input.theta0,
                });
            }
            let dx = ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
                / (input.radial_count as Real - 1.0);
            if !dx.is_finite() || dx <= 0.0 {
                return Err(EelsError::InvalidLogMeshParameter {
                    name: "dx",
                    value: dx,
                });
            }
            1 + (input.collection_angle / input.theta0).ln().div_euclid(dx) as usize
        }
        EelsMeshMode::Uniform => {
            if input.collection_angle + input.convergence_angle <= 0.0 {
                return Err(EelsError::InvalidMeshAngle {
                    name: "angle_sum",
                    value: input.collection_angle + input.convergence_angle,
                });
            }
            (input.collection_angle / (input.collection_angle + input.convergence_angle)
                * input.radial_count as Real) as usize
        }
    };
    if count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "collection_count",
            value: 0,
        });
    }

    let values = match input.mode {
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional => {
            let dx = ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
                / (input.radial_count as Real - 1.0);
            Array1::from_shape_fn(count, |index| {
                if index == 0 {
                    input.theta0
                } else {
                    input.theta0 * (index as Real * dx).exp()
                }
            })
        }
        EelsMeshMode::Uniform => {
            let beta_step =
                (input.convergence_angle + input.collection_angle) / input.radial_count as Real;
            Array1::from_shape_fn(count, |index| beta_step * (index + 1) as Real)
        }
    };
    validate_finite_array("collection_angles", values.view())?;
    Ok(values)
}

fn eels_collection_sweep_weights(
    original: EelsMeshInput,
    collection_angle: Real,
    radial_count: usize,
) -> Result<(RealVec, EelsMeshSetup), EelsError> {
    let angular_count = if original.mode == EelsMeshMode::OneDimensional {
        1
    } else {
        original.angular_count
    };
    let point_count = match original.mode {
        EelsMeshMode::Uniform | EelsMeshMode::Logarithmic => radial_count
            .checked_mul(radial_count)
            .and_then(|value| value.checked_mul(angular_count))
            .ok_or(EelsError::MeshSizeOverflow)?,
        EelsMeshMode::OneDimensional => radial_count,
    };
    let setup = EelsMeshSetup {
        radial_count,
        angular_count,
        point_count,
        theta_part: (collection_angle + original.convergence_angle) / (2.0 * radial_count as Real),
        mode: original.mode,
    };
    let mesh = EelsMeshInput {
        collection_angle,
        radial_count,
        angular_count,
        ..original
    };
    validate_angle("collection_angle", collection_angle)?;
    let zero_mesh = angular_mesh_with_setup(
        EelsMeshInput {
            theta_x_center: 0.0,
            theta_y_center: 0.0,
            ..mesh
        },
        setup,
    )?;
    let mut weights = calculate_weights(mesh, setup, &zero_mesh)?;
    if setup.point_count == 1 {
        weights[0] = if original.convergence_angle > 1.0e-5 {
            std::f64::consts::PI
                * ((original.convergence_angle + collection_angle)
                    * original.convergence_angle.min(collection_angle)
                    / original.convergence_angle)
                    .powi(2)
        } else {
            std::f64::consts::PI * (original.convergence_angle + collection_angle).powi(2)
        };
    }
    validate_finite_array("collection_weights", weights.view())?;
    Ok((weights, setup))
}
