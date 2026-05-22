use super::*;

/// Return FEFF EELS mesh metadata after `init_work` rules are applied.
pub fn eels_mesh_setup(input: EelsMeshInput) -> Result<EelsMeshSetup, EelsError> {
    validate_mesh_inputs(input)?;

    let mut radial_count = input.radial_count;
    let mut angular_count = input.angular_count;
    let angle_sum = input.collection_angle + input.convergence_angle;
    let theta_part = if input.collection_angle > 1.0e-6 || input.convergence_angle > 1.0e-6 {
        angle_sum / (2.0 * radial_count as Real)
    } else if radial_count
        .checked_add(angular_count)
        .ok_or(EelsError::MeshSizeOverflow)?
        > 2
    {
        radial_count = 1;
        angular_count = 1;
        0.0
    } else {
        0.0
    };

    let point_count = match input.mode {
        EelsMeshMode::Uniform | EelsMeshMode::Logarithmic => radial_count
            .checked_mul(radial_count)
            .and_then(|value| value.checked_mul(angular_count))
            .ok_or(EelsError::MeshSizeOverflow)?,
        EelsMeshMode::OneDimensional => {
            angular_count = 1;
            radial_count
        }
    };

    if matches!(
        input.mode,
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
    ) && point_count > 1
    {
        if radial_count <= 1 {
            return Err(EelsError::InvalidMeshCount {
                name: "radial_count",
                value: radial_count,
            });
        }
        if input.theta0 <= 0.0 || !input.theta0.is_finite() {
            return Err(EelsError::InvalidLogMeshParameter {
                name: "theta0",
                value: input.theta0,
            });
        }
        if angle_sum <= 0.0 || !angle_sum.is_finite() {
            return Err(EelsError::InvalidLogMeshParameter {
                name: "angle_sum",
                value: angle_sum,
            });
        }
    }

    Ok(EelsMeshSetup {
        radial_count,
        angular_count,
        point_count,
        theta_part,
        mode: input.mode,
    })
}

/// Port of FEFF `EELS/angularmesh.f90`.
///
/// The returned coordinates are FEFF `ThXV` and `ThYV` after applying the
/// requested detector center and q-mesh mode.
pub fn eels_angular_mesh(input: EelsMeshInput) -> Result<EelsAngularMesh, EelsError> {
    let setup = eels_mesh_setup(input)?;
    angular_mesh_with_setup(input, setup)
}

/// Port of FEFF `EELS/calculateweights.f90`.
///
/// FEFF computes `WeightV` from a zero-centered angular mesh, then regenerates
/// `ThXV` and `ThYV` around the detector center for the rest of the EELS
/// calculation. This function returns that final centered mesh and the weights.
pub fn eels_integration_mesh(input: EelsMeshInput) -> Result<EelsIntegrationMesh, EelsError> {
    let setup = eels_mesh_setup(input)?;
    let zero_center = EelsMeshInput {
        theta_x_center: 0.0,
        theta_y_center: 0.0,
        ..input
    };
    let zero_mesh = angular_mesh_with_setup(zero_center, setup)?;
    let weights = calculate_weights(input, setup, &zero_mesh)?;
    let centered_mesh = angular_mesh_with_setup(input, setup)?;

    Ok(EelsIntegrationMesh {
        theta_x: centered_mesh.theta_x,
        theta_y: centered_mesh.theta_y,
        weights,
        setup,
    })
}

pub(super) fn validate_mesh_inputs(input: EelsMeshInput) -> Result<(), EelsError> {
    validate_angle("collection_angle", input.collection_angle)?;
    validate_angle("convergence_angle", input.convergence_angle)?;
    validate_finite("theta0", input.theta0)?;
    validate_finite("theta_x_center", input.theta_x_center)?;
    validate_finite("theta_y_center", input.theta_y_center)?;
    if input.radial_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "radial_count",
            value: input.radial_count,
        });
    }
    if input.angular_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "angular_count",
            value: input.angular_count,
        });
    }
    Ok(())
}

pub(super) fn validate_angle(name: &'static str, value: Real) -> Result<(), EelsError> {
    if value < 0.0 || !value.is_finite() {
        return Err(EelsError::InvalidMeshAngle { name, value });
    }
    Ok(())
}

pub(super) fn angular_mesh_with_setup(
    input: EelsMeshInput,
    setup: EelsMeshSetup,
) -> Result<EelsAngularMesh, EelsError> {
    let mut theta_x = Vec::with_capacity(setup.point_count);
    let mut theta_y = Vec::with_capacity(setup.point_count);
    if setup.point_count == 1 {
        theta_x.push(input.theta_x_center);
        theta_y.push(input.theta_y_center);
        return Ok(EelsAngularMesh {
            theta_x: Array1::from_vec(theta_x),
            theta_y: Array1::from_vec(theta_y),
            setup,
        });
    }

    let dxx = if matches!(
        setup.mode,
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
    ) {
        ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
            / (setup.radial_count as Real - 1.0)
    } else {
        0.0
    };
    let exp_dxx = dxx.exp();

    for iray in 1..=setup.radial_count {
        let present_tour = if setup.mode == EelsMeshMode::OneDimensional {
            1
        } else {
            setup.angular_count * (2 * iray - 1)
        };
        let inter_angle = std::f64::consts::TAU / present_tour as Real;
        for itour in 1..=present_tour {
            let (sin_angle, cos_angle) = (inter_angle * itour as Real).sin_cos();
            let radius = if matches!(
                setup.mode,
                EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
            ) {
                if iray == 1 {
                    input.theta0 / 2.0
                } else {
                    input.theta0 * (dxx * (iray as Real - 2.0)).exp() * (1.0 + exp_dxx) / 2.0
                }
            } else {
                setup.theta_part * (2 * iray - 1) as Real
            };
            theta_x.push(input.theta_x_center + radius * cos_angle);
            theta_y.push(input.theta_y_center + radius * sin_angle);
        }
    }

    ensure_point_count(setup.point_count, theta_x.len())?;
    Ok(EelsAngularMesh {
        theta_x: Array1::from_vec(theta_x),
        theta_y: Array1::from_vec(theta_y),
        setup,
    })
}

pub(super) fn calculate_weights(
    input: EelsMeshInput,
    setup: EelsMeshSetup,
    zero_mesh: &EelsAngularMesh,
) -> Result<RealVec, EelsError> {
    let mut weights = Vec::with_capacity(setup.point_count);
    if setup.point_count == 1 {
        weights.push(1.0);
        return Ok(Array1::from_vec(weights));
    }

    let dxx = if matches!(
        setup.mode,
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
    ) {
        ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
            / (setup.radial_count as Real - 1.0)
    } else {
        0.0
    };
    let exp_2dxx = (2.0 * dxx).exp();
    let sa = input.collection_angle;
    let ca = input.convergence_angle;
    let mut index_pos = 0usize;

    for iray in 1..=setup.radial_count {
        let present_tour = if setup.mode == EelsMeshMode::OneDimensional {
            1
        } else {
            setup.angular_count * (2 * iray - 1)
        };
        let theta = *zero_mesh.theta_x.get(index_pos + present_tour - 1).ok_or(
            EelsError::MeshSizeMismatch {
                expected: setup.point_count,
                actual: zero_mesh.theta_x.len(),
            },
        )?;
        let convol_value = convolution_overlap_value(theta, sa, ca);
        for _ in 0..present_tour {
            let mut weight = setup.theta_part.powi(2) / present_tour as Real
                * std::f64::consts::PI
                * 4.0
                * (2 * iray - 1) as Real
                * convol_value;
            if matches!(
                setup.mode,
                EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
            ) {
                let lfactor = if iray == 1 {
                    (setup.radial_count as Real * input.theta0 / (sa + ca)).powi(2)
                        * setup.angular_count as Real
                        / present_tour as Real
                } else {
                    (setup.radial_count as Real * input.theta0 * (dxx * (iray as Real - 2.0)).exp()
                        / (sa + ca))
                        .powi(2)
                        * (exp_2dxx - 1.0)
                        * setup.angular_count as Real
                        / present_tour as Real
                };
                weight *= lfactor;
            }
            if !weight.is_finite() {
                return Err(EelsError::NonFiniteResult {
                    name: "weight",
                    value: weight,
                });
            }
            weights.push(weight);
        }
        index_pos += present_tour;
    }

    ensure_point_count(setup.point_count, weights.len())?;
    Ok(Array1::from_vec(weights))
}

fn convolution_overlap_value(theta: Real, collection_angle: Real, convergence_angle: Real) -> Real {
    let sa = collection_angle;
    let ca = convergence_angle;
    if theta <= (sa - ca).abs() {
        if ca > 1.0e-6 && sa > 1.0e-6 {
            sa.min(ca).powi(2) / ca.powi(2)
        } else {
            1.0
        }
    } else if theta >= sa + ca {
        0.0
    } else {
        let p = (theta * theta + ca * ca - sa * sa) / (2.0 * theta);
        let value = std::f64::consts::PI / 2.0 * (ca * ca + sa * sa)
            - p * (ca * ca - p * p).sqrt()
            - (theta - p) * (sa * sa - (theta - p) * (theta - p)).sqrt()
            - sa * sa * ((theta - p) / sa).asin()
            - ca * ca * (p / ca).asin();
        value / (std::f64::consts::PI * ca * ca)
    }
}

fn ensure_point_count(expected: usize, actual: usize) -> Result<(), EelsError> {
    if expected != actual {
        return Err(EelsError::MeshSizeMismatch { expected, actual });
    }
    Ok(())
}
