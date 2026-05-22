use super::*;

/// Port of FEFF `EELS/qmesh.f90`.
///
/// Builds momentum-transfer vectors for one scattered-electron energy from the
/// detector-plane angular mesh. FEFF currently rotates the observer-frame
/// q-vector into a single local basis using the Euler angles implied by
/// `xivec`; this function returns the same `(3, npos)` q-vector table together
/// with the relativistic and classical q lengths.
pub fn eels_qmesh(input: EelsQMeshInput<'_>) -> Result<EelsQMesh, EelsError> {
    validate_qmesh_input(input)?;

    let euler_angles = eels_qmesh_euler_angles(input.beam_direction);
    let rotation_matrix =
        eels_euler_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])?;
    let incident_wave_number =
        std::f64::consts::TAU / electron_wavelength_atomic_units(input.incident_energy_ev)?;
    let scattered_wave_number =
        std::f64::consts::TAU / electron_wavelength_atomic_units(input.scattered_energy_ev)?;
    let beta = ((2.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV)
        / (2.0
            + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV
            + FEFF_ELECTRON_REST_ENERGY_EV / input.incident_energy_ev))
        .sqrt();
    let relativistic_factor = if input.relativistic {
        1.0 - beta * beta
    } else {
        1.0
    };

    let position_count = input.theta_x.len();
    let mut q_vectors = Array2::<Real>::zeros((3, position_count).f());
    let mut q_lengths = Array1::<Real>::zeros(position_count);
    let mut classical_q_lengths = Array1::<Real>::zeros(position_count);

    for position in 0..position_count {
        let theta_x = input.theta_x[position];
        let theta_y = input.theta_y[position];
        let theta = theta_x.hypot(theta_y);
        let phi = eels_qmesh_phi(theta_x, theta_y);
        let mut q = [
            -scattered_wave_number * theta.sin() * phi.cos(),
            -scattered_wave_number * theta.sin() * phi.sin(),
            scattered_wave_number * theta.cos() - incident_wave_number,
        ];
        classical_q_lengths[position] = q[0].hypot(q[1]).hypot(q[2]);
        q[2] *= relativistic_factor;
        q_lengths[position] = q[0].hypot(q[1]).hypot(q[2]);

        for row in 0..3 {
            q_vectors[(row, position)] = (0..3)
                .map(|column| rotation_matrix[(row, column)] * q[column])
                .sum::<Real>();
        }
    }

    validate_finite_matrix("q_vectors", q_vectors.view())?;
    validate_finite_array("q_lengths", q_lengths.view())?;
    validate_finite_array("classical_q_lengths", classical_q_lengths.view())?;

    Ok(EelsQMesh {
        q_vectors,
        q_lengths,
        classical_q_lengths,
        euler_angles,
        rotation_matrix,
    })
}

fn eels_qmesh_euler_angles(beam_direction: [Real; 3]) -> [Real; 3] {
    let alpha1 = if beam_direction[0].abs() < 0.0001 {
        if beam_direction[1] > 0.0001 {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        }
    } else {
        (beam_direction[1] / beam_direction[0]).atan()
    };
    let alpha2 = if beam_direction[2].abs() < 0.0001 {
        std::f64::consts::FRAC_PI_2
    } else {
        (beam_direction[0].hypot(beam_direction[1]) / beam_direction[2]).atan()
    };
    [alpha1, alpha2, 0.0]
}

fn eels_qmesh_phi(theta_x: Real, theta_y: Real) -> Real {
    if theta_x.abs() < 0.000001 {
        if theta_y > 0.0 {
            std::f64::consts::FRAC_PI_2
        } else {
            -std::f64::consts::FRAC_PI_2
        }
    } else {
        let mut phi = (theta_y / theta_x).atan().abs();
        if theta_y < 0.0 && theta_x < 0.0 {
            phi += std::f64::consts::PI;
        } else if theta_x < 0.0 {
            phi = std::f64::consts::PI - phi;
        } else if theta_y < 0.0 {
            phi = -phi;
        }
        phi
    }
}
