use super::*;

/// Port of FEFF `mrb`: build leg distances and scattering-angle cosines.
///
/// `atom_positions` is indexed by FEFF atom number, with row `0` as the
/// absorber/central atom. `path_indices` are the scattering atoms in the path;
/// the final return to atom `0` is added internally. FEFF performs this
/// calculation in single precision through `sdist`, so Rust casts coordinates
/// through `f32` before evaluating distances and cosines.
pub fn path_geometry(
    atom_positions: ArrayView2<'_, Real>,
    path_indices: &[usize],
) -> Result<PathGeometry, PathError> {
    validate_position_shape(atom_positions)?;
    for (position, &atom_index) in path_indices.iter().enumerate() {
        validate_atom_index(atom_positions, position, atom_index)?;
    }

    let legs = path_indices.len() + 1;
    let mut leg_distances = Vec::with_capacity(legs);
    let mut angle_cosines = Vec::with_capacity(legs);
    let mut total_path_length = 0.0_f32;

    for leg in 0..legs {
        let previous = if leg == 0 { legs - 1 } else { leg - 1 };
        let next = if leg + 1 == legs { 0 } else { leg + 1 };
        let current_atom = path_atom_for_leg(path_indices, leg);
        let previous_atom = path_atom_for_leg(path_indices, previous);
        let next_atom = path_atom_for_leg(path_indices, next);

        let current = atom_position(atom_positions, leg, current_atom)?;
        let previous = atom_position(atom_positions, previous, previous_atom)?;
        let next = atom_position(atom_positions, next, next_atom)?;

        let distance = single_precision_distance_between(current, previous);
        total_path_length += distance;
        leg_distances.push(Real::from(distance));
        angle_cosines.push(Real::from(dot_cosine(previous, current, next)));
    }

    Ok(PathGeometry {
        leg_distances,
        angle_cosines,
        total_path_length: Real::from(total_path_length),
    })
}

/// Port of FEFF `mpprmd`: output distances, scattering angles, and eta angles.
///
/// `path_geometry` is the lightweight criteria helper and returns `cos(beta)`.
/// FEFF `mpprmd` is used for path output and returns `beta` as an angle in
/// radians plus the adjacent Euler-angle phase `eta`.
pub fn path_output_parameters(
    atom_positions: ArrayView2<'_, Real>,
    path_indices: &[usize],
) -> Result<PathOutputParameters, PathError> {
    validate_position_shape(atom_positions)?;
    let path_atoms = validate_nonempty_path(path_indices)?;
    for (position, &atom_index) in path_indices.iter().enumerate() {
        validate_atom_index(atom_positions, position, atom_index)?;
    }

    let legs = path_atoms + 1;
    let mut leg_distances = Vec::with_capacity(legs);
    let mut angle_cosines = Vec::with_capacity(legs);
    let mut alpha = Vec::with_capacity(legs);
    let mut gamma = Vec::with_capacity(legs + 1);

    for leg in 0..legs {
        let (current_atom, next_atom, previous_atom) = output_parameter_atoms(path_indices, leg);
        let current = atom_position(atom_positions, leg, current_atom)?;
        let next = atom_position(atom_positions, leg, next_atom)?;
        let previous = atom_position(atom_positions, leg, previous_atom)?;

        let (ct, st, cp, sp) = direction_trig(subtract_f32(next, current));
        let (ctp, stp, cpp, spp) = direction_trig(subtract_f32(current, previous));
        let cppp = cp * cpp + sp * spp;
        let sppp = spp * cp - cpp * sp;

        let alpha_real = st * ctp - ct * stp * cppp;
        let alpha_imag = -stp * sppp;
        let mut beta_cosine = ct * ctp + st * stp * cppp;
        beta_cosine = beta_cosine.clamp(-1.0, 1.0);
        let gamma_real = st * ctp * cppp - ct * stp;
        let gamma_imag = st * sppp;

        alpha.push((alpha_real, alpha_imag));
        gamma.push((gamma_real, gamma_imag));
        angle_cosines.push(beta_cosine);
        leg_distances.push(Real::from(single_precision_distance_between(
            current, previous,
        )));
    }

    gamma.push(gamma[0]);
    let eta_angles = alpha
        .iter()
        .zip(gamma.iter().skip(1))
        .map(|(&(alpha_real, alpha_imag), &(gamma_real, gamma_imag))| {
            let real = alpha_real * gamma_real - alpha_imag * gamma_imag;
            let imag = alpha_real * gamma_imag + alpha_imag * gamma_real;
            complex_argument_with_zero(real, imag)
        })
        .collect();
    let scattering_angles = angle_cosines
        .into_iter()
        .map(|cosine| cosine.clamp(-1.0, 1.0).acos())
        .collect();

    Ok(PathOutputParameters {
        leg_distances,
        scattering_angles,
        eta_angles,
    })
}

fn path_atom_for_leg(path_indices: &[usize], leg: usize) -> usize {
    if leg == path_indices.len() {
        0
    } else {
        path_indices[leg]
    }
}

fn output_parameter_atoms(path_indices: &[usize], leg: usize) -> (usize, usize, usize) {
    let path_atoms = path_indices.len();
    if leg == path_atoms {
        (0, path_indices[0], path_indices[path_atoms - 1])
    } else if leg == path_atoms - 1 {
        (
            path_indices[leg],
            0,
            if path_atoms == 1 {
                0
            } else {
                path_indices[path_atoms - 2]
            },
        )
    } else if leg == 0 {
        (
            path_indices[0],
            if path_atoms == 1 { 0 } else { path_indices[1] },
            0,
        )
    } else {
        (
            path_indices[leg],
            path_indices[leg + 1],
            path_indices[leg - 1],
        )
    }
}

fn subtract_f32(left: [f32; 3], right: [f32; 3]) -> [Real; 3] {
    [
        Real::from(left[0] - right[0]),
        Real::from(left[1] - right[1]),
        Real::from(left[2] - right[2]),
    ]
}

fn direction_trig(vector: [Real; 3]) -> (Real, Real, Real, Real) {
    let radius = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    let xy_radius = (vector[0] * vector[0] + vector[1] * vector[1]).sqrt();
    let (cos_theta, sin_theta) = if radius < 1.0e-6 {
        (1.0, 0.0)
    } else {
        (vector[2] / radius, xy_radius / radius)
    };
    let (cos_phi, sin_phi) = if xy_radius < 1.0e-6 {
        (1.0, 0.0)
    } else {
        (vector[0] / xy_radius, vector[1] / xy_radius)
    };
    (cos_theta, sin_theta, cos_phi, sin_phi)
}

fn complex_argument_with_zero(mut real: Real, mut imag: Real) -> Real {
    const EPSILON: Real = 1.0e-6;
    if real.abs() < EPSILON {
        real = 0.0;
    }
    if imag.abs() < EPSILON {
        imag = 0.0;
    }
    if real.abs() < EPSILON && imag.abs() < EPSILON {
        0.0
    } else {
        imag.atan2(real)
    }
}

fn dot_cosine(previous: [f32; 3], current: [f32; 3], next: [f32; 3]) -> f32 {
    let mut cosine = 0.0_f32;
    for component in 0..3 {
        cosine +=
            (current[component] - previous[component]) * (next[component] - current[component]);
    }
    let denominator = single_precision_distance_between(current, previous)
        * single_precision_distance_between(next, current);
    if denominator > DOT_COSINE_EPSILON {
        cosine / denominator
    } else {
        0.0
    }
}
