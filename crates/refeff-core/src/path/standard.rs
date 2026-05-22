use super::*;

/// Port of FEFF `phash`: hash standardized path coordinates and potentials.
///
/// `local_positions` contains one row per scattering atom after FEFF's path
/// frame normalization, using columns `x`, `y`, and `z`. `potential_indices`
/// are the corresponding FEFF potential IDs, equivalent to `ipot(ipat(j))`.
/// FEFF hashes single-precision coordinates after multiplying by `1000` and
/// applying `NINT`; this function preserves that quantization.
pub fn path_degeneracy_hash(
    local_positions: ArrayView2<'_, Real>,
    potential_indices: &[i32],
) -> Result<Real, PathError> {
    validate_hash_input(local_positions, potential_indices)?;

    let mut hash = 0.0;
    let mut coordinate_factor = 1.0_f32;
    for position in 0..local_positions.nrows() {
        let x = rounded_hash_coordinate(local_positions[(position, 0)], position, 0)? as f32;
        let y = rounded_hash_coordinate(local_positions[(position, 1)], position, 1)? as f32;
        let z = rounded_hash_coordinate(local_positions[(position, 2)], position, 2)? as f32;
        let coordinate_term = x + y * PATH_HASH_Y_WEIGHT + z * PATH_HASH_Z_WEIGHT;
        hash += Real::from(coordinate_factor) * Real::from(coordinate_term);
        coordinate_factor *= PATH_HASH_FACTOR;
    }

    let mut potential_factor = 1.0_f32;
    for &potential in potential_indices {
        hash += Real::from(potential_factor) * Real::from(PATH_HASH_SCALE) * Real::from(potential);
        potential_factor *= PATH_HASH_POTENTIAL_FACTOR;
    }

    Ok(hash + local_positions.nrows() as Real * PATH_HASH_LENGTH_OFFSET)
}

/// Port of FEFF `mpprmp`: transform a path into its standard frame.
///
/// FEFF uses this normalized frame before path degeneracy hashing. The selected
/// symmetry case follows the original `ipol`/`ispin`/`evec`/`xivec` decision
/// tree unless `symmetry_case_override` is set. Coordinates are returned as an
/// `npat x 3` ndarray with columns `x`, `y`, and `z`.
pub fn path_standard_coordinates(
    input: PathStandardCoordinatesInput<'_>,
) -> Result<PathStandardCoordinates, PathError> {
    let PathStandardCoordinatesInput {
        atom_positions,
        path_indices,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
    } = input;
    let path_atoms = validate_nonempty_path(path_indices)?;
    validate_position_shape(atom_positions)?;
    let electric_vector = validate_standard_vector("electric vector", electric_vector)?;
    let incident_vector = validate_standard_vector("incident vector", incident_vector)?;
    let symmetry_case = path_symmetry_case(
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
    )?;

    let absorber = atom_position(atom_positions, 0, 0)?;
    let relative_positions = path_indices
        .iter()
        .enumerate()
        .map(|(position, &atom_index)| {
            let atom = atom_position(atom_positions, position, atom_index)?;
            Ok([
                Real::from(atom[0] - absorber[0]),
                Real::from(atom[1] - absorber[1]),
                Real::from(atom[2] - absorber[2]),
            ])
        })
        .collect::<Result<Vec<_>, PathError>>()?;

    let mut x_axis = [0.0; 3];
    let mut y_axis = [0.0; 3];
    let mut z_axis = standard_z_axis(symmetry_case, &relative_positions, electric_vector)?;
    let mut z_coordinates = relative_positions
        .iter()
        .map(|position| dot3(z_axis, *position))
        .collect::<Vec<_>>();

    if symmetry_case == 7 {
        x_axis[0] = 1.0;
        y_axis[1] = 1.0;
    } else {
        if symmetry_case != 1 && symmetry_case < 4 {
            orient_positive_z(&mut z_axis, &mut z_coordinates);
        }
        if let Some((new_x_axis, new_y_axis)) = standard_xy_axes(
            symmetry_case,
            &relative_positions,
            &z_coordinates,
            z_axis,
            incident_vector,
        ) {
            x_axis = new_x_axis;
            y_axis = new_y_axis;
        }
    }

    let mut x_coordinates = relative_positions
        .iter()
        .map(|position| dot3(x_axis, *position))
        .collect::<Vec<_>>();
    let mut y_coordinates = relative_positions
        .iter()
        .map(|position| dot3(y_axis, *position))
        .collect::<Vec<_>>();

    if symmetry_case == 3 {
        orient_first_positive(&mut x_coordinates);
    }
    if symmetry_case < 4 {
        orient_first_positive(&mut y_coordinates);
    }

    let mut coordinates = Array2::zeros((path_atoms, 3));
    for row in 0..path_atoms {
        coordinates[(row, 0)] = Real::from(x_coordinates[row] as f32);
        coordinates[(row, 1)] = Real::from(y_coordinates[row] as f32);
        coordinates[(row, 2)] = Real::from(z_coordinates[row] as f32);
    }

    Ok(PathStandardCoordinates {
        coordinates,
        symmetry_case,
    })
}

/// Port of FEFF `timrep`: choose the canonical time direction for a path.
///
/// FEFF hashes both the input path and its time reversal after transforming
/// each into standard coordinates, then keeps the representation with the
/// smaller hash. Reversal is disabled for spin-dependent polarized
/// calculations, matching the `ispin != 0 && ipol != 0` branch in `timrep`.
pub fn path_canonical_representation(
    input: PathCanonicalRepresentationInput<'_>,
) -> Result<PathCanonicalRepresentation, PathError> {
    let PathCanonicalRepresentationInput {
        atom_positions,
        path_indices,
        atom_potentials,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
        force_no_symmetry,
    } = input;
    validate_nonempty_path(path_indices)?;
    let symmetry_case_override =
        timrep_symmetry_case_override(symmetry_case_override, force_no_symmetry)?;

    let forward = path_representation_for_order(PathRepresentationForOrderInput {
        atom_positions,
        path_indices,
        atom_potentials,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
        reversed: false,
    })?;

    if path_indices.len() <= 1 || (spin != 0 && polarization != 0) {
        return Ok(forward);
    }

    let reversed_path = path_indices.iter().rev().copied().collect::<Vec<_>>();
    let reversed = path_representation_for_order(PathRepresentationForOrderInput {
        atom_positions,
        path_indices: &reversed_path,
        atom_potentials,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
        reversed: true,
    })?;

    if reversed.degeneracy_hash < forward.degeneracy_hash {
        Ok(reversed)
    } else {
        Ok(forward)
    }
}

struct PathRepresentationForOrderInput<'a> {
    atom_positions: ArrayView2<'a, Real>,
    path_indices: &'a [usize],
    atom_potentials: &'a [usize],
    polarization: i32,
    spin: i32,
    electric_vector: [Real; 3],
    incident_vector: [Real; 3],
    symmetry_case_override: Option<u8>,
    reversed: bool,
}

fn path_representation_for_order(
    input: PathRepresentationForOrderInput<'_>,
) -> Result<PathCanonicalRepresentation, PathError> {
    let PathRepresentationForOrderInput {
        atom_positions,
        path_indices,
        atom_potentials,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
        reversed,
    } = input;
    let standard = path_standard_coordinates(PathStandardCoordinatesInput {
        atom_positions,
        path_indices,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
    })?;
    let potential_indices = path_hash_potential_indices(path_indices, atom_potentials)?;
    let degeneracy_hash = path_degeneracy_hash(standard.coordinates.view(), &potential_indices)?;
    Ok(PathCanonicalRepresentation {
        path_indices: path_indices.to_vec(),
        coordinates: standard.coordinates,
        degeneracy_hash,
        reversed,
        symmetry_case: standard.symmetry_case,
    })
}

fn timrep_symmetry_case_override(
    symmetry_case_override: Option<u8>,
    force_no_symmetry: bool,
) -> Result<Option<u8>, PathError> {
    if let Some(symmetry_case @ 1..=7) = symmetry_case_override {
        Ok(Some(symmetry_case))
    } else if let Some(symmetry_case) = symmetry_case_override {
        Err(PathError::InvalidPathSymmetryCase { symmetry_case })
    } else if force_no_symmetry {
        Ok(Some(7))
    } else {
        Ok(None)
    }
}

fn path_hash_potential_indices(
    path_indices: &[usize],
    atom_potentials: &[usize],
) -> Result<Vec<i32>, PathError> {
    path_indices
        .iter()
        .enumerate()
        .map(|(position, &atom_index)| {
            let Some(&potential) = atom_potentials.get(atom_index) else {
                return Err(PathError::PathCriteriaAtomIndexOutOfRange {
                    position,
                    atom_index,
                    atoms: atom_potentials.len(),
                });
            };
            i32::try_from(potential).map_err(|_| PathError::PathHashPotentialOutOfRange {
                position,
                potential,
            })
        })
        .collect()
}

fn path_symmetry_case(
    polarization: i32,
    spin: i32,
    electric_vector: [Real; 3],
    incident_vector: [Real; 3],
    symmetry_case_override: Option<u8>,
) -> Result<u8, PathError> {
    if let Some(symmetry_case @ 1..=7) = symmetry_case_override {
        return Ok(symmetry_case);
    }
    if let Some(symmetry_case) = symmetry_case_override {
        return Err(PathError::InvalidPathSymmetryCase { symmetry_case });
    }

    let incident_is_vector = norm_squared(incident_vector) > PATH_STANDARD_EPSILON;
    let incident_is_z = incident_vector[0] * incident_vector[0]
        + incident_vector[1] * incident_vector[1]
        <= PATH_STANDARD_EPSILON;
    let electric_is_z = electric_vector[0] * electric_vector[0]
        + electric_vector[1] * electric_vector[1]
        <= PATH_STANDARD_EPSILON;

    let mut symmetry_case = 7;
    if polarization == 0 {
        symmetry_case = 1;
    } else if spin == 0 {
        if polarization == 1 && !incident_is_vector {
            symmetry_case = 2;
        }
        if polarization == 1 && incident_is_vector {
            symmetry_case = 3;
        }
        if polarization == 2 {
            symmetry_case = 4;
        }
    } else {
        if polarization == 2 && incident_is_z {
            symmetry_case = 5;
        }
        if polarization == 1 && !incident_is_vector && electric_is_z {
            symmetry_case = 5;
        }
        if polarization == 1
            && incident_is_z
            && electric_vector[2] * electric_vector[2] < PATH_STANDARD_EPSILON
        {
            symmetry_case = 6;
        }
    }
    Ok(symmetry_case)
}

fn standard_z_axis(
    symmetry_case: u8,
    relative_positions: &[[Real; 3]],
    electric_vector: [Real; 3],
) -> Result<[Real; 3], PathError> {
    if symmetry_case == 1 {
        let norm = norm_squared(relative_positions[0]).sqrt();
        if norm < PATH_STANDARD_EPSILON {
            return Err(PathError::DegeneratePathStandardAxis { symmetry_case });
        }
        Ok(scale3(relative_positions[0], 1.0 / norm))
    } else if symmetry_case == 2 || symmetry_case == 3 {
        Ok(electric_vector)
    } else {
        Ok([0.0, 0.0, 1.0])
    }
}

fn orient_positive_z(z_axis: &mut [Real; 3], z_coordinates: &mut [Real]) {
    let should_flip = z_coordinates
        .iter()
        .copied()
        .find(|coordinate| coordinate.abs() > PATH_STANDARD_EPSILON)
        .is_some_and(|coordinate| coordinate < 0.0);
    if should_flip {
        for component in z_axis {
            *component = -*component;
        }
        for coordinate in z_coordinates {
            *coordinate = -*coordinate;
        }
    }
}

fn standard_xy_axes(
    symmetry_case: u8,
    relative_positions: &[[Real; 3]],
    z_coordinates: &[Real],
    z_axis: [Real; 3],
    incident_vector: [Real; 3],
) -> Option<([Real; 3], [Real; 3])> {
    relative_positions
        .iter()
        .zip(z_coordinates)
        .find_map(|(&position, &z_coordinate)| {
            let radial_squared = (norm_squared(position) - z_coordinate * z_coordinate).abs();
            let radial = radial_squared.sqrt();
            if radial < PATH_STANDARD_EPSILON {
                return None;
            }

            let x_axis = match symmetry_case {
                1 | 2 | 4 | 5 => scale3(sub3(position, scale3(z_axis, z_coordinate)), 1.0 / radial),
                3 => incident_vector,
                _ => [if position[0] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
            };
            let y_axis = cross3(z_axis, x_axis);
            Some((x_axis, y_axis))
        })
}

fn orient_first_positive(values: &mut [Real]) {
    let should_flip = values
        .iter()
        .copied()
        .find(|value| value.abs() >= PATH_STANDARD_EPSILON)
        .is_some_and(|value| value < 0.0);
    if should_flip {
        for value in values {
            *value = -*value;
        }
    }
}

fn validate_standard_vector(
    vector_name: &'static str,
    vector: [Real; 3],
) -> Result<[Real; 3], PathError> {
    for (component, &value) in vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(PathError::NonFinitePathStandardVector {
                vector: vector_name,
                component,
                value,
            });
        }
    }
    Ok(vector)
}

fn dot3(left: [Real; 3], right: [Real; 3]) -> Real {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm_squared(vector: [Real; 3]) -> Real {
    dot3(vector, vector)
}

fn scale3(vector: [Real; 3], scale: Real) -> [Real; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn sub3(left: [Real; 3], right: [Real; 3]) -> [Real; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross3(left: [Real; 3], right: [Real; 3]) -> [Real; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn validate_hash_input(
    local_positions: ArrayView2<'_, Real>,
    potential_indices: &[i32],
) -> Result<(), PathError> {
    if local_positions.ncols() != 3 || local_positions.nrows() != potential_indices.len() {
        return Err(PathError::InvalidPathHashShape {
            rows: local_positions.nrows(),
            columns: local_positions.ncols(),
            potentials: potential_indices.len(),
        });
    }

    for (position, &value) in potential_indices.iter().enumerate() {
        if value < 0 {
            return Err(PathError::NegativePathPotential { position, value });
        }
    }
    Ok(())
}

fn rounded_hash_coordinate(
    value: Real,
    position: usize,
    component: usize,
) -> Result<i32, PathError> {
    let single = value as f32;
    if !value.is_finite() || !single.is_finite() {
        return Err(PathError::PathHashCoordinateOutOfRange {
            position,
            component,
            value,
        });
    }

    let scaled = single * PATH_HASH_SCALE;
    let rounded = scaled.round();
    let rounded_real = Real::from(rounded);
    if !rounded.is_finite()
        || rounded_real < Real::from(i32::MIN)
        || rounded_real > Real::from(i32::MAX)
    {
        return Err(PathError::PathHashCoordinateOutOfRange {
            position,
            component,
            value,
        });
    }
    Ok(rounded as i32)
}
