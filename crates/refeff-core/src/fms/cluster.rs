use super::*;

/// Port FEFF `yprep` absorber-centered FMS cluster-prefix selection.
///
/// The helper finds the first atom with `central_potential`, shifts all
/// coordinates so that atom is at the origin, sorts by FEFF's `athep` radial
/// key, counts the atoms inside `cluster_radius`, and truncates that prefix to
/// `cluster_capacity`. Rotation matrices and spherical-harmonic normalization
/// tables are prepared by separate FMS helpers.
pub fn fms_yprep_cluster(input: FmsYprepClusterInput<'_>) -> Result<FmsYprepCluster, FmsError> {
    let (rows, columns) = input.positions.dim();
    if columns != 3 {
        return Err(FmsError::AtomPositionColumnCount { columns });
    }
    if rows != input.potentials.len() {
        return Err(FmsError::AtomCountMismatch {
            potentials: input.potentials.len(),
            positions: rows,
        });
    }
    if !input.cluster_radius.is_finite() || input.cluster_radius < 0.0 {
        return Err(FmsError::InvalidClusterRadius);
    }
    if input.cluster_capacity == 0 {
        return Err(FmsError::EmptyClusterCapacity);
    }

    let mut central_atom = None;
    for (index, &potential) in input.potentials.iter().enumerate() {
        if potential == input.central_potential {
            if input.central_potential == 0 && central_atom.is_some() {
                return Err(FmsError::DuplicateAbsorber);
            }
            central_atom.get_or_insert(index);
        }
    }
    let central_atom = central_atom.ok_or(FmsError::MissingCentralAtom {
        potential: input.central_potential,
    })?;

    let center = [
        input.positions[(central_atom, 0)],
        input.positions[(central_atom, 1)],
        input.positions[(central_atom, 2)],
    ];
    ensure_finite_position(central_atom, center)?;

    let mut atoms = Vec::with_capacity(rows);
    for (atom, &potential) in input.potentials.iter().enumerate() {
        let position = [
            input.positions[(atom, 0)] - center[0],
            input.positions[(atom, 1)] - center[1],
            input.positions[(atom, 2)] - center[2],
        ];
        ensure_finite_position(atom, position)?;
        atoms.push(FmsAtom {
            position,
            potential,
        });
    }
    sort_atoms_by_radius(&mut atoms)?;

    let radius_squared = input.cluster_radius * input.cluster_radius;
    let first_outside = atoms
        .iter()
        .position(|atom| {
            let [x, y, z] = atom.position;
            x * x + y * y + z * z > radius_squared
        })
        .map_or(atoms.len(), |index| index);
    let untruncated_count = if first_outside == 0 {
        atoms.len()
    } else {
        first_outside
    };
    let included_count = untruncated_count.min(input.cluster_capacity);
    atoms.truncate(included_count);

    Ok(FmsYprepCluster {
        central_atom,
        untruncated_count,
        atoms,
    })
}

/// Port of FEFF `athep`: sort atoms by radius from the central atom.
///
/// The sort key is `x^2 + y^2 + z^2 + (input_index + 1) * 1e-6`, matching the
/// FEFF tie-breaker that preserves the old order for equidistant atoms. The
/// returned vector contains the sorted FEFF `ra` keys.
pub fn sort_atoms_by_radius(atoms: &mut [FmsAtom]) -> Result<Vec<f64>, FmsError> {
    let mut keyed_atoms = atoms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, atom)| sort_radius_key(index, atom).map(|key| (key, atom)))
        .collect::<Result<Vec<_>, _>>()?;

    keyed_atoms.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut keys = Vec::with_capacity(keyed_atoms.len());
    for (slot, (key, atom)) in atoms.iter_mut().zip(keyed_atoms) {
        *slot = atom;
        keys.push(key);
    }
    Ok(keys)
}

/// Port of FEFF `sortat`: move representative atoms into the FMS prefix.
///
/// The input atoms must already be sorted by radial distance. `max_potential`
/// is FEFF's inclusive `npot` loop bound; potential indices `0..=npot` are
/// considered. The returned vector maps each potential to its representative
/// zero-based atom index when that potential is present.
pub fn sort_representative_atoms(
    central_potential: i32,
    max_potential: usize,
    atoms: &mut [FmsAtom],
) -> Result<Vec<Option<usize>>, FmsError> {
    let central = checked_potential(central_potential, max_potential)?;
    let first = atoms
        .first()
        .ok_or(FmsError::AtomIndexOutOfRange { index: 0, len: 0 })?;
    if first.potential != central_potential {
        return Err(FmsError::CentralAtomMismatch {
            expected: central_potential,
            actual: first.potential,
        });
    }

    for (index, atom) in atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
        checked_potential(atom.potential, max_potential)?;
    }

    let mut representative = vec![None; max_potential + 1];
    representative[central] = Some(0);
    for (potential, slot) in representative.iter_mut().enumerate() {
        if potential == central {
            continue;
        }
        *slot = atoms
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, atom)| atom.potential == potential as i32)
            .map(|(index, _)| index);
    }

    for potential in 0..=max_potential {
        let Some(point) = representative[potential] else {
            continue;
        };
        if point <= potential {
            continue;
        }

        atoms.swap(potential, point);
        for slot in representative
            .iter_mut()
            .take(max_potential + 1)
            .skip(potential + 1)
        {
            if *slot == Some(potential) {
                *slot = Some(point);
            }
        }
        representative[potential] = Some(potential);
    }

    let prefix_len = atoms.len().min(max_potential + 1);
    for (potential, representative_slot) in representative.iter_mut().enumerate() {
        let Some(point) = *representative_slot else {
            continue;
        };
        let last_in_prefix = atoms
            .iter()
            .take(prefix_len)
            .enumerate()
            .filter(|(_, atom)| atom.potential == potential as i32)
            .map(|(index, _)| index)
            .next_back();

        if let Some(last_in_prefix) = last_in_prefix
            && last_in_prefix != point
        {
            let position = atoms[last_in_prefix].position;
            atoms[last_in_prefix].position = atoms[point].position;
            atoms[point].position = position;
            *representative_slot = Some(last_in_prefix);
        }
    }

    Ok(representative)
}

/// Port of FEFF `getang`: polar angles for the vector `positions[i] - positions[j]`.
///
/// Rust indices are zero-based. The returned values are `(theta, phi)` in
/// radians using FEFF's single-precision thresholds.
pub fn pair_polar_angles(
    positions: &[[f32; 3]],
    i: usize,
    j: usize,
) -> Result<(f32, f32), FmsError> {
    let left = checked_position(positions, i)?;
    let right = checked_position(positions, j)?;
    if i == j {
        return Ok((0.0, 0.0));
    }

    let x = left[0] - right[0];
    let y = left[1] - right[1];
    let z = left[2] - right[2];
    let r = (x * x + y * y + z * z).sqrt();

    const TINY: f32 = 1.0e-7;
    let phi = if x.abs() < TINY {
        if y.abs() < TINY {
            0.0
        } else if y > TINY {
            std::f32::consts::FRAC_PI_2
        } else {
            -std::f32::consts::FRAC_PI_2
        }
    } else {
        y.atan2(x)
    };

    let theta = if r <= TINY {
        0.0
    } else if z <= -r {
        std::f32::consts::PI
    } else if z < r {
        (z / r).acos()
    } else {
        0.0
    };

    Ok((theta, phi))
}
