//! FEFF DMDW path descriptor and path-motion helpers.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::Real;

use super::{
    DebyeError, DmdwExpandedPath, DmdwPathDescriptor, DmdwPathMotion, dmdw_director_sum, dot,
    ensure_finite_output, ensure_nonnegative, ensure_positive, validate_dmdw_atom_positions,
    validate_dmdw_atoms,
};

/// Port the DMDW `Calc_DW` path mass and initial-vector setup.
///
/// `atom_positions_angstrom` is the DMDW atom table as `(atom, xyz)`,
/// `atom_masses` is FEFF `dym_In%am`, and `path_atoms` is FEFF `lpath` after
/// conversion to zero-based local atom indices. The returned `initial_vector`
/// matches FEFF's component-major `qj0` layout: all x components, then all y,
/// then all z components.
pub fn dmdw_path_motion(
    atom_positions_angstrom: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    path_atoms: &[usize],
) -> Result<DmdwPathMotion, DebyeError> {
    validate_dmdw_path_input(atom_positions_angstrom, atom_masses, path_atoms)?;

    let inverse_reduced_mass = if path_atoms.len() == 1 {
        1.0 / atom_masses[path_atoms[0]]
    } else {
        path_atoms
            .iter()
            .enumerate()
            .map(|(path_index, &atom)| {
                let previous = path_atoms[(path_index + path_atoms.len() - 1) % path_atoms.len()];
                let next = path_atoms[(path_index + 1) % path_atoms.len()];
                let director_sum =
                    dmdw_director_sum(atom_positions_angstrom, atom, previous, next)?;
                Ok(dot(director_sum, director_sum) / (4.0 * atom_masses[atom]))
            })
            .sum::<Result<Real, DebyeError>>()?
    };

    ensure_positive("mu_inv", inverse_reduced_mass)?;
    let reduced_mass = 1.0 / inverse_reduced_mass;
    let mut initial_vector = Array1::<Real>::zeros(atom_positions_angstrom.nrows() * 3);

    if path_atoms.len() > 1 {
        for (path_index, &atom) in path_atoms.iter().enumerate() {
            let previous = path_atoms[(path_index + path_atoms.len() - 1) % path_atoms.len()];
            let next = path_atoms[(path_index + 1) % path_atoms.len()];
            let director_sum = dmdw_director_sum(atom_positions_angstrom, atom, previous, next)?;
            let scale = 0.5 * (reduced_mass / atom_masses[atom]).sqrt();
            for component in 0..3 {
                initial_vector[component * atom_positions_angstrom.nrows() + atom] =
                    scale * director_sum[component];
            }
        }
    }

    ensure_finite_output("mu", reduced_mass)?;
    for value in initial_vector.iter().copied() {
        ensure_finite_output("qj0", value)?;
    }

    Ok(DmdwPathMotion {
        reduced_mass,
        inverse_reduced_mass,
        initial_vector,
    })
}

/// Port FEFF DMDW `Paths_Init` descriptor expansion and path pruning.
///
/// `atom_positions` is the DMDW atom table as `(atom, xyz)`. It must use the
/// same distance unit as `descriptor.max_effective_length`. Descriptor
/// selectors preserve FEFF's convention: `0` expands over all atoms, while a
/// positive selector chooses that 1-based atom index. Returned atom indices are
/// zero-based for Rust callers.
pub fn dmdw_expand_path_descriptor(
    atom_positions: ArrayView2<'_, Real>,
    descriptor: &DmdwPathDescriptor,
) -> Result<Vec<DmdwExpandedPath>, DebyeError> {
    validate_dmdw_path_descriptor(atom_positions, descriptor)?;

    if descriptor.selectors.len() == 1 {
        return Ok(dmdw_expand_single_atom_descriptor(
            atom_positions.nrows(),
            descriptor.selectors[0],
        ));
    }

    let atom_count = atom_positions.nrows();
    let selector_ranges = descriptor
        .selectors
        .iter()
        .copied()
        .map(|selector| dmdw_selector_range(selector, atom_count))
        .collect::<Result<Vec<_>, DebyeError>>()?;
    let strides = dmdw_descriptor_loop_strides(
        selector_ranges.iter().map(Vec::len),
        descriptor.selectors.len(),
        atom_count,
    )?;
    let total_paths = strides.total_count;
    let mut paths = Vec::new();

    for path_index in 0..total_paths {
        let atoms = dmdw_descriptor_atoms_for_index(path_index, &selector_ranges, &strides.strides);
        if dmdw_path_has_pruned_repetition(&atoms) {
            continue;
        }

        let effective_length = dmdw_effective_path_length(atom_positions, &atoms)?;
        if effective_length <= descriptor.max_effective_length {
            paths.push(DmdwExpandedPath {
                atoms,
                effective_length,
            });
        }
    }

    Ok(paths)
}

/// Expand a sequence of FEFF DMDW path descriptors in input order.
pub fn dmdw_expand_path_descriptors(
    atom_positions: ArrayView2<'_, Real>,
    descriptors: &[DmdwPathDescriptor],
) -> Result<Vec<DmdwExpandedPath>, DebyeError> {
    let mut paths = Vec::new();
    for descriptor in descriptors {
        paths.extend(dmdw_expand_path_descriptor(atom_positions, descriptor)?);
    }
    Ok(paths)
}

struct DmdwDescriptorLoopStrides {
    strides: Vec<usize>,
    total_count: usize,
}

fn validate_dmdw_path_descriptor(
    atom_positions: ArrayView2<'_, Real>,
    descriptor: &DmdwPathDescriptor,
) -> Result<(), DebyeError> {
    validate_dmdw_atom_positions(atom_positions)?;
    if descriptor.selectors.is_empty() {
        return Err(DebyeError::EmptyDmdwPath);
    }
    ensure_nonnegative(
        "DMDW path descriptor maximum effective length",
        descriptor.max_effective_length,
    )?;
    for &selector in &descriptor.selectors {
        validate_dmdw_path_selector(selector, atom_positions.nrows())?;
    }
    Ok(())
}

fn validate_dmdw_path_selector(selector: i32, atom_count: usize) -> Result<(), DebyeError> {
    if selector < 0 || selector as usize > atom_count {
        Err(DebyeError::InvalidDmdwPathSelector {
            selector,
            atom_count,
        })
    } else {
        Ok(())
    }
}

fn dmdw_expand_single_atom_descriptor(atom_count: usize, selector: i32) -> Vec<DmdwExpandedPath> {
    let atoms = if selector == 0 {
        (0..atom_count).collect::<Vec<_>>()
    } else {
        vec![selector as usize - 1]
    };
    atoms
        .into_iter()
        .map(|atom| DmdwExpandedPath {
            atoms: vec![atom],
            effective_length: 0.0,
        })
        .collect()
}

fn dmdw_selector_range(selector: i32, atom_count: usize) -> Result<Vec<usize>, DebyeError> {
    validate_dmdw_path_selector(selector, atom_count)?;
    if selector == 0 {
        Ok((0..atom_count).collect())
    } else {
        Ok(vec![selector as usize - 1])
    }
}

fn dmdw_descriptor_loop_strides(
    lengths: impl Iterator<Item = usize>,
    selector_count: usize,
    atom_count: usize,
) -> Result<DmdwDescriptorLoopStrides, DebyeError> {
    let lengths = lengths.collect::<Vec<_>>();
    let mut strides = vec![1; lengths.len()];
    let mut total_count = 1usize;
    for index in (0..lengths.len()).rev() {
        strides[index] = total_count;
        total_count = total_count.checked_mul(lengths[index]).ok_or(
            DebyeError::DmdwPathExpansionTooLarge {
                selectors: selector_count,
                atom_count,
            },
        )?;
    }
    Ok(DmdwDescriptorLoopStrides {
        strides,
        total_count,
    })
}

fn dmdw_descriptor_atoms_for_index(
    path_index: usize,
    selector_ranges: &[Vec<usize>],
    strides: &[usize],
) -> Vec<usize> {
    let mut remainder = path_index;
    selector_ranges
        .iter()
        .zip(strides.iter())
        .map(|(range, &stride)| {
            let range_index = remainder / stride;
            remainder %= stride;
            range[range_index]
        })
        .collect()
}

fn dmdw_path_has_pruned_repetition(atoms: &[usize]) -> bool {
    let has_consecutive_repeat = atoms.windows(2).any(|pair| pair[0] == pair[1]);
    let closes_on_same_atom = match (atoms.first(), atoms.last()) {
        (Some(first), Some(last)) => first == last,
        _ => false,
    };
    has_consecutive_repeat || closes_on_same_atom
}

fn dmdw_effective_path_length(
    atom_positions: ArrayView2<'_, Real>,
    atoms: &[usize],
) -> Result<Real, DebyeError> {
    if atoms.len() <= 1 {
        return Ok(0.0);
    }

    let segment_length = atoms
        .windows(2)
        .map(|pair| dmdw_atom_distance(atom_positions, pair[0], pair[1]))
        .sum::<Result<Real, DebyeError>>()?;
    let closing_length = dmdw_atom_distance(atom_positions, atoms[0], atoms[atoms.len() - 1])?;
    let effective_length = 0.5 * (segment_length + closing_length);
    ensure_finite_output("DMDW effective path length", effective_length)?;
    Ok(effective_length)
}

fn dmdw_atom_distance(
    atom_positions: ArrayView2<'_, Real>,
    left: usize,
    right: usize,
) -> Result<Real, DebyeError> {
    let squared = (0..3)
        .map(|component| {
            let difference = atom_positions[(left, component)] - atom_positions[(right, component)];
            difference * difference
        })
        .sum::<Real>();
    let distance = squared.sqrt();
    ensure_finite_output("DMDW atom distance", distance)?;
    Ok(distance)
}

fn validate_dmdw_path_input(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    path_atoms: &[usize],
) -> Result<(), DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    if path_atoms.is_empty() {
        return Err(DebyeError::EmptyDmdwPath);
    }
    for &index in path_atoms {
        if index >= atom_positions.nrows() {
            return Err(DebyeError::InvalidDmdwPathAtomIndex {
                index,
                atom_count: atom_positions.nrows(),
            });
        }
    }
    Ok(())
}
