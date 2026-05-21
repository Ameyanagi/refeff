//! FEFF path-finder packing helpers.
//!
//! `PATH/ipack.f90` stores up to eight path atom indices in three signed
//! integers by treating each packed field as base 1290. These helpers preserve
//! that representation for compatibility with FEFF path-degeneracy logic.
//! This module also ports the small companion-index min-heap maintenance
//! routines from `PATH/heap.f90`, the path distance/angle builder from
//! `PATH/mrb.f90`, and the path-pruning criteria from `PATH/mcrith.f90` and
//! `PATH/mcritk.f90`. Errors are structured instead of calling `par_stop`.

use ndarray::{Array2, ArrayView2, ArrayView3};

use crate::{Real, quadrature::strap, vector::single_precision_distance_between};

mod error;
mod phase;
mod types;

pub use error::PathError;
pub use phase::path_phase_criteria_tables;
pub use types::*;

#[cfg(test)]
pub(crate) use phase::single_precision_path_value;

const PATH_PACK_BASE: i32 = 1_290;
const PATH_PACK_BASE_SQUARED: i32 = PATH_PACK_BASE * PATH_PACK_BASE;
const MAX_PACKED_PATH_INDICES: usize = 8;
const MAX_PACKED_PATH_VALUE: i32 = PATH_PACK_BASE - 1;
const DOT_COSINE_EPSILON: f32 = 1.0e-8;
const PATH_HASH_SCALE: f32 = 1_000.0;
const PATH_HASH_FACTOR: f32 = 16.123_457;
const PATH_HASH_POTENTIAL_FACTOR: f32 = 8.576_543;
const PATH_HASH_Y_WEIGHT: f32 = 0.894_375;
const PATH_HASH_Z_WEIGHT: f32 = 0.573_498;
const PATH_HASH_LENGTH_OFFSET: Real = 40_000_000.0;
const PATH_OUTPUT_MIN_ABS_ANGLE_COSINE: f32 = 0.3;
const PATH_STANDARD_EPSILON: Real = 1.0e-4;

/// Port of FEFF `ipack`: pack up to eight path indices into three integers.
///
/// Each path index must be in `0..=1289`, matching the base used by FEFF. The
/// first packed integer stores the path length and the first two indices; the
/// second and third packed integers store the remaining six indices.
pub fn pack_path_indices(indices: &[i32]) -> Result<[i32; 3], PathError> {
    if indices.len() > MAX_PACKED_PATH_INDICES {
        return Err(PathError::TooManyIndices {
            count: indices.len(),
            max: MAX_PACKED_PATH_INDICES,
        });
    }

    let mut padded = [0; MAX_PACKED_PATH_INDICES];
    for (position, &value) in indices.iter().enumerate() {
        validate_path_index(position, value)?;
        padded[position] = value;
    }

    Ok([
        i32::try_from(indices.len()).map_err(|_| PathError::TooManyIndices {
            count: indices.len(),
            max: MAX_PACKED_PATH_INDICES,
        })? + padded[0] * PATH_PACK_BASE
            + padded[1] * PATH_PACK_BASE_SQUARED,
        padded[2] + padded[3] * PATH_PACK_BASE + padded[4] * PATH_PACK_BASE_SQUARED,
        padded[5] + padded[6] * PATH_PACK_BASE + padded[7] * PATH_PACK_BASE_SQUARED,
    ])
}

/// Port of FEFF `upack`: unpack a three-integer path representation.
///
/// `capacity` mirrors FEFF's caller-provided maximum `n`: it must be at most
/// eight and must be no smaller than the packed path length.
pub fn unpack_path_indices(packed: [i32; 3], capacity: usize) -> Result<Vec<i32>, PathError> {
    if capacity > MAX_PACKED_PATH_INDICES {
        return Err(PathError::InvalidUnpackCapacity {
            capacity,
            max: MAX_PACKED_PATH_INDICES,
        });
    }
    for (position, &value) in packed.iter().enumerate() {
        if value < 0 {
            return Err(PathError::NegativePackedValue { position, value });
        }
    }

    let packed_count = usize::try_from(packed[0] % PATH_PACK_BASE).map_err(|_| {
        PathError::NegativePackedValue {
            position: 0,
            value: packed[0],
        }
    })?;
    if packed_count > MAX_PACKED_PATH_INDICES {
        return Err(PathError::TooManyIndices {
            count: packed_count,
            max: MAX_PACKED_PATH_INDICES,
        });
    }
    if packed_count > capacity {
        return Err(PathError::UnpackCapacityTooSmall {
            packed_count,
            capacity,
        });
    }

    let unpacked = [
        (packed[0] % PATH_PACK_BASE_SQUARED) / PATH_PACK_BASE,
        packed[0] / PATH_PACK_BASE_SQUARED,
        packed[1] % PATH_PACK_BASE,
        (packed[1] % PATH_PACK_BASE_SQUARED) / PATH_PACK_BASE,
        packed[1] / PATH_PACK_BASE_SQUARED,
        packed[2] % PATH_PACK_BASE,
        (packed[2] % PATH_PACK_BASE_SQUARED) / PATH_PACK_BASE,
        packed[2] / PATH_PACK_BASE_SQUARED,
    ];

    Ok(unpacked[..packed_count].to_vec())
}

/// Port of FEFF `hup`: bubble the last min-heap element upward.
///
/// `keys` and `indices` are swapped together so callers can keep path metadata
/// associated with the heap key, matching FEFF's `h` and `ih` arrays.
pub fn path_heap_bubble_up(keys: &mut [Real], indices: &mut [i32]) -> Result<(), PathError> {
    validate_heap_inputs(keys, indices)?;
    let mut child = keys.len().saturating_sub(1);
    while child > 0 {
        let parent = (child - 1) / 2;
        if keys[child] < keys[parent] {
            keys.swap(child, parent);
            indices.swap(child, parent);
            child = parent;
        } else {
            break;
        }
    }
    Ok(())
}

/// Port of FEFF `hdown`: bubble the root min-heap element downward.
///
/// This is used after the root has been replaced. The function preserves FEFF's
/// choice of the smaller child and swaps the companion index array with the key.
pub fn path_heap_bubble_down(keys: &mut [Real], indices: &mut [i32]) -> Result<(), PathError> {
    validate_heap_inputs(keys, indices)?;
    let mut parent = 0;
    loop {
        let left = 2 * parent + 1;
        if left >= keys.len() {
            break;
        }
        let right = left + 1;
        let child = if right < keys.len() && keys[left] > keys[right] {
            right
        } else {
            left
        };

        if keys[parent] > keys[child] {
            keys.swap(parent, child);
            indices.swap(parent, child);
            parent = child;
        } else {
            break;
        }
    }
    Ok(())
}

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

/// Port of FEFF `mcrith`: heap importance for extending a partial path.
///
/// `None` corresponds to FEFF's undefined `xheap = -1` branch for paths ending
/// at the absorber or paths with two or fewer atoms. All arithmetic is performed
/// in single precision to match the original PATH code.
pub fn path_heap_criterion(
    path_indices: &[usize],
    leg_distances: &[Real],
    beta_indices: &[i32],
    atom_potentials: &[usize],
    fbeta_critical: ArrayView3<'_, Real>,
    wave_numbers: &[Real],
) -> Result<Option<Real>, PathError> {
    let path_atoms = validate_nonempty_path(path_indices)?;
    if path_indices[path_atoms - 1] == 0 || path_atoms <= 2 {
        return Ok(None);
    }

    let beta_offset = validate_criteria_inputs(
        path_indices,
        path_atoms,
        leg_distances,
        beta_indices,
        atom_potentials,
        fbeta_critical,
        wave_numbers,
    )?;

    let mut heap = 0.0_f32;
    let mut inverse_wave_sum = 0.0_f32;
    for criterion in 0..wave_numbers.len() {
        let wave_number = positive_f32("wave number", criterion, wave_numbers[criterion])?;
        let mut value = wave_number.powi(-((path_atoms as i32) - 1))
            * positive_f32(
                "leg distance",
                path_atoms - 2,
                leg_distances[path_atoms - 2],
            )?;

        for atom_position in 0..path_atoms - 2 {
            let potential = criteria_potential(
                path_indices,
                atom_potentials,
                fbeta_critical.dim().1,
                atom_position,
            )?;
            let beta_row =
                criteria_beta_row(beta_indices[atom_position], beta_offset, atom_position)?;
            let fbeta = finite_f32(
                "fbeta",
                atom_position,
                fbeta_critical[(beta_row, potential, criterion)],
            )?;
            let distance =
                positive_f32("leg distance", atom_position, leg_distances[atom_position])?;
            value = value * fbeta / distance;
        }

        inverse_wave_sum += 1.0 / wave_number;
        heap += value;
    }

    Ok(Some(Real::from(100.0_f32 * heap / inverse_wave_sum)))
}

/// Port of FEFF `mcritk`: output importance for keeping a complete path.
///
/// `None` corresponds to FEFF's undefined `xout = -1` branch for paths ending
/// at the absorber. `current_normalization` is FEFF `xcalcx`; when it is
/// nonpositive, the returned normalization is initialized from the current
/// path's raw `xcalc`.
pub fn path_output_criterion(
    input: PathOutputCriterionInput<'_>,
) -> Result<PathOutputCriterion, PathError> {
    let PathOutputCriterionInput {
        path_indices,
        leg_distances,
        angle_cosines,
        beta_indices,
        atom_potentials,
        fbeta_critical,
        mean_free_paths,
        wave_numbers,
        current_normalization,
    } = input;

    let path_atoms = validate_nonempty_path(path_indices)?;
    let normalization = finite_f32("normalization", 0, current_normalization)?;
    if path_indices[path_atoms - 1] == 0 {
        return Ok(PathOutputCriterion {
            output_importance: None,
            normalization: Real::from(normalization),
        });
    }

    let beta_offset = validate_criteria_inputs(
        path_indices,
        path_atoms,
        leg_distances,
        beta_indices,
        atom_potentials,
        fbeta_critical,
        wave_numbers,
    )?;
    if mean_free_paths.len() != wave_numbers.len() {
        return Err(PathError::PathCriteriaMeanFreePathCountMismatch {
            wave_numbers: wave_numbers.len(),
            mean_free_paths: mean_free_paths.len(),
        });
    }
    if angle_cosines.len() != path_atoms + 1 {
        return Err(PathError::PathCriteriaLengthMismatch {
            expected: path_atoms + 1,
            leg_distances: leg_distances.len(),
            beta_entries: angle_cosines.len(),
        });
    }

    let total_distance = leg_distances
        .iter()
        .enumerate()
        .map(|(index, &value)| positive_f32("leg distance", index, value))
        .try_fold(0.0_f32, |sum, value| Ok(sum + value?))?;

    let mut raw_output = 0.0_f32;
    for criterion in 0..wave_numbers.len() {
        let wave_number = positive_f32("wave number", criterion, wave_numbers[criterion])?;
        let mean_free_path = positive_f32("mean free path", criterion, mean_free_paths[criterion])?;
        let return_distance = positive_f32("leg distance", path_atoms, leg_distances[path_atoms])?;
        let mut value = finite_f32("angle cosine", path_atoms, angle_cosines[path_atoms])?
            .abs()
            .max(PATH_OUTPUT_MIN_ABS_ANGLE_COSINE)
            / (return_distance * wave_number);

        for atom_position in 0..path_atoms {
            let potential = criteria_potential(
                path_indices,
                atom_potentials,
                fbeta_critical.dim().1,
                atom_position,
            )?;
            let beta_row =
                criteria_beta_row(beta_indices[atom_position], beta_offset, atom_position)?;
            let fbeta = finite_f32(
                "fbeta",
                atom_position,
                fbeta_critical[(beta_row, potential, criterion)],
            )?;
            let distance =
                positive_f32("leg distance", atom_position, leg_distances[atom_position])?;
            value = value * fbeta / (distance * wave_number);
        }

        value *= (-total_distance / mean_free_path).exp();
        raw_output += value;
    }

    let updated_normalization = if normalization <= 0.0 {
        raw_output
    } else {
        normalization
    };
    Ok(PathOutputCriterion {
        output_importance: Some(Real::from(100.0_f32 * raw_output / updated_normalization)),
        normalization: Real::from(updated_normalization),
    })
}

/// Port of FEFF `ccrit`: decide whether a path is kept in the heap/output.
///
/// This combines FEFF `mrb`, beta-bin quantization, `mcrith`, `mcritk`, the
/// maximum-path-length cutoff, the central-atom partial-path branch, and the
/// cluster filter. FEFF's `xheap = -1` and `xout = -1` sentinels are represented
/// as `None` in the returned importance fields.
pub fn path_criteria_decision(
    input: PathCriteriaDecisionInput<'_>,
) -> Result<PathCriteriaDecision, PathError> {
    let PathCriteriaDecisionInput {
        atom_positions,
        path_indices,
        atom_potentials,
        cluster_outside,
        fbeta_critical,
        mean_free_paths,
        wave_numbers,
        max_path_length,
        heap_cutoff,
        output_cutoff,
        current_normalization,
    } = input;

    let path_atoms = validate_nonempty_path(path_indices)?;
    let max_path_length = finite_f32("max path length", 0, max_path_length)?;
    let heap_cutoff = finite_f32("heap cutoff", 0, heap_cutoff)?;
    let output_cutoff = finite_f32("output cutoff", 0, output_cutoff)?;
    let current_normalization = finite_f32("normalization", 0, current_normalization)?;

    let geometry = path_geometry(atom_positions, path_indices)?;
    let total_path_length = geometry.total_path_length;
    if (total_path_length as f32) > max_path_length {
        return Ok(PathCriteriaDecision {
            total_path_length,
            add_to_heap: false,
            keep_for_output: false,
            normalization: Real::from(current_normalization),
            heap_importance: None,
            output_importance: None,
        });
    }

    if path_indices[path_atoms - 1] == 0 {
        return Ok(PathCriteriaDecision {
            total_path_length,
            add_to_heap: true,
            keep_for_output: false,
            normalization: Real::from(current_normalization),
            heap_importance: None,
            output_importance: None,
        });
    }

    let beta_indices = path_beta_indices(&geometry.angle_cosines)?;
    let mut heap_importance = None;
    if heap_cutoff > 0.0 {
        heap_importance = path_heap_criterion(
            path_indices,
            &geometry.leg_distances,
            &beta_indices,
            atom_potentials,
            fbeta_critical,
            wave_numbers,
        )?;
        if heap_importance.is_some_and(|importance| importance < Real::from(heap_cutoff)) {
            return Ok(PathCriteriaDecision {
                total_path_length,
                add_to_heap: false,
                keep_for_output: false,
                normalization: Real::from(current_normalization),
                heap_importance,
                output_importance: None,
            });
        }
    }

    let mut keep_for_output = true;
    let mut normalization = Real::from(current_normalization);
    let mut output_importance = None;
    if output_cutoff > 0.0 {
        let output = path_output_criterion(PathOutputCriterionInput {
            path_indices,
            leg_distances: &geometry.leg_distances,
            angle_cosines: &geometry.angle_cosines,
            beta_indices: &beta_indices,
            atom_potentials,
            fbeta_critical,
            mean_free_paths,
            wave_numbers,
            current_normalization: normalization,
        })?;
        normalization = output.normalization;
        output_importance = output.output_importance;
        keep_for_output =
            output_importance.is_some_and(|importance| importance >= Real::from(output_cutoff));
    }

    if !path_has_cluster_atom(path_indices, cluster_outside)? {
        keep_for_output = false;
    }

    Ok(PathCriteriaDecision {
        total_path_length,
        add_to_heap: true,
        keep_for_output,
        normalization,
        heap_importance,
        output_importance,
    })
}

/// Port of FEFF `outcrt`: recalculate output importance and heap criteria.
///
/// This helper is used after path degeneracy/time-reversal handling. It
/// integrates the full-energy path importance (`xport`), computes heap
/// importance for both the current and time-reversed path directions, and
/// reuses `mcritk` for the output keep importance. FEFF's `-1` sentinels for
/// undefined heap/output values are represented as `None`.
pub fn path_output_importance(
    input: PathOutputImportanceInput<'_>,
) -> Result<PathOutputImportance, PathError> {
    let PathOutputImportanceInput {
        atom_positions,
        path_indices,
        atom_potentials,
        fbeta,
        wave_numbers,
        mean_free_paths,
        start_energy_index,
        fbeta_critical,
        critical_wave_numbers,
        critical_mean_free_paths,
        current_normalization,
    } = input;

    let geometry = path_geometry(atom_positions, path_indices)?;
    let beta_indices = path_beta_indices(&geometry.angle_cosines)?;
    let port_importance = path_port_importance(PathPortImportanceInput {
        path_indices,
        leg_distances: &geometry.leg_distances,
        angle_cosines: &geometry.angle_cosines,
        beta_indices: &beta_indices,
        atom_potentials,
        fbeta,
        wave_numbers,
        mean_free_paths,
        start_energy_index,
    })?;

    let heap_importance = path_heap_criterion(
        path_indices,
        &geometry.leg_distances,
        &beta_indices,
        atom_potentials,
        fbeta_critical,
        critical_wave_numbers,
    )?;

    let (reversed_path, reversed_distances, reversed_beta_indices) =
        reversed_heap_path(path_indices, &geometry.leg_distances, &beta_indices);
    let reversed_heap_importance = path_heap_criterion(
        &reversed_path,
        &reversed_distances,
        &reversed_beta_indices,
        atom_potentials,
        fbeta_critical,
        critical_wave_numbers,
    )?;

    let output = path_output_criterion(PathOutputCriterionInput {
        path_indices,
        leg_distances: &geometry.leg_distances,
        angle_cosines: &geometry.angle_cosines,
        beta_indices: &beta_indices,
        atom_potentials,
        fbeta_critical,
        mean_free_paths: critical_mean_free_paths,
        wave_numbers: critical_wave_numbers,
        current_normalization,
    })?;

    Ok(PathOutputImportance {
        port_importance,
        heap_importance,
        reversed_heap_importance,
        output_importance: output.output_importance,
        normalization: output.normalization,
    })
}

/// Convert FEFF `beta` angle cosines to `fbetac` grid indices.
///
/// This is the grid quantization used by `ccrit` and `outcrt`: nearest
/// multiple of `0.025`, with the sign of the original cosine preserved.
pub fn path_beta_indices(angle_cosines: &[Real]) -> Result<Vec<i32>, PathError> {
    angle_cosines
        .iter()
        .enumerate()
        .map(|(index, &angle)| path_beta_index(angle, index))
        .collect()
}

#[derive(Clone, Copy)]
struct PathPortImportanceInput<'a> {
    path_indices: &'a [usize],
    leg_distances: &'a [Real],
    angle_cosines: &'a [Real],
    beta_indices: &'a [i32],
    atom_potentials: &'a [usize],
    fbeta: ArrayView3<'a, Real>,
    wave_numbers: &'a [Real],
    mean_free_paths: &'a [Real],
    start_energy_index: usize,
}

fn path_port_importance(input: PathPortImportanceInput<'_>) -> Result<Real, PathError> {
    let PathPortImportanceInput {
        path_indices,
        leg_distances,
        angle_cosines,
        beta_indices,
        atom_potentials,
        fbeta,
        wave_numbers,
        mean_free_paths,
        start_energy_index,
    } = input;
    let path_atoms = validate_nonempty_path(path_indices)?;
    let beta_offset = validate_importance_inputs(input, path_atoms)?;

    let total_distance = leg_distances
        .iter()
        .enumerate()
        .map(|(index, &value)| positive_f32("leg distance", index, value))
        .try_fold(0.0_f32, |sum, value| Ok(sum + value?))?;

    let mut integration_waves = Vec::with_capacity(wave_numbers.len() - start_energy_index);
    let mut port_values = Vec::with_capacity(wave_numbers.len() - start_energy_index);
    for energy in start_energy_index..wave_numbers.len() {
        let wave_number = positive_f32("wave number", energy, wave_numbers[energy])?;
        let mean_free_path = positive_f32("mean free path", energy, mean_free_paths[energy])?;
        let return_distance = positive_f32("leg distance", path_atoms, leg_distances[path_atoms])?;
        let mut value = finite_f32("angle cosine", path_atoms, angle_cosines[path_atoms])?
            .abs()
            .max(PATH_OUTPUT_MIN_ABS_ANGLE_COSINE)
            / (return_distance * wave_number);

        for atom_position in 0..path_atoms {
            let potential =
                criteria_potential(path_indices, atom_potentials, fbeta.dim().1, atom_position)?;
            let beta_row =
                criteria_beta_row(beta_indices[atom_position], beta_offset, atom_position)?;
            let fbeta_value =
                finite_f32("fbeta", atom_position, fbeta[(beta_row, potential, energy)])?;
            let distance =
                positive_f32("leg distance", atom_position, leg_distances[atom_position])?;
            value = value * fbeta_value / (distance * wave_number);
        }

        value *= (-total_distance / mean_free_path).exp();
        integration_waves.push(Real::from(wave_number));
        port_values.push(Real::from(value.abs()));
    }

    strap(&integration_waves, &port_values).map_err(PathError::PathImportanceIntegration)
}

fn reversed_heap_path(
    path_indices: &[usize],
    leg_distances: &[Real],
    beta_indices: &[i32],
) -> (Vec<usize>, Vec<Real>, Vec<i32>) {
    let path_atoms = path_indices.len();
    let legs = path_atoms + 1;
    let reversed_distances = (0..legs)
        .map(|index| leg_distances[legs - 1 - index])
        .collect();
    let mut reversed_beta_indices = vec![0; legs];
    reversed_beta_indices[legs - 1] = beta_indices[legs - 1];
    for index in 0..path_atoms {
        reversed_beta_indices[index] = beta_indices[path_atoms - 1 - index];
    }
    let reversed_path = path_indices.iter().rev().copied().collect();
    (reversed_path, reversed_distances, reversed_beta_indices)
}

#[derive(Clone, Copy)]
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

fn path_beta_index(angle_cosine: Real, index: usize) -> Result<i32, PathError> {
    let angle = finite_f32("angle cosine", index, angle_cosine)?;
    let absolute = angle.abs();
    let mut beta_index = (absolute / 0.025_f32).trunc() as i32;
    let delta = absolute - beta_index as f32 * 0.025_f32;
    if delta > 0.0125_f32 {
        beta_index += 1;
    }
    if angle < 0.0 {
        beta_index = -beta_index;
    }
    Ok(beta_index)
}

fn path_has_cluster_atom(
    path_indices: &[usize],
    cluster_outside: &[bool],
) -> Result<bool, PathError> {
    let mut has_cluster_atom = false;
    for (position, &atom_index) in path_indices.iter().enumerate() {
        let Some(&outside) = cluster_outside.get(atom_index) else {
            return Err(PathError::PathCriteriaClusterIndexOutOfRange {
                position,
                atom_index,
                atoms: cluster_outside.len(),
            });
        };
        has_cluster_atom |= outside;
    }
    Ok(has_cluster_atom)
}

fn validate_nonempty_path(path_indices: &[usize]) -> Result<usize, PathError> {
    if path_indices.is_empty() {
        Err(PathError::EmptyPathCriteria)
    } else {
        Ok(path_indices.len())
    }
}

fn validate_criteria_inputs(
    path_indices: &[usize],
    path_atoms: usize,
    leg_distances: &[Real],
    beta_indices: &[i32],
    atom_potentials: &[usize],
    fbeta_critical: ArrayView3<'_, Real>,
    wave_numbers: &[Real],
) -> Result<i32, PathError> {
    let expected = path_atoms + 1;
    if leg_distances.len() != expected || beta_indices.len() != expected {
        return Err(PathError::PathCriteriaLengthMismatch {
            expected,
            leg_distances: leg_distances.len(),
            beta_entries: beta_indices.len(),
        });
    }

    let (beta_rows, potentials, criteria) = fbeta_critical.dim();
    if beta_rows == 0 || beta_rows % 2 == 0 || potentials == 0 || criteria == 0 {
        return Err(PathError::InvalidPathCriteriaTableShape {
            beta_rows,
            potentials,
            criteria,
        });
    }
    if wave_numbers.is_empty() || wave_numbers.len() > criteria {
        return Err(PathError::PathCriteriaWaveCountMismatch {
            wave_numbers: wave_numbers.len(),
            table_criteria: criteria,
        });
    }

    let beta_offset = (beta_rows as i32 - 1) / 2;
    for (atom_position, &beta_index) in beta_indices.iter().take(path_atoms).enumerate() {
        criteria_potential(path_indices, atom_potentials, potentials, atom_position)?;
        criteria_beta_row(beta_index, beta_offset, atom_position)?;
    }
    for (index, &value) in leg_distances.iter().enumerate() {
        positive_f32("leg distance", index, value)?;
    }
    for (index, &value) in wave_numbers.iter().enumerate() {
        positive_f32("wave number", index, value)?;
    }

    Ok(beta_offset)
}

fn validate_importance_inputs(
    input: PathPortImportanceInput<'_>,
    path_atoms: usize,
) -> Result<i32, PathError> {
    let PathPortImportanceInput {
        path_indices,
        leg_distances,
        angle_cosines,
        beta_indices,
        atom_potentials,
        fbeta,
        wave_numbers,
        mean_free_paths,
        start_energy_index,
    } = input;
    let expected = path_atoms + 1;
    if leg_distances.len() != expected || beta_indices.len() != expected {
        return Err(PathError::PathCriteriaLengthMismatch {
            expected,
            leg_distances: leg_distances.len(),
            beta_entries: beta_indices.len(),
        });
    }
    if angle_cosines.len() != expected {
        return Err(PathError::PathCriteriaLengthMismatch {
            expected,
            leg_distances: leg_distances.len(),
            beta_entries: angle_cosines.len(),
        });
    }

    let (beta_rows, potentials, energies) = fbeta.dim();
    if beta_rows == 0 || beta_rows % 2 == 0 || potentials == 0 || energies == 0 {
        return Err(PathError::InvalidPathImportanceTableShape {
            beta_rows,
            potentials,
            energies,
        });
    }
    if wave_numbers.is_empty() || wave_numbers.len() > energies {
        return Err(PathError::PathImportanceEnergyCountMismatch {
            wave_numbers: wave_numbers.len(),
            table_energies: energies,
        });
    }
    if mean_free_paths.len() != wave_numbers.len() {
        return Err(PathError::PathImportanceMeanFreePathCountMismatch {
            wave_numbers: wave_numbers.len(),
            mean_free_paths: mean_free_paths.len(),
        });
    }
    let remaining = wave_numbers.len().saturating_sub(start_energy_index);
    if remaining < 2 {
        return Err(PathError::PathImportanceStartOutOfRange {
            start: start_energy_index,
            remaining,
        });
    }

    let beta_offset = (beta_rows as i32 - 1) / 2;
    for (atom_position, &beta_index) in beta_indices.iter().take(path_atoms).enumerate() {
        criteria_potential(path_indices, atom_potentials, potentials, atom_position)?;
        criteria_beta_row(beta_index, beta_offset, atom_position)?;
    }
    for (index, &value) in leg_distances.iter().enumerate() {
        positive_f32("leg distance", index, value)?;
    }
    for (index, &value) in angle_cosines.iter().enumerate() {
        finite_f32("angle cosine", index, value)?;
    }
    for (index, &value) in wave_numbers.iter().enumerate() {
        positive_f32("wave number", index, value)?;
    }
    for (index, &value) in mean_free_paths.iter().enumerate() {
        positive_f32("mean free path", index, value)?;
    }

    Ok(beta_offset)
}

fn criteria_potential(
    path_indices: &[usize],
    atom_potentials: &[usize],
    potential_count: usize,
    position: usize,
) -> Result<usize, PathError> {
    let atom_index = path_indices[position];
    let Some(&potential) = atom_potentials.get(atom_index) else {
        return Err(PathError::PathCriteriaAtomIndexOutOfRange {
            position,
            atom_index,
            atoms: atom_potentials.len(),
        });
    };
    if potential >= potential_count {
        return Err(PathError::PathCriteriaPotentialOutOfRange {
            position,
            potential,
            potentials: potential_count,
        });
    }
    Ok(potential)
}

fn criteria_beta_row(
    beta_index: i32,
    beta_offset: i32,
    position: usize,
) -> Result<usize, PathError> {
    let min = -beta_offset;
    let max = beta_offset;
    if beta_index < min || beta_index > max {
        return Err(PathError::PathCriteriaBetaIndexOutOfRange {
            position,
            beta_index,
            min,
            max,
        });
    }
    Ok((beta_index + beta_offset) as usize)
}

fn finite_f32(quantity: &'static str, index: usize, value: Real) -> Result<f32, PathError> {
    let single = value as f32;
    if value.is_finite() && single.is_finite() {
        Ok(single)
    } else {
        Err(PathError::NonFinitePathCriteriaValue {
            quantity,
            index,
            value,
        })
    }
}

fn positive_f32(quantity: &'static str, index: usize, value: Real) -> Result<f32, PathError> {
    let single = finite_f32(quantity, index, value)?;
    if single > 0.0 {
        Ok(single)
    } else {
        Err(PathError::NonPositivePathCriteriaValue {
            quantity,
            index,
            value,
        })
    }
}

fn validate_path_index(position: usize, value: i32) -> Result<(), PathError> {
    if (0..=MAX_PACKED_PATH_VALUE).contains(&value) {
        Ok(())
    } else {
        Err(PathError::IndexOutOfRange {
            position,
            value,
            max: MAX_PACKED_PATH_VALUE,
        })
    }
}

fn validate_heap_inputs(keys: &[Real], indices: &[i32]) -> Result<(), PathError> {
    if keys.len() != indices.len() {
        return Err(PathError::HeapLengthMismatch {
            keys_len: keys.len(),
            indices_len: indices.len(),
        });
    }
    for (index, &value) in keys.iter().enumerate() {
        if !value.is_finite() {
            return Err(PathError::NonFiniteHeapKey { index, value });
        }
    }
    Ok(())
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

fn validate_position_shape(atom_positions: ArrayView2<'_, Real>) -> Result<(), PathError> {
    if atom_positions.nrows() == 0 || atom_positions.ncols() != 3 {
        Err(PathError::InvalidAtomPositionShape {
            rows: atom_positions.nrows(),
            columns: atom_positions.ncols(),
        })
    } else {
        Ok(())
    }
}

fn validate_atom_index(
    atom_positions: ArrayView2<'_, Real>,
    position: usize,
    atom_index: usize,
) -> Result<(), PathError> {
    if atom_index >= atom_positions.nrows() {
        Err(PathError::AtomIndexOutOfRange {
            position,
            atom_index,
            atoms: atom_positions.nrows(),
        })
    } else {
        Ok(())
    }
}

fn atom_position(
    atom_positions: ArrayView2<'_, Real>,
    position: usize,
    atom_index: usize,
) -> Result<[f32; 3], PathError> {
    validate_atom_index(atom_positions, position, atom_index)?;
    let mut point = [0.0_f32; 3];
    for component in 0..3 {
        let value = atom_positions[(atom_index, component)];
        if !value.is_finite() {
            return Err(PathError::NonFiniteAtomPosition {
                atom_index,
                component,
                value,
            });
        }
        point[component] = value as f32;
    }
    Ok(point)
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

#[cfg(test)]
mod tests;
