use super::*;

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
