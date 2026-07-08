use super::*;
use crate::{sortir_order_1based, vector::single_precision_distance_between};

const PATHFINDER_BIG_DISTANCE: Real = 1.0e3;

/// Port of the FEFF `paths.f90` atom and neighbor-table preparation stage.
///
/// FEFF converts coordinates to single precision, moves the first `ipot == 0`
/// absorber to atom row `0`, computes `iclus`, and builds the sorted `m` table
/// used by the heap search. The first-bounce row corresponds to FEFF
/// `m(-1,0:nat)`.
pub fn pathfinder_preparation(
    input: PathfinderPreparationInput<'_>,
) -> Result<PathfinderPreparation, PathError> {
    validate_pathfinder_input(input)?;

    let fms_radius = input.fms_radius as f32;
    let mut atom_positions = single_precision_positions(input.atom_positions)?;
    let mut atom_potentials = input.atom_potentials.to_vec();
    let mut first_bounce_degeneracies = input.first_bounce_degeneracies.to_vec();
    first_bounce_degeneracies[0] = 0;

    let absorber_source_index = atom_potentials
        .iter()
        .position(|&potential| potential == 0)
        .ok_or(PathError::PathfinderMissingAbsorber)?;
    let absorber_position = row_position(atom_positions.view(), absorber_source_index);
    let mut cluster_outside = (0..atom_positions.nrows())
        .map(|atom| {
            single_precision_distance_between(
                row_position(atom_positions.view(), atom),
                absorber_position,
            ) > fms_radius
        })
        .collect::<Vec<_>>();

    if absorber_source_index != 0 {
        swap_position_rows(&mut atom_positions, 0, absorber_source_index);
        atom_potentials.swap(0, absorber_source_index);
        cluster_outside.swap(0, absorber_source_index);
    }

    let first_bounce_count = first_bounce_degeneracies
        .iter()
        .enumerate()
        .skip(1)
        .filter(|&(_, &degeneracy)| degeneracy > 0)
        .count();
    if first_bounce_count == 0 {
        return Err(PathError::PathfinderMissingFirstBounce);
    }

    let first_bounce_neighbors =
        pathfinder_neighbor_row(atom_positions.view(), None, &first_bounce_degeneracies)?;
    let mut neighbor_rows = Vec::with_capacity(atom_positions.nrows());
    for atom in 0..atom_positions.nrows() {
        neighbor_rows.push(pathfinder_neighbor_row(
            atom_positions.view(),
            Some(atom),
            &first_bounce_degeneracies,
        )?);
    }

    Ok(PathfinderPreparation {
        atom_positions,
        atom_potentials,
        first_bounce_degeneracies,
        cluster_outside,
        absorber_source_index,
        first_bounce_neighbors,
        neighbor_rows,
        first_bounce_count,
    })
}

/// Port of the FEFF `paths.f90` heap search that writes `paths.bin` records.
///
/// This consumes the prepared `m` rows, applies `ccrit` through the existing
/// Rust criteria port, and returns owned candidate records in FEFF traversal
/// order. File headers and final `pathsd` degeneracy handling are intentionally
/// kept outside this numerical core.
pub fn pathfinder_search(input: PathfinderSearchInput<'_>) -> Result<PathfinderSearch, PathError> {
    validate_pathfinder_search_input(input)?;
    if input.max_output_paths == 0 {
        return Ok(PathfinderSearch {
            records: Vec::new(),
            normalization: input.current_normalization,
            max_heap_size: 0,
            max_path_atoms_reached: 0,
            skipped_count: 0,
            complete: false,
        });
    }

    let mut normalization = input.current_normalization;
    let mut records = Vec::new();
    let mut skipped_count = 0_usize;
    let mut nodes = Vec::new();
    let mut heap_keys = Vec::new();
    let mut heap_indices = Vec::new();

    let initial_path = vec![input.preparation.first_bounce_neighbors[0]];
    let initial = pathfinder_decision(input, &initial_path, normalization)?;
    normalization = initial.normalization;
    if initial.add_to_heap && initial.total_path_length <= input.max_path_length {
        nodes.push(PathfinderHeapNode {
            path_indices: initial_path,
            neighbor_source: None,
            neighbor_column: 0,
            keep_for_output: initial.keep_for_output,
        });
        heap_keys.push(initial.total_path_length);
        heap_indices.push(0);
    }

    let mut max_heap_size = heap_keys.len();
    let mut max_path_atoms_reached = nodes.first().map_or(0, |node| node.path_indices.len());
    let mut complete = heap_keys.is_empty();

    while !heap_keys.is_empty() {
        if heap_keys[0] > input.max_path_length {
            complete = input.max_path_length < 1.0;
            break;
        }
        if records.len() >= input.max_output_paths {
            complete = false;
            break;
        }

        let root_index = heap_indices[0] as usize;
        let saved_path = nodes[root_index].path_indices.clone();
        let saved_length = heap_keys[0];
        let saved_keep = nodes[root_index].keep_for_output;
        if saved_path.last().copied() != Some(0) && saved_keep {
            records.push(PathfinderRecord {
                total_path_length: saved_length,
                path_indices: saved_path.clone(),
            });
            if records.len() >= input.max_output_paths {
                complete = false;
                break;
            }
        }

        if advance_root_candidate(
            input,
            &mut nodes[root_index],
            &mut normalization,
            &mut heap_keys[0],
        )? {
            path_heap_bubble_down(&mut heap_keys, &mut heap_indices)?;
        } else {
            remove_heap_root(&mut heap_keys, &mut heap_indices)?;
            skipped_count += 1;
        }

        if saved_path.len() < input.max_path_atoms {
            if add_extended_candidate(
                input,
                &saved_path,
                &mut normalization,
                &mut nodes,
                &mut heap_keys,
                &mut heap_indices,
            )? {
                max_heap_size = max_heap_size.max(heap_keys.len());
            } else {
                skipped_count += 1;
            }
        }

        if let Some(&heap_node_index) = heap_indices.first() {
            max_path_atoms_reached =
                max_path_atoms_reached.max(nodes[heap_node_index as usize].path_indices.len());
        } else {
            complete = true;
        }
    }

    Ok(PathfinderSearch {
        records,
        normalization,
        max_heap_size,
        max_path_atoms_reached,
        skipped_count,
        complete,
    })
}

/// Compose FEFF `paths.f90` heap search with the `pathsd` reduction core.
///
/// This is the numerical path from normalized atom inputs to unique retained
/// path groups. Text/binary handoff files and `paths.dat` rendering remain in
/// the IO/CLI layers.
pub fn pathfinder_reduction(
    input: PathfinderReductionInput<'_>,
) -> Result<PathfinderReduction, PathError> {
    let preparation = pathfinder_preparation(PathfinderPreparationInput {
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        first_bounce_degeneracies: input.first_bounce_degeneracies,
        fms_radius: input.fms_radius,
    })?;
    let search = pathfinder_search(PathfinderSearchInput {
        preparation: &preparation,
        max_path_length: input.max_path_length,
        max_path_atoms: input.max_path_atoms,
        max_output_paths: input.max_output_paths,
        heap_cutoff: input.heap_cutoff,
        output_cutoff: input.output_cutoff,
        fbeta_critical: input.fbeta_critical,
        critical_wave_numbers: input.critical_wave_numbers,
        critical_mean_free_paths: input.critical_mean_free_paths,
        current_normalization: input.search_normalization,
    })?;
    let records = search
        .records
        .iter()
        .map(|record| PathDegeneracyRecord {
            total_path_length: record.total_path_length,
            path_indices: &record.path_indices,
        })
        .collect::<Vec<_>>();
    let reduction = path_degeneracy_reduction(PathDegeneracyReductionInput {
        atom_positions: preparation.atom_positions.view(),
        atom_potentials: &preparation.atom_potentials,
        first_bounce_degeneracies: &preparation.first_bounce_degeneracies,
        records: &records,
        polarization: input.polarization,
        spin: input.spin,
        electric_vector: input.electric_vector,
        incident_vector: input.incident_vector,
        symmetry_case_override: input.symmetry_case_override,
        force_no_symmetry: input.force_no_symmetry,
        fbeta: input.fbeta,
        wave_numbers: input.wave_numbers,
        mean_free_paths: input.mean_free_paths,
        start_energy_index: input.start_energy_index,
        fbeta_critical: input.fbeta_critical,
        critical_wave_numbers: input.critical_wave_numbers,
        critical_mean_free_paths: input.critical_mean_free_paths,
        current_normalization: input.reduction_normalization,
        criterion_percent: input.criterion_percent,
        retention_reference: input.retention_reference,
    })?;

    Ok(PathfinderReduction {
        preparation,
        search,
        reduction,
    })
}

#[derive(Debug, Clone)]
struct PathfinderHeapNode {
    path_indices: Vec<usize>,
    neighbor_source: Option<usize>,
    neighbor_column: usize,
    keep_for_output: bool,
}

fn validate_pathfinder_input(input: PathfinderPreparationInput<'_>) -> Result<(), PathError> {
    validate_position_shape(input.atom_positions)?;
    let positions = input.atom_positions.nrows();
    if input.atom_potentials.len() != positions
        || input.first_bounce_degeneracies.len() != positions
    {
        return Err(PathError::PathfinderPreparationLengthMismatch {
            positions,
            potentials: input.atom_potentials.len(),
            first_bounce_degeneracies: input.first_bounce_degeneracies.len(),
        });
    }
    if !input.fms_radius.is_finite() {
        return Err(PathError::NonFinitePathfinderFmsRadius {
            value: input.fms_radius,
        });
    }
    Ok(())
}

fn validate_pathfinder_search_input(input: PathfinderSearchInput<'_>) -> Result<(), PathError> {
    if input.max_path_atoms == 0 {
        return Err(PathError::InvalidPathfinderSearchLimit {
            quantity: "max_path_atoms",
            value: input.max_path_atoms,
        });
    }
    if input.preparation.first_bounce_neighbors.is_empty() {
        return Err(PathError::PathfinderMissingFirstBounce);
    }
    Ok(())
}

fn single_precision_positions(
    atom_positions: ArrayView2<'_, Real>,
) -> Result<Array2<Real>, PathError> {
    let mut positions = Array2::zeros((atom_positions.nrows(), 3));
    for atom in 0..atom_positions.nrows() {
        let point = atom_position(atom_positions, atom, atom)?;
        for component in 0..3 {
            positions[(atom, component)] = Real::from(point[component]);
        }
    }
    Ok(positions)
}

fn pathfinder_decision(
    input: PathfinderSearchInput<'_>,
    path_indices: &[usize],
    normalization: Real,
) -> Result<PathCriteriaDecision, PathError> {
    path_criteria_decision(PathCriteriaDecisionInput {
        atom_positions: input.preparation.atom_positions.view(),
        path_indices,
        atom_potentials: &input.preparation.atom_potentials,
        cluster_outside: &input.preparation.cluster_outside,
        fbeta_critical: input.fbeta_critical,
        mean_free_paths: input.critical_mean_free_paths,
        wave_numbers: input.critical_wave_numbers,
        max_path_length: input.max_path_length,
        heap_cutoff: input.heap_cutoff,
        output_cutoff: input.output_cutoff,
        current_normalization: normalization,
    })
}

fn advance_root_candidate(
    input: PathfinderSearchInput<'_>,
    node: &mut PathfinderHeapNode,
    normalization: &mut Real,
    heap_key: &mut Real,
) -> Result<bool, PathError> {
    node.neighbor_column += 1;
    let Some(candidate) = replacement_candidate(input.preparation, node)? else {
        return Ok(false);
    };
    let Some(last) = node.path_indices.len().checked_sub(1) else {
        return Ok(false);
    };
    node.path_indices[last] = candidate;
    let decision = pathfinder_decision(input, &node.path_indices, *normalization)?;
    *normalization = decision.normalization;
    if decision.total_path_length > input.max_path_length || !decision.add_to_heap {
        return Ok(false);
    }
    *heap_key = decision.total_path_length;
    node.keep_for_output = decision.keep_for_output;
    Ok(true)
}

fn replacement_candidate(
    preparation: &PathfinderPreparation,
    node: &PathfinderHeapNode,
) -> Result<Option<usize>, PathError> {
    let atom_count = preparation.atom_positions.nrows();
    let row = neighbor_row(preparation, node.neighbor_source);
    if node.neighbor_column >= row.len() {
        return Ok(None);
    }
    let candidate = row[node.neighbor_column];
    if node.neighbor_source.is_none()
        && preparation
            .first_bounce_degeneracies
            .get(candidate)
            .copied()
            .unwrap_or(0)
            == 0
    {
        return Ok(None);
    }
    if node.neighbor_column >= atom_count.saturating_sub(1) {
        return Ok(None);
    }
    Ok(Some(candidate))
}

fn add_extended_candidate(
    input: PathfinderSearchInput<'_>,
    saved_path: &[usize],
    normalization: &mut Real,
    nodes: &mut Vec<PathfinderHeapNode>,
    heap_keys: &mut Vec<Real>,
    heap_indices: &mut Vec<i32>,
) -> Result<bool, PathError> {
    let Some(&last_atom) = saved_path.last() else {
        return Ok(false);
    };
    let Some(&next_atom) = input
        .preparation
        .neighbor_rows
        .get(last_atom)
        .and_then(|row| row.first())
    else {
        return Ok(false);
    };

    let mut path_indices = Vec::with_capacity(saved_path.len() + 1);
    path_indices.extend_from_slice(saved_path);
    path_indices.push(next_atom);
    let decision = pathfinder_decision(input, &path_indices, *normalization)?;
    *normalization = decision.normalization;
    if decision.total_path_length > input.max_path_length || !decision.add_to_heap {
        return Ok(false);
    }

    let node_index = i32::try_from(nodes.len())
        .map_err(|_| PathError::PathfinderHeapNodeOverflow { nodes: nodes.len() })?;
    nodes.push(PathfinderHeapNode {
        path_indices,
        neighbor_source: Some(last_atom),
        neighbor_column: 0,
        keep_for_output: decision.keep_for_output,
    });
    heap_keys.push(decision.total_path_length);
    heap_indices.push(node_index);
    path_heap_bubble_up(heap_keys, heap_indices)?;
    Ok(true)
}

fn remove_heap_root(
    heap_keys: &mut Vec<Real>,
    heap_indices: &mut Vec<i32>,
) -> Result<(), PathError> {
    let Some(last) = heap_keys.len().checked_sub(1) else {
        return Ok(());
    };
    heap_keys.swap(0, last);
    heap_indices.swap(0, last);
    heap_keys.pop();
    heap_indices.pop();
    if !heap_keys.is_empty() {
        path_heap_bubble_down(heap_keys, heap_indices)?;
    }
    Ok(())
}

fn neighbor_row(preparation: &PathfinderPreparation, source_atom: Option<usize>) -> &[usize] {
    match source_atom {
        None => &preparation.first_bounce_neighbors,
        Some(atom) => &preparation.neighbor_rows[atom],
    }
}

fn pathfinder_neighbor_row(
    atom_positions: ArrayView2<'_, Real>,
    source_atom: Option<usize>,
    first_bounce_degeneracies: &[usize],
) -> Result<Vec<usize>, PathError> {
    let row = source_atom.map_or(-1, |atom| atom as isize);
    let source = source_atom.unwrap_or(0);
    let source_position = row_position(atom_positions, source);
    let absorber_position = row_position(atom_positions, 0);
    let mut keys = Vec::with_capacity(atom_positions.nrows());

    for (atom, &first_bounce_degeneracy) in first_bounce_degeneracies
        .iter()
        .enumerate()
        .take(atom_positions.nrows())
    {
        let atom_position = row_position(atom_positions, atom);
        let mut key = Real::from(
            single_precision_distance_between(source_position, atom_position)
                + single_precision_distance_between(atom_position, absorber_position),
        );
        if atom == source {
            key = PATHFINDER_BIG_DISTANCE;
        }
        if source_atom.is_none() && first_bounce_degeneracy == 0 {
            key = PATHFINDER_BIG_DISTANCE;
        }
        keys.push(key);
    }

    sortir_order_1based(&keys)
        .map_err(|source| PathError::PathfinderNeighborSort { row, source })
        .map(|order| order.into_iter().map(|index| index - 1).collect())
}

fn row_position(atom_positions: ArrayView2<'_, Real>, atom: usize) -> [f32; 3] {
    [
        atom_positions[(atom, 0)] as f32,
        atom_positions[(atom, 1)] as f32,
        atom_positions[(atom, 2)] as f32,
    ]
}

fn swap_position_rows(atom_positions: &mut Array2<Real>, left: usize, right: usize) {
    for component in 0..3 {
        atom_positions.swap((left, component), (right, component));
    }
}
