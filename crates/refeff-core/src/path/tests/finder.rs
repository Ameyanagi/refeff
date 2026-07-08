use super::{support::*, *};
use ndarray::{Array3, arr2};

#[test]
fn pathfinder_search_emits_paths_bin_order_for_disabled_criteria() -> Result<(), PathError> {
    let preparation = simple_pathfinder_preparation()?;
    let fbetac = Array3::zeros((1, 1, 1));

    let search = pathfinder_search(PathfinderSearchInput {
        preparation: &preparation,
        max_path_length: 4.1,
        max_path_atoms: 2,
        max_output_paths: 8,
        heap_cutoff: 0.0,
        output_cutoff: 0.0,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &[],
        critical_mean_free_paths: &[],
        current_normalization: -1.0,
    })?;

    let paths = search
        .records
        .iter()
        .map(|record| record.path_indices.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec![&[1][..], &[4][..], &[4, 1][..], &[1, 4][..]]);
    assert_close(search.records[0].total_path_length, 2.0);
    assert_close(search.records[1].total_path_length, 2.828_427_076);
    assert_close(search.records[2].total_path_length, 3.414_213_657);
    assert_close(search.records[3].total_path_length, 3.414_213_657);
    assert_eq!(search.max_heap_size, 2);
    assert_eq!(search.max_path_atoms_reached, 2);
    assert!(search.complete);
    Ok(())
}

#[test]
fn pathfinder_search_honors_output_limit() -> Result<(), PathError> {
    let preparation = simple_pathfinder_preparation()?;
    let fbetac = Array3::zeros((1, 1, 1));

    let search = pathfinder_search(PathfinderSearchInput {
        preparation: &preparation,
        max_path_length: 4.1,
        max_path_atoms: 2,
        max_output_paths: 1,
        heap_cutoff: 0.0,
        output_cutoff: 0.0,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &[],
        critical_mean_free_paths: &[],
        current_normalization: -1.0,
    })?;

    assert_eq!(search.records.len(), 1);
    assert_eq!(search.records[0].path_indices, vec![1]);
    assert!(!search.complete);
    Ok(())
}

#[test]
fn pathfinder_search_rejects_invalid_limits() -> Result<(), PathError> {
    let preparation = simple_pathfinder_preparation()?;
    let fbetac = Array3::zeros((1, 1, 1));

    assert!(matches!(
        pathfinder_search(PathfinderSearchInput {
            preparation: &preparation,
            max_path_length: 4.1,
            max_path_atoms: 0,
            max_output_paths: 8,
            heap_cutoff: 0.0,
            output_cutoff: 0.0,
            fbeta_critical: fbetac.view(),
            critical_wave_numbers: &[],
            critical_mean_free_paths: &[],
            current_normalization: -1.0,
        }),
        Err(PathError::InvalidPathfinderSearchLimit {
            quantity: "max_path_atoms",
            value: 0,
        })
    ));
    Ok(())
}

#[test]
fn pathfinder_reduction_composes_search_and_pathsd() -> Result<(), PathError> {
    let atom_positions = simple_pathfinder_positions();
    let fbeta = reference_fbeta_output_table();
    let fbetac = reference_fbeta_table();
    let wave_numbers = [1.2, 2.0, 3.25, 4.5, 6.0];
    let mean_free_paths = [6.0, 7.5, 9.0, 11.0, 14.0];
    let critical_wave_numbers = [2.0, 3.5, 5.0];
    let critical_mean_free_paths = [7.5, 10.0, 12.0];

    let output = pathfinder_reduction(PathfinderReductionInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &[0, 1, 1, 2, 2],
        first_bounce_degeneracies: &[0, 2, 0, 3, 1],
        fms_radius: 0.5,
        max_path_length: 4.1,
        max_path_atoms: 2,
        max_output_paths: 8,
        heap_cutoff: 0.0,
        output_cutoff: 0.0,
        search_normalization: -1.0,
        fbeta: fbeta.view(),
        wave_numbers: &wave_numbers,
        mean_free_paths: &mean_free_paths,
        start_energy_index: 1,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &critical_wave_numbers,
        critical_mean_free_paths: &critical_mean_free_paths,
        reduction_normalization: -1.0,
        criterion_percent: 0.0,
        retention_reference: None,
        polarization: 0,
        spin: 0,
        electric_vector: [0.0, 0.0, 1.0],
        incident_vector: [0.0, 0.0, 0.0],
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;

    assert_eq!(output.search.records.len(), 4);
    assert_eq!(output.search.records[0].path_indices, vec![1]);
    assert_eq!(output.reduction.ranges.len(), 3);
    assert_eq!(output.reduction.retained_unique_count, 3);
    assert_eq!(output.reduction.retained_total_degeneracy, 6);
    assert_eq!(output.reduction.ranges[2].range.groups[0].degeneracy, 3);
    assert!(output.reduction.retention_reference.is_some());
    Ok(())
}

fn simple_pathfinder_preparation() -> Result<PathfinderPreparation, PathError> {
    let atom_positions = simple_pathfinder_positions();
    pathfinder_preparation(PathfinderPreparationInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &[0, 1, 1, 2, 2],
        first_bounce_degeneracies: &[0, 2, 0, 3, 1],
        fms_radius: 0.5,
    })
}

fn simple_pathfinder_positions() -> ndarray::Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 0.0],
    ])
}
