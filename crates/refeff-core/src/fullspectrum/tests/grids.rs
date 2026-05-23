use super::{support::*, *};

#[test]
fn linear_energy_grid_matches_active_feff_egrid_lin_branch() -> Result<(), FullSpectrumError> {
    let grid = full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
        point_count: 5,
        min_energy: 0.1,
        max_energy: 1.1,
    })?;

    assert_close(grid[0], 0.1, 0.0);
    assert_close(grid[1], 0.35, 1.0e-15);
    assert_close(grid[2], 0.6, 1.0e-15);
    assert_close(grid[3], 0.85, 1.0e-15);
    assert_close(grid[4], 1.1, 1.0e-15);
    Ok(())
}

#[test]
fn linear_energy_grid_applies_feff_positive_floor() -> Result<(), FullSpectrumError> {
    let grid = full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
        point_count: 3,
        min_energy: -1.0,
        max_energy: 0.5,
    })?;

    assert_close(grid[0], super::FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY, 0.0);
    assert_close(
        grid[1],
        (super::FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY + 0.5) / 2.0,
        1.0e-15,
    );
    assert_close(grid[2], 0.5, 1.0e-15);
    Ok(())
}

#[test]
fn linear_energy_grid_rejects_invalid_inputs() {
    assert!(matches!(
        full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
            point_count: 1,
            min_energy: 0.1,
            max_energy: 1.1,
        }),
        Err(FullSpectrumError::TooFewRows {
            name: "linear_grid",
            len: 1
        })
    ));
    assert!(matches!(
        full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
            point_count: 2,
            min_energy: f64::NAN,
            max_energy: 1.1,
        }),
        Err(FullSpectrumError::NonFiniteInput {
            name: "min_energy",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
            point_count: 2,
            min_energy: 1.1,
            max_energy: 1.1,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "linear_grid",
            ..
        })
    ));
}

#[test]
fn default_energy_grid_prefers_fine_structure_edges_like_feff_rdop() -> Result<(), FullSpectrumError>
{
    let grid = full_spectrum_default_energy_grid(&[
        FullSpectrumDefaultGridEdge {
            atomic_number: 8,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 1,
            fine_structure: true,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 4,
            fine_structure: false,
        },
    ])?;

    assert!(grid.used_fine_structure_edges);
    assert_close(grid.min_energy, 328.134_580_448_300_37, 1.0e-12);
    assert_close(grid.max_energy, 366.721_354_901_180_35, 1.0e-12);
    assert_eq!(grid.point_count, 2100);
    Ok(())
}

#[test]
fn default_energy_grid_falls_back_to_all_edges_like_feff_rdop() -> Result<(), FullSpectrumError> {
    let grid = full_spectrum_default_energy_grid(&[
        FullSpectrumDefaultGridEdge {
            atomic_number: 8,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 4,
            fine_structure: false,
        },
    ])?;

    assert!(!grid.used_fine_structure_edges);
    assert_close(grid.min_energy, 18.121_084_049_374_577, 1.0e-12);
    assert_close(grid.max_energy, 366.721_354_901_180_35, 1.0e-12);
    assert_eq!(grid.point_count, 18_971);
    Ok(())
}

#[test]
fn default_energy_grid_rejects_invalid_or_missing_edges() {
    assert!(matches!(
        full_spectrum_default_energy_grid(&[]),
        Err(FullSpectrumError::EmptyTable {
            name: "rdop_edge_grid"
        })
    ));
    assert!(matches!(
        full_spectrum_default_energy_grid(&[FullSpectrumDefaultGridEdge {
            atomic_number: 6,
            hole_index: 20,
            fine_structure: true,
        }]),
        Err(FullSpectrumError::MissingElamEdge {
            atomic_number: 6,
            hole_index: 20,
        })
    ));
    assert!(matches!(
        full_spectrum_default_energy_grid(&[FullSpectrumDefaultGridEdge {
            atomic_number: 101,
            hole_index: 1,
            fine_structure: true,
        }]),
        Err(FullSpectrumError::ElamEdgeTable {
            component: 0,
            source: ElamError::AtomicNumberOutOfRange { z: 101, .. },
        })
    ));
}

#[test]
fn edge_energy_grid_matches_feff_egrid_edge_restarts() -> Result<(), FullSpectrumError> {
    let edges = array![0.4, 0.8];

    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        min_energy: 0.0,
        max_energy: 1.0,
        edge_energies: edges.view(),
        wave_number_step: 0.2,
        max_points: 20,
    })?;

    assert!(!grid.clipped);
    assert_eq!(grid.point_count(), 15);
    assert_close(grid.energy[0], FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY, 0.0);
    assert_close(grid.energy[4], 0.326_895_256_108_847_07, 1.0e-14);
    assert_close(grid.energy[5], 0.4, 1.0e-14);
    assert_close(grid.energy[10], 0.8, 1.0e-14);
    assert_close(grid.energy[14], 1.0, 1.0e-14);
    Ok(())
}

#[test]
fn edge_energy_grid_matches_feff_egrid_without_edges() -> Result<(), FullSpectrumError> {
    let edges = Array1::<Real>::zeros(0);

    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        min_energy: 0.1,
        max_energy: 1.0,
        edge_energies: edges.view(),
        wave_number_step: 0.2,
        max_points: 20,
    })?;

    assert!(!grid.clipped);
    assert_eq!(grid.point_count(), 6);
    assert_close(grid.energy[0], 0.1, 0.0);
    assert_close(grid.energy[1], 0.209_442_719_099_991_6, 1.0e-14);
    assert_close(grid.energy[4], 0.777_770_876_399_966_3, 1.0e-14);
    assert_close(grid.energy[5], 1.0, 1.0e-14);
    Ok(())
}

#[test]
fn edge_energy_grid_reports_feff_capacity_clipping() -> Result<(), FullSpectrumError> {
    let edges = Array1::<Real>::zeros(0);

    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        min_energy: 0.1,
        max_energy: 10.0,
        edge_energies: edges.view(),
        wave_number_step: 0.2,
        max_points: 5,
    })?;

    assert!(grid.clipped);
    assert_eq!(grid.point_count(), 5);
    assert_close(grid.energy[4], 0.777_770_876_399_966_3, 1.0e-14);
    Ok(())
}

#[test]
fn elam_edge_energy_adapter_matches_feff_preved_nexted_scan() -> Result<(), FullSpectrumError> {
    let edges = full_spectrum_elam_edge_energies(&[29, 8, 79])?;

    assert_eq!(edges.len(), 30);
    assert!(
        edges
            .iter()
            .zip(edges.iter().skip(1))
            .all(|(left, right)| left < right)
    );

    let previous = edges
        .iter()
        .copied()
        .filter(|&energy| energy < 35.0)
        .fold(0.0, Real::max);
    let next = edges
        .iter()
        .copied()
        .filter(|&energy| energy > 35.0)
        .fold(1.0e8, Real::min);

    assert_close(previous, 3.499_636_651_471_202e1, 1.0e-14);
    assert_close(next, 4.030_296_538_890_82e1, 1.0e-14);
    Ok(())
}

#[test]
fn elam_edge_energy_adapter_rejects_out_of_range_components() {
    assert!(matches!(
        full_spectrum_elam_edge_energies(&[29, 101]),
        Err(FullSpectrumError::ElamEdgeTable {
            component: 1,
            source: ElamError::AtomicNumberOutOfRange { z: 101, .. },
        })
    ));
}

#[test]
fn edge_energy_grid_rejects_invalid_inputs() {
    let edges = array![0.4];

    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 0.1,
            max_energy: 1.0,
            edge_energies: edges.view(),
            wave_number_step: 0.2,
            max_points: 1,
        }),
        Err(FullSpectrumError::TooFewRows {
            name: "edge_grid",
            len: 1
        })
    ));
    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 0.1,
            max_energy: 1.0,
            edge_energies: edges.view(),
            wave_number_step: 0.0,
            max_points: 2,
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "wave_number_step",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 0.1,
            max_energy: 1.0,
            edge_energies: array![f64::NAN].view(),
            wave_number_step: 0.2,
            max_points: 2,
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "edge_energies",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 1.0,
            max_energy: 1.0,
            edge_energies: edges.view(),
            wave_number_step: 0.2,
            max_points: 2,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "edge_grid",
            ..
        })
    ));
}
