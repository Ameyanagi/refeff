use super::*;

fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "{actual} != {expected}"
    );
}

#[test]
fn elam_edge_lookup_matches_feff_getedg_reference() -> Result<(), ElamError> {
    assert_close(
        elam_edge_energy_hartree(29, 1)?.unwrap_or(-1.0),
        3.299_720_455_356_278e2,
    );
    assert_eq!(elam_edge_energy_ev(6, 20)?, None);
    assert_eq!(elam_edge_energy_ev(101, 1)?, None);
    Ok(())
}

#[test]
fn elam_neighbor_edges_match_feff_preved_nexted_reference() -> Result<(), ElamError> {
    let components = [29, 8, 79];
    assert_close(
        previous_elam_edge_hartree(35.0, &components)?,
        3.499_636_651_471_202e1,
    );
    assert_close(
        next_elam_edge_hartree(35.0, &components)?,
        4.030_296_538_890_82e1,
    );
    assert_eq!(previous_elam_edge_hartree(0.1, &components)?, 0.0);
    assert_eq!(
        next_elam_edge_hartree(4000.0, &components)?,
        ELAM_NEXT_EDGE_SENTINEL_HARTREE
    );
    Ok(())
}

#[test]
fn elam_component_edge_list_matches_feff_table_row() -> Result<(), ElamError> {
    let copper_edges = elam_component_edge_energies_hartree(29)?;

    assert_eq!(copper_edges.len(), 9);
    assert_eq!(copper_edges[0].hole_index, 1);
    assert_close(copper_edges[0].energy_hartree, 3.299_720_455_356_278e2);
    assert_eq!(copper_edges[8].hole_index, 9);
    assert_close(copper_edges[8].energy_hartree, 1.837_465_450_137_141e-1);
    assert_eq!(elam_component_edge_energies_hartree(99)?.len(), 0);
    Ok(())
}

#[test]
fn elam_edge_helpers_reject_invalid_inputs() {
    assert_eq!(
        elam_edge_energy_ev(0, 1),
        Err(ElamError::InvalidAtomicNumber { z: 0 })
    );
    assert_eq!(
        elam_edge_energy_ev(29, 0),
        Err(ElamError::InvalidHole {
            ihole: 0,
            max: ELAM_EDGE_HOLE_COUNT,
        })
    );
    assert_eq!(
        previous_elam_edge_hartree(1.0, &[101]),
        Err(ElamError::AtomicNumberOutOfRange {
            z: 101,
            max: ELAM_EDGE_ATOMIC_NUMBER_MAX,
        })
    );
    assert!(matches!(
        next_elam_edge_hartree(Real::NAN, &[29]),
        Err(ElamError::NonFiniteEnergy {
            name: "current_energy",
            ..
        })
    ));
}
