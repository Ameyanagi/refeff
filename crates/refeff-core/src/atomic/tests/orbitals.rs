#![allow(clippy::excessive_precision)]

use super::*;

#[allow(clippy::excessive_precision)]
#[test]
fn atom_orbital_initialization_matches_feff_inmuat_reference() -> Result<(), AtomMathError> {
    let open_shell = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
        atomic_number: 4,
        ionicity: 0.0,
        principal_quantum_numbers: &[2, 3, 1],
        kappas: &[1, 1, -1],
        occupations: &[0.4, 1.6, 2.0],
    })?;

    assert_eq!(open_shell.orbital_count, 3);
    assert_eq!(open_shell.self_consistent_count, 3);
    assert_eq!(open_shell.lagrange_pair_count, 1);
    assert_eq!(open_shell.radial_count, 251);
    assert_eq!(open_shell.development_order, 10);
    assert_eq!(open_shell.attempt_count, 50);
    assert_eq!(open_shell.nucleus_index, 11);
    assert_close_with(
        open_shell.wavefunction_precision,
        1.000_000_000_000_000_08e-5,
        1.0e-20,
    );
    assert_close_with(
        open_shell.energy_precision,
        5.000_000_000_000_000_41e-6,
        1.0e-20,
    );
    assert_close(open_shell.precision_ratios[0], 100.0);
    assert_close(open_shell.precision_ratios[1], 10.0);
    assert_close_with(open_shell.primary_matching_precision, 1.0e-7, 1.0e-20);
    assert_close_with(open_shell.secondary_matching_precision, 1.0e-6, 1.0e-20);
    assert_eq!(open_shell.shell_markers.to_vec(), vec![1, 1, -1]);
    assert_eq!(open_shell.active_lengths.to_vec(), vec![251, 251, 251]);
    assert_close_with(open_shell.convergence_acceleration[0], 1.0, 1.0e-16);
    assert_close_with(
        open_shell.convergence_acceleration[1],
        3.000_000_119_209_289_55e-1,
        1.0e-16,
    );
    assert_close_with(
        open_shell.convergence_acceleration[2],
        3.000_000_119_209_289_55e-1,
        1.0e-16,
    );
    assert!(
        open_shell
            .orbital_energies
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        open_shell
            .wavefunction_errors
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(open_shell.energy_errors.iter().all(|&value| value == 0.0));
    assert_eq!(open_shell.lagrange_parameters.len(), 820);
    assert!(
        open_shell
            .lagrange_parameters
            .iter()
            .all(|&value| value == 0.0)
    );

    let closed_shell = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
        atomic_number: 10,
        ionicity: 0.0,
        principal_quantum_numbers: &[1, 2, 2, 2],
        kappas: &[-1, -1, 1, -2],
        occupations: &[2.0, 2.0, 2.0, 4.0],
    })?;
    assert_eq!(closed_shell.orbital_count, 4);
    assert_eq!(closed_shell.self_consistent_count, 4);
    assert_eq!(closed_shell.lagrange_pair_count, 0);
    assert_eq!(closed_shell.shell_markers.to_vec(), vec![-1, -1, -1, -1]);
    assert_eq!(
        closed_shell.active_lengths.to_vec(),
        vec![251, 251, 251, 251]
    );
    for value in closed_shell.convergence_acceleration {
        assert_close_with(value, 3.000_000_119_209_289_55e-1, 1.0e-16);
    }
    Ok(())
}
