use super::{support::*, *};

#[test]
fn dmdw_path_descriptor_expands_single_atom_feff_branches() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let all_atoms = DmdwPathDescriptor {
        selectors: vec![0],
        max_effective_length: 0.0,
    };
    let selected_atom = DmdwPathDescriptor {
        selectors: vec![2],
        max_effective_length: 0.0,
    };

    assert_eq!(
        dmdw_expand_path_descriptor(positions.view(), &all_atoms)?,
        vec![
            DmdwExpandedPath {
                atoms: vec![0],
                effective_length: 0.0,
            },
            DmdwExpandedPath {
                atoms: vec![1],
                effective_length: 0.0,
            },
            DmdwExpandedPath {
                atoms: vec![2],
                effective_length: 0.0,
            },
        ]
    );
    assert_eq!(
        dmdw_expand_path_descriptor(positions.view(), &selected_atom)?,
        vec![DmdwExpandedPath {
            atoms: vec![1],
            effective_length: 0.0,
        }]
    );
    Ok(())
}

#[test]
fn dmdw_path_descriptor_expands_multi_atom_feff_order_and_pruning() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let pairs = DmdwPathDescriptor {
        selectors: vec![0, 0],
        max_effective_length: 2.1,
    };

    let expanded = dmdw_expand_path_descriptor(positions.view(), &pairs)?;
    let expanded_atoms = expanded
        .iter()
        .map(|path| path.atoms.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        expanded_atoms,
        vec![vec![0, 1], vec![0, 2], vec![1, 0], vec![2, 0]]
    );
    for path in &expanded {
        assert_dmdw_close(path.effective_length, 2.0);
    }

    let triple = DmdwPathDescriptor {
        selectors: vec![1, 0, 3],
        max_effective_length: 3.5,
    };
    let expanded = dmdw_expand_path_descriptor(positions.view(), &triple)?;
    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].atoms, vec![0, 1, 2]);
    assert_dmdw_close(
        expanded[0].effective_length,
        0.5 * (2.0 + 8.0_f64.sqrt() + 2.0),
    );
    Ok(())
}

#[test]
fn dmdw_path_descriptor_rejects_invalid_inputs() {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let bad_shape = ndarray::Array2::<Real>::zeros((3, 2));

    assert!(matches!(
        dmdw_expand_path_descriptor(
            bad_shape.view(),
            &DmdwPathDescriptor {
                selectors: vec![0, 0],
                max_effective_length: 1.0,
            }
        ),
        Err(DebyeError::InvalidDmdwAtomShape { .. })
    ));
    assert!(matches!(
        dmdw_expand_path_descriptor(
            positions.view(),
            &DmdwPathDescriptor {
                selectors: Vec::new(),
                max_effective_length: 1.0,
            }
        ),
        Err(DebyeError::EmptyDmdwPath)
    ));
    assert!(matches!(
        dmdw_expand_path_descriptor(
            positions.view(),
            &DmdwPathDescriptor {
                selectors: vec![-1],
                max_effective_length: 1.0,
            }
        ),
        Err(DebyeError::InvalidDmdwPathSelector { selector: -1, .. })
    ));
    assert!(matches!(
        dmdw_expand_path_descriptor(
            positions.view(),
            &DmdwPathDescriptor {
                selectors: vec![4],
                max_effective_length: 1.0,
            }
        ),
        Err(DebyeError::InvalidDmdwPathSelector { selector: 4, .. })
    ));
    assert!(matches!(
        dmdw_expand_path_descriptor(
            positions.view(),
            &DmdwPathDescriptor {
                selectors: vec![1],
                max_effective_length: -1.0,
            }
        ),
        Err(DebyeError::Negative {
            name: "DMDW path descriptor maximum effective length",
            ..
        })
    ));
}

#[test]
fn dmdw_path_motion_matches_feff_two_atom_path() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let masses = ndarray::arr1(&[10.0, 20.0]);
    let motion = dmdw_path_motion(positions.view(), masses.view(), &[0, 1])?;

    assert_dmdw_close(motion.inverse_reduced_mass, 0.15);
    assert_dmdw_close(motion.reduced_mass, 6.666_666_666_666_667);
    assert_vector_close(
        &motion.initial_vector,
        &[
            -0.816_496_580_927_726,
            0.577_350_269_189_625_8,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    );
    assert_dmdw_close(
        motion
            .initial_vector
            .iter()
            .map(|value| value * value)
            .sum(),
        1.0,
    );
    Ok(())
}

#[test]
fn dmdw_path_motion_matches_feff_bent_three_atom_path() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let masses = ndarray::arr1(&[10.0, 20.0, 30.0]);
    let motion = dmdw_path_motion(positions.view(), masses.view(), &[0, 1, 2])?;

    assert_dmdw_close(motion.inverse_reduced_mass, 0.121_129_449_216_106_15);
    assert_dmdw_close(motion.reduced_mass, 8.255_630_703_115_866);
    assert_vector_close(
        &motion.initial_vector,
        &[
            -0.454_302_506_682_383,
            0.548_391_636_526_351_4,
            -0.185_468_221_706_530_54,
            -0.454_302_506_682_383,
            -0.227_151_253_341_191_5,
            0.447_759_896_233_126_1,
            0.0,
            0.0,
            0.0,
        ],
    );
    assert_dmdw_close(
        motion
            .initial_vector
            .iter()
            .map(|value| value * value)
            .sum(),
        1.0,
    );
    Ok(())
}

#[test]
fn dmdw_path_motion_matches_feff_single_atom_mass_branch() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
    let masses = ndarray::arr1(&[63.546]);
    let motion = dmdw_path_motion(positions.view(), masses.view(), &[0])?;

    assert_dmdw_close(motion.inverse_reduced_mass, 1.0 / 63.546);
    assert_dmdw_close(motion.reduced_mass, 63.546);
    assert_vector_close(&motion.initial_vector, &[0.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn dmdw_path_motion_rejects_invalid_inputs() {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let masses = ndarray::arr1(&[10.0, 20.0]);
    assert!(matches!(
        dmdw_path_motion(positions.view(), masses.view(), &[]),
        Err(DebyeError::EmptyDmdwPath)
    ));
    assert!(matches!(
        dmdw_path_motion(positions.view(), masses.view(), &[0, 2]),
        Err(DebyeError::InvalidDmdwPathAtomIndex { index: 2, .. })
    ));
    assert!(matches!(
        dmdw_path_motion(positions.view(), masses.view(), &[0, 1]),
        Err(DebyeError::ZeroLengthDmdwAtomPair {
            first: 0,
            second: 1
        })
    ));

    let bad_masses = ndarray::arr1(&[10.0]);
    assert!(matches!(
        dmdw_path_motion(positions.view(), bad_masses.view(), &[0]),
        Err(DebyeError::InvalidDmdwMassCount { .. })
    ));
    let bad_shape = ndarray::Array2::<Real>::zeros((2, 2));
    assert!(matches!(
        dmdw_path_motion(bad_shape.view(), masses.view(), &[0]),
        Err(DebyeError::InvalidDmdwAtomShape { .. })
    ));
}

#[test]
fn dmdw_ir_dipole_seed_matches_feff_type4_branch() -> Result<(), DebyeError> {
    let masses = ndarray::arr1(&[4.0, 9.0]);
    let dipoles = ndarray::arr3(&[
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
    ]);

    let seed = dmdw_ir_dipole_seed_vector(masses.view(), dipoles.view())?;

    assert_vector_close(
        &seed,
        &[
            0.007_160_718_421_688_271,
            0.324_917_598_384_105_3,
            0.044_754_490_135_551_696,
            0.526_312_803_994_088,
            0.114_571_494_747_012_34,
            0.776_042_858_950_466_4,
        ],
    );
    Ok(())
}

#[test]
fn dmdw_ir_dipole_seed_rejects_invalid_inputs() {
    let masses = ndarray::arr1(&[4.0, 9.0]);
    let bad_shape = ndarray::Array3::<Real>::zeros((2, 3, 2));
    assert!(matches!(
        dmdw_ir_dipole_seed_vector(masses.view(), bad_shape.view()),
        Err(DebyeError::InvalidDmdwDipoleDerivativeShape { .. })
    ));

    let zero_dipoles = ndarray::Array3::<Real>::zeros((2, 3, 3));
    assert!(matches!(
        dmdw_ir_dipole_seed_vector(masses.view(), zero_dipoles.view()),
        Err(DebyeError::ZeroDmdwSeedNorm)
    ));
}
