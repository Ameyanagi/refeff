use super::*;

#[test]
fn morse_einstein_cumulants_match_feff_reference() -> Result<(), DebyeError> {
    let first = morse_einstein_cumulants(0.003, 300.0, 1.0e-5, 400.0)?;
    assert_close(first.first, 1.190_648_842_682_321_3e-8);
    assert_close(first.third, 5.526_344_214_607_83e-11);
    assert_close(first.scaled_thermal_expansion, 5.291_772_49e-6);

    let second = morse_einstein_cumulants(0.0075, 800.0, 2.5e-5, 250.0)?;
    assert_close(second.first, 7.441_554_786_684_262e-8);
    assert_close(second.third, 1.098_357_016_560_439_2e-9);
    assert_close(second.scaled_thermal_expansion, 1.322_943_122_5e-5);

    let negative_alpha = morse_einstein_cumulants(0.0012, 120.0, -7.0e-6, 350.0)?;
    assert_close(negative_alpha.first, -3.333_816_545_419_16e-9);
    assert_close(negative_alpha.third, -3.706_146_208_663_239e-12);
    assert_close(negative_alpha.scaled_thermal_expansion, -3.704_240_743e-6);
    Ok(())
}

#[test]
fn morse_einstein_cumulants_reject_invalid_inputs() {
    assert!(matches!(
        morse_einstein_cumulants(0.0, 300.0, 1.0e-5, 400.0),
        Err(DebyeError::NonPositive { name: "sig2", .. })
    ));
    assert!(matches!(
        morse_einstein_cumulants(0.003, Real::NAN, 1.0e-5, 400.0),
        Err(DebyeError::NonFinite { name: "tk", .. })
    ));
    assert!(matches!(
        morse_einstein_cumulants(0.003, 300.0, Real::INFINITY, 400.0),
        Err(DebyeError::NonFinite { name: "alphat", .. })
    ));
    assert!(matches!(
        morse_einstein_cumulants(0.003, 300.0, 1.0e-5, -1.0),
        Err(DebyeError::NonPositive { name: "thetae", .. })
    ));
}

#[test]
fn thermal_expansion_cumulants_match_feff_reference() -> Result<(), DebyeError> {
    let copper = thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, 2.55)?;
    assert_relative_close(copper.first, -3.563_418_839_026_406e17);
    assert_relative_close(copper.third, -2.138_051_303_415_843_5e15);

    let copper_oxygen = thermal_expansion_cumulants(29, 8, 0.0042, 1.8e-5, 650.0, 1.91)?;
    assert_relative_close(copper_oxygen.first, -7.144_230_125_822_932e17);
    assert_relative_close(copper_oxygen.third, -6.001_153_305_691_263e15);

    let carbon_hydrogen = thermal_expansion_cumulants(6, 1, 0.0015, -6.0e-6, 300.0, 1.09)?;
    assert_relative_close(carbon_hydrogen.first, 7.521_958_969_413_031e14);
    assert_relative_close(carbon_hydrogen.third, 2.256_587_690_823_909e12);
    Ok(())
}

#[test]
fn thermal_expansion_cumulants_reject_invalid_inputs() {
    assert!(matches!(
        thermal_expansion_cumulants(0, 29, 0.003, 1.0e-5, 400.0, 2.55),
        Err(DebyeError::InvalidAtomicNumber { z: 0 })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 140, 0.003, 1.0e-5, 400.0, 2.55),
        Err(DebyeError::InvalidAtomicNumber { z: 140 })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 29, -0.003, 1.0e-5, 400.0, 2.55),
        Err(DebyeError::NonPositive { name: "sig2", .. })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 0.0, 2.55),
        Err(DebyeError::NonPositive { name: "thetad", .. })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, Real::NAN),
        Err(DebyeError::NonFinite { name: "reff", .. })
    ));
}

#[test]
fn debye_correlations_match_feff_reference() -> Result<(), DebyeError> {
    let zero = quantum_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(zero.value, 4.501_999_849_393_054e-3);
    let copper = quantum_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(copper.value, 1.691_640_883_386_128e-3);
    let copper_oxygen = quantum_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
    assert_close(copper_oxygen.value, 7.447_746_368_694_431e-4);

    let classical_zero = classical_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(classical_zero.value, 4.293_628_582_101_32e-3);
    let classical_copper = classical_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(classical_copper.value, 1.685_437_153_407_153e-3);
    let classical_copper_oxygen = classical_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
    assert_close(classical_copper_oxygen.value, 6.129_399_740_209_465e-4);
    Ok(())
}

#[test]
fn debye_correlations_reject_invalid_inputs() {
    assert!(matches!(
        quantum_debye_correlation(-1.0, 400.0, 300.0, 29, 29, 2.7),
        Err(DebyeError::Negative { name: "rij", .. })
    ));
    assert!(matches!(
        quantum_debye_correlation(1.0, 400.0, -1.0, 29, 29, 2.7),
        Err(DebyeError::Negative { name: "tk", .. })
    ));
    assert!(matches!(
        classical_debye_correlation(1.0, 400.0, 300.0, 29, 0, 2.7),
        Err(DebyeError::InvalidAtomicNumber { z: 0 })
    ));
}

#[test]
fn debye_waller_factors_match_feff_reference() -> Result<(), DebyeError> {
    let copper_path = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.55, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let copper_atomic_numbers = [29, 29, 29];
    assert_close(
        quantum_debye_waller_factor(
            300.0,
            400.0,
            2.7,
            copper_path.view(),
            &copper_atomic_numbers,
        )?,
        5.620_717_932_013_852e-3,
    );
    assert_close(
        classical_debye_waller_factor(
            300.0,
            400.0,
            2.7,
            copper_path.view(),
            &copper_atomic_numbers,
        )?,
        5.216_382_857_388_334e-3,
    );

    let triangle_path = ndarray::arr2(&[
        [0.0, 0.0, 0.0],
        [1.91, 0.25, 0.10],
        [2.60, 1.40, -0.20],
        [0.0, 0.0, 0.0],
    ]);
    let triangle_atomic_numbers = [29, 8, 29, 29];
    assert_close(
        quantum_debye_waller_factor(
            180.0,
            650.0,
            2.3,
            triangle_path.view(),
            &triangle_atomic_numbers,
        )?,
        2.623_124_881_997_499_5e-3,
    );
    assert_close(
        classical_debye_waller_factor(
            180.0,
            650.0,
            2.3,
            triangle_path.view(),
            &triangle_atomic_numbers,
        )?,
        1.796_449_763_322_294e-3,
    );
    Ok(())
}

#[test]
fn debye_waller_factors_reject_invalid_inputs() {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    assert!(matches!(
        quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29, 29]),
        Err(DebyeError::ZeroLengthPathLeg { leg: 1 })
    ));
    assert!(matches!(
        quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29]),
        Err(DebyeError::InvalidAtomicNumberCount { .. })
    ));
    let bad_shape = ndarray::Array2::<Real>::zeros((1, 3));
    assert!(matches!(
        quantum_debye_waller_factor(300.0, 400.0, 2.7, bad_shape.view(), &[29]),
        Err(DebyeError::InvalidPathShape { .. })
    ));
}

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

#[test]
fn dmdw_mass_weighted_dynamical_matrix_matches_feff_make_dm() -> Result<(), DebyeError> {
    let mut blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
    blocks[(0, 0, 0, 0)] = 2.0;
    blocks[(0, 1, 0, 1)] = 3.0;
    blocks[(1, 0, 1, 0)] = 6.0;
    blocks[(1, 1, 2, 2)] = 18.0;
    let masses = ndarray::arr1(&[4.0, 9.0]);
    let scale = 1_556.892_791_61 * 602.214_198_280;

    let result = dmdw_mass_weighted_dynamical_matrix(blocks.view(), masses.view())?;

    assert_eq!(result.matrix.shape(), &[6, 6]);
    assert_dmdw_close(result.matrix[(0, 0)], 0.5 * scale);
    assert_dmdw_close(result.matrix[(0, 3)], 0.5 * scale);
    assert_dmdw_close(result.matrix[(3, 0)], scale);
    assert_dmdw_close(result.matrix[(5, 5)], 2.0 * scale);
    assert_dmdw_close(result.average_value, scale / 9.0);
    assert_dmdw_close(result.average_asymmetry, scale / 36.0);
    assert_dmdw_close(result.asymmetry_percent_average, 25.0);
    assert_dmdw_close(result.asymmetry_percent_max, 100.0 / 72.0);
    assert!(result.passes_feff_symmetry_check());
    Ok(())
}

#[test]
fn dmdw_mass_weighted_dynamical_matrix_reports_feff_asymmetry_warning() -> Result<(), DebyeError> {
    let mut blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
    blocks[(0, 1, 0, 1)] = 6.0;
    let masses = ndarray::arr1(&[4.0, 9.0]);

    let result = dmdw_mass_weighted_dynamical_matrix(blocks.view(), masses.view())?;

    assert_dmdw_close(result.asymmetry_percent_average, 200.0);
    assert_dmdw_close(result.asymmetry_percent_max, 100.0 / 18.0);
    assert!(!result.passes_feff_symmetry_check());
    Ok(())
}

#[test]
fn dmdw_mass_weighted_dynamical_matrix_rejects_invalid_inputs() {
    let masses = ndarray::arr1(&[4.0, 9.0]);
    let bad_shape = ndarray::Array4::<Real>::zeros((1, 2, 3, 3));
    assert!(matches!(
        dmdw_mass_weighted_dynamical_matrix(bad_shape.view(), masses.view()),
        Err(DebyeError::InvalidDmdwBlockShape { .. })
    ));

    let empty_blocks = ndarray::Array4::<Real>::zeros((0, 0, 3, 3));
    let empty_masses = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_mass_weighted_dynamical_matrix(empty_blocks.view(), empty_masses.view()),
        Err(DebyeError::EmptyDmdwAtomTable)
    ));

    let bad_masses = ndarray::arr1(&[4.0, 0.0]);
    let blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
    assert!(matches!(
        dmdw_mass_weighted_dynamical_matrix(blocks.view(), bad_masses.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW atom mass",
            ..
        })
    ));
}

#[test]
fn dmdw_lanczos_coefficients_match_feff_recurrence() -> Result<(), DebyeError> {
    let matrix = ndarray::arr2(&[[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]);
    let seed = ndarray::arr1(&[1.0, 1.0, 1.0]);

    let coefficients = dmdw_lanczos_coefficients(matrix.view(), seed.view(), 1)?;

    assert_vector_close(
        &coefficients.alpha,
        &[4.666_666_666_666_667, 5.639_455_782_312_925],
    );
    assert_vector_close(
        &coefficients.beta,
        &[0.0, 3.299_831_645_537_221_6, 2.120_878_539_880_258],
    );
    assert_dmdw_close(coefficients.single_pole_frequency, 0.343_813_972_349_477_75);
    Ok(())
}

#[test]
fn dmdw_lanczos_coefficients_preserve_feff_column_product() -> Result<(), DebyeError> {
    let matrix = ndarray::arr2(&[[1.0, 10.0], [0.0, 2.0]]);
    let seed = ndarray::arr1(&[1.0, 0.0]);

    let coefficients = dmdw_lanczos_coefficients(matrix.view(), seed.view(), 1)?;

    assert_vector_close(&coefficients.alpha, &[1.0, 2.0]);
    assert_vector_close(&coefficients.beta, &[0.0, 10.0, 10.0]);
    Ok(())
}

#[test]
fn dmdw_lanczos_coefficients_reject_invalid_inputs() {
    let matrix = ndarray::arr2(&[[1.0, 0.0], [0.0, 2.0]]);
    let seed = ndarray::arr1(&[1.0, 0.0]);
    assert!(matches!(
        dmdw_lanczos_coefficients(matrix.view(), seed.view(), 0),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole count",
            ..
        })
    ));

    let bad_matrix = ndarray::Array2::<Real>::zeros((2, 3));
    assert!(matches!(
        dmdw_lanczos_coefficients(bad_matrix.view(), seed.view(), 1),
        Err(DebyeError::InvalidDmdwLanczosShape { .. })
    ));

    let eigen_seed = ndarray::arr1(&[1.0, 0.0]);
    assert!(matches!(
        dmdw_lanczos_coefficients(matrix.view(), eigen_seed.view(), 1),
        Err(DebyeError::DmdwLanczosBreakdown { iteration: 1 })
    ));
}

#[test]
fn dmdw_lanczos_polynomials_match_feff_recurrences() -> Result<(), DebyeError> {
    let alpha = ndarray::arr1(&[4.666_666_666_666_667, 5.639_455_782_312_925]);
    let beta = ndarray::arr1(&[0.0, 3.299_831_645_537_221_6]);

    assert_dmdw_close(
        dmdw_lanczos_s_polynomial(2, 7.0, alpha.view(), beta.view())?,
        -7.714_285_714_285_713_5,
    );
    assert_dmdw_close(
        dmdw_lanczos_r_polynomial(2, 7.0, alpha.view(), beta.view())?,
        1.360_544_217_687_074_6,
    );
    assert_dmdw_close(
        dmdw_lanczos_s_polynomial_derivative(2, 7.0, alpha.view(), beta.view())?,
        3.693_877_551_020_406_7,
    );
    assert_dmdw_close(
        dmdw_lanczos_s_polynomial(1, 7.0, alpha.view(), beta.view())?,
        2.333_333_333_333_333,
    );
    assert_dmdw_close(
        dmdw_lanczos_r_polynomial(1, 7.0, alpha.view(), beta.view())?,
        1.0,
    );
    assert_dmdw_close(
        dmdw_lanczos_s_polynomial_derivative(1, 7.0, alpha.view(), beta.view())?,
        1.0,
    );
    Ok(())
}

#[test]
fn dmdw_lanczos_polynomials_reject_invalid_inputs() {
    let alpha = ndarray::arr1(&[1.0]);
    let beta = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_lanczos_s_polynomial(0, 1.0, alpha.view(), beta.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos polynomial order",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_s_polynomial(2, 1.0, alpha.view(), beta.view()),
        Err(DebyeError::InvalidDmdwLanczosPolynomialShape { .. })
    ));
    assert!(matches!(
        dmdw_lanczos_s_polynomial(1, Real::NAN, alpha.view(), beta.view()),
        Err(DebyeError::NonFinite {
            name: "DMDW Lanczos polynomial x",
            ..
        })
    ));
}

#[test]
fn dmdw_lanczos_pole_spectrum_matches_feff_scan() -> Result<(), DebyeError> {
    let alpha = ndarray::arr1(&[16.0, 16.0]);
    let beta = ndarray::arr1(&[0.0, 8.0]);
    let spectrum =
        dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 32.0, 8192)?;

    assert!(spectrum.has_expected_pole_count());
    assert_vector_close(&spectrum.squared_angular_frequencies, &[8.0, 24.0]);
    assert_vector_close(
        &spectrum.angular_frequencies,
        &[8.0_f64.sqrt(), 24.0_f64.sqrt()],
    );
    assert_vector_close(
        &spectrum.frequencies,
        &[
            8.0_f64.sqrt() / (2.0 * std::f64::consts::PI),
            24.0_f64.sqrt() / (2.0 * std::f64::consts::PI),
        ],
    );
    assert_vector_close(&spectrum.weights, &[0.5, 0.5]);
    assert!(spectrum.imaginary_warnings.is_empty());
    Ok(())
}

#[test]
fn dmdw_lanczos_pole_spectrum_reports_imaginary_weight_warnings() -> Result<(), DebyeError> {
    let alpha = ndarray::arr1(&[-16.0, -16.0]);
    let beta = ndarray::arr1(&[0.0, 8.0]);
    let spectrum =
        dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 32.0, 8192)?;

    assert!(spectrum.has_expected_pole_count());
    assert_vector_close(&spectrum.squared_angular_frequencies, &[-24.0, -8.0]);
    assert_vector_close(
        &spectrum.angular_frequencies,
        &[-24.0_f64.sqrt(), -8.0_f64.sqrt()],
    );
    assert_vector_close(&spectrum.weights, &[0.5, 0.5]);
    assert_eq!(spectrum.imaginary_warnings.len(), 2);
    assert_eq!(
        spectrum.imaginary_warnings[0].severity,
        DmdwImaginaryPoleSeverity::LargeWeight
    );
    assert_eq!(spectrum.imaginary_warnings[0].pole_index, 0);
    assert_dmdw_close(spectrum.imaginary_warnings[0].weight, 0.5);
    Ok(())
}

#[test]
fn dmdw_lanczos_pole_spectrum_rejects_invalid_inputs() {
    let alpha = ndarray::arr1(&[1.0, 1.0]);
    let beta = ndarray::arr1(&[0.0, 0.0]);
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(0, alpha.view(), beta.view(), 2.0, 1),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos polynomial order",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(1, alpha.view(), beta.view(), 0.0, 1),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole search limit",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(1, alpha.view(), beta.view(), 2.0, 0),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole samples per pole",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 2.0, 2),
        Err(DebyeError::ZeroDmdwLanczosPoleDerivative { .. })
    ));
}

#[test]
fn dmdw_debye_waller_factors_from_poles_match_feff_accumulation() -> Result<(), DebyeError> {
    let temperatures = ndarray::arr1(&[300.0, 600.0]);
    let angular_frequencies = ndarray::arr1(&[2.0, 4.0, -3.0, 0.0]);
    let weights = ndarray::arr1(&[0.25, 0.75, 0.9, 10.0]);
    let factors = dmdw_debye_waller_factors_from_poles(
        temperatures.view(),
        5.0,
        angular_frequencies.view(),
        weights.view(),
    )?;

    assert_vector_close(&factors, &[5.459_186_287_610_058, 10.914_330_842_743_967]);
    Ok(())
}

#[test]
fn dmdw_debye_waller_factors_use_zero_temperature_coth_limit() -> Result<(), DebyeError> {
    let temperatures = ndarray::arr1(&[0.001]);
    let angular_frequencies = ndarray::arr1(&[2.0]);
    let weights = ndarray::arr1(&[1.0]);
    let factors = dmdw_debye_waller_factors_from_poles(
        temperatures.view(),
        5.0,
        angular_frequencies.view(),
        weights.view(),
    )?;

    assert_vector_close(&factors, &[0.317_544_517_206_879_8]);
    Ok(())
}

#[test]
fn dmdw_vibrational_free_energy_from_poles_matches_feff_accumulation() -> Result<(), DebyeError> {
    let temperatures = ndarray::arr1(&[300.0, 600.0]);
    let angular_frequencies = ndarray::arr1(&[2.0, 4.0, -3.0, 0.0]);
    let weights = ndarray::arr1(&[0.25, 0.75, 0.9, 10.0]);
    let free_energy = dmdw_vibrational_free_energy_from_poles(
        temperatures.view(),
        angular_frequencies.view(),
        weights.view(),
    )?;

    assert_vector_close(
        &free_energy,
        &[-6_129.431_830_672_452, -15_718.169_449_997_833],
    );
    Ok(())
}

#[test]
fn dmdw_einstein_and_moment_summaries_match_feff_print_formulas() -> Result<(), DebyeError> {
    let reduced_mass = 10.0;
    let summary = dmdw_single_pole_einstein_summary(3.5, reduced_mass)?;
    assert_dmdw_close(summary.frequency_thz, 3.5);
    assert_dmdw_close(summary.temperature_kelvin, 3.5 * DMDW_THZ_TO_KELVIN);
    assert_dmdw_close(
        summary.effective_force_constant_n_per_m,
        reduced_mass
            * (2.0 * std::f64::consts::PI * 3.5).powi(2)
            * DMDW_AMU_THZ2_TO_NEWTON_PER_METER,
    );

    let frequencies = ndarray::arr1(&[-1.0, 2.0, 4.0]);
    let weights = ndarray::arr1(&[0.2, 0.2, 0.6]);
    let moments =
        dmdw_moment_summaries_from_poles(reduced_mass, frequencies.view(), weights.view())?;

    assert_eq!(
        moments
            .iter()
            .map(|moment| moment.order)
            .collect::<Vec<_>>(),
        vec![-2, -1, 0, 1, 2]
    );
    assert_moment_summary(
        &moments[0],
        0.109_375,
        0.109_375_f64.powf(-0.5),
        reduced_mass,
    )?;
    assert_moment_summary(&moments[1], 0.312_5, 3.2, reduced_mass)?;
    assert_dmdw_close(moments[2].moment_thz_power_n, 1.0);
    assert_eq!(moments[2].frequency_thz, None);
    assert_eq!(moments[2].temperature_kelvin, None);
    assert_eq!(moments[2].effective_force_constant_n_per_m, None);
    assert_moment_summary(&moments[3], 3.5, 3.5, reduced_mass)?;
    assert_moment_summary(&moments[4], 13.0, 13.0_f64.sqrt(), reduced_mass)?;
    Ok(())
}

#[test]
fn dmdw_pole_thermal_helpers_reject_invalid_inputs() {
    let temperatures = ndarray::arr1(&[300.0]);
    let frequencies = ndarray::arr1(&[1.0, 2.0]);
    let weights = ndarray::arr1(&[1.0]);
    assert!(matches!(
        dmdw_debye_waller_factors_from_poles(
            temperatures.view(),
            1.0,
            frequencies.view(),
            weights.view()
        ),
        Err(DebyeError::InvalidDmdwPoleTableShape { .. })
    ));

    let empty_temperatures = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_vibrational_free_energy_from_poles(
            empty_temperatures.view(),
            weights.view(),
            weights.view()
        ),
        Err(DebyeError::EmptyDmdwTemperatureTable)
    ));

    let bad_temperatures = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_vibrational_free_energy_from_poles(
            bad_temperatures.view(),
            weights.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW temperature",
            ..
        })
    ));

    assert!(matches!(
        dmdw_debye_waller_factors_from_poles(
            temperatures.view(),
            0.0,
            weights.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW reduced mass",
            ..
        })
    ));
}

#[test]
fn dmdw_pole_summary_helpers_reject_invalid_inputs() {
    assert!(matches!(
        dmdw_single_pole_einstein_summary(0.0, 1.0),
        Err(DebyeError::NonPositive {
            name: "DMDW Einstein frequency",
            ..
        })
    ));

    let empty = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_moment_summaries_from_poles(1.0, empty.view(), empty.view()),
        Err(DebyeError::EmptyDmdwPoleTable)
    ));

    let imaginary_frequencies = ndarray::arr1(&[-1.0]);
    let weights = ndarray::arr1(&[1.0]);
    assert!(matches!(
        dmdw_moment_summaries_from_poles(1.0, imaginary_frequencies.view(), weights.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW positive pole weight normalization",
            ..
        })
    ));
}

#[test]
fn dmdw_phonon_coupling_matches_feff_phonon_coupling_formulas() -> Result<(), DebyeError> {
    let pds_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let phonon_dos = ndarray::arr1(&[10.0, 20.0, 30.0]);
    let a2f_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let eliashberg = ndarray::arr1(&[0.5, 1.0, 1.5]);

    let coupling = dmdw_phonon_coupling(
        pds_energy.view(),
        phonon_dos.view(),
        a2f_energy.view(),
        eliashberg.view(),
    )?;

    assert_eq!(coupling.point_count(), 3);
    assert_vector_close(&coupling.energy_hartree, &[0.001, 0.002, 0.004]);
    assert_vector_close(
        &coupling.energy_ev,
        &[
            0.001 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.002 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.004 * DMDW_COUPLING_ENERGY_HARTREE_EV,
        ],
    );
    assert_vector_close(&coupling.eliashberg, &[0.5, 1.0, 1.5]);
    assert_vector_close(&coupling.matrix_element, &[0.05, 0.05, 0.05]);
    assert_dmdw_close(
        coupling.normalization,
        (10.0 * 0.001 + 20.0 * 0.001 + 30.0 * 0.002) * DMDW_COUPLING_NORM_HARTREE_EV,
    );
    Ok(())
}

#[test]
fn dmdw_pole_weighted_a2f_matches_feff_diagnostic_formulas() -> Result<(), DebyeError> {
    let pds_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let phonon_dos = ndarray::arr1(&[10.0, 20.0, 30.0]);
    let a2f_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let eliashberg = ndarray::arr1(&[0.5, 1.0, 1.5]);
    let coupling = dmdw_phonon_coupling(
        pds_energy.view(),
        phonon_dos.view(),
        a2f_energy.view(),
        eliashberg.view(),
    )?;
    let spectrum = DmdwLanczosPoleSpectrum {
        expected_poles: 3,
        squared_angular_frequencies: ndarray::arr1(&[
            (5.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2),
            (10.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2),
            (20.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2),
        ]),
        angular_frequencies: ndarray::arr1(&[
            5.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI,
            10.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI,
            20.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI,
        ]),
        frequencies: ndarray::arr1(&[5.0, 10.0, 20.0]),
        weights: ndarray::arr1(&[0.2, 0.3, 0.5]),
        imaginary_warnings: Vec::new(),
    };

    let diagnostic = dmdw_pole_weighted_a2f(&spectrum, &coupling)?;
    let expected_pole_weights = [
        0.05 * 0.2 * coupling.normalization,
        0.05 * 0.3 * coupling.normalization,
        0.05 * 0.5 * coupling.normalization,
    ];
    let expected_energies = [
        5.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI * DMDW_A2F_POLE_ANGULAR_TO_EV,
        10.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI * DMDW_A2F_POLE_ANGULAR_TO_EV,
        20.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI * DMDW_A2F_POLE_ANGULAR_TO_EV,
    ];
    let expected_total = expected_pole_weights.iter().sum::<Real>();
    let expected_w0 = expected_energies
        .iter()
        .zip(expected_pole_weights.iter())
        .map(|(&energy, &weight)| energy * weight)
        .sum::<Real>()
        / expected_total;
    let expected_lambda = expected_energies
        .iter()
        .zip(expected_pole_weights.iter())
        .map(|(&energy, &weight)| 2.0 * weight / energy)
        .sum::<Real>();

    assert_vector_close(&diagnostic.lanczos_frequency_thz, &[5.0, 10.0, 20.0]);
    assert_vector_close(&diagnostic.lanczos_weight, &[0.2, 0.3, 0.5]);
    assert_dmdw_close(diagnostic.normalization, coupling.normalization);
    assert_vector_close(&diagnostic.pole_energy_ev, &expected_energies);
    assert_vector_close(&diagnostic.pole_weight, &expected_pole_weights);
    assert_dmdw_close(diagnostic.mass_enhancement, expected_lambda);
    assert_dmdw_close(diagnostic.characteristic_energy_ev, expected_w0);
    Ok(())
}

#[test]
fn dmdw_pole_weighted_a2f_rejects_unmatched_and_zero_weight_inputs() {
    let coupling = DmdwPhononCoupling {
        energy_hartree: ndarray::arr1(&[0.001]),
        energy_ev: ndarray::arr1(&[0.001 * DMDW_COUPLING_ENERGY_HARTREE_EV]),
        eliashberg: ndarray::arr1(&[0.5]),
        matrix_element: ndarray::arr1(&[0.05]),
        normalization: 1.0,
    };
    let unmatched = DmdwLanczosPoleSpectrum {
        expected_poles: 1,
        squared_angular_frequencies: ndarray::arr1(&[
            (100.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2)
        ]),
        angular_frequencies: ndarray::arr1(&[100.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI]),
        frequencies: ndarray::arr1(&[100.0]),
        weights: ndarray::arr1(&[1.0]),
        imaginary_warnings: Vec::new(),
    };
    assert!(matches!(
        dmdw_pole_weighted_a2f(&unmatched, &coupling),
        Err(DebyeError::UnmatchedDmdwA2fPole { pole_index: 0, .. })
    ));

    let imaginary = DmdwLanczosPoleSpectrum {
        expected_poles: 1,
        squared_angular_frequencies: ndarray::arr1(&[-1.0]),
        angular_frequencies: ndarray::arr1(&[-1.0]),
        frequencies: ndarray::arr1(&[-1.0 / (2.0 * std::f64::consts::PI)]),
        weights: ndarray::arr1(&[1.0]),
        imaginary_warnings: Vec::new(),
    };
    assert!(matches!(
        dmdw_pole_weighted_a2f(&imaginary, &coupling),
        Err(DebyeError::NonPositive {
            name: "DMDW a2f total pole weight",
            ..
        })
    ));
}

#[test]
fn dmdw_type2_pole_weighted_a2f_matches_feff_unit_seed_accumulation() -> Result<(), DebyeError> {
    let (force_blocks, masses) = sample_dmdw_type2_blocks();
    let coupling = sample_dmdw_type2_coupling();
    let groups = vec![DmdwType2AtomGroup {
        center_atom_indices: vec![0],
    }];

    let all_displacements =
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &groups, 0, 1, &coupling)?;
    let selected_y =
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &groups, 2, 1, &coupling)?;

    let scale = DMDW_DYNAMICAL_MATRIX_SCALE / masses[0];
    let expected_all_angular = (0..3)
        .map(|component| (force_blocks[(0, 0, component, component)] * scale).sqrt())
        .sum::<Real>()
        / 3.0;
    let expected_y_angular = (force_blocks[(0, 0, 1, 1)] * scale).sqrt();
    let expected_all_energy = expected_all_angular * DMDW_A2F_POLE_ANGULAR_TO_EV;
    let expected_y_energy = expected_y_angular * DMDW_A2F_POLE_ANGULAR_TO_EV;
    let expected_weight = coupling.matrix_element[0] * coupling.normalization;

    assert_vector_close(
        &all_displacements.lanczos_frequency_thz,
        &[expected_all_angular / DMDW_A2F_DIAGNOSTIC_TWO_PI],
    );
    assert_vector_close(&all_displacements.lanczos_weight, &[1.0]);
    assert_vector_close(&all_displacements.pole_energy_ev, &[expected_all_energy]);
    assert_vector_close(&all_displacements.pole_weight, &[expected_weight]);
    assert_dmdw_close(
        all_displacements.mass_enhancement,
        2.0 * expected_weight / expected_all_energy,
    );
    assert_dmdw_close(
        all_displacements.characteristic_energy_ev,
        expected_all_energy,
    );

    assert_vector_close(
        &selected_y.lanczos_frequency_thz,
        &[expected_y_angular / DMDW_A2F_DIAGNOSTIC_TWO_PI],
    );
    assert_vector_close(&selected_y.lanczos_weight, &[1.0]);
    assert_vector_close(&selected_y.pole_energy_ev, &[expected_y_energy]);
    assert_vector_close(&selected_y.pole_weight, &[expected_weight]);
    assert_dmdw_close(selected_y.characteristic_energy_ev, expected_y_energy);
    Ok(())
}

#[test]
fn dmdw_type2_pole_weighted_a2f_rejects_invalid_metadata() {
    let (force_blocks, masses) = sample_dmdw_type2_blocks();
    let coupling = sample_dmdw_type2_coupling();
    let groups = vec![DmdwType2AtomGroup {
        center_atom_indices: vec![masses.len()],
    }];

    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &[], 0, 1, &coupling),
        Err(DebyeError::EmptyDmdwType2UniqueAtomTable)
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(
            force_blocks.view(),
            masses.view(),
            &[DmdwType2AtomGroup {
                center_atom_indices: Vec::new()
            }],
            0,
            1,
            &coupling
        ),
        Err(DebyeError::EmptyDmdwType2CenterAtomGroup { group: 0 })
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &groups, 0, 1, &coupling),
        Err(DebyeError::InvalidDmdwType2CenterAtomIndex { group: 0, .. })
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(
            force_blocks.view(),
            masses.view(),
            &[DmdwType2AtomGroup {
                center_atom_indices: vec![0]
            }],
            4,
            1,
            &coupling
        ),
        Err(DebyeError::InvalidDmdwType2DisplacementOption { option: 4 })
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(
            force_blocks.view(),
            masses.view(),
            &[DmdwType2AtomGroup {
                center_atom_indices: vec![0]
            }],
            0,
            0,
            &coupling
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW type 2 Lanczos pole count",
            ..
        })
    ));
}

#[test]
fn dmdw_self_energy_matches_zero_energy_feff_identity() -> Result<(), DebyeError> {
    let temperature = 300.0;
    let pole_energy = ndarray::arr1(&[0.012, 0.024]);
    let pole_weight = ndarray::arr1(&[0.35, 0.65]);

    let self_energy = dmdw_self_energy_from_a2f_poles(
        temperature,
        Complex::new(0.0, 0.0),
        pole_energy.view(),
        pole_weight.view(),
    )?;
    let expected_imaginary = pole_energy
        .iter()
        .zip(pole_weight.iter())
        .map(|(&energy, &weight)| {
            let argument = energy / (DMDW_SELF_ENERGY_BOLTZMANN_EV_PER_K * temperature);
            -DMDW_SELF_ENERGY_TWO_PI * weight / argument.sinh()
        })
        .sum::<Real>();

    assert_complex_dmdw_close_tol(self_energy, Complex::new(0.0, expected_imaginary), 1.0e-10);
    Ok(())
}

#[test]
fn dmdw_self_energy_grid_matches_scalar_evaluation() -> Result<(), DebyeError> {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[3.0, 7.0]),
        lanczos_weight: ndarray::arr1(&[0.4, 0.6]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.010, 0.030]),
        pole_weight: ndarray::arr1(&[0.15, 0.25]),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.0225,
    };
    let energies = ndarray::arr1(&[-0.02, 0.0, 0.04]);

    let grid = dmdw_self_energy_grid_from_a2f_poles(450.0, energies.view(), &diagnostic)?;

    assert_eq!(grid.point_count(), energies.len());
    assert_vector_close(&grid.energy_ev, &[-0.02, 0.0, 0.04]);
    for (&energy, &actual) in energies.iter().zip(grid.self_energy.iter()) {
        let expected = dmdw_self_energy_from_a2f_poles(
            450.0,
            Complex::new(energy, 0.0),
            diagnostic.pole_energy_ev.view(),
            diagnostic.pole_weight.view(),
        )?;
        assert_complex_dmdw_close(actual, expected);
    }
    Ok(())
}

#[test]
fn dmdw_self_energy_rejects_invalid_inputs() {
    let energies = ndarray::arr1(&[0.01]);
    let weights = ndarray::arr1(&[0.2]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            0.0,
            Complex::new(0.0, 0.0),
            energies.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW self-energy temperature",
            ..
        })
    ));
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(Real::NAN, 0.0),
            energies.view(),
            weights.view()
        ),
        Err(DebyeError::NonFiniteComplex {
            name: "DMDW self-energy energy",
            ..
        })
    ));

    let short_weights = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(0.0, 0.0),
            energies.view(),
            short_weights.view()
        ),
        Err(DebyeError::InvalidDmdwSelfEnergyPoleTableShape { .. })
    ));

    let empty = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(300.0, Complex::new(0.0, 0.0), empty.view(), empty.view()),
        Err(DebyeError::EmptyDmdwPoleTable)
    ));

    let zero_energy = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(0.0, 0.0),
            zero_energy.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW self-energy pole energy",
            ..
        })
    ));

    let zero_weight = ndarray::arr1(&[0.0]);
    assert_eq!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(0.0, 0.0),
            zero_energy.view(),
            zero_weight.view()
        ),
        Ok(Complex::new(0.0, 0.0))
    );
}

#[test]
fn dmdw_self_energy_grid_rejects_empty_energy_grid() {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[1.0]),
        lanczos_weight: ndarray::arr1(&[1.0]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.01]),
        pole_weight: ndarray::arr1(&[0.2]),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.01,
    };
    let empty = ndarray::arr1(&[]);

    assert!(matches!(
        dmdw_self_energy_grid_from_a2f_poles(300.0, empty.view(), &diagnostic),
        Err(DebyeError::EmptyDmdwSelfEnergyGrid)
    ));
}

#[test]
fn dmdw_spectral_function_handles_zero_coupling_symmetry() -> Result<(), DebyeError> {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[3.0]),
        lanczos_weight: ndarray::arr1(&[1.0]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.020]),
        pole_weight: ndarray::arr1(&[0.0]),
        mass_enhancement: 0.0,
        characteristic_energy_ev: 0.020,
    };
    let energy = ndarray::arr1(&[-1.0, -0.5, 0.0, 0.5, 1.0]);

    let spectral = dmdw_spectral_function_from_a2f_poles(
        300.0,
        energy.view(),
        0.0,
        diagnostic.characteristic_energy_ev,
        &diagnostic,
        20.0,
        101,
    )?;

    assert_eq!(spectral.point_count(), energy.len());
    assert_close(spectral.gamma_w0, 0.005);
    assert!(spectral.normalization.is_finite());
    assert!(spectral.normalization > 0.0);
    for value in &spectral.spectral_function {
        assert!(value.re.is_finite());
        assert!(value.im.is_finite());
    }
    assert_complex_dmdw_close_tol(
        spectral.spectral_function[0],
        spectral.spectral_function[4].conj(),
        1.0e-10,
    );
    assert_complex_dmdw_close_tol(
        spectral.spectral_function[1],
        spectral.spectral_function[3].conj(),
        1.0e-10,
    );
    Ok(())
}

#[test]
fn dmdw_spectral_function_rejects_invalid_grids() {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[3.0]),
        lanczos_weight: ndarray::arr1(&[1.0]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.020]),
        pole_weight: ndarray::arr1(&[0.1]),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.020,
    };
    let one_point = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_spectral_function_from_a2f_poles(
            300.0,
            one_point.view(),
            0.0,
            diagnostic.characteristic_energy_ev,
            &diagnostic,
            20.0,
            101,
        ),
        Err(DebyeError::InvalidDmdwSpectralEnergyGrid { points: 1 })
    ));

    let nonuniform = ndarray::arr1(&[-1.0, 0.0, 0.25]);
    assert!(matches!(
        dmdw_spectral_function_from_a2f_poles(
            300.0,
            nonuniform.view(),
            0.0,
            diagnostic.characteristic_energy_ev,
            &diagnostic,
            20.0,
            101,
        ),
        Err(DebyeError::NonUniformDmdwSpectralEnergyGrid { .. })
    ));

    let energy = ndarray::arr1(&[-1.0, 0.0, 1.0]);
    assert!(matches!(
        dmdw_spectral_function_from_a2f_poles(
            300.0,
            energy.view(),
            0.0,
            diagnostic.characteristic_energy_ev,
            &diagnostic,
            20.0,
            100,
        ),
        Err(DebyeError::InvalidDmdwSpectralTimeGrid { points: 100 })
    ));
}

#[test]
fn dmdw_phonon_coupling_rejects_invalid_inputs() {
    let energy = ndarray::arr1(&[0.001, 0.002]);
    let short_energy = ndarray::arr1(&[0.001]);
    let phonon_dos = ndarray::arr1(&[10.0, 20.0]);
    let eliashberg = ndarray::arr1(&[0.5, 1.0]);

    assert!(matches!(
        dmdw_phonon_coupling(
            short_energy.view(),
            phonon_dos.view(),
            energy.view(),
            eliashberg.view()
        ),
        Err(DebyeError::InvalidDmdwCouplingTableShape { .. })
    ));

    let empty = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_phonon_coupling(empty.view(), empty.view(), empty.view(), empty.view()),
        Err(DebyeError::EmptyDmdwCouplingTable)
    ));

    let bad_dos = ndarray::arr1(&[10.0, 0.0]);
    assert!(matches!(
        dmdw_phonon_coupling(
            energy.view(),
            bad_dos.view(),
            energy.view(),
            eliashberg.view()
        ),
        Err(DebyeError::NonPositiveDmdwPhononDensity { row: 2, .. })
    ));

    let shifted_energy = ndarray::arr1(&[0.001, 0.002_1]);
    assert!(matches!(
        dmdw_phonon_coupling(
            energy.view(),
            phonon_dos.view(),
            shifted_energy.view(),
            eliashberg.view()
        ),
        Err(DebyeError::MismatchedDmdwCouplingEnergyGrid { row: 2, .. })
    ));

    let nonfinite = ndarray::arr1(&[0.5, Real::NAN]);
    assert!(matches!(
        dmdw_phonon_coupling(
            energy.view(),
            phonon_dos.view(),
            energy.view(),
            nonfinite.view()
        ),
        Err(DebyeError::NonFinite {
            name: "DMDW Eliashberg coupling",
            ..
        })
    ));
}

#[test]
fn dmdw_center_of_mass_and_inertia_match_feff_reference_formulas() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]);
    let masses = ndarray::arr1(&[2.0, 3.0, 5.0]);

    let center = dmdw_center_of_mass(positions.view(), masses.view())?;
    assert_slice_close(&center, &[0.6, 1.5, 0.0]);

    let centered = ndarray::arr2(&[[-0.6, -1.5, 0.0], [1.4, -1.5, 0.0], [-0.6, 1.5, 0.0]]);
    let tensor = dmdw_inertia_tensor(centered.view(), masses.view())?;
    assert_matrix_close(
        tensor.view(),
        &[[22.5, 9.0, 0.0], [9.0, 8.4, 0.0], [0.0, 0.0, 30.9]],
    );
    Ok(())
}

#[test]
fn dmdw_rigid_body_projection_modes_match_feff_make_trfd_formulas() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -3.0],
    ]);
    let masses = ndarray::arr1(&[1.0; 6]);
    let modes = dmdw_rigid_body_projection_modes(positions.view(), masses.view())?;

    assert_slice_close(&modes.center_of_mass, &[0.0, 0.0, 0.0]);
    assert_vector_close(&modes.moments_of_inertia, &[10.0, 20.0, 26.0]);
    assert_matrix_abs_close(
        modes.principal_axes.view(),
        &[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    );

    let projection = modes.projection_modes;
    assert_eq!(projection.shape(), &[18, 6]);
    for left in 0..6 {
        assert_dmdw_close(column_dot(projection.view(), left, left), 1.0);
        for right in (left + 1)..6 {
            assert_dmdw_close(column_dot(projection.view(), left, right), 0.0);
        }
    }

    let translation_scale = 1.0 / 6.0_f64.sqrt();
    for atom in 0..6 {
        assert_dmdw_close(projection[(atom, 0)], translation_scale);
        assert_dmdw_close(projection[(6 + atom, 1)], translation_scale);
        assert_dmdw_close(projection[(12 + atom, 2)], translation_scale);
    }

    let rotation_z = ndarray::arr1(&[
        0.0,
        0.0,
        -2.0 / 10.0_f64.sqrt(),
        2.0 / 10.0_f64.sqrt(),
        0.0,
        0.0,
        1.0 / 10.0_f64.sqrt(),
        -1.0 / 10.0_f64.sqrt(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    let rotation_y = ndarray::arr1(&[
        0.0,
        0.0,
        0.0,
        0.0,
        3.0 / 20.0_f64.sqrt(),
        -3.0 / 20.0_f64.sqrt(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0 / 20.0_f64.sqrt(),
        1.0 / 20.0_f64.sqrt(),
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    let rotation_x = ndarray::arr1(&[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -3.0 / 26.0_f64.sqrt(),
        3.0 / 26.0_f64.sqrt(),
        0.0,
        0.0,
        2.0 / 26.0_f64.sqrt(),
        -2.0 / 26.0_f64.sqrt(),
        0.0,
        0.0,
    ]);
    assert_dmdw_close(projection.column(3).dot(&rotation_z).abs(), 1.0);
    assert_dmdw_close(projection.column(4).dot(&rotation_y).abs(), 1.0);
    assert_dmdw_close(projection.column(5).dot(&rotation_x).abs(), 1.0);
    Ok(())
}

#[test]
fn dmdw_seed_projection_matches_feff_qj0_loop() -> Result<(), DebyeError> {
    let seed = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
    let inv_sqrt_two = 0.5_f64.sqrt();
    let modes = ndarray::arr2(&[
        [1.0, 0.0],
        [0.0, inv_sqrt_two],
        [0.0, inv_sqrt_two],
        [0.0, 0.0],
    ]);

    let projected = dmdw_project_seed_vector(seed.view(), modes.view())?;
    assert_vector_close(
        &projected,
        &[
            0.0,
            -0.123_091_490_979_332_72,
            0.123_091_490_979_332_72,
            0.984_731_927_834_661_8,
        ],
    );
    assert_dmdw_close(projected.iter().map(|value| value * value).sum(), 1.0);

    let normalized = dmdw_normalize_seed_vector(seed.view())?;
    assert_vector_close(
        &normalized,
        &[
            0.182_574_185_835_055_36,
            0.365_148_371_670_110_7,
            0.547_722_557_505_166_1,
            0.730_296_743_340_221_4,
        ],
    );
    Ok(())
}

#[test]
fn dmdw_rigid_body_helpers_reject_invalid_inputs() {
    let empty_positions = ndarray::Array2::<Real>::zeros((0, 3));
    let empty_masses = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_center_of_mass(empty_positions.view(), empty_masses.view()),
        Err(DebyeError::EmptyDmdwAtomTable)
    ));

    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
    let bad_masses = ndarray::arr1(&[-1.0]);
    assert!(matches!(
        dmdw_inertia_tensor(positions.view(), bad_masses.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW atom mass",
            ..
        })
    ));

    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
    let masses = ndarray::arr1(&[1.0]);
    assert!(matches!(
        dmdw_rigid_body_projection_modes(positions.view(), masses.view()),
        Err(DebyeError::TooFewDmdwRigidBodyAtoms { atoms: 1 })
    ));

    let collinear_positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let collinear_masses = ndarray::arr1(&[1.0, 1.0]);
    assert!(matches!(
        dmdw_rigid_body_projection_modes(collinear_positions.view(), collinear_masses.view()),
        Err(DebyeError::ZeroDmdwProjectionModeNorm { .. })
    ));
}

#[test]
fn dmdw_seed_projection_rejects_invalid_inputs() {
    let seed = ndarray::arr1(&[1.0, 2.0]);
    let bad_modes = ndarray::Array2::<Real>::zeros((3, 1));
    assert!(matches!(
        dmdw_project_seed_vector(seed.view(), bad_modes.view()),
        Err(DebyeError::InvalidDmdwProjectionShape { .. })
    ));

    let empty_seed = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_normalize_seed_vector(empty_seed.view()),
        Err(DebyeError::EmptyDmdwSeed)
    ));

    let zero_seed = ndarray::arr1(&[0.0, 0.0]);
    assert!(matches!(
        dmdw_normalize_seed_vector(zero_seed.view()),
        Err(DebyeError::ZeroDmdwSeedNorm)
    ));
}

fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-18,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

fn assert_slice_close(actual: &[Real], expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_dmdw_close(*actual, *expected);
    }
}

fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: &[[Real; 3]; 3]) {
    assert_eq!(actual.shape(), &[3, 3]);
    for row in 0..3 {
        for column in 0..3 {
            assert_dmdw_close(actual[(row, column)], expected[row][column]);
        }
    }
}

fn assert_matrix_abs_close(actual: ArrayView2<'_, Real>, expected: &[[Real; 3]; 3]) {
    assert_eq!(actual.shape(), &[3, 3]);
    for row in 0..3 {
        for column in 0..3 {
            assert_dmdw_close(actual[(row, column)].abs(), expected[row][column].abs());
        }
    }
}

fn sample_dmdw_type2_blocks() -> (ndarray::Array4<Real>, ndarray::Array1<Real>) {
    let masses = ndarray::arr1(&[63.546, 63.546, 63.546]);
    let mut force_blocks = ndarray::Array4::zeros((3, 3, 3, 3));
    for atom in 0..3 {
        for component in 0..3 {
            force_blocks[(atom, atom, component, component)] =
                0.02 + 0.003 * atom as Real + 0.001 * component as Real;
        }
    }
    for component in 0..3 {
        force_blocks[(0, 1, component, component)] = -0.004;
        force_blocks[(1, 0, component, component)] = -0.004;
        force_blocks[(1, 2, component, component)] = -0.003;
        force_blocks[(2, 1, component, component)] = -0.003;
    }
    (force_blocks, masses)
}

fn sample_dmdw_type2_coupling() -> DmdwPhononCoupling {
    DmdwPhononCoupling {
        energy_hartree: ndarray::arr1(&[0.001, 0.002, 0.004]),
        energy_ev: ndarray::arr1(&[
            0.001 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.002 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.004 * DMDW_COUPLING_ENERGY_HARTREE_EV,
        ]),
        eliashberg: ndarray::arr1(&[0.5, 1.0, 1.5]),
        matrix_element: ndarray::arr1(&[0.05, 0.05, 0.05]),
        normalization: 1.0,
    }
}

fn assert_vector_close(actual: &Array1<Real>, expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_dmdw_close(*actual, *expected);
    }
}

fn assert_complex_dmdw_close(actual: Complex, expected: Complex) {
    assert_dmdw_close(actual.re, expected.re);
    assert_dmdw_close(actual.im, expected.im);
}

fn assert_complex_dmdw_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "actual={} expected={} diff={}",
        actual.re,
        expected.re,
        (actual.re - expected.re).abs()
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "actual={} expected={} diff={}",
        actual.im,
        expected.im,
        (actual.im - expected.im).abs()
    );
}

fn column_dot(matrix: ArrayView2<'_, Real>, left: usize, right: usize) -> Real {
    let left_column = matrix.column(left);
    let right_column = matrix.column(right);
    left_column
        .iter()
        .zip(right_column.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

fn assert_moment_summary(
    actual: &DmdwMomentSummary,
    expected_moment: Real,
    expected_frequency: Real,
    reduced_mass: Real,
) -> Result<(), DebyeError> {
    assert_dmdw_close(actual.moment_thz_power_n, expected_moment);
    let expected = dmdw_single_pole_einstein_summary(expected_frequency, reduced_mass)?;
    assert_dmdw_close(
        actual.frequency_thz.ok_or(DebyeError::NonFiniteOutput {
            name: "test moment frequency",
            value: Real::NAN,
        })?,
        expected.frequency_thz,
    );
    assert_dmdw_close(
        actual
            .temperature_kelvin
            .ok_or(DebyeError::NonFiniteOutput {
                name: "test moment temperature",
                value: Real::NAN,
            })?,
        expected.temperature_kelvin,
    );
    assert_dmdw_close(
        actual
            .effective_force_constant_n_per_m
            .ok_or(DebyeError::NonFiniteOutput {
                name: "test moment force constant",
                value: Real::NAN,
            })?,
        expected.effective_force_constant_n_per_m,
    );
    Ok(())
}

fn assert_dmdw_close(actual: Real, expected: Real) {
    let tolerance = expected.abs().max(1.0) * 1.0e-14;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

fn assert_relative_close(actual: Real, expected: Real) {
    let tolerance = expected.abs().max(1.0) * 1.0e-14;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}
