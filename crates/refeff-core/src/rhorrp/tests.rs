use ndarray::{Array1, Array3, arr2};

use super::*;

#[test]
fn density_grid_points_match_feff_reference() -> Result<(), RhorrpError> {
    let axes = reference_axes();
    let input = reference_grid_input(&axes);
    let points = rhorrp_density_grid_points(input)?;

    assert_eq!(points.points.dim(), (3, 24));
    assert_vector_close(column(&points.points, 0), [0.1, -0.2, 0.3]);
    assert_vector_close(column(&points.points, 1), [0.7, -0.4, 0.4]);
    assert_vector_close(column(&points.points, 3), [-0.2, 0.7, 0.8]);
    assert_vector_close(
        column(&points.points, 6),
        [0.233333333333333, -0.166666666666667, 0.666666666666667],
    );
    assert_vector_close(column(&points.points, 23), [1.4, 0.4, 2.1]);
    Ok(())
}

#[test]
fn evaluate_density_grid_matches_feff_reference() -> Result<(), RhorrpError> {
    let axes = reference_axes();
    let input = reference_grid_input(&axes);
    let evaluated = rhorrp_evaluate_density_grid(input, |point| Ok(sample_density(point)))?;

    assert_eq!(evaluated.point_count(), 24);
    assert_eq!(evaluated.points.dim(), (3, 24));
    assert_vector_close(column(&evaluated.points, 0), [0.1, -0.2, 0.3]);
    assert_real_close(evaluated.density_per_bohr3[0], -0.470_000_000_000_000_1);
    assert_vector_close(column(&evaluated.points, 1), [0.7, -0.4, 0.4]);
    assert_real_close(evaluated.density_per_bohr3[1], -0.580_000_000_000_000_1);
    assert_vector_close(column(&evaluated.points, 3), [-0.2, 0.7, 0.8]);
    assert_real_close(evaluated.density_per_bohr3[3], 0.659_999_999_999_999_9);
    assert_vector_close(
        column(&evaluated.points, 6),
        [0.233333333333333, -0.166666666666667, 0.666666666666667],
    );
    assert_real_close(evaluated.density_per_bohr3[6], -0.472_222_222_222_222_27);
    assert_vector_close(column(&evaluated.points, 23), [1.4, 0.4, 2.1]);
    assert_real_close(evaluated.density_per_bohr3[23], 1.709_999_999_999_999_5);
    Ok(())
}

#[test]
fn point_and_next_index_match_feff_order() -> Result<(), RhorrpError> {
    let axes = reference_axes();
    let input = reference_grid_input(&axes);
    let mut index = vec![1, 1, 1];
    assert_vector_close(rhorrp_point_at_index(input, &index)?, [0.1, -0.2, 0.3]);
    rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
    assert_eq!(index, vec![2, 1, 1]);
    rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
    assert_eq!(index, vec![3, 1, 1]);
    rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
    assert_eq!(index, vec![1, 2, 1]);
    Ok(())
}

#[test]
fn process_ranges_match_feff_reference() -> Result<(), RhorrpError> {
    assert_eq!(
        rhorrp_process_ranges(10, 3)?,
        vec![
            RhorrpProcessRange {
                process: 0,
                start_1based: 1,
                end_1based: 4,
            },
            RhorrpProcessRange {
                process: 1,
                start_1based: 5,
                end_1based: 7,
            },
            RhorrpProcessRange {
                process: 2,
                start_1based: 8,
                end_1based: 10,
            },
        ]
    );
    assert_eq!(
        rhorrp_process_ranges(3, 5)?,
        vec![
            RhorrpProcessRange {
                process: 0,
                start_1based: 1,
                end_1based: 1,
            },
            RhorrpProcessRange {
                process: 1,
                start_1based: 2,
                end_1based: 2,
            },
            RhorrpProcessRange {
                process: 2,
                start_1based: 3,
                end_1based: 3,
            },
            RhorrpProcessRange {
                process: 3,
                start_1based: 4,
                end_1based: 3,
            },
            RhorrpProcessRange {
                process: 4,
                start_1based: 4,
                end_1based: 3,
            },
        ]
    );
    assert_eq!(rhorrp_process_ranges(24, 4)?[3].len(), 6);
    assert!(rhorrp_process_ranges(3, 5)?[3].is_empty());
    assert!(matches!(
        rhorrp_process_ranges(10, 0),
        Err(RhorrpError::InvalidProcessCount)
    ));
    Ok(())
}

#[test]
fn fms_inclusion_counts_match_feff_reference() -> Result<(), RhorrpError> {
    let positions = reference_inclusion_positions();
    let counts = rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
        atom_positions: positions.view(),
        representative_atoms: &[0, 1, 3, 5],
        fms_radius: 1.25,
    })?;

    assert_eq!(counts, vec![4, 2, 2, 4]);
    Ok(())
}

#[test]
fn nearest_atom_matches_feff_reference() -> Result<(), RhorrpError> {
    let positions = reference_positions();
    let potentials = [0, 2, 1, 3];
    let first = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: [0.7, 0.2, 0.1],
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: Some(3),
    })?;
    assert_eq!(first.atom_index_1based, 2);
    assert_eq!(first.potential_index, 2);
    assert_vector_close(first.displacement, [-0.3, 0.2, 0.1]);

    let z_limited = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: [0.0, 0.1, 0.8],
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: Some(3),
    })?;
    assert_eq!(z_limited.atom_index_1based, 1);
    assert_eq!(z_limited.potential_index, 0);
    assert_vector_close(z_limited.displacement, [0.0, 0.1, 0.8]);

    let z_all = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: [0.0, 0.1, 0.8],
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: None,
    })?;
    assert_eq!(z_all.atom_index_1based, 4);
    assert_eq!(z_all.potential_index, 3);
    assert_vector_close(z_all.displacement, [0.0, 0.1, -0.2]);
    Ok(())
}

#[test]
fn nearest_atom_table_matches_feff_reference() -> Result<(), RhorrpError> {
    let positions = reference_positions();
    let potentials = [0, 2, 1, 3];
    let points = reference_nearest_points();
    let table = rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
        points: points.view(),
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: None,
    })?;

    assert_eq!(table.point_count(), 4);
    assert_vector_close(
        row(&table.displacement_bohr, 0),
        [-0.300_000_000_000_000_04, 0.2, 0.1],
    );
    assert_vector_close(row(&table.displacement_bohr, 1), [0.0, 0.1, -0.2]);
    assert_vector_close(row(&table.displacement_bohr, 2), [0.2, -0.1, 0.1]);
    assert_vector_close(row(&table.displacement_bohr, 3), [0.0, 0.5, 0.5]);
    assert_eq!(table.atom_indices, vec![1, 3, 2, 0]);
    assert_eq!(table.atom_indices_1based, vec![2, 4, 3, 1]);
    assert_eq!(table.potential_indices, vec![2, 3, 1, 0]);
    Ok(())
}

#[test]
fn rhorrp_helpers_reject_invalid_inputs() {
    let axes = arr2(&[[1.0], [0.0], [0.0]]);
    assert!(matches!(
        rhorrp_density_grid_points(RhorrpDensityGridInput {
            origin: [0.0; 3],
            axes: axes.view(),
            points_per_axis: &[1],
        }),
        Err(RhorrpError::InvalidPointCount { axis: 0, value: 1 })
    ));
    assert!(matches!(
        rhorrp_point_at_index(
            RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[2],
            },
            &[3],
        ),
        Err(RhorrpError::InvalidGridIndex {
            axis: 0,
            index: 3,
            limit: 2,
        })
    ));
    assert!(matches!(
        rhorrp_evaluate_density_grid(
            RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[2],
            },
            |_| Ok(f64::NAN),
        ),
        Err(RhorrpError::NonFiniteDensityValue { point: 0, .. })
    ));
    assert!(matches!(
        rhorrp_evaluate_density_grid(
            RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[2],
            },
            |_| Err(RhorrpError::InvalidProcessCount),
        ),
        Err(RhorrpError::InvalidProcessCount)
    ));

    let positions = reference_positions();
    assert!(matches!(
        rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.0; 3],
            atom_positions: positions.view(),
            atom_potentials: &[0, 1],
            fms_atom_count: None,
        }),
        Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: 2,
            atoms: 4,
        })
    ));
    assert!(matches!(
        rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.0; 3],
            atom_positions: positions.view(),
            atom_potentials: &[0, 1, 2, 3],
            fms_atom_count: Some(5),
        }),
        Err(RhorrpError::InvalidFmsAtomCount {
            fms_atom_count: 5,
            atoms: 4,
        })
    ));
    let bad_points = arr2(&[[0.0, 1.0]]);
    assert!(matches!(
        rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
            points: bad_points.view(),
            atom_positions: positions.view(),
            atom_potentials: &[0, 1, 2, 3],
            fms_atom_count: None,
        }),
        Err(RhorrpError::InvalidPointTableShape {
            rows: 1,
            columns: 2,
        })
    ));
    assert!(matches!(
        rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
            atom_positions: positions.view(),
            representative_atoms: &[0, 4],
            fms_radius: 1.0,
        }),
        Err(RhorrpError::InvalidRepresentativeAtom {
            potential: 1,
            representative: 4,
            atoms: 4,
        })
    ));
    assert!(matches!(
        rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
            atom_positions: positions.view(),
            representative_atoms: &[0],
            fms_radius: f64::NAN,
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "fms_radius",
            ..
        })
    ));
}

#[test]
fn radial_interpolation_location_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference = [
        (0.0, 1, 0.0),
        (3.011_942_119_122_021_4e-1, 1, 0.0),
        (5.220_457_767_610_16e-1, 1, 2.499_999_999_999_995_6e-1),
        (1.061_836_546_545_359_6, 4, 7.999_999_999_999_998e-1),
        (1.0_f64.exp(), 6, 0.0),
    ];

    for (radius, index, fraction) in reference {
        let location = rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
            radius,
            x0: 0.7,
            dx: 0.2,
            radial_count: 6,
        })?;
        assert_eq!(location.index_below_1based, index);
        assert_real_close(location.fraction, fraction);
    }
    Ok(())
}

#[test]
fn energy_prefactor_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference_energy = Complex::new(0.03, -0.01);
    let reference = [
        (
            Complex::new(0.2, 0.05),
            Complex::new(7.535_556_933_393_025e-1, 1.290_779_920_176_434_2e-1),
        ),
        (
            Complex::new(-0.1, 0.0),
            Complex::new(2.495_199_004_442_948e-2, 6.497_077_620_608_896e-1),
        ),
        (
            Complex::new(1.5, -0.2),
            Complex::new(2.187_643_962_932_14, -1.407_872_105_075_902e-1),
        ),
    ];

    for (energy, expected) in reference {
        let actual = rhorrp_energy_prefactor(RhorrpEnergyPrefactorInput {
            energy_hartree: energy,
            reference_energy_hartree: reference_energy,
        })?;
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn energy_density_finish_matches_feff_reference() -> Result<(), RhorrpError> {
    let energies = Array1::from_vec(vec![
        Complex::new(0.2, 0.05),
        Complex::new(-0.1, 0.0),
        Complex::new(1.5, -0.2),
    ]);
    let green = Array1::from_vec(vec![
        Complex::new(0.002_385_790_539_293_98, -0.001_327_985_363_644_39),
        Complex::new(0.004_561_327_948_938_82, -0.002_352_045_463_805_32),
        Complex::new(0.007_803_700_836_496_53, -0.003_398_486_727_838_54),
    ]);
    let density = rhorrp_finish_energy_density(RhorrpEnergyDensityInput {
        energies_hartree: energies.view(),
        green_function: green.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        radius: 0.85,
        prime_radius: 1.25,
    })?;

    assert_complex_close(
        density[0],
        Complex::new(1.853_402_097_099_352_3e-3, -6.520_074_157_729_285e-4),
    );
    assert_complex_close(
        density[1],
        Complex::new(1.545_370_733_294_796_5e-3, 2.733_968_902_337_800_6e-3),
    );
    assert_complex_close(
        density[2],
        Complex::new(1.561_718_170_082_886_1e-2, -8.031_379_054_745_488e-3),
    );
    Ok(())
}

#[test]
fn pair_energy_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let tables = reference_pair_energy_tables();
    let energies = reference_pair_energies();
    let density = rhorrp_pair_energy_density(RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.first_regular_large.view(),
        first_irregular_large: tables.first_irregular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        first_irregular_small: tables.first_irregular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: true,
        first_displacement: [0.22, -0.18, 0.44],
        second_displacement: [-0.31, 0.28, 0.36],
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    })?;

    assert_complex_close_tol(
        density[0],
        Complex::new(4.920_268_421_420_252e-3, 1.431_508_175_602_251_9e-3),
        5.0e-11,
    );
    assert_complex_close_tol(
        density[1],
        Complex::new(-2.189_314_228_048_695e-4, 1.597_291_659_336_723_4e-2),
        5.0e-11,
    );
    assert_complex_close_tol(
        density[2],
        Complex::new(1.170_398_112_832_168_2e-1, -4.062_425_109_588_009e-3),
        5.0e-11,
    );
    Ok(())
}

#[test]
fn pair_density_matches_composed_feff_rhorrp_flow() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let pair_energy = RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.first_regular_large.view(),
        first_irregular_large: tables.first_irregular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        first_irregular_small: tables.first_irregular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: true,
        first_displacement: [0.22, -0.18, 0.44],
        second_displacement: [-0.31, 0.28, 0.36],
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    };
    let energy_density = rhorrp_pair_energy_density(pair_energy)?;
    let expected = rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: energies.view(),
        energy_density: energy_density.view(),
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;
    let actual = rhorrp_pair_density(RhorrpPairDensityInput {
        pair_energy,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    assert_real_close(actual, expected);
    Ok(())
}

#[test]
fn same_site_green_matches_feff_reference() -> Result<(), RhorrpError> {
    let tables = reference_same_site_wavefunctions();
    let same = rhorrp_same_site_green(RhorrpSameSiteGreenInput {
        regular_large: tables.regular_large.view(),
        irregular_large: tables.irregular_large.view(),
        regular_small: tables.regular_small.view(),
        irregular_small: tables.irregular_small.view(),
        first_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 1,
            fraction: 0.25,
        },
        second_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 3,
            fraction: 0.60,
        },
        cosine_between: 0.35,
    })?;
    assert_complex_close(
        same[0],
        Complex::new(2.385_790_539_293_98e-3, -1.327_985_363_644_393_7e-3),
    );
    assert_complex_close(
        same[1],
        Complex::new(4.561_327_948_938_822e-3, -2.352_045_463_805_323e-3),
    );
    assert_complex_close(
        same[2],
        Complex::new(7.803_700_836_496_527e-3, -3.398_486_727_838_544e-3),
    );

    let swapped = rhorrp_same_site_green(RhorrpSameSiteGreenInput {
        regular_large: tables.regular_large.view(),
        irregular_large: tables.irregular_large.view(),
        regular_small: tables.regular_small.view(),
        irregular_small: tables.irregular_small.view(),
        first_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 3,
            fraction: 0.70,
        },
        second_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 1,
            fraction: 0.20,
        },
        cosine_between: -0.40,
    })?;
    assert_complex_close(
        swapped[0],
        Complex::new(5.306_066_647_740_74e-5, -2.530_189_581_044_869_3e-3),
    );
    assert_complex_close(
        swapped[1],
        Complex::new(-5.026_528_497_243_523e-3, -4.721_792_936_156_043e-3),
    );
    assert_complex_close(
        swapped[2],
        Complex::new(-1.351_999_119_028_561e-2, -6.841_776_566_875_865e-3),
    );
    Ok(())
}

#[test]
fn scattering_green_matches_feff_reference() -> Result<(), RhorrpError> {
    let tables = reference_scattering_green_tables();
    let scattering = rhorrp_scattering_green(RhorrpScatteringGreenInput {
        first_regular_large: tables.first_regular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: tables.scattering_matrix.view(),
        first_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 1,
            fraction: 0.25,
        },
        second_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 2,
            fraction: 0.40,
        },
        first_displacement: [0.4, -0.2, 0.6],
        second_displacement: [-0.3, 0.5, 0.7],
    })?;

    assert_complex_close_tol(
        scattering[0],
        Complex::new(-3.113_692_815_836_969e-5, -1.736_972_186_353_566e-5),
        5.0e-12,
    );
    assert_complex_close_tol(
        scattering[1],
        Complex::new(-8.550_104_470_011_752e-5, -4.571_303_454_320_471e-5),
        5.0e-12,
    );
    assert_complex_close_tol(
        scattering[2],
        Complex::new(-1.728_024_650_639_756_4e-4, -5.819_987_091_037_537_5e-5),
        5.0e-12,
    );
    Ok(())
}

#[test]
fn wavefunction_interpolation_matches_feff_reference() -> Result<(), RhorrpError> {
    let wavefunctions = reference_wavefunctions();

    let negative = rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
        wavefunctions: wavefunctions.view(),
        index_below_1based: -1,
        fraction: 0.4,
    })?;
    assert_complex_close(negative[(1, 1)], Complex::new(0.0, 0.0));

    let zero = rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
        wavefunctions: wavefunctions.view(),
        index_below_1based: 0,
        fraction: 0.4,
    })?;
    assert_complex_close(zero[(1, 1)], Complex::new(4.48, -2.06));

    let two = rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
        wavefunctions: wavefunctions.view(),
        index_below_1based: 2,
        fraction: 0.35,
    })?;
    assert_complex_close(two[(0, 0)], Complex::new(23.6, -11.95));
    assert_complex_close(two[(2, 2)], Complex::new(25.799999999999997, -11.85));
    Ok(())
}

#[test]
fn fermi_distribution_matches_feff_reference() -> Result<(), RhorrpError> {
    let complex = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
        energy_hartree: Complex::new(0.2, 0.05),
        chemical_potential_hartree: 0.1,
        temperature_hartree: 0.025,
        chemical_potential_override_hartree: None,
    })?;
    assert_complex_close(
        complex,
        Complex::new(-7.396_808_073_316_784e-3, -1.690_641_303_994_834_5e-2),
    );

    let override_mu = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
        energy_hartree: Complex::new(0.2, 0.05),
        chemical_potential_hartree: 0.1,
        temperature_hartree: 0.025,
        chemical_potential_override_hartree: Some(0.22),
    })?;
    assert_complex_close(
        override_mu,
        Complex::new(9.819_914_491_359_244e-1, -4.934_924_358_596_282_6e-1),
    );

    let zero_low = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
        energy_hartree: Complex::new(0.05, 0.0),
        chemical_potential_hartree: 0.1,
        temperature_hartree: 1.0e-6,
        chemical_potential_override_hartree: None,
    })?;
    assert_complex_close(zero_low, Complex::new(1.0, 0.0));

    let zero_high = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
        energy_hartree: Complex::new(0.15, 0.0),
        chemical_potential_hartree: 0.1,
        temperature_hartree: 1.0e-6,
        chemical_potential_override_hartree: None,
    })?;
    assert_complex_close(zero_high, Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn wavefunction_and_fermi_helpers_reject_invalid_inputs() {
    let wavefunctions = reference_wavefunctions();
    assert!(matches!(
        rhorrp_energy_prefactor(RhorrpEnergyPrefactorInput {
            energy_hartree: Complex::new(f64::NAN, 0.0),
            reference_energy_hartree: Complex::new(0.0, 0.0),
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "energy_hartree.real",
            ..
        })
    ));
    assert!(matches!(
        rhorrp_energy_prefactor(RhorrpEnergyPrefactorInput {
            energy_hartree: Complex::new(0.1, 0.0),
            reference_energy_hartree: Complex::new(0.0, f64::NAN),
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "reference_energy_hartree.imag",
            ..
        })
    ));
    let one_energy = Array1::from_vec(vec![Complex::new(0.1, 0.0)]);
    let two_green = Array1::from_vec(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]);
    assert!(matches!(
        rhorrp_finish_energy_density(RhorrpEnergyDensityInput {
            energies_hartree: one_energy.view(),
            green_function: two_green.view(),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            radius: 1.0,
            prime_radius: 1.0,
        }),
        Err(RhorrpError::EnergyDensityLengthMismatch {
            energies: 1,
            green: 2,
        })
    ));
    assert!(matches!(
        rhorrp_finish_energy_density(RhorrpEnergyDensityInput {
            energies_hartree: one_energy.view(),
            green_function: one_energy.view(),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            radius: 0.0,
            prime_radius: 1.0,
        }),
        Err(RhorrpError::InvalidPositiveRadius {
            name: "radius",
            value: 0.0,
        })
    ));
    let pair_tables = reference_pair_energy_tables();
    assert!(matches!(
        rhorrp_pair_energy_density(RhorrpPairEnergyDensityInput {
            energies_hartree: one_energy.view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            first_regular_large: pair_tables.first_regular_large.view(),
            first_irregular_large: pair_tables.first_irregular_large.view(),
            first_regular_small: pair_tables.first_regular_small.view(),
            first_irregular_small: pair_tables.first_irregular_small.view(),
            second_regular_large: pair_tables.second_regular_large.view(),
            second_regular_small: pair_tables.second_regular_small.view(),
            first_phase: pair_tables.first_phase.view(),
            second_phase: pair_tables.second_phase.view(),
            scattering_matrix: Some(pair_tables.scattering_matrix.view()),
            same_atom: true,
            first_displacement: [0.22, -0.18, 0.44],
            second_displacement: [-0.31, 0.28, 0.36],
            radial_x0: 0.7,
            radial_dx: 0.2,
            radial_count: 6,
        }),
        Err(RhorrpError::EnergyDensityLengthMismatch {
            energies: 1,
            green: 3,
        })
    ));
    assert!(matches!(
        rhorrp_pair_energy_density(RhorrpPairEnergyDensityInput {
            energies_hartree: reference_pair_energies().view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            first_regular_large: pair_tables.first_regular_large.view(),
            first_irregular_large: pair_tables.first_irregular_large.view(),
            first_regular_small: pair_tables.first_regular_small.view(),
            first_irregular_small: pair_tables.first_irregular_small.view(),
            second_regular_large: pair_tables.second_regular_large.view(),
            second_regular_small: pair_tables.second_regular_small.view(),
            first_phase: pair_tables.first_phase.view(),
            second_phase: pair_tables.second_phase.view(),
            scattering_matrix: Some(pair_tables.scattering_matrix.view()),
            same_atom: true,
            first_displacement: [f64::NAN, -0.18, 0.44],
            second_displacement: [-0.31, 0.28, 0.36],
            radial_x0: 0.7,
            radial_dx: 0.2,
            radial_count: 6,
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "first_displacement",
            ..
        })
    ));
    assert!(matches!(
        rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
            radius: -1.0,
            x0: 0.7,
            dx: 0.2,
            radial_count: 6,
        }),
        Err(RhorrpError::InvalidRadius { value: -1.0 })
    ));
    assert!(matches!(
        rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
            radius: 0.5,
            x0: 0.7,
            dx: 0.0,
            radial_count: 6,
        }),
        Err(RhorrpError::InvalidRadialStep { value: 0.0 })
    ));
    assert!(matches!(
        rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
            radius: 0.5,
            x0: 0.7,
            dx: 0.2,
            radial_count: 0,
        }),
        Err(RhorrpError::InvalidRadialCount { radial_count: 0 })
    ));

    let same_site_tables = reference_same_site_wavefunctions();
    let bad_component = Array3::<Complex>::zeros((3, 2, 4));
    assert!(matches!(
        rhorrp_same_site_green(RhorrpSameSiteGreenInput {
            regular_large: same_site_tables.regular_large.view(),
            irregular_large: bad_component.view(),
            regular_small: same_site_tables.regular_small.view(),
            irregular_small: same_site_tables.irregular_small.view(),
            first_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 1,
                fraction: 0.25,
            },
            second_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 2,
                fraction: 0.60,
            },
            cosine_between: 0.35,
        }),
        Err(RhorrpError::WavefunctionComponentShapeMismatch {
            component: "irregular_large",
            actual_angular: 2,
            ..
        })
    ));
    assert!(matches!(
        rhorrp_same_site_green(RhorrpSameSiteGreenInput {
            regular_large: same_site_tables.regular_large.view(),
            irregular_large: same_site_tables.irregular_large.view(),
            regular_small: same_site_tables.regular_small.view(),
            irregular_small: same_site_tables.irregular_small.view(),
            first_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 1,
                fraction: 0.25,
            },
            second_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 2,
                fraction: 0.60,
            },
            cosine_between: f64::NAN,
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "cosine_between",
            ..
        })
    ));
    let scattering_tables = reference_scattering_green_tables();
    let bad_phase = Array2::<Complex>::zeros((3, 1));
    assert!(matches!(
        rhorrp_scattering_green(RhorrpScatteringGreenInput {
            first_regular_large: scattering_tables.first_regular_large.view(),
            first_regular_small: scattering_tables.first_regular_small.view(),
            second_regular_large: scattering_tables.second_regular_large.view(),
            second_regular_small: scattering_tables.second_regular_small.view(),
            first_phase: bad_phase.view(),
            second_phase: scattering_tables.second_phase.view(),
            scattering_matrix: scattering_tables.scattering_matrix.view(),
            first_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 1,
                fraction: 0.25,
            },
            second_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 2,
                fraction: 0.40,
            },
            first_displacement: [0.4, -0.2, 0.6],
            second_displacement: [-0.3, 0.5, 0.7],
        }),
        Err(RhorrpError::PhaseShapeMismatch {
            component: "first_phase",
            actual_angular: 1,
            ..
        })
    ));
    let bad_scattering = Array3::<Complex>::zeros((3, 3, 4));
    assert!(matches!(
        rhorrp_scattering_green(RhorrpScatteringGreenInput {
            first_regular_large: scattering_tables.first_regular_large.view(),
            first_regular_small: scattering_tables.first_regular_small.view(),
            second_regular_large: scattering_tables.second_regular_large.view(),
            second_regular_small: scattering_tables.second_regular_small.view(),
            first_phase: scattering_tables.first_phase.view(),
            second_phase: scattering_tables.second_phase.view(),
            scattering_matrix: bad_scattering.view(),
            first_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 1,
                fraction: 0.25,
            },
            second_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 2,
                fraction: 0.40,
            },
            first_displacement: [0.4, -0.2, 0.6],
            second_displacement: [-0.3, 0.5, 0.7],
        }),
        Err(RhorrpError::ScatteringMatrixShapeMismatch {
            actual_rows: 3,
            actual_columns: 4,
            ..
        })
    ));
    assert!(matches!(
        rhorrp_scattering_green(RhorrpScatteringGreenInput {
            first_regular_large: scattering_tables.first_regular_large.view(),
            first_regular_small: scattering_tables.first_regular_small.view(),
            second_regular_large: scattering_tables.second_regular_large.view(),
            second_regular_small: scattering_tables.second_regular_small.view(),
            first_phase: scattering_tables.first_phase.view(),
            second_phase: scattering_tables.second_phase.view(),
            scattering_matrix: scattering_tables.scattering_matrix.view(),
            first_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 1,
                fraction: 0.25,
            },
            second_location: RhorrpRadialInterpolationLocation {
                index_below_1based: 2,
                fraction: 0.40,
            },
            first_displacement: [f64::NAN, -0.2, 0.6],
            second_displacement: [-0.3, 0.5, 0.7],
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "first_displacement",
            ..
        })
    ));
    assert!(matches!(
        rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
            wavefunctions: wavefunctions.view(),
            index_below_1based: 4,
            fraction: 0.0,
        }),
        Err(RhorrpError::InvalidWavefunctionIndex {
            index: 4,
            radial: 4,
        })
    ));

    let empty = Array3::<Complex>::zeros((1, 1, 0));
    assert!(matches!(
        rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
            wavefunctions: empty.view(),
            index_below_1based: 0,
            fraction: 0.0,
        }),
        Err(RhorrpError::InvalidWavefunctionShape {
            energy: 1,
            angular: 1,
            radial: 0,
        })
    ));

    assert!(matches!(
        rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: Complex::new(f64::NAN, 0.0),
            chemical_potential_hartree: 0.1,
            temperature_hartree: 0.025,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "energy_hartree.real",
            ..
        })
    ));
}

#[test]
fn fix_irregular_origin_matches_feff_reference() -> Result<(), RhorrpError> {
    let (radii, values) = reference_irregular_solution();
    let fixed = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
        radii: &radii,
        values: values.view(),
    })?;

    assert_complex_close_tol(
        fixed[0],
        Complex::new(9.791_151_469_085_387, 3.741_459_448_683_99),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[49],
        Complex::new(-2.047_179_619_930_901_1e-1, -8.434_737_680_311_137e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[74],
        Complex::new(-6.916_158_567_064_077e-1, -8.929_639_586_361_882e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[99],
        Complex::new(8.811_645_823_831e-1, 1.866_102_289_679_183_5e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[100],
        Complex::new(9.101_077_089_878_837e-1, 2.302_339_202_367_545e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[119],
        Complex::new(1.094_598_908_088_280_5, 8.401_702_866_503_66e-1),
        1.0e-8,
    );
    Ok(())
}

#[test]
fn fix_irregular_origin_rejects_invalid_inputs() {
    let (radii, values) = reference_irregular_solution();
    assert!(matches!(
        rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii[..99],
            values: values.slice_axis(Axis(0), Slice::from(..99)),
        }),
        Err(RhorrpError::InsufficientIrregularFixPoints {
            points: 99,
            required: 100,
        })
    ));

    assert!(matches!(
        rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii[..100],
            values: values.view(),
        }),
        Err(RhorrpError::IrregularFixLengthMismatch {
            radii: 100,
            values: 120,
        })
    ));
}

#[test]
fn atomic_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference = reference_atomic_density_tables();

    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.08, 0.04, -0.03],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        9.746_265_921_948_757,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.72, -0.15, 0.18],
            orbital_index_1based: 2,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        2.182_748_347_338_233e1,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 3,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        7.107_185_239_762_148e6,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [4.2, 3.9, -2.5],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        0.0,
    );
    Ok(())
}

#[test]
fn atomic_density_rejects_invalid_inputs() {
    let reference = reference_atomic_density_tables();
    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 0,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::InvalidAtomicDensityOrbital {
            orbital: 0,
            orbital_count: 3,
        })
    ));

    let bad_potentials = [0, 1, 3, 1];
    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &bad_potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::InvalidAtomicDensityPotential {
            atom_index_1based: 3,
            potential: 3,
            max_potential: 2,
        })
    ));

    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii[..11],
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::AtomicDensityRadialLengthMismatch {
            radii: 11,
            components: 12,
        })
    ));
}

#[test]
fn integrate_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let (energies, energy_density) = reference_density_integration_inputs();

    assert_real_close(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        })?,
        -4.627_669_214_946_009e-2,
    );
    assert_real_close(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: -0.010,
            temperature_hartree: 0.000_001,
            chemical_potential_override_hartree: None,
        })?,
        -1.115_611_780_024_965e-3,
    );
    Ok(())
}

#[test]
fn integrate_density_rejects_invalid_inputs() {
    let (energies, energy_density) = reference_density_integration_inputs();

    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.slice_axis(Axis(0), Slice::from(..7)),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::DensityIntegrationLengthMismatch {
            energies: 7,
            densities: 8,
        })
    ));
    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 1,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::InvalidDensityIntegrationRealAxisCount {
            real_axis_count: 1,
            energy_count: 8,
        })
    ));

    let vertical_only = Array1::from_vec(vec![
        Complex::new(-0.03, 0.09),
        Complex::new(-0.03, 0.06),
        Complex::new(-0.03, 0.03),
        Complex::new(-0.03, 0.00),
    ]);
    let vertical_density = Array1::from_vec(vec![Complex::new(0.3, 0.1); 4]);
    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: vertical_only.view(),
            energy_density: vertical_density.view(),
            real_axis_count: 4,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::MissingDensityIntegrationCorner)
    ));
}

fn reference_grid_input<'a>(axes: &'a Array2<Real>) -> RhorrpDensityGridInput<'a> {
    RhorrpDensityGridInput {
        origin: [0.1, -0.2, 0.3],
        axes: axes.view(),
        points_per_axis: &[3, 2, 4],
    }
}

fn reference_axes() -> Array2<Real> {
    arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]])
}

fn reference_positions() -> Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

fn reference_nearest_points() -> Array2<Real> {
    arr2(&[
        [0.7, 0.0, 0.2, 0.0],
        [0.2, 0.1, 0.9, 0.5],
        [0.1, 0.8, 0.1, 0.5],
    ])
}

fn reference_inclusion_positions() -> Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 1.4],
        [1.5, 1.5, 0.0],
        [-0.5, 0.2, 0.3],
    ])
}

fn sample_density(point: Vector3) -> Real {
    point[0] + 2.0 * point[1] - 0.5 * point[2] + point[0] * point[1]
}

fn reference_wavefunctions() -> Array3<Complex> {
    Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
        let ie = (energy + 1) as Real;
        let il = angular as Real;
        let ir = (radial + 1) as Real;
        Complex::new(10.0 * ir + il + 0.1 * ie, -5.0 * ir + 0.25 * il - 0.2 * ie)
    })
}

struct ReferenceSameSiteWavefunctions {
    regular_large: Array3<Complex>,
    irregular_large: Array3<Complex>,
    regular_small: Array3<Complex>,
    irregular_small: Array3<Complex>,
}

fn reference_same_site_wavefunctions() -> ReferenceSameSiteWavefunctions {
    ReferenceSameSiteWavefunctions {
        regular_large: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.10 * ie + 0.03 * il + 0.01 * ir,
                -0.06 * ie + 0.02 * il - 0.015 * ir,
            )
        }),
        irregular_large: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                -0.08 * ie + 0.04 * il + 0.025 * ir,
                0.05 * ie - 0.01 * il + 0.02 * ir,
            )
        }),
        regular_small: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.07 * ie - 0.02 * il + 0.018 * ir,
                0.04 * ie + 0.015 * il - 0.012 * ir,
            )
        }),
        irregular_small: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                -0.03 * ie + 0.025 * il - 0.02 * ir,
                0.02 * ie + 0.018 * il + 0.017 * ir,
            )
        }),
    }
}

struct ReferencePairEnergyTables {
    first_regular_large: Array3<Complex>,
    first_irregular_large: Array3<Complex>,
    first_regular_small: Array3<Complex>,
    first_irregular_small: Array3<Complex>,
    second_regular_large: Array3<Complex>,
    second_regular_small: Array3<Complex>,
    first_phase: Array2<Complex>,
    second_phase: Array2<Complex>,
    scattering_matrix: Array3<Complex>,
}

fn reference_pair_energies() -> Array1<Complex> {
    Array1::from_vec(vec![
        Complex::new(0.2, 0.05),
        Complex::new(-0.1, 0.0),
        Complex::new(1.5, -0.2),
    ])
}

fn reference_pair_energy_tables() -> ReferencePairEnergyTables {
    reference_pair_energy_tables_with_energy_count(3)
}

fn reference_pair_energy_tables_with_energy_count(
    energy_count: usize,
) -> ReferencePairEnergyTables {
    ReferencePairEnergyTables {
        first_regular_large: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    0.10 * ie + 0.03 * il + 0.01 * ir,
                    -0.06 * ie + 0.02 * il - 0.015 * ir,
                )
            },
        ),
        first_irregular_large: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    -0.08 * ie + 0.04 * il + 0.025 * ir,
                    0.05 * ie - 0.01 * il + 0.02 * ir,
                )
            },
        ),
        first_regular_small: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    0.07 * ie - 0.02 * il + 0.018 * ir,
                    0.04 * ie + 0.015 * il - 0.012 * ir,
                )
            },
        ),
        first_irregular_small: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    -0.03 * ie + 0.025 * il - 0.02 * ir,
                    0.02 * ie + 0.018 * il + 0.017 * ir,
                )
            },
        ),
        second_regular_large: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    -0.05 * ie + 0.02 * il + 0.014 * ir,
                    0.03 * ie - 0.012 * il + 0.011 * ir,
                )
            },
        ),
        second_regular_small: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    0.045 * ie + 0.018 * il - 0.009 * ir,
                    -0.025 * ie + 0.013 * il + 0.016 * ir,
                )
            },
        ),
        first_phase: Array2::from_shape_fn((energy_count, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(0.015 * ie + 0.04 * il, -0.006 * ie + 0.02 * il)
        }),
        second_phase: Array2::from_shape_fn((energy_count, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(-0.011 * ie + 0.03 * il, 0.007 * ie - 0.015 * il)
        }),
        scattering_matrix: Array3::from_shape_fn((energy_count, 4, 4), |(energy, row, column)| {
            let ie = (energy + 1) as Real;
            let row = (row + 1) as Real;
            let column = (column + 1) as Real;
            Complex::new(
                0.002 * ie + 0.004 * row - 0.003 * column,
                -0.0015 * ie + 0.0025 * row + 0.001 * column,
            )
        }),
    }
}

struct ReferenceScatteringGreenTables {
    first_regular_large: Array3<Complex>,
    first_regular_small: Array3<Complex>,
    second_regular_large: Array3<Complex>,
    second_regular_small: Array3<Complex>,
    first_phase: Array2<Complex>,
    second_phase: Array2<Complex>,
    scattering_matrix: Array3<Complex>,
}

fn reference_scattering_green_tables() -> ReferenceScatteringGreenTables {
    ReferenceScatteringGreenTables {
        first_regular_large: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.10 * ie + 0.03 * il + 0.01 * ir,
                -0.06 * ie + 0.02 * il - 0.015 * ir,
            )
        }),
        first_regular_small: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.07 * ie - 0.02 * il + 0.018 * ir,
                0.04 * ie + 0.015 * il - 0.012 * ir,
            )
        }),
        second_regular_large: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                -0.05 * ie + 0.02 * il + 0.014 * ir,
                0.03 * ie - 0.012 * il + 0.011 * ir,
            )
        }),
        second_regular_small: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.045 * ie + 0.018 * il - 0.009 * ir,
                -0.025 * ie + 0.013 * il + 0.016 * ir,
            )
        }),
        first_phase: Array2::from_shape_fn((3, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(0.015 * ie + 0.04 * il, -0.006 * ie + 0.02 * il)
        }),
        second_phase: Array2::from_shape_fn((3, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(-0.011 * ie + 0.03 * il, 0.007 * ie - 0.015 * il)
        }),
        scattering_matrix: Array3::from_shape_fn((3, 4, 4), |(energy, row, column)| {
            let ie = (energy + 1) as Real;
            let row = (row + 1) as Real;
            let column = (column + 1) as Real;
            Complex::new(
                0.002 * ie + 0.004 * row - 0.003 * column,
                -0.0015 * ie + 0.0025 * row + 0.001 * column,
            )
        }),
    }
}

fn reference_irregular_solution() -> (Vec<Real>, ComplexVec) {
    let radii = (1..=120)
        .map(|index| {
            let index = index as Real;
            0.02 * index + 0.0001 * index * index
        })
        .collect::<Vec<_>>();
    let values = ComplexVec::from_shape_fn(120, |index| {
        let one_based = (index + 1) as Real;
        Complex::new(
            (0.07 * one_based).sin() + 0.002 * one_based,
            (0.05 * one_based).cos() - 0.001 * one_based,
        )
    });
    (radii, values)
}

struct ReferenceAtomicDensityTables {
    radii: Vec<Real>,
    positions: Array2<Real>,
    potentials: [usize; 4],
    large: Array3<Real>,
    small: Array3<Real>,
}

fn reference_atomic_density_tables() -> ReferenceAtomicDensityTables {
    let radii = (1..=12)
        .map(|index| 0.015 + 0.035 * index as Real + 0.001 * (index as Real - 1.0).powi(2))
        .collect::<Vec<_>>();
    let positions = arr2(&[
        [0.0, 0.0, 0.0],
        [0.7, -0.2, 0.15],
        [-0.5, 0.55, -0.25],
        [1.85, 0.2, -0.1],
    ]);
    let potentials = [0, 1, 2, 1];
    let large = Array3::from_shape_fn((12, 3, 3), |(radial, orbital, potential)| {
        let index = (radial + 1) as Real;
        let orbital = (orbital + 1) as Real;
        (0.13 * index).sin() + 0.031 * orbital + 0.047 * potential as Real + 0.12 * radii[radial]
    });
    let small = Array3::from_shape_fn((12, 3, 3), |(radial, orbital, potential)| {
        let index = (radial + 1) as Real;
        let orbital = (orbital + 1) as Real;
        (0.09 * index).cos() - 0.019 * orbital + 0.023 * potential as Real - 0.08 * radii[radial]
    });
    ReferenceAtomicDensityTables {
        radii,
        positions,
        potentials,
        large,
        small,
    }
}

fn reference_density_integration_inputs() -> (Array1<Complex>, Array1<Complex>) {
    let energies = Array1::from_vec(vec![
        Complex::new(-0.030, 0.070),
        Complex::new(-0.030, 0.035),
        Complex::new(-0.030, 0.000),
        Complex::new(0.010, 0.000),
        Complex::new(0.065, 0.000),
        Complex::new(0.130, 0.000),
        Complex::new(0.045, 0.021_991_148_575_128_55),
        Complex::new(0.045, 0.043_982_297_150_257_1),
    ]);
    let energy_density = Array1::from_shape_fn(8, |index| {
        let energy = energies[index];
        let one_based = (index + 1) as Real;
        Complex::new(
            0.40 + 0.07 * one_based + 0.02 * energy.re - 0.15 * energy.im,
            -0.25 + 0.04 * one_based + 0.18 * energy.re + 0.03 * energy.im,
        )
    });
    (energies, energy_density)
}

fn column(points: &RealMat, index: usize) -> Vector3 {
    [points[(0, index)], points[(1, index)], points[(2, index)]]
}

fn row(points: &RealMat, index: usize) -> Vector3 {
    [points[(index, 0)], points[(index, 1)], points[(index, 2)]]
}

fn assert_vector_close(actual: Vector3, expected: Vector3) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
            (actual - expected).abs()
        );
    }
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_complex_close_tol(actual, expected, 1.0e-12);
}

fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
    assert!(
        (actual.re - expected.re).abs() < tolerance,
        "real actual={:.17e}, expected={:.17e}, diff={:.17e}",
        actual.re,
        expected.re,
        (actual.re - expected.re).abs()
    );
    assert!(
        (actual.im - expected.im).abs() < tolerance,
        "imag actual={:.17e}, expected={:.17e}, diff={:.17e}",
        actual.im,
        expected.im,
        (actual.im - expected.im).abs()
    );
}

fn assert_real_close_scaled(actual: Real, expected: Real) {
    let tolerance = 1.0e-11 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() < tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}, tolerance={tolerance:.17e}",
        (actual - expected).abs()
    );
}

fn assert_real_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
        (actual - expected).abs()
    );
}
