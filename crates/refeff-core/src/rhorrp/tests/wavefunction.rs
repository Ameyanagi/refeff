use super::{support::*, *};

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
