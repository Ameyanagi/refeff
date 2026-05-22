use super::{support::*, *};

#[test]
fn xsph_update_nrixs_lg_spectrum_matches_feff_reference() -> Result<(), XsphError> {
    let index_map = arr1(&[1, -1, 2, 1, -2]);
    let orbital_l = arr1(&[0, 1, 2, 3, 4]);
    let final_lj = arr1(&[0, 1, 2, 3, 1]);
    let radial_integrals = arr1(&[
        Complex::new(0.12, -0.03),
        Complex::new(-0.08, 0.19),
        Complex::new(0.31, 0.07),
        Complex::new(-0.22, -0.11),
    ]);
    let q_cosines = arr2(&[[0.25, -0.35], [0.60, -0.40]]);
    let mut transition_weights = Array3::<Real>::zeros((2, 5, 4).f());
    for state in 0..5 {
        let state_feff = state as Real + 1.0;
        for spin in 0..2 {
            let spin_feff = spin as Real;
            for (magnetic_index, magnetic_j2) in [-3, -1, 1, 3].iter().enumerate() {
                let magnetic = Real::from(*magnetic_j2);
                transition_weights[(spin, state, magnetic_index)] = 0.05 * state_feff
                    + 0.11 * spin_feff
                    + 0.017 * magnetic
                    + 0.003 * state_feff * magnetic;
            }
        }
    }

    let q_weights = arr1(&[Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
    xsph_update_nrixs_lg_spectrum(
        XsphLgSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 1,
            index_map: index_map.view(),
            orbital_l: orbital_l.view(),
            final_lj: final_lj.view(),
            initial_j2: 3,
            transition_weights: transition_weights.view(),
            radial_integrals: radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: q_cosines.view(),
            mix_dff: false,
            mdff_mode: 0,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
    )?;
    let expected = [
        Complex::new(1.100_552_32e-2, -1.768_391_84e-2),
        Complex::new(1.004_038_768e-2, -2.057_271_974e-2),
        Complex::new(1.0e-2, -2.0e-2),
        Complex::new(1.156_784_538_79e-2, -2.277_795_550_092_5e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }

    let q_weights = arr1(&[Complex::new(0.7, -0.2), Complex::new(-0.15, 0.45)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(-0.03, 0.04));
    xsph_update_nrixs_lg_spectrum(
        XsphLgSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 0,
            index_map: index_map.view(),
            orbital_l: orbital_l.view(),
            final_lj: final_lj.view(),
            initial_j2: 3,
            transition_weights: transition_weights.view(),
            radial_integrals: radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: q_cosines.view(),
            mix_dff: true,
            mdff_mode: 1,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Irregular,
        },
        spectrum.view_mut(),
    )?;
    let expected = [
        Complex::new(-3.066_69e-2, 3.953_56e-2),
        Complex::new(-2.859_349_665e-2, 3.854_721_595e-2),
        Complex::new(-3.0e-2, 4.0e-2),
        Complex::new(-4.005_742_490_156_249e-2, 3.762_661_973_593_751e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }

    let q_weights = arr1(&[Complex::new(0.25, 0.1), Complex::new(0.3, -0.35)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(0.02, 0.01));
    xsph_update_nrixs_lg_spectrum(
        XsphLgSpectrumUpdateInput {
            calculation_index: 2,
            spin_index: 1,
            index_map: index_map.view(),
            orbital_l: orbital_l.view(),
            final_lj: final_lj.view(),
            initial_j2: 3,
            transition_weights: transition_weights.view(),
            radial_integrals: radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: q_cosines.view(),
            mix_dff: true,
            mdff_mode: 2,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
    )?;
    let expected = [
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(1.995_779_884_1e-2, 8.875_159_533_25e-3),
        Complex::new(2.0e-2, 1.0e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn xsph_update_nrixs_lj_and_atom_spectra_match_feff_reference() -> Result<(), XsphError> {
    let fixture = xsph_spectrum_fixture();

    let q_weights = arr1(&[Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
    let mut spectrum_norm = 0.02;
    xsph_update_nrixs_lj_spectrum(
        XsphLjSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 1,
            index_map: fixture.index_map.view(),
            final_lj: fixture.final_lj.view(),
            initial_j2: 3,
            transition_weights: fixture.transition_weights.view(),
            radial_integrals: fixture.radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: fixture.q_cosines.view(),
            mix_dff: false,
            mdff_mode: 0,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
        &mut spectrum_norm,
    )?;
    let expected = [
        Complex::new(1.100_552_319_999_999_9e-2, -1.768_391_84e-2),
        Complex::new(1.004_038_767_999_999_9e-2, -2.057_271_974e-2),
        Complex::new(1.0e-2, -2.0e-2),
        Complex::new(1.156_784_538_790_000_2e-2, -2.277_795_550_092_500_2e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    assert_close(spectrum_norm, 7.678_319_047_619_049e-2);

    let q_weights = arr1(&[Complex::new(0.7, -0.2), Complex::new(-0.15, 0.45)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(-0.03, 0.04));
    let mut spectrum_norm = -0.01;
    xsph_update_nrixs_lj_spectrum(
        XsphLjSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 0,
            index_map: fixture.index_map.view(),
            final_lj: fixture.final_lj.view(),
            initial_j2: 3,
            transition_weights: fixture.transition_weights.view(),
            radial_integrals: fixture.radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: fixture.q_cosines.view(),
            mix_dff: true,
            mdff_mode: 1,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Irregular,
        },
        spectrum.view_mut(),
        &mut spectrum_norm,
    )?;
    let expected = [
        Complex::new(-3.066_689_999_999_999_7e-2, 3.953_560_000_000_000_4e-2),
        Complex::new(-2.859_349_665e-2, 3.854_721_595_000_000_5e-2),
        Complex::new(-3e-2, 4e-2),
        Complex::new(-4.005_742_490_156_249e-2, 3.762_661_973_593_750_5e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    assert_close(spectrum_norm, -1.0e-2);

    let q_weights = arr1(&[Complex::new(0.25, 0.1), Complex::new(0.3, -0.35)]);
    let mut spectrum = Array1::from_elem(5, Complex::new(0.02, 0.01));
    let mut spectrum_norm = 0.005;
    xsph_update_nrixs_atom_spectrum(
        XsphLjSpectrumUpdateInput {
            calculation_index: 2,
            spin_index: 1,
            index_map: fixture.index_map.view(),
            final_lj: fixture.final_lj.view(),
            initial_j2: 3,
            transition_weights: fixture.transition_weights.view(),
            radial_integrals: fixture.radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: fixture.q_cosines.view(),
            mix_dff: true,
            mdff_mode: 2,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
        &mut spectrum_norm,
    )?;
    let expected = [
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(1.995_779_884_1e-2, 8.875_159_533_25e-3),
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(1.969_139_016e-2, 1.094_586_912e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    assert_close(spectrum_norm, 8.780_333_333_333_334e-3);

    Ok(())
}
