use super::{support::*, *};
use crate::{FovrgYkZkExchangeInput, fovrg_yk_zk_exchange, somm2};

#[test]
fn xsph_xsect_spin_merge_matches_feff_two_spin_output() -> Result<(), XsphError> {
    let spectrum_norms = arr1(&[2.0, 6.0]);
    let cross_sections = arr1(&[Complex::new(0.3, 0.1), Complex::new(0.7, -0.4)]);
    let reduced_matrix_elements = Array3::from_shape_fn((1, 3, 2).f(), |(_, transition, spin)| {
        Complex::new(transition as Real + 1.0, spin as Real + 0.25)
    });

    let result = xsph_xsect_spin_merge(XsphXsectSpinMergeInput {
        spin_polarized: true,
        spectrum_norms: spectrum_norms.view(),
        cross_sections: cross_sections.view(),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        q_count: 1,
        transition_count: 3,
    })?;

    assert_close(result.spectrum_norm, 4.0);
    assert_complex_close(result.cross_section, Complex::new(1.0, -0.3));
    let [first_scale, last_scale] = result.spin_scales.expect("nq=1 scales both spin channels");
    assert_close(first_scale, 0.5_f64.sqrt());
    assert_close(last_scale, 1.5_f64.sqrt());
    for transition in 0..3 {
        assert_complex_close(
            result.reduced_matrix_elements[(0, transition, 0)],
            reduced_matrix_elements[(0, transition, 0)] * first_scale,
        );
        assert_complex_close(
            result.reduced_matrix_elements[(0, transition, 1)],
            reduced_matrix_elements[(0, transition, 1)] * last_scale,
        );
    }

    Ok(())
}

#[test]
fn xsph_xsect_spin_merge_preserves_first_spin_for_ordinary_output() -> Result<(), XsphError> {
    let spectrum_norms = arr1(&[3.5, 7.0]);
    let cross_sections = arr1(&[Complex::new(-0.2, 0.9), Complex::new(0.4, -0.1)]);
    let reduced_matrix_elements = Array3::from_shape_fn((2, 2, 2).f(), |(iq, transition, spin)| {
        Complex::new(iq as Real + 0.1 * transition as Real, spin as Real - 0.3)
    });

    let result = xsph_xsect_spin_merge(XsphXsectSpinMergeInput {
        spin_polarized: false,
        spectrum_norms: spectrum_norms.view(),
        cross_sections: cross_sections.view(),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        q_count: 2,
        transition_count: 2,
    })?;

    assert_close(result.spectrum_norm, spectrum_norms[0]);
    assert_complex_close(result.cross_section, cross_sections[0]);
    assert_eq!(result.spin_scales, None);
    assert_eq!(result.reduced_matrix_elements, reduced_matrix_elements);

    Ok(())
}

#[test]
fn xsph_xsect_spin_merge_rejects_invalid_spin_average_norm() {
    let spectrum_norms = arr1(&[1.0, -1.0]);
    let cross_sections = arr1(&[Complex::new(0.1, 0.0), Complex::new(0.2, 0.0)]);
    let reduced_matrix_elements = Array3::<Complex>::zeros((1, 1, 2).f());

    let error = xsph_xsect_spin_merge(XsphXsectSpinMergeInput {
        spin_polarized: true,
        spectrum_norms: spectrum_norms.view(),
        cross_sections: cross_sections.view(),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        q_count: 1,
        transition_count: 1,
    })
    .expect_err("spin-average rkk scaling needs a positive normalization sum");

    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "xsect_spin_merge_norm_sum",
            ..
        }
    ));
}

#[test]
fn xsph_tdlda_screened_dipoles_match_feff_dmscf_reference() -> Result<(), XsphError> {
    let mut response = Array3::<Complex>::zeros((2, 2, 2));
    let mut kernel = Array3::<Complex>::zeros((2, 2, 2));

    kernel[(0, 0, 0)] = Complex::new(0.12, 0.03);
    kernel[(0, 0, 1)] = Complex::new(-0.04, 0.02);
    kernel[(0, 1, 0)] = Complex::new(0.07, -0.05);
    kernel[(0, 1, 1)] = Complex::new(0.10, 0.04);
    response[(0, 0, 0)] = Complex::new(0.20, -0.10);
    response[(0, 0, 1)] = Complex::new(0.05, 0.02);
    response[(0, 1, 0)] = Complex::new(-0.03, 0.04);
    response[(0, 1, 1)] = Complex::new(0.16, -0.06);

    kernel[(1, 0, 0)] = Complex::new(0.20, 0.10);
    kernel[(1, 1, 1)] = Complex::new(0.10, -0.20);
    response[(1, 0, 0)] = Complex::new(0.50, -0.20);
    response[(1, 1, 1)] = Complex::new(0.30, 0.10);

    let dipole_matrix = arr2(&[[1.4, -0.6], [2.0, 0.5]]);
    let solved = xsph_tdlda_screened_dipoles(XsphTdldaScreenedDipoleInput {
        energy_count: 2,
        matrix_size: 2,
        response: response.view(),
        kernel: kernel.view(),
        dipole_matrix: dipole_matrix.view(),
    })?;

    let expected = arr2(&[
        [
            Complex::new(1.439_364_075_660_705_6, -0.018_076_853_826_642_036),
            Complex::new(-0.607_857_823_371_887_2, -0.020_563_922_822_475_433),
        ],
        [
            Complex::new(2.272_433_757_781_982_4, 0.025_823_105_126_619_34),
            Complex::new(0.524_861_872_196_197_5, -0.027_624_310_925_602_913),
        ],
    ]);
    for energy in 0..2 {
        for row in 0..2 {
            assert_close_tol(
                solved.screened_dipoles[(energy, row)].re,
                expected[(energy, row)].re,
                2.0e-7,
            );
            assert_close_tol(
                solved.screened_dipoles[(energy, row)].im,
                expected[(energy, row)].im,
                2.0e-7,
            );
        }
    }

    Ok(())
}

#[test]
fn xsph_tdlda_screened_dipoles_reject_invalid_inputs() {
    let mut response = Array3::<Complex>::zeros((1, 1, 1));
    response[(0, 0, 0)] = Complex::new(1.0, 0.0);
    let mut singular_kernel = Array3::<Complex>::zeros((1, 1, 1));
    singular_kernel[(0, 0, 0)] = Complex::new(1.0, 0.0);
    let dipole_matrix = arr2(&[[1.0]]);

    let error = xsph_tdlda_screened_dipoles(XsphTdldaScreenedDipoleInput {
        energy_count: 0,
        matrix_size: 1,
        response: response.view(),
        kernel: singular_kernel.view(),
        dipole_matrix: dipole_matrix.view(),
    })
    .expect_err("dmscf needs at least one active energy row");
    assert_eq!(error, XsphError::EmptyIndexSet);

    let error = xsph_tdlda_screened_dipoles(XsphTdldaScreenedDipoleInput {
        energy_count: 1,
        matrix_size: 1,
        response: response.view(),
        kernel: singular_kernel.view(),
        dipole_matrix: dipole_matrix.view(),
    })
    .expect_err("1 - K*chi0 must be nonsingular");
    assert!(matches!(
        error,
        XsphError::Linalg(refeff_linalg::LinalgError::SingularMatrix { pivot: 0 })
    ));
}

#[test]
fn xsph_tdlda_separation_function_matches_feff_xsectd_ramp() -> Result<(), XsphError> {
    let energies = arr1(&[
        80.0 / XSPH_HARTREE_EV,
        100.0 / XSPH_HARTREE_EV,
        125.0 / XSPH_HARTREE_EV,
        150.0 / XSPH_HARTREE_EV,
        175.0 / XSPH_HARTREE_EV,
    ]);
    let values = xsph_tdlda_separation_function(0, energies.view(), energies.len())?;
    let expected = arr1(&[0.0, 0.0, 0.5, 1.0, 1.0]);
    for (&actual, &expected) in values.iter().zip(expected.iter()) {
        assert_close(actual, expected);
    }

    let mixed = xsph_tdlda_separation_function(1, energies.view(), energies.len())?;
    assert_eq!(mixed, Array1::from_elem(energies.len(), 1.0));
    let pmbse_only = xsph_tdlda_separation_function(2, energies.view(), energies.len())?;
    assert_eq!(pmbse_only, Array1::from_elem(energies.len(), 0.0));
    let combined = xsph_tdlda_separation_function(3, energies.view(), energies.len())?;
    assert_eq!(combined, Array1::from_elem(energies.len(), 1.0));

    Ok(())
}

#[test]
fn xsph_tdlda_separation_function_rejects_invalid_inputs() {
    let empty = arr1(&[]);
    assert_eq!(
        xsph_tdlda_separation_function(0, empty.view(), 0),
        Err(XsphError::EmptyIndexSet)
    );

    let invalid = arr1(&[Real::NAN]);
    assert!(matches!(
        xsph_tdlda_separation_function(0, invalid.view(), 1),
        Err(XsphError::NonFiniteScalar {
            name: "tdlda_sfun_energy",
            ..
        })
    ));
}

#[test]
fn xsph_tdlda_energy_rows_match_feff_xsectd_setup() -> Result<(), XsphError> {
    let energy_hartree = arr1(&[-11.0, 0.0, 125.0 / XSPH_HARTREE_EV]);
    let reference_energy = arr1(&[
        Complex::new(0.0, 0.0),
        Complex::new(-0.05, 0.01),
        Complex::new(0.1, -0.02),
    ]);
    let spin_orbit_split = 0.03;

    let rows = xsph_tdlda_energy_rows(XsphTdldaEnergyRowsInput {
        energy_count: energy_hartree.len(),
        energy_hartree: energy_hartree.view(),
        reference_energy: reference_energy.view(),
        edge_energy: 0.2,
        chemical_potential: 0.05,
        spin_orbit_split,
        ipmbse: 0,
    })?;

    assert_eq!(rows.active_rows.to_vec(), vec![false, true, true]);
    assert_close(rows.photon_energy[0], 0.1 / XSPH_HARTREE_EV);
    assert_close(rows.photon_energy[1], 0.1 / XSPH_HARTREE_EV);
    assert_close(rows.photon_energy[2], energy_hartree[2] - 0.2 + 0.05);
    assert_close(rows.separation_function[0], 0.0);
    assert_close(rows.separation_function[1], 0.0);
    assert_close(rows.separation_function[2], 0.5);

    for index in 0..energy_hartree.len() {
        let plus = Complex::new(energy_hartree[index], 0.0) - reference_energy[index];
        let minus =
            Complex::new(energy_hartree[index] - spin_orbit_split, 0.0) - reference_energy[index];
        let expected_plus = (2.0 * plus
            + (plus * XSPH_FINE_STRUCTURE_ALPHA) * (plus * XSPH_FINE_STRUCTURE_ALPHA))
            .sqrt();
        let expected_minus = (2.0 * minus
            + (minus * XSPH_FINE_STRUCTURE_ALPHA) * (minus * XSPH_FINE_STRUCTURE_ALPHA))
            .sqrt();
        assert_close_tol(rows.plus_wave_number[index], expected_plus.re, 1.0e-14);
        assert_close_tol(rows.minus_wave_number[index], expected_minus.re, 1.0e-14);
    }

    let pmbse_only = xsph_tdlda_energy_rows(XsphTdldaEnergyRowsInput {
        ipmbse: 2,
        ..XsphTdldaEnergyRowsInput {
            energy_count: energy_hartree.len(),
            energy_hartree: energy_hartree.view(),
            reference_energy: reference_energy.view(),
            edge_energy: 0.2,
            chemical_potential: 0.05,
            spin_orbit_split,
            ipmbse: 0,
        }
    })?;
    assert!(
        pmbse_only
            .separation_function
            .iter()
            .all(|value| *value == 0.0)
    );

    Ok(())
}

#[test]
fn xsph_tdlda_energy_rows_reject_invalid_inputs() {
    let energy_hartree = arr1(&[0.0]);
    let empty_reference = arr1(&[]);
    let error = xsph_tdlda_energy_rows(XsphTdldaEnergyRowsInput {
        energy_count: 1,
        energy_hartree: energy_hartree.view(),
        reference_energy: empty_reference.view(),
        edge_energy: 0.0,
        chemical_potential: 0.0,
        spin_orbit_split: 0.0,
        ipmbse: 0,
    })
    .expect_err("TDLDA energy rows require matching xcpot references");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_energy_rows_reference",
            required: 1,
            actual: 0,
        }
    );

    let bad_reference = arr1(&[Complex::new(0.0, Real::NAN)]);
    assert!(matches!(
        xsph_tdlda_energy_rows(XsphTdldaEnergyRowsInput {
            energy_count: 1,
            energy_hartree: energy_hartree.view(),
            reference_energy: bad_reference.view(),
            edge_energy: 0.0,
            chemical_potential: 0.0,
            spin_orbit_split: 0.0,
            ipmbse: 0,
        }),
        Err(XsphError::NonFiniteComplex {
            name: "tdlda_energy_rows_reference",
            ..
        })
    ));
}

#[test]
fn xsph_tdlda_row_wave_numbers_match_feff_getchi0_reference_shift_rule() -> Result<(), XsphError> {
    let reference_shifts = arr1(&[0.0, -0.4, 0.2]);
    let rows = xsph_tdlda_row_wave_numbers(XsphTdldaRowWaveNumbersInput {
        matrix_size: reference_shifts.len(),
        energy_hartree: 0.3,
        reference_energy: Complex::new(0.1, 0.9),
        reference_shifts: reference_shifts.view(),
    })?;

    assert_eq!(
        rows.positive_momentum_rows.to_vec(),
        vec![true, false, true]
    );
    for row in 0..reference_shifts.len() {
        let expected_p2 = 0.3 - 0.1 + reference_shifts[row];
        let expected_p2_complex = Complex::new(expected_p2, 0.0);
        let expected_wave = (2.0 * expected_p2_complex
            + (expected_p2_complex * XSPH_FINE_STRUCTURE_ALPHA)
                * (expected_p2_complex * XSPH_FINE_STRUCTURE_ALPHA))
            .sqrt();
        assert_close_tol(rows.momentum_squared[row], expected_p2, 1.0e-14);
        assert_close_tol(rows.row_wave_numbers[row], expected_wave.re, 1.0e-14);
    }

    Ok(())
}

#[test]
fn xsph_tdlda_row_wave_numbers_reject_invalid_inputs() {
    let reference_shifts = arr1(&[0.0, 0.1]);
    let error = xsph_tdlda_row_wave_numbers(XsphTdldaRowWaveNumbersInput {
        matrix_size: 0,
        energy_hartree: 0.0,
        reference_energy: Complex::new(0.0, 0.0),
        reference_shifts: reference_shifts.view(),
    })
    .expect_err("TDLDA row wave numbers require at least one row");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_row_wave_number_matrix_size",
            required: 1,
            actual: 0,
        }
    );

    let error = xsph_tdlda_row_wave_numbers(XsphTdldaRowWaveNumbersInput {
        matrix_size: 3,
        energy_hartree: 0.0,
        reference_energy: Complex::new(0.0, 0.0),
        reference_shifts: reference_shifts.view(),
    })
    .expect_err("TDLDA row wave numbers require one reference shift per row");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_row_wave_number_refsh",
            required: 3,
            actual: 2,
        }
    );
}

#[test]
fn xsph_tdlda_kramers_kronig_response_matches_feff_kkchi_reference() -> Result<(), XsphError> {
    let energy_hartree = arr1(&[-0.1, 0.2, 0.7]);
    let reference_shifts = arr1(&[0.0, 0.15]);
    let imaginary_response = Array3::from_shape_vec(
        (3, 2, 2),
        vec![
            0.20, -0.10, 0.05, 0.30, 0.35, 0.02, -0.04, 0.45, 0.10, 0.25, 0.18, -0.12,
        ],
    )
    .expect("test response shape");

    let response = xsph_tdlda_kramers_kronig_response(XsphTdldaKramersKronigInput {
        energy_count: energy_hartree.len(),
        matrix_size: 2,
        energy_hartree: energy_hartree.view(),
        chemical_potential: 0.4,
        edge_energy: 0.0,
        reference_shifts: reference_shifts.view(),
        imaginary_response: imaginary_response.view(),
    })?;

    let expected = Array3::from_shape_vec(
        (3, 2, 2),
        vec![
            0.238_436_098_486_306_87,
            0.054_834_101_693_156_31,
            0.125_432_725_093_321_65,
            0.895_971_535_433_094_2,
            0.102_078_317_969_040_5,
            0.132_731_700_414_864_84,
            0.053_783_646_710_659_31,
            -0.015_638_765_143_053_443,
            -0.153_022_625_312_698_2,
            -0.053_920_086_391_990_446,
            -0.021_835_317_847_618_54,
            -0.156_072_824_277_906_2,
        ],
    )
    .expect("expected response shape");

    for energy in 0..3 {
        for row in 0..2 {
            for column in 0..2 {
                assert_close_tol(
                    response.real_response[(energy, row, column)],
                    expected[(energy, row, column)],
                    2.0e-11,
                );
            }
        }
    }

    Ok(())
}

#[test]
fn xsph_tdlda_condition_response_broadens_and_assembles_complex_chi0() -> Result<(), XsphError> {
    let energy_hartree = arr1(&[-0.1, 0.2, 0.7]);
    let reference_shifts = arr1(&[0.0, 0.15]);
    let row_broadenings = arr1(&[0.08, 0.11]);
    let imaginary_response = Array3::from_shape_vec(
        (3, 2, 2),
        vec![
            0.20, -0.10, 0.05, 0.30, 0.35, 0.02, -0.04, 0.45, 0.10, 0.25, 0.18, -0.12,
        ],
    )
    .expect("test response shape");

    let conditioned = xsph_tdlda_condition_response(XsphTdldaResponseConditioningInput {
        energy_count: energy_hartree.len(),
        matrix_size: 2,
        energy_hartree: energy_hartree.view(),
        chemical_potential: 0.4,
        edge_energy: 0.0,
        reference_shifts: reference_shifts.view(),
        row_broadenings: row_broadenings.view(),
        imaginary_response: imaginary_response.view(),
    })?;

    let energies = energy_hartree.to_vec();
    for row in 0..2 {
        for column in 0..2 {
            let raw = (0..energy_hartree.len())
                .map(|energy| Complex::new(imaginary_response[(energy, row, column)], 0.0))
                .collect::<Vec<_>>();
            let broadened = crate::conv(&energies, &raw, row_broadenings[row])?;
            for energy in 0..energy_hartree.len() {
                assert_close_tol(
                    conditioned.broadened_imaginary_response[(energy, row, column)],
                    broadened[energy].re,
                    broadened[energy].re.abs() * 1.0e-12 + 1.0e-11,
                );
                assert_close_tol(
                    conditioned.response[(energy, row, column)].re,
                    conditioned.real_response[(energy, row, column)],
                    1.0e-12,
                );
                assert_close_tol(
                    conditioned.response[(energy, row, column)].im,
                    conditioned.broadened_imaginary_response[(energy, row, column)],
                    1.0e-12,
                );
            }
        }
    }

    Ok(())
}

#[test]
fn xsph_tdlda_response_conditioning_rejects_invalid_inputs() {
    let energy_hartree = arr1(&[0.0, 1.0]);
    let reference_shifts = arr1(&[0.0]);
    let row_broadenings = arr1(&[0.1]);
    let response = Array3::from_elem((2, 1, 1), 0.25);

    let short_energy = arr1(&[0.0]);
    let short_response = Array3::from_elem((1, 1, 1), 0.25);
    let error = xsph_tdlda_kramers_kronig_response(XsphTdldaKramersKronigInput {
        energy_count: 1,
        matrix_size: 1,
        energy_hartree: short_energy.view(),
        chemical_potential: 0.4,
        edge_energy: 0.0,
        reference_shifts: reference_shifts.view(),
        imaginary_response: short_response.view(),
    })
    .expect_err("kkchi needs at least two active energy rows");
    assert_eq!(
        error,
        XsphError::SizeOutOfRange {
            name: "tdlda_kk_energy_count",
            value: 1,
        }
    );

    let repeated_energy = arr1(&[0.0, 0.0]);
    let error = xsph_tdlda_kramers_kronig_response(XsphTdldaKramersKronigInput {
        energy_hartree: repeated_energy.view(),
        ..XsphTdldaKramersKronigInput {
            energy_count: 2,
            matrix_size: 1,
            energy_hartree: energy_hartree.view(),
            chemical_potential: 0.4,
            edge_energy: 0.0,
            reference_shifts: reference_shifts.view(),
            imaginary_response: response.view(),
        }
    })
    .expect_err("kkchi interpolation requires a strictly increasing grid");
    assert_eq!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "tdlda_kk_energy_step",
            value: 0.0,
        }
    );

    let zero_broadening = arr1(&[0.0]);
    let error = xsph_tdlda_condition_response(XsphTdldaResponseConditioningInput {
        row_broadenings: zero_broadening.view(),
        ..XsphTdldaResponseConditioningInput {
            energy_count: 2,
            matrix_size: 1,
            energy_hartree: energy_hartree.view(),
            chemical_potential: 0.4,
            edge_energy: 0.0,
            reference_shifts: reference_shifts.view(),
            row_broadenings: row_broadenings.view(),
            imaginary_response: response.view(),
        }
    })
    .expect_err("xsectd response broadening must be positive");
    assert_eq!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "tdlda_response_conditioning_broadening",
            value: 0.0,
        }
    );

    let invalid_response = Array3::from_elem((2, 1, 1), Real::NAN);
    assert!(matches!(
        xsph_tdlda_kramers_kronig_response(XsphTdldaKramersKronigInput {
            imaginary_response: invalid_response.view(),
            ..XsphTdldaKramersKronigInput {
                energy_count: 2,
                matrix_size: 1,
                energy_hartree: energy_hartree.view(),
                chemical_potential: 0.4,
                edge_energy: 0.0,
                reference_shifts: reference_shifts.view(),
                imaginary_response: response.view(),
            }
        }),
        Err(XsphError::NonFiniteScalar {
            name: "tdlda_kk_imaginary_response",
            ..
        })
    ));
}

#[test]
fn xsph_tdlda_channel_multipliers_match_feff_ridxmu_single_edge() -> Result<(), XsphError> {
    let photon = arr1(&[100.0, 101.0, 102.0, 103.0]);
    let relative = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let wave_number = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let background = arr1(&[2.0, 2.0, 2.0, 2.0]);
    let chi = arr1(&[0.0, 2.0, 4.0, 6.0]);

    let multipliers = xsph_tdlda_channel_multipliers(XsphTdldaChannelMultipliersInput {
        initial_kappa: -1,
        energy_capacity: 10,
        dominant_plus: xmu_channel(
            photon.view(),
            relative.view(),
            wave_number.view(),
            background.view(),
            chi.view(),
        ),
        split_plus: None,
        dominant_minus: None,
        split_minus: None,
    })?;

    assert_eq!(multipliers.energy_hartree.len(), 4);
    assert_eq!(multipliers.spin_orbit_split, 0.0);
    for row in 0..4 {
        assert_close_tol(
            multipliers.energy_hartree[row],
            relative[row] / XSPH_HARTREE_EV,
            1.0e-14,
        );
    }
    assert_eq!(
        multipliers.channel_multipliers.column(0).to_owned(),
        arr1(&[2.0, 2.0, 3.0, 4.0])
    );
    assert_eq!(
        multipliers.channel_multipliers.column(1).to_owned(),
        Array1::from_elem(4, 1.0)
    );
    assert_eq!(
        multipliers.channel_multipliers.column(2).to_owned(),
        Array1::from_elem(4, 1.0)
    );
    assert_eq!(
        multipliers.channel_multipliers.column(3).to_owned(),
        Array1::from_elem(4, 1.0)
    );

    Ok(())
}

#[test]
fn xsph_tdlda_channel_multipliers_match_feff_ridxmu_split_edges() -> Result<(), XsphError> {
    let odd_plus_photon = arr1(&[100.0, 101.0, 102.0, 103.0]);
    let even_plus_photon = arr1(&[102.0, 102.5, 103.0, 103.5]);
    let odd_photon = arr1(&[100.0, 101.0, 102.0, 103.0]);
    let odd_relative = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let even_relative = arr1(&[0.0, 0.5, 1.0, 1.5]);
    let wave_number = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let background = arr1(&[1.0, 1.0, 1.0, 1.0]);
    let odd_plus_chi = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let even_plus_chi = arr1(&[9.0, 19.0, 29.0, 39.0]);
    let odd_minus_chi = arr1(&[4.0, 5.0, 6.0, 7.0]);
    let even_minus_chi = arr1(&[49.0, 59.0, 69.0, 79.0]);

    let multipliers = xsph_tdlda_channel_multipliers(XsphTdldaChannelMultipliersInput {
        initial_kappa: -2,
        energy_capacity: 10,
        dominant_plus: xmu_channel(
            odd_plus_photon.view(),
            odd_relative.view(),
            wave_number.view(),
            background.view(),
            odd_plus_chi.view(),
        ),
        split_plus: Some(xmu_channel(
            even_plus_photon.view(),
            even_relative.view(),
            wave_number.view(),
            background.view(),
            even_plus_chi.view(),
        )),
        dominant_minus: Some(xmu_channel(
            odd_photon.view(),
            odd_relative.view(),
            wave_number.view(),
            background.view(),
            odd_minus_chi.view(),
        )),
        split_minus: Some(xmu_channel(
            even_plus_photon.view(),
            even_relative.view(),
            wave_number.view(),
            background.view(),
            even_minus_chi.view(),
        )),
    })?;

    let expected_energy_ev = arr1(&[0.0, 1.0, 2.0, 2.5, 3.0, 3.5]);
    assert_eq!(multipliers.energy_hartree.len(), expected_energy_ev.len());
    for row in 0..expected_energy_ev.len() {
        assert_close_tol(
            multipliers.energy_hartree[row],
            expected_energy_ev[row] / XSPH_HARTREE_EV,
            1.0e-14,
        );
    }
    assert_close_tol(multipliers.spin_orbit_split, 2.0 / XSPH_HARTREE_EV, 1.0e-14);

    let expected = arr2(&[
        [2.0, 10.0, 6.0, 50.0],
        [2.0, 10.0, 6.0, 50.0],
        [3.0, 10.0, 7.0, 50.0],
        [3.5, 20.0, 7.5, 60.0],
        [4.0, 30.0, 8.0, 70.0],
        [1.0, 40.0, 1.0, 80.0],
    ]);
    for row in 0..expected.nrows() {
        for channel in 0..4 {
            assert_close_tol(
                multipliers.channel_multipliers[(row, channel)],
                expected[(row, channel)],
                2.0e-12,
            );
        }
    }

    Ok(())
}

#[test]
fn xsph_tdlda_channel_multipliers_reject_invalid_inputs() {
    let photon = arr1(&[100.0, 101.0, 102.0, 103.0]);
    let relative = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let wave_number = arr1(&[0.0, 1.0, 2.0, 3.0]);
    let background = arr1(&[1.0, 1.0, 1.0, 1.0]);
    let chi = arr1(&[0.0, 1.0, 2.0, 3.0]);

    let error = xsph_tdlda_channel_multipliers(XsphTdldaChannelMultipliersInput {
        initial_kappa: -2,
        energy_capacity: 10,
        dominant_plus: xmu_channel(
            photon.view(),
            relative.view(),
            wave_number.view(),
            background.view(),
            chi.view(),
        ),
        split_plus: None,
        dominant_minus: None,
        split_minus: None,
    })
    .expect_err("split-edge kappa requires the even-plus channel");
    assert_eq!(
        error,
        XsphError::MissingTdldaChannel {
            name: "tdlda_ridxmu_even_plus",
        }
    );

    let bad_background = arr1(&[1.0, 1.0, 0.0, 1.0]);
    let error = xsph_tdlda_channel_multipliers(XsphTdldaChannelMultipliersInput {
        initial_kappa: -1,
        energy_capacity: 10,
        dominant_plus: xmu_channel(
            photon.view(),
            relative.view(),
            wave_number.view(),
            bad_background.view(),
            chi.view(),
        ),
        split_plus: None,
        dominant_minus: None,
        split_minus: None,
    })
    .expect_err("ridxmu divides chi by mu0");
    assert_eq!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "tdlda_ridxmu_background",
            value: 0.0,
        }
    );

    let no_edge_wave_number = arr1(&[0.1, 1.0, 2.0, 3.0]);
    assert!(matches!(
        xsph_tdlda_channel_multipliers(XsphTdldaChannelMultipliersInput {
            initial_kappa: -1,
            energy_capacity: 10,
            dominant_plus: xmu_channel(
                photon.view(),
                relative.view(),
                no_edge_wave_number.view(),
                background.view(),
                chi.view(),
            ),
            split_plus: None,
            dominant_minus: None,
            split_minus: None,
        }),
        Err(XsphError::MissingTdldaEdge {
            name: "tdlda_ridxmu_odd_plus",
        })
    ));
}

#[test]
fn xsph_tdlda_weight_response_matches_feff_xsectd_channel_selection() -> Result<(), XsphError> {
    let initial_kappas = arr1(&[-2, 1, -2, 1]);
    let final_kappas = arr1(&[2, 2, -1, -1]);
    let raw = Array3::from_shape_fn((2, 4, 4), |(energy, row, column)| {
        100.0 * energy as Real + 10.0 * row as Real + column as Real + 1.0
    });
    let multipliers = arr2(&[[10.0, 20.0, 30.0, 40.0], [1.5, 2.5, 3.5, 4.5]]);

    let weighted = xsph_tdlda_weight_response(XsphTdldaWeightedResponseInput {
        energy_count: 2,
        matrix_size: 4,
        initial_kappas: initial_kappas.view(),
        final_kappas: final_kappas.view(),
        raw_imaginary_response: raw.view(),
        channel_multipliers: multipliers.view(),
    })?;

    assert_eq!(weighted.row_channels.to_vec(), vec![0, 1, 2, 3]);
    for energy in 0..2 {
        for row in 0..4 {
            for column in 0..4 {
                assert_close(
                    weighted.imaginary_response[(energy, row, column)],
                    raw[(energy, row, column)] * multipliers[(energy, row)],
                );
            }
        }
    }

    Ok(())
}

#[test]
fn xsph_tdlda_raw_response_matches_feff_getchi0_overlap_rules() -> Result<(), XsphError> {
    let matrix_size = 24;
    let mut reference_shifts = Array1::<Real>::zeros(matrix_size);
    reference_shifts[5] = -0.5;
    let row_wave_numbers = Array1::from_shape_fn(matrix_size, |row| 0.5 + 0.01 * row as Real);
    let overlaps = Array1::from_shape_fn(matrix_size, |row| 0.1 + 0.02 * row as Real);
    let localized_dipoles = Array1::from_shape_fn(matrix_size, |row| 1.0 + row as Real);
    let full_dipoles = Array1::from_shape_fn(matrix_size, |row| 2.0 + row as Real);

    let raw = xsph_tdlda_raw_response(XsphTdldaRawResponseInput {
        matrix_size,
        plus_basis_count: 2,
        minus_basis_count: 2,
        initial_l: 1,
        energy_hartree: 0.2,
        edge_energy: 0.0,
        reference_shifts: reference_shifts.view(),
        row_wave_numbers: row_wave_numbers.view(),
        overlaps: overlaps.view(),
        localized_dipoles: localized_dipoles.view(),
        full_dipoles: full_dipoles.view(),
    })?;

    let plus_row = 10;
    let plus_column = plus_row - 9;
    assert_close(
        raw.raw_imaginary_response[(plus_row, plus_row)],
        -2.0 * row_wave_numbers[plus_row] * overlaps[plus_row] * overlaps[plus_row],
    );
    assert_close(
        raw.raw_imaginary_response[(plus_row, plus_column)],
        -2.0 * row_wave_numbers[plus_row] * overlaps[plus_row] * overlaps[plus_column],
    );
    assert_close(
        raw.raw_imaginary_response[(plus_column, plus_row)],
        raw.raw_imaginary_response[(plus_row, plus_column)],
    );

    let minus_row = 21;
    let minus_column = minus_row - 3;
    assert_close(
        raw.raw_imaginary_response[(minus_row, minus_column)],
        -2.0 * row_wave_numbers[minus_row] * overlaps[minus_row] * overlaps[minus_column],
    );
    assert_close(
        raw.raw_imaginary_response[(minus_column, minus_row)],
        raw.raw_imaginary_response[(minus_row, minus_column)],
    );

    assert!(!raw.occupied_rows[5]);
    assert_close(raw.raw_imaginary_response[(5, 5)], 0.0);
    assert_close(raw.localized_dipoles[5], 0.0);
    assert_close(raw.full_dipoles[5], 0.0);
    assert!(raw.occupied_rows[6]);
    assert_close(raw.localized_dipoles[6], localized_dipoles[6]);
    assert_close(raw.full_dipoles[6], full_dipoles[6]);

    Ok(())
}

#[test]
fn xsph_tdlda_raw_response_rejects_invalid_inputs() {
    let reference_shifts = Array1::<Real>::zeros(9);
    let row_wave_numbers = Array1::<Real>::from_elem(9, 1.0);
    let overlaps = Array1::<Real>::from_elem(9, 0.2);
    let localized_dipoles = Array1::<Real>::from_elem(9, 0.3);
    let full_dipoles = Array1::<Real>::from_elem(9, 0.4);

    let error = xsph_tdlda_raw_response(XsphTdldaRawResponseInput {
        matrix_size: 8,
        plus_basis_count: 1,
        minus_basis_count: 0,
        initial_l: 1,
        energy_hartree: 0.0,
        edge_energy: 0.0,
        reference_shifts: reference_shifts.view(),
        row_wave_numbers: row_wave_numbers.view(),
        overlaps: overlaps.view(),
        localized_dipoles: localized_dipoles.view(),
        full_dipoles: full_dipoles.view(),
    })
    .expect_err("TDLDA raw response requires the FEFF getmat matrix size");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_raw_response_matrix_size",
            required: 9,
            actual: 8,
        }
    );

    let short_reference_shifts = Array1::<Real>::zeros(8);
    let error = xsph_tdlda_raw_response(XsphTdldaRawResponseInput {
        matrix_size: 9,
        reference_shifts: short_reference_shifts.view(),
        ..XsphTdldaRawResponseInput {
            matrix_size: 9,
            plus_basis_count: 1,
            minus_basis_count: 0,
            initial_l: 1,
            energy_hartree: 0.0,
            edge_energy: 0.0,
            reference_shifts: reference_shifts.view(),
            row_wave_numbers: row_wave_numbers.view(),
            overlaps: overlaps.view(),
            localized_dipoles: localized_dipoles.view(),
            full_dipoles: full_dipoles.view(),
        }
    })
    .expect_err("TDLDA raw response requires per-row reference shifts");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_raw_response_refsh",
            required: 9,
            actual: 8,
        }
    );
}

#[test]
fn xsph_tdlda_projected_kernel_matches_feff_getchi0_row_folding() -> Result<(), XsphError> {
    let matrix_size = 33;
    let raw_kernel = Array2::from_shape_fn((matrix_size, matrix_size), |(row, column)| {
        Complex::new(
            100.0 * row as Real + column as Real + 1.0,
            -(10.0 * row as Real + column as Real),
        )
    });

    let folded = xsph_tdlda_projected_kernel(XsphTdldaProjectedKernelInput {
        matrix_size,
        plus_basis_count: 3,
        minus_basis_count: 2,
        initial_l: 1,
        projected_kernel: raw_kernel.view(),
    })?;

    assert_complex_close(folded.projected_kernel[(0, 7)], raw_kernel[(0, 7)]);
    assert_complex_close(folded.projected_kernel[(8, 2)], raw_kernel[(8, 2)]);
    assert_complex_close(folded.projected_kernel[(14, 4)], Complex::new(0.0, 0.0));
    assert_complex_close(folded.projected_kernel[(26, 1)], Complex::new(0.0, 0.0));

    for column in 0..matrix_size {
        assert_complex_close(
            folded.projected_kernel[(9, column)],
            raw_kernel[(27, column)],
        );
        assert_complex_close(
            folded.projected_kernel[(10, column)],
            raw_kernel[(28, column)],
        );
        assert_complex_close(
            folded.projected_kernel[(11, column)],
            raw_kernel[(29, column)],
        );
        assert_complex_close(
            folded.projected_kernel[(27, column)],
            Complex::new(0.0, 0.0),
        );
        assert_complex_close(
            folded.projected_kernel[(28, column)],
            Complex::new(0.0, 0.0),
        );
        assert_complex_close(
            folded.projected_kernel[(29, column)],
            Complex::new(0.0, 0.0),
        );
        assert_complex_close(
            folded.projected_kernel[(30, column)],
            Complex::new(0.0, 0.0),
        );
        assert_complex_close(
            folded.projected_kernel[(31, column)],
            Complex::new(0.0, 0.0),
        );
        assert_complex_close(
            folded.projected_kernel[(32, column)],
            Complex::new(0.0, 0.0),
        );
    }

    let single_block = Array2::from_shape_fn((9, 9), |(row, column)| {
        Complex::new(row as Real, column as Real)
    });
    let unchanged = xsph_tdlda_projected_kernel(XsphTdldaProjectedKernelInput {
        matrix_size: 9,
        plus_basis_count: 1,
        minus_basis_count: 0,
        initial_l: 1,
        projected_kernel: single_block.view(),
    })?;
    assert_eq!(unchanged.projected_kernel, single_block);

    Ok(())
}

#[test]
fn xsph_tdlda_projected_kernel_rejects_invalid_inputs() {
    let projected_kernel = Array2::from_elem((9, 9), Complex::new(0.0, 0.0));

    let error = xsph_tdlda_projected_kernel(XsphTdldaProjectedKernelInput {
        matrix_size: 8,
        plus_basis_count: 1,
        minus_basis_count: 0,
        initial_l: 1,
        projected_kernel: projected_kernel.view(),
    })
    .expect_err("TDLDA projected kernel requires the FEFF getmat matrix size");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_projected_kernel_matrix_size",
            required: 9,
            actual: 8,
        }
    );

    let short_projected_kernel = Array2::from_elem((9, 8), Complex::new(0.0, 0.0));
    let error = xsph_tdlda_projected_kernel(XsphTdldaProjectedKernelInput {
        matrix_size: 9,
        plus_basis_count: 1,
        minus_basis_count: 0,
        initial_l: 1,
        projected_kernel: short_projected_kernel.view(),
    })
    .expect_err("TDLDA projected kernel requires a square matrix");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_projected_kernel_cols",
            required: 9,
            actual: 8,
        }
    );
}

#[test]
fn xsph_tdlda_direct_kernel_matches_feff_getchi0_core_hole_potential_rules() -> Result<(), XsphError>
{
    let matrix_size = 24;
    let active_len = 4;
    let radii = arr1(&[1.0, 1.5, 2.5, 4.0]);
    let core_hole_potential = arr1(&[0.5, 0.8, 1.1, 1.4]);
    let localized_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        0.10 * (radial as Real + 1.0) + 0.01 * (row as Real + 1.0)
    });
    let localized_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        0.05 * (radial as Real + 1.0) + 0.003 * row as Real
    });
    let full_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        localized_large[(radial, row)] + 0.20 + 0.002 * row as Real
    });
    let full_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        localized_small[(radial, row)] + 0.10 + 0.001 * radial as Real
    });
    let reference_shifts = Array1::<Real>::zeros(matrix_size);
    let mut momentum_squared = Array1::<Real>::from_elem(matrix_size, 0.2);
    momentum_squared[5] = -0.1;
    let separation_function = 0.25;
    let direct_scale = 1.0 - separation_function;
    let expected_integral = |factor: &dyn Fn(usize) -> Real| {
        let mut integral = 0.0;
        let mut previous = direct_scale * core_hole_potential[0] * factor(0);
        for radial in 1..active_len {
            let current = direct_scale * core_hole_potential[radial] * factor(radial);
            integral += (current + previous) * (radii[radial] - radii[radial - 1]) / 2.0;
            previous = current;
        }
        integral
    };

    let direct = xsph_tdlda_direct_kernel(XsphTdldaDirectKernelInput {
        matrix_size,
        plus_basis_count: 2,
        minus_basis_count: 2,
        initial_l: 1,
        active_len,
        energy_hartree: 0.5,
        edge_energy: 0.0,
        separation_function,
        reference_shifts: reference_shifts.view(),
        momentum_squared: momentum_squared.view(),
        radii: radii.view(),
        core_hole_potential: core_hole_potential.view(),
        localized_large: localized_large.view(),
        localized_small: localized_small.view(),
        full_large: full_large.view(),
        full_small: full_small.view(),
    })?;

    let diag_row = 6;
    let expected_diag = expected_integral(&|radial| {
        localized_large[(radial, diag_row)] * localized_large[(radial, diag_row)]
            + localized_small[(radial, diag_row)] * localized_small[(radial, diag_row)]
    });
    assert_complex_close(
        direct.kernel[(diag_row, diag_row)],
        Complex::new(expected_diag, 0.0),
    );

    let projected_row = 10;
    let projected_target = 1;
    let expected_projected = expected_integral(&|radial| {
        localized_large[(radial, projected_row)] * full_large[(radial, projected_row)]
            + localized_small[(radial, projected_row)] * full_small[(radial, projected_row)]
    });
    assert_complex_close(
        direct.projected_kernel[(projected_target, projected_row)],
        Complex::new(expected_projected, 0.0),
    );

    let plus_row = 12;
    let plus_column = plus_row - 9;
    let expected_plus = expected_integral(&|radial| {
        localized_large[(radial, plus_row)] * localized_large[(radial, plus_column)]
            + localized_small[(radial, plus_row)] * localized_small[(radial, plus_column)]
    });
    assert_complex_close(
        direct.kernel[(plus_row, plus_column)],
        Complex::new(expected_plus, 0.0),
    );
    assert_complex_close(
        direct.kernel[(plus_column, plus_row)],
        Complex::new(expected_plus, 0.0),
    );

    let minus_row = 21;
    let minus_column = minus_row - 3;
    let expected_minus = expected_integral(&|radial| {
        localized_large[(radial, minus_row)] * localized_large[(radial, minus_column)]
            + localized_small[(radial, minus_row)] * localized_small[(radial, minus_column)]
    });
    assert_complex_close(
        direct.kernel[(minus_row, minus_column)],
        Complex::new(expected_minus, 0.0),
    );
    assert_complex_close(
        direct.kernel[(minus_column, minus_row)],
        Complex::new(expected_minus, 0.0),
    );

    assert_complex_close(direct.kernel[(5, 5)], Complex::new(0.0, 0.0));
    assert_complex_close(direct.projected_kernel[(5, 5)], Complex::new(0.0, 0.0));

    Ok(())
}

#[test]
fn xsph_tdlda_direct_kernel_rejects_invalid_inputs() {
    let radii = arr1(&[1.0, 2.0]);
    let core_hole_potential = arr1(&[0.5, 0.6]);
    let matrix = Array2::<Real>::from_elem((2, 9), 0.1);
    let reference_shifts = Array1::<Real>::zeros(9);
    let momentum_squared = Array1::<Real>::from_elem(9, 0.2);

    let error = xsph_tdlda_direct_kernel(XsphTdldaDirectKernelInput {
        matrix_size: 9,
        plus_basis_count: 1,
        minus_basis_count: 0,
        initial_l: 1,
        active_len: 1,
        energy_hartree: 0.0,
        edge_energy: 0.0,
        separation_function: 0.0,
        reference_shifts: reference_shifts.view(),
        momentum_squared: momentum_squared.view(),
        radii: radii.view(),
        core_hole_potential: core_hole_potential.view(),
        localized_large: matrix.view(),
        localized_small: matrix.view(),
        full_large: matrix.view(),
        full_small: matrix.view(),
    })
    .expect_err("TDLDA direct kernel requires at least two radial rows");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_direct_kernel_active_len",
            required: 2,
            actual: 1,
        }
    );

    let short_matrix = Array2::<Real>::from_elem((2, 8), 0.1);
    let error = xsph_tdlda_direct_kernel(XsphTdldaDirectKernelInput {
        full_small: short_matrix.view(),
        ..XsphTdldaDirectKernelInput {
            matrix_size: 9,
            plus_basis_count: 1,
            minus_basis_count: 0,
            initial_l: 1,
            active_len: 2,
            energy_hartree: 0.0,
            edge_energy: 0.0,
            separation_function: 0.0,
            reference_shifts: reference_shifts.view(),
            momentum_squared: momentum_squared.view(),
            radii: radii.view(),
            core_hole_potential: core_hole_potential.view(),
            localized_large: matrix.view(),
            localized_small: matrix.view(),
            full_large: matrix.view(),
            full_small: matrix.view(),
        }
    })
    .expect_err("TDLDA direct kernel requires every radial matrix column");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_direct_kernel_full_small",
            required: 9,
            actual: 8,
        }
    );
}

#[test]
fn xsph_tdlda_coulomb_fields_match_feff_yzktd_exchange_source() -> Result<(), XsphError> {
    let matrix_size = 3;
    let active_len = 7;
    let source_len = 6;
    let coefficient_count = 4;
    let step = 0.05;
    let radii =
        Array1::from_iter((0..active_len).map(|index| 0.025 * (step * index as Real).exp()));
    let core_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        0.10 + 0.02 * radial as Real + 0.007 * row as Real
    });
    let core_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        0.04 + 0.01 * radial as Real + 0.003 * row as Real
    });
    let core_large_coefficients =
        Array2::from_shape_fn((coefficient_count, matrix_size), |(coefficient, row)| {
            0.20 + 0.03 * coefficient as Real + 0.01 * row as Real
        });
    let core_small_coefficients =
        Array2::from_shape_fn((coefficient_count, matrix_size), |(coefficient, row)| {
            0.06 + 0.02 * coefficient as Real + 0.004 * row as Real
        });
    let core_powers = arr1(&[0.9, 1.1, 1.3]);
    let core_lengths = Array1::from_vec(vec![6_usize, 5, 7]);
    let target_powers = arr1(&[1.0, 1.2, 1.4]);
    let target_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.30 + 0.015 * radial as Real + 0.006 * row as Real,
            0.002 * (radial + row) as Real,
        )
    });
    let target_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.05 + 0.012 * radial as Real + 0.002 * row as Real,
            -0.001 * row as Real,
        )
    });
    let target_large_coefficients =
        Array2::from_shape_fn((coefficient_count, matrix_size), |(coefficient, row)| {
            Complex::new(
                0.40 + 0.02 * coefficient as Real + 0.01 * row as Real,
                0.005 * coefficient as Real,
            )
        });
    let target_small_coefficients =
        Array2::from_shape_fn((coefficient_count, matrix_size), |(coefficient, row)| {
            Complex::new(
                0.08 + 0.01 * coefficient as Real + 0.003 * row as Real,
                -0.002 * row as Real,
            )
        });

    let fields = xsph_tdlda_coulomb_fields(XsphTdldaCoulombFieldsInput {
        matrix_size,
        active_len,
        source_len,
        coefficient_count,
        step,
        multipole: 1,
        radii: radii.view(),
        core_large: core_large.view(),
        core_small: core_small.view(),
        core_large_coefficients: core_large_coefficients.view(),
        core_small_coefficients: core_small_coefficients.view(),
        core_powers: core_powers.view(),
        core_lengths: core_lengths.view(),
        target_large: target_large.view(),
        target_small: target_small.view(),
        target_large_coefficients: target_large_coefficients.view(),
        target_small_coefficients: target_small_coefficients.view(),
        target_powers: target_powers.view(),
    })?;

    for row in 0..matrix_size {
        let expected = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
            large_component: core_large.column(row),
            small_component: core_small.column(row),
            large_coefficients: core_large_coefficients.column(row),
            small_coefficients: core_small_coefficients.column(row),
            partner_large_component: target_large.column(row),
            partner_small_component: target_small.column(row),
            partner_large_coefficients: target_large_coefficients.column(row),
            partner_small_coefficients: target_small_coefficients.column(row),
            radii: radii.view(),
            orbital_power: core_powers[row],
            partner_power: target_powers[row],
            step,
            angular_momentum: 1,
            coefficient_count,
            orbital_len: core_lengths[row],
            source_len,
            active_len,
        })?;
        assert_eq!(fields.computed_lengths[row], expected.computed_len);
        assert_complex_close(fields.origin_constants[row], expected.origin_constant);
        for radial in 0..active_len {
            assert_complex_close(fields.fields[(radial, row)], expected.yk[radial]);
        }
    }

    Ok(())
}

#[test]
fn xsph_tdlda_coulomb_fields_reject_invalid_inputs() {
    let radii = arr1(&[1.0, 2.0]);
    let real_matrix = Array2::<Real>::from_elem((2, 1), 0.1);
    let complex_matrix = Array2::<Complex>::from_elem((2, 1), Complex::new(0.1, 0.0));
    let real_coefficients = Array2::<Real>::from_elem((1, 1), 0.1);
    let complex_coefficients = Array2::<Complex>::from_elem((1, 1), Complex::new(0.1, 0.0));
    let powers = arr1(&[1.0]);
    let lengths = Array1::from_vec(vec![2_usize]);
    let target_powers = arr1(&[1.0]);

    let error = xsph_tdlda_coulomb_fields(XsphTdldaCoulombFieldsInput {
        matrix_size: 1,
        active_len: 1,
        source_len: 1,
        coefficient_count: 1,
        step: 0.05,
        multipole: 1,
        radii: radii.view(),
        core_large: real_matrix.view(),
        core_small: real_matrix.view(),
        core_large_coefficients: real_coefficients.view(),
        core_small_coefficients: real_coefficients.view(),
        core_powers: powers.view(),
        core_lengths: lengths.view(),
        target_large: complex_matrix.view(),
        target_small: complex_matrix.view(),
        target_large_coefficients: complex_coefficients.view(),
        target_small_coefficients: complex_coefficients.view(),
        target_powers: target_powers.view(),
    })
    .expect_err("TDLDA Coulomb fields require at least two radial rows");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_coulomb_field_active_len",
            required: 2,
            actual: 1,
        }
    );

    let short_target = Array2::<Complex>::from_elem((2, 0), Complex::new(0.1, 0.0));
    let error = xsph_tdlda_coulomb_fields(XsphTdldaCoulombFieldsInput {
        target_small: short_target.view(),
        ..XsphTdldaCoulombFieldsInput {
            matrix_size: 1,
            active_len: 2,
            source_len: 1,
            coefficient_count: 1,
            step: 0.05,
            multipole: 1,
            radii: radii.view(),
            core_large: real_matrix.view(),
            core_small: real_matrix.view(),
            core_large_coefficients: real_coefficients.view(),
            core_small_coefficients: real_coefficients.view(),
            core_powers: powers.view(),
            core_lengths: lengths.view(),
            target_large: complex_matrix.view(),
            target_small: complex_matrix.view(),
            target_large_coefficients: complex_coefficients.view(),
            target_small_coefficients: complex_coefficients.view(),
            target_powers: target_powers.view(),
        }
    })
    .expect_err("TDLDA Coulomb fields require every target column");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_coulomb_field_target_small",
            required: 1,
            actual: 0,
        }
    );
}

#[test]
fn xsph_tdlda_nonlocal_exchange_integrals_match_feff_yzktd_bound_bound_source()
-> Result<(), XsphError> {
    let matrix_size = 3;
    let active_len = 6;
    let source_len = 5;
    let coefficient_count = 4;
    let step = 0.06;
    let direct_scale = 0.65;
    let radii =
        Array1::from_iter((0..active_len).map(|index| 0.035 * (step * index as Real).exp()));
    let positive_momentum_rows = arr1(&[true, true, true]);
    let initial_kappas = arr1(&[-2, 1, -2]);
    let core_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        0.16 + 0.021 * radial as Real + 0.009 * row as Real
    });
    let core_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        0.05 + 0.012 * radial as Real + 0.004 * row as Real
    });
    let core_large_coefficients =
        Array2::from_shape_fn((coefficient_count, matrix_size), |(coefficient, row)| {
            0.18 + 0.025 * coefficient as Real + 0.007 * row as Real
        });
    let core_small_coefficients =
        Array2::from_shape_fn((coefficient_count, matrix_size), |(coefficient, row)| {
            0.07 + 0.014 * coefficient as Real + 0.003 * row as Real
        });
    let core_powers = arr1(&[0.8, 1.1, 1.4]);
    let core_lengths = Array1::from_vec(vec![5_usize, 4, 6]);
    let localized_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.24 + 0.018 * radial as Real + 0.011 * row as Real,
            0.002 * (radial + row) as Real,
        )
    });
    let localized_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.045 + 0.009 * radial as Real + 0.003 * row as Real,
            -0.001 * row as Real,
        )
    });
    let full_large = localized_large.mapv(|value| value + Complex::new(0.08, -0.006));
    let full_small = localized_small.mapv(|value| value + Complex::new(0.025, 0.002));

    let nonlocal = xsph_tdlda_nonlocal_exchange_integrals(XsphTdldaNonlocalExchangeInput {
        matrix_size,
        active_len,
        source_len,
        coefficient_count,
        step,
        multipole: 2,
        direct_scale,
        positive_momentum_rows: positive_momentum_rows.view(),
        initial_kappas: initial_kappas.view(),
        radii: radii.view(),
        core_large: core_large.view(),
        core_small: core_small.view(),
        core_large_coefficients: core_large_coefficients.view(),
        core_small_coefficients: core_small_coefficients.view(),
        core_powers: core_powers.view(),
        core_lengths: core_lengths.view(),
        localized_large: localized_large.view(),
        localized_small: localized_small.view(),
        full_large: full_large.view(),
        full_small: full_small.view(),
    })?;

    let expected_pair =
        |row: usize, column: usize, projected: bool| -> Result<Complex, XsphError> {
            let partner_large = core_large
                .column(row)
                .mapv(|value| Complex::new(value, 0.0));
            let partner_small = core_small
                .column(row)
                .mapv(|value| Complex::new(value, 0.0));
            let partner_large_coefficients = core_large_coefficients
                .column(row)
                .mapv(|value| Complex::new(value, 0.0));
            let partner_small_coefficients = core_small_coefficients
                .column(row)
                .mapv(|value| Complex::new(value, 0.0));
            let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                large_component: core_large.column(column),
                small_component: core_small.column(column),
                large_coefficients: core_large_coefficients.column(column),
                small_coefficients: core_small_coefficients.column(column),
                partner_large_component: partner_large.view(),
                partner_small_component: partner_small.view(),
                partner_large_coefficients: partner_large_coefficients.view(),
                partner_small_coefficients: partner_small_coefficients.view(),
                radii: radii.view(),
                orbital_power: core_powers[column],
                partner_power: core_powers[row],
                step,
                angular_momentum: 2,
                coefficient_count,
                orbital_len: core_lengths[column].min(core_lengths[row]),
                source_len,
                active_len,
            })?;
            let integrand = |radial: usize| {
                let product = if projected {
                    (localized_large[(radial, column)] * full_large[(radial, row)]
                        + localized_small[(radial, column)] * full_small[(radial, row)])
                        .re
                } else {
                    (localized_large[(radial, column)] * localized_large[(radial, row)]
                        + localized_small[(radial, column)] * localized_small[(radial, row)])
                        .re
                };
                Complex::new(
                    product * transform.yk[radial].re / radii[radial] * direct_scale,
                    0.0,
                )
            };
            let mut integral = Complex::new(0.0, 0.0);
            let mut previous = integrand(0);
            for radial in 1..active_len {
                let current = integrand(radial);
                integral += (current + previous) * (radii[radial] - radii[radial - 1]) / 2.0;
                previous = current;
            }
            Ok(integral)
        };

    assert_complex_close(
        nonlocal.radial_integrals[(1, 0)],
        expected_pair(1, 0, false)?,
    );
    assert_complex_close(
        nonlocal.projected_radial_integrals[(1, 0)],
        expected_pair(1, 0, true)?,
    );
    assert_complex_close(
        nonlocal.radial_integrals[(0, 1)],
        expected_pair(0, 1, false)?,
    );
    assert_complex_close(
        nonlocal.projected_radial_integrals[(0, 1)],
        expected_pair(0, 1, true)?,
    );
    assert_eq!(nonlocal.radial_integrals[(0, 2)], Complex::new(0.0, 0.0));
    assert_eq!(
        nonlocal.projected_radial_integrals[(2, 0)],
        Complex::new(0.0, 0.0)
    );

    Ok(())
}

#[test]
fn xsph_tdlda_nonlocal_exchange_integrals_reject_invalid_inputs() {
    let radii = arr1(&[1.0, 2.0]);
    let real_matrix = Array2::<Real>::from_elem((2, 1), 0.1);
    let real_coefficients = Array2::<Real>::from_elem((1, 1), 0.1);
    let complex_matrix = Array2::<Complex>::from_elem((2, 1), Complex::new(0.1, 0.0));
    let powers = arr1(&[1.0]);
    let lengths = Array1::from_vec(vec![2_usize]);
    let positive_momentum_rows = arr1(&[true]);
    let initial_kappas = arr1(&[-1]);

    let error = xsph_tdlda_nonlocal_exchange_integrals(XsphTdldaNonlocalExchangeInput {
        matrix_size: 1,
        active_len: 1,
        source_len: 1,
        coefficient_count: 1,
        step: 0.05,
        multipole: 2,
        direct_scale: 0.5,
        positive_momentum_rows: positive_momentum_rows.view(),
        initial_kappas: initial_kappas.view(),
        radii: radii.view(),
        core_large: real_matrix.view(),
        core_small: real_matrix.view(),
        core_large_coefficients: real_coefficients.view(),
        core_small_coefficients: real_coefficients.view(),
        core_powers: powers.view(),
        core_lengths: lengths.view(),
        localized_large: complex_matrix.view(),
        localized_small: complex_matrix.view(),
        full_large: complex_matrix.view(),
        full_small: complex_matrix.view(),
    })
    .expect_err("TDLDA nonlocal exchange requires at least two radial rows");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_nonlocal_exchange_active_len",
            required: 2,
            actual: 1,
        }
    );

    let short_full = Array2::<Complex>::from_elem((2, 0), Complex::new(0.1, 0.0));
    let error = xsph_tdlda_nonlocal_exchange_integrals(XsphTdldaNonlocalExchangeInput {
        full_small: short_full.view(),
        ..XsphTdldaNonlocalExchangeInput {
            matrix_size: 1,
            active_len: 2,
            source_len: 1,
            coefficient_count: 1,
            step: 0.05,
            multipole: 2,
            direct_scale: 0.5,
            positive_momentum_rows: positive_momentum_rows.view(),
            initial_kappas: initial_kappas.view(),
            radii: radii.view(),
            core_large: real_matrix.view(),
            core_small: real_matrix.view(),
            core_large_coefficients: real_coefficients.view(),
            core_small_coefficients: real_coefficients.view(),
            core_powers: powers.view(),
            core_lengths: lengths.view(),
            localized_large: complex_matrix.view(),
            localized_small: complex_matrix.view(),
            full_large: complex_matrix.view(),
            full_small: complex_matrix.view(),
        }
    })
    .expect_err("TDLDA nonlocal exchange requires every full-wave column");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_nonlocal_exchange_full_small",
            required: 1,
            actual: 0,
        }
    );
}

#[test]
fn xsph_tdlda_projector_orthogonalization_matches_feff_getwf_tail() -> Result<(), XsphError> {
    let active_len = 5;
    let output_len = 6;
    let log_step = 0.08;
    let norman_radius = 1.9;
    let final_l = 1;
    let radii = arr1(&[0.45, 0.62, 0.85, 1.20, 1.70]);
    let candidate_large = Array1::from_shape_fn(output_len, |index| 0.35 + 0.04 * index as Real);
    let candidate_small = Array1::from_shape_fn(output_len, |index| 0.06 + 0.015 * index as Real);
    let previous_large = Array2::from_shape_fn((output_len, 2), |(radial, previous)| {
        0.10 + 0.02 * radial as Real + 0.015 * previous as Real
    });
    let previous_small = Array2::from_shape_fn((output_len, 2), |(radial, previous)| {
        0.03 + 0.007 * radial as Real + 0.004 * previous as Real
    });

    let result =
        xsph_tdlda_projector_orthogonalization(XsphTdldaProjectorOrthogonalizationInput {
            active_len,
            log_step,
            norman_radius,
            final_l,
            radii: radii.view(),
            candidate_large: candidate_large.view(),
            candidate_small: candidate_small.view(),
            previous_large: previous_large.view(),
            previous_small: previous_small.view(),
        })?;

    let near_origin_power = 2.0 * final_l as Real + 2.0;
    let mut expected_large = candidate_large.clone();
    let mut expected_small = candidate_small.clone();
    let mut expected_overlaps = Array1::<Real>::zeros(2);
    let active_radii = radii.to_vec();
    for previous in 0..2 {
        let samples = (0..active_len)
            .map(|radial| {
                expected_large[radial] * previous_large[(radial, previous)]
                    + expected_small[radial] * previous_small[(radial, previous)]
            })
            .collect::<Vec<_>>();
        let overlap = somm2(
            &active_radii,
            &samples,
            log_step,
            near_origin_power,
            norman_radius,
            0,
        )?;
        expected_overlaps[previous] = overlap;
        for radial in 0..output_len {
            expected_large[radial] -= overlap * previous_large[(radial, previous)];
            expected_small[radial] -= overlap * previous_small[(radial, previous)];
        }
    }
    let norm_samples = (0..active_len)
        .map(|radial| expected_large[radial].powi(2) + expected_small[radial].powi(2))
        .collect::<Vec<_>>();
    let expected_norm = somm2(
        &active_radii,
        &norm_samples,
        log_step,
        near_origin_power,
        norman_radius,
        0,
    )?;
    let expected_sqrt = expected_norm.sqrt();
    expected_large.mapv_inplace(|value| value / expected_sqrt);
    expected_small.mapv_inplace(|value| value / expected_sqrt);

    assert_close(result.norm_integral, expected_norm);
    assert_close(result.norm_sqrt, expected_sqrt);
    for index in 0..2 {
        assert_close(result.overlaps[index], expected_overlaps[index]);
    }
    for radial in 0..output_len {
        assert_close(result.large[radial], expected_large[radial]);
        assert_close(result.small[radial], expected_small[radial]);
    }

    Ok(())
}

#[test]
fn xsph_tdlda_projector_orthogonalization_accepts_first_projector() -> Result<(), XsphError> {
    let active_len = 5;
    let log_step = 0.08;
    let norman_radius = 1.9;
    let final_l = 1;
    let radii = arr1(&[0.45, 0.62, 0.85, 1.20, 1.70]);
    let candidate_large = Array1::from_shape_fn(active_len, |index| 0.35 + 0.04 * index as Real);
    let candidate_small = Array1::from_shape_fn(active_len, |index| 0.06 + 0.015 * index as Real);
    let previous_large = Array2::<Real>::zeros((active_len, 0));
    let previous_small = Array2::<Real>::zeros((active_len, 0));

    let result =
        xsph_tdlda_projector_orthogonalization(XsphTdldaProjectorOrthogonalizationInput {
            active_len,
            log_step,
            norman_radius,
            final_l,
            radii: radii.view(),
            candidate_large: candidate_large.view(),
            candidate_small: candidate_small.view(),
            previous_large: previous_large.view(),
            previous_small: previous_small.view(),
        })?;

    let near_origin_power = 2.0 * final_l as Real + 2.0;
    let norm_samples = (0..active_len)
        .map(|radial| candidate_large[radial].powi(2) + candidate_small[radial].powi(2))
        .collect::<Vec<_>>();
    let expected_norm = somm2(
        &radii.to_vec(),
        &norm_samples,
        log_step,
        near_origin_power,
        norman_radius,
        0,
    )?;
    let expected_sqrt = expected_norm.sqrt();

    assert_eq!(result.overlaps.len(), 0);
    assert_close(result.norm_integral, expected_norm);
    assert_close(result.norm_sqrt, expected_sqrt);
    for radial in 0..active_len {
        assert_close(
            result.large[radial],
            candidate_large[radial] / expected_sqrt,
        );
        assert_close(
            result.small[radial],
            candidate_small[radial] / expected_sqrt,
        );
    }
    Ok(())
}

#[test]
fn xsph_tdlda_projector_orthogonalization_rejects_invalid_inputs() {
    let radii = arr1(&[0.5, 0.7, 1.0, 1.4]);
    let candidate = Array1::<Real>::from_elem(4, 0.2);
    let previous = Array2::<Real>::from_elem((4, 1), 0.1);

    let error = xsph_tdlda_projector_orthogonalization(XsphTdldaProjectorOrthogonalizationInput {
        active_len: 3,
        log_step: 0.08,
        norman_radius: 1.4,
        final_l: 1,
        radii: radii.view(),
        candidate_large: candidate.view(),
        candidate_small: candidate.view(),
        previous_large: previous.view(),
        previous_small: previous.view(),
    })
    .expect_err("TDLDA projector cleanup requires enough somm2 rows");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_projector_active_len",
            required: 4,
            actual: 3,
        }
    );

    let short_previous = Array2::<Real>::from_elem((4, 0), 0.1);
    let error = xsph_tdlda_projector_orthogonalization(XsphTdldaProjectorOrthogonalizationInput {
        previous_small: short_previous.view(),
        ..XsphTdldaProjectorOrthogonalizationInput {
            active_len: 4,
            log_step: 0.08,
            norman_radius: 1.4,
            final_l: 1,
            radii: radii.view(),
            candidate_large: candidate.view(),
            candidate_small: candidate.view(),
            previous_large: previous.view(),
            previous_small: previous.view(),
        }
    })
    .expect_err("TDLDA projector cleanup requires paired previous components");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_projector_previous_small_columns",
            required: 1,
            actual: 0,
        }
    );
}

#[test]
fn xsph_tdlda_radial_kernel_integrals_match_feff_getchi0_ykgr_and_fxc_rules()
-> Result<(), XsphError> {
    let matrix_size = 4;
    let active_len = 4;
    let radii = arr1(&[0.5, 0.8, 1.3, 2.1]);
    let positive_momentum_rows = arr1(&[true, true, true, false]);
    let initial_kappas = arr1(&[-2, -2, 1, -2]);
    let fxc0 = arr1(&[0.10, 0.12, 0.14, 0.16]);
    let fxc = arr1(&[0.20, 0.23, 0.26, 0.29]);
    let fxcim = arr1(&[0.01, 0.02, 0.03, 0.04]);
    let response_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.20 + 0.03 * radial as Real + 0.01 * row as Real,
            0.004 * (radial + row) as Real,
        )
    });
    let response_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.06 + 0.02 * radial as Real + 0.005 * row as Real,
            -0.002 * row as Real,
        )
    });
    let localized_large = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.30 + 0.02 * radial as Real + 0.015 * row as Real,
            0.003 * radial as Real,
        )
    });
    let localized_small = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.04 + 0.01 * radial as Real + 0.006 * row as Real,
            0.001 * (radial + row) as Real,
        )
    });
    let full_large = localized_large.mapv(|value| value + Complex::new(0.12, -0.01));
    let full_small = localized_small.mapv(|value| value + Complex::new(0.03, 0.004));
    let coulomb_fields = Array2::from_shape_fn((active_len, matrix_size), |(radial, row)| {
        Complex::new(
            0.40 + 0.04 * radial as Real + 0.02 * row as Real,
            10.0 + row as Real,
        )
    });
    let direct_scale = 0.75;

    let radial = xsph_tdlda_radial_kernel_integrals(XsphTdldaRadialKernelInput {
        matrix_size,
        active_len,
        positive_momentum_rows: positive_momentum_rows.view(),
        initial_kappas: initial_kappas.view(),
        exchange_correlation_selector: 1,
        direct_scale,
        radii: radii.view(),
        exchange_correlation_same_edge: fxc0.view(),
        exchange_correlation_real: fxc.view(),
        exchange_correlation_imaginary: fxcim.view(),
        response_large: response_large.view(),
        response_small: response_small.view(),
        localized_large: localized_large.view(),
        localized_small: localized_small.view(),
        full_large: full_large.view(),
        full_small: full_small.view(),
        coulomb_fields: coulomb_fields.view(),
    })?;

    for &(row, column) in &[(0, 1), (2, 0), (0, 2)] {
        assert_complex_close(
            radial.radial_integrals[(row, column)],
            expected_tdlda_radial_kernel_integral(
                row,
                column,
                false,
                direct_scale,
                1,
                radii.view(),
                initial_kappas.view(),
                fxc0.view(),
                fxc.view(),
                fxcim.view(),
                response_large.view(),
                response_small.view(),
                localized_large.view(),
                localized_small.view(),
                full_large.view(),
                full_small.view(),
                coulomb_fields.view(),
            ),
        );
        assert_complex_close(
            radial.projected_radial_integrals[(row, column)],
            expected_tdlda_radial_kernel_integral(
                row,
                column,
                true,
                direct_scale,
                1,
                radii.view(),
                initial_kappas.view(),
                fxc0.view(),
                fxc.view(),
                fxcim.view(),
                response_large.view(),
                response_small.view(),
                localized_large.view(),
                localized_small.view(),
                full_large.view(),
                full_small.view(),
                coulomb_fields.view(),
            ),
        );
    }

    assert_close(radial.radial_integrals[(0, 1)].im, 0.0);
    assert!(radial.radial_integrals[(2, 0)].im > 0.0);
    assert!(radial.radial_integrals[(0, 2)].im < 0.0);
    assert_complex_close(radial.radial_integrals[(3, 0)], Complex::new(0.0, 0.0));
    assert_complex_close(
        radial.projected_radial_integrals[(0, 3)],
        Complex::new(0.0, 0.0),
    );

    Ok(())
}

#[test]
fn xsph_tdlda_radial_kernel_integrals_reject_invalid_inputs() {
    let radii = arr1(&[1.0, 2.0]);
    let positive_momentum_rows = arr1(&[true]);
    let initial_kappas = arr1(&[-2]);
    let kernels = arr1(&[0.1, 0.2]);
    let matrix = Array2::<Complex>::from_elem((2, 1), Complex::new(0.1, 0.0));

    let error = xsph_tdlda_radial_kernel_integrals(XsphTdldaRadialKernelInput {
        matrix_size: 1,
        active_len: 1,
        positive_momentum_rows: positive_momentum_rows.view(),
        initial_kappas: initial_kappas.view(),
        exchange_correlation_selector: 1,
        direct_scale: 1.0,
        radii: radii.view(),
        exchange_correlation_same_edge: kernels.view(),
        exchange_correlation_real: kernels.view(),
        exchange_correlation_imaginary: kernels.view(),
        response_large: matrix.view(),
        response_small: matrix.view(),
        localized_large: matrix.view(),
        localized_small: matrix.view(),
        full_large: matrix.view(),
        full_small: matrix.view(),
        coulomb_fields: matrix.view(),
    })
    .expect_err("TDLDA radial kernel requires a trapezoid interval");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_radial_kernel_active_len",
            required: 2,
            actual: 1,
        }
    );

    let short_matrix = Array2::<Complex>::from_elem((2, 0), Complex::new(0.1, 0.0));
    let error = xsph_tdlda_radial_kernel_integrals(XsphTdldaRadialKernelInput {
        response_small: short_matrix.view(),
        ..XsphTdldaRadialKernelInput {
            matrix_size: 1,
            active_len: 2,
            positive_momentum_rows: positive_momentum_rows.view(),
            initial_kappas: initial_kappas.view(),
            exchange_correlation_selector: 1,
            direct_scale: 1.0,
            radii: radii.view(),
            exchange_correlation_same_edge: kernels.view(),
            exchange_correlation_real: kernels.view(),
            exchange_correlation_imaginary: kernels.view(),
            response_large: matrix.view(),
            response_small: matrix.view(),
            localized_large: matrix.view(),
            localized_small: matrix.view(),
            full_large: matrix.view(),
            full_small: matrix.view(),
            coulomb_fields: matrix.view(),
        }
    })
    .expect_err("TDLDA radial kernel requires every response column");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_radial_kernel_response_small",
            required: 1,
            actual: 0,
        }
    );
}

#[allow(clippy::too_many_arguments)]
fn expected_tdlda_radial_kernel_integral(
    row: usize,
    column: usize,
    projected: bool,
    direct_scale: Real,
    ifxc: i32,
    radii: ArrayView1<'_, Real>,
    initial_kappas: ArrayView1<'_, i32>,
    fxc0: ArrayView1<'_, Real>,
    fxc: ArrayView1<'_, Real>,
    fxcim: ArrayView1<'_, Real>,
    response_large: ArrayView2<'_, Complex>,
    response_small: ArrayView2<'_, Complex>,
    localized_large: ArrayView2<'_, Complex>,
    localized_small: ArrayView2<'_, Complex>,
    full_large: ArrayView2<'_, Complex>,
    full_small: ArrayView2<'_, Complex>,
    coulomb_fields: ArrayView2<'_, Complex>,
) -> Complex {
    let mut integral = Complex::new(0.0, 0.0);
    let mut previous = expected_tdlda_radial_kernel_integrand(
        0,
        row,
        column,
        projected,
        direct_scale,
        ifxc,
        radii,
        initial_kappas,
        fxc0,
        fxc,
        fxcim,
        response_large,
        response_small,
        localized_large,
        localized_small,
        full_large,
        full_small,
        coulomb_fields,
    );
    for radial in 1..radii.len() {
        let current = expected_tdlda_radial_kernel_integrand(
            radial,
            row,
            column,
            projected,
            direct_scale,
            ifxc,
            radii,
            initial_kappas,
            fxc0,
            fxc,
            fxcim,
            response_large,
            response_small,
            localized_large,
            localized_small,
            full_large,
            full_small,
            coulomb_fields,
        );
        integral += (current + previous) * (radii[radial] - radii[radial - 1]) / 2.0;
        previous = current;
    }
    integral
}

#[allow(clippy::too_many_arguments)]
fn expected_tdlda_radial_kernel_integrand(
    radial: usize,
    row: usize,
    column: usize,
    projected: bool,
    direct_scale: Real,
    ifxc: i32,
    radii: ArrayView1<'_, Real>,
    initial_kappas: ArrayView1<'_, i32>,
    fxc0: ArrayView1<'_, Real>,
    fxc: ArrayView1<'_, Real>,
    fxcim: ArrayView1<'_, Real>,
    response_large: ArrayView2<'_, Complex>,
    response_small: ArrayView2<'_, Complex>,
    localized_large: ArrayView2<'_, Complex>,
    localized_small: ArrayView2<'_, Complex>,
    full_large: ArrayView2<'_, Complex>,
    full_small: ArrayView2<'_, Complex>,
    coulomb_fields: ArrayView2<'_, Complex>,
) -> Complex {
    let row_product = if projected {
        (response_large[(radial, row)] * full_large[(radial, row)]
            + response_small[(radial, row)] * full_small[(radial, row)])
            .re
    } else {
        (response_large[(radial, row)] * localized_large[(radial, row)]
            + response_small[(radial, row)] * localized_small[(radial, row)])
            .re
    };
    let column_product = (response_large[(radial, column)] * localized_large[(radial, column)]
        + response_small[(radial, column)] * localized_small[(radial, column)])
        .re;
    let exchange = if initial_kappas[row] == initial_kappas[column] && ifxc != 2 {
        Complex::new(fxc0[radial], 0.0)
    } else if initial_kappas[row] > 0 || ifxc == 2 {
        Complex::new(fxc[radial], fxcim[radial])
    } else {
        Complex::new(fxc[radial], -fxcim[radial])
    };
    Complex::new(
        row_product * coulomb_fields[(radial, column)].re / radii[radial] * direct_scale,
        0.0,
    ) + exchange * row_product * column_product
}

#[test]
fn xsph_tdlda_angular_kernel_matches_feff_getchi0_wigner_accumulation() -> Result<(), XsphError> {
    let matrix_size = 4;
    let initial_j2 = arr1(&[3, 3, 1, 1]);
    let initial_m2 = arr1(&[-3, -1, -1, 1]);
    let initial_kappas = arr1(&[-2, 1, 1, -2]);
    let final_j2 = arr1(&[5, 5, 1, 1]);
    let final_m2 = arr1(&[-1, 1, -1, -1]);
    let positive_momentum_rows = arr1(&[true, true, true, false]);
    let radial_integrals = Array2::from_shape_fn((matrix_size, matrix_size), |(row, column)| {
        Complex::new(
            1.0 + 0.25 * row as Real + 0.05 * column as Real,
            0.01 * (row as Real - column as Real),
        )
    });
    let projected_radial_integrals =
        Array2::from_shape_fn((matrix_size, matrix_size), |(row, column)| {
            Complex::new(
                0.5 + 0.11 * row as Real + 0.07 * column as Real,
                0.02 * (row as Real + column as Real),
            )
        });
    let nonlocal_radial_integrals =
        Array2::from_shape_fn((matrix_size, matrix_size), |(row, column)| {
            Complex::new(
                0.2 + 0.03 * row as Real + 0.02 * column as Real,
                -0.01 * column as Real,
            )
        });
    let nonlocal_projected_radial_integrals =
        Array2::from_shape_fn((matrix_size, matrix_size), |(row, column)| {
            Complex::new(
                0.1 + 0.04 * row as Real + 0.01 * column as Real,
                0.015 * row as Real,
            )
        });

    let angular = xsph_tdlda_angular_kernel(XsphTdldaAngularKernelInput {
        matrix_size,
        initial_j2: initial_j2.view(),
        initial_m2: initial_m2.view(),
        initial_kappas: initial_kappas.view(),
        final_j2: final_j2.view(),
        final_m2: final_m2.view(),
        positive_momentum_rows: positive_momentum_rows.view(),
        radial_integrals: radial_integrals.view(),
        projected_radial_integrals: projected_radial_integrals.view(),
        nonlocal_radial_integrals: Some(nonlocal_radial_integrals.view()),
        nonlocal_projected_radial_integrals: Some(nonlocal_projected_radial_integrals.view()),
    })?;

    let main_prefactor = expected_tdlda_angular_prefactor(5, -1, 3, -3, 5, 1, 3, -1, 1)?;
    let nonlocal_prefactor = expected_tdlda_angular_prefactor(5, -1, 5, 1, 3, -3, 3, -1, 2)?;
    let row = 0;
    let column = 1;
    assert_close(angular.prefactors[(row, column)], main_prefactor);
    assert_close(
        angular.nonlocal_prefactors[(row, column)],
        nonlocal_prefactor,
    );
    assert_complex_close(
        angular.kernel[(row, column)],
        radial_integrals[(row, column)] * main_prefactor
            - nonlocal_radial_integrals[(row, column)] * nonlocal_prefactor,
    );
    assert_complex_close(
        angular.projected_kernel[(row, column)],
        projected_radial_integrals[(row, column)] * main_prefactor
            - nonlocal_projected_radial_integrals[(row, column)] * nonlocal_prefactor,
    );

    assert_complex_close(angular.kernel[(0, 2)], Complex::new(0.0, 0.0));
    assert_close(angular.prefactors[(0, 2)], 0.0);
    assert_close(angular.nonlocal_prefactors[(0, 2)], 0.0);
    assert_complex_close(angular.kernel[(3, 0)], Complex::new(0.0, 0.0));
    assert_complex_close(angular.projected_kernel[(0, 3)], Complex::new(0.0, 0.0));

    Ok(())
}

#[test]
fn xsph_tdlda_angular_kernel_rejects_invalid_inputs() {
    let initial_j2 = arr1(&[3]);
    let initial_m2 = arr1(&[-3]);
    let initial_kappas = arr1(&[-2]);
    let final_j2 = arr1(&[5]);
    let final_m2 = arr1(&[-1]);
    let positive_momentum_rows = arr1(&[true]);
    let matrix = Array2::<Complex>::zeros((1, 1));
    let short_matrix = Array2::<Complex>::zeros((1, 0));

    let error = xsph_tdlda_angular_kernel(XsphTdldaAngularKernelInput {
        matrix_size: 1,
        initial_j2: initial_j2.view(),
        initial_m2: initial_m2.view(),
        initial_kappas: initial_kappas.view(),
        final_j2: final_j2.view(),
        final_m2: final_m2.view(),
        positive_momentum_rows: positive_momentum_rows.view(),
        radial_integrals: matrix.view(),
        projected_radial_integrals: short_matrix.view(),
        nonlocal_radial_integrals: None,
        nonlocal_projected_radial_integrals: None,
    })
    .expect_err("TDLDA angular kernel requires full projected radial matrix");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_angular_kernel_projected_radial",
            required: 1,
            actual: 0,
        }
    );

    let error = xsph_tdlda_angular_kernel(XsphTdldaAngularKernelInput {
        matrix_size: 1,
        initial_j2: initial_j2.view(),
        initial_m2: initial_m2.view(),
        initial_kappas: initial_kappas.view(),
        final_j2: final_j2.view(),
        final_m2: final_m2.view(),
        positive_momentum_rows: positive_momentum_rows.view(),
        radial_integrals: matrix.view(),
        projected_radial_integrals: matrix.view(),
        nonlocal_radial_integrals: Some(matrix.view()),
        nonlocal_projected_radial_integrals: None,
    })
    .expect_err("TDLDA angular nonlocal radial inputs must be supplied as a pair");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_angular_kernel_nonlocal_pair",
            required: 2,
            actual: 1,
        }
    );
}

#[allow(clippy::too_many_arguments)]
fn expected_tdlda_angular_prefactor(
    ja2: i32,
    ma2: i32,
    jc2: i32,
    mc2: i32,
    jd2: i32,
    md2: i32,
    jb2: i32,
    mb2: i32,
    multipole: i32,
) -> Result<Real, XsphError> {
    let multipole2 = 2 * multipole;
    let phase = expected_tdlda_phase(ma2) * expected_tdlda_phase(md2);
    Ok(phase
        * wigner_3j(ja2, multipole2, jc2, 1, 0, 2)?
        * wigner_3j(ja2, multipole2, jc2, -ma2, ma2 - mc2, 2)?
        * wigner_3j(jd2, multipole2, jb2, 1, 0, 2)?
        * wigner_3j(jd2, multipole2, jb2, -md2, md2 - mb2, 2)?
        * (((ja2 + 1) * (jd2 + 1) * (jc2 + 1) * (jb2 + 1)) as Real).sqrt())
}

fn expected_tdlda_phase(m2: i32) -> Real {
    if ((m2 + 1) / 2).rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

#[test]
fn xsph_tdlda_weight_response_rejects_invalid_inputs() {
    let initial_kappas = arr1(&[0]);
    let final_kappas = arr1(&[1]);
    let raw = Array3::from_elem((1, 1, 1), 1.0);
    let multipliers = arr2(&[[1.0, 1.0, 1.0, 1.0]]);

    let error = xsph_tdlda_weight_response(XsphTdldaWeightedResponseInput {
        energy_count: 1,
        matrix_size: 1,
        initial_kappas: initial_kappas.view(),
        final_kappas: final_kappas.view(),
        raw_imaginary_response: raw.view(),
        channel_multipliers: multipliers.view(),
    })
    .expect_err("TDLDA weighted response rows require physical kappa values");
    assert_eq!(error, XsphError::ZeroKappa);

    let valid_initial_kappas = arr1(&[-1]);
    let too_few_channels = arr2(&[[1.0, 1.0, 1.0]]);
    let error = xsph_tdlda_weight_response(XsphTdldaWeightedResponseInput {
        initial_kappas: valid_initial_kappas.view(),
        channel_multipliers: too_few_channels.view(),
        ..XsphTdldaWeightedResponseInput {
            energy_count: 1,
            matrix_size: 1,
            initial_kappas: valid_initial_kappas.view(),
            final_kappas: final_kappas.view(),
            raw_imaginary_response: raw.view(),
            channel_multipliers: multipliers.view(),
        }
    })
    .expect_err("TDLDA weighted response requires all four PMBSE channels");
    assert_eq!(
        error,
        XsphError::LengthTooShort {
            name: "tdlda_weighted_response_multiplier_channels",
            required: 4,
            actual: 3,
        }
    );
}

#[test]
fn xsph_tdlda_channel_spectra_match_feff_xsectd_accumulation() -> Result<(), XsphError> {
    let photon_energy = arr1(&[2.0]);
    let plus_wave_number = arr1(&[1.5]);
    let minus_wave_number = arr1(&[0.75]);
    let initial_kappas = arr1(&[-1, 1, -1]);
    let dipole_matrix = arr2(&[[2.0, 3.0, 4.0]]);
    let mut response = Array3::<Complex>::zeros((1, 3, 3));
    let mut projected_kernel = Array3::<Complex>::zeros((1, 3, 3));
    let screened_dipoles = arr2(&[[
        Complex::new(1.0, 0.5),
        Complex::new(0.5, -0.25),
        Complex::new(-0.25, 0.75),
    ]]);

    response[(0, 0, 0)] = Complex::new(0.2, -0.1);
    response[(0, 1, 1)] = Complex::new(0.1, 0.2);
    response[(0, 2, 2)] = Complex::new(-0.1, 0.05);
    projected_kernel[(0, 0, 0)] = Complex::new(0.5, 0.1);
    projected_kernel[(0, 1, 1)] = Complex::new(-0.2, 0.05);
    projected_kernel[(0, 2, 2)] = Complex::new(0.1, -0.2);

    let spectra = xsph_tdlda_channel_spectra(XsphTdldaChannelSpectraInput {
        energy_count: 1,
        matrix_size: 3,
        primary_channel_count: 2,
        channel_count: 4,
        photon_energy: photon_energy.view(),
        plus_wave_number: plus_wave_number.view(),
        minus_wave_number: minus_wave_number.view(),
        initial_kappas: initial_kappas.view(),
        dipole_matrix: dipole_matrix.view(),
        response: response.view(),
        projected_kernel: projected_kernel.view(),
        screened_dipoles: screened_dipoles.view(),
    })?;

    let prefactor = -4.0 * std::f64::consts::PI / XSPH_FINE_STRUCTURE_ALPHA / 2.0
        * XSPH_BOHR_ANGSTROM.powi(2)
        * 100.0;
    let plus_prefactor = -2.0 * 1.5 * prefactor;
    let minus_prefactor = -2.0 * 0.75 * prefactor;
    assert_close_tol(spectra.plus_prefactors[0], plus_prefactor, 1.0e-8);
    assert_close_tol(spectra.minus_prefactors[0], minus_prefactor, 1.0e-8);

    let expected_single = arr2(&[[
        4.0 * plus_prefactor,
        9.0 * minus_prefactor,
        16.0 * plus_prefactor,
        0.0,
    ]]);
    let expected_screened = arr2(&[[
        4.51625 * plus_prefactor,
        8.858_164_062_5 * minus_prefactor,
        15.850_390_625 * plus_prefactor,
        0.0,
    ]]);
    for channel in 0..4 {
        assert_close_tol(
            spectra.single_particle_channels[(0, channel)],
            expected_single[(0, channel)],
            expected_single[(0, channel)].abs() * 1.0e-12 + 1.0e-8,
        );
        assert_close_tol(
            spectra.screened_channels[(0, channel)],
            expected_screened[(0, channel)],
            expected_screened[(0, channel)].abs() * 1.0e-12 + 1.0e-8,
        );
    }

    let two_channel = xsph_tdlda_channel_spectra(XsphTdldaChannelSpectraInput {
        channel_count: 2,
        ..XsphTdldaChannelSpectraInput {
            energy_count: 1,
            matrix_size: 3,
            primary_channel_count: 2,
            channel_count: 4,
            photon_energy: photon_energy.view(),
            plus_wave_number: plus_wave_number.view(),
            minus_wave_number: minus_wave_number.view(),
            initial_kappas: initial_kappas.view(),
            dipole_matrix: dipole_matrix.view(),
            response: response.view(),
            projected_kernel: projected_kernel.view(),
            screened_dipoles: screened_dipoles.view(),
        }
    })?;
    assert_close_tol(
        two_channel.single_particle_channels[(0, 0)],
        expected_single[(0, 0)],
        1.0e-8,
    );
    assert_close_tol(
        two_channel.single_particle_channels[(0, 1)],
        expected_single[(0, 1)],
        1.0e-8,
    );
    assert_eq!(two_channel.single_particle_channels[(0, 2)], 0.0);
    assert_eq!(two_channel.screened_channels[(0, 2)], 0.0);

    Ok(())
}

#[test]
fn xsph_tdlda_channel_spectra_reject_invalid_inputs() {
    let photon_energy = arr1(&[0.0]);
    let wave_number = arr1(&[1.0]);
    let initial_kappas = arr1(&[-1]);
    let dipole_matrix = arr2(&[[1.0]]);
    let response = Array3::from_elem((1, 1, 1), Complex::new(0.0, 0.0));
    let projected_kernel = Array3::from_elem((1, 1, 1), Complex::new(0.0, 0.0));
    let screened_dipoles = arr2(&[[Complex::new(1.0, 0.0)]]);

    let error = xsph_tdlda_channel_spectra(XsphTdldaChannelSpectraInput {
        energy_count: 1,
        matrix_size: 1,
        primary_channel_count: 1,
        channel_count: 1,
        photon_energy: photon_energy.view(),
        plus_wave_number: wave_number.view(),
        minus_wave_number: wave_number.view(),
        initial_kappas: initial_kappas.view(),
        dipole_matrix: dipole_matrix.view(),
        response: response.view(),
        projected_kernel: projected_kernel.view(),
        screened_dipoles: screened_dipoles.view(),
    })
    .expect_err("omega appears in the denominator of FEFF xsectd prefactor");
    assert_eq!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "tdlda_channel_spectra_omega",
            value: 0.0,
        }
    );

    let valid_photon_energy = arr1(&[1.0]);
    let error = xsph_tdlda_channel_spectra(XsphTdldaChannelSpectraInput {
        photon_energy: valid_photon_energy.view(),
        primary_channel_count: 2,
        ..XsphTdldaChannelSpectraInput {
            energy_count: 1,
            matrix_size: 1,
            primary_channel_count: 1,
            channel_count: 1,
            photon_energy: valid_photon_energy.view(),
            plus_wave_number: wave_number.view(),
            minus_wave_number: wave_number.view(),
            initial_kappas: initial_kappas.view(),
            dipole_matrix: dipole_matrix.view(),
            response: response.view(),
            projected_kernel: projected_kernel.view(),
            screened_dipoles: screened_dipoles.view(),
        }
    })
    .expect_err("primary channel split must fit inside matsize");
    assert_eq!(
        error,
        XsphError::SizeOutOfRange {
            name: "tdlda_channel_spectra_primary_channel_count",
            value: 2,
        }
    );

    let error = xsph_tdlda_channel_spectra(XsphTdldaChannelSpectraInput {
        channel_count: 3,
        ..XsphTdldaChannelSpectraInput {
            energy_count: 1,
            matrix_size: 1,
            primary_channel_count: 1,
            channel_count: 1,
            photon_energy: valid_photon_energy.view(),
            plus_wave_number: wave_number.view(),
            minus_wave_number: wave_number.view(),
            initial_kappas: initial_kappas.view(),
            dipole_matrix: dipole_matrix.view(),
            response: response.view(),
            projected_kernel: projected_kernel.view(),
            screened_dipoles: screened_dipoles.view(),
        }
    })
    .expect_err("FEFF xsectd only writes 1, 2, or 4 channel tables");
    assert_eq!(
        error,
        XsphError::SizeOutOfRange {
            name: "tdlda_channel_count",
            value: 3,
        }
    );
}

#[test]
fn xsph_tdlda_broaden_channel_spectra_matches_feff_xsectd_thresholds() -> Result<(), XsphError> {
    let energy_hartree = arr1(&[-0.2, 0.0, 0.3, 0.8]);
    let single_particle_channels = arr2(&[
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ]);
    let screened_channels = arr2(&[
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
        [2.0, 3.0, 4.0, 5.0],
        [2.5, 3.5, 4.5, 5.5],
    ]);
    let edge_energy = 0.0;
    let spin_orbit_split = 0.25;
    let plus_broadening = 0.12;
    let minus_broadening = 0.18;

    let broadened = xsph_tdlda_broaden_channel_spectra(XsphTdldaChannelBroadeningInput {
        energy_count: energy_hartree.len(),
        channel_count: 4,
        energy_hartree: energy_hartree.view(),
        edge_energy,
        spin_orbit_split,
        plus_broadening,
        minus_broadening,
        single_particle_channels: single_particle_channels.view(),
        screened_channels: screened_channels.view(),
    })?;

    let energies = energy_hartree.to_vec();
    for channel in 0..4 {
        let plus_channel = matches!(channel, 0 | 2);
        let threshold = if plus_channel {
            edge_energy
        } else {
            edge_energy + spin_orbit_split
        };
        let width = if plus_channel {
            plus_broadening
        } else {
            minus_broadening
        };
        let expected_single = expected_thresholded_convolution(
            &energies,
            single_particle_channels.view(),
            channel,
            threshold,
            width,
        )?;
        let expected_screened = expected_thresholded_convolution(
            &energies,
            screened_channels.view(),
            channel,
            threshold,
            width,
        )?;

        for energy in 0..energy_hartree.len() {
            assert_close_tol(
                broadened.single_particle_channels[(energy, channel)],
                expected_single[energy],
                expected_single[energy].abs() * 1.0e-12 + 1.0e-10,
            );
            assert_close_tol(
                broadened.screened_channels[(energy, channel)],
                expected_screened[energy],
                expected_screened[energy].abs() * 1.0e-12 + 1.0e-10,
            );
        }
    }

    let two_channel = xsph_tdlda_broaden_channel_spectra(XsphTdldaChannelBroadeningInput {
        channel_count: 2,
        ..XsphTdldaChannelBroadeningInput {
            energy_count: energy_hartree.len(),
            channel_count: 4,
            energy_hartree: energy_hartree.view(),
            edge_energy,
            spin_orbit_split,
            plus_broadening,
            minus_broadening,
            single_particle_channels: single_particle_channels.view(),
            screened_channels: screened_channels.view(),
        }
    })?;
    for energy in 0..energy_hartree.len() {
        assert_close_tol(
            two_channel.single_particle_channels[(energy, 0)],
            broadened.single_particle_channels[(energy, 0)],
            1.0e-10,
        );
        assert_close_tol(
            two_channel.single_particle_channels[(energy, 1)],
            broadened.single_particle_channels[(energy, 1)],
            1.0e-10,
        );
        assert_eq!(two_channel.single_particle_channels[(energy, 2)], 0.0);
        assert_eq!(two_channel.screened_channels[(energy, 2)], 0.0);
        assert_eq!(two_channel.single_particle_channels[(energy, 3)], 0.0);
        assert_eq!(two_channel.screened_channels[(energy, 3)], 0.0);
    }

    Ok(())
}

#[test]
fn xsph_tdlda_broaden_channel_spectra_reject_invalid_inputs() {
    let energy_hartree = arr1(&[0.0, 1.0]);
    let channels = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    let error = xsph_tdlda_broaden_channel_spectra(XsphTdldaChannelBroadeningInput {
        energy_count: 2,
        channel_count: 3,
        energy_hartree: energy_hartree.view(),
        edge_energy: 0.0,
        spin_orbit_split: 0.25,
        plus_broadening: 0.1,
        minus_broadening: 0.2,
        single_particle_channels: channels.view(),
        screened_channels: channels.view(),
    })
    .expect_err("FEFF xsectd only broadens 1, 2, or 4 channel tables");
    assert_eq!(
        error,
        XsphError::SizeOutOfRange {
            name: "tdlda_channel_count",
            value: 3,
        }
    );

    let error = xsph_tdlda_broaden_channel_spectra(XsphTdldaChannelBroadeningInput {
        plus_broadening: 0.0,
        ..XsphTdldaChannelBroadeningInput {
            energy_count: 2,
            channel_count: 1,
            energy_hartree: energy_hartree.view(),
            edge_energy: 0.0,
            spin_orbit_split: 0.25,
            plus_broadening: 0.1,
            minus_broadening: 0.2,
            single_particle_channels: channels.view(),
            screened_channels: channels.view(),
        }
    })
    .expect_err("Lorentzian broadening width must be positive");
    assert_eq!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "tdlda_channel_broadening_plus_width",
            value: 0.0,
        }
    );

    let invalid_energy = arr1(&[Real::NAN, 1.0]);
    assert!(matches!(
        xsph_tdlda_broaden_channel_spectra(XsphTdldaChannelBroadeningInput {
            energy_hartree: invalid_energy.view(),
            ..XsphTdldaChannelBroadeningInput {
                energy_count: 2,
                channel_count: 1,
                energy_hartree: energy_hartree.view(),
                edge_energy: 0.0,
                spin_orbit_split: 0.25,
                plus_broadening: 0.1,
                minus_broadening: 0.2,
                single_particle_channels: channels.view(),
                screened_channels: channels.view(),
            }
        }),
        Err(XsphError::NonFiniteScalar {
            name: "tdlda_channel_broadening_energy",
            ..
        })
    ));
}

fn expected_thresholded_convolution(
    energies: &[Real],
    channels: ArrayView2<'_, Real>,
    channel: usize,
    threshold: Real,
    width: Real,
) -> Result<Vec<Real>, XsphError> {
    let values = energies
        .iter()
        .enumerate()
        .map(|(energy, &energy_hartree)| {
            if energy_hartree < threshold {
                Complex::new(0.0, 0.0)
            } else {
                Complex::new(channels[(energy, channel)], 0.0)
            }
        })
        .collect::<Vec<_>>();
    Ok(crate::conv(energies, &values, width)?
        .iter()
        .map(|value| value.re)
        .collect())
}

fn xmu_channel<'a>(
    photon_energy_ev: ArrayView1<'a, Real>,
    relative_energy_ev: ArrayView1<'a, Real>,
    wave_number: ArrayView1<'a, Real>,
    background: ArrayView1<'a, Real>,
    fine_structure: ArrayView1<'a, Real>,
) -> XsphTdldaXmuChannelInput<'a> {
    XsphTdldaXmuChannelInput {
        point_count: photon_energy_ev.len(),
        photon_energy_ev,
        relative_energy_ev,
        wave_number,
        background,
        fine_structure,
    }
}

#[test]
fn xsph_tdlda_xsedge_rows_match_feff_channel_sums() -> Result<(), XsphError> {
    let energy_hartree = arr1(&[1.0, 2.5]);
    let single_particle_channels = arr2(&[[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]]);
    let screened_channels = arr2(&[[100.0, 200.0, 300.0, 400.0], [5.0, 6.0, 7.0, 8.0]]);
    let channel_multipliers = arr2(&[[1.0, 0.5, 2.0, 0.25], [2.0, 3.0, 4.0, 5.0]]);

    let four_channel = xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
        energy_count: energy_hartree.len(),
        channel_count: 4,
        energy_hartree: energy_hartree.view(),
        single_particle_channels: single_particle_channels.view(),
        screened_channels: screened_channels.view(),
        channel_multipliers: channel_multipliers.view(),
    })?;

    for (actual, expected) in four_channel
        .energy_ev
        .iter()
        .zip([XSPH_HARTREE_EV, 2.5 * XSPH_HARTREE_EV])
    {
        assert_close(*actual, expected);
    }
    assert_eq!(four_channel.total_single_particle, arr1(&[90.0, 40.0]));
    assert_eq!(four_channel.total_screened, arr1(&[900.0, 96.0]));
    assert_eq!(
        four_channel.plus_branch_single_particle,
        arr1(&[70.0, 14.0])
    );
    assert_eq!(
        four_channel.minus_branch_single_particle,
        arr1(&[20.0, 26.0])
    );
    assert_eq!(four_channel.plus_branch_screened, arr1(&[700.0, 38.0]));
    assert_eq!(four_channel.minus_branch_screened, arr1(&[200.0, 58.0]));

    let two_channel = xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
        channel_count: 2,
        ..XsphTdldaXsedgeRowsInput {
            energy_count: energy_hartree.len(),
            channel_count: 4,
            energy_hartree: energy_hartree.view(),
            single_particle_channels: single_particle_channels.view(),
            screened_channels: screened_channels.view(),
            channel_multipliers: channel_multipliers.view(),
        }
    })?;
    assert_eq!(two_channel.total_single_particle, arr1(&[20.0, 8.0]));
    assert_eq!(two_channel.total_screened, arr1(&[200.0, 28.0]));
    assert_eq!(two_channel.plus_branch_single_particle, arr1(&[10.0, 2.0]));
    assert_eq!(two_channel.minus_branch_single_particle, arr1(&[10.0, 6.0]));

    let one_channel = xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
        channel_count: 1,
        ..XsphTdldaXsedgeRowsInput {
            energy_count: energy_hartree.len(),
            channel_count: 4,
            energy_hartree: energy_hartree.view(),
            single_particle_channels: single_particle_channels.view(),
            screened_channels: screened_channels.view(),
            channel_multipliers: channel_multipliers.view(),
        }
    })?;
    assert_eq!(one_channel.total_single_particle, arr1(&[10.0, 2.0]));
    assert_eq!(one_channel.total_screened, arr1(&[100.0, 10.0]));
    assert_eq!(
        one_channel.plus_branch_single_particle,
        one_channel.total_single_particle
    );
    assert_eq!(
        one_channel.minus_branch_single_particle,
        Array1::<Real>::zeros(energy_hartree.len())
    );

    Ok(())
}

#[test]
fn xsph_tdlda_xsedge_rows_reject_invalid_inputs() {
    let energy_hartree = arr1(&[1.0]);
    let channels = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
    let error = xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
        energy_count: 1,
        channel_count: 3,
        energy_hartree: energy_hartree.view(),
        single_particle_channels: channels.view(),
        screened_channels: channels.view(),
        channel_multipliers: channels.view(),
    })
    .expect_err("FEFF xsectd only writes 1, 2, or 4 channel tables");
    assert_eq!(
        error,
        XsphError::SizeOutOfRange {
            name: "tdlda_xsedge_channel_count",
            value: 3,
        }
    );

    let empty = arr1(&[]);
    let empty_channels = Array2::<Real>::zeros((0, 4));
    assert_eq!(
        xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
            energy_count: 0,
            channel_count: 1,
            energy_hartree: empty.view(),
            single_particle_channels: empty_channels.view(),
            screened_channels: empty_channels.view(),
            channel_multipliers: empty_channels.view(),
        }),
        Err(XsphError::EmptyIndexSet)
    );

    let invalid_energy = arr1(&[Real::NAN]);
    assert!(matches!(
        xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
            energy_count: 1,
            channel_count: 1,
            energy_hartree: invalid_energy.view(),
            single_particle_channels: channels.view(),
            screened_channels: channels.view(),
            channel_multipliers: channels.view(),
        }),
        Err(XsphError::NonFiniteScalar {
            name: "tdlda_xsedge_energy",
            ..
        })
    ));

    let short_channels = arr2(&[[1.0, 2.0, 3.0]]);
    assert!(matches!(
        xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
            energy_count: 1,
            channel_count: 1,
            energy_hartree: energy_hartree.view(),
            single_particle_channels: short_channels.view(),
            screened_channels: channels.view(),
            channel_multipliers: channels.view(),
        }),
        Err(XsphError::LengthTooShort {
            name: "tdlda_xsedge_single_particle_channels",
            required: 4,
            actual: 3,
        })
    ));
}

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
