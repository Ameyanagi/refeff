use super::{support::*, *};

#[test]
fn fine_structure_from_segments_matches_feff_rdst_reference_algorithm()
-> Result<(), FullSpectrumError> {
    let omega = array![
        10.0 / FEFF_HARTREE_EV,
        12.5 / FEFF_HARTREE_EV,
        15.0 / FEFF_HARTREE_EV,
        20.0 / FEFF_HARTREE_EV,
    ];
    let fms_energy_ev = array![5.0, 10.0, 15.0];
    let fms_wave_number = array![1.0, 3.0, 4.0];
    let real_fms = array![1.0, 2.0, 3.0];
    let real_fms_background = array![0.5, 1.0, 1.5];
    let imaginary_fms = array![0.0, 2.0, 4.0];
    let imaginary_fms_background = array![0.0, 1.0, 2.0];
    let path_energy_ev = array![10.0, 20.0, 30.0];
    let path_wave_number = array![3.0, 5.0, 7.0];
    let real_path = array![10.0, 20.0, 30.0];
    let real_path_background = array![5.0, 10.0, 15.0];
    let imaginary_path = array![20.0, 40.0, 60.0];
    let imaginary_path_background = array![10.0, 20.0, 30.0];

    let fine_structure =
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: real_fms.view(),
                background: real_fms_background.view(),
            },
            real_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: real_path.view(),
                background: real_path_background.view(),
            },
            imaginary_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: imaginary_fms.view(),
                background: imaginary_fms_background.view(),
            },
            imaginary_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: imaginary_path.view(),
                background: imaginary_path_background.view(),
            },
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        })?;

    assert_eq!(fine_structure.point_count(), 4);
    assert_close(
        fine_structure.real_transition_interval[0],
        10.0 / FEFF_HARTREE_EV,
        1.0e-14,
    );
    assert_close(
        fine_structure.real_transition_interval[1],
        15.0 / FEFF_HARTREE_EV,
        1.0e-14,
    );
    assert_close(fine_structure.scattering_factor[0].re, 2.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[0].im, 2.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[1].re, 7.5, 1.0e-14);
    assert_close(fine_structure.scattering_factor[1].im, 14.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[2].re, 15.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[2].im, 30.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[3].re, 20.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[3].im, 40.0, 1.0e-14);
    assert_close(fine_structure.background[1].re, 3.75, 1.0e-14);
    assert_close(fine_structure.background[1].im, 7.0, 1.0e-14);
    assert_close(fine_structure.background[3].re, 10.0, 1.0e-14);
    assert_close(fine_structure.background[3].im, 20.0, 1.0e-14);
    Ok(())
}

#[test]
fn fine_structure_from_segments_clamps_negative_imaginary_parts() -> Result<(), FullSpectrumError> {
    let omega = array![12.5 / FEFF_HARTREE_EV];
    let fms_energy_ev = array![5.0, 10.0, 15.0];
    let fms_wave_number = array![1.0, 3.0, 4.0];
    let real = array![1.0, 2.0, 3.0];
    let real_background = array![0.5, 1.0, 1.5];
    let imaginary_fms = array![0.0, -2.0, -4.0];
    let imaginary_fms_background = array![0.0, -1.0, -2.0];
    let path_energy_ev = array![10.0, 20.0, 30.0];
    let path_wave_number = array![3.0, 5.0, 7.0];
    let imaginary_path = array![-20.0, -40.0, -60.0];
    let imaginary_path_background = array![-10.0, -20.0, -30.0];

    let fine_structure =
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: real.view(),
                background: real_background.view(),
            },
            real_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: real.view(),
                background: real_background.view(),
            },
            imaginary_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: imaginary_fms.view(),
                background: imaginary_fms_background.view(),
            },
            imaginary_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: imaginary_path.view(),
                background: imaginary_path_background.view(),
            },
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        })?;

    assert_close(fine_structure.scattering_factor[0].im, 0.0, 0.0);
    assert_close(fine_structure.background[0].im, 0.0, 0.0);
    Ok(())
}

#[test]
fn fine_structure_from_segments_rejects_invalid_inputs() {
    let omega = array![10.0 / FEFF_HARTREE_EV];
    let energy_ev = array![5.0, 10.0, 15.0];
    let wave_number = array![1.0, 3.0, 4.0];
    let signal = array![1.0, 2.0, 3.0];
    let background = array![0.5, 1.0, 1.5];
    let valid_segment = FullSpectrumFineStructureSegmentInput {
        photon_energy_ev: energy_ev.view(),
        wave_number_inverse_angstrom: wave_number.view(),
        scattering_factor: signal.view(),
        background: background.view(),
    };
    let empty_omega = Array1::<Real>::zeros(0);

    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: empty_omega.view(),
            real_fms: valid_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        }),
        Err(FullSpectrumError::EmptyTable { name: "omega" })
    ));
    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: valid_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 4.0,
            high_wave_number: 3.0,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "fine_structure_wave_number",
            ..
        })
    ));

    let too_short_energy = array![5.0];
    let too_short_wave_number = array![1.0];
    let too_short_signal = array![1.0];
    let too_short_background = array![0.5];
    let too_short_segment = FullSpectrumFineStructureSegmentInput {
        photon_energy_ev: too_short_energy.view(),
        wave_number_inverse_angstrom: too_short_wave_number.view(),
        scattering_factor: too_short_signal.view(),
        background: too_short_background.view(),
    };
    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: too_short_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        }),
        Err(FullSpectrumError::SegmentTooShort {
            name: "fine_structure",
            segment: 0,
            len: 1
        })
    ));

    let high_wave_number_only = array![5.0, 6.0, 7.0];
    let missing_transition_segment = FullSpectrumFineStructureSegmentInput {
        photon_energy_ev: energy_ev.view(),
        wave_number_inverse_angstrom: high_wave_number_only.view(),
        scattering_factor: signal.view(),
        background: background.view(),
    };
    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: missing_transition_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        }),
        Err(FullSpectrumError::MissingTransitionThreshold {
            name: "real_fms",
            ..
        })
    ));
}
