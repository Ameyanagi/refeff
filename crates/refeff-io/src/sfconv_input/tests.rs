use crate::{
    EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh, FeffDocument, FeffInput,
    IoError, ListDatData, ListDatEntry, rdinp,
};
use refeff_core::{SfconvExafsConvolution, SfconvPathAverage, SfconvXanesConvolution};

use super::{
    SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvInput, SfconvSo2convTarget, SfconvSo2convTargetData,
    SfconvSo2convTargetKind, sfconv_input_string, sfconv_so2conv_chi_data_from_convolution_rows,
    sfconv_so2conv_feff_path_data_from_averages, sfconv_so2conv_header_from_text,
    sfconv_so2conv_material_input_from_header, sfconv_so2conv_target_data_from_text,
    sfconv_so2conv_target_data_string, sfconv_so2conv_targets,
    sfconv_so2conv_xmu_data_from_convolution_rows,
};

#[test]
fn parses_generated_sfconv_input() -> crate::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SFCONV
XANES
PRINT 0 0 0 0 0 3
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let sfconv = SfconvInput::parse_str("sfconv.inp", &rdinp::sfconv_inp_string(&document)?)?;

    assert_eq!(sfconv.control.msfconv, 1);
    assert_eq!(sfconv.control.ipse, 0);
    assert_eq!(sfconv.control.ipsk, 0);
    assert_eq!(sfconv.window.wsigk, 0.0);
    assert_eq!(sfconv.window.cen, 0.0);
    assert_eq!(sfconv.spectrum.ispec, 1);
    assert_eq!(sfconv.spectrum.ipr6, 3);
    assert_eq!(sfconv.cfname, "NULL");
    Ok(())
}

#[test]
fn renders_generated_sfconv_input() -> crate::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SFCONV
XANES
PRINT 0 0 0 0 0 3
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let expected = rdinp::sfconv_inp_string(&document)?;
    let sfconv = SfconvInput::parse_str("sfconv.inp", &expected)?;
    let rendered = sfconv_input_string(&sfconv)?;
    let reparsed = SfconvInput::parse_str("sfconv.inp", &rendered)?;

    assert_eq!(rendered, expected);
    assert_eq!(reparsed, sfconv);
    Ok(())
}

#[test]
fn rejects_invalid_sfconv_rendering() {
    let input = SfconvInput {
        control: super::SfconvControl {
            msfconv: 1,
            ipse: 0,
            ipsk: 0,
        },
        window: super::SfconvWindow {
            wsigk: f64::NAN,
            cen: 0.0,
        },
        spectrum: super::SfconvSpectrum { ispec: 1, ipr6: 0 },
        cfname: "NULL".to_string(),
    };
    assert!(sfconv_input_string(&input).is_err());
}

#[test]
fn so2conv_targets_match_concrete_feff_reference_exafs_branches() -> crate::Result<()> {
    let list = so2conv_target_list();

    let base = sfconv_so2conv_targets(&sfconv_target_input(0, 1), None, None)?;
    assert_eq!(
        base,
        vec![
            target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            target("chi.dat", SfconvSo2convTargetKind::Chi),
        ]
    );

    let chip = sfconv_so2conv_targets(&sfconv_target_input(0, 2), Some(&list), None)?;
    assert_eq!(
        chip,
        vec![
            target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            target("chi.dat", SfconvSo2convTargetKind::Chi),
            target("chip0001.dat", SfconvSo2convTargetKind::Chi),
            target("chip0023.dat", SfconvSo2convTargetKind::Chi),
            target("chip4567.dat", SfconvSo2convTargetKind::Chi),
        ]
    );

    let feff = sfconv_so2conv_targets(&sfconv_target_input(0, 3), Some(&list), None)?;
    assert_eq!(
        feff,
        vec![
            target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            target("chi.dat", SfconvSo2convTargetKind::Chi),
            target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
            target("feff0023.dat", SfconvSo2convTargetKind::FeffPath),
            target("feff4567.dat", SfconvSo2convTargetKind::FeffPath),
        ]
    );
    Ok(())
}

#[test]
fn so2conv_targets_match_feff_reference_elnes_polarizations() -> crate::Result<()> {
    let eels = eels_target_input(true, 1, 4, 9);
    let targets = sfconv_so2conv_targets(&sfconv_target_input(1, 3), None, Some(&eels))?;

    assert_eq!(
        targets,
        vec![
            target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            target("chi.dat", SfconvSo2convTargetKind::Chi),
            target("xmu05.dat", SfconvSo2convTargetKind::Xmu),
            target("chi05.d", SfconvSo2convTargetKind::Chi),
            target("xmu09.dat", SfconvSo2convTargetKind::Xmu),
            target("chi09.d", SfconvSo2convTargetKind::Chi),
        ]
    );
    Ok(())
}

#[test]
fn so2conv_targets_reject_missing_list_and_invalid_controls() {
    let missing_list = sfconv_so2conv_targets(&sfconv_target_input(0, 2), None, None).err();
    assert!(missing_list.is_some_and(|error| {
        error
            .to_string()
            .contains("requires list.dat when ipr6 >= 2")
    }));

    let invalid_ispec = sfconv_so2conv_targets(&sfconv_target_input(2, 0), None, None).err();
    assert!(
        invalid_ispec.is_some_and(|error| error.to_string().contains("ispec must be -1, 0, or 1"))
    );

    let bad_step = eels_target_input(true, 1, 0, 9);
    let invalid_range =
        sfconv_so2conv_targets(&sfconv_target_input(1, 0), None, Some(&bad_step)).err();
    assert!(invalid_range.is_some_and(|error| {
        error
            .to_string()
            .contains("EELS polarization step must be nonzero")
    }));
}

#[test]
fn parses_so2conv_material_header_like_feff_reference() -> crate::Result<()> {
    let header = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "  1.0 2.0 3.0\n",
    );

    let material = sfconv_so2conv_material_input_from_header("xmu.dat", header)?;

    assert_eq!(material.core_hole_width_ev, 1.729);
    assert_eq!(material.wigner_seitz_radius, 2.05);
    assert_eq!(material.interstitial_potential_ev, 12.34);
    assert_eq!(material.chemical_potential_ev, 18.76);
    assert_eq!(material.fermi_wave_number_inv_angstrom, 1.23);
    Ok(())
}

#[test]
fn so2conv_material_header_uses_last_value_before_separator_like_feff() -> crate::Result<()> {
    let header = concat!(
        "# one Gam_ch= 4.000000 Rs_int= 9.99 Vint= 99.00000 ",
        "Mu= 88.00000 kf= 7.000000\n",
        "# two Gam_ch= 5.533000 Rs_int= 1.42 Vint=-3.250000 ",
        "Mu=  0.80000 kf= 0.780000\n",
        " ------------------------------------------------------------------------------\n",
        "# later Gam_ch= 9.000000 Rs_int= 8.88 Vint= 77.00000 ",
        "Mu= 66.00000 kf= 5.000000\n",
    );

    let material = sfconv_so2conv_material_input_from_header("chip0001.dat", header)?;

    assert_eq!(material.core_hole_width_ev, 5.533);
    assert_eq!(material.wigner_seitz_radius, 1.42);
    assert_eq!(material.interstitial_potential_ev, -3.25);
    assert_eq!(material.chemical_potential_ev, 0.8);
    assert_eq!(material.fermi_wave_number_inv_angstrom, 0.78);
    Ok(())
}

#[test]
fn so2conv_material_header_rejects_missing_or_invalid_values() {
    let missing = "# Header Gam_ch= 1.729000 Rs_int= 2.05\n";
    let missing_error = sfconv_so2conv_material_input_from_header("chi.dat", missing).err();
    assert!(missing_error.is_some_and(|error| {
        error
            .to_string()
            .contains("missing SO2CONV header field Vint")
    }));

    let invalid = concat!(
        "# Header Gam_ch= not_real Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
    );
    let invalid_error = sfconv_so2conv_material_input_from_header("chi.dat", invalid).err();
    assert!(invalid_error.is_some_and(|error| {
        error
            .to_string()
            .contains("invalid SO2CONV header field Gam_ch")
    }));
}

#[test]
fn so2conv_header_detects_previous_convolution_before_separator_like_feff() -> crate::Result<()> {
    let before = format!(
        concat!(
            "{}                                                     \n",
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
        ),
        SFCONV_SO2CONV_CONVOLUTED_MARKER
    );
    let before_scan = sfconv_so2conv_header_from_text("xmu.dat", &before)?;
    assert!(before_scan.already_convoluted);
    assert_eq!(before_scan.material.core_hole_width_ev, 1.729);

    let after = format!(
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "{}\n",
        ),
        SFCONV_SO2CONV_CONVOLUTED_MARKER
    );
    let after_scan = sfconv_so2conv_header_from_text("xmu.dat", &after)?;
    assert!(!after_scan.already_convoluted);
    assert_eq!(after_scan.material.fermi_wave_number_inv_angstrom, 1.23);
    Ok(())
}

#[test]
fn parses_so2conv_xmu_target_data() -> crate::Result<()> {
    let text = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
    );
    let parsed = sfconv_so2conv_target_data_from_text(
        "xmu.dat",
        &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
        text,
    )?;

    match parsed {
        SfconvSo2convTargetData::Xmu { header, data } => {
            assert_eq!(header.material.core_hole_width_ev, 1.729);
            assert_eq!(data.point_count(), 1);
            assert_eq!(data.mu[0], 1.0);
            assert_eq!(data.chi[0], 0.1);
        }
        _ => return Err(wrong_target_variant("xmu target")),
    }
    Ok(())
}

#[test]
fn parses_so2conv_chi_target_data() -> crate::Result<()> {
    let text = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "  0.5000  1.10000E-01 2.20000E+00 -3.30000E-01\n",
        "  0.6000  1.20000E-01 2.30000E+00 -3.40000E-01\n",
    );
    let parsed = sfconv_so2conv_target_data_from_text(
        "chi.dat",
        &target("chi.dat", SfconvSo2convTargetKind::Chi),
        text,
    )?;

    match parsed {
        SfconvSo2convTargetData::Chi { header, data } => {
            assert_eq!(header.material.fermi_wave_number_inv_angstrom, 1.23);
            assert_eq!(data.point_count(), 2);
            assert_eq!(data.wave_number[1], 0.6);
            assert_eq!(data.phase[1], -0.34);
        }
        _ => return Err(wrong_target_variant("chi target")),
    }
    Ok(())
}

#[test]
fn parses_so2conv_feff_path_target_data_like_feff_reference() -> crate::Result<()> {
    let text = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "#    3   4.250   2.7500 reff path metadata\n",
        "#       k          phase @#\n",
        "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
    );
    let parsed = sfconv_so2conv_target_data_from_text(
        "feff0001.dat",
        &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
        text,
    )?;

    match parsed {
        SfconvSo2convTargetData::FeffPath { header, data } => {
            assert_eq!(header.material.interstitial_potential_ev, 12.34);
            assert_eq!(data.leg_count, 3);
            assert_eq!(data.degeneracy, 4.25);
            assert_eq!(data.effective_half_path_length_angstrom, 2.75);
            assert_eq!(data.point_count(), 1);
            assert_eq!(data.wave_number_inverse_angstrom[0], 0.5);
            assert_eq!(data.central_phase[0], 0.11);
            assert_eq!(data.effective_amplitude[0], 2.2);
            assert_eq!(data.effective_phase[0], -0.33);
            assert_eq!(data.reduction_factor[0], 0.88);
            assert_eq!(data.mean_free_path_angstrom[0], 4.4);
            assert_eq!(data.real_momentum_inverse_angstrom[0], 0.66);
        }
        _ => return Err(wrong_target_variant("feff path target")),
    }
    Ok(())
}

#[test]
fn rejects_bad_so2conv_feff_path_target_data() {
    let missing_reff = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "#       k          phase @#\n",
        "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
    );
    let missing_error = sfconv_so2conv_target_data_from_text(
        "feff0001.dat",
        &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
        missing_reff,
    )
    .err();
    assert!(missing_error.is_some_and(|error| {
        error
            .to_string()
            .contains("missing SO2CONV feff path reff leg count")
    }));

    let short_row = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "#    3   4.250   2.7500 reff path metadata\n",
        "#       k          phase @#\n",
        "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01\n",
    );
    let row_error = sfconv_so2conv_target_data_from_text(
        "feff0001.dat",
        &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
        short_row,
    )
    .err();
    assert!(row_error.is_some_and(|error| {
        error
            .to_string()
            .contains("SO2CONV feff path row requires 7 fields")
    }));
}

#[test]
fn renders_so2conv_feff_path_target_data_like_feff_reference() -> crate::Result<()> {
    let parsed = sfconv_so2conv_target_data_from_text(
        "feff0001.dat",
        &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#    3   4.250   2.7500 reff path metadata\n",
            "#       k          phase @#\n",
            "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
        ),
    )?;

    let rendered = sfconv_so2conv_target_data_string(&parsed)?;

    assert_eq!(
        rendered,
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#    3   4.250   2.7500 reff path metadata\n",
            "#       k          phase @#\n",
            "  0.500  1.1000E-01  2.2000E+00 -3.3000E-01  8.800E-01  4.4000E+00  6.6000E-01 \n",
        )
    );
    assert_eq!(
        sfconv_so2conv_target_data_from_text(
            "feff0001.dat",
            &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
            &rendered,
        )?,
        parsed
    );
    Ok(())
}

#[test]
fn renders_so2conv_convoluted_target_marker() -> crate::Result<()> {
    let parsed = sfconv_so2conv_target_data_from_text(
        "xmu.dat",
        &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
        ),
    )?;

    let rendered = super::sfconv_so2conv_convoluted_target_data_string(&parsed)?;

    assert!(rendered.starts_with(SFCONV_SO2CONV_CONVOLUTED_MARKER));
    assert_eq!(
        super::sfconv_so2conv_convoluted_target_data_string(
            &sfconv_so2conv_target_data_from_text(
                "xmu.dat",
                &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
                &rendered,
            )?
        )?,
        rendered
    );
    Ok(())
}

#[test]
fn writes_so2conv_convoluted_target_with_marker() -> crate::Result<()> {
    let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
    let path = temp.path().join("xmu.dat");
    let parsed = sfconv_so2conv_target_data_from_text(
        "xmu.dat",
        &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
        ),
    )?;

    super::write_sfconv_so2conv_convoluted_target_data(&path, &parsed)?;

    let rendered = std::fs::read_to_string(&path).map_err(|source| IoError::io(&path, source))?;
    assert_eq!(
        rendered.lines().next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    assert!(
        sfconv_so2conv_target_data_from_text(
            &path,
            &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            &rendered,
        )?
        .header()
        .already_convoluted
    );
    Ok(())
}

#[test]
fn builds_so2conv_xmu_target_data_from_convolution_rows() -> crate::Result<()> {
    let parsed = sfconv_so2conv_target_data_from_text(
        "xmu.dat",
        &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
            "  105.000 5.000 1.000 1.10000E+00 9.50000E-01 1.50000E-01\n",
        ),
    )?;
    let source = match parsed {
        SfconvSo2convTargetData::Xmu { data, .. } => data,
        _ => return Err(wrong_target_variant("xmu target")),
    };

    let output = sfconv_so2conv_xmu_data_from_convolution_rows(
        &source,
        &[
            SfconvXanesConvolution {
                absorption: 1.25,
                embedded_background: 0.80,
                fine_structure: 0.45,
            },
            SfconvXanesConvolution {
                absorption: 1.50,
                embedded_background: 0.90,
                fine_structure: 0.60,
            },
        ],
    )?;

    assert_eq!(output.photon_energy_ev, source.photon_energy_ev);
    assert_eq!(output.relative_energy_ev, source.relative_energy_ev);
    assert_eq!(output.wave_number, source.wave_number);
    assert_eq!(output.mu.to_vec(), vec![1.25, 1.50]);
    assert_eq!(output.mu0.to_vec(), vec![0.80, 0.90]);
    assert_eq!(output.chi.to_vec(), vec![0.45, 0.60]);
    Ok(())
}

#[test]
fn builds_so2conv_chi_target_data_from_convolution_rows() -> crate::Result<()> {
    let parsed = sfconv_so2conv_target_data_from_text(
        "chi.dat",
        &target("chi.dat", SfconvSo2convTargetKind::Chi),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  0.500  1.00000E-01  2.00000E+00  3.00000E-01\n",
            "  0.550  2.00000E-01  2.10000E+00  4.00000E-01\n",
        ),
    )?;
    let source = match parsed {
        SfconvSo2convTargetData::Chi { data, .. } => data,
        _ => return Err(wrong_target_variant("chi target")),
    };

    let output = sfconv_so2conv_chi_data_from_convolution_rows(
        &source,
        &[
            SfconvExafsConvolution {
                real: 0.70,
                imaginary: 0.11,
                magnitude: 0.71,
                output_phase: 0.12,
                output_phase_minus_original: 0.02,
                amplitude_reduction: 0.90,
                phase_shift: 0.01,
                previous_phase: 0.12,
                phase_jump_count: 0,
            },
            SfconvExafsConvolution {
                real: 0.80,
                imaginary: 0.22,
                magnitude: 0.83,
                output_phase: 0.24,
                output_phase_minus_original: 0.04,
                amplitude_reduction: 0.95,
                phase_shift: 0.02,
                previous_phase: 0.24,
                phase_jump_count: 0,
            },
        ],
    )?;

    assert_eq!(output.wave_number, source.wave_number);
    assert_eq!(output.chi.to_vec(), vec![0.11, 0.22]);
    assert_eq!(output.magnitude.to_vec(), vec![0.71, 0.83]);
    assert_eq!(output.phase.to_vec(), vec![0.12, 0.24]);
    assert_eq!(
        output
            .phase_minus_2kr
            .as_ref()
            .map(|values| values.to_vec()),
        Some(vec![0.02, 0.04])
    );
    assert!(output.ckp_real.is_none());
    assert!(output.ckp_imag.is_none());
    Ok(())
}

#[test]
fn builds_so2conv_feff_path_data_from_path_averages() -> crate::Result<()> {
    let parsed = sfconv_so2conv_target_data_from_text(
        "feff0001.dat",
        &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#    3   4.250   2.7500 reff path metadata\n",
            "#       k          phase @#\n",
            "  0.500  1.00000E-01  2.20000E+00 -3.30000E-01  8.00000E-01  4.40000E+00  6.60000E-01\n",
            "  0.550  2.00000E-01  2.30000E+00 -3.10000E-01  7.00000E-01  4.50000E+00  6.70000E-01\n",
        ),
    )?;
    let source = match parsed {
        SfconvSo2convTargetData::FeffPath { data, .. } => data,
        _ => return Err(wrong_target_variant("feff path target")),
    };

    let output = sfconv_so2conv_feff_path_data_from_averages(
        &source,
        &[
            SfconvPathAverage {
                amplitude_reduction: 0.50,
                phase_shift: 0.03,
                normalization: 0.05,
            },
            SfconvPathAverage {
                amplitude_reduction: 0.25,
                phase_shift: -0.02,
                normalization: 0.05,
            },
        ],
    )?;

    assert_eq!(
        output.wave_number_inverse_angstrom,
        source.wave_number_inverse_angstrom
    );
    assert_close(output.central_phase[0], 0.13, 1.0e-12);
    assert_close(output.central_phase[1], 0.18, 1.0e-12);
    assert_close(output.reduction_factor[0], 0.40, 1.0e-12);
    assert_close(output.reduction_factor[1], 0.175, 1.0e-12);
    assert_eq!(output.effective_amplitude, source.effective_amplitude);
    assert_eq!(output.effective_phase, source.effective_phase);
    Ok(())
}

#[test]
fn rejects_so2conv_result_row_count_mismatches() -> crate::Result<()> {
    let xmu = match sfconv_so2conv_target_data_from_text(
        "xmu.dat",
        &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
        ),
    )? {
        SfconvSo2convTargetData::Xmu { data, .. } => data,
        _ => return Err(wrong_target_variant("xmu target")),
    };
    assert!(matches!(
        sfconv_so2conv_xmu_data_from_convolution_rows(&xmu, &[]),
        Err(IoError::XmuDatShape {
            field: "SO2CONV XANES rows",
            actual: 0,
            expected: 1
        })
    ));

    let chi = match sfconv_so2conv_target_data_from_text(
        "chi.dat",
        &target("chi.dat", SfconvSo2convTargetKind::Chi),
        concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  0.500  1.00000E-01  2.00000E+00  3.00000E-01\n",
        ),
    )? {
        SfconvSo2convTargetData::Chi { data, .. } => data,
        _ => return Err(wrong_target_variant("chi target")),
    };
    assert!(matches!(
        sfconv_so2conv_chi_data_from_convolution_rows(&chi, &[]),
        Err(IoError::ChiDatShape {
            field: "SO2CONV EXAFS rows",
            actual: 0,
            expected: 1
        })
    ));

    Ok(())
}

fn sfconv_target_input(ispec: i32, ipr6: i32) -> SfconvInput {
    SfconvInput {
        control: super::SfconvControl {
            msfconv: 1,
            ipse: 0,
            ipsk: 0,
        },
        window: super::SfconvWindow {
            wsigk: 0.0,
            cen: 0.0,
        },
        spectrum: super::SfconvSpectrum { ispec, ipr6 },
        cfname: "NULL".to_string(),
    }
}

fn so2conv_target_list() -> ListDatData {
    ListDatData {
        titles: vec!["target oracle".to_string()],
        entries: [1, 23, 4567]
            .into_iter()
            .map(|path_index| ListDatEntry {
                path_index,
                sigma2: 0.0,
                amplitude_ratio: 1.0,
                degeneracy: 4.0,
                leg_count: 2,
                effective_half_path_length_angstrom: 2.5,
            })
            .collect(),
    }
}

fn eels_target_input(calculate_elnes: bool, min: i32, step: i32, max: i32) -> EelsInput {
    EelsInput {
        calculate_elnes,
        calculation_mode: i32::from(calculate_elnes),
        control: EelsControl {
            average: 0,
            relativistic: 0,
            cross_terms: 0,
            input: 1,
            spectrum_column: 0,
        },
        polarization: EelsPolarization { min, step, max },
        beam_energy: 200.0,
        beam_direction: [0.0, 0.0, 1.0],
        angles: EelsAngles {
            collection: 0.0,
            convergence: 0.0,
        },
        qmesh: EelsQMesh {
            radial: 1,
            angular: 1,
        },
        detector: [0.0, 0.0],
        magic: 0,
        magic_energy: 0.0,
    }
}

fn target(file_name: &str, kind: SfconvSo2convTargetKind) -> SfconvSo2convTarget {
    SfconvSo2convTarget {
        file_name: file_name.to_string(),
        kind,
    }
}

fn wrong_target_variant(context: &'static str) -> IoError {
    IoError::Parse {
        path: "test".into(),
        line: 0,
        message: format!("wrong SO2CONV target variant for {context}"),
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual} differs from expected {expected} by more than {tolerance}"
    );
}
