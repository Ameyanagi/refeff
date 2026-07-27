use super::*;

#[test]
fn parses_generated_reference_spectrum_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated spectrum parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut xmu_spectra = Vec::new();
    collect_named_files(&golden_dir, "referencexmu.dat", &mut xmu_spectra)?;
    collect_named_files(&golden_dir, "reference_xmu.dat", &mut xmu_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut xmu_spectra, &is_xmu_spectrum_name)?;
    xmu_spectra.sort();

    let mut axafs_spectra = Vec::new();
    collect_matching_nonempty_files(&golden_dir, &mut axafs_spectra, &is_axafs_spectrum_name)?;
    axafs_spectra.sort();

    let mut chi_spectra = Vec::new();
    collect_named_files(&golden_dir, "referencechi.dat", &mut chi_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut chi_spectra, &is_chi_spectrum_name)?;
    chi_spectra.sort();

    let mut eels_spectra = Vec::new();
    collect_named_files(&golden_dir, "reference_eels.dat", &mut eels_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut eels_spectra, &|name| name == "eels.dat")?;
    eels_spectra.sort();

    let mut danes_spectra = Vec::new();
    collect_named_files(&golden_dir, "referencedanes.dat", &mut danes_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut danes_spectra, &|name| name == "danes.dat")?;
    danes_spectra.sort();

    let mut ldos_spectra = Vec::new();
    collect_named_files(&golden_dir, "referenceldos00.dat", &mut ldos_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut ldos_spectra, &is_ldos_spectrum_name)?;
    ldos_spectra.sort();

    let mut rhoc_spectra = Vec::new();
    collect_matching_nonempty_files(&golden_dir, &mut rhoc_spectra, &is_rhoc_spectrum_name)?;
    rhoc_spectra.sort();

    let mut compton_spectra = Vec::new();
    collect_named_files(&golden_dir, "reference_compton.dat", &mut compton_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut compton_spectra, &|name| {
        name == "compton.dat"
    })?;
    compton_spectra.sort();

    let mut rhozzp_spectra = Vec::new();
    collect_named_files(&golden_dir, "reference_rhozzp.dat", &mut rhozzp_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut rhozzp_spectra, &|name| {
        name == "rhozzp.dat"
    })?;
    rhozzp_spectra.sort();

    let mut crpa_spectra = Vec::new();
    collect_named_files(&golden_dir, "referencecrpa.dat", &mut crpa_spectra)?;
    collect_matching_nonempty_files(&golden_dir, &mut crpa_spectra, &|name| name == "crpa.dat")?;
    crpa_spectra.sort();

    let mut loss_spectra = Vec::new();
    collect_named_files(&golden_dir, "loss.dat", &mut loss_spectra)?;
    loss_spectra.sort();

    let mut eps_tables = Vec::new();
    collect_named_files(&golden_dir, "eps.dat", &mut eps_tables)?;
    eps_tables.sort();

    let mut osc_str_tables = Vec::new();
    collect_named_files(&golden_dir, "osc_str.dat", &mut osc_str_tables)?;
    osc_str_tables.sort();

    let mut opcons_tables = Vec::new();
    for name in ["opcons.dat", "opconsKK.dat", "opcons0.dat"] {
        collect_named_files(&golden_dir, name, &mut opcons_tables)?;
    }
    opcons_tables.sort();

    let mut sumrules_tables = Vec::new();
    collect_named_files(&golden_dir, "sumrules.dat", &mut sumrules_tables)?;
    sumrules_tables.sort();

    let mut drude_tables = Vec::new();
    collect_named_files(&golden_dir, "drude.dat", &mut drude_tables)?;
    drude_tables.sort();

    let mut hamaker_tables = Vec::new();
    collect_named_files(&golden_dir, "hamaker.dat", &mut hamaker_tables)?;
    hamaker_tables.sort();

    let mut exc_tables = Vec::new();
    collect_named_files(&golden_dir, "exc.dat", &mut exc_tables)?;
    exc_tables.sort();

    let mut mpse_spectra = Vec::new();
    collect_named_files(&golden_dir, "mpse.dat", &mut mpse_spectra)?;
    mpse_spectra.sort();

    let mut rixs_maps = Vec::new();
    collect_named_files(&golden_dir, "referencerixsET.dat", &mut rixs_maps)?;
    collect_matching_nonempty_files(&golden_dir, &mut rixs_maps, &|name| name == "rixsET.dat")?;
    rixs_maps.sort();

    let mut rixs_lines = Vec::new();
    collect_named_files(&golden_dir, "referenceherfd.dat", &mut rixs_lines)?;
    collect_named_files(&golden_dir, "referenceherfd-sat.dat", &mut rixs_lines)?;
    collect_matching_nonempty_files(&golden_dir, &mut rixs_lines, &is_rixs_line_name)?;
    rixs_lines.sort();

    let mut highz_outputs = Vec::new();
    collect_named_files(&golden_dir, "HighZ.out", &mut highz_outputs)?;
    highz_outputs.sort();

    let mut xsecl_outputs = Vec::new();
    collect_named_files(&golden_dir, "xsecl.dat", &mut xsecl_outputs)?;
    xsecl_outputs.sort();

    let mut xsecl2_outputs = Vec::new();
    collect_named_files(&golden_dir, "xsecl2.dat", &mut xsecl2_outputs)?;
    xsecl2_outputs.sort();

    let mut xmul_outputs = Vec::new();
    collect_named_files(&golden_dir, "xmul.dat", &mut xmul_outputs)?;
    xmul_outputs.sort();

    ensure!(
        !(xmu_spectra.is_empty()
            && axafs_spectra.is_empty()
            && chi_spectra.is_empty()
            && eels_spectra.is_empty()
            && danes_spectra.is_empty()
            && ldos_spectra.is_empty()
            && rhoc_spectra.is_empty()
            && compton_spectra.is_empty()
            && rhozzp_spectra.is_empty()
            && crpa_spectra.is_empty()
            && loss_spectra.is_empty()
            && eps_tables.is_empty()
            && osc_str_tables.is_empty()
            && opcons_tables.is_empty()
            && sumrules_tables.is_empty()
            && drude_tables.is_empty()
            && hamaker_tables.is_empty()
            && exc_tables.is_empty()
            && mpse_spectra.is_empty()
            && rixs_maps.is_empty()
            && rixs_lines.is_empty()
            && highz_outputs.is_empty()
            && xsecl_outputs.is_empty()
            && xsecl2_outputs.is_empty()
            && xmul_outputs.is_empty()),
        "no generated FEFF spectrum reference outputs found"
    );

    for spectrum in &xmu_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_xmu_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = xmu_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "xmu.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &axafs_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_axafs_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = axafs_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "axafs.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &chi_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_chi_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = chi_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "chi.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &eels_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_eels_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = eels_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "eels.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &danes_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_danes_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = danes_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "danes.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &ldos_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_ldos_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = ldos_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "ldos.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &rhoc_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_rhoc_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        ensure!(
            matches!(parsed.density.ncols(), 3 | 4),
            "{} has an unexpected rhoc density width",
            spectrum.display()
        );
        let rendered = rhoc_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "rhoc.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &compton_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_compton_dat_lossless(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = compton_dat_lossless_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "compton.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &rhozzp_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_rhozzp_dat_lossless(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = rhozzp_dat_lossless_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "rhozzp.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &crpa_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_crpa_dat_lossless(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = crpa_dat_lossless_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "crpa.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &loss_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_loss_dat_lossless(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = loss_dat_lossless_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "loss.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for table in &eps_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed =
            parse_eps_dat(&text).with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = eps_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "eps.dat roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for table in &osc_str_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed = parse_osc_str_dat(&text)
            .with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = osc_str_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "osc_str.dat roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for table in &opcons_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed = parse_opcons_dat(&text)
            .with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = opcons_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "opcons optical-constants roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for table in &sumrules_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed = parse_sumrules_dat(&text)
            .with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = sumrules_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "sumrules.dat roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for table in &drude_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed = parse_drude_dat(&text)
            .with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = drude_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "drude.dat roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for table in &hamaker_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed = parse_hamaker_dat(&text)
            .with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = hamaker_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "hamaker.dat roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for table in &exc_tables {
        let text = std::fs::read_to_string(table)
            .with_context(|| format!("failed to read {}", table.display()))?;
        let parsed =
            parse_exc_dat(&text).with_context(|| format!("failed to parse {}", table.display()))?;
        let rendered = exc_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", table.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "exc.dat roundtrip mismatch for {}: {mismatch}",
                table.display()
            );
        }
    }
    for spectrum in &mpse_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_mpse_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = mpse_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "mpse.dat roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &rixs_maps {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_rixs_map_lossless(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = rixs_map_lossless_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "rixs map roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for spectrum in &rixs_lines {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_rixs_line(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        let rendered = rixs_line_string(&parsed)
            .with_context(|| format!("failed to render {}", spectrum.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "rixs line roundtrip mismatch for {}: {mismatch}",
                spectrum.display()
            );
        }
    }
    for output in &highz_outputs {
        let text = std::fs::read_to_string(output)
            .with_context(|| format!("failed to read {}", output.display()))?;
        let parsed = parse_highz_out(&text)
            .with_context(|| format!("failed to parse {}", output.display()))?;
        ensure!(
            parsed.row_count() >= 100,
            "{} has too few HIGHZ rows",
            output.display()
        );
        let rendered = highz_out_string(&parsed)
            .with_context(|| format!("failed to render {}", output.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "HighZ.out roundtrip mismatch for {}: {mismatch}",
                output.display()
            );
        }
    }
    for output in &xsecl_outputs {
        let text = std::fs::read_to_string(output)
            .with_context(|| format!("failed to read {}", output.display()))?;
        let parsed = parse_xsecl_dat(&text)
            .with_context(|| format!("failed to parse {}", output.display()))?;
        ensure!(
            parsed.row_count() >= parsed.header.real_energy_count,
            "{} has fewer rows than ne1",
            output.display()
        );
        let rendered = xsecl_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", output.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "xsecl.dat roundtrip mismatch for {}: {mismatch}",
                output.display()
            );
        }
    }
    for output in &xsecl2_outputs {
        let text = std::fs::read_to_string(output)
            .with_context(|| format!("failed to read {}", output.display()))?;
        let parsed = parse_xsecl2_dat(&text)
            .with_context(|| format!("failed to parse {}", output.display()))?;
        ensure!(
            parsed.row_count() >= parsed.header.real_energy_count,
            "{} has fewer rows than ne1",
            output.display()
        );
        let rendered = xsecl2_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", output.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "xsecl2.dat roundtrip mismatch for {}: {mismatch}",
                output.display()
            );
        }
    }
    for output in &xmul_outputs {
        let text = std::fs::read_to_string(output)
            .with_context(|| format!("failed to read {}", output.display()))?;
        let parsed = parse_xmul_dat(&text)
            .with_context(|| format!("failed to parse {}", output.display()))?;
        ensure!(
            parsed.point_count() >= 1,
            "{} has no xmul.dat rows",
            output.display()
        );
        ensure!(
            parsed.channel_count() == parsed.max_decomposition_channel + 1,
            "{} has inconsistent xmul.dat channel metadata",
            output.display()
        );
        let rendered = xmul_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", output.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "xmul.dat roundtrip mismatch for {}: {mismatch}",
                output.display()
            );
        }
    }
    Ok(())
}
