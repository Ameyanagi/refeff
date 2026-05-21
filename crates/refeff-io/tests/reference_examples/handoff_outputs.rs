use super::*;

#[test]
fn parses_generated_reference_handoff_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated handoff parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut parsed_count = 0_usize;
    for input in inputs {
        let output_dir = input
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }

        parsed_count +=
            parse_handoff_file(output_dir, ".dimensions.dat", DimensionsDat::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "atoms.dat", AtomsDat::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "band.inp", BandInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "global.inp", GlobalInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "compton.inp", ComptonInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "config.inp", |_, text| {
            ConfigInput::parse_str(text)
        })?;
        parsed_count += roundtrip_handoff_config_dat(output_dir)?;
        parsed_count += parse_handoff_file(output_dir, "crpa.inp", CrpaInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "density.inp", DensityInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "dmdw.inp", DmdwInput::parse_str)?;
        parsed_count += roundtrip_handoff_dym(output_dir)?;
        parsed_count += parse_handoff_file(output_dir, "log.dat", |_, text| parse_log_dat(text))?;
        parsed_count += parse_handoff_file(output_dir, "eels.inp", EelsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "hubbard.inp", HubbardInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "pot.inp", PotInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "screen.inp", ScreenInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "sfconv.inp", SfconvInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "xsph.inp", XsphInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "fms.inp", FmsInput::parse_str)?;
        parsed_count +=
            parse_handoff_file(output_dir, "fullspectrum.inp", FullSpectrumInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "geom.dat", GeomDat::parse_str)?;
        parsed_count +=
            parse_handoff_file(output_dir, "grid.inp", |_, text| GridInput::parse_str(text))?;
        parsed_count += parse_handoff_file(output_dir, "ldos.inp", LdosInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "opcons.inp", OpconsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "paths.inp", PathsInput::parse_str)?;
        parsed_count += roundtrip_handoff_paths_dat(output_dir)?;
        parsed_count += parse_handoff_file(output_dir, "genfmt.inp", GenfmtInput::parse_str)?;
        parsed_count +=
            parse_handoff_file(output_dir, "reciprocal.inp", ReciprocalInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "ff2x.inp", Ff2xInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "rixs.inp", RixsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "spring.inp", |_, text| {
            SpringInput::parse_str(text)
        })?;
        parsed_count += parse_handoff_pot_bin(output_dir)?;
        parsed_count += roundtrip_handoff_phase_bin(output_dir)?;
        parsed_count += roundtrip_handoff_feff_bin(output_dir)?;
        parsed_count += parse_feffl_bin_when_present(output_dir)?;
        parsed_count += roundtrip_handoff_list_dat(output_dir)?;
        parsed_count += roundtrip_handoff_xsect_dat(output_dir)?;
        parsed_count += parse_handoff_fms_bin(output_dir)?;
        parsed_count += parse_fmsl_bin_when_present(output_dir)?;
        parsed_count += parse_xsecl_bin_when_present(output_dir)?;
    }

    ensure!(parsed_count > 0, "no generated handoff files parsed");
    Ok(())
}

#[test]
fn parses_generated_reference_energy_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated energy-output parser coverage; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut edges_files = Vec::new();
    collect_named_files(&golden_dir, "edges.dat", &mut edges_files)?;
    edges_files.sort();

    let mut chemical_files = Vec::new();
    collect_named_files(&golden_dir, "chemical.dat", &mut chemical_files)?;
    chemical_files.sort();

    let mut emesh_files = Vec::new();
    collect_named_files(&golden_dir, "emesh.dat", &mut emesh_files)?;
    emesh_files.sort();

    let mut emesh_bin_files = Vec::new();
    collect_named_files(&golden_dir, "emesh.bin", &mut emesh_bin_files)?;
    emesh_bin_files.sort();

    let mut fpf0_files = Vec::new();
    collect_named_files(&golden_dir, "fpf0.dat", &mut fpf0_files)?;
    fpf0_files.sort();

    ensure!(
        !(edges_files.is_empty()
            && chemical_files.is_empty()
            && emesh_files.is_empty()
            && emesh_bin_files.is_empty()
            && fpf0_files.is_empty()),
        "no generated FEFF energy reference outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &edges_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_edges_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.row_count() >= 1,
            "{} has no edge rows",
            path.display()
        );
        let rendered = edges_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "edges.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &chemical_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_chemical_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = chemical_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "chemical.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &emesh_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_emesh_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.point_count() >= parsed.fermi_index,
            "{} has an out-of-range ik0",
            path.display()
        );
        let rendered = emesh_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "emesh.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &emesh_bin_files {
        let parsed =
            read_emesh_bin(path).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.point_count() == parsed.point_count_declared,
            "{} has a mismatched binary energy count",
            path.display()
        );
        ensure!(
            parsed.horizontal_count <= parsed.point_count_declared,
            "{} has an out-of-range binary ne1",
            path.display()
        );
        if let Some(output_dir) = path.parent() {
            let text_path = output_dir.join("emesh.dat");
            if text_path.exists() {
                let text = std::fs::read_to_string(&text_path)
                    .with_context(|| format!("failed to read {}", text_path.display()))?;
                let text_grid = parse_emesh_dat(&text)
                    .with_context(|| format!("failed to parse {}", text_path.display()))?;
                ensure!(
                    parsed.point_count() == text_grid.point_count(),
                    "{} does not match sibling emesh.dat point count",
                    path.display()
                );
            }
        }
        let expected = std::fs::read(path)
            .with_context(|| format!("failed to read binary {}", path.display()))?;
        let rendered = emesh_bin_bytes(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        ensure!(
            rendered == expected,
            "emesh.bin byte roundtrip mismatch for {}",
            path.display()
        );
        parsed_count += 1;
    }
    for path in &fpf0_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_fpf0_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.oscillator_count() >= 1,
            "{} has no oscillator rows",
            path.display()
        );
        ensure!(
            parsed.form_factor_count() >= 1,
            "{} has no form-factor rows",
            path.display()
        );
        let rendered = fpf0_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "fpf0.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }

    ensure!(parsed_count > 0, "no generated energy files parsed");
    Ok(())
}

#[test]
fn parses_generated_reference_xscorr_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated XSCORR parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut prexmu_files = Vec::new();
    collect_named_files(&golden_dir, "prexmu.dat", &mut prexmu_files)?;
    prexmu_files.sort();

    let mut residue_files = Vec::new();
    collect_named_files(&golden_dir, "residue.dat", &mut residue_files)?;
    residue_files.sort();

    let mut contour_files = Vec::new();
    collect_named_files(&golden_dir, "contour.dat", &mut contour_files)?;
    contour_files.sort();

    let mut curve_files = Vec::new();
    collect_named_files(&golden_dir, "curve.dat", &mut curve_files)?;
    curve_files.sort();

    let mut raw_files = Vec::new();
    collect_named_files(&golden_dir, "raw.dat", &mut raw_files)?;
    raw_files.sort();

    ensure!(
        !(prexmu_files.is_empty()
            && residue_files.is_empty()
            && contour_files.is_empty()
            && curve_files.is_empty()
            && raw_files.is_empty()),
        "no generated FEFF XSCORR reference outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &prexmu_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_prexmu_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = prexmu_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "prexmu.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &residue_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_residue_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = residue_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "residue.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &contour_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_contour_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = contour_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "contour.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &curve_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_curve_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = curve_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "curve.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &raw_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_xscorr_raw_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = xscorr_raw_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "raw.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }

    ensure!(parsed_count > 0, "no generated XSCORR files parsed");
    Ok(())
}

#[test]
fn parses_generated_reference_fms_diagnostics_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated FMS diagnostic parser coverage; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut gtr_files = Vec::new();
    collect_named_files(&golden_dir, "gtr.dat", &mut gtr_files)?;
    gtr_files.sort();

    let mut gtr_bin_files = Vec::new();
    collect_matching_nonempty_files(&golden_dir, &mut gtr_bin_files, &is_gtr_bin_name)?;
    gtr_bin_files.sort();

    let mut gg_files = Vec::new();
    collect_named_files(&golden_dir, "gg.dat", &mut gg_files)?;
    gg_files.sort();

    let mut gg_bin_files = Vec::new();
    collect_named_files(&golden_dir, "gg.bin", &mut gg_bin_files)?;
    gg_bin_files.sort();

    let mut gtrl_files = Vec::new();
    collect_named_files(&golden_dir, "gtrl.dat", &mut gtrl_files)?;
    gtrl_files.sort();

    ensure!(
        !(gtr_files.is_empty()
            && gtr_bin_files.is_empty()
            && gg_files.is_empty()
            && gg_bin_files.is_empty()
            && gtrl_files.is_empty()),
        "no generated FEFF FMS diagnostic reference outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &gtr_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_gtr_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = gtr_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "gtr.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &gtr_bin_files {
        let parsed =
            read_gtr_bin(path).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.energy_count() == parsed.point_count_declared,
            "{} has a mismatched gtrNN.bin energy count",
            path.display()
        );
        ensure!(
            parsed.potential_count() == parsed.highest_potential_index + 1,
            "{} has a mismatched gtrNN.bin potential count",
            path.display()
        );
        ensure!(
            parsed.angular_channel_count() >= 1,
            "{} has no gtrNN.bin angular channels",
            path.display()
        );
        let expected = std::fs::read(path)
            .with_context(|| format!("failed to read binary {}", path.display()))?;
        let rendered = gtr_bin_bytes(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        ensure!(
            rendered == expected,
            "gtrNN.bin byte roundtrip mismatch for {}",
            path.display()
        );
        parsed_count += 1;
    }
    for path in &gg_files {
        let parsed =
            read_gg_dat(path).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.section_count() >= 1,
            "{} has no sections",
            path.display()
        );
        ensure!(
            parsed
                .sections
                .iter()
                .all(|section| section.shape() == (16, 16)),
            "{} has an unexpected gg matrix shape",
            path.display()
        );
        let expected = std::fs::read(path)
            .with_context(|| format!("failed to read binary {}", path.display()))?;
        let rendered = gg_dat_bytes(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        ensure!(
            rendered == expected,
            "gg.dat byte roundtrip mismatch for {}",
            path.display()
        );
        parsed_count += 1;
    }
    for path in &gg_bin_files {
        let parsed =
            read_gg_bin(path).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.section_count() >= 1,
            "{} has no sections",
            path.display()
        );
        ensure!(
            parsed
                .sections
                .iter()
                .all(|section| section.shape() == (16, 16)),
            "{} has an unexpected gg.bin matrix shape",
            path.display()
        );
        let expected = std::fs::read(path)
            .with_context(|| format!("failed to read binary {}", path.display()))?;
        let rendered = gg_bin_bytes(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        ensure!(
            rendered == expected,
            "gg.bin byte roundtrip mismatch for {}",
            path.display()
        );
        parsed_count += 1;
    }
    for path in &gtrl_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_gtrl_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        ensure!(
            parsed.component_count() >= 3,
            "{} has too few decomposition components",
            path.display()
        );
        let rendered = gtrl_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "gtrl.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    ensure!(parsed_count > 0, "no generated FMS diagnostics parsed");
    Ok(())
}

#[test]
fn parses_generated_reference_per_potential_path_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated per-potential path parser coverage; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut list_files = Vec::new();
    collect_matching_nonempty_files(&golden_dir, &mut list_files, &is_indexed_list_dat_name)?;
    list_files.sort();

    let mut feff_bin_files = Vec::new();
    collect_matching_nonempty_files(&golden_dir, &mut feff_bin_files, &is_indexed_feff_bin_name)?;
    feff_bin_files.sort();

    ensure!(
        !(list_files.is_empty() && feff_bin_files.is_empty()),
        "no generated FEFF per-potential path outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &list_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_list_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            !parsed.titles.is_empty() || !parsed.entries.is_empty(),
            "{} has neither header titles nor selected path rows",
            path.display()
        );
        let rendered = list_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "listNN.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &feff_bin_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_feff_bin(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.energy_count() > 0 && parsed.potential_count() > 0,
            "{} has empty feffNN.bin metadata",
            path.display()
        );
        parsed_count += 1;
    }

    ensure!(
        parsed_count > 0,
        "no generated per-potential path outputs parsed"
    );
    Ok(())
}

#[test]
fn parses_generated_reference_dmdw_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated DMDW parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut outputs = Vec::new();
    collect_named_files(&golden_dir, "dmdw.out", &mut outputs)?;
    outputs.sort();
    ensure!(
        !outputs.is_empty(),
        "no generated FEFF DMDW reference outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_dmdw_out(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        if text.trim().is_empty() {
            ensure!(
                parsed.header.is_none(),
                "{} should have no header",
                path.display()
            );
            ensure!(
                parsed.sections.is_empty(),
                "{} should have no sections",
                path.display()
            );
        } else {
            let header = parsed
                .header
                .as_ref()
                .with_context(|| format!("{} has no DMDW header", path.display()))?;
            ensure!(
                parsed.section_count() >= 1,
                "{} has no DMDW sections",
                path.display()
            );
            ensure!(
                parsed.sections.iter().all(|section| {
                    section.pdos_poles.is_empty()
                        || section.pdos_poles.len() == header.lanczos_recursion_order
                }),
                "{} has a PDOS pole count that disagrees with its header",
                path.display()
            );
        }
        let rendered = dmdw_out_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "dmdw.out roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    ensure!(parsed_count > 0, "no generated DMDW outputs parsed");
    Ok(())
}

#[test]
fn parses_generated_reference_screen_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated screened-core-hole parser coverage; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut wscrn_outputs = Vec::new();
    collect_named_files(&golden_dir, "wscrn.dat", &mut wscrn_outputs)?;
    wscrn_outputs.sort();

    let mut vtot_outputs = Vec::new();
    collect_named_files(&golden_dir, "vtot.dat", &mut vtot_outputs)?;
    vtot_outputs.sort();

    ensure!(
        !(wscrn_outputs.is_empty() && vtot_outputs.is_empty()),
        "no generated FEFF screened-core-hole reference outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &wscrn_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_wscrn_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = wscrn_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "wscrn.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &vtot_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_vtot_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        let rendered = vtot_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "vtot.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }

    ensure!(
        parsed_count > 0,
        "no generated screened-core-hole outputs parsed"
    );
    Ok(())
}

#[test]
fn parses_generated_reference_apot_bin_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated apot.bin parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut outputs = Vec::new();
    collect_named_files(&golden_dir, "apot.bin", &mut outputs)?;
    outputs.sort();
    ensure!(
        !outputs.is_empty(),
        "no generated FEFF apot.bin outputs found"
    );

    for path in &outputs {
        let parsed =
            read_apot_bin(path).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.section_count() >= 20,
            "{} has too few apot.bin sections",
            path.display()
        );
        ensure!(
            parsed.matrix_count() >= 10,
            "{} has too few apot.bin matrix sections",
            path.display()
        );

        let first = parsed.sections.first().with_context(|| {
            format!(
                "{} did not contain a first apot.bin section",
                path.display()
            )
        })?;
        let ApotBinPayload::Records(records) = &first.payload else {
            bail!(
                "{} first apot.bin section is not scalar records",
                path.display()
            );
        };
        ensure!(
            records.row_count() == 1 && records.column_count() >= 6,
            "{} first apot.bin section has an unexpected scalar shape",
            path.display()
        );
        ensure!(
            matches!(records.rows[0].first(), Some(ApotBinValue::Int(_))),
            "{} first apot.bin scalar is not nph",
            path.display()
        );

        let expected = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read text {}", path.display()))?;
        let rendered = apot_bin_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        ensure!(
            rendered == expected,
            "apot.bin roundtrip mismatch for {}: {}",
            path.display(),
            first_mismatch(&expected, &rendered)
        );
    }
    Ok(())
}

#[test]
fn renders_generated_reference_wpot_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated wpot coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut pot_outputs = Vec::new();
    collect_named_files(&golden_dir, "pot.bin", &mut pot_outputs)?;
    pot_outputs.sort();
    ensure!(
        !pot_outputs.is_empty(),
        "no generated FEFF pot.bin outputs found"
    );

    let mut rendered_count = 0_usize;
    for pot_path in &pot_outputs {
        let work_dir = pot_path
            .parent()
            .with_context(|| format!("{} has no parent directory", pot_path.display()))?;
        let apot_path = work_dir.join("apot.bin");
        if !apot_path.is_file() {
            continue;
        }

        let pot = read_pot_bin(pot_path)
            .with_context(|| format!("failed to parse {}", pot_path.display()))?;
        let apot = read_apot_bin(&apot_path)
            .with_context(|| format!("failed to parse {}", apot_path.display()))?;
        let outputs = potential_dat_outputs_from_bins(&pot, &apot)
            .with_context(|| format!("failed to render wpot outputs for {}", work_dir.display()))?;
        ensure!(
            outputs.len() == pot.potential_count(),
            "{} rendered {} wpot files, expected {}",
            work_dir.display(),
            outputs.len(),
            pot.potential_count()
        );
        let pot00 = outputs
            .get("pot00.dat")
            .with_context(|| format!("{} did not render pot00.dat", work_dir.display()))?;
        ensure!(
            pot00.lines().count() >= 250,
            "{} pot00.dat output was unexpectedly short",
            work_dir.display()
        );
        rendered_count += outputs.len();
    }

    ensure!(
        rendered_count > 0,
        "no generated FEFF pot.bin/apot.bin pairs rendered wpot output"
    );
    Ok(())
}

#[test]
fn parses_generated_reference_module_logs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated module-log parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut logs = Vec::new();
    collect_matching_files(&golden_dir, &mut logs, &is_module_log_name)?;
    logs.sort();
    ensure!(!logs.is_empty(), "no generated FEFF module logs found");

    for path in &logs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_module_log_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = module_log_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "module-log roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        if path
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len()
            > 0
        {
            ensure!(
                !parsed.is_empty(),
                "{} parsed as empty despite nonempty file",
                path.display()
            );
        }
    }
    Ok(())
}

#[test]
fn accounts_for_generated_reference_file_names_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated reference filename coverage; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut names = BTreeSet::new();
    collect_file_names(&golden_dir, &mut names)?;
    ensure!(!names.is_empty(), "no generated FEFF reference files found");

    let unaccounted = names
        .iter()
        .filter(|name| !is_accounted_reference_file_name(name))
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        unaccounted.is_empty(),
        "generated FEFF reference file names need parser coverage or explicit ignore: {}",
        unaccounted.join(", ")
    );
    Ok(())
}

#[test]
fn parses_generated_reference_run_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated run-output parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut stdout_outputs = Vec::new();
    collect_named_files(&golden_dir, "feff.stdout", &mut stdout_outputs)?;
    stdout_outputs.sort();

    let mut stderr_outputs = Vec::new();
    collect_named_files(&golden_dir, "feff.stderr", &mut stderr_outputs)?;
    collect_named_files(&golden_dir, "rdinp.stderr", &mut stderr_outputs)?;
    stderr_outputs.sort();

    let mut fort11_outputs = Vec::new();
    collect_named_files(&golden_dir, "fort.11", &mut fort11_outputs)?;
    fort11_outputs.sort();

    ensure!(
        !(stdout_outputs.is_empty() && stderr_outputs.is_empty() && fort11_outputs.is_empty()),
        "no generated FEFF run outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &stdout_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_run_stdout(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.line_count() >= 1,
            "{} has no stdout lines",
            path.display()
        );
        ensure!(
            parsed.completion_count() >= 1,
            "{} has no module-completion events",
            path.display()
        );
        let rendered = run_stdout_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "stdout roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &stderr_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_run_stderr(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        if text.contains("floating-point exceptions") {
            ensure!(
                parsed.floating_point_note_count() >= 1,
                "{} has no floating-point exception notes",
                path.display()
            );
        }
        let rendered = run_stderr_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "stderr roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &fort11_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_fort11(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.completion_count() >= 1,
            "{} has no fort.11 module-completion event",
            path.display()
        );
        let rendered = run_stdout_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "fort.11 roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }

    ensure!(parsed_count > 0, "no generated run outputs parsed");
    Ok(())
}

#[test]
fn parses_generated_reference_pot_diagnostics_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated potential diagnostic parser coverage; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut convergence_outputs = Vec::new();
    collect_named_files(&golden_dir, "convergence.scf", &mut convergence_outputs)?;
    convergence_outputs.sort();

    let mut fine_outputs = Vec::new();
    collect_named_files(&golden_dir, "convergence.scf.fine", &mut fine_outputs)?;
    fine_outputs.sort();

    let mut fort16_outputs = Vec::new();
    collect_named_files(&golden_dir, "fort.16", &mut fort16_outputs)?;
    fort16_outputs.sort();

    let mut misc_outputs = Vec::new();
    collect_named_files(&golden_dir, "misc.dat", &mut misc_outputs)?;
    misc_outputs.sort();

    ensure!(
        !(convergence_outputs.is_empty()
            && fine_outputs.is_empty()
            && fort16_outputs.is_empty()
            && misc_outputs.is_empty()),
        "no generated FEFF potential diagnostic reference outputs found"
    );

    let mut parsed_count = 0_usize;
    for path in &convergence_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_convergence_scf(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = convergence_scf_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "convergence.scf roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &fine_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_convergence_scf_fine(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = convergence_scf_fine_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "convergence.scf.fine roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &fort16_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_fort16(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.row_count() >= 1,
            "{} has no total-energy rows",
            path.display()
        );
        let rendered = fort16_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "fort.16 roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }
    for path in &misc_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_misc_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.title_count() >= 1,
            "{} has no misc.dat title records",
            path.display()
        );
        let rendered = misc_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "misc.dat roundtrip mismatch for {}: {mismatch}",
                path.display()
            );
        }
        parsed_count += 1;
    }

    ensure!(
        parsed_count > 0,
        "no generated potential diagnostics parsed"
    );
    Ok(())
}
