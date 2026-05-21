use super::*;

#[test]
fn roundtrips_generated_reference_reciprocal_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping reciprocal.inp roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut reciprocal_inputs = Vec::new();
    collect_named_files(&golden_dir, "reciprocal.inp", &mut reciprocal_inputs)?;
    reciprocal_inputs.sort();
    ensure!(
        !reciprocal_inputs.is_empty(),
        "no generated reciprocal.inp files found"
    );

    for path in reciprocal_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = ReciprocalInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = reciprocal_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "reciprocal.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
    }

    Ok(())
}

#[test]
fn roundtrips_generated_reference_shared_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping shared-module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut global_inputs = Vec::new();
    collect_named_files(&golden_dir, "global.inp", &mut global_inputs)?;
    global_inputs.sort();
    for path in global_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = GlobalInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = global_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "global.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut compton_inputs = Vec::new();
    collect_named_files(&golden_dir, "compton.inp", &mut compton_inputs)?;
    compton_inputs.sort();
    for path in compton_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = ComptonInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = compton_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "compton.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut eels_inputs = Vec::new();
    collect_named_files(&golden_dir, "eels.inp", &mut eels_inputs)?;
    eels_inputs.sort();
    for path in eels_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = EelsInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = eels_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "eels.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(compared > 0, "no generated shared-module input files found");
    Ok(())
}

#[test]
fn roundtrips_generated_reference_module_control_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping module-control input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut band_inputs = Vec::new();
    collect_named_files(&golden_dir, "band.inp", &mut band_inputs)?;
    band_inputs.sort();
    for path in band_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = BandInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = band_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "band.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut fullspectrum_inputs = Vec::new();
    collect_named_files(&golden_dir, "fullspectrum.inp", &mut fullspectrum_inputs)?;
    fullspectrum_inputs.sort();
    for path in fullspectrum_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = FullSpectrumInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = fullspectrum_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "fullspectrum.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut opcons_inputs = Vec::new();
    collect_named_files(&golden_dir, "opcons.inp", &mut opcons_inputs)?;
    opcons_inputs.sort();
    for path in opcons_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = OpconsInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = opcons_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "opcons.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(
        compared > 0,
        "no generated module-control input files found"
    );
    Ok(())
}

#[test]
fn roundtrips_generated_reference_scalar_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping scalar module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut crpa_inputs = Vec::new();
    collect_named_files(&golden_dir, "crpa.inp", &mut crpa_inputs)?;
    crpa_inputs.sort();
    for path in crpa_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = CrpaInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = crpa_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "crpa.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut hubbard_inputs = Vec::new();
    collect_named_files(&golden_dir, "hubbard.inp", &mut hubbard_inputs)?;
    hubbard_inputs.sort();
    for path in hubbard_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = HubbardInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = hubbard_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "hubbard.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut screen_inputs = Vec::new();
    collect_named_files(&golden_dir, "screen.inp", &mut screen_inputs)?;
    screen_inputs.sort();
    for path in screen_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = ScreenInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = screen_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "screen.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(compared > 0, "no generated scalar module input files found");
    Ok(())
}

#[test]
fn roundtrips_generated_reference_path_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping path-module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut paths_inputs = Vec::new();
    collect_named_files(&golden_dir, "paths.inp", &mut paths_inputs)?;
    paths_inputs.sort();
    for path in paths_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = PathsInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = paths_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "paths.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut sfconv_inputs = Vec::new();
    collect_named_files(&golden_dir, "sfconv.inp", &mut sfconv_inputs)?;
    sfconv_inputs.sort();
    for path in sfconv_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = SfconvInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = sfconv_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "sfconv.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut dmdw_inputs = Vec::new();
    collect_named_files(&golden_dir, "dmdw.inp", &mut dmdw_inputs)?;
    dmdw_inputs.sort();
    for path in dmdw_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = DmdwInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = dmdw_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "dmdw.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(compared > 0, "no generated path-module input files found");
    Ok(())
}

#[test]
fn roundtrips_generated_reference_scattering_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping scattering-module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut fms_inputs = Vec::new();
    collect_named_files(&golden_dir, "fms.inp", &mut fms_inputs)?;
    fms_inputs.sort();
    for path in fms_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = FmsInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = fms_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(false, "fms.inp mismatch for {}: {mismatch}", path.display());
        }
        compared += 1;
    }

    let mut genfmt_inputs = Vec::new();
    collect_named_files(&golden_dir, "genfmt.inp", &mut genfmt_inputs)?;
    genfmt_inputs.sort();
    for path in genfmt_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = GenfmtInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = genfmt_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "genfmt.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(
        compared > 0,
        "no generated scattering-module input files found"
    );
    Ok(())
}

#[test]
fn roundtrips_generated_reference_phase_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping phase-module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut xsph_inputs = Vec::new();
    collect_named_files(&golden_dir, "xsph.inp", &mut xsph_inputs)?;
    xsph_inputs.sort();
    for path in xsph_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = XsphInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = xsph_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "xsph.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(compared > 0, "no generated phase-module input files found");
    Ok(())
}

#[test]
fn roundtrips_generated_reference_potential_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping potential-module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut pot_inputs = Vec::new();
    collect_named_files(&golden_dir, "pot.inp", &mut pot_inputs)?;
    pot_inputs.sort();
    for path in pot_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = PotInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = pot_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(false, "pot.inp mismatch for {}: {mismatch}", path.display());
        }
        compared += 1;
    }

    ensure!(
        compared > 0,
        "no generated potential-module input files found"
    );
    Ok(())
}

#[test]
fn roundtrips_generated_reference_spectrum_module_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping spectrum-module input roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;
    let mut ff2x_inputs = Vec::new();
    collect_named_files(&golden_dir, "ff2x.inp", &mut ff2x_inputs)?;
    ff2x_inputs.sort();
    for path in ff2x_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = Ff2xInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = ff2x_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "ff2x.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut ldos_inputs = Vec::new();
    collect_named_files(&golden_dir, "ldos.inp", &mut ldos_inputs)?;
    ldos_inputs.sort();
    for path in ldos_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = LdosInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = ldos_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "ldos.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut rixs_inputs = Vec::new();
    collect_named_files(&golden_dir, "rixs.inp", &mut rixs_inputs)?;
    rixs_inputs.sort();
    for path in rixs_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = RixsInput::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = rixs_input_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "rixs.inp mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(
        compared > 0,
        "no generated spectrum-module input files found"
    );
    Ok(())
}
