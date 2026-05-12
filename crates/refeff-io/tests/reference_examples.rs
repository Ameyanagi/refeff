use std::path::{Path, PathBuf};

use anyhow::{Context as _, ensure};
use refeff_io::{
    AtomsDat, BandInput, ComptonInput, ConfigInput, CrpaInput, DensityInput, DimensionsDat,
    DmdwInput, EelsInput, FeffDocument, FeffInput, Ff2xInput, FmsInput, FullSpectrumInput,
    GenfmtInput, GeomDat, GlobalInput, GridInput, HubbardInput, LdosInput, OpconsInput, PathsInput,
    PotInput, ReciprocalInput, RixsInput, ScreenInput, SfconvInput, SpringInput, XsphInput,
    parse_chi_dat, parse_dym, parse_feff_bin, parse_feffl_bin, parse_fms_bin, parse_fmsl_bin,
    parse_list_dat, parse_paths_dat, parse_phase_bin, parse_pot_bin, parse_xmu_dat,
    parse_xsecl_bin, parse_xsect_dat, rdinp,
};

#[test]
fn parses_all_local_reference_examples_when_present() -> anyhow::Result<()> {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples");
    if !examples_dir.exists() {
        eprintln!("skipping local reference example parse; feff10/examples not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();
    ensure!(
        inputs.len() == 44,
        "unexpected local FEFF example count: {}",
        inputs.len()
    );

    for input in inputs {
        let parsed = FeffInput::parse_file(&input)
            .with_context(|| format!("failed to parse {}", input.display()))?;
        FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input.display()))?;
    }

    let mut spring_inputs = Vec::new();
    collect_named_files(&examples_dir, "spring.inp", &mut spring_inputs)?;
    spring_inputs.sort();
    for input in &spring_inputs {
        let text = std::fs::read_to_string(input)
            .with_context(|| format!("failed to read {}", input.display()))?;
        SpringInput::parse_str(&text)
            .with_context(|| format!("failed to parse {}", input.display()))?;
    }
    ensure!(
        !spring_inputs.is_empty(),
        "no local FEFF spring.inp examples found"
    );
    Ok(())
}

#[test]
fn matches_generated_reference_rdinp_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated reference output comparison; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input in inputs {
        let output_dir = input
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input)
            .with_context(|| format!("failed to parse {}", input.display()))?;
        let document = FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input.display()))?;
        let outputs = rdinp::text_outputs(&document)
            .with_context(|| format!("failed to render rdinp outputs for {}", input.display()))?;

        for (name, actual) in outputs {
            let expected_path = output_dir.join(name);
            if !expected_path.exists() {
                continue;
            }
            let expected = std::fs::read_to_string(&expected_path)
                .with_context(|| format!("failed to read {}", expected_path.display()))?;

            ensure!(
                actual == expected,
                "{name} mismatch for {}",
                input.display()
            );
            compared += 1;
        }
    }

    ensure!(compared > 0, "no generated rdinp outputs found");
    Ok(())
}

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
        parsed_count += parse_handoff_file(output_dir, "crpa.inp", CrpaInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "density.inp", DensityInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "dmdw.inp", DmdwInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "feff.dym", |_, text| parse_dym(text))?;
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
        parsed_count +=
            parse_handoff_file(output_dir, "paths.dat", |_, text| parse_paths_dat(text))?;
        parsed_count += parse_handoff_file(output_dir, "genfmt.inp", GenfmtInput::parse_str)?;
        parsed_count +=
            parse_handoff_file(output_dir, "reciprocal.inp", ReciprocalInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "ff2x.inp", Ff2xInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "rixs.inp", RixsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "spring.inp", |_, text| {
            SpringInput::parse_str(text)
        })?;
        parsed_count += parse_handoff_file(output_dir, "pot.bin", |_, text| parse_pot_bin(text))?;
        parsed_count +=
            parse_handoff_file(output_dir, "phase.bin", |_, text| parse_phase_bin(text))?;
        parsed_count += parse_handoff_file(output_dir, "feff.bin", |_, text| parse_feff_bin(text))?;
        parsed_count += parse_feffl_bin_when_present(output_dir)?;
        parsed_count += parse_handoff_file(output_dir, "list.dat", |_, text| parse_list_dat(text))?;
        parsed_count +=
            parse_handoff_file(output_dir, "xsect.dat", |_, text| parse_xsect_dat(text))?;
        parsed_count += parse_handoff_file(output_dir, "fms.bin", |_, text| parse_fms_bin(text))?;
        parsed_count += parse_fmsl_bin_when_present(output_dir)?;
        parsed_count += parse_xsecl_bin_when_present(output_dir)?;
    }

    ensure!(parsed_count > 0, "no generated handoff files parsed");
    Ok(())
}

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
    xmu_spectra.sort();

    let mut chi_spectra = Vec::new();
    collect_named_files(&golden_dir, "referencechi.dat", &mut chi_spectra)?;
    chi_spectra.sort();
    ensure!(
        !(xmu_spectra.is_empty() && chi_spectra.is_empty()),
        "no generated FEFF spectrum reference outputs found"
    );

    for spectrum in &xmu_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_xmu_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &chi_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_chi_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    Ok(())
}

fn parse_handoff_file<T>(
    output_dir: &Path,
    name: &str,
    parse: impl FnOnce(PathBuf, &str) -> refeff_io::Result<T>,
) -> anyhow::Result<usize> {
    let path = output_dir.join(name);
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse(path.clone(), &text).with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(1)
}

fn parse_feffl_bin_when_present(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("feffl.bin");
    if !path.exists() {
        return Ok(0);
    }

    let feff_bin_path = output_dir.join("feff.bin");
    let feff_bin_text = std::fs::read_to_string(&feff_bin_path)
        .with_context(|| format!("failed to read {}", feff_bin_path.display()))?;
    let feff_bin = parse_feff_bin(&feff_bin_text)
        .with_context(|| format!("failed to parse {}", feff_bin_path.display()))?;

    let genfmt_input_path = output_dir.join("genfmt.inp");
    let genfmt_input_text = std::fs::read_to_string(&genfmt_input_path)
        .with_context(|| format!("failed to read {}", genfmt_input_path.display()))?;
    let genfmt_input = GenfmtInput::parse_str(genfmt_input_path.clone(), &genfmt_input_text)
        .with_context(|| format!("failed to parse {}", genfmt_input_path.display()))?;
    ensure!(
        genfmt_input.decomposition_channels >= 0,
        "feffl.bin exists but genfmt.inp has negative decomposition channel count"
    );
    let max_decomposition_channel = usize::try_from(genfmt_input.decomposition_channels)
        .with_context(|| "failed to convert GENFMT decomposition channel count")?;

    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_feffl_bin(
        &text,
        feff_bin.pad_width,
        feff_bin.paths.len(),
        feff_bin.energy_count(),
        max_decomposition_channel,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(1)
}

fn parse_fmsl_bin_when_present(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("fmsl.bin");
    if !path.exists() {
        return Ok(0);
    }

    let fms_bin_path = output_dir.join("fms.bin");
    let fms_bin_text = std::fs::read_to_string(&fms_bin_path)
        .with_context(|| format!("failed to read {}", fms_bin_path.display()))?;
    let fms_bin = parse_fms_bin(&fms_bin_text)
        .with_context(|| format!("failed to parse {}", fms_bin_path.display()))?;

    let fms_input_path = output_dir.join("fms.inp");
    let fms_input_text = std::fs::read_to_string(&fms_input_path)
        .with_context(|| format!("failed to read {}", fms_input_path.display()))?;
    let fms_input = FmsInput::parse_str(fms_input_path.clone(), &fms_input_text)
        .with_context(|| format!("failed to parse {}", fms_input_path.display()))?;
    ensure!(
        fms_input.decomposition_channels >= 0,
        "fmsl.bin exists but fms.inp has negative decomposition channel count"
    );
    let max_decomposition_channel = usize::try_from(fms_input.decomposition_channels)
        .with_context(|| "failed to convert FMS decomposition channel count")?;

    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_fmsl_bin(
        &text,
        fms_bin.pad_width,
        fms_bin.energy_count,
        max_decomposition_channel,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(1)
}

fn parse_xsecl_bin_when_present(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("xsecl.bin");
    if !path.exists() {
        return Ok(0);
    }

    let phase_path = output_dir.join("phase.bin");
    let phase_text = std::fs::read_to_string(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let phase = parse_phase_bin(&phase_text)
        .with_context(|| format!("failed to parse {}", phase_path.display()))?;

    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_xsecl_bin(&text, phase.pad_width, phase.energy_count)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(1)
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    collect_named_files(dir, "feff.inp", inputs)
}

fn collect_named_files(dir: &Path, name: &str, inputs: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_named_files(&path, name, inputs)?;
        } else if path.file_name().is_some_and(|file_name| file_name == name) {
            inputs.push(path);
        }
    }
    Ok(())
}
