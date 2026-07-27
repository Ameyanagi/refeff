#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail, ensure};
use refeff_io::{
    ApotBinPayload, ApotBinValue, AtomsDat, BandInput, ComptonInput, ConfigInput, CrpaInput,
    DensityInput, DimensionsDat, DmdwInput, EelsInput, FeffDocument, FeffInput, Ff2xInput,
    FmsInput, FullSpectrumInput, GenfmtInput, GeomDat, GlobalInput, GridInput,
    HUBBARD_APHASE_ENERGY_COUNT, HubbardInput, LdosInput, OpconsInput, PathsInput, PotInput,
    ReciprocalInput, RixsInput, ScreenInput, SfconvInput, SpringInput, XsphInput,
    aphase_hubbard_bin_bytes, apot_bin_string, atoms_dat_string, axafs_dat_string,
    band_input_string, chemical_dat_string, chi_dat_string, compton_dat_lossless_string,
    compton_input_string, config_dat_string, contour_dat_string, convergence_scf_fine_string,
    convergence_scf_string, crpa_dat_lossless_string, crpa_input_string, curve_dat_string,
    danes_dat_string, dimensions_dat_string, dmdw_input_string, dmdw_out_string, drude_dat_string,
    dym_string, edges_dat_string, eels_dat_string, eels_input_string, eels_magic_dat_string,
    emesh_bin_bytes, emesh_dat_string, eps_dat_string, exc_dat_string, expand_cif_cluster,
    feff_bin_string, feffl_bin_string, ff2x_input_string, fms_bin_lossless_string,
    fms_input_string, fmsl_bin_string, fort16_string, fpf0_dat_string, fullspectrum_input_string,
    genfmt_input_string, geom_dat_string, gg_bin_bytes, gg_dat_bytes, global_input_string,
    gtr_bin_bytes, gtr_dat_string, gtrl_dat_string, hamaker_dat_string, highz_out_string,
    hubbard_input_string, jzzp_dat_lossless_string, kmesh_dat_string, ldos_dat_string,
    ldos_input_string, list_dat_string, lmdos_dat_string, loss_dat_lossless_string,
    misc_dat_string, module_log_dat_string, mpse_dat_string, opcons_dat_string,
    opcons_input_string, osc_str_dat_string, parse_aphase_hubbard_bin_inferred, parse_axafs_dat,
    parse_chemical_dat, parse_chi_dat, parse_cif, parse_compton_dat_lossless, parse_config_dat,
    parse_contour_dat, parse_convergence_scf, parse_convergence_scf_fine, parse_crpa_dat_lossless,
    parse_curve_dat, parse_danes_dat, parse_dmdw_out, parse_drude_dat, parse_dym, parse_edges_dat,
    parse_eels_dat, parse_eels_magic_dat, parse_emesh_dat, parse_eps_dat, parse_exc_dat,
    parse_feff_bin, parse_feffl_bin, parse_fms_bin, parse_fms_bin_lossless, parse_fmsl_bin,
    parse_fort11, parse_fort16, parse_fpf0_dat, parse_gtr_dat, parse_gtrl_dat, parse_hamaker_dat,
    parse_highz_out, parse_jzzp_dat_lossless, parse_kmesh_dat, parse_ldos_dat, parse_list_dat,
    parse_lmdos_dat, parse_log_dat, parse_loss_dat_lossless, parse_misc_dat, parse_module_log_dat,
    parse_mpse_dat, parse_opcons_dat, parse_osc_str_dat, parse_paths_dat, parse_phase_bin,
    parse_pot_bin, parse_prexmu_dat, parse_residue_dat_lossless, parse_rhoc_dat, parse_rhocm_dat,
    parse_rhozzp_dat_lossless, parse_rixs_line, parse_rixs_map_lossless, parse_run_stderr,
    parse_run_stdout, parse_specfunct_dat, parse_sumrules_dat, parse_thermal_curve_dat,
    parse_transformation_hubbard_bin_inferred, parse_v_hubbard_bin_inferred, parse_vtot_dat,
    parse_wscrn_dat, parse_xmu_dat, parse_xmul_dat, parse_xscorr_raw_dat_lossless, parse_xsecl_bin,
    parse_xsecl_dat, parse_xsecl2_dat, parse_xsect_dat, paths_dat_string, paths_input_string,
    phase_bin_string, pot_bin_string, pot_input_string, potential_dat_outputs_from_bins,
    prexmu_dat_string, rdinp, read_apot_bin, read_emesh_bin, read_gg_bin, read_gg_dat,
    read_gtr_bin, read_pot_bin, reciprocal_input_string, residue_dat_lossless_string,
    rhoc_dat_string, rhocm_dat_string, rhozzp_dat_lossless_string, rixs_input_string,
    rixs_line_string, rixs_map_lossless_string, run_stderr_string, run_stdout_string,
    screen_input_string, sfconv_input_string, specfunct_dat_bytes, sumrules_dat_string,
    thermal_curve_dat_string, transformation_hubbard_bin_bytes, v_hubbard_bin_bytes,
    vtot_dat_string, wscrn_dat_string, xmu_dat_string, xmul_dat_string,
    xscorr_raw_dat_lossless_string, xsecl_bin_string, xsecl_dat_string, xsecl2_dat_string,
    xsect_dat_string, xsph_input_string,
};

#[path = "reference_examples/handoff_outputs.rs"]
mod handoff_outputs;
#[path = "reference_examples/local_examples.rs"]
mod local_examples;
#[path = "reference_examples/module_inputs.rs"]
mod module_inputs;
#[path = "reference_examples/rdinp_outputs.rs"]
mod rdinp_outputs;
#[path = "reference_examples/spectrum_outputs.rs"]
mod spectrum_outputs;
#[path = "reference_examples/structure.rs"]
mod structure;

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

fn roundtrip_handoff_config_dat(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("config.dat");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_config_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered = config_dat_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "config.dat roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn roundtrip_handoff_dym(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("feff.dym");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed = parse_dym(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered =
        dym_string(&parsed).with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "feff.dym roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn roundtrip_handoff_list_dat(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("list.dat");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_list_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered =
        list_dat_string(&parsed).with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "list.dat roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn roundtrip_handoff_paths_dat(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("paths.dat");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_paths_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered = paths_dat_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "paths.dat roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn parse_handoff_pot_bin(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("pot.bin");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_pot_bin(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let raw_titles = text
        .lines()
        .skip(1)
        .take(parsed.titles.len())
        .map(str::to_string)
        .collect::<Vec<_>>();
    ensure!(
        parsed.titles == raw_titles,
        "pot.bin title spacing mismatch for {}",
        path.display()
    );
    let rendered =
        pot_bin_string(&parsed).with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "pot.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn roundtrip_handoff_phase_bin(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("phase.bin");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_phase_bin(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered = phase_bin_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "phase.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn roundtrip_handoff_feff_bin(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("feff.bin");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_feff_bin(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered =
        feff_bin_string(&parsed).with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "feff.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn parse_handoff_fms_bin(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("fms.bin");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed = parse_fms_bin_lossless(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    if let Some(declared) = declared_fms_spectrum_count(&text)
        .with_context(|| format!("failed to inspect {}", path.display()))?
    {
        ensure!(
            parsed.data.declared_spectrum_count == Some(declared),
            "fms.bin declared nip mismatch for {}: expected {declared}, got {:?}",
            path.display(),
            parsed.data.declared_spectrum_count
        );
    }
    let rendered = fms_bin_lossless_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "fms.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn roundtrip_handoff_xsect_dat(output_dir: &Path) -> anyhow::Result<usize> {
    let path = output_dir.join("xsect.dat");
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed =
        parse_xsect_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered = xsect_dat_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "xsect.dat roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn declared_fms_spectrum_count(text: &str) -> anyhow::Result<Option<usize>> {
    let Some(counts_line) = text.lines().filter(|line| !line.trim().is_empty()).nth(1) else {
        return Ok(None);
    };
    let tokens = counts_line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 6 {
        return Ok(None);
    }
    let declared = tokens[5]
        .parse::<usize>()
        .with_context(|| format!("invalid fms.bin declared nip token {:?}", tokens[5]))?;
    Ok(Some(declared))
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
    let parsed = parse_feffl_bin(
        &text,
        feff_bin.pad_width,
        feff_bin.paths.len(),
        feff_bin.energy_count(),
        max_decomposition_channel,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered = feffl_bin_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "feffl.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
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
    let parsed = parse_fmsl_bin(
        &text,
        fms_bin.pad_width,
        fms_bin.energy_count,
        max_decomposition_channel,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let rendered =
        fmsl_bin_string(&parsed).with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "fmsl.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
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
    let parsed = parse_xsecl_bin(&text, phase.pad_width, phase.energy_count)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    ensure!(
        parsed.energy_count() >= phase.energy_count,
        "xsecl.bin has fewer energy blocks than phase.bin for {}",
        path.display()
    );
    ensure!(
        parsed.final_state_count() == phase.final_state_count,
        "xsecl.bin final-state count differs from phase.bin for {}",
        path.display()
    );
    ensure!(
        parsed.transition_index_count() == phase.transition_count,
        "xsecl.bin transition-index count differs from phase.bin for {}",
        path.display()
    );
    let rendered = xsecl_bin_string(&parsed)
        .with_context(|| format!("failed to render {}", path.display()))?;
    if rendered != text {
        let mismatch = first_mismatch(&text, &rendered);
        ensure!(
            false,
            "xsecl.bin roundtrip mismatch for {}: {mismatch}",
            path.display()
        );
    }
    Ok(1)
}

fn ensure_supported_reference_rdinp_outputs_are_rendered<'a, S>(
    output_dir: &Path,
    actual_names: impl Iterator<Item = &'a S>,
) -> anyhow::Result<()>
where
    S: AsRef<str> + 'a,
{
    let actual_names = actual_names
        .map(std::convert::AsRef::as_ref)
        .collect::<BTreeSet<_>>();
    for expected_name in supported_reference_rdinp_output_names(output_dir)? {
        ensure!(
            actual_names.contains(expected_name.as_str()),
            "missing supported rdinp output {expected_name} for {}",
            output_dir.display()
        );
    }
    Ok(())
}

fn supported_reference_rdinp_output_names(output_dir: &Path) -> anyhow::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(output_dir)
        .with_context(|| format!("failed to read {}", output_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", output_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name == "density.inp"
            && path
                .metadata()
                .with_context(|| format!("failed to stat {}", path.display()))?
                .len()
                == 0
        {
            // Full FEFF runs create an empty density.inp even when RDINP did not
            // receive a DENSITY block. Do not treat that as a required RDINP
            // handoff output.
            continue;
        }
        if is_supported_reference_rdinp_output(name) {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}

fn is_supported_reference_rdinp_output(name: &str) -> bool {
    matches!(
        name,
        ".dimensions.dat"
            | "atoms.dat"
            | "band.inp"
            | "compton.inp"
            | "config.inp"
            | "crpa.inp"
            | "density.inp"
            | "dmdw.inp"
            | "eels.inp"
            | "ff2x.inp"
            | "fms.inp"
            | "fullspectrum.inp"
            | "genfmt.inp"
            | "geom.dat"
            | "global.inp"
            | "grid.inp"
            | "hubbard.inp"
            | "ldos.inp"
            | "opcons.inp"
            | "paths.inp"
            | "pot.inp"
            | "reciprocal.inp"
            | "rixs.inp"
            | "screen.inp"
            | "sfconv.inp"
            | "spring.inp"
            | "xsph.inp"
    ) || name.ends_with(".dym")
}

fn is_accounted_reference_file_name(name: &str) -> bool {
    is_supported_reference_rdinp_output(name)
        || is_xmu_spectrum_name(name)
        || is_axafs_spectrum_name(name)
        || is_chi_spectrum_name(name)
        || is_ldos_spectrum_name(name)
        || is_rhoc_spectrum_name(name)
        || is_lmdos_magnetic_name(name)
        || is_rhocm_magnetic_name(name)
        || is_hubbard_binary_name(name)
        || is_specialized_codec_reference_file_name(name)
        || is_gtr_bin_name(name)
        || is_module_log_name(name)
        || is_indexed_list_dat_name(name)
        || is_indexed_feff_bin_name(name)
        || is_indexed_atom_dat_name(name)
        || is_indexed_pot_dat_name(name)
        || is_rixs_line_name(name)
        || known_unparsed_reference_file_reason(name).is_some()
        || name.ends_with(".cif")
        || matches!(
            name,
            ".feff.error"
                | ".hubbard-mkgtr-provenance.json"
                | ".reference-fallback.json"
                | ".rixs-native-provenance.json"
                | "REFERENCE.zip"
                | "README.txt"
                | "about.txt"
                | "apot.bin"
                | "chemical.dat"
                | "chi.dat"
                | "compton.dat"
                | "config.dat"
                | "contour.dat"
                | "convergence.scf"
                | "convergence.scf.fine"
                | "crpa.dat"
                | "curve.dat"
                | "danes.dat"
                | "drude.dat"
                | "dmdw.out"
                | "density.bin"
                | "density.dat"
                | "edges.dat"
                | "eels.dat"
                | "emesh.bin"
                | "emesh.dat"
                | "eps.dat"
                | "exc.dat"
                | "feff.bin"
                | "feff.inp"
                | "feff-ldos.stderr"
                | "feff-ldos.stdout"
                | "feff.stderr"
                | "feff.stdout"
                | "feffl.bin"
                | "fms.bin"
                | "fmsl.bin"
                | "fort.11"
                | "fort.16"
                | "fpf0.dat"
                | "gg.bin"
                | "gg.dat"
                | "gtr.dat"
                | "gtrl.dat"
                | "gg_diag.bin"
                | "gg_slice.bin"
                | "hamaker.dat"
                | "HighZ.out"
                | "list.dat"
                | "listedges.pmbse"
                | "log.dat"
                | "loss.dat"
                | "manifest.json"
                | "misc.dat"
                | "mpse.dat"
                | "opcons.dat"
                | "opcons0.dat"
                | "opconsKK.dat"
                | "osc_str.dat"
                | "paths.dat"
                | "phase.bin"
                | "pot.bin"
                | "pot.dat"
                | "prexmu.dat"
                | "raw.dat"
                | "readme.txt"
                | "reference_compton.dat"
                | "reference_eels.dat"
                | "reference_rhozzp.dat"
                | "reference_xmu.dat"
                | "referencechi.dat"
                | "referencecrpa.dat"
                | "referencedanes.dat"
                | "referenceherfd-sat.dat"
                | "referenceherfd.dat"
                | "referenceldos00.dat"
                | "referencerixsET.dat"
                | "referencexmu.dat"
                | "residue.dat"
                | "rdinp.stderr"
                | "rdinp.stdout"
                | "rhozzp.dat"
                | "rixs.sh"
                | "rixsET.dat"
                | "runall"
                | "sumrules.dat"
                | "test"
                | "vtot.dat"
                | "wscrn.dat"
                | "xmul.dat"
                | "xmu.dat"
                | "xscorr.raw"
                | "xsecl.bin"
                | "xsecl.dat"
                | "xsecl2.dat"
                | "xsect.dat"
                | "xsedge.dat"
                // FEFF upstream diagnostic scratch units retained for
                // provenance; no stable public format is claimed.
                | "fort.18"
                | "fort.43"
                | "fort.77"
                | "fort.78"
                // Generated subprogram logs use the shared run-output codec.
                | "fms.stderr"
                | "fms.stdout"
                | "rhorrp.stderr"
                | "rhorrp.stdout"
                | "xsph.stderr"
                | "xsph.stdout"
        )
}

fn first_mismatch(expected: &str, actual: &str) -> String {
    for (index, (expected_line, actual_line)) in expected.lines().zip(actual.lines()).enumerate() {
        if expected_line != actual_line {
            return format!(
                "line {} expected {:?}, got {:?}",
                index + 1,
                expected_line.escape_debug().to_string(),
                actual_line.escape_debug().to_string()
            );
        }
    }
    if let Some(index) = expected
        .as_bytes()
        .iter()
        .zip(actual.as_bytes().iter())
        .position(|(expected, actual)| expected != actual)
    {
        return format!(
            "byte {} expected 0x{:02x}, got 0x{:02x}",
            index + 1,
            expected.as_bytes()[index],
            actual.as_bytes()[index]
        );
    }
    if expected.len() != actual.len() {
        return format!(
            "byte length differs: expected {}, got {}",
            expected.len(),
            actual.len()
        );
    }
    format!(
        "line count differs: expected {}, got {}",
        expected.lines().count(),
        actual.lines().count()
    )
}

fn ensure_periodic_structure_matches(
    name: &str,
    expected_path: &Path,
    expected: &str,
    actual: &str,
) -> anyhow::Result<()> {
    match name {
        "atoms.dat" => {
            let expected = AtomsDat::parse_str(expected_path, expected)?;
            let actual = AtomsDat::parse_str("actual atoms.dat", actual)?;
            ensure!(
                expected.natx == actual.natx,
                "atoms.dat natx mismatch: expected {}, got {}",
                expected.natx,
                actual.natx
            );

            let mut expected_atoms = expected
                .atoms
                .iter()
                .copied()
                .map(rounded_atoms_dat_key)
                .collect::<Vec<_>>();
            let mut actual_atoms = actual
                .atoms
                .iter()
                .copied()
                .map(rounded_atoms_dat_key)
                .collect::<Vec<_>>();
            expected_atoms.sort_unstable();
            actual_atoms.sort_unstable();
            ensure!(
                expected_atoms == actual_atoms,
                "atoms.dat atom set mismatch"
            );
        }
        "geom.dat" => {
            let expected = GeomDat::parse_str(expected_path, expected)?;
            let actual = GeomDat::parse_str("actual geom.dat", actual)?;
            ensure!(
                expected.nat == actual.nat && expected.nph == actual.nph,
                "geom.dat header mismatch: expected nat/nph {}/{}, got {}/{}",
                expected.nat,
                expected.nph,
                actual.nat,
                actual.nph
            );
            ensure!(
                expected.model_atoms == actual.model_atoms,
                "geom.dat model atom mismatch"
            );

            let mut expected_atoms = expected
                .atoms
                .iter()
                .copied()
                .map(rounded_geom_dat_key)
                .collect::<Vec<_>>();
            let mut actual_atoms = actual
                .atoms
                .iter()
                .copied()
                .map(rounded_geom_dat_key)
                .collect::<Vec<_>>();
            expected_atoms.sort_unstable();
            actual_atoms.sort_unstable();
            ensure!(expected_atoms == actual_atoms, "geom.dat atom set mismatch");
        }
        _ => bail!("unsupported periodic structural output {name}"),
    }
    Ok(())
}

fn rounded_cluster_atom_key(atom: refeff_io::CifClusterAtom) -> (i64, i64, i64, i32, i64) {
    (
        rounded(atom.x),
        rounded(atom.y),
        rounded(atom.z),
        atom.potential,
        rounded(cluster_atom_distance(atom)),
    )
}

fn rounded_atoms_dat_key(atom: refeff_io::AtomsDatRow) -> (i64, i64, i64, i32, i64) {
    (
        rounded(atom.x),
        rounded(atom.y),
        rounded(atom.z),
        atom.iph,
        rounded(atom.distance),
    )
}

fn rounded_geom_dat_key(atom: refeff_io::GeomDatRow) -> (i64, i64, i64, i32) {
    (rounded(atom.x), rounded(atom.y), rounded(atom.z), atom.iph)
}

fn cluster_atom_distance(atom: refeff_io::CifClusterAtom) -> f64 {
    (atom
        .x
        .mul_add(atom.x, atom.y.mul_add(atom.y, atom.z * atom.z)))
    .sqrt()
}

fn rounded(value: f64) -> i64 {
    let rounded = (value * 100_000.0).round() as i64;
    if rounded == 0 { 0 } else { rounded }
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    collect_named_files(dir, "feff.inp", inputs)
}

fn collect_file_names(dir: &Path, names: &mut BTreeSet<String>) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_file_names(&path, names)?;
        } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            names.insert(name.to_string());
        }
    }
    Ok(())
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

fn collect_matching_nonempty_files(
    dir: &Path,
    inputs: &mut Vec<PathBuf>,
    matches_name: &impl Fn(&str) -> bool,
) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_matching_nonempty_files(&path, inputs, matches_name)?;
            continue;
        }

        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if matches_name(name)
            && path
                .metadata()
                .with_context(|| format!("failed to stat {}", path.display()))?
                .len()
                > 0
        {
            inputs.push(path);
        }
    }
    Ok(())
}

fn collect_matching_files(
    dir: &Path,
    inputs: &mut Vec<PathBuf>,
    matches_name: &impl Fn(&str) -> bool,
) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_matching_files(&path, inputs, matches_name)?;
            continue;
        }

        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if matches_name(name) {
            inputs.push(path);
        }
    }
    Ok(())
}

fn collect_extension_files(
    dir: &Path,
    extension: &str,
    inputs: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_extension_files(&path, extension, inputs)?;
        } else if path
            .extension()
            .is_some_and(|found_extension| found_extension == extension)
        {
            inputs.push(path);
        }
    }
    Ok(())
}

fn is_xmu_spectrum_name(name: &str) -> bool {
    name == "xmu.dat"
        || name
            .strip_prefix("xmu")
            .and_then(|suffix| suffix.strip_suffix(".dat"))
            .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_axafs_spectrum_name(name: &str) -> bool {
    name == "axafs.dat"
}

fn is_chi_spectrum_name(name: &str) -> bool {
    name == "chi.dat"
        || ["chi", "chip"].iter().any(|prefix| {
            name.strip_prefix(prefix)
                .and_then(|suffix| suffix.strip_suffix(".dat"))
                .is_some_and(|index| {
                    !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit())
                })
        })
}

fn is_lmdos_magnetic_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "lmdos", ".dat")
}

fn is_rhocm_magnetic_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "rhocm", ".dat")
}

fn is_hubbard_binary_name(name: &str) -> bool {
    matches!(
        name,
        "aphase_hubbard.bin" | "transformation_hubbard.bin" | "v_hubbard.bin"
    )
}

fn is_specialized_codec_reference_file_name(name: &str) -> bool {
    matches!(
        name,
        "jzzp.dat" | "kmesh.dat" | "leg1.dat" | "magic.dat" | "specfunct.dat"
    )
}

/// Exact FEFF-generated artifacts for which no lossless parser exists yet.
/// Every exception carries its own format reason so filename accounting cannot
/// silently grow into a generic skip list.
fn known_unparsed_reference_file_reason(name: &str) -> Option<&'static str> {
    match name {
        "apl.dat" => Some(
            "SFCONV apl.dat has a renderer for derived pole data but no independent file parser",
        ),
        "crit.dat" => Some("KSPACE convergence scratch table has no public FEFF file contract"),
        "files.dat" => Some("FINITE_T internal filename manifest has no reusable data schema"),
        "fort.66" => Some("MPSE retained Fortran scratch unit has no stable public format"),
        "mbheader.temp" => Some("MPSE transient header fragment is not a standalone data format"),
        "opconsCu.dat" => Some(
            "atomic three-column epsilon table is distinct from the eight-column opcons.dat codec",
        ),
        "phase.dat" => Some("legacy text phase diagnostic is distinct from the phase.bin codec"),
        "ratio.dat" => Some("FINITE_T internal convergence-ratio scratch table is undocumented"),
        "s2_em.dat" => Some("DEBYE equation-of-motion diagnostic table has no output codec"),
        "s2_rm1.dat" => Some("DEBYE recursion diagnostic table has no output codec"),
        "s2_rm2.dat" => Some("DEBYE recursion diagnostic table has no output codec"),
        "spring.dat" => {
            Some("DEBYE generated spring-network diagnostic is distinct from the spring.inp codec")
        }
        _ => None,
    }
}

fn is_ldos_spectrum_name(name: &str) -> bool {
    name.strip_prefix("ldos")
        .and_then(|suffix| suffix.strip_suffix(".dat"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_rhoc_spectrum_name(name: &str) -> bool {
    name.strip_prefix("rhoc")
        .and_then(|suffix| suffix.strip_suffix(".dat"))
        .is_some_and(|index| index.len() == 2 && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_gtr_bin_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "gtr", ".bin")
}

fn is_module_log_name(name: &str) -> bool {
    name != "log.dat" && name.starts_with("log") && name.ends_with(".dat")
}

fn is_indexed_list_dat_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "list", ".dat")
}

fn is_indexed_feff_bin_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "feff", ".bin")
}

fn is_indexed_atom_dat_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "atom", ".dat")
}

fn is_indexed_pot_dat_name(name: &str) -> bool {
    has_two_digit_stem_index(name, "pot", ".dat")
}

fn has_two_digit_stem_index(name: &str, prefix: &str, suffix: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
        .is_some_and(|index| index.len() == 2 && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_rixs_line_name(name: &str) -> bool {
    name.starts_with("herfd") && name.ends_with(".dat")
}
