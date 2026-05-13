use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail, ensure};
use refeff_io::{
    ApotBinPayload, ApotBinValue, AtomsDat, BandInput, ComptonInput, ConfigInput, CrpaInput,
    DensityInput, DimensionsDat, DmdwInput, EelsInput, FeffDocument, FeffInput, Ff2xInput,
    FmsInput, FullSpectrumInput, GenfmtInput, GeomDat, GlobalInput, GridInput, HubbardInput,
    LdosInput, OpconsInput, PathsInput, PotInput, ReciprocalInput, RixsInput, ScreenInput,
    SfconvInput, SpringInput, XsphInput, band_input_string, crpa_input_string, expand_cif_cluster,
    fullspectrum_input_string, hubbard_input_string, opcons_input_string, parse_chemical_dat,
    parse_chi_dat, parse_cif, parse_compton_dat, parse_config_dat, parse_contour_dat,
    parse_convergence_scf, parse_convergence_scf_fine, parse_crpa_dat, parse_curve_dat,
    parse_danes_dat, parse_dmdw_out, parse_dym, parse_edges_dat, parse_eels_dat, parse_emesh_dat,
    parse_feff_bin, parse_feffl_bin, parse_fms_bin, parse_fmsl_bin, parse_fort11, parse_fort16,
    parse_fpf0_dat, parse_gtr_dat, parse_gtrl_dat, parse_highz_out, parse_ldos_dat, parse_list_dat,
    parse_log_dat, parse_loss_dat, parse_misc_dat, parse_module_log_dat, parse_mpse_dat,
    parse_paths_dat, parse_phase_bin, parse_pot_bin, parse_prexmu_dat, parse_residue_dat,
    parse_rhoc_dat, parse_rhozzp_dat, parse_rixs_line, parse_rixs_map, parse_run_stderr,
    parse_run_stdout, parse_vtot_dat, parse_wscrn_dat, parse_xmu_dat, parse_xmul_dat,
    parse_xscorr_raw_dat, parse_xsecl_bin, parse_xsecl_dat, parse_xsecl2_dat, parse_xsect_dat,
    rdinp, read_apot_bin, read_emesh_bin, read_gg_bin, read_gg_dat, read_gtr_bin,
    reciprocal_input_string, screen_input_string,
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
        if input
            .components()
            .any(|component| component.as_os_str() == "HIGHZ")
        {
            let error = FeffDocument::from_input(&parsed).err().with_context(|| {
                format!("HIGHZ template should be invalid: {}", input.display())
            })?;
            ensure!(
                error.to_string().contains("XXX"),
                "unexpected HIGHZ template error for {}: {error}",
                input.display()
            );
        } else {
            FeffDocument::from_input(&parsed)
                .with_context(|| format!("failed to extract {}", input.display()))?;
        }
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

    let mpse_dat = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/src/SELF/mpse.dat");
    if mpse_dat.exists() {
        let text = std::fs::read_to_string(&mpse_dat)
            .with_context(|| format!("failed to read {}", mpse_dat.display()))?;
        parse_mpse_dat(&text).with_context(|| format!("failed to parse {}", mpse_dat.display()))?;
    }
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
        ensure_supported_reference_rdinp_outputs_are_rendered(output_dir, outputs.keys())?;

        let generated_periodic_structure = parsed.card("CIF").is_some()
            || (parsed.card("RECIPROCAL").is_some() && parsed.card("LATTICE").is_some());
        for (name, actual) in outputs {
            let expected_path = output_dir.join(name.as_ref());
            if !expected_path.exists() {
                bail!(
                    "unexpected generated rdinp output {name} for {}",
                    input.display()
                );
            }
            let expected = std::fs::read_to_string(&expected_path)
                .with_context(|| format!("failed to read {}", expected_path.display()))?;

            if generated_periodic_structure && matches!(name.as_ref(), "atoms.dat" | "geom.dat") {
                // Periodic equal-distance shells are sensitive to FEFF's compiler-level
                // floating-point tie order. Keep this semantic until that ordering is
                // reproduced byte-for-byte for CIF and reciprocal LATTICE expansion.
                ensure_periodic_structure_matches(&name, &expected_path, &expected, &actual)
                    .with_context(|| {
                        format!("{name} structural mismatch for {}", input.display())
                    })?;
                compared += 1;
                continue;
            }

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
fn matches_generated_reference_rdinp_log_dat_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated log.dat comparison; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input_path in inputs {
        let output_dir = input_path
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input_path.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }
        let expected_path = output_dir.join("log.dat");
        if !expected_path.exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input_path)
            .with_context(|| format!("failed to parse {}", input_path.display()))?;
        let document = FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input_path.display()))?;
        let actual = rdinp::rdinp_log_dat_string(&document)
            .with_context(|| format!("failed to render log.dat for {}", input_path.display()))?;
        let expected = std::fs::read_to_string(&expected_path)
            .with_context(|| format!("failed to read {}", expected_path.display()))?;

        ensure!(
            actual == expected,
            "log.dat mismatch for {}",
            input_path.display()
        );
        compared += 1;
    }

    ensure!(compared > 0, "no generated log.dat examples found");
    Ok(())
}

#[test]
fn matches_generated_reference_rdinp_error_log_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated rdinp error-log comparison; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input_path in inputs {
        let output_dir = input_path
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input_path.display()))?;
        if !output_dir.join(".feff.error").exists() {
            continue;
        }
        let expected_path = output_dir.join("log.dat");
        if !expected_path.exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input_path)
            .with_context(|| format!("failed to parse {}", input_path.display()))?;
        let error = FeffDocument::from_input(&parsed).err().with_context(|| {
            format!(
                "expected rdinp extraction to fail for {}",
                input_path.display()
            )
        })?;
        let actual = rdinp::rdinp_error_log_string(&parsed, &error).with_context(|| {
            format!(
                "failed to render rdinp error log for {}",
                input_path.display()
            )
        })?;
        let expected = std::fs::read_to_string(&expected_path)
            .with_context(|| format!("failed to read {}", expected_path.display()))?;

        ensure!(
            actual == expected,
            "rdinp error log mismatch for {}",
            input_path.display()
        );
        compared += 1;
    }

    ensure!(compared > 0, "no generated rdinp error-log examples found");
    Ok(())
}

#[test]
fn matches_generated_reference_rdinp_stdout_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated rdinp stdout comparison; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input_path in inputs {
        let output_dir = input_path
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input_path.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }
        let expected_path = output_dir.join("rdinp.stdout");
        if !expected_path.exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input_path)
            .with_context(|| format!("failed to parse {}", input_path.display()))?;
        let document = FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input_path.display()))?;
        let actual = rdinp::rdinp_stdout_string(&document).with_context(|| {
            format!("failed to render rdinp stdout for {}", input_path.display())
        })?;
        let expected = std::fs::read_to_string(&expected_path)
            .with_context(|| format!("failed to read {}", expected_path.display()))?;

        ensure!(
            actual == expected,
            "rdinp stdout mismatch for {}",
            input_path.display()
        );
        compared += 1;
    }

    ensure!(compared > 0, "no generated rdinp stdout examples found");
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
        parsed_count +=
            parse_handoff_file(output_dir, "config.dat", |_, text| parse_config_dat(text))?;
        parsed_count += parse_handoff_file(output_dir, "crpa.inp", CrpaInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "density.inp", DensityInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "dmdw.inp", DmdwInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "feff.dym", |_, text| parse_dym(text))?;
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
fn parses_generated_reference_cif_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated CIF parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut cif_inputs = Vec::new();
    collect_extension_files(&golden_dir, "cif", &mut cif_inputs)?;
    cif_inputs.sort();
    ensure!(!cif_inputs.is_empty(), "no generated CIF inputs found");

    for path in cif_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_cif(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.cell.a > 0.0 && parsed.cell.b > 0.0 && parsed.cell.c > 0.0,
            "{} has non-positive cell length",
            path.display()
        );
        ensure!(
            parsed.space_group_number.is_some() || parsed.space_group_hm.is_some(),
            "{} has no space-group metadata",
            path.display()
        );
        ensure!(
            !parsed.symmetry_operations.is_empty(),
            "{} has no symmetry operations",
            path.display()
        );
        ensure!(
            !parsed.atom_sites.is_empty(),
            "{} has no atom sites",
            path.display()
        );
    }

    Ok(())
}

#[test]
fn expands_generated_reference_cif_clusters_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated CIF cluster coverage; reference-work/golden not found");
        return Ok(());
    }

    struct Case {
        name: &'static str,
        cif_name: &'static str,
        target: usize,
        rmax: f64,
        potentials: &'static [(i32, i32, &'static str, bool)],
    }

    let cases = [
        Case {
            name: "CRPA",
            cif_name: "Ce-Cerium.cif",
            target: 1,
            rmax: 5.0,
            potentials: &[(0, 58, "Ce", true), (1, 58, "Ce", false)],
        },
        Case {
            name: "HUBBARD/CeO2",
            cif_name: "CeO2.cif",
            target: 2,
            rmax: 8.0,
            potentials: &[(0, 8, "O", true), (1, 58, "Ce", false), (2, 8, "O", false)],
        },
        Case {
            name: "KSPACE/Cr2GeC",
            cif_name: "Cr2GeC.cif",
            target: 3,
            rmax: 6.0,
            potentials: &[
                (0, 32, "Ge", true),
                (1, 6, "C", false),
                (2, 24, "Cr", false),
                (3, 32, "Ge", false),
            ],
        },
    ];

    let mut compared = 0_usize;
    for case in &cases {
        let case_dir = golden_dir.join(case.name);
        let cif_path = case_dir.join(case.cif_name);
        let atoms_path = case_dir.join("atoms.dat");
        if !cif_path.exists() || !atoms_path.exists() {
            continue;
        }

        let cif_text = std::fs::read_to_string(&cif_path)
            .with_context(|| format!("failed to read {}", cif_path.display()))?;
        let cif = parse_cif(&cif_text)
            .with_context(|| format!("failed to parse {}", cif_path.display()))?;
        let cluster = expand_cif_cluster(&cif, case.target, case.rmax)
            .with_context(|| format!("failed to expand {}", cif_path.display()))?;

        let atoms_text = std::fs::read_to_string(&atoms_path)
            .with_context(|| format!("failed to read {}", atoms_path.display()))?;
        let expected = AtomsDat::parse_str(atoms_path.clone(), &atoms_text)
            .with_context(|| format!("failed to parse {}", atoms_path.display()))?;

        ensure!(
            expected.natx == expected.atoms.len(),
            "{} atoms.dat natx does not match row count",
            case.name
        );
        ensure!(
            cluster.atoms.len() == expected.atoms.len(),
            "{} CIF cluster row count mismatch: expected {}, got {}",
            case.name,
            expected.atoms.len(),
            cluster.atoms.len()
        );
        ensure!(
            cluster.atoms.first().is_some_and(|atom| atom.potential == 0
                && rounded(atom.x) == 0
                && rounded(atom.y) == 0
                && rounded(atom.z) == 0),
            "{} CIF cluster does not start with the absorber at the origin",
            case.name
        );
        ensure!(
            cluster.atoms.windows(2).all(|pair| {
                cluster_atom_distance(pair[0]) <= cluster_atom_distance(pair[1]) + 1.0e-9
            }),
            "{} CIF cluster is not sorted by distance",
            case.name
        );

        let mut actual_atoms = cluster
            .atoms
            .iter()
            .copied()
            .map(rounded_cluster_atom_key)
            .collect::<Vec<_>>();
        let mut expected_atoms = expected
            .atoms
            .iter()
            .copied()
            .map(rounded_atoms_dat_key)
            .collect::<Vec<_>>();
        actual_atoms.sort_unstable();
        expected_atoms.sort_unstable();
        ensure!(
            actual_atoms == expected_atoms,
            "{} CIF cluster atom set differs from atoms.dat",
            case.name
        );

        let actual_potentials = cluster
            .potentials
            .iter()
            .map(|potential| {
                (
                    potential.ipot,
                    potential.atomic_number,
                    potential.label.as_str(),
                    potential.absorber,
                )
            })
            .collect::<Vec<_>>();
        ensure!(
            actual_potentials.as_slice() == case.potentials,
            "{} CIF potential metadata mismatch",
            case.name
        );

        compared += 1;
    }

    ensure!(
        compared == cases.len(),
        "expected {} CIF cluster cases, compared {compared}",
        cases.len()
    );
    Ok(())
}

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
            && chi_spectra.is_empty()
            && eels_spectra.is_empty()
            && danes_spectra.is_empty()
            && ldos_spectra.is_empty()
            && rhoc_spectra.is_empty()
            && compton_spectra.is_empty()
            && rhozzp_spectra.is_empty()
            && crpa_spectra.is_empty()
            && loss_spectra.is_empty()
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
        parse_xmu_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &chi_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_chi_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &eels_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_eels_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &danes_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_danes_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &ldos_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_ldos_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &rhoc_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        let parsed = parse_rhoc_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
        ensure!(
            parsed.density.ncols() == 4,
            "{} has an unexpected rhoc density width",
            spectrum.display()
        );
    }
    for spectrum in &compton_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_compton_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &rhozzp_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_rhozzp_dat(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &crpa_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_crpa_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &loss_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_loss_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &mpse_spectra {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_mpse_dat(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &rixs_maps {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_rixs_map(&text).with_context(|| format!("failed to parse {}", spectrum.display()))?;
    }
    for spectrum in &rixs_lines {
        let text = std::fs::read_to_string(spectrum)
            .with_context(|| format!("failed to read {}", spectrum.display()))?;
        parse_rixs_line(&text)
            .with_context(|| format!("failed to parse {}", spectrum.display()))?;
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
    }
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
        parsed_count += 1;
    }
    for path in &chemical_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        parse_chemical_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
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
        parsed_count += 1;
    }
    for path in &residue_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_residue_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        parsed_count += 1;
    }
    for path in &contour_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_contour_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        parsed_count += 1;
    }
    for path in &curve_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_curve_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
        parsed_count += 1;
    }
    for path in &raw_files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = parse_xscorr_raw_dat(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
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
        parsed_count += 1;
    }
    for path in &vtot_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_vtot_dat(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(parsed.row_count() >= 1, "{} has no rows", path.display());
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
    }
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
        parse_convergence_scf(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        parsed_count += 1;
    }
    for path in &fine_outputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        parse_convergence_scf_fine(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
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
        parsed_count += 1;
    }

    ensure!(
        parsed_count > 0,
        "no generated potential diagnostics parsed"
    );
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

fn is_chi_spectrum_name(name: &str) -> bool {
    name == "chi.dat"
        || name
            .strip_prefix("chip")
            .and_then(|suffix| suffix.strip_suffix(".dat"))
            .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
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

fn has_two_digit_stem_index(name: &str, prefix: &str, suffix: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
        .is_some_and(|index| index.len() == 2 && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_rixs_line_name(name: &str) -> bool {
    name.starts_with("herfd") && name.ends_with(".dat")
}
