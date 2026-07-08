use super::*;

#[test]
fn wpot_module_writes_potential_dat_outputs_from_bin_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;
    write_misc_dat(temp.path().join("misc.dat"), &sample_misc_dat())?;
    write_convergence_scf(
        temp.path().join("convergence.scf"),
        &sample_convergence_scf(),
    )?;
    write_convergence_scf_fine(
        temp.path().join("convergence.scf.fine"),
        &sample_convergence_scf_fine(),
    )?;
    write_fort16(temp.path().join("fort.16"), &sample_fort16())?;
    let expected_misc = read_misc_dat(temp.path().join("misc.dat"))?;
    let expected_convergence = read_convergence_scf(temp.path().join("convergence.scf"))?;
    let expected_convergence_fine =
        read_convergence_scf_fine(temp.path().join("convergence.scf.fine"))?;
    let expected_fort16 = read_fort16(temp.path().join("fort.16"))?;

    let count = wpot::run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("pot00.dat"))?
            .lines()
            .nth(4)
            .context("missing first potential data row")?,
        "    1  1.5073E-04 -7.6250E-01  1.1937E-03 -1.2200E+00 -4.4700E-01  2.7852E-03"
    );
    assert_eq!(read_misc_dat(temp.path().join("misc.dat"))?, expected_misc);
    assert_eq!(
        read_convergence_scf(temp.path().join("convergence.scf"))?,
        expected_convergence
    );
    assert_eq!(
        read_convergence_scf_fine(temp.path().join("convergence.scf.fine"))?,
        expected_convergence_fine
    );
    assert_eq!(read_fort16(temp.path().join("fort.16"))?, expected_fort16);
    Ok(())
}

#[test]
fn pot_module_alias_writes_potential_dat_outputs_from_bin_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;

    run_module("pot", temp.path().join("feff.inp"))?;

    assert_eq!(
        std::fs::read_to_string(temp.path().join("pot00.dat"))?
            .lines()
            .nth(4)
            .context("missing first potential data row")?,
        "    1  1.5073E-04 -7.6250E-01  1.1937E-03 -1.2200E+00 -4.4700E-01  2.7852E-03"
    );
    Ok(())
}

#[test]
fn atomic_module_alias_validates_cached_apot_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_minimal_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;
    let expected = read_apot_bin(temp.path().join("apot.bin"))?;

    run_module("atomic", input)?;

    assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected);
    Ok(())
}

#[test]
fn atomic_module_runner_validates_cached_apot_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_minimal_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;
    let expected = read_apot_bin(temp.path().join("apot.bin"))?;

    let count = atomic::run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected);
    assert!(temp.path().join("log1.dat").is_file());
    Ok(())
}

#[test]
fn atomic_module_alias_generates_source_apot_handoff_from_rdinp_geometry() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_single_potential_atomic_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;

    run_module("atomic", input)?;

    let config = refeff_io::read_config_dat(temp.path().join("config.dat"))?;
    assert_eq!(config.potential_count(), 1);
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    assert!(temp.path().join("fpf0.dat").is_file());
    assert!(temp.path().join("log1.dat").is_file());
    Ok(())
}

#[test]
fn band_module_alias_validates_cached_bandstructure_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_bandstructure_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_bandstructure_dat(
        temp.path().join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;

    run_module("band", input)?;

    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
    Ok(())
}

#[test]
fn band_module_alias_validates_source_phase_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_bandstructure_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_band_handoff_phase_bin(),
    )?;

    let error = run_module("band", input)
        .err()
        .context("BAND source phase handoff should require complete source state")?;
    assert!(
        error
            .to_string()
            .contains(band::BAND_SOURCE_REQUIREMENT_ERROR),
        "{error:?}"
    );

    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn band_module_alias_generates_kmesh_from_reciprocal_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_reciprocal_bandstructure_module_input(&input)?;
    execute_rdinp(&input, temp.path())?;

    run_module("band", input)?;

    let kmesh = refeff_io::read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert_eq!(
        kmesh.rows[0].metadata,
        Some(refeff_io::KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn band_module_alias_generates_bandstructure_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_single_potential_reciprocal_bandstructure_module_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_band_handoff_phase_bin(),
    )?;

    run_module("band", input)?;

    assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat"))?.k_point_count() > 0);
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn rixs_module_alias_validates_source_phase_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_rixs_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;

    run_module("rixs", input)?;

    assert!(!temp.path().join("rixsET.dat").exists());
    assert!(!temp.path().join("herfd.dat").exists());
    assert!(!temp.path().join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn screen_module_alias_recovers_wscrn_from_vtot_and_apot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_screen_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    let vtot = sample_vtot_dat();
    write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;

    run_module("screen", input)?;

    let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(!temp.path().join("logscreen.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_alias_generates_missing_ldos_from_rhoc() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_ldos_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    let mut rhoc = sample_ldos_dat()?;
    rhoc.header_lines.clear();
    rhoc.fermi_level_ev = None;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    run_module("ldos", input)?;

    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    assert!(temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_alias_generates_kmesh_from_reciprocal_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_reciprocal_ldos_module_input(&input)?;
    execute_rdinp(&input, temp.path())?;

    run_module("ldos", input)?;

    let kmesh = refeff_io::read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert_eq!(
        kmesh.rows[0].metadata,
        Some(refeff_io::KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert!(!temp.path().join("ldos00.dat").exists());
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn xsph_module_alias_generates_initial_emesh_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_xsph_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;

    run_module("xsph", input)?;

    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(!temp.path().join("log2.dat").exists());
    Ok(())
}

#[test]
fn band_module_runner_validates_cached_bandstructure_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_bandstructure_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_bandstructure_dat(
        temp.path().join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;

    let count = band::run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn eelsmdff_module_alias_validates_cached_mdff_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_eelsmdff_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_mdff_dat(temp.path().join("mdff.dat"), &sample_mdff_dat()?)?;
    let expected = read_mdff_dat(temp.path().join("mdff.dat"))?;

    run_module("mdff", input)?;

    assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
    Ok(())
}

#[test]
fn eelsmdff_module_runner_validates_cached_mdff_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_eelsmdff_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_mdff_dat(temp.path().join("mdff.dat"), &sample_mdff_dat()?)?;
    let expected = read_mdff_dat(temp.path().join("mdff.dat"))?;

    let count = eelsmdff::run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
    assert!(temp.path().join("logmdff.dat").is_file());
    Ok(())
}

#[test]
fn self_module_alias_validates_cached_exc_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_self_cached_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;
    let expected = read_exc_dat(temp.path().join("exc.dat"))?;

    run_module("self", input)?;

    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, expected);
    Ok(())
}

#[test]
fn path_module_roundtrips_cached_paths_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_path_cached_input(&temp.path().join("feff.inp"))?;
    let document = FeffDocument::from_input(&FeffInput::parse_file(temp.path().join("feff.inp"))?)?;
    std::fs::write(
        temp.path().join("paths.inp"),
        rdinp::paths_inp_string(&document)?,
    )?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

    let count = paths::run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        read_paths_dat(temp.path().join("paths.dat"))?,
        sample_paths_dat()
    );
    Ok(())
}

#[test]
fn opcons_module_writes_loss_and_epsilon_from_tables() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
    write_single_component_opcons_input(temp.path(), true)?;
    std::fs::write(
        temp.path().join("opconsCu.dat"),
        concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n",),
    )?;

    let count = opcons::run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let loss = parse_loss_dat(&std::fs::read_to_string(temp.path().join("loss.dat"))?)?;
    assert_eq!(loss.point_count(), 3);
    Tol::EXACT_ECHO.assert(loss.energy_ev[0], 1.0);
    Tol {
        rel: 1.0e-6,
        abs: 1.0e-6,
    }
    .assert(loss.loss[0], 0.5 / (2.0_f64.powi(2) + 0.5_f64.powi(2)));
    assert!(temp.path().join("epsilon.dat").is_file());
    Ok(())
}

#[test]
fn opcons_module_does_not_claim_malformed_table_inputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
    write_single_component_opcons_input(temp.path(), false)?;
    std::fs::write(temp.path().join("opconsCu.dat"), b"not an opcons table\n")?;

    assert!(!opcons::has_complete_table_inputs(temp.path())?);
    Ok(())
}

fn write_single_component_opcons_input(work_dir: &Path, print_eps: bool) -> Result<()> {
    let print_eps = if print_eps { "T" } else { "F" };
    std::fs::write(
        work_dir.join("opcons.inp"),
        format!("run_opcons\n T\nprint_eps\n {print_eps}\nNumDens(0:nphx)\n  1.0000000000000000\n"),
    )?;
    Ok(())
}

fn write_single_potential_atomic_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu atomic config handoff test
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    Ok(())
}

fn write_reciprocal_bandstructure_module_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu reciprocal band module run
EDGE K
BANDSTRUCTURE -5.0 10.0 0.25 2 8 T
RECIPROCAL
KMESH 8 0
TARGET 1
SGROUP 221
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

fn write_single_potential_reciprocal_bandstructure_module_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu reciprocal band module source run
EDGE K
BANDSTRUCTURE -5.0 10.0 0.25 2 8 F
RECIPROCAL
KMESH 8 0
TARGET 1
SGROUP 221
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
POTENTIALS
0 29 Cu0
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    Ok(())
}

fn write_reciprocal_ldos_module_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu reciprocal LDOS module run
LDOS -1 1 0.1 3 0
RECIPROCAL
KMESH 8 0
TARGET 1
SGROUP 221
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

#[test]
fn opcons_module_matches_feff_reference_loss_when_present() -> Result<()> {
    let Some(zip_path) = reference_opcons_zip()? else {
        require_fixture!("OPCONS reference test; Cu_OPCONS REFERENCE.zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("OPCONS reference test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["feff.inp", "opconsCu.dat"] {
        std::fs::write(
            temp.path().join(name),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
        )?;
    }
    std::fs::write(
        temp.path().join("opcons.inp"),
        concat!(
            "run_opcons\n",
            " T\n",
            "print_eps\n",
            " F\n",
            "NumDens(0:nphx)\n",
            "  8.640712681512044E-004  8.640712681512043E-002\n",
        ),
    )?;
    let expected_loss = parse_loss_dat(&String::from_utf8(unzip_reference_entry(
        &zip_path,
        "REFERENCE/loss.dat",
    )?)?)?;

    let count = opcons::run_in_dir(temp.path())?;

    let actual_loss = parse_loss_dat(&std::fs::read_to_string(temp.path().join("loss.dat"))?)?;
    assert_eq!(count, expected_loss.point_count());
    assert_eq!(actual_loss.point_count(), expected_loss.point_count());
    for ((actual_energy, expected_energy), (actual_loss, expected_loss)) in actual_loss
        .energy_ev
        .iter()
        .zip(expected_loss.energy_ev.iter())
        .zip(actual_loss.loss.iter().zip(expected_loss.loss.iter()))
    {
        Tol::REFERENCE_ENERGY.assert(*actual_energy, *expected_energy);
        Tol::REFERENCE_LOSS.assert(*actual_loss, *expected_loss);
    }
    Ok(())
}
