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

    assert_eq!(count, 1);
    assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected);
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

    assert_eq!(count, 1);
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
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
    std::fs::write(
        temp.path().join("opcons.inp"),
        concat!(
            "run_opcons\n",
            " T\n",
            "print_eps\n",
            " T\n",
            "NumDens(0:nphx)\n",
            "  1.0000000000000000\n",
        ),
    )?;
    std::fs::write(
        temp.path().join("opconsCu.dat"),
        concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n",),
    )?;

    let count = opcons::run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let loss = parse_loss_dat(&std::fs::read_to_string(temp.path().join("loss.dat"))?)?;
    assert_eq!(loss.point_count(), 3);
    assert_close(loss.energy_ev[0], 1.0, 1.0e-12);
    assert_close(
        loss.loss[0],
        0.5 / (2.0_f64.powi(2) + 0.5_f64.powi(2)),
        1.0e-6,
    );
    assert!(temp.path().join("epsilon.dat").is_file());
    Ok(())
}

#[test]
fn opcons_module_matches_feff_reference_loss_when_present() -> Result<()> {
    let Some(zip_path) = reference_opcons_zip()? else {
        eprintln!("skipping OPCONS reference test; Cu_OPCONS REFERENCE.zip not found");
        return Ok(());
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        eprintln!("skipping OPCONS reference test; unzip command not found");
        return Ok(());
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
        assert_close(*actual_energy, *expected_energy, 2.0e-6);
        assert_close(*actual_loss, *expected_loss, 2.0e-5);
    }
    Ok(())
}
