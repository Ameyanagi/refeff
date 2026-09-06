use super::{
    has_cached_atomic_output, has_supported_atomic_source_handoff, has_supported_config_handoff,
    run_in_dir,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::{Complex32, Complex64};
use refeff_core::{
    BroydenWorkspace, GridError, NormanRadiusInput, PotScfContourRunStatus,
    PotScfContourSourceRows, PotScfIterationStatus, PotScfOuterIterationStatus, PotScfState,
    ScmtEnergyGrid, norman_radius_from_density,
};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    APOT_ATOMIC_COULOMB_SECTION_NUMBER, APOT_ATOMIC_DENSITY_SECTION_NUMBER,
    APOT_ATOMIC_KAPPA_SECTION_NUMBER, APOT_ATOMIC_NORB_SECTION_NUMBER,
    APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER, APOT_ATOMIC_ORBITAL_SECTION_START,
    APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER, APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection, ApotBinType,
    ApotBinValue, ConfigDatData, ConfigDatPotential, FEFF_BOHR_ANGSTROM, FeffDocument, FeffInput,
    Fpf0DatData, Fpf0Oscillator, GeomDat, GeomDatRow, ModuleLogData, MtdpData, PotBinData,
    PotBinScalars, PotInput, PotOverlapShell, PotScfFmsSourceGridHandoff,
    PotScfFovrgSourceGridHandoff, apot_bin_string, apot_core_hole_coulomb_from_density,
    apot_core_hole_radii, geom_dat_string, parse_apot_bin, pot_bin_string,
    potential_dat_outputs_from_bins, rdinp, read_apot_bin, read_config_dat, read_fort16,
    read_fpf0_dat, read_module_log_dat, read_pot_bin, write_apot_bin, write_config_dat,
    write_fpf0_dat, write_module_log_dat, write_mtdp, write_pot_bin,
};
use std::path::{Path, PathBuf};

#[test]
fn atomic_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 0)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_atomic_output(temp.path())?);
    Ok(())
}

#[test]
fn atomic_module_does_not_advertise_apot_sidecar_without_pot_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

    assert!(!has_cached_atomic_output(temp.path())?);
    Ok(())
}

#[test]
fn atomic_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    let apot = read_apot_bin(temp.path().join("apot.bin"))?;
    let pot = read_pot_bin(temp.path().join("pot.bin"))?;
    std::fs::write(temp.path().join("pot.inp"), "not a pot.inp handoff\n")?;

    assert!(!has_cached_atomic_output(temp.path())?);
    assert!(!has_supported_atomic_source_handoff(temp.path())?);
    assert!(!has_supported_config_handoff(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed pot.inp should fail through the explicit ATOMIC runner")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("pot.inp"), "{chain}");
    assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, apot);
    assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, pot);
    assert!(!temp.path().join("config.dat").exists());
    assert!(!temp.path().join("log1.dat").exists());
    Ok(())
}

#[test]
fn atomic_module_requires_geometry_before_source_apot_generation() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled ATOM should require source geometry")?;

    assert!(
        error
            .to_string()
            .contains("ATOM source apot.bin generation requires geom.dat handoff")
    );
    Ok(())
}

#[test]
fn atomic_module_generates_config_before_missing_geometry_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled ATOM should still require source geometry")?;

    assert!(
        error
            .to_string()
            .contains("ATOM source apot.bin generation requires geom.dat handoff")
    );
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    Ok(())
}

#[test]
fn atomic_module_generates_supported_config_handoff_without_apot_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;

    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    assert!(has_supported_config_handoff(temp.path())?);
    assert!(!has_cached_atomic_output(temp.path())?);
    Ok(())
}

#[test]
fn atomic_module_generates_supported_config_handoff_from_pot_input_without_pot_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;

    assert!(!has_supported_atomic_source_handoff(temp.path())?);
    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    assert!(has_supported_config_handoff(temp.path())?);
    assert!(!has_cached_atomic_output(temp.path())?);
    assert!(!temp.path().join("apot.bin").exists());
    assert!(!temp.path().join("log1.dat").exists());
    Ok(())
}

#[test]
fn atomic_module_generates_full_apot_from_source_handoffs_without_cached_apot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_beryllium_atomic_source_handoffs(temp.path())?;

    assert!(has_supported_atomic_source_handoff(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        1
    );
    let apot = read_apot_bin(temp.path().join("apot.bin"))?;
    assert_eq!(
        apot.sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        (1..=29).collect::<Vec<_>>()
    );
    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert!(temp.path().join("log1.dat").is_file());
    assert!(has_cached_atomic_output(temp.path())?);
    Ok(())
}

#[test]
fn atomic_module_generates_full_apot_from_geometry_source_handoff_without_pot_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_beryllium_atomic_geometry_source_handoffs(temp.path())?;

    assert!(has_supported_atomic_source_handoff(temp.path())?);
    assert!(!temp.path().join("pot.bin").exists());

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        1
    );
    let apot = read_apot_bin(temp.path().join("apot.bin"))?;
    assert_eq!(
        apot.sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        (1..=29).collect::<Vec<_>>()
    );
    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert!(temp.path().join("log1.dat").is_file());
    assert!(has_cached_atomic_output(temp.path())?);
    assert!(!temp.path().join("pot.bin").exists());
    Ok(())
}

#[test]
fn atomic_module_generates_finite_nucleus_apot_from_geometry_source_handoff_without_pot_bin()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.finite_nucleus = true;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;
    let finite_states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    assert!(
        finite_states
            .iter()
            .all(|state| state.initial_orbitals.nucleus_index > 1)
    );
    let expected = super::generated_atomic_apot_bin_from_sources(
        &super::AtomicCachePaths::new(temp.path()),
        &input,
    )?;

    assert!(has_supported_atomic_source_handoff(temp.path())?);
    assert!(!temp.path().join("pot.bin").exists());

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        1
    );
    let apot = read_apot_bin(temp.path().join("apot.bin"))?;
    assert_eq!(apot, rendered_apot_bin(&expected)?);
    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert!(temp.path().join("log1.dat").is_file());
    assert!(has_cached_atomic_output(temp.path())?);
    assert!(!temp.path().join("pot.bin").exists());
    Ok(())
}

#[test]
fn atomic_module_writes_highz_finite_nucleus_diagnostic() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.finite_nucleus = true;
    input.control.ipr1 = 5;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let atom = std::fs::read_to_string(temp.path().join("atom00.dat"))?;
    let row = atom
        .lines()
        .find(|line| line.trim_start().starts_with("1s"))
        .context("atom00.dat is missing its 1s orbital row")?;
    let binding_energy_ev = row
        .split_whitespace()
        .nth(2)
        .context("atom00.dat 1s row is missing its binding energy")?
        .parse::<f64>()
        .context("atom00.dat 1s binding energy is not numeric")?;
    let highz_reference_ev = 1.287_986e2_f64;
    let relative_error = (binding_energy_ev - highz_reference_ev).abs() / highz_reference_ev;
    assert!(
        relative_error <= 1.0e-3,
        "Be finite-nucleus 1s binding energy {binding_energy_ev:.8e} differs from the pinned FEFF HIGHZ reference {highz_reference_ev:.8e} by {relative_error:.3e}"
    );
    Ok(())
}

#[test]
fn atomic_finite_nucleus_binding_energies_match_highz_reference_range() -> Result<()> {
    let Some(report_path) = reference_highz_report() else {
        crate::require_fixture!("ATOM finite-nucleus HIGHZ reference report; source not found");
    };
    let report = std::fs::read_to_string(&report_path)
        .with_context(|| format!("failed to read {}", report_path.display()))?;

    for atomic_number in [4_usize, 29, 79, 92] {
        let reference_ev = highz_finite_binding_energy(&report, atomic_number)?;
        let input = highz_pot_input(atomic_number)?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))
            .with_context(|| format!("failed to solve HIGHZ Z={atomic_number}"))?;
        let tabulation = super::atomic_tabulation_from_state(
            states
                .first()
                .context("HIGHZ atomic solve returned no SCF state")?,
        )?;
        let actual_ev = tabulation
            .orbitals
            .iter()
            .find(|orbital| {
                orbital.principal_quantum_number == 1 && orbital.orbital_label.trim() == "s"
            })
            .context("HIGHZ atomic tabulation is missing its 1s orbital")?
            .binding_energy_ev;
        let relative_error = (actual_ev - reference_ev).abs() / reference_ev;
        assert!(
            relative_error <= 1.0e-3,
            "Z={atomic_number} finite-nucleus 1s binding energy {actual_ev:.8e} differs from pinned FEFF HIGHZ {reference_ev:.8e} by {relative_error:.3e}"
        );
    }
    Ok(())
}

#[test]
fn atomic_finite_nucleus_reports_upstream_z119_matching_failure() -> Result<()> {
    let input = highz_pot_input(119)?;
    let error = super::generated_atomic_scf_states(&input, Path::new("config.inp"))
        .err()
        .context("HIGHZ Z=119 unexpectedly completed its production SCF path")?;
    let source = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<refeff_core::AtomMathError>())
        .context("HIGHZ Z=119 failure chain has no typed atomic source")?;
    assert_eq!(
        source,
        &refeff_core::AtomMathError::ScfDiracAttemptFailed { orbital_1based: 32 },
        "unexpected HIGHZ Z=119 typed failure: {error:#}"
    );
    Ok(())
}

#[test]
fn atomic_module_generates_ybco_reference_apot_from_no_scf_pot_handoff() -> Result<()> {
    let Some(reference_dir) = reference_exafs_ybco_dir()? else {
        crate::require_fixture!("ATOM YBCO APOT source reference test; source not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let input = super::read_input(temp.path())?;
    let caches = super::AtomicCachePaths::new(temp.path());
    let pot = super::generated_no_scf_pot_bin_from_sources(&caches, &input)
        .context("failed to prepare YBCO no-SCF pot.bin handoff")?;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;

    let apot = super::generated_atomic_apot_bin_from_sources(&caches, &input)
        .context("failed to generate YBCO apot.bin from no-SCF pot.bin handoff")?;

    assert_eq!(pot.potential_count(), 5);
    assert!(apot.sections.len() >= 29);
    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
        (POT_BIN_RADIAL_POINTS, 6)
    );
    potential_dat_outputs_from_bins(&pot, &apot)?;
    Ok(())
}

#[test]
fn atomic_no_scf_cu_istprm_scalars_match_feff_reference() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        crate::require_fixture!(
            "POT no-SCF Cu ISTPRM regression; generated EXAFS/Cu reference not found"
        );
    };
    if !reference_dir.join("geom.dat").is_file() || !reference_dir.join("pot.bin").is_file() {
        crate::require_fixture!(
            "POT no-SCF Cu ISTPRM regression; geom.dat or pot.bin reference not found"
        );
    }

    let pot_path = reference_dir.join("pot.inp");
    let input = PotInput::parse_str(&pot_path, &std::fs::read_to_string(&pot_path)?)?;
    let geom = super::read_geom_dat(&reference_dir.join("geom.dat"))?;
    let actual = super::generated_no_scf_pot_bin(&input, &reference_dir.join("config.inp"), &geom)?;
    let expected = read_pot_bin(reference_dir.join("pot.bin"))?;

    for potential in 0..expected.potential_count() {
        for (label, actual_value, expected_value) in [
            (
                "muffin-tin radius",
                actual.muffin_tin_radii[potential],
                expected.muffin_tin_radii[potential],
            ),
            (
                "overlap factor",
                actual.overlap_factors[potential],
                expected.overlap_factors[potential],
            ),
            (
                "maximum overlap factor",
                actual.max_overlap_factors[potential],
                expected.max_overlap_factors[potential],
            ),
        ] {
            let tolerance = 1.0e-10_f64.max(1.0e-5 * expected_value.abs());
            assert_close(
                actual_value,
                expected_value,
                tolerance,
                &format!("{label} potential {potential}"),
            );
        }
    }
    for (label, actual_value, expected_value) in [
        (
            "Fermi level",
            actual.scalars.fermi_level,
            expected.scalars.fermi_level,
        ),
        (
            "interstitial density",
            actual.scalars.interstitial_density,
            expected.scalars.interstitial_density,
        ),
    ] {
        let tolerance = 1.0e-10_f64.max(1.0e-5 * expected_value.abs());
        assert_close(actual_value, expected_value, tolerance, label);
    }
    assert_eq!(
        actual.norman_charges, expected.norman_charges,
        "nscmt=0 must preserve FEFF's zero-initialized qnrm output"
    );
    assert_close(
        actual.scalars.total_volume,
        expected.scalars.total_volume,
        0.0,
        "nscmt=0 non-positive totvol",
    );
    Ok(())
}

#[test]
fn atomic_module_preserves_no_scf_qnrm_and_totvol_output_conventions() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 0;
    input.scattering.totvol = -0.0;
    let geom = beryllium_single_potential_geom_dat();
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;

    let no_volume = super::generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
        &input,
        Path::new("config.inp"),
        &geom,
        &states,
        None,
    )?;
    assert!(
        no_volume.norman_charges.iter().all(|charge| *charge == 0.0),
        "FEFF leaves qnrm at its zero initialization when nscmt=0"
    );
    assert_eq!(
        no_volume.scalars.total_volume.to_bits(),
        (-0.0_f64).to_bits(),
        "derived ISTPRM volume must not replace signed non-positive totvol"
    );

    input.scattering.totvol = 12.5;
    let positive_volume = super::generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
        &input,
        Path::new("config.inp"),
        &geom,
        &states,
        None,
    )?;
    let expected = super::pot_input_total_volume_bohr3(&input)?;
    assert_close(
        positive_volume.scalars.total_volume,
        expected,
        1.0e-12,
        "positive totvol Angstrom-to-Bohr conversion",
    );
    assert!(positive_volume.scalars.total_volume > input.scattering.totvol);
    assert!(
        positive_volume
            .norman_charges
            .iter()
            .all(|charge| *charge == 0.0)
    );
    Ok(())
}

#[test]
fn atomic_module_derives_no_scf_pot_multiplicities_from_pot_input() -> Result<()> {
    let mut input = copper_two_potential_pot_input()?;
    input.potentials[0].xnatph = 100.0;
    input.potentials[1].xnatph = 200.0;

    let multiplicities = super::generated_pot_potential_multiplicities(&input, 2)?;

    assert_eq!(multiplicities.to_vec(), vec![100.0, 200.0]);
    Ok(())
}

#[test]
fn atomic_module_derives_scf_pot_multiplicities_from_pot_input() -> Result<()> {
    let mut input = copper_two_potential_pot_input()?;
    input.run.nscmt = 4;
    input.potentials[0].xnatph = 100.0;
    input.potentials[1].xnatph = 200.0;

    let multiplicities = super::generated_pot_potential_multiplicities(&input, 2)?;

    assert_eq!(multiplicities.to_vec(), vec![100.0, 200.0]);
    Ok(())
}

#[test]
fn atomic_module_rejects_invalid_pot_input_multiplicities() -> Result<()> {
    let mut input = copper_two_potential_pot_input()?;
    input.run.nscmt = 0;

    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        input.potentials[1].xnatph = invalid;
        let error = super::generated_pot_potential_multiplicities(&input, 2)
            .err()
            .context("invalid xnatph should fail closed")?;
        assert!(
            error
                .to_string()
                .contains("POT xnatph for potential 1 must be positive and finite"),
            "{error:#}"
        );
    }
    Ok(())
}

#[test]
fn atomic_module_selects_only_the_implemented_feff_thermal_scf_method() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.thermal.scf_temperature = 0.0;
    input.thermal.iscfth = 1;
    assert!(!super::scf_pot_uses_thermal_occupations(&input)?);

    input.thermal.scf_temperature = 0.025_852_026_9;
    let error = super::scf_pot_uses_thermal_occupations(&input)
        .err()
        .context("positive-temperature Sommerfeld SCF should fail closed")?;
    assert!(error.to_string().contains("Sommerfeld"), "{error:#}");

    input.thermal.iscfth = 2;
    assert!(super::scf_pot_uses_thermal_occupations(&input)?);

    input.thermal.iscfth = 0;
    let error = super::scf_pot_uses_thermal_occupations(&input)
        .err()
        .context("unknown positive-temperature SCF method should fail closed")?;
    assert!(error.to_string().contains("unsupported"), "{error:#}");
    Ok(())
}

#[test]
fn atomic_module_builds_feff_thermal_grids_at_ag_reference_temperatures() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.thermal.negrid = 400;
    input.thermal.emaxscf = 5.0;
    let mut pot = beryllium_single_potential_pot_bin();
    pot.scalars.core_valence_energy = -40.0 / refeff_core::FEFF_HARTREE_EV;
    pot.scalars.fermi_level = -2.345 / refeff_core::FEFF_HARTREE_EV;

    for (temperature_ev, expected_poles) in
        [(0.025_852_026_9, 26), (0.861_734_23, 1), (1.723_47, 1)]
    {
        input.thermal.scf_temperature = temperature_ev;
        let grid = super::scf_pot_thermal_grid(&input, &pot, pot.scalars.fermi_level)?;
        assert_eq!(grid.energies.len(), 400);
        assert_eq!(grid.pole_count, expected_poles);
        assert_eq!(
            grid.energies[0].re.to_bits(),
            pot.scalars.core_valence_energy.to_bits()
        );
        assert!(
            grid.energies[399].re
                > pot.scalars.fermi_level
                    + super::POT_THERMAL_INTERPOLATION_WINDOW * temperature_ev
                        / refeff_core::FEFF_HARTREE_EV
        );
    }
    Ok(())
}

#[test]
fn atomic_module_fails_closed_on_thermal_grid_and_iteration_limits() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.thermal.scf_temperature = 0.025_852_026_9;
    input.thermal.negrid = super::POT_THERMAL_VERTICAL_POINTS as i32;
    let pot = beryllium_single_potential_pot_bin();
    let error = super::scf_pot_thermal_grid(&input, &pot, pot.scalars.fermi_level)
        .err()
        .context("undersized thermal grid should fail closed")?;
    assert!(error.to_string().contains("negrid"));

    input.thermal.negrid = 400;
    input.thermal.nmu = 0;
    let error = super::scf_pot_thermal_chemical_iteration_count(&input)
        .err()
        .context("zero thermal chemical iterations should fail closed")?;
    assert!(error.to_string().contains("nmu 0"));

    input.thermal.nmu = super::POT_THERMAL_MAX_CHEMICAL_ITERATIONS as i32 + 1;
    assert!(
        super::scf_pot_thermal_chemical_iteration_count(&input).is_err(),
        "excessive thermal chemical iteration count should fail closed"
    );
    Ok(())
}

#[test]
fn atomic_module_enforces_explicit_matsubara_resource_boundary() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.thermal.negrid = 400;
    input.thermal.emaxscf = 5.0;
    let pot = beryllium_single_potential_pot_bin();
    let maximum = super::POT_THERMAL_MAX_MATSUBARA_POLES as f64;

    let supported_ratio = maximum - 0.25;
    input.thermal.scf_temperature = super::POT_THERMAL_MAX_IMAGINARY_HARTREE
        / (2.0 * std::f64::consts::PI * supported_ratio)
        * refeff_core::FEFF_HARTREE_EV;
    let grid = super::scf_pot_thermal_grid(&input, &pot, pot.scalars.fermi_level)?;
    assert_eq!(grid.pole_count, super::POT_THERMAL_MAX_MATSUBARA_POLES);

    let unsupported_ratio = maximum + 0.25;
    input.thermal.scf_temperature = super::POT_THERMAL_MAX_IMAGINARY_HARTREE
        / (2.0 * std::f64::consts::PI * unsupported_ratio)
        * refeff_core::FEFF_HARTREE_EV;
    let error = super::scf_pot_thermal_grid(&input, &pot, pot.scalars.fermi_level)
        .err()
        .context("one-pole-over-limit thermal grid should fail closed")?;
    let message = error.to_string();
    assert!(message.contains("Matsubara pole count"), "{error:#}");
    assert!(message.contains("explicit resource limit"), "{error:#}");
    assert!(message.contains("supported minimum"), "{error:#}");
    Ok(())
}

#[test]
fn atomic_module_accepts_feff_thermal_chemical_potential_plateau() -> Result<()> {
    let current = 0.0;
    assert!(super::scf_pot_thermal_chemical_update_stalled(
        current, current
    )?);
    assert!(super::scf_pot_thermal_chemical_update_stalled(
        current,
        current + 0.5 * super::POT_THERMAL_CHEMICAL_STALL_HARTREE
    )?);
    assert!(!super::scf_pot_thermal_chemical_update_stalled(
        current,
        current + 2.0 * super::POT_THERMAL_CHEMICAL_STALL_HARTREE
    )?);
    assert!(super::scf_pot_thermal_chemical_update_stalled(current, f64::NAN).is_err());
    Ok(())
}

#[test]
fn atomic_module_builds_iterative_pot_scf_initial_state_from_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 4;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;

    assert_eq!(initial.pot.potential_count(), 1);
    assert_eq!(
        initial.last_indices,
        super::scf_pot_rholie_last_indices(&initial.pot)?
    );
    assert_eq!(
        initial.state.fermi_energy,
        initial.istprm.fermi.chemical_potential
    );
    assert_eq!(initial.pot.scalars.fermi_level, initial.state.fermi_energy);
    assert!(initial.pot.scalars.interstitial_density > 0.0);
    assert_eq!(initial.pot.total_potential, initial.istprm.total_potential);
    assert_eq!(
        initial.state.overlapped_density,
        initial.pot.electron_density
    );
    assert!(initial.energy_grid.active_len > 0);
    assert_eq!(
        initial.energy_grid.energies.len(),
        super::POT_SCMT_MAX_ENERGY_POINTS
    );
    assert_eq!(initial.energy_grid.steps.len(), super::POT_SCMT_FLOOR_COUNT);
    assert_eq!(
        initial.energy_grid.energies[0].re,
        initial.pot.scalars.core_valence_energy
    );
    let fovrg_grid = initial.fovrg_grid.as_ref().unwrap_or_else(|| {
        panic!(
            "missing FOVRG source grid: {}",
            initial
                .fovrg_grid_unavailable
                .as_deref()
                .unwrap_or("missing unavailable reason")
        )
    });
    assert!(initial.fovrg_grid_unavailable.is_none());
    assert_eq!(fovrg_grid.rholie_active_counts, initial.last_indices);
    assert!(
        fovrg_grid
            .radial_active_counts
            .iter()
            .zip(fovrg_grid.rholie_active_counts.iter())
            .all(|(radial_count, rholie_count)| radial_count >= rholie_count)
    );
    let source_energy_count = fovrg_grid.energies_hartree.len();
    assert!(source_energy_count > 0);
    assert_eq!(
        fovrg_grid.wave_numbers.dim(),
        (source_energy_count, initial.pot.potential_count())
    );
    assert_eq!(fovrg_grid.regular_large.dim().0, source_energy_count);
    assert_eq!(
        fovrg_grid.regular_large.dim().1,
        initial.pot.potential_count()
    );
    assert!(fovrg_grid.regular_large.dim().2 > 0);
    assert_eq!(
        fovrg_grid.regular_large.dim().3,
        fovrg_grid.source_radii.len()
    );
    if let Some(fms_grid) = &initial.fms_grid {
        assert!(initial.fms_grid_unavailable.is_none());
        assert_eq!(fms_grid.energies_hartree.len(), source_energy_count);
        assert_eq!(fms_grid.scattering_trace.dim().0, source_energy_count);
        assert_eq!(
            fms_grid.scattering_trace.dim().2,
            initial.pot.potential_count()
        );
    } else {
        let reason = initial
            .fms_grid_unavailable
            .as_deref()
            .expect("missing FMS source-grid reason");
        assert!(!reason.is_empty());
    }
    if let Some(contour_rows) = &initial.contour_rows {
        assert!(initial.contour_rows_unavailable.is_none());
        assert_eq!(contour_rows.source_energies.len(), source_energy_count);
        assert_eq!(contour_rows.scattering_trace.dim().0, source_energy_count);
        assert_eq!(
            contour_rows.scattering_trace.dim().2,
            initial.pot.potential_count()
        );
    } else {
        let reason = initial
            .contour_rows_unavailable
            .as_deref()
            .expect("missing contour source-row reason");
        assert!(!reason.is_empty());
    }
    if let Some(advance) = &initial.state_advance {
        assert!(initial.state_advance_unavailable.is_none());
        assert_eq!(
            advance.iteration.contour.embedded_ldos.ncols(),
            initial.pot.potential_count()
        );
        assert_eq!(
            advance.state.overlapped_density.dim(),
            initial.state.overlapped_density.dim()
        );
    } else {
        let reason = initial
            .state_advance_unavailable
            .as_deref()
            .expect("missing state-advance reason");
        assert!(!reason.is_empty());
    }
    if let Some(next) = &initial.next_iteration {
        assert!(initial.next_iteration_unavailable.is_none());
        assert_eq!(next.iteration, 2);
        assert_eq!(next.pot.potential_count(), initial.pot.potential_count());
        assert_eq!(
            next.last_indices,
            super::scf_pot_rholie_last_indices(&next.pot)?
        );
        assert_eq!(next.pot.scalars.fermi_level, next.state.fermi_energy);
        if let Some(contour_rows) = &next.contour_rows {
            assert!(next.contour_rows_unavailable.is_none());
            assert!(!contour_rows.source_energies.is_empty());
            assert_eq!(
                contour_rows.scattering_trace.dim().2,
                next.pot.potential_count()
            );
        } else {
            let reason = next
                .contour_rows_unavailable
                .as_deref()
                .expect("missing next contour source-row reason");
            assert!(!reason.is_empty());
        }
        if let Some(advance) = &next.state_advance {
            assert!(next.state_advance_unavailable.is_none());
            assert_eq!(
                advance.iteration.contour.embedded_ldos.ncols(),
                next.pot.potential_count()
            );
            assert_eq!(
                advance.state.overlapped_density.dim(),
                next.state.overlapped_density.dim()
            );
        } else {
            let reason = next
                .state_advance_unavailable
                .as_deref()
                .expect("missing next state-advance reason");
            assert!(!reason.is_empty());
        }
    } else {
        let reason = initial
            .next_iteration_unavailable
            .as_deref()
            .expect("missing next-iteration reason");
        assert!(!reason.is_empty());
    }
    assert!(!temp.path().join("pot.bin").exists());
    assert!(!temp.path().join("apot.bin").exists());
    Ok(())
}

#[test]
fn atomic_module_imports_external_potential_as_scf_initial_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 2;
    input.external_pot = true;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    let geom = beryllium_single_potential_geom_dat();
    std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;
    let mut seed_input = input.clone();
    seed_input.run.nscmt = 0;
    seed_input.external_pot = false;
    let seed_pot = super::generated_no_scf_pot_bin(&seed_input, Path::new("config.inp"), &geom)?;
    let mtdp = sample_scf_mtdp_data(&seed_pot);
    let mtdp_path = temp.path().join("GeCl4.04.dft.mtdp");
    write_mtdp(&mtdp_path, &mtdp)?;
    let imported_mtdp = refeff_io::read_mtdp(&mtdp_path)?;
    let expected_potential_last = imported_mtdp.atom_potential[(POT_BIN_RADIAL_POINTS - 1, 0)];
    let expected_density0 = imported_mtdp.atom_density[(0, 0)];
    let expected_density2 = imported_mtdp.atom_density[(2, 0)];
    std::fs::write(temp.path().join("sort.aip"), "0\n")?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;
    assert!(super::can_prepare_scf_pot_initial_state_from_sources_in_dir(temp.path())?);

    assert!(initial.external_pot_imported);
    assert!(!initial.restart_pot_imported);
    assert_eq!(initial.pot.muffin_tin_indices[0], 7);
    assert!((initial.pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
    assert!((initial.pot.scalars.interstitial_potential + 0.75).abs() < 1.0e-10);
    assert!((initial.pot.scalars.fermi_level + 0.10).abs() < 1.0e-10);
    assert!((initial.pot.total_potential[(0, 0)] + 1.0).abs() < 1.0e-10);
    assert!((initial.pot.total_potential[(2, 0)] + 1.2).abs() < 1.0e-10);
    assert!(
        (initial.pot.total_potential[(POT_BIN_RADIAL_POINTS - 1, 0)] - expected_potential_last)
            .abs()
            < 1.0e-10
    );
    assert!((initial.pot.electron_density[(0, 0)] - expected_density0).abs() < 1.0e-10);
    assert!((initial.pot.electron_density[(2, 0)] - expected_density2).abs() < 1.0e-10);
    assert_eq!(
        initial.state.overlapped_density,
        initial.pot.electron_density
    );
    assert_eq!(initial.state.fermi_energy, initial.pot.scalars.fermi_level);
    assert!(!temp.path().join("pot.bin").exists());
    assert!(!temp.path().join("apot.bin").exists());
    Ok(())
}

#[test]
fn atomic_module_applies_start_from_file_after_external_potential_for_scf_initial_state()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 2;
    input.external_pot = true;
    input.start_from_file = true;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    let geom = beryllium_single_potential_geom_dat();
    std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;

    let mut seed_input = input.clone();
    seed_input.run.nscmt = 0;
    seed_input.external_pot = false;
    seed_input.start_from_file = false;
    let seed_pot = super::generated_no_scf_pot_bin(&seed_input, Path::new("config.inp"), &geom)?;
    write_mtdp(
        temp.path().join("GeCl4.04.dft.mtdp"),
        &sample_scf_mtdp_data(&seed_pot),
    )?;
    std::fs::write(temp.path().join("sort.aip"), "0\n")?;

    let mut restart = beryllium_single_potential_pot_bin();
    restart.scalars.fermi_level = -0.125;
    restart.scalars.interstitial_potential = -0.275;
    restart.scalars.interstitial_density = 0.019;
    restart.electron_density.fill(0.023);
    restart.total_potential.fill(-0.41);
    restart.coulomb_potential.fill(123.0);
    restart.valence_density.fill(456.0);
    write_pot_bin(temp.path().join("pot.bin"), &restart)?;
    let restart = read_pot_bin(temp.path().join("pot.bin"))?;

    assert!(super::can_prepare_scf_pot_initial_state_from_sources_in_dir(temp.path())?);

    let caches = super::AtomicCachePaths::new(temp.path());
    let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;

    assert!(initial.external_pot_imported);
    assert!(initial.restart_pot_imported);
    assert_eq!(initial.pot.muffin_tin_indices[0], 7);
    assert!((initial.pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
    assert_eq!(initial.pot.total_potential, restart.total_potential);
    assert_eq!(initial.pot.electron_density, restart.electron_density);
    assert_eq!(initial.pot.scalars.fermi_level, restart.scalars.fermi_level);
    assert_eq!(
        initial.pot.scalars.interstitial_potential,
        restart.scalars.interstitial_potential
    );
    assert_eq!(
        initial.pot.scalars.interstitial_density,
        restart.scalars.interstitial_density
    );
    assert_eq!(initial.state.overlapped_density, restart.electron_density);
    assert_eq!(initial.state.fermi_energy, restart.scalars.fermi_level);
    assert_ne!(initial.pot.coulomb_potential, restart.coulomb_potential);
    assert_ne!(initial.pot.valence_density, restart.valence_density);
    assert!(!temp.path().join("apot.bin").exists());
    Ok(())
}

#[test]
fn atomic_module_generates_finite_nucleus_scf_state_from_pot_input() -> Result<()> {
    let point_input = beryllium_pot_input()?;
    let point_states = super::generated_atomic_scf_states(&point_input, Path::new("config.inp"))?;
    let mut finite_input = point_input.clone();
    finite_input.finite_nucleus = true;

    let finite_states = super::generated_atomic_scf_states(&finite_input, Path::new("config.inp"))?;

    assert!(!finite_states.is_empty());
    assert_eq!(finite_states.len(), point_states.len());
    for (potential, (point, finite)) in point_states.iter().zip(&finite_states).enumerate() {
        assert_eq!(point.initial_orbitals.nucleus_index, 1);
        assert!(finite.initial_orbitals.nucleus_index > 1);
        assert!(
            finite.initial_orbitals.radii[0] < super::atomic_first_radius_times_charge(4) / 4.0
        );
        assert_ne!(
            finite.initial_orbitals.radii[0],
            point.initial_orbitals.radii[0]
        );
        assert_ne!(
            finite.initial_orbitals.nuclear_potential[0],
            point.initial_orbitals.nuclear_potential[0]
        );
        assert_ne!(finite.scf.density_4pi[0], point.scf.density_4pi[0]);
        assert_ne!(
            finite.scf.large_components[(0, 0)],
            point.scf.large_components[(0, 0)]
        );

        let remapped = super::atomic_apot_free_spin_density_from_state(potential, finite)?;
        let moment = finite.spin_magnetization.sum();
        let normalization = if moment > 0.0 { moment } else { 1.0 };
        let source_log_radii = finite
            .initial_orbitals
            .radii
            .iter()
            .map(|radius| radius.ln())
            .collect::<Vec<_>>();
        let native = Array1::from_shape_fn(finite.initial_orbitals.radii.len(), |row| {
            let weighted = finite
                .spin_magnetization
                .iter()
                .enumerate()
                .map(|(orbital, weight)| {
                    weight
                        * (finite.scf.large_components[(row, orbital)].powi(2)
                            + finite.scf.small_components[(row, orbital)].powi(2))
                })
                .sum::<f64>();
            weighted / normalization / finite.initial_orbitals.radii[row].powi(2)
        });
        let native = native
            .as_slice()
            .context("finite spin-density test storage is not contiguous")?;
        let target_radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);
        for &row in &[0, 31, POT_BIN_RADIAL_POINTS - 1] {
            let expected =
                refeff_core::terp(&source_log_radii, native, 3, target_radii[row].ln())?.value;
            assert_close(
                remapped[row],
                expected,
                1.0e-12 * expected.abs().max(1.0),
                &format!("finite-nucleus remapped dmag potential {potential} row {row}"),
            );
        }
    }

    let unique_count = super::apot_unique_potential_count(&finite_input)?;
    let alternate = super::atomic_apot_free_spin_density_from_state(
        unique_count,
        &finite_states[unique_count],
    )?;
    let arrays = super::atomic_apot_overlap_arrays_from_states(
        &finite_input,
        &beryllium_single_potential_static_arrays()?,
        &finite_states,
    )?;
    for &row in &[0, 31, POT_BIN_RADIAL_POINTS - 1] {
        assert_eq!(
            arrays.magnetization_density[(row, unique_count)].to_bits(),
            alternate[row].to_bits(),
            "alternate absorber must store the remapped finite-nucleus dmag"
        );
    }
    Ok(())
}

#[test]
fn atomic_module_preserves_saved_scmt_call_state_across_retries() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.finite_nucleus = true;
    input.run.nscmt = 2;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;

    let initial_advance = run
        .initial
        .state_advance
        .as_ref()
        .context("finite-nucleus initial SCF advance should be available")?;
    assert_eq!(
        initial_advance.iteration.contour.status,
        PotScfContourRunStatus::Bracketed,
        "finite-nucleus contour exhausted {} points at energy {:?} with electron delta {}",
        initial_advance.iteration.contour.energy_points_used,
        initial_advance.iteration.contour.current_energy,
        initial_advance.iteration.contour.current_electron_delta
    );
    assert!(
        initial_advance.iteration.contour.energy_points_used
            > super::POT_SCMT_MAX_ENERGY_POINTS * 2,
        "finite-nucleus contour should exercise the expanded adaptive source-row cap"
    );
    assert_eq!(
        initial_advance.outer.status,
        PotScfOuterIterationStatus::RepeatRequired
    );
    assert_eq!(
        run.final_status,
        Some(PotScfOuterIterationStatus::RepeatRequired)
    );
    assert_eq!(run.final_iteration, Some(1));
    assert!(run.prepared_iterations.is_empty());
    assert!(run.final_pot.is_none());
    assert!(run.final_apot.is_none());
    assert!(
        run.final_pot_unavailable
            .as_deref()
            .is_some_and(|reason| reason.contains("FEFF-style start attempt")),
        "missing finite-nucleus repeat retry exhaustion reason: {:?}",
        run.final_pot_unavailable
    );
    Ok(())
}

#[test]
fn atomic_module_imports_start_from_file_pot_as_scf_initial_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 2;
    input.start_from_file = true;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let mut restart = beryllium_single_potential_pot_bin();
    restart.scalars.fermi_level = -0.125;
    restart.scalars.interstitial_potential = -0.275;
    restart.scalars.interstitial_density = 0.019;
    restart.electron_density.fill(0.023);
    restart.total_potential.fill(-0.41);
    restart.coulomb_potential.fill(123.0);
    restart.valence_density.fill(456.0);
    write_pot_bin(temp.path().join("pot.bin"), &restart)?;
    let restart = read_pot_bin(temp.path().join("pot.bin"))?;

    assert!(super::can_prepare_scf_pot_initial_state_from_sources_in_dir(temp.path())?);

    let caches = super::AtomicCachePaths::new(temp.path());
    let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;

    assert!(initial.restart_pot_imported);
    assert_eq!(initial.pot.total_potential, restart.total_potential);
    assert_eq!(initial.pot.electron_density, restart.electron_density);
    assert_eq!(initial.pot.scalars.fermi_level, restart.scalars.fermi_level);
    assert_eq!(
        initial.pot.scalars.interstitial_potential,
        restart.scalars.interstitial_potential
    );
    assert_eq!(
        initial.pot.scalars.interstitial_density,
        restart.scalars.interstitial_density
    );
    assert_eq!(initial.state.overlapped_density, restart.electron_density);
    assert_eq!(initial.state.fermi_energy, restart.scalars.fermi_level);
    assert_ne!(initial.pot.coulomb_potential, restart.coulomb_potential);
    assert_ne!(initial.pot.valence_density, restart.valence_density);
    Ok(())
}

#[test]
fn atomic_module_runs_iterative_pot_scf_loop_from_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 4;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;
    let initial_fovrg = run
        .initial
        .fovrg_grid
        .as_ref()
        .expect("missing initial POT SCF FOVRG source grid");
    assert!(
        initial_fovrg
            .phase_amplitudes
            .iter()
            .all(|&value| value != Complex64::new(0.0, 0.0))
    );

    assert!(super::can_write_scf_pot_bin_from_sources_in_dir(
        temp.path()
    )?);
    assert_eq!(run.initial.pot.potential_count(), 1);
    assert!(
        run.final_status
            .is_some_and(super::scf_pot_status_has_final_pot),
        "expected final POT status, got {:?}",
        run.final_status
    );
    assert!(
        run.final_iteration
            .is_some_and(|iteration| { iteration >= 1 && iteration <= input.run.nscmt as usize })
    );
    assert!(!run.prepared_iterations.is_empty());
    assert!(
        run.terminal_unavailable.is_none(),
        "unexpected terminal POT SCF boundary: {:?}",
        run.terminal_unavailable
    );
    let (final_contour_rows, final_advance) = run
        .prepared_iterations
        .last()
        .and_then(|prepared| {
            prepared
                .contour_rows
                .as_ref()
                .zip(prepared.state_advance.as_ref())
        })
        .or_else(|| {
            run.initial
                .contour_rows
                .as_ref()
                .zip(run.initial.state_advance.as_ref())
        })
        .expect("missing final POT SCF contour source rows");
    assert!(
        final_contour_rows.source_energies.len() < super::POT_SCMT_MAX_ADAPTIVE_SOURCE_POINTS,
        "corrected core-valence contour should avoid exhausting the adaptive source grid"
    );
    assert!(!final_contour_rows.source_energies.is_empty());
    assert_eq!(
        final_advance.iteration.contour.energy_points_used,
        final_contour_rows.source_energies.len()
    );
    if let Some(iteration) = run.final_iteration {
        assert!(iteration >= 1 && iteration <= input.run.nscmt as usize);
    }
    if let Some(status) = run.final_status {
        let expected = run
            .prepared_iterations
            .last()
            .and_then(|prepared| prepared.state_advance.as_ref())
            .map(|advance| advance.outer.status)
            .or_else(|| {
                run.initial
                    .state_advance
                    .as_ref()
                    .map(|advance| advance.outer.status)
            })
            .expect("final status should have an advance source");
        assert_eq!(status, expected);
    } else {
        assert!(run.terminal_unavailable.is_some());
    }
    for (offset, prepared) in run.prepared_iterations.iter().enumerate() {
        assert_eq!(prepared.iteration, offset + 2);
        assert_eq!(
            prepared.pot.potential_count(),
            run.initial.pot.potential_count()
        );
        assert_eq!(
            prepared.last_indices,
            super::scf_pot_rholie_last_indices(&prepared.pot)?
        );
    }
    if let Some(reason) = &run.terminal_unavailable {
        assert!(!reason.is_empty());
    }
    if matches!(
        run.final_status,
        Some(
            PotScfOuterIterationStatus::Converged
                | PotScfOuterIterationStatus::ReachedIterationLimit
        )
    ) {
        let final_pot = run.final_pot.as_ref().expect("missing final pot.bin");
        assert!(run.final_pot_unavailable.is_none());
        assert_eq!(
            final_pot.electron_density.dim(),
            run.initial.pot.electron_density.dim()
        );
        assert!(final_pot.raw_text.is_none());
        assert!(!pot_bin_string(final_pot)?.is_empty());
        let final_apot = run.final_apot.as_ref().expect("missing final apot.bin");
        assert!(!apot_bin_string(final_apot)?.is_empty());
        assert!(!potential_dat_outputs_from_bins(final_pot, final_apot)?.is_empty());
    } else {
        assert!(run.final_pot.is_none());
        assert!(run.final_apot.is_none());
        match run.final_pot_unavailable.as_ref() {
            Some(reason) => assert!(!reason.is_empty()),
            None => panic!("missing final pot.bin unavailable reason"),
        }
    }
    assert!(!temp.path().join("pot.bin").exists());
    assert!(!temp.path().join("apot.bin").exists());
    Ok(())
}

#[test]
fn atomic_module_builds_true_scf_istprm_from_positive_totvol_reference() -> Result<()> {
    let Some(reference_dir) = reference_bn_true_scf_dir()? else {
        crate::require_fixture!("POT true-SCF istprm reference test; XANES/BN source not found");
    };
    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let input = super::read_input(temp.path())?;
    assert!(input.run.nscmt > 0);
    assert!(input.scattering.totvol > 0.0);

    let caches = super::AtomicCachePaths::new(temp.path());
    let geom = super::prepare_scf_pot_initial_state_source_handoff(&caches, &input)?;
    let unique_count = super::apot_unique_potential_count(&input)?;
    let static_arrays = super::atomic_apot_static_arrays_from_source_geometry(
        &input,
        &geom,
        Array1::from_elem(unique_count, 1.0),
    )?;
    let pot = super::generated_no_scf_pot_bin(&input, &caches.config_inp, &geom)?;

    let converted_volume = super::pot_input_total_volume_bohr3(&input)?;
    assert!(converted_volume > input.scattering.totvol);
    assert_close(
        pot.scalars.total_volume,
        converted_volume,
        1.0e-12,
        "positive totvol serialized in Bohr^3",
    );
    let istprm = super::scf_pot_istprm_from_initial_state(&input, &static_arrays, &pot)?;

    assert_eq!(istprm.muffin_tin_radii.len(), unique_count);
    assert!(istprm.interstitial_volume > 0.0);
    assert!(istprm.interstitial_density > 0.0);
    assert!(istprm.fermi.chemical_potential.is_finite());
    assert!(
        istprm
            .muffin_tin_radii
            .iter()
            .zip(istprm.norman_radii.iter())
            .all(|(muffin_tin, norman)| muffin_tin.is_finite()
                && *muffin_tin > 0.0
                && norman.is_finite()
                && *norman > *muffin_tin)
    );
    Ok(())
}

#[test]
fn atomic_module_preserves_bn_bounded_scf_final_valence_density() -> Result<()> {
    let Some(reference_dir) = reference_bn_true_scf_dir()? else {
        crate::require_fixture!(
            "POT BN bounded SCF valence-density test; XANES/BN source not found"
        );
    };
    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let mut input = super::read_input(temp.path())?;
    input.run.nscmt = 1;

    let caches = super::AtomicCachePaths::new(temp.path());
    let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;
    let advance = run
        .initial
        .state_advance
        .as_ref()
        .context("missing bounded BN initial SCF advance")?;
    let final_pot = run
        .final_pot
        .as_ref()
        .context("missing bounded BN final pot.bin")?;

    assert_eq!(
        run.final_status,
        Some(PotScfOuterIterationStatus::ReachedIterationLimit)
    );
    assert_close(
        advance.outer.fermi_energy * refeff_core::FEFF_HARTREE_EV,
        -10.374_905_860_627,
        1.0e-6,
        "bounded BN first SCMT Fermi energy",
    );
    assert_close(
        advance.outer.charge_distance,
        0.129_419_310_645,
        1.0e-8,
        "bounded BN first SCMT charge distance",
    );
    assert_close(
        advance.outer.partial_charge_distance,
        3.614_303_291_509,
        2.0e-8,
        "bounded BN first SCMT partial charge distance",
    );
    assert_eq!(
        run.initial.pot.scalars.plasmon_frequency,
        run.initial.pot.scalars.interstitial_density.sqrt(),
        "FEFF POT plasmon frequency is sqrt(rhoint)"
    );
    assert_eq!(
        final_pot.scalars.plasmon_frequency, run.initial.pot.scalars.plasmon_frequency,
        "FEFF freezes the pre-SCF plasmon frequency across SCMT updates"
    );
    assert_eq!(
        final_pot.norman_charges, advance.outer.reported_charge_transfer,
        "terminal pot.bin must serialize FEFF's -qnrm + xion charge transfer"
    );
    for potential in 0..final_pot.potential_count() {
        assert_close(
            final_pot.norman_charges[potential],
            -advance.state.norman_charges[potential] + run.initial.pot.ionization[potential],
            1.0e-12,
            &format!("terminal reported charge transfer potential {potential}"),
        );
    }
    assert_close(
        run.initial.pot.valence_density[(0, 0)],
        64.89293612,
        1.0e-6,
        "bounded BN initial absorber valence density",
    );
    assert_close(
        advance.iteration.overlapped_valence_density[(0, 0)],
        run.initial.pot.valence_density[(0, 0)],
        1.0e-10,
        "bounded BN SCMT iteration absorber valence density",
    );
    assert_close(
        advance.outer.overlapped_valence_density[(0, 0)],
        run.initial.pot.valence_density[(0, 0)],
        1.0e-10,
        "bounded BN outer absorber valence density",
    );
    assert_close(
        advance.state.overlapped_valence_density[(0, 0)],
        run.initial.pot.valence_density[(0, 0)],
        1.0e-10,
        "bounded BN state absorber valence density",
    );
    assert_close(
        final_pot.valence_density[(0, 0)],
        run.initial.pot.valence_density[(0, 0)],
        1.0e-10,
        "bounded BN final absorber valence density",
    );
    Ok(())
}

#[test]
fn atomic_module_reaches_core_hole_bounded_iteration_limit_from_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_core_hole_pot_input()?;
    input.run.nohole = -1;
    input.run.nscmt = 2;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;

    assert_eq!(
        run.final_status,
        Some(PotScfOuterIterationStatus::ReachedIterationLimit)
    );
    assert_eq!(run.final_iteration, Some(2));
    assert_eq!(run.prepared_iterations.len(), 1);
    let prepared = run
        .prepared_iterations
        .last()
        .context("missing prepared bounded iteration")?;
    assert_close(
        run.initial.pot.scalars.core_valence_energy,
        input.scattering.ecv / refeff_core::FEFF_HARTREE_EV,
        1.0e-12,
        "converted core-hole ecv",
    );
    assert_close(
        prepared.pot.scalars.core_valence_energy,
        run.initial.pot.scalars.core_valence_energy,
        1.0e-12,
        "prepared bounded ecv",
    );
    let advance = prepared
        .state_advance
        .as_ref()
        .context("missing bounded state advance")?;
    assert_eq!(
        advance.outer.status,
        PotScfOuterIterationStatus::ReachedIterationLimit
    );
    assert_eq!(advance.iteration.bad_occupation_count, 0);
    assert!(advance.iteration.density_step.is_some());
    assert!(run.final_pot.is_some());
    assert!(run.final_apot.is_some());
    assert!(run.final_pot_unavailable.is_none());
    assert!(run.terminal_unavailable.is_none());
    Ok(())
}

#[test]
fn atomic_module_updates_scf_retry_controls_like_feff_nstarts() -> Result<()> {
    let mut input = beryllium_core_hole_pot_input()?;
    input.scattering.ecv = -refeff_core::FEFF_HARTREE_EV;
    input.scattering.ca1 = 0.2;

    let next = super::update_scf_pot_retry_controls(&mut input, -0.8, 1)?;

    assert_eq!(next, 2);
    assert_close(
        input.scattering.ecv,
        -0.8 * refeff_core::FEFF_HARTREE_EV,
        1.0e-12,
        "changed ecv retry",
    );
    assert_close(
        input.scattering.ca1,
        0.2,
        1.0e-12,
        "unchanged first retry ca1",
    );

    let next = super::update_scf_pot_retry_controls(&mut input, -0.79, next)?;

    assert_eq!(next, super::POT_SCF_MAX_START_ATTEMPTS);
    assert_close(
        input.scattering.ecv,
        -0.79 * refeff_core::FEFF_HARTREE_EV,
        1.0e-12,
        "carried ecv retry",
    );
    assert_close(
        input.scattering.ca1,
        0.04,
        1.0e-12,
        "third-start reduced ca1",
    );

    let mut skipped = beryllium_core_hole_pot_input()?;
    skipped.scattering.ecv = -0.5 * refeff_core::FEFF_HARTREE_EV;
    skipped.scattering.ca1 = 0.04;

    let next = super::update_scf_pot_retry_controls(&mut skipped, -0.49, 1)?;

    assert_eq!(next, super::POT_SCF_MAX_START_ATTEMPTS);
    assert_close(
        skipped.scattering.ca1,
        super::POT_SCF_MIN_RETRY_MIXING,
        1.0e-12,
        "small ecv change skips to reduced mixing",
    );

    let error = super::update_scf_pot_retry_controls(
        &mut skipped,
        -0.48,
        super::POT_SCF_MAX_START_ATTEMPTS,
    )
    .err()
    .context("max FEFF-style start attempt should not advance")?;
    assert!(
        error.to_string().contains("cannot advance"),
        "unexpected max-attempt retry error: {error:?}"
    );
    Ok(())
}

#[test]
fn atomic_module_keeps_second_istprm_density_projection_positive() -> Result<()> {
    use refeff_core::{
        MuffinTinOverlapMatrixInput, MuffinTinOverlapProjectionInput,
        MuffinTinOverlapProjectionMode, muffin_tin_overlap_matrix, overlap_density_indices,
        project_muffin_tin_overlap,
    };

    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 4;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let context = super::scf_pot_source_context_from_sources(&caches, &input, true)?;
    let mut fms_cache = crate::fms::PotScfFmsPipelineCache::default();
    let initial = super::scf_pot_initial_state_from_generated_pot(
        temp.path(),
        &input,
        &context.static_arrays,
        &context.config,
        context.pot,
        context.external_source.as_ref(),
        context.restart_pot.as_ref(),
        true,
        &mut fms_cache,
    )?;
    let advance = initial
        .state_advance
        .as_ref()
        .context("missing first SCMT advance")?;
    if let Some(density_step) = &advance.iteration.density_step {
        assert!(
            density_step
                .norman_charges
                .iter()
                .all(|charge| charge.abs() <= 1.0e-10),
            "first SCMT Norman-charge history should remain neutral for single-potential Be"
        );
    }
    let unique_count = super::apot_unique_potential_count(&input)?;
    let static_arrays = &context.static_arrays;
    let atom_potentials = super::atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let atom_positions = super::atomic_apot_core_atom_positions(static_arrays)?;
    let representative_atoms =
        super::atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let explicit_overlaps = super::no_scf_pot_muffin_tin_overlaps(static_arrays, unique_count)?;
    let explicit_overlap_refs = explicit_overlaps
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let near_neighbor_flags = Array1::<bool>::from_elem(unique_count, false);
    let interstitial_selector = usize::try_from(input.run.inters)?;

    for (label, density) in [
        ("initial", initial.pot.electron_density.view()),
        ("after_scmt", advance.state.overlapped_density.view()),
    ] {
        let valence_density = if label == "initial" {
            initial.pot.valence_density.view()
        } else {
            advance.state.overlapped_valence_density.view()
        };
        let mut muffin_tin_indices = Array1::<usize>::zeros(unique_count);
        let mut norman_indices = Array1::<usize>::zeros(unique_count);
        let mut norman_radii = initial.pot.norman_radii.clone();
        for potential in 0..unique_count {
            let indices = overlap_density_indices(refeff_core::OverlapDensityIndicesInput {
                overlapped_density: density.column(potential),
                muffin_tin_radius: initial.pot.muffin_tin_radii[potential],
                norman_radius: norman_radii[potential],
            })?;
            muffin_tin_indices[potential] = indices.muffin_tin_index;
            norman_indices[potential] = indices.norman_index;
            norman_radii[potential] = indices.norman_radius;
        }
        let volume_seed = if input.scattering.totvol <= 0.0 {
            let mut norman_volume = 0.0;
            let mut interstitial_volume = 0.0;
            for potential in 0..unique_count {
                norman_volume += initial.pot.potential_multiplicities[potential]
                    * norman_radii[potential].powi(3);
                interstitial_volume -= initial.pot.potential_multiplicities[potential]
                    * initial.pot.muffin_tin_radii[potential].powi(3);
            }
            4.0 * std::f64::consts::PI / 3.0 * (interstitial_volume + norman_volume)
        } else {
            let mut interstitial_volume = 0.0;
            for potential in 0..unique_count {
                interstitial_volume -= initial.pot.potential_multiplicities[potential]
                    * initial.pot.muffin_tin_radii[potential].powi(3);
            }
            4.0 * std::f64::consts::PI / 3.0 * interstitial_volume + input.scattering.totvol
        };
        let overlap_matrix = muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
            highest_potential_index: unique_count - 1,
            atom_potentials: atom_potentials.view(),
            atom_positions: atom_positions.view(),
            representative_atoms: representative_atoms.view(),
            potential_multiplicities: initial.pot.potential_multiplicities.view(),
            explicit_overlaps: &explicit_overlap_refs,
            muffin_tin_indices: muffin_tin_indices.view(),
            muffin_tin_radii: initial.pot.muffin_tin_radii.view(),
            norman_radii: norman_radii.view(),
            near_neighbor_flags: near_neighbor_flags.view(),
            interstitial_selector,
            interstitial_volume: volume_seed,
        })?;
        let projected_density = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
            highest_potential_index: unique_count - 1,
            values: density,
            radii: overlap_matrix.radii.view(),
            potential_multiplicities: initial.pot.potential_multiplicities.view(),
            norman_indices: norman_indices.view(),
            muffin_tin_indices: muffin_tin_indices.view(),
            muffin_tin_radii: initial.pot.muffin_tin_radii.view(),
            norman_radii: norman_radii.view(),
            near_neighbor_flags: near_neighbor_flags.view(),
            overlap_matrix: &overlap_matrix,
            interstitial_selector,
            interstitial_value: 0.0,
            mode: MuffinTinOverlapProjectionMode::Density {
                total_charge: initial.pot.scalars.total_charge,
            },
        })?;
        let projected_valence = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
            highest_potential_index: unique_count - 1,
            values: valence_density,
            radii: overlap_matrix.radii.view(),
            potential_multiplicities: initial.pot.potential_multiplicities.view(),
            norman_indices: norman_indices.view(),
            muffin_tin_indices: muffin_tin_indices.view(),
            muffin_tin_radii: initial.pot.muffin_tin_radii.view(),
            norman_radii: norman_radii.view(),
            near_neighbor_flags: near_neighbor_flags.view(),
            overlap_matrix: &overlap_matrix,
            interstitial_selector,
            interstitial_value: 0.0,
            mode: MuffinTinOverlapProjectionMode::Density { total_charge: 0.0 },
        })?;
        let outside_charge = projected_density.interstitial_value;
        let inside_charge = initial.pot.scalars.total_charge - outside_charge;
        let valence_inside = -projected_valence.interstitial_value;
        let interstitial_density =
            4.0 * std::f64::consts::PI * outside_charge / overlap_matrix.interstitial_volume;
        assert!(
            outside_charge.is_finite()
                && inside_charge.is_finite()
                && valence_inside.is_finite()
                && interstitial_density.is_finite()
                && outside_charge > 0.0
                && inside_charge > 0.0
                && valence_inside > 0.0
                && interstitial_density > 0.0,
            "{label} POT density projection must keep positive interstitial charge, got outside_charge={outside_charge}, inside_charge={inside_charge}, valence_inside={valence_inside}, rhoint={interstitial_density}"
        );
    }

    Ok(())
}

#[test]
fn atomic_module_assembles_terminal_scf_final_pot_candidate() -> Result<()> {
    let mut pot = beryllium_single_potential_pot_bin();
    pot.scalars.fermi_level = -0.10;
    pot.electron_density.fill(0.25);
    pot.valence_density.fill(0.05);
    pot.coulomb_potential.fill(-0.20);

    let potential_count = pot.potential_count();
    let mut final_density = pot.electron_density.clone();
    final_density.fill(0.33);
    let mut final_valence = pot.valence_density.clone();
    final_valence.fill(0.11);
    let mut final_coulomb = pot.coulomb_potential.clone();
    final_coulomb.fill(-0.42);
    let occupancy_by_l = Array2::from_shape_fn(pot.valence_occupancy.dim(), |(angular, _)| {
        angular as f64 + 0.25
    });
    let state = PotScfState {
        fermi_energy: -0.32,
        norman_charges: Array1::from_vec(vec![3.75]),
        norman_charge_reference: Array1::from_vec(vec![3.75]),
        occupancy_by_l: occupancy_by_l.clone(),
        overlapped_density: final_density.clone(),
        overlapped_valence_density: final_valence.clone(),
        coulomb_potential: final_coulomb.clone(),
        workspace: BroydenWorkspace::zeros(2, potential_count),
    };
    let config = beryllium_single_potential_config_dat();
    let reported_charge_transfer = Array1::from_vec(vec![-2.25]);

    let final_pot = super::scf_pot_final_pot_from_state(
        &pot,
        &state,
        reported_charge_transfer.view(),
        PotScfOuterIterationStatus::Converged,
        &config,
    )?;

    assert_eq!(final_pot.scalars.fermi_level, state.fermi_energy);
    assert_eq!(final_pot.norman_charges, reported_charge_transfer);
    assert_eq!(
        state.norman_charges,
        Array1::from_vec(vec![3.75]),
        "final serialization must not convert the iterative raw qnrm state"
    );
    assert_eq!(final_pot.valence_occupancy, occupancy_by_l);
    assert_eq!(final_pot.electron_density, final_density);
    assert_eq!(final_pot.valence_density, final_valence);
    assert_eq!(final_pot.coulomb_potential, final_coulomb);
    assert_eq!(final_pot.total_potential, pot.total_potential);
    assert_eq!(final_pot.orbital_occupancy[(0, 0)], 2.0);
    assert!(final_pot.raw_text.is_none());
    assert!(pot_bin_string(&final_pot)?.contains("ATOM source APOT Be smoke test"));

    let iteration_limit_pot = super::scf_pot_final_pot_from_state(
        &pot,
        &state,
        reported_charge_transfer.view(),
        PotScfOuterIterationStatus::ReachedIterationLimit,
        &config,
    )?;
    assert_eq!(iteration_limit_pot, final_pot);

    for status in [
        PotScfOuterIterationStatus::NeedsMoreSourcePoints,
        PotScfOuterIterationStatus::RepeatRequired,
        PotScfOuterIterationStatus::NeedsNextIteration,
    ] {
        let error = super::scf_pot_final_pot_from_state(
            &pot,
            &state,
            reported_charge_transfer.view(),
            status,
            &config,
        )
        .err()
        .with_context(|| {
            format!("nonterminal SCF status {status:?} should not produce final pot.bin")
        })?;
        assert!(
            error
                .to_string()
                .contains("requires converged or iteration-limit SCF status"),
            "{error:?}"
        );
    }
    Ok(())
}

#[test]
fn atomic_module_lifts_fovrg_fms_grids_into_pot_contour_source_rows() -> Result<()> {
    let point_count = 2;
    let potential_count = 2;
    let angular_count = 2;
    let radial_count = 7;
    let source_radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let energies = Array1::from_vec(vec![Complex64::new(0.20, 0.04), Complex64::new(0.28, 0.04)]);
    let mut pot = sample_pot_bin();
    pot.norman_radii = Array1::from_vec(vec![0.64, 0.72]);

    let wave_numbers =
        Array2::from_shape_fn((point_count, potential_count), |(point, potential)| {
            Complex64::new(
                0.70 + 0.05 * point as f64 + 0.03 * potential as f64,
                0.06 + 0.01 * potential as f64,
            )
        });
    let regular_large = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, potential, angular, radial)| {
            Complex64::new(
                0.10 + 0.02 * radial as f64 + 0.03 * angular as f64 + 0.01 * point as f64,
                0.02 - 0.004 * potential as f64 + 0.003 * radial as f64,
            )
        },
    );
    let regular_small = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, potential, angular, radial)| {
            Complex64::new(
                0.015 + 0.003 * radial as f64 + 0.004 * angular as f64,
                -0.004 + 0.002 * point as f64 + 0.001 * potential as f64,
            )
        },
    );
    let irregular_large = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, potential, angular, radial)| {
            Complex64::new(
                -0.35 + 0.04 * radial as f64 + 0.02 * potential as f64,
                0.24 - 0.015 * angular as f64 + 0.01 * point as f64,
            )
        },
    );
    let irregular_small = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, potential, angular, radial)| {
            Complex64::new(
                -0.030 + 0.004 * radial as f64 + 0.002 * angular as f64,
                0.014 - 0.002 * potential as f64 + 0.001 * point as f64,
            )
        },
    );
    let fovrg_grid = PotScfFovrgSourceGridHandoff {
        source_radii: source_radii.clone(),
        energies_hartree: energies.clone(),
        reference_energies_hartree: Array2::zeros((point_count, potential_count)),
        wave_numbers,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        phase_shifts: Array3::zeros((point_count, angular_count, potential_count)),
        phase_amplitudes: Array3::zeros((point_count, angular_count, potential_count)),
        radial_active_counts: Array1::from_vec(vec![radial_count, radial_count]),
        rholie_active_counts: Array1::from_vec(vec![radial_count, radial_count]),
        muffin_tin_indices_1based: Array1::from_vec(vec![4, 4]),
        norman_indices_1based: Array1::from_vec(vec![6, 6]),
        radial_handoffs: Vec::new(),
    };
    let fms_grid = PotScfFmsSourceGridHandoff {
        energies_hartree: energies,
        scattering_trace: Array3::from_shape_fn(
            (point_count, angular_count, potential_count),
            |(point, angular, potential)| {
                Complex64::new(
                    0.04 + 0.01 * point as f64 + 0.02 * angular as f64,
                    -0.03 + 0.015 * potential as f64,
                )
            },
        ),
    };

    let rows = super::scf_pot_contour_source_rows_from_initial_state(&pot, &fovrg_grid, &fms_grid)?;

    assert_eq!(rows.source_energies, fovrg_grid.energies_hartree);
    assert_eq!(
        rows.scattering_trace.dim(),
        (point_count, angular_count, potential_count)
    );
    assert_eq!(
        rows.scattering_density.dim(),
        (
            point_count,
            POT_BIN_RADIAL_POINTS,
            angular_count,
            potential_count
        )
    );
    assert_eq!(
        rows.embedded_density_source.dim(),
        (point_count, POT_BIN_RADIAL_POINTS, potential_count)
    );
    let density_norm = rows
        .embedded_density_source
        .iter()
        .map(|value| value.norm())
        .sum::<f64>();
    assert!(density_norm > 0.0);
    Ok(())
}

#[test]
fn atomic_module_masks_fovrg_and_fms_channels_by_local_lmaxsc() -> Result<()> {
    let mut input = copper_two_potential_pot_input()?;
    input.potentials[0].lmaxsc = 1;
    input.potentials[1].lmaxsc = 0;
    let energy_count = 2;
    let potential_count = 2;
    let angular_count = 2;
    let radial_count = 3;
    let one = Complex64::new(1.0, -0.5);
    let mut fovrg = PotScfFovrgSourceGridHandoff {
        source_radii: Array1::from_vec(vec![0.1, 0.2, 0.3]),
        energies_hartree: Array1::from_elem(energy_count, one),
        reference_energies_hartree: Array2::from_elem((energy_count, potential_count), one),
        wave_numbers: Array2::from_elem((energy_count, potential_count), one),
        regular_large: Array4::from_elem(
            (energy_count, potential_count, angular_count, radial_count),
            one,
        ),
        regular_small: Array4::from_elem(
            (energy_count, potential_count, angular_count, radial_count),
            one,
        ),
        irregular_large: Array4::from_elem(
            (energy_count, potential_count, angular_count, radial_count),
            one,
        ),
        irregular_small: Array4::from_elem(
            (energy_count, potential_count, angular_count, radial_count),
            one,
        ),
        phase_shifts: Array3::from_elem((energy_count, angular_count, potential_count), one),
        phase_amplitudes: Array3::from_elem((energy_count, angular_count, potential_count), one),
        radial_active_counts: Array1::from_elem(potential_count, radial_count),
        rholie_active_counts: Array1::from_elem(potential_count, radial_count),
        muffin_tin_indices_1based: Array1::ones(potential_count),
        norman_indices_1based: Array1::ones(potential_count),
        radial_handoffs: Vec::new(),
    };

    super::scf_pot_mask_inactive_fovrg_angular_channels(&input, &mut fovrg)?;

    for energy in 0..energy_count {
        assert_eq!(fovrg.phase_shifts[(energy, 1, 0)], one);
        assert_eq!(fovrg.phase_shifts[(energy, 1, 1)], Complex64::new(0.0, 0.0));
        assert_eq!(fovrg.regular_large[(energy, 0, 1, 0)], one);
        assert_eq!(
            fovrg.regular_large[(energy, 1, 1, 0)],
            Complex64::new(0.0, 0.0)
        );
        assert_eq!(
            fovrg.irregular_small[(energy, 1, 1, radial_count - 1)],
            Complex64::new(0.0, 0.0)
        );
    }

    let mut fms = PotScfFmsSourceGridHandoff {
        energies_hartree: Array1::from_elem(energy_count, one),
        scattering_trace: Array3::from_elem((energy_count, angular_count, potential_count), one),
    };
    super::scf_pot_mask_inactive_fms_angular_channels(&input, &mut fms)?;
    assert_eq!(fms.scattering_trace[(0, 1, 0)], one);
    assert_eq!(fms.scattering_trace[(0, 1, 1)], Complex64::new(0.0, 0.0));

    let mut uniform = input.clone();
    uniform.potentials[1].lmaxsc = 1;
    let unchanged = PotScfFmsSourceGridHandoff {
        energies_hartree: Array1::from_elem(energy_count, one),
        scattering_trace: Array3::from_elem((energy_count, angular_count, potential_count), one),
    };
    let mut actual = unchanged.clone();
    super::scf_pot_mask_inactive_fms_angular_channels(&uniform, &mut actual)?;
    assert_eq!(actual, unchanged);

    let mut high_l = uniform.clone();
    high_l.potentials[0].lmaxsc = 4;
    high_l.potentials[1].lmaxsc = 4;
    let fovrg_counts = super::scf_pot_local_angular_counts(&high_l, potential_count, 5)?;
    assert_eq!(
        fovrg_counts.to_vec(),
        vec![5, 5],
        "atomic/FOVRG background must retain l=4"
    );
    assert_eq!(
        super::scf_pot_fms_solve_angular_count(&high_l, 5)?,
        5,
        "zero-temperature FMS must retain the configured lmax"
    );
    high_l.thermal.scf_temperature = 0.025_852_026_9;
    high_l.thermal.iscfth = 2;
    assert_eq!(
        super::scf_pot_fms_solve_angular_count(&high_l, 5)?,
        super::POT_THERMAL_FMS_MAX_LMAX + 1,
        "only thermal FMS is capped at lmax=3"
    );

    input.potentials[1].lmaxsc = -1;
    let error = super::scf_pot_mask_inactive_fms_angular_channels(&input, &mut actual)
        .err()
        .context("negative local lmaxsc should fail closed")?;
    assert!(
        error.to_string().contains("must be non-negative"),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn atomic_module_advances_initial_pot_scf_state_from_contour_rows() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 4;
    let static_arrays = beryllium_single_potential_static_arrays()?;
    let mut pot = beryllium_single_potential_pot_bin();
    pot.valence_occupancy[(0, 0)] = 1.0;
    let potential_count = pot.potential_count();
    let angular_count = pot.valence_occupancy.nrows();
    let radial_count = POT_BIN_RADIAL_POINTS;
    let energy_grid = ScmtEnergyGrid {
        energies: Array1::from_vec(vec![Complex64::new(0.10, 0.02)]),
        steps: Array1::from_vec(vec![0.01]),
        active_len: 1,
        lower_imaginary_count: 1,
        real_axis_count: 0,
        upper_imaginary_count: 0,
    };
    let rows = PotScfContourSourceRows {
        source_energies: Array1::from_vec(vec![energy_grid.energies[0]]),
        scattering_trace: Array3::<Complex32>::zeros((
            energy_grid.active_len,
            angular_count,
            potential_count,
        )),
        scattering_ldos: Array3::<Complex64>::zeros((
            energy_grid.active_len,
            angular_count,
            potential_count,
        )),
        embedded_ldos_source: Array3::<Complex64>::zeros((
            energy_grid.active_len,
            angular_count,
            potential_count,
        )),
        scattering_density: Array4::<Complex64>::zeros((
            energy_grid.active_len,
            radial_count,
            angular_count,
            potential_count,
        )),
        embedded_density_source: Array3::<Complex64>::zeros((
            energy_grid.active_len,
            radial_count,
            potential_count,
        )),
        density_scale: Array3::<Complex64>::zeros((
            energy_grid.active_len,
            angular_count,
            potential_count,
        )),
    };
    let last_indices = Array1::from_vec(vec![radial_count]);
    let workspace = BroydenWorkspace::zeros(input.run.nscmt as usize, potential_count);
    let initial_norman_charges = Array1::zeros(potential_count);
    let state = PotScfState {
        fermi_energy: pot.scalars.fermi_level,
        norman_charges: initial_norman_charges.clone(),
        norman_charge_reference: initial_norman_charges,
        occupancy_by_l: Array2::zeros(pot.valence_occupancy.dim()),
        overlapped_density: pot.electron_density.clone(),
        overlapped_valence_density: pot.valence_density.clone(),
        coulomb_potential: pot.coulomb_potential.clone(),
        workspace,
    };

    let advance = super::scf_pot_state_advance_from_rows(
        &input,
        &static_arrays,
        &pot,
        &energy_grid,
        &rows,
        last_indices.view(),
        &state,
        1,
        true,
    )?;

    assert_eq!(
        advance.iteration.status,
        PotScfIterationStatus::NeedsMoreSourcePoints
    );
    assert_eq!(
        advance.outer.status,
        PotScfOuterIterationStatus::NeedsMoreSourcePoints
    );
    assert_eq!(advance.iteration.contour.energy_points_used, 1);
    assert_eq!(advance.state.fermi_energy, state.fermi_energy);
    assert_eq!(advance.state.overlapped_density, state.overlapped_density);
    assert_eq!(advance.state.coulomb_potential, state.coulomb_potential);

    let second = super::scf_pot_state_advance_from_rows(
        &input,
        &static_arrays,
        &pot,
        &energy_grid,
        &rows,
        last_indices.view(),
        &state,
        2,
        false,
    )?;
    assert_eq!(
        second.iteration.status,
        PotScfIterationStatus::NeedsMoreSourcePoints
    );
    assert_eq!(
        second.outer.status,
        PotScfOuterIterationStatus::NeedsMoreSourcePoints
    );
    assert_eq!(second.iteration.contour.energy_points_used, 1);
    assert_eq!(second.state.fermi_energy, state.fermi_energy);
    Ok(())
}

#[test]
fn atomic_module_prepares_next_pot_scf_iteration_after_state_advance() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = beryllium_pot_input()?;
    input.run.nscmt = 4;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    let geom = beryllium_single_potential_geom_dat();
    std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;

    let caches = super::AtomicCachePaths::new(temp.path());
    let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;
    let static_arrays = super::atomic_apot_static_arrays_from_source_geometry(
        &input,
        &geom,
        Array1::from_elem(initial.pot.potential_count(), 1.0),
    )?;
    let mut state = initial.state.clone();
    state.fermi_energy += 1.0e-4;
    state.overlapped_valence_density = state.overlapped_valence_density.mapv(|value| {
        if value.is_finite() {
            value * 0.99
        } else {
            value
        }
    });
    let config = super::generated_config_dat(&input, &temp.path().join("config.inp"))?;

    let mut fms_cache = crate::fms::PotScfFmsPipelineCache::default();
    let next = super::scf_pot_next_iteration_preparation_from_state(
        temp.path(),
        &input,
        &static_arrays,
        &config,
        &initial.pot,
        &state,
        2,
        &mut fms_cache,
    )?;

    assert_eq!(next.iteration, 2);
    assert_eq!(next.pot.potential_count(), initial.pot.potential_count());
    assert_eq!(next.state.fermi_energy, state.fermi_energy);
    assert_eq!(next.pot.scalars.fermi_level, state.fermi_energy);
    assert_eq!(next.pot.electron_density, state.overlapped_density);
    assert_eq!(next.pot.valence_density, state.overlapped_valence_density);
    assert_eq!(next.pot.coulomb_potential, state.coulomb_potential);
    assert_eq!(next.pot.total_potential, next.istprm.total_potential);
    assert_eq!(
        next.last_indices,
        super::scf_pot_rholie_last_indices(&next.pot)?
    );
    assert!(next.energy_grid.active_len > 0);
    assert_eq!(
        next.energy_grid.energies[0].re,
        next.pot.scalars.core_valence_energy
    );
    assert_eq!(next.energy_grid.steps.len(), super::POT_SCMT_FLOOR_COUNT);
    if let Some(fovrg_grid) = &next.fovrg_grid {
        assert!(next.fovrg_grid_unavailable.is_none());
        assert_eq!(fovrg_grid.rholie_active_counts, next.last_indices);
        assert!(
            fovrg_grid
                .radial_active_counts
                .iter()
                .zip(fovrg_grid.rholie_active_counts.iter())
                .all(|(radial_count, rholie_count)| radial_count >= rholie_count)
        );
        assert!(fovrg_grid.energies_hartree.len() >= next.energy_grid.active_len);
        assert_eq!(fovrg_grid.wave_numbers.dim().1, next.pot.potential_count());
    } else {
        let reason = next
            .fovrg_grid_unavailable
            .as_deref()
            .expect("missing next FOVRG source-grid reason");
        assert!(!reason.is_empty());
    }
    if let Some(contour_rows) = &next.contour_rows {
        assert!(next.contour_rows_unavailable.is_none());
        assert!(contour_rows.source_energies.len() >= next.energy_grid.active_len);
        assert_eq!(
            contour_rows.scattering_trace.dim().2,
            next.pot.potential_count()
        );
    } else {
        let reason = next
            .contour_rows_unavailable
            .as_deref()
            .expect("missing next contour source-row reason");
        assert!(!reason.is_empty());
    }
    if let Some(advance) = &next.state_advance {
        assert!(next.state_advance_unavailable.is_none());
        assert!(next.contour_rows.as_ref().is_some_and(|rows| {
            advance.iteration.contour.energy_points_used <= rows.source_energies.len()
        }));
        assert_eq!(
            advance.iteration.contour.embedded_ldos.ncols(),
            next.pot.potential_count()
        );
        assert_eq!(
            advance.state.overlapped_density.dim(),
            next.state.overlapped_density.dim()
        );
    } else {
        let reason = next
            .state_advance_unavailable
            .as_deref()
            .expect("missing next state-advance reason");
        assert!(!reason.is_empty());
    }
    Ok(())
}

#[test]
fn atomic_module_replaces_malformed_apot_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_beryllium_atomic_source_handoffs(temp.path())?;
    std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

    assert!(has_supported_atomic_source_handoff(temp.path())?);
    assert!(!has_cached_atomic_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let apot = read_apot_bin(temp.path().join("apot.bin"))?;
    assert!(super::apot_has_section(&apot, 1));
    assert!(super::apot_has_section(
        &apot,
        APOT_ATOMIC_DENSITY_SECTION_NUMBER
    ));
    Ok(())
}

#[test]
fn atomic_module_generates_source_scf_apot_sections_from_pot_input() -> Result<()> {
    let input = beryllium_pot_input()?;

    let apot = super::generated_atomic_scf_apot_bin(&input, Path::new("config.inp"))?;

    assert_eq!(
        apot.sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        vec![3, 8, 10, 11, 13, 14, 20, 22, 23, 24, 25, 26, 27, 28, 29]
    );
    let apot = rendered_apot_bin(&apot)?;

    let ApotBinPayload::Records(records) =
        &super::apot_section(&apot, APOT_ATOMIC_NORB_SECTION_NUMBER, "norb")?.payload
    else {
        anyhow::bail!("ATOM generated norb should be row records");
    };
    assert_eq!(records.rows.len(), 2);
    assert!(matches!(records.rows[0][0], ApotBinValue::Int(value) if value > 0));
    assert!(matches!(records.rows[1][0], ApotBinValue::Int(value) if value > 0));

    let density = super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?;
    assert_eq!(density.dim(), (POT_BIN_RADIAL_POINTS, 2));
    assert!(density[[0, 0]].is_finite());
    assert!(density[[0, 0]] > 0.0);
    assert!(density[[0, 1]].is_finite());

    let valence = super::real_matrix_section(
        &apot,
        APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
        "xnval",
    )?;
    assert_eq!(valence.dim(), (POT_BIN_ORBITALS, 2));

    let energies =
        super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER, "eorb")?;
    assert_eq!(energies.dim(), (POT_BIN_ORBITALS, 2));
    assert!(energies[[0, 0]].is_finite());

    let kappa = super::int_matrix_section(&apot, APOT_ATOMIC_KAPPA_SECTION_NUMBER, "kappa")?;
    assert_eq!(kappa.dim(), (POT_BIN_ORBITALS, 2));
    assert_eq!(kappa[[0, 0]], -1);

    let first_dgc = super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_SECTION_START, "dgc")?;
    assert_eq!(first_dgc.dim(), (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS));
    assert!(first_dgc[[0, 0]].is_finite());

    let final_adpc =
        super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_SECTION_START + 7, "adpc")?;
    assert_eq!(final_adpc.dim(), (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS));
    assert!(final_adpc[[0, 0]].is_finite());
    Ok(())
}

#[test]
fn atomic_module_generates_source_scf_apot_sections_for_core_hole_state() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;

    let apot = super::generated_atomic_scf_apot_bin(&input, Path::new("config.inp"))?;
    let apot = rendered_apot_bin(&apot)?;

    let ApotBinPayload::Records(records) =
        &super::apot_section(&apot, APOT_ATOMIC_NORB_SECTION_NUMBER, "norb")?.payload
    else {
        anyhow::bail!("ATOM generated norb should be row records");
    };
    assert_eq!(records.rows.len(), 2);
    assert!(matches!(records.rows[0][0], ApotBinValue::Int(value) if value > 0));
    assert!(matches!(records.rows[1][0], ApotBinValue::Int(value) if value > 0));

    let density = super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?;
    assert!(density[[0, 0]].is_finite());
    assert!(density[[0, 1]].is_finite());
    assert_ne!(density[[0, 0]], density[[0, 1]]);

    let kappa = super::int_matrix_section(&apot, APOT_ATOMIC_KAPPA_SECTION_NUMBER, "kappa")?;
    assert_eq!(kappa[[0, 1]], -1);

    let final_dgc =
        super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_SECTION_START + 1, "dgc")?;
    assert!(final_dgc[[0, 0]].is_finite());
    Ok(())
}

#[test]
fn atomic_module_uses_warnion_custom_config_charge_for_scf_states() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let feff = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE WARNION Cu custom configuration smoke test
EDGE K
CONTROL 1 0 0 0 0 0
CONFIG card 1
0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0
WARNION
COREHOLE FSR
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let document = FeffDocument::from_input(&feff)?;
    let input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
    std::fs::write(
        temp.path().join("config.inp"),
        rdinp::config_inp_string(&document)?,
    )?;

    assert!(input.warn_ion);
    assert_eq!(input.config_type, 2);

    let states = super::generated_atomic_scf_states(&input, &temp.path().join("config.inp"))?;

    assert_eq!(states.len(), 3);
    assert!((states[0].occupations.sum() - 28.0).abs() < 1.0e-10);
    assert!((states[1].occupations.sum() - 29.0).abs() < 1.0e-10);
    assert!((states[2].occupations.sum() - 28.0).abs() < 1.0e-10);
    Ok(())
}

#[test]
fn atomic_module_assembles_full_apot_sections_from_source_arrays() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;
    let static_arrays = beryllium_single_potential_static_arrays()?;

    let apot = ApotBinData {
        sections: super::generated_atomic_apot_sections_from_static_arrays(
            &input,
            Path::new("config.inp"),
            &static_arrays,
        )?,
    };
    assert_eq!(
        apot.sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        (1..=29).collect::<Vec<_>>()
    );
    let apot = rendered_apot_bin(&apot)?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let overlap_arrays =
        super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    let ApotBinPayload::Records(records) = &super::apot_section(&apot, 1, "scalars")?.payload
    else {
        anyhow::bail!("ATOM full apot section 1 should be scalar records");
    };
    assert_eq!(records.rows.len(), 1);
    assert!(matches!(records.rows[0][0], ApotBinValue::Int(0)));
    assert!(matches!(records.rows[0][1], ApotBinValue::Int(1)));
    assert!(matches!(records.rows[0][2], ApotBinValue::Int(1)));
    let ApotBinValue::Real(relaxation_energy) = &records.rows[0][3] else {
        anyhow::bail!("ATOM full apot erelax should be real-valued");
    };
    let ApotBinValue::Real(edge_energy) = &records.rows[0][4] else {
        anyhow::bail!("ATOM full apot emu should be real-valued");
    };
    let ApotBinValue::Real(amplitude_reduction) = &records.rows[0][5] else {
        anyhow::bail!("ATOM full apot s02 should be real-valued");
    };
    assert!(relaxation_energy.is_finite());
    assert!(edge_energy.is_finite());
    assert!(amplitude_reduction.is_finite());
    assert!(
        *amplitude_reduction > 0.0 && *amplitude_reduction <= 1.0,
        "unexpected full apot s02 {amplitude_reduction}"
    );

    let ApotBinPayload::Records(unique_records) =
        &super::apot_section(&apot, 2, "unique potentials")?.payload
    else {
        anyhow::bail!("ATOM full apot section 2 should be unique-potential records");
    };
    assert_eq!(unique_records.rows.len(), 1);
    assert!(matches!(unique_records.rows[0][0], ApotBinValue::Int(4)));
    let ApotBinValue::Real(norman_radius) = &unique_records.rows[0][3] else {
        anyhow::bail!("ATOM full apot rnrm should be real-valued");
    };
    assert_close(
        *norman_radius,
        overlap_arrays.norman_radii[0],
        1.0e-9 * overlap_arrays.norman_radii[0].abs().max(1.0),
        "full apot rnrm",
    );

    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER, "rhoval")?
            .dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert_eq!(
        super::real_matrix_section(&apot, APOT_ATOMIC_COULOMB_SECTION_NUMBER, "vcoul")?.dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert_eq!(
        super::real_matrix_section(&apot, 12, "xnvmu")?.dim(),
        (4, 2)
    );
    assert_eq!(
        super::real_matrix_section(&apot, 17, "edens")?.dim(),
        (POT_BIN_RADIAL_POINTS, 1)
    );
    assert_eq!(
        super::real_matrix_section(&apot, 19, "vclap")?.dim(),
        (POT_BIN_RADIAL_POINTS, 1)
    );
    assert_eq!(
        super::int_matrix_section(&apot, 21, "iorb")?.dim(),
        (POT_BIN_IORB_SLOTS, 2)
    );
    Ok(())
}

#[test]
fn atomic_module_derives_iorb_section_matrix_from_scf_configurations() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;

    let indices = super::atomic_orbital_indices_by_kappa(&input, Path::new("config.inp"))?;

    assert_eq!(indices.dim(), (10, 2));
    assert_eq!(indices[[4, 0]], 2);
    assert_eq!(indices[[4, 1]], 2);
    assert_eq!(indices[[6, 1]], 3);
    for row in 0..10 {
        if row != 4 {
            assert_eq!(indices[[row, 0]], 0);
        }
        if row != 4 && row != 6 {
            assert_eq!(indices[[row, 1]], 0);
        }
    }
    Ok(())
}

#[test]
fn atomic_module_derives_norman_valence_counts_from_scf_configurations() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;
    let configurations =
        super::atomic_scf_configurations_from_pot_input(&input, Path::new("config.inp"))?;

    let counts = super::atomic_norman_valence_counts_by_l(&input, Path::new("config.inp"))?;

    assert_eq!(counts.dim(), (4, 2));
    for (state_index, configuration) in configurations.iter().enumerate() {
        let expected = configuration
            .valence_counts
            .iter()
            .take(configuration.orbital_count)
            .sum::<f64>();
        let actual = counts.column(state_index).sum();
        assert_close(
            actual,
            expected,
            1.0e-12,
            &format!("xnvmu total state {state_index}"),
        );
    }
    assert_eq!(counts[(2, 0)], 0.0);
    assert_eq!(counts[(3, 0)], 0.0);
    assert!(
        counts[(0, 0)] > 0.0,
        "initial Be absorber should carry s-channel valence count"
    );
    assert!(
        counts[(1, 1)] > 0.0,
        "final Be K-edge state should carry p-channel screening valence count"
    );
    Ok(())
}

#[test]
fn atomic_module_converts_generated_pot_core_valence_energy_from_input_ev() -> Result<()> {
    let input = beryllium_pot_input()?;
    let geom = beryllium_single_potential_geom_dat();

    let pot = super::generated_no_scf_pot_bin(&input, Path::new("config.inp"), &geom)?;

    let guard = super::POT_CORVAL_TOLERANCE_EV / refeff_core::FEFF_HARTREE_EV;
    let input_ecv_hartree = input.scattering.ecv / refeff_core::FEFF_HARTREE_EV;
    assert!(pot.scalars.core_valence_energy.is_finite());
    assert_close(
        pot.scalars.core_valence_energy,
        input_ecv_hartree,
        1.0e-12,
        "converted core-valence separation",
    );
    assert!(
        pot.scalars.core_valence_energy <= pot.scalars.interstitial_potential - guard + 1.0e-12,
        "core-valence separation {} should respect vint {} minus FEFF guard {}",
        pot.scalars.core_valence_energy,
        pot.scalars.interstitial_potential,
        guard
    );
    Ok(())
}

#[test]
fn atomic_module_builds_corval_scan_grid_with_feff_spacing() -> Result<()> {
    let grid = super::pot_corval_scan_energy_grid(-1.2, -70.0)?;

    assert!(grid.len() > 2);
    assert_close(
        grid[0].re,
        -70.0 / refeff_core::FEFF_HARTREE_EV,
        1.0e-12,
        "corval lower energy",
    );
    assert_close(
        grid[grid.len() - 1].re,
        super::POT_CORVAL_HIGH_EV / refeff_core::FEFF_HARTREE_EV,
        1.0e-12,
        "corval upper energy",
    );
    assert_close(
        grid[0].im,
        super::POT_CORVAL_LDOS_IMAGINARY_EV / refeff_core::FEFF_HARTREE_EV,
        1.0e-12,
        "corval imaginary broadening",
    );
    assert_close(
        (grid[1].re - grid[0].re) * refeff_core::FEFF_HARTREE_EV,
        0.5,
        5.0e-3,
        "corval approximate energy step",
    );
    Ok(())
}

#[test]
fn atomic_module_detects_corval_ldos_peak_from_previous_sample() -> Result<()> {
    let energies = Array1::from_vec(vec![
        Complex64::new(-2.0, 0.05),
        Complex64::new(-1.5, 0.05),
        Complex64::new(-1.0, 0.05),
        Complex64::new(-0.5, 0.05),
    ]);
    let angular_count = 2;
    let potentials = 1;
    let mut embedded_ldos = Array3::<Complex64>::zeros((energies.len(), angular_count, potentials));
    let threshold = 1.0
        / (6.0
            * (super::POT_CORVAL_LDOS_IMAGINARY_EV / refeff_core::FEFF_HARTREE_EV)
            * std::f64::consts::PI);
    embedded_ldos[(0, 0, 0)] = Complex64::new(0.0, 0.1);
    embedded_ldos[(1, 0, 0)] = Complex64::new(0.0, threshold + 0.25);
    embedded_ldos[(2, 0, 0)] = Complex64::new(0.0, threshold + 0.10);
    embedded_ldos[(3, 0, 0)] = Complex64::new(0.0, threshold + 0.05);
    embedded_ldos[(0, 1, 0)] = Complex64::new(0.0, 0.1);
    embedded_ldos[(1, 1, 0)] = Complex64::new(0.0, 0.2);
    embedded_ldos[(2, 1, 0)] = Complex64::new(0.0, 0.15);
    embedded_ldos[(3, 1, 0)] = Complex64::new(0.0, 0.05);

    let peaks = super::pot_corval_ldos_peak_energies(
        energies.view(),
        embedded_ldos.view(),
        angular_count,
        true,
    )?;

    assert_close(peaks[(0, 0)], -1.5, 1.0e-12, "l=0 corval peak");
    assert!(
        peaks[(1, 0)].is_nan(),
        "below-threshold l=1 channel should not report a peak"
    );
    Ok(())
}

#[test]
fn atomic_module_does_not_treat_corval_batch_end_as_scan_end() -> Result<()> {
    let energies = Array1::from_vec(vec![
        Complex64::new(-2.0, 0.05),
        Complex64::new(-1.5, 0.05),
        Complex64::new(-1.0, 0.05),
        Complex64::new(-0.5, 0.05),
    ]);
    let threshold = 1.0
        / (6.0
            * (super::POT_CORVAL_LDOS_IMAGINARY_EV / refeff_core::FEFF_HARTREE_EV)
            * std::f64::consts::PI);
    let embedded_ldos = Array3::from_shape_vec(
        (energies.len(), 1, 1),
        vec![
            Complex64::new(0.0, 0.1),
            Complex64::new(0.0, threshold + 0.1),
            Complex64::new(0.0, threshold + 0.2),
            Complex64::new(0.0, threshold + 0.05),
        ],
    )?;
    let partial_energies = Array1::from_vec(energies.iter().take(3).copied().collect::<Vec<_>>());
    let partial_ldos = Array3::from_shape_vec(
        (3, 1, 1),
        embedded_ldos.iter().take(3).copied().collect::<Vec<_>>(),
    )?;

    let partial = super::pot_corval_ldos_peak_energies(
        partial_energies.view(),
        partial_ldos.view(),
        1,
        false,
    )?;
    assert!(
        partial[(0, 0)].is_nan(),
        "a rising LDOS at a partial batch endpoint is not a completed peak"
    );

    let terminal =
        super::pot_corval_ldos_peak_energies(energies.view(), embedded_ldos.view(), 1, false)?;
    assert_close(
        terminal[(0, 0)],
        -1.0,
        1.0e-12,
        "corval peak after the batch boundary",
    );
    Ok(())
}

#[test]
fn atomic_module_uses_scf_exchange_selector_for_no_scf_ground_state_potential() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.control.ixc = 0;
    let density = 3.0;
    let coulomb = -0.2;
    let density_radius = (density / 3.0_f64).powf(-1.0 / 3.0);
    let overlap = super::AtomicApotOverlapArrays {
        norman_radii: Array1::from_vec(vec![1.5]),
        magnetization_density: Array2::from_elem((1, 1), -1.25),
        overlapped_density: Array2::from_shape_vec((1, 1), vec![density])?,
        overlapped_valence_density: Array2::from_shape_vec((1, 1), vec![0.5 * density])?,
        overlapped_coulomb_potential: Array2::from_shape_vec((1, 1), vec![coulomb])?,
    };

    input.control.iscfxc = 11;
    let vbh = super::no_scf_pot_total_potential(&input, &overlap)?;
    assert_close(
        vbh[(0, 0)],
        coulomb + refeff_core::von_barth_hedin_potential(density_radius, 1.0)?,
        1.0e-12,
        "vBH no-SCF total potential",
    );

    input.control.iscfxc = 12;
    let pz = super::no_scf_pot_total_potential(&input, &overlap)?;
    assert_close(
        pz[(0, 0)],
        coulomb + refeff_core::perdew_zunger_vxc(density_radius)?,
        1.0e-12,
        "PZ no-SCF total potential",
    );
    assert!((vbh[(0, 0)] - pz[(0, 0)]).abs() > 1.0e-6);
    Ok(())
}

#[test]
fn atomic_module_builds_no_scf_valence_potential_for_high_exchange_selector() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.control.iscfxc = 11;
    let density = 3.0;
    let valence_density = 0.375;
    let coulomb = -0.2;
    let overlap = super::AtomicApotOverlapArrays {
        norman_radii: Array1::from_vec(vec![1.5]),
        magnetization_density: Array2::from_elem((1, 1), -1.25),
        overlapped_density: Array2::from_shape_vec((1, 1), vec![density])?,
        overlapped_valence_density: Array2::from_shape_vec((1, 1), vec![valence_density])?,
        overlapped_coulomb_potential: Array2::from_shape_vec((1, 1), vec![coulomb])?,
    };

    input.control.ixc = 5;
    let total = super::no_scf_pot_total_potential(&input, &overlap)?;
    let valence = super::no_scf_pot_valence_potential(&input, &overlap, &total)?;
    let valence_radius = (valence_density / 3.0_f64).powf(-1.0 / 3.0);
    assert_close(
        valence[(0, 0)],
        coulomb + refeff_core::von_barth_hedin_potential(valence_radius, 1.0)?,
        1.0e-12,
        "EXCHANGE 5 no-SCF valence potential",
    );
    assert!((valence[(0, 0)] - total[(0, 0)]).abs() > 1.0e-6);

    input.control.ixc = 6;
    let high_branch = super::no_scf_pot_valence_potential(&input, &overlap, &total)?;
    let core_radius = ((density - valence_density) / 3.0_f64).powf(-1.0 / 3.0);
    let magnetized_radius = (density / 3.0_f64).powf(-1.0 / 3.0);
    let magnetized_fermi_momentum = refeff_core::FEFF_FERMI_MOMENTUM_FACTOR / magnetized_radius;
    assert_close(
        high_branch[(0, 0)],
        total[(0, 0)]
            - refeff_core::dirac_hara_exchange_potential(core_radius, magnetized_fermi_momentum)?,
        1.0e-12,
        "EXCHANGE 6 no-SCF valence potential",
    );
    assert!((high_branch[(0, 0)] - total[(0, 0)]).abs() > 1.0e-6);
    Ok(())
}

#[test]
fn atomic_module_uses_corval_ldos_peak_overrides_in_selection() -> Result<()> {
    let mut input = beryllium_pot_input()?;
    input.scattering.corval_emin = -70.0;
    let geom = beryllium_single_potential_geom_dat();
    let pot = super::generated_no_scf_pot_bin(&input, Path::new("config.inp"), &geom)?;
    let mut states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    states[0].scf.orbital_energies.fill(0.0);
    states[0].scf.orbital_energies[0] = super::POT_CORVAL_HIGH_EV / refeff_core::FEFF_HARTREE_EV
        - 2.0 * super::POT_CORVAL_TOLERANCE_EV / refeff_core::FEFF_HARTREE_EV;
    states[0].valence_occupations[0] = 0.0;
    let atomic_numbers = super::no_scf_pot_atomic_numbers(&input, pot.potential_count())?;
    let base = super::no_scf_pot_core_valence_selection(
        &input,
        &states,
        &atomic_numbers,
        pot.scalars.interstitial_potential,
        None,
    )?;
    let marker = base
        .markers
        .first()
        .copied()
        .context("Be corval selection should expose at least one marker")?;
    let shifted_energy = marker.energy + 0.01;
    let mut peaks = Array2::<f64>::from_elem((4, pot.potential_count()), f64::NAN);
    peaks[(marker.angular, marker.potential)] = shifted_energy;

    let shifted = super::no_scf_pot_core_valence_selection(
        &input,
        &states,
        &atomic_numbers,
        pot.scalars.interstitial_potential,
        Some(peaks.view()),
    )?;
    let shifted_marker = shifted
        .markers
        .iter()
        .find(|candidate| {
            candidate.angular == marker.angular && candidate.potential == marker.potential
        })
        .context("shifted corval marker should still be present")?;

    assert_close(
        shifted_marker.energy,
        shifted_energy,
        1.0e-12,
        "LDOS peak override energy",
    );
    Ok(())
}

#[test]
fn atomic_module_applies_core_valence_reassignment_to_pot_state() -> Result<()> {
    let potentials = 1;
    let mut orbital_occupancy = Array2::<f64>::zeros((POT_BIN_ORBITALS, potentials));
    let mut valence_occupancy = Array2::<f64>::zeros((4, potentials));
    let mut valence_density = Array2::<f64>::from_elem((POT_BIN_RADIAL_POINTS, potentials), 0.5);
    let radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);
    let mut large_components =
        Array3::<f64>::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
    let mut small_components =
        Array3::<f64>::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
    for radial in 0..POT_BIN_RADIAL_POINTS {
        let radius = radii[radial];
        large_components[(radial, 2, 0)] = radius * 0.25;
        small_components[(radial, 2, 0)] = radius * 0.05;
        large_components[(radial, 1, 0)] = radius * 0.10;
        small_components[(radial, 1, 0)] = radius * 0.02;
    }
    let selection = super::PotCoreValenceSelection {
        core_valence_energy: -0.35,
        markers: vec![
            super::PotCoreValenceMarker {
                potential: 0,
                angular: 1,
                orbital: 2,
                energy: -0.40,
                initial_is_valence: false,
                is_valence: true,
            },
            super::PotCoreValenceMarker {
                potential: 0,
                angular: 0,
                orbital: 0,
                energy: -0.10,
                initial_is_valence: true,
                is_valence: true,
            },
        ],
    };

    super::no_scf_pot_apply_core_valence_selection(
        &selection,
        &mut orbital_occupancy,
        &mut valence_occupancy,
        &mut valence_density,
        large_components.view(),
        small_components.view(),
    )?;

    assert_eq!(valence_occupancy[(1, 0)], 6.0);
    assert_eq!(valence_occupancy[(0, 0)], 0.0);
    assert_eq!(orbital_occupancy[(2, 0)], 4.0);
    assert_eq!(orbital_occupancy[(1, 0)], 2.0);
    assert_eq!(orbital_occupancy[(0, 0)], 0.0);

    let expected_density = 0.5
        + 4.0 * (0.25_f64.powi(2) + 0.05_f64.powi(2))
        + 2.0 * (0.10_f64.powi(2) + 0.02_f64.powi(2));
    assert_close(
        valence_density[(0, 0)],
        expected_density,
        1.0e-12,
        "core-valence density row 1",
    );
    assert_close(
        valence_density[(POT_BIN_RADIAL_POINTS - 1, 0)],
        expected_density,
        1.0e-12,
        "core-valence density final row",
    );
    Ok(())
}

#[test]
fn atomic_module_ignores_high_l_norman_valence_count_channels() -> Result<()> {
    let configuration = super::OrbitalConfiguration {
        orbital_count: 3,
        core_orbital_count: 0,
        projection_orbitals: Array1::zeros(10),
        hole_position: 0,
        principal_quantum_numbers: Array1::from_vec(vec![2, 4, 5]),
        kappa: Array1::from_vec(vec![-1, 3, -5]),
        electron_counts: Array1::from_vec(vec![2.0, 1.5, 0.5]),
        valence_counts: Array1::from_vec(vec![2.0, 1.5, 0.5]),
        spin_magnetization: Array1::zeros(3),
        ionization_orbital: 0,
        screening_orbital: 0,
        last_occupied_orbital: 0,
        template_atomic_number: 4,
        ionicity_delta: 0.0,
    };

    let counts = super::atomic_norman_valence_counts_from_configurations(&[configuration], 1)?;

    assert_eq!(counts.dim(), (4, 1));
    assert_eq!(counts[(0, 0)], 2.0);
    assert_eq!(counts[(3, 0)], 1.5);
    assert_eq!(counts.column(0).sum(), 3.5);
    Ok(())
}

#[test]
fn atomic_module_keeps_unit_amplitude_reduction_without_core_hole() -> Result<()> {
    let input = beryllium_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;

    let s02 = super::atomic_apot_amplitude_reduction_from_states(
        &input,
        Path::new("config.inp"),
        &states,
    )?;

    assert_eq!(s02, 1.0);
    Ok(())
}

#[test]
fn atomic_module_derives_amplitude_reduction_from_relaxed_absorber_overlaps() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let mut swapped = input.clone();
    swapped.run.nohole = -1;
    let swapped_states = super::generated_atomic_scf_states(&swapped, Path::new("config.inp"))?;

    let s02 = super::atomic_apot_amplitude_reduction_from_states(
        &input,
        Path::new("config.inp"),
        &states,
    )?;
    let swapped_s02 = super::atomic_apot_amplitude_reduction_from_states(
        &swapped,
        Path::new("config.inp"),
        &swapped_states,
    )?;

    assert!(s02.is_finite());
    assert!(s02 > 0.0 && s02 <= 1.0, "unexpected s02 {s02}");
    assert_close(swapped_s02, s02, 1.0e-9 * s02.abs().max(1.0), "swapped s02");
    Ok(())
}

#[test]
fn atomic_module_keeps_zero_energy_scalars_without_core_hole() -> Result<()> {
    let input = beryllium_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let static_arrays = beryllium_single_potential_static_arrays()?;
    let overlap_arrays =
        super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    let scalars = super::atomic_apot_energy_scalars_from_states(
        &input,
        Path::new("config.inp"),
        &states,
        &overlap_arrays,
    )?;
    let initial_total = super::atomic_total_energy_from_state(&input, 0, &states[0])?;

    assert_close(
        scalars.initial_total_energy,
        initial_total.total,
        1.0e-9 * initial_total.total.abs().max(1.0),
        "no-hole initial total energy",
    );
    assert_close(
        scalars.final_total_energy,
        initial_total.total,
        1.0e-9 * initial_total.total.abs().max(1.0),
        "no-hole final total energy",
    );
    assert_eq!(scalars.frozen_orbital_energy, 0.0);
    assert_eq!(scalars.relaxation_energy, 0.0);
    assert_eq!(scalars.edge_energy, 0.0);
    Ok(())
}

#[test]
fn atomic_module_derives_energy_scalars_from_generated_states() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let static_arrays = beryllium_single_potential_static_arrays()?;
    let overlap_arrays =
        super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    let scalars = super::atomic_apot_energy_scalars_from_states(
        &input,
        Path::new("config.inp"),
        &states,
        &overlap_arrays,
    )?;
    let state_count = super::apot_state_count(&input)?;
    let (initial_column, final_column) =
        super::atomic_apot_absorber_state_columns(&input, state_count)?;
    let initial_total =
        super::atomic_total_energy_from_state(&input, initial_column, &states[initial_column])?;
    let final_total =
        super::atomic_total_energy_from_state(&input, final_column, &states[final_column])?;
    let frozen_orbital_energy =
        super::atomic_apot_frozen_orbital_energy(&input, Path::new("config.inp"), &states[0])?;
    let adiabatic_edge = final_total.total - initial_total.total;
    let expected_relaxation = -frozen_orbital_energy - adiabatic_edge;
    let expected_edge = if adiabatic_edge <= 0.0 {
        -frozen_orbital_energy
    } else {
        adiabatic_edge
    } + states[0].scf.coulomb_potential[0]
        - overlap_arrays.overlapped_coulomb_potential[(0, 0)];

    assert!(scalars.frozen_orbital_energy.is_finite());
    assert!(scalars.frozen_orbital_energy < 0.0);
    assert_close(
        scalars.initial_total_energy,
        initial_total.total,
        1.0e-9 * initial_total.total.abs().max(1.0),
        "initial total energy",
    );
    assert_close(
        scalars.final_total_energy,
        final_total.total,
        1.0e-9 * final_total.total.abs().max(1.0),
        "final total energy",
    );
    assert_close(
        scalars.frozen_orbital_energy,
        frozen_orbital_energy,
        1.0e-12 * frozen_orbital_energy.abs().max(1.0),
        "frozen orbital energy",
    );
    assert_close(
        scalars.relaxation_energy,
        expected_relaxation,
        1.0e-9 * expected_relaxation.abs().max(1.0),
        "relaxation energy",
    );
    assert_close(
        scalars.edge_energy,
        expected_edge,
        1.0e-9 * expected_edge.abs().max(1.0),
        "edge energy",
    );
    Ok(())
}

#[test]
fn atomic_module_preserves_absorber_energy_scalars_when_nohole_swaps_columns() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let static_arrays = beryllium_single_potential_static_arrays()?;
    let overlap_arrays =
        super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;
    let scalars = super::atomic_apot_energy_scalars_from_states(
        &input,
        Path::new("config.inp"),
        &states,
        &overlap_arrays,
    )?;

    let mut swapped = input.clone();
    swapped.run.nohole = -1;
    let swapped_states = super::generated_atomic_scf_states(&swapped, Path::new("config.inp"))?;
    let swapped_overlap_arrays =
        super::atomic_apot_overlap_arrays_from_states(&swapped, &static_arrays, &swapped_states)?;
    let swapped_scalars = super::atomic_apot_energy_scalars_from_states(
        &swapped,
        Path::new("config.inp"),
        &swapped_states,
        &swapped_overlap_arrays,
    )?;

    assert_close(
        swapped_scalars.initial_total_energy,
        scalars.initial_total_energy,
        1.0e-9 * scalars.initial_total_energy.abs().max(1.0),
        "swapped initial total energy",
    );
    assert_close(
        swapped_scalars.final_total_energy,
        scalars.final_total_energy,
        1.0e-9 * scalars.final_total_energy.abs().max(1.0),
        "swapped final total energy",
    );
    assert_close(
        swapped_scalars.frozen_orbital_energy,
        scalars.frozen_orbital_energy,
        1.0e-12 * scalars.frozen_orbital_energy.abs().max(1.0),
        "swapped frozen orbital energy",
    );
    assert_close(
        swapped_scalars.relaxation_energy,
        scalars.relaxation_energy,
        1.0e-9 * scalars.relaxation_energy.abs().max(1.0),
        "swapped relaxation energy",
    );
    assert!(
        swapped_scalars.edge_energy.is_finite(),
        "swapped edge energy should stay finite"
    );
    Ok(())
}

#[test]
fn atomic_module_derives_nohole_core_hole_columns_with_zero_density() -> Result<()> {
    let input = beryllium_core_hole_pot_input()?;

    let columns = super::generated_atomic_core_hole_columns(&input, Path::new("config.inp"))?;

    assert_eq!(columns.large_component.len(), POT_BIN_RADIAL_POINTS);
    assert_eq!(columns.small_component.len(), POT_BIN_RADIAL_POINTS);
    assert_eq!(columns.density.len(), POT_BIN_RADIAL_POINTS);
    assert_eq!(columns.coulomb_potential.len(), POT_BIN_RADIAL_POINTS);
    assert!(
        columns.large_component.iter().any(|value| *value != 0.0),
        "core-hole large component should still be copied for fpf0 input"
    );
    assert!(columns.density.iter().all(|value| *value == 0.0));
    assert!(columns.coulomb_potential.iter().all(|value| *value == 0.0));
    Ok(())
}

#[test]
fn atomic_module_derives_nohole_one_core_hole_density_from_initial_orbital() -> Result<()> {
    let mut input = beryllium_core_hole_pot_input()?;
    input.run.nohole = 1;

    let columns = super::generated_atomic_core_hole_columns(&input, Path::new("config.inp"))?;
    let radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);

    assert!(columns.large_component.iter().any(|value| *value != 0.0));
    assert!(columns.small_component.iter().any(|value| *value != 0.0));
    assert!(columns.density.iter().any(|value| *value > 0.0));
    for row in [0_usize, 23, 97] {
        let expected_density = (columns.large_component[row] * columns.large_component[row]
            + columns.small_component[row] * columns.small_component[row])
            / (2.0 * radii[row] * radii[row]);
        assert_close(
            columns.density[row],
            expected_density,
            1.0e-9 * expected_density.abs().max(1.0),
            &format!("drho[{row}]"),
        );
    }

    let expected_coulomb =
        apot_core_hole_coulomb_from_density(columns.density.view(), input.run.nohole)?;
    for row in [0_usize, 23, 97] {
        assert_close(
            columns.coulomb_potential[row],
            expected_coulomb[row],
            1.0e-9 * expected_coulomb[row].abs().max(1.0),
            &format!("dvcoul[{row}]"),
        );
    }
    Ok(())
}

#[test]
fn atomic_module_derives_transition_core_hole_density_from_state_difference() -> Result<()> {
    let mut input = beryllium_core_hole_pot_input()?;
    input.run.nohole = 2;

    let columns = super::generated_atomic_core_hole_columns(&input, Path::new("config.inp"))?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let final_state = states.last().context("missing final absorber state")?;

    assert!(columns.density.iter().any(|value| *value != 0.0));
    for row in [0_usize, 23, 97] {
        let expected_density = 0.5
            * (states[0].scf.density_4pi[row]
                - states[0].scf.valence_density_4pi[row]
                - final_state.scf.density_4pi[row]
                + final_state.scf.valence_density_4pi[row]);
        assert_close(
            columns.density[row],
            expected_density,
            1.0e-9 * expected_density.abs().max(1.0),
            &format!("transition drho[{row}]"),
        );
    }

    let expected_coulomb =
        apot_core_hole_coulomb_from_density(columns.density.view(), input.run.nohole)?;
    for row in [0_usize, 23, 97] {
        assert_close(
            columns.coulomb_potential[row],
            expected_coulomb[row],
            1.0e-9 * expected_coulomb[row].abs().max(1.0),
            &format!("transition dvcoul[{row}]"),
        );
    }
    Ok(())
}

#[test]
fn atomic_module_derives_static_apot_arrays_from_pot_geom_handoffs() -> Result<()> {
    let mut input = copper_two_potential_pot_input()?;
    input.overlap_shells = vec![
        vec![PotOverlapShell {
            iphovr: 1,
            nnovr: 2,
            rovr: 2.0 * FEFF_BOHR_ANGSTROM,
        }],
        vec![
            PotOverlapShell {
                iphovr: 0,
                nnovr: 1,
                rovr: FEFF_BOHR_ANGSTROM,
            },
            PotOverlapShell {
                iphovr: 1,
                nnovr: 3,
                rovr: 3.0 * FEFF_BOHR_ANGSTROM,
            },
        ],
    ];
    let geom = sample_atomic_geom_dat();
    let pot = sample_pot_bin();

    let arrays = super::atomic_apot_static_arrays_from_handoffs(&input, &geom, &pot)?;

    assert_eq!(arrays.unique_potential_count, 2);
    assert_eq!(arrays.atom_count, 3);
    assert_eq!(arrays.atomic_numbers.to_vec(), vec![29, 29]);
    assert_eq!(arrays.model_atom_indices.to_vec(), vec![1, 2]);
    assert_eq!(arrays.overlap_shell_counts.to_vec(), vec![1, 2]);
    assert_eq!(arrays.norman_radii.to_vec(), vec![2.1, 2.1]);
    assert_eq!(arrays.atom_potential_indices.to_vec(), vec![0, 1, 1]);
    assert_eq!(arrays.atom_positions.dim(), (3, 3));
    assert_close(arrays.atom_positions[(0, 1)], 1.0, 1.0e-12, "rat x atom 2");
    assert_close(arrays.atom_positions[(1, 2)], 2.0, 1.0e-12, "rat y atom 3");

    assert_eq!(arrays.overlap_potential_indices.dim(), (2, 2));
    assert_eq!(arrays.overlap_potential_indices[[0, 0]], 1);
    assert_eq!(arrays.overlap_potential_indices[[1, 0]], 0);
    assert_eq!(arrays.overlap_potential_indices[[0, 1]], 0);
    assert_eq!(arrays.overlap_potential_indices[[1, 1]], 1);
    assert_eq!(arrays.overlap_shell_atom_counts[[0, 0]], 2);
    assert_eq!(arrays.overlap_shell_atom_counts[[1, 1]], 3);
    assert_close(arrays.overlap_radii[[0, 0]], 2.0, 1.0e-12, "rovr 0,0");
    assert_close(arrays.overlap_radii[[0, 1]], 1.0, 1.0e-12, "rovr 0,1");
    assert_close(arrays.overlap_radii[[1, 1]], 3.0, 1.0e-12, "rovr 1,1");
    Ok(())
}

#[test]
fn atomic_module_rejects_static_apot_atomic_number_mismatch() -> Result<()> {
    let input = copper_two_potential_pot_input()?;
    let geom = sample_atomic_geom_dat();
    let mut pot = sample_pot_bin();
    pot.atomic_numbers[1] = 30;

    let error = super::atomic_apot_static_arrays_from_handoffs(&input, &geom, &pot)
        .err()
        .context("mismatched pot.bin atomic number should fail")?;

    assert!(error.to_string().contains("does not match pot.inp"));
    Ok(())
}

#[test]
fn atomic_module_derives_overlap_apot_arrays_from_single_potential_shell() -> Result<()> {
    let input = beryllium_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let static_arrays = super::AtomicApotStaticArrays {
        unique_potential_count: 1,
        atom_count: 1,
        atomic_numbers: Array1::from_vec(vec![4]),
        model_atom_indices: Array1::from_vec(vec![1]),
        overlap_shell_counts: Array1::from_vec(vec![1]),
        norman_radii: Array1::from_vec(vec![1.0]),
        atom_potential_indices: Array1::from_vec(vec![0]),
        atom_positions: Array2::zeros((3, 1)),
        overlap_potential_indices: Array2::from_shape_vec((1, 1), vec![0])?,
        overlap_shell_atom_counts: Array2::from_shape_vec((1, 1), vec![1])?,
        overlap_radii: Array2::from_shape_vec((1, 1), vec![2.0])?,
    };

    let arrays = super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    assert_eq!(arrays.norman_radii.len(), 1);
    assert!(arrays.norman_radii[0].is_finite());
    assert!(arrays.norman_radii[0] > 0.0);
    assert_eq!(
        arrays.magnetization_density.dim(),
        (POT_BIN_RADIAL_POINTS, 2)
    );
    assert_eq!(arrays.overlapped_density.dim(), (POT_BIN_RADIAL_POINTS, 1));
    assert_eq!(
        arrays.overlapped_valence_density.dim(),
        (POT_BIN_RADIAL_POINTS, 1)
    );
    assert_eq!(
        arrays.overlapped_coulomb_potential.dim(),
        (POT_BIN_RADIAL_POINTS, 1)
    );
    assert!(
        arrays
            .magnetization_density
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        arrays
            .magnetization_density
            .column(0)
            .iter()
            .any(|value| *value != 0.0),
        "FEFF9 Be spin-polarizable orbital should produce an atomic dmag profile"
    );
    assert!(arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23]);
    assert!(arrays.overlapped_valence_density[(23, 0)] > states[0].scf.valence_density_4pi[23]);
    Ok(())
}

#[test]
fn atomic_module_builds_free_spin_density_from_normalized_atomic_moment() -> Result<()> {
    let input = beryllium_pot_input()?;
    let mut states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let state = &mut states[0];

    state.spin_magnetization.fill(0.0);
    assert!(
        super::atomic_apot_free_spin_density_from_state(0, state)?
            .iter()
            .all(|value| *value == 0.0),
        "a zero xmag configuration should keep an exactly zero spin density"
    );

    state.spin_magnetization[0] = 5.0;
    let density = super::atomic_apot_free_spin_density_from_state(0, state)?;
    for &row in &[0, 31, POT_BIN_RADIAL_POINTS - 1] {
        let radius = state.initial_orbitals.radii[row];
        let expected = (state.scf.large_components[(row, 0)].powi(2)
            + state.scf.small_components[(row, 0)].powi(2))
            / radius.powi(2);
        assert_close(
            density[row],
            expected,
            1.0e-12 * expected.abs().max(1.0),
            &format!("normalized atomic spin density row {row}"),
        );
    }

    state.spin_magnetization.fill(0.0);
    state.spin_magnetization[0] = 1.0;
    state.spin_magnetization[1] = -1.0;
    let zero_moment_density = super::atomic_apot_free_spin_density_from_state(0, state)?;
    for &row in &[0, 31, POT_BIN_RADIAL_POINTS - 1] {
        let radius = state.initial_orbitals.radii[row];
        let expected = (state.scf.large_components[(row, 0)].powi(2)
            + state.scf.small_components[(row, 0)].powi(2)
            - state.scf.large_components[(row, 1)].powi(2)
            - state.scf.small_components[(row, 1)].powi(2))
            / radius.powi(2);
        assert_close(
            zero_moment_density[row],
            expected,
            1.0e-12 * expected.abs().max(1.0),
            &format!("unnormalized zero-moment atomic spin density row {row}"),
        );
    }

    state.spin_magnetization.fill(0.0);
    state.spin_magnetization[0] = -2.0;
    let negative_density = super::atomic_apot_free_spin_density_from_state(0, state)?;
    for &row in &[0, 31, POT_BIN_RADIAL_POINTS - 1] {
        let radius = state.initial_orbitals.radii[row];
        let expected = -2.0
            * (state.scf.large_components[(row, 0)].powi(2)
                + state.scf.small_components[(row, 0)].powi(2))
            / radius.powi(2);
        assert_close(
            negative_density[row],
            expected,
            1.0e-12 * expected.abs().max(1.0),
            &format!("unnormalized negative atomic spin density row {row}"),
        );
    }

    state.spin_magnetization[0] = f64::NAN;
    let error = super::atomic_apot_free_spin_density_from_state(0, state)
        .err()
        .context("non-finite spin occupation should fail closed")?;
    assert!(error.to_string().contains("non-finite"), "{error:?}");
    Ok(())
}

#[test]
fn atomic_module_normalizes_isolated_apot_density_for_norman_radius() -> Result<()> {
    let input = beryllium_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let static_arrays = super::AtomicApotStaticArrays {
        unique_potential_count: 1,
        atom_count: 1,
        atomic_numbers: Array1::from_vec(vec![4]),
        model_atom_indices: Array1::from_vec(vec![1]),
        overlap_shell_counts: Array1::from_vec(vec![0]),
        norman_radii: Array1::from_vec(vec![1.0]),
        atom_potential_indices: Array1::from_vec(vec![0]),
        atom_positions: Array2::zeros((3, 1)),
        overlap_potential_indices: Array2::zeros((1, 1)),
        overlap_shell_atom_counts: Array2::zeros((1, 1)),
        overlap_radii: Array2::zeros((1, 1)),
    };

    let raw = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: states[0].scf.density_4pi.view(),
        atomic_number: 4,
    });
    assert!(
        matches!(raw, Err(GridError::InsufficientNormanCharge { .. })),
        "raw isolated generated Be density should exercise the APOT normalization edge"
    );

    let arrays = super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    assert_eq!(arrays.overlapped_density.dim(), (POT_BIN_RADIAL_POINTS, 1));
    assert!(arrays.norman_radii[0].is_finite());
    assert!(arrays.norman_radii[0] > 0.0);
    assert!(
        arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23],
        "normalized isolated density should scale the source density upward"
    );
    Ok(())
}

#[test]
fn atomic_module_applies_explicit_overlap_shells_from_static_arrays() -> Result<()> {
    let input = beryllium_two_potential_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let mut atom_positions = Array2::<f64>::zeros((3, 2));
    atom_positions[(0, 1)] = 20.0;
    let static_arrays = super::AtomicApotStaticArrays {
        unique_potential_count: 2,
        atom_count: 2,
        atomic_numbers: Array1::from_vec(vec![4, 4]),
        model_atom_indices: Array1::from_vec(vec![1, 2]),
        overlap_shell_counts: Array1::from_vec(vec![1, 1]),
        norman_radii: Array1::from_vec(vec![1.0, 1.0]),
        atom_potential_indices: Array1::from_vec(vec![0, 1]),
        atom_positions,
        overlap_potential_indices: Array2::from_shape_vec((1, 2), vec![1, 0])?,
        overlap_shell_atom_counts: Array2::from_shape_vec((1, 2), vec![2, 1])?,
        overlap_radii: Array2::from_shape_vec((1, 2), vec![2.0, 2.5])?,
    };

    let arrays = super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    assert_eq!(arrays.norman_radii.len(), 2);
    assert_eq!(arrays.overlapped_density.dim(), (POT_BIN_RADIAL_POINTS, 2));
    assert!(arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23]);
    assert!(arrays.overlapped_valence_density[(23, 0)] > states[0].scf.valence_density_4pi[23]);
    assert!(arrays.overlapped_density[(23, 1)] > states[1].scf.density_4pi[23]);
    Ok(())
}

#[test]
fn atomic_module_applies_geometry_overlap_from_static_arrays() -> Result<()> {
    let input = beryllium_two_potential_pot_input()?;
    let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
    let mut atom_positions = Array2::<f64>::zeros((3, 2));
    atom_positions[(0, 1)] = 2.0;
    let static_arrays = super::AtomicApotStaticArrays {
        unique_potential_count: 2,
        atom_count: 2,
        atomic_numbers: Array1::from_vec(vec![4, 4]),
        model_atom_indices: Array1::from_vec(vec![1, 2]),
        overlap_shell_counts: Array1::from_vec(vec![0, 0]),
        norman_radii: Array1::from_vec(vec![1.0, 1.0]),
        atom_potential_indices: Array1::from_vec(vec![0, 1]),
        atom_positions,
        overlap_potential_indices: Array2::zeros((1, 2)),
        overlap_shell_atom_counts: Array2::zeros((1, 2)),
        overlap_radii: Array2::zeros((1, 2)),
    };

    let arrays = super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

    assert!(arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23]);
    assert!(arrays.overlapped_density[(23, 1)] > states[1].scf.density_4pi[23]);
    Ok(())
}

#[test]
fn atomic_module_recovers_malformed_config_handoff_without_apot_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;

    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    assert!(has_supported_config_handoff(temp.path())?);
    assert!(!has_cached_atomic_output(temp.path())?);
    Ok(())
}

#[test]
fn atomic_module_does_not_claim_malformed_custom_config_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = copper_two_potential_pot_input()?;
    input.config_type = 2;
    std::fs::write(
        temp.path().join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    std::fs::write(temp.path().join("config.inp"), "not a config.inp handoff\n")?;

    assert!(!has_supported_config_handoff(temp.path())?);

    let error = super::run_supported_config_handoff_in_dir(temp.path())
        .err()
        .context("malformed custom config.inp should fail through explicit config handoff")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("config.inp"), "{chain}");
    assert!(!temp.path().join("config.dat").exists());
    assert!(!temp.path().join("log1.dat").exists());
    Ok(())
}

#[test]
fn atomic_module_validates_existing_supported_config_handoff_without_apot_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    super::run_supported_config_handoff_in_dir(temp.path())?;
    let expected = read_config_dat(temp.path().join("config.dat"))?;

    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(read_config_dat(temp.path().join("config.dat"))?, expected);
    assert!(!temp.path().join("apot.bin").exists());
    assert!(!temp.path().join("log1.dat").exists());
    Ok(())
}

#[test]
fn atomic_module_recovers_existing_malformed_log_for_config_handoff_without_apot_solver()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    super::run_supported_config_handoff_in_dir(temp.path())?;
    std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    assert!(!temp.path().join("apot.bin").exists());
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_log_contains(&log, "Calculating atomic potentials ...");
    assert_log_contains(&log, "Done with module: atomic potentials.");
    Ok(())
}

#[test]
fn atomic_module_recovers_stale_existing_config_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    write_config_dat(temp.path().join("config.dat"), &sample_config_dat())?;

    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    assert!(has_supported_config_handoff(temp.path())?);
    assert!(!has_cached_atomic_output(temp.path())?);
    Ok(())
}

#[test]
fn atomic_module_does_not_advertise_malformed_apot_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

    assert!(!has_cached_atomic_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed apot.bin should fail through the explicit ATOM runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("apot.bin"), "{chain}");
    Ok(())
}

#[test]
fn atomic_module_validates_config_handoff_when_malformed_apot_cache_exists() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

    assert!(!has_cached_atomic_output(temp.path())?);
    assert!(has_supported_config_handoff(temp.path())?);
    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );

    let error = run_in_dir(temp.path())
        .err()
        .context("source-backed config handoff should reach the ATOM geometry gate")?;

    assert!(
        error
            .to_string()
            .contains("ATOM source apot.bin generation requires geom.dat handoff"),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn atomic_module_recovers_config_log_when_pot_cache_is_unusable() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
    std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;
    std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;
    std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_supported_config_handoff(temp.path())?);

    let count = super::run_supported_config_handoff_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_log_contains(&log, "Calculating atomic potentials ...");
    assert_log_contains(&log, "Done with module: atomic potentials.");
    Ok(())
}

#[test]
fn atomic_module_recovers_malformed_config_sidecar_from_pot_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
    std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;

    assert!(has_cached_atomic_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_log_contains(&log, "Calculating atomic potentials ...");
    assert_log_contains(&log, "Done with module: atomic potentials.");
    Ok(())
}

#[test]
fn atomic_module_recovers_stale_config_sidecar_from_pot_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
    write_config_dat(temp.path().join("config.dat"), &sample_config_dat())?;

    assert!(has_cached_atomic_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_log_contains(&log, "Calculating atomic potentials ...");
    assert_log_contains(&log, "Done with module: atomic potentials.");
    Ok(())
}

#[test]
fn atomic_module_recovers_malformed_module_log_for_config_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
    std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;
    std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_cached_atomic_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_log_contains(&log, "Calculating atomic potentials ...");
    assert_log_contains(&log, "Done with module: atomic potentials.");
    Ok(())
}

#[test]
fn atomic_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
    write_generated_config_dat(temp.path())?;
    std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(!has_cached_atomic_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed cached log1.dat should fail through the explicit ATOM runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("log1.dat"), "{chain}");
    Ok(())
}

#[test]
fn atomic_module_does_not_advertise_malformed_fpf0_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
    std::fs::write(temp.path().join("fpf0.dat"), "not fpf0.dat\n")?;

    assert!(!has_cached_atomic_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed fpf0.dat should fail through the explicit ATOM runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("fpf0.dat"), "{chain}");
    Ok(())
}

#[test]
fn atomic_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    let apot_path = temp.path().join("apot.bin");
    let config_path = temp.path().join("config.dat");
    let fpf0_path = temp.path().join("fpf0.dat");
    let log_path = temp.path().join("log1.dat");
    write_apot_bin(&apot_path, &sample_apot_bin())?;
    write_generated_config_dat(temp.path())?;
    write_fpf0_dat(&fpf0_path, &sample_fpf0_dat())?;
    write_module_log_dat(&log_path, &sample_module_log())?;
    let expected_apot = read_apot_bin(&apot_path)?;
    let expected_config = read_config_dat(&config_path)?;
    let expected_fpf0 = read_fpf0_dat(&fpf0_path)?;
    let expected_log = read_module_log_dat(&log_path)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(has_cached_atomic_output(temp.path())?);
    assert_eq!(read_apot_bin(&apot_path)?, expected_apot);
    assert_eq!(read_config_dat(&config_path)?, expected_config);
    assert_eq!(read_fpf0_dat(&fpf0_path)?, expected_fpf0);
    assert_eq!(read_module_log_dat(&log_path)?, expected_log);
    Ok(())
}

#[test]
fn atomic_module_generates_missing_module_log_from_cached_apot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_log_contains(&log, "Calculating atomic potentials ...");
    assert_log_contains(
        &log,
        "    overlapped atomic potential and density for unique potential    0",
    );
    assert_log_contains(
        &log,
        "    overlapped atomic potential and density for unique potential    1",
    );
    assert_log_contains(&log, "Done with module: atomic potentials.");
    Ok(())
}

#[test]
fn atomic_module_generates_second_overlap_pass_for_ionized_potentials() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    let pot_path = temp.path().join("pot.inp");
    let pot_text = std::fs::read_to_string(&pot_path)?;
    let mut input = PotInput::parse_str(&pot_path, &pot_text)?;
    input.potentials[1].xion = 0.25;
    std::fs::write(&pot_path, refeff_io::pot_input_string(&input)?)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let log = read_module_log_dat(temp.path().join("log1.dat"))?;
    assert_eq!(
        log_line_occurrences(
            &log,
            "    overlapped atomic potential and density for unique potential    0"
        ),
        2
    );
    assert_eq!(
        log_line_occurrences(
            &log,
            "    overlapped atomic potential and density for unique potential    1"
        ),
        2
    );
    Ok(())
}

#[test]
fn atomic_module_validates_cached_core_hole_coulomb_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    set_nohole(temp.path(), 2)?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_transition_apot_bin()?)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    Ok(())
}

#[test]
fn atomic_module_regenerates_corrupt_cached_core_hole_coulomb() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    set_nohole(temp.path(), 2)?;
    let mut apot = sample_transition_apot_bin()?;
    let expected = rendered_apot_bin(&sample_transition_apot_bin()?)?;
    let records = apot.sections[0]
        .records()
        .context("sample transition apot should have core-hole records")?;
    let mut rows = records.rows.clone();
    rows[10][3] = ApotBinValue::Real(10.0);
    if let ApotBinPayload::Records(records) = &mut apot.sections[0].payload {
        records.rows = rows;
    }
    write_apot_bin(temp.path().join("apot.bin"), &apot)?;

    let count = run_in_dir(temp.path())?;
    let actual = read_apot_bin(temp.path().join("apot.bin"))?;

    assert_eq!(count, 3);
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn atomic_module_regenerates_nohole_core_hole_coulomb_to_zero() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_input(temp.path(), 1)?;
    let mut apot = sample_apot_bin();
    let records = apot.sections[0]
        .records()
        .context("sample apot should have core-hole records")?;
    let mut rows = records.rows.clone();
    rows[10][3] = ApotBinValue::Real(10.0);
    if let ApotBinPayload::Records(records) = &mut apot.sections[0].payload {
        records.rows = rows;
    }
    write_apot_bin(temp.path().join("apot.bin"), &apot)?;

    let count = run_in_dir(temp.path())?;
    let actual = read_apot_bin(temp.path().join("apot.bin"))?;

    assert_eq!(count, 3);
    assert_eq!(actual, rendered_apot_bin(&sample_apot_bin())?);
    Ok(())
}

#[test]
fn atomic_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        crate::require_fixture!("ATOM reference test; generated EXAFS/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    for name in ["config.dat", "fpf0.dat", "log1.dat"] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
    let expected_config = optional_config_dat(temp.path().join("config.dat"))?;
    let expected_fpf0 = optional_fpf0_dat(temp.path().join("fpf0.dat"))?;
    let expected_log = optional_module_log(temp.path().join("log1.dat"))?;

    let count = run_in_dir(temp.path())?;

    let optional_count = [
        expected_config.as_ref().map(|_| 1_usize),
        expected_fpf0.as_ref().map(|_| 1_usize),
        Some(1_usize),
    ]
    .into_iter()
    .flatten()
    .sum::<usize>();
    assert_eq!(count, 1 + optional_count);
    assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
    if let Some(expected) = expected_config {
        assert_eq!(read_config_dat(temp.path().join("config.dat"))?, expected);
    }
    if let Some(expected) = expected_fpf0 {
        assert_eq!(read_fpf0_dat(temp.path().join("fpf0.dat"))?, expected);
    }
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log1.dat"))?, expected);
    } else {
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(&log, "Done with module: atomic potentials.");
    }
    Ok(())
}

#[test]
fn atomic_module_generates_missing_reference_config_when_absent() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        crate::require_fixture!(
            "ATOM config generation test; generated EXAFS/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_config = read_config_dat(reference_dir.join("config.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?,
        expected_config
    );
    Ok(())
}

#[test]
fn atomic_module_generates_reference_config_before_missing_geometry_error_when_present()
-> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        crate::require_fixture!(
            "ATOM pre-solver config test; generated EXAFS/Cu reference not found"
        );
    };
    let expected_path = reference_dir.join("config.dat");
    if !expected_path.is_file() {
        crate::require_fixture!(
            "ATOM pre-solver config test; generated EXAFS/Cu config.dat not found"
        );
    }

    let temp = tempfile::tempdir()?;
    std::fs::copy(reference_dir.join("pot.inp"), temp.path().join("pot.inp"))?;
    let expected_config = read_config_dat(expected_path)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("ATOM without apot.bin should still require source geometry")?;

    assert!(
        error
            .to_string()
            .contains("ATOM source apot.bin generation requires geom.dat handoff")
    );
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?,
        expected_config
    );
    Ok(())
}

#[test]
fn atomic_module_generates_missing_reference_fpf0_when_absent() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        crate::require_fixture!(
            "ATOM fpf0 generation test; generated EXAFS/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("fort.16").is_file());
    assert_fpf0_close(
        &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
        &expected_fpf0,
    );
    Ok(())
}

#[test]
fn atomic_total_energy_from_reference_apot_matches_fort16() -> Result<()> {
    let cases = [
        ("EXAFS/Cu", reference_atomic_dir()?),
        ("XANES/Cu", reference_atomic_transition_dir()?),
        ("NRIXS/GeCl_4", reference_atomic_nohole_dir()?),
    ];
    for (label, reference_dir) in cases {
        let Some(reference_dir) = reference_dir else {
            crate::record_missing_fixture!("{label}; reference not found");
            continue;
        };
        let pot_path = reference_dir.join("pot.inp");
        let pot_text = std::fs::read_to_string(&pot_path)?;
        let input = PotInput::parse_str(&pot_path, &pot_text)?;
        let apot = read_apot_bin(reference_dir.join("apot.bin"))?;
        assert!(super::has_fpf0_total_energy_source_sections(&apot, &input)?);
        let config_inp = reference_dir.join("config.inp");
        let actual = super::generated_atomic_total_energy(&apot, &input, &config_inp)?;
        let fort16 = read_fort16(reference_dir.join("fort.16"))?;
        let expected = fort16
            .total_energy_hartree
            .iter()
            .next_back()
            .copied()
            .with_context(|| format!("{label} reference fort.16 has no total-energy rows"))?;
        let difference = (actual.total - expected).abs();
        assert!(
            difference <= 7.5e-5,
            concat!(
                "{label} total_energy differs by {difference:e}: ",
                "actual={actual_total:e}, expected={expected:e}, ",
                "direct={direct:e}, exchange={exchange:e}, ",
                "magnetic_breit={magnetic_breit:e}, retarded_breit={retarded_breit:e}"
            ),
            label = label,
            difference = difference,
            actual_total = actual.total,
            expected = expected,
            direct = actual.direct_coulomb,
            exchange = actual.exchange_coulomb,
            magnetic_breit = actual.magnetic_breit,
            retarded_breit = actual.retarded_breit,
        );
    }
    Ok(())
}

#[test]
fn atomic_module_matches_mnf2_and_gd_atomic_magnetization_density() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to resolve workspace root")?;
    for example in ["XMCD/MnF2_SPXAS", "XMCD/Gd_L1"] {
        let reference_dir = workspace.join("reference-work/golden").join(example);
        if !reference_dir.join("apot.bin").is_file() {
            crate::require_fixture!(
                "ATOM magnetic-density reference regression; {example} apot.bin not found"
            );
        }
        let pot_path = reference_dir.join("pot.inp");
        let input = PotInput::parse_str(&pot_path, &std::fs::read_to_string(&pot_path)?)?;
        let geom = super::read_geom_dat(&reference_dir.join("geom.dat"))?;
        let pot = read_pot_bin(reference_dir.join("pot.bin"))?;
        let actual = super::generated_atomic_apot_bin(
            &input,
            &reference_dir.join("config.inp"),
            &geom,
            &pot,
        )?;
        let expected = read_apot_bin(reference_dir.join("apot.bin"))?;
        let actual_dmag = super::real_matrix_section(&actual, 9, "dmag")?;
        let expected_dmag = super::real_matrix_section(&expected, 9, "dmag")?;
        assert_eq!(
            actual_dmag.dim(),
            expected_dmag.dim(),
            "{example} dmag shape"
        );

        for potential in 0..actual_dmag.ncols() {
            let mut difference_squared = 0.0;
            let mut expected_squared = 0.0;
            for row in 0..actual_dmag.nrows() {
                difference_squared +=
                    (actual_dmag[(row, potential)] - expected_dmag[(row, potential)]).powi(2);
                expected_squared += expected_dmag[(row, potential)].powi(2);
            }
            if expected_squared == 0.0 {
                assert!(
                    difference_squared == 0.0,
                    "{example} dmag potential {potential} should stay exactly zero"
                );
            } else {
                let relative_l2 = (difference_squared / expected_squared).sqrt();
                assert!(
                    relative_l2 <= 5.0e-6,
                    "{example} dmag potential {potential} relative L2 {relative_l2:e} exceeds 5e-6"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn atomic_module_recovers_stale_reference_fpf0_when_source_handoff_present() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        crate::require_fixture!(
            "ATOM stale fpf0 recovery test; generated EXAFS/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;
    let mut stale_fpf0 = expected_fpf0.clone();
    stale_fpf0.atomic_number = 8;
    write_fpf0_dat(temp.path().join("fpf0.dat"), &stale_fpf0)?;

    assert!(has_cached_atomic_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("fort.16").is_file());
    assert_fpf0_close(
        &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
        &expected_fpf0,
    );
    Ok(())
}

#[test]
fn atomic_module_generates_missing_reference_nohole_config_when_absent() -> Result<()> {
    let Some(reference_dir) = reference_atomic_nohole_dir()? else {
        crate::require_fixture!(
            "ATOM nohole config generation test; generated NRIXS/GeCl_4 reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_config = read_config_dat(reference_dir.join("config.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_config_dat(temp.path().join("config.dat"))?,
        expected_config
    );
    Ok(())
}

#[test]
fn atomic_module_generates_missing_reference_nohole_fpf0_when_absent() -> Result<()> {
    let Some(reference_dir) = reference_atomic_nohole_dir()? else {
        crate::require_fixture!(
            "ATOM nohole fpf0 generation test; generated NRIXS/GeCl_4 reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("fort.16").is_file());
    assert_fpf0_close(
        &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
        &expected_fpf0,
    );
    Ok(())
}

#[test]
fn atomic_module_validates_reference_transition_core_hole_when_present() -> Result<()> {
    let Some(reference_dir) = reference_atomic_transition_dir()? else {
        crate::require_fixture!(
            "ATOM transition reference test; generated XANES/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }

    run_in_dir(temp.path())?;
    Ok(())
}

#[test]
fn atomic_module_generates_missing_reference_transition_fpf0_when_absent() -> Result<()> {
    let Some(reference_dir) = reference_atomic_transition_dir()? else {
        crate::require_fixture!(
            "ATOM transition fpf0 generation test; generated XANES/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "apot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("fort.16").is_file());
    assert_fpf0_close(
        &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
        &expected_fpf0,
    );
    Ok(())
}

fn write_pot_input(work_dir: &Path, mpot: i32) -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu atomic smoke test
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let mut pot_input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
    pot_input.control.mpot = mpot;
    std::fs::write(
        work_dir.join("pot.inp"),
        refeff_io::pot_input_string(&pot_input)?,
    )?;
    Ok(())
}

fn beryllium_pot_input() -> Result<PotInput> {
    let mut input = beryllium_core_hole_pot_input()?;
    input.control.ihole = 0;
    input.run.nohole = 0;
    Ok(input)
}

fn highz_pot_input(atomic_number: usize) -> Result<PotInput> {
    let input = FeffInput::parse_str(
        "feff.inp",
        &format!(
            r#"
TITLE HIGHZ Z={atomic_number}
CONTROL 1 0 0 0 0 0
PRINT 5 0 0 0 0 0
NOHOLE
HIGHZ
POTENTIALS
0 {atomic_number} X
ATOMS
0.0 0.0 0.0 0 X0
END
"#
        ),
    )?;
    let document = FeffDocument::from_input(&input)?;
    let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
    input.control.ihole = 0;
    input.control.ipr1 = 5;
    input.run.nohole = 0;
    input.finite_nucleus = true;
    Ok(input)
}

fn highz_finite_binding_energy(report: &str, atomic_number: usize) -> Result<f64> {
    let prefix = format!("{atomic_number}:");
    let line = report
        .lines()
        .find(|line| line.starts_with(&prefix))
        .with_context(|| format!("HIGHZ report is missing Z={atomic_number}"))?;
    line.split_whitespace()
        .nth(2)
        .context("HIGHZ row is missing its finite-nucleus binding energy")?
        .parse::<f64>()
        .with_context(|| format!("HIGHZ Z={atomic_number} binding energy is not numeric"))
}

fn beryllium_core_hole_pot_input() -> Result<PotInput> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Be atomic SCF smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
    input.run.nohole = 0;
    Ok(input)
}

fn copper_two_potential_pot_input() -> Result<PotInput> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu atomic static APOT smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
0.0 1.0 0.0 1 Cu2
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    Ok(PotInput::parse_str(
        "pot.inp",
        &rdinp::pot_inp_string(&document)?,
    )?)
}

fn beryllium_two_potential_pot_input() -> Result<PotInput> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Be two-potential overlap smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 4 Be
1 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
1.0 0.0 0.0 1 Be1
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    Ok(PotInput::parse_str(
        "pot.inp",
        &rdinp::pot_inp_string(&document)?,
    )?)
}

fn beryllium_single_potential_static_arrays() -> Result<super::AtomicApotStaticArrays> {
    Ok(super::AtomicApotStaticArrays {
        unique_potential_count: 1,
        atom_count: 1,
        atomic_numbers: Array1::from_vec(vec![4]),
        model_atom_indices: Array1::from_vec(vec![1]),
        overlap_shell_counts: Array1::from_vec(vec![1]),
        norman_radii: Array1::from_vec(vec![1.0]),
        atom_potential_indices: Array1::from_vec(vec![0]),
        atom_positions: Array2::zeros((3, 1)),
        overlap_potential_indices: Array2::from_shape_vec((1, 1), vec![0])?,
        overlap_shell_atom_counts: Array2::from_shape_vec((1, 1), vec![1])?,
        overlap_radii: Array2::from_shape_vec((1, 1), vec![2.0])?,
    })
}

fn write_beryllium_atomic_source_handoffs(work_dir: &Path) -> Result<()> {
    let input = beryllium_pot_input()?;
    std::fs::write(
        work_dir.join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    write_pot_bin(
        work_dir.join("pot.bin"),
        &beryllium_single_potential_pot_bin(),
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;
    Ok(())
}

fn write_beryllium_atomic_geometry_source_handoffs(work_dir: &Path) -> Result<()> {
    let input = beryllium_pot_input()?;
    std::fs::write(
        work_dir.join("pot.inp"),
        refeff_io::pot_input_string(&input)?,
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&beryllium_single_potential_geom_dat())?,
    )?;
    Ok(())
}

fn beryllium_single_potential_geom_dat() -> GeomDat {
    GeomDat {
        nat: 1,
        nph: 0,
        model_atoms: vec![1],
        atoms: vec![GeomDatRow {
            index: 1,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            iph: 0,
            boundary: 0,
        }],
    }
}

fn beryllium_single_potential_config_dat() -> ConfigDatData {
    let mut occupations = Array1::zeros(40);
    let mut valence_occupations = Array1::zeros(40);
    occupations[0] = 2.0;
    valence_occupations[0] = 2.0;
    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![ConfigDatPotential {
            potential_index: 0,
            atomic_number: 4,
            element: "Be".to_string(),
            occupations,
            valence_occupations,
            spin_occupations: None,
        }],
    }
}

fn beryllium_single_potential_pot_bin() -> PotBinData {
    let potentials = 1;
    PotBinData {
        titles: vec!["ATOM source APOT Be smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 0,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.0,
            fermi_level: 0.0,
            interstitial_potential: 0.0,
            interstitial_density: 0.0,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![12]),
        muffin_tin_radii: Array1::from_vec(vec![1.1]),
        norman_indices: Array1::from_vec(vec![40]),
        atomic_numbers: Array1::from_vec(vec![4]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.1]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        electron_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_scf_mtdp_data(seed: &PotBinData) -> MtdpData {
    let mut atom_density = Array2::zeros((POT_BIN_RADIAL_POINTS, 1));
    let mut atom_potential = Array2::zeros((POT_BIN_RADIAL_POINTS, 1));
    for row in 0..POT_BIN_RADIAL_POINTS {
        atom_density[(row, 0)] = seed.electron_density[(row, 0)];
        atom_potential[(row, 0)] = seed.total_potential[(row, 0)];
    }
    atom_density[(0, 0)] += 1.0e-6;
    atom_density[(2, 0)] += 2.0e-6;
    atom_potential[(0, 0)] = -1.0;
    atom_potential[(1, 0)] = -1.1;
    atom_potential[(2, 0)] = -1.2;
    MtdpData {
        radial_count: POT_BIN_RADIAL_POINTS,
        atomic_numbers: Array1::from_vec(vec![4]),
        atom_coordinates: Array2::zeros((1, 3)),
        atom_radii: Array1::from_vec(vec![1.25]),
        atom_radius_indices: Array1::from_vec(vec![7]),
        atom_density,
        atom_potential,
        empty_sphere_coordinates: Array2::zeros((0, 3)),
        empty_sphere_radii: Array1::zeros(0),
        empty_sphere_radius_indices: Array1::zeros(0),
        empty_sphere_density: Array2::zeros((POT_BIN_RADIAL_POINTS, 0)),
        empty_sphere_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, 0)),
        interstitial_potential: -0.75,
        homo_energy: -0.12,
        lumo_energy: -0.08,
    }
}

fn sample_atomic_geom_dat() -> GeomDat {
    GeomDat {
        nat: 3,
        nph: 1,
        model_atoms: vec![1, 2],
        atoms: vec![
            GeomDatRow {
                index: 1,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: 0,
                boundary: 0,
            },
            GeomDatRow {
                index: 2,
                x: FEFF_BOHR_ANGSTROM,
                y: 0.0,
                z: 0.0,
                iph: 1,
                boundary: 1,
            },
            GeomDatRow {
                index: 3,
                x: 0.0,
                y: 2.0 * FEFF_BOHR_ANGSTROM,
                z: 0.0,
                iph: 1,
                boundary: 1,
            },
        ],
    }
}

fn set_nohole(work_dir: &Path, nohole: i32) -> Result<()> {
    let pot_path = work_dir.join("pot.inp");
    let pot_text = std::fs::read_to_string(&pot_path)?;
    let mut input = PotInput::parse_str(&pot_path, &pot_text)?;
    input.run.nohole = nohole;
    std::fs::write(&pot_path, refeff_io::pot_input_string(&input)?)?;
    Ok(())
}

fn sample_apot_bin() -> ApotBinData {
    ApotBinData {
        sections: vec![
            sample_core_hole_section(None).expect("zero core-hole sample should be renderable"),
            sample_atomic_density_section(),
        ],
    }
}

fn sample_transition_apot_bin() -> Result<ApotBinData> {
    let drho = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
        0.012 + 0.0002 * row as f64 + 0.001 * (-0.03 * row as f64).exp()
    });
    Ok(ApotBinData {
        sections: vec![
            sample_core_hole_section(Some(drho.view()))?,
            sample_atomic_density_section(),
        ],
    })
}

fn sample_pot_bin() -> PotBinData {
    let potentials = 2;
    PotBinData {
        titles: vec!["ATOM config handoff smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.0,
            fermi_level: 0.0,
            interstitial_potential: 0.0,
            interstitial_density: 0.0,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![12, 12]),
        muffin_tin_radii: Array1::from_vec(vec![1.1, 1.1]),
        norman_indices: Array1::from_vec(vec![40, 40]),
        atomic_numbers: Array1::from_vec(vec![29, 29]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.1, 2.1]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        electron_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_core_hole_section(drho: Option<ndarray::ArrayView1<'_, f64>>) -> Result<ApotBinSection> {
    let drho = drho
        .map(|values| values.to_owned())
        .unwrap_or_else(|| Array1::zeros(POT_BIN_RADIAL_POINTS));
    let dvcoul = apot_core_hole_coulomb_from_density(drho.view(), 2)?;
    Ok(ApotBinSection {
        section_number: 5,
        headers: vec![
            "dgc0   - upper component of core hole orbital".to_string(),
            "dpc0   - lower component of core hole orbital".to_string(),
            "drho   - core hole density.".to_string(),
            "dvcoul - core hole coulomb potential.".to_string(),
        ],
        header_texts: vec![
            " dgc0   - upper component of core hole orbital".to_string(),
            " dpc0   - lower component of core hole orbital".to_string(),
            " drho   - core hole density.".to_string(),
            " dvcoul - core hole coulomb potential.".to_string(),
        ],
        column_labels: vec![
            "dgc0".to_string(),
            "dpc0".to_string(),
            "drho".to_string(),
            "dvcoul".to_string(),
        ],
        column_label_text: Some(
            "            dgc0                 dpc0                 drho               dvcoul "
                .to_string(),
        ),
        payload: ApotBinPayload::Records(refeff_io::ApotBinRecords {
            column_types: vec![ApotBinType::Double; 4],
            rows: (0..POT_BIN_RADIAL_POINTS)
                .map(|row| {
                    vec![
                        ApotBinValue::Real(0.05 + 0.001 * row as f64),
                        ApotBinValue::Real(-0.005 - 0.0001 * row as f64),
                        ApotBinValue::Real(drho[row]),
                        ApotBinValue::Real(dvcoul[row]),
                    ]
                })
                .collect(),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    })
}

fn sample_atomic_density_section() -> ApotBinSection {
    ApotBinSection {
        section_number: 8,
        headers: vec!["rho(r,0:nphx+1) - atomic density for each unique potential".to_string()],
        header_texts: vec![
            " rho(r,0:nphx+1) - atomic density for each unique potential".to_string(),
        ],
        column_labels: vec![],
        column_label_text: None,
        payload: ApotBinPayload::Matrix(ApotBinMatrix {
            value_type: ApotBinType::Double,
            values: ApotBinMatrixValues::Real(Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, 2),
                |(row, potential)| 0.015 * (row + 1) as f64 + 0.25 * potential as f64,
            )),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    }
}

fn sample_config_dat() -> ConfigDatData {
    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![ConfigDatPotential {
            potential_index: 0,
            atomic_number: 29,
            element: "Cu".to_string(),
            occupations: Array1::from_shape_fn(40, |index| index as f64 * 0.1),
            valence_occupations: Array1::from_shape_fn(40, |index| index as f64 * 0.01),
            spin_occupations: None,
        }],
    }
}

fn write_generated_config_dat(work_dir: &Path) -> Result<()> {
    let input_path = work_dir.join("pot.inp");
    let input_text = std::fs::read_to_string(&input_path)?;
    let input = PotInput::parse_str(&input_path, &input_text)?;
    let data = super::generated_config_dat(&input, &work_dir.join("config.inp"))?;
    write_config_dat(work_dir.join("config.dat"), &data)?;
    Ok(())
}

fn sample_fpf0_dat() -> Fpf0DatData {
    Fpf0DatData {
        atomic_number: 29,
        total_energy_fprime: -0.125,
        relativistic_correction: 0.075,
        oscillators: vec![Fpf0Oscillator {
            oscillator_strength: 1.25,
            excitation_energy: -8.98,
            orbital_index: 1,
        }],
        form_factor_momentum: Array1::from_vec(vec![0.0, 0.5, 1.0]),
        form_factor: Array1::from_vec(vec![29.0, 28.1, 25.7]),
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating atomic potentials ...".to_string(),
            "Done with module: atomic potentials.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn optional_config_dat(path: impl AsRef<Path>) -> Result<Option<ConfigDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        return Ok(Some(read_config_dat(path)?));
    }
    Ok(None)
}

fn optional_fpf0_dat(path: impl AsRef<Path>) -> Result<Option<Fpf0DatData>> {
    let path = path.as_ref();
    if path.is_file() {
        return Ok(Some(read_fpf0_dat(path)?));
    }
    Ok(None)
}

fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
    let path = path.as_ref();
    if path.is_file() {
        return Ok(Some(read_module_log_dat(path)?));
    }
    Ok(None)
}

fn rendered_apot_bin(data: &ApotBinData) -> Result<ApotBinData> {
    Ok(parse_apot_bin(&apot_bin_string(data)?)?)
}

fn assert_log_contains(log: &ModuleLogData, expected: &str) {
    assert!(
        log.lines.iter().any(|line| line.contains(expected)),
        "expected log to contain {expected:?}, got {:?}",
        log.lines
    );
}

fn log_line_occurrences(log: &ModuleLogData, expected: &str) -> usize {
    log.lines.iter().filter(|line| line == &expected).count()
}

fn assert_fpf0_close(actual: &Fpf0DatData, expected: &Fpf0DatData) {
    assert_eq!(actual.atomic_number, expected.atomic_number);
    assert_close(
        actual.total_energy_fprime,
        expected.total_energy_fprime,
        1.0e-6,
        "total_energy_fprime",
    );
    assert_close(
        actual.relativistic_correction,
        expected.relativistic_correction,
        1.0e-6,
        "relativistic_correction",
    );
    assert_eq!(actual.oscillators, expected.oscillators);
    assert_eq!(actual.form_factor_momentum, expected.form_factor_momentum);
    assert_eq!(actual.form_factor_count(), expected.form_factor_count());
    for (index, (&actual, &expected)) in actual
        .form_factor
        .iter()
        .zip(expected.form_factor.iter())
        .enumerate()
    {
        assert_close(actual, expected, 1.5e-4, &format!("form_factor[{index}]"));
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "{label} differs by {difference:e}: actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

fn reference_atomic_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("reference-work/golden/EXAFS/Cu"));
    Ok(reference.filter(|path| path.join("pot.inp").is_file() && path.join("apot.bin").is_file()))
}

fn reference_atomic_transition_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("reference-work/golden/XANES/Cu"));
    Ok(reference.filter(|path| path.join("pot.inp").is_file() && path.join("apot.bin").is_file()))
}

fn reference_atomic_nohole_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("reference-work/golden/NRIXS/GeCl_4"));
    Ok(reference.filter(|path| {
        path.join("pot.inp").is_file()
            && path.join("apot.bin").is_file()
            && path.join("config.dat").is_file()
    }))
}

fn reference_bn_true_scf_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("reference-work/golden/XANES/BN"));
    Ok(reference.filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
}

fn reference_exafs_ybco_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("reference-work/golden/EXAFS/YBCO"));
    Ok(reference.filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
}

fn reference_highz_report() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("reference-work/golden/HIGHZ/HighZ.out"))
        .filter(|path| path.is_file())
}
