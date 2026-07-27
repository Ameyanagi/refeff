use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{
    CrpaDatData, CrpaInput, CrpaResponseAssemblyHandoff, CrpaResponseAssemblyHandoffInput,
    ModuleLogData, ScreenInput, crpa_dat_string, crpa_response_assembly_handoff, read_crpa_dat,
    read_module_log_dat, read_wscrn_dat, write_crpa_dat, write_module_log_dat, write_wscrn_dat,
    wscrn_dat_string,
};

use crate::{screen, work_dir_for_input};

const CRPA_SOURCE_REQUIREMENT_ERROR: &str =
    "CRPA generation requires cached crpa.dat or complete SCREEN/FMS source handoffs";

/// Run the supported FEFF CRPA cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_cached_crpa_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_wscrn_handoff(work_dir)? {
        return run_supported_wscrn_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Whether a FEFF CRPA run can be satisfied from an existing `crpa.dat` cache.
pub(crate) fn has_cached_crpa_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("crpa.inp").is_file() || !work_dir.join("crpa.dat").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    Ok(input.enabled && can_use_cached_crpa_output_for_discovery(work_dir, &input))
}

/// Whether FEFF CRPA can recover its optional screened-potential sidecar from
/// source handoffs when full CRPA output is not yet available.
pub(crate) fn has_supported_wscrn_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("crpa.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.enabled && can_use_cached_crpa_output_for_discovery(work_dir, &input) {
        return Ok(false);
    }
    if work_dir.join("screen.inp").is_file() && read_screen_input(work_dir).is_err() {
        return Ok(false);
    }
    Ok(input.enabled && screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir))
}

/// Whether FEFF CRPA can generate `crpa.dat` and `wscrn.dat` from complete
/// SCREEN/FMS source handoffs.
pub(crate) fn has_supported_crpa_source_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("crpa.inp").is_file() || !work_dir.join("screen.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !input.enabled || can_use_cached_crpa_output_for_discovery(work_dir, &input) {
        return Ok(false);
    }
    let screen_input = match read_screen_input(work_dir) {
        Ok(input) => input,
        Err(_) => return Ok(false),
    };
    match screen::build_source_screen_response_components(work_dir, &screen_input) {
        Ok(components) => Ok(components.is_some()),
        Err(_) => Ok(false),
    }
}

/// Recover only the Rust-backed CRPA `wscrn.dat` sidecar.
///
/// This intentionally does not report a full CRPA stage without `crpa.dat`.
pub(crate) fn run_supported_wscrn_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    if !work_dir.join("crpa.inp").is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if input.enabled && can_use_cached_crpa_output(work_dir, &input)? {
        return Ok(0);
    }
    if !input.enabled || !screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir) {
        return Ok(0);
    }

    screen::recover_wscrn_from_vtot_and_apot_in_dir(work_dir)
        .context("failed to recover CRPA wscrn.dat sidecar")
}

/// Run the FEFF CRPA path from cached outputs or complete source handoffs.
///
/// Cached FEFF output passes through the typed Rust renderer for module-level
/// orchestration tests. When complete SCREEN/FMS source handoffs are present,
/// Rust assembles paired `crpa.dat` and CRPA-owned `wscrn.dat` outputs before
/// rendering sidecars and the deterministic `logscrn.dat` wrapper. Missing or
/// unreadable optional `wscrn.dat` sidecars can still be regenerated from valid
/// `vtot.dat`/`apot.bin` handoffs when a full CRPA source bundle is absent.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.enabled {
        return Ok(0);
    }

    if !can_use_cached_crpa_output(work_dir, &input)? {
        write_source_crpa_outputs(work_dir, &input)
            .context("failed to generate CRPA outputs from source handoffs")?;
    }

    let output_path = work_dir.join("crpa.dat");
    if !output_path.is_file() {
        recover_optional_wscrn_from_vtot_and_apot(work_dir)?;
        bail!(CRPA_SOURCE_REQUIREMENT_ERROR);
    }

    let data = match read_crpa_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))
    {
        Ok(data) => data,
        Err(error) => {
            if validate_or_recover_wscrn_handoff(work_dir)? {
                bail!(CRPA_SOURCE_REQUIREMENT_ERROR);
            }
            return Err(error);
        }
    };
    write_cached_output(&output_path, &data)?;
    let wscrn_source_handoff = recover_optional_wscrn_from_vtot_and_apot(work_dir)?;
    Ok(1 + write_optional_wscrn_cache(&work_dir.join("wscrn.dat"))?
        + write_or_recover_module_log(&work_dir.join("logscrn.dat"), wscrn_source_handoff)?)
}

fn read_input(work_dir: &Path) -> Result<CrpaInput> {
    let input_path = work_dir.join("crpa.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    CrpaInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn read_screen_input(work_dir: &Path) -> Result<ScreenInput> {
    let input_path = work_dir.join("screen.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    ScreenInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &CrpaDatData) -> Result<()> {
    write_crpa_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_source_crpa_outputs(work_dir: &Path, input: &CrpaInput) -> Result<usize> {
    let Some(handoff) = generated_source_crpa_handoff(work_dir, input)? else {
        return Ok(0);
    };

    write_crpa_dat(work_dir.join("crpa.dat"), &handoff.outputs.crpa.crpa)
        .with_context(|| format!("failed to write {}", work_dir.join("crpa.dat").display()))?;
    write_wscrn_dat(work_dir.join("wscrn.dat"), &handoff.outputs.wscrn)
        .with_context(|| format!("failed to write {}", work_dir.join("wscrn.dat").display()))?;
    Ok(1 + handoff.outputs.wscrn.row_count())
}

fn generated_source_crpa_handoff(
    work_dir: &Path,
    input: &CrpaInput,
) -> Result<Option<CrpaResponseAssemblyHandoff>> {
    if !work_dir.join("screen.inp").is_file() {
        return Ok(None);
    }

    let screen_input = read_screen_input(work_dir)?;
    let Some(components) =
        screen::build_source_screen_response_components(work_dir, &screen_input)?
    else {
        return Ok(None);
    };

    Ok(Some(
        crpa_response_assembly_handoff(CrpaResponseAssemblyHandoffInput {
            crpa: input,
            potential: &components.potential,
            fms: &components.fms,
            reference_energies_hartree: components.radial.reference_energies_hartree.view(),
            fermi_level_hartree: components.fermi_level_hartree,
            regular_solutions: components
                .radial
                .matched
                .solved
                .radial_cubes
                .regular_large
                .view(),
            irregular_solutions: components
                .radial
                .matched
                .solved
                .radial_cubes
                .irregular_large
                .view(),
            crpa_header_lines: &[],
            wscrn_header_lines: &[],
        })
        .context("failed to assemble CRPA response from source handoffs")?,
    ))
}

fn can_use_cached_crpa_output(work_dir: &Path, input: &CrpaInput) -> Result<bool> {
    if !crpa_cache_is_usable(work_dir) {
        return Ok(false);
    }

    if cached_crpa_output_is_stale_against_source(work_dir, input)? {
        return Ok(false);
    }

    let wscrn_source_handoff = screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir);
    let wscrn_path = work_dir.join("wscrn.dat");
    if wscrn_path.is_file() && read_wscrn_dat(&wscrn_path).is_err() && !wscrn_source_handoff {
        return Ok(false);
    }

    if wscrn_source_handoff {
        return Ok(true);
    }

    Ok(prepare_module_log_cache(&work_dir.join("logscrn.dat")).is_ok())
}

fn can_use_cached_crpa_output_for_discovery(work_dir: &Path, input: &CrpaInput) -> bool {
    can_use_cached_crpa_output(work_dir, input).unwrap_or(false)
}

fn crpa_cache_is_usable(work_dir: &Path) -> bool {
    let output_path = work_dir.join("crpa.dat");
    output_path.is_file() && read_crpa_dat(&output_path).is_ok()
}

fn cached_crpa_output_is_stale_against_source(work_dir: &Path, input: &CrpaInput) -> Result<bool> {
    let Some(handoff) = generated_source_crpa_handoff(work_dir, input)? else {
        return Ok(false);
    };

    let crpa_path = work_dir.join("crpa.dat");
    let Ok(cached_crpa) = read_crpa_dat(&crpa_path) else {
        return Ok(false);
    };
    if crpa_dat_string(&cached_crpa)? != crpa_dat_string(&handoff.outputs.crpa.crpa)? {
        return Ok(true);
    }

    let wscrn_path = work_dir.join("wscrn.dat");
    let Ok(cached_wscrn) = read_wscrn_dat(&wscrn_path) else {
        return Ok(true);
    };
    Ok(wscrn_dat_string(&cached_wscrn)? != wscrn_dat_string(&handoff.outputs.wscrn)?)
}

fn validate_or_recover_wscrn_handoff(work_dir: &Path) -> Result<bool> {
    if screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir) {
        return recover_optional_wscrn_from_vtot_and_apot(work_dir);
    }
    Ok(work_dir.join("vtot.dat").is_file()
        && work_dir.join("apot.bin").is_file()
        && screen::has_usable_wscrn_handoff_in_dir(work_dir))
}

fn recover_optional_wscrn_from_vtot_and_apot(work_dir: &Path) -> Result<bool> {
    if !screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir) {
        return Ok(false);
    }
    let row_count = screen::recover_wscrn_from_vtot_and_apot_in_dir(work_dir)
        .context("failed to recover CRPA wscrn.dat sidecar")?;
    Ok(row_count > 0)
}

fn write_optional_wscrn_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_wscrn_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    let row_count = data.row_count();
    write_wscrn_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(row_count)
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)?;
    Ok(1)
}

fn prepare_module_log_cache(path: &Path) -> Result<()> {
    if path.is_file() {
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(())
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_generate_module_log(path: &Path) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_module_log(path, &generated_crpa_module_log())?;
    Ok(1)
}

fn write_or_recover_module_log(path: &Path, source_handoff_written: bool) -> Result<usize> {
    if source_handoff_written && path.is_file() && read_module_log_dat(path).is_err() {
        write_module_log(path, &generated_crpa_module_log())?;
        return Ok(1);
    }
    write_or_generate_module_log(path)
}

fn generated_crpa_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            " Calculating Hubbard U.".to_string(),
            " Done with Hubbard U calculation.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(); 2],
    }
}

#[cfg(test)]
mod tests {
    use super::{
        has_cached_crpa_output, has_supported_crpa_source_handoff, has_supported_wscrn_handoff,
        run_in_dir, run_supported_wscrn_handoff_in_dir,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, ArrayView1, array};
    use refeff_core::screen::{screen_bare_core_hole_potential, screen_radial_grid};
    use refeff_io::pot_bin::POT_BIN_RADIAL_POINTS;
    use refeff_io::{
        ApotBinData, ApotBinPayload, ApotBinSection, ApotBinType, ApotBinValue, CrpaDatData,
        ModuleLogData, VtotDatData, parse_crpa_dat, parse_module_log_dat, read_crpa_dat,
        read_module_log_dat, read_wscrn_dat, write_apot_bin, write_crpa_dat, write_module_log_dat,
        write_vtot_dat, write_wscrn_dat,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn crpa_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("crpa.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled CRPA should require cached output or source handoffs")?;

        assert!(error.to_string().contains(
            "CRPA generation requires cached crpa.dat or complete SCREEN/FMS source handoffs"
        ));
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_dat(temp.path().join("crpa.dat"), &sample_crpa_dat())?;

        assert!(!has_cached_crpa_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_advertise_malformed_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        std::fs::write(temp.path().join("crpa.dat"), "not a crpa.dat table\n")?;

        assert!(!has_cached_crpa_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled CRPA should reject malformed crpa.dat")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("crpa.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let expected = sample_crpa_dat();
        std::fs::write(temp.path().join("crpa.inp"), b"not a crpa.inp handoff\n")?;
        std::fs::write(temp.path().join("screen.inp"), "not a screen.inp handoff\n")?;
        write_crpa_dat(temp.path().join("crpa.dat"), &expected)?;

        assert!(!has_cached_crpa_output(temp.path())?);
        assert!(!has_supported_wscrn_handoff(temp.path())?);
        assert!(!has_supported_crpa_source_handoff(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed CRPA input should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("crpa.inp"), "{chain}");
        assert_eq!(read_crpa_dat(temp.path().join("crpa.dat"))?, expected);
        assert!(!temp.path().join("logscrn.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_claim_malformed_screen_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        std::fs::write(temp.path().join("screen.inp"), "not a screen.inp handoff\n")?;

        assert!(!has_supported_crpa_source_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed CRPA screen.inp source should fail through explicit CRPA")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("screen.inp"), "{chain}");
        assert!(!temp.path().join("crpa.dat").exists());
        assert!(!temp.path().join("wscrn.dat").exists());
        assert!(!temp.path().join("logscrn.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_claim_cached_output_with_malformed_screen_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        let expected = sample_crpa_dat();
        write_crpa_input(temp.path(), true)?;
        write_crpa_dat(temp.path().join("crpa.dat"), &expected)?;
        std::fs::write(temp.path().join("screen.inp"), "not a screen.inp handoff\n")?;

        assert!(!has_cached_crpa_output(temp.path())?);
        assert!(!has_supported_wscrn_handoff(temp.path())?);
        assert!(!has_supported_crpa_source_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed CRPA screen source should fail through explicit CRPA")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("screen.inp"), "{chain}");
        assert_eq!(read_crpa_dat(temp.path().join("crpa.dat"))?, expected);
        assert!(!temp.path().join("wscrn.dat").exists());
        assert!(!temp.path().join("logscrn.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_keeps_malformed_crpa_cache_strict_with_standalone_wscrn() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        std::fs::write(temp.path().join("crpa.dat"), "not a crpa.dat table\n")?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;

        assert!(!has_cached_crpa_output(temp.path())?);
        assert!(!has_supported_wscrn_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("standalone wscrn.dat should not hide malformed crpa.dat")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("crpa.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_advertise_malformed_wscrn_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        write_crpa_dat(temp.path().join("crpa.dat"), &sample_crpa_dat())?;
        std::fs::write(temp.path().join("wscrn.dat"), "not a wscrn.dat table\n")?;

        assert!(!has_cached_crpa_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled CRPA should reject malformed wscrn.dat sidecar")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("wscrn.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn crpa_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        write_crpa_dat(temp.path().join("crpa.dat"), &sample_crpa_dat())?;
        std::fs::write(temp.path().join("logscrn.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_crpa_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled CRPA should reject malformed logscrn.dat")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logscrn.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn crpa_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let expected = CrpaDatData {
            header_lines: vec!["U, n, U_Bare".to_string()],
            hubbard_u: 0.197_879_035_252_010,
            occupation: 1.0,
            bare_u: 0.694_283_422_651_496,
        };
        write_crpa_dat(temp.path().join("crpa.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_crpa_dat(temp.path().join("crpa.dat"))?, expected);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn crpa_module_generates_missing_module_log_from_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        write_crpa_dat(temp.path().join("crpa.dat"), &sample_crpa_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn crpa_module_roundtrips_cached_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let expected_output = CrpaDatData {
            header_lines: vec!["U, n, U_Bare".to_string()],
            hubbard_u: 0.197_879_035_252_010,
            occupation: 1.0,
            bare_u: 0.694_283_422_651_496,
        };
        let expected_log = sample_module_log();
        write_crpa_dat(temp.path().join("crpa.dat"), &expected_output)?;
        write_module_log_dat(temp.path().join("logscrn.dat"), &expected_log)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_crpa_dat(temp.path().join("crpa.dat"))?,
            expected_output
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn crpa_module_preserves_cached_wscrn_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let expected_crpa = sample_crpa_dat();
        write_crpa_dat(temp.path().join("crpa.dat"), &expected_crpa)?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;
        let expected_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1 + expected_wscrn.row_count() + 1);
        assert_eq!(read_crpa_dat(temp.path().join("crpa.dat"))?, expected_crpa);
        assert_wscrn_close(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn,
            1.0e-12,
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn crpa_module_recovers_malformed_wscrn_from_vtot_and_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let expected_crpa = sample_crpa_dat();
        let vtot = sample_vtot_dat();
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, vtot.row_count())?;
        write_crpa_dat(temp.path().join("crpa.dat"), &expected_crpa)?;
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;
        std::fs::write(temp.path().join("wscrn.dat"), "not a wscrn.dat table\n")?;
        std::fs::write(temp.path().join("logscrn.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_crpa_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1 + vtot.row_count() + 1);
        assert_eq!(read_crpa_dat(temp.path().join("crpa.dat"))?, expected_crpa);
        let recovered = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_array_close(
            recovered.radius_bohr.view(),
            vtot.radius_bohr.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.screened_potential.view(),
            vtot.screened_core_hole_potential.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.core_hole_potential.view(),
            expected_core_hole.view(),
            1.0e-9,
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn crpa_module_generates_supported_wscrn_handoff_without_crpa_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let vtot = sample_vtot_dat();
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, vtot.row_count())?;
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_supported_wscrn_handoff_in_dir(temp.path())?;

        assert_eq!(count, vtot.row_count());
        let recovered = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_array_close(
            recovered.radius_bohr.view(),
            vtot.radius_bohr.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.screened_potential.view(),
            vtot.screened_core_hole_potential.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.core_hole_potential.view(),
            expected_core_hole.view(),
            1.0e-9,
        );
        assert!(!temp.path().join("crpa.dat").exists());
        assert!(!temp.path().join("logscrn.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_recovers_wscrn_handoff_before_missing_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let vtot = sample_vtot_dat();
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, vtot.row_count())?;
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("missing final CRPA output should still require source handoffs")?;

        assert!(
            error.to_string().contains(
                "CRPA generation requires cached crpa.dat or complete SCREEN/FMS source handoffs"
            ),
            "{error:?}"
        );
        let recovered = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_array_close(
            recovered.radius_bohr.view(),
            vtot.radius_bohr.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.screened_potential.view(),
            vtot.screened_core_hole_potential.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.core_hole_potential.view(),
            expected_core_hole.view(),
            1.0e-9,
        );
        assert!(!temp.path().join("crpa.dat").exists());
        assert!(!temp.path().join("logscrn.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_recovers_wscrn_handoff_when_malformed_crpa_cache_exists() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let vtot = sample_vtot_dat();
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, vtot.row_count())?;
        std::fs::write(temp.path().join("crpa.dat"), "not a crpa.dat table\n")?;
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        assert!(!has_cached_crpa_output(temp.path())?);
        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_supported_wscrn_handoff_in_dir(temp.path())?;

        assert_eq!(count, vtot.row_count());
        let recovered = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_array_close(
            recovered.radius_bohr.view(),
            vtot.radius_bohr.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.screened_potential.view(),
            vtot.screened_core_hole_potential.view(),
            1.0e-12,
        );
        assert_array_close(
            recovered.core_hole_potential.view(),
            expected_core_hole.view(),
            1.0e-9,
        );
        assert!(!temp.path().join("logscrn.dat").exists());

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed final CRPA cache should require source handoffs")?;
        assert!(
            error.to_string().contains(
                "CRPA generation requires cached crpa.dat or complete SCREEN/FMS source handoffs"
            ),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn crpa_module_roundtrips_reference_zip_when_present() -> Result<()> {
        let Some(zip_path) = reference_crpa_zip()? else {
            crate::require_fixture!("CRPA reference test; CRPA REFERENCE.zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("CRPA reference test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("crpa.inp"),
            unzip_reference_entry(&zip_path, "REFERENCE/crpa.inp")?,
        )?;
        std::fs::write(
            temp.path().join("crpa.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/crpa.dat")?,
        )?;
        std::fs::write(
            temp.path().join("wscrn.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/wscrn.dat")?,
        )?;
        std::fs::write(
            temp.path().join("logscrn.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/logscrn.dat")?,
        )?;
        let expected_output = parse_crpa_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/crpa.dat",
        )?)?)?;
        let expected_log = parse_module_log_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/logscrn.dat",
        )?)?)?;
        let expected_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2 + expected_wscrn.row_count());
        assert_eq!(
            read_crpa_dat(temp.path().join("crpa.dat"))?,
            expected_output
        );
        assert_wscrn_close(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn,
            1.0e-9,
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn crpa_module_generates_reference_zip_from_source_without_phase_or_gg_cache() -> Result<()> {
        let Some(zip_path) = reference_crpa_zip()? else {
            crate::require_fixture!("CRPA source reference test; CRPA REFERENCE.zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("CRPA source reference test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        for entry in ["crpa.inp", "pot.bin", "config.dat", "geom.dat", "fms.inp"] {
            std::fs::write(
                temp.path().join(entry),
                unzip_reference_entry(&zip_path, &format!("REFERENCE/{entry}"))?,
            )?;
        }
        let mut screen_input = unzip_reference_entry(&zip_path, "REFERENCE/screen.inp")?;
        if !String::from_utf8_lossy(&screen_input).contains("icore") {
            screen_input.extend_from_slice(b" icore          -1\n");
        }
        std::fs::write(temp.path().join("screen.inp"), screen_input)?;

        assert!(has_supported_crpa_source_handoff(temp.path())?);

        let expected_crpa = parse_crpa_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/crpa.dat",
        )?)?)?;
        let expected = tempfile::tempdir()?;
        std::fs::write(
            expected.path().join("wscrn.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/wscrn.dat")?,
        )?;
        let expected_wscrn = read_wscrn_dat(expected.path().join("wscrn.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2 + expected_wscrn.row_count());
        assert!(!temp.path().join("phase.bin").is_file());
        assert!(!temp.path().join("gg.bin").is_file());
        assert_crpa_close(
            read_crpa_dat(temp.path().join("crpa.dat"))?,
            expected_crpa,
            1.0e-5,
        );
        assert_wscrn_screened_close(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn,
            1.0e-5,
        );
        assert!(read_module_log_dat(temp.path().join("logscrn.dat")).is_ok());
        Ok(())
    }

    #[test]
    fn crpa_module_regenerates_stale_cache_from_source_reference_handoffs() -> Result<()> {
        let Some(zip_path) = reference_crpa_zip()? else {
            crate::require_fixture!(
                "CRPA stale source reference test; CRPA REFERENCE.zip not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("CRPA stale source reference test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        for entry in ["crpa.inp", "pot.bin", "config.dat", "geom.dat", "fms.inp"] {
            std::fs::write(
                temp.path().join(entry),
                unzip_reference_entry(&zip_path, &format!("REFERENCE/{entry}"))?,
            )?;
        }
        let mut screen_input = unzip_reference_entry(&zip_path, "REFERENCE/screen.inp")?;
        if !String::from_utf8_lossy(&screen_input).contains("icore") {
            screen_input.extend_from_slice(b" icore          -1\n");
        }
        std::fs::write(temp.path().join("screen.inp"), screen_input)?;

        let expected_crpa = parse_crpa_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/crpa.dat",
        )?)?)?;
        let expected = tempfile::tempdir()?;
        std::fs::write(
            expected.path().join("wscrn.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/wscrn.dat")?,
        )?;
        let expected_wscrn = read_wscrn_dat(expected.path().join("wscrn.dat"))?;
        let mut stale_crpa = expected_crpa.clone();
        stale_crpa.hubbard_u += 0.25;
        write_crpa_dat(temp.path().join("crpa.dat"), &stale_crpa)?;
        let mut stale_wscrn = expected_wscrn.clone();
        stale_wscrn.screened_potential[0] += 0.25;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &stale_wscrn)?;

        assert!(!has_cached_crpa_output(temp.path())?);
        assert!(has_supported_crpa_source_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2 + expected_wscrn.row_count());
        assert_crpa_close(
            read_crpa_dat(temp.path().join("crpa.dat"))?,
            expected_crpa,
            1.0e-5,
        );
        assert_wscrn_screened_close(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn,
            1.0e-5,
        );
        assert!(read_module_log_dat(temp.path().join("logscrn.dat")).is_ok());
        Ok(())
    }

    fn write_crpa_input(work_dir: &Path, enabled: bool) -> Result<()> {
        std::fs::write(
            work_dir.join("crpa.inp"),
            format!(
                concat!(" do_CRPA{:12}\n", " rcut{:21.16}     \n", " l_crpa{:12}\n",),
                i32::from(enabled),
                3.5,
                2
            ),
        )?;
        Ok(())
    }

    fn sample_crpa_dat() -> CrpaDatData {
        CrpaDatData {
            header_lines: vec!["U, n, U_Bare".to_string()],
            hubbard_u: 0.197_879_035_252_010,
            occupation: 1.0,
            bare_u: 0.694_283_422_651_496,
        }
    }

    fn sample_wscrn_dat() -> refeff_io::WscrnDatData {
        refeff_io::WscrnDatData {
            header_lines: vec!["# r       w_scrn(r)      v_ch(r)".to_string()],
            radius_bohr: array![
                0.000_150_733_075_095,
                0.000_158_462_349_092,
                0.000_166_587_928_075
            ],
            screened_potential: array![0.103_467_493_981, 0.108_315_930_699, 0.113_339_518_784,],
            core_hole_potential: array![0.995_451_023_116, 0.993_984_758_512, 0.992_447_997_456,],
        }
    }

    fn sample_vtot_dat() -> VtotDatData {
        VtotDatData {
            header_lines: Vec::new(),
            radius_bohr: array![
                0.000_150_733_075_095,
                0.000_158_462_349_092,
                0.000_166_587_928_075
            ],
            total_potential: array![-182_900.15, -182_900.13, -182_900.10],
            screened_core_hole_potential: array![26.728_823_46, 26.728_816_78, 26.728_803_06],
        }
    }

    fn sample_core_hole_components() -> (Array1<f64>, Array1<f64>) {
        let large_component = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            if row < 3 {
                0.45 + 0.05 * row as f64
            } else {
                0.0
            }
        });
        let small_component = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            if row < 3 {
                -0.06 - 0.01 * row as f64
            } else {
                0.0
            }
        });
        (large_component, small_component)
    }

    fn expected_core_hole_potential(
        large_component: &Array1<f64>,
        small_component: &Array1<f64>,
        row_count: usize,
    ) -> Result<Array1<f64>> {
        let radii = screen_radial_grid(0.05, 8.8, row_count)?;
        Ok(screen_bare_core_hole_potential(
            radii
                .as_slice()
                .context("radial grid storage is not contiguous")?,
            large_component
                .as_slice()
                .context("large component storage is not contiguous")?,
            small_component
                .as_slice()
                .context("small component storage is not contiguous")?,
            0.05,
            row_count,
        )?)
    }

    fn sample_apot_bin(
        large_component: &Array1<f64>,
        small_component: &Array1<f64>,
    ) -> ApotBinData {
        ApotBinData {
            sections: vec![ApotBinSection {
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
                                ApotBinValue::Real(large_component[row]),
                                ApotBinValue::Real(small_component[row]),
                                ApotBinValue::Real(0.0),
                                ApotBinValue::Real(0.0),
                            ]
                        })
                        .collect(),
                }),
                trailing_headers: vec![],
                trailing_header_texts: vec![],
            }],
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                " Calculating Hubbard U.".to_string(),
                " Done with Hubbard U calculation.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn assert_wscrn_close(
        actual: refeff_io::WscrnDatData,
        expected: refeff_io::WscrnDatData,
        tolerance: f64,
    ) {
        assert_eq!(actual.row_count(), expected.row_count());
        assert_array_close(
            actual.radius_bohr.view(),
            expected.radius_bohr.view(),
            tolerance,
        );
        assert_array_close(
            actual.screened_potential.view(),
            expected.screened_potential.view(),
            tolerance,
        );
        assert_array_close(
            actual.core_hole_potential.view(),
            expected.core_hole_potential.view(),
            tolerance,
        );
    }

    fn assert_wscrn_screened_close(
        actual: refeff_io::WscrnDatData,
        expected: refeff_io::WscrnDatData,
        tolerance: f64,
    ) {
        assert_eq!(actual.row_count(), expected.row_count());
        assert_array_close(
            actual.radius_bohr.view(),
            expected.radius_bohr.view(),
            tolerance,
        );
        assert_array_close(
            actual.screened_potential.view(),
            expected.screened_potential.view(),
            tolerance,
        );
    }

    fn assert_crpa_close(actual: CrpaDatData, expected: CrpaDatData, tolerance: f64) {
        for (label, actual, expected) in [
            ("Hubbard U", actual.hubbard_u, expected.hubbard_u),
            ("occupation", actual.occupation, expected.occupation),
            ("bare U", actual.bare_u, expected.bare_u),
        ] {
            let scaled_tolerance = tolerance * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= scaled_tolerance,
                "{label}: actual={actual}, expected={expected}, diff={}, tolerance={scaled_tolerance}",
                (actual - expected).abs()
            );
        }
    }

    fn assert_array_close(
        actual: ArrayView1<'_, f64>,
        expected: ArrayView1<'_, f64>,
        tolerance: f64,
    ) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            let scaled_tolerance = tolerance * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= scaled_tolerance,
                "actual={actual}, expected={expected}, diff={}, tolerance={scaled_tolerance}",
                (actual - expected).abs()
            );
        }
    }

    fn reference_crpa_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/CRPA/REFERENCE.zip");
        Ok(path.is_file().then_some(path))
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to extract {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }
}
