use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_core::{PathfinderReductionInput, pathfinder_reduction};
use refeff_io::{
    EelsInput, GeomDat, GlobalInput, ModuleLogData, PathsDatData, PathsInput,
    PathsdPathsDatDataInput, pathsd_paths_dat_data, read_module_log_dat, read_paths_dat,
    read_phase_bin, write_module_log_dat, write_paths_dat,
};

use crate::work_dir_for_input;

const FEFF_PATH_MAX_PATH_ATOMS: usize = 8;
const FEFF_PATH_MAX_OUTPUT_PATHS: usize = 5_000_001;

/// Run the supported FEFF PATH output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF PATH run can be satisfied from `paths.dat` or source handoffs.
pub(crate) fn has_cached_paths_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("paths.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !path_enabled(&input) {
        return Ok(false);
    }
    let output_path = work_dir.join("paths.dat");
    if output_path.is_file() {
        if read_paths_dat(&output_path).is_ok() {
            if validate_declared_path_source_handoffs(work_dir, &input).is_err() {
                return Ok(false);
            }
            return Ok(true);
        }
        return Ok(path_zero_output_branch(&input)
            || (path_generation_handoffs_present(work_dir)
                && generate_paths_dat_from_handoffs(work_dir, &input).is_ok()));
    }
    if path_zero_output_branch(&input) {
        return Ok(true);
    }
    if !path_generation_handoffs_present(work_dir) {
        return Ok(false);
    }
    Ok(generate_paths_dat_from_handoffs(work_dir, &input).is_ok())
}

/// Run FEFF PATH from source handoffs or an existing `paths.dat`.
///
/// When `geom.dat`, `phase.bin`, and `global.inp` are present, this composes the
/// ported `prcrit`, `paths`, and `pathsd` Rust kernels to generate `paths.dat`.
/// Existing `paths.dat` caches are still validated and re-rendered. When the
/// zero-output branch or complete PATH source handoffs can regenerate the
/// result, readable stale caches are replaced from source. FEFF's trivial
/// `rmax < 1.0` branch writes a zero-path header.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !path_enabled(&input) {
        return Ok(0);
    }

    let output_path = work_dir.join("paths.dat");
    let data = if output_path.is_file() {
        match read_paths_dat(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))
        {
            Ok(data) => {
                generate_paths_if_stale_against_source(work_dir, &input, &data)?.unwrap_or(data)
            }
            Err(error) => recover_malformed_paths_dat(work_dir, &input, error)?,
        }
    } else if path_zero_output_branch(&input) {
        empty_paths_dat_for_input(&input)?
    } else if path_generation_handoffs_present(work_dir) {
        generate_paths_dat_from_handoffs(work_dir, &input)?
    } else {
        bail!(
            "PATH pathfinder generation requires cached paths.dat output or phase.bin/geom.dat/global.inp source handoffs"
        );
    };
    let path_count = data.paths.len();
    write_cached_output(&output_path, &data)?;
    write_or_generate_module_log(&work_dir.join("log4.dat"), &input, &data)?;
    Ok(path_count)
}

fn generate_paths_if_stale_against_source(
    work_dir: &Path,
    input: &PathsInput,
    cached: &PathsDatData,
) -> Result<Option<PathsDatData>> {
    let generated = if path_zero_output_branch(input) {
        empty_paths_dat_for_input(input)?
    } else if path_generation_handoffs_present(work_dir) {
        generate_paths_dat_from_handoffs(work_dir, input)?
    } else {
        validate_present_path_source_handoffs(work_dir)?;
        return Ok(None);
    };
    if paths_dat_matches_source(cached, &generated) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn recover_malformed_paths_dat(
    work_dir: &Path,
    input: &PathsInput,
    cache_error: anyhow::Error,
) -> Result<PathsDatData> {
    if path_zero_output_branch(input) {
        return empty_paths_dat_for_input(input);
    }
    if path_generation_handoffs_present(work_dir) {
        return generate_paths_dat_from_handoffs(work_dir, input);
    }
    Err(cache_error)
}

fn paths_dat_matches_source(cached: &PathsDatData, source: &PathsDatData) -> bool {
    cached.titles == source.titles
        && cached.paths.len() == source.paths.len()
        && cached
            .paths
            .iter()
            .zip(source.paths.iter())
            .all(|(cached, source)| paths_dat_path_matches_source(cached, source))
}

fn paths_dat_path_matches_source(
    cached: &refeff_io::PathsDatPath,
    source: &refeff_io::PathsDatPath,
) -> bool {
    const PATH_HEADER_TOLERANCE: f64 = 5.0e-4;

    cached.index == source.index
        && cached.row_header == source.row_header
        && cached.atoms.len() == source.atoms.len()
        && scalar_matches(cached.degeneracy, source.degeneracy, PATH_HEADER_TOLERANCE)
        && scalar_matches(
            cached.effective_half_path_length_angstrom,
            source.effective_half_path_length_angstrom,
            PATH_HEADER_TOLERANCE,
        )
        && cached
            .atoms
            .iter()
            .zip(source.atoms.iter())
            .all(|(cached, source)| paths_dat_atom_matches_source(cached, source))
}

fn paths_dat_atom_matches_source(
    cached: &refeff_io::PathsDatAtom,
    source: &refeff_io::PathsDatAtom,
) -> bool {
    const POSITION_TOLERANCE: f64 = 5.0e-6;
    const LEG_TOLERANCE: f64 = 5.0e-4;

    cached.potential_index == source.potential_index
        && cached.label == source.label
        && cached
            .position_angstrom
            .iter()
            .zip(source.position_angstrom.iter())
            .all(|(cached, source)| scalar_matches(*cached, *source, POSITION_TOLERANCE))
        && optional_scalar_matches(
            cached.leg_distance_angstrom,
            source.leg_distance_angstrom,
            LEG_TOLERANCE,
        )
        && optional_scalar_matches(cached.beta_degrees, source.beta_degrees, LEG_TOLERANCE)
        && optional_scalar_matches(cached.eta_degrees, source.eta_degrees, LEG_TOLERANCE)
}

fn optional_scalar_matches(
    cached: Option<f64>,
    source: Option<f64>,
    absolute_tolerance: f64,
) -> bool {
    match (cached, source) {
        (Some(cached), Some(source)) => scalar_matches(cached, source, absolute_tolerance),
        (None, None) => true,
        _ => false,
    }
}

fn scalar_matches(cached: f64, source: f64, absolute_tolerance: f64) -> bool {
    (cached - source).abs() <= absolute_tolerance
}

fn path_enabled(input: &PathsInput) -> bool {
    input.control.mpath == 1 && input.control.ms == 1
}

fn read_input(work_dir: &Path) -> Result<PathsInput> {
    let input_path = work_dir.join("paths.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    PathsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn path_zero_output_branch(input: &PathsInput) -> bool {
    input.criteria.rmax.is_finite() && input.criteria.rmax < 1.0
}

fn path_generation_handoffs_present(work_dir: &Path) -> bool {
    ["phase.bin", "geom.dat", "global.inp"]
        .iter()
        .all(|name| work_dir.join(name).is_file())
}

fn validate_declared_path_source_handoffs(work_dir: &Path, input: &PathsInput) -> Result<()> {
    if path_zero_output_branch(input) {
        return Ok(());
    }
    if path_generation_handoffs_present(work_dir) {
        generate_paths_dat_from_handoffs(work_dir, input)?;
    } else {
        validate_present_path_source_handoffs(work_dir)?;
    }
    Ok(())
}

fn validate_present_path_source_handoffs(work_dir: &Path) -> Result<()> {
    let geom_path = work_dir.join("geom.dat");
    if geom_path.is_file() {
        let geom = read_geom_dat(work_dir)?;
        geom.to_pathfinder_handoff()
            .context("failed to validate PATH geom.dat source handoff")?;
    }
    let phase_path = work_dir.join("phase.bin");
    if phase_path.is_file() {
        let phase = read_phase_bin(&phase_path)
            .with_context(|| format!("failed to read {}", phase_path.display()))?;
        let handoff = phase
            .to_path_handoff()
            .context("failed to validate PATH phase.bin source handoff")?;
        handoff
            .criteria_tables()
            .context("failed to validate PATH phase.bin criteria tables")?;
    }
    let global_path = work_dir.join("global.inp");
    if global_path.is_file() {
        read_global_input(work_dir)?;
    }
    Ok(())
}

fn empty_paths_dat_for_input(input: &PathsInput) -> Result<PathsDatData> {
    Ok(PathsDatData {
        titles: paths_dat_titles_for_input(input)?,
        paths: Vec::new(),
    })
}

fn paths_dat_titles_for_input(input: &PathsInput) -> Result<Vec<String>> {
    if !input.criteria.rmax.is_finite()
        || !input.criteria.pcritk.is_finite()
        || !input.criteria.pcrith.is_finite()
        || !input.criteria.critpw.is_finite()
    {
        bail!("PATH paths.dat generation requires finite path criteria");
    }
    Ok(vec![format!(
        "PATH  Rmax={:6.3},  Keep_limit={:5.2}, Heap_limit{:5.2}  Pwcrit={:5.2}%",
        input.criteria.rmax, input.criteria.pcritk, input.criteria.pcrith, input.criteria.critpw
    )])
}

fn generate_paths_dat_from_handoffs(work_dir: &Path, input: &PathsInput) -> Result<PathsDatData> {
    let geom = read_geom_dat(work_dir)?;
    let geom_handoff = geom.to_pathfinder_handoff()?;
    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let phase_handoff = phase.to_path_handoff()?;
    let criteria_tables = phase_handoff.criteria_tables()?;
    let global = read_global_input(work_dir)?;
    let output_energy_count = criteria_tables.output_energy_count;
    let wave_numbers = criteria_tables
        .wave_numbers
        .get(..output_energy_count)
        .context("PATH phase wave-number table is shorter than neout")?;
    let mean_free_paths = criteria_tables
        .mean_free_paths
        .get(..output_energy_count)
        .context("PATH phase mean-free-path table is shorter than neout")?;
    let max_path_length = doubled_path_length(input.criteria.rmax)?;
    let output = pathfinder_reduction(PathfinderReductionInput {
        atom_positions: geom_handoff.atom_positions_angstrom.view(),
        atom_potentials: &geom_handoff.atom_potentials,
        first_bounce_degeneracies: &geom_handoff.first_bounce_degeneracies,
        fms_radius: input.criteria.rfms2,
        max_path_length,
        max_path_atoms: max_path_atoms(input)?,
        max_output_paths: FEFF_PATH_MAX_OUTPUT_PATHS,
        heap_cutoff: input.criteria.pcrith,
        output_cutoff: input.criteria.pcritk,
        search_normalization: -1.0,
        fbeta: criteria_tables.fbeta.view(),
        wave_numbers,
        mean_free_paths,
        start_energy_index: criteria_tables.zero_wave_energy_index,
        fbeta_critical: criteria_tables.fbeta_critical.view(),
        critical_wave_numbers: &criteria_tables.critical_wave_numbers,
        critical_mean_free_paths: &criteria_tables.critical_mean_free_paths,
        reduction_normalization: -1.0,
        criterion_percent: input.criteria.critpw,
        retention_reference: None,
        polarization: global.control.ipol,
        spin: path_spin_for_global(phase.spin_count, &global),
        electric_vector: global.evec,
        incident_vector: global.xivec,
        symmetry_case_override: symmetry_case_override(input),
        force_no_symmetry: force_no_symmetry(work_dir, &global)?,
    })?;
    let potential_labels = phase_handoff
        .potential_labels
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    pathsd_paths_dat_data(PathsdPathsDatDataInput {
        titles: &paths_dat_titles_for_input(input)?,
        reduction: &output.reduction,
        atom_positions: output.preparation.atom_positions.view(),
        atom_potentials: &output.preparation.atom_potentials,
        potential_labels: &potential_labels,
    })
    .with_context(|| {
        format!(
            "failed to generate {}",
            work_dir.join("paths.dat").display()
        )
    })
}

fn read_geom_dat(work_dir: &Path) -> Result<GeomDat> {
    let geom_path = work_dir.join("geom.dat");
    let geom_text = std::fs::read_to_string(&geom_path)
        .with_context(|| format!("failed to read {}", geom_path.display()))?;
    GeomDat::parse_str(&geom_path, &geom_text)
        .with_context(|| format!("failed to parse {}", geom_path.display()))
}

fn read_global_input(work_dir: &Path) -> Result<GlobalInput> {
    let global_path = work_dir.join("global.inp");
    let global_text = std::fs::read_to_string(&global_path)
        .with_context(|| format!("failed to read {}", global_path.display()))?;
    GlobalInput::parse_str(&global_path, &global_text)
        .with_context(|| format!("failed to parse {}", global_path.display()))
}

fn max_path_atoms(input: &PathsInput) -> Result<usize> {
    let max_from_input = input
        .control
        .nlegxx
        .checked_sub(1)
        .context("PATH source generation requires nlegxx > 1")?;
    if max_from_input <= 0 {
        bail!(
            "PATH source generation requires nlegxx > 1, got {}",
            input.control.nlegxx
        );
    }
    Ok(FEFF_PATH_MAX_PATH_ATOMS.min(max_from_input as usize))
}

fn doubled_path_length(rmax: f64) -> Result<f64> {
    let max_path_length = rmax * 2.0;
    if max_path_length.is_finite() {
        Ok(max_path_length)
    } else {
        bail!("PATH source generation requires finite doubled rmax")
    }
}

fn path_spin_for_global(spin_count: usize, global: &GlobalInput) -> i32 {
    if spin_count > 1 {
        global.control.ispin.abs()
    } else {
        global.control.ispin
    }
}

fn symmetry_case_override(input: &PathsInput) -> Option<u8> {
    (1..=7).contains(&input.ica).then_some(input.ica as u8)
}

fn force_no_symmetry(work_dir: &Path, global: &GlobalInput) -> Result<bool> {
    if global.control.do_nrixs == 1 {
        return Ok(true);
    }

    let eels_path = work_dir.join("eels.inp");
    if !eels_path.is_file() {
        return Ok(false);
    }
    let eels_text = std::fs::read_to_string(&eels_path)
        .with_context(|| format!("failed to read {}", eels_path.display()))?;
    let eels = EelsInput::parse_str(&eels_path, &eels_text)
        .with_context(|| format!("failed to parse {}", eels_path.display()))?;
    Ok(eels.calculate_elnes && eels.calculation_mode == 1)
}

fn write_cached_output(path: &Path, data: &PathsDatData) -> Result<()> {
    write_paths_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_generate_module_log(
    path: &Path,
    input: &PathsInput,
    paths: &PathsDatData,
) -> Result<()> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    let data = generated_path_module_log(input, paths)?;
    write_module_log_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

fn generated_path_module_log(input: &PathsInput, paths: &PathsDatData) -> Result<ModuleLogData> {
    let mut lines = vec![
        "Pathfinder: finding scattering paths...".to_string(),
        "Preparing plane wave scattering amplitudes".to_string(),
        "Searching for paths".to_string(),
    ];
    if input.criteria.rmax > 0.1 {
        lines.push(format!(
            "    Rmax{:8.4}  keep and heap limits{:12.7}{:12.7}",
            input.criteria.rmax, input.criteria.pcritk, input.criteria.pcrith
        ));
        lines.push("    Preparing neighbor table".to_string());
    }
    lines.push("Eliminating path degeneracies".to_string());
    if paths.paths.is_empty() {
        lines.push("0 paths retained.".to_string());
    } else {
        lines.push(format!(
            "    Unique paths{:7},  total paths{:8}",
            paths.paths.len(),
            rounded_total_path_degeneracy(paths)?
        ));
    }
    lines.push("Done with module: pathfinder.".to_string());

    let line_terminators = vec!["\n".to_string(); lines.len()];
    Ok(ModuleLogData {
        lines,
        line_terminators,
    })
}

fn rounded_total_path_degeneracy(paths: &PathsDatData) -> Result<usize> {
    let total = paths
        .paths
        .iter()
        .map(|path| path.degeneracy.round())
        .sum::<f64>();
    if total.is_finite() && total >= 0.0 && total <= usize::MAX as f64 {
        Ok(total as usize)
    } else {
        bail!("paths.dat degeneracy total is out of range for generated PATH log")
    }
}

#[cfg(test)]
mod tests {
    use super::{has_cached_paths_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
    use num_complex::Complex64;
    use refeff_io::phase_bin::PHASE_BIN_DEFAULT_PAD_WIDTH;
    use refeff_io::{
        CfAverage, GeomDat, GeomDatRow, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl,
        ModuleLogData, PathsControl, PathsCriteria, PathsDatAtom, PathsDatData, PathsDatPath,
        PathsInput, PhaseBinData, PhaseBinPotential, PhaseBinScalars, geom_dat_string,
        global_input_string, paths_input_string, read_module_log_dat, read_paths_dat,
        write_module_log_dat, write_paths_dat, write_phase_bin,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn path_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 0)?;
        write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_paths_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn path_module_requires_cache_or_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled PATH should require cache or source handoffs")?;

        assert!(error.to_string().contains(
            "PATH pathfinder generation requires cached paths.dat output or phase.bin/geom.dat/global.inp source handoffs"
        ));
        Ok(())
    }

    #[test]
    fn path_module_generates_zero_path_output_for_small_rmax() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input_with_rmax(temp.path(), 1, 0.75)?;

        assert!(has_cached_paths_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert_eq!(
            read_paths_dat(temp.path().join("paths.dat"))?,
            PathsDatData {
                titles: vec![
                    "PATH  Rmax= 0.750,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%"
                        .to_string(),
                ],
                paths: Vec::new(),
            }
        );
        let log = read_module_log_dat(temp.path().join("log4.dat"))?;
        assert_log_contains(&log, "Pathfinder: finding scattering paths...");
        assert_log_contains(&log, "Searching for paths");
        assert_log_contains(&log, "Eliminating path degeneracies");
        assert_log_contains(&log, "0 paths retained.");
        assert_log_contains(&log, "Done with module: pathfinder.");
        Ok(())
    }

    #[test]
    fn path_module_generates_paths_dat_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input_for_generation(temp.path())?;
        write_path_generation_handoffs(temp.path())?;

        assert!(has_cached_paths_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let paths = read_paths_dat(temp.path().join("paths.dat"))?;
        assert_eq!(
            paths.titles,
            vec![
                "PATH  Rmax= 2.050,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 0.00%".to_string()
            ]
        );
        assert_eq!(paths.paths.len(), 3);
        assert_eq!(paths.paths[0].index, 1);
        assert_eq!(paths.paths[0].degeneracy, 2.0);
        assert_eq!(paths.paths[0].atoms[0].potential_index, 1);
        assert_eq!(paths.paths[0].atoms[0].label, "Cu1");
        assert_close(paths.paths[0].effective_half_path_length_angstrom, 1.0);
        let retained_degeneracy = paths
            .paths
            .iter()
            .map(|path| path.degeneracy.round() as usize)
            .sum::<usize>();
        assert_eq!(retained_degeneracy, 6);
        let log = read_module_log_dat(temp.path().join("log4.dat"))?;
        assert_log_contains(&log, "Pathfinder: finding scattering paths...");
        assert_log_contains(&log, "    Preparing neighbor table");
        assert_log_contains(
            &log,
            &format!("    Unique paths{:7},  total paths{:8}", 3, 6),
        );
        Ok(())
    }

    #[test]
    fn path_module_does_not_advertise_malformed_cached_output_without_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 1)?;
        std::fs::write(temp.path().join("paths.dat"), b"not a paths.dat cache\n")?;

        assert!(!has_cached_paths_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed paths.dat should fail without source handoffs")?;

        assert!(error.to_string().contains("failed to read"));
        Ok(())
    }

    #[test]
    fn path_module_does_not_claim_cached_output_with_malformed_phase_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_paths_input_for_generation(temp.path())?;
        write_path_generation_handoffs(temp.path())?;
        let expected = sample_paths_dat();
        write_paths_dat(temp.path().join("paths.dat"), &expected)?;
        std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;

        assert!(!has_cached_paths_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed PATH phase source should block cached PATH completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("phase.bin"), "{chain}");
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        assert!(!temp.path().join("log4.dat").exists());
        Ok(())
    }

    #[test]
    fn path_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let expected = sample_paths_dat();
        std::fs::write(temp.path().join("paths.inp"), b"not a paths.inp handoff\n")?;
        write_paths_dat(temp.path().join("paths.dat"), &expected)?;

        assert!(!has_cached_paths_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed PATH input should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("paths.inp"), "{chain}");
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        assert!(!temp.path().join("log4.dat").exists());
        Ok(())
    }

    #[test]
    fn path_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let expected = sample_paths_dat();
        write_paths_dat(temp.path().join("paths.dat"), &expected)?;

        assert!(!has_cached_paths_output(temp.path())?);
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        assert!(!temp.path().join("log4.dat").exists());
        Ok(())
    }

    #[test]
    fn path_module_recovers_malformed_paths_dat_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input_for_generation(temp.path())?;
        write_path_generation_handoffs(temp.path())?;
        std::fs::write(temp.path().join("paths.dat"), b"not a paths.dat cache\n")?;

        assert!(has_cached_paths_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let paths = read_paths_dat(temp.path().join("paths.dat"))?;
        assert_eq!(paths.paths.len(), 3);
        assert_eq!(paths.paths[0].index, 1);
        assert_eq!(paths.paths[0].degeneracy, 2.0);
        let log = read_module_log_dat(temp.path().join("log4.dat"))?;
        assert_log_contains(&log, "Pathfinder: finding scattering paths...");
        assert_log_contains(
            &log,
            &format!("    Unique paths{:7},  total paths{:8}", 3, 6),
        );
        Ok(())
    }

    #[test]
    fn path_module_regenerates_stale_paths_dat_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input_for_generation(temp.path())?;
        write_path_generation_handoffs(temp.path())?;
        let stale = sample_paths_dat();
        write_paths_dat(temp.path().join("paths.dat"), &stale)?;

        assert!(has_cached_paths_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let paths = read_paths_dat(temp.path().join("paths.dat"))?;
        assert_eq!(paths.paths.len(), 3);
        assert_ne!(paths.paths.len(), stale.paths.len());
        assert_eq!(paths.paths[0].index, 1);
        assert_eq!(paths.paths[0].degeneracy, 2.0);
        let log = read_module_log_dat(temp.path().join("log4.dat"))?;
        assert_log_contains(&log, "Pathfinder: finding scattering paths...");
        assert_log_contains(
            &log,
            &format!("    Unique paths{:7},  total paths{:8}", 3, 6),
        );
        Ok(())
    }

    #[test]
    fn path_module_skips_when_ms_branch_is_disabled_like_feff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input_with_ms(temp.path(), 1, 0)?;
        write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_paths_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn path_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 1)?;
        let expected = sample_paths_dat();
        write_paths_dat(temp.path().join("paths.dat"), &expected)?;
        write_module_log_dat(temp.path().join("log4.dat"), &sample_module_log())?;
        let expected_log = read_module_log_dat(temp.path().join("log4.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert!(has_cached_paths_output(temp.path())?);
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        assert_eq!(
            read_module_log_dat(temp.path().join("log4.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn path_module_generates_missing_module_log_from_cached_paths() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 1)?;
        write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        let log = read_module_log_dat(temp.path().join("log4.dat"))?;
        assert_log_contains(&log, "Pathfinder: finding scattering paths...");
        assert_log_contains(&log, "Preparing plane wave scattering amplitudes");
        assert_log_contains(
            &log,
            "    Rmax  5.5000  keep and heap limits   0.0000000   0.0000000",
        );
        assert_log_contains(&log, "Eliminating path degeneracies");
        assert_log_contains(&log, "    Unique paths      1,  total paths      12");
        assert_log_contains(&log, "Done with module: pathfinder.");
        Ok(())
    }

    #[test]
    fn path_module_generates_zero_path_module_log_from_cached_paths() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input_with_rmax(temp.path(), 1, 0.1)?;
        write_paths_dat(temp.path().join("paths.dat"), &empty_paths_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        let log = read_module_log_dat(temp.path().join("log4.dat"))?;
        assert_log_contains(&log, "Pathfinder: finding scattering paths...");
        assert_log_contains(&log, "Searching for paths");
        assert_log_contains(&log, "Eliminating path degeneracies");
        assert_log_contains(&log, "0 paths retained.");
        assert_log_contains(&log, "Done with module: pathfinder.");
        Ok(())
    }

    #[test]
    fn path_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_paths_dir()? else {
            crate::require_fixture!("PATH reference test; generated EXAFS/Cu reference not found");
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(
            reference_dir.join("paths.inp"),
            temp.path().join("paths.inp"),
        )?;
        std::fs::copy(
            reference_dir.join("paths.dat"),
            temp.path().join("paths.dat"),
        )?;
        let log_source = reference_dir.join("log4.dat");
        if log_source.is_file() {
            std::fs::copy(log_source, temp.path().join("log4.dat"))?;
        }
        let expected = read_paths_dat(temp.path().join("paths.dat"))?;
        let expected_log = optional_module_log(temp.path().join("log4.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, expected.paths.len());
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        if let Some(expected) = expected_log {
            assert_eq!(read_module_log_dat(temp.path().join("log4.dat"))?, expected);
        }
        Ok(())
    }

    fn write_paths_input(work_dir: &Path, mpath: i32) -> Result<()> {
        write_paths_input_with_ms(work_dir, mpath, mpath)
    }

    fn write_paths_input_with_ms(work_dir: &Path, mpath: i32, ms: i32) -> Result<()> {
        write_paths_input_with_ms_and_rmax(work_dir, mpath, ms, 5.5)
    }

    fn write_paths_input_with_rmax(work_dir: &Path, mpath: i32, rmax: f64) -> Result<()> {
        write_paths_input_with_ms_and_rmax(work_dir, mpath, mpath, rmax)
    }

    fn write_paths_input_with_ms_and_rmax(
        work_dir: &Path,
        mpath: i32,
        ms: i32,
        rmax: f64,
    ) -> Result<()> {
        let input = PathsInput {
            control: PathsControl {
                mpath,
                ms,
                nncrit: 0,
                nlegxx: 7,
                ipr4: 0,
            },
            criteria: PathsCriteria {
                critpw: 2.5,
                pcritk: 0.0,
                pcrith: 0.0,
                rmax,
                rfms2: -1.0,
            },
            ica: -1,
        };
        std::fs::write(work_dir.join("paths.inp"), paths_input_string(&input)?)?;
        Ok(())
    }

    fn write_paths_input_for_generation(work_dir: &Path) -> Result<()> {
        let input = PathsInput {
            control: PathsControl {
                mpath: 1,
                ms: 1,
                nncrit: 0,
                nlegxx: 3,
                ipr4: 0,
            },
            criteria: PathsCriteria {
                critpw: 0.0,
                pcritk: 0.0,
                pcrith: 0.0,
                rmax: 2.05,
                rfms2: 0.5,
            },
            ica: -1,
        };
        std::fs::write(work_dir.join("paths.inp"), paths_input_string(&input)?)?;
        Ok(())
    }

    fn write_path_generation_handoffs(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&sample_path_generation_geom())?,
        )?;
        std::fs::write(
            work_dir.join("global.inp"),
            global_input_string(&sample_global_input())?,
        )?;
        write_phase_bin(work_dir.join("phase.bin"), &sample_path_phase_bin())?;
        Ok(())
    }

    fn empty_paths_dat() -> PathsDatData {
        PathsDatData {
            titles: vec![
                "PATH  Rmax= 0.100,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%".to_string(),
            ],
            paths: Vec::new(),
        }
    }

    fn sample_path_generation_geom() -> GeomDat {
        GeomDat {
            nat: 5,
            nph: 2,
            model_atoms: vec![1, 2, 4],
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
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 2,
                },
                GeomDatRow {
                    index: 3,
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 0,
                },
                GeomDatRow {
                    index: 4,
                    x: 0.0,
                    y: 0.0,
                    z: 3.0,
                    iph: 2,
                    boundary: 3,
                },
                GeomDatRow {
                    index: 5,
                    x: 1.0,
                    y: 1.0,
                    z: 0.0,
                    iph: 2,
                    boundary: 1,
                },
            ],
        }
    }

    fn sample_global_input() -> GlobalInput {
        GlobalInput {
            cfaverage: CfAverage {
                nabs: 1,
                iphabs: 0,
                rclabs: 0.0,
            },
            control: GlobalControl {
                ipol: 0,
                ispin: 0,
                le2: 0,
                elpty: 0.0,
                angks: 0.0,
                l2lp: 0,
                do_nrixs: 0,
                ldecmx: 0,
                lj: 0,
            },
            evec: [0.0, 0.0, 1.0],
            xivec: [1.0, 0.0, 0.0],
            spvec: [0.0, 0.0, 1.0],
            polarization_tensor: [[0.0; 6]; 3],
            norms: GlobalNorms {
                evnorm: 1.0,
                xivnorm: 1.0,
                spvnorm: 1.0,
            },
            q_control: GlobalQControl {
                nq: 0,
                imdff: 0,
                qaverage: false,
                mixdff: false,
            },
            q_vectors: Vec::new(),
            mdff: None,
        }
    }

    fn sample_path_phase_bin() -> PhaseBinData {
        let energy_count = 5;
        let spin_count = 1;
        let transition_count = 1;
        let energy_grid =
            Array1::from_iter((0..energy_count).map(|energy| {
                Complex64::new(0.1 + 0.2 * energy as f64, 0.002 * (energy + 1) as f64)
            }));
        let reference_energy = Array2::zeros((energy_count, spin_count));
        let potentials = (0..3)
            .map(|potential| PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: format!("Cu{potential}"),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, 3, spin_count),
                    |(energy, signed_l, _)| {
                        Complex64::new(
                            0.03 * (potential + 1) as f64
                                + 0.01 * energy as f64
                                + 0.02 * signed_l as f64,
                            0.002 * (signed_l + 1) as f64,
                        )
                    },
                ),
            })
            .collect();
        let transition_moments = Array4::<Complex64>::from_shape_fn(
            (energy_count, 1, transition_count, spin_count).f(),
            |(energy, _, _, _)| Complex64::new(0.1 + 0.01 * energy as f64, 0.0),
        );

        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 2,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: transition_count,
            transition_count,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.2,
                fermi_level: 0.0,
                edge_energy: 8_979.0,
            },
            energy_grid,
            reference_energy,
            potentials,
            transition_moments,
            raw_pads: None,
        }
    }

    fn sample_paths_dat() -> PathsDatData {
        PathsDatData {
            titles: vec![
                "PATH  Rmax= 5.500,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%".to_string(),
            ],
            paths: vec![PathsDatPath {
                index: 1,
                degeneracy: 12.0,
                effective_half_path_length_angstrom: 2.5527,
                row_header:
                    "      x           y           z     ipot  label      rleg      beta        eta"
                        .to_string(),
                atoms: vec![
                    PathsDatAtom {
                        position_angstrom: [-1.805, -1.805, 0.0],
                        potential_index: 1,
                        label: "Cu".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(180.0),
                        eta_degrees: Some(0.0),
                    },
                    PathsDatAtom {
                        position_angstrom: [0.0, 0.0, 0.0],
                        potential_index: 0,
                        label: "Cu".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(180.0),
                        eta_degrees: Some(0.0),
                    },
                ],
            }],
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Pathfinder: finding scattering paths...".to_string(),
                "Preparing plane wave scattering amplitudes".to_string(),
                "Searching for paths".to_string(),
                "Done with module: pathfinder.".to_string(),
            ],
            line_terminators: vec![
                "\n".to_string(),
                "\n".to_string(),
                "\n".to_string(),
                "\n".to_string(),
            ],
        }
    }

    fn assert_log_contains(log: &ModuleLogData, expected: &str) {
        assert!(
            log.lines.iter().any(|line| line == expected),
            "expected log to contain {expected:?}, got {:?}",
            log.lines
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1.0e-6;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
        let path = path.as_ref();
        if path.is_file() {
            Ok(Some(read_module_log_dat(path)?))
        } else {
            Ok(None)
        }
    }

    fn reference_paths_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = ["paths.inp", "paths.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
