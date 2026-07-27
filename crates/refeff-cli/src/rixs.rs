use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2, Array3, Array4, Axis, Slice};
use refeff_core::{
    Complex, FEFF_HARTREE_EV, RixsCoreHolePotentialInput, RixsIncidentAmplitudeConvolutionInput,
    RixsInitialAmplitudeInput, RixsPoleNormalization, RixsPoleNormalizationInput,
    RixsPostRawSpectrum, RixsPostRawSpectrumInput, RixsRadialOverlapInput,
    RixsRawCrossSectionInput, RixsSatelliteSpectrumInput, RixsSelfEnergyGridInput,
    RixsTransitionMatrixSetup, RixsWaveNumberInput, RixsWaveNumbers,
    rixs_core_hole_potential_difference, rixs_default_pole_normalization,
    rixs_incident_amplitude_convolution, rixs_initial_transition_amplitudes, rixs_normalize_poles,
    rixs_post_raw_spectrum, rixs_prepare_self_energy_grid, rixs_radial_transition_overlaps,
    rixs_raw_cross_section, rixs_satellite_spectrum, rixs_wave_numbers,
};
use refeff_io::{
    EdgesDatData, EdgesDatRow, GgDatRixsHandoff, GlobalInput, ModuleLogData, MpseDatData,
    PhaseBinRixsHandoff, RixsDatFromFinalSpectrum, RixsInput, RixsLineData, RixsMapData,
    RixsSkipCalcEdgeOffsets, WscrnDatData, XsectDatRixsHandoff, XsphRlDatRixsHandoff,
    gg_dat_rixs_handoff, phase_bin_rixs_handoff_from_phase_bin,
    phase_bin_rixs_transition_phase_shifts_from_handoff,
    phase_bin_rixs_transition_setup_from_handoffs, read_edges_dat, read_gg_bin,
    read_module_log_dat, read_mpse_dat, read_phase_bin, read_pot_bin, read_rixs_line,
    read_rixs_map, read_wscrn_dat, read_xmu_dat, read_xsect_dat, read_xsph_rl_dat,
    rixs_dat_from_final_spectrum, rixs_line_from_map_diagonal,
    rixs_skip_calc_edge_offsets_from_edges_dat, rixs_skip_calc_herfd_from_rixs_et_map,
    rixs_skip_calc_outputs_from_rixs_et_map, rixs_skip_calc_satellite_outputs, write_edges_dat,
    write_module_log_dat, write_rixs_line, write_rixs_map, write_wscrn_dat, xsect_dat_rixs_handoff,
    xsph_rl_dat_rixs_handoff,
};

use crate::{screen, work_dir_for_input, xsph};

const RIXS_SOURCE_REQUIREMENT_ERROR: &str =
    "RIXS generation requires cached spectra or complete phase/rl/wscrn/xsect source handoffs";

const RIXS_SOLVER_HANDOFF_FILES: &[&str] = &[
    "phase.bin",
    "phase_1.bin",
    "phase_2.bin",
    "rl_1.dat",
    "rl_2.dat",
    "rl.dat",
    "wscrn_1.dat",
    "wscrn_2.dat",
    "wscrn.dat",
    "gg_1.bin",
    "gg_2.bin",
    "gg.bin",
    "xsect_2.dat",
    "xsect.dat",
];

const RIXS_EXPLICIT_EDGE_HANDOFFS: &[(&str, &str, &str)] = &[
    ("phase.bin", "phase_1.bin", "phase_2.bin"),
    ("rl.dat", "rl_1.dat", "rl_2.dat"),
    ("wscrn.dat", "wscrn_1.dat", "wscrn_2.dat"),
    ("gg.bin", "gg_1.bin", "gg_2.bin"),
];

/// Materialize the two stock FEFF edge calculations used by RIXS.
///
/// The source codecs retain every contour row (`ne`). The RIXS solver later
/// projects those handoffs to the `xsect.dat` main grid (`ne1`).
pub(crate) fn prepare_two_edge_handoffs(input_path: &Path, work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !rixs_enabled(&input) || input.switches.skip_calc {
        return Ok(0);
    }
    if work_dir.join("rixsET.dat").is_file() {
        return Ok(0);
    }

    let explicit_paths = RIXS_EXPLICIT_EDGE_HANDOFFS
        .iter()
        .flat_map(|(_, first, second)| [work_dir.join(first), work_dir.join(second)])
        .chain(std::iter::once(work_dir.join("xsect_2.dat")))
        .collect::<Vec<_>>();
    let explicit_count = explicit_paths.iter().filter(|path| path.is_file()).count();
    if explicit_count == explicit_paths.len() {
        return Ok(0);
    }
    ensure!(
        explicit_count == 0,
        "incomplete RIXS two-edge handoff: found {explicit_count} of {} required files; refusing to mix edge calculations",
        explicit_paths.len()
    );
    if input.edges.len() != 2 {
        return Ok(0);
    }

    let source = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read RIXS source input {}", input_path.display()))?;
    let incident_edge = &input.edges[0];
    let final_edge = &input.edges[1];
    validate_rixs_edge_directory_name(incident_edge)?;
    validate_rixs_edge_directory_name(final_edge)?;

    let first_dir = work_dir.join(incident_edge);
    let second_dir = work_dir.join(final_edge);
    run_rixs_edge_pipeline(&source, incident_edge, incident_edge, &first_dir)?;
    run_rixs_edge_pipeline(&source, incident_edge, final_edge, &second_dir)?;
    recover_edge_wscrn_if_needed(&first_dir, None)?;
    recover_edge_wscrn_if_needed(&second_dir, Some(&first_dir.join("wscrn.dat")))?;

    for (source_name, _, _) in RIXS_EXPLICIT_EDGE_HANDOFFS {
        ensure!(
            first_dir.join(source_name).is_file(),
            "RIXS incident-edge calculation did not produce {}",
            first_dir.join(source_name).display()
        );
        ensure!(
            second_dir.join(source_name).is_file(),
            "RIXS final-edge calculation did not produce {}",
            second_dir.join(source_name).display()
        );
    }
    ensure!(
        second_dir.join("xsect.dat").is_file(),
        "RIXS final-edge calculation did not produce {}",
        second_dir.join("xsect.dat").display()
    );
    write_rixs_edges_if_needed(work_dir, &input, &first_dir, &second_dir)?;

    let mut staged = Vec::with_capacity(explicit_paths.len());
    for (source_name, first_name, second_name) in RIXS_EXPLICIT_EDGE_HANDOFFS {
        stage_rixs_handoff_copy(
            work_dir,
            &first_dir.join(source_name),
            first_name,
            &mut staged,
        )?;
        stage_rixs_handoff_copy(
            work_dir,
            &second_dir.join(source_name),
            second_name,
            &mut staged,
        )?;
    }
    stage_rixs_handoff_copy(
        work_dir,
        &second_dir.join("xsect.dat"),
        "xsect_2.dat",
        &mut staged,
    )?;
    for (staged_path, destination) in &staged {
        std::fs::rename(staged_path, destination).with_context(|| {
            format!(
                "failed to publish RIXS handoff {} as {}",
                staged_path.display(),
                destination.display()
            )
        })?;
    }
    Ok(staged.len())
}

fn write_rixs_edges_if_needed(
    work_dir: &Path,
    input: &RixsInput,
    first_dir: &Path,
    second_dir: &Path,
) -> Result<()> {
    let path = work_dir.join("edges.dat");
    if !input.switches.read_poles || path.is_file() {
        return Ok(());
    }
    let incident_path = first_dir.join("xsect.dat");
    let incident = read_xsect_dat(&incident_path)
        .with_context(|| format!("failed to read {}", incident_path.display()))?;
    let incident_pot_path = first_dir.join("pot.bin");
    let incident_pot = read_pot_bin(&incident_pot_path)
        .with_context(|| format!("failed to read {}", incident_pot_path.display()))?;
    ensure!(
        incident_pot.scalars.edge_position.is_finite(),
        "{} edge energy is not finite",
        incident_pot_path.display()
    );
    let final_path = second_dir.join("xsect.dat");
    let final_state = read_xsect_dat(&final_path)
        .with_context(|| format!("failed to read {}", final_path.display()))?;
    let final_row = if input
        .edges
        .get(1)
        .is_some_and(|edge| edge.eq_ignore_ascii_case("VAL"))
    {
        EdgesDatRow {
            chemical_potential: 0.0,
            matrix_element: 1.0,
            core_hole_width: input.broadening.gam_exp_2,
        }
    } else {
        EdgesDatRow {
            chemical_potential: final_state.scalars.chemical_potential,
            matrix_element: 1.0,
            core_hole_width: final_state.core_hole_width_ev / FEFF_HARTREE_EV,
        }
    };
    let edges = EdgesDatData {
        header_lines: vec!["# emu, M_kk, gam".to_string()],
        rows: vec![
            EdgesDatRow {
                // FEFF's RIXS scheduler retains the binary POT `emu` scalar.
                // `xsect.dat` renders this value at limited precision, which
                // otherwise shifts the complete incident-energy axis.
                chemical_potential: incident_pot.scalars.edge_position,
                matrix_element: 1.0,
                core_hole_width: incident.core_hole_width_ev / FEFF_HARTREE_EV,
            },
            final_row,
        ],
    };
    write_edges_dat(&path, &edges).with_context(|| format!("failed to write {}", path.display()))
}

fn recover_edge_wscrn_if_needed(edge_dir: &Path, zero_grid_source: Option<&Path>) -> Result<()> {
    if edge_dir.join("wscrn.dat").is_file() {
        return Ok(());
    }
    screen::recover_wscrn_from_vtot_and_apot_in_dir(edge_dir).with_context(|| {
        format!(
            "failed to recover the RIXS edge screened potential in {}",
            edge_dir.display()
        )
    })?;
    if edge_dir.join("wscrn.dat").is_file() {
        return Ok(());
    }
    let Some(zero_grid_source) = zero_grid_source else {
        return Ok(());
    };
    let source = read_wscrn_dat(zero_grid_source)
        .with_context(|| format!("failed to read {}", zero_grid_source.display()))?;
    let row_count = source.row_count();
    let zero_final_state = WscrnDatData {
        header_lines: source.header_lines,
        radius_bohr: source.radius_bohr,
        screened_potential: Array1::zeros(row_count),
        core_hole_potential: Array1::zeros(row_count),
    };
    let path = edge_dir.join("wscrn.dat");
    write_wscrn_dat(&path, &zero_final_state)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn run_rixs_edge_pipeline(
    source: &str,
    incident_edge: &str,
    calculation_edge: &str,
    output_dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let derived_input = rixs_edge_input_string(source, incident_edge, calculation_edge)?;
    let derived_path = output_dir.join("feff.inp");
    std::fs::write(&derived_path, derived_input)
        .with_context(|| format!("failed to write {}", derived_path.display()))?;
    crate::execute_rixs_edge_pipeline(&derived_path, output_dir).with_context(|| {
        format!(
            "failed to run RIXS edge calculation {calculation_edge} in {}",
            output_dir.display()
        )
    })
}

fn rixs_edge_input_string(
    source: &str,
    incident_edge: &str,
    calculation_edge: &str,
) -> Result<String> {
    let mut output = String::new();
    if calculation_edge.eq_ignore_ascii_case("VAL") {
        output.push_str("COREHOLE NONE\n");
    } else {
        let core_state = rixs_core_state(incident_edge)
            .with_context(|| format!("unsupported RIXS incident edge {incident_edge:?}"))?;
        output.push_str(&format!("ICORE {core_state}\nCOREHOLE RPA\n"));
    }
    output.push_str("RLPRINT\n");
    output.push_str(&format!("EDGE {incident_edge}\nXANES 20\n"));

    for line in source.lines() {
        let card = line
            .trim_start()
            .split_ascii_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();
        if matches!(
            card.as_str(),
            "COREHOLE" | "HOLE" | "ICORE" | "RLPRINT" | "EDGE" | "RIXS" | "XES" | "XANES"
        ) {
            continue;
        }
        output.push_str(line.trim_end_matches('\r'));
        output.push('\n');
    }
    Ok(output)
}

fn rixs_core_state(edge: &str) -> Option<i32> {
    match edge.to_ascii_uppercase().as_str() {
        "K" => Some(1),
        "L1" => Some(2),
        "L2" => Some(3),
        "L3" => Some(4),
        "M1" => Some(5),
        "M2" => Some(6),
        "M3" => Some(7),
        "M4" => Some(8),
        "M5" => Some(10),
        "N1" => Some(11),
        "N2" => Some(12),
        "N3" => Some(13),
        "N4" => Some(14),
        "N5" => Some(15),
        _ => None,
    }
}

fn validate_rixs_edge_directory_name(edge: &str) -> Result<()> {
    ensure!(
        !edge.is_empty()
            && edge
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_'),
        "RIXS edge label {edge:?} is not safe to use as a calculation directory"
    );
    Ok(())
}

fn stage_rixs_handoff_copy(
    work_dir: &Path,
    source: &Path,
    destination_name: &str,
    staged: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<()> {
    let destination = work_dir.join(destination_name);
    let temporary = work_dir.join(format!(".{destination_name}.rixs-tmp"));
    std::fs::copy(source, &temporary).with_context(|| {
        format!(
            "failed to stage RIXS handoff {} as {}",
            source.display(),
            temporary.display()
        )
    })?;
    staged.push((temporary, destination));
    Ok(())
}

/// Run the supported FEFF RIXS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_cached_rixs_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_solver_handoff(work_dir)? {
        return run_supported_solver_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Whether a FEFF RIXS run can be satisfied from existing spectrum caches.
pub(crate) fn has_cached_rixs_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("rixs.inp").is_file() {
        return Ok(false);
    }
    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !rixs_enabled(&input) {
        return Ok(false);
    }
    if input.switches.skip_calc {
        return can_run_skip_calc_from_cached_map(work_dir, &input);
    }
    if validate_declared_rixs_solver_handoffs(work_dir, &input).is_err() {
        return Ok(false);
    }
    Ok(can_use_cached_rixs_output(work_dir, &input, &outputs))
}

fn can_run_skip_calc_from_cached_map(work_dir: &Path, input: &RixsInput) -> Result<bool> {
    let map_path = work_dir.join("rixsET.dat");
    if !map_path.is_file() {
        return Ok(false);
    }

    let map = match read_rixs_map(&map_path) {
        Ok(map) => map,
        Err(_) => return Ok(false),
    };
    let edges = match read_skip_calc_edge_offsets(work_dir, input) {
        Ok(edges) => edges,
        Err(_) => return Ok(false),
    };
    let supported = if let Some(edges) = edges {
        rixs_skip_calc_outputs_from_rixs_et_map(&map, input.energy_window, edges).is_ok()
            && skip_calc_satellite_source_is_usable(work_dir, &map, input, edges)
    } else {
        rixs_skip_calc_herfd_from_rixs_et_map(&map).is_ok()
    };

    if !supported {
        return Ok(false);
    }
    if prepare_skip_calc_cached_outputs(work_dir, input).is_err() {
        return Ok(false);
    }
    Ok(prepare_module_log_cache(work_dir).is_ok() || module_log_is_recoverable(work_dir, true))
}

fn skip_calc_satellite_source_is_usable(
    work_dir: &Path,
    map: &RixsMapData,
    input: &RixsInput,
    edges: RixsSkipCalcEdgeOffsets,
) -> bool {
    if !input.switches.mbconv || !work_dir.join("XES").join("xmu.dat").is_file() {
        return true;
    }
    skip_calc_satellite_outputs(work_dir, map, input.energy_window, edges, input.xmu).is_ok()
}

fn prepare_skip_calc_cached_outputs(work_dir: &Path, input: &RixsInput) -> Result<()> {
    prepare_missing_final_outputs_from_cached_rixs_maps(work_dir, input)?;
    prepare_missing_diagonal_line_outputs(work_dir)?;
    let outputs = cached_output_paths(work_dir)?;
    for output in &outputs {
        prepare_cached_output_or_recoverable(work_dir, input, output)?;
    }
    Ok(())
}

fn can_use_cached_rixs_output(
    work_dir: &Path,
    input: &RixsInput,
    outputs: &[CachedOutputPath],
) -> bool {
    prepare_cached_rixs_output(work_dir, input, outputs).is_ok()
}

fn prepare_cached_rixs_output(
    work_dir: &Path,
    input: &RixsInput,
    outputs: &[CachedOutputPath],
) -> Result<()> {
    let source_outputs_available = can_generate_any_rixs_output_from_sources(work_dir, input)?;
    prepare_missing_final_outputs_from_cached_rixs_maps(work_dir, input)?;
    prepare_missing_diagonal_line_outputs(work_dir)?;
    for output in outputs {
        prepare_cached_output_or_recoverable(work_dir, input, output)?;
    }
    if module_log_is_recoverable(work_dir, source_outputs_available) {
        return Ok(());
    }
    prepare_module_log_cache(work_dir)
}

/// Whether partial RIXS source state can be parsed and validated before full
/// spectrum generation is available.
pub(crate) fn has_supported_solver_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("rixs.inp").is_file() {
        return Ok(false);
    }
    if !has_rixs_solver_handoff_files(work_dir) {
        return Ok(false);
    }
    if has_cached_rixs_output(work_dir)? {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !rixs_enabled(&input) || input.switches.skip_calc {
        return Ok(false);
    }

    let bundle = match build_optional_rixs_solver_handoffs(work_dir, &input) {
        Ok(bundle) => bundle,
        Err(_) => return Ok(false),
    };
    let _raw_solver_ready = bundle.has_raw_solver_inputs();
    Ok(true)
}

/// Validate or write Rust-backed RIXS solver handoff outputs.
///
/// Complete source-backed handoff bundles write the RIXS map/line outputs,
/// including optional satellite outputs. Partial bundles still stop before the
/// remaining solver boundary, preserving validation-only handoff reporting for
/// incomplete source state.
pub(crate) fn run_supported_solver_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !rixs_enabled(&input) || input.switches.skip_calc {
        return Ok(0);
    }
    if !has_rixs_solver_handoff_files(work_dir) {
        return Ok(0);
    }
    if has_cached_rixs_output(work_dir)? {
        return Ok(0);
    }

    let bundle = match build_optional_rixs_solver_handoffs(work_dir, &input) {
        Ok(bundle) => bundle,
        Err(_) => return Ok(0),
    };
    match write_optional_rixs_source_outputs(work_dir, &bundle) {
        Ok(Some(written)) => return Ok(written),
        Ok(None) => {}
        Err(_) => return Ok(0),
    }
    Ok(rixs_solver_handoff_file_count(work_dir))
}

/// Run the FEFF RIXS path from cached spectra or complete source handoffs.
///
/// This preserves cached FEFF RIXS output directories by validating and
/// re-rendering typed two-axis maps (`rixsET*`, `rixsEE*`, `rixs0.dat`,
/// `rixs1.dat`) and one-axis line spectra (`herfd*`, `xasEI*`, `xasEF*`,
/// `xas0.dat`, `xas1.dat`), plus `logrixs.dat` module diagnostics. Complete
/// phase/radial/screening/cross-section source handoffs generate standard and
/// optional satellite spectra directly. FEFF `SkipCalc` mode can also derive
/// cached-map outputs from `rixsET.dat`; when `edges.dat` is available this
/// includes the FEFF-style `xasEI.dat`, `xasEF.dat`, and `rixsEE.dat`
/// transforms.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !rixs_enabled(&input) {
        return Ok(0);
    }

    let outputs = if input.switches.skip_calc {
        run_skip_calc_from_cached_map(work_dir)?
    } else {
        validate_declared_rixs_solver_handoffs(work_dir, &input)?;
        let outputs = cached_output_paths(work_dir)?;
        let cached_output_usable =
            !outputs.is_empty() && can_use_cached_rixs_output(work_dir, &input, &outputs);
        let source_outputs_preferred =
            cached_output_usable && source_outputs_should_fill_partial_cache(work_dir, &input);
        if source_outputs_preferred
            && let Ok(bundle) = build_optional_rixs_solver_handoffs(work_dir, &input)
            && let Some(written) = write_optional_rixs_source_outputs(work_dir, &bundle)?
        {
            return Ok(written);
        }
        if outputs.is_empty() || (!cached_output_usable && has_rixs_solver_handoff_files(work_dir))
        {
            let bundle = build_optional_rixs_solver_handoffs(work_dir, &input)?;
            if let Some(written) = write_optional_rixs_source_outputs(work_dir, &bundle)? {
                return Ok(written);
            }
            bail!(RIXS_SOURCE_REQUIREMENT_ERROR);
        }
        let source_outputs_written =
            write_missing_final_outputs_from_cached_rixs_maps(work_dir, &input)?
                + write_missing_diagonal_line_outputs(work_dir)?
                > 0;
        let outputs = cached_output_paths(work_dir)?;
        for output in &outputs {
            roundtrip_cached_output(output)?;
        }
        return Ok(outputs.len()
            + write_or_recover_module_log(
                &work_dir.join("logrixs.dat"),
                &outputs,
                source_outputs_written,
            )?);
    };

    Ok(outputs.len() + write_or_recover_module_log(&work_dir.join("logrixs.dat"), &outputs, true)?)
}

fn source_outputs_should_fill_partial_cache(work_dir: &Path, input: &RixsInput) -> bool {
    if !has_rixs_solver_handoff_files(work_dir) {
        return false;
    }

    !source_final_outputs_are_usable(work_dir, regular_final_output_names())
        || (input.switches.mbconv
            && work_dir.join("XES").join("xmu.dat").is_file()
            && !source_final_outputs_are_usable(work_dir, satellite_final_output_names()))
}

fn rixs_enabled(input: &RixsInput) -> bool {
    input.run
}

fn has_rixs_solver_handoff_files(work_dir: &Path) -> bool {
    has_declared_rixs_solver_handoff_files(work_dir)
        || has_recoverable_shared_wscrn_handoff(work_dir)
}

fn has_declared_rixs_solver_handoff_files(work_dir: &Path) -> bool {
    RIXS_SOLVER_HANDOFF_FILES
        .iter()
        .any(|name| work_dir.join(name).is_file())
}

fn has_recoverable_shared_wscrn_handoff(work_dir: &Path) -> bool {
    !work_dir.join("wscrn_1.dat").is_file()
        && screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir)
}

fn rixs_solver_handoff_file_count(work_dir: &Path) -> usize {
    RIXS_SOLVER_HANDOFF_FILES
        .iter()
        .filter(|name| work_dir.join(name).is_file())
        .count()
}

fn read_input(work_dir: &Path) -> Result<RixsInput> {
    let input_path = work_dir.join("rixs.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    RixsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn build_optional_rixs_solver_handoffs(
    work_dir: &Path,
    input: &RixsInput,
) -> Result<RixsSolverHandoffBundle> {
    let first_phase = read_optional_rixs_phase_handoff(work_dir, "phase_1.bin")?;
    let second_phase = read_optional_rixs_phase_handoff(work_dir, "phase_2.bin")?;
    let shared_phase = if first_phase.is_none() && second_phase.is_none() {
        read_optional_rixs_phase_handoff(work_dir, "phase.bin")?
    } else {
        None
    };
    let (incident_phase, final_phase) = resolve_rixs_edge_pair(
        "phase_1.bin",
        first_phase,
        "phase_2.bin",
        second_phase,
        shared_phase,
    )?;
    let solver_input = rixs_solver_input(input, incident_phase.as_ref());
    let global = if incident_phase.is_some() || final_phase.is_some() {
        read_optional_rixs_global_input(work_dir)?
    } else {
        None
    };
    let incident_transition_setup =
        build_optional_rixs_transition_setup(work_dir, global.as_ref(), incident_phase.as_ref())?;
    let final_transition_setup =
        build_optional_rixs_transition_setup(work_dir, global.as_ref(), final_phase.as_ref())?;
    validate_optional_rixs_transition_setup_compatibility(
        incident_transition_setup.as_ref(),
        final_transition_setup.as_ref(),
    )?;
    let incident_phase_shifts = validate_optional_rixs_transition_phase_shifts(
        work_dir,
        incident_phase.as_ref(),
        incident_transition_setup.as_ref(),
    )?;
    let final_phase_shifts = validate_optional_rixs_transition_phase_shifts(
        work_dir,
        final_phase.as_ref(),
        final_transition_setup.as_ref(),
    )?;

    let first_radial = read_optional_rixs_rl_handoff(work_dir, "rl_1.dat")?;
    let second_radial = read_optional_rixs_rl_handoff(work_dir, "rl_2.dat")?;
    let shared_radial = if first_radial.is_none() && second_radial.is_none() {
        read_optional_rixs_rl_handoff(work_dir, "rl.dat")?
    } else {
        None
    };
    let (first_radial, second_radial) = resolve_rixs_edge_pair(
        "rl_1.dat",
        first_radial,
        "rl_2.dat",
        second_radial,
        shared_radial,
    )?;
    validate_optional_rixs_rl_handoff_against_phase(
        work_dir,
        first_radial.as_ref(),
        incident_phase.as_ref(),
    )?;
    validate_optional_rixs_rl_handoff_against_phase(
        work_dir,
        second_radial.as_ref(),
        final_phase.as_ref(),
    )?;
    recover_shared_wscrn_handoff_if_needed(work_dir)
        .context("failed to recover shared RIXS wscrn.dat handoff")?;
    let potential_difference = validate_optional_rixs_wscrn_handoffs(
        work_dir,
        first_radial.as_ref().map(|source| &source.handoff),
        second_radial.as_ref().map(|source| &source.handoff),
    )?;

    let first_green = read_optional_rixs_green_handoff(work_dir, "gg_1.bin")?;
    let second_green = read_optional_rixs_green_handoff(work_dir, "gg_2.bin")?;
    let shared_green = if first_green.is_none() && second_green.is_none() {
        read_optional_rixs_green_handoff(work_dir, "gg.bin")?
    } else {
        None
    };
    let (first_green, second_green) = resolve_rixs_edge_pair(
        "gg_1.bin",
        first_green,
        "gg_2.bin",
        second_green,
        shared_green,
    )?;
    validate_optional_rixs_green_handoff_against_phase(
        work_dir,
        first_green.as_ref(),
        incident_phase.as_ref(),
    )?;
    validate_optional_rixs_green_handoff_against_phase(
        work_dir,
        second_green.as_ref(),
        final_phase.as_ref(),
    )?;

    let xsect = validate_optional_rixs_xsect_handoff(
        work_dir,
        incident_phase.as_ref(),
        second_radial.as_ref().map(|source| &source.handoff),
    )?;
    let radial_overlaps = validate_optional_rixs_radial_overlaps(
        work_dir,
        &solver_input,
        incident_transition_setup.as_ref(),
        first_radial.as_ref().map(|source| &source.handoff),
        second_radial.as_ref().map(|source| &source.handoff),
        potential_difference.as_ref(),
        xsect.as_ref().map(|source| &source.handoff),
    )?;
    let initial_transition_amplitudes = validate_optional_rixs_initial_transition_amplitudes(
        work_dir,
        &solver_input,
        incident_transition_setup.as_ref(),
        incident_phase.as_ref(),
        incident_phase_shifts.as_ref(),
        first_green.as_ref(),
        xsect.as_ref().map(|source| &source.handoff),
        radial_overlaps.as_ref(),
    )?;
    let wave_numbers = validate_optional_rixs_wave_numbers(
        work_dir,
        incident_phase.as_ref(),
        final_phase.as_ref(),
        xsect.as_ref().map(|source| &source.handoff),
    )?;
    let convolved_transition_amplitudes = validate_optional_rixs_incident_amplitude_convolution(
        work_dir,
        &solver_input,
        incident_transition_setup.as_ref(),
        final_phase.as_ref(),
        final_phase_shifts.as_ref(),
        xsect.as_ref().map(|source| &source.handoff),
        initial_transition_amplitudes.as_ref(),
        wave_numbers.as_ref(),
    )?;
    let raw_cross_section = validate_optional_rixs_raw_cross_section(
        work_dir,
        final_phase.as_ref(),
        incident_transition_setup.as_ref(),
        final_phase_shifts.as_ref(),
        second_green.as_ref(),
        convolved_transition_amplitudes.as_ref(),
    )?;
    let pole_normalization = read_optional_rixs_poles(work_dir, &solver_input)?;
    let self_energy = validate_optional_rixs_self_energy(
        work_dir,
        &solver_input,
        xsect.as_ref().map(|source| &source.handoff),
        raw_cross_section.as_ref(),
        pole_normalization.as_ref(),
    )?;
    let post_raw_spectrum = validate_optional_rixs_post_raw_spectrum(
        work_dir,
        &solver_input,
        xsect.as_ref().map(|source| &source.handoff),
        raw_cross_section.as_ref(),
        pole_normalization.as_ref(),
        self_energy.as_ref(),
    )?;
    let standard_outputs = build_optional_rixs_standard_outputs(
        work_dir,
        xsect.as_ref().map(|source| &source.handoff),
        post_raw_spectrum.as_ref(),
    )?;
    let satellite_outputs = build_optional_rixs_satellite_outputs(
        work_dir,
        &solver_input,
        xsect.as_ref().map(|source| &source.handoff),
        post_raw_spectrum.as_ref(),
        pole_normalization.as_ref(),
    )?;

    Ok(RixsSolverHandoffBundle {
        global,
        incident_phase,
        final_phase,
        incident_transition_setup,
        final_transition_setup,
        incident_phase_shifts,
        final_phase_shifts,
        incident_radial: first_radial,
        final_radial: second_radial,
        potential_difference,
        incident_green: first_green,
        final_green: second_green,
        xsect,
        radial_overlaps,
        initial_transition_amplitudes,
        wave_numbers,
        convolved_transition_amplitudes,
        raw_cross_section,
        pole_normalization,
        self_energy,
        post_raw_spectrum,
        standard_outputs,
        satellite_outputs,
    })
}

fn resolve_rixs_edge_pair<T: Clone>(
    first_name: &str,
    first: Option<T>,
    second_name: &str,
    second: Option<T>,
    shared: Option<T>,
) -> Result<(Option<T>, Option<T>)> {
    match (first, second) {
        (Some(first), Some(second)) => Ok((Some(first), Some(second))),
        (Some(_), None) => bail!(
            "incomplete RIXS two-edge handoff: {first_name} is present but {second_name} is missing"
        ),
        (None, Some(_)) => bail!(
            "incomplete RIXS two-edge handoff: {second_name} is present but {first_name} is missing"
        ),
        (None, None) => Ok((shared.clone(), shared)),
    }
}

fn validate_declared_rixs_solver_handoffs(work_dir: &Path, input: &RixsInput) -> Result<()> {
    if has_declared_rixs_solver_handoff_files(work_dir) {
        build_optional_rixs_solver_handoffs(work_dir, input)?;
    }
    Ok(())
}

#[derive(Debug)]
struct RixsSolverHandoffBundle {
    global: Option<GlobalInput>,
    incident_phase: Option<PhaseBinRixsHandoff>,
    final_phase: Option<PhaseBinRixsHandoff>,
    incident_transition_setup: Option<RixsTransitionMatrixSetup>,
    final_transition_setup: Option<RixsTransitionMatrixSetup>,
    incident_phase_shifts: Option<Array2<Complex>>,
    final_phase_shifts: Option<Array2<Complex>>,
    incident_radial: Option<RixsRlHandoffSource>,
    final_radial: Option<RixsRlHandoffSource>,
    potential_difference: Option<RixsPotentialDifferenceHandoffSource>,
    incident_green: Option<RixsGreenHandoffSource>,
    final_green: Option<RixsGreenHandoffSource>,
    xsect: Option<RixsXsectHandoffSource>,
    radial_overlaps: Option<Array3<f64>>,
    initial_transition_amplitudes: Option<Array4<Complex>>,
    wave_numbers: Option<RixsWaveNumbers>,
    convolved_transition_amplitudes: Option<Array4<Complex>>,
    raw_cross_section: Option<Array3<f64>>,
    pole_normalization: Option<RixsPoleNormalization>,
    self_energy: Option<Array1<Complex>>,
    post_raw_spectrum: Option<RixsPostRawSpectrum>,
    standard_outputs: Option<RixsDatFromFinalSpectrum>,
    satellite_outputs: Option<RixsDatFromFinalSpectrum>,
}

impl RixsSolverHandoffBundle {
    fn has_raw_solver_inputs(&self) -> bool {
        self.global.is_some()
            && self.incident_phase.is_some()
            && self.final_phase.is_some()
            && self.incident_transition_setup.is_some()
            && self.final_transition_setup.is_some()
            && self.incident_phase_shifts.is_some()
            && self.final_phase_shifts.is_some()
            && self.incident_radial.is_some()
            && self.final_radial.is_some()
            && self.potential_difference.is_some()
            && self.incident_green.is_some()
            && self.final_green.is_some()
            && self.xsect.is_some()
            && self.radial_overlaps.is_some()
            && self.initial_transition_amplitudes.is_some()
            && self.wave_numbers.is_some()
            && self.convolved_transition_amplitudes.is_some()
            && self.raw_cross_section.is_some()
            && self.pole_normalization.is_some()
            && self.self_energy.is_some()
            && self.post_raw_spectrum.is_some()
            && self.standard_outputs.is_some()
    }
}

fn recover_shared_wscrn_handoff_if_needed(work_dir: &Path) -> Result<()> {
    if work_dir.join("wscrn_1.dat").is_file() {
        return Ok(());
    }

    let shared_path = work_dir.join("wscrn.dat");
    if shared_path.is_file()
        && let Ok(cached) = read_wscrn_dat(&shared_path)
    {
        if let Ok(Some(generated)) = screen::generated_wscrn_from_vtot_and_apot_in_dir(work_dir)
            && !rixs_wscrn_matches_source_wscrn(&cached, &generated)
        {
            write_wscrn_dat(&shared_path, &generated)
                .with_context(|| format!("failed to write {}", shared_path.display()))?;
        }
        return Ok(());
    }

    screen::recover_wscrn_from_vtot_and_apot_in_dir(work_dir)?;
    Ok(())
}

fn rixs_wscrn_matches_source_wscrn(cached: &WscrnDatData, source: &WscrnDatData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-8;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    real_slices_match(
        &cached.radius_bohr,
        &source.radius_bohr,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.screened_potential,
        &source.screened_potential,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.core_hole_potential,
        &source.core_hole_potential,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
}

#[derive(Debug, Clone)]
struct RixsRlHandoffSource {
    name: &'static str,
    handoff: XsphRlDatRixsHandoff,
}

fn read_optional_rixs_phase_handoff(
    work_dir: &Path,
    name: &str,
) -> Result<Option<PhaseBinRixsHandoff>> {
    let path = work_dir.join(name);
    if !path.is_file() {
        return Ok(None);
    }

    let phase =
        read_phase_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let handoff = phase_bin_rixs_handoff_from_phase_bin(&phase)
        .with_context(|| format!("failed to build RIXS phase handoff from {}", path.display()))?;
    Ok(Some(handoff))
}

fn read_optional_rixs_global_input(work_dir: &Path) -> Result<Option<GlobalInput>> {
    let path = work_dir.join("global.inp");
    if !path.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let global = GlobalInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(global))
}

fn build_optional_rixs_transition_setup(
    work_dir: &Path,
    global: Option<&GlobalInput>,
    phase: Option<&PhaseBinRixsHandoff>,
) -> Result<Option<RixsTransitionMatrixSetup>> {
    let (Some(global), Some(phase)) = (global, phase) else {
        return Ok(None);
    };

    let setup =
        phase_bin_rixs_transition_setup_from_handoffs(global, phase).with_context(|| {
            format!(
                "failed to build RIXS transition setup from {} and phase handoff",
                work_dir.join("global.inp").display()
            )
        })?;
    Ok(Some(setup))
}

fn validate_optional_rixs_transition_phase_shifts(
    work_dir: &Path,
    phase: Option<&PhaseBinRixsHandoff>,
    setup: Option<&RixsTransitionMatrixSetup>,
) -> Result<Option<Array2<Complex>>> {
    let (Some(phase), Some(setup)) = (phase, setup) else {
        return Ok(None);
    };

    let phase_shifts = phase_bin_rixs_transition_phase_shifts_from_handoff(phase, setup)
        .with_context(|| {
            format!(
                "failed to select RIXS transition phase shifts from phase handoff in {}",
                work_dir.display()
            )
        })?;
    Ok(Some(phase_shifts))
}

fn validate_optional_rixs_transition_setup_compatibility(
    incident: Option<&RixsTransitionMatrixSetup>,
    final_setup: Option<&RixsTransitionMatrixSetup>,
) -> Result<()> {
    let (Some(incident), Some(final_setup)) = (incident, final_setup) else {
        return Ok(());
    };

    ensure!(
        incident.transition_angular_momenta == final_setup.transition_angular_momenta,
        "RIXS incident/final transition angular-momentum labels do not match"
    );
    ensure!(
        incident.transition_kappas == final_setup.transition_kappas,
        "RIXS incident/final transition kappa labels do not match"
    );
    Ok(())
}

fn read_optional_rixs_rl_handoff(
    work_dir: &Path,
    name: &'static str,
) -> Result<Option<RixsRlHandoffSource>> {
    let path = work_dir.join(name);
    if !path.is_file() {
        return Ok(None);
    }

    let data =
        read_xsph_rl_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let handoff = xsph_rl_dat_rixs_handoff(&data).with_context(|| {
        format!(
            "failed to build RIXS rl.dat handoff from {}",
            path.display()
        )
    })?;
    Ok(Some(RixsRlHandoffSource { name, handoff }))
}

fn validate_optional_rixs_rl_handoff_against_phase(
    work_dir: &Path,
    radial: Option<&RixsRlHandoffSource>,
    phase: Option<&PhaseBinRixsHandoff>,
) -> Result<()> {
    let (Some(radial), Some(phase)) = (radial, phase) else {
        return Ok(());
    };
    let path = work_dir.join(radial.name);
    ensure!(
        radial.handoff.energy_count == phase.main_energy_count
            || radial.handoff.energy_count == phase.energy_count,
        "{} energy count {} does not match phase handoff ne1 {} or ne {}",
        path.display(),
        radial.handoff.energy_count,
        phase.main_energy_count,
        phase.energy_count
    );
    ensure!(
        radial.handoff.angular_count >= phase.max_angular_limit_plus_one,
        "{} angular count {} is smaller than phase handoff lmaxp1 {}",
        path.display(),
        radial.handoff.angular_count,
        phase.max_angular_limit_plus_one
    );
    Ok(())
}

fn validate_optional_rixs_radial_overlaps(
    work_dir: &Path,
    input: &RixsInput,
    transition_setup: Option<&RixsTransitionMatrixSetup>,
    first_radial: Option<&XsphRlDatRixsHandoff>,
    second_radial: Option<&XsphRlDatRixsHandoff>,
    potential: Option<&RixsPotentialDifferenceHandoffSource>,
    xsect: Option<&XsectDatRixsHandoff>,
) -> Result<Option<Array3<f64>>> {
    let (
        Some(transition_setup),
        Some(first_radial),
        Some(second_radial),
        Some(potential),
        Some(xsect),
    ) = (
        transition_setup,
        first_radial,
        second_radial,
        potential,
        xsect,
    )
    else {
        return Ok(None);
    };

    let first_radial_count = first_radial.table.radial_functions.dim().0;
    let second_radial_count = second_radial.table.radial_functions.dim().0;
    ensure!(
        first_radial_count == second_radial_count,
        "RIXS radial overlap row count mismatch: first rl.dat has {}, second rl.dat has {}",
        first_radial_count,
        second_radial_count
    );
    let radial_count = first_radial_count;
    ensure!(
        potential.potential_difference.len() >= radial_count,
        "{} DeltaV row count {} is smaller than radial function row count {}",
        work_dir.join(potential.name).display(),
        potential.potential_difference.len(),
        radial_count
    );
    ensure!(
        potential.radii.len() >= radial_count,
        "{} radius row count {} is smaller than radial function row count {}",
        work_dir.join(potential.name).display(),
        potential.radii.len(),
        radial_count
    );

    let radii = potential
        .radii
        .iter()
        .take(radial_count)
        .copied()
        .collect::<Array1<_>>();
    let potential_difference = potential
        .potential_difference
        .iter()
        .take(radial_count)
        .copied()
        .collect::<Array1<_>>();
    let muffin_tin_radius = *radii
        .last()
        .with_context(|| "RIXS radial overlap requires at least one radial row")?;
    let solver_energy_count = xsect.energy_count();
    ensure_solver_energy_count(
        "incident rl.dat",
        first_radial.energy_count,
        solver_energy_count,
    )?;
    ensure_solver_energy_count(
        "final rl.dat",
        second_radial.energy_count,
        solver_energy_count,
    )?;

    let overlaps = rixs_radial_transition_overlaps(RixsRadialOverlapInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        radii: radii.view(),
        initial_radial_functions: first_radial
            .table
            .radial_functions
            .slice_axis(Axis(2), Slice::from(..solver_energy_count)),
        final_radial_functions: second_radial
            .table
            .radial_functions
            .slice_axis(Axis(2), Slice::from(..solver_energy_count)),
        potential_difference: potential_difference.view(),
        transition_angular_momenta: &transition_setup.transition_angular_momenta,
        fermi_level: input.xmu,
        log_step: first_radial.log_step,
        muffin_tin_radius,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS radial DeltaV overlaps from {}, rl handoffs, and xsect handoff",
            work_dir.join(potential.name).display()
        )
    })?;
    Ok(Some(overlaps))
}

#[allow(clippy::too_many_arguments)]
fn validate_optional_rixs_initial_transition_amplitudes(
    work_dir: &Path,
    input: &RixsInput,
    transition_setup: Option<&RixsTransitionMatrixSetup>,
    phase: Option<&PhaseBinRixsHandoff>,
    phase_shifts: Option<&Array2<Complex>>,
    green: Option<&RixsGreenHandoffSource>,
    xsect: Option<&XsectDatRixsHandoff>,
    radial_overlaps: Option<&Array3<f64>>,
) -> Result<Option<Array4<Complex>>> {
    let (
        Some(transition_setup),
        Some(phase),
        Some(phase_shifts),
        Some(green),
        Some(xsect),
        Some(radial_overlaps),
    ) = (
        transition_setup,
        phase,
        phase_shifts,
        green,
        xsect,
        radial_overlaps,
    )
    else {
        return Ok(None);
    };
    let solver_energy_count = xsect.energy_count();
    ensure_solver_energy_count(
        "incident phase transition moments",
        phase.transition_moments.dim().0,
        solver_energy_count,
    )?;
    ensure_solver_energy_count(
        "incident phase shifts",
        phase_shifts.dim().0,
        solver_energy_count,
    )?;
    ensure_solver_energy_count(green.name, green.handoff.energy_count, solver_energy_count)?;

    let amplitudes = rixs_initial_transition_amplitudes(RixsInitialAmplitudeInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        radial_overlaps: radial_overlaps.view(),
        incident_transition_moments: phase
            .transition_moments
            .slice_axis(Axis(0), Slice::from(..solver_energy_count)),
        incident_phase_shifts: phase_shifts
            .slice_axis(Axis(0), Slice::from(..solver_energy_count)),
        incident_green: green
            .handoff
            .green
            .slice_axis(Axis(2), Slice::from(..solver_energy_count)),
        normalization: xsect.normalization.view(),
        transition_angular_momenta: &transition_setup.transition_angular_momenta,
        fermi_level: input.xmu,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS initial TLb amplitudes from phase, {}, xsect, and radial overlaps",
            work_dir.join(green.name).display()
        )
    })?;
    Ok(Some(amplitudes))
}

fn validate_optional_rixs_wave_numbers(
    work_dir: &Path,
    incident_phase: Option<&PhaseBinRixsHandoff>,
    final_phase: Option<&PhaseBinRixsHandoff>,
    xsect: Option<&XsectDatRixsHandoff>,
) -> Result<Option<RixsWaveNumbers>> {
    let (Some(incident_phase), Some(final_phase), Some(xsect)) =
        (incident_phase, final_phase, xsect)
    else {
        return Ok(None);
    };

    let incident_reference = rixs_reference_energy_for_last_spin(incident_phase)
        .with_context(|| "failed to select incident RIXS reference energy")?;
    let final_reference = rixs_reference_energy_for_last_spin(final_phase)
        .with_context(|| "failed to select final RIXS reference energy")?;
    let wave_numbers = rixs_wave_numbers(RixsWaveNumberInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        incident_reference_energy: incident_reference,
        final_reference_energy: final_reference,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS wave-number handoff from phase handoffs in {}",
            work_dir.display()
        )
    })?;
    Ok(Some(wave_numbers))
}

fn rixs_reference_energy_for_last_spin(phase: &PhaseBinRixsHandoff) -> Result<Complex> {
    let spin = phase
        .spin_count
        .checked_sub(1)
        .with_context(|| "RIXS phase handoff has no spin channels")?;
    ensure!(
        !phase.reference_energy.is_empty(),
        "RIXS phase handoff has no reference-energy rows"
    );
    ensure!(
        phase.reference_energy.dim().1 > spin,
        "RIXS phase handoff reference-energy spin columns {} do not cover spin index {}",
        phase.reference_energy.dim().1,
        spin
    );
    Ok(phase.reference_energy[(0, spin)])
}

#[allow(clippy::too_many_arguments)]
fn validate_optional_rixs_incident_amplitude_convolution(
    work_dir: &Path,
    input: &RixsInput,
    transition_setup: Option<&RixsTransitionMatrixSetup>,
    final_phase: Option<&PhaseBinRixsHandoff>,
    final_phase_shifts: Option<&Array2<Complex>>,
    xsect: Option<&XsectDatRixsHandoff>,
    initial_transition_amplitudes: Option<&Array4<Complex>>,
    wave_numbers: Option<&RixsWaveNumbers>,
) -> Result<Option<Array4<Complex>>> {
    let (
        Some(transition_setup),
        Some(final_phase),
        Some(final_phase_shifts),
        Some(xsect),
        Some(initial_transition_amplitudes),
        Some(wave_numbers),
    ) = (
        transition_setup,
        final_phase,
        final_phase_shifts,
        xsect,
        initial_transition_amplitudes,
        wave_numbers,
    )
    else {
        return Ok(None);
    };
    let solver_energy_count = xsect.energy_count();
    ensure_solver_energy_count(
        "final phase transition moments",
        final_phase.transition_moments.dim().0,
        solver_energy_count,
    )?;
    ensure_solver_energy_count(
        "final phase shifts",
        final_phase_shifts.dim().0,
        solver_energy_count,
    )?;

    let convolved = rixs_incident_amplitude_convolution(RixsIncidentAmplitudeConvolutionInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        transition_amplitudes: initial_transition_amplitudes.view(),
        final_transition_moments: final_phase
            .transition_moments
            .slice_axis(Axis(0), Slice::from(..solver_energy_count)),
        final_phase_shifts: final_phase_shifts
            .slice_axis(Axis(0), Slice::from(..solver_energy_count)),
        final_wave_numbers: wave_numbers.final_wave_numbers.view(),
        normalization: xsect.normalization.view(),
        b_matrix_diagonal: transition_setup.b_matrix_diagonal.view(),
        transition_angular_momenta: &transition_setup.transition_angular_momenta,
        fermi_level: input.xmu,
        core_width: xsect.core_hole_width_hartree,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS incident-amplitude convolution from phase, xsect, and wave-number handoffs in {}",
            work_dir.display()
        )
    })?;
    Ok(Some(convolved))
}

fn validate_optional_rixs_raw_cross_section(
    work_dir: &Path,
    final_phase: Option<&PhaseBinRixsHandoff>,
    transition_setup: Option<&RixsTransitionMatrixSetup>,
    final_phase_shifts: Option<&Array2<Complex>>,
    final_green: Option<&RixsGreenHandoffSource>,
    transition_amplitudes: Option<&Array4<Complex>>,
) -> Result<Option<Array3<f64>>> {
    let (
        Some(final_phase),
        Some(transition_setup),
        Some(final_phase_shifts),
        Some(final_green),
        Some(transition_amplitudes),
    ) = (
        final_phase,
        transition_setup,
        final_phase_shifts,
        final_green,
        transition_amplitudes,
    )
    else {
        return Ok(None);
    };
    let solver_energy_count = transition_amplitudes.dim().0;
    ensure_solver_energy_count(
        "final phase shifts",
        final_phase_shifts.dim().0,
        solver_energy_count,
    )?;
    ensure_solver_energy_count(
        final_green.name,
        final_green.handoff.energy_count,
        solver_energy_count,
    )?;

    let raw_cross_section = rixs_raw_cross_section(RixsRawCrossSectionInput {
        transition_amplitudes: transition_amplitudes.view(),
        final_green: final_green
            .handoff
            .green
            .slice_axis(Axis(2), Slice::from(..solver_energy_count)),
        final_phase_shifts: final_phase_shifts
            .slice_axis(Axis(0), Slice::from(..solver_energy_count)),
        transition_angular_momenta: &transition_setup.transition_angular_momenta,
        spin_channel_count: final_phase.spin_count,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS raw cross-section from {}, final phase shifts, and convolved amplitudes",
            work_dir.join(final_green.name).display()
        )
    })?;
    Ok(Some(raw_cross_section))
}

fn ensure_solver_energy_count(name: &str, available: usize, required: usize) -> Result<()> {
    ensure!(
        available >= required,
        "RIXS {name} has {available} energy row(s), but xsect.ne1 requires {required}"
    );
    Ok(())
}

fn validate_optional_rixs_post_raw_spectrum(
    work_dir: &Path,
    input: &RixsInput,
    xsect: Option<&XsectDatRixsHandoff>,
    raw_cross_section: Option<&Array3<f64>>,
    poles: Option<&RixsPoleNormalization>,
    self_energy: Option<&Array1<Complex>>,
) -> Result<Option<RixsPostRawSpectrum>> {
    let (Some(xsect), Some(raw_cross_section), Some(poles), Some(self_energy)) =
        (xsect, raw_cross_section, poles, self_energy)
    else {
        return Ok(None);
    };

    let post_raw = rixs_post_raw_spectrum(RixsPostRawSpectrumInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        raw_cross_section: raw_cross_section.view(),
        self_energy: self_energy.view(),
        fermi_level: input.xmu,
        core_width: poles.core_width,
        incident_width: input.broadening.gam_exp_1,
        final_width_base: input.broadening.gam_exp_2,
        edge_splits: poles.edge_splits.view(),
        edge_widths: poles.edge_widths.view(),
        edge_amplitudes: poles.edge_amplitudes.view(),
        incident_window: (input.energy_window.emin_i, input.energy_window.emax_i),
        final_window: (input.energy_window.emin_f, input.energy_window.emax_f),
        incident_edge: poles.incident_edge,
        final_edge: poles.final_edge,
        hartree_ev: FEFF_HARTREE_EV,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS post-raw spectra from {}, raw cross-section, and rixs.inp",
            work_dir.join("edges.dat").display()
        )
    })?;
    Ok(Some(post_raw))
}

fn read_optional_rixs_poles(
    work_dir: &Path,
    input: &RixsInput,
) -> Result<Option<RixsPoleNormalization>> {
    if !input.switches.read_poles {
        let default_poles = rixs_default_pole_normalization(input.broadening.gam_ch)
            .with_context(|| "failed to build default RIXS pole data for ReadPoles = false")?;
        return Ok(Some(default_poles));
    }
    let path = work_dir.join("edges.dat");
    if !path.is_file() {
        return Ok(None);
    }

    let edges =
        read_edges_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let pole_energies = edges
        .rows
        .iter()
        .map(|row| row.chemical_potential)
        .collect::<Array1<_>>();
    let pole_amplitudes = edges
        .rows
        .iter()
        .map(|row| row.matrix_element)
        .collect::<Array1<_>>();
    let pole_widths = edges
        .rows
        .iter()
        .map(|row| row.core_hole_width)
        .collect::<Array1<_>>();
    let normalized = rixs_normalize_poles(RixsPoleNormalizationInput {
        pole_energies: pole_energies.view(),
        pole_amplitudes: pole_amplitudes.view(),
        pole_widths: pole_widths.view(),
        fermi_level: input.xmu,
    })
    .with_context(|| format!("failed to normalize RIXS poles from {}", path.display()))?;
    Ok(Some(normalized))
}

fn validate_optional_rixs_self_energy(
    work_dir: &Path,
    input: &RixsInput,
    xsect: Option<&XsectDatRixsHandoff>,
    raw_cross_section: Option<&Array3<f64>>,
    poles: Option<&RixsPoleNormalization>,
) -> Result<Option<Array1<Complex>>> {
    let Some(xsect) = xsect else {
        return Ok(None);
    };
    if raw_cross_section.is_none() || poles.is_none() {
        return Ok(None);
    }
    if !input.switches.read_sigma {
        return Ok(Some(Array1::zeros(xsect.energy_count())));
    }

    let mpse_path = work_dir.join("mpse.dat");
    let mpse = read_or_generate_rixs_mpse_handoff(work_dir)?;
    let self_energy = rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        mpse_energy_ev: mpse.energy_ev.view(),
        mpse_self_energy_ev: mpse.self_energy.view(),
        fermi_level: input.xmu,
        hartree_ev: FEFF_HARTREE_EV,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS self-energy grid from {} or XSPH source handoff",
            mpse_path.display()
        )
    })?;
    Ok(Some(self_energy))
}

fn read_or_generate_rixs_mpse_handoff(work_dir: &Path) -> Result<MpseDatData> {
    let path = work_dir.join("mpse.dat");
    match read_mpse_dat(&path) {
        Ok(mpse) => Ok(select_source_rixs_mpse_if_stale(work_dir, mpse)),
        Err(error) => {
            if let Some(mpse) = xsph::generate_mpse_dat_from_source_handoff(work_dir)
                .context("failed to generate RIXS self-energy mpse.dat from XSPH source handoff")?
            {
                return Ok(mpse);
            }
            Err(error).with_context(|| format!("failed to read {}", path.display()))
        }
    }
}

fn select_source_rixs_mpse_if_stale(work_dir: &Path, cached: MpseDatData) -> MpseDatData {
    match xsph::generate_mpse_dat_from_source_handoff(work_dir) {
        Ok(Some(source)) if !rixs_mpse_matches_source_mpse(&cached, &source) => source,
        _ => cached,
    }
}

fn rixs_mpse_matches_source_mpse(cached: &MpseDatData, source: &MpseDatData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-8;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    real_slices_match(
        &cached.energy_ev,
        &source.energy_ev,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && complex_slices_match(
        &cached.self_energy,
        &source.self_energy,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
}

fn real_slices_match(
    cached: &Array1<f64>,
    source: &Array1<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.len() == source.len()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_slices_match(
    cached: &Array1<Complex>,
    source: &Array1<Complex>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.len() == source.len()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            scalar_matches(cached.re, source.re, absolute_tolerance, relative_tolerance)
                && scalar_matches(cached.im, source.im, absolute_tolerance, relative_tolerance)
        })
}

fn scalar_matches(
    cached: f64,
    source: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    (cached - source).abs() <= absolute_tolerance + relative_tolerance * source.abs()
}

fn build_optional_rixs_standard_outputs(
    work_dir: &Path,
    xsect: Option<&XsectDatRixsHandoff>,
    post_raw: Option<&RixsPostRawSpectrum>,
) -> Result<Option<RixsDatFromFinalSpectrum>> {
    let (Some(xsect), Some(post_raw)) = (xsect, post_raw) else {
        return Ok(None);
    };

    let outputs = rixs_dat_from_final_spectrum(&post_raw.spectrum, xsect.energy_count())
        .with_context(|| {
            format!(
                "failed to build RIXS standard output payloads from source-backed spectrum in {}",
                work_dir.display()
            )
        })?;
    Ok(Some(outputs))
}

fn build_optional_rixs_satellite_outputs(
    work_dir: &Path,
    input: &RixsInput,
    xsect: Option<&XsectDatRixsHandoff>,
    post_raw: Option<&RixsPostRawSpectrum>,
    poles: Option<&RixsPoleNormalization>,
) -> Result<Option<RixsDatFromFinalSpectrum>> {
    if !input.switches.mbconv {
        return Ok(None);
    }
    let xes_path = work_dir.join("XES").join("xmu.dat");
    if !xes_path.is_file() {
        return Ok(None);
    }
    let (Some(xsect), Some(post_raw), Some(poles)) = (xsect, post_raw, poles) else {
        return Ok(None);
    };

    let xes = read_xmu_dat(&xes_path)
        .with_context(|| format!("failed to read {}", xes_path.display()))?;
    let satellite = rixs_satellite_spectrum(RixsSatelliteSpectrumInput {
        relative_energies: xsect.relative_energies_hartree.view(),
        cross_section: post_raw.summed_cross_section.view(),
        xes_energy_ev: xes.relative_energy_ev.view(),
        xes_mu: xes.mu.view(),
        fermi_level: input.xmu,
        incident_window: (input.energy_window.emin_i, input.energy_window.emax_i),
        final_window: (input.energy_window.emin_f, input.energy_window.emax_f),
        incident_edge: poles.incident_edge,
        final_edge: poles.final_edge,
        hartree_ev: FEFF_HARTREE_EV,
    })
    .with_context(|| {
        format!(
            "failed to build RIXS MBConv satellite spectrum from {} and source-backed spectrum",
            xes_path.display()
        )
    })?;
    let outputs = rixs_dat_from_final_spectrum(&satellite.spectrum, xsect.energy_count())
        .with_context(|| {
            format!(
                "failed to build RIXS MBConv satellite output payloads from {}",
                xes_path.display()
            )
        })?;
    Ok(Some(outputs))
}

fn validate_optional_rixs_wscrn_handoffs(
    work_dir: &Path,
    first_radial: Option<&XsphRlDatRixsHandoff>,
    second_radial: Option<&XsphRlDatRixsHandoff>,
) -> Result<Option<RixsPotentialDifferenceHandoffSource>> {
    let explicit_first = read_optional_rixs_wscrn_handoff(work_dir, "wscrn_1.dat")?;
    let explicit_second = read_optional_rixs_wscrn_handoff(work_dir, "wscrn_2.dat")?;
    let shared = if explicit_first.is_none() && explicit_second.is_none() {
        read_optional_rixs_wscrn_handoff(work_dir, "wscrn.dat")?
    } else {
        None
    };
    let (first, second) = resolve_rixs_edge_pair(
        "wscrn_1.dat",
        explicit_first,
        "wscrn_2.dat",
        explicit_second,
        shared,
    )?;
    validate_optional_wscrn_against_radial(first.as_ref(), first_radial)?;
    validate_optional_wscrn_against_radial(second.as_ref(), second_radial)?;

    let Some(first) = first.as_ref() else {
        return Ok(None);
    };
    if let Some(second) = second.as_ref() {
        ensure!(
            first.data.row_count() == second.data.row_count(),
            "{} row count {} does not match {} row count {}",
            work_dir.join(second.name).display(),
            second.data.row_count(),
            work_dir.join(first.name).display(),
            first.data.row_count()
        );
        ensure_matching_wscrn_radius(
            &first.data,
            &second.data,
            work_dir.join(second.name),
            work_dir.join(first.name),
        )?;
    }

    let potential_difference = rixs_core_hole_potential_difference(RixsCoreHolePotentialInput {
        incident_screened_core_hole: first.data.screened_potential.view(),
        final_screened_core_hole: second
            .as_ref()
            .map(|source| source.data.screened_potential.view()),
    })
    .with_context(|| {
        format!(
            "failed to build RIXS screened-core handoff from {}",
            work_dir.join(first.name).display()
        )
    })?;
    Ok(Some(RixsPotentialDifferenceHandoffSource {
        name: first.name,
        radii: first.data.radius_bohr.clone(),
        potential_difference,
    }))
}

#[derive(Clone)]
struct RixsWscrnHandoffSource {
    name: &'static str,
    data: WscrnDatData,
}

#[derive(Debug, Clone)]
struct RixsPotentialDifferenceHandoffSource {
    name: &'static str,
    radii: Array1<f64>,
    potential_difference: Array1<f64>,
}

fn read_optional_rixs_wscrn_handoff(
    work_dir: &Path,
    name: &'static str,
) -> Result<Option<RixsWscrnHandoffSource>> {
    let path = work_dir.join(name);
    if !path.is_file() {
        return Ok(None);
    }

    let data =
        read_wscrn_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(Some(RixsWscrnHandoffSource { name, data }))
}

fn validate_optional_wscrn_against_radial(
    data: Option<&RixsWscrnHandoffSource>,
    radial: Option<&XsphRlDatRixsHandoff>,
) -> Result<()> {
    let (Some(data), Some(radial)) = (data, radial) else {
        return Ok(());
    };
    let radial_count = radial.table.radial_functions.dim().0;
    ensure!(
        data.data.row_count() >= radial_count,
        "{} row count {} is smaller than rl.dat radial count {}",
        data.name,
        data.data.row_count(),
        radial_count
    );
    Ok(())
}

fn ensure_matching_wscrn_radius(
    first: &WscrnDatData,
    second: &WscrnDatData,
    second_path: PathBuf,
    first_path: PathBuf,
) -> Result<()> {
    for (row, (&left, &right)) in first
        .radius_bohr
        .iter()
        .zip(second.radius_bohr.iter())
        .enumerate()
    {
        let tolerance = 1.0e-8 * left.abs().max(right.abs()).max(1.0);
        ensure!(
            (left - right).abs() <= tolerance,
            "{} radius row {} {} does not match {} radius {}",
            second_path.display(),
            row + 1,
            right,
            first_path.display(),
            left
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct RixsGreenHandoffSource {
    name: &'static str,
    handoff: GgDatRixsHandoff,
}

fn read_optional_rixs_green_handoff(
    work_dir: &Path,
    name: &'static str,
) -> Result<Option<RixsGreenHandoffSource>> {
    let path = work_dir.join(name);
    if !path.is_file() {
        return Ok(None);
    }

    let data = read_gg_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let handoff = gg_dat_rixs_handoff(&data).with_context(|| {
        format!(
            "failed to build RIXS Green-function handoff from {}",
            path.display()
        )
    })?;
    Ok(Some(RixsGreenHandoffSource { name, handoff }))
}

fn validate_optional_rixs_green_handoff_against_phase(
    work_dir: &Path,
    green: Option<&RixsGreenHandoffSource>,
    phase: Option<&PhaseBinRixsHandoff>,
) -> Result<()> {
    let Some(green) = green else {
        return Ok(());
    };

    let path = work_dir.join(green.name);
    let Some(phase) = phase else {
        return Ok(());
    };
    let section_count = green.handoff.energy_count;
    ensure!(
        section_count == phase.main_energy_count || section_count == phase.energy_count,
        "{} section count {} does not match phase handoff ne1 {} or ne {}",
        path.display(),
        section_count,
        phase.main_energy_count,
        phase.energy_count
    );
    ensure!(
        green.handoff.angular_count >= phase.max_angular_limit_plus_one,
        "{} matrix order {} is smaller than phase handoff lmaxp1 {}",
        path.display(),
        green.handoff.angular_count,
        phase.max_angular_limit_plus_one
    );
    Ok(())
}

fn validate_optional_rixs_xsect_handoff(
    work_dir: &Path,
    phase: Option<&PhaseBinRixsHandoff>,
    radial: Option<&XsphRlDatRixsHandoff>,
) -> Result<Option<RixsXsectHandoffSource>> {
    let xsect = match read_optional_rixs_xsect_handoff(work_dir, "xsect_2.dat")? {
        Some(xsect) => Some(xsect),
        None => read_optional_rixs_xsect_handoff(work_dir, "xsect.dat")?,
    };
    let Some(mut xsect) = xsect else {
        return Ok(None);
    };
    let path = work_dir.join(xsect.name);

    if let Some(phase) = phase {
        project_rixs_solver_energy_grid_from_phase(&path, &mut xsect.handoff, phase)?;
    }
    if let Some(radial) = radial {
        ensure!(
            radial.energy_count >= xsect.handoff.energy_count(),
            "{} ne1 {} exceeds rl.dat energy count {}",
            path.display(),
            xsect.handoff.energy_count(),
            radial.energy_count
        );
    }
    Ok(Some(xsect))
}

/// FEFF's RIXS solver assigns `rem(1:ne1)` from binary `phase.bin`, not from
/// the rounded energy column rendered in `xsect.dat`. Preserve that handoff:
/// a text-rounding error at `xmu` otherwise changes a physical threshold row
/// from included (`rem == xmu`) to excluded (`rem < xmu`).
fn project_rixs_solver_energy_grid_from_phase(
    path: &Path,
    xsect: &mut XsectDatRixsHandoff,
    phase: &PhaseBinRixsHandoff,
) -> Result<()> {
    ensure!(
        xsect.main_energy_count == phase.main_energy_count,
        "{} ne1 {} does not match phase handoff ne1 {}",
        path.display(),
        xsect.main_energy_count,
        phase.main_energy_count
    );
    let energy_grid = phase
        .energy_grid
        .iter()
        .take(phase.main_energy_count)
        .copied()
        .collect::<Array1<_>>();
    ensure!(
        energy_grid.len() == xsect.energy_count(),
        "{} phase energy count {} does not match xsect solver energy count {}",
        path.display(),
        energy_grid.len(),
        xsect.energy_count()
    );
    xsect.relative_energies_hartree = energy_grid.mapv(|energy| energy.re);
    xsect.energy_grid_hartree = energy_grid;
    Ok(())
}

#[derive(Debug, Clone)]
struct RixsXsectHandoffSource {
    name: &'static str,
    handoff: XsectDatRixsHandoff,
}

fn read_optional_rixs_xsect_handoff(
    work_dir: &Path,
    name: &'static str,
) -> Result<Option<RixsXsectHandoffSource>> {
    let path = work_dir.join(name);
    if !path.is_file() {
        return Ok(None);
    }

    let data =
        read_xsect_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let handoff = xsect_dat_rixs_handoff(&data).with_context(|| {
        format!(
            "failed to build RIXS xsect.dat handoff from {}",
            path.display()
        )
    })?;
    Ok(Some(RixsXsectHandoffSource { name, handoff }))
}

fn write_rixs_map_cache(path: &Path, data: &RixsMapData) -> Result<()> {
    write_rixs_map(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_rixs_line_cache(path: &Path, data: &RixsLineData) -> Result<()> {
    write_rixs_line(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_rixs_source_outputs(
    work_dir: &Path,
    bundle: &RixsSolverHandoffBundle,
) -> Result<Option<usize>> {
    let Some(outputs) = bundle.standard_outputs.as_ref() else {
        return Ok(None);
    };

    let mut written = write_rixs_source_outputs(work_dir, outputs, regular_final_output_names())?;
    if let Some(satellite) = bundle.satellite_outputs.as_ref() {
        written += write_rixs_source_outputs(work_dir, satellite, satellite_final_output_names())?;
    }
    let outputs = cached_output_paths(work_dir)?;
    let log_written =
        write_or_recover_module_log(&work_dir.join("logrixs.dat"), &outputs, written > 0)?;
    Ok(Some(written + log_written))
}

fn write_rixs_source_outputs(
    work_dir: &Path,
    outputs: &RixsDatFromFinalSpectrum,
    names: RixsFinalOutputNames,
) -> Result<usize> {
    write_rixs_map_cache(&work_dir.join(names.map), &outputs.rixs_et)?;
    write_rixs_line_cache(&work_dir.join(names.herfd), &outputs.herfd)?;
    write_rixs_line_cache(&work_dir.join(names.xas_ei), &outputs.xas_ei)?;
    write_rixs_line_cache(&work_dir.join(names.xas_ef), &outputs.xas_ef)?;
    write_rixs_map_cache(&work_dir.join(names.rixs_ee), &outputs.rixs_ee)?;
    Ok(5)
}

fn roundtrip_cached_output(output: &CachedOutputPath) -> Result<()> {
    match output.kind {
        CachedOutputKind::Map => {
            let data = read_rixs_map(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
            write_rixs_map_cache(&output.path, &data)
        }
        CachedOutputKind::Line => {
            let data = read_rixs_line(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
            write_rixs_line_cache(&output.path, &data)
        }
    }
}

fn prepare_cached_output(output: &CachedOutputPath) -> Result<()> {
    match output.kind {
        CachedOutputKind::Map => {
            read_rixs_map(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
        }
        CachedOutputKind::Line => {
            read_rixs_line(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
        }
    }
    Ok(())
}

fn prepare_cached_output_or_recoverable(
    work_dir: &Path,
    input: &RixsInput,
    output: &CachedOutputPath,
) -> Result<()> {
    match prepare_cached_output(output) {
        Ok(()) => Ok(()),
        Err(error) => {
            if cached_output_can_be_regenerated(work_dir, input, output)? {
                Ok(())
            } else {
                Err(error)
            }
        }
    }
}

fn cached_output_can_be_regenerated(
    work_dir: &Path,
    input: &RixsInput,
    output: &CachedOutputPath,
) -> Result<bool> {
    let Some(name) = output.path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };

    match name {
        "herfd.dat" => Ok(can_derive_diagonal_line(work_dir, "rixsET.dat")?
            || can_derive_final_outputs(work_dir, input, regular_final_output_names())?),
        "herfd-sat.dat" => Ok(can_derive_diagonal_line(work_dir, "rixsET-sat.dat")?
            || can_derive_final_outputs(work_dir, input, satellite_final_output_names())?),
        "xas0.dat" => can_derive_diagonal_line(work_dir, "rixs0.dat"),
        "xas1.dat" => can_derive_diagonal_line(work_dir, "rixs1.dat"),
        "xasEI.dat" | "xasEF.dat" | "rixsEE.dat" => {
            can_derive_final_outputs(work_dir, input, regular_final_output_names())
        }
        "xasEI-sat.dat" | "xasEF-sat.dat" | "rixsEE-sat.dat" => {
            can_derive_final_outputs(work_dir, input, satellite_final_output_names())
        }
        _ => Ok(false),
    }
}

fn can_generate_any_rixs_output_from_sources(work_dir: &Path, input: &RixsInput) -> Result<bool> {
    for names in [regular_final_output_names(), satellite_final_output_names()] {
        if !final_outputs_are_usable(work_dir, names)
            && can_derive_final_outputs(work_dir, input, names)?
        {
            return Ok(true);
        }
    }

    for (map_name, line_name) in [
        ("rixsET.dat", "herfd.dat"),
        ("rixsET-sat.dat", "herfd-sat.dat"),
        ("rixs0.dat", "xas0.dat"),
        ("rixs1.dat", "xas1.dat"),
    ] {
        let line_path = work_dir.join(line_name);
        if diagonal_line_output_needs_generation(&line_path)
            && can_derive_diagonal_line(work_dir, map_name)?
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn can_derive_diagonal_line(work_dir: &Path, map_name: &str) -> Result<bool> {
    let map_path = work_dir.join(map_name);
    if !map_path.is_file() {
        return Ok(false);
    }
    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    Ok(rixs_line_from_map_diagonal(&map).is_ok())
}

fn can_derive_final_outputs(
    work_dir: &Path,
    input: &RixsInput,
    names: RixsFinalOutputNames,
) -> Result<bool> {
    let map_path = work_dir.join(names.map);
    if !map_path.is_file() {
        return Ok(false);
    }
    let Some(edges) = read_skip_calc_edge_offsets(work_dir, input)? else {
        return Ok(false);
    };
    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    Ok(rixs_skip_calc_outputs_from_rixs_et_map(&map, input.energy_window, edges).is_ok())
}

fn prepare_missing_diagonal_line_outputs(work_dir: &Path) -> Result<()> {
    for (map_name, line_name) in [
        ("rixsET.dat", "herfd.dat"),
        ("rixsET-sat.dat", "herfd-sat.dat"),
        ("rixs0.dat", "xas0.dat"),
        ("rixs1.dat", "xas1.dat"),
    ] {
        prepare_missing_diagonal_line_output(work_dir, map_name, line_name)?;
    }
    Ok(())
}

fn prepare_missing_diagonal_line_output(
    work_dir: &Path,
    map_name: &str,
    line_name: &str,
) -> Result<()> {
    let map_path = work_dir.join(map_name);
    let line_path = work_dir.join(line_name);
    if !diagonal_line_output_needs_generation(&line_path) || !map_path.is_file() {
        return Ok(());
    }

    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    rixs_line_from_map_diagonal(&map).with_context(|| {
        format!(
            "failed to derive {} from {}",
            line_path.display(),
            map_path.display()
        )
    })?;
    Ok(())
}

fn write_missing_diagonal_line_outputs(work_dir: &Path) -> Result<usize> {
    let mut written = 0;
    for (map_name, line_name) in [
        ("rixsET.dat", "herfd.dat"),
        ("rixsET-sat.dat", "herfd-sat.dat"),
        ("rixs0.dat", "xas0.dat"),
        ("rixs1.dat", "xas1.dat"),
    ] {
        written += write_missing_diagonal_line_output(work_dir, map_name, line_name)?;
    }
    Ok(written)
}

fn write_missing_diagonal_line_output(
    work_dir: &Path,
    map_name: &str,
    line_name: &str,
) -> Result<usize> {
    let map_path = work_dir.join(map_name);
    let line_path = work_dir.join(line_name);
    if !diagonal_line_output_needs_generation(&line_path) || !map_path.is_file() {
        return Ok(0);
    }

    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    let line = rixs_line_from_map_diagonal(&map).with_context(|| {
        format!(
            "failed to derive {} from {}",
            line_path.display(),
            map_path.display()
        )
    })?;
    write_rixs_line_cache(&line_path, &line)?;
    Ok(1)
}

fn diagonal_line_output_needs_generation(path: &Path) -> bool {
    !path.is_file() || read_rixs_line(path).is_err()
}

fn write_missing_final_outputs_from_cached_rixs_maps(
    work_dir: &Path,
    input: &RixsInput,
) -> Result<usize> {
    let mut written = 0;
    written += write_missing_final_outputs_from_cached_map(
        work_dir,
        input,
        RixsFinalOutputNames {
            map: "rixsET.dat",
            herfd: "herfd.dat",
            xas_ei: "xasEI.dat",
            xas_ef: "xasEF.dat",
            rixs_ee: "rixsEE.dat",
        },
    )?;
    written += write_missing_final_outputs_from_cached_map(
        work_dir,
        input,
        RixsFinalOutputNames {
            map: "rixsET-sat.dat",
            herfd: "herfd-sat.dat",
            xas_ei: "xasEI-sat.dat",
            xas_ef: "xasEF-sat.dat",
            rixs_ee: "rixsEE-sat.dat",
        },
    )?;
    Ok(written)
}

#[derive(Debug, Clone, Copy)]
struct RixsFinalOutputNames {
    map: &'static str,
    herfd: &'static str,
    xas_ei: &'static str,
    xas_ef: &'static str,
    rixs_ee: &'static str,
}

fn regular_final_output_names() -> RixsFinalOutputNames {
    RixsFinalOutputNames {
        map: "rixsET.dat",
        herfd: "herfd.dat",
        xas_ei: "xasEI.dat",
        xas_ef: "xasEF.dat",
        rixs_ee: "rixsEE.dat",
    }
}

fn satellite_final_output_names() -> RixsFinalOutputNames {
    RixsFinalOutputNames {
        map: "rixsET-sat.dat",
        herfd: "herfd-sat.dat",
        xas_ei: "xasEI-sat.dat",
        xas_ef: "xasEF-sat.dat",
        rixs_ee: "rixsEE-sat.dat",
    }
}

fn prepare_missing_final_outputs_from_cached_rixs_maps(
    work_dir: &Path,
    input: &RixsInput,
) -> Result<()> {
    prepare_missing_final_outputs_from_cached_map(
        work_dir,
        input,
        RixsFinalOutputNames {
            map: "rixsET.dat",
            herfd: "herfd.dat",
            xas_ei: "xasEI.dat",
            xas_ef: "xasEF.dat",
            rixs_ee: "rixsEE.dat",
        },
    )?;
    prepare_missing_final_outputs_from_cached_map(
        work_dir,
        input,
        RixsFinalOutputNames {
            map: "rixsET-sat.dat",
            herfd: "herfd-sat.dat",
            xas_ei: "xasEI-sat.dat",
            xas_ef: "xasEF-sat.dat",
            rixs_ee: "rixsEE-sat.dat",
        },
    )
}

fn prepare_missing_final_outputs_from_cached_map(
    work_dir: &Path,
    input: &RixsInput,
    names: RixsFinalOutputNames,
) -> Result<()> {
    let map_path = work_dir.join(names.map);
    if !map_path.is_file() || final_outputs_are_usable(work_dir, names) {
        return Ok(());
    }

    let Some(edges) = read_skip_calc_edge_offsets(work_dir, input)? else {
        return Ok(());
    };
    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    rixs_skip_calc_outputs_from_rixs_et_map(&map, input.energy_window, edges).with_context(
        || {
            format!(
                "failed to derive RIXS final outputs from cached {}",
                names.map
            )
        },
    )?;
    Ok(())
}

fn write_missing_final_outputs_from_cached_map(
    work_dir: &Path,
    input: &RixsInput,
    names: RixsFinalOutputNames,
) -> Result<usize> {
    let map_path = work_dir.join(names.map);
    if !map_path.is_file() || final_outputs_are_usable(work_dir, names) {
        return Ok(0);
    }

    let Some(edges) = read_skip_calc_edge_offsets(work_dir, input)? else {
        return Ok(0);
    };
    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    let derived = rixs_skip_calc_outputs_from_rixs_et_map(&map, input.energy_window, edges)
        .with_context(|| {
            format!(
                "failed to derive RIXS final outputs from cached {}",
                names.map
            )
        })?;

    let mut written = 0;
    written += write_missing_final_line_output(work_dir, names.herfd, &derived.herfd)?;
    written += write_missing_final_line_output(work_dir, names.xas_ei, &derived.xas_ei)?;
    written += write_missing_final_line_output(work_dir, names.xas_ef, &derived.xas_ef)?;
    written += write_missing_final_map_output(work_dir, names.rixs_ee, &derived.rixs_ee)?;
    Ok(written)
}

fn final_outputs_are_usable(work_dir: &Path, names: RixsFinalOutputNames) -> bool {
    final_line_output_is_usable(&work_dir.join(names.herfd))
        && final_line_output_is_usable(&work_dir.join(names.xas_ei))
        && final_line_output_is_usable(&work_dir.join(names.xas_ef))
        && final_map_output_is_usable(&work_dir.join(names.rixs_ee))
}

fn source_final_outputs_are_usable(work_dir: &Path, names: RixsFinalOutputNames) -> bool {
    final_map_output_is_usable(&work_dir.join(names.map))
        && final_outputs_are_usable(work_dir, names)
}

fn final_line_output_is_usable(path: &Path) -> bool {
    path.is_file() && read_rixs_line(path).is_ok()
}

fn final_map_output_is_usable(path: &Path) -> bool {
    path.is_file() && read_rixs_map(path).is_ok()
}

fn write_missing_final_line_output(
    work_dir: &Path,
    name: &str,
    data: &RixsLineData,
) -> Result<usize> {
    let path = work_dir.join(name);
    if final_line_output_is_usable(&path) {
        return Ok(0);
    }
    write_rixs_line_cache(&path, data)?;
    Ok(1)
}

fn write_missing_final_map_output(
    work_dir: &Path,
    name: &str,
    data: &RixsMapData,
) -> Result<usize> {
    let path = work_dir.join(name);
    if final_map_output_is_usable(&path) {
        return Ok(0);
    }
    write_rixs_map_cache(&path, data)?;
    Ok(1)
}

fn run_skip_calc_from_cached_map(work_dir: &Path) -> Result<Vec<CachedOutputPath>> {
    let input = read_input(work_dir)?;
    let map_path = work_dir.join("rixsET.dat");
    if !map_path.is_file() {
        bail!("RIXS SkipCalc requires cached rixsET.dat");
    }

    let map = read_rixs_map(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    let edges = read_skip_calc_edge_offsets(work_dir, &input)?;
    write_rixs_map_cache(&map_path, &map)?;
    if let Some(edges) = edges {
        let derived = rixs_skip_calc_outputs_from_rixs_et_map(&map, input.energy_window, edges)
            .context("failed to build RIXS SkipCalc output tables")?;
        write_rixs_line_cache(&work_dir.join("herfd.dat"), &derived.herfd)?;
        write_rixs_line_cache(&work_dir.join("xasEI.dat"), &derived.xas_ei)?;
        write_rixs_line_cache(&work_dir.join("xasEF.dat"), &derived.xas_ef)?;
        write_rixs_map_cache(&work_dir.join("rixsEE.dat"), &derived.rixs_ee)?;
        if input.switches.mbconv
            && let Some(satellite) =
                skip_calc_satellite_outputs(work_dir, &map, input.energy_window, edges, input.xmu)?
        {
            write_rixs_map_cache(&work_dir.join("rixsET-sat.dat"), &satellite.rixs_et)?;
            write_rixs_line_cache(&work_dir.join("herfd-sat.dat"), &satellite.herfd)?;
            write_rixs_line_cache(&work_dir.join("xasEI-sat.dat"), &satellite.xas_ei)?;
            write_rixs_line_cache(&work_dir.join("xasEF-sat.dat"), &satellite.xas_ef)?;
            write_rixs_map_cache(&work_dir.join("rixsEE-sat.dat"), &satellite.rixs_ee)?;
        }
    } else {
        let herfd = rixs_skip_calc_herfd_from_rixs_et_map(&map)
            .context("failed to extract RIXS SkipCalc herfd.dat from cached rixsET.dat")?;
        write_rixs_line_cache(&work_dir.join("herfd.dat"), &herfd)?;
    }
    write_missing_final_outputs_from_cached_rixs_maps(work_dir, &input)?;
    write_missing_diagonal_line_outputs(work_dir)?;

    let outputs = cached_output_paths(work_dir)?;
    for output in &outputs {
        roundtrip_cached_output(output)?;
    }
    Ok(outputs)
}

fn read_skip_calc_edge_offsets(
    work_dir: &Path,
    input: &RixsInput,
) -> Result<Option<RixsSkipCalcEdgeOffsets>> {
    if !input.switches.read_poles {
        return Ok(None);
    }
    let path = work_dir.join("edges.dat");
    if !path.is_file() {
        return Ok(None);
    }

    let data =
        read_edges_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(Some(
        rixs_skip_calc_edge_offsets_from_edges_dat(&data, input.xmu)
            .context("failed to normalize RIXS SkipCalc edges.dat poles")?,
    ))
}

fn skip_calc_satellite_outputs(
    work_dir: &Path,
    map: &RixsMapData,
    energy_window: refeff_io::RixsEnergyWindow,
    edges: RixsSkipCalcEdgeOffsets,
    chemical_potential_hartree: f64,
) -> Result<Option<refeff_io::RixsDatFromFinalSpectrum>> {
    let xes_path = work_dir.join("XES").join("xmu.dat");
    if !xes_path.is_file() {
        return Ok(None);
    }

    let xes = read_xmu_dat(&xes_path)
        .with_context(|| format!("failed to read {}", xes_path.display()))?;
    rixs_skip_calc_satellite_outputs(map, energy_window, edges, chemical_potential_hartree, &xes)
        .map(Some)
        .context("failed to build RIXS SkipCalc MBConv satellite output tables")
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)?;
    Ok(1)
}

fn prepare_module_log_cache(work_dir: &Path) -> Result<()> {
    let path = work_dir.join("logrixs.dat");
    if path.is_file() {
        read_module_log_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(())
}

fn module_log_is_recoverable(work_dir: &Path, source_outputs_written: bool) -> bool {
    let path = work_dir.join("logrixs.dat");
    source_outputs_written && path.is_file() && read_module_log_dat(&path).is_err()
}

fn write_or_generate_module_log(path: &Path, outputs: &[CachedOutputPath]) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_generated_module_log(path, outputs)
}

fn write_or_recover_module_log(
    path: &Path,
    outputs: &[CachedOutputPath],
    source_outputs_written: bool,
) -> Result<usize> {
    if source_outputs_written && path.is_file() && read_module_log_dat(path).is_err() {
        return write_generated_module_log(path, outputs);
    }
    write_or_generate_module_log(path, outputs)
}

fn write_generated_module_log(path: &Path, outputs: &[CachedOutputPath]) -> Result<usize> {
    write_module_log(
        path,
        &generated_rixs_module_log(has_satellite_outputs(outputs)),
    )?;
    Ok(1)
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn generated_rixs_module_log(has_satellite_outputs: bool) -> ModuleLogData {
    let mut lines = vec![
        " ".to_string(),
        "Reading data.".to_string(),
        "Writing results.".to_string(),
    ];
    if has_satellite_outputs {
        lines.push("Writing results.".to_string());
    }
    let line_terminators = vec!["\n".to_string(); lines.len()];
    ModuleLogData {
        lines,
        line_terminators,
    }
}

fn has_satellite_outputs(outputs: &[CachedOutputPath]) -> bool {
    outputs.iter().any(|output| {
        output
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("-sat."))
    })
}

fn rixs_solver_input(input: &RixsInput, phase: Option<&PhaseBinRixsHandoff>) -> RixsInput {
    let mut solver_input = input.clone();
    if solver_input.xmu <= -100.0
        && let Some(phase) = phase
    {
        solver_input.xmu = phase.scalars.fermi_level;
    }
    solver_input
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Map,
    Line,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedOutputPath {
    path: PathBuf,
    kind: CachedOutputKind,
}

fn cached_output_paths(work_dir: &Path) -> Result<Vec<CachedOutputPath>> {
    let mut outputs = Vec::new();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", work_dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }

        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let kind = if is_rixs_map_name(name) {
            Some(CachedOutputKind::Map)
        } else if is_rixs_line_name(name) {
            Some(CachedOutputKind::Line)
        } else {
            None
        };
        if let Some(kind) = kind {
            outputs.push(CachedOutputPath {
                path: entry.path(),
                kind,
            });
        }
    }

    outputs.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(outputs)
}

fn is_rixs_map_name(name: &str) -> bool {
    matches!(
        name,
        "rixsET.dat"
            | "rixsET-sat.dat"
            | "rixsEE.dat"
            | "rixsEE-sat.dat"
            | "rixsEI.dat"
            | "rixsEI-sat.dat"
            | "rixs0.dat"
            | "rixs1.dat"
    )
}

fn is_rixs_line_name(name: &str) -> bool {
    matches!(
        name,
        "herfd.dat"
            | "herfd-sat.dat"
            | "xasEI.dat"
            | "xasEI-sat.dat"
            | "xasEF.dat"
            | "xasEF-sat.dat"
            | "xas0.dat"
            | "xas1.dat"
    )
}

#[cfg(test)]
mod tests {
    use super::{
        RIXS_SOURCE_REQUIREMENT_ERROR, build_optional_rixs_solver_handoffs,
        generated_rixs_module_log, has_cached_rixs_output, has_supported_solver_handoff,
        project_rixs_solver_energy_grid_from_phase, read_input, resolve_rixs_edge_pair,
        rixs_edge_input_string, rixs_solver_input, run_for_input, run_in_dir,
        run_supported_solver_handoff_in_dir, write_rixs_edges_if_needed,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4};
    use num_complex::Complex64;
    use refeff_core::{FEFF_HARTREE_EV, RixsSelfEnergyGridInput, rixs_prepare_self_energy_grid};
    use refeff_io::phase_bin::PHASE_BIN_DEFAULT_PAD_WIDTH;
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::{
        APOT_CORE_HOLE_RADIAL_POINTS, ApotBinData, ApotBinPayload, ApotBinRecords, ApotBinSection,
        ApotBinType, ApotBinValue, CfAverage, EdgesDatData, EdgesDatRow, GgDatData, GgDatSection,
        GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, ModuleLogData, MpseDatData,
        PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotBinData, PotBinScalars,
        RixsBroadening, RixsEnergyWindow, RixsInput, RixsLineData, RixsMapData, RixsSwitches,
        VtotDatData, WscrnDatData, XmuDatData, XsectDatData, XsectDatScalars, XsphAdvanced,
        XsphControl, XsphGrid, XsphInput, XsphInputSourceFormat, XsphRlDatData, XsphRlDatRecord,
        global_input_string, phase_bin_rixs_handoff_from_phase_bin, read_edges_dat,
        read_module_log_dat, read_mpse_dat, read_phase_bin, read_pot_bin, read_rixs_line,
        read_rixs_map, read_wscrn_dat, rixs_input_string, write_apot_bin, write_edges_dat,
        write_gg_bin, write_module_log_dat, write_mpse_dat, write_phase_bin, write_pot_bin,
        write_rixs_line, write_rixs_map, write_vtot_dat, write_wscrn_dat, write_xmu_dat,
        write_xsect_dat, write_xsph_rl_dat, xsect_dat_rixs_handoff, xsph_input_string,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn stock_rixs_edge_inputs_replace_conflicting_cards_and_enable_rlprint() -> Result<()> {
        let source = "TITLE sample\r\nEDGE L3 VAL\r\nCOREHOLE RPA\r\nRIXS\r\nXES -20 15 0.05\r\nEGRID\r\ne_grid -15 -1 1\r\nEND\r\n";

        let incident = rixs_edge_input_string(source, "L3", "L3")?;
        assert!(incident.starts_with("ICORE 4\nCOREHOLE RPA\nRLPRINT\nEDGE L3\nXANES 20\n"));
        assert_eq!(incident.matches("EDGE L3").count(), 1);
        assert_eq!(incident.matches("COREHOLE").count(), 1);
        assert!(!incident.lines().any(|line| line.trim() == "RIXS"));
        assert!(!incident.lines().any(|line| line.trim().starts_with("XES")));
        assert!(incident.contains("e_grid -15 -1 1\n"));

        let final_state = rixs_edge_input_string(source, "L3", "VAL")?;
        assert!(final_state.starts_with("COREHOLE NONE\nRLPRINT\nEDGE L3\nXANES 20\n"));
        assert!(!final_state.contains("ICORE"));
        Ok(())
    }

    #[test]
    fn rixs_edges_use_unrounded_binary_pot_edge_scalar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let first_dir = temp.path().join("L3");
        let second_dir = temp.path().join("VAL");
        std::fs::create_dir_all(&first_dir)?;
        std::fs::create_dir_all(&second_dir)?;

        let binary_edge = 424.914_983_172_784_56;
        let mut pot = sample_rixs_mpse_pot_bin();
        pot.scalars.edge_position = binary_edge;
        write_pot_bin(first_dir.join("pot.bin"), &pot)?;
        let decoded_binary_edge = read_pot_bin(first_dir.join("pot.bin"))?
            .scalars
            .edge_position;

        let mut rounded_incident = sample_rixs_xsect_dat(3, 3);
        rounded_incident.scalars.chemical_potential = 424.914_7;
        write_xsect_dat(first_dir.join("xsect.dat"), &rounded_incident)?;
        write_xsect_dat(second_dir.join("xsect.dat"), &sample_rixs_xsect_dat(3, 3))?;

        write_rixs_input(temp.path(), true)?;
        let input = read_input(temp.path())?;
        write_rixs_edges_if_needed(temp.path(), &input, &first_dir, &second_dir)?;

        let edges = read_edges_dat(temp.path().join("edges.dat"))?;
        assert_eq!(edges.rows[0].chemical_potential, decoded_binary_edge);
        assert_ne!(
            edges.rows[0].chemical_potential,
            rounded_incident.scalars.chemical_potential
        );
        Ok(())
    }

    #[test]
    fn explicit_rixs_edge_pair_fails_closed_when_one_side_is_missing() {
        let error = resolve_rixs_edge_pair("phase_1.bin", Some(1), "phase_2.bin", None, Some(9))
            .expect_err("one explicit edge must not fall back to a shared handoff");
        assert!(error.to_string().contains("phase_2.bin is missing"));
    }

    #[test]
    fn rixs_solver_xmu_uses_phase_value_for_stock_input_sentinel() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        let sentinel_input = read_input(temp.path())?;
        let phase = sample_rixs_phase_bin(8).to_rixs_handoff()?;

        assert_eq!(
            rixs_solver_input(&sentinel_input, Some(&phase)).xmu,
            phase.scalars.fermi_level
        );

        let mut explicit_input = sentinel_input;
        explicit_input.xmu = -0.5;
        assert_eq!(rixs_solver_input(&explicit_input, Some(&phase)).xmu, -0.5);
        Ok(())
    }

    #[test]
    fn rixs_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), false)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_rixs_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_advertise_shared_wscrn_without_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_rixs_wscrn_dat(4))?;

        assert!(!has_cached_rixs_output(temp.path())?);
        assert!(!has_supported_solver_handoff(temp.path())?);
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        let map = sample_rixs_map();
        write_rixs_map(temp.path().join("rixsET.dat"), &map)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        std::fs::write(temp.path().join("rixs.inp"), "not a rixs.inp handoff\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);
        assert!(!has_supported_solver_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed rixs.inp should fail through the explicit RIXS runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("rixs.inp"), "{chain}");
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let map = sample_rixs_map();
        write_rixs_map(temp.path().join("rixsET.dat"), &map)?;

        assert!(!has_cached_rixs_output(temp.path())?);
        assert!(!has_supported_solver_handoff(temp.path())?);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
        assert!(!temp.path().join("herfd.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require cached output or source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_validates_supported_solver_handoff_without_entering_solver() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;

        assert!(has_supported_solver_handoff(temp.path())?);
        let count = run_supported_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("herfd.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_validates_global_transition_setup_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_rixs_global_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_builds_radial_overlap_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_rixs_global_input(temp.path())?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_solver_handoff_bundle_retains_raw_solver_inputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;

        let input = read_input(temp.path())?;
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;

        assert!(handoffs.has_raw_solver_inputs());
        assert_eq!(
            handoffs
                .incident_phase_shifts
                .as_ref()
                .context("missing incident phase shifts")?
                .dim(),
            (3, 8)
        );
        assert_eq!(
            handoffs
                .final_phase_shifts
                .as_ref()
                .context("missing final phase shifts")?
                .dim(),
            (3, 8)
        );
        assert_eq!(
            handoffs
                .radial_overlaps
                .as_ref()
                .context("missing radial overlaps")?
                .dim(),
            (3, 3, 8)
        );
        assert_eq!(
            handoffs
                .initial_transition_amplitudes
                .as_ref()
                .context("missing initial TLb amplitudes")?
                .dim(),
            (3, 3, 4, 8)
        );
        assert_eq!(
            handoffs
                .wave_numbers
                .as_ref()
                .context("missing wave numbers")?
                .final_wave_numbers
                .len(),
            3
        );
        assert_eq!(
            handoffs
                .convolved_transition_amplitudes
                .as_ref()
                .context("missing convolved TLb amplitudes")?
                .dim(),
            (3, 3, 4, 8)
        );
        assert_eq!(
            handoffs
                .raw_cross_section
                .as_ref()
                .context("missing raw cross-section")?
                .dim(),
            (3, 3, 2)
        );
        let poles = handoffs
            .pole_normalization
            .as_ref()
            .context("missing normalized poles")?;
        assert_eq!(poles.edge_splits.len(), 1);
        assert_eq!(
            handoffs
                .self_energy
                .as_ref()
                .context("missing self-energy grid")?
                .len(),
            3
        );
        let post_raw = handoffs
            .post_raw_spectrum
            .as_ref()
            .context("missing post-raw spectrum")?;
        assert_eq!(post_raw.edge_contributions.dim(), (3, 3, 2, 1));
        assert_eq!(post_raw.summed_cross_section.dim(), (3, 3, 2));
        assert_eq!(post_raw.spectrum.rixs_et.nrows(), 9);
        let standard_outputs = handoffs
            .standard_outputs
            .as_ref()
            .context("missing standard output payloads")?;
        assert_eq!(standard_outputs.rixs_et.point_count(), 9);
        assert_eq!(standard_outputs.herfd.point_count(), 3);
        assert_eq!(standard_outputs.rixs_ee.point_count(), 9);
        assert!(handoffs.satellite_outputs.is_none());
        Ok(())
    }

    #[test]
    fn rixs_module_writes_standard_outputs_from_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEI.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE.dat"))?.point_count(),
            9
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        Ok(())
    }

    #[test]
    fn rixs_module_writes_full_source_outputs_when_partial_cache_exists() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;

        assert!(has_cached_rixs_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEI.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE.dat"))?.point_count(),
            9
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_claim_cached_output_with_malformed_phase_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed RIXS phase source should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("phase.bin"), "{chain}");
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
            sample_rixs_line().point_count()
        );
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_writes_standard_outputs_from_shared_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_shared_standard_source_handoff(temp.path())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEI.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE.dat"))?.point_count(),
            9
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        Ok(())
    }

    #[test]
    fn rixs_module_writes_standard_outputs_from_no_readpoles_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        std::fs::remove_file(temp.path().join("edges.dat"))?;
        let mut input = read_input(temp.path())?;
        input.switches.read_poles = false;
        std::fs::write(temp.path().join("rixs.inp"), rixs_input_string(&input)?)?;

        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;
        let poles = handoffs
            .pole_normalization
            .as_ref()
            .context("missing default ReadPoles=false pole data")?;
        assert_eq!(poles.incident_edge, 0.0);
        assert_eq!(poles.final_edge, 0.0);
        assert_eq!(poles.edge_splits.as_slice().unwrap(), &[0.0]);
        assert_eq!(poles.edge_amplitudes.as_slice().unwrap(), &[1.0]);
        assert_eq!(poles.edge_widths.as_slice().unwrap(), &[0.0]);
        assert_eq!(poles.core_width, input.broadening.gam_ch);
        assert!(handoffs.standard_outputs.is_some());

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert!(!temp.path().join("edges.dat").exists());
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
            3
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        Ok(())
    }

    #[test]
    fn rixs_module_writes_satellite_outputs_from_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        std::fs::create_dir(temp.path().join("XES"))?;
        write_xmu_dat(
            temp.path().join("XES").join("xmu.dat"),
            &sample_xes_xmu_dat(),
        )?;

        let input = read_input(temp.path())?;
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;
        assert!(handoffs.satellite_outputs.is_some());

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 11);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET-sat.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd-sat.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEI-sat.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF-sat.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE-sat.dat"))?.point_count(),
            9
        );
        assert_eq!(
            optional_module_log(temp.path().join("logrixs.dat"))?
                .context("missing generated RIXS log")?,
            generated_rixs_module_log(true)
        );
        Ok(())
    }

    #[test]
    fn rixs_module_writes_satellite_source_outputs_when_partial_cache_exists() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        std::fs::create_dir(temp.path().join("XES"))?;
        write_xmu_dat(
            temp.path().join("XES").join("xmu.dat"),
            &sample_xes_xmu_dat(),
        )?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;

        assert!(has_cached_rixs_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 11);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEI.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET-sat.dat"))?.point_count(),
            9
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd-sat.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEI-sat.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF-sat.dat"))?.point_count(),
            3
        );
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE-sat.dat"))?.point_count(),
            9
        );
        assert_eq!(
            optional_module_log(temp.path().join("logrixs.dat"))?
                .context("missing generated RIXS log")?,
            generated_rixs_module_log(true)
        );
        Ok(())
    }

    #[test]
    fn rixs_module_writes_standard_outputs_from_read_sigma_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        write_rixs_input_with_read_sigma(temp.path(), true)?;
        write_mpse_dat(temp.path().join("mpse.dat"), &sample_rixs_mpse_dat())?;

        let input = read_input(temp.path())?;
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;
        let self_energy = handoffs
            .self_energy
            .as_ref()
            .context("missing ReadSigma self-energy grid")?;
        assert!(self_energy.iter().any(|value| value.norm() > 0.0));

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        Ok(())
    }

    #[test]
    fn rixs_module_read_sigma_consumes_mpse_cu_reference_zip() -> Result<()> {
        let Some(zip_path) = reference_mpse_cu_zip()? else {
            crate::require_fixture!("RIXS MPSE reference test; MPSE/Cu REFERENCE.zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("RIXS MPSE reference test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        write_rixs_input_with_read_sigma(temp.path(), true)?;
        std::fs::write(
            temp.path().join("mpse.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/mpse.dat")?,
        )?;

        let input = read_input(temp.path())?;
        let mpse = read_mpse_dat(temp.path().join("mpse.dat"))?;
        assert!(mpse.energy_ev.len() > 10);
        assert!(mpse.renormalization.is_some());
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;
        let xsect = &handoffs
            .xsect
            .as_ref()
            .context("missing RIXS xsect handoff")?
            .handoff;
        let self_energy = handoffs
            .self_energy
            .as_ref()
            .context("missing MPSE/Cu ReadSigma self-energy grid")?;
        let solver_input = rixs_solver_input(&input, handoffs.incident_phase.as_ref());
        let expected = rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
            relative_energies: xsect.relative_energies_hartree.view(),
            mpse_energy_ev: mpse.energy_ev.view(),
            mpse_self_energy_ev: mpse.self_energy.view(),
            fermi_level: solver_input.xmu,
            hartree_ev: FEFF_HARTREE_EV,
        })?;

        assert_eq!(self_energy.len(), expected.len());
        assert!(self_energy.iter().any(|value| value.norm() > 0.0));
        for (&actual, &expected) in self_energy.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected, 1.0e-12);
        }

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert!(temp.path().join("rixsET.dat").is_file());
        Ok(())
    }

    #[test]
    fn rixs_module_writes_read_sigma_outputs_from_xsph_mpse_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        write_rixs_input_with_read_sigma(temp.path(), true)?;
        write_xsph_mpse_source_handoff(temp.path())?;

        let input = read_input(temp.path())?;
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;
        let self_energy = handoffs
            .self_energy
            .as_ref()
            .context("missing XSPH-generated RIXS self-energy grid")?;
        assert!(self_energy.iter().any(|value| value.norm() > 0.0));
        assert!(handoffs.standard_outputs.is_some());
        assert!(!temp.path().join("mpse.dat").exists());

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        assert!(!temp.path().join("mpse.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_read_sigma_prefers_xsph_mpse_source_over_stale_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        write_rixs_input_with_read_sigma(temp.path(), true)?;
        write_xsph_mpse_source_handoff(temp.path())?;
        write_mpse_dat(temp.path().join("mpse.dat"), &sample_rixs_mpse_dat())?;

        let input = read_input(temp.path())?;
        let stale_mpse = read_mpse_dat(temp.path().join("mpse.dat"))?;
        let source_mpse = crate::xsph::generate_mpse_dat_from_source_handoff(temp.path())?
            .context("missing generated XSPH MPSE source handoff")?;
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;
        let xsect = &handoffs
            .xsect
            .as_ref()
            .context("missing RIXS xsect handoff")?
            .handoff;
        let actual = handoffs
            .self_energy
            .as_ref()
            .context("missing source-backed RIXS self-energy grid")?;
        let solver_input = rixs_solver_input(&input, handoffs.incident_phase.as_ref());
        let expected_source = rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
            relative_energies: xsect.relative_energies_hartree.view(),
            mpse_energy_ev: source_mpse.energy_ev.view(),
            mpse_self_energy_ev: source_mpse.self_energy.view(),
            fermi_level: solver_input.xmu,
            hartree_ev: FEFF_HARTREE_EV,
        })?;
        let expected_stale = rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
            relative_energies: xsect.relative_energies_hartree.view(),
            mpse_energy_ev: stale_mpse.energy_ev.view(),
            mpse_self_energy_ev: stale_mpse.self_energy.view(),
            fermi_level: solver_input.xmu,
            hartree_ev: FEFF_HARTREE_EV,
        })?;

        assert_eq!(actual.len(), expected_source.len());
        for (&actual, &expected) in actual.iter().zip(expected_source.iter()) {
            assert_complex_close(actual, expected, 1.0e-12);
        }
        assert!(
            actual
                .iter()
                .zip(expected_stale.iter())
                .any(|(&actual, &stale)| (actual - stale).norm() > 1.0e-12),
            "self-energy should come from XSPH source, not stale cached mpse.dat"
        );
        assert_eq!(read_mpse_dat(temp.path().join("mpse.dat"))?, stale_mpse);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert!(temp.path().join("rixsET.dat").is_file());
        assert_eq!(read_mpse_dat(temp.path().join("mpse.dat"))?, stale_mpse);
        Ok(())
    }

    #[test]
    fn rixs_module_read_sigma_uses_xsph_mpse_source_for_malformed_cache() -> Result<()> {
        let source_only = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(source_only.path())?;
        write_rixs_input_with_read_sigma(source_only.path(), true)?;
        write_xsph_mpse_source_handoff(source_only.path())?;

        let source_count = run_in_dir(source_only.path())?;

        assert_eq!(source_count, 6);
        let source_map = read_rixs_map(source_only.path().join("rixsET.dat"))?;

        let malformed_with_source = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(malformed_with_source.path())?;
        write_rixs_input_with_read_sigma(malformed_with_source.path(), true)?;
        write_xsph_mpse_source_handoff(malformed_with_source.path())?;
        let malformed_mpse = b"not an mpse.dat cache\n";
        std::fs::write(
            malformed_with_source.path().join("mpse.dat"),
            malformed_mpse,
        )?;

        let count = run_in_dir(malformed_with_source.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(malformed_with_source.path().join("rixsET.dat"))?,
            source_map
        );
        assert_eq!(
            std::fs::read(malformed_with_source.path().join("mpse.dat"))?,
            malformed_mpse
        );
        Ok(())
    }

    #[test]
    fn rixs_module_read_sigma_requires_mpse_for_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;
        write_rixs_input_with_read_sigma(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("ReadSigma source handoff should require mpse.dat")?;
        let chain = format!("{error:#?}");

        assert!(chain.contains("mpse.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn rixs_run_for_input_writes_standard_outputs_from_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_standard_source_handoff(temp.path())?;

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 6);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
            9
        );
        assert!(optional_module_log(temp.path().join("logrixs.dat"))?.is_some());
        Ok(())
    }

    #[test]
    fn rixs_module_rejects_short_radial_overlap_grid_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_rixs_global_input(temp.path())?;
        write_xsph_rl_dat(
            temp.path().join("rl_1.dat"),
            &sample_rixs_rl_dat_with_radial_count(3, 1, 3),
        )?;
        write_xsph_rl_dat(
            temp.path().join("rl_2.dat"),
            &sample_rixs_rl_dat_with_radial_count(3, 1, 3),
        )?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(3))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(3))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("short radial grid should fail before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("failed to build RIXS radial DeltaV overlaps"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_rejects_malformed_global_transition_handoff_before_source_requirement()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed global.inp should fail before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("global.inp"), "{chain}");
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_claim_malformed_global_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

        assert!(!has_supported_solver_handoff(temp.path())?);
        assert_eq!(run_supported_solver_handoff_in_dir(temp.path())?, 0);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed RIXS global.inp should fail through the explicit RIXS runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("global.inp"), "{chain}");
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_validates_explicit_green_handoffs_without_entering_solver() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        write_gg_bin(temp.path().join("gg_1.bin"), &sample_rixs_gg_bin(3, 2))?;
        write_gg_bin(temp.path().join("gg_2.bin"), &sample_rixs_gg_bin(3, 2))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;

        assert!(has_supported_solver_handoff(temp.path())?);
        let count = run_supported_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 8);
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("herfd.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_recover_unneeded_shared_wscrn_when_edges_are_explicit() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;
        std::fs::write(temp.path().join("vtot.dat"), "not a vtot.dat table\n")?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_zero_core_hole_apot_bin(),
        )?;

        assert!(has_supported_solver_handoff(temp.path())?);
        let count = run_supported_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert!(!temp.path().join("wscrn.dat").exists());
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_ignores_malformed_shared_handoffs_when_edge_handoffs_are_explicit() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        std::fs::write(temp.path().join("phase.bin"), "not a phase.bin cache\n")?;
        write_phase_bin(temp.path().join("phase_1.bin"), &sample_rixs_phase_bin(8))?;
        write_phase_bin(temp.path().join("phase_2.bin"), &sample_rixs_phase_bin(8))?;
        std::fs::write(temp.path().join("rl.dat"), "not an rl.dat handoff\n")?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        std::fs::write(temp.path().join("wscrn.dat"), "not a wscrn.dat handoff\n")?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        std::fs::write(temp.path().join("xsect.dat"), "not an xsect.dat handoff\n")?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;

        assert!(has_supported_solver_handoff(temp.path())?);
        let count = run_supported_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 11);
        assert!(read_phase_bin(temp.path().join("phase.bin")).is_err());
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("herfd.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_skips_supported_solver_handoff_without_handoff_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;

        assert!(!has_supported_solver_handoff(temp.path())?);
        assert_eq!(run_supported_solver_handoff_in_dir(temp.path())?, 0);
        Ok(())
    }

    #[test]
    fn rixs_module_detects_recoverable_shared_wscrn_handoff_without_other_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        let vtot = sample_rixs_vtot_dat(4);
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_zero_core_hole_apot_bin(),
        )?;

        assert!(!temp.path().join("wscrn.dat").exists());
        assert!(has_supported_solver_handoff(temp.path())?);
        assert!(temp.path().join("wscrn.dat").exists());

        let count = run_supported_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_vector_close(
            wscrn
                .radius_bohr
                .as_slice()
                .context("wscrn radius storage")?,
            vtot.radius_bohr.as_slice().context("vtot radius storage")?,
        );
        assert_vector_close(
            wscrn
                .screened_potential
                .as_slice()
                .context("wscrn screened-potential storage")?,
            vtot.screened_core_hole_potential
                .as_slice()
                .context("vtot screened-potential storage")?,
        );
        assert!(wscrn.core_hole_potential.iter().all(|value| *value == 0.0));
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_advertise_unrecoverable_shared_wscrn_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        std::fs::write(temp.path().join("vtot.dat"), "not a vtot.dat table\n")?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_zero_core_hole_apot_bin(),
        )?;

        assert!(!has_supported_solver_handoff(temp.path())?);
        assert_eq!(run_supported_solver_handoff_in_dir(temp.path())?, 0);
        assert!(!temp.path().join("wscrn.dat").exists());
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_validates_phase_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(7))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should validate phase.bin before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("failed to build RIXS phase handoff from"),
            "{chain}"
        );
        assert!(
            chain.contains("RIXS phase handoff requires at least 8 transition channels"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_accepts_phase_handoff_then_requires_complete_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_validates_rl_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        let mut radial = sample_rixs_rl_dat(3, 1);
        radial.records.pop();
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &radial)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should validate rl_1.dat before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("failed to build RIXS rl.dat handoff from"),
            "{chain}"
        );
        assert!(
            chain.contains("record count 5 is not a multiple of angular count 2"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_accepts_phase_and_rl_handoffs_then_requires_complete_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_validates_shared_rl_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        let mut radial = sample_rixs_rl_dat(3, 1);
        radial.records.pop();
        write_xsph_rl_dat(temp.path().join("rl.dat"), &radial)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should validate shared rl.dat before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("failed to build RIXS rl.dat handoff from"),
            "{chain}"
        );
        assert!(chain.contains("rl.dat"), "{chain}");
        assert!(
            chain.contains("record count 5 is not a multiple of angular count 2"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_accepts_shared_rl_handoff_then_requires_complete_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl.dat"), &sample_rixs_rl_dat(3, 1))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_validates_wscrn_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(3))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(2))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should validate wscrn handoffs before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("wscrn_2.dat") && chain.contains("row count 2"),
            "{chain}"
        );
        assert!(
            chain.contains("wscrn_1.dat") && chain.contains("row count 3"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_validates_shared_wscrn_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_xsph_rl_dat(temp.path().join("rl.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_rixs_wscrn_dat(3))?;

        let error = run_in_dir(temp.path()).err().context(
            "enabled RIXS should validate shared wscrn.dat before the source requirement",
        )?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("wscrn.dat row count 3 is smaller than rl.dat radial count 4"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_accepts_shared_wscrn_handoff_then_requires_complete_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_xsph_rl_dat(temp.path().join("rl.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_rixs_wscrn_dat(4))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_validates_shared_green_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_rixs_gg_bin(2, 2))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should validate gg.bin before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("gg.bin section count 2 does not match phase handoff ne1 3 or ne 3"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_recovers_shared_wscrn_handoff_from_vtot_and_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl.dat"), &sample_rixs_rl_dat(3, 1))?;
        let vtot = sample_rixs_vtot_dat(4);
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_zero_core_hole_apot_bin(),
        )?;

        assert!(!temp.path().join("wscrn.dat").exists());
        let count = run_supported_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_vector_close(
            wscrn
                .radius_bohr
                .as_slice()
                .context("wscrn radius storage")?,
            vtot.radius_bohr.as_slice().context("vtot radius storage")?,
        );
        assert_vector_close(
            wscrn
                .screened_potential
                .as_slice()
                .context("wscrn screened-potential storage")?,
            vtot.screened_core_hole_potential
                .as_slice()
                .context("vtot screened-potential storage")?,
        );
        assert!(wscrn.core_hole_potential.iter().all(|value| *value == 0.0));
        assert!(!temp.path().join("rixsET.dat").exists());
        assert!(!temp.path().join("logrixs.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_refreshes_stale_shared_wscrn_handoff_from_vtot_and_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_complete_rixs_shared_standard_source_handoff(temp.path())?;
        let vtot = sample_rixs_vtot_dat(4);
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_zero_core_hole_apot_bin(),
        )?;
        let mut stale = sample_rixs_wscrn_dat(4);
        stale.screened_potential[0] += 0.25;
        stale.core_hole_potential[0] += 0.25;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &stale)?;

        let input = read_input(temp.path())?;
        let handoffs = build_optional_rixs_solver_handoffs(temp.path(), &input)?;

        assert!(handoffs.potential_difference.is_some());
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_vector_close(
            wscrn
                .radius_bohr
                .as_slice()
                .context("wscrn radius storage")?,
            vtot.radius_bohr.as_slice().context("vtot radius storage")?,
        );
        assert_vector_close(
            wscrn
                .screened_potential
                .as_slice()
                .context("wscrn screened-potential storage")?,
            vtot.screened_core_hole_potential
                .as_slice()
                .context("vtot screened-potential storage")?,
        );
        assert!(wscrn.core_hole_potential.iter().all(|value| *value == 0.0));
        assert_ne!(read_wscrn_dat(temp.path().join("wscrn.dat"))?, stale);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert!(temp.path().join("rixsET.dat").is_file());
        Ok(())
    }

    #[test]
    fn rixs_module_validates_xsect_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase_1.bin"), &sample_rixs_phase_bin(8))?;
        write_phase_bin(temp.path().join("phase_2.bin"), &sample_rixs_phase_bin(8))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(2, 2),
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should validate xsect_2.dat before the source requirement")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("xsect_2.dat ne1 2 does not match phase handoff ne1 3"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_validates_shared_xsect_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsect_dat(temp.path().join("xsect.dat"), &sample_rixs_xsect_dat(2, 2))?;

        let error = run_in_dir(temp.path()).err().context(
            "enabled RIXS should validate shared xsect.dat before the source requirement",
        )?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("xsect.dat ne1 2 does not match phase handoff ne1 3"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn rixs_solver_uses_binary_phase_grid_at_fermi_threshold() -> Result<()> {
        const PHASE_XMU: f64 = -0.245_073_443_460_620_4;
        const ROUNDED_XSECT_ENERGY: f64 = -0.245_073_443_494_042_0;

        let mut phase_data = sample_rixs_phase_bin(8);
        phase_data.scalars.fermi_level = PHASE_XMU;
        phase_data.energy_grid[1] = Complex64::new(PHASE_XMU, 2.5e-4);
        let phase = phase_bin_rixs_handoff_from_phase_bin(&phase_data)?;

        let mut xsect = xsect_dat_rixs_handoff(&sample_rixs_xsect_dat(3, 3))?;
        xsect.relative_energies_hartree[1] = ROUNDED_XSECT_ENERGY;
        xsect.energy_grid_hartree[1].re = ROUNDED_XSECT_ENERGY;

        assert!(
            ROUNDED_XSECT_ENERGY < PHASE_XMU,
            "the rounded text sentinel must reproduce the lost threshold row"
        );
        project_rixs_solver_energy_grid_from_phase(Path::new("xsect_2.dat"), &mut xsect, &phase)?;

        assert_eq!(xsect.relative_energies_hartree[1], PHASE_XMU);
        assert_eq!(xsect.energy_grid_hartree[1], phase.energy_grid[1]);
        Ok(())
    }

    #[test]
    fn rixs_module_accepts_shared_xsect_handoff_then_requires_complete_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsect_dat(temp.path().join("xsect.dat"), &sample_rixs_xsect_dat(3, 3))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_accepts_full_sidecar_handoffs_then_requires_complete_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_xsph_rl_dat(temp.path().join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(temp.path().join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(temp.path().join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(temp.path().join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        write_xsect_dat(
            temp.path().join("xsect_2.dat"),
            &sample_rixs_xsect_dat(3, 3),
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require complete source handoffs")?;

        assert!(error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_generates_herfd_from_cached_rixs_et() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip(temp.path(), true, true)?;
        let map = sample_rixs_map();
        write_rixs_map(temp.path().join("rixsET.dat"), &map)?;

        assert!(has_cached_rixs_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert!(has_cached_rixs_output(temp.path())?);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            RixsLineData {
                header_lines: Vec::new(),
                energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0]),
                channels: Array2::from_shape_vec((2, 2), vec![1.0e-6, 1.2e-6, 4.0e-6, 4.2e-6])?,
            }
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            generated_rixs_module_log(false)
        );
        assert!(!temp.path().join("xasEI.dat").exists());
        assert!(!temp.path().join("xasEF.dat").exists());
        assert!(!temp.path().join("rixsEE.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_recovers_malformed_module_log_from_cached_map() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip(temp.path(), true, true)?;
        let map = sample_rixs_map();
        write_rixs_map(temp.path().join("rixsET.dat"), &map)?;
        std::fs::write(temp.path().join("logrixs.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_rixs_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            generated_rixs_module_log(false)
        );
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_rejects_unowned_malformed_final_output_without_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip(temp.path(), true, true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        std::fs::write(temp.path().join("xasEI.dat"), "not a RIXS line\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("RIXS SkipCalc should validate unowned cached side outputs")?;
        let chain = format!("{error:#?}");

        assert!(chain.contains("xasEI.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_generates_xas_and_emission_outputs_with_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            true,
            RixsEnergyWindow {
                emin_i: -0.5,
                emax_i: 2.0,
                emin_f: -0.5,
                emax_f: 2.0,
            },
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;

        assert!(has_cached_rixs_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEI.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[8.0, 208.0], [10.0, 210.0], [12.0, 212.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEF.dat"))?,
            &[6.0, 7.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[4.0, 204.0], [10.0, 210.0], [16.0, 216.0]],
        );

        let rixs_ee = read_rixs_map(temp.path().join("rixsEE.dat"))?;
        assert_eq!(rixs_ee.block_lengths, vec![3, 3, 3]);
        assert_vector_close(
            rixs_ee.first_energy_ev.as_slice().unwrap(),
            &[10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 10.0, 11.0, 12.0]
                .map(|energy| energy * FEFF_HARTREE_EV),
        );
        assert_vector_close(
            rixs_ee.second_energy_ev.as_slice().unwrap(),
            &[4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
        );
        assert_matrix_close(
            &rixs_ee.channels,
            &[
                [3.0, 103.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 101.0],
                [5.0, 105.0],
                [9.0, 109.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7.0, 107.0],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_rewrites_stale_final_outputs_with_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            true,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        write_rixs_line(temp.path().join("xasEI.dat"), &sample_rixs_line())?;
        write_rixs_map(temp.path().join("rixsEE.dat"), &sample_rixs_map())?;

        assert!(has_cached_rixs_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEI.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[8.0, 208.0], [10.0, 210.0], [12.0, 212.0]],
        );
        let rixs_ee = read_rixs_map(temp.path().join("rixsEE.dat"))?;
        assert_eq!(rixs_ee.block_lengths, vec![3, 3, 3]);
        assert_matrix_close(
            &rixs_ee.channels,
            &[
                [3.0, 103.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 101.0],
                [5.0, 105.0],
                [9.0, 109.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7.0, 107.0],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_generates_satellite_outputs_with_xes_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            true,
            RixsEnergyWindow {
                emin_i: -0.5,
                emax_i: 2.0,
                emin_f: -0.5,
                emax_f: 2.0,
            },
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        std::fs::create_dir(temp.path().join("XES"))?;
        write_xmu_dat(
            temp.path().join("XES").join("xmu.dat"),
            &sample_xes_xmu_dat(),
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 11);
        assert_line_close(
            &read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[0.5, 50.5], [3.5, 103.5], [7.5, 107.5]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEI-sat.dat"))?,
            &[0.0, 1.0, 2.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[5.5, 180.5], [7.25, 182.25], [9.0, 184.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEF-sat.dat"))?,
            &[0.0, 1.0, 2.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[2.0, 102.0], [7.0, 207.0], [13.0, 213.0]],
        );

        let rixs_ee_sat = read_rixs_map(temp.path().join("rixsEE-sat.dat"))?;
        assert_matrix_close(
            &rixs_ee_sat.channels,
            &[
                [1.5, 51.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.5, 50.5],
                [3.5, 103.5],
                [7.5, 107.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [5.5, 105.5],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_does_not_advertise_malformed_xes_satellite_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            true,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        std::fs::create_dir(temp.path().join("XES"))?;
        std::fs::write(
            temp.path().join("XES").join("xmu.dat"),
            "not an xmu cache\n",
        )?;

        assert!(!has_cached_rixs_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed XES satellite source should fail the explicit RIXS run")?;
        let chain = format!("{error:#?}");

        assert!(chain.contains("XES/xmu.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_derives_satellite_outputs_from_cached_rixs_et_sat_with_edges()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            true,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        write_rixs_map(
            temp.path().join("rixsET-sat.dat"),
            &sample_skip_calc_rixs_map(),
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 11);
        assert_line_close(
            &read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEI-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[8.0, 208.0], [10.0, 210.0], [12.0, 212.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEF-sat.dat"))?,
            &[6.0, 7.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[4.0, 204.0], [10.0, 210.0], [16.0, 216.0]],
        );

        let rixs_ee_sat = read_rixs_map(temp.path().join("rixsEE-sat.dat"))?;
        assert_eq!(rixs_ee_sat.block_lengths, vec![3, 3, 3]);
        assert_matrix_close(
            &rixs_ee_sat.channels,
            &[
                [3.0, 103.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 101.0],
                [5.0, 105.0],
                [9.0, 109.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7.0, 107.0],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_derives_satellite_herfd_from_cached_rixs_et_sat_without_edges()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip(temp.path(), true, true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        write_rixs_map(
            temp.path().join("rixsET-sat.dat"),
            &sample_skip_calc_rixs_map(),
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 5);
        assert_line_close(
            &read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        assert!(!temp.path().join("xasEI-sat.dat").exists());
        assert!(!temp.path().join("xasEF-sat.dat").exists());
        assert!(!temp.path().join("rixsEE-sat.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_does_not_advertise_malformed_rixs_et() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip(temp.path(), true, true)?;
        std::fs::write(temp.path().join("rixsET.dat"), "not a RIXS map\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("RIXS SkipCalc should reject malformed rixsET.dat")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("rixsET.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_requires_cached_rixs_et() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip(temp.path(), true, true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("RIXS SkipCalc should require rixsET.dat")?;

        assert!(
            error
                .to_string()
                .contains("RIXS SkipCalc requires cached rixsET.dat")
        );
        Ok(())
    }

    #[test]
    fn rixs_module_derives_missing_herfd_from_cached_rixs_et() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        let map = sample_rixs_map();
        write_rixs_map(temp.path().join("rixsET.dat"), &map)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            RixsLineData {
                header_lines: Vec::new(),
                energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0]),
                channels: Array2::from_shape_vec((2, 2), vec![1.0e-6, 1.2e-6, 4.0e-6, 4.2e-6])?,
            }
        );
        Ok(())
    }

    #[test]
    fn rixs_module_recovers_malformed_herfd_from_cached_rixs_et() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        let map = sample_rixs_map();
        write_rixs_map(temp.path().join("rixsET.dat"), &map)?;
        std::fs::write(temp.path().join("herfd.dat"), "not a RIXS line\n")?;

        assert!(has_cached_rixs_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            RixsLineData {
                header_lines: Vec::new(),
                energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0]),
                channels: Array2::from_shape_vec((2, 2), vec![1.0e-6, 1.2e-6, 4.0e-6, 4.2e-6])?,
            }
        );
        Ok(())
    }

    #[test]
    fn rixs_module_derives_final_outputs_from_cached_rixs_et_with_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            false,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_line_close(
            &read_rixs_line(temp.path().join("herfd.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEI.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[8.0, 208.0], [10.0, 210.0], [12.0, 212.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEF.dat"))?,
            &[6.0, 7.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[4.0, 204.0], [10.0, 210.0], [16.0, 216.0]],
        );

        let rixs_ee = read_rixs_map(temp.path().join("rixsEE.dat"))?;
        assert_eq!(rixs_ee.block_lengths, vec![3, 3, 3]);
        assert_matrix_close(
            &rixs_ee.channels,
            &[
                [3.0, 103.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 101.0],
                [5.0, 105.0],
                [9.0, 109.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7.0, 107.0],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_module_recovers_malformed_final_line_from_cached_rixs_et_with_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            false,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        std::fs::write(temp.path().join("xasEF.dat"), "not a RIXS line\n")?;

        assert!(has_cached_rixs_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEF.dat"))?,
            &[6.0, 7.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[4.0, 204.0], [10.0, 210.0], [16.0, 216.0]],
        );
        assert!(temp.path().join("herfd.dat").is_file());
        assert!(temp.path().join("xasEI.dat").is_file());
        assert!(temp.path().join("rixsEE.dat").is_file());
        Ok(())
    }

    #[test]
    fn rixs_module_derives_satellite_outputs_from_cached_rixs_et_sat_with_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            false,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(
            temp.path().join("rixsET-sat.dat"),
            &sample_skip_calc_rixs_map(),
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_line_close(
            &read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEI-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[8.0, 208.0], [10.0, 210.0], [12.0, 212.0]],
        );
        assert_line_close(
            &read_rixs_line(temp.path().join("xasEF-sat.dat"))?,
            &[6.0, 7.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[4.0, 204.0], [10.0, 210.0], [16.0, 216.0]],
        );

        let rixs_ee = read_rixs_map(temp.path().join("rixsEE-sat.dat"))?;
        assert_eq!(rixs_ee.block_lengths, vec![3, 3, 3]);
        assert_matrix_close(
            &rixs_ee.channels,
            &[
                [3.0, 103.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 101.0],
                [5.0, 105.0],
                [9.0, 109.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7.0, 107.0],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_module_derives_satellite_herfd_from_cached_rixs_et_sat_without_edges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(
            temp.path().join("rixsET-sat.dat"),
            &sample_skip_calc_rixs_map(),
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_line_close(
            &read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        assert!(!temp.path().join("xasEI-sat.dat").exists());
        assert!(!temp.path().join("xasEF-sat.dat").exists());
        assert!(!temp.path().join("rixsEE-sat.dat").exists());
        Ok(())
    }

    #[test]
    fn rixs_module_preserves_existing_final_output_when_deriving_missing_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input_with_skip_window_and_xmu(
            temp.path(),
            true,
            false,
            sample_skip_calc_window(),
            0.0,
        )?;
        write_edges_dat(temp.path().join("edges.dat"), &sample_skip_calc_edges_dat())?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_skip_calc_rixs_map())?;
        let existing_xas = sample_rixs_line();
        write_rixs_line(temp.path().join("xasEI.dat"), &existing_xas)?;
        let expected_xas = read_rixs_line(temp.path().join("xasEI.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(read_rixs_line(temp.path().join("xasEI.dat"))?, expected_xas);
        assert!(temp.path().join("herfd.dat").is_file());
        assert!(temp.path().join("xasEF.dat").is_file());
        assert!(temp.path().join("rixsEE.dat").is_file());
        Ok(())
    }

    #[test]
    fn rixs_module_derives_missing_intermediate_xas_from_cached_maps() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixs0.dat"), &sample_rixs_map())?;
        write_rixs_map(temp.path().join("rixs1.dat"), &sample_rixs_map())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 5);
        let expected = RixsLineData {
            header_lines: Vec::new(),
            energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0]),
            channels: Array2::from_shape_vec((2, 2), vec![1.0e-6, 1.2e-6, 4.0e-6, 4.2e-6])?,
        };
        assert_eq!(read_rixs_line(temp.path().join("xas0.dat"))?, expected);
        assert_eq!(read_rixs_line(temp.path().join("xas1.dat"))?, expected);
        Ok(())
    }

    #[test]
    fn rixs_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        write_rixs_map(temp.path().join("rixsEE-sat.dat"), &sample_rixs_map())?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;
        write_rixs_line(temp.path().join("xasEF-sat.dat"), &sample_rixs_line())?;
        write_module_log_dat(temp.path().join("logrixs.dat"), &sample_module_log())?;

        let expected_map = read_rixs_map(temp.path().join("rixsET.dat"))?;
        let expected_map_sat = read_rixs_map(temp.path().join("rixsEE-sat.dat"))?;
        let expected_line = read_rixs_line(temp.path().join("herfd.dat"))?;
        let expected_line_sat = read_rixs_line(temp.path().join("xasEF-sat.dat"))?;
        let expected_log = read_module_log_dat(temp.path().join("logrixs.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 5);
        assert!(has_cached_rixs_output(temp.path())?);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, expected_map);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE-sat.dat"))?,
            expected_map_sat
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            expected_line
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF-sat.dat"))?,
            expected_line_sat
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_advertise_malformed_cached_map() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        std::fs::write(temp.path().join("rixsET.dat"), "not a RIXS map\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed cached RIXS map should be rejected during explicit run")?;

        assert!(format!("{error:#}").contains("rixsET.dat"));
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_advertise_malformed_cached_line() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        std::fs::write(temp.path().join("herfd.dat"), "not a RIXS line\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed cached RIXS line should be rejected during explicit run")?;

        assert!(format!("{error:#}").contains("herfd.dat"));
        Ok(())
    }

    #[test]
    fn rixs_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;
        std::fs::write(temp.path().join("logrixs.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_rixs_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed cached RIXS log should be rejected during explicit run")?;

        assert!(format!("{error:#}").contains("logrixs.dat"));
        Ok(())
    }

    #[test]
    fn rixs_module_recovers_malformed_module_log_from_cached_map_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        std::fs::write(temp.path().join("logrixs.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_rixs_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            RixsLineData {
                header_lines: Vec::new(),
                energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0]),
                channels: Array2::from_shape_vec((2, 2), vec![1.0e-6, 1.2e-6, 4.0e-6, 4.2e-6])?,
            }
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            generated_rixs_module_log(false)
        );
        Ok(())
    }

    #[test]
    fn rixs_module_validates_handoff_when_malformed_cache_exists() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_rixs_phase_bin(8))?;
        std::fs::write(temp.path().join("herfd.dat"), "not a RIXS line\n")?;

        assert!(!has_cached_rixs_output(temp.path())?);
        assert!(has_supported_solver_handoff(temp.path())?);
        assert_eq!(run_supported_solver_handoff_in_dir(temp.path())?, 1);

        let error = run_in_dir(temp.path())
            .err()
            .context("source-backed handoff should reach the RIXS source requirement")?;

        assert!(
            error.to_string().contains(RIXS_SOURCE_REQUIREMENT_ERROR),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn rixs_module_generates_missing_module_log_from_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            generated_rixs_module_log(false)
        );
        Ok(())
    }

    #[test]
    fn rixs_module_generates_satellite_results_log_marker() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        write_rixs_line(temp.path().join("herfd-sat.dat"), &sample_rixs_line())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            generated_rixs_module_log(true)
        );
        Ok(())
    }

    #[test]
    fn rixs_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_rixs_dir()? else {
            crate::require_fixture!("RIXS reference test; generated RIXS reference not found");
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("rixs.inp"), temp.path().join("rixs.inp"))?;
        std::fs::copy(
            reference_dir.join("referencerixsET.dat"),
            temp.path().join("rixsET.dat"),
        )?;
        std::fs::copy(
            reference_dir.join("referenceherfd.dat"),
            temp.path().join("herfd.dat"),
        )?;
        std::fs::copy(
            reference_dir.join("referenceherfd-sat.dat"),
            temp.path().join("herfd-sat.dat"),
        )?;
        let log_source = reference_dir.join("logrixs.dat");
        if log_source.is_file() {
            std::fs::copy(log_source, temp.path().join("logrixs.dat"))?;
        }

        let expected_map = read_rixs_map(temp.path().join("rixsET.dat"))?;
        let expected_herfd = read_rixs_line(temp.path().join("herfd.dat"))?;
        let expected_herfd_sat = read_rixs_line(temp.path().join("herfd-sat.dat"))?;
        let expected_log = optional_module_log(temp.path().join("logrixs.dat"))?
            .unwrap_or_else(|| generated_rixs_module_log(true));

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, expected_map);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            expected_herfd
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            expected_herfd_sat
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logrixs.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn rixs_module_skip_calc_generates_reference_herfd_when_present() -> Result<()> {
        let Some(reference_dir) = reference_rixs_dir()? else {
            crate::require_fixture!(
                "RIXS SkipCalc reference test; generated RIXS reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        let input_text = std::fs::read_to_string(reference_dir.join("rixs.inp"))?;
        let mut input = RixsInput::parse_str(reference_dir.join("rixs.inp"), &input_text)?;
        input.switches.skip_calc = true;
        std::fs::write(temp.path().join("rixs.inp"), rixs_input_string(&input)?)?;
        std::fs::copy(
            reference_dir.join("referencerixsET.dat"),
            temp.path().join("rixsET.dat"),
        )?;
        let expected_herfd = read_rixs_line(reference_dir.join("referenceherfd.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            expected_herfd
        );
        Ok(())
    }

    fn write_rixs_input(work_dir: &Path, run: bool) -> Result<()> {
        write_rixs_input_with_skip(work_dir, run, false)
    }

    fn write_rixs_input_with_skip(work_dir: &Path, run: bool, skip_calc: bool) -> Result<()> {
        write_rixs_input_with_skip_window_and_xmu(
            work_dir,
            run,
            skip_calc,
            RixsEnergyWindow {
                emin_i: 0.0,
                emax_i: 0.0,
                emin_f: 0.0,
                emax_f: 0.0,
            },
            -367_493_090.027_428_2,
        )
    }

    fn write_rixs_input_with_read_sigma(work_dir: &Path, run: bool) -> Result<()> {
        write_rixs_input_with_switches(
            work_dir,
            run,
            RixsEnergyWindow {
                emin_i: 0.0,
                emax_i: 0.0,
                emin_f: 0.0,
                emax_f: 0.0,
            },
            -367_493_090.027_428_2,
            RixsSwitches {
                read_poles: true,
                skip_calc: false,
                mbconv: true,
                read_sigma: true,
            },
        )
    }

    fn write_rixs_input_with_skip_window_and_xmu(
        work_dir: &Path,
        run: bool,
        skip_calc: bool,
        energy_window: RixsEnergyWindow,
        xmu: f64,
    ) -> Result<()> {
        write_rixs_input_with_switches(
            work_dir,
            run,
            energy_window,
            xmu,
            RixsSwitches {
                read_poles: true,
                skip_calc,
                mbconv: true,
                read_sigma: false,
            },
        )
    }

    fn write_rixs_input_with_switches(
        work_dir: &Path,
        run: bool,
        energy_window: RixsEnergyWindow,
        xmu: f64,
        switches: RixsSwitches,
    ) -> Result<()> {
        let input = RixsInput {
            run,
            broadening: RixsBroadening {
                gam_ch: 0.000_135_051_2,
                gam_exp_1: 0.000_135_051_2,
                gam_exp_2: 0.000_135_051_2,
            },
            energy_window,
            xmu,
            switches,
            edges: vec!["L3".to_string(), "VAL".to_string()],
        };
        std::fs::write(work_dir.join("rixs.inp"), rixs_input_string(&input)?)?;
        Ok(())
    }

    fn write_rixs_global_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("global.inp"),
            global_input_string(&sample_rixs_global_input())?,
        )?;
        Ok(())
    }

    fn write_complete_rixs_standard_source_handoff(work_dir: &Path) -> Result<()> {
        write_rixs_input(work_dir, true)?;
        write_phase_bin(work_dir.join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_rixs_global_input(work_dir)?;
        write_xsph_rl_dat(work_dir.join("rl_1.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_xsph_rl_dat(work_dir.join("rl_2.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(work_dir.join("wscrn_1.dat"), &sample_rixs_wscrn_dat(4))?;
        write_wscrn_dat(work_dir.join("wscrn_2.dat"), &sample_rixs_wscrn_dat(4))?;
        write_gg_bin(work_dir.join("gg_1.bin"), &sample_rixs_gg_bin(3, 4))?;
        write_gg_bin(work_dir.join("gg_2.bin"), &sample_rixs_gg_bin(3, 4))?;
        write_xsect_dat(work_dir.join("xsect_2.dat"), &sample_rixs_xsect_dat(3, 3))?;
        write_edges_dat(work_dir.join("edges.dat"), &sample_skip_calc_edges_dat())?;
        Ok(())
    }

    fn write_complete_rixs_shared_standard_source_handoff(work_dir: &Path) -> Result<()> {
        write_rixs_input(work_dir, true)?;
        write_phase_bin(work_dir.join("phase.bin"), &sample_rixs_phase_bin(8))?;
        write_rixs_global_input(work_dir)?;
        write_xsph_rl_dat(work_dir.join("rl.dat"), &sample_rixs_rl_dat(3, 1))?;
        write_wscrn_dat(work_dir.join("wscrn.dat"), &sample_rixs_wscrn_dat(4))?;
        write_gg_bin(work_dir.join("gg.bin"), &sample_rixs_gg_bin(3, 4))?;
        write_xsect_dat(work_dir.join("xsect.dat"), &sample_rixs_xsect_dat(3, 3))?;
        write_edges_dat(work_dir.join("edges.dat"), &sample_skip_calc_edges_dat())?;
        Ok(())
    }

    fn write_xsph_mpse_source_handoff(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("xsph.inp"),
            xsph_input_string(&sample_rixs_xsph_input())?,
        )?;
        write_phase_bin(work_dir.join("phase.bin"), &sample_rixs_mpse_phase_bin(8))?;
        write_pot_bin(work_dir.join("pot.bin"), &sample_rixs_mpse_pot_bin())?;
        Ok(())
    }

    fn sample_skip_calc_edges_dat() -> EdgesDatData {
        EdgesDatData {
            header_lines: vec!["# emu, M_kk, gam".to_string()],
            rows: vec![
                EdgesDatRow {
                    chemical_potential: 10.0,
                    matrix_element: 1.0,
                    core_hole_width: 0.1,
                },
                EdgesDatRow {
                    chemical_potential: 4.0,
                    matrix_element: 1.0,
                    core_hole_width: 0.1,
                },
            ],
        }
    }

    fn sample_skip_calc_window() -> RixsEnergyWindow {
        RixsEnergyWindow {
            emin_i: -0.5,
            emax_i: 2.0,
            emin_f: -0.5,
            emax_f: 2.0,
        }
    }

    fn sample_skip_calc_rixs_map() -> RixsMapData {
        let mut first_energy_ev = Vec::new();
        let mut second_energy_ev = Vec::new();
        let mut channels = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                let value = (row * 3 + col + 1) as f64;
                first_energy_ev.push((10.0 + col as f64) * FEFF_HARTREE_EV);
                second_energy_ev.push((4.0 + row as f64) * FEFF_HARTREE_EV);
                channels.push(value);
                channels.push(value + 100.0);
            }
        }

        RixsMapData {
            header_lines: Vec::new(),
            block_lengths: vec![3, 3, 3],
            first_energy_ev: Array1::from_vec(first_energy_ev),
            second_energy_ev: Array1::from_vec(second_energy_ev),
            channels: Array2::from_shape_vec((9, 2), channels).expect("valid channel shape"),
        }
    }

    fn sample_xes_xmu_dat() -> XmuDatData {
        XmuDatData {
            header_lines: Vec::new(),
            normalization: None,
            photon_energy_ev: Array1::from_vec(vec![0.0, 1.0]),
            relative_energy_ev: Array1::from_vec(vec![-FEFF_HARTREE_EV, 0.0]),
            wave_number: Array1::from_vec(vec![0.0, 0.0]),
            mu: Array1::from_vec(vec![1.0, 1.0]),
            mu0: Array1::from_vec(vec![0.0, 0.0]),
            chi: Array1::from_vec(vec![1.0, 1.0]),
        }
    }

    fn sample_rixs_mpse_dat() -> MpseDatData {
        MpseDatData {
            header_lines: vec!["# sample mpse.dat".to_string()],
            energy_ev: Array1::from_vec(vec![0.0, 15.0, 30.0]),
            self_energy: Array1::from_vec(vec![
                Complex64::new(0.05, -0.01),
                Complex64::new(0.06, -0.02),
                Complex64::new(0.07, -0.03),
            ]),
            renormalization: None,
            renormalization_magnitude: None,
            renormalization_phase: None,
            inelastic_mean_free_path: None,
        }
    }

    fn sample_rixs_map() -> RixsMapData {
        RixsMapData {
            header_lines: vec!["# sample RIXS map".to_string()],
            block_lengths: vec![2, 2],
            first_energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0, 11_540.0, 11_541.0]),
            second_energy_ev: Array1::from_vec(vec![-15.0, -15.0, -14.0, -14.0]),
            channels: Array2::from_shape_fn((4, 2), |(row, channel)| {
                1.0e-6 * (row + 1) as f64 + 2.0e-7 * channel as f64
            }),
        }
    }

    fn sample_rixs_line() -> RixsLineData {
        RixsLineData {
            header_lines: vec!["# sample RIXS line".to_string()],
            energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0, 11_542.0]),
            channels: Array2::from_shape_fn((3, 2), |(row, channel)| {
                2.5e-6 * (row + 1) as f64 + 1.0e-7 * channel as f64
            }),
        }
    }

    fn sample_rixs_gg_bin(section_count: usize, order: usize) -> GgDatData {
        GgDatData {
            sections: (0..section_count)
                .map(|section| GgDatSection {
                    section_number: section + 1,
                    values: Array2::from_shape_fn((order, order), |(row, column)| {
                        Complex64::new(
                            0.1 * (section + 1) as f64 + 0.01 * row as f64,
                            -0.02 * column as f64,
                        )
                    }),
                    raw_prefix_lines: None,
                })
                .collect(),
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating RIXS spectrum ...".to_string(),
                "Done with module: RIXS.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn sample_rixs_rl_dat(energy_count: usize, angular_limit: usize) -> XsphRlDatData {
        sample_rixs_rl_dat_with_radial_count(energy_count, angular_limit, 4)
    }

    fn sample_rixs_rl_dat_with_radial_count(
        energy_count: usize,
        angular_limit: usize,
        radial_count: usize,
    ) -> XsphRlDatData {
        let mut records = Vec::new();
        for energy in 0..energy_count {
            for angular in 0..=angular_limit {
                records.push(XsphRlDatRecord {
                    energy: Complex64::new(0.25 + energy as f64, 0.01 * energy as f64),
                    angular_momentum: angular,
                    phase_shift: Complex64::new(0.02 * (energy + 1) as f64, 0.01 * angular as f64),
                    regular_large: Array1::from_shape_fn(radial_count, |radial| {
                        Complex64::new(
                            0.1 * (energy + 1) as f64
                                + 0.01 * angular as f64
                                + 0.001 * radial as f64,
                            0.0,
                        )
                    }),
                    regular_small: Array1::from_shape_fn(radial_count, |radial| {
                        Complex64::new(
                            0.05 * (energy + 1) as f64
                                + 0.005 * angular as f64
                                + 0.0005 * radial as f64,
                            0.0,
                        )
                    }),
                });
            }
        }

        XsphRlDatData {
            muffin_tin_radius: 1.4,
            angular_limit,
            radial_match_index_1based: radial_count,
            log_step: 0.05,
            grid_origin: 8.8,
            records,
        }
    }

    fn sample_rixs_wscrn_dat(row_count: usize) -> WscrnDatData {
        WscrnDatData {
            header_lines: vec!["# r       w_scrn(r)      v_ch(r)".to_string()],
            radius_bohr: Array1::from_shape_fn(row_count, |row| 0.1 + 0.05 * row as f64),
            screened_potential: Array1::from_shape_fn(row_count, |row| -0.4 - 0.01 * row as f64),
            core_hole_potential: Array1::from_shape_fn(row_count, |row| 0.6 + 0.02 * row as f64),
        }
    }

    fn sample_rixs_vtot_dat(row_count: usize) -> VtotDatData {
        VtotDatData {
            header_lines: Vec::new(),
            radius_bohr: Array1::from_shape_fn(row_count, |row| 0.1 + 0.05 * row as f64),
            total_potential: Array1::from_shape_fn(row_count, |row| -1.2 - 0.1 * row as f64),
            screened_core_hole_potential: Array1::from_shape_fn(row_count, |row| {
                -0.4 - 0.01 * row as f64
            }),
        }
    }

    fn sample_zero_core_hole_apot_bin() -> ApotBinData {
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
                payload: ApotBinPayload::Records(ApotBinRecords {
                    column_types: vec![ApotBinType::Double; 4],
                    rows: (0..APOT_CORE_HOLE_RADIAL_POINTS)
                        .map(|_| vec![ApotBinValue::Real(0.0); 4])
                        .collect(),
                }),
                trailing_headers: Vec::new(),
                trailing_header_texts: Vec::new(),
            }],
        }
    }

    fn sample_rixs_xsect_dat(main_energy_count: usize, energy_count: usize) -> XsectDatData {
        XsectDatData {
            titles: vec!["sample RIXS xsect".to_string()],
            scalars: XsectDatScalars {
                amplitude_reduction: 1.0,
                relaxation_energy: 0.0,
                plasmon_frequency: 0.0,
                edge_energy: 8.0,
                chemical_potential: 0.2,
            },
            core_hole_width_ev: 0.15,
            main_energy_count,
            fermi_index: 1,
            energy_grid_ev: Array1::from_shape_fn(energy_count, |energy| {
                Complex64::new(10.0 + energy as f64, 0.01 * energy as f64)
            }),
            normalized_background: Array1::from_shape_fn(energy_count, |energy| {
                1.0 + 0.1 * energy as f64
            }),
            cross_section: Array1::from_shape_fn(energy_count, |energy| {
                Complex64::new(0.2 + 0.01 * energy as f64, -0.002 * energy as f64)
            }),
        }
    }

    fn sample_rixs_global_input() -> GlobalInput {
        GlobalInput {
            cfaverage: CfAverage {
                nabs: 1,
                iphabs: 0,
                rclabs: 100000.0,
            },
            control: GlobalControl {
                ipol: 1,
                ispin: 1,
                le2: 0,
                elpty: 0.0,
                angks: 0.0,
                l2lp: 0,
                do_nrixs: 0,
                ldecmx: -1,
                lj: -1,
            },
            evec: [0.0, 0.0, 1.0],
            xivec: [1.0, 0.0, 0.0],
            spvec: [0.0, 0.0, 1.0],
            polarization_tensor: [
                [0.20, -0.05, -0.10, 0.04, 0.03, 0.02],
                [0.11, -0.07, 0.50, 0.00, -0.08, 0.09],
                [0.06, 0.01, 0.13, -0.02, 0.17, 0.03],
            ],
            norms: GlobalNorms {
                evnorm: 1.0,
                xivnorm: 1.0,
                spvnorm: 1.0,
            },
            q_control: GlobalQControl {
                nq: 0,
                imdff: 0,
                qaverage: true,
                mixdff: false,
            },
            q_vectors: Vec::new(),
            mdff: None,
        }
    }

    fn sample_rixs_phase_bin(transition_count: usize) -> PhaseBinData {
        let spin_count = 1;
        let energy_count = 3;
        let q_count = 1;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: 3,
            auxiliary_energy_count: 0,
            ihole: 2,
            fermi_index: 1,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: 8,
            transition_count,
            q_count,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.2,
                fermi_level: -0.35,
                edge_energy: 9.8,
            },
            energy_grid: Array1::from_shape_fn(energy_count, |energy| {
                Complex64::new(0.5 + energy as f64, 0.01 * energy as f64)
            }),
            reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
                Complex64::new(-1.0 + 0.2 * energy as f64, 0.0)
            }),
            potentials: vec![sample_rixs_phase_potential(energy_count, spin_count)],
            transition_moments: Array4::from_shape_fn(
                (energy_count, q_count, transition_count, spin_count),
                |(energy, _, transition, _)| {
                    Complex64::new(0.01 * (energy + 1) as f64 + transition as f64, 0.0)
                },
            ),
            raw_pads: None,
        }
    }

    fn sample_rixs_mpse_phase_bin(transition_count: usize) -> PhaseBinData {
        let mut phase = sample_rixs_phase_bin(transition_count);
        for energy in 0..phase.energy_count {
            phase.reference_energy[(energy, 0)].im = 0.03 + 0.005 * energy as f64;
        }
        phase
    }

    fn sample_rixs_phase_potential(energy_count: usize, spin_count: usize) -> PhaseBinPotential {
        let lmax = 1;
        PhaseBinPotential {
            lmax,
            atomic_number: 29,
            label: "Cu".to_string(),
            phase_shifts: Array3::from_shape_fn(
                (energy_count, 2 * lmax + 1, spin_count),
                |(energy, signed_l, _)| {
                    Complex64::new(0.02 * (energy + 1) as f64 + 0.01 * signed_l as f64, 0.0)
                },
            ),
        }
    }

    fn sample_rixs_xsph_input() -> XsphInput {
        XsphInput {
            control: XsphControl {
                mphase: 1,
                ipr2: 0,
                ixc: 0,
                ixc0: 0,
                ispec: 0,
                lreal: 0,
                lfms2: 0,
                nph: 0,
                l2lp: 0,
                i_plsmn: 0,
                n_poles: 100,
                i_gamma_ch: 0,
                i_grid: 0,
                i_core_state: -1,
                iscfxc: 11,
            },
            vr0: 0.0,
            vi0: 0.0,
            lmaxph: vec![1],
            pot_labels: vec!["Cu".to_string()],
            grid: XsphGrid {
                rgrd: 0.05,
                rfms2: 0.0,
                gamach: 1.0,
                xkstep: 0.05,
                xkmax: 10.0,
                vixan: 0.0,
                eps0: 0.0,
                egap: 0.0,
            },
            spinph: vec![0.0],
            advanced: XsphAdvanced {
                izstd: 0,
                ifxc: 0,
                ipmbse: 0,
                itdlda: 0,
                nonlocal: 0,
                ibasis: 0,
            },
            electronic_temperature: 0.0,
            chsh_type: 0,
            decomposition_channels: -1,
            lopt: false,
            print_rl: false,
            source_format: XsphInputSourceFormat::modern(),
        }
    }

    fn sample_rixs_mpse_pot_bin() -> PotBinData {
        let potentials = 1;
        PotBinData {
            titles: vec!["RIXS XSPH MPSE source handoff".to_string()],
            pad_width: 8,
            nohole: 0,
            ihole: 2,
            interstitial_selector: 0,
            automatic_folp: 0,
            jump_mode: 0,
            unfreeze_f: 0,
            scalars: PotBinScalars {
                average_norman_radius: 1.2,
                fermi_level: -0.35,
                interstitial_potential: -0.75,
                interstitial_density: 0.08,
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
            muffin_tin_indices: Array1::from_vec(vec![20]),
            muffin_tin_radii: Array1::from_vec(vec![0.0123]),
            norman_indices: Array1::from_vec(vec![40]),
            atomic_numbers: Array1::from_vec(vec![29]),
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
            electron_density: Array2::from_elem((POT_BIN_RADIAL_POINTS, potentials), 0.08),
            coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            total_potential: Array2::from_elem((POT_BIN_RADIAL_POINTS, potentials), -0.5),
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

    fn reference_rixs_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/RIXS");
        let required = [
            "rixs.inp",
            "referencerixsET.dat",
            "referenceherfd.dat",
            "referenceherfd-sat.dat",
        ];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }

    fn reference_mpse_cu_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/MPSE/Cu/REFERENCE.zip");
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

    fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
        let path = path.as_ref();
        if path.is_file() {
            Ok(Some(read_module_log_dat(path)?))
        } else {
            Ok(None)
        }
    }

    fn assert_line_close(data: &RixsLineData, expected_energy: &[f64], expected: &[[f64; 2]]) {
        assert_vector_close(data.energy_ev.as_slice().unwrap(), expected_energy);
        assert_matrix_close(&data.channels, expected);
    }

    fn assert_matrix_close<const N: usize>(actual: &Array2<f64>, expected: &[[f64; N]]) {
        assert_eq!(actual.dim(), (expected.len(), N));
        for (row, expected_row) in expected.iter().enumerate() {
            for (col, &expected_value) in expected_row.iter().enumerate() {
                assert_close(actual[(row, col)], expected_value);
            }
        }
    }

    fn assert_vector_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual_value, &expected_value) in actual.iter().zip(expected.iter()) {
            assert_close(actual_value, expected_value);
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!(
            (actual - expected).norm() <= tolerance,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }
}
