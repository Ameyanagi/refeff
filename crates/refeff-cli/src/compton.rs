use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use refeff_core::{
    ComptonGrid as CoreComptonGrid, ComptonGridInput as CoreComptonGridInput, ComptonRhoZzpInput,
    ComptonWindow as CoreComptonWindow, FEFF_HARTREE_EV, compton_build_grid,
    compton_jzzp_from_rhorrp, compton_profiles, compton_rhozzp_slice_from_rhorrp,
};
use refeff_io::{
    ComptonDatData, ComptonInput, JzzpDatData, ModuleLogData, RhozzpDatData, jzzp_dat_string,
    read_jzzp_dat, read_module_log_dat, read_rhozzp_dat, rhozzp_dat_string, write_compton_dat,
    write_jzzp_dat, write_module_log_dat, write_rhozzp_dat,
};

use crate::{rhorrp, work_dir_for_input};

const CACHE_GRID_TOLERANCE: f64 = 1.0e-6;
const RHOZZP_SAMPLE_COUNT: usize = 1000;
const RHOZZP_BASE_Z_BOHR: f64 = 0.01;

/// Run the supported FEFF COMPTON profile stage beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether cached outputs or RHORRP handoffs can satisfy the requested work.
pub(crate) fn has_supported_outputs(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("compton.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !input.run {
        return Ok(false);
    }
    if (input.switches.jpq || input.switches.rhozzp)
        && rhorrp::validate_declared_rhorrp_density_callback_source(work_dir).is_err()
    {
        return Ok(false);
    }

    let has_density_callback = rhorrp::has_rhorrp_density_callback_source(work_dir);
    let has_profile_source = !input.switches.jpq
        || has_density_callback
        || (!input.switches.force_recalc_jzzp && has_compatible_jzzp_cache(work_dir, &input));
    let has_rhozzp_source =
        !input.switches.rhozzp || has_density_callback || has_parseable_rhozzp_cache(work_dir);
    Ok(has_profile_source && has_rhozzp_source && (input.switches.jpq || input.switches.rhozzp))
}

fn has_compatible_jzzp_cache(work_dir: &Path, input: &ComptonInput) -> bool {
    let path = work_dir.join("jzzp.dat");
    if !path.is_file() {
        return false;
    }
    match read_jzzp_dat(&path) {
        Ok(cache) => validate_cache_matches_input(input, &cache).is_ok(),
        Err(_) => false,
    }
}

fn has_parseable_rhozzp_cache(work_dir: &Path) -> bool {
    let path = work_dir.join("rhozzp.dat");
    path.is_file() && read_rhozzp_dat(&path).is_ok()
}

/// Run the FEFF COMPTON output path from caches or RHORRP handoff files.
///
/// Profile output is generated from an existing `jzzp.dat` cache. The `jzzp.dat`
/// cache and requested `rhozzp.dat` diagnostics are validated and re-rendered
/// from cached text outputs when present, along with `logcompton.dat`. Missing,
/// malformed, forced, or readable-but-stale COMPTON caches are generated from
/// the ported RHORRP density callback when the required handoff files are
/// present.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.run {
        return Ok(0);
    }
    if input.switches.jpq || input.switches.rhozzp {
        rhorrp::validate_declared_rhorrp_density_callback_source(work_dir)?;
    }
    let mut rhorrp_source = None;
    let mut jzzp_log_mode = JzzpLogMode::None;

    let rhozzp_rows = if input.switches.rhozzp {
        let rhozzp = if work_dir.join("rhozzp.dat").is_file() {
            match read_cached_rhozzp(work_dir) {
                Ok(rhozzp) => regenerate_stale_rhozzp_from_source_handoff(
                    work_dir,
                    &input,
                    &mut rhorrp_source,
                    rhozzp,
                )?,
                Err(error) => recover_malformed_rhozzp_from_source_handoff(
                    work_dir,
                    &input,
                    &mut rhorrp_source,
                    error,
                )?,
            }
        } else if rhorrp::has_rhorrp_density_callback_source(work_dir) {
            let source = load_rhorrp_source(work_dir, &mut rhorrp_source)?;
            generate_rhozzp(&input, source)?
        } else {
            read_cached_rhozzp(work_dir)?
        };
        let point_count = rhozzp.point_count();
        write_cached_rhozzp(work_dir, &rhozzp)?;
        point_count
    } else {
        0
    };

    let row_count = if input.switches.jpq {
        let (cache, generated) = read_or_generate_jzzp(work_dir, &input, &mut rhorrp_source)?;
        jzzp_log_mode = if generated {
            JzzpLogMode::Saved
        } else {
            JzzpLogMode::Reused
        };
        let profile = calculate_profile(&input, &cache)?;
        write_cached_jzzp(work_dir, &cache)?;
        let point_count = profile.point_count();
        let output_path = work_dir.join("compton.dat");
        write_compton_dat(&output_path, &profile)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        point_count + rhozzp_rows
    } else {
        rhozzp_rows
    };

    write_or_generate_module_log(&work_dir.join("logcompton.dat"), &input, jzzp_log_mode)?;
    Ok(row_count)
}

fn read_input(work_dir: &Path) -> Result<ComptonInput> {
    let input_path = work_dir.join("compton.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    ComptonInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn calculate_profile(input: &ComptonInput, cache: &JzzpDatData) -> Result<ComptonDatData> {
    validate_cache_matches_input(input, cache)?;
    let grid = compton_build_grid(CoreComptonGridInput {
        ns: cache.ns,
        nphi: cache.nphi,
        nz: cache.nz,
        nzp: cache.nzp,
        smax: cache.smax,
        phimax: cache.phimax,
        zmax: cache.zmax,
        zpmax: cache.zpmax,
        norman_radius: 0.0,
        qhat: input.qhat,
    })
    .context("failed to build COMPTON integration grid from jzzp.dat")?;
    let window = profile_window(input.window.window_type);
    let momentum = compton_momentum_grid(input)?;
    let profile = compton_profiles(
        &grid,
        cache.values.view(),
        momentum.view(),
        window,
        input.window.cutoff,
    )
    .context("failed to evaluate COMPTON profile")?;

    Ok(ComptonDatData {
        header_lines: compton_dat_header(input, cache),
        ns: Some(cache.ns),
        nphi: Some(cache.nphi),
        nz: Some(cache.nz),
        nzp: Some(cache.nzp),
        zpmax: Some(cache.zpmax),
        temperature_ev: Some(input.temperature),
        momentum,
        profile,
    })
}

fn read_or_generate_jzzp(
    work_dir: &Path,
    input: &ComptonInput,
    rhorrp_source: &mut Option<rhorrp::TableDensitySource>,
) -> Result<(JzzpDatData, bool)> {
    let cache_path = work_dir.join("jzzp.dat");
    if !input.switches.force_recalc_jzzp && cache_path.is_file() {
        let cache = match read_cached_jzzp(work_dir) {
            Ok(cache) => cache,
            Err(error) => {
                return Ok((
                    recover_malformed_jzzp_from_source_handoff(
                        work_dir,
                        input,
                        rhorrp_source,
                        error,
                    )?,
                    true,
                ));
            }
        };
        match validate_cache_matches_input(input, &cache) {
            Ok(()) => {
                return regenerate_stale_jzzp_from_source_handoff(
                    work_dir,
                    input,
                    rhorrp_source,
                    cache,
                );
            }
            Err(_) if rhorrp::has_rhorrp_density_callback_source(work_dir) => {
                let source = load_rhorrp_source(work_dir, rhorrp_source)?;
                return Ok((generate_jzzp(input, source)?, true));
            }
            Err(error) => return Err(error),
        }
    }

    if rhorrp::has_rhorrp_density_callback_source(work_dir) {
        let source = load_rhorrp_source(work_dir, rhorrp_source)?;
        Ok((generate_jzzp(input, source)?, true))
    } else if input.switches.force_recalc_jzzp {
        bail!(
            "COMPTON forced jzzp.dat recalculation requires RHORRP density callback handoff files"
        )
    } else {
        Ok((read_cached_jzzp(work_dir)?, false))
    }
}

fn regenerate_stale_jzzp_from_source_handoff(
    work_dir: &Path,
    input: &ComptonInput,
    rhorrp_source: &mut Option<rhorrp::TableDensitySource>,
    cache: JzzpDatData,
) -> Result<(JzzpDatData, bool)> {
    if !rhorrp::has_rhorrp_density_callback_source(work_dir) {
        return Ok((cache, false));
    }

    let source = match load_rhorrp_source(work_dir, rhorrp_source) {
        Ok(source) => source,
        Err(_) => return Ok((cache, false)),
    };
    let generated = match generate_jzzp(input, source) {
        Ok(generated) => generated,
        Err(_) => return Ok((cache, false)),
    };
    if jzzp_dat_string(&cache)? == jzzp_dat_string(&generated)? {
        Ok((cache, false))
    } else {
        Ok((generated, true))
    }
}

fn recover_malformed_jzzp_from_source_handoff(
    work_dir: &Path,
    input: &ComptonInput,
    rhorrp_source: &mut Option<rhorrp::TableDensitySource>,
    cache_error: anyhow::Error,
) -> Result<JzzpDatData> {
    if !rhorrp::has_rhorrp_density_callback_source(work_dir) {
        return Err(cache_error);
    }

    let source = load_rhorrp_source(work_dir, rhorrp_source).with_context(|| {
        format!(
            "failed to recover malformed jzzp.dat from RHORRP source handoff after cache read failed: {cache_error:#}"
        )
    })?;
    generate_jzzp(input, source).with_context(|| {
        format!(
            "failed to recover malformed jzzp.dat from RHORRP source handoff after cache read failed: {cache_error:#}"
        )
    })
}

fn generate_jzzp(input: &ComptonInput, source: &rhorrp::TableDensitySource) -> Result<JzzpDatData> {
    let grid = build_compton_grid(input, source.central_norman_radius_bohr())?;
    let values = compton_jzzp_from_rhorrp(
        &grid,
        source.compton_density_input(chemical_potential_override_hartree(input)?)?,
    )
    .context("failed to generate COMPTON jzzp.dat from RHORRP density callback")?;

    Ok(JzzpDatData {
        ns: grid.ns(),
        nphi: grid.nphi(),
        nz: grid.nz(),
        nzp: grid.nzp(),
        smax: grid_extent(&grid.s),
        phimax: grid_extent(&grid.phi),
        zmax: symmetric_grid_extent(&grid.z),
        zpmax: symmetric_grid_extent(&grid.zp),
        values,
    })
}

fn generate_rhozzp(
    input: &ComptonInput,
    source: &rhorrp::TableDensitySource,
) -> Result<RhozzpDatData> {
    let grid = build_compton_grid(input, source.central_norman_radius_bohr())?;
    let slice = compton_rhozzp_slice_from_rhorrp(
        &grid,
        ComptonRhoZzpInput {
            sample_count: RHOZZP_SAMPLE_COUNT,
            base_z: RHOZZP_BASE_Z_BOHR,
        },
        source.compton_density_input(chemical_potential_override_hartree(input)?)?,
    )
    .context("failed to generate COMPTON rhozzp.dat from RHORRP density callback")?;

    Ok(RhozzpDatData {
        header_lines: Vec::new(),
        z_prime: slice.z_prime,
        density: slice.rho,
    })
}

fn regenerate_stale_rhozzp_from_source_handoff(
    work_dir: &Path,
    input: &ComptonInput,
    rhorrp_source: &mut Option<rhorrp::TableDensitySource>,
    cache: RhozzpDatData,
) -> Result<RhozzpDatData> {
    if !rhorrp::has_rhorrp_density_callback_source(work_dir) {
        return Ok(cache);
    }

    let source = match load_rhorrp_source(work_dir, rhorrp_source) {
        Ok(source) => source,
        Err(_) => return Ok(cache),
    };
    let generated = match generate_rhozzp(input, source) {
        Ok(generated) => generated,
        Err(_) => return Ok(cache),
    };
    if rhozzp_dat_string(&cache)? == rhozzp_dat_string(&generated)? {
        Ok(cache)
    } else {
        Ok(generated)
    }
}

fn recover_malformed_rhozzp_from_source_handoff(
    work_dir: &Path,
    input: &ComptonInput,
    rhorrp_source: &mut Option<rhorrp::TableDensitySource>,
    cache_error: anyhow::Error,
) -> Result<RhozzpDatData> {
    if !rhorrp::has_rhorrp_density_callback_source(work_dir) {
        return Err(cache_error);
    }

    let source = load_rhorrp_source(work_dir, rhorrp_source).with_context(|| {
        format!(
            "failed to recover malformed rhozzp.dat from RHORRP source handoff after cache read failed: {cache_error:#}"
        )
    })?;
    generate_rhozzp(input, source).with_context(|| {
        format!(
            "failed to recover malformed rhozzp.dat from RHORRP source handoff after cache read failed: {cache_error:#}"
        )
    })
}

fn build_compton_grid(input: &ComptonInput, norman_radius: f64) -> Result<CoreComptonGrid> {
    compton_build_grid(CoreComptonGridInput {
        ns: positive_i32_to_usize("ns", input.grid.ns)?,
        nphi: positive_i32_to_usize("nphi", input.grid.nphi)?,
        nz: positive_i32_to_usize("nz", input.grid.nz)?,
        nzp: positive_i32_to_usize("nzp", input.grid.nzp)?,
        smax: input.limits.smax,
        phimax: input.limits.phimax,
        zmax: input.limits.zmax,
        zpmax: input.limits.zpmax,
        norman_radius,
        qhat: input.qhat,
    })
    .context("failed to build COMPTON integration grid from compton.inp")
}

fn load_rhorrp_source<'a>(
    work_dir: &Path,
    source: &'a mut Option<rhorrp::TableDensitySource>,
) -> Result<&'a rhorrp::TableDensitySource> {
    if source.is_none() {
        *source = Some(rhorrp::read_rhorrp_density_callback_source(work_dir)?);
    }
    source
        .as_ref()
        .context("missing RHORRP density callback source after loading")
}

fn chemical_potential_override_hartree(input: &ComptonInput) -> Result<Option<f64>> {
    if !input.chemical_potential.enabled {
        return Ok(None);
    }
    if !input.chemical_potential.value.is_finite() {
        bail!(
            "COMPTON chemical potential must be finite, got {}",
            input.chemical_potential.value
        );
    }
    Ok(Some(input.chemical_potential.value / FEFF_HARTREE_EV))
}

fn grid_extent(values: &Array1<f64>) -> f64 {
    values.last().copied().unwrap_or(0.0)
}

fn symmetric_grid_extent(values: &Array1<f64>) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn read_cached_jzzp(work_dir: &Path) -> Result<JzzpDatData> {
    let path = work_dir.join("jzzp.dat");
    read_jzzp_dat(&path).with_context(|| format!("failed to read {}", path.display()))
}

fn write_cached_jzzp(work_dir: &Path, data: &JzzpDatData) -> Result<()> {
    let path = work_dir.join("jzzp.dat");
    write_jzzp_dat(&path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn read_cached_rhozzp(work_dir: &Path) -> Result<RhozzpDatData> {
    let path = work_dir.join("rhozzp.dat");
    read_rhozzp_dat(&path).with_context(|| format!("failed to read {}", path.display()))
}

fn write_cached_rhozzp(work_dir: &Path, data: &RhozzpDatData) -> Result<()> {
    let path = work_dir.join("rhozzp.dat");
    write_rhozzp_dat(&path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_generate_module_log(
    path: &Path,
    input: &ComptonInput,
    jzzp_log_mode: JzzpLogMode,
) -> Result<()> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_module_log(path, &generated_compton_module_log(input, jzzp_log_mode))
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JzzpLogMode {
    None,
    Reused,
    Saved,
}

fn generated_compton_module_log(input: &ComptonInput, jzzp_log_mode: JzzpLogMode) -> ModuleLogData {
    let mut lines = vec![
        "Calculating Compton scattering ...".to_string(),
        "FEFF-serial using 1 thread.".to_string(),
    ];

    if input.switches.rhozzp {
        lines.push("Calculating rho(z,z')".to_string());
    }

    if input.switches.jpq {
        lines.push("Calculating Compton profile".to_string());
        match jzzp_log_mode {
            JzzpLogMode::None => {}
            JzzpLogMode::Reused => {
                lines.push("Reusing previously calculated j(z,z')".to_string());
            }
            JzzpLogMode::Saved => {
                lines.push("Saving j(z,z')".to_string());
            }
        }
        lines.push("Calculate j(pq)".to_string());
    }

    lines.push("Done with module: Compton scattering.".to_string());
    let line_terminators = vec!["\n".to_string(); lines.len()];
    ModuleLogData {
        lines,
        line_terminators,
    }
}

fn validate_cache_matches_input(input: &ComptonInput, cache: &JzzpDatData) -> Result<()> {
    let ns = positive_i32_to_usize("ns", input.grid.ns)?;
    let nphi = positive_i32_to_usize("nphi", input.grid.nphi)?;
    let nz = positive_i32_to_usize("nz", input.grid.nz)?;
    let nzp = positive_i32_to_usize("nzp", input.grid.nzp)?;
    if (ns, nphi, nz, nzp) != (cache.ns, cache.nphi, cache.nz, cache.nzp) {
        bail!(
            "COMPTON jzzp.dat grid ({}, {}, {}, {}) does not match compton.inp grid ({ns}, {nphi}, {nz}, {nzp})",
            cache.ns,
            cache.nphi,
            cache.nz,
            cache.nzp
        );
    }
    validate_cache_limit_matches_input("smax", input.limits.smax, cache.smax)?;
    validate_cache_limit_matches_input("phimax", input.limits.phimax, cache.phimax)?;
    validate_cache_limit_matches_input("zmax", input.limits.zmax, cache.zmax)?;
    validate_cache_limit_matches_input("zpmax", input.limits.zpmax, cache.zpmax)?;
    Ok(())
}

fn validate_cache_limit_matches_input(name: &'static str, input: f64, cache: f64) -> Result<()> {
    if input == 0.0 {
        return Ok(());
    }
    if (input - cache).abs() >= CACHE_GRID_TOLERANCE {
        return Err(anyhow::anyhow!(
            "COMPTON cached jzzp.dat {name} ({cache}) differs from compton.inp {name} ({input}); recalculation requires RHORRP density callback handoff files"
        ));
    }
    Ok(())
}

fn positive_i32_to_usize(name: &'static str, value: i32) -> Result<usize> {
    if value <= 0 {
        bail!("COMPTON {name} must be positive, got {value}");
    }
    usize::try_from(value).with_context(|| format!("COMPTON {name} is out of range: {value}"))
}

fn compton_momentum_grid(input: &ComptonInput) -> Result<Array1<f64>> {
    if !input.momentum.pqmax.is_finite() || input.momentum.pqmax < 0.0 {
        bail!(
            "COMPTON pqmax must be finite and nonnegative, got {}",
            input.momentum.pqmax
        );
    }
    let npq = positive_i32_to_usize("npq", input.momentum.npq)?;
    if npq < 2 {
        bail!("COMPTON npq must be at least 2, got {npq}");
    }
    let scale = input.momentum.pqmax / (npq - 1) as f64;
    Ok(Array1::from_iter(
        (0..npq).map(|index| index as f64 * scale),
    ))
}

fn profile_window(window_type: i32) -> CoreComptonWindow {
    match window_type {
        0 => CoreComptonWindow::Rectangular,
        1 => CoreComptonWindow::CosineSquared,
        _ => CoreComptonWindow::Unwindowed,
    }
}

fn compton_dat_header(input: &ComptonInput, cache: &JzzpDatData) -> Vec<String> {
    vec![
        " # Compton profile, J(pq)".to_string(),
        format!(" # ns:          {:4}", cache.ns),
        format!(" # nphi:        {:4}", cache.nphi),
        format!(" # nz:          {:4}", cache.nz),
        format!(" # nzp:         {:4}", cache.nzp),
        format!(" # zpmax:   {:17.13}", cache.zpmax),
        format!(" # temperature (eV): {:14.7E}", input.temperature),
        " #----------------------------".to_string(),
        " # pq               J".to_string(),
    ]
}

#[cfg(test)]
mod tests;
