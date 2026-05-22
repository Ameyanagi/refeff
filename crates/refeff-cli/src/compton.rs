use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use refeff_core::{
    ComptonGridInput as CoreComptonGridInput, ComptonWindow as CoreComptonWindow,
    compton_build_grid, compton_profiles,
};
use refeff_io::{
    ComptonDatData, ComptonInput, JzzpDatData, ModuleLogData, RhozzpDatData, read_jzzp_dat,
    read_module_log_dat, read_rhozzp_dat, write_compton_dat, write_jzzp_dat, write_module_log_dat,
    write_rhozzp_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF COMPTON profile stage beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether cached FEFF COMPTON outputs can satisfy the requested work.
pub(crate) fn has_cached_outputs(work_dir: &Path) -> Result<bool> {
    let input = read_input(work_dir)?;
    if !input.run || input.switches.force_recalc_jzzp {
        return Ok(false);
    }

    let has_profile_cache = !input.switches.jpq || work_dir.join("jzzp.dat").is_file();
    let has_rhozzp_cache = !input.switches.rhozzp || work_dir.join("rhozzp.dat").is_file();
    Ok(has_profile_cache && has_rhozzp_cache && (input.switches.jpq || input.switches.rhozzp))
}

/// Run the FEFF COMPTON cached-output path.
///
/// Profile output is generated from an existing `jzzp.dat` cache. The `jzzp.dat`
/// cache and requested `rhozzp.dat` diagnostics are validated and re-rendered
/// from cached text outputs when present, along with optional `logcompton.dat`
/// diagnostics. Rebuilding either cache from RHORRP density callbacks is still
/// outside the supported path.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.run {
        return Ok(0);
    }
    if input.switches.force_recalc_jzzp {
        bail!("COMPTON forced jzzp.dat recalculation requires the unported density callback path");
    }

    let rhozzp_rows = if input.switches.rhozzp {
        let rhozzp = read_cached_rhozzp(work_dir)?;
        let point_count = rhozzp.point_count();
        write_cached_rhozzp(work_dir, &rhozzp)?;
        point_count
    } else {
        0
    };

    let row_count = if input.switches.jpq {
        let cache = read_cached_jzzp(work_dir)?;
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

    write_optional_module_log(&work_dir.join("logcompton.dat"))?;
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

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
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
