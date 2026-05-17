#![forbid(unsafe_code)]

mod atomic;
mod band;
mod compton;
mod crpa;
mod dmdw;
mod eels;
mod eelsmdff;
mod ff2x;
mod fms;
mod fullspectrum;
mod genfmt;
mod ldos;
mod opcons;
mod paths;
mod rhorrp;
mod rixs;
mod screen;
mod sfconv;
mod wpot;
mod xsph;

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use refeff_io::{FeffDocument, FeffInput, rdinp};

#[derive(Debug, Parser)]
#[command(
    name = "refeff",
    version,
    about = "Pure-Rust FEFF10 compatibility port"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Inspect {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Rdinp {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Run {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Module {
        name: String,
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
}

/// Summary of the parsed input handled by the `rdinp` compatibility stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdinpReport {
    /// Number of active FEFF cards parsed from the input.
    pub cards: usize,
    /// Number of atoms extracted from the `ATOMS` table.
    pub atoms: usize,
    /// Number of unique potential rows extracted from `POTENTIALS`.
    pub potentials: usize,
    /// FEFF-style RDINP stdout summary, when currently renderable.
    pub stdout: Option<String>,
}

/// Dispatch a parsed `refeff` command.
pub fn run_cli(cli: Cli) -> Result<()> {
    let command = match cli.command {
        Some(command) => command,
        None => Command::Run {
            input: PathBuf::from("feff.inp"),
        },
    };
    match command {
        Command::Inspect { input } => inspect(input),
        Command::Rdinp { input } => run_rdinp(input),
        Command::Run { input } => run_feff(input),
        Command::Module { name, input } => run_module(&name, input),
    }
}

fn inspect(input: PathBuf) -> Result<()> {
    let parsed = FeffInput::parse_file(&input)?;
    let cards = parsed.cards().count();
    let atom_rows = parsed.section_rows("ATOMS").count();
    let potential_rows = parsed.section_rows("POTENTIALS").count();
    println!(
        "input={} cards={} atoms={} potentials={}",
        input.display(),
        cards,
        atom_rows,
        potential_rows
    );
    Ok(())
}

/// Run the supported FEFF `rdinp` compatibility stage in the current directory.
pub fn run_rdinp(input: PathBuf) -> Result<()> {
    let report = execute_rdinp(&input, Path::new("."))?;
    let mut stdout = std::io::stdout().lock();
    if let Some(summary) = &report.stdout {
        stdout.write_all(summary.as_bytes())?;
    } else {
        writeln!(
            stdout,
            "rdinp: parsed {} cards, {} atoms, {} potentials",
            report.cards, report.atoms, report.potentials
        )?;
    }
    Ok(())
}

/// Run the supported FEFF `pot` compatibility stage in the input directory.
pub fn run_pot(input: PathBuf) -> Result<()> {
    run_potential_output_module("pot", input)
}

/// Run the supported FEFF `atomic` compatibility stage in the input directory.
pub fn run_atomic(input: PathBuf) -> Result<()> {
    run_atomic_module(input)
}

/// Run the supported FEFF `band` compatibility stage in the input directory.
pub fn run_band(input: PathBuf) -> Result<()> {
    run_band_module(input)
}

/// Run the supported FEFF `mdff` compatibility stage in the input directory.
pub fn run_mdff(input: PathBuf) -> Result<()> {
    run_mdff_module(input)
}

fn run_feff(input: PathBuf) -> Result<()> {
    run_feff_to_dir(&input, Path::new("."))
}

fn run_feff_to_dir(input: &Path, output_dir: &Path) -> Result<()> {
    let report = execute_rdinp(input, output_dir)?;
    let module_reports = run_supported_cached_modules(output_dir)?;
    bail!(
        "full FEFF numerical execution is not implemented yet; completed rdinp for {} cards, {} atoms, {} potentials from {}; {}",
        report.cards,
        report.atoms,
        report.potentials,
        input.display(),
        supported_module_summary(&module_reports)
    )
}

fn run_module(name: &str, input: PathBuf) -> Result<()> {
    if name.eq_ignore_ascii_case("rdinp") {
        return run_rdinp(input);
    }
    if name.eq_ignore_ascii_case("pot") {
        return run_pot(input);
    }
    if name.eq_ignore_ascii_case("atomic") || name.eq_ignore_ascii_case("atom") {
        return run_atomic(input);
    }
    if name.eq_ignore_ascii_case("band") {
        return run_band(input);
    }
    if name.eq_ignore_ascii_case("mdff") || name.eq_ignore_ascii_case("eelsmdff") {
        return run_mdff(input);
    }
    if name.eq_ignore_ascii_case("wpot") {
        return run_potential_output_module("wpot", input);
    }
    if name.eq_ignore_ascii_case("opcons") || name.eq_ignore_ascii_case("opconsat") {
        let count = opcons::run_for_input(&input)?;
        println!(
            "opcons: wrote loss.dat with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("compton") {
        let count = compton::run_for_input(&input)?;
        println!(
            "compton: wrote cached output with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("fullspectrum") {
        let count = fullspectrum::run_for_input(&input)?;
        println!(
            "fullspectrum: wrote optical constants with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("crpa") {
        let count = crpa::run_for_input(&input)?;
        println!(
            "crpa: wrote crpa.dat with {count} result row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("screen") {
        let count = screen::run_for_input(&input)?;
        println!(
            "screen: wrote cached output with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("ldos") {
        let count = ldos::run_for_input(&input)?;
        println!(
            "ldos: validated {count} cached output file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("eels") {
        let count = eels::run_for_input(&input)?;
        println!(
            "eels: wrote eels.dat with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("dmdw") {
        let count = dmdw::run_for_input(&input)?;
        println!(
            "dmdw: wrote dmdw.out with {count} section(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("path") || name.eq_ignore_ascii_case("paths") {
        let count = paths::run_for_input(&input)?;
        println!(
            "path: wrote paths.dat with {count} path(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("genfmt") {
        let count = genfmt::run_for_input(&input)?;
        println!(
            "genfmt: validated {count} cached output file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("ff2x") {
        let count = ff2x::run_for_input(&input)?;
        println!(
            "ff2x: validated {count} cached spectrum file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("xsph") {
        let count = xsph::run_for_input(&input)?;
        println!(
            "xsph: validated {count} cached output file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("fms") || name.eq_ignore_ascii_case("mkgtr") {
        let count = fms::run_for_input(&input)?;
        println!(
            "fms: validated {count} cached Green's-function file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("rixs") {
        let count = rixs::run_for_input(&input)?;
        println!(
            "rixs: validated {count} cached spectrum file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("rhorrp") {
        let count = rhorrp::run_for_input(&input)?;
        println!(
            "rhorrp: validated {count} cached density file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("sfconv") {
        sfconv::run_for_input(&input)?;
        println!("sfconv: wrote logsfconv.dat beside {}", input.display());
        return Ok(());
    }
    if name.eq_ignore_ascii_case("self") || name.eq_ignore_ascii_case("selfenergy") {
        let count = sfconv::run_self_for_input(&input)?;
        println!(
            "self: validated {count} excitation pole(s) beside {}",
            input.display()
        );
        return Ok(());
    }

    let parsed = FeffInput::parse_file(&input)?;
    bail!(
        "module {name} is not implemented yet; parsed {} active lines from {}",
        parsed.lines.len(),
        input.display()
    )
}

fn run_potential_output_module(label: &str, input: PathBuf) -> Result<()> {
    let count = wpot::run_for_input(&input)?;
    println!(
        "{label}: wrote {count} potential output file(s) beside {}",
        input.display()
    );
    Ok(())
}

fn run_atomic_module(input: PathBuf) -> Result<()> {
    let count = atomic::run_for_input(&input)?;
    println!(
        "atomic: validated {count} cached atomic output file(s) beside {}",
        input.display()
    );
    Ok(())
}

fn run_band_module(input: PathBuf) -> Result<()> {
    let count = band::run_for_input(&input)?;
    println!(
        "band: validated {count} cached band output file(s) beside {}",
        input.display()
    );
    Ok(())
}

fn run_mdff_module(input: PathBuf) -> Result<()> {
    let count = eelsmdff::run_for_input(&input)?;
    println!(
        "mdff: validated {count} cached EELS-MDFF row(s) beside {}",
        input.display()
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SupportedModuleReport {
    name: &'static str,
    count: usize,
    unit: &'static str,
}

fn run_supported_cached_modules(work_dir: &Path) -> Result<Vec<SupportedModuleReport>> {
    let mut reports = Vec::new();
    if atomic::has_cached_atomic_output(work_dir)? {
        let count = atomic::run_in_dir(work_dir).context("failed to run supported atomic stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "atomic",
                count,
                unit: "file(s)",
            });
        }
    }

    if work_dir.join("pot.bin").is_file() && work_dir.join("apot.bin").is_file() {
        reports.push(SupportedModuleReport {
            name: "wpot",
            count: wpot::run_in_dir(work_dir).context("failed to run supported wpot stage")?,
            unit: "file(s)",
        });
    }

    if xsph::has_cached_xsph_output(work_dir)? {
        let count = xsph::run_in_dir(work_dir).context("failed to run supported xsph stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "xsph",
                count,
                unit: "file(s)",
            });
        }
    }

    if fms::has_cached_fms_output(work_dir)? {
        let count = fms::run_in_dir(work_dir).context("failed to run supported fms stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "fms",
                count,
                unit: "file(s)",
            });
        }
    }

    if band::has_cached_band_output(work_dir)? {
        let count = band::run_in_dir(work_dir).context("failed to run supported band stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "band",
                count,
                unit: "file(s)",
            });
        }
    }

    if rixs::has_cached_rixs_output(work_dir)? {
        let count = rixs::run_in_dir(work_dir).context("failed to run supported rixs stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "rixs",
                count,
                unit: "file(s)",
            });
        }
    }

    if rhorrp::has_cached_rhorrp_output(work_dir)? {
        let count = rhorrp::run_in_dir(work_dir).context("failed to run supported rhorrp stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "rhorrp",
                count,
                unit: "file(s)",
            });
        }
    }

    if opcons::has_complete_table_inputs(work_dir)? {
        let count = opcons::run_in_dir(work_dir).context("failed to run supported opcons stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "opcons",
                count,
                unit: "row(s)",
            });
        }
    }

    if compton::has_cached_outputs(work_dir)? {
        let count =
            compton::run_in_dir(work_dir).context("failed to run supported compton stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "compton",
                count,
                unit: "row(s)",
            });
        }
    }

    if fullspectrum::has_cached_optical_inputs(work_dir)? {
        let count = fullspectrum::run_in_dir(work_dir)
            .context("failed to run supported fullspectrum stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "fullspectrum",
                count,
                unit: "row(s)",
            });
        }
    }

    if crpa::has_cached_crpa_output(work_dir)? {
        let count = crpa::run_in_dir(work_dir).context("failed to run supported crpa stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "crpa",
                count,
                unit: "row(s)",
            });
        }
    }

    if screen::has_cached_screen_output(work_dir)? {
        let count = screen::run_in_dir(work_dir).context("failed to run supported screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
            });
        }
    }

    if ldos::has_cached_ldos_output(work_dir)? {
        let count = ldos::run_in_dir(work_dir).context("failed to run supported ldos stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
            });
        }
    }

    if eels::has_cached_eels_output(work_dir)? {
        let count = eels::run_in_dir(work_dir).context("failed to run supported eels stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "eels",
                count,
                unit: "row(s)",
            });
        }
    }

    if eelsmdff::has_cached_mdff_output(work_dir)? {
        let count =
            eelsmdff::run_in_dir(work_dir).context("failed to run supported eelsmdff stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "eelsmdff",
                count,
                unit: "row(s)",
            });
        }
    }

    if dmdw::has_cached_dmdw_output(work_dir)? {
        let count = dmdw::run_in_dir(work_dir).context("failed to run supported dmdw stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "dmdw",
                count,
                unit: "section(s)",
            });
        }
    }

    if paths::has_cached_paths_output(work_dir)? {
        let count = paths::run_in_dir(work_dir).context("failed to run supported path stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "path",
                count,
                unit: "path(s)",
            });
        }
    }

    if genfmt::has_cached_genfmt_output(work_dir)? {
        let count = genfmt::run_in_dir(work_dir).context("failed to run supported genfmt stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "genfmt",
                count,
                unit: "file(s)",
            });
        }
    }

    if ff2x::has_cached_ff2x_output(work_dir)? {
        let count = ff2x::run_in_dir(work_dir).context("failed to run supported ff2x stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ff2x",
                count,
                unit: "file(s)",
            });
        }
    }

    if sfconv::has_cached_self_output(work_dir)? {
        let count =
            sfconv::run_self_in_dir(work_dir).context("failed to run supported self stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "self",
                count,
                unit: "pole(s)",
            });
        }
    }

    Ok(reports)
}

fn supported_module_summary(reports: &[SupportedModuleReport]) -> String {
    if reports.is_empty() {
        return "no supported cached stages were run".to_string();
    }

    let details = reports
        .iter()
        .map(|report| format!("{}={} {}", report.name, report.count, report.unit))
        .collect::<Vec<_>>()
        .join(", ");
    format!("supported cached stages run: {details}")
}

pub(crate) fn work_dir_for_input(input: &Path) -> &Path {
    match input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        Some(parent) => parent,
        None => Path::new("."),
    }
}

fn execute_rdinp(input: &Path, output_dir: &Path) -> Result<RdinpReport> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let error_sentinel = output_dir.join(".feff.error");
    std::fs::write(&error_sentinel, rdinp::rdinp_error_sentinel_string())
        .with_context(|| format!("failed to write {}", error_sentinel.display()))?;

    let parsed = FeffInput::parse_file(input)?;
    let document = match FeffDocument::from_input(&parsed) {
        Ok(document) => document,
        Err(error) => {
            if let Ok(content) = rdinp::rdinp_error_log_string(&parsed, &error) {
                let output_path = output_dir.join("log.dat");
                std::fs::write(&output_path, content)
                    .with_context(|| format!("failed to write {}", output_path.display()))?;
            }
            return Err(error.into());
        }
    };
    let outputs = rdinp::text_outputs(&document)?;
    let log_dat = rdinp::rdinp_log_dat_string(&document).ok();
    let stdout = rdinp::rdinp_stdout_string(&document).ok();
    for (name, content) in outputs {
        let output_path = output_dir.join(name.as_ref());
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    if let Some(content) = &log_dat {
        let output_path = output_dir.join("log.dat");
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    match std::fs::remove_file(&error_sentinel) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to remove {}", error_sentinel.display()));
        }
    }

    Ok(RdinpReport {
        cards: parsed.cards().count(),
        atoms: document.atoms.len(),
        potentials: document.potentials.len(),
        stdout,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        atomic, band, eelsmdff, execute_rdinp, opcons, paths, run_feff_to_dir, run_module, wpot,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4};
    use num_complex::Complex64;
    use refeff_io::feff_bin::{FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH};
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::rdinp;
    use refeff_io::{
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, BandstructureDatData, BandstructureRow, ChiDatData, CrpaDatData, DanesDatData,
        DmdwOutData, DmdwOutHeader, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature,
        EelsDatData, EmeshBinData, EmeshDatData, EpsDatData, ExcDatData, FeffBinData, FeffBinPath,
        FeffBinPotential, FeffDocument, FeffInput, FmsBinData, Fort16Data, HamakerDatData,
        JzzpDatData, LdosDatData, ListDatData, ListDatEntry, MdffDatData, MiscDatData,
        ModuleLogData, MpseDatData, OscStrDatData, OscStrRow, PhaseBinData, PhaseBinPotential,
        PhaseBinScalars, PotBinData, PotBinScalars, RhorrpDensityTextData,
        RhorrpNearestAtomColumns, RhozzpDatData, RixsMapData, ScfConvergenceData,
        ScfConvergenceLine, ScfConvergenceRow, VtotDatData, WscrnDatData, XmuDatData,
        XscorrComplexTable, XscorrCurveDatData, XscorrRawDatData, XsectDatData, XsectDatScalars,
        parse_loss_dat, read_apot_bin, read_bandstructure_dat, read_chi_dat, read_compton_dat,
        read_contour_dat, read_convergence_scf, read_convergence_scf_fine, read_crpa_dat,
        read_curve_dat, read_danes_dat, read_dmdw_out, read_eels_dat, read_emesh_bin,
        read_emesh_dat, read_exc_dat, read_feff_bin, read_fms_bin, read_fort16, read_hamaker_dat,
        read_jzzp_dat, read_ldos_dat, read_list_dat, read_mdff_dat, read_misc_dat,
        read_module_log_dat, read_mpse_dat, read_opcons_dat, read_osc_str_dat, read_paths_dat,
        read_phase_bin, read_prexmu_dat, read_residue_dat, read_rhorrp_density_text,
        read_rhozzp_dat, read_rixs_map, read_sumrules_dat, read_vtot_dat, read_wscrn_dat,
        read_xmu_dat, read_xscorr_raw_dat, read_xsect_dat, write_apot_bin, write_bandstructure_dat,
        write_chi_dat, write_contour_dat, write_convergence_scf, write_convergence_scf_fine,
        write_crpa_dat, write_curve_dat, write_danes_dat, write_dmdw_out, write_eels_dat,
        write_emesh_bin, write_emesh_dat, write_eps_dat, write_exc_dat, write_feff_bin,
        write_fms_bin, write_fort16, write_hamaker_dat, write_jzzp_dat, write_ldos_dat,
        write_list_dat, write_mdff_dat, write_misc_dat, write_module_log_dat, write_mpse_dat,
        write_osc_str_dat, write_paths_dat, write_phase_bin, write_pot_bin, write_prexmu_dat,
        write_residue_dat, write_rhorrp_density_text, write_rhozzp_dat, write_rixs_map,
        write_vtot_dat, write_wscrn_dat, write_xmu_dat, write_xscorr_raw_dat, write_xsect_dat,
    };
    use refeff_io::{PathsDatAtom, PathsDatData, PathsDatPath};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn write_minimal_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu smoke test
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
        Ok(())
    }

    fn write_bandstructure_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu band smoke test
BANDSTRUCTURE -5.0 10.0 0.25 2 64 T
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_dmdw_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
DEBYE 450 315 5 dym/force.dym 6 0 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_highz_template_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE test_element
HIGHZ
POTENTIALS
       0    XXX   Te
END
"#,
        )?;
        Ok(())
    }

    fn write_opcons_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu opcons run
OPCONS
NUMDENS 0 1.0
POTENTIALS
0 29 Cu
END
"#,
        )?;
        Ok(())
    }

    fn write_xsph_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu XSPH cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
        )?;
        Ok(())
    }

    fn write_self_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu SELF cache run
SELF
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
        )?;
        Ok(())
    }

    fn write_fms_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu FMS cache run
CONTROL 1 1 1 1 1 1
FMS 5.5
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
        )?;
        Ok(())
    }

    fn write_rixs_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu RIXS cache run
EDGE L3 VAL
RIXS 0.1 0.1
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
        )?;
        Ok(())
    }

    fn write_rhorrp_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu RHORRP cache run
EDGE K
DENSITY
line density.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 2
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
        )?;
        Ok(())
    }

    fn write_compton_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu compton cache run
COMPTON 1.0 3 0
CGRID 1.0 2 2 3 3
END
"#,
        )?;
        Ok(())
    }

    fn write_compton_rhozzp_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu compton rhozzp cache run
COMPTON 1.0 3 0
RHOZZP
CGRID 1.0 2 2 3 3
END
"#,
        )?;
        Ok(())
    }

    fn write_crpa_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Ce CRPA cache run
CRPA 2 3.5
END
"#,
        )?;
        Ok(())
    }

    fn write_ldos_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu LDOS cache run
LDOS -1 1 0.1 3 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_eels_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu EELS cache run
ELNES
300
0 1 0
2.4 0.0
5 3
0.0 0.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_eelsmdff_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu EELS-MDFF cache run
ELNES
300
0 1 0
2.4 0.0
5 3
0.0 0.0
MDFF 3
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_dmdw_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
DEBYE 450 315 5 feff.dym 2 0 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_path_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu PATH cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_genfmt_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu GENFMT cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_ff2x_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu FF2X cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_fullspectrum_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu fullspectrum cache run
FULLSPECTRUM
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
        Ok(())
    }

    fn sample_fullspectrum_eps_dat() -> EpsDatData {
        EpsDatData {
            header_lines: vec!["# sample eps.dat".to_string()],
            omega: Array1::from_vec(vec![1.0, 2.0, 4.0, 7.0]),
            epsilon: Array1::from_vec(vec![
                Complex64::new(0.2, 0.05),
                Complex64::new(0.4, 0.12),
                Complex64::new(0.1, 0.07),
                Complex64::new(0.3, 0.03),
            ]),
            background_epsilon: Array1::from_vec(vec![
                Complex64::new(0.1, 0.02),
                Complex64::new(0.2, 0.04),
                Complex64::new(0.05, 0.025),
                Complex64::new(0.15, 0.01),
            ]),
            sigma: Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04]),
        }
    }

    fn sample_fullspectrum_osc_str_dat() -> OscStrDatData {
        OscStrDatData {
            header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
            rows: vec![OscStrRow {
                component: "Cu".to_string(),
                edge: "K".to_string(),
                core_hole_index: 1,
                effective_electron_count: 5.123,
            }],
        }
    }

    fn sample_fullspectrum_hamaker_dat() -> HamakerDatData {
        HamakerDatData {
            header_lines: vec!["# cached hamaker transform".to_string()],
            omega: Array1::from_vec(vec![1.0, 2.0, 4.0]),
            imaginary_axis_epsilon: Array1::from_vec(vec![
                Complex64::new(0.35, 0.0),
                Complex64::new(0.25, 0.0),
                Complex64::new(0.10, 0.0),
            ]),
        }
    }

    fn sample_crpa_dat() -> CrpaDatData {
        CrpaDatData {
            header_lines: vec!["U, n, U_Bare".to_string()],
            hubbard_u: 0.197_879_035_252_010,
            occupation: 1.0,
            bare_u: 0.694_283_422_651_496,
        }
    }

    fn sample_wscrn_dat() -> WscrnDatData {
        WscrnDatData {
            header_lines: vec![" # r       w_scrn(r)      v_ch(r)".to_string()],
            radius_bohr: Array1::from_vec(vec![
                0.150_733_046_3E-03,
                0.158_461_294_9E-03,
                0.166_585_779_2E-03,
            ]),
            screened_potential: Array1::from_vec(vec![
                0.267_288_234_6E+02,
                0.267_288_167_8E+02,
                0.267_288_030_6E+02,
            ]),
            core_hole_potential: Array1::from_vec(vec![
                0.291_616_524_4E+02,
                0.291_616_457_6E+02,
                0.291_616_320_4E+02,
            ]),
        }
    }

    fn sample_vtot_dat() -> VtotDatData {
        VtotDatData {
            header_lines: Vec::new(),
            radius_bohr: Array1::from_vec(vec![
                0.150_733_046_3E-03,
                0.158_461_294_9E-03,
                0.166_585_779_2E-03,
            ]),
            total_potential: Array1::from_vec(vec![
                -0.182_900_150_0E+06,
                -0.182_900_133_6E+06,
                -0.182_900_100_2E+06,
            ]),
            screened_core_hole_potential: Array1::from_vec(vec![
                0.267_288_234_6E+02,
                0.267_288_167_8E+02,
                0.267_288_030_6E+02,
            ]),
        }
    }

    fn sample_screen_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating screened core-hole potential ...".to_string(),
                "Done with module: screened core-hole potential.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn sample_ldos_dat() -> Result<LdosDatData> {
        Ok(LdosDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e        sDOS           pDOS          dDOS          fDOS".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: Array1::from_vec(vec![-1.0, 0.0, 1.0]),
            density: Array2::from_shape_vec(
                (3, 4),
                vec![
                    1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 1.2E-4, 2.2E-4,
                    3.2E-4, 4.2E-4,
                ],
            )?,
        })
    }

    fn sample_eels_dat() -> EelsDatData {
        EelsDatData {
            header_lines: vec![
                "# Orientation averaged EELS calculation".to_string(),
                "#  Energy       total         atomic-bg     fine-struct".to_string(),
            ],
            energy_loss_ev: Array1::from_vec(vec![8979.41, 8980.98, 8982.40]),
            total: Array1::from_vec(vec![0.123_014E-12, 0.146_285E-12, 0.176_683E-12]),
            atomic_background: Array1::from_vec(vec![0.138_430E-12, 0.166_322E-12, 0.203_202E-12]),
            fine_structure: Array1::from_vec(vec![-0.154_167E-13, -0.200_377E-13, -0.265_188E-13]),
            tensor: None,
        }
    }

    fn sample_mdff_dat() -> Result<MdffDatData> {
        Ok(MdffDatData {
            header_lines: vec![
                "# Orientation sensitive EELS calculation - beam energy =    300keV".to_string(),
                "#  Energy       total".to_string(),
            ],
            energy_loss_ev: Array1::from_vec(vec![10.0, 12.5]),
            spectrum: Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.25),
                    Complex64::new(0.5, -0.1),
                    Complex64::new(1.2, 0.2),
                    Complex64::new(0.8, -0.05),
                ],
            )?,
        })
    }

    fn sample_mpse_dat() -> MpseDatData {
        MpseDatData {
            header_lines: vec!["# XSPH MPSE self-energy sidecar".to_string()],
            energy_ev: Array1::from_vec(vec![0.038_099_840_30, 0.152_399_361_2]),
            self_energy: Array1::from_vec(vec![
                Complex64::new(0.001_436_696_198, -0.000_007_842_984_015),
                Complex64::new(0.005_774_807_411, -0.000_124_742_315_9),
            ]),
            renormalization: Some(Array1::from_vec(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])),
            renormalization_magnitude: Some(Array1::from_vec(vec![1.0, 1.0])),
            renormalization_phase: Some(Array1::from_vec(vec![0.0, 0.0])),
            inelastic_mean_free_path: Some(Array1::from_vec(vec![48_578.245_52, 6_108.567_091])),
        }
    }

    fn sample_emesh_dat() -> EmeshDatData {
        EmeshDatData {
            edge_hartree: 333.333,
            bohr_angstrom: 0.529_177_249,
            edge_ev: 9_071.2,
            spectrum: 0,
            fermi_index: 1,
            indices: Array1::from_vec(vec![1, 2, 3]),
            energy_ev: Array1::from_vec(vec![0.0, 1.5, 3.0]),
            wave_number_inverse_angstrom: Array1::from_vec(vec![0.0, 0.627, 0.887]),
        }
    }

    fn sample_emesh_bin() -> EmeshBinData {
        EmeshBinData {
            point_count_declared: 3,
            horizontal_count: 2,
            danes_extension_count: 1,
            energy_hartree: Array1::from_vec(vec![
                Complex64::new(-0.25, 0.01),
                Complex64::new(0.0, 0.02),
                Complex64::new(0.5, 0.03),
            ]),
        }
    }

    fn sample_exc_dat() -> ExcDatData {
        ExcDatData {
            header_lines: vec!["# SELF excitation poles".to_string()],
            energy_ev: Array1::from_vec(vec![15.0, 27.5]),
            broadening_ev: Array1::from_vec(vec![0.15, 0.275]),
            oscillator_strength: Array1::from_vec(vec![0.75, 0.25]),
            auxiliary_weight: Some(Array1::from_vec(vec![1.0, 0.5])),
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

    fn sample_phase_bin_data() -> PhaseBinData {
        let spin_count = 1;
        let energy_count = 2;
        let transition_count = 2;
        let q_count = 1;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: 2,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 1,
            pad_width: 8,
            final_state_count: 4,
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
            potentials: vec![
                sample_phase_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
                sample_phase_potential(1, 8, "O", energy_count, spin_count, 0.2),
            ],
            transition_moments: Array4::from_shape_fn(
                (energy_count, q_count, transition_count, spin_count),
                |(energy, q_index, transition, spin)| {
                    Complex64::new(
                        0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                        -0.02 * spin as f64,
                    )
                },
            ),
            raw_pads: None,
        }
    }

    fn sample_phase_potential(
        lmax: usize,
        atomic_number: usize,
        label: &str,
        energy_count: usize,
        spin_count: usize,
        scale: f64,
    ) -> PhaseBinPotential {
        let l_count = 2 * lmax + 1;
        PhaseBinPotential {
            lmax,
            atomic_number,
            label: label.to_string(),
            phase_shifts: Array3::from_shape_fn(
                (energy_count, l_count, spin_count),
                |(energy, l_slot, spin)| {
                    Complex64::new(
                        scale + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                        0.001 * spin as f64,
                    )
                },
            ),
        }
    }

    fn sample_xsect_dat() -> XsectDatData {
        XsectDatData {
            titles: vec!["Cu crystal".to_string()],
            scalars: XsectDatScalars {
                amplitude_reduction: 0.85,
                relaxation_energy: 0.15,
                plasmon_frequency: 2.4,
                edge_energy: 9.1,
                chemical_potential: -0.4,
            },
            core_hole_width_ev: 1.23,
            main_energy_count: 2,
            fermi_index: 1,
            energy_grid_ev: Array1::from_vec(vec![
                Complex64::new(1.25, 0.01),
                Complex64::new(1.5, 0.02),
            ]),
            normalized_background: Array1::from_vec(vec![2.0, 2.5]),
            cross_section: Array1::from_vec(vec![
                Complex64::new(3.0, -0.4),
                Complex64::new(3.5, -0.5),
            ]),
        }
    }

    fn sample_fms_bin_data() -> FmsBinData {
        FmsBinData {
            cluster_radius_angstrom: 5.5,
            energy_count: 2,
            main_energy_count: 1,
            auxiliary_energy_count: 0,
            highest_potential_index: 1,
            pad_width: 8,
            declared_spectrum_count: Some(2),
            spectra: Array2::from_shape_fn((2, 2), |(spectrum, energy)| {
                Complex64::new(
                    0.25 * (energy + 1) as f64 + spectrum as f64,
                    -0.05 * (energy + 1) as f64 - spectrum as f64,
                )
            }),
        }
    }

    fn sample_rixs_map_data() -> RixsMapData {
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

    fn sample_rhorrp_density_text_data() -> RhorrpDensityTextData {
        RhorrpDensityTextData {
            points_angstrom: Array2::from_shape_fn((2, 3), |(row, coordinate)| {
                if row == 1 && coordinate == 0 {
                    0.529_177_249
                } else {
                    0.0
                }
            }),
            density_per_angstrom3: Array1::from_vec(vec![1.0, 2.0]),
            nearest: Some(RhorrpNearestAtomColumns {
                displacement_bohr: Array2::from_shape_fn((2, 3), |(row, coordinate)| {
                    if row == 1 && coordinate == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }),
                atom_indices: Array1::from_vec(vec![0, 0]),
                potential_indices: Array1::from_vec(vec![0, 0]),
            }),
        }
    }

    fn sample_feff_bin_data() -> FeffBinData {
        FeffBinData {
            version: "refeff-test".to_string(),
            pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
            ihole: 1,
            order: 2,
            initial_angular_momentum: 0,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potentials: vec![
                FeffBinPotential {
                    label: "Cu".to_string(),
                    atomic_number: 29,
                },
                FeffBinPotential {
                    label: "O".to_string(),
                    atomic_number: 8,
                },
            ],
            central_phase_shift: Array1::from_vec(vec![
                Complex64::new(0.1, -0.01),
                Complex64::new(0.2, -0.02),
                Complex64::new(0.3, -0.03),
            ]),
            complex_momentum: Array1::from_vec(vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(1.1, 0.2),
                Complex64::new(1.2, 0.3),
            ]),
            real_momentum: Array1::from_vec(vec![0.5, 0.6, 0.7]),
            paths: vec![FeffBinPath {
                index: 17,
                degeneracy: 4.0,
                effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
                criterion: 12.5,
                potential_indices: Array1::from_vec(vec![0, 1, 0]),
                positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                    (0, 0..=2) => 0.0,
                    (1, 0) => 1.0,
                    (1, 1) => 0.5,
                    (1, 2) => 0.0,
                    (2, 0) => -1.0,
                    (2, 1) => 0.25,
                    (2, 2) => 0.0,
                    _ => 0.0,
                }),
                beta: Array1::from_vec(vec![0.1, 0.2, 0.3]),
                eta: Array1::from_vec(vec![0.4, 0.5, 0.6]),
                leg_distances: Array1::from_vec(vec![1.0, 1.1, 1.2]),
                amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
                phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
            }],
            raw_text: None,
        }
    }

    fn sample_list_dat() -> ListDatData {
        ListDatData {
            titles: vec!["PATH  Rmax= 6.000".to_string()],
            entries: vec![ListDatEntry {
                path_index: 17,
                sigma2: 0.0,
                amplitude_ratio: 12.5,
                degeneracy: 4.0,
                leg_count: 3,
                effective_half_path_length_angstrom: 2.5,
            }],
        }
    }

    fn sample_xmu_dat() -> XmuDatData {
        XmuDatData {
            header_lines: vec![
                "# # Cu                                                           FEFF 10.0"
                    .to_string(),
                "# xsedge+ 50, used to normalize mu           1.234500E+00".to_string(),
            ],
            normalization: Some(1.2345),
            photon_energy_ev: Array1::from_vec(vec![8979.0, 8980.0, 8981.0]),
            relative_energy_ev: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            wave_number: Array1::from_vec(vec![0.0, 0.512, 0.724]),
            mu: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            mu0: Array1::from_vec(vec![0.9, 0.95, 1.0]),
            chi: Array1::from_vec(vec![0.1, 0.15, 0.2]),
        }
    }

    fn sample_chi_dat() -> ChiDatData {
        ChiDatData {
            header_lines: vec![
                "# # Cu                                                           FEFF 10.0"
                    .to_string(),
                "#       k          chi          mag           phase @#".to_string(),
            ],
            wave_number: Array1::from_vec(vec![0.0, 0.05, 0.1]),
            chi: Array1::from_vec(vec![-0.115_938_3, -0.119_413_8, -0.122_912_6]),
            magnitude: Array1::from_vec(vec![0.270_227_8, 0.272_670_8, 0.275_083_6]),
            phase: Array1::from_vec(vec![-2.698_164, -2.688_285, -2.678_386]),
            phase_minus_2kr: None,
            ckp_real: None,
            ckp_imag: None,
        }
    }

    fn sample_danes_dat() -> DanesDatData {
        DanesDatData {
            header_lines: vec!["# E  matsub. sommerf. anomal. tale, total, differ.".to_string()],
            energy_ev: Array1::from_vec(vec![-18.690, -17.122, -15.703]),
            matsubara: Array1::from_vec(vec![0.0, 0.0, 0.0]),
            sommerfeld: Array1::from_vec(vec![0.0, 0.0, 0.0]),
            anomalous: Array1::from_vec(vec![10.097, 10.603, 11.159]),
            tail: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
            total: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
            difference: Array1::from_vec(vec![-5.4576, -5.6591, -5.8651]),
        }
    }

    fn sample_xscorr_complex_table() -> XscorrComplexTable {
        XscorrComplexTable {
            energy_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
            values: Array1::from_vec(vec![
                Complex64::new(-0.000_020_637_731_56, 0.000_120_322_770_8),
                Complex64::new(-0.000_021_177_763_91, 0.000_123_685_052_9),
            ]),
        }
    }

    fn sample_xscorr_curve_dat() -> XscorrCurveDatData {
        XscorrCurveDatData {
            energy: Array1::from_vec(vec![
                Complex64::new(-0.138_801_301_5, 0.000_183_746_545),
                Complex64::new(-0.138_801_301_5, 0.000_367_493_09),
            ]),
            values: Array1::from_vec(vec![
                Complex64::new(-0.000_028_662, 0.000_237_48),
                Complex64::new(-0.000_028_683, 0.000_237_44),
            ]),
        }
    }

    fn sample_xscorr_raw_dat() -> XscorrRawDatData {
        XscorrRawDatData {
            temperature_hartree: 0.0,
            electronic_temperature_ev: 0.0,
            loss_ev: 0.864_59,
            fermi_energy_ev: -3.776_977_18,
            pole_count: 0,
            omega_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
            cchi: Array1::from_vec(vec![
                Complex64::new(-0.000_016_299_5, 0.000_115_24),
                Complex64::new(-0.000_016_898_337_65, 0.000_118_558_222_9),
            ]),
            one_minus_fermi: Array1::from_vec(vec![0.5, 0.514_017_875_2]),
            xmu0: Array1::from_vec(vec![
                Complex64::new(-0.000_032_599, 0.000_230_48),
                Complex64::new(-0.000_032_875, 0.000_230_65),
            ]),
        }
    }

    fn sample_dmdw_out() -> DmdwOutData {
        let mut section = DmdwOutSection::new(DmdwOutSubject::PathIndices(vec![1, 2]));
        section.reduced_mass_amu = Some(31.773);
        section.path_length_angstrom = Some(2.5323);
        section.sigma2_1e_minus_3_angstrom2 = Some(11.8576);

        DmdwOutData {
            header: Some(DmdwOutHeader {
                lanczos_recursion_order: 2,
                temperature: DmdwOutTemperature::Single(450.0),
                dynamical_matrix_file: "feff.dym".to_string(),
            }),
            sections: vec![section],
        }
    }

    fn minimal_dym_text() -> &'static str {
        concat!(
            "    1\n",
            "    1\n",
            "   29\n",
            "   63.546000\n",
            "    0.00000000    0.00000000    0.00000000\n",
            "    1    1\n",
            "  1.000000E+00  0.000000E+00  0.000000E+00\n",
            "  0.000000E+00  1.000000E+00  0.000000E+00\n",
            "  0.000000E+00  0.000000E+00  1.000000E+00\n",
        )
    }

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
        let document =
            FeffDocument::from_input(&FeffInput::parse_file(temp.path().join("feff.inp"))?)?;
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

    #[test]
    fn rdinp_stage_writes_supported_outputs_to_requested_dir() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_minimal_input(&input)?;

        let report = execute_rdinp(&input, &output)?;

        assert_eq!(report.cards, 6);
        assert_eq!(report.atoms, 2);
        assert_eq!(report.potentials, 2);
        assert!(
            report
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.starts_with("Launching FEFF version"))
        );
        assert!(output.join("atoms.dat").is_file());
        assert!(output.join("geom.dat").is_file());
        assert!(output.join(".dimensions.dat").is_file());
        assert!(output.join("log.dat").is_file());
        assert!(output.join("rixs.inp").is_file());
        assert!(!output.join(".feff.error").exists());
        Ok(())
    }

    #[test]
    fn rdinp_stage_copies_relative_dmdw_auxiliary_to_requested_dir() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        let dym_dir = temp.path().join("dym");
        std::fs::create_dir_all(&dym_dir)?;
        write_dmdw_input(&input)?;
        std::fs::write(dym_dir.join("force.dym"), minimal_dym_text())?;

        execute_rdinp(&input, &output)?;

        assert_eq!(
            std::fs::read_to_string(output.join("dym").join("force.dym"))?,
            minimal_dym_text()
        );
        Ok(())
    }

    #[test]
    fn full_run_writes_rdinp_outputs_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_minimal_input(&input)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(error.to_string().contains("completed rdinp"));
        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(output.join("pot.inp").is_file());
        assert!(output.join("xsph.inp").is_file());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_wpot_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_minimal_input(&input)?;
        write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
        write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
        write_misc_dat(output.join("misc.dat"), &sample_misc_dat())?;
        write_convergence_scf(output.join("convergence.scf"), &sample_convergence_scf())?;
        write_convergence_scf_fine(
            output.join("convergence.scf.fine"),
            &sample_convergence_scf_fine(),
        )?;
        write_fort16(output.join("fort.16"), &sample_fort16())?;
        let expected_misc = read_misc_dat(output.join("misc.dat"))?;
        let expected_convergence = read_convergence_scf(output.join("convergence.scf"))?;
        let expected_convergence_fine =
            read_convergence_scf_fine(output.join("convergence.scf.fine"))?;
        let expected_fort16 = read_fort16(output.join("fort.16"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        let message = error.to_string();
        assert!(message.contains("atomic=1 file(s)"));
        assert!(message.contains("wpot=5 file(s)"));
        assert!(output.join("pot00.dat").is_file());
        assert_eq!(read_misc_dat(output.join("misc.dat"))?, expected_misc);
        assert_eq!(
            read_convergence_scf(output.join("convergence.scf"))?,
            expected_convergence
        );
        assert_eq!(
            read_convergence_scf_fine(output.join("convergence.scf.fine"))?,
            expected_convergence_fine
        );
        assert_eq!(read_fort16(output.join("fort.16"))?, expected_fort16);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_atomic_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_minimal_input(&input)?;
        write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
        let expected = read_apot_bin(output.join("apot.bin"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: atomic=1 file(s)")
        );
        assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_xsph_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_xsph_cached_input(&input)?;
        write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
        write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;
        write_mpse_dat(output.join("mpse.dat"), &sample_mpse_dat())?;
        write_emesh_dat(output.join("emesh.dat"), &sample_emesh_dat())?;
        write_emesh_bin(output.join("emesh.bin"), &sample_emesh_bin())?;
        let expected_phase = read_phase_bin(output.join("phase.bin"))?;
        let expected_xsect = read_xsect_dat(output.join("xsect.dat"))?;
        let expected_mpse = read_mpse_dat(output.join("mpse.dat"))?;
        let expected_emesh = read_emesh_dat(output.join("emesh.dat"))?;
        let expected_emesh_bin = read_emesh_bin(output.join("emesh.bin"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: xsph=5 file(s)")
        );
        assert_eq!(read_phase_bin(output.join("phase.bin"))?, expected_phase);
        assert_eq!(read_xsect_dat(output.join("xsect.dat"))?, expected_xsect);
        assert_eq!(read_mpse_dat(output.join("mpse.dat"))?, expected_mpse);
        assert_eq!(read_emesh_dat(output.join("emesh.dat"))?, expected_emesh);
        assert_eq!(
            read_emesh_bin(output.join("emesh.bin"))?,
            expected_emesh_bin
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_self_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_self_cached_input(&input)?;
        write_exc_dat(output.join("exc.dat"), &sample_exc_dat())?;
        let expected = read_exc_dat(output.join("exc.dat"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: self=2 pole(s)")
        );
        assert_eq!(read_exc_dat(output.join("exc.dat"))?, expected);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_fms_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_fms_cached_input(&input)?;
        write_fms_bin(output.join("fms.bin"), &sample_fms_bin_data())?;
        let expected_fms = read_fms_bin(output.join("fms.bin"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: fms=1 file(s)")
        );
        assert_eq!(read_fms_bin(output.join("fms.bin"))?, expected_fms);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_band_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_bandstructure_input(&input)?;
        write_bandstructure_dat(
            output.join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        let expected = read_bandstructure_dat(output.join("bandstructure.dat"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: band=1 file(s)")
        );
        assert_eq!(
            read_bandstructure_dat(output.join("bandstructure.dat"))?,
            expected
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_rixs_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_rixs_cached_input(&input)?;
        write_rixs_map(output.join("rixsET.dat"), &sample_rixs_map_data())?;
        let expected_map = read_rixs_map(output.join("rixsET.dat"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: rixs=1 file(s)")
        );
        assert_eq!(read_rixs_map(output.join("rixsET.dat"))?, expected_map);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_rhorrp_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_rhorrp_cached_input(&input)?;
        write_rhorrp_density_text(
            output.join("density.dat"),
            &sample_rhorrp_density_text_data(),
        )?;
        let expected_density = read_rhorrp_density_text(output.join("density.dat"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: rhorrp=1 file(s)")
        );
        assert_eq!(
            read_rhorrp_density_text(output.join("density.dat"))?,
            expected_density
        );
        Ok(())
    }

    #[test]
    fn full_run_skips_incomplete_wpot_cache_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_minimal_input(&input)?;
        write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(!output.join("pot00.dat").exists());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_opcons_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_opcons_input(&input)?;
        std::fs::write(
            output.join("opconsCu.dat"),
            concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
        )?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: opcons=3 row(s)")
        );
        assert!(output.join("loss.dat").is_file());
        Ok(())
    }

    #[test]
    fn full_run_skips_opcons_stage_when_tables_are_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_opcons_input(&input)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(!output.join("loss.dat").exists());
        Ok(())
    }

    #[test]
    fn full_run_skips_compton_stage_when_jzzp_cache_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_compton_cached_input(&input)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(!output.join("compton.dat").exists());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_compton_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_compton_cached_input(&input)?;
        write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: compton=3 row(s)")
        );
        assert_eq!(
            read_compton_dat(output.join("compton.dat"))?.point_count(),
            3
        );
        assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, sample_jzzp_data());
        Ok(())
    }

    #[test]
    fn full_run_preserves_cached_compton_rhozzp_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_compton_rhozzp_cached_input(&input)?;
        write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;
        write_rhozzp_dat(output.join("rhozzp.dat"), &sample_rhozzp_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: compton=6 row(s)")
        );
        assert_eq!(
            read_compton_dat(output.join("compton.dat"))?.point_count(),
            3
        );
        assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, sample_jzzp_data());
        assert_eq!(
            read_rhozzp_dat(output.join("rhozzp.dat"))?,
            sample_rhozzp_data()
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_crpa_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_crpa_cached_input(&input)?;
        write_crpa_dat(output.join("crpa.dat"), &sample_crpa_dat())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: crpa=1 row(s)")
        );
        assert_eq!(read_crpa_dat(output.join("crpa.dat"))?, sample_crpa_dat());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_screen_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_minimal_input(&input)?;
        write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;
        write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
        write_module_log_dat(output.join("logscreen.dat"), &sample_screen_module_log())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: screen=6 row(s)")
        );
        assert_eq!(
            read_wscrn_dat(output.join("wscrn.dat"))?,
            sample_wscrn_dat()
        );
        assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, sample_vtot_dat());
        assert_eq!(
            read_module_log_dat(output.join("logscreen.dat"))?,
            sample_screen_module_log()
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_ldos_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_ldos_cached_input(&input)?;
        write_ldos_dat(output.join("ldos00.dat"), &sample_ldos_dat()?)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: ldos=1 file(s)")
        );
        assert_eq!(
            read_ldos_dat(output.join("ldos00.dat"))?,
            sample_ldos_dat()?
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_eels_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_eels_cached_input(&input)?;
        write_eels_dat(output.join("eels.dat"), &sample_eels_dat())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: eels=3 row(s)")
        );
        assert_eq!(read_eels_dat(output.join("eels.dat"))?, sample_eels_dat());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_eelsmdff_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_eelsmdff_cached_input(&input)?;
        write_mdff_dat(output.join("mdff.dat"), &sample_mdff_dat()?)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: eelsmdff=2 row(s)")
        );
        assert_eq!(read_mdff_dat(output.join("mdff.dat"))?, sample_mdff_dat()?);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_dmdw_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_dmdw_cached_input(&input)?;
        std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())?;
        write_dmdw_out(output.join("dmdw.out"), &sample_dmdw_out())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: dmdw=1 section(s)")
        );
        assert_eq!(read_dmdw_out(output.join("dmdw.out"))?, sample_dmdw_out());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_path_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_path_cached_input(&input)?;
        write_paths_dat(output.join("paths.dat"), &sample_paths_dat())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: path=1 path(s)")
        );
        assert_eq!(
            read_paths_dat(output.join("paths.dat"))?,
            sample_paths_dat()
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_genfmt_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_genfmt_cached_input(&input)?;
        write_feff_bin(output.join("feff.bin"), &sample_feff_bin_data())?;
        write_list_dat(output.join("list.dat"), &sample_list_dat())?;
        let expected_feff = read_feff_bin(output.join("feff.bin"))?;
        let expected_list = read_list_dat(output.join("list.dat"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: genfmt=2 file(s)")
        );
        assert_eq!(read_feff_bin(output.join("feff.bin"))?, expected_feff);
        assert_eq!(read_list_dat(output.join("list.dat"))?, expected_list);
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_ff2x_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_ff2x_cached_input(&input)?;
        write_xmu_dat(output.join("xmu.dat"), &sample_xmu_dat())?;
        write_chi_dat(output.join("chi.dat"), &sample_chi_dat())?;
        write_danes_dat(output.join("danes.dat"), &sample_danes_dat())?;
        write_prexmu_dat(output.join("prexmu.dat"), &sample_xscorr_complex_table())?;
        write_residue_dat(output.join("residue.dat"), &sample_xscorr_complex_table())?;
        write_contour_dat(output.join("contour.dat"), &sample_xscorr_complex_table())?;
        write_curve_dat(output.join("curve.dat"), &sample_xscorr_curve_dat())?;
        write_xscorr_raw_dat(output.join("raw.dat"), &sample_xscorr_raw_dat())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: ff2x=8 file(s)")
        );
        assert_eq!(read_xmu_dat(output.join("xmu.dat"))?, sample_xmu_dat());
        assert_eq!(read_chi_dat(output.join("chi.dat"))?, sample_chi_dat());
        assert_eq!(
            read_danes_dat(output.join("danes.dat"))?,
            sample_danes_dat()
        );
        assert_eq!(
            read_prexmu_dat(output.join("prexmu.dat"))?,
            sample_xscorr_complex_table()
        );
        assert_eq!(
            read_residue_dat(output.join("residue.dat"))?,
            sample_xscorr_complex_table()
        );
        assert_eq!(
            read_contour_dat(output.join("contour.dat"))?,
            sample_xscorr_complex_table()
        );
        assert_eq!(
            read_curve_dat(output.join("curve.dat"))?,
            sample_xscorr_curve_dat()
        );
        assert_eq!(
            read_xscorr_raw_dat(output.join("raw.dat"))?,
            sample_xscorr_raw_dat()
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_fullspectrum_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_fullspectrum_cached_input(&input)?;
        write_eps_dat(output.join("eps.dat"), &sample_fullspectrum_eps_dat())?;
        write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
        write_osc_str_dat(
            output.join("osc_str.dat"),
            &sample_fullspectrum_osc_str_dat(),
        )?;
        write_hamaker_dat(
            output.join("hamaker.dat"),
            &sample_fullspectrum_hamaker_dat(),
        )?;
        let expected_osc_str = read_osc_str_dat(output.join("osc_str.dat"))?;
        let expected_hamaker = read_hamaker_dat(output.join("hamaker.dat"))?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: fullspectrum=4 row(s)")
        );
        assert_eq!(
            read_opcons_dat(output.join("opconsKK.dat"))?.point_count(),
            4
        );
        assert_eq!(
            read_sumrules_dat(output.join("sumrules.dat"))?.point_count(),
            4
        );
        assert!(output.join("opcons.dat").is_file());
        assert!(output.join("opcons0.dat").is_file());
        assert_eq!(
            read_osc_str_dat(output.join("osc_str.dat"))?,
            expected_osc_str
        );
        assert_eq!(
            read_hamaker_dat(output.join("hamaker.dat"))?,
            expected_hamaker
        );
        Ok(())
    }

    #[test]
    fn failed_rdinp_writes_feff_style_error_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_highz_template_input(&input)?;

        let error = execute_rdinp(&input, &output)
            .err()
            .context("HIGHZ template should fail during rdinp extraction")?;

        assert!(error.to_string().contains("XXX"));
        assert_eq!(
            std::fs::read_to_string(output.join("log.dat"))?,
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Using finite nucleus.\n",
                " Error reading input, bad line follows:\n",
                " 0    XXX   Te\n",
                "RDINP fatal error.\n",
            )
        );
        assert_eq!(
            std::fs::read_to_string(output.join(".feff.error"))?,
            rdinp::rdinp_error_sentinel_string()
        );
        Ok(())
    }

    fn reference_opcons_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/MPSE/Cu_OPCONS/REFERENCE.zip");
        Ok(path.is_file().then_some(path))
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to read {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn sample_jzzp_data() -> JzzpDatData {
        JzzpDatData {
            ns: 2,
            nphi: 2,
            nz: 3,
            nzp: 3,
            smax: 1.0,
            phimax: std::f64::consts::PI,
            zmax: 1.0,
            zpmax: 1.0,
            values: Array2::from_shape_fn((3, 3), |(z, zp)| {
                0.2 + z as f64 * 0.1 + zp as f64 * 0.05
            }),
        }
    }

    fn sample_rhozzp_data() -> RhozzpDatData {
        RhozzpDatData {
            header_lines: vec![" # rhozzp diagnostic".to_string()],
            z_prime: Array1::from_vec(vec![0.01, 0.51, 1.01]),
            density: Array1::from_vec(vec![0.45, 0.35, 0.15]),
        }
    }

    fn sample_misc_dat() -> MiscDatData {
        MiscDatData {
            titles: vec![
                "Cu".to_string(),
                "absorbing".to_string(),
                " POT  SCF 100  5.5000   0, core-hole, AFOLP (folp(0)= 1.150)".to_string(),
            ],
        }
    }

    fn sample_convergence_scf() -> ScfConvergenceData {
        let header =
            " # it. E_fermi(eV)  Charge Distance  Partial Chg. D.  Convergence".to_string();
        let first = ScfConvergenceRow {
            iteration: 0,
            fermi_level_ev: -4.006,
            charge_distance: 0.0,
            partial_charge_distance: 0.0,
            converged: false,
        };
        let second = ScfConvergenceRow {
            iteration: 1,
            fermi_level_ev: -4.125,
            charge_distance: 0.3252,
            partial_charge_distance: 0.5599,
            converged: true,
        };
        ScfConvergenceData {
            detail_lines: vec![header.clone()],
            rows: vec![first.clone(), second.clone()],
            lines: vec![
                ScfConvergenceLine::Detail(header),
                ScfConvergenceLine::Row(first),
                ScfConvergenceLine::Row(second),
            ],
        }
    }

    fn sample_convergence_scf_fine() -> ScfConvergenceData {
        let title = " Electronic configuration".to_string();
        let detail = " 0     2   10.466".to_string();
        let row = ScfConvergenceRow {
            iteration: 2,
            fermi_level_ev: -4.250,
            charge_distance: 0.1025,
            partial_charge_distance: 0.2250,
            converged: true,
        };
        ScfConvergenceData {
            detail_lines: vec![title.clone(), detail.clone()],
            rows: vec![row.clone()],
            lines: vec![
                ScfConvergenceLine::Detail(title),
                ScfConvergenceLine::Detail(detail),
                ScfConvergenceLine::Row(row),
            ],
        }
    }

    fn sample_fort16() -> Fort16Data {
        Fort16Data {
            total_energy_hartree: Array1::from_vec(vec![
                -1_322.522_518_926_127_5,
                -1_652.786_043_284_159_6,
            ]),
        }
    }

    fn sample_pot_bin_data() -> PotBinData {
        let potentials = 1;
        PotBinData {
            titles: vec!["CLI wpot smoke test".to_string()],
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
            muffin_tin_indices: Array1::from_vec(vec![12]),
            muffin_tin_radii: Array1::from_vec(vec![1.1]),
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
            electron_density: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| 0.035 * (row + 1) as f64,
            ),
            coulomb_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| -1.2 - 0.02 * (row + 1) as f64,
            ),
            total_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| -0.45 + 0.003 * (row + 1) as f64,
            ),
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

    fn sample_apot_bin_data() -> ApotBinData {
        ApotBinData {
            sections: vec![
                apot_matrix_section(
                    8,
                    "rho(r,0:nphx+1) - atomic density for each unique potential",
                    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, 2), |(row, potential)| {
                        0.015 * (row + 1) as f64 + 0.25 * potential as f64
                    }),
                ),
                apot_matrix_section(
                    11,
                    "vcoul(r,nph) - coulomb potential for each unique potential.",
                    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, 2), |(row, potential)| {
                        -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
                    }),
                ),
            ],
        }
    }

    fn sample_bandstructure_dat() -> BandstructureDatData {
        BandstructureDatData {
            header_lines: vec![
                " # grid of            2  k-points.".to_string(),
                " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000".to_string(),
                " # Found between            1  and            2  number of bands.".to_string(),
            ],
            rows: vec![
                BandstructureRow {
                    index: 1,
                    k_point: [0.0, 0.5, 0.25],
                    bands: Array1::from_vec(vec![-5.0, 1.25]),
                },
                BandstructureRow {
                    index: 2,
                    k_point: [0.5, 0.25, 0.0],
                    bands: Array1::from_vec(vec![0.75]),
                },
            ],
        }
    }

    fn apot_matrix_section(
        section_number: usize,
        header: &str,
        values: Array2<f64>,
    ) -> ApotBinSection {
        ApotBinSection {
            section_number,
            headers: vec![header.to_string()],
            header_texts: vec![format!(" {header}")],
            column_labels: vec![],
            column_label_text: None,
            payload: ApotBinPayload::Matrix(ApotBinMatrix {
                value_type: ApotBinType::Double,
                values: ApotBinMatrixValues::Real(values),
            }),
            trailing_headers: vec![],
            trailing_header_texts: vec![],
        }
    }
}
