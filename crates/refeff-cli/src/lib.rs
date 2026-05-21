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
mod tests;
