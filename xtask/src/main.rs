#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use refeff_io::{FeffDocument, FeffInput};

#[derive(Debug, Parser)]
#[command(name = "xtask")]
struct Xtask {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ReferenceTests {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
    },
    GenerateGolden {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
        #[arg(long, default_value = "reference-work/golden")]
        out_dir: PathBuf,
        #[arg(long)]
        example: Vec<String>,
        #[arg(long)]
        no_build: bool,
        #[arg(long)]
        force: bool,
        #[arg(long, value_enum, default_value_t = ReferenceProgram::Feff)]
        program: ReferenceProgram,
    },
    BenchE2e,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReferenceProgram {
    Feff,
    Rdinp,
}

impl ReferenceProgram {
    fn binary_candidates(self, ref_dir: &Path) -> [PathBuf; 2] {
        match self {
            Self::Feff => [ref_dir.join("bin/Seq/feff"), ref_dir.join("bin/feff")],
            Self::Rdinp => [ref_dir.join("bin/Seq/rdinp"), ref_dir.join("bin/rdinp")],
        }
    }

    fn log_prefix(self) -> &'static str {
        match self {
            Self::Feff => "feff",
            Self::Rdinp => "rdinp",
        }
    }
}

fn main() -> Result<()> {
    let xtask = Xtask::parse();
    match xtask.command {
        Command::ReferenceTests { ref_dir } => run_reference_tests(ref_dir)?,
        Command::GenerateGolden {
            ref_dir,
            out_dir,
            example,
            no_build,
            force,
            program,
        } => generate_golden(ref_dir, &out_dir, &example, !no_build, force, program)?,
        Command::BenchE2e => {
            println!(
                "end-to-end benchmark orchestration will compare Rust and FEFF10 once execution is available"
            );
        }
    }
    Ok(())
}

fn generate_golden(
    ref_dir: Option<PathBuf>,
    out_dir: &Path,
    examples: &[String],
    build_reference: bool,
    force: bool,
    program: ReferenceProgram,
) -> Result<()> {
    let ref_dir = ref_dir
        .or_else(|| env::var_os("FEFF10_REF").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("feff10"));
    let ref_dir = ref_dir.canonicalize()?;
    let examples_dir = ref_dir.join("examples");

    if build_reference {
        build_reference_feff(&ref_dir)?;
    }
    let driver = reference_driver(&ref_dir, program)?;

    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();

    for input in inputs {
        let parent = input
            .parent()
            .with_context(|| format!("{} has no parent directory", input.display()))?;
        let rel = parent.strip_prefix(&examples_dir)?;
        let rel_string = rel.to_string_lossy();
        if !examples.is_empty() && !examples.iter().any(|pattern| rel_string.contains(pattern)) {
            continue;
        }

        let dest = out_dir.join(rel);
        if dest.exists() {
            if force {
                std::fs::remove_dir_all(&dest)?;
            } else {
                anyhow::bail!(
                    "{} already exists; pass --force to replace it",
                    dest.display()
                );
            }
        }
        std::fs::create_dir_all(&dest)?;
        copy_dir(parent, &dest)?;

        let output = std::process::Command::new(&driver)
            .current_dir(&dest)
            .output()?;
        std::fs::write(
            dest.join(format!("{}.stdout", program.log_prefix())),
            &output.stdout,
        )?;
        std::fs::write(
            dest.join(format!("{}.stderr", program.log_prefix())),
            &output.stderr,
        )?;
        if !output.status.success() {
            anyhow::bail!(
                "{} reference failed for {} with status {}",
                program.log_prefix(),
                rel.display(),
                output.status
            );
        }
        println!("generated {}", dest.display());
    }

    Ok(())
}

fn build_reference_feff(ref_dir: &Path) -> Result<()> {
    let src = ref_dir.join("src");
    let mut command = std::process::Command::new("make");
    command.arg("all").current_dir(&src);
    if !command_exists("ifort") && command_exists("gfortran") {
        let flags = "-ffree-line-length-none -cpp -O3 -fallow-argument-mismatch";
        command
            .arg("F90=gfortran")
            .arg(format!("FLAGS={flags}"))
            .arg("MPIF90=gfortran")
            .arg(format!("MPIFLAGS={flags}"));
    }

    let status = command.status()?;
    if !status.success() {
        anyhow::bail!("failed to build FEFF reference in {}", src.display());
    }
    Ok(())
}

fn command_exists(command: &str) -> bool {
    let command_path = Path::new(command);
    if command_path.components().count() > 1 {
        return command_path.is_file();
    }

    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path).any(|dir| dir.join(command).is_file())
}

fn reference_driver(ref_dir: &Path, program: ReferenceProgram) -> Result<PathBuf> {
    program
        .binary_candidates(ref_dir)
        .into_iter()
        .find(|candidate| candidate.exists())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no {} reference driver found under {}; run xtask generate-golden without --no-build or build FEFF manually",
                program.log_prefix(),
                ref_dir.display()
            )
        })
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            std::fs::create_dir_all(&dst)?;
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

fn run_reference_tests(ref_dir: Option<PathBuf>) -> Result<()> {
    let ref_dir = ref_dir
        .or_else(|| env::var_os("FEFF10_REF").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("feff10"));
    let examples_dir = ref_dir.join("examples");
    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();

    let mut total_cards = 0_usize;
    let mut total_atoms = 0_usize;
    let mut total_potentials = 0_usize;
    for input in &inputs {
        let parsed = FeffInput::parse_file(input)?;
        let document = FeffDocument::from_input(&parsed)?;
        total_cards += parsed.cards().count();
        total_atoms += document.atoms.len();
        total_potentials += document.potentials.len();
    }

    println!(
        "parsed {} FEFF examples: cards={} atoms={} potentials={}",
        inputs.len(),
        total_cards,
        total_atoms,
        total_potentials
    );
    Ok(())
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_feff_inputs(&path, inputs)?;
        } else if path.file_name().is_some_and(|name| name == "feff.inp") {
            inputs.push(path);
        }
    }
    Ok(())
}
