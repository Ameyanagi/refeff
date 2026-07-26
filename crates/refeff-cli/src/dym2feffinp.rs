use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use refeff_io::{
    DymSpectrum, DymToFeffOptions, convert_dym_to_feff, read_dym, write_dym_feff_outputs,
};

#[derive(Debug, Parser)]
#[command(
    name = "dym2feffinp",
    version,
    about = "Create a FEFF input and matching reordered .dym file"
)]
struct Dym2FeffInpCli {
    /// Center the FEFF input on one-based atom iAbs.
    #[arg(long = "c", value_name = "iAbs", default_value_t = 1)]
    center: usize,

    /// Write FEFF input to this file.
    #[arg(long = "f", value_name = "fname", default_value = "feff.inp")]
    feff_output: PathBuf,

    /// Write the adjusted dynamical matrix to this file.
    #[arg(long = "d", value_name = "dname", default_value = "feff.dym")]
    dym_output: PathBuf,

    /// Write only ATOMS and POTENTIALS, for JFEFF.
    #[arg(long = "j")]
    jfeff: bool,

    /// Select the EXAFS or XANES FEFF input template.
    #[arg(long = "s", value_name = "EXAFS/XANES", default_value = "EXAFS")]
    spectrum: String,

    /// File containing the dynamical matrix.
    #[arg(value_name = "dymfile")]
    dym_file: PathBuf,
}

impl Dym2FeffInpCli {
    fn spectrum(&self) -> DymSpectrum {
        if self.spectrum == "XANES" {
            DymSpectrum::Xanes
        } else {
            // The production converter maps every value other than the exact
            // uppercase `XANES` token to EXAFS.
            DymSpectrum::Exafs
        }
    }
}

pub(crate) fn main() -> Result<()> {
    let cli = Dym2FeffInpCli::parse();
    run(
        &cli.dym_file,
        cli.center,
        &cli.feff_output,
        &cli.dym_output,
        cli.spectrum(),
        !cli.jfeff,
    )
}

pub(crate) fn run(
    dym_file: &Path,
    center_atom: usize,
    feff_output: &Path,
    dym_output: &Path,
    spectrum: DymSpectrum,
    write_header: bool,
) -> Result<()> {
    if center_atom == 0 {
        bail!("center atom is 1-based and must be at least 1");
    }
    let data =
        read_dym(dym_file).with_context(|| format!("failed to read {}", dym_file.display()))?;
    let conversion = convert_dym_to_feff(
        &data,
        DymToFeffOptions {
            center_atom_index: center_atom - 1,
            spectrum,
            write_header,
        },
    )
    .with_context(|| {
        format!(
            "failed to center {} on atom {center_atom}",
            dym_file.display()
        )
    })?;
    write_dym_feff_outputs(feff_output, dym_output, &conversion).with_context(|| {
        format!(
            "failed to write {} and {}",
            feff_output.display(),
            dym_output.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use refeff_io::read_dym;
    use std::process::Command;

    use super::*;

    #[test]
    fn parser_matches_production_option_spellings_and_defaults() -> Result<()> {
        let defaults = Dym2FeffInpCli::try_parse_from(["dym2feffinp", "input.dym"])?;
        assert_eq!(defaults.center, 1);
        assert_eq!(defaults.feff_output, Path::new("feff.inp"));
        assert_eq!(defaults.dym_output, Path::new("feff.dym"));
        assert!(!defaults.jfeff);
        assert_eq!(defaults.spectrum(), DymSpectrum::Exafs);
        assert_eq!(defaults.dym_file, Path::new("input.dym"));

        let options = Dym2FeffInpCli::try_parse_from([
            "dym2feffinp",
            "--c",
            "2",
            "--f",
            "centered.inp",
            "--d",
            "centered.dym",
            "--j",
            "--s",
            "XANES",
            "input.dym",
        ])?;
        assert_eq!(options.center, 2);
        assert_eq!(options.feff_output, Path::new("centered.inp"));
        assert_eq!(options.dym_output, Path::new("centered.dym"));
        assert!(options.jfeff);
        assert_eq!(options.spectrum(), DymSpectrum::Xanes);

        assert!(Dym2FeffInpCli::try_parse_from(["dym2feffinp", "-c", "2", "input.dym"]).is_err());
        Ok(())
    }

    #[test]
    fn parser_preserves_production_spectrum_fallback() -> Result<()> {
        let lowercase =
            Dym2FeffInpCli::try_parse_from(["dym2feffinp", "--s", "xanes", "input.dym"])?;
        assert_eq!(lowercase.spectrum(), DymSpectrum::Exafs);
        let unknown =
            Dym2FeffInpCli::try_parse_from(["dym2feffinp", "--s", "anything", "input.dym"])?;
        assert_eq!(unknown.spectrum(), DymSpectrum::Exafs);
        Ok(())
    }

    #[test]
    fn matches_pinned_production_converter_semantically() -> Result<()> {
        const PINNED_FEFF10_REVISION: &str = "0a4fbd797cf72938f64dda034a438ce009ec6eb7";
        let upstream_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10");
        let source = upstream_root.join("src/DMDW/Test/H2O.g03.dym");
        let upstream_executable = upstream_root.join("bin/Seq/dym2feffinp");
        if !source.is_file() || !upstream_executable.is_file() {
            eprintln!("skipping pinned dym2feffinp parity; FEFF10 source or executable not found");
            return Ok(());
        }
        let revision = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&upstream_root)
            .output()?;
        if !revision.status.success()
            || String::from_utf8_lossy(&revision.stdout).trim() != PINNED_FEFF10_REVISION
        {
            eprintln!("skipping pinned dym2feffinp parity; FEFF10 revision does not match");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        // The pinned Fortran command-line parser stores filenames in a
        // 50-character buffer. Keep its arguments relative to the temporary
        // working directory so an absolute checkout path is not truncated.
        std::fs::copy(&source, temp.path().join("input.dym"))?;
        let rust_feff_output = temp.path().join("rust.inp");
        let rust_dym_output = temp.path().join("rust.dym");
        run(
            &source,
            2,
            &rust_feff_output,
            &rust_dym_output,
            DymSpectrum::Xanes,
            true,
        )?;

        let reference_feff_output = temp.path().join("reference.inp");
        let reference_dym_output = temp.path().join("reference.dym");
        let upstream_status = Command::new(&upstream_executable)
            .args([
                "--c",
                "2",
                "--f",
                "reference.inp",
                "--d",
                "reference.dym",
                "--s",
                "XANES",
                "input.dym",
            ])
            .current_dir(temp.path())
            .status()?;
        assert!(upstream_status.success());

        // This legacy converter intentionally emits shorthand cards such as
        // `EXCHANGE 0`, which the runtime FEFF parser rejects as incomplete.
        // Byte equality is both stricter and faithful to the converter's
        // standalone output contract.
        assert_eq!(
            std::fs::read(&rust_feff_output)?,
            std::fs::read(&reference_feff_output)?
        );

        let adjusted = read_dym(&rust_dym_output)?;
        let reference_adjusted = read_dym(&reference_dym_output)?;
        assert_eq!(adjusted, reference_adjusted);
        assert_eq!(adjusted.atomic_numbers.to_vec(), vec![1, 8, 1]);
        assert_eq!(
            adjusted.coordinates.cartesian_positions().row(0).to_vec(),
            vec![0.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn rejects_zero_as_a_one_based_center() {
        let error = run(
            Path::new("missing.dym"),
            0,
            Path::new("feff.inp"),
            Path::new("feff.dym"),
            DymSpectrum::Exafs,
            true,
        )
        .expect_err("zero center must fail before file access");
        assert!(error.to_string().contains("1-based"));
    }
}
