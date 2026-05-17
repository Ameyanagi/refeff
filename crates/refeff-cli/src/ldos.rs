use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{LdosInput, read_ldos_dat, read_rhoc_dat, write_ldos_dat, write_rhoc_dat};

use crate::work_dir_for_input;

/// Run the supported FEFF LDOS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF LDOS run can be satisfied from existing `ldosNN.dat` caches.
pub(crate) fn has_cached_ldos_output(work_dir: &Path) -> Result<bool> {
    let tables = cached_output_paths(work_dir)?;
    if tables.is_empty() {
        return Ok(false);
    }
    let input = read_input(work_dir)?;
    Ok(input.control.mldos == 1)
}

/// Run the FEFF LDOS cached-output path from existing `ldosNN.dat`/`rhocNN.dat`.
///
/// The LDOS FMS/density solver is still unported. This preserves the module
/// boundary for FEFF cache directories by validating and re-rendering the
/// per-potential LDOS and embedded-density tables that downstream modules read.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if input.control.mldos != 1 {
        return Ok(0);
    }

    let tables = cached_output_paths(work_dir)?;
    if tables.is_empty() {
        bail!("LDOS density-of-states generation requires the unported LDOS numerical solver");
    }

    for table in &tables {
        write_cached_output(table)?;
    }
    Ok(tables.len())
}

fn read_input(work_dir: &Path) -> Result<LdosInput> {
    let input_path = work_dir.join("ldos.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    LdosInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(table: &CachedTable) -> Result<()> {
    match table.kind {
        CachedTableKind::Ldos => {
            let data = read_ldos_dat(&table.path)
                .with_context(|| format!("failed to read {}", table.path.display()))?;
            write_ldos_dat(&table.path, &data)
                .with_context(|| format!("failed to write {}", table.path.display()))
        }
        CachedTableKind::Rhoc => {
            let data = read_rhoc_dat(&table.path)
                .with_context(|| format!("failed to read {}", table.path.display()))?;
            write_rhoc_dat(&table.path, &data)
                .with_context(|| format!("failed to write {}", table.path.display()))
        }
    }
}

fn cached_output_paths(work_dir: &Path) -> Result<Vec<CachedTable>> {
    let mut tables = Vec::new();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", work_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(kind) = cached_table_kind(name) {
            tables.push(CachedTable { path, kind });
        }
    }
    tables.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(tables)
}

fn cached_table_kind(name: &str) -> Option<CachedTableKind> {
    let index = name
        .strip_prefix("ldos")
        .and_then(|suffix| suffix.strip_suffix(".dat"));
    if index.is_some_and(is_feff_potential_index) {
        return Some(CachedTableKind::Ldos);
    }

    let index = name
        .strip_prefix("rhoc")
        .and_then(|suffix| suffix.strip_suffix(".dat"));
    if index.is_some_and(is_feff_potential_index) {
        return Some(CachedTableKind::Rhoc);
    }

    None
}

fn is_feff_potential_index(index: &str) -> bool {
    index.len() == 2 && index.chars().all(|digit| digit.is_ascii_digit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedTable {
    path: PathBuf,
    kind: CachedTableKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedTableKind {
    Ldos,
    Rhoc,
}

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use ndarray::array;
    use refeff_io::{LdosDatData, read_ldos_dat, read_rhoc_dat, write_ldos_dat, write_rhoc_dat};
    use std::path::{Path, PathBuf};

    #[test]
    fn ldos_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled LDOS should require the numerical solver")?;

        assert!(error.to_string().contains(
            "LDOS density-of-states generation requires the unported LDOS numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn ldos_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        Ok(())
    }

    #[test]
    fn ldos_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_ldos_dir()? else {
            eprintln!("skipping LDOS reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "ldos.inp",
            "ldos00.dat",
            "ldos01.dat",
            "rhoc00.dat",
            "rhoc01.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_ldos_dat(temp.path().join("ldos00.dat"))?,
            expected_ldos
        );
        assert_eq!(
            read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
            expected_rhoc
        );
        Ok(())
    }

    fn write_ldos_input(work_dir: &Path, enabled: bool) -> Result<()> {
        std::fs::write(
            work_dir.join("ldos.inp"),
            format!(
                concat!(
                    "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                    "{:4}{:4}{:4}{:4}{:4} {:7} {:4}\n",
                    "rfms2, emin, emax, eimag, rgrd\n",
                    "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}\n",
                    "rdirec, toler1, toler2\n",
                    "{:13.5}{:13.5}{:13.5}\n",
                    " lmaxph(0:nph)\n",
                    "{:4}{:4}\n",
                    "ldostype\n",
                    "{:4}\n"
                ),
                i32::from(enabled),
                0,
                0,
                0,
                0,
                3,
                11,
                -1.0,
                -1.0,
                1.0,
                0.1,
                0.05,
                -1.0,
                0.001,
                0.001,
                3,
                3,
                0
            ),
        )?;
        Ok(())
    }

    fn sample_ldos_dat() -> LdosDatData {
        LdosDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e        sDOS           pDOS          dDOS          fDOS".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4],
                [1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4],
                [1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4]
            ],
        }
    }

    fn sample_rhoc_dat() -> LdosDatData {
        LdosDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4],
                [5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4],
                [5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4]
            ],
        }
    }

    fn reference_ldos_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = [
            "ldos.inp",
            "ldos00.dat",
            "ldos01.dat",
            "rhoc00.dat",
            "rhoc01.dat",
        ];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
