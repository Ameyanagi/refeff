//! FEFF `misc.dat` quick-reference header codec.
//!
//! FEFF writes `misc.dat` from the potential stage when `PRINT` enables the
//! diagnostic quick-reference file. The file is the same `wthead` title block
//! used by `potXX.dat` and `xsect.dat`, with one `# `-prefixed record per
//! title line.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

/// Parsed FEFF `misc.dat` contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiscDatData {
    /// Header title records written by FEFF `wthead`.
    pub titles: Vec<String>,
}

impl MiscDatData {
    /// Number of title records.
    #[must_use]
    pub fn title_count(&self) -> usize {
        self.titles.len()
    }

    /// Whether this diagnostic file contains no title records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.titles.is_empty()
    }
}

/// Render FEFF-compatible `misc.dat` text.
pub fn misc_dat_string(data: &MiscDatData) -> Result<String> {
    validate_misc_dat(data)?;

    let mut out = String::new();
    for title in &data.titles {
        writeln!(out, "# {}", title.trim_end())?;
    }
    Ok(out)
}

/// Parse FEFF `misc.dat` text.
pub fn parse_misc_dat(text: &str) -> Result<MiscDatData> {
    let mut titles = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let Some(title) = strip_wthead_title(line) else {
            return Err(IoError::InvalidMiscDat {
                line: index + 1,
                message: "expected a FEFF wthead '# ' title record".to_string(),
            });
        };
        titles.push(title.to_string());
    }
    Ok(MiscDatData { titles })
}

/// Write FEFF `misc.dat` text to a file.
pub fn write_misc_dat(path: impl AsRef<Path>, data: &MiscDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, misc_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `misc.dat` text from a file.
pub fn read_misc_dat(path: impl AsRef<Path>) -> Result<MiscDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_misc_dat(&text)
}

fn validate_misc_dat(data: &MiscDatData) -> Result<()> {
    for (index, title) in data.titles.iter().enumerate() {
        if title.contains('\n') || title.contains('\r') {
            return Err(IoError::InvalidMiscDat {
                line: index + 1,
                message: "title record must not contain line terminators".to_string(),
            });
        }
    }
    Ok(())
}

fn strip_wthead_title(line: &str) -> Option<&str> {
    line.strip_prefix("# ")
        .or_else(|| line.strip_prefix('#').map(str::trim_start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_misc_titles() -> Result<()> {
        let data = parse_misc_dat(MISC_DAT)?;
        assert_eq!(data.title_count(), 3);
        assert!(!data.is_empty());
        assert_eq!(data.titles[0], "Cu");
        assert_eq!(
            data.titles[2],
            " POT  SCF 100  5.5000   0, core-hole, AFOLP (folp(0)= 1.150)"
        );
        Ok(())
    }

    #[test]
    fn roundtrips_misc_text() -> Result<()> {
        let data = parse_misc_dat(MISC_DAT)?;
        assert_eq!(parse_misc_dat(&misc_dat_string(&data)?)?, data);
        assert!(parse_misc_dat("")?.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_bad_misc_inputs() {
        assert!(parse_misc_dat("Cu\n").is_err());
        let bad = MiscDatData {
            titles: vec!["Cu\nbad".to_string()],
        };
        assert!(misc_dat_string(&bad).is_err());
    }

    const MISC_DAT: &str = r#"# Cu
# absorbing
#  POT  SCF 100  5.5000   0, core-hole, AFOLP (folp(0)= 1.150)
"#;
}
