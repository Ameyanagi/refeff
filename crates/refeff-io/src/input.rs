//! FEFF input reader and line tokenizer.
//!
//! FEFF input is line-oriented, permits nested `include`/`load` files, treats
//! `*`, `;`, `%`, and `#` as full-line comments, and uses `!`, `#`, and `%` as
//! inline comment markers outside protected delimiters.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{IoError, Result};

const MAX_INCLUDE_DEPTH: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// Source file containing the logical line.
    pub path: PathBuf,
    /// One-based source line number.
    pub line: usize,
}

/// Parsed logical FEFF line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineKind {
    /// A card line such as `EDGE K` or `CONTROL 1 1 1 1 1 1`.
    Card {
        /// Canonical uppercase keyword.
        keyword: String,
        /// Tokenized arguments after the keyword.
        args: Vec<String>,
        /// Un-tokenized argument text after the keyword.
        raw_args: String,
    },
    /// A data row belonging to the nearest active block card.
    SectionData {
        /// Canonical uppercase block card name.
        section: String,
        /// Tokenized row fields.
        fields: Vec<String>,
    },
}

/// One active FEFF input line after include expansion and comment removal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeffLine {
    /// Original file and line number.
    pub location: SourceLocation,
    /// Comment-stripped logical text.
    pub raw: String,
    /// Parsed card or section row.
    pub kind: LineKind,
}

/// Parsed FEFF input file with includes expanded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeffInput {
    /// Root input file.
    pub source: PathBuf,
    /// Active logical lines in FEFF read order.
    pub lines: Vec<FeffLine>,
}

impl FeffInput {
    /// Parse a FEFF input file, expanding nested `include` and `load` files.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self> {
        let source = path.as_ref().to_path_buf();
        let mut reader = Reader::default();
        let lines = reader.read_file(&source, 0)?;
        Ok(Self { source, lines })
    }

    /// Parse FEFF input from a string. This is mainly used for unit tests.
    pub fn parse_str(source_name: impl Into<PathBuf>, input: &str) -> Result<Self> {
        let source = source_name.into();
        let lines = parse_logical_lines(&source, input)?;
        Ok(Self { source, lines })
    }

    /// Iterate over card lines only.
    pub fn cards(&self) -> impl Iterator<Item = &FeffLine> {
        self.lines
            .iter()
            .filter(|line| matches!(line.kind, LineKind::Card { .. }))
    }

    /// Return the first card with a case-insensitive keyword match.
    pub fn card(&self, keyword: &str) -> Option<&FeffLine> {
        let keyword = keyword.to_ascii_uppercase();
        self.cards().find(|line| match &line.kind {
            LineKind::Card { keyword: found, .. } => found == &keyword,
            LineKind::SectionData { .. } => false,
        })
    }

    /// Iterate over section rows belonging to a block card.
    pub fn section_rows<'a>(&'a self, section: &'a str) -> impl Iterator<Item = &'a FeffLine> + 'a {
        let section = section.to_ascii_uppercase();
        self.lines.iter().filter(move |line| match &line.kind {
            LineKind::SectionData { section: found, .. } => found == &section,
            LineKind::Card { .. } => false,
        })
    }
}

#[derive(Default)]
struct Reader {
    stack: Vec<PathBuf>,
    seen: HashSet<PathBuf>,
}

impl Reader {
    fn read_file(&mut self, path: &Path, depth: usize) -> Result<Vec<FeffLine>> {
        if depth >= MAX_INCLUDE_DEPTH {
            return Err(IoError::IncludeDepth {
                path: path.to_path_buf(),
            });
        }

        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if !self.seen.insert(canonical.clone()) {
            return Err(IoError::RecursiveInclude { path: canonical });
        }

        self.stack.push(canonical.clone());
        let text = fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
        let mut parsed = Vec::new();
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

        for line in parse_logical_lines(path, &text)? {
            if let LineKind::Card { keyword, args, .. } = &line.kind
                && (keyword == "INCLUDE" || keyword == "LOAD")
            {
                let include = args.first().ok_or_else(|| IoError::Parse {
                    path: line.location.path.clone(),
                    line: line.location.line,
                    message: format!("{keyword} requires a file name"),
                })?;
                let include_path = base_dir.join(strip_delimiters(include));
                parsed.extend(self.read_file(&include_path, depth + 1)?);
                continue;
            }

            parsed.push(line);
        }

        self.stack.pop();
        self.seen.remove(&canonical);
        Ok(parsed)
    }
}

fn parse_logical_lines(path: &Path, input: &str) -> Result<Vec<FeffLine>> {
    let mut lines = Vec::new();
    let mut active_section: Option<String> = None;

    for (idx, raw_line) in input.lines().enumerate() {
        let line_number = idx + 1;
        let raw = raw_line.replace('\t', " ");
        let trimmed = raw.trim_start();

        if trimmed.is_empty() || is_comment_line(trimmed) {
            continue;
        }

        let uncommented = strip_inline_comment(trimmed).trim().to_string();
        if uncommented.is_empty() {
            continue;
        }

        let tokens = bwords(&uncommented);
        if tokens.is_empty() {
            continue;
        }

        let first = &tokens[0];
        let location = SourceLocation {
            path: path.to_path_buf(),
            line: line_number,
        };

        let egrid_payload =
            active_section.as_deref() == Some("EGRID") && is_egrid_payload_keyword(first);
        let density_payload =
            active_section.as_deref() == Some("DENSITY") && is_density_payload_keyword(first);
        let is_card = !egrid_payload
            && !density_payload
            && first
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic());
        let kind = if is_card {
            let keyword = canonical_keyword(&first.to_ascii_uppercase()).to_string();
            if keyword == "END" {
                lines.push(FeffLine {
                    location,
                    raw: uncommented,
                    kind: LineKind::Card {
                        keyword,
                        args: tokens[1..].to_vec(),
                        raw_args: String::new(),
                    },
                });
                break;
            } else if is_block_card(&keyword) {
                active_section = Some(keyword.clone());
            } else if keyword != "TITLE" {
                active_section = None;
            }

            let raw_args = uncommented[first.len()..].trim_start().to_string();
            LineKind::Card {
                keyword,
                args: tokens[1..].to_vec(),
                raw_args,
            }
        } else if let Some(section) = active_section.clone() {
            LineKind::SectionData {
                section,
                fields: tokens,
            }
        } else {
            return Err(IoError::Parse {
                path: path.to_path_buf(),
                line: line_number,
                message: "data row found outside a block card".to_string(),
            });
        };

        lines.push(FeffLine {
            location,
            raw: uncommented,
            kind,
        });
    }

    Ok(lines)
}

fn is_comment_line(line: &str) -> bool {
    line.chars()
        .next()
        .is_some_and(|ch| matches!(ch, ';' | '*' | '%' | '#'))
}

fn is_block_card(keyword: &str) -> bool {
    matches!(
        keyword,
        "ATOMS"
            | "POTENTIALS"
            | "OVERLAP"
            | "LATTICE"
            | "EGRID"
            | "DENSITY"
            | "CONFIG"
            | "STRETCHES"
            | "ANGLES"
            | "ELNES"
            | "EXELFS"
            | "NRIXS"
            | "MDFF"
    )
}

fn is_egrid_payload_keyword(keyword: &str) -> bool {
    matches!(
        keyword.to_ascii_lowercase().as_str(),
        "e_grid" | "k_grid" | "exp_grid" | "user_grid"
    )
}

fn is_density_payload_keyword(keyword: &str) -> bool {
    matches!(
        keyword.to_ascii_lowercase().as_str(),
        "line" | "plane" | "volume"
    )
}

fn canonical_keyword(keyword: &str) -> &str {
    match keyword {
        "ATOM" => "ATOMS",
        "CONF" | "CONFIGURATION" => "CONFIG",
        "DENS" => "DENSITY",
        "POTENTIAL" => "POTENTIALS",
        other => other,
    }
}

fn strip_delimiters(value: &str) -> &str {
    let pairs = [
        ('"', '"'),
        ('\'', '\''),
        ('{', '}'),
        ('(', ')'),
        ('<', '>'),
        ('[', ']'),
    ];
    for (open, close) in pairs {
        if value.starts_with(open) && value.ends_with(close) && value.len() >= 2 {
            return &value[1..value.len() - 1];
        }
    }
    value
}

/// Strip FEFF inline comments outside protected delimiters.
pub fn strip_inline_comment(line: &str) -> String {
    let open = ['[', '{', '"', '\'', '('];
    let close = [']', '}', '"', '\'', ')'];
    let mut protected: Option<usize> = None;

    for (byte_idx, ch) in line.char_indices() {
        if let Some(idx) = protected {
            if ch == close[idx] {
                protected = None;
            }
            continue;
        }

        if let Some(idx) = open.iter().position(|candidate| *candidate == ch) {
            protected = Some(idx);
            continue;
        }

        if matches!(ch, '!' | '#' | '%') {
            return line[..byte_idx].to_string();
        }
    }

    line.to_string()
}

/// Tokenize using FEFF `BWORDS` semantics: whitespace separates words and
/// commas may introduce empty fields.
pub fn bwords(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    let mut comma_found = true;

    for (idx, ch) in line.char_indices() {
        if ch == ' ' || ch == '\t' {
            if let Some(begin) = start.take() {
                words.push(line[begin..idx].to_string());
                comma_found = false;
            }
        } else if ch == ',' {
            if let Some(begin) = start.take() {
                words.push(line[begin..idx].to_string());
            } else if comma_found {
                words.push(String::new());
            }
            comma_found = true;
        } else if start.is_none() {
            start = Some(idx);
        }
    }

    if let Some(begin) = start {
        words.push(line[begin..].trim_end().to_string());
    }

    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context as _, ensure};
    use std::io::Write;

    #[test]
    fn removes_feff_comments_but_keeps_protected_text() {
        assert_eq!(strip_inline_comment("EDGE K ! comment"), "EDGE K ");
        assert_eq!(
            strip_inline_comment("TITLE \"a # b\" # comment"),
            "TITLE \"a # b\" "
        );
        assert_eq!(
            strip_inline_comment("ATOMS * inline star survives"),
            "ATOMS * inline star survives"
        );
    }

    #[test]
    fn tokenizes_like_bwords_for_commas_and_spaces() {
        assert_eq!(bwords("A  B,C,,D"), vec!["A", "B", "C", "", "D"]);
    }

    #[test]
    fn parses_cards_and_section_rows() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
* comment
EDGE K
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS * comment after card is data to FEFF but not a data row
0.0 0.0 0.0 0 Cu
END
"#,
        )?;

        let edge = input.card("edge").context("missing EDGE card")?;
        ensure!(edge.raw == "EDGE K", "unexpected EDGE card: {}", edge.raw);
        ensure!(
            input.section_rows("POTENTIALS").count() == 2,
            "unexpected POTENTIALS row count"
        );
        ensure!(
            input.section_rows("ATOMS").count() == 1,
            "unexpected ATOMS row count"
        );
        Ok(())
    }

    #[test]
    fn parses_density_payload_as_section_rows() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DENSITY
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
plane plane.dat 0.0 0.0 0.0
1.0 0.0 0.0 11
0.0 1.0 0.0 12
EDGE K
END
"#,
        )?;

        ensure!(
            input.section_rows("DENSITY").count() == 5,
            "unexpected DENSITY row count"
        );
        ensure!(
            input.card("EDGE").is_some(),
            "DENSITY block did not terminate before EDGE"
        );
        Ok(())
    }

    #[test]
    fn parses_density_alias_as_section_rows() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DENS
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
EDGE K
END
"#,
        )?;

        ensure!(
            input.card("DENSITY").is_some(),
            "DENS alias did not canonicalize to DENSITY"
        );
        ensure!(
            input.section_rows("DENSITY").count() == 2,
            "unexpected DENSITY row count"
        );
        ensure!(
            input.card("EDGE").is_some(),
            "DENSITY block did not terminate before EDGE"
        );
        Ok(())
    }

    #[test]
    fn parses_configuration_alias_as_section_rows() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONF card 1
2 1 0
EDGE K
END
"#,
        )?;

        ensure!(
            input.card("CONFIG").is_some(),
            "CONF alias did not canonicalize to CONFIG"
        );
        ensure!(
            input.section_rows("CONFIG").count() == 1,
            "unexpected CONFIG row count"
        );
        ensure!(
            input.card("EDGE").is_some(),
            "CONFIG block did not terminate before EDGE"
        );
        Ok(())
    }

    #[test]
    fn expands_include_files_relative_to_parent() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let include_path = dir.path().join("more.inp");
        let root_path = dir.path().join("feff.inp");
        fs::write(&include_path, "EDGE K\n")?;
        let mut root = fs::File::create(&root_path)?;
        writeln!(root, "include more.inp")?;

        let parsed = FeffInput::parse_file(&root_path)?;
        ensure!(parsed.card("EDGE").is_some(), "missing included EDGE card");
        Ok(())
    }
}
