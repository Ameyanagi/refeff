use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{IoError, Result};

const MAX_INCLUDE_DEPTH: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub path: PathBuf,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineKind {
    Card {
        keyword: String,
        args: Vec<String>,
        raw_args: String,
    },
    SectionData {
        section: String,
        fields: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeffLine {
    pub location: SourceLocation,
    pub raw: String,
    pub kind: LineKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeffInput {
    pub source: PathBuf,
    pub lines: Vec<FeffLine>,
}

impl FeffInput {
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self> {
        let source = path.as_ref().to_path_buf();
        let mut reader = Reader::default();
        let lines = reader.read_file(&source, 0)?;
        Ok(Self { source, lines })
    }

    pub fn parse_str(source_name: impl Into<PathBuf>, input: &str) -> Result<Self> {
        let source = source_name.into();
        let lines = parse_logical_lines(&source, input)?;
        Ok(Self { source, lines })
    }

    pub fn cards(&self) -> impl Iterator<Item = &FeffLine> {
        self.lines
            .iter()
            .filter(|line| matches!(line.kind, LineKind::Card { .. }))
    }

    pub fn card(&self, keyword: &str) -> Option<&FeffLine> {
        let keyword = keyword.to_ascii_uppercase();
        self.cards().find(|line| match &line.kind {
            LineKind::Card { keyword: found, .. } => found == &keyword,
            LineKind::SectionData { .. } => false,
        })
    }

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

        let is_card = first
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic());
        let kind = if is_card {
            let keyword = first.to_ascii_uppercase();
            if keyword == "END" {
                active_section = None;
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
            | "CONFIG"
            | "STRETCHES"
            | "ANGLES"
    )
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
    fn parses_cards_and_section_rows() {
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
        )
        .expect("parse");

        assert_eq!(input.card("edge").unwrap().raw, "EDGE K");
        assert_eq!(input.section_rows("POTENTIALS").count(), 2);
        assert_eq!(input.section_rows("ATOMS").count(), 1);
    }

    #[test]
    fn expands_include_files_relative_to_parent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let include_path = dir.path().join("more.inp");
        let root_path = dir.path().join("feff.inp");
        fs::write(&include_path, "EDGE K\n").expect("write include");
        let mut root = fs::File::create(&root_path).expect("create root");
        writeln!(root, "include more.inp").expect("write root");

        let parsed = FeffInput::parse_file(&root_path).expect("parse include");
        assert!(parsed.card("EDGE").is_some());
    }
}
