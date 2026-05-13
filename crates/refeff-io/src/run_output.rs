//! FEFF run stdout/stderr diagnostic output support.
//!
//! FEFF modules report progress through `wlog`, which writes both terminal
//! output and the currently-open unit 11 log. Some module-completion messages
//! can therefore appear in `feff.stdout` or, if unit 11 has already been
//! closed, a compiler-created `fort.11`. Fortran runtimes also write
//! floating-point exception notes to stderr. This module keeps those files
//! line-preserving while extracting the common module-completion and exception
//! metadata used by compatibility tests.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

const MODULE_START_PREFIX: &str = "Calculating ";
const MODULE_DONE_PREFIX: &str = "Done with module:";
const FPE_PREFIX: &str = "Note: The following floating-point exceptions are signalling:";

/// Parsed FEFF stdout-like diagnostic output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunStdoutData {
    /// Text lines in file order, without line terminators.
    pub lines: Vec<String>,
    /// Original line ending for each line; empty means render all lines with LF.
    pub line_endings: Vec<RunLineEnding>,
    /// Progress events extracted from the lines.
    pub module_events: Vec<RunModuleEvent>,
}

/// Parsed FEFF stderr-like diagnostic output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunStderrData {
    /// Text lines in file order, without line terminators.
    pub lines: Vec<String>,
    /// Original line ending for each line; empty means render all lines with LF.
    pub line_endings: Vec<RunLineEnding>,
    /// Floating-point exception notes emitted by the Fortran runtime.
    pub floating_point_notes: Vec<FloatingPointNote>,
}

/// Line terminator style preserved for FEFF terminal/log output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunLineEnding {
    /// No trailing line terminator.
    None,
    /// Unix line feed (`\n`).
    Lf,
    /// Carriage return plus line feed (`\r\n`).
    CrLf,
}

/// Module-progress event extracted from FEFF stdout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunModuleEvent {
    /// 1-based source line number.
    pub line: usize,
    /// Event kind.
    pub kind: RunModuleEventKind,
    /// Human-readable module message after removing the FEFF prefix.
    pub message: String,
}

/// FEFF module-progress event kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunModuleEventKind {
    /// A module calculation start line, for example `Calculating atomic potentials ...`.
    Start,
    /// A module completion line, for example `Done with module: potentials.`.
    Completed,
}

/// Floating-point exception note extracted from FEFF stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloatingPointNote {
    /// 1-based source line number.
    pub line: usize,
    /// Runtime exception flag names, such as `IEEE_UNDERFLOW_FLAG`.
    pub flags: Vec<String>,
}

impl RunStdoutData {
    /// Number of output lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Number of module-completion events.
    #[must_use]
    pub fn completion_count(&self) -> usize {
        self.module_events
            .iter()
            .filter(|event| event.kind == RunModuleEventKind::Completed)
            .count()
    }
}

impl RunStderrData {
    /// Number of output lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Number of floating-point exception note lines.
    #[must_use]
    pub fn floating_point_note_count(&self) -> usize {
        self.floating_point_notes.len()
    }
}

/// Parse FEFF stdout text.
pub fn parse_run_stdout(text: &str) -> Result<RunStdoutData> {
    let (lines, line_endings) = parse_lines(text);
    let module_events = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| parse_module_event(index + 1, line))
        .collect();
    Ok(RunStdoutData {
        lines,
        line_endings,
        module_events,
    })
}

/// Render FEFF stdout text.
pub fn run_stdout_string(data: &RunStdoutData) -> Result<String> {
    validate_lines("stdout", &data.lines, &data.line_endings)?;
    lines_string(&data.lines, &data.line_endings)
}

/// Read FEFF stdout text from a file.
pub fn read_run_stdout(path: impl AsRef<Path>) -> Result<RunStdoutData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_run_stdout(&text)
}

/// Write FEFF stdout text to a file.
pub fn write_run_stdout(path: impl AsRef<Path>, data: &RunStdoutData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, run_stdout_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `fort.11` scratch/progress text.
pub fn parse_fort11(text: &str) -> Result<RunStdoutData> {
    parse_run_stdout(text)
}

/// Read FEFF `fort.11` scratch/progress text from a file.
pub fn read_fort11(path: impl AsRef<Path>) -> Result<RunStdoutData> {
    read_run_stdout(path)
}

/// Parse FEFF stderr text.
pub fn parse_run_stderr(text: &str) -> Result<RunStderrData> {
    let (lines, line_endings) = parse_lines(text);
    let floating_point_notes = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| parse_floating_point_note(index + 1, line))
        .collect();
    Ok(RunStderrData {
        lines,
        line_endings,
        floating_point_notes,
    })
}

/// Render FEFF stderr text.
pub fn run_stderr_string(data: &RunStderrData) -> Result<String> {
    validate_lines("stderr", &data.lines, &data.line_endings)?;
    lines_string(&data.lines, &data.line_endings)
}

/// Read FEFF stderr text from a file.
pub fn read_run_stderr(path: impl AsRef<Path>) -> Result<RunStderrData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_run_stderr(&text)
}

/// Write FEFF stderr text to a file.
pub fn write_run_stderr(path: impl AsRef<Path>, data: &RunStderrData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, run_stderr_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_module_event(line_number: usize, line: &str) -> Option<RunModuleEvent> {
    let trimmed = line.trim();
    if let Some(message) = trimmed.strip_prefix(MODULE_DONE_PREFIX) {
        return Some(RunModuleEvent {
            line: line_number,
            kind: RunModuleEventKind::Completed,
            message: clean_module_message(message),
        });
    }
    trimmed
        .strip_prefix(MODULE_START_PREFIX)
        .map(|message| RunModuleEvent {
            line: line_number,
            kind: RunModuleEventKind::Start,
            message: clean_module_message(message),
        })
}

fn parse_floating_point_note(line_number: usize, line: &str) -> Option<FloatingPointNote> {
    let flags = line
        .trim()
        .strip_prefix(FPE_PREFIX)?
        .split_whitespace()
        .map(|flag| flag.trim_end_matches(',').to_string())
        .filter(|flag| !flag.is_empty())
        .collect::<Vec<_>>();
    Some(FloatingPointNote {
        line: line_number,
        flags,
    })
}

fn parse_lines(text: &str) -> (Vec<String>, Vec<RunLineEnding>) {
    let mut lines = Vec::new();
    let mut endings = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            let (end, ending) = if index > start && bytes[index - 1] == b'\r' {
                (index - 1, RunLineEnding::CrLf)
            } else {
                (index, RunLineEnding::Lf)
            };
            lines.push(text[start..end].to_string());
            endings.push(ending);
            index += 1;
            start = index;
        } else {
            index += 1;
        }
    }
    if start < text.len() {
        lines.push(text[start..].to_string());
        endings.push(RunLineEnding::None);
    }
    (lines, endings)
}

fn lines_string(lines: &[String], line_endings: &[RunLineEnding]) -> Result<String> {
    let mut out = String::new();
    for (index, line) in lines.iter().enumerate() {
        write!(out, "{line}")?;
        match line_endings
            .get(index)
            .copied()
            .unwrap_or(RunLineEnding::Lf)
        {
            RunLineEnding::None => {}
            RunLineEnding::Lf => out.push('\n'),
            RunLineEnding::CrLf => out.push_str("\r\n"),
        }
    }
    Ok(out)
}

fn clean_module_message(message: &str) -> String {
    message
        .trim()
        .trim_end_matches('.')
        .trim_end_matches("...")
        .trim()
        .to_string()
}

fn validate_lines(
    field: &'static str,
    lines: &[String],
    line_endings: &[RunLineEnding],
) -> Result<()> {
    if !line_endings.is_empty() && line_endings.len() != lines.len() {
        return Err(IoError::InvalidRunOutput {
            field,
            message: format!(
                "line ending count {} does not match line count {}",
                line_endings.len(),
                lines.len()
            ),
        });
    }
    if lines
        .iter()
        .any(|line| line.contains('\n') || line.contains('\r'))
    {
        Err(IoError::InvalidRunOutput {
            field,
            message: "line data must not contain embedded line terminators".to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stdout_module_events() -> Result<()> {
        let parsed = parse_run_stdout(STDOUT)?;
        assert_eq!(parsed.line_count(), 5);
        assert_eq!(parsed.module_events.len(), 4);
        assert_eq!(parsed.completion_count(), 2);
        assert_eq!(parsed.module_events[0].kind, RunModuleEventKind::Start);
        assert_eq!(parsed.module_events[0].message, "atomic potentials");
        assert_eq!(parsed.module_events[1].kind, RunModuleEventKind::Completed);
        assert_eq!(parsed.module_events[1].message, "atomic potentials");
        assert_eq!(parsed.module_events[2].message, "LDOS");
        assert_eq!(
            parsed.module_events[3].message,
            "screened core-hole potential"
        );
        let rendered = run_stdout_string(&parsed)?;
        assert_eq!(rendered, STDOUT);
        assert_eq!(parse_run_stdout(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_fort11_completion_message() -> Result<()> {
        let text = "Done with module: screened core-hole potential.\r\n\n";
        let parsed = parse_fort11(text)?;
        assert_eq!(parsed.line_count(), 2);
        assert_eq!(parsed.completion_count(), 1);
        assert_eq!(
            parsed.module_events[0].message,
            "screened core-hole potential"
        );
        assert_eq!(run_stdout_string(&parsed)?, text);
        Ok(())
    }

    #[test]
    fn parses_stderr_floating_point_notes() -> Result<()> {
        let parsed = parse_run_stderr(STDERR)?;
        assert_eq!(parsed.line_count(), 2);
        assert_eq!(parsed.floating_point_note_count(), 2);
        assert_eq!(parsed.floating_point_notes[0].line, 1);
        assert_eq!(
            parsed.floating_point_notes[0].flags,
            vec!["IEEE_UNDERFLOW_FLAG".to_string()]
        );
        assert_eq!(
            parsed.floating_point_notes[1].flags,
            vec![
                "IEEE_INVALID_FLAG".to_string(),
                "IEEE_DIVIDE_BY_ZERO".to_string()
            ]
        );
        let rendered = run_stderr_string(&parsed)?;
        assert_eq!(rendered, STDERR);
        assert_eq!(parse_run_stderr(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_embedded_line_terminators() {
        let data = RunStdoutData {
            lines: vec!["bad\nline".to_string()],
            line_endings: Vec::new(),
            module_events: Vec::new(),
        };
        assert!(run_stdout_string(&data).is_err());
        let data = RunStderrData {
            lines: vec!["one".to_string()],
            line_endings: vec![RunLineEnding::Lf, RunLineEnding::Lf],
            floating_point_notes: Vec::new(),
        };
        assert!(run_stderr_string(&data).is_err());
    }

    const STDOUT: &str = r#"Launching FEFF version FEFF 10.0.0
Calculating atomic potentials ...
Done with module: atomic potentials.
 Calculating LDOS ...
Done with module: screened core-hole potential.
"#;

    const STDERR: &str = r#"Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG
Note: The following floating-point exceptions are signalling: IEEE_INVALID_FLAG, IEEE_DIVIDE_BY_ZERO
"#;
}
