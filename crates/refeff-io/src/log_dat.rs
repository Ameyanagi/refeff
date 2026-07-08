//! FEFF `log.dat` run-summary and module-log codecs.
//!
//! FEFF writes a short log during `rdinp` containing the version banner,
//! warnings, optional core-hole lifetime, title summary, enabled feature list,
//! and cards used by the calculation. Failed input parsing can stop before the
//! calculation-summary block, so this codec also accepts banner-plus-message
//! logs.
//!
//! Later FEFF modules also write simpler `log1.dat`, `logdos.dat`,
//! `logscreen.dat`, and similar files. Those module logs do not share the
//! structured `rdinp` banner, so they are represented as raw line-preserving
//! text via [`ModuleLogData`].

use std::fmt::Write as _;
use std::path::Path;

use refeff_core::{GenfmtJasPathOutputs, GenfmtOrdinaryPathOutputs, GenfmtPathOutputSummary};

use crate::error::{IoError, Result};
use crate::format::fortran_zero_scaled_exp;

const VERSION_PREFIX: &str = "Launching FEFF version ";
const CORE_HOLE_PREFIX: &str = "Core hole lifetime is";
const YOUR_CALCULATION: &str = "Your calculation:";
const USING_PREFIX: &str = "Using:";
const USING_CARDS_PREFIX: &str = "Using cards:";

/// Parsed FEFF `log.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct LogDatData {
    /// FEFF version text from the launch banner, for example `FEFF 10.0.0`.
    pub version: String,
    /// Messages before the optional core-hole lifetime line.
    pub preamble_lines: Vec<String>,
    /// Optional core-hole lifetime in eV.
    pub core_hole_lifetime_ev: Option<f64>,
    /// Messages after the core-hole lifetime and before `Your calculation:`.
    pub post_core_lines: Vec<String>,
    /// FEFF title records printed below `Your calculation:`.
    pub titles: Vec<String>,
    /// Calculation summary line, such as `Cu K edge XANES using RPA corehole`.
    pub calculation_summary: Option<String>,
    /// Enabled feature descriptions parsed from the starred `Using:` line.
    pub features: Vec<String>,
    /// FEFF card names parsed from the `Using cards:` line.
    pub cards: Vec<String>,
    /// Any lines after the card summary.
    pub trailing_lines: Vec<String>,
}

/// Raw FEFF module-log contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleLogData {
    /// Text lines in file order, without line terminators.
    pub lines: Vec<String>,
    /// Original line terminator for each line, preserving mixed `LF`, `CRLF`,
    /// and final unterminated lines from FEFF module logs.
    pub line_terminators: Vec<String>,
}

/// FEFF GENFMT path-log format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenfmtPathLogMode {
    /// Ordinary `GENFMT/genfmtsub.f90` logging, gated by `ipr3 > 2`.
    Ordinary,
    /// `GENFMT/genfmtjas.f90` logging, always emitted.
    Jas,
}

/// One FEFF GENFMT path progress-log row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtPathLogEntry {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// Whether FEFF retained this path for output.
    pub retained: bool,
    /// FEFF `crit`, the curved-wave amplitude ratio in percent.
    pub criterion_percent: f64,
    /// FEFF `deg`.
    pub degeneracy: f64,
    /// FEFF `nleg`.
    pub leg_count: usize,
    /// FEFF `reff*bohr`, printed in Angstrom.
    pub effective_half_path_length_angstrom: f64,
}

/// Inputs for FEFF GENFMT path progress-log generation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathLogInput<'a> {
    /// FEFF logging variant.
    pub mode: GenfmtPathLogMode,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, printed in the table heading.
    pub curved_wave_criterion_percent: f64,
    /// Retained and neglected paths in the order FEFF examined them.
    pub entries: &'a [GenfmtPathLogEntry],
}

impl From<GenfmtPathOutputSummary> for GenfmtPathLogEntry {
    fn from(summary: GenfmtPathOutputSummary) -> Self {
        Self::from(&summary)
    }
}

impl From<&GenfmtPathOutputSummary> for GenfmtPathLogEntry {
    fn from(summary: &GenfmtPathOutputSummary) -> Self {
        Self {
            path_index: summary.path_index,
            retained: summary.retained,
            criterion_percent: summary.criterion_percent,
            degeneracy: summary.degeneracy,
            leg_count: summary.leg_count,
            effective_half_path_length_angstrom: summary.effective_half_path_length_angstrom,
        }
    }
}

impl LogDatData {
    /// Whether this log contains the calculation-summary block.
    #[must_use]
    pub fn has_calculation_summary(&self) -> bool {
        self.calculation_summary.is_some()
    }
}

impl ModuleLogData {
    /// Number of log lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Whether this module log contains no text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// Render the FEFF GENFMT path progress lines as raw module-log data.
///
/// Ordinary GENFMT emits these lines only when `ipr3 > 2`, matching the final
/// `write_to_screen = ipr3 .gt. 2` assignment in `genfmtsub.f90`. GENFMTJAS
/// always emits them.
pub fn genfmt_path_module_log(input: GenfmtPathLogInput<'_>) -> Result<ModuleLogData> {
    validate_genfmt_path_log_input(input)?;

    let emit = matches!(input.mode, GenfmtPathLogMode::Jas) || input.print_level > 2;
    if !emit {
        return Ok(ModuleLogData {
            lines: Vec::new(),
            line_terminators: Vec::new(),
        });
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "    Curved wave chi amplitude ratio{:7.2}%",
        input.curved_wave_criterion_percent
    ));
    if input.print_level <= 0 {
        lines.push(format!(
            "    Discard feff.dat for paths with cw ratio <{:7.2}%",
            2.0 * input.curved_wave_criterion_percent / 3.0
        ));
    }
    lines.push("    path  cw ratio     deg    nleg  reff".to_string());

    let mut retained_count = 0_usize;
    for entry in input.entries {
        if entry.retained {
            retained_count += 1;
        }
        lines.push(genfmt_path_log_entry_line(input.mode, *entry));
    }
    lines.push(format!(
        " {:4} paths kept, {:4} examined.",
        retained_count,
        input.entries.len()
    ));

    let line_terminators = vec!["\n".to_string(); lines.len()];
    Ok(ModuleLogData {
        lines,
        line_terminators,
    })
}

/// Render ordinary FEFF GENFMT collector output as path progress module-log data.
pub fn genfmt_ordinary_path_outputs_module_log(
    print_level: i32,
    curved_wave_criterion_percent: f64,
    outputs: &GenfmtOrdinaryPathOutputs,
) -> Result<ModuleLogData> {
    genfmt_path_outputs_module_log(
        GenfmtPathLogMode::Ordinary,
        print_level,
        curved_wave_criterion_percent,
        &outputs.path_summaries,
    )
}

/// Render FEFF GENFMTJAS collector output as path progress module-log data.
pub fn genfmt_jas_path_outputs_module_log(
    print_level: i32,
    curved_wave_criterion_percent: f64,
    outputs: &GenfmtJasPathOutputs,
) -> Result<ModuleLogData> {
    genfmt_path_outputs_module_log(
        GenfmtPathLogMode::Jas,
        print_level,
        curved_wave_criterion_percent,
        &outputs.path_summaries,
    )
}

/// Render the deterministic FEFF POT module wrapper used for cached POT runs.
#[must_use]
pub fn pot_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating SCF potentials ...".to_string(),
            "FEFF-serial using 1 thread.".to_string(),
            "Done with module: potentials.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(); 3],
    }
}

/// Whether a `log1.dat` payload is the ATOM wrapper, not the POT wrapper.
#[must_use]
pub fn is_atomic_potential_module_log(data: &ModuleLogData) -> bool {
    data.lines
        .iter()
        .any(|line| line.contains("Calculating atomic potentials ..."))
        && data
            .lines
            .iter()
            .any(|line| line.contains("Done with module: atomic potentials."))
}

/// Choose the `log1.dat` payload after a cached POT stage has run.
///
/// FEFF reuses `log1.dat` for ATOM and POT. Cached directories may still carry
/// an ATOM wrapper from the preceding stage, so cached POT orchestration
/// replaces that specific wrapper and preserves existing non-ATOM POT detail.
#[must_use]
pub fn cached_pot_stage_module_log(existing: Option<&ModuleLogData>) -> ModuleLogData {
    existing
        .filter(|data| !is_atomic_potential_module_log(data))
        .cloned()
        .unwrap_or_else(pot_module_log)
}

/// Render FEFF-compatible `log.dat` text.
pub fn log_dat_string(data: &LogDatData) -> Result<String> {
    validate_log_dat(data)?;

    let mut out = String::new();
    writeln!(out, "{VERSION_PREFIX}{}", data.version)?;
    for line in &data.preamble_lines {
        writeln!(out, "{line}")?;
    }
    if let Some(core_hole) = data.core_hole_lifetime_ev {
        writeln!(out, "Core hole lifetime is {core_hole:7.3} eV.")?;
    }
    for line in &data.post_core_lines {
        writeln!(out, "{line}")?;
    }

    if let Some(summary) = &data.calculation_summary {
        writeln!(out, "{YOUR_CALCULATION}")?;
        for title in &data.titles {
            writeln!(out, "{title}")?;
        }
        writeln!(out, "{summary}")?;
        if data.features.is_empty() {
            writeln!(out, "{USING_PREFIX}")?;
        } else {
            writeln!(out, "{USING_PREFIX}     * {}", data.features.join("   * "))?;
        }
        writeln!(out, "{USING_CARDS_PREFIX}   {}", data.cards.join(" "))?;
    }

    for line in &data.trailing_lines {
        writeln!(out, "{line}")?;
    }
    Ok(out)
}

/// Render FEFF-compatible raw module-log text.
pub fn module_log_dat_string(data: &ModuleLogData) -> Result<String> {
    if data.lines.len() != data.line_terminators.len() {
        return Err(IoError::Parse {
            path: "module log".into(),
            line: 0,
            message: format!(
                "line/terminator count mismatch: {} line(s), {} terminator(s)",
                data.lines.len(),
                data.line_terminators.len()
            ),
        });
    }
    let mut out = String::new();
    for (line, terminator) in data.lines.iter().zip(data.line_terminators.iter()) {
        out.push_str(line);
        out.push_str(terminator);
    }
    Ok(out)
}

/// Parse FEFF `log.dat` text.
pub fn parse_log_dat(text: &str) -> Result<LogDatData> {
    let mut lines = text.lines().enumerate();
    let (_, version_line) = lines
        .next()
        .ok_or(IoError::LogDatMissing { field: "version" })?;
    let version = version_line
        .strip_prefix(VERSION_PREFIX)
        .ok_or_else(|| invalid_log_dat("version", "missing FEFF launch banner"))?
        .trim()
        .to_string();
    if version.is_empty() {
        return Err(invalid_log_dat("version", "version must not be empty"));
    }

    let mut preamble_lines = Vec::new();
    let mut core_hole_lifetime_ev = None;
    let mut post_core_lines = Vec::new();
    let mut before_core = true;
    let mut calculation_lines = Vec::new();

    for (index, raw) in lines {
        let line_number = index + 1;
        let line = raw.trim_end();
        if line == YOUR_CALCULATION {
            calculation_lines = text
                .lines()
                .skip(line_number)
                .map(str::trim_end)
                .map(str::to_string)
                .collect();
            break;
        }

        if line.starts_with(CORE_HOLE_PREFIX) {
            if core_hole_lifetime_ev.is_some() {
                return Err(invalid_log_dat(
                    "core_hole_lifetime_ev",
                    "duplicate core-hole lifetime line",
                ));
            }
            core_hole_lifetime_ev = Some(parse_core_hole_lifetime(line_number, line)?);
            before_core = false;
        } else if before_core {
            preamble_lines.push(line.to_string());
        } else {
            post_core_lines.push(line.to_string());
        }
    }

    let calculation = parse_calculation_block(&calculation_lines)?;

    let data = LogDatData {
        version,
        preamble_lines,
        core_hole_lifetime_ev,
        post_core_lines,
        titles: calculation.titles,
        calculation_summary: calculation.summary,
        features: calculation.features,
        cards: calculation.cards,
        trailing_lines: calculation.trailing_lines,
    };
    validate_log_dat(&data)?;
    Ok(data)
}

/// Parse a FEFF module log such as `log1.dat` or `logdos.dat`.
pub fn parse_module_log_dat(text: &str) -> Result<ModuleLogData> {
    let (lines, line_terminators) = split_module_log_lines(text);
    Ok(ModuleLogData {
        lines,
        line_terminators,
    })
}

fn split_module_log_lines(text: &str) -> (Vec<String>, Vec<String>) {
    let mut lines = Vec::new();
    let mut terminators = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\r' => {
                let next = if bytes.get(index + 1) == Some(&b'\n') {
                    index + 2
                } else {
                    index + 1
                };
                lines.push(text[start..index].to_string());
                terminators.push(text[index..next].to_string());
                index = next;
                start = next;
            }
            b'\n' => {
                let next = index + 1;
                lines.push(text[start..index].to_string());
                terminators.push(text[index..next].to_string());
                index = next;
                start = next;
            }
            _ => {
                index += 1;
            }
        }
    }
    if start < text.len() {
        lines.push(text[start..].to_string());
        terminators.push(String::new());
    }
    (lines, terminators)
}

/// Write FEFF `log.dat` text to a file.
pub fn write_log_dat(path: impl AsRef<Path>, data: &LogDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, log_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write a FEFF raw module log to a file.
pub fn write_module_log_dat(path: impl AsRef<Path>, data: &ModuleLogData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, module_log_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `log.dat` text from a file.
pub fn read_log_dat(path: impl AsRef<Path>) -> Result<LogDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_log_dat(&text)
}

/// Read a FEFF raw module log from a file.
pub fn read_module_log_dat(path: impl AsRef<Path>) -> Result<ModuleLogData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_module_log_dat(&text)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedCalculationBlock {
    titles: Vec<String>,
    summary: Option<String>,
    features: Vec<String>,
    cards: Vec<String>,
    trailing_lines: Vec<String>,
}

fn parse_calculation_block(lines: &[String]) -> Result<ParsedCalculationBlock> {
    if lines.is_empty() {
        return Ok(ParsedCalculationBlock {
            titles: Vec::new(),
            summary: None,
            features: Vec::new(),
            cards: Vec::new(),
            trailing_lines: Vec::new(),
        });
    }

    let using_index = lines
        .iter()
        .position(|line| line.starts_with(USING_PREFIX))
        .ok_or(IoError::LogDatMissing { field: "Using" })?;
    if using_index == 0 {
        return Err(invalid_log_dat(
            "calculation_summary",
            "calculation block must include a summary line before Using",
        ));
    }
    let using_cards_index = lines
        .iter()
        .skip(using_index + 1)
        .position(|line| line.starts_with(USING_CARDS_PREFIX))
        .map(|offset| using_index + 1 + offset)
        .ok_or(IoError::LogDatMissing {
            field: "Using cards",
        })?;
    if using_cards_index != using_index + 1 {
        return Err(invalid_log_dat(
            "Using cards",
            "Using cards must immediately follow Using",
        ));
    }

    let titles = lines[..using_index - 1].to_vec();
    let calculation_summary = Some(lines[using_index - 1].clone());
    let features = parse_features(&lines[using_index]);
    let cards = parse_cards(&lines[using_cards_index]);
    let trailing_lines = lines[using_cards_index + 1..].to_vec();
    Ok(ParsedCalculationBlock {
        titles,
        summary: calculation_summary,
        features,
        cards,
        trailing_lines,
    })
}

fn parse_core_hole_lifetime(line: usize, text: &str) -> Result<f64> {
    let token = text
        .split_whitespace()
        .find(|token| token.parse::<f64>().is_ok())
        .ok_or(IoError::LogDatMissing {
            field: "core_hole_lifetime_ev",
        })?;
    token.parse::<f64>().map_err(|_| IoError::LogDatParse {
        field: "core_hole_lifetime_ev",
        line,
        token: token.to_string(),
    })
}

fn parse_features(line: &str) -> Vec<String> {
    line.strip_prefix(USING_PREFIX)
        .unwrap_or("")
        .split('*')
        .map(str::trim)
        .filter(|feature| !feature.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_cards(line: &str) -> Vec<String> {
    line.strip_prefix(USING_CARDS_PREFIX)
        .unwrap_or("")
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn validate_log_dat(data: &LogDatData) -> Result<()> {
    if data.version.trim().is_empty() {
        return Err(invalid_log_dat("version", "version must not be empty"));
    }
    if let Some(core_hole) = data.core_hole_lifetime_ev
        && !core_hole.is_finite()
    {
        return Err(invalid_log_dat(
            "core_hole_lifetime_ev",
            "value must be finite",
        ));
    }
    if data.calculation_summary.is_some() {
        if data.cards.is_empty() {
            return Err(invalid_log_dat(
                "cards",
                "calculation summary logs must include at least one card",
            ));
        }
    } else if !data.titles.is_empty() || !data.features.is_empty() || !data.cards.is_empty() {
        return Err(invalid_log_dat(
            "calculation_summary",
            "titles, features, and cards require a calculation summary",
        ));
    }
    Ok(())
}

fn validate_genfmt_path_log_input(input: GenfmtPathLogInput<'_>) -> Result<()> {
    ensure_finite_log_value(
        "curved_wave_criterion_percent",
        input.curved_wave_criterion_percent,
    )?;
    for entry in input.entries {
        if entry.path_index == 0 {
            return Err(invalid_log_dat(
                "path_index",
                "GENFMT path index must be positive",
            ));
        }
        if entry.leg_count == 0 {
            return Err(invalid_log_dat(
                "nleg",
                "GENFMT path leg count must be positive",
            ));
        }
        ensure_finite_log_value("crit", entry.criterion_percent)?;
        ensure_finite_log_value("deg", entry.degeneracy)?;
        ensure_finite_log_value("reff", entry.effective_half_path_length_angstrom)?;
    }
    Ok(())
}

fn genfmt_path_outputs_module_log(
    mode: GenfmtPathLogMode,
    print_level: i32,
    curved_wave_criterion_percent: f64,
    path_summaries: &[GenfmtPathOutputSummary],
) -> Result<ModuleLogData> {
    let entries: Vec<_> = path_summaries
        .iter()
        .map(GenfmtPathLogEntry::from)
        .collect();
    genfmt_path_module_log(GenfmtPathLogInput {
        mode,
        print_level,
        curved_wave_criterion_percent,
        entries: &entries,
    })
}

fn genfmt_path_log_entry_line(mode: GenfmtPathLogMode, entry: GenfmtPathLogEntry) -> String {
    match (mode, entry.retained) {
        (GenfmtPathLogMode::Ordinary, true) => format!(
            "   {:4}{}{:10.3}{:6}{:9.4}",
            entry.path_index,
            fortran_zero_scaled_exp(entry.criterion_percent, 15, 4),
            entry.degeneracy,
            entry.leg_count,
            entry.effective_half_path_length_angstrom
        ),
        (GenfmtPathLogMode::Ordinary, false) => format!(
            "   {:4}{:10.3}{:10.3}{:6}{:9.4} neglected",
            entry.path_index,
            entry.criterion_percent,
            entry.degeneracy,
            entry.leg_count,
            entry.effective_half_path_length_angstrom
        ),
        (GenfmtPathLogMode::Jas, true) => format!(
            "   {:4}{:12.5}{:12.5}{:6}{:9.4}",
            entry.path_index,
            entry.criterion_percent,
            entry.degeneracy,
            entry.leg_count,
            entry.effective_half_path_length_angstrom
        ),
        (GenfmtPathLogMode::Jas, false) => format!(
            "   {:4}{:12.5}{:12.5}{:6}{:9.4} neglected",
            entry.path_index,
            entry.criterion_percent,
            entry.degeneracy,
            entry.leg_count,
            entry.effective_half_path_length_angstrom
        ),
    }
}

fn ensure_finite_log_value(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_log_dat(field, "value must be finite"))
    }
}

fn invalid_log_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidLogDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_log_summary() -> Result<()> {
        let data = parse_log_dat(STANDARD_LOG)?;
        assert_eq!(data.version, "FEFF 10.0.0");
        assert_eq!(data.core_hole_lifetime_ev, Some(1.729));
        assert_eq!(data.titles, vec![" Cu crystal"]);
        assert_eq!(
            data.calculation_summary,
            Some("Cu K edge XANES using RPA corehole.".to_string())
        );
        assert_eq!(data.features, vec!["Self-Consistent Field potentials"]);
        assert_eq!(data.cards[0], "ATOMS");
        assert_eq!(data.cards[data.cards.len() - 1], "COREHOLE");
        Ok(())
    }

    #[test]
    fn parses_empty_features_and_post_core_messages() -> Result<()> {
        let data = parse_log_dat(SPIN_LOG)?;
        assert_eq!(data.features, Vec::<String>::new());
        assert_eq!(data.post_core_lines.len(), 6);
        assert_eq!(data.core_hole_lifetime_ev, Some(5.533));
        Ok(())
    }

    #[test]
    fn parses_error_log_without_calculation_summary() -> Result<()> {
        let data = parse_log_dat(ERROR_LOG)?;
        assert!(!data.has_calculation_summary());
        assert_eq!(data.core_hole_lifetime_ev, None);
        assert_eq!(data.preamble_lines.len(), 4);
        assert_eq!(data.preamble_lines[3], "RDINP fatal error.");
        Ok(())
    }

    #[test]
    fn roundtrips_log_text() -> Result<()> {
        for text in [STANDARD_LOG, SPIN_LOG, ERROR_LOG] {
            let data = parse_log_dat(text)?;
            assert_eq!(parse_log_dat(&log_dat_string(&data)?)?, data);
        }
        Ok(())
    }

    #[test]
    fn parses_and_roundtrips_module_log_text() -> Result<()> {
        let data = parse_module_log_dat(MODULE_LOG)?;
        assert_eq!(data.line_count(), 3);
        assert!(!data.is_empty());
        assert_eq!(data.lines[0], "Calculating SCF potentials ...");
        assert_eq!(parse_module_log_dat(&module_log_dat_string(&data)?)?, data);
        assert_eq!(module_log_dat_string(&data)?, MODULE_LOG);
        assert!(parse_module_log_dat("")?.is_empty());
        Ok(())
    }

    #[test]
    fn preserves_module_log_terminators() -> Result<()> {
        let text = "one\ntwo\r\nthree\rfour";
        let data = parse_module_log_dat(text)?;
        assert_eq!(data.lines, vec!["one", "two", "three", "four"]);
        assert_eq!(data.line_terminators, vec!["\n", "\r\n", "\r", ""]);
        assert_eq!(module_log_dat_string(&data)?, text);
        Ok(())
    }

    #[test]
    fn genfmt_path_module_log_matches_genfmtsub_reference() -> Result<()> {
        let data = genfmt_path_module_log(GenfmtPathLogInput {
            mode: GenfmtPathLogMode::Ordinary,
            print_level: 3,
            curved_wave_criterion_percent: 12.5,
            entries: &sample_genfmt_path_log_entries(),
        })?;

        assert_eq!(data.line_count(), 5);
        assert_eq!(data.lines[0], "    Curved wave chi amplitude ratio  12.50%");
        assert_eq!(data.lines[1], "    path  cw ratio     deg    nleg  reff");
        assert!(data.lines[2].contains("0.1250E+02"));
        assert!(data.lines[3].ends_with(" neglected"));
        assert_eq!(data.lines[4], "    1 paths kept,    2 examined.");
        assert!(module_log_dat_string(&data)?.ends_with('\n'));
        Ok(())
    }

    #[test]
    fn genfmt_path_module_log_suppresses_ordinary_low_print_level() -> Result<()> {
        let data = genfmt_path_module_log(GenfmtPathLogInput {
            mode: GenfmtPathLogMode::Ordinary,
            print_level: 2,
            curved_wave_criterion_percent: 12.5,
            entries: &sample_genfmt_path_log_entries(),
        })?;

        assert!(data.is_empty());
        assert_eq!(module_log_dat_string(&data)?, "");
        Ok(())
    }

    #[test]
    fn genfmt_path_module_log_matches_genfmtjas_reference() -> Result<()> {
        let data = genfmt_path_module_log(GenfmtPathLogInput {
            mode: GenfmtPathLogMode::Jas,
            print_level: 0,
            curved_wave_criterion_percent: 12.5,
            entries: &sample_genfmt_path_log_entries(),
        })?;

        assert_eq!(data.line_count(), 6);
        assert_eq!(
            data.lines[1],
            "    Discard feff.dat for paths with cw ratio <   8.33%"
        );
        assert_eq!(
            data.lines[3],
            "     17    12.50000     4.00000     3   1.2700"
        );
        assert_eq!(
            data.lines[4],
            "     23     6.25000     2.00000     4   2.5000 neglected"
        );
        assert_eq!(data.lines[5], "    1 paths kept,    2 examined.");
        Ok(())
    }

    #[test]
    fn genfmt_path_log_entry_from_genfmt_summary() {
        let summary = sample_genfmt_path_output_summaries()[0];

        let entry = GenfmtPathLogEntry::from(summary);

        assert_eq!(GenfmtPathLogEntry::from(&summary), entry);
        assert_eq!(entry.path_index, 17);
        assert!(entry.retained);
        assert_eq!(entry.criterion_percent, 12.5);
        assert_eq!(entry.degeneracy, 4.0);
        assert_eq!(entry.leg_count, 3);
        assert_eq!(entry.effective_half_path_length_angstrom, 1.27);
    }

    #[test]
    fn genfmt_ordinary_path_outputs_module_log_matches_summary_entries() -> Result<()> {
        let outputs = sample_genfmt_ordinary_path_outputs();

        let data = genfmt_ordinary_path_outputs_module_log(3, 12.5, &outputs)?;
        let expected = genfmt_path_module_log(GenfmtPathLogInput {
            mode: GenfmtPathLogMode::Ordinary,
            print_level: 3,
            curved_wave_criterion_percent: 12.5,
            entries: &sample_genfmt_path_log_entries(),
        })?;

        assert_eq!(data, expected);
        assert_eq!(data.lines[4], "    1 paths kept,    2 examined.");
        Ok(())
    }

    #[test]
    fn genfmt_jas_path_outputs_module_log_matches_summary_entries() -> Result<()> {
        let outputs = sample_genfmt_jas_path_outputs();

        let data = genfmt_jas_path_outputs_module_log(0, 12.5, &outputs)?;
        let expected = genfmt_path_module_log(GenfmtPathLogInput {
            mode: GenfmtPathLogMode::Jas,
            print_level: 0,
            curved_wave_criterion_percent: 12.5,
            entries: &sample_genfmt_path_log_entries(),
        })?;

        assert_eq!(data, expected);
        assert_eq!(
            data.lines[1],
            "    Discard feff.dat for paths with cw ratio <   8.33%"
        );
        Ok(())
    }

    #[test]
    fn pot_module_log_matches_cached_pot_wrapper() -> Result<()> {
        let data = pot_module_log();

        assert_eq!(data.line_count(), 3);
        assert_eq!(data.lines[0], "Calculating SCF potentials ...");
        assert_eq!(data.lines[1], "FEFF-serial using 1 thread.");
        assert_eq!(data.lines[2], "Done with module: potentials.");
        assert_eq!(module_log_dat_string(&data)?, MODULE_LOG);
        Ok(())
    }

    #[test]
    fn cached_pot_stage_module_log_replaces_atomic_wrapper() {
        let atomic = ModuleLogData {
            lines: vec![
                "Calculating atomic potentials ...".to_string(),
                "Done with module: atomic potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 2],
        };

        assert!(is_atomic_potential_module_log(&atomic));
        assert_eq!(cached_pot_stage_module_log(Some(&atomic)), pot_module_log());
    }

    #[test]
    fn cached_pot_stage_module_log_preserves_existing_pot_detail() {
        let detailed = ModuleLogData {
            lines: vec![
                "Calculating SCF potentials ...".to_string(),
                "cached FEFF POT detail".to_string(),
                "Done with module: potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 3],
        };

        assert!(!is_atomic_potential_module_log(&detailed));
        assert_eq!(
            cached_pot_stage_module_log(Some(&detailed)),
            detailed.clone()
        );
        assert_eq!(cached_pot_stage_module_log(None), pot_module_log());
    }

    #[test]
    fn genfmt_path_module_log_rejects_invalid_inputs() {
        let mut entries = sample_genfmt_path_log_entries();
        entries[0].criterion_percent = f64::NAN;
        assert!(matches!(
            genfmt_path_module_log(GenfmtPathLogInput {
                mode: GenfmtPathLogMode::Jas,
                print_level: 0,
                curved_wave_criterion_percent: 12.5,
                entries: &entries,
            }),
            Err(IoError::InvalidLogDat { field: "crit", .. })
        ));

        let mut entries = sample_genfmt_path_log_entries();
        entries[0].path_index = 0;
        assert!(matches!(
            genfmt_path_module_log(GenfmtPathLogInput {
                mode: GenfmtPathLogMode::Jas,
                print_level: 0,
                curved_wave_criterion_percent: 12.5,
                entries: &entries,
            }),
            Err(IoError::InvalidLogDat {
                field: "path_index",
                ..
            })
        ));
    }

    #[test]
    fn rejects_bad_log_inputs() {
        assert!(parse_log_dat("").is_err());
        assert!(parse_log_dat("not a launch line\n").is_err());
        assert!(parse_log_dat("Launching FEFF version \n").is_err());
        assert!(
            parse_log_dat("Launching FEFF version FEFF 10\nYour calculation:\nUsing:\n").is_err()
        );

        let bad = LogDatData {
            version: "FEFF 10".to_string(),
            preamble_lines: Vec::new(),
            core_hole_lifetime_ev: Some(f64::NAN),
            post_core_lines: Vec::new(),
            titles: Vec::new(),
            calculation_summary: None,
            features: Vec::new(),
            cards: Vec::new(),
            trailing_lines: Vec::new(),
        };
        assert!(log_dat_string(&bad).is_err());
    }

    const STANDARD_LOG: &str = r#"Launching FEFF version FEFF 10.0.0
Resetting lmaxsc to 2 for iph =    0.  Use  UNFREEZE to prevent this.
Core hole lifetime is   1.729 eV.
Your calculation:
 Cu crystal
Cu K edge XANES using RPA corehole.
Using:     * Self-Consistent Field potentials
Using cards:   ATOMS CONTROL EXCHANGE TITLE POTENTIALS XANES SCF FMS COREHOLE
"#;

    const SPIN_LOG: &str = r#"Launching FEFF version FEFF 10.0.0
 RGRID, rgrd;   1.00000E-02
Core hole lifetime is   5.533 eV.
No spin set in POTENTIALS card. Using default spins:
iph   spinph
  0 7.0
No spin set in POTENTIALS card. Using default spins:
iph   spinph
  1 7.0
Your calculation:
 Gd_L1 hcp
Gd L1 edge XMCD using FSR corehole.
Using:
Using cards:   ATOMS CONTROL EXCHANGE TITLE RPATH DEBYE POTENTIALS CRITERIA XANES RGRID SPIN EDGE XMCD
"#;

    const ERROR_LOG: &str = r#"Launching FEFF version FEFF 10.0.0
Using finite nucleus.
 Error reading input, bad line follows:
 0    XXX   Te
RDINP fatal error.
"#;

    const MODULE_LOG: &str = r#"Calculating SCF potentials ...
FEFF-serial using 1 thread.
Done with module: potentials.
"#;

    fn sample_genfmt_path_log_entries() -> [GenfmtPathLogEntry; 2] {
        [
            GenfmtPathLogEntry {
                path_index: 17,
                retained: true,
                criterion_percent: 12.5,
                degeneracy: 4.0,
                leg_count: 3,
                effective_half_path_length_angstrom: 1.27,
            },
            GenfmtPathLogEntry {
                path_index: 23,
                retained: false,
                criterion_percent: 6.25,
                degeneracy: 2.0,
                leg_count: 4,
                effective_half_path_length_angstrom: 2.5,
            },
        ]
    }

    fn sample_genfmt_path_output_summaries() -> [GenfmtPathOutputSummary; 2] {
        [
            GenfmtPathOutputSummary {
                path_index: 17,
                retained: true,
                criterion_percent: 12.5,
                degeneracy: 4.0,
                leg_count: 3,
                effective_half_path_length_bohr: 2.4,
                effective_half_path_length_angstrom: 1.27,
            },
            GenfmtPathOutputSummary {
                path_index: 23,
                retained: false,
                criterion_percent: 6.25,
                degeneracy: 2.0,
                leg_count: 4,
                effective_half_path_length_bohr: 4.7,
                effective_half_path_length_angstrom: 2.5,
            },
        ]
    }

    fn sample_genfmt_ordinary_path_outputs() -> GenfmtOrdinaryPathOutputs {
        let path_summaries = sample_genfmt_path_output_summaries().to_vec();
        GenfmtOrdinaryPathOutputs {
            examined_path_count: path_summaries.len(),
            retained_path_count: path_summaries
                .iter()
                .filter(|summary| summary.retained)
                .count(),
            final_normalization: Some(1.0),
            path_summaries,
            retained_paths: Vec::new(),
        }
    }

    fn sample_genfmt_jas_path_outputs() -> GenfmtJasPathOutputs {
        let path_summaries = sample_genfmt_path_output_summaries().to_vec();
        GenfmtJasPathOutputs {
            examined_path_count: path_summaries.len(),
            retained_path_count: path_summaries
                .iter()
                .filter(|summary| summary.retained)
                .count(),
            final_normalization: Some(1.0),
            path_summaries,
            retained_paths: Vec::new(),
            decomposed_paths: None,
        }
    }
}
