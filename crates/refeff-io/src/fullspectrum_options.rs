//! FEFF `FULLSPECTRUM/rdop.f90` option-card reader.
//!
//! FEFF10's full-spectrum driver reads a card-oriented `fullspectrum.inp`
//! stream after the module on/off handoff flag. This module covers that
//! standalone option grammar: `CONTROL`, `EGRID`, `DRUDE`, `VALENCE`, `EELS`,
//! global `DETAIL`, and per-`COMPONENT` edge selection blocks.

use std::path::{Path, PathBuf};

use refeff_core::{FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV, standard_edge_label};

use crate::error::{IoError, Result};
use crate::input::{bwords, strip_inline_comment};

const CONTROL_COUNT: usize = 6;
const COMPONENT_NAME_WIDTH: usize = 3;

/// Parsed FEFF `FULLSPECTRUM/rdop.f90` options.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumOptions {
    /// FEFF `CONTROL` switches. Missing switches keep FEFF's default `1`.
    pub control: [i32; CONTROL_COUNT],
    /// Optional `EGRID` request, converted from eV to Hartree.
    pub energy_grid: FullSpectrumOptionsEnergyGrid,
    /// Component rows and their edge-selection policy.
    pub components: Vec<FullSpectrumComponent>,
    /// Optional `DRUDE` free-electron contribution settings.
    pub drude: Option<FullSpectrumDrudeOptions>,
    /// Whether `VALENCE` requested inclusion of `xmu.val`.
    pub valence: bool,
    /// Whether `EELS` was requested.
    pub eels: bool,
}

/// FEFF `EGRID emin emax [npts]` request after unit conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullSpectrumOptionsEnergyGrid {
    /// Lower photon-energy bound in Hartree, or `None` for FEFF-derived bounds.
    pub min_hartree: Option<f64>,
    /// Upper photon-energy bound in Hartree, or `None` for FEFF-derived bounds.
    pub max_hartree: Option<f64>,
    /// Optional point count. FEFF's edge-adaptive grid later overwrites this.
    pub point_count: Option<usize>,
}

/// FEFF `DRUDE tau [ndrude]` options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullSpectrumDrudeOptions {
    /// Drude lifetime `tau`, in seconds.
    pub lifetime_seconds: f64,
    /// Optional free-electron density parameter `ndrude`.
    pub electron_density: Option<f64>,
}

/// One FEFF `COMPONENT` row and the edge policy following it.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumComponent {
    /// FEFF component name, truncated to the `character*3` width used by rdop.
    pub name: String,
    /// Component atomic number.
    pub atomic_number: i32,
    /// Optional number density converted from inverse cubic Angstrom to FEFF
    /// inverse cubic Bohr units.
    pub number_density_bohr3: Option<f64>,
    /// Explicit edge rows or FEFF automatic edge selection.
    pub edge_source: FullSpectrumComponentEdgeSource,
}

/// Component edge source from `rdop.f90`.
#[derive(Debug, Clone, PartialEq)]
pub enum FullSpectrumComponentEdgeSource {
    /// `COMPONENT ... EDGES` followed by explicit edge rows.
    Explicit(Vec<FullSpectrumComponentEdge>),
    /// No `EDGES` block: FEFF selects occupied edges from atomic occupations.
    Automatic {
        /// Fine-structure policy for automatically selected edges.
        fine_structure: FullSpectrumAutomaticFineStructure,
    },
}

/// Fine-structure policy for automatically selected edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullSpectrumAutomaticFineStructure {
    /// FEFF default: background-only unless a global `DETAIL` card appeared.
    None,
    /// Global `DETAIL` requests fine structure for all selected edges.
    All,
    /// `COMPONENT ... DETAIL` followed by edge labels to enable.
    Listed(Vec<String>),
}

/// One explicit edge row following `COMPONENT ... EDGES`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullSpectrumComponentEdge {
    /// Canonical FEFF edge label such as `K` or `L3`.
    pub label: String,
    /// Whether `CONV`/`CONVOLUTION` requested LDOS convolution.
    pub convolve: bool,
    /// Whether FEFF should compute fine structure for this edge.
    pub fine_structure: bool,
}

impl FullSpectrumOptions {
    /// Parse FEFF full-spectrum option cards from a string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        parse_fullspectrum_options_with_source(source.into(), text)
    }

    /// Whether any component edge requests LDOS convolution.
    #[must_use]
    pub fn requires_ldos(&self) -> bool {
        self.components
            .iter()
            .any(FullSpectrumComponent::requires_ldos)
    }
}

impl FullSpectrumOptionsEnergyGrid {
    /// Return true when `EGRID` supplied both FEFF energy bounds.
    #[must_use]
    pub fn is_explicit(&self) -> bool {
        self.min_hartree.is_some() && self.max_hartree.is_some()
    }
}

impl FullSpectrumComponent {
    /// Whether this component has an explicit convolved edge.
    #[must_use]
    pub fn requires_ldos(&self) -> bool {
        match &self.edge_source {
            FullSpectrumComponentEdgeSource::Explicit(edges) => {
                edges.iter().any(|edge| edge.convolve)
            }
            FullSpectrumComponentEdgeSource::Automatic { .. } => false,
        }
    }
}

/// Parse FEFF full-spectrum option cards from a string.
pub fn parse_fullspectrum_options(text: &str) -> Result<FullSpectrumOptions> {
    FullSpectrumOptions::parse_str("fullspectrum.inp", text)
}

/// Read FEFF full-spectrum option cards from a file.
pub fn read_fullspectrum_options(path: impl AsRef<Path>) -> Result<FullSpectrumOptions> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    FullSpectrumOptions::parse_str(path, &text)
}

#[derive(Debug, Clone)]
struct OptionLine {
    line: usize,
    tokens: Vec<String>,
}

fn parse_fullspectrum_options_with_source(
    source: PathBuf,
    text: &str,
) -> Result<FullSpectrumOptions> {
    let lines = option_lines(text);
    let mut options = FullSpectrumOptions {
        control: [1; CONTROL_COUNT],
        energy_grid: FullSpectrumOptionsEnergyGrid {
            min_hartree: None,
            max_hartree: None,
            point_count: None,
        },
        components: Vec::new(),
        drude: None,
        valence: false,
        eels: false,
    };
    let mut global_detail = false;
    let mut index = 0;

    while index < lines.len() {
        let line = &lines[index];
        let key = key3(&line.tokens[0]);
        match key.as_str() {
            "con" => {
                parse_control(&source, line, &mut options.control)?;
                index += 1;
            }
            "eel" => {
                options.eels = true;
                index += 1;
            }
            "val" => {
                options.valence = true;
                index += 1;
            }
            "dru" => {
                options.drude = Some(parse_drude(&source, line)?);
                index += 1;
            }
            "det" => {
                global_detail = true;
                index += 1;
            }
            "egr" => {
                options.energy_grid = parse_egrid(&source, line)?;
                index += 1;
            }
            "com" => {
                let (component, next_index) =
                    parse_component(&source, &lines, index, global_detail)?;
                options.components.push(component);
                index = next_index;
            }
            _ => {
                index += 1;
            }
        }
    }

    if options.components.is_empty() {
        return Err(parse_error(
            &source,
            0,
            "FULLSPECTRUM options require at least one COMPONENT card",
        ));
    }
    Ok(options)
}

fn parse_control(
    source: &Path,
    line: &OptionLine,
    control: &mut [i32; CONTROL_COUNT],
) -> Result<()> {
    for (index, token) in line.tokens.iter().skip(1).take(CONTROL_COUNT).enumerate() {
        control[index] = parse_i32(source, line.line, "CONTROL switch", token)?;
    }
    Ok(())
}

fn parse_drude(source: &Path, line: &OptionLine) -> Result<FullSpectrumDrudeOptions> {
    let Some(tau) = line.tokens.get(1) else {
        return Err(parse_error(
            source,
            line.line,
            "DRUDE requires a lifetime value",
        ));
    };
    let lifetime_seconds = parse_f64(source, line.line, "DRUDE tau", tau)?;
    validate_finite_positive(source, line.line, "DRUDE tau", lifetime_seconds)?;
    let electron_density = line
        .tokens
        .get(2)
        .map(|token| parse_f64(source, line.line, "DRUDE ndrude", token))
        .transpose()?;
    if let Some(value) = electron_density {
        validate_finite_positive(source, line.line, "DRUDE ndrude", value)?;
    }
    Ok(FullSpectrumDrudeOptions {
        lifetime_seconds,
        electron_density,
    })
}

fn parse_egrid(source: &Path, line: &OptionLine) -> Result<FullSpectrumOptionsEnergyGrid> {
    let Some(min) = line.tokens.get(1) else {
        return Err(parse_error(
            source,
            line.line,
            "EGRID requires emin and emax",
        ));
    };
    let Some(max) = line.tokens.get(2) else {
        return Err(parse_error(
            source,
            line.line,
            "EGRID requires emin and emax",
        ));
    };
    let min_ev = parse_f64(source, line.line, "EGRID emin", min)?;
    let max_ev = parse_f64(source, line.line, "EGRID emax", max)?;
    if !min_ev.is_finite() || min_ev < 0.0 {
        return Err(parse_error(
            source,
            line.line,
            "EGRID emin must be finite and nonnegative",
        ));
    }
    validate_finite_positive(source, line.line, "EGRID emax", max_ev)?;
    if max_ev <= min_ev {
        return Err(parse_error(
            source,
            line.line,
            "EGRID emax must be greater than emin",
        ));
    }

    let point_count = line
        .tokens
        .get(3)
        .map(|token| parse_positive_usize(source, line.line, "EGRID point count", token))
        .transpose()?;
    Ok(FullSpectrumOptionsEnergyGrid {
        min_hartree: Some(min_ev / FEFF_HARTREE_EV),
        max_hartree: Some(max_ev / FEFF_HARTREE_EV),
        point_count,
    })
}

fn parse_component(
    source: &Path,
    lines: &[OptionLine],
    component_index: usize,
    global_detail: bool,
) -> Result<(FullSpectrumComponent, usize)> {
    let line = &lines[component_index];
    let Some(name) = line.tokens.get(1) else {
        return Err(parse_error(
            source,
            line.line,
            "COMPONENT requires a name and atomic number",
        ));
    };
    let Some(atomic_number) = line.tokens.get(2) else {
        return Err(parse_error(
            source,
            line.line,
            "COMPONENT requires a name and atomic number",
        ));
    };
    let atomic_number = parse_i32(source, line.line, "COMPONENT atomic number", atomic_number)?;
    if atomic_number <= 0 {
        return Err(parse_error(
            source,
            line.line,
            "COMPONENT atomic number must be positive",
        ));
    }

    let mut selector_index = 3;
    let number_density_bohr3 = match line.tokens.get(3) {
        Some(token) => match token.parse::<f64>() {
            Ok(value) if value.is_finite() && value > 0.0 => {
                selector_index = 4;
                Some(value * FEFF_BOHR_ANGSTROM.powi(3))
            }
            Ok(_) => {
                selector_index = 4;
                None
            }
            Err(_) => None,
        },
        None => None,
    };

    let selector = line
        .tokens
        .get(selector_index)
        .map(|token| key3(token))
        .unwrap_or_default();
    let mut next_index = component_index + 1;
    let edge_source = if selector == "edg" {
        let mut edges = Vec::new();
        while next_index < lines.len() {
            let edge_line = &lines[next_index];
            let Some(label) = standard_edge_label(&edge_line.tokens[0]) else {
                break;
            };
            let edge_mode = edge_line.tokens.get(1).map(|token| key3(token));
            let convolve = edge_mode.as_deref() == Some("con");
            let fine_structure = !matches!(edge_mode.as_deref(), Some("bac"));
            edges.push(FullSpectrumComponentEdge {
                label: label.to_string(),
                convolve,
                fine_structure,
            });
            next_index += 1;
        }
        FullSpectrumComponentEdgeSource::Explicit(edges)
    } else if selector == "det" {
        let mut detail_edges = Vec::new();
        while next_index < lines.len() {
            let edge_line = &lines[next_index];
            let Some(label) = standard_edge_label(&edge_line.tokens[0]) else {
                break;
            };
            detail_edges.push(label.to_string());
            next_index += 1;
        }
        FullSpectrumComponentEdgeSource::Automatic {
            fine_structure: FullSpectrumAutomaticFineStructure::Listed(detail_edges),
        }
    } else {
        FullSpectrumComponentEdgeSource::Automatic {
            fine_structure: if global_detail {
                FullSpectrumAutomaticFineStructure::All
            } else {
                FullSpectrumAutomaticFineStructure::None
            },
        }
    };

    Ok((
        FullSpectrumComponent {
            name: fortran_fixed_string(name, COMPONENT_NAME_WIDTH),
            atomic_number,
            number_density_bohr3,
            edge_source,
        },
        next_index,
    ))
}

fn option_lines(text: &str) -> Vec<OptionLine> {
    text.lines()
        .enumerate()
        .filter_map(|(index, raw)| {
            let trimmed = raw.trim_start();
            if trimmed.is_empty() || is_comment_line(trimmed) {
                return None;
            }
            let line = strip_inline_comment(trimmed);
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let tokens = bwords(line);
            if tokens.is_empty() {
                return None;
            }
            Some(OptionLine {
                line: index + 1,
                tokens,
            })
        })
        .collect()
}

fn is_comment_line(line: &str) -> bool {
    line.chars()
        .next()
        .is_some_and(|ch| matches!(ch, ';' | '*' | '%' | '#' | '!'))
}

fn key3(token: &str) -> String {
    token
        .chars()
        .take(3)
        .collect::<String>()
        .to_ascii_lowercase()
}

fn parse_i32(source: &Path, line: usize, field: &'static str, token: &str) -> Result<i32> {
    token.parse::<i32>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("invalid {field} value {token:?}"),
    })
}

fn parse_f64(source: &Path, line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("invalid {field} value {token:?}"),
    })
}

fn parse_positive_usize(
    source: &Path,
    line: usize,
    field: &'static str,
    token: &str,
) -> Result<usize> {
    let value = token.parse::<usize>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("invalid {field} value {token:?}"),
    })?;
    if value == 0 {
        return Err(parse_error(
            source,
            line,
            format!("{field} must be positive"),
        ));
    }
    Ok(value)
}

fn validate_finite_positive(
    source: &Path,
    line: usize,
    field: &'static str,
    value: f64,
) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(parse_error(
            source,
            line,
            format!("{field} must be finite and positive"),
        ))
    }
}

fn parse_error(source: &Path, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: message.into(),
    }
}

fn fortran_fixed_string(value: &str, width: usize) -> String {
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > width {
            break;
        }
        end = next;
    }
    value[..end].to_string()
}

#[cfg(test)]
mod tests {
    use refeff_core::{FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV};

    use super::{
        FullSpectrumAutomaticFineStructure, FullSpectrumComponentEdgeSource, FullSpectrumOptions,
        parse_fullspectrum_options,
    };
    use crate::{IoError, Result};

    #[test]
    fn parses_feff_fullspectrum_rdop_options() -> Result<()> {
        let options = FullSpectrumOptions::parse_str("fullspectrum.inp", FULLSPECTRUM_OPTIONS)?;

        assert_eq!(options.control, [1, 0, 1, 0, 1, 0]);
        assert!(options.energy_grid.is_explicit());
        assert_close(options.energy_grid.min_hartree, Some(5.0 / FEFF_HARTREE_EV))?;
        assert_close(
            options.energy_grid.max_hartree,
            Some(120.0 / FEFF_HARTREE_EV),
        )?;
        assert_eq!(options.energy_grid.point_count, Some(230));
        assert_eq!(
            options.drude.map(|drude| drude.lifetime_seconds),
            Some(1.5e-15)
        );
        assert_eq!(
            options.drude.and_then(|drude| drude.electron_density),
            Some(0.025)
        );
        assert!(options.valence);
        assert!(options.eels);
        assert!(options.requires_ldos());
        assert_eq!(options.components.len(), 2);

        let copper = &options.components[0];
        assert_eq!(copper.name, "Cu2");
        assert_eq!(copper.atomic_number, 29);
        assert_close(
            copper.number_density_bohr3,
            Some(0.0847 * FEFF_BOHR_ANGSTROM.powi(3)),
        )?;
        let edges = match &copper.edge_source {
            FullSpectrumComponentEdgeSource::Explicit(edges) => edges,
            FullSpectrumComponentEdgeSource::Automatic { .. } => {
                return Err(test_error("expected explicit copper edges"));
            }
        };
        assert_eq!(edges.len(), 3);
        assert_eq!(edges[0].label, "K");
        assert!(edges[0].convolve);
        assert!(edges[0].fine_structure);
        assert_eq!(edges[1].label, "L3");
        assert!(!edges[1].convolve);
        assert!(edges[1].fine_structure);
        assert_eq!(edges[2].label, "M1");
        assert!(!edges[2].convolve);
        assert!(!edges[2].fine_structure);

        let oxygen = &options.components[1];
        assert_eq!(oxygen.name, "O1");
        assert_eq!(oxygen.atomic_number, 8);
        assert_eq!(oxygen.number_density_bohr3, None);
        assert!(!oxygen.requires_ldos());
        let fine_structure = match &oxygen.edge_source {
            FullSpectrumComponentEdgeSource::Automatic { fine_structure } => fine_structure,
            FullSpectrumComponentEdgeSource::Explicit(_) => {
                return Err(test_error("expected automatic oxygen edges"));
            }
        };
        assert_eq!(
            fine_structure,
            &FullSpectrumAutomaticFineStructure::Listed(vec!["K".to_string(), "L1".to_string()])
        );
        Ok(())
    }

    #[test]
    fn applies_global_detail_to_automatic_edges() -> Result<()> {
        let options = parse_fullspectrum_options(
            r#"
DETAIL
COMPONENT FeLongName 26
"#,
        )?;
        let component = &options.components[0];
        assert_eq!(component.name, "FeL");
        let fine_structure = match &component.edge_source {
            FullSpectrumComponentEdgeSource::Automatic { fine_structure } => fine_structure,
            FullSpectrumComponentEdgeSource::Explicit(_) => {
                return Err(test_error("expected automatic edges"));
            }
        };
        assert_eq!(fine_structure, &FullSpectrumAutomaticFineStructure::All);
        Ok(())
    }

    #[test]
    fn rejects_invalid_rdop_options() {
        assert!(parse_fullspectrum_options("CONTROL 1 no\nCOMPONENT Cu 29\n").is_err());
        assert!(parse_fullspectrum_options("EGRID -1 10\nCOMPONENT Cu 29\n").is_err());
        assert!(parse_fullspectrum_options("DRUDE\nCOMPONENT Cu 29\n").is_err());
        assert!(parse_fullspectrum_options("COMPONENT Cu 0\n").is_err());
        assert!(parse_fullspectrum_options("VALENCE\n").is_err());
    }

    fn assert_close(actual: Option<f64>, expected: Option<f64>) -> Result<()> {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() <= 1.0e-12,
                "actual {actual} expected {expected}"
            ),
            (None, None) => {}
            _ => {
                return Err(test_error(format!(
                    "actual {actual:?} expected {expected:?}"
                )));
            }
        }
        Ok(())
    }

    fn test_error(message: impl Into<String>) -> IoError {
        IoError::Parse {
            path: "test".into(),
            line: 0,
            message: message.into(),
        }
    }

    const FULLSPECTRUM_OPTIONS: &str = r#"
CONTROL 1 0 1 0 1 0
EGRID 5.0 120.0 230
DRUDE 1.5E-15 0.025
VALENCE
EELS
DETAIL
COMPONENT Cu2 29 0.0847 EDGES
K CONV
4 DETAIL
M1 BACKGROUND
COMPONENT O1 8 DETAIL
1
L1
"#;
}
