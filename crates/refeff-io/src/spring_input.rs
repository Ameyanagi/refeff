//! Typed reader/writer for FEFF `spring.inp` Debye force-field files.
//!
//! The FEFF Debye equation-of-motion and recursion-method drivers read
//! `spring.inp` with the `VDOS`, `PRINT`/`PRDOS`, `STRETCHES`, and `ANGLES`
//! cards documented in the FEFF10 user guide and implemented by
//! `DEBYE/sigrem.f90`.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::{IoError, Result};

/// Default VDOS spectral resolution used by FEFF when `VDOS` is omitted.
pub const SPRING_DEFAULT_RESOLUTION: f64 = 0.05;
/// Default VDOS maximum-frequency multiplier used by FEFF.
pub const SPRING_DEFAULT_WMAX: f64 = 1.0;
/// Default low-frequency fitting fraction used by FEFF's reader.
pub const SPRING_DEFAULT_DOSFIT: f64 = 0.0;
/// Default VDOS time-integration cutoff used by FEFF.
pub const SPRING_DEFAULT_ACUT: f64 = 3.0;

/// Parsed FEFF `spring.inp` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct SpringInput {
    /// Optional VDOS integration settings.
    pub vdos: Option<SpringVdos>,
    /// Optional number of projected VDOS path files to print.
    pub print_projected: Option<usize>,
    /// Bond-stretching force constants.
    pub stretches: Vec<SpringStretch>,
    /// Angle-bending force constants.
    pub angles: Vec<SpringAngle>,
}

impl SpringInput {
    /// Parse FEFF `spring.inp` text.
    pub fn parse_str(text: &str) -> Result<Self> {
        parse_spring_inp(text)
    }

    /// Return explicit VDOS settings or FEFF's reader defaults.
    #[must_use]
    pub fn vdos_or_default(&self) -> SpringVdos {
        self.vdos.unwrap_or_default()
    }

    /// Atom-index pairs for stretch rows as an `n x 2` ndarray.
    #[must_use]
    pub fn stretch_indices(&self) -> Array2<usize> {
        Array2::from_shape_fn((self.stretches.len(), 2), |(row, column)| {
            let stretch = &self.stretches[row];
            if column == 0 {
                stretch.first_atom
            } else {
                stretch.second_atom
            }
        })
    }

    /// Stretch force constants as an ndarray.
    #[must_use]
    pub fn stretch_force_constants(&self) -> Array1<f64> {
        self.stretches
            .iter()
            .map(|stretch| stretch.force_constant)
            .collect()
    }

    /// Stretch distance tolerances as FEFF's normalized fractions.
    #[must_use]
    pub fn normalized_stretch_tolerances(&self) -> Array1<f64> {
        self.stretches
            .iter()
            .map(SpringStretch::normalized_distance_tolerance)
            .collect()
    }

    /// Atom-index triplets for angle rows as an `n x 3` ndarray.
    #[must_use]
    pub fn angle_indices(&self) -> Array2<usize> {
        Array2::from_shape_fn((self.angles.len(), 3), |(row, column)| {
            let angle = &self.angles[row];
            match column {
                0 => angle.first_atom,
                1 => angle.center_atom,
                _ => angle.third_atom,
            }
        })
    }

    /// Angle-bending force constants as an ndarray.
    #[must_use]
    pub fn angle_force_constants(&self) -> Array1<f64> {
        self.angles
            .iter()
            .map(|angle| angle.force_constant)
            .collect()
    }

    /// Angle tolerances as FEFF's normalized fractions.
    #[must_use]
    pub fn normalized_angle_tolerances(&self) -> Array1<f64> {
        self.angles
            .iter()
            .map(SpringAngle::normalized_angle_tolerance)
            .collect()
    }
}

/// VDOS integration settings from a `VDOS` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringVdos {
    /// VDOS spectral resolution width.
    pub resolution: f64,
    /// Maximum-frequency multiplier.
    pub wmax: f64,
    /// Fraction of the low-frequency VDOS fitted to Debye-like behavior.
    pub dosfit: f64,
    /// Time-integration cutoff.
    pub acut: f64,
}

impl Default for SpringVdos {
    fn default() -> Self {
        Self {
            resolution: SPRING_DEFAULT_RESOLUTION,
            wmax: SPRING_DEFAULT_WMAX,
            dosfit: SPRING_DEFAULT_DOSFIT,
            acut: SPRING_DEFAULT_ACUT,
        }
    }
}

/// One `STRETCHES` force-constant row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringStretch {
    /// First FEFF atom index.
    pub first_atom: usize,
    /// Second FEFF atom index.
    pub second_atom: usize,
    /// Central stretching force constant.
    pub force_constant: f64,
    /// Bond-length matching tolerance in percent.
    pub distance_tolerance_percent: f64,
}

impl SpringStretch {
    /// Return FEFF's normalized distance tolerance, `abs(dR_ij) / 100`.
    #[must_use]
    pub fn normalized_distance_tolerance(&self) -> f64 {
        self.distance_tolerance_percent.abs() / 100.0
    }
}

/// One `ANGLES` force-constant row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringAngle {
    /// First FEFF atom index in the angle triplet.
    pub first_atom: usize,
    /// Center FEFF atom index in the angle triplet.
    pub center_atom: usize,
    /// Third FEFF atom index in the angle triplet.
    pub third_atom: usize,
    /// Angle-bending force constant.
    pub force_constant: f64,
    /// Angle matching tolerance in percent.
    pub angle_tolerance_percent: f64,
}

impl SpringAngle {
    /// Return FEFF's normalized angle tolerance, `abs(dtheta) / 100`.
    #[must_use]
    pub fn normalized_angle_tolerance(&self) -> f64 {
        self.angle_tolerance_percent.abs() / 100.0
    }
}

/// Render FEFF `spring.inp` text.
pub fn spring_inp_string(input: &SpringInput) -> Result<String> {
    validate_spring_input(input)?;

    let mut out = String::new();
    if let Some(vdos) = input.vdos {
        writeln!(
            out,
            " VDOS {} {} {} {}",
            vdos.resolution, vdos.wmax, vdos.dosfit, vdos.acut
        )?;
    }
    if let Some(count) = input.print_projected {
        writeln!(out, " PRINT {count}")?;
    }
    if !input.stretches.is_empty() {
        writeln!(out, " STRETCHES")?;
        for stretch in &input.stretches {
            writeln!(
                out,
                " {} {} {} {}",
                stretch.first_atom,
                stretch.second_atom,
                stretch.force_constant,
                stretch.distance_tolerance_percent
            )?;
        }
    }
    if !input.angles.is_empty() {
        writeln!(out, " ANGLES")?;
        for angle in &input.angles {
            writeln!(
                out,
                " {} {} {} {} {}",
                angle.first_atom,
                angle.center_atom,
                angle.third_atom,
                angle.force_constant,
                angle.angle_tolerance_percent
            )?;
        }
    }
    Ok(out)
}

/// Parse FEFF `spring.inp` text.
pub fn parse_spring_inp(text: &str) -> Result<SpringInput> {
    let mut input = SpringInput {
        vdos: None,
        print_projected: None,
        stretches: Vec::new(),
        angles: Vec::new(),
    };
    let mut section = SpringSection::Top;

    for line in spring_lines(text) {
        let tokens = split_spring_line(line.text);
        if tokens.is_empty() {
            continue;
        }

        if let Some(card) = parse_spring_card(tokens[0]) {
            section = match card {
                SpringCard::Vdos => {
                    input.vdos = Some(parse_vdos(line.line, &tokens)?);
                    SpringSection::Top
                }
                SpringCard::Print => {
                    input.print_projected = Some(parse_print(line.line, &tokens)?);
                    SpringSection::Top
                }
                SpringCard::Stretches => SpringSection::Stretches,
                SpringCard::Angles => SpringSection::Angles,
                SpringCard::End => break,
            };
            continue;
        }

        match section {
            SpringSection::Top => {
                return Err(IoError::SpringInpParse {
                    field: "card",
                    line: line.line,
                    token: tokens[0].to_string(),
                });
            }
            SpringSection::Stretches => input.stretches.push(parse_stretch(line.line, &tokens)?),
            SpringSection::Angles => input.angles.push(parse_angle(line.line, &tokens)?),
        }
    }

    validate_spring_input(&input)?;
    Ok(input)
}

/// Write FEFF `spring.inp` text to a file.
pub fn write_spring_inp(path: impl AsRef<Path>, input: &SpringInput) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, spring_inp_string(input)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `spring.inp` text from a file.
pub fn read_spring_inp(path: impl AsRef<Path>) -> Result<SpringInput> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_spring_inp(&text)
}

fn parse_vdos(line: usize, tokens: &[&str]) -> Result<SpringVdos> {
    if tokens.len() < 4 {
        return Err(IoError::SpringInpRowWidth {
            line,
            actual: tokens.len(),
            expected: 4,
        });
    }
    Ok(SpringVdos {
        resolution: parse_f64(line, "VDOS resolution", tokens[1])?,
        wmax: parse_f64(line, "VDOS wmax", tokens[2])?,
        dosfit: parse_f64(line, "VDOS dosfit", tokens[3])?,
        acut: tokens.get(4).map_or(Ok(SPRING_DEFAULT_ACUT), |token| {
            parse_f64(line, "VDOS acut", token)
        })?,
    })
}

fn parse_print(line: usize, tokens: &[&str]) -> Result<usize> {
    tokens
        .get(1)
        .map_or(Ok(1), |token| parse_usize(line, "PRINT iprdos", token))
}

fn parse_stretch(line: usize, tokens: &[&str]) -> Result<SpringStretch> {
    if tokens.len() < 4 {
        return Err(IoError::SpringInpRowWidth {
            line,
            actual: tokens.len(),
            expected: 4,
        });
    }
    Ok(SpringStretch {
        first_atom: parse_usize(line, "stretch first atom", tokens[0])?,
        second_atom: parse_usize(line, "stretch second atom", tokens[1])?,
        force_constant: parse_f64(line, "stretch force constant", tokens[2])?,
        distance_tolerance_percent: parse_f64(line, "stretch tolerance", tokens[3])?,
    })
}

fn parse_angle(line: usize, tokens: &[&str]) -> Result<SpringAngle> {
    if tokens.len() < 5 {
        return Err(IoError::SpringInpRowWidth {
            line,
            actual: tokens.len(),
            expected: 5,
        });
    }
    Ok(SpringAngle {
        first_atom: parse_usize(line, "angle first atom", tokens[0])?,
        center_atom: parse_usize(line, "angle center atom", tokens[1])?,
        third_atom: parse_usize(line, "angle third atom", tokens[2])?,
        force_constant: parse_f64(line, "angle force constant", tokens[3])?,
        angle_tolerance_percent: parse_f64(line, "angle tolerance", tokens[4])?,
    })
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::SpringInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::SpringInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn validate_spring_input(input: &SpringInput) -> Result<()> {
    if let Some(vdos) = input.vdos {
        validate_finite("VDOS resolution", vdos.resolution)?;
        validate_finite("VDOS wmax", vdos.wmax)?;
        validate_finite("VDOS dosfit", vdos.dosfit)?;
        validate_finite("VDOS acut", vdos.acut)?;
    }

    for stretch in &input.stretches {
        validate_finite("stretch force constant", stretch.force_constant)?;
        validate_finite("stretch tolerance", stretch.distance_tolerance_percent)?;
    }
    for angle in &input.angles {
        validate_finite("angle force constant", angle.force_constant)?;
        validate_finite("angle tolerance", angle.angle_tolerance_percent)?;
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidSpringInp {
            field,
            message: "value must be finite".to_string(),
        })
    }
}

fn spring_lines(text: &str) -> impl Iterator<Item = SpringLine<'_>> {
    text.lines().enumerate().filter_map(|(index, raw)| {
        let line = strip_inline_comment(raw).trim();
        if line.is_empty() || is_comment_line(line) {
            None
        } else {
            Some(SpringLine {
                line: index + 1,
                text: line,
            })
        }
    })
}

fn split_spring_line(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

fn strip_inline_comment(line: &str) -> &str {
    let comment_index = line
        .char_indices()
        .find_map(|(index, ch)| matches!(ch, '#' | '!' | '%' | '*').then_some(index));
    comment_index.map_or(line, |index| &line[..index])
}

fn is_comment_line(line: &str) -> bool {
    line.chars()
        .next()
        .is_some_and(|ch| matches!(ch, '#' | '!' | '*' | 'C' | 'c'))
}

fn parse_spring_card(token: &str) -> Option<SpringCard> {
    let token = token.to_ascii_uppercase();
    if token.starts_with("STRE") {
        Some(SpringCard::Stretches)
    } else if token.starts_with("ANGL") {
        Some(SpringCard::Angles)
    } else if token.starts_with("VDOS") {
        Some(SpringCard::Vdos)
    } else if token.starts_with("PRDO") || token.starts_with("PRIN") {
        Some(SpringCard::Print)
    } else if token.starts_with("END") {
        Some(SpringCard::End)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct SpringLine<'a> {
    line: usize,
    text: &'a str,
}

#[derive(Debug, Clone, Copy)]
enum SpringSection {
    Top,
    Stretches,
    Angles,
}

#[derive(Debug, Clone, Copy)]
enum SpringCard {
    Vdos,
    Print,
    Stretches,
    Angles,
    End,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_spring_file() -> Result<()> {
        let parsed = parse_spring_inp(DOCUMENTED_SPRING_INP)?;
        assert_eq!(
            parsed.vdos,
            Some(SpringVdos {
                resolution: 0.02,
                wmax: 1.0,
                dosfit: 1.2,
                acut: 3.0,
            })
        );
        assert_eq!(parsed.print_projected, Some(5));
        assert_eq!(
            parsed.stretches,
            vec![
                SpringStretch {
                    first_atom: 0,
                    second_atom: 2,
                    force_constant: 110.0,
                    distance_tolerance_percent: 2.0,
                },
                SpringStretch {
                    first_atom: 1,
                    second_atom: 2,
                    force_constant: 626.0,
                    distance_tolerance_percent: 5.0,
                },
            ]
        );
        assert_eq!(
            parsed.angles,
            vec![
                SpringAngle {
                    first_atom: 2,
                    center_atom: 0,
                    third_atom: 5,
                    force_constant: 37.0,
                    angle_tolerance_percent: 10.0,
                },
                SpringAngle {
                    first_atom: 1,
                    center_atom: 2,
                    third_atom: 3,
                    force_constant: 2590.0,
                    angle_tolerance_percent: 10.0,
                },
            ]
        );
        assert_eq!(parsed.stretch_indices().shape(), &[2, 2]);
        assert_eq!(parsed.angle_indices().shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn uses_defaults_for_optional_fields() -> Result<()> {
        let parsed = parse_spring_inp(
            r#"
PRINT
VDOS 0.03 0.5 1
STRETCHES
0 1 27.9 -2.0
"#,
        )?;
        assert_eq!(parsed.print_projected, Some(1));
        assert_eq!(
            parsed.vdos,
            Some(SpringVdos {
                resolution: 0.03,
                wmax: 0.5,
                dosfit: 1.0,
                acut: SPRING_DEFAULT_ACUT,
            })
        );
        assert_eq!(parsed.stretches[0].normalized_distance_tolerance(), 0.02);
        Ok(())
    }

    #[test]
    fn roundtrips_spring_text() -> Result<()> {
        let parsed = parse_spring_inp(DOCUMENTED_SPRING_INP)?;
        let rendered = spring_inp_string(&parsed)?;
        assert_eq!(parse_spring_inp(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_spring_rows() {
        assert!(parse_spring_inp("VDOS 0.03 1\n").is_err());
        assert!(parse_spring_inp("STRETCHES\n0 1\n").is_err());
        assert!(parse_spring_inp("ANGLES\n0 1 2 3\n").is_err());
        assert!(parse_spring_inp("UNKNOWN\n").is_err());
        assert!(parse_spring_inp("VDOS NaN 1 0\n").is_err());
    }

    const DOCUMENTED_SPRING_INP: &str = r#"
* 13-atom model of zinc tetraimidazole
*                 res          wmax         dosfit         acut
VDOS             0.02             1            1.2            3
PRINT 5
STRETCHES  *   i         j        k_ij       dR_ij[%]
               0         2        110.        2.
               1         2        626.        5.
ANGLES  * i     j    k     ktheta      dtheta[%]
          2     0    5         37.           10.
          1     2    3       2590.           10.
"#;
}
