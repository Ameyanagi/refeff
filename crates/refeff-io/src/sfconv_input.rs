//! Typed reader for FEFF `sfconv.inp` module handoff files.
//!
//! `sfconv.inp` controls spectral-function convolution after spectrum
//! assembly.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use refeff_core::SfconvSo2convMaterialInput;

use crate::eels_input::EelsInput;
use crate::list_dat::ListDatData;
use crate::{IoError, Result};

/// FEFF marker written at the top of files already processed by `SO2CONV`.
pub const SFCONV_SO2CONV_CONVOLUTED_MARKER: &str = "# Convoluted with A(omega).";

/// Parsed contents of a FEFF `sfconv.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvInput {
    /// Spectral-function convolution switches.
    pub control: SfconvControl,
    /// Width and center controls.
    pub window: SfconvWindow,
    /// Spectrum type and print flag.
    pub spectrum: SfconvSpectrum,
    /// Convolution filename, or `NULL`.
    pub cfname: String,
}

/// First control line of `sfconv.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfconvControl {
    pub msfconv: i32,
    pub ipse: i32,
    pub ipsk: i32,
}

/// Width and center controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvWindow {
    pub wsigk: f64,
    pub cen: f64,
}

/// Spectrum type and print controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfconvSpectrum {
    pub ispec: i32,
    pub ipr6: i32,
}

/// Kind of FEFF spectrum file selected by `SFCONV/so2conv.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfconvSo2convTargetKind {
    /// `chi.dat` or `chipNNNN.dat` EXAFS-like spectrum columns.
    Chi,
    /// `xmu.dat` XANES spectrum columns.
    Xmu,
    /// `feffNNNN.dat` path file.
    FeffPath,
}

/// One concrete file that FEFF `SO2CONV` would attempt to convolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SfconvSo2convTarget {
    /// FEFF file name as selected by the legacy `so2conv.f90` dispatch logic.
    pub file_name: String,
    /// Spectrum layout expected for the file.
    pub kind: SfconvSo2convTargetKind,
}

/// Header scan result for a FEFF spectrum file consumed by `SO2CONV`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSo2convHeader {
    /// Material constants read from fixed-width FEFF header fields.
    pub material: SfconvSo2convMaterialInput,
    /// Whether FEFF's previous-convolution marker was found before the table.
    pub already_convoluted: bool,
}

impl SfconvInput {
    /// Parse a FEFF `sfconv.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = SfconvInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `sfconv.inp` text.
pub fn sfconv_input_string(input: &SfconvInput) -> Result<String> {
    validate_sfconv_input(input)?;

    let mut out = String::new();
    out.push_str("msfconv, ipse, ipsk\n");
    push_i4_row(
        &mut out,
        [
            input.control.msfconv,
            input.control.ipse,
            input.control.ipsk,
        ],
    );
    out.push_str("wsigk, cen\n");
    out.push_str(&format!(
        "{:13.5}{:13.5}\n",
        input.window.wsigk, input.window.cen
    ));
    out.push_str("ispec, ipr6\n");
    push_i4_row(&mut out, [input.spectrum.ispec, input.spectrum.ipr6]);
    out.push_str("cfname\n");
    out.push_str(&sfconv_cfname_line(&input.cfname));
    out.push('\n');
    Ok(out)
}

/// Select the FEFF spectrum files consumed by `SO2CONV`.
///
/// This mirrors the filename branch in `SFCONV/so2conv.f90`: base `xmu.dat`
/// and `chi.dat` are always selected for EXAFS-style runs, `ipr6 >= 2` adds
/// path-expanded files from `list.dat`, and ELNES `eels.inp` polarization
/// ranges choose the legacy `xmuNN.dat`/`chiNN.dat` variants.
pub fn sfconv_so2conv_targets(
    input: &SfconvInput,
    list: Option<&ListDatData>,
    eels: Option<&EelsInput>,
) -> Result<Vec<SfconvSo2convTarget>> {
    let polarizations = so2conv_polarization_indices(eels)?;
    let mut targets = Vec::new();

    for polarization in polarizations {
        match input.spectrum.ispec.abs() {
            0 => push_so2conv_exafs_targets(&mut targets, input.spectrum.ipr6, list, polarization)?,
            1 => push_so2conv_xanes_targets(&mut targets, polarization)?,
            _ => {
                return Err(IoError::Parse {
                    path: "sfconv.inp".into(),
                    line: 0,
                    message: format!(
                        "SO2CONV ispec must be -1, 0, or 1, got {}",
                        input.spectrum.ispec
                    ),
                });
            }
        }
    }

    Ok(targets)
}

/// Extract `SO2CONV` material header values from a FEFF spectrum header.
///
/// FEFF `SFCONV/so2conv.f90` scans each header line with fixed-width
/// substrings until the dashed table separator is reached. This parser keeps
/// those legacy widths so values from `xmu.dat`, `chi.dat`, `chipNNNN.dat`,
/// and `feffNNNN.dat` headers feed the same unit-conversion path as FEFF.
pub fn sfconv_so2conv_material_input_from_header(
    source: impl Into<PathBuf>,
    text: &str,
) -> Result<SfconvSo2convMaterialInput> {
    sfconv_so2conv_header_from_text(source, text).map(|header| header.material)
}

/// Scan a FEFF spectrum header as `SO2CONV` does before reading data rows.
///
/// The scan stops at the dashed table separator and therefore ignores any
/// later marker or metadata rows, matching the legacy `so2conv.f90` flow.
pub fn sfconv_so2conv_header_from_text(
    source: impl Into<PathBuf>,
    text: &str,
) -> Result<SfconvSo2convHeader> {
    let source = source.into();
    let mut tokens = SfconvSo2convHeaderTokens::default();
    let mut already_convoluted = false;

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        if line.trim_end_matches(' ') == SFCONV_SO2CONV_CONVOLUTED_MARKER {
            already_convoluted = true;
        }
        scan_so2conv_material_header_line(line, line_number, &mut tokens);
        if fixed_width_field(line, 5, 9) == "---------" {
            break;
        }
    }

    Ok(SfconvSo2convHeader {
        material: SfconvSo2convMaterialInput {
            core_hole_width_ev: parse_header_token(&source, tokens.core_hole_width, "Gam_ch")?,
            wigner_seitz_radius: parse_header_token(&source, tokens.wigner_seitz_radius, "Rs_int")?,
            interstitial_potential_ev: parse_header_token(
                &source,
                tokens.interstitial_potential,
                "Vint",
            )?,
            chemical_potential_ev: parse_header_token(&source, tokens.chemical_potential, "Mu")?,
            fermi_wave_number_inv_angstrom: parse_header_token(
                &source,
                tokens.fermi_wave_number,
                "kf",
            )?,
        },
        already_convoluted,
    })
}

fn validate_sfconv_input(input: &SfconvInput) -> Result<()> {
    validate_finite("wsigk", input.window.wsigk)?;
    validate_finite("cen", input.window.cen)?;
    if input
        .cfname
        .chars()
        .any(|character| matches!(character, '\n' | '\r'))
    {
        return Err(IoError::Parse {
            path: "sfconv.inp".into(),
            line: 0,
            message: "SFCONV filename must not contain a line terminator".to_string(),
        });
    }
    Ok(())
}

fn push_so2conv_exafs_targets(
    targets: &mut Vec<SfconvSo2convTarget>,
    ipr6: i32,
    list: Option<&ListDatData>,
    polarization: i32,
) -> Result<()> {
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_xmu_file_name(polarization)?,
        kind: SfconvSo2convTargetKind::Xmu,
    });
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_chi_file_name(polarization, false)?,
        kind: SfconvSo2convTargetKind::Chi,
    });

    if ipr6 >= 2 {
        let list = list.ok_or_else(|| IoError::Parse {
            path: "list.dat".into(),
            line: 0,
            message: "SO2CONV target selection requires list.dat when ipr6 >= 2".to_string(),
        })?;
        let kind = if ipr6 >= 3 {
            SfconvSo2convTargetKind::FeffPath
        } else {
            SfconvSo2convTargetKind::Chi
        };
        let prefix = if ipr6 >= 3 { "feff" } else { "chip" };
        targets.extend(list.entries.iter().map(|entry| SfconvSo2convTarget {
            file_name: format!("{prefix}{}.dat", so2conv_path_suffix(entry.path_index)),
            kind,
        }));
    }

    Ok(())
}

fn push_so2conv_xanes_targets(
    targets: &mut Vec<SfconvSo2convTarget>,
    polarization: i32,
) -> Result<()> {
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_xmu_file_name(polarization)?,
        kind: SfconvSo2convTargetKind::Xmu,
    });
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_chi_file_name(polarization, true)?,
        kind: SfconvSo2convTargetKind::Chi,
    });
    Ok(())
}

fn so2conv_polarization_indices(eels: Option<&EelsInput>) -> Result<Vec<i32>> {
    let Some(eels) = eels.filter(|input| input.calculate_elnes) else {
        return Ok(vec![1]);
    };
    let min = eels.polarization.min;
    let step = eels.polarization.step;
    let max = eels.polarization.max;
    if step == 0 {
        return Err(IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: "SO2CONV EELS polarization step must be nonzero".to_string(),
        });
    }
    if (step > 0 && min > max) || (step < 0 && min < max) {
        return Err(IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: format!(
                "SO2CONV EELS polarization range {min}:{step}:{max} does not reach max"
            ),
        });
    }

    let mut values = Vec::new();
    let mut value = min;
    while if step > 0 { value <= max } else { value >= max } {
        values.push(value);
        value = value.checked_add(step).ok_or_else(|| IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: "SO2CONV EELS polarization range overflows i32".to_string(),
        })?;
    }
    Ok(values)
}

fn so2conv_xmu_file_name(polarization: i32) -> Result<String> {
    so2conv_polarized_file_name("xmu", polarization, false)
}

fn so2conv_chi_file_name(polarization: i32, legacy_xanes_width: bool) -> Result<String> {
    so2conv_polarized_file_name("chi", polarization, legacy_xanes_width)
}

fn so2conv_polarized_file_name(
    stem: &'static str,
    polarization: i32,
    legacy_xanes_width: bool,
) -> Result<String> {
    let full = match polarization {
        1 => format!("{stem}.dat"),
        2..=9 => format!("{stem}0{polarization}.dat"),
        10 => format!("{stem}10.dat"),
        _ => {
            return Err(IoError::Parse {
                path: "eels.inp".into(),
                line: 0,
                message: format!("SO2CONV polarization index must be 1..=10, got {polarization}"),
            });
        }
    };

    if legacy_xanes_width && stem == "chi" && full.len() > 7 {
        Ok(full.chars().take(7).collect())
    } else {
        Ok(full)
    }
}

fn so2conv_path_suffix(path_index: usize) -> String {
    let digits = path_index.to_string();
    match digits.len() {
        0 => "0000".to_string(),
        1 => format!("000{digits}"),
        2 => format!("00{digits}"),
        3 => format!("0{digits}"),
        4 => digits,
        _ => digits.chars().take(4).collect(),
    }
}

#[derive(Debug, Clone, Default)]
struct SfconvSo2convHeaderTokens {
    core_hole_width: Option<HeaderToken>,
    wigner_seitz_radius: Option<HeaderToken>,
    interstitial_potential: Option<HeaderToken>,
    chemical_potential: Option<HeaderToken>,
    fermi_wave_number: Option<HeaderToken>,
}

#[derive(Debug, Clone)]
struct HeaderToken {
    line: usize,
    text: String,
}

fn scan_so2conv_material_header_line(
    line: &str,
    line_number: usize,
    tokens: &mut SfconvSo2convHeaderTokens,
) {
    let last = fortran_trimmed_len(line);
    for start in 0..last.saturating_sub(7) {
        if has_fixed_token(line, start, b"Gam_ch=") {
            tokens.core_hole_width = Some(HeaderToken {
                line: line_number,
                text: fixed_width_field(line, start + 7, 9),
            });
        } else if has_fixed_token(line, start, b"Rs_int=") {
            tokens.wigner_seitz_radius = Some(HeaderToken {
                line: line_number,
                text: fixed_width_field(line, start + 8, 5),
            });
        } else if has_fixed_token(line, start, b"Vint=") {
            tokens.interstitial_potential = Some(HeaderToken {
                line: line_number,
                text: fixed_width_field(line, start + 5, 10),
            });
        } else if has_fixed_token(line, start, b"Mu=") {
            tokens.chemical_potential = Some(HeaderToken {
                line: line_number,
                text: fixed_width_field(line, start + 3, 10),
            });
        } else if has_fixed_token(line, start, b"kf=") {
            tokens.fermi_wave_number = Some(HeaderToken {
                line: line_number,
                text: fixed_width_field(line, start + 3, 9),
            });
        }
    }
}

fn parse_header_token(
    source: &Path,
    token: Option<HeaderToken>,
    field: &'static str,
) -> Result<f64> {
    let token = token.ok_or_else(|| IoError::Parse {
        path: source.to_path_buf(),
        line: 0,
        message: format!("missing SO2CONV header field {field}"),
    })?;
    let normalized = token.text.trim().replace(['D', 'd'], "E");
    let value = normalized.parse::<f64>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line: token.line,
        message: format!("invalid SO2CONV header field {field}: {:?}", token.text),
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IoError::Parse {
            path: source.to_path_buf(),
            line: token.line,
            message: format!("SO2CONV header field {field} must be finite"),
        })
    }
}

fn has_fixed_token(line: &str, start: usize, expected: &[u8]) -> bool {
    let bytes = line.as_bytes();
    bytes
        .get(start..start + expected.len())
        .is_some_and(|actual| actual == expected)
}

fn fixed_width_field(line: &str, start: usize, width: usize) -> String {
    let bytes = line.as_bytes();
    (start..start + width)
        .map(|index| match bytes.get(index) {
            Some(byte) => char::from(*byte),
            None => ' ',
        })
        .collect()
}

fn fortran_trimmed_len(line: &str) -> usize {
    match line.as_bytes().iter().rposition(|byte| *byte != b' ') {
        Some(index) => index + 1,
        None => 0,
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "sfconv.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn sfconv_cfname_line(cfname: &str) -> String {
    if cfname.len() <= 12 {
        format!("{cfname:<12}")
    } else {
        cfname.to_string()
    }
}

fn push_i4_row(out: &mut String, values: impl IntoIterator<Item = i32>) {
    for value in values {
        out.push_str(&format!("{value:4}"));
    }
    out.push('\n');
}

struct SfconvInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> SfconvInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<SfconvInput> {
        self.expect_header("msfconv, ipse, ipsk")?;
        let control_values = self.parse_values::<i32>(3, "SFCONV control line")?;
        let control = SfconvControl {
            msfconv: control_values[0],
            ipse: control_values[1],
            ipsk: control_values[2],
        };

        self.expect_header("wsigk, cen")?;
        let window_values = self.parse_values::<f64>(2, "SFCONV window line")?;
        let window = SfconvWindow {
            wsigk: window_values[0],
            cen: window_values[1],
        };

        self.expect_header("ispec, ipr6")?;
        let spectrum_values = self.parse_values::<i32>(2, "SFCONV spectrum line")?;
        let spectrum = SfconvSpectrum {
            ispec: spectrum_values[0],
            ipr6: spectrum_values[1],
        };

        self.expect_header("cfname")?;
        let (_, line) = self.next_line("SFCONV filename line")?;

        Ok(SfconvInput {
            control,
            window,
            spectrum,
            cfname: line.trim().to_string(),
        })
    }

    fn expect_header(&mut self, expected: &str) -> Result<()> {
        let (line_number, line) = self.next_line(expected)?;
        if line.trim() == expected {
            Ok(())
        } else {
            Err(self.parse_error(
                line_number,
                format!("expected header {expected:?}, found {line:?}"),
            ))
        }
    }

    fn parse_values<T>(&mut self, count: usize, description: &str) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < count {
            return Err(self.parse_error(
                line_number,
                format!("{description} requires {count} fields"),
            ));
        }
        fields
            .iter()
            .take(count)
            .map(|field| parse_field(&self.source, line_number, field))
            .collect()
    }

    fn next_line(&mut self, description: &str) -> Result<(usize, &'a str)> {
        self.lines
            .next()
            .map(|(index, line)| (index + 1, line))
            .ok_or_else(|| self.parse_error(0, format!("expected {description}")))
    }

    fn parse_error(&self, line: usize, message: impl Into<String>) -> IoError {
        IoError::Parse {
            path: self.source.clone(),
            line,
            message: message.into(),
        }
    }
}

fn parse_field<T>(source: &Path, line: usize, field: &str) -> Result<T>
where
    T: FromStr,
{
    field.parse::<T>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("invalid numeric field {field:?}"),
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh, FeffDocument, FeffInput,
        ListDatData, ListDatEntry, rdinp,
    };

    use super::{
        SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvInput, SfconvSo2convTarget,
        SfconvSo2convTargetKind, sfconv_input_string, sfconv_so2conv_header_from_text,
        sfconv_so2conv_material_input_from_header, sfconv_so2conv_targets,
    };

    #[test]
    fn parses_generated_sfconv_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SFCONV
XANES
PRINT 0 0 0 0 0 3
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let sfconv = SfconvInput::parse_str("sfconv.inp", &rdinp::sfconv_inp_string(&document)?)?;

        assert_eq!(sfconv.control.msfconv, 1);
        assert_eq!(sfconv.control.ipse, 0);
        assert_eq!(sfconv.control.ipsk, 0);
        assert_eq!(sfconv.window.wsigk, 0.0);
        assert_eq!(sfconv.window.cen, 0.0);
        assert_eq!(sfconv.spectrum.ispec, 1);
        assert_eq!(sfconv.spectrum.ipr6, 3);
        assert_eq!(sfconv.cfname, "NULL");
        Ok(())
    }

    #[test]
    fn renders_generated_sfconv_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SFCONV
XANES
PRINT 0 0 0 0 0 3
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let expected = rdinp::sfconv_inp_string(&document)?;
        let sfconv = SfconvInput::parse_str("sfconv.inp", &expected)?;
        let rendered = sfconv_input_string(&sfconv)?;
        let reparsed = SfconvInput::parse_str("sfconv.inp", &rendered)?;

        assert_eq!(rendered, expected);
        assert_eq!(reparsed, sfconv);
        Ok(())
    }

    #[test]
    fn rejects_invalid_sfconv_rendering() {
        let input = SfconvInput {
            control: super::SfconvControl {
                msfconv: 1,
                ipse: 0,
                ipsk: 0,
            },
            window: super::SfconvWindow {
                wsigk: f64::NAN,
                cen: 0.0,
            },
            spectrum: super::SfconvSpectrum { ispec: 1, ipr6: 0 },
            cfname: "NULL".to_string(),
        };
        assert!(sfconv_input_string(&input).is_err());
    }

    #[test]
    fn so2conv_targets_match_concrete_feff_reference_exafs_branches() -> crate::Result<()> {
        let list = so2conv_target_list();

        let base = sfconv_so2conv_targets(&sfconv_target_input(0, 1), None, None)?;
        assert_eq!(
            base,
            vec![
                target("xmu.dat", SfconvSo2convTargetKind::Xmu),
                target("chi.dat", SfconvSo2convTargetKind::Chi),
            ]
        );

        let chip = sfconv_so2conv_targets(&sfconv_target_input(0, 2), Some(&list), None)?;
        assert_eq!(
            chip,
            vec![
                target("xmu.dat", SfconvSo2convTargetKind::Xmu),
                target("chi.dat", SfconvSo2convTargetKind::Chi),
                target("chip0001.dat", SfconvSo2convTargetKind::Chi),
                target("chip0023.dat", SfconvSo2convTargetKind::Chi),
                target("chip4567.dat", SfconvSo2convTargetKind::Chi),
            ]
        );

        let feff = sfconv_so2conv_targets(&sfconv_target_input(0, 3), Some(&list), None)?;
        assert_eq!(
            feff,
            vec![
                target("xmu.dat", SfconvSo2convTargetKind::Xmu),
                target("chi.dat", SfconvSo2convTargetKind::Chi),
                target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
                target("feff0023.dat", SfconvSo2convTargetKind::FeffPath),
                target("feff4567.dat", SfconvSo2convTargetKind::FeffPath),
            ]
        );
        Ok(())
    }

    #[test]
    fn so2conv_targets_match_feff_reference_elnes_polarizations() -> crate::Result<()> {
        let eels = eels_target_input(true, 1, 4, 9);
        let targets = sfconv_so2conv_targets(&sfconv_target_input(1, 3), None, Some(&eels))?;

        assert_eq!(
            targets,
            vec![
                target("xmu.dat", SfconvSo2convTargetKind::Xmu),
                target("chi.dat", SfconvSo2convTargetKind::Chi),
                target("xmu05.dat", SfconvSo2convTargetKind::Xmu),
                target("chi05.d", SfconvSo2convTargetKind::Chi),
                target("xmu09.dat", SfconvSo2convTargetKind::Xmu),
                target("chi09.d", SfconvSo2convTargetKind::Chi),
            ]
        );
        Ok(())
    }

    #[test]
    fn so2conv_targets_reject_missing_list_and_invalid_controls() {
        let missing_list = sfconv_so2conv_targets(&sfconv_target_input(0, 2), None, None).err();
        assert!(missing_list.is_some_and(|error| {
            error
                .to_string()
                .contains("requires list.dat when ipr6 >= 2")
        }));

        let invalid_ispec = sfconv_so2conv_targets(&sfconv_target_input(2, 0), None, None).err();
        assert!(
            invalid_ispec
                .is_some_and(|error| error.to_string().contains("ispec must be -1, 0, or 1"))
        );

        let bad_step = eels_target_input(true, 1, 0, 9);
        let invalid_range =
            sfconv_so2conv_targets(&sfconv_target_input(1, 0), None, Some(&bad_step)).err();
        assert!(invalid_range.is_some_and(|error| {
            error
                .to_string()
                .contains("EELS polarization step must be nonzero")
        }));
    }

    #[test]
    fn parses_so2conv_material_header_like_feff_reference() -> crate::Result<()> {
        let header = concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  1.0 2.0 3.0\n",
        );

        let material = sfconv_so2conv_material_input_from_header("xmu.dat", header)?;

        assert_eq!(material.core_hole_width_ev, 1.729);
        assert_eq!(material.wigner_seitz_radius, 2.05);
        assert_eq!(material.interstitial_potential_ev, 12.34);
        assert_eq!(material.chemical_potential_ev, 18.76);
        assert_eq!(material.fermi_wave_number_inv_angstrom, 1.23);
        Ok(())
    }

    #[test]
    fn so2conv_material_header_uses_last_value_before_separator_like_feff() -> crate::Result<()> {
        let header = concat!(
            "# one Gam_ch= 4.000000 Rs_int= 9.99 Vint= 99.00000 ",
            "Mu= 88.00000 kf= 7.000000\n",
            "# two Gam_ch= 5.533000 Rs_int= 1.42 Vint=-3.250000 ",
            "Mu=  0.80000 kf= 0.780000\n",
            " ------------------------------------------------------------------------------\n",
            "# later Gam_ch= 9.000000 Rs_int= 8.88 Vint= 77.00000 ",
            "Mu= 66.00000 kf= 5.000000\n",
        );

        let material = sfconv_so2conv_material_input_from_header("chip0001.dat", header)?;

        assert_eq!(material.core_hole_width_ev, 5.533);
        assert_eq!(material.wigner_seitz_radius, 1.42);
        assert_eq!(material.interstitial_potential_ev, -3.25);
        assert_eq!(material.chemical_potential_ev, 0.8);
        assert_eq!(material.fermi_wave_number_inv_angstrom, 0.78);
        Ok(())
    }

    #[test]
    fn so2conv_material_header_rejects_missing_or_invalid_values() {
        let missing = "# Header Gam_ch= 1.729000 Rs_int= 2.05\n";
        let missing_error = sfconv_so2conv_material_input_from_header("chi.dat", missing).err();
        assert!(missing_error.is_some_and(|error| {
            error
                .to_string()
                .contains("missing SO2CONV header field Vint")
        }));

        let invalid = concat!(
            "# Header Gam_ch= not_real Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
        );
        let invalid_error = sfconv_so2conv_material_input_from_header("chi.dat", invalid).err();
        assert!(invalid_error.is_some_and(|error| {
            error
                .to_string()
                .contains("invalid SO2CONV header field Gam_ch")
        }));
    }

    #[test]
    fn so2conv_header_detects_previous_convolution_before_separator_like_feff() -> crate::Result<()>
    {
        let before = format!(
            concat!(
                "{}                                                     \n",
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
                "Mu= 18.76000 kf= 1.230000\n",
                " ------------------------------------------------------------------------------\n",
            ),
            SFCONV_SO2CONV_CONVOLUTED_MARKER
        );
        let before_scan = sfconv_so2conv_header_from_text("xmu.dat", &before)?;
        assert!(before_scan.already_convoluted);
        assert_eq!(before_scan.material.core_hole_width_ev, 1.729);

        let after = format!(
            concat!(
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
                "Mu= 18.76000 kf= 1.230000\n",
                " ------------------------------------------------------------------------------\n",
                "{}\n",
            ),
            SFCONV_SO2CONV_CONVOLUTED_MARKER
        );
        let after_scan = sfconv_so2conv_header_from_text("xmu.dat", &after)?;
        assert!(!after_scan.already_convoluted);
        assert_eq!(after_scan.material.fermi_wave_number_inv_angstrom, 1.23);
        Ok(())
    }

    fn sfconv_target_input(ispec: i32, ipr6: i32) -> SfconvInput {
        SfconvInput {
            control: super::SfconvControl {
                msfconv: 1,
                ipse: 0,
                ipsk: 0,
            },
            window: super::SfconvWindow {
                wsigk: 0.0,
                cen: 0.0,
            },
            spectrum: super::SfconvSpectrum { ispec, ipr6 },
            cfname: "NULL".to_string(),
        }
    }

    fn so2conv_target_list() -> ListDatData {
        ListDatData {
            titles: vec!["target oracle".to_string()],
            entries: [1, 23, 4567]
                .into_iter()
                .map(|path_index| ListDatEntry {
                    path_index,
                    sigma2: 0.0,
                    amplitude_ratio: 1.0,
                    degeneracy: 4.0,
                    leg_count: 2,
                    effective_half_path_length_angstrom: 2.5,
                })
                .collect(),
        }
    }

    fn eels_target_input(calculate_elnes: bool, min: i32, step: i32, max: i32) -> EelsInput {
        EelsInput {
            calculate_elnes,
            control: EelsControl {
                average: 0,
                relativistic: 0,
                cross_terms: 0,
                input: 1,
                spectrum_column: 0,
            },
            polarization: EelsPolarization { min, step, max },
            beam_energy: 200.0,
            beam_direction: [0.0, 0.0, 1.0],
            angles: EelsAngles {
                collection: 0.0,
                convergence: 0.0,
            },
            qmesh: EelsQMesh {
                radial: 1,
                angular: 1,
            },
            detector: [0.0, 0.0],
            magic: 0,
            magic_energy: 0.0,
        }
    }

    fn target(file_name: &str, kind: SfconvSo2convTargetKind) -> SfconvSo2convTarget {
        SfconvSo2convTarget {
            file_name: file_name.to_string(),
            kind,
        }
    }
}
