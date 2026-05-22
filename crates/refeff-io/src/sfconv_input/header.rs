use std::path::{Path, PathBuf};

use refeff_core::SfconvSo2convMaterialInput;

use crate::{IoError, Result};

use super::types::{SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvSo2convHeader};

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

pub(super) fn fixed_width_field(line: &str, start: usize, width: usize) -> String {
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
