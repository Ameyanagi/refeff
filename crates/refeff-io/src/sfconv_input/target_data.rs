use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use ndarray::Array1;
use refeff_core::{SfconvExafsConvolution, SfconvPathAverage, SfconvXanesConvolution};

use crate::chi_dat::{ChiDatData, chi_dat_string, parse_chi_dat, validate_chi_dat};
use crate::format::write_fortran_exp;
use crate::xmu_dat::{XmuDatData, parse_xmu_dat, validate_xmu_dat, xmu_dat_string};
use crate::{IoError, Result};

use super::header::{fixed_width_field, sfconv_so2conv_header_from_text};
use super::types::{
    SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvSo2convFeffPathData, SfconvSo2convTarget,
    SfconvSo2convTargetData, SfconvSo2convTargetKind,
};

const SFCONV_SO2CONV_FEFF_PATH_ROW_WIDTH: usize = 7;

pub fn sfconv_so2conv_target_data_from_text(
    source: impl Into<PathBuf>,
    target: &SfconvSo2convTarget,
    text: &str,
) -> Result<SfconvSo2convTargetData> {
    let source = source.into();
    let header = sfconv_so2conv_header_from_text(source.clone(), text)?;
    match target.kind {
        SfconvSo2convTargetKind::Xmu => Ok(SfconvSo2convTargetData::Xmu {
            header,
            data: parse_xmu_dat(text)?,
        }),
        SfconvSo2convTargetKind::Chi => Ok(SfconvSo2convTargetData::Chi {
            header,
            data: parse_chi_dat(text)?,
        }),
        SfconvSo2convTargetKind::FeffPath => Ok(SfconvSo2convTargetData::FeffPath {
            header,
            data: parse_so2conv_feff_path_data(&source, text)?,
        }),
    }
}

/// Render one selected `SO2CONV` target file.
///
/// This renders the parsed target data without adding the previous-convolution
/// marker. Use [`sfconv_so2conv_convoluted_target_data_string`] for FEFF's
/// overwritten-output form after applying many-body convolution.
pub fn sfconv_so2conv_target_data_string(data: &SfconvSo2convTargetData) -> Result<String> {
    match data {
        SfconvSo2convTargetData::Xmu { data, .. } => xmu_dat_string(data),
        SfconvSo2convTargetData::Chi { data, .. } => chi_dat_string(data),
        SfconvSo2convTargetData::FeffPath { data, .. } => {
            sfconv_so2conv_feff_path_data_string(data)
        }
    }
}

/// Render one selected `SO2CONV` target file with FEFF's convolution marker.
pub fn sfconv_so2conv_convoluted_target_data_string(
    data: &SfconvSo2convTargetData,
) -> Result<String> {
    let rendered = sfconv_so2conv_target_data_string(data)?;
    if rendered.lines().next() == Some(SFCONV_SO2CONV_CONVOLUTED_MARKER) {
        Ok(rendered)
    } else {
        Ok(format!("{SFCONV_SO2CONV_CONVOLUTED_MARKER}\n{rendered}"))
    }
}

/// Write one selected `SO2CONV` target file without adding a marker.
pub fn write_sfconv_so2conv_target_data(
    path: impl AsRef<Path>,
    data: &SfconvSo2convTargetData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, sfconv_so2conv_target_data_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Render one FEFF `feffNNNN.dat` path file consumed by `SO2CONV`.
pub fn sfconv_so2conv_feff_path_data_string(data: &SfconvSo2convFeffPathData) -> Result<String> {
    validate_so2conv_feff_path_data(Path::new("feffNNNN.dat"), data)?;
    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for ((((((xk2, caph2), xmfeff2), phfeff2), redfac2), xlam2), realck2) in data
        .wave_number_inverse_angstrom
        .iter()
        .zip(data.central_phase.iter())
        .zip(data.effective_amplitude.iter())
        .zip(data.effective_phase.iter())
        .zip(data.reduction_factor.iter())
        .zip(data.mean_free_path_angstrom.iter())
        .zip(data.real_momentum_inverse_angstrom.iter())
    {
        write_so2conv_feff_path_row(
            &mut out,
            [*xk2, *caph2, *xmfeff2, *phfeff2, *redfac2, *xlam2, *realck2],
        )?;
    }
    Ok(out)
}

/// Write one FEFF `feffNNNN.dat` path file consumed by `SO2CONV`.
pub fn write_sfconv_so2conv_feff_path_data(
    path: impl AsRef<Path>,
    data: &SfconvSo2convFeffPathData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, sfconv_so2conv_feff_path_data_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Write one selected `SO2CONV` target file with FEFF's convolution marker.
pub fn write_sfconv_so2conv_convoluted_target_data(
    path: impl AsRef<Path>,
    data: &SfconvSo2convTargetData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, sfconv_so2conv_convoluted_target_data_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Build a convolved `xmu.dat` table from row-level SO2CONV XANES results.
///
/// The FEFF driver preserves the incident-energy, edge-relative energy, and
/// wave-number columns from the source table, then replaces `mu`, `mu0`, and
/// `chi` with the SO2CONV absorption, embedded-background, and fine-structure
/// rows.
pub fn sfconv_so2conv_xmu_data_from_convolution_rows(
    source: &XmuDatData,
    rows: &[SfconvXanesConvolution],
) -> Result<XmuDatData> {
    if rows.len() != source.point_count() {
        return Err(IoError::XmuDatShape {
            field: "SO2CONV XANES rows",
            actual: rows.len(),
            expected: source.point_count(),
        });
    }

    let output = XmuDatData {
        header_lines: source.header_lines.clone(),
        normalization: source.normalization,
        photon_energy_ev: source.photon_energy_ev.clone(),
        relative_energy_ev: source.relative_energy_ev.clone(),
        wave_number: source.wave_number.clone(),
        mu: Array1::from_iter(rows.iter().map(|row| row.absorption)),
        mu0: Array1::from_iter(rows.iter().map(|row| row.embedded_background)),
        chi: Array1::from_iter(rows.iter().map(|row| row.fine_structure)),
    };
    validate_xmu_dat(&output)?;
    Ok(output)
}

/// Build a convolved `chi.dat` or `chipNNNN.dat` table from EXAFS row results.
///
/// FEFF writes the many-body imaginary EXAFS signal as the `chi` column,
/// followed by the combined magnitude, unwrapped phase, and phase correction.
/// Diagnostic complex-wave-number columns are intentionally dropped because the
/// legacy SO2CONV output format is always the five-column EXAFS form.
pub fn sfconv_so2conv_chi_data_from_convolution_rows(
    source: &ChiDatData,
    rows: &[SfconvExafsConvolution],
) -> Result<ChiDatData> {
    if rows.len() != source.point_count() {
        return Err(IoError::ChiDatShape {
            field: "SO2CONV EXAFS rows",
            actual: rows.len(),
            expected: source.point_count(),
        });
    }

    let output = ChiDatData {
        header_lines: source.header_lines.clone(),
        wave_number: source.wave_number.clone(),
        chi: Array1::from_iter(rows.iter().map(|row| row.imaginary)),
        magnitude: Array1::from_iter(rows.iter().map(|row| row.magnitude)),
        phase: Array1::from_iter(rows.iter().map(|row| row.output_phase)),
        phase_minus_2kr: Some(Array1::from_iter(
            rows.iter().map(|row| row.output_phase_minus_original),
        )),
        ckp_real: None,
        ckp_imag: None,
    };
    validate_chi_dat(&output)?;
    Ok(output)
}

/// Build a convolved `feffNNNN.dat` path table from SO2CONV path averages.
///
/// FEFF applies averaged many-body amplitude reductions to `redfac2` and adds
/// averaged phase shifts to `caph2`, while preserving the path grid and
/// scattering columns.
pub fn sfconv_so2conv_feff_path_data_from_averages(
    source: &SfconvSo2convFeffPathData,
    averages: &[SfconvPathAverage],
) -> Result<SfconvSo2convFeffPathData> {
    if averages.len() != source.point_count() {
        return Err(parse_error(
            Path::new("feffNNNN.dat"),
            0,
            format!(
                "SO2CONV path averages row count {} does not match target row count {}",
                averages.len(),
                source.point_count()
            ),
        ));
    }

    let output = SfconvSo2convFeffPathData {
        header_lines: source.header_lines.clone(),
        leg_count: source.leg_count,
        degeneracy: source.degeneracy,
        effective_half_path_length_angstrom: source.effective_half_path_length_angstrom,
        wave_number_inverse_angstrom: source.wave_number_inverse_angstrom.clone(),
        central_phase: Array1::from_iter(
            source
                .central_phase
                .iter()
                .zip(averages.iter())
                .map(|(&phase, average)| phase + average.phase_shift),
        ),
        effective_amplitude: source.effective_amplitude.clone(),
        effective_phase: source.effective_phase.clone(),
        reduction_factor: Array1::from_iter(
            source
                .reduction_factor
                .iter()
                .zip(averages.iter())
                .map(|(&reduction, average)| reduction * average.amplitude_reduction),
        ),
        mean_free_path_angstrom: source.mean_free_path_angstrom.clone(),
        real_momentum_inverse_angstrom: source.real_momentum_inverse_angstrom.clone(),
    };
    validate_so2conv_feff_path_data(Path::new("feffNNNN.dat"), &output)?;
    Ok(output)
}

fn parse_so2conv_feff_path_data(source: &Path, text: &str) -> Result<SfconvSo2convFeffPathData> {
    let mut header_lines = Vec::new();
    let mut leg_count = None;
    let mut degeneracy = None;
    let mut effective_half_path_length_angstrom = None;
    let mut after_separator = false;
    let mut after_column_marker = false;
    let mut wave_number_inverse_angstrom = Vec::new();
    let mut central_phase = Vec::new();
    let mut effective_amplitude = Vec::new();
    let mut effective_phase = Vec::new();
    let mut reduction_factor = Vec::new();
    let mut mean_free_path_angstrom = Vec::new();
    let mut real_momentum_inverse_angstrom = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        if fixed_width_field(line, 5, 9) == "---------" {
            after_separator = true;
            header_lines.push(raw.to_string());
            continue;
        }

        if after_separator && line.contains("reff") {
            let metadata = parse_so2conv_reff_metadata(source, line_number, line)?;
            leg_count = Some(metadata.0);
            degeneracy = Some(metadata.1);
            effective_half_path_length_angstrom = Some(metadata.2);
            header_lines.push(raw.to_string());
            continue;
        }

        if after_separator && line.contains("@#") {
            after_column_marker = true;
            header_lines.push(raw.to_string());
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let row_can_start =
            tokens.first().is_some_and(|token| is_numeric_token(token)) && after_separator;
        if row_can_start
            && (after_column_marker || tokens.len() == SFCONV_SO2CONV_FEFF_PATH_ROW_WIDTH)
        {
            if tokens.len() != SFCONV_SO2CONV_FEFF_PATH_ROW_WIDTH {
                return Err(parse_error(
                    source,
                    line_number,
                    format!(
                        "SO2CONV feff path row requires {SFCONV_SO2CONV_FEFF_PATH_ROW_WIDTH} fields, got {}",
                        tokens.len()
                    ),
                ));
            }
            wave_number_inverse_angstrom.push(parse_so2conv_f64(
                source,
                line_number,
                "xk2",
                tokens[0],
            )?);
            central_phase.push(parse_so2conv_f64(source, line_number, "caph2", tokens[1])?);
            effective_amplitude.push(parse_so2conv_f64(
                source,
                line_number,
                "xmfeff2",
                tokens[2],
            )?);
            effective_phase.push(parse_so2conv_f64(
                source,
                line_number,
                "phfeff2",
                tokens[3],
            )?);
            reduction_factor.push(parse_so2conv_f64(
                source,
                line_number,
                "redfac2",
                tokens[4],
            )?);
            mean_free_path_angstrom.push(parse_so2conv_f64(
                source,
                line_number,
                "xlam2",
                tokens[5],
            )?);
            real_momentum_inverse_angstrom.push(parse_so2conv_f64(
                source,
                line_number,
                "realck2",
                tokens[6],
            )?);
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let data = SfconvSo2convFeffPathData {
        header_lines,
        leg_count: leg_count
            .ok_or_else(|| parse_error(source, 0, "missing SO2CONV feff path reff leg count"))?,
        degeneracy: degeneracy
            .ok_or_else(|| parse_error(source, 0, "missing SO2CONV feff path reff degeneracy"))?,
        effective_half_path_length_angstrom: effective_half_path_length_angstrom
            .ok_or_else(|| parse_error(source, 0, "missing SO2CONV feff path reff distance"))?,
        wave_number_inverse_angstrom: Array1::from_vec(wave_number_inverse_angstrom),
        central_phase: Array1::from_vec(central_phase),
        effective_amplitude: Array1::from_vec(effective_amplitude),
        effective_phase: Array1::from_vec(effective_phase),
        reduction_factor: Array1::from_vec(reduction_factor),
        mean_free_path_angstrom: Array1::from_vec(mean_free_path_angstrom),
        real_momentum_inverse_angstrom: Array1::from_vec(real_momentum_inverse_angstrom),
    };
    validate_so2conv_feff_path_data(source, &data)?;
    Ok(data)
}

fn parse_so2conv_reff_metadata(
    source: &Path,
    line_number: usize,
    line: &str,
) -> Result<(usize, f64, f64)> {
    let content = match line.as_bytes().first() {
        Some(first) if first.is_ascii_digit() => line,
        Some(_) => line.get(1..).map_or("", |rest| rest),
        None => "",
    };
    let numeric = content
        .split_whitespace()
        .filter(|token| is_numeric_token(token))
        .take(3)
        .collect::<Vec<_>>();
    if numeric.len() != 3 {
        return Err(parse_error(
            source,
            line_number,
            "SO2CONV feff path reff metadata requires nleg, degeneracy, and reff",
        ));
    }
    let leg_count = numeric[0].parse::<usize>().map_err(|_| {
        parse_error(
            source,
            line_number,
            format!("invalid SO2CONV feff path nleg {:?}", numeric[0]),
        )
    })?;
    let degeneracy = parse_so2conv_f64(source, line_number, "deg", numeric[1])?;
    let effective_half_path_length_angstrom =
        parse_so2conv_f64(source, line_number, "reff", numeric[2])?;
    Ok((leg_count, degeneracy, effective_half_path_length_angstrom))
}

fn validate_so2conv_feff_path_data(source: &Path, data: &SfconvSo2convFeffPathData) -> Result<()> {
    if data.leg_count == 0 {
        return Err(parse_error(
            source,
            0,
            "SO2CONV feff path leg count must be positive",
        ));
    }
    validate_so2conv_finite(source, 0, "deg", data.degeneracy)?;
    validate_so2conv_finite(source, 0, "reff", data.effective_half_path_length_angstrom)?;
    if data.point_count() == 0 {
        return Err(parse_error(
            source,
            0,
            "SO2CONV feff path data requires at least one row",
        ));
    }
    let point_count = data.point_count();
    validate_so2conv_len(source, "caph2", data.central_phase.len(), point_count)?;
    validate_so2conv_len(
        source,
        "xmfeff2",
        data.effective_amplitude.len(),
        point_count,
    )?;
    validate_so2conv_len(source, "phfeff2", data.effective_phase.len(), point_count)?;
    validate_so2conv_len(source, "redfac2", data.reduction_factor.len(), point_count)?;
    validate_so2conv_len(
        source,
        "xlam2",
        data.mean_free_path_angstrom.len(),
        point_count,
    )?;
    validate_so2conv_len(
        source,
        "realck2",
        data.real_momentum_inverse_angstrom.len(),
        point_count,
    )?;

    for (index, values) in data
        .wave_number_inverse_angstrom
        .iter()
        .zip(data.central_phase.iter())
        .zip(data.effective_amplitude.iter())
        .zip(data.effective_phase.iter())
        .zip(data.reduction_factor.iter())
        .zip(data.mean_free_path_angstrom.iter())
        .zip(data.real_momentum_inverse_angstrom.iter())
        .enumerate()
    {
        let ((((((xk2, caph2), xmfeff2), phfeff2), redfac2), xlam2), realck2) = values;
        let line = index + 1;
        validate_so2conv_finite(source, line, "xk2", *xk2)?;
        validate_so2conv_finite(source, line, "caph2", *caph2)?;
        validate_so2conv_finite(source, line, "xmfeff2", *xmfeff2)?;
        validate_so2conv_finite(source, line, "phfeff2", *phfeff2)?;
        validate_so2conv_finite(source, line, "redfac2", *redfac2)?;
        validate_so2conv_finite(source, line, "xlam2", *xlam2)?;
        validate_so2conv_finite(source, line, "realck2", *realck2)?;
    }
    Ok(())
}

fn validate_so2conv_len(
    source: &Path,
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(parse_error(
            source,
            0,
            format!("SO2CONV feff path {field} length {actual} does not match {expected}"),
        ))
    }
}

fn validate_so2conv_finite(
    source: &Path,
    line: usize,
    field: &'static str,
    value: f64,
) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(parse_error(
            source,
            line,
            format!("SO2CONV feff path {field} must be finite"),
        ))
    }
}

fn write_so2conv_feff_path_row(out: &mut String, fields: [f64; 7]) -> Result<()> {
    write!(out, " {:6.3} ", fields[0])?;
    write_fortran_exp(out, fields[1], 11, 4)?;
    out.push(' ');
    write_fortran_exp(out, fields[2], 11, 4)?;
    out.push(' ');
    write_fortran_exp(out, fields[3], 11, 4)?;
    out.push(' ');
    write_fortran_exp(out, fields[4], 10, 3)?;
    out.push(' ');
    write_fortran_exp(out, fields[5], 11, 4)?;
    out.push(' ');
    write_fortran_exp(out, fields[6], 11, 4)?;
    out.push(' ');
    out.push('\n');
    Ok(())
}

fn parse_so2conv_f64(source: &Path, line: usize, field: &'static str, token: &str) -> Result<f64> {
    let normalized = token.replace(['D', 'd'], "E");
    let value = normalized.parse::<f64>().map_err(|_| {
        parse_error(
            source,
            line,
            format!("invalid SO2CONV feff path {field}: {token:?}"),
        )
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error(
            source,
            line,
            format!("SO2CONV feff path {field} must be finite"),
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

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}
