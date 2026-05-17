//! Typed reader for FEFF `sfconv.inp` module handoff files.
//!
//! `sfconv.inp` controls spectral-function convolution after spectrum
//! assembly.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ndarray::Array1;
use refeff_core::SfconvSo2convMaterialInput;

use crate::chi_dat::{ChiDatData, chi_dat_string, parse_chi_dat};
use crate::eels_input::EelsInput;
use crate::format::write_fortran_exp;
use crate::list_dat::ListDatData;
use crate::xmu_dat::{XmuDatData, parse_xmu_dat, xmu_dat_string};
use crate::{IoError, Result};

/// FEFF marker written at the top of files already processed by `SO2CONV`.
pub const SFCONV_SO2CONV_CONVOLUTED_MARKER: &str = "# Convoluted with A(omega).";
const SFCONV_SO2CONV_FEFF_PATH_ROW_WIDTH: usize = 7;

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

/// Parsed contents of one selected `SO2CONV` input file.
#[derive(Debug, Clone, PartialEq)]
pub enum SfconvSo2convTargetData {
    /// `xmu.dat` XANES-style table.
    Xmu {
        /// Header material and previous-convolution status.
        header: SfconvSo2convHeader,
        /// Parsed six-column FEFF `xmu.dat` data.
        data: XmuDatData,
    },
    /// `chi.dat` or `chipNNNN.dat` EXAFS-style table.
    Chi {
        /// Header material and previous-convolution status.
        header: SfconvSo2convHeader,
        /// Parsed FEFF `chi.dat`/`chipNNNN.dat` data.
        data: ChiDatData,
    },
    /// `feffNNNN.dat` path table consumed by `SO2CONV`.
    FeffPath {
        /// Header material and previous-convolution status.
        header: SfconvSo2convHeader,
        /// Parsed seven-column path data plus `reff` metadata.
        data: SfconvSo2convFeffPathData,
    },
}

/// Plain-text `feffNNNN.dat` path data consumed by `SO2CONV`.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSo2convFeffPathData {
    /// Header and metadata lines before the numeric path table.
    pub header_lines: Vec<String>,
    /// Number of legs from the `reff` metadata line.
    pub leg_count: usize,
    /// Path degeneracy from the `reff` metadata line.
    pub degeneracy: f64,
    /// Effective scattering half-path length in Angstrom from the file.
    pub effective_half_path_length_angstrom: f64,
    /// `feffNNNN.dat` path momentum grid in inverse Angstrom, FEFF `xk2`.
    pub wave_number_inverse_angstrom: Array1<f64>,
    /// Central atom phase shift, FEFF `caph2`.
    pub central_phase: Array1<f64>,
    /// Effective scattering amplitude, FEFF `xmfeff2`.
    pub effective_amplitude: Array1<f64>,
    /// Effective scattering phase, FEFF `phfeff2`.
    pub effective_phase: Array1<f64>,
    /// Reduction factor, FEFF `redfac2`.
    pub reduction_factor: Array1<f64>,
    /// Mean free path in Angstrom, FEFF `xlam2`.
    pub mean_free_path_angstrom: Array1<f64>,
    /// Real part of the complex momentum in inverse Angstrom, FEFF `realck2`.
    pub real_momentum_inverse_angstrom: Array1<f64>,
}

impl SfconvSo2convTargetData {
    /// Header scan result for this target data.
    #[must_use]
    pub fn header(&self) -> SfconvSo2convHeader {
        match self {
            Self::Xmu { header, .. } | Self::Chi { header, .. } | Self::FeffPath { header, .. } => {
                *header
            }
        }
    }

    /// Number of numeric spectrum or path rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        match self {
            Self::Xmu { data, .. } => data.point_count(),
            Self::Chi { data, .. } => data.point_count(),
            Self::FeffPath { data, .. } => data.point_count(),
        }
    }
}

impl SfconvSo2convFeffPathData {
    /// Number of path data rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.wave_number_inverse_angstrom.len()
    }
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

/// Parse one selected `SO2CONV` target file.
///
/// Existing FEFF `xmu.dat` and `chi.dat` codecs are reused for spectrum
/// tables. `feffNNNN.dat` uses the plain seven-column text layout read inside
/// `SO2CONV`, not the PAD-backed `feff.bin` handoff format.
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

fn sfconv_so2conv_feff_path_data_string(data: &SfconvSo2convFeffPathData) -> Result<String> {
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
        IoError, ListDatData, ListDatEntry, rdinp,
    };

    use super::{
        SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvInput, SfconvSo2convTarget,
        SfconvSo2convTargetData, SfconvSo2convTargetKind, sfconv_input_string,
        sfconv_so2conv_header_from_text, sfconv_so2conv_material_input_from_header,
        sfconv_so2conv_target_data_from_text, sfconv_so2conv_target_data_string,
        sfconv_so2conv_targets,
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

    #[test]
    fn parses_so2conv_xmu_target_data() -> crate::Result<()> {
        let text = concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
        );
        let parsed = sfconv_so2conv_target_data_from_text(
            "xmu.dat",
            &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            text,
        )?;

        match parsed {
            SfconvSo2convTargetData::Xmu { header, data } => {
                assert_eq!(header.material.core_hole_width_ev, 1.729);
                assert_eq!(data.point_count(), 1);
                assert_eq!(data.mu[0], 1.0);
                assert_eq!(data.chi[0], 0.1);
            }
            _ => return Err(wrong_target_variant("xmu target")),
        }
        Ok(())
    }

    #[test]
    fn parses_so2conv_chi_target_data() -> crate::Result<()> {
        let text = concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "  0.5000  1.10000E-01 2.20000E+00 -3.30000E-01\n",
            "  0.6000  1.20000E-01 2.30000E+00 -3.40000E-01\n",
        );
        let parsed = sfconv_so2conv_target_data_from_text(
            "chi.dat",
            &target("chi.dat", SfconvSo2convTargetKind::Chi),
            text,
        )?;

        match parsed {
            SfconvSo2convTargetData::Chi { header, data } => {
                assert_eq!(header.material.fermi_wave_number_inv_angstrom, 1.23);
                assert_eq!(data.point_count(), 2);
                assert_eq!(data.wave_number[1], 0.6);
                assert_eq!(data.phase[1], -0.34);
            }
            _ => return Err(wrong_target_variant("chi target")),
        }
        Ok(())
    }

    #[test]
    fn parses_so2conv_feff_path_target_data_like_feff_reference() -> crate::Result<()> {
        let text = concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#    3   4.250   2.7500 reff path metadata\n",
            "#       k          phase @#\n",
            "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
        );
        let parsed = sfconv_so2conv_target_data_from_text(
            "feff0001.dat",
            &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
            text,
        )?;

        match parsed {
            SfconvSo2convTargetData::FeffPath { header, data } => {
                assert_eq!(header.material.interstitial_potential_ev, 12.34);
                assert_eq!(data.leg_count, 3);
                assert_eq!(data.degeneracy, 4.25);
                assert_eq!(data.effective_half_path_length_angstrom, 2.75);
                assert_eq!(data.point_count(), 1);
                assert_eq!(data.wave_number_inverse_angstrom[0], 0.5);
                assert_eq!(data.central_phase[0], 0.11);
                assert_eq!(data.effective_amplitude[0], 2.2);
                assert_eq!(data.effective_phase[0], -0.33);
                assert_eq!(data.reduction_factor[0], 0.88);
                assert_eq!(data.mean_free_path_angstrom[0], 4.4);
                assert_eq!(data.real_momentum_inverse_angstrom[0], 0.66);
            }
            _ => return Err(wrong_target_variant("feff path target")),
        }
        Ok(())
    }

    #[test]
    fn rejects_bad_so2conv_feff_path_target_data() {
        let missing_reff = concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#       k          phase @#\n",
            "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
        );
        let missing_error = sfconv_so2conv_target_data_from_text(
            "feff0001.dat",
            &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
            missing_reff,
        )
        .err();
        assert!(missing_error.is_some_and(|error| {
            error
                .to_string()
                .contains("missing SO2CONV feff path reff leg count")
        }));

        let short_row = concat!(
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#    3   4.250   2.7500 reff path metadata\n",
            "#       k          phase @#\n",
            "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01\n",
        );
        let row_error = sfconv_so2conv_target_data_from_text(
            "feff0001.dat",
            &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
            short_row,
        )
        .err();
        assert!(row_error.is_some_and(|error| {
            error
                .to_string()
                .contains("SO2CONV feff path row requires 7 fields")
        }));
    }

    #[test]
    fn renders_so2conv_feff_path_target_data_like_feff_reference() -> crate::Result<()> {
        let parsed = sfconv_so2conv_target_data_from_text(
            "feff0001.dat",
            &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
            concat!(
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
                "Mu= 18.76000 kf= 1.230000\n",
                " ------------------------------------------------------------------------------\n",
                "#    3   4.250   2.7500 reff path metadata\n",
                "#       k          phase @#\n",
                "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
            ),
        )?;

        let rendered = sfconv_so2conv_target_data_string(&parsed)?;

        assert_eq!(
            rendered,
            concat!(
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
                "Mu= 18.76000 kf= 1.230000\n",
                " ------------------------------------------------------------------------------\n",
                "#    3   4.250   2.7500 reff path metadata\n",
                "#       k          phase @#\n",
                "  0.500  1.1000E-01  2.2000E+00 -3.3000E-01  8.800E-01  4.4000E+00  6.6000E-01 \n",
            )
        );
        assert_eq!(
            sfconv_so2conv_target_data_from_text(
                "feff0001.dat",
                &target("feff0001.dat", SfconvSo2convTargetKind::FeffPath),
                &rendered,
            )?,
            parsed
        );
        Ok(())
    }

    #[test]
    fn renders_so2conv_convoluted_target_marker() -> crate::Result<()> {
        let parsed = sfconv_so2conv_target_data_from_text(
            "xmu.dat",
            &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
            concat!(
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
                "Mu= 18.76000 kf= 1.230000\n",
                " ------------------------------------------------------------------------------\n",
                "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
            ),
        )?;

        let rendered = super::sfconv_so2conv_convoluted_target_data_string(&parsed)?;

        assert!(rendered.starts_with(SFCONV_SO2CONV_CONVOLUTED_MARKER));
        assert_eq!(
            super::sfconv_so2conv_convoluted_target_data_string(
                &sfconv_so2conv_target_data_from_text(
                    "xmu.dat",
                    &target("xmu.dat", SfconvSo2convTargetKind::Xmu),
                    &rendered,
                )?
            )?,
            rendered
        );
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

    fn wrong_target_variant(context: &'static str) -> IoError {
        IoError::Parse {
            path: "test".into(),
            line: 0,
            message: format!("wrong SO2CONV target variant for {context}"),
        }
    }
}
