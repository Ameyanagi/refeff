//! FEFF RIXS output table codecs.
//!
//! `rixsET.dat` stores a two-axis RIXS map followed by one or more channel
//! columns. `herfd.dat` and `herfd-sat.dat` store one-axis line spectra with
//! the same channel-column convention. FEFF separates each map block with a
//! blank line; the parser records those block lengths so rendering can preserve
//! the same table structure.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, Axis};
use refeff_core::{
    FEFF_HARTREE_EV, RixsFinalSpectrum, RixsFinalSpectrumInput, RixsPoleNormalizationInput,
    RixsSatelliteSpectrumInput, rixs_final_spectrum, rixs_normalize_poles, rixs_satellite_spectrum,
};

use crate::energy_output::EdgesDatData;
use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;
use crate::rixs_input::RixsEnergyWindow;
use crate::xmu_dat::XmuDatData;

const RIXS_MAP_MIN_ROW_WIDTH: usize = 3;
const RIXS_LINE_MIN_ROW_WIDTH: usize = 2;
const RIXS_MAP_ALLOWED_ROW_WIDTHS: &str = "at least 3";
const RIXS_LINE_ALLOWED_ROW_WIDTHS: &str = "at least 2";
const RIXS_CHANNEL_COLUMNS: &str = "at least 1";

/// Parsed FEFF two-axis RIXS map such as `rixsET.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsMapData {
    /// Header and comment lines before or around the numeric map.
    pub header_lines: Vec<String>,
    /// Number of contiguous numeric rows in each FEFF block.
    pub block_lengths: Vec<usize>,
    /// First energy axis in eV.
    pub first_energy_ev: Array1<f64>,
    /// Second energy axis in eV.
    pub second_energy_ev: Array1<f64>,
    /// RIXS channel columns for each map row.
    pub channels: Array2<f64>,
}

impl RixsMapData {
    /// Number of numeric map rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.first_energy_ev.len()
    }

    /// Number of channel columns after the two energy axes.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.ncols()
    }
}

/// Parsed FEFF one-axis RIXS line spectrum such as `herfd.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsLineData {
    /// Header and comment lines before or around the numeric line spectrum.
    pub header_lines: Vec<String>,
    /// Energy axis in eV.
    pub energy_ev: Array1<f64>,
    /// Spectrum channel columns for each row.
    pub channels: Array2<f64>,
}

/// FEFF RIXS text outputs produced from the final spectrum assembly block.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsDatFromFinalSpectrum {
    /// Renderable `rixsET.dat` transfer/incident-energy map.
    pub rixs_et: RixsMapData,
    /// Renderable `herfd.dat` diagonal line spectrum.
    pub herfd: RixsLineData,
    /// Renderable `xasEI.dat` incident-energy XAS line spectrum.
    pub xas_ei: RixsLineData,
    /// Renderable `xasEF.dat` final-energy XAS line spectrum.
    pub xas_ef: RixsLineData,
    /// Renderable `rixsEE.dat` incident/emission-energy map.
    pub rixs_ee: RixsMapData,
}

/// FEFF `ReadPoles` edge offsets normalized from `edges.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RixsSkipCalcEdgeOffsets {
    /// Incident edge in Hartree.
    pub incident_edge_hartree: f64,
    /// Final edge in Hartree.
    pub final_edge_hartree: f64,
}

impl RixsLineData {
    /// Number of line-spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Number of channel columns after the energy axis.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.ncols()
    }
}

/// Normalize FEFF RIXS `edges.dat` rows for cached `SkipCalc` output assembly.
pub fn rixs_skip_calc_edge_offsets_from_edges_dat(
    edges: &EdgesDatData,
    chemical_potential_hartree: f64,
) -> Result<RixsSkipCalcEdgeOffsets> {
    let pole_energies = edges
        .rows
        .iter()
        .map(|row| row.chemical_potential)
        .collect::<Array1<_>>();
    let pole_amplitudes = edges
        .rows
        .iter()
        .map(|row| row.matrix_element)
        .collect::<Array1<_>>();
    let pole_widths = edges
        .rows
        .iter()
        .map(|row| row.core_hole_width)
        .collect::<Array1<_>>();
    let normalized = rixs_normalize_poles(RixsPoleNormalizationInput {
        pole_energies: pole_energies.view(),
        pole_amplitudes: pole_amplitudes.view(),
        pole_widths: pole_widths.view(),
        fermi_level: chemical_potential_hartree,
    })
    .map_err(|source| invalid_rixs_dat("edges.dat", source.to_string()))?;

    Ok(RixsSkipCalcEdgeOffsets {
        incident_edge_hartree: normalized.incident_edge,
        final_edge_hartree: normalized.final_edge,
    })
}

/// Build FEFF RIXS `SkipCalc` xas/HERFD/map payloads from cached `rixsET.dat`.
pub fn rixs_skip_calc_outputs_from_rixs_et_map(
    map: &RixsMapData,
    energy_window: RixsEnergyWindow,
    edges: RixsSkipCalcEdgeOffsets,
) -> Result<RixsDatFromFinalSpectrum> {
    validate_finite("RIXS SkipCalc EMinI", energy_window.emin_i)?;
    validate_finite("RIXS SkipCalc EMaxI", energy_window.emax_i)?;
    validate_finite("RIXS SkipCalc EMinF", energy_window.emin_f)?;
    validate_finite("RIXS SkipCalc EMaxF", energy_window.emax_f)?;

    let order = rixs_skip_calc_square_order(map)?;
    let grid = rixs_skip_calc_energy_grid(map, order, edges)?;
    let cross_section = rixs_skip_calc_cross_section_table(map, order)?;
    let spectrum = rixs_final_spectrum(RixsFinalSpectrumInput {
        relative_energies: grid.relative_hartree.view(),
        cross_section: cross_section.view(),
        incident_window: (energy_window.emin_i, energy_window.emax_i),
        final_window: (energy_window.emin_f, energy_window.emax_f),
        incident_edge: edges.incident_edge_hartree,
        final_edge: edges.final_edge_hartree,
        hartree_ev: FEFF_HARTREE_EV,
    })
    .map_err(|source| invalid_rixs_dat("SkipCalc final spectrum", source.to_string()))?;
    rixs_dat_from_final_spectrum(&spectrum, order)
}

/// Build FEFF RIXS `SkipCalc` MBConv satellite payloads from cached inputs.
pub fn rixs_skip_calc_satellite_outputs(
    map: &RixsMapData,
    energy_window: RixsEnergyWindow,
    edges: RixsSkipCalcEdgeOffsets,
    chemical_potential_hartree: f64,
    xes: &XmuDatData,
) -> Result<RixsDatFromFinalSpectrum> {
    let order = rixs_skip_calc_square_order(map)?;
    let grid = rixs_skip_calc_energy_grid(map, order, edges)?;
    let cross_section = rixs_skip_calc_cross_section_table(map, order)?;
    let satellite = rixs_satellite_spectrum(RixsSatelliteSpectrumInput {
        relative_energies: grid.relative_hartree.view(),
        cross_section: cross_section.view(),
        xes_energy_ev: xes.relative_energy_ev.view(),
        xes_mu: xes.mu.view(),
        fermi_level: chemical_potential_hartree,
        incident_window: (energy_window.emin_i, energy_window.emax_i),
        final_window: (energy_window.emin_f, energy_window.emax_f),
        incident_edge: edges.incident_edge_hartree,
        final_edge: edges.final_edge_hartree,
        hartree_ev: FEFF_HARTREE_EV,
    })
    .map_err(|source| invalid_rixs_dat("SkipCalc satellite spectrum", source.to_string()))?;
    rixs_dat_from_final_spectrum(&satellite.spectrum, order)
}

/// Extract FEFF RIXS `SkipCalc` HERFD from the diagonal of cached `rixsET.dat`.
pub fn rixs_skip_calc_herfd_from_rixs_et_map(map: &RixsMapData) -> Result<RixsLineData> {
    rixs_line_from_map_diagonal(map)
}

/// Extract a FEFF one-axis line spectrum from the diagonal of a square RIXS map.
///
/// FEFF writes `herfd.dat` from the diagonal of `rixsET.dat`, and writes the
/// intermediate `xas0.dat`/`xas1.dat` diagnostics from the diagonals of
/// `rixs0.dat`/`rixs1.dat`. This adapter keeps that deterministic output
/// step available when the map cache exists but the matching line cache is
/// missing.
pub fn rixs_line_from_map_diagonal(map: &RixsMapData) -> Result<RixsLineData> {
    let order = rixs_square_map_order(map, "RIXS map diagonal extraction")?;
    let mut energy_ev = Vec::with_capacity(order);
    let mut channels = Vec::with_capacity(order * map.channel_count());
    for diagonal in 0..order {
        let row = diagonal * order + diagonal;
        energy_ev.push(map.first_energy_ev[row]);
        channels.extend(map.channels.row(row).iter().copied());
    }

    let channels = Array2::from_shape_vec((order, map.channel_count()), channels)
        .map_err(|_| invalid_rixs_dat("herfd.dat channels", "channel table shape mismatch"))?;
    Ok(RixsLineData {
        header_lines: Vec::new(),
        energy_ev: Array1::from_vec(energy_ev),
        channels,
    })
}

/// Build FEFF RIXS text-table payloads from a core final-spectrum result.
///
/// This ports the last text-output handoff in `RIXS/rixs.f90` after
/// `xsect_tmp` has been broadened and projected into [`RixsFinalSpectrum`].
/// `map_order` is the FEFF square `ne1` order used for `rixsET.dat` and
/// `rixsEE.dat` blank-line block layout.
pub fn rixs_dat_from_final_spectrum(
    spectrum: &RixsFinalSpectrum,
    map_order: usize,
) -> Result<RixsDatFromFinalSpectrum> {
    if map_order == 0 {
        return Err(invalid_rixs_dat(
            "map_order",
            "map_order must be at least 1",
        ));
    }
    let block_lengths = vec![map_order; map_order];
    let rixs_et = RixsMapData {
        header_lines: Vec::new(),
        block_lengths: block_lengths.clone(),
        first_energy_ev: spectrum.rixs_et_first_energy_ev.clone(),
        second_energy_ev: spectrum.rixs_et_second_energy_ev.clone(),
        channels: spectrum.rixs_et.clone(),
    };
    validate_rixs_map(&rixs_et)?;

    let herfd = RixsLineData {
        header_lines: Vec::new(),
        energy_ev: spectrum.herfd_energy_ev.clone(),
        channels: spectrum.herfd.clone(),
    };
    validate_rixs_line(&herfd)?;

    let xas_ei = RixsLineData {
        header_lines: Vec::new(),
        energy_ev: spectrum.incident_xas_energy_ev.clone(),
        channels: spectrum.incident_xas.clone(),
    };
    validate_rixs_line(&xas_ei)?;

    let xas_ef = RixsLineData {
        header_lines: Vec::new(),
        energy_ev: spectrum.final_xas_energy_ev.clone(),
        channels: spectrum.final_xas.clone(),
    };
    validate_rixs_line(&xas_ef)?;

    let rixs_ee = RixsMapData {
        header_lines: Vec::new(),
        block_lengths,
        first_energy_ev: spectrum.rixs_ee_incident_energy_ev.clone(),
        second_energy_ev: spectrum.rixs_ee_emission_energy_ev.clone(),
        channels: spectrum.rixs_ee.clone(),
    };
    validate_rixs_map(&rixs_ee)?;

    Ok(RixsDatFromFinalSpectrum {
        rixs_et,
        herfd,
        xas_ei,
        xas_ef,
        rixs_ee,
    })
}

/// Render FEFF-compatible two-axis RIXS map text.
pub fn rixs_map_string(data: &RixsMapData) -> Result<String> {
    validate_rixs_map(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    let block_lengths = normalized_block_lengths(data.block_lengths.as_slice(), data.point_count());
    let mut row_index = 0_usize;
    for block_len in block_lengths {
        for _ in 0..block_len {
            write_fortran_zero_scaled_exp(&mut out, data.first_energy_ev[row_index], 30, 15)?;
            write_fortran_zero_scaled_exp(&mut out, data.second_energy_ev[row_index], 30, 15)?;
            for value in data.channels.index_axis(Axis(0), row_index) {
                write_fortran_zero_scaled_exp(&mut out, *value, 30, 15)?;
            }
            out.push('\n');
            row_index += 1;
        }
        writeln!(out, " ")?;
    }
    Ok(out)
}

/// Parse FEFF two-axis RIXS map text.
pub fn parse_rixs_map(text: &str) -> Result<RixsMapData> {
    let mut header_lines = Vec::new();
    let mut block_lengths = Vec::new();
    let mut current_block_len = 0_usize;
    let mut row_width = None;
    let mut first_energy_ev = Vec::new();
    let mut second_energy_ev = Vec::new();
    let mut channels = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            if current_block_len > 0 {
                block_lengths.push(current_block_len);
                current_block_len = 0;
            }
            continue;
        }

        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if width < RIXS_MAP_MIN_ROW_WIDTH {
                return Err(IoError::RixsDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: RIXS_MAP_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::RixsDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: "same width as previous numeric rows",
                    });
                }
            } else {
                row_width = Some(width);
            }

            first_energy_ev.push(parse_f64(line_number, "first energy", tokens[0])?);
            second_energy_ev.push(parse_f64(line_number, "second energy", tokens[1])?);
            for token in &tokens[2..] {
                channels.push(parse_f64(line_number, "channel", token)?);
            }
            current_block_len += 1;
        } else {
            header_lines.push(raw.to_string());
        }
    }

    if current_block_len > 0 {
        block_lengths.push(current_block_len);
    }

    let point_count = first_energy_ev.len();
    let channel_count = row_width.map_or(0, |width| width - 2);
    let channels = if channel_count == 0 {
        Array2::zeros((0, 0))
    } else {
        Array2::from_shape_vec((point_count, channel_count), channels)
            .map_err(|_| invalid_rixs_dat("channels", "channel payload did not match map shape"))?
    };

    let data = RixsMapData {
        header_lines,
        block_lengths,
        first_energy_ev: Array1::from_vec(first_energy_ev),
        second_energy_ev: Array1::from_vec(second_energy_ev),
        channels,
    };
    validate_rixs_map(&data)?;
    Ok(data)
}

/// Write FEFF two-axis RIXS map text to a file.
pub fn write_rixs_map(path: impl AsRef<Path>, data: &RixsMapData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rixs_map_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF two-axis RIXS map text from a file.
pub fn read_rixs_map(path: impl AsRef<Path>) -> Result<RixsMapData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rixs_map(&text)
}

/// Render FEFF-compatible one-axis RIXS line spectrum text.
pub fn rixs_line_string(data: &RixsLineData) -> Result<String> {
    validate_rixs_line(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, row) in data.energy_ev.iter().zip(data.channels.axis_iter(Axis(0))) {
        write_fortran_zero_scaled_exp(&mut out, *energy, 30, 15)?;
        for value in row {
            write_fortran_zero_scaled_exp(&mut out, *value, 30, 15)?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Parse FEFF one-axis RIXS line spectrum text.
pub fn parse_rixs_line(text: &str) -> Result<RixsLineData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut channels = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }

        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if width < RIXS_LINE_MIN_ROW_WIDTH {
                return Err(IoError::RixsDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: RIXS_LINE_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::RixsDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: "same width as previous numeric rows",
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            for token in &tokens[1..] {
                channels.push(parse_f64(line_number, "channel", token)?);
            }
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let point_count = energy_ev.len();
    let channel_count = row_width.map_or(0, |width| width - 1);
    let channels = if channel_count == 0 {
        Array2::zeros((0, 0))
    } else {
        Array2::from_shape_vec((point_count, channel_count), channels)
            .map_err(|_| invalid_rixs_dat("channels", "channel payload did not match line shape"))?
    };

    let data = RixsLineData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        channels,
    };
    validate_rixs_line(&data)?;
    Ok(data)
}

/// Write FEFF one-axis RIXS line spectrum text to a file.
pub fn write_rixs_line(path: impl AsRef<Path>, data: &RixsLineData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rixs_line_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF one-axis RIXS line spectrum text from a file.
pub fn read_rixs_line(path: impl AsRef<Path>) -> Result<RixsLineData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rixs_line(&text)
}

#[derive(Debug, Clone, PartialEq)]
struct RixsSkipCalcEnergyGrid {
    relative_hartree: Array1<f64>,
}

fn rixs_skip_calc_energy_grid(
    map: &RixsMapData,
    order: usize,
    edges: RixsSkipCalcEdgeOffsets,
) -> Result<RixsSkipCalcEnergyGrid> {
    validate_finite("RIXS SkipCalc incident edge", edges.incident_edge_hartree)?;
    validate_finite("RIXS SkipCalc final edge", edges.final_edge_hartree)?;

    let incident_energy_ev = Array1::from_iter(map.first_energy_ev.iter().take(order).copied());
    let relative_hartree = incident_energy_ev
        .iter()
        .map(|energy| {
            validate_finite("RIXS SkipCalc incident energy", *energy)?;
            Ok(*energy / FEFF_HARTREE_EV - edges.incident_edge_hartree)
        })
        .collect::<Result<Array1<_>>>()?;
    validate_increasing_grid("RIXS SkipCalc relative energy", relative_hartree.view())?;

    Ok(RixsSkipCalcEnergyGrid { relative_hartree })
}

fn rixs_skip_calc_channel_table(
    map: &RixsMapData,
    order: usize,
    channel: usize,
) -> Result<Array2<f64>> {
    if channel >= map.channel_count() {
        return Err(invalid_rixs_dat(
            "channel",
            format!(
                "requested channel {channel}, but rixsET.dat has {} channel(s)",
                map.channel_count()
            ),
        ));
    }
    Ok(Array2::from_shape_fn((order, order), |(row, col)| {
        map.channels[(row * order + col, channel)]
    }))
}

fn rixs_skip_calc_cross_section_table(map: &RixsMapData, order: usize) -> Result<Array3<f64>> {
    let channel_count = map.channel_count();
    let mut cross_section = Array3::<f64>::zeros((order, order, channel_count));
    for channel in 0..channel_count {
        let table = rixs_skip_calc_channel_table(map, order, channel)?;
        for row in 0..order {
            for col in 0..order {
                cross_section[(row, col, channel)] = table[(row, col)];
            }
        }
    }
    Ok(cross_section)
}

fn rixs_skip_calc_square_order(map: &RixsMapData) -> Result<usize> {
    rixs_square_map_order(map, "RIXS SkipCalc rixsET.dat")
}

fn rixs_square_map_order(map: &RixsMapData, context: &'static str) -> Result<usize> {
    let point_count = map.point_count();
    if point_count == 0 {
        return Err(invalid_rixs_dat(
            "rows",
            format!("{context} requires at least one map row"),
        ));
    }
    let order = if map.block_lengths.len() > 1 {
        let order = map.block_lengths.len();
        if map
            .block_lengths
            .iter()
            .any(|&block_len| block_len != order)
        {
            return Err(invalid_rixs_dat(
                "block lengths",
                format!(
                    "{context} blocks must form a square map, got block lengths {:?}",
                    map.block_lengths
                ),
            ));
        }
        order
    } else {
        square_order_from_point_count(point_count)?
    };

    if order
        .checked_mul(order)
        .is_none_or(|expected| expected != point_count)
    {
        return Err(invalid_rixs_dat(
            "rows",
            format!("{context} has {point_count} row(s), expected {order}x{order}"),
        ));
    }
    Ok(order)
}

fn square_order_from_point_count(point_count: usize) -> Result<usize> {
    let mut order = 1_usize;
    while order
        .checked_mul(order)
        .is_some_and(|square| square < point_count)
    {
        order += 1;
    }
    if order
        .checked_mul(order)
        .is_some_and(|square| square == point_count)
    {
        Ok(order)
    } else {
        Err(invalid_rixs_dat(
            "rows",
            format!("RIXS SkipCalc rixsET.dat row count {point_count} is not a square"),
        ))
    }
}

fn validate_rixs_map(data: &RixsMapData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_rixs_dat(
            "rows",
            "at least one RIXS map row is required",
        ));
    }
    validate_len("second_energy_ev", data.second_energy_ev.len(), point_count)?;
    validate_channels("channels", data.channels.dim(), point_count)?;
    validate_block_lengths(data.block_lengths.as_slice(), point_count)?;

    for (row, (first, second)) in data
        .first_energy_ev
        .iter()
        .zip(data.second_energy_ev.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite_row("first energy", *first, row)?;
        validate_finite_row("second energy", *second, row)?;
    }
    validate_channel_values(&data.channels)
}

fn validate_rixs_line(data: &RixsLineData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_rixs_dat(
            "rows",
            "at least one RIXS line row is required",
        ));
    }
    validate_channels("channels", data.channels.dim(), point_count)?;

    for (row, energy) in data.energy_ev.iter().enumerate() {
        validate_finite_row("energy", *energy, row + 1)?;
    }
    validate_channel_values(&data.channels)
}

fn validate_finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_rixs_dat(
            name,
            format!("must be finite, got {value}"),
        ))
    }
}

fn validate_increasing_grid(
    name: &'static str,
    values: ndarray::ArrayView1<'_, f64>,
) -> Result<()> {
    if values.len() < 2 {
        return Err(invalid_rixs_dat(
            name,
            format!("requires at least two points, got {}", values.len()),
        ));
    }
    let mut previous = values[0];
    validate_finite(name, previous)?;
    for (index, &value) in values.iter().enumerate().skip(1) {
        validate_finite(name, value)?;
        if value <= previous {
            return Err(invalid_rixs_dat(
                name,
                format!("must increase at index {index}: previous={previous}, current={value}"),
            ));
        }
        previous = value;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::RixsDatShape {
            field,
            rows: actual,
            cols: 1,
            expected_rows: expected,
            expected_cols: "1",
        })
    }
}

fn validate_channels(
    field: &'static str,
    (rows, cols): (usize, usize),
    expected_rows: usize,
) -> Result<()> {
    if rows == expected_rows && cols > 0 {
        Ok(())
    } else {
        Err(IoError::RixsDatShape {
            field,
            rows,
            cols,
            expected_rows,
            expected_cols: RIXS_CHANNEL_COLUMNS,
        })
    }
}

fn validate_block_lengths(block_lengths: &[usize], point_count: usize) -> Result<()> {
    if block_lengths.is_empty() {
        return Ok(());
    }
    let total = block_lengths.iter().try_fold(0_usize, |total, block| {
        if *block == 0 {
            Err(invalid_rixs_dat(
                "block lengths",
                "block length must be positive",
            ))
        } else {
            total
                .checked_add(*block)
                .ok_or_else(|| invalid_rixs_dat("block lengths", "block length sum overflows"))
        }
    })?;
    if total == point_count {
        Ok(())
    } else {
        Err(invalid_rixs_dat(
            "block lengths",
            format!("block length sum {total} does not match row count {point_count}"),
        ))
    }
}

fn validate_channel_values(channels: &Array2<f64>) -> Result<()> {
    let cols = channels.ncols();
    for (index, value) in channels.iter().enumerate() {
        let row = index / cols + 1;
        validate_finite_row("channel", *value, row)?;
    }
    Ok(())
}

fn normalized_block_lengths(block_lengths: &[usize], point_count: usize) -> Vec<usize> {
    if block_lengths.is_empty() {
        vec![point_count]
    } else {
        block_lengths.to_vec()
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::RixsDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidRixsDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_rixs_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidRixsDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rixs_map_blocks_and_channels() -> Result<()> {
        let data = parse_rixs_map(RIXS_MAP)?;
        assert_eq!(data.point_count(), 4);
        assert_eq!(data.channel_count(), 4);
        assert_eq!(data.block_lengths, vec![2, 2]);
        assert_eq!(data.first_energy_ev[0], 1.15408611465090e4);
        assert_eq!(data.second_energy_ev[0], -15.0000004124133);
        assert_eq!(data.channels[[0, 0]], 1.05265355183925e-6);
        assert_eq!(data.channels[[3, 3]], 3.0e-3);
        Ok(())
    }

    #[test]
    fn parses_rixs_line_reference_shape() -> Result<()> {
        let data = parse_rixs_line(RIXS_LINE)?;
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.channel_count(), 4);
        assert_eq!(data.energy_ev[0], 1.15408611465090e4);
        assert_eq!(data.channels[[1, 0]], 1.14359805938722e-6);
        assert_eq!(data.channels[[2, 3]], 0.0);
        Ok(())
    }

    #[test]
    fn roundtrips_rixs_map_and_line_text() -> Result<()> {
        let map = parse_rixs_map(RIXS_MAP)?;
        assert_eq!(parse_rixs_map(&rixs_map_string(&map)?)?, map);

        let line = parse_rixs_line(RIXS_LINE)?;
        assert_eq!(parse_rixs_line(&rixs_line_string(&line)?)?, line);
        Ok(())
    }

    #[test]
    fn builds_rixs_tables_from_final_spectrum() -> Result<()> {
        let spectrum = sample_final_spectrum();

        let tables = rixs_dat_from_final_spectrum(&spectrum, 2)?;

        assert_eq!(tables.rixs_et.block_lengths, vec![2, 2]);
        assert_eq!(tables.rixs_et.first_energy_ev[0], 10.0);
        assert_eq!(tables.rixs_et.second_energy_ev[3], 103.0);
        assert_eq!(tables.rixs_et.channels[(2, 1)], 0.6);
        assert_eq!(tables.herfd.energy_ev[1], 21.0);
        assert_eq!(tables.herfd.channels[(1, 1)], 1.4);
        assert_eq!(tables.xas_ei.energy_ev[0], 30.0);
        assert_eq!(tables.xas_ef.channels[(1, 0)], 3.3);
        assert_eq!(tables.rixs_ee.block_lengths, vec![2, 2]);
        assert_eq!(tables.rixs_ee.first_energy_ev[3], 63.0);
        assert_eq!(tables.rixs_ee.second_energy_ev[0], 70.0);

        let rendered_map = rixs_map_string(&tables.rixs_et)?;
        assert_eq!(parse_rixs_map(&rendered_map)?, tables.rixs_et);
        let rendered_line = rixs_line_string(&tables.xas_ei)?;
        assert_eq!(parse_rixs_line(&rendered_line)?, tables.xas_ei);
        Ok(())
    }

    #[test]
    fn skip_calc_edge_offsets_from_edges_dat_match_readpoles_reference() -> Result<()> {
        let offsets =
            rixs_skip_calc_edge_offsets_from_edges_dat(&sample_skip_calc_edges_dat(), 0.0)?;

        assert_close(offsets.incident_edge_hartree, 10.0);
        assert_close(offsets.final_edge_hartree, 4.0);
        Ok(())
    }

    #[test]
    fn skip_calc_herfd_from_rixs_et_map_extracts_diagonal() -> Result<()> {
        let herfd = rixs_skip_calc_herfd_from_rixs_et_map(&sample_skip_calc_rixs_map())?;

        assert_line_close(
            &herfd,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        Ok(())
    }

    #[test]
    fn line_from_map_diagonal_extracts_square_map_diagonal() -> Result<()> {
        let line = rixs_line_from_map_diagonal(&sample_skip_calc_rixs_map())?;

        assert_line_close(
            &line,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[1.0, 101.0], [5.0, 105.0], [9.0, 109.0]],
        );
        Ok(())
    }

    #[test]
    fn skip_calc_outputs_from_rixs_et_map_builds_xas_and_emission_tables() -> Result<()> {
        let edges = rixs_skip_calc_edge_offsets_from_edges_dat(&sample_skip_calc_edges_dat(), 0.0)?;
        let tables = rixs_skip_calc_outputs_from_rixs_et_map(
            &sample_skip_calc_rixs_map(),
            sample_skip_calc_window(),
            edges,
        )?;

        assert_line_close(
            &tables.xas_ei,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[8.0, 208.0], [10.0, 210.0], [12.0, 212.0]],
        );
        assert_line_close(
            &tables.xas_ef,
            &[6.0, 7.0, 8.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[4.0, 204.0], [10.0, 210.0], [16.0, 216.0]],
        );
        assert_eq!(tables.rixs_ee.block_lengths, vec![3, 3, 3]);
        assert_matrix_close(
            &tables.rixs_ee.channels,
            &[
                [3.0, 103.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 101.0],
                [5.0, 105.0],
                [9.0, 109.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7.0, 107.0],
            ],
        );
        Ok(())
    }

    #[test]
    fn skip_calc_satellite_outputs_use_xes_convolution() -> Result<()> {
        let edges = rixs_skip_calc_edge_offsets_from_edges_dat(&sample_skip_calc_edges_dat(), 0.0)?;
        let tables = rixs_skip_calc_satellite_outputs(
            &sample_skip_calc_rixs_map(),
            sample_skip_calc_window(),
            edges,
            0.0,
            &sample_xes_xmu_dat(),
        )?;

        assert_line_close(
            &tables.herfd,
            &[10.0, 11.0, 12.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[0.5, 50.5], [3.5, 103.5], [7.5, 107.5]],
        );
        assert_line_close(
            &tables.xas_ei,
            &[0.0, 1.0, 2.0].map(|energy| energy * FEFF_HARTREE_EV),
            &[[5.5, 180.5], [7.25, 182.25], [9.0, 184.0]],
        );
        assert_matrix_close(
            &tables.rixs_ee.channels,
            &[
                [1.5, 51.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.5, 50.5],
                [3.5, 103.5],
                [7.5, 107.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [5.5, 105.5],
            ],
        );
        Ok(())
    }

    #[test]
    fn rixs_tables_from_final_spectrum_rejects_wrong_map_order() {
        let spectrum = sample_final_spectrum();

        let error = rixs_dat_from_final_spectrum(&spectrum, 3).unwrap_err();

        assert!(error.to_string().contains("block length sum"));
    }

    #[test]
    fn rejects_bad_rixs_inputs() {
        assert!(parse_rixs_map("# no rows\n").is_err());
        assert!(parse_rixs_map("1 2\n").is_err());
        assert!(parse_rixs_map("1 2 3\n4 5 6 7\n").is_err());
        assert!(parse_rixs_map("1 2 NaN\n").is_err());
        assert!(parse_rixs_line("1\n").is_err());
        assert!(parse_rixs_line("1 2\n3 4 5\n").is_err());
        assert!(parse_rixs_line("1 NaN\n").is_err());

        let bad_map = RixsMapData {
            header_lines: Vec::new(),
            block_lengths: vec![3],
            first_energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            second_energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            channels: Array2::zeros((2, 1)),
        };
        assert!(rixs_map_string(&bad_map).is_err());
    }

    fn sample_final_spectrum() -> RixsFinalSpectrum {
        RixsFinalSpectrum {
            incident_xas_energy_ev: Array1::from_vec(vec![30.0, 31.0]),
            incident_xas: Array2::from_shape_vec((2, 2), vec![2.0, 2.1, 2.2, 2.3]).unwrap(),
            final_xas_energy_ev: Array1::from_vec(vec![40.0, 41.0]),
            final_xas: Array2::from_shape_vec((2, 2), vec![3.0, 3.1, 3.3, 3.4]).unwrap(),
            herfd_energy_ev: Array1::from_vec(vec![20.0, 21.0]),
            herfd: Array2::from_shape_vec((2, 2), vec![1.0, 1.1, 1.3, 1.4]).unwrap(),
            rixs_et_first_energy_ev: Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0]),
            rixs_et_second_energy_ev: Array1::from_vec(vec![100.0, 101.0, 102.0, 103.0]),
            rixs_et: Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap(),
            rixs_ee_incident_energy_ev: Array1::from_vec(vec![60.0, 61.0, 62.0, 63.0]),
            rixs_ee_emission_energy_ev: Array1::from_vec(vec![70.0, 71.0, 72.0, 73.0]),
            rixs_ee: Array2::from_shape_vec((4, 2), vec![4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8])
                .unwrap(),
        }
    }

    fn sample_skip_calc_window() -> RixsEnergyWindow {
        RixsEnergyWindow {
            emin_i: -0.5,
            emax_i: 2.0,
            emin_f: -0.5,
            emax_f: 2.0,
        }
    }

    fn sample_skip_calc_edges_dat() -> EdgesDatData {
        EdgesDatData {
            header_lines: vec!["# emu, M_kk, gam".to_string()],
            rows: vec![
                crate::energy_output::EdgesDatRow {
                    chemical_potential: 10.0,
                    matrix_element: 1.0,
                    core_hole_width: 0.1,
                },
                crate::energy_output::EdgesDatRow {
                    chemical_potential: 4.0,
                    matrix_element: 1.0,
                    core_hole_width: 0.1,
                },
            ],
        }
    }

    fn sample_skip_calc_rixs_map() -> RixsMapData {
        let mut first_energy_ev = Vec::new();
        let mut second_energy_ev = Vec::new();
        let mut channels = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                let value = (row * 3 + col + 1) as f64;
                first_energy_ev.push((10.0 + col as f64) * FEFF_HARTREE_EV);
                second_energy_ev.push((4.0 + row as f64) * FEFF_HARTREE_EV);
                channels.push(value);
                channels.push(value + 100.0);
            }
        }

        RixsMapData {
            header_lines: Vec::new(),
            block_lengths: vec![3, 3, 3],
            first_energy_ev: Array1::from_vec(first_energy_ev),
            second_energy_ev: Array1::from_vec(second_energy_ev),
            channels: Array2::from_shape_vec((9, 2), channels).expect("valid channel shape"),
        }
    }

    fn sample_xes_xmu_dat() -> XmuDatData {
        XmuDatData {
            header_lines: Vec::new(),
            normalization: None,
            photon_energy_ev: Array1::from_vec(vec![0.0, 1.0]),
            relative_energy_ev: Array1::from_vec(vec![-FEFF_HARTREE_EV, 0.0]),
            wave_number: Array1::from_vec(vec![0.0, 0.0]),
            mu: Array1::from_vec(vec![1.0, 1.0]),
            mu0: Array1::from_vec(vec![0.0, 0.0]),
            chi: Array1::from_vec(vec![1.0, 1.0]),
        }
    }

    fn assert_line_close(data: &RixsLineData, expected_energy: &[f64], expected: &[[f64; 2]]) {
        assert_vector_close(data.energy_ev.as_slice().unwrap(), expected_energy);
        assert_matrix_close(&data.channels, expected);
    }

    fn assert_matrix_close<const N: usize>(actual: &Array2<f64>, expected: &[[f64; N]]) {
        assert_eq!(actual.dim(), (expected.len(), N));
        for (row, expected_row) in expected.iter().enumerate() {
            for (col, &expected_value) in expected_row.iter().enumerate() {
                assert_close(actual[(row, col)], expected_value);
            }
        }
    }

    fn assert_vector_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual_value, &expected_value) in actual.iter().zip(expected.iter()) {
            assert_close(actual_value, expected_value);
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    const RIXS_MAP: &str = r#"         0.115408611465090E+05        -0.150000004124133E+02         0.105265355183925E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115418611463549E+05        -0.150000004124133E+02         0.100255925206130E-05         0.100000000000000E-02         0.200000000000000E-02         0.300000000000000E-02

         0.115408611465090E+05        -0.140000004124133E+02         0.200000000000000E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115418611463549E+05        -0.140000004124133E+02         0.300000000000000E-05         0.100000000000000E-02         0.200000000000000E-02         0.300000000000000E-02
"#;

    const RIXS_LINE: &str = r#"         0.115408611465090E+05         0.105265355183925E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115418611463549E+05         0.114359805938722E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115428611462008E+05         0.125101882959870E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
"#;
}
