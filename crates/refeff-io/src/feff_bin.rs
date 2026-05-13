//! FEFF `feff.bin` text/PAD path-data codec.
//!
//! `GENFMT/genfmtsub.f90` writes this printable handoff file and FF2X reads it
//! via `FF2X/rdfbin.f90`. The format uses tagged text records for metadata and
//! Packed ASCII Data (PAD) blocks for shared energy arrays and per-path data.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::exp;
use crate::pad::{decode_f64, encode_complex, encode_reals};

/// FEFF's bohr-to-Angstrom conversion used when writing `reff` records.
pub const FEFF_BIN_BOHR: f64 = 0.529_177_249;
/// FEFF v03 `feff.bin` default PAD width.
pub const FEFF_BIN_DEFAULT_PAD_WIDTH: usize = 8;

/// Potential label and atomic number entry from the `#@` record.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffBinPotential {
    /// FEFF six-character potential label.
    pub label: String,
    /// Atomic number for this potential.
    pub atomic_number: usize,
}

/// One path block from a FEFF v03 `feff.bin` file.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffBinPath {
    /// FEFF path index, `ipath`.
    pub index: usize,
    /// Path degeneracy.
    pub degeneracy: f64,
    /// Effective half path length in bohr. The text file stores Angstrom.
    pub effective_half_path_length_bohr: f64,
    /// Path importance criterion.
    pub criterion: f64,
    /// Potential index for each leg.
    pub potential_indices: Array1<usize>,
    /// Cartesian leg positions as `(leg, xyz)` in bohr.
    pub positions: Array2<f64>,
    /// First Euler angle for each leg.
    pub beta: Array1<f64>,
    /// Second Euler angle for each leg.
    pub eta: Array1<f64>,
    /// Leg distances in bohr.
    pub leg_distances: Array1<f64>,
    /// FEFF amplitude array, `amff`.
    pub amplitude: Array1<f64>,
    /// FEFF phase array, `phff`.
    pub phase: Array1<f64>,
}

impl FeffBinPath {
    /// Number of legs in this path.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.potential_indices.len()
    }
}

/// FEFF v03 `feff.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffBinData {
    /// Version suffix written after `#_feff.bin v03:`.
    pub version: String,
    /// PAD field width, `mpadx`.
    pub pad_width: usize,
    /// Core-hole index, `ihole`.
    pub ihole: i32,
    /// GENFMT matrix order, `iorder`.
    pub order: i32,
    /// Initial-state angular momentum, `ilinit`.
    pub initial_angular_momentum: i32,
    /// Average Norman radius, `rnrmav`.
    pub average_norman_radius: f64,
    /// Fermi level, `xmu`.
    pub fermi_level: f64,
    /// Edge energy.
    pub edge_energy: f64,
    /// Potential table for FEFF indices `0:npot`.
    pub potentials: Vec<FeffBinPotential>,
    /// Central atom phase shift, `phc`.
    pub central_phase_shift: Array1<Complex64>,
    /// Complex momentum, `ck`.
    pub complex_momentum: Array1<Complex64>,
    /// Real momentum, `xk`.
    pub real_momentum: Array1<f64>,
    /// Path records.
    pub paths: Vec<FeffBinPath>,
    /// Raw parsed `feff.bin` text for exact re-emission when the typed content
    /// is unchanged.
    pub raw_text: Option<String>,
}

impl FeffBinData {
    /// Number of energy points, `ne`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.central_phase_shift.len()
    }

    /// Number of potential entries represented by FEFF indices `0:npot`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potentials.len()
    }
}

/// Render FEFF v03 `feff.bin` text.
pub fn feff_bin_string(data: &FeffBinData) -> Result<String> {
    validate_feff_bin(data)?;

    if let Some(raw_text) = &data.raw_text
        && raw_feff_bin_matches(data, raw_text)?
    {
        return Ok(raw_text.clone());
    }

    let mut out = String::new();
    writeln!(out, "#_feff.bin v03: {}", data.version.trim_end())?;
    writeln!(
        out,
        "#_{:>5}{:>5}{:>5}",
        data.potential_count() - 1,
        data.energy_count(),
        data.pad_width
    )?;
    writeln!(
        out,
        "#={:>8}{:>8}{:>8} {} {} {}",
        data.ihole,
        data.order,
        data.initial_angular_momentum,
        exp(data.average_norman_radius, 14, 7),
        exp(data.fermi_level, 14, 7),
        exp(data.edge_energy, 14, 7)
    )?;
    write_potential_line(&mut out, &data.potentials)?;
    write_complex_pad(
        &mut out,
        &data.central_phase_shift.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_complex_pad(
        &mut out,
        &data.complex_momentum.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_real_pad(
        &mut out,
        &data.real_momentum.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;

    for path in &data.paths {
        write_path_header(&mut out, path)?;
        write_real_pad(
            &mut out,
            &flatten_positions(path.positions.view()),
            data.pad_width,
        )?;
        write_real_pad(
            &mut out,
            &path.beta.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?;
        write_real_pad(
            &mut out,
            &path.eta.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?;
        write_real_pad(
            &mut out,
            &path.leg_distances.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?;
        write_real_pad(
            &mut out,
            &path.amplitude.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?;
        write_real_pad(
            &mut out,
            &path.phase.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?;
    }
    Ok(out)
}

/// Parse FEFF v03 `feff.bin` text.
pub fn parse_feff_bin(text: &str) -> Result<FeffBinData> {
    let mut lines = FeffBinLines::new(text);
    let version = lines.version()?;
    let counts = lines.counts()?;
    let potential_count = usize_from_i64(counts[0], "npot")?
        .checked_add(1)
        .ok_or_else(|| invalid_feff_bin("npot", "potential count overflowed"))?;
    let energy_count = usize_from_i64(counts[1], "ne")?;
    let pad_width = usize_from_i64(counts[2], "mpadx")?;

    let misc = lines.misc()?;
    let ihole = i32_from_i64(misc.0[0], "ihole")?;
    let order = i32_from_i64(misc.0[1], "iorder")?;
    let initial_angular_momentum = i32_from_i64(misc.0[2], "ilinit")?;
    let average_norman_radius = misc.1[0];
    let fermi_level = misc.1[1];
    let edge_energy = misc.1[2];
    let potentials = lines.potentials(potential_count)?;

    let central_phase_shift =
        Array1::from_vec(lines.pad_complex("phc", pad_width, energy_count)?);
    let complex_momentum = Array1::from_vec(lines.pad_complex("ck", pad_width, energy_count)?);
    let real_momentum = Array1::from_vec(lines.pad_reals("xk", pad_width, energy_count)?);

    let mut paths = Vec::new();
    while let Some(header) = lines.next_path_header()? {
        let leg_count = header.potential_indices.len();
        let positions = positions_from_values(
            lines.pad_reals("rat", pad_width, checked_count2("rat", leg_count, 3)?)?,
            leg_count,
        )?;
        let beta = Array1::from_vec(lines.pad_reals("beta", pad_width, leg_count)?);
        let eta = Array1::from_vec(lines.pad_reals("eta", pad_width, leg_count)?);
        let leg_distances = Array1::from_vec(lines.pad_reals("ri", pad_width, leg_count)?);
        let amplitude = Array1::from_vec(lines.pad_reals("achi", pad_width, energy_count)?);
        let phase = Array1::from_vec(lines.pad_reals("phchi", pad_width, energy_count)?);
        paths.push(FeffBinPath {
            index: header.index,
            degeneracy: header.degeneracy,
            effective_half_path_length_bohr: header.effective_half_path_length_bohr,
            criterion: header.criterion,
            potential_indices: Array1::from_vec(header.potential_indices),
            positions,
            beta,
            eta,
            leg_distances,
            amplitude,
            phase,
        });
    }

    let data = FeffBinData {
        version,
        pad_width,
        ihole,
        order,
        initial_angular_momentum,
        average_norman_radius,
        fermi_level,
        edge_energy,
        potentials,
        central_phase_shift,
        complex_momentum,
        real_momentum,
        paths,
        raw_text: Some(text.to_string()),
    };
    validate_feff_bin(&data)?;
    Ok(data)
}

/// Write FEFF v03 `feff.bin` text to a file.
pub fn write_feff_bin(path: impl AsRef<Path>, data: &FeffBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, feff_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF v03 `feff.bin` text from a file.
pub fn read_feff_bin(path: impl AsRef<Path>) -> Result<FeffBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_feff_bin(&text)
}

fn validate_feff_bin(data: &FeffBinData) -> Result<()> {
    if data.pad_width <= 2 {
        return Err(IoError::InvalidPadWidth(data.pad_width));
    }
    if data.energy_count() == 0 {
        return Err(invalid_feff_bin("ne", "at least one energy is required"));
    }
    if data.potential_count() == 0 {
        return Err(invalid_feff_bin(
            "npot",
            "at least one potential is required",
        ));
    }
    if data.version.trim().is_empty() || !data.version.is_ascii() {
        return Err(invalid_feff_bin(
            "version",
            "version must be non-empty ASCII text",
        ));
    }
    validate_len("ck", data.complex_momentum.len(), data.energy_count())?;
    validate_len("xk", data.real_momentum.len(), data.energy_count())?;
    validate_finite_complex("phc", data.central_phase_shift.iter().copied())?;
    validate_finite_complex("ck", data.complex_momentum.iter().copied())?;
    validate_finite_reals("xk", data.real_momentum.iter().copied())?;
    validate_finite_reals(
        "misc",
        [
            data.average_norman_radius,
            data.fermi_level,
            data.edge_energy,
        ],
    )?;

    for potential in &data.potentials {
        validate_label(&potential.label)?;
        check_fixed_int(i64_from_usize(potential.atomic_number, "iz")?, 3, "iz")?;
    }
    for path in &data.paths {
        validate_path(path, data.energy_count(), data.potential_count())?;
    }
    Ok(())
}

fn raw_feff_bin_matches(data: &FeffBinData, raw_text: &str) -> Result<bool> {
    let mut parsed = parse_feff_bin(raw_text)?;
    parsed.raw_text = None;
    let mut expected = data.clone();
    expected.raw_text = None;
    Ok(parsed == expected)
}

fn validate_path(path: &FeffBinPath, energy_count: usize, potential_count: usize) -> Result<()> {
    let leg_count = path.leg_count();
    if leg_count == 0 {
        return Err(invalid_feff_bin(
            "nleg",
            "at least one path leg is required",
        ));
    }
    check_fixed_int(i64_from_usize(path.index, "index")?, 6, "index")?;
    check_fixed_int(i64_from_usize(leg_count, "nleg")?, 3, "nleg")?;
    validate_shape2("rat", path.positions.dim(), (leg_count, 3))?;
    validate_len("beta", path.beta.len(), leg_count)?;
    validate_len("eta", path.eta.len(), leg_count)?;
    validate_len("ri", path.leg_distances.len(), leg_count)?;
    validate_len("achi", path.amplitude.len(), energy_count)?;
    validate_len("phchi", path.phase.len(), energy_count)?;
    validate_finite_reals(
        "path",
        [
            path.degeneracy,
            path.effective_half_path_length_bohr,
            path.criterion,
        ],
    )?;
    validate_finite_reals("rat", path.positions.iter().copied())?;
    validate_finite_reals("beta", path.beta.iter().copied())?;
    validate_finite_reals("eta", path.eta.iter().copied())?;
    validate_finite_reals("ri", path.leg_distances.iter().copied())?;
    validate_finite_reals("achi", path.amplitude.iter().copied())?;
    validate_finite_reals("phchi", path.phase.iter().copied())?;
    for &potential in &path.potential_indices {
        if potential >= potential_count {
            return Err(invalid_feff_bin(
                "ipot",
                format!("potential index {potential} is outside 0..{potential_count}"),
            ));
        }
        check_fixed_int(i64_from_usize(potential, "ipot")?, 2, "ipot")?;
    }
    Ok(())
}

fn write_potential_line(out: &mut String, potentials: &[FeffBinPotential]) -> Result<()> {
    out.push_str("#@");
    for potential in potentials {
        validate_label(&potential.label)?;
        write!(out, " {:<6}", potential.label)?;
    }
    for potential in potentials {
        check_fixed_int(i64_from_usize(potential.atomic_number, "iz")?, 3, "iz")?;
        write!(out, " {:>3}", potential.atomic_number)?;
    }
    out.push('\n');
    Ok(())
}

fn write_path_header(out: &mut String, path: &FeffBinPath) -> Result<()> {
    check_fixed_int(i64_from_usize(path.index, "index")?, 6, "index")?;
    check_fixed_int(i64_from_usize(path.leg_count(), "nleg")?, 3, "nleg")?;
    out.push_str("##");
    write!(
        out,
        "{:>6} {:>3} {:>7.3} {:>11.7} {}",
        path.index,
        path.leg_count(),
        path.degeneracy,
        path.effective_half_path_length_bohr * FEFF_BIN_BOHR,
        exp(path.criterion, 15, 4)
    )?;
    for &potential in &path.potential_indices {
        check_fixed_int(i64_from_usize(potential, "ipot")?, 2, "ipot")?;
        write!(out, " {:>2}", potential)?;
    }
    out.push('\n');
    Ok(())
}

fn write_real_pad(out: &mut String, values: &[f64], pad_width: usize) -> Result<()> {
    out.push_str(&encode_reals(values, pad_width)?);
    Ok(())
}

fn write_complex_pad(out: &mut String, values: &[Complex64], pad_width: usize) -> Result<()> {
    out.push_str(&encode_complex(values, pad_width)?);
    Ok(())
}

fn flatten_positions(positions: ArrayView2<'_, f64>) -> Vec<f64> {
    let (legs, axes) = positions.dim();
    let mut flat = Vec::with_capacity(legs * axes);
    for leg in 0..legs {
        for axis in 0..axes {
            flat.push(positions[(leg, axis)]);
        }
    }
    flat
}

fn positions_from_values(values: Vec<f64>, legs: usize) -> Result<Array2<f64>> {
    let expected = checked_count2("rat", legs, 3)?;
    if values.len() != expected {
        return Err(IoError::FeffBinShape {
            field: "rat",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }
    Array2::from_shape_vec((legs, 3), values).map_err(|_| IoError::FeffBinShape {
        field: "rat",
        actual: vec![legs, 3],
        expected: vec![legs, 3],
    })
}

struct ParsedPathHeader {
    index: usize,
    degeneracy: f64,
    effective_half_path_length_bohr: f64,
    criterion: f64,
    potential_indices: Vec<usize>,
}

struct FeffBinLines<'a> {
    lines: Vec<&'a str>,
    position: usize,
}

impl<'a> FeffBinLines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines().collect(),
            position: 0,
        }
    }

    fn version(&mut self) -> Result<String> {
        let line = self.next_line("version")?.trim_end();
        if !line.starts_with("#_feff.bin v03") {
            return Err(invalid_feff_bin(
                "version",
                format!("expected #_feff.bin v03 marker, got {line:?}"),
            ));
        }
        Ok(line
            .split_once(':')
            .map_or("", |(_, suffix)| suffix)
            .trim()
            .to_string())
    }

    fn counts(&mut self) -> Result<[i64; 3]> {
        let values = self.tagged_ints("#_", "counts")?;
        if values.len() != 3 {
            return Err(IoError::FeffBinShape {
                field: "counts",
                actual: vec![values.len()],
                expected: vec![3],
            });
        }
        Ok([values[0], values[1], values[2]])
    }

    fn misc(&mut self) -> Result<([i64; 3], [f64; 3])> {
        let line = self.next_tagged_line("#=", "misc")?;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 6 {
            return Err(IoError::FeffBinShape {
                field: "misc",
                actual: vec![tokens.len()],
                expected: vec![6],
            });
        }
        Ok((
            [
                parse_i64("ihole", tokens[0])?,
                parse_i64("iorder", tokens[1])?,
                parse_i64("ilinit", tokens[2])?,
            ],
            [
                parse_f64("rnrmav", tokens[3])?,
                parse_f64("xmu", tokens[4])?,
                parse_f64("edge", tokens[5])?,
            ],
        ))
    }

    fn potentials(&mut self, count: usize) -> Result<Vec<FeffBinPotential>> {
        let line = self.next_tagged_line("#@", "potentials")?;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let expected = checked_count2("potentials", count, 2)?;
        if tokens.len() != expected {
            return Err(IoError::FeffBinShape {
                field: "potentials",
                actual: vec![tokens.len()],
                expected: vec![expected],
            });
        }
        let mut potentials = Vec::with_capacity(count);
        for index in 0..count {
            let label = tokens[index].to_string();
            validate_label(&label)?;
            potentials.push(FeffBinPotential {
                label,
                atomic_number: usize_from_i64(parse_i64("iz", tokens[count + index])?, "iz")?,
            });
        }
        Ok(potentials)
    }

    fn next_path_header(&mut self) -> Result<Option<ParsedPathHeader>> {
        if self.position >= self.lines.len() {
            return Ok(None);
        }
        if self.lines[self.position].trim().is_empty() {
            self.position += 1;
            return self.next_path_header();
        }
        let line = self.next_tagged_line("##", "path")?;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 5 {
            return Err(IoError::FeffBinShape {
                field: "path",
                actual: vec![tokens.len()],
                expected: vec![5],
            });
        }
        let leg_count = usize_from_i64(parse_i64("nleg", tokens[1])?, "nleg")?;
        if tokens.len() != 5 + leg_count {
            return Err(IoError::FeffBinShape {
                field: "path",
                actual: vec![tokens.len()],
                expected: vec![5 + leg_count],
            });
        }
        let mut potential_indices = Vec::with_capacity(leg_count);
        for token in &tokens[5..] {
            potential_indices.push(usize_from_i64(parse_i64("ipot", token)?, "ipot")?);
        }
        Ok(Some(ParsedPathHeader {
            index: usize_from_i64(parse_i64("index", tokens[0])?, "index")?,
            degeneracy: parse_f64("deg", tokens[2])?,
            effective_half_path_length_bohr: parse_f64("reff", tokens[3])? / FEFF_BIN_BOHR,
            criterion: parse_f64("crit", tokens[4])?,
            potential_indices,
        }))
    }

    fn pad_reals(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            for value in decode_real_pad_line(field, line, pad_width)? {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(values)
    }

    fn pad_complex(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<Vec<Complex64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            for value in decode_complex_pad_line(field, line, pad_width)? {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(values)
    }

    fn tagged_ints(&mut self, tag: &'static str, field: &'static str) -> Result<Vec<i64>> {
        self.next_tagged_line(tag, field)?
            .split_whitespace()
            .map(|token| parse_i64(field, token))
            .collect()
    }

    fn next_tagged_line(&mut self, tag: &'static str, field: &'static str) -> Result<&'a str> {
        let line = self.next_line(field)?.trim_end();
        if !line.starts_with(tag) {
            return Err(invalid_feff_bin(
                field,
                format!("expected {tag:?} record, got {line:?}"),
            ));
        }
        Ok(line[tag.len()..].trim())
    }

    fn next_line(&mut self, field: &'static str) -> Result<&'a str> {
        let line = self
            .lines
            .get(self.position)
            .copied()
            .ok_or(IoError::FeffBinMissing { field })?;
        self.position += 1;
        Ok(line)
    }
}

fn decode_real_pad_line(field: &'static str, line: &str, pad_width: usize) -> Result<Vec<f64>> {
    decode_pad_chunks(field, line, pad_width, '!', |chunk| {
        decode_f64(chunk, pad_width)
    })
}

fn decode_complex_pad_line(
    field: &'static str,
    line: &str,
    pad_width: usize,
) -> Result<Vec<Complex64>> {
    decode_pad_chunks(field, line, 2 * pad_width, '$', |chunk| {
        let (re, im) = chunk.split_at(pad_width);
        Ok(Complex64::new(
            decode_f64(re, pad_width)?,
            decode_f64(im, pad_width)?,
        ))
    })
}

fn decode_pad_chunks<T>(
    field: &'static str,
    line: &str,
    unit_width: usize,
    marker: char,
    mut decode: impl FnMut(&str) -> Result<T>,
) -> Result<Vec<T>> {
    let trimmed = line.trim_start().trim_end();
    let Some(found) = trimmed.chars().next() else {
        return Err(IoError::FeffBinMissing { field });
    };
    if found != marker {
        return Err(IoError::PadMarker {
            expected: marker,
            found,
        });
    }
    let payload = &trimmed[found.len_utf8()..];
    if payload.is_empty() || !payload.len().is_multiple_of(unit_width) {
        return Err(IoError::PadPayload {
            payload_len: payload.len(),
            unit_len: unit_width,
        });
    }

    let mut values = Vec::with_capacity(payload.len() / unit_width);
    for chunk in payload.as_bytes().chunks(unit_width) {
        let chunk =
            std::str::from_utf8(chunk).map_err(|source| IoError::PadChunkUtf8 { source })?;
        values.push(decode(chunk)?);
    }
    Ok(values)
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::FeffBinShape {
            field,
            actual: vec![actual],
            expected: vec![expected],
        })
    }
}

fn validate_shape2(
    field: &'static str,
    actual: (usize, usize),
    expected: (usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::FeffBinShape {
            field,
            actual: vec![actual.0, actual.1],
            expected: vec![expected.0, expected.1],
        })
    }
}

fn validate_finite_reals(field: &'static str, values: impl IntoIterator<Item = f64>) -> Result<()> {
    for value in values {
        if !value.is_finite() {
            return Err(invalid_feff_bin(
                field,
                format!("value must be finite, got {value}"),
            ));
        }
    }
    Ok(())
}

fn validate_finite_complex(
    field: &'static str,
    values: impl IntoIterator<Item = Complex64>,
) -> Result<()> {
    for value in values {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(invalid_feff_bin(
                field,
                format!("value must be finite, got {value}"),
            ));
        }
    }
    Ok(())
}

fn validate_label(label: &str) -> Result<()> {
    if !label.is_empty() && label.len() <= 6 && label.is_ascii() {
        Ok(())
    } else {
        Err(invalid_feff_bin(
            "potlbl",
            format!("label {label:?} must be non-empty and at most 6 ASCII bytes"),
        ))
    }
}

fn checked_count2(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_feff_bin(field, "array element count overflowed"))
}

fn parse_i64(field: &'static str, token: &str) -> Result<i64> {
    token.parse::<i64>().map_err(|_| IoError::FeffBinParse {
        field,
        token: token.to_string(),
    })
}

fn parse_f64(field: &'static str, token: &str) -> Result<f64> {
    let value = token.parse::<f64>().map_err(|_| IoError::FeffBinParse {
        field,
        token: token.to_string(),
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_feff_bin(
            field,
            format!("value must be finite, got {value}"),
        ))
    }
}

fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_feff_bin(field, format!("value {value} does not fit in i64")))
}

fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_feff_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_feff_bin(field, format!("value {value} does not fit in usize")))
}

fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_feff_bin(field, format!("value {value} does not fit in i32")))
}

fn check_fixed_int(value: i64, width: usize, field: &'static str) -> Result<()> {
    let limit = 10_i64.pow(u32::try_from(width).map_err(|_| {
        invalid_feff_bin(field, format!("integer field width {width} is too large"))
    })?);
    let negative_limit = -(limit / 10 - 1);
    if value < negative_limit || value >= limit {
        Err(invalid_feff_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn invalid_feff_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidFeffBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_path_records_like_feff() -> Result<()> {
        let data = sample_feff_bin_data();
        let text = feff_bin_string(&data)?;
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("#_feff.bin v03: refeff-test"));
        assert_eq!(lines.next(), Some("#_    1    3    8"));
        assert!(text.lines().any(|line| line == "#@ Cu     O       29   8"));
        assert!(
            text.lines()
                .any(|line| line == "##    17   3   4.000   2.5000000        1.2500e1  0  1  0")
        );
        Ok(())
    }

    #[test]
    fn roundtrips_feff_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_feff_bin_data();
        let parsed = parse_feff_bin(&feff_bin_string(&data)?)?;
        assert_eq!(parsed.version, data.version);
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(parsed.ihole, data.ihole);
        assert_eq!(parsed.order, data.order);
        assert_eq!(
            parsed.initial_angular_momentum,
            data.initial_angular_momentum
        );
        assert_eq!(parsed.potentials, data.potentials);
        assert_close_reals(
            [
                parsed.average_norman_radius,
                parsed.fermi_level,
                parsed.edge_energy,
            ],
            [
                data.average_norman_radius,
                data.fermi_level,
                data.edge_energy,
            ],
        );
        assert_close_complex(parsed.central_phase_shift, data.central_phase_shift);
        assert_close_complex(parsed.complex_momentum, data.complex_momentum);
        assert_close_reals(parsed.real_momentum, data.real_momentum);
        assert_eq!(parsed.paths.len(), data.paths.len());
        for (actual, expected) in parsed.paths.iter().zip(&data.paths) {
            assert_eq!(actual.index, expected.index);
            assert_close_reals(
                [
                    actual.degeneracy,
                    actual.effective_half_path_length_bohr,
                    actual.criterion,
                ],
                [
                    expected.degeneracy,
                    expected.effective_half_path_length_bohr,
                    expected.criterion,
                ],
            );
            assert_eq!(actual.potential_indices, expected.potential_indices);
            assert_close_reals(
                actual.positions.iter().copied(),
                expected.positions.iter().copied(),
            );
            assert_close_reals(actual.beta.iter().copied(), expected.beta.iter().copied());
            assert_close_reals(actual.eta.iter().copied(), expected.eta.iter().copied());
            assert_close_reals(
                actual.leg_distances.iter().copied(),
                expected.leg_distances.iter().copied(),
            );
            assert_close_reals(
                actual.amplitude.iter().copied(),
                expected.amplitude.iter().copied(),
            );
            assert_close_reals(actual.phase.iter().copied(), expected.phase.iter().copied());
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_shapes_and_tokens() {
        let mut bad = sample_feff_bin_data();
        bad.real_momentum = Array1::from_vec(vec![1.0]);
        assert!(matches!(
            feff_bin_string(&bad),
            Err(IoError::FeffBinShape {
                field: "xk",
                actual,
                expected,
            }) if actual == vec![1] && expected == vec![3]
        ));

        assert!(matches!(
            parse_feff_bin("#_not-feff"),
            Err(IoError::InvalidFeffBin {
                field: "version",
                ..
            })
        ));
    }

    #[test]
    fn preserves_matching_raw_text() -> Result<()> {
        let data = sample_feff_bin_data();
        let text = feff_bin_string(&data)?;
        let mut parsed = parse_feff_bin(&text)?;
        let raw_text = parsed
            .raw_text
            .as_mut()
            .ok_or(IoError::FeffBinMissing { field: "raw_text" })?;
        raw_text.push('\n');

        let mut expected = text.clone();
        expected.push('\n');
        assert_eq!(feff_bin_string(&parsed)?, expected);

        parsed.edge_energy += 1.0;
        assert_ne!(feff_bin_string(&parsed)?, expected);
        Ok(())
    }

    fn sample_feff_bin_data() -> FeffBinData {
        FeffBinData {
            version: "refeff-test".to_string(),
            pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
            ihole: 1,
            order: 2,
            initial_angular_momentum: 0,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potentials: vec![
                FeffBinPotential {
                    label: "Cu".to_string(),
                    atomic_number: 29,
                },
                FeffBinPotential {
                    label: "O".to_string(),
                    atomic_number: 8,
                },
            ],
            central_phase_shift: Array1::from_vec(vec![
                Complex64::new(0.1, -0.01),
                Complex64::new(0.2, -0.02),
                Complex64::new(0.3, -0.03),
            ]),
            complex_momentum: Array1::from_vec(vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(1.1, 0.2),
                Complex64::new(1.2, 0.3),
            ]),
            real_momentum: Array1::from_vec(vec![0.5, 0.6, 0.7]),
            paths: vec![FeffBinPath {
                index: 17,
                degeneracy: 4.0,
                effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
                criterion: 12.5,
                potential_indices: Array1::from_vec(vec![0, 1, 0]),
                positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                    (0, 0..=2) => 0.0,
                    (1, 0) => 1.0,
                    (1, 1) => 0.5,
                    (1, 2) => 0.0,
                    (2, 0) => -1.0,
                    (2, 1) => 0.25,
                    (2, 2) => 0.0,
                    _ => 0.0,
                }),
                beta: Array1::from_vec(vec![0.1, 0.2, 0.3]),
                eta: Array1::from_vec(vec![0.4, 0.5, 0.6]),
                leg_distances: Array1::from_vec(vec![1.0, 1.1, 1.2]),
                amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
                phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
            }],
            raw_text: None,
        }
    }

    fn assert_close_reals(
        actual: impl IntoIterator<Item = f64>,
        expected: impl IntoIterator<Item = f64>,
    ) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
        }
    }

    fn assert_close_complex(
        actual: impl IntoIterator<Item = Complex64>,
        expected: impl IntoIterator<Item = Complex64>,
    ) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual.re - expected.re).abs() <= expected.re.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
            assert!(
                (actual.im - expected.im).abs() <= expected.im.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
        }
    }
}
