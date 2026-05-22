use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::pad::decode_f64;

use super::common::{
    checked_count2, i32_from_i64, invalid_feff_bin, parse_f64, parse_i64, usize_from_i64,
    validate_label,
};
use super::types::{FEFF_BIN_BOHR, FeffBinData, FeffBinPath, FeffBinPotential};
use super::validate::validate_feff_bin;

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
