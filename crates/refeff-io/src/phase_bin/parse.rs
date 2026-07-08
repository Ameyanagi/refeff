use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::pad::decode_f64;

use super::common::{
    checked_count2, checked_count3, checked_l_count, i32_from_i64, invalid_phase_bin,
    usize_from_i64, validate_label,
};
use super::types::{
    PHASE_BIN_DEFAULT_TRANSITION_COUNT, PHASE_BIN_SCALARS, PhaseBinData, PhaseBinPotential,
    PhaseBinRawPads, PhaseBinScalars,
};
use super::validate::validate_phase_bin;

/// Parse FEFF `phase.bin` text.
pub fn parse_phase_bin(text: &str) -> Result<PhaseBinData> {
    let mut lines = PhaseBinLines::new(text);
    let header = lines.header()?;
    let spin_count = usize_from_i64(header[0], "nsp")?;
    let energy_count = usize_from_i64(header[1], "ne")?;
    let main_energy_count = usize_from_i64(header[2], "ne1")?;
    let auxiliary_energy_count = usize_from_i64(header[3], "ne3")?;
    let potential_count = usize_from_i64(header[4], "nph")?
        .checked_add(1)
        .ok_or_else(|| invalid_phase_bin("nph", "potential count overflowed"))?;
    let ihole = i32_from_i64(header[5], "ihole")?;
    let fermi_index = i32_from_i64(header[6], "ik0")?;
    let pad_width = usize_from_i64(header[7], "npadx")?;
    let (final_state_count, transition_count, q_count) = match header.len() {
        11 => (
            usize_from_i64(header[8], "kfinmax")?,
            usize_from_i64(header[9], "indmax")?,
            usize_from_i64(header[10], "nq")?,
        ),
        10 => (
            usize_from_i64(header[8], "kfinmax")?,
            usize_from_i64(header[9], "indmax")?,
            1,
        ),
        _ => (
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            1,
        ),
    };

    let scalars_block = lines.pad_reals_block("dum", pad_width, PHASE_BIN_SCALARS)?;
    let scalars = PhaseBinScalars::from_slice(&scalars_block.values)?;
    let energy_grid_block = lines.pad_complex_block("em", pad_width, energy_count)?;
    let energy_grid = Array1::from_vec(energy_grid_block.values);
    let reference_energy_block = lines.pad_complex_block(
        "eref",
        pad_width,
        checked_count2("eref", energy_count, spin_count)?,
    )?;
    let reference_energy = array2_complex_from_fortran(
        "eref",
        reference_energy_block.values,
        energy_count,
        spin_count,
    )?;

    let mut potentials = Vec::with_capacity(potential_count);
    let mut raw_phase_shifts = Vec::with_capacity(potential_count);
    for _ in 0..potential_count {
        let (lmax, atomic_number, label) = lines.potential_header()?;
        let l_count = checked_l_count(lmax)?;
        let mut phase_shifts = Array3::<Complex64>::zeros((energy_count, l_count, spin_count));
        let mut raw_spin_blocks = Vec::with_capacity(spin_count);
        for spin in 0..spin_count {
            let block = lines.pad_complex_block(
                "ph",
                pad_width,
                checked_count2("ph", energy_count, l_count)?,
            )?;
            fill_phase_spin(&mut phase_shifts, spin, &block.values)?;
            raw_spin_blocks.push(Some(block.raw));
        }
        raw_phase_shifts.push(raw_spin_blocks);
        potentials.push(PhaseBinPotential {
            lmax,
            atomic_number,
            label,
            phase_shifts,
        });
    }

    let mut transition_moments =
        Array4::<Complex64>::zeros((energy_count, q_count, transition_count, spin_count));
    let mut raw_transition_moments = Vec::with_capacity(q_count);
    for q_index in 0..q_count {
        let block = lines.pad_complex_block(
            "rkk",
            pad_width,
            checked_count3("rkk", energy_count, transition_count, spin_count)?,
        )?;
        fill_transition_q(&mut transition_moments, q_index, &block.values)?;
        raw_transition_moments.push(Some(block.raw));
    }
    lines.finish()?;

    let data = PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count,
        auxiliary_energy_count,
        ihole,
        fermi_index,
        pad_width,
        final_state_count,
        transition_count,
        q_count,
        scalars,
        energy_grid,
        reference_energy,
        potentials,
        transition_moments,
        raw_pads: Some(PhaseBinRawPads {
            scalars: Some(scalars_block.raw),
            energy_grid: Some(energy_grid_block.raw),
            reference_energy: Some(reference_energy_block.raw),
            phase_shifts: raw_phase_shifts,
            transition_moments: raw_transition_moments,
        }),
    };
    validate_phase_bin(&data)?;
    Ok(data)
}

fn fill_phase_spin(
    phase_shifts: &mut Array3<Complex64>,
    spin: usize,
    values: &[Complex64],
) -> Result<()> {
    let (energies, l_count, _) = phase_shifts.dim();
    let expected = checked_count2("ph", energies, l_count)?;
    if values.len() != expected {
        return Err(IoError::PhaseBinShape {
            field: "ph",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }
    let mut index = 0;
    for energy in 0..energies {
        for l_slot in 0..l_count {
            phase_shifts[(energy, l_slot, spin)] = values[index];
            index += 1;
        }
    }
    Ok(())
}

fn fill_transition_q(
    transition_moments: &mut Array4<Complex64>,
    q_index: usize,
    values: &[Complex64],
) -> Result<()> {
    let (energies, _, transitions, spins) = transition_moments.dim();
    let expected = checked_count3("rkk", energies, transitions, spins)?;
    if values.len() != expected {
        return Err(IoError::PhaseBinShape {
            field: "rkk",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }
    let mut index = 0;
    for spin in 0..spins {
        for transition in 0..transitions {
            for energy in 0..energies {
                transition_moments[(energy, q_index, transition, spin)] = values[index];
                index += 1;
            }
        }
    }
    Ok(())
}

fn array2_complex_from_fortran(
    field: &'static str,
    values: Vec<Complex64>,
    rows: usize,
    cols: usize,
) -> Result<Array2<Complex64>> {
    Array2::from_shape_vec((rows, cols).f(), values).map_err(|_| IoError::PhaseBinShape {
        field,
        actual: vec![rows, cols],
        expected: vec![rows, cols],
    })
}

#[derive(Debug, Clone)]
struct PadBlock<T> {
    values: Vec<T>,
    raw: String,
}

struct PhaseBinLines<'a> {
    lines: Vec<&'a str>,
    position: usize,
}

impl<'a> PhaseBinLines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines().collect(),
            position: 0,
        }
    }

    fn finish(self) -> Result<()> {
        let count = self.lines[self.position..]
            .iter()
            .filter(|line| !line.trim().is_empty())
            .count();
        if count == 0 {
            Ok(())
        } else {
            Err(IoError::PhaseBinTrailingLines { count })
        }
    }

    fn header(&mut self) -> Result<Vec<i64>> {
        let line = self.next_line("header")?;
        let values = parse_int_line("header", line)?;
        if matches!(values.len(), 8 | 10 | 11) {
            Ok(values)
        } else {
            Err(IoError::PhaseBinShape {
                field: "header",
                actual: vec![values.len()],
                expected: vec![8, 10, 11],
            })
        }
    }

    fn potential_header(&mut self) -> Result<(usize, usize, String)> {
        let line = self.next_line("potential_header")?;
        let mut parts = line.split_whitespace();
        let lmax_token = parts.next().ok_or(IoError::PhaseBinMissing {
            field: "potential_header",
        })?;
        let iz_token = parts.next().ok_or(IoError::PhaseBinMissing {
            field: "potential_header",
        })?;
        let label = parts.next().map_or_else(String::new, str::to_string);
        let lmax = lmax_token
            .parse::<i64>()
            .map_err(|_| IoError::PhaseBinParse {
                field: "lmax",
                token: lmax_token.to_string(),
            })
            .and_then(|value| usize_from_i64(value, "lmax"))?;
        let atomic_number = iz_token
            .parse::<i64>()
            .map_err(|_| IoError::PhaseBinParse {
                field: "iz",
                token: iz_token.to_string(),
            })
            .and_then(|value| usize_from_i64(value, "iz"))?;
        validate_label(&label)?;
        Ok((lmax, atomic_number, label))
    }

    fn pad_reals_block(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<PadBlock<f64>> {
        let mut values = Vec::with_capacity(expected);
        let mut raw = String::new();
        while values.len() < expected {
            let line = self.next_line(field)?;
            raw.push_str(line);
            raw.push('\n');
            let decoded = decode_real_pad_line(field, line, pad_width)?;
            for value in decoded {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(PadBlock { values, raw })
    }

    fn pad_complex_block(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<PadBlock<Complex64>> {
        let mut values = Vec::with_capacity(expected);
        let mut raw = String::new();
        while values.len() < expected {
            let line = self.next_line(field)?;
            raw.push_str(line);
            raw.push('\n');
            let decoded = decode_complex_pad_line(field, line, pad_width)?;
            for value in decoded {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(PadBlock { values, raw })
    }

    fn next_line(&mut self, field: &'static str) -> Result<&'a str> {
        let line = self
            .lines
            .get(self.position)
            .copied()
            .ok_or(IoError::PhaseBinMissing { field })?;
        self.position += 1;
        Ok(line)
    }
}

fn parse_int_line(field: &'static str, line: &str) -> Result<Vec<i64>> {
    line.split_whitespace()
        .map(|token| {
            token.parse::<i64>().map_err(|_| IoError::PhaseBinParse {
                field,
                token: token.to_string(),
            })
        })
        .collect()
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
        return Err(IoError::PhaseBinMissing { field });
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

pub(super) fn decode_raw_real_pad(
    field: &'static str,
    raw: &str,
    pad_width: usize,
) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        values.extend(decode_real_pad_line(field, line, pad_width)?);
    }
    Ok(values)
}

pub(super) fn decode_raw_complex_pad(
    field: &'static str,
    raw: &str,
    pad_width: usize,
) -> Result<Vec<Complex64>> {
    let mut values = Vec::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        values.extend(decode_complex_pad_line(field, line, pad_width)?);
    }
    Ok(values)
}
