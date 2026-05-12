//! FEFF `phase.bin` text/PAD phase-shift codec.
//!
//! `XSPH/wrxsph.f90` writes this handoff file for downstream FMS and FF2X
//! stages. The file is formatted text: a fixed-width integer header, a small
//! real PAD block, and several complex PAD blocks. This module preserves that
//! order while exposing phase shifts and transition moments as `ndarray`
//! values.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::repeated_ints;
use crate::pad::{decode_f64, encode_complex, encode_reals};

/// FEFF10 default PAD width used by `wrxsph`.
pub const PHASE_BIN_DEFAULT_PAD_WIDTH: usize = 8;
/// Number of scalar values in the FEFF `dum(3)` phase header block.
pub const PHASE_BIN_SCALARS: usize = 3;
/// Historical non-NRIXS transition-moment count read by old `rdxsph`.
pub const PHASE_BIN_DEFAULT_TRANSITION_COUNT: usize = 8;

/// Scalar `dum(3)` block from FEFF `phase.bin`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseBinScalars {
    /// Average Norman radius, `rnrmav`.
    pub average_norman_radius: f64,
    /// Fermi level position, `xmu`.
    pub fermi_level: f64,
    /// Edge energy, `edge`.
    pub edge_energy: f64,
}

impl PhaseBinScalars {
    /// Return the FEFF `dum(3)` values in `wrxsph` order.
    #[must_use]
    pub fn as_array(self) -> [f64; PHASE_BIN_SCALARS] {
        [
            self.average_norman_radius,
            self.fermi_level,
            self.edge_energy,
        ]
    }

    fn from_slice(values: &[f64]) -> Result<Self> {
        if values.len() != PHASE_BIN_SCALARS {
            return Err(IoError::PhaseBinShape {
                field: "dum",
                actual: vec![values.len()],
                expected: vec![PHASE_BIN_SCALARS],
            });
        }
        Ok(Self {
            average_norman_radius: values[0],
            fermi_level: values[1],
            edge_energy: values[2],
        })
    }
}

/// Per-potential phase-shift block from FEFF `phase.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinPotential {
    /// Maximum angular momentum written for this potential.
    pub lmax: usize,
    /// Atomic number, `iz(iph)`.
    pub atomic_number: usize,
    /// FEFF six-character potential label, `potlbl(iph)`.
    pub label: String,
    /// Phase shifts as `(energy, l_slot -lmax..lmax, spin)`.
    pub phase_shifts: Array3<Complex64>,
}

/// FEFF `phase.bin` contents from `XSPH/wrxsph.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinData {
    /// Spin channel count, `nsp`.
    pub spin_count: usize,
    /// Energy grid count, `ne`.
    pub energy_count: usize,
    /// Main horizontal-axis energy count, `ne1`.
    pub main_energy_count: usize,
    /// Auxiliary horizontal-axis energy count, `ne3`.
    pub auxiliary_energy_count: usize,
    /// Core-hole index, `ihole`.
    pub ihole: i32,
    /// Fermi-level grid index, `ik0`.
    pub fermi_index: i32,
    /// PAD field width, `npadx`.
    pub pad_width: usize,
    /// FEFFQ final-state channel count, `kfinmax`.
    pub final_state_count: usize,
    /// Number of transition-moment channels written, `indmax`.
    pub transition_count: usize,
    /// Momentum-transfer vector count, `nq`.
    pub q_count: usize,
    /// FEFF scalar `dum(3)` block.
    pub scalars: PhaseBinScalars,
    /// Complex energy mesh, `em(1:ne)`.
    pub energy_grid: Array1<Complex64>,
    /// Reference/self-energy mesh as `(energy, spin)`, `eref`.
    pub reference_energy: Array2<Complex64>,
    /// Per-potential phase-shift blocks for FEFF `iph=0:nph`.
    pub potentials: Vec<PhaseBinPotential>,
    /// Transition moments as `(energy, q, transition, spin)`, `rkk`.
    pub transition_moments: Array4<Complex64>,
}

impl PhaseBinData {
    /// Number of FEFF potential types represented by `0:nph`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potentials.len()
    }
}

/// Render FEFF `phase.bin` text.
pub fn phase_bin_string(data: &PhaseBinData) -> Result<String> {
    validate_phase_bin(data)?;

    let mut out = String::new();
    write_int_line(
        &mut out,
        &[
            i64_from_usize(data.spin_count, "nsp")?,
            i64_from_usize(data.energy_count, "ne")?,
            i64_from_usize(data.main_energy_count, "ne1")?,
            i64_from_usize(data.auxiliary_energy_count, "ne3")?,
            i64_from_usize(data.potential_count() - 1, "nph")?,
            i64::from(data.ihole),
            i64::from(data.fermi_index),
            i64_from_usize(data.pad_width, "npadx")?,
            i64_from_usize(data.final_state_count, "kfinmax")?,
            i64_from_usize(data.transition_count, "indmax")?,
            i64_from_usize(data.q_count, "nq")?,
        ],
        4,
        "header",
    )?;
    write_pad_values(&mut out, &data.scalars.as_array(), data.pad_width)?;
    write_complex_pad(
        &mut out,
        &data.energy_grid.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_complex_pad(
        &mut out,
        &flatten_reference_energy(data.reference_energy.view()),
        data.pad_width,
    )?;

    for potential in &data.potentials {
        write_potential_header(&mut out, potential)?;
        for spin in 0..data.spin_count {
            write_complex_pad(
                &mut out,
                &flatten_phase_spin(potential.phase_shifts.view(), spin),
                data.pad_width,
            )?;
        }
    }

    for q_index in 0..data.q_count {
        write_complex_pad(
            &mut out,
            &flatten_transition_q(data.transition_moments.view(), q_index),
            data.pad_width,
        )?;
    }
    Ok(out)
}

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
    let (final_state_count, transition_count, q_count) = if header.len() >= 11 {
        (
            usize_from_i64(header[8], "kfinmax")?,
            usize_from_i64(header[9], "indmax")?,
            usize_from_i64(header[10], "nq")?,
        )
    } else {
        (
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            1,
        )
    };

    let scalars =
        PhaseBinScalars::from_slice(&lines.pad_reals("dum", pad_width, PHASE_BIN_SCALARS)?)?;
    let energy_grid = Array1::from_vec(lines.pad_complex("em", pad_width, energy_count)?);
    let reference_energy = array2_complex_from_fortran(
        "eref",
        lines.pad_complex(
            "eref",
            pad_width,
            checked_count2("eref", energy_count, spin_count)?,
        )?,
        energy_count,
        spin_count,
    )?;

    let mut potentials = Vec::with_capacity(potential_count);
    for _ in 0..potential_count {
        let (lmax, atomic_number, label) = lines.potential_header()?;
        let l_count = checked_l_count(lmax)?;
        let mut phase_shifts = Array3::<Complex64>::zeros((energy_count, l_count, spin_count));
        for spin in 0..spin_count {
            let values = lines.pad_complex(
                "ph",
                pad_width,
                checked_count2("ph", energy_count, l_count)?,
            )?;
            fill_phase_spin(&mut phase_shifts, spin, &values)?;
        }
        potentials.push(PhaseBinPotential {
            lmax,
            atomic_number,
            label,
            phase_shifts,
        });
    }

    let mut transition_moments =
        Array4::<Complex64>::zeros((energy_count, q_count, transition_count, spin_count));
    for q_index in 0..q_count {
        let values = lines.pad_complex(
            "rkk",
            pad_width,
            checked_count3("rkk", energy_count, transition_count, spin_count)?,
        )?;
        fill_transition_q(&mut transition_moments, q_index, &values)?;
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
    };
    validate_phase_bin(&data)?;
    Ok(data)
}

/// Write FEFF `phase.bin` text to a file.
pub fn write_phase_bin(path: impl AsRef<Path>, data: &PhaseBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, phase_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `phase.bin` text from a file.
pub fn read_phase_bin(path: impl AsRef<Path>) -> Result<PhaseBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_phase_bin(&text)
}

fn validate_phase_bin(data: &PhaseBinData) -> Result<()> {
    if data.pad_width <= 2 {
        return Err(IoError::InvalidPadWidth(data.pad_width));
    }
    if data.spin_count == 0 {
        return Err(invalid_phase_bin("nsp", "at least one spin is required"));
    }
    if data.energy_count == 0 {
        return Err(invalid_phase_bin("ne", "at least one energy is required"));
    }
    if data.potential_count() == 0 {
        return Err(invalid_phase_bin(
            "nph",
            "at least one potential is required",
        ));
    }
    if data.q_count == 0 {
        return Err(invalid_phase_bin("nq", "at least one q-vector is required"));
    }
    if data.transition_count == 0 || data.transition_count > data.final_state_count {
        return Err(invalid_phase_bin(
            "indmax",
            "transition count must be in 1..=kfinmax",
        ));
    }

    for (field, value) in [
        ("nsp", i64_from_usize(data.spin_count, "nsp")?),
        ("ne", i64_from_usize(data.energy_count, "ne")?),
        ("ne1", i64_from_usize(data.main_energy_count, "ne1")?),
        ("ne3", i64_from_usize(data.auxiliary_energy_count, "ne3")?),
        ("nph", i64_from_usize(data.potential_count() - 1, "nph")?),
        ("ihole", i64::from(data.ihole)),
        ("ik0", i64::from(data.fermi_index)),
        ("npadx", i64_from_usize(data.pad_width, "npadx")?),
        (
            "kfinmax",
            i64_from_usize(data.final_state_count, "kfinmax")?,
        ),
        ("indmax", i64_from_usize(data.transition_count, "indmax")?),
        ("nq", i64_from_usize(data.q_count, "nq")?),
    ] {
        check_fixed_int(value, 4, field)?;
    }

    validate_len("em", data.energy_grid.len(), data.energy_count)?;
    validate_shape2(
        "eref",
        data.reference_energy.dim(),
        (data.energy_count, data.spin_count),
    )?;
    validate_shape4(
        "rkk",
        data.transition_moments.dim(),
        (
            data.energy_count,
            data.q_count,
            data.transition_count,
            data.spin_count,
        ),
    )?;

    validate_finite_reals("dum", data.scalars.as_array())?;
    validate_finite_complex("em", data.energy_grid.iter().copied())?;
    validate_finite_complex("eref", data.reference_energy.iter().copied())?;
    validate_finite_complex("rkk", data.transition_moments.iter().copied())?;

    for potential in &data.potentials {
        let l_count = checked_l_count(potential.lmax)?;
        check_fixed_int(i64_from_usize(potential.lmax, "lmax")?, 3, "lmax")?;
        check_fixed_int(i64_from_usize(potential.atomic_number, "iz")?, 3, "iz")?;
        validate_label(&potential.label)?;
        validate_shape3(
            "ph",
            potential.phase_shifts.dim(),
            (data.energy_count, l_count, data.spin_count),
        )?;
        validate_finite_complex("ph", potential.phase_shifts.iter().copied())?;
    }
    Ok(())
}

fn write_potential_header(out: &mut String, potential: &PhaseBinPotential) -> Result<()> {
    check_fixed_int(i64_from_usize(potential.lmax, "lmax")?, 3, "lmax")?;
    check_fixed_int(i64_from_usize(potential.atomic_number, "iz")?, 3, "iz")?;
    validate_label(&potential.label)?;
    writeln!(
        out,
        " {:>3} {:>3} {:<6}",
        potential.lmax, potential.atomic_number, potential.label
    )?;
    Ok(())
}

fn write_int_line(
    out: &mut String,
    values: &[i64],
    width: usize,
    field: &'static str,
) -> Result<()> {
    for &value in values {
        check_fixed_int(value, width, field)?;
    }
    writeln!(out, "{}", repeated_ints(values.iter().copied(), width))?;
    Ok(())
}

fn write_pad_values(out: &mut String, values: &[f64], pad_width: usize) -> Result<()> {
    out.push_str(&encode_reals(values, pad_width)?);
    Ok(())
}

fn write_complex_pad(out: &mut String, values: &[Complex64], pad_width: usize) -> Result<()> {
    out.push_str(&encode_complex(values, pad_width)?);
    Ok(())
}

fn flatten_reference_energy(values: ndarray::ArrayView2<'_, Complex64>) -> Vec<Complex64> {
    let (energies, spins) = values.dim();
    let mut flat = Vec::with_capacity(energies * spins);
    for spin in 0..spins {
        for energy in 0..energies {
            flat.push(values[(energy, spin)]);
        }
    }
    flat
}

fn flatten_phase_spin(values: ndarray::ArrayView3<'_, Complex64>, spin: usize) -> Vec<Complex64> {
    let (energies, l_count, _) = values.dim();
    let mut flat = Vec::with_capacity(energies * l_count);
    for energy in 0..energies {
        for l_slot in 0..l_count {
            flat.push(values[(energy, l_slot, spin)]);
        }
    }
    flat
}

fn flatten_transition_q(
    values: ndarray::ArrayView4<'_, Complex64>,
    q_index: usize,
) -> Vec<Complex64> {
    let (energies, _, transitions, spins) = values.dim();
    let mut flat = Vec::with_capacity(energies * transitions * spins);
    for spin in 0..spins {
        for transition in 0..transitions {
            for energy in 0..energies {
                flat.push(values[(energy, q_index, transition, spin)]);
            }
        }
    }
    flat
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
        if values.len() == 8 || values.len() == 11 {
            Ok(values)
        } else {
            Err(IoError::PhaseBinShape {
                field: "header",
                actual: vec![values.len()],
                expected: vec![11],
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
        let label = parts.next().unwrap_or_default().to_string();
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

    fn pad_reals(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            let decoded = decode_real_pad_line(field, line, pad_width)?;
            for value in decoded {
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
            let decoded = decode_complex_pad_line(field, line, pad_width)?;
            for value in decoded {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(values)
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

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PhaseBinShape {
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
        Err(IoError::PhaseBinShape {
            field,
            actual: vec![actual.0, actual.1],
            expected: vec![expected.0, expected.1],
        })
    }
}

fn validate_shape3(
    field: &'static str,
    actual: (usize, usize, usize),
    expected: (usize, usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PhaseBinShape {
            field,
            actual: vec![actual.0, actual.1, actual.2],
            expected: vec![expected.0, expected.1, expected.2],
        })
    }
}

fn validate_shape4(
    field: &'static str,
    actual: (usize, usize, usize, usize),
    expected: (usize, usize, usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PhaseBinShape {
            field,
            actual: vec![actual.0, actual.1, actual.2, actual.3],
            expected: vec![expected.0, expected.1, expected.2, expected.3],
        })
    }
}

fn validate_finite_reals(field: &'static str, values: impl IntoIterator<Item = f64>) -> Result<()> {
    for value in values {
        if !value.is_finite() {
            return Err(invalid_phase_bin(
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
            return Err(invalid_phase_bin(
                field,
                format!("value must be finite, got {value}"),
            ));
        }
    }
    Ok(())
}

fn validate_label(label: &str) -> Result<()> {
    if label.len() <= 6 && label.is_ascii() {
        Ok(())
    } else {
        Err(invalid_phase_bin(
            "potlbl",
            format!("label {label:?} must be at most 6 ASCII bytes"),
        ))
    }
}

fn checked_l_count(lmax: usize) -> Result<usize> {
    lmax.checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid_phase_bin("lmax", "l channel count overflowed"))
}

fn checked_count2(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_phase_bin(field, "array element count overflowed"))
}

fn checked_count3(field: &'static str, rows: usize, cols: usize, planes: usize) -> Result<usize> {
    checked_count2(field, rows, cols)?
        .checked_mul(planes)
        .ok_or_else(|| invalid_phase_bin(field, "array element count overflowed"))
}

fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("value {value} does not fit in i64")))
}

fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_phase_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("value {value} does not fit in usize")))
}

fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("value {value} does not fit in i32")))
}

fn check_fixed_int(value: i64, width: usize, field: &'static str) -> Result<()> {
    let limit = 10_i64.pow(u32::try_from(width).map_err(|_| {
        invalid_phase_bin(field, format!("integer field width {width} is too large"))
    })?);
    let negative_limit = -(limit / 10 - 1);
    if value < negative_limit || value >= limit {
        Err(invalid_phase_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn invalid_phase_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidPhaseBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_phase_bin_header_like_feff() -> Result<()> {
        let data = sample_phase_bin_data();
        let text = phase_bin_string(&data)?;
        assert_eq!(
            text.lines().next(),
            Some("    2    3    2    1    1    4    2    8    4    3    2")
        );
        assert!(text.lines().any(|line| line == "   1  29 Cu    "));
        assert!(text.lines().any(|line| line == "   2   8 O     "));
        Ok(())
    }

    #[test]
    fn roundtrips_phase_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_phase_bin_data();
        let parsed = parse_phase_bin(&phase_bin_string(&data)?)?;
        assert_eq!(parsed.spin_count, data.spin_count);
        assert_eq!(parsed.energy_count, data.energy_count);
        assert_eq!(parsed.main_energy_count, data.main_energy_count);
        assert_eq!(parsed.auxiliary_energy_count, data.auxiliary_energy_count);
        assert_eq!(parsed.ihole, data.ihole);
        assert_eq!(parsed.fermi_index, data.fermi_index);
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(parsed.final_state_count, data.final_state_count);
        assert_eq!(parsed.transition_count, data.transition_count);
        assert_eq!(parsed.q_count, data.q_count);
        assert_close_reals(parsed.scalars.as_array(), data.scalars.as_array());
        assert_close_complex(parsed.energy_grid, data.energy_grid);
        assert_close_complex(parsed.reference_energy, data.reference_energy);
        assert_close_complex(parsed.transition_moments, data.transition_moments);
        assert_eq!(parsed.potentials.len(), data.potentials.len());
        for (actual, expected) in parsed.potentials.iter().zip(data.potentials.iter()) {
            assert_eq!(actual.lmax, expected.lmax);
            assert_eq!(actual.atomic_number, expected.atomic_number);
            assert_eq!(actual.label, expected.label);
            assert_close_complex(
                actual.phase_shifts.iter().copied(),
                expected.phase_shifts.iter().copied(),
            );
        }
        Ok(())
    }

    #[test]
    fn parses_legacy_eight_integer_header() -> Result<()> {
        let mut text = phase_bin_string(&legacy_phase_bin_data())?;
        text.replace_range(
            0..text.lines().next().map_or(0, str::len),
            "    2    3    2    1    1    4    2    8",
        );
        let parsed = parse_phase_bin(&text)?;
        assert_eq!(parsed.final_state_count, PHASE_BIN_DEFAULT_TRANSITION_COUNT);
        assert_eq!(parsed.transition_count, PHASE_BIN_DEFAULT_TRANSITION_COUNT);
        assert_eq!(parsed.q_count, 1);
        Ok(())
    }

    #[test]
    fn rejects_invalid_shapes_and_tokens() {
        let mut bad = sample_phase_bin_data();
        bad.energy_grid = Array1::from_vec(vec![Complex64::new(1.0, 0.0)]);
        assert!(matches!(
            phase_bin_string(&bad),
            Err(IoError::PhaseBinShape {
                field: "em",
                actual,
                expected,
            }) if actual == vec![1] && expected == vec![3]
        ));

        assert!(matches!(
            parse_phase_bin("not-an-int"),
            Err(IoError::PhaseBinParse {
                field: "header",
                ..
            })
        ));
    }

    fn sample_phase_bin_data() -> PhaseBinData {
        let spin_count = 2;
        let energy_count = 3;
        let q_count = 2;
        let transition_count = 3;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: 2,
            auxiliary_energy_count: 1,
            ihole: 4,
            fermi_index: 2,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: 4,
            transition_count,
            q_count,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.2,
                fermi_level: -0.35,
                edge_energy: 9.8,
            },
            energy_grid: Array1::from_shape_fn(energy_count, |energy| {
                Complex64::new(0.5 + energy as f64, 0.1 * energy as f64)
            }),
            reference_energy: Array2::from_shape_fn(
                (energy_count, spin_count),
                |(energy, spin)| Complex64::new(-1.0 + energy as f64 * 0.2, 0.05 * spin as f64),
            ),
            potentials: vec![
                sample_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
                sample_potential(2, 8, "O", energy_count, spin_count, 0.2),
            ],
            transition_moments: Array4::from_shape_fn(
                (energy_count, q_count, transition_count, spin_count),
                |(energy, q_index, transition, spin)| {
                    Complex64::new(
                        0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                        -0.02 * spin as f64,
                    )
                },
            ),
        }
    }

    fn legacy_phase_bin_data() -> PhaseBinData {
        let mut data = sample_phase_bin_data();
        data.final_state_count = PHASE_BIN_DEFAULT_TRANSITION_COUNT;
        data.transition_count = PHASE_BIN_DEFAULT_TRANSITION_COUNT;
        data.q_count = 1;
        data.transition_moments = Array4::from_shape_fn(
            (
                data.energy_count,
                data.q_count,
                data.transition_count,
                data.spin_count,
            ),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                    -0.02 * spin as f64,
                )
            },
        );
        data
    }

    fn sample_potential(
        lmax: usize,
        atomic_number: usize,
        label: &str,
        energy_count: usize,
        spin_count: usize,
        scale: f64,
    ) -> PhaseBinPotential {
        let l_count = 2 * lmax + 1;
        PhaseBinPotential {
            lmax,
            atomic_number,
            label: label.to_string(),
            phase_shifts: Array3::from_shape_fn(
                (energy_count, l_count, spin_count),
                |(energy, l_slot, spin)| {
                    Complex64::new(
                        scale + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                        0.001 * spin as f64,
                    )
                },
            ),
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
