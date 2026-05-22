use std::fmt::Write as _;
use std::path::Path;

use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::repeated_ints;
use crate::pad::{encode_complex, encode_reals};

use super::common::{check_fixed_int, i64_from_usize, validate_label};
use super::parse::{decode_raw_complex_pad, decode_raw_real_pad, parse_phase_bin};
use super::types::{PhaseBinData, PhaseBinPotential};
use super::validate::validate_phase_bin;

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
    let raw_pads = data.raw_pads.as_ref();
    let scalars = data.scalars.as_array();
    write_real_pad_block(
        &mut out,
        raw_pads.and_then(|raw| raw.scalars.as_deref()),
        "dum",
        data.pad_width,
        &scalars,
    )?;
    write_complex_pad_block(
        &mut out,
        raw_pads.and_then(|raw| raw.energy_grid.as_deref()),
        "em",
        &data.energy_grid.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_complex_pad_block(
        &mut out,
        raw_pads.and_then(|raw| raw.reference_energy.as_deref()),
        "eref",
        &flatten_reference_energy(data.reference_energy.view()),
        data.pad_width,
    )?;

    for (potential_index, potential) in data.potentials.iter().enumerate() {
        write_potential_header(&mut out, potential)?;
        for spin in 0..data.spin_count {
            let raw_pad = raw_pads
                .and_then(|raw| raw.phase_shifts.get(potential_index))
                .and_then(|per_spin| per_spin.get(spin))
                .and_then(Option::as_deref);
            write_complex_pad_block(
                &mut out,
                raw_pad,
                "ph",
                &flatten_phase_spin(potential.phase_shifts.view(), spin),
                data.pad_width,
            )?;
        }
    }

    for q_index in 0..data.q_count {
        let raw_pad = raw_pads
            .and_then(|raw| raw.transition_moments.get(q_index))
            .and_then(Option::as_deref);
        write_complex_pad_block(
            &mut out,
            raw_pad,
            "rkk",
            &flatten_transition_q(data.transition_moments.view(), q_index),
            data.pad_width,
        )?;
    }
    Ok(out)
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

fn write_real_pad_block(
    out: &mut String,
    raw: Option<&str>,
    field: &'static str,
    pad_width: usize,
    values: &[f64],
) -> Result<()> {
    if let Some(raw) = raw
        && raw_real_pad_matches(field, raw, pad_width, values)?
    {
        out.push_str(raw);
        return Ok(());
    }
    write_pad_values(out, values, pad_width)
}

fn write_complex_pad_block(
    out: &mut String,
    raw: Option<&str>,
    field: &'static str,
    values: &[Complex64],
    pad_width: usize,
) -> Result<()> {
    if let Some(raw) = raw
        && raw_complex_pad_matches(field, raw, pad_width, values)?
    {
        out.push_str(raw);
        return Ok(());
    }
    write_complex_pad(out, values, pad_width)
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

fn raw_real_pad_matches(
    field: &'static str,
    raw: &str,
    pad_width: usize,
    expected: &[f64],
) -> Result<bool> {
    let raw_values = decode_raw_real_pad(field, raw, pad_width)?;
    Ok(raw_values.len() == expected.len()
        && raw_values
            .iter()
            .zip(expected.iter())
            .all(|(raw, typed)| raw == typed))
}

fn raw_complex_pad_matches(
    field: &'static str,
    raw: &str,
    pad_width: usize,
    expected: &[Complex64],
) -> Result<bool> {
    let raw_values = decode_raw_complex_pad(field, raw, pad_width)?;
    Ok(raw_values.len() == expected.len()
        && raw_values
            .iter()
            .zip(expected.iter())
            .all(|(raw, typed)| raw == typed))
}
