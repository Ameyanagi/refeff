use std::fmt::Write as _;
use std::path::Path;

use ndarray::ArrayView2;
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::exp;
use crate::pad::{encode_complex, encode_reals};

use super::common::{check_fixed_int, i64_from_usize, validate_label};
use super::parse::parse_feff_bin;
use super::types::{FEFF_BIN_BOHR, FeffBinData, FeffBinPath, FeffBinPotential};
use super::validate::validate_feff_bin;

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

fn raw_feff_bin_matches(data: &FeffBinData, raw_text: &str) -> Result<bool> {
    let mut parsed = parse_feff_bin(raw_text)?;
    parsed.raw_text = None;
    let mut expected = data.clone();
    expected.raw_text = None;
    Ok(parsed == expected)
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
