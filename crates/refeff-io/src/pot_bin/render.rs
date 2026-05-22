use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, ArrayView2, ArrayView3};

use crate::error::{IoError, Result};
use crate::format::repeated_ints;
use crate::pad::encode_reals;

use super::common::{check_fixed_int, i64_from_usize, validate_finite_values};
use super::parse::parse_pot_bin;
use super::types::{POT_BIN_IORB_SLOTS, PotBinData};
use super::validate::validate_pot_bin;

/// Render FEFF `pot.bin` text.
pub fn pot_bin_string(data: &PotBinData) -> Result<String> {
    validate_pot_bin(data)?;

    if let Some(raw_text) = &data.raw_text
        && raw_pot_bin_matches(data, raw_text)?
    {
        return Ok(raw_text.clone());
    }

    let potential_count = data.potential_count();
    let mut out = String::new();
    write_int_line(
        &mut out,
        &[
            i64_from_usize(data.titles.len(), "ntitle")?,
            i64_from_usize(potential_count - 1, "nph")?,
            i64_from_usize(data.pad_width, "npadx")?,
            i64::from(data.nohole),
            i64::from(data.ihole),
            i64::from(data.interstitial_selector),
            i64::from(data.automatic_folp),
            i64::from(data.jump_mode),
            i64::from(data.unfreeze_f),
        ],
        4,
    )?;

    for title in &data.titles {
        writeln!(out, "{title}")?;
    }

    write_pad_values(&mut out, &data.scalars.as_array(), data.pad_width)?;
    write_i4_chunks(
        &mut out,
        &usize_array_to_i64("imt", &data.muffin_tin_indices)?,
    )?;
    write_pad_values(
        &mut out,
        &data.muffin_tin_radii.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_i4_chunks(&mut out, &usize_array_to_i64("inrm", &data.norman_indices)?)?;
    write_i4_chunks(&mut out, &usize_array_to_i64("iz", &data.atomic_numbers)?)?;
    write_i4_chunks(&mut out, &i32_array_to_i64(&data.kappa))?;

    for (field, values) in [
        ("rnrm", data.norman_radii.view()),
        ("folp", data.overlap_factors.view()),
        ("folpx", data.max_overlap_factors.view()),
        ("xnatph", data.potential_multiplicities.view()),
        ("xion", data.ionization.view()),
        ("dgc0", data.initial_large_component.view()),
        ("dpc0", data.initial_small_component.view()),
    ] {
        let flat = values.iter().copied().collect::<Vec<_>>();
        validate_finite_values(field, flat.iter().copied())?;
        write_pad_values(&mut out, &flat, data.pad_width)?;
    }

    for values in [
        data.large_components.view(),
        data.small_components.view(),
        data.large_coefficients.view(),
        data.small_coefficients.view(),
    ] {
        write_pad_values(&mut out, &flatten3(values), data.pad_width)?;
    }

    for values in [
        data.electron_density.view(),
        data.coulomb_potential.view(),
        data.total_potential.view(),
        data.valence_density.view(),
        data.valence_potential.view(),
        data.magnetization_density.view(),
        data.orbital_occupancy.view(),
    ] {
        write_pad_values(&mut out, &flatten2(values), data.pad_width)?;
    }

    write_pad_values(
        &mut out,
        &data.orbital_energies.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;

    for potential in 0..potential_count {
        let mut values = Vec::with_capacity(POT_BIN_IORB_SLOTS);
        for slot in 0..POT_BIN_IORB_SLOTS {
            values.push(i64::from(data.occupied_orbital_indices[(slot, potential)]));
        }
        write_i2_chunks(&mut out, &values)?;
    }

    write_pad_values(
        &mut out,
        &data.norman_charges.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_pad_values(
        &mut out,
        &flatten2(data.valence_occupancy.view()),
        data.pad_width,
    )?;
    Ok(out)
}

/// Write FEFF `pot.bin` text to a file.
pub fn write_pot_bin(path: impl AsRef<Path>, data: &PotBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, pot_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

fn raw_pot_bin_matches(data: &PotBinData, raw_text: &str) -> Result<bool> {
    let mut parsed = parse_pot_bin(raw_text)?;
    parsed.raw_text = None;
    let mut expected = data.clone();
    expected.raw_text = None;
    Ok(parsed == expected)
}

fn write_i4_chunks(out: &mut String, values: &[i64]) -> Result<()> {
    for chunk in values.chunks(20) {
        write_int_line(out, chunk, 4)?;
    }
    Ok(())
}

fn write_i2_chunks(out: &mut String, values: &[i64]) -> Result<()> {
    for chunk in values.chunks(8) {
        write_int_line(out, chunk, 2)?;
    }
    Ok(())
}

fn write_int_line(out: &mut String, values: &[i64], width: usize) -> Result<()> {
    for &value in values {
        check_fixed_int(value, width, "integer")?;
    }
    writeln!(out, "{}", repeated_ints(values.iter().copied(), width))?;
    Ok(())
}

fn write_pad_values(out: &mut String, values: &[f64], pad_width: usize) -> Result<()> {
    out.push_str(&encode_reals(values, pad_width)?);
    Ok(())
}

fn flatten2(values: ArrayView2<'_, f64>) -> Vec<f64> {
    let (rows, cols) = values.dim();
    let mut flat = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            flat.push(values[(row, col)]);
        }
    }
    flat
}

fn flatten3(values: ArrayView3<'_, f64>) -> Vec<f64> {
    let (rows, cols, planes) = values.dim();
    let mut flat = Vec::with_capacity(rows * cols * planes);
    for plane in 0..planes {
        for col in 0..cols {
            for row in 0..rows {
                flat.push(values[(row, col, plane)]);
            }
        }
    }
    flat
}

fn usize_array_to_i64(field: &'static str, values: &Array1<usize>) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&value| i64_from_usize(value, field))
        .collect()
}

fn i32_array_to_i64(values: &Array1<i32>) -> Vec<i64> {
    values.iter().map(|&value| i64::from(value)).collect()
}
