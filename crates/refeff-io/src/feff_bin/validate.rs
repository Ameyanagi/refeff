use num_complex::Complex64;

use crate::error::{IoError, Result};

use super::common::{check_fixed_int, i64_from_usize, invalid_feff_bin, validate_label};
use super::types::{FeffBinData, FeffBinPath};

pub(super) fn validate_feff_bin(data: &FeffBinData) -> Result<()> {
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
