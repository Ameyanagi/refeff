use num_complex::Complex64;

use crate::error::{IoError, Result};

use super::common::{
    check_fixed_int, checked_l_count, i64_from_usize, invalid_phase_bin, validate_label,
};
use super::types::PhaseBinData;

pub(super) fn validate_phase_bin(data: &PhaseBinData) -> Result<()> {
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
