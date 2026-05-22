use super::*;

/// Port of FEFF `XSPH/axafs.f90`.
///
/// FEFF writes `axafs.dat` directly. This pure helper returns the same six data
/// columns as an `ndarray` table so callers can render the file or feed the
/// values into downstream analysis without filesystem side effects.
pub fn xsph_axafs(input: XsphAxafsInput<'_>) -> Result<XsphAxafs, XsphError> {
    validate_finite_real("fermi_energy", input.fermi_energy)?;
    validate_active_len("energies", input.energies.len(), input.horizontal_count)?;
    validate_active_len(
        "cross_section",
        input.cross_section.len(),
        input.horizontal_count,
    )?;
    if input.zero_wave_index >= input.horizontal_count {
        return Err(XsphError::InvalidAxafsGridIndex {
            zero_wave_index: input.zero_wave_index,
            horizontal_count: input.horizontal_count,
        });
    }
    let point_count = input.horizontal_count - input.zero_wave_index - 1;
    if point_count < 3 {
        return Err(XsphError::InsufficientAxafsPoints { point_count });
    }

    let reference_energy = input.energies[input.zero_wave_index];
    validate_finite_complex("energies", input.zero_wave_index, reference_energy)?;
    let mut energies = Vec::with_capacity(point_count);
    let mut absorption = Vec::with_capacity(point_count);
    for row in 0..point_count {
        let source = input.zero_wave_index + row + 1;
        let energy = input.energies[source];
        let cross_section = input.cross_section[source];
        validate_finite_complex("energies", source, energy)?;
        validate_finite_complex("cross_section", source, cross_section)?;
        energies.push((energy - reference_energy).re + input.fermi_energy);
        absorption.push(cross_section.im);
    }

    let weights = xsph_axafs_weights(&energies, input.fermi_energy);
    let mut moments = [0.0; 5];
    let mut rhs = [0.0; 3];
    for row in 0..point_count {
        let energy = energies[row];
        let weight = weights[row];
        for (power, moment) in moments.iter_mut().enumerate() {
            *moment += weight * energy.powi(power as i32);
        }
        for (power, value) in rhs.iter_mut().enumerate() {
            *value += weight * absorption[row] * energy.powi(power as i32);
        }
    }

    let coefficients = xsph_axafs_quadratic_coefficients(moments, rhs)?;
    let normalization_energy = energies[0] + 100.0 / XSPH_HARTREE_EV;
    let normalization = xsph_axafs_background(coefficients, normalization_energy);
    validate_finite_real("axafs normalization", normalization)?;
    if normalization == 0.0 {
        return Err(XsphError::ZeroAxafsNormalization);
    }

    let mut rows = Array2::<Real>::zeros((point_count, XSPH_AXAFS_COLUMN_COUNT));
    for row in 0..point_count {
        let energy = energies[row];
        let background = xsph_axafs_background(coefficients, energy);
        validate_finite_real("axafs background", background)?;
        if background == 0.0 {
            return Err(XsphError::ZeroAxafsBackground { index: row });
        }
        let relative_energy = energy - input.fermi_energy;
        let wave_number = if relative_energy >= 0.0 {
            (2.0 * relative_energy).sqrt() / XSPH_BOHR_ANGSTROM
        } else {
            -(-2.0 * relative_energy).sqrt() / XSPH_BOHR_ANGSTROM
        };
        let chi_atomic = (absorption[row] - background) / background;
        rows[(row, 0)] = energy * XSPH_HARTREE_EV;
        rows[(row, 1)] = relative_energy * XSPH_HARTREE_EV;
        rows[(row, 2)] = wave_number;
        rows[(row, 3)] = absorption[row] / normalization;
        rows[(row, 4)] = background / normalization;
        rows[(row, 5)] = chi_atomic;
        for column in 0..XSPH_AXAFS_COLUMN_COUNT {
            validate_finite_real("axafs row", rows[(row, column)])?;
        }
    }

    Ok(XsphAxafs {
        rows,
        coefficients,
        normalization,
    })
}
fn xsph_axafs_weights(energies: &[Real], fermi_energy: Real) -> Vec<Real> {
    let point_count = energies.len();
    (0..point_count)
        .map(|row| {
            if row == 0 {
                (energies[1] - fermi_energy) * (energies[0] - fermi_energy).abs()
            } else if row + 1 == point_count {
                (energies[row] - energies[row - 1]) * (energies[row] - fermi_energy)
            } else {
                (energies[row + 1] - energies[row - 1]) * (energies[row] - fermi_energy)
            }
        })
        .collect()
}

fn xsph_axafs_quadratic_coefficients(
    moments: [Real; 5],
    rhs: [Real; 3],
) -> Result<[Real; 3], XsphError> {
    let base = Array2::from_shape_fn((3, 3), |(row, column)| moments[row + column]);
    let denominator = feff_determinant(base.view())?;
    validate_finite_real("axafs fit denominator", denominator)?;
    if denominator == 0.0 {
        return Err(XsphError::SingularAxafsFit);
    }

    let mut coefficients = [0.0; 3];
    for (target_column, coefficient) in coefficients.iter_mut().enumerate() {
        let matrix = Array2::from_shape_fn((3, 3), |(row, column)| {
            if column == target_column {
                rhs[row]
            } else {
                moments[row + column]
            }
        });
        let numerator = feff_determinant(matrix.view())?;
        validate_finite_real("axafs fit numerator", numerator)?;
        *coefficient = numerator / denominator;
        validate_finite_real("axafs fit coefficient", *coefficient)?;
    }
    Ok(coefficients)
}

fn xsph_axafs_background(coefficients: [Real; 3], energy: Real) -> Real {
    coefficients[0] + coefficients[1] * energy + coefficients[2] * energy * energy
}
