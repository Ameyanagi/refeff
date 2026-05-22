use super::*;

/// Return FEFF's relativistic electron wavelength in atomic units.
///
/// This ports `EELS/wavelength.f90`:
/// `HOnSqrtTwoMe / sqrt(E + E**2 / (2 * MeC2))`, with `E` in eV.
pub fn electron_wavelength_atomic_units(energy_ev: Real) -> Result<Real, EelsError> {
    validate_finite("energy_ev", energy_ev)?;
    if energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy { value: energy_ev });
    }

    let denominator =
        (energy_ev + energy_ev * energy_ev / (2.0 * FEFF_ELECTRON_REST_ENERGY_EV)).sqrt();
    let wavelength = FEFF_H_ON_SQRT_TWO_ME / denominator;
    if !wavelength.is_finite() {
        return Err(EelsError::NonFiniteResult {
            name: "wavelength",
            value: wavelength,
        });
    }
    Ok(wavelength)
}

/// Build FEFF's EELS Euler rotation matrix.
///
/// The three angles correspond to FEFF `a`, `b`, and `g`. The returned matrix
/// is shaped `(3, 3)` in Fortran-order `ndarray` storage and preserves the
/// `E(row,column)` assignments in `EELS/euler.f90`.
pub fn eels_euler_rotation_matrix(
    alpha: Real,
    beta: Real,
    gamma: Real,
) -> Result<RealMat, EelsError> {
    validate_finite("alpha", alpha)?;
    validate_finite("beta", beta)?;
    validate_finite("gamma", gamma)?;

    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let (sin_beta, cos_beta) = beta.sin_cos();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();

    let mut matrix = Array2::zeros((3, 3).f());
    matrix[(0, 0)] = cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma;
    matrix[(1, 0)] = sin_alpha * cos_beta * cos_gamma + cos_alpha * sin_gamma;
    matrix[(0, 1)] = -cos_alpha * cos_beta * sin_gamma - sin_alpha * cos_gamma;
    matrix[(1, 1)] = -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma;
    matrix[(0, 2)] = cos_alpha * sin_beta;
    matrix[(1, 2)] = sin_alpha * sin_beta;
    matrix[(2, 2)] = cos_beta;
    matrix[(2, 0)] = -sin_beta * cos_gamma;
    matrix[(2, 1)] = sin_beta * sin_gamma;

    for &value in &matrix {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteResult {
                name: "euler_matrix",
                value,
            });
        }
    }
    Ok(matrix)
}

/// Port of FEFF `EELS/productmatvect.f90`.
///
/// Multiplies a `3 x 3` matrix by a 3-vector with FEFF's row/column
/// convention: `Vout(i) = sum_k M(i,k) * Vin(k)`.
pub fn eels_product_matrix_vector(
    matrix: ArrayView2<'_, Real>,
    vector: ArrayView1<'_, Real>,
) -> Result<RealVec, EelsError> {
    let (rows, columns) = matrix.dim();
    if (rows, columns) != (3, 3) {
        return Err(EelsError::InvalidMatrixShape { rows, columns });
    }
    if vector.len() != 3 {
        return Err(EelsError::InvalidVectorLength {
            length: vector.len(),
        });
    }
    for &value in matrix.iter() {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteInput {
                name: "matrix",
                value,
            });
        }
    }
    for &value in &vector {
        validate_finite("vector", value)?;
    }

    let product = Array1::from_shape_fn(3, |row| {
        (0..3)
            .map(|column| matrix[(row, column)] * vector[column])
            .sum::<Real>()
    });
    for &value in &product {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteResult {
                name: "matrix_vector_product",
                value,
            });
        }
    }
    Ok(product)
}
