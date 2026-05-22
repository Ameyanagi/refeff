use super::lambda::checked_i32;
use super::validation::*;
use super::*;

/// Build FEFF `snlm` associated-Legendre normalization factors.
///
/// The returned table uses FEFF's GENFMT layout `xnlm(il, im)` with Rust
/// indices `(l, m)`. Entries where `m > l` are retained as zeroes. The
/// implementation follows FEFF's scaled-factorial calculation so the values
/// match `snlm.f90` while keeping invalid dimensions as explicit errors.
pub fn genfmt_legendre_normalization_table(
    input: GenfmtLegendreNormalizationInput,
) -> Result<Array2<Real>, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;

    let l_max = input.lmaxp1 - 1;
    let m_max = input.mmaxp1 - 1;
    let active_m_max = l_max.min(m_max);
    let factorial_limit =
        l_max
            .checked_add(active_m_max)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "lmaxp1+mmaxp1",
                value: input.lmaxp1,
            })?;
    if factorial_limit > SNLM_FACTORIAL_LIMIT {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1+mmaxp1",
            value: factorial_limit,
        });
    }

    let scaled_factorials = snlm_scaled_factorials();
    let mut table = Array2::<Real>::zeros((input.lmaxp1, input.mmaxp1).f());
    for il in 1..=input.lmaxp1 {
        let active_m = input.mmaxp1.min(il);
        for im in 1..=active_m {
            let l = il - 1;
            let m = im - 1;
            let odd_factor = checked_double_plus_one("lmaxp1", l)? as Real;
            let value = (odd_factor * scaled_factorials[l - m] / scaled_factorials[l + m]).sqrt()
                * SNLM_AFAC.powi(checked_i32("mmaxp1", m)?);
            if !value.is_finite() {
                return Err(GenfmtError::NonFiniteScalar {
                    field: "xnlm",
                    value,
                });
            }
            table[(l, m)] = value;
        }
    }

    Ok(table)
}

/// Build FEFF `sclmz` curved-wave Rehr-Albers polynomial factors.
///
/// FEFF stores the result in `clmi(il, im, ileg)`. This Rust helper returns the
/// active two-dimensional leg table in Fortran-order ndarray storage, with
/// FEFF one-based indices mapped to Rust `(il - 1, im - 1)`. The row dimension
/// is `lmaxp1 + 1` because FEFF fills the `im + 1` row for diagonal magnetic
/// recurrences; the column dimension is the requested `mmaxp1`, with columns
/// above `lmaxp1` left at zero.
pub fn curved_wave_polynomials(
    input: CurvedWavePolynomialInput,
) -> Result<Array2<Complex>, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;
    validate_finite_complex("rho", input.rho)?;
    if input.rho == Complex::new(0.0, 0.0) {
        return Err(GenfmtError::ZeroComplex { field: "rho" });
    }

    let rows = input
        .lmaxp1
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?;
    let mut table = Array2::zeros((rows, input.mmaxp1).f());
    let one = Complex::new(1.0, 0.0);
    let z = -Complex::new(0.0, 1.0) / input.rho;

    table[(0, 0)] = one;
    table[(1, 0)] = table[(0, 0)] - z;

    let lmax = input.lmaxp1 - 1;
    for il in 2..=lmax {
        table[(il, 0)] = table[(il - 2, 0)]
            - z * checked_odd_factor(il, "lmaxp1", input.lmaxp1)? * table[(il - 1, 0)];
    }

    let mut cmm = one;
    let mmxp1 = input.mmaxp1.min(input.lmaxp1);
    for im in 2..=mmxp1 {
        let m = im - 1;
        cmm = -cmm * checked_odd_factor(m, "mmaxp1", input.mmaxp1)? * z;
        table[(im - 1, im - 1)] = cmm;
        table[(im, im - 1)] =
            cmm * checked_odd_factor(im, "mmaxp1", input.mmaxp1)? * (one - (im as Real) * z);

        for il in (im + 1)..=lmax {
            let l = il - 1;
            table[(il, im - 1)] = table[(l - 1, im - 1)]
                - checked_odd_factor(il, "lmaxp1", input.lmaxp1)?
                    * z
                    * (table[(il - 1, im - 1)] + table[(il - 1, m - 1)]);
        }
    }

    Ok(table)
}

fn snlm_scaled_factorials() -> [Real; SNLM_FACTORIAL_COUNT] {
    let mut factors = [0.0; SNLM_FACTORIAL_COUNT];
    factors[0] = 1.0;
    factors[1] = SNLM_AFAC;
    for i in 2..=SNLM_FACTORIAL_LIMIT {
        factors[i] = factors[i - 1] * (i as Real) * SNLM_AFAC;
    }
    factors
}

pub(super) fn alternating_sign(power: usize) -> Real {
    if power.is_multiple_of(2) { 1.0 } else { -1.0 }
}
