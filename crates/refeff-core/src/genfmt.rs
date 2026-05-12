//! FEFF `GENFMT` helper routines.
//!
//! This module ports small, self-contained setup routines used by FEFF's
//! curved-wave multiple-scattering formatter. `lambda_indices` is the Rust
//! equivalent of `GENFMT/setlam.f90`: it builds the Rehr-Albers lambda index
//! arrays `(m, n)` from FEFF's `icalc` mode, path order, and dimension limits.

use ndarray::{Array1, Array2, Array3, ShapeBuilder};
use thiserror::Error;

use crate::{Complex, Real};

const ONE_DEGREE_RADIANS: Real = 0.017_453_292_52;

/// Inputs for FEFF `GENFMT/rot3i.f90` initial-state rotation matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialStateRotationInput {
    /// FEFF `lxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mxp1`, equal to `mmax + 1`.
    pub mmaxp1: usize,
    /// FEFF `beta(ileg)` scattering angle in radians.
    pub beta_angle: Real,
}

/// Inputs for FEFF `GENFMT/setlam.f90` lambda-index selection.
#[derive(Debug, Clone, Copy)]
pub struct LambdaIndexInput<'a> {
    /// FEFF `icalc` selector: `0..=9` for exact order, `10` for the cute
    /// heuristic, or a negative encoded `(nmax, mmax, iord)` request.
    pub calculation: i32,
    /// FEFF one-based energy index `ie`; the cute heuristic raises `nmax` for
    /// `ie >= 42`.
    pub energy_index: usize,
    /// FEFF `nsc`, used to detect single-scattering paths.
    pub scattering_count: usize,
    /// FEFF `ilinit`, the initial-state angular momentum.
    pub initial_l: usize,
    /// FEFF `beta(1:nleg)` path scattering angles in radians.
    pub beta_angles: &'a [Real],
    /// FEFF `lamtot`, the capacity of `mlam` and `nlam`.
    pub lambda_capacity: usize,
    /// FEFF `mtot`, the maximum magnetic index dimension.
    pub max_m: usize,
    /// FEFF `ntot`, the maximum order index dimension.
    pub max_n: usize,
}

/// Inputs for FEFF `GENFMT/xstar.f90` central-atom plane-wave factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XStarInput {
    /// FEFF `eps1`: primary polarization vector.
    pub primary_polarization: [Real; 3],
    /// FEFF `eps2`: secondary polarization vector for elliptic polarization.
    pub secondary_polarization: [Real; 3],
    /// FEFF `vec1`: direction to the first atom in the path.
    pub first_leg: [Real; 3],
    /// FEFF `vec2`: direction to the last atom in the path.
    pub last_leg: [Real; 3],
    /// FEFF `ndeg`, the path degeneracy used for this approximation.
    pub degeneracy: Real,
    /// FEFF `ilinit`, supported by the embedded Legendre table for `1..=4`.
    pub initial_l: usize,
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
}

/// Inputs for FEFF `GENFMT/sclmz.f90` curved-wave polynomial tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvedWavePolynomialInput {
    /// FEFF `lmaxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mmaxp1`; columns above `lmaxp1` are retained as zeroes.
    pub mmaxp1: usize,
    /// FEFF complex path length `rho(ileg)`.
    pub rho: Complex,
}

/// Compact FEFF `rot3i` rotation table for one path leg.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialStateRotation {
    /// FEFF `dri(il,m1+mtot+1,m2+mtot+1,ileg)` without unused global padding.
    ///
    /// Rust indices are `(il - 1, m1 + magnetic_offset, m2 + magnetic_offset)`.
    pub matrix: Array3<Real>,
    /// Offset added to signed magnetic indices before indexing `matrix`.
    pub magnetic_offset: usize,
}

/// FEFF lambda index arrays and associated `setlam` metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaIndexSet {
    /// FEFF `mlam(1:lamx)` magnetic indices.
    pub m_indices: Array1<i32>,
    /// FEFF `nlam(1:lamx)` order indices.
    pub n_indices: Array1<i32>,
    /// FEFF `laml0x`: prefix count whose entries are within `ilinit`.
    pub initial_l_prefix_len: usize,
    /// FEFF `mmaxp1`, computed after capacity truncation and ordering.
    pub max_m_plus_one: usize,
    /// FEFF final `nmax`, computed after capacity truncation and ordering.
    pub max_n: usize,
    /// FEFF `iord`, the requested Rehr-Albers order.
    pub order: i32,
    /// Requested `nmax` before lambda-capacity truncation.
    pub requested_n_max: usize,
    /// Requested `mmax` before lambda-capacity truncation.
    pub requested_m_max: usize,
    /// Whether FEFF would have logged `Lambda array filled, some order lost`.
    pub truncated: bool,
}

/// Error returned by FEFF `GENFMT` helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GenfmtError {
    /// FEFF only defines nonnegative `icalc` values through `10`.
    #[error("undefined FEFF lambda calculation {calculation}")]
    UndefinedLambdaCalculation { calculation: i32 },
    /// A negative `icalc` could not be decoded safely.
    #[error("lambda calculation code {calculation} cannot be decoded safely")]
    LambdaCodeOverflow { calculation: i32 },
    /// The cute heuristic needs finite beta angles.
    #[error("beta angle at index {index} must be finite, got {value}")]
    NonFiniteBetaAngle { index: usize, value: Real },
    /// A generated FEFF integer field would overflow.
    #[error("lambda field {field}={value} does not fit in i32")]
    IntegerOverflow { field: &'static str, value: usize },
    /// GENFMT angular limits must be positive and fit index calculations.
    #[error("invalid GENFMT angular limit {name}={value}")]
    InvalidAngularLimit { name: &'static str, value: usize },
    /// FEFF `rot3i` requires a finite beta angle.
    #[error("rotation beta angle must be finite")]
    NonFiniteRotationAngle,
    /// FEFF `sclmz` needs a finite complex path length.
    #[error("{field} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        field: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// FEFF `sclmz` divides by the complex path length.
    #[error("{field} must be nonzero")]
    ZeroComplex { field: &'static str },
    /// FEFF `xstar` only tabulates Legendre coefficients through `ilinit=4`.
    #[error("initial angular momentum {initial_l} is outside GENFMT xstar table range 1..=4")]
    InvalidInitialAngularMomentum { initial_l: usize },
    /// Scalar GENFMT inputs must be finite.
    #[error("{field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Vector GENFMT inputs must have finite components.
    #[error("{field}[{index}] must be finite, got {value}")]
    NonFiniteVector {
        field: &'static str,
        index: usize,
        value: Real,
    },
    /// FEFF `xxcos` is undefined for zero-length vectors.
    #[error("{field} must have nonzero length")]
    ZeroVector { field: &'static str },
    /// Generated lambda indices exceed the caller's FEFF dimensions.
    #[error(
        "lambda selection exceeded dimensions: mmaxp1={max_m_plus_one}, nmax={max_n}, mtot={max_m}, ntot={max_n_limit}"
    )]
    DimensionExceeded {
        max_m_plus_one: usize,
        max_n: usize,
        max_m: usize,
        max_n_limit: usize,
    },
}

/// Build FEFF `rot3i` real rotation matrices for a single path leg.
///
/// The recursion is the Edmonds small-`d` rotation used by FEFF before GENFMT
/// matrix assembly. FEFF writes into a globally padded `dri` array; this helper
/// returns only the active magnetic range `-(mxp1-1)..=(mxp1-1)` for each
/// `il`, with zeroes retained where FEFF would not fill entries.
pub fn initial_state_rotation(
    input: InitialStateRotationInput,
) -> Result<InitialStateRotation, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;
    if !input.beta_angle.is_finite() {
        return Err(GenfmtError::NonFiniteRotationAngle);
    }

    let magnetic_offset = input.mmaxp1 - 1;
    let m_dim = checked_double_plus_one("mmaxp1", magnetic_offset)?;
    let mut matrix = Array3::<Real>::zeros((input.lmaxp1, m_dim, m_dim).f());

    let work_l = input.lmaxp1.max(2);
    let ndm = input
        .lmaxp1
        .checked_add(input.mmaxp1)
        .and_then(|value| value.checked_sub(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?;
    let work_m = checked_double_plus_one("lmaxp1", work_l)?
        .checked_sub(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?
        .max(ndm)
        .max(3);
    let mut work = Array3::<Real>::zeros((work_l + 1, work_m + 1, work_m + 1).f());
    fill_initial_state_rotation_work(input.lmaxp1, input.mmaxp1, input.beta_angle, &mut work);
    copy_initial_state_rotation(
        input.lmaxp1,
        input.mmaxp1,
        magnetic_offset,
        &work,
        &mut matrix,
    )?;

    Ok(InitialStateRotation {
        matrix,
        magnetic_offset,
    })
}

/// Compute FEFF `xstar`, the central-atom plane-wave polarization factor.
///
/// FEFF evaluates the orientationally averaged `ystar` expression for the
/// primary polarization and, when `ellipticity != 0`, adds the secondary
/// polarization weighted by `ellipticity^2`. The vector cosines match
/// `xxcos` from `xstar.f90`, but zero-length and non-finite inputs are reported
/// as errors instead of allowing division by zero.
pub fn xstar(input: XStarInput) -> Result<Real, GenfmtError> {
    if !(1..=4).contains(&input.initial_l) {
        return Err(GenfmtError::InvalidInitialAngularMomentum {
            initial_l: input.initial_l,
        });
    }
    validate_finite_scalar("degeneracy", input.degeneracy)?;
    validate_finite_scalar("ellipticity", input.ellipticity)?;

    let x = normalized_dot("first_leg", input.first_leg, "last_leg", input.last_leg)?;
    let primary_y = normalized_dot(
        "primary_polarization",
        input.primary_polarization,
        "first_leg",
        input.first_leg,
    )?;
    let primary_z = normalized_dot(
        "primary_polarization",
        input.primary_polarization,
        "last_leg",
        input.last_leg,
    )?;
    let mut value = ystar(input.initial_l, x, primary_y, primary_z);

    if input.ellipticity != 0.0 {
        let secondary_y = normalized_dot(
            "secondary_polarization",
            input.secondary_polarization,
            "first_leg",
            input.first_leg,
        )?;
        let secondary_z = normalized_dot(
            "secondary_polarization",
            input.secondary_polarization,
            "last_leg",
            input.last_leg,
        )?;
        value += input.ellipticity
            * input.ellipticity
            * ystar(input.initial_l, x, secondary_y, secondary_z);
    }

    Ok(input.degeneracy * value / (1.0 + input.ellipticity * input.ellipticity))
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

/// Build FEFF `mlam` and `nlam` arrays from `GENFMT/setlam.f90` rules.
///
/// The returned arrays preserve FEFF's insertion order, including `-m` before
/// `+m`, and then apply FEFF's second pass that moves entries satisfying
/// `n <= ilinit && abs(m) <= ilinit` to the front to minimize `laml0x`.
/// Capacity handling also follows FEFF: if `lamtot` fills, the result is
/// truncated and flagged instead of failing.
pub fn lambda_indices(input: LambdaIndexInput<'_>) -> Result<LambdaIndexSet, GenfmtError> {
    let request = lambda_request(input)?;
    let mut raw = Vec::with_capacity(input.lambda_capacity.min(128));
    let mut truncated = false;

    if request.order >= 0 {
        let order = usize::try_from(request.order).map_err(|_| GenfmtError::IntegerOverflow {
            field: "iord",
            value: request.order.unsigned_abs() as usize,
        })?;
        let valid_n_max = request.n_max.min(order / 2);

        'outer: for n in 0..=valid_n_max {
            let valid_m_max = request.m_max.min(order - 2 * n);
            for m in 0..=valid_m_max {
                if raw.len() >= input.lambda_capacity {
                    truncated = true;
                    break 'outer;
                }
                raw.push((-checked_i32("m", m)?, checked_i32("n", n)?));

                if m != 0 {
                    if raw.len() >= input.lambda_capacity {
                        truncated = true;
                        break 'outer;
                    }
                    raw.push((checked_i32("m", m)?, checked_i32("n", n)?));
                }
            }
        }
    }

    let mut pairs = Vec::with_capacity(raw.len());
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| within_initial_l(m, n, input.initial_l)),
    );
    let initial_l_prefix_len = pairs.len();
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| !within_initial_l(m, n, input.initial_l)),
    );

    let max_m_plus_one = pairs
        .iter()
        .filter_map(|&(m, _)| usize::try_from(m.saturating_add(1)).ok())
        .max()
        .unwrap_or(0);
    let max_n = pairs
        .iter()
        .filter_map(|&(_, n)| usize::try_from(n).ok())
        .max()
        .unwrap_or(0);

    if max_n > input.max_n || max_m_plus_one > input.max_m.saturating_add(1) {
        return Err(GenfmtError::DimensionExceeded {
            max_m_plus_one,
            max_n,
            max_m: input.max_m,
            max_n_limit: input.max_n,
        });
    }

    let (m_values, n_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok(LambdaIndexSet {
        m_indices: Array1::from_vec(m_values),
        n_indices: Array1::from_vec(n_values),
        initial_l_prefix_len,
        max_m_plus_one,
        max_n,
        order: request.order,
        requested_n_max: request.n_max,
        requested_m_max: request.m_max,
        truncated,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LambdaRequest {
    order: i32,
    n_max: usize,
    m_max: usize,
}

fn lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    if input.calculation < 0 {
        return decode_lambda_request(input.calculation);
    }

    if input.scattering_count == 1 {
        return Ok(LambdaRequest {
            order: checked_order(input.initial_l, input.initial_l)?,
            n_max: input.initial_l,
            m_max: input.initial_l,
        });
    }

    if input.calculation < 10 {
        let order = input.calculation;
        return Ok(LambdaRequest {
            order,
            n_max: usize::try_from(order / 2).map_err(|_| GenfmtError::IntegerOverflow {
                field: "nmax",
                value: order.unsigned_abs() as usize,
            })?,
            m_max: usize::try_from(order).map_err(|_| GenfmtError::IntegerOverflow {
                field: "mmax",
                value: order.unsigned_abs() as usize,
            })?,
        });
    }

    if input.calculation == 10 {
        return cute_lambda_request(input);
    }

    Err(GenfmtError::UndefinedLambdaCalculation {
        calculation: input.calculation,
    })
}

fn decode_lambda_request(calculation: i32) -> Result<LambdaRequest, GenfmtError> {
    let code = calculation
        .checked_neg()
        .ok_or(GenfmtError::LambdaCodeOverflow { calculation })?;
    let order = (code / 10_000) - 1;
    Ok(LambdaRequest {
        order,
        n_max: usize::try_from(code % 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
        m_max: usize::try_from((code % 10_000) / 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
    })
}

fn cute_lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    let mut m_max = input.initial_l;
    for (index, &angle) in input.beta_angles.iter().enumerate() {
        if !angle.is_finite() {
            return Err(GenfmtError::NonFiniteBetaAngle {
                index,
                value: angle,
            });
        }
        let magnitude = angle.abs();
        let pi_distance = (magnitude - std::f64::consts::PI).abs();
        if magnitude > ONE_DEGREE_RADIANS && pi_distance > ONE_DEGREE_RADIANS {
            m_max = 3;
        }
    }

    let n_max = if input.energy_index >= 42 {
        9
    } else {
        input.initial_l
    };

    Ok(LambdaRequest {
        order: checked_order(n_max, m_max)?,
        n_max,
        m_max,
    })
}

fn checked_order(n_max: usize, m_max: usize) -> Result<i32, GenfmtError> {
    let order = n_max
        .checked_mul(2)
        .and_then(|value| value.checked_add(m_max))
        .ok_or(GenfmtError::IntegerOverflow {
            field: "iord",
            value: n_max,
        })?;
    checked_i32("iord", order)
}

fn checked_i32(field: &'static str, value: usize) -> Result<i32, GenfmtError> {
    i32::try_from(value).map_err(|_| GenfmtError::IntegerOverflow { field, value })
}

fn within_initial_l(m: i32, n: i32, initial_l: usize) -> bool {
    let abs_m = m.unsigned_abs() as usize;
    let Ok(n) = usize::try_from(n) else {
        return false;
    };
    n <= initial_l && abs_m <= initial_l
}

fn validate_positive_limit(name: &'static str, value: usize) -> Result<(), GenfmtError> {
    if value == 0 || isize::try_from(value).is_err() {
        Err(GenfmtError::InvalidAngularLimit { name, value })
    } else {
        Ok(())
    }
}

fn checked_double_plus_one(name: &'static str, value: usize) -> Result<usize, GenfmtError> {
    value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit { name, value })
}

fn checked_odd_factor(value: usize, name: &'static str, limit: usize) -> Result<Real, GenfmtError> {
    let factor = value.checked_mul(2).and_then(|value| value.checked_sub(1));
    factor
        .map(|value| value as Real)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value: limit })
}

fn fill_initial_state_rotation_work(
    lmaxp1: usize,
    mmaxp1: usize,
    beta: Real,
    work: &mut Array3<Real>,
) {
    let ndm = lmaxp1 + mmaxp1 - 1;
    let half_beta = beta / 2.0;
    let xc = half_beta.cos();
    let xs = half_beta.sin();
    let s = beta.sin();

    work[(1, 1, 1)] = 1.0;
    work[(2, 1, 1)] = xc * xc;
    work[(2, 1, 2)] = s / 2.0_f64.sqrt();
    work[(2, 1, 3)] = xs * xs;
    work[(2, 2, 1)] = -work[(2, 1, 2)];
    work[(2, 2, 2)] = beta.cos();
    work[(2, 2, 3)] = work[(2, 1, 2)];
    work[(2, 3, 1)] = work[(2, 1, 3)];
    work[(2, 3, 2)] = -work[(2, 2, 3)];
    work[(2, 3, 3)] = work[(2, 1, 1)];

    for l in 3..=lmaxp1 {
        let ln = (2 * l - 1).min(ndm);
        let lm = (2 * l - 3).min(ndm);
        for n in 1..=ln {
            for m in 1..=lm {
                let l_signed = l as isize;
                let n_signed = n as isize;
                let m_signed = m as isize;
                let t1 = ((2 * l_signed - 1 - n_signed) * (2 * l_signed - 2 - n_signed)) as Real;
                let t = ((2 * l_signed - 1 - m_signed) * (2 * l_signed - 2 - m_signed)) as Real;
                let f1 = (t1 / t).sqrt();
                let f2 = (((2 * l_signed - 1 - n_signed) * (n_signed - 1)) as Real / t).sqrt();
                let f3 = if n > 2 {
                    (((n - 2) * (n - 1)) as Real / t).sqrt()
                } else {
                    0.0
                };

                let mut dlnm = f1 * xc * xc * work[(l - 1, n, m)];
                if n > 1 {
                    dlnm -= f2 * s * work[(l - 1, n - 1, m)];
                }
                if n > 2 {
                    dlnm += f3 * xs * xs * work[(l - 1, n - 2, m)];
                }
                work[(l, n, m)] = dlnm;

                if n > 2 * l - 3 {
                    work[(l, m, n)] = alternating_sign(n - m) * dlnm;
                }
            }

            if n > 2 * l - 3 {
                work[(l, 2 * l - 2, 2 * l - 2)] = work[(l, 2, 2)];
                work[(l, 2 * l - 1, 2 * l - 2)] = -work[(l, 1, 2)];
                work[(l, 2 * l - 2, 2 * l - 1)] = -work[(l, 2, 1)];
                work[(l, 2 * l - 1, 2 * l - 1)] = work[(l, 1, 1)];
            }
        }
    }
}

fn copy_initial_state_rotation(
    lmaxp1: usize,
    mmaxp1: usize,
    magnetic_offset: usize,
    work: &Array3<Real>,
    matrix: &mut Array3<Real>,
) -> Result<(), GenfmtError> {
    let magnetic_offset =
        isize::try_from(magnetic_offset).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
        })?;

    for il in 1..=lmaxp1 {
        let mx = (il - 1).min(mmaxp1 - 1);
        let mx_signed = isize::try_from(mx).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
        })?;
        let il_signed = isize::try_from(il).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: lmaxp1,
        })?;

        for m1_slot in 0..=(2 * mx) {
            let m1 = isize::try_from(m1_slot).map_err(|_| GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: mmaxp1,
            })? - mx_signed;
            for m2_slot in 0..=(2 * mx) {
                let m2 =
                    isize::try_from(m2_slot).map_err(|_| GenfmtError::InvalidAngularLimit {
                        name: "mmaxp1",
                        value: mmaxp1,
                    })? - mx_signed;
                let row = shifted_index(m1, magnetic_offset, "mmaxp1", mmaxp1)?;
                let column = shifted_index(m2, magnetic_offset, "mmaxp1", mmaxp1)?;
                let work_row = shifted_index(m1, il_signed, "lmaxp1", lmaxp1)?;
                let work_column = shifted_index(m2, il_signed, "lmaxp1", lmaxp1)?;
                matrix[(il - 1, row, column)] = work[(il, work_row, work_column)];
            }
        }
    }
    Ok(())
}

fn shifted_index(
    value: isize,
    offset: isize,
    name: &'static str,
    limit: usize,
) -> Result<usize, GenfmtError> {
    let index = value
        .checked_add(offset)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value: limit })?;
    usize::try_from(index).map_err(|_| GenfmtError::InvalidAngularLimit { name, value: limit })
}

fn alternating_sign(power: usize) -> Real {
    if power.is_multiple_of(2) { 1.0 } else { -1.0 }
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), GenfmtError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GenfmtError::NonFiniteScalar { field, value })
    }
}

fn validate_finite_complex(field: &'static str, value: Complex) -> Result<(), GenfmtError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(GenfmtError::NonFiniteComplex {
            field,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn normalized_dot(
    left_field: &'static str,
    left: [Real; 3],
    right_field: &'static str,
    right: [Real; 3],
) -> Result<Real, GenfmtError> {
    validate_vector(left_field, left)?;
    validate_vector(right_field, right)?;

    let dot = left.iter().zip(right).map(|(&a, b)| a * b).sum::<Real>();
    let left_norm = left.iter().map(|value| value * value).sum::<Real>();
    let right_norm = right.iter().map(|value| value * value).sum::<Real>();

    if left_norm == 0.0 {
        return Err(GenfmtError::ZeroVector { field: left_field });
    }
    if right_norm == 0.0 {
        return Err(GenfmtError::ZeroVector { field: right_field });
    }

    Ok(dot / (left_norm * right_norm).sqrt())
}

fn validate_vector(field: &'static str, vector: [Real; 3]) -> Result<(), GenfmtError> {
    for (index, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn ystar(initial_l: usize, x: Real, y: Real, z: Real) -> Real {
    const LEGENDRE: [[Real; 5]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.5, 0.0, 1.5, 0.0, 0.0],
        [0.0, -1.5, 0.0, 2.5, 0.0],
        [0.375, 0.0, -3.75, 0.0, 4.375],
    ];
    let coefficients = LEGENDRE[initial_l];
    let l = initial_l as Real;

    let pln0 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .map(|(power, coefficient)| coefficient * x.powi(power as i32))
        .sum::<Real>();
    let pln1 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .skip(1)
        .map(|(power, coefficient)| {
            let power_real = power as Real;
            coefficient * power_real * x.powi(power as i32 - 1)
        })
        .sum::<Real>();
    let pln2 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .skip(2)
        .map(|(power, coefficient)| {
            let power_real = power as Real;
            coefficient * power_real * (power_real - 1.0) * x.powi(power as i32 - 2)
        })
        .sum::<Real>();

    let ytemp = -l * pln0 + pln1 * (x + y * z) - pln2 * (y * y + z * z - 2.0 * x * y * z);
    ytemp * 3.0 / l / (4.0 * l * l - 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        CurvedWavePolynomialInput, GenfmtError, InitialStateRotation, InitialStateRotationInput,
        LambdaIndexInput, XStarInput, curved_wave_polynomials, initial_state_rotation,
        lambda_indices, xstar,
    };
    use crate::Complex;

    fn input<'a>(
        calculation: i32,
        energy_index: usize,
        scattering_count: usize,
        initial_l: usize,
        beta_angles: &'a [f64],
        lambda_capacity: usize,
    ) -> LambdaIndexInput<'a> {
        LambdaIndexInput {
            calculation,
            energy_index,
            scattering_count,
            initial_l,
            beta_angles,
            lambda_capacity,
            max_m: 10,
            max_n: 10,
        }
    }

    #[test]
    fn exact_order_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [0.0, std::f64::consts::PI, 0.5, 2.8];
        let lambda = lambda_indices(input(2, 10, 2, 3, &beta, 40))?;

        assert_eq!(lambda.order, 2);
        assert_eq!(lambda.requested_n_max, 1);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 6);
        assert_eq!(lambda.max_n, 1);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert!(!lambda.truncated);
        assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2, 0]);
        assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0, 1]);
        Ok(())
    }

    #[test]
    fn single_scattering_uses_initial_l_exact_reference() -> Result<(), GenfmtError> {
        let beta = [0.3, 1.2];
        let lambda = lambda_indices(input(10, 8, 1, 2, &beta, 40))?;

        assert_eq!(lambda.order, 6);
        assert_eq!(lambda.requested_n_max, 2);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 15);
        assert_eq!(lambda.max_n, 2);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(
            lambda.m_indices.to_vec(),
            vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2]
        );
        assert_eq!(
            lambda.n_indices.to_vec(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        );
        Ok(())
    }

    #[test]
    fn cute_linear_low_energy_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [
            0.0,
            std::f64::consts::PI,
            0.010,
            std::f64::consts::PI - 0.010,
        ];
        let lambda = lambda_indices(input(10, 41, 2, 4, &beta, 80))?;

        assert_eq!(lambda.order, 12);
        assert_eq!(lambda.requested_n_max, 4);
        assert_eq!(lambda.requested_m_max, 4);
        assert_eq!(lambda.initial_l_prefix_len, 45);
        assert_eq!(lambda.max_n, 4);
        assert_eq!(lambda.max_m_plus_one, 5);
        assert_eq!(lambda.m_indices.len(), 45);
        assert_eq!(
            &lambda.m_indices.to_vec()[..9],
            &[0, -1, 1, -2, 2, -3, 3, -4, 4]
        );
        assert_eq!(
            &lambda.n_indices.to_vec()[36..],
            &[4, 4, 4, 4, 4, 4, 4, 4, 4]
        );
        Ok(())
    }

    #[test]
    fn cute_nonlinear_high_energy_sorts_initial_l_prefix() -> Result<(), GenfmtError> {
        let beta = [0.0, 0.25, std::f64::consts::PI];
        let lambda = lambda_indices(input(10, 42, 2, 4, &beta, 80))?;

        assert_eq!(lambda.order, 21);
        assert_eq!(lambda.requested_n_max, 9);
        assert_eq!(lambda.requested_m_max, 3);
        assert_eq!(lambda.m_indices.len(), 70);
        assert_eq!(lambda.initial_l_prefix_len, 35);
        assert_eq!(lambda.max_n, 9);
        assert_eq!(lambda.max_m_plus_one, 4);
        assert_eq!(&lambda.n_indices.to_vec()[..7], &[0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&lambda.n_indices.to_vec()[28..35], &[4, 4, 4, 4, 4, 4, 4]);
        assert_eq!(&lambda.n_indices.to_vec()[35..42], &[5, 5, 5, 5, 5, 5, 5]);
        assert_eq!(&lambda.n_indices.to_vec()[63..], &[9, 9, 9, 9, 9, 9, 9]);
        Ok(())
    }

    #[test]
    fn negative_calculation_decodes_requested_limits() -> Result<(), GenfmtError> {
        let beta = [0.0, 0.5];
        let lambda = lambda_indices(input(-80_205, 12, 2, 2, &beta, 80))?;

        assert_eq!(lambda.order, 7);
        assert_eq!(lambda.requested_n_max, 5);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 15);
        assert_eq!(lambda.max_n, 3);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(
            lambda.m_indices.to_vec(),
            vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1]
        );
        assert_eq!(
            lambda.n_indices.to_vec(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
        );
        Ok(())
    }

    #[test]
    fn capacity_truncation_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [0.0, 1.0];
        let lambda = lambda_indices(input(4, 10, 2, 1, &beta, 5))?;

        assert!(lambda.truncated);
        assert_eq!(lambda.order, 4);
        assert_eq!(lambda.requested_n_max, 2);
        assert_eq!(lambda.requested_m_max, 4);
        assert_eq!(lambda.initial_l_prefix_len, 3);
        assert_eq!(lambda.max_n, 0);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2]);
        assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn cute_calculation_rejects_nonfinite_beta() {
        let beta = [f64::NAN];

        assert!(matches!(
            lambda_indices(input(10, 42, 2, 4, &beta, 80)),
            Err(GenfmtError::NonFiniteBetaAngle { index: 0, .. })
        ));
    }

    #[test]
    fn undefined_calculation_is_an_error_for_multiple_scattering() {
        assert_eq!(
            lambda_indices(input(11, 1, 2, 0, &[], 10)),
            Err(GenfmtError::UndefinedLambdaCalculation { calculation: 11 })
        );
    }

    #[test]
    fn dimension_overflow_is_reported() {
        let mut bad = input(10, 42, 2, 4, &[0.25], 80);
        bad.max_n = 8;

        assert!(matches!(
            lambda_indices(bad),
            Err(GenfmtError::DimensionExceeded {
                max_n: 9,
                max_n_limit: 8,
                ..
            })
        ));
    }

    #[test]
    fn initial_state_rotation_matches_feff_full_reference() -> Result<(), GenfmtError> {
        let rotation = initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 4,
            mmaxp1: 4,
            beta_angle: 0.7,
        })?;

        assert_eq!(rotation.matrix.shape(), &[4, 7, 7]);
        assert_eq!(rotation.matrix.strides(), &[1, 4, 28]);
        assert_eq!(rotation.magnetic_offset, 3);
        assert_close(rotation_sum(&rotation), 14.508_147_433_950_487);
        assert_eq!(rotation_nonzero_count(&rotation), 84);
        assert_close(rotation_value(&rotation, 1, 0, 0), 1.0);
        assert_close(
            rotation_value(&rotation, 2, -1, -1),
            0.882_421_093_642_244_2,
        );
        assert_close(
            rotation_value(&rotation, 2, -1, 0),
            0.455_530_695_206_085_63,
        );
        assert_close(rotation_value(&rotation, 2, 0, 1), 0.455_530_695_206_085_63);
        assert_close(
            rotation_value(&rotation, 3, -2, 1),
            0.075_746_411_121_730_47,
        );
        assert_close(
            rotation_value(&rotation, 4, -3, 3),
            0.001_625_504_772_936_771_3,
        );
        assert_close(
            rotation_value(&rotation, 4, 0, 0),
            -0.028_712_995_143_227_615,
        );
        Ok(())
    }

    #[test]
    fn initial_state_rotation_matches_feff_limited_m_reference() -> Result<(), GenfmtError> {
        let rotation = initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 5,
            mmaxp1: 2,
            beta_angle: -0.4,
        })?;

        assert_eq!(rotation.matrix.shape(), &[5, 3, 3]);
        assert_eq!(rotation.matrix.strides(), &[1, 5, 15]);
        assert_eq!(rotation.magnetic_offset, 1);
        assert_close(rotation_sum(&rotation), 10.424_101_881_334_796);
        assert_eq!(rotation_nonzero_count(&rotation), 37);
        assert_close(rotation_value(&rotation, 1, 0, 0), 1.0);
        assert_close(
            rotation_value(&rotation, 2, -1, -1),
            0.960_530_497_001_442_6,
        );
        assert_close(rotation_value(&rotation, 2, -1, 0), -0.275_360_350_564_871);
        assert_close(rotation_value(&rotation, 2, 0, 1), -0.275_360_350_564_871);
        assert_close(
            rotation_value(&rotation, 3, -1, 1),
            0.112_177_142_327_859_86,
        );
        assert_close(rotation_value(&rotation, 5, -1, 1), 0.307_544_785_027_699_8);
        assert_close(rotation_value(&rotation, 5, 0, 0), 0.342_377_357_912_471_87);
        Ok(())
    }

    #[test]
    fn initial_state_rotation_rejects_invalid_inputs() {
        assert_eq!(
            initial_state_rotation(InitialStateRotationInput {
                lmaxp1: 0,
                mmaxp1: 1,
                beta_angle: 0.0,
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            initial_state_rotation(InitialStateRotationInput {
                lmaxp1: 1,
                mmaxp1: 0,
                beta_angle: 0.0,
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            initial_state_rotation(InitialStateRotationInput {
                lmaxp1: 1,
                mmaxp1: 1,
                beta_angle: f64::NAN,
            }),
            Err(GenfmtError::NonFiniteRotationAngle)
        );
    }

    #[test]
    fn curved_wave_polynomials_match_feff_sclmz_reference() -> Result<(), GenfmtError> {
        let table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 4,
            mmaxp1: 4,
            rho: Complex::new(1.25, 0.4),
        })?;

        assert_eq!(table.shape(), &[5, 4]);
        assert_eq!(table.strides(), &[1, 5]);
        assert_eq!(complex_nonzero_count(&table), 11);
        assert_complex_close(table[(0, 0)], Complex::new(1.0, 0.0));
        assert_complex_close(
            table[(1, 0)],
            Complex::new(1.232_220_609_579_100_2, 0.725_689_404_934_687_9),
        );
        assert_complex_close(
            table[(2, 0)],
            Complex::new(0.278_565_725_973_782_6, 3.188_188_430_678_23),
        );
        assert_complex_close(
            table[(3, 1)],
            Complex::new(-28.733_692_908_170_283, 2.550_923_127_350_68),
        );
        assert_complex_close(table[(4, 2)], Complex::new(0.0, 0.0));
        assert_complex_close(
            complex_sum(&table),
            Complex::new(-58.983_990_231_020_26, -154.618_863_530_600_9),
        );
        Ok(())
    }

    #[test]
    fn curved_wave_polynomials_match_limited_m_reference() -> Result<(), GenfmtError> {
        let table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 5,
            mmaxp1: 3,
            rho: Complex::new(-0.8, 1.1),
        })?;

        assert_eq!(table.shape(), &[6, 3]);
        assert_eq!(table.strides(), &[1, 6]);
        assert_eq!(complex_nonzero_count(&table), 12);
        assert_complex_close(
            table[(1, 0)],
            Complex::new(1.594_594_594_594_594_5, -0.432_432_432_432_432_35),
        );
        assert_complex_close(
            table[(2, 0)],
            Complex::new(3.283_418_553_688_824, -2.840_029_218_407_596),
        );
        assert_complex_close(
            table[(3, 1)],
            Complex::new(3.013_207_509_920_446_7, -35.022_288_906_876_184),
        );
        assert_complex_close(
            table[(4, 2)],
            Complex::new(-180.487_514_146_329_86, -250.055_955_704_979_3),
        );
        assert_complex_close(
            complex_sum(&table),
            Complex::new(-306.259_756_232_255_1, -662.066_424_389_366_5),
        );
        Ok(())
    }

    #[test]
    fn curved_wave_polynomials_retain_requested_zero_columns() -> Result<(), GenfmtError> {
        let table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 2,
            mmaxp1: 4,
            rho: Complex::new(1.0, 0.25),
        })?;

        assert_eq!(table.shape(), &[3, 4]);
        assert!(
            table
                .column(2)
                .iter()
                .all(|&value| value == Complex::new(0.0, 0.0))
        );
        assert!(
            table
                .column(3)
                .iter()
                .all(|&value| value == Complex::new(0.0, 0.0))
        );
        Ok(())
    }

    #[test]
    fn curved_wave_polynomials_reject_invalid_inputs() {
        assert_eq!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 0,
                mmaxp1: 1,
                rho: Complex::new(1.0, 0.0),
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 1,
                mmaxp1: 0,
                rho: Complex::new(1.0, 0.0),
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 1,
                mmaxp1: 1,
                rho: Complex::new(0.0, 0.0),
            }),
            Err(GenfmtError::ZeroComplex { field: "rho" })
        );
        assert!(matches!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 1,
                mmaxp1: 1,
                rho: Complex::new(f64::NAN, 0.0),
            }),
            Err(GenfmtError::NonFiniteComplex { field: "rho", .. })
        ));
    }

    #[test]
    fn xstar_matches_feff_linear_references() -> Result<(), GenfmtError> {
        assert_close(
            xstar(XStarInput {
                primary_polarization: [1.0, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [2.0, 0.0, 0.0],
                last_leg: [0.0, 3.0, 0.0],
                degeneracy: 3.5,
                initial_l: 1,
                ellipticity: 0.0,
            })?,
            0.0,
        );
        assert_close(
            xstar(XStarInput {
                primary_polarization: [0.2, 0.9, 0.4],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [1.0, 0.5, -0.25],
                last_leg: [0.4, -0.3, 1.2],
                degeneracy: 1.75,
                initial_l: 1,
                ellipticity: 0.0,
            })?,
            0.185_559_995_771_885_34,
        );
        Ok(())
    }

    #[test]
    fn xstar_matches_feff_elliptic_references() -> Result<(), GenfmtError> {
        assert_close(
            xstar(XStarInput {
                primary_polarization: [0.3, 1.0, -0.2],
                secondary_polarization: [-0.4, 0.2, 1.5],
                first_leg: [1.2, -0.5, 0.8],
                last_leg: [-0.7, 1.4, 0.6],
                degeneracy: 2.25,
                initial_l: 2,
                ellipticity: 0.7,
            })?,
            -0.014_836_343_260_557_886,
        );
        assert_close(
            xstar(XStarInput {
                primary_polarization: [1.0, 2.0, 3.0],
                secondary_polarization: [2.0, -1.0, 0.5],
                first_leg: [-0.25, 0.75, 1.50],
                last_leg: [1.1, -0.9, 0.4],
                degeneracy: 5.0,
                initial_l: 4,
                ellipticity: -0.35,
            })?,
            0.254_890_323_398_489_77,
        );
        Ok(())
    }

    #[test]
    fn xstar_rejects_invalid_inputs() {
        assert_eq!(
            xstar(XStarInput {
                primary_polarization: [1.0, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [1.0, 0.0, 0.0],
                last_leg: [0.0, 1.0, 0.0],
                degeneracy: 1.0,
                initial_l: 5,
                ellipticity: 0.0,
            }),
            Err(GenfmtError::InvalidInitialAngularMomentum { initial_l: 5 })
        );
        assert!(matches!(
            xstar(XStarInput {
                primary_polarization: [f64::NAN, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [1.0, 0.0, 0.0],
                last_leg: [0.0, 1.0, 0.0],
                degeneracy: 1.0,
                initial_l: 1,
                ellipticity: 0.0,
            }),
            Err(GenfmtError::NonFiniteVector {
                field: "primary_polarization",
                index: 0,
                ..
            })
        ));
        assert_eq!(
            xstar(XStarInput {
                primary_polarization: [1.0, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [0.0, 0.0, 0.0],
                last_leg: [0.0, 1.0, 0.0],
                degeneracy: 1.0,
                initial_l: 1,
                ellipticity: 0.0,
            }),
            Err(GenfmtError::ZeroVector { field: "first_leg" })
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    fn complex_sum(table: &ndarray::Array2<Complex>) -> Complex {
        table
            .iter()
            .copied()
            .fold(Complex::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn complex_nonzero_count(table: &ndarray::Array2<Complex>) -> usize {
        table
            .iter()
            .filter(|&&value| value.re.abs() > 1.0e-14 || value.im.abs() > 1.0e-14)
            .count()
    }

    fn rotation_value(rotation: &InitialStateRotation, il: usize, m1: isize, m2: isize) -> f64 {
        let row = (m1 + rotation.magnetic_offset as isize) as usize;
        let column = (m2 + rotation.magnetic_offset as isize) as usize;
        rotation.matrix[(il - 1, row, column)]
    }

    fn rotation_sum(rotation: &InitialStateRotation) -> f64 {
        rotation.matrix.iter().sum()
    }

    fn rotation_nonzero_count(rotation: &InitialStateRotation) -> usize {
        rotation
            .matrix
            .iter()
            .filter(|&&value| value.abs() > 1.0e-14)
            .count()
    }
}
