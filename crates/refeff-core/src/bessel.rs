//! FEFF spherical Bessel and Hankel helpers.
//!
//! This module ports `MATH/bjnser.f90`, `MATH/besjn.f90`, and
//! `MATH/besjh.f90`. FEFF combines a power series for small arguments,
//! upward/downward recursion for intermediate arguments, and asymptotic
//! Abramowitz-Stegun forms for larger arguments. The Rust API returns typed
//! errors instead of terminating the process.

use ndarray::Array1;
use thiserror::Error;

use crate::{Complex, ComplexVec, Real};

const SERIES_ITERATIONS: usize = 160;
const SERIES_TOLERANCE: Real = 1.0e-15;
const SMALL_CUT: Real = 1.0;
const MEDIUM_CUT: Real = 7.51;
const NEUMANN_SERIES_CUT: Real = 5.01;

/// Spherical Bessel `j_l` and Neumann `y_l` values for `l = 0..=max_l`.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalBessel {
    /// Abramowitz-Stegun spherical Bessel values `j_l(x)`.
    pub j: ComplexVec,
    /// Abramowitz-Stegun spherical Neumann values `y_l(x)`.
    pub y: ComplexVec,
}

/// Spherical Bessel `j_l` and FEFF/Messiah Hankel values for `l = 0..=max_l`.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalHankel {
    /// Abramowitz-Stegun spherical Bessel values `j_l(x)`.
    pub j: ComplexVec,
    /// FEFF spherical Hankel values.
    ///
    /// FEFF uses `h_l^+` when `Im(x) >= 0` and `h_l^-` when `Im(x) < 0`.
    pub h: ComplexVec,
}

/// Error returned by spherical Bessel helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BesselError {
    /// FEFF `besjn` requires a positive real argument.
    #[error("besjn requires Re(x) > 0, got {real}")]
    NonPositiveRealArgument { real: Real },
    /// FEFF `besjh` rejects negative real arguments.
    #[error("besjh requires Re(x) >= 0, got {real}")]
    NegativeRealArgument { real: Real },
    /// Bessel arguments must be finite.
    #[error("Bessel argument must be finite, got ({real}, {imaginary})")]
    NonFiniteArgument { real: Real, imaginary: Real },
    /// Neumann and Hankel values are singular at exactly zero.
    #[error("spherical Neumann/Hankel values are singular at x = 0")]
    ZeroArgument,
    /// A series expansion did not converge within the FEFF iteration limit.
    #[error("{function} series for l={l} did not converge in {iterations} iterations")]
    SeriesDidNotConverge {
        function: &'static str,
        l: usize,
        iterations: usize,
    },
}

/// Port of FEFF `besjn`: compute `j_l(x)` and `y_l(x)` for `l = 0..=max_l`.
pub fn besjn(x: Complex, max_l: usize) -> Result<SphericalBessel, BesselError> {
    spherical_bessel_j_y(x, max_l)
}

/// Compute FEFF spherical Bessel `j_l(x)` and Neumann `y_l(x)` values.
pub fn spherical_bessel_j_y(x: Complex, max_l: usize) -> Result<SphericalBessel, BesselError> {
    validate_besjn_argument(x)?;
    let (j, y) = spherical_j_y_values(x, max_l)?;
    Ok(SphericalBessel {
        j: Array1::from_vec(j),
        y: Array1::from_vec(y),
    })
}

/// Port of FEFF `besjh`: compute `j_l(x)` and FEFF Hankel values.
pub fn besjh(x: Complex, max_l: usize) -> Result<SphericalHankel, BesselError> {
    spherical_bessel_j_h(x, max_l)
}

/// Compute FEFF spherical Bessel `j_l(x)` and Hankel values.
pub fn spherical_bessel_j_h(x: Complex, max_l: usize) -> Result<SphericalHankel, BesselError> {
    validate_besjh_argument(x)?;

    let (j, h) = if is_large_argument(x) {
        spherical_j_h_asymptotic(x, max_l)
    } else {
        let (j, y) = spherical_j_y_values(x, max_l)?;
        let imaginary_unit = Complex::new(0.0, 1.0);
        let h = y
            .iter()
            .zip(j.iter())
            .map(|(&y_value, &j_value)| -y_value + imaginary_unit * j_value)
            .collect();
        (j, h)
    };

    Ok(SphericalHankel {
        j: Array1::from_vec(j),
        h: Array1::from_vec(h),
    })
}

fn spherical_j_y_values(
    x: Complex,
    max_l: usize,
) -> Result<(Vec<Complex>, Vec<Complex>), BesselError> {
    if is_small_argument(x) {
        return spherical_j_y_series_range(x, max_l);
    }
    if is_large_argument(x) {
        return Ok(spherical_j_y_asymptotic(x, max_l));
    }
    spherical_j_y_recurrence(x, max_l)
}

fn spherical_j_y_series_range(
    x: Complex,
    max_l: usize,
) -> Result<(Vec<Complex>, Vec<Complex>), BesselError> {
    let mut j = Vec::with_capacity(max_l + 1);
    let mut y = Vec::with_capacity(max_l + 1);
    for l in 0..=max_l {
        let values = bjnser(x, l, SeriesMode::Both)?;
        j.push(values.j);
        y.push(values.y);
    }
    Ok((j, y))
}

fn spherical_j_y_recurrence(
    x: Complex,
    max_l: usize,
) -> Result<(Vec<Complex>, Vec<Complex>), BesselError> {
    let mut j = vec![Complex::new(0.0, 0.0); max_l + 1];
    let mut y = vec![Complex::new(0.0, 0.0); max_l + 1];

    if max_l == 0 {
        let values = bjnser(x, 0, SeriesMode::Both)?;
        j[0] = values.j;
        y[0] = values.y;
        return Ok((j, y));
    }

    j[max_l] = bjnser(x, max_l, SeriesMode::JOnly)?.j;
    j[max_l - 1] = bjnser(x, max_l - 1, SeriesMode::JOnly)?.j;

    if x.re < NEUMANN_SERIES_CUT && x.im.abs() < NEUMANN_SERIES_CUT {
        y[0] = bjnser(x, 0, SeriesMode::YOnly)?.y;
        y[1] = bjnser(x, 1, SeriesMode::YOnly)?.y;
    } else {
        let xi = Complex::new(1.0, 0.0) / x;
        let sin_x = x.sin();
        let cos_x = x.cos();
        y[0] = -cos_x * xi;
        y[1] = -cos_x * xi * xi - sin_x * xi;
    }

    for target_l in 2..=max_l {
        let coefficient = (2 * target_l - 1) as Real;
        y[target_l] = y[target_l - 1] * coefficient / x - y[target_l - 2];
    }

    for target_l in (0..=(max_l - 2)).rev() {
        let coefficient = (2 * target_l + 3) as Real;
        j[target_l] = j[target_l + 1] * coefficient / x - j[target_l + 2];
    }

    Ok((j, y))
}

fn spherical_j_y_asymptotic(x: Complex, max_l: usize) -> (Vec<Complex>, Vec<Complex>) {
    let (sjl, cjl) = asymptotic_j_tables(x, max_l);
    let sin_x = x.sin();
    let cos_x = x.cos();

    let mut j = Vec::with_capacity(max_l + 1);
    let mut y = Vec::with_capacity(max_l + 1);
    for l in 0..=max_l {
        j.push(sin_x * sjl[l] + cos_x * cjl[l]);
        y.push(sin_x * cjl[l] - cos_x * sjl[l]);
    }
    (j, y)
}

fn spherical_j_h_asymptotic(x: Complex, max_l: usize) -> (Vec<Complex>, Vec<Complex>) {
    let (sjl, cjl) = asymptotic_j_tables(x, max_l);
    let sin_x = x.sin();
    let cos_x = x.cos();
    let imaginary_unit = Complex::new(0.0, 1.0);
    let outgoing = x.im >= 0.0;
    let phase = if outgoing {
        (imaginary_unit * x).exp()
    } else {
        (-imaginary_unit * x).exp()
    };

    let mut j = Vec::with_capacity(max_l + 1);
    let mut h = Vec::with_capacity(max_l + 1);
    for l in 0..=max_l {
        j.push(sin_x * sjl[l] + cos_x * cjl[l]);
        let hankel_factor = if outgoing {
            sjl[l] + imaginary_unit * cjl[l]
        } else {
            sjl[l] - imaginary_unit * cjl[l]
        };
        h.push(hankel_factor * phase);
    }
    (j, h)
}

fn asymptotic_j_tables(x: Complex, max_l: usize) -> (Vec<Complex>, Vec<Complex>) {
    let xi = Complex::new(1.0, 0.0) / x;
    let powers = complex_powers(xi, 11);
    let zero = Complex::new(0.0, 0.0);

    let s_seed = [
        powers[1],
        powers[2],
        powers[3] * 3.0 - powers[1],
        powers[4] * 15.0 - powers[2] * 6.0,
        powers[5] * 105.0 - powers[3] * 45.0 + powers[1],
        powers[6] * 945.0 - powers[4] * 420.0 + powers[2] * 15.0,
        powers[7] * 10395.0 - powers[5] * 4725.0 + powers[3] * 210.0 - powers[1],
        powers[8] * 135135.0 - powers[6] * 62370.0 + powers[4] * 3150.0 - powers[2] * 28.0,
        powers[9] * 2027025.0 - powers[7] * 945945.0 + powers[5] * 51975.0 - powers[3] * 630.0
            + powers[1],
        powers[10] * 34459425.0 - powers[8] * 16216200.0 + powers[6] * 945945.0
            - powers[4] * 13860.0
            + powers[2] * 45.0,
        powers[11] * 654729075.0 - powers[9] * 310134825.0 + powers[7] * 18918900.0
            - powers[5] * 315315.0
            + powers[3] * 1485.0
            - powers[1],
    ];
    let c_seed = [
        zero,
        -powers[1],
        -powers[2] * 3.0,
        -powers[3] * 15.0 + powers[1],
        -powers[4] * 105.0 + powers[2] * 10.0,
        -powers[5] * 945.0 + powers[3] * 105.0 - powers[1],
        -powers[6] * 10395.0 + powers[4] * 1260.0 - powers[2] * 21.0,
        -powers[7] * 135135.0 + powers[5] * 17325.0 - powers[3] * 378.0 + powers[1],
        -powers[8] * 2027025.0 + powers[6] * 270270.0 - powers[4] * 6930.0 + powers[2] * 36.0,
        -powers[9] * 34459425.0 + powers[7] * 4729725.0 - powers[5] * 135135.0 + powers[3] * 990.0
            - powers[1],
        -powers[10] * 654729075.0 + powers[8] * 91891800.0 - powers[6] * 2837835.0
            + powers[4] * 25740.0
            - powers[2] * 55.0,
    ];

    let mut sjl = vec![zero; max_l + 1];
    let mut cjl = vec![zero; max_l + 1];
    let seed_count = (max_l + 1).min(s_seed.len());
    sjl[..seed_count].copy_from_slice(&s_seed[..seed_count]);
    cjl[..seed_count].copy_from_slice(&c_seed[..seed_count]);

    for target_l in 11..=max_l {
        let coefficient = (2 * target_l - 1) as Real;
        sjl[target_l] = sjl[target_l - 1] * coefficient * xi - sjl[target_l - 2];
        cjl[target_l] = cjl[target_l - 1] * coefficient * xi - cjl[target_l - 2];
    }

    (sjl, cjl)
}

fn complex_powers(base: Complex, max_power: usize) -> Vec<Complex> {
    let mut powers = vec![Complex::new(1.0, 0.0); max_power + 1];
    for power in 1..=max_power {
        powers[power] = powers[power - 1] * base;
    }
    powers
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeriesMode {
    Both,
    JOnly,
    YOnly,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SeriesValues {
    j: Complex,
    y: Complex,
}

fn bjnser(x: Complex, l: usize, mode: SeriesMode) -> Result<SeriesValues, BesselError> {
    let u = x * x / 2.0;
    let djl = odd_product_through(2 * l + 1);
    let dnl = djl / (2 * l + 1) as Real;

    let j = if mode == SeriesMode::YOnly {
        Complex::new(0.0, 0.0)
    } else {
        bjnser_j(x, l, u, djl)?
    };

    let y = if mode == SeriesMode::JOnly {
        Complex::new(0.0, 0.0)
    } else {
        bjnser_y(x, l, u, dnl)?
    };

    Ok(SeriesValues { j, y })
}

fn bjnser_j(x: Complex, l: usize, u: Complex, djl: Real) -> Result<Complex, BesselError> {
    let mut series_sum = Complex::new(1.0, 0.0);
    let mut nf = 1.0;
    let mut nfac = (2 * l + 3) as Real;
    let mut denominator = nfac;
    let mut sign = -1.0;
    let mut ux = u;

    for _ in 0..SERIES_ITERATIONS {
        let delta = ux * (sign / denominator);
        series_sum += delta;
        if (delta / series_sum).norm() <= SERIES_TOLERANCE {
            return Ok(series_sum * complex_pow_usize(x, l) / djl);
        }
        sign = -sign;
        ux *= u;
        nf += 1.0;
        nfac += 2.0;
        denominator *= nf * nfac;
    }

    Err(BesselError::SeriesDidNotConverge {
        function: "jl",
        l,
        iterations: SERIES_ITERATIONS,
    })
}

fn bjnser_y(x: Complex, l: usize, u: Complex, dnl: Real) -> Result<Complex, BesselError> {
    let mut series_sum = Complex::new(1.0, 0.0);
    let mut nf = 1.0;
    let mut nfac = 1.0 - 2.0 * l as Real;
    let mut denominator = nfac;
    let mut sign = -1.0;
    let mut ux = u;

    for _ in 0..SERIES_ITERATIONS {
        let delta = ux * (sign / denominator);
        series_sum += delta;
        if (delta / series_sum).norm() <= SERIES_TOLERANCE {
            return Ok(-series_sum * dnl / complex_pow_usize(x, l + 1));
        }
        sign = -sign;
        ux *= u;
        nf += 1.0;
        nfac += 2.0;
        denominator *= nf * nfac;
    }

    Err(BesselError::SeriesDidNotConverge {
        function: "nl",
        l,
        iterations: SERIES_ITERATIONS,
    })
}

fn odd_product_through(last_odd: usize) -> Real {
    (1..=last_odd)
        .step_by(2)
        .fold(1.0, |product, value| product * value as Real)
}

fn complex_pow_usize(value: Complex, power: usize) -> Complex {
    (0..power).fold(Complex::new(1.0, 0.0), |product, _| product * value)
}

fn is_small_argument(x: Complex) -> bool {
    x.re < SMALL_CUT && x.im.abs() < SMALL_CUT
}

fn is_large_argument(x: Complex) -> bool {
    !(x.re < MEDIUM_CUT && x.im.abs() < MEDIUM_CUT)
}

fn validate_besjn_argument(x: Complex) -> Result<(), BesselError> {
    validate_finite(x)?;
    if x.re <= 0.0 {
        return Err(BesselError::NonPositiveRealArgument { real: x.re });
    }
    validate_nonzero(x)
}

fn validate_besjh_argument(x: Complex) -> Result<(), BesselError> {
    validate_finite(x)?;
    if x.re < 0.0 {
        return Err(BesselError::NegativeRealArgument { real: x.re });
    }
    validate_nonzero(x)
}

fn validate_finite(x: Complex) -> Result<(), BesselError> {
    if !(x.re.is_finite() && x.im.is_finite()) {
        return Err(BesselError::NonFiniteArgument {
            real: x.re,
            imaginary: x.im,
        });
    }
    Ok(())
}

fn validate_nonzero(x: Complex) -> Result<(), BesselError> {
    if x == Complex::new(0.0, 0.0) {
        return Err(BesselError::ZeroArgument);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: Complex, expected: Complex) {
        let tolerance = 2.0e-12;
        assert!(
            (actual - expected).norm() < tolerance,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn assert_first_values(actual: &ComplexVec, expected: &[Complex]) {
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected);
        }
    }

    #[test]
    fn besjn_matches_feff_small_argument_series() -> Result<(), BesselError> {
        let result = besjn(Complex::new(0.7, 0.2), 5)?;

        assert_first_values(
            &result.j,
            &[
                Complex::new(0.9260369547275705, -0.04459588911838349),
                Complex::new(0.22474428544876177, 0.05737066815224228),
                Complex::new(0.029407196845533484, 0.01748613789282074),
            ],
        );
        assert_first_values(
            &result.y,
            &[
                Complex::new(-0.9814947532033563, 0.465718806507681),
                Complex::new(-2.046_607_833_861_181, 1.030_071_578_167_531),
                Complex::new(-5.961587330207891, 5.932611409470666),
            ],
        );
        Ok(())
    }

    #[test]
    fn besjn_matches_feff_medium_argument_recurrence() -> Result<(), BesselError> {
        let result = besjn(Complex::new(3.5, 0.4), 17)?;

        assert_first_values(
            &result.j,
            &[
                Complex::new(-0.11935035461609986, -0.09626046299195844),
                Complex::new(0.24411415352002147, -0.09656890029317053),
                Complex::new(0.31655550623644, -0.009050611730226591),
                Complex::new(0.20081897586573566, 0.032789954463046014),
                Complex::new(0.08730241328064724, 0.028475329854365797),
                Complex::new(0.029038977866424266, 0.014162841878790623),
            ],
        );
        assert_first_values(
            &result.y,
            &[
                Complex::new(0.280_877_282_154_44, -0.07326741128225883),
                Complex::new(0.1962047883814211, 0.06654355305241916),
                Complex::new(-0.10843549797492152, 0.11059710999238742),
            ],
        );
        Ok(())
    }

    #[test]
    fn besjn_matches_feff_large_argument_asymptotic() -> Result<(), BesselError> {
        let result = besjn(Complex::new(12.0, 0.5), 17)?;

        assert_first_values(
            &result.j,
            &[
                Complex::new(-0.04880955623263242, 0.038677759544511026),
                Complex::new(-0.08405406049108705, -0.016575069902921128),
                Complex::new(0.027660101934306482, -0.04194029975781106),
            ],
        );
        assert_first_values(
            &result.y,
            &[
                Complex::new(-0.08012771182544176, -0.019961814392698934),
                Complex::new(0.04207462764267297, -0.040060622052654295),
                Complex::new(0.09021156390209511, 0.009526498376341473),
            ],
        );
        Ok(())
    }

    #[test]
    fn besjh_matches_feff_medium_argument_hankel() -> Result<(), BesselError> {
        let result = besjh(Complex::new(3.5, 0.4), 5)?;

        assert_first_values(
            &result.h,
            &[
                Complex::new(-0.18461681916248157, -0.046082943333841025),
                Complex::new(-0.09963588808825058, 0.1775706004676023),
                Complex::new(0.11748610970514811, 0.20595839624405254),
                Complex::new(0.2985016183322633, 0.09392816785315697),
            ],
        );
        Ok(())
    }

    #[test]
    fn besjh_matches_feff_large_argument_outgoing_and_incoming() -> Result<(), BesselError> {
        let outgoing = besjh(Complex::new(12.0, 0.5), 5)?;
        let incoming = besjh(Complex::new(12.0, -0.5), 5)?;

        assert_first_values(
            &outgoing.h,
            &[
                Complex::new(0.04144995228093075, -0.028847741839933494),
                Complex::new(-0.025499557739751846, -0.04399343843843277),
                Complex::new(-0.04827126414428406, 0.018133603557965016),
            ],
        );
        assert_first_values(
            &incoming.h,
            &[
                Complex::new(0.04144995228093075, 0.028847741839933494),
                Complex::new(-0.025499557739751846, 0.04399343843843277),
                Complex::new(-0.04827126414428406, -0.018133603557965016),
            ],
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(matches!(
            besjn(Complex::new(0.0, 0.1), 2),
            Err(BesselError::NonPositiveRealArgument { .. })
        ));
        assert!(matches!(
            besjh(Complex::new(-0.1, 0.0), 2),
            Err(BesselError::NegativeRealArgument { .. })
        ));
        assert!(matches!(
            besjh(Complex::new(0.0, 0.0), 2),
            Err(BesselError::ZeroArgument)
        ));
    }
}
