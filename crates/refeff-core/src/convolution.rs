//! FEFF analytic Lorentzian convolution helpers.
//!
//! This module ports `MATH/conv.f90`. FEFF linearly interpolates a complex
//! spectrum on each energy interval and integrates the Lorentzian kernel
//! analytically with `conv1`; `conv` applies that segment integral to every
//! requested output energy and adds one extrapolated endpoint interval.

use ndarray::Array1;
use thiserror::Error;

use crate::{Complex, ComplexVec, Real};

const FEFF_REAL_PI: Real = std::f32::consts::PI as Real;

/// Error returned by FEFF convolution helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ConvolutionError {
    /// Energy and spectrum arrays must have identical lengths.
    #[error("convolution length mismatch: omega has {omega_len}, spectrum has {spectrum_len}")]
    LengthMismatch {
        omega_len: usize,
        spectrum_len: usize,
    },
    /// FEFF `conv` needs at least two points to extrapolate the final interval.
    #[error("convolution requires at least two points, got {points}")]
    InsufficientPoints { points: usize },
    /// The Lorentzian width must be positive and finite.
    #[error("Lorentzian width must be positive and finite, got {width}")]
    InvalidWidth { width: Real },
    /// Energy values must be finite.
    #[error("energy value {name} must be finite, got {value}")]
    NonFiniteEnergy { name: &'static str, value: Real },
    /// Spectrum values must be finite.
    #[error("spectrum value {name} must be finite, got ({real}, {imaginary})")]
    NonFiniteSpectrum {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// FEFF's endpoint extrapolation divides by the final energy spacing.
    #[error("last two energy values must be distinct for endpoint extrapolation")]
    DuplicateEndpointEnergy,
}

/// Port of FEFF `conv1`, the analytic integral over one linear segment.
pub fn conv1(
    x1: Real,
    x2: Real,
    y1: Complex,
    y2: Complex,
    x0: Real,
    width: Real,
) -> Result<Complex, ConvolutionError> {
    validate_width(width)?;
    validate_energy("x1", x1)?;
    validate_energy("x2", x2)?;
    validate_energy("x0", x0)?;
    validate_spectrum("y1", y1)?;
    validate_spectrum("y2", y2)?;

    let half_width = (x2 - x1) / 2.0;
    let denominator = (x1 + x2) / 2.0 - x0;
    let t = Complex::new(half_width, 0.0) / Complex::new(denominator, -width);

    let real_part = conv1_component((y2.re - y1.re) / 2.0, (y2.re + y1.re) / 2.0, t);
    let imaginary_part = conv1_component((y2.im - y1.im) / 2.0, (y2.im + y1.im) / 2.0, t);
    Ok(Complex::new(real_part, imaginary_part))
}

/// Convolve a FEFF spectrum with the Lorentzian broadening kernel.
///
/// This ports `conv` and returns a new `ndarray` vector. Use
/// [`conv_in_place`] when the caller needs FEFF's in-place mutation behavior.
pub fn conv(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<ComplexVec, ConvolutionError> {
    let values = convolved_values(omega, spectrum, width)?;
    Ok(Array1::from_vec(values))
}

/// In-place FEFF `conv` behavior for a mutable spectrum slice.
pub fn conv_in_place(
    omega: &[Real],
    spectrum: &mut [Complex],
    width: Real,
) -> Result<(), ConvolutionError> {
    let values = convolved_values(omega, spectrum, width)?;
    for (target, value) in spectrum.iter_mut().zip(values) {
        *target = value;
    }
    Ok(())
}

fn convolved_values(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<Vec<Complex>, ConvolutionError> {
    validate_inputs(omega, spectrum, width)?;
    let last = omega.len() - 1;
    let previous = omega.len() - 2;
    let final_spacing = omega[last] - omega[previous];
    if final_spacing == 0.0 {
        return Err(ConvolutionError::DuplicateEndpointEnergy);
    }

    let extrapolated_width = final_spacing.max(50.0 * width);
    let xlast = omega[last] + extrapolated_width;
    let slope_scale = extrapolated_width / final_spacing;
    let ylast = spectrum[last] + (spectrum[last] - spectrum[previous]) * slope_scale;

    omega
        .iter()
        .map(|&omega0| {
            let intervals = omega
                .windows(2)
                .zip(spectrum.windows(2))
                .map(|(x_window, y_window)| {
                    conv1(
                        x_window[0],
                        x_window[1],
                        y_window[0],
                        y_window[1],
                        omega0,
                        width,
                    )
                })
                .try_fold(Complex::new(0.0, 0.0), |sum, value| {
                    value.map(|value| sum + value)
                })?;
            let endpoint = conv1(omega[last], xlast, spectrum[last], ylast, omega0, width)?;
            Ok((intervals + endpoint) / FEFF_REAL_PI)
        })
        .collect()
}

fn conv1_component(slope: Real, midpoint: Real, t: Complex) -> Real {
    let slope = Complex::new(slope, 0.0);
    let midpoint = Complex::new(midpoint, 0.0);
    let value = if t.norm() >= 0.1 {
        slope * 2.0
            + (midpoint - slope / t)
                * ((Complex::new(1.0, 0.0) + t) / (Complex::new(1.0, 0.0) - t)).ln()
    } else {
        midpoint * (2.0 * (t + t * t * t / 3.0)) - slope * (2.0 * t * t / 3.0)
    };
    value.im
}

fn validate_inputs(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<(), ConvolutionError> {
    if omega.len() != spectrum.len() {
        return Err(ConvolutionError::LengthMismatch {
            omega_len: omega.len(),
            spectrum_len: spectrum.len(),
        });
    }
    if omega.len() < 2 {
        return Err(ConvolutionError::InsufficientPoints {
            points: omega.len(),
        });
    }
    validate_width(width)?;
    for &energy in omega {
        validate_energy("omega", energy)?;
    }
    for &value in spectrum {
        validate_spectrum("spectrum", value)?;
    }
    Ok(())
}

fn validate_width(width: Real) -> Result<(), ConvolutionError> {
    if !(width.is_finite() && width > 0.0) {
        return Err(ConvolutionError::InvalidWidth { width });
    }
    Ok(())
}

fn validate_energy(name: &'static str, value: Real) -> Result<(), ConvolutionError> {
    if !value.is_finite() {
        return Err(ConvolutionError::NonFiniteEnergy { name, value });
    }
    Ok(())
}

fn validate_spectrum(name: &'static str, value: Complex) -> Result<(), ConvolutionError> {
    if !(value.re.is_finite() && value.im.is_finite()) {
        return Err(ConvolutionError::NonFiniteSpectrum {
            name,
            real: value.re,
            imaginary: value.im,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn fixture() -> (Vec<Real>, Vec<Complex>) {
        (
            vec![-2.0, -0.5, 0.25, 1.5, 3.0],
            vec![
                Complex::new(0.0, -0.2),
                Complex::new(1.0, 0.3),
                Complex::new(0.25, 0.8),
                Complex::new(2.0, -0.1),
                Complex::new(1.5, 0.5),
            ],
        )
    }

    #[test]
    fn conv1_matches_feff_reference() -> Result<(), ConvolutionError> {
        let (omega, spectrum) = fixture();

        let value = conv1(omega[1], omega[2], spectrum[1], spectrum[2], 0.75, 0.2)?;

        assert_complex_close(
            value,
            Complex::new(0.11548114784993968, 0.137_468_645_861_804_9),
        );
        Ok(())
    }

    #[test]
    fn conv_matches_feff_reference() -> Result<(), ConvolutionError> {
        let (omega, spectrum) = fixture();
        let values = conv(&omega, &spectrum, 0.2)?;
        let expected = [
            Complex::new(0.11859251775642239, -0.02108887292319812),
            Complex::new(0.7780612100410474, 0.3262922950470956),
            Complex::new(0.565152671501454, 0.5900533350655084),
            Complex::new(1.6561626472085706, 0.10570908207293085),
            Complex::new(1.4178395137621587, 0.5320419613503408),
        ];

        for (&actual, expected) in values.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn conv_in_place_replaces_spectrum_values() -> Result<(), ConvolutionError> {
        let (omega, mut spectrum) = fixture();

        conv_in_place(&omega, &mut spectrum, 0.2)?;

        assert_complex_close(
            spectrum[0],
            Complex::new(0.11859251775642239, -0.02108887292319812),
        );
        assert_complex_close(
            spectrum[4],
            Complex::new(1.4178395137621587, 0.5320419613503408),
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(matches!(
            conv(&[1.0], &[Complex::new(1.0, 0.0)], 0.2),
            Err(ConvolutionError::InsufficientPoints { .. })
        ));
        assert!(matches!(
            conv(
                &[1.0, 1.0],
                &[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
                0.2
            ),
            Err(ConvolutionError::DuplicateEndpointEnergy)
        ));
        assert!(matches!(
            conv1(
                0.0,
                1.0,
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                0.5,
                0.0
            ),
            Err(ConvolutionError::InvalidWidth { .. })
        ));
    }
}
