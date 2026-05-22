use super::*;

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
