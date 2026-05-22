use super::*;

/// Port of FEFF `iniptz`: construct a polarization tensor.
///
/// The returned `3x3` matrix is indexed in signed spherical order
/// `[-1, 0, 1]` on both axes. Selector `10` is the orientational average
/// `diag(1/3, 1/3, 1/3)` in both modes.
pub fn polarization_tensor(
    selector: usize,
    mode: PolarizationTensorMode,
) -> Result<Array2<Complex>, AngularError> {
    if !(1..=10).contains(&selector) {
        return Err(AngularError::InvalidPolarizationTensorIndex { index: selector });
    }

    let mut tensor = Array2::zeros((3, 3).f());
    if selector == 10 {
        for magnetic in -1..=1 {
            tensor[(polarization_index(magnetic), polarization_index(magnetic))] =
                Complex::new(1.0 / 3.0, 0.0);
        }
        return Ok(tensor);
    }

    match mode {
        PolarizationTensorMode::Spherical => {
            let zero_based = selector - 1;
            tensor[(zero_based / 3, zero_based % 3)] = Complex::new(1.0, 0.0);
        }
        PolarizationTensorMode::Cartesian => {
            fill_cartesian_polarization_tensor(&mut tensor, selector);
        }
    }
    Ok(tensor)
}

fn fill_cartesian_polarization_tensor(tensor: &mut Array2<Complex>, selector: usize) {
    let one = Complex::new(1.0, 0.0);
    let imaginary = Complex::new(0.0, 1.0);
    let half = 0.5;
    let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();

    match selector {
        1 => {
            set_polarization(tensor, 1, 1, one * half);
            set_polarization(tensor, -1, -1, one * half);
            set_polarization(tensor, -1, 1, -one * half);
            set_polarization(tensor, 1, -1, -one * half);
        }
        2 => {
            set_polarization(tensor, 1, 1, imaginary * half);
            set_polarization(tensor, -1, -1, -imaginary * half);
            set_polarization(tensor, -1, 1, -imaginary * half);
            set_polarization(tensor, 1, -1, imaginary * half);
        }
        3 => {
            set_polarization(tensor, -1, 0, one * inv_sqrt_two);
            set_polarization(tensor, 1, 0, -one * inv_sqrt_two);
        }
        4 => {
            set_polarization(tensor, 1, 1, -imaginary * half);
            set_polarization(tensor, -1, -1, imaginary * half);
            set_polarization(tensor, -1, 1, -imaginary * half);
            set_polarization(tensor, 1, -1, imaginary * half);
        }
        5 => {
            set_polarization(tensor, 1, 1, one * half);
            set_polarization(tensor, -1, -1, one * half);
            set_polarization(tensor, -1, 1, one * half);
            set_polarization(tensor, 1, -1, one * half);
        }
        6 => {
            set_polarization(tensor, -1, 0, -imaginary * inv_sqrt_two);
            set_polarization(tensor, 1, 0, -imaginary * inv_sqrt_two);
        }
        7 => {
            set_polarization(tensor, 0, -1, one * inv_sqrt_two);
            set_polarization(tensor, 0, 1, -one * inv_sqrt_two);
        }
        8 => {
            set_polarization(tensor, 0, -1, imaginary * inv_sqrt_two);
            set_polarization(tensor, 0, 1, imaginary * inv_sqrt_two);
        }
        9 => set_polarization(tensor, 0, 0, one),
        _ => {}
    }
}

fn set_polarization(tensor: &mut Array2<Complex>, row: isize, column: isize, value: Complex) {
    tensor[(polarization_index(row), polarization_index(column))] = value;
}

fn polarization_index(magnetic: isize) -> usize {
    match magnetic {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 0,
    }
}
