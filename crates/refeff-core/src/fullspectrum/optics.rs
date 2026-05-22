//! FULLSPECTRUM dielectric and optical-constant kernels.

use ndarray::Array1;

use crate::Complex;

use super::constants::{FEFF_ALPHA_INV, FEFF_BOHR_ANGSTROM};
use super::types::*;
use super::validation::{validate_finite_value, validate_matching_len, validate_positive};

/// Convert assembled scattering factors to dielectric contributions.
///
/// This is the `fullspectrum.f90` step immediately after `addedg`: FEFF turns
/// `f` and `f0` into dielectric response with
/// `-4*pi*numden*f/omega**2`, while the original imaginary scattering factor
/// contributes to the `sigma` diagnostic column.
pub fn full_spectrum_scattering_to_dielectric(
    input: FullSpectrumScatteringDielectricInput<'_>,
) -> Result<FullSpectrumScatteringDielectric, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "scattering_to_dielectric",
        });
    }
    validate_matching_len(
        "scattering_factor",
        input.scattering_factor.len(),
        input.omega.len(),
    )?;
    validate_matching_len(
        "background_scattering_factor",
        input.background_scattering_factor.len(),
        input.omega.len(),
    )?;

    let mut epsilon_minus_one = Vec::with_capacity(input.omega.len());
    let mut background_epsilon_minus_one = Vec::with_capacity(input.omega.len());
    let mut sigma = Vec::with_capacity(input.omega.len());
    let density_scale = -4.0 * std::f64::consts::PI * input.number_density;
    let bohr_squared = FEFF_BOHR_ANGSTROM.powi(2);

    for row in 0..input.omega.len() {
        let omega = input.omega[row];
        validate_finite_value("omega", row, omega)?;
        if omega <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value: omega,
            });
        }

        let scattering = input.scattering_factor[row];
        validate_finite_value("scattering_factor real", row, scattering.re)?;
        validate_finite_value("scattering_factor imaginary", row, scattering.im)?;
        let background = input.background_scattering_factor[row];
        validate_finite_value("background_scattering_factor real", row, background.re)?;
        validate_finite_value("background_scattering_factor imaginary", row, background.im)?;

        let scale = density_scale / omega.powi(2);
        let epsilon = scattering * scale;
        let background_epsilon = background * scale;
        let sigma_value = -scattering.im / FEFF_ALPHA_INV / bohr_squared / omega;

        validate_finite_value("epsilon real", row, epsilon.re)?;
        validate_finite_value("epsilon imaginary", row, epsilon.im)?;
        validate_finite_value("background epsilon real", row, background_epsilon.re)?;
        validate_finite_value("background epsilon imaginary", row, background_epsilon.im)?;
        validate_finite_value("sigma", row, sigma_value)?;

        epsilon_minus_one.push(epsilon);
        background_epsilon_minus_one.push(background_epsilon);
        sigma.push(sigma_value);
    }

    Ok(FullSpectrumScatteringDielectric {
        omega: input.omega.to_owned(),
        epsilon_minus_one: Array1::from_vec(epsilon_minus_one),
        background_epsilon_minus_one: Array1::from_vec(background_epsilon_minus_one),
        sigma: Array1::from_vec(sigma),
    })
}

/// Port of `FULLSPECTRUM/opcons.f90`: derive optical constants from `eps - 1`.
///
/// FEFF keeps the dielectric response offset by one in this routine. The
/// returned refractive-index column is also offset by one, matching the
/// `opcons.dat`/`opconsKK.dat` text layout.
pub fn full_spectrum_optical_constants(
    input: FullSpectrumOpticalConstantsInput<'_>,
) -> Result<FullSpectrumOpticalConstants, FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "optical_constants",
        });
    }
    validate_matching_len(
        "epsilon_minus_one",
        input.epsilon_minus_one.len(),
        input.omega.len(),
    )?;

    let mut refractive_index_minus_one = Vec::with_capacity(input.omega.len());
    let mut absorption_coefficient = Vec::with_capacity(input.omega.len());
    let mut reflectivity = Vec::with_capacity(input.omega.len());
    let mut loss = Vec::with_capacity(input.omega.len());
    let one = Complex::new(1.0, 0.0);
    let alpha = 1.0 / FEFF_ALPHA_INV;

    for row in 0..input.omega.len() {
        let omega = input.omega[row];
        validate_finite_value("omega", row, omega)?;
        if omega <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value: omega,
            });
        }

        let epsilon_minus_one = input.epsilon_minus_one[row];
        validate_finite_value("epsilon real", row, epsilon_minus_one.re)?;
        validate_finite_value("epsilon imaginary", row, epsilon_minus_one.im)?;

        let dielectric = epsilon_minus_one + one;
        let refractive_index = dielectric.sqrt();
        let refractive_minus_one = refractive_index - one;
        let absorption = 2.0 * omega * alpha * refractive_index.im / FEFF_BOHR_ANGSTROM * 1000.0;
        let reflectance = ((refractive_index - one) / (refractive_index + one)).norm_sqr();
        let loss_value = -(one / dielectric).im;

        validate_finite_value("refractive index real", row, refractive_minus_one.re)?;
        validate_finite_value("refractive index imaginary", row, refractive_minus_one.im)?;
        validate_finite_value("absorption coefficient", row, absorption)?;
        validate_finite_value("reflectivity", row, reflectance)?;
        validate_finite_value("loss", row, loss_value)?;

        refractive_index_minus_one.push(refractive_minus_one);
        absorption_coefficient.push(absorption);
        reflectivity.push(reflectance);
        loss.push(loss_value);
    }

    Ok(FullSpectrumOpticalConstants {
        omega: input.omega.to_owned(),
        epsilon_minus_one: input.epsilon_minus_one.to_owned(),
        refractive_index_minus_one: Array1::from_vec(refractive_index_minus_one),
        absorption_coefficient: Array1::from_vec(absorption_coefficient),
        reflectivity: Array1::from_vec(reflectivity),
        loss: Array1::from_vec(loss),
    })
}
