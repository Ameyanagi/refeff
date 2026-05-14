//! FEFF SCREEN helper kernels.
//!
//! These routines cover small, self-contained pieces from `SCREEN/frgrid.f90`
//! and `SCREEN/fxc.f90`. The full SCREEN/CRPA drivers also depend on phase,
//! potential, and FMS handoff state; keeping these kernels separate makes them
//! usable and testable while those drivers are ported incrementally.

use ndarray::Array1;
use thiserror::Error;

use crate::{Complex, ComplexVec, Real, RealVec};

/// Error returned by FEFF SCREEN helper kernels.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ScreenError {
    #[error("SCREEN input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    #[error("SCREEN input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    #[error("SCREEN radial count must be positive")]
    EmptyRadialGrid,
    #[error("SCREEN active radial count {active_count} exceeds input length {len}")]
    ActiveCountOutOfRange { active_count: usize, len: usize },
    #[error("SCREEN radial index is outside isize range: {value}")]
    RadialIndexOutOfRange { value: Real },
    #[error("SCREEN {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    #[error("SCREEN input {upper_name} must exceed {lower_name}: {upper} <= {lower}")]
    NonIncreasingInput {
        lower_name: &'static str,
        upper_name: &'static str,
        lower: Real,
        upper: Real,
    },
    #[error("SCREEN energy grid requires {required} points but capacity is {available}")]
    EnergyGridTooLong { required: usize, available: usize },
    #[error("SCREEN energy grid size overflow for {name}")]
    EnergyGridSizeOverflow { name: &'static str },
    #[error("SCREEN energy grid unexpectedly has no points")]
    EmptyEnergyGrid,
}

/// Inputs for SCREEN `setegi`: rectangular complex-energy contour setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenContourEnergyGridInput {
    /// Lower real-axis energy `emin`.
    pub min_real_energy: Real,
    /// Upper real-axis energy `emax`.
    pub max_real_energy: Real,
    /// Maximum imaginary-axis energy `eimax`.
    pub max_imaginary_energy: Real,
    /// Minimum imaginary-axis offset `ermin`; FEFF clamps non-positive values to 0.05.
    pub min_imaginary_energy: Real,
    /// Number of real-axis divisions `ner`.
    pub real_points: usize,
    /// Number of imaginary-axis divisions `nei`.
    pub imaginary_points: usize,
    /// Capacity of the output energy table, equivalent to FEFF `nex`.
    pub max_points: usize,
}

/// SCREEN complex-energy contour with FEFF's active-length convention.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenContourEnergyGrid {
    /// Complex contour energies `em`, zero-filled after [`ScreenContourEnergyGrid::active_len`].
    pub energies: ComplexVec,
    /// Number of active contour points returned as FEFF `ne`.
    pub active_len: usize,
    /// Effective `ermin` after FEFF's non-positive clamp.
    pub effective_min_imaginary_energy: Real,
}

/// Port of SCREEN `setri`: build the logarithmic radial grid.
///
/// FEFF stores radial samples as `ri(i) = exp(-x0 + (i-1)*dx)` using 1-based
/// loop bounds. This helper returns the same values in Rust's zero-based
/// [`ndarray::Array1`] layout.
pub fn screen_radial_grid(dx: Real, x0: Real, count: usize) -> Result<RealVec, ScreenError> {
    validate_positive("dx", dx)?;
    validate_finite("x0", x0)?;
    if count == 0 {
        return Err(ScreenError::EmptyRadialGrid);
    }

    Ok(Array1::from_iter(
        (0..count).map(|index| (-x0 + index as Real * dx).exp()),
    ))
}

/// Port of SCREEN `SetEGrid`: exponential grid on the imaginary axis.
///
/// FEFF fills `em(ie) = i * (exp((ne-ie)*dx)-1)`, where
/// `dx = log(emax+1)/(ne-1)`. The resulting table runs from `i*emax`
/// down to zero, matching the reference routine's storage order.
pub fn screen_exponential_energy_grid(
    max_imaginary_energy: Real,
    count: usize,
) -> Result<ComplexVec, ScreenError> {
    validate_positive("max_imaginary_energy", max_imaginary_energy)?;
    validate_count_at_least("energy", count, 2)?;

    let denominator = (count - 1) as Real;
    let dx = (max_imaginary_energy + 1.0).ln() / denominator;
    Ok(Array1::from_iter((1..=count).map(|index_1based| {
        let scaled = (count - index_1based) as Real * dx;
        Complex::new(0.0, scaled.exp() - 1.0)
    })))
}

/// Port of SCREEN `setegi`: rectangular complex-energy contour.
///
/// FEFF starts at `emax + i*ermin`, climbs the imaginary branch to `eimax`,
/// steps across the top edge toward `emin`, descends back to `ermin`, and then
/// reverses the table. Non-positive `ermin` is clamped to `0.05` before any
/// step sizes are computed.
pub fn screen_contour_energy_grid(
    input: ScreenContourEnergyGridInput,
) -> Result<ScreenContourEnergyGrid, ScreenError> {
    validate_finite("min_real_energy", input.min_real_energy)?;
    validate_finite("max_real_energy", input.max_real_energy)?;
    validate_finite("max_imaginary_energy", input.max_imaginary_energy)?;
    validate_finite("min_imaginary_energy", input.min_imaginary_energy)?;
    validate_count_at_least("real_points", input.real_points, 2)?;
    validate_count_at_least("imaginary_points", input.imaginary_points, 2)?;
    validate_count_at_least("max_points", input.max_points, 1)?;
    validate_increasing(
        "min_real_energy",
        input.min_real_energy,
        "max_real_energy",
        input.max_real_energy,
    )?;

    let effective_min_imaginary_energy = if input.min_imaginary_energy <= 0.0 {
        0.05
    } else {
        input.min_imaginary_energy
    };
    validate_increasing(
        "min_imaginary_energy",
        effective_min_imaginary_energy,
        "max_imaginary_energy",
        input.max_imaginary_energy,
    )?;

    let max_iterations = input
        .max_points
        .checked_mul(input.max_points)
        .ok_or(ScreenError::EnergyGridSizeOverflow { name: "max_points" })?;
    let real_step =
        (input.max_real_energy - input.min_real_energy) / (input.real_points - 1) as Real;
    let imaginary_step = Complex::new(
        0.0,
        (input.max_imaginary_energy - effective_min_imaginary_energy)
            / (input.imaginary_points - 1) as Real,
    );

    let mut points = Vec::with_capacity(input.max_points.min(max_iterations));
    points.push(Complex::new(
        input.max_real_energy,
        effective_min_imaginary_energy,
    ));
    let mut accumulated_imaginary = effective_min_imaginary_energy;
    let mut delta = imaginary_step;

    for index_1based in 2..=max_iterations {
        let previous = points.last().copied().ok_or(ScreenError::EmptyEnergyGrid)?;
        if previous.re < input.min_real_energy {
            delta = -imaginary_step;
            if previous.im <= effective_min_imaginary_energy {
                let active_len = if previous.im <= 0.0 {
                    index_1based - 2
                } else {
                    index_1based - 1
                };
                points.truncate(active_len);
                break;
            }
        } else if accumulated_imaginary.abs() >= input.max_imaginary_energy {
            delta = Complex::new(-real_step, 0.0);
            accumulated_imaginary = 0.0;
        }

        accumulated_imaginary += delta.im.abs();
        points.push(previous + delta);
    }

    if points.len() > input.max_points {
        return Err(ScreenError::EnergyGridTooLong {
            required: points.len(),
            available: input.max_points,
        });
    }

    let active_len = points.len();
    let mut energies = Array1::<Complex>::zeros(input.max_points);
    for (index, energy) in points.into_iter().rev().enumerate() {
        energies[index] = energy;
    }

    Ok(ScreenContourEnergyGrid {
        energies,
        active_len,
        effective_min_imaginary_energy,
    })
}

/// Port of SCREEN `getiat`: map a radius to FEFF's 1-based radial index.
///
/// Fortran assigns the floating-point expression to an integer, which truncates
/// toward zero. Returning an `isize` preserves that behavior for callers that
/// need to handle out-of-grid locations explicitly. Values reconstructed from
/// the same logarithmic grid are snapped back to exact integer boundaries when
/// roundoff alone would move them just below the FEFF index.
pub fn screen_radial_index_1based(x0: Real, dx: Real, radius: Real) -> Result<isize, ScreenError> {
    validate_finite("x0", x0)?;
    validate_positive("dx", dx)?;
    validate_positive("radius", radius)?;

    let value = (radius.ln() + x0) / dx + 1.0;
    if value < isize::MIN as Real || value > isize::MAX as Real {
        return Err(ScreenError::RadialIndexOutOfRange { value });
    }
    Ok(feff_truncated_index(value))
}

fn feff_truncated_index(value: Real) -> isize {
    let nearest = value.round();
    let tolerance = 1.0e-12 * nearest.abs().max(1.0);
    if value >= 0.0 && (value - nearest).abs() <= tolerance {
        nearest as isize
    } else {
        value.trunc() as isize
    }
}

/// Port of SCREEN `ldafxc`: local-density exchange-correlation kernel.
///
/// FEFF evaluates only the first `active_count` rows, sets non-positive
/// electron-density rows to zero, and uses a pure-exchange branch when
/// `exchange_selector == 2`.
pub fn screen_lda_exchange_correlation_kernel(
    radii: &[Real],
    electron_density: &[Real],
    exchange_selector: i32,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, electron_density.len())?;

    let mut output = Array1::zeros(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let density = electron_density[index];
        validate_positive("radius", radius)?;
        validate_finite("electron_density", density)?;
        if density <= 0.0 {
            continue;
        }

        let rs = (density / 3.0).powf(-1.0 / 3.0);
        let exchange = -1.222 / rs;
        let correlation = if exchange_selector == 2 {
            0.0
        } else {
            -0.75924 / (11.4 + rs)
        };
        output[index] = rs.powi(3) / radius.powi(2) / 6.0 * (exchange + correlation);
    }
    Ok(output)
}

fn validate_active_count(active_count: usize, len: usize) -> Result<(), ScreenError> {
    if active_count > len {
        Err(ScreenError::ActiveCountOutOfRange { active_count, len })
    } else {
        Ok(())
    }
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), ScreenError> {
    if actual < minimum {
        Err(ScreenError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteInput { name, value })
    }
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveInput { name, value })
    }
}

fn validate_increasing(
    lower_name: &'static str,
    lower: Real,
    upper_name: &'static str,
    upper: Real,
) -> Result<(), ScreenError> {
    if upper > lower {
        Ok(())
    } else {
        Err(ScreenError::NonIncreasingInput {
            lower_name,
            upper_name,
            lower,
            upper,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ScreenContourEnergyGridInput, ScreenError, screen_contour_energy_grid,
        screen_exponential_energy_grid, screen_lda_exchange_correlation_kernel, screen_radial_grid,
        screen_radial_index_1based,
    };

    #[test]
    fn exponential_energy_grid_matches_feff_setegrid_reference() -> Result<(), ScreenError> {
        let grid = screen_exponential_energy_grid(8.0, 5)?;

        assert_complex_close(grid[0], 0.0, 8.000_000_000_000_002, 1.0e-14);
        assert_complex_close(grid[1], 0.0, 4.196_152_422_706_632, 1.0e-14);
        assert_complex_close(grid[2], 0.0, 2.000_000_000_000_000_4, 1.0e-14);
        assert_complex_close(grid[3], 0.0, 0.732_050_807_568_877_4, 1.0e-14);
        assert_complex_close(grid[4], 0.0, 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn contour_energy_grid_matches_feff_setegi_reference() -> Result<(), ScreenError> {
        let grid = screen_contour_energy_grid(ScreenContourEnergyGridInput {
            min_real_energy: -0.2,
            max_real_energy: 0.4,
            max_imaginary_energy: 0.5,
            min_imaginary_energy: 0.0,
            real_points: 4,
            imaginary_points: 4,
            max_points: 20,
        })?;

        assert_eq!(grid.active_len, 10);
        assert_close(grid.effective_min_imaginary_energy, 0.05, 1.0e-15);
        assert_complex_close(grid.energies[0], -0.2, 0.05, 1.0e-14);
        assert_complex_close(grid.energies[1], -0.2, 0.2, 1.0e-14);
        assert_complex_close(grid.energies[2], -0.2, 0.35, 1.0e-14);
        assert_complex_close(grid.energies[3], -0.2, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[4], -5.551_115_123_125_783e-17, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[5], 0.2, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[6], 0.4, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[7], 0.4, 0.35, 1.0e-14);
        assert_complex_close(grid.energies[8], 0.4, 0.2, 1.0e-14);
        assert_complex_close(grid.energies[9], 0.4, 0.05, 1.0e-14);
        assert_complex_close(grid.energies[10], 0.0, 0.0, 1.0e-15);
        Ok(())
    }

    #[test]
    fn radial_grid_matches_feff_setri_reference() -> Result<(), ScreenError> {
        let grid = screen_radial_grid(0.05, 8.8, 5)?;

        assert_close(grid[0], 0.000_150_733_075_095_476_5, 1.0e-15);
        assert_close(grid[1], 0.000_158_461_325_115_751_26, 1.0e-15);
        assert_close(grid[2], 0.000_166_585_810_987_633_24, 1.0e-15);
        assert_close(grid[3], 0.000_175_126_848_157_658_42, 1.0e-15);
        assert_close(grid[4], 0.000_184_105_793_667_578_87, 1.0e-15);
        assert_eq!(screen_radial_index_1based(8.8, 0.05, grid[2])?, 3);
        assert_eq!(screen_radial_index_1based(8.8, 0.05, 1.0)?, 177);
        assert_eq!(screen_radial_index_1based(0.0, 1.0, 0.01)?, -3);
        Ok(())
    }

    #[test]
    fn lda_exchange_correlation_kernel_matches_feff_ldafxc_reference() -> Result<(), ScreenError> {
        let radii = [0.5, 0.75, 1.0, 1.5, 2.0];
        let density = [0.04, 0.10, 0.0, -1.0, 0.25];

        let full = screen_lda_exchange_correlation_kernel(&radii, &density, 0, radii.len())?;
        assert_close(full[0], -16.919_199_214_545_813, 1.0e-13);
        assert_close(full[1], -3.960_989_192_391_738_6, 1.0e-13);
        assert_close(full[2], 0.0, 1.0e-15);
        assert_close(full[3], 0.0, 1.0e-15);
        assert_close(full[4], -0.294_609_719_384_913, 1.0e-13);

        let exchange_only =
            screen_lda_exchange_correlation_kernel(&radii, &density, 2, radii.len())?;
        assert_close(exchange_only[0], -14.488_412_060_289_518, 1.0e-13);
        assert_close(exchange_only[1], -3.495_786_749_594_309_6, 1.0e-13);
        assert_close(exchange_only[4], -0.266_878_831_976_939_35, 1.0e-13);
        Ok(())
    }

    #[test]
    fn screen_helpers_reject_invalid_inputs() {
        assert!(matches!(
            screen_radial_grid(0.0, 8.8, 5),
            Err(ScreenError::NonPositiveInput { name: "dx", .. })
        ));
        assert!(matches!(
            screen_radial_grid(0.05, 8.8, 0),
            Err(ScreenError::EmptyRadialGrid)
        ));
        assert!(matches!(
            screen_exponential_energy_grid(8.0, 1),
            Err(ScreenError::CountTooSmall { name: "energy", .. })
        ));
        assert!(matches!(
            screen_radial_index_1based(8.8, 0.05, -1.0),
            Err(ScreenError::NonPositiveInput { name: "radius", .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[1.0], &[0.1], 0, 2),
            Err(ScreenError::ActiveCountOutOfRange { .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[0.0], &[0.1], 0, 1),
            Err(ScreenError::NonPositiveInput { name: "radius", .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[1.0], &[f64::NAN], 0, 1),
            Err(ScreenError::NonFiniteInput {
                name: "electron_density",
                ..
            })
        ));
        assert!(matches!(
            screen_contour_energy_grid(ScreenContourEnergyGridInput {
                min_real_energy: 0.4,
                max_real_energy: 0.4,
                max_imaginary_energy: 0.5,
                min_imaginary_energy: 0.05,
                real_points: 4,
                imaginary_points: 4,
                max_points: 20,
            }),
            Err(ScreenError::NonIncreasingInput {
                upper_name: "max_real_energy",
                ..
            })
        ));
        assert!(matches!(
            screen_contour_energy_grid(ScreenContourEnergyGridInput {
                min_real_energy: -0.2,
                max_real_energy: 0.4,
                max_imaginary_energy: 0.04,
                min_imaginary_energy: 0.0,
                real_points: 4,
                imaginary_points: 4,
                max_points: 20,
            }),
            Err(ScreenError::NonIncreasingInput {
                upper_name: "max_imaginary_energy",
                ..
            })
        ));
        assert!(matches!(
            screen_contour_energy_grid(ScreenContourEnergyGridInput {
                min_real_energy: -0.2,
                max_real_energy: 0.4,
                max_imaginary_energy: 0.5,
                min_imaginary_energy: 0.0,
                real_points: 4,
                imaginary_points: 4,
                max_points: 8,
            }),
            Err(ScreenError::EnergyGridTooLong {
                required: 10,
                available: 8
            })
        ));
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn assert_complex_close(
        actual: crate::Complex,
        expected_re: f64,
        expected_im: f64,
        tolerance: f64,
    ) {
        assert_close(actual.re, expected_re, tolerance);
        assert_close(actual.im, expected_im, tolerance);
    }
}
