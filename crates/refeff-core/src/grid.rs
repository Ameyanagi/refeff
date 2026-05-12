//! FEFF common energy and radial-grid helpers.
//!
//! These functions port the small common routines `getxk.f90`, `xx.f90`, and
//! `m_ifuns.f90`. FEFF uses a 1-based logarithmic radial grid with
//! `x = -8.8 + (j - 1) * delta` and `r = exp(x)`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use thiserror::Error;

use crate::Real;
use crate::interpolation::{InterpolationError, terp};

/// Default FEFF logarithmic radial-grid spacing.
pub const LOUCKS_DELTA: Real = 0.05;

/// Offset used by FEFF's Loucks radial grid.
pub const LOUCKS_X_OFFSET: Real = 8.8;

const SPINOR_ZERO_THRESHOLD: Real = 1.0e-11;

/// Inputs for FEFF `COMMON/fixdsp.f90` spinor grid interpolation.
#[derive(Debug, Clone, Copy)]
pub struct DiracSpinorGridInput<'a> {
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// Original large Dirac component on the source grid.
    pub large_component: ArrayView1<'a, Real>,
    /// Original small Dirac component on the source grid.
    pub small_component: ArrayView1<'a, Real>,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixdsp` spinor components on a target logarithmic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DiracSpinorGrid {
    /// Interpolated large Dirac component.
    pub large_component: Array1<Real>,
    /// Interpolated small Dirac component.
    pub small_component: Array1<Real>,
    /// Number of target-grid points filled before the zero tail.
    pub active_len: usize,
}

/// Inputs for FEFF `COMMON/fixdsx.f90` multi-orbital spinor interpolation.
#[derive(Debug, Clone, Copy)]
pub struct DiracSpinorOrbitalsGridInput<'a> {
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// Original large Dirac components as `(source_radial, orbital)`.
    pub large_components: ArrayView2<'a, Real>,
    /// Original small Dirac components as `(source_radial, orbital)`.
    pub small_components: ArrayView2<'a, Real>,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixdsx` spinor components on a target logarithmic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DiracSpinorOrbitalsGrid {
    /// Interpolated large Dirac components as `(target_radial, orbital)`.
    pub large_components: Array2<Real>,
    /// Interpolated small Dirac components as `(target_radial, orbital)`.
    pub small_components: Array2<Real>,
    /// Per-orbital target-grid active lengths before zero tails.
    pub active_lengths: Array1<usize>,
}

/// Error returned by radial-grid indexing helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GridError {
    /// The radius must be positive and finite before `ln(r)` is meaningful.
    #[error("radius must be positive and finite, got {radius}")]
    InvalidRadius { radius: Real },
    /// The logarithmic grid spacing must be positive and finite.
    #[error("grid delta must be positive and finite, got {delta}")]
    InvalidDelta { delta: Real },
    /// Source spinor component arrays must have matching lengths.
    #[error("spinor component length mismatch: large={large_len}, small={small_len}")]
    SpinorLengthMismatch { large_len: usize, small_len: usize },
    /// Source spinor component tables must have matching shapes.
    #[error(
        "spinor component shape mismatch: large=({large_rows},{large_columns}), small=({small_rows},{small_columns})"
    )]
    SpinorShapeMismatch {
        large_rows: usize,
        large_columns: usize,
        small_rows: usize,
        small_columns: usize,
    },
    /// A grid length must be positive.
    #[error("{name} length must be positive")]
    InvalidGridLength { name: &'static str },
    /// A source or interpolated grid value must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteGridValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// The caller's output grid is too short for FEFF's active interpolation range.
    #[error("output grid length {available} is shorter than required active length {required}")]
    OutputGridTooShort { required: usize, available: usize },
    /// FEFF `terp` failed while resampling grid data.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
}

/// Convert energy in Hartrees to FEFF's signed photoelectron wave number.
///
/// This ports `getxk`: `sqrt(2E)` above the edge and `-sqrt(-2E)` below it.
#[must_use]
pub fn wave_number_from_hartree(energy: Real) -> Real {
    let magnitude = (2.0 * energy).abs().sqrt();
    if energy < 0.0 { -magnitude } else { magnitude }
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a 1-based index.
#[must_use]
pub fn loucks_x(index_1based: usize) -> Real {
    radial_x(index_1based, LOUCKS_DELTA)
}

/// Return the radial coordinate for a 1-based Loucks grid index.
#[must_use]
pub fn loucks_radius(index_1based: usize) -> Real {
    loucks_x(index_1based).exp()
}

/// Return the 1-based Loucks grid index immediately below `radius`.
pub fn loucks_index_below(radius: Real) -> Result<usize, GridError> {
    radial_index_below(radius, LOUCKS_DELTA)
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a custom spacing.
#[must_use]
pub fn radial_x(index_1based: usize, delta: Real) -> Real {
    -LOUCKS_X_OFFSET + (index_1based as Real - 1.0) * delta
}

/// Return the radial coordinate for a custom logarithmic spacing.
#[must_use]
pub fn radial_radius(index_1based: usize, delta: Real) -> Real {
    radial_x(index_1based, delta).exp()
}

/// Return the 1-based grid index immediately below `radius` for a custom spacing.
pub fn radial_index_below(radius: Real, delta: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(GridError::InvalidDelta { delta });
    }
    let index = ((radius.ln() + LOUCKS_X_OFFSET) / delta + 1.0).trunc();
    Ok(index as usize)
}

/// Interpolate one FEFF Dirac spinor pair from `dxorg` to `dxnew`.
///
/// This ports the deterministic numerical part of `COMMON/fixdsp.f90`. FEFF
/// finds the last nonzero source-grid spinor point, adds one source point as
/// the zero boundary, interpolates both components with cubic `terp` on the
/// logarithmic `x` grid, and zero-fills the target tail.
pub fn fix_dirac_spinor_grid(
    input: DiracSpinorGridInput<'_>,
) -> Result<DiracSpinorGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;

    let source_len = input.large_component.len();
    if source_len != input.small_component.len() {
        return Err(GridError::SpinorLengthMismatch {
            large_len: source_len,
            small_len: input.small_component.len(),
        });
    }
    validate_positive_grid_length("source", source_len)?;
    validate_component_values("large_component", input.large_component)?;
    validate_component_values("small_component", input.small_component)?;

    let mut large_component = Array1::<Real>::zeros(input.output_len);
    let mut small_component = Array1::<Real>::zeros(input.output_len);
    let Some(last_nonzero) =
        last_nonzero_spinor_index(input.large_component, input.small_component)
    else {
        return Ok(DiracSpinorGrid {
            large_component,
            small_component,
            active_len: 0,
        });
    };

    let source_window_len = (last_nonzero + 2).min(source_len);
    let source_x = (1..=source_window_len)
        .map(|index| radial_x(index, input.original_delta))
        .collect::<Vec<_>>();
    let source_large = input
        .large_component
        .iter()
        .take(source_window_len)
        .copied()
        .collect::<Vec<_>>();
    let source_small = input
        .small_component
        .iter()
        .take(source_window_len)
        .copied()
        .collect::<Vec<_>>();

    let rmax = radial_radius(source_window_len, input.original_delta);
    let active_len = radial_index_below(rmax, input.new_delta)?;
    if active_len > input.output_len {
        return Err(GridError::OutputGridTooShort {
            required: active_len,
            available: input.output_len,
        });
    }

    for target_index in 1..=active_len {
        let x = radial_x(target_index, input.new_delta);
        let index = target_index - 1;
        large_component[index] = terp(&source_x, &source_large, 3, x)?.value;
        small_component[index] = terp(&source_x, &source_small, 3, x)?.value;
    }

    Ok(DiracSpinorGrid {
        large_component,
        small_component,
        active_len,
    })
}

/// Interpolate FEFF Dirac spinor orbital columns from `dxorg` to `dxnew`.
///
/// This ports the deterministic resampling behavior of `COMMON/fixdsx.f90`.
/// Each orbital column is treated independently with the same zero-tail
/// detection and cubic interpolation used by [`fix_dirac_spinor_grid`].
pub fn fix_dirac_spinor_orbitals_grid(
    input: DiracSpinorOrbitalsGridInput<'_>,
) -> Result<DiracSpinorOrbitalsGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;

    let large_shape = input.large_components.shape();
    let small_shape = input.small_components.shape();
    if large_shape != small_shape {
        return Err(GridError::SpinorShapeMismatch {
            large_rows: large_shape[0],
            large_columns: large_shape[1],
            small_rows: small_shape[0],
            small_columns: small_shape[1],
        });
    }
    validate_positive_grid_length("source", large_shape[0])?;
    validate_positive_grid_length("orbital", large_shape[1])?;

    let orbital_count = large_shape[1];
    let mut large_components = Array2::<Real>::zeros((input.output_len, orbital_count).f());
    let mut small_components = Array2::<Real>::zeros((input.output_len, orbital_count).f());
    let mut active_lengths = Array1::<usize>::zeros(orbital_count);

    for orbital in 0..orbital_count {
        let spinor = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: input.original_delta,
            new_delta: input.new_delta,
            large_component: input.large_components.column(orbital),
            small_component: input.small_components.column(orbital),
            output_len: input.output_len,
        })?;
        large_components
            .column_mut(orbital)
            .assign(&spinor.large_component);
        small_components
            .column_mut(orbital)
            .assign(&spinor.small_component);
        active_lengths[orbital] = spinor.active_len;
    }

    Ok(DiracSpinorOrbitalsGrid {
        large_components,
        small_components,
        active_lengths,
    })
}

fn validate_delta(delta: Real) -> Result<(), GridError> {
    if delta.is_finite() && delta > 0.0 {
        Ok(())
    } else {
        Err(GridError::InvalidDelta { delta })
    }
}

fn validate_positive_grid_length(name: &'static str, len: usize) -> Result<(), GridError> {
    if len > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridLength { name })
    }
}

fn validate_component_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn last_nonzero_spinor_index(
    large_component: ArrayView1<'_, Real>,
    small_component: ArrayView1<'_, Real>,
) -> Option<usize> {
    large_component
        .iter()
        .zip(small_component.iter())
        .enumerate()
        .rev()
        .find_map(|(index, (&large, &small))| {
            (large.abs() >= SPINOR_ZERO_THRESHOLD || small.abs() >= SPINOR_ZERO_THRESHOLD)
                .then_some(index)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ShapeBuilder};

    #[test]
    fn converts_energy_to_signed_wave_number() {
        assert_eq!(wave_number_from_hartree(2.0), 2.0);
        assert_eq!(wave_number_from_hartree(-2.0), -2.0);
        assert_eq!(wave_number_from_hartree(0.0), 0.0);
    }

    #[test]
    fn reproduces_loucks_log_grid_points() {
        assert!((loucks_x(1) + 8.8).abs() < 1.0e-12);
        assert!((loucks_x(2) + 8.75).abs() < 1.0e-12);
        assert!((loucks_radius(1) - (-8.8_f64).exp()).abs() < 1.0e-16);
    }

    #[test]
    fn maps_radius_to_index_below() -> Result<(), GridError> {
        let radius = loucks_radius(42);
        assert_eq!(loucks_index_below(radius)?, 42);

        let midpoint = (loucks_x(42) + 0.5 * LOUCKS_DELTA).exp();
        assert_eq!(loucks_index_below(midpoint)?, 42);
        Ok(())
    }

    #[test]
    fn rejects_invalid_radius_or_delta() {
        assert!(matches!(
            loucks_index_below(0.0),
            Err(GridError::InvalidRadius { .. })
        ));
        assert!(matches!(
            radial_index_below(1.0, 0.0),
            Err(GridError::InvalidDelta { .. })
        ));
    }

    #[test]
    fn fix_dirac_spinor_grid_matches_feff_fixdsp_reference() -> Result<(), GridError> {
        let mut large = vec![0.0; 251];
        let mut small = vec![0.0; 251];
        for i in 1..=80 {
            let i_real = i as Real;
            large[i - 1] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
            small[i - 1] = (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
        }
        let large = Array1::from_vec(large);
        let small = Array1::from_vec(small);

        let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 180,
        })?;

        assert_eq!(result.active_len, 161);
        assert_spinor_value(
            &result,
            1,
            0.098_856_582_548_901_49,
            0.981_461_262_295_415_9,
        );
        assert_spinor_value(&result, 2, 0.146_525_001_614_189, 0.969_970_868_040_543_4);
        assert_spinor_value(
            &result,
            3,
            0.192_879_394_911_354_22,
            0.957_050_307_749_104_5,
        );
        assert_spinor_value(&result, 10, 0.473_738_853_193_487_96, 0.830_355_320_320_026);
        assert_spinor_value(
            &result,
            80,
            -0.310_280_702_093_608_3,
            -0.562_325_207_440_241_6,
        );
        assert_spinor_value(
            &result,
            120,
            -0.008_407_166_503_866_128,
            0.021_105_137_955_943_806,
        );
        assert_spinor_value(
            &result,
            160,
            0.191_266_534_139_204_64,
            0.176_750_359_590_577_94,
        );
        assert_spinor_value(&result, 161, 0.0, 0.0);
        assert_spinor_value(&result, 180, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_grid_zero_fills_empty_spinor() -> Result<(), GridError> {
        let large = Array1::zeros(251);
        let small = Array1::zeros(251);

        let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 16,
        })?;

        assert_eq!(result.active_len, 0);
        assert!(result.large_component.iter().all(|&value| value == 0.0));
        assert!(result.small_component.iter().all(|&value| value == 0.0));
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_grid_rejects_invalid_inputs() {
        let large = Array1::zeros(4);
        let small = Array1::zeros(3);
        assert_eq!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: large.view(),
                small_component: small.view(),
                output_len: 16,
            }),
            Err(GridError::SpinorLengthMismatch {
                large_len: 4,
                small_len: 3,
            })
        );

        let nonfinite = Array1::from_vec(vec![0.0, f64::NAN, 0.0, 0.0]);
        let zeros = Array1::zeros(4);
        assert!(matches!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: nonfinite.view(),
                small_component: zeros.view(),
                output_len: 16,
            }),
            Err(GridError::NonFiniteGridValue {
                name: "large_component",
                index: 1,
                ..
            })
        ));

        assert_eq!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.0,
                new_delta: 0.025,
                large_component: zeros.view(),
                small_component: zeros.view(),
                output_len: 16,
            }),
            Err(GridError::InvalidDelta { delta: 0.0 })
        );
    }

    #[test]
    fn fix_dirac_spinor_orbitals_grid_matches_feff_fixdsx_reference() -> Result<(), GridError> {
        let mut large = Array2::<Real>::zeros((251, 4).f());
        let mut small = Array2::<Real>::zeros((251, 4).f());
        for i in 1..=40 {
            let i_real = i as Real;
            large[(i - 1, 0)] = (0.07 * i_real).sin() * (-0.01 * i_real).exp();
            small[(i - 1, 0)] = (0.05 * i_real).cos() * (-0.02 * i_real).exp();
        }
        for i in 1..=75 {
            let i_real = i as Real;
            large[(i - 1, 2)] = 0.2 * (0.11 * i_real).sin() + 0.002 * i_real;
            small[(i - 1, 2)] = 0.3 * (0.09 * i_real).cos() - 0.001 * i_real;
        }
        for i in 1..=5 {
            let i_real = i as Real;
            large[(i - 1, 3)] = 0.05 * i_real;
            small[(i - 1, 3)] = -0.04 * i_real;
        }

        let result = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_components: large.view(),
            small_components: small.view(),
            output_len: 260,
        })?;

        assert_eq!(result.large_components.shape(), &[260, 4]);
        assert_eq!(result.large_components.strides(), &[1, 260]);
        assert_eq!(result.active_lengths.to_vec(), vec![81, 0, 151, 11]);
        assert_orbital_value(
            &result,
            1,
            1,
            0.069_246_904_378_467_77,
            0.978_973_680_203_922_3,
        );
        assert_orbital_value(&result, 81, 1, 0.0, 0.0);
        assert_orbital_value(&result, 82, 1, 0.0, 0.0);
        assert_orbital_value(&result, 1, 2, 0.0, 0.0);
        assert_orbital_value(&result, 100, 2, 0.0, 0.0);
        assert_orbital_value(
            &result,
            1,
            3,
            0.023_955_660_167_434_965,
            0.297_785_819_903_598_26,
        );
        assert_orbital_value(
            &result,
            150,
            3,
            0.228_834_221_332_933_4,
            0.130_219_461_349_623_98,
        );
        assert_orbital_value(&result, 151, 3, 0.0, 0.0);
        assert_orbital_value(&result, 1, 4, 0.05, -0.04);
        assert_orbital_value(&result, 11, 4, 0.0, 0.0);
        assert_orbital_value(&result, 12, 4, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_orbitals_grid_rejects_shape_mismatch() {
        let large = Array2::<Real>::zeros((4, 2));
        let small = Array2::<Real>::zeros((4, 3));

        assert_eq!(
            fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_components: large.view(),
                small_components: small.view(),
                output_len: 16,
            }),
            Err(GridError::SpinorShapeMismatch {
                large_rows: 4,
                large_columns: 2,
                small_rows: 4,
                small_columns: 3,
            })
        );
    }

    fn assert_spinor_value(
        spinor: &DiracSpinorGrid,
        index_1based: usize,
        expected_large: Real,
        expected_small: Real,
    ) {
        let index = index_1based - 1;
        assert_close(spinor.large_component[index], expected_large);
        assert_close(spinor.small_component[index], expected_small);
    }

    fn assert_orbital_value(
        spinor: &DiracSpinorOrbitalsGrid,
        index_1based: usize,
        orbital_1based: usize,
        expected_large: Real,
        expected_small: Real,
    ) {
        let radial = index_1based - 1;
        let orbital = orbital_1based - 1;
        assert_close(spinor.large_components[(radial, orbital)], expected_large);
        assert_close(spinor.small_components[(radial, orbital)], expected_small);
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}
