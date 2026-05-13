//! FEFF COMPTON grid, rotation, and profile helpers.
//!
//! This module ports the compact numerical kernels from
//! `COMPTON/m_rotation.f90` and the `compton_build_grid`/`jpq` routines in
//! `COMPTON/m_compton.f90`. The routines preserve FEFF's grid and Fourier
//! transform formulas while replacing implicit NaN/Inf behavior with typed
//! validation errors.

use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder};
use num_complex::Complex64;
use thiserror::Error;

use crate::{Real, RealMat, RealVec, Vector3};

const COMPTON_ROTATION_TOLERANCE: Real = 1.0e-10;
const ROTATION_RATIO_TOLERANCE: Real = 1.0e-12;

/// FEFF COMPTON apodization mode used by `jpq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComptonWindow {
    /// FEFF `window = 0`: rectangular cutoff in `z'`.
    Rectangular,
    /// FEFF `window = 1`: squared cosine taper up to the cutoff.
    CosineSquared,
    /// FEFF fallback branch for any other integer `window` value.
    Unwindowed,
}

/// Input values for FEFF `compton_build_grid`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonGridInput {
    /// Number of cylindrical-radius samples, FEFF `ns`.
    pub ns: usize,
    /// Number of azimuth samples, FEFF `nphi`.
    pub nphi: usize,
    /// Number of `z` samples, FEFF `nz`.
    pub nz: usize,
    /// Number of `z'` samples, FEFF `nzp`.
    pub nzp: usize,
    /// Maximum cylindrical radius. A zero value uses [`Self::norman_radius`].
    pub smax: Real,
    /// Maximum azimuth angle.
    pub phimax: Real,
    /// Maximum `z` coordinate. A zero value uses [`Self::norman_radius`].
    pub zmax: Real,
    /// Maximum `z'` coordinate.
    pub zpmax: Real,
    /// FEFF `rnrm(0)` fallback used when `smax` or `zmax` is zero.
    pub norman_radius: Real,
    /// Momentum-transfer direction, FEFF `qhat`.
    pub qhat: Vector3,
}

/// FEFF COMPTON integration grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ComptonGrid {
    /// Cylindrical radial grid `s`.
    pub s: RealVec,
    /// Azimuth grid `phi`.
    pub phi: RealVec,
    /// Longitudinal grid `z`.
    pub z: RealVec,
    /// Companion longitudinal grid `z'`.
    pub zp: RealVec,
    /// Whether FEFF rotates sample points from the q-axis frame.
    pub rotate: bool,
    /// FEFF rotation matrix from q-axis coordinates to cluster coordinates.
    pub rotation_matrix: RealMat,
}

impl ComptonGrid {
    /// Number of cylindrical-radius samples.
    #[must_use]
    pub fn ns(&self) -> usize {
        self.s.len()
    }

    /// Number of azimuth samples.
    #[must_use]
    pub fn nphi(&self) -> usize {
        self.phi.len()
    }

    /// Number of `z` samples.
    #[must_use]
    pub fn nz(&self) -> usize {
        self.z.len()
    }

    /// Number of `z'` samples.
    #[must_use]
    pub fn nzp(&self) -> usize {
        self.zp.len()
    }
}

/// Inputs for FEFF `jpq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonProfileInput {
    /// Projected momentum value `p_q`.
    pub pq: Real,
    /// FEFF window branch.
    pub window: ComptonWindow,
    /// FEFF `window_cutoff`; zero means use the upper end of `grid.zp`.
    pub window_cutoff: Real,
}

/// Rotation axis and angle returned by FEFF `rotation_axis_angle`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonRotationAxisAngle {
    /// Cross-product axis `a x b`.
    pub axis: Vector3,
    /// Rotation angle in radians.
    pub theta: Real,
}

/// Error returned by FEFF COMPTON helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ComptonError {
    /// Scalar inputs must be finite real values.
    #[error("COMPTON input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Vector components must be finite.
    #[error("COMPTON vector {name}[{axis}] must be finite, got {value}")]
    NonFiniteVector {
        name: &'static str,
        axis: usize,
        value: Real,
    },
    /// Grid counts must allow FEFF's `(n - 1)` denominators.
    #[error("COMPTON grid count {name} must be at least 2, got {value}")]
    InvalidGridCount { name: &'static str, value: usize },
    /// Extents used to build linear grids must be nonnegative and finite.
    #[error("COMPTON grid extent {name} must be nonnegative and finite, got {value}")]
    InvalidGridExtent { name: &'static str, value: Real },
    /// A vector norm is required in the corresponding FEFF formula.
    #[error("COMPTON vector {name} must be nonzero")]
    ZeroVector { name: &'static str },
    /// The computed Wigner-style rotation ratio is outside the real asin domain.
    #[error("COMPTON rotation ratio must be in [0, 1], got {value}")]
    InvalidRotationRatio { value: Real },
    /// Rotation matrices must have FEFF's 3x3 shape.
    #[error("COMPTON rotation matrix must have shape (3, 3), got ({rows}, {columns})")]
    InvalidRotationMatrixShape { rows: usize, columns: usize },
    /// `J(z,z')` must match the supplied grid dimensions.
    #[error(
        "COMPTON jzzp shape ({rows}, {columns}) does not match grid shape ({expected_rows}, {expected_columns})"
    )]
    InvalidJzzpShape {
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    /// Piecewise-linear Fourier intervals cannot have zero width.
    #[error("COMPTON {axis} interval {index} has zero width")]
    ZeroFourierInterval { axis: &'static str, index: usize },
    /// Active FEFF windows require a positive cutoff after defaulting.
    #[error("COMPTON window cutoff must be positive, got {value}")]
    InvalidWindowCutoff { value: Real },
    /// A computed result became non-finite.
    #[error("COMPTON result {name} must be finite, got {value}")]
    NonFiniteResult { name: &'static str, value: Real },
}

/// FEFF `cross`: return `a x b` for two 3-vectors.
#[must_use]
pub fn compton_cross_product(a: Vector3, b: Vector3) -> Vector3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Port of FEFF `rotation_axis_angle`.
///
/// FEFF obtains the axis from `a x b` and the angle from
/// `asin(|a x b| / (|a| |b|))`. This intentionally preserves that convention,
/// including the zero angle for parallel and antiparallel vectors.
pub fn compton_rotation_axis_angle(
    a: Vector3,
    b: Vector3,
) -> Result<ComptonRotationAxisAngle, ComptonError> {
    validate_vector("a", a)?;
    validate_vector("b", b)?;
    let a_norm2 = vector_norm2(a);
    let b_norm2 = vector_norm2(b);
    if a_norm2 == 0.0 {
        return Err(ComptonError::ZeroVector { name: "a" });
    }
    if b_norm2 == 0.0 {
        return Err(ComptonError::ZeroVector { name: "b" });
    }

    let axis = compton_cross_product(a, b);
    let ratio = vector_norm2(axis) / (a_norm2 * b_norm2);
    if !ratio.is_finite() {
        return Err(ComptonError::InvalidRotationRatio { value: ratio });
    }
    let clamped_ratio = if ratio > 1.0 && ratio <= 1.0 + ROTATION_RATIO_TOLERANCE {
        1.0
    } else {
        ratio
    };
    if !(0.0..=1.0).contains(&clamped_ratio) {
        return Err(ComptonError::InvalidRotationRatio { value: ratio });
    }

    let theta = clamped_ratio.sqrt().asin();
    if !theta.is_finite() {
        return Err(ComptonError::NonFiniteResult {
            name: "theta",
            value: theta,
        });
    }
    Ok(ComptonRotationAxisAngle { axis, theta })
}

/// Port of FEFF `rotation_matrix`: build a 3x3 axis-angle rotation matrix.
///
/// The returned `ndarray` uses Fortran-order storage to match FEFF matrix
/// traversal while retaining normal Rust `(row, column)` indexing.
pub fn compton_rotation_matrix(axis: Vector3, theta: Real) -> Result<RealMat, ComptonError> {
    validate_vector("axis", axis)?;
    validate_finite("theta", theta)?;
    let axis_norm = vector_norm2(axis).sqrt();
    if axis_norm == 0.0 {
        return Err(ComptonError::ZeroVector { name: "axis" });
    }

    let u = [
        axis[0] / axis_norm,
        axis[1] / axis_norm,
        axis[2] / axis_norm,
    ];
    let (sine, cosine) = theta.sin_cos();
    let delta = 1.0 - cosine;
    let mut rotation = Array2::zeros((3, 3).f());
    rotation[(0, 0)] = cosine + u[0] * u[0] * delta;
    rotation[(0, 1)] = u[0] * u[1] * delta - u[2] * sine;
    rotation[(0, 2)] = u[0] * u[2] * delta + u[1] * sine;
    rotation[(1, 0)] = u[1] * u[0] * delta + u[2] * sine;
    rotation[(1, 1)] = cosine + u[1] * u[1] * delta;
    rotation[(1, 2)] = u[1] * u[2] * delta - u[0] * sine;
    rotation[(2, 0)] = u[2] * u[0] * delta - u[1] * sine;
    rotation[(2, 1)] = u[2] * u[1] * delta + u[0] * sine;
    rotation[(2, 2)] = cosine + u[2] * u[2] * delta;
    validate_matrix_finite("rotation_matrix", rotation.view())?;
    Ok(rotation)
}

/// Port of FEFF `rotate`: multiply a 3x3 matrix by a 3-vector.
pub fn compton_rotate_vector(
    rotation: ArrayView2<'_, Real>,
    vector: Vector3,
) -> Result<Vector3, ComptonError> {
    validate_rotation_shape(rotation)?;
    validate_matrix_finite("rotation", rotation)?;
    validate_vector("vector", vector)?;

    let mut rotated = [0.0; 3];
    for row in 0..3 {
        rotated[row] = (0..3)
            .map(|column| rotation[(row, column)] * vector[column])
            .sum();
    }
    validate_vector("rotated", rotated)?;
    Ok(rotated)
}

/// Port of FEFF `rotate_in_place`.
pub fn compton_rotate_vector_in_place(
    rotation: ArrayView2<'_, Real>,
    vector: &mut Vector3,
) -> Result<(), ComptonError> {
    *vector = compton_rotate_vector(rotation, *vector)?;
    Ok(())
}

/// Port of FEFF `compton_build_grid`.
///
/// `smax` and `zmax` use `norman_radius` when supplied as zero, matching the
/// Fortran assignments from `rnrm(0)`. The returned grid contains the FEFF
/// q-axis rotation matrix or identity when the rotation angle is negligible.
pub fn compton_build_grid(input: ComptonGridInput) -> Result<ComptonGrid, ComptonError> {
    validate_grid_count("ns", input.ns)?;
    validate_grid_count("nphi", input.nphi)?;
    validate_grid_count("nz", input.nz)?;
    validate_grid_count("nzp", input.nzp)?;
    validate_extent("smax", input.smax)?;
    validate_extent("phimax", input.phimax)?;
    validate_extent("zmax", input.zmax)?;
    validate_extent("zpmax", input.zpmax)?;
    validate_vector("qhat", input.qhat)?;

    let smax = default_extent(input.smax, input.norman_radius, "smax", "norman_radius")?;
    let zmax = default_extent(input.zmax, input.norman_radius, "zmax", "norman_radius")?;
    let s = linspace(0.0, smax, input.ns);
    let phi = linspace(0.0, input.phimax, input.nphi);
    let z = linspace(-zmax, zmax, input.nz);
    let zp = linspace(-input.zpmax, input.zpmax, input.nzp);

    let rotation_axis = compton_rotation_axis_angle([0.0, 0.0, 1.0], input.qhat)?;
    let (rotate, rotation_matrix) = if rotation_axis.theta.abs() > COMPTON_ROTATION_TOLERANCE {
        (
            true,
            compton_rotation_matrix(rotation_axis.axis, rotation_axis.theta)?,
        )
    } else {
        (false, identity_rotation_matrix())
    };

    Ok(ComptonGrid {
        s,
        phi,
        z,
        zp,
        rotate,
        rotation_matrix,
    })
}

/// Port of FEFF `jpq`: Fourier transform `J(z,z')` into `J(p_q)`.
///
/// For `pq = 0`, FEFF uses a trapezoid rule over both axes. For nonzero `pq`,
/// it applies a piecewise-linear Fourier transform in `z'` and then `z`.
/// The returned scalar is the real part of FEFF's complex accumulator.
pub fn compton_profile(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    input: ComptonProfileInput,
) -> Result<Real, ComptonError> {
    validate_finite("pq", input.pq)?;
    validate_finite("window_cutoff", input.window_cutoff)?;
    validate_grid_for_profile(grid, jzzp)?;
    validate_matrix_finite("jzzp", jzzp)?;

    let cutoff = match input.window {
        ComptonWindow::Rectangular | ComptonWindow::CosineSquared => {
            let cutoff = if input.window_cutoff == 0.0 {
                grid.zp[grid.nzp() - 1]
            } else {
                input.window_cutoff
            };
            if cutoff <= 0.0 || !cutoff.is_finite() {
                return Err(ComptonError::InvalidWindowCutoff { value: cutoff });
            }
            cutoff
        }
        ComptonWindow::Unwindowed => input.window_cutoff,
    };

    if input.pq == 0.0 {
        return compton_profile_zero_pq(grid, jzzp, input.window, cutoff);
    }
    compton_profile_finite_pq(grid, jzzp, input, cutoff)
}

fn compton_profile_zero_pq(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    window: ComptonWindow,
    cutoff: Real,
) -> Result<Real, ComptonError> {
    let mut profile = 0.0;
    let mut previous_z_integral = 0.0;
    let mut previous_z = 0.0;
    for iz in 0..grid.nz() {
        let z = grid.z[iz];
        let mut z_integral = 0.0;
        let mut previous_zp = grid.zp[0];

        for izp in 1..grid.nzp() {
            let zp = grid.zp[izp];
            let weight = compton_window_weight(window, zp, cutoff);
            z_integral +=
                (jzzp[(iz, izp)] + jzzp[(iz, izp - 1)]) * 0.5 * (zp - previous_zp) * weight;
            previous_zp = zp;
        }

        if iz > 0 {
            profile += (z_integral + previous_z_integral) * 0.5 * (z - previous_z);
        }
        previous_z_integral = z_integral;
        previous_z = z;
    }

    validate_profile_result(profile)
}

fn compton_profile_finite_pq(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    input: ComptonProfileInput,
    cutoff: Real,
) -> Result<Real, ComptonError> {
    let i_over_pq = Complex64::new(0.0, 1.0 / input.pq);
    let i_pq = Complex64::new(0.0, input.pq);
    let minus_i_pq = -i_pq;
    let inv_i_pq = Complex64::new(0.0, -1.0 / input.pq);
    let inv_minus_i_pq = Complex64::new(0.0, 1.0 / input.pq);
    let zp_phases = grid
        .zp
        .iter()
        .map(|&zp| (i_pq * zp).exp())
        .collect::<Vec<_>>();
    let z_phases = grid
        .z
        .iter()
        .map(|&z| (minus_i_pq * z).exp())
        .collect::<Vec<_>>();
    let zp_weights = grid
        .zp
        .iter()
        .map(|&zp| compton_window_weight(input.window, zp, cutoff))
        .collect::<Vec<_>>();
    let mut profile = Complex64::new(0.0, 0.0);
    let mut previous_z_integral = Complex64::new(0.0, 0.0);
    let mut previous_z = 0.0;
    for iz in 0..grid.nz() {
        let z = grid.z[iz];
        let mut z_integral = Complex64::new(0.0, 0.0);
        let mut previous_zp = grid.zp[0];

        for izp in 1..grid.nzp() {
            let zp = grid.zp[izp];
            let dzp = zp - previous_zp;
            if dzp == 0.0 {
                return Err(ComptonError::ZeroFourierInterval {
                    axis: "z'",
                    index: izp,
                });
            }
            let slope = Complex64::new((jzzp[(iz, izp)] - jzzp[(iz, izp - 1)]) / dzp, 0.0);
            let previous_value = Complex64::new(jzzp[(iz, izp - 1)], 0.0);
            let a = previous_value + slope * (Complex64::new(dzp, 0.0) + i_over_pq);
            let b = previous_value + slope * i_over_pq;
            z_integral += (zp_phases[izp] * a * inv_i_pq - zp_phases[izp - 1] * b * inv_i_pq)
                * zp_weights[izp];
            previous_zp = zp;
        }

        if iz > 0 {
            let dz = z - previous_z;
            if dz == 0.0 {
                return Err(ComptonError::ZeroFourierInterval {
                    axis: "z",
                    index: iz,
                });
            }
            let slope = (z_integral - previous_z_integral) / dz;
            let a = previous_z_integral + slope * (Complex64::new(dz, 0.0) - i_over_pq);
            let b = previous_z_integral - slope * i_over_pq;
            profile += z_phases[iz] * a * inv_minus_i_pq - z_phases[iz - 1] * b * inv_minus_i_pq;
        }
        previous_z_integral = z_integral;
        previous_z = z;
    }

    validate_profile_result(profile.re)
}

fn validate_profile_result(profile: Real) -> Result<Real, ComptonError> {
    if !profile.is_finite() {
        return Err(ComptonError::NonFiniteResult {
            name: "profile",
            value: profile,
        });
    }
    Ok(profile)
}

fn compton_window_weight(window: ComptonWindow, zp: Real, cutoff: Real) -> Real {
    match window {
        ComptonWindow::Rectangular => {
            if zp.abs() > cutoff {
                0.0
            } else {
                1.0
            }
        }
        ComptonWindow::CosineSquared => {
            if zp.abs() > cutoff {
                0.0
            } else {
                (std::f64::consts::PI * zp.abs() / (2.0 * cutoff))
                    .cos()
                    .powi(2)
            }
        }
        ComptonWindow::Unwindowed => 1.0,
    }
}

fn validate_grid_for_profile(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
) -> Result<(), ComptonError> {
    validate_grid_count("nz", grid.nz())?;
    validate_grid_count("nzp", grid.nzp())?;
    validate_real_vec("z", &grid.z)?;
    validate_real_vec("zp", &grid.zp)?;
    let (rows, columns) = jzzp.dim();
    if rows != grid.nz() || columns != grid.nzp() {
        return Err(ComptonError::InvalidJzzpShape {
            rows,
            columns,
            expected_rows: grid.nz(),
            expected_columns: grid.nzp(),
        });
    }
    Ok(())
}

fn default_extent(
    value: Real,
    default: Real,
    value_name: &'static str,
    default_name: &'static str,
) -> Result<Real, ComptonError> {
    if value == 0.0 {
        validate_extent(default_name, default)?;
        if default == 0.0 {
            return Err(ComptonError::InvalidGridExtent {
                name: default_name,
                value: default,
            });
        }
        Ok(default)
    } else {
        validate_extent(value_name, value)?;
        Ok(value)
    }
}

fn linspace(start: Real, end: Real, count: usize) -> RealVec {
    let step = (end - start) / (count as Real - 1.0);
    Array1::from_iter((0..count).map(|index| start + step * index as Real))
}

fn identity_rotation_matrix() -> RealMat {
    let mut matrix = Array2::zeros((3, 3).f());
    for axis in 0..3 {
        matrix[(axis, axis)] = 1.0;
    }
    matrix
}

fn validate_real_vec(name: &'static str, values: &RealVec) -> Result<(), ComptonError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteVector {
                name,
                axis: index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_vector(name: &'static str, vector: Vector3) -> Result<(), ComptonError> {
    for (axis, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteVector { name, axis, value });
        }
    }
    Ok(())
}

fn validate_matrix_finite(
    name: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), ComptonError> {
    for &value in &matrix {
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteResult { name, value });
        }
    }
    Ok(())
}

fn validate_rotation_shape(rotation: ArrayView2<'_, Real>) -> Result<(), ComptonError> {
    let (rows, columns) = rotation.dim();
    if rows != 3 || columns != 3 {
        return Err(ComptonError::InvalidRotationMatrixShape { rows, columns });
    }
    Ok(())
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), ComptonError> {
    if !value.is_finite() {
        return Err(ComptonError::NonFiniteInput { name, value });
    }
    Ok(())
}

fn validate_grid_count(name: &'static str, value: usize) -> Result<(), ComptonError> {
    if value < 2 {
        return Err(ComptonError::InvalidGridCount { name, value });
    }
    Ok(())
}

fn validate_extent(name: &'static str, value: Real) -> Result<(), ComptonError> {
    if value < 0.0 || !value.is_finite() {
        return Err(ComptonError::InvalidGridExtent { name, value });
    }
    Ok(())
}

fn vector_norm2(vector: Vector3) -> Real {
    vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
}

#[cfg(test)]
mod tests {
    use ndarray::ShapeBuilder;

    use super::*;

    #[test]
    fn compton_rotation_matches_feff_reference() -> Result<(), ComptonError> {
        let axis_angle = compton_rotation_axis_angle([0.0, 0.0, 1.0], [0.35, -0.25, 0.92])?;
        assert_vector_close(axis_angle.axis, [0.25, 0.35, -0.0], 1.0e-15);
        assert_close(axis_angle.theta, 0.437325754687105, 1.0e-15);

        let rotation = compton_rotation_matrix(axis_angle.axis, axis_angle.theta)?;
        assert_close(rotation[(0, 0)], 0.937682259061576, 1.0e-15);
        assert_close(rotation[(1, 0)], 0.044512672098874, 1.0e-15);
        assert_close(rotation[(2, 0)], -0.344631111572645, 1.0e-15);
        assert_close(rotation[(0, 1)], 0.044512672098874, 1.0e-15);
        assert_close(rotation[(1, 1)], 0.968205234215090, 1.0e-15);
        assert_close(rotation[(2, 1)], 0.246165079694746, 1.0e-15);
        assert_close(rotation[(0, 2)], 0.344631111572645, 1.0e-15);
        assert_close(rotation[(1, 2)], -0.246165079694746, 1.0e-15);
        assert_close(rotation[(2, 2)], 0.905887493276667, 1.0e-15);

        let rotated = compton_rotate_vector(rotation.view(), [0.7, -0.2, 1.1])?;
        assert_vector_close(
            rotated,
            [1.026569269653238, -0.433263764038027, 0.706001448564533],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn compton_grid_matches_feff_reference() -> Result<(), ComptonError> {
        let grid = reference_grid()?;
        assert_eq!(grid.ns(), 4);
        assert_eq!(grid.nphi(), 5);
        assert_eq!(grid.nz(), 4);
        assert_eq!(grid.nzp(), 5);
        assert!(grid.rotate);
        assert_slice_close(
            grid.s.as_slice().unwrap_or(&[]),
            &[0.0, 0.75, 1.5, 2.25],
            1.0e-15,
        );
        assert_slice_close(
            grid.phi.as_slice().unwrap_or(&[]),
            &[
                0.0,
                std::f64::consts::FRAC_PI_4,
                std::f64::consts::FRAC_PI_2,
                3.0 * std::f64::consts::FRAC_PI_4,
                std::f64::consts::PI,
            ],
            1.0e-15,
        );
        assert_slice_close(
            grid.z.as_slice().unwrap_or(&[]),
            &[-1.2, -0.4, 0.4, 1.2],
            1.0e-15,
        );
        assert_slice_close(
            grid.zp.as_slice().unwrap_or(&[]),
            &[-1.5, -0.75, 0.0, 0.75, 1.5],
            1.0e-15,
        );
        assert_close(grid.rotation_matrix[(0, 0)], 0.937682259061576, 1.0e-15);
        assert_close(grid.rotation_matrix[(2, 2)], 0.905887493276667, 1.0e-15);
        Ok(())
    }

    #[test]
    fn compton_profile_matches_feff_reference() -> Result<(), ComptonError> {
        let grid = reference_grid()?;
        let jzzp = reference_jzzp();

        assert_close(
            compton_profile(
                &grid,
                jzzp.view(),
                ComptonProfileInput {
                    pq: 0.0,
                    window: ComptonWindow::Rectangular,
                    window_cutoff: 0.0,
                },
            )?,
            4.481999999999999,
            1.0e-14,
        );
        assert_close(
            compton_profile(
                &grid,
                jzzp.view(),
                ComptonProfileInput {
                    pq: 1.35,
                    window: ComptonWindow::Rectangular,
                    window_cutoff: 0.0,
                },
            )?,
            1.284270509089719,
            1.0e-14,
        );
        assert_close(
            compton_profile(
                &grid,
                jzzp.view(),
                ComptonProfileInput {
                    pq: 1.35,
                    window: ComptonWindow::CosineSquared,
                    window_cutoff: 1.0,
                },
            )?,
            0.541117104926063,
            1.0e-14,
        );
        assert_close(
            compton_profile(
                &grid,
                jzzp.view(),
                ComptonProfileInput {
                    pq: 0.65,
                    window: ComptonWindow::Unwindowed,
                    window_cutoff: 0.0,
                },
            )?,
            3.454016879329959,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn compton_helpers_reject_invalid_inputs() -> Result<(), ComptonError> {
        assert!(matches!(
            compton_rotation_axis_angle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            Err(ComptonError::ZeroVector { name: "a" })
        ));
        assert!(matches!(
            compton_rotation_matrix([0.0, 0.0, 0.0], 0.3),
            Err(ComptonError::ZeroVector { name: "axis" })
        ));
        assert!(matches!(
            compton_build_grid(ComptonGridInput {
                ns: 1,
                ..reference_grid_input()
            }),
            Err(ComptonError::InvalidGridCount {
                name: "ns",
                value: 1
            })
        ));

        let grid = reference_grid()?;
        let bad = Array2::zeros((grid.nz(), grid.nzp() + 1).f());
        assert!(matches!(
            compton_profile(
                &grid,
                bad.view(),
                ComptonProfileInput {
                    pq: 0.0,
                    window: ComptonWindow::Rectangular,
                    window_cutoff: 0.0,
                },
            ),
            Err(ComptonError::InvalidJzzpShape { .. })
        ));
        assert!(matches!(
            compton_profile(
                &grid,
                reference_jzzp().view(),
                ComptonProfileInput {
                    pq: 1.0,
                    window: ComptonWindow::CosineSquared,
                    window_cutoff: -1.0,
                },
            ),
            Err(ComptonError::InvalidWindowCutoff { value: -1.0 })
        ));
        Ok(())
    }

    fn reference_grid_input() -> ComptonGridInput {
        ComptonGridInput {
            ns: 4,
            nphi: 5,
            nz: 4,
            nzp: 5,
            smax: 0.0,
            phimax: std::f64::consts::PI,
            zmax: 1.2,
            zpmax: 1.5,
            norman_radius: 2.25,
            qhat: [0.35, -0.25, 0.92],
        }
    }

    fn reference_grid() -> Result<ComptonGrid, ComptonError> {
        compton_build_grid(reference_grid_input())
    }

    fn reference_jzzp() -> Array2<Real> {
        Array2::from_shape_fn((4, 5).f(), |(iz, izp)| {
            let iz = iz as Real + 1.0;
            let izp = izp as Real + 1.0;
            0.12 * iz + 0.07 * izp + 0.015 * iz * izp
        })
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
            (actual - expected).abs()
        );
    }

    fn assert_vector_close(actual: Vector3, expected: Vector3, tolerance: Real) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(actual, expected, tolerance);
        }
    }

    fn assert_slice_close(actual: &[Real], expected: &[Real], tolerance: Real) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_close(actual, expected, tolerance);
        }
    }
}
