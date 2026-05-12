//! Small vector helpers shared by FEFF input preparation and solvers.
//!
//! These routines keep FEFF's q-vector conventions explicit for NRIxs and
//! reciprocal-space handoff data while staying in simple fixed-size arrays.

use crate::Real;

/// Three-dimensional Cartesian vector.
pub type Vector3 = [Real; 3];

/// Return the FEFF `dist` distance between two Cartesian points.
#[must_use]
pub fn distance_between(left: Vector3, right: Vector3) -> Real {
    vector_norm([left[0] - right[0], left[1] - right[1], left[2] - right[2]])
}

/// Return the FEFF `sdist` distance using single-precision arithmetic.
///
/// Despite the historical comment saying "distance squared", FEFF `sdist`
/// returns the square root. This helper keeps the single-precision arithmetic
/// for paths that need byte-for-byte compatible ordering thresholds.
#[must_use]
pub fn single_precision_distance_between(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Return the Euclidean norm of a 3D vector.
#[must_use]
pub fn vector_norm(vector: Vector3) -> Real {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

/// Return a unit vector, or zero for a zero-length input.
#[must_use]
pub fn normalize_vector(vector: Vector3) -> Vector3 {
    let norm = vector_norm(vector);
    if norm == 0.0 {
        [0.0; 3]
    } else {
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    }
}

/// Rotate `vector` into the reference frame defined by `axis`.
#[must_use]
pub fn rotate_into_reference_frame(vector: Vector3, axis: Vector3) -> Vector3 {
    let Some(rotation) = ReferenceFrameRotation::from_axis(axis) else {
        return vector;
    };
    rotation.rotate(vector)
}

/// Trigonometric q-vector factors written by FEFF for NRIxs calculations.
#[must_use]
pub fn nrixs_qtrig(qvec: Vector3, qnorm: Real) -> [Real; 4] {
    let xy_norm2 = qvec[0] * qvec[0] + qvec[1] * qvec[1];
    if qnorm <= 0.0 {
        return [1.0, 0.0, 1.0, 0.0];
    }
    if xy_norm2 == 0.0 {
        return [-1.0, 0.0, 1.0, 0.0];
    }
    if qvec[2] < 0.0 {
        let xy_norm = xy_norm2.sqrt();
        return [
            qvec[2] / qnorm,
            xy_norm / qnorm,
            qvec[0] / xy_norm,
            qvec[1] / xy_norm,
        ];
    }
    [1.0, 0.0, 1.0, 0.0]
}

/// Rotation coefficients for FEFF's q-vector reference frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceFrameRotation {
    cst: Real,
    snt: Real,
    csf: Real,
    snf: Real,
}

impl ReferenceFrameRotation {
    /// Build the reference-frame rotation for an axis, if a rotation is needed.
    #[must_use]
    pub fn from_axis(axis: Vector3) -> Option<Self> {
        let norm = vector_norm(axis);
        if norm == 0.0 {
            return None;
        }

        let xy_norm2 = axis[0] * axis[0] + axis[1] * axis[1];
        if xy_norm2 == 0.0 && axis[2] >= 0.0 {
            return None;
        }

        let (cst, snt, csf, snf) = if xy_norm2 == 0.0 {
            (-1.0, 0.0, 1.0, 0.0)
        } else {
            let xy_norm = xy_norm2.sqrt();
            (
                axis[2] / norm,
                xy_norm / norm,
                axis[0] / xy_norm,
                axis[1] / xy_norm,
            )
        };

        Some(Self { cst, snt, csf, snf })
    }

    /// Apply this rotation to a vector.
    #[must_use]
    pub fn rotate(self, vector: Vector3) -> Vector3 {
        [
            vector[0] * self.cst * self.csf + vector[1] * self.cst * self.snf
                - vector[2] * self.snt,
            -vector[0] * self.snf + vector[1] * self.csf,
            vector[0] * self.csf * self.snt
                + vector[1] * self.snt * self.snf
                + vector[2] * self.cst,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ReferenceFrameRotation, distance_between, normalize_vector, nrixs_qtrig,
        rotate_into_reference_frame, single_precision_distance_between, vector_norm,
    };

    #[test]
    fn normalizes_vectors() {
        assert_close(vector_norm([3.0, 4.0, 12.0]), 13.0);
        assert_eq!(normalize_vector([0.0, 0.0, 0.0]), [0.0; 3]);
        assert_close(normalize_vector([0.0, 3.0, 4.0])[1], 0.6);
        assert_close(normalize_vector([0.0, 3.0, 4.0])[2], 0.8);
    }

    #[test]
    fn computes_feff_cartesian_distances() {
        let left = [1.0, -2.0, 0.5];
        let right = [-3.0, 4.0, 2.5];

        assert_close(distance_between(left, right), 7.483314773547883);
        assert_close(
            f64::from(single_precision_distance_between(
                [1.0, -2.0, 0.5],
                [-3.0, 4.0, 2.5],
            )),
            7.4833149909973145,
        );
    }

    #[test]
    fn rotates_vectors_into_negative_z_reference_frame() {
        assert_eq!(
            rotate_into_reference_frame([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            rotate_into_reference_frame([1.0, 2.0, 3.0], [0.0, 0.0, -1.0]),
            [-1.0, 2.0, -3.0]
        );
    }

    #[test]
    fn builds_general_reference_rotation() {
        let rotation = ReferenceFrameRotation::from_axis([1.0, 1.0, -2.0]);
        assert!(rotation.is_some());
        let rotated = rotation.map_or([0.0; 3], |rotation| rotation.rotate([1.0, 0.0, 0.0]));

        assert_close(rotated[0], -1.0 / 3.0_f64.sqrt());
        assert_close(rotated[1], -1.0 / 2.0_f64.sqrt());
        assert_close(rotated[2], 1.0 / 6.0_f64.sqrt());
    }

    #[test]
    fn computes_nrixs_qtrig() {
        assert_eq!(nrixs_qtrig([0.0, 0.0, 0.0], 0.0), [1.0, 0.0, 1.0, 0.0]);
        assert_eq!(nrixs_qtrig([0.0, 0.0, 2.0], 2.0), [-1.0, 0.0, 1.0, 0.0]);

        let values = nrixs_qtrig([1.0, 1.0, -2.0_f64.sqrt()], 2.0);
        assert_close(values[0], -1.0 / 2.0_f64.sqrt());
        assert_close(values[1], 1.0 / 2.0_f64.sqrt());
        assert_close(values[2], 1.0 / 2.0_f64.sqrt());
        assert_close(values[3], 1.0 / 2.0_f64.sqrt());
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
