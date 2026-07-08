//! Small vector helpers shared by FEFF input preparation and solvers.
//!
//! These routines keep FEFF's q-vector conventions explicit for NRIxs and
//! reciprocal-space handoff data while staying in simple fixed-size arrays.

use ndarray::{Array2, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::Real;

/// Three-dimensional Cartesian vector.
pub type Vector3 = [Real; 3];

const MOVEH_MAX_NEAREST_DISTANCE: Real = 100.0;
const MOVEH_MAX_HEAVY_BOND_DISTANCE: Real = 10.0;
const MOVEH_EXTENSION_NUMERATOR: Real = 4.0;
const MOVEH_HEAVY_BOND_WEIGHT: Real = 0.95_f32 as Real;
const MOVEH_ORIGINAL_BOND_WEIGHT: Real = 0.05_f32 as Real;
const MOVEH_MAX_ITERATIONS: usize = 64;

/// Inputs for FEFF `POT/moveh.f90` hydrogen bond adjustment.
#[derive(Debug, Clone, Copy)]
pub struct HydrogenBondAdjustmentInput<'a> {
    /// Potential index for each atom, matching FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic number for each potential index, matching FEFF `iz`.
    pub potential_atomic_numbers: ArrayView1<'a, usize>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, matching FEFF `rath`.
    pub positions: ArrayView2<'a, Real>,
}

/// Result of FEFF hydrogen coordinate adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct HydrogenBondAdjustment {
    /// Adjusted coordinates in Bohr as `(atom, xyz)`.
    pub positions: Array2<Real>,
    /// Number of hydrogen atoms whose coordinates changed.
    pub moved_count: usize,
}

/// Error returned by vector and geometry helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum VectorError {
    /// FEFF Cartesian coordinate tables must be shaped `(atoms, 3)`.
    #[error("positions must have shape (atoms, 3), got ({rows}, {columns})")]
    InvalidPositionShape { rows: usize, columns: usize },
    /// Coordinate rows and atom-potential assignments must have matching lengths.
    #[error("atom potential length {potentials} does not match position rows {positions}")]
    AtomPotentialLengthMismatch { potentials: usize, positions: usize },
    /// An atom referenced a potential index outside the atomic-number table.
    #[error(
        "atom {atom_index} references potential index {potential_index}, but only {available} potentials are available"
    )]
    InvalidPotentialIndex {
        atom_index: usize,
        potential_index: usize,
        available: usize,
    },
    /// Coordinate values must be finite.
    #[error("positions[{atom_index},{axis}] must be finite, got {value}")]
    NonFiniteCoordinate {
        atom_index: usize,
        axis: usize,
        value: Real,
    },
    /// FEFF `moveh` needs at least one neighboring atom for every hydrogen.
    #[error("hydrogen atom {atom_index} has no neighboring atom within FEFF moveh search bounds")]
    NoHydrogenNeighbor { atom_index: usize },
    /// FEFF `moveh` divides by the nearest-neighbor H-A distance.
    #[error("hydrogen atom {atom_index} is coincident with nearest atom {neighbor_index}")]
    CoincidentHydrogenNeighbor {
        atom_index: usize,
        neighbor_index: usize,
    },
    /// A denominator in FEFF `moveh` became zero.
    #[error("hydrogen atom {atom_index} produced a zero moveh adjustment denominator")]
    ZeroHydrogenAdjustmentDenominator { atom_index: usize },
    /// The FEFF nearest-neighbor preservation loop did not converge.
    #[error("hydrogen atom {atom_index} did not converge after {iterations} moveh iterations")]
    HydrogenAdjustmentDidNotConverge {
        atom_index: usize,
        iterations: usize,
    },
}

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

/// Increase hydrogen bond lengths using FEFF's `POT/moveh.f90` rule.
///
/// FEFF moves hydrogens away from their nearest non-hydrogen atom to avoid
/// pathological muffin-tin geometry. The adjustment is applied in atom order and
/// uses already-moved coordinates for later hydrogens, matching the in-place
/// Fortran routine.
pub fn adjust_hydrogen_bonds(
    input: HydrogenBondAdjustmentInput<'_>,
) -> Result<HydrogenBondAdjustment, VectorError> {
    validate_moveh_inputs(input)?;

    let mut positions = input.positions.to_owned();
    let mut moved_count = 0;
    for atom_index in 0..positions.nrows() {
        if potential_atomic_number(input, atom_index)? != 1 {
            continue;
        }

        let Some((anchor_index, rah)) =
            nearest_atom(positions.view(), atom_index, MOVEH_MAX_NEAREST_DISTANCE)
        else {
            return Err(VectorError::NoHydrogenNeighbor { atom_index });
        };
        if rah == 0.0 {
            return Err(VectorError::CoincidentHydrogenNeighbor {
                atom_index,
                neighbor_index: anchor_index,
            });
        }
        if potential_atomic_number(input, anchor_index)? == 1 {
            continue;
        }

        let original_position = row_vector(positions.view(), atom_index);
        let mut ratmax = rah + MOVEH_EXTENSION_NUMERATOR / rah.powi(2);
        let rab = shortest_non_hydrogen_bond(positions.view(), input, anchor_index)?
            .unwrap_or(MOVEH_MAX_HEAVY_BOND_DISTANCE);
        if rab < ratmax {
            ratmax = MOVEH_HEAVY_BOND_WEIGHT * rab + MOVEH_ORIGINAL_BOND_WEIGHT * rah;
        }
        if rah > ratmax {
            continue;
        }

        let mut ratmin = rah;
        let mut converged = false;
        for _ in 0..MOVEH_MAX_ITERATIONS {
            move_hydrogen_along_anchor(&mut positions, atom_index, anchor_index, ratmax, ratmin);
            let Some((nearest_index, rbh)) =
                nearest_atom(positions.view(), atom_index, MOVEH_MAX_HEAVY_BOND_DISTANCE)
            else {
                return Err(VectorError::NoHydrogenNeighbor { atom_index });
            };
            if nearest_index == anchor_index {
                converged = true;
                break;
            }

            let rab = distance_between(
                row_vector(positions.view(), anchor_index),
                row_vector(positions.view(), nearest_index),
            );
            let denominator = ratmax.powi(2) + rab.powi(2) - rbh.powi(2);
            if denominator == 0.0 {
                return Err(VectorError::ZeroHydrogenAdjustmentDenominator { atom_index });
            }
            let rattmp = ratmax * rab.powi(2) / denominator;
            ratmin = ratmax;
            ratmax = MOVEH_HEAVY_BOND_WEIGHT * rattmp + MOVEH_ORIGINAL_BOND_WEIGHT * rah;
        }

        if !converged {
            return Err(VectorError::HydrogenAdjustmentDidNotConverge {
                atom_index,
                iterations: MOVEH_MAX_ITERATIONS,
            });
        }
        if row_vector(positions.view(), atom_index) != original_position {
            moved_count += 1;
        }
    }

    Ok(HydrogenBondAdjustment {
        positions,
        moved_count,
    })
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

fn validate_moveh_inputs(input: HydrogenBondAdjustmentInput<'_>) -> Result<(), VectorError> {
    let [rows, columns] = input.positions.shape() else {
        return Err(VectorError::InvalidPositionShape {
            rows: input.positions.nrows(),
            columns: input.positions.ncols(),
        });
    };
    if *columns != 3 {
        return Err(VectorError::InvalidPositionShape {
            rows: *rows,
            columns: *columns,
        });
    }
    if input.atom_potentials.len() != *rows {
        return Err(VectorError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: *rows,
        });
    }
    for atom_index in 0..*rows {
        let potential_index = input.atom_potentials[atom_index];
        if potential_index >= input.potential_atomic_numbers.len() {
            return Err(VectorError::InvalidPotentialIndex {
                atom_index,
                potential_index,
                available: input.potential_atomic_numbers.len(),
            });
        }
        for axis in 0..3 {
            let value = input.positions[(atom_index, axis)];
            if !value.is_finite() {
                return Err(VectorError::NonFiniteCoordinate {
                    atom_index,
                    axis,
                    value,
                });
            }
        }
    }
    Ok(())
}

fn potential_atomic_number(
    input: HydrogenBondAdjustmentInput<'_>,
    atom_index: usize,
) -> Result<usize, VectorError> {
    let potential_index = input.atom_potentials[atom_index];
    input
        .potential_atomic_numbers
        .get(potential_index)
        .copied()
        .ok_or(VectorError::InvalidPotentialIndex {
            atom_index,
            potential_index,
            available: input.potential_atomic_numbers.len(),
        })
}

fn nearest_atom(
    positions: ArrayView2<'_, Real>,
    atom_index: usize,
    initial_distance: Real,
) -> Option<(usize, Real)> {
    (0..positions.nrows())
        .filter(|&candidate| candidate != atom_index)
        .fold(None, |nearest, candidate| {
            let distance = distance_between(
                row_vector(positions, atom_index),
                row_vector(positions, candidate),
            );
            let current_distance = nearest.map_or(initial_distance, |(_, distance)| distance);
            if distance < current_distance {
                Some((candidate, distance))
            } else {
                nearest
            }
        })
}

fn shortest_non_hydrogen_bond(
    positions: ArrayView2<'_, Real>,
    input: HydrogenBondAdjustmentInput<'_>,
    anchor_index: usize,
) -> Result<Option<Real>, VectorError> {
    let mut shortest = None;
    for candidate in 0..positions.nrows() {
        if candidate == anchor_index || potential_atomic_number(input, candidate)? == 1 {
            continue;
        }
        let distance = distance_between(
            row_vector(positions, anchor_index),
            row_vector(positions, candidate),
        );
        let current_distance = shortest.map_or(MOVEH_MAX_HEAVY_BOND_DISTANCE, |distance| distance);
        if distance < current_distance {
            shortest = Some(distance);
        }
    }
    Ok(shortest)
}

fn move_hydrogen_along_anchor(
    positions: &mut Array2<Real>,
    atom_index: usize,
    anchor_index: usize,
    ratmax: Real,
    ratmin: Real,
) {
    let scale = ratmax / ratmin;
    for axis in 0..3 {
        positions[(atom_index, axis)] = positions[(anchor_index, axis)]
            + scale * (positions[(atom_index, axis)] - positions[(anchor_index, axis)]);
    }
}

fn row_vector(positions: ArrayView2<'_, Real>, index: usize) -> Vector3 {
    [
        positions[(index, 0)],
        positions[(index, 1)],
        positions[(index, 2)],
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        HydrogenBondAdjustmentInput, ReferenceFrameRotation, VectorError, adjust_hydrogen_bonds,
        distance_between, normalize_vector, nrixs_qtrig, rotate_into_reference_frame,
        single_precision_distance_between, vector_norm,
    };
    use ndarray::{Array2, arr1, arr2};

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

    #[test]
    fn adjusts_hydrogen_bonds_like_feff_moveh_reference() -> Result<(), VectorError> {
        let potential_atomic_numbers = arr1(&[8, 1]);

        let two_atom = adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
            atom_potentials: arr1(&[0, 1]).view(),
            potential_atomic_numbers: potential_atomic_numbers.view(),
            positions: arr2(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]).view(),
        })?;
        assert_eq!(two_atom.moved_count, 1);
        assert_position(&two_atom.positions, 0, [0.0, 0.0, 0.0]);
        assert_position(&two_atom.positions, 1, [9.524_999_881_163_24, 0.0, 0.0]);

        let capped = adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
            atom_potentials: arr1(&[0, 1, 0]).view(),
            potential_atomic_numbers: potential_atomic_numbers.view(),
            positions: arr2(&[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.0, 0.0, 0.0]]).view(),
        })?;
        assert_eq!(capped.moved_count, 1);
        assert_position(&capped.positions, 0, [0.0, 0.0, 0.0]);
        assert_position(&capped.positions, 1, [9.899_999_886_751_174e-1, 0.0, 0.0]);
        assert_position(&capped.positions, 2, [2.0, 0.0, 0.0]);

        let hh_skip = adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
            atom_potentials: arr1(&[1, 1, 0]).view(),
            potential_atomic_numbers: potential_atomic_numbers.view(),
            positions: arr2(&[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0]]).view(),
        })?;
        assert_eq!(hh_skip.moved_count, 0);
        assert_position(&hh_skip.positions, 0, [0.0, 0.0, 0.0]);
        assert_position(&hh_skip.positions, 1, [0.4, 0.0, 0.0]);
        assert_position(&hh_skip.positions, 2, [3.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn adjust_hydrogen_bonds_rejects_invalid_inputs() {
        let potential_atomic_numbers = arr1(&[8, 1]);
        assert_eq!(
            adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
                atom_potentials: arr1(&[0, 1]).view(),
                potential_atomic_numbers: potential_atomic_numbers.view(),
                positions: arr2(&[[0.0, 0.0], [0.5, 0.0]]).view(),
            }),
            Err(VectorError::InvalidPositionShape {
                rows: 2,
                columns: 2,
            })
        );

        assert_eq!(
            adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
                atom_potentials: arr1(&[0]).view(),
                potential_atomic_numbers: potential_atomic_numbers.view(),
                positions: arr2(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]).view(),
            }),
            Err(VectorError::AtomPotentialLengthMismatch {
                potentials: 1,
                positions: 2,
            })
        );

        assert_eq!(
            adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
                atom_potentials: arr1(&[0, 2]).view(),
                potential_atomic_numbers: potential_atomic_numbers.view(),
                positions: arr2(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]).view(),
            }),
            Err(VectorError::InvalidPotentialIndex {
                atom_index: 1,
                potential_index: 2,
                available: 2,
            })
        );

        assert!(matches!(
            adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
                atom_potentials: arr1(&[0, 1]).view(),
                potential_atomic_numbers: potential_atomic_numbers.view(),
                positions: arr2(&[[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]]).view(),
            }),
            Err(VectorError::NonFiniteCoordinate {
                atom_index: 1,
                axis: 0,
                ..
            })
        ));

        assert_eq!(
            adjust_hydrogen_bonds(HydrogenBondAdjustmentInput {
                atom_potentials: arr1(&[1]).view(),
                potential_atomic_numbers: potential_atomic_numbers.view(),
                positions: arr2(&[[0.0, 0.0, 0.0]]).view(),
            }),
            Err(VectorError::NoHydrogenNeighbor { atom_index: 0 })
        );
    }

    fn assert_position(positions: &Array2<f64>, atom_index: usize, expected: [f64; 3]) {
        for axis in 0..3 {
            assert_close(positions[(atom_index, axis)], expected[axis]);
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
