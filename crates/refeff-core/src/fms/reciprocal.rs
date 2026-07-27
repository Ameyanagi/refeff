//! Reciprocal-space full multiple-scattering Green-function integration.
//!
//! FEFF `FMS/kkrintegral.f90` evaluates one energy at a time.  For every
//! k-point it forms `(I - G(k) T)`, solves that system against `G(k)`, and
//! accumulates the result in fixed mesh order.  A core-hole impurity is then
//! applied with a local Dyson update.  Keeping the plan and accumulator
//! separate lets callers stream KSPACE structure factors without allocating an
//! `energy x kpoint x state x state` grid.

use ndarray::{Array2, ArrayView2, ShapeBuilder};
use num_complex::Complex32;
use refeff_linalg::{LinalgError, complex32_faer_lu_factor, complex32_faer_lu_solve};
use thiserror::Error;

/// Errors returned by reciprocal-space FMS integration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FmsReciprocalError {
    /// A matrix must be nonempty and square.
    #[error("{name} must be a nonempty square matrix, got {rows}x{columns}")]
    InvalidMatrixShape {
        name: &'static str,
        rows: usize,
        columns: usize,
    },
    /// Two matrices that participate in one solve must have the same shape.
    #[error(
        "{left_name} shape {left_rows}x{left_columns} does not match {right_name} shape {right_rows}x{right_columns}"
    )]
    MatrixShapeMismatch {
        left_name: &'static str,
        left_rows: usize,
        left_columns: usize,
        right_name: &'static str,
        right_rows: usize,
        right_columns: usize,
    },
    /// Reciprocal-space matrices must contain only finite values.
    #[error("{name}[{row},{column}] must be finite, got {value}")]
    NonFiniteMatrixValue {
        name: &'static str,
        row: usize,
        column: usize,
        value: Complex32,
    },
    /// Every streamed k-point must have a positive finite integration weight.
    #[error("reciprocal FMS k-point weight must be positive and finite, got {weight}")]
    InvalidKPointWeight { weight: f64 },
    /// At least one k-point must be accumulated.
    #[error("reciprocal FMS requires at least one k-point")]
    EmptyKPointMesh,
    /// The total k-point integration weight must remain positive and finite.
    #[error("reciprocal FMS total k-point weight must be positive and finite, got {weight}")]
    InvalidTotalKPointWeight { weight: f64 },
    /// A local absorber block must fit in the integrated Green matrix.
    #[error("reciprocal FMS absorber block [{offset}..{end}) exceeds Green-matrix order {order}")]
    AbsorberBlockOutOfRange {
        offset: usize,
        end: usize,
        order: usize,
    },
    /// Linear algebra failed, including singular KKR or Dyson matrices.
    #[error(transparent)]
    Linalg(#[from] LinalgError),
}

/// One-energy reciprocal FMS solve plan.
///
/// FEFF lattice T matrices are sparse (diagonal for one spin and at most one
/// spin-mixing partner per row for two spins).  The plan records nonzero
/// entries column-by-column once so every streamed k-point can assemble
/// `I-GT` without a dense cubic matrix product.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsReciprocalPlan {
    order: usize,
    t_columns: Vec<Vec<(usize, Complex32)>>,
}

impl FmsReciprocalPlan {
    /// Validate and cache one energy's full lattice T matrix.
    pub fn new(t_matrix: ArrayView2<'_, Complex32>) -> Result<Self, FmsReciprocalError> {
        validate_square_matrix("reciprocal_fms_t_matrix", t_matrix)?;
        validate_finite_matrix("reciprocal_fms_t_matrix", t_matrix)?;
        let order = t_matrix.nrows();
        let mut t_columns = Vec::with_capacity(order);
        for column in 0..order {
            let mut entries = Vec::new();
            for row in 0..order {
                let value = t_matrix[(row, column)];
                if value != Complex32::new(0.0, 0.0) {
                    entries.push((row, value));
                }
            }
            t_columns.push(entries);
        }
        Ok(Self { order, t_columns })
    }

    /// Matrix order for this reciprocal solve.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Solve one FEFF k-point contribution `(I-G(k)T)^-1 G(k)`.
    pub fn solve_k_point(
        &self,
        structure_factor: ArrayView2<'_, Complex32>,
    ) -> Result<Array2<Complex32>, FmsReciprocalError> {
        validate_square_matrix("reciprocal_fms_structure_factor", structure_factor)?;
        validate_finite_matrix("reciprocal_fms_structure_factor", structure_factor)?;
        if structure_factor.nrows() != self.order {
            return Err(FmsReciprocalError::MatrixShapeMismatch {
                left_name: "reciprocal_fms_structure_factor",
                left_rows: structure_factor.nrows(),
                left_columns: structure_factor.ncols(),
                right_name: "reciprocal_fms_t_matrix",
                right_rows: self.order,
                right_columns: self.order,
            });
        }

        let mut system = Array2::<Complex32>::zeros((self.order, self.order).f());
        for diagonal in 0..self.order {
            system[(diagonal, diagonal)] = Complex32::new(1.0, 0.0);
        }
        for column in 0..self.order {
            for &(inner, t_value) in &self.t_columns[column] {
                for row in 0..self.order {
                    system[(row, column)] -= structure_factor[(row, inner)] * t_value;
                }
            }
        }

        let lu = complex32_faer_lu_factor(system.view())?;
        let solved = complex32_faer_lu_solve(&lu, structure_factor)?;
        validate_finite_matrix("reciprocal_fms_k_point_green", solved.view())?;
        Ok(solved)
    }
}

/// Fixed-order weighted accumulator for one reciprocal FMS energy.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsReciprocalAccumulator {
    weighted_green: Array2<Complex32>,
    total_weight: f64,
    point_count: usize,
}

impl FmsReciprocalAccumulator {
    /// Create an empty accumulator for a known state order.
    pub fn new(order: usize) -> Result<Self, FmsReciprocalError> {
        if order == 0 {
            return Err(FmsReciprocalError::InvalidMatrixShape {
                name: "reciprocal_fms_accumulator",
                rows: 0,
                columns: 0,
            });
        }
        Ok(Self {
            weighted_green: Array2::zeros((order, order).f()),
            total_weight: 0.0,
            point_count: 0,
        })
    }

    /// Add one k-point result in caller-provided mesh order.
    pub fn push(
        &mut self,
        weight: f64,
        green: ArrayView2<'_, Complex32>,
    ) -> Result<(), FmsReciprocalError> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(FmsReciprocalError::InvalidKPointWeight { weight });
        }
        validate_square_matrix("reciprocal_fms_k_point_green", green)?;
        validate_finite_matrix("reciprocal_fms_k_point_green", green)?;
        if green.dim() != self.weighted_green.dim() {
            return Err(FmsReciprocalError::MatrixShapeMismatch {
                left_name: "reciprocal_fms_k_point_green",
                left_rows: green.nrows(),
                left_columns: green.ncols(),
                right_name: "reciprocal_fms_accumulator",
                right_rows: self.weighted_green.nrows(),
                right_columns: self.weighted_green.ncols(),
            });
        }

        let weight32 = weight as f32;
        for column in 0..green.ncols() {
            for row in 0..green.nrows() {
                self.weighted_green[(row, column)] += green[(row, column)] * weight32;
            }
        }
        self.total_weight += weight;
        self.point_count += 1;
        Ok(())
    }

    /// Finish the Brillouin-zone average.
    pub fn finish(self) -> Result<Array2<Complex32>, FmsReciprocalError> {
        if self.point_count == 0 {
            return Err(FmsReciprocalError::EmptyKPointMesh);
        }
        if !self.total_weight.is_finite() || self.total_weight <= 0.0 {
            return Err(FmsReciprocalError::InvalidTotalKPointWeight {
                weight: self.total_weight,
            });
        }
        let mut green = self.weighted_green;
        green.mapv_inplace(|value| {
            Complex32::new(
                (f64::from(value.re) / self.total_weight) as f32,
                (f64::from(value.im) / self.total_weight) as f32,
            )
        });
        validate_finite_matrix("reciprocal_fms_integrated_green", green.view())?;
        Ok(green)
    }
}

/// Inputs for FEFF's local core-hole Dyson correction.
#[derive(Debug, Clone, Copy)]
pub struct FmsReciprocalCoreHoleInput<'a> {
    /// Ground-state periodic Green matrix after Brillouin-zone integration.
    pub green: ArrayView2<'a, Complex32>,
    /// First state of the absorber site in the lattice state order.
    pub absorber_state_offset: usize,
    /// Number of states in one lattice-site block.
    pub site_block_order: usize,
    /// FEFF `C = strength * (T_ground - T_core_hole)`.
    pub t_difference: ArrayView2<'a, Complex32>,
}

/// Apply FEFF `kkrintegral.f90` core-hole method 1.
pub fn fms_reciprocal_apply_core_hole(
    input: FmsReciprocalCoreHoleInput<'_>,
) -> Result<Array2<Complex32>, FmsReciprocalError> {
    validate_square_matrix("reciprocal_fms_green", input.green)?;
    validate_finite_matrix("reciprocal_fms_green", input.green)?;
    validate_square_matrix("reciprocal_fms_core_hole_t_difference", input.t_difference)?;
    validate_finite_matrix("reciprocal_fms_core_hole_t_difference", input.t_difference)?;
    if input.t_difference.dim() != (input.site_block_order, input.site_block_order) {
        return Err(FmsReciprocalError::MatrixShapeMismatch {
            left_name: "reciprocal_fms_core_hole_t_difference",
            left_rows: input.t_difference.nrows(),
            left_columns: input.t_difference.ncols(),
            right_name: "reciprocal_fms_site_block",
            right_rows: input.site_block_order,
            right_columns: input.site_block_order,
        });
    }
    let end = input
        .absorber_state_offset
        .checked_add(input.site_block_order)
        .unwrap_or(usize::MAX);
    if input.site_block_order == 0 || end > input.green.nrows() {
        return Err(FmsReciprocalError::AbsorberBlockOutOfRange {
            offset: input.absorber_state_offset,
            end,
            order: input.green.nrows(),
        });
    }

    let block = input.site_block_order;
    let offset = input.absorber_state_offset;
    let mut local_green = Array2::<Complex32>::zeros((block, block).f());
    for column in 0..block {
        for row in 0..block {
            local_green[(row, column)] = input.green[(offset + row, offset + column)];
        }
    }

    let c_times_green = matrix_product(input.t_difference, local_green.view());
    let mut dyson = c_times_green;
    for diagonal in 0..block {
        dyson[(diagonal, diagonal)] += Complex32::new(1.0, 0.0);
    }
    let lu = complex32_faer_lu_factor(dyson.view())?;
    let effective_t = complex32_faer_lu_solve(&lu, input.t_difference)?;

    let order = input.green.nrows();
    let mut green_to_absorber = Array2::<Complex32>::zeros((order, block).f());
    let mut absorber_to_green = Array2::<Complex32>::zeros((block, order).f());
    for column in 0..block {
        for row in 0..order {
            green_to_absorber[(row, column)] = input.green[(row, offset + column)];
        }
    }
    for column in 0..order {
        for row in 0..block {
            absorber_to_green[(row, column)] = input.green[(offset + row, column)];
        }
    }

    let left = matrix_product(green_to_absorber.view(), effective_t.view());
    let correction = matrix_product(left.view(), absorber_to_green.view());
    let mut corrected = input.green.to_owned();
    corrected -= &correction;
    validate_finite_matrix("reciprocal_fms_core_hole_green", corrected.view())?;
    Ok(corrected)
}

fn matrix_product(
    left: ArrayView2<'_, Complex32>,
    right: ArrayView2<'_, Complex32>,
) -> Array2<Complex32> {
    debug_assert_eq!(left.ncols(), right.nrows());
    let mut output = Array2::<Complex32>::zeros((left.nrows(), right.ncols()).f());
    for column in 0..right.ncols() {
        for inner in 0..left.ncols() {
            let right_value = right[(inner, column)];
            if right_value == Complex32::new(0.0, 0.0) {
                continue;
            }
            for row in 0..left.nrows() {
                output[(row, column)] += left[(row, inner)] * right_value;
            }
        }
    }
    output
}

fn validate_square_matrix(
    name: &'static str,
    matrix: ArrayView2<'_, Complex32>,
) -> Result<(), FmsReciprocalError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(FmsReciprocalError::InvalidMatrixShape {
            name,
            rows: matrix.nrows(),
            columns: matrix.ncols(),
        });
    }
    Ok(())
}

fn validate_finite_matrix(
    name: &'static str,
    matrix: ArrayView2<'_, Complex32>,
) -> Result<(), FmsReciprocalError> {
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[(row, column)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(FmsReciprocalError::NonFiniteMatrixValue {
                    name,
                    row,
                    column,
                    value,
                });
            }
        }
    }
    Ok(())
}
