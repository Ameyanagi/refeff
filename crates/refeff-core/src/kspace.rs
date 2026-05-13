//! Reciprocal-space lattice helpers ported from FEFF BAND/KSPACE routines.
//!
//! The routines here cover the small deterministic helpers used before the
//! heavier KSPACE solvers: Bravais classification from `BAND/ibravais.f90`,
//! high-symmetry K-path segment generation from `BAND/kpath.f90`, point-group
//! operation discovery and closure checks from `KSPACE/pointgroup.f90` and
//! `KSPACE/symmetrycheck.f90`, and the coordinate reductions from
//! `KSPACE/subtract_a.f90` and `change_car.f90`. FEFF exits the process for
//! unsupported lattices; Rust returns typed errors.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use refeff_linalg::{LinalgError, feff_inverse};
use thiserror::Error;

use crate::{Real, RealMat, Vector3};

const PI2: Real = std::f64::consts::TAU;
const POINT_GROUP_EPSILON: Real = 1.0e-8;
const REDUCE_NEGATIVE_EPSILON: Real = -1.0e-8;
const LATTICE_VOLUME_EPSILON: Real = Real::EPSILON;

/// FEFF Bravais lattice selector from `BAND/ibravais.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BravaisLattice {
    /// Triclinic primitive.
    TriclinicPrimitive = 1,
    /// Monoclinic primitive.
    MonoclinicPrimitive = 2,
    /// Monoclinic base-centered.
    MonoclinicBaseCentered = 3,
    /// Orthorhombic primitive.
    OrthorhombicPrimitive = 4,
    /// Orthorhombic base-centered.
    OrthorhombicBaseCentered = 5,
    /// Orthorhombic body-centered.
    OrthorhombicBodyCentered = 6,
    /// Orthorhombic face-centered.
    OrthorhombicFaceCentered = 7,
    /// Tetragonal primitive.
    TetragonalPrimitive = 8,
    /// Tetragonal body-centered.
    TetragonalBodyCentered = 9,
    /// Trigonal primitive.
    TrigonalPrimitive = 10,
    /// Hexagonal primitive.
    HexagonalPrimitive = 11,
    /// Cubic primitive.
    CubicPrimitive = 12,
    /// Cubic face-centered.
    CubicFaceCentered = 13,
    /// Cubic body-centered.
    CubicBodyCentered = 14,
}

impl BravaisLattice {
    /// Return FEFF's integer Bravais index.
    #[must_use]
    pub fn index(self) -> i32 {
        self as i32
    }

    /// Convert FEFF's integer Bravais index into a typed selector.
    pub fn from_index(index: i32) -> Result<Self, KSpaceError> {
        match index {
            1 => Ok(Self::TriclinicPrimitive),
            2 => Ok(Self::MonoclinicPrimitive),
            3 => Ok(Self::MonoclinicBaseCentered),
            4 => Ok(Self::OrthorhombicPrimitive),
            5 => Ok(Self::OrthorhombicBaseCentered),
            6 => Ok(Self::OrthorhombicBodyCentered),
            7 => Ok(Self::OrthorhombicFaceCentered),
            8 => Ok(Self::TetragonalPrimitive),
            9 => Ok(Self::TetragonalBodyCentered),
            10 => Ok(Self::TrigonalPrimitive),
            11 => Ok(Self::HexagonalPrimitive),
            12 => Ok(Self::CubicPrimitive),
            13 => Ok(Self::CubicFaceCentered),
            14 => Ok(Self::CubicBodyCentered),
            _ => Err(KSpaceError::InvalidBravaisIndex { index }),
        }
    }
}

/// High-symmetry path segments returned by FEFF `define_kpath`.
#[derive(Debug, Clone, PartialEq)]
pub struct KPath {
    /// Bravais lattice used to select the segment table.
    pub bravais: BravaisLattice,
    /// User-provided FEFF `KPATH` value.
    pub requested_kpath: i32,
    /// FEFF-adjusted `KPATH` value after defaulting rules.
    pub effective_kpath: i32,
    /// FEFF eight-character segment labels.
    pub labels: Vec<String>,
    /// Segment start vectors as `(segment, xyz)`.
    pub starts: RealMat,
    /// Segment end vectors as `(segment, xyz)`.
    pub ends: RealMat,
}

impl KPath {
    /// Number of active FEFF K-path segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the path contains no active segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Return one segment's start vector.
    #[must_use]
    pub fn start(&self, index: usize) -> Option<Vector3> {
        if index < self.len() {
            Some([
                self.starts[(index, 0)],
                self.starts[(index, 1)],
                self.starts[(index, 2)],
            ])
        } else {
            None
        }
    }

    /// Return one segment's end vector.
    #[must_use]
    pub fn end(&self, index: usize) -> Option<Vector3> {
        if index < self.len() {
            Some([
                self.ends[(index, 0)],
                self.ends[(index, 1)],
                self.ends[(index, 2)],
            ])
        } else {
            None
        }
    }
}

/// Vector reduction result from FEFF reciprocal-space helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedVector {
    /// Reduced vector. `subtract_a` returns reduced coordinates; `reduce`
    /// returns Cartesian coordinates reconstructed inside the unit cell.
    pub vector: Vector3,
    /// Sum of absolute nearest-integer translations FEFF accumulated in `l`.
    pub translation_count: usize,
}

/// Point-group operations returned by FEFF `KSPACE/pointgroup.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointGroup {
    /// Rotation operations as `(operation, row, column)`.
    pub operations: Array3<Real>,
}

impl PointGroup {
    /// Number of point-group operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.shape()[0]
    }

    /// Whether the point group has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return one operation as a fixed 3 by 3 matrix.
    #[must_use]
    pub fn operation(&self, index: usize) -> Option<[[Real; 3]; 3]> {
        if index < self.len() {
            Some([
                [
                    self.operations[(index, 0, 0)],
                    self.operations[(index, 0, 1)],
                    self.operations[(index, 0, 2)],
                ],
                [
                    self.operations[(index, 1, 0)],
                    self.operations[(index, 1, 1)],
                    self.operations[(index, 1, 2)],
                ],
                [
                    self.operations[(index, 2, 0)],
                    self.operations[(index, 2, 1)],
                    self.operations[(index, 2, 2)],
                ],
            ])
        } else {
            None
        }
    }
}

/// FEFF `symmetrycheck` multiplication table and error selector.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryCheck {
    /// FEFF-style multiplication table as `(left_operation, right_operation)`.
    ///
    /// Positive entries are one-based operation indices. A `-1` entry means
    /// the rotational product exists but the corresponding translation check
    /// failed, matching FEFF `mult(i,j) = -1`.
    pub multiplication: Array2<i32>,
    /// FEFF `ierr`: zero when no invalid entries were found, otherwise the
    /// one-based operation index with the largest number of invalid products.
    pub ierr: usize,
}

impl SymmetryCheck {
    /// Number of operations represented by the square multiplication table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.multiplication.nrows()
    }

    /// Whether the check contains no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert FEFF's one-based `ierr` value into a Rust zero-based index.
    #[must_use]
    pub fn invalid_operation_index(&self) -> Option<usize> {
        self.ierr.checked_sub(1)
    }
}

/// Error returned by reciprocal-space helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum KSpaceError {
    /// FEFF space-group numbers are in the crystallographic range 1..=230.
    #[error("space group must be in 1..=230, got {space_group}")]
    InvalidSpaceGroup { space_group: i32 },
    /// FEFF produced no valid Bravais lattice for this lattice centering.
    #[error("invalid Bravais lattice for space group {space_group} and lattice '{lattice}'")]
    InvalidBravaisResult { space_group: i32, lattice: char },
    /// FEFF Bravais indices are 1..=14.
    #[error("invalid Bravais lattice index {index}")]
    InvalidBravaisIndex { index: i32 },
    /// `define_kpath` only implements the FEFF Bravais tables that do not stop.
    #[error("Bravais lattice {bravais} has no FEFF K-path table")]
    UnsupportedBravais { bravais: i32 },
    /// The requested `KPATH` selector is not defined for the Bravais lattice.
    #[error("invalid KPATH {kpath} for Bravais lattice {bravais}")]
    InvalidKPath { bravais: i32, kpath: i32 },
    /// Matrix inputs must be 3 by 3.
    #[error("{name} must have shape (3, 3), got ({rows}, {columns})")]
    InvalidMatrixShape {
        name: &'static str,
        rows: usize,
        columns: usize,
    },
    /// Coordinate values must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// Reduced coordinates must fit in FEFF's nearest-integer translation count.
    #[error("nearest-integer translation for component {component} is too large: {value}")]
    TranslationOverflow { component: usize, value: Real },
    /// The accumulated translation count overflowed a Rust `usize`.
    #[error("translation count overflowed")]
    TranslationCountOverflow,
    /// Internal guard for malformed segment tables.
    #[error(
        "KPATH {kpath} for Bravais lattice {bravais} produced only {available} of {required} segments"
    )]
    KPathDefinitionIncomplete {
        bravais: i32,
        kpath: i32,
        available: usize,
        required: usize,
    },
    /// FEFF `pointgroup` requires a positive output capacity.
    #[error("point-group operation capacity must be positive, got {capacity}")]
    InvalidPointGroupCapacity { capacity: usize },
    /// FEFF `pointgroup` divides by the metric diagonal values.
    #[error("point-group metric diagonal {index} must be positive and non-degenerate, got {value}")]
    DegenerateMetricDiagonal { index: usize, value: Real },
    /// FEFF `pointgroup` stops when a cutoff denominator is too small.
    #[error("point-group metric denominator {index} is degenerate: {value}")]
    DegenerateMetricDenominator { index: usize, value: Real },
    /// Candidate reciprocal-vector enumeration exceeded addressable memory.
    #[error("point-group candidate vector count overflowed")]
    PointGroupSearchOverflow,
    /// FEFF `pointgroup` stops when the supplied operation array is too small.
    #[error("point group produced more than {capacity} operations")]
    TooManyPointGroupOperations { capacity: usize },
    /// FEFF `pointgroup` expects at least one matching operation.
    #[error("no point-group operations matched the supplied metric")]
    NoPointGroupOperations,
    /// FEFF `GBASS` divides by the cell volume.
    #[error("lattice basis has a degenerate reciprocal volume determinant: {determinant}")]
    DegenerateLatticeVolume { determinant: Real },
    /// FEFF `symmetrycheck` requires at least one operation.
    #[error("at least one symmetry operation is required")]
    NoSymmetryOperations,
    /// FEFF `symmetrycheck` expects operations as `(operation, 3, 3)`.
    #[error("symmetry operations must have shape (n, 3, 3), got ({operations}, {rows}, {columns})")]
    InvalidSymmetryOperationShape {
        operations: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF `symmetrycheck` expects one 3-vector translation per operation.
    #[error("symmetry translations must have shape ({operations}, 3), got ({rows}, {columns})")]
    InvalidSymmetryTranslationShape {
        operations: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF-compatible multiplication entries must fit in a signed integer.
    #[error("symmetry operation count {count} does not fit FEFF multiplication entries")]
    SymmetryOperationCountOverflow { count: usize },
    /// FEFF stops when the rotational products are not closed.
    #[error(
        "symmetry product for operations {left} and {right} is missing from the operation table"
    )]
    SymmetryProductMissing { left: usize, right: usize },
    /// FEFF matrix inversion failed while preparing point-group operations.
    #[error(transparent)]
    Linalg(#[from] LinalgError),
}

/// Port of FEFF `BAND/ibravais.f90`.
///
/// `lattice` is interpreted as FEFF does: a one-character centering code such
/// as `P`, `I`, `F`, `C`, or `R`. The result is the typed Bravais selector whose
/// [`BravaisLattice::index`] matches the original integer return value.
pub fn bravais_lattice(space_group: i32, lattice: char) -> Result<BravaisLattice, KSpaceError> {
    if !(1..=230).contains(&space_group) {
        return Err(KSpaceError::InvalidSpaceGroup { space_group });
    }

    let lattice = lattice.to_ascii_uppercase();
    let index = if space_group <= 2 {
        1
    } else if space_group <= 15 {
        if lattice == 'P' { 2 } else { 3 }
    } else if space_group <= 74 {
        match lattice {
            'P' => 4,
            'I' => 6,
            'F' => 7,
            _ => 5,
        }
    } else if space_group <= 142 {
        if lattice == 'P' { 8 } else { 9 }
    } else if space_group <= 167 {
        10
    } else if space_group <= 194 {
        11
    } else {
        match lattice {
            'P' => 12,
            'F' => 13,
            'I' => 14,
            _ => {
                return Err(KSpaceError::InvalidBravaisResult {
                    space_group,
                    lattice,
                });
            }
        }
    };

    BravaisLattice::from_index(index)
}

/// Return FEFF's integer Bravais lattice selector.
pub fn bravais_lattice_index(space_group: i32, lattice: char) -> Result<i32, KSpaceError> {
    bravais_lattice(space_group, lattice).map(BravaisLattice::index)
}

/// Port of FEFF `BAND/kpath.f90` for supported Bravais lattices.
///
/// `reciprocal_basis` stores the three reciprocal basis vectors as
/// `[g1, g2, g3]`. FEFF leaves some lattices as hard `STOP` cases; those return
/// [`KSpaceError::UnsupportedBravais`]. Segment starts and ends are returned as
/// `ndarray` matrices with shape `(segment, xyz)`.
pub fn define_k_path(
    bravais: BravaisLattice,
    kpath: i32,
    reciprocal_basis: [Vector3; 3],
) -> Result<KPath, KSpaceError> {
    validate_basis(reciprocal_basis)?;

    let mut effective_kpath = kpath;
    let mut segments = Vec::with_capacity(12);
    let take = match bravais {
        BravaisLattice::OrthorhombicPrimitive => {
            orthorhombic_primitive_segments(kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::HexagonalPrimitive => {
            hexagonal_primitive_segments(kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::CubicPrimitive => {
            cubic_primitive_segments(kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::CubicFaceCentered => {
            if effective_kpath == 0 {
                effective_kpath = 4;
            }
            cubic_face_centered_segments(effective_kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::CubicBodyCentered => {
            if effective_kpath == 0 {
                effective_kpath = 5;
            }
            cubic_body_centered_segments(effective_kpath, reciprocal_basis, &mut segments)?
        }
        _ => {
            return Err(KSpaceError::UnsupportedBravais {
                bravais: bravais.index(),
            });
        }
    };

    if segments.len() < take {
        return Err(KSpaceError::KPathDefinitionIncomplete {
            bravais: bravais.index(),
            kpath: effective_kpath,
            available: segments.len(),
            required: take,
        });
    }

    let mut labels = Vec::with_capacity(take);
    let mut starts = Array2::<Real>::zeros((take, 3));
    let mut ends = Array2::<Real>::zeros((take, 3));
    for (row, segment) in segments.into_iter().take(take).enumerate() {
        labels.push(segment.label.to_string());
        for axis in 0..3 {
            starts[(row, axis)] = segment.start[axis];
            ends[(row, axis)] = segment.end[axis];
        }
    }

    Ok(KPath {
        bravais,
        requested_kpath: kpath,
        effective_kpath,
        labels,
        starts,
        ends,
    })
}

/// Port of FEFF `KSPACE/subtract_a.f90`.
///
/// The input vector is projected through reciprocal vectors, divided by
/// `2*pi`, and shifted by nearest integers. The returned vector is in reduced
/// lattice coordinates, matching FEFF `r_red` on return from `subtract_a`.
pub fn subtract_lattice_translation(
    reciprocal_vectors: ArrayView2<'_, Real>,
    vector: Vector3,
) -> Result<ReducedVector, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_vector("vector", vector)?;

    let mut reduced = reciprocal_coordinates(reciprocal_vectors, vector);
    let translation_count = shift_reduced_coordinates(&mut reduced, false)?;
    Ok(ReducedVector {
        vector: reduced,
        translation_count,
    })
}

/// Port of FEFF `KSPACE/reduce` from `subtract_a.f90`.
///
/// The input vector is reduced into the unit cell and then transformed back to
/// Cartesian coordinates with the direct lattice vectors.
pub fn reduce_to_lattice_cell(
    direct_vectors: ArrayView2<'_, Real>,
    reciprocal_vectors: ArrayView2<'_, Real>,
    vector: Vector3,
) -> Result<ReducedVector, KSpaceError> {
    validate_matrix("direct_vectors", direct_vectors)?;
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_vector("vector", vector)?;

    let mut reduced = reciprocal_coordinates(reciprocal_vectors, vector);
    let translation_count = shift_reduced_coordinates(&mut reduced, true)?;
    let mut cartesian = [0.0; 3];
    for row in 0..3 {
        for col in 0..3 {
            cartesian[row] += direct_vectors[(row, col)] * reduced[col];
        }
    }
    Ok(ReducedVector {
        vector: cartesian,
        translation_count,
    })
}

/// Port of FEFF `KSPACE/change_car.f90`.
///
/// Calculates `bvs' * operation * avs`, where `operation` is a 3 by 3 integer
/// symmetry matrix. The returned matrix has shape `(3, 3)`.
pub fn change_cartesian_basis(
    reciprocal_vectors: ArrayView2<'_, Real>,
    direct_vectors: ArrayView2<'_, Real>,
    operation: ArrayView2<'_, i32>,
) -> Result<RealMat, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_matrix("direct_vectors", direct_vectors)?;
    if operation.nrows() != 3 || operation.ncols() != 3 {
        return Err(KSpaceError::InvalidMatrixShape {
            name: "operation",
            rows: operation.nrows(),
            columns: operation.ncols(),
        });
    }

    let mut result = Array2::<Real>::zeros((3, 3));
    for row in 0..3 {
        for col in 0..3 {
            for left in 0..3 {
                for right in 0..3 {
                    result[(row, col)] += reciprocal_vectors[(left, row)]
                        * Real::from(operation[(left, right)])
                        * direct_vectors[(right, col)];
                }
            }
        }
    }
    Ok(result)
}

/// Port of FEFF `KSPACE/kmesh.f90` `GBASS`.
///
/// The input basis is stored as columns, matching FEFF `rbas(3,3)`. The result
/// is the reciprocal basis `2*pi * inverse(basis)^T`, also stored as columns.
/// Applying this routine twice returns the original basis within floating-point
/// roundoff, matching FEFF's "real space or vice versa" behavior.
pub fn reciprocal_lattice_vectors(
    lattice_vectors: ArrayView2<'_, Real>,
) -> Result<RealMat, KSpaceError> {
    validate_matrix("lattice_vectors", lattice_vectors)?;

    let mut reciprocal = Array2::<Real>::zeros((3, 3));
    reciprocal[(0, 0)] = lattice_vectors[(1, 1)] * lattice_vectors[(2, 2)]
        - lattice_vectors[(2, 1)] * lattice_vectors[(1, 2)];
    reciprocal[(1, 0)] = lattice_vectors[(2, 1)] * lattice_vectors[(0, 2)]
        - lattice_vectors[(0, 1)] * lattice_vectors[(2, 2)];
    reciprocal[(2, 0)] = lattice_vectors[(0, 1)] * lattice_vectors[(1, 2)]
        - lattice_vectors[(1, 1)] * lattice_vectors[(0, 2)];
    reciprocal[(0, 1)] = lattice_vectors[(1, 2)] * lattice_vectors[(2, 0)]
        - lattice_vectors[(2, 2)] * lattice_vectors[(1, 0)];
    reciprocal[(1, 1)] = lattice_vectors[(2, 2)] * lattice_vectors[(0, 0)]
        - lattice_vectors[(0, 2)] * lattice_vectors[(2, 0)];
    reciprocal[(2, 1)] = lattice_vectors[(0, 2)] * lattice_vectors[(1, 0)]
        - lattice_vectors[(1, 2)] * lattice_vectors[(0, 0)];
    reciprocal[(0, 2)] = lattice_vectors[(1, 0)] * lattice_vectors[(2, 1)]
        - lattice_vectors[(2, 0)] * lattice_vectors[(1, 1)];
    reciprocal[(1, 2)] = lattice_vectors[(2, 0)] * lattice_vectors[(0, 1)]
        - lattice_vectors[(0, 0)] * lattice_vectors[(2, 1)];
    reciprocal[(2, 2)] = lattice_vectors[(0, 0)] * lattice_vectors[(1, 1)]
        - lattice_vectors[(1, 0)] * lattice_vectors[(0, 1)];

    let determinant = (0..3)
        .map(|row| reciprocal[(row, 0)] * lattice_vectors[(row, 0)])
        .sum::<Real>();
    if !determinant.is_finite() || determinant.abs() <= LATTICE_VOLUME_EPSILON {
        return Err(KSpaceError::DegenerateLatticeVolume { determinant });
    }

    let scale = PI2 / determinant;
    reciprocal.mapv_inplace(|value| value * scale);
    for ((row, column), &value) in reciprocal.indexed_iter() {
        validate_vector_component("reciprocal_lattice_vectors", row * 3 + column, value)?;
    }
    Ok(reciprocal)
}

/// Build FEFF's reciprocal-space metric `b(i,j) = dot(b_i, b_j)`.
///
/// `reciprocal_vectors` stores the reciprocal basis vectors as columns,
/// matching `KSPACE/pointgroup.f90` inputs.
pub fn reciprocal_metric(reciprocal_vectors: ArrayView2<'_, Real>) -> Result<RealMat, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    let mut metric = Array2::<Real>::zeros((3, 3));
    for row in 0..3 {
        for col in 0..3 {
            for axis in 0..3 {
                metric[(row, col)] +=
                    reciprocal_vectors[(axis, row)] * reciprocal_vectors[(axis, col)];
            }
        }
    }
    Ok(metric)
}

/// Port of FEFF `KSPACE/pointgroup.f90`.
///
/// `reciprocal_vectors` stores reciprocal basis vectors as columns. `metric`
/// should normally come from [`reciprocal_metric`]. `max_operations` is FEFF's
/// `ntot` capacity; if the lattice produces more operations, a typed error is
/// returned instead of stopping the process.
pub fn point_group_operations(
    reciprocal_vectors: ArrayView2<'_, Real>,
    metric: ArrayView2<'_, Real>,
    max_operations: usize,
) -> Result<PointGroup, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_matrix("metric", metric)?;
    if max_operations == 0 {
        return Err(KSpaceError::InvalidPointGroupCapacity {
            capacity: max_operations,
        });
    }

    let transposed = Array2::from_shape_fn((3, 3), |(row, col)| reciprocal_vectors[(col, row)]);
    let inverse_transposed = feff_inverse(transposed.view())?;
    let cutoffs = point_group_cutoffs(metric)?;
    let candidates = point_group_candidate_vectors(transposed.view(), cutoffs)?;
    let mut operations = Vec::<[[Real; 3]; 3]>::new();

    for first in 0..candidates.len() {
        if (candidates[first].norm - metric[(0, 0)]).abs() > POINT_GROUP_EPSILON {
            continue;
        }
        for second in 0..candidates.len() {
            if second == first
                || (candidates[second].norm - metric[(1, 1)]).abs() > POINT_GROUP_EPSILON
            {
                continue;
            }
            let first_second_dot = dot(candidates[first].vector, candidates[second].vector);
            if (first_second_dot - metric[(1, 0)]).abs() > POINT_GROUP_EPSILON {
                continue;
            }
            for third in 0..candidates.len() {
                if third == first
                    || third == second
                    || (candidates[third].norm - metric[(2, 2)]).abs() > POINT_GROUP_EPSILON
                {
                    continue;
                }
                let first_third_dot = dot(candidates[first].vector, candidates[third].vector);
                let second_third_dot = dot(candidates[second].vector, candidates[third].vector);
                if (first_third_dot - metric[(0, 2)]).abs() <= POINT_GROUP_EPSILON
                    && (second_third_dot - metric[(1, 2)]).abs() <= POINT_GROUP_EPSILON
                {
                    if operations.len() >= max_operations {
                        return Err(KSpaceError::TooManyPointGroupOperations {
                            capacity: max_operations,
                        });
                    }
                    operations.push(point_group_operation(
                        inverse_transposed.view(),
                        candidates[first].vector,
                        candidates[second].vector,
                        candidates[third].vector,
                    ));
                }
            }
        }
    }

    if operations.is_empty() {
        return Err(KSpaceError::NoPointGroupOperations);
    }

    Ok(PointGroup {
        operations: Array3::from_shape_fn((operations.len(), 3, 3), |(op, row, col)| {
            operations[op][row][col]
        }),
    })
}

/// Port of FEFF `KSPACE/symmetrycheck.f90`.
///
/// `operations` stores integer rotation matrices as `(operation, row, column)`.
/// `translations` stores FEFF translation vectors as `(operation, xyz)` in
/// radians. The returned multiplication table intentionally keeps FEFF's
/// positive one-based operation indices and `-1` translation-failure sentinel
/// so callers can compare it directly with legacy outputs.
pub fn symmetry_check(
    operations: ArrayView3<'_, i32>,
    translations: ArrayView2<'_, Real>,
) -> Result<SymmetryCheck, KSpaceError> {
    validate_symmetry_inputs(operations, translations)?;

    let operation_count = operations.shape()[0];
    if operation_count > i32::MAX as usize {
        return Err(KSpaceError::SymmetryOperationCountOverflow {
            count: operation_count,
        });
    }
    if operation_count == 1 {
        return Ok(SymmetryCheck {
            multiplication: Array2::<i32>::zeros((1, 1)),
            ierr: 0,
        });
    }

    let mut multiplication = Array2::<i32>::zeros((operation_count, operation_count));
    for left in 0..operation_count {
        for right in 0..operation_count {
            let product_index = symmetry_product_index(operations, left, right)?.ok_or(
                KSpaceError::SymmetryProductMissing {
                    left: left + 1,
                    right: right + 1,
                },
            )?;
            multiplication[(left, right)] = product_index;
        }
    }

    mark_invalid_symmetry_translations(operations, translations, &mut multiplication);
    let ierr = symmetry_error_index(multiplication.view());
    Ok(SymmetryCheck {
        multiplication,
        ierr,
    })
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    label: &'static str,
    start: Vector3,
    end: Vector3,
}

#[derive(Debug, Clone, Copy)]
struct PointGroupCandidate {
    vector: Vector3,
    norm: Real,
}

fn orthorhombic_primitive_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let y = lc3(0.0, 0.5, 0.0, basis);
    let x = lc3(0.5, 0.0, 0.0, basis);
    let z = lc3(0.0, 0.0, 0.5, basis);
    let u = lc3(0.5, 0.0, 0.5, basis);
    let t = lc3(0.0, 0.5, 0.5, basis);
    let s = lc3(0.5, 0.5, 0.0, basis);
    let r = lc3(0.5, 0.5, 0.5, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 12,
        2 => 7,
        3 => 4,
        4 => 3,
        5 => 1,
        6 => 1,
        7 => 2,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::OrthorhombicPrimitive.index(),
                kpath,
            });
        }
    };

    if !matches!(kpath, 4 | 6 | 7) {
        push_segment(segments, "GG-GS-X ", gamma, x);
        push_segment(segments, "X -G -U ", x, u);
        push_segment(segments, "U -A -Z ", u, z);
        push_segment(segments, "Z -GL-GG", z, gamma);
    }
    if kpath == 7 {
        push_segment(segments, "X -GS-GG", x, gamma);
    }
    push_segment(segments, "GG-GD-Y ", gamma, y);
    push_segment(segments, "Y -H -T ", y, t);
    push_segment(segments, "T -B -Z ", t, z);
    push_segment(segments, "X -D -S ", x, s);
    push_segment(segments, "S -C -Y ", s, y);
    push_segment(segments, "U -P -R ", u, r);
    push_segment(segments, "R -E -T ", r, t);
    push_segment(segments, "S -Q -T ", s, t);
    Ok(take)
}

fn hexagonal_primitive_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let m = lc3(0.0, 0.5, 0.0, basis);
    let a = lc3(0.0, 0.0, 0.5, basis);
    let l = lc3(0.0, 0.5, 0.5, basis);
    let k = lc3(-1.0 / 3.0, 2.0 / 3.0, 0.0, basis);
    let h = lc3(-1.0 / 3.0, 2.0 / 3.0, 0.5, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 9,
        2 => 7,
        3 => 4,
        4 => 1,
        5 => 1,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::HexagonalPrimitive.index(),
                kpath,
            });
        }
    };

    if kpath != 5 {
        push_segment(segments, "GG-GS-M ", gamma, m);
        push_segment(segments, "M -T'-K ", m, k);
    }
    push_segment(segments, "K -T -GG", k, gamma);
    push_segment(segments, "GG-GD-A ", gamma, a);
    push_segment(segments, "A -R -L ", a, l);
    push_segment(segments, "L -S'-H ", l, h);
    push_segment(segments, "H -S -A ", h, a);
    push_segment(segments, "M -U -L ", m, l);
    push_segment(segments, "K -P -H ", k, h);
    Ok(take)
}

fn cubic_primitive_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let x = lc3(0.0, 0.5, 0.0, basis);
    let m = lc3(0.5, 0.5, 0.0, basis);
    let r = lc3(0.5, 0.5, 0.5, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        0 => 1,
        1 => 5,
        2 => 4,
        3 => 3,
        4 => 2,
        5 => 3,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::CubicPrimitive.index(),
                kpath,
            });
        }
    };

    if kpath == 5 {
        push_segment(segments, "GG-GD-X ", gamma, [0.5, 0.0, 0.0]);
        push_segment(segments, "GG-GD-Y ", gamma, [0.0, 0.5, 0.0]);
        push_segment(segments, "GG-GD-Z ", gamma, [0.0, 0.0, 0.5]);
    }
    push_segment(segments, "GG-GD-X ", gamma, x);
    push_segment(segments, "X -Y -M ", x, m);
    push_segment(segments, "M -V -R ", m, r);
    push_segment(segments, "R -GL-GG", r, gamma);
    push_segment(segments, "GG-GS-M ", gamma, m);
    Ok(take)
}

fn cubic_face_centered_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let x = lc3(0.5, 0.0, 0.5, basis);
    let l = lc3(0.5, 0.5, 0.5, basis);
    let w = lc3(0.75, 0.25, 0.5, basis);
    let k = lc3(0.75, 0.375, 0.375, basis);
    let u = lc3(0.625, 0.25, 0.625, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 9,
        2 => 5,
        3 => 2,
        4 => 1,
        5 => 1,
        6 => 3,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::CubicFaceCentered.index(),
                kpath,
            });
        }
    };

    if kpath == 4 {
        push_segment(segments, "GG-GD-X ", gamma, x);
    }
    if kpath == 6 {
        push_segment(segments, "GG-GD-X ", gamma, [1.0, 0.0, 0.0]);
        push_segment(segments, "GG-GD-Y ", gamma, [0.0, 1.0, 0.0]);
        push_segment(segments, "GG-GD-Z ", gamma, [0.0, 0.0, 1.0]);
    }
    if kpath != 5 {
        push_segment(segments, "X -GD-GG", x, gamma);
    }
    push_segment(segments, "GG-GL-L ", gamma, l);
    push_segment(segments, "L -Q -W ", l, w);
    push_segment(segments, "W -N -K ", w, k);
    push_segment(segments, "K -GS-GG", k, gamma);
    push_segment(segments, "L -M -U ", l, u);
    push_segment(segments, "U -S -X ", u, x);
    push_segment(segments, "X -Z -W ", x, w);
    push_segment(segments, "W -B -U ", w, u);
    Ok(take)
}

fn cubic_body_centered_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let h = lc3(0.5, 0.5, -0.5, basis);
    let p = lc3(0.25, 0.25, 0.25, basis);
    let n = lc3(0.5, 0.0, 0.0, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 6,
        2 => 5,
        3 => 4,
        4 => 3,
        5 => 1,
        6 => 3,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::CubicBodyCentered.index(),
                kpath,
            });
        }
    };

    if kpath == 6 {
        push_segment(segments, "GG-GD-X ", gamma, [1.0, 0.0, 0.0]);
        push_segment(segments, "GG-GD-Y ", gamma, [0.0, 1.0, 0.0]);
        push_segment(segments, "GG-GD-Z ", gamma, [0.0, 0.0, 1.0]);
    }
    push_segment(segments, "GG-GD-H ", gamma, h);
    push_segment(segments, "H -G -N ", h, n);
    push_segment(segments, "N -GS-GG", n, gamma);
    push_segment(segments, "GG-GL-P ", gamma, p);
    push_segment(segments, "P -F -H ", p, h);
    push_segment(segments, "N -D -P ", n, p);
    Ok(take)
}

fn push_segment(segments: &mut Vec<Segment>, label: &'static str, start: Vector3, end: Vector3) {
    segments.push(Segment { label, start, end });
}

fn lc3(c1: Real, c2: Real, c3: Real, basis: [Vector3; 3]) -> Vector3 {
    [
        c1 * basis[0][0] + c2 * basis[1][0] + c3 * basis[2][0],
        c1 * basis[0][1] + c2 * basis[1][1] + c3 * basis[2][1],
        c1 * basis[0][2] + c2 * basis[1][2] + c3 * basis[2][2],
    ]
}

fn point_group_cutoffs(metric: ArrayView2<'_, Real>) -> Result<[i32; 3], KSpaceError> {
    let mut max_metric = 0.0;
    let mut diagonal = [0.0; 3];
    for index in 0..3 {
        let value = metric[(index, index)];
        if value <= POINT_GROUP_EPSILON {
            return Err(KSpaceError::DegenerateMetricDiagonal { index, value });
        }
        if value > max_metric {
            max_metric = value;
        }
        diagonal[index] = 1.0 / value;
    }

    let bhelp = metric[(0, 1)] * metric[(0, 2)] * metric[(1, 2)];
    let first = metric[(0, 0)]
        - metric[(0, 1)].powi(2) * diagonal[1]
        - metric[(2, 0)].powi(2) * diagonal[2]
        + 2.0 * bhelp * diagonal[1] * diagonal[2];
    let second = metric[(1, 1)]
        - metric[(0, 1)].powi(2) * diagonal[0]
        - metric[(2, 1)].powi(2) * diagonal[2]
        + 2.0 * bhelp * diagonal[0] * diagonal[2];
    let third = metric[(2, 2)]
        - metric[(0, 2)].powi(2) * diagonal[0]
        - metric[(2, 1)].powi(2) * diagonal[1]
        + 2.0 * bhelp * diagonal[0] * diagonal[1];

    Ok([
        point_group_cutoff(0, max_metric, first)?,
        point_group_cutoff(1, max_metric, second)?,
        point_group_cutoff(2, max_metric, third)?,
    ])
}

fn point_group_cutoff(index: usize, max_metric: Real, value: Real) -> Result<i32, KSpaceError> {
    if value <= POINT_GROUP_EPSILON {
        return Err(KSpaceError::DegenerateMetricDenominator { index, value });
    }
    let cutoff = (max_metric / value).sqrt() + 1.0;
    if cutoff > Real::from(i32::MAX) {
        return Err(KSpaceError::PointGroupSearchOverflow);
    }
    Ok(cutoff as i32)
}

fn point_group_candidate_vectors(
    transposed_basis: ArrayView2<'_, Real>,
    cutoffs: [i32; 3],
) -> Result<Vec<PointGroupCandidate>, KSpaceError> {
    let first_count = point_group_axis_count(cutoffs[0])?;
    let second_count = point_group_axis_count(cutoffs[1])?;
    let third_count = point_group_axis_count(cutoffs[2])?;
    let capacity = first_count
        .checked_mul(second_count)
        .and_then(|value| value.checked_mul(third_count))
        .ok_or(KSpaceError::PointGroupSearchOverflow)?;
    let mut candidates = Vec::with_capacity(capacity);

    for i in -cutoffs[0]..=cutoffs[0] {
        let first = [
            Real::from(i) * transposed_basis[(0, 0)],
            Real::from(i) * transposed_basis[(1, 0)],
            Real::from(i) * transposed_basis[(2, 0)],
        ];
        for j in -cutoffs[1]..=cutoffs[1] {
            let second = [
                first[0] + Real::from(j) * transposed_basis[(0, 1)],
                first[1] + Real::from(j) * transposed_basis[(1, 1)],
                first[2] + Real::from(j) * transposed_basis[(2, 1)],
            ];
            for k in -cutoffs[2]..=cutoffs[2] {
                let vector = [
                    second[0] + Real::from(k) * transposed_basis[(0, 2)],
                    second[1] + Real::from(k) * transposed_basis[(1, 2)],
                    second[2] + Real::from(k) * transposed_basis[(2, 2)],
                ];
                candidates.push(PointGroupCandidate {
                    vector,
                    norm: dot(vector, vector),
                });
            }
        }
    }
    Ok(candidates)
}

fn point_group_axis_count(cutoff: i32) -> Result<usize, KSpaceError> {
    let cutoff = usize::try_from(cutoff).map_err(|_| KSpaceError::PointGroupSearchOverflow)?;
    cutoff
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(KSpaceError::PointGroupSearchOverflow)
}

fn point_group_operation(
    inverse_transposed: ArrayView2<'_, Real>,
    first: Vector3,
    second: Vector3,
    third: Vector3,
) -> [[Real; 3]; 3] {
    let columns = [first, second, third];
    let mut operation = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for axis in 0..3 {
                operation[row][col] += columns[axis][row] * inverse_transposed[(axis, col)];
            }
        }
    }
    operation
}

fn symmetry_product_index(
    operations: ArrayView3<'_, i32>,
    left: usize,
    right: usize,
) -> Result<Option<i32>, KSpaceError> {
    for candidate in 0..operations.shape()[0] {
        if symmetry_product_matches(operations, left, right, candidate) {
            let index = i32::try_from(candidate + 1).map_err(|_| {
                KSpaceError::SymmetryOperationCountOverflow {
                    count: operations.shape()[0],
                }
            })?;
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn symmetry_product_matches(
    operations: ArrayView3<'_, i32>,
    left: usize,
    right: usize,
    candidate: usize,
) -> bool {
    (0..3).all(|row| {
        (0..3).all(|col| {
            let product = (0..3)
                .map(|axis| {
                    i128::from(operations[(left, row, axis)])
                        * i128::from(operations[(right, axis, col)])
                })
                .sum::<i128>();
            product == i128::from(operations[(candidate, row, col)])
        })
    })
}

fn mark_invalid_symmetry_translations(
    operations: ArrayView3<'_, i32>,
    translations: ArrayView2<'_, Real>,
    multiplication: &mut Array2<i32>,
) {
    let operation_count = operations.shape()[0];
    for left in 0..operation_count {
        for right in 0..operation_count {
            let product = multiplication[(left, right)];
            if product <= 0 {
                continue;
            }
            let product_index = product as usize - 1;
            for axis in 0..3 {
                let mut ttest = translations[(right, axis)];
                for component in 0..3 {
                    ttest += Real::from(operations[(left, component, axis)])
                        * (translations[(left, component)]
                            - translations[(product_index, component)]);
                }
                let normalized = ttest.abs() / PI2;
                let integer_part = (normalized * 1.001) as i32;
                if (normalized - Real::from(integer_part)).abs() >= 0.0001 {
                    multiplication[(left, right)] = -1;
                    break;
                }
            }
        }
    }
}

fn symmetry_error_index(multiplication: ArrayView2<'_, i32>) -> usize {
    let operation_count = multiplication.nrows();
    let mut errors = vec![0usize; operation_count];
    for left in 0..operation_count {
        for right in 0..operation_count {
            if multiplication[(left, right)] <= 0 {
                errors[left] += 1;
                errors[right] += 1;
            }
        }
    }

    let mut ierr = 0usize;
    let mut max_errors = 0usize;
    for (index, count) in errors.into_iter().enumerate() {
        if count > max_errors {
            max_errors = count;
            ierr = index + 1;
        }
    }
    ierr
}

fn dot(left: Vector3, right: Vector3) -> Real {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn reciprocal_coordinates(reciprocal_vectors: ArrayView2<'_, Real>, vector: Vector3) -> Vector3 {
    let mut reduced = [0.0; 3];
    for row in 0..3 {
        for col in 0..3 {
            reduced[row] += reciprocal_vectors[(row, col)] * vector[col];
        }
        reduced[row] /= PI2;
    }
    reduced
}

fn shift_reduced_coordinates(
    reduced: &mut Vector3,
    wrap_negative: bool,
) -> Result<usize, KSpaceError> {
    let mut translation_count = 0usize;
    for (component, value) in reduced.iter_mut().enumerate() {
        let shift = nearest_integer(*value, component)?;
        let magnitude = usize::try_from(shift.unsigned_abs())
            .map_err(|_| KSpaceError::TranslationCountOverflow)?;
        translation_count = translation_count
            .checked_add(magnitude)
            .ok_or(KSpaceError::TranslationCountOverflow)?;
        *value -= shift as Real;
        if wrap_negative {
            if *value < 0.0 && *value > REDUCE_NEGATIVE_EPSILON {
                *value = 0.0;
            } else if *value < 0.0 {
                *value += 1.0;
            }
        }
    }
    Ok(translation_count)
}

fn nearest_integer(value: Real, component: usize) -> Result<i64, KSpaceError> {
    let rounded = value.round();
    if rounded < i64::MIN as Real || rounded > i64::MAX as Real {
        return Err(KSpaceError::TranslationOverflow { component, value });
    }
    Ok(rounded as i64)
}

fn validate_basis(basis: [Vector3; 3]) -> Result<(), KSpaceError> {
    for (vector_index, vector) in basis.into_iter().enumerate() {
        validate_vector_component("reciprocal_basis", vector_index * 3, vector[0])?;
        validate_vector_component("reciprocal_basis", vector_index * 3 + 1, vector[1])?;
        validate_vector_component("reciprocal_basis", vector_index * 3 + 2, vector[2])?;
    }
    Ok(())
}

fn validate_vector(name: &'static str, vector: Vector3) -> Result<(), KSpaceError> {
    for (index, value) in vector.into_iter().enumerate() {
        validate_vector_component(name, index, value)?;
    }
    Ok(())
}

fn validate_matrix(name: &'static str, matrix: ArrayView2<'_, Real>) -> Result<(), KSpaceError> {
    if matrix.nrows() != 3 || matrix.ncols() != 3 {
        return Err(KSpaceError::InvalidMatrixShape {
            name,
            rows: matrix.nrows(),
            columns: matrix.ncols(),
        });
    }
    for ((row, column), &value) in matrix.indexed_iter() {
        validate_vector_component(name, row * 3 + column, value)?;
    }
    Ok(())
}

fn validate_symmetry_inputs(
    operations: ArrayView3<'_, i32>,
    translations: ArrayView2<'_, Real>,
) -> Result<(), KSpaceError> {
    let shape = operations.shape();
    let operation_count = shape[0];
    if operation_count == 0 {
        return Err(KSpaceError::NoSymmetryOperations);
    }
    if shape[1] != 3 || shape[2] != 3 {
        return Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: operation_count,
            rows: shape[1],
            columns: shape[2],
        });
    }
    if translations.nrows() != operation_count || translations.ncols() != 3 {
        return Err(KSpaceError::InvalidSymmetryTranslationShape {
            operations: operation_count,
            rows: translations.nrows(),
            columns: translations.ncols(),
        });
    }
    for ((row, column), &value) in translations.indexed_iter() {
        validate_vector_component("translations", row * 3 + column, value)?;
    }
    Ok(())
}

fn validate_vector_component(
    name: &'static str,
    index: usize,
    value: Real,
) -> Result<(), KSpaceError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(KSpaceError::NonFiniteValue { name, index, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, array};

    use super::*;

    const BASIS: [Vector3; 3] = [[1.1, 0.2, 0.05], [-0.1, 1.3, 0.04], [0.03, 0.2, 0.9]];

    #[test]
    fn bravais_lattice_matches_feff_ibravais_reference() -> Result<(), KSpaceError> {
        let cases = [
            (1, 'P', BravaisLattice::TriclinicPrimitive),
            (2, 'P', BravaisLattice::TriclinicPrimitive),
            (3, 'P', BravaisLattice::MonoclinicPrimitive),
            (15, 'C', BravaisLattice::MonoclinicBaseCentered),
            (16, 'P', BravaisLattice::OrthorhombicPrimitive),
            (74, 'I', BravaisLattice::OrthorhombicBodyCentered),
            (74, 'F', BravaisLattice::OrthorhombicFaceCentered),
            (75, 'P', BravaisLattice::TetragonalPrimitive),
            (142, 'I', BravaisLattice::TetragonalBodyCentered),
            (143, 'R', BravaisLattice::TrigonalPrimitive),
            (168, 'P', BravaisLattice::HexagonalPrimitive),
            (195, 'P', BravaisLattice::CubicPrimitive),
            (225, 'F', BravaisLattice::CubicFaceCentered),
            (229, 'I', BravaisLattice::CubicBodyCentered),
        ];

        for (space_group, lattice, expected) in cases {
            let bravais = bravais_lattice(space_group, lattice)?;
            assert_eq!(bravais, expected);
            assert_eq!(
                bravais_lattice_index(space_group, lattice)?,
                expected.index()
            );
        }
        Ok(())
    }

    #[test]
    fn kpath_segments_match_feff_reference() -> Result<(), KSpaceError> {
        let orthorhombic = define_k_path(BravaisLattice::OrthorhombicPrimitive, 7, BASIS)?;
        assert_eq!(orthorhombic.effective_kpath, 7);
        assert_eq!(orthorhombic.labels, ["X -GS-GG", "GG-GD-Y "]);
        assert_vector_close(orthorhombic.start(0), [0.55, 0.1, 0.025])?;
        assert_vector_close(orthorhombic.end(0), [0.0, 0.0, 0.0])?;
        assert_vector_close(orthorhombic.end(1), [-0.05, 0.65, 0.02])?;

        let hexagonal = define_k_path(BravaisLattice::HexagonalPrimitive, 5, BASIS)?;
        assert_eq!(hexagonal.labels, ["K -T -GG"]);
        assert_vector_close(hexagonal.start(0), [-0.433_333_333_333_333_35, 0.8, 0.01])?;
        assert_vector_close(hexagonal.end(0), [0.0, 0.0, 0.0])?;

        let cubic = define_k_path(BravaisLattice::CubicPrimitive, 5, BASIS)?;
        assert_eq!(cubic.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
        assert_vector_close(cubic.end(0), [0.5, 0.0, 0.0])?;
        assert_vector_close(cubic.end(1), [0.0, 0.5, 0.0])?;
        assert_vector_close(cubic.end(2), [0.0, 0.0, 0.5])?;

        let face_default = define_k_path(BravaisLattice::CubicFaceCentered, 0, BASIS)?;
        assert_eq!(face_default.effective_kpath, 4);
        assert_eq!(face_default.labels, ["GG-GD-X "]);
        assert_vector_close(face_default.end(0), [0.565, 0.2, 0.475])?;

        let face_axes = define_k_path(BravaisLattice::CubicFaceCentered, 6, BASIS)?;
        assert_eq!(face_axes.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
        assert_vector_close(face_axes.end(0), [1.0, 0.0, 0.0])?;
        assert_vector_close(face_axes.end(1), [0.0, 1.0, 0.0])?;
        assert_vector_close(face_axes.end(2), [0.0, 0.0, 1.0])?;

        let body_default = define_k_path(BravaisLattice::CubicBodyCentered, 0, BASIS)?;
        assert_eq!(body_default.effective_kpath, 5);
        assert_eq!(body_default.labels, ["GG-GD-H "]);
        assert_vector_close(body_default.end(0), [0.485, 0.65, -0.405])?;

        let body_axes = define_k_path(BravaisLattice::CubicBodyCentered, 6, BASIS)?;
        assert_eq!(body_axes.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
        assert_vector_close(body_axes.end(0), [1.0, 0.0, 0.0])?;
        assert_vector_close(body_axes.end(1), [0.0, 1.0, 0.0])?;
        assert_vector_close(body_axes.end(2), [0.0, 0.0, 1.0])?;
        Ok(())
    }

    #[test]
    fn reciprocal_coordinate_helpers_match_feff_reference() -> Result<(), KSpaceError> {
        let direct = arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
        let reciprocal = arr2(&[
            [PI2 / 2.0, 0.0, 0.0],
            [0.0, PI2 / 3.0, 0.0],
            [0.0, 0.0, PI2 / 4.0],
        ]);
        let vector = [3.2, -1.55, 8.2];

        let subtracted = subtract_lattice_translation(reciprocal.view(), vector)?;
        assert_eq!(subtracted.translation_count, 5);
        assert_close(subtracted.vector[0], -0.4);
        assert_close(subtracted.vector[1], 0.483_333_333_333_333_4);
        assert_close(subtracted.vector[2], 0.05);

        let reduced = reduce_to_lattice_cell(direct.view(), reciprocal.view(), vector)?;
        assert_eq!(reduced.translation_count, 5);
        assert_close(reduced.vector[0], 1.2);
        assert_close(reduced.vector[1], 1.45);
        assert_close(reduced.vector[2], 0.2);

        let operation = array![[1, -2, 0], [3, 0, 1], [-1, 2, 1]];
        let changed = change_cartesian_basis(reciprocal.view(), direct.view(), operation.view())?;
        assert_close(changed[(0, 0)], PI2);
        assert_close(changed[(0, 1)], -3.0 * PI2);
        assert_close(changed[(0, 2)], 0.0);
        assert_close(changed[(1, 0)], 2.0 * PI2);
        assert_close(changed[(1, 1)], 0.0);
        assert_close(changed[(1, 2)], 4.0 * PI2 / 3.0);
        assert_close(changed[(2, 0)], -std::f64::consts::PI);
        assert_close(changed[(2, 1)], 3.0 * std::f64::consts::PI);
        assert_close(changed[(2, 2)], PI2);
        Ok(())
    }

    #[test]
    fn reciprocal_lattice_vectors_match_feff_gbass_reference() -> Result<(), KSpaceError> {
        let direct = arr2(&[[2.0, 0.3, -0.2], [0.1, 3.0, 0.5], [0.2, 0.4, 4.0]]);
        let reciprocal = reciprocal_lattice_vectors(direct.view())?;
        let expected = arr2(&[
            [
                3.138_666_777_779_998_4,
                -7.979_661_299_440_674e-2,
                -1.489_536_775_895_592_7e-1,
            ],
            [
                -3.404_655_487_761_354_4e-1,
                2.138_549_228_250_1,
                -1.968_316_453_862_033e-1,
            ],
            [
                1.994_915_324_860_168_6e-1,
                -2.713_084_841_809_829e-1,
                1.587_952_598_588_694,
            ],
        ]);
        assert_matrix_close(reciprocal.view(), expected.view());

        let roundtrip = reciprocal_lattice_vectors(reciprocal.view())?;
        assert_matrix_close(roundtrip.view(), direct.view());
        Ok(())
    }

    #[test]
    fn point_group_operations_match_feff_reference() -> Result<(), KSpaceError> {
        let cubic = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let cubic_metric = reciprocal_metric(cubic.view())?;
        let cubic_group = point_group_operations(cubic.view(), cubic_metric.view(), 64)?;
        assert_eq!(cubic_group.len(), 48);
        assert_operation_close(
            cubic_group.operation(0),
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        )?;
        assert_operation_close(
            cubic_group.operation(8),
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        )?;
        assert_operation_close(
            cubic_group.operation(47),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )?;

        let orthorhombic = arr2(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        let orthorhombic_metric = reciprocal_metric(orthorhombic.view())?;
        let orthorhombic_group =
            point_group_operations(orthorhombic.view(), orthorhombic_metric.view(), 64)?;
        assert_eq!(orthorhombic_group.len(), 8);
        assert_operation_close(
            orthorhombic_group.operation(0),
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        )?;
        assert_operation_close(
            orthorhombic_group.operation(7),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )?;
        Ok(())
    }

    #[test]
    fn symmetry_check_matches_feff_reference() -> Result<(), KSpaceError> {
        let operations = sign_flip_symmetry_operations();
        let translations = Array2::<Real>::zeros((4, 3));
        let checked = symmetry_check(operations.view(), translations.view())?;
        assert_eq!(checked.ierr, 0);
        assert_eq!(checked.invalid_operation_index(), None);
        assert_eq!(
            checked.multiplication,
            arr2(&[[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]])
        );

        let mut bad_translations = translations;
        bad_translations[(1, 0)] = PI2 / 2.0;
        let checked = symmetry_check(operations.view(), bad_translations.view())?;
        assert_eq!(checked.ierr, 2);
        assert_eq!(checked.invalid_operation_index(), Some(1));
        assert_eq!(
            checked.multiplication,
            arr2(&[[1, 2, 3, 4], [2, 1, -1, -1], [3, -1, 1, -1], [4, -1, -1, 1]])
        );
        Ok(())
    }

    #[test]
    fn kspace_helpers_reject_invalid_inputs() {
        assert_eq!(
            bravais_lattice(0, 'P'),
            Err(KSpaceError::InvalidSpaceGroup { space_group: 0 })
        );
        assert_eq!(
            bravais_lattice(225, 'C'),
            Err(KSpaceError::InvalidBravaisResult {
                space_group: 225,
                lattice: 'C',
            })
        );
        assert_eq!(
            define_k_path(BravaisLattice::TetragonalPrimitive, 1, BASIS),
            Err(KSpaceError::UnsupportedBravais { bravais: 8 })
        );
        assert_eq!(
            define_k_path(BravaisLattice::CubicPrimitive, 99, BASIS),
            Err(KSpaceError::InvalidKPath {
                bravais: 12,
                kpath: 99,
            })
        );

        let bad_matrix = Array2::<Real>::zeros((2, 3));
        assert_eq!(
            subtract_lattice_translation(bad_matrix.view(), [0.0; 3]),
            Err(KSpaceError::InvalidMatrixShape {
                name: "reciprocal_vectors",
                rows: 2,
                columns: 3,
            })
        );
        let matrix = Array2::<Real>::zeros((3, 3));
        assert_eq!(
            reciprocal_lattice_vectors(matrix.view()),
            Err(KSpaceError::DegenerateLatticeVolume { determinant: 0.0 })
        );
        assert!(matches!(
            reduce_to_lattice_cell(matrix.view(), matrix.view(), [Real::NAN, 0.0, 0.0]),
            Err(KSpaceError::NonFiniteValue {
                name: "vector",
                index: 0,
                value,
            }) if value.is_nan()
        ));
        assert_eq!(
            point_group_operations(matrix.view(), matrix.view(), 0),
            Err(KSpaceError::InvalidPointGroupCapacity { capacity: 0 })
        );
        let identity = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(
            point_group_operations(identity.view(), matrix.view(), 4),
            Err(KSpaceError::DegenerateMetricDiagonal {
                index: 0,
                value: 0.0,
            })
        );

        let no_operations = Array3::<i32>::zeros((0, 3, 3));
        let no_translations = Array2::<Real>::zeros((0, 3));
        assert_eq!(
            symmetry_check(no_operations.view(), no_translations.view()),
            Err(KSpaceError::NoSymmetryOperations)
        );
        let bad_operations = Array3::<i32>::zeros((2, 2, 3));
        let translations = Array2::<Real>::zeros((2, 3));
        assert_eq!(
            symmetry_check(bad_operations.view(), translations.view()),
            Err(KSpaceError::InvalidSymmetryOperationShape {
                operations: 2,
                rows: 2,
                columns: 3,
            })
        );
        let operations = sign_flip_symmetry_operations();
        let bad_translations = Array2::<Real>::zeros((3, 3));
        assert_eq!(
            symmetry_check(operations.view(), bad_translations.view()),
            Err(KSpaceError::InvalidSymmetryTranslationShape {
                operations: 4,
                rows: 3,
                columns: 3,
            })
        );
        let rotating_operation = array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ];
        let translations = Array2::<Real>::zeros((2, 3));
        assert_eq!(
            symmetry_check(rotating_operation.view(), translations.view()),
            Err(KSpaceError::SymmetryProductMissing { left: 2, right: 2 })
        );
    }

    fn sign_flip_symmetry_operations() -> Array3<i32> {
        array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        ]
    }

    fn assert_vector_close(actual: Option<Vector3>, expected: Vector3) -> Result<(), KSpaceError> {
        let Some(actual) = actual else {
            return Err(KSpaceError::KPathDefinitionIncomplete {
                bravais: 0,
                kpath: 0,
                available: 0,
                required: 1,
            });
        };
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
        Ok(())
    }

    fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
        assert_eq!(actual.shape(), expected.shape());
        for ((row, column), &actual) in actual.indexed_iter() {
            assert_close(actual, expected[(row, column)]);
        }
    }

    fn assert_operation_close(
        actual: Option<[[Real; 3]; 3]>,
        expected: [[Real; 3]; 3],
    ) -> Result<(), KSpaceError> {
        let Some(actual) = actual else {
            return Err(KSpaceError::NoPointGroupOperations);
        };
        for row in 0..3 {
            for col in 0..3 {
                assert_close(actual[row][col], expected[row][col]);
            }
        }
        Ok(())
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }
}
