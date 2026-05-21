//! Public reciprocal-space data types.

use ndarray::{Array1, Array2, Array3};
use refeff_linalg::LinalgError;
use thiserror::Error;

use crate::{Real, RealMat, Vector3};

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

/// FEFF `basdiv` k-mesh division result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KMeshDivisions {
    /// Number of intervals for each reciprocal-lattice vector.
    pub divisions: [usize; 3],
    /// FEFF-adjusted `(n1 + 1) * (n2 + 1) * (n3 + 1)` mesh-point count.
    pub mesh_points: usize,
}

/// FEFF `ARBMSH` generated k-mesh data.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshArbitraryMesh {
    /// Requested FEFF k-point count, `nka`.
    pub requested_point_count: usize,
    /// Number of intervals for each reciprocal-lattice vector, FEFF `n`.
    pub divisions: [usize; 3],
    /// FEFF work-mesh point count, `nkw = (n1 + 1) * (n2 + 1) * (n3 + 1)`.
    pub work_point_count: usize,
    /// FEFF full Brillouin-zone point count, `nkf = n1 * n2 * n3`.
    pub full_point_count: usize,
    /// FEFF irreducible Brillouin-zone point count, `nki`.
    pub irreducible_point_count: usize,
    /// FEFF unnormalized reduction weight sum, `sumwgt`.
    pub total_weight: Real,
    /// FEFF work, full, and irreducible k-point arrays from `REDUZ`.
    pub reduction: KMeshReduction,
    /// Optional FEFF `TETCNT` records when tetrahedra are requested.
    pub tetrahedra: Option<KMeshTetrahedronRecords>,
}

/// FEFF `divisi` common-factor reduction result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KMeshDivisionReduction {
    /// Reduced integer k-point list as `(point, xyz)`.
    pub k_list: Array2<i32>,
    /// FEFF-adjusted divisor value after integer division and lower-bound clamp.
    pub division: usize,
    /// Product of prime factors removed from every k-point component.
    pub common_divisor: usize,
}

/// FEFF `TETCNT` tetrahedron record table.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshTetrahedronRecords {
    /// Number of irreducible k-points FEFF writes as `nki`.
    pub irreducible_point_count: usize,
    /// Total tetrahedra before equivalent records are merged.
    pub tetrahedron_count: usize,
    /// Number of unique tetrahedron records after FEFF ordering and merging.
    pub unique_tetrahedron_count: usize,
    /// Per-tetrahedron integration weight `1 / (6 * n1 * n2 * n3)`.
    pub tetrahedron_weight: Real,
    /// FEFF write chunk size from `m_tetrahedra.f90`.
    pub write_chunk_size: usize,
    /// Number of FEFF record chunks required for `records`.
    pub record_count: usize,
    /// Rows are FEFF `ITTFL` records: multiplicity and four 1-based corners.
    pub records: Array2<usize>,
}

/// FEFF `REDUZ` k-mesh reduction data.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshReduction {
    /// FEFF half-mesh shift selector `ishift`.
    pub shift: [usize; 3],
    /// Sum of unnormalized work-point boundary weights before normalization.
    pub total_weight: Real,
    /// Integer work-mesh coordinates as `(point, xyz)`.
    pub work_grid: Array2<usize>,
    /// Work-mesh Cartesian k-vectors as `(point, xyz)`, FEFF `bkw`.
    pub work_vectors: RealMat,
    /// Full-mesh Cartesian k-vectors as `(point, xyz)`, FEFF `bkf`.
    pub full_vectors: RealMat,
    /// Irreducible Cartesian k-vectors as `(point, xyz)`, FEFF `bki`.
    pub irreducible_vectors: RealMat,
    /// Irreducible fractional k-vectors as `(point, xyz)`, FEFF `bki2`.
    pub irreducible_fractional_vectors: RealMat,
    /// Normalized work-point weights, FEFF `ww`.
    pub work_weights: Array1<Real>,
    /// Normalized full-mesh weights, FEFF `wf`.
    pub full_weights: Array1<Real>,
    /// Normalized irreducible weights, FEFF `wi`.
    pub irreducible_weights: Array1<Real>,
    /// Final 1-based work-to-irreducible links, FEFF `linkw`.
    pub work_links: Vec<usize>,
    /// 1-based full-to-irreducible links, FEFF `linkf`.
    pub full_links: Vec<usize>,
    /// 1-based symmetry operation used for each work point, FEFF `lsymw`.
    pub work_symmetry: Vec<usize>,
    /// 1-based symmetry operation used for each full point, FEFF `lsymf`.
    pub full_symmetry: Vec<usize>,
}

/// FEFF `bravais` lattice-basis construction result.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshBravaisBasis {
    /// FEFF-adjusted lengths after centering-specific scaling.
    pub adjusted_lengths: Vector3,
    /// Direct lattice vectors as `(vector, xyz)`, matching FEFF `rbas(i,*)`.
    pub direct_vectors: RealMat,
    /// Reciprocal lattice matrix exactly as returned by FEFF `bravais`.
    ///
    /// FEFF calls `GBASS` immediately afterward when row-vector reciprocal
    /// storage is needed for the mesh generator.
    pub reciprocal_vectors: RealMat,
    /// FEFF `afact` centering factor.
    pub afact: Real,
    /// FEFF `iarb` dependency flags for equal k-mesh divisions.
    pub dependencies: [bool; 3],
    /// FEFF `ortho` flag.
    pub orthogonal: bool,
    /// FEFF Brillouin-zone volume `v` from `KSPACE/kmesh.f90` `bravais`.
    pub brillouin_zone_volume: Real,
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
    /// FEFF `basdiv` requires a positive requested mesh-point count.
    #[error("requested k-mesh point count must be positive, got {mesh_points}")]
    InvalidKMeshPointTarget { mesh_points: usize },
    /// FEFF `basdiv` scales by reciprocal-vector lengths.
    #[error("reciprocal vector {index} has degenerate length {length}")]
    DegenerateReciprocalVector { index: usize, length: Real },
    /// FEFF `basdiv` could not compute a finite scale from the input basis.
    #[error(
        "k-mesh scale is invalid for requested point count {mesh_points} and reciprocal-vector length product {length_product}"
    )]
    InvalidKMeshScale {
        mesh_points: usize,
        length_product: Real,
    },
    /// FEFF-compatible k-mesh divisions must fit addressable Rust memory.
    #[error("k-mesh division component {component} is too large: {value}")]
    KMeshDivisionOverflow { component: usize, value: Real },
    /// FEFF `basdiv` mesh-point count overflowed Rust `usize`.
    #[error("k-mesh point count overflowed")]
    KMeshPointCountOverflow,
    /// FEFF `tetdiv` divides by each k-mesh division component.
    #[error("k-mesh division component {component} must be positive, got {value}")]
    InvalidKMeshDivision { component: usize, value: usize },
    /// FEFF `tetcnt` expects six tetrahedra with four 3-D corner offsets.
    #[error(
        "tetrahedron offsets must have shape (6, 4, 3), got ({tetrahedra}, {corners}, {coordinates})"
    )]
    InvalidTetrahedronOffsetShape {
        tetrahedra: usize,
        corners: usize,
        coordinates: usize,
    },
    /// FEFF `tetcnt` consumes zero/one offsets generated by `tetdiv`.
    #[error("tetrahedron offset ({tetrahedron}, {corner}, {axis}) must be 0 or 1, got {value}")]
    InvalidTetrahedronOffset {
        tetrahedron: usize,
        corner: usize,
        axis: usize,
        value: i32,
    },
    /// FEFF `tetcnt` needs a positive irreducible k-point count.
    #[error("irreducible k-point count must be positive, got {count}")]
    InvalidIrreducibleKPointCount { count: usize },
    /// FEFF `tetcnt` needs one link for every work mesh point.
    #[error("work mesh link count must be {expected}, got {actual}")]
    InvalidWorkMeshLinkCount { expected: usize, actual: usize },
    /// FEFF `tetcnt` uses 1-based irreducible k-point links.
    #[error("work mesh link {index} must be in 1..={irreducible_point_count}, got {value}")]
    InvalidWorkMeshLink {
        index: usize,
        value: usize,
        irreducible_point_count: usize,
    },
    /// FEFF `tetcnt` tetrahedron count overflowed Rust `usize`.
    #[error("k-mesh tetrahedron count overflowed")]
    KMeshTetrahedronCountOverflow,
    /// FEFF `REDUZ` full-mesh links are internal 1-based indices.
    #[error("full mesh link {index} must be in 1..={full_point_count}, got {value}")]
    InvalidFullMeshLink {
        index: usize,
        value: usize,
        full_point_count: usize,
    },
    /// FEFF `REDUZ` checks that each irreducible point is emitted exactly once.
    #[error("k-mesh reduction emitted {actual} irreducible points, expected {expected}")]
    KMeshReductionIrreducibleCountMismatch { expected: usize, actual: usize },
    /// FEFF `REDUZ` checks that full points do not duplicate link/symmetry pairs.
    #[error(
        "full mesh points {first} and {second} both map to link {link} with symmetry {symmetry}"
    )]
    DuplicateFullMeshLinkSymmetry {
        first: usize,
        second: usize,
        link: usize,
        symmetry: usize,
    },
    /// FEFF `divisi` expects a non-empty `(n, 3)` integer k-point list.
    #[error("k-mesh list must have shape (n, 3) with n > 0, got ({rows}, {columns})")]
    InvalidKMeshListShape { rows: usize, columns: usize },
    /// FEFF `divisi` common divisor overflowed Rust `usize`.
    #[error("k-mesh common divisor overflowed")]
    KMeshCommonDivisorOverflow,
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
    /// FEFF-compatible symmetry entries must round back into signed integers.
    #[error(
        "transformed symmetry operation {operation} entry ({row}, {column}) is not representable as i32: {value}"
    )]
    SymmetryOperationValueOverflow {
        operation: usize,
        row: usize,
        column: usize,
        value: Real,
    },
    /// FEFF matrix inversion failed while preparing point-group operations.
    #[error(transparent)]
    Linalg(#[from] LinalgError),
}
