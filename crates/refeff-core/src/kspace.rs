//! Reciprocal-space lattice helpers ported from FEFF BAND/KSPACE routines.
//!
//! The routines here cover the small deterministic helpers used before the
//! heavier KSPACE solvers: Bravais classification from `BAND/ibravais.f90`,
//! high-symmetry K-path segment generation from `BAND/kpath.f90`, point-group
//! operation discovery and closure checks from `KSPACE/pointgroup.f90` and
//! `KSPACE/symmetrycheck.f90`, k-mesh division helpers from `KSPACE/kmesh.f90`,
//! and the coordinate reductions from `KSPACE/subtract_a.f90` and
//! `change_car.f90`. FEFF exits the process for unsupported lattices; Rust
//! returns typed errors.

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use refeff_linalg::{LinalgError, feff_inverse};
use thiserror::Error;

use crate::{Real, RealMat, Vector3};

const PI2: Real = std::f64::consts::TAU;
const BRAVAIS_PI2: Real = 2.0 * (std::f32::consts::PI as Real);
const BRAVAIS_RIGHT_ANGLE: Real = 1_570_796.0 / 1_000_000.0;
const BRAVAIS_ANGLE_EPSILON: Real = 0.0001;
const POINT_GROUP_EPSILON: Real = 1.0e-8;
const REDUCE_NEGATIVE_EPSILON: Real = -1.0e-8;
const LATTICE_VOLUME_EPSILON: Real = Real::EPSILON;
const BASDIV_OFFSET: Real = 1.0e-6;
/// FEFF `KSPACE/m_tetrahedra.f90` `mwrit` chunk size for tetrahedron records.
pub const KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE: usize = 101;
const DIVISI_PRIMES: [i32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const DIVISI_ITERATIONS: usize = 10;

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

/// Port of FEFF `KSPACE/kmesh.f90` `bravais`.
///
/// `lattice` is FEFF's three-character lattice code. `lengths` are `(a, b, c)`,
/// and `angles` are `(alpha, beta, gamma)` in radians. The returned matrices
/// use FEFF's direct row-vector storage and FEFF `bravais` reciprocal matrix
/// orientation. This routine intentionally keeps the single-precision `pi`
/// constants used by FEFF `bravais`; call [`reciprocal_lattice_vectors`] on
/// `direct_vectors` when the subsequent double-precision `GBASS` row-vector
/// result is needed.
pub fn kmesh_bravais_basis(
    lattice: &str,
    lengths: Vector3,
    angles: Vector3,
) -> Result<KMeshBravaisBasis, KSpaceError> {
    validate_vector("lattice_lengths", lengths)?;
    validate_vector("lattice_angles", angles)?;

    let lattice = lattice_code3(lattice);
    let mut adjusted_lengths = lengths;
    let mut direct_vectors = Array2::<Real>::zeros((3, 3));
    let mut dependencies = [true; 3];
    let mut afact = 1.0;
    let orthogonal;

    if lattice[0] == b'H' {
        direct_vectors[(0, 0)] = adjusted_lengths[0] * Real::from(0.75_f32.sqrt());
        direct_vectors[(0, 1)] = -adjusted_lengths[0] / 2.0;
        direct_vectors[(1, 1)] = adjusted_lengths[0];
        direct_vectors[(2, 2)] = adjusted_lengths[2];
        dependencies[1] = false;
        dependencies[2] = false;
        orthogonal = false;
    } else if lattice[0] == b'F' {
        adjusted_lengths = adjusted_lengths.map(|length| length * 0.5);
        direct_vectors[(0, 1)] = adjusted_lengths[1];
        direct_vectors[(0, 2)] = adjusted_lengths[2];
        direct_vectors[(1, 0)] = adjusted_lengths[0];
        direct_vectors[(1, 2)] = adjusted_lengths[2];
        direct_vectors[(2, 0)] = adjusted_lengths[0];
        direct_vectors[(2, 1)] = adjusted_lengths[1];
        afact = 0.5;
        orthogonal = true;
    } else if lattice[0] == b'B' {
        adjusted_lengths = adjusted_lengths.map(|length| length * 0.5);
        direct_vectors[(0, 0)] = -adjusted_lengths[0];
        direct_vectors[(0, 1)] = adjusted_lengths[1];
        direct_vectors[(0, 2)] = adjusted_lengths[2];
        direct_vectors[(1, 0)] = adjusted_lengths[0];
        direct_vectors[(1, 1)] = -adjusted_lengths[1];
        direct_vectors[(1, 2)] = adjusted_lengths[2];
        direct_vectors[(2, 0)] = adjusted_lengths[0];
        direct_vectors[(2, 1)] = adjusted_lengths[1];
        direct_vectors[(2, 2)] = -adjusted_lengths[2];
        afact = 0.5;
        orthogonal = true;
    } else if lattice[0] == b'P' && has_non_right_angle(angles) {
        let cos_gamma_1 = (angles[2].cos() - angles[0].cos() * angles[1].cos())
            / angles[0].sin()
            / angles[1].sin();
        let gamma_0 = cos_gamma_1.acos();
        direct_vectors[(0, 0)] = adjusted_lengths[0] * gamma_0.sin() * angles[1].sin();
        direct_vectors[(0, 1)] = adjusted_lengths[0] * gamma_0.cos() * angles[1].sin();
        direct_vectors[(1, 1)] = adjusted_lengths[1] * angles[0].sin();
        direct_vectors[(0, 2)] = adjusted_lengths[0] * angles[1].cos();
        direct_vectors[(1, 2)] = adjusted_lengths[1] * angles[0].cos();
        direct_vectors[(2, 2)] = adjusted_lengths[2];
        dependencies = [false; 3];
        orthogonal = false;
    } else if (lattice[0] == b'C' && !is_feff_right_angle(angles[2]))
        || lattice == [b'M', b'X', b'Z']
    {
        let ay = adjusted_lengths[0] * angles[2].cos() / 2.0;
        adjusted_lengths[0] = adjusted_lengths[0] * angles[2].sin() / 2.0;
        adjusted_lengths[2] /= 2.0;
        direct_vectors[(0, 0)] = adjusted_lengths[0];
        direct_vectors[(0, 1)] = ay;
        direct_vectors[(0, 2)] = -adjusted_lengths[2];
        direct_vectors[(1, 1)] = adjusted_lengths[1];
        direct_vectors[(2, 0)] = adjusted_lengths[0];
        direct_vectors[(2, 1)] = ay;
        direct_vectors[(2, 2)] = adjusted_lengths[2];
        dependencies[0] = false;
        dependencies[2] = false;
        orthogonal = false;
    } else if lattice[0] == b'S' || lattice[0] == b'P' {
        direct_vectors[(0, 0)] = adjusted_lengths[0];
        direct_vectors[(1, 1)] = adjusted_lengths[1];
        direct_vectors[(2, 2)] = adjusted_lengths[2];
        dependencies = [false; 3];
        orthogonal = true;
    } else if lattice[0] == b'C' {
        if lattice[1] == b'X' && lattice[2] == b'Z' {
            direct_vectors[(0, 0)] = adjusted_lengths[0] * 0.5;
            direct_vectors[(0, 2)] = -adjusted_lengths[2] * 0.5;
            direct_vectors[(2, 0)] = adjusted_lengths[0] * 0.5;
            direct_vectors[(2, 2)] = adjusted_lengths[2] * 0.5;
            direct_vectors[(1, 1)] = adjusted_lengths[1];
            dependencies[0] = false;
            dependencies[2] = false;
        } else if lattice[1] == b'Y' && lattice[2] == b'Z' {
            direct_vectors[(1, 1)] = adjusted_lengths[1] * 0.5;
            direct_vectors[(1, 2)] = -adjusted_lengths[2] * 0.5;
            direct_vectors[(2, 1)] = adjusted_lengths[1] * 0.5;
            direct_vectors[(2, 2)] = adjusted_lengths[2] * 0.5;
            direct_vectors[(0, 0)] = adjusted_lengths[0];
            dependencies[0] = false;
            dependencies[1] = false;
        } else {
            direct_vectors[(0, 0)] = adjusted_lengths[0] * 0.5;
            direct_vectors[(0, 1)] = -adjusted_lengths[1] * 0.5;
            direct_vectors[(1, 0)] = adjusted_lengths[0] * 0.5;
            direct_vectors[(1, 1)] = adjusted_lengths[1] * 0.5;
            direct_vectors[(2, 2)] = adjusted_lengths[2];
            dependencies[1] = false;
            dependencies[2] = false;
        }
        orthogonal = true;
    } else if lattice == [b'M', b' ', b' '] {
        direct_vectors[(0, 0)] = adjusted_lengths[0] * angles[2].sin();
        direct_vectors[(0, 1)] = adjusted_lengths[0] * angles[2].cos();
        direct_vectors[(1, 1)] = adjusted_lengths[1];
        direct_vectors[(2, 2)] = adjusted_lengths[2];
        dependencies = [false; 3];
        orthogonal = false;
    } else if lattice[0] == b'R' {
        direct_vectors[(0, 0)] = adjusted_lengths[0] / 2.0 / 3.0_f64.sqrt();
        direct_vectors[(0, 1)] = -adjusted_lengths[0] / 2.0;
        direct_vectors[(0, 2)] = adjusted_lengths[2] / 3.0;
        direct_vectors[(1, 0)] = adjusted_lengths[0] / 2.0 / 3.0_f64.sqrt();
        direct_vectors[(1, 1)] = adjusted_lengths[0] * 0.5;
        direct_vectors[(1, 2)] = adjusted_lengths[2] / 3.0;
        direct_vectors[(2, 0)] = -adjusted_lengths[0] / 3.0_f64.sqrt();
        direct_vectors[(2, 2)] = adjusted_lengths[2] / 3.0;
        orthogonal = false;
    } else {
        adjusted_lengths[0] *= 0.5;
        direct_vectors[(0, 0)] = -adjusted_lengths[0];
        direct_vectors[(1, 0)] = adjusted_lengths[0];
        direct_vectors[(2, 0)] = adjusted_lengths[0];
        direct_vectors[(0, 1)] = adjusted_lengths[0];
        direct_vectors[(1, 1)] = -adjusted_lengths[0];
        direct_vectors[(2, 1)] = adjusted_lengths[0];
        direct_vectors[(0, 2)] = adjusted_lengths[0];
        direct_vectors[(1, 2)] = adjusted_lengths[0];
        direct_vectors[(2, 2)] = -adjusted_lengths[0];
        afact = 0.5;
        orthogonal = true;
    }

    let (gbass_reciprocal_vectors, determinant) =
        reciprocal_lattice_vectors_with_scale(direct_vectors.view(), BRAVAIS_PI2)?;
    let reciprocal_vectors = gbass_reciprocal_vectors.t().to_owned();
    Ok(KMeshBravaisBasis {
        adjusted_lengths,
        direct_vectors,
        reciprocal_vectors,
        afact,
        dependencies,
        orthogonal,
        brillouin_zone_volume: BRAVAIS_PI2.powi(3) / determinant,
    })
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
/// The input basis is stored as row vectors, matching the way `GBASS` is used
/// by FEFF `basdiv`. The result is the reciprocal basis `2*pi *
/// inverse(basis)^T`, also stored as row vectors.
/// Applying this routine twice returns the original basis within floating-point
/// roundoff, matching FEFF's "real space or vice versa" behavior.
pub fn reciprocal_lattice_vectors(
    lattice_vectors: ArrayView2<'_, Real>,
) -> Result<RealMat, KSpaceError> {
    reciprocal_lattice_vectors_with_scale(lattice_vectors, PI2).map(|(reciprocal, _)| reciprocal)
}

fn reciprocal_lattice_vectors_with_scale(
    lattice_vectors: ArrayView2<'_, Real>,
    pi2: Real,
) -> Result<(RealMat, Real), KSpaceError> {
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

    let scale = pi2 / determinant;
    reciprocal.mapv_inplace(|value| value * scale);
    for ((row, column), &value) in reciprocal.indexed_iter() {
        validate_vector_component("reciprocal_lattice_vectors", row * 3 + column, value)?;
    }
    Ok((reciprocal, determinant))
}

/// Port of FEFF `KSPACE/kmesh.f90` `basdiv`.
///
/// `reciprocal_vectors` stores reciprocal-lattice vectors as rows, matching
/// FEFF `gbas(i,*)`. `dependencies` is FEFF `iarb`: `[0]` couples divisions
/// 1 and 2, `[1]` couples 1 and 3, and `[2]` couples 2 and 3. The returned
/// point count is FEFF's adjusted `(n1 + 1) * (n2 + 1) * (n3 + 1)` value.
pub fn kmesh_basis_divisions(
    reciprocal_vectors: ArrayView2<'_, Real>,
    requested_mesh_points: usize,
    dependencies: [bool; 3],
) -> Result<KMeshDivisions, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    if requested_mesh_points == 0 {
        return Err(KSpaceError::InvalidKMeshPointTarget {
            mesh_points: requested_mesh_points,
        });
    }

    let lengths = [
        row_norm(reciprocal_vectors, 0),
        row_norm(reciprocal_vectors, 1),
        row_norm(reciprocal_vectors, 2),
    ];
    for (index, length) in lengths.into_iter().enumerate() {
        if length <= LATTICE_VOLUME_EPSILON {
            return Err(KSpaceError::DegenerateReciprocalVector { index, length });
        }
    }

    let length_product = lengths[0] * lengths[1] * lengths[2];
    if !length_product.is_finite() || length_product <= LATTICE_VOLUME_EPSILON {
        return Err(KSpaceError::InvalidKMeshScale {
            mesh_points: requested_mesh_points,
            length_product,
        });
    }

    let scale = ((requested_mesh_points as Real) / length_product).powf(1.0 / 3.0);
    if !scale.is_finite() {
        return Err(KSpaceError::InvalidKMeshScale {
            mesh_points: requested_mesh_points,
            length_product,
        });
    }
    let rn = [lengths[0] * scale, lengths[1] * scale, lengths[2] * scale];

    let divisions = if dependencies[0] && dependencies[1] {
        let value = (rn[0] * rn[1] * rn[2]).powf(1.0 / 3.0) + BASDIV_OFFSET;
        let division = mesh_division_from_real(0, value)?;
        [division; 3]
    } else if dependencies[0] {
        let division = mesh_division_from_real(0, (rn[0] * rn[1]).sqrt() + BASDIV_OFFSET)?;
        [division, division, mesh_division_from_real(2, rn[2])?]
    } else if dependencies[1] {
        let division = mesh_division_from_real(0, (rn[0] * rn[2]).sqrt() + BASDIV_OFFSET)?;
        [division, mesh_division_from_real(1, rn[1])?, division]
    } else if dependencies[2] {
        let division = mesh_division_from_real(1, (rn[1] * rn[2]).sqrt() + BASDIV_OFFSET)?;
        [mesh_division_from_real(0, rn[0])?, division, division]
    } else {
        [
            mesh_division_from_real(0, rn[0] + BASDIV_OFFSET)?,
            mesh_division_from_real(1, rn[1] + BASDIV_OFFSET)?,
            mesh_division_from_real(2, rn[2] + BASDIV_OFFSET)?,
        ]
    };

    Ok(KMeshDivisions {
        divisions,
        mesh_points: kmesh_point_count(divisions)?,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `TETDIV`.
///
/// Splits one reciprocal-lattice parallelepiped cell into six tetrahedra using
/// FEFF's shortest body-diagonal rule. `divisions` is FEFF `n`, and
/// `reciprocal_vectors` stores `gbas(i,*)` row vectors. The returned array has
/// shape `(tetrahedron, corner, xyz)` and stores the zero/one corner offsets
/// that FEFF returns as `TET0(:, corner, tetrahedron)`.
pub fn kmesh_tetrahedron_division(
    divisions: [usize; 3],
    reciprocal_vectors: ArrayView2<'_, Real>,
) -> Result<Array3<i32>, KSpaceError> {
    validate_kmesh_divisions(divisions)?;
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;

    let mut points = [[0.0; 3]; 8];
    for first in 0..=1 {
        for second in 0..=1 {
            for third in 0..=1 {
                let index = 4 * first + 2 * second + third;
                for axis in 0..3 {
                    points[index][axis] = reciprocal_vectors[(0, axis)] * (first as Real)
                        / (divisions[0] as Real)
                        + reciprocal_vectors[(1, axis)] * (second as Real) / (divisions[1] as Real)
                        + reciprocal_vectors[(2, axis)] * (third as Real) / (divisions[2] as Real);
                }
            }
        }
    }

    let mut diagonal_lengths = [0.0; 4];
    for diagonal in 0..4 {
        diagonal_lengths[diagonal] = (0..3)
            .map(|axis| {
                let delta = points[diagonal][axis] - points[7 - diagonal][axis];
                delta * delta
            })
            .sum();
    }
    let shortest = (1..4).fold(0, |best, diagonal| {
        if diagonal_lengths[diagonal] < diagonal_lengths[best] {
            diagonal
        } else {
            best
        }
    });

    let vertex_order = tetrahedron_vertex_order(shortest);
    let tetrahedra = tetrahedron_vertices(vertex_order);
    Ok(Array3::from_shape_fn(
        (6, 4, 3),
        |(tetrahedron, corner, axis)| vertex_coordinate(tetrahedra[tetrahedron][corner], axis),
    ))
}

/// Port of FEFF `KSPACE/kmesh.f90` `TETCNT`.
///
/// `tetrahedron_offsets` is the `(6, 4, 3)` table returned by
/// [`kmesh_tetrahedron_division`]. `point_links` is FEFF `linkw` after
/// irreducible-zone reduction: one 1-based irreducible k-point id for each
/// work-mesh point in `(n1 + 1) * (n2 + 1) * (n3 + 1)` order. The returned
/// `records` table mirrors FEFF `ITTFL`, with each row storing multiplicity
/// followed by four sorted 1-based tetrahedron corners.
pub fn kmesh_tetrahedron_records(
    tetrahedron_offsets: ArrayView3<'_, i32>,
    divisions: [usize; 3],
    point_links: &[usize],
    irreducible_point_count: usize,
) -> Result<KMeshTetrahedronRecords, KSpaceError> {
    validate_kmesh_divisions(divisions)?;
    validate_tetrahedron_offsets(tetrahedron_offsets)?;
    validate_work_mesh_links(
        point_links,
        kmesh_point_count(divisions)?,
        irreducible_point_count,
    )?;

    let cell_count = kmesh_cell_count(divisions)?;
    let tetrahedron_count = cell_count
        .checked_mul(6)
        .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)?;
    let row_stride = divisions[2]
        .checked_add(1)
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let plane_stride = divisions[1]
        .checked_add(1)
        .and_then(|value| value.checked_mul(row_stride))
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let corner_offsets =
        tetrahedron_work_mesh_offsets(tetrahedron_offsets, row_stride, plane_stride)?;

    let mut tetrahedra = Vec::with_capacity(tetrahedron_count);
    for first in 0..divisions[0] {
        for second in 0..divisions[1] {
            for third in 0..divisions[2] {
                let base = first
                    .checked_mul(plane_stride)
                    .and_then(|value| {
                        second
                            .checked_mul(row_stride)
                            .and_then(|offset| value.checked_add(offset))
                    })
                    .and_then(|value| value.checked_add(third))
                    .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                for tetrahedron_offsets in &corner_offsets {
                    let mut corners = [0usize; 4];
                    for (corner, &corner_offset) in tetrahedron_offsets.iter().enumerate() {
                        let point_index = base
                            .checked_add(corner_offset)
                            .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                        corners[corner] = *point_links.get(point_index).ok_or(
                            KSpaceError::InvalidWorkMeshLinkCount {
                                expected: point_index + 1,
                                actual: point_links.len(),
                            },
                        )?;
                    }
                    corners.sort_unstable();
                    tetrahedra.push(corners);
                }
            }
        }
    }
    tetrahedra.sort_unstable();

    let mut rows = Vec::new();
    let mut index = 0;
    while index < tetrahedra.len() {
        let current = tetrahedra[index];
        let mut multiplicity = 1usize;
        index += 1;
        while index < tetrahedra.len() && tetrahedra[index] == current {
            multiplicity = multiplicity
                .checked_add(1)
                .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)?;
            index += 1;
        }
        rows.extend([multiplicity, current[0], current[1], current[2], current[3]]);
    }

    let unique_tetrahedron_count = rows.len() / 5;
    let record_count = unique_tetrahedron_count.div_ceil(KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE);
    let records = Array2::from_shape_vec((unique_tetrahedron_count, 5), rows)
        .map_err(|_| KSpaceError::KMeshTetrahedronCountOverflow)?;

    Ok(KMeshTetrahedronRecords {
        irreducible_point_count,
        tetrahedron_count,
        unique_tetrahedron_count,
        tetrahedron_weight: 1.0 / (tetrahedron_count as Real),
        write_chunk_size: KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE,
        record_count,
        records,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `ARBMSH`.
///
/// FEFF `ARBMSH` composes `basdiv`, `REDUZ`, `TETDIV`, and `TETCNT`: it chooses
/// reciprocal-lattice divisions for the requested mesh size, reduces the work
/// mesh to irreducible k-points, and optionally creates FEFF tetrahedron
/// integration records. The returned fields preserve FEFF's `nka`, `nkw`,
/// `nkf`, `nki`, and one-based link semantics.
pub fn kmesh_arbitrary_mesh(
    reciprocal_vectors: ArrayView2<'_, Real>,
    operations: ArrayView3<'_, i32>,
    requested_mesh_points: usize,
    dependencies: [bool; 3],
    include_tetrahedra: bool,
) -> Result<KMeshArbitraryMesh, KSpaceError> {
    let division_result =
        kmesh_basis_divisions(reciprocal_vectors, requested_mesh_points, dependencies)?;
    let divisions = division_result.divisions;
    let full_point_count = kmesh_cell_count(divisions)?;
    let reduction = reduce_kmesh_irreducible_points(divisions, operations, reciprocal_vectors)?;
    let irreducible_point_count = reduction.irreducible_weights.len();
    let total_weight = reduction.total_weight;

    let tetrahedra = if include_tetrahedra {
        let tetrahedron_offsets = kmesh_tetrahedron_division(divisions, reciprocal_vectors)?;
        Some(kmesh_tetrahedron_records(
            tetrahedron_offsets.view(),
            divisions,
            &reduction.work_links,
            irreducible_point_count,
        )?)
    } else {
        None
    };

    Ok(KMeshArbitraryMesh {
        requested_point_count: requested_mesh_points,
        divisions,
        work_point_count: division_result.mesh_points,
        full_point_count,
        irreducible_point_count,
        total_weight,
        reduction,
        tetrahedra,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `REDUZ`.
///
/// Builds FEFF's work, full, and irreducible k-mesh arrays from integer
/// symmetry operations in reciprocal-lattice coordinates. Link and symmetry
/// outputs keep FEFF's 1-based numbering so they can be fed directly into
/// [`kmesh_tetrahedron_records`] and FEFF-compatible text/binary handoff code.
pub fn reduce_kmesh_irreducible_points(
    divisions: [usize; 3],
    operations: ArrayView3<'_, i32>,
    reciprocal_vectors: ArrayView2<'_, Real>,
) -> Result<KMeshReduction, KSpaceError> {
    validate_kmesh_divisions(divisions)?;
    validate_symmetry_operation_shape(operations)?;
    if operations.shape()[0] == 0 {
        return Err(KSpaceError::NoSymmetryOperations);
    }
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;

    let work_point_count = kmesh_point_count(divisions)?;
    let full_point_count = kmesh_cell_count(divisions)?;
    let row_stride = divisions[2]
        .checked_add(1)
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let plane_stride = divisions[1]
        .checked_add(1)
        .and_then(|value| value.checked_mul(row_stride))
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let work_grid = Array2::from_shape_fn((work_point_count, 3), |(point, axis)| {
        work_mesh_coordinate(point, row_stride, plane_stride, axis)
    });
    let shift = kmesh_submesh_shift(operations);

    let mut work_links = (1..=work_point_count).collect::<Vec<_>>();
    let mut work_symmetry = vec![0usize; work_point_count];
    let mut full_work_links = vec![0usize; work_point_count];
    let mut full_link_count = 0usize;

    for operation in 0..operations.shape()[0] {
        for point in 0..work_point_count {
            let mapped = mapped_work_mesh_index(
                operations,
                operation,
                [
                    work_grid[(point, 0)],
                    work_grid[(point, 1)],
                    work_grid[(point, 2)],
                ],
                divisions,
                shift,
                row_stride,
                plane_stride,
            )?;
            let mapped_link = mapped + 1;
            if mapped_link < work_links[point]
                || (mapped_link == work_links[point] && work_symmetry[point] == 0)
            {
                work_symmetry[point] = operation + 1;
                work_links[point] = work_links[point].min(mapped_link);
            }

            if operation == 0 {
                if mapped == point {
                    full_link_count = full_link_count
                        .checked_add(1)
                        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                    full_work_links[point] = full_link_count;
                } else if mapped < point {
                    full_work_links[point] = full_work_links[mapped].min(mapped_link);
                } else {
                    full_link_count = full_link_count
                        .checked_add(1)
                        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                    full_work_links[point] = full_link_count;
                }
            }
        }
    }

    let irreducible_point_count = work_links
        .iter()
        .enumerate()
        .filter(|(point, link)| **link == point + 1)
        .count();
    let mut work_weights = Array1::<Real>::zeros(work_point_count);
    let mut full_weights = Array1::<Real>::zeros(full_point_count);
    let mut irreducible_weight_sums = vec![0.0; work_point_count];
    let mut total_weight = 0.0;
    for point in 0..work_point_count {
        let weight = boundary_weight(
            [
                work_grid[(point, 0)],
                work_grid[(point, 1)],
                work_grid[(point, 2)],
            ],
            divisions,
        );
        work_weights[point] = weight;
        let full_link = checked_full_work_link(full_work_links[point], point, full_point_count)?;
        full_weights[full_link - 1] += weight;
        irreducible_weight_sums[work_links[point] - 1] += weight;
        total_weight += weight;
    }

    let mut irreducible_weights = Array1::<Real>::zeros(irreducible_point_count);
    let mut irreducible_index = 0usize;
    for (point, &link) in work_links.iter().enumerate() {
        if link == point + 1 {
            irreducible_weights[irreducible_index] = irreducible_weight_sums[link - 1];
            irreducible_index += 1;
        }
    }

    work_weights.mapv_inplace(|weight| weight / total_weight);
    full_weights.mapv_inplace(|weight| weight / total_weight);
    irreducible_weights.mapv_inplace(|weight| weight / total_weight);

    let mut work_vectors = Array2::<Real>::zeros((work_point_count, 3));
    let mut full_vectors = Array2::<Real>::zeros((full_point_count, 3));
    let mut irreducible_vectors = Array2::<Real>::zeros((irreducible_point_count, 3));
    let mut irreducible_fractional_vectors = Array2::<Real>::zeros((irreducible_point_count, 3));
    let mut full_links = vec![0usize; full_point_count];
    let mut full_symmetry = vec![0usize; full_point_count];
    let mut final_irreducible_count = 0usize;
    let mut final_full_count = 0usize;

    for point in 0..work_point_count {
        let fractional = [
            fractional_coordinate(work_grid[(point, 0)], shift[0], divisions[0]),
            fractional_coordinate(work_grid[(point, 1)], shift[1], divisions[1]),
            fractional_coordinate(work_grid[(point, 2)], shift[2], divisions[2]),
        ];
        let vector = kmesh_vector(reciprocal_vectors, fractional);
        for axis in 0..3 {
            work_vectors[(point, axis)] = vector[axis];
        }

        if work_links[point] == point + 1 {
            final_irreducible_count += 1;
            work_links[point] = final_irreducible_count;
            for axis in 0..3 {
                irreducible_vectors[(final_irreducible_count - 1, axis)] = vector[axis];
                irreducible_fractional_vectors[(final_irreducible_count - 1, axis)] =
                    fractional[axis];
            }
        } else {
            work_links[point] = work_links[work_links[point] - 1];
        }

        if full_work_links[point] == final_full_count + 1 {
            final_full_count += 1;
            full_links[final_full_count - 1] = work_links[point];
            full_symmetry[final_full_count - 1] = work_symmetry[point];
            for axis in 0..3 {
                full_vectors[(final_full_count - 1, axis)] = vector[axis];
            }
        }
    }

    if final_irreducible_count != irreducible_point_count {
        return Err(KSpaceError::KMeshReductionIrreducibleCountMismatch {
            expected: irreducible_point_count,
            actual: final_irreducible_count,
        });
    }
    validate_full_mesh_link_symmetry(&full_links, &full_symmetry)?;

    Ok(KMeshReduction {
        shift,
        total_weight,
        work_grid,
        work_vectors,
        full_vectors,
        irreducible_vectors,
        irreducible_fractional_vectors,
        work_weights,
        full_weights,
        irreducible_weights,
        work_links,
        full_links,
        work_symmetry,
        full_symmetry,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `divisi`.
///
/// FEFF removes repeated common prime factors from every k-list component and
/// divides `idiv` by the same factor. Its control flow exits completely on the
/// first failed prime divisibility test, so this intentionally does not compute
/// a full greatest common divisor.
pub fn reduce_kmesh_common_divisor(
    k_list: ArrayView2<'_, i32>,
    division: usize,
) -> Result<KMeshDivisionReduction, KSpaceError> {
    if k_list.nrows() == 0 || k_list.ncols() != 3 {
        return Err(KSpaceError::InvalidKMeshListShape {
            rows: k_list.nrows(),
            columns: k_list.ncols(),
        });
    }

    let mut reduced = k_list.to_owned();
    let mut common_divisor = 1usize;
    'prime_search: for prime in DIVISI_PRIMES {
        for _ in 0..DIVISI_ITERATIONS {
            if reduced.iter().any(|value| value % prime != 0) {
                break 'prime_search;
            }
            common_divisor = common_divisor
                .checked_mul(prime as usize)
                .ok_or(KSpaceError::KMeshCommonDivisorOverflow)?;
            reduced.mapv_inplace(|value| value / prime);
        }
    }

    Ok(KMeshDivisionReduction {
        k_list: reduced,
        division: (division / common_divisor).max(1),
        common_divisor,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `sdef`.
///
/// FEFF rewrites symmetry-operation matrices for alternate centered-lattice
/// labels derived from CXY settings. `lattice` is interpreted as FEFF's
/// three-character `lattic` code, padded with spaces and uppercased for Rust
/// caller convenience.
pub fn redefine_lattice_symmetry_operations(
    operations: ArrayView3<'_, i32>,
    lattice: &str,
) -> Result<Array3<i32>, KSpaceError> {
    validate_symmetry_operation_shape(operations)?;

    let mut redefined = operations.to_owned();
    match lattice_code3(lattice) {
        [b'C', b'X', b'Z'] | [b'B', b'O', b' '] => {
            for operation in 0..redefined.shape()[0] {
                swap_operation_entries(&mut redefined, operation, (1, 1), (2, 2));
                swap_operation_entries(&mut redefined, operation, (0, 1), (0, 2));
                swap_operation_entries(&mut redefined, operation, (1, 2), (2, 1));
                swap_operation_entries(&mut redefined, operation, (1, 0), (2, 0));
            }
        }
        [b'C', b'Y', b'Z'] | [b'A', b'O', b' '] => {
            for operation in 0..redefined.shape()[0] {
                swap_operation_entries(&mut redefined, operation, (0, 0), (2, 2));
                swap_operation_entries(&mut redefined, operation, (0, 1), (2, 1));
                swap_operation_entries(&mut redefined, operation, (0, 2), (2, 0));
                swap_operation_entries(&mut redefined, operation, (1, 2), (1, 0));
            }
        }
        _ => {}
    }
    Ok(redefined)
}

/// Port of FEFF `KSPACE/kmesh.f90` `sdefl`.
///
/// FEFF transforms LAPW-style symmetry operations into the current lattice
/// basis as `rbas * operation * transpose(gbas) / (2*pi)` for orthogonal
/// lattices, and also for the special non-orthogonal `CXZ` setting. Other
/// non-orthogonal settings keep the integer operation table unchanged.
pub fn transform_lapw_symmetry_operations(
    direct_vectors: ArrayView2<'_, Real>,
    reciprocal_vectors: ArrayView2<'_, Real>,
    operations: ArrayView3<'_, i32>,
    lattice: &str,
    orthogonal: bool,
) -> Result<Array3<i32>, KSpaceError> {
    validate_matrix("direct_vectors", direct_vectors)?;
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_symmetry_operation_shape(operations)?;

    if !orthogonal && lattice_code3(lattice) != [b'C', b'X', b'Z'] {
        return Ok(operations.to_owned());
    }

    let mut transformed = Array3::<i32>::zeros(operations.raw_dim());
    for operation in 0..operations.shape()[0] {
        for row in 0..3 {
            for column in 0..3 {
                let value = transformed_symmetry_entry(
                    direct_vectors,
                    reciprocal_vectors,
                    operations,
                    operation,
                    row,
                    column,
                );
                transformed[(operation, row, column)] =
                    round_symmetry_operation_entry(value, operation, row, column)?;
            }
        }
    }
    Ok(transformed)
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

fn row_norm(matrix: ArrayView2<'_, Real>, row: usize) -> Real {
    (matrix[(row, 0)].powi(2) + matrix[(row, 1)].powi(2) + matrix[(row, 2)].powi(2)).sqrt()
}

fn mesh_division_from_real(component: usize, value: Real) -> Result<usize, KSpaceError> {
    if !value.is_finite() || value < 0.0 || value >= usize::MAX as Real {
        return Err(KSpaceError::KMeshDivisionOverflow { component, value });
    }
    Ok((value as usize).max(1))
}

fn kmesh_point_count(divisions: [usize; 3]) -> Result<usize, KSpaceError> {
    divisions
        .into_iter()
        .map(|division| division.checked_add(1))
        .try_fold(1usize, |product, value| {
            let value = value.ok_or(KSpaceError::KMeshPointCountOverflow)?;
            product
                .checked_mul(value)
                .ok_or(KSpaceError::KMeshPointCountOverflow)
        })
}

fn kmesh_cell_count(divisions: [usize; 3]) -> Result<usize, KSpaceError> {
    divisions.into_iter().try_fold(1usize, |product, division| {
        product
            .checked_mul(division)
            .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)
    })
}

fn work_mesh_coordinate(
    point: usize,
    row_stride: usize,
    plane_stride: usize,
    axis: usize,
) -> usize {
    let first = point / plane_stride;
    let second = (point - first * plane_stride) / row_stride;
    let third = point - first * plane_stride - second * row_stride;
    match axis {
        0 => first,
        1 => second,
        _ => third,
    }
}

fn kmesh_submesh_shift(operations: ArrayView3<'_, i32>) -> [usize; 3] {
    for operation in 0..operations.shape()[0] {
        for vertex in 1usize..=8 {
            let first = (vertex - 1) / 4;
            let second = (vertex - first * 4 - 1) / 2;
            let third = vertex - first * 4 - second * 2 - 1;
            let doubled = [
                2 * (first as i128) + 1,
                2 * (second as i128) + 1,
                2 * (third as i128) + 1,
            ];
            if (0..3)
                .any(|axis| (operation_row_dot(operations, operation, axis, doubled) - 1) % 2 != 0)
            {
                return [0, 0, 0];
            }
        }
    }
    [1, 1, 1]
}

fn mapped_work_mesh_index(
    operations: ArrayView3<'_, i32>,
    operation: usize,
    coordinates: [usize; 3],
    divisions: [usize; 3],
    shift: [usize; 3],
    row_stride: usize,
    plane_stride: usize,
) -> Result<usize, KSpaceError> {
    let doubled = [
        2 * i128::from(coordinates[0] as i64) + i128::from(shift[0] as i64),
        2 * i128::from(coordinates[1] as i64) + i128::from(shift[1] as i64),
        2 * i128::from(coordinates[2] as i64) + i128::from(shift[2] as i64),
    ];
    let mut mapped = [0usize; 3];
    for axis in 0..3 {
        let modulus = 2 * i128::from(divisions[axis] as i64);
        let mut value = operation_row_dot(operations, operation, axis, doubled) % modulus;
        if value < 0 {
            value += modulus;
        }
        value = (value - i128::from(shift[axis] as i64)) / 2;
        mapped[axis] = usize::try_from(value).map_err(|_| KSpaceError::KMeshPointCountOverflow)?;
    }
    mapped[0]
        .checked_mul(plane_stride)
        .and_then(|first| {
            mapped[1]
                .checked_mul(row_stride)
                .and_then(|second| first.checked_add(second))
        })
        .and_then(|value| value.checked_add(mapped[2]))
        .ok_or(KSpaceError::KMeshPointCountOverflow)
}

fn operation_row_dot(
    operations: ArrayView3<'_, i32>,
    operation: usize,
    row: usize,
    vector: [i128; 3],
) -> i128 {
    (0..3)
        .map(|column| i128::from(operations[(operation, row, column)]) * vector[column])
        .sum()
}

fn boundary_weight(coordinates: [usize; 3], divisions: [usize; 3]) -> Real {
    (0..3).fold(1.0, |weight, axis| {
        if coordinates[axis].is_multiple_of(divisions[axis]) {
            weight / 2.0
        } else {
            weight
        }
    })
}

fn checked_full_work_link(
    link: usize,
    index: usize,
    full_point_count: usize,
) -> Result<usize, KSpaceError> {
    if link == 0 || link > full_point_count {
        Err(KSpaceError::InvalidFullMeshLink {
            index,
            value: link,
            full_point_count,
        })
    } else {
        Ok(link)
    }
}

fn fractional_coordinate(coordinate: usize, shift: usize, division: usize) -> Real {
    (coordinate as Real + (shift as Real) / 2.0) / (division as Real)
}

fn kmesh_vector(reciprocal_vectors: ArrayView2<'_, Real>, fractional: Vector3) -> Vector3 {
    let mut vector = [0.0; 3];
    for axis in 0..3 {
        vector[axis] = reciprocal_vectors[(0, axis)] * fractional[0]
            + reciprocal_vectors[(1, axis)] * fractional[1]
            + reciprocal_vectors[(2, axis)] * fractional[2];
    }
    vector
}

fn validate_full_mesh_link_symmetry(
    full_links: &[usize],
    full_symmetry: &[usize],
) -> Result<(), KSpaceError> {
    for first in 0..full_links.len() {
        for second in first + 1..full_links.len() {
            if full_links[first] == full_links[second]
                && full_symmetry[first] == full_symmetry[second]
            {
                return Err(KSpaceError::DuplicateFullMeshLinkSymmetry {
                    first,
                    second,
                    link: full_links[first],
                    symmetry: full_symmetry[first],
                });
            }
        }
    }
    Ok(())
}

fn validate_kmesh_divisions(divisions: [usize; 3]) -> Result<(), KSpaceError> {
    for (component, value) in divisions.into_iter().enumerate() {
        if value == 0 {
            return Err(KSpaceError::InvalidKMeshDivision { component, value });
        }
    }
    Ok(())
}

fn validate_tetrahedron_offsets(
    tetrahedron_offsets: ArrayView3<'_, i32>,
) -> Result<(), KSpaceError> {
    let shape = tetrahedron_offsets.shape();
    if shape[0] != 6 || shape[1] != 4 || shape[2] != 3 {
        return Err(KSpaceError::InvalidTetrahedronOffsetShape {
            tetrahedra: shape[0],
            corners: shape[1],
            coordinates: shape[2],
        });
    }
    for ((tetrahedron, corner, axis), &value) in tetrahedron_offsets.indexed_iter() {
        if value != 0 && value != 1 {
            return Err(KSpaceError::InvalidTetrahedronOffset {
                tetrahedron,
                corner,
                axis,
                value,
            });
        }
    }
    Ok(())
}

fn validate_work_mesh_links(
    point_links: &[usize],
    expected_point_count: usize,
    irreducible_point_count: usize,
) -> Result<(), KSpaceError> {
    if irreducible_point_count == 0 {
        return Err(KSpaceError::InvalidIrreducibleKPointCount {
            count: irreducible_point_count,
        });
    }
    if point_links.len() != expected_point_count {
        return Err(KSpaceError::InvalidWorkMeshLinkCount {
            expected: expected_point_count,
            actual: point_links.len(),
        });
    }
    for (index, &value) in point_links.iter().enumerate() {
        if value == 0 || value > irreducible_point_count {
            return Err(KSpaceError::InvalidWorkMeshLink {
                index,
                value,
                irreducible_point_count,
            });
        }
    }
    Ok(())
}

fn tetrahedron_corner_index(
    tetrahedron_offsets: ArrayView3<'_, i32>,
    tetrahedron: usize,
    corner: usize,
) -> Result<[usize; 3], KSpaceError> {
    let mut offset = [0usize; 3];
    for axis in 0..3 {
        offset[axis] = match tetrahedron_offsets[(tetrahedron, corner, axis)] {
            0 => 0,
            1 => 1,
            value => {
                return Err(KSpaceError::InvalidTetrahedronOffset {
                    tetrahedron,
                    corner,
                    axis,
                    value,
                });
            }
        };
    }
    Ok(offset)
}

fn tetrahedron_work_mesh_offsets(
    tetrahedron_offsets: ArrayView3<'_, i32>,
    row_stride: usize,
    plane_stride: usize,
) -> Result<[[usize; 4]; 6], KSpaceError> {
    let mut offsets = [[0usize; 4]; 6];
    for (tetrahedron, row_offsets) in offsets.iter_mut().enumerate() {
        for (corner, corner_offset) in row_offsets.iter_mut().enumerate() {
            *corner_offset = offset_work_mesh_index(
                0,
                row_stride,
                plane_stride,
                tetrahedron_corner_index(tetrahedron_offsets, tetrahedron, corner)?,
            )?;
        }
    }
    Ok(offsets)
}

fn offset_work_mesh_index(
    base: usize,
    row_stride: usize,
    plane_stride: usize,
    offset: [usize; 3],
) -> Result<usize, KSpaceError> {
    base.checked_add(
        offset[0]
            .checked_mul(plane_stride)
            .and_then(|first| {
                offset[1]
                    .checked_mul(row_stride)
                    .and_then(|second| first.checked_add(second))
            })
            .and_then(|value| value.checked_add(offset[2]))
            .ok_or(KSpaceError::KMeshPointCountOverflow)?,
    )
    .ok_or(KSpaceError::KMeshPointCountOverflow)
}

fn tetrahedron_vertex_order(shortest_diagonal: usize) -> [usize; 8] {
    match shortest_diagonal {
        1 => [2, 1, 4, 3, 6, 5, 8, 7],
        2 => [3, 4, 1, 2, 7, 8, 5, 6],
        3 => [5, 6, 7, 8, 1, 2, 3, 4],
        _ => [1, 2, 3, 4, 5, 6, 7, 8],
    }
}

fn tetrahedron_vertices(vertex_order: [usize; 8]) -> [[usize; 4]; 6] {
    [
        [
            vertex_order[0],
            vertex_order[1],
            vertex_order[3],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[3],
            vertex_order[2],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[2],
            vertex_order[6],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[6],
            vertex_order[4],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[4],
            vertex_order[5],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[5],
            vertex_order[1],
            vertex_order[7],
        ],
    ]
}

fn vertex_coordinate(vertex: usize, axis: usize) -> i32 {
    let first = (vertex - 1) / 4;
    let second = (vertex - first * 4 - 1) / 2;
    let third = vertex - first * 4 - second * 2 - 1;
    match axis {
        0 => first as i32,
        1 => second as i32,
        _ => third as i32,
    }
}

fn swap_operation_entries(
    operations: &mut Array3<i32>,
    operation: usize,
    first: (usize, usize),
    second: (usize, usize),
) {
    let first_value = operations[(operation, first.0, first.1)];
    operations[(operation, first.0, first.1)] = operations[(operation, second.0, second.1)];
    operations[(operation, second.0, second.1)] = first_value;
}

fn lattice_code3(lattice: &str) -> [u8; 3] {
    let mut code = [b' '; 3];
    for (index, byte) in lattice.bytes().take(3).enumerate() {
        code[index] = byte.to_ascii_uppercase();
    }
    code
}

fn has_non_right_angle(angles: Vector3) -> bool {
    angles.into_iter().any(|angle| !is_feff_right_angle(angle))
}

fn is_feff_right_angle(angle: Real) -> bool {
    (angle - BRAVAIS_RIGHT_ANGLE).abs() <= BRAVAIS_ANGLE_EPSILON
}

fn transformed_symmetry_entry(
    direct_vectors: ArrayView2<'_, Real>,
    reciprocal_vectors: ArrayView2<'_, Real>,
    operations: ArrayView3<'_, i32>,
    operation: usize,
    row: usize,
    column: usize,
) -> Real {
    let mut value = 0.0;
    for left in 0..3 {
        for right in 0..3 {
            value += direct_vectors[(row, left)]
                * Real::from(operations[(operation, left, right)])
                * reciprocal_vectors[(column, right)]
                / PI2;
        }
    }
    value
}

fn round_symmetry_operation_entry(
    value: Real,
    operation: usize,
    row: usize,
    column: usize,
) -> Result<i32, KSpaceError> {
    let rounded = value.round();
    if rounded.is_finite() && rounded >= i32::MIN as Real && rounded <= i32::MAX as Real {
        Ok(rounded as i32)
    } else {
        Err(KSpaceError::SymmetryOperationValueOverflow {
            operation: operation + 1,
            row: row + 1,
            column: column + 1,
            value,
        })
    }
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
    validate_symmetry_operation_shape(operations)?;
    let operation_count = operations.shape()[0];
    if operation_count == 0 {
        return Err(KSpaceError::NoSymmetryOperations);
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

fn validate_symmetry_operation_shape(operations: ArrayView3<'_, i32>) -> Result<(), KSpaceError> {
    let shape = operations.shape();
    if shape[1] != 3 || shape[2] != 3 {
        return Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: shape[0],
            rows: shape[1],
            columns: shape[2],
        });
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
    use ndarray::{ArrayView1, arr2, array};

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
        let expected = skew_reciprocal_basis();
        assert_matrix_close(reciprocal.view(), expected.view());

        let roundtrip = reciprocal_lattice_vectors(reciprocal.view())?;
        assert_matrix_close(roundtrip.view(), direct.view());
        Ok(())
    }

    #[test]
    fn kmesh_bravais_basis_matches_feff_bravais_reference() -> Result<(), KSpaceError> {
        let right_angles = [BRAVAIS_RIGHT_ANGLE; 3];
        let triclinic_angles = [1.2, 1.3, 1.1];
        let monoclinic_angles = [BRAVAIS_RIGHT_ANGLE, BRAVAIS_RIGHT_ANGLE, 1.2];
        let cases = vec![
            (
                "H  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [true, false, false],
                false,
                17.901_484_003_701_512,
                arr2(&[
                    [1.732_050_776_481_628_4, -1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]),
            ),
            (
                "F  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [1.0, 1.5, 2.0],
                0.5,
                [true, true, true],
                true,
                41.341_705_691_712_875,
                arr2(&[[0.0, 1.5, 2.0], [1.0, 0.0, 2.0], [1.0, 1.5, 0.0]]),
            ),
            (
                "B  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [1.0, 1.5, 2.0],
                0.5,
                [true, true, true],
                true,
                20.670_852_845_856_437,
                arr2(&[[-1.0, 1.5, 2.0], [1.0, -1.5, 2.0], [1.0, 1.5, -2.0]]),
            ),
            (
                "P  ",
                [2.0, 3.0, 4.0],
                triclinic_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [false, false, false],
                false,
                12.539_759_914_879_173,
                arr2(&[
                    [
                        1.768_622_103_620_578_5,
                        0.765_345_256_288_045_5,
                        0.534_997_657_249_174_7,
                    ],
                    [0.0, 2.796_117_257_901_679, 1.087_073_263_430_020_9],
                    [0.0, 0.0, 4.0],
                ]),
            ),
            (
                "C  ",
                [2.0, 3.0, 4.0],
                monoclinic_angles,
                [0.932_039_085_967_226_3, 3.0, 2.0],
                1.0,
                [false, true, false],
                false,
                22.178_096_559_550_61,
                arr2(&[
                    [0.932_039_085_967_226_3, 0.362_357_754_476_673_6, -2.0],
                    [0.0, 3.0, 0.0],
                    [0.932_039_085_967_226_3, 0.362_357_754_476_673_6, 2.0],
                ]),
            ),
            (
                "P  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [false, false, false],
                true,
                10.335_426_422_928_219,
                arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
            ),
            (
                "CXZ",
                [2.0, 3.0, 4.0],
                right_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [false, true, false],
                true,
                20.670_852_845_856_437,
                arr2(&[[1.0, 0.0, -2.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]]),
            ),
            (
                "CYZ",
                [2.0, 3.0, 4.0],
                right_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [false, false, true],
                true,
                20.670_852_845_856_437,
                arr2(&[[2.0, 0.0, 0.0], [0.0, 1.5, -2.0], [0.0, 1.5, 2.0]]),
            ),
            (
                "C  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [true, false, false],
                true,
                20.670_852_845_856_437,
                arr2(&[[1.0, -1.5, 0.0], [1.0, 1.5, 0.0], [0.0, 0.0, 4.0]]),
            ),
            (
                "M  ",
                [2.0, 3.0, 4.0],
                monoclinic_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [false, false, false],
                false,
                11.089_048_279_775_305,
                arr2(&[
                    [1.864_078_171_934_452_6, 0.724_715_508_953_347_2, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]),
            ),
            (
                "R  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [2.0, 3.0, 4.0],
                1.0,
                [true, true, true],
                false,
                53.704_451_047_204_61,
                arr2(&[
                    [0.577_350_269_189_625_8, -1.0, 1.333_333_333_333_333_3],
                    [0.577_350_269_189_625_8, 1.0, 1.333_333_333_333_333_3],
                    [-1.154_700_538_379_251_7, 0.0, 1.333_333_333_333_333_3],
                ]),
            ),
            (
                "I  ",
                [2.0, 3.0, 4.0],
                right_angles,
                [1.0, 3.0, 4.0],
                0.5,
                [true, true, true],
                true,
                62.012_558_537_569_31,
                arr2(&[[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
            ),
        ];

        for (
            lattice,
            lengths,
            angles,
            adjusted_lengths,
            afact,
            dependencies,
            orthogonal,
            brillouin_zone_volume,
            direct_vectors,
        ) in cases
        {
            let basis = kmesh_bravais_basis(lattice, lengths, angles)?;
            assert_vector_values_close(basis.adjusted_lengths, adjusted_lengths);
            assert_close(basis.afact, afact);
            assert_eq!(basis.dependencies, dependencies);
            assert_eq!(basis.orthogonal, orthogonal);
            assert_close(basis.brillouin_zone_volume, brillouin_zone_volume);
            assert_matrix_close(basis.direct_vectors.view(), direct_vectors.view());
        }

        let hexagonal = kmesh_bravais_basis("H  ", [2.0, 3.0, 4.0], right_angles)?;
        assert_matrix_close(
            hexagonal.reciprocal_vectors.view(),
            arr2(&[
                [3.627_598_894_524_551_6, 1.813_799_447_262_275_8, 0.0],
                [0.0, 3.141_592_741_012_573_2, 0.0],
                [0.0, 0.0, 1.570_796_370_506_286_6],
            ])
            .view(),
        );

        let body = kmesh_bravais_basis("I  ", [2.0, 3.0, 4.0], right_angles)?;
        assert_matrix_close(
            body.reciprocal_vectors.view(),
            arr2(&[
                [0.0, 3.141_592_741_012_573_2, 3.141_592_741_012_573_2],
                [3.141_592_741_012_573_2, 0.0, 3.141_592_741_012_573_2],
                [3.141_592_741_012_573_2, 3.141_592_741_012_573_2, 0.0],
            ])
            .view(),
        );
        Ok(())
    }

    #[test]
    fn kmesh_basis_divisions_match_feff_basdiv_reference() -> Result<(), KSpaceError> {
        let reciprocal = skew_reciprocal_basis();
        let cases = [
            ([false, false, false], 120, [6, 4, 3], 140),
            ([true, false, false], 120, [5, 5, 3], 144),
            ([false, true, false], 120, [4, 4, 4], 125),
            ([false, false, true], 120, [6, 4, 4], 175),
            ([true, true, false], 120, [4, 4, 4], 125),
            ([false, false, false], 4, [2, 1, 1], 12),
        ];

        for (dependencies, requested, divisions, mesh_points) in cases {
            assert_eq!(
                kmesh_basis_divisions(reciprocal.view(), requested, dependencies)?,
                KMeshDivisions {
                    divisions,
                    mesh_points,
                }
            );
        }
        Ok(())
    }

    #[test]
    fn kmesh_tetrahedron_division_matches_feff_tetdiv_reference() -> Result<(), KSpaceError> {
        let branch_one = array![
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
            [[0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],
        ];
        let branch_two = array![
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]],
        ];
        let branch_three = array![
            [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
            [[0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1]],
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
            [[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]],
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]],
        ];
        let branch_four = array![
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            [[1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
            [[1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        ];

        let cases = [
            (
                [1, 1, 1],
                arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                branch_one.clone(),
            ),
            (
                [1, 1, 1],
                arr2(&[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                branch_two.clone(),
            ),
            (
                [1, 1, 1],
                arr2(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                branch_three.clone(),
            ),
            (
                [1, 1, 1],
                arr2(&[[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                branch_four,
            ),
            (
                [2, 3, 4],
                arr2(&[[2.0, 0.5, 0.0], [0.0, 3.0, 0.25], [0.1, 0.0, 4.0]]),
                branch_three,
            ),
        ];

        for (divisions, reciprocal, expected) in cases {
            assert_eq!(
                kmesh_tetrahedron_division(divisions, reciprocal.view())?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn kmesh_tetrahedron_records_match_feff_tetcnt_reference() -> Result<(), KSpaceError> {
        let reciprocal = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let offsets = kmesh_tetrahedron_division([1, 1, 1], reciprocal.view())?;

        let identity =
            kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1, 2, 3, 4, 5, 6, 7, 8], 8)?;
        assert_eq!(identity.irreducible_point_count, 8);
        assert_eq!(identity.tetrahedron_count, 6);
        assert_eq!(identity.unique_tetrahedron_count, 6);
        assert_close(identity.tetrahedron_weight, 1.0 / 6.0);
        assert_eq!(
            identity.write_chunk_size,
            KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE
        );
        assert_eq!(identity.record_count, 1);
        assert_eq!(
            identity.records,
            array![
                [1_usize, 1, 2, 4, 8],
                [1, 1, 2, 6, 8],
                [1, 1, 3, 4, 8],
                [1, 1, 3, 7, 8],
                [1, 1, 5, 6, 8],
                [1, 1, 5, 7, 8],
            ]
        );

        let collapsed =
            kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1, 2, 2, 3, 2, 3, 3, 4], 4)?;
        assert_eq!(collapsed.irreducible_point_count, 4);
        assert_eq!(collapsed.tetrahedron_count, 6);
        assert_eq!(collapsed.unique_tetrahedron_count, 1);
        assert_close(collapsed.tetrahedron_weight, 1.0 / 6.0);
        assert_eq!(collapsed.record_count, 1);
        assert_eq!(collapsed.records, array![[6_usize, 1, 2, 3, 4]]);

        let stretched_offsets = kmesh_tetrahedron_division([2, 1, 1], reciprocal.view())?;
        let stretched_links = (1..=12).collect::<Vec<_>>();
        let stretched =
            kmesh_tetrahedron_records(stretched_offsets.view(), [2, 1, 1], &stretched_links, 12)?;
        assert_eq!(stretched.irreducible_point_count, 12);
        assert_eq!(stretched.tetrahedron_count, 12);
        assert_eq!(stretched.unique_tetrahedron_count, 12);
        assert_close(stretched.tetrahedron_weight, 1.0 / 12.0);
        assert_eq!(stretched.record_count, 1);
        assert_eq!(
            stretched.records,
            array![
                [1_usize, 1, 2, 4, 8],
                [1, 1, 2, 6, 8],
                [1, 1, 3, 4, 8],
                [1, 1, 3, 7, 8],
                [1, 1, 5, 6, 8],
                [1, 1, 5, 7, 8],
                [1, 5, 6, 8, 12],
                [1, 5, 6, 10, 12],
                [1, 5, 7, 8, 12],
                [1, 5, 7, 11, 12],
                [1, 5, 9, 10, 12],
                [1, 5, 9, 11, 12],
            ]
        );
        Ok(())
    }

    #[test]
    fn reduce_kmesh_irreducible_points_matches_feff_reduz_reference() -> Result<(), KSpaceError> {
        let reciprocal = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let identity_operations = array![[[1, 0, 0], [0, 1, 0], [0, 0, 1]]];
        let identity = reduce_kmesh_irreducible_points(
            [1, 1, 1],
            identity_operations.view(),
            reciprocal.view(),
        )?;
        assert_eq!(identity.shift, [1, 1, 1]);
        assert_close(identity.total_weight, 1.0);
        assert_eq!(identity.work_links, vec![1; 8]);
        assert_eq!(identity.work_symmetry, vec![1; 8]);
        assert_eq!(identity.full_links, vec![1]);
        assert_eq!(identity.full_symmetry, vec![1]);
        assert_eq!(
            identity.work_grid,
            array![
                [0_usize, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        );
        assert_array1_close(
            identity.work_weights.view(),
            array![0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125].view(),
        );
        assert_array1_close(identity.full_weights.view(), array![1.0].view());
        assert_array1_close(identity.irreducible_weights.view(), array![1.0].view());
        assert_matrix_close(
            identity.work_vectors.view(),
            arr2(&[
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 1.5],
                [0.5, 1.5, 0.5],
                [0.5, 1.5, 1.5],
                [1.5, 0.5, 0.5],
                [1.5, 0.5, 1.5],
                [1.5, 1.5, 0.5],
                [1.5, 1.5, 1.5],
            ])
            .view(),
        );
        assert_matrix_close(
            identity.full_vectors.view(),
            arr2(&[[0.5, 0.5, 0.5]]).view(),
        );
        assert_matrix_close(
            identity.irreducible_fractional_vectors.view(),
            arr2(&[[0.5, 0.5, 0.5]]).view(),
        );

        let sign = reduce_kmesh_irreducible_points(
            [2, 1, 1],
            sign_flip_symmetry_operations().view(),
            reciprocal.view(),
        )?;
        assert_eq!(sign.shift, [1, 1, 1]);
        assert_close(sign.total_weight, 2.0);
        assert_eq!(sign.work_links, vec![1; 12]);
        assert_eq!(sign.work_symmetry, vec![1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1]);
        assert_eq!(sign.full_links, vec![1, 1]);
        assert_eq!(sign.full_symmetry, vec![1, 3]);
        assert_array1_close(
            sign.work_weights.view(),
            array![
                0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625,
                0.0625
            ]
            .view(),
        );
        assert_array1_close(sign.full_weights.view(), array![0.5, 0.5].view());
        assert_array1_close(sign.irreducible_weights.view(), array![1.0].view());
        assert_matrix_close(
            sign.full_vectors.view(),
            arr2(&[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]).view(),
        );
        assert_matrix_close(
            sign.irreducible_fractional_vectors.view(),
            arr2(&[[0.25, 0.5, 0.5]]).view(),
        );

        let shear_operations = array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        ];
        let skew_reciprocal = arr2(&[[2.0, 0.5, 0.0], [0.0, 3.0, 0.25], [0.1, 0.0, 4.0]]);
        let shear = reduce_kmesh_irreducible_points(
            [2, 2, 1],
            shear_operations.view(),
            skew_reciprocal.view(),
        )?;
        assert_eq!(shear.shift, [0, 0, 0]);
        assert_close(shear.total_weight, 4.0);
        assert_eq!(
            shear.work_links,
            vec![1, 1, 2, 2, 1, 1, 3, 3, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1]
        );
        assert_eq!(
            shear.work_symmetry,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(shear.full_links, vec![1, 2, 3, 2]);
        assert_eq!(shear.full_symmetry, vec![1, 1, 1, 2]);
        assert_array1_close(
            shear.full_weights.view(),
            array![0.25, 0.25, 0.25, 0.25].view(),
        );
        assert_array1_close(
            shear.irreducible_weights.view(),
            array![0.25, 0.5, 0.25].view(),
        );
        assert_matrix_close(
            shear.full_vectors.view(),
            arr2(&[
                [0.0, 0.0, 0.0],
                [0.0, 1.5, 0.125],
                [1.0, 0.25, 0.0],
                [1.0, 1.75, 0.125],
            ])
            .view(),
        );
        assert_matrix_close(
            shear.irreducible_vectors.view(),
            arr2(&[[0.0, 0.0, 0.0], [0.0, 1.5, 0.125], [1.0, 0.25, 0.0]]).view(),
        );
        assert_matrix_close(
            shear.irreducible_fractional_vectors.view(),
            arr2(&[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]]).view(),
        );
        Ok(())
    }

    #[test]
    fn kmesh_arbitrary_mesh_matches_feff_arbmsh_flow_reference() -> Result<(), KSpaceError> {
        let reciprocal = skew_reciprocal_basis();
        let mesh = kmesh_arbitrary_mesh(
            reciprocal.view(),
            sign_flip_symmetry_operations().view(),
            4,
            [false, false, false],
            true,
        )?;

        assert_eq!(mesh.requested_point_count, 4);
        assert_eq!(mesh.divisions, [2, 1, 1]);
        assert_eq!(mesh.work_point_count, 12);
        assert_eq!(mesh.full_point_count, 2);
        assert_eq!(mesh.irreducible_point_count, 1);
        assert_close(mesh.total_weight, 2.0);
        assert_eq!(mesh.reduction.shift, [1, 1, 1]);
        assert_eq!(mesh.reduction.work_links, vec![1; 12]);
        assert_eq!(
            mesh.reduction.work_symmetry,
            vec![1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1]
        );
        assert_eq!(mesh.reduction.full_links, vec![1, 1]);
        assert_eq!(mesh.reduction.full_symmetry, vec![1, 3]);
        assert_array1_close(
            mesh.reduction.work_weights.view(),
            array![
                0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625,
                0.0625
            ]
            .view(),
        );
        assert_array1_close(mesh.reduction.full_weights.view(), array![0.5, 0.5].view());
        assert_array1_close(
            mesh.reduction.irreducible_weights.view(),
            array![1.0].view(),
        );
        assert_matrix_close(
            mesh.reduction.irreducible_fractional_vectors.view(),
            arr2(&[[0.25, 0.5, 0.5]]).view(),
        );

        let tetrahedra = mesh
            .tetrahedra
            .as_ref()
            .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)?;
        assert_eq!(tetrahedra.irreducible_point_count, 1);
        assert_eq!(tetrahedra.tetrahedron_count, 12);
        assert_eq!(tetrahedra.unique_tetrahedron_count, 1);
        assert_close(tetrahedra.tetrahedron_weight, 1.0 / 12.0);
        assert_eq!(tetrahedra.record_count, 1);
        assert_eq!(tetrahedra.records, array![[12_usize, 1, 1, 1, 1]]);

        let mesh_without_tetrahedra = kmesh_arbitrary_mesh(
            reciprocal.view(),
            sign_flip_symmetry_operations().view(),
            4,
            [false, false, false],
            false,
        )?;
        assert!(mesh_without_tetrahedra.tetrahedra.is_none());
        Ok(())
    }

    #[test]
    fn reduce_kmesh_common_divisor_matches_feff_divisi_reference() -> Result<(), KSpaceError> {
        let cases = [
            (
                arr2(&[[3, 6, 9], [12, 15, 18]]),
                9,
                arr2(&[[3, 6, 9], [12, 15, 18]]),
                9,
                1,
            ),
            (
                arr2(&[[6, 12, 18], [24, 30, 36]]),
                12,
                arr2(&[[3, 6, 9], [12, 15, 18]]),
                6,
                2,
            ),
            (
                arr2(&[[8, 12, 16], [20, 24, 28]]),
                8,
                arr2(&[[2, 3, 4], [5, 6, 7]]),
                2,
                4,
            ),
            (
                arr2(&[[2, 4, 6], [4, 8, 12], [6, 12, 18]]),
                3,
                arr2(&[[1, 2, 3], [2, 4, 6], [3, 6, 9]]),
                1,
                2,
            ),
        ];

        for (k_list, division, expected_k_list, expected_division, common_divisor) in cases {
            assert_eq!(
                reduce_kmesh_common_divisor(k_list.view(), division)?,
                KMeshDivisionReduction {
                    k_list: expected_k_list,
                    division: expected_division,
                    common_divisor,
                }
            );
        }
        Ok(())
    }

    #[test]
    fn redefine_lattice_symmetry_operations_matches_feff_sdef_reference() -> Result<(), KSpaceError>
    {
        let operations = sample_sdef_operations();
        let cxz_expected = array![
            [[111, 113, 112], [131, 133, 132], [121, 123, 122]],
            [[211, 213, 212], [231, 233, 232], [221, 223, 222]]
        ];
        for lattice in ["CXZ", "BO ", "bo"] {
            assert_eq!(
                redefine_lattice_symmetry_operations(operations.view(), lattice)?,
                cxz_expected
            );
        }

        let cyz_expected = array![
            [[133, 132, 131], [123, 122, 121], [113, 112, 111]],
            [[233, 232, 231], [223, 222, 221], [213, 212, 211]]
        ];
        for lattice in ["CYZ", "AO ", "ao"] {
            assert_eq!(
                redefine_lattice_symmetry_operations(operations.view(), lattice)?,
                cyz_expected
            );
        }

        assert_eq!(
            redefine_lattice_symmetry_operations(operations.view(), "P  ")?,
            operations
        );
        Ok(())
    }

    #[test]
    fn transform_lapw_symmetry_operations_matches_feff_sdefl_reference() -> Result<(), KSpaceError>
    {
        let operations = sample_sdefl_operations();
        let shear_direct = arr2(&[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let shear_reciprocal = reciprocal_lattice_vectors(shear_direct.view())?;
        let transformed_expected = array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 2, 0], [0, 1, 0], [0, 0, 1]]
        ];

        assert_eq!(
            transform_lapw_symmetry_operations(
                shear_direct.view(),
                shear_reciprocal.view(),
                operations.view(),
                "P  ",
                true,
            )?,
            transformed_expected
        );
        assert_eq!(
            transform_lapw_symmetry_operations(
                shear_direct.view(),
                shear_reciprocal.view(),
                operations.view(),
                "P  ",
                false,
            )?,
            operations
        );
        assert_eq!(
            transform_lapw_symmetry_operations(
                shear_direct.view(),
                shear_reciprocal.view(),
                operations.view(),
                "CXZ",
                false,
            )?,
            transformed_expected
        );

        let diagonal_direct = arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
        let diagonal_reciprocal = reciprocal_lattice_vectors(diagonal_direct.view())?;
        assert_eq!(
            transform_lapw_symmetry_operations(
                diagonal_direct.view(),
                diagonal_reciprocal.view(),
                operations.view(),
                "P  ",
                true,
            )?,
            operations
        );
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
        assert_eq!(
            kmesh_bravais_basis("P  ", [0.0, 3.0, 4.0], [BRAVAIS_RIGHT_ANGLE; 3]),
            Err(KSpaceError::DegenerateLatticeVolume { determinant: 0.0 })
        );
        assert!(matches!(
            kmesh_bravais_basis(
                "P  ",
                [Real::NAN, 3.0, 4.0],
                [BRAVAIS_RIGHT_ANGLE; 3],
            ),
            Err(KSpaceError::NonFiniteValue {
                name: "lattice_lengths",
                index: 0,
                value,
            }) if value.is_nan()
        ));
        assert_eq!(
            kmesh_basis_divisions(matrix.view(), 0, [false; 3]),
            Err(KSpaceError::InvalidKMeshPointTarget { mesh_points: 0 })
        );
        assert_eq!(
            kmesh_arbitrary_mesh(
                matrix.view(),
                sign_flip_symmetry_operations().view(),
                0,
                [false; 3],
                false,
            ),
            Err(KSpaceError::InvalidKMeshPointTarget { mesh_points: 0 })
        );
        assert_eq!(
            kmesh_basis_divisions(matrix.view(), 16, [false; 3]),
            Err(KSpaceError::DegenerateReciprocalVector {
                index: 0,
                length: 0.0,
            })
        );
        assert_eq!(
            kmesh_tetrahedron_division([0, 1, 1], matrix.view()),
            Err(KSpaceError::InvalidKMeshDivision {
                component: 0,
                value: 0,
            })
        );
        assert_eq!(
            kmesh_tetrahedron_division([1, 1, 1], bad_matrix.view()),
            Err(KSpaceError::InvalidMatrixShape {
                name: "reciprocal_vectors",
                rows: 2,
                columns: 3,
            })
        );
        let offsets = Array3::<i32>::zeros((6, 4, 3));
        assert_eq!(
            kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1; 8], 0),
            Err(KSpaceError::InvalidIrreducibleKPointCount { count: 0 })
        );
        assert_eq!(
            kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1; 7], 1),
            Err(KSpaceError::InvalidWorkMeshLinkCount {
                expected: 8,
                actual: 7,
            })
        );
        assert_eq!(
            kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1, 2, 0, 1, 1, 1, 1, 1], 2),
            Err(KSpaceError::InvalidWorkMeshLink {
                index: 2,
                value: 0,
                irreducible_point_count: 2,
            })
        );
        let bad_offsets = Array3::<i32>::zeros((6, 4, 2));
        assert_eq!(
            kmesh_tetrahedron_records(bad_offsets.view(), [1, 1, 1], &[1; 8], 1),
            Err(KSpaceError::InvalidTetrahedronOffsetShape {
                tetrahedra: 6,
                corners: 4,
                coordinates: 2,
            })
        );
        let mut bad_offsets = Array3::<i32>::zeros((6, 4, 3));
        bad_offsets[(0, 0, 0)] = 2;
        assert_eq!(
            kmesh_tetrahedron_records(bad_offsets.view(), [1, 1, 1], &[1; 8], 1),
            Err(KSpaceError::InvalidTetrahedronOffset {
                tetrahedron: 0,
                corner: 0,
                axis: 0,
                value: 2,
            })
        );
        let bad_klist = Array2::<i32>::zeros((2, 2));
        assert_eq!(
            reduce_kmesh_common_divisor(bad_klist.view(), 12),
            Err(KSpaceError::InvalidKMeshListShape {
                rows: 2,
                columns: 2,
            })
        );
        let bad_operations = Array3::<i32>::zeros((2, 2, 3));
        assert_eq!(
            redefine_lattice_symmetry_operations(bad_operations.view(), "CXZ"),
            Err(KSpaceError::InvalidSymmetryOperationShape {
                operations: 2,
                rows: 2,
                columns: 3,
            })
        );
        assert_eq!(
            transform_lapw_symmetry_operations(
                bad_matrix.view(),
                matrix.view(),
                sign_flip_symmetry_operations().view(),
                "P  ",
                true,
            ),
            Err(KSpaceError::InvalidMatrixShape {
                name: "direct_vectors",
                rows: 2,
                columns: 3,
            })
        );
        assert_eq!(
            reduce_kmesh_irreducible_points([1, 1, 1], bad_operations.view(), matrix.view(),),
            Err(KSpaceError::InvalidSymmetryOperationShape {
                operations: 2,
                rows: 2,
                columns: 3,
            })
        );
        let reciprocal = skew_reciprocal_basis();
        assert_eq!(
            kmesh_arbitrary_mesh(
                reciprocal.view(),
                bad_operations.view(),
                4,
                [false; 3],
                false,
            ),
            Err(KSpaceError::InvalidSymmetryOperationShape {
                operations: 2,
                rows: 2,
                columns: 3,
            })
        );
        let no_operations = Array3::<i32>::zeros((0, 3, 3));
        assert_eq!(
            reduce_kmesh_irreducible_points([1, 1, 1], no_operations.view(), matrix.view()),
            Err(KSpaceError::NoSymmetryOperations)
        );
        assert_eq!(
            reduce_kmesh_irreducible_points(
                [1, 1, 1],
                sign_flip_symmetry_operations().view(),
                bad_matrix.view(),
            ),
            Err(KSpaceError::InvalidMatrixShape {
                name: "reciprocal_vectors",
                rows: 2,
                columns: 3,
            })
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

        let no_translations = Array2::<Real>::zeros((0, 3));
        assert_eq!(
            symmetry_check(no_operations.view(), no_translations.view()),
            Err(KSpaceError::NoSymmetryOperations)
        );
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

    fn sample_sdef_operations() -> Array3<i32> {
        array![
            [[111, 112, 113], [121, 122, 123], [131, 132, 133]],
            [[211, 212, 213], [221, 222, 223], [231, 232, 233]]
        ]
    }

    fn sample_sdefl_operations() -> Array3<i32> {
        array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ]
    }

    fn sign_flip_symmetry_operations() -> Array3<i32> {
        array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        ]
    }

    fn skew_reciprocal_basis() -> RealMat {
        arr2(&[
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
        ])
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

    fn assert_vector_values_close(actual: Vector3, expected: Vector3) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }

    fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
        assert_eq!(actual.shape(), expected.shape());
        for ((row, column), &actual) in actual.indexed_iter() {
            assert_close(actual, expected[(row, column)]);
        }
    }

    fn assert_array1_close(actual: ArrayView1<'_, Real>, expected: ArrayView1<'_, Real>) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(*actual, *expected);
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
