//! Reciprocal-space lattice helpers ported from FEFF BAND/KSPACE routines.
//!
//! The routines here cover the small deterministic helpers used before the
//! heavier KSPACE solvers: Bravais classification from `BAND/ibravais.f90`,
//! high-symmetry K-path segment generation from `BAND/kpath.f90`, and the
//! coordinate reductions from `KSPACE/subtract_a.f90` and `change_car.f90`.
//! FEFF exits the process for unsupported lattices; Rust returns typed errors.

use ndarray::{Array2, ArrayView2};
use thiserror::Error;

use crate::{Real, RealMat, Vector3};

const PI2: Real = std::f64::consts::TAU;
const REDUCE_NEGATIVE_EPSILON: Real = -1.0e-8;

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

#[derive(Debug, Clone, Copy)]
struct Segment {
    label: &'static str,
    start: Vector3,
    end: Vector3,
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
        assert!(matches!(
            reduce_to_lattice_cell(matrix.view(), matrix.view(), [Real::NAN, 0.0, 0.0]),
            Err(KSpaceError::NonFiniteValue {
                name: "vector",
                index: 0,
                value,
            }) if value.is_nan()
        ));
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

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }
}
