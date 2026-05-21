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

use crate::{Real, RealMat, Vector3};

const PI2: Real = std::f64::consts::TAU;
const BRAVAIS_PI2: Real = 2.0 * (std::f32::consts::PI as Real);
const BRAVAIS_RIGHT_ANGLE: Real = 1_570_796.0 / 1_000_000.0;
const BRAVAIS_ANGLE_EPSILON: Real = 0.0001;
const REDUCE_NEGATIVE_EPSILON: Real = -1.0e-8;
const LATTICE_VOLUME_EPSILON: Real = Real::EPSILON;
const BASDIV_OFFSET: Real = 1.0e-6;
/// FEFF `KSPACE/m_tetrahedra.f90` `mwrit` chunk size for tetrahedron records.
pub const KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE: usize = 101;
const DIVISI_PRIMES: [i32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const DIVISI_ITERATIONS: usize = 10;

mod path;
mod symmetry;
mod types;

pub use path::define_k_path;
pub use symmetry::{
    point_group_operations, reciprocal_metric, redefine_lattice_symmetry_operations,
    symmetry_check, transform_lapw_symmetry_operations,
};
pub use types::*;

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
mod tests;
