use super::support::*;
use super::*;

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
