use super::*;

/// Build FEFF `yprep` pair azimuths and FMS rotation tables.
///
/// For each ordered atom pair, this runs the same `getang`/`rotxan` sequence as
/// `FMS/yprep.f90`: `xphi(i,j)` is recorded for all pairs, while `rotxan`
/// stores the corresponding rotation at `drix(...,j,i)`. Diagonal rotations
/// remain zero, and off-diagonal pairs receive forward (`k=0`) and backward
/// (`k=1`) rotation tables.
pub fn fms_yprep_geometry(
    lmax: usize,
    mmax: usize,
    atoms: &[FmsAtom],
) -> Result<FmsYprepGeometry, FmsError> {
    validate_rotation_limits(lmax, mmax)?;
    if atoms.is_empty() {
        return Err(FmsError::AtomIndexOutOfRange { index: 0, len: 0 });
    }

    let mut positions = Vec::with_capacity(atoms.len());
    for (index, atom) in atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
        positions.push(atom.position);
    }

    let atom_count = atoms.len();
    let magnetic_count = lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lmax",
            value: lmax,
            lx: FMS_ROTATION_LMAX,
        })?;
    let angular_count = lmax.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lmax",
        value: lmax,
        lx: FMS_ROTATION_LMAX,
    })?;
    let mut phi = Array2::zeros((atom_count, atom_count).f());
    let mut rotations = Array6::zeros(
        (
            magnetic_count,
            magnetic_count,
            angular_count,
            2,
            atom_count,
            atom_count,
        )
            .f(),
    );

    for atom1 in 0..atom_count {
        for atom2 in 0..atom_count {
            let (beta, pair_phi) = pair_polar_angles(&positions, atom1, atom2)?;
            phi[(atom1, atom2)] = pair_phi;
            if atom1 == atom2 {
                continue;
            }
            let forward =
                fms_rotation_matrix(lmax, mmax, beta, pair_phi, FmsRotationDirection::Forward)?;
            copy_rotation_table(
                &forward.view(),
                &mut rotations,
                atom2,
                atom1,
                FmsRotationDirection::Forward,
            );
            let backward =
                fms_rotation_matrix(lmax, mmax, -beta, pair_phi, FmsRotationDirection::Backward)?;
            copy_rotation_table(
                &backward.view(),
                &mut rotations,
                atom2,
                atom1,
                FmsRotationDirection::Backward,
            );
        }
    }

    Ok(FmsYprepGeometry { phi, rotations })
}

/// Port of FEFF `rotxan`: build a phased FMS rotation table.
///
/// The returned array is indexed as `drix(m2, m1, l)` with signed magnetic
/// indices shifted by `lmax`, so FEFF `drix(m2,m1,l,k,j,i)` is
/// `table[(m2 + lmax, m1 + lmax, l)]`.
pub fn fms_rotation_matrix(
    lmax: usize,
    mmax: usize,
    beta: f32,
    phi: f32,
    direction: FmsRotationDirection,
) -> Result<Array3<Complex32>, FmsError> {
    validate_rotation_limits(lmax, mmax)?;
    if !beta.is_finite() {
        return Err(FmsError::NonFiniteRotationAngle { name: "beta" });
    }
    if !phi.is_finite() {
        return Err(FmsError::NonFiniteRotationAngle { name: "phi" });
    }

    let mut drix = Array3::zeros((2 * lmax + 1, 2 * lmax + 1, lmax + 1).f());
    let mut dri0 = Array3::<f32>::zeros(
        (
            FMS_ROTATION_LMAX + 2,
            2 * FMS_ROTATION_LMAX + 2,
            2 * FMS_ROTATION_LMAX + 2,
        )
            .f(),
    );
    fill_rotxan_small_d(lmax, mmax, beta, &mut dri0);
    copy_rotxan_small_d(lmax, mmax, &dri0.view(), &mut drix)?;
    apply_rotxan_phase(lmax, phi, direction, &mut drix)?;
    Ok(drix)
}
