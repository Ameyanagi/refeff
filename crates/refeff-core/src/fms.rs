//! Full multiple-scattering helpers.
//!
//! FEFF's FMS routines use Rehr-Albers polynomial tables when building
//! multiple-scattering propagators. The helpers here keep the legacy table
//! layout explicit while returning Rust-owned `ndarray` storage.

use ndarray::{
    Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, ArrayView6, Axis, ShapeBuilder,
};
use num_complex::Complex32;
use thiserror::Error;

use crate::{Real, state::StateKet};

/// Atom record used by FEFF FMS cluster preparation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsAtom {
    /// Cartesian position in FEFF FMS single-precision arithmetic.
    pub position: [f32; 3],
    /// FEFF potential index for this atom.
    pub potential: i32,
}

/// Direction branch used by FEFF `rotxan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmsRotationDirection {
    /// FEFF `k=0`, used for forward rotations.
    Forward,
    /// FEFF `k=1`, used for backward rotations.
    Backward,
}

/// FEFF FMS pair tables for one energy point and spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsPairTables {
    /// `xrho(atom2, atom1)` complex distance table.
    pub rho: Array2<Complex32>,
    /// `xclm(m, l, atom2, atom1)` Rehr-Albers polynomial table.
    pub polynomials: Array4<Complex32>,
}

/// Inputs for one FEFF FMS free-propagator matrix element.
#[derive(Debug, Clone)]
pub struct FmsFreePropagatorInput<'a> {
    /// Bra-side FEFF state.
    pub first: StateKet,
    /// Ket-side FEFF state.
    pub second: StateKet,
    /// Pair `rho = ck * |R_i - R_j|`.
    pub rho: Complex32,
    /// Complex wave number `ck`.
    pub wave_number: Complex32,
    /// Pair mean-square displacement in Angstrom squared.
    pub mean_square_displacement: f32,
    /// FEFF `xclm(m,l,atom2,atom1)` table.
    pub xclm: ArrayView4<'a, Complex32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(...,k=1,atom2,atom1)` backward rotation table.
    pub backward_rotation: ArrayView3<'a, Complex32>,
    /// FEFF `drix(...,k=0,atom2,atom1)` forward rotation table.
    pub forward_rotation: ArrayView3<'a, Complex32>,
}

/// Inputs for building the FEFF FMS free-propagator matrix.
#[derive(Debug, Clone)]
pub struct FmsFreePropagatorMatrixInput<'a> {
    /// FEFF state kets in matrix row/column order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// Direct-space cutoff `rdirec` in Angstrom.
    pub direct_cutoff: f32,
    /// FEFF `xrho(atom2,atom1)` table.
    pub rho: ArrayView2<'a, Complex32>,
    /// Complex wave number `ck`.
    pub wave_number: Complex32,
    /// FEFF `sigsqr(atom2,atom1)` mean-square displacement table.
    pub mean_square_displacements: ArrayView2<'a, f32>,
    /// FEFF `xclm(m,l,atom2,atom1)` table.
    pub xclm: ArrayView4<'a, Complex32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(m2,m1,l,k,atom2,atom1)` rotation table.
    pub rotations: ArrayView6<'a, Complex32>,
}

/// Error returned by FEFF FMS helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FmsError {
    /// FEFF angular limits must fit the allocated `clm(lx+2, 2*lx+3)` table.
    #[error("{name}={value} is invalid for lx={lx}")]
    InvalidAngularLimit {
        name: &'static str,
        value: usize,
        lx: usize,
    },
    /// FEFF state-ket atom indices are one-based.
    #[error("state atom index must be one-based, got {atom}")]
    InvalidStateAtom { atom: usize },
    /// A zero-based Rust atom index was outside the supplied cluster table.
    #[error("atom index {index} is outside cluster length {len}")]
    AtomIndexOutOfRange { index: usize, len: usize },
    /// FMS cluster coordinates must be finite.
    #[error("atom {atom} coordinate axis {axis} must be finite")]
    NonFiniteCoordinate { atom: usize, axis: usize },
    /// FEFF FMS rotation angles must be finite.
    #[error("rotation angle {name} must be finite")]
    NonFiniteRotationAngle { name: &'static str },
    /// FMS potential indices must fit the caller-provided potential range.
    #[error("potential {potential} is outside 0..={max_potential}")]
    PotentialOutOfRange {
        potential: i32,
        max_potential: usize,
    },
    /// FEFF `sortat` requires the first atom to be the central potential.
    #[error("first atom potential {actual} does not match central potential {expected}")]
    CentralAtomMismatch { expected: i32, actual: i32 },
    /// FEFF `xgllm` is called with `mu <= l1`.
    #[error("mu={mu} is invalid for angular momentum l={angular_momentum}")]
    MuOutOfRange { mu: usize, angular_momentum: usize },
    /// An input table is too small for a required FEFF index.
    #[error("{table} table is too small for {axis} index {index}")]
    TableIndexOutOfRange {
        table: &'static str,
        axis: &'static str,
        index: usize,
    },
    /// FEFF `xnlm(mu,l)` must be finite and nonzero when used as a divisor.
    #[error("xnlm({mu},{angular_momentum}) must be finite and nonzero")]
    InvalidNormalization { mu: usize, angular_momentum: usize },
    /// `rho` appears in the denominator of FEFF `xclmz`.
    #[error("rho must be nonzero")]
    ZeroRho,
    /// `rho` must contain finite real and imaginary parts.
    #[error("rho must be finite")]
    NonFiniteRho,
    /// The complex wave number used for FMS pair tables must be finite.
    #[error("wave number must be finite")]
    NonFiniteWaveNumber,
    /// Pair mean-square displacement must be finite.
    #[error("mean-square displacement must be finite")]
    NonFiniteMeanSquareDisplacement,
    /// The direct FMS cutoff must be finite and nonnegative.
    #[error("direct FMS cutoff must be finite and nonnegative")]
    InvalidDirectCutoff,
}

/// Port of FEFF `xclmz`: Rehr-Albers Hankel-like polynomial table.
///
/// The returned matrix has FEFF's work shape `clm(lx+2, 2*lx+3)` and
/// Fortran-order strides. Rust indices are zero-based, so FEFF `clm(il, im)`
/// is `table[(il - 1, im - 1)]`.
pub fn rehr_albers_polynomials(
    lx: usize,
    lmaxp1: usize,
    mmaxp1: usize,
    rho: Complex32,
) -> Result<Array2<Complex32>, FmsError> {
    let max_lmaxp1 = lx.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lx,
        lx,
    })?;
    if lmaxp1 == 0 || lmaxp1 > max_lmaxp1 {
        return Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: lmaxp1,
            lx,
        });
    }
    if mmaxp1 == 0 {
        return Err(FmsError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
            lx,
        });
    }
    if !(rho.re.is_finite() && rho.im.is_finite()) {
        return Err(FmsError::NonFiniteRho);
    }
    if rho == Complex32::new(0.0, 0.0) {
        return Err(FmsError::ZeroRho);
    }

    let rows = lx.checked_add(2).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lx,
        lx,
    })?;
    let cols = lx
        .checked_mul(2)
        .and_then(|value| value.checked_add(3))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    let mut clm = Array2::zeros((rows, cols).f());

    let one = Complex32::new(1.0, 0.0);
    let z = Complex32::new(0.0, -1.0) / rho;
    clm[(0, 0)] = one;
    clm[(1, 0)] = one - z;

    let lmax = lmaxp1 - 1;
    for il in 2..=lmax {
        let factor = odd_factor(il, lx)? * z;
        clm[(il, 0)] = clm[(il - 2, 0)] - factor * clm[(il - 1, 0)];
    }

    let mut cmm = one;
    let mmxp1 = lmaxp1.min(mmaxp1);
    for im in 2..=mmxp1 {
        let m = im - 1;
        let cmm_factor = odd_factor(m, lx)? * z;
        cmm = -cmm * cmm_factor;
        clm[(im - 1, im - 1)] = cmm;
        clm[(im, im - 1)] = cmm * odd_factor(im, lx)? * (one - Complex32::new(im as f32, 0.0) * z);

        for il in (im + 1)..=lmax {
            let factor = odd_factor(il, lx)? * z;
            clm[(il, im - 1)] =
                clm[(il - 2, im - 1)] - factor * (clm[(il - 1, im - 1)] + clm[(il - 1, im - 2)]);
        }
    }

    Ok(clm)
}

/// Port of FEFF `athep`: sort atoms by radius from the central atom.
///
/// The sort key is `x^2 + y^2 + z^2 + (input_index + 1) * 1e-6`, matching the
/// FEFF tie-breaker that preserves the old order for equidistant atoms. The
/// returned vector contains the sorted FEFF `ra` keys.
pub fn sort_atoms_by_radius(atoms: &mut [FmsAtom]) -> Result<Vec<f64>, FmsError> {
    let mut keyed_atoms = atoms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, atom)| sort_radius_key(index, atom).map(|key| (key, atom)))
        .collect::<Result<Vec<_>, _>>()?;

    keyed_atoms.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut keys = Vec::with_capacity(keyed_atoms.len());
    for (slot, (key, atom)) in atoms.iter_mut().zip(keyed_atoms) {
        *slot = atom;
        keys.push(key);
    }
    Ok(keys)
}

/// Port of FEFF `sortat`: move representative atoms into the FMS prefix.
///
/// The input atoms must already be sorted by radial distance. `max_potential`
/// is FEFF's inclusive `npot` loop bound; potential indices `0..=npot` are
/// considered. The returned vector maps each potential to its representative
/// zero-based atom index when that potential is present.
pub fn sort_representative_atoms(
    central_potential: i32,
    max_potential: usize,
    atoms: &mut [FmsAtom],
) -> Result<Vec<Option<usize>>, FmsError> {
    let central = checked_potential(central_potential, max_potential)?;
    let first = atoms
        .first()
        .ok_or(FmsError::AtomIndexOutOfRange { index: 0, len: 0 })?;
    if first.potential != central_potential {
        return Err(FmsError::CentralAtomMismatch {
            expected: central_potential,
            actual: first.potential,
        });
    }

    for (index, atom) in atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
        checked_potential(atom.potential, max_potential)?;
    }

    let mut representative = vec![None; max_potential + 1];
    representative[central] = Some(0);
    for (potential, slot) in representative.iter_mut().enumerate() {
        if potential == central {
            continue;
        }
        *slot = atoms
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, atom)| atom.potential == potential as i32)
            .map(|(index, _)| index);
    }

    for potential in 0..=max_potential {
        let Some(point) = representative[potential] else {
            continue;
        };
        if point <= potential {
            continue;
        }

        atoms.swap(potential, point);
        for slot in representative
            .iter_mut()
            .take(max_potential + 1)
            .skip(potential + 1)
        {
            if *slot == Some(potential) {
                *slot = Some(point);
            }
        }
        representative[potential] = Some(potential);
    }

    let prefix_len = atoms.len().min(max_potential + 1);
    for (potential, representative_slot) in representative.iter_mut().enumerate() {
        let Some(point) = *representative_slot else {
            continue;
        };
        let last_in_prefix = atoms
            .iter()
            .take(prefix_len)
            .enumerate()
            .filter(|(_, atom)| atom.potential == potential as i32)
            .map(|(index, _)| index)
            .next_back();

        if let Some(last_in_prefix) = last_in_prefix
            && last_in_prefix != point
        {
            let position = atoms[last_in_prefix].position;
            atoms[last_in_prefix].position = atoms[point].position;
            atoms[point].position = position;
            *representative_slot = Some(last_in_prefix);
        }
    }

    Ok(representative)
}

/// Port of FEFF `getang`: polar angles for the vector `positions[i] - positions[j]`.
///
/// Rust indices are zero-based. The returned values are `(theta, phi)` in
/// radians using FEFF's single-precision thresholds.
pub fn pair_polar_angles(
    positions: &[[f32; 3]],
    i: usize,
    j: usize,
) -> Result<(f32, f32), FmsError> {
    let left = checked_position(positions, i)?;
    let right = checked_position(positions, j)?;
    if i == j {
        return Ok((0.0, 0.0));
    }

    let x = left[0] - right[0];
    let y = left[1] - right[1];
    let z = left[2] - right[2];
    let r = (x * x + y * y + z * z).sqrt();

    const TINY: f32 = 1.0e-7;
    let phi = if x.abs() < TINY {
        if y.abs() < TINY {
            0.0
        } else if y > TINY {
            std::f32::consts::FRAC_PI_2
        } else {
            -std::f32::consts::FRAC_PI_2
        }
    } else {
        y.atan2(x)
    };

    let theta = if r <= TINY {
        0.0
    } else if z <= -r {
        std::f32::consts::PI
    } else if z < r {
        (z / r).acos()
    } else {
        0.0
    };

    Ok((theta, phi))
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
    const LXX: usize = 24;
    if lmax > LXX {
        return Err(FmsError::InvalidAngularLimit {
            name: "lmax",
            value: lmax,
            lx: LXX,
        });
    }
    if mmax > lmax {
        return Err(FmsError::InvalidAngularLimit {
            name: "mmax",
            value: mmax,
            lx: lmax,
        });
    }
    if !beta.is_finite() {
        return Err(FmsError::NonFiniteRotationAngle { name: "beta" });
    }
    if !phi.is_finite() {
        return Err(FmsError::NonFiniteRotationAngle { name: "phi" });
    }

    let mut drix = Array3::zeros((2 * lmax + 1, 2 * lmax + 1, lmax + 1).f());
    let mut dri0 = Array3::<f32>::zeros((LXX + 2, 2 * LXX + 2, 2 * LXX + 2).f());
    fill_rotxan_small_d(lmax, mmax, beta, &mut dri0);
    copy_rotxan_small_d(lmax, mmax, &dri0.view(), &mut drix)?;
    apply_rotxan_phase(lmax, phi, direction, &mut drix)?;
    Ok(drix)
}

/// Build FEFF `xrho` and `xclm` pair tables for an FMS cluster.
///
/// This ports the pair loop in `fmspack`: `rho = ck * |R_i - R_j|`, diagonal
/// polynomial entries are zero, and off-diagonal `xclm(m,l,j,i)` values are
/// copied from [`rehr_albers_polynomials`] in FEFF axis order.
pub fn fms_pair_tables(
    lmax: usize,
    wave_number: Complex32,
    atoms: &[FmsAtom],
) -> Result<FmsPairTables, FmsError> {
    if !(wave_number.re.is_finite() && wave_number.im.is_finite()) {
        return Err(FmsError::NonFiniteWaveNumber);
    }
    for (index, atom) in atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
    }

    let angular_len = lmax.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lmax",
        value: lmax,
        lx: lmax,
    })?;
    let atom_count = atoms.len();
    let mut rho = Array2::zeros((atom_count, atom_count).f());
    let mut polynomials = Array4::zeros((angular_len, angular_len, atom_count, atom_count).f());

    for i in 0..atom_count {
        for j in 0..=i {
            let distance = fms_atom_distance(atoms[i].position, atoms[j].position);
            let pair_rho = wave_number * distance;
            rho[(i, j)] = pair_rho;
            rho[(j, i)] = pair_rho;
            if i == j {
                continue;
            }

            let clm = rehr_albers_polynomials(lmax, angular_len, angular_len, pair_rho)?;
            for l in 0..=lmax {
                for m in 0..=lmax {
                    polynomials[(m, l, j, i)] = clm[(l, m)];
                    polynomials[(m, l, i, j)] = clm[(l, m)];
                }
            }
        }
    }

    Ok(FmsPairTables { rho, polynomials })
}

/// Port of the off-diagonal FEFF FMS free-propagator element.
///
/// This evaluates the `fmspack` Eq. 9 branch for different atoms with matching
/// spin: the Rehr-Albers angular sum, `exp(i*rho)/rho`, and the correlated
/// Debye damping factor. Same-atom or spin-mismatched states return zero, as in
/// FEFF's `g0` construction.
pub fn fms_free_propagator_element(
    input: FmsFreePropagatorInput<'_>,
) -> Result<Complex32, FmsError> {
    if input.first.atom == input.second.atom || input.first.spin != input.second.spin {
        return Ok(Complex32::new(0.0, 0.0));
    }
    if !(input.rho.re.is_finite() && input.rho.im.is_finite()) {
        return Err(FmsError::NonFiniteRho);
    }
    if input.rho == Complex32::new(0.0, 0.0) {
        return Err(FmsError::ZeroRho);
    }
    if !(input.wave_number.re.is_finite() && input.wave_number.im.is_finite()) {
        return Err(FmsError::NonFiniteWaveNumber);
    }
    if !input.mean_square_displacement.is_finite() {
        return Err(FmsError::NonFiniteMeanSquareDisplacement);
    }

    let l1 = input.first.angular_momentum;
    let l2 = input.second.angular_momentum;
    let l1_signed = isize::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
        name: "l1",
        value: l1,
        lx: l1,
    })?;

    let mut sum = Complex32::new(0.0, 0.0);
    for mu in -l1_signed..=l1_signed {
        let gllmz = rehr_albers_z_axis_propagator(
            mu.unsigned_abs(),
            input.first,
            input.second,
            input.xclm,
            input.xnlm,
        )?;
        let backward = rotation_table_value(
            input.backward_rotation,
            mu,
            input.first.magnetic,
            l1,
            "backward_rotation",
        )?;
        let forward = rotation_table_value(
            input.forward_rotation,
            input.second.magnetic,
            mu,
            l2,
            "forward_rotation",
        )?;
        sum += backward * gllmz * forward;
    }

    let prefactor =
        fms_free_propagator_prefactor(input.rho, input.wave_number, input.mean_square_displacement);
    Ok(prefactor * sum)
}

/// Build the FEFF off-diagonal FMS free-propagator matrix `g0`.
///
/// This ports the `fmspack` state-pair loop for the `G0` part only. Same-atom
/// and spin-mismatched pairs are left zero, and different-atom pairs outside
/// `direct_cutoff` are skipped before evaluating the Rehr-Albers angular sum.
/// The returned matrix is Fortran-order, matching FEFF/LAPACK storage.
pub fn fms_free_propagator_matrix(
    input: FmsFreePropagatorMatrixInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    if !input.direct_cutoff.is_finite() || input.direct_cutoff < 0.0 {
        return Err(FmsError::InvalidDirectCutoff);
    }
    if !(input.wave_number.re.is_finite() && input.wave_number.im.is_finite()) {
        return Err(FmsError::NonFiniteWaveNumber);
    }
    for (index, atom) in input.atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
    }

    let cutoff_squared = input.direct_cutoff * input.direct_cutoff;
    let mut matrix = Array2::zeros((input.states.len(), input.states.len()).f());
    for (row, &first) in input.states.iter().enumerate() {
        let atom1 = checked_atom_index(first.atom)?;
        ensure_atom_table_index(atom1, input.atoms.len())?;
        for (column, &second) in input.states.iter().enumerate() {
            let atom2 = checked_atom_index(second.atom)?;
            ensure_atom_table_index(atom2, input.atoms.len())?;
            if first.atom == second.atom || first.spin != second.spin {
                continue;
            }

            let distance_squared =
                fms_atom_distance_squared(input.atoms[atom1].position, input.atoms[atom2].position);
            if distance_squared > cutoff_squared {
                continue;
            }

            ensure_axis_len("xrho", "atom2", input.rho.shape()[0], atom2)?;
            ensure_axis_len("xrho", "atom1", input.rho.shape()[1], atom1)?;
            ensure_axis_len(
                "sigsqr",
                "atom2",
                input.mean_square_displacements.shape()[0],
                atom2,
            )?;
            ensure_axis_len(
                "sigsqr",
                "atom1",
                input.mean_square_displacements.shape()[1],
                atom1,
            )?;

            matrix[(row, column)] = fms_free_propagator_element(FmsFreePropagatorInput {
                first,
                second,
                rho: input.rho[(atom2, atom1)],
                wave_number: input.wave_number,
                mean_square_displacement: input.mean_square_displacements[(atom2, atom1)],
                xclm: input.xclm,
                xnlm: input.xnlm,
                backward_rotation: rotation_pair_view(
                    input.rotations,
                    FmsRotationDirection::Backward,
                    atom2,
                    atom1,
                )?,
                forward_rotation: rotation_pair_view(
                    input.rotations,
                    FmsRotationDirection::Forward,
                    atom2,
                    atom1,
                )?,
            })?;
        }
    }

    Ok(matrix)
}

/// Port of FEFF `xgllm`: z-axis Rehr-Albers propagator term.
///
/// `xclm` is indexed as `xclm(m, l, atom2 - 1, atom1 - 1)` and `xnlm` as
/// `xnlm(mu, l)`, matching FEFF's zero-based angular axes and one-based atom
/// labels. The state atoms in [`StateKet`] are therefore interpreted as FEFF
/// one-based atom indices.
pub fn rehr_albers_z_axis_propagator(
    mu: usize,
    first: StateKet,
    second: StateKet,
    xclm: ArrayView4<'_, Complex32>,
    xnlm: ArrayView2<'_, Real>,
) -> Result<Complex32, FmsError> {
    let iat1 = checked_atom_index(first.atom)?;
    let iat2 = checked_atom_index(second.atom)?;
    let l1 = first.angular_momentum;
    let l2 = second.angular_momentum;

    if mu > l1 {
        return Err(FmsError::MuOutOfRange {
            mu,
            angular_momentum: l1,
        });
    }

    ensure_axis_len("xclm", "m", xclm.shape()[0], l1.max(l2))?;
    ensure_axis_len("xclm", "l", xclm.shape()[1], l1.max(l2))?;
    ensure_axis_len("xclm", "atom2", xclm.shape()[2], iat2)?;
    ensure_axis_len("xclm", "atom1", xclm.shape()[3], iat1)?;
    ensure_axis_len("xnlm", "mu", xnlm.shape()[0], mu)?;
    ensure_axis_len("xnlm", "l", xnlm.shape()[1], l1.max(l2))?;

    if mu > l2 {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let norm_l1 = normalization_value(xnlm, mu, l1)?;
    let norm_l2 = normalization_value(xnlm, mu, l2)?;
    let angular_weight = angular_weight(l1)?;
    let sign = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };
    let numax = l1.min(l2 - mu);

    let sum = (0..=numax).try_fold(Complex32::new(0.0, 0.0), |sum, nu| {
        let mn = mu.checked_add(nu).ok_or(FmsError::InvalidAngularLimit {
            name: "mu",
            value: mu,
            lx: l2,
        })?;
        let gamtl = angular_weight * xclm[(nu, l1, iat2, iat1)] / norm_l1;
        let gam = xclm[(mn, l2, iat2, iat1)] * (sign * norm_l2);
        Ok(sum + gamtl * gam)
    })?;

    Ok(sum)
}

fn fill_rotxan_small_d(lmax: usize, mmax: usize, beta: f32, dri0: &mut Array3<f32>) {
    let lxp1 = lmax + 1;
    let mxp1 = mmax + 1;
    let ndm = lxp1 + mxp1 - 1;
    let xc = (beta / 2.0).cos();
    let xs = (beta / 2.0).sin();
    let s = beta.sin();

    dri0[(1, 1, 1)] = 1.0;
    if lxp1 < 2 {
        return;
    }
    dri0[(2, 1, 1)] = xc * xc;
    dri0[(2, 1, 2)] = s / 2.0_f32.sqrt();
    dri0[(2, 1, 3)] = xs * xs;
    dri0[(2, 2, 1)] = -dri0[(2, 1, 2)];
    dri0[(2, 2, 2)] = beta.cos();
    dri0[(2, 2, 3)] = dri0[(2, 1, 2)];
    dri0[(2, 3, 1)] = dri0[(2, 1, 3)];
    dri0[(2, 3, 2)] = -dri0[(2, 2, 3)];
    dri0[(2, 3, 3)] = dri0[(2, 1, 1)];

    for l in 3..=lxp1 {
        let mut ln = 2 * l - 1;
        let mut lm = 2 * l - 3;
        if ln > ndm {
            ln = ndm;
        }
        if lm > ndm {
            lm = ndm;
        }
        for n in 1..=ln {
            for m in 1..=lm {
                let l_i = l as i32;
                let n_i = n as i32;
                let m_i = m as i32;
                let t1 = ((2 * l_i - 1 - n_i) * (2 * l_i - 2 - n_i)) as f32;
                let t = ((2 * l_i - 1 - m_i) * (2 * l_i - 2 - m_i)) as f32;
                let f1 = (t1 / t).sqrt();
                let f2 = (((2 * l_i - 1 - n_i) * (n_i - 1)) as f32 / t).sqrt();
                let t3 = ((n_i - 2) * (n_i - 1)) as f32;
                let f3 = (t3 / t).sqrt();
                let mut dlnm = f1 * xc * xc * dri0[(l - 1, n, m)];
                if n > 1 {
                    dlnm -= f2 * s * dri0[(l - 1, n - 1, m)];
                }
                if n > 2 {
                    dlnm += f3 * xs * xs * dri0[(l - 1, n - 2, m)];
                }
                dri0[(l, n, m)] = dlnm;
                if n > (2 * l - 3) {
                    dri0[(l, m, n)] = alternating_f32(n - m) * dlnm;
                }
            }

            if n > (2 * l - 3) {
                dri0[(l, 2 * l - 2, 2 * l - 2)] = dri0[(l, 2, 2)];
                dri0[(l, 2 * l - 1, 2 * l - 2)] = -dri0[(l, 1, 2)];
                dri0[(l, 2 * l - 2, 2 * l - 1)] = -dri0[(l, 2, 1)];
                dri0[(l, 2 * l - 1, 2 * l - 1)] = dri0[(l, 1, 1)];
            }
        }
    }
}

fn copy_rotxan_small_d(
    lmax: usize,
    mmax: usize,
    dri0: &ArrayView3<'_, f32>,
    drix: &mut Array3<Complex32>,
) -> Result<(), FmsError> {
    for il in 1..=lmax + 1 {
        let mmx = (il - 1).min(mmax);
        for m1 in -(mmx as isize)..=(mmx as isize) {
            for m2 in -(mmx as isize)..=(mmx as isize) {
                let row = signed_magnetic_index(m2, lmax)?;
                let column = signed_magnetic_index(m1, lmax)?;
                drix[(row, column, il - 1)] = Complex32::new(
                    dri0[(il, (m1 + il as isize) as usize, (m2 + il as isize) as usize)],
                    0.0,
                );
            }
        }
    }
    Ok(())
}

fn apply_rotxan_phase(
    lmax: usize,
    phi: f32,
    direction: FmsRotationDirection,
    drix: &mut Array3<Complex32>,
) -> Result<(), FmsError> {
    for il in 0..=lmax {
        for m1 in -(il as isize)..=(il as isize) {
            let angle = match direction {
                FmsRotationDirection::Forward => m1 as f32 * (phi - std::f32::consts::PI),
                FmsRotationDirection::Backward => -m1 as f32 * (phi - std::f32::consts::PI),
            };
            let phase = Complex32::new(0.0, angle).exp();
            for m2 in -(il as isize)..=(il as isize) {
                match direction {
                    FmsRotationDirection::Forward => {
                        let row = signed_magnetic_index(m1, lmax)?;
                        let column = signed_magnetic_index(m2, lmax)?;
                        drix[(row, column, il)] *= phase;
                    }
                    FmsRotationDirection::Backward => {
                        let row = signed_magnetic_index(m2, lmax)?;
                        let column = signed_magnetic_index(m1, lmax)?;
                        drix[(row, column, il)] *= phase;
                    }
                }
            }
        }
    }
    Ok(())
}

fn signed_magnetic_index(magnetic: isize, lmax: usize) -> Result<usize, FmsError> {
    let lmax_isize = isize::try_from(lmax).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lmax",
        value: lmax,
        lx: lmax,
    })?;
    let index = magnetic + lmax_isize;
    usize::try_from(index).map_err(|_| FmsError::InvalidAngularLimit {
        name: "magnetic",
        value: magnetic.unsigned_abs(),
        lx: lmax,
    })
}

fn alternating_f32(value: usize) -> f32 {
    if value.is_multiple_of(2) { 1.0 } else { -1.0 }
}

fn fms_atom_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    fms_atom_distance_squared(left, right).sqrt()
}

fn fms_atom_distance_squared(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

fn fms_free_propagator_prefactor(
    rho: Complex32,
    wave_number: Complex32,
    mean_square_displacement: f32,
) -> Complex32 {
    const BOHR: f32 = 0.529_177_25;
    let phase = (Complex32::new(0.0, 1.0) * rho).exp() / rho;
    let damping_factor = Complex32::new(-mean_square_displacement / (BOHR * BOHR), 0.0);
    let damping = (damping_factor * wave_number * wave_number).exp();
    phase * damping
}

fn rotation_table_value(
    table: ArrayView3<'_, Complex32>,
    m2: isize,
    m1: isize,
    angular_momentum: usize,
    table_name: &'static str,
) -> Result<Complex32, FmsError> {
    let shape = table.shape();
    if shape[0] == 0 || shape[0] != shape[1] || shape[0].is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: table_name,
            value: shape[0],
            lx: shape[0],
        });
    }
    ensure_axis_len(table_name, "l", shape[2], angular_momentum)?;
    let lmax = (shape[0] - 1) / 2;
    let row = signed_magnetic_index(m2, lmax)?;
    let column = signed_magnetic_index(m1, lmax)?;
    ensure_axis_len(table_name, "m2", shape[0], row)?;
    ensure_axis_len(table_name, "m1", shape[1], column)?;
    Ok(table[(row, column, angular_momentum)])
}

fn rotation_pair_view<'a>(
    rotations: ArrayView6<'a, Complex32>,
    direction: FmsRotationDirection,
    atom2: usize,
    atom1: usize,
) -> Result<ArrayView3<'a, Complex32>, FmsError> {
    let shape = rotations.shape();
    if shape[0] == 0 || shape[0] != shape[1] || shape[0].is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: "rotations",
            value: shape[0],
            lx: shape[0],
        });
    }
    ensure_axis_len("rotations", "k", shape[3], 1)?;
    ensure_axis_len("rotations", "atom2", shape[4], atom2)?;
    ensure_axis_len("rotations", "atom1", shape[5], atom1)?;

    let branch = match direction {
        FmsRotationDirection::Forward => 0,
        FmsRotationDirection::Backward => 1,
    };
    Ok(rotations
        .index_axis_move(Axis(5), atom1)
        .index_axis_move(Axis(4), atom2)
        .index_axis_move(Axis(3), branch))
}

fn sort_radius_key(index: usize, atom: FmsAtom) -> Result<f64, FmsError> {
    ensure_finite_position(index, atom.position)?;
    Ok(f64::from(atom.position[0]) * f64::from(atom.position[0])
        + f64::from(atom.position[1]) * f64::from(atom.position[1])
        + f64::from(atom.position[2]) * f64::from(atom.position[2])
        + (index as f64 + 1.0) * 1.0e-6)
}

fn checked_potential(potential: i32, max_potential: usize) -> Result<usize, FmsError> {
    let Ok(potential_index) = usize::try_from(potential) else {
        return Err(FmsError::PotentialOutOfRange {
            potential,
            max_potential,
        });
    };
    if potential_index <= max_potential {
        Ok(potential_index)
    } else {
        Err(FmsError::PotentialOutOfRange {
            potential,
            max_potential,
        })
    }
}

fn checked_position(positions: &[[f32; 3]], index: usize) -> Result<[f32; 3], FmsError> {
    let position = positions
        .get(index)
        .copied()
        .ok_or(FmsError::AtomIndexOutOfRange {
            index,
            len: positions.len(),
        })?;
    ensure_finite_position(index, position)?;
    Ok(position)
}

fn ensure_finite_position(atom: usize, position: [f32; 3]) -> Result<(), FmsError> {
    for (axis, value) in position.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(FmsError::NonFiniteCoordinate { atom, axis });
        }
    }
    Ok(())
}

fn checked_atom_index(atom: usize) -> Result<usize, FmsError> {
    atom.checked_sub(1)
        .ok_or(FmsError::InvalidStateAtom { atom })
}

fn ensure_atom_table_index(index: usize, len: usize) -> Result<(), FmsError> {
    if index < len {
        Ok(())
    } else {
        Err(FmsError::AtomIndexOutOfRange { index, len })
    }
}

fn ensure_axis_len(
    table: &'static str,
    axis: &'static str,
    len: usize,
    index: usize,
) -> Result<(), FmsError> {
    if index < len {
        Ok(())
    } else {
        Err(FmsError::TableIndexOutOfRange { table, axis, index })
    }
}

fn normalization_value(
    xnlm: ArrayView2<'_, Real>,
    mu: usize,
    angular_momentum: usize,
) -> Result<f32, FmsError> {
    let value = xnlm[(mu, angular_momentum)] as f32;
    if value.is_finite() && value != 0.0 {
        Ok(value)
    } else {
        Err(FmsError::InvalidNormalization {
            mu,
            angular_momentum,
        })
    }
}

fn angular_weight(angular_momentum: usize) -> Result<Complex32, FmsError> {
    let value = angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "angular_momentum",
            value: angular_momentum,
            lx: angular_momentum,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}

fn odd_factor(index: usize, lx: usize) -> Result<Complex32, FmsError> {
    let value = index
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}

#[cfg(test)]
mod tests {
    use super::{
        FmsAtom, FmsFreePropagatorInput, FmsFreePropagatorMatrixInput, FmsRotationDirection,
        fms_free_propagator_element, fms_free_propagator_matrix, fms_pair_tables,
        fms_rotation_matrix, pair_polar_angles, sort_atoms_by_radius, sort_representative_atoms,
    };
    use super::{FmsError, rehr_albers_polynomials, rehr_albers_z_axis_propagator};
    use crate::{Real, angular::legendre_normalization_table, state::StateKet};
    use ndarray::{
        Array2, Array3, Array4, Array6, ArrayView2, ArrayView3, ArrayView4, ShapeBuilder,
    };
    use num_complex::Complex32;
    use std::error::Error;

    #[test]
    fn xclmz_matches_feff_reference_lx3() -> Result<(), FmsError> {
        let table = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;

        assert_eq!(table.shape(), &[5, 9]);
        assert_eq!(table.strides(), &[1, 5]);
        assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(table[(1, 0)], Complex32::new(1.2322206, 0.725_689_4));
        assert_complex32_close(table[(3, 0)], Complex32::new(-10.012509, 5.438_266));
        assert_complex32_close(table[(2, 1)], Complex32::new(-2.1395304, 4.1993084));
        assert_complex32_close(table[(3, 2)], Complex32::new(-23.036537, -6.8588142));
        assert_complex32_close(table[(4, 3)], Complex32::new(8.928_719, -161.62775));
        assert_complex32_close(
            matrix_sum(table.view()),
            Complex32::new(-58.983994, -154.61885),
        );
        assert_eq!(nonzero_count(table.view()), 11);
        Ok(())
    }

    #[test]
    fn xclmz_matches_feff_reference_with_limited_m() -> Result<(), FmsError> {
        let table = rehr_albers_polynomials(4, 3, 2, Complex32::new(-0.8, 1.1))?;

        assert_eq!(table.shape(), &[6, 11]);
        assert_eq!(table.strides(), &[1, 6]);
        assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(table[(1, 0)], Complex32::new(1.5945946, -0.432_432_4));
        assert_complex32_close(table[(2, 0)], Complex32::new(3.2834187, -2.840029));
        assert_complex32_close(table[(1, 1)], Complex32::new(0.5945946, -0.432_432_4));
        assert_complex32_close(table[(2, 1)], Complex32::new(2.7830534, -4.382761));
        assert_complex32_close(
            matrix_sum(table.view()),
            Complex32::new(9.255661, -8.087655),
        );
        assert_eq!(nonzero_count(table.view()), 5);
        Ok(())
    }

    #[test]
    fn xclmz_rejects_invalid_inputs() {
        assert_eq!(
            rehr_albers_polynomials(3, 0, 1, Complex32::new(1.0, 0.0)),
            Err(FmsError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
                lx: 3,
            })
        );
        assert_eq!(
            rehr_albers_polynomials(3, 5, 1, Complex32::new(1.0, 0.0)),
            Err(FmsError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 5,
                lx: 3,
            })
        );
        assert_eq!(
            rehr_albers_polynomials(3, 1, 1, Complex32::new(0.0, 0.0)),
            Err(FmsError::ZeroRho)
        );
        assert_eq!(
            rehr_albers_polynomials(3, 1, 1, Complex32::new(f32::NAN, 0.0)),
            Err(FmsError::NonFiniteRho)
        );
    }

    #[test]
    fn rotxan_matches_feff_reference_forward_and_backward() -> Result<(), FmsError> {
        let forward = fms_rotation_matrix(3, 3, 0.7, 1.1, FmsRotationDirection::Forward)?;
        let backward = fms_rotation_matrix(3, 3, 0.7, 1.1, FmsRotationDirection::Backward)?;

        assert_eq!(forward.shape(), &[7, 7, 4]);
        assert_eq!(forward.strides(), &[1, 7, 49]);
        assert_complex32_close(
            rotation_sum(forward.view()),
            Complex32::new(1.159_583_6, 0.288_981_8),
        );
        assert_complex32_close(
            rotation_sum(backward.view()),
            Complex32::new(1.159_583_1, 0.288_981_74),
        );
        assert_eq!(rotation_nonzero_count(forward.view()), 84);
        assert_eq!(rotation_nonzero_count(backward.view()), 84);

        assert_complex32_close(rotation_value(&forward, 0, 0, 0), Complex32::new(1.0, 0.0));
        assert_complex32_close(
            rotation_value(&forward, 1, -1, 1),
            Complex32::new(-0.053_333_33, -0.104_787_19),
        );
        assert_complex32_close(
            rotation_value(&forward, -1, 1, 1),
            Complex32::new(-0.053_333_33, 0.104_787_19),
        );
        assert_complex32_close(
            rotation_value(&forward, 2, -1, 2),
            Complex32::new(-0.044_576_85, 0.061_240_695),
        );
        assert_complex32_close(
            rotation_value(&forward, -2, 1, 3),
            Complex32::new(0.116_102_73, 0.159_504_58),
        );
        assert_complex32_close(
            rotation_value(&forward, 3, 3, 3),
            Complex32::new(0.678_509_35, 0.108_389_09),
        );

        assert_complex32_close(
            rotation_value(&backward, 2, -1, 2),
            Complex32::new(-0.034_358_274, -0.067_505_76),
        );
        assert_complex32_close(
            rotation_value(&backward, -2, 1, 3),
            Complex32::new(0.089_487_91, -0.175_822_26),
        );
        assert_complex32_close(
            rotation_value(&backward, 3, 3, 3),
            Complex32::new(0.678_509_35, -0.108_389_09),
        );
        Ok(())
    }

    #[test]
    fn rotxan_rejects_invalid_inputs() {
        assert_eq!(
            fms_rotation_matrix(25, 1, 0.0, 0.0, FmsRotationDirection::Forward),
            Err(FmsError::InvalidAngularLimit {
                name: "lmax",
                value: 25,
                lx: 24,
            })
        );
        assert_eq!(
            fms_rotation_matrix(3, 4, 0.0, 0.0, FmsRotationDirection::Forward),
            Err(FmsError::InvalidAngularLimit {
                name: "mmax",
                value: 4,
                lx: 3,
            })
        );
        assert_eq!(
            fms_rotation_matrix(3, 3, f32::NAN, 0.0, FmsRotationDirection::Forward),
            Err(FmsError::NonFiniteRotationAngle { name: "beta" })
        );
    }

    #[test]
    fn fms_pair_tables_match_feff_reference() -> Result<(), FmsError> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
            FmsAtom {
                position: [-1.0, 0.0, 0.5],
                potential: 2,
            },
        ];

        let tables = fms_pair_tables(2, Complex32::new(1.2, 0.3), &atoms)?;

        assert_eq!(tables.rho.shape(), &[3, 3]);
        assert_eq!(tables.rho.strides(), &[1, 3]);
        assert_eq!(tables.polynomials.shape(), &[3, 3, 3, 3]);
        assert_eq!(tables.polynomials.strides(), &[1, 3, 9, 27]);
        assert_complex32_close(
            tables.rho[(0, 1)],
            Complex32::new(3.600_000_1, 0.900_000_04),
        );
        assert_complex32_close(tables.rho[(0, 2)], Complex32::new(1.341_640_8, 0.335_410_2));
        assert_complex32_close(tables.rho[(1, 2)], Complex32::new(3.841_874_8, 0.960_468_7));
        assert_complex32_close(
            pair_table_sum(tables.polynomials.view()),
            Complex32::new(8.870_853, 26.772_633),
        );
        assert_eq!(pair_table_nonzero_count(tables.polynomials.view()), 36);
        assert_complex32_close(tables.polynomials[(0, 0, 1, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(
            tables.polynomials[(1, 1, 1, 0)],
            Complex32::new(0.065_359_47, 0.261_437_9),
        );
        assert_complex32_close(
            tables.polynomials[(2, 2, 2, 0)],
            Complex32::new(-1.384_083, 0.738_177_6),
        );
        assert_complex32_close(
            tables.polynomials[(1, 2, 2, 1)],
            Complex32::new(-0.153_847_35, 0.914_978_6),
        );
        assert_complex32_close(tables.polynomials[(1, 1, 0, 0)], Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn fms_pair_tables_reject_invalid_inputs() {
        assert_eq!(
            fms_pair_tables(
                1,
                Complex32::new(f32::NAN, 0.0),
                &[FmsAtom {
                    position: [0.0, 0.0, 0.0],
                    potential: 0,
                }],
            ),
            Err(FmsError::NonFiniteWaveNumber)
        );
        assert_eq!(
            fms_pair_tables(
                1,
                Complex32::new(1.0, 0.0),
                &[FmsAtom {
                    position: [0.0, f32::INFINITY, 0.0],
                    potential: 0,
                }],
            ),
            Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 1 })
        );
    }

    #[test]
    fn fms_free_propagator_matches_feff_reference() -> Result<(), Box<dyn Error>> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
        ];
        let wave_number = Complex32::new(1.2, 0.3);
        let tables = fms_pair_tables(2, wave_number, &atoms)?;
        let xnlm = legendre_normalization_table(2)?;
        let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
        let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 2,
            magnetic: -1,
            spin: 1,
        };

        let value = fms_free_propagator_element(FmsFreePropagatorInput {
            first,
            second,
            rho: tables.rho[(0, 1)],
            wave_number,
            mean_square_displacement: 0.05,
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            backward_rotation: backward.view(),
            forward_rotation: forward.view(),
        })?;

        assert_complex32_close(value, Complex32::new(-0.103_387_31, 0.105_749_39));
        Ok(())
    }

    #[test]
    fn fms_free_propagator_returns_zero_for_excluded_state_pairs() -> Result<(), Box<dyn Error>> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
        ];
        let wave_number = Complex32::new(1.2, 0.3);
        let tables = fms_pair_tables(2, wave_number, &atoms)?;
        let xnlm = legendre_normalization_table(2)?;
        let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
        let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        };

        let same_atom = fms_free_propagator_element(FmsFreePropagatorInput {
            second: StateKet { atom: 1, ..second },
            first,
            rho: Complex32::new(0.0, 0.0),
            wave_number,
            mean_square_displacement: 0.05,
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            backward_rotation: backward.view(),
            forward_rotation: forward.view(),
        })?;
        let spin_mismatch = fms_free_propagator_element(FmsFreePropagatorInput {
            second: StateKet { spin: 2, ..second },
            first,
            rho: tables.rho[(0, 1)],
            wave_number,
            mean_square_displacement: 0.05,
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            backward_rotation: backward.view(),
            forward_rotation: forward.view(),
        })?;

        assert_complex32_close(same_atom, Complex32::new(0.0, 0.0));
        assert_complex32_close(spin_mismatch, Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn fms_free_propagator_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
        ];
        let wave_number = Complex32::new(1.2, 0.3);
        let tables = fms_pair_tables(2, wave_number, &atoms)?;
        let xnlm = legendre_normalization_table(2)?;
        let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
        let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        };
        let input = |rho, wave_number, mean_square_displacement| FmsFreePropagatorInput {
            first,
            second,
            rho,
            wave_number,
            mean_square_displacement,
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            backward_rotation: backward.view(),
            forward_rotation: forward.view(),
        };

        assert_eq!(
            fms_free_propagator_element(input(tables.rho[(0, 1)], wave_number, f32::INFINITY,)),
            Err(FmsError::NonFiniteMeanSquareDisplacement)
        );
        assert_eq!(
            fms_free_propagator_element(input(Complex32::new(0.0, 0.0), wave_number, 0.05)),
            Err(FmsError::ZeroRho)
        );
        assert_eq!(
            fms_free_propagator_element(input(
                tables.rho[(0, 1)],
                Complex32::new(f32::NAN, 0.0),
                0.05,
            )),
            Err(FmsError::NonFiniteWaveNumber)
        );
        Ok(())
    }

    #[test]
    fn fms_free_propagator_matrix_matches_feff_reference_element() -> Result<(), Box<dyn Error>> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
        ];
        let wave_number = Complex32::new(1.2, 0.3);
        let tables = fms_pair_tables(2, wave_number, &atoms)?;
        let xnlm = legendre_normalization_table(2)?;
        let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
        let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
        let mut rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
        copy_rotation_pair(
            &mut rotations,
            1,
            0,
            FmsRotationDirection::Backward,
            &backward,
        );
        copy_rotation_pair(
            &mut rotations,
            1,
            0,
            FmsRotationDirection::Forward,
            &forward,
        );
        let mut sigsqr = Array2::zeros((2, 2).f());
        sigsqr[(1, 0)] = 0.05;
        sigsqr[(0, 1)] = 0.05;
        let states = [
            StateKet {
                atom: 1,
                angular_momentum: 2,
                magnetic: 1,
                spin: 1,
            },
            StateKet {
                atom: 2,
                angular_momentum: 2,
                magnetic: -1,
                spin: 1,
            },
        ];

        let matrix = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
            states: &states,
            atoms: &atoms,
            direct_cutoff: 3.0,
            rho: tables.rho.view(),
            wave_number,
            mean_square_displacements: sigsqr.view(),
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            rotations: rotations.view(),
        })?;

        assert_eq!(matrix.shape(), &[2, 2]);
        assert_eq!(matrix.strides(), &[1, 2]);
        assert_complex32_close(matrix[(0, 0)], Complex32::new(0.0, 0.0));
        assert_complex32_close(matrix[(0, 1)], Complex32::new(-0.103_387_31, 0.105_749_39));
        assert_complex32_close(matrix[(1, 0)], Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn fms_free_propagator_matrix_applies_direct_cutoff() -> Result<(), Box<dyn Error>> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
        ];
        let wave_number = Complex32::new(1.2, 0.3);
        let tables = fms_pair_tables(2, wave_number, &atoms)?;
        let xnlm = legendre_normalization_table(2)?;
        let rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
        let sigsqr = Array2::zeros((2, 2).f());
        let states = [
            StateKet {
                atom: 1,
                angular_momentum: 2,
                magnetic: 1,
                spin: 1,
            },
            StateKet {
                atom: 2,
                angular_momentum: 2,
                magnetic: -1,
                spin: 1,
            },
        ];

        let matrix = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
            states: &states,
            atoms: &atoms,
            direct_cutoff: 2.99,
            rho: tables.rho.view(),
            wave_number,
            mean_square_displacements: sigsqr.view(),
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            rotations: rotations.view(),
        })?;

        assert_complex32_close(matrix[(0, 1)], Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn fms_free_propagator_matrix_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
        let atoms = [
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 2.0, 2.0],
                potential: 1,
            },
        ];
        let wave_number = Complex32::new(1.2, 0.3);
        let tables = fms_pair_tables(2, wave_number, &atoms)?;
        let xnlm = legendre_normalization_table(2)?;
        let rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
        let sigsqr = Array2::zeros((2, 2).f());
        let states = [
            StateKet {
                atom: 1,
                angular_momentum: 1,
                magnetic: 0,
                spin: 1,
            },
            StateKet {
                atom: 2,
                angular_momentum: 1,
                magnetic: 0,
                spin: 1,
            },
        ];

        let result = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
            states: &states,
            atoms: &atoms,
            direct_cutoff: f32::NAN,
            rho: tables.rho.view(),
            wave_number,
            mean_square_displacements: sigsqr.view(),
            xclm: tables.polynomials.view(),
            xnlm: xnlm.view(),
            rotations: rotations.view(),
        });

        assert!(matches!(result, Err(FmsError::InvalidDirectCutoff)));
        Ok(())
    }

    #[test]
    fn atheap_matches_feff_reference_sort_order() -> Result<(), FmsError> {
        let mut atoms = vec![
            FmsAtom {
                position: [2.0, 0.0, 0.0],
                potential: 1,
            },
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [-1.0, 0.0, 0.0],
                potential: 2,
            },
            FmsAtom {
                position: [1.0, 0.0, 0.0],
                potential: 3,
            },
            FmsAtom {
                position: [0.0, 2.0, 0.0],
                potential: 4,
            },
        ];

        let keys = sort_atoms_by_radius(&mut atoms)?;

        assert_eq!(
            atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
            vec![0, 2, 3, 1, 4]
        );
        assert_eq!(atoms[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(atoms[1].position, [-1.0, 0.0, 0.0]);
        assert_close_f64(keys[0], 2.0e-6);
        assert_close_f64(keys[1], 1.000_003);
        assert_close_f64(keys[2], 1.000_004);
        assert_close_f64(keys[3], 4.000_001);
        assert_close_f64(keys[4], 4.000_005);
        Ok(())
    }

    #[test]
    fn getang_matches_feff_reference_angles() -> Result<(), FmsError> {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [0.0, 5.0e-8, 2.0e-7],
            [0.0, 2.0e-7, 0.0],
        ];

        let (theta, phi) = pair_polar_angles(&positions, 1, 0)?;
        assert_close_f32(theta, 0.841_068_6);
        assert_close_f32(phi, 1.107_148_8);

        let (theta, phi) = pair_polar_angles(&positions, 3, 2)?;
        assert_close_f32(theta, 2.498_091_5);
        assert_close_f32(phi, 1.570_796_4);

        assert_eq!(pair_polar_angles(&positions, 0, 0)?, (0.0, 0.0));
        Ok(())
    }

    #[test]
    fn sortat_matches_feff_reference_representative_order() -> Result<(), FmsError> {
        let mut atoms = vec![
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 0.0, 0.0],
                potential: 2,
            },
            FmsAtom {
                position: [2.0, 0.0, 0.0],
                potential: 1,
            },
            FmsAtom {
                position: [3.0, 0.0, 0.0],
                potential: 3,
            },
            FmsAtom {
                position: [4.0, 0.0, 0.0],
                potential: 2,
            },
            FmsAtom {
                position: [5.0, 0.0, 0.0],
                potential: 1,
            },
        ];

        let representatives = sort_representative_atoms(0, 3, &mut atoms)?;

        assert_eq!(representatives, vec![Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(
            atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 2, 1]
        );
        assert_eq!(atoms[1].position, [2.0, 0.0, 0.0]);
        assert_eq!(atoms[2].position, [1.0, 0.0, 0.0]);
        assert_eq!(atoms[3].position, [3.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn fms_cluster_helpers_reject_invalid_inputs() {
        let positions = [[0.0, 0.0, 0.0]];
        assert_eq!(
            pair_polar_angles(&positions, 1, 0),
            Err(FmsError::AtomIndexOutOfRange { index: 1, len: 1 })
        );

        let mut atoms = [FmsAtom {
            position: [f32::NAN, 0.0, 0.0],
            potential: 0,
        }];
        assert_eq!(
            sort_atoms_by_radius(&mut atoms),
            Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 0 })
        );

        let mut atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        assert_eq!(
            sort_representative_atoms(0, 1, &mut atoms),
            Err(FmsError::CentralAtomMismatch {
                expected: 0,
                actual: 1,
            })
        );
        assert_eq!(
            sort_representative_atoms(-1, 1, &mut atoms),
            Err(FmsError::PotentialOutOfRange {
                potential: -1,
                max_potential: 1,
            })
        );
    }

    #[test]
    fn xgllm_matches_feff_reference() -> Result<(), Box<dyn Error>> {
        let (xclm, xnlm) = reference_xgllm_tables()?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 3,
            magnetic: 0,
            spin: 1,
        };

        assert_complex32_close(
            rehr_albers_z_axis_propagator(0, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(415.546_9, -1006.2809),
        );
        assert_complex32_close(
            rehr_albers_z_axis_propagator(1, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(-307.497_3, 722.469_5),
        );
        assert_complex32_close(
            rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(115.08963, -235.94589),
        );
        Ok(())
    }

    #[test]
    fn xgllm_matches_feff_empty_sum_case() -> Result<(), Box<dyn Error>> {
        let (xclm, xnlm) = reference_xgllm_tables()?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 3,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        };

        assert_complex32_close(
            rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(0.0, 0.0),
        );
        Ok(())
    }

    #[test]
    fn xgllm_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
        let (xclm, xnlm) = reference_xgllm_tables()?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 3,
            magnetic: 0,
            spin: 1,
        };

        assert_eq!(
            rehr_albers_z_axis_propagator(3, first, second, xclm.view(), xnlm.view()),
            Err(FmsError::MuOutOfRange {
                mu: 3,
                angular_momentum: 2,
            })
        );
        assert_eq!(
            rehr_albers_z_axis_propagator(
                0,
                StateKet { atom: 0, ..first },
                second,
                xclm.view(),
                xnlm.view(),
            ),
            Err(FmsError::InvalidStateAtom { atom: 0 })
        );

        let mut bad_xnlm = xnlm.clone();
        bad_xnlm[(0, 2)] = 0.0;
        assert_eq!(
            rehr_albers_z_axis_propagator(0, first, second, xclm.view(), bad_xnlm.view()),
            Err(FmsError::InvalidNormalization {
                mu: 0,
                angular_momentum: 2,
            })
        );
        Ok(())
    }

    fn reference_xgllm_tables() -> Result<(Array4<Complex32>, Array2<Real>), Box<dyn Error>> {
        let clm = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;
        let mut xclm = Array4::zeros((4, 4, 2, 2).f());
        for l in 0..=3 {
            for m in 0..=3 {
                xclm[(m, l, 1, 0)] = clm[(l, m)];
                xclm[(m, l, 0, 1)] = clm[(l, m)];
            }
        }
        Ok((xclm, legendre_normalization_table(3)?))
    }

    fn matrix_sum(matrix: ArrayView2<'_, Complex32>) -> Complex32 {
        matrix
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn nonzero_count(matrix: ArrayView2<'_, Complex32>) -> usize {
        matrix
            .iter()
            .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
            .count()
    }

    fn rotation_sum(matrix: ArrayView3<'_, Complex32>) -> Complex32 {
        matrix
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn rotation_nonzero_count(matrix: ArrayView3<'_, Complex32>) -> usize {
        matrix
            .iter()
            .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
            .count()
    }

    fn pair_table_sum(table: ArrayView4<'_, Complex32>) -> Complex32 {
        table
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn pair_table_nonzero_count(table: ArrayView4<'_, Complex32>) -> usize {
        table
            .iter()
            .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
            .count()
    }

    fn rotation_value(
        matrix: &Array3<Complex32>,
        m2: isize,
        m1: isize,
        angular_momentum: usize,
    ) -> Complex32 {
        let offset = 3_isize;
        matrix[(
            (m2 + offset) as usize,
            (m1 + offset) as usize,
            angular_momentum,
        )]
    }

    fn copy_rotation_pair(
        rotations: &mut Array6<Complex32>,
        atom2: usize,
        atom1: usize,
        direction: FmsRotationDirection,
        table: &Array3<Complex32>,
    ) {
        let branch = match direction {
            FmsRotationDirection::Forward => 0,
            FmsRotationDirection::Backward => 1,
        };
        for l in 0..table.shape()[2] {
            for m1 in 0..table.shape()[1] {
                for m2 in 0..table.shape()[0] {
                    rotations[(m2, m1, l, branch, atom2, atom1)] = table[(m2, m1, l)];
                }
            }
        }
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32) {
        assert!(
            (actual - expected).norm() < 2.0e-4,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_close_f32(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 2.0e-6,
            "actual={actual} expected={expected}"
        );
    }

    fn assert_close_f64(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
