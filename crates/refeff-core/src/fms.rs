//! Full multiple-scattering helpers.
//!
//! FEFF's FMS routines use Rehr-Albers polynomial tables when building
//! multiple-scattering propagators. The helpers here keep the legacy table
//! layout explicit while returning Rust-owned `ndarray` storage.

use ndarray::{
    Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, ArrayView6, Axis, ShapeBuilder,
};
use num_complex::Complex32;
use refeff_linalg::{LinalgError, complex32_lu_factor, complex32_lu_solve};
use thiserror::Error;

use crate::{Real, angular::SpinOrbitCouplingTables, state::StateKet};

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

/// Inputs for one FEFF FMS single-site T-matrix element.
#[derive(Debug, Clone)]
pub struct FmsTMatrixInput<'a> {
    /// Bra-side FEFF state.
    pub first: StateKet,
    /// Ket-side FEFF state.
    pub second: StateKet,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// Zero-based potential index for the shared atom.
    pub potential: usize,
    /// FEFF `xphase(spin,l,potential)` table with signed `l` centered.
    pub phase_shifts: ArrayView3<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for building FEFF's compact `tmatrx(spin_band,state)` table.
#[derive(Debug, Clone)]
pub struct FmsTMatrixTableInput<'a> {
    /// FEFF state kets in compact-table column order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase(spin,l,potential)` table with signed `l` centered.
    pub phase_shifts: ArrayView3<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for FEFF iterative FMS matrix assembly.
#[derive(Debug, Clone)]
pub struct FmsIterativeSystemInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `toler2` cutoff applied to `abs(g0)` terms.
    pub zero_tolerance: f32,
}

/// Inputs for FEFF's BiCGStab FMS branch, `ggbi`.
#[derive(Debug, Clone)]
pub struct FmsBiCgStabInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `1 - T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's BiCGStab FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsBiCgStabResult {
    /// The assembled `1 - T*G0` matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's recursion-method FMS branch, `ggrm`.
#[derive(Debug, Clone)]
pub struct FmsRecursionInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `1 - T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's recursion-method FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsRecursionResult {
    /// The assembled `1 - T*G0` matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's Graves-Morris/Salam FMS branch, `gggm`.
#[derive(Debug, Clone)]
pub struct FmsGravesMorrisInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's Graves-Morris/Salam FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsGravesMorrisResult {
    /// The assembled FEFF `T*G0` work matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's TFQMR FMS branch, `ggtf`.
#[derive(Debug, Clone)]
pub struct FmsTfqmrInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `1 - T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's TFQMR FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsTfqmrResult {
    /// The assembled `1 - T*G0` matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's LU FMS branch, `gglu`.
#[derive(Debug, Clone)]
pub struct FmsLuInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
}

/// Result from FEFF's LU FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsLuResult {
    /// The assembled `1 - G0*T` matrix before LU factorization.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
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
    /// Iterative FMS tolerances must be finite and nonnegative.
    #[error("{name} tolerance must be finite and nonnegative, got {value}")]
    InvalidTolerance { name: &'static str, value: f32 },
    /// FEFF FMS supports one or two spin channels.
    #[error("invalid spin channel count {value}; expected 1 or 2")]
    InvalidSpinChannelCount { value: usize },
    /// FEFF state spins are one-based and must fit the active spin channels.
    #[error("state spin {spin} is outside 1..={spin_channels}")]
    InvalidStateSpin { spin: usize, spin_channels: usize },
    /// FEFF phase shifts used by the FMS T-matrix must be finite.
    #[error("xphase(spin={spin}, l={angular_momentum}, potential={potential}) must be finite")]
    NonFinitePhaseShift {
        spin: usize,
        angular_momentum: isize,
        potential: usize,
    },
    /// A requested potential has no representative state offset.
    #[error("missing representative state offset for potential {potential}")]
    MissingRepresentativePotential { potential: usize },
    /// FEFF-compatible LU factorization or solve failed.
    #[error("linear algebra failure: {0}")]
    LinearAlgebra(#[from] LinalgError),
    /// Iterative FMS solver encountered a zero denominator.
    #[error("{solver} solver breakdown at {step}")]
    IterativeSolverBreakdown {
        solver: &'static str,
        step: &'static str,
    },
    /// Iterative FMS solver did not converge before the Rust safety limit.
    #[error("{solver} solver did not converge after {restarts} restarts")]
    IterativeSolverNoConvergence {
        solver: &'static str,
        restarts: usize,
    },
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

/// Port of the FEFF FMS single-site T-matrix branch.
///
/// This evaluates the same-atom portion of `fmspack`'s state-pair loop. The
/// scalar non-spin branch uses the diagonal phase-shift expression directly;
/// the spin-orbit branch combines `j=l-1/2` and `j=l+1/2` phase shifts with
/// FEFF's `t3jm` and `t3jp` Clebsch-Gordon tables. Non-single-site pairs and
/// disallowed spin-mixing pairs return zero.
pub fn fms_t_matrix_element(input: FmsTMatrixInput<'_>) -> Result<Complex32, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    ensure_state_spin(input.first.spin, input.spin_channels)?;
    ensure_state_spin(input.second.spin, input.spin_channels)?;
    if input.first.atom != input.second.atom {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let l1 = input.first.angular_momentum;
    let l2 = input.second.angular_momentum;
    let l1_signed = isize::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
        name: "l",
        value: l1,
        lx: l1,
    })?;

    if input.spin_channels == 1 && input.spin_selector == 0 {
        return if input.first == input.second {
            let phase = phase_shift_value(
                input.phase_shifts,
                input.first.spin,
                l1_signed,
                input.potential,
            )?;
            Ok(t_matrix_phase(phase))
        } else {
            Ok(Complex32::new(0.0, 0.0))
        };
    }

    if input.first == input.second {
        let coupling_spin = if input.spin_channels == 1 {
            if input.spin_selector > 0 { 2 } else { 1 }
        } else {
            input.first.spin
        };
        let minus = spin_orbit_coefficient(
            input.spin_orbit,
            false,
            l1,
            input.first.magnetic,
            coupling_spin,
        )?;
        let plus = spin_orbit_coefficient(
            input.spin_orbit,
            true,
            l1,
            input.first.magnetic,
            coupling_spin,
        )?;
        let phase_minus = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            l1_signed,
            input.potential,
        )?;
        let phase_plus = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            -l1_signed,
            input.potential,
        )?;
        return Ok(t_matrix_phase(phase_minus) * (minus * minus)
            + t_matrix_phase(phase_plus) * (plus * plus));
    }

    if input.spin_channels == 2
        && l1 == l2
        && input.first.magnetic + input.first.spin as isize
            == input.second.magnetic + input.second.spin as isize
    {
        let minus_first = spin_orbit_coefficient(
            input.spin_orbit,
            false,
            l1,
            input.first.magnetic,
            input.first.spin,
        )?;
        let minus_second = spin_orbit_coefficient(
            input.spin_orbit,
            false,
            l1,
            input.second.magnetic,
            input.second.spin,
        )?;
        let plus_first = spin_orbit_coefficient(
            input.spin_orbit,
            true,
            l1,
            input.first.magnetic,
            input.first.spin,
        )?;
        let plus_second = spin_orbit_coefficient(
            input.spin_orbit,
            true,
            l1,
            input.second.magnetic,
            input.second.spin,
        )?;
        let phase_minus_first = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            l1_signed,
            input.potential,
        )?;
        let phase_minus_second = phase_shift_value(
            input.phase_shifts,
            input.second.spin,
            l1_signed,
            input.potential,
        )?;
        let phase_plus_first = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            -l1_signed,
            input.potential,
        )?;
        let phase_plus_second = phase_shift_value(
            input.phase_shifts,
            input.second.spin,
            -l1_signed,
            input.potential,
        )?;
        let minus_phase =
            (t_matrix_phase(phase_minus_first) + t_matrix_phase(phase_minus_second)) * 0.5;
        let plus_phase =
            (t_matrix_phase(phase_plus_first) + t_matrix_phase(phase_plus_second)) * 0.5;
        return Ok(minus_phase * minus_first * minus_second + plus_phase * plus_first * plus_second);
    }

    Ok(Complex32::new(0.0, 0.0))
}

/// Build FEFF's compact FMS T-matrix table `tmatrx`.
///
/// The first row contains the same-site diagonal T element for each state. When
/// `spin_channels == 2`, the second row contains the one allowed spin-mixing
/// partner for that state, matching FEFF's compact storage used by `gglu`.
/// The returned table is Fortran-order with shape `(spin_channels, states)`.
pub fn fms_t_matrix_table(input: FmsTMatrixTableInput<'_>) -> Result<Array2<Complex32>, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    let mut table = Array2::zeros((input.spin_channels, input.states.len()).f());

    for (column, &first) in input.states.iter().enumerate() {
        ensure_state_spin(first.spin, input.spin_channels)?;
        let atom = checked_atom_index(first.atom)?;
        ensure_atom_table_index(atom, input.atoms.len())?;
        let potential = checked_phase_potential(input.atoms[atom].potential, input.phase_shifts)?;

        table[(0, column)] = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: input.spin_channels,
            spin_selector: input.spin_selector,
            potential,
            phase_shifts: input.phase_shifts,
            spin_orbit: input.spin_orbit,
        })?;

        if input.spin_channels == 2 {
            for &second in input.states {
                if second == first {
                    continue;
                }
                let value = fms_t_matrix_element(FmsTMatrixInput {
                    first,
                    second,
                    spin_channels: input.spin_channels,
                    spin_selector: input.spin_selector,
                    potential,
                    phase_shifts: input.phase_shifts,
                    spin_orbit: input.spin_orbit,
                })?;
                if value != Complex32::new(0.0, 0.0) {
                    table[(1, column)] = value;
                    break;
                }
            }
        }
    }

    Ok(table)
}

/// Assemble FEFF's iterative FMS system matrix `1 - T*G0`.
///
/// This is the shared matrix-building branch used by FEFF `ggbi`, `ggrm`, and
/// `ggtf`. It differs from [`fms_lu_scattering`] because the compact
/// single-site T-matrix multiplies `G0` from the left, and it applies FEFF's
/// `toler2` cutoff to individual `G0` elements before adding each contribution.
/// The returned matrix is Fortran-order for LAPACK-compatible downstream use.
pub fn fms_iterative_system_matrix(
    input: FmsIterativeSystemInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    fms_compact_tg_work_matrix(input, Complex32::new(-1.0, 0.0), Complex32::new(1.0, 0.0))
}

fn fms_graves_morris_system_matrix(
    input: FmsIterativeSystemInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    fms_compact_tg_work_matrix(input, Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0))
}

fn fms_compact_tg_work_matrix(
    input: FmsIterativeSystemInput<'_>,
    factor: Complex32,
    diagonal: Complex32,
) -> Result<Array2<Complex32>, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    if !input.zero_tolerance.is_finite() || input.zero_tolerance < 0.0 {
        return Err(FmsError::InvalidTolerance {
            name: "toler2",
            value: input.zero_tolerance,
        });
    }
    ensure_square_table("g0", input.free_propagator, input.states.len())?;
    ensure_axis_len(
        "tmatrx",
        "spin_band",
        input.t_matrix.shape()[0],
        input.spin_channels - 1,
    )?;
    ensure_axis_len(
        "tmatrx",
        "state",
        input.t_matrix.shape()[1],
        input.states.len() - 1,
    )?;

    let mut system_matrix = Array2::zeros((input.states.len(), input.states.len()).f());
    for column in 0..input.states.len() {
        for (row, &state) in input.states.iter().enumerate() {
            ensure_state_spin(state.spin, input.spin_channels)?;
            let diagonal_g0 = input.free_propagator[(row, column)];
            if diagonal_g0.norm() > input.zero_tolerance {
                system_matrix[(row, column)] += factor * input.t_matrix[(0, row)] * diagonal_g0;
            }

            if input.spin_channels == 2
                && let Some(partner) = fms_spin_partner_index(state, row, input.states.len())?
            {
                let spin_flip_g0 = input.free_propagator[(partner, column)];
                if spin_flip_g0.norm() > input.zero_tolerance {
                    system_matrix[(row, column)] +=
                        factor * input.t_matrix[(1, partner)] * spin_flip_g0;
                }
            }
        }
        system_matrix[(column, column)] += diagonal;
    }

    Ok(system_matrix)
}

/// Port of FEFF `ggbi`: BiCGStab-style iterative FMS scattering.
///
/// FEFF's `ggbi` solves columns of `(1 - T*G0) * x = e_j` and packs
/// `G0*x` into `gg`. This implementation preserves the FEFF single-precision
/// control flow and compact spin-orbit T-matrix storage, while returning
/// explicit errors for invalid tolerances or zero solver denominators.
pub fn fms_bicgstab_scattering(input: FmsBiCgStabInput<'_>) -> Result<FmsBiCgStabResult, FmsError> {
    let result = fms_iterative_scattering(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        fms_bicgstab_solve,
    )?;

    Ok(FmsBiCgStabResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `ggrm`: recursion-method iterative FMS scattering.
///
/// This branch solves the same `(1 - T*G0) * x = e_j` systems as
/// [`fms_bicgstab_scattering`], but follows FEFF's bi-orthogonal recursion
/// update with a bounded restart loop and explicit breakdown errors.
pub fn fms_recursion_scattering(
    input: FmsRecursionInput<'_>,
) -> Result<FmsRecursionResult, FmsError> {
    let result = fms_iterative_scattering(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        fms_recursion_solve,
    )?;

    Ok(FmsRecursionResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `gggm`: Graves-Morris/Salam iterative FMS scattering.
///
/// Unlike the other iterative branches, FEFF's `gggm` builds the compact
/// `T*G0` work matrix directly and applies the GMS update to recover
/// `(1 - T*G0)^-1 * e_j` before packing `G0*x` into `gg`.
pub fn fms_graves_morris_scattering(
    input: FmsGravesMorrisInput<'_>,
) -> Result<FmsGravesMorrisResult, FmsError> {
    let system_matrix = fms_graves_morris_system_matrix(FmsIterativeSystemInput {
        states: input.states,
        spin_channels: input.spin_channels,
        free_propagator: input.free_propagator,
        t_matrix: input.t_matrix,
        zero_tolerance: input.zero_tolerance,
    })?;
    let result = fms_iterative_scattering_with_system(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        system_matrix,
        fms_graves_morris_solve,
    )?;

    Ok(FmsGravesMorrisResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `ggtf`: TFQMR iterative FMS scattering.
///
/// This branch solves the same `(1 - T*G0) * x = e_j` systems as
/// [`fms_bicgstab_scattering`], but uses FEFF's TFQMR iteration from `ggtf`.
pub fn fms_tfqmr_scattering(input: FmsTfqmrInput<'_>) -> Result<FmsTfqmrResult, FmsError> {
    let result = fms_iterative_scattering(
        FmsIterativeScatteringInput {
            states: input.states,
            spin_channels: input.spin_channels,
            global_lmax: input.global_lmax,
            potential_lmax: input.potential_lmax,
            representative_offsets: input.representative_offsets,
            potential_start: input.potential_start,
            potential_end: input.potential_end,
            free_propagator: input.free_propagator,
            t_matrix: input.t_matrix,
            calculated_l: input.calculated_l,
            convergence_tolerance: input.convergence_tolerance,
            zero_tolerance: input.zero_tolerance,
        },
        fms_tfqmr_solve,
    )?;

    Ok(FmsTfqmrResult {
        system_matrix: result.system_matrix,
        scattering: result.scattering,
        multiple_scattering_order: result.multiple_scattering_order,
    })
}

/// Port of FEFF `gglu`: solve `(1 - G0*T) * G = G0` and pack `gg`.
///
/// This is the LU branch used by FEFF FMS. It preserves the compact `tmatrx`
/// multiplication, including the spin-orbit off-diagonal band when
/// `spin_channels == 2`, then solves with FEFF-compatible single-precision
/// complex LU factors from `refeff-linalg`.
pub fn fms_lu_scattering(input: FmsLuInput<'_>) -> Result<FmsLuResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    ensure_axis_len(
        "states",
        "potential_start",
        input.representative_offsets.len(),
        input.potential_start,
    )?;
    ensure_axis_len(
        "states",
        "potential_end",
        input.representative_offsets.len(),
        input.potential_end,
    )?;
    if input.potential_start > input.potential_end {
        return Err(FmsError::TableIndexOutOfRange {
            table: "potential_range",
            axis: "potential",
            index: input.potential_start,
        });
    }
    ensure_square_table("g0", input.free_propagator, input.states.len())?;
    ensure_axis_len(
        "tmatrx",
        "spin_band",
        input.t_matrix.shape()[0],
        input.spin_channels - 1,
    )?;
    ensure_axis_len(
        "tmatrx",
        "state",
        input.t_matrix.shape()[1],
        input.states.len() - 1,
    )?;

    let system_matrix = fms_lu_system_matrix(
        input.states,
        input.spin_channels,
        input.free_propagator,
        input.t_matrix,
    )?;
    let lu = complex32_lu_factor(system_matrix.view())?;
    let channel_count = input
        .global_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(input.spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "global_lmax",
            value: input.global_lmax,
            lx: input.global_lmax,
        })?;
    let mut scattering = Array3::zeros(
        (
            channel_count,
            channel_count,
            input.representative_offsets.len(),
        )
            .f(),
    );

    for potential in input.potential_start..=input.potential_end {
        let lmax = potential_lmax_for(input.potential_lmax, potential)?.min(input.global_lmax);
        let ipart = lmax
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .and_then(|value| value.checked_mul(input.spin_channels))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lipotx",
                value: lmax,
                lx: input.global_lmax,
            })?;
        let offset = representative_offset(input.representative_offsets, potential)?;
        ensure_axis_len(
            "g0",
            "representative_state",
            input.free_propagator.shape()[1],
            offset,
        )?;
        ensure_axis_len(
            "g0",
            "representative_block",
            input.free_propagator.shape()[1],
            offset + ipart - 1,
        )?;

        let mut rhs = Array2::zeros((input.states.len(), ipart).f());
        for row in 0..input.states.len() {
            for column in 0..ipart {
                rhs[(row, column)] = input.free_propagator[(row, offset + column)];
            }
        }
        let solved = complex32_lu_solve(&lu, rhs.view())?;
        for column in 0..ipart {
            for row in 0..ipart {
                scattering[(row, column, potential)] = solved[(offset + row, column)];
            }
        }
    }

    Ok(FmsLuResult {
        system_matrix,
        scattering,
    })
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
        Ok::<Complex32, FmsError>(sum + gamtl * gam)
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

fn ensure_spin_channels(spin_channels: usize) -> Result<(), FmsError> {
    if (1..=2).contains(&spin_channels) {
        Ok(())
    } else {
        Err(FmsError::InvalidSpinChannelCount {
            value: spin_channels,
        })
    }
}

fn ensure_state_spin(spin: usize, spin_channels: usize) -> Result<(), FmsError> {
    if (1..=spin_channels).contains(&spin) {
        Ok(())
    } else {
        Err(FmsError::InvalidStateSpin {
            spin,
            spin_channels,
        })
    }
}

fn phase_shift_value(
    phase_shifts: ArrayView3<'_, Complex32>,
    spin: usize,
    angular_momentum: isize,
    potential: usize,
) -> Result<Complex32, FmsError> {
    let spin_index = spin.checked_sub(1).ok_or(FmsError::InvalidStateSpin {
        spin,
        spin_channels: phase_shifts.shape()[0],
    })?;
    ensure_axis_len("xphase", "spin", phase_shifts.shape()[0], spin_index)?;
    ensure_axis_len("xphase", "potential", phase_shifts.shape()[2], potential)?;
    let angular_len = phase_shifts.shape()[1];
    if angular_len == 0 || angular_len.is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: "xphase",
            value: angular_len,
            lx: angular_len,
        });
    }
    let lmax = (angular_len - 1) / 2;
    let angular_index = signed_magnetic_index(angular_momentum, lmax)?;
    ensure_axis_len("xphase", "l", angular_len, angular_index)?;
    let value = phase_shifts[(spin_index, angular_index, potential)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(FmsError::NonFinitePhaseShift {
            spin,
            angular_momentum,
            potential,
        })
    }
}

fn t_matrix_phase(phase: Complex32) -> Complex32 {
    let two_i = Complex32::new(0.0, 2.0);
    ((two_i * phase).exp() - Complex32::new(1.0, 0.0)) / two_i
}

fn spin_orbit_coefficient(
    tables: &SpinOrbitCouplingTables,
    plus: bool,
    angular_momentum: usize,
    magnetic: isize,
    spin: usize,
) -> Result<f32, FmsError> {
    ensure_state_spin(spin, 2)?;
    let table = if plus { &tables.plus } else { &tables.minus };
    let table_name = if plus { "t3jp" } else { "t3jm" };
    ensure_axis_len(table_name, "l", table.shape()[0], angular_momentum)?;
    let offset = isize::try_from(tables.m_offset).map_err(|_| FmsError::InvalidAngularLimit {
        name: table_name,
        value: tables.m_offset,
        lx: tables.m_offset,
    })?;
    let magnetic_index =
        usize::try_from(magnetic + offset).map_err(|_| FmsError::InvalidAngularLimit {
            name: table_name,
            value: magnetic.unsigned_abs(),
            lx: tables.m_offset,
        })?;
    ensure_axis_len(table_name, "m", table.shape()[1], magnetic_index)?;
    let spin_index = spin - 1;
    ensure_axis_len(table_name, "spin", table.shape()[2], spin_index)?;
    Ok(table[(angular_momentum, magnetic_index, spin_index)] as f32)
}

struct FmsIterativeScatteringInput<'a> {
    states: &'a [StateKet],
    spin_channels: usize,
    global_lmax: usize,
    potential_lmax: &'a [usize],
    representative_offsets: &'a [Option<usize>],
    potential_start: usize,
    potential_end: usize,
    free_propagator: ArrayView2<'a, Complex32>,
    t_matrix: ArrayView2<'a, Complex32>,
    calculated_l: &'a [bool],
    convergence_tolerance: f32,
    zero_tolerance: f32,
}

struct FmsIterativeScatteringResult {
    system_matrix: Array2<Complex32>,
    scattering: Array3<Complex32>,
    multiple_scattering_order: usize,
}

fn fms_iterative_scattering(
    input: FmsIterativeScatteringInput<'_>,
    solve: impl Fn(ArrayView2<'_, Complex32>, usize, f32) -> Result<(Vec<Complex32>, usize), FmsError>,
) -> Result<FmsIterativeScatteringResult, FmsError> {
    let system_matrix = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: input.states,
        spin_channels: input.spin_channels,
        free_propagator: input.free_propagator,
        t_matrix: input.t_matrix,
        zero_tolerance: input.zero_tolerance,
    })?;
    fms_iterative_scattering_with_system(input, system_matrix, solve)
}

fn fms_iterative_scattering_with_system(
    input: FmsIterativeScatteringInput<'_>,
    system_matrix: Array2<Complex32>,
    solve: impl Fn(ArrayView2<'_, Complex32>, usize, f32) -> Result<(Vec<Complex32>, usize), FmsError>,
) -> Result<FmsIterativeScatteringResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    ensure_axis_len(
        "states",
        "potential_start",
        input.representative_offsets.len(),
        input.potential_start,
    )?;
    ensure_axis_len(
        "states",
        "potential_end",
        input.representative_offsets.len(),
        input.potential_end,
    )?;
    if input.potential_start > input.potential_end {
        return Err(FmsError::TableIndexOutOfRange {
            table: "potential_range",
            axis: "potential",
            index: input.potential_start,
        });
    }
    if !input.convergence_tolerance.is_finite() || input.convergence_tolerance < 0.0 {
        return Err(FmsError::InvalidTolerance {
            name: "toler1",
            value: input.convergence_tolerance,
        });
    }
    ensure_square_table("g0t", system_matrix.view(), input.states.len())?;

    let channel_count = input
        .global_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(input.spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "global_lmax",
            value: input.global_lmax,
            lx: input.global_lmax,
        })?;
    let mut scattering = Array3::zeros(
        (
            channel_count,
            channel_count,
            input.representative_offsets.len(),
        )
            .f(),
    );
    let mut multiple_scattering_order = 0;

    for potential in input.potential_start..=input.potential_end {
        let lmax = potential_lmax_for(input.potential_lmax, potential)?.min(input.global_lmax);
        let ipart = lmax
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .and_then(|value| value.checked_mul(input.spin_channels))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lipotx",
                value: lmax,
                lx: input.global_lmax,
            })?;
        let offset = representative_offset(input.representative_offsets, potential)?;
        ensure_axis_len(
            "g0",
            "representative_state",
            input.free_propagator.shape()[0],
            offset,
        )?;
        ensure_axis_len(
            "g0",
            "representative_block",
            input.free_propagator.shape()[0],
            offset
                .checked_add(ipart - 1)
                .ok_or(FmsError::TableIndexOutOfRange {
                    table: "g0",
                    axis: "representative_block",
                    index: ipart,
                })?,
        )?;

        for source_column in 0..ipart {
            let source_state =
                offset
                    .checked_add(source_column)
                    .ok_or(FmsError::TableIndexOutOfRange {
                        table: "states",
                        axis: "source_state",
                        index: source_column,
                    })?;
            ensure_axis_len("states", "source_state", input.states.len(), source_state)?;
            let angular_momentum = input.states[source_state].angular_momentum;
            ensure_axis_len("lcalc", "l", input.calculated_l.len(), angular_momentum)?;
            if !input.calculated_l[angular_momentum] {
                continue;
            }

            let (solution, msord) = solve(
                system_matrix.view(),
                source_state,
                input.convergence_tolerance,
            )?;
            multiple_scattering_order = msord;
            for row in 0..ipart {
                let target_state =
                    offset
                        .checked_add(row)
                        .ok_or(FmsError::TableIndexOutOfRange {
                            table: "g0",
                            axis: "row_state",
                            index: row,
                        })?;
                ensure_axis_len(
                    "g0",
                    "row_state",
                    input.free_propagator.shape()[0],
                    target_state,
                )?;
                let value = (0..input.states.len())
                    .map(|state| input.free_propagator[(target_state, state)] * solution[state])
                    .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value);
                scattering[(row, source_column, potential)] = value;
            }
        }
    }

    Ok(FmsIterativeScatteringResult {
        system_matrix,
        scattering,
        multiple_scattering_order,
    })
}

fn fms_bicgstab_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];
    let mut rvec = vec![zero; state_count];
    rvec[source_state] = Complex32::new(1.0, 0.0);

    if fms_vector_within_tolerance(&rvec, tolerance) {
        return Ok((xvec, multiple_scattering_order));
    }

    let pvec = rvec.clone();
    let avec = fms_matvec(system_matrix, &pvec);
    multiple_scattering_order += 1;

    let mut aa = fms_cdot(&avec, &avec);
    let wa = fms_cdot(&rvec, &avec);
    let aw = wa.conj();
    let mut ww = fms_cdot(&rvec, &rvec);
    fms_checked_nonzero(aa, "ggbi", "avec dot avec")?;
    fms_checked_nonzero(ww, "ggbi", "rvec dot rvec")?;
    let dd = aa * ww - aw * wa;
    let scaled_dd = fms_checked_divide(
        fms_checked_divide(dd, aa, "ggbi", "dd/aa")?,
        ww,
        "ggbi",
        "dd/ww",
    )?;
    let yvec = if scaled_dd.norm() < 1.0e-8 {
        rvec.iter().map(|&value| value / ww).collect::<Vec<_>>()
    } else {
        fms_checked_nonzero(dd, "ggbi", "Gram determinant")?;
        ww = (ww - aw) / dd;
        aa = (wa - aa) / dd;
        rvec.iter()
            .zip(avec.iter())
            .map(|(&residual, &matrix_residual)| residual * aa + matrix_residual * ww)
            .collect::<Vec<_>>()
    };
    let del = fms_cdot(&yvec, &rvec);
    let delp = fms_cdot(&yvec, &avec);
    let omega = fms_checked_divide(del, delp, "ggbi", "omega")?;
    let svec = rvec
        .iter()
        .zip(avec.iter())
        .map(|(&residual, &matrix_residual)| residual - omega * matrix_residual)
        .collect::<Vec<_>>();

    if fms_vector_within_tolerance(&svec, tolerance) {
        for (solution, &direction) in xvec.iter_mut().zip(pvec.iter()) {
            *solution += omega * direction;
        }
        return Ok((xvec, multiple_scattering_order));
    }

    let asve = fms_matvec(system_matrix, &svec);
    multiple_scattering_order += 1;
    aa = fms_cdot(&asve, &asve);
    let wa = fms_cdot(&asve, &svec);
    let chi = fms_checked_divide(wa, aa, "ggbi", "chi")?;
    for ((solution, &direction), &shadow) in xvec.iter_mut().zip(pvec.iter()).zip(svec.iter()) {
        *solution += omega * direction + chi * shadow;
    }

    // FEFF `ggbi` resets `ipass` before label 380, so this branch exits after
    // the first residual update even when the residual is still above tolerance.
    Ok((xvec, multiple_scattering_order))
}

fn fms_recursion_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    const MAX_RESTARTS: usize = 128;
    const MAX_ITERATIONS: usize = 100;

    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let one = Complex32::new(1.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];

    for restart in 0..MAX_RESTARTS {
        let mut rvec = if restart > 0 {
            fms_matvec(system_matrix, &xvec)
        } else {
            vec![zero; state_count]
        };
        rvec[source_state] -= one;

        let mut xket = rvec.iter().map(|&value| -value).collect::<Vec<_>>();
        let residual_norm = fms_cdot(&xket, &xket);
        if residual_norm == zero {
            return Ok((xvec, multiple_scattering_order));
        }

        let xfnorm =
            1.0 / fms_checked_positive_real(residual_norm.re, "ggrm", "initial residual norm")?;
        let mut xbra = xket.iter().map(|&value| value * xfnorm).collect::<Vec<_>>();
        let mut tket = fms_matvec(system_matrix, &xket);
        multiple_scattering_order += 1;

        let mut aa = fms_cdot(&xbra, &tket);
        let mut aac = aa.conj();
        let mut bb = zero;
        let mut bbc = zero;
        let mut betac = aa;
        fms_checked_nonzero(betac, "ggrm", "initial beta")?;

        let mut yy = one;
        let mut xketp = vec![zero; state_count];
        let mut xbrap = vec![zero; state_count];
        let mut zvec = xket.clone();
        for (solution, &basis) in xvec.iter_mut().zip(zvec.iter()) {
            *solution += basis / betac;
        }
        let mut svec = tket.clone();
        for (residual, &matrix_basis) in rvec.iter_mut().zip(svec.iter()) {
            *residual += matrix_basis / betac;
        }

        for _ in 0..MAX_ITERATIONS {
            for ((matrix_basis, &basis), &previous_basis) in
                tket.iter_mut().zip(xket.iter()).zip(xketp.iter())
            {
                *matrix_basis -= aa * basis + bb * previous_basis;
            }

            let mut tbra = fms_adjoint_matvec(system_matrix, &xbra);
            for ((matrix_bra, &bra), &previous_bra) in
                tbra.iter_mut().zip(xbra.iter()).zip(xbrap.iter())
            {
                *matrix_bra -= aac * bra + bbc * previous_bra;
            }

            let recurrence_norm = fms_cdot(&tbra, &tket);
            if recurrence_norm == zero {
                return Ok((xvec, multiple_scattering_order));
            }
            bb = recurrence_norm.sqrt();
            bbc = bb.conj();
            fms_checked_nonzero(bb, "ggrm", "recursion norm")?;
            fms_checked_nonzero(bbc, "ggrm", "adjoint recursion norm")?;

            xketp = xket;
            xbrap = xbra;
            xket = tket.iter().map(|&value| value / bb).collect();
            xbra = tbra.iter().map(|&value| value / bbc).collect();

            tket = fms_matvec(system_matrix, &xket);
            multiple_scattering_order += 1;
            aa = fms_cdot(&xbra, &tket);
            aac = aa.conj();

            let alphac = fms_checked_divide(bb, betac, "ggrm", "alpha")?;
            for ((basis, &current), (matrix_basis, &matrix_current)) in zvec
                .iter_mut()
                .zip(xket.iter())
                .zip(svec.iter_mut().zip(tket.iter()))
            {
                *basis = current - alphac * *basis;
                *matrix_basis = matrix_current - alphac * *matrix_basis;
            }

            betac = aa - alphac * bb;
            fms_checked_nonzero(betac, "ggrm", "beta")?;
            yy = -alphac * yy;
            let gamma = fms_checked_divide(yy, betac, "ggrm", "gamma")?;
            for ((solution, residual), (&basis, &matrix_basis)) in xvec
                .iter_mut()
                .zip(rvec.iter_mut())
                .zip(zvec.iter().zip(svec.iter()))
            {
                *solution += gamma * basis;
                *residual += gamma * matrix_basis;
            }

            if fms_vector_within_tolerance(&rvec, tolerance) {
                return Ok((xvec, multiple_scattering_order));
            }
        }
    }

    Err(FmsError::IterativeSolverNoConvergence {
        solver: "ggrm",
        restarts: MAX_RESTARTS,
    })
}

fn fms_graves_morris_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    const MAX_RESTARTS: usize = 128;
    const MAX_ITERATIONS: usize = 10;

    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let one = Complex32::new(1.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];
    let mut bvec = vec![zero; state_count];
    let mut x0 = vec![zero; state_count];
    let mut q0 = one;
    bvec[source_state] = one;

    for restart in 0..MAX_RESTARTS {
        if restart > 0 {
            fms_checked_nonzero(q0, "gggm", "restart q0")?;
            for (solution, &basis) in xvec.iter_mut().zip(x0.iter()) {
                *solution += basis / q0;
            }
            let avec = fms_matvec(system_matrix, &xvec);
            for ((rhs, &matrix_solution), &solution) in
                bvec.iter_mut().zip(avec.iter()).zip(xvec.iter())
            {
                *rhs = matrix_solution - solution;
            }
            bvec[source_state] += one;
        }

        let mut r0 = bvec.clone();
        x0.fill(zero);
        let mut x1 = bvec.clone();
        let mut r1 = fms_matvec(system_matrix, &bvec);
        multiple_scattering_order += 1;

        let mut ww = fms_cdot(&r0, &r0);
        let mut aa = fms_cdot(&r1, &r1);
        let wa = fms_cdot(&r0, &r1);
        let aw = wa.conj();
        fms_checked_nonzero(aa, "gggm", "r1 norm")?;
        fms_checked_nonzero(ww, "gggm", "r0 norm")?;
        let dd = aa * ww - aw * wa;
        let scaled_dd = fms_checked_divide(
            fms_checked_divide(dd, aa, "gggm", "dd/aa")?,
            ww,
            "gggm",
            "dd/ww",
        )?;
        let wvec = if scaled_dd.norm() < 1.0e-8 {
            r0.iter().map(|&value| value / ww).collect::<Vec<_>>()
        } else {
            fms_checked_nonzero(dd, "gggm", "Gram determinant")?;
            ww = (ww - aw) / dd;
            aa = (wa - aa) / dd;
            r0.iter()
                .zip(r1.iter())
                .map(|(&current, &matrix_current)| current * aa + matrix_current * ww)
                .collect::<Vec<_>>()
        };

        let mut e0 = fms_cdot(&wvec, &r0);
        let mut e1 = fms_cdot(&wvec, &r1);
        q0 = one;
        let mut q1 = one;

        for _ in 0..MAX_ITERATIONS {
            let tol = fms_scaled_tolerance(tolerance, q1.norm() / 10.0, "gggm", "r1 tolerance")?;
            if fms_vector_within_tolerance(&r1, tol) {
                fms_checked_nonzero(q1, "gggm", "q1")?;
                for (solution, &basis) in xvec.iter_mut().zip(x1.iter()) {
                    *solution += basis / q1;
                }
                return Ok((xvec, multiple_scattering_order));
            }

            let alpha = fms_checked_divide(e1, e0, "gggm", "alpha")?;
            let mut t0 = r1
                .iter()
                .zip(r0.iter())
                .map(|(&current, &previous)| current - alpha * previous)
                .collect::<Vec<_>>();
            let t1 = fms_matvec(system_matrix, &t0);
            multiple_scattering_order += 1;

            let wa = fms_cdot(&t0, &t1);
            let ww = fms_cdot(&t0, &t0);
            let aa = fms_cdot(&t1, &t1);
            let aw = wa.conj();
            let theta = fms_checked_divide(wa - aa, ww - aw, "gggm", "theta")?;

            for ((residual, &matrix_basis), &basis) in r0.iter_mut().zip(t1.iter()).zip(t0.iter()) {
                *residual = matrix_basis - theta * basis;
            }
            let dd = one - theta;
            for ((basis, &current), &previous) in x0.iter_mut().zip(t0.iter()).zip(x1.iter()) {
                *basis = current + dd * (previous - alpha * *basis);
            }
            q0 = dd * (q1 - alpha * q0);
            let tol = fms_scaled_tolerance(tolerance, q0.norm(), "gggm", "r0 tolerance")?;
            if fms_vector_within_tolerance(&r0, tol) {
                fms_checked_nonzero(q0, "gggm", "q0")?;
                for (solution, &basis) in xvec.iter_mut().zip(x0.iter()) {
                    *solution += basis / q0;
                }
                return Ok((xvec, multiple_scattering_order));
            }

            e0 = fms_cdot(&wvec, &r0);
            let beta = fms_checked_divide(e0, e1, "gggm", "beta")?;
            for ((basis, &current), &previous) in t0.iter_mut().zip(r0.iter()).zip(r1.iter()) {
                *basis = current - beta * previous;
            }
            let avec = fms_matvec(system_matrix, &t0);
            multiple_scattering_order += 1;
            let dd = beta * theta;
            for (residual, &matrix_basis) in r1.iter_mut().zip(avec.iter()) {
                *residual = matrix_basis + dd * *residual;
            }
            e1 = fms_cdot(&wvec, &r1);

            let dd = beta * (one - theta);
            for ((basis, &current), &correction) in x1.iter_mut().zip(x0.iter()).zip(t0.iter()) {
                *basis = current - dd * *basis + correction;
            }
            q1 = q0 - (one - theta) * beta * q1;
        }
    }

    Err(FmsError::IterativeSolverNoConvergence {
        solver: "gggm",
        restarts: MAX_RESTARTS,
    })
}

fn fms_tfqmr_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    const MAX_RESTARTS: usize = 128;
    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];
    let mut avec = vec![zero; state_count];

    for restart in 0..MAX_RESTARTS {
        if restart > 0 {
            avec = fms_matvec(system_matrix, &xvec);
        }
        let mut uvec = avec.iter().map(|&value| -value).collect::<Vec<_>>();
        uvec[source_state] += Complex32::new(1.0, 0.0);
        avec = fms_matvec(system_matrix, &uvec);
        multiple_scattering_order += 1;

        let mut wvec = uvec.clone();
        let mut vvec = avec.clone();
        let mut dvec = vec![zero; state_count];
        let aa = fms_cdot(&uvec, &uvec);
        fms_checked_nonzero(aa, "ggtf", "initial residual norm")?;
        let mut tau = fms_checked_positive_real(aa.re, "ggtf", "tau")?.sqrt();
        let mut nu = 0.0;
        let mut eta = zero;
        let rvec = uvec.iter().map(|&value| value / aa).collect::<Vec<_>>();
        let mut rho = Complex32::new(1.0, 0.0);
        let mut alpha = zero;

        for nit in 0..=20 {
            if nit % 2 == 0 {
                let aa = fms_cdot(&rvec, &vvec);
                alpha = fms_checked_divide(rho, aa, "ggtf", "alpha")?;
            } else {
                avec = fms_matvec(system_matrix, &uvec);
                multiple_scattering_order += 1;
            }

            for (w, &matrix_direction) in wvec.iter_mut().zip(avec.iter()) {
                *w -= alpha * matrix_direction;
            }
            let aa = fms_checked_divide((nu * nu) * eta, alpha, "ggtf", "dvec factor")?;
            let previous_dvec = dvec.clone();
            for ((direction, &basis), &previous) in
                dvec.iter_mut().zip(uvec.iter()).zip(previous_dvec.iter())
            {
                *direction = basis + aa * previous;
            }
            let aa = fms_cdot(&wvec, &wvec);
            let norm = fms_checked_nonnegative_real(aa.re, "ggtf", "wvec norm")?.sqrt();
            nu = norm / tau;
            let cm = 1.0 / (1.0 + nu * nu).sqrt();
            tau *= nu * cm;
            eta = (cm * cm) * alpha;
            for (solution, &direction) in xvec.iter_mut().zip(dvec.iter()) {
                *solution += eta * direction;
            }

            let err = tau * (((1.0 + nit as f32) / state_count as f32).sqrt()) * 10.0;
            if err.abs() < tolerance {
                return Ok((xvec, multiple_scattering_order));
            }

            if nit % 2 != 0 {
                let previous_rho = rho;
                rho = fms_cdot(&rvec, &wvec);
                let beta = fms_checked_divide(rho, previous_rho, "ggtf", "beta")?;
                for (basis, &shadow) in uvec.iter_mut().zip(wvec.iter()) {
                    *basis = shadow + beta * *basis;
                }
                for (matrix_direction, &current) in vvec.iter_mut().zip(avec.iter()) {
                    *matrix_direction = beta * (current + beta * *matrix_direction);
                }
                avec = fms_matvec(system_matrix, &uvec);
                multiple_scattering_order += 1;
                for (matrix_direction, &current) in vvec.iter_mut().zip(avec.iter()) {
                    *matrix_direction += current;
                }
            } else {
                for (basis, &matrix_direction) in uvec.iter_mut().zip(vvec.iter()) {
                    *basis -= alpha * matrix_direction;
                }
            }
        }
    }

    Err(FmsError::IterativeSolverNoConvergence {
        solver: "ggtf",
        restarts: MAX_RESTARTS,
    })
}

fn fms_vector_within_tolerance(vector: &[Complex32], tolerance: f32) -> bool {
    vector
        .iter()
        .all(|value| value.re.abs() <= tolerance && value.im.abs() <= tolerance)
}

fn fms_scaled_tolerance(
    tolerance: f32,
    scale: f32,
    solver: &'static str,
    step: &'static str,
) -> Result<f32, FmsError> {
    let scaled = tolerance * scale;
    if scaled.is_finite() && scaled >= 0.0 {
        Ok(scaled)
    } else {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    }
}

fn fms_cdot(left: &[Complex32], right: &[Complex32]) -> Complex32 {
    left.iter()
        .zip(right.iter())
        .map(|(&bra, &ket)| bra.conj() * ket)
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn fms_matvec(matrix: ArrayView2<'_, Complex32>, vector: &[Complex32]) -> Vec<Complex32> {
    let mut output = vec![Complex32::new(0.0, 0.0); vector.len()];
    for column in 0..vector.len() {
        for row in 0..vector.len() {
            output[row] += matrix[(row, column)] * vector[column];
        }
    }
    output
}

fn fms_adjoint_matvec(matrix: ArrayView2<'_, Complex32>, vector: &[Complex32]) -> Vec<Complex32> {
    let mut output = vec![Complex32::new(0.0, 0.0); vector.len()];
    for column in 0..vector.len() {
        for row in 0..vector.len() {
            output[column] += matrix[(row, column)].conj() * vector[row];
        }
    }
    output
}

fn fms_checked_divide(
    numerator: Complex32,
    denominator: Complex32,
    solver: &'static str,
    step: &'static str,
) -> Result<Complex32, FmsError> {
    fms_checked_nonzero(denominator, solver, step)?;
    Ok(numerator / denominator)
}

fn fms_checked_nonzero(
    value: Complex32,
    solver: &'static str,
    step: &'static str,
) -> Result<(), FmsError> {
    if value == Complex32::new(0.0, 0.0) {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    } else {
        Ok(())
    }
}

fn fms_checked_positive_real(
    value: f32,
    solver: &'static str,
    step: &'static str,
) -> Result<f32, FmsError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    }
}

fn fms_checked_nonnegative_real(
    value: f32,
    solver: &'static str,
    step: &'static str,
) -> Result<f32, FmsError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    }
}

fn fms_lu_system_matrix(
    states: &[StateKet],
    spin_channels: usize,
    free_propagator: ArrayView2<'_, Complex32>,
    t_matrix: ArrayView2<'_, Complex32>,
) -> Result<Array2<Complex32>, FmsError> {
    if states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }

    let mut system_matrix = Array2::zeros((states.len(), states.len()).f());
    for (column, &state) in states.iter().enumerate() {
        ensure_state_spin(state.spin, spin_channels)?;
        for row in 0..states.len() {
            system_matrix[(row, column)] = -free_propagator[(row, column)] * t_matrix[(0, column)];
        }

        if spin_channels == 2
            && let Some(partner) = fms_spin_partner_index(state, column, states.len())?
        {
            for row in 0..states.len() {
                system_matrix[(row, column)] -=
                    free_propagator[(row, partner)] * t_matrix[(1, column)];
            }
        }
        system_matrix[(column, column)] += Complex32::new(1.0, 0.0);
    }

    Ok(system_matrix)
}

fn fms_spin_partner_index(
    state: StateKet,
    column: usize,
    state_count: usize,
) -> Result<Option<usize>, FmsError> {
    let angular_momentum =
        isize::try_from(state.angular_momentum).map_err(|_| FmsError::InvalidAngularLimit {
            name: "l",
            value: state.angular_momentum,
            lx: state.angular_momentum,
        })?;
    let projection = state.magnetic + state.spin as isize;
    if projection <= -angular_momentum + 1 || projection >= angular_momentum + 2 {
        return Ok(None);
    }

    let column = isize::try_from(column).map_err(|_| FmsError::TableIndexOutOfRange {
        table: "states",
        axis: "state",
        index: column,
    })?;
    let partner = match state.spin {
        1 => column - 1,
        2 => column + 1,
        spin => {
            return Err(FmsError::InvalidStateSpin {
                spin,
                spin_channels: 2,
            });
        }
    };
    let partner = usize::try_from(partner).map_err(|_| FmsError::TableIndexOutOfRange {
        table: "states",
        axis: "spin_partner",
        index: 0,
    })?;
    ensure_axis_len("states", "spin_partner", state_count, partner)?;
    Ok(Some(partner))
}

fn ensure_square_table(
    table: &'static str,
    matrix: ArrayView2<'_, Complex32>,
    expected_order: usize,
) -> Result<(), FmsError> {
    if matrix.shape() == [expected_order, expected_order] {
        Ok(())
    } else {
        Err(FmsError::TableIndexOutOfRange {
            table,
            axis: "shape",
            index: expected_order,
        })
    }
}

fn potential_lmax_for(potential_lmax: &[usize], potential: usize) -> Result<usize, FmsError> {
    potential_lmax
        .get(potential)
        .copied()
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "lipotx",
            axis: "potential",
            index: potential,
        })
}

fn representative_offset(
    representative_offsets: &[Option<usize>],
    potential: usize,
) -> Result<usize, FmsError> {
    representative_offsets
        .get(potential)
        .copied()
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "i0",
            axis: "potential",
            index: potential,
        })?
        .ok_or(FmsError::MissingRepresentativePotential { potential })
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

fn checked_phase_potential(
    potential: i32,
    phase_shifts: ArrayView3<'_, Complex32>,
) -> Result<usize, FmsError> {
    let potential_count = phase_shifts.shape()[2];
    if potential_count == 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "xphase",
            axis: "potential",
            index: 0,
        });
    }
    checked_potential(potential, potential_count - 1)
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
        FmsAtom, FmsBiCgStabInput, FmsFreePropagatorInput, FmsFreePropagatorMatrixInput,
        FmsGravesMorrisInput, FmsIterativeSystemInput, FmsLuInput, FmsRecursionInput,
        FmsRotationDirection, FmsTMatrixInput, FmsTMatrixTableInput, FmsTfqmrInput,
        fms_bicgstab_scattering, fms_free_propagator_element, fms_free_propagator_matrix,
        fms_graves_morris_scattering, fms_iterative_system_matrix, fms_lu_scattering,
        fms_pair_tables, fms_recursion_scattering, fms_rotation_matrix, fms_t_matrix_element,
        fms_t_matrix_table, fms_tfqmr_scattering, pair_polar_angles, sort_atoms_by_radius,
        sort_representative_atoms,
    };
    use super::{FmsError, rehr_albers_polynomials, rehr_albers_z_axis_propagator};
    use crate::{
        Real,
        angular::{legendre_normalization_table, spin_orbit_coupling_tables},
        state::{StateKet, construct_state_kets},
    };
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
    fn fms_t_matrix_matches_feff_reference_branches() -> Result<(), Box<dyn Error>> {
        let phases = reference_phase_shifts();
        let spin_orbit = spin_orbit_coupling_tables(2)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };

        let non_spin = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: 1,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;
        let spin_orbit_diagonal = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: 2,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;
        let spin_mixing = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: StateKet {
                magnetic: 0,
                spin: 2,
                ..first
            },
            spin_channels: 2,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;

        assert_complex32_close(non_spin, Complex32::new(0.176_180_14, 0.083_294_78));
        assert_complex32_close(
            spin_orbit_diagonal,
            Complex32::new(0.068_288_13, 0.065_378_49),
        );
        assert_complex32_close(spin_mixing, Complex32::new(-0.087_964_38, -0.001_144_098_1));
        Ok(())
    }

    #[test]
    fn fms_t_matrix_returns_zero_for_disallowed_pairs() -> Result<(), Box<dyn Error>> {
        let phases = reference_phase_shifts();
        let spin_orbit = spin_orbit_coupling_tables(2)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };

        let different_atom = fms_t_matrix_element(FmsTMatrixInput {
            second: StateKet { atom: 2, ..first },
            first,
            spin_channels: 2,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;
        let disallowed_spin_mix = fms_t_matrix_element(FmsTMatrixInput {
            second: StateKet {
                magnetic: -1,
                spin: 2,
                ..first
            },
            first,
            spin_channels: 2,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;

        assert_complex32_close(different_atom, Complex32::new(0.0, 0.0));
        assert_complex32_close(disallowed_spin_mix, Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn fms_t_matrix_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
        let mut phases = reference_phase_shifts();
        let spin_orbit = spin_orbit_coupling_tables(2)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };

        let invalid_spin_count = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: 3,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        });
        assert!(matches!(
            invalid_spin_count,
            Err(FmsError::InvalidSpinChannelCount { value: 3 })
        ));

        let invalid_state_spin = fms_t_matrix_element(FmsTMatrixInput {
            first: StateKet { spin: 2, ..first },
            second: StateKet { spin: 2, ..first },
            spin_channels: 1,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        });
        assert!(matches!(
            invalid_state_spin,
            Err(FmsError::InvalidStateSpin {
                spin: 2,
                spin_channels: 1,
            })
        ));

        phases[(0, 4, 1)] = Complex32::new(f32::NAN, 0.0);
        let nonfinite_phase = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: 1,
            spin_selector: 0,
            potential: 1,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        });
        assert!(matches!(
            nonfinite_phase,
            Err(FmsError::NonFinitePhaseShift {
                spin: 1,
                angular_momentum: 2,
                potential: 1,
            })
        ));
        Ok(())
    }

    #[test]
    fn fms_t_matrix_table_matches_feff_compact_layout() -> Result<(), Box<dyn Error>> {
        let phases = reference_phase_shifts();
        let spin_orbit = spin_orbit_coupling_tables(2)?;
        let atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };
        let states = [
            first,
            StateKet {
                magnetic: 0,
                spin: 2,
                ..first
            },
        ];

        let table = fms_t_matrix_table(FmsTMatrixTableInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            spin_selector: 0,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;

        assert_eq!(table.shape(), &[2, 2]);
        assert_eq!(table.strides(), &[1, 2]);
        assert_complex32_close(table[(0, 0)], Complex32::new(0.068_288_13, 0.065_378_49));
        assert_complex32_close(
            table[(1, 0)],
            Complex32::new(-0.087_964_38, -0.001_144_098_1),
        );
        Ok(())
    }

    #[test]
    fn fms_t_matrix_table_handles_non_spin_branch() -> Result<(), Box<dyn Error>> {
        let phases = reference_phase_shifts();
        let spin_orbit = spin_orbit_coupling_tables(2)?;
        let atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        let states = [StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        }];

        let table = fms_t_matrix_table(FmsTMatrixTableInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 1,
            spin_selector: 0,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        })?;

        assert_eq!(table.shape(), &[1, 1]);
        assert_complex32_close(table[(0, 0)], Complex32::new(0.176_180_14, 0.083_294_78));
        Ok(())
    }

    #[test]
    fn fms_t_matrix_table_rejects_invalid_potential() -> Result<(), Box<dyn Error>> {
        let phases = reference_phase_shifts();
        let spin_orbit = spin_orbit_coupling_tables(2)?;
        let atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 2,
        }];
        let states = [StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        }];

        let result = fms_t_matrix_table(FmsTMatrixTableInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 1,
            spin_selector: 0,
            phase_shifts: phases.view(),
            spin_orbit: &spin_orbit,
        });

        assert!(matches!(
            result,
            Err(FmsError::PotentialOutOfRange {
                potential: 2,
                max_potential: 1,
            })
        ));
        Ok(())
    }

    #[test]
    fn fms_iterative_system_matrix_matches_feff_reference() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let system = fms_iterative_system_matrix(FmsIterativeSystemInput {
            states: &state_set.states,
            spin_channels: 2,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            zero_tolerance: 0.0,
        })?;

        assert_eq!(system.shape(), &[8, 8]);
        assert_eq!(system.strides(), &[1, 8]);
        assert_complex32_close(
            matrix_sum(system.view()),
            Complex32::new(7.909_579_3, -0.516_9),
        );
        assert_complex32_close(system[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(system[(1, 3)], Complex32::new(0.001_4, -0.003_199_999_7));
        assert_complex32_close(
            system[(4, 5)],
            Complex32::new(0.001_230_000_3, -0.011_239_999),
        );
        assert_complex32_close(system[(6, 7)], Complex32::new(0.001_789_999_7, -0.020_9));

        let cutoff_system = fms_iterative_system_matrix(FmsIterativeSystemInput {
            states: &state_set.states,
            spin_channels: 2,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            zero_tolerance: 0.09,
        })?;

        assert_complex32_close(
            matrix_sum(cutoff_system.view()),
            Complex32::new(7.922_833_4, -0.471_125_07),
        );
        assert_complex32_close(cutoff_system[(1, 3)], Complex32::new(0.0, 0.0));
        assert_complex32_close(
            cutoff_system[(4, 5)],
            Complex32::new(0.001_230_000_3, -0.011_239_999),
        );
        Ok(())
    }

    #[test]
    fn fms_iterative_system_matrix_rejects_invalid_tolerance() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_iterative_system_matrix(FmsIterativeSystemInput {
            states: &state_set.states,
            spin_channels: 2,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            zero_tolerance: -1.0,
        });

        assert!(matches!(
            result,
            Err(FmsError::InvalidTolerance {
                name: "toler2",
                value: -1.0,
            })
        ));
        Ok(())
    }

    #[test]
    fn fms_bicgstab_scattering_matches_feff_ggbi_reference() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_bicgstab_scattering(FmsBiCgStabInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &state_set.representative_offsets,
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            calculated_l: &[true, true],
            convergence_tolerance: 1.0e-5,
            zero_tolerance: 0.0,
        })?;

        assert_eq!(result.system_matrix.shape(), &[8, 8]);
        assert_eq!(result.system_matrix.strides(), &[1, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.scattering.strides(), &[1, 8, 64]);
        assert_eq!(result.multiple_scattering_order, 2);
        assert_complex32_close(
            matrix_sum(result.system_matrix.view()),
            Complex32::new(7.909_579_3, -0.516_9),
        );
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            Complex32::new(-2.949_217_6, 4.806_942),
        );
        assert_complex32_close(
            result.scattering[(0, 0, 0)],
            Complex32::new(-0.007_855_818, -0.003_201_462_3),
        );
        assert_complex32_close(
            result.scattering[(1, 3, 0)],
            Complex32::new(-0.066_029_795, 0.044_123_195),
        );
        assert_complex32_close(
            result.scattering[(6, 7, 0)],
            Complex32::new(-0.096_492_656, 0.140_840_8),
        );
        Ok(())
    }

    #[test]
    fn fms_bicgstab_scattering_respects_lcalc_mask() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_bicgstab_scattering(FmsBiCgStabInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &state_set.representative_offsets,
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            calculated_l: &[true, false],
            convergence_tolerance: 1.0e-5,
            zero_tolerance: 0.0,
        })?;

        assert_complex32_close(
            result.scattering[(0, 0, 0)],
            Complex32::new(-0.007_855_818, -0.003_201_462_3),
        );
        assert_complex32_close(result.scattering[(2, 2, 0)], Complex32::new(0.0, 0.0));
        assert_complex32_close(result.scattering[(7, 7, 0)], Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn fms_recursion_scattering_matches_feff_ggrm_reference() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_recursion_scattering(FmsRecursionInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &state_set.representative_offsets,
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            calculated_l: &[true, true],
            convergence_tolerance: 1.0e-5,
            zero_tolerance: 0.0,
        })?;

        assert_eq!(result.system_matrix.shape(), &[8, 8]);
        assert_eq!(result.system_matrix.strides(), &[1, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.scattering.strides(), &[1, 8, 64]);
        assert_eq!(result.multiple_scattering_order, 3);
        assert_complex32_close(
            matrix_sum(result.system_matrix.view()),
            Complex32::new(7.909_579_3, -0.516_9),
        );
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            Complex32::new(-2.944_324, 4.799_402),
        );
        assert_complex32_close(
            result.scattering[(0, 0, 0)],
            Complex32::new(-0.007_797_021, -0.003_244_287_3),
        );
        assert_complex32_close(
            result.scattering[(1, 3, 0)],
            Complex32::new(-0.065_967_52, 0.044_093_154),
        );
        assert_complex32_close(
            result.scattering[(6, 7, 0)],
            Complex32::new(-0.096_285_72, 0.140_520_17),
        );
        Ok(())
    }

    #[test]
    fn fms_graves_morris_scattering_matches_feff_gggm_reference() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_graves_morris_scattering(FmsGravesMorrisInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &state_set.representative_offsets,
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            calculated_l: &[true, true],
            convergence_tolerance: 1.0e-5,
            zero_tolerance: 0.0,
        })?;

        assert_eq!(result.system_matrix.shape(), &[8, 8]);
        assert_eq!(result.system_matrix.strides(), &[1, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.scattering.strides(), &[1, 8, 64]);
        assert_eq!(result.multiple_scattering_order, 4);
        assert_complex32_close(
            matrix_sum(result.system_matrix.view()),
            Complex32::new(0.090_419_99, 0.516_9),
        );
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            Complex32::new(-2.944_321_6, 4.799_405),
        );
        assert_complex32_close(
            result.scattering[(0, 0, 0)],
            Complex32::new(-0.007_797_049_4, -0.003_244_209),
        );
        assert_complex32_close(
            result.scattering[(1, 3, 0)],
            Complex32::new(-0.065_967_47, 0.044_093_188),
        );
        assert_complex32_close(
            result.scattering[(6, 7, 0)],
            Complex32::new(-0.096_285_895, 0.140_520_08),
        );
        Ok(())
    }

    #[test]
    fn fms_tfqmr_scattering_matches_feff_ggtf_reference() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_tfqmr_scattering(FmsTfqmrInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &state_set.representative_offsets,
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
            calculated_l: &[true, true],
            convergence_tolerance: 1.0e-5,
            zero_tolerance: 0.0,
        })?;

        assert_eq!(result.system_matrix.shape(), &[8, 8]);
        assert_eq!(result.system_matrix.strides(), &[1, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.scattering.strides(), &[1, 8, 64]);
        assert_eq!(result.multiple_scattering_order, 4);
        assert_complex32_close(
            matrix_sum(result.system_matrix.view()),
            Complex32::new(7.909_579_3, -0.516_9),
        );
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            Complex32::new(-2.944_320_7, 4.799_402_7),
        );
        assert_complex32_close(
            result.scattering[(0, 0, 0)],
            Complex32::new(-0.007_797_021_4, -0.003_244_287_3),
        );
        assert_complex32_close(
            result.scattering[(1, 3, 0)],
            Complex32::new(-0.065_967_43, 0.044_093_173),
        );
        assert_complex32_close(
            result.scattering[(6, 7, 0)],
            Complex32::new(-0.096_285_91, 0.140_520_1),
        );
        Ok(())
    }

    #[test]
    fn fms_lu_scattering_matches_feff_gglu_reference() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_lu_scattering(FmsLuInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &state_set.representative_offsets,
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
        })?;

        assert_eq!(result.system_matrix.shape(), &[8, 8]);
        assert_eq!(result.system_matrix.strides(), &[1, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.scattering.strides(), &[1, 8, 64]);
        assert_complex32_close(
            matrix_sum(result.system_matrix.view()),
            Complex32::new(8.107_28, -0.542_959_87),
        );
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            Complex32::new(-2.944_320_4, 4.799_401_3),
        );
        assert_complex32_close(
            result.scattering[(0, 0, 0)],
            Complex32::new(-0.007_797_020_5, -0.003_244_286_6),
        );
        assert_complex32_close(
            result.scattering[(1, 3, 0)],
            Complex32::new(-0.065_967_42, 0.044_093_15),
        );
        assert_complex32_close(
            result.scattering[(6, 7, 0)],
            Complex32::new(-0.096_285_9, 0.140_520_07),
        );
        Ok(())
    }

    #[test]
    fn fms_lu_scattering_rejects_missing_representative() -> Result<(), Box<dyn Error>> {
        let state_set = construct_state_kets(2, &[0], &[1], 1)?;
        let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

        let result = fms_lu_scattering(FmsLuInput {
            states: &state_set.states,
            spin_channels: 2,
            global_lmax: 1,
            potential_lmax: &[1],
            representative_offsets: &[None],
            potential_start: 0,
            potential_end: 0,
            free_propagator: free_propagator.view(),
            t_matrix: t_matrix.view(),
        });

        assert!(matches!(
            result,
            Err(FmsError::MissingRepresentativePotential { potential: 0 })
        ));
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

    fn reference_phase_shifts() -> Array3<Complex32> {
        let mut phases = Array3::zeros((2, 5, 2).f());
        phases[(0, 4, 1)] = Complex32::new(0.2, 0.05);
        phases[(0, 0, 1)] = Complex32::new(-0.1, 0.03);
        phases[(1, 4, 1)] = Complex32::new(0.15, -0.02);
        phases[(1, 0, 1)] = Complex32::new(0.07, 0.04);
        phases
    }

    fn reference_gglu_inputs(state_count: usize) -> (Array2<Complex32>, Array2<Complex32>) {
        let mut free_propagator = Array2::zeros((state_count, state_count).f());
        let mut t_matrix = Array2::zeros((2, state_count).f());
        for column in 0..state_count {
            for row in 0..state_count {
                let row_feff = row as f32 + 1.0;
                let column_feff = column as f32 + 1.0;
                if row != column {
                    free_propagator[(row, column)] = Complex32::new(
                        0.01 * row_feff - 0.02 * column_feff,
                        0.015 * row_feff + 0.005 * column_feff,
                    );
                }
            }
            let column_feff = column as f32 + 1.0;
            t_matrix[(0, column)] = Complex32::new(0.02 * column_feff, -0.01 * column_feff);
            t_matrix[(1, column)] = Complex32::new(-0.005 * column_feff, 0.003 * column_feff);
        }
        (free_propagator, t_matrix)
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

    fn scattering_sum(table: ArrayView3<'_, Complex32>) -> Complex32 {
        table
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
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
