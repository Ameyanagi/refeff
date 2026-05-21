//! Full multiple-scattering helpers.
//!
//! FEFF's FMS routines use Rehr-Albers polynomial tables when building
//! multiple-scattering propagators. The helpers here keep the legacy table
//! layout explicit while returning Rust-owned `ndarray` storage.

use ndarray::{
    Array2, Array3, Array4, Array5, Array6, ArrayView2, ArrayView3, ArrayView4, ArrayView6, Axis,
    ShapeBuilder,
};
use num_complex::Complex32;
use refeff_linalg::{complex32_lu_factor, complex32_lu_solve};

use crate::{
    Real,
    angular::SpinOrbitCouplingTables,
    state::{StateKet, StateKetError, construct_state_kets_with_limit},
};

const FMS_ROTATION_LMAX: usize = 24;

mod internals;
mod mkgtr;
pub use mkgtr::{MkgtrGreenTraceInput, MkgtrGreenTraceResult, mkgtr_green_trace};

use internals::*;

mod types;
pub use types::*;

/// Port the setup prelude in FEFF `fmspack.f90`.
///
/// This performs the non-numerical work before `fmspack` allocates the solver
/// matrices: `lipotx` values are clamped to `0..=lx` with negative values
/// replaced by `lx`, the active `gg` potential range is selected from `lfms`,
/// FEFF `getkts` state kets are generated, and every requested potential is
/// checked for a representative state offset.
pub fn fms_driver_setup(input: FmsDriverSetupInput<'_>) -> Result<FmsDriverSetup, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.atoms.is_empty() {
        return Err(FmsError::EmptyCluster);
    }
    if input.max_potential >= input.raw_potential_lmax.len() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "lipotx",
            axis: "potential",
            index: input.max_potential,
        });
    }

    let potential_count = input
        .max_potential
        .checked_add(1)
        .ok_or(FmsError::IntegerOverflow {
            field: "max_potential",
            value: input.max_potential,
        })?;
    let potential_lmax = input
        .raw_potential_lmax
        .iter()
        .take(potential_count)
        .map(|&lmax| clamp_fms_lipotx(lmax, input.global_lmax))
        .collect::<Vec<_>>();

    let atom_potentials = input
        .atoms
        .iter()
        .map(|atom| checked_potential(atom.potential, input.max_potential))
        .collect::<Result<Vec<_>, _>>()?;
    let absorber_potential = atom_potentials
        .first()
        .copied()
        .ok_or(FmsError::EmptyCluster)?;
    let (potential_start, potential_end) = if input.lfms == 0 {
        (absorber_potential, absorber_potential)
    } else {
        (0, input.max_potential)
    };

    let state_kets = construct_state_kets_with_limit(
        input.spin_channels,
        &atom_potentials,
        &potential_lmax,
        input.global_lmax,
        input.state_capacity,
    )
    .map_err(fms_state_ket_error)?;

    for potential in potential_start..=potential_end {
        representative_offset(&state_kets.representative_offsets, potential)?;
    }

    Ok(FmsDriverSetup {
        potential_lmax,
        potential_start,
        potential_end,
        state_kets,
    })
}

/// Select the FEFF FMS scattering branch for a raw `minv` value.
///
/// FEFF dispatches `minv=0` to LU, `1` to BiCGStab/VdV, `2` to recursion,
/// `3` to Graves-Morris/Salam, and every other value to TFQMR. When a full
/// scattering matrix is requested, FEFF forces all non-LU choices back to LU.
pub fn fms_scattering_method_selection(
    minv: i32,
    full_scattering_matrix_requested: bool,
) -> FmsScatteringMethodSelection {
    let forced_lu_for_full_scattering = full_scattering_matrix_requested && minv != 0;
    let effective_minv = if forced_lu_for_full_scattering {
        0
    } else {
        minv
    };
    let method = match effective_minv {
        0 => FmsScatteringMethod::Lu,
        1 => FmsScatteringMethod::BiCgStab,
        2 => FmsScatteringMethod::Recursion,
        3 => FmsScatteringMethod::GravesMorris,
        _ => FmsScatteringMethod::Tfqmr,
    };

    FmsScatteringMethodSelection {
        effective_minv,
        method,
        forced_lu_for_full_scattering,
    }
}

/// Assemble and solve one real-space FEFF FMS energy point.
///
/// This wires the top-level `fmspack` sequence for real-space FMS after
/// `xprep` has prepared geometry tables: setup state kets, build spin-resolved
/// `xrho`/`xclm`, assemble `g0`, build the compact T-matrix, normalize `minv`,
/// and dispatch the selected scattering solver.
pub fn fms_real_space_energy(
    input: FmsRealSpaceEnergyInput<'_>,
) -> Result<FmsRealSpaceEnergyResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.wave_numbers.len() != input.spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "ck",
            expected: input.spin_channels,
            actual: input.wave_numbers.len(),
        });
    }
    if input.phase_shifts.shape()[0] != input.spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "xphase",
            expected: input.spin_channels,
            actual: input.phase_shifts.shape()[0],
        });
    }

    let setup = fms_driver_setup(FmsDriverSetupInput {
        lfms: input.lfms,
        spin_channels: input.spin_channels,
        atoms: input.atoms,
        max_potential: input.max_potential,
        global_lmax: input.global_lmax,
        raw_potential_lmax: input.raw_potential_lmax,
        state_capacity: input.state_capacity,
    })?;
    let pair_tables = fms_spin_pair_tables(input.global_lmax, input.wave_numbers, input.atoms)?;
    let free_propagator = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
        states: &setup.state_kets.states,
        atoms: input.atoms,
        direct_cutoff: input.direct_cutoff,
        rho: pair_tables.rho.view(),
        wave_numbers: input.wave_numbers,
        mean_square_displacements: input.mean_square_displacements,
        xclm: pair_tables.polynomials.view(),
        xnlm: input.xnlm,
        rotations: input.rotations,
    })?;
    let t_matrix = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &setup.state_kets.states,
        atoms: input.atoms,
        spin_channels: input.spin_channels,
        spin_selector: input.spin_selector,
        phase_shifts: input.phase_shifts,
        spin_orbit: input.spin_orbit,
    })?;
    let method_selection =
        fms_scattering_method_selection(input.minv, input.full_scattering_matrix_requested);
    let scattering = fms_scattering(FmsScatteringInput {
        method: method_selection.method,
        calculate_full_scattering: input.full_scattering_matrix_requested,
        states: &setup.state_kets.states,
        spin_channels: input.spin_channels,
        global_lmax: input.global_lmax,
        potential_lmax: &setup.potential_lmax,
        representative_offsets: &setup.state_kets.representative_offsets,
        potential_start: setup.potential_start,
        potential_end: setup.potential_end,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: input.calculated_l,
        convergence_tolerance: input.convergence_tolerance,
        zero_tolerance: input.zero_tolerance,
    })?;

    Ok(FmsRealSpaceEnergyResult {
        setup,
        method_selection,
        pair_tables,
        free_propagator,
        t_matrix,
        scattering,
    })
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

/// Port FEFF `yprep` absorber-centered FMS cluster-prefix selection.
///
/// The helper finds the first atom with `central_potential`, shifts all
/// coordinates so that atom is at the origin, sorts by FEFF's `athep` radial
/// key, counts the atoms inside `cluster_radius`, and truncates that prefix to
/// `cluster_capacity`. Rotation matrices and spherical-harmonic normalization
/// tables are prepared by separate FMS helpers.
pub fn fms_yprep_cluster(input: FmsYprepClusterInput<'_>) -> Result<FmsYprepCluster, FmsError> {
    let (rows, columns) = input.positions.dim();
    if columns != 3 {
        return Err(FmsError::AtomPositionColumnCount { columns });
    }
    if rows != input.potentials.len() {
        return Err(FmsError::AtomCountMismatch {
            potentials: input.potentials.len(),
            positions: rows,
        });
    }
    if !input.cluster_radius.is_finite() || input.cluster_radius < 0.0 {
        return Err(FmsError::InvalidClusterRadius);
    }
    if input.cluster_capacity == 0 {
        return Err(FmsError::EmptyClusterCapacity);
    }

    let mut central_atom = None;
    for (index, &potential) in input.potentials.iter().enumerate() {
        if potential == input.central_potential {
            if input.central_potential == 0 && central_atom.is_some() {
                return Err(FmsError::DuplicateAbsorber);
            }
            central_atom.get_or_insert(index);
        }
    }
    let central_atom = central_atom.ok_or(FmsError::MissingCentralAtom {
        potential: input.central_potential,
    })?;

    let center = [
        input.positions[(central_atom, 0)],
        input.positions[(central_atom, 1)],
        input.positions[(central_atom, 2)],
    ];
    ensure_finite_position(central_atom, center)?;

    let mut atoms = Vec::with_capacity(rows);
    for (atom, &potential) in input.potentials.iter().enumerate() {
        let position = [
            input.positions[(atom, 0)] - center[0],
            input.positions[(atom, 1)] - center[1],
            input.positions[(atom, 2)] - center[2],
        ];
        ensure_finite_position(atom, position)?;
        atoms.push(FmsAtom {
            position,
            potential,
        });
    }
    sort_atoms_by_radius(&mut atoms)?;

    let radius_squared = input.cluster_radius * input.cluster_radius;
    let first_outside = atoms
        .iter()
        .position(|atom| {
            let [x, y, z] = atom.position;
            x * x + y * y + z * z > radius_squared
        })
        .map_or(atoms.len(), |index| index);
    let untruncated_count = if first_outside == 0 {
        atoms.len()
    } else {
        first_outside
    };
    let included_count = untruncated_count.min(input.cluster_capacity);
    atoms.truncate(included_count);

    Ok(FmsYprepCluster {
        central_atom,
        untruncated_count,
        atoms,
    })
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

/// Build FEFF `yprep` pair azimuths and FMS rotation tables.
///
/// For each ordered atom pair, this runs the same `getang`/`rotxan` sequence as
/// `FMS/yprep.f90`: `xphi(atom2,atom1)` is recorded for all pairs, diagonal
/// rotations remain zero, and off-diagonal pairs receive forward (`k=0`) and
/// backward (`k=1`) rotation tables.
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

    for atom2 in 0..atom_count {
        for atom1 in 0..atom_count {
            let (beta, pair_phi) = pair_polar_angles(&positions, atom2, atom1)?;
            phi[(atom2, atom1)] = pair_phi;
            if atom2 == atom1 {
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

/// Build FEFF spin-resolved `xrho` and `xclm` pair tables.
///
/// FEFF stores these tables with a trailing spin index and evaluates the
/// Rehr-Albers polynomial table separately for each `ck(isp)`. This helper
/// preserves the same layout while reusing [`fms_pair_tables`] for each spin.
pub fn fms_spin_pair_tables(
    lmax: usize,
    wave_numbers: &[Complex32],
    atoms: &[FmsAtom],
) -> Result<FmsSpinPairTables, FmsError> {
    ensure_spin_channels(wave_numbers.len())?;
    let angular_len = lmax.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lmax",
        value: lmax,
        lx: lmax,
    })?;
    let atom_count = atoms.len();
    let mut rho = Array3::zeros((atom_count, atom_count, wave_numbers.len()).f());
    let mut polynomials = Array5::zeros(
        (
            angular_len,
            angular_len,
            atom_count,
            atom_count,
            wave_numbers.len(),
        )
            .f(),
    );

    for (spin, &wave_number) in wave_numbers.iter().enumerate() {
        let tables = fms_pair_tables(lmax, wave_number, atoms)?;
        for atom2 in 0..atom_count {
            for atom1 in 0..atom_count {
                rho[(atom2, atom1, spin)] = tables.rho[(atom2, atom1)];
                for l in 0..angular_len {
                    for m in 0..angular_len {
                        polynomials[(m, l, atom2, atom1, spin)] =
                            tables.polynomials[(m, l, atom2, atom1)];
                    }
                }
            }
        }
    }

    Ok(FmsSpinPairTables { rho, polynomials })
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

/// Build FEFF's spin-resolved off-diagonal FMS free-propagator matrix `g0`.
///
/// This is the spin-aware form of [`fms_free_propagator_matrix`]. It matches
/// FEFF's `fmspack` loop by selecting `ck(isp)` and `xclm(...,isp)` from the
/// row state's spin channel when same-spin states are coupled.
pub fn fms_spin_free_propagator_matrix(
    input: FmsSpinFreePropagatorMatrixInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    ensure_spin_channels(input.wave_numbers.len())?;
    if !input.direct_cutoff.is_finite() || input.direct_cutoff < 0.0 {
        return Err(FmsError::InvalidDirectCutoff);
    }
    for (spin, &wave_number) in input.wave_numbers.iter().enumerate() {
        if !(wave_number.re.is_finite() && wave_number.im.is_finite()) {
            return Err(FmsError::NonFiniteWaveNumber);
        }
        ensure_axis_len("xrho", "spin", input.rho.shape()[2], spin)?;
        ensure_axis_len("xclm", "spin", input.xclm.shape()[4], spin)?;
    }
    for (index, atom) in input.atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
    }

    let cutoff_squared = input.direct_cutoff * input.direct_cutoff;
    let mut matrix = Array2::zeros((input.states.len(), input.states.len()).f());
    for (row, &first) in input.states.iter().enumerate() {
        ensure_state_spin(first.spin, input.wave_numbers.len())?;
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

            let spin = first.spin - 1;
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
                rho: input.rho[(atom2, atom1, spin)],
                wave_number: input.wave_numbers[spin],
                mean_square_displacement: input.mean_square_displacements[(atom2, atom1)],
                xclm: input.xclm.index_axis(Axis(4), spin),
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

/// Dispatch FEFF's compact FMS scattering branches.
///
/// This mirrors the final `minv` branch in `fmspack.f90` after setup and
/// matrix assembly are complete. The LU branch ignores iterative tolerances
/// and `lcalc`, while iterative branches return FEFF's reported
/// multiple-scattering order in [`FmsScatteringResult::multiple_scattering_order`].
pub fn fms_scattering(input: FmsScatteringInput<'_>) -> Result<FmsScatteringResult, FmsError> {
    match input.method {
        FmsScatteringMethod::Lu => {
            let result = fms_lu_scattering(FmsLuInput {
                states: input.states,
                calculate_full_scattering: input.calculate_full_scattering,
                spin_channels: input.spin_channels,
                global_lmax: input.global_lmax,
                potential_lmax: input.potential_lmax,
                representative_offsets: input.representative_offsets,
                potential_start: input.potential_start,
                potential_end: input.potential_end,
                free_propagator: input.free_propagator,
                t_matrix: input.t_matrix,
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: result.full_scattering,
                multiple_scattering_order: None,
            })
        }
        FmsScatteringMethod::BiCgStab => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_bicgstab_scattering(FmsBiCgStabInput {
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
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
        FmsScatteringMethod::Recursion => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_recursion_scattering(FmsRecursionInput {
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
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
        FmsScatteringMethod::GravesMorris => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_graves_morris_scattering(FmsGravesMorrisInput {
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
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
        FmsScatteringMethod::Tfqmr => {
            if input.calculate_full_scattering {
                return Err(FmsError::FullScatteringRequiresLu {
                    method: input.method,
                });
            }
            let result = fms_tfqmr_scattering(FmsTfqmrInput {
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
            })?;
            Ok(FmsScatteringResult {
                method: input.method,
                system_matrix: result.system_matrix,
                scattering: result.scattering,
                full_scattering: None,
                multiple_scattering_order: Some(result.multiple_scattering_order),
            })
        }
    }
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

    let full_scattering = if input.calculate_full_scattering {
        Some(complex32_lu_solve(&lu, input.free_propagator)?)
    } else {
        None
    };

    Ok(FmsLuResult {
        system_matrix,
        scattering,
        full_scattering,
    })
}

/// Port of FEFF `gglufullpot`: LU FMS scattering with a full T-matrix.
///
/// FEFF's full-potential branch accepts `tmatrx(state,state)` rather than the
/// compact spin-band table used by [`fms_lu_scattering`]. The assembled work
/// matrix follows the original `gglufullpot` diagonal assignment before the
/// pure-Rust LU solve.
pub fn fms_full_potential_lu_scattering(
    input: FmsFullPotentialLuInput<'_>,
) -> Result<FmsFullPotentialLuResult, FmsError> {
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
    ensure_square_table("tmatrx", input.t_matrix, input.states.len())?;
    for &state in input.states {
        ensure_state_spin(state.spin, input.spin_channels)?;
    }

    let system_matrix =
        fms_full_potential_lu_system_matrix(input.states, input.free_propagator, input.t_matrix)?;
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

    Ok(FmsFullPotentialLuResult {
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

#[cfg(test)]
mod tests;
