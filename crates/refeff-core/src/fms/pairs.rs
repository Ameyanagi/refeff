use super::*;
use ndarray::ArrayView5;

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

    let state_count = input.states.len();
    if state_count == 0 {
        return Ok(Array2::zeros((0, 0).f()));
    }
    let limits = propagator_state_limits(input.states, input.atoms.len(), None)?;
    validate_free_propagator_tables(
        limits,
        input.rho.shape(),
        input.mean_square_displacements.shape(),
        Some(input.xclm.shape()),
        None,
        input.xnlm.shape(),
    )?;
    let rotation_lmax = validate_rotation_matrix_tables(input.rotations, limits)?;
    let states = cache_propagator_states(input.states, input.atoms.len(), None, rotation_lmax)?;
    let rho = IndexedView2::new(input.rho);
    let sigsqr = IndexedView2::new(input.mean_square_displacements);
    let xclm = IndexedView4::new(input.xclm);
    let xnlm = validated_normalization_table(input.xnlm, limits.max_l)?;
    let rotations = IndexedView6::new(input.rotations);
    let cutoff_squared = input.direct_cutoff * input.direct_cutoff;
    let mut matrix = Array2::<Complex32>::zeros((state_count, state_count).f());
    for (row, first) in states.iter().enumerate() {
        for (column, second) in states.iter().enumerate() {
            if first.atom_index == second.atom_index || first.state.spin != second.state.spin {
                continue;
            }

            let distance_squared = fms_atom_distance_squared(
                input.atoms[first.atom_index].position,
                input.atoms[second.atom_index].position,
            );
            if distance_squared > cutoff_squared {
                continue;
            }

            let value = fms_free_propagator_element_fast4(FmsFastPropagatorInput4 {
                first: *first,
                second: *second,
                rho: rho.get(second.atom_index, first.atom_index),
                wave_number: input.wave_number,
                mean_square_displacement: sigsqr.get(second.atom_index, first.atom_index),
                xclm: &xclm,
                xnlm: &xnlm,
                rotations: &rotations,
                rotation_lmax,
            })?;
            matrix[[row, column]] = value;
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

    let state_count = input.states.len();
    if state_count == 0 {
        return Ok(Array2::zeros((0, 0).f()));
    }
    let limits = propagator_state_limits(
        input.states,
        input.atoms.len(),
        Some(input.wave_numbers.len()),
    )?;
    validate_free_propagator_tables(
        limits,
        &input.rho.shape()[..2],
        input.mean_square_displacements.shape(),
        None,
        Some(input.xclm.shape()),
        input.xnlm.shape(),
    )?;
    let rotation_lmax = validate_rotation_matrix_tables(input.rotations, limits)?;
    let states = cache_propagator_states(
        input.states,
        input.atoms.len(),
        Some(input.wave_numbers.len()),
        rotation_lmax,
    )?;
    let rho = IndexedView3::new(input.rho);
    let sigsqr = IndexedView2::new(input.mean_square_displacements);
    let xclm = IndexedView5::new(input.xclm);
    let xnlm = validated_normalization_table(input.xnlm, limits.max_l)?;
    let rotations = IndexedView6::new(input.rotations);
    let cutoff_squared = input.direct_cutoff * input.direct_cutoff;
    let mut matrix = Array2::<Complex32>::zeros((state_count, state_count).f());
    for (row, first) in states.iter().enumerate() {
        for (column, second) in states.iter().enumerate() {
            if first.atom_index == second.atom_index || first.state.spin != second.state.spin {
                continue;
            }

            let distance_squared = fms_atom_distance_squared(
                input.atoms[first.atom_index].position,
                input.atoms[second.atom_index].position,
            );
            if distance_squared > cutoff_squared {
                continue;
            }

            let spin = first.state.spin - 1;
            let value = fms_free_propagator_element_fast5(FmsFastPropagatorInput5 {
                first: *first,
                second: *second,
                spin,
                rho: rho.get(second.atom_index, first.atom_index, spin),
                wave_number: input.wave_numbers[spin],
                mean_square_displacement: sigsqr.get(second.atom_index, first.atom_index),
                xclm: &xclm,
                xnlm: &xnlm,
                rotations: &rotations,
                rotation_lmax,
            })?;
            matrix[[row, column]] = value;
        }
    }

    Ok(matrix)
}

#[derive(Clone, Copy)]
struct CachedPropagatorState {
    state: StateKet,
    atom_index: usize,
    magnetic_index: usize,
}

#[derive(Clone, Copy)]
struct PropagatorStateLimits {
    max_l: usize,
    max_atom: usize,
}

struct FmsFastPropagatorInput4<'tables, 'data> {
    first: CachedPropagatorState,
    second: CachedPropagatorState,
    rho: Complex32,
    wave_number: Complex32,
    mean_square_displacement: f32,
    xclm: &'tables IndexedView4<'data, Complex32>,
    xnlm: &'tables NormalizationTable,
    rotations: &'tables IndexedView6<'data, Complex32>,
    rotation_lmax: usize,
}

struct FmsFastPropagatorInput5<'tables, 'data> {
    first: CachedPropagatorState,
    second: CachedPropagatorState,
    spin: usize,
    rho: Complex32,
    wave_number: Complex32,
    mean_square_displacement: f32,
    xclm: &'tables IndexedView5<'data, Complex32>,
    xnlm: &'tables NormalizationTable,
    rotations: &'tables IndexedView6<'data, Complex32>,
    rotation_lmax: usize,
}

#[inline(always)]
fn fms_free_propagator_element_fast4(
    input: FmsFastPropagatorInput4<'_, '_>,
) -> Result<Complex32, FmsError> {
    validate_fast_propagator_scalars(input.rho, input.wave_number, input.mean_square_displacement)?;

    let l1 = input.first.state.angular_momentum;
    let l2 = input.second.state.angular_momentum;
    let l1_signed = isize::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
        name: "l1",
        value: l1,
        lx: l1,
    })?;

    let mut sum = Complex32::new(0.0, 0.0);
    for mu in -l1_signed..=l1_signed {
        let gllmz = rehr_albers_z_axis_propagator_fast4(
            mu.unsigned_abs(),
            input.first,
            input.second,
            input.xclm,
            input.xnlm,
        )?;
        let mu_index = signed_mu_index(mu, input.rotation_lmax);
        let backward = input.rotations.get(
            mu_index,
            input.first.magnetic_index,
            l1,
            1,
            input.second.atom_index,
            input.first.atom_index,
        );
        let forward = input.rotations.get(
            input.second.magnetic_index,
            mu_index,
            l2,
            0,
            input.second.atom_index,
            input.first.atom_index,
        );
        sum += backward * gllmz * forward;
    }

    Ok(
        fms_free_propagator_prefactor(input.rho, input.wave_number, input.mean_square_displacement)
            * sum,
    )
}

#[inline(always)]
fn fms_free_propagator_element_fast5(
    input: FmsFastPropagatorInput5<'_, '_>,
) -> Result<Complex32, FmsError> {
    validate_fast_propagator_scalars(input.rho, input.wave_number, input.mean_square_displacement)?;

    let l1 = input.first.state.angular_momentum;
    let l2 = input.second.state.angular_momentum;
    let l1_signed = isize::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
        name: "l1",
        value: l1,
        lx: l1,
    })?;

    let mut sum = Complex32::new(0.0, 0.0);
    for mu in -l1_signed..=l1_signed {
        let gllmz = rehr_albers_z_axis_propagator_fast5(
            mu.unsigned_abs(),
            input.first,
            input.second,
            input.spin,
            input.xclm,
            input.xnlm,
        )?;
        let mu_index = signed_mu_index(mu, input.rotation_lmax);
        let backward = input.rotations.get(
            mu_index,
            input.first.magnetic_index,
            l1,
            1,
            input.second.atom_index,
            input.first.atom_index,
        );
        let forward = input.rotations.get(
            input.second.magnetic_index,
            mu_index,
            l2,
            0,
            input.second.atom_index,
            input.first.atom_index,
        );
        sum += backward * gllmz * forward;
    }

    Ok(
        fms_free_propagator_prefactor(input.rho, input.wave_number, input.mean_square_displacement)
            * sum,
    )
}

fn validate_fast_propagator_scalars(
    rho: Complex32,
    wave_number: Complex32,
    mean_square_displacement: f32,
) -> Result<(), FmsError> {
    if !(rho.re.is_finite() && rho.im.is_finite()) {
        return Err(FmsError::NonFiniteRho);
    }
    if rho == Complex32::new(0.0, 0.0) {
        return Err(FmsError::ZeroRho);
    }
    if !(wave_number.re.is_finite() && wave_number.im.is_finite()) {
        return Err(FmsError::NonFiniteWaveNumber);
    }
    if !mean_square_displacement.is_finite() {
        return Err(FmsError::NonFiniteMeanSquareDisplacement);
    }
    Ok(())
}

#[inline(always)]
fn rehr_albers_z_axis_propagator_fast4(
    mu: usize,
    first: CachedPropagatorState,
    second: CachedPropagatorState,
    xclm: &IndexedView4<'_, Complex32>,
    xnlm: &NormalizationTable,
) -> Result<Complex32, FmsError> {
    let l1 = first.state.angular_momentum;
    let l2 = second.state.angular_momentum;
    if mu > l2 {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let norm_l1 = xnlm.get(mu, l1);
    let norm_l2 = xnlm.get(mu, l2);
    let angular_weight = xnlm.angular_weight(l1);
    let sign = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };
    let numax = l1.min(l2 - mu);

    let mut sum = Complex32::new(0.0, 0.0);
    for nu in 0..=numax {
        let mn = mu.checked_add(nu).ok_or(FmsError::InvalidAngularLimit {
            name: "mu",
            value: mu,
            lx: l2,
        })?;
        let gamtl =
            angular_weight * xclm.get(nu, l1, second.atom_index, first.atom_index) / norm_l1;
        let gam = xclm.get(mn, l2, second.atom_index, first.atom_index) * (sign * norm_l2);
        sum += gamtl * gam;
    }

    Ok(sum)
}

#[inline(always)]
fn rehr_albers_z_axis_propagator_fast5(
    mu: usize,
    first: CachedPropagatorState,
    second: CachedPropagatorState,
    spin: usize,
    xclm: &IndexedView5<'_, Complex32>,
    xnlm: &NormalizationTable,
) -> Result<Complex32, FmsError> {
    let l1 = first.state.angular_momentum;
    let l2 = second.state.angular_momentum;
    if mu > l2 {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let norm_l1 = xnlm.get(mu, l1);
    let norm_l2 = xnlm.get(mu, l2);
    let angular_weight = xnlm.angular_weight(l1);
    let sign = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };
    let numax = l1.min(l2 - mu);

    let mut sum = Complex32::new(0.0, 0.0);
    for nu in 0..=numax {
        let mn = mu.checked_add(nu).ok_or(FmsError::InvalidAngularLimit {
            name: "mu",
            value: mu,
            lx: l2,
        })?;
        let gamtl =
            angular_weight * xclm.get(nu, l1, second.atom_index, first.atom_index, spin) / norm_l1;
        let gam = xclm.get(mn, l2, second.atom_index, first.atom_index, spin) * (sign * norm_l2);
        sum += gamtl * gam;
    }

    Ok(sum)
}

struct NormalizationTable {
    values: Vec<f32>,
    angular_weights: Vec<Complex32>,
    stride: usize,
}

impl NormalizationTable {
    #[inline(always)]
    fn get(&self, mu: usize, angular_momentum: usize) -> f32 {
        self.values[mu * self.stride + angular_momentum]
    }

    #[inline(always)]
    fn angular_weight(&self, angular_momentum: usize) -> Complex32 {
        self.angular_weights[angular_momentum]
    }
}

fn validated_normalization_table(
    xnlm: ArrayView2<'_, Real>,
    max_l: usize,
) -> Result<NormalizationTable, FmsError> {
    let stride = max_l.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lmax",
        value: max_l,
        lx: max_l,
    })?;
    let view = IndexedView2::new(xnlm);
    let mut values = Vec::with_capacity(stride * stride);
    let mut angular_weights = Vec::with_capacity(stride);
    for angular_momentum in 0..=max_l {
        let weight = angular_momentum
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "angular_momentum",
                value: angular_momentum,
                lx: max_l,
            })?;
        angular_weights.push(Complex32::new(weight as f32, 0.0));
    }
    for mu in 0..=max_l {
        for angular_momentum in 0..=max_l {
            let value = view.get(mu, angular_momentum) as f32;
            if mu <= angular_momentum && (!value.is_finite() || value == 0.0) {
                return Err(FmsError::InvalidNormalization {
                    mu,
                    angular_momentum,
                });
            }
            values.push(value);
        }
    }
    Ok(NormalizationTable {
        values,
        angular_weights,
        stride,
    })
}

fn propagator_state_limits(
    states: &[StateKet],
    atom_count: usize,
    spin_channels: Option<usize>,
) -> Result<PropagatorStateLimits, FmsError> {
    let mut max_l = 0;
    let mut max_atom = 0;
    for &state in states {
        if let Some(spin_channels) = spin_channels {
            ensure_state_spin(state.spin, spin_channels)?;
        }
        let atom_index = checked_atom_index(state.atom)?;
        ensure_atom_table_index(atom_index, atom_count)?;
        max_l = max_l.max(state.angular_momentum);
        max_atom = max_atom.max(atom_index);
    }
    Ok(PropagatorStateLimits { max_l, max_atom })
}

fn validate_free_propagator_tables(
    limits: PropagatorStateLimits,
    rho_shape: &[usize],
    sigsqr_shape: &[usize],
    xclm4_shape: Option<&[usize]>,
    xclm5_shape: Option<&[usize]>,
    xnlm_shape: &[usize],
) -> Result<(), FmsError> {
    ensure_axis_len("xrho", "atom2", rho_shape[0], limits.max_atom)?;
    ensure_axis_len("xrho", "atom1", rho_shape[1], limits.max_atom)?;
    ensure_axis_len("sigsqr", "atom2", sigsqr_shape[0], limits.max_atom)?;
    ensure_axis_len("sigsqr", "atom1", sigsqr_shape[1], limits.max_atom)?;
    if let Some(shape) = xclm4_shape {
        ensure_axis_len("xclm", "m", shape[0], limits.max_l)?;
        ensure_axis_len("xclm", "l", shape[1], limits.max_l)?;
        ensure_axis_len("xclm", "atom2", shape[2], limits.max_atom)?;
        ensure_axis_len("xclm", "atom1", shape[3], limits.max_atom)?;
    }
    if let Some(shape) = xclm5_shape {
        ensure_axis_len("xclm", "m", shape[0], limits.max_l)?;
        ensure_axis_len("xclm", "l", shape[1], limits.max_l)?;
        ensure_axis_len("xclm", "atom2", shape[2], limits.max_atom)?;
        ensure_axis_len("xclm", "atom1", shape[3], limits.max_atom)?;
    }
    ensure_axis_len("xnlm", "mu", xnlm_shape[0], limits.max_l)?;
    ensure_axis_len("xnlm", "l", xnlm_shape[1], limits.max_l)?;
    Ok(())
}

fn validate_rotation_matrix_tables(
    rotations: ArrayView6<'_, Complex32>,
    limits: PropagatorStateLimits,
) -> Result<usize, FmsError> {
    let shape = rotations.shape();
    if shape[0] == 0 || shape[0] != shape[1] || shape[0].is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: "rotations",
            value: shape[0],
            lx: shape[0],
        });
    }
    let rotation_lmax = (shape[0] - 1) / 2;
    if limits.max_l > rotation_lmax {
        return Err(FmsError::InvalidAngularLimit {
            name: "rotations",
            value: shape[0],
            lx: limits.max_l,
        });
    }
    ensure_axis_len("rotations", "l", shape[2], limits.max_l)?;
    ensure_axis_len("rotations", "k", shape[3], 1)?;
    ensure_axis_len("rotations", "atom2", shape[4], limits.max_atom)?;
    ensure_axis_len("rotations", "atom1", shape[5], limits.max_atom)?;
    Ok(rotation_lmax)
}

fn cache_propagator_states(
    states: &[StateKet],
    atom_count: usize,
    spin_channels: Option<usize>,
    rotation_lmax: usize,
) -> Result<Vec<CachedPropagatorState>, FmsError> {
    let rotation_axis_len = 2 * rotation_lmax + 1;
    let mut cached = Vec::with_capacity(states.len());
    for &state in states {
        if let Some(spin_channels) = spin_channels {
            ensure_state_spin(state.spin, spin_channels)?;
        }
        let atom_index = checked_atom_index(state.atom)?;
        ensure_atom_table_index(atom_index, atom_count)?;
        let magnetic_index = signed_magnetic_index(state.magnetic, rotation_lmax)?;
        ensure_axis_len("rotations", "m", rotation_axis_len, magnetic_index)?;
        cached.push(CachedPropagatorState {
            state,
            atom_index,
            magnetic_index,
        });
    }
    Ok(cached)
}

#[inline(always)]
fn signed_mu_index(mu: isize, rotation_lmax: usize) -> usize {
    (mu + rotation_lmax as isize) as usize
}

enum IndexedView2<'a, T: Copy> {
    Strided {
        values: &'a [T],
        strides: [isize; 2],
    },
    View(ArrayView2<'a, T>),
}

impl<'a, T: Copy> IndexedView2<'a, T> {
    fn new(view: ArrayView2<'a, T>) -> Self {
        let strides = [view.strides()[0], view.strides()[1]];
        match view.to_slice_memory_order() {
            Some(values) => Self::Strided { values, strides },
            None => Self::View(view),
        }
    }

    #[inline(always)]
    fn get(&self, first: usize, second: usize) -> T {
        match self {
            Self::Strided { values, strides } => values[strided_offset(*strides, [first, second])],
            Self::View(view) => (*view)[(first, second)],
        }
    }
}

enum IndexedView3<'a, T: Copy> {
    Strided {
        values: &'a [T],
        strides: [isize; 3],
    },
    View(ArrayView3<'a, T>),
}

impl<'a, T: Copy> IndexedView3<'a, T> {
    fn new(view: ArrayView3<'a, T>) -> Self {
        let strides = [view.strides()[0], view.strides()[1], view.strides()[2]];
        match view.to_slice_memory_order() {
            Some(values) => Self::Strided { values, strides },
            None => Self::View(view),
        }
    }

    #[inline(always)]
    fn get(&self, first: usize, second: usize, third: usize) -> T {
        match self {
            Self::Strided { values, strides } => {
                values[strided_offset(*strides, [first, second, third])]
            }
            Self::View(view) => (*view)[(first, second, third)],
        }
    }
}

enum IndexedView4<'a, T: Copy> {
    Strided {
        values: &'a [T],
        strides: [isize; 4],
    },
    View(ArrayView4<'a, T>),
}

impl<'a, T: Copy> IndexedView4<'a, T> {
    fn new(view: ArrayView4<'a, T>) -> Self {
        let strides = [
            view.strides()[0],
            view.strides()[1],
            view.strides()[2],
            view.strides()[3],
        ];
        match view.to_slice_memory_order() {
            Some(values) => Self::Strided { values, strides },
            None => Self::View(view),
        }
    }

    #[inline(always)]
    fn get(&self, first: usize, second: usize, third: usize, fourth: usize) -> T {
        match self {
            Self::Strided { values, strides } => {
                values[strided_offset(*strides, [first, second, third, fourth])]
            }
            Self::View(view) => (*view)[(first, second, third, fourth)],
        }
    }
}

enum IndexedView5<'a, T: Copy> {
    Strided {
        values: &'a [T],
        strides: [isize; 5],
    },
    View(ArrayView5<'a, T>),
}

impl<'a, T: Copy> IndexedView5<'a, T> {
    fn new(view: ArrayView5<'a, T>) -> Self {
        let strides = [
            view.strides()[0],
            view.strides()[1],
            view.strides()[2],
            view.strides()[3],
            view.strides()[4],
        ];
        match view.to_slice_memory_order() {
            Some(values) => Self::Strided { values, strides },
            None => Self::View(view),
        }
    }

    #[inline(always)]
    fn get(&self, first: usize, second: usize, third: usize, fourth: usize, fifth: usize) -> T {
        match self {
            Self::Strided { values, strides } => {
                values[strided_offset(*strides, [first, second, third, fourth, fifth])]
            }
            Self::View(view) => (*view)[(first, second, third, fourth, fifth)],
        }
    }
}

enum IndexedView6<'a, T: Copy> {
    Strided {
        values: &'a [T],
        strides: [isize; 6],
    },
    View(ArrayView6<'a, T>),
}

impl<'a, T: Copy> IndexedView6<'a, T> {
    fn new(view: ArrayView6<'a, T>) -> Self {
        let strides = [
            view.strides()[0],
            view.strides()[1],
            view.strides()[2],
            view.strides()[3],
            view.strides()[4],
            view.strides()[5],
        ];
        match view.to_slice_memory_order() {
            Some(values) => Self::Strided { values, strides },
            None => Self::View(view),
        }
    }

    #[inline(always)]
    fn get(
        &self,
        first: usize,
        second: usize,
        third: usize,
        fourth: usize,
        fifth: usize,
        sixth: usize,
    ) -> T {
        match self {
            Self::Strided { values, strides } => {
                values[strided_offset(*strides, [first, second, third, fourth, fifth, sixth])]
            }
            Self::View(view) => (*view)[(first, second, third, fourth, fifth, sixth)],
        }
    }
}

#[inline(always)]
fn strided_offset<const N: usize>(strides: [isize; N], indices: [usize; N]) -> usize {
    let mut offset = 0;
    let mut axis = 0;
    while axis < N {
        offset += strides[axis] * indices[axis] as isize;
        axis += 1;
    }
    debug_assert!(offset >= 0);
    offset as usize
}
