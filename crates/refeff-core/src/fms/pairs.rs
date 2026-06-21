use super::*;
use ndarray::{ArrayView2, ArrayView3, ArrayView5, ArrayView6};
use rayon::prelude::*;

const FMS_PARALLEL_STATE_THRESHOLD: usize = 128;

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

    let state_count = input.states.len();
    if state_count == 0 {
        return Ok(Array2::zeros((0, 0).f()));
    }
    let limits = propagator_state_limits(
        input.states,
        input.atoms.len(),
        Some(input.wave_numbers.len()),
    )?;
    ensure_axis_len("xclm", "m", input.xclm.shape()[0], limits.max_l)?;
    ensure_axis_len("xclm", "l", input.xclm.shape()[1], limits.max_l)?;
    ensure_axis_len("rotations", "l", input.rotations.shape()[2], limits.max_l)?;
    let states = cache_propagator_states(
        input.states,
        input.atoms.len(),
        Some(input.wave_numbers.len()),
        rotation_lmax_for(input.rotations)?,
    )?;
    let xnlm = validated_normalization_table(input.xnlm, limits.max_l)?;
    let tables = SpinPropagatorTables::new(&input)?;
    let pair_cache = spin_pair_prefactor_cache(&input, &states, &tables)?;
    let mut matrix = vec![Complex32::new(0.0, 0.0); state_count * state_count];
    if state_count >= FMS_PARALLEL_STATE_THRESHOLD {
        matrix
            .par_chunks_mut(state_count)
            .enumerate()
            .try_for_each(|(column, values)| {
                fill_spin_free_propagator_column(
                    values,
                    column,
                    &states,
                    &tables,
                    &xnlm,
                    &pair_cache,
                )
            })?;
    } else {
        for (column, values) in matrix.chunks_mut(state_count).enumerate() {
            fill_spin_free_propagator_column(values, column, &states, &tables, &xnlm, &pair_cache)?;
        }
    }

    Ok(Array2::from_shape_vec((state_count, state_count).f(), matrix).expect("g0 shape"))
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
}

#[derive(Clone, Copy)]
struct SpinPairPrefactor {
    value: Complex32,
}

struct SpinPairPrefactorCache {
    values: Vec<Option<SpinPairPrefactor>>,
    atom_count: usize,
    spin_count: usize,
}

impl SpinPairPrefactorCache {
    fn new(atom_count: usize, spin_count: usize) -> Self {
        Self {
            values: vec![None; atom_count * atom_count * spin_count],
            atom_count,
            spin_count,
        }
    }

    #[inline(always)]
    fn key(&self, atom2: usize, atom1: usize, spin: usize) -> usize {
        debug_assert!(atom2 < self.atom_count);
        debug_assert!(atom1 < self.atom_count);
        debug_assert!(spin < self.spin_count);
        (atom2 * self.atom_count + atom1) * self.spin_count + spin
    }

    fn insert(&mut self, atom2: usize, atom1: usize, spin: usize, value: SpinPairPrefactor) {
        let key = self.key(atom2, atom1, spin);
        self.values[key] = Some(value);
    }

    #[inline(always)]
    fn get(&self, atom2: usize, atom1: usize, spin: usize) -> Option<SpinPairPrefactor> {
        self.values[self.key(atom2, atom1, spin)]
    }
}

struct SpinPropagatorTables<'a> {
    rho: IndexedView3<'a, Complex32>,
    sigsqr: IndexedView2<'a, f32>,
    xclm: IndexedView5<'a, Complex32>,
    rotations: IndexedView6<'a, Complex32>,
    rotation_lmax: usize,
}

impl<'a> SpinPropagatorTables<'a> {
    fn new(input: &FmsSpinFreePropagatorMatrixInput<'a>) -> Result<Self, FmsError> {
        let rotation_lmax = rotation_lmax_for(input.rotations)?;
        Ok(Self {
            rho: IndexedView3::new(input.rho),
            sigsqr: IndexedView2::new(input.mean_square_displacements),
            xclm: IndexedView5::new(input.xclm),
            rotations: IndexedView6::new(input.rotations),
            rotation_lmax,
        })
    }
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

fn fill_spin_free_propagator_column(
    column_values: &mut [Complex32],
    column: usize,
    states: &[CachedPropagatorState],
    tables: &SpinPropagatorTables<'_>,
    xnlm: &NormalizationTable,
    pair_cache: &SpinPairPrefactorCache,
) -> Result<(), FmsError> {
    let second = states[column];
    for (row, first) in states.iter().copied().enumerate() {
        if first.atom_index == second.atom_index || first.state.spin != second.state.spin {
            continue;
        }

        let spin = first.state.spin - 1;
        let Some(pair) = pair_cache.get(second.atom_index, first.atom_index, spin) else {
            continue;
        };
        column_values[row] =
            fms_free_propagator_element_fast5(first, second, spin, pair, tables, xnlm)?;
    }
    Ok(())
}

fn spin_pair_prefactor_cache(
    input: &FmsSpinFreePropagatorMatrixInput<'_>,
    states: &[CachedPropagatorState],
    tables: &SpinPropagatorTables<'_>,
) -> Result<SpinPairPrefactorCache, FmsError> {
    let mut reachable = SpinPairPrefactorCache::new(input.atoms.len(), input.wave_numbers.len());
    let cutoff_squared = input.direct_cutoff * input.direct_cutoff;
    for first in states {
        for second in states {
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
            let key = reachable.key(second.atom_index, first.atom_index, spin);
            if reachable.values[key].is_some() {
                continue;
            }
            ensure_axis_len("xrho", "atom2", input.rho.shape()[0], second.atom_index)?;
            ensure_axis_len("xrho", "atom1", input.rho.shape()[1], first.atom_index)?;
            ensure_axis_len(
                "sigsqr",
                "atom2",
                input.mean_square_displacements.shape()[0],
                second.atom_index,
            )?;
            ensure_axis_len(
                "sigsqr",
                "atom1",
                input.mean_square_displacements.shape()[1],
                first.atom_index,
            )?;
            ensure_axis_len("xclm", "atom2", input.xclm.shape()[2], second.atom_index)?;
            ensure_axis_len("xclm", "atom1", input.xclm.shape()[3], first.atom_index)?;
            ensure_axis_len(
                "rotations",
                "atom2",
                input.rotations.shape()[4],
                second.atom_index,
            )?;
            ensure_axis_len(
                "rotations",
                "atom1",
                input.rotations.shape()[5],
                first.atom_index,
            )?;

            let rho = tables.rho.get(second.atom_index, first.atom_index, spin);
            let mean_square_displacement = tables.sigsqr.get(second.atom_index, first.atom_index);
            validate_fast_propagator_scalars(
                rho,
                input.wave_numbers[spin],
                mean_square_displacement,
            )?;
            reachable.insert(
                second.atom_index,
                first.atom_index,
                spin,
                SpinPairPrefactor {
                    value: fms_free_propagator_prefactor(
                        rho,
                        input.wave_numbers[spin],
                        mean_square_displacement,
                    ),
                },
            );
        }
    }
    Ok(reachable)
}

fn fms_free_propagator_element_fast5(
    first: CachedPropagatorState,
    second: CachedPropagatorState,
    spin: usize,
    pair: SpinPairPrefactor,
    tables: &SpinPropagatorTables<'_>,
    xnlm: &NormalizationTable,
) -> Result<Complex32, FmsError> {
    let l1 = first.state.angular_momentum;
    let l2 = second.state.angular_momentum;
    let l1_signed = isize::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
        name: "l1",
        value: l1,
        lx: l1,
    })?;

    let mut sum = Complex32::new(0.0, 0.0);
    for mu in -l1_signed..=l1_signed {
        let gllmz = rehr_albers_z_axis_propagator_fast5(
            mu.unsigned_abs(),
            first,
            second,
            spin,
            &tables.xclm,
            xnlm,
        )?;
        let mu_index = signed_mu_index(mu, tables.rotation_lmax);
        let backward = tables.rotations.get(
            mu_index,
            first.magnetic_index,
            l1,
            1,
            second.atom_index,
            first.atom_index,
        );
        let forward = tables.rotations.get(
            second.magnetic_index,
            mu_index,
            l2,
            0,
            second.atom_index,
            first.atom_index,
        );
        sum += backward * gllmz * forward;
    }

    Ok(pair.value * sum)
}

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

fn validated_normalization_table(
    xnlm: ArrayView2<'_, Real>,
    max_l: usize,
) -> Result<NormalizationTable, FmsError> {
    let stride = max_l.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lmax",
        value: max_l,
        lx: max_l,
    })?;
    ensure_axis_len("xnlm", "mu", xnlm.shape()[0], max_l)?;
    ensure_axis_len("xnlm", "l", xnlm.shape()[1], max_l)?;
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
            let value = xnlm[(mu, angular_momentum)] as f32;
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
    for &state in states {
        if let Some(spin_channels) = spin_channels {
            ensure_state_spin(state.spin, spin_channels)?;
        }
        let atom_index = checked_atom_index(state.atom)?;
        ensure_atom_table_index(atom_index, atom_count)?;
        max_l = max_l.max(state.angular_momentum);
    }
    Ok(PropagatorStateLimits { max_l })
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

fn rotation_lmax_for(rotations: ArrayView6<'_, Complex32>) -> Result<usize, FmsError> {
    let rows = rotations.shape()[0];
    if rows == 0 || rows.is_multiple_of(2) {
        return Err(FmsError::TableIndexOutOfRange {
            table: "rotations",
            axis: "m",
            index: rows,
        });
    }
    let rotation_lmax = (rows - 1) / 2;
    ensure_axis_len("rotations", "m", rotations.shape()[1], 2 * rotation_lmax)?;
    ensure_axis_len("rotations", "k", rotations.shape()[3], 1)?;
    Ok(rotation_lmax)
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
    fn get(&self, row: usize, col: usize) -> T {
        match self {
            Self::Strided { values, strides } => {
                let index = row as isize * strides[0] + col as isize * strides[1];
                values[index as usize]
            }
            Self::View(view) => view[(row, col)],
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
                let index = first as isize * strides[0]
                    + second as isize * strides[1]
                    + third as isize * strides[2];
                values[index as usize]
            }
            Self::View(view) => view[(first, second, third)],
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
                let index = first as isize * strides[0]
                    + second as isize * strides[1]
                    + third as isize * strides[2]
                    + fourth as isize * strides[3]
                    + fifth as isize * strides[4];
                values[index as usize]
            }
            Self::View(view) => view[(first, second, third, fourth, fifth)],
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
                let index = first as isize * strides[0]
                    + second as isize * strides[1]
                    + third as isize * strides[2]
                    + fourth as isize * strides[3]
                    + fifth as isize * strides[4]
                    + sixth as isize * strides[5];
                values[index as usize]
            }
            Self::View(view) => view[(first, second, third, fourth, fifth, sixth)],
        }
    }
}
