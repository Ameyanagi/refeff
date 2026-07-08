//! RIXS-facing views derived from FEFF `phase.bin`.
//!
//! `RIXS/rdxsphrxs.f90` reads the XSPH `phase.bin` handoff, keeps the first
//! eight transition-moment channels from the first `q` block, and recomputes
//! per-energy active angular limits from nonzero positive-`l` phase shifts.

use ndarray::{Array1, Array2, Array3, Axis};
use num_complex::Complex64;
use refeff_core::{
    RixsTransitionMatrixInput, RixsTransitionMatrixSetup, RixsTransitionPhaseShiftInput,
    rixs_transition_matrix_setup, rixs_transition_phase_shifts,
};

use crate::error::Result;
use crate::global_input::GlobalInput;

use super::common::invalid_phase_bin;
use super::types::{
    PHASE_BIN_DEFAULT_TRANSITION_COUNT, PhaseBinData, PhaseBinPotential, PhaseBinScalars,
};
use super::validate::validate_phase_bin;

const RIXS_PHASE_MIN: f64 = 1.0e-7;

/// RIXS-ready values read from FEFF `phase.bin`.
///
/// FEFF `RIXS/rdxsphrxs.f90` is a specialized reader rather than a general
/// `phase.bin` codec. It preserves the full phase-shift blocks, exposes the
/// legacy eight `rkk` transition channels used by `RIXS/rixs.f90`, and derives
/// `lmax(ie, iph)` by scanning positive signed-`l` phase shifts from high to
/// low until either the first or last spin channel has `abs(sin(ph)) > 1e-7`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinRixsHandoff {
    /// Spin channel count, `nsp`.
    pub spin_count: usize,
    /// Complex energy count, `ne`.
    pub energy_count: usize,
    /// Main horizontal-axis energy count, `ne1`.
    pub main_energy_count: usize,
    /// Auxiliary horizontal-axis energy count, `ne3`.
    pub auxiliary_energy_count: usize,
    /// Core-hole index, `ihole`.
    pub ihole: i32,
    /// Fermi-level grid index, `ik0`.
    pub fermi_index: i32,
    /// FEFF scalar `dum(3)` block.
    pub scalars: PhaseBinScalars,
    /// Complex contour energies `em`.
    pub energy_grid: Array1<Complex64>,
    /// Reference/self-energy mesh `eref(ie, isp)`.
    pub reference_energy: Array2<Complex64>,
    /// Per-potential phase-shift blocks as read from `phase.bin`.
    pub potentials: Vec<PhaseBinPotential>,
    /// Legacy RIXS transition moments `rkk(ie, kdif, isp)` for `kdif=1..8`.
    pub transition_moments: Array3<Complex64>,
    /// FEFF `lmax(ie, iph)` derived from nonzero positive-`l` phase shifts.
    pub angular_limits: Array2<usize>,
    /// FEFF `lmaxp1`, the largest derived angular limit plus one.
    pub max_angular_limit_plus_one: usize,
}

impl PhaseBinRixsHandoff {
    /// Number of FEFF potential blocks represented by the phase table.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potentials.len()
    }
}

/// Build the FEFF RIXS transition matrix setup from `global.inp` and
/// RIXS-normalized `phase.bin`.
///
/// This ports the source-backed `setkap`/`bcoef` setup boundary before the
/// RIXS scattering loops. The returned setup carries FEFF `lnd/knd` transition
/// labels and the diagonal `bmat` entries used by the later amplitude
/// convolution.
pub fn phase_bin_rixs_transition_setup_from_handoffs(
    global: &GlobalInput,
    phase: &PhaseBinRixsHandoff,
) -> Result<RixsTransitionMatrixSetup> {
    rixs_transition_matrix_setup(RixsTransitionMatrixInput {
        lmax: phase.max_angular_limit_plus_one.saturating_sub(1),
        hole: phase.ihole,
        polarization: global.control.ipol,
        polarization_tensor: rixs_polarization_tensor(global),
        multipole: global.control.le2,
        trace_orbital: false,
        spin: global.control.ispin,
        spin_channel_count: phase.spin_count,
        spin_vector_angle: global.control.angks,
    })
    .map_err(|source| invalid_phase_bin("rixs_transition_setup", source.to_string()))
}

/// Select the absorber phase shifts consumed by FEFF RIXS transition kernels.
///
/// FEFF indexes the absorber branch as `ph(iE, -lnd(ind), 1, 0)`: the first
/// spin channel and first potential block, with signed angular columns selected
/// from the transition setup's `lnd` labels.
pub fn phase_bin_rixs_transition_phase_shifts_from_handoff(
    phase: &PhaseBinRixsHandoff,
    setup: &RixsTransitionMatrixSetup,
) -> Result<Array2<Complex64>> {
    let potential = phase
        .potentials
        .first()
        .ok_or_else(|| invalid_phase_bin("potentials", "phase.bin contains no RIXS potentials"))?;
    let signed_l_min = isize::try_from(potential.lmax)
        .map_err(|_| invalid_phase_bin("lmax", "phase lmax exceeds isize"))?
        .checked_neg()
        .ok_or_else(|| invalid_phase_bin("lmax", "phase lmax exceeds signed range"))?;
    let phase_shifts = potential.phase_shifts.index_axis(Axis(2), 0);

    rixs_transition_phase_shifts(RixsTransitionPhaseShiftInput {
        phase_shifts,
        signed_l_min,
        transition_angular_momenta: &setup.transition_angular_momenta,
    })
    .map_err(|source| invalid_phase_bin("rixs_transition_phase_shifts", source.to_string()))
}

impl PhaseBinData {
    /// Normalize this `phase.bin` payload for RIXS setup.
    pub fn to_rixs_handoff(&self) -> Result<PhaseBinRixsHandoff> {
        phase_bin_rixs_handoff_from_phase_bin(self)
    }
}

/// Build RIXS phase-shift handoff values from parsed FEFF `phase.bin` data.
pub fn phase_bin_rixs_handoff_from_phase_bin(phase: &PhaseBinData) -> Result<PhaseBinRixsHandoff> {
    validate_rixs_phase_handoff(phase)?;
    let angular_limits = rixs_angular_limits_from_phase_bin(phase)?;
    let max_angular_limit_plus_one = match angular_limits.iter().copied().max() {
        Some(limit) => limit
            .checked_add(1)
            .ok_or_else(|| invalid_phase_bin("lmaxp1", "angular limit overflowed"))?,
        None => 0,
    };
    let transition_moments = rixs_transition_moments_from_phase_bin(phase)?;

    Ok(PhaseBinRixsHandoff {
        spin_count: phase.spin_count,
        energy_count: phase.energy_count,
        main_energy_count: phase.main_energy_count,
        auxiliary_energy_count: phase.auxiliary_energy_count,
        ihole: phase.ihole,
        fermi_index: phase.fermi_index,
        scalars: phase.scalars,
        energy_grid: phase.energy_grid.clone(),
        reference_energy: phase.reference_energy.clone(),
        potentials: phase.potentials.clone(),
        transition_moments,
        angular_limits,
        max_angular_limit_plus_one,
    })
}

/// Recompute FEFF `RIXS/rdxsphrxs.f90` `lmax(ie, iph)` from phase shifts.
pub fn rixs_angular_limits_from_phase_bin(phase: &PhaseBinData) -> Result<Array2<usize>> {
    validate_phase_bin(phase)?;
    let mut angular_limits = Array2::zeros((phase.energy_count, phase.potential_count()));
    for (potential_index, potential) in phase.potentials.iter().enumerate() {
        for energy in 0..phase.energy_count {
            angular_limits[(energy, potential_index)] =
                rixs_active_angular_limit(phase, potential, energy)?;
        }
    }
    Ok(angular_limits)
}

/// Extract the legacy eight-channel `rkk(ie, kdif, isp)` table used by RIXS.
pub fn rixs_transition_moments_from_phase_bin(phase: &PhaseBinData) -> Result<Array3<Complex64>> {
    validate_rixs_phase_handoff(phase)?;
    Ok(Array3::from_shape_fn(
        (
            phase.energy_count,
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            phase.spin_count,
        ),
        |(energy, transition, spin)| phase.transition_moments[(energy, 0, transition, spin)],
    ))
}

fn rixs_active_angular_limit(
    phase: &PhaseBinData,
    potential: &PhaseBinPotential,
    energy: usize,
) -> Result<usize> {
    for angular in (0..=potential.lmax).rev() {
        let signed_l_slot = potential
            .lmax
            .checked_add(angular)
            .ok_or_else(|| invalid_phase_bin("lmax", "positive signed-l slot overflowed"))?;
        let first_spin = potential.phase_shifts[(energy, signed_l_slot, 0)];
        let last_spin = potential.phase_shifts[(energy, signed_l_slot, phase.spin_count - 1)];
        if first_spin.sin().norm() > RIXS_PHASE_MIN || last_spin.sin().norm() > RIXS_PHASE_MIN {
            return Ok(angular);
        }
    }
    Ok(0)
}

fn validate_rixs_phase_handoff(phase: &PhaseBinData) -> Result<()> {
    validate_phase_bin(phase)?;
    if phase.q_count == 0 {
        return Err(invalid_phase_bin(
            "nq",
            "RIXS phase handoff requires at least one q block",
        ));
    }
    if phase.transition_count < PHASE_BIN_DEFAULT_TRANSITION_COUNT {
        return Err(invalid_phase_bin(
            "rkk",
            format!(
                "RIXS phase handoff requires at least {} transition channels, got {}",
                PHASE_BIN_DEFAULT_TRANSITION_COUNT, phase.transition_count
            ),
        ));
    }
    Ok(())
}

fn rixs_polarization_tensor(global: &GlobalInput) -> [[Complex64; 3]; 3] {
    let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    for (row_index, row) in global.polarization_tensor.iter().enumerate() {
        tensor[row_index] = [
            Complex64::new(row[0], row[1]),
            Complex64::new(row[2], row[3]),
            Complex64::new(row[4], row[5]),
        ];
    }
    tensor
}
