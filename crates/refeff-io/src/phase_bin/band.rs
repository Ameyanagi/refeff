//! BAND-facing views derived from FEFF `phase.bin`.
//!
//! BAND consumes the same signed-`l` phase-shift table shape used by the
//! `bandtot.f90` interpolation loop. Parsed `phase.bin` stores each potential
//! with its own `lmax`; this adapter normalizes those blocks onto one shared
//! signed-`l` axis for `refeff-core` BAND setup helpers.

use ndarray::{Array1, Array2, Array4, Axis, Slice};
use num_complex::Complex64;
use refeff_core::{
    BandEnergySearchMesh, BandEnergySearchMeshInput, BandPhaseSearchInterpolation,
    BandPhaseSearchInterpolationInput, Real, band_energy_search_mesh,
    band_phase_search_interpolation,
};

use crate::BandInput;
use crate::error::{IoError, Result};

use super::common::{checked_l_count, invalid_phase_bin};
use super::types::PhaseBinData;
use super::validate::validate_phase_bin;

/// `phase.bin` data normalized for BAND search-mesh interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinBandData {
    /// Offset used to map signed angular momentum onto `spin_phase_shifts`.
    pub signed_angular_offset: usize,
    /// FEFF `em(1:ne)` complex energy grid.
    pub energies_hartree: Array1<Complex64>,
    /// FEFF `eref(1:ne,1:nsp)` reference energies.
    pub reference_energies_hartree: Array2<Complex64>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nsp,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin,
    /// potential)`. Slots outside a potential's own `lmax` are zero-filled.
    pub spin_phase_shifts: Array4<Complex64>,
    /// Inclusive active `lmax` for each potential.
    pub potential_lmax: Vec<usize>,
    /// FEFF `potlbl(0:npot)` labels.
    pub potential_labels: Vec<String>,
}

/// BAND search-mesh and phase/reference setup derived from FEFF handoffs.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinBandSearchSetup {
    /// Normalized `phase.bin` data used to build the interpolated tables.
    pub phase_handoff: PhaseBinBandData,
    /// FEFF `bandtot.f90` clipped search mesh.
    pub energy_mesh: BandEnergySearchMesh,
    /// Interpolated reference energies, momenta, and phase shifts.
    pub phase_interpolation: BandPhaseSearchInterpolation,
}

impl PhaseBinData {
    /// Normalize this `phase.bin` payload for BAND search setup.
    pub fn to_band_data(&self) -> Result<PhaseBinBandData> {
        phase_bin_band_data(self)
    }

    /// Build BAND search-mesh phase/reference setup from this phase handoff.
    pub fn to_band_search_setup(&self, band: &BandInput) -> Result<PhaseBinBandSearchSetup> {
        band_search_setup_from_handoffs(band, self)
    }
}

/// Build BAND phase/reference handoff values from parsed FEFF `phase.bin` data.
pub fn phase_bin_band_data(data: &PhaseBinData) -> Result<PhaseBinBandData> {
    validate_phase_bin(data)?;

    let signed_angular_offset = data
        .potentials
        .iter()
        .map(|potential| potential.lmax)
        .max()
        .ok_or_else(|| invalid_phase_bin("lmax", "phase.bin contains no potentials"))?;
    let signed_l_count = checked_l_count(signed_angular_offset)?;
    let mut spin_phase_shifts = Array4::zeros((
        data.energy_count,
        signed_l_count,
        data.spin_count,
        data.potential_count(),
    ));

    for (potential_index, potential) in data.potentials.iter().enumerate() {
        let potential_l_count = checked_l_count(potential.lmax)?;
        let destination_start = signed_angular_offset - potential.lmax;
        for energy in 0..data.energy_count {
            for l_slot in 0..potential_l_count {
                let destination_l_slot = destination_start + l_slot;
                for spin in 0..data.spin_count {
                    spin_phase_shifts[(energy, destination_l_slot, spin, potential_index)] =
                        potential.phase_shifts[(energy, l_slot, spin)];
                }
            }
        }
    }

    Ok(PhaseBinBandData {
        signed_angular_offset,
        energies_hartree: data.energy_grid.clone(),
        reference_energies_hartree: data.reference_energy.clone(),
        spin_phase_shifts,
        potential_lmax: data
            .potentials
            .iter()
            .map(|potential| potential.lmax)
            .collect(),
        potential_labels: data
            .potentials
            .iter()
            .map(|potential| potential.label.clone())
            .collect(),
    })
}

/// Build FEFF BAND pre-solver phase/reference setup from parsed handoff files.
///
/// This covers the `bandtot.f90` work before the KKR solve: clipping the
/// requested `band.inp` energy range to active XSPH phase coverage, preserving
/// FEFF's cubic interpolation order, and producing BAND-ready `xphase`,
/// `refene`, and `ck` tables on the search mesh.
pub fn band_search_setup_from_handoffs(
    band: &BandInput,
    phase: &PhaseBinData,
) -> Result<PhaseBinBandSearchSetup> {
    let phase_handoff = phase_bin_band_data(phase)?;
    let potential_lmax = phase_handoff.potential_lmax.clone();
    band_search_setup_from_phase_handoff(band, phase, phase_handoff, &potential_lmax)
}

/// Build BAND phase/reference setup using FEFF `fms.inp` `lmaxph(0:nph)`.
///
/// FEFF `bandtot.f90` reads raw phase shifts from `phase.bin`, but the
/// interpolation loop only fills `xphase` over the active `lmaxph` range passed
/// through `reafms`/`kprep`. Callers that have parsed `fms.inp` should use this
/// handoff so the BAND KSPACE and T-matrix dimensions follow FEFF's active
/// FMS angular cutoff instead of the larger raw phase-write range.
pub fn band_search_setup_from_handoffs_with_lmaxph(
    band: &BandInput,
    phase: &PhaseBinData,
    lmaxph: &[usize],
) -> Result<PhaseBinBandSearchSetup> {
    let phase_handoff = phase_bin_band_data(phase)?;
    band_search_setup_from_phase_handoff(band, phase, phase_handoff, lmaxph)
}

fn band_search_setup_from_phase_handoff(
    band: &BandInput,
    phase: &PhaseBinData,
    phase_handoff: PhaseBinBandData,
    potential_lmax: &[usize],
) -> Result<PhaseBinBandSearchSetup> {
    validate_active_phase_prefix(phase)?;

    let active_len = phase.main_energy_count;
    let source_energies_hartree: Array1<Real> = Array1::from_iter(
        phase_handoff
            .energies_hartree
            .iter()
            .take(active_len)
            .map(|energy| energy.re),
    );
    let active_references = phase_handoff
        .reference_energies_hartree
        .slice_axis(Axis(0), Slice::from(..active_len));
    let active_phase_shifts = phase_handoff
        .spin_phase_shifts
        .slice_axis(Axis(0), Slice::from(..active_len));

    let energy_mesh = band_energy_search_mesh(BandEnergySearchMeshInput {
        requested_min_ev: band.energy_mesh.emin,
        requested_max_ev: band.energy_mesh.emax,
        requested_step_ev: band.energy_mesh.estep,
        phase_energies_hartree: phase_handoff.energies_hartree.view(),
        phase_active_len: active_len,
        fermi_level_hartree: phase.scalars.fermi_level,
    })
    .map_err(band_setup_error)?;

    let phase_interpolation = band_phase_search_interpolation(BandPhaseSearchInterpolationInput {
        search_energies_hartree: energy_mesh.energies_hartree.view(),
        source_energies_hartree: source_energies_hartree.view(),
        source_reference_energies_hartree: active_references,
        source_phase_shifts: active_phase_shifts,
        potential_lmax,
        interpolation_order: 3,
    })
    .map_err(band_setup_error)?;

    Ok(PhaseBinBandSearchSetup {
        phase_handoff,
        energy_mesh,
        phase_interpolation,
    })
}

fn validate_active_phase_prefix(phase: &PhaseBinData) -> Result<()> {
    if phase.main_energy_count == 0 || phase.main_energy_count > phase.energy_count {
        return Err(invalid_phase_bin(
            "ne1",
            format!(
                "active BAND phase prefix must be in 1..={}, got {}",
                phase.energy_count, phase.main_energy_count
            ),
        ));
    }
    Ok(())
}

fn band_setup_error(source: refeff_core::BandError) -> IoError {
    IoError::BandSetup { source }
}
