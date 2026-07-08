//! RHORRP-facing views derived from FEFF `phase.bin`.

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use refeff_core::{GenfmtSpinChannelCountInput, genfmt_spin_channel_count};

use crate::error::{IoError, Result};

use super::common::invalid_phase_bin;
use super::types::PhaseBinData;
use super::validate::validate_phase_bin;

/// RHORRP-ready values read from FEFF `phase.bin`.
///
/// FEFF `RHORRP/m_rhorrp.f90::init_rdxsph` reads `phase.bin` for the complex
/// contour mesh, chemical potential, and `ne1` split used by the final density
/// integration. The XSPH phase shifts are retained for diagnostics and FEFF
/// comparisons; the RHORRP `init_wavefunctions` path recomputes the `ph2`
/// table consumed by `rhoerrp`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpPhaseBinHandoff {
    /// Complex contour energies `em`, in Hartree.
    pub energies_hartree: Array1<Complex64>,
    /// FEFF chemical potential `xmu`, in Hartree.
    pub chemical_potential_hartree: f64,
    /// FEFF `ne1`: contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// XSPH phase shifts from `phase.bin` as `(energy, l, potential)`.
    pub xsph_phase_shifts: Array3<Complex64>,
}

impl RhorrpPhaseBinHandoff {
    /// Number of complex contour energies.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.energies_hartree.len()
    }

    /// Shared ordinary angular-momentum channel count in `xsph_phase_shifts`.
    #[must_use]
    pub fn angular_momentum_count(&self) -> usize {
        self.xsph_phase_shifts.dim().1
    }

    /// Number of FEFF potential blocks represented by the phase table.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.xsph_phase_shifts.dim().2
    }
}

impl PhaseBinData {
    /// Normalize this `phase.bin` payload for RHORRP density setup.
    pub fn to_rhorrp_handoff(&self, spin_selector: i32) -> Result<RhorrpPhaseBinHandoff> {
        rhorrp_phase_handoff_from_phase_bin(self, spin_selector)
    }
}

/// Build RHORRP density handoff values from parsed FEFF `phase.bin` data.
pub fn rhorrp_phase_handoff_from_phase_bin(
    phase: &PhaseBinData,
    spin_selector: i32,
) -> Result<RhorrpPhaseBinHandoff> {
    validate_rhorrp_phase_handoff(phase)?;
    let xsph_phase_shifts = rhorrp_phase_table_from_phase_bin(phase, spin_selector)?;

    Ok(RhorrpPhaseBinHandoff {
        energies_hartree: phase.energy_grid.clone(),
        chemical_potential_hartree: phase.scalars.fermi_level,
        real_axis_count: phase.main_energy_count,
        xsph_phase_shifts,
    })
}

/// Extract FEFF `rhoerrp` phase shifts as `(energy, l, potential)`.
///
/// `phase.bin` stores signed angular momentum slots `-lmax..lmax` per
/// potential, while the scalar RHORRP density path indexes ordinary
/// non-negative `l` channels. This helper copies the positive signed-`l`
/// branch into a shared angular axis and zero-fills channels beyond a
/// potential's own `lmax`.
///
/// The spin selector follows FEFF's ordinary scalar setup rule: `ispin == 1`
/// activates both available channels, and a two-spin table is averaged; every
/// other selector uses the first spin slot.
pub fn rhorrp_phase_table_from_phase_bin(
    phase: &PhaseBinData,
    spin_selector: i32,
) -> Result<Array3<Complex64>> {
    validate_phase_bin(phase)?;
    let angular_count = phase
        .potentials
        .iter()
        .map(|potential| potential.lmax)
        .max()
        .and_then(|lmax| lmax.checked_add(1))
        .ok_or_else(|| invalid_phase_bin("lmax", "phase.bin contains no potentials"))?;
    let spin_channel_count = genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
        spin_selector,
        available_spin_channels: phase.spin_count,
    })
    .map_err(rhorrp_phase_error)?;

    let mut output = Array3::zeros((phase.energy_count, angular_count, phase.potential_count()));
    for (potential_index, potential) in phase.potentials.iter().enumerate() {
        for energy in 0..phase.energy_count {
            for angular in 0..=potential.lmax {
                let signed_l_slot = potential
                    .lmax
                    .checked_add(angular)
                    .ok_or_else(|| invalid_phase_bin("lmax", "signed l slot overflowed"))?;
                output[(energy, angular, potential_index)] = selected_phase_shift(
                    phase.spin_count,
                    spin_channel_count,
                    potential.phase_shifts[(energy, signed_l_slot, 0)],
                    potential.phase_shifts[(energy, signed_l_slot, phase.spin_count - 1)],
                );
            }
        }
    }

    Ok(output)
}

fn selected_phase_shift(
    available_spin_channels: usize,
    active_spin_channels: usize,
    first_spin: Complex64,
    last_spin: Complex64,
) -> Complex64 {
    if available_spin_channels == 1 || active_spin_channels == 1 {
        first_spin
    } else {
        (first_spin + last_spin) * 0.5
    }
}

fn rhorrp_phase_error(source: refeff_core::GenfmtError) -> IoError {
    invalid_phase_bin("rhorrp_phase", source.to_string())
}

fn validate_rhorrp_phase_handoff(phase: &PhaseBinData) -> Result<()> {
    validate_phase_bin(phase)?;
    if phase.main_energy_count < 2 || phase.main_energy_count > phase.energy_count {
        return Err(invalid_phase_bin(
            "ne1",
            format!(
                "RHORRP real-axis count {} is outside 2..={}",
                phase.main_energy_count, phase.energy_count
            ),
        ));
    }
    let used_energy_count = phase
        .main_energy_count
        .checked_add(phase.auxiliary_energy_count)
        .ok_or_else(|| invalid_phase_bin("ne3", "main plus auxiliary energy count overflowed"))?;
    if used_energy_count > phase.energy_count {
        return Err(invalid_phase_bin(
            "ne3",
            format!(
                "main plus auxiliary energy counts {} exceed total energy count {}",
                used_energy_count, phase.energy_count
            ),
        ));
    }
    Ok(())
}
