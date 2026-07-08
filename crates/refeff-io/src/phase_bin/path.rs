//! PATH-facing views derived from FEFF `phase.bin`.
//!
//! `PATH/prcrit.f90` reads `phase.bin`, selects the first spin channel, copies
//! the negative signed-`l` phase-shift branch into ordinary non-negative
//! angular channels, and then builds the single-precision pathfinder criteria
//! tables. This module keeps the file decoding in `refeff-io` and delegates the
//! numerical table construction to `refeff-core`.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use refeff_core::{PathPhaseCriteriaInput, PathPhaseCriteriaTables, path_phase_criteria_tables};

use crate::error::{IoError, Result};

use super::common::invalid_phase_bin;
use super::types::PhaseBinData;
use super::validate::validate_phase_bin;

/// PATH-ready values read from FEFF `phase.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinPathHandoff {
    /// FEFF `potlbl(0:npot)` labels.
    pub potential_labels: Vec<String>,
    /// FEFF `em(1:ne)` complex energy grid.
    pub energies: Array1<Complex64>,
    /// FEFF `eref(1:ne,1)` first-spin reference energies.
    pub reference_energies: Array1<Complex64>,
    /// FEFF `ph(1:ne,1:lmaxp1,0:npot)` copied from `ph4(ie,-l,1,iph)`.
    pub phase_shifts: Array3<Complex64>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: Array2<usize>,
    /// FEFF `neout`, normally `ne1`.
    pub output_energy_count: usize,
    /// Zero-based Rust equivalent of FEFF `ik0`.
    pub zero_wave_energy_index: usize,
}

impl PhaseBinPathHandoff {
    /// Number of complex contour energies.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.energies.len()
    }

    /// Number of FEFF potential blocks represented by the phase table.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potential_labels.len()
    }

    /// Build PATH criteria tables from this handoff.
    pub fn criteria_tables(&self) -> Result<PathPhaseCriteriaTables> {
        path_phase_criteria_tables(PathPhaseCriteriaInput {
            energies: self
                .energies
                .as_slice()
                .ok_or_else(|| invalid_phase_bin("em", "energy grid must be contiguous"))?,
            reference_energies: self.reference_energies.as_slice().ok_or_else(|| {
                invalid_phase_bin("eref", "reference energy grid must be contiguous")
            })?,
            phase_shifts: self.phase_shifts.view(),
            angular_limits: self.angular_limits.view(),
            output_energy_count: self.output_energy_count,
            zero_wave_energy_index: self.zero_wave_energy_index,
        })
        .map_err(path_phase_error)
    }
}

impl PhaseBinData {
    /// Normalize this `phase.bin` payload for PATH criteria setup.
    pub fn to_path_handoff(&self) -> Result<PhaseBinPathHandoff> {
        phase_bin_path_handoff_from_phase_bin(self)
    }
}

/// Build PATH phase handoff values from parsed FEFF `phase.bin` data.
pub fn phase_bin_path_handoff_from_phase_bin(phase: &PhaseBinData) -> Result<PhaseBinPathHandoff> {
    validate_path_phase_handoff(phase)?;
    let angular_count = phase
        .potentials
        .iter()
        .map(|potential| potential.lmax)
        .max()
        .and_then(|lmax| lmax.checked_add(1))
        .ok_or_else(|| invalid_phase_bin("lmax", "phase.bin contains no potentials"))?;
    let potential_count = phase.potential_count();
    let mut phase_shifts = Array3::zeros((phase.energy_count, angular_count, potential_count));
    let mut angular_limits = Array2::zeros((phase.energy_count, potential_count));

    for (potential_index, potential) in phase.potentials.iter().enumerate() {
        for energy in 0..phase.energy_count {
            angular_limits[(energy, potential_index)] = potential.lmax;
            for angular in 0..=potential.lmax {
                let signed_l_slot = potential.lmax.checked_sub(angular).ok_or_else(|| {
                    invalid_phase_bin("lmax", "negative signed-l slot underflowed")
                })?;
                phase_shifts[(energy, angular, potential_index)] =
                    potential.phase_shifts[(energy, signed_l_slot, 0)];
            }
        }
    }

    Ok(PhaseBinPathHandoff {
        potential_labels: phase
            .potentials
            .iter()
            .map(|potential| potential.label.clone())
            .collect(),
        energies: phase.energy_grid.clone(),
        reference_energies: phase.reference_energy.column(0).to_owned(),
        phase_shifts,
        angular_limits,
        output_energy_count: phase.main_energy_count,
        zero_wave_energy_index: phase_zero_wave_index(phase)?,
    })
}

/// Build FEFF PATH criteria tables directly from parsed `phase.bin` data.
pub fn path_phase_criteria_tables_from_phase_bin(
    phase: &PhaseBinData,
) -> Result<PathPhaseCriteriaTables> {
    phase_bin_path_handoff_from_phase_bin(phase)?.criteria_tables()
}

fn validate_path_phase_handoff(phase: &PhaseBinData) -> Result<()> {
    validate_phase_bin(phase)?;
    if phase.main_energy_count == 0 || phase.main_energy_count > phase.energy_count {
        return Err(invalid_phase_bin(
            "ne1",
            format!(
                "PATH output energy count {} is outside 1..={}",
                phase.main_energy_count, phase.energy_count
            ),
        ));
    }
    let zero_wave_index = phase_zero_wave_index(phase)?;
    if zero_wave_index >= phase.main_energy_count {
        return Err(invalid_phase_bin(
            "ik0",
            "PATH zero-wave index must fall within the main energy grid",
        ));
    }
    Ok(())
}

fn phase_zero_wave_index(phase: &PhaseBinData) -> Result<usize> {
    if phase.fermi_index <= 0 {
        return Err(invalid_phase_bin(
            "ik0",
            "PATH zero-wave index must be one-based and positive",
        ));
    }
    let index = usize::try_from(phase.fermi_index - 1)
        .map_err(|_| invalid_phase_bin("ik0", "failed to convert zero-wave index"))?;
    if index >= phase.energy_count {
        return Err(invalid_phase_bin(
            "ik0",
            "PATH zero-wave index must fall within the energy grid",
        ));
    }
    Ok(index)
}

fn path_phase_error(source: refeff_core::PathError) -> IoError {
    invalid_phase_bin("path_phase", source.to_string())
}
