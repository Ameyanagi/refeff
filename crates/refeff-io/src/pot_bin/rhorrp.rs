//! RHORRP-facing views derived from FEFF `pot.bin`.

use ndarray::ArrayView1;
use refeff_core::{
    Complex, Real, RhorrpPreparedWavefunctionTablesInput, RhorrpWavefunctionGridPreparation,
    RhorrpWavefunctionGridPreparationInput, RhorrpWavefunctionTables,
    rhorrp_density_reference_energy_hartree, rhorrp_prepare_wavefunction_grids,
    rhorrp_prepared_wavefunction_tables,
};

use crate::config_dat::{
    ConfigDatData, RhorrpConfigOrbitalTables, rhorrp_orbital_tables_from_config_dat,
};
use crate::error::Result;
use crate::{RhorrpFmsInputHandoff, RhorrpPhaseBinHandoff, RhorrpPotInputControls};

use super::common::invalid_pot_bin;
use super::types::PotBinData;
use super::validate::validate_pot_bin;

/// FEFF `rdpot` source-grid spacing used by RHORRP before `fixvar`/`fixdsx`.
pub const RHORRP_POT_BIN_RADIAL_DX: Real = 0.05;
/// FEFF RHORRP logarithmic radial-grid offset, `x0`.
pub const RHORRP_WAVEFUNCTION_RADIAL_X0: Real = 8.8;
/// FEFF RHORRP wavefunction radial table length, `nr`.
pub const RHORRP_WAVEFUNCTION_RADIAL_COUNT: usize = 251;

/// Parsed FEFF handoff files needed to run RHORRP `init_wavefunctions`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionTablesHandoffInput<'a> {
    /// FEFF `pot.bin` data from `rdpot`.
    pub pot: &'a PotBinData,
    /// FEFF `config.dat` electron occupations from `getorb`.
    pub config: &'a ConfigDatData,
    /// RHORRP contour data from FEFF `phase.bin`.
    pub phase: &'a RhorrpPhaseBinHandoff,
    /// RHORRP FMS controls from FEFF `fms.inp`.
    pub fms: RhorrpFmsInputHandoff,
    /// RHORRP POT controls from FEFF `pot.inp`.
    pub controls: RhorrpPotInputControls,
    /// Target RHORRP radial table length, normally `nr = 251`.
    pub radial_count: usize,
}

/// Owned result of the FEFF RHORRP `init_wavefunctions` handoff sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpWavefunctionTablesHandoff {
    /// Prepared grids after `fixvar`, `fixdsx`, and potential-local `eref0` shifts.
    pub prepared: RhorrpWavefunctionGridPreparation,
    /// Final module-level `eref0` used later by `rhoerrp`.
    pub reference_energy_hartree: Complex,
    /// All-potential `ph2`, `prel`, `qrel`, `pnel`, and `qnel` tables.
    pub wavefunctions: RhorrpWavefunctionTables,
    /// FEFF logarithmic radial-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic radial-grid step `dx`.
    pub radial_dx: Real,
}

/// Borrowed RHORRP wavefunction handoff data from FEFF `pot.bin`.
///
/// FEFF `RHORRP/rhorrp.f90::rhorrp_init` reads `pot.bin`, then
/// `init_wavefunctions` forwards these fixed potential, bound-spinor, and
/// coefficient arrays through `fixvar`, `fixdsx`, and `dfovrg`. The adjusted
/// compact orbital occupations are supplied separately by FEFF `getorb`; Rust
/// callers should pair this view with [`RhorrpConfigOrbitalTables`].
#[derive(Debug, Clone)]
pub struct RhorrpPotBinWavefunctionHandoff<'a> {
    pot: &'a PotBinData,
    muffin_tin_radii: &'a [Real],
    norman_radii: &'a [Real],
    atomic_numbers: Vec<Real>,
}

impl RhorrpPotBinWavefunctionHandoff<'_> {
    /// Number of FEFF potential blocks.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.pot.potential_count()
    }

    /// Muffin-tin radii `rmt(iph)` in Bohr.
    #[must_use]
    pub fn muffin_tin_radii(&self) -> &[Real] {
        self.muffin_tin_radii
    }

    /// Norman radii `rnrm(iph)` in Bohr.
    #[must_use]
    pub fn norman_radii(&self) -> &[Real] {
        self.norman_radii
    }

    /// Atomic numbers `iz(iph)` converted to the `Real` type used by FOVRG.
    #[must_use]
    pub fn atomic_numbers(&self) -> &[Real] {
        &self.atomic_numbers
    }

    /// Build the FEFF `init_wavefunctions` grid-preparation input.
    ///
    /// `target_radial_dx` is FEFF `rgrd`; `exchange_index` is FEFF `ixc`.
    /// `radial_count` is normally `nrptx`.
    #[must_use]
    pub fn grid_preparation_input(
        &self,
        target_radial_dx: Real,
        exchange_index: i32,
        radial_count: usize,
    ) -> RhorrpWavefunctionGridPreparationInput<'_> {
        RhorrpWavefunctionGridPreparationInput {
            muffin_tin_radii: self.muffin_tin_radii,
            electron_density: self.pot.electron_density.view(),
            total_potential: self.pot.total_potential.view(),
            valence_density: self.pot.valence_density.view(),
            valence_potential: self.pot.valence_potential.view(),
            magnetization: self.pot.magnetization_density.view(),
            bound_large_components: self.pot.large_components.view(),
            bound_small_components: self.pot.small_components.view(),
            interstitial_potential: self.pot.scalars.interstitial_potential,
            interstitial_density: self.pot.scalars.interstitial_density,
            original_radial_dx: RHORRP_POT_BIN_RADIAL_DX,
            target_radial_dx,
            jump_mode: self.pot.jump_mode,
            potential_jump: 0.0,
            exchange_index,
            radial_count,
        }
    }

    /// Build the all-potential FEFF `init_wavefunctions` table input.
    ///
    /// The resulting input combines `pot.bin` bound spinor coefficients and
    /// SCF valence occupations with already-adjusted compact total occupations
    /// from `config.dat`.
    pub fn prepared_wavefunction_tables_input<'a>(
        &'a self,
        prepared: &'a RhorrpWavefunctionGridPreparation,
        orbital_tables: &'a RhorrpConfigOrbitalTables,
        energies_hartree: ArrayView1<'a, Complex>,
        exchange_index: i32,
        angular_momentum_count: usize,
    ) -> Result<RhorrpPreparedWavefunctionTablesInput<'a>> {
        validate_prepared_table_sources(self, prepared, orbital_tables)?;

        Ok(RhorrpPreparedWavefunctionTablesInput {
            prepared,
            energies_hartree,
            muffin_tin_radii: self.muffin_tin_radii,
            norman_radii: self.norman_radii,
            bound_large_coefficients_by_potential: self.pot.large_coefficients.view(),
            bound_small_coefficients_by_potential: self.pot.small_coefficients.view(),
            electron_counts_by_potential: orbital_tables.electron_counts_by_potential.view(),
            valence_counts_by_potential: self.pot.orbital_occupancy.view(),
            kappa_by_potential: orbital_tables.kappa_by_potential.view(),
            atomic_numbers: &self.atomic_numbers,
            exchange_index,
            angular_momentum_count,
            bound_orbital_counts: &orbital_tables.bound_orbital_counts,
        })
    }
}

/// Borrow FEFF `pot.bin` data in the shape consumed by RHORRP wavefunction setup.
pub fn rhorrp_wavefunction_handoff_from_pot_bin(
    pot: &PotBinData,
) -> Result<RhorrpPotBinWavefunctionHandoff<'_>> {
    validate_pot_bin(pot)?;
    let muffin_tin_radii = contiguous_potential_slice("rmt", &pot.muffin_tin_radii)?;
    let norman_radii = contiguous_potential_slice("rnrm", &pot.norman_radii)?;
    let atomic_numbers = pot
        .atomic_numbers
        .iter()
        .map(|&atomic_number| atomic_number as Real)
        .collect::<Vec<_>>();

    Ok(RhorrpPotBinWavefunctionHandoff {
        pot,
        muffin_tin_radii,
        norman_radii,
        atomic_numbers,
    })
}

/// Compose parsed FEFF handoff files into RHORRP `init_wavefunctions` tables.
///
/// This mirrors the non-I/O part of `RHORRP/m_rhorrp.f90::rhorrp_init` after
/// `rdpot`, `init_rdxsph`, and `fms_read`: prepare the potential grids,
/// compact `config.dat` occupations, build all-potential wavefunction tables,
/// and expose the final `eref0` scalar retained by `rhoerrp`.
pub fn rhorrp_wavefunction_tables_from_handoffs(
    input: RhorrpWavefunctionTablesHandoffInput<'_>,
) -> Result<RhorrpWavefunctionTablesHandoff> {
    let pot = rhorrp_wavefunction_handoff_from_pot_bin(input.pot)?;
    validate_rhorrp_wavefunction_handoff_counts(&pot, input.config, input.phase, input.fms)?;

    let orbital_tables = rhorrp_orbital_tables_from_config_dat(input.config)?;
    let prepared = rhorrp_prepare_wavefunction_grids(pot.grid_preparation_input(
        input.controls.target_radial_dx,
        input.controls.exchange_index,
        input.radial_count,
    ))
    .map_err(|source| invalid_pot_bin("rhorrp_wavefunction_grids", source.to_string()))?;
    let tables_input = pot.prepared_wavefunction_tables_input(
        &prepared,
        &orbital_tables,
        input.phase.energies_hartree.view(),
        input.controls.exchange_index,
        input.fms.angular_momentum_count,
    )?;
    let wavefunctions = rhorrp_prepared_wavefunction_tables(tables_input)
        .map_err(|source| invalid_pot_bin("rhorrp_wavefunction_tables", source.to_string()))?;
    let reference_energy_hartree = rhorrp_density_reference_energy_hartree(&prepared)
        .map_err(|source| invalid_pot_bin("rhorrp_eref0", source.to_string()))?;
    let radial_dx = prepared.radial_dx;

    Ok(RhorrpWavefunctionTablesHandoff {
        prepared,
        reference_energy_hartree,
        wavefunctions,
        radial_x0: RHORRP_WAVEFUNCTION_RADIAL_X0,
        radial_dx,
    })
}

fn contiguous_potential_slice<'a>(
    field: &'static str,
    values: &'a ndarray::Array1<Real>,
) -> Result<&'a [Real]> {
    values
        .as_slice()
        .ok_or_else(|| invalid_pot_bin(field, "potential vector must be contiguous"))
}

fn validate_rhorrp_wavefunction_handoff_counts(
    pot: &RhorrpPotBinWavefunctionHandoff<'_>,
    config: &ConfigDatData,
    phase: &RhorrpPhaseBinHandoff,
    fms: RhorrpFmsInputHandoff,
) -> Result<()> {
    let potential_count = pot.potential_count();
    validate_rhorrp_potential_axis("config.dat", potential_count, config.potential_count())?;
    validate_rhorrp_potential_axis("phase.bin", potential_count, phase.potential_count())?;
    validate_rhorrp_potential_axis("fms.inp", potential_count, fms.potential_count)?;
    Ok(())
}

fn validate_prepared_table_sources(
    handoff: &RhorrpPotBinWavefunctionHandoff<'_>,
    prepared: &RhorrpWavefunctionGridPreparation,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> Result<()> {
    let potential_count = handoff.potential_count();
    if prepared.potential_count() != potential_count {
        return Err(invalid_pot_bin(
            "rhorrp_prepared",
            format!(
                "prepared wavefunctions have {} potential(s), expected {potential_count}",
                prepared.potential_count()
            ),
        ));
    }

    validate_rhorrp_potential_axis(
        "config electron_counts",
        potential_count,
        orbital_tables.electron_counts_by_potential.dim().1,
    )?;
    validate_rhorrp_potential_axis(
        "config kappa",
        potential_count,
        orbital_tables.kappa_by_potential.dim().1,
    )?;
    validate_rhorrp_potential_axis(
        "config bound_orbital_counts",
        potential_count,
        orbital_tables.bound_orbital_counts.len(),
    )?;

    let valence_shape = handoff.pot.orbital_occupancy.dim();
    if valence_shape.1 != potential_count {
        return Err(invalid_pot_bin(
            "xnval",
            format!(
                "valence occupation table has {} potential(s), expected {potential_count}",
                valence_shape.1
            ),
        ));
    }
    let max_bound_orbitals = orbital_tables
        .bound_orbital_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if valence_shape.0 < max_bound_orbitals {
        return Err(invalid_pot_bin(
            "xnval",
            format!(
                "valence occupation table has {} orbital row(s), expected at least {max_bound_orbitals}",
                valence_shape.0
            ),
        ));
    }

    Ok(())
}

fn validate_rhorrp_potential_axis(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_pot_bin(
            "rhorrp_config",
            format!("{field} has {actual} potential(s), expected {expected}"),
        ))
    }
}
