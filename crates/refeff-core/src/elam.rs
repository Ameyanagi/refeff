//! Elam edge-energy table adapters used by FEFF.
//!
//! FEFF's `XSPH/getedg.f90` stores corrected Williams/Elam edge energies in
//! eV, then converts them to Hartree for phase and full-spectrum grids.  This
//! module keeps the positive table entries and exposes explicit
//! `Result<Option<_>>` behavior instead of FEFF's implicit unchanged-output
//! convention for missing entries.

use crate::{FEFF_HARTREE_EV, Real};

mod table;

use table::ELAM_EDGE_ENERGIES_EV;

/// Highest atomic number covered by FEFF's Elam edge table.
pub const ELAM_EDGE_ATOMIC_NUMBER_MAX: i32 = 100;

/// Number of FEFF hole indices stored in each Elam table row.
pub const ELAM_EDGE_HOLE_COUNT: i32 = 29;

/// FEFF `nexted` sentinel returned when no higher edge exists.
pub const ELAM_NEXT_EDGE_SENTINEL_HARTREE: Real = 1.0e8;

/// Error returned by FEFF Elam edge-energy helpers.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ElamError {
    /// FEFF atomic numbers are one-based.
    #[error("atomic number must be positive, got {z}")]
    InvalidAtomicNumber { z: i32 },
    /// `preved` and `nexted` require table-backed components.
    #[error("Elam edge table supports atomic numbers through Z={max}, got {z}")]
    AtomicNumberOutOfRange { z: i32, max: i32 },
    /// FEFF `getedg` scans hole indices `1..=29`.
    #[error("hole index must be in 1..={max}, got {ihole}")]
    InvalidHole { ihole: i32, max: i32 },
    /// Edge scans require a finite current energy.
    #[error("{name} must be finite, got {value}")]
    NonFiniteEnergy { name: &'static str, value: Real },
}

/// One positive edge entry from FEFF's Elam table, converted to Hartree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElamEdgeEnergy {
    /// One-based FEFF core-hole index.
    pub hole_index: i32,
    /// Edge onset in Hartree.
    pub energy_hartree: Real,
}

/// Return FEFF Elam edge energy in eV.
///
/// This ports the table lookup in `XSPH/getedg.f90`. Missing or negative FEFF
/// table entries return `Ok(None)`. Atomic numbers above the table limit also
/// return `Ok(None)`, matching FEFF's `SETEDGE`-ignore branch.
pub fn elam_edge_energy_ev(z: i32, ihole: i32) -> Result<Option<Real>, ElamError> {
    validate_hole(ihole)?;
    if z <= 0 {
        return Err(ElamError::InvalidAtomicNumber { z });
    }
    if z > ELAM_EDGE_ATOMIC_NUMBER_MAX {
        return Ok(None);
    }

    let row_index = usize::try_from(z - 1).map_err(|_| ElamError::InvalidAtomicNumber { z })?;
    let row = ELAM_EDGE_ENERGIES_EV[row_index];
    Ok(row
        .iter()
        .find_map(|&(hole, energy)| (hole == ihole).then_some(Real::from(energy))))
}

/// Return FEFF Elam edge energy in Hartree.
pub fn elam_edge_energy_hartree(z: i32, ihole: i32) -> Result<Option<Real>, ElamError> {
    Ok(elam_edge_energy_ev(z, ihole)?.map(|energy| energy / FEFF_HARTREE_EV))
}

/// Return all positive FEFF Elam edge entries for one component in Hartree.
///
/// This materializes the table row used by `XSPH/preved` and `XSPH/nexted`.
/// Atomic numbers outside the FEFF table return an error so callers do not
/// silently build an incomplete material-wide edge grid.
pub fn elam_component_edge_energies_hartree(z: i32) -> Result<Vec<ElamEdgeEnergy>, ElamError> {
    let row = elam_component_row(z)?;
    Ok(row
        .iter()
        .map(|&(hole_index, energy_ev)| ElamEdgeEnergy {
            hole_index,
            energy_hartree: Real::from(energy_ev) / FEFF_HARTREE_EV,
        })
        .collect())
}

/// Return the closest FEFF Elam edge below `current_energy` in Hartree.
///
/// This ports `XSPH/preved` and returns FEFF's `0.0` sentinel when no lower
/// edge exists.
pub fn previous_elam_edge_hartree(
    current_energy: Real,
    atomic_numbers: &[i32],
) -> Result<Real, ElamError> {
    validate_energy("current_energy", current_energy)?;
    atomic_numbers.iter().try_fold(0.0, |previous, &z| {
        let row = elam_component_row(z)?;
        Ok(row
            .iter()
            .map(|&(_, energy)| Real::from(energy) / FEFF_HARTREE_EV)
            .filter(|&energy| energy < current_energy && energy > previous)
            .fold(previous, Real::max))
    })
}

/// Return the closest FEFF Elam edge above `current_energy` in Hartree.
///
/// This ports `XSPH/nexted` and returns FEFF's `1.0e8` sentinel when no higher
/// edge exists.
pub fn next_elam_edge_hartree(
    current_energy: Real,
    atomic_numbers: &[i32],
) -> Result<Real, ElamError> {
    validate_energy("current_energy", current_energy)?;
    atomic_numbers
        .iter()
        .try_fold(ELAM_NEXT_EDGE_SENTINEL_HARTREE, |next, &z| {
            let row = elam_component_row(z)?;
            Ok(row
                .iter()
                .map(|&(_, energy)| Real::from(energy) / FEFF_HARTREE_EV)
                .filter(|&energy| energy > current_energy && energy < next)
                .fold(next, Real::min))
        })
}

fn validate_hole(ihole: i32) -> Result<(), ElamError> {
    if (1..=ELAM_EDGE_HOLE_COUNT).contains(&ihole) {
        Ok(())
    } else {
        Err(ElamError::InvalidHole {
            ihole,
            max: ELAM_EDGE_HOLE_COUNT,
        })
    }
}

fn validate_energy(name: &'static str, value: Real) -> Result<(), ElamError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ElamError::NonFiniteEnergy { name, value })
    }
}

fn elam_component_row(z: i32) -> Result<&'static [(i32, f32)], ElamError> {
    if z <= 0 {
        return Err(ElamError::InvalidAtomicNumber { z });
    }
    if z > ELAM_EDGE_ATOMIC_NUMBER_MAX {
        return Err(ElamError::AtomicNumberOutOfRange {
            z,
            max: ELAM_EDGE_ATOMIC_NUMBER_MAX,
        });
    }
    let row_index = usize::try_from(z - 1).map_err(|_| ElamError::InvalidAtomicNumber { z })?;
    Ok(ELAM_EDGE_ENERGIES_EV[row_index])
}

#[cfg(test)]
mod tests;
