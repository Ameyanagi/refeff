//! SCREEN geometry and phase-potential setup helpers.

use ndarray::{Array2, ShapeBuilder};

use super::constants::{SCREEN_BOHR_ANGSTROM, SCREEN_HARTREE_EV};
use super::types::*;
use super::validation::{
    validate_count_at_least, validate_finite, validate_finite_matrix, validate_result_finite,
};

/// Port the unit setup block from SCREEN `rdgeom.f90`.
///
/// FEFF clamps `ScreenI%maxl` to `lx + 1`, converts atomic coordinates and
/// FMS radii from Angstrom to bohr, and converts SCREEN contour energies from
/// eV to Hartree before the screening driver starts. This helper keeps that
/// setup separate from the full file-reading routine so callers can apply it
/// to already-parsed Rust inputs.
pub fn screen_rdgeom_atomic_units(
    input: ScreenRdgeomAtomicUnitsInput<'_>,
) -> Result<ScreenRdgeomAtomicUnits, ScreenError> {
    let (_, columns) = input.atom_positions_angstrom.dim();
    if columns != 3 {
        return Err(ScreenError::AtomPositionColumnCount { columns });
    }

    validate_finite("rfms2_angstrom", input.rfms2_angstrom)?;
    validate_finite("direct_radius_angstrom", input.direct_radius_angstrom)?;
    validate_finite("min_real_energy_ev", input.min_real_energy_ev)?;
    validate_finite("max_real_energy_ev", input.max_real_energy_ev)?;
    validate_finite("max_imaginary_energy_ev", input.max_imaginary_energy_ev)?;
    validate_finite("screen_rfms_angstrom", input.screen_rfms_angstrom)?;
    validate_finite("min_imaginary_energy_ev", input.min_imaginary_energy_ev)?;

    let mut atom_positions_bohr =
        Array2::zeros((input.atom_positions_angstrom.nrows(), columns).f());
    for ((row, column), value) in input.atom_positions_angstrom.indexed_iter() {
        validate_finite_matrix("atom_positions_angstrom", row, column, *value)?;
        let converted = *value / SCREEN_BOHR_ANGSTROM;
        validate_result_finite("atom_position_bohr", converted)?;
        atom_positions_bohr[(row, column)] = converted;
    }

    let angular_count_cap =
        input
            .angular_capacity_lx
            .checked_add(1)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "angular_capacity_lx",
            })?;
    let converted = ScreenRdgeomAtomicUnits {
        atom_positions_bohr,
        rfms2_bohr: input.rfms2_angstrom / SCREEN_BOHR_ANGSTROM,
        direct_radius_bohr: input.direct_radius_angstrom / SCREEN_BOHR_ANGSTROM,
        min_real_energy_hartree: input.min_real_energy_ev / SCREEN_HARTREE_EV,
        max_real_energy_hartree: input.max_real_energy_ev / SCREEN_HARTREE_EV,
        max_imaginary_energy_hartree: input.max_imaginary_energy_ev / SCREEN_HARTREE_EV,
        screen_rfms_bohr: input.screen_rfms_angstrom / SCREEN_BOHR_ANGSTROM,
        min_imaginary_energy_hartree: input.min_imaginary_energy_ev / SCREEN_HARTREE_EV,
        max_l: input.max_l.min(angular_count_cap),
    };
    validate_result_finite("rfms2_bohr", converted.rfms2_bohr)?;
    validate_result_finite("direct_radius_bohr", converted.direct_radius_bohr)?;
    validate_result_finite("min_real_energy_hartree", converted.min_real_energy_hartree)?;
    validate_result_finite("max_real_energy_hartree", converted.max_real_energy_hartree)?;
    validate_result_finite(
        "max_imaginary_energy_hartree",
        converted.max_imaginary_energy_hartree,
    )?;
    validate_result_finite("screen_rfms_bohr", converted.screen_rfms_bohr)?;
    validate_result_finite(
        "min_imaginary_energy_hartree",
        converted.min_imaginary_energy_hartree,
    )?;

    Ok(converted)
}

/// Port the phase-potential reference shift from SCREEN `prep.f90`.
///
/// After `fixvar`, FEFF chooses `eref(1) = vtotph(jri1)`, subtracts that
/// reference from `vtotph(1:jri1)`, and either subtracts it from
/// `vvalph(1:jri1)` (`ixc >= 5`) or copies the shifted total potential into
/// `vvalph(1:jri1)`. Entries after `jri1` are left untouched, matching the
/// Fortran loop bounds.
pub fn screen_phase_potential_reference_shift(
    input: ScreenPhasePotentialInput<'_>,
) -> Result<ScreenPhasePotential, ScreenError> {
    validate_count_at_least(
        "muffin_tin_next_index_1based",
        input.muffin_tin_next_index_1based,
        1,
    )?;
    if input.muffin_tin_next_index_1based > input.total_potential.len() {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: input.muffin_tin_next_index_1based,
            capacity: input.total_potential.len(),
        });
    }
    if input.muffin_tin_next_index_1based > input.valence_potential.len() {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: input.muffin_tin_next_index_1based,
            capacity: input.valence_potential.len(),
        });
    }

    let prefix_len = input.muffin_tin_next_index_1based;
    let reference_index = prefix_len - 1;
    let reference_energy = input.total_potential[reference_index];
    validate_finite("reference_potential", reference_energy)?;

    let mut total_potential = input.total_potential.to_owned();
    let mut valence_potential = input.valence_potential.to_owned();
    for index in 0..prefix_len {
        let total = input.total_potential[index];
        let valence = input.valence_potential[index];
        validate_finite("total_potential", total)?;
        validate_finite("valence_potential", valence)?;

        let shifted_total = total - reference_energy;
        validate_result_finite("shifted_total_potential", shifted_total)?;
        total_potential[index] = shifted_total;
        valence_potential[index] = if input.exchange_selector >= 5 {
            let shifted_valence = valence - reference_energy;
            validate_result_finite("shifted_valence_potential", shifted_valence)?;
            shifted_valence
        } else {
            shifted_total
        };
    }

    Ok(ScreenPhasePotential {
        reference_energy,
        total_potential,
        valence_potential,
    })
}
