//! FEFF `apot.bin` TXT section-stream codec.
//!
//! The atomic-potential stage writes `apot.bin` through FEFF's generic
//! `WriteData`, `WriteArrayData`, and `Write2D` helpers. The generated file is
//! a text stream of `#SN#` sections even though the suffix is `.bin`. This
//! module keeps the stream typed while preserving section headers and FEFF's
//! column-major matrix shapes in [`ndarray::Array2`] values.

mod atomic_sections;
mod common;
mod core_hole;
mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use atomic_sections::{
    APOT_ATOMIC_COEFFICIENTS, APOT_ATOMIC_COULOMB_SECTION_NUMBER,
    APOT_ATOMIC_DENSITY_SECTION_NUMBER, APOT_ATOMIC_KAPPA_SECTION_NUMBER,
    APOT_ATOMIC_NORB_SECTION_NUMBER, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER,
    APOT_ATOMIC_ORBITAL_SECTION_START, APOT_ATOMIC_ORBITAL_SLOTS, APOT_ATOMIC_RADIAL_POINTS,
    APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER, APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
    ApotAtomicPotsSectionsInput, ApotAtomicScfStateRef, ApotAtomicScfStateSectionsInput,
    apot_atomic_pots_sections, apot_atomic_scf_sections, apot_atomic_scf_sections_from_states,
    apot_atomic_scf_state_sections, apot_atomic_scf_state_sections_from_state,
};
pub use core_hole::{
    APOT_CORE_HOLE_GRID_ORIGIN, APOT_CORE_HOLE_GRID_STEP, APOT_CORE_HOLE_RADIAL_POINTS,
    APOT_CORE_HOLE_SECTION_NUMBER, APOT_CORE_HOLE_TOLERANCE, ApotCoreHoleColumns,
    apot_core_hole_columns, apot_core_hole_coulomb_from_density, apot_core_hole_radii,
    refresh_apot_core_hole_coulomb_payload,
};
pub use parse::parse_apot_bin;
pub use render::{apot_bin_string, read_apot_bin, write_apot_bin};
pub use types::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords,
    ApotBinSection, ApotBinType, ApotBinValue,
};
