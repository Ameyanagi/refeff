//! Canonical FEFF physical constants, with explicitly named legacy variants.
//!
//! FEFF hardcodes the Hartree-to-eV and Bohr-to-Angstrom conversions
//! independently in most Fortran modules instead of sharing one canonical
//! value, and the literals occasionally disagree in the low digits.
//! [`HARTREE_EV`] and [`BOHR_ANGSTROM`] are the canonical values from
//! `COMMON/m_constants.f90`; every other value used in this crate is a
//! byte-identical *legacy* variant, named for and doc-commented with the
//! FEFF Fortran source that hardcodes it. This module only deduplicates the
//! `const` declarations that used to live in each port module — the numeric
//! value used at every call site is unchanged, since a changed legacy value
//! would change FEFF-format output.

use crate::Real;

/// Canonical FEFF Hartree energy in eV, `hart = 2.0_dp*ryd` from
/// `COMMON/m_constants.f90` (`ryd = 13.605698`).
pub const HARTREE_EV: Real = 27.211_396;

/// Canonical FEFF Bohr radius in Angstrom, `bohr` from
/// `COMMON/m_constants.f90`.
pub const BOHR_ANGSTROM: Real = 0.529_177_249;

/// Legacy Hartree/eV conversion hardcoded in `SFCONV/so2conv.f90`
/// (`parameter (eV=1.d0/27.21160d0)`), duplicated verbatim in
/// `SFCONV/mkspectf.f90` and `Utility/edgec.f`.
pub const HARTREE_EV_SFCONV_LEGACY: Real = 27.21160;

/// Legacy Bohr/Angstrom conversion hardcoded in `SFCONV/so2conv.f90`
/// (`parameter (aangstrom=1.d0/0.52917706d0)`), duplicated verbatim in
/// `SFCONV/mkspectf.f90` and `Utility/edgec.f`.
pub const BOHR_ANGSTROM_SFCONV_LEGACY: Real = 0.529_177_06;

/// Legacy Hartree/eV conversion hardcoded in `DMDW/m_dmdw.f90`
/// (`a2fall(1,j)=w*27.211396132`).
pub const HARTREE_EV_DMDW_COUPLING_LEGACY: Real = 27.211_396_132;

/// Legacy Bohr/Angstrom conversion hardcoded in
/// `EELS/writeangulardependence3.f90` (`a0=dble(0.529177)`).
pub const BOHR_ANGSTROM_EELS_LEGACY: Real = 0.529177;
