//! FEFF `ldosNN.dat` and `rhocNN.dat` local density-of-states text codecs.
//!
//! FEFF writes non-spin LDOS files with energy plus `s`, `p`, `d`, and,
//! when the potential angular cutoff includes it, `f` orbital density
//! columns. Spin-resolved LDOS output keeps the same energy column and appends
//! orbital channels for spin up followed by the same channels for spin down;
//! older Hubbard references can stop at `d` and omit the `f` pair.
//!
//! The LDOS module also writes `rhocNN.dat` embedded-atom reference density
//! files from the same `ff2rho` data path. Those files omit the descriptive
//! header but keep the same energy plus angular-momentum table shape, so the
//! explicit `rhoc` helpers intentionally share this data model.
//!
//! Hubbard LDOS also writes magnetic-orbital `lmdosNN.dat` and `rhocmNN.dat`
//! tables. Those rows carry spin-major `(l,m)` density columns instead of the
//! ordinary four orbital channels, so they use a separate typed model below.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
use num_complex::Complex64;
use refeff_core::{
    FEFF_HARTREE_EV, LdosFf2rhoInput, LdosHubbardMagneticFf2rhoInput, LdosSpinFf2rhoInput,
    ldos_ff2rho_tables, ldos_hubbard_magnetic_ff2rho_tables, ldos_spin_ff2rho_tables,
};

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const LDOS_DAT_TRUNCATED_NON_SPIN_ROW_WIDTH: usize = 4;
const LDOS_DAT_NON_SPIN_ROW_WIDTH: usize = 5;
const LDOS_DAT_TRUNCATED_SPIN_ROW_WIDTH: usize = 7;
const LDOS_DAT_SPIN_ROW_WIDTH: usize = 9;
const LDOS_DAT_TRUNCATED_NON_SPIN_DENSITY_COLUMNS: usize = 3;
const LDOS_DAT_NON_SPIN_DENSITY_COLUMNS: usize = 4;
const LDOS_DAT_TRUNCATED_SPIN_DENSITY_COLUMNS: usize = 6;
const LDOS_DAT_SPIN_DENSITY_COLUMNS: usize = 8;
const LDOS_DAT_ALLOWED_ROW_WIDTHS: &str = "4, 5, 7, or 9";
const LDOS_DAT_ALLOWED_DENSITY_COLUMNS: &str = "3, 4, 6, or 8";
const LDOS_MAGNETIC_ALLOWED_ROW_WIDTHS: &str = "1 + 2 * (lx + 1)^2";
const LDOS_MAGNETIC_ALLOWED_DENSITY_COLUMNS: &str = "2 * (lx + 1)^2";

/// FEFF non-spin LDOS column labels after the energy column.
pub const LDOS_ORBITAL_LABELS: [&str; LDOS_DAT_NON_SPIN_DENSITY_COLUMNS] =
    ["sDOS", "pDOS", "dDOS", "fDOS"];

/// FEFF spin-resolved LDOS column labels after the energy column.
pub const LDOS_SPIN_ORBITAL_LABELS: [&str; LDOS_DAT_SPIN_DENSITY_COLUMNS] = [
    "sDOS_up",
    "pDOS_up",
    "dDOS_up",
    "fDOS_up",
    "sDOS_down",
    "pDOS_down",
    "dDOS_down",
    "fDOS_down",
];

/// Parsed electron count from a FEFF `ldosNN.dat` header.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosElectronCount {
    /// Orbital angular momentum quantum number.
    pub angular_momentum: usize,
    /// Electron count in the corresponding angular momentum channel.
    pub count: f64,
}

/// Parsed FEFF `ldosNN.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosDatData {
    /// Header and comment lines before and around the numeric LDOS table.
    pub header_lines: Vec<String>,
    /// Fermi level in eV when present in the header.
    pub fermi_level_ev: Option<f64>,
    /// Charge transfer when present in the header.
    pub charge_transfer: Option<f64>,
    /// Header electron counts by orbital angular momentum.
    pub electron_counts: Vec<LdosElectronCount>,
    /// Number of atoms in the cluster when present in the header.
    pub atom_count: Option<usize>,
    /// Lorentzian half-width at half-height broadening in eV when present.
    pub lorentzian_hwhh_ev: Option<f64>,
    /// Energy grid in eV.
    pub energy_ev: Array1<f64>,
    /// DOS columns. Non-spin files have three or four columns in
    /// [`LDOS_ORBITAL_LABELS`] order; spin-resolved files usually have eight
    /// columns in [`LDOS_SPIN_ORBITAL_LABELS`] order, with older Hubbard
    /// references sometimes carrying six `s,p,d` spin columns.
    pub density: Array2<f64>,
}

/// FEFF FULLSPECTRUM `rdldos.f90` view of an `ldosNN.dat` table.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumLdosData {
    /// Fermi level from the LDOS header, converted from eV to Hartree.
    pub fermi_level_hartree: f64,
    /// Photon-energy grid, converted from eV to Hartree.
    pub energy_hartree: Array1<f64>,
    /// `s`, `p`, `d`, and `f` DOS columns converted to states/Hartree/atom.
    pub density_states_per_hartree_atom: Array2<f64>,
}

/// Parsed FEFF `rhocNN.dat` contents.
///
/// FEFF `rhocNN.dat` files use the same energy-grid and angular-momentum
/// density table as non-spin `ldosNN.dat`, usually without header metadata.
pub type RhocDatData = LdosDatData;

/// Parsed FEFF Hubbard magnetic-orbital LDOS text contents.
///
/// FEFF writes `lmdosNN.dat` and `rhocmNN.dat` rows in spin-major order:
/// all `(l,m)` columns for spin up followed by the same `(l,m)` columns for
/// spin down. For each spin, the magnetic columns traverse `l = 0..lx` and
/// `m = l**2 + 1..(l + 1)**2`, yielding `(lx + 1)^2` columns per spin.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosMagneticDatData {
    /// Header and comment lines before and around the numeric table.
    pub header_lines: Vec<String>,
    /// Fermi level in eV when present in the header.
    pub fermi_level_ev: Option<f64>,
    /// Charge transfer when present in the header.
    pub charge_transfer: Option<f64>,
    /// Header electron counts by orbital/angular-magnetic momentum.
    pub electron_counts: Vec<LdosElectronCount>,
    /// Number of atoms in the cluster when present in the header.
    pub atom_count: Option<usize>,
    /// Lorentzian half-width at half-height broadening in eV when present.
    pub lorentzian_hwhh_ev: Option<f64>,
    /// Highest angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Energy grid in eV.
    pub energy_ev: Array1<f64>,
    /// Magnetic DOS columns as `(energy, spin-major magnetic column)`.
    pub density: Array2<f64>,
}

/// Parsed FEFF Hubbard `lmdosNN.dat` text contents.
pub type LmdosDatData = LdosMagneticDatData;

/// Parsed FEFF Hubbard `rhocmNN.dat` text contents.
pub type RhocmDatData = LdosMagneticDatData;

/// Inputs for building FEFF `ldosNN.dat`/`rhocNN.dat` from `ff2rho` work arrays.
#[derive(Debug, Clone, Copy)]
pub struct LdosDatFromFf2rhoInput<'a> {
    /// Optional header/comment lines for `ldosNN.dat`; default FEFF headers are
    /// generated when this is empty.
    pub header_lines: &'a [String],
    /// Fermi level `xmu`, in Hartree.
    pub fermi_level_hartree: Option<f64>,
    /// Charge transfer `qnrm(iph)`.
    pub charge_transfer: Option<f64>,
    /// Electron counts `xnmues(l,iph)`.
    pub electron_counts: &'a [LdosElectronCount],
    /// Cluster inclusion count `inclus(iph)`.
    pub atom_count: Option<usize>,
    /// Lorentzian HWHH broadening, usually `dimag(em(1))`, in Hartree.
    pub lorentzian_hwhh_hartree: Option<f64>,
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex64>,
    /// Embedded-atom LDOS, FEFF `xrhoce(l,ie)`.
    pub embedded_ldos: ArrayView2<'a, f64>,
    /// Scattering LDOS, FEFF `xrhole(l,ie)`.
    pub scattering_ldos: ArrayView2<'a, Complex64>,
    /// FMS trace copied into FEFF `cchi(l,ie)`.
    pub scattering_trace: ArrayView2<'a, Complex64>,
    /// Active angular-momentum channel count, FEFF runtime `lx + 1`.
    pub angular_count: usize,
    /// Whether FEFF applies the `msapp.ne.1` scattering correction.
    pub apply_scattering: bool,
}

/// Inputs for building spin-resolved FEFF `ldosNN.dat`/`rhocNN.dat` from
/// `ff2rho_h` work arrays.
#[derive(Debug, Clone, Copy)]
pub struct LdosSpinDatFromFf2rhoInput<'a> {
    /// Optional header/comment lines for `ldosNN.dat`; default FEFF headers are
    /// generated when this is empty.
    pub header_lines: &'a [String],
    /// Fermi level `xmu`, in Hartree.
    pub fermi_level_hartree: Option<f64>,
    /// Charge transfer `qnrm(iph)`.
    pub charge_transfer: Option<f64>,
    /// Electron counts `xnmues(l,iph)`.
    pub electron_counts: &'a [LdosElectronCount],
    /// Cluster inclusion count `inclus(iph)`.
    pub atom_count: Option<usize>,
    /// Lorentzian HWHH broadening, usually `dimag(em(1))`, in Hartree.
    pub lorentzian_hwhh_hartree: Option<f64>,
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex64>,
    /// Embedded spin LDOS, FEFF `xrhoce(l,is,ie)`.
    pub embedded_ldos: ArrayView3<'a, f64>,
    /// Scattering spin LDOS, FEFF `xrhole(l,is,ie)`.
    pub scattering_ldos: ArrayView3<'a, Complex64>,
    /// FMS spin trace copied into FEFF `cchi(l,is,ie)`.
    pub scattering_trace: ArrayView3<'a, Complex64>,
    /// Whether FEFF applies the `msapp.ne.1` scattering correction.
    pub apply_scattering: bool,
}

/// Inputs for building FEFF Hubbard `lmdosNN.dat`/`rhocmNN.dat` from
/// `ff2rho_h_step2` magnetic-orbital work arrays.
#[derive(Debug, Clone, Copy)]
pub struct LdosMagneticDatFromFf2rhoInput<'a> {
    /// Optional header/comment lines for `lmdosNN.dat`; default FEFF headers are
    /// generated when this is empty.
    pub header_lines: &'a [String],
    /// Fermi level `xmu`, in Hartree.
    pub fermi_level_hartree: Option<f64>,
    /// Charge transfer `qnrm(iph)`.
    pub charge_transfer: Option<f64>,
    /// Electron counts `xnmues(l,iph)`.
    pub electron_counts: &'a [LdosElectronCount],
    /// Cluster inclusion count `inclus(iph)`.
    pub atom_count: Option<usize>,
    /// Lorentzian HWHH broadening, usually `dimag(em(1))`, in Hartree.
    pub lorentzian_hwhh_hartree: Option<f64>,
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex64>,
    /// Embedded magnetic LDOS, FEFF `xmrhoce(l,im,is,ie)`.
    pub embedded_magnetic_ldos: ndarray::ArrayView4<'a, f64>,
    /// Scattering magnetic LDOS, FEFF `xmrhole(l,im,is,ie)`.
    pub scattering_magnetic_ldos: ndarray::ArrayView4<'a, Complex64>,
    /// Magnetic FMS trace, FEFF `gtr_m(l,im,is,iph,ie)` after selecting `iph`.
    pub magnetic_scattering_trace: ndarray::ArrayView4<'a, Complex64>,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
}

/// FEFF LDOS and embedded-density text payloads from the `ff2rho` handoff.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosDatFromFf2rho {
    /// Renderable FEFF `ldosNN.dat` payload.
    pub ldos: LdosDatData,
    /// Renderable FEFF `rhocNN.dat` payload.
    pub rhoc: RhocDatData,
}

/// FEFF Hubbard magnetic LDOS and embedded-density text payloads from the
/// `ff2rho_h_step2` handoff.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosMagneticDatFromFf2rho {
    /// Renderable FEFF `lmdosNN.dat` payload.
    pub lmdos: LmdosDatData,
    /// Renderable FEFF `rhocmNN.dat` payload.
    pub rhocm: RhocmDatData,
}

impl LdosDatData {
    /// Number of energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Whether this table has spin-resolved up/down orbital channels.
    #[must_use]
    pub fn is_spin_resolved(&self) -> bool {
        matches!(
            self.density.ncols(),
            LDOS_DAT_TRUNCATED_SPIN_DENSITY_COLUMNS | LDOS_DAT_SPIN_DENSITY_COLUMNS
        )
    }
}

impl FullSpectrumLdosData {
    /// Number of energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_hartree.len()
    }

    /// Number of angular momentum channels.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.density_states_per_hartree_atom.ncols()
    }
}

impl LdosMagneticDatData {
    /// Number of energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Number of magnetic `(l,m)` columns per spin, equal to `(lx + 1)^2`.
    #[must_use]
    pub fn magnetic_columns_per_spin(&self) -> usize {
        self.density.ncols() / 2
    }

    /// Number of density columns across both spin projections.
    #[must_use]
    pub fn density_column_count(&self) -> usize {
        self.density.ncols()
    }
}

/// Build FEFF `ldosNN.dat` and `rhocNN.dat` data from `LDOS/ff2rho.f90` arrays.
///
/// This covers the non-full-potential table path: `rhocNN.dat` receives the
/// embedded `xrhoce` table, while `ldosNN.dat` receives the optional
/// `imag(cchi*xrhole)` scattering update.
pub fn ldos_dat_from_ff2rho(input: LdosDatFromFf2rhoInput<'_>) -> Result<LdosDatFromFf2rho> {
    let tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: input.energy_grid_hartree,
        embedded_ldos: input.embedded_ldos,
        scattering_ldos: input.scattering_ldos,
        scattering_trace: input.scattering_trace,
        angular_count: input.angular_count,
        apply_scattering: input.apply_scattering,
    })
    .map_err(|source| invalid_ldos_dat("ff2rho", source.to_string()))?;

    let fermi_level_ev = input
        .fermi_level_hartree
        .map(|value| value * FEFF_HARTREE_EV);
    let lorentzian_hwhh_ev = input
        .lorentzian_hwhh_hartree
        .map(|value| value * FEFF_HARTREE_EV);
    let ldos = LdosDatData {
        header_lines: if input.header_lines.is_empty() {
            default_ldos_header_lines(
                fermi_level_ev,
                input.charge_transfer,
                input.electron_counts,
                input.atom_count,
                lorentzian_hwhh_ev,
            )
        } else {
            input.header_lines.to_vec()
        },
        fermi_level_ev,
        charge_transfer: input.charge_transfer,
        electron_counts: input.electron_counts.to_vec(),
        atom_count: input.atom_count,
        lorentzian_hwhh_ev,
        energy_ev: tables.energy_ev.clone(),
        density: tables.ldos_density,
    };
    validate_ldos_dat(&ldos)?;

    let rhoc = RhocDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: tables.energy_ev,
        density: tables.rhoc_density,
    };
    validate_ldos_dat(&rhoc)?;

    Ok(LdosDatFromFf2rho { ldos, rhoc })
}

/// Build spin-resolved FEFF `ldosNN.dat` and `rhocNN.dat` data from
/// `LDOS/ff2rho_h.f90` arrays.
///
/// The resulting tables use FEFF's spin-major column order: four orbital
/// channels for spin up followed by the same four channels for spin down.
pub fn ldos_spin_dat_from_ff2rho(
    input: LdosSpinDatFromFf2rhoInput<'_>,
) -> Result<LdosDatFromFf2rho> {
    let tables = ldos_spin_ff2rho_tables(LdosSpinFf2rhoInput {
        energy_grid_hartree: input.energy_grid_hartree,
        embedded_ldos: input.embedded_ldos,
        scattering_ldos: input.scattering_ldos,
        scattering_trace: input.scattering_trace,
        angular_count: LDOS_DAT_NON_SPIN_DENSITY_COLUMNS,
        apply_scattering: input.apply_scattering,
    })
    .map_err(|source| invalid_ldos_dat("ff2rho_h", source.to_string()))?;

    let fermi_level_ev = input
        .fermi_level_hartree
        .map(|value| value * FEFF_HARTREE_EV);
    let lorentzian_hwhh_ev = input
        .lorentzian_hwhh_hartree
        .map(|value| value * FEFF_HARTREE_EV);
    let ldos = LdosDatData {
        header_lines: if input.header_lines.is_empty() {
            default_spin_ldos_header_lines(
                fermi_level_ev,
                input.charge_transfer,
                input.electron_counts,
                input.atom_count,
                lorentzian_hwhh_ev,
            )
        } else {
            input.header_lines.to_vec()
        },
        fermi_level_ev,
        charge_transfer: input.charge_transfer,
        electron_counts: input.electron_counts.to_vec(),
        atom_count: input.atom_count,
        lorentzian_hwhh_ev,
        energy_ev: tables.energy_ev.clone(),
        density: tables.ldos_density,
    };
    validate_ldos_dat(&ldos)?;

    let rhoc = RhocDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: tables.energy_ev,
        density: tables.rhoc_density,
    };
    validate_ldos_dat(&rhoc)?;

    Ok(LdosDatFromFf2rho { ldos, rhoc })
}

/// Build FEFF Hubbard `lmdosNN.dat` and `rhocmNN.dat` data from
/// `LDOS/ff2rho_h_step2.f90` magnetic-orbital arrays.
pub fn ldos_magnetic_dat_from_ff2rho(
    input: LdosMagneticDatFromFf2rhoInput<'_>,
) -> Result<LdosMagneticDatFromFf2rho> {
    let tables = ldos_hubbard_magnetic_ff2rho_tables(LdosHubbardMagneticFf2rhoInput {
        energy_grid_hartree: input.energy_grid_hartree,
        embedded_magnetic_ldos: input.embedded_magnetic_ldos,
        scattering_magnetic_ldos: input.scattering_magnetic_ldos,
        magnetic_scattering_trace: input.magnetic_scattering_trace,
        angular_count: input.angular_count,
    })
    .map_err(|source| invalid_ldos_dat("ff2rho_h_step2", source.to_string()))?;

    let angular_limit = input
        .angular_count
        .checked_sub(1)
        .ok_or_else(|| invalid_ldos_dat("angular_count", "must be positive"))?;
    let fermi_level_ev = input
        .fermi_level_hartree
        .map(|value| value * FEFF_HARTREE_EV);
    let lorentzian_hwhh_ev = input
        .lorentzian_hwhh_hartree
        .map(|value| value * FEFF_HARTREE_EV);
    let lmdos = LmdosDatData {
        header_lines: if input.header_lines.is_empty() {
            default_magnetic_ldos_header_lines(
                fermi_level_ev,
                input.charge_transfer,
                input.electron_counts,
                input.atom_count,
                lorentzian_hwhh_ev,
                angular_limit,
            )
        } else {
            input.header_lines.to_vec()
        },
        fermi_level_ev,
        charge_transfer: input.charge_transfer,
        electron_counts: input.electron_counts.to_vec(),
        atom_count: input.atom_count,
        lorentzian_hwhh_ev,
        angular_limit,
        energy_ev: tables.energy_ev.clone(),
        density: tables.lmdos_density,
    };
    validate_ldos_magnetic_dat(&lmdos)?;

    let rhocm = RhocmDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        angular_limit,
        energy_ev: tables.energy_ev,
        density: tables.rhocm_density,
    };
    validate_ldos_magnetic_dat(&rhocm)?;

    Ok(LdosMagneticDatFromFf2rho { lmdos, rhocm })
}

/// Render FEFF-compatible `ldosNN.dat` text.
pub fn ldos_dat_string(data: &LdosDatData) -> Result<String> {
    validate_ldos_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, row) in data.energy_ev.iter().zip(data.density.axis_iter(Axis(0))) {
        write!(out, "{energy:11.4} ")?;
        for (column, value) in row.iter().enumerate() {
            if column > 0 {
                out.push(' ');
            }
            write_fortran_exp(&mut out, *value, 13, 6)?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Render FEFF-compatible `rhocNN.dat` text.
///
/// The rendered table uses the same numeric layout as [`ldos_dat_string`].
pub fn rhoc_dat_string(data: &RhocDatData) -> Result<String> {
    ldos_dat_string(data)
}

/// Render FEFF-compatible Hubbard magnetic-orbital LDOS text.
pub fn ldos_magnetic_dat_string(data: &LdosMagneticDatData) -> Result<String> {
    validate_ldos_magnetic_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, row) in data.energy_ev.iter().zip(data.density.axis_iter(Axis(0))) {
        write!(out, "{energy:11.4} ")?;
        for (column, value) in row.iter().enumerate() {
            if column > 0 {
                out.push(' ');
            }
            write_fortran_exp(&mut out, *value, 13, 6)?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Render FEFF-compatible Hubbard `lmdosNN.dat` text.
pub fn lmdos_dat_string(data: &LmdosDatData) -> Result<String> {
    ldos_magnetic_dat_string(data)
}

/// Render FEFF-compatible Hubbard `rhocmNN.dat` text.
pub fn rhocm_dat_string(data: &RhocmDatData) -> Result<String> {
    ldos_magnetic_dat_string(data)
}

/// Parse FEFF `ldosNN.dat` text.
pub fn parse_ldos_dat(text: &str) -> Result<LdosDatData> {
    let mut header_lines = Vec::new();
    let mut fermi_level_ev = None;
    let mut charge_transfer = None;
    let mut electron_counts = Vec::new();
    let mut atom_count = None;
    let mut lorentzian_hwhh_ev = None;
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut density = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(
                width,
                LDOS_DAT_TRUNCATED_NON_SPIN_ROW_WIDTH
                    | LDOS_DAT_NON_SPIN_ROW_WIDTH
                    | LDOS_DAT_TRUNCATED_SPIN_ROW_WIDTH
                    | LDOS_DAT_SPIN_ROW_WIDTH
            ) {
                return Err(IoError::LdosDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: LDOS_DAT_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::LdosDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: row_width_label(expected),
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            for token in &tokens[1..] {
                density.push(parse_f64(line_number, "density", token)?);
            }
        } else {
            parse_header_metadata(
                line,
                line_number,
                &mut fermi_level_ev,
                &mut charge_transfer,
                &mut electron_counts,
                &mut atom_count,
                &mut lorentzian_hwhh_ev,
            )?;
            header_lines.push(raw.to_string());
        }
    }

    let point_count = energy_ev.len();
    let density_columns = match row_width {
        Some(LDOS_DAT_TRUNCATED_NON_SPIN_ROW_WIDTH) => LDOS_DAT_TRUNCATED_NON_SPIN_DENSITY_COLUMNS,
        Some(LDOS_DAT_NON_SPIN_ROW_WIDTH) => LDOS_DAT_NON_SPIN_DENSITY_COLUMNS,
        Some(LDOS_DAT_TRUNCATED_SPIN_ROW_WIDTH) => LDOS_DAT_TRUNCATED_SPIN_DENSITY_COLUMNS,
        Some(LDOS_DAT_SPIN_ROW_WIDTH) => LDOS_DAT_SPIN_DENSITY_COLUMNS,
        _ => 0,
    };
    let density = if density_columns == 0 {
        Array2::zeros((0, 0))
    } else {
        Array2::from_shape_vec((point_count, density_columns), density).map_err(|_| {
            invalid_ldos_dat("density", "density payload did not match LDOS table shape")
        })?
    };

    let data = LdosDatData {
        header_lines,
        fermi_level_ev,
        charge_transfer,
        electron_counts,
        atom_count,
        lorentzian_hwhh_ev,
        energy_ev: Array1::from_vec(energy_ev),
        density,
    };
    validate_ldos_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `rhocNN.dat` embedded-density text.
///
/// FEFF writes `rhocNN.dat` with the same five-column table accepted by
/// [`parse_ldos_dat`].
pub fn parse_rhoc_dat(text: &str) -> Result<RhocDatData> {
    parse_ldos_dat(text)
}

/// Parse FEFF Hubbard magnetic-orbital LDOS text.
pub fn parse_ldos_magnetic_dat(text: &str) -> Result<LdosMagneticDatData> {
    let mut header_lines = Vec::new();
    let mut fermi_level_ev = None;
    let mut charge_transfer = None;
    let mut electron_counts = Vec::new();
    let mut atom_count = None;
    let mut lorentzian_hwhh_ev = None;
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut density = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if width < 3 {
                return Err(IoError::LdosDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: LDOS_MAGNETIC_ALLOWED_ROW_WIDTHS,
                });
            }
            let density_columns = width - 1;
            magnetic_angular_limit_from_density_columns(density_columns).map_err(|_| {
                IoError::LdosDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: LDOS_MAGNETIC_ALLOWED_ROW_WIDTHS,
                }
            })?;
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::LdosDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: "consistent magnetic LDOS width",
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            for token in &tokens[1..] {
                density.push(parse_f64(line_number, "density", token)?);
            }
        } else {
            parse_header_metadata(
                line,
                line_number,
                &mut fermi_level_ev,
                &mut charge_transfer,
                &mut electron_counts,
                &mut atom_count,
                &mut lorentzian_hwhh_ev,
            )?;
            header_lines.push(raw.to_string());
        }
    }

    let point_count = energy_ev.len();
    let density_columns = row_width.map(|width| width - 1).unwrap_or(0);
    let angular_limit = magnetic_angular_limit_from_density_columns(density_columns)?;
    let density =
        Array2::from_shape_vec((point_count, density_columns), density).map_err(|_| {
            invalid_ldos_dat(
                "density",
                "density payload did not match magnetic LDOS table shape",
            )
        })?;

    let data = LdosMagneticDatData {
        header_lines,
        fermi_level_ev,
        charge_transfer,
        electron_counts,
        atom_count,
        lorentzian_hwhh_ev,
        angular_limit,
        energy_ev: Array1::from_vec(energy_ev),
        density,
    };
    validate_ldos_magnetic_dat(&data)?;
    Ok(data)
}

/// Parse FEFF Hubbard `lmdosNN.dat` magnetic-orbital text.
pub fn parse_lmdos_dat(text: &str) -> Result<LmdosDatData> {
    parse_ldos_magnetic_dat(text)
}

/// Parse FEFF Hubbard `rhocmNN.dat` magnetic-orbital text.
pub fn parse_rhocm_dat(text: &str) -> Result<RhocmDatData> {
    parse_ldos_magnetic_dat(text)
}

/// Convert parsed `ldosNN.dat` content to FEFF `FULLSPECTRUM/rdldos.f90` units.
///
/// The FULLSPECTRUM reader consumes only non-spin LDOS tables with four
/// angular-momentum columns. It converts energies from eV to Hartree and DOS
/// values from states/eV/atom to states/Hartree/atom.
pub fn fullspectrum_ldos_from_ldos_dat(data: &LdosDatData) -> Result<FullSpectrumLdosData> {
    validate_ldos_dat(data)?;
    if data.point_count() < 2 {
        return Err(invalid_ldos_dat(
            "rows",
            "FULLSPECTRUM rdldos requires at least two LDOS rows",
        ));
    }
    if data.is_spin_resolved() || data.density.ncols() != LDOS_DAT_NON_SPIN_DENSITY_COLUMNS {
        return Err(invalid_ldos_dat(
            "density",
            "FULLSPECTRUM rdldos supports only non-spin four-column LDOS data",
        ));
    }
    let fermi_level_ev = data.fermi_level_ev.ok_or_else(|| {
        invalid_ldos_dat(
            "fermi level",
            "FULLSPECTRUM rdldos requires a Fermi-level header",
        )
    })?;

    let fermi_level_hartree = fermi_level_ev / FEFF_HARTREE_EV;
    validate_finite("fermi_level_hartree", fermi_level_hartree)?;

    let energy_hartree = data.energy_ev.mapv(|energy| energy / FEFF_HARTREE_EV);
    for (row, energy) in energy_hartree.iter().copied().enumerate() {
        validate_finite_row("energy_hartree", energy, row + 1)?;
    }

    let density_states_per_hartree_atom = data.density.mapv(|density| density * FEFF_HARTREE_EV);
    for (index, density) in density_states_per_hartree_atom.iter().copied().enumerate() {
        let row = index / LDOS_DAT_NON_SPIN_DENSITY_COLUMNS + 1;
        validate_finite_row("density_states_per_hartree_atom", density, row)?;
    }

    Ok(FullSpectrumLdosData {
        fermi_level_hartree,
        energy_hartree,
        density_states_per_hartree_atom,
    })
}

/// Write FEFF `ldosNN.dat` text to a file.
pub fn write_ldos_dat(path: impl AsRef<Path>, data: &LdosDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, ldos_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `rhocNN.dat` text to a file.
pub fn write_rhoc_dat(path: impl AsRef<Path>, data: &RhocDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhoc_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF Hubbard `lmdosNN.dat` text to a file.
pub fn write_lmdos_dat(path: impl AsRef<Path>, data: &LmdosDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, lmdos_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF Hubbard `rhocmNN.dat` text to a file.
pub fn write_rhocm_dat(path: impl AsRef<Path>, data: &RhocmDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhocm_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `ldosNN.dat` text from a file.
pub fn read_ldos_dat(path: impl AsRef<Path>) -> Result<LdosDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_ldos_dat(&text)
}

/// Read FEFF `rhocNN.dat` embedded-density text from a file.
pub fn read_rhoc_dat(path: impl AsRef<Path>) -> Result<RhocDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rhoc_dat(&text)
}

/// Read FEFF Hubbard `lmdosNN.dat` text from a file.
pub fn read_lmdos_dat(path: impl AsRef<Path>) -> Result<LmdosDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_lmdos_dat(&text)
}

/// Read FEFF Hubbard `rhocmNN.dat` text from a file.
pub fn read_rhocm_dat(path: impl AsRef<Path>) -> Result<RhocmDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rhocm_dat(&text)
}

fn default_ldos_header_lines(
    fermi_level_ev: Option<f64>,
    charge_transfer: Option<f64>,
    electron_counts: &[LdosElectronCount],
    atom_count: Option<usize>,
    lorentzian_hwhh_ev: Option<f64>,
) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(value) = fermi_level_ev {
        lines.push(format!("#  Fermi level (eV): {value:7.3}"));
    }
    if let Some(value) = charge_transfer {
        lines.push(format!("#  Charge transfer : {value:7.3}"));
    }
    if !electron_counts.is_empty() {
        lines.push("#    Electron counts for each orbital momentum:".to_string());
        for count in electron_counts {
            lines.push(format!(
                "#       {}   {:8.3}",
                count.angular_momentum, count.count
            ));
        }
    }
    if let Some(value) = atom_count {
        lines.push(format!("#  Number of atoms in cluster: {value:3}"));
    }
    if let Some(value) = lorentzian_hwhh_ev {
        lines.push(format!(
            "#  Lorentzian broadening with HWHH {value:10.4} eV"
        ));
    }
    lines.push(
        "# -----------------------------------------------------------------------".to_string(),
    );
    lines.push("#      e        sDOS           pDOS          dDOS          fDOS    @#".to_string());
    lines
}

fn default_spin_ldos_header_lines(
    fermi_level_ev: Option<f64>,
    charge_transfer: Option<f64>,
    electron_counts: &[LdosElectronCount],
    atom_count: Option<usize>,
    lorentzian_hwhh_ev: Option<f64>,
) -> Vec<String> {
    let mut lines = default_ldos_header_lines(
        fermi_level_ev,
        charge_transfer,
        electron_counts,
        atom_count,
        lorentzian_hwhh_ev,
    );
    if let Some(label) = lines.last_mut() {
        *label = concat!(
            "#      e        sDOS(up)   pDOS(up)      dDOS(up)    fDOS(up)",
            "   sDOS(down)    pDOS(down)   dDOS(down)   fDOS(down)    @#"
        )
        .to_string();
    }
    lines
}

fn default_magnetic_ldos_header_lines(
    fermi_level_ev: Option<f64>,
    charge_transfer: Option<f64>,
    electron_counts: &[LdosElectronCount],
    atom_count: Option<usize>,
    lorentzian_hwhh_ev: Option<f64>,
    angular_limit: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(value) = fermi_level_ev {
        lines.push(format!("#  Fermi level (eV): {value:7.3}"));
    }
    if let Some(value) = charge_transfer {
        lines.push(format!("#  Charge transfer : {value:7.3}"));
    }
    if !electron_counts.is_empty() {
        lines.push("#  Electron counts for each magnetic-orbital momentum:".to_string());
        for angular in 0..=angular_limit {
            for magnetic in angular_magnetic_range(angular) {
                let count = electron_counts
                    .iter()
                    .find(|count| count.angular_momentum == angular)
                    .map(|count| count.count)
                    .unwrap_or(0.0);
                let orbital_1based = magnetic + 1;
                lines.push(format!(
                    "#       {angular}   {orbital_1based:3}   {count:8.3}"
                ));
            }
        }
    }
    if let Some(value) = atom_count {
        lines.push(format!("#  Number of atoms in cluster: {value:3}"));
    }
    if let Some(value) = lorentzian_hwhh_ev {
        lines.push(format!(
            "#  Lorentzian broadening with HWHH {value:10.4} eV"
        ));
    }
    lines.push(
        "# -----------------------------------------------------------------------".to_string(),
    );
    lines.push(magnetic_ldos_column_header(angular_limit));
    lines
}

fn magnetic_ldos_column_header(angular_limit: usize) -> String {
    let mut line = "#      e".to_string();
    for spin_label in ["up", "dn"] {
        for angular in 0..=angular_limit {
            let magnetic_min = -(angular as isize);
            for (offset, _) in angular_magnetic_range(angular).enumerate() {
                let magnetic_quantum = magnetic_min + offset as isize;
                let label = match angular {
                    0 => "s",
                    1 => "p",
                    2 => "d",
                    3 => "f",
                    _ => "l",
                };
                line.push_str(&format!("   {label}({magnetic_quantum:+})DOS-{spin_label}"));
            }
        }
    }
    line.push_str("   @#");
    line
}

fn angular_magnetic_range(angular: usize) -> std::ops::Range<usize> {
    (angular * angular)..((angular + 1) * (angular + 1))
}

fn parse_header_metadata(
    line: &str,
    line_number: usize,
    fermi_level_ev: &mut Option<f64>,
    charge_transfer: &mut Option<f64>,
    electron_counts: &mut Vec<LdosElectronCount>,
    atom_count: &mut Option<usize>,
    lorentzian_hwhh_ev: &mut Option<f64>,
) -> Result<()> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("fermi level") {
        *fermi_level_ev = Some(parse_f64(
            line_number,
            "fermi level",
            last_numeric_token(line)
                .ok_or_else(|| invalid_ldos_dat("fermi level", "missing numeric header value"))?,
        )?);
    } else if lower.contains("charge transfer") {
        *charge_transfer = Some(parse_f64(
            line_number,
            "charge transfer",
            last_numeric_token(line).ok_or_else(|| {
                invalid_ldos_dat("charge transfer", "missing numeric header value")
            })?,
        )?);
    } else if lower.contains("number of atoms in cluster") {
        *atom_count = Some(parse_usize(
            line_number,
            "number of atoms",
            last_numeric_token(line).ok_or_else(|| {
                invalid_ldos_dat("number of atoms", "missing numeric header value")
            })?,
        )?);
    } else if lower.contains("lorentzian broadening") {
        *lorentzian_hwhh_ev = Some(parse_f64(
            line_number,
            "lorentzian broadening",
            last_numeric_token(line).ok_or_else(|| {
                invalid_ldos_dat("lorentzian broadening", "missing numeric header value")
            })?,
        )?);
    } else if let Some(count) = parse_electron_count_header(line, line_number)? {
        electron_counts.push(count);
    }
    Ok(())
}

fn parse_electron_count_header(
    line: &str,
    line_number: usize,
) -> Result<Option<LdosElectronCount>> {
    let stripped = line.trim_start().strip_prefix('#').map(str::trim);
    let Some(stripped) = stripped else {
        return Ok(None);
    };
    let tokens = stripped.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 2 || !is_usize_token(tokens[0]) || !is_numeric_token(tokens[1]) {
        return Ok(None);
    }
    Ok(Some(LdosElectronCount {
        angular_momentum: parse_usize(line_number, "electron count angular momentum", tokens[0])?,
        count: parse_f64(line_number, "electron count", tokens[1])?,
    }))
}

fn validate_ldos_dat(data: &LdosDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_ldos_dat(
            "rows",
            "at least one LDOS row is required",
        ));
    }
    let (rows, cols) = data.density.dim();
    if rows != point_count {
        return Err(IoError::LdosDatShape {
            field: "density",
            rows,
            cols,
            expected_rows: point_count,
            expected_cols: LDOS_DAT_ALLOWED_DENSITY_COLUMNS,
        });
    }
    if !matches!(
        cols,
        LDOS_DAT_TRUNCATED_NON_SPIN_DENSITY_COLUMNS
            | LDOS_DAT_NON_SPIN_DENSITY_COLUMNS
            | LDOS_DAT_TRUNCATED_SPIN_DENSITY_COLUMNS
            | LDOS_DAT_SPIN_DENSITY_COLUMNS
    ) {
        return Err(IoError::LdosDatShape {
            field: "density",
            rows,
            cols,
            expected_rows: point_count,
            expected_cols: LDOS_DAT_ALLOWED_DENSITY_COLUMNS,
        });
    }

    if let Some(value) = data.fermi_level_ev {
        validate_finite("fermi level", value)?;
    }
    if let Some(value) = data.charge_transfer {
        validate_finite("charge transfer", value)?;
    }
    if let Some(value) = data.lorentzian_hwhh_ev {
        validate_finite("lorentzian broadening", value)?;
    }
    for count in &data.electron_counts {
        validate_finite("electron count", count.count)?;
    }

    for (row, energy) in data.energy_ev.iter().enumerate() {
        validate_finite_row("energy", *energy, row + 1)?;
    }
    for (index, value) in data.density.iter().enumerate() {
        let row = index / cols + 1;
        validate_finite_row("density", *value, row)?;
    }
    Ok(())
}

fn validate_ldos_magnetic_dat(data: &LdosMagneticDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_ldos_dat(
            "rows",
            "at least one magnetic LDOS row is required",
        ));
    }
    let (rows, cols) = data.density.dim();
    if rows != point_count {
        return Err(IoError::LdosDatShape {
            field: "density",
            rows,
            cols,
            expected_rows: point_count,
            expected_cols: LDOS_MAGNETIC_ALLOWED_DENSITY_COLUMNS,
        });
    }
    let expected_cols = ldos_magnetic_density_columns(data.angular_limit)?;
    if cols != expected_cols {
        return Err(IoError::LdosDatShape {
            field: "density",
            rows,
            cols,
            expected_rows: point_count,
            expected_cols: LDOS_MAGNETIC_ALLOWED_DENSITY_COLUMNS,
        });
    }

    if let Some(value) = data.fermi_level_ev {
        validate_finite("fermi level", value)?;
    }
    if let Some(value) = data.charge_transfer {
        validate_finite("charge transfer", value)?;
    }
    if let Some(value) = data.lorentzian_hwhh_ev {
        validate_finite("lorentzian broadening", value)?;
    }
    for count in &data.electron_counts {
        validate_finite("electron count", count.count)?;
    }

    for (row, energy) in data.energy_ev.iter().enumerate() {
        validate_finite_row("energy", *energy, row + 1)?;
    }
    for (index, value) in data.density.iter().enumerate() {
        let row = index / cols + 1;
        validate_finite_row("density", *value, row)?;
    }
    Ok(())
}

fn magnetic_angular_limit_from_density_columns(columns: usize) -> Result<usize> {
    if columns == 0 || !columns.is_multiple_of(2) {
        return Err(invalid_ldos_dat(
            "density",
            "magnetic LDOS density columns must be 2 * (lx + 1)^2",
        ));
    }
    let per_spin = columns / 2;
    let mut angular_count = 1_usize;
    loop {
        let square = angular_count
            .checked_mul(angular_count)
            .ok_or_else(|| invalid_ldos_dat("density", "integer overflow"))?;
        if square == per_spin {
            return angular_count
                .checked_sub(1)
                .ok_or_else(|| invalid_ldos_dat("density", "angular count underflow"));
        }
        if square > per_spin {
            return Err(invalid_ldos_dat(
                "density",
                "magnetic LDOS density columns must be 2 * (lx + 1)^2",
            ));
        }
        angular_count = angular_count
            .checked_add(1)
            .ok_or_else(|| invalid_ldos_dat("density", "integer overflow"))?;
    }
}

fn ldos_magnetic_density_columns(angular_limit: usize) -> Result<usize> {
    let angular_count = angular_limit
        .checked_add(1)
        .ok_or_else(|| invalid_ldos_dat("density", "integer overflow"))?;
    angular_count
        .checked_mul(angular_count)
        .and_then(|per_spin| per_spin.checked_mul(2))
        .ok_or_else(|| invalid_ldos_dat("density", "integer overflow"))
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::LdosDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::LdosDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_ldos_dat(field, "value must be finite"))
    }
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidLdosDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_ldos_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidLdosDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

fn is_usize_token(token: &str) -> bool {
    token.parse::<usize>().is_ok()
}

fn last_numeric_token(line: &str) -> Option<&str> {
    line.split_whitespace()
        .rev()
        .find(|token| is_numeric_token(token))
}

fn row_width_label(width: usize) -> &'static str {
    match width {
        LDOS_DAT_TRUNCATED_NON_SPIN_ROW_WIDTH => "4",
        LDOS_DAT_NON_SPIN_ROW_WIDTH => "5",
        LDOS_DAT_TRUNCATED_SPIN_ROW_WIDTH => "7",
        LDOS_DAT_SPIN_ROW_WIDTH => "9",
        _ => LDOS_DAT_ALLOWED_ROW_WIDTHS,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::{Array3, Array4};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn parses_feff_ldos_reference_shape_and_metadata() -> Result<()> {
        let data = parse_ldos_dat(LDOS_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert!(!data.is_spin_resolved());
        assert_eq!(data.fermi_level_ev, Some(-14.683));
        assert_eq!(data.charge_transfer, Some(0.711));
        assert_eq!(data.atom_count, Some(0));
        assert_eq!(data.lorentzian_hwhh_ev, Some(0.0100));
        assert_eq!(data.electron_counts.len(), 4);
        assert_eq!(data.electron_counts[2].angular_momentum, 2);
        assert_eq!(data.electron_counts[2].count, 10.223);
        assert_eq!(data.energy_ev[0], -30.0);
        assert_eq!(data.density[[0, 0]], 1.342776e-4);
        assert_eq!(data.density[[1, 3]], 2.564170e-5);
        Ok(())
    }

    #[test]
    fn parses_spin_resolved_ldos_shape() -> Result<()> {
        let data = parse_ldos_dat(SPIN_LDOS_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(data.is_spin_resolved());
        assert_eq!(data.density.ncols(), LDOS_SPIN_ORBITAL_LABELS.len());
        assert_eq!(data.density[[0, 4]], 5.0e-3);
        assert_eq!(data.density[[1, 7]], 1.6e-2);
        Ok(())
    }

    #[test]
    fn parses_truncated_spin_resolved_ldos_shape() -> Result<()> {
        let data = parse_ldos_dat(
            r#"#      e        sDOS(up)   pDOS(up)      dDOS(up)   sDOS(down)    pDOS(down)   dDOS(down)    @#
   -25.0000  5.966260E-05  1.925367E-03  1.296122E-05  5.938916E-05  1.749502E-03  1.299633E-05
   -24.8141  5.829510E-05  2.229671E-03  1.312382E-05  5.814359E-05  2.013812E-03  1.316047E-05
"#,
        )?;

        assert_eq!(data.point_count(), 2);
        assert!(data.is_spin_resolved());
        assert_eq!(
            data.density.ncols(),
            LDOS_DAT_TRUNCATED_SPIN_DENSITY_COLUMNS
        );
        assert_eq!(data.density[[0, 3]], 5.938916e-5);
        assert_eq!(parse_ldos_dat(&ldos_dat_string(&data)?)?, data);
        Ok(())
    }

    #[test]
    fn parses_truncated_non_spin_ldos_shape() -> Result<()> {
        let data = parse_ldos_dat(
            r#"#      e        sDOS           pDOS          dDOS          fDOS    @#
   -20.0000  1.144897E-01  9.892354E-02  1.168724E-02
   -19.7000  1.304320E-01  1.115281E-01  1.427987E-02
"#,
        )?;

        assert_eq!(data.point_count(), 2);
        assert!(!data.is_spin_resolved());
        assert_eq!(
            data.density.ncols(),
            LDOS_DAT_TRUNCATED_NON_SPIN_DENSITY_COLUMNS
        );
        assert_eq!(data.density[[0, 2]], 1.168724e-2);
        assert_eq!(parse_ldos_dat(&ldos_dat_string(&data)?)?, data);
        Ok(())
    }

    #[test]
    fn roundtrips_ldos_text() -> Result<()> {
        let data = parse_ldos_dat(LDOS_DAT)?;
        let rendered = ldos_dat_string(&data)?;
        assert_eq!(parse_ldos_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn parses_and_roundtrips_rhoc_text() -> Result<()> {
        let data = parse_rhoc_dat(RHOC_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(!data.is_spin_resolved());
        assert_eq!(data.header_lines.len(), 0);
        assert_eq!(data.energy_ev[0], -10.0);
        assert_eq!(data.density[[1, 3]], 8.0);
        let rendered = rhoc_dat_string(&data)?;
        assert_eq!(parse_rhoc_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn parses_and_roundtrips_magnetic_ldos_text() -> Result<()> {
        let data = parse_lmdos_dat(MAGNETIC_LDOS_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.angular_limit, 1);
        assert_eq!(data.magnetic_columns_per_spin(), 4);
        assert_eq!(data.density_column_count(), 8);
        assert_eq!(data.fermi_level_ev, Some(-1.0));
        assert_eq!(data.energy_ev[0], -10.0);
        assert_eq!(data.density[[0, 0]], 1.0);
        assert_eq!(data.density[[0, 4]], 5.0);
        assert_eq!(data.density[[1, 7]], 16.0);

        let rendered = lmdos_dat_string(&data)?;
        let reparsed = parse_lmdos_dat(&rendered)?;
        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn parses_hubbard_nio_magnetic_ldos_reference_zip() -> Result<()> {
        let Some(zip_path) = workspace_reference_zip("HUBBARD/NiO") else {
            eprintln!("skipping Hubbard magnetic LDOS reference test; NiO REFERENCE.zip not found");
            return Ok(());
        };

        let lmdos_text = unzip_reference_text(&zip_path, "REFERENCE/lmdos00.dat")?;
        let lmdos = parse_lmdos_dat(&lmdos_text)?;
        assert_eq!(lmdos.point_count(), 200);
        assert_eq!(lmdos.angular_limit, 2);
        assert_eq!(lmdos.magnetic_columns_per_spin(), 9);
        assert_eq!(lmdos.density_column_count(), 18);
        assert_eq!(lmdos.fermi_level_ev, Some(-12.717));
        assert_eq!(parse_lmdos_dat(&lmdos_dat_string(&lmdos)?)?, lmdos);

        let rhocm_text = unzip_reference_text(&zip_path, "REFERENCE/rhocm00.dat")?;
        let rhocm = parse_rhocm_dat(&rhocm_text)?;
        assert_eq!(rhocm.point_count(), 200);
        assert_eq!(rhocm.angular_limit, 2);
        assert_eq!(rhocm.magnetic_columns_per_spin(), 9);
        assert_eq!(rhocm.density_column_count(), 18);
        assert_eq!(rhocm.header_lines.len(), 0);
        assert_eq!(parse_rhocm_dat(&rhocm_dat_string(&rhocm)?)?, rhocm);
        Ok(())
    }

    #[test]
    fn converts_ldos_to_feff_fullspectrum_rdldos_units() -> Result<()> {
        let data = parse_ldos_dat(LDOS_DAT)?;
        let fullspectrum = fullspectrum_ldos_from_ldos_dat(&data)?;

        assert_eq!(fullspectrum.point_count(), 3);
        assert_eq!(fullspectrum.angular_count(), 4);
        assert_close(fullspectrum.fermi_level_hartree, -14.683 / FEFF_HARTREE_EV);
        assert_close(fullspectrum.energy_hartree[0], -30.0 / FEFF_HARTREE_EV);
        assert_close(
            fullspectrum.density_states_per_hartree_atom[[0, 0]],
            1.342_776E-04 * FEFF_HARTREE_EV,
        );
        assert_close(
            fullspectrum.density_states_per_hartree_atom[[1, 3]],
            2.564_170E-05 * FEFF_HARTREE_EV,
        );
        Ok(())
    }

    #[test]
    fn builds_ldos_and_rhoc_from_ff2rho_tables() -> Result<()> {
        let energy = Array1::from_vec(vec![Complex64::new(0.5, 0.01), Complex64::new(0.75, 0.01)]);
        let embedded =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let scattering = Array2::from_shape_vec(
            (4, 2),
            vec![
                Complex64::new(1.5, -0.4),
                Complex64::new(0.5, 0.2),
                Complex64::new(-0.3, 0.7),
                Complex64::new(1.1, -0.6),
                Complex64::new(0.8, 0.9),
                Complex64::new(-1.2, 0.4),
                Complex64::new(2.0, -1.0),
                Complex64::new(-0.5, 0.3),
            ],
        )
        .unwrap();
        let trace = Array2::from_shape_vec(
            (4, 2),
            vec![
                Complex64::new(0.2, 0.3),
                Complex64::new(-0.1, 0.4),
                Complex64::new(0.5, -0.2),
                Complex64::new(0.3, 0.1),
                Complex64::new(-0.4, 0.6),
                Complex64::new(0.7, -0.5),
                Complex64::new(0.9, 0.2),
                Complex64::new(-0.8, 0.25),
            ],
        )
        .unwrap();
        let electron_counts = vec![
            LdosElectronCount {
                angular_momentum: 0,
                count: 1.25,
            },
            LdosElectronCount {
                angular_momentum: 1,
                count: 2.50,
            },
        ];

        let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: Some(0.2),
            charge_transfer: Some(0.125),
            electron_counts: &electron_counts,
            atom_count: Some(3),
            lorentzian_hwhh_hartree: Some(0.01),
            energy_grid_hartree: energy.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: scattering.view(),
            scattering_trace: trace.view(),
            angular_count: 4,
            apply_scattering: true,
        })?;

        assert_eq!(handoff.ldos.point_count(), 2);
        assert!(!handoff.ldos.is_spin_resolved());
        assert_eq!(handoff.ldos.electron_counts, electron_counts);
        assert_close(handoff.ldos.fermi_level_ev.unwrap(), 0.2 * FEFF_HARTREE_EV);
        assert_close(
            handoff.ldos.lorentzian_hwhh_ev.unwrap(),
            0.01 * FEFF_HARTREE_EV,
        );
        assert_close(handoff.ldos.energy_ev[0], 0.5 * FEFF_HARTREE_EV);
        assert_close(handoff.ldos.density[(0, 0)], 1.37);
        assert_close(handoff.ldos.density[(1, 0)], 2.18);
        assert_close(handoff.ldos.density[(0, 3)], 6.5);
        assert!(handoff.ldos.header_lines[0].contains("Fermi level"));

        assert_eq!(handoff.rhoc.header_lines, Vec::<String>::new());
        assert_eq!(handoff.rhoc.fermi_level_ev, None);
        assert_close(handoff.rhoc.density[(0, 0)], 1.0);
        assert_close(handoff.rhoc.density[(1, 3)], 8.0);

        let rendered_ldos = ldos_dat_string(&handoff.ldos)?;
        let parsed_ldos = parse_ldos_dat(&rendered_ldos)?;
        assert_close_tol(
            parsed_ldos.density[(0, 0)],
            handoff.ldos.density[(0, 0)],
            1.0e-6,
        );

        let rendered_rhoc = rhoc_dat_string(&handoff.rhoc)?;
        let parsed_rhoc = parse_rhoc_dat(&rendered_rhoc)?;
        assert_close_tol(
            parsed_rhoc.density[(1, 3)],
            handoff.rhoc.density[(1, 3)],
            1.0e-6,
        );
        Ok(())
    }

    #[test]
    fn builds_spin_ldos_and_rhoc_from_ff2rho_h_tables() -> Result<()> {
        let energy = Array1::from_vec(vec![Complex64::new(0.5, 0.01)]);
        let mut embedded = Array3::<f64>::zeros((4, 2, 1));
        for angular in 0..4 {
            embedded[(angular, 0, 0)] = 1.0 + angular as f64;
            embedded[(angular, 1, 0)] = 10.0 + angular as f64;
        }
        let mut scattering = Array3::<Complex64>::zeros((4, 2, 1));
        let mut trace = Array3::<Complex64>::zeros((4, 2, 1));
        scattering[(0, 0, 0)] = Complex64::new(1.5, -0.4);
        trace[(0, 0, 0)] = Complex64::new(0.2, 0.3);
        scattering[(0, 1, 0)] = Complex64::new(-0.2, 0.6);
        trace[(0, 1, 0)] = Complex64::new(0.5, -0.1);

        let handoff = ldos_spin_dat_from_ff2rho(LdosSpinDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: Some(0.2),
            charge_transfer: Some(0.125),
            electron_counts: &[],
            atom_count: Some(3),
            lorentzian_hwhh_hartree: Some(0.01),
            energy_grid_hartree: energy.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: scattering.view(),
            scattering_trace: trace.view(),
            apply_scattering: true,
        })?;

        assert!(handoff.ldos.is_spin_resolved());
        assert!(handoff.rhoc.is_spin_resolved());
        assert_close(handoff.ldos.energy_ev[0], 0.5 * FEFF_HARTREE_EV);
        assert_close(handoff.ldos.density[(0, 0)], 1.37);
        assert_close(handoff.ldos.density[(0, 4)], 10.32);
        assert_close(handoff.rhoc.density[(0, 0)], 1.0);
        assert_close(handoff.rhoc.density[(0, 4)], 10.0);
        let label = handoff.ldos.header_lines.last().unwrap();
        assert!(label.contains("sDOS(up)") && label.contains("sDOS(down)"));

        let parsed_ldos = parse_ldos_dat(&ldos_dat_string(&handoff.ldos)?)?;
        assert!(parsed_ldos.is_spin_resolved());
        assert_close_tol(parsed_ldos.density[(0, 4)], 10.32, 1.0e-6);

        let parsed_rhoc = parse_rhoc_dat(&rhoc_dat_string(&handoff.rhoc)?)?;
        assert!(parsed_rhoc.is_spin_resolved());
        assert_close_tol(parsed_rhoc.density[(0, 4)], 10.0, 1.0e-6);
        Ok(())
    }

    #[test]
    fn builds_magnetic_lmdos_and_rhocm_from_ff2rho_h_step2_tables() -> Result<()> {
        let energy = Array1::from_vec(vec![Complex64::new(0.5, 0.01)]);
        let mut embedded = Array4::<f64>::zeros((2, 4, 2, 1));
        for angular in 0..2 {
            for magnetic in (angular * angular)..((angular + 1) * (angular + 1)) {
                for spin in 0..2 {
                    embedded[(angular, magnetic, spin, 0)] =
                        1.0 + angular as f64 + magnetic as f64 + 10.0 * spin as f64;
                }
            }
        }
        embedded[(0, 0, 0, 0)] = 2.0;
        embedded[(1, 2, 1, 0)] = 10.0;

        let mut scattering = Array4::<Complex64>::zeros((2, 4, 2, 1));
        let mut trace = Array4::<Complex64>::zeros((2, 4, 2, 1));
        scattering[(0, 0, 0, 0)] = Complex64::new(1.5, -0.4);
        trace[(0, 0, 0, 0)] = Complex64::new(0.2, 0.3);
        scattering[(1, 2, 1, 0)] = Complex64::new(1.0, -0.2);
        trace[(1, 2, 1, 0)] = Complex64::new(0.5, 0.25);
        let electron_counts = vec![
            LdosElectronCount {
                angular_momentum: 0,
                count: 1.25,
            },
            LdosElectronCount {
                angular_momentum: 1,
                count: 2.50,
            },
        ];

        let handoff = ldos_magnetic_dat_from_ff2rho(LdosMagneticDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: Some(0.2),
            charge_transfer: Some(0.125),
            electron_counts: &electron_counts,
            atom_count: Some(3),
            lorentzian_hwhh_hartree: Some(0.01),
            energy_grid_hartree: energy.view(),
            embedded_magnetic_ldos: embedded.view(),
            scattering_magnetic_ldos: scattering.view(),
            magnetic_scattering_trace: trace.view(),
            angular_count: 2,
        })?;

        assert_eq!(handoff.lmdos.angular_limit, 1);
        assert_eq!(handoff.lmdos.density_column_count(), 8);
        assert_eq!(handoff.rhocm.density_column_count(), 8);
        assert_close(handoff.lmdos.fermi_level_ev.unwrap(), 0.2 * FEFF_HARTREE_EV);
        assert_close(
            handoff.lmdos.lorentzian_hwhh_ev.unwrap(),
            0.01 * FEFF_HARTREE_EV,
        );
        assert_close(handoff.rhocm.density[(0, 0)], 2.0);
        assert_close(handoff.lmdos.density[(0, 0)], 2.37);
        assert_close(handoff.rhocm.density[(0, 6)], 10.0);
        assert_close(handoff.lmdos.density[(0, 6)], 10.0 / 3.0 + 0.15);
        assert!(
            handoff
                .lmdos
                .header_lines
                .last()
                .is_some_and(|line| line.contains("p(+1)DOS-dn"))
        );

        let parsed_lmdos = parse_lmdos_dat(&lmdos_dat_string(&handoff.lmdos)?)?;
        assert_close_tol(parsed_lmdos.density[(0, 6)], 10.0 / 3.0 + 0.15, 1.0e-6);
        let parsed_rhocm = parse_rhocm_dat(&rhocm_dat_string(&handoff.rhocm)?)?;
        assert_close_tol(parsed_rhocm.density[(0, 6)], 10.0, 1.0e-6);
        Ok(())
    }

    #[test]
    fn ldos_dat_from_ff2rho_rejects_short_orbital_tables() {
        let energy = Array1::from_vec(vec![Complex64::new(0.5, 0.01)]);
        let embedded = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let scattering = Array2::<Complex64>::zeros((4, 1));

        let error = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: None,
            charge_transfer: None,
            electron_counts: &[],
            atom_count: None,
            lorentzian_hwhh_hartree: None,
            energy_grid_hartree: energy.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: scattering.view(),
            scattering_trace: scattering.view(),
            angular_count: 4,
            apply_scattering: true,
        })
        .unwrap_err();

        assert!(error.to_string().contains("embedded_ldos"));
    }

    #[test]
    fn ldos_dat_from_ff2rho_emits_runtime_lx_columns_with_static_header() -> Result<()> {
        let energy = Array1::from_vec(vec![Complex64::new(0.5, 0.01)]);
        let embedded = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let scattering = Array2::<Complex64>::zeros((3, 1));
        let counts = (0..3)
            .map(|angular_momentum| LdosElectronCount {
                angular_momentum,
                count: angular_momentum as f64,
            })
            .collect::<Vec<_>>();

        let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: None,
            charge_transfer: None,
            electron_counts: &counts,
            atom_count: None,
            lorentzian_hwhh_hartree: None,
            energy_grid_hartree: energy.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: scattering.view(),
            scattering_trace: scattering.view(),
            angular_count: 3,
            apply_scattering: false,
        })?;

        assert_eq!(handoff.ldos.density.ncols(), 3);
        assert_eq!(handoff.rhoc.density.ncols(), 3);
        assert_eq!(handoff.ldos.electron_counts, counts);
        assert!(
            handoff
                .ldos
                .header_lines
                .last()
                .is_some_and(|line| line.contains("fDOS"))
        );
        Ok(())
    }

    #[test]
    fn rejects_ldos_tables_not_supported_by_fullspectrum_rdldos() -> Result<()> {
        let spin = parse_ldos_dat(SPIN_LDOS_DAT)?;
        assert!(fullspectrum_ldos_from_ldos_dat(&spin).is_err());

        let truncated_non_spin = parse_ldos_dat("1 2 3 4\n2 3 4 5\n")?;
        assert!(fullspectrum_ldos_from_ldos_dat(&truncated_non_spin).is_err());

        let missing_fermi = parse_rhoc_dat(RHOC_DAT)?;
        assert!(fullspectrum_ldos_from_ldos_dat(&missing_fermi).is_err());

        let one_row = LdosDatData {
            header_lines: vec!["#  Fermi level (eV): -14.683".to_string()],
            fermi_level_ev: Some(-14.683),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: Array1::from_vec(vec![1.0]),
            density: Array2::zeros((1, LDOS_DAT_NON_SPIN_DENSITY_COLUMNS)),
        };
        assert!(fullspectrum_ldos_from_ldos_dat(&one_row).is_err());
        Ok(())
    }

    #[test]
    fn rejects_bad_ldos_inputs() {
        assert!(parse_ldos_dat("# no data\n").is_err());
        assert!(parse_ldos_dat("1 2 3 4 5 6\n").is_err());
        assert!(parse_ldos_dat("1 2 3 NaN 5\n").is_err());
        assert!(parse_ldos_dat("1 2 3 4 5\n2 3 4 5 6 7 8 9 10\n").is_err());

        let bad_shape = LdosDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            density: Array2::zeros((1, LDOS_DAT_NON_SPIN_DENSITY_COLUMNS)),
        };
        assert!(ldos_dat_string(&bad_shape).is_err());
    }

    #[test]
    fn rejects_bad_magnetic_ldos_inputs() {
        assert!(parse_lmdos_dat("# no data\n").is_err());
        assert!(parse_lmdos_dat("1 2 3 4\n").is_err());
        assert!(parse_lmdos_dat("1 2 3 4 5 6 7 8 9\n2 3 4\n").is_err());

        let bad_shape = LdosMagneticDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            angular_limit: 1,
            energy_ev: Array1::from_vec(vec![1.0]),
            density: Array2::zeros((1, 6)),
        };
        assert!(lmdos_dat_string(&bad_shape).is_err());
    }

    fn assert_close(actual: f64, expected: f64) {
        assert_close_tol(actual, expected, 1.0e-12);
    }

    fn assert_close_tol(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected} tolerance {tolerance}"
        );
    }

    fn workspace_reference_zip(relative: &str) -> Option<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.parent()?.parent()?;
        let zip_path = workspace
            .join("reference-work")
            .join("golden")
            .join(relative)
            .join("REFERENCE.zip");
        zip_path.is_file().then_some(zip_path)
    }

    fn unzip_reference_text(zip_path: &PathBuf, entry: &str) -> Result<String> {
        let output = std::process::Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .map_err(|source| IoError::io(zip_path, source))?;
        if !output.status.success() {
            return Err(invalid_ldos_dat(
                "reference",
                format!("failed to extract {entry} from {}", zip_path.display()),
            ));
        }
        String::from_utf8(output.stdout)
            .map_err(|_| invalid_ldos_dat("reference", "reference text is not UTF-8"))
    }

    const LDOS_DAT: &str = r#"#  Fermi level (eV): -14.683
#  Charge transfer :   0.711
#    Electron counts for each orbital momentum:
#       0      1.428
#       1      1.637
#       2     10.223
#       3      0.000
#  Number of atoms in cluster:   0
#  Lorentzian broadening with HWHH     0.0100 eV
# -----------------------------------------------------------------------
#      e        sDOS           pDOS          dDOS          fDOS    @#
   -30.0000  1.342776E-04  7.462376E-05  4.190590E-04  2.449744E-05
   -29.5500  1.777093E-04  8.534908E-05  3.865858E-04  2.564170E-05
   -29.1000  3.428764E-04  1.101810E-04  3.605636E-04  2.691077E-05
"#;

    const SPIN_LDOS_DAT: &str = r#"#      e        sDOS(up)   pDOS(up)      dDOS(up)    fDOS(up)   sDOS(down)    pDOS(down)   dDOS9(down)   fDOS(down)    @#
     1.0000  1.000000E-03  2.000000E-03  3.000000E-03  4.000000E-03  5.000000E-03  6.000000E-03  7.000000E-03  8.000000E-03
     2.0000  2.000000E-03  4.000000E-03  6.000000E-03  8.000000E-03  1.000000E-02  1.200000E-02  1.400000E-02  1.600000E-02
"#;

    const RHOC_DAT: &str = r#"    -10.0000  1.000000E+00  2.000000E+00  3.000000E+00  4.000000E+00
     -9.5000  5.000000E+00  6.000000E+00  7.000000E+00  8.000000E+00
"#;

    const MAGNETIC_LDOS_DAT: &str = r#"#  Fermi level (eV):  -1.000
#  Lorentzian broadening with HWHH     0.0100 eV
# -----------------------------------------------------------------------
#      e      s(0)DOS-up   p(-1)DOS-up   p(0)DOS-up   p(+1)DOS-up  s(0)DOS-dn   p(-1)DOS-dn   p(0)DOS-dn   p(+1)DOS-dn   @#
   -10.0000  1.000000E+00  2.000000E+00  3.000000E+00  4.000000E+00  5.000000E+00  6.000000E+00  7.000000E+00  8.000000E+00
    -9.5000  9.000000E+00  1.000000E+01  1.100000E+01  1.200000E+01  1.300000E+01  1.400000E+01  1.500000E+01  1.600000E+01
"#;
}
