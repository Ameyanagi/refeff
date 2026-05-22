use ndarray::{Array1, Array2};
use refeff_core::FEFF_ORBITAL_SLOT_COUNT;

use crate::error::Result;

use super::parse::parse_config_inp;

pub const CONFIG_RECORD_WIDTH: usize = 150;

/// Parsed FEFF `config.inp` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigInput {
    /// Configuration records in file order.
    pub records: Vec<ConfigRecord>,
}

impl ConfigInput {
    /// Parse FEFF `config.inp` text.
    pub fn parse_str(text: &str) -> Result<Self> {
        parse_config_inp(text)
    }

    /// Potential indices as an ndarray.
    #[must_use]
    pub fn potential_indices(&self) -> Array1<i32> {
        self.records
            .iter()
            .map(|record| record.potential_index)
            .collect()
    }
}

/// One configuration row.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigRecord {
    /// FEFF potential index, or a negative index to apply to all atoms of the element.
    pub potential_index: i32,
    /// Element symbol from the configuration row.
    pub element: String,
    /// Optional noble-gas base configuration.
    pub noble_gas: Option<String>,
    /// Explicit orbital occupations following the optional noble-gas token.
    pub states: Vec<ConfigState>,
}

impl ConfigRecord {
    /// Flatten explicit occupations in record order.
    ///
    /// This preserves the grouped input order. Use [`super::config_record_slot_rows`]
    /// when the caller needs FEFF's expanded 40-slot orbital arrays.
    #[must_use]
    pub fn occupations(&self) -> Array1<f64> {
        self.states
            .iter()
            .flat_map(|state| {
                state
                    .occupations
                    .iter()
                    .map(|occupation| occupation.occupation)
            })
            .collect()
    }
}

/// Expanded FEFF `config.inp` row arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigSlotRows {
    /// Occupation row in FEFF's 40 relativistic orbital slots.
    pub occupations: Array1<f64>,
    /// Valence occupation row in FEFF's 40 relativistic orbital slots.
    pub valence_occupations: Array1<f64>,
    /// Spin marker row in FEFF's 40 relativistic orbital slots.
    pub spin_occupations: Array1<f64>,
    /// Sum of absolute explicit occupations plus the supplied base-row occupation sum.
    pub electron_count: f64,
}

impl ConfigSlotRows {
    /// Zero-filled FEFF configuration slot rows.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            occupations: Array1::zeros(FEFF_ORBITAL_SLOT_COUNT),
            valence_occupations: Array1::zeros(FEFF_ORBITAL_SLOT_COUNT),
            spin_occupations: Array1::zeros(FEFF_ORBITAL_SLOT_COUNT),
            electron_count: 0.0,
        }
    }
}

/// Expanded FEFF `config.inp` rows for all potential indices.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigSlotTable {
    /// Occupation table indexed as `(potential, FEFF orbital slot)`.
    pub occupations: Array2<f64>,
    /// Valence occupation table indexed as `(potential, FEFF orbital slot)`.
    pub valence_occupations: Array2<f64>,
    /// Spin marker table indexed as `(potential, FEFF orbital slot)`.
    pub spin_occupations: Array2<f64>,
    /// Electron-count diagnostic for each potential row.
    pub electron_counts: Array1<f64>,
}

impl ConfigSlotTable {
    /// Zero-filled FEFF configuration slot table.
    #[must_use]
    pub fn zeros(potential_count: usize) -> Self {
        Self {
            occupations: Array2::zeros((potential_count, FEFF_ORBITAL_SLOT_COUNT)),
            valence_occupations: Array2::zeros((potential_count, FEFF_ORBITAL_SLOT_COUNT)),
            spin_occupations: Array2::zeros((potential_count, FEFF_ORBITAL_SLOT_COUNT)),
            electron_counts: Array1::zeros(potential_count),
        }
    }

    /// Number of potential rows represented by this table.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.electron_counts.len()
    }
}

/// One orbital configuration entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigState {
    /// FEFF orbital label such as `1s`, `2p`, or `4f`.
    pub orbital: String,
    /// Occupations for the orbital; one for s states, two otherwise.
    pub occupations: Vec<ConfigOccupation>,
}

/// One occupation value, with optional `s` spin marker value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfigOccupation {
    /// Signed occupation. Negative values mark core states in FEFF input.
    pub occupation: f64,
    /// Optional spin marker value following an `s` token.
    pub spin: Option<f64>,
}
