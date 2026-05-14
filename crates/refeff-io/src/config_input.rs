//! Typed reader/writer for FEFF `config.inp` electron-configuration files.
//!
//! `RDINP` creates `config.inp` from `CONFIG card` payload lines by writing
//! fixed-width 150-character records. The potential stage later parses those
//! records as `iph element [NobleGas] orbital occupation...` rows.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2};
use refeff_core::FEFF_ORBITAL_SLOT_COUNT;

use crate::error::{IoError, Result};

/// FEFF fixed record width used when `RDINP` writes `config.inp`.
pub const CONFIG_RECORD_WIDTH: usize = 150;

const HE_OCCUPATIONS: &[(usize, f64)] = &[(1, 2.0)];
const HE_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(1, 2.0)];
const HE_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(1, 1.0)];

const NE_OCCUPATIONS: &[(usize, f64)] = &[(1, 2.0), (2, 2.0), (3, 2.0), (4, 4.0)];
const NE_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(3, 2.0), (4, 4.0)];
const NE_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(4, 1.0)];

const AR_OCCUPATIONS: &[(usize, f64)] = &[
    (1, 2.0),
    (2, 2.0),
    (3, 2.0),
    (4, 4.0),
    (5, 2.0),
    (6, 2.0),
    (7, 4.0),
];
const AR_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(5, 2.0), (6, 2.0), (7, 4.0)];
const AR_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(7, 1.0)];

const KR_OCCUPATIONS: &[(usize, f64)] = &[
    (1, 2.0),
    (2, 2.0),
    (3, 2.0),
    (4, 4.0),
    (5, 2.0),
    (6, 2.0),
    (7, 4.0),
    (8, 4.0),
    (9, 6.0),
    (10, 2.0),
    (11, 2.0),
    (12, 4.0),
];
const KR_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(10, 2.0), (11, 2.0), (12, 4.0)];
const KR_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(12, 1.0)];

const XE_OCCUPATIONS: &[(usize, f64)] = &[
    (1, 2.0),
    (2, 2.0),
    (3, 2.0),
    (4, 4.0),
    (5, 2.0),
    (6, 2.0),
    (7, 4.0),
    (8, 4.0),
    (9, 6.0),
    (10, 2.0),
    (11, 2.0),
    (12, 4.0),
    (13, 4.0),
    (14, 6.0),
    (17, 2.0),
    (18, 2.0),
    (19, 4.0),
];
const XE_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(17, 2.0), (18, 2.0), (19, 4.0)];
const XE_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(19, 1.0)];

const HG_OCCUPATIONS: &[(usize, f64)] = &[
    (1, 2.0),
    (2, 2.0),
    (3, 2.0),
    (4, 4.0),
    (5, 2.0),
    (6, 2.0),
    (7, 4.0),
    (8, 4.0),
    (9, 6.0),
    (10, 2.0),
    (11, 2.0),
    (12, 4.0),
    (13, 4.0),
    (14, 6.0),
    (15, 6.0),
    (16, 8.0),
    (17, 2.0),
    (18, 2.0),
    (19, 4.0),
    (20, 4.0),
    (21, 6.0),
    (24, 2.0),
];
const HG_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(20, 4.0), (21, 6.0), (24, 2.0)];
const HG_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(24, 1.0)];

const RN_OCCUPATIONS: &[(usize, f64)] = &[
    (1, 2.0),
    (2, 2.0),
    (3, 2.0),
    (4, 4.0),
    (5, 2.0),
    (6, 2.0),
    (7, 4.0),
    (8, 4.0),
    (9, 6.0),
    (10, 2.0),
    (11, 2.0),
    (12, 4.0),
    (13, 4.0),
    (14, 6.0),
    (15, 6.0),
    (16, 8.0),
    (17, 2.0),
    (18, 2.0),
    (19, 4.0),
    (20, 4.0),
    (21, 6.0),
    (24, 2.0),
    (25, 2.0),
    (26, 4.0),
];
const RN_VALENCE_OCCUPATIONS: &[(usize, f64)] = &[(24, 2.0), (25, 2.0), (26, 4.0)];
const RN_SPIN_OCCUPATIONS: &[(usize, f64)] = &[(26, 1.0)];

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
    /// This preserves the grouped input order. Use [`config_record_slot_rows`]
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

/// Render typed FEFF `config.inp` text.
pub fn config_inp_string(input: &ConfigInput) -> Result<String> {
    validate_config_input(input)?;
    let lines = input
        .records
        .iter()
        .map(config_record_line)
        .collect::<Result<Vec<_>>>()?;
    config_inp_lines_string(&lines)
}

/// Render raw `CONFIG card` payload lines as FEFF `config.inp` records.
pub fn config_inp_lines_string(lines: &[String]) -> Result<String> {
    let mut out = String::new();
    for line in lines {
        write_config_record_line(&mut out, line)?;
    }
    Ok(out)
}

/// Parse FEFF `config.inp` text.
pub fn parse_config_inp(text: &str) -> Result<ConfigInput> {
    let records = config_lines(text)
        .map(|line| {
            let tokens = line.text.split_whitespace().collect::<Vec<_>>();
            parse_record(line.line, &tokens)
        })
        .collect::<Result<Vec<_>>>()?;
    let input = ConfigInput { records };
    validate_config_input(&input)?;
    Ok(input)
}

/// Expand a parsed configuration record into FEFF's 40 orbital slots.
///
/// `base_rows` can override FEFF's noble-gas initialization step. When it is
/// `None`, records with a recognized noble-gas token use FEFF10's built-in
/// default rows. Negative occupations update total occupation but leave the
/// valence row unchanged, matching `COMMON/m_config.f90::ParseConfig`.
pub fn config_record_slot_rows(
    record: &ConfigRecord,
    base_rows: Option<&ConfigSlotRows>,
) -> Result<ConfigSlotRows> {
    let mut rows = match (base_rows, record.noble_gas.as_deref()) {
        (Some(rows), _) => rows.clone(),
        (None, Some(noble_gas)) => config_noble_gas_slot_rows(noble_gas)?,
        (None, None) => ConfigSlotRows::zeros(),
    };
    validate_slot_rows(&rows)?;

    for state in &record.states {
        let first_slot = config_orbital_slot(&state.orbital)?;
        for (offset, occupation) in state.occupations.iter().enumerate() {
            let slot = first_slot + offset;
            if slot > FEFF_ORBITAL_SLOT_COUNT {
                return Err(invalid_config_inp(
                    "orbital",
                    format!("orbital {} extends past FEFF slot 40", state.orbital),
                ));
            }
            let index = slot - 1;
            let value = occupation.occupation.abs();
            rows.occupations[index] = value;
            if occupation.occupation >= 0.0 {
                rows.valence_occupations[index] = value;
            }
            if let Some(spin) = occupation.spin {
                rows.spin_occupations[index] = spin;
            }
            rows.electron_count += value;
        }
    }

    validate_slot_rows(&rows)?;
    Ok(rows)
}

/// Apply FEFF `config.inp` records to a potential-indexed 40-slot table.
///
/// `potential_elements` is indexed by FEFF potential index `iph = 0..=nph`.
/// Positive record indices replace one matching potential row. Negative record
/// indices are first expanded at `abs(iph)` and then copied to every potential
/// with the same element symbol, matching `COMMON/m_config.f90::ParseConfig`.
pub fn config_input_slot_table<S: AsRef<str>>(
    input: &ConfigInput,
    potential_elements: &[S],
) -> Result<ConfigSlotTable> {
    let potential_symbols = potential_elements
        .iter()
        .map(|symbol| parse_element(0, symbol.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    let mut table = ConfigSlotTable::zeros(potential_symbols.len());

    for record in &input.records {
        let target = record_target_index(record)?;
        let Some(target_symbol) = potential_symbols.get(target) else {
            return Err(invalid_config_inp(
                "potential index",
                format!(
                    "record references potential {target}, but only {} potential(s) are available",
                    potential_symbols.len()
                ),
            ));
        };
        if target_symbol != &record.element {
            return Err(invalid_config_inp(
                "element",
                format!(
                    "record element {} does not match potential {target} element {target_symbol}",
                    record.element
                ),
            ));
        }

        let rows = config_record_slot_rows(record, None)?;
        if record.potential_index < 0 {
            for (index, symbol) in potential_symbols.iter().enumerate() {
                if symbol == &record.element {
                    assign_table_row(&mut table, index, &rows);
                }
            }
        } else {
            assign_table_row(&mut table, target, &rows);
        }
    }

    Ok(table)
}

/// Return FEFF10's default 40-slot rows for a noble-gas shorthand token.
///
/// FEFF accepts `Hg` as a filled-shell shorthand in addition to the noble gases.
/// The returned rows match `COMMON/m_config.f90`'s `iocc9`, `ival9`, and
/// `ispn9` defaults used when `configtype` is not the legacy FEFF7 mode.
pub fn config_noble_gas_slot_rows(noble_gas: &str) -> Result<ConfigSlotRows> {
    let canonical = canonical_noble_gas(noble_gas)
        .ok_or_else(|| invalid_config_inp("noble gas", "unknown noble-gas token"))?;
    let base = noble_gas_base(&canonical)
        .ok_or_else(|| invalid_config_inp("noble gas", "unsupported noble-gas base"))?;
    slot_rows_from_pairs(
        base.electron_count,
        base.occupations,
        base.valence_occupations,
        base.spin_occupations,
    )
}

/// Return the one-based FEFF slot index for a grouped orbital label.
pub fn config_orbital_slot(orbital: &str) -> Result<usize> {
    match orbital.to_ascii_lowercase().as_str() {
        "1s" => Ok(1),
        "2s" => Ok(2),
        "2p" => Ok(3),
        "3s" => Ok(5),
        "3p" => Ok(6),
        "3d" => Ok(8),
        "4s" => Ok(10),
        "4p" => Ok(11),
        "4d" => Ok(13),
        "4f" => Ok(15),
        "5s" => Ok(17),
        "5p" => Ok(18),
        "5d" => Ok(20),
        "5f" => Ok(22),
        "6s" => Ok(24),
        "6p" => Ok(25),
        "6d" => Ok(27),
        "7s" => Ok(29),
        "7p" => Ok(30),
        "8s" => Ok(32),
        "8p" => Ok(33),
        "7d" => Ok(35),
        "6f" => Ok(37),
        "5g" => Ok(39),
        _ => Err(invalid_config_inp("orbital", "unknown FEFF orbital label")),
    }
}

/// Write FEFF `config.inp` text to a file.
pub fn write_config_inp(path: impl AsRef<Path>, input: &ConfigInput) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, config_inp_string(input)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `config.inp` text from a file.
pub fn read_config_inp(path: impl AsRef<Path>) -> Result<ConfigInput> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_config_inp(&text)
}

fn parse_record(line: usize, tokens: &[&str]) -> Result<ConfigRecord> {
    if tokens.len() < 3 {
        return Err(IoError::ConfigInpRowWidth {
            line,
            actual: tokens.len(),
            expected: 3,
        });
    }

    let potential_index = parse_i32(line, "potential index", tokens[0])?;
    let element = parse_element(line, tokens[1])?;
    let (noble_gas, mut index) = if is_zero_base_token(tokens[2]) {
        (None, 3)
    } else if let Some(noble_gas) = canonical_noble_gas(tokens[2]) {
        (Some(noble_gas), 3)
    } else {
        (None, 2)
    };
    let mut states = Vec::new();

    while index < tokens.len() {
        let orbital = parse_orbital(line, tokens[index])?;
        index += 1;
        let occupation_count = if is_s_orbital(&orbital) { 1 } else { 2 };
        let mut occupations = Vec::with_capacity(occupation_count);

        for _ in 0..occupation_count {
            let Some(token) = tokens.get(index) else {
                return Err(IoError::ConfigInpMissing {
                    field: "occupation",
                    line,
                });
            };
            let occupation = parse_f64(line, "occupation", token)?;
            index += 1;
            let spin = if tokens
                .get(index)
                .is_some_and(|token| token.eq_ignore_ascii_case("s"))
            {
                index += 1;
                let Some(token) = tokens.get(index) else {
                    return Err(IoError::ConfigInpMissing {
                        field: "spin",
                        line,
                    });
                };
                let spin = parse_f64(line, "spin", token)?;
                index += 1;
                Some(spin)
            } else {
                None
            };
            occupations.push(ConfigOccupation { occupation, spin });
        }

        states.push(ConfigState {
            orbital,
            occupations,
        });
    }

    Ok(ConfigRecord {
        potential_index,
        element,
        noble_gas,
        states,
    })
}

fn config_record_line(record: &ConfigRecord) -> Result<String> {
    let mut line = format!("{} {}", record.potential_index, record.element);
    if let Some(noble_gas) = &record.noble_gas {
        write!(line, " {noble_gas}")?;
    }
    for state in &record.states {
        write!(line, " {}", state.orbital)?;
        for occupation in &state.occupations {
            write!(line, " {}", occupation.occupation)?;
            if let Some(spin) = occupation.spin {
                write!(line, " s {spin}")?;
            }
        }
    }
    Ok(line)
}

fn write_config_record_line(out: &mut String, line: &str) -> Result<()> {
    if line.len() > CONFIG_RECORD_WIDTH {
        return Err(IoError::InvalidConfigInp {
            field: "record",
            message: format!(
                "record length {} exceeds FEFF width {CONFIG_RECORD_WIDTH}",
                line.len()
            ),
        });
    }
    writeln!(out, "{line:<width$}", width = CONFIG_RECORD_WIDTH)?;
    Ok(())
}

fn validate_config_input(input: &ConfigInput) -> Result<()> {
    for record in &input.records {
        parse_element(0, &record.element)?;
        if let Some(noble_gas) = &record.noble_gas
            && canonical_noble_gas(noble_gas).is_none()
        {
            return Err(invalid_config_inp("noble gas", "unknown noble-gas token"));
        }
        for state in &record.states {
            let orbital = parse_orbital(0, &state.orbital)?;
            let expected = if is_s_orbital(&orbital) { 1 } else { 2 };
            if state.occupations.len() != expected {
                return Err(invalid_config_inp(
                    "occupation",
                    format!(
                        "orbital {orbital} requires {expected} occupation value(s), got {}",
                        state.occupations.len()
                    ),
                ));
            }
            for occupation in &state.occupations {
                validate_finite("occupation", occupation.occupation)?;
                if let Some(spin) = occupation.spin {
                    validate_finite("spin", spin)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_slot_rows(rows: &ConfigSlotRows) -> Result<()> {
    validate_slot_row_len("occupations", rows.occupations.len())?;
    validate_slot_row_len("valence_occupations", rows.valence_occupations.len())?;
    validate_slot_row_len("spin_occupations", rows.spin_occupations.len())?;
    validate_finite("electron count", rows.electron_count)?;
    for value in rows
        .occupations
        .iter()
        .chain(rows.valence_occupations.iter())
        .chain(rows.spin_occupations.iter())
    {
        validate_finite("slot row", *value)?;
    }
    Ok(())
}

fn validate_slot_row_len(name: &'static str, len: usize) -> Result<()> {
    if len == FEFF_ORBITAL_SLOT_COUNT {
        Ok(())
    } else {
        Err(invalid_config_inp(
            name,
            format!("length {len} does not match FEFF slot count {FEFF_ORBITAL_SLOT_COUNT}"),
        ))
    }
}

fn record_target_index(record: &ConfigRecord) -> Result<usize> {
    usize::try_from(i64::from(record.potential_index).abs()).map_err(|_| {
        invalid_config_inp(
            "potential index",
            format!(
                "potential index {} cannot be represented",
                record.potential_index
            ),
        )
    })
}

fn assign_table_row(table: &mut ConfigSlotTable, potential_index: usize, rows: &ConfigSlotRows) {
    table
        .occupations
        .row_mut(potential_index)
        .assign(&rows.occupations);
    table
        .valence_occupations
        .row_mut(potential_index)
        .assign(&rows.valence_occupations);
    table
        .spin_occupations
        .row_mut(potential_index)
        .assign(&rows.spin_occupations);
    table.electron_counts[potential_index] = rows.electron_count;
}

fn slot_rows_from_pairs(
    electron_count: f64,
    occupations: &[(usize, f64)],
    valence_occupations: &[(usize, f64)],
    spin_occupations: &[(usize, f64)],
) -> Result<ConfigSlotRows> {
    let mut rows = ConfigSlotRows::zeros();
    rows.electron_count = electron_count;
    set_slot_pairs("occupations", &mut rows.occupations, occupations)?;
    set_slot_pairs(
        "valence_occupations",
        &mut rows.valence_occupations,
        valence_occupations,
    )?;
    set_slot_pairs(
        "spin_occupations",
        &mut rows.spin_occupations,
        spin_occupations,
    )?;
    validate_slot_rows(&rows)?;
    Ok(rows)
}

fn set_slot_pairs(name: &'static str, row: &mut Array1<f64>, pairs: &[(usize, f64)]) -> Result<()> {
    for (slot, value) in pairs {
        if !(1..=FEFF_ORBITAL_SLOT_COUNT).contains(slot) {
            return Err(invalid_config_inp(
                name,
                format!("slot {slot} is outside FEFF's 1..=40 range"),
            ));
        }
        row[*slot - 1] = *value;
    }
    Ok(())
}

fn noble_gas_base(symbol: &str) -> Option<NobleGasBase> {
    match symbol {
        "He" => Some(NobleGasBase {
            electron_count: 2.0,
            occupations: HE_OCCUPATIONS,
            valence_occupations: HE_VALENCE_OCCUPATIONS,
            spin_occupations: HE_SPIN_OCCUPATIONS,
        }),
        "Ne" => Some(NobleGasBase {
            electron_count: 10.0,
            occupations: NE_OCCUPATIONS,
            valence_occupations: NE_VALENCE_OCCUPATIONS,
            spin_occupations: NE_SPIN_OCCUPATIONS,
        }),
        "Ar" => Some(NobleGasBase {
            electron_count: 18.0,
            occupations: AR_OCCUPATIONS,
            valence_occupations: AR_VALENCE_OCCUPATIONS,
            spin_occupations: AR_SPIN_OCCUPATIONS,
        }),
        "Kr" => Some(NobleGasBase {
            electron_count: 36.0,
            occupations: KR_OCCUPATIONS,
            valence_occupations: KR_VALENCE_OCCUPATIONS,
            spin_occupations: KR_SPIN_OCCUPATIONS,
        }),
        "Xe" => Some(NobleGasBase {
            electron_count: 54.0,
            occupations: XE_OCCUPATIONS,
            valence_occupations: XE_VALENCE_OCCUPATIONS,
            spin_occupations: XE_SPIN_OCCUPATIONS,
        }),
        "Hg" => Some(NobleGasBase {
            electron_count: 80.0,
            occupations: HG_OCCUPATIONS,
            valence_occupations: HG_VALENCE_OCCUPATIONS,
            spin_occupations: HG_SPIN_OCCUPATIONS,
        }),
        "Rn" => Some(NobleGasBase {
            electron_count: 86.0,
            occupations: RN_OCCUPATIONS,
            valence_occupations: RN_VALENCE_OCCUPATIONS,
            spin_occupations: RN_SPIN_OCCUPATIONS,
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct NobleGasBase {
    electron_count: f64,
    occupations: &'static [(usize, f64)],
    valence_occupations: &'static [(usize, f64)],
    spin_occupations: &'static [(usize, f64)],
}

fn parse_element(line: usize, token: &str) -> Result<String> {
    let valid_len = (1..=2).contains(&token.len());
    let valid_chars = token.chars().all(|ch| ch.is_ascii_alphabetic());
    if valid_len && valid_chars {
        Ok(canonical_symbol(token))
    } else {
        Err(IoError::ConfigInpParse {
            field: "element",
            line,
            token: token.to_string(),
        })
    }
}

fn parse_orbital(line: usize, token: &str) -> Result<String> {
    let orbital = token.to_ascii_lowercase();
    if is_allowed_orbital(&orbital) {
        Ok(orbital)
    } else {
        Err(IoError::ConfigInpParse {
            field: "orbital",
            line,
            token: token.to_string(),
        })
    }
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token.parse::<i32>().map_err(|_| IoError::ConfigInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::ConfigInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_config_inp(field, "value must be finite"))
    }
}

fn invalid_config_inp(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidConfigInp {
        field,
        message: message.into(),
    }
}

fn config_lines(text: &str) -> impl Iterator<Item = ConfigLine<'_>> {
    text.lines().enumerate().filter_map(|(index, raw)| {
        let line = strip_inline_comment(raw).trim();
        if line.is_empty() || is_comment_line(line) {
            None
        } else {
            Some(ConfigLine {
                line: index + 1,
                text: line,
            })
        }
    })
}

fn strip_inline_comment(line: &str) -> &str {
    let comment_index = line
        .char_indices()
        .find_map(|(index, ch)| matches!(ch, '#' | '!' | '%').then_some(index));
    comment_index.map_or(line, |index| &line[..index])
}

fn is_comment_line(line: &str) -> bool {
    line.chars()
        .next()
        .is_some_and(|ch| matches!(ch, '#' | '!' | '*' | ';' | 'C' | 'c'))
}

fn canonical_noble_gas(token: &str) -> Option<String> {
    match token.to_ascii_uppercase().as_str() {
        "HE" => Some("He".to_string()),
        "NE" => Some("Ne".to_string()),
        "AR" => Some("Ar".to_string()),
        "KR" => Some("Kr".to_string()),
        "XE" => Some("Xe".to_string()),
        "HG" => Some("Hg".to_string()),
        "RN" => Some("Rn".to_string()),
        _ => None,
    }
}

fn is_zero_base_token(token: &str) -> bool {
    token == "0"
}

fn canonical_symbol(token: &str) -> String {
    let mut chars = token.chars();
    let mut symbol = String::new();
    if let Some(first) = chars.next() {
        symbol.push(first.to_ascii_uppercase());
    }
    if let Some(second) = chars.next() {
        symbol.push(second.to_ascii_lowercase());
    }
    symbol
}

fn is_allowed_orbital(orbital: &str) -> bool {
    matches!(
        orbital,
        "1s" | "2s"
            | "2p"
            | "3s"
            | "3p"
            | "3d"
            | "4s"
            | "4p"
            | "4d"
            | "4f"
            | "5s"
            | "5p"
            | "5d"
            | "5f"
            | "5g"
            | "6s"
            | "6p"
            | "6d"
            | "6f"
            | "7s"
            | "7p"
            | "7d"
            | "8s"
            | "8p"
    )
}

fn is_s_orbital(orbital: &str) -> bool {
    orbital.ends_with('s')
}

#[derive(Debug, Clone, Copy)]
struct ConfigLine<'a> {
    line: usize,
    text: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_generated_config_record() -> Result<()> {
        let parsed = parse_config_inp(GENERATED_CONFIG_INP)?;
        assert_eq!(parsed.records.len(), 1);
        let record = &parsed.records[0];
        assert_eq!(record.potential_index, 0);
        assert_eq!(record.element, "Cu");
        assert_eq!(record.noble_gas, None);
        assert_eq!(record.states.len(), 8);
        assert_eq!(record.states[2].orbital, "2p");
        assert_eq!(
            record.states[2].occupations,
            vec![
                ConfigOccupation {
                    occupation: -2.0,
                    spin: None,
                },
                ConfigOccupation {
                    occupation: -4.0,
                    spin: None,
                },
            ]
        );
        assert_eq!(parsed.potential_indices().to_vec(), vec![0]);
        Ok(())
    }

    #[test]
    fn parses_noble_gas_and_spin_markers() -> Result<()> {
        let parsed = parse_config_inp("4 Cr Ar 3d 4 0 4s 1 4p 1 s 1 0 s 0\n")?;
        let record = &parsed.records[0];
        assert_eq!(record.noble_gas.as_deref(), Some("Ar"));
        let state = &record.states[2];
        assert_eq!(state.orbital, "4p");
        assert_eq!(state.occupations[0].spin, Some(1.0));
        assert_eq!(state.occupations[1].spin, Some(0.0));
        Ok(())
    }

    #[test]
    fn parses_zero_noble_gas_placeholder() -> Result<()> {
        let parsed = parse_config_inp("-1 C 0 1s -2 2s 2 2p 1 1\n")?;
        let record = &parsed.records[0];
        assert_eq!(record.potential_index, -1);
        assert_eq!(record.element, "C");
        assert_eq!(record.noble_gas, None);

        let rows = config_record_slot_rows(record, None)?;
        assert_eq!(first_slots(&rows.occupations, 5), [2.0, 2.0, 1.0, 1.0, 0.0]);
        assert_eq!(
            first_slots(&rows.valence_occupations, 5),
            [0.0, 2.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(rows.electron_count, 6.0);
        Ok(())
    }

    #[test]
    fn expands_config_record_to_feff_orbital_slots() -> Result<()> {
        let parsed = parse_config_inp(GENERATED_CONFIG_INP)?;
        let rows = config_record_slot_rows(&parsed.records[0], None)?;

        assert_eq!(rows.occupations.len(), FEFF_ORBITAL_SLOT_COUNT);
        assert_eq!(
            first_slots(&rows.occupations, 12),
            [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 4.0, 6.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            first_slots(&rows.valence_occupations, 12),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 6.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(first_slots(&rows.spin_occupations, 12), [0.0; 12]);
        assert_eq!(rows.electron_count, 28.0);
        assert_eq!(config_orbital_slot("5g")?, 39);
        Ok(())
    }

    #[test]
    fn expands_config_record_with_feff_noble_gas_base() -> Result<()> {
        let parsed = parse_config_inp("4 Cr Ar 3d 4 0 4s 1 4p 1 s 1 0 s 0\n")?;
        let rows = config_record_slot_rows(&parsed.records[0], None)?;

        assert_eq!(
            first_slots(&rows.occupations, 12),
            [2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 4.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            first_slots(&rows.valence_occupations, 12),
            [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            first_slots(&rows.spin_occupations, 12),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(rows.electron_count, 24.0);
        Ok(())
    }

    #[test]
    fn expands_config_record_with_supplied_noble_gas_base() -> Result<()> {
        let parsed = parse_config_inp("4 Cr Ar 3d 4 0 4s 1 4p 1 s 1 0 s 0\n")?;
        let mut base = ConfigSlotRows::zeros();
        for (slot, occupation) in [
            (1, 2.0),
            (2, 2.0),
            (3, 2.0),
            (4, 4.0),
            (5, 2.0),
            (6, 2.0),
            (7, 4.0),
        ] {
            base.occupations[slot - 1] = occupation;
        }
        base.valence_occupations[5] = 2.0;
        base.valence_occupations[6] = 4.0;
        base.electron_count = 18.0;

        let rows = config_record_slot_rows(&parsed.records[0], Some(&base))?;

        assert_eq!(
            first_slots(&rows.occupations, 12),
            [2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 4.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            first_slots(&rows.valence_occupations, 12),
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            first_slots(&rows.spin_occupations, 12),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(rows.electron_count, 24.0);
        Ok(())
    }

    #[test]
    fn noble_gas_slot_rows_match_feff_defaults() -> Result<()> {
        let cases = [
            ("He", 2.0, 1, 2.0, 1, 2.0, 1, 1.0),
            ("Ne", 10.0, 4, 4.0, 4, 4.0, 4, 1.0),
            ("Ar", 18.0, 7, 4.0, 7, 4.0, 7, 1.0),
            ("Kr", 36.0, 12, 4.0, 12, 4.0, 12, 1.0),
            ("Xe", 54.0, 19, 4.0, 19, 4.0, 19, 1.0),
            ("Hg", 80.0, 24, 2.0, 24, 2.0, 24, 1.0),
            ("Rn", 86.0, 26, 4.0, 26, 4.0, 26, 1.0),
        ];

        for (symbol, electron_count, occ_slot, occ, val_slot, val, spin_slot, spin) in cases {
            let rows = config_noble_gas_slot_rows(symbol)?;
            assert_eq!(rows.electron_count, electron_count);
            assert_eq!(rows.occupations[occ_slot - 1], occ);
            assert_eq!(rows.valence_occupations[val_slot - 1], val);
            assert_eq!(rows.spin_occupations[spin_slot - 1], spin);
        }
        assert!(config_noble_gas_slot_rows("Cu").is_err());
        Ok(())
    }

    #[test]
    fn applies_config_records_to_potential_slot_table() -> Result<()> {
        let parsed = parse_config_inp(
            "-2 Cu 0 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1\n\
             1 O 0 1s -2 2s 2 2p 2 2\n",
        )?;

        let table = config_input_slot_table(&parsed, &["Cu", "O", "Cu"])?;

        assert_eq!(table.potential_count(), 3);
        assert_eq!(table.electron_counts.to_vec(), vec![28.0, 8.0, 28.0]);
        assert_eq!(
            first_slots(&table.occupations.row(0).to_owned(), 12),
            [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 4.0, 6.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            first_slots(&table.occupations.row(2).to_owned(), 12),
            [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 4.0, 6.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            first_slots(&table.valence_occupations.row(1).to_owned(), 5),
            [0.0, 2.0, 2.0, 2.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_config_slot_table_application() -> Result<()> {
        let mismatch = parse_config_inp("1 Cu 1s 2\n")?;
        assert!(config_input_slot_table(&mismatch, &["Cu", "O"]).is_err());

        let out_of_range = parse_config_inp("2 Cu 1s 2\n")?;
        assert!(config_input_slot_table(&out_of_range, &["Cu"]).is_err());
        Ok(())
    }

    #[test]
    fn renders_fixed_width_records() -> Result<()> {
        let parsed = parse_config_inp(GENERATED_CONFIG_INP)?;
        let rendered = config_inp_string(&parsed)?;
        assert_eq!(rendered.lines().next().map(str::len), Some(150));
        assert_eq!(parse_config_inp(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_config_records() -> Result<()> {
        assert!(parse_config_inp("0 Cu 2p 1\n").is_err());
        assert!(parse_config_inp("0 Copper 1s 2\n").is_err());
        assert!(parse_config_inp("0 Cu 9z 1\n").is_err());
        assert!(parse_config_inp("0 Cu 1s NaN\n").is_err());
        assert!(config_inp_lines_string(&["x".repeat(151)]).is_err());
        Ok(())
    }

    const GENERATED_CONFIG_INP: &str =
        "0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0\n";

    fn first_slots<const N: usize>(row: &Array1<f64>, count: usize) -> [f64; N] {
        let mut values = [0.0; N];
        for index in 0..count {
            values[index] = row[index];
        }
        values
    }
}
