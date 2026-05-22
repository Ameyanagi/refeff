use refeff_core::FEFF_ORBITAL_SLOT_COUNT;

use crate::error::Result;

use super::common::{invalid_config_inp, parse_element, validate_finite};
use super::noble_gas::config_noble_gas_slot_rows;
use super::types::{ConfigInput, ConfigRecord, ConfigSlotRows, ConfigSlotTable};

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

pub(super) fn validate_slot_rows(rows: &ConfigSlotRows) -> Result<()> {
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
