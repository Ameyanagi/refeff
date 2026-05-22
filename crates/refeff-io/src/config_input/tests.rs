use super::*;

use ndarray::Array1;
use refeff_core::FEFF_ORBITAL_SLOT_COUNT;

use crate::error::Result;

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

const GENERATED_CONFIG_INP: &str = "0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0\n";

fn first_slots<const N: usize>(row: &Array1<f64>, count: usize) -> [f64; N] {
    let mut values = [0.0; N];
    for index in 0..count {
        values[index] = row[index];
    }
    values
}
