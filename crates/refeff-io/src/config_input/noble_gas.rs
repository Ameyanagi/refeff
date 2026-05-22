use ndarray::Array1;
use refeff_core::FEFF_ORBITAL_SLOT_COUNT;

use crate::error::Result;

use super::common::{canonical_noble_gas, invalid_config_inp};
use super::slots::validate_slot_rows;
use super::types::ConfigSlotRows;

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
