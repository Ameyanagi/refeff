//! Typed reader/writer for FEFF `config.inp` electron-configuration files.
//!
//! `RDINP` creates `config.inp` from `CONFIG card` payload lines by writing
//! fixed-width 150-character records. The potential stage later parses those
//! records as `iph element [NobleGas] orbital occupation...` rows.

mod common;
mod noble_gas;
mod parse;
mod render;
mod slots;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use noble_gas::config_noble_gas_slot_rows;
pub use parse::parse_config_inp;
pub use render::{config_inp_lines_string, config_inp_string, read_config_inp, write_config_inp};
pub use slots::{config_input_slot_table, config_orbital_slot, config_record_slot_rows};
pub use types::{
    CONFIG_RECORD_WIDTH, ConfigInput, ConfigOccupation, ConfigRecord, ConfigSlotRows,
    ConfigSlotTable, ConfigState,
};
