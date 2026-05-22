use crate::error::Result;

use super::common::{
    canonical_noble_gas, invalid_config_inp, is_s_orbital, parse_element, parse_orbital,
    validate_finite,
};
use super::types::ConfigInput;

pub(super) fn validate_config_input(input: &ConfigInput) -> Result<()> {
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
