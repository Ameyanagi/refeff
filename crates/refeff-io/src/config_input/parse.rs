use crate::error::{IoError, Result};

use super::common::{
    canonical_noble_gas, is_s_orbital, is_zero_base_token, parse_element, parse_f64, parse_i32,
    parse_orbital,
};
use super::types::{ConfigInput, ConfigOccupation, ConfigRecord, ConfigState};
use super::validate::validate_config_input;

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

struct ConfigLine<'a> {
    line: usize,
    text: &'a str,
}
