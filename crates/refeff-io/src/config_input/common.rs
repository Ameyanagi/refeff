use crate::error::{IoError, Result};

pub(super) fn parse_element(line: usize, token: &str) -> Result<String> {
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

pub(super) fn parse_orbital(line: usize, token: &str) -> Result<String> {
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

pub(super) fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token.parse::<i32>().map_err(|_| IoError::ConfigInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

pub(super) fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::ConfigInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

pub(super) fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_config_inp(field, "value must be finite"))
    }
}

pub(super) fn invalid_config_inp(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidConfigInp {
        field,
        message: message.into(),
    }
}

pub(super) fn canonical_noble_gas(token: &str) -> Option<String> {
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

pub(super) fn is_zero_base_token(token: &str) -> bool {
    token == "0"
}

pub(super) fn canonical_symbol(token: &str) -> String {
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

pub(super) fn is_allowed_orbital(orbital: &str) -> bool {
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

pub(super) fn is_s_orbital(orbital: &str) -> bool {
    orbital.ends_with('s')
}
