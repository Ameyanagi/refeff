use std::str::FromStr;

use crate::error::{IoError, Result};

#[derive(Debug, Clone, Copy)]
struct DymToken<'a> {
    line: usize,
    text: &'a str,
}

#[derive(Debug)]
pub(super) struct DymTokenCursor<'a> {
    tokens: Vec<DymToken<'a>>,
    index: usize,
}

impl<'a> DymTokenCursor<'a> {
    pub(super) fn new(text: &'a str) -> Self {
        let tokens = text
            .lines()
            .enumerate()
            .flat_map(|(line, text)| {
                text.split_whitespace().map(move |token| DymToken {
                    line: line + 1,
                    text: token,
                })
            })
            .collect();
        Self { tokens, index: 0 }
    }

    pub(super) fn parse<T>(&mut self, field: &'static str) -> Result<T>
    where
        T: FromStr,
    {
        let token = self.next_token(field)?;
        token.text.parse::<T>().map_err(|_| IoError::DymParse {
            field,
            line: token.line,
            token: token.text.to_string(),
        })
    }

    fn next_token(&mut self, field: &'static str) -> Result<DymToken<'a>> {
        let Some(token) = self.tokens.get(self.index).copied() else {
            return Err(IoError::DymMissing { field });
        };
        self.index += 1;
        Ok(token)
    }

    pub(super) fn remaining_count(&self) -> usize {
        self.tokens.len().saturating_sub(self.index)
    }
}
