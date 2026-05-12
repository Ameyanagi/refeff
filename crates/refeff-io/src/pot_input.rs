//! Typed reader for FEFF `pot.inp` module handoff files.
//!
//! The POT solver consumes this file after `rdinp` has normalized FEFF cards
//! into fixed module inputs. Keeping the reader typed gives the Rust numerical
//! port a stable boundary and lets tests compare writer and reader behavior
//! before the full potential solver is implemented.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `pot.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct PotInput {
    /// POT module control header.
    pub control: PotControl,
    /// POT run-control switches.
    pub run: PotRun,
    /// Title lines passed through from `feff.inp`.
    pub titles: Vec<String>,
    /// Scalar scattering-potential settings.
    pub scattering: PotScattering,
    /// Per-potential rows from the `iz, lmaxsc, xnatph, xion, folp` block.
    pub potentials: Vec<PotPotential>,
}

/// First integer control line of `pot.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PotControl {
    pub mpot: i32,
    pub nph: i32,
    pub ntitle: i32,
    pub ihole: i32,
    pub ipr1: i32,
    pub iafolp: i32,
    pub ixc: i32,
    pub ispec: i32,
    pub iscfxc: i32,
}

/// Second integer control line of `pot.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PotRun {
    pub nmix: i32,
    pub nohole: i32,
    pub jumprm: i32,
    pub inters: i32,
    pub nscmt: i32,
    pub icoul: i32,
    pub lfms1: i32,
    pub iunf: i32,
}

/// Scalar potential settings from the `gamach` block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotScattering {
    pub gamach: f64,
    pub rgrd: f64,
    pub ca1: f64,
    pub ecv: f64,
    pub totvol: f64,
    pub rfms1: f64,
    pub corval_emin: f64,
}

/// One potential row from `pot.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotPotential {
    pub z: i32,
    pub lmaxsc: i32,
    pub xnatph: f64,
    pub xion: f64,
    pub folp: f64,
}

impl PotInput {
    /// Parse a FEFF `pot.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = PotInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct PotInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> PotInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<PotInput> {
        self.expect_header("mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc")?;
        let control_values = self.parse_values::<i32>(9, "POT control line")?;
        let control = PotControl {
            mpot: control_values[0],
            nph: control_values[1],
            ntitle: control_values[2],
            ihole: control_values[3],
            ipr1: control_values[4],
            iafolp: control_values[5],
            ixc: control_values[6],
            ispec: control_values[7],
            iscfxc: control_values[8],
        };

        self.expect_header("nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf")?;
        let run_values = self.parse_values::<i32>(8, "POT run line")?;
        let run = PotRun {
            nmix: run_values[0],
            nohole: run_values[1],
            jumprm: run_values[2],
            inters: run_values[3],
            nscmt: run_values[4],
            icoul: run_values[5],
            lfms1: run_values[6],
            iunf: run_values[7],
        };

        let title_count = control.ntitle.max(0) as usize;
        let mut titles = Vec::with_capacity(title_count);
        for _ in 0..title_count {
            let (_, line) = self.next_line("title line")?;
            titles.push(line.trim_end().to_string());
        }

        self.expect_header("gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin")?;
        let scattering_values = self.parse_values::<f64>(7, "POT scattering line")?;
        let scattering = PotScattering {
            gamach: scattering_values[0],
            rgrd: scattering_values[1],
            ca1: scattering_values[2],
            ecv: scattering_values[3],
            totvol: scattering_values[4],
            rfms1: scattering_values[5],
            corval_emin: scattering_values[6],
        };

        self.expect_header("iz, lmaxsc, xnatph, xion, folp")?;
        let potential_count = control.nph.max(0) as usize + 1;
        let mut potentials = Vec::with_capacity(potential_count);
        for _ in 0..potential_count {
            let (line_number, line) = self.next_line("potential row")?;
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 5 {
                return Err(self.parse_error(line_number, "potential row requires 5 fields"));
            }
            potentials.push(PotPotential {
                z: parse_field(&self.source, line_number, fields[0])?,
                lmaxsc: parse_field(&self.source, line_number, fields[1])?,
                xnatph: parse_field(&self.source, line_number, fields[2])?,
                xion: parse_field(&self.source, line_number, fields[3])?,
                folp: parse_field(&self.source, line_number, fields[4])?,
            });
        }

        Ok(PotInput {
            control,
            run,
            titles,
            scattering,
            potentials,
        })
    }

    fn expect_header(&mut self, expected: &str) -> Result<()> {
        let (line_number, line) = self.next_line(expected)?;
        if line.trim() == expected {
            Ok(())
        } else {
            Err(self.parse_error(
                line_number,
                format!("expected header {expected:?}, found {line:?}"),
            ))
        }
    }

    fn parse_values<T>(&mut self, count: usize, description: &str) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < count {
            return Err(self.parse_error(
                line_number,
                format!("{description} requires {count} fields"),
            ));
        }
        fields
            .iter()
            .take(count)
            .map(|field| parse_field(&self.source, line_number, field))
            .collect()
    }

    fn next_line(&mut self, description: &str) -> Result<(usize, &'a str)> {
        self.lines
            .next()
            .map(|(index, line)| (index + 1, line))
            .ok_or_else(|| self.parse_error(0, format!("expected {description}")))
    }

    fn parse_error(&self, line: usize, message: impl Into<String>) -> IoError {
        IoError::Parse {
            path: self.source.clone(),
            line,
            message: message.into(),
        }
    }
}

fn parse_field<T>(source: &Path, line: usize, field: &str) -> Result<T>
where
    T: FromStr,
{
    field.parse::<T>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("invalid numeric field {field:?}"),
    })
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::PotInput;

    #[test]
    fn parses_generated_copper_pot_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
SCF 4.0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::pot_inp_string(&document)?;
        let pot = PotInput::parse_str("pot.inp", &text)?;

        assert_eq!(pot.control.ihole, 1);
        assert_eq!(pot.control.ixc, 0);
        assert_eq!(pot.run.nscmt, 100);
        assert_eq!(pot.titles, ["Cu crystal".to_string()]);
        assert_eq!(pot.potentials.len(), 2);
        assert_eq!(pot.potentials[0].z, 29);
        assert_eq!(pot.potentials[1].xnatph, 1.0);
        assert!((pot.scattering.gamach - 1.72919).abs() < 1.0e-5);
        Ok(())
    }
}
