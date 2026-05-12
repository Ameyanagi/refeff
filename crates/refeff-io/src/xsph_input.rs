//! Typed reader for FEFF `xsph.inp` module handoff files.
//!
//! XSPH consumes phase-scattering controls written by `rdinp`. This reader
//! keeps that module boundary explicit for the Rust numerical port.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `xsph.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphInput {
    /// Integer module controls from the first data line.
    pub control: XsphControl,
    /// Real exchange correction values.
    pub vr0: f64,
    pub vi0: f64,
    /// Phase angular momentum cutoffs indexed by potential.
    pub lmaxph: Vec<i32>,
    /// Fixed-width potential labels indexed by potential.
    pub pot_labels: Vec<String>,
    /// Energy and grid controls.
    pub grid: XsphGrid,
    /// Spin moments indexed by potential.
    pub spinph: Vec<f64>,
    /// Optional advanced solver controls.
    pub advanced: XsphAdvanced,
    /// Electronic temperature in eV.
    pub electronic_temperature: f64,
    /// Chemical-shift correction mode.
    pub chsh_type: i32,
    /// NRIXS decomposition-channel count.
    pub decomposition_channels: i32,
    /// Optical calculation switch.
    pub lopt: bool,
    /// Radial wavefunction print switch.
    pub print_rl: bool,
}

/// First integer control line of `xsph.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphControl {
    pub mphase: i32,
    pub ipr2: i32,
    pub ixc: i32,
    pub ixc0: i32,
    pub ispec: i32,
    pub lreal: i32,
    pub lfms2: i32,
    pub nph: i32,
    pub l2lp: i32,
    pub i_plsmn: i32,
    pub n_poles: i32,
    pub i_gamma_ch: i32,
    pub i_grid: i32,
    pub i_core_state: i32,
    pub iscfxc: i32,
}

/// Grid and broadening controls from the `rgrd` data line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphGrid {
    pub rgrd: f64,
    pub rfms2: f64,
    pub gamach: f64,
    pub xkstep: f64,
    pub xkmax: f64,
    pub vixan: f64,
    pub eps0: f64,
    pub egap: f64,
}

/// Advanced XSPH switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphAdvanced {
    pub izstd: i32,
    pub ifxc: i32,
    pub ipmbse: i32,
    pub itdlda: i32,
    pub nonlocal: i32,
    pub ibasis: i32,
}

impl XsphInput {
    /// Parse a FEFF `xsph.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = XsphInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct XsphInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> XsphInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<XsphInput> {
        self.expect_header(
            "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc",
        )?;
        let control_values = self.parse_values::<i32>(15, "XSPH control line")?;
        let control = XsphControl {
            mphase: control_values[0],
            ipr2: control_values[1],
            ixc: control_values[2],
            ixc0: control_values[3],
            ispec: control_values[4],
            lreal: control_values[5],
            lfms2: control_values[6],
            nph: control_values[7],
            l2lp: control_values[8],
            i_plsmn: control_values[9],
            n_poles: control_values[10],
            i_gamma_ch: control_values[11],
            i_grid: control_values[12],
            i_core_state: control_values[13],
            iscfxc: control_values[14],
        };
        let potential_count = control.nph.max(0) as usize + 1;

        self.expect_header("vr0, vi0")?;
        let exchange = self.parse_values::<f64>(2, "XSPH exchange line")?;

        self.expect_header("lmaxph(0:nph)")?;
        let lmaxph = self.parse_values::<i32>(potential_count, "XSPH lmaxph line")?;

        self.expect_header("potlbl(iph)")?;
        let (_, label_line) = self.next_line("XSPH potential labels")?;
        let pot_labels = fixed_a6_labels(label_line, potential_count);

        self.expect_header("rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap")?;
        let grid_values = self.parse_values::<f64>(8, "XSPH grid line")?;
        let grid = XsphGrid {
            rgrd: grid_values[0],
            rfms2: grid_values[1],
            gamach: grid_values[2],
            xkstep: grid_values[3],
            xkmax: grid_values[4],
            vixan: grid_values[5],
            eps0: grid_values[6],
            egap: grid_values[7],
        };

        self.expect_header("spinph(0:nph)")?;
        let spinph = self.parse_values::<f64>(potential_count, "XSPH spinph line")?;

        self.expect_header("izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis")?;
        let advanced_values = self.parse_values::<i32>(6, "XSPH advanced-control line")?;
        let advanced = XsphAdvanced {
            izstd: advanced_values[0],
            ifxc: advanced_values[1],
            ipmbse: advanced_values[2],
            itdlda: advanced_values[3],
            nonlocal: advanced_values[4],
            ibasis: advanced_values[5],
        };

        self.expect_header("electronic temperature")?;
        let electronic_temperature = self.parse_values::<f64>(1, "XSPH temperature line")?[0];
        self.expect_header("ChSh_Type:")?;
        let chsh_type = self.parse_values::<i32>(1, "XSPH chemical-shift line")?[0];
        self.expect_header("the number of decomposition channels ; only used for nrixs")?;
        let decomposition_channels = self.parse_values::<i32>(1, "XSPH decomposition line")?[0];
        self.expect_header("lopt")?;
        let lopt = self.parse_bool_line("XSPH lopt line")?;
        self.expect_header("PrintRL")?;
        let print_rl = self.parse_bool_line("XSPH PrintRL line")?;

        Ok(XsphInput {
            control,
            vr0: exchange[0],
            vi0: exchange[1],
            lmaxph,
            pot_labels,
            grid,
            spinph,
            advanced,
            electronic_temperature,
            chsh_type,
            decomposition_channels,
            lopt,
            print_rl,
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

    fn parse_bool_line(&mut self, description: &str) -> Result<bool> {
        let (line_number, line) = self.next_line(description)?;
        match line.trim() {
            "T" => Ok(true),
            "F" => Ok(false),
            value => Err(self.parse_error(line_number, format!("invalid FEFF bool {value:?}"))),
        }
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

fn fixed_a6_labels(line: &str, count: usize) -> Vec<String> {
    let bytes = line.as_bytes();
    (0..count)
        .map(|idx| {
            let start = idx * 6;
            let end = (start + 6).min(bytes.len());
            if start >= bytes.len() {
                String::new()
            } else {
                String::from_utf8_lossy(&bytes[start..end])
                    .trim_end()
                    .to_string()
            }
        })
        .collect()
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

    use super::XsphInput;

    #[test]
    fn parses_generated_copper_xsph_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
EXAFS 20
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
        let text = rdinp::xsph_inp_string(&document)?;
        let xsph = XsphInput::parse_str("xsph.inp", &text)?;

        assert_eq!(xsph.control.mphase, 1);
        assert_eq!(xsph.control.nph, 1);
        assert_eq!(xsph.control.n_poles, 100);
        assert_eq!(xsph.pot_labels, ["Cu".to_string(), "Cu".to_string()]);
        assert_eq!(xsph.lmaxph, [3, 3]);
        assert_eq!(xsph.spinph, [0.0, 0.0]);
        assert!((xsph.grid.gamach - 1.72919).abs() < 1.0e-5);
        assert!(!xsph.lopt);
        assert!(!xsph.print_rl);
        Ok(())
    }
}
