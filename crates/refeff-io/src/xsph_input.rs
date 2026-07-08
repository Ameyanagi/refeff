//! Typed reader for FEFF `xsph.inp` module handoff files.
//!
//! XSPH consumes phase-scattering controls written by `rdinp`. This reader
//! keeps that module boundary explicit for the Rust numerical port.

use std::fmt::Write as _;
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
    /// Structural format used by the source `xsph.inp` handoff.
    pub source_format: XsphInputSourceFormat,
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

/// Structural format used by a parsed FEFF `xsph.inp` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphInputSourceFormat {
    pub control_header: XsphControlHeaderFormat,
    pub grid_header: XsphGridHeaderFormat,
}

impl XsphInputSourceFormat {
    pub const fn modern() -> Self {
        Self {
            control_header: XsphControlHeaderFormat::Modern,
            grid_header: XsphGridHeaderFormat::Modern,
        }
    }

    pub const fn uses_legacy_fixed_xanes_mesh(self) -> bool {
        matches!(
            self.control_header,
            XsphControlHeaderFormat::LegacyWithPoles | XsphControlHeaderFormat::LegacyWithoutPoles
        ) || matches!(self.grid_header, XsphGridHeaderFormat::Legacy)
    }
}

/// Integer-control header shape used by a parsed FEFF `xsph.inp` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphControlHeaderFormat {
    Modern,
    LegacyWithPoles,
    LegacyWithoutPoles,
}

impl XsphControlHeaderFormat {
    fn field_count(self) -> usize {
        match self {
            Self::Modern => 15,
            Self::LegacyWithPoles => 13,
            Self::LegacyWithoutPoles => 12,
        }
    }

    fn n_poles(self, values: &[i32]) -> i32 {
        match self {
            Self::Modern | Self::LegacyWithPoles => values[10],
            Self::LegacyWithoutPoles => 100,
        }
    }

    fn i_gamma_ch(self, values: &[i32]) -> i32 {
        match self {
            Self::Modern | Self::LegacyWithPoles => values[11],
            Self::LegacyWithoutPoles => values[10],
        }
    }

    fn i_grid(self, values: &[i32]) -> i32 {
        match self {
            Self::Modern | Self::LegacyWithPoles => values[12],
            Self::LegacyWithoutPoles => values[11],
        }
    }

    fn i_core_state(self, values: &[i32]) -> i32 {
        match self {
            Self::Modern => values[13],
            Self::LegacyWithPoles | Self::LegacyWithoutPoles => 0,
        }
    }

    fn iscfxc(self, values: &[i32]) -> i32 {
        match self {
            Self::Modern => values[14],
            Self::LegacyWithPoles | Self::LegacyWithoutPoles => 11,
        }
    }
}

/// Grid header shape used by a parsed FEFF `xsph.inp` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphGridHeaderFormat {
    Modern,
    Legacy,
}

impl XsphGridHeaderFormat {
    fn field_count(self) -> usize {
        match self {
            Self::Modern => 8,
            Self::Legacy => 6,
        }
    }

    fn eps0(self, values: &[f64]) -> f64 {
        match self {
            Self::Modern => values[6],
            Self::Legacy => 0.0,
        }
    }

    fn egap(self, values: &[f64]) -> f64 {
        match self {
            Self::Modern => values[7],
            Self::Legacy => 0.0,
        }
    }

    fn has_labeled_following_rows(self) -> bool {
        matches!(self, Self::Modern)
    }
}

impl XsphInput {
    /// Parse a FEFF `xsph.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = XsphInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `xsph.inp` text.
pub fn xsph_input_string(input: &XsphInput) -> Result<String> {
    validate_xsph_input(input)?;

    let mut out = String::new();
    writeln!(
        out,
        "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc"
    )?;
    push_i4_row(
        &mut out,
        [
            input.control.mphase,
            input.control.ipr2,
            input.control.ixc,
            input.control.ixc0,
            input.control.ispec,
            input.control.lreal,
            input.control.lfms2,
            input.control.nph,
            input.control.l2lp,
            input.control.i_plsmn,
            input.control.n_poles,
            input.control.i_gamma_ch,
            input.control.i_grid,
            input.control.i_core_state,
            input.control.iscfxc,
        ],
    )?;
    writeln!(out, "vr0, vi0")?;
    writeln!(out, "{:13.5}{:13.5}", input.vr0, input.vi0)?;
    writeln!(out, " lmaxph(0:nph)")?;
    push_i4_row(&mut out, input.lmaxph.iter().copied())?;
    writeln!(out, " potlbl(iph)")?;
    for label in &input.pot_labels {
        out.push_str(&fixed_a6(label));
    }
    out.push('\n');
    writeln!(out, "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        input.grid.rgrd,
        input.grid.rfms2,
        input.grid.gamach,
        input.grid.xkstep,
        input.grid.xkmax,
        input.grid.vixan,
        input.grid.eps0,
        input.grid.egap
    )?;
    writeln!(out, "spinph(0:nph)")?;
    for spin in &input.spinph {
        write!(out, "{spin:13.5}")?;
    }
    out.push('\n');
    writeln!(out, "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis")?;
    push_i4_row(
        &mut out,
        [
            input.advanced.izstd,
            input.advanced.ifxc,
            input.advanced.ipmbse,
            input.advanced.itdlda,
            input.advanced.nonlocal,
            input.advanced.ibasis,
        ],
    )?;
    writeln!(out, "electronic temperature")?;
    writeln!(out, "{:13.5}", input.electronic_temperature)?;
    writeln!(out, "ChSh_Type:")?;
    writeln!(out, "{:4}", input.chsh_type)?;
    writeln!(
        out,
        " the number of decomposition channels ; only used for nrixs"
    )?;
    writeln!(out, "{:5}", input.decomposition_channels)?;
    writeln!(out, "lopt")?;
    writeln!(out, " {}", fortran_bool_field(input.lopt))?;
    writeln!(out, "PrintRL")?;
    writeln!(out, " {}", fortran_bool_field(input.print_rl))?;
    Ok(out)
}

fn validate_xsph_input(input: &XsphInput) -> Result<()> {
    let expected = input.control.nph.max(0) as usize + 1;
    for (field, len) in [
        ("lmaxph", input.lmaxph.len()),
        ("pot_labels", input.pot_labels.len()),
        ("spinph", input.spinph.len()),
    ] {
        if len != expected {
            return Err(IoError::Parse {
                path: "xsph.inp".into(),
                line: 0,
                message: format!(
                    "{field} length {len} does not match nph-derived count {expected}"
                ),
            });
        }
    }
    validate_finite("vr0", input.vr0)?;
    validate_finite("vi0", input.vi0)?;
    validate_finite("rgrd", input.grid.rgrd)?;
    validate_finite("rfms2", input.grid.rfms2)?;
    validate_finite("gamach", input.grid.gamach)?;
    validate_finite("xkstep", input.grid.xkstep)?;
    validate_finite("xkmax", input.grid.xkmax)?;
    validate_finite("vixan", input.grid.vixan)?;
    validate_finite("eps0", input.grid.eps0)?;
    validate_finite("egap", input.grid.egap)?;
    for spin in &input.spinph {
        validate_finite("spinph", *spin)?;
    }
    validate_finite("electronic_temperature", input.electronic_temperature)?;
    for label in &input.pot_labels {
        if label.contains(['\n', '\r']) {
            return Err(IoError::Parse {
                path: "xsph.inp".into(),
                line: 0,
                message: "XSPH potential labels cannot contain line terminators".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "xsph.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn fortran_bool_field(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

fn push_i4_row(out: &mut String, values: impl IntoIterator<Item = i32>) -> Result<()> {
    for value in values {
        write!(out, "{value:4}")?;
    }
    out.push('\n');
    Ok(())
}

fn fixed_a6(value: &str) -> String {
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > 6 {
            break;
        }
        end = next;
    }
    let mut out = value[..end].to_string();
    while out.len() < 6 {
        out.push(' ');
    }
    out
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
        let control_header = self.expect_control_header()?;
        let control_values =
            self.parse_values::<i32>(control_header.field_count(), "XSPH control line")?;
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
            n_poles: control_header.n_poles(&control_values),
            i_gamma_ch: control_header.i_gamma_ch(&control_values),
            i_grid: control_header.i_grid(&control_values),
            i_core_state: control_header.i_core_state(&control_values),
            iscfxc: control_header.iscfxc(&control_values),
        };
        let potential_count = control.nph.max(0) as usize + 1;

        self.expect_header("vr0, vi0")?;
        let exchange = self.parse_values::<f64>(2, "XSPH exchange line")?;

        self.expect_header("lmaxph(0:nph)")?;
        let lmaxph = self.parse_values::<i32>(potential_count, "XSPH lmaxph line")?;

        self.expect_header("potlbl(iph)")?;
        let (_, label_line) = self.next_line("XSPH potential labels")?;
        let pot_labels = fixed_a6_labels(label_line, potential_count);

        let grid_header = self.expect_grid_header()?;
        let grid_values = self.parse_values::<f64>(grid_header.field_count(), "XSPH grid line")?;
        let grid = XsphGrid {
            rgrd: grid_values[0],
            rfms2: grid_values[1],
            gamach: grid_values[2],
            xkstep: grid_values[3],
            xkmax: grid_values[4],
            vixan: grid_values[5],
            eps0: grid_header.eps0(&grid_values),
            egap: grid_header.egap(&grid_values),
        };

        let optional_following_labels = !matches!(control_header, XsphControlHeaderFormat::Modern);
        let spinph = if grid_header.has_labeled_following_rows() && optional_following_labels {
            self.parse_values_after_optional_header(
                "spinph(0:nph)",
                potential_count,
                "XSPH spinph line",
            )?
        } else {
            if grid_header.has_labeled_following_rows() {
                self.expect_header("spinph(0:nph)")?;
            }
            self.parse_values::<f64>(potential_count, "XSPH spinph line")?
        };

        let advanced_values =
            if grid_header.has_labeled_following_rows() && optional_following_labels {
                self.parse_values_after_optional_header(
                    "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis",
                    6,
                    "XSPH advanced-control line",
                )?
            } else {
                if grid_header.has_labeled_following_rows() {
                    self.expect_header("izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis")?;
                }
                self.parse_values::<i32>(6, "XSPH advanced-control line")?
            };
        let advanced = XsphAdvanced {
            izstd: advanced_values[0],
            ifxc: advanced_values[1],
            ipmbse: advanced_values[2],
            itdlda: advanced_values[3],
            nonlocal: advanced_values[4],
            ibasis: advanced_values[5],
        };

        let electronic_temperature = self.parse_optional_electronic_temperature()?;
        let chsh_type = self.parse_values::<i32>(1, "XSPH chemical-shift line")?[0];
        self.expect_header("the number of decomposition channels ; only used for nrixs")?;
        let decomposition_channels = self.parse_values::<i32>(1, "XSPH decomposition line")?[0];
        self.expect_header("lopt")?;
        let lopt = self.parse_bool_line("XSPH lopt line")?;
        let print_rl = self.parse_optional_print_rl()?;

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
            source_format: XsphInputSourceFormat {
                control_header,
                grid_header,
            },
        })
    }

    fn expect_control_header(&mut self) -> Result<XsphControlHeaderFormat> {
        let (line_number, line) = self.next_line("XSPH control header")?;
        match line.trim() {
            "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc" => {
                Ok(XsphControlHeaderFormat::Modern)
            }
            "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid" => {
                Ok(XsphControlHeaderFormat::LegacyWithPoles)
            }
            "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,iGammaCH,iGrid" => {
                Ok(XsphControlHeaderFormat::LegacyWithoutPoles)
            }
            _ => Err(self.parse_error(
                line_number,
                format!("expected XSPH control header, found {line:?}"),
            )),
        }
    }

    fn expect_grid_header(&mut self) -> Result<XsphGridHeaderFormat> {
        let (line_number, line) = self.next_line("XSPH grid header")?;
        match line.trim() {
            "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap" => {
                Ok(XsphGridHeaderFormat::Modern)
            }
            "rgrd, rfms2, gamach, xkstep, xkmax, vixan" => Ok(XsphGridHeaderFormat::Legacy),
            _ => Err(self.parse_error(
                line_number,
                format!("expected XSPH grid header, found {line:?}"),
            )),
        }
    }

    fn parse_optional_electronic_temperature(&mut self) -> Result<f64> {
        let (line_number, line) =
            self.next_line("XSPH electronic temperature or chemical-shift header")?;
        match line.trim() {
            "electronic temperature" => {
                let temperature = self.parse_values::<f64>(1, "XSPH temperature line")?[0];
                self.expect_header("ChSh_Type:")?;
                Ok(temperature)
            }
            "ChSh_Type:" => Ok(0.0),
            _ => Err(self.parse_error(
                line_number,
                format!(
                    "expected XSPH electronic temperature or chemical-shift header, found {line:?}"
                ),
            )),
        }
    }

    fn parse_optional_print_rl(&mut self) -> Result<bool> {
        let Some((index, line)) = self.lines.next() else {
            return Ok(false);
        };
        let line_number = index + 1;
        if line.trim() != "PrintRL" {
            return Err(self.parse_error(
                line_number,
                format!("expected header {:?}, found {line:?}", "PrintRL"),
            ));
        }
        self.parse_bool_line("XSPH PrintRL line")
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
        self.parse_values_from_line(line_number, line, count, description)
    }

    fn parse_values_after_optional_header<T>(
        &mut self,
        header: &str,
        count: usize,
        description: &str,
    ) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let (line_number, line) = self.next_line(description)?;
        if line.trim() == header {
            self.parse_values::<T>(count, description)
        } else {
            self.parse_values_from_line(line_number, line, count, description)
        }
    }

    fn parse_values_from_line<T>(
        &self,
        line_number: usize,
        line: &str,
        count: usize,
        description: &str,
    ) -> Result<Vec<T>>
    where
        T: FromStr,
    {
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

    use super::{XsphControlHeaderFormat, XsphGridHeaderFormat, XsphInput, xsph_input_string};

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
        assert_eq!(
            xsph.source_format.control_header,
            XsphControlHeaderFormat::Modern
        );
        assert_eq!(xsph.source_format.grid_header, XsphGridHeaderFormat::Modern);
        assert!(!xsph.source_format.uses_legacy_fixed_xanes_mesh());
        assert_eq!(xsph.pot_labels, ["Cu".to_string(), "Cu".to_string()]);
        assert_eq!(xsph.lmaxph, [3, 3]);
        assert_eq!(xsph.spinph, [0.0, 0.0]);
        assert!((xsph.grid.gamach - 1.72919).abs() < 1.0e-5);
        assert!(!xsph.lopt);
        assert!(!xsph.print_rl);
        assert_eq!(xsph_input_string(&xsph)?, text);
        Ok(())
    }

    #[test]
    fn parses_legacy_xsph_input_without_core_state_or_scxc() -> crate::Result<()> {
        let text = "\
mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid
   1   0   0   2   3   0   0   2   0   0 100   0   0
vr0, vi0
      0.00000      1.00000
 lmaxph(0:nph)
   2   2   2
 potlbl(iph)
B     B     N
rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap
      0.05000      5.00000      0.06473      0.07000      4.00000      0.00000      0.00000      0.00000
spinph(0:nph)
      0.00000      0.00000      0.00000
izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis
   0   0   0   0   0   0
ChSh_Type:
   0
 the number of decomposition channels ; only used for nrixs
   -1
lopt
 F
";
        let xsph = XsphInput::parse_str("legacy-xsph.inp", text)?;

        assert_eq!(xsph.control.n_poles, 100);
        assert_eq!(xsph.control.i_core_state, 0);
        assert_eq!(xsph.control.iscfxc, 11);
        assert_eq!(
            xsph.source_format.control_header,
            XsphControlHeaderFormat::LegacyWithPoles
        );
        assert_eq!(xsph.source_format.grid_header, XsphGridHeaderFormat::Modern);
        assert!(xsph.source_format.uses_legacy_fixed_xanes_mesh());
        assert_eq!(xsph.electronic_temperature, 0.0);
        assert!(!xsph.print_rl);
        assert_eq!(
            xsph.pot_labels,
            ["B".to_string(), "B".to_string(), "N".to_string()]
        );
        Ok(())
    }

    #[test]
    fn parses_legacy_xsph_input_with_modern_grid_but_unlabeled_following_rows() -> crate::Result<()>
    {
        let text = "\
mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid
   1   0   0   2   1   1   1   1   0   0 100   0   0
vr0, vi0
      0.00000      0.00000
 lmaxph(0:nph)
   3   3
 potlbl(iph)
Ge    Cl
rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap
      0.05000      3.00000      2.28438      0.05000      8.00000      0.00000      0.00000      0.00000
      0.00000      0.00000
   0   0   0   0   0   0
ChSh_Type:
   0
 the number of decomposition channels ; only used for nrixs
    2
lopt
 F
";
        let xsph = XsphInput::parse_str("legacy-hybrid-xsph.inp", text)?;

        assert_eq!(
            xsph.source_format.control_header,
            XsphControlHeaderFormat::LegacyWithPoles
        );
        assert_eq!(xsph.source_format.grid_header, XsphGridHeaderFormat::Modern);
        assert_eq!(xsph.spinph, [0.0, 0.0]);
        assert_eq!(xsph.advanced.ibasis, 0);
        assert_eq!(xsph.decomposition_channels, 2);
        Ok(())
    }

    #[test]
    fn parses_older_xsph_input_without_poles_or_labeled_spin_rows() -> crate::Result<()> {
        let text = "\
mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,iGammaCH,iGrid
   1   0   0   2   3   1   1   1   0   0   0   0
vr0, vi0
      0.00000      0.00000
 lmaxph(0:nph)
   3   3
 potlbl(iph)
Ge    Cl
rgrd, rfms2, gamach, xkstep, xkmax, vixan
      0.05000      3.00000      2.28438      0.05000      8.00000      0.00000
      0.00000      0.00000
   0   0   0   0   0   0
ChSh_Type:
   0
 the number of decomposition channels ; only used for nrixs
   -1
lopt
 F
";
        let xsph = XsphInput::parse_str("older-xsph.inp", text)?;

        assert_eq!(xsph.control.n_poles, 100);
        assert_eq!(xsph.control.i_gamma_ch, 0);
        assert_eq!(xsph.control.i_grid, 0);
        assert_eq!(xsph.control.i_core_state, 0);
        assert_eq!(xsph.control.iscfxc, 11);
        assert_eq!(
            xsph.source_format.control_header,
            XsphControlHeaderFormat::LegacyWithoutPoles
        );
        assert_eq!(xsph.source_format.grid_header, XsphGridHeaderFormat::Legacy);
        assert!(xsph.source_format.uses_legacy_fixed_xanes_mesh());
        assert_eq!(xsph.grid.eps0, 0.0);
        assert_eq!(xsph.grid.egap, 0.0);
        assert_eq!(xsph.spinph, [0.0, 0.0]);
        assert_eq!(xsph.advanced.ibasis, 0);
        assert!(!xsph.print_rl);
        Ok(())
    }

    #[test]
    fn rejects_invalid_xsph_rendering() {
        let input = XsphInput {
            control: super::XsphControl {
                mphase: 1,
                ipr2: 0,
                ixc: 0,
                ixc0: 0,
                ispec: 1,
                lreal: 0,
                lfms2: 0,
                nph: 0,
                l2lp: 0,
                i_plsmn: 0,
                n_poles: 100,
                i_gamma_ch: 0,
                i_grid: 1,
                i_core_state: -1,
                iscfxc: 11,
            },
            vr0: f64::NAN,
            vi0: 0.0,
            lmaxph: vec![3],
            pot_labels: vec!["Cu".to_string()],
            grid: super::XsphGrid {
                rgrd: 0.05,
                rfms2: -1.0,
                gamach: 1.0,
                xkstep: 0.07,
                xkmax: 20.0,
                vixan: 0.0,
                eps0: 0.0,
                egap: 0.0,
            },
            spinph: vec![0.0],
            advanced: super::XsphAdvanced {
                izstd: 0,
                ifxc: 0,
                ipmbse: 0,
                itdlda: 0,
                nonlocal: 0,
                ibasis: 0,
            },
            electronic_temperature: 0.0,
            chsh_type: 0,
            decomposition_channels: -1,
            lopt: false,
            print_rl: false,
            source_format: super::XsphInputSourceFormat::modern(),
        };
        assert!(xsph_input_string(&input).is_err());

        let mut bad_count = input;
        bad_count.vr0 = 0.0;
        bad_count.lmaxph.clear();
        assert!(xsph_input_string(&bad_count).is_err());
    }
}
