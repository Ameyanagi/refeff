//! Typed reader for FEFF `pot.inp` module handoff files.
//!
//! The POT solver consumes this file after `rdinp` has normalized FEFF cards
//! into fixed module inputs. Keeping the reader typed gives the Rust numerical
//! port a stable boundary and lets tests compare writer and reader behavior
//! before the full potential solver is implemented.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use refeff_core::FEFF_HARTREE_EV;

use crate::{IoError, Result};

const RHORRP_MIN_TEMPERATURE_HARTREE: f64 = 1.0e-3;

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
    /// Whether POT should read external muffin-tin potentials.
    pub external_pot: bool,
    /// Whether POT should restart from a prior `pot.bin` file.
    pub start_from_file: bool,
    /// Manual overlap-shell rows grouped by potential index.
    pub overlap_shells: Vec<Vec<PotOverlapShell>>,
    /// Chemical-shift correction mode.
    pub chsh_type: i32,
    /// Atomic configuration selection mode.
    pub config_type: i32,
    /// Thermal-SCF and electronic-temperature controls.
    pub thermal: PotThermal,
    /// Finite-nucleus calculation switch.
    pub finite_nucleus: bool,
    /// Ionicity warning switch.
    pub warn_ion: bool,
    /// SCF radius ramp controls.
    pub ramp: PotRamp,
    /// SCF convergence tolerances.
    pub tolerances: PotTolerances,
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

/// One manual overlap-shell row for a potential in `pot.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotOverlapShell {
    pub iphovr: i32,
    pub nnovr: i32,
    pub rovr: f64,
}

/// Electronic-temperature and thermal-SCF settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotThermal {
    pub scf_temperature: f64,
    pub scf_thermal_vxc: i32,
    pub iscfth: i32,
    pub xntol: f64,
    pub nmu: i32,
    pub negrid: i32,
    pub emaxscf: f64,
}

/// SCF radial cutoff ramp settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotRamp {
    pub ramp_scf: bool,
    pub rfms_start: f64,
    pub nramp: i32,
}

/// POT self-consistency convergence tolerances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotTolerances {
    pub tolmu: f64,
    pub tolq: f64,
    pub tolqp: f64,
}

/// RHORRP controls imported from FEFF `potential_inp`.
///
/// `RHORRP/m_rhorrp.f90` reads the potential module inputs for the exchange
/// selector, target logarithmic radial step, and SCF electronic temperature.
/// The temperature is converted from eV to Hartree and floored with the same
/// `max(scf_temperature/hart, 0.001)` rule used before RHORRP contour
/// integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpPotInputControls {
    /// FEFF `ixc` exchange-correlation selector.
    pub exchange_index: i32,
    /// FEFF `rgrd` target logarithmic radial-grid step.
    pub target_radial_dx: f64,
    /// FEFF `scf_temperature` converted from eV to Hartree before flooring.
    pub raw_temperature_hartree: f64,
    /// RHORRP effective electronic temperature in Hartree.
    pub temperature_hartree: f64,
}

impl PotInput {
    /// Parse a FEFF `pot.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = PotInputParser::new(source.into(), text);
        parser.parse()
    }

    /// Extract RHORRP controls from this parsed `pot.inp`.
    pub fn to_rhorrp_controls(&self) -> Result<RhorrpPotInputControls> {
        rhorrp_controls_from_pot_input(self)
    }
}

/// Build RHORRP potential-module controls from parsed FEFF `pot.inp`.
pub fn rhorrp_controls_from_pot_input(input: &PotInput) -> Result<RhorrpPotInputControls> {
    validate_pot_input(input)?;
    if input.scattering.rgrd <= 0.0 {
        return Err(invalid_rhorrp_pot_input(
            "rgrd",
            format!(
                "RHORRP target radial step must be positive, got {}",
                input.scattering.rgrd
            ),
        ));
    }

    let raw_temperature_hartree = input.thermal.scf_temperature / FEFF_HARTREE_EV;
    Ok(RhorrpPotInputControls {
        exchange_index: input.control.ixc,
        target_radial_dx: input.scattering.rgrd,
        raw_temperature_hartree,
        temperature_hartree: raw_temperature_hartree.max(RHORRP_MIN_TEMPERATURE_HARTREE),
    })
}

/// Render FEFF-compatible `pot.inp` text.
pub fn pot_input_string(input: &PotInput) -> Result<String> {
    validate_pot_input(input)?;

    let mut out = String::new();
    writeln!(
        out,
        "mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc"
    )?;
    push_i4_row(
        &mut out,
        [
            input.control.mpot,
            input.control.nph,
            input.control.ntitle,
            input.control.ihole,
            input.control.ipr1,
            input.control.iafolp,
            input.control.ixc,
            input.control.ispec,
            input.control.iscfxc,
        ],
    )?;
    writeln!(
        out,
        "nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf"
    )?;
    push_i4_row(
        &mut out,
        [
            input.run.nmix,
            input.run.nohole,
            input.run.jumprm,
            input.run.inters,
            input.run.nscmt,
            input.run.icoul,
            input.run.lfms1,
            input.run.iunf,
        ],
    )?;
    for title in &input.titles {
        writeln!(out, "{}", fixed_title(title))?;
    }
    writeln!(out, "gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        input.scattering.gamach,
        input.scattering.rgrd,
        input.scattering.ca1,
        input.scattering.ecv,
        input.scattering.totvol,
        input.scattering.rfms1,
        input.scattering.corval_emin
    )?;
    writeln!(out, " iz, lmaxsc, xnatph, xion, folp")?;
    for potential in &input.potentials {
        writeln!(
            out,
            "{:5}{:5}{:20.10}{:20.10}{:20.10}",
            potential.z, potential.lmaxsc, potential.xnatph, potential.xion, potential.folp
        )?;
    }
    writeln!(out, "ExternalPot switch, StartFromFile switch")?;
    writeln!(
        out,
        " {} {}",
        fortran_bool_field(input.external_pot),
        fortran_bool_field(input.start_from_file)
    )?;
    writeln!(out, "OVERLAP option: novr(iph)")?;
    push_i4_row(
        &mut out,
        input
            .overlap_shells
            .iter()
            .map(|shells| i32::try_from(shells.len()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| IoError::Parse {
                path: "pot.inp".into(),
                line: 0,
                message: "overlap shell count exceeds i32 range".to_string(),
            })?,
    )?;
    writeln!(out, " iphovr  nnovr rovr ")?;
    for shells in &input.overlap_shells {
        for shell in shells {
            writeln!(
                out,
                "{:5}{:5}{:13.5}",
                shell.iphovr, shell.nnovr, shell.rovr
            )?;
        }
    }
    writeln!(out, "ChSh_Type:")?;
    writeln!(out, "{:4}", input.chsh_type)?;
    writeln!(out, "ConfigType:")?;
    writeln!(out, "{:4}", input.config_type)?;
    writeln!(out, "Temperature (in eV):")?;
    write_pot_temperature(
        &mut out,
        input.thermal.scf_temperature,
        input.thermal.scf_thermal_vxc,
    )?;
    writeln!(out, "scf_th,  xntol,  nmu")?;
    write_pot_thermal_scf(
        &mut out,
        input.thermal.iscfth,
        input.thermal.xntol,
        input.thermal.nmu,
    )?;
    writeln!(out, "negrid,  emaxscf")?;
    writeln!(
        out,
        "{:12}{:21.16}     ",
        input.thermal.negrid, input.thermal.emaxscf
    )?;
    writeln!(out, "FiniteNucleus, WarnIon")?;
    writeln!(
        out,
        " {} {}",
        fortran_bool_field(input.finite_nucleus),
        fortran_bool_field(input.warn_ion)
    )?;
    writeln!(out, "ramp_scf  rfms_start  nramp")?;
    writeln!(
        out,
        " {}{:13.8}{:16}",
        fortran_bool_field(input.ramp.ramp_scf),
        input.ramp.rfms_start,
        input.ramp.nramp
    )?;
    writeln!(out, "tolmu, tolq, tolqp")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.tolerances.tolmu, input.tolerances.tolq, input.tolerances.tolqp
    )?;
    Ok(out)
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
        let control_values = self.parse_array::<i32, 9>("POT control line")?;
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
        if control.nph < 0 {
            return Err(self.parse_error(0, "POT nph cannot be negative"));
        }
        if control.ntitle < 0 {
            return Err(self.parse_error(0, "POT ntitle cannot be negative"));
        }

        self.expect_header("nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf")?;
        let run_values = self.parse_array::<i32, 8>("POT run line")?;
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

        let title_count = control.ntitle as usize;
        let mut titles = Vec::with_capacity(title_count);
        for _ in 0..title_count {
            let (_, line) = self.next_line("title line")?;
            titles.push(line.trim_end().to_string());
        }

        self.expect_header("gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin")?;
        let scattering_values = self.parse_array::<f64, 7>("POT scattering line")?;
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
        let potential_count = control.nph as usize + 1;
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

        self.expect_header("ExternalPot switch, StartFromFile switch")?;
        let switches = self.parse_bool_array::<2>("POT external-potential switch line")?;
        let external_pot = switches[0];
        let start_from_file = switches[1];

        self.expect_header("OVERLAP option: novr(iph)")?;
        let overlap_counts = self.parse_values::<i32>(potential_count, "POT overlap count line")?;
        self.expect_header("iphovr  nnovr rovr")?;
        let mut overlap_shells = Vec::with_capacity(potential_count);
        for count in overlap_counts {
            if count < 0 {
                return Err(self.parse_error(0, "POT overlap count cannot be negative"));
            }
            let mut shells = Vec::with_capacity(count as usize);
            for _ in 0..count {
                shells.push(self.parse_overlap_shell()?);
            }
            overlap_shells.push(shells);
        }

        self.expect_header("ChSh_Type:")?;
        let chsh_type = self.parse_array::<i32, 1>("POT chemical-shift line")?[0];
        self.expect_header("ConfigType:")?;
        let config_type = self.parse_array::<i32, 1>("POT config-type line")?[0];
        self.expect_header("Temperature (in eV):")?;
        let temperature_values = self.parse_float_int("POT electronic-temperature line")?;
        self.expect_header("scf_th,  xntol,  nmu")?;
        let scf_values = self.parse_int_float_int("POT thermal-SCF line")?;
        self.expect_header("negrid,  emaxscf")?;
        let grid_values = self.parse_int_float("POT thermal grid line")?;
        self.expect_header("FiniteNucleus, WarnIon")?;
        let finite_switches = self.parse_bool_array::<2>("POT finite-nucleus switch line")?;
        self.expect_header("ramp_scf  rfms_start  nramp")?;
        let ramp_values = self.parse_mixed_bool_float_int("POT SCF ramp line")?;
        self.expect_header("tolmu, tolq, tolqp")?;
        let tolerance_values = self.parse_array::<f64, 3>("POT tolerance line")?;

        Ok(PotInput {
            control,
            run,
            titles,
            scattering,
            potentials,
            external_pot,
            start_from_file,
            overlap_shells,
            chsh_type,
            config_type,
            thermal: PotThermal {
                scf_temperature: temperature_values.0,
                scf_thermal_vxc: temperature_values.1,
                iscfth: scf_values.0,
                xntol: scf_values.1,
                nmu: scf_values.2,
                negrid: grid_values.0,
                emaxscf: grid_values.1,
            },
            finite_nucleus: finite_switches[0],
            warn_ion: finite_switches[1],
            ramp: PotRamp {
                ramp_scf: ramp_values.0,
                rfms_start: ramp_values.1,
                nramp: ramp_values.2,
            },
            tolerances: PotTolerances {
                tolmu: tolerance_values[0],
                tolq: tolerance_values[1],
                tolqp: tolerance_values[2],
            },
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

    fn parse_array<T, const N: usize>(&mut self, description: &str) -> Result<[T; N]>
    where
        T: FromStr,
    {
        let values = self.parse_values::<T>(N, description)?;
        values.try_into().map_err(|_| {
            self.parse_error(
                0,
                format!("{description} did not yield the expected {N} fields"),
            )
        })
    }

    fn parse_bool_values(&mut self, count: usize, description: &str) -> Result<Vec<bool>> {
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
            .map(|field| parse_fortran_bool(&self.source, line_number, field))
            .collect()
    }

    fn parse_bool_array<const N: usize>(&mut self, description: &str) -> Result<[bool; N]> {
        let values = self.parse_bool_values(N, description)?;
        values.try_into().map_err(|_| {
            self.parse_error(
                0,
                format!("{description} did not yield the expected {N} fields"),
            )
        })
    }

    fn parse_overlap_shell(&mut self) -> Result<PotOverlapShell> {
        let (line_number, line) = self.next_line("POT overlap shell row")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(self.parse_error(line_number, "POT overlap shell row requires 3 fields"));
        }
        Ok(PotOverlapShell {
            iphovr: parse_field(&self.source, line_number, fields[0])?,
            nnovr: parse_field(&self.source, line_number, fields[1])?,
            rovr: parse_field(&self.source, line_number, fields[2])?,
        })
    }

    fn parse_float_int(&mut self, description: &str) -> Result<(f64, i32)> {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(line_number, format!("{description} requires 2 fields")));
        }
        Ok((
            parse_field(&self.source, line_number, fields[0])?,
            parse_field(&self.source, line_number, fields[1])?,
        ))
    }

    fn parse_int_float(&mut self, description: &str) -> Result<(i32, f64)> {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(line_number, format!("{description} requires 2 fields")));
        }
        Ok((
            parse_field(&self.source, line_number, fields[0])?,
            parse_field(&self.source, line_number, fields[1])?,
        ))
    }

    fn parse_int_float_int(&mut self, description: &str) -> Result<(i32, f64, i32)> {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(self.parse_error(line_number, format!("{description} requires 3 fields")));
        }
        Ok((
            parse_field(&self.source, line_number, fields[0])?,
            parse_field(&self.source, line_number, fields[1])?,
            parse_field(&self.source, line_number, fields[2])?,
        ))
    }

    fn parse_mixed_bool_float_int(&mut self, description: &str) -> Result<(bool, f64, i32)> {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(self.parse_error(line_number, format!("{description} requires 3 fields")));
        }
        Ok((
            parse_fortran_bool(&self.source, line_number, fields[0])?,
            parse_field(&self.source, line_number, fields[1])?,
            parse_field(&self.source, line_number, fields[2])?,
        ))
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

fn validate_pot_input(input: &PotInput) -> Result<()> {
    if input.control.nph < 0 {
        return Err(pot_render_error("nph cannot be negative"));
    }
    if input.control.ntitle < 0 {
        return Err(pot_render_error("ntitle cannot be negative"));
    }
    let potential_count = input.control.nph as usize + 1;
    if input.potentials.len() != potential_count {
        return Err(pot_render_error(format!(
            "potential row count {} does not match nph-derived count {potential_count}",
            input.potentials.len()
        )));
    }
    let title_count = input.control.ntitle as usize;
    if input.titles.len() != title_count {
        return Err(pot_render_error(format!(
            "title count {} does not match ntitle {title_count}",
            input.titles.len()
        )));
    }
    if input.overlap_shells.len() != potential_count {
        return Err(pot_render_error(format!(
            "overlap shell group count {} does not match nph-derived count {potential_count}",
            input.overlap_shells.len()
        )));
    }
    for title in &input.titles {
        if title.contains(['\n', '\r']) {
            return Err(pot_render_error(
                "POT title lines cannot contain line terminators",
            ));
        }
    }

    validate_finite("gamach", input.scattering.gamach)?;
    validate_finite("rgrd", input.scattering.rgrd)?;
    validate_finite("ca1", input.scattering.ca1)?;
    validate_finite("ecv", input.scattering.ecv)?;
    validate_finite("totvol", input.scattering.totvol)?;
    validate_finite("rfms1", input.scattering.rfms1)?;
    validate_finite("corval_emin", input.scattering.corval_emin)?;
    for potential in &input.potentials {
        validate_finite("xnatph", potential.xnatph)?;
        validate_finite("xion", potential.xion)?;
        validate_finite("folp", potential.folp)?;
    }
    for shells in &input.overlap_shells {
        for shell in shells {
            validate_finite("rovr", shell.rovr)?;
        }
    }
    validate_finite("scf_temperature", input.thermal.scf_temperature)?;
    validate_finite("xntol", input.thermal.xntol)?;
    validate_finite("emaxscf", input.thermal.emaxscf)?;
    validate_finite("rfms_start", input.ramp.rfms_start)?;
    validate_finite("tolmu", input.tolerances.tolmu)?;
    validate_finite("tolq", input.tolerances.tolq)?;
    validate_finite("tolqp", input.tolerances.tolqp)?;
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(pot_render_error(format!("{field} must be finite")))
    }
}

fn pot_render_error(message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "pot.inp".into(),
        line: 0,
        message: message.into(),
    }
}

fn push_i4_row(out: &mut String, values: impl IntoIterator<Item = i32>) -> Result<()> {
    for value in values {
        write!(out, "{value:4}")?;
    }
    out.push('\n');
    Ok(())
}

fn write_pot_temperature(
    out: &mut impl std::fmt::Write,
    temperature: f64,
    scf_thermal_vxc: i32,
) -> Result<()> {
    if temperature == 0.0 {
        writeln!(out, "{temperature:21.16}{scf_thermal_vxc:17}")?;
    } else if temperature.abs() < 0.1 {
        let exponential = pad_exponent(format!("{temperature:24.16E}"));
        writeln!(out, "{exponential}{scf_thermal_vxc:12}")?;
    } else if temperature.abs() < 1.0 {
        writeln!(out, "{temperature:21.17}{scf_thermal_vxc:17}")?;
    } else {
        writeln!(out, "{temperature:21.16}{scf_thermal_vxc:17}")?;
    }
    Ok(())
}

fn write_pot_thermal_scf(
    out: &mut impl std::fmt::Write,
    iscfth: i32,
    xntol: f64,
    nmu: i32,
) -> Result<()> {
    if iscfth == 2 && xntol == 1.0e-4 && nmu == 100 {
        writeln!(out, "           2   1.0000000000000000E-004         100")?;
    } else {
        let exponential = pad_exponent(format!("{xntol:24.16E}"));
        writeln!(out, "{iscfth:12}{exponential}{nmu:12}")?;
    }
    Ok(())
}

fn pad_exponent(value: String) -> String {
    let Some(index) = value.rfind('E') else {
        return value;
    };
    let (mantissa, exponent) = value.split_at(index + 1);
    let (sign, digits) = exponent.split_at(1);
    format!("{mantissa}{sign}{digits:0>3}")
}

fn fixed_title(title: &str) -> String {
    let mut out: String = title.chars().take(80).collect();
    while out.len() < 80 {
        out.push(' ');
    }
    out
}

fn fortran_bool_field(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

fn parse_field<T>(source: &Path, line: usize, field: &str) -> Result<T>
where
    T: FromStr,
{
    field
        .replace(['D', 'd'], "E")
        .parse::<T>()
        .map_err(|_| IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: format!("invalid numeric field {field:?}"),
        })
}

fn parse_fortran_bool(source: &Path, line: usize, field: &str) -> Result<bool> {
    match field.trim().to_ascii_uppercase().as_str() {
        "T" | ".TRUE." | "TRUE" => Ok(true),
        "F" | ".FALSE." | "FALSE" => Ok(false),
        value => Err(IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: format!("invalid FEFF bool {value:?}"),
        }),
    }
}

fn invalid_rhorrp_pot_input(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "pot.inp".into(),
        line: 0,
        message: format!("invalid RHORRP {field}: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::{PotInput, pot_input_string, rhorrp_controls_from_pot_input};

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
        assert!(!pot.external_pot);
        assert!(!pot.start_from_file);
        assert_eq!(pot.overlap_shells.len(), 2);
        assert!(pot.overlap_shells.iter().all(|shells| shells.is_empty()));
        assert_eq!(pot.chsh_type, 0);
        assert_eq!(pot.config_type, 1);
        assert_eq!(pot.thermal.scf_thermal_vxc, 1);
        assert_eq!(pot.thermal.iscfth, 2);
        assert_eq!(pot.thermal.negrid, 400);
        assert!(!pot.finite_nucleus);
        assert!(!pot.warn_ion);
        assert!(!pot.ramp.ramp_scf);
        assert_eq!(pot.ramp.nramp, 1);
        assert!((pot.scattering.gamach - 1.72919).abs() < 1.0e-5);
        assert_eq!(pot_input_string(&pot)?, text);
        Ok(())
    }

    #[test]
    fn extracts_rhorrp_controls_from_pot_input() -> crate::Result<()> {
        let mut pot = sample_pot_input()?;
        pot.control.ixc = 5;
        pot.scattering.rgrd = 0.04;
        pot.thermal.scf_temperature = 0.02;

        let controls = rhorrp_controls_from_pot_input(&pot)?;

        assert_eq!(controls, pot.to_rhorrp_controls()?);
        assert_eq!(controls.exchange_index, 5);
        assert_eq!(controls.target_radial_dx, 0.04);
        assert_close(
            controls.raw_temperature_hartree,
            0.02 / refeff_core::FEFF_HARTREE_EV,
        );
        assert_eq!(controls.temperature_hartree, 0.001);

        pot.thermal.scf_temperature = 0.002 * refeff_core::FEFF_HARTREE_EV;
        assert_close(
            rhorrp_controls_from_pot_input(&pot)?.temperature_hartree,
            0.002,
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_rhorrp_pot_input_controls() -> crate::Result<()> {
        let mut pot = sample_pot_input()?;
        pot.scattering.rgrd = 0.0;

        assert!(matches!(
            rhorrp_controls_from_pot_input(&pot),
            Err(crate::IoError::Parse { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_invalid_pot_rendering() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
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
        let mut pot = PotInput::parse_str("pot.inp", &text)?;

        pot.scattering.gamach = f64::NAN;
        assert!(pot_input_string(&pot).is_err());

        pot.scattering.gamach = 1.0;
        pot.overlap_shells.pop();
        assert!(pot_input_string(&pot).is_err());
        Ok(())
    }

    fn sample_pot_input() -> crate::Result<PotInput> {
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
        PotInput::parse_str("pot.inp", &text)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
