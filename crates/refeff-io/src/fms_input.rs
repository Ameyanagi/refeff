//! Typed reader for FEFF `fms.inp` module handoff files.
//!
//! The FMS module consumes cluster, Debye-Waller, and angular-momentum
//! controls from `rdinp`. This reader preserves that boundary for the Rust
//! full multiple-scattering implementation.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use refeff_core::{RhorrpFmsInclusionInput, rhorrp_fms_inclusion_counts};

use crate::control_input::FEFF_BOHR_ANGSTROM;
use crate::structure_output::RhorrpGeomHandoff;
use crate::{IoError, Result};

/// Parsed contents of a FEFF `fms.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsInput {
    /// FMS run-control switches.
    pub control: FmsControl,
    /// FMS cluster and convergence settings.
    pub cluster: FmsCluster,
    /// Temperature and Debye-Waller settings.
    pub debye: FmsDebye,
    /// Angular-momentum cutoffs indexed by potential.
    pub lmaxph: Vec<i32>,
    /// NRIXS decomposition-channel count.
    pub decomposition_channels: i32,
    /// Whether to save a scattering-matrix slice.
    pub save_gg_slice: bool,
    /// Whether the FMS algorithm should contribute to the spectrum.
    pub do_fms: i32,
}

/// First integer control line of `fms.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmsControl {
    pub mfms: i32,
    pub idwopt: i32,
    pub minv: i32,
}

/// Cluster and convergence controls from `fms.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsCluster {
    pub rfms2: f64,
    pub rdirec: f64,
    pub toler1: f64,
    pub toler2: f64,
}

/// Temperature and Debye-Waller controls from `fms.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsDebye {
    pub tk: f64,
    pub thetad: f64,
    pub sig2g: f64,
}

/// RHORRP FMS controls imported from FEFF `fms_inp`.
///
/// RHORRP converts `rfms2` from Angstrom to Bohr before `init_inclus`, and uses
/// `lmaxph` to choose the wavefunction angular table width. FEFF's RHORRP
/// source notes that saved `gg_slice` indexing currently assumes a single
/// shared `lmaxph` across active potentials; this handoff enforces that
/// constraint before callers build `init_wavefunctions` tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpFmsInputHandoff {
    /// FEFF `rfms2` converted from Angstrom to Bohr for `init_inclus`.
    pub fms_radius_bohr: f64,
    /// Number of potential entries checked from `lmaxph`.
    pub potential_count: usize,
    /// Shared FEFF `lmaxph` value.
    pub max_angular_momentum: usize,
    /// Ordinary angular momentum channel count, `lmaxph + 1`.
    pub angular_momentum_count: usize,
}

impl RhorrpFmsInputHandoff {
    /// Compute FEFF `init_inclus` counts for this FMS setup and RHORRP geometry.
    pub fn fms_inclusion_counts(&self, geometry: &RhorrpGeomHandoff) -> Result<Vec<usize>> {
        if geometry.potential_count() != self.potential_count {
            return Err(invalid_rhorrp_fms_input(
                "rfms2",
                format!(
                    "geometry has {} potential representative(s), expected {}",
                    geometry.potential_count(),
                    self.potential_count
                ),
            ));
        }
        rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
            atom_positions: geometry.atom_positions_bohr.view(),
            representative_atoms: &geometry.representative_atoms,
            fms_radius: self.fms_radius_bohr,
        })
        .map_err(|source| invalid_rhorrp_fms_input("rfms2", source.to_string()))
    }

    /// FEFF `inclus(0)`, the central-potential atom count used by `nearest_atom(..., fmsF=.true.)`.
    pub fn central_fms_atom_count(&self, geometry: &RhorrpGeomHandoff) -> Result<usize> {
        self.fms_inclusion_counts(geometry)?
            .into_iter()
            .next()
            .ok_or_else(|| invalid_rhorrp_fms_input("rfms2", "missing central inclusion count"))
    }
}

impl FmsInput {
    /// Parse a FEFF `fms.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = FmsInputParser::new(source.into(), text);
        parser.parse()
    }

    /// Extract RHORRP angular-momentum controls from this parsed `fms.inp`.
    pub fn to_rhorrp_handoff(&self, potential_count: usize) -> Result<RhorrpFmsInputHandoff> {
        rhorrp_handoff_from_fms_input(self, potential_count)
    }
}

/// Build RHORRP FMS controls from parsed FEFF `fms.inp`.
pub fn rhorrp_handoff_from_fms_input(
    input: &FmsInput,
    potential_count: usize,
) -> Result<RhorrpFmsInputHandoff> {
    validate_fms_input(input)?;
    if potential_count == 0 {
        return Err(invalid_rhorrp_fms_input(
            "lmaxph",
            "RHORRP requires at least one potential",
        ));
    }
    if input.lmaxph.len() < potential_count {
        return Err(invalid_rhorrp_fms_input(
            "lmaxph",
            format!(
                "fms.inp has {} lmaxph value(s), expected at least {potential_count}",
                input.lmaxph.len()
            ),
        ));
    }

    let first = rhorrp_lmax(input.lmaxph[0], 0)?;
    for potential in 1..potential_count {
        let lmax = rhorrp_lmax(input.lmaxph[potential], potential)?;
        if lmax != first {
            return Err(invalid_rhorrp_fms_input(
                "lmaxph",
                format!(
                    "RHORRP requires uniform lmaxph values, got lmaxph(0)={first} and lmaxph({potential})={lmax}"
                ),
            ));
        }
    }
    let angular_momentum_count = first
        .checked_add(1)
        .ok_or_else(|| invalid_rhorrp_fms_input("lmaxph", "angular channel count overflowed"))?;
    let fms_radius_bohr = angstrom_to_bohr("rfms2", input.cluster.rfms2)?;

    Ok(RhorrpFmsInputHandoff {
        fms_radius_bohr,
        potential_count,
        max_angular_momentum: first,
        angular_momentum_count,
    })
}

/// Render FEFF-compatible `fms.inp` text.
pub fn fms_input_string(input: &FmsInput) -> Result<String> {
    validate_fms_input(input)?;

    let mut out = String::new();
    writeln!(out, "mfms, idwopt, minv")?;
    push_i4_row(
        &mut out,
        [input.control.mfms, input.control.idwopt, input.control.minv],
    )?;
    writeln!(out, "rfms2, rdirec, toler1, toler2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        input.cluster.rfms2, input.cluster.rdirec, input.cluster.toler1, input.cluster.toler2
    )?;
    writeln!(out, "tk, thetad, sig2g")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.debye.tk, input.debye.thetad, input.debye.sig2g
    )?;
    writeln!(out, " lmaxph(0:nph)")?;
    push_i4_row(&mut out, input.lmaxph.iter().copied())?;
    writeln!(out, " the number of decomposi")?;
    writeln!(out, "{:5}", input.decomposition_channels)?;
    writeln!(out, " save_gg_slice")?;
    writeln!(out, "{}", fortran_bool_field(input.save_gg_slice))?;
    writeln!(out, "do_fms")?;
    push_i4_row(&mut out, [input.do_fms])?;
    Ok(out)
}

fn validate_fms_input(input: &FmsInput) -> Result<()> {
    validate_finite("rfms2", input.cluster.rfms2)?;
    validate_finite("rdirec", input.cluster.rdirec)?;
    validate_finite("toler1", input.cluster.toler1)?;
    validate_finite("toler2", input.cluster.toler2)?;
    validate_finite("tk", input.debye.tk)?;
    validate_finite("thetad", input.debye.thetad)?;
    validate_finite("sig2g", input.debye.sig2g)?;
    if input.lmaxph.is_empty() {
        return Err(IoError::Parse {
            path: "fms.inp".into(),
            line: 0,
            message: "FMS input requires at least one lmaxph value".to_string(),
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "fms.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn angstrom_to_bohr(field: &'static str, value: f64) -> Result<f64> {
    validate_finite(field, value)?;
    let converted = value / FEFF_BOHR_ANGSTROM;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(invalid_rhorrp_fms_input(
            field,
            "conversion to Bohr produced a non-finite value",
        ))
    }
}

fn rhorrp_lmax(value: i32, potential: usize) -> Result<usize> {
    if value < 0 {
        return Err(invalid_rhorrp_fms_input(
            "lmaxph",
            format!("lmaxph({potential}) must be nonnegative, got {value}"),
        ));
    }
    usize::try_from(value).map_err(|_| {
        invalid_rhorrp_fms_input(
            "lmaxph",
            format!("lmaxph({potential}) does not fit in usize"),
        )
    })
}

fn invalid_rhorrp_fms_input(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "fms.inp".into(),
        line: 0,
        message: format!("invalid RHORRP {field}: {}", message.into()),
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

struct FmsInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> FmsInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<FmsInput> {
        self.expect_header("mfms, idwopt, minv")?;
        let control_values = self.parse_values::<i32>(3, "FMS control line")?;
        let control = FmsControl {
            mfms: control_values[0],
            idwopt: control_values[1],
            minv: control_values[2],
        };

        self.expect_header("rfms2, rdirec, toler1, toler2")?;
        let cluster_values = self.parse_values::<f64>(4, "FMS cluster line")?;
        let cluster = FmsCluster {
            rfms2: cluster_values[0],
            rdirec: cluster_values[1],
            toler1: cluster_values[2],
            toler2: cluster_values[3],
        };

        self.expect_header("tk, thetad, sig2g")?;
        let debye_values = self.parse_values::<f64>(3, "FMS Debye line")?;
        let debye = FmsDebye {
            tk: debye_values[0],
            thetad: debye_values[1],
            sig2g: debye_values[2],
        };

        self.expect_header("lmaxph(0:nph)")?;
        let lmaxph = self.parse_lmaxph()?;

        self.expect_header("the number of decomposi")?;
        let decomposition_channels = self.parse_values::<i32>(1, "FMS decomposition line")?[0];
        let (save_gg_slice, do_fms) = if self.next_optional_header("save_gg_slice")? {
            let save_gg_slice = self.parse_bool_line("FMS save_gg_slice line")?;
            self.expect_header("do_fms")?;
            let do_fms = self.parse_values::<i32>(1, "FMS do_fms line")?[0];
            (save_gg_slice, do_fms)
        } else {
            (false, control.mfms)
        };

        Ok(FmsInput {
            control,
            cluster,
            debye,
            lmaxph,
            decomposition_channels,
            save_gg_slice,
            do_fms,
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

    fn next_optional_header(&mut self, expected: &str) -> Result<bool> {
        let Some((index, line)) = self.lines.next() else {
            return Ok(false);
        };
        let line_number = index + 1;
        if line.trim() == expected {
            Ok(true)
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

    fn parse_lmaxph(&mut self) -> Result<Vec<i32>> {
        let (line_number, line) = self.next_line("FMS lmaxph line")?;
        line.split_whitespace()
            .map(|field| parse_field(&self.source, line_number, field))
            .collect()
    }

    fn parse_bool_line(&mut self, description: &str) -> Result<bool> {
        let (line_number, line) = self.next_line(description)?;
        parse_fortran_bool(&self.source, line_number, line.trim())
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

#[cfg(test)]
mod tests {
    use crate::{FEFF_BOHR_ANGSTROM, FeffDocument, FeffInput, GeomDat, GeomDatRow, rdinp};

    use super::{FmsInput, fms_input_string, rhorrp_handoff_from_fms_input};

    #[test]
    fn parses_generated_copper_fms_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FMS 7.0 1 2 0.002 0.003 10.0
DEBYE 300.0 400.0
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
        let text = rdinp::fms_inp_string(&document)?;
        let fms = FmsInput::parse_str("fms.inp", &text)?;

        assert_eq!(fms.control.mfms, 1);
        assert_eq!(fms.control.idwopt, 0);
        assert_eq!(fms.control.minv, 2);
        assert_eq!(fms.cluster.rfms2, 7.0);
        assert_eq!(fms.cluster.rdirec, 10.0);
        assert_eq!(fms.cluster.toler1, 0.002);
        assert_eq!(fms.cluster.toler2, 0.003);
        assert_eq!(fms.debye.tk, 300.0);
        assert_eq!(fms.debye.thetad, 400.0);
        assert_eq!(fms.lmaxph, [3, 3]);
        assert_eq!(fms.decomposition_channels, -1);
        assert!(!fms.save_gg_slice);
        assert_eq!(fms.do_fms, 1);
        Ok(())
    }

    #[test]
    fn parses_legacy_fms_input_without_save_slice_tail() -> crate::Result<()> {
        let text = concat!(
            "mfms, idwopt, minv\n",
            "   1  -1   0\n",
            "rfms2, rdirec, toler1, toler2\n",
            "      6.00000     12.00000      0.00100      0.00100\n",
            "tk, thetad, sig2g\n",
            "      0.00000      0.00000      0.00000\n",
            " lmaxph(0:nph)\n",
            "   2   2   2\n",
            " the number of decomposi\n",
            "   -1\n",
        );

        let fms = FmsInput::parse_str("legacy-fms.inp", text)?;

        assert_eq!(fms.control.mfms, 1);
        assert_eq!(fms.control.idwopt, -1);
        assert_eq!(fms.control.minv, 0);
        assert_eq!(fms.lmaxph, [2, 2, 2]);
        assert_eq!(fms.decomposition_channels, -1);
        assert!(!fms.save_gg_slice);
        assert_eq!(fms.do_fms, 1);
        Ok(())
    }

    #[test]
    fn renders_generated_copper_fms_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FMS 7.0 1 2 0.002 0.003 10.0
DEBYE 300.0 400.0
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
        let text = rdinp::fms_inp_string(&document)?;
        let fms = FmsInput::parse_str("fms.inp", &text)?;

        assert_eq!(fms_input_string(&fms)?, text);
        Ok(())
    }

    #[test]
    fn extracts_rhorrp_handoff_from_fms_input() -> crate::Result<()> {
        let mut fms = sample_fms_input()?;
        fms.cluster.rfms2 = 0.75 * FEFF_BOHR_ANGSTROM;
        fms.lmaxph = vec![2, 2, 2];

        let handoff = rhorrp_handoff_from_fms_input(&fms, 3)?;

        assert_eq!(handoff, fms.to_rhorrp_handoff(3)?);
        assert_close(handoff.fms_radius_bohr, 0.75);
        assert_eq!(handoff.potential_count, 3);
        assert_eq!(handoff.max_angular_momentum, 2);
        assert_eq!(handoff.angular_momentum_count, 3);
        assert_eq!(
            handoff.fms_inclusion_counts(&sample_rhorrp_geom_handoff()?)?,
            vec![2, 2, 1]
        );
        assert_eq!(
            handoff.central_fms_atom_count(&sample_rhorrp_geom_handoff()?)?,
            2
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_rhorrp_fms_handoff_inputs() -> crate::Result<()> {
        let mut fms = sample_fms_input()?;

        assert!(matches!(
            rhorrp_handoff_from_fms_input(&fms, 0),
            Err(crate::IoError::Parse { .. })
        ));

        fms.lmaxph = vec![2];
        assert!(matches!(
            rhorrp_handoff_from_fms_input(&fms, 2),
            Err(crate::IoError::Parse { .. })
        ));

        fms.lmaxph = vec![2, 3];
        assert!(matches!(
            rhorrp_handoff_from_fms_input(&fms, 2),
            Err(crate::IoError::Parse { .. })
        ));

        fms.lmaxph = vec![2, -1];
        assert!(matches!(
            rhorrp_handoff_from_fms_input(&fms, 2),
            Err(crate::IoError::Parse { .. })
        ));

        fms = sample_fms_input()?;
        fms.lmaxph = vec![2, 2];
        let handoff = rhorrp_handoff_from_fms_input(&fms, 2)?;
        assert!(matches!(
            handoff.fms_inclusion_counts(&sample_rhorrp_geom_handoff()?),
            Err(crate::IoError::Parse { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_invalid_fms_rendering() {
        let input = FmsInput {
            control: super::FmsControl {
                mfms: 1,
                idwopt: -1,
                minv: 0,
            },
            cluster: super::FmsCluster {
                rfms2: f64::NAN,
                rdirec: 10.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: super::FmsDebye {
                tk: 0.0,
                thetad: 0.0,
                sig2g: 0.0,
            },
            lmaxph: vec![3],
            decomposition_channels: -1,
            save_gg_slice: false,
            do_fms: 1,
        };
        assert!(fms_input_string(&input).is_err());

        let mut empty_lmax = input;
        empty_lmax.cluster.rfms2 = 7.0;
        empty_lmax.lmaxph.clear();
        assert!(fms_input_string(&empty_lmax).is_err());
    }

    fn sample_fms_input() -> crate::Result<FmsInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FMS 7.0 1 2 0.002 0.003 10.0
DEBYE 300.0 400.0
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
        let text = rdinp::fms_inp_string(&document)?;
        FmsInput::parse_str("fms.inp", &text)
    }

    fn sample_rhorrp_geom_handoff() -> crate::Result<crate::RhorrpGeomHandoff> {
        GeomDat {
            nat: 4,
            nph: 2,
            model_atoms: vec![1, 2, 4],
            atoms: vec![
                GeomDatRow {
                    index: 1,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    iph: 0,
                    boundary: 1,
                },
                GeomDatRow {
                    index: 2,
                    x: FEFF_BOHR_ANGSTROM,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 1,
                },
                GeomDatRow {
                    index: 3,
                    x: 0.4 * FEFF_BOHR_ANGSTROM,
                    y: 0.0,
                    z: 0.0,
                    iph: 0,
                    boundary: 1,
                },
                GeomDatRow {
                    index: 4,
                    x: 0.0,
                    y: 2.0 * FEFF_BOHR_ANGSTROM,
                    z: 0.0,
                    iph: 2,
                    boundary: 1,
                },
            ],
        }
        .to_rhorrp_handoff()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
