//! Typed reader for FEFF `global.inp` module handoff files.
//!
//! `global.inp` carries shared polarization, spin, CFAVERAGE, and NRIXS
//! controls. Several FEFF modules read this file, so keeping it typed gives
//! the Rust port one common source of truth for those cross-module settings.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `global.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalInput {
    /// Configuration-average controls from the `CFAVERAGE` bridge line.
    pub cfaverage: CfAverage,
    /// Shared polarization, spin, and NRIXS switches.
    pub control: GlobalControl,
    /// Electric-field vector, stored as FEFF writes the row-wise table.
    pub evec: [f64; 3],
    /// Incidence, beam, or momentum-transfer vector depending on spectroscopy.
    pub xivec: [f64; 3],
    /// Spin vector.
    pub spvec: [f64; 3],
    /// FEFF polarization tensor rows for magnetic quantum numbers -1, 0, 1.
    pub polarization_tensor: [[f64; 6]; 3],
    /// Vector norms used by NRIXS.
    pub norms: GlobalNorms,
    /// NRIXS q-list controls.
    pub q_control: GlobalQControl,
    /// NRIXS q-vectors. Empty for non-NRIXS inputs with `nq == 0`.
    pub q_vectors: Vec<GlobalQVector>,
    /// Optional MDFF pair controls written when `mixdff` is enabled.
    pub mdff: Option<GlobalMdff>,
}

/// Configuration-average controls from `global.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfAverage {
    /// Number of absorbers included in the configuration average.
    pub nabs: i32,
    /// Potential index used as the absorbing atom type.
    pub iphabs: i32,
    /// Cluster radius used for the per-absorber reduced geometry list.
    pub rclabs: f64,
}

/// Main `global.inp` control line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalControl {
    pub ipol: i32,
    pub ispin: i32,
    pub le2: i32,
    pub elpty: f64,
    pub angks: f64,
    pub l2lp: i32,
    pub do_nrixs: i32,
    pub ldecmx: i32,
    pub lj: i32,
}

/// Vector norms written after the polarization tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalNorms {
    pub evnorm: f64,
    pub xivnorm: f64,
    pub spvnorm: f64,
}

/// NRIXS q-list controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalQControl {
    pub nq: i32,
    pub imdff: i32,
    pub qaverage: bool,
    pub mixdff: bool,
}

/// One NRIXS q-vector row from `global.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalQVector {
    pub q: [f64; 3],
    pub norm: f64,
    /// Complex q weight as written by FEFF's real-valued table.
    pub weight: [f64; 2],
    /// Rotation helper values: costh, sinth, cosfi, sinfi.
    pub trig: [f64; 4],
}

/// MDFF pair-control payload from `global.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalMdff {
    /// Generated q-prime norm. FEFF uses -1 to reuse the input q-list.
    pub qqmdff: f64,
    /// Flattened row-major `cos<q,q'>` matrix.
    pub cosines: Vec<f64>,
}

impl GlobalInput {
    /// Parse a FEFF `global.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = GlobalInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `global.inp` text.
pub fn global_input_string(input: &GlobalInput) -> Result<String> {
    validate_global_input(input)?;

    let mut out = String::new();
    writeln!(out, " nabs, iphabs - CFAVERAGE data")?;
    writeln!(
        out,
        "{:8}{:8}{:13.5}",
        input.cfaverage.nabs, input.cfaverage.iphabs, input.cfaverage.rclabs
    )?;
    writeln!(
        out,
        " ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj"
    )?;
    writeln!(
        out,
        "{:5}{:5}{:5}{:12.4}{:12.4}{:5}{:5}{:5}{:5}",
        input.control.ipol,
        input.control.ispin,
        input.control.le2,
        input.control.elpty,
        input.control.angks,
        input.control.l2lp,
        input.control.do_nrixs,
        input.control.ldecmx,
        input.control.lj
    )?;
    writeln!(out, "evec\t\t  xivec \t   spvec")?;
    for idx in 0..3 {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}",
            input.evec[idx], input.xivec[idx], input.spvec[idx]
        )?;
    }
    writeln!(out, " polarization tensor ")?;
    for row in input.polarization_tensor {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
            row[0], row[1], row[2], row[3], row[4], row[5]
        )?;
    }
    writeln!(out, "evnorm, xivnorm, spvnorm - only used for nrixs")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.norms.evnorm, input.norms.xivnorm, input.norms.spvnorm
    )?;
    writeln!(out, "nq,    imdff,   qaverage,   mixdff")?;
    writeln!(
        out,
        "{:12}{:12} {} {}",
        input.q_control.nq,
        input.q_control.imdff,
        fortran_bool_field(input.q_control.qaverage),
        fortran_bool_field(input.q_control.mixdff)
    )?;
    writeln!(
        out,
        " q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi"
    )?;
    for vector in &input.q_vectors {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
            vector.q[0],
            vector.q[1],
            vector.q[2],
            vector.norm,
            vector.weight[0],
            vector.weight[1],
            vector.trig[0],
            vector.trig[1],
            vector.trig[2],
            vector.trig[3]
        )?;
    }
    if let Some(mdff) = &input.mdff {
        writeln!(out, "    qqmdff,   cos<q,q'>")?;
        write!(out, "{:22.16}", mdff.qqmdff)?;
        for value in &mdff.cosines {
            write!(out, "{value:22.16}")?;
        }
        out.push('\n');
    }
    Ok(out)
}

fn validate_global_input(input: &GlobalInput) -> Result<()> {
    validate_finite("rclabs", input.cfaverage.rclabs)?;
    validate_finite("elpty", input.control.elpty)?;
    validate_finite("angks", input.control.angks)?;
    validate_finite("evnorm", input.norms.evnorm)?;
    validate_finite("xivnorm", input.norms.xivnorm)?;
    validate_finite("spvnorm", input.norms.spvnorm)?;
    for (name, values) in [
        ("evec", input.evec),
        ("xivec", input.xivec),
        ("spvec", input.spvec),
    ] {
        for value in values {
            validate_finite(name, value)?;
        }
    }
    for row in input.polarization_tensor {
        for value in row {
            validate_finite("polarization_tensor", value)?;
        }
    }
    let expected_q_vectors = input.q_control.nq.max(0) as usize;
    if input.q_vectors.len() != expected_q_vectors {
        return Err(IoError::Parse {
            path: "global.inp".into(),
            line: 0,
            message: format!(
                "global q-vector count {} does not match nq {expected_q_vectors}",
                input.q_vectors.len()
            ),
        });
    }
    for vector in &input.q_vectors {
        for value in vector.q {
            validate_finite("q_vector", value)?;
        }
        validate_finite("q_norm", vector.norm)?;
        for value in vector.weight {
            validate_finite("q_weight", value)?;
        }
        for value in vector.trig {
            validate_finite("q_trig", value)?;
        }
    }
    match (&input.mdff, input.q_control.mixdff) {
        (Some(mdff), true) => {
            validate_finite("qqmdff", mdff.qqmdff)?;
            let expected = expected_q_vectors.saturating_mul(expected_q_vectors);
            if mdff.cosines.len() != expected {
                return Err(IoError::Parse {
                    path: "global.inp".into(),
                    line: 0,
                    message: format!(
                        "MDFF cosine count {} does not match nq-derived count {expected}",
                        mdff.cosines.len()
                    ),
                });
            }
            for value in &mdff.cosines {
                validate_finite("mdff_cosine", *value)?;
            }
        }
        (None, true) => {
            return Err(IoError::Parse {
                path: "global.inp".into(),
                line: 0,
                message: "global.inp mixdff requires MDFF pair data".to_string(),
            });
        }
        (Some(_), false) => {
            return Err(IoError::Parse {
                path: "global.inp".into(),
                line: 0,
                message: "global.inp MDFF pair data requires mixdff".to_string(),
            });
        }
        (None, false) => {}
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "global.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn fortran_bool_field(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

struct GlobalInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlobalQHeader {
    Standard,
    InlineMdff,
}

impl<'a> GlobalInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<GlobalInput> {
        self.expect_header("nabs, iphabs - CFAVERAGE data")?;
        let cfaverage_values = self.parse_values::<f64>(3, "CFAVERAGE data line")?;
        let cfaverage = CfAverage {
            nabs: cfaverage_values[0] as i32,
            iphabs: cfaverage_values[1] as i32,
            rclabs: cfaverage_values[2],
        };

        self.expect_header("ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj")?;
        let control_values = self.parse_values::<f64>(9, "global control line")?;
        let control = GlobalControl {
            ipol: control_values[0] as i32,
            ispin: control_values[1] as i32,
            le2: control_values[2] as i32,
            elpty: control_values[3],
            angks: control_values[4],
            l2lp: control_values[5] as i32,
            do_nrixs: control_values[6] as i32,
            ldecmx: control_values[7] as i32,
            lj: control_values[8] as i32,
        };

        self.expect_header("evec\t\t  xivec \t   spvec")?;
        let vector_rows = self.parse_f64_rows::<3, 3>("global vector row")?;
        let evec = [vector_rows[0][0], vector_rows[1][0], vector_rows[2][0]];
        let xivec = [vector_rows[0][1], vector_rows[1][1], vector_rows[2][1]];
        let spvec = [vector_rows[0][2], vector_rows[1][2], vector_rows[2][2]];

        self.expect_header("polarization tensor")?;
        let polarization_tensor = self.parse_f64_rows::<3, 6>("polarization tensor row")?;

        self.expect_header("evnorm, xivnorm, spvnorm - only used for nrixs")?;
        let norm_values = self.parse_values::<f64>(3, "global norm line")?;
        let norms = GlobalNorms {
            evnorm: norm_values[0],
            xivnorm: norm_values[1],
            spvnorm: norm_values[2],
        };

        let Some((line_number, line)) = self.lines.next().map(|(index, line)| (index + 1, line))
        else {
            return Ok(GlobalInput {
                cfaverage,
                control,
                evec,
                xivec,
                spvec,
                polarization_tensor,
                norms,
                q_control: GlobalQControl {
                    nq: 0,
                    imdff: 0,
                    qaverage: true,
                    mixdff: false,
                },
                q_vectors: Vec::new(),
                mdff: None,
            });
        };
        let q_header = self.parse_q_control_header(line_number, line)?;
        let (q_control, inline_mdff_values) = self.parse_q_control(q_header)?;

        self.expect_header(
            "q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi",
        )?;
        let q_count = q_control.nq.max(0) as usize;
        let mut q_vectors = Vec::with_capacity(q_count);
        for _ in 0..q_count {
            let values = self.parse_values::<f64>(10, "global q-vector line")?;
            q_vectors.push(GlobalQVector {
                q: [values[0], values[1], values[2]],
                norm: values[3],
                weight: [values[4], values[5]],
                trig: [values[6], values[7], values[8], values[9]],
            });
        }
        let mdff = if q_control.mixdff {
            Some(self.parse_mdff(q_count, inline_mdff_values)?)
        } else {
            None
        };

        Ok(GlobalInput {
            cfaverage,
            control,
            evec,
            xivec,
            spvec,
            polarization_tensor,
            norms,
            q_control,
            q_vectors,
            mdff,
        })
    }

    fn expect_header(&mut self, expected: &str) -> Result<()> {
        let (line_number, line) = self.next_line(expected)?;
        self.expect_header_at(line_number, line, expected)
    }

    fn expect_header_at(&self, line_number: usize, line: &str, expected: &str) -> Result<()> {
        if line.trim() == expected {
            Ok(())
        } else {
            Err(self.parse_error(
                line_number,
                format!("expected header {expected:?}, found {line:?}"),
            ))
        }
    }

    fn parse_q_control_header(&self, line_number: usize, line: &str) -> Result<GlobalQHeader> {
        match line.trim() {
            "nq,    imdff,   qaverage,   mixdff" => Ok(GlobalQHeader::Standard),
            "nq,    imdff,   qaverage,   mixdff,   qqmdff,   cos<q,q'>" => {
                Ok(GlobalQHeader::InlineMdff)
            }
            _ => Err(self.parse_error(
                line_number,
                format!(
                    "expected header {:?}, found {line:?}",
                    "nq,    imdff,   qaverage,   mixdff"
                ),
            )),
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

    fn parse_f64_rows<const ROWS: usize, const COLS: usize>(
        &mut self,
        description: &str,
    ) -> Result<[[f64; COLS]; ROWS]> {
        let mut rows = [[0.0; COLS]; ROWS];
        for row in rows.iter_mut() {
            let values = self.parse_values::<f64>(COLS, description)?;
            for (target, value) in row.iter_mut().zip(values) {
                *target = value;
            }
        }
        Ok(rows)
    }

    fn parse_q_control(&mut self, header: GlobalQHeader) -> Result<(GlobalQControl, Vec<f64>)> {
        let (line_number, line) = self.next_line("global q-control line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(self.parse_error(line_number, "global q-control line requires 4 fields"));
        }
        let q_control = GlobalQControl {
            nq: parse_field(&self.source, line_number, fields[0])?,
            imdff: parse_field(&self.source, line_number, fields[1])?,
            qaverage: parse_fortran_bool(&self.source, line_number, fields[2])?,
            mixdff: parse_fortran_bool(&self.source, line_number, fields[3])?,
        };
        let inline_mdff_values = if matches!(header, GlobalQHeader::InlineMdff) {
            fields
                .iter()
                .skip(4)
                .map(|field| parse_field(&self.source, line_number, field))
                .collect::<Result<Vec<f64>>>()?
        } else {
            Vec::new()
        };
        Ok((q_control, inline_mdff_values))
    }

    fn parse_mdff(&mut self, q_count: usize, mut values: Vec<f64>) -> Result<GlobalMdff> {
        let expected = 1 + q_count.saturating_mul(q_count);
        if values.is_empty() {
            self.expect_header("qqmdff,   cos<q,q'>")?;
        }
        values.reserve(expected.saturating_sub(values.len()));
        while values.len() < expected {
            let (line_number, line) = self.next_line("MDFF data line")?;
            for field in line.split_whitespace() {
                values.push(parse_field(&self.source, line_number, field)?);
                if values.len() == expected {
                    break;
                }
            }
        }
        let qqmdff = values[0];
        Ok(GlobalMdff {
            qqmdff,
            cosines: values.into_iter().skip(1).collect(),
        })
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
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::{GlobalInput, global_input_string};

    #[test]
    fn parses_generated_default_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::global_inp_string(&document)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.cfaverage.nabs, 1);
        assert_eq!(global.cfaverage.iphabs, 0);
        assert_eq!(global.control.ipol, 0);
        assert_eq!(global.control.do_nrixs, 0);
        assert_eq!(global.evec, [0.0, 0.0, 0.0]);
        assert!((global.polarization_tensor[0][0] - 1.0 / 3.0).abs() < 1.0e-5);
        assert_eq!(global.q_control.nq, 0);
        assert!(global.q_control.qaverage);
        assert!(global.q_vectors.is_empty());
        assert_eq!(global_input_string(&global)?, text);
        Ok(())
    }

    #[test]
    fn parses_generated_nrixs_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 1 1.0 2.0 -3.0
LDEC 4
LJMAX 2
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::global_inp_string(&document)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.control.ipol, 1);
        assert_eq!(global.control.do_nrixs, 1);
        assert_eq!(global.control.ldecmx, 4);
        assert_eq!(global.control.lj, 2);
        assert_eq!(global.control.l2lp, 30);
        assert_eq!(global.q_control.nq, 1);
        assert!(!global.q_control.qaverage);
        assert_eq!(global.q_vectors.len(), 1);

        let q_vector = global.q_vectors[0];
        assert_eq!(q_vector.q, [1.0, 2.0, -3.0]);
        assert!((q_vector.norm - 3.74166).abs() < 1.0e-5);
        assert_eq!(q_vector.weight, [1.0, 0.0]);
        assert!((q_vector.trig[0] + 0.80178).abs() < 1.0e-5);
        assert!((q_vector.trig[1] - 0.59761).abs() < 1.0e-5);
        assert_eq!(global_input_string(&global)?, text);
        Ok(())
    }

    #[test]
    fn parses_legacy_nrixs_global_input_with_inline_mdff_header() -> crate::Result<()> {
        let text = "\
 nabs, iphabs - CFAVERAGE data
       1       0 100000.00000
 ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj
    1    0   10      1.0000      0.0000    0    1    2   10
evec		  xivec 	   spvec
      0.70711      0.00000      0.00000
      0.70711      0.00000      0.00000
      0.00000      1.00000      0.00000
 polarization tensor
      0.50000      0.00000      0.00000      0.00000      0.00000      0.00000
      0.00000      0.00000      0.00000      0.00000      0.00000      0.00000
      0.00000      0.00000      0.00000      0.00000      0.50000      0.00000
evnorm, xivnorm, spvnorm - only used for nrixs
      0.00000     24.00000      0.00000
nq,    imdff,   qaverage,   mixdff,   qqmdff,   cos<q,q'>
           1           0 T F  -1.00000000000000        1.00000000000000
 q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi
      0.00000      0.00000     24.00000     24.00000      1.00000      0.00000     -1.00000      0.00000      1.00000      0.00000
";
        let global = GlobalInput::parse_str("legacy-global.inp", text)?;

        assert_eq!(global.control.do_nrixs, 1);
        assert_eq!(global.q_control.nq, 1);
        assert!(global.q_control.qaverage);
        assert!(!global.q_control.mixdff);
        assert!(global.mdff.is_none());
        assert_eq!(global.q_vectors.len(), 1);
        assert_eq!(global.q_vectors[0].q, [0.0, 0.0, 24.0]);
        Ok(())
    }

    #[test]
    fn parses_generated_nrixs_qaverage_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE NRIXS qaverage multi-q smoke
EDGE K
XANES
NRIXS -2 2.0 0.25 0.10
3.0 0.75 0.20
LDEC 4
LJMAX 2
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::global_inp_string(&document)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.control.ipol, 1);
        assert_eq!(global.control.do_nrixs, 1);
        assert_eq!(global.control.ldecmx, 4);
        assert_eq!(global.control.lj, 2);
        assert_eq!(global.control.l2lp, 30);
        assert_eq!(global.q_control.nq, 2);
        assert!(global.q_control.qaverage);
        assert!(!global.q_control.mixdff);
        assert_eq!(global.q_vectors.len(), 2);

        assert_eq!(global.q_vectors[0].q, [0.0, 0.0, 2.0]);
        assert_eq!(global.q_vectors[0].norm, 2.0);
        assert_eq!(global.q_vectors[0].weight, [0.25, 0.10]);
        assert_eq!(global.q_vectors[0].trig, [-1.0, 0.0, 1.0, 0.0]);
        assert_eq!(global.q_vectors[1].q, [0.0, 0.0, 3.0]);
        assert_eq!(global.q_vectors[1].norm, 3.0);
        assert_eq!(global.q_vectors[1].weight, [0.75, 0.20]);
        assert_eq!(global.q_vectors[1].trig, [-1.0, 0.0, 1.0, 0.0]);
        assert_eq!(global_input_string(&global)?, text);
        Ok(())
    }

    #[test]
    fn parses_generated_mdff_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 1 0.0 0.0 2.0
LDEC 4
LJMAX 2
MDFF 1
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::global_inp_string(&document)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.q_control.nq, 1);
        assert_eq!(global.q_control.imdff, 1);
        assert!(global.q_control.mixdff);
        let mdff = global.mdff.as_ref().ok_or_else(|| crate::IoError::Parse {
            path: "global.inp".into(),
            line: 0,
            message: "missing MDFF data".to_string(),
        })?;
        assert_eq!(mdff.qqmdff, -1.0);
        assert!((mdff.cosines[0] - 0.9998476951563913).abs() < 1.0e-12);
        assert_eq!(global_input_string(&global)?, text);
        Ok(())
    }

    #[test]
    fn parses_generated_multi_q_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 2 0.0 0.0 2.0 0.25
1.0 0.0 0.0 0.75
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::global_inp_string(&document)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.q_control.nq, 2);
        assert!(!global.q_control.mixdff);
        assert_eq!(global.q_vectors.len(), 2);
        assert_eq!(global.q_vectors[0].q, [0.0, 0.0, 2.0]);
        assert_eq!(global.q_vectors[0].weight, [0.25, 0.0]);
        assert_eq!(global.q_vectors[1].q, [1.0, 0.0, 0.0]);
        assert_eq!(global.q_vectors[1].weight, [0.75, 0.0]);
        assert_eq!(global_input_string(&global)?, text);
        Ok(())
    }

    #[test]
    fn parses_generated_mdff2_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 2 0.0 0.0 2.0 1.0
1.0 0.0 0.0 1.0
MDFF 2 3.0 45.0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::global_inp_string(&document)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.q_control.nq, 2);
        assert_eq!(global.q_control.imdff, 2);
        assert!(global.q_control.mixdff);
        assert_eq!(global.q_vectors.len(), 2);
        assert_eq!(global.q_vectors[1].norm, 1.0);
        assert!((global.q_vectors[1].q[1] - 2.12132).abs() < 1.0e-5);
        assert!((global.q_vectors[1].q[2] - 2.12132).abs() < 1.0e-5);
        let mdff = global.mdff.as_ref().ok_or_else(|| crate::IoError::Parse {
            path: "global.inp".into(),
            line: 0,
            message: "missing MDFF data".to_string(),
        })?;
        assert_eq!(mdff.qqmdff, 3.0);
        assert_eq!(mdff.cosines.len(), 4);
        assert!((mdff.cosines[0] - 0.9998476951563913).abs() < 1.0e-12);
        assert!((mdff.cosines[1] - 0.9993146890949606).abs() < 1.0e-12);
        assert!((mdff.cosines[2] - 0.9993146890949606).abs() < 1.0e-12);
        assert!((mdff.cosines[3] - 0.9876883405951378).abs() < 1.0e-12);
        assert_eq!(global_input_string(&global)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_global_rendering() {
        let input = GlobalInput {
            cfaverage: super::CfAverage {
                nabs: 1,
                iphabs: 0,
                rclabs: f64::NAN,
            },
            control: super::GlobalControl {
                ipol: 0,
                ispin: 0,
                le2: 0,
                elpty: 0.0,
                angks: 0.0,
                l2lp: 0,
                do_nrixs: 0,
                ldecmx: -1,
                lj: -1,
            },
            evec: [0.0; 3],
            xivec: [0.0; 3],
            spvec: [0.0; 3],
            polarization_tensor: [[0.0; 6]; 3],
            norms: super::GlobalNorms {
                evnorm: 0.0,
                xivnorm: 0.0,
                spvnorm: 0.0,
            },
            q_control: super::GlobalQControl {
                nq: 0,
                imdff: 0,
                qaverage: true,
                mixdff: false,
            },
            q_vectors: Vec::new(),
            mdff: None,
        };
        assert!(global_input_string(&input).is_err());

        let mut bad_count = input;
        bad_count.cfaverage.rclabs = 100000.0;
        bad_count.q_control.nq = 1;
        assert!(global_input_string(&bad_count).is_err());
    }
}
