//! Typed reader for FEFF `global.inp` module handoff files.
//!
//! `global.inp` carries shared polarization, spin, CFAVERAGE, and NRIXS
//! controls. Several FEFF modules read this file, so keeping it typed gives
//! the Rust port one common source of truth for those cross-module settings.

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
}

/// Configuration-average controls from `global.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfAverage {
    pub nabs: i32,
    pub iphabs: i32,
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

impl GlobalInput {
    /// Parse a FEFF `global.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = GlobalInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct GlobalInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
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
            });
        };
        self.expect_header_at(line_number, line, "nq,    imdff,   qaverage,   mixdff")?;
        let q_control = self.parse_q_control()?;

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

    fn parse_q_control(&mut self) -> Result<GlobalQControl> {
        let (line_number, line) = self.next_line("global q-control line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(self.parse_error(line_number, "global q-control line requires 4 fields"));
        }
        Ok(GlobalQControl {
            nq: parse_field(&self.source, line_number, fields[0])?,
            imdff: parse_field(&self.source, line_number, fields[1])?,
            qaverage: parse_fortran_bool(&self.source, line_number, fields[2])?,
            mixdff: parse_fortran_bool(&self.source, line_number, fields[3])?,
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

    use super::GlobalInput;

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
        Ok(())
    }

    #[test]
    fn parses_generated_nrixs_global_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
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
        Ok(())
    }
}
