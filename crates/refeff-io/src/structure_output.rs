//! Typed readers for FEFF structural handoff files.
//!
//! These files are emitted by `rdinp` and consumed by downstream modules:
//! `.dimensions.dat` carries derived array limits, `atoms.dat` carries the
//! untrimmed atom cluster, and `geom.dat` carries the sorted cluster used by
//! scattering, potential, and density modules.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of FEFF `.dimensions.dat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionsDat {
    /// Maximum cluster size selected for later modules.
    pub nclusx: usize,
    /// Maximum angular momentum channel.
    pub lx: i32,
    /// Highest unique potential index.
    pub nphu: i32,
    /// Spin channel count.
    pub nspu: i32,
}

/// Parsed contents of FEFF `atoms.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomsDat {
    /// Number of atom rows advertised by the `natx` header.
    pub natx: usize,
    /// Atom rows in FEFF input order.
    pub atoms: Vec<AtomsDatRow>,
}

/// One atom row from FEFF `atoms.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomsDatRow {
    /// Cartesian x coordinate.
    pub x: f64,
    /// Cartesian y coordinate.
    pub y: f64,
    /// Cartesian z coordinate.
    pub z: f64,
    /// FEFF potential index (`iph`).
    pub iph: i32,
    /// Distance from the absorber.
    pub distance: f64,
}

/// Parsed contents of FEFF `geom.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeomDat {
    /// Number of atom rows.
    pub nat: usize,
    /// Highest unique potential index.
    pub nph: usize,
    /// Representative atom index for each potential, indexed from `iph = 0`.
    pub model_atoms: Vec<usize>,
    /// Sorted geometry rows.
    pub atoms: Vec<GeomDatRow>,
}

/// One sorted atom row from FEFF `geom.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeomDatRow {
    /// FEFF one-based row index.
    pub index: usize,
    /// Cartesian x coordinate.
    pub x: f64,
    /// Cartesian y coordinate.
    pub y: f64,
    /// Cartesian z coordinate.
    pub z: f64,
    /// FEFF potential index (`iph`).
    pub iph: i32,
    /// Boundary marker used by path finding.
    pub boundary: i32,
}

impl DimensionsDat {
    /// Parse a FEFF `.dimensions.dat` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = StructureParser::new(source.into(), text);
        parser.parse_dimensions()
    }
}

impl AtomsDat {
    /// Parse a FEFF `atoms.dat` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = StructureParser::new(source.into(), text);
        parser.parse_atoms_dat()
    }
}

impl GeomDat {
    /// Parse a FEFF `geom.dat` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = StructureParser::new(source.into(), text);
        parser.parse_geom_dat()
    }
}

/// Render FEFF-compatible `.dimensions.dat` text.
pub fn dimensions_dat_string(data: &DimensionsDat) -> Result<String> {
    Ok(format!(
        "{:12}{:12}{:12}{:12}\n",
        data.nclusx, data.lx, data.nphu, data.nspu
    ))
}

/// Render FEFF-compatible `atoms.dat` text.
pub fn atoms_dat_string(data: &AtomsDat) -> Result<String> {
    validate_atoms_dat(data)?;

    let mut out = String::new();
    writeln!(out, "natx =  {:>7}", data.natx)?;
    writeln!(out, "    x       y        z       iph  ")?;
    for atom in &data.atoms {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:4}{:13.5}",
            atom.x, atom.y, atom.z, atom.iph, atom.distance
        )?;
    }
    Ok(out)
}

/// Render FEFF-compatible `geom.dat` text.
pub fn geom_dat_string(data: &GeomDat) -> Result<String> {
    validate_geom_dat(data)?;

    let mut out = String::new();
    writeln!(out, "nat, nph = {:5}{:5}", data.nat, data.nph)?;
    for model_atom in &data.model_atoms {
        write!(out, "{model_atom:5}")?;
    }
    out.push('\n');
    writeln!(out, " iat     x       y        z       iph  ")?;
    writeln!(out, " {}", "-".repeat(71))?;
    for atom in &data.atoms {
        writeln!(
            out,
            "{:4}{:13.5}{:13.5}{:13.5}{:4}{:4}",
            atom.index, atom.x, atom.y, atom.z, atom.iph, atom.boundary
        )?;
    }
    Ok(out)
}

fn validate_atoms_dat(data: &AtomsDat) -> Result<()> {
    if data.natx != data.atoms.len() {
        return Err(structure_render_error(format!(
            "atoms.dat natx {} does not match row count {}",
            data.natx,
            data.atoms.len()
        )));
    }
    for atom in &data.atoms {
        validate_finite("atoms.dat x", atom.x)?;
        validate_finite("atoms.dat y", atom.y)?;
        validate_finite("atoms.dat z", atom.z)?;
        validate_finite("atoms.dat distance", atom.distance)?;
    }
    Ok(())
}

fn validate_geom_dat(data: &GeomDat) -> Result<()> {
    if data.nat != data.atoms.len() {
        return Err(structure_render_error(format!(
            "geom.dat nat {} does not match row count {}",
            data.nat,
            data.atoms.len()
        )));
    }
    let expected_model_atoms = data.nph + 1;
    if data.model_atoms.len() != expected_model_atoms {
        return Err(structure_render_error(format!(
            "geom.dat model atom count {} does not match nph-derived count {expected_model_atoms}",
            data.model_atoms.len()
        )));
    }
    for atom in &data.atoms {
        validate_finite("geom.dat x", atom.x)?;
        validate_finite("geom.dat y", atom.y)?;
        validate_finite("geom.dat z", atom.z)?;
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(structure_render_error(format!("{field} must be finite")))
    }
}

fn structure_render_error(message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "structure output".into(),
        line: 0,
        message: message.into(),
    }
}

struct StructureParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> StructureParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse_dimensions(&mut self) -> Result<DimensionsDat> {
        let (line_number, line) = self.next_nonempty_line("dimensions values")?;
        let fields = fields(line);
        if fields.len() < 4 {
            return Err(self.parse_error(line_number, ".dimensions.dat requires 4 fields"));
        }

        Ok(DimensionsDat {
            nclusx: parse_field(&self.source, line_number, fields[0])?,
            lx: parse_field(&self.source, line_number, fields[1])?,
            nphu: parse_field(&self.source, line_number, fields[2])?,
            nspu: parse_field(&self.source, line_number, fields[3])?,
        })
    }

    fn parse_atoms_dat(&mut self) -> Result<AtomsDat> {
        let (line_number, line) = self.next_nonempty_line("atoms.dat natx header")?;
        let natx = parse_named_count(&self.source, line_number, line, "natx")?;
        self.next_line("atoms.dat column header")?;

        let atoms = (0..natx)
            .map(|_| self.parse_atoms_dat_row())
            .collect::<Result<Vec<_>>>()?;

        Ok(AtomsDat { natx, atoms })
    }

    fn parse_atoms_dat_row(&mut self) -> Result<AtomsDatRow> {
        let (line_number, line) = self.next_nonempty_line("atoms.dat atom row")?;
        let fields = fields(line);
        if fields.len() < 5 {
            return Err(self.parse_error(line_number, "atoms.dat row requires 5 fields"));
        }
        Ok(AtomsDatRow {
            x: parse_field(&self.source, line_number, fields[0])?,
            y: parse_field(&self.source, line_number, fields[1])?,
            z: parse_field(&self.source, line_number, fields[2])?,
            iph: parse_field(&self.source, line_number, fields[3])?,
            distance: parse_field(&self.source, line_number, fields[4])?,
        })
    }

    fn parse_geom_dat(&mut self) -> Result<GeomDat> {
        let (line_number, line) = self.next_nonempty_line("geom.dat nat/nph header")?;
        let (nat, nph) = parse_geom_counts(&self.source, line_number, line)?;
        let model_atoms = self.parse_model_atoms(nph + 1)?;
        self.next_line("geom.dat column header")?;
        self.next_line("geom.dat separator")?;

        let atoms = (0..nat)
            .map(|_| self.parse_geom_dat_row())
            .collect::<Result<Vec<_>>>()?;

        Ok(GeomDat {
            nat,
            nph,
            model_atoms,
            atoms,
        })
    }

    fn parse_model_atoms(&mut self, count: usize) -> Result<Vec<usize>> {
        let mut model_atoms = Vec::with_capacity(count);
        while model_atoms.len() < count {
            let (line_number, line) = self.next_nonempty_line("geom.dat model atom list")?;
            let remaining = count - model_atoms.len();
            for field in fields(line).into_iter().take(remaining) {
                model_atoms.push(parse_field(&self.source, line_number, field)?);
            }
        }
        Ok(model_atoms)
    }

    fn parse_geom_dat_row(&mut self) -> Result<GeomDatRow> {
        let (line_number, line) = self.next_nonempty_line("geom.dat atom row")?;
        let fields = fields(line);
        if fields.len() < 6 {
            return Err(self.parse_error(line_number, "geom.dat row requires 6 fields"));
        }
        Ok(GeomDatRow {
            index: parse_field(&self.source, line_number, fields[0])?,
            x: parse_field(&self.source, line_number, fields[1])?,
            y: parse_field(&self.source, line_number, fields[2])?,
            z: parse_field(&self.source, line_number, fields[3])?,
            iph: parse_field(&self.source, line_number, fields[4])?,
            boundary: parse_field(&self.source, line_number, fields[5])?,
        })
    }

    fn next_nonempty_line(&mut self, description: &str) -> Result<(usize, &'a str)> {
        loop {
            let (line_number, line) = self.next_line(description)?;
            if !line.trim().is_empty() {
                return Ok((line_number, line));
            }
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

fn fields(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

fn parse_named_count(source: &Path, line: usize, text: &str, name: &str) -> Result<usize> {
    let fields = fields(text);
    if fields.len() < 3 || fields[0] != name || fields[1] != "=" {
        return Err(IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: format!("expected {name} = count header"),
        });
    }
    parse_field(source, line, fields[2])
}

fn parse_geom_counts(source: &Path, line: usize, text: &str) -> Result<(usize, usize)> {
    let fields = fields(text);
    if fields.len() < 5 || fields[0] != "nat," || fields[1] != "nph" || fields[2] != "=" {
        return Err(IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: "expected nat, nph = counts header".to_string(),
        });
    }
    Ok((
        parse_field(source, line, fields[3])?,
        parse_field(source, line, fields[4])?,
    ))
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
    use crate::structure_output::{
        AtomsDat, AtomsDatRow, DimensionsDat, GeomDat, GeomDatRow, atoms_dat_string,
        dimensions_dat_string, geom_dat_string,
    };
    use crate::{FeffDocument, FeffInput, IoError, rdinp};

    #[test]
    fn parses_dimensions_dat_from_writer() -> crate::Result<()> {
        let document = copper_cluster_document()?;
        let text = rdinp::dimensions_dat_string(&document)?;
        let dimensions = DimensionsDat::parse_str(".dimensions.dat", &text)?;

        assert_eq!(
            dimensions,
            DimensionsDat {
                nclusx: 2,
                lx: 3,
                nphu: 1,
                nspu: 1,
            }
        );
        assert_eq!(dimensions_dat_string(&dimensions)?, text);
        Ok(())
    }

    #[test]
    fn parses_atoms_dat_from_writer() -> crate::Result<()> {
        let document = copper_cluster_document()?;
        let text = rdinp::atoms_dat_string(&document)?;
        let atoms = AtomsDat::parse_str("atoms.dat", &text)?;
        let first = atoms
            .atoms
            .first()
            .ok_or_else(|| parse_error("atoms.dat"))?;
        let second = atoms.atoms.get(1).ok_or_else(|| parse_error("atoms.dat"))?;

        assert_eq!(atoms.natx, 3);
        assert_eq!(
            *first,
            AtomsDatRow {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: 0,
                distance: 0.0,
            }
        );
        assert_eq!(second.iph, 1);
        assert_eq!(second.distance, 1.0);
        assert_eq!(atoms_dat_string(&atoms)?, text);
        Ok(())
    }

    #[test]
    fn parses_geom_dat_from_writer() -> crate::Result<()> {
        let document = copper_cluster_document()?;
        let text = rdinp::geom_dat_string(&document)?;
        let geom = GeomDat::parse_str("geom.dat", &text)?;
        let absorber = geom.atoms.first().ok_or_else(|| parse_error("geom.dat"))?;
        let scatterer = geom.atoms.get(1).ok_or_else(|| parse_error("geom.dat"))?;

        assert_eq!(geom.nat, 3);
        assert_eq!(geom.nph, 1);
        assert_eq!(geom.model_atoms, vec![1, 2]);
        assert_eq!(
            *absorber,
            GeomDatRow {
                index: 1,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: 0,
                boundary: 1,
            }
        );
        assert_eq!(scatterer.index, 2);
        assert_eq!(scatterer.x, 1.0);
        assert_eq!(scatterer.iph, 1);
        assert_eq!(geom_dat_string(&geom)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_structure_rendering() {
        let atoms = AtomsDat {
            natx: 2,
            atoms: vec![AtomsDatRow {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: 0,
                distance: 0.0,
            }],
        };
        assert!(atoms_dat_string(&atoms).is_err());

        let geom = GeomDat {
            nat: 1,
            nph: 1,
            model_atoms: vec![1],
            atoms: vec![GeomDatRow {
                index: 1,
                x: f64::NAN,
                y: 0.0,
                z: 0.0,
                iph: 0,
                boundary: 1,
            }],
        };
        assert!(geom_dat_string(&geom).is_err());
    }

    #[test]
    fn parses_wrapped_geom_model_atom_list() -> crate::Result<()> {
        let geom = GeomDat::parse_str(
            "geom.dat",
            concat!(
                "nat, nph =     1   17\n",
                "    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16\n",
                "   17   18\n",
                " iat     x       y        z       iph  \n",
                " -----------------------------------------------------------------------\n",
                "   1      0.00000      0.00000      0.00000   0   1\n",
            ),
        )?;
        let sixteenth = geom
            .model_atoms
            .get(15)
            .ok_or_else(|| parse_error("geom.dat"))?;
        let eighteenth = geom
            .model_atoms
            .get(17)
            .ok_or_else(|| parse_error("geom.dat"))?;

        assert_eq!(geom.model_atoms.len(), 18);
        assert_eq!(*sixteenth, 16);
        assert_eq!(*eighteenth, 18);
        Ok(())
    }

    fn copper_cluster_document() -> crate::Result<FeffDocument> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RPATH 2.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
3.0 0.0 0.0 1 Cu2
END
"#,
        )?;
        FeffDocument::from_input(&input)
    }

    fn parse_error(path: &str) -> IoError {
        IoError::Parse {
            path: path.into(),
            line: 0,
            message: "expected parsed test row".to_string(),
        }
    }
}
