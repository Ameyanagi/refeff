//! Typed readers for FEFF structural handoff files.
//!
//! These files are emitted by `rdinp` and consumed by downstream modules:
//! `.dimensions.dat` carries derived array limits, `atoms.dat` carries the
//! untrimmed atom cluster, and `geom.dat` carries the sorted cluster used by
//! scattering, potential, and density modules.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ndarray::Array2;
use refeff_core::{RhorrpFmsInclusionInput, rhorrp_fms_inclusion_counts};

use crate::control_input::FEFF_BOHR_ANGSTROM;
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

/// RHORRP-ready atom geometry imported from FEFF `geom.dat`.
///
/// `RHORRP/m_rhorrp.f90::rhorrp_init` reads the atom cluster and then converts
/// both coordinates and `rfms2` from Angstrom to Bohr before `init_inclus` and
/// `nearest_atom` run. This handoff preserves the zero-based Rust shape while
/// applying the same coordinate conversion at the IO boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpGeomHandoff {
    /// Atom Cartesian positions in Bohr as `(atom, xyz)`.
    pub atom_positions_bohr: Array2<f64>,
    /// Potential index for each atom.
    pub atom_potentials: Vec<usize>,
    /// Representative atom index for each potential, converted from FEFF one-based `iatph`.
    pub representative_atoms: Vec<usize>,
}

/// PATH-ready atom geometry imported from FEFF `geom.dat`.
///
/// `PATH/repath.f90` reads `geom.dat` through `atoms_read` and passes Angstrom
/// coordinates, potential IDs, and first-bounce flags into `paths.f90`. The
/// Rust pathfinder still performs the FEFF absorber normalization itself, so
/// this handoff preserves the `geom.dat` row order and units.
#[derive(Debug, Clone, PartialEq)]
pub struct PathfinderGeomHandoff {
    /// Atom Cartesian positions in Angstroms as `(atom, xyz)`.
    pub atom_positions_angstrom: Array2<f64>,
    /// Potential index for each atom.
    pub atom_potentials: Vec<usize>,
    /// First-bounce degeneracy flags from the final `geom.dat` integer column.
    pub first_bounce_degeneracies: Vec<usize>,
}

impl RhorrpGeomHandoff {
    /// Number of atom rows represented by this handoff.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atom_potentials.len()
    }

    /// Number of potential representatives represented by this handoff.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.representative_atoms.len()
    }

    /// Compute FEFF `init_inclus` counts for an `rfms2` value in Angstrom.
    pub fn fms_inclusion_counts(&self, fms_radius_angstrom: f64) -> Result<Vec<usize>> {
        rhorrp_fms_inclusion_counts_from_geom_handoff(self, fms_radius_angstrom)
    }
}

impl PathfinderGeomHandoff {
    /// Number of atom rows represented by this handoff.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atom_potentials.len()
    }
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

    /// Convert this FEFF `geom.dat` payload into RHORRP atom geometry.
    pub fn to_rhorrp_handoff(&self) -> Result<RhorrpGeomHandoff> {
        rhorrp_geom_handoff_from_geom_dat(self)
    }

    /// Convert this FEFF `geom.dat` payload into PATH pathfinder atom geometry.
    pub fn to_pathfinder_handoff(&self) -> Result<PathfinderGeomHandoff> {
        pathfinder_geom_handoff_from_geom_dat(self)
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

/// Build RHORRP atom geometry from FEFF `geom.dat`.
pub fn rhorrp_geom_handoff_from_geom_dat(data: &GeomDat) -> Result<RhorrpGeomHandoff> {
    validate_geom_dat(data)?;

    let mut atom_positions_bohr = Array2::zeros((data.atoms.len(), 3));
    let mut atom_potentials = Vec::with_capacity(data.atoms.len());
    for (row, atom) in data.atoms.iter().enumerate() {
        let potential = usize::try_from(atom.iph).map_err(|_| {
            invalid_rhorrp_geom(format!(
                "atom {} has negative potential index {}",
                row + 1,
                atom.iph
            ))
        })?;
        if potential > data.nph {
            return Err(invalid_rhorrp_geom(format!(
                "atom {} potential {potential} exceeds geom.dat nph {}",
                row + 1,
                data.nph
            )));
        }
        atom_positions_bohr[(row, 0)] = angstrom_to_bohr("geom.dat x", atom.x)?;
        atom_positions_bohr[(row, 1)] = angstrom_to_bohr("geom.dat y", atom.y)?;
        atom_positions_bohr[(row, 2)] = angstrom_to_bohr("geom.dat z", atom.z)?;
        atom_potentials.push(potential);
    }

    let representative_atoms = data
        .model_atoms
        .iter()
        .enumerate()
        .map(|(potential, &atom_index_1based)| {
            if atom_index_1based == 0 || atom_index_1based > data.atoms.len() {
                return Err(invalid_rhorrp_geom(format!(
                    "representative atom for potential {potential} is {atom_index_1based}, expected 1..={}",
                    data.atoms.len()
                )));
            }
            Ok(atom_index_1based - 1)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RhorrpGeomHandoff {
        atom_positions_bohr,
        atom_potentials,
        representative_atoms,
    })
}

/// Build PATH pathfinder atom geometry from FEFF `geom.dat`.
pub fn pathfinder_geom_handoff_from_geom_dat(data: &GeomDat) -> Result<PathfinderGeomHandoff> {
    validate_geom_dat(data)?;

    let mut atom_positions_angstrom = Array2::zeros((data.atoms.len(), 3));
    let mut atom_potentials = Vec::with_capacity(data.atoms.len());
    let mut first_bounce_degeneracies = Vec::with_capacity(data.atoms.len());
    for (row, atom) in data.atoms.iter().enumerate() {
        let potential = usize::try_from(atom.iph).map_err(|_| {
            invalid_pathfinder_geom(format!(
                "atom {} has negative potential index {}",
                row + 1,
                atom.iph
            ))
        })?;
        if potential > data.nph {
            return Err(invalid_pathfinder_geom(format!(
                "atom {} potential {potential} exceeds geom.dat nph {}",
                row + 1,
                data.nph
            )));
        }
        let first_bounce = usize::try_from(atom.boundary).map_err(|_| {
            invalid_pathfinder_geom(format!(
                "atom {} has negative first-bounce flag {}",
                row + 1,
                atom.boundary
            ))
        })?;
        atom_positions_angstrom[(row, 0)] = atom.x;
        atom_positions_angstrom[(row, 1)] = atom.y;
        atom_positions_angstrom[(row, 2)] = atom.z;
        atom_potentials.push(potential);
        first_bounce_degeneracies.push(first_bounce);
    }

    Ok(PathfinderGeomHandoff {
        atom_positions_angstrom,
        atom_potentials,
        first_bounce_degeneracies,
    })
}

/// Compute FEFF `init_inclus` counts from a RHORRP geometry handoff.
pub fn rhorrp_fms_inclusion_counts_from_geom_handoff(
    handoff: &RhorrpGeomHandoff,
    fms_radius_angstrom: f64,
) -> Result<Vec<usize>> {
    let fms_radius_bohr = angstrom_to_bohr("rfms2", fms_radius_angstrom)?;
    rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
        atom_positions: handoff.atom_positions_bohr.view(),
        representative_atoms: &handoff.representative_atoms,
        fms_radius: fms_radius_bohr,
    })
    .map_err(|source| invalid_rhorrp_geom(source.to_string()))
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

fn angstrom_to_bohr(field: &'static str, value: f64) -> Result<f64> {
    validate_finite(field, value)?;
    let converted = value / FEFF_BOHR_ANGSTROM;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(invalid_rhorrp_geom(format!(
            "{field} conversion produced a non-finite Bohr value"
        )))
    }
}

fn invalid_rhorrp_geom(message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "geom.dat".into(),
        line: 0,
        message: format!("invalid RHORRP geometry: {}", message.into()),
    }
}

fn invalid_pathfinder_geom(message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "geom.dat".into(),
        line: 0,
        message: format!("invalid PATH geometry: {}", message.into()),
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
        dimensions_dat_string, geom_dat_string, pathfinder_geom_handoff_from_geom_dat,
        rhorrp_geom_handoff_from_geom_dat,
    };
    use crate::{FEFF_BOHR_ANGSTROM, FeffDocument, FeffInput, IoError, rdinp};

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
    fn extracts_rhorrp_geometry_handoff_from_geom_dat() -> crate::Result<()> {
        let geom = GeomDat {
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
        };

        let handoff = rhorrp_geom_handoff_from_geom_dat(&geom)?;

        assert_eq!(handoff, geom.to_rhorrp_handoff()?);
        assert_eq!(handoff.atom_count(), 4);
        assert_eq!(handoff.potential_count(), 3);
        assert_eq!(handoff.atom_potentials, vec![0, 1, 0, 2]);
        assert_eq!(handoff.representative_atoms, vec![0, 1, 3]);
        assert_close(handoff.atom_positions_bohr[(1, 0)], 1.0);
        assert_close(handoff.atom_positions_bohr[(2, 0)], 0.4);
        assert_close(handoff.atom_positions_bohr[(3, 1)], 2.0);
        assert_eq!(
            handoff.fms_inclusion_counts(0.75 * FEFF_BOHR_ANGSTROM)?,
            vec![2, 2, 1]
        );
        Ok(())
    }

    #[test]
    fn extracts_pathfinder_geometry_handoff_from_geom_dat() -> crate::Result<()> {
        let geom = GeomDat {
            nat: 3,
            nph: 2,
            model_atoms: vec![1, 2, 3],
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
                    x: 1.25,
                    y: -0.5,
                    z: 0.0,
                    iph: 1,
                    boundary: 2,
                },
                GeomDatRow {
                    index: 3,
                    x: 0.0,
                    y: 2.5,
                    z: 1.0,
                    iph: 2,
                    boundary: 0,
                },
            ],
        };

        let handoff = pathfinder_geom_handoff_from_geom_dat(&geom)?;

        assert_eq!(handoff, geom.to_pathfinder_handoff()?);
        assert_eq!(handoff.atom_count(), 3);
        assert_eq!(handoff.atom_potentials, vec![0, 1, 2]);
        assert_eq!(handoff.first_bounce_degeneracies, vec![1, 2, 0]);
        assert_eq!(handoff.atom_positions_angstrom[(1, 0)], 1.25);
        assert_eq!(handoff.atom_positions_angstrom[(1, 1)], -0.5);
        assert_eq!(handoff.atom_positions_angstrom[(2, 2)], 1.0);
        Ok(())
    }

    #[test]
    fn rejects_invalid_rhorrp_geometry_handoffs() {
        let mut geom = GeomDat {
            nat: 1,
            nph: 0,
            model_atoms: vec![1],
            atoms: vec![GeomDatRow {
                index: 1,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: -1,
                boundary: 1,
            }],
        };
        assert!(matches!(
            rhorrp_geom_handoff_from_geom_dat(&geom),
            Err(IoError::Parse { .. })
        ));

        geom.atoms[0].iph = 0;
        geom.model_atoms[0] = 0;
        assert!(matches!(
            rhorrp_geom_handoff_from_geom_dat(&geom),
            Err(IoError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_invalid_pathfinder_geometry_handoffs() {
        let mut geom = GeomDat {
            nat: 1,
            nph: 0,
            model_atoms: vec![1],
            atoms: vec![GeomDatRow {
                index: 1,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: -1,
                boundary: 1,
            }],
        };
        assert!(matches!(
            pathfinder_geom_handoff_from_geom_dat(&geom),
            Err(IoError::Parse { .. })
        ));

        geom.atoms[0].iph = 0;
        geom.atoms[0].boundary = -1;
        assert!(matches!(
            pathfinder_geom_handoff_from_geom_dat(&geom),
            Err(IoError::Parse { .. })
        ));
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

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
