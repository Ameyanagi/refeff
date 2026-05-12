//! Typed readers for small FEFF module-control handoff files.
//!
//! These files mostly carry switches or compact lists emitted by `rdinp`:
//! `band.inp`, `density.inp`, `fullspectrum.inp`, `opcons.inp`, and
//! `reciprocal.inp`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of FEFF `band.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandInput {
    /// Band module run flag.
    pub mband: i32,
    /// Energy mesh definition.
    pub energy_mesh: BandEnergyMesh,
    /// Number of k-path points.
    pub nkp: usize,
    /// K-path selector.
    pub ikpath: i32,
    /// Empty-lattice propagation switch.
    pub freeprop: bool,
}

/// Energy mesh row from `band.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandEnergyMesh {
    pub emin: f64,
    pub emax: f64,
    pub estep: f64,
}

/// Parsed contents of FEFF `density.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityInput {
    /// Requested density grids.
    pub grids: Vec<DensityGrid>,
}

/// One density grid request.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityGrid {
    /// Grid dimensionality and command type.
    pub kind: DensityGridKind,
    /// Output filename requested by FEFF input.
    pub filename: String,
    /// Origin in the file's Angstrom coordinate units.
    pub origin: [f64; 3],
    /// Whether the optional `core` flag is present.
    pub core: bool,
    /// Axis rows in the file's Angstrom coordinate units.
    pub axes: Vec<DensityAxis>,
}

/// Density grid command kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityGridKind {
    Line,
    Plane,
    Volume,
}

/// One axis row from a density grid request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityAxis {
    /// Axis vector in the file's Angstrom coordinate units.
    pub vector: [f64; 3],
    /// Number of points along the axis.
    pub points: usize,
}

/// Parsed contents of FEFF `fullspectrum.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullSpectrumInput {
    /// Full-spectrum module run flag.
    pub m_full_spectrum: i32,
}

/// Parsed contents of FEFF `opcons.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpconsInput {
    /// Whether optical constants should run.
    pub run_opcons: bool,
    /// Whether epsilon output should be printed.
    pub print_eps: bool,
    /// Number densities for potential indices.
    pub number_densities: Vec<f64>,
}

/// Parsed contents of FEFF `reciprocal.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalInput {
    /// FEFF space selector: `1` for real space, `0` for reciprocal space.
    pub ispace: i32,
    /// Reciprocal-space cell block, present only when `ispace == 0`.
    pub cell: Option<ReciprocalCell>,
}

/// Reciprocal-space cell block from `reciprocal.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalCell {
    /// Lattice vectors in Angstrom Cartesian coordinates.
    pub lattice_vectors: [[f64; 3]; 3],
    /// Volume scaling factor.
    pub volume_scale: f64,
    /// Imaginary energy broadening.
    pub imaginary_energy: f64,
    /// Core-hole strength selector.
    pub core_hole_strength: f64,
    /// FEFF lattice name.
    pub lattice_name: String,
    /// Hermann-Mauguin space-group label.
    pub space_group_hm: String,
    /// Numeric space-group identifier.
    pub space_group: i32,
    /// Number of atoms in the unit cell.
    pub atom_count: usize,
    /// Absorber position selector.
    pub absorber: i32,
    /// Core-hole selector.
    pub core_hole: i32,
    /// K-point mesh controls.
    pub k_mesh: ReciprocalKMesh,
    /// Unit-cell atom positions.
    pub positions: Vec<[f64; 3]>,
    /// Potential index for each unit-cell atom.
    pub potentials: Vec<i32>,
    /// Atom labels for each unit-cell atom.
    pub labels: Vec<String>,
    /// `streta`, `strgmax`, and `strrmax` controls.
    pub stretch: [f64; 3],
}

/// K-point mesh controls from reciprocal-space input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalKMesh {
    pub total: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub kind: i32,
    pub use_symmetry: bool,
}

impl BandInput {
    /// Parse a FEFF `band.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_band()
    }
}

impl DensityInput {
    /// Parse a FEFF `density.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_density()
    }
}

impl FullSpectrumInput {
    /// Parse a FEFF `fullspectrum.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_fullspectrum()
    }
}

impl OpconsInput {
    /// Parse a FEFF `opcons.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_opcons()
    }
}

impl ReciprocalInput {
    /// Parse a FEFF `reciprocal.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_reciprocal()
    }
}

/// Render FEFF-compatible `reciprocal.inp` text.
pub fn reciprocal_input_string(input: &ReciprocalInput) -> Result<String> {
    validate_reciprocal_input(input)?;

    let mut out = String::new();
    writeln!(out, "ispace")?;
    writeln!(out, "{:4}", input.ispace)?;
    if let Some(cell) = &input.cell {
        writeln!(out, "lattice vectors  (in A, in Carthesian coordinates)")?;
        for row in cell.lattice_vectors {
            write_f13_5_line(&mut out, row)?;
        }
        writeln!(out, "Volume scaling factor (A^3); eimag; core hole")?;
        write_f13_5_line(
            &mut out,
            [
                cell.volume_scale,
                cell.imaginary_energy,
                cell.core_hole_strength,
            ],
        )?;
        writeln!(out, "lattice type  (P,I,F,R,B,CXY,CYZ,CXZ)")?;
        writeln!(
            out,
            "{}{}{:>3}",
            fixed_left(&cell.lattice_name, 7),
            fixed_left(&cell.space_group_hm, 13),
            cell.space_group
        )?;
        writeln!(out, "#atoms in unit cell ; position absorber ; corehole?")?;
        writeln!(
            out,
            "{:4}{:4}{:4}",
            cell.atom_count, cell.absorber, cell.core_hole
        )?;
        writeln!(out, "# k-points total/x/y/z ; ktype; use symmetry?")?;
        writeln!(
            out,
            "{:12}{:12}{:12}{:12}{:12}{:12}",
            cell.k_mesh.total,
            cell.k_mesh.x,
            cell.k_mesh.y,
            cell.k_mesh.z,
            cell.k_mesh.kind,
            i32::from(cell.k_mesh.use_symmetry)
        )?;
        writeln!(out, "ppos")?;
        for position in &cell.positions {
            write_f13_5_line(&mut out, *position)?;
        }
        writeln!(out, "ppot")?;
        for potential in &cell.potentials {
            write!(out, "{potential:12}")?;
        }
        writeln!(out)?;
        writeln!(out, "label")?;
        writeln!(out, "{}", reciprocal_label_line(&cell.labels))?;
        writeln!(out, "streta,strgmax,strrmax")?;
        write_f13_5_line(&mut out, cell.stretch)?;
    }
    Ok(out)
}

fn validate_reciprocal_input(input: &ReciprocalInput) -> Result<()> {
    match (input.ispace, input.cell.as_ref()) {
        (0, Some(cell)) => {
            if cell.positions.len() != cell.atom_count {
                return Err(IoError::Parse {
                    path: "reciprocal.inp".into(),
                    line: 0,
                    message: format!(
                        "reciprocal.inp atom_count is {} but has {} positions",
                        cell.atom_count,
                        cell.positions.len()
                    ),
                });
            }
            if cell.potentials.len() != cell.atom_count {
                return Err(IoError::Parse {
                    path: "reciprocal.inp".into(),
                    line: 0,
                    message: format!(
                        "reciprocal.inp atom_count is {} but has {} potentials",
                        cell.atom_count,
                        cell.potentials.len()
                    ),
                });
            }
            Ok(())
        }
        (0, None) => Err(IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: "reciprocal-space input requires a cell block".to_string(),
        }),
        (1, None) => Ok(()),
        (1, Some(_)) => Err(IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: "real-space reciprocal.inp must not include a cell block".to_string(),
        }),
        (ispace, _) => Err(IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: format!("unsupported reciprocal ispace {ispace}"),
        }),
    }
}

fn write_f13_5_line(out: &mut String, values: [f64; 3]) -> Result<()> {
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        values[0], values[1], values[2]
    )?;
    Ok(())
}

fn fixed_left(value: &str, width: usize) -> String {
    let mut out: String = value.chars().take(width).collect();
    while out.len() < width {
        out.push(' ');
    }
    out
}

fn reciprocal_label_line(labels: &[String]) -> String {
    let mut out = String::new();
    for label in labels {
        out.push_str(&fixed_left(label, 3));
    }
    for _ in 0..14 {
        out.push(' ');
    }
    out
}

impl DensityGridKind {
    fn dimensions(self) -> usize {
        match self {
            DensityGridKind::Line => 1,
            DensityGridKind::Plane => 2,
            DensityGridKind::Volume => 3,
        }
    }
}

struct ControlParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> ControlParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse_band(&mut self) -> Result<BandInput> {
        self.next_line("BAND mband header")?;
        let mband = self.parse_single("BAND mband line")?;

        self.next_line("BAND energy header")?;
        let energy = self.parse_array3("BAND energy line")?;

        self.next_line("BAND nkp header")?;
        let nkp = self.parse_single("BAND nkp line")?;

        self.next_line("BAND ikpath header")?;
        let ikpath = self.parse_single("BAND ikpath line")?;

        self.next_line("BAND freeprop header")?;
        let freeprop = self.parse_bool_line("BAND freeprop line")?;

        Ok(BandInput {
            mband,
            energy_mesh: BandEnergyMesh {
                emin: energy[0],
                emax: energy[1],
                estep: energy[2],
            },
            nkp,
            ikpath,
            freeprop,
        })
    }

    fn parse_density(&mut self) -> Result<DensityInput> {
        let mut grids = Vec::new();
        while let Some((line_number, line)) = self.next_density_line()? {
            grids.push(self.parse_density_grid(line_number, line)?);
        }
        Ok(DensityInput { grids })
    }

    fn parse_density_grid(&mut self, line_number: usize, line: &str) -> Result<DensityGrid> {
        let fields = fields(line);
        if fields.len() < 5 {
            return Err(
                self.parse_error(line_number, "density grid row requires at least 5 fields")
            );
        }

        let kind = parse_density_kind(&self.source, line_number, fields[0])?;
        let axes = (0..kind.dimensions())
            .map(|_| self.parse_density_axis())
            .collect::<Result<Vec<_>>>()?;

        Ok(DensityGrid {
            kind,
            filename: fields[1].to_string(),
            origin: [
                parse_field(&self.source, line_number, fields[2])?,
                parse_field(&self.source, line_number, fields[3])?,
                parse_field(&self.source, line_number, fields[4])?,
            ],
            core: fields
                .iter()
                .skip(5)
                .any(|field| field.eq_ignore_ascii_case("core")),
            axes,
        })
    }

    fn parse_density_axis(&mut self) -> Result<DensityAxis> {
        let (line_number, line) = self
            .next_density_line()?
            .ok_or_else(|| self.parse_error(0, "expected density axis row"))?;
        let fields = fields(line);
        if fields.len() < 4 {
            return Err(self.parse_error(line_number, "density axis row requires 4 fields"));
        }
        Ok(DensityAxis {
            vector: [
                parse_field(&self.source, line_number, fields[0])?,
                parse_field(&self.source, line_number, fields[1])?,
                parse_field(&self.source, line_number, fields[2])?,
            ],
            points: parse_field(&self.source, line_number, fields[3])?,
        })
    }

    fn parse_fullspectrum(&mut self) -> Result<FullSpectrumInput> {
        self.next_line("FULLSPECTRUM header")?;
        Ok(FullSpectrumInput {
            m_full_spectrum: self.parse_single("FULLSPECTRUM flag line")?,
        })
    }

    fn parse_opcons(&mut self) -> Result<OpconsInput> {
        self.next_line("OPCONS run header")?;
        let run_opcons = self.parse_bool_line("OPCONS run line")?;
        self.next_line("OPCONS print header")?;
        let print_eps = self.parse_bool_line("OPCONS print line")?;
        self.next_line("OPCONS density header")?;
        let number_densities = self.parse_remaining_numeric_values()?;

        Ok(OpconsInput {
            run_opcons,
            print_eps,
            number_densities,
        })
    }

    fn parse_reciprocal(&mut self) -> Result<ReciprocalInput> {
        self.next_line("RECIPROCAL ispace header")?;
        let ispace = self.parse_single("RECIPROCAL ispace line")?;
        let cell = if ispace == 0 {
            Some(self.parse_reciprocal_cell()?)
        } else {
            None
        };

        Ok(ReciprocalInput { ispace, cell })
    }

    fn parse_reciprocal_cell(&mut self) -> Result<ReciprocalCell> {
        self.next_line("RECIPROCAL lattice-vector header")?;
        let a1 = self.parse_array3("RECIPROCAL lattice vector a1")?;
        let a2 = self.parse_array3("RECIPROCAL lattice vector a2")?;
        let a3 = self.parse_array3("RECIPROCAL lattice vector a3")?;

        self.next_line("RECIPROCAL volume/eimag/core-hole header")?;
        let scaling = self.parse_array3("RECIPROCAL volume/eimag/core-hole line")?;

        self.next_line("RECIPROCAL lattice-type header")?;
        let (lattice_name, space_group_hm, space_group) = self.parse_lattice_type()?;

        self.next_line("RECIPROCAL atom-count header")?;
        let atom_counts = self.parse_values::<i32>(3, "RECIPROCAL atom-count line")?;
        let atom_count = checked_usize(&self.source, 0, atom_counts[0], "atom count")?;

        self.next_line("RECIPROCAL k-point header")?;
        let k_mesh = self.parse_k_mesh()?;

        self.next_line("RECIPROCAL position header")?;
        let positions = (0..atom_count)
            .map(|_| self.parse_array3("RECIPROCAL atom position"))
            .collect::<Result<Vec<_>>>()?;

        self.next_line("RECIPROCAL potential header")?;
        let potentials = self.parse_repeated_fields(atom_count, "RECIPROCAL potential list")?;

        self.next_line("RECIPROCAL label header")?;
        let labels = self.parse_label_line("RECIPROCAL label list")?;

        self.next_line("RECIPROCAL stretch header")?;
        let stretch = self.parse_array3("RECIPROCAL stretch line")?;

        Ok(ReciprocalCell {
            lattice_vectors: [a1, a2, a3],
            volume_scale: scaling[0],
            imaginary_energy: scaling[1],
            core_hole_strength: scaling[2],
            lattice_name,
            space_group_hm,
            space_group,
            atom_count,
            absorber: atom_counts[1],
            core_hole: atom_counts[2],
            k_mesh,
            positions,
            potentials,
            labels,
            stretch,
        })
    }

    fn parse_lattice_type(&mut self) -> Result<(String, String, i32)> {
        let (line_number, line) = self.next_line("RECIPROCAL lattice-type line")?;
        let fields = fields(line);
        if fields.len() < 3 {
            return Err(self.parse_error(
                line_number,
                "RECIPROCAL lattice-type line requires 3 fields",
            ));
        }
        Ok((
            fields[0].to_string(),
            fields[1].to_string(),
            parse_field(&self.source, line_number, fields[2])?,
        ))
    }

    fn parse_k_mesh(&mut self) -> Result<ReciprocalKMesh> {
        let (line_number, line) = self.next_line("RECIPROCAL k-point line")?;
        let fields = fields(line);
        if fields.len() < 6 {
            return Err(self.parse_error(line_number, "RECIPROCAL k-point line requires 6 fields"));
        }

        Ok(ReciprocalKMesh {
            total: parse_field(&self.source, line_number, fields[0])?,
            x: parse_field(&self.source, line_number, fields[1])?,
            y: parse_field(&self.source, line_number, fields[2])?,
            z: parse_field(&self.source, line_number, fields[3])?,
            kind: parse_field(&self.source, line_number, fields[4])?,
            use_symmetry: parse_bool_field(&self.source, line_number, fields[5])?,
        })
    }

    fn parse_values<T>(&mut self, count: usize, description: &str) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let (line_number, line) = self.next_line(description)?;
        parse_line_values(&self.source, line_number, line, count, description)
    }

    fn parse_single<T>(&mut self, description: &str) -> Result<T>
    where
        T: FromStr,
    {
        let values = self.parse_values(1, description)?;
        values
            .into_iter()
            .next()
            .ok_or_else(|| self.parse_error(0, format!("expected {description}")))
    }

    fn parse_array3(&mut self, description: &str) -> Result<[f64; 3]> {
        let values = self.parse_values::<f64>(3, description)?;
        match values.as_slice() {
            [x, y, z] => Ok([*x, *y, *z]),
            _ => Err(self.parse_error(0, format!("expected {description}"))),
        }
    }

    fn parse_bool_line(&mut self, description: &str) -> Result<bool> {
        let (line_number, line) = self.next_line(description)?;
        let fields = fields(line);
        let Some(field) = fields.first() else {
            return Err(self.parse_error(line_number, format!("{description} requires 1 field")));
        };
        parse_bool_field(&self.source, line_number, field)
    }

    fn parse_remaining_numeric_values<T>(&mut self) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let mut values = Vec::new();
        for (index, line) in self.lines.by_ref() {
            let line_number = index + 1;
            for field in fields(line) {
                values.push(parse_field(&self.source, line_number, field)?);
            }
        }
        Ok(values)
    }

    fn parse_repeated_fields<T>(&mut self, count: usize, description: &str) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let remaining = count - values.len();
            let (line_number, line) = self.next_line(description)?;
            for field in fields(line).into_iter().take(remaining) {
                values.push(parse_field(&self.source, line_number, field)?);
            }
        }
        Ok(values)
    }

    fn parse_label_line(&mut self, description: &str) -> Result<Vec<String>> {
        let (_, line) = self.next_line(description)?;
        Ok(fields(line)
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect())
    }

    fn next_density_line(&mut self) -> Result<Option<(usize, &'a str)>> {
        for (index, line) in self.lines.by_ref() {
            let trimmed = line.trim();
            if trimmed.is_empty() || is_density_comment(trimmed) {
                continue;
            }
            return Ok(Some((index + 1, line)));
        }
        Ok(None)
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

fn parse_line_values<T>(
    source: &Path,
    line_number: usize,
    line: &str,
    count: usize,
    description: &str,
) -> Result<Vec<T>>
where
    T: FromStr,
{
    let fields = fields(line);
    if fields.len() < count {
        return Err(IoError::Parse {
            path: source.to_path_buf(),
            line: line_number,
            message: format!("{description} requires {count} fields"),
        });
    }
    fields
        .iter()
        .take(count)
        .map(|field| parse_field(source, line_number, field))
        .collect()
}

fn parse_density_kind(source: &Path, line: usize, field: &str) -> Result<DensityGridKind> {
    match field {
        "line" => Ok(DensityGridKind::Line),
        "plane" => Ok(DensityGridKind::Plane),
        "volume" => Ok(DensityGridKind::Volume),
        _ => Err(IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: format!("unknown density grid type {field:?}"),
        }),
    }
}

fn parse_bool_field(source: &Path, line: usize, field: &str) -> Result<bool> {
    let normalized = field.trim_matches('.').to_ascii_uppercase();
    match normalized.as_str() {
        "T" | "TRUE" | "1" => Ok(true),
        "F" | "FALSE" | "0" => Ok(false),
        _ => Err(IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: format!("invalid logical field {field:?}"),
        }),
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

fn checked_usize(source: &Path, line: usize, value: i32, description: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("{description} must be nonnegative"),
    })
}

fn is_density_comment(trimmed: &str) -> bool {
    trimmed.starts_with('#')
        || trimmed.starts_with('!')
        || trimmed.starts_with('*')
        || trimmed.starts_with('C')
}

#[cfg(test)]
mod tests {
    use crate::control_input::{
        BandInput, DensityGridKind, DensityInput, FullSpectrumInput, OpconsInput, ReciprocalInput,
        reciprocal_input_string,
    };
    use crate::{FeffDocument, FeffInput, rdinp};

    #[test]
    fn parses_generated_band_input() -> crate::Result<()> {
        let band = BandInput::parse_str("band.inp", &rdinp::band_inp_string())?;

        assert_eq!(band.mband, 0);
        assert_eq!(band.energy_mesh.emin, 0.0);
        assert_eq!(band.energy_mesh.emax, 0.0);
        assert_eq!(band.energy_mesh.estep, 0.0);
        assert_eq!(band.nkp, 0);
        assert_eq!(band.ikpath, -1);
        assert!(!band.freeprop);
        Ok(())
    }

    #[test]
    fn parses_empty_density_input() -> crate::Result<()> {
        let density = DensityInput::parse_str("density.inp", &rdinp::density_inp_string())?;

        assert!(density.grids.is_empty());
        Ok(())
    }

    #[test]
    fn parses_density_grid_requests() -> crate::Result<()> {
        let density = DensityInput::parse_str(
            "density.inp",
            concat!(
                "# comment\n",
                "line line.dat 0.0 1.0 2.0 core\n",
                "1.0 0.0 0.0 101\n",
                "plane plane.dat 0.0 0.0 0.0\n",
                "1.0 0.0 0.0 11\n",
                "0.0 1.0 0.0 12\n",
            ),
        )?;
        let line = density.grids.first().ok_or_else(|| crate::IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: "expected line grid".to_string(),
        })?;
        let plane = density.grids.get(1).ok_or_else(|| crate::IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: "expected plane grid".to_string(),
        })?;
        let line_axis = line.axes.first().ok_or_else(|| crate::IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: "expected line axis".to_string(),
        })?;

        assert_eq!(line.kind, DensityGridKind::Line);
        assert_eq!(line.filename, "line.dat");
        assert!(line.core);
        assert_eq!(line.axes.len(), 1);
        assert_eq!(line_axis.points, 101);
        assert_eq!(plane.kind, DensityGridKind::Plane);
        assert_eq!(plane.axes.len(), 2);
        Ok(())
    }

    #[test]
    fn parses_generated_fullspectrum_input() -> crate::Result<()> {
        let fullspectrum =
            FullSpectrumInput::parse_str("fullspectrum.inp", &rdinp::fullspectrum_inp_string())?;

        assert_eq!(fullspectrum.m_full_spectrum, 0);
        Ok(())
    }

    #[test]
    fn parses_generated_opcons_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
OPCONS
POTENTIALS
0 29 Cu
1 29 Cu
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let opcons = OpconsInput::parse_str("opcons.inp", &rdinp::opcons_inp_string(&document))?;

        assert!(opcons.run_opcons);
        assert!(!opcons.print_eps);
        assert_eq!(opcons.number_densities, vec![-1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn parses_generated_reciprocal_input() -> crate::Result<()> {
        let text = rdinp::reciprocal_inp_string();
        let reciprocal = ReciprocalInput::parse_str("reciprocal.inp", &text)?;

        assert_eq!(reciprocal.ispace, 1);
        assert!(reciprocal.cell.is_none());
        assert_eq!(reciprocal_input_string(&reciprocal)?, text);
        Ok(())
    }

    #[test]
    fn parses_reciprocal_cell_block() -> crate::Result<()> {
        let reciprocal = ReciprocalInput::parse_str(
            "reciprocal.inp",
            concat!(
                "ispace\n",
                "   0\n",
                "lattice vectors  (in A, in Carthesian coordinates)\n",
                "      1.00000      0.00000      0.00000\n",
                "      0.00000      1.00000      0.00000\n",
                "      0.00000      0.00000      1.00000\n",
                "Volume scaling factor (A^3); eimag; core hole\n",
                "     -1.00000      0.00000      1.00000\n",
                "lattice type  (P,I,F,R,B,CXY,CYZ,CXZ)\n",
                "P      P1          1\n",
                "#atoms in unit cell ; position absorber ; corehole?\n",
                "   2   1   1\n",
                "# k-points total/x/y/z ; ktype; use symmetry?\n",
                "8 2 2 2 0 T\n",
                "ppos\n",
                "      0.00000      0.00000      0.00000\n",
                "      0.50000      0.50000      0.50000\n",
                "ppot\n",
                "0 1\n",
                "label\n",
                "Cu Zn\n",
                "streta,strgmax,strrmax\n",
                "      0.10000      2.00000      3.00000\n",
            ),
        )?;
        let cell = reciprocal.cell.ok_or_else(|| crate::IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: "expected reciprocal cell".to_string(),
        })?;

        assert_eq!(reciprocal.ispace, 0);
        assert_eq!(cell.atom_count, 2);
        assert_eq!(cell.k_mesh.total, 8);
        assert!(cell.k_mesh.use_symmetry);
        assert_eq!(cell.potentials, vec![0, 1]);
        assert_eq!(cell.labels, vec!["Cu".to_string(), "Zn".to_string()]);
        assert!(
            reciprocal_input_string(&ReciprocalInput {
                ispace: 0,
                cell: Some(cell)
            })?
            .contains("# k-points total/x/y/z ; ktype; use symmetry?\n")
        );
        Ok(())
    }
}
