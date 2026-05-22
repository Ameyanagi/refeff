use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

use super::common::{DENSITY_FILENAME_WIDTH, fortran_fixed_string};
use super::types::{
    BandEnergyMesh, BandInput, DensityAxis, DensityGrid, DensityGridKind, DensityInput,
    FullSpectrumInput, OpconsInput, ReciprocalCell, ReciprocalInput, ReciprocalKMesh,
};

pub(super) struct ControlParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> ControlParser<'a> {
    pub(super) fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    pub(super) fn parse_band(&mut self) -> Result<BandInput> {
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

    pub(super) fn parse_density(&mut self) -> Result<DensityInput> {
        let mut grids = Vec::new();
        while let Some((line_number, line)) = self.next_density_line()? {
            grids.push(self.parse_density_grid(line_number, line)?);
        }
        Ok(DensityInput { grids })
    }

    pub(super) fn parse_fullspectrum(&mut self) -> Result<FullSpectrumInput> {
        self.next_line("FULLSPECTRUM header")?;
        Ok(FullSpectrumInput {
            m_full_spectrum: self.parse_single("FULLSPECTRUM flag line")?,
        })
    }

    pub(super) fn parse_opcons(&mut self) -> Result<OpconsInput> {
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

    pub(super) fn parse_reciprocal(&mut self) -> Result<ReciprocalInput> {
        self.next_line("RECIPROCAL ispace header")?;
        let ispace = self.parse_single("RECIPROCAL ispace line")?;
        let cell = if ispace == 0 {
            Some(self.parse_reciprocal_cell()?)
        } else {
            None
        };

        Ok(ReciprocalInput { ispace, cell })
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
            filename: fortran_fixed_string(fields[1], DENSITY_FILENAME_WIDTH),
            origin: [
                parse_field(&self.source, line_number, fields[2])?,
                parse_field(&self.source, line_number, fields[3])?,
                parse_field(&self.source, line_number, fields[4])?,
            ],
            core: fields.get(5).is_some_and(|field| *field == "core"),
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

pub(super) fn fields(line: &str) -> Vec<&str> {
    // FEFF `bwords`: blanks/tabs separate words; commas also separate and
    // leading or consecutive commas produce blank fields.
    let mut fields = Vec::new();
    let mut between_words = true;
    let mut comma_found = true;
    let mut start = 0;

    for (index, byte) in line.bytes().enumerate() {
        match byte {
            b' ' | b'\t' => {
                if !between_words {
                    fields.push(&line[start..index]);
                    between_words = true;
                    comma_found = false;
                }
            }
            b',' => {
                if between_words {
                    if comma_found {
                        fields.push("");
                    }
                } else {
                    fields.push(&line[start..index]);
                    between_words = true;
                }
                comma_found = true;
            }
            _ => {
                if between_words {
                    between_words = false;
                    start = index;
                }
            }
        }
    }

    if !between_words {
        fields.push(&line[start..]);
    }
    fields
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
