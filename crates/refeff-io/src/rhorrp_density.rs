//! FEFF RHORRP ASCII density-output codec.
//!
//! `RHORRP/rhorrp.f90` writes text density grids as Cartesian coordinates and
//! density values. In its current diagnostic mode it also appends nearest-atom
//! displacement and atom/potential indices. Coordinates and density are written
//! in Angstrom units; the nearest-atom displacement follows FEFF's diagnostic
//! output and remains in Bohr.

use std::fmt::Write as _;
use std::path::Path;
use std::str::SplitWhitespace;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::control_input::FEFF_BOHR_ANGSTROM;
use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const RHORRP_BASIC_ROW_WIDTH: usize = 4;
const RHORRP_NEAREST_ROW_WIDTH: usize = 9;
const RHORRP_COORDINATE_COLUMNS: usize = 3;

/// Optional nearest-atom diagnostic columns from RHORRP text output.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpNearestAtomColumns {
    /// Displacement from the nearest atom to the grid point, as `(point, xyz)`.
    pub displacement_bohr: Array2<f64>,
    /// FEFF text-output atom index after `iat = iat - 1`, so absorber is zero.
    pub atom_indices: Array1<usize>,
    /// Potential index `iph` for the nearest atom.
    pub potential_indices: Array1<usize>,
}

/// Parsed RHORRP ASCII density output.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpDensityTextData {
    /// Cartesian grid coordinates in Angstroms as `(point, xyz)`.
    pub points_angstrom: Array2<f64>,
    /// Charge density in inverse cubic Angstroms.
    pub density_per_angstrom3: Array1<f64>,
    /// Optional nearest-atom diagnostic columns.
    pub nearest: Option<RhorrpNearestAtomColumns>,
}

/// Bohr-unit RHORRP density data ready for FEFF text-output conversion.
#[derive(Debug, Clone)]
pub struct RhorrpDensityTextBohrInput<'a> {
    /// Grid points in Bohr as `(xyz, point)`, matching FEFF `points(3, totpts)`.
    pub points_bohr: ArrayView2<'a, f64>,
    /// Charge density in inverse cubic Bohr, matching RHORRP calculation units.
    pub density_per_bohr3: ArrayView1<'a, f64>,
    /// Optional nearest-atom diagnostic columns. Displacements remain in Bohr.
    pub nearest: Option<RhorrpNearestAtomColumns>,
}

impl RhorrpDensityTextData {
    /// Number of grid points in this density output.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.density_per_angstrom3.len()
    }

    /// Whether this output includes RHORRP nearest-atom diagnostics.
    #[must_use]
    pub fn has_nearest_atom_columns(&self) -> bool {
        self.nearest.is_some()
    }
}

/// Convert RHORRP Bohr-unit calculation output to FEFF ASCII density data.
///
/// FEFF `calculate_density` multiplies grid coordinates by `bohr`, divides
/// density by `bohr**3`, and leaves nearest-atom displacement diagnostics in
/// Bohr. The returned data can be passed to [`rhorrp_density_text_string`] or
/// [`write_rhorrp_density_text`].
pub fn rhorrp_density_text_from_bohr(
    input: RhorrpDensityTextBohrInput<'_>,
) -> Result<RhorrpDensityTextData> {
    let (coordinate_rows, point_count) = input.points_bohr.dim();
    if coordinate_rows != RHORRP_COORDINATE_COLUMNS {
        return Err(IoError::RhorrpDensityShape {
            field: "points_bohr",
            rows: coordinate_rows,
            columns: point_count,
            expected: "3xN",
        });
    }
    validate_length(
        "density_per_bohr3",
        input.density_per_bohr3.len(),
        point_count,
    )?;

    let coordinate_scale = FEFF_BOHR_ANGSTROM;
    let density_scale = 1.0 / (FEFF_BOHR_ANGSTROM * FEFF_BOHR_ANGSTROM * FEFF_BOHR_ANGSTROM);

    let mut points_angstrom = Array2::zeros((point_count, RHORRP_COORDINATE_COLUMNS));
    for point in 0..point_count {
        for coordinate in 0..RHORRP_COORDINATE_COLUMNS {
            points_angstrom[(point, coordinate)] =
                input.points_bohr[(coordinate, point)] * coordinate_scale;
        }
    }
    let density_per_angstrom3 = input
        .density_per_bohr3
        .mapv(|density| density * density_scale);

    let data = RhorrpDensityTextData {
        points_angstrom,
        density_per_angstrom3,
        nearest: input.nearest,
    };
    validate_rhorrp_density_text(&data)?;
    Ok(data)
}

/// Render FEFF-compatible RHORRP ASCII density output.
pub fn rhorrp_density_text_string(data: &RhorrpDensityTextData) -> Result<String> {
    validate_rhorrp_density_text(data)?;

    let row_capacity = if data.nearest.is_some() { 112 } else { 56 };
    let mut out = String::with_capacity(data.point_count().saturating_mul(row_capacity));
    for row in 0..data.point_count() {
        write_real_fields(
            &mut out,
            [
                data.points_angstrom[(row, 0)],
                data.points_angstrom[(row, 1)],
                data.points_angstrom[(row, 2)],
                data.density_per_angstrom3[row],
            ],
        )?;

        if let Some(nearest) = &data.nearest {
            out.push(' ');
            write_fortran_exp(&mut out, nearest.displacement_bohr[(row, 0)], 12, 5)?;
            out.push(' ');
            write_fortran_exp(&mut out, nearest.displacement_bohr[(row, 1)], 12, 5)?;
            out.push(' ');
            write_fortran_exp(&mut out, nearest.displacement_bohr[(row, 2)], 12, 5)?;
            write!(
                out,
                "  {:>2} {:>1}",
                nearest.atom_indices[row], nearest.potential_indices[row],
            )?;
        }
        writeln!(out)?;
    }

    Ok(out)
}

/// Parse FEFF RHORRP ASCII density output.
pub fn parse_rhorrp_density_text(text: &str) -> Result<RhorrpDensityTextData> {
    let mut points = Vec::new();
    let mut density = Vec::new();
    let mut displacement = Vec::new();
    let mut atom_indices = Vec::new();
    let mut potential_indices = Vec::new();
    let mut expected_width: Option<usize> = None;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || is_comment_line(line) {
            continue;
        }

        let mut tokens = line.split_whitespace();
        points.push(parse_rhorrp_f64(
            line_number,
            "x",
            next_rhorrp_field(&mut tokens, line_number, line, "x")?,
        )?);
        points.push(parse_rhorrp_f64(
            line_number,
            "y",
            next_rhorrp_field(&mut tokens, line_number, line, "y")?,
        )?);
        points.push(parse_rhorrp_f64(
            line_number,
            "z",
            next_rhorrp_field(&mut tokens, line_number, line, "z")?,
        )?);
        density.push(parse_rhorrp_f64(
            line_number,
            "density",
            next_rhorrp_field(&mut tokens, line_number, line, "density")?,
        )?);

        let width = if let Some(dx) = tokens.next() {
            displacement.push(parse_rhorrp_f64(line_number, "dx", dx)?);
            displacement.push(parse_rhorrp_f64(
                line_number,
                "dy",
                next_rhorrp_field(&mut tokens, line_number, line, "dy")?,
            )?);
            displacement.push(parse_rhorrp_f64(
                line_number,
                "dz",
                next_rhorrp_field(&mut tokens, line_number, line, "dz")?,
            )?);
            atom_indices.push(parse_rhorrp_usize(
                line_number,
                "atom index",
                next_rhorrp_field(&mut tokens, line_number, line, "atom index")?,
            )?);
            potential_indices.push(parse_rhorrp_usize(
                line_number,
                "potential index",
                next_rhorrp_field(&mut tokens, line_number, line, "potential index")?,
            )?);
            if tokens.next().is_some() {
                return Err(IoError::RhorrpDensityRowWidth {
                    line: line_number,
                    actual: RHORRP_NEAREST_ROW_WIDTH + 1 + tokens.count(),
                    expected: "4 or 9".to_string(),
                });
            }
            RHORRP_NEAREST_ROW_WIDTH
        } else {
            RHORRP_BASIC_ROW_WIDTH
        };

        if let Some(expected) = expected_width {
            if width != expected {
                return Err(IoError::RhorrpDensityRowWidth {
                    line: line_number,
                    actual: width,
                    expected: expected.to_string(),
                });
            }
        } else {
            expected_width = Some(width);
        }
    }

    let point_count = density.len();
    let points_angstrom = Array2::from_shape_vec((point_count, RHORRP_COORDINATE_COLUMNS), points)
        .map_err(|_| IoError::InvalidRhorrpDensity {
            field: "points_angstrom",
            message: "coordinate payload did not match RHORRP table shape".to_string(),
        })?;
    let density_per_angstrom3 = Array1::from_vec(density);
    let nearest = if expected_width == Some(RHORRP_NEAREST_ROW_WIDTH) {
        Some(RhorrpNearestAtomColumns {
            displacement_bohr: Array2::from_shape_vec(
                (point_count, RHORRP_COORDINATE_COLUMNS),
                displacement,
            )
            .map_err(|_| IoError::InvalidRhorrpDensity {
                field: "displacement_bohr",
                message: "nearest-atom displacement payload did not match RHORRP table shape"
                    .to_string(),
            })?,
            atom_indices: Array1::from_vec(atom_indices),
            potential_indices: Array1::from_vec(potential_indices),
        })
    } else {
        None
    };

    let data = RhorrpDensityTextData {
        points_angstrom,
        density_per_angstrom3,
        nearest,
    };
    validate_rhorrp_density_text(&data)?;
    Ok(data)
}

/// Write FEFF RHORRP ASCII density output to a file.
pub fn write_rhorrp_density_text(
    path: impl AsRef<Path>,
    data: &RhorrpDensityTextData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhorrp_density_text_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Read FEFF RHORRP ASCII density output from a file.
pub fn read_rhorrp_density_text(path: impl AsRef<Path>) -> Result<RhorrpDensityTextData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rhorrp_density_text(&text)
}

fn write_real_fields<const N: usize>(out: &mut String, values: [f64; N]) -> Result<()> {
    if let Some((first, rest)) = values.split_first() {
        write_fortran_exp(out, *first, 12, 5)?;
        for value in rest {
            out.push(' ');
            write_fortran_exp(out, *value, 12, 5)?;
        }
    }
    Ok(())
}

fn validate_rhorrp_density_text(data: &RhorrpDensityTextData) -> Result<()> {
    let (rows, columns) = data.points_angstrom.dim();
    if columns != RHORRP_COORDINATE_COLUMNS {
        return Err(IoError::RhorrpDensityShape {
            field: "points_angstrom",
            rows,
            columns,
            expected: "Nx3",
        });
    }
    validate_length(
        "density_per_angstrom3",
        data.density_per_angstrom3.len(),
        rows,
    )?;

    for (index, value) in data.points_angstrom.iter().enumerate() {
        validate_finite("points_angstrom", *value, index)?;
    }
    for (index, value) in data.density_per_angstrom3.iter().enumerate() {
        validate_finite("density_per_angstrom3", *value, index)?;
    }

    if let Some(nearest) = &data.nearest {
        let (displacement_rows, displacement_columns) = nearest.displacement_bohr.dim();
        if displacement_rows != rows || displacement_columns != RHORRP_COORDINATE_COLUMNS {
            return Err(IoError::RhorrpDensityShape {
                field: "displacement_bohr",
                rows: displacement_rows,
                columns: displacement_columns,
                expected: "Nx3 matching points_angstrom",
            });
        }
        validate_length("atom_indices", nearest.atom_indices.len(), rows)?;
        validate_length("potential_indices", nearest.potential_indices.len(), rows)?;
        for (index, value) in nearest.displacement_bohr.iter().enumerate() {
            validate_finite("displacement_bohr", *value, index)?;
        }
    }

    Ok(())
}

fn validate_length(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::RhorrpDensityLength {
            field,
            actual,
            expected,
        })
    }
}

fn validate_finite(field: &'static str, value: f64, index: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidRhorrpDensity {
            field,
            message: format!("value at index {index} is not finite: {value}"),
        })
    }
}

fn parse_rhorrp_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    let normalized;
    let candidate = if token.contains('D') || token.contains('d') {
        normalized = token.replace(['D', 'd'], "E");
        normalized.as_str()
    } else {
        token
    };
    candidate
        .parse::<f64>()
        .map_err(|_| IoError::RhorrpDensityParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn parse_rhorrp_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| IoError::RhorrpDensityParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn next_rhorrp_field<'a>(
    tokens: &mut SplitWhitespace<'a>,
    line: usize,
    original_line: &str,
    expected_field: &'static str,
) -> Result<&'a str> {
    tokens.next().ok_or_else(|| IoError::RhorrpDensityRowWidth {
        line,
        actual: original_line.split_whitespace().count(),
        expected: format!("4 or 9 fields including {expected_field}"),
    })
}

fn is_comment_line(line: &str) -> bool {
    matches!(line.as_bytes().first(), Some(b'#' | b'!' | b'*'))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_REFERENCE: &str = concat!(
        " 0.00000E+00 -2.50000E-01  1.50000E+00  1.23457E-04\n",
        " 1.23457E+00  2.50000E-03 -9.87654E+00  2.50000E+00\n",
        "-1.20000E+01  3.33333E-01  4.20000E+00 -3.75000E-02\n",
    );

    const NEAREST_REFERENCE: &str = concat!(
        " 0.00000E+00 -2.50000E-01  1.50000E+00  1.23457E-04  1.00000E-01 -2.00000E-01  3.00000E-01   0 2\n",
        " 1.23457E+00  2.50000E-03 -9.87654E+00  2.50000E+00 -1.00000E-03  2.25000E+00 -3.50000E+00  12 0\n",
        "-1.20000E+01  3.33333E-01  4.20000E+00 -3.75000E-02  4.40000E+00 -5.50000E+00  6.60000E+00   7 5\n",
    );

    #[test]
    fn renders_basic_density_text_like_feff_reference() -> Result<()> {
        let data = reference_basic_data()?;

        assert_eq!(rhorrp_density_text_string(&data)?, BASIC_REFERENCE);

        let parsed = parse_rhorrp_density_text(BASIC_REFERENCE)?;
        assert!(!parsed.has_nearest_atom_columns());
        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.points_angstrom[(1, 0)], 1.23457);
        assert_eq!(parsed.density_per_angstrom3[2], -3.75e-2);
        Ok(())
    }

    #[test]
    fn renders_nearest_atom_density_text_like_feff_reference() -> Result<()> {
        let data = reference_nearest_data()?;

        assert_eq!(rhorrp_density_text_string(&data)?, NEAREST_REFERENCE);

        let parsed = parse_rhorrp_density_text(NEAREST_REFERENCE)?;
        assert_eq!(parsed.point_count(), 3);
        let Some(nearest) = parsed.nearest else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "nearest",
                message: "missing nearest-atom columns".to_string(),
            });
        };
        assert_eq!(nearest.displacement_bohr[(0, 2)], 0.3);
        assert_eq!(nearest.atom_indices[1], 12);
        assert_eq!(nearest.potential_indices[2], 5);
        Ok(())
    }

    #[test]
    fn rhorrp_density_text_roundtrips_files() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|source| IoError::Io {
            path: "rhorrp-density-tempdir".into(),
            source,
        })?;
        let path = dir.path().join("density.out");
        let data = reference_nearest_data()?;

        write_rhorrp_density_text(&path, &data)?;
        let parsed = read_rhorrp_density_text(&path)?;

        assert_eq!(parsed, parse_rhorrp_density_text(NEAREST_REFERENCE)?);
        Ok(())
    }

    #[test]
    fn rhorrp_density_text_rejects_bad_shapes_and_rows() {
        let bad_shape = RhorrpDensityTextData {
            points_angstrom: Array2::zeros((2, 2)),
            density_per_angstrom3: Array1::zeros(2),
            nearest: None,
        };
        assert!(matches!(
            rhorrp_density_text_string(&bad_shape),
            Err(IoError::RhorrpDensityShape {
                field: "points_angstrom",
                ..
            })
        ));

        let bad_len = RhorrpDensityTextData {
            points_angstrom: Array2::zeros((2, 3)),
            density_per_angstrom3: Array1::zeros(1),
            nearest: None,
        };
        assert!(matches!(
            rhorrp_density_text_string(&bad_len),
            Err(IoError::RhorrpDensityLength {
                field: "density_per_angstrom3",
                ..
            })
        ));

        assert!(matches!(
            parse_rhorrp_density_text("0 1 2 3\n0 1 2 3 4 5 6 7 8\n"),
            Err(IoError::RhorrpDensityRowWidth { line: 2, .. })
        ));
        assert!(matches!(
            parse_rhorrp_density_text("0 1 nope 3\n"),
            Err(IoError::RhorrpDensityParse { field: "z", .. })
        ));

        assert!(matches!(
            rhorrp_density_text_from_bohr(RhorrpDensityTextBohrInput {
                points_bohr: Array2::zeros((2, 3)).view(),
                density_per_bohr3: Array1::zeros(3).view(),
                nearest: None,
            }),
            Err(IoError::RhorrpDensityShape {
                field: "points_bohr",
                ..
            })
        ));

        assert!(matches!(
            rhorrp_density_text_from_bohr(RhorrpDensityTextBohrInput {
                points_bohr: Array2::zeros((3, 3)).view(),
                density_per_bohr3: Array1::zeros(2).view(),
                nearest: None,
            }),
            Err(IoError::RhorrpDensityLength {
                field: "density_per_bohr3",
                ..
            })
        ));
    }

    #[test]
    fn converts_bohr_density_text_like_feff_reference() -> Result<()> {
        let points_bohr = ndarray::arr2(&[[0.1, 1.5, -0.25], [-0.2, 0.0, 2.0], [0.3, 0.75, -1.0]]);
        let density_per_bohr3 = ndarray::arr1(&[0.5, 2.0, -0.125]);
        let data = rhorrp_density_text_from_bohr(RhorrpDensityTextBohrInput {
            points_bohr: points_bohr.view(),
            density_per_bohr3: density_per_bohr3.view(),
            nearest: None,
        })?;

        assert_close(data.points_angstrom[(0, 0)], 0.052_917_724_9);
        assert_close(data.points_angstrom[(0, 1)], -0.105_835_449_8);
        assert_close(data.points_angstrom[(0, 2)], 0.158_753_174_699_999_97);
        assert_close(data.points_angstrom[(1, 0)], 0.793_765_873_499_999_9);
        assert_close(data.points_angstrom[(1, 1)], 0.0);
        assert_close(data.points_angstrom[(1, 2)], 0.396_882_936_749_999_97);
        assert_close(data.points_angstrom[(2, 0)], -0.132_294_312_25);
        assert_close(data.points_angstrom[(2, 1)], 1.058_354_498);
        assert_close(data.points_angstrom[(2, 2)], -0.529_177_249);
        assert_close(data.density_per_angstrom3[0], 3.374_166_518_552_075_3);
        assert_close(data.density_per_angstrom3[1], 13.496_666_074_208_301);
        assert_close(data.density_per_angstrom3[2], -0.843_541_629_638_018_8);
        Ok(())
    }

    fn reference_basic_data() -> Result<RhorrpDensityTextData> {
        let points_angstrom = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0,
                -0.25,
                1.5,
                1.23456789,
                2.5e-3,
                -9.87654321,
                -12.0,
                0.333333333333,
                4.2,
            ],
        )
        .map_err(|_| IoError::InvalidRhorrpDensity {
            field: "points_angstrom",
            message: "test fixture has invalid coordinate shape".to_string(),
        })?;

        Ok(RhorrpDensityTextData {
            points_angstrom,
            density_per_angstrom3: Array1::from_vec(vec![1.23456789e-4, 2.5, -3.75e-2]),
            nearest: None,
        })
    }

    fn reference_nearest_data() -> Result<RhorrpDensityTextData> {
        let displacement_bohr = Array2::from_shape_vec(
            (3, 3),
            vec![0.1, -0.2, 0.3, -1.0e-3, 2.25, -3.5, 4.4, -5.5, 6.6],
        )
        .map_err(|_| IoError::InvalidRhorrpDensity {
            field: "displacement_bohr",
            message: "test fixture has invalid displacement shape".to_string(),
        })?;
        let mut data = reference_basic_data()?;
        data.nearest = Some(RhorrpNearestAtomColumns {
            displacement_bohr,
            atom_indices: Array1::from_vec(vec![0, 12, 7]),
            potential_indices: Array1::from_vec(vec![2, 0, 5]),
        });
        Ok(data)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
