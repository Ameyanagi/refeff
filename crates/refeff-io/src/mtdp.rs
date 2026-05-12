//! FEFF muffin-tin density and potential text format.
//!
//! This module ports the read/write layout from `POT/m_mtdp.f90`. FEFF writes
//! every scalar with either `i12` or `1p,e20.10`, traversing matrix data in
//! Fortran column-major order. The Rust API stores atom coordinates as
//! `(atom, xyz)` and radial tables as `(radial, atom_or_empty_sphere)`, then
//! emits and parses the FEFF-compatible scalar stream explicitly.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, ShapeBuilder};

use crate::error::{IoError, Result};
use crate::format::fortran_exp;

/// FEFF `Mtdp_Data_Type` data from `POT/m_mtdp.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct MtdpData {
    /// Number of radial grid points `nR`.
    pub radial_count: usize,
    /// Atomic numbers for muffin-tin atoms, `At_AN`.
    pub atomic_numbers: Array1<usize>,
    /// Atom coordinates as `(atom, xyz)`, matching the values in `At_XYZ`.
    pub atom_coordinates: Array2<f64>,
    /// Muffin-tin radii `At_R`.
    pub atom_radii: Array1<f64>,
    /// 1-based muffin-tin radius indices `At_iR`.
    pub atom_radius_indices: Array1<usize>,
    /// Electron density inside each muffin tin as `(radial, atom)`, `At_Den`.
    pub atom_density: Array2<f64>,
    /// Potential inside each muffin tin as `(radial, atom)`, `At_Pot`.
    pub atom_potential: Array2<f64>,
    /// Empty-sphere coordinates as `(empty_sphere, xyz)`, `ESph_XYZ`.
    pub empty_sphere_coordinates: Array2<f64>,
    /// Empty-sphere radii `ESph_R`.
    pub empty_sphere_radii: Array1<f64>,
    /// 1-based empty-sphere radius indices `ESph_iR`.
    pub empty_sphere_radius_indices: Array1<usize>,
    /// Electron density inside each empty sphere as `(radial, empty_sphere)`, `ESph_Den`.
    pub empty_sphere_density: Array2<f64>,
    /// Potential inside each empty sphere as `(radial, empty_sphere)`, `ESph_Pot`.
    pub empty_sphere_potential: Array2<f64>,
    /// Interstitial potential `V_Int`.
    pub interstitial_potential: f64,
    /// HOMO energy `V_HOMO`.
    pub homo_energy: f64,
    /// LUMO energy `V_LUMO`.
    pub lumo_energy: f64,
}

/// Render FEFF `m_mtdp` text.
pub fn mtdp_string(data: &MtdpData) -> Result<String> {
    validate_mtdp(data)?;

    let mut out = String::new();
    write_i12(&mut out, data.radial_count)?;
    write_i12(&mut out, data.atomic_numbers.len())?;

    for &atomic_number in &data.atomic_numbers {
        write_i12(&mut out, atomic_number)?;
    }
    for atom in 0..data.atomic_numbers.len() {
        for axis in 0..3 {
            write_e20_10(&mut out, data.atom_coordinates[(atom, axis)])?;
        }
    }
    for &radius in &data.atom_radii {
        write_e20_10(&mut out, radius)?;
    }
    for &radius_index in &data.atom_radius_indices {
        write_i12(&mut out, radius_index)?;
    }
    write_radial_columns(&mut out, data.atom_density.view())?;
    write_radial_columns(&mut out, data.atom_potential.view())?;

    let empty_count = data.empty_sphere_radii.len();
    write_i12(&mut out, empty_count)?;
    for empty_sphere in 0..empty_count {
        for axis in 0..3 {
            write_e20_10(
                &mut out,
                data.empty_sphere_coordinates[(empty_sphere, axis)],
            )?;
        }
    }
    for &radius in &data.empty_sphere_radii {
        write_e20_10(&mut out, radius)?;
    }
    for &radius_index in &data.empty_sphere_radius_indices {
        write_i12(&mut out, radius_index)?;
    }
    write_radial_columns(&mut out, data.empty_sphere_density.view())?;
    write_radial_columns(&mut out, data.empty_sphere_potential.view())?;

    write_e20_10(&mut out, data.interstitial_potential)?;
    write_e20_10(&mut out, data.homo_energy)?;
    write_e20_10(&mut out, data.lumo_energy)?;
    Ok(out)
}

/// Parse FEFF `m_mtdp` text.
pub fn parse_mtdp(text: &str) -> Result<MtdpData> {
    let mut tokens = MtdpTokens::new(text);
    let radial_count = tokens.usize("nR")?;
    let atom_count = tokens.usize("nAt")?;

    let atomic_numbers = tokens.usize_array("At_AN", atom_count)?;
    let atom_coordinates = tokens.real_matrix_row_major("At_XYZ", atom_count, 3)?;
    let atom_radii = tokens.real_array("At_R", atom_count)?;
    let atom_radius_indices = tokens.usize_array("At_iR", atom_count)?;
    let atom_density = tokens.real_matrix_fortran("At_Den", radial_count, atom_count)?;
    let atom_potential = tokens.real_matrix_fortran("At_Pot", radial_count, atom_count)?;

    let empty_count = tokens.usize("nESph")?;
    let empty_sphere_coordinates = tokens.real_matrix_row_major("ESph_XYZ", empty_count, 3)?;
    let empty_sphere_radii = tokens.real_array("ESph_R", empty_count)?;
    let empty_sphere_radius_indices = tokens.usize_array("ESph_iR", empty_count)?;
    let empty_sphere_density = tokens.real_matrix_fortran("ESph_Den", radial_count, empty_count)?;
    let empty_sphere_potential =
        tokens.real_matrix_fortran("ESph_Pot", radial_count, empty_count)?;

    let interstitial_potential = tokens.real("V_Int")?;
    let homo_energy = tokens.real("V_HOMO")?;
    let lumo_energy = tokens.real("V_LUMO")?;
    tokens.finish()?;

    let data = MtdpData {
        radial_count,
        atomic_numbers,
        atom_coordinates,
        atom_radii,
        atom_radius_indices,
        atom_density,
        atom_potential,
        empty_sphere_coordinates,
        empty_sphere_radii,
        empty_sphere_radius_indices,
        empty_sphere_density,
        empty_sphere_potential,
        interstitial_potential,
        homo_energy,
        lumo_energy,
    };
    validate_mtdp(&data)?;
    Ok(data)
}

/// Write FEFF `m_mtdp` text to a file.
pub fn write_mtdp(path: impl AsRef<Path>, data: &MtdpData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, mtdp_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `m_mtdp` text from a file.
pub fn read_mtdp(path: impl AsRef<Path>) -> Result<MtdpData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_mtdp(&text)
}

fn validate_mtdp(data: &MtdpData) -> Result<()> {
    let atom_count = data.atomic_numbers.len();
    let empty_count = data.empty_sphere_radii.len();
    validate_len("At_R", data.atom_radii.len(), atom_count)?;
    validate_len("At_iR", data.atom_radius_indices.len(), atom_count)?;
    validate_len(
        "ESph_iR",
        data.empty_sphere_radius_indices.len(),
        empty_count,
    )?;
    validate_shape("At_XYZ", data.atom_coordinates.dim(), atom_count, 3)?;
    validate_shape(
        "At_Den",
        data.atom_density.dim(),
        data.radial_count,
        atom_count,
    )?;
    validate_shape(
        "At_Pot",
        data.atom_potential.dim(),
        data.radial_count,
        atom_count,
    )?;
    validate_shape(
        "ESph_XYZ",
        data.empty_sphere_coordinates.dim(),
        empty_count,
        3,
    )?;
    validate_shape(
        "ESph_Den",
        data.empty_sphere_density.dim(),
        data.radial_count,
        empty_count,
    )?;
    validate_shape(
        "ESph_Pot",
        data.empty_sphere_potential.dim(),
        data.radial_count,
        empty_count,
    )?;

    for (field, value) in [
        ("V_Int", data.interstitial_potential),
        ("V_HOMO", data.homo_energy),
        ("V_LUMO", data.lumo_energy),
    ] {
        validate_finite(field, value)?;
    }
    for (field, values) in [
        ("At_XYZ", data.atom_coordinates.view()),
        ("At_Den", data.atom_density.view()),
        ("At_Pot", data.atom_potential.view()),
        ("ESph_XYZ", data.empty_sphere_coordinates.view()),
        ("ESph_Den", data.empty_sphere_density.view()),
        ("ESph_Pot", data.empty_sphere_potential.view()),
    ] {
        for &value in values {
            validate_finite(field, value)?;
        }
    }
    for (field, values) in [
        ("At_R", data.atom_radii.view()),
        ("ESph_R", data.empty_sphere_radii.view()),
    ] {
        for &value in values {
            validate_finite(field, value)?;
        }
    }
    Ok(())
}

fn validate_len(field: &'static str, len: usize, expected: usize) -> Result<()> {
    if len == expected {
        Ok(())
    } else {
        Err(IoError::MtdpLength {
            field,
            len,
            expected,
        })
    }
}

fn validate_shape(
    field: &'static str,
    (rows, cols): (usize, usize),
    expected_rows: usize,
    expected_cols: usize,
) -> Result<()> {
    if rows == expected_rows && cols == expected_cols {
        Ok(())
    } else {
        Err(IoError::MtdpShape {
            field,
            rows,
            cols,
            expected_rows,
            expected_cols,
        })
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidMtdp {
            field,
            message: format!("value must be finite, got {value}"),
        })
    }
}

fn write_i12(out: &mut String, value: usize) -> Result<()> {
    let signed = i64::try_from(value).map_err(|_| IoError::InvalidMtdp {
        field: "integer",
        message: format!("value {value} does not fit FEFF i12 integer output"),
    })?;
    if signed > 999_999_999_999 {
        return Err(IoError::InvalidMtdp {
            field: "integer",
            message: format!("value {value} does not fit FEFF i12 integer output"),
        });
    }
    writeln!(out, "{signed:>12}")?;
    Ok(())
}

fn write_e20_10(out: &mut String, value: f64) -> Result<()> {
    validate_finite("real", value)?;
    writeln!(out, "{}", fortran_exp(value, 20, 10))?;
    Ok(())
}

fn write_radial_columns(out: &mut String, values: ndarray::ArrayView2<'_, f64>) -> Result<()> {
    for column in 0..values.ncols() {
        for row in 0..values.nrows() {
            write_e20_10(out, values[(row, column)])?;
        }
    }
    Ok(())
}

struct MtdpTokens<'a> {
    tokens: Vec<&'a str>,
    position: usize,
}

impl<'a> MtdpTokens<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            tokens: text.split_whitespace().collect(),
            position: 0,
        }
    }

    fn finish(self) -> Result<()> {
        if self.position == self.tokens.len() {
            Ok(())
        } else {
            Err(IoError::MtdpTrailingTokens {
                count: self.tokens.len() - self.position,
            })
        }
    }

    fn next(&mut self, field: &'static str) -> Result<&'a str> {
        let token = self
            .tokens
            .get(self.position)
            .copied()
            .ok_or(IoError::MtdpMissing { field })?;
        self.position += 1;
        Ok(token)
    }

    fn usize(&mut self, field: &'static str) -> Result<usize> {
        let token = self.next(field)?;
        token.parse::<usize>().map_err(|_| IoError::MtdpParse {
            field,
            token: token.to_string(),
        })
    }

    fn real(&mut self, field: &'static str) -> Result<f64> {
        let token = self.next(field)?;
        let value = token.parse::<f64>().map_err(|_| IoError::MtdpParse {
            field,
            token: token.to_string(),
        })?;
        validate_finite(field, value)?;
        Ok(value)
    }

    fn usize_array(&mut self, field: &'static str, len: usize) -> Result<Array1<usize>> {
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.usize(field)?);
        }
        Ok(Array1::from_vec(values))
    }

    fn real_array(&mut self, field: &'static str, len: usize) -> Result<Array1<f64>> {
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.real(field)?);
        }
        Ok(Array1::from_vec(values))
    }

    fn real_matrix_row_major(
        &mut self,
        field: &'static str,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<f64>> {
        let count = checked_count(field, rows, cols)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.real(field)?);
        }
        Array2::from_shape_vec((rows, cols), values).map_err(|_| IoError::MtdpShape {
            field,
            rows: count,
            cols: 1,
            expected_rows: rows,
            expected_cols: cols,
        })
    }

    fn real_matrix_fortran(
        &mut self,
        field: &'static str,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<f64>> {
        let count = checked_count(field, rows, cols)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.real(field)?);
        }
        Array2::from_shape_vec((rows, cols).f(), values).map_err(|_| IoError::MtdpShape {
            field,
            rows: count,
            cols: 1,
            expected_rows: rows,
            expected_cols: cols,
        })
    }
}

fn checked_count(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols).ok_or(IoError::InvalidMtdp {
        field,
        message: "matrix element count overflowed".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MTDP_ORACLE: &str = "           2\n           2\n          29\n           8\n    1.0000000000E-01\n   -2.0000000000E-01\n    3.0000000000E-01\n    1.1000000000E+00\n    1.2000000000E+00\n   -1.3000000000E+00\n    4.5000000000E-01\n    6.7000000000E-01\n          12\n          15\n    1.0000000000E-03\n    2.0000000000E-03\n    3.0000000000E-03\n    4.0000000000E-03\n   -1.0000000000E+00\n   -2.0000000000E+00\n   -3.0000000000E+00\n   -4.0000000000E+00\n           1\n    2.1000000000E+00\n    2.2000000000E+00\n    2.3000000000E+00\n    2.5000000000E-01\n           9\n    5.0000000000E-03\n    6.0000000000E-03\n   -5.0000000000E+00\n   -6.0000000000E+00\n   -7.5000000000E-01\n   -1.2000000000E-01\n    3.4000000000E-01\n";

    #[test]
    fn writes_mtdp_like_feff_reference() -> Result<()> {
        assert_eq!(mtdp_string(&sample_mtdp_data())?, MTDP_ORACLE);
        Ok(())
    }

    #[test]
    fn parses_mtdp_reference_text() -> Result<()> {
        assert_eq!(parse_mtdp(MTDP_ORACLE)?, sample_mtdp_data());
        Ok(())
    }

    #[test]
    fn rejects_invalid_mtdp_shapes_and_tokens() {
        let mut bad = sample_mtdp_data();
        bad.atom_coordinates = Array2::zeros((2, 2));
        assert!(matches!(
            mtdp_string(&bad),
            Err(IoError::MtdpShape {
                field: "At_XYZ",
                rows: 2,
                cols: 2,
                expected_rows: 2,
                expected_cols: 3,
            })
        ));

        assert!(matches!(
            parse_mtdp("2 1 not-an-int"),
            Err(IoError::MtdpParse { field: "At_AN", .. })
        ));
    }

    fn sample_mtdp_data() -> MtdpData {
        let atom_coordinates = Array2::from_shape_fn((2, 3), |(atom, axis)| match (atom, axis) {
            (0, 0) => 0.1,
            (0, 1) => -0.2,
            (0, 2) => 0.3,
            (1, 0) => 1.1,
            (1, 1) => 1.2,
            (1, 2) => -1.3,
            _ => 0.0,
        });
        let atom_density = Array2::from_shape_fn((2, 2), |(radial, atom)| {
            1.0e-3 + (atom * 2 + radial) as f64 * 1.0e-3
        });
        let atom_potential =
            Array2::from_shape_fn((2, 2), |(radial, atom)| -1.0 - (atom * 2 + radial) as f64);
        let empty_sphere_coordinates = Array2::from_shape_fn((1, 3), |(_, axis)| match axis {
            0 => 2.1,
            1 => 2.2,
            2 => 2.3,
            _ => 0.0,
        });
        let empty_sphere_density =
            Array2::from_shape_fn((2, 1), |(radial, _)| 5.0e-3 + radial as f64 * 1.0e-3);
        let empty_sphere_potential =
            Array2::from_shape_fn((2, 1), |(radial, _)| -5.0 - radial as f64);

        MtdpData {
            radial_count: 2,
            atomic_numbers: Array1::from_vec(vec![29, 8]),
            atom_coordinates,
            atom_radii: Array1::from_vec(vec![0.45, 0.67]),
            atom_radius_indices: Array1::from_vec(vec![12, 15]),
            atom_density,
            atom_potential,
            empty_sphere_coordinates,
            empty_sphere_radii: Array1::from_vec(vec![0.25]),
            empty_sphere_radius_indices: Array1::from_vec(vec![9]),
            empty_sphere_density,
            empty_sphere_potential,
            interstitial_potential: -0.75,
            homo_energy: -0.12,
            lumo_energy: 0.34,
        }
    }
}
