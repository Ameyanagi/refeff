//! FEFF-compatible writers for potential output files.
//!
//! FEFF `wpot` emits `potXX.dat` files after the overlap-density stage. These
//! helpers preserve that text layout while accepting typed `ndarray` views from
//! the Rust numerical pipeline.

use std::collections::BTreeMap;

use ndarray::ArrayView2;

use crate::format::fortran_exp;
use crate::{IoError, Result};

const FEFF_WPOT_RADIAL_POINTS: usize = 251;
const FEFF_WPOT_RADIUS_LIMIT: f64 = 38.0;
const PI4: f64 = 4.0 * std::f64::consts::PI;

/// Inputs for rendering one FEFF `potXX.dat` file.
#[derive(Debug, Clone, Copy)]
pub struct PotentialDatInput<'a> {
    /// Unique potential index (`iph`) to render.
    pub potential_index: usize,
    /// Muffin-tin radial index (`imt(iph)`).
    pub muffin_tin_index: usize,
    /// Norman-radius radial index (`inrm(iph)`).
    pub norman_index: usize,
    /// Header title lines written by FEFF `wthead`.
    pub titles: &'a [String],
    /// Overlapped electron density, equivalent to FEFF `edens`.
    pub electron_density: ArrayView2<'a, f64>,
    /// Free-atom density, equivalent to FEFF `rho`.
    pub free_density: ArrayView2<'a, f64>,
    /// Overlapped Coulomb potential, equivalent to FEFF `vclap`.
    pub overlapped_coulomb: ArrayView2<'a, f64>,
    /// Free-atom Coulomb potential, equivalent to FEFF `vcoul`.
    pub free_coulomb: ArrayView2<'a, f64>,
    /// Overlapped total potential, equivalent to FEFF `vtot`.
    pub total_potential: ArrayView2<'a, f64>,
}

/// Inputs for rendering every `potXX.dat` file from one potential set.
#[derive(Debug, Clone, Copy)]
pub struct PotentialDatSetInput<'a> {
    /// Highest unique potential index (`nph`); files are rendered for `0..=nph`.
    pub highest_potential_index: usize,
    /// Muffin-tin radial indices indexed by potential.
    pub muffin_tin_indices: &'a [usize],
    /// Norman-radius radial indices indexed by potential.
    pub norman_indices: &'a [usize],
    /// Header title lines written by FEFF `wthead`.
    pub titles: &'a [String],
    /// Overlapped electron density, equivalent to FEFF `edens`.
    pub electron_density: ArrayView2<'a, f64>,
    /// Free-atom density, equivalent to FEFF `rho`.
    pub free_density: ArrayView2<'a, f64>,
    /// Overlapped Coulomb potential, equivalent to FEFF `vclap`.
    pub overlapped_coulomb: ArrayView2<'a, f64>,
    /// Free-atom Coulomb potential, equivalent to FEFF `vcoul`.
    pub free_coulomb: ArrayView2<'a, f64>,
    /// Overlapped total potential, equivalent to FEFF `vtot`.
    pub total_potential: ArrayView2<'a, f64>,
}

/// Return the FEFF filename for one unique-potential output.
#[must_use]
pub fn potential_dat_filename(potential_index: usize) -> String {
    format!("pot{potential_index:02}.dat")
}

/// Render FEFF-compatible `potXX.dat` content for one potential.
pub fn pot_dat_string(input: PotentialDatInput<'_>) -> Result<String> {
    let mut out = String::new();
    write_potential_dat(input, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `potXX.dat` content for all potentials.
pub fn potential_dat_outputs(input: PotentialDatSetInput<'_>) -> Result<BTreeMap<String, String>> {
    validate_potential_set(input)?;
    let mut outputs = BTreeMap::new();
    for potential_index in 0..=input.highest_potential_index {
        let muffin_tin_index = checked_index_value(
            "muffin_tin_indices",
            input.muffin_tin_indices,
            potential_index,
        )?;
        let norman_index =
            checked_index_value("norman_indices", input.norman_indices, potential_index)?;
        let content = pot_dat_string(PotentialDatInput {
            potential_index,
            muffin_tin_index,
            norman_index,
            titles: input.titles,
            electron_density: input.electron_density,
            free_density: input.free_density,
            overlapped_coulomb: input.overlapped_coulomb,
            free_coulomb: input.free_coulomb,
            total_potential: input.total_potential,
        })?;
        outputs.insert(potential_dat_filename(potential_index), content);
    }
    Ok(outputs)
}

/// Write FEFF-compatible `potXX.dat` content for one potential.
pub fn write_potential_dat(
    input: PotentialDatInput<'_>,
    out: &mut impl std::fmt::Write,
) -> Result<()> {
    validate_potential_input(input)?;

    for title in input.titles {
        writeln!(out, "# {}", trim_feff_title(title))?;
    }
    writeln!(
        out,
        " {potential_index:>4}{muffin_tin_index:>4}{norman_index:>4}  Unique potential, I_mt, I_norman.    Following data in atomic units.",
        potential_index = input.potential_index,
        muffin_tin_index = input.muffin_tin_index,
        norman_index = input.norman_index
    )?;
    writeln!(
        out,
        "  iph {potential_index:>12}",
        potential_index = input.potential_index
    )?;
    writeln!(
        out,
        "   i      r         vcoul        rho     ovrlp vcoul  ovrlp vtot  ovrlp rho"
    )?;

    for radial_index in 1..=wpot_output_radial_count() {
        let row = radial_index - 1;
        let radius = legacy_wpot_radius(radial_index);
        writeln!(
            out,
            " {radial_index:>4}{}{}{}{}{}{}",
            e12_4(radius),
            e12_4(input.free_coulomb[(row, input.potential_index)]),
            e12_4(input.free_density[(row, input.potential_index)] / PI4),
            e12_4(input.overlapped_coulomb[(row, input.potential_index)]),
            e12_4(input.total_potential[(row, input.potential_index)]),
            e12_4(input.electron_density[(row, input.potential_index)] / PI4)
        )?;
    }

    Ok(())
}

fn validate_potential_set(input: PotentialDatSetInput<'_>) -> Result<()> {
    ensure_i4("highest_potential_index", input.highest_potential_index)?;
    let required_len = checked_count("highest_potential_index", input.highest_potential_index)?;
    ensure_slice_len(
        "muffin_tin_indices",
        input.muffin_tin_indices.len(),
        required_len,
    )?;
    ensure_slice_len("norman_indices", input.norman_indices.len(), required_len)?;
    validate_matrix_shape(
        "electron_density",
        input.electron_density,
        input.highest_potential_index,
    )?;
    validate_matrix_shape(
        "free_density",
        input.free_density,
        input.highest_potential_index,
    )?;
    validate_matrix_shape(
        "overlapped_coulomb",
        input.overlapped_coulomb,
        input.highest_potential_index,
    )?;
    validate_matrix_shape(
        "free_coulomb",
        input.free_coulomb,
        input.highest_potential_index,
    )?;
    validate_matrix_shape(
        "total_potential",
        input.total_potential,
        input.highest_potential_index,
    )
}

fn validate_potential_input(input: PotentialDatInput<'_>) -> Result<()> {
    ensure_i4("potential_index", input.potential_index)?;
    ensure_i4("muffin_tin_index", input.muffin_tin_index)?;
    ensure_i4("norman_index", input.norman_index)?;
    validate_matrix_shape(
        "electron_density",
        input.electron_density,
        input.potential_index,
    )?;
    validate_matrix_shape("free_density", input.free_density, input.potential_index)?;
    validate_matrix_shape(
        "overlapped_coulomb",
        input.overlapped_coulomb,
        input.potential_index,
    )?;
    validate_matrix_shape("free_coulomb", input.free_coulomb, input.potential_index)?;
    validate_matrix_shape(
        "total_potential",
        input.total_potential,
        input.potential_index,
    )?;
    validate_matrix_values(
        "electron_density",
        input.electron_density,
        input.potential_index,
    )?;
    validate_matrix_values("free_density", input.free_density, input.potential_index)?;
    validate_matrix_values(
        "overlapped_coulomb",
        input.overlapped_coulomb,
        input.potential_index,
    )?;
    validate_matrix_values("free_coulomb", input.free_coulomb, input.potential_index)?;
    validate_matrix_values(
        "total_potential",
        input.total_potential,
        input.potential_index,
    )
}

fn validate_matrix_shape(
    field: &'static str,
    values: ArrayView2<'_, f64>,
    column: usize,
) -> Result<()> {
    let (rows, cols) = values.dim();
    let min_cols = checked_count(field, column)?;
    if rows < FEFF_WPOT_RADIAL_POINTS || cols < min_cols {
        return Err(IoError::PotentialOutputShape {
            field,
            rows,
            cols,
            min_rows: FEFF_WPOT_RADIAL_POINTS,
            min_cols,
        });
    }
    Ok(())
}

fn validate_matrix_values(
    field: &'static str,
    values: ArrayView2<'_, f64>,
    column: usize,
) -> Result<()> {
    for row in 0..wpot_output_radial_count() {
        if !values[(row, column)].is_finite() {
            return Err(IoError::InvalidPotentialOutput {
                field,
                message: format!("non-finite value at row {}, column {}", row + 1, column),
            });
        }
    }
    Ok(())
}

fn ensure_slice_len(field: &'static str, actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(IoError::InvalidPotentialOutput {
            field,
            message: format!("expected at least {required} entries, got {actual}"),
        });
    }
    Ok(())
}

fn checked_index_value(field: &'static str, values: &[usize], index: usize) -> Result<usize> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| IoError::InvalidPotentialOutput {
            field,
            message: format!("missing entry for potential index {index}"),
        })
}

fn ensure_i4(field: &'static str, value: usize) -> Result<()> {
    if value > 9_999 {
        return Err(IoError::InvalidPotentialOutput {
            field,
            message: format!("value {value} exceeds FEFF i4 output width"),
        });
    }
    Ok(())
}

fn checked_count(field: &'static str, zero_based_index: usize) -> Result<usize> {
    zero_based_index
        .checked_add(1)
        .ok_or_else(|| IoError::InvalidPotentialOutput {
            field,
            message: "zero-based index overflowed required count".to_string(),
        })
}

fn trim_feff_title(title: &str) -> &str {
    title.trim_end_matches(' ')
}

fn e12_4(value: f64) -> String {
    fortran_exp(value, 12, 4)
}

fn wpot_output_radial_count() -> usize {
    (1..=FEFF_WPOT_RADIAL_POINTS)
        .take_while(|radial_index| legacy_wpot_radius(*radial_index) <= FEFF_WPOT_RADIUS_LIMIT)
        .count()
}

fn legacy_wpot_radius(radial_index: usize) -> f64 {
    (-8.8 + (radial_index - 1) as f64 * 0.05).exp()
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::pot_output::{
        PotentialDatInput, PotentialDatSetInput, pot_dat_string, potential_dat_filename,
        potential_dat_outputs,
    };
    use crate::{IoError, Result};

    #[test]
    fn writes_potential_dat_like_feff_wpot_oracle() -> Result<()> {
        let state = sample_wpot_state();
        let actual = pot_dat_string(PotentialDatInput {
            potential_index: 0,
            muffin_tin_index: 12,
            norman_index: 40,
            titles: &state.titles,
            electron_density: state.electron_density.view(),
            free_density: state.free_density.view(),
            overlapped_coulomb: state.overlapped_coulomb.view(),
            free_coulomb: state.free_coulomb.view(),
            total_potential: state.total_potential.view(),
        })?;

        assert_eq!(actual.lines().count(), 255);
        assert_eq!(nth_line(&actual, 0)?, "# First title");
        assert_eq!(nth_line(&actual, 1)?, "# Second title with trailing blanks");
        assert_eq!(nth_line(&actual, 2)?, "# Third title 123");
        assert_eq!(
            nth_line(&actual, 3)?,
            "    0  12  40  Unique potential, I_mt, I_norman.    Following data in atomic units."
        );
        assert_eq!(nth_line(&actual, 4)?, "  iph            0");
        assert_eq!(
            nth_line(&actual, 5)?,
            "   i      r         vcoul        rho     ovrlp vcoul  ovrlp vtot  ovrlp rho"
        );
        assert_eq!(
            nth_line(&actual, 6)?,
            "    1  1.5073E-04 -7.6250E-01  1.1937E-03 -1.2200E+00 -4.4700E-01  2.7852E-03"
        );
        assert_eq!(
            nth_line(&actual, 254)?,
            "  249  3.6598E+01 -3.8625E+00  2.9722E-01 -6.1800E+00  2.9700E-01  6.9352E-01"
        );
        Ok(())
    }

    #[test]
    fn writes_all_potential_dat_outputs() -> Result<()> {
        let state = sample_wpot_state();
        let outputs = potential_dat_outputs(PotentialDatSetInput {
            highest_potential_index: 1,
            muffin_tin_indices: &[12, 13],
            norman_indices: &[40, 42],
            titles: &state.titles,
            electron_density: state.electron_density.view(),
            free_density: state.free_density.view(),
            overlapped_coulomb: state.overlapped_coulomb.view(),
            free_coulomb: state.free_coulomb.view(),
            total_potential: state.total_potential.view(),
        })?;

        assert_eq!(potential_dat_filename(1), "pot01.dat");
        let pot01 = outputs
            .get("pot01.dat")
            .ok_or_else(|| parse_error("pot01.dat"))?;
        assert_eq!(
            nth_line(pot01, 3)?,
            "    1  13  42  Unique potential, I_mt, I_norman.    Following data in atomic units."
        );
        assert_eq!(
            nth_line(pot01, 6)?,
            "    1  1.5073E-04 -1.5125E+00  2.1088E-02 -2.4200E+00 -8.9700E-01  1.2732E-02"
        );
        assert_eq!(
            nth_line(pot01, 254)?,
            "  249  3.6598E+01 -4.6125E+00  3.1712E-01 -7.3800E+00 -1.5300E-01  7.0346E-01"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_potential_dat_inputs() -> Result<()> {
        let state = sample_wpot_state();
        let too_small = Array2::zeros((250, 1));
        let err = match pot_dat_string(PotentialDatInput {
            potential_index: 0,
            muffin_tin_index: 12,
            norman_index: 40,
            titles: &state.titles,
            electron_density: too_small.view(),
            free_density: state.free_density.view(),
            overlapped_coulomb: state.overlapped_coulomb.view(),
            free_coulomb: state.free_coulomb.view(),
            total_potential: state.total_potential.view(),
        }) {
            Ok(_) => return Err(parse_error("short electron-density grid accepted")),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            IoError::PotentialOutputShape {
                field: "electron_density",
                ..
            }
        ));

        let mut invalid = state.free_density.clone();
        invalid[(0, 0)] = f64::NAN;
        let err = match pot_dat_string(PotentialDatInput {
            potential_index: 0,
            muffin_tin_index: 12,
            norman_index: 40,
            titles: &state.titles,
            electron_density: state.electron_density.view(),
            free_density: invalid.view(),
            overlapped_coulomb: state.overlapped_coulomb.view(),
            free_coulomb: state.free_coulomb.view(),
            total_potential: state.total_potential.view(),
        }) {
            Ok(_) => return Err(parse_error("non-finite density accepted")),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            IoError::InvalidPotentialOutput {
                field: "free_density",
                ..
            }
        ));
        Ok(())
    }

    struct WpotState {
        titles: Vec<String>,
        electron_density: Array2<f64>,
        free_density: Array2<f64>,
        overlapped_coulomb: Array2<f64>,
        free_coulomb: Array2<f64>,
        total_potential: Array2<f64>,
    }

    fn sample_wpot_state() -> WpotState {
        let rows = 251;
        let cols = 33;
        WpotState {
            titles: vec![
                "First title".to_string(),
                "Second title with trailing blanks   ".to_string(),
                "Third title 123".to_string(),
            ],
            electron_density: Array2::from_shape_fn((rows, cols), |(row, potential)| {
                0.035 * (row + 1) as f64 + 0.125 * potential as f64
            }),
            free_density: Array2::from_shape_fn((rows, cols), |(row, potential)| {
                0.015 * (row + 1) as f64 + 0.25 * potential as f64
            }),
            overlapped_coulomb: Array2::from_shape_fn((rows, cols), |(row, potential)| {
                -1.2 * (potential + 1) as f64 - 0.02 * (row + 1) as f64
            }),
            free_coulomb: Array2::from_shape_fn((rows, cols), |(row, potential)| {
                -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
            }),
            total_potential: Array2::from_shape_fn((rows, cols), |(row, potential)| {
                -0.45 * (potential + 1) as f64 + 0.003 * (row + 1) as f64
            }),
        }
    }

    fn nth_line(text: &str, index: usize) -> Result<&str> {
        text.lines()
            .nth(index)
            .ok_or_else(|| parse_error("potXX.dat"))
    }

    fn parse_error(path: &str) -> IoError {
        IoError::Parse {
            path: path.into(),
            line: 0,
            message: "expected generated potential output line".to_string(),
        }
    }
}
