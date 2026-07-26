//! OPCONS optical-constant helpers from FEFF.
//!
//! This module ports FEFF10's bundled `OPCONSAT/epsdb.f90` elemental
//! dielectric database and the numerical core of `OPCONSAT/addeps.f90`:
//! combine weighted epsilon tables on FEFF's legacy `AddEps` energy grid, then
//! compute the optical loss function used in `loss.dat`.

use std::sync::OnceLock;

use ndarray::Array1;
use thiserror::Error;

use crate::Real;

// This compact text asset is mechanically generated from the pinned FEFF10
// `src/OPCONSAT/epsdb.f90`; its header records the source SHA-256.
const EPSDB_DATA: &str = include_str!("opcons/epsdb.db");
const EPSDB_MAX_ATOMIC_NUMBER: usize = 99;

/// Input dielectric-function contribution for FEFF `AddEps`.
#[derive(Debug, Clone, PartialEq)]
pub struct EpsilonTable {
    /// Energy grid in eV. FEFF expects this grid to be strictly increasing.
    pub energy_ev: Array1<Real>,
    /// Real epsilon contribution excluding the FEFF background `+1`.
    pub epsilon1_minus_one: Array1<Real>,
    /// Imaginary epsilon contribution.
    pub epsilon2: Array1<Real>,
}

impl EpsilonTable {
    /// Number of epsilon samples in the table.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Combined dielectric function and FEFF loss function.
#[derive(Debug, Clone, PartialEq)]
pub struct CombinedEpsilon {
    /// FEFF `AddEps` output energy grid in eV.
    pub energy_ev: Array1<Real>,
    /// Combined real dielectric function, including FEFF's background `+1`.
    pub epsilon1: Array1<Real>,
    /// Combined imaginary dielectric function.
    pub epsilon2: Array1<Real>,
    /// FEFF optical loss function `eps2 / (eps1**2 + eps2**2)`.
    pub loss: Array1<Real>,
}

impl CombinedEpsilon {
    /// Number of combined samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Error returned by OPCONS optical-constant helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum OpconsError {
    /// At least one epsilon table is required.
    #[error("at least one epsilon table is required")]
    EmptyTables,
    /// The number of weights must match the number of tables.
    #[error("weight count mismatch: got {actual} but expected {expected}")]
    WeightCount { actual: usize, expected: usize },
    /// Each table must contain enough points for FEFF's order-1 interpolation.
    #[error("epsilon table {table} has {actual} points but needs at least {required}")]
    TooFewTablePoints {
        table: usize,
        actual: usize,
        required: usize,
    },
    /// A table column must have the same length as its energy grid.
    #[error("epsilon table {table} field {field} has {actual} values but expected {expected}")]
    Shape {
        table: usize,
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// An ndarray column was not available as a contiguous 1D slice.
    #[error("epsilon table {table} field {field} is not contiguous")]
    NonContiguous { table: usize, field: &'static str },
    /// Input weights must be finite.
    #[error("epsilon weight {index} must be finite, got {value}")]
    NonFiniteWeight { index: usize, value: Real },
    /// Input table values must be finite.
    #[error("epsilon table {table} field {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        table: usize,
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// FEFF interpolation assumes each source grid never decreases and ends
    /// with a nonzero-width final interval.
    #[error(
        "epsilon table {table} energy row {row} must not close a non-increasing interval, got {current} after {previous}"
    )]
    NonIncreasingEnergy {
        table: usize,
        row: usize,
        previous: Real,
        current: Real,
    },
    /// The computed loss value must be finite.
    #[error("combined loss row {row} must be finite, got {value}")]
    NonFiniteLoss { row: usize, value: Real },
    /// FEFF's bundled `epsdb` has no source table for this atomic number.
    #[error("FEFF OPCONS epsilon database has no entry for atomic number {atomic_number}")]
    DatabaseUnavailable { atomic_number: usize },
    /// The bundled FEFF `epsdb.f90` source did not satisfy its expected schema.
    #[error("bundled FEFF OPCONS epsilon database is malformed")]
    MalformedDatabase,
}

struct EpsilonSlices<'a> {
    energy_ev: &'a [Real],
    epsilon1_minus_one: &'a [Real],
    epsilon2: &'a [Real],
}

#[derive(Debug)]
struct DatabaseElement {
    table: EpsilonTable,
    sum_rule_error_percent: Real,
}

#[derive(Debug)]
struct EpsilonDatabase {
    elements: Vec<Option<DatabaseElement>>,
}

/// Load an elemental dielectric-function table from FEFF10's `epsdb`.
///
/// The bundled data is mechanically generated from the pinned
/// `feff10/src/OPCONSAT/epsdb.f90`. FEFF10 supplies entries for atomic numbers
/// 1 through 99; atomic number 100 is named by `getelement.f90` but
/// intentionally has no epsilon data.
pub fn epsilon_table_from_database(atomic_number: usize) -> Result<EpsilonTable, OpconsError> {
    epsilon_database()?
        .elements
        .get(atomic_number)
        .and_then(Option::as_ref)
        .map(|element| element.table.clone())
        .ok_or(OpconsError::DatabaseUnavailable { atomic_number })
}

/// Return FEFF10's relative sum-rule error annotation for an `epsdb` element.
pub fn epsilon_database_sum_rule_error_percent(atomic_number: usize) -> Result<Real, OpconsError> {
    epsilon_database()?
        .elements
        .get(atomic_number)
        .and_then(Option::as_ref)
        .map(|element| element.sum_rule_error_percent)
        .ok_or(OpconsError::DatabaseUnavailable { atomic_number })
}

/// Combine weighted epsilon tables and compute FEFF's optical loss function.
///
/// This follows `OPCONSAT/AddEps`: build FEFF's legacy sorted output grid,
/// interpolate every table to each output point using FEFF's order-1 `terp`
/// window semantics, sum the weighted epsilon contributions, add the real
/// background `+1`, and compute `eps2 / (eps1**2 + eps2**2)`.
///
/// FEFF's grid builder is not a strict mathematical set union: repeated source
/// grids can produce duplicate output rows. This function preserves that legacy
/// behavior because downstream FEFF-compatible `loss.dat` files include those
/// duplicate rows.
pub fn combine_epsilon_tables(
    tables: &[EpsilonTable],
    weights: &[Real],
) -> Result<CombinedEpsilon, OpconsError> {
    if tables.is_empty() {
        return Err(OpconsError::EmptyTables);
    }
    if weights.len() != tables.len() {
        return Err(OpconsError::WeightCount {
            actual: weights.len(),
            expected: tables.len(),
        });
    }
    for (index, &value) in weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(OpconsError::NonFiniteWeight { index, value });
        }
    }

    let slices = tables
        .iter()
        .enumerate()
        .map(|(table, data)| validate_table(table, data))
        .collect::<Result<Vec<_>, _>>()?;

    let energy = feff_add_eps_energy_grid(&slices);

    let mut epsilon1 = Vec::with_capacity(energy.len());
    let mut epsilon2 = Vec::with_capacity(energy.len());
    let mut loss = Vec::with_capacity(energy.len());
    let mut cursors = vec![0_usize; slices.len()];

    for &energy_point in &energy {
        let mut epsilon1_minus_one = 0.0;
        let mut epsilon2_value = 0.0;
        for ((table, &weight), cursor) in slices.iter().zip(weights.iter()).zip(cursors.iter_mut())
        {
            epsilon1_minus_one += weight
                * interpolate_order1_cached(
                    table.energy_ev,
                    table.epsilon1_minus_one,
                    energy_point,
                    cursor,
                );
            epsilon2_value += weight
                * interpolate_order1_cached(table.energy_ev, table.epsilon2, energy_point, cursor);
        }
        let epsilon1_value = epsilon1_minus_one + 1.0;
        let denominator = epsilon1_value.mul_add(epsilon1_value, epsilon2_value * epsilon2_value);
        let loss_value = epsilon2_value / denominator;
        if !loss_value.is_finite() {
            return Err(OpconsError::NonFiniteLoss {
                row: loss.len(),
                value: loss_value,
            });
        }
        epsilon1.push(epsilon1_value);
        epsilon2.push(epsilon2_value);
        loss.push(loss_value);
    }

    Ok(CombinedEpsilon {
        energy_ev: Array1::from_vec(energy),
        epsilon1: Array1::from_vec(epsilon1),
        epsilon2: Array1::from_vec(epsilon2),
        loss: Array1::from_vec(loss),
    })
}

fn feff_add_eps_energy_grid(slices: &[EpsilonSlices<'_>]) -> Vec<Real> {
    let point_capacity = slices.iter().map(|table| table.energy_ev.len()).sum();
    let mut sorted_energy = slices
        .iter()
        .flat_map(|table| table.energy_ev.iter().copied())
        .collect::<Vec<_>>();
    sorted_energy.sort_by(|left, right| left.total_cmp(right));
    if !sorted_energy
        .windows(2)
        .any(|window| window[0] == window[1])
    {
        return sorted_energy;
    }

    let mut energy = vec![0.0; point_capacity];
    let mut total = 0_usize;

    for table in slices {
        for &point in table.energy_ev {
            insert_feff_add_eps_energy_point(&mut energy, &mut total, point);
        }
    }

    energy.truncate(total);
    energy
}

fn insert_feff_add_eps_energy_point(energy: &mut [Real], total: &mut usize, point: Real) {
    *total += 1;
    let mut new_index = *total;

    if *total == 1 {
        energy[0] = point;
    }

    let loop_total = *total;
    let mut inserted = false;
    for sort_index in (1..loop_total).rev() {
        let current = energy[sort_index - 1];
        if point < current {
            new_index = sort_index;
        } else if point == current {
            *total -= 1;
        } else {
            assign_feff_add_eps_energy_point(energy, *total, new_index, point);
            inserted = true;
            break;
        }
    }

    if !inserted && *total == loop_total {
        energy.copy_within(0..(loop_total - 1), 1);
        energy[0] = point;
    }
}

fn assign_feff_add_eps_energy_point(
    energy: &mut [Real],
    total: usize,
    new_index: usize,
    point: Real,
) {
    if new_index == 0 || new_index > energy.len() {
        return;
    }

    if new_index != total && new_index < total {
        energy.copy_within((new_index - 1)..(total - 1), new_index);
    }

    energy[new_index - 1] = point;
}

fn epsilon_database() -> Result<&'static EpsilonDatabase, OpconsError> {
    static DATABASE: OnceLock<Result<EpsilonDatabase, OpconsError>> = OnceLock::new();
    match DATABASE.get_or_init(parse_epsilon_database) {
        Ok(database) => Ok(database),
        Err(error) => Err(*error),
    }
}

fn parse_epsilon_database() -> Result<EpsilonDatabase, OpconsError> {
    let mut rows_by_atomic_number = (0..=EPSDB_MAX_ATOMIC_NUMBER)
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut sum_rule_errors: Vec<Option<Real>> = vec![None; EPSDB_MAX_ATOMIC_NUMBER + 1];
    let mut current_atomic_number = None;

    for line in EPSDB_DATA.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(metadata) = line.strip_prefix('@') {
            let mut fields = metadata.split_whitespace();
            let atomic_number = fields
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or(OpconsError::MalformedDatabase)?;
            let value = fields
                .next()
                .and_then(parse_fortran_default_real)
                .ok_or(OpconsError::MalformedDatabase)?;
            if fields.next().is_some() || atomic_number == 0 {
                return Err(OpconsError::MalformedDatabase);
            }
            let error = sum_rule_errors
                .get_mut(atomic_number)
                .ok_or(OpconsError::MalformedDatabase)?;
            if error.is_some() {
                return Err(OpconsError::MalformedDatabase);
            }
            *error = Some(value);
            current_atomic_number = Some(atomic_number);
            continue;
        }

        let atomic_number = current_atomic_number.ok_or(OpconsError::MalformedDatabase)?;
        let mut fields = line.split_whitespace().map(parse_fortran_default_real);
        let energy = fields
            .next()
            .flatten()
            .ok_or(OpconsError::MalformedDatabase)?;
        let epsilon1_minus_one = fields
            .next()
            .flatten()
            .ok_or(OpconsError::MalformedDatabase)?;
        let epsilon2 = fields
            .next()
            .flatten()
            .ok_or(OpconsError::MalformedDatabase)?;
        if fields.next().is_some() {
            return Err(OpconsError::MalformedDatabase);
        }
        rows_by_atomic_number[atomic_number].push((energy, epsilon1_minus_one, epsilon2));
    }

    let mut elements = (0..=EPSDB_MAX_ATOMIC_NUMBER)
        .map(|_| None)
        .collect::<Vec<_>>();
    for atomic_number in 1..=EPSDB_MAX_ATOMIC_NUMBER {
        let rows = &rows_by_atomic_number[atomic_number];
        if rows.len() < 2 {
            return Err(OpconsError::MalformedDatabase);
        }

        let mut energy_ev = Vec::with_capacity(rows.len());
        let mut epsilon1_minus_one = Vec::with_capacity(rows.len());
        let mut epsilon2 = Vec::with_capacity(rows.len());
        for &(energy, epsilon1, epsilon2_value) in rows {
            energy_ev.push(energy);
            epsilon1_minus_one.push(epsilon1);
            epsilon2.push(epsilon2_value);
        }
        let table = EpsilonTable {
            energy_ev: Array1::from_vec(energy_ev),
            epsilon1_minus_one: Array1::from_vec(epsilon1_minus_one),
            epsilon2: Array1::from_vec(epsilon2),
        };
        validate_table(atomic_number, &table)?;
        let sum_rule_error_percent =
            sum_rule_errors[atomic_number].ok_or(OpconsError::MalformedDatabase)?;
        elements[atomic_number] = Some(DatabaseElement {
            table,
            sum_rule_error_percent,
        });
    }

    Ok(EpsilonDatabase { elements })
}

fn parse_fortran_default_real(token: &str) -> Option<Real> {
    let value = if token.contains(['D', 'd']) {
        token.replace(['D', 'd'], "E").parse::<f32>().ok()?
    } else {
        token.parse::<f32>().ok()?
    };
    Some(Real::from(value))
}

fn validate_table(table: usize, data: &EpsilonTable) -> Result<EpsilonSlices<'_>, OpconsError> {
    let point_count = data.point_count();
    if point_count < 2 {
        return Err(OpconsError::TooFewTablePoints {
            table,
            actual: point_count,
            required: 2,
        });
    }
    ensure_len(
        table,
        "epsilon1_minus_one",
        data.epsilon1_minus_one.len(),
        point_count,
    )?;
    ensure_len(table, "epsilon2", data.epsilon2.len(), point_count)?;

    let energy_ev = contiguous(table, "energy_ev", &data.energy_ev)?;
    let epsilon1_minus_one = contiguous(table, "epsilon1_minus_one", &data.epsilon1_minus_one)?;
    let epsilon2 = contiguous(table, "epsilon2", &data.epsilon2)?;

    for row in 0..point_count {
        ensure_finite(table, "energy_ev", row, energy_ev[row])?;
        ensure_finite(table, "epsilon1_minus_one", row, epsilon1_minus_one[row])?;
        ensure_finite(table, "epsilon2", row, epsilon2[row])?;
        if row > 0 && energy_ev[row] < energy_ev[row - 1] {
            return Err(OpconsError::NonIncreasingEnergy {
                table,
                row,
                previous: energy_ev[row - 1],
                current: energy_ev[row],
            });
        }
    }
    let last = point_count - 1;
    if energy_ev[last] <= energy_ev[last - 1] {
        return Err(OpconsError::NonIncreasingEnergy {
            table,
            row: last,
            previous: energy_ev[last - 1],
            current: energy_ev[last],
        });
    }

    Ok(EpsilonSlices {
        energy_ev,
        epsilon1_minus_one,
        epsilon2,
    })
}

fn ensure_len(
    table: usize,
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), OpconsError> {
    if actual == expected {
        Ok(())
    } else {
        Err(OpconsError::Shape {
            table,
            field,
            actual,
            expected,
        })
    }
}

fn contiguous<'a>(
    table: usize,
    field: &'static str,
    values: &'a Array1<Real>,
) -> Result<&'a [Real], OpconsError> {
    values
        .as_slice()
        .ok_or(OpconsError::NonContiguous { table, field })
}

fn ensure_finite(
    table: usize,
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), OpconsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OpconsError::NonFiniteValue {
            table,
            field,
            row,
            value,
        })
    }
}

fn interpolate_order1_cached(xs: &[Real], ys: &[Real], x: Real, cursor: &mut usize) -> Real {
    while *cursor + 1 < xs.len() && xs[*cursor + 1] <= x {
        *cursor += 1;
    }

    let lower = (*cursor).min(xs.len() - 2);
    let upper = lower + 1;
    let denominator = xs[upper] - xs[lower];
    ys[lower] + (x - xs[lower]) * (ys[upper] - ys[lower]) / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation::terp;
    use std::error::Error;

    #[test]
    fn combines_weighted_epsilon_tables_like_feff_addeps() -> Result<(), OpconsError> {
        let first = EpsilonTable {
            energy_ev: Array1::from_vec(vec![0.0, 2.0, 4.0]),
            epsilon1_minus_one: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            epsilon2: Array1::from_vec(vec![0.5, 1.0, 1.5]),
        };
        let second = EpsilonTable {
            energy_ev: Array1::from_vec(vec![1.0, 3.0]),
            epsilon1_minus_one: Array1::from_vec(vec![10.0, 30.0]),
            epsilon2: Array1::from_vec(vec![2.0, 6.0]),
        };

        let combined = combine_epsilon_tables(&[first, second], &[1.0, 0.5])?;

        assert_eq!(
            combined.energy_ev.as_slice(),
            Some([0.0, 1.0, 2.0, 3.0, 4.0].as_slice())
        );
        assert_close(combined.epsilon1[0], 2.0);
        assert_close(combined.epsilon1[1], 7.5);
        assert_close(combined.epsilon1[2], 13.0);
        assert_close(combined.epsilon1[3], 18.5);
        assert_close(combined.epsilon1[4], 24.0);
        assert_close(combined.epsilon2[0], 0.5);
        assert_close(combined.epsilon2[1], 1.75);
        assert_close(combined.epsilon2[2], 3.0);
        assert_close(combined.epsilon2[3], 4.25);
        assert_close(combined.epsilon2[4], 5.5);
        assert_close(combined.loss[0], 0.5 / (2.0_f64.powi(2) + 0.5_f64.powi(2)));
        assert_close(combined.loss[2], 3.0 / (13.0_f64.powi(2) + 3.0_f64.powi(2)));
        assert_close(combined.loss[4], 5.5 / (24.0_f64.powi(2) + 5.5_f64.powi(2)));
        Ok(())
    }

    #[test]
    fn bundled_epsdb_matches_feff10_source_rows() -> Result<(), OpconsError> {
        let hydrogen = epsilon_table_from_database(1)?;
        let copper = epsilon_table_from_database(29)?;
        let einsteinium = epsilon_table_from_database(99)?;

        assert_eq!(hydrogen.point_count(), 150);
        assert_eq!(copper.point_count(), 182);
        assert_eq!(einsteinium.point_count(), 314);
        assert_eq!(
            epsilon_database()?
                .elements
                .iter()
                .filter_map(Option::as_ref)
                .map(|element| element.table.point_count())
                .sum::<usize>(),
            23_266
        );

        // The DATA literals in epsdb.f90 have default REAL kind and are then
        // assigned into REAL(8), so FEFF first rounds every token through f32.
        assert_eq!(hydrogen.energy_ev[0], Real::from(0.250_658_000_0e-2_f32));
        assert_eq!(
            hydrogen.epsilon1_minus_one[0],
            Real::from(-0.972_640_000_0e2_f32)
        );
        assert_eq!(hydrogen.epsilon2[0], Real::from(0.196_079_000_0e3_f32));
        assert_eq!(copper.energy_ev[0], Real::from(0.250_658_000_0e-2_f32));
        assert_eq!(
            copper.epsilon1_minus_one[0],
            Real::from(-0.711_152_000_0e2_f32)
        );
        assert_eq!(copper.epsilon2[0], Real::from(0.435_246_000_0e3_f32));
        assert_eq!(copper.energy_ev[181], Real::from(0.100_000_000_0e6_f32));
        assert_eq!(
            copper.epsilon1_minus_one[181],
            Real::from(-0.384_787_000_0e-5_f32)
        );
        assert_eq!(copper.epsilon2[181], Real::from(0.499_733_000_0e-8_f32));
        assert_eq!(
            einsteinium.energy_ev[313],
            Real::from(0.100_000_000_0e6_f32)
        );
        assert_eq!(
            epsilon_database_sum_rule_error_percent(29)?,
            Real::from(4.382_76_f32)
        );
        assert!(matches!(
            epsilon_table_from_database(100),
            Err(OpconsError::DatabaseUnavailable { atomic_number: 100 })
        ));
        Ok(())
    }

    #[test]
    fn repeated_component_grids_preserve_feff_add_eps_duplicate_rows() -> Result<(), OpconsError> {
        let table = EpsilonTable {
            energy_ev: Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            epsilon1_minus_one: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            epsilon2: Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]),
        };

        let combined = combine_epsilon_tables(&[table.clone(), table], &[0.25, 0.75])?;

        assert_eq!(
            combined.energy_ev.as_slice(),
            Some([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0].as_slice())
        );
        assert_eq!(
            combined.epsilon1.as_slice(),
            Some([2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0].as_slice())
        );
        assert_eq!(
            combined.epsilon2.as_slice(),
            Some([0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.5].as_slice())
        );
        assert_close(combined.loss[1], 1.0 / (3.0_f64.powi(2) + 1.0_f64.powi(2)));
        assert_close(combined.loss[2], combined.loss[1]);
        Ok(())
    }

    #[test]
    fn accepts_interior_duplicate_source_energy_like_feff_opcons() -> Result<(), OpconsError> {
        let table = EpsilonTable {
            energy_ev: Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]),
            epsilon1_minus_one: Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]),
            epsilon2: Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5]),
        };

        let combined = combine_epsilon_tables(&[table], &[1.0])?;

        assert_eq!(
            combined.energy_ev.as_slice(),
            Some([0.0, 1.0, 2.0].as_slice())
        );
        assert_close(combined.epsilon1[1], 3.0);
        assert_close(combined.epsilon2[1], 1.0);
        Ok(())
    }

    #[test]
    fn cached_order1_interpolation_matches_feff_terp_window() -> Result<(), Box<dyn Error>> {
        let table = EpsilonTable {
            energy_ev: Array1::from_vec(vec![0.0, 2.0, 4.0, 8.0]),
            epsilon1_minus_one: Array1::from_vec(vec![1.0, 3.0, 2.0, 6.0]),
            epsilon2: Array1::from_vec(vec![0.5, 1.0, 1.5, 2.5]),
        };
        let query = [-1.0, 0.0, 1.0, 2.0, 3.5, 4.0, 6.0, 8.0, 10.0];
        let validated = validate_table(0, &table)?;
        let mut epsilon1_cursor = 0;
        let mut epsilon2_cursor = 0;

        for energy in query {
            let actual_epsilon1 = interpolate_order1_cached(
                validated.energy_ev,
                validated.epsilon1_minus_one,
                energy,
                &mut epsilon1_cursor,
            );
            let expected_epsilon1 =
                terp(validated.energy_ev, validated.epsilon1_minus_one, 1, energy)?.value;
            assert_close(actual_epsilon1, expected_epsilon1);

            let actual_epsilon2 = interpolate_order1_cached(
                validated.energy_ev,
                validated.epsilon2,
                energy,
                &mut epsilon2_cursor,
            );
            let expected_epsilon2 = terp(validated.energy_ev, validated.epsilon2, 1, energy)?.value;
            assert_close(actual_epsilon2, expected_epsilon2);
        }

        Ok(())
    }

    #[test]
    fn rejects_invalid_epsilon_inputs() {
        let table = EpsilonTable {
            energy_ev: Array1::from_vec(vec![0.0, 0.0]),
            epsilon1_minus_one: Array1::from_vec(vec![1.0, 2.0]),
            epsilon2: Array1::from_vec(vec![0.0, 1.0]),
        };
        assert!(matches!(
            combine_epsilon_tables(&[table], &[1.0]),
            Err(OpconsError::NonIncreasingEnergy {
                table: 0,
                row: 1,
                ..
            })
        ));

        let table = EpsilonTable {
            energy_ev: Array1::from_vec(vec![0.0, 1.0]),
            epsilon1_minus_one: Array1::from_vec(vec![1.0]),
            epsilon2: Array1::from_vec(vec![0.0, 1.0]),
        };
        assert!(matches!(
            combine_epsilon_tables(&[table], &[1.0]),
            Err(OpconsError::Shape {
                table: 0,
                field: "epsilon1_minus_one",
                actual: 1,
                expected: 2
            })
        ));

        assert!(matches!(
            combine_epsilon_tables(&[], &[]),
            Err(OpconsError::EmptyTables)
        ));
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 2.0e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}
