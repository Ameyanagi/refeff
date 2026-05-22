//! FULLSPECTRUM energy-grid helpers.

use ndarray::{Array1, ArrayView1};

use crate::Real;
use crate::elam::{
    ELAM_EDGE_ATOMIC_NUMBER_MAX, ElamError, elam_component_edge_energies_hartree,
    elam_edge_energy_hartree,
};

use super::constants::{
    FEFF_FULLSPECTRUM_DEFAULT_EDGE_HIGH_PADDING_EV, FEFF_FULLSPECTRUM_DEFAULT_EDGE_LOW_PADDING_EV,
    FEFF_FULLSPECTRUM_DEFAULT_EDGE_MIN_EV, FEFF_FULLSPECTRUM_DEFAULT_EDGE_STEP_EV,
    FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY, FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY, FEFF_HARTREE_EV,
};
use super::types::*;
use super::validation::{
    validate_finite, validate_finite_value, validate_positive, validate_transform_len,
};

/// Port of the active branch in `FULLSPECTRUM/egrid_lin.f90`.
///
/// The historical logarithmic branch is skipped by a `goto` in FEFF10. The live
/// path clamps non-positive lower bounds to `0.01 eV` in Hartree units and then
/// advances by a fixed linear step from `min_energy` to `max_energy`.
pub fn full_spectrum_linear_energy_grid(
    input: FullSpectrumLinearGridInput,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_transform_len("linear_grid", input.point_count)?;
    validate_finite("min_energy", input.min_energy)?;
    validate_finite("max_energy", input.max_energy)?;

    let min_energy = if input.min_energy <= 0.0 {
        FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY
    } else {
        input.min_energy
    };
    if input.max_energy <= min_energy {
        return Err(FullSpectrumError::InvalidEnergyRange {
            name: "linear_grid",
            min: min_energy,
            max: input.max_energy,
        });
    }

    let step = (input.max_energy - min_energy) / (input.point_count - 1) as Real;
    let mut grid = Vec::with_capacity(input.point_count);
    grid.push(min_energy);
    for row in 1..input.point_count {
        grid.push(grid[row - 1] + step);
    }
    Ok(Array1::from_vec(grid))
}

/// Build the FEFF Elam edge-energy list consumed by `FULLSPECTRUM/egrid.f90`.
///
/// FEFF's grid code calls `preved`/`nexted` against every component atomic
/// number each time it crosses an edge. This adapter materializes the same
/// positive table entries once, sorted and de-duplicated, so the Rust grid
/// routine can use an `ndarray` view without re-scanning the Elam table.
pub fn full_spectrum_elam_edge_energies(
    atomic_numbers: &[i32],
) -> Result<Array1<Real>, FullSpectrumError> {
    let mut energies = Vec::new();
    for (component, &atomic_number) in atomic_numbers.iter().enumerate() {
        let component_edges = elam_component_edge_energies_hartree(atomic_number)
            .map_err(|source| FullSpectrumError::ElamEdgeTable { component, source })?;
        energies.extend(component_edges.into_iter().map(|edge| edge.energy_hartree));
    }

    energies.sort_by(|left, right| left.total_cmp(right));
    energies.dedup_by(|left, right| *left == *right);
    Ok(Array1::from_vec(energies))
}

/// Port of `FULLSPECTRUM/rdop.f90`: infer default energy-grid bounds.
///
/// FEFF first considers only edges that requested fine structure. If none of
/// those produce a valid edge energy, it repeats the search over all selected
/// edges. The chosen window starts 50 eV before the lowest edge, clamped to
/// 0.1 eV, and ends 1000 eV above the highest edge. The point count matches
/// FEFF's integer truncation of a nominal 0.5 eV spacing.
pub fn full_spectrum_default_energy_grid(
    edges: &[FullSpectrumDefaultGridEdge],
) -> Result<FullSpectrumDefaultEnergyGrid, FullSpectrumError> {
    if edges.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "rdop_edge_grid",
        });
    }

    if let Some(grid) = default_energy_grid_from_edges(edges, true)? {
        return Ok(FullSpectrumDefaultEnergyGrid {
            used_fine_structure_edges: true,
            ..grid
        });
    }
    if let Some(grid) = default_energy_grid_from_edges(edges, false)? {
        return Ok(grid);
    }

    Err(FullSpectrumError::EmptyTable {
        name: "rdop_edge_grid",
    })
}

/// Port of `FULLSPECTRUM/egrid.f90`: edge-aware full-spectrum energy grid.
///
/// FEFF builds a grid that is regular in `k = sqrt(2 * (omega - edge))`
/// relative to the previous absorption edge, then restarts the k grid whenever
/// the next edge is crossed. This helper accepts already-tabulated edge
/// energies directly; [`full_spectrum_elam_edge_energies`] provides the FEFF
/// Elam-table adapter for component atomic numbers.
pub fn full_spectrum_edge_energy_grid(
    input: FullSpectrumEdgeGridInput<'_>,
) -> Result<FullSpectrumEdgeGrid, FullSpectrumError> {
    validate_transform_len("edge_grid", input.max_points)?;
    validate_finite("min_energy", input.min_energy)?;
    validate_finite("max_energy", input.max_energy)?;
    validate_positive("wave_number_step", input.wave_number_step)?;
    for (row, value) in input.edge_energies.iter().copied().enumerate() {
        validate_finite_value("edge_energies", row, value)?;
    }

    let min_energy = input.min_energy.max(FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY);
    if input.max_energy <= min_energy {
        return Err(FullSpectrumError::InvalidEnergyRange {
            name: "edge_grid",
            min: min_energy,
            max: input.max_energy,
        });
    }

    let mut energy = Vec::with_capacity(input.max_points);
    energy.push(min_energy);
    let mut next_edge = next_edge_above(input.edge_energies, min_energy);
    let mut current_edge = previous_edge_below(input.edge_energies, min_energy);
    let mut current_k = (2.0 * (min_energy - current_edge)).sqrt();
    let mut next_k = (2.0 * (next_edge - current_edge)).sqrt();

    for _ in 1..input.max_points {
        current_k += input.wave_number_step;
        let mut value = if current_k >= next_k {
            current_k = 0.0;
            current_edge = next_edge;
            next_edge = next_edge_above(input.edge_energies, current_edge);
            next_k = (2.0 * (next_edge - current_edge)).sqrt();
            current_edge
        } else {
            current_edge + current_k.powi(2) / 2.0
        };

        if value > input.max_energy {
            value = input.max_energy;
            energy.push(value);
            return Ok(FullSpectrumEdgeGrid {
                energy: Array1::from_vec(energy),
                clipped: false,
            });
        }
        energy.push(value);
    }

    Ok(FullSpectrumEdgeGrid {
        energy: Array1::from_vec(energy),
        clipped: true,
    })
}

fn default_energy_grid_from_edges(
    edges: &[FullSpectrumDefaultGridEdge],
    fine_structure_only: bool,
) -> Result<Option<FullSpectrumDefaultEnergyGrid>, FullSpectrumError> {
    let mut low = Real::INFINITY;
    let mut high: Real = 0.0;
    for (row, edge) in edges.iter().enumerate() {
        if fine_structure_only && !edge.fine_structure {
            continue;
        }
        if edge.atomic_number > ELAM_EDGE_ATOMIC_NUMBER_MAX {
            return Err(FullSpectrumError::ElamEdgeTable {
                component: row,
                source: ElamError::AtomicNumberOutOfRange {
                    z: edge.atomic_number,
                    max: ELAM_EDGE_ATOMIC_NUMBER_MAX,
                },
            });
        }
        let energy = elam_edge_energy_hartree(edge.atomic_number, edge.hole_index)
            .map_err(|source| FullSpectrumError::ElamEdgeTable {
                component: row,
                source,
            })?
            .ok_or(FullSpectrumError::MissingElamEdge {
                atomic_number: edge.atomic_number,
                hole_index: edge.hole_index,
            })?;
        let rounded_energy = Real::from(energy as f32);
        low = low.min(rounded_energy);
        high = high.max(rounded_energy);
    }

    if low <= high {
        let min_energy = (low - FEFF_FULLSPECTRUM_DEFAULT_EDGE_LOW_PADDING_EV / FEFF_HARTREE_EV)
            .max(FEFF_FULLSPECTRUM_DEFAULT_EDGE_MIN_EV / FEFF_HARTREE_EV);
        let max_energy = high + FEFF_FULLSPECTRUM_DEFAULT_EDGE_HIGH_PADDING_EV / FEFF_HARTREE_EV;
        let point_count = ((max_energy - min_energy) * FEFF_HARTREE_EV
            / FEFF_FULLSPECTRUM_DEFAULT_EDGE_STEP_EV) as usize;
        Ok(Some(FullSpectrumDefaultEnergyGrid {
            min_energy,
            max_energy,
            point_count,
            used_fine_structure_edges: false,
        }))
    } else {
        Ok(None)
    }
}

pub(super) fn previous_edge_below(edge_energies: ArrayView1<'_, Real>, current: Real) -> Real {
    edge_energies
        .iter()
        .copied()
        .map(|edge| edge.max(0.0))
        .filter(|&edge| edge < current)
        .fold(0.0, Real::max)
}

pub(super) fn next_edge_above(edge_energies: ArrayView1<'_, Real>, current: Real) -> Real {
    edge_energies
        .iter()
        .copied()
        .map(|edge| edge.max(0.0))
        .filter(|&edge| edge > current)
        .fold(1.0e8, Real::min)
}
