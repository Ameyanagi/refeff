//! FEFF FULLSPECTRUM numerical helpers.
//!
//! This module covers small kernels from `FULLSPECTRUM/` that can be tested
//! independently of the full driver. Larger spectrum assembly remains in the
//! module runner layer until the surrounding FEFF state is ported.

use ndarray::{Array1, ArrayView1};

use crate::elam::{
    ELAM_EDGE_ATOMIC_NUMBER_MAX, ElamError, elam_component_edge_energies_hartree,
    elam_edge_energy_hartree,
};
use crate::interpolation::{LintCache, lint_with_cache};
use crate::{Complex, Real};

/// FEFF Hartree energy in eV, matching `COMMON/m_constants.f90`.
pub const FEFF_HARTREE_EV: Real = 27.211_396;
/// FEFF Bohr radius in Angstrom, matching `COMMON/m_constants.f90`.
pub const FEFF_BOHR_ANGSTROM: Real = 0.529_177_249;
/// Inverse fine-structure constant used by FEFF optical sum rules.
pub const FEFF_ALPHA_INV: Real = 137.035_989_56;
/// Reduced Planck constant in eV seconds used by `FULLSPECTRUM/drdtrm.f90`.
pub const FEFF_HBAR_EV_SECONDS: Real = 6.58E-16;
/// FEFF lower energy floor for `FULLSPECTRUM/egrid_lin.f90`, in Hartree.
pub const FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY: Real = 0.01 / FEFF_HARTREE_EV;
/// FEFF lower energy floor for `FULLSPECTRUM/egrid.f90`, in Hartree.
pub const FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY: Real = 0.001 / FEFF_HARTREE_EV;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` default k-space step.
pub const FEFF_FULLSPECTRUM_XK_STEP: Real = 0.005;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` energy-grid capacity.
pub const FEFF_FULLSPECTRUM_GRID_CAPACITY: usize = 200_001;
/// FEFF `FULLSPECTRUM/gtedgs.f90` DOS-convolution edge threshold in Hartree.
pub const FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE: Real = 1.837_465_5;
/// Number of core-hole slots scanned by FEFF `FULLSPECTRUM/gtedgs.f90`.
pub const FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT: usize = 40;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` lower sum-rule grid bound.
pub const FEFF_FULLSPECTRUM_BACKGROUND_SUM_MIN: Real = 0.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` upper sum-rule grid bound.
pub const FEFF_FULLSPECTRUM_BACKGROUND_SUM_MAX: Real = 18_383.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` lower path-expansion transition k.
pub const FEFF_FULLSPECTRUM_FINE_STRUCTURE_LOW_K: Real = 3.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` upper FMS transition k.
pub const FEFF_FULLSPECTRUM_FINE_STRUCTURE_HIGH_K: Real = 4.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` entry transition size, in Hartree.
pub const FEFF_FULLSPECTRUM_EDGE_TRANSITION_SIZE: Real = 0.05;
/// FEFF `FULLSPECTRUM/addedg.f90` multiplier for the imaginary exit transition.
pub const FEFF_FULLSPECTRUM_IMAGINARY_EXIT_MULTIPLIER: usize = 10;
/// FEFF `FULLSPECTRUM/rdop.f90` lower padding before selected edges, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_LOW_PADDING_EV: Real = 50.0;
/// FEFF `FULLSPECTRUM/rdop.f90` upper padding after selected edges, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_HIGH_PADDING_EV: Real = 1000.0;
/// FEFF `FULLSPECTRUM/rdop.f90` lower default energy-grid floor, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_MIN_EV: Real = 0.1;
/// FEFF `FULLSPECTRUM/rdop.f90` default point-count spacing, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_STEP_EV: Real = 0.5;

const FEFF_FULLSPECTRUM_EDGE_LABELS: [&str; FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT] = [
    "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "P1", "P2", "P3", "P4", "P5", "P6", "P7",
    "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3",
];

mod types;

pub use types::*;

/// Port of `FULLSPECTRUM/qsum.f90`: compute the effective electron count.
///
/// FEFF applies a trapezoid integral to `omega * eps2`, then scales it by
/// `1 / (2*pi^2*numden)`. An active length of zero or one follows the Fortran
/// loop semantics and returns zero.
pub fn full_spectrum_effective_electron_count(
    input: FullSpectrumQSumInput<'_>,
) -> Result<Real, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    validate_active_len("epsilon2", input.active_len, input.epsilon2.len())?;
    validate_active_len("omega", input.active_len, input.omega.len())?;

    for row in 0..input.active_len {
        validate_finite_value("epsilon2", row, input.epsilon2[row])?;
        validate_finite_value("omega", row, input.omega[row])?;
        if row > 0 && input.omega[row] < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: input.omega[row],
            });
        }
    }

    let integral = (0..input.active_len.saturating_sub(1))
        .map(|row| {
            let left = input.omega[row] * input.epsilon2[row];
            let right = input.omega[row + 1] * input.epsilon2[row + 1];
            0.5 * (left + right) * (input.omega[row + 1] - input.omega[row])
        })
        .sum::<Real>();
    let result = integral / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value: result })
    }
}

/// Port of `FULLSPECTRUM/drdtrm.f90`: Drude free-electron epsilon term.
///
/// FEFF converts `tau` to a Hartree-scale width, uses `wp2 = 4*pi*numden`,
/// and writes both the eV-scaled width/plasma frequency and the complex
/// dielectric response on the active energy grid.
pub fn full_spectrum_drude_term(
    input: FullSpectrumDrudeInput<'_>,
) -> Result<FullSpectrumDrudeTerm, FullSpectrumError> {
    validate_positive("lifetime_seconds", input.lifetime_seconds)?;
    validate_positive("number_density", input.number_density)?;
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "drude_term" });
    }

    let gamma_hartree = FEFF_HBAR_EV_SECONDS / input.lifetime_seconds / FEFF_HARTREE_EV;
    let plasma_squared = 4.0 * std::f64::consts::PI * input.number_density;
    let mut epsilon = Vec::with_capacity(input.omega.len());

    for (row, omega) in input.omega.iter().copied().enumerate() {
        validate_finite_value("omega", row, omega)?;
        if omega <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value: omega,
            });
        }
        let denominator = omega.powi(2) + gamma_hartree.powi(2);
        let epsilon2 = plasma_squared * gamma_hartree / omega / denominator;
        let epsilon1 = -plasma_squared / denominator;
        let value = Complex::new(epsilon1, epsilon2);
        validate_finite_value("epsilon real", row, value.re)?;
        validate_finite_value("epsilon imaginary", row, value.im)?;
        epsilon.push(value);
    }

    Ok(FullSpectrumDrudeTerm {
        gamma_ev: gamma_hartree * FEFF_HARTREE_EV,
        plasma_frequency_ev: plasma_squared.sqrt() * FEFF_HARTREE_EV,
        omega: input.omega.to_owned(),
        epsilon: Array1::from_vec(epsilon),
    })
}

/// Port of `FULLSPECTRUM/rdval.f90`: valence `xmu.dat` contribution to eps2.
///
/// FEFF reads a valence `xmu.dat`, converts the normalized absorption to an
/// absolute cross section before this step, then linearly interpolates
/// `mu * 4*pi*alpha_inv*bohr^2*numden` onto the FULLSPECTRUM grid and divides
/// by the target photon energy.
pub fn full_spectrum_valence_epsilon2(
    input: FullSpectrumValenceInput<'_>,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    validate_matching_len(
        "source_absorption_angstrom2",
        input.source_absorption_angstrom2.len(),
        input.source_energy_ev.len(),
    )?;
    validate_transform_len("source_energy_ev", input.source_energy_ev.len())?;

    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("omega", row, value)?;
        if value <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value,
            });
        }
    }

    let mut source_energy = Vec::with_capacity(input.source_energy_ev.len());
    for (row, energy_ev) in input.source_energy_ev.iter().copied().enumerate() {
        validate_finite_value("source_energy_ev", row, energy_ev)?;
        if energy_ev <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "source_energy_ev",
                row,
                value: energy_ev,
            });
        }
        let energy_hartree = energy_ev / FEFF_HARTREE_EV;
        if row > 0 && energy_hartree <= source_energy[row - 1] {
            return Err(FullSpectrumError::NonIncreasingOmega {
                row,
                previous: source_energy[row - 1],
                current: energy_hartree,
            });
        }
        source_energy.push(energy_hartree);
    }

    let scale = 4.0
        * std::f64::consts::PI
        * FEFF_ALPHA_INV
        * FEFF_BOHR_ANGSTROM.powi(2)
        * input.number_density;
    let mut source_absorption = Vec::with_capacity(input.source_absorption_angstrom2.len());
    for (row, absorption) in input
        .source_absorption_angstrom2
        .iter()
        .copied()
        .enumerate()
    {
        validate_finite_value("source_absorption_angstrom2", row, absorption)?;
        source_absorption.push(absorption * scale);
    }

    let final_source_energy = source_energy[source_energy.len() - 1];
    let mut epsilon2 = vec![0.0; input.omega.len()];
    let mut cache = LintCache::new();
    for (row, omega) in input.omega.iter().copied().enumerate() {
        if omega < final_source_energy {
            let interpolated =
                lint_with_cache(&source_energy, &source_absorption, omega, &mut cache)
                    .map_err(|source| FullSpectrumError::Interpolation { source })?;
            let value = interpolated / omega;
            if !value.is_finite() {
                return Err(FullSpectrumError::NonFiniteResult { value });
            }
            epsilon2[row] = value;
        }
    }

    Ok(Array1::from_vec(epsilon2))
}

/// Port of `FULLSPECTRUM/rddens.f90`: estimate a species number density.
///
/// FEFF divides the total multiplicity of potentials with atomic number `iz`
/// by the sum of Norman-sphere volumes `xnatph * 4*pi*rnrm^3/3`.
pub fn full_spectrum_number_density(
    input: FullSpectrumNumberDensityInput<'_>,
) -> Result<Real, FullSpectrumError> {
    if input.target_atomic_number == 0 {
        return Err(FullSpectrumError::InvalidAtomicNumber {
            atomic_number: input.target_atomic_number,
        });
    }
    let potential_count = input.atomic_numbers.len();
    validate_matching_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    validate_matching_len("norman_radii", input.norman_radii.len(), potential_count)?;
    if potential_count == 0 {
        return Err(FullSpectrumError::EmptyTable {
            name: "number_density",
        });
    }

    let mut target_atoms = 0.0;
    let mut total_volume = 0.0;
    for row in 0..potential_count {
        let atomic_number = input.atomic_numbers[row];
        if atomic_number == 0 {
            return Err(FullSpectrumError::InvalidAtomicNumber { atomic_number });
        }

        let multiplicity = input.potential_multiplicities[row];
        validate_finite_value("potential_multiplicities", row, multiplicity)?;
        if multiplicity <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "potential_multiplicities",
                row,
                value: multiplicity,
            });
        }

        let radius = input.norman_radii[row];
        validate_finite_value("norman_radii", row, radius)?;
        if radius <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "norman_radii",
                row,
                value: radius,
            });
        }

        if atomic_number == input.target_atomic_number {
            target_atoms += multiplicity;
        }
        total_volume += multiplicity * radius.powi(3) * 4.0 * std::f64::consts::PI / 3.0;
    }

    let value = target_atoms / total_volume;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value })
    }
}

/// Port of `FULLSPECTRUM/gtedgs.f90`: select occupied component edges.
///
/// FEFF scans the 40 relativistic orbital slots from `getorb`, emits an edge
/// when the occupation is at least one, and sets the convolution flag when the
/// tabulated onset is at or below 50 eV (`1.8374655` Hartree). Fractionally
/// occupied states below one electron are skipped, matching FEFF's warning-only
/// branch.
pub fn full_spectrum_edges_from_occupations(
    input: FullSpectrumEdgeSelectionInput<'_>,
) -> Result<FullSpectrumEdgeSelection, FullSpectrumError> {
    validate_matching_len(
        "occupations",
        input.occupations.len(),
        FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT,
    )?;
    validate_matching_len(
        "edge_onsets_hartree",
        input.edge_onsets_hartree.len(),
        FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT,
    )?;

    let mut edges = Vec::new();
    for (row, label) in FEFF_FULLSPECTRUM_EDGE_LABELS.iter().copied().enumerate() {
        let occupation = input.occupations[row];
        validate_finite_value("occupations", row, occupation)?;
        if occupation < 0.0 {
            return Err(FullSpectrumError::NegativeValue {
                field: "occupations",
                row,
                value: occupation,
            });
        }

        let onset = input.edge_onsets_hartree[row];
        validate_finite_value("edge_onsets_hartree", row, onset)?;
        if occupation >= 1.0 {
            edges.push(FullSpectrumSelectedEdge {
                hole_index: row + 1,
                label,
                occupation,
                convolve: onset <= FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE,
            });
        }
    }

    Ok(FullSpectrumEdgeSelection { edges })
}

/// Port of `FULLSPECTRUM/sumrules.f90`: cumulative optical sum rules.
///
/// FEFF reads `opconsKK.dat`, recomputes the loss function from `epsilon - 1`,
/// converts energy and absorption units to atomic units, then writes a row after
/// every input point. This helper keeps the same integration rules, including
/// the right-endpoint rule for `mu` columns and trapezoids for energy-weighted
/// dielectric/loss columns.
pub fn full_spectrum_sum_rules(
    input: FullSpectrumSumRulesInput<'_>,
) -> Result<FullSpectrumSumRules, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    if input.energy_ev.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "sum_rules" });
    }
    validate_matching_len(
        "epsilon_minus_one",
        input.epsilon_minus_one.len(),
        input.energy_ev.len(),
    )?;
    validate_matching_len(
        "refractive_index_minus_one",
        input.refractive_index_minus_one.len(),
        input.energy_ev.len(),
    )?;
    validate_matching_len(
        "absorption_coefficient",
        input.absorption_coefficient.len(),
        input.energy_ev.len(),
    )?;

    let mut epsilon2_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut absorption_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut loss_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut absorption_refractive_sum = Vec::with_capacity(input.energy_ev.len());
    let mut refractive_index_sum_ratio = Vec::with_capacity(input.energy_ev.len());
    let mut log_loss_moment_ratio = Vec::with_capacity(input.energy_ev.len());

    let scale = 1.0 / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    let mut previous_energy_hartree = 0.0;
    let mut previous_epsilon2 = 0.0;
    let mut previous_loss = 0.0;
    let mut sum_epsilon2 = 0.0;
    let mut sum_absorption = 0.0;
    let mut sum_loss = 0.0;
    let mut sum_absorption_refractive = 0.0;
    let mut sum_refractive = 0.0;
    let mut sum_abs_refractive = 0.0;
    let mut sum_log_loss = 0.0;

    for row in 0..input.energy_ev.len() {
        let energy_ev = input.energy_ev[row];
        let epsilon = input.epsilon_minus_one[row];
        let refractive_index = input.refractive_index_minus_one[row];
        let absorption = input.absorption_coefficient[row];

        validate_finite_value("energy_ev", row, energy_ev)?;
        validate_finite_value("epsilon_minus_one real", row, epsilon.re)?;
        validate_finite_value("epsilon_minus_one imaginary", row, epsilon.im)?;
        validate_finite_value("refractive_index_minus_one real", row, refractive_index.re)?;
        validate_finite_value(
            "refractive_index_minus_one imaginary",
            row,
            refractive_index.im,
        )?;
        validate_finite_value("absorption_coefficient", row, absorption)?;
        if row > 0 && energy_ev < input.energy_ev[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.energy_ev[row - 1],
                current: energy_ev,
            });
        }

        let energy_hartree = energy_ev / FEFF_HARTREE_EV;
        let mu_atomic = absorption * FEFF_BOHR_ANGSTROM / 1000.0;
        let epsilon2 = epsilon.im;
        let refractive_real = refractive_index.re;
        let loss = epsilon2 / (epsilon2.powi(2) + (epsilon.re + 1.0).powi(2));
        let delta_energy = energy_hartree - previous_energy_hartree;

        sum_epsilon2 += 0.5
            * (energy_hartree * epsilon2 + previous_energy_hartree * previous_epsilon2)
            * delta_energy;
        sum_absorption += mu_atomic * delta_energy;
        sum_loss +=
            0.5 * (energy_hartree * loss + previous_energy_hartree * previous_loss) * delta_energy;
        sum_absorption_refractive += mu_atomic * refractive_real * delta_energy;
        sum_refractive += delta_energy * refractive_real;
        sum_abs_refractive += delta_energy * refractive_real.abs();
        if previous_energy_hartree > 0.0 {
            sum_log_loss += 0.5
                * (energy_hartree.ln() * energy_hartree * loss
                    + previous_energy_hartree.ln() * previous_energy_hartree * previous_loss)
                * delta_energy;
        } else {
            sum_log_loss += energy_hartree.ln() * energy_hartree * loss * energy_hartree;
        }

        push_sum_rule_value(
            &mut epsilon2_effective_electrons,
            "epsilon2_effective_electrons",
            row,
            scale * sum_epsilon2,
        )?;
        push_sum_rule_value(
            &mut absorption_effective_electrons,
            "absorption_effective_electrons",
            row,
            FEFF_ALPHA_INV * scale * sum_absorption,
        )?;
        push_sum_rule_value(
            &mut loss_effective_electrons,
            "loss_effective_electrons",
            row,
            scale * sum_loss,
        )?;
        push_sum_rule_value(
            &mut absorption_refractive_sum,
            "absorption_refractive_sum",
            row,
            FEFF_ALPHA_INV * scale * sum_absorption_refractive,
        )?;
        push_sum_rule_value(
            &mut refractive_index_sum_ratio,
            "refractive_index_sum_ratio",
            row,
            sum_refractive / sum_abs_refractive,
        )?;
        push_sum_rule_value(
            &mut log_loss_moment_ratio,
            "log_loss_moment_ratio",
            row,
            sum_log_loss / sum_loss,
        )?;

        previous_energy_hartree = energy_hartree;
        previous_epsilon2 = epsilon2;
        previous_loss = loss;
    }

    Ok(FullSpectrumSumRules {
        energy_ev: input.energy_ev.to_owned(),
        epsilon2_effective_electrons: Array1::from_vec(epsilon2_effective_electrons),
        absorption_effective_electrons: Array1::from_vec(absorption_effective_electrons),
        loss_effective_electrons: Array1::from_vec(loss_effective_electrons),
        absorption_refractive_sum: Array1::from_vec(absorption_refractive_sum),
        refractive_index_sum_ratio: Array1::from_vec(refractive_index_sum_ratio),
        log_loss_moment_ratio: Array1::from_vec(log_loss_moment_ratio),
    })
}

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

/// Port of `FULLSPECTRUM/rdbkg.f90`: assemble FPRIME background factors.
///
/// FEFF reads `fprime1/xmu.dat`, `fprime2/xmu.dat`, and following numbered
/// files, then processes them in reverse so lower-numbered files overwrite
/// overlapping higher-numbered intervals. This helper accepts those parsed
/// segments directly in that file-priority order and returns the complex
/// background scattering factor on the caller's Hartree grid.
pub fn full_spectrum_background_from_fprime(
    input: FullSpectrumBackgroundInput<'_>,
) -> Result<FullSpectrumBackground, FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "omega" });
    }
    if input.segments.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "background_segments",
        });
    }
    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("background omega", row, value)?;
        if row > 0 && value < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: value,
            });
        }
    }

    let mut prepared = input
        .segments
        .iter()
        .enumerate()
        .map(|(segment, source)| prepare_background_segment(segment, *source))
        .collect::<Result<Vec<_>, _>>()?;

    let mut f_prime = Array1::<Real>::zeros(input.omega.len());
    let mut f_double_prime = Array1::<Real>::zeros(input.omega.len());
    let sum_grid = full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
        point_count: FEFF_FULLSPECTRUM_GRID_CAPACITY,
        min_energy: FEFF_FULLSPECTRUM_BACKGROUND_SUM_MIN,
        max_energy: FEFF_FULLSPECTRUM_BACKGROUND_SUM_MAX,
    })?;
    let mut sum_f_double_prime = Array1::<Real>::zeros(sum_grid.len());

    let mut zero_energy_fprime = 0.0;
    let mut zero_energy = Real::INFINITY;
    for segment in prepared.iter_mut().rev() {
        if segment.low_energy < zero_energy {
            zero_energy = segment.low_energy;
            zero_energy_fprime = segment.f_prime[0];
        }
        interpolate_background_segment(
            segment,
            input.omega,
            zero_energy_fprime,
            &mut f_prime,
            &mut f_double_prime,
        )?;
        interpolate_background_sum_segment(segment, sum_grid.view(), &mut sum_f_double_prime)?;
    }

    for (row, value) in sum_f_double_prime.iter_mut().enumerate() {
        *value /= sum_grid[row].powi(2);
    }
    let effective_electron_count = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
        number_density: 1.0 / (4.0 * std::f64::consts::PI),
        epsilon2: sum_f_double_prime.view(),
        omega: sum_grid.view(),
        active_len: sum_grid.len(),
    })?;

    let scattering_factor = Array1::from_shape_fn(input.omega.len(), |row| {
        if input.omega[row] <= zero_energy {
            Complex::new(zero_energy_fprime, 0.0)
        } else {
            Complex::new(f_prime[row], f_double_prime[row])
        }
    });

    Ok(FullSpectrumBackground {
        scattering_factor,
        effective_electron_count,
        zero_energy_fprime,
    })
}

/// Port of `FULLSPECTRUM/rdst.f90`: combine FMS and path fine structure.
///
/// The four input segments correspond to FEFF's `fms_re`, `path_re`,
/// `fms_im`, and `path_im` `xmu.dat` files after parsing and unit conversion.
/// Values are interpolated onto `omega`; in the transition interval selected
/// from the FMS wave-number grid, FEFF mixes FMS and path values with
/// `sin(theta)^2`/`cos(theta)^2` weights.
pub fn full_spectrum_fine_structure_from_segments(
    input: FullSpectrumFineStructureInput<'_>,
) -> Result<FullSpectrumFineStructure, FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "omega" });
    }
    validate_positive("low_wave_number", input.low_wave_number)?;
    validate_positive("high_wave_number", input.high_wave_number)?;
    if input.high_wave_number <= input.low_wave_number {
        return Err(FullSpectrumError::InvalidEnergyRange {
            name: "fine_structure_wave_number",
            min: input.low_wave_number,
            max: input.high_wave_number,
        });
    }
    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("fine_structure omega", row, value)?;
        if row > 0 && value < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: value,
            });
        }
    }

    let mut real_fms = prepare_fine_structure_segment(0, input.real_fms)?;
    let mut real_path = prepare_fine_structure_segment(1, input.real_path)?;
    let mut imaginary_fms = prepare_fine_structure_segment(2, input.imaginary_fms)?;
    let mut imaginary_path = prepare_fine_structure_segment(3, input.imaginary_path)?;

    let real_transition = fine_structure_transition_interval(
        "real_fms",
        &real_fms,
        input.low_wave_number,
        input.high_wave_number,
    )?;
    let imaginary_transition = fine_structure_transition_interval(
        "imaginary_fms",
        &imaginary_fms,
        input.low_wave_number,
        input.high_wave_number,
    )?;

    let mut real_part = Array1::<Real>::zeros(input.omega.len());
    let mut imaginary_part = Array1::<Real>::zeros(input.omega.len());
    let mut real_background = Array1::<Real>::zeros(input.omega.len());
    let mut imaginary_background = Array1::<Real>::zeros(input.omega.len());

    interpolate_fine_structure_fms(
        &mut real_fms,
        input.omega,
        real_transition[1],
        FineStructureFmsBounds {
            include_low: true,
            include_high: true,
        },
        false,
        &mut real_part,
        &mut real_background,
    )?;
    interpolate_fine_structure_path(
        &mut real_path,
        input.omega,
        real_transition,
        false,
        &mut real_part,
        &mut real_background,
    )?;
    interpolate_fine_structure_fms(
        &mut imaginary_fms,
        input.omega,
        imaginary_transition[1],
        FineStructureFmsBounds {
            include_low: false,
            include_high: false,
        },
        true,
        &mut imaginary_part,
        &mut imaginary_background,
    )?;
    interpolate_fine_structure_path(
        &mut imaginary_path,
        input.omega,
        imaginary_transition,
        true,
        &mut imaginary_part,
        &mut imaginary_background,
    )?;

    let scattering_factor = Array1::from_shape_fn(input.omega.len(), |row| {
        Complex::new(real_part[row], imaginary_part[row])
    });
    let background = Array1::from_shape_fn(input.omega.len(), |row| {
        Complex::new(real_background[row], imaginary_background[row])
    });

    Ok(FullSpectrumFineStructure {
        scattering_factor,
        background,
        real_energy_interval: [real_fms.low_energy, real_path.high_energy],
        imaginary_energy_interval: [imaginary_fms.low_energy, imaginary_path.high_energy],
        real_transition_interval: real_transition,
        imaginary_transition_interval: imaginary_transition,
    })
}

/// Port of `FULLSPECTRUM/addedg.f90`: assemble one edge contribution.
///
/// FEFF first reads the smooth FPRIME background, then overlays FMS/path
/// fine-structure data with separate real and imaginary transition intervals.
/// Before mixing, FEFF conjugates all scattering factors to match its optical
/// sign convention. After the seams are blended, it subtracts `fp0` from the
/// full signal and atomic background so insulating spectra satisfy `f(0)=0`.
pub fn full_spectrum_assemble_edge(
    input: FullSpectrumEdgeAssemblyInput<'_>,
) -> Result<FullSpectrumEdgeAssembly, FullSpectrumError> {
    validate_edge_assembly_input(input)?;

    let mut background = input
        .background
        .scattering_factor
        .mapv(|value| value.conj());
    let mut scattering = input
        .fine_structure
        .scattering_factor
        .mapv(|value| value.conj());
    let fine_background = input.fine_structure.background.mapv(|value| value.conj());

    let real_low = input.fine_structure.real_energy_interval[0];
    let real_high = input.fine_structure.real_energy_interval[1];
    let imaginary_low = input.fine_structure.imaginary_energy_interval[0];
    let imaginary_high = input.fine_structure.imaginary_energy_interval[1];
    let real_low_row = last_omega_le(input.omega, real_low);
    let real_high_row = last_omega_le(input.omega, real_high);
    let imaginary_low_row = last_omega_le(input.omega, imaginary_low);
    let imaginary_high_row = last_omega_le(input.omega, imaginary_high);
    let overlap_points = edge_transition_overlap(
        input.omega,
        imaginary_low_row,
        imaginary_high_row,
        imaginary_low,
        input.transition_size,
    );
    let background_overlap_points = (overlap_points / 5).max(1);

    if input.omega[0] <= real_low
        && let Some(end) = real_low_row
    {
        for row in 0..=end {
            scattering[row] = Complex::new(background[row].re, scattering[row].im);
        }
    }
    if input.omega[0] <= imaginary_low
        && let Some(end) = imaginary_low_row
    {
        for row in 0..=end {
            scattering[row] = Complex::new(scattering[row].re, 0.0);
            background[row] = Complex::new(background[row].re, 0.0);
        }
    }

    if input.omega[0] <= real_low
        && let Some(start) = real_low_row
    {
        apply_real_background_entry(
            &mut background,
            &fine_background,
            start,
            background_overlap_points,
            overlap_points,
        );
        apply_real_scattering_entry(&mut scattering, &background, start, overlap_points);
    }
    if input.omega[0] <= imaginary_low
        && let Some(start) = imaginary_low_row
    {
        apply_imaginary_background_entry(
            &mut background,
            &fine_background,
            start,
            background_overlap_points,
            overlap_points,
        );
        apply_imaginary_scattering_entry(&mut scattering, &background, start, overlap_points);
    }

    if let Some(end) = real_high_row
        && end >= overlap_points
    {
        apply_real_exit(
            &mut scattering,
            &mut background,
            &fine_background,
            end - overlap_points,
            end,
            overlap_points,
        );
    }

    let imaginary_exit_start = match (imaginary_low_row, imaginary_high_row) {
        (Some(low), Some(high)) => {
            let wide_start = high.saturating_sub(
                overlap_points.saturating_mul(FEFF_FULLSPECTRUM_IMAGINARY_EXIT_MULTIPLIER),
            );
            Some((low + overlap_points).max(wide_start))
        }
        _ => None,
    };
    if let (Some(start), Some(end)) = (imaginary_exit_start, imaginary_high_row)
        && start < end
    {
        apply_imaginary_exit(
            &mut scattering,
            &mut background,
            &fine_background,
            start,
            end,
        );
    }

    if let (Some(start), Some(end)) = (real_low_row, real_high_row) {
        let middle_start = start + overlap_points - 1;
        let middle_end = end.saturating_sub(overlap_points).saturating_add(1);
        if middle_start <= middle_end {
            for row in middle_start..=middle_end.min(background.len() - 1) {
                background[row] = Complex::new(fine_background[row].re, background[row].im);
            }
        }
    }
    if let (Some(low), Some(exit_start)) = (imaginary_low_row, imaginary_exit_start) {
        let middle_start = low + overlap_points;
        if middle_start <= exit_start {
            for row in middle_start..=exit_start.min(background.len() - 1) {
                background[row] = Complex::new(background[row].re, fine_background[row].im);
            }
        }
    }

    let real_tail_start = real_high_row.unwrap_or(0);
    for row in real_tail_start..scattering.len() {
        scattering[row] = Complex::new(background[row].re, scattering[row].im);
    }
    let imaginary_tail_start = imaginary_high_row.unwrap_or(0);
    for row in imaginary_tail_start..scattering.len() {
        scattering[row] = Complex::new(scattering[row].re, background[row].im);
    }

    let shift = Complex::new(input.background.zero_energy_fprime, 0.0);
    for row in 0..scattering.len() {
        scattering[row] -= shift;
        background[row] -= shift;
    }

    Ok(FullSpectrumEdgeAssembly {
        scattering_factor: scattering,
        background,
        effective_electron_count: input.background.effective_electron_count,
        zero_energy_fprime: input.background.zero_energy_fprime,
        overlap_points,
    })
}

/// Convert assembled scattering factors to dielectric contributions.
///
/// This is the `fullspectrum.f90` step immediately after `addedg`: FEFF turns
/// `f` and `f0` into dielectric response with
/// `-4*pi*numden*f/omega**2`, while the original imaginary scattering factor
/// contributes to the `sigma` diagnostic column.
pub fn full_spectrum_scattering_to_dielectric(
    input: FullSpectrumScatteringDielectricInput<'_>,
) -> Result<FullSpectrumScatteringDielectric, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "scattering_to_dielectric",
        });
    }
    validate_matching_len(
        "scattering_factor",
        input.scattering_factor.len(),
        input.omega.len(),
    )?;
    validate_matching_len(
        "background_scattering_factor",
        input.background_scattering_factor.len(),
        input.omega.len(),
    )?;

    let mut epsilon_minus_one = Vec::with_capacity(input.omega.len());
    let mut background_epsilon_minus_one = Vec::with_capacity(input.omega.len());
    let mut sigma = Vec::with_capacity(input.omega.len());
    let density_scale = -4.0 * std::f64::consts::PI * input.number_density;
    let bohr_squared = FEFF_BOHR_ANGSTROM.powi(2);

    for row in 0..input.omega.len() {
        let omega = input.omega[row];
        validate_finite_value("omega", row, omega)?;
        if omega <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value: omega,
            });
        }

        let scattering = input.scattering_factor[row];
        validate_finite_value("scattering_factor real", row, scattering.re)?;
        validate_finite_value("scattering_factor imaginary", row, scattering.im)?;
        let background = input.background_scattering_factor[row];
        validate_finite_value("background_scattering_factor real", row, background.re)?;
        validate_finite_value("background_scattering_factor imaginary", row, background.im)?;

        let scale = density_scale / omega.powi(2);
        let epsilon = scattering * scale;
        let background_epsilon = background * scale;
        let sigma_value = -scattering.im / FEFF_ALPHA_INV / bohr_squared / omega;

        validate_finite_value("epsilon real", row, epsilon.re)?;
        validate_finite_value("epsilon imaginary", row, epsilon.im)?;
        validate_finite_value("background epsilon real", row, background_epsilon.re)?;
        validate_finite_value("background epsilon imaginary", row, background_epsilon.im)?;
        validate_finite_value("sigma", row, sigma_value)?;

        epsilon_minus_one.push(epsilon);
        background_epsilon_minus_one.push(background_epsilon);
        sigma.push(sigma_value);
    }

    Ok(FullSpectrumScatteringDielectric {
        omega: input.omega.to_owned(),
        epsilon_minus_one: Array1::from_vec(epsilon_minus_one),
        background_epsilon_minus_one: Array1::from_vec(background_epsilon_minus_one),
        sigma: Array1::from_vec(sigma),
    })
}

/// Port of `FULLSPECTRUM/opcons.f90`: derive optical constants from `eps - 1`.
///
/// FEFF keeps the dielectric response offset by one in this routine. The
/// returned refractive-index column is also offset by one, matching the
/// `opcons.dat`/`opconsKK.dat` text layout.
pub fn full_spectrum_optical_constants(
    input: FullSpectrumOpticalConstantsInput<'_>,
) -> Result<FullSpectrumOpticalConstants, FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "optical_constants",
        });
    }
    validate_matching_len(
        "epsilon_minus_one",
        input.epsilon_minus_one.len(),
        input.omega.len(),
    )?;

    let mut refractive_index_minus_one = Vec::with_capacity(input.omega.len());
    let mut absorption_coefficient = Vec::with_capacity(input.omega.len());
    let mut reflectivity = Vec::with_capacity(input.omega.len());
    let mut loss = Vec::with_capacity(input.omega.len());
    let one = Complex::new(1.0, 0.0);
    let alpha = 1.0 / FEFF_ALPHA_INV;

    for row in 0..input.omega.len() {
        let omega = input.omega[row];
        validate_finite_value("omega", row, omega)?;
        if omega <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value: omega,
            });
        }

        let epsilon_minus_one = input.epsilon_minus_one[row];
        validate_finite_value("epsilon real", row, epsilon_minus_one.re)?;
        validate_finite_value("epsilon imaginary", row, epsilon_minus_one.im)?;

        let dielectric = epsilon_minus_one + one;
        let refractive_index = dielectric.sqrt();
        let refractive_minus_one = refractive_index - one;
        let absorption = 2.0 * omega * alpha * refractive_index.im / FEFF_BOHR_ANGSTROM * 1000.0;
        let reflectance = ((refractive_index - one) / (refractive_index + one)).norm_sqr();
        let loss_value = -(one / dielectric).im;

        validate_finite_value("refractive index real", row, refractive_minus_one.re)?;
        validate_finite_value("refractive index imaginary", row, refractive_minus_one.im)?;
        validate_finite_value("absorption coefficient", row, absorption)?;
        validate_finite_value("reflectivity", row, reflectance)?;
        validate_finite_value("loss", row, loss_value)?;

        refractive_index_minus_one.push(refractive_minus_one);
        absorption_coefficient.push(absorption);
        reflectivity.push(reflectance);
        loss.push(loss_value);
    }

    Ok(FullSpectrumOpticalConstants {
        omega: input.omega.to_owned(),
        epsilon_minus_one: input.epsilon_minus_one.to_owned(),
        refractive_index_minus_one: Array1::from_vec(refractive_index_minus_one),
        absorption_coefficient: Array1::from_vec(absorption_coefficient),
        reflectivity: Array1::from_vec(reflectivity),
        loss: Array1::from_vec(loss),
    })
}

/// Port of `FULLSPECTRUM/kk.f90`: Kramers-Kronig transform of `eps2`.
///
/// FEFF evaluates the principal-value integral at interval midpoints, using an
/// analytic linear-`eps2` contribution near the singularity and trapezoid
/// contributions elsewhere, then linearly interpolates the midpoint result back
/// to the input grid. The first and last output values copy their neighbors,
/// matching the Fortran endpoint convention.
pub fn full_spectrum_kramers_kronig(
    input: FullSpectrumKramersKronigInput<'_>,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_transform_len("kramers_kronig", input.omega.len())?;
    validate_matching_len("epsilon2", input.epsilon2.len(), input.omega.len())?;
    validate_strictly_increasing_grid(input.omega)?;
    for (row, value) in input.epsilon2.iter().copied().enumerate() {
        validate_finite_value("epsilon2", row, value)?;
    }

    let midpoints = midpoint_grid(input.omega);
    let midpoint_values = midpoints
        .iter()
        .copied()
        .map(|midpoint| kk_midpoint_value(input.omega, input.epsilon2, midpoint))
        .collect::<Result<Vec<_>, _>>()?;
    interpolate_midpoints_to_grid(input.omega, &midpoints, &midpoint_values)
}

/// Port of `FULLSPECTRUM/hamaker.f90`: dielectric transform on the imaginary axis.
///
/// The FEFF routine integrates `omega' * eps2(omega') / (omega'^2 + omega^2)`
/// at interval midpoints, applies the `2/pi` factor, and interpolates back to
/// the input grid. The real part of the input epsilon is ignored by FEFF.
pub fn full_spectrum_hamaker_transform(
    input: FullSpectrumHamakerInput<'_>,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_transform_len("hamaker", input.omega.len())?;
    validate_matching_len("epsilon", input.epsilon.len(), input.omega.len())?;
    validate_strictly_increasing_grid(input.omega)?;
    for row in 0..input.epsilon.len() {
        validate_finite_value("epsilon real", row, input.epsilon[row].re)?;
        validate_finite_value("epsilon imaginary", row, input.epsilon[row].im)?;
    }

    let midpoints = midpoint_grid(input.omega);
    let midpoint_values = midpoints
        .iter()
        .copied()
        .map(|midpoint| hamaker_midpoint_value(input.omega, input.epsilon, midpoint))
        .collect::<Result<Vec<_>, _>>()?;
    interpolate_midpoints_to_grid(input.omega, &midpoints, &midpoint_values)
}

#[derive(Debug, Clone)]
struct PreparedBackgroundSegment {
    energy_hartree: Vec<Real>,
    f_prime: Vec<Real>,
    f_double_prime: Vec<Real>,
    low_energy: Real,
    high_energy: Real,
    output_cache: LintCache,
    sum_cache: LintCache,
}

fn prepare_background_segment(
    segment_index: usize,
    input: FullSpectrumBackgroundSegmentInput<'_>,
) -> Result<PreparedBackgroundSegment, FullSpectrumError> {
    let len = input.photon_energy_ev.len();
    if len < 2 {
        return Err(FullSpectrumError::SegmentTooShort {
            name: "background",
            segment: segment_index,
            len,
        });
    }
    validate_segment_len("f_prime", segment_index, input.f_prime.len(), len)?;
    validate_segment_len(
        "f_double_prime",
        segment_index,
        input.f_double_prime.len(),
        len,
    )?;

    let mut energy_ev = input.photon_energy_ev.to_vec();
    let mut f_prime = input.f_prime.to_vec();
    let mut f_double_prime = input.f_double_prime.to_vec();
    for row in 0..len {
        validate_finite_value("background photon_energy_ev", row, energy_ev[row])?;
        validate_finite_value("background f_prime", row, f_prime[row])?;
        validate_finite_value("background f_double_prime", row, f_double_prime[row])?;
        if row > 0 && energy_ev[row] <= energy_ev[row - 1] {
            return Err(FullSpectrumError::SegmentNonIncreasingEnergy {
                segment: segment_index,
                row,
                previous: energy_ev[row - 1],
                current: energy_ev[row],
            });
        }
    }

    if energy_ev[0] <= 0.0 {
        let shifted_energy = 0.01;
        let mut cache = LintCache::new();
        f_prime[0] = lint_with_cache(&energy_ev, &f_prime, shifted_energy, &mut cache)
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
        cache.reset();
        f_double_prime[0] =
            lint_with_cache(&energy_ev, &f_double_prime, shifted_energy, &mut cache)
                .map_err(|source| FullSpectrumError::Interpolation { source })?;
        energy_ev[0] = shifted_energy;
        if energy_ev[1] <= energy_ev[0] {
            return Err(FullSpectrumError::SegmentNonIncreasingEnergy {
                segment: segment_index,
                row: 1,
                previous: energy_ev[0],
                current: energy_ev[1],
            });
        }
    }

    let energy_hartree = energy_ev
        .into_iter()
        .map(|energy| energy / FEFF_HARTREE_EV)
        .collect::<Vec<_>>();
    let low_energy = energy_hartree[0];
    let high_energy = energy_hartree[len - 1];

    Ok(PreparedBackgroundSegment {
        energy_hartree,
        f_prime,
        f_double_prime,
        low_energy,
        high_energy,
        output_cache: LintCache::new(),
        sum_cache: LintCache::new(),
    })
}

fn validate_segment_len(
    field: &'static str,
    segment: usize,
    actual: usize,
    expected: usize,
) -> Result<(), FullSpectrumError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FullSpectrumError::SegmentLengthMismatch {
            field,
            segment,
            actual,
            expected,
        })
    }
}

fn interpolate_background_segment(
    segment: &mut PreparedBackgroundSegment,
    omega: ArrayView1<'_, Real>,
    zero_energy_fprime: Real,
    f_prime_out: &mut Array1<Real>,
    f_double_prime_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    let scratch = Vec::from_iter(
        segment
            .energy_hartree
            .iter()
            .zip(segment.f_prime.iter())
            .map(|(energy, f_prime)| {
                if *energy != 0.0 {
                    (*f_prime - zero_energy_fprime) / energy.powi(2)
                } else {
                    0.0
                }
            }),
    );

    segment.output_cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if energy > segment.low_energy && energy < segment.high_energy {
            let interpolated = lint_with_cache(
                &segment.energy_hartree,
                &scratch,
                energy,
                &mut segment.output_cache,
            )
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
            f_prime_out[row] = interpolated * energy.powi(2) + zero_energy_fprime;
        }
    }

    segment.output_cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if energy > segment.low_energy && energy < segment.high_energy {
            let interpolated = lint_with_cache(
                &segment.energy_hartree,
                &segment.f_double_prime,
                energy,
                &mut segment.output_cache,
            )
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
            f_double_prime_out[row] = interpolated.max(0.0);
        }
    }
    Ok(())
}

fn interpolate_background_sum_segment(
    segment: &mut PreparedBackgroundSegment,
    sum_grid: ArrayView1<'_, Real>,
    f_double_prime_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    segment.sum_cache.reset();
    for (row, energy) in sum_grid.iter().copied().enumerate() {
        if energy > segment.low_energy && energy < segment.high_energy {
            let interpolated = lint_with_cache(
                &segment.energy_hartree,
                &segment.f_double_prime,
                energy,
                &mut segment.sum_cache,
            )
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
            f_double_prime_out[row] = interpolated.max(0.0);
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PreparedFineStructureSegment {
    energy_hartree: Vec<Real>,
    wave_number: Vec<Real>,
    scattering_factor: Vec<Real>,
    background: Vec<Real>,
    low_energy: Real,
    high_energy: Real,
    cache: LintCache,
}

#[derive(Debug, Clone, Copy)]
struct FineStructureFmsBounds {
    include_low: bool,
    include_high: bool,
}

fn prepare_fine_structure_segment(
    segment_index: usize,
    input: FullSpectrumFineStructureSegmentInput<'_>,
) -> Result<PreparedFineStructureSegment, FullSpectrumError> {
    let len = input.photon_energy_ev.len();
    if len < 2 {
        return Err(FullSpectrumError::SegmentTooShort {
            name: "fine_structure",
            segment: segment_index,
            len,
        });
    }
    validate_segment_len(
        "wave_number_inverse_angstrom",
        segment_index,
        input.wave_number_inverse_angstrom.len(),
        len,
    )?;
    validate_segment_len(
        "scattering_factor",
        segment_index,
        input.scattering_factor.len(),
        len,
    )?;
    validate_segment_len("background", segment_index, input.background.len(), len)?;

    let energy_ev = input.photon_energy_ev.to_vec();
    let wave_number = input.wave_number_inverse_angstrom.to_vec();
    let scattering_factor = input.scattering_factor.to_vec();
    let background = input.background.to_vec();
    for row in 0..len {
        validate_finite_value("fine_structure photon_energy_ev", row, energy_ev[row])?;
        validate_finite_value("fine_structure wave_number", row, wave_number[row])?;
        validate_finite_value(
            "fine_structure scattering_factor",
            row,
            scattering_factor[row],
        )?;
        validate_finite_value("fine_structure background", row, background[row])?;
        if row > 0 && energy_ev[row] <= energy_ev[row - 1] {
            return Err(FullSpectrumError::SegmentNonIncreasingEnergy {
                segment: segment_index,
                row,
                previous: energy_ev[row - 1],
                current: energy_ev[row],
            });
        }
    }

    let energy_hartree = energy_ev
        .into_iter()
        .map(|energy| energy / FEFF_HARTREE_EV)
        .collect::<Vec<_>>();
    let low_energy = energy_hartree[0];
    let high_energy = energy_hartree[len - 1];

    Ok(PreparedFineStructureSegment {
        energy_hartree,
        wave_number,
        scattering_factor,
        background,
        low_energy,
        high_energy,
        cache: LintCache::new(),
    })
}

fn fine_structure_transition_interval(
    name: &'static str,
    segment: &PreparedFineStructureSegment,
    low_wave_number: Real,
    high_wave_number: Real,
) -> Result<[Real; 2], FullSpectrumError> {
    let low = transition_energy_at_wave_number(name, segment, low_wave_number)?;
    let high = transition_energy_at_wave_number(name, segment, high_wave_number)?;
    if high <= low {
        return Err(FullSpectrumError::InvalidEnergyRange {
            name: "fine_structure_transition",
            min: low,
            max: high,
        });
    }
    Ok([low, high])
}

fn transition_energy_at_wave_number(
    name: &'static str,
    segment: &PreparedFineStructureSegment,
    threshold: Real,
) -> Result<Real, FullSpectrumError> {
    segment
        .energy_hartree
        .iter()
        .copied()
        .zip(segment.wave_number.iter().copied())
        .filter_map(|(energy, wave_number)| (wave_number <= threshold).then_some(energy))
        .next_back()
        .ok_or(FullSpectrumError::MissingTransitionThreshold { name, threshold })
}

fn interpolate_fine_structure_fms(
    segment: &mut PreparedFineStructureSegment,
    omega: ArrayView1<'_, Real>,
    high_transition: Real,
    bounds: FineStructureFmsBounds,
    clamp_nonnegative: bool,
    scattering_factor_out: &mut Array1<Real>,
    background_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    segment.cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if within_fms_interval(energy, segment.low_energy, high_transition, bounds) {
            scattering_factor_out[row] = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.scattering_factor,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
            background_out[row] = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.background,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
        }
    }
    Ok(())
}

fn interpolate_fine_structure_path(
    segment: &mut PreparedFineStructureSegment,
    omega: ArrayView1<'_, Real>,
    transition: [Real; 2],
    clamp_nonnegative: bool,
    scattering_factor_out: &mut Array1<Real>,
    background_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    segment.cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if energy >= transition[0] && energy <= segment.high_energy {
            let (path_weight, fms_weight) = fine_structure_transition_weights(energy, transition);
            let scattering_factor = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.scattering_factor,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
            scattering_factor_out[row] =
                scattering_factor * path_weight + scattering_factor_out[row] * fms_weight;
            let background = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.background,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
            background_out[row] = background * path_weight + background_out[row] * fms_weight;
        }
    }
    Ok(())
}

fn within_fms_interval(
    energy: Real,
    low_energy: Real,
    high_transition: Real,
    bounds: FineStructureFmsBounds,
) -> bool {
    let above_low = if bounds.include_low {
        energy >= low_energy
    } else {
        energy > low_energy
    };
    let below_high = if bounds.include_high {
        energy <= high_transition
    } else {
        energy < high_transition
    };
    above_low && below_high
}

fn fine_structure_transition_weights(energy: Real, transition: [Real; 2]) -> (Real, Real) {
    if energy >= transition[0] && energy <= transition[1] {
        let fraction = (transition[1] - energy) / (transition[1] - transition[0]);
        let theta = fraction * std::f64::consts::FRAC_PI_2;
        (theta.cos().powi(2), theta.sin().powi(2))
    } else {
        (1.0, 0.0)
    }
}

fn maybe_clamp_fine_structure(value: Real, clamp_nonnegative: bool) -> Real {
    if clamp_nonnegative {
        value.max(0.0)
    } else {
        value
    }
}

fn validate_edge_assembly_input(
    input: FullSpectrumEdgeAssemblyInput<'_>,
) -> Result<(), FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "edge_assembly",
        });
    }
    validate_positive("transition_size", input.transition_size)?;
    validate_finite_value("zero_energy_fprime", 0, input.background.zero_energy_fprime)?;
    validate_finite_value(
        "effective_electron_count",
        0,
        input.background.effective_electron_count,
    )?;
    validate_matching_len(
        "background scattering_factor",
        input.background.scattering_factor.len(),
        input.omega.len(),
    )?;
    validate_matching_len(
        "fine_structure scattering_factor",
        input.fine_structure.scattering_factor.len(),
        input.omega.len(),
    )?;
    validate_matching_len(
        "fine_structure background",
        input.fine_structure.background.len(),
        input.omega.len(),
    )?;
    validate_energy_interval(
        "real_energy_interval",
        input.fine_structure.real_energy_interval,
    )?;
    validate_energy_interval(
        "imaginary_energy_interval",
        input.fine_structure.imaginary_energy_interval,
    )?;
    validate_energy_interval(
        "real_transition_interval",
        input.fine_structure.real_transition_interval,
    )?;
    validate_energy_interval(
        "imaginary_transition_interval",
        input.fine_structure.imaginary_transition_interval,
    )?;

    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("edge_assembly omega", row, value)?;
        if row > 0 && value < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: value,
            });
        }
    }
    validate_complex_grid(
        "background scattering_factor",
        input.background.scattering_factor.view(),
    )?;
    validate_complex_grid(
        "fine_structure scattering_factor",
        input.fine_structure.scattering_factor.view(),
    )?;
    validate_complex_grid(
        "fine_structure background",
        input.fine_structure.background.view(),
    )?;
    Ok(())
}

fn validate_energy_interval(
    name: &'static str,
    interval: [Real; 2],
) -> Result<(), FullSpectrumError> {
    validate_finite_value(name, 0, interval[0])?;
    validate_finite_value(name, 1, interval[1])?;
    if interval[1] < interval[0] {
        Err(FullSpectrumError::InvalidEnergyRange {
            name,
            min: interval[0],
            max: interval[1],
        })
    } else {
        Ok(())
    }
}

fn validate_complex_grid(
    field: &'static str,
    values: ArrayView1<'_, Complex>,
) -> Result<(), FullSpectrumError> {
    for (row, value) in values.iter().copied().enumerate() {
        validate_finite_value(field, row, value.re)?;
        validate_finite_value(field, row, value.im)?;
    }
    Ok(())
}

fn last_omega_le(omega: ArrayView1<'_, Real>, limit: Real) -> Option<usize> {
    omega
        .iter()
        .copied()
        .enumerate()
        .rev()
        .find_map(|(row, energy)| (energy <= limit).then_some(row))
}

fn edge_transition_overlap(
    omega: ArrayView1<'_, Real>,
    low_row: Option<usize>,
    high_row: Option<usize>,
    low_energy: Real,
    transition_size: Real,
) -> usize {
    let mut overlap_points = 5;
    if let (Some(start), Some(end)) = (low_row, high_row) {
        for row in start..=end {
            if omega[row] <= low_energy + transition_size {
                overlap_points = row - start + 1;
            }
        }
    }
    overlap_points.max(1)
}

fn apply_real_background_entry(
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    background_overlap: usize,
    overlap: usize,
) {
    if start >= background.len() {
        return;
    }
    for row in start..=(start + background_overlap).min(background.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, background_overlap);
        background[row] = Complex::new(
            background[row].re * cos_squared + fine_background[row].re * sin_squared,
            background[row].im,
        );
    }
    for row in (start + background_overlap)..=(start + overlap).min(background.len() - 1) {
        background[row] = Complex::new(fine_background[row].re, background[row].im);
    }
}

fn apply_imaginary_background_entry(
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    background_overlap: usize,
    overlap: usize,
) {
    if start >= background.len() {
        return;
    }
    for row in start..=(start + background_overlap).min(background.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, background_overlap);
        background[row] = Complex::new(
            background[row].re,
            background[row].im * cos_squared + fine_background[row].im * sin_squared,
        );
    }
    for row in (start + background_overlap)..=(start + overlap).min(background.len() - 1) {
        background[row] = Complex::new(background[row].re, fine_background[row].im);
    }
}

fn apply_real_scattering_entry(
    scattering: &mut Array1<Complex>,
    background: &Array1<Complex>,
    start: usize,
    overlap: usize,
) {
    if start >= scattering.len() {
        return;
    }
    for row in start..=(start + overlap).min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, overlap);
        scattering[row] = Complex::new(
            background[row].re * cos_squared + scattering[row].re * sin_squared,
            scattering[row].im,
        );
    }
}

fn apply_imaginary_scattering_entry(
    scattering: &mut Array1<Complex>,
    background: &Array1<Complex>,
    start: usize,
    overlap: usize,
) {
    if start >= scattering.len() {
        return;
    }
    for row in start..=(start + overlap).min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, overlap);
        scattering[row] = Complex::new(
            scattering[row].re,
            background[row].im * cos_squared + scattering[row].im * sin_squared,
        );
    }
}

fn apply_real_exit(
    scattering: &mut Array1<Complex>,
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    end: usize,
    overlap: usize,
) {
    if start >= scattering.len() {
        return;
    }
    for row in start..=end.min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(end - row, overlap);
        background[row] = Complex::new(
            background[row].re * cos_squared + fine_background[row].re * sin_squared,
            background[row].im,
        );
        scattering[row] = Complex::new(
            background[row].re * cos_squared + scattering[row].re * sin_squared,
            scattering[row].im,
        );
    }
}

fn apply_imaginary_exit(
    scattering: &mut Array1<Complex>,
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    end: usize,
) {
    if start >= scattering.len() || start >= end {
        return;
    }
    let width = end - start;
    for row in start..=end.min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(end - row, width);
        scattering[row] = Complex::new(
            scattering[row].re,
            background[row].im * cos_squared + scattering[row].im * sin_squared,
        );
        background[row] = Complex::new(
            background[row].re,
            background[row].im * cos_squared + fine_background[row].im * sin_squared,
        );
    }
}

fn transition_weights(offset: usize, width: usize) -> (Real, Real) {
    let fraction = offset as Real / width.max(1) as Real;
    let theta = fraction * std::f64::consts::FRAC_PI_2;
    (theta.cos().powi(2), theta.sin().powi(2))
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), FullSpectrumError> {
    if !value.is_finite() {
        Err(FullSpectrumError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FullSpectrumError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteInput { name, value })
    }
}

fn validate_transform_len(name: &'static str, len: usize) -> Result<(), FullSpectrumError> {
    if len >= 2 {
        Ok(())
    } else {
        Err(FullSpectrumError::TooFewRows { name, len })
    }
}

fn validate_strictly_increasing_grid(omega: ArrayView1<'_, Real>) -> Result<(), FullSpectrumError> {
    for (row, value) in omega.iter().copied().enumerate() {
        validate_finite_value("omega", row, value)?;
        if row > 0 && value <= omega[row - 1] {
            return Err(FullSpectrumError::NonIncreasingOmega {
                row,
                previous: omega[row - 1],
                current: value,
            });
        }
    }
    Ok(())
}

fn validate_matching_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), FullSpectrumError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FullSpectrumError::LengthMismatch {
            field,
            actual,
            expected,
        })
    }
}

fn midpoint_grid(omega: ArrayView1<'_, Real>) -> Vec<Real> {
    (0..omega.len() - 1)
        .map(|row| 0.5 * (omega[row + 1] + omega[row]))
        .collect()
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

fn previous_edge_below(edge_energies: ArrayView1<'_, Real>, current: Real) -> Real {
    edge_energies
        .iter()
        .copied()
        .map(|edge| edge.max(0.0))
        .filter(|&edge| edge < current)
        .fold(0.0, Real::max)
}

fn next_edge_above(edge_energies: ArrayView1<'_, Real>, current: Real) -> Real {
    edge_energies
        .iter()
        .copied()
        .map(|edge| edge.max(0.0))
        .filter(|&edge| edge > current)
        .fold(1.0e8, Real::min)
}

fn kk_midpoint_value(
    omega: ArrayView1<'_, Real>,
    epsilon2: ArrayView1<'_, Real>,
    midpoint: Real,
) -> Result<Real, FullSpectrumError> {
    const ANALYTIC_WINDOW: Real = 25.0;
    const TOO_BIG: Real = 1.0e8;

    let mut integral = 0.0;
    for row in (0..omega.len() - 1).rev() {
        let left = omega[row];
        let right = omega[row + 1];
        let contribution =
            if (left - midpoint).abs().max((right - midpoint).abs()) < ANALYTIC_WINDOW {
                let delta = right - left;
                let mut value = epsilon2[row + 1] - epsilon2[row];
                let slope = value / delta;
                let intercept = epsilon2[row] - slope * left;

                let plus_factor = (right + midpoint) / (left + midpoint);
                if plus_factor <= 0.0 || plus_factor >= TOO_BIG {
                    continue;
                }
                value += plus_factor.ln() * (intercept - slope * midpoint) / 2.0;

                let minus_factor = (right - midpoint) / (left - midpoint);
                if minus_factor == 0.0 {
                    continue;
                }
                value += minus_factor.abs().ln() * (intercept + midpoint * slope) / 2.0;
                if value.abs() < TOO_BIG {
                    value
                } else {
                    continue;
                }
            } else {
                let left_denominator = left.powi(2) - midpoint.powi(2);
                let right_denominator = right.powi(2) - midpoint.powi(2);
                let value = right * epsilon2[row + 1] / right_denominator
                    + left * epsilon2[row] / left_denominator;
                value * (right - left) / 2.0
            };
        integral += contribution;
    }

    let value = 2.0 * integral / std::f64::consts::PI;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value })
    }
}

fn hamaker_midpoint_value(
    omega: ArrayView1<'_, Real>,
    epsilon: ArrayView1<'_, Complex>,
    midpoint: Real,
) -> Result<Real, FullSpectrumError> {
    let integral = (0..omega.len() - 1)
        .rev()
        .map(|row| {
            let left = omega[row];
            let right = omega[row + 1];
            let left_value = left * epsilon[row].im / (left.powi(2) + midpoint.powi(2));
            let right_value = right * epsilon[row + 1].im / (right.powi(2) + midpoint.powi(2));
            (left_value + right_value) * (right - left) / 2.0
        })
        .sum::<Real>();
    let value = 2.0 * integral / std::f64::consts::PI;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value })
    }
}

fn interpolate_midpoints_to_grid(
    omega: ArrayView1<'_, Real>,
    midpoints: &[Real],
    midpoint_values: &[Real],
) -> Result<Array1<Real>, FullSpectrumError> {
    let mut output = vec![0.0; omega.len()];
    if omega.len() == 2 {
        output[0] = midpoint_values[0];
        output[1] = midpoint_values[0];
        return Ok(Array1::from_vec(output));
    }

    let mut cache = LintCache::new();
    for row in 1..omega.len() - 1 {
        output[row] = lint_with_cache(midpoints, midpoint_values, omega[row], &mut cache)
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
    }
    output[0] = output[1];
    output[omega.len() - 1] = output[omega.len() - 2];
    Ok(Array1::from_vec(output))
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), FullSpectrumError> {
    if active_len > len {
        Err(FullSpectrumError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_value(
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteValue { field, row, value })
    }
}

fn push_sum_rule_value(
    values: &mut Vec<Real>,
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        values.push(value);
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteSumRule { field, row, value })
    }
}

#[cfg(test)]
mod tests;
