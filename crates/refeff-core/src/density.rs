//! FEFF density and LDOS accumulation helpers.
//!
//! This module ports compact numerical routines that update radial valence
//! densities and angular-momentum-resolved density of states after scattering
//! terms have been computed.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex32;
use thiserror::Error;

use crate::grid::{
    CoulombPotentialSlwInput, GridError, LoucksSphericalOverlapInput, NormanRadius,
    NormanRadiusInput, coulomb_potential_slw, norman_radius_from_density,
    sum_loucks_spherical_overlap,
};
use crate::vector::distance_between;
use crate::{Complex, Real};

const OVRLP_DENSITY_POINTS: usize = 251;
const OVRLP_GEOMETRY_CUTOFF: Real = 12.0;
const COULOM_DELTA: Real = 0.05;
const COULOM_LITERAL_DELTA: Real = 0.05_f32 as Real;
const COULOM_LITERAL_OFFSET: Real = 8.8_f32 as Real;

/// FEFF `POT/coulom.f90` normalization branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoulombUpdateMode {
    /// FEFF default branch (`icoul != 1`) using Norman-sphere charge matching.
    Norman,
    /// FEFF long-range branch (`icoul == 1`) using explicit cluster charge deltas.
    LongRange,
}

/// Inputs for FEFF `POT/coulom.f90` Coulomb-potential correction.
#[derive(Debug, Clone, Copy)]
pub struct CoulombPotentialUpdateInput<'a> {
    /// Normalization branch matching FEFF `icoul`.
    pub mode: CoulombUpdateMode,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are updated.
    pub highest_potential_index: usize,
    /// Last active radial index for each potential, FEFF `ilast`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Valence density `rhoval(radial, potential)`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Overlapped valence density `edenvl(radial, potential)`.
    pub overlapped_valence_density: ArrayView2<'a, Real>,
    /// Overlapped electron density `edens(radial, potential)`.
    pub overlapped_density: ArrayView2<'a, Real>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Norman radii `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Charge deltas `dq`.
    pub charge_deltas: ArrayView1<'a, Real>,
    /// Atomic numbers `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Coulomb potential to correct, FEFF `vclap`.
    pub coulomb_potential: ArrayView2<'a, Real>,
}

/// Corrected FEFF Coulomb potentials from `coulom`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoulombPotentialUpdate {
    /// Updated Coulomb potential `vclap(radial, potential)`.
    pub coulomb_potential: Array2<Real>,
}

/// One FEFF `POT/ovrlp.f90` explicit overlap contribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotentialOverlapNeighbor {
    /// Source potential index `infr` whose free-atom grids are overlapped in.
    pub source_potential: usize,
    /// Number of equivalent neighbors `ann`, from FEFF `nnovr`.
    pub multiplicity: Real,
    /// Neighbor distance `rnn`, in Bohr.
    pub distance: Real,
}

/// Inputs for overlapping one FEFF potential's Coulomb potential and densities.
#[derive(Debug, Clone, Copy)]
pub struct PotentialOverlapInput<'a> {
    /// Potential index `iph` to overlap.
    pub potential_index: usize,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Atomic number for each potential, FEFF `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Explicit overlap list for `potential_index`; empty means geometry mode.
    pub explicit_overlaps: &'a [PotentialOverlapNeighbor],
    /// Free-atom electron density `rho(radial, potential)`.
    pub electron_density: ArrayView2<'a, Real>,
    /// Free-atom spin-density or magnetization `dmag(radial, potential)`.
    pub spin_density: ArrayView2<'a, Real>,
    /// Free-atom valence density `rhoval(radial, potential)`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Free-atom Coulomb potential `vcoul(radial, potential)`.
    pub coulomb_potential: ArrayView2<'a, Real>,
}

/// FEFF `ovrlp` output for one potential.
#[derive(Debug, Clone, PartialEq)]
pub struct PotentialOverlap {
    /// Overlapped Coulomb potential `vclap(:,iph)`.
    pub coulomb_potential: Array1<Real>,
    /// Overlapped electron density `edens(:,iph)`.
    pub electron_density: Array1<Real>,
    /// Overlapped valence-density accumulator `edenvl(:,iph)`.
    pub valence_density: Array1<Real>,
    /// FEFF `dmag(:,iph)` after conversion to `dmag / edens`.
    pub spin_density_ratio: Array1<Real>,
    /// Norman radius computed from the overlapped density.
    pub norman_radius: NormanRadius,
}

/// Inputs for FEFF `POT/ff2g.f90` valence-density accumulation.
#[derive(Debug, Clone, Copy)]
pub struct ValenceDensityUpdateInput<'a> {
    /// Single-precision FMS scattering trace `gtr(0:lx)`.
    pub scattering_trace: ArrayView1<'a, Complex32>,
    /// Zero-based potential column corresponding to FEFF `iph`.
    pub potential_index: usize,
    /// One-based energy index `ie`; `ie == 1` initializes previous-energy work arrays.
    pub energy_index: usize,
    /// One-based last radial point to update, FEFF `ilast`.
    pub last_radial_index: usize,
    /// Scattering contribution to angular-momentum LDOS, `xrhole(0:lx)`.
    pub scattering_ldos: ArrayView1<'a, Complex>,
    /// Embedded-atom LDOS table, indexed as `(l, potential)`, FEFF `xrhoce`.
    pub embedded_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy LDOS table, indexed as `(l, potential)`, FEFF `xrhocp`.
    pub previous_ldos: ArrayView2<'a, Complex>,
    /// Scattering radial density table, indexed as `(radial, l)`, FEFF `yrhole`.
    pub scattering_density: ArrayView2<'a, Complex>,
    /// Embedded radial density for the current potential, FEFF `yrhoce`.
    pub embedded_density: ArrayView1<'a, Complex>,
    /// Previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: ArrayView1<'a, Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: ArrayView1<'a, Real>,
    /// Energy-integrated electron count per angular momentum, FEFF `xnmues`.
    pub occupancy_by_l: ArrayView1<'a, Real>,
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Number of atoms with this potential type, FEFF `xnatph`.
    pub potential_multiplicity: Real,
    /// Current contour-floor flag `iflr`.
    pub current_floor: i32,
    /// Previous contour-floor flag `iflrp`.
    pub previous_floor: i32,
    /// Running left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Running right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Running total valence electron count `xntot`.
    pub total_electron_count: Real,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// Updated FEFF `ff2g` density and LDOS state.
#[derive(Debug, Clone, PartialEq)]
pub struct ValenceDensityUpdate {
    /// Updated embedded-atom LDOS table, FEFF `xrhoce`.
    pub embedded_ldos: Array2<Complex>,
    /// Updated previous-energy LDOS table, FEFF `xrhocp`.
    pub previous_ldos: Array2<Complex>,
    /// Updated embedded radial density, FEFF `yrhoce`.
    pub embedded_density: Array1<Complex>,
    /// Updated previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: Array1<Complex>,
    /// Updated energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: Array1<Real>,
    /// Updated angular-momentum electron counts, FEFF `xnmues`.
    pub occupancy_by_l: Array1<Real>,
    /// Updated left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Updated right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Updated total valence electron count `xntot`.
    pub total_electron_count: Real,
}

/// Error returned by density accumulation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum DensityError {
    /// FEFF indices that select a point count or energy point are 1-based.
    #[error("{name} must be 1-based and positive, got {index}")]
    InvalidIndex { name: &'static str, index: usize },
    /// A vector must contain enough values for the FEFF loop bounds.
    #[error("{name} length {actual} is shorter than required length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// A pair of angular-momentum vectors must have matching lengths.
    #[error("{left_name} length {left_len} does not match {right_name} length {right_len}")]
    LengthMismatch {
        left_name: &'static str,
        left_len: usize,
        right_name: &'static str,
        right_len: usize,
    },
    /// A matrix must have enough rows and columns for the FEFF loop bounds.
    #[error(
        "{name} shape ({rows},{columns}) is smaller than required ({required_rows},{required_columns})"
    )]
    ShapeTooSmall {
        name: &'static str,
        rows: usize,
        columns: usize,
        required_rows: usize,
        required_columns: usize,
    },
    /// A real scalar must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// A real scalar must be positive and finite.
    #[error("{name} must be positive and finite, got {value}")]
    NonPositiveScalar { name: &'static str, value: Real },
    /// A complex scalar must have finite components.
    #[error("{name} must be finite, got ({real},{imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// A real vector entry must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// A complex vector or matrix entry must have finite components.
    #[error("{name}[{index}] must be finite, got ({real},{imaginary})")]
    NonFiniteComplexValue {
        name: &'static str,
        index: usize,
        real: Real,
        imaginary: Real,
    },
    /// Coordinate rows and atom-potential assignments must have matching lengths.
    #[error("atom potential length {potentials} does not match position rows {positions}")]
    AtomPotentialLengthMismatch { potentials: usize, positions: usize },
    /// FEFF Cartesian coordinate tables must be shaped `(atoms, 3)`.
    #[error("atom_positions must have shape (atoms, 3), got ({rows}, {columns})")]
    InvalidPositionShape { rows: usize, columns: usize },
    /// A potential index referenced outside a potential-indexed table.
    #[error(
        "{name} references potential index {index}, but only {available} potentials are available"
    )]
    InvalidPotentialIndex {
        name: &'static str,
        index: usize,
        available: usize,
    },
    /// FEFF radial-grid overlap or Norman-radius calculation failed.
    #[error(transparent)]
    Grid(#[from] GridError),
}

/// Accumulate valence LDOS and radial density from FEFF scattering terms.
///
/// This ports `POT/ff2g.f90`. The routine first folds the single-precision
/// scattering trace into the embedded LDOS, then integrates the current and
/// previous energy endpoints. For `energy_index == 1`, FEFF initializes the
/// previous-energy work arrays from the current values; for later energies it
/// preserves the caller-provided previous state.
pub fn update_valence_density(
    input: ValenceDensityUpdateInput<'_>,
) -> Result<ValenceDensityUpdate, DensityError> {
    validate_valence_density_input(input)?;

    let l_count = input.scattering_trace.len();
    let radial_count = input.last_radial_index;
    let potential = input.potential_index;
    let mut embedded_ldos = input.embedded_ldos.to_owned();
    let mut previous_ldos = input.previous_ldos.to_owned();
    let mut embedded_density = input.embedded_density.to_owned();
    let mut previous_density = input.previous_density.to_owned();
    let mut valence_density = input.valence_density.to_owned();
    let mut occupancy_by_l = input.occupancy_by_l.to_owned();
    let mut left_sum = input.left_sum;
    let mut right_sum = input.right_sum;
    let mut total_electron_count = input.total_electron_count;

    for angular in 0..l_count {
        embedded_ldos[(angular, potential)] +=
            widen_complex32(input.scattering_trace[angular]) * input.scattering_ldos[angular];
        if input.energy_index == 1 {
            previous_ldos[(angular, potential)] = embedded_ldos[(angular, potential)];
        }
    }

    let mut left_step = input.current_energy - input.previous_energy;
    let mut right_step = left_step;
    if input.current_floor == 1 {
        right_step -= Complex::new(0.0, 2.0 * input.current_energy.im);
    }
    if input.previous_floor == 1 {
        left_step += Complex::new(0.0, 2.0 * input.previous_energy.im);
    }

    for angular in 0..l_count {
        if includes_angular_channel(angular, input.include_high_l) {
            left_sum += previous_ldos[(angular, potential)] * (2.0 * input.potential_multiplicity);
            right_sum += embedded_ldos[(angular, potential)] * (2.0 * input.potential_multiplicity);
            occupancy_by_l[angular] += (embedded_ldos[(angular, potential)] * right_step
                + previous_ldos[(angular, potential)] * left_step)
                .im;
            total_electron_count += occupancy_by_l[angular] * input.potential_multiplicity;
        }
    }

    for angular in 0..l_count {
        if includes_angular_channel(angular, input.include_high_l) {
            let trace = widen_complex32(input.scattering_trace[angular]);
            for radial in 0..radial_count {
                embedded_density[radial] += trace * input.scattering_density[(radial, angular)];
                if input.energy_index == 1 {
                    previous_density[radial] = embedded_density[radial];
                }
            }
        }
    }

    for radial in 0..radial_count {
        valence_density[radial] +=
            (embedded_density[radial] * right_step + previous_density[radial] * left_step).im;
    }

    Ok(ValenceDensityUpdate {
        embedded_ldos,
        previous_ldos,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
        left_sum,
        right_sum,
        total_electron_count,
    })
}

/// Correct FEFF Coulomb potentials after a valence-density update.
///
/// This ports `POT/coulom.f90`. FEFF builds a radial Coulomb correction from
/// `rhoval - edenvl`, fixes its additive constant either from cluster charge
/// deltas (`icoul == 1`) or from Norman-sphere charge normalization, then
/// zero-fills the inactive radial tail for each potential.
pub fn update_coulomb_potential(
    input: CoulombPotentialUpdateInput<'_>,
) -> Result<CoulombPotentialUpdate, DensityError> {
    validate_coulomb_update_input(input)?;

    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    let radii = coulom_radii();
    let mut updated = input.coulomb_potential.to_owned();
    for potential in 0..potential_count {
        let active_len = input.last_indices[potential];
        let mut density_delta = Array1::<Real>::zeros(OVRLP_DENSITY_POINTS);
        for radial in 0..active_len {
            density_delta[radial] = (input.valence_density[(radial, potential)]
                - input.overlapped_valence_density[(radial, potential)])
                * radii[radial].powi(2);
        }
        let correction = coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density_delta.view(),
            radii: radii.view(),
            delta: COULOM_DELTA,
            active_len,
        })?;
        let constant_shift = match input.mode {
            CoulombUpdateMode::LongRange => coulom_long_range_shift(
                input,
                potential,
                &radii,
                &density_delta,
                &correction.potential,
            )?,
            CoulombUpdateMode::Norman => {
                coulom_norman_shift(input, potential, &radii, &correction.potential)?
            }
        };

        for radial in 0..active_len {
            updated[(radial, potential)] += correction.potential[radial] + constant_shift;
        }
        for radial in active_len..OVRLP_DENSITY_POINTS {
            updated[(radial, potential)] = 0.0;
        }
    }

    Ok(CoulombPotentialUpdate {
        coulomb_potential: updated,
    })
}

/// Overlap one FEFF potential's free-atom Coulomb potential and densities.
///
/// This ports `POT/ovrlp.f90` for a single `iph`. The routine starts from the
/// current free-atom columns, applies either explicit `OVERLAP`-style neighbor
/// specifications or all geometry neighbors within 12 Bohr, computes the Norman
/// radius from the overlapped electron density, and converts the current
/// spin-density column into FEFF's `dmag / edens` ratio. FEFF initializes
/// `edenvl` from `rhoval(:,iph)` but adds source `rho` during overlaps; this
/// port preserves that behavior.
pub fn overlap_potential_density(
    input: PotentialOverlapInput<'_>,
) -> Result<PotentialOverlap, DensityError> {
    validate_potential_overlap_input(input)?;

    let potential = input.potential_index;
    let mut coulomb_potential = table_column_prefix(input.coulomb_potential, potential);
    let mut electron_density = table_column_prefix(input.electron_density, potential);
    let mut valence_density = table_column_prefix(input.valence_density, potential);

    let neighbors = overlap_neighbors(input)?;
    for neighbor in neighbors {
        let source_coulomb =
            table_column_prefix(input.coulomb_potential, neighbor.source_potential);
        let source_density = table_column_prefix(input.electron_density, neighbor.source_potential);
        coulomb_potential = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: neighbor.distance,
            multiplicity: neighbor.multiplicity,
            source: source_coulomb.view(),
            accumulated: coulomb_potential.view(),
        })?
        .accumulated;
        electron_density = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: neighbor.distance,
            multiplicity: neighbor.multiplicity,
            source: source_density.view(),
            accumulated: electron_density.view(),
        })?
        .accumulated;
        valence_density = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: neighbor.distance,
            multiplicity: neighbor.multiplicity,
            source: source_density.view(),
            accumulated: valence_density.view(),
        })?
        .accumulated;
    }

    let norman_atomic_number = input.atomic_numbers[potential].max(1);
    let norman_radius = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: electron_density.view(),
        atomic_number: norman_atomic_number,
    })?;
    let spin_density_ratio = table_column_prefix(input.spin_density, potential)
        .iter()
        .zip(electron_density.iter())
        .map(|(&spin, &density)| if density > 0.0 { spin / density } else { 0.0 })
        .collect::<Array1<_>>();

    Ok(PotentialOverlap {
        coulomb_potential,
        electron_density,
        valence_density,
        spin_density_ratio,
        norman_radius,
    })
}

fn validate_valence_density_input(
    input: ValenceDensityUpdateInput<'_>,
) -> Result<(), DensityError> {
    if input.energy_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "energy_index",
            index: input.energy_index,
        });
    }
    if input.last_radial_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "last_radial_index",
            index: input.last_radial_index,
        });
    }

    let l_count = input.scattering_trace.len();
    ensure_length_match(
        "scattering_trace",
        l_count,
        "scattering_ldos",
        input.scattering_ldos.len(),
    )?;
    ensure_len("occupancy_by_l", input.occupancy_by_l.len(), l_count)?;
    ensure_len(
        "embedded_density",
        input.embedded_density.len(),
        input.last_radial_index,
    )?;
    ensure_len(
        "previous_density",
        input.previous_density.len(),
        input.last_radial_index,
    )?;
    ensure_len(
        "valence_density",
        input.valence_density.len(),
        input.last_radial_index,
    )?;
    ensure_shape(
        "embedded_ldos",
        input.embedded_ldos.shape(),
        l_count,
        input.potential_index + 1,
    )?;
    ensure_shape(
        "previous_ldos",
        input.previous_ldos.shape(),
        l_count,
        input.potential_index + 1,
    )?;
    ensure_shape(
        "scattering_density",
        input.scattering_density.shape(),
        input.last_radial_index,
        l_count,
    )?;

    validate_complex32_values("scattering_trace", input.scattering_trace)?;
    validate_complex_values("scattering_ldos", input.scattering_ldos.iter().copied())?;
    validate_complex_values("embedded_ldos", input.embedded_ldos.iter().copied())?;
    validate_complex_values("previous_ldos", input.previous_ldos.iter().copied())?;
    validate_complex_values(
        "scattering_density",
        input.scattering_density.iter().copied(),
    )?;
    validate_complex_values("embedded_density", input.embedded_density.iter().copied())?;
    validate_complex_values("previous_density", input.previous_density.iter().copied())?;
    validate_real_values("valence_density", input.valence_density)?;
    validate_real_values("occupancy_by_l", input.occupancy_by_l)?;
    validate_complex_scalar("current_energy", input.current_energy)?;
    validate_complex_scalar("previous_energy", input.previous_energy)?;
    validate_complex_scalar("left_sum", input.left_sum)?;
    validate_complex_scalar("right_sum", input.right_sum)?;
    validate_real_scalar("potential_multiplicity", input.potential_multiplicity)?;
    validate_real_scalar("total_electron_count", input.total_electron_count)?;

    Ok(())
}

fn validate_potential_overlap_input(input: PotentialOverlapInput<'_>) -> Result<(), DensityError> {
    ensure_len(
        "atomic_numbers",
        input.atomic_numbers.len(),
        input.potential_index + 1,
    )?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        input.potential_index + 1,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(DensityError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values(
        "atom_potentials",
        input.atom_potentials,
        input.atomic_numbers.len(),
    )?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;

    let potential_count = input.atomic_numbers.len();
    ensure_shape(
        "electron_density",
        input.electron_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "spin_density",
        input.spin_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "valence_density",
        input.valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "coulomb_potential",
        input.coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("electron_density", input.electron_density)?;
    validate_real_table_values("spin_density", input.spin_density)?;
    validate_real_table_values("valence_density", input.valence_density)?;
    validate_real_table_values("coulomb_potential", input.coulomb_potential)?;

    for neighbor in input.explicit_overlaps {
        if neighbor.source_potential >= potential_count {
            return Err(DensityError::InvalidPotentialIndex {
                name: "explicit_overlaps.source_potential",
                index: neighbor.source_potential,
                available: potential_count,
            });
        }
        validate_positive_real_scalar("explicit_overlaps.distance", neighbor.distance)?;
        validate_real_scalar("explicit_overlaps.multiplicity", neighbor.multiplicity)?;
    }

    Ok(())
}

fn validate_coulomb_update_input(
    input: CoulombPotentialUpdateInput<'_>,
) -> Result<(), DensityError> {
    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    ensure_len("last_indices", input.last_indices.len(), potential_count)?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len("charge_deltas", input.charge_deltas.len(), potential_count)?;
    ensure_len(
        "atomic_numbers",
        input.atomic_numbers.len(),
        potential_count,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(DensityError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values("atom_potentials", input.atom_potentials, potential_count)?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;
    validate_usize_positive_values("atomic_numbers", input.atomic_numbers)?;
    validate_usize_radial_indices("last_indices", input.last_indices)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_real_values("charge_deltas", input.charge_deltas)?;
    for &radius in input.norman_radii.iter().take(potential_count) {
        validate_positive_real_scalar("norman_radii", radius)?;
    }

    ensure_shape(
        "valence_density",
        input.valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "overlapped_valence_density",
        input.overlapped_valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "overlapped_density",
        input.overlapped_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "coulomb_potential",
        input.coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("valence_density", input.valence_density)?;
    validate_real_table_values(
        "overlapped_valence_density",
        input.overlapped_valence_density,
    )?;
    validate_real_table_values("overlapped_density", input.overlapped_density)?;
    validate_real_table_values("coulomb_potential", input.coulomb_potential)
}

fn coulom_long_range_shift(
    input: CoulombPotentialUpdateInput<'_>,
    potential: usize,
    radii: &Array1<Real>,
    density_delta: &Array1<Real>,
    correction: &Array1<Real>,
) -> Result<Real, DensityError> {
    let norman_radius = input.norman_radii[potential];
    let norman_index = coulom_index_for_radius(norman_radius, 2.0)?;
    ensure_coulom_grid_index("norman_radius", norman_index, 2)?;

    let representative = input.representative_atoms[potential];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut boundary_value = input.charge_deltas[potential] / norman_radius;
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        let distance = distance_between(position, center).max(norman_radius);
        boundary_value += input.charge_deltas[input.atom_potentials[atom]] / distance;
    }

    let row = norman_index - 1;
    let dr = radii[row] - norman_radius;
    let slope = (density_delta[row] - density_delta[row - 1]) / (radii[row] - radii[row - 1]);
    boundary_value -= dr / 2.0
        * (input.charge_deltas[potential] / norman_radius.powi(2)
            + (input.charge_deltas[potential] + density_delta[row] * dr
                - slope / 2.0 * dr.powi(2))
                / radii[row].powi(2));
    validate_real_scalar("long_range_shift", boundary_value - correction[row])?;
    Ok(boundary_value - correction[row])
}

fn coulom_norman_shift(
    input: CoulombPotentialUpdateInput<'_>,
    potential: usize,
    radii: &Array1<Real>,
    correction: &Array1<Real>,
) -> Result<Real, DensityError> {
    let density = table_column_prefix(input.overlapped_density, potential);
    let mut combined = Array1::<Real>::zeros(OVRLP_DENSITY_POINTS);
    for radial in 0..OVRLP_DENSITY_POINTS {
        combined[radial] = input.overlapped_density[(radial, potential)]
            - input.overlapped_valence_density[(radial, potential)]
            + input.valence_density[(radial, potential)];
    }
    let radius_original = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: input.atomic_numbers[potential],
    })?
    .radius;
    let radius_updated = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: combined.view(),
        atomic_number: input.atomic_numbers[potential],
    })?
    .radius;

    let rmin = radius_original.min(radius_updated);
    let inrm = coulom_index_for_radius(rmin, 1.0)?;
    ensure_coulom_grid_index("norman_min", inrm, 1)?;
    ensure_coulom_grid_index("norman_min_next", inrm + 1, 1)?;
    let row = inrm - 1;
    let r0 = radii[row];
    let mut delta = 0.0;
    if radius_updated > radius_original {
        let slope = (combined[row + 1] - combined[row]) / (radii[row + 1] - radii[row]);
        let intercept = combined[row] - slope * radii[row];
        delta -= coulom_fab(slope, intercept, r0, radius_original, radius_updated);
    } else {
        let slope = (input.overlapped_density[(row, potential)]
            - input.overlapped_density[(row + 1, potential)])
            / (radii[row + 1] - radii[row]);
        let intercept = -input.overlapped_density[(row, potential)] - slope * radii[row];
        delta -= coulom_fab(slope, intercept, r0, radius_updated, radius_original);
    }
    let slope = (combined[row + 1] - combined[row] + input.overlapped_density[(row, potential)]
        - input.overlapped_density[(row + 1, potential)])
        / (radii[row + 1] - radii[row]);
    let intercept = combined[row] - input.overlapped_density[(row, potential)] - slope * radii[row];
    delta -= coulom_fab(slope, intercept, r0, r0, rmin);
    validate_real_scalar("norman_shift", delta - correction[row])?;
    Ok(delta - correction[row])
}

fn coulom_fab(slope: Real, intercept: Real, r0: Real, r1: Real, r2: Real) -> Real {
    let a2 = (r2.powi(2) - r1.powi(2)) / 2.0;
    let a3 = (r2.powi(3) - r1.powi(3)) / 3.0;
    let a4 = (r2.powi(4) - r1.powi(4)) / 4.0;
    slope * (a4 / r0 - a3) + intercept * (a3 / r0 - a2)
}

fn coulom_index_for_radius(radius: Real, offset: Real) -> Result<usize, DensityError> {
    validate_positive_real_scalar("radius", radius)?;
    Ok(((radius.ln() + COULOM_LITERAL_OFFSET) / COULOM_LITERAL_DELTA + offset).trunc() as usize)
}

fn potential_count_from_highest(index: usize) -> Result<usize, DensityError> {
    index.checked_add(1).ok_or(DensityError::InvalidIndex {
        name: "highest_potential_index",
        index,
    })
}

fn ensure_coulom_grid_index(
    name: &'static str,
    index: usize,
    minimum: usize,
) -> Result<(), DensityError> {
    if index < minimum || index > OVRLP_DENSITY_POINTS {
        return Err(DensityError::InvalidIndex { name, index });
    }
    Ok(())
}

fn coulom_radii() -> Array1<Real> {
    (1..=OVRLP_DENSITY_POINTS)
        .map(|index| (-COULOM_LITERAL_OFFSET + COULOM_DELTA * (index - 1) as Real).exp())
        .collect::<Array1<_>>()
}

fn overlap_neighbors(
    input: PotentialOverlapInput<'_>,
) -> Result<Vec<PotentialOverlapNeighbor>, DensityError> {
    if !input.explicit_overlaps.is_empty() {
        return Ok(input.explicit_overlaps.to_vec());
    }

    let representative = input.representative_atoms[input.potential_index];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut neighbors = Vec::new();
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        let distance = distance_between(position, center);
        if distance <= OVRLP_GEOMETRY_CUTOFF {
            neighbors.push(PotentialOverlapNeighbor {
                source_potential: input.atom_potentials[atom],
                multiplicity: 1.0,
                distance,
            });
        }
    }
    Ok(neighbors)
}

fn table_column_prefix(values: ArrayView2<'_, Real>, column: usize) -> Array1<Real> {
    (0..OVRLP_DENSITY_POINTS)
        .map(|row| values[(row, column)])
        .collect::<Array1<_>>()
}

fn includes_angular_channel(angular: usize, include_high_l: bool) -> bool {
    angular <= 2 || include_high_l
}

fn widen_complex32(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}

fn ensure_len(name: &'static str, actual: usize, required: usize) -> Result<(), DensityError> {
    if actual >= required {
        Ok(())
    } else {
        Err(DensityError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}

fn ensure_length_match(
    left_name: &'static str,
    left_len: usize,
    right_name: &'static str,
    right_len: usize,
) -> Result<(), DensityError> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(DensityError::LengthMismatch {
            left_name,
            left_len,
            right_name,
            right_len,
        })
    }
}

fn ensure_shape(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
) -> Result<(), DensityError> {
    let rows = shape[0];
    let columns = shape[1];
    if rows >= required_rows && columns >= required_columns {
        Ok(())
    } else {
        Err(DensityError::ShapeTooSmall {
            name,
            rows,
            columns,
            required_rows,
            required_columns,
        })
    }
}

fn validate_real_scalar(name: &'static str, value: Real) -> Result<(), DensityError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DensityError::NonFiniteScalar { name, value })
    }
}

fn validate_positive_real_scalar(name: &'static str, value: Real) -> Result<(), DensityError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(DensityError::NonPositiveScalar { name, value })
    }
}

fn validate_complex_scalar(name: &'static str, value: Complex) -> Result<(), DensityError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(DensityError::NonFiniteComplex {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_real_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DensityError::NonFiniteValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_real_table_values(
    name: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DensityError::NonFiniteValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_position_table(positions: ArrayView2<'_, Real>) -> Result<(), DensityError> {
    if positions.ncols() != 3 {
        return Err(DensityError::InvalidPositionShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    validate_real_table_values("atom_positions", positions)
}

fn validate_usize_potential_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
    available: usize,
) -> Result<(), DensityError> {
    for &index in values {
        if index >= available {
            return Err(DensityError::InvalidPotentialIndex {
                name,
                index,
                available,
            });
        }
    }
    Ok(())
}

fn validate_usize_positive_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
) -> Result<(), DensityError> {
    for &index in values {
        if index == 0 {
            return Err(DensityError::InvalidIndex { name, index });
        }
    }
    Ok(())
}

fn validate_usize_radial_indices(
    name: &'static str,
    values: ArrayView1<'_, usize>,
) -> Result<(), DensityError> {
    for &index in values {
        if index == 0 || index > OVRLP_DENSITY_POINTS {
            return Err(DensityError::InvalidIndex { name, index });
        }
    }
    Ok(())
}

fn validate_complex32_values(
    name: &'static str,
    values: ArrayView1<'_, Complex32>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re as Real,
                imaginary: value.im as Real,
            });
        }
    }
    Ok(())
}

fn validate_complex_values(
    name: &'static str,
    values: impl Iterator<Item = Complex>,
) -> Result<(), DensityError> {
    for (index, value) in values.enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re,
                imaginary: value.im,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn valence_density_update_matches_feff_ff2g_first_energy_reference() -> Result<(), DensityError>
    {
        let sample = sample_ff2g_state();

        let result = update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: sample.scattering_trace.view(),
            potential_index: 1,
            energy_index: 1,
            last_radial_index: 5,
            scattering_ldos: sample.scattering_ldos.view(),
            embedded_ldos: sample.embedded_ldos.view(),
            previous_ldos: sample.previous_ldos.view(),
            scattering_density: sample.scattering_density.view(),
            embedded_density: sample.embedded_density.view(),
            previous_density: sample.previous_density.view(),
            valence_density: sample.valence_density.view(),
            occupancy_by_l: sample.occupancy_by_l.view(),
            current_energy: Complex::new(0.72, 0.11),
            previous_energy: Complex::new(0.61, -0.04),
            potential_multiplicity: 2.5,
            current_floor: 1,
            previous_floor: 0,
            left_sum: Complex::new(0.2, -0.1),
            right_sum: Complex::new(-0.3, 0.25),
            total_electron_count: 1.25,
            include_high_l: false,
        })?;

        assert_complex_close(
            result.embedded_ldos[(0, 1)],
            Complex::new(0.451_099_999_919_533_76, -0.215_299_999_862_909_35),
        );
        assert_complex_close(
            result.embedded_ldos[(2, 1)],
            Complex::new(0.539_700_002_484_023_6, -0.211_100_000_292_062_77),
        );
        assert_complex_close(
            result.embedded_ldos[(3, 1)],
            Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
        );
        assert_complex_close(result.previous_ldos[(2, 1)], result.embedded_ldos[(2, 1)]);
        assert_complex_close(
            result.embedded_density[0],
            Complex::new(6.406_000_025_570_392e-2, -1.129_999_976_605_176_9e-2),
        );
        assert_complex_close(
            result.embedded_density[4],
            Complex::new(2.775_000_003_725_290_3e-1, -9.609_999_980_777_503e-2),
        );
        assert_complex_close(result.previous_density[4], result.embedded_density[4]);
        assert_close(result.valence_density[0], 1.263_880_007_192_492_5e-2);
        assert_close(result.valence_density[3], 4.145_320_007_205_01e-2);
        assert_close(result.valence_density[4], 5.105_800_007_209_182e-2);
        assert_close(result.occupancy_by_l[0], -4.127_799_997_627_735e-2);
        assert_close(result.occupancy_by_l[2], -3.265_999_865_531_922_5e-3);
        assert_close(result.occupancy_by_l[3], 1.5e-2);
        assert_close(result.total_electron_count, 1.082_550_000_302_493_7);
        assert_complex_close(
            result.left_sum,
            Complex::new(7.618_000_007_234_514, -3.296_999_999_880_791),
        );
        assert_complex_close(
            result.right_sum,
            Complex::new(7.118_000_007_234_514, -2.946_999_999_880_791_4),
        );
        Ok(())
    }

    #[test]
    fn valence_density_update_matches_feff_ff2g_high_l_reference() -> Result<(), DensityError> {
        let sample = sample_ff2g_state();

        let result = update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: sample.scattering_trace.view(),
            potential_index: 1,
            energy_index: 2,
            last_radial_index: 4,
            scattering_ldos: sample.scattering_ldos.view(),
            embedded_ldos: sample.embedded_ldos.view(),
            previous_ldos: sample.previous_ldos.view(),
            scattering_density: sample.scattering_density.view(),
            embedded_density: sample.embedded_density.view(),
            previous_density: sample.previous_density.view(),
            valence_density: sample.valence_density.view(),
            occupancy_by_l: sample.occupancy_by_l.view(),
            current_energy: Complex::new(0.91, -0.08),
            previous_energy: Complex::new(0.77, 0.05),
            potential_multiplicity: 1.75,
            current_floor: 0,
            previous_floor: 1,
            left_sum: Complex::new(-0.15, 0.09),
            right_sum: Complex::new(0.05, -0.12),
            total_electron_count: -0.2,
            include_high_l: true,
        })?;

        assert_complex_close(
            result.embedded_ldos[(3, 1)],
            Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
        );
        assert_complex_close(result.previous_ldos[(2, 1)], Complex::new(-4.0e-2, 4.5e-2));
        assert_complex_close(
            result.embedded_density[0],
            Complex::new(8.203_999_945_521_354e-2, -1.959_999_881_684_781e-3),
        );
        assert_complex_close(result.embedded_density[4], Complex::new(2.5e-1, -1.0e-1));
        assert_complex_close(result.previous_density[4], Complex::new(-1.5e-1, 2.0e-1));
        assert_close(result.valence_density[0], 5.560_400_087_386_373e-3);
        assert_close(result.valence_density[3], 2.428_160_011_395_812e-2);
        assert_close(result.valence_density[4], 5.0e-2);
        assert_close(result.occupancy_by_l[0], -1.041_849_999_703_466_9e-1);
        assert_close(result.occupancy_by_l[2], -9.221_500_036_381_186e-2);
        assert_close(result.occupancy_by_l[3], -8.732_799_936_085_94e-2);
        assert_close(result.total_electron_count, -8.677_334_992_048_331e-1);
        assert_complex_close(result.left_sum, Complex::new(-8.85e-1, 8.6e-1));
        assert_complex_close(
            result.right_sum,
            Complex::new(7.313_899_995_405_228, -3.091_499_992_907_047_5),
        );
        Ok(())
    }

    #[test]
    fn valence_density_update_rejects_invalid_inputs() {
        let sample = sample_ff2g_state();
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                energy_index: 0,
                ..sample.input()
            }),
            Err(DensityError::InvalidIndex {
                name: "energy_index",
                index: 0,
            })
        );
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                last_radial_index: 0,
                ..sample.input()
            }),
            Err(DensityError::InvalidIndex {
                name: "last_radial_index",
                index: 0,
            })
        );

        let short_ldos = Array1::<Complex>::zeros(2);
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                scattering_ldos: short_ldos.view(),
                ..sample.input()
            }),
            Err(DensityError::LengthMismatch {
                left_name: "scattering_trace",
                left_len: 4,
                right_name: "scattering_ldos",
                right_len: 2,
            })
        );

        let short_density = Array1::<Complex>::zeros(3);
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                embedded_density: short_density.view(),
                ..sample.input()
            }),
            Err(DensityError::LengthTooShort {
                name: "embedded_density",
                required: 5,
                actual: 3,
            })
        );

        let small_matrix = Array2::<Complex>::zeros((2, 2));
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                embedded_ldos: small_matrix.view(),
                ..sample.input()
            }),
            Err(DensityError::ShapeTooSmall {
                name: "embedded_ldos",
                rows: 2,
                columns: 2,
                required_rows: 4,
                required_columns: 2,
            })
        );

        let mut bad_trace = sample.scattering_trace.clone();
        bad_trace[1] = Complex32::new(f32::NAN, 0.0);
        assert!(matches!(
            update_valence_density(ValenceDensityUpdateInput {
                scattering_trace: bad_trace.view(),
                ..sample.input()
            }),
            Err(DensityError::NonFiniteComplexValue {
                name: "scattering_trace",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn coulomb_update_matches_feff_coulom_norman_reference() -> Result<(), DensityError> {
        let sample = sample_coulom_state();
        let result = update_coulomb_potential(CoulombPotentialUpdateInput {
            mode: CoulombUpdateMode::Norman,
            ..sample.input()
        })?;

        assert_coulom_values(
            &result.coulomb_potential,
            0,
            [
                -1.775_572_357_598_355,
                -1.771_572_355_686_939_8,
                -1.523_562_494_397_465,
                -1.201_342_285_954_037_5,
                0.0,
                0.0,
            ],
        );
        assert_coulom_values(
            &result.coulomb_potential,
            1,
            [
                -1.995_771_090_609_550_5,
                -1.991_771_087_961_443_9,
                -1.743_757_425_966_650_6,
                -1.460_139_524_464_028_5,
                0.0,
                0.0,
            ],
        );
        Ok(())
    }

    #[test]
    fn coulomb_update_matches_feff_coulom_long_range_reference() -> Result<(), DensityError> {
        let sample = sample_coulom_state();
        let result = update_coulomb_potential(CoulombPotentialUpdateInput {
            mode: CoulombUpdateMode::LongRange,
            ..sample.input()
        })?;

        assert_coulom_values(
            &result.coulomb_potential,
            0,
            [
                -1.593_233_968_914_050_2,
                -1.589_233_967_002_635,
                -1.341_224_105_713_160_2,
                -1.019_003_897_269_732_6,
                0.0,
                0.0,
            ],
        );
        assert_coulom_values(
            &result.coulomb_potential,
            1,
            [
                -2.000_638_675_823_475_3,
                -1.996_638_673_175_368_7,
                -1.748_625_011_180_575_5,
                -1.465_007_109_677_953_3,
                0.0,
                0.0,
            ],
        );
        Ok(())
    }

    #[test]
    fn coulomb_update_rejects_invalid_inputs() {
        let sample = sample_coulom_state();
        let short = Array1::<usize>::zeros(1);
        assert_eq!(
            update_coulomb_potential(CoulombPotentialUpdateInput {
                last_indices: short.view(),
                ..sample.input()
            }),
            Err(DensityError::LengthTooShort {
                name: "last_indices",
                required: 2,
                actual: 1,
            })
        );

        let bad_last = Array1::from_vec(vec![140, 252]);
        assert_eq!(
            update_coulomb_potential(CoulombPotentialUpdateInput {
                last_indices: bad_last.view(),
                ..sample.input()
            }),
            Err(DensityError::InvalidIndex {
                name: "last_indices",
                index: 252,
            })
        );

        let bad_atoms = Array1::from_vec(vec![0, 2, 1]);
        assert_eq!(
            update_coulomb_potential(CoulombPotentialUpdateInput {
                atom_potentials: bad_atoms.view(),
                ..sample.input()
            }),
            Err(DensityError::InvalidPotentialIndex {
                name: "atom_potentials",
                index: 2,
                available: 2,
            })
        );
    }

    #[test]
    fn potential_overlap_matches_feff_ovrlp_explicit_reference() -> Result<(), DensityError> {
        let sample = sample_ovrlp_state();

        let result = overlap_potential_density(PotentialOverlapInput {
            potential_index: 1,
            explicit_overlaps: &[
                PotentialOverlapNeighbor {
                    source_potential: 0,
                    multiplicity: 2.0,
                    distance: 1.6,
                },
                PotentialOverlapNeighbor {
                    source_potential: 2,
                    multiplicity: 1.0,
                    distance: 2.4,
                },
            ],
            ..sample.input()
        })?;

        assert_overlap_grid_values(
            &result.electron_density,
            [
                1.147_581_324_726_077_3e2,
                1.159_343_448_785_123_5e2,
                1.182_847_850_423_736_6e2,
                7.780_182_969_116_142e1,
            ],
        );
        assert_overlap_grid_values(
            &result.valence_density,
            [
                9.268_692_173_721_425e1,
                9.345_625_940_677_17e1,
                9.499_826_182_402_593e1,
                6.839_302_990_519_607e1,
            ],
        );
        assert_overlap_grid_values(
            &result.coulomb_potential,
            [
                -6.116_080_385_415_219,
                -6.053_852_541_970_13,
                -5.705_837_372_088_443,
                -5.416_140_791_110_99,
            ],
        );
        assert_overlap_grid_values(
            &result.spin_density_ratio,
            [
                2.204_636_783_021_805e-4,
                2.803_310_790_607_974_4e-4,
                4.649_794_982_532_802e-4,
                1.015_400_284_461_108_2e-3,
            ],
        );
        assert_close(result.norman_radius.radius, 6.257_226_100_235_719e-1);
        Ok(())
    }

    #[test]
    fn potential_overlap_matches_feff_ovrlp_geometry_reference() -> Result<(), DensityError> {
        let sample = sample_ovrlp_state();

        let result = overlap_potential_density(PotentialOverlapInput {
            potential_index: 0,
            explicit_overlaps: &[],
            ..sample.input()
        })?;

        assert_overlap_grid_values(
            &result.electron_density,
            [
                8.079_917_195_503_039e1,
                8.198_343_896_737_67e1,
                8.480_691_764_162_866e1,
                5.743_104_082_268_246e1,
            ],
        );
        assert_overlap_grid_values(
            &result.valence_density,
            [
                6.503_424_582_159_577e1,
                6.580_881_910_411_395e1,
                6.765_853_259_870_691e1,
                4.938_868_124_806_859e1,
            ],
        );
        assert_overlap_grid_values(
            &result.coulomb_potential,
            [
                -4.792_432_321_760_901,
                -4.716_935_029_071_595,
                -4.417_627_169_440_431,
                -4.116_381_332_024_653,
            ],
        );
        assert_overlap_grid_values(
            &result.spin_density_ratio,
            [
                2.512_401_984_923_580_4e-4,
                3.354_335_991_070_458_7e-4,
                5.895_745_463_982_858e-4,
                1.288_501_809_125_729_8e-3,
            ],
        );
        assert_close(result.norman_radius.radius, 6.302_380_902_894_656e-1);
        Ok(())
    }

    #[test]
    fn potential_overlap_rejects_invalid_inputs() {
        let sample = sample_ovrlp_state();
        assert_eq!(
            overlap_potential_density(PotentialOverlapInput {
                potential_index: 8,
                ..sample.input()
            }),
            Err(DensityError::LengthTooShort {
                name: "atomic_numbers",
                required: 9,
                actual: 3,
            })
        );

        let bad_positions = Array2::<Real>::zeros((4, 2));
        assert_eq!(
            overlap_potential_density(PotentialOverlapInput {
                atom_positions: bad_positions.view(),
                ..sample.input()
            }),
            Err(DensityError::InvalidPositionShape {
                rows: 4,
                columns: 2,
            })
        );

        let bad_potentials = Array1::from_vec(vec![0, 4, 2, 1]);
        assert_eq!(
            overlap_potential_density(PotentialOverlapInput {
                atom_potentials: bad_potentials.view(),
                ..sample.input()
            }),
            Err(DensityError::InvalidPotentialIndex {
                name: "atom_potentials",
                index: 4,
                available: 3,
            })
        );

        let bad_overlap = [PotentialOverlapNeighbor {
            source_potential: 0,
            multiplicity: 1.0,
            distance: 0.0,
        }];
        assert_eq!(
            overlap_potential_density(PotentialOverlapInput {
                explicit_overlaps: &bad_overlap,
                ..sample.input()
            }),
            Err(DensityError::NonPositiveScalar {
                name: "explicit_overlaps.distance",
                value: 0.0,
            })
        );
    }

    #[derive(Debug, Clone)]
    struct Ff2gSample {
        scattering_trace: Array1<Complex32>,
        scattering_ldos: Array1<Complex>,
        embedded_ldos: Array2<Complex>,
        previous_ldos: Array2<Complex>,
        scattering_density: Array2<Complex>,
        embedded_density: Array1<Complex>,
        previous_density: Array1<Complex>,
        valence_density: Array1<Real>,
        occupancy_by_l: Array1<Real>,
    }

    #[derive(Debug, Clone)]
    struct CoulomSample {
        last_indices: Array1<usize>,
        valence_density: Array2<Real>,
        overlapped_valence_density: Array2<Real>,
        overlapped_density: Array2<Real>,
        atom_positions: Array2<Real>,
        representative_atoms: Array1<usize>,
        atom_potentials: Array1<usize>,
        norman_radii: Array1<Real>,
        charge_deltas: Array1<Real>,
        atomic_numbers: Array1<usize>,
        coulomb_potential: Array2<Real>,
    }

    #[derive(Debug, Clone)]
    struct OvrlpSample {
        atom_potentials: Array1<usize>,
        atom_positions: Array2<Real>,
        representative_atoms: Array1<usize>,
        atomic_numbers: Array1<usize>,
        electron_density: Array2<Real>,
        spin_density: Array2<Real>,
        valence_density: Array2<Real>,
        coulomb_potential: Array2<Real>,
    }

    impl Ff2gSample {
        fn input(&self) -> ValenceDensityUpdateInput<'_> {
            ValenceDensityUpdateInput {
                scattering_trace: self.scattering_trace.view(),
                potential_index: 1,
                energy_index: 1,
                last_radial_index: 5,
                scattering_ldos: self.scattering_ldos.view(),
                embedded_ldos: self.embedded_ldos.view(),
                previous_ldos: self.previous_ldos.view(),
                scattering_density: self.scattering_density.view(),
                embedded_density: self.embedded_density.view(),
                previous_density: self.previous_density.view(),
                valence_density: self.valence_density.view(),
                occupancy_by_l: self.occupancy_by_l.view(),
                current_energy: Complex::new(0.72, 0.11),
                previous_energy: Complex::new(0.61, -0.04),
                potential_multiplicity: 2.5,
                current_floor: 1,
                previous_floor: 0,
                left_sum: Complex::new(0.2, -0.1),
                right_sum: Complex::new(-0.3, 0.25),
                total_electron_count: 1.25,
                include_high_l: false,
            }
        }
    }

    impl CoulomSample {
        fn input(&self) -> CoulombPotentialUpdateInput<'_> {
            CoulombPotentialUpdateInput {
                mode: CoulombUpdateMode::Norman,
                highest_potential_index: 1,
                last_indices: self.last_indices.view(),
                valence_density: self.valence_density.view(),
                overlapped_valence_density: self.overlapped_valence_density.view(),
                overlapped_density: self.overlapped_density.view(),
                atom_positions: self.atom_positions.view(),
                representative_atoms: self.representative_atoms.view(),
                atom_potentials: self.atom_potentials.view(),
                norman_radii: self.norman_radii.view(),
                charge_deltas: self.charge_deltas.view(),
                atomic_numbers: self.atomic_numbers.view(),
                coulomb_potential: self.coulomb_potential.view(),
            }
        }
    }

    impl OvrlpSample {
        fn input(&self) -> PotentialOverlapInput<'_> {
            PotentialOverlapInput {
                potential_index: 1,
                atom_potentials: self.atom_potentials.view(),
                atom_positions: self.atom_positions.view(),
                representative_atoms: self.representative_atoms.view(),
                atomic_numbers: self.atomic_numbers.view(),
                explicit_overlaps: &[],
                electron_density: self.electron_density.view(),
                spin_density: self.spin_density.view(),
                valence_density: self.valence_density.view(),
                coulomb_potential: self.coulomb_potential.view(),
            }
        }
    }

    fn sample_ff2g_state() -> Ff2gSample {
        let l_count = 4;
        let potential_count = 3;
        let radial_count = 251;
        let scattering_trace = (0..l_count)
            .map(|angular| {
                let l = angular as Real;
                Complex32::new(
                    ((0.05_f32 as Real) * l + 0.11_f32 as Real) as f32,
                    ((-0.03_f32 as Real) * l + 0.07_f32 as Real) as f32,
                )
            })
            .collect::<Array1<_>>();
        let scattering_ldos = (0..l_count)
            .map(|angular| {
                let l = angular as Real;
                Complex::new(0.2 + 0.04 * l, -0.13 + 0.02 * l)
            })
            .collect::<Array1<_>>();
        let mut embedded_ldos = Array2::<Complex>::zeros((l_count, potential_count));
        let mut previous_ldos = Array2::<Complex>::zeros((l_count, potential_count));
        for angular in 0..l_count {
            let l = angular as Real;
            for potential in 0..potential_count {
                let p = potential as Real;
                embedded_ldos[(angular, potential)] =
                    Complex::new(0.4 + 0.03 * l + 0.02 * p, -0.2 + 0.01 * l - 0.015 * p);
                previous_ldos[(angular, potential)] =
                    Complex::new(-0.1 + 0.025 * l + 0.01 * p, 0.08 - 0.02 * l + 0.005 * p);
            }
        }
        let embedded_density = (1..=radial_count)
            .map(|radial| {
                let r = radial as Real;
                Complex::new(0.05 * r, -0.02 * r)
            })
            .collect::<Array1<_>>();
        let previous_density = (1..=radial_count)
            .map(|radial| {
                let r = radial as Real;
                Complex::new(-0.03 * r, 0.04 * r)
            })
            .collect::<Array1<_>>();
        let valence_density = (1..=radial_count)
            .map(|radial| 0.01 * radial as Real)
            .collect::<Array1<_>>();
        let mut scattering_density = Array2::<Complex>::zeros((radial_count, l_count));
        for radial in 0..radial_count {
            let r = (radial + 1) as Real;
            for angular in 0..l_count {
                let l = angular as Real;
                scattering_density[(radial, angular)] =
                    Complex::new(0.006 * r + 0.02 * l, -0.004 * r + 0.015 * l);
            }
        }
        let occupancy_by_l = (0..l_count)
            .map(|angular| -0.03 + 0.015 * angular as Real)
            .collect::<Array1<_>>();

        Ff2gSample {
            scattering_trace,
            scattering_ldos,
            embedded_ldos,
            previous_ldos,
            scattering_density,
            embedded_density,
            previous_density,
            valence_density,
            occupancy_by_l,
        }
    }

    fn sample_coulom_state() -> CoulomSample {
        let last_indices = Array1::from_vec(vec![140, 132]);
        let atom_potentials = Array1::from_vec(vec![0, 1, 1]);
        let representative_atoms = Array1::from_vec(vec![0, 1]);
        let norman_radii = Array1::from_vec(vec![0.65, 0.82]);
        let charge_deltas = Array1::from_vec(vec![0.15, -0.07]);
        let atomic_numbers = Array1::from_vec(vec![8, 14]);
        let mut atom_positions = Array2::<Real>::zeros((3, 3));
        for (atom, position) in [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.1, 0.0]]
            .into_iter()
            .enumerate()
        {
            for axis in 0..3 {
                atom_positions[(atom, axis)] = position[axis];
            }
        }

        let mut valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
        let mut overlapped_valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
        let mut overlapped_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
        let mut coulomb_potential = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
        for potential in 0..=1 {
            let p = potential as Real;
            for index in 1..=OVRLP_DENSITY_POINTS {
                let radius = (-8.8 + 0.05 * (index - 1) as Real).exp();
                let density = (80.0 + 15.0 * p) * (-0.85 * radius).exp() / (1.0 + 0.12 * radius);
                overlapped_density[(index - 1, potential)] = density;
                overlapped_valence_density[(index - 1, potential)] = (0.42 + 0.03 * p) * density;
                valence_density[(index - 1, potential)] = (0.36 + 0.02 * p) * density;
                coulomb_potential[(index - 1, potential)] = -1.7 - 0.25 * p + 0.004 * index as Real;
            }
        }

        CoulomSample {
            last_indices,
            valence_density,
            overlapped_valence_density,
            overlapped_density,
            atom_positions,
            representative_atoms,
            atom_potentials,
            norman_radii,
            charge_deltas,
            atomic_numbers,
            coulomb_potential,
        }
    }

    fn sample_ovrlp_state() -> OvrlpSample {
        let atom_potentials = Array1::from_vec(vec![0, 1, 2, 1]);
        let mut atom_positions = Array2::<Real>::zeros((4, 3));
        for (atom, position) in [
            [0.0, 0.0, 0.0],
            [1.35, 0.2, -0.15],
            [3.10, -0.4, 0.25],
            [13.5, 0.0, 0.0],
        ]
        .into_iter()
        .enumerate()
        {
            for axis in 0..3 {
                atom_positions[(atom, axis)] = position[axis];
            }
        }
        let representative_atoms = Array1::from_vec(vec![0, 1, 2]);
        let atomic_numbers = Array1::from_vec(vec![6, 8, 14]);
        let mut electron_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
        let mut spin_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
        let mut valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
        let mut coulomb_potential = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
        for potential in 0..4 {
            let p = potential as Real;
            for index in 1..=OVRLP_DENSITY_POINTS {
                let i = index as Real;
                let radius = legacy_loucks_radius(index);
                let density =
                    (45.0 + 18.0 * p) * (-(1.0 + 0.08 * p) * radius).exp() + 0.05 * (i + p);
                electron_density[(index - 1, potential)] = density;
                valence_density[(index - 1, potential)] = 0.65 * density + 0.01 * p + 0.0002 * i;
                coulomb_potential[(index - 1, potential)] =
                    -2.0 - 0.12 * p + 0.004 * i + 0.03 * (0.05 * i + p).cos();
                spin_density[(index - 1, potential)] = 0.02 + 0.0003 * i + 0.005 * p;
            }
        }

        OvrlpSample {
            atom_potentials,
            atom_positions,
            representative_atoms,
            atomic_numbers,
            electron_density,
            spin_density,
            valence_density,
            coulomb_potential,
        }
    }

    fn legacy_loucks_radius(index_1based: usize) -> Real {
        ((0.05_f32 as Real) * (index_1based as Real - 1.0) - 8.8_f32 as Real).exp()
    }

    fn assert_coulom_values(values: &Array2<Real>, potential: usize, expected: [Real; 6]) {
        let indices = [
            1,
            2,
            64,
            if potential == 0 { 140 } else { 132 },
            if potential == 0 { 141 } else { 133 },
            251,
        ];
        for (index, expected_value) in indices.into_iter().zip(expected) {
            assert_close(values[(index - 1, potential)], expected_value);
        }
    }

    fn assert_overlap_grid_values(values: &Array1<Real>, expected: [Real; 4]) {
        const OVRLP_ORACLE_TOLERANCE: Real = 5.0e-7;

        for (index, expected_value) in [1, 25, 100, 180].into_iter().zip(expected) {
            assert!(
                (values[index - 1] - expected_value).abs() <= OVRLP_ORACLE_TOLERANCE,
                "{} != {}",
                values[index - 1],
                expected_value
            );
        }
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-8_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}
