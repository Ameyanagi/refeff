//! Source-backed FEFF `apot.bin` sections produced from atomic SCF state data.

use ndarray::{Array2, ArrayView1, ArrayView2};
use refeff_core::atomic::AtomicScfState;

use crate::error::Result;
use crate::pot_bin::{POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS};

use super::common::{invalid_apot_bin, invalid_apot_bin_value, validate_finite};
use super::types::{
    ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords, ApotBinSection,
    ApotBinType, ApotBinValue,
};

/// Section number of FEFF `norb(0:nphx+1)`.
pub const APOT_ATOMIC_NORB_SECTION_NUMBER: usize = 3;
/// Section number of FEFF `rho(r,0:nphx+1)`.
pub const APOT_ATOMIC_DENSITY_SECTION_NUMBER: usize = 8;
/// Section number of FEFF `rhoval(r,0:nphx+1)`.
pub const APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER: usize = 10;
/// Section number of FEFF `vcoul(r,0:nphx+1)`.
pub const APOT_ATOMIC_COULOMB_SECTION_NUMBER: usize = 11;
/// Section number of FEFF `xnval(i,0:nphx+1)`.
pub const APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER: usize = 13;
/// Section number of FEFF `eorb(i,0:nphx+1)`.
pub const APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER: usize = 14;
/// Section number of FEFF `kappa(i,0:nphx+1)`.
pub const APOT_ATOMIC_KAPPA_SECTION_NUMBER: usize = 20;
/// First section number of the FEFF per-state `dgc`, `dpc`, `adgc`, `adpc` blocks.
pub const APOT_ATOMIC_ORBITAL_SECTION_START: usize = 22;

/// FEFF radial mesh size used by atomic `apot.bin` arrays.
pub const APOT_ATOMIC_RADIAL_POINTS: usize = POT_BIN_RADIAL_POINTS;
/// FEFF orbital slot count used by atomic `apot.bin` arrays.
pub const APOT_ATOMIC_ORBITAL_SLOTS: usize = POT_BIN_ORBITALS;
/// FEFF coefficient count used by atomic `apot.bin` arrays.
pub const APOT_ATOMIC_COEFFICIENTS: usize = POT_BIN_COEFFICIENTS;

/// Input for building a FEFF `WriteAtomicPots`-ordered `apot.bin` section stream.
#[derive(Debug, Clone, Copy)]
pub struct ApotAtomicPotsSectionsInput<'a, 'states> {
    /// FEFF `nph`, the highest unique-potential index.
    pub unique_potential_count: usize,
    /// FEFF `nat`, the number of atoms in the cluster.
    pub atom_count: usize,
    /// FEFF `ihole` core-hole index.
    pub hole_index: i64,
    /// FEFF relaxation energy estimate `erelax`.
    pub relaxation_energy: f64,
    /// FEFF edge energy `emu`.
    pub edge_energy: f64,
    /// FEFF many-body amplitude reduction `s02`.
    pub amplitude_reduction: f64,
    /// Atomic number for each unique potential, FEFF `iz(0:nph)`.
    pub atomic_numbers: ArrayView1<'a, i64>,
    /// Model atom index for each unique potential, FEFF `iatph(0:nph)`.
    pub model_atom_indices: ArrayView1<'a, i64>,
    /// Overlap-shell count for each unique potential, FEFF `novr(0:nph)`.
    pub overlap_shell_counts: ArrayView1<'a, i64>,
    /// Norman radius for each unique potential, FEFF `rnrm(0:nph)`.
    pub norman_radii: ArrayView1<'a, f64>,
    /// Unique-potential index for each atom, FEFF `iphat(nat)`.
    pub atom_potential_indices: ArrayView1<'a, i64>,
    /// Core-hole large component, FEFF `dgc0`.
    pub core_hole_large_component: ArrayView1<'a, f64>,
    /// Core-hole small component, FEFF `dpc0`.
    pub core_hole_small_component: ArrayView1<'a, f64>,
    /// Core-hole density, FEFF `drho`.
    pub core_hole_density: ArrayView1<'a, f64>,
    /// Core-hole Coulomb potential, FEFF `dvcoul`.
    pub core_hole_coulomb_potential: ArrayView1<'a, f64>,
    /// Unique potential per overlap shell, FEFF `iphovr(novr,0:nph)`.
    pub overlap_potential_indices: ArrayView2<'a, i64>,
    /// Atom count per overlap shell, FEFF `nnovr(novr,0:nph)`.
    pub overlap_shell_atom_counts: ArrayView2<'a, i64>,
    /// Magnetization density table, FEFF `dmag(r,0:nph+1)`.
    pub magnetization_density: ArrayView2<'a, f64>,
    /// Norman-sphere valence counts per angular channel, FEFF `xnvmu`.
    pub norman_valence_counts: ArrayView2<'a, f64>,
    /// Cartesian atom positions, FEFF `rat(3,nat)`.
    pub atom_positions: ArrayView2<'a, f64>,
    /// Overlap-shell radii, FEFF `rovr(novr,0:nph)`.
    pub overlap_radii: ArrayView2<'a, f64>,
    /// Overlapped density for each unique potential, FEFF `edens(r,0:nph)`.
    pub overlapped_density: ArrayView2<'a, f64>,
    /// Overlapped valence density for each unique potential, FEFF `edenvl(r,0:nph)`.
    pub overlapped_valence_density: ArrayView2<'a, f64>,
    /// Overlapped Coulomb potential for each unique potential, FEFF `vclap(r,0:nph)`.
    pub overlapped_coulomb_potential: ArrayView2<'a, f64>,
    /// Last occupied orbital per kappa slot and state, FEFF `iorb(-5:4,0:nph+1)`.
    pub orbital_indices_by_kappa: ArrayView2<'a, i64>,
    /// Complete source-backed SCF states for FEFF state columns `0..=nph+1`.
    pub states: &'states [ApotAtomicScfStateSectionsInput<'a>],
}

/// Input for building the source-backed `apot.bin` sections for one FEFF atomic state.
///
/// `state_count` is the full FEFF `0:nph+1` column count so the per-state orbital
/// sections can be numbered exactly as FEFF's `WriteAtomicPots` does. Only
/// `state_index` is populated; the other columns in shared matrices are zero.
#[derive(Debug, Clone)]
pub struct ApotAtomicScfStateSectionsInput<'a> {
    /// Total number of FEFF atomic states/potential columns in `apot.bin`.
    pub state_count: usize,
    /// Zero-based state column populated by this source-backed state.
    pub state_index: usize,
    /// Number of occupied/active orbitals to expose from the compact SCF tables.
    pub orbital_count: usize,
    /// Total electron density multiplied by `4*pi`, on FEFF's 251-point radial grid.
    pub density_4pi: ArrayView1<'a, f64>,
    /// Coulomb potential on FEFF's 251-point radial grid.
    pub coulomb_potential: ArrayView1<'a, f64>,
    /// Valence electron density multiplied by `4*pi`, on FEFF's radial grid.
    pub valence_density_4pi: ArrayView1<'a, f64>,
    /// Valence occupations for the first `orbital_count` compact orbitals.
    pub valence_occupations: ArrayView1<'a, f64>,
    /// Orbital eigenvalues for the first `orbital_count` compact orbitals.
    pub orbital_energies: ArrayView1<'a, f64>,
    /// Dirac kappa values for the first `orbital_count` compact orbitals.
    pub kappas: ArrayView1<'a, i32>,
    /// Large Dirac radial components shaped `(251, orbital_count_or_more)`.
    pub large_components: ArrayView2<'a, f64>,
    /// Small Dirac radial components shaped `(251, orbital_count_or_more)`.
    pub small_components: ArrayView2<'a, f64>,
    /// Large-component origin coefficients shaped `(10, orbital_count_or_more)`.
    pub large_coefficients: ArrayView2<'a, f64>,
    /// Small-component origin coefficients shaped `(10, orbital_count_or_more)`.
    pub small_coefficients: ArrayView2<'a, f64>,
}

impl<'a> ApotAtomicScfStateSectionsInput<'a> {
    /// Borrow the SCF-dependent `apot.bin` columns from a core ATOM state.
    #[must_use]
    pub fn from_atomic_scf_state(
        state_count: usize,
        state_index: usize,
        state: &'a AtomicScfState,
    ) -> Self {
        Self {
            state_count,
            state_index,
            orbital_count: state.occupations.len(),
            density_4pi: state.scf.density_4pi.view(),
            coulomb_potential: state.scf.coulomb_potential.view(),
            valence_density_4pi: state.scf.valence_density_4pi.view(),
            valence_occupations: state.valence_occupations.view(),
            orbital_energies: state.scf.orbital_energies.view(),
            kappas: state.kappas.view(),
            large_components: state.scf.large_components.view(),
            small_components: state.scf.small_components.view(),
            large_coefficients: state.scf.large_coefficients.view(),
            small_coefficients: state.scf.small_coefficients.view(),
        }
    }
}

/// Borrowed core ATOM state with its FEFF `apot.bin` state column index.
#[derive(Debug, Clone, Copy)]
pub struct ApotAtomicScfStateRef<'a> {
    /// Zero-based state column populated by this source-backed state.
    pub state_index: usize,
    /// Source-backed core ATOM state.
    pub state: &'a AtomicScfState,
}

/// Build the `apot.bin` sections consumed by current source-backed ATOM handoffs
/// for one populated FEFF atomic state.
///
/// The returned stream contains section numbers 3, 8, 11, 13, 14, 20 and the
/// populated state's `dgc`, `dpc`, `adgc`, and `adpc` matrix sections. It is a
/// focused subset of FEFF's full `WriteAtomicPots` output, intended for
/// downstream readers that query sections by number.
pub fn apot_atomic_scf_state_sections(
    input: ApotAtomicScfStateSectionsInput<'_>,
) -> Result<Vec<ApotBinSection>> {
    apot_atomic_scf_sections(&[input])
}

/// Build source-backed `apot.bin` sections directly from one core ATOM state.
pub fn apot_atomic_scf_state_sections_from_state(
    state_count: usize,
    state_index: usize,
    state: &AtomicScfState,
) -> Result<Vec<ApotBinSection>> {
    apot_atomic_scf_state_sections(ApotAtomicScfStateSectionsInput::from_atomic_scf_state(
        state_count,
        state_index,
        state,
    ))
}

/// Build merged `apot.bin` sections for one or more source-backed atomic SCF states.
///
/// Shared FEFF matrices are emitted once with one column per `state_count`.
/// Per-state orbital matrices are emitted in FEFF `WriteAtomicPots` group
/// order: all `dgc` state sections, then all `dpc`, `adgc`, and `adpc`
/// sections for the provided states. Missing states remain zero-filled in the
/// shared matrices and do not get per-state orbital sections.
pub fn apot_atomic_scf_sections<'view>(
    states: &[ApotAtomicScfStateSectionsInput<'view>],
) -> Result<Vec<ApotBinSection>> {
    let states = validate_state_inputs(states)?;
    let state_count = states[0].state_count;
    scf_state_sections_from_sorted(state_count, &states)
}

/// Build merged source-backed `apot.bin` sections directly from core ATOM states.
pub fn apot_atomic_scf_sections_from_states(
    state_count: usize,
    states: &[ApotAtomicScfStateRef<'_>],
) -> Result<Vec<ApotBinSection>> {
    let inputs = states
        .iter()
        .map(|state| {
            ApotAtomicScfStateSectionsInput::from_atomic_scf_state(
                state_count,
                state.state_index,
                state.state,
            )
        })
        .collect::<Vec<_>>();
    apot_atomic_scf_sections(&inputs)
}

/// Build a FEFF `WriteAtomicPots`-ordered `apot.bin` section stream.
///
/// This assembles sections 1 through 21 plus the per-state orbital matrix
/// sections from complete source-backed SCF states. It does not run the ATOM
/// numerical solver; callers must supply the already-computed source arrays.
pub fn apot_atomic_pots_sections<'view>(
    input: ApotAtomicPotsSectionsInput<'view, '_>,
) -> Result<Vec<ApotBinSection>> {
    let unique_count = checked_add_one(input.unique_potential_count, "nph+1")?;
    let state_count = checked_add(input.unique_potential_count, 2, "nph+2")?;
    validate_atomic_pots_input(&input, unique_count, state_count)?;

    let states = validate_state_inputs(input.states)?;
    validate_complete_states(&states, state_count)?;
    let mut state_sections = scf_state_sections_from_sorted(state_count, &states)?;

    let mut sections = vec![
        atomic_scalar_section(&input)?,
        unique_potential_section(&input, unique_count)?,
        remove_section(&mut state_sections, APOT_ATOMIC_NORB_SECTION_NUMBER)?,
        atom_potential_section(input.atom_potential_indices)?,
        core_hole_section(&input),
        int_matrix_section(
            6,
            "iphovr(novrx,0:nphx) - unique pot for each overlap shell",
            input.overlap_potential_indices.to_owned(),
        ),
        int_matrix_section(
            7,
            "nnovr(novrx,0:nphx) - number of atoms in overlap shell",
            input.overlap_shell_atom_counts.to_owned(),
        ),
        remove_section(&mut state_sections, APOT_ATOMIC_DENSITY_SECTION_NUMBER)?,
        real_matrix_section(
            9,
            "dmag(r,nph+1) - magnetization density",
            input.magnetization_density.to_owned(),
        ),
        remove_section(
            &mut state_sections,
            APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER,
        )?,
        remove_section(&mut state_sections, APOT_ATOMIC_COULOMB_SECTION_NUMBER)?,
        real_matrix_section(
            12,
            "xnvmu(0:lx,0:nphx+1) - valence electrons within Norman sphere",
            input.norman_valence_counts.to_owned(),
        ),
        remove_section(
            &mut state_sections,
            APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
        )?,
        remove_section(
            &mut state_sections,
            APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER,
        )?,
        real_matrix_section(
            15,
            "rat(3,nat) - cartesian coordinates for each atom",
            input.atom_positions.to_owned(),
        ),
        real_matrix_section(
            16,
            "rovr(novrx,0:nphx) - r for overlap shell",
            input.overlap_radii.to_owned(),
        ),
        real_matrix_section(
            17,
            "edens(r,0:nphx) - overlapped density for each unique potential",
            input.overlapped_density.to_owned(),
        ),
        real_matrix_section(
            18,
            "edenvl(r,0:nphx) - overlapped valence density for each unique potential",
            input.overlapped_valence_density.to_owned(),
        ),
        real_matrix_section(
            19,
            "vclap(r,0:nphx) - overlapped coulomb potential",
            input.overlapped_coulomb_potential.to_owned(),
        ),
        remove_section(&mut state_sections, APOT_ATOMIC_KAPPA_SECTION_NUMBER)?,
        int_matrix_section(
            21,
            "iorb(-5:4,0:nphx+1) - last occupied orbital of a particular kappa",
            input.orbital_indices_by_kappa.to_owned(),
        ),
    ];

    sections.extend(state_sections);
    Ok(sections)
}

fn scf_state_sections_from_sorted(
    state_count: usize,
    states: &[&ApotAtomicScfStateSectionsInput<'_>],
) -> Result<Vec<ApotBinSection>> {
    let mut sections = vec![
        norb_section(state_count, states)?,
        real_state_matrix_section_for_states(
            APOT_ATOMIC_DENSITY_SECTION_NUMBER,
            "rho(r,0:nphx+1) - atomic density for each unique potential",
            state_count,
            states,
            |state| state.density_4pi,
        ),
        real_state_matrix_section_for_states(
            APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER,
            "rhoval(r,0:nphx+1) - valence density for each unique potential",
            state_count,
            states,
            |state| state.valence_density_4pi,
        ),
        real_state_matrix_section_for_states(
            APOT_ATOMIC_COULOMB_SECTION_NUMBER,
            "vcoul(r,0:nphx+1) - coulomb potential for each unique potential",
            state_count,
            states,
            |state| state.coulomb_potential,
        ),
        real_orbital_state_matrix_section_for_states(
            APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
            "xnval(i,0:nphx+1) - valence orbital occupations",
            state_count,
            states,
            |state| state.valence_occupations,
        ),
        real_orbital_state_matrix_section_for_states(
            APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER,
            "eorb(i,0:nphx+1) - orbital energies",
            state_count,
            states,
            |state| state.orbital_energies,
        ),
        kappa_section(state_count, states),
    ];

    for group_index in 0..4 {
        for state in states {
            let (header, rows, source) = match group_index {
                0 => (
                    "dgc(r,i,iph) - large Dirac component",
                    APOT_ATOMIC_RADIAL_POINTS,
                    state.large_components,
                ),
                1 => (
                    "dpc(r,i,iph) - small Dirac component",
                    APOT_ATOMIC_RADIAL_POINTS,
                    state.small_components,
                ),
                2 => (
                    "adgc(j,i,iph) - large-component origin coefficients",
                    APOT_ATOMIC_COEFFICIENTS,
                    state.large_coefficients,
                ),
                _ => (
                    "adpc(j,i,iph) - small-component origin coefficients",
                    APOT_ATOMIC_COEFFICIENTS,
                    state.small_coefficients,
                ),
            };
            sections.push(real_orbital_component_section(
                orbital_section_number(state_count, state.state_index, group_index)?,
                header,
                rows,
                state.orbital_count,
                source,
            ));
        }
    }

    Ok(sections)
}

fn validate_atomic_pots_input(
    input: &ApotAtomicPotsSectionsInput<'_, '_>,
    unique_count: usize,
    state_count: usize,
) -> Result<()> {
    validate_finite_scalar("erelax", input.relaxation_energy)?;
    validate_finite_scalar("emu", input.edge_energy)?;
    validate_finite_scalar("s02", input.amplitude_reduction)?;
    validate_vector_len("iz", input.atomic_numbers.len(), unique_count)?;
    validate_vector_len("iatph", input.model_atom_indices.len(), unique_count)?;
    validate_vector_len("novr", input.overlap_shell_counts.len(), unique_count)?;
    validate_vector_len("rnrm", input.norman_radii.len(), unique_count)?;
    validate_vector_len(
        "iphat",
        input.atom_potential_indices.len(),
        input.atom_count,
    )?;
    validate_vector_len(
        "dgc0",
        input.core_hole_large_component.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    validate_vector_len(
        "dpc0",
        input.core_hole_small_component.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    validate_vector_len(
        "drho",
        input.core_hole_density.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    validate_vector_len(
        "dvcoul",
        input.core_hole_coulomb_potential.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    let overlap_shape = input.overlap_potential_indices.dim();
    validate_matrix_nonempty_columns("iphovr", overlap_shape, unique_count)?;
    validate_matrix_exact(
        "nnovr",
        input.overlap_shell_atom_counts.dim(),
        overlap_shape.0,
        unique_count,
    )?;
    validate_matrix_exact(
        "rovr",
        input.overlap_radii.dim(),
        overlap_shape.0,
        unique_count,
    )?;
    validate_matrix_exact(
        "dmag",
        input.magnetization_density.dim(),
        APOT_ATOMIC_RADIAL_POINTS,
        state_count,
    )?;
    validate_matrix_nonempty_columns("xnvmu", input.norman_valence_counts.dim(), state_count)?;
    validate_matrix_exact("rat", input.atom_positions.dim(), 3, input.atom_count)?;
    validate_matrix_exact(
        "edens",
        input.overlapped_density.dim(),
        APOT_ATOMIC_RADIAL_POINTS,
        unique_count,
    )?;
    validate_matrix_exact(
        "edenvl",
        input.overlapped_valence_density.dim(),
        APOT_ATOMIC_RADIAL_POINTS,
        unique_count,
    )?;
    validate_matrix_exact(
        "vclap",
        input.overlapped_coulomb_potential.dim(),
        APOT_ATOMIC_RADIAL_POINTS,
        unique_count,
    )?;
    validate_matrix_exact(
        "iorb",
        input.orbital_indices_by_kappa.dim(),
        10,
        state_count,
    )?;

    validate_finite_vector("rnrm", input.norman_radii)?;
    validate_finite_vector("dgc0", input.core_hole_large_component)?;
    validate_finite_vector("dpc0", input.core_hole_small_component)?;
    validate_finite_vector("drho", input.core_hole_density)?;
    validate_finite_vector("dvcoul", input.core_hole_coulomb_potential)?;
    validate_finite_matrix("dmag", input.magnetization_density)?;
    validate_finite_matrix("xnvmu", input.norman_valence_counts)?;
    validate_finite_matrix("rat", input.atom_positions)?;
    validate_finite_matrix("rovr", input.overlap_radii)?;
    validate_finite_matrix("edens", input.overlapped_density)?;
    validate_finite_matrix("edenvl", input.overlapped_valence_density)?;
    validate_finite_matrix("vclap", input.overlapped_coulomb_potential)
}

fn validate_complete_states(
    states: &[&ApotAtomicScfStateSectionsInput<'_>],
    state_count: usize,
) -> Result<()> {
    if states.len() != state_count {
        return invalid_apot_bin(
            0,
            format!(
                "full atomic apot stream requires {state_count} SCF state(s), got {}",
                states.len()
            ),
        );
    }
    for (expected, state) in states.iter().enumerate() {
        if state.state_index != expected {
            return invalid_apot_bin(
                0,
                format!(
                    "full atomic apot stream missing state {expected}; found state {} instead",
                    state.state_index
                ),
            );
        }
    }
    Ok(())
}

fn remove_section(
    sections: &mut Vec<ApotBinSection>,
    section_number: usize,
) -> Result<ApotBinSection> {
    let index = sections
        .iter()
        .position(|section| section.section_number == section_number)
        .ok_or_else(|| {
            invalid_apot_bin_value(
                0,
                format!("internal atomic apot section {section_number} was not assembled"),
            )
        })?;
    Ok(sections.remove(index))
}

fn validate_state_inputs<'slice, 'view>(
    states: &'slice [ApotAtomicScfStateSectionsInput<'view>],
) -> Result<Vec<&'slice ApotAtomicScfStateSectionsInput<'view>>> {
    if states.is_empty() {
        return invalid_apot_bin(0, "at least one atomic SCF state is required");
    }

    let state_count = states[0].state_count;
    let mut seen = vec![false; state_count];
    let mut sorted = Vec::with_capacity(states.len());
    for state in states {
        validate_input(state)?;
        if state.state_count != state_count {
            return invalid_apot_bin(
                0,
                format!(
                    "atomic SCF state {} has state_count {}, expected {state_count}",
                    state.state_index, state.state_count
                ),
            );
        }
        if seen[state.state_index] {
            return invalid_apot_bin(
                0,
                format!("duplicate atomic SCF state_index {}", state.state_index),
            );
        }
        seen[state.state_index] = true;
        sorted.push(state);
    }
    sorted.sort_by_key(|state| state.state_index);
    Ok(sorted)
}

fn orbital_section_number(
    state_count: usize,
    state_index: usize,
    group_index: usize,
) -> Result<usize> {
    group_index
        .checked_mul(state_count)
        .and_then(|offset| offset.checked_add(state_index))
        .and_then(|offset| APOT_ATOMIC_ORBITAL_SECTION_START.checked_add(offset))
        .ok_or_else(|| invalid_apot_bin_value(0, "atomic SCF orbital section number overflowed"))
}

fn validate_input(input: &ApotAtomicScfStateSectionsInput<'_>) -> Result<()> {
    if input.state_count == 0 {
        return invalid_apot_bin(0, "atomic SCF state_count must be positive");
    }
    if input.state_index >= input.state_count {
        return invalid_apot_bin(
            0,
            format!(
                "atomic SCF state_index {} exceeds state_count {}",
                input.state_index, input.state_count
            ),
        );
    }
    if input.orbital_count == 0 || input.orbital_count > APOT_ATOMIC_ORBITAL_SLOTS {
        return invalid_apot_bin(
            0,
            format!(
                "atomic SCF orbital_count {} must be in 1..={APOT_ATOMIC_ORBITAL_SLOTS}",
                input.orbital_count
            ),
        );
    }

    validate_vector_len(
        "density_4pi",
        input.density_4pi.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    validate_vector_len(
        "coulomb_potential",
        input.coulomb_potential.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    validate_vector_len(
        "valence_density_4pi",
        input.valence_density_4pi.len(),
        APOT_ATOMIC_RADIAL_POINTS,
    )?;
    validate_prefix_len(
        "valence_occupations",
        input.valence_occupations.len(),
        input.orbital_count,
    )?;
    validate_prefix_len(
        "orbital_energies",
        input.orbital_energies.len(),
        input.orbital_count,
    )?;
    validate_prefix_len("kappas", input.kappas.len(), input.orbital_count)?;
    validate_matrix_shape(
        "large_components",
        input.large_components.dim(),
        APOT_ATOMIC_RADIAL_POINTS,
        input.orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components.dim(),
        APOT_ATOMIC_RADIAL_POINTS,
        input.orbital_count,
    )?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients.dim(),
        APOT_ATOMIC_COEFFICIENTS,
        input.orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients.dim(),
        APOT_ATOMIC_COEFFICIENTS,
        input.orbital_count,
    )?;

    validate_finite_vector("density_4pi", input.density_4pi)?;
    validate_finite_vector("coulomb_potential", input.coulomb_potential)?;
    validate_finite_vector("valence_density_4pi", input.valence_density_4pi)?;
    validate_finite_prefix(
        "valence_occupations",
        input.valence_occupations,
        input.orbital_count,
    )?;
    validate_finite_prefix(
        "orbital_energies",
        input.orbital_energies,
        input.orbital_count,
    )?;
    validate_finite_matrix_prefix(
        "large_components",
        input.large_components,
        APOT_ATOMIC_RADIAL_POINTS,
        input.orbital_count,
    )?;
    validate_finite_matrix_prefix(
        "small_components",
        input.small_components,
        APOT_ATOMIC_RADIAL_POINTS,
        input.orbital_count,
    )?;
    validate_finite_matrix_prefix(
        "large_coefficients",
        input.large_coefficients,
        APOT_ATOMIC_COEFFICIENTS,
        input.orbital_count,
    )?;
    validate_finite_matrix_prefix(
        "small_coefficients",
        input.small_coefficients,
        APOT_ATOMIC_COEFFICIENTS,
        input.orbital_count,
    )
}

fn validate_vector_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_apot_bin(
            0,
            format!("atomic SCF {field} has length {actual}, expected {expected}"),
        )
    }
}

fn validate_prefix_len(field: &'static str, actual: usize, minimum: usize) -> Result<()> {
    if actual >= minimum {
        Ok(())
    } else {
        invalid_apot_bin(
            0,
            format!("atomic SCF {field} has length {actual}, expected at least {minimum}"),
        )
    }
}

fn validate_matrix_shape(
    field: &'static str,
    actual: (usize, usize),
    rows: usize,
    columns: usize,
) -> Result<()> {
    if actual.0 == rows && actual.1 >= columns {
        Ok(())
    } else {
        invalid_apot_bin(
            0,
            format!(
                "atomic SCF {field} has shape ({}, {}), expected ({rows}, at least {columns})",
                actual.0, actual.1
            ),
        )
    }
}

fn validate_matrix_exact(
    field: &'static str,
    actual: (usize, usize),
    rows: usize,
    columns: usize,
) -> Result<()> {
    if actual == (rows, columns) {
        Ok(())
    } else {
        invalid_apot_bin(
            0,
            format!(
                "atomic apot {field} has shape ({}, {}), expected ({rows}, {columns})",
                actual.0, actual.1
            ),
        )
    }
}

fn validate_matrix_nonempty_columns(
    field: &'static str,
    actual: (usize, usize),
    columns: usize,
) -> Result<()> {
    if actual.0 > 0 && actual.1 == columns {
        Ok(())
    } else {
        invalid_apot_bin(
            0,
            format!(
                "atomic apot {field} has shape ({}, {}), expected (positive, {columns})",
                actual.0, actual.1
            ),
        )
    }
}

fn validate_finite_scalar(field: &'static str, value: f64) -> Result<()> {
    validate_finite(0, field, value)
}

fn validate_finite_vector(field: &'static str, values: ArrayView1<'_, f64>) -> Result<()> {
    for value in values {
        validate_finite(0, field, *value)?;
    }
    Ok(())
}

fn validate_finite_matrix(field: &'static str, values: ArrayView2<'_, f64>) -> Result<()> {
    for value in values {
        validate_finite(0, field, *value)?;
    }
    Ok(())
}

fn validate_finite_prefix(
    field: &'static str,
    values: ArrayView1<'_, f64>,
    count: usize,
) -> Result<()> {
    for index in 0..count {
        validate_finite(0, field, values[index])?;
    }
    Ok(())
}

fn validate_finite_matrix_prefix(
    field: &'static str,
    values: ArrayView2<'_, f64>,
    rows: usize,
    columns: usize,
) -> Result<()> {
    for row in 0..rows {
        for column in 0..columns {
            validate_finite(0, field, values[[row, column]])?;
        }
    }
    Ok(())
}

fn norb_section(
    state_count: usize,
    states: &[&ApotAtomicScfStateSectionsInput<'_>],
) -> Result<ApotBinSection> {
    let mut orbital_counts = vec![0_i64; state_count];
    for state in states {
        orbital_counts[state.state_index] = i64::try_from(state.orbital_count)
            .map_err(|_| invalid_apot_bin_value(0, "orbital_count overflows i64"))?;
    }
    Ok(records_section(
        APOT_ATOMIC_NORB_SECTION_NUMBER,
        "norb(0:nphx+1) - number of orbitals for each unique potential",
        vec!["norb"],
        vec![ApotBinType::Int],
        (0..state_count)
            .map(|state| vec![ApotBinValue::Int(orbital_counts[state])])
            .collect(),
    ))
}

fn atomic_scalar_section(input: &ApotAtomicPotsSectionsInput<'_, '_>) -> Result<ApotBinSection> {
    Ok(records_section(
        1,
        "This file contains information about the free atom potentials.",
        vec!["nph", "nat", "ihole", "erelax", "emu", "s02"],
        vec![
            ApotBinType::Int,
            ApotBinType::Int,
            ApotBinType::Int,
            ApotBinType::Double,
            ApotBinType::Double,
            ApotBinType::Double,
        ],
        vec![vec![
            ApotBinValue::Int(usize_to_i64(input.unique_potential_count, "nph")?),
            ApotBinValue::Int(usize_to_i64(input.atom_count, "nat")?),
            ApotBinValue::Int(input.hole_index),
            ApotBinValue::Real(input.relaxation_energy),
            ApotBinValue::Real(input.edge_energy),
            ApotBinValue::Real(input.amplitude_reduction),
        ]],
    ))
}

fn unique_potential_section(
    input: &ApotAtomicPotsSectionsInput<'_, '_>,
    unique_count: usize,
) -> Result<ApotBinSection> {
    Ok(records_section(
        2,
        "iz/iatph/novr/rnrm for each unique potential",
        vec!["iz", "iatph", "novr", "rnrm"],
        vec![
            ApotBinType::Int,
            ApotBinType::Int,
            ApotBinType::Int,
            ApotBinType::Double,
        ],
        (0..unique_count)
            .map(|row| {
                vec![
                    ApotBinValue::Int(input.atomic_numbers[row]),
                    ApotBinValue::Int(input.model_atom_indices[row]),
                    ApotBinValue::Int(input.overlap_shell_counts[row]),
                    ApotBinValue::Real(input.norman_radii[row]),
                ]
            })
            .collect(),
    ))
}

fn atom_potential_section(atom_potential_indices: ArrayView1<'_, i64>) -> Result<ApotBinSection> {
    Ok(records_section(
        4,
        "iphat(natx) - given specific atom, which unique pot?",
        vec!["iphat"],
        vec![ApotBinType::Int],
        atom_potential_indices
            .iter()
            .map(|&value| vec![ApotBinValue::Int(value)])
            .collect(),
    ))
}

fn core_hole_section(input: &ApotAtomicPotsSectionsInput<'_, '_>) -> ApotBinSection {
    records_section(
        5,
        "dgc0/dpc0/drho/dvcoul core-hole columns",
        vec!["dgc0", "dpc0", "drho", "dvcoul"],
        vec![ApotBinType::Double; 4],
        (0..APOT_ATOMIC_RADIAL_POINTS)
            .map(|row| {
                vec![
                    ApotBinValue::Real(input.core_hole_large_component[row]),
                    ApotBinValue::Real(input.core_hole_small_component[row]),
                    ApotBinValue::Real(input.core_hole_density[row]),
                    ApotBinValue::Real(input.core_hole_coulomb_potential[row]),
                ]
            })
            .collect(),
    )
}

fn kappa_section(
    state_count: usize,
    states: &[&ApotAtomicScfStateSectionsInput<'_>],
) -> ApotBinSection {
    let mut values = Array2::<i64>::zeros((APOT_ATOMIC_ORBITAL_SLOTS, state_count));
    for state in states {
        for orbital in 0..state.orbital_count {
            values[[orbital, state.state_index]] = i64::from(state.kappas[orbital]);
        }
    }
    int_matrix_section(
        APOT_ATOMIC_KAPPA_SECTION_NUMBER,
        "kappa(i,0:nphx+1) - Dirac kappa for each orbital",
        values,
    )
}

fn real_state_matrix_section_for_states<'slice, 'view>(
    section_number: usize,
    header: &'static str,
    state_count: usize,
    states: &[&'slice ApotAtomicScfStateSectionsInput<'view>],
    column: impl Fn(&ApotAtomicScfStateSectionsInput<'view>) -> ArrayView1<'view, f64>,
) -> ApotBinSection {
    let mut values = Array2::<f64>::zeros((APOT_ATOMIC_RADIAL_POINTS, state_count));
    for state in states {
        let column = column(state);
        for row in 0..APOT_ATOMIC_RADIAL_POINTS {
            values[[row, state.state_index]] = column[row];
        }
    }
    real_matrix_section(section_number, header, values)
}

fn real_orbital_state_matrix_section_for_states<'slice, 'view>(
    section_number: usize,
    header: &'static str,
    state_count: usize,
    states: &[&'slice ApotAtomicScfStateSectionsInput<'view>],
    column: impl Fn(&ApotAtomicScfStateSectionsInput<'view>) -> ArrayView1<'view, f64>,
) -> ApotBinSection {
    let mut values = Array2::<f64>::zeros((APOT_ATOMIC_ORBITAL_SLOTS, state_count));
    for state in states {
        let column = column(state);
        for orbital in 0..state.orbital_count {
            values[[orbital, state.state_index]] = column[orbital];
        }
    }
    real_matrix_section(section_number, header, values)
}

fn real_orbital_component_section(
    section_number: usize,
    header: &'static str,
    rows: usize,
    orbital_count: usize,
    source: ArrayView2<'_, f64>,
) -> ApotBinSection {
    let mut values = Array2::<f64>::zeros((rows, APOT_ATOMIC_ORBITAL_SLOTS));
    for row in 0..rows {
        for orbital in 0..orbital_count {
            values[[row, orbital]] = source[[row, orbital]];
        }
    }
    real_matrix_section(section_number, header, values)
}

fn records_section(
    section_number: usize,
    header: &'static str,
    labels: Vec<&'static str>,
    column_types: Vec<ApotBinType>,
    rows: Vec<Vec<ApotBinValue>>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: vec![header.to_string()],
        header_texts: vec![format!(" {header}")],
        column_labels: labels.iter().map(|label| (*label).to_string()).collect(),
        column_label_text: Some(format!(
            " {} ",
            labels
                .iter()
                .map(|label| format!("{label:>10}"))
                .collect::<Vec<_>>()
                .join(" ")
        )),
        payload: ApotBinPayload::Records(ApotBinRecords { column_types, rows }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    }
}

fn real_matrix_section(
    section_number: usize,
    header: &'static str,
    values: Array2<f64>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: vec![header.to_string()],
        header_texts: vec![format!(" {header}")],
        column_labels: vec![],
        column_label_text: None,
        payload: ApotBinPayload::Matrix(ApotBinMatrix {
            value_type: ApotBinType::Double,
            values: ApotBinMatrixValues::Real(values),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    }
}

fn int_matrix_section(
    section_number: usize,
    header: &'static str,
    values: Array2<i64>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: vec![header.to_string()],
        header_texts: vec![format!(" {header}")],
        column_labels: vec![],
        column_label_text: None,
        payload: ApotBinPayload::Matrix(ApotBinMatrix {
            value_type: ApotBinType::Int,
            values: ApotBinMatrixValues::Int(values),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    }
}

fn checked_add(left: usize, right: usize, field: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_apot_bin_value(0, format!("atomic apot {field} overflowed")))
}

fn checked_add_one(value: usize, field: &'static str) -> Result<usize> {
    checked_add(value, 1, field)
}

fn usize_to_i64(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_apot_bin_value(0, format!("atomic apot {field} overflows i64")))
}
