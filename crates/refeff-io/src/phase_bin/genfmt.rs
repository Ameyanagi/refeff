//! GENFMT-facing views derived from FEFF `phase.bin`.
//!
//! `phase.bin` stores each potential with its own `lmax`, while GENFMT works
//! with one shared signed-`l` axis. These helpers normalize the parsed
//! `phase.bin` payload into the tensor layout expected by `refeff-core` GENFMT
//! driver setup functions.

use ndarray::{Array1, Array2, Array3, Array4, Axis, Slice};
use num_complex::Complex64;
use refeff_core::{
    Complex, GenfmtDriverSetup, GenfmtDriverSetupInput, GenfmtJasDriverSetup,
    GenfmtJasDriverSetupInput, GenfmtJasPathSetupInput, GenfmtJasTransitionSetup,
    GenfmtJasTransitionSetupInput, GenfmtLegendreNormalizationInput, GenfmtNStarDriverInput,
    GenfmtNStarRows, GenfmtNStarRowsInput, GenfmtOrdinaryPathSetupInput,
    GenfmtOrdinaryTransitionMatrices, GenfmtOrdinaryTransitionMatricesInput, GenfmtPathSetup,
    GenfmtSpinChannelCountInput, JasOneSidedTransitionInput, JasQAngleInput, JasQAngles,
    JasSpinTransitionInput, TransitionBMatrix, TransitionBMatrixInput, TransitionRotationInput,
    XsphNrixsTransitionIndices, XsphNrixsTransitionIndicesInput, core_hole_quantum_numbers,
    genfmt_driver_setup, genfmt_jas_driver_setup, genfmt_jas_path_setup,
    genfmt_jas_transition_setup, genfmt_legendre_normalization_table, genfmt_nstar_rows,
    genfmt_ordinary_path_setup, genfmt_ordinary_transition_matrices, genfmt_spin_channel_count,
    jas_q_angles, transition_b_matrix, xsph_nrixs_transition_indices,
};

use crate::error::{IoError, Result};
use crate::{GenfmtInput, GlobalInput};

use super::common::{checked_l_count, invalid_phase_bin};
use super::rixs::rixs_angular_limits_from_phase_bin;
use super::types::{PHASE_BIN_DEFAULT_TRANSITION_COUNT, PhaseBinData};
use super::validate::validate_phase_bin;
use crate::paths_dat::{PathsDatGenfmtPath, genfmt_nstar_path_inputs};

const GENFMT_LAMBDA_CAPACITY: usize = 15;
const GENFMT_MAX_L: usize = 24;
const GENFMT_MAX_M: usize = 4;
const GENFMT_MAX_N: usize = 2;

/// `phase.bin` data normalized for GENFMT driver setup.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinGenfmtData {
    /// Offset used to map signed angular momentum onto `spin_phase_shifts`.
    pub signed_angular_offset: usize,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: Array2<usize>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin,
    /// potential)`. Slots outside a potential's own `lmax` are zero-filled and
    /// masked by `angular_limits`.
    pub spin_phase_shifts: Array4<Complex64>,
    /// FEFF `potlbl(0:npot)` labels.
    pub potential_labels: Vec<String>,
    /// FEFF `iz(0:npot)` atomic numbers.
    pub atomic_numbers: Array1<usize>,
}

impl PhaseBinGenfmtData {
    /// Borrow potential labels in the form required by core GENFMT setup.
    #[must_use]
    pub fn potential_label_refs(&self) -> Vec<&str> {
        self.potential_labels.iter().map(String::as_str).collect()
    }
}

impl PhaseBinData {
    /// Normalize this `phase.bin` payload for core GENFMT driver setup.
    pub fn to_genfmt_data(&self) -> Result<PhaseBinGenfmtData> {
        phase_bin_genfmt_data(self)
    }
}

/// Build GENFMT driver setup tables from parsed `phase.bin` data.
pub fn phase_bin_genfmt_data(data: &PhaseBinData) -> Result<PhaseBinGenfmtData> {
    validate_phase_bin(data)?;

    let signed_angular_offset = data
        .potentials
        .iter()
        .map(|potential| potential.lmax)
        .max()
        .unwrap_or(0);
    let signed_l_count = checked_l_count(signed_angular_offset)?;
    let angular_limits = rixs_angular_limits_from_phase_bin(data)?;
    let mut spin_phase_shifts = Array4::zeros((
        data.energy_count,
        signed_l_count,
        data.spin_count,
        data.potential_count(),
    ));

    for (potential_index, potential) in data.potentials.iter().enumerate() {
        let potential_l_count = checked_l_count(potential.lmax)?;
        let destination_start = signed_angular_offset - potential.lmax;
        for energy in 0..data.energy_count {
            for l_slot in 0..potential_l_count {
                let destination_l_slot = destination_start + l_slot;
                for spin in 0..data.spin_count {
                    spin_phase_shifts[(energy, destination_l_slot, spin, potential_index)] =
                        potential.phase_shifts[(energy, l_slot, spin)];
                }
            }
        }
    }

    Ok(PhaseBinGenfmtData {
        signed_angular_offset,
        angular_limits,
        spin_phase_shifts,
        potential_labels: data
            .potentials
            .iter()
            .map(|potential| potential.label.clone())
            .collect(),
        atomic_numbers: Array1::from_iter(
            data.potentials
                .iter()
                .map(|potential| potential.atomic_number),
        ),
    })
}

/// Build ordinary GENFMT driver setup from parsed FEFF handoff files.
pub fn genfmt_driver_setup_from_handoffs(
    version: &str,
    genfmt: &GenfmtInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<GenfmtDriverSetup> {
    let genfmt_data = phase_bin_genfmt_data(phase)?;
    let potential_labels = genfmt_data.potential_label_refs();
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .map_err(|source| invalid_phase_bin("ihole", source.to_string()))?;
    let initial_orbital_l = usize::try_from(core_hole.angular_momentum)
        .map_err(|_| invalid_phase_bin("ihole", "core-hole angular momentum is negative"))?;

    genfmt_driver_setup(GenfmtDriverSetupInput {
        version,
        pad_width: phase.pad_width,
        core_hole: phase.ihole,
        order: genfmt.control.iorder,
        average_norman_radius: phase.scalars.average_norman_radius,
        fermi_level: phase.scalars.fermi_level,
        edge_energy: phase.scalars.edge_energy,
        spin_selector: global.control.ispin,
        available_spin_channels: phase.spin_count,
        energies: phase.energy_grid.view(),
        spin_reference_energies: phase.reference_energy.view(),
        spin_phase_shifts: genfmt_data.spin_phase_shifts.view(),
        angular_limits: genfmt_data.angular_limits.view(),
        signed_angular_offset: genfmt_data.signed_angular_offset,
        initial_orbital_l,
        initial_kappa: core_hole.kappa,
        potential_labels: &potential_labels,
        atomic_numbers: genfmt_data.atomic_numbers.view(),
    })
    .map_err(genfmt_setup_error)
}

/// Build GENFMTJAS driver setup from parsed FEFF handoff files.
pub fn genfmt_jas_driver_setup_from_handoffs(
    version: &str,
    genfmt: &GenfmtInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<GenfmtJasDriverSetup> {
    let genfmt_data = phase_bin_genfmt_data(phase)?;
    let potential_labels = genfmt_data.potential_label_refs();
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .map_err(|source| invalid_phase_bin("ihole", source.to_string()))?;
    let initial_orbital_l = usize::try_from(core_hole.angular_momentum)
        .map_err(|_| invalid_phase_bin("ihole", "core-hole angular momentum is negative"))?;

    genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version,
        pad_width: phase.pad_width,
        core_hole: phase.ihole,
        order: genfmt.control.iorder,
        average_norman_radius: phase.scalars.average_norman_radius,
        fermi_level: phase.scalars.fermi_level,
        edge_energy: phase.scalars.edge_energy,
        spin_selector: global.control.ispin,
        available_spin_channels: phase.spin_count,
        energies: phase.energy_grid.view(),
        spin_reference_energies: phase.reference_energy.view(),
        spin_phase_shifts: genfmt_data.spin_phase_shifts.view(),
        spin_radial_factors: phase.transition_moments.view(),
        angular_limits: genfmt_data.angular_limits.view(),
        signed_angular_offset: genfmt_data.signed_angular_offset,
        initial_orbital_l,
        initial_kappa: core_hole.kappa,
        potential_labels: &potential_labels,
        atomic_numbers: genfmt_data.atomic_numbers.view(),
    })
    .map_err(genfmt_setup_error)
}

/// Build ordinary GENFMT path-local setup from parsed FEFF handoff files.
pub fn genfmt_ordinary_path_setups_from_handoffs(
    genfmt: &GenfmtInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    paths: &[PathsDatGenfmtPath],
) -> Result<Vec<GenfmtPathSetup>> {
    let initial_l = genfmt_initial_l_from_phase(phase)?;
    let lmaxp1 = genfmt_lmaxp1_from_phase(phase)?;
    let polarized = global.control.ipol > 0;
    paths
        .iter()
        .map(|path| {
            genfmt_ordinary_path_setup(GenfmtOrdinaryPathSetupInput {
                positions: path.positions_bohr.view(),
                polarized,
                calculation: genfmt.control.iorder,
                initial_l,
                lambda_capacity: GENFMT_LAMBDA_CAPACITY,
                max_m: GENFMT_MAX_M,
                max_n: GENFMT_MAX_N,
                lmaxp1,
            })
            .map_err(genfmt_setup_error)
        })
        .collect()
}

/// Build GENFMTJAS path-local setup from parsed FEFF handoff files.
pub fn genfmt_jas_path_setups_from_handoffs(
    genfmt: &GenfmtInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    paths: &[PathsDatGenfmtPath],
) -> Result<Vec<GenfmtPathSetup>> {
    let initial_l = genfmt_initial_l_from_phase(phase)?;
    let lmaxp1 = genfmt_lmaxp1_from_phase(phase)?;
    let polarized = global.control.ipol > 0;
    paths
        .iter()
        .map(|path| {
            genfmt_jas_path_setup(GenfmtJasPathSetupInput {
                positions: path.positions_bohr.view(),
                polarized,
                calculation: genfmt.control.iorder,
                initial_l,
                lambda_capacity: GENFMT_LAMBDA_CAPACITY,
                max_m: GENFMT_MAX_M,
                max_n: GENFMT_MAX_N,
                lmaxp1,
            })
            .map_err(genfmt_setup_error)
        })
        .collect()
}

/// Rebuild FEFF `nrixs_inp` transition indices for GENFMTJAS handoffs.
///
/// In Fortran this state is generated by `COMMON/m_nrixs.f90:nrixs_init` and
/// kept in module globals for both XSPH and GENFMTJAS. The Rust CLI stages are
/// independent, so GENFMTJAS reconstructs the deterministic transition index
/// table from `global.inp` and `phase.bin` before entering the path loop.
pub fn genfmt_jas_transition_indices_from_handoffs(
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<XsphNrixsTransitionIndices> {
    validate_phase_bin(phase)?;
    if global.control.do_nrixs != 1 {
        return Err(invalid_phase_bin(
            "do_nrixs",
            "GENFMTJAS transition indices require NRIXS handoff controls",
        ));
    }
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .map_err(|source| invalid_phase_bin("ihole", source.to_string()))?;
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: core_hole.kappa,
        multipole: global.control.le2,
        max_angular_momentum: GENFMT_MAX_L,
    })
    .map_err(|source| invalid_phase_bin("nrixs_indices", source.to_string()))?;
    if phase.transition_count != indices.transitions.len() {
        return Err(invalid_phase_bin(
            "indmax",
            format!(
                "phase.bin indmax {} does not match generated NRIXS indmax {}",
                phase.transition_count,
                indices.transitions.len()
            ),
        ));
    }
    Ok(indices)
}

/// Rebuild FEFF GENFMTJAS q-vector weights and rotation angles from handoffs.
pub fn genfmt_jas_q_angles_from_handoffs(
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<JasQAngles> {
    let q_weights = Array1::from_iter(
        global
            .q_vectors
            .iter()
            .map(|vector| Complex::new(vector.weight[0], vector.weight[1])),
    );
    let q_trig = Array2::from_shape_fn((global.q_vectors.len(), 4), |(row, column)| {
        global.q_vectors[row].trig[column]
    });
    let q_angles = jas_q_angles(JasQAngleInput {
        qaverage: global.q_control.qaverage,
        q_trig: q_trig.view(),
        q_weights: q_weights.view(),
    })
    .map_err(genfmt_setup_error)?;
    if q_angles.weights.len() != phase.q_count {
        return Err(invalid_phase_bin(
            "nq",
            format!(
                "phase.bin nq {} does not match GENFMTJAS q angle count {}",
                phase.q_count,
                q_angles.weights.len()
            ),
        ));
    }
    Ok(q_angles)
}

/// Build FEFF GENFMTJAS `mmtrjas`/`mmtrjas0` transition setup per path.
pub fn genfmt_jas_transition_setups_from_handoff_setups(
    global: &GlobalInput,
    phase: &PhaseBinData,
    path_setups: &[GenfmtPathSetup],
) -> Result<Vec<GenfmtJasTransitionSetup>> {
    let indices = genfmt_jas_transition_indices_from_handoffs(global, phase)?;
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .map_err(|source| invalid_phase_bin("ihole", source.to_string()))?;
    let spin_channels = phase.spin_count;
    let transition_angular_momenta = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.orbital_angular_momentum),
    );
    let final_lg_momenta = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.decomposition_channel),
    );
    let final_lj_momenta = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.total_angular_momentum_channel),
    );
    let q_angles = genfmt_jas_q_angles_from_handoffs(global, phase)?;
    let max_angular_momentum = nonnegative_usize(global.control.l2lp, "l2lp")?;

    path_setups
        .iter()
        .map(|setup| {
            let rotations = setup
                .rotations
                .transition_rotations(setup.angles.eta_values.view(), global.control.ipol > 0)
                .map_err(genfmt_setup_error)?;
            let TransitionRotationInput::Polarized {
                first_rotation,
                last_rotation,
                first_eta,
                last_eta,
            } = rotations
            else {
                return Err(invalid_phase_bin(
                    "ipol",
                    "GENFMTJAS transition setup requires polarized path rotations",
                ));
            };

            genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
                ellipticity: global.control.elpty,
                phase_transition_count: phase.transition_count,
                requested_transition_count: indices.transitions.len(),
                left_right: JasOneSidedTransitionInput {
                    initial_kappa: core_hole.kappa,
                    initial_j2: indices.initial_j2,
                    spin_channels,
                    transition_angular_momenta: transition_angular_momenta.view(),
                    final_lg_momenta: final_lg_momenta.view(),
                    final_lj_momenta: final_lj_momenta.view(),
                    final_lj_max: indices.final_lj_max,
                    final_j2_max: indices.final_j2_max,
                    max_angular_momentum,
                    q_phases: q_angles.phases.view(),
                    q_beta_angles: q_angles.beta_angles.view(),
                    first_rotation,
                    last_rotation,
                    rotation_magnetic_offset: setup.rotations.rotation_magnetic_offset,
                    first_eta,
                    last_eta,
                },
                spherical: JasSpinTransitionInput {
                    initial_kappa: core_hole.kappa,
                    initial_j2: indices.initial_j2,
                    spin_channels,
                    transition_angular_momenta: transition_angular_momenta.view(),
                    final_lj_max: indices.final_lj_max,
                    final_j2_max: indices.final_j2_max,
                    max_angular_momentum,
                    first_rotation,
                    last_rotation,
                    rotation_magnetic_offset: setup.rotations.rotation_magnetic_offset,
                    first_eta,
                    last_eta,
                },
            })
            .map_err(genfmt_setup_error)
        })
        .collect()
}

/// Build ordinary GENFMT `bcoef` transition data from parsed FEFF handoff files.
///
/// FEFF `GENFMT/mmtr.f90` calls `bcoef` before applying path rotations. The
/// Rust handoff layer uses the maximum parsed `phase.bin` angular basis as the
/// current `lx` source, matching the angular range carried forward from XSPH.
pub fn genfmt_ordinary_transition_b_matrix_from_handoffs(
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<TransitionBMatrix> {
    validate_phase_bin(phase)?;
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .map_err(|source| invalid_phase_bin("ihole", source.to_string()))?;
    transition_b_matrix(TransitionBMatrixInput {
        lmax: genfmt_lmax_from_phase(phase)?,
        initial_kappa: core_hole.kappa,
        polarization: global.control.ipol,
        polarization_tensor: genfmt_polarization_tensor(global),
        multipole: global.control.le2,
        trace_orbital: false,
        spin: global.control.ispin,
        spin_channels: phase.spin_count,
        spin_vector_angle: global.control.angks,
    })
    .map_err(genfmt_transition_error)
}

/// Apply ordinary GENFMT `mmtr` transition setup to every prepared path.
pub fn genfmt_ordinary_transition_matrices_from_handoff_setups(
    global: &GlobalInput,
    phase: &PhaseBinData,
    path_setups: &[GenfmtPathSetup],
    transition_b_matrix: &TransitionBMatrix,
) -> Result<Vec<GenfmtOrdinaryTransitionMatrices>> {
    validate_phase_bin(phase)?;
    let initial_l = genfmt_initial_l_from_phase(phase)?;
    let active_spin_channel_count = genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
        spin_selector: global.control.ispin,
        available_spin_channels: phase.spin_count,
    })
    .map_err(genfmt_setup_error)?;
    let transition_angular_momenta = Array1::from_vec(transition_b_matrix.orbital_momenta.to_vec());
    let polarized = global.control.ipol > 0;

    path_setups
        .iter()
        .map(|setup| {
            let rotations = setup
                .rotations
                .transition_rotations(setup.angles.eta_values.view(), polarized)
                .map_err(genfmt_setup_error)?;
            genfmt_ordinary_transition_matrices(GenfmtOrdinaryTransitionMatricesInput {
                spin_selector: global.control.ispin,
                active_spin_channel_count,
                available_spin_channels: phase.spin_count,
                transition_angular_momenta: transition_angular_momenta.view(),
                transition_b_matrix: transition_b_matrix.matrix.view(),
                transition_magnetic_offset: transition_b_matrix.l_offset,
                initial_l,
                magnetic_limit: GENFMT_MAX_M,
                rotation_magnetic_offset: setup.rotations.rotation_magnetic_offset,
                rotations,
            })
            .map_err(genfmt_setup_error)
        })
        .collect()
}

/// Extract ordinary GENFMT `rkk2(1:ne,1:8,1:nspx)` radial factors.
///
/// FEFF `COMMON/rdxsph.f90` reads exactly eight transition channels for the
/// ordinary GENFMT branch. Modern `phase.bin` files may include the explicit
/// FEFFQ header fields, but ordinary GENFMT still consumes the first q slice and
/// the historical eight transition slots.
pub fn genfmt_ordinary_spin_radial_factors_from_phase(
    phase: &PhaseBinData,
) -> Result<Array3<Complex64>> {
    validate_phase_bin(phase)?;
    if phase.transition_count < PHASE_BIN_DEFAULT_TRANSITION_COUNT {
        return Err(invalid_phase_bin(
            "indmax",
            "ordinary GENFMT requires at least eight transition channels",
        ));
    }

    Ok(phase
        .transition_moments
        .index_axis(Axis(1), 0)
        .slice_axis(Axis(1), Slice::from(..PHASE_BIN_DEFAULT_TRANSITION_COUNT))
        .to_owned())
}

/// Build optional FEFF `nstar` driver controls from parsed GENFMT handoff files.
pub fn genfmt_nstar_driver_input_from_handoffs(
    genfmt: &GenfmtInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<Option<GenfmtNStarDriverInput>> {
    if !genfmt.control.wnstar {
        return Ok(None);
    }

    let ellipticity = if global.control.do_nrixs == 1 {
        0.0
    } else {
        global.control.elpty
    };
    Ok(Some(GenfmtNStarDriverInput {
        primary_polarization: global.evec,
        ellipticity_vector: global.xivec,
        initial_l: genfmt_initial_l_from_phase(phase)?,
        ellipticity,
    }))
}

/// Convert FEFF one-based `ik0` from `phase.bin` into Rust's zero-based index.
pub fn genfmt_edge_start_index_from_phase(phase: &PhaseBinData) -> Result<usize> {
    validate_phase_bin(phase)?;
    if phase.main_energy_count > phase.energy_count {
        return Err(invalid_phase_bin(
            "ne1",
            "main energy count exceeds total energy count",
        ));
    }
    if phase.fermi_index <= 0 {
        return Err(invalid_phase_bin(
            "ik0",
            "GENFMT edge index must be one-based and positive",
        ));
    }

    let edge_start_index = usize::try_from(phase.fermi_index - 1)
        .map_err(|_| invalid_phase_bin("ik0", "failed to convert edge index"))?;
    if edge_start_index >= phase.main_energy_count {
        return Err(invalid_phase_bin(
            "ik0",
            "edge index must fall within the main energy grid",
        ));
    }
    if phase.main_energy_count - edge_start_index < 2 {
        return Err(invalid_phase_bin(
            "ne1",
            "ordinary GENFMT importance integration requires at least two points from ik0",
        ));
    }
    Ok(edge_start_index)
}

/// Build optional FEFF `nstar.dat` rows from parsed GENFMT handoff files.
///
/// Ordinary GENFMT passes global `elpty` to `xstar`, while GENFMTJAS follows
/// `genfmtjas.f90` and uses the local `elpty0 = 0` value for this bookkeeping
/// output.
pub fn genfmt_nstar_rows_from_handoffs(
    genfmt: &GenfmtInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    paths: &[PathsDatGenfmtPath],
) -> Result<Option<GenfmtNStarRows>> {
    if !genfmt.control.wnstar {
        return Ok(None);
    }

    let Some(driver_input) = genfmt_nstar_driver_input_from_handoffs(genfmt, global, phase)? else {
        return Ok(None);
    };
    let path_inputs = genfmt_nstar_path_inputs(paths);

    genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization: driver_input.primary_polarization,
        ellipticity_vector: driver_input.ellipticity_vector,
        initial_l: driver_input.initial_l,
        ellipticity: driver_input.ellipticity,
        path_inputs: &path_inputs,
    })
    .map(Some)
    .map_err(genfmt_setup_error)
}

/// Build FEFF GENFMT `xnlm` normalization factors using GENFMT source dims.
///
/// Both `genfmtsub.f90` and `genfmtjas.f90` call `snlm(ltot+1, mtot+1)` once
/// before the path loop. The Rust core validates the table against the active
/// angular basis, so source-backed handoffs retain the FEFF `ltot` ceiling on
/// both axes while path setup still clamps lambda selection to `mtot`.
pub fn genfmt_legendre_normalization_from_feff_dims() -> Result<Array2<f64>> {
    genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
        lmaxp1: GENFMT_MAX_L + 1,
        mmaxp1: GENFMT_MAX_L + 1,
    })
    .map_err(genfmt_setup_error)
}

/// Build GENFMT normalization factors in the `(m, l)` layout used by core path loops.
pub fn genfmt_core_legendre_normalization_from_feff_dims() -> Result<Array2<f64>> {
    let snlm = genfmt_legendre_normalization_from_feff_dims()?;
    Ok(snlm.reversed_axes().as_standard_layout().to_owned())
}

fn genfmt_initial_l_from_phase(phase: &PhaseBinData) -> Result<usize> {
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .map_err(|source| invalid_phase_bin("ihole", source.to_string()))?;
    let initial_orbital_l = usize::try_from(core_hole.angular_momentum)
        .map_err(|_| invalid_phase_bin("ihole", "core-hole angular momentum is negative"))?;
    initial_orbital_l
        .checked_add(1)
        .ok_or_else(|| invalid_phase_bin("ihole", "initial angular momentum overflowed"))
}

fn genfmt_lmaxp1_from_phase(phase: &PhaseBinData) -> Result<usize> {
    validate_phase_bin(phase)?;
    let mut lmaxp1 = None;
    for potential in &phase.potentials {
        let potential_lmaxp1 = potential
            .lmax
            .checked_add(1)
            .ok_or_else(|| invalid_phase_bin("lmax", "lmaxp1 overflowed"))?;
        lmaxp1 = Some(lmaxp1.map_or(potential_lmaxp1, |maximum: usize| {
            maximum.max(potential_lmaxp1)
        }));
    }
    lmaxp1.ok_or_else(|| invalid_phase_bin("potentials", "phase.bin contains no potentials"))
}

fn genfmt_lmax_from_phase(phase: &PhaseBinData) -> Result<usize> {
    genfmt_lmaxp1_from_phase(phase)?
        .checked_sub(1)
        .ok_or_else(|| invalid_phase_bin("lmax", "lmax underflowed"))
}

fn nonnegative_usize(value: i32, field: &'static str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("{field} must be nonnegative")))
}

fn genfmt_polarization_tensor(global: &GlobalInput) -> [[Complex; 3]; 3] {
    let mut tensor = [[Complex::new(0.0, 0.0); 3]; 3];
    for (row_index, row) in global.polarization_tensor.iter().enumerate() {
        tensor[row_index] = [
            Complex::new(row[0], row[1]),
            Complex::new(row[2], row[3]),
            Complex::new(row[4], row[5]),
        ];
    }
    tensor
}

fn genfmt_setup_error(source: refeff_core::GenfmtError) -> IoError {
    invalid_phase_bin("genfmt_setup", source.to_string())
}

fn genfmt_transition_error(source: refeff_core::AngularError) -> IoError {
    invalid_phase_bin("genfmt_transition_b_matrix", source.to_string())
}
