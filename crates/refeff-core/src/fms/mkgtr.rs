//! MKGTR Green's-function trace folding for FMS outputs.

use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ShapeBuilder,
};
use num_complex::Complex32;

use crate::{
    Complex, Real,
    angular::{
        TransitionBMatrix, legendre_polynomials_into, mkgtr_clebsch_gordan_coefficients,
        wigner_rotation,
    },
    xsph_nrixs_transition_weights,
};

use super::{FmsError, ensure_axis_len, ensure_spin_channels};

/// Inputs for FEFF `MKGTR/getgtr.f90` Green's-function trace folding.
#[derive(Debug, Clone)]
pub struct MkgtrGreenTraceInput<'a> {
    /// Active spin channels used by `getgtr` after FEFF's `ispin` selection.
    pub active_spin_channels: usize,
    /// `gg(energy, channel1, channel2)` FMS Green's-function matrices for
    /// absorber potential `iph=0`.
    pub green_functions: ArrayView3<'a, Complex32>,
    /// Transition B matrices for the spectra selected by `ipmin:ipstep:ipmax`.
    pub transition_matrices: &'a [TransitionBMatrix],
    /// FEFF transition moments `rkk(energy, transition, spin)`.
    pub transition_moments: ArrayView3<'a, Complex>,
}

/// FEFF MKGTR folded FMS trace spectra.
#[derive(Debug, Clone, PartialEq)]
pub struct MkgtrGreenTraceResult {
    /// `gtr(spectrum, energy)` values ready for `fms.bin` or `gtr.dat`.
    pub traces: Array2<Complex>,
}

/// One active FEFF `nrixs_inp` transition row used by `getgtrjas`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MkgtrJasTransition {
    /// Relativistic final-state kappa, FEFF `kind`.
    pub final_state_kappa: i32,
    /// Angular-decomposition channel, FEFF `lgind`.
    pub decomposition_channel: usize,
    /// Spherical transition multipole, FEFF `ljind`.
    pub multipole: usize,
    /// Final-state orbital angular momentum, FEFF `lind`.
    pub orbital_angular_momentum: usize,
}

/// FEFF `getgtrjas` q/q-prime traversal selected by `mixdff` and `imdff`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MkgtrJasQPairMode {
    /// Ordinary NRIXS density input: use only `(q,q)` elements.
    Diagonal,
    /// MDFF task 1: sum all `(q,q')` elements.
    AllPairs,
    /// MDFF task 2: use only FEFF's `(q=1,q'=2)` element.
    FirstToSecond,
}

/// Inputs for FEFF `MKGTR/getgtrjas.f90` NRIXS/JAS trace folding.
#[derive(Debug, Clone)]
pub struct MkgtrJasGreenTraceInput<'a> {
    /// FEFF `nsp`: one or two active spin channels.
    pub active_spin_channels: usize,
    /// FEFF FMS angular limit `lx` used to index the absorber `gg` block.
    pub max_angular_momentum: usize,
    /// Absorber Green functions as `(energy, channel1, channel2)`.
    pub green_functions: ArrayView3<'a, Complex32>,
    /// NRIXS radial transition moments as `(energy, q, transition, spin)`.
    pub transition_moments: ArrayView4<'a, Complex>,
    /// Core-hole relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Doubled initial-state total angular momentum, FEFF `jinit`.
    pub initial_j2: i32,
    /// Largest doubled final-state total angular momentum, FEFF `jmax`.
    pub final_j2_max: i32,
    /// Largest spherical transition multipole, FEFF `ljmax`.
    pub final_lj_max: usize,
    /// Active deterministic `kind/lgind/ljind/lind` transition rows.
    pub transitions: &'a [MkgtrJasTransition],
    /// Conjugated q azimuthal phases, FEFF `pha`.
    pub q_phases: ArrayView1<'a, Complex>,
    /// Q polar rotation angles, FEFF `beta`.
    pub q_beta_angles: ArrayView1<'a, Real>,
    /// Effective complex q weights, FEFF `qweights`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// FEFF `cosmdff(q,q')` table.
    pub q_pair_cosines: ArrayView2<'a, Real>,
    /// FEFF `elpty`; negative values select spherical averaging.
    pub ellipticity: Real,
    /// Q/q-prime traversal selected by `mixdff` and `imdff`.
    pub q_pair_mode: MkgtrJasQPairMode,
    /// Optional inclusive FEFF `ldecmx` decomposition channel.
    pub max_decomposition_channel: Option<usize>,
}

/// FEFF NRIXS/JAS FMS trace and optional `gtrl(lg2,lg1,ie)` decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct MkgtrJasGreenTraceResult {
    /// Total `gtr(ie)` values.
    pub trace: Array1<Complex>,
    /// Optional decomposed traces as `(energy, lg2, lg1)`.
    pub decomposed_traces: Option<Array3<Complex>>,
}

/// Fold FEFF FMS Green functions through the NRIXS/JAS `getgtrjas` branch.
///
/// This includes q-axis rotation of `gg`, the `bcoefjas` non-spherical
/// contraction, the `calclbcoef` spherical-average contraction, MDFF q/q-prime
/// selection, Legendre factors, and optional `ldecmx` channel decomposition.
pub fn mkgtr_jas_green_trace(
    input: MkgtrJasGreenTraceInput<'_>,
) -> Result<MkgtrJasGreenTraceResult, FmsError> {
    validate_mkgtr_jas_input(&input)?;

    let energy_count = input.green_functions.shape()[0];
    let q_count = input.transition_moments.shape()[1];
    let mut trace = Array1::<Complex>::zeros(energy_count);
    let mut decomposed_traces = input
        .max_decomposition_channel
        .map(|maximum| Array3::<Complex>::zeros((energy_count, maximum + 1, maximum + 1).f()));
    let pairs = mkgtr_jas_q_pairs(input.q_pair_mode, q_count)?;
    let max_multipole = input
        .transitions
        .iter()
        .map(|transition| transition.multipole)
        .max()
        .unwrap_or(0);
    let mut legendre = vec![0.0; max_multipole + 1];

    let transition_weights = if input.ellipticity >= 0.0 {
        Some(mkgtr_jas_transition_weights(&input)?)
    } else {
        None
    };
    let clebsch = if input.ellipticity < 0.0 {
        let j_lmax =
            input
                .max_angular_momentum
                .checked_add(2)
                .ok_or(FmsError::InvalidAngularLimit {
                    name: "lx",
                    value: input.max_angular_momentum,
                    lx: input.max_angular_momentum,
                })?;
        let mj_lmax = input
            .max_angular_momentum
            .checked_mul(2)
            .and_then(|value| value.checked_add(2))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lx",
                value: input.max_angular_momentum,
                lx: input.max_angular_momentum,
            })?;
        Some(mkgtr_clebsch_gordan_coefficients(
            input.max_angular_momentum,
            j_lmax,
            mj_lmax,
        )?)
    } else {
        None
    };

    for energy in 0..energy_count {
        for q in 0..q_count {
            let rotated = mkgtr_jas_rotate_green(
                input.green_functions.index_axis(ndarray::Axis(0), energy),
                input.active_spin_channels,
                input.max_angular_momentum,
                input.q_phases[q],
                input.q_beta_angles[q],
                input.ellipticity >= 0.0,
            )?;
            for &(left_q, right_q) in pairs.iter().filter(|&&(left, _)| left == q) {
                legendre_polynomials_into(input.q_pair_cosines[(left_q, right_q)], &mut legendre);
                if input.ellipticity >= 0.0 {
                    let Some(transition_weights) = transition_weights.as_ref() else {
                        return Err(FmsError::TableIndexOutOfRange {
                            table: "hbmat",
                            axis: "branch",
                            index: 0,
                        });
                    };
                    mkgtr_jas_accumulate_nonspherical(
                        &input,
                        energy,
                        left_q,
                        right_q,
                        rotated.view(),
                        transition_weights,
                        &legendre,
                        &mut trace,
                        decomposed_traces.as_mut(),
                    )?;
                } else {
                    let Some(clebsch) = clebsch.as_ref() else {
                        return Err(FmsError::TableIndexOutOfRange {
                            table: "clbcoef",
                            axis: "branch",
                            index: 0,
                        });
                    };
                    mkgtr_jas_accumulate_spherical(
                        &input,
                        energy,
                        left_q,
                        right_q,
                        rotated.view(),
                        clebsch,
                        &legendre,
                        &mut trace,
                        decomposed_traces.as_mut(),
                    )?;
                }
            }
        }
    }

    for (index, value) in trace.iter().copied().enumerate() {
        validate_finite_complex_value("gtrjas", index, value)?;
    }
    if let Some(decomposed) = decomposed_traces.as_ref() {
        for (index, value) in decomposed.iter().copied().enumerate() {
            validate_finite_complex_value("gtrl", index, value)?;
        }
    }
    Ok(MkgtrJasGreenTraceResult {
        trace,
        decomposed_traces,
    })
}

fn validate_mkgtr_jas_input(input: &MkgtrJasGreenTraceInput<'_>) -> Result<(), FmsError> {
    ensure_spin_channels(input.active_spin_channels)?;
    let green_shape = input.green_functions.shape();
    if green_shape[0] == 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "energy",
            index: 0,
        });
    }
    if green_shape[1] == 0 || green_shape[1] != green_shape[2] {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "shape",
            index: green_shape[1],
        });
    }
    let angular_count =
        input
            .max_angular_momentum
            .checked_add(1)
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lx",
                value: input.max_angular_momentum,
                lx: input.max_angular_momentum,
            })?;
    let required_green_channels = angular_count
        .checked_mul(angular_count)
        .and_then(|value| value.checked_mul(input.active_spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: input.max_angular_momentum,
            lx: input.max_angular_momentum,
        })?;
    ensure_axis_len("gg", "channel", green_shape[1], required_green_channels - 1)?;

    if input.initial_kappa == 0 || input.initial_j2 < 0 || input.final_j2_max < 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "nrixs",
            axis: "angular",
            index: 0,
        });
    }
    if input.transitions.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "nrixs",
            axis: "transition",
            index: 0,
        });
    }

    let rkk_shape = input.transition_moments.shape();
    ensure_axis_len("rkk", "energy", rkk_shape[0], green_shape[0] - 1)?;
    if rkk_shape[1] == 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "rkk",
            axis: "q",
            index: 0,
        });
    }
    ensure_axis_len(
        "rkk",
        "transition",
        rkk_shape[2],
        input.transitions.len() - 1,
    )?;
    if rkk_shape[3] < input.active_spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "rkk",
            expected: input.active_spin_channels,
            actual: rkk_shape[3],
        });
    }
    let q_count = rkk_shape[1];
    ensure_axis_len("pha", "q", input.q_phases.len(), q_count - 1)?;
    ensure_axis_len("beta", "q", input.q_beta_angles.len(), q_count - 1)?;
    ensure_axis_len("qweights", "q", input.q_weights.len(), q_count - 1)?;
    ensure_axis_len("cosmdff", "q", input.q_pair_cosines.shape()[0], q_count - 1)?;
    ensure_axis_len(
        "cosmdff",
        "qprime",
        input.q_pair_cosines.shape()[1],
        q_count - 1,
    )?;
    if input.q_pair_mode == MkgtrJasQPairMode::FirstToSecond && q_count < 2 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "cosmdff",
            axis: "qprime",
            index: 1,
        });
    }

    for (q, ((phase, beta), weight)) in input
        .q_phases
        .iter()
        .zip(input.q_beta_angles.iter())
        .zip(input.q_weights.iter())
        .take(q_count)
        .enumerate()
    {
        validate_finite_complex_value("pha", q, *phase)?;
        if !beta.is_finite() {
            return Err(FmsError::NonFiniteRotationAngle { name: "beta" });
        }
        validate_finite_complex_value("qweights", q, *weight)?;
    }
    for (index, cosine) in input
        .q_pair_cosines
        .iter()
        .copied()
        .take(q_count * q_count)
        .enumerate()
    {
        if !cosine.is_finite() {
            return Err(FmsError::NonFiniteComplexValue {
                table: "cosmdff",
                index,
            });
        }
    }
    for (index, transition) in input.transitions.iter().enumerate() {
        if transition.final_state_kappa == 0 {
            return Err(FmsError::TableIndexOutOfRange {
                table: "nrixs",
                axis: "transition",
                index,
            });
        }
    }
    Ok(())
}

fn mkgtr_jas_q_pairs(
    mode: MkgtrJasQPairMode,
    q_count: usize,
) -> Result<Vec<(usize, usize)>, FmsError> {
    match mode {
        MkgtrJasQPairMode::Diagonal => Ok((0..q_count).map(|q| (q, q)).collect()),
        MkgtrJasQPairMode::AllPairs => Ok((0..q_count)
            .flat_map(|left| (0..q_count).map(move |right| (left, right)))
            .collect()),
        MkgtrJasQPairMode::FirstToSecond if q_count >= 2 => Ok(vec![(0, 1)]),
        MkgtrJasQPairMode::FirstToSecond => Err(FmsError::TableIndexOutOfRange {
            table: "cosmdff",
            axis: "qprime",
            index: 1,
        }),
    }
}

fn mkgtr_jas_transition_weights(
    input: &MkgtrJasGreenTraceInput<'_>,
) -> Result<Array3<Real>, FmsError> {
    let transition_count = input.transitions.len();
    let mj_count = usize::try_from(input.initial_j2)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "hbmat",
            axis: "mj",
            index: 0,
        })?;
    let lgind = Array1::from_iter(
        input
            .transitions
            .iter()
            .map(|transition| i32::try_from(transition.decomposition_channel))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| FmsError::IntegerOverflow {
                field: "lgind",
                value: usize::MAX,
            })?,
    );
    let ljind = Array1::from_iter(
        input
            .transitions
            .iter()
            .map(|transition| i32::try_from(transition.multipole))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| FmsError::IntegerOverflow {
                field: "ljind",
                value: usize::MAX,
            })?,
    );
    let transition_lmax = input
        .transitions
        .iter()
        .map(|transition| transition.orbital_angular_momentum)
        .max()
        .unwrap_or(input.max_angular_momentum);
    let final_lj_max =
        i32::try_from(input.final_lj_max).map_err(|_| FmsError::IntegerOverflow {
            field: "ljmax",
            value: input.final_lj_max,
        })?;
    let mut weights = Array3::<Real>::zeros((mj_count, 2, transition_count).f());
    for mj_row in 0..mj_count {
        let mj = i32::try_from(mj_row)
            .ok()
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_sub(input.initial_j2))
            .ok_or(FmsError::TableIndexOutOfRange {
                table: "hbmat",
                axis: "mj",
                index: mj_row,
            })?;
        let row = xsph_nrixs_transition_weights(
            input.initial_kappa,
            mj,
            transition_lmax,
            input.final_j2_max,
            final_lj_max,
            lgind.view(),
            ljind.view(),
            transition_count,
        )?;
        for spin in 0..2 {
            for transition in 0..transition_count {
                weights[(mj_row, spin, transition)] = row[(spin, transition)];
            }
        }
    }
    Ok(weights)
}

fn mkgtr_jas_rotate_green(
    green: ArrayView2<'_, Complex32>,
    spin_channels: usize,
    lmax: usize,
    phase: Complex,
    beta: Real,
    rotate: bool,
) -> Result<Array2<Complex>, FmsError> {
    let angular_count = lmax.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lmax,
        lx: lmax,
    })?;
    let channel_count = angular_count
        .checked_mul(angular_count)
        .and_then(|value| value.checked_mul(spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lmax,
            lx: lmax,
        })?;
    let mut rotated = Array2::<Complex>::zeros((channel_count, channel_count).f());
    if !rotate {
        for row in 0..channel_count {
            for column in 0..channel_count {
                rotated[(row, column)] = widen_complex32(green[(row, column)]);
            }
        }
        return Ok(rotated);
    }

    for spin1 in 0..spin_channels {
        for spin2 in 0..spin_channels {
            for l1 in 0..=lmax {
                let l1_i32 = i32::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
                    name: "lx",
                    value: lmax,
                    lx: lmax,
                })?;
                for m1 in -l1_i32..=l1_i32 {
                    let output1 = mkgtr_jas_channel_index(spin_channels, l1, m1, spin1)?;
                    for l2 in 0..=lmax {
                        let l2_i32 =
                            i32::try_from(l2).map_err(|_| FmsError::InvalidAngularLimit {
                                name: "lx",
                                value: lmax,
                                lx: lmax,
                            })?;
                        for m2 in -l2_i32..=l2_i32 {
                            let output2 = mkgtr_jas_channel_index(spin_channels, l2, m2, spin2)?;
                            let mut value = Complex::new(0.0, 0.0);
                            for mp1 in -l1_i32..=l1_i32 {
                                let input1 =
                                    mkgtr_jas_channel_index(spin_channels, l1, mp1, spin1)?;
                                let rotation1 =
                                    mkgtr_jas_rotation_entry(phase, beta, l1_i32, m1, mp1)?;
                                for mp2 in -l2_i32..=l2_i32 {
                                    let input2 =
                                        mkgtr_jas_channel_index(spin_channels, l2, mp2, spin2)?;
                                    let rotation2 =
                                        mkgtr_jas_rotation_entry(phase, beta, l2_i32, m2, mp2)?;
                                    value += rotation1
                                        * widen_complex32(green[(input1, input2)])
                                        * rotation2.conj();
                                }
                            }
                            // FEFF `rotgmatrix` stores `ggrot(ig2,ig1)`.
                            rotated[(output2, output1)] = value;
                        }
                    }
                }
            }
        }
    }
    Ok(rotated)
}

fn mkgtr_jas_rotation_entry(
    phase: Complex,
    beta: Real,
    angular: i32,
    output_magnetic: i32,
    input_magnetic: i32,
) -> Result<Complex, FmsError> {
    let azimuth = phase.conj().powi(output_magnetic);
    Ok(azimuth * wigner_rotation(-beta, angular, output_magnetic, input_magnetic, 1)?)
}

#[allow(clippy::too_many_arguments)]
fn mkgtr_jas_accumulate_nonspherical(
    input: &MkgtrJasGreenTraceInput<'_>,
    energy: usize,
    left_q: usize,
    right_q: usize,
    green: ArrayView2<'_, Complex>,
    transition_weights: &Array3<Real>,
    legendre: &[Real],
    trace: &mut Array1<Complex>,
    mut decomposed: Option<&mut Array3<Complex>>,
) -> Result<(), FmsError> {
    let mj_count = transition_weights.shape()[0];
    for (transition1, first) in input.transitions.iter().enumerate() {
        if first.decomposition_channel > input.max_angular_momentum {
            continue;
        }
        let jfin1 = doubled_j_from_kappa(first.final_state_kappa)?;
        let lg1 = first.decomposition_channel;
        let angular_phase1 = imaginary_unit_power(first.multipole as i32).conj()
            * imaginary_unit_power(lg1 as i32).conj();
        for mj_row in 0..mj_count {
            let mj = i32::try_from(mj_row)
                .ok()
                .and_then(|value| value.checked_mul(2))
                .and_then(|value| value.checked_sub(input.initial_j2))
                .ok_or(FmsError::TableIndexOutOfRange {
                    table: "hbmat",
                    axis: "mj",
                    index: mj_row,
                })?;
            for spin1 in 0..input.active_spin_channels {
                let magnetic1 = mkgtr_jas_magnetic_from_mj(mj, spin1)?;
                if magnetic1.unsigned_abs() as usize > lg1 || mj.abs() > jfin1 {
                    continue;
                }
                let channel1 =
                    mkgtr_jas_channel_index(input.active_spin_channels, lg1, magnetic1, spin1)?;
                let radial1 = input.transition_moments[(energy, left_q, transition1, spin1)]
                    * input.q_weights[left_q]
                    * transition_weights[(mj_row, spin1, transition1)];
                for (transition2, second) in input.transitions.iter().enumerate() {
                    if second.decomposition_channel > input.max_angular_momentum {
                        continue;
                    }
                    let jfin2 = doubled_j_from_kappa(second.final_state_kappa)?;
                    let lg2 = second.decomposition_channel;
                    let angular_phase2 = imaginary_unit_power(second.multipole as i32)
                        * imaginary_unit_power(lg2 as i32);
                    for spin2 in 0..input.active_spin_channels {
                        let magnetic2 = mkgtr_jas_magnetic_from_mj(mj, spin2)?;
                        if magnetic2.unsigned_abs() as usize > lg2 || mj.abs() > jfin2 {
                            continue;
                        }
                        let channel2 = mkgtr_jas_channel_index(
                            input.active_spin_channels,
                            lg2,
                            magnetic2,
                            spin2,
                        )?;
                        let radial2 = input.transition_moments
                            [(energy, right_q, transition2, spin2)]
                            * input.q_weights[right_q]
                            * transition_weights[(mj_row, spin2, transition2)];
                        let value = green[(channel2, channel1)]
                            * radial1
                            * radial2
                            * angular_phase1
                            * angular_phase2
                            * legendre[first.multipole];
                        trace[energy] += value;
                        if let (Some(maximum), Some(decomposed)) =
                            (input.max_decomposition_channel, decomposed.as_deref_mut())
                            && lg1 <= maximum
                            && lg2 <= maximum
                        {
                            decomposed[(energy, lg2, lg1)] += value;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mkgtr_jas_accumulate_spherical(
    input: &MkgtrJasGreenTraceInput<'_>,
    energy: usize,
    left_q: usize,
    right_q: usize,
    green: ArrayView2<'_, Complex>,
    clebsch: &ndarray::Array4<Real>,
    legendre: &[Real],
    trace: &mut Array1<Complex>,
    mut decomposed: Option<&mut Array3<Complex>>,
) -> Result<(), FmsError> {
    for (transition, state) in input.transitions.iter().enumerate() {
        if state.decomposition_channel > input.max_angular_momentum {
            continue;
        }
        let jfin = doubled_j_from_kappa(state.final_state_kappa)?;
        let lg = state.decomposition_channel;
        let j_index =
            usize::try_from((jfin - 1) / 2).map_err(|_| FmsError::TableIndexOutOfRange {
                table: "clbcoef",
                axis: "j",
                index: 0,
            })?;
        let mut mj = -jfin;
        while mj <= jfin {
            let mj_index =
                usize::try_from((mj + jfin) / 2).map_err(|_| FmsError::TableIndexOutOfRange {
                    table: "clbcoef",
                    axis: "mj",
                    index: 0,
                })?;
            for spin1 in 0..input.active_spin_channels {
                let magnetic1 = mkgtr_jas_magnetic_from_mj(mj, spin1)?;
                if magnetic1.unsigned_abs() as usize > lg {
                    continue;
                }
                let channel1 =
                    mkgtr_jas_channel_index(input.active_spin_channels, lg, magnetic1, spin1)?;
                for spin2 in 0..input.active_spin_channels {
                    let magnetic2 = mkgtr_jas_magnetic_from_mj(mj, spin2)?;
                    if magnetic2.unsigned_abs() as usize > lg {
                        continue;
                    }
                    let channel2 =
                        mkgtr_jas_channel_index(input.active_spin_channels, lg, magnetic2, spin2)?;
                    let denominator = (2 * state.multipole + 1) as Real;
                    let value = green[(channel1, channel2)]
                        * input.transition_moments[(energy, left_q, transition, spin1)]
                        * input.transition_moments[(energy, right_q, transition, spin2)]
                        * input.q_weights[left_q]
                        * input.q_weights[right_q]
                        * clebsch[(mj_index, j_index, spin1, lg)]
                        * clebsch[(mj_index, j_index, spin2, lg)]
                        * legendre[state.multipole]
                        / denominator;
                    trace[energy] += value;
                    if let (Some(maximum), Some(decomposed)) =
                        (input.max_decomposition_channel, decomposed.as_deref_mut())
                        && lg <= maximum
                    {
                        decomposed[(energy, lg, lg)] += value;
                    }
                }
            }
            mj += 2;
        }
    }
    Ok(())
}

fn doubled_j_from_kappa(kappa: i32) -> Result<i32, FmsError> {
    kappa
        .checked_abs()
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_sub(1))
        .filter(|value| *value >= 0)
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "nrixs",
            axis: "kappa",
            index: 0,
        })
}

fn mkgtr_jas_magnetic_from_mj(mj: i32, spin: usize) -> Result<i32, FmsError> {
    let spin_mj = i32::try_from(spin)
        .ok()
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_sub(1))
        .ok_or(FmsError::InvalidSpinChannelCount { value: spin })?;
    mj.checked_sub(spin_mj)
        .map(|value| value / 2)
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "nrixs",
            axis: "magnetic",
            index: 0,
        })
}

fn mkgtr_jas_channel_index(
    spin_channels: usize,
    angular: usize,
    magnetic: i32,
    spin: usize,
) -> Result<usize, FmsError> {
    if spin >= spin_channels || magnetic.unsigned_abs() as usize > angular {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "channel",
            index: spin,
        });
    }
    let orbital_center = angular
        .checked_mul(angular)
        .and_then(|value| value.checked_add(angular))
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "channel",
            index: spin,
        })?;
    let orbital = isize::try_from(orbital_center)
        .ok()
        .and_then(|value| value.checked_add(magnetic as isize))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "channel",
            index: spin,
        })?;
    orbital
        .checked_mul(spin_channels)
        .and_then(|value| value.checked_add(spin))
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "channel",
            index: spin,
        })
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

/// Fold FEFF FMS Green's-function matrices into MKGTR trace spectra.
///
/// This ports the non-NRIXS `Form gtr` loop in `MKGTR/getgtr.f90`. The input
/// Green's functions are the absorber-potential `gg` matrices for each energy,
/// while `transition_matrices` corresponds to the per-spectrum `bmat` blocks
/// built by FEFF `bcoef`.
pub fn mkgtr_green_trace(
    input: MkgtrGreenTraceInput<'_>,
) -> Result<MkgtrGreenTraceResult, FmsError> {
    ensure_spin_channels(input.active_spin_channels)?;
    let shape = input.green_functions.shape();
    if shape[0] == 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "energy",
            index: 0,
        });
    }
    if shape[1] == 0 || shape[1] != shape[2] {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "shape",
            index: shape[1],
        });
    }
    if input.transition_matrices.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "bmat",
            axis: "spectrum",
            index: 0,
        });
    }
    ensure_axis_len(
        "rkk",
        "energy",
        input.transition_moments.shape()[0],
        shape[0] - 1,
    )?;
    ensure_axis_len("rkk", "transition", input.transition_moments.shape()[1], 7)?;
    if input.transition_moments.shape()[2] < input.active_spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "rkk",
            expected: input.active_spin_channels,
            actual: input.transition_moments.shape()[2],
        });
    }

    for (spectrum, matrix) in input.transition_matrices.iter().enumerate() {
        validate_mkgtr_transition_matrix(spectrum, matrix)?;
        validate_mkgtr_green_channels(
            input.green_functions.shape()[1],
            input.active_spin_channels,
            matrix,
        )?;
    }

    let mut traces = Array2::zeros((input.transition_matrices.len(), shape[0]).f());
    for (spectrum, matrix) in input.transition_matrices.iter().enumerate() {
        for energy in 0..shape[0] {
            traces[(spectrum, energy)] = mkgtr_green_trace_energy(&input, matrix, energy)?;
        }
    }
    Ok(MkgtrGreenTraceResult { traces })
}

fn mkgtr_green_trace_energy(
    input: &MkgtrGreenTraceInput<'_>,
    transition_matrix: &TransitionBMatrix,
    energy: usize,
) -> Result<Complex, FmsError> {
    let mut trace = Complex::new(0.0, 0.0);
    for transition1 in 0..8 {
        let angular1 = transition_matrix.orbital_momenta[transition1];
        if angular1 < 0 {
            continue;
        }
        let angular1 = usize::try_from(angular1).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: 0,
            lx: transition_matrix.l_offset,
        })?;
        for spin1 in 0..input.active_spin_channels {
            let rkk1 = input.transition_moments[(energy, transition1, spin1)];
            validate_finite_complex_value(
                "rkk",
                flat_index3(input.transition_moments.shape(), energy, transition1, spin1),
                rkk1,
            )?;
            for transition2 in 0..8 {
                let angular2 = transition_matrix.orbital_momenta[transition2];
                if angular2 < 0 {
                    continue;
                }
                let angular2 =
                    usize::try_from(angular2).map_err(|_| FmsError::InvalidAngularLimit {
                        name: "lnd",
                        value: 0,
                        lx: transition_matrix.l_offset,
                    })?;
                for spin2 in 0..input.active_spin_channels {
                    let rkk2 = input.transition_moments[(energy, transition2, spin2)];
                    validate_finite_complex_value(
                        "rkk",
                        flat_index3(input.transition_moments.shape(), energy, transition2, spin2),
                        rkk2,
                    )?;
                    for magnetic1 in signed_magnetic_range(angular1)? {
                        let row = mkgtr_channel_index(
                            input.active_spin_channels,
                            angular1,
                            magnetic1,
                            spin1,
                        )?;
                        for magnetic2 in signed_magnetic_range(angular2)? {
                            let column = mkgtr_channel_index(
                                input.active_spin_channels,
                                angular2,
                                magnetic2,
                                spin2,
                            )?;
                            let green = input.green_functions[(energy, row, column)];
                            validate_finite_complex32_value(
                                "gg",
                                flat_index3(input.green_functions.shape(), energy, row, column),
                                green,
                            )?;
                            let bmat = transition_matrix
                                .value(
                                    magnetic2 as isize,
                                    spin2,
                                    transition2 + 1,
                                    magnetic1 as isize,
                                    spin1,
                                    transition1 + 1,
                                )
                                .ok_or(FmsError::TableIndexOutOfRange {
                                    table: "bmat",
                                    axis: "magnetic",
                                    index: transition_matrix.l_offset,
                                })?;
                            validate_finite_complex_value(
                                "bmat",
                                flat_index6(
                                    transition_matrix.matrix.shape(),
                                    [
                                        signed_to_shifted_magnetic(
                                            magnetic2,
                                            transition_matrix.l_offset,
                                        )?,
                                        spin2,
                                        transition2,
                                        signed_to_shifted_magnetic(
                                            magnetic1,
                                            transition_matrix.l_offset,
                                        )?,
                                        spin1,
                                        transition1,
                                    ],
                                ),
                                bmat,
                            )?;
                            trace += widen_complex32(green) * bmat * rkk1 * rkk2;
                        }
                    }
                }
            }
        }
    }
    validate_finite_complex_value("gtr", energy, trace)?;
    Ok(trace)
}

fn validate_mkgtr_transition_matrix(
    _spectrum: usize,
    matrix: &TransitionBMatrix,
) -> Result<(), FmsError> {
    let shape = matrix.matrix.shape();
    ensure_axis_len("bmat", "ml2", shape[0], matrix.l_offset)?;
    ensure_axis_len("bmat", "ms2", shape[1], 1)?;
    ensure_axis_len("bmat", "transition2", shape[2], 7)?;
    ensure_axis_len("bmat", "ml1", shape[3], matrix.l_offset)?;
    ensure_axis_len("bmat", "ms1", shape[4], 1)?;
    ensure_axis_len("bmat", "transition1", shape[5], 7)?;

    for angular in matrix.orbital_momenta {
        if angular < 0 {
            continue;
        }
        let angular = usize::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: 0,
            lx: matrix.l_offset,
        })?;
        if matrix.l_offset < angular {
            return Err(FmsError::TableIndexOutOfRange {
                table: "bmat",
                axis: "magnetic",
                index: angular,
            });
        }
        let high = matrix
            .l_offset
            .checked_add(angular)
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lnd",
                value: angular,
                lx: matrix.l_offset,
            })?;
        ensure_axis_len("bmat", "ml2", shape[0], high)?;
        ensure_axis_len("bmat", "ml1", shape[3], high)?;
    }
    Ok(())
}

fn validate_mkgtr_green_channels(
    channel_count: usize,
    spin_channels: usize,
    matrix: &TransitionBMatrix,
) -> Result<(), FmsError> {
    for angular in matrix.orbital_momenta {
        if angular < 0 {
            continue;
        }
        let angular = usize::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: 0,
            lx: matrix.l_offset,
        })?;
        let magnetic = i32::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: angular,
            lx: matrix.l_offset,
        })?;
        let channel = mkgtr_channel_index(spin_channels, angular, magnetic, spin_channels - 1)?;
        ensure_axis_len("gg", "channel", channel_count, channel)?;
    }
    Ok(())
}

fn mkgtr_channel_index(
    spin_channels: usize,
    angular: usize,
    magnetic: i32,
    spin: usize,
) -> Result<usize, FmsError> {
    let angular_isize = isize::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lnd",
        value: angular,
        lx: angular,
    })?;
    let magnetic_isize = magnetic as isize;
    let orbital = angular_isize
        .checked_mul(angular_isize)
        .and_then(|value| value.checked_add(angular_isize))
        .and_then(|value| value.checked_add(magnetic_isize))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lnd",
            value: angular,
            lx: angular,
        })?;
    let orbital = usize::try_from(orbital).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lnd",
        value: angular,
        lx: angular,
    })?;
    orbital
        .checked_mul(spin_channels)
        .and_then(|value| value.checked_add(spin))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lnd",
            value: angular,
            lx: angular,
        })
}

fn signed_magnetic_range(angular: usize) -> Result<std::ops::RangeInclusive<i32>, FmsError> {
    let angular = i32::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lnd",
        value: angular,
        lx: angular,
    })?;
    Ok(-angular..=angular)
}

fn signed_to_shifted_magnetic(magnetic: i32, offset: usize) -> Result<usize, FmsError> {
    let offset_i32 = i32::try_from(offset).map_err(|_| FmsError::InvalidAngularLimit {
        name: "bmat",
        value: offset,
        lx: offset,
    })?;
    let shifted = magnetic
        .checked_add(offset_i32)
        .ok_or(FmsError::InvalidAngularLimit {
            name: "bmat",
            value: offset,
            lx: offset,
        })?;
    usize::try_from(shifted).map_err(|_| FmsError::TableIndexOutOfRange {
        table: "bmat",
        axis: "magnetic",
        index: 0,
    })
}

fn validate_finite_complex32_value(
    table: &'static str,
    index: usize,
    value: Complex32,
) -> Result<(), FmsError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FmsError::NonFiniteComplexValue { table, index })
    }
}

fn validate_finite_complex_value(
    table: &'static str,
    index: usize,
    value: Complex,
) -> Result<(), FmsError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FmsError::NonFiniteComplexValue { table, index })
    }
}

fn widen_complex32(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}

fn flat_index3(shape: &[usize], axis0: usize, axis1: usize, axis2: usize) -> usize {
    let dim1 = match shape.get(1) {
        Some(value) => *value,
        None => 0,
    };
    let dim2 = match shape.get(2) {
        Some(value) => *value,
        None => 0,
    };
    axis0
        .saturating_mul(dim1)
        .saturating_add(axis1)
        .saturating_mul(dim2)
        .saturating_add(axis2)
}

fn flat_index6(shape: &[usize], axes: [usize; 6]) -> usize {
    axes.into_iter()
        .enumerate()
        .fold(0usize, |index, (axis, value)| {
            let dimension = match shape.get(axis) {
                Some(value) => *value,
                None => 0,
            };
            index.saturating_mul(dimension).saturating_add(value)
        })
}
