//! FEFF XSPH phase-shift matching helpers.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, Axis, ShapeBuilder};

use crate::{
    Complex, DiracSpinorOrbitalsGridInput, FovrgDiracSolverInput, PotentialGridInput, besjn,
    fix_dirac_spinor_orbitals_grid, fix_potential_grid, fovrg_dirac_solver,
    muffin_tin_phase_amplitude,
};

use super::{
    XSPH_BOHR_ANGSTROM, XSPH_FINE_STRUCTURE_ALPHA, XSPH_HARTREE_EV, XsphEmptyCellPhase,
    XsphEmptyCellPhaseInput, XsphError, XsphHubbardPhaseAssignment,
    XsphHubbardPhaseAssignmentInput, XsphHubbardPhasePotentialInput,
    XsphHubbardPhasePotentialShift, XsphPhaseAngularLimit, XsphPhaseAngularLimitInput,
    XsphPhaseChannel, XsphPhaseChannelPlan, XsphPhaseChannelPlanInput, XsphPhaseCutoff,
    XsphPhaseCutoffInput, XsphPhaseEnergyDecision, XsphPhaseEnergyDynamics, XsphPhaseEnergySetup,
    XsphPhaseEnergySetupInput, XsphPhaseGridPreparation, XsphPhaseGridPreparationInput,
    XsphPhasePlasmonPole, XsphPhasePlasmonPoleSetup, XsphPhasePlasmonPoleSetupInput,
    XsphPhaseRadialHeader, XsphPhaseRadialHeaderInput, XsphPhaseRadialIndices,
    XsphPhaseRadialIndicesInput, XsphPhaseRadialOutput, XsphPhaseRadialOutputInput,
    XsphPhaseReferenceTail, XsphPhaseSelfEnergySummary, XsphPhaseSelfEnergySummaryInput,
    XsphRegularPhase, XsphRegularPhaseChannel, XsphRegularPhaseInput, nint, usize_to_i32,
    validate_active_len, validate_finite_complex, validate_finite_real,
};

const PHASE_MIN_ACTIVE_ENERGY: f64 = -10.0;
const PHASE_MAX_ACTIVE_ENERGY: f64 = 300.0;
const PHASE_LOCAL_EXCHANGE_THRESHOLD: f64 = 0.50;
const PHASE_INITIAL_STOP_CUTOFF: f64 = 1.0e-6;
const PHASE_SCATTERING_ZERO_CUTOFF: f64 = 1.0e-5;
const PHASE_FINAL_STOP_CUTOFF: f64 = 1.0e-5;
const PHASE_STOP_MIN_CHANNEL: i32 = 4;
const PHASE_LMAX_PREFAC: f64 = 0.7;
const PHASE_LMAX_FLOOR: usize = 5;

/// Prepare unreferenced XSPH phase grids from FEFF `pot.bin`-style arrays.
///
/// This mirrors the `fixvar`/`fixdsx` setup in `XSPH/xsphsub.f90` immediately
/// before calling `phase.f90`. Unlike RHORRP preparation, this intentionally
/// does not subtract an `eref0` reference; XSPH calls `xcpot` later for every
/// energy row and that routine performs the energy-dependent reference shift.
pub fn xsph_phase_grid_preparation(
    input: XsphPhaseGridPreparationInput<'_>,
) -> Result<XsphPhaseGridPreparation, XsphError> {
    validate_phase_grid_preparation_input(&input)?;

    let (_, potential_count) = input.total_potential.dim();
    let (_, orbital_count, _) = input.bound_large_components.dim();
    let mut radii = Array1::<f64>::zeros(input.radial_count);
    let mut potential_jumps = Array1::<f64>::zeros(potential_count);
    let mut total_potential = Array2::<f64>::zeros((input.radial_count, potential_count).f());
    let mut valence_potential = Array2::<f64>::zeros((input.radial_count, potential_count).f());
    let mut electron_density = Array2::<f64>::zeros((input.radial_count, potential_count).f());
    let mut valence_density = Array2::<f64>::zeros((input.radial_count, potential_count).f());
    let mut magnetization = Array2::<f64>::zeros((input.radial_count, potential_count).f());
    let mut bound_large_components =
        Array3::<f64>::zeros((input.radial_count, orbital_count, potential_count).f());
    let mut bound_small_components =
        Array3::<f64>::zeros((input.radial_count, orbital_count, potential_count).f());
    let mut bound_active_lengths = Array2::<usize>::zeros((orbital_count, potential_count).f());
    let prepare_valence_potential = input.exchange_selector.rem_euclid(10) >= 5;

    for potential in 0..potential_count {
        let total_grid = fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: input.muffin_tin_radii[potential],
            electron_density: input.electron_density.index_axis(Axis(1), potential),
            total_potential: input.total_potential.index_axis(Axis(1), potential),
            magnetization: input.magnetization.index_axis(Axis(1), potential),
            interstitial_potential: input.interstitial_potential,
            interstitial_density: input.interstitial_density,
            original_delta: input.original_radial_dx,
            new_delta: input.target_radial_dx,
            jump_mode: input.jump_mode,
            potential_jump: input.potential_jump,
            output_len: input.radial_count,
        })?;
        if potential == 0 {
            radii.assign(&total_grid.radii);
        }
        potential_jumps[potential] = total_grid.potential_jump;
        total_potential
            .index_axis_mut(Axis(1), potential)
            .assign(&total_grid.total_potential);
        electron_density
            .index_axis_mut(Axis(1), potential)
            .assign(&total_grid.charge_density);
        magnetization
            .index_axis_mut(Axis(1), potential)
            .assign(&total_grid.magnetization);

        if prepare_valence_potential {
            let valence_jump_mode = if input.jump_mode > 0 {
                2
            } else {
                input.jump_mode
            };
            let valence_grid = fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: input.muffin_tin_radii[potential],
                electron_density: input.valence_density.index_axis(Axis(1), potential),
                total_potential: input.valence_potential.index_axis(Axis(1), potential),
                magnetization: input.magnetization.index_axis(Axis(1), potential),
                interstitial_potential: input.interstitial_potential,
                interstitial_density: input.interstitial_density,
                original_delta: input.original_radial_dx,
                new_delta: input.target_radial_dx,
                jump_mode: valence_jump_mode,
                potential_jump: total_grid.potential_jump,
                output_len: input.radial_count,
            })?;
            valence_potential
                .index_axis_mut(Axis(1), potential)
                .assign(&valence_grid.total_potential);
            valence_density
                .index_axis_mut(Axis(1), potential)
                .assign(&valence_grid.charge_density);
        } else {
            valence_potential
                .index_axis_mut(Axis(1), potential)
                .assign(&total_grid.total_potential);
        }

        let spinors = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: input.original_radial_dx,
            new_delta: input.target_radial_dx,
            large_components: input.bound_large_components.index_axis(Axis(2), potential),
            small_components: input.bound_small_components.index_axis(Axis(2), potential),
            output_len: input.radial_count,
        })?;
        bound_large_components
            .index_axis_mut(Axis(2), potential)
            .assign(&spinors.large_components);
        bound_small_components
            .index_axis_mut(Axis(2), potential)
            .assign(&spinors.small_components);
        bound_active_lengths
            .index_axis_mut(Axis(1), potential)
            .assign(&spinors.active_lengths);
    }

    Ok(XsphPhaseGridPreparation {
        radii,
        radial_dx: input.target_radial_dx,
        potential_jumps,
        total_potential,
        valence_potential,
        electron_density,
        valence_density,
        magnetization,
        bound_large_components,
        bound_small_components,
        bound_active_lengths,
    })
}

/// Port of the `lmax` planning block in FEFF `XSPH/phase.f90`.
///
/// FEFF scans only `em(1:ne-ne3)` for the largest real energy, converts that
/// to `sqrt(2E)`, applies `int(0.7 * rmt * kmax)`, floors the result at five,
/// and finally caps it at the compiled `ltot`. The diagnostic wave number is
/// returned when the uncapped limit exceeds `ltot`, matching the value FEFF
/// writes in its warning.
pub fn xsph_phase_angular_limit(
    input: XsphPhaseAngularLimitInput<'_>,
) -> Result<XsphPhaseAngularLimit, XsphError> {
    validate_phase_angular_limit_input(&input)?;

    let scan_count = input.energy_count - input.auxiliary_count;
    let mut max_energy = 0.0;
    for index in 0..scan_count {
        let energy = input.energies[index];
        if max_energy < energy.re {
            max_energy = energy.re;
        }
    }
    let max_wave_number = (2.0 * max_energy).sqrt();
    let raw_limit = phase_nonnegative_integer_assignment(
        PHASE_LMAX_PREFAC * input.muffin_tin_radius * max_wave_number,
    );
    let uncapped_limit = raw_limit.max(PHASE_LMAX_FLOOR);
    let accuracy_warning_wave_number = if uncapped_limit > input.max_angular_momentum {
        Some(nint(
            input.max_angular_momentum as f64
                / input.muffin_tin_radius
                / XSPH_BOHR_ANGSTROM
                / PHASE_LMAX_PREFAC,
        ))
    } else {
        None
    };

    Ok(XsphPhaseAngularLimit {
        angular_limit: uncapped_limit.min(input.max_angular_momentum),
        uncapped_limit,
        max_wave_number,
        accuracy_warning_wave_number,
    })
}

/// Port of the per-energy setup branch in FEFF `XSPH/phase.f90`.
///
/// This mirrors the block between `xcpot` and the per-`ll` radial loop: FEFF
/// skips real energies outside `[-10, 300]`, builds `p2` and `p2EC`, optionally
/// forces `p2` real on the `ne1` prefix, computes relativistic wave numbers,
/// skips nonpositive `p2`, and finally chooses the `dfovrg` cycle count from
/// `ixc`.
pub fn xsph_phase_energy_setup(
    input: XsphPhaseEnergySetupInput,
) -> Result<XsphPhaseEnergySetup, XsphError> {
    validate_phase_energy_setup_input(input)?;

    if input.energy.re < PHASE_MIN_ACTIVE_ENERGY || input.energy.re > PHASE_MAX_ACTIVE_ENERGY {
        return Ok(XsphPhaseEnergySetup {
            decision: XsphPhaseEnergyDecision::OutsideEnergyWindow,
            dynamics: None,
            cycle_count: None,
        });
    }

    let mut momentum_squared = input.energy - input.reference_energy;
    let empty_cell_momentum_squared =
        momentum_squared - Complex::new(input.muffin_tin_potential, 0.0);
    if input.lreal > 1 && input.energy_index < input.real_mesh_count {
        momentum_squared = Complex::new(momentum_squared.re, 0.0);
    }

    let wave_number = phase_relativistic_wave_number(momentum_squared);
    let empty_cell_wave_number = phase_relativistic_wave_number(empty_cell_momentum_squared);
    let muffin_tin_argument = wave_number * input.muffin_tin_radius;
    let empty_cell_muffin_tin_argument = empty_cell_wave_number * input.muffin_tin_radius;

    validate_phase_energy_dynamics(
        momentum_squared,
        empty_cell_momentum_squared,
        wave_number,
        empty_cell_wave_number,
        muffin_tin_argument,
        empty_cell_muffin_tin_argument,
    )?;

    let dynamics = XsphPhaseEnergyDynamics {
        momentum_squared,
        empty_cell_momentum_squared,
        wave_number,
        empty_cell_wave_number,
        muffin_tin_argument,
        empty_cell_muffin_tin_argument,
    };
    if momentum_squared.re <= 0.0 && momentum_squared.im <= 0.0 {
        return Ok(XsphPhaseEnergySetup {
            decision: XsphPhaseEnergyDecision::NonPositiveMomentum,
            dynamics: Some(dynamics),
            cycle_count: None,
        });
    }

    Ok(XsphPhaseEnergySetup {
        decision: XsphPhaseEnergyDecision::Active,
        dynamics: Some(dynamics),
        cycle_count: Some(phase_cycle_count(input.exchange_selector)),
    })
}

/// Port of the per-`ll` channel setup loop in FEFF `XSPH/phase.f90`.
///
/// FEFF traverses `ll = -lmax..lmax`, derives the one-based `il`/`ilp` radial
/// component indices and relativistic `ikap`, optionally removes spin-orbit
/// splitting for spin-averaged calculations, and forces local exchange when
/// `il * dx > 0.50`.
pub fn xsph_phase_channel_plan(
    input: XsphPhaseChannelPlanInput,
) -> Result<XsphPhaseChannelPlan, XsphError> {
    validate_phase_channel_plan_input(input)?;

    let lmax = usize_to_i32("angular_limit", input.angular_limit)?;
    let capacity = input
        .angular_limit
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::SizeOutOfRange {
            name: "phase_channel_count",
            value: input.angular_limit,
        })?;
    let mut cycle_count = input.initial_cycle_count;
    let mut channels = Vec::with_capacity(capacity);

    for angular_channel in -lmax..=lmax {
        let abs_channel = angular_channel
            .checked_abs()
            .ok_or(XsphError::IntegerOutOfRange {
                name: "angular_channel",
                value: angular_channel,
            })?;
        let orbital_index = usize::try_from(abs_channel)
            .map_err(|_| XsphError::IntegerOutOfRange {
                name: "angular_channel",
                value: angular_channel,
            })?
            .checked_add(1)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "angular_channel",
                value: angular_channel,
            })?;
        let forces_local_exchange =
            orbital_index as f64 * input.log_step > PHASE_LOCAL_EXCHANGE_THRESHOLD;
        if forces_local_exchange {
            cycle_count = 0;
        }

        let mut kappa = phase_channel_kappa(angular_channel)?;
        let mut partner_orbital_index = if kappa > 0 {
            orbital_index
                .checked_sub(1)
                .ok_or(XsphError::IntegerOutOfRange {
                    name: "angular_channel",
                    value: angular_channel,
                })?
        } else {
            orbital_index
                .checked_add(1)
                .ok_or(XsphError::SizeOutOfRange {
                    name: "partner_orbital_index",
                    value: orbital_index,
                })?
        };
        let mut c3_derivative = 0;

        if input.spin_channels == 1 && input.spin == 0 {
            if angular_channel != 0 {
                c3_derivative = 1;
            }
            kappa = abs_channel
                .checked_add(1)
                .and_then(i32::checked_neg)
                .ok_or(XsphError::IntegerOutOfRange {
                    name: "angular_channel",
                    value: angular_channel,
                })?;
            partner_orbital_index =
                orbital_index
                    .checked_add(1)
                    .ok_or(XsphError::SizeOutOfRange {
                        name: "partner_orbital_index",
                        value: orbital_index,
                    })?;
        }

        channels.push(XsphPhaseChannel {
            angular_channel,
            orbital_index,
            partner_orbital_index,
            kappa,
            c3_derivative,
            cycle_count,
            forces_local_exchange,
        });
    }

    Ok(XsphPhaseChannelPlan { channels })
}

/// Port of the small-phase cutoff block after `phamp` in FEFF `XSPH/phase.f90`.
///
/// FEFF first stops high positive angular channels when `abs(ph) < 1e-6`.
/// Only otherwise does it zero phases whose scattering factor is effectively
/// one, followed by a second high-`ll` stop at `abs(ph) < 1e-5`.
pub fn xsph_phase_cutoff(input: XsphPhaseCutoffInput) -> Result<XsphPhaseCutoff, XsphError> {
    validate_finite_complex("phase_shift", 0, input.phase_shift)?;

    let high_positive_channel = input.angular_channel >= PHASE_STOP_MIN_CHANNEL;
    let mut phase_shift = input.phase_shift;
    if phase_shift.norm() < PHASE_INITIAL_STOP_CUTOFF && high_positive_channel {
        return Ok(XsphPhaseCutoff {
            phase_shift,
            zeroed: false,
            terminate_energy: true,
        });
    }

    let scattering_change = (Complex::new(0.0, 2.0) * phase_shift).exp() - Complex::new(1.0, 0.0);
    validate_finite_complex("phase_scattering_change", 0, scattering_change)?;
    let zeroed = scattering_change.norm() < PHASE_SCATTERING_ZERO_CUTOFF;
    if zeroed {
        phase_shift = Complex::new(0.0, 0.0);
    }
    let terminate_energy = phase_shift.norm() < PHASE_FINAL_STOP_CUTOFF && high_positive_channel;

    Ok(XsphPhaseCutoff {
        phase_shift,
        zeroed,
        terminate_energy,
    })
}

/// Port of the FEFF `XSPH/phase.f90` final `eref` tail copy.
///
/// FEFF computes `ne12 = ne - ne3`, then copies `eref(ne1)` into
/// `eref(max(1, ne12 + 1):ne)`. The `max(1, ...)` behavior is preserved here:
/// when the auxiliary count covers or exceeds the active energy count, the
/// whole active prefix is filled from `ne1`.
pub fn xsph_phase_reference_tail(
    reference_energies: ArrayViewMut1<'_, Complex>,
    energy_count: usize,
    real_mesh_count: usize,
    auxiliary_count: usize,
) -> Result<XsphPhaseReferenceTail, XsphError> {
    if reference_energies.len() < energy_count {
        return Err(XsphError::LengthTooShort {
            name: "reference_energies",
            required: energy_count,
            actual: reference_energies.len(),
        });
    }

    let start_index_1based = if auxiliary_count >= energy_count {
        1
    } else {
        energy_count - auxiliary_count + 1
    };
    fill_phase_reference_tail(
        reference_energies,
        energy_count,
        real_mesh_count,
        start_index_1based,
    )
}

/// Port of the FEFF `XSPH/phase_h.f90` final `eref` tail copy.
///
/// The Hubbard variant copies `eref(ne1)` into `eref(ne-ne3+1:ne)` without the
/// defensive lower-bound clamp used by `phase.f90`. This safe Rust wrapper
/// rejects `ne3 > ne`, which would make the FEFF start index nonpositive.
pub fn xsph_hubbard_phase_reference_tail(
    reference_energies: ArrayViewMut1<'_, Complex>,
    energy_count: usize,
    real_mesh_count: usize,
    auxiliary_count: usize,
) -> Result<XsphPhaseReferenceTail, XsphError> {
    if reference_energies.len() < energy_count {
        return Err(XsphError::LengthTooShort {
            name: "reference_energies",
            required: energy_count,
            actual: reference_energies.len(),
        });
    }
    if auxiliary_count > energy_count {
        return Err(XsphError::InvalidAuxiliaryEnergyCount {
            auxiliary_count,
            energy_count,
        });
    }

    fill_phase_reference_tail(
        reference_energies,
        energy_count,
        real_mesh_count,
        energy_count - auxiliary_count + 1,
    )
}

fn fill_phase_reference_tail(
    mut reference_energies: ArrayViewMut1<'_, Complex>,
    energy_count: usize,
    real_mesh_count: usize,
    start_index_1based: usize,
) -> Result<XsphPhaseReferenceTail, XsphError> {
    if energy_count == 0 || start_index_1based > energy_count {
        return Ok(XsphPhaseReferenceTail {
            start_index_1based,
            filled_count: 0,
        });
    }
    if real_mesh_count == 0 || real_mesh_count > energy_count {
        return Err(XsphError::InvalidRealEnergyCount {
            real_mesh_count,
            energy_count,
        });
    }

    let reference = reference_energies[real_mesh_count - 1];
    let start_index = start_index_1based - 1;
    for index in start_index..energy_count {
        reference_energies[index] = reference;
    }

    Ok(XsphPhaseReferenceTail {
        start_index_1based,
        filled_count: energy_count - start_index,
    })
}

/// Port of the FEFF `XSPH/phase.f90` `imt`/`jri` radial-index setup.
///
/// FEFF computes `imt = (log(rmt) + x0) / dx + 1` by assigning the real value
/// to an integer, then sets `jri = imt + 1` and `jri1 = jri + 1`. The real to
/// integer assignment truncates toward zero.
pub fn xsph_phase_radial_indices(
    input: XsphPhaseRadialIndicesInput,
) -> Result<XsphPhaseRadialIndices, XsphError> {
    validate_phase_radial_indices_input(input)?;

    let raw_muffin_tin_index =
        (input.muffin_tin_radius.ln() + input.grid_origin) / input.log_step + 1.0;
    let muffin_tin_index =
        phase_real_to_integer_assignment("muffin_tin_index", raw_muffin_tin_index)?;
    let radial_match_index =
        muffin_tin_index
            .checked_add(1)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "radial_match_index_1based",
                value: muffin_tin_index,
            })?;
    if radial_match_index <= 0 {
        return Err(XsphError::IntegerOutOfRange {
            name: "radial_match_index_1based",
            value: radial_match_index,
        });
    }
    let reference_index =
        radial_match_index
            .checked_add(1)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "reference_index_1based",
                value: radial_match_index,
            })?;
    let reference_index_1based =
        usize::try_from(reference_index).map_err(|_| XsphError::IntegerOutOfRange {
            name: "reference_index_1based",
            value: reference_index,
        })?;
    if reference_index_1based > input.radial_capacity {
        return Err(XsphError::LengthTooShort {
            name: "radial_grid",
            required: reference_index_1based,
            actual: input.radial_capacity,
        });
    }

    Ok(XsphPhaseRadialIndices {
        raw_muffin_tin_index,
        muffin_tin_index,
        radial_match_index_1based: usize::try_from(radial_match_index).map_err(|_| {
            XsphError::IntegerOutOfRange {
                name: "radial_match_index_1based",
                value: radial_match_index,
            }
        })?,
        reference_index_1based,
    })
}

/// Port of the FEFF `XSPH/phase.f90` `mpse.dat` self-energy summary row.
///
/// FEFF samples `edens(jri+1)`, writes `rs = (3/(4*pi*edens))**third`, and
/// writes the corresponding plasma frequency `sqrt(3/rs**3)*hart` in eV.
pub fn xsph_phase_self_energy_summary(
    input: XsphPhaseSelfEnergySummaryInput<'_>,
) -> Result<XsphPhaseSelfEnergySummary, XsphError> {
    let density = phase_reference_density(input)?;
    let wigner_seitz_radius = (3.0 / (4.0 * std::f64::consts::PI * density)).powf(1.0 / 3.0);
    let plasma_frequency_ev = (3.0 / wigner_seitz_radius.powi(3)).sqrt() * XSPH_HARTREE_EV;
    validate_finite_real("wigner_seitz_radius", wigner_seitz_radius)?;
    validate_finite_real("plasma_frequency_ev", plasma_frequency_ev)?;

    Ok(XsphPhaseSelfEnergySummary {
        electron_density: density,
        wigner_seitz_radius,
        plasma_frequency_ev,
    })
}

/// Port of the FEFF `XSPH/phase.f90` MPSE pole setup after `MkExc`.
///
/// When `iPl > 0` and `ixc == 0`, FEFF reads `loss.dat`, calls `MkExc`, then
/// converts pole widths from eV to Hartree and stores pole energies as
/// Hartree pole energies divided by the local density-derived plasma
/// frequency. This helper takes the already-generated `MkExc` pole rows as its
/// input so the `phase` code can stay focused on its own scaling branch.
pub fn xsph_phase_plasmon_pole_setup(
    input: XsphPhasePlasmonPoleSetupInput<'_>,
) -> Result<Option<XsphPhasePlasmonPoleSetup>, XsphError> {
    if input.plasmon_selector <= 0 || input.exchange_selector != 0 {
        return Ok(None);
    }
    if input.excitation_poles.is_empty() {
        return Err(XsphError::EmptyIndexSet);
    }

    let summary = xsph_phase_self_energy_summary(XsphPhaseSelfEnergySummaryInput {
        electron_density: input.electron_density,
        reference_index_1based: input.reference_index_1based,
    })?;
    let plasma_frequency_hartree = summary.plasma_frequency_ev / XSPH_HARTREE_EV;
    validate_finite_real("plasma_frequency_hartree", plasma_frequency_hartree)?;
    if plasma_frequency_hartree <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "plasma_frequency_hartree",
            value: plasma_frequency_hartree,
        });
    }

    let mut poles = Vec::with_capacity(input.excitation_poles.len());
    for (index, pole) in input.excitation_poles.iter().enumerate() {
        validate_phase_plasmon_pole(index, pole)?;
        let energy_over_plasma = (pole.energy / XSPH_HARTREE_EV) / plasma_frequency_hartree;
        let width_hartree = pole.width / XSPH_HARTREE_EV;
        validate_finite_real("plasmon_energy_over_plasma", energy_over_plasma)?;
        validate_finite_real("plasmon_width_hartree", width_hartree)?;
        poles.push(XsphPhasePlasmonPole {
            energy_over_plasma,
            width_hartree,
            amplitude: pole.amplitude,
        });
    }

    Ok(Some(XsphPhasePlasmonPoleSetup {
        electron_density: summary.electron_density,
        wigner_seitz_radius: summary.wigner_seitz_radius,
        plasma_frequency_hartree,
        plasma_frequency_ev: summary.plasma_frequency_ev,
        poles,
    }))
}

/// Port of FEFF `XSPH/phase.f90` `PrintRl` `rl.dat` header emission.
///
/// FEFF writes the header only for the absorber potential (`iph == 0`) when the
/// `PrintRl` flag is set. The two header rows carry `rmt`, `lmax`, `jri`, `dx`,
/// and `x0` unchanged.
pub fn xsph_phase_radial_header(
    input: XsphPhaseRadialHeaderInput,
) -> Result<Option<XsphPhaseRadialHeader>, XsphError> {
    if !input.print_radial || input.potential_index != 0 {
        return Ok(None);
    }
    validate_phase_radial_header_input(input)?;

    Ok(Some(XsphPhaseRadialHeader {
        muffin_tin_radius: input.muffin_tin_radius,
        angular_limit: input.angular_limit,
        radial_match_index_1based: input.radial_match_index_1based,
        log_step: input.log_step,
        grid_origin: input.grid_origin,
    }))
}

/// Port of FEFF `XSPH/phase_h.f90` Hubbard `Vnlm` potential shifts.
///
/// For each magnetic channel `imm = ll**2 + 1 .. (ll + 1)**2`, FEFF creates
/// temporary total and valence potential arrays by adding `abs(Vnlm(ll, imm))`
/// for `is_p = 2` or subtracting it for `is_p = 1`.
pub fn xsph_hubbard_phase_potential_shifts(
    input: XsphHubbardPhasePotentialInput<'_>,
) -> Result<Vec<XsphHubbardPhasePotentialShift>, XsphError> {
    validate_hubbard_phase_potential_input(&input)?;

    let angular_channel =
        usize::try_from(input.angular_channel).map_err(|_| XsphError::NegativeAngularMomentum {
            name: "angular_channel",
            index: 0,
            value: input.angular_channel,
        })?;
    let (first_magnetic, last_exclusive) = hubbard_magnetic_channel_range(angular_channel)?;
    validate_hubbard_potential_shape(input.hubbard_potential, angular_channel, last_exclusive)?;

    let sign = match input.spin_projection {
        1 => -1.0,
        2 => 1.0,
        spin_projection => {
            return Err(XsphError::InvalidHubbardSpinProjection { spin_projection });
        }
    };

    let mut shifts = Vec::with_capacity(last_exclusive - first_magnetic);
    for magnetic_channel in first_magnetic..last_exclusive {
        let raw_shift = input.hubbard_potential[(angular_channel, magnetic_channel)];
        validate_finite_real("hubbard_potential", raw_shift)?;
        let shift = sign * raw_shift.abs();
        let total_potential =
            shifted_phase_potential(input.total_potential, input.active_len, shift);
        let valence_potential =
            shifted_phase_potential(input.valence_potential, input.active_len, shift);
        shifts.push(XsphHubbardPhasePotentialShift {
            magnetic_channel,
            shift,
            total_potential,
            valence_potential,
        });
    }

    Ok(shifts)
}

/// Port of FEFF `XSPH/phase_h.f90` Hubbard `aph` workspace assignments.
///
/// FEFF only enters this branch for `0 <= ll <= lx`, then stores each
/// per-`imm` perturbed phase shift into `aph(ie,ll+1,imm)`. Rust keeps
/// `energy_index`, `angular_channel`, and `magnetic_channel` zero-based, so
/// FEFF `imm` is represented as `magnetic_channel + 1`.
pub fn xsph_hubbard_phase_assignments(
    input: XsphHubbardPhaseAssignmentInput<'_>,
) -> Result<Vec<XsphHubbardPhaseAssignment>, XsphError> {
    usize_to_i32("energy_index", input.energy_index)?;
    usize_to_i32("hubbard_angular_limit", input.hubbard_angular_limit)?;

    if input.angular_channel < 0 {
        return Ok(Vec::new());
    }
    let angular_channel =
        usize::try_from(input.angular_channel).map_err(|_| XsphError::IntegerOutOfRange {
            name: "angular_channel",
            value: input.angular_channel,
        })?;
    if angular_channel > input.hubbard_angular_limit {
        return Ok(Vec::new());
    }

    let (first_magnetic, last_exclusive) = hubbard_magnetic_channel_range(angular_channel)?;
    let magnetic_count = last_exclusive - first_magnetic;
    validate_active_len(
        "hubbard_phase_shifts",
        input.magnetic_phase_shifts.len(),
        magnetic_count,
    )?;

    let mut assignments = Vec::with_capacity(magnetic_count);
    for (offset, magnetic_channel) in (first_magnetic..last_exclusive).enumerate() {
        let phase_shift = input.magnetic_phase_shifts[offset];
        validate_finite_complex("hubbard_phase_shift", offset, phase_shift)?;
        assignments.push(XsphHubbardPhaseAssignment {
            energy_index: input.energy_index,
            angular_channel,
            magnetic_channel,
            phase_shift,
        });
    }

    Ok(assignments)
}

/// Port of the empty-cell phase branch in FEFF `XSPH/phase.f90`.
///
/// When `iz == 0`, FEFF does not call `dfovrg`; instead it builds `pu` and
/// `qu` directly from the empty-cell Bessel solution at `ckEC*rmt`, then calls
/// `phamp` against the active-potential Bessel/Neumann values at `ck*rmt`.
pub fn xsph_empty_cell_phase(
    input: XsphEmptyCellPhaseInput,
) -> Result<XsphEmptyCellPhase, XsphError> {
    validate_empty_cell_phase_input(input)?;

    let large_l = l_from_kappa(input.kappa)?;
    let small_l = small_l_from_kappa(input.kappa, large_l)?;
    let max_l = large_l.max(small_l);
    let active_argument = input.wave_number * input.muffin_tin_radius;
    let empty_argument = input.empty_cell_wave_number * input.muffin_tin_radius;
    validate_finite_complex("phase_active_argument", 0, active_argument)?;
    validate_finite_complex("phase_empty_cell_argument", 0, empty_argument)?;

    let active = besjn(active_argument, max_l)?;
    let empty_cell = besjn(empty_argument, max_l)?;
    let sign = if input.kappa < 0 { -1.0 } else { 1.0 };
    let scaled_wave_number = input.wave_number * XSPH_FINE_STRUCTURE_ALPHA;
    let one = Complex::new(1.0, 0.0);
    let small_component_factor =
        sign * scaled_wave_number / (one + (one + scaled_wave_number * scaled_wave_number).sqrt());
    validate_finite_complex("phase_small_component_factor", 0, small_component_factor)?;

    let regular_large_at_muffin_tin = empty_cell.j[large_l] * input.muffin_tin_radius;
    let regular_small_at_muffin_tin =
        empty_cell.j[small_l] * input.muffin_tin_radius * small_component_factor;
    let amplitude_phase = muffin_tin_phase_amplitude(
        input.muffin_tin_radius,
        regular_large_at_muffin_tin,
        regular_small_at_muffin_tin,
        input.wave_number,
        active.j[large_l],
        active.y[large_l],
        active.j[small_l],
        active.y[small_l],
        input.kappa,
    )?;

    Ok(XsphEmptyCellPhase {
        phase_shift: amplitude_phase.phase,
        phase_amplitude: amplitude_phase.amplitude,
        regular_large_at_muffin_tin,
        regular_small_at_muffin_tin,
        large_l,
        small_l,
        bessel_j_large: active.j[large_l],
        neumann_large: active.y[large_l],
        bessel_j_small: active.j[small_l],
        neumann_small: active.y[small_l],
        empty_cell_bessel_j_large: empty_cell.j[large_l],
        empty_cell_bessel_j_small: empty_cell.j[small_l],
    })
}

/// Port of the normal-potential phase match in FEFF `XSPH/phase.f90`.
///
/// FEFF obtains `pu` and `qu` from `dfovrg`, evaluates the active-potential
/// Bessel/Neumann arrays at `ck*rmt`, and calls `phamp` to recover the
/// scattering phase and amplitude. This helper covers that post-`dfovrg`
/// matching block while leaving the radial solve to the caller.
pub fn xsph_regular_phase(input: XsphRegularPhaseInput) -> Result<XsphRegularPhase, XsphError> {
    validate_regular_phase_input(input)?;

    let large_l = l_from_kappa(input.kappa)?;
    let small_l = small_l_from_kappa(input.kappa, large_l)?;
    let max_l = large_l.max(small_l);
    let active_argument = input.wave_number * input.muffin_tin_radius;
    validate_finite_complex("phase_active_argument", 0, active_argument)?;

    let active = besjn(active_argument, max_l)?;
    let amplitude_phase = muffin_tin_phase_amplitude(
        input.muffin_tin_radius,
        input.regular_large_at_muffin_tin,
        input.regular_small_at_muffin_tin,
        input.wave_number,
        active.j[large_l],
        active.y[large_l],
        active.j[small_l],
        active.y[small_l],
        input.kappa,
    )?;

    Ok(XsphRegularPhase {
        phase_shift: amplitude_phase.phase,
        phase_amplitude: amplitude_phase.amplitude,
        regular_large_at_muffin_tin: input.regular_large_at_muffin_tin,
        regular_small_at_muffin_tin: input.regular_small_at_muffin_tin,
        large_l,
        small_l,
        bessel_j_large: active.j[large_l],
        neumann_large: active.y[large_l],
        bessel_j_small: active.j[small_l],
        neumann_small: active.y[small_l],
    })
}

/// Run the regular FOVRG branch for one normal-potential XSPH phase channel.
///
/// This mirrors the `iz > 0` branch in FEFF `XSPH/phase.f90`: solve the
/// regular radial Dirac equation with `dfovrg`, then call the same phase-match
/// helper used by the empty-cell path with the solver's muffin-tin `pu`/`qu`
/// values. The caller supplies the already prepared, energy-referenced FOVRG
/// input and the corresponding relativistic wave number.
pub fn xsph_regular_phase_channel(
    solver: FovrgDiracSolverInput<'_>,
    wave_number: Complex,
) -> Result<XsphRegularPhaseChannel, XsphError> {
    validate_finite_complex("regular_phase_channel_wave_number", 0, wave_number)?;

    let zero = Complex::new(0.0, 0.0);
    let regular_input = FovrgDiracSolverInput {
        irregular: false,
        muffin_tin_large_component: zero,
        muffin_tin_small_component: zero,
        ..solver
    };
    let regular_solution = fovrg_dirac_solver(regular_input)?;
    let phase = xsph_regular_phase(XsphRegularPhaseInput {
        muffin_tin_radius: solver.muffin_tin_radius,
        wave_number,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: solver.target_kappa,
    })?;

    Ok(XsphRegularPhaseChannel {
        regular_solution,
        phase,
    })
}

/// Port of FEFF `XSPH/phase.f90` `PrintRl` radial-output normalization.
///
/// FEFF writes only absorber (`iph == 0`) channels with `ll <= 0` and
/// `abs(ll) <= lmax`. For those rows it divides `p(1:jri)` and `q(1:jri)` by
/// the `phamp` amplitude `temp` before writing the `rl.dat` arrays.
pub fn xsph_phase_radial_output(
    input: XsphPhaseRadialOutputInput<'_>,
) -> Result<Option<XsphPhaseRadialOutput>, XsphError> {
    if !input.print_radial || input.potential_index != 0 || input.angular_channel > 0 {
        return Ok(None);
    }

    let output_angular_momentum = phase_abs_angular_channel(input.angular_channel)?;
    if output_angular_momentum > input.angular_limit {
        return Ok(None);
    }

    validate_phase_radial_output_input(&input)?;
    let mut regular_large = Vec::with_capacity(input.active_len);
    let mut regular_small = Vec::with_capacity(input.active_len);
    for index in 0..input.active_len {
        let large = input.regular_large[index] / input.phase_amplitude;
        let small = input.regular_small[index] / input.phase_amplitude;
        validate_finite_complex("phase_radial_regular_large", index, large)?;
        validate_finite_complex("phase_radial_regular_small", index, small)?;
        regular_large.push(large);
        regular_small.push(small);
    }

    Ok(Some(XsphPhaseRadialOutput {
        energy: input.energy,
        angular_channel: input.angular_channel,
        output_angular_momentum,
        phase_shift: input.phase_shift,
        regular_large: Array1::from_vec(regular_large),
        regular_small: Array1::from_vec(regular_small),
    }))
}

fn validate_phase_energy_setup_input(input: XsphPhaseEnergySetupInput) -> Result<(), XsphError> {
    validate_finite_complex("energy", input.energy_index, input.energy)?;
    validate_finite_complex(
        "reference_energy",
        input.energy_index,
        input.reference_energy,
    )?;
    validate_finite_real("muffin_tin_potential", input.muffin_tin_potential)?;
    validate_finite_real("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    usize_to_i32("energy_index", input.energy_index)?;
    usize_to_i32("real_mesh_count", input.real_mesh_count)?;
    Ok(())
}

fn validate_phase_grid_preparation_input(
    input: &XsphPhaseGridPreparationInput<'_>,
) -> Result<(), XsphError> {
    if input.radial_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "radial_count",
            required: 1,
            actual: 0,
        });
    }
    validate_finite_real(
        "phase_grid_interstitial_potential",
        input.interstitial_potential,
    )?;
    validate_finite_real(
        "phase_grid_interstitial_density",
        input.interstitial_density,
    )?;
    validate_finite_real("phase_grid_original_radial_dx", input.original_radial_dx)?;
    if input.original_radial_dx <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "phase_grid_original_radial_dx",
            value: input.original_radial_dx,
        });
    }
    validate_finite_real("phase_grid_target_radial_dx", input.target_radial_dx)?;
    if input.target_radial_dx <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "phase_grid_target_radial_dx",
            value: input.target_radial_dx,
        });
    }
    validate_finite_real("phase_grid_potential_jump", input.potential_jump)?;

    let (source_radial, potential_count) = input.total_potential.dim();
    if potential_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "phase_grid_potential_count",
            required: 1,
            actual: 0,
        });
    }
    if input.muffin_tin_radii.len() != potential_count {
        return Err(XsphError::LengthTooShort {
            name: "muffin_tin_radii",
            required: potential_count,
            actual: input.muffin_tin_radii.len(),
        });
    }
    validate_phase_grid_matrix_shape(
        "electron_density",
        source_radial,
        potential_count,
        input.electron_density.dim(),
    )?;
    validate_phase_grid_matrix_shape(
        "valence_density",
        source_radial,
        potential_count,
        input.valence_density.dim(),
    )?;
    validate_phase_grid_matrix_shape(
        "valence_potential",
        source_radial,
        potential_count,
        input.valence_potential.dim(),
    )?;
    validate_phase_grid_matrix_shape(
        "magnetization",
        source_radial,
        potential_count,
        input.magnetization.dim(),
    )?;

    let (large_radial, large_orbital, large_potential) = input.bound_large_components.dim();
    let (small_radial, small_orbital, small_potential) = input.bound_small_components.dim();
    if large_radial == 0 || large_orbital == 0 || large_potential != potential_count {
        return Err(XsphError::ShapeTooSmall {
            name: "bound_large_components",
            required: [1, 1, potential_count],
            actual: [large_radial, large_orbital, large_potential],
        });
    }
    if small_radial != large_radial
        || small_orbital != large_orbital
        || small_potential != large_potential
    {
        return Err(XsphError::ShapeTooSmall {
            name: "bound_small_components",
            required: [large_radial, large_orbital, large_potential],
            actual: [small_radial, small_orbital, small_potential],
        });
    }
    for (potential, &radius) in input.muffin_tin_radii.iter().enumerate() {
        validate_finite_real("phase_grid_muffin_tin_radius", radius)?;
        if radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "phase_grid_muffin_tin_radius",
                value: radius,
            });
        }
        usize_to_i32("phase_grid_potential_index", potential)?;
    }
    Ok(())
}

fn validate_phase_grid_matrix_shape(
    name: &'static str,
    expected_radial: usize,
    expected_potentials: usize,
    actual: (usize, usize),
) -> Result<(), XsphError> {
    if actual.0 != expected_radial || actual.1 != expected_potentials {
        return Err(XsphError::MatrixTooSmall {
            name,
            required: [expected_radial, expected_potentials],
            actual: [actual.0, actual.1],
        });
    }
    Ok(())
}

fn validate_phase_radial_indices_input(
    input: XsphPhaseRadialIndicesInput,
) -> Result<(), XsphError> {
    validate_finite_real("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    validate_finite_real("grid_origin", input.grid_origin)?;
    validate_finite_real("log_step", input.log_step)?;
    if input.log_step <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "log_step",
            value: input.log_step,
        });
    }
    usize_to_i32("radial_capacity", input.radial_capacity)?;
    Ok(())
}

fn phase_real_to_integer_assignment(name: &'static str, value: f64) -> Result<i32, XsphError> {
    if !value.is_finite() || value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err(XsphError::RealIntegerOutOfRange { name, value });
    }
    Ok(value.trunc() as i32)
}

fn phase_reference_density(input: XsphPhaseSelfEnergySummaryInput<'_>) -> Result<f64, XsphError> {
    if input.reference_index_1based == 0 {
        return Err(XsphError::InvalidPhaseRadialReferenceIndex {
            index_1based: input.reference_index_1based,
        });
    }
    if input.electron_density.len() < input.reference_index_1based {
        return Err(XsphError::LengthTooShort {
            name: "electron_density",
            required: input.reference_index_1based,
            actual: input.electron_density.len(),
        });
    }
    let density = input.electron_density[input.reference_index_1based - 1];
    validate_finite_real("electron_density", density)?;
    if density <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "electron_density",
            value: density,
        });
    }
    Ok(density)
}

fn validate_phase_plasmon_pole(
    index: usize,
    pole: &crate::ExcitationPole,
) -> Result<(), XsphError> {
    validate_finite_real("plasmon_pole_energy", pole.energy)?;
    if pole.energy <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "plasmon_pole_energy",
            value: pole.energy,
        });
    }
    validate_finite_real("plasmon_pole_width", pole.width)?;
    if pole.width <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "plasmon_pole_width",
            value: pole.width,
        });
    }
    validate_finite_real("plasmon_pole_amplitude", pole.amplitude)?;
    usize_to_i32("plasmon_pole_index", index)?;
    Ok(())
}

fn validate_phase_radial_header_input(input: XsphPhaseRadialHeaderInput) -> Result<(), XsphError> {
    validate_finite_real("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    usize_to_i32("angular_limit", input.angular_limit)?;
    if input.radial_match_index_1based == 0 {
        return Err(XsphError::InvalidPhaseRadialMatchIndex {
            index_1based: input.radial_match_index_1based,
        });
    }
    usize_to_i32("radial_match_index_1based", input.radial_match_index_1based)?;
    validate_finite_real("log_step", input.log_step)?;
    if input.log_step <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "log_step",
            value: input.log_step,
        });
    }
    validate_finite_real("grid_origin", input.grid_origin)
}

fn validate_phase_energy_dynamics(
    momentum_squared: Complex,
    empty_cell_momentum_squared: Complex,
    wave_number: Complex,
    empty_cell_wave_number: Complex,
    muffin_tin_argument: Complex,
    empty_cell_muffin_tin_argument: Complex,
) -> Result<(), XsphError> {
    validate_finite_complex("phase_momentum_squared", 0, momentum_squared)?;
    validate_finite_complex(
        "phase_empty_cell_momentum_squared",
        0,
        empty_cell_momentum_squared,
    )?;
    validate_finite_complex("phase_wave_number", 0, wave_number)?;
    validate_finite_complex("phase_empty_cell_wave_number", 0, empty_cell_wave_number)?;
    validate_finite_complex("phase_muffin_tin_argument", 0, muffin_tin_argument)?;
    validate_finite_complex(
        "phase_empty_cell_muffin_tin_argument",
        0,
        empty_cell_muffin_tin_argument,
    )
}

fn phase_relativistic_wave_number(momentum_squared: Complex) -> Complex {
    (2.0 * momentum_squared
        + (momentum_squared * XSPH_FINE_STRUCTURE_ALPHA)
            * (momentum_squared * XSPH_FINE_STRUCTURE_ALPHA))
        .sqrt()
}

fn phase_cycle_count(exchange_selector: i32) -> usize {
    if exchange_selector % 10 < 5 { 0 } else { 3 }
}

fn validate_hubbard_phase_potential_input(
    input: &XsphHubbardPhasePotentialInput<'_>,
) -> Result<(), XsphError> {
    if input.angular_channel < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "angular_channel",
            index: 0,
            value: input.angular_channel,
        });
    }
    validate_active_len(
        "total_potential",
        input.total_potential.len(),
        input.active_len,
    )?;
    validate_active_len(
        "valence_potential",
        input.valence_potential.len(),
        input.active_len,
    )?;
    for index in 0..input.active_len {
        validate_finite_complex("total_potential", index, input.total_potential[index])?;
        validate_finite_complex("valence_potential", index, input.valence_potential[index])?;
    }
    Ok(())
}

fn hubbard_magnetic_channel_range(angular_channel: usize) -> Result<(usize, usize), XsphError> {
    let first_magnetic =
        angular_channel
            .checked_mul(angular_channel)
            .ok_or(XsphError::SizeOutOfRange {
                name: "magnetic_channel",
                value: angular_channel,
            })?;
    let last_exclusive = angular_channel
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(XsphError::SizeOutOfRange {
            name: "magnetic_channel",
            value: angular_channel,
        })?;
    Ok((first_magnetic, last_exclusive))
}

fn validate_hubbard_potential_shape(
    hubbard_potential: ArrayView2<'_, f64>,
    angular_channel: usize,
    magnetic_channel_count: usize,
) -> Result<(), XsphError> {
    let actual = hubbard_potential.shape();
    let required = [angular_channel + 1, magnetic_channel_count];
    if actual[0] < required[0] || actual[1] < required[1] {
        return Err(XsphError::MatrixTooSmall {
            name: "hubbard_potential",
            required,
            actual: [actual[0], actual[1]],
        });
    }
    Ok(())
}

fn shifted_phase_potential(
    potential: ArrayView1<'_, Complex>,
    active_len: usize,
    shift: f64,
) -> Array1<Complex> {
    let shift = Complex::new(shift, 0.0);
    Array1::from_iter((0..active_len).map(|index| potential[index] + shift))
}

fn validate_phase_radial_output_input(
    input: &XsphPhaseRadialOutputInput<'_>,
) -> Result<(), XsphError> {
    usize_to_i32("angular_limit", input.angular_limit)?;
    validate_active_len("regular_large", input.regular_large.len(), input.active_len)?;
    validate_active_len("regular_small", input.regular_small.len(), input.active_len)?;
    validate_finite_complex("phase_radial_energy", 0, input.energy)?;
    validate_finite_complex("phase_shift", 0, input.phase_shift)?;
    validate_finite_complex("phase_amplitude", 0, input.phase_amplitude)?;
    if input.phase_amplitude == Complex::new(0.0, 0.0) {
        return Err(XsphError::ZeroPhaseAmplitude);
    }
    for index in 0..input.active_len {
        validate_finite_complex("regular_large", index, input.regular_large[index])?;
        validate_finite_complex("regular_small", index, input.regular_small[index])?;
    }
    Ok(())
}

fn phase_abs_angular_channel(angular_channel: i32) -> Result<usize, XsphError> {
    let abs_channel = angular_channel
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "angular_channel",
            value: angular_channel,
        })?;
    usize::try_from(abs_channel).map_err(|_| XsphError::IntegerOutOfRange {
        name: "angular_channel",
        value: angular_channel,
    })
}

fn validate_phase_channel_plan_input(input: XsphPhaseChannelPlanInput) -> Result<(), XsphError> {
    usize_to_i32("angular_limit", input.angular_limit)?;
    usize_to_i32("initial_cycle_count", input.initial_cycle_count)?;
    validate_finite_real("log_step", input.log_step)
}

fn phase_channel_kappa(angular_channel: i32) -> Result<i32, XsphError> {
    if angular_channel > 0 {
        Ok(angular_channel)
    } else {
        angular_channel
            .checked_sub(1)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "angular_channel",
                value: angular_channel,
            })
    }
}

fn validate_empty_cell_phase_input(input: XsphEmptyCellPhaseInput) -> Result<(), XsphError> {
    validate_finite_real("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    validate_finite_complex("wave_number", 0, input.wave_number)?;
    validate_finite_complex("empty_cell_wave_number", 0, input.empty_cell_wave_number)?;
    if input.kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    Ok(())
}

fn validate_regular_phase_input(input: XsphRegularPhaseInput) -> Result<(), XsphError> {
    validate_finite_real("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    validate_finite_complex("wave_number", 0, input.wave_number)?;
    validate_finite_complex(
        "regular_large_at_muffin_tin",
        0,
        input.regular_large_at_muffin_tin,
    )?;
    validate_finite_complex(
        "regular_small_at_muffin_tin",
        0,
        input.regular_small_at_muffin_tin,
    )?;
    if input.kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    Ok(())
}

fn validate_phase_angular_limit_input(
    input: &XsphPhaseAngularLimitInput<'_>,
) -> Result<(), XsphError> {
    validate_finite_real("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    usize_to_i32("max_angular_momentum", input.max_angular_momentum)?;
    if input.energy_count == 0 {
        return Err(XsphError::EmptyPhaseMesh);
    }
    if input.auxiliary_count > input.energy_count {
        return Err(XsphError::InvalidAuxiliaryEnergyCount {
            auxiliary_count: input.auxiliary_count,
            energy_count: input.energy_count,
        });
    }
    if input.energies.len() < input.energy_count {
        return Err(XsphError::LengthTooShort {
            name: "energies",
            required: input.energy_count,
            actual: input.energies.len(),
        });
    }
    validate_phase_energy_prefix(input.energies, input.energy_count)
}

fn validate_phase_energy_prefix(
    energies: ArrayView1<'_, Complex>,
    energy_count: usize,
) -> Result<(), XsphError> {
    for index in 0..energy_count {
        validate_finite_complex("energies", index, energies[index])?;
    }
    Ok(())
}

fn phase_nonnegative_integer_assignment(value: f64) -> usize {
    let nearest = value.round();
    if (value - nearest).abs() <= 1.0e-12 {
        nearest as usize
    } else {
        value as usize
    }
}

fn l_from_kappa(kappa: i32) -> Result<usize, XsphError> {
    let value = if kappa > 0 {
        kappa
    } else {
        kappa
            .checked_neg()
            .and_then(|value| value.checked_sub(1))
            .ok_or(XsphError::IntegerOutOfRange {
                name: "kappa",
                value: kappa,
            })?
    };
    usize::try_from(value).map_err(|_| XsphError::IntegerOutOfRange {
        name: "kappa",
        value: kappa,
    })
}

fn small_l_from_kappa(kappa: i32, large_l: usize) -> Result<usize, XsphError> {
    if kappa > 0 {
        large_l.checked_sub(1).ok_or(XsphError::IntegerOutOfRange {
            name: "kappa",
            value: kappa,
        })
    } else {
        large_l.checked_add(1).ok_or(XsphError::SizeOutOfRange {
            name: "small_l",
            value: large_l,
        })
    }
}
