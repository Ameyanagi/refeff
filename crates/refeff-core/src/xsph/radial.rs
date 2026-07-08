//! FEFF XSPH radial matrix-element integrals.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use refeff_linalg::{complex_lu_factor, complex_lu_solve};

use crate::{
    Complex, Real, besjh, csomm, csomm2, csommjas, fovrg_dirac_solver, fovrg_initial_photoelectron,
    somm, somm2, wigner_3j,
};

use super::{
    XsphError, XsphJasOrthogonalityCorrection, XsphJasOrthogonalityCorrectionInput, XsphJasOverlap,
    XsphJasOverlapInput, XsphJasRadialCrossIntegral, XsphJasRadialCrossIntegralInput,
    XsphJasRadialIntegral, XsphJasRadialIntegralInput, XsphRadialCrossIntegral,
    XsphRadialCrossIntegralBranch, XsphRadialCrossIntegralInput, XsphRadialIntegral,
    XsphRadialIntegralInput, XsphRadialIntegralMode, XsphRegularPhaseInput,
    XsphRelativisticMultipoleFactors, XsphTransitionMultipole, XsphXrayBesselTable,
    XsphXrayBesselTableInput, XsphXsectBcoefCentralCrossSectionInput,
    XsphXsectBcoefCrossTermAccumulationInput, XsphXsectBcoefCrossTermStateAccumulationInput,
    XsphXsectBcoefDirectTransitionInput, XsphXsectBcoefDirectTransitionUpdate,
    XsphXsectBcoefDirectTransitionUpdateInput, XsphXsectBcoefNonstandardChannelRow,
    XsphXsectBcoefNonstandardChannelRowInput, XsphXsectBcoefNonstandardEnergyRow,
    XsphXsectBcoefNonstandardEnergyRowInput, XsphXsectBcoefOrdinaryRow,
    XsphXsectBcoefOrdinaryRowInput, XsphXsectBcoefStandardChannelRow,
    XsphXsectBcoefStandardChannelRowInput, XsphXsectBcoefStandardEnergyRow,
    XsphXsectBcoefStandardEnergyRowFieldsInput, XsphXsectBcoefStandardEnergyRowInput,
    XsphXsectBcoefStandardTransitionField, XsphXsectCentralCrossSection,
    XsphXsectCentralCrossSectionInput, XsphXsectCrossTermAccumulation,
    XsphXsectCrossTermAccumulationInput, XsphXsectCrossTermMode, XsphXsectCrossTermPlan,
    XsphXsectCrossTermPlanInput, XsphXsectCrossTermState, XsphXsectCrossTermStateReuse,
    XsphXsectCrossTermStateReuseInput, XsphXsectCrossTermStateSaveInput, XsphXsectDensityBranch,
    XsphXsectDensityBranchInput, XsphXsectDirectTransition, XsphXsectDirectTransitionInput,
    XsphXsectEmbeddedDensity, XsphXsectEmbeddedDensityInput, XsphXsectEnergyDecision,
    XsphXsectEnergySetup, XsphXsectEnergySetupInput, XsphXsectFscfComponentPart,
    XsphXsectFscfIntegral, XsphXsectFscfIntegralInput, XsphXsectFscfSelection, XsphXsectFscfWeight,
    XsphXsectFscfWeights, XsphXsectFscfWeightsInput, XsphXsectHoleNormalization,
    XsphXsectHoleNormalizationInput, XsphXsectIrregularChannel, XsphXsectIrregularChannelInput,
    XsphXsectIrregularInitialCondition, XsphXsectIrregularInitialConditionInput,
    XsphXsectIrregularTransform, XsphXsectIrregularTransformInput, XsphXsectOutputNormalization,
    XsphXsectOutputNormalizationInput, XsphXsectPhiscfAccumulatedResponse,
    XsphXsectPhiscfAccumulatedResponseInput, XsphXsectPhiscfAngularChannels,
    XsphXsectPhiscfContributionPlan, XsphXsectPhiscfContributionPlanInput,
    XsphXsectPhiscfContributionPlanRow, XsphXsectPhiscfContributionRule,
    XsphXsectPhiscfContributionRuleInput, XsphXsectPhiscfFieldAssemblyInput, XsphXsectPhiscfFields,
    XsphXsectPhiscfIrregularSeed, XsphXsectPhiscfIrregularSeedInput, XsphXsectPhiscfLinearSolve,
    XsphXsectPhiscfLinearSolveInput, XsphXsectPhiscfLipman, XsphXsectPhiscfLipmanInput,
    XsphXsectPhiscfLocalField, XsphXsectPhiscfLocalFieldInput, XsphXsectPhiscfPoleEnergy,
    XsphXsectPhiscfPoleEnergyInput, XsphXsectPhiscfRadialContribution,
    XsphXsectPhiscfRadialContributionInput, XsphXsectPhiscfRadialSolverSetup,
    XsphXsectPhiscfRadialSolverSetupInput, XsphXsectPhiscfResponseContributionInput,
    XsphXsectPhiscfScreenedContributionsInput, XsphXsectPhiscfScreenedSolution,
    XsphXsectPhiscfScreenedSolutionInput, XsphXsectPhiscfWfirdcContribution,
    XsphXsectPhiscfWfirdcContributionInput, XsphXsectPhiscfWfirdcContributions,
    XsphXsectPhiscfWfirdcContributionsInput, XsphXsectPhiscfWorkspace, XsphXsectProjectedDensity,
    XsphXsectProjectedDensityInput, XsphXsectRadialPass, XsphXsectRadialPassInput,
    XsphXsectRadialPassKind, XsphXsectRegularChannel, XsphXsectRegularChannelInput,
    XsphXsectRegularSolution, XsphXsectRegularSolutionInput, XsphXsectScreenedField,
    XsphXsectScreenedFieldInput, XsphXsectScreenedFieldMode, XsphXsectTransition,
    XsphXsectTransitionPlan, XsphXsectTransitionPlanInput, XsphXsectWeightedRadialCrossIntegral,
    XsphXsectWeightedRadialCrossIntegralInput, XsphXsectWeightedRadialIntegral,
    XsphXsectWeightedRadialIntegralInput, doubled_j_from_kappa, validate_active_len,
    validate_cwig3j_doubled_argument, validate_cwig3j_integer_argument, validate_finite_complex,
    validate_finite_real, xsph_longitudinal_multipole_factor, xsph_regular_phase,
    xsph_relativistic_multipole_factors,
};

const RADINT_BESSEL_ROWS: usize = 3;
const XRAY_BESSEL_SERIES_TOLERANCE: Real = 1.0e-15;
const XRAY_BESSEL_SERIES_MAX_ITERATIONS: usize = 160;
const XSECT_BCOEF_TRANSITION_SLOTS: usize = 8;
const XSECT_OMEGA_FLOOR: Real = 0.001 / super::XSPH_HARTREE_EV;
const XSECT_HOLE_NORMALIZATION_TOLERANCE: Real = 1.0e-2;
const XSECT_NORMAN_TAIL_ROWS: usize = 7;
const XSECT_PHISCF_REGULAR_WFIRDC_COEFFICIENT_COUNT: usize = 3;
const XSECT_PHISCF_IRREGULAR_WFIRDC_COEFFICIENT_COUNT: usize = 2;

#[derive(Debug, Clone, Copy)]
struct RadintMultipoleFactors {
    xm1: Complex,
    xm2: Complex,
    xm3: Complex,
    xm4: Complex,
}

#[derive(Debug, Clone, Copy)]
struct RadialCouplingInput<'a> {
    mode: XsphRadialIntegralMode,
    multipole: XsphTransitionMultipole,
    initial_large: ArrayView1<'a, Real>,
    initial_small: ArrayView1<'a, Real>,
    final_large: ArrayView1<'a, Complex>,
    final_small: ArrayView1<'a, Complex>,
    xray_bessel: ArrayView2<'a, Real>,
    radii: ArrayView1<'a, Real>,
    active_len: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct XsectBcoefNonstandardRadialComponents {
    reduced_radial_pass: XsphXsectRadialPass,
    central_radial_pass: XsphXsectRadialPass,
    reduced_radial_integral: XsphRadialIntegral,
    central_cross_integral: XsphRadialCrossIntegral,
}

#[derive(Debug, Clone, PartialEq)]
struct XsectBcoefStandardRadialComponents {
    fscf_weights: XsphXsectFscfWeights,
    reduced_radial_pass: XsphXsectRadialPass,
    central_radial_pass: XsphXsectRadialPass,
    reduced_component_integrals: Vec<XsphXsectWeightedRadialIntegral>,
    reduced_fscf_integrals: Vec<XsphXsectFscfIntegral>,
    reduced_matrix_integral: Complex,
    central_component_integrals: Vec<XsphXsectWeightedRadialCrossIntegral>,
    central_fscf_integrals: Vec<XsphXsectFscfIntegral>,
    central_cross_integral: Complex,
}

/// Port of FEFF `XSPH/xsect.f90` initial hole-orbital normalization check.
///
/// Before the energy loop FEFF squares the core-hole spinor components,
/// integrates them through `jnrm` with `somm`, and logs a warning when
/// `abs(abs(xinorm) - 1)` exceeds `1e-2`. The warning is surfaced as data so a
/// future driver can preserve the diagnostic without treating it as fatal.
pub fn xsph_xsect_hole_normalization(
    input: XsphXsectHoleNormalizationInput<'_>,
) -> Result<XsphXsectHoleNormalization, XsphError> {
    validate_xsect_hole_normalization_input(&input)?;

    let active_len = input.norman_index_1based;
    let radii = active_real_prefix(input.radii, active_len);
    let small_squared = input
        .initial_small
        .iter()
        .take(active_len)
        .map(|&value| value * value)
        .collect::<Vec<_>>();
    let large_squared = input
        .initial_large
        .iter()
        .take(active_len)
        .map(|&value| value * value)
        .collect::<Vec<_>>();
    let near_origin_power = 2.0 * input.initial_l as Real + 2.0;
    validate_finite_real("xsect_hole_near_origin_power", near_origin_power)?;

    let normalization = somm(
        &radii,
        &small_squared,
        &large_squared,
        input.log_step,
        near_origin_power,
        0,
    )?;
    validate_finite_real("xsect_hole_normalization", normalization)?;
    let deviation = (normalization.abs() - 1.0).abs();
    validate_finite_real("xsect_hole_normalization_deviation", deviation)?;

    Ok(XsphXsectHoleNormalization {
        near_origin_power,
        normalization,
        deviation,
        warning_required: deviation > XSECT_HOLE_NORMALIZATION_TOLERANCE,
    })
}

/// Port of FEFF `XSPH/xsect.f90` per-energy setup before radial solves.
///
/// FEFF builds the self-energy-referenced momentum `p2`, relativistic wave
/// number `ck`, photon energy `omega` with a 0.001 eV floor, photon wave number
/// `xk0`, and the capped radial prefix `ilast` before deciding whether a row
/// proceeds to the transition loops.
pub fn xsph_xsect_energy_setup(
    input: XsphXsectEnergySetupInput,
) -> Result<XsphXsectEnergySetup, XsphError> {
    validate_xsect_energy_setup_input(input)?;

    let momentum_squared = input.energy - input.reference_energy;
    let edge_momentum_squared = input.edge_energy - input.reference_energy.re;
    let alpha_scaled = momentum_squared * super::XSPH_FINE_STRUCTURE_ALPHA;
    let wave_number = (2.0 * momentum_squared + alpha_scaled * alpha_scaled).sqrt();
    let muffin_tin_argument = input.muffin_tin_radius * wave_number;
    let cycle_count = if input.exchange_selector % 10 < 5 {
        0
    } else {
        3
    };

    let raw_photon_energy = input.energy.re - input.edge_energy + input.chemical_potential;
    let photon_energy = raw_photon_energy.max(XSECT_OMEGA_FLOOR);
    let photon_wave_number = photon_energy * super::XSPH_FINE_STRUCTURE_ALPHA;
    let norman_tail = input
        .norman_index_1based
        .checked_add(XSECT_NORMAN_TAIL_ROWS)
        .ok_or(XsphError::SizeOutOfRange {
            name: "xsect_norman_tail_index",
            value: input.norman_index_1based,
        })?;
    let active_radial_len = norman_tail
        .max(input.new_grid_index_1based)
        .min(input.radial_capacity);

    validate_finite_complex("xsect_momentum_squared", 0, momentum_squared)?;
    validate_finite_real("xsect_edge_momentum_squared", edge_momentum_squared)?;
    validate_finite_complex("xsect_wave_number", 0, wave_number)?;
    validate_finite_complex("xsect_muffin_tin_argument", 0, muffin_tin_argument)?;
    validate_finite_real("xsect_photon_energy", photon_energy)?;
    validate_finite_real("xsect_photon_wave_number", photon_wave_number)?;

    let decision = if input.energy.re < -10.0 {
        XsphXsectEnergyDecision::BelowEnergyWindow
    } else if momentum_squared.im <= 0.0 && momentum_squared.re <= 0.0 {
        XsphXsectEnergyDecision::NonPositiveMomentum
    } else {
        XsphXsectEnergyDecision::Active
    };

    Ok(XsphXsectEnergySetup {
        decision,
        momentum_squared,
        edge_momentum_squared,
        wave_number,
        muffin_tin_argument,
        cycle_count,
        photon_energy,
        photon_wave_number,
        active_radial_len,
    })
}

/// Port of FEFF `XSPH/xsect.f90` transition-loop planning.
///
/// FEFF traverses `mult = 0..2`, derives the `kx`/`ks` pair, applies the `le2`
/// higher-multipole selector, then scans `kdif = -kx..kx`. It skips inactive
/// `kiind(ind)` rows and preserves the `l2lp` direction filters before radial
/// solves are attempted.
pub fn xsph_xsect_transition_plan(
    input: XsphXsectTransitionPlanInput<'_>,
) -> Result<XsphXsectTransitionPlan, XsphError> {
    validate_xsect_transition_plan_input(&input)?;

    let mut transitions = Vec::new();
    for multipole in [
        XsphTransitionMultipole::ElectricDipole,
        XsphTransitionMultipole::MagneticDipole,
        XsphTransitionMultipole::ElectricQuadrupole,
    ] {
        if !xsect_multipole_enabled(multipole, input.selected_higher_multipole) {
            continue;
        }

        let multipole_order = xsect_transition_multipole_order(multipole);
        let transition_slot_offset = xsect_transition_slot_offset(multipole);
        for transition_delta in -(multipole_order as i32)..=(multipole_order as i32) {
            if input.photon_energy <= 0.0 {
                continue;
            }

            let transition_index = transition_slot_offset + transition_delta;
            if transition_index <= 0 {
                return Err(XsphError::IntegerOutOfRange {
                    name: "xsect_transition_index",
                    value: transition_index,
                });
            }
            let transition_index_1based =
                usize::try_from(transition_index).map_err(|_| XsphError::IntegerOutOfRange {
                    name: "xsect_transition_index",
                    value: transition_index,
                })?;
            if transition_index_1based > input.active_len {
                continue;
            }

            let final_kappa = input.final_kappas[transition_index_1based - 1];
            if final_kappa == 0
                || xsect_l2lp_skips(
                    input.transition_direction,
                    input.initial_kappa,
                    transition_index_1based,
                )
            {
                continue;
            }

            transitions.push(XsphXsectTransition {
                multipole,
                transition_delta,
                transition_index_1based,
                final_kappa,
                final_l: input.orbital_l[transition_index_1based - 1],
                multipole_order,
            });
        }
    }

    Ok(XsphXsectTransitionPlan { transitions })
}

/// Port of FEFF `XSPH/xsect.f90` screened-dipole field setup.
///
/// FEFF first computes `ww = dble(emu+p2-edge)`. For dipole transitions on a
/// standard atom it prepares `phiscf`, optionally runs `correorb`, and sets
/// `wse = dble(p2-eng(1,ihole))`; otherwise it uses the unity field and
/// `wse = ww`. The returned `field_scale` is FEFF's final `sqrt(wse/ww)`.
pub fn xsph_xsect_screened_field_setup(
    input: XsphXsectScreenedFieldInput,
) -> Result<XsphXsectScreenedField, XsphError> {
    validate_xsect_screened_field_input(input)?;

    let work_energy = (input.momentum_squared + input.chemical_potential - input.edge_energy).re;
    validate_finite_real("xsect_screened_work_energy", work_energy)?;

    let screened_dipole =
        input.multipole == XsphTransitionMultipole::ElectricDipole && input.standard_potential;
    let (
        mode,
        screened_transition_energy,
        unity_fscf,
        orbital_correction_required,
        orbital_correction_pending_after,
        phiscf_workspace,
    ) = if screened_dipole {
        validate_finite_real("screened_orbital_energy", input.screened_orbital_energy)?;
        (
            XsphXsectScreenedFieldMode::ScreenedDipole,
            (input.momentum_squared - input.screened_orbital_energy).re,
            false,
            input.orbital_correction_pending,
            false,
            Some(XsphXsectPhiscfWorkspace {
                max_size: 1,
                matrix_size: 0,
                scale_function: 1.0,
            }),
        )
    } else {
        (
            XsphXsectScreenedFieldMode::UnityField,
            work_energy,
            true,
            false,
            input.orbital_correction_pending,
            None,
        )
    };
    validate_finite_real(
        "xsect_screened_transition_energy",
        screened_transition_energy,
    )?;

    let field_scale = (screened_transition_energy / work_energy).sqrt();
    validate_finite_real("xsect_screened_field_scale", field_scale)?;

    Ok(XsphXsectScreenedField {
        mode,
        work_energy,
        screened_transition_energy,
        field_scale,
        unity_fscf,
        orbital_correction_required,
        orbital_correction_pending_after,
        phiscf_workspace,
    })
}

/// Port of FEFF `TDLDA/phiscf.f90` local exchange-field setup.
///
/// FEFF computes this `fxc` array before solving the Zangwill-Soven effective
/// field. `ifxc == 0` is the RPA branch and zeros the local term; nonzero
/// selectors use the Zangwill-Soven coefficients from `phiscf.f90`.
pub fn xsph_xsect_phiscf_local_field(
    input: XsphXsectPhiscfLocalFieldInput<'_>,
) -> Result<XsphXsectPhiscfLocalField, XsphError> {
    validate_xsect_phiscf_local_field_input(&input)?;

    let values = (0..input.active_len)
        .map(|index| {
            if input.exchange_correlation_selector == 0 {
                return 0.0;
            }
            let density = input.electron_density[index];
            let radius = input.radii[index];
            let rs = if density <= 0.0 {
                100.0
            } else {
                (4.0 * std::f64::consts::PI * density / 3.0).powf(-1.0 / 3.0)
            };
            rs.powi(3) / radius.powi(2) / 6.0 * (-1.222 / rs - 0.759_24 / (11.4 + rs))
        })
        .collect::<Array1<_>>();

    for value in values.iter().copied() {
        validate_finite_real("xsect_phiscf_local_field", value)?;
    }

    Ok(XsphXsectPhiscfLocalField { values })
}

/// Port of FEFF `TDLDA/chiklu.f90` screened-field linear solve.
///
/// FEFF solves `(1 - K*chi0) * f = r` on the coarse 0.05 grid with single
/// complex LAPACK (`cgetrf`/`cgetrs`), divides by the radial grid, and linearly
/// interpolates the four fine-grid points between each coarse pair. Optional
/// `yvec` source columns go through the same factorization.
pub fn xsph_xsect_phiscf_linear_solve(
    input: XsphXsectPhiscfLinearSolveInput<'_>,
) -> Result<XsphXsectPhiscfLinearSolve, XsphError> {
    validate_xsect_phiscf_linear_solve_input(&input)?;

    let fine_len = phiscf_linear_fine_len(input.coarse_count)?;
    let mut system = Array2::<Complex>::zeros((input.coarse_count, input.coarse_count));
    for row in 0..input.coarse_count {
        for column in 0..input.coarse_count {
            system[(row, column)] = -input.response[(row, column)];
        }
        system[(row, row)] += Complex::new(1.0, 0.0);
    }

    let rhs_count = input.basis_count + 1;
    let mut rhs = Array2::<Complex>::zeros((input.coarse_count, rhs_count));
    for coarse_index in 0..input.coarse_count {
        let fine_index = phiscf_linear_fine_index(coarse_index);
        rhs[(coarse_index, 0)] = Complex::new(input.radii[fine_index], 0.0);
        for basis_index in 0..input.basis_count {
            rhs[(coarse_index, basis_index + 1)] =
                Complex::new(input.basis_fields[(fine_index, basis_index)].re, 0.0);
        }
    }

    let system_scale = system
        .iter()
        .map(|value| value.re.abs() + value.im.abs())
        .fold(0.0, Real::max);
    validate_finite_real("xsect_phiscf_linear_system_scale", system_scale)?;
    if system_scale > 1.0 {
        system.mapv_inplace(|value| value / system_scale);
        rhs.mapv_inplace(|value| value / system_scale);
    }

    let solution = solve_phiscf_scaled_linear_system(system.view(), rhs.view())?;
    let screened_field = phiscf_interpolated_solution_column(
        solution.column(0).to_owned().view(),
        input.radii,
        fine_len,
    )?;
    let mut screened_basis_fields = Array2::<Complex>::zeros((fine_len, input.basis_count));
    for basis_index in 0..input.basis_count {
        let interpolated = phiscf_interpolated_solution_column(
            solution.column(basis_index + 1).to_owned().view(),
            input.radii,
            fine_len,
        )?;
        for index in 0..fine_len {
            screened_basis_fields[(index, basis_index)] = interpolated[index];
        }
    }

    Ok(XsphXsectPhiscfLinearSolve {
        screened_field,
        screened_basis_fields,
    })
}

/// Port of FEFF `TDLDA/lipman.f90` `K*chi0` response assembly.
///
/// This builds the coarse `chik` matrix that `chiklu` later solves. FEFF samples
/// every fifth point of the fine radial grid, so `coarse_count` rows cover fine
/// indices `1 + 5*(i0-1)` in Fortran indexing.
pub fn xsph_xsect_phiscf_lipman_response(
    input: XsphXsectPhiscfLipmanInput<'_>,
) -> Result<XsphXsectPhiscfLipman, XsphError> {
    validate_xsect_phiscf_lipman_input(&input)?;

    let mut regular = Vec::with_capacity(input.active_len);
    let mut irregular = Vec::with_capacity(input.active_len);
    for index in 0..input.active_len {
        regular.push(
            input.regular_large[index] * input.orbital_large[index]
                + input.regular_small[index] * input.orbital_small[index],
        );
        irregular.push(
            input.irregular_large[index] * input.orbital_large[index]
                + input.irregular_small[index] * input.orbital_small[index],
        );
    }

    let regular_over_r2 = (0..input.active_len)
        .map(|index| regular[index] / input.radii[index].powi(2))
        .collect::<Vec<_>>();
    let regular_times_r = (0..input.active_len)
        .map(|index| regular[index] * input.radii[index])
        .collect::<Vec<_>>();
    let irregular_over_r2 = (0..input.active_len)
        .map(|index| irregular[index] / input.radii[index].powi(2))
        .collect::<Vec<_>>();
    let irregular_times_r = (0..input.active_len)
        .map(|index| irregular[index] * input.radii[index])
        .collect::<Vec<_>>();

    let f1 = phiscf_lipman_prefix_integral(&regular_over_r2, input.radii, input.active_len);
    let f2 = phiscf_lipman_prefix_integral(&regular_times_r, input.radii, input.active_len);
    let f3 = phiscf_lipman_tail_integral(
        &irregular_over_r2,
        input.radii,
        input.active_len,
        input.match_index_1based,
    );
    let f4 = phiscf_lipman_tail_integral(
        &irregular_times_r,
        input.radii,
        input.active_len,
        input.match_index_1based,
    );

    let quadrature_step = 0.05;
    let weighted_regular = (0..input.active_len)
        .map(|index| regular[index] * quadrature_step * input.radii[index])
        .collect::<Vec<_>>();
    let weighted_irregular = (0..input.active_len)
        .map(|index| irregular[index] * quadrature_step * input.radii[index])
        .collect::<Vec<_>>();

    let mut response = Array2::<Complex>::zeros((input.coarse_count, input.coarse_count));
    for row in 0..input.coarse_count {
        let fine_row = phiscf_linear_fine_index(row);
        if fine_row >= input.active_len {
            continue;
        }
        let row_radius = input.radii[fine_row];
        let row_radius_squared = row_radius.powi(2);
        for column in 0..input.coarse_count {
            let fine_column = phiscf_linear_fine_index(column);
            if fine_column >= input.active_len {
                continue;
            }
            let value = if row <= column {
                let kernel = f2[fine_row] / row_radius_squared
                    + (f1[fine_column] - f1[fine_row]) * row_radius;
                kernel * weighted_irregular[fine_column]
                    + f3[fine_column] * row_radius * weighted_regular[fine_column]
                    + input.local_field[fine_row]
                        * regular[fine_row]
                        * weighted_irregular[fine_column]
            } else {
                let kernel = f2[fine_column] / row_radius_squared
                    + (f4[fine_column] - f4[fine_row]) / row_radius_squared
                    + f3[fine_row] * row_radius;
                kernel * weighted_regular[fine_column]
                    + input.local_field[fine_row]
                        * irregular[fine_row]
                        * weighted_regular[fine_column]
            };
            validate_finite_complex("xsect_phiscf_lipman_response", row, value)?;
            response[(row, column)] = value;
        }
    }

    Ok(XsphXsectPhiscfLipman { response })
}

/// Port of FEFF `TDLDA/phiscf.f90` accumulation of `lipman` responses into `cchik`.
///
/// In the production branch (`itest.eq.0`), FEFF keeps the imaginary part only
/// for the forward pole above the edge; otherwise it contributes only
/// `dble(chik)`. The screened-response branch can generate very large
/// below-edge second-pole rows, so Rust keeps the accumulated `cchik` matrix in
/// double precision through the matching linear solve.
pub fn xsph_xsect_phiscf_accumulated_response(
    input: XsphXsectPhiscfAccumulatedResponseInput<'_>,
) -> Result<XsphXsectPhiscfAccumulatedResponse, XsphError> {
    if input.coarse_count == 0 {
        return Err(XsphError::EmptyIndexSet);
    }

    let mut response = Array2::<Complex>::zeros((input.coarse_count, input.coarse_count));
    for (contribution_index, contribution) in input.contributions.iter().enumerate() {
        validate_finite_real("xsect_phiscf_response_scale", contribution.scale)?;
        if contribution.response.nrows() < input.coarse_count
            || contribution.response.ncols() < input.coarse_count
        {
            return Err(XsphError::MatrixTooSmall {
                name: "xsect_phiscf_response_contribution",
                required: [input.coarse_count, input.coarse_count],
                actual: [contribution.response.nrows(), contribution.response.ncols()],
            });
        }
        for row in 0..input.coarse_count {
            for column in 0..input.coarse_count {
                let source = contribution.response[(row, column)];
                validate_finite_complex(
                    "xsect_phiscf_response_contribution",
                    contribution_index,
                    source,
                )?;
                let real = source.re * contribution.scale;
                let imaginary = if contribution.include_imaginary {
                    source.im * contribution.scale
                } else {
                    0.0
                };
                let contribution_value = Complex::new(real, imaginary);
                validate_finite_complex(
                    "xsect_phiscf_response_contribution",
                    contribution_index,
                    contribution_value,
                )?;
                response[(row, column)] += contribution_value;
            }
        }
    }

    Ok(XsphXsectPhiscfAccumulatedResponse { response })
}

/// Port of FEFF `TDLDA/phiscf.f90` `aa` scaling and production pole branch.
///
/// FEFF multiplies each `lipman` response by an angular `cwig3j` factor, shell
/// occupation, photon-energy correction, and `sfun`. In the production branch
/// (`itest.eq.0`), only the first pole above `edge` contributes the imaginary
/// part of `chik`; other poles contribute the real part only.
pub fn xsph_xsect_phiscf_contribution_rule(
    input: XsphXsectPhiscfContributionRuleInput,
) -> Result<XsphXsectPhiscfContributionRule, XsphError> {
    if input.initial_kappa == 0 {
        return Err(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: input.initial_kappa,
        });
    }
    if input.final_kappa == 0 {
        return Err(XsphError::IntegerOutOfRange {
            name: "final_kappa",
            value: input.final_kappa,
        });
    }
    if input.pole_index_1based == 0 || input.pole_index_1based > 2 {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_pole_index",
            index_1based: input.pole_index_1based,
            active_len: 2,
        });
    }
    validate_finite_real(
        "xsect_phiscf_shell_occupation_fraction",
        input.shell_occupation_fraction,
    )?;
    validate_finite_real(
        "xsect_phiscf_photon_energy_correction",
        input.photon_energy_correction,
    )?;
    validate_finite_real("xsect_phiscf_scale_function", input.scale_function)?;
    validate_finite_complex("xsect_phiscf_pole_energy", 0, input.pole_energy)?;
    validate_finite_real("xsect_phiscf_edge_energy", input.edge_energy)?;

    let jfin2 = doubled_j_from_kappa("final_kappa", input.final_kappa)?;
    let jin2 = doubled_j_from_kappa("initial_kappa", input.initial_kappa)?;
    validate_cwig3j_doubled_argument("final_kappa", input.final_kappa, jfin2)?;
    validate_cwig3j_doubled_argument("initial_kappa", input.initial_kappa, jin2)?;

    let angular = wigner_3j(jfin2, 2, jin2, 1, 0, 2)?;
    let angular_scale = -angular.powi(2) * Real::from((jfin2 + 1) * (jin2 + 1)) / 3.0;
    let scale = angular_scale
        * input.shell_occupation_fraction
        * input.photon_energy_correction
        * input.scale_function;
    validate_finite_real("xsect_phiscf_contribution_scale", scale)?;

    Ok(XsphXsectPhiscfContributionRule {
        angular_scale,
        scale,
        include_imaginary: input.pole_index_1based == 1 && input.pole_energy.re > input.edge_energy,
    })
}

/// Port of FEFF `TDLDA/phiscf.f90` pole-energy and broadening setup.
///
/// FEFF computes `ww = p2 - eng(1,ihole)` once, uses
/// `dble(ww) / dble(p2 + emu - edge)` as the photon-energy correction, and
/// builds two pole energies for each occupied DOS row. Below-edge poles replace
/// the imaginary part with `max(dimag(ww), (edge - yy)/10)`.
pub fn xsph_xsect_phiscf_pole_energy(
    input: XsphXsectPhiscfPoleEnergyInput,
) -> Result<XsphXsectPhiscfPoleEnergy, XsphError> {
    if input.pole_index_1based == 0 || input.pole_index_1based > 2 {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_pole_index",
            index_1based: input.pole_index_1based,
            active_len: 2,
        });
    }
    validate_finite_complex("xsect_phiscf_momentum_squared", 0, input.momentum_squared)?;
    validate_finite_real("xsect_phiscf_edge_energy", input.edge_energy)?;
    validate_finite_real("xsect_phiscf_chemical_potential", input.chemical_potential)?;
    validate_finite_real(
        "xsect_phiscf_hole_orbital_energy",
        input.hole_orbital_energy,
    )?;
    validate_finite_real(
        "xsect_phiscf_occupied_orbital_energy",
        input.occupied_orbital_energy,
    )?;

    let photon_energy = (input.momentum_squared + input.chemical_potential - input.edge_energy).re;
    validate_finite_real("xsect_phiscf_photon_energy", photon_energy)?;

    let response_energy = input.momentum_squared - input.hole_orbital_energy;
    validate_finite_complex("xsect_phiscf_response_energy", 0, response_energy)?;

    let photon_energy_correction = response_energy.re / photon_energy;
    validate_finite_real(
        "xsect_phiscf_photon_energy_correction",
        photon_energy_correction,
    )?;

    let raw_pole_energy = if input.pole_index_1based == 1 {
        Complex::new(input.occupied_orbital_energy, 0.0) + response_energy
    } else {
        Complex::new(
            input.occupied_orbital_energy - response_energy.re,
            response_energy.im,
        )
    };
    validate_finite_complex("xsect_phiscf_raw_pole_energy", 0, raw_pole_energy)?;

    let below_edge_broadening_applied = raw_pole_energy.re < input.edge_energy;
    let pole_energy = if below_edge_broadening_applied {
        let resonance_reference_energy = if input.pole_index_1based == 1 {
            input.occupied_orbital_energy + response_energy.re
        } else {
            input.occupied_orbital_energy
        };
        validate_finite_real(
            "xsect_phiscf_resonance_reference_energy",
            resonance_reference_energy,
        )?;
        let broadening = response_energy
            .im
            .max((input.edge_energy - resonance_reference_energy) / 10.0);
        validate_finite_real("xsect_phiscf_pole_broadening", broadening)?;
        Complex::new(raw_pole_energy.re, broadening)
    } else {
        raw_pole_energy
    };
    validate_finite_complex("xsect_phiscf_pole_energy", 0, pole_energy)?;

    Ok(XsphXsectPhiscfPoleEnergy {
        photon_energy,
        response_energy,
        photon_energy_correction,
        raw_pole_energy,
        pole_energy,
        below_edge_broadening_applied,
        broadening: pole_energy.im,
    })
}

/// Port of FEFF `TDLDA/phiscf.f90` occupied-state contribution traversal.
///
/// FEFF loops over occupied orbitals (`iorb`), projected DOS rows (`ieg`), two
/// response poles (`ind`), then dipole final-state candidates (`ik=-1..1`).
/// This helper emits those rows with pole-energy setup and `aa` scaling already
/// evaluated. Radial wavefunction generation and `lipman` response assembly are
/// intentionally left to the downstream producer.
pub fn xsph_xsect_phiscf_contribution_plan(
    input: XsphXsectPhiscfContributionPlanInput<'_>,
) -> Result<XsphXsectPhiscfContributionPlan, XsphError> {
    validate_xsect_phiscf_contribution_plan_input(&input)?;

    let mut rows = Vec::new();
    for orbital_index in 0..input.active_orbital_count {
        let initial_kappa = input.orbital_kappas[orbital_index];
        if initial_kappa == 0 {
            return Err(XsphError::ZeroKappa);
        }
        let energy_count = input.orbital_energy_counts[orbital_index];
        for energy_index in 0..energy_count {
            let occupied_orbital_energy = input.occupied_energies[(energy_index, orbital_index)];
            let shell_occupation_fraction =
                input.occupation_fractions[(energy_index, orbital_index)];
            for pole_index_1based in 1..=2 {
                let pole = xsph_xsect_phiscf_pole_energy(XsphXsectPhiscfPoleEnergyInput {
                    momentum_squared: input.momentum_squared,
                    edge_energy: input.edge_energy,
                    chemical_potential: input.chemical_potential,
                    hole_orbital_energy: input.hole_orbital_energy,
                    occupied_orbital_energy,
                    pole_index_1based,
                })?;

                for dipole_delta in -1..=1 {
                    let Some(final_kappa) = phiscf_dipole_final_kappa(initial_kappa, dipole_delta)?
                    else {
                        continue;
                    };
                    let rule = xsph_xsect_phiscf_contribution_rule(
                        XsphXsectPhiscfContributionRuleInput {
                            initial_kappa,
                            final_kappa,
                            shell_occupation_fraction,
                            photon_energy_correction: pole.photon_energy_correction,
                            scale_function: input.scale_function,
                            pole_index_1based,
                            pole_energy: pole.pole_energy,
                            edge_energy: input.edge_energy,
                        },
                    )?;
                    rows.push(XsphXsectPhiscfContributionPlanRow {
                        orbital_index_1based: orbital_index + 1,
                        energy_index_1based: energy_index + 1,
                        pole_index_1based,
                        dipole_delta,
                        initial_kappa,
                        final_kappa,
                        occupied_orbital_energy,
                        shell_occupation_fraction,
                        pole,
                        rule,
                    });
                }
            }
        }
    }

    Ok(XsphXsectPhiscfContributionPlan { rows })
}

/// Port of FEFF `TDLDA/phiscf.f90` `ck`/`jrip`/`iwkb` setup before `wfirdc`.
pub fn xsph_xsect_phiscf_radial_solver_setup(
    input: XsphXsectPhiscfRadialSolverSetupInput<'_>,
) -> Result<XsphXsectPhiscfRadialSolverSetup, XsphError> {
    validate_xsect_phiscf_radial_solver_setup_input(&input)?;

    let alpha_scaled = input.pole_energy * super::XSPH_FINE_STRUCTURE_ALPHA;
    let wave_number = (input.pole_energy * 2.0 + alpha_scaled * alpha_scaled).sqrt();
    validate_finite_complex("xsect_phiscf_wave_number", 0, wave_number)?;

    let mut matching_radius_limit = 10.0 / wave_number.im.abs();
    if matching_radius_limit > input.muffin_tin_radius {
        matching_radius_limit = input.muffin_tin_radius;
    }
    validate_finite_real("xsect_phiscf_matching_radius_limit", matching_radius_limit)?;
    if matching_radius_limit <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "xsect_phiscf_matching_radius_limit",
            value: matching_radius_limit,
        });
    }

    let match_index_1based = feff_positive_float_to_usize(
        (matching_radius_limit.ln() + input.origin_shift) / input.log_step + 2.0,
        "xsect_phiscf_match_index",
    )?;
    if match_index_1based == 0 || match_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_match_index",
            index_1based: match_index_1based,
            active_len: input.active_len,
        });
    }
    let match_index = match_index_1based - 1;
    let match_radius = input.radii[match_index] - 1.0e-20;
    validate_finite_real("xsect_phiscf_match_radius", match_radius)?;
    if match_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "xsect_phiscf_match_radius",
            value: match_radius,
        });
    }

    let rwkb = 0.5 / input.log_step / wave_number.norm();
    validate_finite_real("xsect_phiscf_rwkb", rwkb)?;
    let mut wkb_index_1based = feff_positive_float_to_usize(
        (rwkb.ln() + input.origin_shift) / input.log_step + 2.0,
        "xsect_phiscf_wkb_index",
    )?;
    if wkb_index_1based > input.active_len {
        wkb_index_1based = input.active_len;
    }
    if wkb_index_1based < 10 {
        wkb_index_1based = 10;
    }
    if wkb_index_1based >= input.target_last_index_1based.saturating_sub(1) {
        wkb_index_1based = input.active_len;
    }
    if wkb_index_1based == 0 || wkb_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_wkb_index",
            index_1based: wkb_index_1based,
            active_len: input.active_len,
        });
    }
    let wkb_index = wkb_index_1based - 1;

    Ok(XsphXsectPhiscfRadialSolverSetup {
        wave_number,
        matching_radius_limit,
        match_index_1based,
        match_index,
        match_radius,
        wkb_index_1based,
        wkb_index,
        match_argument_inside: wave_number * match_radius,
        match_argument_grid: wave_number * input.radii[match_index],
    })
}

/// Select FEFF `TDLDA/phiscf.f90` large/small angular channels from `kfin`.
pub fn xsph_xsect_phiscf_angular_channels(
    final_kappa: i32,
) -> Result<XsphXsectPhiscfAngularChannels, XsphError> {
    if final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }

    let large_l_i32 = l_from_kappa(final_kappa)?;
    let small_l_i32 = if final_kappa > 0 {
        large_l_i32
            .checked_sub(1)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "final_kappa",
                value: final_kappa,
            })?
    } else {
        large_l_i32
            .checked_add(1)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "final_kappa",
                value: final_kappa,
            })?
    };

    let large_l = usize::try_from(large_l_i32).map_err(|_| XsphError::IntegerOutOfRange {
        name: "final_kappa",
        value: final_kappa,
    })?;
    let small_l = usize::try_from(small_l_i32).map_err(|_| XsphError::IntegerOutOfRange {
        name: "final_kappa",
        value: final_kappa,
    })?;

    Ok(XsphXsectPhiscfAngularChannels { large_l, small_l })
}

/// Port of FEFF `TDLDA/phiscf.f90` irregular `wfirdc` seed setup.
pub fn xsph_xsect_phiscf_irregular_seed(
    input: XsphXsectPhiscfIrregularSeedInput,
) -> Result<XsphXsectPhiscfIrregularSeed, XsphError> {
    validate_xsect_phiscf_irregular_seed_input(input)?;

    let channels = xsph_xsect_phiscf_angular_channels(input.final_kappa)?;
    let scales = xsph_xsect_relativistic_scales(input.wave_number, input.final_kappa)?;
    let max_l = channels.large_l.max(channels.small_l);
    let hankel = besjh(input.wave_number * input.match_radius, max_l)?;
    let large_coefficient =
        hankel.h[channels.large_l] * input.match_radius * scales.relativistic_scale;
    let small_coefficient = hankel.h[channels.small_l]
        * input.match_radius
        * scales.relativistic_scale
        * scales.small_component_factor;

    validate_finite_complex("xsect_phiscf_irregular_seed_large", 0, large_coefficient)?;
    validate_finite_complex("xsect_phiscf_irregular_seed_small", 0, small_coefficient)?;

    Ok(XsphXsectPhiscfIrregularSeed {
        channels,
        small_component_factor: scales.small_component_factor,
        relativistic_scale: scales.relativistic_scale,
        large_coefficient,
        small_coefficient,
    })
}

/// Port of FEFF `TDLDA/phiscf.f90` post-`wfirdc` Wronskian/outside assembly.
///
/// FEFF first normalizes `ph/qh` inside `jrip` with the regular/irregular
/// Wronskian, then continues `pir/qir` and `ph/qh` outside `jrip` with the
/// spherical Hankel/Bessel fields at `ck*ri`.
pub fn xsph_xsect_phiscf_field_assembly(
    input: XsphXsectPhiscfFieldAssemblyInput<'_>,
) -> Result<XsphXsectPhiscfFields, XsphError> {
    validate_xsect_phiscf_field_assembly_input(&input)?;

    let channels = xsph_xsect_phiscf_angular_channels(input.final_kappa)?;
    let max_l = channels.large_l.max(channels.small_l);
    let match_index = input.match_index_1based - 1;
    let match_radius = input.radii[match_index];
    let match_argument = input.wave_number * match_radius;
    validate_xsect_phiscf_nonzero_complex_result(
        "xsect_phiscf_match_argument_grid",
        match_argument,
    )?;

    let match_hankel = besjh(match_argument, max_l)?;
    let hankel_large_at_match = match_hankel.h[channels.large_l];
    let hankel_small_at_match = match_hankel.h[channels.small_l];
    validate_xsect_phiscf_nonzero_complex_result(
        "xsect_phiscf_hankel_large_at_match",
        hankel_large_at_match,
    )?;
    validate_xsect_phiscf_nonzero_complex_result(
        "xsect_phiscf_hankel_small_at_match",
        hankel_small_at_match,
    )?;

    let wronskian_denominator = 2.0
        * (1.0 / super::XSPH_FINE_STRUCTURE_ALPHA)
        * (input.irregular_large[match_index] * input.regular_small[match_index]
            - input.regular_large[match_index] * input.irregular_small[match_index]);
    validate_xsect_phiscf_nonzero_complex_result(
        "xsect_phiscf_wronskian_denominator",
        wronskian_denominator,
    )?;
    let wronskian_scale = Complex::new(2.0, 0.0) / wronskian_denominator;
    validate_finite_complex("xsect_phiscf_wronskian_scale", 0, wronskian_scale)?;

    let mut regular_large =
        Array1::from_iter((0..input.active_len).map(|index| input.regular_large[index]));
    let mut regular_small =
        Array1::from_iter((0..input.active_len).map(|index| input.regular_small[index]));
    let mut irregular_large =
        Array1::from_iter((0..input.active_len).map(|index| input.irregular_large[index]));
    let mut irregular_small =
        Array1::from_iter((0..input.active_len).map(|index| input.irregular_small[index]));

    for index in 0..=match_index {
        regular_large[index] *= wronskian_scale;
        regular_small[index] *= wronskian_scale;
    }

    let tail_denominator = match_argument * 2.0;
    validate_xsect_phiscf_nonzero_complex_result(
        "xsect_phiscf_tail_denominator",
        tail_denominator,
    )?;
    let tail_coefficient = (regular_large[match_index] / tail_denominator
        - match_hankel.j[channels.large_l])
        / hankel_large_at_match;
    validate_finite_complex("xsect_phiscf_tail_coefficient", 0, tail_coefficient)?;

    for index in (match_index + 1)..input.active_len {
        let argument = input.wave_number * input.radii[index];
        validate_xsect_phiscf_nonzero_complex_result("xsect_phiscf_argument_grid", argument)?;
        let hankel = besjh(argument, max_l)?;
        let radius_ratio = input.radii[index] / match_radius;
        irregular_large[index] =
            input.irregular_large[match_index] * radius_ratio * hankel.h[channels.large_l]
                / hankel_large_at_match;
        irregular_small[index] =
            input.irregular_small[match_index] * radius_ratio * hankel.h[channels.small_l]
                / hankel_small_at_match;
        regular_large[index] = 2.0
            * argument
            * (hankel.j[channels.large_l] + tail_coefficient * hankel.h[channels.large_l]);
        regular_small[index] = 2.0
            * argument
            * (hankel.j[channels.small_l] + tail_coefficient * hankel.h[channels.small_l]);
    }

    for index in 0..input.active_len {
        validate_finite_complex("xsect_phiscf_regular_large", index, regular_large[index])?;
        validate_finite_complex("xsect_phiscf_regular_small", index, regular_small[index])?;
        validate_finite_complex(
            "xsect_phiscf_irregular_large",
            index,
            irregular_large[index],
        )?;
        validate_finite_complex(
            "xsect_phiscf_irregular_small",
            index,
            irregular_small[index],
        )?;
    }

    Ok(XsphXsectPhiscfFields {
        channels,
        wronskian_scale,
        tail_coefficient,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
    })
}

/// Compose one FEFF `TDLDA/phiscf.f90` radial contribution from `wfirdc` rows.
///
/// This is the source-order bridge from the regular/irregular `wfirdc`
/// solutions through Wronskian matching into the `lipman` response matrix. The
/// returned contribution owns its response so the full `phiscf` driver can
/// collect contributions in FEFF traversal order before the `cchik` solve.
pub fn xsph_xsect_phiscf_radial_contribution(
    input: XsphXsectPhiscfRadialContributionInput<'_>,
) -> Result<XsphXsectPhiscfRadialContribution, XsphError> {
    validate_finite_real("xsect_phiscf_response_scale", input.response_scale)?;

    let fields = xsph_xsect_phiscf_field_assembly(XsphXsectPhiscfFieldAssemblyInput {
        final_kappa: input.final_kappa,
        wave_number: input.wave_number,
        radii: input.radii,
        regular_large: input.regular_large,
        regular_small: input.regular_small,
        irregular_large: input.irregular_large,
        irregular_small: input.irregular_small,
        active_len: input.active_len,
        match_index_1based: input.match_index_1based,
    })?;
    let response = xsph_xsect_phiscf_lipman_response(XsphXsectPhiscfLipmanInput {
        coarse_count: input.coarse_count,
        active_len: input.active_len,
        match_index_1based: input.match_index_1based,
        radii: input.radii,
        orbital_large: input.orbital_large,
        orbital_small: input.orbital_small,
        regular_large: fields.regular_large.view(),
        regular_small: fields.regular_small.view(),
        irregular_large: fields.irregular_large.view(),
        irregular_small: fields.irregular_small.view(),
        local_field: input.local_field,
    })?;

    Ok(XsphXsectPhiscfRadialContribution {
        fields,
        response,
        scale: input.response_scale,
        include_imaginary: input.include_response_imaginary,
    })
}

/// Generate one FEFF `phiscf` contribution from regular/irregular `wfirdc`.
///
/// The supplied `wfirdc_input` is the already-prepared per-contribution source
/// state: target `kappa`, pole energy, `vxcp`, `vm`, `rmtp`, `jrip`, `iwkb`,
/// and bound-orbital data. This helper owns the FEFF branch switches:
/// regular `irr=-1`, irregular Hankel seed, irregular `irr=1`, then the
/// matched-field `lipman` contribution.
pub fn xsph_xsect_phiscf_wfirdc_contribution(
    input: XsphXsectPhiscfWfirdcContributionInput<'_>,
) -> Result<XsphXsectPhiscfWfirdcContribution, XsphError> {
    validate_active_len(
        "xsect_phiscf_wfirdc_kappa",
        input.wfirdc_input.kappa.len(),
        input.wfirdc_input.orbital_count,
    )?;
    let target_index = input
        .wfirdc_input
        .orbital_count
        .checked_sub(1)
        .ok_or(XsphError::EmptyIndexSet)?;
    let final_kappa = input.wfirdc_input.kappa[target_index];
    if final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    let match_index_1based =
        input
            .wfirdc_input
            .radial_match_index
            .checked_add(1)
            .ok_or(XsphError::SizeOutOfRange {
                name: "xsect_phiscf_match_index",
                value: input.wfirdc_input.radial_match_index,
            })?;

    let zero = Complex::new(0.0, 0.0);
    let regular_solution = fovrg_initial_photoelectron(crate::FovrgInitialPhotoelectronInput {
        irregular: false,
        initial_large_coefficient: zero,
        initial_small_coefficient: zero,
        coefficient_count: XSECT_PHISCF_REGULAR_WFIRDC_COEFFICIENT_COUNT,
        ..input.wfirdc_input
    })?;
    let irregular_seed = xsph_xsect_phiscf_irregular_seed(XsphXsectPhiscfIrregularSeedInput {
        final_kappa,
        wave_number: input.wave_number,
        match_radius: input.wfirdc_input.muffin_tin_radius,
    })?;
    let irregular_solution = fovrg_initial_photoelectron(crate::FovrgInitialPhotoelectronInput {
        irregular: true,
        initial_large_coefficient: irregular_seed.large_coefficient,
        initial_small_coefficient: irregular_seed.small_coefficient,
        coefficient_count: XSECT_PHISCF_IRREGULAR_WFIRDC_COEFFICIENT_COUNT,
        ..input.wfirdc_input
    })?;
    let contribution =
        xsph_xsect_phiscf_radial_contribution(XsphXsectPhiscfRadialContributionInput {
            coarse_count: input.coarse_count,
            active_len: input.wfirdc_input.active_len,
            match_index_1based,
            final_kappa,
            wave_number: input.wave_number,
            radii: regular_solution.nuclear_potential.radii.view(),
            orbital_large: input.orbital_large,
            orbital_small: input.orbital_small,
            regular_large: regular_solution.large_component.view(),
            regular_small: regular_solution.small_component.view(),
            irregular_large: irregular_solution.large_component.view(),
            irregular_small: irregular_solution.small_component.view(),
            local_field: input.local_field,
            response_scale: input.response_scale,
            include_response_imaginary: input.include_response_imaginary,
        })?;

    Ok(XsphXsectPhiscfWfirdcContribution {
        regular_solution,
        irregular_seed,
        irregular_solution,
        contribution,
    })
}

/// Collect source-backed FEFF `phiscf` contributions from `wfirdc` and solve `fscf`.
///
/// Production `phiscf` builds every occupied-orbital/pole contribution before
/// solving the accumulated `cchik` response. This helper is the source-order
/// bridge from per-row `wfirdc` inputs to that screened-field solve; callers
/// still own construction of each pole-specific `wfirdc_input`.
pub fn xsph_xsect_phiscf_wfirdc_contributions(
    input: XsphXsectPhiscfWfirdcContributionsInput<'_>,
) -> Result<XsphXsectPhiscfWfirdcContributions, XsphError> {
    validate_active_len(
        "xsect_phiscf_wfirdc_contributions",
        input.contribution_inputs.len(),
        1,
    )?;

    let mut contributions = Vec::with_capacity(input.contribution_inputs.len());
    for contribution_input in input.contribution_inputs.iter().copied() {
        if contribution_input.coarse_count != input.coarse_count {
            return Err(XsphError::SizeOutOfRange {
                name: "xsect_phiscf_wfirdc_coarse_count",
                value: contribution_input.coarse_count,
            });
        }
        contributions.push(xsph_xsect_phiscf_wfirdc_contribution(contribution_input)?);
    }

    let response_contributions = contributions
        .iter()
        .map(|contribution| XsphXsectPhiscfResponseContributionInput {
            response: contribution.contribution.response.response.view(),
            scale: contribution.contribution.scale,
            include_imaginary: contribution.contribution.include_imaginary,
        })
        .collect::<Vec<_>>();
    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: input.coarse_count,
            contributions: &response_contributions,
        })?;
    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: input.coarse_count,
        radii: input.radii,
        response: accumulated.response.view(),
        basis_fields: input.basis_fields,
        basis_count: input.basis_count,
    })?;

    Ok(XsphXsectPhiscfWfirdcContributions {
        contributions,
        screened_solution: XsphXsectPhiscfScreenedSolution {
            response: accumulated.response,
            screened_field: solved.screened_field,
            screened_basis_fields: solved.screened_basis_fields,
        },
    })
}

/// Accumulate FEFF `phiscf` radial contributions and solve the screened field.
pub fn xsph_xsect_phiscf_screened_contributions(
    input: XsphXsectPhiscfScreenedContributionsInput<'_>,
) -> Result<XsphXsectPhiscfScreenedSolution, XsphError> {
    let contribution_inputs = input
        .contributions
        .iter()
        .map(|contribution| XsphXsectPhiscfResponseContributionInput {
            response: contribution.response.response.view(),
            scale: contribution.scale,
            include_imaginary: contribution.include_imaginary,
        })
        .collect::<Vec<_>>();
    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: input.coarse_count,
            contributions: &contribution_inputs,
        })?;
    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: input.coarse_count,
        radii: input.radii,
        response: accumulated.response.view(),
        basis_fields: input.basis_fields,
        basis_count: input.basis_count,
    })?;

    Ok(XsphXsectPhiscfScreenedSolution {
        response: accumulated.response,
        screened_field: solved.screened_field,
        screened_basis_fields: solved.screened_basis_fields,
    })
}

/// Port of the FEFF `TDLDA/phiscf.f90` `lipman` + `chiklu` solve chain.
///
/// This produces a single source-backed screened-field solve from one
/// `lipman` response contribution with the same `cchik` accumulation rule FEFF
/// applies inside the occupied-orbital/pole loops. The full FEFF `phiscf`
/// driver still has to generate all contributions before this result can retire
/// the positive-`izstd` XSPH gate.
pub fn xsph_xsect_phiscf_screened_solution(
    input: XsphXsectPhiscfScreenedSolutionInput<'_>,
) -> Result<XsphXsectPhiscfScreenedSolution, XsphError> {
    let response = xsph_xsect_phiscf_lipman_response(XsphXsectPhiscfLipmanInput {
        coarse_count: input.coarse_count,
        active_len: input.active_len,
        match_index_1based: input.match_index_1based,
        radii: input.radii,
        orbital_large: input.orbital_large,
        orbital_small: input.orbital_small,
        regular_large: input.regular_large,
        regular_small: input.regular_small,
        irregular_large: input.irregular_large,
        irregular_small: input.irregular_small,
        local_field: input.local_field,
    })?;
    let contribution = XsphXsectPhiscfResponseContributionInput {
        response: response.response.view(),
        scale: input.response_scale,
        include_imaginary: input.include_response_imaginary,
    };
    let contributions = [contribution];
    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: input.coarse_count,
            contributions: &contributions,
        })?;
    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: input.coarse_count,
        radii: input.radii,
        response: accumulated.response.view(),
        basis_fields: input.basis_fields,
        basis_count: input.basis_count,
    })?;

    Ok(XsphXsectPhiscfScreenedSolution {
        response: accumulated.response,
        screened_field: solved.screened_field,
        screened_basis_fields: solved.screened_basis_fields,
    })
}

/// Port of FEFF `XSPH/xsect.f90` real/imaginary `fscf` pass setup.
///
/// FEFF loops over `id = 1, 2`, using `dble(fscf)` first and `dimag(fscf)`
/// second, but exits before the imaginary pass for nonstandard atoms
/// (`izstd.le.0`). This helper returns those radial weights in FEFF traversal
/// order for the later `radint` calls.
pub fn xsph_xsect_fscf_weights(
    input: XsphXsectFscfWeightsInput<'_>,
) -> Result<XsphXsectFscfWeights, XsphError> {
    validate_xsect_fscf_weights_input(&input)?;

    let mut components = Vec::with_capacity(if input.standard_potential { 2 } else { 1 });
    components.push(XsphXsectFscfWeight {
        component_id: 1,
        part: XsphXsectFscfComponentPart::Real,
        weights: (0..input.active_len)
            .map(|index| input.fscf[index].re)
            .collect::<Array1<_>>(),
    });

    if input.standard_potential {
        components.push(XsphXsectFscfWeight {
            component_id: 2,
            part: XsphXsectFscfComponentPart::Imaginary,
            weights: (0..input.active_len)
                .map(|index| input.fscf[index].im)
                .collect::<Array1<_>>(),
        });
    }

    Ok(XsphXsectFscfWeights { components })
}

/// Port of FEFF `XSPH/xsect.f90` `ifl` and post-`radint` scaling setup.
///
/// FEFF calls `radint` with positive `ifl` for nonstandard atoms and negative
/// `ifl` for standard atoms. The negative branches are then scaled by `xk0` in
/// the reduced-matrix section and by `xk0**2 * ww**2` in the central
/// cross-section section.
pub fn xsph_xsect_radial_pass(
    input: XsphXsectRadialPassInput,
) -> Result<XsphXsectRadialPass, XsphError> {
    validate_xsect_radial_pass_input(input)?;

    let base_ifl = match input.kind {
        XsphXsectRadialPassKind::ReducedMatrixElement => 1,
        XsphXsectRadialPassKind::CentralCrossSection => 2,
    };

    if input.standard_potential {
        let post_radint_scale = match input.kind {
            XsphXsectRadialPassKind::ReducedMatrixElement => input.photon_wave_number,
            XsphXsectRadialPassKind::CentralCrossSection => {
                input.photon_wave_number.powi(2) * input.screened_field_scale.powi(2)
            }
        };
        validate_finite_real("xsect_radial_pass_scale", post_radint_scale)?;
        Ok(XsphXsectRadialPass {
            feff_ifl: -base_ifl,
            post_radint_scale,
        })
    } else {
        Ok(XsphXsectRadialPass {
            feff_ifl: base_ifl,
            post_radint_scale: 1.0,
        })
    }
}

/// Port of FEFF `XSPH/xsect.f90` photon Bessel-table setup for `radint`.
///
/// FEFF uses a `bjnser` series for `abs(xk0 * ri(i)) < 1` and closed
/// spherical-Bessel formulas otherwise. The returned table is shaped like
/// FEFF `bf(0:2, 1:ilast)`, with rows `j_0`, `j_1`, and `j_2`.
pub fn xsph_xray_bessel_table(
    input: XsphXrayBesselTableInput<'_>,
) -> Result<XsphXrayBesselTable, XsphError> {
    validate_xray_bessel_input(&input)?;

    let mut values = Array2::<Real>::zeros((RADINT_BESSEL_ROWS, input.active_len).f());
    for index in 0..input.active_len {
        let argument = input.photon_wave_number * input.radii[index];
        validate_finite_real("xray_bessel_argument", argument)?;
        let sample = if argument.abs() < 1.0 {
            [
                xray_bessel_series(argument, 0),
                xray_bessel_series(argument, 1),
                xray_bessel_series(argument, 2),
            ]
        } else {
            xray_bessel_formula(argument)
        };
        for (row, value) in sample.into_iter().enumerate() {
            values[(row, index)] = value;
        }
    }

    Ok(XsphXrayBesselTable { values })
}

/// Port of FEFF `XSPH/xsect.f90` regular-solution normalization.
///
/// After the regular `dfovrg` solve and `phamp`, FEFF computes the
/// kappa-signed lower-component `factor`, relativistic `dum1`, and
/// `xfnorm = dum1 / temp`, then multiplies both radial components by
/// `xfnorm` over `1:ilast`.
pub fn xsph_xsect_regular_solution(
    input: XsphXsectRegularSolutionInput<'_>,
) -> Result<XsphXsectRegularSolution, XsphError> {
    validate_xsect_regular_solution_input(&input)?;

    let scales = xsph_xsect_relativistic_scales(input.wave_number, input.final_kappa)?;
    let regular_solution_scale = scales.relativistic_scale / input.phase_amplitude;
    validate_finite_complex("xsect_regular_solution_scale", 0, regular_solution_scale)?;

    let regular_large = input
        .regular_large
        .iter()
        .take(input.active_len)
        .map(|&value| value * regular_solution_scale)
        .collect::<Array1<_>>();
    let regular_small = input
        .regular_small
        .iter()
        .take(input.active_len)
        .map(|&value| value * regular_solution_scale)
        .collect::<Array1<_>>();

    for index in 0..input.active_len {
        validate_finite_complex("xsect_regular_large", index, regular_large[index])?;
        validate_finite_complex("xsect_regular_small", index, regular_small[index])?;
    }

    Ok(XsphXsectRegularSolution {
        small_component_factor: scales.small_component_factor,
        relativistic_scale: scales.relativistic_scale,
        regular_solution_scale,
        regular_large,
        regular_small,
    })
}

/// Compose the regular FEFF `XSPH/xsect.f90` radial channel.
///
/// This runs the regular `dfovrg` branch, matches the muffin-tin phase with
/// `phamp`, then applies the `xfnorm` scale used by the cross-section radial
/// integrals. It is the regular-channel building block for a source-backed
/// `xsect.dat` driver.
pub fn xsph_xsect_regular_channel(
    input: XsphXsectRegularChannelInput<'_>,
) -> Result<XsphXsectRegularChannel, XsphError> {
    validate_finite_complex("xsect_regular_channel_wave_number", 0, input.wave_number)?;

    let zero = Complex::new(0.0, 0.0);
    let target_kappa = input.solver.target_kappa;
    let muffin_tin_radius = input.solver.muffin_tin_radius;
    let regular_input = crate::FovrgDiracSolverInput {
        irregular: false,
        muffin_tin_large_component: zero,
        muffin_tin_small_component: zero,
        ..input.solver
    };
    let regular_solution = fovrg_dirac_solver(regular_input)?;
    let active_len =
        regular_solution
            .target_last_index
            .checked_add(1)
            .ok_or(XsphError::SizeOutOfRange {
                name: "xsect_regular_channel_active_len",
                value: regular_solution.target_last_index,
            })?;

    let phase = xsph_regular_phase(XsphRegularPhaseInput {
        muffin_tin_radius,
        wave_number: input.wave_number,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: target_kappa,
    })?;
    let normalized_solution = xsph_xsect_regular_solution(XsphXsectRegularSolutionInput {
        wave_number: input.wave_number,
        phase_amplitude: phase.phase_amplitude,
        final_kappa: target_kappa,
        regular_large: regular_solution.large_component.view(),
        regular_small: regular_solution.small_component.view(),
        active_len,
    })?;

    Ok(XsphXsectRegularChannel {
        regular_solution,
        phase,
        normalized_solution,
    })
}

/// Port of FEFF `XSPH/xsect.f90` irregular-solution boundary setup.
///
/// For complex-energy points FEFF initializes the irregular `dfovrg` solve with
/// `N*cos(ph0)+J*sin(ph0)` at the muffin-tin boundary, using the same
/// kappa-signed `factor` and relativistic `dum1` as the regular branch.
pub fn xsph_xsect_irregular_initial_condition(
    input: XsphXsectIrregularInitialConditionInput,
) -> Result<XsphXsectIrregularInitialCondition, XsphError> {
    validate_xsect_irregular_initial_condition_input(input)?;

    let scales = xsph_xsect_relativistic_scales(input.wave_number, input.final_kappa)?;
    let cos_phase = input.phase_shift.cos();
    let sin_phase = input.phase_shift.sin();
    validate_finite_complex("xsect_irregular_cos_phase", 0, cos_phase)?;
    validate_finite_complex("xsect_irregular_sin_phase", 0, sin_phase)?;

    let radius_scale = input.muffin_tin_radius * scales.relativistic_scale;
    let large_component =
        (input.neumann_l * cos_phase + input.bessel_j_l * sin_phase) * radius_scale;
    let small_component = (input.neumann_l_plus_1 * cos_phase
        + input.bessel_j_l_plus_1 * sin_phase)
        * scales.small_component_factor
        * radius_scale;

    validate_finite_complex("xsect_irregular_large_component", 0, large_component)?;
    validate_finite_complex("xsect_irregular_small_component", 0, small_component)?;

    Ok(XsphXsectIrregularInitialCondition {
        large_component,
        small_component,
        small_component_factor: scales.small_component_factor,
        relativistic_scale: scales.relativistic_scale,
    })
}

/// Port of FEFF `XSPH/xsect.f90` irregular post-`dfovrg` transform.
///
/// FEFF replaces the outgoing-Hankel radial solution returned by `dfovrg` with
/// `N = i*R - exp(i*ph0)*H`, where `R` is the already normalized regular
/// solution and `H` is the irregular solve result.
pub fn xsph_xsect_irregular_transform(
    input: XsphXsectIrregularTransformInput<'_>,
) -> Result<XsphXsectIrregularTransform, XsphError> {
    validate_xsect_irregular_transform_input(&input)?;

    let imaginary_unit = Complex::new(0.0, 1.0);
    let phase_factor = (imaginary_unit * input.phase_shift).exp();
    validate_finite_complex("xsect_irregular_phase_factor", 0, phase_factor)?;

    let irregular_large = (0..input.active_len)
        .map(|index| {
            imaginary_unit * input.regular_large[index]
                - phase_factor * input.irregular_large[index]
        })
        .collect::<Array1<_>>();
    let irregular_small = (0..input.active_len)
        .map(|index| {
            imaginary_unit * input.regular_small[index]
                - phase_factor * input.irregular_small[index]
        })
        .collect::<Array1<_>>();

    for index in 0..input.active_len {
        validate_finite_complex("xsect_irregular_large", index, irregular_large[index])?;
        validate_finite_complex("xsect_irregular_small", index, irregular_small[index])?;
    }

    Ok(XsphXsectIrregularTransform {
        phase_factor,
        irregular_large,
        irregular_small,
    })
}

/// Compose the irregular FEFF `XSPH/xsect.f90` radial channel.
///
/// FEFF builds the irregular muffin-tin boundary from the regular `phamp`
/// result, runs `dfovrg` with `irr > 0`, then transforms the outgoing Hankel
/// solution into the irregular rows used by the central cross-section radial
/// integral.
pub fn xsph_xsect_irregular_channel(
    input: XsphXsectIrregularChannelInput<'_, '_>,
) -> Result<XsphXsectIrregularChannel, XsphError> {
    validate_finite_complex("xsect_irregular_channel_wave_number", 0, input.wave_number)?;

    let target_kappa = input.solver.target_kappa;
    let phase = &input.regular_channel.phase;
    let initial_condition =
        xsph_xsect_irregular_initial_condition(XsphXsectIrregularInitialConditionInput {
            muffin_tin_radius: input.solver.muffin_tin_radius,
            phase_shift: phase.phase_shift,
            wave_number: input.wave_number,
            final_kappa: target_kappa,
            bessel_j_l: phase.bessel_j_large,
            neumann_l: phase.neumann_large,
            bessel_j_l_plus_1: phase.bessel_j_small,
            neumann_l_plus_1: phase.neumann_small,
        })?;

    let irregular_input = crate::FovrgDiracSolverInput {
        irregular: true,
        muffin_tin_large_component: initial_condition.large_component,
        muffin_tin_small_component: initial_condition.small_component,
        ..input.solver
    };
    let irregular_solution = fovrg_dirac_solver(irregular_input)?;
    let active_len = input
        .regular_channel
        .normalized_solution
        .regular_large
        .len();
    let transformed_solution = xsph_xsect_irregular_transform(XsphXsectIrregularTransformInput {
        phase_shift: phase.phase_shift,
        regular_large: input
            .regular_channel
            .normalized_solution
            .regular_large
            .view(),
        regular_small: input
            .regular_channel
            .normalized_solution
            .regular_small
            .view(),
        irregular_large: irregular_solution.large_component.view(),
        irregular_small: irregular_solution.small_component.view(),
        active_len,
    })?;

    Ok(XsphXsectIrregularChannel {
        initial_condition,
        irregular_solution,
        transformed_solution,
    })
}

/// Port of FEFF `XSPH/xsect.f90` positive-omega output finalization.
///
/// At the end of each positive-energy row FEFF scales `xsnorm` and `xsec` by
/// the relativistic `prefac`, then normalizes the first reduced matrix-element
/// channels by `sqrt(prefac * 2*ck) / sqrt(xsnorm)` and applies the central atom
/// phase factor `exp(i*phx)`.
pub fn xsph_xsect_output_normalization(
    input: XsphXsectOutputNormalizationInput<'_>,
) -> Result<XsphXsectOutputNormalization, XsphError> {
    validate_xsect_output_normalization_input(&input)?;

    let prefactor = 4.0 * std::f64::consts::PI / super::XSPH_FINE_STRUCTURE_ALPHA
        * super::XSPH_BOHR_ANGSTROM.powi(2)
        / input.photon_energy;
    validate_finite_real("xsect_prefactor", prefactor)?;

    let spectrum_norm = input.spectrum_norm * prefactor * 2.0 * input.wave_number.norm();
    if !spectrum_norm.is_finite() || spectrum_norm <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "xsect_scaled_spectrum_norm",
            value: spectrum_norm,
        });
    }
    let spectrum_norm_sqrt = spectrum_norm.sqrt();
    validate_finite_real("xsect_spectrum_norm_sqrt", spectrum_norm_sqrt)?;

    let cross_section_scale = input.wave_number * (2.0 * prefactor);
    validate_finite_complex("xsect_cross_section_scale", 0, cross_section_scale)?;
    let cross_section = input.cross_section * cross_section_scale;
    validate_finite_complex("xsect_cross_section", 0, cross_section)?;

    let mut reduced_matrix_root_scale = cross_section_scale.sqrt();
    if reduced_matrix_root_scale.im < 0.0 {
        reduced_matrix_root_scale = -reduced_matrix_root_scale;
    }
    validate_finite_complex(
        "xsect_reduced_matrix_root_scale",
        0,
        reduced_matrix_root_scale,
    )?;
    let reduced_matrix_scale = reduced_matrix_root_scale / spectrum_norm_sqrt;
    validate_finite_complex("xsect_reduced_matrix_scale", 0, reduced_matrix_scale)?;

    let imaginary_unit = Complex::new(0.0, 1.0);
    let reduced_matrix_elements = (0..input.active_channel_count)
        .map(|index| {
            let phase_factor = (imaginary_unit * input.phase_shifts[index]).exp();
            input.reduced_matrix_elements[index] * reduced_matrix_scale * phase_factor
        })
        .collect::<Array1<_>>();
    for index in 0..input.active_channel_count {
        validate_finite_complex(
            "xsect_reduced_matrix_elements",
            index,
            reduced_matrix_elements[index],
        )?;
    }

    Ok(XsphXsectOutputNormalization {
        prefactor,
        spectrum_norm,
        spectrum_norm_sqrt,
        cross_section,
        reduced_matrix_root_scale,
        reduced_matrix_scale,
        reduced_matrix_elements,
    })
}

/// Port of FEFF `XSPH/xsect.f90` real/imaginary `fscf` integral combination.
///
/// FEFF evaluates the radial integral once with `dble(fscf)` and, when present,
/// once with `dimag(fscf)`. The second pass combines magnitudes without changing
/// the phase of the dominant complex value. This helper preserves the exact
/// zero and dominance branches used in both `xsect` radial-integral sections.
pub fn xsph_xsect_fscf_integral(
    input: XsphXsectFscfIntegralInput,
) -> Result<XsphXsectFscfIntegral, XsphError> {
    validate_finite_complex("xsect_fscf_accumulated", 0, input.accumulated)?;
    validate_finite_complex("xsect_fscf_contribution", 0, input.contribution)?;

    if input.first_component {
        return Ok(XsphXsectFscfIntegral {
            value: input.contribution,
            selection: XsphXsectFscfSelection::FirstComponent,
            scale: 1.0,
        });
    }

    let accumulated_abs = input.accumulated.norm();
    let contribution_abs = input.contribution.norm();
    let (value, selection, scale) = if accumulated_abs == 0.0 {
        (
            input.contribution,
            XsphXsectFscfSelection::AccumulatedZero,
            1.0,
        )
    } else if contribution_abs == 0.0 {
        (
            input.accumulated,
            XsphXsectFscfSelection::ContributionZero,
            1.0,
        )
    } else if contribution_abs < accumulated_abs {
        let ratio = contribution_abs / accumulated_abs;
        let scale = (1.0 + ratio * ratio).sqrt();
        (
            input.accumulated * scale,
            XsphXsectFscfSelection::AccumulatedDominant,
            scale,
        )
    } else {
        let ratio = accumulated_abs / contribution_abs;
        let scale = (1.0 + ratio * ratio).sqrt();
        (
            input.contribution * scale,
            XsphXsectFscfSelection::ContributionDominant,
            scale,
        )
    };

    validate_finite_real("xsect_fscf_scale", scale)?;
    validate_finite_complex("xsect_fscf_integral", 0, value)?;
    Ok(XsphXsectFscfIntegral {
        value,
        selection,
        scale,
    })
}

/// Port of FEFF `XSPH/xsect.f90` direct transition accumulation.
///
/// For the first spin-orbit pass FEFF optionally stores `rkk(ie,ind)` and
/// `phx(ind)`, increments `xsnorm` by `abs(xirf)**2/(2*kx+1)`, and updates
/// `xsec` with `-aa*bmat`, where `aa = -i*xirf**2`.
pub fn xsph_xsect_direct_transition(
    input: XsphXsectDirectTransitionInput,
) -> Result<XsphXsectDirectTransition, XsphError> {
    validate_finite_complex("xsect_transition_integral", 0, input.radial_integral)?;
    validate_finite_complex("xsect_transition_phase_shift", 0, input.phase_shift)?;
    validate_finite_complex("xsect_transition_angular_weight", 0, input.angular_weight)?;
    validate_finite_real("xsect_spectrum_norm", input.spectrum_norm)?;
    validate_finite_complex("xsect_cross_section", 0, input.cross_section)?;

    let multipole_order = xsect_transition_multipole_order(input.multipole);
    let spectrum_norm_increment =
        input.radial_integral.norm_sqr() / (2 * multipole_order + 1) as Real;
    let imaginary_unit = Complex::new(0.0, 1.0);
    let cross_section_increment =
        imaginary_unit * input.radial_integral * input.radial_integral * input.angular_weight;
    let spectrum_norm = input.spectrum_norm + spectrum_norm_increment;
    let cross_section = input.cross_section + cross_section_increment;
    let store_reduced_matrix =
        xsect_stores_reduced_matrix(input.multipole, input.selected_higher_multipole);

    validate_finite_real("xsect_spectrum_norm_increment", spectrum_norm_increment)?;
    validate_finite_real("xsect_updated_spectrum_norm", spectrum_norm)?;
    validate_finite_complex("xsect_cross_section_increment", 0, cross_section_increment)?;
    validate_finite_complex("xsect_updated_cross_section", 0, cross_section)?;

    Ok(XsphXsectDirectTransition {
        store_reduced_matrix,
        reduced_matrix_element: store_reduced_matrix.then_some(input.radial_integral),
        phase_shift: store_reduced_matrix.then_some(input.phase_shift),
        spectrum_norm_increment,
        spectrum_norm,
        cross_section_increment,
        cross_section,
    })
}

/// Direct transition accumulation using source-backed traced `bcoef` weights.
///
/// This selects FEFF `bmat(0,isp,ind,0,isp,ind)` from the eight-slot traced
/// table before delegating to [`xsph_xsect_direct_transition`].
pub fn xsph_xsect_bcoef_direct_transition(
    input: XsphXsectBcoefDirectTransitionInput<'_>,
) -> Result<XsphXsectDirectTransition, XsphError> {
    validate_xsect_bcoef_transition_index(input.transition_index_1based)?;
    validate_xsect_bcoef_diagonal_weights(input.diagonal_weights)?;
    let angular_weight = input.diagonal_weights[input.transition_index_1based - 1];

    xsph_xsect_direct_transition(XsphXsectDirectTransitionInput {
        multipole: input.multipole,
        selected_higher_multipole: input.selected_higher_multipole,
        radial_integral: input.radial_integral,
        phase_shift: input.phase_shift,
        angular_weight,
        spectrum_norm: input.spectrum_norm,
        cross_section: input.cross_section,
    })
}

/// Direct transition accumulation plus FEFF `rkk/phx` row-workspace update.
///
/// FEFF stores `rkk(ie,ind)` and `phx(ind)` only when the selected multipole
/// contributes to the final reduced-matrix output. This wrapper makes that
/// workspace update explicit while keeping the angular weight source-backed by
/// the traced `bcoef` table.
pub fn xsph_xsect_bcoef_direct_transition_update(
    input: XsphXsectBcoefDirectTransitionUpdateInput<'_>,
) -> Result<XsphXsectBcoefDirectTransitionUpdate, XsphError> {
    validate_xsect_bcoef_transition_index(input.transition_index_1based)?;
    validate_xsect_reduced_matrix_workspace(input.reduced_matrix_elements, input.phase_shifts)?;

    let transition = xsph_xsect_bcoef_direct_transition(XsphXsectBcoefDirectTransitionInput {
        multipole: input.multipole,
        selected_higher_multipole: input.selected_higher_multipole,
        transition_index_1based: input.transition_index_1based,
        diagonal_weights: input.diagonal_weights,
        radial_integral: input.radial_integral,
        phase_shift: input.phase_shift,
        spectrum_norm: input.spectrum_norm,
        cross_section: input.cross_section,
    })?;

    let mut reduced_matrix_elements =
        xsect_transition_workspace_copy(input.reduced_matrix_elements);
    let mut phase_shifts = xsect_transition_workspace_copy(input.phase_shifts);
    if transition.store_reduced_matrix {
        let index = input.transition_index_1based - 1;
        if let Some(reduced_matrix_element) = transition.reduced_matrix_element {
            reduced_matrix_elements[index] = reduced_matrix_element;
        }
        if let Some(phase_shift) = transition.phase_shift {
            phase_shifts[index] = phase_shift;
        }
    }

    Ok(XsphXsectBcoefDirectTransitionUpdate {
        transition,
        spectrum_norm: transition.spectrum_norm,
        cross_section: transition.cross_section,
        reduced_matrix_elements,
        phase_shifts,
    })
}

/// Port of FEFF `XSPH/xsect.f90` diagonal central cross-section accumulation.
///
/// After the `radint(ifl=2)` and `fscf` combination block, FEFF updates
/// `xsec(ie)` with `-xirf * bmat(0,isp,ind,0,isp,ind)` only on the ordinary
/// pass (`ic3 = 0`). Spin-orbit-removed retry passes skip this diagonal update.
pub fn xsph_xsect_central_cross_section(
    input: XsphXsectCentralCrossSectionInput,
) -> Result<Option<XsphXsectCentralCrossSection>, XsphError> {
    if input.spin_orbit_removed_pass {
        return Ok(None);
    }

    Ok(Some(xsect_central_cross_section_update(
        input.radial_integral,
        input.angular_weight,
        input.cross_section,
    )?))
}

fn xsect_central_cross_section_update(
    radial_integral: Complex,
    angular_weight: Complex,
    cross_section: Complex,
) -> Result<XsphXsectCentralCrossSection, XsphError> {
    validate_finite_complex("xsect_central_cross_integral", 0, radial_integral)?;
    validate_finite_complex("xsect_central_angular_weight", 0, angular_weight)?;
    validate_finite_complex("xsect_cross_section", 0, cross_section)?;

    let cross_section_increment = -radial_integral * angular_weight;
    let cross_section = cross_section + cross_section_increment;

    validate_finite_complex(
        "xsect_central_cross_section_increment",
        0,
        cross_section_increment,
    )?;
    validate_finite_complex("xsect_updated_cross_section", 0, cross_section)?;

    Ok(XsphXsectCentralCrossSection {
        cross_section_increment,
        cross_section,
    })
}

/// Central cross-section accumulation using source-backed traced `bcoef` weights.
///
/// This selects FEFF `bmat(0,isp,ind,0,isp,ind)` from the same traced diagonal
/// table used by direct transitions before applying the `radint(ifl=2)` diagonal
/// central-atom update.
pub fn xsph_xsect_bcoef_central_cross_section(
    input: XsphXsectBcoefCentralCrossSectionInput<'_>,
) -> Result<Option<XsphXsectCentralCrossSection>, XsphError> {
    if input.spin_orbit_removed_pass {
        return Ok(None);
    }
    validate_xsect_bcoef_transition_index(input.transition_index_1based)?;
    validate_xsect_bcoef_diagonal_weights(input.diagonal_weights)?;
    let angular_weight = input.diagonal_weights[input.transition_index_1based - 1];

    Ok(Some(xsect_central_cross_section_update(
        input.radial_integral,
        angular_weight,
        input.cross_section,
    )?))
}

/// Ordinary FEFF `xsect` transition row using source-backed traced `bcoef`.
///
/// This composes the ordinary-pass `ic3 = 0` row in FEFF order: first the
/// direct reduced-matrix / `xsnorm` / direct `xsec` update, then the diagonal
/// central cross-section update from `radint(ifl=2)`.
pub fn xsph_xsect_bcoef_ordinary_row(
    input: XsphXsectBcoefOrdinaryRowInput<'_>,
) -> Result<XsphXsectBcoefOrdinaryRow, XsphError> {
    let direct_transition =
        xsph_xsect_bcoef_direct_transition_update(XsphXsectBcoefDirectTransitionUpdateInput {
            multipole: input.multipole,
            selected_higher_multipole: input.selected_higher_multipole,
            transition_index_1based: input.transition_index_1based,
            diagonal_weights: input.diagonal_weights,
            radial_integral: input.reduced_matrix_integral,
            phase_shift: input.phase_shift,
            spectrum_norm: input.spectrum_norm,
            cross_section: input.cross_section,
            reduced_matrix_elements: input.reduced_matrix_elements,
            phase_shifts: input.phase_shifts,
        })?;

    let angular_weight = input.diagonal_weights[input.transition_index_1based - 1];
    let central_cross_section = xsect_central_cross_section_update(
        input.central_cross_integral,
        angular_weight,
        direct_transition.cross_section,
    )?;

    Ok(XsphXsectBcoefOrdinaryRow {
        spectrum_norm: direct_transition.spectrum_norm,
        cross_section: central_cross_section.cross_section,
        reduced_matrix_elements: direct_transition.reduced_matrix_elements.clone(),
        phase_shifts: direct_transition.phase_shifts.clone(),
        direct_transition,
        central_cross_section,
    })
}

/// Nonstandard-potential ordinary FEFF `xsect` transition row from radial channels.
///
/// This covers the `izstd <= 0` branch used by ordinary source-backed XSPH
/// runs: FEFF calls `radint(ifl=1)` for the reduced matrix element,
/// `radint(ifl=2)` for the central cross section, then applies the traced
/// `bcoef` ordinary-row updates.
pub fn xsph_xsect_bcoef_nonstandard_channel_row(
    input: XsphXsectBcoefNonstandardChannelRowInput<'_>,
) -> Result<XsphXsectBcoefNonstandardChannelRow, XsphError> {
    let radial = xsect_bcoef_nonstandard_radial_components(
        &input,
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
    )?;

    let reduced_matrix_integral =
        radial.reduced_radial_integral.value * radial.reduced_radial_pass.post_radint_scale;
    let central_cross_value =
        radial.central_cross_integral.value * radial.central_radial_pass.post_radint_scale;
    let row = xsph_xsect_bcoef_ordinary_row(XsphXsectBcoefOrdinaryRowInput {
        multipole: input.transition.multipole,
        selected_higher_multipole: input.selected_higher_multipole,
        transition_index_1based: input.transition.transition_index_1based,
        diagonal_weights: input.diagonal_weights,
        reduced_matrix_integral,
        central_cross_integral: central_cross_value,
        phase_shift: input.regular_channel.phase.phase_shift,
        spectrum_norm: input.spectrum_norm,
        cross_section: input.cross_section,
        reduced_matrix_elements: input.reduced_matrix_elements,
        phase_shifts: input.phase_shifts,
    })?;

    Ok(XsphXsectBcoefNonstandardChannelRow {
        reduced_radial_pass: radial.reduced_radial_pass,
        central_radial_pass: radial.central_radial_pass,
        reduced_radial_integral: radial.reduced_radial_integral,
        central_cross_integral: radial.central_cross_integral,
        row,
    })
}

/// Standard-potential ordinary FEFF `xsect` transition row from `fscf` passes.
///
/// This covers the `izstd > 0` branch after the screened field has been
/// prepared: FEFF evaluates real and imaginary `fscf` radial passes, applies
/// the standard-atom post-`radint` scales, combines component magnitudes, then
/// applies the same traced `bcoef` ordinary-row updates as the nonstandard
/// branch.
pub fn xsph_xsect_bcoef_standard_channel_row(
    input: XsphXsectBcoefStandardChannelRowInput<'_>,
) -> Result<XsphXsectBcoefStandardChannelRow, XsphError> {
    let radial = xsect_bcoef_standard_radial_components(
        &input,
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
    )?;

    let row = xsph_xsect_bcoef_ordinary_row(XsphXsectBcoefOrdinaryRowInput {
        multipole: input.transition.multipole,
        selected_higher_multipole: input.selected_higher_multipole,
        transition_index_1based: input.transition.transition_index_1based,
        diagonal_weights: input.diagonal_weights,
        reduced_matrix_integral: radial.reduced_matrix_integral,
        central_cross_integral: radial.central_cross_integral,
        phase_shift: input.regular_channel.phase.phase_shift,
        spectrum_norm: input.spectrum_norm,
        cross_section: input.cross_section,
        reduced_matrix_elements: input.reduced_matrix_elements,
        phase_shifts: input.phase_shifts,
    })?;

    Ok(XsphXsectBcoefStandardChannelRow {
        fscf_weights: radial.fscf_weights,
        reduced_radial_pass: radial.reduced_radial_pass,
        central_radial_pass: radial.central_radial_pass,
        reduced_component_integrals: radial.reduced_component_integrals,
        reduced_fscf_integrals: radial.reduced_fscf_integrals,
        central_component_integrals: radial.central_component_integrals,
        central_fscf_integrals: radial.central_fscf_integrals,
        row,
    })
}

fn xsect_bcoef_standard_radial_components<'a, 'b>(
    input: &'b XsphXsectBcoefStandardChannelRowInput<'a>,
    central_branch: XsphRadialCrossIntegralBranch<'b>,
) -> Result<XsectBcoefStandardRadialComponents, XsphError>
where
    'a: 'b,
{
    let active_len = input
        .regular_channel
        .normalized_solution
        .regular_large
        .len();
    let fscf_weights = xsph_xsect_fscf_weights(XsphXsectFscfWeightsInput {
        standard_potential: true,
        fscf: input.fscf,
        active_len,
    })?;

    let reduced_radial_pass = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::ReducedMatrixElement,
        standard_potential: true,
        photon_wave_number: input.photon_wave_number,
        screened_field_scale: input.screened_field_scale,
    })?;
    let reduced_mode = xsect_radial_integral_mode_from_ifl(reduced_radial_pass.feff_ifl)?;
    let mut reduced_component_integrals = Vec::with_capacity(fscf_weights.components.len());
    let mut reduced_fscf_integrals = Vec::with_capacity(fscf_weights.components.len());
    let mut reduced_matrix_integral = Complex::new(0.0, 0.0);
    for (component_index, component) in fscf_weights.components.iter().enumerate() {
        let integral = xsph_xsect_weighted_radial_integral(XsphXsectWeightedRadialIntegralInput {
            mode: reduced_mode,
            multipole: input.transition.multipole,
            initial_kappa: input.initial_kappa,
            final_kappa: input.transition.final_kappa,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large_regular: input
                .regular_channel
                .normalized_solution
                .regular_large
                .view(),
            final_small_regular: input
                .regular_channel
                .normalized_solution
                .regular_small
                .view(),
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            log_step: input.log_step,
            radial_weights: component.weights.view(),
            active_len,
        })?;
        let combined = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
            accumulated: reduced_matrix_integral,
            contribution: integral.integral.value * reduced_radial_pass.post_radint_scale,
            first_component: component_index == 0,
        })?;
        reduced_matrix_integral = combined.value;
        reduced_component_integrals.push(integral);
        reduced_fscf_integrals.push(combined);
    }

    let central_radial_pass = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::CentralCrossSection,
        standard_potential: true,
        photon_wave_number: input.photon_wave_number,
        screened_field_scale: input.screened_field_scale,
    })?;
    let central_mode = xsect_radial_integral_mode_from_ifl(central_radial_pass.feff_ifl)?;
    let mut central_component_integrals = Vec::with_capacity(fscf_weights.components.len());
    let mut central_fscf_integrals = Vec::with_capacity(fscf_weights.components.len());
    let mut central_cross_integral = Complex::new(0.0, 0.0);
    for (component_index, component) in fscf_weights.components.iter().enumerate() {
        let integral =
            xsph_xsect_weighted_radial_cross_integral(XsphXsectWeightedRadialCrossIntegralInput {
                mode: central_mode,
                branch: central_branch.clone(),
                multipole: input.transition.multipole,
                initial_kappa: input.initial_kappa,
                final_kappa: input.transition.final_kappa,
                initial_large: input.initial_large,
                initial_small: input.initial_small,
                final_large_regular: input
                    .regular_channel
                    .normalized_solution
                    .regular_large
                    .view(),
                final_small_regular: input
                    .regular_channel
                    .normalized_solution
                    .regular_small
                    .view(),
                final_large_irregular: input
                    .irregular_channel
                    .transformed_solution
                    .irregular_large
                    .view(),
                final_small_irregular: input
                    .irregular_channel
                    .transformed_solution
                    .irregular_small
                    .view(),
                xray_bessel: input.xray_bessel,
                radii: input.radii,
                log_step: input.log_step,
                regular_weights: component.weights.view(),
                irregular_weights: component.weights.view(),
                active_len,
            })?;
        let combined = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
            accumulated: central_cross_integral,
            contribution: integral.integral.value * central_radial_pass.post_radint_scale,
            first_component: component_index == 0,
        })?;
        central_cross_integral = combined.value;
        central_component_integrals.push(integral);
        central_fscf_integrals.push(combined);
    }

    Ok(XsectBcoefStandardRadialComponents {
        fscf_weights,
        reduced_radial_pass,
        central_radial_pass,
        reduced_component_integrals,
        reduced_fscf_integrals,
        reduced_matrix_integral,
        central_component_integrals,
        central_fscf_integrals,
        central_cross_integral,
    })
}

/// Standard-potential ordinary FEFF `xsect` energy row from transition channels.
///
/// This folds the source-backed standard transition rows in FEFF traversal
/// order, then applies the positive-omega output normalization that produces
/// the per-spin `xsnorm`, `xsec`, and `rkk` rows consumed by `xsect.dat` spin
/// merging. For spin-polarized cross terms it also reuses the saved same-`l`
/// regular/irregular state to apply the FEFF spin-orbit retry branches.
pub fn xsph_xsect_bcoef_standard_energy_row(
    input: XsphXsectBcoefStandardEnergyRowInput<'_>,
) -> Result<XsphXsectBcoefStandardEnergyRow, XsphError> {
    let field = XsphXsectBcoefStandardTransitionField {
        screened_field_scale: input.screened_field_scale,
        fscf: input.fscf,
    };
    xsect_bcoef_standard_energy_row_impl(
        XsectBcoefStandardEnergyRowBaseInput {
            transitions: input.transitions,
            regular_channels: input.regular_channels,
            irregular_channels: input.irregular_channels,
            selected_higher_multipole: input.selected_higher_multipole,
            initial_kappa: input.initial_kappa,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            log_step: input.log_step,
            photon_wave_number: input.photon_wave_number,
            diagonal_weights: input.diagonal_weights,
            spin_polarized_cross_terms: input.spin_polarized_cross_terms,
            orbital_l: input.orbital_l,
            trace_weights: input.trace_weights,
            spin_orbit_removed_regular_channels: input.spin_orbit_removed_regular_channels,
            spin_orbit_removed_irregular_channels: input.spin_orbit_removed_irregular_channels,
            photon_energy: input.photon_energy,
            wave_number: input.wave_number,
            active_channel_count: input.active_channel_count,
        },
        |_| field,
    )
}

/// Standard-potential FEFF `xsect` energy row with per-transition `fscf`.
///
/// FEFF uses a screened `phiscf` field only for standard-atom dipole rows; other
/// multipoles in the same transition traversal keep the unity field. This entry
/// point preserves that per-transition branch while sharing the same row folding
/// and spin-orbit retry implementation as [`xsph_xsect_bcoef_standard_energy_row`].
pub fn xsph_xsect_bcoef_standard_energy_row_with_transition_fields(
    input: XsphXsectBcoefStandardEnergyRowFieldsInput<'_>,
) -> Result<XsphXsectBcoefStandardEnergyRow, XsphError> {
    validate_active_len(
        "xsect_standard_transition_fields",
        input.transition_fields.len(),
        input.transitions.len(),
    )?;
    xsect_bcoef_standard_energy_row_impl(
        XsectBcoefStandardEnergyRowBaseInput {
            transitions: input.transitions,
            regular_channels: input.regular_channels,
            irregular_channels: input.irregular_channels,
            selected_higher_multipole: input.selected_higher_multipole,
            initial_kappa: input.initial_kappa,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            log_step: input.log_step,
            photon_wave_number: input.photon_wave_number,
            diagonal_weights: input.diagonal_weights,
            spin_polarized_cross_terms: input.spin_polarized_cross_terms,
            orbital_l: input.orbital_l,
            trace_weights: input.trace_weights,
            spin_orbit_removed_regular_channels: input.spin_orbit_removed_regular_channels,
            spin_orbit_removed_irregular_channels: input.spin_orbit_removed_irregular_channels,
            photon_energy: input.photon_energy,
            wave_number: input.wave_number,
            active_channel_count: input.active_channel_count,
        },
        |index| input.transition_fields[index],
    )
}

#[derive(Debug, Clone, Copy)]
struct XsectBcoefStandardEnergyRowBaseInput<'a> {
    transitions: &'a [XsphXsectTransition],
    regular_channels: &'a [XsphXsectRegularChannel],
    irregular_channels: &'a [XsphXsectIrregularChannel],
    selected_higher_multipole: Option<XsphTransitionMultipole>,
    initial_kappa: i32,
    initial_large: ArrayView1<'a, Real>,
    initial_small: ArrayView1<'a, Real>,
    xray_bessel: ArrayView2<'a, Real>,
    radii: ArrayView1<'a, Real>,
    log_step: Real,
    photon_wave_number: Real,
    diagonal_weights: ArrayView1<'a, Complex>,
    spin_polarized_cross_terms: bool,
    orbital_l: ArrayView1<'a, i32>,
    trace_weights: ArrayView2<'a, Complex>,
    spin_orbit_removed_regular_channels: Option<&'a [XsphXsectRegularChannel]>,
    spin_orbit_removed_irregular_channels: Option<&'a [XsphXsectIrregularChannel]>,
    photon_energy: Real,
    wave_number: Complex,
    active_channel_count: usize,
}

fn xsect_bcoef_standard_energy_row_impl<'a>(
    input: XsectBcoefStandardEnergyRowBaseInput<'a>,
    transition_field: impl Fn(usize) -> XsphXsectBcoefStandardTransitionField<'a>,
) -> Result<XsphXsectBcoefStandardEnergyRow, XsphError> {
    validate_active_len("xsect_transitions", input.transitions.len(), 1)?;
    validate_active_len(
        "xsect_regular_channels",
        input.regular_channels.len(),
        input.transitions.len(),
    )?;
    validate_active_len(
        "xsect_irregular_channels",
        input.irregular_channels.len(),
        input.transitions.len(),
    )?;
    if input.active_channel_count == 0 || input.active_channel_count > XSECT_BCOEF_TRANSITION_SLOTS
    {
        return Err(XsphError::SizeOutOfRange {
            name: "xsect_active_channel_count",
            value: input.active_channel_count,
        });
    }
    validate_xsect_bcoef_diagonal_weights(input.diagonal_weights)?;
    let cross_term_channels = if input.spin_polarized_cross_terms {
        validate_active_len(
            "xsect_orbital_l",
            input.orbital_l.len(),
            input.active_channel_count,
        )?;
        validate_xsect_bcoef_trace_weights(input.trace_weights)?;
        let regular_channels =
            input
                .spin_orbit_removed_regular_channels
                .ok_or(XsphError::LengthTooShort {
                    name: "xsect_spin_orbit_removed_regular_channels",
                    required: input.transitions.len(),
                    actual: 0,
                })?;
        let irregular_channels =
            input
                .spin_orbit_removed_irregular_channels
                .ok_or(XsphError::LengthTooShort {
                    name: "xsect_spin_orbit_removed_irregular_channels",
                    required: input.transitions.len(),
                    actual: 0,
                })?;
        validate_active_len(
            "xsect_spin_orbit_removed_regular_channels",
            regular_channels.len(),
            input.transitions.len(),
        )?;
        validate_active_len(
            "xsect_spin_orbit_removed_irregular_channels",
            irregular_channels.len(),
            input.transitions.len(),
        )?;
        Some((regular_channels, irregular_channels))
    } else {
        None
    };

    let mut spectrum_norm = 0.0;
    let mut cross_section = Complex::new(0.0, 0.0);
    let mut reduced_matrix_elements = Array1::<Complex>::zeros(XSECT_BCOEF_TRANSITION_SLOTS);
    let mut phase_shifts = Array1::<Complex>::zeros(XSECT_BCOEF_TRANSITION_SLOTS);
    let mut transition_rows = Vec::with_capacity(input.transitions.len());
    let mut cross_term_updates = Vec::new();
    let mut saved_cross_term_state: Option<XsphXsectCrossTermState> = None;

    for (index, transition) in input.transitions.iter().copied().enumerate() {
        let field = transition_field(index);
        let row = xsph_xsect_bcoef_standard_channel_row(XsphXsectBcoefStandardChannelRowInput {
            transition,
            selected_higher_multipole: input.selected_higher_multipole,
            initial_kappa: input.initial_kappa,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            regular_channel: &input.regular_channels[index],
            irregular_channel: &input.irregular_channels[index],
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            log_step: input.log_step,
            photon_wave_number: input.photon_wave_number,
            screened_field_scale: field.screened_field_scale,
            fscf: field.fscf,
            diagonal_weights: input.diagonal_weights,
            spectrum_norm,
            cross_section,
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
        })?;
        spectrum_norm = row.row.spectrum_norm;
        cross_section = row.row.cross_section;
        reduced_matrix_elements.assign(&row.row.reduced_matrix_elements);
        phase_shifts.assign(&row.row.phase_shifts);
        transition_rows.push(row);

        if let Some((retry_regular_channels, retry_irregular_channels)) = cross_term_channels {
            let Some(cross_term_plan) = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
                spin_polarized: true,
                spin_orbit_removed_pass: false,
                transition_index_1based: transition.transition_index_1based,
                orbital_l: input.orbital_l,
                active_len: input.active_channel_count,
            })?
            else {
                continue;
            };
            let retry_input = XsphXsectBcoefStandardChannelRowInput {
                transition,
                selected_higher_multipole: input.selected_higher_multipole,
                initial_kappa: input.initial_kappa,
                initial_large: input.initial_large,
                initial_small: input.initial_small,
                regular_channel: &retry_regular_channels[index],
                irregular_channel: &retry_irregular_channels[index],
                xray_bessel: input.xray_bessel,
                radii: input.radii,
                log_step: input.log_step,
                photon_wave_number: input.photon_wave_number,
                screened_field_scale: field.screened_field_scale,
                fscf: field.fscf,
                diagonal_weights: input.diagonal_weights,
                spectrum_norm,
                cross_section,
                reduced_matrix_elements: reduced_matrix_elements.view(),
                phase_shifts: phase_shifts.view(),
            };
            let retry_radial = xsect_bcoef_standard_radial_components(
                &retry_input,
                XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            )?;

            match cross_term_plan.mode {
                XsphXsectCrossTermMode::SaveCurrentForNext => {
                    saved_cross_term_state =
                        xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
                            plan: cross_term_plan,
                            transition_index_1based: transition.transition_index_1based,
                            radial_integral: retry_radial.reduced_matrix_integral,
                            phase_shift: retry_input.regular_channel.phase.phase_shift,
                            regular_coupling: retry_radial.reduced_component_integrals[0]
                                .unweighted_coupling
                                .view(),
                            irregular_coupling: retry_radial.central_component_integrals[0]
                                .unweighted_irregular_coupling
                                .view(),
                            active_len: retry_radial.reduced_component_integrals[0]
                                .unweighted_coupling
                                .len(),
                        })?;
                }
                XsphXsectCrossTermMode::UsePreviousForCurrent => {
                    if !saved_cross_term_state.as_ref().is_some_and(|state| {
                        state.transition_index_1based == cross_term_plan.partner_index_1based
                            && state.partner_index_1based == transition.transition_index_1based
                    }) {
                        let partner_position = input
                            .transitions
                            .iter()
                            .position(|candidate| {
                                candidate.transition_index_1based
                                    == cross_term_plan.partner_index_1based
                            })
                            .ok_or(XsphError::InvalidOneBasedIndex {
                                name: "xsect_cross_term_saved_transition",
                                index_1based: cross_term_plan.partner_index_1based,
                                active_len: input.active_channel_count,
                            })?;
                        let partner_transition = input.transitions[partner_position];
                        let partner_field = transition_field(partner_position);
                        let partner_retry_input = XsphXsectBcoefStandardChannelRowInput {
                            transition: partner_transition,
                            selected_higher_multipole: input.selected_higher_multipole,
                            initial_kappa: input.initial_kappa,
                            initial_large: input.initial_large,
                            initial_small: input.initial_small,
                            regular_channel: &retry_regular_channels[partner_position],
                            irregular_channel: &retry_irregular_channels[partner_position],
                            xray_bessel: input.xray_bessel,
                            radii: input.radii,
                            log_step: input.log_step,
                            photon_wave_number: input.photon_wave_number,
                            screened_field_scale: partner_field.screened_field_scale,
                            fscf: partner_field.fscf,
                            diagonal_weights: input.diagonal_weights,
                            spectrum_norm,
                            cross_section,
                            reduced_matrix_elements: reduced_matrix_elements.view(),
                            phase_shifts: phase_shifts.view(),
                        };
                        let partner_retry_radial = xsect_bcoef_standard_radial_components(
                            &partner_retry_input,
                            XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
                        )?;
                        let synthetic_save_plan = XsphXsectCrossTermPlan {
                            iold: 1,
                            mode: XsphXsectCrossTermMode::SaveCurrentForNext,
                            partner_index_1based: transition.transition_index_1based,
                        };
                        saved_cross_term_state =
                            xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
                                plan: synthetic_save_plan,
                                transition_index_1based: partner_transition.transition_index_1based,
                                radial_integral: partner_retry_radial.reduced_matrix_integral,
                                phase_shift: partner_retry_input.regular_channel.phase.phase_shift,
                                regular_coupling: partner_retry_radial.reduced_component_integrals
                                    [0]
                                .unweighted_coupling
                                .view(),
                                irregular_coupling: partner_retry_radial
                                    .central_component_integrals[0]
                                    .unweighted_irregular_coupling
                                    .view(),
                                active_len: partner_retry_radial.reduced_component_integrals[0]
                                    .unweighted_coupling
                                    .len(),
                            })?;
                    }
                    let state =
                        saved_cross_term_state
                            .as_ref()
                            .ok_or(XsphError::InvalidOneBasedIndex {
                                name: "xsect_cross_term_saved_transition",
                                index_1based: 0,
                                active_len: cross_term_plan.partner_index_1based,
                            })?;
                    let state_reuse =
                        xsph_xsect_cross_term_state_reuse(XsphXsectCrossTermStateReuseInput {
                            plan: cross_term_plan,
                            transition_index_1based: transition.transition_index_1based,
                            state,
                        })?
                        .ok_or(XsphError::InvalidOneBasedIndex {
                            name: "xsect_cross_term_saved_transition",
                            index_1based: 0,
                            active_len: cross_term_plan.partner_index_1based,
                        })?;
                    let radint3 = xsect_bcoef_standard_radial_components(
                        &retry_input,
                        state_reuse.radint3_branch.clone(),
                    )?;
                    let radint4 = xsect_bcoef_standard_radial_components(
                        &retry_input,
                        state_reuse.radint4_branch.clone(),
                    )?;
                    if let Some(cross_term_update) = xsph_xsect_bcoef_cross_term_state_accumulation(
                        XsphXsectBcoefCrossTermStateAccumulationInput {
                            transition_index_1based: transition.transition_index_1based,
                            orbital_l: input.orbital_l,
                            active_len: input.active_channel_count,
                            trace_weights: input.trace_weights,
                            state_reuse: &state_reuse,
                            current_radial_integral: retry_radial.reduced_matrix_integral,
                            current_phase_shift: retry_input.regular_channel.phase.phase_shift,
                            radint3_integral: radint3.central_cross_integral,
                            radint4_integral: radint4.central_cross_integral,
                            cross_section,
                        },
                    )? {
                        cross_section = cross_term_update.cross_section;
                        cross_term_updates.push(cross_term_update);
                    }
                }
            }
        }
    }

    let output_normalization =
        xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
            photon_energy: input.photon_energy,
            wave_number: input.wave_number,
            spectrum_norm,
            cross_section,
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
            active_channel_count: input.active_channel_count,
        })?;

    Ok(XsphXsectBcoefStandardEnergyRow {
        transition_rows,
        cross_term_updates,
        unnormalized_spectrum_norm: spectrum_norm,
        unnormalized_cross_section: cross_section,
        unnormalized_reduced_matrix_elements: reduced_matrix_elements,
        phase_shifts,
        output_normalization,
    })
}

fn xsect_bcoef_nonstandard_radial_components<'a>(
    input: &XsphXsectBcoefNonstandardChannelRowInput<'a>,
    central_branch: XsphRadialCrossIntegralBranch<'a>,
) -> Result<XsectBcoefNonstandardRadialComponents, XsphError> {
    let active_len = input
        .regular_channel
        .normalized_solution
        .regular_large
        .len();

    let reduced_radial_pass = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::ReducedMatrixElement,
        standard_potential: false,
        photon_wave_number: 1.0,
        screened_field_scale: 1.0,
    })?;
    let reduced_mode = xsect_radial_integral_mode_from_ifl(reduced_radial_pass.feff_ifl)?;
    let reduced_radial_integral = xsph_radial_integral(XsphRadialIntegralInput {
        mode: reduced_mode,
        multipole: input.transition.multipole,
        initial_kappa: input.initial_kappa,
        final_kappa: input.transition.final_kappa,
        initial_large: input.initial_large,
        initial_small: input.initial_small,
        final_large_regular: input
            .regular_channel
            .normalized_solution
            .regular_large
            .view(),
        final_small_regular: input
            .regular_channel
            .normalized_solution
            .regular_small
            .view(),
        xray_bessel: input.xray_bessel,
        radii: input.radii,
        log_step: input.log_step,
        active_len,
    })?;

    let central_radial_pass = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::CentralCrossSection,
        standard_potential: false,
        photon_wave_number: 1.0,
        screened_field_scale: 1.0,
    })?;
    let central_mode = xsect_radial_integral_mode_from_ifl(central_radial_pass.feff_ifl)?;
    let central_cross_integral = xsph_radial_cross_integral(XsphRadialCrossIntegralInput {
        mode: central_mode,
        branch: central_branch,
        multipole: input.transition.multipole,
        initial_kappa: input.initial_kappa,
        final_kappa: input.transition.final_kappa,
        initial_large: input.initial_large,
        initial_small: input.initial_small,
        final_large_regular: input
            .regular_channel
            .normalized_solution
            .regular_large
            .view(),
        final_small_regular: input
            .regular_channel
            .normalized_solution
            .regular_small
            .view(),
        final_large_irregular: input
            .irregular_channel
            .transformed_solution
            .irregular_large
            .view(),
        final_small_irregular: input
            .irregular_channel
            .transformed_solution
            .irregular_small
            .view(),
        xray_bessel: input.xray_bessel,
        radii: input.radii,
        log_step: input.log_step,
        active_len,
    })?;

    Ok(XsectBcoefNonstandardRadialComponents {
        reduced_radial_pass,
        central_radial_pass,
        reduced_radial_integral,
        central_cross_integral,
    })
}

/// Nonstandard-potential ordinary FEFF `xsect` energy row from transition channels.
///
/// This folds the source-backed nonstandard transition rows in FEFF traversal
/// order, then applies the positive-omega output normalization that produces
/// the per-spin `xsnorm`, `xsec`, and `rkk` rows consumed by `xsect.dat`
/// spin merging.
pub fn xsph_xsect_bcoef_nonstandard_energy_row(
    input: XsphXsectBcoefNonstandardEnergyRowInput<'_>,
) -> Result<XsphXsectBcoefNonstandardEnergyRow, XsphError> {
    validate_active_len("xsect_transitions", input.transitions.len(), 1)?;
    validate_active_len(
        "xsect_regular_channels",
        input.regular_channels.len(),
        input.transitions.len(),
    )?;
    validate_active_len(
        "xsect_irregular_channels",
        input.irregular_channels.len(),
        input.transitions.len(),
    )?;
    if input.active_channel_count == 0 || input.active_channel_count > XSECT_BCOEF_TRANSITION_SLOTS
    {
        return Err(XsphError::SizeOutOfRange {
            name: "xsect_active_channel_count",
            value: input.active_channel_count,
        });
    }
    validate_xsect_bcoef_diagonal_weights(input.diagonal_weights)?;
    let cross_term_channels = if input.spin_polarized_cross_terms {
        validate_active_len(
            "xsect_orbital_l",
            input.orbital_l.len(),
            input.active_channel_count,
        )?;
        validate_xsect_bcoef_trace_weights(input.trace_weights)?;
        let regular_channels =
            input
                .spin_orbit_removed_regular_channels
                .ok_or(XsphError::LengthTooShort {
                    name: "xsect_spin_orbit_removed_regular_channels",
                    required: input.transitions.len(),
                    actual: 0,
                })?;
        let irregular_channels =
            input
                .spin_orbit_removed_irregular_channels
                .ok_or(XsphError::LengthTooShort {
                    name: "xsect_spin_orbit_removed_irregular_channels",
                    required: input.transitions.len(),
                    actual: 0,
                })?;
        validate_active_len(
            "xsect_spin_orbit_removed_regular_channels",
            regular_channels.len(),
            input.transitions.len(),
        )?;
        validate_active_len(
            "xsect_spin_orbit_removed_irregular_channels",
            irregular_channels.len(),
            input.transitions.len(),
        )?;
        Some((regular_channels, irregular_channels))
    } else {
        None
    };

    let mut spectrum_norm = 0.0;
    let mut cross_section = Complex::new(0.0, 0.0);
    let mut reduced_matrix_elements = Array1::<Complex>::zeros(XSECT_BCOEF_TRANSITION_SLOTS);
    let mut phase_shifts = Array1::<Complex>::zeros(XSECT_BCOEF_TRANSITION_SLOTS);
    let mut transition_rows = Vec::with_capacity(input.transitions.len());
    let mut cross_term_updates = Vec::new();
    let mut saved_cross_term_state: Option<XsphXsectCrossTermState> = None;

    for (index, transition) in input.transitions.iter().copied().enumerate() {
        let row =
            xsph_xsect_bcoef_nonstandard_channel_row(XsphXsectBcoefNonstandardChannelRowInput {
                transition,
                selected_higher_multipole: input.selected_higher_multipole,
                initial_kappa: input.initial_kappa,
                initial_large: input.initial_large,
                initial_small: input.initial_small,
                regular_channel: &input.regular_channels[index],
                irregular_channel: &input.irregular_channels[index],
                xray_bessel: input.xray_bessel,
                radii: input.radii,
                log_step: input.log_step,
                diagonal_weights: input.diagonal_weights,
                spectrum_norm,
                cross_section,
                reduced_matrix_elements: reduced_matrix_elements.view(),
                phase_shifts: phase_shifts.view(),
            })?;
        spectrum_norm = row.row.spectrum_norm;
        cross_section = row.row.cross_section;
        reduced_matrix_elements.assign(&row.row.reduced_matrix_elements);
        phase_shifts.assign(&row.row.phase_shifts);
        transition_rows.push(row);

        if let Some((retry_regular_channels, retry_irregular_channels)) = cross_term_channels {
            let Some(cross_term_plan) = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
                spin_polarized: true,
                spin_orbit_removed_pass: false,
                transition_index_1based: transition.transition_index_1based,
                orbital_l: input.orbital_l,
                active_len: input.active_channel_count,
            })?
            else {
                continue;
            };
            let retry_input = XsphXsectBcoefNonstandardChannelRowInput {
                transition,
                selected_higher_multipole: input.selected_higher_multipole,
                initial_kappa: input.initial_kappa,
                initial_large: input.initial_large,
                initial_small: input.initial_small,
                regular_channel: &retry_regular_channels[index],
                irregular_channel: &retry_irregular_channels[index],
                xray_bessel: input.xray_bessel,
                radii: input.radii,
                log_step: input.log_step,
                diagonal_weights: input.diagonal_weights,
                spectrum_norm,
                cross_section,
                reduced_matrix_elements: reduced_matrix_elements.view(),
                phase_shifts: phase_shifts.view(),
            };
            let retry_radial = xsect_bcoef_nonstandard_radial_components(
                &retry_input,
                XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            )?;
            let retry_reduced_matrix_integral = retry_radial.reduced_radial_integral.value
                * retry_radial.reduced_radial_pass.post_radint_scale;

            match cross_term_plan.mode {
                XsphXsectCrossTermMode::SaveCurrentForNext => {
                    saved_cross_term_state =
                        xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
                            plan: cross_term_plan,
                            transition_index_1based: transition.transition_index_1based,
                            radial_integral: retry_reduced_matrix_integral,
                            phase_shift: retry_input.regular_channel.phase.phase_shift,
                            regular_coupling: retry_radial.reduced_radial_integral.coupling.view(),
                            irregular_coupling: retry_radial
                                .central_cross_integral
                                .irregular_coupling
                                .view(),
                            active_len: retry_radial.reduced_radial_integral.coupling.len(),
                        })?;
                }
                XsphXsectCrossTermMode::UsePreviousForCurrent => {
                    if !saved_cross_term_state.as_ref().is_some_and(|state| {
                        state.transition_index_1based == cross_term_plan.partner_index_1based
                            && state.partner_index_1based == transition.transition_index_1based
                    }) {
                        let partner_position = input
                            .transitions
                            .iter()
                            .position(|candidate| {
                                candidate.transition_index_1based
                                    == cross_term_plan.partner_index_1based
                            })
                            .ok_or(XsphError::InvalidOneBasedIndex {
                                name: "xsect_cross_term_saved_transition",
                                index_1based: cross_term_plan.partner_index_1based,
                                active_len: input.active_channel_count,
                            })?;
                        let partner_transition = input.transitions[partner_position];
                        let partner_retry_input = XsphXsectBcoefNonstandardChannelRowInput {
                            transition: partner_transition,
                            selected_higher_multipole: input.selected_higher_multipole,
                            initial_kappa: input.initial_kappa,
                            initial_large: input.initial_large,
                            initial_small: input.initial_small,
                            regular_channel: &retry_regular_channels[partner_position],
                            irregular_channel: &retry_irregular_channels[partner_position],
                            xray_bessel: input.xray_bessel,
                            radii: input.radii,
                            log_step: input.log_step,
                            diagonal_weights: input.diagonal_weights,
                            spectrum_norm,
                            cross_section,
                            reduced_matrix_elements: reduced_matrix_elements.view(),
                            phase_shifts: phase_shifts.view(),
                        };
                        let partner_retry_radial = xsect_bcoef_nonstandard_radial_components(
                            &partner_retry_input,
                            XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
                        )?;
                        let partner_retry_reduced_matrix_integral =
                            partner_retry_radial.reduced_radial_integral.value
                                * partner_retry_radial.reduced_radial_pass.post_radint_scale;
                        let synthetic_save_plan = XsphXsectCrossTermPlan {
                            iold: 1,
                            mode: XsphXsectCrossTermMode::SaveCurrentForNext,
                            partner_index_1based: transition.transition_index_1based,
                        };
                        saved_cross_term_state =
                            xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
                                plan: synthetic_save_plan,
                                transition_index_1based: partner_transition.transition_index_1based,
                                radial_integral: partner_retry_reduced_matrix_integral,
                                phase_shift: partner_retry_input.regular_channel.phase.phase_shift,
                                regular_coupling: partner_retry_radial
                                    .reduced_radial_integral
                                    .coupling
                                    .view(),
                                irregular_coupling: partner_retry_radial
                                    .central_cross_integral
                                    .irregular_coupling
                                    .view(),
                                active_len: partner_retry_radial
                                    .reduced_radial_integral
                                    .coupling
                                    .len(),
                            })?;
                    }
                    let state =
                        saved_cross_term_state
                            .as_ref()
                            .ok_or(XsphError::InvalidOneBasedIndex {
                                name: "xsect_cross_term_saved_transition",
                                index_1based: 0,
                                active_len: cross_term_plan.partner_index_1based,
                            })?;
                    let state_reuse =
                        xsph_xsect_cross_term_state_reuse(XsphXsectCrossTermStateReuseInput {
                            plan: cross_term_plan,
                            transition_index_1based: transition.transition_index_1based,
                            state,
                        })?
                        .ok_or(XsphError::InvalidOneBasedIndex {
                            name: "xsect_cross_term_saved_transition",
                            index_1based: 0,
                            active_len: cross_term_plan.partner_index_1based,
                        })?;
                    let radint3 = xsect_bcoef_nonstandard_radial_components(
                        &retry_input,
                        state_reuse.radint3_branch.clone(),
                    )?;
                    let radint4 = xsect_bcoef_nonstandard_radial_components(
                        &retry_input,
                        state_reuse.radint4_branch.clone(),
                    )?;
                    let radint3_integral = radint3.central_cross_integral.value
                        * radint3.central_radial_pass.post_radint_scale;
                    let radint4_integral = radint4.central_cross_integral.value
                        * radint4.central_radial_pass.post_radint_scale;
                    if let Some(cross_term_update) = xsph_xsect_bcoef_cross_term_state_accumulation(
                        XsphXsectBcoefCrossTermStateAccumulationInput {
                            transition_index_1based: transition.transition_index_1based,
                            orbital_l: input.orbital_l,
                            active_len: input.active_channel_count,
                            trace_weights: input.trace_weights,
                            state_reuse: &state_reuse,
                            current_radial_integral: retry_reduced_matrix_integral,
                            current_phase_shift: retry_input.regular_channel.phase.phase_shift,
                            radint3_integral,
                            radint4_integral,
                            cross_section,
                        },
                    )? {
                        cross_section = cross_term_update.cross_section;
                        cross_term_updates.push(cross_term_update);
                    }
                }
            }
        }
    }

    let output_normalization =
        xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
            photon_energy: input.photon_energy,
            wave_number: input.wave_number,
            spectrum_norm,
            cross_section,
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
            active_channel_count: input.active_channel_count,
        })?;

    Ok(XsphXsectBcoefNonstandardEnergyRow {
        transition_rows,
        cross_term_updates,
        unnormalized_spectrum_norm: spectrum_norm,
        unnormalized_cross_section: cross_section,
        unnormalized_reduced_matrix_elements: reduced_matrix_elements,
        phase_shifts,
        output_normalization,
    })
}

/// Port of the FEFF `XSPH/xsect.f90` density-branch predicate.
///
/// FEFF computes `jproj = iorb(ikap)`, falls back to `iorb(-ikap-1)` for
/// negative kappa channels with no direct projector, then runs the `xrhoce` and
/// `xrhopr` block only for the first spin-orbit pass, a matching `kdif`, and a
/// positive one-based projector index.
pub fn xsph_xsect_density_branch(
    input: XsphXsectDensityBranchInput<'_>,
) -> Result<Option<XsphXsectDensityBranch>, XsphError> {
    if input.initial_kappa == 0 || input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len(
        "orbital_projector_map",
        input.orbital_projector_map.len(),
        1,
    )?;

    let mut projector = xsph_xsect_projector_lookup(
        input.orbital_projector_map,
        input.min_projector_kappa,
        input.final_kappa,
    )?;
    if projector == 0 && input.final_kappa < 0 {
        let fallback_kappa = input
            .final_kappa
            .checked_neg()
            .and_then(|value| value.checked_sub(1))
            .ok_or(XsphError::IntegerOutOfRange {
                name: "final_kappa",
                value: input.final_kappa,
            })?;
        projector = xsph_xsect_projector_lookup(
            input.orbital_projector_map,
            input.min_projector_kappa,
            fallback_kappa,
        )?;
    }

    let required_transition_delta = if input.initial_kappa > 0 { 1 } else { -1 };
    if input.transition_delta == required_transition_delta
        && !input.spin_orbit_removed_pass
        && projector > 0
    {
        let projector_index_1based =
            usize::try_from(projector).map_err(|_| XsphError::IntegerOutOfRange {
                name: "projector_index",
                value: projector,
            })?;
        Ok(Some(XsphXsectDensityBranch {
            required_transition_delta,
            projector_index_1based,
        }))
    } else {
        Ok(None)
    }
}

/// Port of FEFF `XSPH/xsect.f90` `iold` cross-term planning.
///
/// In spin-polarized runs FEFF repeats selected adjacent final-state channels
/// with spin-orbit removed. The first row of an adjacent same-`l` pair saves its
/// radial samples for the next channel (`iold = 1`); later rows reuse the
/// previous samples (`iold = 2`).
pub fn xsph_xsect_cross_term_plan(
    input: XsphXsectCrossTermPlanInput<'_>,
) -> Result<Option<XsphXsectCrossTermPlan>, XsphError> {
    validate_xsect_cross_term_plan_input(&input)?;
    if !input.spin_polarized || input.spin_orbit_removed_pass {
        return Ok(None);
    }

    let index = input.transition_index_1based - 1;
    let current_l = input.orbital_l[index];
    if current_l <= 0 {
        return Ok(None);
    }

    if input.transition_index_1based == 1 {
        let partner_index_1based = 2;
        if partner_index_1based <= input.active_len && input.orbital_l[1] == current_l {
            return Ok(Some(XsphXsectCrossTermPlan {
                iold: 1,
                mode: XsphXsectCrossTermMode::SaveCurrentForNext,
                partner_index_1based,
            }));
        }
    } else {
        let partner_index_1based = input.transition_index_1based - 1;
        if input.orbital_l[partner_index_1based - 1] == current_l {
            return Ok(Some(XsphXsectCrossTermPlan {
                iold: 2,
                mode: XsphXsectCrossTermMode::UsePreviousForCurrent,
                partner_index_1based,
            }));
        }
    }

    Ok(None)
}

/// Save FEFF `rkk1/phold/xrcold/xncold` state for an adjacent same-`l` retry.
///
/// FEFF populates this state during the spin-orbit-removed `iold = 1` retry. The
/// explicit Rust state lets the row driver pass the saved couplings back into
/// `radint(ifl=3)` and `radint(ifl=4)` without implicit mutable work arrays.
pub fn xsph_xsect_cross_term_state_save(
    input: XsphXsectCrossTermStateSaveInput<'_>,
) -> Result<Option<XsphXsectCrossTermState>, XsphError> {
    if input.plan.mode != XsphXsectCrossTermMode::SaveCurrentForNext {
        return Ok(None);
    }
    validate_xsect_cross_term_save_input(&input)?;

    Ok(Some(XsphXsectCrossTermState {
        transition_index_1based: input.transition_index_1based,
        partner_index_1based: input.plan.partner_index_1based,
        radial_integral: input.radial_integral,
        phase_shift: input.phase_shift,
        regular_coupling: Array1::from_iter(
            (0..input.active_len).map(|index| input.regular_coupling[index]),
        ),
        irregular_coupling: Array1::from_iter(
            (0..input.active_len).map(|index| input.irregular_coupling[index]),
        ),
    }))
}

/// Reuse saved FEFF cross-term state as `radint(ifl=3)`/`radint(ifl=4)` branches.
///
/// This helper validates that the saved state is the previous partner selected by
/// an `iold = 2` retry and returns the branch selectors consumed by
/// [`xsph_radial_cross_integral`].
pub fn xsph_xsect_cross_term_state_reuse<'a>(
    input: XsphXsectCrossTermStateReuseInput<'a>,
) -> Result<Option<XsphXsectCrossTermStateReuse<'a>>, XsphError> {
    if input.plan.mode != XsphXsectCrossTermMode::UsePreviousForCurrent {
        return Ok(None);
    }
    validate_xsect_cross_term_reuse_input(&input)?;

    Ok(Some(XsphXsectCrossTermStateReuse {
        saved_transition_index_1based: input.state.transition_index_1based,
        saved_radial_integral: input.state.radial_integral,
        saved_phase_shift: input.state.phase_shift,
        radint3_branch: XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling: input.state.regular_coupling.view(),
        },
        radint4_branch: XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling: input.state.irregular_coupling.view(),
        },
    }))
}

/// Port of FEFF `XSPH/xsect.f90` XMCD cross-term accumulation.
///
/// After the spin-orbit-removed retry pass, FEFF couples adjacent same-`l`
/// transitions with the saved `rkk1/phold`, current `rkk0/ph0`, symmetrized
/// off-diagonal `bmat`, and the `radint` branches `ifl = 3` and `ifl = 4`.
pub fn xsph_xsect_cross_term_accumulation(
    input: XsphXsectCrossTermAccumulationInput<'_>,
) -> Result<Option<XsphXsectCrossTermAccumulation>, XsphError> {
    validate_xsect_cross_term_accumulation_input(&input)?;

    if input.transition_index_1based == 1 {
        return Ok(None);
    }
    let partner_index_1based = input.transition_index_1based - 1;
    if partner_index_1based > 8 {
        return Ok(None);
    }

    let partner_l = input.orbital_l[partner_index_1based - 1];
    let current_l = input.orbital_l[input.transition_index_1based - 1];
    if partner_l <= 0 || partner_l != current_l {
        return Ok(None);
    }

    validate_finite_complex(
        "xsect_cross_term_saved_integral",
        0,
        input.saved_radial_integral,
    )?;
    validate_finite_complex(
        "xsect_cross_term_current_integral",
        0,
        input.current_radial_integral,
    )?;
    validate_finite_complex("xsect_cross_term_saved_phase", 0, input.saved_phase_shift)?;
    validate_finite_complex(
        "xsect_cross_term_current_phase",
        0,
        input.current_phase_shift,
    )?;
    validate_finite_complex(
        "xsect_cross_term_partner_current_weight",
        0,
        input.partner_current_weight,
    )?;
    validate_finite_complex(
        "xsect_cross_term_current_partner_weight",
        0,
        input.current_partner_weight,
    )?;
    validate_finite_complex("xsect_cross_term_radint3", 0, input.radint3_integral)?;
    validate_finite_complex("xsect_cross_term_radint4", 0, input.radint4_integral)?;
    validate_finite_complex("xsect_cross_term_cross_section", 0, input.cross_section)?;

    let imaginary_unit = Complex::new(0.0, 1.0);
    let phase_factor =
        (imaginary_unit * (input.current_phase_shift - input.saved_phase_shift)).exp();
    validate_finite_complex("xsect_cross_term_phase_factor", 0, phase_factor)?;
    if phase_factor == Complex::new(0.0, 0.0) {
        return Err(XsphError::ZeroComplexResult {
            name: "xsect_cross_term_phase_factor",
        });
    }

    let inverse_phase_factor = Complex::new(1.0, 0.0) / phase_factor;
    let angular_coupling = -(input.partner_current_weight + input.current_partner_weight) / 2.0;
    let matrix_cross_term_increment = -imaginary_unit
        * input.saved_radial_integral
        * input.current_radial_integral
        * (inverse_phase_factor + phase_factor)
        * angular_coupling;
    let radint3_increment = input.radint3_integral * angular_coupling * inverse_phase_factor;
    let radint4_increment = input.radint4_integral * angular_coupling * phase_factor;
    let cross_section_increment =
        matrix_cross_term_increment + radint3_increment + radint4_increment;
    let cross_section = input.cross_section + cross_section_increment;

    validate_finite_complex(
        "xsect_cross_term_inverse_phase_factor",
        0,
        inverse_phase_factor,
    )?;
    validate_finite_complex("xsect_cross_term_angular_coupling", 0, angular_coupling)?;
    validate_finite_complex(
        "xsect_cross_term_matrix_increment",
        0,
        matrix_cross_term_increment,
    )?;
    validate_finite_complex("xsect_cross_term_radint3_increment", 0, radint3_increment)?;
    validate_finite_complex("xsect_cross_term_radint4_increment", 0, radint4_increment)?;
    validate_finite_complex(
        "xsect_cross_term_cross_section_increment",
        0,
        cross_section_increment,
    )?;
    validate_finite_complex("xsect_cross_term_updated_cross_section", 0, cross_section)?;

    Ok(Some(XsphXsectCrossTermAccumulation {
        partner_index_1based,
        phase_factor,
        inverse_phase_factor,
        angular_coupling,
        matrix_cross_term_increment,
        radint3_increment,
        radint4_increment,
        cross_section_increment,
        cross_section,
    }))
}

/// XMCD cross-term accumulation using source-backed traced `bcoef` weights.
///
/// FEFF reads the off-diagonal pair `bmat(0,isp,k1,0,isp,ind)` and
/// `bmat(0,isp,ind,0,isp,k1)` from the same traced table used for direct
/// transitions. This wrapper performs that lookup only after the FEFF skip
/// conditions have selected an active adjacent same-`l` pair.
pub fn xsph_xsect_bcoef_cross_term_accumulation(
    input: XsphXsectBcoefCrossTermAccumulationInput<'_>,
) -> Result<Option<XsphXsectCrossTermAccumulation>, XsphError> {
    validate_xsect_bcoef_cross_term_accumulation_input(&input)?;

    if input.transition_index_1based == 1 {
        return Ok(None);
    }
    let partner_index_1based = input.transition_index_1based - 1;
    if partner_index_1based > 8 {
        return Ok(None);
    }

    let partner_l = input.orbital_l[partner_index_1based - 1];
    let current_l = input.orbital_l[input.transition_index_1based - 1];
    if partner_l <= 0 || partner_l != current_l {
        return Ok(None);
    }

    let partner_current_weight = xsect_bcoef_trace_weight(
        input.trace_weights,
        partner_index_1based,
        input.transition_index_1based,
    )?;
    let current_partner_weight = xsect_bcoef_trace_weight(
        input.trace_weights,
        input.transition_index_1based,
        partner_index_1based,
    )?;

    xsph_xsect_cross_term_accumulation(XsphXsectCrossTermAccumulationInput {
        transition_index_1based: input.transition_index_1based,
        orbital_l: input.orbital_l,
        active_len: input.active_len,
        saved_radial_integral: input.saved_radial_integral,
        current_radial_integral: input.current_radial_integral,
        saved_phase_shift: input.saved_phase_shift,
        current_phase_shift: input.current_phase_shift,
        partner_current_weight,
        current_partner_weight,
        radint3_integral: input.radint3_integral,
        radint4_integral: input.radint4_integral,
        cross_section: input.cross_section,
    })
}

/// Bcoef-weighted XMCD cross-term accumulation from saved `iold` retry state.
///
/// The eventual `xsect` row driver receives saved `rkk1/phold` through
/// [`XsphXsectCrossTermStateReuse`]. This wrapper validates that the saved state
/// belongs to FEFF's active adjacent same-`l` partner, then delegates the
/// off-diagonal `bcoef` lookup and cross-section update to
/// [`xsph_xsect_bcoef_cross_term_accumulation`].
pub fn xsph_xsect_bcoef_cross_term_state_accumulation(
    input: XsphXsectBcoefCrossTermStateAccumulationInput<'_>,
) -> Result<Option<XsphXsectCrossTermAccumulation>, XsphError> {
    validate_xsect_bcoef_cross_term_state_accumulation_input(&input)?;
    let Some(partner_index_1based) = xsect_active_cross_term_partner(
        input.transition_index_1based,
        input.orbital_l,
        input.active_len,
    )?
    else {
        return Ok(None);
    };
    validate_xsect_cross_term_state_reuse_payload(input.state_reuse, partner_index_1based)?;

    xsph_xsect_bcoef_cross_term_accumulation(XsphXsectBcoefCrossTermAccumulationInput {
        transition_index_1based: input.transition_index_1based,
        orbital_l: input.orbital_l,
        active_len: input.active_len,
        trace_weights: input.trace_weights,
        saved_radial_integral: input.state_reuse.saved_radial_integral,
        current_radial_integral: input.current_radial_integral,
        saved_phase_shift: input.state_reuse.saved_phase_shift,
        current_phase_shift: input.current_phase_shift,
        radint3_integral: input.radint3_integral,
        radint4_integral: input.radint4_integral,
        cross_section: input.cross_section,
    })
}

/// Port of FEFF `XSPH/xsect.f90` embedded central density `xrhoce`.
///
/// This evaluates FEFF's `xrc = N*R - i*R*R` large/small-component density
/// samples, integrates them through the Norman radius with `csomm2`, and applies
/// the common `temp = (2*lfin+1)/(1+factor**2)/pi*4*ck/hart` prefactor.
pub fn xsph_xsect_embedded_density(
    input: XsphXsectEmbeddedDensityInput<'_>,
) -> Result<XsphXsectEmbeddedDensity, XsphError> {
    validate_xsect_density_base_input(
        input.final_kappa,
        input.wave_number,
        input.regular_large,
        input.regular_small,
        input.irregular_large,
        input.irregular_small,
        input.radii,
        input.log_step,
        input.norman_radius,
        input.active_len,
        input.integration_len,
    )?;

    let prefactor =
        xsph_xsect_density_prefactor(input.final_l, input.final_kappa, input.wave_number)?;
    let imaginary_unit = Complex::new(0.0, 1.0);
    let density_samples = (0..input.active_len)
        .map(|index| {
            input.irregular_large[index] * input.regular_large[index]
                - imaginary_unit * input.regular_large[index] * input.regular_large[index]
                + input.irregular_small[index] * input.regular_small[index]
                - imaginary_unit * input.regular_small[index] * input.regular_small[index]
        })
        .collect::<Array1<_>>();
    for index in 0..input.active_len {
        validate_finite_complex(
            "xsect_embedded_density_samples",
            index,
            density_samples[index],
        )?;
    }

    let integral = csomm2(
        &active_real_prefix(input.radii, input.integration_len),
        &active_complex_prefix(density_samples.view(), input.integration_len),
        input.log_step,
        1.0,
        input.norman_radius,
    )?;
    validate_finite_complex("xsect_embedded_density_integral", 0, integral)?;
    let density = -integral * prefactor;
    validate_finite_complex("xsect_embedded_density", 0, density)?;

    Ok(XsphXsectEmbeddedDensity {
        prefactor,
        density_samples,
        integral,
        density,
    })
}

/// Port of FEFF `XSPH/xsect.f90` projected density `xrhopr`.
///
/// FEFF first normalizes the selected atomic projector inside the Norman
/// sphere, accumulates the regular-solution overlap by a radial trapezoid, then
/// integrates the projected `N*P_at*intr - i*R*P_at*intr` density samples with
/// `csomm2`.
pub fn xsph_xsect_projected_density(
    input: XsphXsectProjectedDensityInput<'_>,
) -> Result<XsphXsectProjectedDensity, XsphError> {
    validate_xsect_projected_density_input(&input)?;

    let prefactor =
        xsph_xsect_density_prefactor(input.final_l, input.final_kappa, input.wave_number)?;
    let atomic_norm_samples = (0..input.integration_len)
        .map(|index| input.atomic_large[index].powi(2) + input.atomic_small[index].powi(2))
        .collect::<Vec<_>>();
    let atomic_norm_integral = somm2(
        &active_real_prefix(input.radii, input.integration_len),
        &atomic_norm_samples,
        input.log_step,
        2.0 * input.final_l as Real + 2.0,
        input.norman_radius,
        0,
    )?;
    if !atomic_norm_integral.is_finite() || atomic_norm_integral <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "xsect_projected_atomic_norm",
            value: atomic_norm_integral,
        });
    }
    let atomic_norm_sqrt = atomic_norm_integral.sqrt();
    validate_finite_real("xsect_projected_atomic_norm_sqrt", atomic_norm_sqrt)?;

    let normalized_atomic_large = (0..input.active_len)
        .map(|index| input.atomic_large[index] / atomic_norm_sqrt)
        .collect::<Array1<_>>();
    let normalized_atomic_small = (0..input.active_len)
        .map(|index| input.atomic_small[index] / atomic_norm_sqrt)
        .collect::<Array1<_>>();

    let overlap_samples = (0..input.active_len)
        .map(|index| {
            normalized_atomic_large[index] * input.regular_large[index]
                + normalized_atomic_small[index] * input.regular_small[index]
        })
        .collect::<Array1<_>>();
    let cumulative_overlap =
        xsph_xsect_cumulative_trapezoid(input.radii, overlap_samples.view(), input.active_len)?;

    let imaginary_unit = Complex::new(0.0, 1.0);
    let density_samples = (0..input.active_len)
        .map(|index| {
            let large = normalized_atomic_large[index] * cumulative_overlap[index];
            let small = normalized_atomic_small[index] * cumulative_overlap[index];
            input.irregular_large[index] * large + input.irregular_small[index] * small
                - imaginary_unit
                    * (input.regular_large[index] * large + input.regular_small[index] * small)
        })
        .collect::<Array1<_>>();
    for index in 0..input.active_len {
        validate_finite_real(
            "xsect_projected_atomic_large",
            normalized_atomic_large[index],
        )?;
        validate_finite_real(
            "xsect_projected_atomic_small",
            normalized_atomic_small[index],
        )?;
        validate_finite_complex("xsect_projected_overlap", index, cumulative_overlap[index])?;
        validate_finite_complex(
            "xsect_projected_density_samples",
            index,
            density_samples[index],
        )?;
    }

    let integral = csomm2(
        &active_real_prefix(input.radii, input.integration_len),
        &active_complex_prefix(density_samples.view(), input.integration_len),
        input.log_step,
        1.0,
        input.norman_radius,
    )?;
    validate_finite_complex("xsect_projected_density_integral", 0, integral)?;
    let density = -integral * prefactor;
    validate_finite_complex("xsect_projected_density", 0, density)?;

    Ok(XsphXsectProjectedDensity {
        prefactor,
        atomic_norm_integral,
        atomic_norm_sqrt,
        normalized_atomic_large,
        normalized_atomic_small,
        cumulative_overlap,
        density_samples,
        integral,
        density,
    })
}

/// Port of FEFF `XSPH/radjas.f90` `getcorrection`.
///
/// This builds the NRIXS orthogonality correction `ortcor(0:ljmax, 1:nq)` by
/// normalizing q-Bessel-weighted bound-spinor overlaps against the unweighted
/// bound-spinor norm.
pub fn xsph_jas_orthogonality_correction(
    input: XsphJasOrthogonalityCorrectionInput<'_>,
) -> Result<XsphJasOrthogonalityCorrection, XsphError> {
    validate_jas_orthogonality_input(&input)?;

    let radii = active_real_prefix(input.radii, input.active_len);
    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let norm_samples = (0..input.active_len)
        .map(|index| {
            Complex::new(
                input.large_component[index] * input.large_component[index]
                    + input.small_component[index] * input.small_component[index],
                0.0,
            )
        })
        .collect::<Vec<_>>();
    let normalization = csommjas(
        &radii,
        &norm_samples,
        &zeros,
        input.log_step,
        jas_near_origin_power(input.initial_l, 0)?,
        0,
    )?;
    if normalization == Complex::new(0.0, 0.0) {
        return Err(XsphError::ZeroJasOrthogonalityNormalization);
    }

    let q_count = input.q_bessel.shape()[2];
    let mut corrections = Array2::<Complex>::zeros((input.ljmax + 1, q_count).f());
    for q_index in 0..q_count {
        for angular_l in 0..=input.ljmax {
            if angular_l <= input.initial_j {
                let weighted = (0..input.active_len)
                    .map(|radius_index| {
                        let q_bessel = input.q_bessel[(radius_index, angular_l, q_index)];
                        Complex::new(
                            input.large_component[radius_index]
                                * q_bessel
                                * input.large_component[radius_index]
                                + input.small_component[radius_index]
                                    * q_bessel
                                    * input.small_component[radius_index],
                            0.0,
                        )
                    })
                    .collect::<Vec<_>>();
                corrections[(angular_l, q_index)] = csommjas(
                    &radii,
                    &weighted,
                    &zeros,
                    input.log_step,
                    jas_near_origin_power(input.initial_l, angular_l)?,
                    0,
                )? / normalization;
            }
        }
    }

    Ok(XsphJasOrthogonalityCorrection {
        corrections,
        normalization,
    })
}

/// Port of FEFF `XSPH/radjas.f90` `getorthg`.
///
/// This helper evaluates the large (`ap`) and small (`aq`) component overlaps
/// with the same four-step corrected Simpson quadrature FEFF uses for JAS
/// radial integrals.
pub fn xsph_jas_overlap(input: XsphJasOverlapInput<'_>) -> Result<XsphJasOverlap, XsphError> {
    validate_jas_overlap_input(&input)?;

    let radii = active_real_prefix(input.radii, input.active_len);
    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let near_origin_power = jas_overlap_near_origin_power(input.initial_l, input.final_l)?;
    let large_samples = (0..input.active_len)
        .map(|index| input.final_large[index] * input.initial_large[index])
        .collect::<Vec<_>>();
    let small_samples = (0..input.active_len)
        .map(|index| input.final_small[index] * input.initial_small[index])
        .collect::<Vec<_>>();

    let large_overlap = csommjas(
        &radii,
        &large_samples,
        &zeros,
        input.log_step,
        near_origin_power,
        input.radial_power,
    )?;
    let small_overlap = csommjas(
        &radii,
        &small_samples,
        &zeros,
        input.log_step,
        near_origin_power,
        input.radial_power,
    )?;

    Ok(XsphJasOverlap {
        large_overlap,
        small_overlap,
        total_overlap: large_overlap + small_overlap,
        near_origin_power,
    })
}

/// Port of FEFF `XSPH/radjas.f90` for `ifl = 1`.
///
/// This evaluates the NRIXS/JAS reduced radial matrix elements for all
/// requested `lj` multipoles. Same-kappa channels apply FEFF's orthogonality
/// correction before the `csommjas` integration.
pub fn xsph_jas_radial_integral(
    input: XsphJasRadialIntegralInput<'_>,
) -> Result<XsphJasRadialIntegral, XsphError> {
    validate_jas_radial_integral_input(&input)?;

    let linit = usize::try_from(l_from_kappa(input.initial_kappa)?).map_err(|_| {
        XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: input.initial_kappa,
        }
    })?;
    let lfin = usize::try_from(l_from_kappa(input.final_kappa)?).map_err(|_| {
        XsphError::IntegerOutOfRange {
            name: "final_kappa",
            value: input.final_kappa,
        }
    })?;
    let angular_count = input.ljmax + 1;
    let radii = active_real_prefix(input.radii, input.active_len);
    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let mut radial_integrals = Array1::<Complex>::zeros(angular_count);
    let mut regular_coupling = Array2::<Complex>::zeros((input.active_len, angular_count).f());
    let mut near_origin_powers = Array1::<Real>::zeros(angular_count);

    for angular_l in 0..=input.ljmax {
        if input.needed_multipoles[angular_l] <= 0 {
            continue;
        }
        let angular_l_i32 = i32::try_from(angular_l).map_err(|_| XsphError::SizeOutOfRange {
            name: "angular_l",
            value: angular_l,
        })?;
        let angular_factor = xsph_longitudinal_multipole_factor(
            input.initial_kappa,
            input.final_kappa,
            angular_l_i32,
        )?;
        let coupling = jas_reduced_coupling(&input, angular_l);
        for (index, &value) in coupling.iter().enumerate() {
            regular_coupling[(index, angular_l)] = value;
        }

        let near_origin_power = jas_radial_near_origin_power(linit, lfin, angular_l)?;
        near_origin_powers[angular_l] = near_origin_power;
        radial_integrals[angular_l] = csommjas(
            &radii,
            &coupling,
            &zeros,
            input.log_step,
            near_origin_power,
            0,
        )? * angular_factor;
    }

    Ok(XsphJasRadialIntegral {
        radial_integrals,
        regular_coupling,
        near_origin_powers,
    })
}

/// Port of FEFF `XSPH/radjas.f90` for `ifl = 2`.
///
/// FEFF computes this central-atom NRIXS double radial integral after an
/// `ifl = 1` call has populated `xrc`. This wrapper makes that state explicit:
/// callers pass the saved regular coupling, then the helper builds the
/// irregular coupling, applies the prefix integral, and finishes with
/// `csommjas`.
pub fn xsph_jas_radial_cross_integral(
    input: XsphJasRadialCrossIntegralInput<'_>,
) -> Result<XsphJasRadialCrossIntegral, XsphError> {
    validate_jas_radial_cross_integral_input(&input)?;

    let linit = usize::try_from(l_from_kappa(input.initial_kappa)?).map_err(|_| {
        XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: input.initial_kappa,
        }
    })?;
    let lfin = usize::try_from(l_from_kappa(input.final_kappa)?).map_err(|_| {
        XsphError::IntegerOutOfRange {
            name: "final_kappa",
            value: input.final_kappa,
        }
    })?;
    let angular_count = input.ljmax + 1;
    let radii = active_real_prefix(input.radii, input.active_len);
    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let mut radial_integrals = Array1::<Complex>::zeros(angular_count);
    let mut irregular_coupling = Array2::<Complex>::zeros((input.active_len, angular_count).f());
    let mut regular_prefix_integral =
        Array2::<Complex>::zeros((input.active_len, angular_count).f());
    let mut weighted_irregular_coupling =
        Array2::<Complex>::zeros((input.active_len, angular_count).f());
    let mut first_near_origin_powers = Array1::<Real>::zeros(angular_count);
    let mut second_near_origin_powers = Array1::<Real>::zeros(angular_count);

    for angular_l in 0..=input.ljmax {
        if input.needed_multipoles[angular_l] <= 0 {
            continue;
        }
        let angular_l_i32 = i32::try_from(angular_l).map_err(|_| XsphError::SizeOutOfRange {
            name: "angular_l",
            value: angular_l,
        })?;
        let angular_factor = xsph_longitudinal_multipole_factor(
            input.initial_kappa,
            input.final_kappa,
            angular_l_i32,
        )?;
        let irregular = jas_irregular_coupling(&input, angular_l);
        for (index, &value) in irregular.iter().enumerate() {
            irregular_coupling[(index, angular_l)] = value;
        }

        let first_power = jas_radial_cross_first_power(linit, lfin)?;
        first_near_origin_powers[angular_l] = first_power;
        let mut prefix =
            input.regular_coupling[(0, angular_l)] * (2.0 * radii[0] / (first_power + 1.0));
        regular_prefix_integral[(0, angular_l)] = prefix;
        weighted_irregular_coupling[(0, angular_l)] = irregular[0] * prefix;
        for index in 1..input.active_len {
            prefix += (input.regular_coupling[(index - 1, angular_l)]
                + input.regular_coupling[(index, angular_l)])
                * (radii[index] - radii[index - 1]);
            regular_prefix_integral[(index, angular_l)] = prefix;
            weighted_irregular_coupling[(index, angular_l)] = irregular[index] * prefix;
        }

        let second_power = jas_radial_cross_second_power(linit, lfin)?;
        second_near_origin_powers[angular_l] = second_power;
        let weighted = weighted_irregular_coupling
            .column(angular_l)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        radial_integrals[angular_l] =
            csommjas(&radii, &zeros, &weighted, input.log_step, second_power, 0)?
                * angular_factor
                * angular_factor.conj();
    }

    Ok(XsphJasRadialCrossIntegral {
        radial_integrals,
        irregular_coupling,
        regular_prefix_integral,
        weighted_irregular_coupling,
        first_near_origin_powers,
        second_near_origin_powers,
    })
}

/// Port of FEFF `XSPH/radint.f90` for `abs(ifl) == 1`.
///
/// This evaluates the reduced radial matrix element used for `rkk`, including
/// FEFF's relativistic (`ifl = 1`) and nonrelativistic (`ifl = -1`) branches.
/// The cross-section double-integral branches (`ifl = 2, 3, 4`) are exposed
/// separately by [`xsph_radial_cross_integral`].
pub fn xsph_radial_integral(
    input: XsphRadialIntegralInput<'_>,
) -> Result<XsphRadialIntegral, XsphError> {
    validate_radial_integral_input(&input)?;

    let factors = radint_factors_for_mode(
        input.mode,
        input.final_kappa,
        input.initial_kappa,
        input.multipole,
    )?;
    let coupling = radial_coupling(
        RadialCouplingInput {
            mode: input.mode,
            multipole: input.multipole,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large: input.final_large_regular,
            final_small: input.final_small_regular,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            active_len: input.active_len,
        },
        factors,
    );
    let radii = active_real_prefix(input.radii, input.active_len);
    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let near_origin_power =
        radial_near_origin_power(input.initial_kappa, input.final_kappa, input.multipole)?;
    let value = csomm(
        &radii,
        &coupling,
        &zeros,
        input.log_step,
        near_origin_power,
        0,
    )?;

    Ok(XsphRadialIntegral {
        value,
        coupling: Array1::from_vec(coupling),
        near_origin_power,
    })
}

/// Port the FEFF `fscf`-weighted reduced radial integral used by `xsect`.
///
/// This is the `radint(abs(ifl) == 1)` coupling with a real screened-field
/// component applied pointwise before Simpson integration. It is the missing
/// primitive for the `izstd > 0` standard-atom branch, where FEFF evaluates the
/// real and imaginary `fscf` components separately before combining them.
pub fn xsph_xsect_weighted_radial_integral(
    input: XsphXsectWeightedRadialIntegralInput<'_>,
) -> Result<XsphXsectWeightedRadialIntegral, XsphError> {
    let base = XsphRadialIntegralInput {
        mode: input.mode,
        multipole: input.multipole,
        initial_kappa: input.initial_kappa,
        final_kappa: input.final_kappa,
        initial_large: input.initial_large,
        initial_small: input.initial_small,
        final_large_regular: input.final_large_regular,
        final_small_regular: input.final_small_regular,
        xray_bessel: input.xray_bessel,
        radii: input.radii,
        log_step: input.log_step,
        active_len: input.active_len,
    };
    validate_radial_integral_input(&base)?;
    validate_xsect_radial_weights(
        "xsect_radial_weights",
        input.radial_weights,
        input.active_len,
    )?;

    let factors = radint_factors_for_mode(
        input.mode,
        input.final_kappa,
        input.initial_kappa,
        input.multipole,
    )?;
    let unweighted = radial_coupling(
        RadialCouplingInput {
            mode: input.mode,
            multipole: input.multipole,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large: input.final_large_regular,
            final_small: input.final_small_regular,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            active_len: input.active_len,
        },
        factors,
    );
    let weighted = apply_real_radial_weights(&unweighted, input.radial_weights, input.active_len);
    let radii = active_real_prefix(input.radii, input.active_len);
    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let near_origin_power =
        radial_near_origin_power(input.initial_kappa, input.final_kappa, input.multipole)?;
    let value = csomm(
        &radii,
        &weighted,
        &zeros,
        input.log_step,
        near_origin_power,
        0,
    )?;

    Ok(XsphXsectWeightedRadialIntegral {
        integral: XsphRadialIntegral {
            value,
            coupling: Array1::from_vec(weighted),
            near_origin_power,
        },
        unweighted_coupling: Array1::from_vec(unweighted),
    })
}

/// Port of FEFF `XSPH/radint.f90` for `abs(ifl) == 2`, `ifl == 3`, and `ifl == 4`.
///
/// This evaluates the central-atom cross-section double radial integral. The
/// [`XsphRadialCrossIntegralBranch`] selector mirrors FEFF's `iold` state:
/// compute both couplings for `ifl = 2`, reuse `xrcold` for `ifl = 3`, or
/// reuse `xncold` for `ifl = 4`.
pub fn xsph_radial_cross_integral(
    input: XsphRadialCrossIntegralInput<'_>,
) -> Result<XsphRadialCrossIntegral, XsphError> {
    validate_radial_cross_integral_input(&input)?;

    let factors = radint_factors_for_mode(
        input.mode,
        input.final_kappa,
        input.initial_kappa,
        input.multipole,
    )?;
    let current_regular = radial_coupling(
        RadialCouplingInput {
            mode: input.mode,
            multipole: input.multipole,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large: input.final_large_regular,
            final_small: input.final_small_regular,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            active_len: input.active_len,
        },
        factors,
    );
    let current_irregular = radial_coupling(
        RadialCouplingInput {
            mode: input.mode,
            multipole: input.multipole,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large: input.final_large_irregular,
            final_small: input.final_small_irregular,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            active_len: input.active_len,
        },
        factors,
    );
    let (regular_coupling, irregular_coupling) = match input.branch {
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular => {
            (current_regular, current_irregular)
        }
        XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling,
        } => (
            active_complex_prefix(stored_regular_coupling, input.active_len),
            current_irregular,
        ),
        XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling,
        } => (
            current_regular,
            active_complex_prefix(stored_irregular_coupling, input.active_len),
        ),
    };

    integrate_radial_cross_couplings(&input, regular_coupling, irregular_coupling)
}

/// Port the FEFF `fscf`-weighted central cross-section radial integral.
///
/// This evaluates the `radint(abs(ifl) == 2)` central integral with real
/// screened-field components applied before the prefix and outer radial
/// integrations. The branch selector mirrors FEFF's `iold` retry state so the
/// same primitive can feed ordinary rows and spin-orbit cross-term `radint`
/// branches.
pub fn xsph_xsect_weighted_radial_cross_integral(
    input: XsphXsectWeightedRadialCrossIntegralInput<'_, '_>,
) -> Result<XsphXsectWeightedRadialCrossIntegral, XsphError> {
    let base = XsphRadialCrossIntegralInput {
        mode: input.mode,
        branch: XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
        multipole: input.multipole,
        initial_kappa: input.initial_kappa,
        final_kappa: input.final_kappa,
        initial_large: input.initial_large,
        initial_small: input.initial_small,
        final_large_regular: input.final_large_regular,
        final_small_regular: input.final_small_regular,
        final_large_irregular: input.final_large_irregular,
        final_small_irregular: input.final_small_irregular,
        xray_bessel: input.xray_bessel,
        radii: input.radii,
        log_step: input.log_step,
        active_len: input.active_len,
    };
    validate_radial_cross_integral_input(&base)?;
    match input.branch {
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular => {}
        XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling,
        } => validate_stored_coupling(
            "stored_regular_coupling",
            stored_regular_coupling,
            input.active_len,
        )?,
        XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling,
        } => validate_stored_coupling(
            "stored_irregular_coupling",
            stored_irregular_coupling,
            input.active_len,
        )?,
    }
    validate_xsect_radial_weights(
        "xsect_regular_weights",
        input.regular_weights,
        input.active_len,
    )?;
    validate_xsect_radial_weights(
        "xsect_irregular_weights",
        input.irregular_weights,
        input.active_len,
    )?;

    let factors = radint_factors_for_mode(
        input.mode,
        input.final_kappa,
        input.initial_kappa,
        input.multipole,
    )?;
    let current_regular = radial_coupling(
        RadialCouplingInput {
            mode: input.mode,
            multipole: input.multipole,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large: input.final_large_regular,
            final_small: input.final_small_regular,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            active_len: input.active_len,
        },
        factors,
    );
    let current_irregular = radial_coupling(
        RadialCouplingInput {
            mode: input.mode,
            multipole: input.multipole,
            initial_large: input.initial_large,
            initial_small: input.initial_small,
            final_large: input.final_large_irregular,
            final_small: input.final_small_irregular,
            xray_bessel: input.xray_bessel,
            radii: input.radii,
            active_len: input.active_len,
        },
        factors,
    );
    let (unweighted_regular, unweighted_irregular) = match input.branch {
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular => {
            (current_regular, current_irregular)
        }
        XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling,
        } => (
            active_complex_prefix(stored_regular_coupling, input.active_len),
            current_irregular,
        ),
        XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling,
        } => (
            current_regular,
            active_complex_prefix(stored_irregular_coupling, input.active_len),
        ),
    };
    let weighted_regular =
        apply_real_radial_weights(&unweighted_regular, input.regular_weights, input.active_len);
    let weighted_irregular = apply_real_radial_weights(
        &unweighted_irregular,
        input.irregular_weights,
        input.active_len,
    );
    let integral = integrate_radial_cross_couplings(&base, weighted_regular, weighted_irregular)?;

    Ok(XsphXsectWeightedRadialCrossIntegral {
        integral,
        unweighted_regular_coupling: Array1::from_vec(unweighted_regular),
        unweighted_irregular_coupling: Array1::from_vec(unweighted_irregular),
    })
}

fn validate_xsect_energy_setup_input(input: XsphXsectEnergySetupInput) -> Result<(), XsphError> {
    validate_finite_complex("xsect_energy", 0, input.energy)?;
    validate_finite_complex("xsect_reference_energy", 0, input.reference_energy)?;
    validate_finite_real("edge_energy", input.edge_energy)?;
    validate_finite_real("chemical_potential", input.chemical_potential)?;
    if !input.muffin_tin_radius.is_finite() || input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    validate_active_len("radial_grid", input.radial_capacity, 1)?;
    if input.norman_index_1based == 0 {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "norman_index",
            index_1based: input.norman_index_1based,
            active_len: input.radial_capacity,
        });
    }
    if input.new_grid_index_1based == 0 {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "new_grid_index",
            index_1based: input.new_grid_index_1based,
            active_len: input.radial_capacity,
        });
    }

    Ok(())
}

fn validate_xsect_hole_normalization_input(
    input: &XsphXsectHoleNormalizationInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("radii", input.radii.len(), input.norman_index_1based)?;
    validate_active_len(
        "initial_large",
        input.initial_large.len(),
        input.norman_index_1based,
    )?;
    validate_active_len(
        "initial_small",
        input.initial_small.len(),
        input.norman_index_1based,
    )?;
    if !input.log_step.is_finite() || input.log_step <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "log_step",
            value: input.log_step,
        });
    }
    for index in 0..input.norman_index_1based {
        validate_finite_real("initial_large", input.initial_large[index])?;
        validate_finite_real("initial_small", input.initial_small[index])?;
    }
    Ok(())
}

fn validate_xsect_transition_plan_input(
    input: &XsphXsectTransitionPlanInput<'_>,
) -> Result<(), XsphError> {
    validate_finite_real("photon_energy", input.photon_energy)?;
    if input.initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if !matches!(input.transition_direction, -1..=1) {
        return Err(XsphError::IntegerOutOfRange {
            name: "l2lp",
            value: input.transition_direction,
        });
    }
    if input.active_len > 8 {
        return Err(XsphError::SizeOutOfRange {
            name: "xsect_transition_count",
            value: input.active_len,
        });
    }
    validate_active_len("final_kappas", input.final_kappas.len(), input.active_len)?;
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    Ok(())
}

fn validate_xsect_screened_field_input(
    input: XsphXsectScreenedFieldInput,
) -> Result<(), XsphError> {
    validate_finite_complex("xsect_momentum_squared", 0, input.momentum_squared)?;
    validate_finite_real("edge_energy", input.edge_energy)?;
    validate_finite_real("chemical_potential", input.chemical_potential)?;
    Ok(())
}

fn validate_xsect_fscf_weights_input(
    input: &XsphXsectFscfWeightsInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("fscf", input.fscf.len(), input.active_len)?;
    for index in 0..input.active_len {
        validate_finite_complex("fscf", index, input.fscf[index])?;
    }
    Ok(())
}

fn validate_xsect_phiscf_local_field_input(
    input: &XsphXsectPhiscfLocalFieldInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("radii", input.radii.len(), input.active_len)?;
    validate_active_len(
        "electron_density",
        input.electron_density.len(),
        input.active_len,
    )?;
    for index in 0..input.active_len {
        let radius = input.radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveScalar {
                name: "xsect_phiscf_radius",
                value: radius,
            });
        }
        validate_finite_real("electron_density", input.electron_density[index])?;
    }
    Ok(())
}

fn validate_xsect_phiscf_linear_solve_input(
    input: &XsphXsectPhiscfLinearSolveInput<'_>,
) -> Result<(), XsphError> {
    if input.coarse_count == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    let fine_len = phiscf_linear_fine_len(input.coarse_count)?;
    validate_active_len("radii", input.radii.len(), fine_len)?;
    if input.response.nrows() < input.coarse_count || input.response.ncols() < input.coarse_count {
        return Err(XsphError::MatrixTooSmall {
            name: "xsect_phiscf_response",
            required: [input.coarse_count, input.coarse_count],
            actual: [input.response.nrows(), input.response.ncols()],
        });
    }
    if input.basis_count > 0
        && (input.basis_fields.nrows() < fine_len || input.basis_fields.ncols() < input.basis_count)
    {
        return Err(XsphError::MatrixTooSmall {
            name: "xsect_phiscf_basis_fields",
            required: [fine_len, input.basis_count],
            actual: [input.basis_fields.nrows(), input.basis_fields.ncols()],
        });
    }
    for index in 0..fine_len {
        let radius = input.radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "xsect_phiscf_radius",
                value: radius,
            });
        }
        for basis_index in 0..input.basis_count {
            validate_finite_complex(
                "xsect_phiscf_basis_field",
                index,
                input.basis_fields[(index, basis_index)],
            )?;
        }
    }
    for row in 0..input.coarse_count {
        for column in 0..input.coarse_count {
            validate_finite_complex("xsect_phiscf_response", row, input.response[(row, column)])?;
        }
    }
    Ok(())
}

fn validate_xsect_phiscf_lipman_input(
    input: &XsphXsectPhiscfLipmanInput<'_>,
) -> Result<(), XsphError> {
    if input.coarse_count == 0 || input.active_len == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    if input.match_index_1based == 0 || input.match_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_match_index",
            index_1based: input.match_index_1based,
            active_len: input.active_len,
        });
    }
    validate_active_len("radii", input.radii.len(), input.active_len)?;
    validate_active_len("orbital_large", input.orbital_large.len(), input.active_len)?;
    validate_active_len("orbital_small", input.orbital_small.len(), input.active_len)?;
    validate_active_len("regular_large", input.regular_large.len(), input.active_len)?;
    validate_active_len("regular_small", input.regular_small.len(), input.active_len)?;
    validate_active_len(
        "irregular_large",
        input.irregular_large.len(),
        input.active_len,
    )?;
    validate_active_len(
        "irregular_small",
        input.irregular_small.len(),
        input.active_len,
    )?;
    validate_active_len("local_field", input.local_field.len(), input.active_len)?;
    for index in 0..input.active_len {
        let radius = input.radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "xsect_phiscf_radius",
                value: radius,
            });
        }
        validate_finite_real("xsect_phiscf_orbital_large", input.orbital_large[index])?;
        validate_finite_real("xsect_phiscf_orbital_small", input.orbital_small[index])?;
        validate_finite_complex(
            "xsect_phiscf_regular_large",
            index,
            input.regular_large[index],
        )?;
        validate_finite_complex(
            "xsect_phiscf_regular_small",
            index,
            input.regular_small[index],
        )?;
        validate_finite_complex(
            "xsect_phiscf_irregular_large",
            index,
            input.irregular_large[index],
        )?;
        validate_finite_complex(
            "xsect_phiscf_irregular_small",
            index,
            input.irregular_small[index],
        )?;
        validate_finite_real("xsect_phiscf_local_field", input.local_field[index])?;
    }
    Ok(())
}

fn validate_xsect_phiscf_contribution_plan_input(
    input: &XsphXsectPhiscfContributionPlanInput<'_>,
) -> Result<(), XsphError> {
    if input.active_orbital_count == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    validate_finite_complex("xsect_phiscf_momentum_squared", 0, input.momentum_squared)?;
    validate_finite_real("xsect_phiscf_edge_energy", input.edge_energy)?;
    validate_finite_real("xsect_phiscf_chemical_potential", input.chemical_potential)?;
    validate_finite_real(
        "xsect_phiscf_hole_orbital_energy",
        input.hole_orbital_energy,
    )?;
    validate_finite_real("xsect_phiscf_scale_function", input.scale_function)?;
    validate_active_len(
        "xsect_phiscf_orbital_kappas",
        input.orbital_kappas.len(),
        input.active_orbital_count,
    )?;
    validate_active_len(
        "xsect_phiscf_orbital_energy_counts",
        input.orbital_energy_counts.len(),
        input.active_orbital_count,
    )?;

    let mut max_energy_count = 0;
    for orbital_index in 0..input.active_orbital_count {
        if input.orbital_kappas[orbital_index] == 0 {
            return Err(XsphError::ZeroKappa);
        }
        max_energy_count = max_energy_count.max(input.orbital_energy_counts[orbital_index]);
    }
    if max_energy_count > 0
        && (input.occupied_energies.nrows() < max_energy_count
            || input.occupied_energies.ncols() < input.active_orbital_count)
    {
        return Err(XsphError::MatrixTooSmall {
            name: "xsect_phiscf_occupied_energies",
            required: [max_energy_count, input.active_orbital_count],
            actual: [
                input.occupied_energies.nrows(),
                input.occupied_energies.ncols(),
            ],
        });
    }
    if max_energy_count > 0
        && (input.occupation_fractions.nrows() < max_energy_count
            || input.occupation_fractions.ncols() < input.active_orbital_count)
    {
        return Err(XsphError::MatrixTooSmall {
            name: "xsect_phiscf_occupation_fractions",
            required: [max_energy_count, input.active_orbital_count],
            actual: [
                input.occupation_fractions.nrows(),
                input.occupation_fractions.ncols(),
            ],
        });
    }
    for orbital_index in 0..input.active_orbital_count {
        for energy_index in 0..input.orbital_energy_counts[orbital_index] {
            validate_finite_real(
                "xsect_phiscf_occupied_orbital_energy",
                input.occupied_energies[(energy_index, orbital_index)],
            )?;
            validate_finite_real(
                "xsect_phiscf_shell_occupation_fraction",
                input.occupation_fractions[(energy_index, orbital_index)],
            )?;
        }
    }
    Ok(())
}

fn validate_xsect_phiscf_radial_solver_setup_input(
    input: &XsphXsectPhiscfRadialSolverSetupInput<'_>,
) -> Result<(), XsphError> {
    if input.active_len == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    validate_finite_complex("xsect_phiscf_pole_energy", 0, input.pole_energy)?;
    validate_finite_real("xsect_phiscf_muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "xsect_phiscf_muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    validate_finite_real("xsect_phiscf_log_step", input.log_step)?;
    if input.log_step <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "xsect_phiscf_log_step",
            value: input.log_step,
        });
    }
    validate_finite_real("xsect_phiscf_origin_shift", input.origin_shift)?;
    validate_active_len("radii", input.radii.len(), input.active_len)?;
    if input.target_last_index_1based == 0 || input.target_last_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_target_last_index",
            index_1based: input.target_last_index_1based,
            active_len: input.active_len,
        });
    }
    for index in 0..input.active_len {
        let radius = input.radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "xsect_phiscf_radius",
                value: radius,
            });
        }
    }
    Ok(())
}

fn validate_xsect_phiscf_irregular_seed_input(
    input: XsphXsectPhiscfIrregularSeedInput,
) -> Result<(), XsphError> {
    if input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_finite_complex("xsect_phiscf_wave_number", 0, input.wave_number)?;
    validate_finite_real("xsect_phiscf_match_radius", input.match_radius)?;
    if input.match_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "xsect_phiscf_match_radius",
            value: input.match_radius,
        });
    }
    Ok(())
}

fn validate_xsect_phiscf_field_assembly_input(
    input: &XsphXsectPhiscfFieldAssemblyInput<'_>,
) -> Result<(), XsphError> {
    if input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_finite_complex("xsect_phiscf_wave_number", 0, input.wave_number)?;
    validate_active_len("radii", input.radii.len(), input.active_len)?;
    validate_active_len("regular_large", input.regular_large.len(), input.active_len)?;
    validate_active_len("regular_small", input.regular_small.len(), input.active_len)?;
    validate_active_len(
        "irregular_large",
        input.irregular_large.len(),
        input.active_len,
    )?;
    validate_active_len(
        "irregular_small",
        input.irregular_small.len(),
        input.active_len,
    )?;
    if input.match_index_1based == 0 || input.match_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_match_index",
            index_1based: input.match_index_1based,
            active_len: input.active_len,
        });
    }
    for index in 0..input.active_len {
        let radius = input.radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "xsect_phiscf_radius",
                value: radius,
            });
        }
        validate_finite_complex("regular_large", index, input.regular_large[index])?;
        validate_finite_complex("regular_small", index, input.regular_small[index])?;
        validate_finite_complex("irregular_large", index, input.irregular_large[index])?;
        validate_finite_complex("irregular_small", index, input.irregular_small[index])?;
    }
    Ok(())
}

fn validate_xsect_phiscf_nonzero_complex_result(
    name: &'static str,
    value: Complex,
) -> Result<(), XsphError> {
    validate_finite_complex(name, 0, value)?;
    if value == Complex::new(0.0, 0.0) {
        return Err(XsphError::ZeroComplexResult { name });
    }
    Ok(())
}

fn phiscf_dipole_final_kappa(
    initial_kappa: i32,
    dipole_delta: i32,
) -> Result<Option<i32>, XsphError> {
    if initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if !(-1..=1).contains(&dipole_delta) {
        return Err(XsphError::IntegerOutOfRange {
            name: "xsect_phiscf_dipole_delta",
            value: dipole_delta,
        });
    }
    let shifted = initial_kappa
        .checked_add(dipole_delta)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: initial_kappa,
        })?;
    let final_kappa = if dipole_delta == 0 {
        shifted.checked_neg().ok_or(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: initial_kappa,
        })?
    } else {
        shifted
    };
    Ok((final_kappa != 0).then_some(final_kappa))
}

fn feff_positive_float_to_usize(value: Real, name: &'static str) -> Result<usize, XsphError> {
    validate_finite_real(name, value)?;
    if value < 0.0 || value > usize::MAX as Real {
        return Err(XsphError::RealIntegerOutOfRange { name, value });
    }
    Ok(value.trunc() as usize)
}

fn phiscf_linear_fine_len(coarse_count: usize) -> Result<usize, XsphError> {
    coarse_count
        .checked_sub(1)
        .and_then(|intervals| intervals.checked_mul(5))
        .and_then(|offset| offset.checked_add(1))
        .ok_or(XsphError::SizeOutOfRange {
            name: "xsect_phiscf_coarse_count",
            value: coarse_count,
        })
}

fn phiscf_linear_fine_index(coarse_index: usize) -> usize {
    5 * coarse_index
}

fn phiscf_lipman_prefix_integral(
    values: &[Complex],
    radii: ArrayView1<'_, Real>,
    active_len: usize,
) -> Vec<Complex> {
    let mut integral = vec![Complex::new(0.0, 0.0); active_len];
    for index in 0..active_len.saturating_sub(1) {
        integral[index + 1] = integral[index]
            + (values[index] + values[index + 1]) * (radii[index + 1] - radii[index]) / 2.0;
    }
    integral
}

fn phiscf_lipman_tail_integral(
    values: &[Complex],
    radii: ArrayView1<'_, Real>,
    active_len: usize,
    match_index_1based: usize,
) -> Vec<Complex> {
    let mut integral = vec![Complex::new(0.0, 0.0); active_len];
    let match_index = match_index_1based - 1;
    for index in (0..match_index).rev() {
        integral[index] = integral[index + 1]
            + (values[index] + values[index + 1]) * (radii[index + 1] - radii[index]) / 2.0;
    }
    integral
}

fn solve_phiscf_scaled_linear_system(
    system: ArrayView2<'_, Complex>,
    rhs: ArrayView2<'_, Complex>,
) -> Result<Array2<Complex>, XsphError> {
    let lu = complex_lu_factor(system)?;
    Ok(complex_lu_solve(&lu, rhs)?)
}

fn phiscf_interpolated_solution_column(
    coarse_solution: ArrayView1<'_, Complex>,
    radii: ArrayView1<'_, Real>,
    fine_len: usize,
) -> Result<Array1<Complex>, XsphError> {
    let mut output = Array1::<Complex>::zeros(fine_len);
    for coarse_index in 0..coarse_solution.len() {
        let fine_index = phiscf_linear_fine_index(coarse_index);
        let current = coarse_solution[coarse_index] / radii[fine_index];
        validate_finite_complex("xsect_phiscf_solution", fine_index, current)?;
        output[fine_index] = current;
        if coarse_index > 0 {
            let previous = output[fine_index - 5];
            for offset in 1..=4 {
                let interpolated =
                    (previous * (5 - offset) as Real + current * offset as Real) / 5.0;
                let index = fine_index - 5 + offset;
                validate_finite_complex("xsect_phiscf_solution", index, interpolated)?;
                output[index] = interpolated;
            }
        }
    }
    Ok(output)
}

fn validate_xsect_radial_weights(
    name: &'static str,
    weights: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, weights.len(), active_len)?;
    for index in 0..active_len {
        validate_finite_real(name, weights[index])?;
    }
    Ok(())
}

fn validate_xsect_radial_pass_input(input: XsphXsectRadialPassInput) -> Result<(), XsphError> {
    validate_finite_real("photon_wave_number", input.photon_wave_number)?;
    validate_finite_real("screened_field_scale", input.screened_field_scale)?;
    if input.standard_potential && input.photon_wave_number <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "photon_wave_number",
            value: input.photon_wave_number,
        });
    }
    if input.standard_potential
        && input.kind == XsphXsectRadialPassKind::CentralCrossSection
        && input.screened_field_scale <= 0.0
    {
        return Err(XsphError::InvalidPositiveScalar {
            name: "screened_field_scale",
            value: input.screened_field_scale,
        });
    }
    Ok(())
}

fn apply_real_radial_weights(
    coupling: &[Complex],
    weights: ArrayView1<'_, Real>,
    active_len: usize,
) -> Vec<Complex> {
    (0..active_len)
        .map(|index| coupling[index] * weights[index])
        .collect()
}

fn validate_xray_bessel_input(input: &XsphXrayBesselTableInput<'_>) -> Result<(), XsphError> {
    validate_active_len("radii", input.radii.len(), input.active_len)?;
    if !input.photon_wave_number.is_finite() || input.photon_wave_number <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "photon_wave_number",
            value: input.photon_wave_number,
        });
    }
    for index in 0..input.active_len {
        let radius = input.radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "radius",
                value: radius,
            });
        }
    }
    Ok(())
}

fn validate_xsect_regular_solution_input(
    input: &XsphXsectRegularSolutionInput<'_>,
) -> Result<(), XsphError> {
    if input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len("regular_large", input.regular_large.len(), input.active_len)?;
    validate_active_len("regular_small", input.regular_small.len(), input.active_len)?;
    validate_finite_complex("wave_number", 0, input.wave_number)?;
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

fn validate_xsect_irregular_initial_condition_input(
    input: XsphXsectIrregularInitialConditionInput,
) -> Result<(), XsphError> {
    if input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if !input.muffin_tin_radius.is_finite() || input.muffin_tin_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    validate_finite_complex("phase_shift", 0, input.phase_shift)?;
    validate_finite_complex("wave_number", 0, input.wave_number)?;
    validate_finite_complex("bessel_j_l", 0, input.bessel_j_l)?;
    validate_finite_complex("neumann_l", 0, input.neumann_l)?;
    validate_finite_complex("bessel_j_l_plus_1", 0, input.bessel_j_l_plus_1)?;
    validate_finite_complex("neumann_l_plus_1", 0, input.neumann_l_plus_1)
}

fn validate_xsect_irregular_transform_input(
    input: &XsphXsectIrregularTransformInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("regular_large", input.regular_large.len(), input.active_len)?;
    validate_active_len("regular_small", input.regular_small.len(), input.active_len)?;
    validate_active_len(
        "irregular_large",
        input.irregular_large.len(),
        input.active_len,
    )?;
    validate_active_len(
        "irregular_small",
        input.irregular_small.len(),
        input.active_len,
    )?;
    validate_finite_complex("phase_shift", 0, input.phase_shift)?;
    for index in 0..input.active_len {
        validate_finite_complex("regular_large", index, input.regular_large[index])?;
        validate_finite_complex("regular_small", index, input.regular_small[index])?;
        validate_finite_complex("irregular_large", index, input.irregular_large[index])?;
        validate_finite_complex("irregular_small", index, input.irregular_small[index])?;
    }
    Ok(())
}

fn validate_xsect_output_normalization_input(
    input: &XsphXsectOutputNormalizationInput<'_>,
) -> Result<(), XsphError> {
    if !input.photon_energy.is_finite() || input.photon_energy <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "photon_energy",
            value: input.photon_energy,
        });
    }
    validate_finite_complex("wave_number", 0, input.wave_number)?;
    validate_finite_real("spectrum_norm", input.spectrum_norm)?;
    validate_finite_complex("cross_section", 0, input.cross_section)?;
    validate_active_len(
        "reduced_matrix_elements",
        input.reduced_matrix_elements.len(),
        input.active_channel_count,
    )?;
    validate_active_len(
        "phase_shifts",
        input.phase_shifts.len(),
        input.active_channel_count,
    )?;
    for index in 0..input.active_channel_count {
        validate_finite_complex(
            "reduced_matrix_elements",
            index,
            input.reduced_matrix_elements[index],
        )?;
        validate_finite_complex("phase_shifts", index, input.phase_shifts[index])?;
    }
    Ok(())
}

fn validate_xsect_cross_term_plan_input(
    input: &XsphXsectCrossTermPlanInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    if input.transition_index_1based == 0 || input.transition_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: input.transition_index_1based,
            active_len: input.active_len,
        });
    }
    Ok(())
}

fn validate_xsect_cross_term_accumulation_input(
    input: &XsphXsectCrossTermAccumulationInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    if input.transition_index_1based == 0 || input.transition_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: input.transition_index_1based,
            active_len: input.active_len,
        });
    }
    Ok(())
}

fn validate_xsect_cross_term_save_input(
    input: &XsphXsectCrossTermStateSaveInput<'_>,
) -> Result<(), XsphError> {
    validate_xsect_cross_term_iold(input.plan.iold, XsphXsectCrossTermMode::SaveCurrentForNext)?;
    validate_xsect_bcoef_transition_index(input.transition_index_1based)?;
    validate_xsect_bcoef_transition_index(input.plan.partner_index_1based)?;
    validate_active_len(
        "xsect_cross_term_regular_coupling",
        input.regular_coupling.len(),
        input.active_len,
    )?;
    validate_active_len(
        "xsect_cross_term_irregular_coupling",
        input.irregular_coupling.len(),
        input.active_len,
    )?;
    validate_finite_complex("xsect_cross_term_saved_integral", 0, input.radial_integral)?;
    validate_finite_complex("xsect_cross_term_saved_phase", 0, input.phase_shift)?;
    for index in 0..input.active_len {
        validate_finite_complex(
            "xsect_cross_term_regular_coupling",
            index,
            input.regular_coupling[index],
        )?;
        validate_finite_complex(
            "xsect_cross_term_irregular_coupling",
            index,
            input.irregular_coupling[index],
        )?;
    }
    Ok(())
}

fn validate_xsect_cross_term_reuse_input(
    input: &XsphXsectCrossTermStateReuseInput<'_>,
) -> Result<(), XsphError> {
    validate_xsect_cross_term_iold(
        input.plan.iold,
        XsphXsectCrossTermMode::UsePreviousForCurrent,
    )?;
    validate_xsect_bcoef_transition_index(input.transition_index_1based)?;
    validate_xsect_bcoef_transition_index(input.plan.partner_index_1based)?;
    validate_xsect_cross_term_saved_state(input.state)?;

    if input.state.transition_index_1based != input.plan.partner_index_1based {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_cross_term_saved_transition",
            index_1based: input.state.transition_index_1based,
            active_len: input.plan.partner_index_1based,
        });
    }
    if input.state.partner_index_1based != input.transition_index_1based {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_cross_term_state_partner",
            index_1based: input.state.partner_index_1based,
            active_len: input.transition_index_1based,
        });
    }

    Ok(())
}

fn validate_xsect_cross_term_iold(
    iold: i32,
    mode: XsphXsectCrossTermMode,
) -> Result<(), XsphError> {
    let expected = match mode {
        XsphXsectCrossTermMode::SaveCurrentForNext => 1,
        XsphXsectCrossTermMode::UsePreviousForCurrent => 2,
    };
    if iold != expected {
        return Err(XsphError::IntegerOutOfRange {
            name: "xsect_cross_term_iold",
            value: iold,
        });
    }
    Ok(())
}

fn validate_xsect_cross_term_saved_state(state: &XsphXsectCrossTermState) -> Result<(), XsphError> {
    validate_xsect_bcoef_transition_index(state.transition_index_1based)?;
    validate_xsect_bcoef_transition_index(state.partner_index_1based)?;
    let active_len = state.regular_coupling.len();
    validate_active_len("xsect_cross_term_regular_coupling", active_len, 1)?;
    validate_active_len(
        "xsect_cross_term_irregular_coupling",
        state.irregular_coupling.len(),
        active_len,
    )?;
    validate_finite_complex("xsect_cross_term_saved_integral", 0, state.radial_integral)?;
    validate_finite_complex("xsect_cross_term_saved_phase", 0, state.phase_shift)?;
    for index in 0..active_len {
        validate_finite_complex(
            "xsect_cross_term_regular_coupling",
            index,
            state.regular_coupling[index],
        )?;
        validate_finite_complex(
            "xsect_cross_term_irregular_coupling",
            index,
            state.irregular_coupling[index],
        )?;
    }
    Ok(())
}

fn validate_xsect_bcoef_cross_term_accumulation_input(
    input: &XsphXsectBcoefCrossTermAccumulationInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    if input.transition_index_1based == 0 || input.transition_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: input.transition_index_1based,
            active_len: input.active_len,
        });
    }
    validate_xsect_bcoef_trace_weights(input.trace_weights)?;
    Ok(())
}

fn validate_xsect_bcoef_cross_term_state_accumulation_input(
    input: &XsphXsectBcoefCrossTermStateAccumulationInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    if input.transition_index_1based == 0 || input.transition_index_1based > input.active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: input.transition_index_1based,
            active_len: input.active_len,
        });
    }
    validate_xsect_bcoef_trace_weights(input.trace_weights)?;
    Ok(())
}

fn validate_xsect_cross_term_state_reuse_payload(
    state_reuse: &XsphXsectCrossTermStateReuse<'_>,
    partner_index_1based: usize,
) -> Result<(), XsphError> {
    validate_xsect_bcoef_transition_index(state_reuse.saved_transition_index_1based)?;
    if state_reuse.saved_transition_index_1based != partner_index_1based {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "xsect_cross_term_saved_transition",
            index_1based: state_reuse.saved_transition_index_1based,
            active_len: partner_index_1based,
        });
    }
    validate_finite_complex(
        "xsect_cross_term_saved_integral",
        0,
        state_reuse.saved_radial_integral,
    )?;
    validate_finite_complex(
        "xsect_cross_term_saved_phase",
        0,
        state_reuse.saved_phase_shift,
    )?;
    Ok(())
}

fn xsect_active_cross_term_partner(
    transition_index_1based: usize,
    orbital_l: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<Option<usize>, XsphError> {
    validate_active_len("orbital_l", orbital_l.len(), active_len)?;
    if transition_index_1based == 0 || transition_index_1based > active_len {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: transition_index_1based,
            active_len,
        });
    }
    if transition_index_1based == 1 {
        return Ok(None);
    }
    let partner_index_1based = transition_index_1based - 1;
    if partner_index_1based > XSECT_BCOEF_TRANSITION_SLOTS {
        return Ok(None);
    }

    let partner_l = orbital_l[partner_index_1based - 1];
    let current_l = orbital_l[transition_index_1based - 1];
    if partner_l <= 0 || partner_l != current_l {
        return Ok(None);
    }

    Ok(Some(partner_index_1based))
}

fn validate_xsect_bcoef_diagonal_weights(
    diagonal_weights: ArrayView1<'_, Complex>,
) -> Result<(), XsphError> {
    validate_active_len(
        "xsect_bcoef_diagonal_weights",
        diagonal_weights.len(),
        XSECT_BCOEF_TRANSITION_SLOTS,
    )
}

fn validate_xsect_reduced_matrix_workspace(
    reduced_matrix_elements: ArrayView1<'_, Complex>,
    phase_shifts: ArrayView1<'_, Complex>,
) -> Result<(), XsphError> {
    validate_active_len(
        "xsect_reduced_matrix_workspace",
        reduced_matrix_elements.len(),
        XSECT_BCOEF_TRANSITION_SLOTS,
    )?;
    validate_active_len(
        "xsect_phase_workspace",
        phase_shifts.len(),
        XSECT_BCOEF_TRANSITION_SLOTS,
    )?;
    for index in 0..XSECT_BCOEF_TRANSITION_SLOTS {
        validate_finite_complex(
            "xsect_reduced_matrix_workspace",
            index,
            reduced_matrix_elements[index],
        )?;
        validate_finite_complex("xsect_phase_workspace", index, phase_shifts[index])?;
    }
    Ok(())
}

fn validate_xsect_bcoef_trace_weights(
    trace_weights: ArrayView2<'_, Complex>,
) -> Result<(), XsphError> {
    let shape = trace_weights.shape();
    if shape[0] < XSECT_BCOEF_TRANSITION_SLOTS || shape[1] < XSECT_BCOEF_TRANSITION_SLOTS {
        return Err(XsphError::MatrixTooSmall {
            name: "xsect_bcoef_trace_weights",
            required: [XSECT_BCOEF_TRANSITION_SLOTS, XSECT_BCOEF_TRANSITION_SLOTS],
            actual: [shape[0], shape[1]],
        });
    }
    Ok(())
}

fn xsect_transition_workspace_copy(workspace: ArrayView1<'_, Complex>) -> Array1<Complex> {
    Array1::from_iter((0..XSECT_BCOEF_TRANSITION_SLOTS).map(|index| workspace[index]))
}

fn xsect_bcoef_trace_weight(
    trace_weights: ArrayView2<'_, Complex>,
    transition2_index_1based: usize,
    transition1_index_1based: usize,
) -> Result<Complex, XsphError> {
    validate_xsect_bcoef_transition_index(transition2_index_1based)?;
    validate_xsect_bcoef_transition_index(transition1_index_1based)?;
    Ok(trace_weights[(transition2_index_1based - 1, transition1_index_1based - 1)])
}

fn validate_xsect_bcoef_transition_index(transition_index_1based: usize) -> Result<(), XsphError> {
    if transition_index_1based == 0 || transition_index_1based > XSECT_BCOEF_TRANSITION_SLOTS {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: transition_index_1based,
            active_len: XSECT_BCOEF_TRANSITION_SLOTS,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_xsect_density_base_input(
    final_kappa: i32,
    wave_number: Complex,
    regular_large: ArrayView1<'_, Complex>,
    regular_small: ArrayView1<'_, Complex>,
    irregular_large: ArrayView1<'_, Complex>,
    irregular_small: ArrayView1<'_, Complex>,
    radii: ArrayView1<'_, Real>,
    log_step: Real,
    norman_radius: Real,
    active_len: usize,
    integration_len: usize,
) -> Result<(), XsphError> {
    if final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len("radii", radii.len(), active_len)?;
    validate_solution_pair(
        "regular_large",
        regular_large,
        "regular_small",
        regular_small,
        active_len,
    )?;
    validate_solution_pair(
        "irregular_large",
        irregular_large,
        "irregular_small",
        irregular_small,
        active_len,
    )?;
    validate_integration_len(active_len, integration_len)?;
    validate_finite_complex("wave_number", 0, wave_number)?;
    validate_finite_real("log_step", log_step)?;
    if !norman_radius.is_finite() || norman_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveRadius {
            name: "norman_radius",
            value: norman_radius,
        });
    }
    for index in 0..active_len {
        let radius = radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "radius",
                value: radius,
            });
        }
    }
    Ok(())
}

fn validate_xsect_projected_density_input(
    input: &XsphXsectProjectedDensityInput<'_>,
) -> Result<(), XsphError> {
    validate_xsect_density_base_input(
        input.final_kappa,
        input.wave_number,
        input.regular_large,
        input.regular_small,
        input.irregular_large,
        input.irregular_small,
        input.radii,
        input.log_step,
        input.norman_radius,
        input.active_len,
        input.integration_len,
    )?;
    validate_active_len("atomic_large", input.atomic_large.len(), input.active_len)?;
    validate_active_len("atomic_small", input.atomic_small.len(), input.active_len)?;
    for index in 0..input.active_len {
        validate_finite_real("atomic_large", input.atomic_large[index])?;
        validate_finite_real("atomic_small", input.atomic_small[index])?;
    }
    Ok(())
}

fn validate_integration_len(active_len: usize, integration_len: usize) -> Result<(), XsphError> {
    if integration_len == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    if integration_len > active_len {
        return Err(XsphError::LengthTooShort {
            name: "active_len",
            required: integration_len,
            actual: active_len,
        });
    }
    Ok(())
}

fn validate_jas_orthogonality_input(
    input: &XsphJasOrthogonalityCorrectionInput<'_>,
) -> Result<(), XsphError> {
    validate_jas_common_inputs(
        input.radii,
        input.large_component,
        input.small_component,
        input.log_step,
        input.active_len,
    )?;
    let angular_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    let q_shape = input.q_bessel.shape();
    if q_shape[0] < input.active_len || q_shape[1] < angular_count || q_shape[2] == 0 {
        return Err(XsphError::ShapeTooSmall {
            name: "q_bessel",
            required: [input.active_len, angular_count, 1],
            actual: [q_shape[0], q_shape[1], q_shape[2]],
        });
    }
    for index in 0..input.active_len {
        for angular_l in 0..=input.ljmax {
            for q_index in 0..q_shape[2] {
                validate_finite_real("q_bessel", input.q_bessel[(index, angular_l, q_index)])?;
            }
        }
    }
    Ok(())
}

fn validate_jas_common_inputs(
    radii: ArrayView1<'_, Real>,
    large_component: ArrayView1<'_, Real>,
    small_component: ArrayView1<'_, Real>,
    log_step: Real,
    active_len: usize,
) -> Result<(), XsphError> {
    validate_active_len("radii", radii.len(), active_len)?;
    if active_len < 2 {
        return Err(XsphError::LengthTooShort {
            name: "active_len",
            required: 2,
            actual: active_len,
        });
    }
    validate_active_len("large_component", large_component.len(), active_len)?;
    validate_active_len("small_component", small_component.len(), active_len)?;
    validate_finite_real("log_step", log_step)?;
    for index in 0..active_len {
        let radius = radii[index];
        if !radius.is_finite() || radius <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "radius",
                value: radius,
            });
        }
        validate_finite_real("large_component", large_component[index])?;
        validate_finite_real("small_component", small_component[index])?;
    }
    Ok(())
}

fn validate_jas_radial_integral_input(
    input: &XsphJasRadialIntegralInput<'_>,
) -> Result<(), XsphError> {
    if input.initial_kappa == 0 || input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_jas_common_inputs(
        input.radii,
        input.initial_large,
        input.initial_small,
        input.log_step,
        input.active_len,
    )?;
    validate_solution_pair(
        "final_large_regular",
        input.final_large_regular,
        "final_small_regular",
        input.final_small_regular,
        input.active_len,
    )?;
    let angular_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "needed_multipoles",
        input.needed_multipoles.len(),
        angular_count,
    )?;
    validate_active_len(
        "orthogonality_correction",
        input.orthogonality_correction.len(),
        angular_count,
    )?;
    let q_shape = input.q_bessel.shape();
    if q_shape[0] < input.active_len || q_shape[1] < angular_count {
        return Err(XsphError::MatrixTooSmall {
            name: "q_bessel",
            required: [input.active_len, angular_count],
            actual: [q_shape[0], q_shape[1]],
        });
    }
    for angular_l in 0..=input.ljmax {
        let needed = input.needed_multipoles[angular_l];
        if needed < 0 {
            return Err(XsphError::NegativeAngularMomentum {
                name: "needed_multipoles",
                index: angular_l,
                value: needed,
            });
        }
        validate_finite_complex(
            "orthogonality_correction",
            angular_l,
            input.orthogonality_correction[angular_l],
        )?;
        for index in 0..input.active_len {
            validate_finite_real("q_bessel", input.q_bessel[(index, angular_l)])?;
        }
    }
    Ok(())
}

fn validate_jas_overlap_input(input: &XsphJasOverlapInput<'_>) -> Result<(), XsphError> {
    validate_jas_common_inputs(
        input.radii,
        input.initial_large,
        input.initial_small,
        input.log_step,
        input.active_len,
    )?;
    validate_solution_pair(
        "final_large",
        input.final_large,
        "final_small",
        input.final_small,
        input.active_len,
    )?;
    let _ = jas_overlap_near_origin_power(input.initial_l, input.final_l)?;
    Ok(())
}

fn validate_jas_radial_cross_integral_input(
    input: &XsphJasRadialCrossIntegralInput<'_>,
) -> Result<(), XsphError> {
    if input.initial_kappa == 0 || input.final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_jas_common_inputs(
        input.radii,
        input.initial_large,
        input.initial_small,
        input.log_step,
        input.active_len,
    )?;
    validate_solution_pair(
        "final_large_irregular",
        input.final_large_irregular,
        "final_small_irregular",
        input.final_small_irregular,
        input.active_len,
    )?;
    let angular_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "needed_multipoles",
        input.needed_multipoles.len(),
        angular_count,
    )?;
    validate_active_len(
        "orthogonality_correction",
        input.orthogonality_correction.len(),
        angular_count,
    )?;
    let q_shape = input.q_bessel.shape();
    if q_shape[0] < input.active_len || q_shape[1] < angular_count {
        return Err(XsphError::MatrixTooSmall {
            name: "q_bessel",
            required: [input.active_len, angular_count],
            actual: [q_shape[0], q_shape[1]],
        });
    }
    let regular_shape = input.regular_coupling.shape();
    if regular_shape[0] < input.active_len || regular_shape[1] < angular_count {
        return Err(XsphError::MatrixTooSmall {
            name: "regular_coupling",
            required: [input.active_len, angular_count],
            actual: [regular_shape[0], regular_shape[1]],
        });
    }

    for angular_l in 0..=input.ljmax {
        let needed = input.needed_multipoles[angular_l];
        if needed < 0 {
            return Err(XsphError::NegativeAngularMomentum {
                name: "needed_multipoles",
                index: angular_l,
                value: needed,
            });
        }
        validate_finite_complex(
            "orthogonality_correction",
            angular_l,
            input.orthogonality_correction[angular_l],
        )?;
        for index in 0..input.active_len {
            validate_finite_real("q_bessel", input.q_bessel[(index, angular_l)])?;
            validate_finite_complex(
                "regular_coupling",
                index,
                input.regular_coupling[(index, angular_l)],
            )?;
        }
    }
    Ok(())
}

fn jas_near_origin_power(initial_l: usize, angular_l: usize) -> Result<Real, XsphError> {
    let value = initial_l
        .checked_mul(2)
        .and_then(|value| value.checked_add(angular_l))
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::SizeOutOfRange {
            name: "jas_near_origin_power",
            value: initial_l,
        })?;
    Ok(value as Real)
}

fn jas_radial_near_origin_power(
    initial_l: usize,
    final_l: usize,
    angular_l: usize,
) -> Result<Real, XsphError> {
    let value = final_l
        .checked_add(initial_l)
        .and_then(|value| value.checked_add(angular_l))
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::SizeOutOfRange {
            name: "jas_radial_near_origin_power",
            value: initial_l,
        })?;
    Ok(value as Real)
}

fn jas_overlap_near_origin_power(initial_l: usize, final_l: i32) -> Result<Real, XsphError> {
    let initial_l_i32 = i32::try_from(initial_l).map_err(|_| XsphError::SizeOutOfRange {
        name: "initial_l",
        value: initial_l,
    })?;
    let value = initial_l_i32
        .checked_add(final_l)
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "jas_overlap_near_origin_power",
            value: final_l,
        })?;
    Ok(Real::from(value))
}

fn jas_reduced_coupling(input: &XsphJasRadialIntegralInput<'_>, angular_l: usize) -> Vec<Complex> {
    (0..input.active_len)
        .map(|index| {
            let bessel = Complex::new(input.q_bessel[(index, angular_l)], 0.0);
            let bessel = if input.initial_kappa == input.final_kappa {
                bessel - input.orthogonality_correction[angular_l]
            } else {
                bessel
            };
            let spinor_overlap = input.final_large_regular[index] * input.initial_large[index]
                + input.final_small_regular[index] * input.initial_small[index];
            bessel * spinor_overlap
        })
        .collect()
}

fn jas_irregular_coupling(
    input: &XsphJasRadialCrossIntegralInput<'_>,
    angular_l: usize,
) -> Vec<Complex> {
    (0..input.active_len)
        .map(|index| {
            let bessel = Complex::new(input.q_bessel[(index, angular_l)], 0.0);
            let bessel = if input.initial_kappa == input.final_kappa {
                bessel - input.orthogonality_correction[angular_l]
            } else {
                bessel
            };
            let spinor_overlap = input.final_large_irregular[index] * input.initial_large[index]
                + input.final_small_irregular[index] * input.initial_small[index];
            bessel * spinor_overlap
        })
        .collect()
}

fn jas_radial_cross_first_power(initial_l: usize, final_l: usize) -> Result<Real, XsphError> {
    let value = final_l
        .checked_add(initial_l)
        .and_then(|value| value.checked_add(2))
        .ok_or(XsphError::SizeOutOfRange {
            name: "jas_radial_cross_near_origin_power",
            value: initial_l,
        })?;
    Ok(value as Real)
}

fn jas_radial_cross_second_power(initial_l: usize, final_l: usize) -> Result<Real, XsphError> {
    let first_power = final_l
        .checked_add(initial_l)
        .and_then(|value| value.checked_add(2))
        .ok_or(XsphError::SizeOutOfRange {
            name: "jas_radial_cross_near_origin_power",
            value: initial_l,
        })?;
    let value = first_power
        .checked_add(1)
        .and_then(|value| value.checked_add(initial_l))
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_sub(final_l))
        .ok_or(XsphError::SizeOutOfRange {
            name: "jas_radial_cross_near_origin_power",
            value: first_power,
        })?;
    Ok(value as Real)
}

fn xray_bessel_series(argument: Real, angular_l: usize) -> Real {
    debug_assert!(angular_l < RADINT_BESSEL_ROWS);

    let mut double_factorial = 1.0;
    for factor in (1..=2 * angular_l + 1).step_by(2) {
        double_factorial *= factor as Real;
    }
    let mut term = argument.powi(angular_l as i32) / double_factorial;
    let mut sum = term;

    for iteration in 0..XRAY_BESSEL_SERIES_MAX_ITERATIONS {
        let denominator =
            2.0 * (iteration as Real + 1.0) * (2 * angular_l + 2 * iteration + 3) as Real;
        term *= -argument * argument / denominator;
        sum += term;
        if term.abs() <= XRAY_BESSEL_SERIES_TOLERANCE * sum.abs().max(1.0) {
            break;
        }
    }

    sum
}

fn xray_bessel_formula(argument: Real) -> [Real; RADINT_BESSEL_ROWS] {
    let sinx = argument.sin();
    let cosx = argument.cos();
    let inverse = argument.recip();
    let inverse_squared = inverse * inverse;
    let inverse_cubed = inverse_squared * inverse;

    [
        sinx * inverse,
        sinx * inverse_squared - cosx * inverse,
        sinx * (3.0 * inverse_cubed - inverse) - 3.0 * cosx * inverse_squared,
    ]
}

fn validate_radial_integral_input(input: &XsphRadialIntegralInput<'_>) -> Result<(), XsphError> {
    validate_radial_base_inputs(
        input.initial_kappa,
        input.final_kappa,
        input.initial_large,
        input.initial_small,
        input.xray_bessel,
        input.radii,
        input.log_step,
        input.active_len,
    )?;
    validate_solution_pair(
        "final_large_regular",
        input.final_large_regular,
        "final_small_regular",
        input.final_small_regular,
        input.active_len,
    )?;
    Ok(())
}

fn validate_radial_cross_integral_input(
    input: &XsphRadialCrossIntegralInput<'_>,
) -> Result<(), XsphError> {
    validate_radial_base_inputs(
        input.initial_kappa,
        input.final_kappa,
        input.initial_large,
        input.initial_small,
        input.xray_bessel,
        input.radii,
        input.log_step,
        input.active_len,
    )?;
    validate_solution_pair(
        "final_large_regular",
        input.final_large_regular,
        "final_small_regular",
        input.final_small_regular,
        input.active_len,
    )?;
    validate_solution_pair(
        "final_large_irregular",
        input.final_large_irregular,
        "final_small_irregular",
        input.final_small_irregular,
        input.active_len,
    )?;
    match input.branch {
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular => {}
        XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling,
        } => validate_stored_coupling(
            "stored_regular_coupling",
            stored_regular_coupling,
            input.active_len,
        )?,
        XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling,
        } => validate_stored_coupling(
            "stored_irregular_coupling",
            stored_irregular_coupling,
            input.active_len,
        )?,
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_radial_base_inputs(
    initial_kappa: i32,
    final_kappa: i32,
    initial_large: ArrayView1<'_, Real>,
    initial_small: ArrayView1<'_, Real>,
    xray_bessel: ArrayView2<'_, Real>,
    radii: ArrayView1<'_, Real>,
    log_step: Real,
    active_len: usize,
) -> Result<(), XsphError> {
    if initial_kappa == 0 || final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len("radii", radii.len(), active_len)?;
    validate_active_len("initial_large", initial_large.len(), active_len)?;
    validate_active_len("initial_small", initial_small.len(), active_len)?;
    let bessel_shape = xray_bessel.shape();
    if bessel_shape[0] < RADINT_BESSEL_ROWS || bessel_shape[1] < active_len {
        return Err(XsphError::MatrixTooSmall {
            name: "xray_bessel",
            required: [RADINT_BESSEL_ROWS, active_len],
            actual: [bessel_shape[0], bessel_shape[1]],
        });
    }

    validate_finite_real("log_step", log_step)?;
    for index in 0..active_len {
        validate_finite_real("initial_large", initial_large[index])?;
        validate_finite_real("initial_small", initial_small[index])?;
        validate_finite_real("radius", radii[index])?;
        for row in 0..RADINT_BESSEL_ROWS {
            validate_finite_real("xray_bessel", xray_bessel[(row, index)])?;
        }
    }
    Ok(())
}

fn validate_solution_pair(
    large_name: &'static str,
    large: ArrayView1<'_, Complex>,
    small_name: &'static str,
    small: ArrayView1<'_, Complex>,
    active_len: usize,
) -> Result<(), XsphError> {
    validate_active_len(large_name, large.len(), active_len)?;
    validate_active_len(small_name, small.len(), active_len)?;
    for index in 0..active_len {
        validate_finite_complex(large_name, index, large[index])?;
        validate_finite_complex(small_name, index, small[index])?;
    }
    Ok(())
}

fn validate_stored_coupling(
    name: &'static str,
    coupling: ArrayView1<'_, Complex>,
    active_len: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, coupling.len(), active_len)?;
    for index in 0..active_len {
        validate_finite_complex(name, index, coupling[index])?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct XsphXsectRelativisticScales {
    small_component_factor: Complex,
    relativistic_scale: Complex,
}

fn xsph_xsect_relativistic_scales(
    wave_number: Complex,
    final_kappa: i32,
) -> Result<XsphXsectRelativisticScales, XsphError> {
    let one = Complex::new(1.0, 0.0);
    let zero = Complex::new(0.0, 0.0);
    let alpha_scaled = wave_number * super::XSPH_FINE_STRUCTURE_ALPHA;
    let denominator = one + (one + alpha_scaled * alpha_scaled).sqrt();
    validate_finite_complex("xsect_small_component_denominator", 0, denominator)?;
    if denominator == zero {
        return Err(XsphError::ZeroComplexResult {
            name: "xsect_small_component_denominator",
        });
    }

    let sign = if final_kappa > 0 { 1.0 } else { -1.0 };
    let small_component_factor = sign * alpha_scaled / denominator;
    validate_finite_complex("xsect_small_component_factor", 0, small_component_factor)?;

    let relativistic_denominator = (one + small_component_factor * small_component_factor).sqrt();
    validate_finite_complex(
        "xsect_relativistic_scale_denominator",
        0,
        relativistic_denominator,
    )?;
    if relativistic_denominator == zero {
        return Err(XsphError::ZeroComplexResult {
            name: "xsect_relativistic_scale_denominator",
        });
    }

    let relativistic_scale = one / relativistic_denominator;
    validate_finite_complex("xsect_relativistic_scale", 0, relativistic_scale)?;

    Ok(XsphXsectRelativisticScales {
        small_component_factor,
        relativistic_scale,
    })
}

fn xsph_xsect_density_prefactor(
    final_l: usize,
    final_kappa: i32,
    wave_number: Complex,
) -> Result<Complex, XsphError> {
    let scales = xsph_xsect_relativistic_scales(wave_number, final_kappa)?;
    let denominator =
        Complex::new(1.0, 0.0) + scales.small_component_factor * scales.small_component_factor;
    validate_finite_complex("xsect_density_prefactor_denominator", 0, denominator)?;
    if denominator == Complex::new(0.0, 0.0) {
        return Err(XsphError::ZeroComplexResult {
            name: "xsect_density_prefactor_denominator",
        });
    }

    let prefactor = (2.0 * final_l as Real + 1.0) / std::f64::consts::PI * 4.0 * wave_number
        / denominator
        / super::XSPH_HARTREE_EV;
    validate_finite_complex("xsect_density_prefactor", 0, prefactor)?;
    Ok(prefactor)
}

fn xsect_transition_multipole_order(multipole: XsphTransitionMultipole) -> usize {
    match multipole {
        XsphTransitionMultipole::ElectricDipole | XsphTransitionMultipole::MagneticDipole => 1,
        XsphTransitionMultipole::ElectricQuadrupole => 2,
    }
}

fn xsect_transition_slot_offset(multipole: XsphTransitionMultipole) -> i32 {
    match multipole {
        XsphTransitionMultipole::ElectricDipole => 2,
        XsphTransitionMultipole::MagneticDipole | XsphTransitionMultipole::ElectricQuadrupole => 6,
    }
}

fn xsect_multipole_enabled(
    multipole: XsphTransitionMultipole,
    selected_higher_multipole: Option<XsphTransitionMultipole>,
) -> bool {
    multipole == XsphTransitionMultipole::ElectricDipole
        || selected_higher_multipole.is_some_and(|selected| selected == multipole)
}

fn xsect_l2lp_skips(
    transition_direction: i32,
    initial_kappa: i32,
    transition_index_1based: usize,
) -> bool {
    match transition_direction {
        1 => {
            (initial_kappa < 0 && transition_index_1based >= 3)
                || (initial_kappa > 0 && transition_index_1based != 3)
        }
        -1 => {
            (initial_kappa < 0 && transition_index_1based != 3)
                || (initial_kappa > 0 && transition_index_1based >= 3)
        }
        _ => false,
    }
}

fn xsect_stores_reduced_matrix(
    multipole: XsphTransitionMultipole,
    selected_higher_multipole: Option<XsphTransitionMultipole>,
) -> bool {
    multipole == XsphTransitionMultipole::ElectricDipole
        || selected_higher_multipole.is_some_and(|selected| selected == multipole)
}

fn xsect_radial_integral_mode_from_ifl(feff_ifl: i32) -> Result<XsphRadialIntegralMode, XsphError> {
    match feff_ifl {
        1..=4 => Ok(XsphRadialIntegralMode::RelativisticMatrixElement),
        -4..=-1 => Ok(XsphRadialIntegralMode::NonRelativisticMatrixElement),
        _ => Err(XsphError::IntegerOutOfRange {
            name: "xsect_radial_ifl",
            value: feff_ifl,
        }),
    }
}

fn xsph_xsect_projector_lookup(
    projector_map: ArrayView1<'_, i32>,
    min_kappa: i32,
    kappa: i32,
) -> Result<i32, XsphError> {
    let offset = kappa
        .checked_sub(min_kappa)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "projector_kappa",
            value: kappa,
        })?;
    let index = usize::try_from(offset).map_err(|_| XsphError::IntegerOutOfRange {
        name: "projector_kappa",
        value: kappa,
    })?;
    if index >= projector_map.len() {
        return Err(XsphError::LengthTooShort {
            name: "orbital_projector_map",
            required: index + 1,
            actual: projector_map.len(),
        });
    }
    Ok(projector_map[index])
}

fn xsph_xsect_cumulative_trapezoid(
    radii: ArrayView1<'_, Real>,
    values: ArrayView1<'_, Complex>,
    active_len: usize,
) -> Result<Array1<Complex>, XsphError> {
    validate_active_len("radii", radii.len(), active_len)?;
    validate_active_len("values", values.len(), active_len)?;
    let mut overlap = Array1::<Complex>::zeros(active_len);
    overlap[0] = values[0] * radii[0];
    validate_finite_complex("xsect_projected_overlap", 0, overlap[0])?;
    for index in 1..active_len {
        let delta = radii[index] - radii[index - 1];
        validate_finite_real("xsect_projected_radius_delta", delta)?;
        overlap[index] = overlap[index - 1] + (values[index] + values[index - 1]) * delta;
        validate_finite_complex("xsect_projected_overlap", index, overlap[index])?;
    }
    Ok(overlap)
}

fn radint_factors_for_mode(
    mode: XsphRadialIntegralMode,
    final_kappa: i32,
    initial_kappa: i32,
    multipole: XsphTransitionMultipole,
) -> Result<RadintMultipoleFactors, XsphError> {
    match mode {
        XsphRadialIntegralMode::RelativisticMatrixElement => {
            relativistic_radint_factors(final_kappa, initial_kappa, multipole)
        }
        XsphRadialIntegralMode::NonRelativisticMatrixElement => {
            nonrelativistic_radint_factors(final_kappa, initial_kappa, multipole)
        }
    }
}

fn relativistic_radint_factors(
    final_kappa: i32,
    initial_kappa: i32,
    multipole: XsphTransitionMultipole,
) -> Result<RadintMultipoleFactors, XsphError> {
    match multipole {
        XsphTransitionMultipole::ElectricDipole => {
            let j0 = xsph_relativistic_multipole_factors(final_kappa, initial_kappa, 0, 1)?;
            let j2 = xsph_relativistic_multipole_factors(final_kappa, initial_kappa, 2, 1)?;
            Ok(radint_factors(j0, j2))
        }
        XsphTransitionMultipole::ElectricQuadrupole => {
            let j1 = xsph_relativistic_multipole_factors(final_kappa, initial_kappa, 1, 2)?;
            Ok(radint_factors(j1, zero_multipole_factors()))
        }
        XsphTransitionMultipole::MagneticDipole => {
            let j1 = xsph_relativistic_multipole_factors(final_kappa, initial_kappa, 1, 1)?;
            Ok(radint_factors(j1, zero_multipole_factors()))
        }
    }
}

fn nonrelativistic_radint_factors(
    final_kappa: i32,
    initial_kappa: i32,
    multipole: XsphTransitionMultipole,
) -> Result<RadintMultipoleFactors, XsphError> {
    let angular_l = match multipole {
        XsphTransitionMultipole::ElectricDipole => 1_i32,
        XsphTransitionMultipole::ElectricQuadrupole => 2_i32,
        XsphTransitionMultipole::MagneticDipole => {
            // FEFF XSPH/radint.f90 stops for mult=1 when ifl < 0; do not
            // synthesize a nonrelativistic M1 path that the reference lacks.
            return Err(XsphError::UnsupportedRadialMultipole {
                mode: XsphRadialIntegralMode::NonRelativisticMatrixElement,
                multipole,
            });
        }
    };
    let ji2 = doubled_j_from_kappa("initial_kappa", initial_kappa)?;
    let jf2 = doubled_j_from_kappa("final_kappa", final_kappa)?;
    validate_cwig3j_doubled_argument("initial_kappa", initial_kappa, ji2)?;
    validate_cwig3j_doubled_argument("final_kappa", final_kappa, jf2)?;

    let angular_l2 = angular_l
        .checked_mul(2)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole_l",
            value: angular_l,
        })?;
    validate_cwig3j_doubled_argument("multipole_l", angular_l, angular_l2)?;

    let final_abs = final_kappa
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "final_kappa",
            value: final_kappa,
        })?;
    let temp = (f64::from((ji2 + 1) * (jf2 + 1))).sqrt()
        * wigner_3j(jf2, angular_l2, ji2, 1, 0, 2)?
        * alternating_sign(final_abs);

    let bessel_l = angular_l - 1;
    validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
    validate_cwig3j_integer_argument("multipole_l", angular_l)?;
    let xm1 = Complex::new(
        temp * f64::from(angular_l2 + 1) * f64::from(2 * bessel_l + 1),
        0.0,
    ) * imaginary_unit_power(bessel_l)
        * wigner_3j(bessel_l, 1, angular_l, 0, 0, 1)?
        * wigner_3j(bessel_l, 1, angular_l, 0, 1, 1)?;

    Ok(RadintMultipoleFactors {
        xm1,
        xm2: Complex::new(0.0, 0.0),
        xm3: Complex::new(0.0, 0.0),
        xm4: Complex::new(0.0, 0.0),
    })
}

fn radial_coupling(
    input: RadialCouplingInput<'_>,
    factors: RadintMultipoleFactors,
) -> Vec<Complex> {
    (0..input.active_len)
        .map(|index| match input.mode {
            XsphRadialIntegralMode::RelativisticMatrixElement => {
                relativistic_coupling_sample(input, factors, index)
            }
            XsphRadialIntegralMode::NonRelativisticMatrixElement => {
                nonrelativistic_coupling_sample(input, factors, index)
            }
        })
        .collect()
}

fn relativistic_coupling_sample(
    input: RadialCouplingInput<'_>,
    factors: RadintMultipoleFactors,
    index: usize,
) -> Complex {
    let large_initial = input.initial_large[index];
    let small_initial = input.initial_small[index];
    let large_final = input.final_large[index];
    let small_final = input.final_small[index];
    let bf0 = input.xray_bessel[(0, index)];
    let bf1 = input.xray_bessel[(1, index)];
    let bf2 = input.xray_bessel[(2, index)];

    match input.multipole {
        XsphTransitionMultipole::ElectricDipole => {
            small_final * large_initial * (factors.xm2 * bf0 + factors.xm4 * bf2)
                + large_final * small_initial * (factors.xm1 * bf0 + factors.xm3 * bf2)
        }
        XsphTransitionMultipole::MagneticDipole | XsphTransitionMultipole::ElectricQuadrupole => {
            (small_final * large_initial * factors.xm2 + large_final * small_initial * factors.xm1)
                * bf1
        }
    }
}

fn nonrelativistic_coupling_sample(
    input: RadialCouplingInput<'_>,
    factors: RadintMultipoleFactors,
    index: usize,
) -> Complex {
    let transition_factor = match input.multipole {
        XsphTransitionMultipole::ElectricDipole => {
            factors.xm1 * input.xray_bessel[(0, index)]
                + factors.xm3 * input.xray_bessel[(2, index)]
        }
        XsphTransitionMultipole::ElectricQuadrupole => factors.xm1 * input.xray_bessel[(1, index)],
        XsphTransitionMultipole::MagneticDipole => Complex::new(0.0, 0.0),
    } * Complex::new(0.0, 1.0);

    let spinor_overlap = input.final_large[index] * input.initial_large[index]
        + input.final_small[index] * input.initial_small[index];
    spinor_overlap * transition_factor * input.radii[index]
}

fn radial_near_origin_power(
    initial_kappa: i32,
    final_kappa: i32,
    multipole: XsphTransitionMultipole,
) -> Result<Real, XsphError> {
    let mut power = l_from_kappa(initial_kappa)?
        .checked_add(l_from_kappa(final_kappa)?)
        .and_then(|value| value.checked_add(2))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "radial_near_origin_power",
            value: initial_kappa,
        })?;
    if multipole != XsphTransitionMultipole::ElectricDipole {
        power = power.checked_add(1).ok_or(XsphError::IntegerOutOfRange {
            name: "radial_near_origin_power",
            value: power,
        })?;
    }
    Ok(f64::from(power))
}

fn integrate_radial_cross_couplings(
    input: &XsphRadialCrossIntegralInput<'_>,
    regular_coupling: Vec<Complex>,
    irregular_coupling: Vec<Complex>,
) -> Result<XsphRadialCrossIntegral, XsphError> {
    let first_power = first_cross_near_origin_power(input.initial_kappa, input.final_kappa)?;
    let second_power = second_cross_near_origin_power(input.initial_kappa, input.final_kappa)?;
    let radii = active_real_prefix(input.radii, input.active_len);
    let mut prefix_integral = Vec::with_capacity(input.active_len);
    let mut weighted_irregular = Vec::with_capacity(input.active_len);

    let mut prefix = regular_coupling[0] * (2.0 * radii[0] / (first_power + 1.0));
    prefix_integral.push(prefix);
    weighted_irregular.push(irregular_coupling[0] * prefix);
    for index in 1..input.active_len {
        prefix += (regular_coupling[index - 1] + regular_coupling[index])
            * (radii[index] - radii[index - 1]);
        prefix_integral.push(prefix);
        weighted_irregular.push(irregular_coupling[index] * prefix);
    }

    let zeros = vec![Complex::new(0.0, 0.0); input.active_len];
    let value = csomm(
        &radii,
        &zeros,
        &weighted_irregular,
        input.log_step,
        second_power,
        0,
    )?;

    Ok(XsphRadialCrossIntegral {
        value,
        regular_coupling: Array1::from_vec(regular_coupling),
        irregular_coupling: Array1::from_vec(irregular_coupling),
        regular_prefix_integral: Array1::from_vec(prefix_integral),
        weighted_irregular_coupling: Array1::from_vec(weighted_irregular),
        first_near_origin_power: first_power,
        second_near_origin_power: second_power,
    })
}

fn first_cross_near_origin_power(initial_kappa: i32, final_kappa: i32) -> Result<Real, XsphError> {
    let power = l_from_kappa(final_kappa)?
        .checked_add(l_from_kappa(initial_kappa)?)
        .and_then(|value| value.checked_add(2))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "radial_cross_near_origin_power",
            value: initial_kappa,
        })?;
    Ok(f64::from(power))
}

fn second_cross_near_origin_power(initial_kappa: i32, final_kappa: i32) -> Result<Real, XsphError> {
    let linit = l_from_kappa(initial_kappa)?;
    let lfin = l_from_kappa(final_kappa)?;
    let first_power = lfin
        .checked_add(linit)
        .and_then(|value| value.checked_add(2))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "radial_cross_near_origin_power",
            value: initial_kappa,
        })?;
    let second_power = first_power
        .checked_add(1)
        .and_then(|value| value.checked_add(linit))
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_sub(lfin))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "radial_cross_near_origin_power",
            value: first_power,
        })?;
    Ok(f64::from(second_power))
}

fn l_from_kappa(kappa: i32) -> Result<i32, XsphError> {
    if kappa > 0 {
        Ok(kappa)
    } else {
        kappa
            .checked_neg()
            .and_then(|value| value.checked_sub(1))
            .ok_or(XsphError::IntegerOutOfRange {
                name: "kappa",
                value: kappa,
            })
    }
}

fn radint_factors(
    primary: XsphRelativisticMultipoleFactors,
    secondary: XsphRelativisticMultipoleFactors,
) -> RadintMultipoleFactors {
    RadintMultipoleFactors {
        xm1: primary.p_q_prime,
        xm2: primary.q_p_prime,
        xm3: secondary.p_q_prime,
        xm4: secondary.q_p_prime,
    }
}

fn zero_multipole_factors() -> XsphRelativisticMultipoleFactors {
    XsphRelativisticMultipoleFactors {
        p_q_prime: Complex::new(0.0, 0.0),
        q_p_prime: Complex::new(0.0, 0.0),
    }
}

fn active_real_prefix(values: ArrayView1<'_, Real>, active_len: usize) -> Vec<Real> {
    values.iter().take(active_len).copied().collect()
}

fn active_complex_prefix(values: ArrayView1<'_, Complex>, active_len: usize) -> Vec<Complex> {
    values.iter().take(active_len).copied().collect()
}

fn alternating_sign(exponent: i32) -> Real {
    if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}
