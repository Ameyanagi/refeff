use super::*;

struct BoundOrbitalPass {
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
    matching_small_component: Real,
    energy_correction: Option<AtomicDiracEnergyCorrection>,
    matching_index_1based: usize,
    max_index_1based: usize,
    scan_index_1based: usize,
    needs_reintegration: bool,
}

/// Compose the FEFF `ATOM/soldir.f90` bound-orbital shooting loop.
pub fn atomic_dirac_bound_orbital(
    input: AtomicDiracBoundOrbitalInput<'_>,
) -> Result<AtomicDiracBoundOrbital, AtomMathError> {
    let entry = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: input.asymptotic_large_component,
        method: input.method,
    })?;
    let setup = atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
        energy: input.energy,
        origin_power: input.origin_power,
        initial_large_coefficient: input.initial_large_coefficient,
        initial_small_coefficient: input.initial_small_coefficient,
        principal_quantum_number: input.principal_quantum_number,
        kappa: input.kappa,
        speed_of_light: input.speed_of_light,
        method: input.method,
        radii: input.radii,
        potential: input.potential,
        potential_coefficients: input.potential_coefficients,
        active_len: input.active_len,
    })?;
    let target_node_count = usize::try_from(setup.target_nodes).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;

    let mut method = setup.method;
    loop {
        match solve_bound_orbital_method(&input, entry, setup, target_node_count, method) {
            Ok(solution) => return Ok(solution),
            Err(error) => {
                let recovery =
                    atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
                        requested_method: entry.requested_method,
                        method,
                    })?;
                if !recovery.needs_restart {
                    return Err(error);
                }
                method = recovery.method;
            }
        }
    }
}

fn solve_bound_orbital_method(
    input: &AtomicDiracBoundOrbitalInput<'_>,
    entry: AtomicDiracEntryState,
    setup: AtomicDiracSolverSetup,
    target_node_count: usize,
    method: i32,
) -> Result<AtomicDiracBoundOrbital, AtomMathError> {
    let reset = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method,
        primary_matching_precision: input.primary_matching_precision,
        secondary_matching_precision: input.secondary_matching_precision,
        energy_floor: setup.energy_floor,
        reference_energy: setup.energy,
    })?;
    let mut energy = reset.energy;
    let mut energy_inf = reset.energy_inf;
    let mut energy_sup = reset.energy_sup;
    let mut previous_energy = entry.previous_energy;
    let mut matching_index_1based = 1usize;
    let mut max_index_1based = input.initial_max_index_1based;
    let mut match_attempt_count = reset.match_attempt_count;

    'rematch: loop {
        let mut search_attempt_count = 0usize;
        loop {
            let shooting = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
                energy,
                previous_energy,
            })?;
            previous_energy = shooting.reference_energy;
            let mut integration_mode = shooting.integration_mode;
            let mut relocated = shooting.relocated;

            let pass = loop {
                let pass = run_bound_orbital_pass(
                    input,
                    entry,
                    setup,
                    method,
                    energy,
                    integration_mode,
                    matching_index_1based,
                    max_index_1based,
                    relocated,
                )?;
                matching_index_1based = pass.matching_index_1based;
                max_index_1based = pass.max_index_1based;
                if pass.needs_reintegration {
                    integration_mode = AtomicDiracIntegrationMode::FixedMatchingPoint;
                    relocated = true;
                    continue;
                }
                break pass;
            };

            let nodes = atomic_dirac_node_count(AtomicDiracNodeCountInput {
                large_component: pass.large_component.view(),
                matching_index_1based: pass.matching_index_1based,
                scan_index_1based: pass.scan_index_1based,
            })?;
            let node_search = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy,
                node_count: nodes.node_count,
                target_node_count,
                energy_sup,
                energy_inf,
                energy_floor: setup.energy_floor,
                energy_precision: input.primary_matching_precision,
                search_attempt_count,
                max_attempt_count: input.max_attempt_count,
            })?;
            energy = node_search.energy;
            energy_sup = node_search.energy_sup;
            energy_inf = node_search.energy_inf;
            search_attempt_count = node_search.search_attempt_count;
            if node_search.needs_reintegration {
                continue;
            }

            let norm = atomic_dirac_normalization(AtomicDiracNormalizationInput {
                radii: input.radii,
                large_component: pass.large_component.view(),
                small_component: pass.small_component.view(),
                large_coefficients: pass.large_coefficients.view(),
                small_coefficients: pass.small_coefficients.view(),
                method,
                step: input.step,
                coefficient_count: input.coefficient_count,
                matching_small_component: pass.matching_small_component,
                origin_power: input.origin_power,
                active_len: pass.max_index_1based,
                matching_index_1based: pass.matching_index_1based,
            })?;
            let energy_correction = if method == 1 {
                atomic_dirac_method_one_energy_correction(
                    AtomicDiracMethodOneEnergyCorrectionInput {
                        speed_of_light: input.speed_of_light,
                        norm: norm.norm,
                        large_component: pass.large_component.view(),
                        small_component: pass.small_component.view(),
                        matching_small_component: pass.matching_small_component,
                        matching_index_1based: pass.matching_index_1based,
                    },
                )?
            } else {
                pass.energy_correction.ok_or(
                    AtomMathError::MissingDiracIntegrationMatchingValue {
                        field: "soldir_method2_energy_correction",
                    },
                )?
            };
            let energy_step = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
                energy,
                correction: energy_correction.correction,
                mismatch: energy_correction.mismatch,
                energy_sup,
                energy_inf,
                mismatch_precision: reset.mismatch_precision,
                zero_energy_precision: input.primary_matching_precision,
            })?;
            energy = energy_step.energy;

            let mut attempts_exhausted = node_search.attempts_exhausted;
            if energy_step.needs_rematch {
                let rematch = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
                    mismatch: energy_correction.mismatch,
                    mismatch_precision: reset.mismatch_precision,
                    match_attempt_count,
                    max_attempt_count: input.max_attempt_count,
                })?;
                match_attempt_count = rematch.match_attempt_count;
                if rematch.needs_rematch {
                    continue 'rematch;
                }
                attempts_exhausted |= rematch.attempts_exhausted;
            }

            let normalized =
                atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
                    norm: norm.norm,
                    initial_large_coefficient: input.initial_large_coefficient,
                    initial_small_coefficient: setup.initial_small_coefficient,
                    large_component: pass.large_component.view(),
                    small_component: pass.small_component.view(),
                    large_coefficients: pass.large_coefficients.view(),
                    small_coefficients: pass.small_coefficients.view(),
                    coefficient_count: input.coefficient_count,
                    active_len: pass.max_index_1based,
                })?;

            return Ok(AtomicDiracBoundOrbital {
                energy,
                large_component: normalized.large_component,
                small_component: normalized.small_component,
                large_coefficients: normalized.large_coefficients,
                small_coefficients: normalized.small_coefficients,
                method,
                attempts_exhausted,
                active_len: pass.max_index_1based,
                matching_index_1based: pass.matching_index_1based,
                node_count: nodes.node_count,
                norm: norm.norm,
                search_attempt_count,
                match_attempt_count,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bound_orbital_pass(
    input: &AtomicDiracBoundOrbitalInput<'_>,
    entry: AtomicDiracEntryState,
    setup: AtomicDiracSolverSetup,
    method: i32,
    energy: Real,
    integration_mode: AtomicDiracIntegrationMode,
    matching_index_1based: usize,
    max_index_1based: usize,
    relocated: bool,
) -> Result<BoundOrbitalPass, AtomMathError> {
    let inhomogeneous_seed = atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
        large_source: input.large_source,
        small_source: input.small_source,
        large_source_coefficients: input.large_source_coefficients,
        small_source_coefficients: input.small_source_coefficients,
        coefficient_count: input.coefficient_count,
    })?;
    let inhomogeneous = integrate_bound_orbital_system(
        input,
        entry,
        setup,
        energy,
        inhomogeneous_seed,
        integration_mode,
        matching_index_1based,
        max_index_1based,
    )?;
    let matching_large_component = required_matching_value(
        inhomogeneous.matching_large_component,
        "soldir_inhomogeneous_matching_large_component",
    )?;
    let matching_small_component = required_matching_value(
        inhomogeneous.matching_small_component,
        "soldir_inhomogeneous_matching_small_component",
    )?;
    let matching_index_1based = inhomogeneous.matching_index_1based;
    let max_index_1based = inhomogeneous.max_index_1based;

    let mut large_component = inhomogeneous.large_component;
    let mut small_component = inhomogeneous.small_component;
    let mut large_coefficients = inhomogeneous.large_coefficients;
    let mut small_coefficients = inhomogeneous.small_coefficients;
    let mut energy_correction = None;

    match atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
        requested_method: entry.requested_method,
    })?
    .action
    {
        AtomicDiracInhomogeneousBranchAction::MatchHomogeneousTail => {
            let matched = atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
                large_component: large_component.view(),
                small_component: small_component.view(),
                matching_large_component,
                active_len: max_index_1based,
                matching_index_1based,
            })?;
            large_component = matched.large_component;
            small_component = matched.small_component;
        }
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem => {
            let homogeneous_seed =
                atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
                    radial_len: input.radii.len(),
                    coefficient_len: input.large_source_coefficients.len(),
                })?;
            let homogeneous_pass =
                atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput {
                    method,
                })?;
            let homogeneous = integrate_bound_orbital_system(
                input,
                entry,
                setup,
                energy,
                homogeneous_seed,
                homogeneous_pass.integration_mode,
                matching_index_1based,
                max_index_1based,
            )?;
            if method < 2 {
                let matched =
                    atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
                        large_component: large_component.view(),
                        small_component: small_component.view(),
                        homogeneous_large_component: homogeneous.large_component.view(),
                        homogeneous_small_component: homogeneous.small_component.view(),
                        matching_large_component,
                        active_len: max_index_1based,
                        matching_index_1based,
                    })?;
                large_component = matched.large_component;
                small_component = matched.small_component;
            } else {
                let homogeneous_matching_large_component = required_matching_value(
                    homogeneous.matching_large_component,
                    "soldir_homogeneous_matching_large_component",
                )?;
                let homogeneous_matching_small_component = required_matching_value(
                    homogeneous.matching_small_component,
                    "soldir_homogeneous_matching_small_component",
                )?;
                let matched =
                    atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
                        large_component: large_component.view(),
                        small_component: small_component.view(),
                        large_coefficients: large_coefficients.view(),
                        small_coefficients: small_coefficients.view(),
                        homogeneous_large_component: homogeneous.large_component.view(),
                        homogeneous_small_component: homogeneous.small_component.view(),
                        homogeneous_large_coefficients: homogeneous.large_coefficients.view(),
                        homogeneous_small_coefficients: homogeneous.small_coefficients.view(),
                        matching_large_component,
                        matching_small_component,
                        homogeneous_matching_large_component,
                        homogeneous_matching_small_component,
                        coefficient_count: input.coefficient_count,
                        active_len: max_index_1based,
                        matching_index_1based,
                    })?;
                large_component = matched.large_component;
                small_component = matched.small_component;
                large_coefficients = matched.large_coefficients;
                small_coefficients = matched.small_coefficients;

                let derivative_source = atomic_dirac_energy_disagreement_source(
                    AtomicDiracEnergyDisagreementSourceInput {
                        large_component: large_component.view(),
                        small_component: small_component.view(),
                        large_coefficients: large_coefficients.view(),
                        small_coefficients: small_coefficients.view(),
                        radii: input.radii,
                        speed_of_light: input.speed_of_light,
                        coefficient_count: input.coefficient_count,
                        active_len: max_index_1based,
                    },
                )?;
                let derivative = integrate_bound_orbital_system(
                    input,
                    entry,
                    setup,
                    energy,
                    AtomicDiracIntegrationSeed {
                        large_source: derivative_source.large_source,
                        small_source: derivative_source.small_source,
                        large_coefficients: derivative_source.large_coefficients,
                        small_coefficients: derivative_source.small_coefficients,
                    },
                    AtomicDiracIntegrationMode::FixedMatchingPoint,
                    matching_index_1based,
                    max_index_1based,
                )?;
                let derivative_matching_large_component = required_matching_value(
                    derivative.matching_large_component,
                    "soldir_derivative_matching_large_component",
                )?;
                let derivative_matching_small_component = required_matching_value(
                    derivative.matching_small_component,
                    "soldir_derivative_matching_small_component",
                )?;
                let derivative_match = atomic_dirac_energy_disagreement_match(
                    AtomicDiracEnergyDisagreementMatchInput {
                        large_derivative: derivative.large_component.view(),
                        small_derivative: derivative.small_component.view(),
                        large_derivative_coefficients: derivative.large_coefficients.view(),
                        small_derivative_coefficients: derivative.small_coefficients.view(),
                        homogeneous_large_component: homogeneous.large_component.view(),
                        homogeneous_small_component: homogeneous.small_component.view(),
                        homogeneous_large_coefficients: homogeneous.large_coefficients.view(),
                        homogeneous_small_coefficients: homogeneous.small_coefficients.view(),
                        matching_large_derivative: derivative_matching_large_component,
                        matching_small_derivative: derivative_matching_small_component,
                        homogeneous_matching_large_component,
                        homogeneous_matching_small_component,
                        coefficient_count: input.coefficient_count,
                        active_len: max_index_1based,
                        matching_index_1based,
                    },
                )?;
                let method2_norm = atomic_dirac_normalization(AtomicDiracNormalizationInput {
                    radii: input.radii,
                    large_component: large_component.view(),
                    small_component: small_component.view(),
                    large_coefficients: large_coefficients.view(),
                    small_coefficients: small_coefficients.view(),
                    method,
                    step: input.step,
                    coefficient_count: input.coefficient_count,
                    matching_small_component,
                    origin_power: input.origin_power,
                    active_len: max_index_1based,
                    matching_index_1based,
                })?;
                let correction = atomic_dirac_energy_disagreement_correction(
                    AtomicDiracEnergyDisagreementCorrectionInput {
                        radii: input.radii,
                        large_component: large_component.view(),
                        small_component: small_component.view(),
                        large_derivative: derivative_match.large_derivative.view(),
                        small_derivative: derivative_match.small_derivative.view(),
                        large_coefficients: large_coefficients.view(),
                        small_coefficients: small_coefficients.view(),
                        large_derivative_coefficients: derivative_match
                            .large_derivative_coefficients
                            .view(),
                        small_derivative_coefficients: derivative_match
                            .small_derivative_coefficients
                            .view(),
                        norm: method2_norm.norm,
                        step: input.step,
                        origin_power: input.origin_power,
                        coefficient_count: input.coefficient_count,
                        active_len: max_index_1based,
                    },
                )?;
                energy_correction = Some(AtomicDiracEnergyCorrection {
                    correction: correction.correction,
                    mismatch: correction.normalization_mismatch,
                });
                large_component = correction.large_component;
                small_component = correction.small_component;
                large_coefficients = correction.large_coefficients;
                small_coefficients = correction.small_coefficients;
            }
        }
    }

    let matching_update =
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: large_component.view(),
            active_len: max_index_1based,
            matching_index_1based,
            already_relocated: relocated,
        })?;
    Ok(BoundOrbitalPass {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        matching_small_component,
        energy_correction,
        matching_index_1based: matching_update.matching_index_1based,
        max_index_1based,
        scan_index_1based: matching_update.scan_index_1based,
        needs_reintegration: matching_update.needs_reintegration,
    })
}

#[allow(clippy::too_many_arguments)]
fn integrate_bound_orbital_system(
    input: &AtomicDiracBoundOrbitalInput<'_>,
    entry: AtomicDiracEntryState,
    setup: AtomicDiracSolverSetup,
    energy: Real,
    seed: AtomicDiracIntegrationSeed,
    mode: AtomicDiracIntegrationMode,
    matching_index_1based: usize,
    max_index_1based: usize,
) -> Result<AtomicDiracIntegration, AtomMathError> {
    atomic_dirac_integration(AtomicDiracIntegrationInput {
        large_source: seed.large_source.view(),
        small_source: seed.small_source.view(),
        large_coefficients: seed.large_coefficients.view(),
        small_coefficients: seed.small_coefficients.view(),
        radii: input.radii,
        potential: input.potential,
        potential_coefficients: input.potential_coefficients,
        energy,
        origin_power: input.origin_power,
        initial_large_coefficient: input.initial_large_coefficient,
        initial_small_coefficient: setup.initial_small_coefficient,
        asymptotic_large_component: entry.asymptotic_large_component,
        kappa: input.kappa,
        speed_of_light: input.speed_of_light,
        step: input.step,
        matching_precision: input.primary_matching_precision,
        coefficient_count: input.coefficient_count,
        active_len: input.active_len,
        mode,
        matching_index_1based,
        max_index_1based,
    })
}

fn required_matching_value(
    value: Option<Real>,
    field: &'static str,
) -> Result<Real, AtomMathError> {
    value.ok_or(AtomMathError::MissingDiracIntegrationMatchingValue { field })
}
