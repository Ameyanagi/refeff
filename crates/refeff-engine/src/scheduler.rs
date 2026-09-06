//! FEFF stage scheduling, preserving the explicit scientific branch order.
use super::*;

#[cfg(all(test, feature = "full"))]
pub(super) fn run_supported_cached_modules(work_dir: &Path) -> Result<Vec<SupportedModuleReport>> {
    let mut reports = Vec::new();
    let mut pot_context = pot::PotRunContext::default();
    run_supported_cached_modules_into(work_dir, &mut reports, &mut pot_context)?;
    Ok(reports)
}

#[cfg(feature = "full")]
pub(super) fn run_supported_cached_modules_into(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
    pot_context: &mut pot::PotRunContext,
) -> Result<()> {
    execution::start("atomic")?;
    let stage_start = Instant::now();
    let atomic_cached = atomic::has_cached_atomic_output(work_dir)?;
    let prepared_no_scf_available =
        !atomic_cached && matches!(pot_context.prepared_no_scf(work_dir), Ok(Some(_)));
    let atomic_source_handoff = !atomic_cached
        && (prepared_no_scf_available || atomic::has_supported_atomic_source_handoff(work_dir)?);
    if atomic_cached || atomic_source_handoff {
        let prepared_no_scf = if prepared_no_scf_available {
            pot_context.prepared_no_scf(work_dir)?
        } else {
            None
        };
        let count = atomic::run_in_dir_with_prepared_no_scf(work_dir, prepared_no_scf)
            .context("failed to run supported atomic stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "atomic",
                count,
                unit: "file(s)",
                status: StageStatus::from_cached(atomic_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if atomic::has_supported_config_handoff(work_dir)? {
        let count = atomic::run_supported_config_handoff_in_dir(work_dir)
            .context("failed to run supported atomic config handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "atomic-config",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if rhorrp::has_supported_rhorrp_output(work_dir)? {
        execution::start("rhorrp")?;
        let stage_start = Instant::now();
        let count = rhorrp::run_in_dir(work_dir).context("failed to run supported rhorrp stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "rhorrp",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    execution::start("pot")?;
    let pot_start = Instant::now();
    let pot_cached = pot::has_cached_pot_output_with_context(work_dir, pot_context)?;
    if pot_cached
        || pot::has_supported_pot_source_handoff_with_context(work_dir, pot_context)?
        || pot::has_supported_pot_generation_handoff_with_context(work_dir, pot_context)?
    {
        let count = pot::run_in_dir_with_context(work_dir, pot_context)
            .context("failed to run supported pot stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "pot",
                count,
                unit: "file(s)",
                status: StageStatus::from_cached(pot_cached),
                duration_ms: elapsed_ms(pot_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if let Some(count) = pot::run_supported_pot_scf_source_handoff_once_in_dir(work_dir)? {
        if count > 0 {
            let completed = pot::has_cached_pot_output(work_dir)?;
            reports.push(SupportedModuleReport {
                name: if completed { "pot" } else { "pot-scf-source" },
                count,
                unit: if completed {
                    "file(s)"
                } else {
                    "source bundle(s)"
                },
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(pot_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if pot::has_supported_pot_input_handoff(work_dir)? {
        execution::start("pot-input")?;
        let stage_start = Instant::now();
        let count = pot::run_supported_pot_input_handoff_in_dir(work_dir)
            .context("failed to run supported pot input handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "pot-input",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    // Active Hubbard starts with an ordinary spectrum phase because LDOS
    // has not created v_hubbard.bin yet.  Record that bootstrap boundary
    // before the first XSPH/FMS pass.  After LDOS's two internal passes the
    // normal spectrum must be refreshed from the newly active Hubbard
    // source, matching FEFF's POT -> LDOS -> XSPH -> FMS final ordering.
    let active_hubbard_spectrum_bootstrap_pending =
        ldos::active_hubbard_spectrum_bootstrap_pending(work_dir)?;

    // MPSE XSPH consumes the loss function as its excitation-pole source.
    // OPCONS may need the freshly generated POT Norman radii to determine
    // default number densities, so run it after POT but before SCREEN/XSPH.
    // Deferring OPCONS until the later optical-output block leaves XSPH with
    // only an energy-mesh handoff and then prevents FMS from obtaining
    // phase.bin during a fresh full run.
    if opcons::has_complete_table_inputs(work_dir)? {
        execution::start("opcons")?;
        let stage_start = Instant::now();
        let count = opcons::run_in_dir(work_dir).context("failed to run supported opcons stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "opcons",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if screen::has_completed_screen_output(work_dir)? {
        execution::start("screen")?;
        let stage_start = Instant::now();
        let count = screen::run_in_dir(work_dir).context("failed to run supported screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if screen::has_recoverable_cached_screen_stage(work_dir)? {
        execution::start("screen")?;
        let stage_start = Instant::now();
        let count = screen::run_recoverable_cached_screen_stage_in_dir(work_dir)
            .context("failed to run recoverable cached screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if screen::has_supported_wscrn_handoff(work_dir)? {
        execution::start("screen-wscrn")?;
        let stage_start = Instant::now();
        let count = screen::run_supported_wscrn_handoff_in_dir(work_dir)
            .context("failed to run supported screen wscrn handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen-wscrn",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if screen::has_supported_screen_source_handoff(work_dir)? {
        execution::start("screen")?;
        let stage_start = Instant::now();
        let count = screen::run_in_dir(work_dir).context("failed to run supported screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if xsph::has_supported_xsph_output(work_dir)?
        || xsph::has_supported_tdlda_xsedge_output(work_dir)?
    {
        execution::start("xsph")?;
        let stage_start = Instant::now();
        let count = xsph::run_in_dir(work_dir).context("failed to run supported xsph stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "xsph",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else {
        if xsph::has_supported_phase_handoff(work_dir)? {
            execution::start("xsph-phase")?;
            let stage_start = Instant::now();
            let count = xsph::run_supported_phase_handoff_in_dir(work_dir)
                .context("failed to run supported xsph phase handoff")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "xsph-phase",
                    count,
                    unit: "file(s)",
                    status: StageStatus::Generated,
                    duration_ms: elapsed_ms(stage_start),
                });
                let index = reports.len();
                if let Some(report) = reports.last_mut() {
                    print_stage_line(index, report);
                }
            }
        }
        if xsph::has_supported_phase_text_handoff(work_dir)? {
            execution::start("xsph-phase-text")?;
            let stage_start = Instant::now();
            let count = xsph::run_supported_phase_text_handoff_in_dir(work_dir)
                .context("failed to run supported xsph phase text handoff")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "xsph-phase-text",
                    count,
                    unit: "file(s)",
                    status: StageStatus::Generated,
                    duration_ms: elapsed_ms(stage_start),
                });
                let index = reports.len();
                if let Some(report) = reports.last_mut() {
                    print_stage_line(index, report);
                }
            }
        }
        if xsph::has_supported_phase_mesh_handoff(work_dir)? {
            execution::start("xsph-emesh")?;
            let stage_start = Instant::now();
            let count = xsph::run_supported_phase_mesh_handoff_in_dir(work_dir)
                .context("failed to run supported xsph emesh handoff")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "xsph-emesh",
                    count,
                    unit: "file(s)",
                    status: StageStatus::Generated,
                    duration_ms: elapsed_ms(stage_start),
                });
                let index = reports.len();
                if let Some(report) = reports.last_mut() {
                    print_stage_line(index, report);
                }
            }
        }
    }

    ldos::with_preserved_active_hubbard_ldos_magnetic_sources(work_dir, || {
        if fms::has_runnable_fms_solver(work_dir)? {
            let status = if fms::has_cached_fms_solver_output(work_dir)? {
                StageStatus::Cached
            } else {
                StageStatus::Generated
            };
            execution::start("fms")?;
            let stage_start = Instant::now();
            let count =
                fms::run_fms_in_dir(work_dir).context("failed to run supported fms stage")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "fms",
                    count,
                    unit: "file(s)",
                    status,
                    duration_ms: elapsed_ms(stage_start),
                });
                let index = reports.len();
                if let Some(report) = reports.last_mut() {
                    print_stage_line(index, report);
                }
            }

            let status = if fms::has_cached_mkgtr_output(work_dir)? {
                StageStatus::Cached
            } else {
                StageStatus::Generated
            };
            execution::start("mkgtr")?;
            let stage_start = Instant::now();
            let count =
                fms::run_mkgtr_in_dir(work_dir).context("failed to run supported mkgtr stage")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "mkgtr",
                    count,
                    unit: "file(s)",
                    status,
                    duration_ms: elapsed_ms(stage_start),
                });
                let index = reports.len();
                if let Some(report) = reports.last_mut() {
                    print_stage_line(index, report);
                }
            }
        }
        Ok(())
    })?;

    if band::has_cached_band_output(work_dir)? {
        execution::start("band")?;
        let stage_start = Instant::now();
        let count = band::run_in_dir(work_dir).context("failed to run supported band stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "band",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if band::has_supported_pre_solver_handoff(work_dir)? {
        execution::start("rixs")?;
        let stage_start = Instant::now();
        let count = band::run_supported_pre_solver_handoff_in_dir(work_dir)
            .context("failed to run supported band pre-solver handoff")?;
        if count > 0 {
            let completed = band::has_cached_band_output(work_dir)?;
            reports.push(SupportedModuleReport {
                name: if completed { "band" } else { "band-handoff" },
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if rixs::has_cached_rixs_output(work_dir)? {
        execution::start("rixs")?;
        let stage_start = Instant::now();
        let count = rixs::run_in_dir(work_dir).context("failed to run supported rixs stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "rixs",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if rixs::has_supported_solver_handoff(work_dir)? {
        execution::start("compton")?;
        let stage_start = Instant::now();
        let count = rixs::run_supported_solver_handoff_in_dir(work_dir)
            .context("failed to run supported rixs solver handoff")?;
        if count > 0 {
            let completed = rixs::has_cached_rixs_output(work_dir)?;
            reports.push(SupportedModuleReport {
                name: if completed { "rixs" } else { "rixs-handoff" },
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if compton::has_supported_outputs(work_dir)? {
        execution::start("compton")?;
        let stage_start = Instant::now();
        let count =
            compton::run_in_dir(work_dir).context("failed to run supported compton stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "compton",
                count,
                unit: "row(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if fullspectrum::has_cached_optical_inputs(work_dir)? {
        execution::start("fullspectrum")?;
        let stage_start = Instant::now();
        let count = fullspectrum::run_in_dir(work_dir)
            .context("failed to run supported fullspectrum stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "fullspectrum",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    let crpa_cached = crpa::has_cached_crpa_output(work_dir)?;
    if crpa_cached || crpa::has_supported_crpa_source_handoff(work_dir)? {
        execution::start("crpa")?;
        let stage_start = Instant::now();
        let count = crpa::run_in_dir(work_dir).context("failed to run supported crpa stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "crpa",
                count,
                unit: "row(s)",
                status: StageStatus::from_cached(crpa_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if crpa::has_supported_wscrn_handoff(work_dir)? {
        execution::start("crpa-wscrn")?;
        let stage_start = Instant::now();
        let count = crpa::run_supported_wscrn_handoff_in_dir(work_dir)
            .context("failed to run supported crpa wscrn handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "crpa-wscrn",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if ldos::has_cached_ldos_output(work_dir)? {
        execution::start("ldos")?;
        let stage_start = Instant::now();
        let count = ldos::run_in_dir(work_dir).context("failed to run supported ldos stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if ldos::has_recoverable_ldos_output(work_dir)? {
        execution::start("ldos")?;
        let stage_start = Instant::now();
        let count =
            ldos::run_in_dir(work_dir).context("failed to run supported recoverable ldos stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if ldos::has_supported_source_output_handoff(work_dir)? {
        execution::start("ldos")?;
        let stage_start = Instant::now();
        let count = ldos::run_in_dir(work_dir)
            .context("failed to run supported ldos source-output handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    } else if ldos::has_supported_kmesh_handoff(work_dir)? {
        execution::start("ldos-kmesh")?;
        let stage_start = Instant::now();
        let count = ldos::run_supported_kmesh_handoff_in_dir(work_dir)
            .context("failed to run supported ldos kmesh handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos-kmesh",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if active_hubbard_spectrum_bootstrap_pending && work_dir.join("v_hubbard.bin").is_file() {
        execution::start("xsph")?;
        let stage_start = Instant::now();
        let count = xsph::run_in_dir(work_dir)
            .context("failed to refresh active Hubbard xsph spectrum after ldos")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "xsph",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }

        ldos::with_preserved_active_hubbard_ldos_magnetic_sources(work_dir, || {
            if fms::has_runnable_fms_solver(work_dir)? {
                execution::start("fms")?;
                let stage_start = Instant::now();
                let count = fms::run_fms_in_dir(work_dir)
                    .context("failed to refresh active Hubbard fms spectrum after ldos")?;
                if count > 0 {
                    reports.push(SupportedModuleReport {
                        name: "fms",
                        count,
                        unit: "file(s)",
                        status: StageStatus::Generated,
                        duration_ms: elapsed_ms(stage_start),
                    });
                    let index = reports.len();
                    if let Some(report) = reports.last_mut() {
                        print_stage_line(index, report);
                    }
                }

                execution::start("mkgtr")?;
                let stage_start = Instant::now();
                let count = fms::run_mkgtr_in_dir(work_dir)
                    .context("failed to refresh active Hubbard mkgtr spectrum after ldos")?;
                if count > 0 {
                    reports.push(SupportedModuleReport {
                        name: "mkgtr",
                        count,
                        unit: "file(s)",
                        status: StageStatus::Generated,
                        duration_ms: elapsed_ms(stage_start),
                    });
                    let index = reports.len();
                    if let Some(report) = reports.last_mut() {
                        print_stage_line(index, report);
                    }
                }
            }
            Ok(())
        })?;
    }

    if band::kmesh::has_supported_kmesh_handoff(work_dir)? {
        execution::start("kmesh")?;
        let stage_start = Instant::now();
        let count = band::kmesh::run_supported_kmesh_handoff_in_dir(work_dir)
            .context("failed to run supported kmesh handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "kmesh",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    let eels_cached = eels::has_completed_eels_output(work_dir)?;
    if eels::has_cached_eels_output(work_dir)? {
        execution::start("eels")?;
        let stage_start = Instant::now();
        let count = eels::run_in_dir(work_dir).context("failed to run supported eels stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "eels",
                count,
                unit: "row(s)",
                status: StageStatus::from_cached(eels_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    let eelsmdff_cached = eelsmdff::has_completed_mdff_output(work_dir)?;
    if eelsmdff::has_cached_mdff_output(work_dir)? {
        execution::start("eelsmdff")?;
        let stage_start = Instant::now();
        let count =
            eelsmdff::run_in_dir(work_dir).context("failed to run supported eelsmdff stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "eelsmdff",
                count,
                unit: "row(s)",
                status: StageStatus::from_cached(eelsmdff_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    let dmdw_cached = dmdw::has_cached_dmdw_output(work_dir)?;
    if dmdw_cached || dmdw::has_supported_dmdw_source_handoff(work_dir)? {
        execution::start("dmdw")?;
        let stage_start = Instant::now();
        let count = dmdw::run_in_dir(work_dir).context("failed to run supported dmdw stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "dmdw",
                count,
                unit: "section(s)",
                status: StageStatus::from_cached(dmdw_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if let Some(plan) = paths::prepare_stage(work_dir)? {
        execution::start("path")?;
        let stage_start = Instant::now();
        let count =
            paths::run_prepared(work_dir, plan).context("failed to run supported path stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "path",
                count,
                unit: "path(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    let genfmt_start = Instant::now();
    if let Some(plan) = genfmt::prepare_stage(work_dir)? {
        execution::start("genfmt")?;
        let stage_start = genfmt_start;
        let count =
            genfmt::run_prepared(work_dir, plan).context("failed to run supported genfmt stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "genfmt",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if ff2x::has_cached_ff2x_output(work_dir)? {
        execution::start("ff2x")?;
        let stage_start = Instant::now();
        let count = ff2x::run_in_dir(work_dir).context("failed to run supported ff2x stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ff2x",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    let self_cached = sfconv::has_cached_self_output(work_dir)?;
    if self_cached || sfconv::has_supported_self_source_handoff(work_dir)? {
        execution::start("self")?;
        let stage_start = Instant::now();
        let count =
            sfconv::run_self_in_dir(work_dir).context("failed to run supported self stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "self",
                count,
                unit: "pole(s)",
                status: StageStatus::from_cached(self_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    if sfconv::has_supported_sfconv_source_handoff(work_dir)? {
        execution::start("sfconv")?;
        let stage_start = Instant::now();
        let count = sfconv::run_in_dir(work_dir).context("failed to run supported sfconv stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "sfconv",
                count,
                unit: "target(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            let index = reports.len();
            if let Some(report) = reports.last_mut() {
                print_stage_line(index, report);
            }
        }
    }

    Ok(())
}

pub(super) fn supported_module_summary(reports: &[SupportedModuleReport]) -> String {
    if reports.is_empty() {
        return "no supported cached stages were run".to_string();
    }

    let details = reports
        .iter()
        .map(|report| {
            // Preserve the historical human summary; structured reports use
            // the registry's canonical module identities.
            let name = match report.name {
                "opconsat" => "opcons",
                "eelsmdff" => "mdff",
                name => name,
            };
            format!("{name}={} {}", report.count, report.unit)
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("supported cached stages run: {details}")
}

#[cfg(feature = "full")]
pub(super) fn run_remaining_required_modules(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
    pot_context: &mut pot::PotRunContext,
) -> Result<()> {
    if !atomic::has_cached_atomic_output(work_dir)? {
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "pot" && report.count > 0)
        && !pot::has_cached_pot_output_with_context(work_dir, pot_context)?
    {
        run_required_module(reports, "pot", "file(s)", || {
            pot::run_in_dir_with_context(work_dir, pot_context)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "xsph" && report.count > 0)
        && !xsph::has_supported_xsph_output(work_dir)?
        && !xsph::has_supported_tdlda_xsedge_output(work_dir)?
    {
        run_required_module(reports, "xsph", "file(s)", || {
            xsph::run_required_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "fms" && report.count > 0)
        && !fms::has_cached_fms_solver_output(work_dir)?
    {
        run_required_module(reports, "fms", "file(s)", || fms::run_fms_in_dir(work_dir))?;
    }
    if fms::has_cached_fms_solver_output(work_dir)?
        && !reports
            .iter()
            .any(|report| report.name == "mkgtr" && report.count > 0)
    {
        run_required_module(reports, "mkgtr", "file(s)", || {
            fms::run_mkgtr_in_dir(work_dir)
        })?;
    }
    if !band::has_cached_band_output(work_dir)? {
        run_required_module(reports, "band", "file(s)", || band::run_in_dir(work_dir))?;
    }
    if !rixs::has_cached_rixs_output(work_dir)? {
        run_required_module(reports, "rixs", "file(s)", || rixs::run_in_dir(work_dir))?;
    }
    if !rhorrp::has_supported_rhorrp_output(work_dir)? {
        run_required_module(reports, "rhorrp", "file(s)", || {
            rhorrp::run_in_dir(work_dir)
        })?;
    }
    if !opcons::has_complete_table_inputs(work_dir)? {
        run_required_module(reports, "opcons", "row(s)", || opcons::run_in_dir(work_dir))?;
    }
    if !compton::has_supported_outputs(work_dir)? {
        run_required_module(reports, "compton", "row(s)", || {
            compton::run_in_dir(work_dir)
        })?;
    }
    if !fullspectrum::has_cached_optical_inputs(work_dir)? {
        run_required_module(reports, "fullspectrum", "row(s)", || {
            fullspectrum::run_in_dir(work_dir)
        })?;
    }
    if !(crpa::has_cached_crpa_output(work_dir)?
        || crpa::has_supported_crpa_source_handoff(work_dir)?)
    {
        run_required_module(reports, "crpa", "row(s)", || crpa::run_in_dir(work_dir))?;
    }
    if !screen::has_completed_screen_output(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || screen::run_in_dir(work_dir))?;
    }
    if !ldos::has_cached_ldos_output(work_dir)? {
        run_required_module(reports, "ldos", "file(s)", || ldos::run_in_dir(work_dir))?;
    }
    if !dmdw::has_cached_dmdw_output(work_dir)? {
        run_required_module(reports, "dmdw", "section(s)", || dmdw::run_in_dir(work_dir))?;
    }
    if !paths::has_cached_paths_output(work_dir)? {
        run_required_module(reports, "path", "path(s)", || paths::run_in_dir(work_dir))?;
    }
    if !genfmt::has_cached_genfmt_output(work_dir)? {
        run_required_module(reports, "genfmt", "file(s)", || {
            genfmt::run_in_dir(work_dir)
        })?;
    }
    if !ff2x::has_cached_ff2x_output(work_dir)? {
        run_required_module(reports, "ff2x", "file(s)", || ff2x::run_in_dir(work_dir))?;
    }
    // EELS consumes the polarization-specific xmu/opcons spectra assembled by
    // the path -> GENFMT -> FF2X producer chain. Keep it after those required
    // stages so a cache-free ELNES/EXELFS run can complete in one scheduler
    // pass instead of stopping before its source spectra exist.
    if !eels::has_completed_eels_output(work_dir)? {
        run_required_module(reports, "eels", "row(s)", || eels::run_in_dir(work_dir))?;
    }
    if !eelsmdff::has_completed_mdff_output(work_dir)? {
        run_required_module(reports, "eelsmdff", "row(s)", || {
            eelsmdff::run_in_dir(work_dir)
        })?;
    }
    if !sfconv::has_cached_self_output(work_dir)? {
        run_required_module(reports, "self", "pole(s)", || {
            sfconv::run_self_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "sfconv" && report.count > 0)
    {
        run_required_module(reports, "sfconv", "target(s)", || {
            sfconv::run_in_dir(work_dir)
        })?;
    }
    Ok(())
}

#[cfg(all(feature = "exafs", not(feature = "full")))]
pub(super) fn run_exafs_pipeline(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
) -> Result<()> {
    let mut pot_context = pot::PotRunContext::default();

    let atomic_cached = atomic::has_cached_atomic_output(work_dir)?;
    let prepared_no_scf_available =
        !atomic_cached && matches!(pot_context.prepared_no_scf(work_dir), Ok(Some(_)));
    let atomic_source_handoff =
        prepared_no_scf_available || atomic::has_supported_atomic_source_handoff(work_dir)?;
    if atomic_cached || atomic_source_handoff {
        let prepared_no_scf = if prepared_no_scf_available {
            pot_context.prepared_no_scf(work_dir)?
        } else {
            None
        };
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir_with_prepared_no_scf(work_dir, prepared_no_scf)
        })?;
    } else {
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir(work_dir)
        })?;
    }

    run_required_module(reports, "pot", "file(s)", || {
        pot::run_in_dir_with_context(work_dir, &mut pot_context)
    })?;

    if screen::has_completed_screen_output(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || screen::run_in_dir(work_dir))?;
    } else if screen::has_recoverable_cached_screen_stage(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || {
            screen::run_recoverable_cached_screen_stage_in_dir(work_dir)
        })?;
    } else if screen::has_supported_wscrn_handoff(work_dir)? {
        run_required_module(reports, "screen-wscrn", "row(s)", || {
            screen::run_supported_wscrn_handoff_in_dir(work_dir)
        })?;
    } else if screen::has_supported_screen_source_handoff(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || screen::run_in_dir(work_dir))?;
    }

    // `has_supported_xsph_output` means the stage can run from either caches
    // or source handoffs, not that its outputs already exist. Always invoke
    // the stage so a fresh EXAFS workspace materializes phase.bin before
    // PATH.
    let mut xsph_context = xsph::XsphRunContext::default();
    run_required_module(reports, "xsph", "file(s)", || {
        let xsph_satisfiable =
            xsph::has_supported_xsph_output_with_context(work_dir, &mut xsph_context)?;
        if xsph_satisfiable {
            xsph::run_in_dir_with_context(work_dir, &mut xsph_context)
        } else if xsph::has_supported_tdlda_xsedge_output(work_dir)? {
            xsph::run_in_dir(work_dir)
        } else {
            xsph::run_required_in_dir(work_dir)
        }
    })?;
    // These compatibility predicates report whether a stage is satisfiable
    // from either an existing cache or source handoffs. Invoke each stage so
    // fresh handoffs are actually serialized for its downstream consumer.
    run_required_module(reports, "path", "path(s)", || paths::run_in_dir(work_dir))?;
    run_required_module(reports, "genfmt", "file(s)", || {
        genfmt::run_in_dir(work_dir)
    })?;
    run_required_module(reports, "ff2x", "file(s)", || ff2x::run_in_dir(work_dir))?;

    #[cfg(feature = "sfconv")]
    if sfconv::has_supported_sfconv_source_handoff(work_dir)? {
        run_required_module(reports, "sfconv", "target(s)", || {
            sfconv::run_in_dir(work_dir)
        })?;
    }

    Ok(())
}

/// Runs a single module for which no cached/handoff state satisfied it (the
/// `run_remaining_required_modules` fallback), pushing a `Generated`
/// [`SupportedModuleReport`] into the shared `reports` vec — the same vec
/// [`run_supported_cached_modules_into`] populates — so the final summary
/// and `--json` [`RunReport`] cover every stage the run touched, not just
/// the ones satisfied from cache.
pub(super) fn run_required_module(
    reports: &mut Vec<SupportedModuleReport>,
    name: &'static str,
    unit: &'static str,
    run: impl FnOnce() -> Result<usize>,
) -> Result<()> {
    execution::start(name)?;
    let stage_start = Instant::now();
    let count = run().with_context(|| format!("failed to run FEFF {name} stage"))?;
    if count > 0 {
        reports.push(SupportedModuleReport {
            name,
            count,
            unit,
            status: StageStatus::Generated,
            duration_ms: elapsed_ms(stage_start),
        });
        let index = reports.len();
        if let Some(report) = reports.last_mut() {
            print_stage_line(index, report);
        }
    }
    Ok(())
}
