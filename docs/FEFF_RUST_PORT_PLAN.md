# FEFF Rust Port Plan

This is the active implementation plan for finishing the FEFF10 Rust port in
`refeff`. It is deliberately tied to:

- `cargo run -p xtask -- port-status --detail`
- `cargo run --profile release -p xtask -- port-status --detail --json-out target/port-status.json`
- `cargo run -p xtask -- compatibility-matrix --detail`
- `cargo run -p xtask -- compatibility-matrix --open-only --detail`
- `cargo run --profile release -p xtask -- compatibility-matrix --open-only --detail --json-out target/compatibility-matrix.json`
- source-backed numerical drivers, not cached-output pass-throughs
- focused verification first, with full suites reserved for release gates

## Goal

Complete the Rust port by replacing every remaining production `unported` CLI
gate with source-backed Rust numerical execution, wiring those drivers into
`refeff run`, and proving parity with FEFF reference data.

Done means:

- `cargo run -p xtask -- port-status --fail-on-unported --fail-on-guarded-branches --fail-on-ignored-parity`
  passes.
- `cargo run -p xtask -- compatibility-matrix --fail-on-open` passes.
- `refeff module <name>` can run each enabled FEFF module from prepared source
  handoffs without relying on final-output caches.
- `refeff run --input feff.inp --output <dir>` executes the same
  source-backed path through full-run orchestration and writes the requested
  FEFF outputs into `<dir>`.
- Each removed gate has focused module tests, source-handoff tests, and at
  least one generated-reference or reference-zip parity test where the repo has
  suitable fixture data.

## Current Port Status

There are no current production `unported` CLI gates according to the module
inventory:

- `cargo run -p xtask -- port-status --detail`

Branch-level guarded production paths are tracked separately by the same
command and are currently clean. XSPH NRIXS/JAS `xsectjas` production now has
readable `xsecl.dat`/`xsecl2.dat`/`xsecl.bin` cache validation plus a
q-resolved, source-backed one-spin normal-potential writer that produces
`xsect.dat`, the `xsecl*` sidecars, and matching `phase.bin` transition
moments. Broader spin normalization and additional branch parity remain
follow-up coverage work, not a current guarded release blocker.

The explicit POT gate is retired: complete no-SCF or supported SCF source
handoffs now produce `pot.bin`, `apot.bin`, `potNN.dat`, and `log1.dat`, while
missing or incomplete POT source state reports a normal source/cache
requirement instead of an unported numerical solver.

Post-completion parity broadening remains for supported modules:

- `pot`: broaden SCF convergence/source coverage beyond the current reference
  gates; bounded NiO Hubbard and BN positive-`totvol` SCF FEFF parity is
  covered at module and scheduler boundaries, SF6/YBCO/XMCD no-SCF FEFF
  reference parity is covered at module and scheduler boundaries, and
  release-profile gates now cover high-`EXCHANGE`, restart/external source runs,
  and FEFF-style `nstarts` retry controls, including the hard max-start
  boundary. Terminal convergence/iteration-limit final `pot.bin` candidate
  gating, finite-nucleus repeat-exhaustion boundaries, the core SCF
  contour/outer-iteration formulas, and CLI SCF source-loop/reference-output
  gates are also release covered. Broader convergence/exhaustion parity remains
  open.
- `atomic`: broaden finite-nucleus/generated-reference parity beyond the
  release-profile no-SCF source-output, direct APOT source-generation,
  iterative repeat-boundary, finite-vs-point generated SCF state-selection
  handoffs, core `nucdev` nuclear-potential parity, and core finite-nucleus SCF
  state construction now covered.
- `xsph`: multi-fixture source-backed `phase.bin`/`xsect.dat` parity and
  direct occupied/file-basis/generated-basis TDLDA/PMBSE `xsedge.dat`
  generation are covered release gates; core FEFF phase primitives and core
  TDLDA/`xsectd` formulas are release-gated in `refeff-core`. Positive-`izstd`
  PMBSE-reset behavior is covered at module, scheduler, and full-run
  boundaries. LDOS FMS/spin-FMS phase/xsect scheduler parity and NRIXS/JAS
  sidecar generation are also release covered. A broad CLI source-generation
  sweep now release-gates empty-cell, Hubbard, `izstd`, FPRIME, E2/L2LP, MPSE,
  AXAFS, NRIXS, phase-text, and two-spin branch outputs, and the broader source
  parity sweep now includes NRIXS/GeCl4 sidecars plus MnF2 XMCD `ltot`
  capacity/`xsect.dat` parity and Gd L1 XMCD fine-radial-grid
  capacity/`xsect.dat` parity coverage. Remaining NRIXS/MgB2, XMCD
  phase-shift numeric parity, phase-shift reference broadening, and broader
  TDLDA/PMBSE reference parity remain open.
- `band`: broaden generated-output parity for production `bandstructure.dat`;
  one-spin relativistic KKR/`freeprop` and multi-spin relativistic source
  dispatch now run through the CLI/full-run scheduler, with Cr2GeC scheduler
  reference parity, standalone `freeprop`, degenerate two-spin, direct CLI
  branch-generation, and release-gated stale-cache repair covered. The BAND
  KSPACE structure-factor bridge and KKR/`freeprop` final-row paths are
  release-gated in `refeff-core`, and the diagonalization bridge uses a
  release-gated pure-Rust `faer` general-complex eigenvalue adapter.
- `ldos`: broaden source-generated final tables into spin-Hubbard and
  full-potential branches; production, nonzero, and ordinary-spin FMS
  final-table parity plus
  active-Hubbard NiO magnetic sidecar cache contracts are covered release
  gates, and release-profile scheduler gates cover no-FMS source/stale repair
  plus active-Hubbard cache/source contract validation. Magnetic Hubbard
  `ff2rho_h_step2` table assembly, non-full-potential `ff2rho` final-table
  formulas, non-full-potential `fmsdos` trace projection, and active-Hubbard
  full-potential FMS `save_gg_slice` sidecars are covered. The no-FMS
  active-Hubbard path can now
  repair one-sided `lmdosNN.dat`/`rhocmNN.dat` magnetic sidecars through the
  zero-scattering `ff2rho_h_step2` adapter, and broad direct CLI generation and
  repair sweeps are release-gated. Broader spin-Hubbard final-table source
  generation and full-potential LDOS parity remain open.

## Current Snapshot

Baseline command:

- `cargo run -p xtask -- port-status --detail`
- `cargo run -p xtask -- compatibility-matrix --detail`
- `cargo run -p xtask -- compatibility-matrix --open-only --detail`

Current baseline:

- modules: 21
- module support: 21/21 = 100.0%
- explicit unported gates: 0
- unported gates with reference coverage: 0
- source-handoff coverage: 21/21 = 100.0%
- guarded production branches: 0 across 0 modules
- ignored parity/release checks: 0 across 0 modules
- branch compatibility blockers: 6 open rows

The module-support, source-handoff, guarded-branch, and ignored parity metrics
are now clean in the current inventory. The branch compatibility blockers
remain the release-completion backlog.

The compatibility matrix is intentionally stricter than the module inventory:
it records branch-level FEFF10 workflows that are covered, need broader
reference coverage, or still need implementation before `refeff` can be called
a complete FEFF10 replacement.

`cargo test --profile release -p refeff-cli full_run_completes_minimal_cu_smoke_input`
now runs as covered compatibility-matrix evidence that a fresh Rust full run
from `feff.inp` reaches `phase.bin`, `xsect.dat`, `chi.dat`, and `xmu.dat`.
`screen_module_matches_no_cache_inline_fms_generated_reference_when_present`
now runs as normal default coverage for SCREEN inline source-FMS parity.
`screen_module_matches_graphite_reference_zip_without_phase_or_gg_cache` now
runs as normal default coverage for non-Cu SCREEN source parity.
`cargo test --profile release -p refeff-cli pot_module_matches_` now runs as
covered compatibility-matrix evidence for bounded NiO Hubbard and BN
positive-`totvol` module SCF source runs matching local FEFF `pot.bin`
artifacts.
`cargo test --profile release -p refeff-cli bounded_feff_pot_reference` now
runs as covered compatibility-matrix evidence that full-run scheduling carries
the bounded NiO Hubbard and BN positive-`totvol` source runs through matching
FEFF `pot.bin` parity.
`cargo test --profile release -p refeff-cli true_scf_outputs_from_source_handoffs`
now runs as covered compatibility-matrix evidence for GeCl4, NiO Hubbard, LDOS
spin, and BN positive-`totvol` true-SCF POT source-output gates.
`cargo test --profile release -p refeff-cli true_scf_pot` plus
`full_run_scheduler_runs_bn_positive_totvol_pot_source_output` now run as
covered compatibility-matrix evidence that those source-output gates are also
reported as completed `pot` stages by full-run supported-module scheduling.
`cargo test --profile release -p refeff-cli reference_no_scf_outputs` now runs
as covered compatibility-matrix evidence for SF6, YBCO, MnF2 XMCD, and Gd L1
no-SCF source-generated POT outputs matching FEFF references.
`cargo test --profile release -p refeff-cli no_scf_pot_source_output` now runs
as covered compatibility-matrix evidence that full-run scheduling reports those
same no-SCF reference source bundles as completed `pot` stages.
`cargo test --profile release -p refeff-cli iterative_scf_outputs_with_high_exchange`,
`cargo test --profile release -p refeff-cli high_exchange_iterative`, and
`cargo test --profile release -p refeff-cli high_exchange_scf` now run as
covered compatibility-matrix evidence for high-`EXCHANGE` iterative SCF POT
module generation, scheduler source validation, and stale-cache repair.
`cargo test --profile release -p refeff-cli restart_iterative_scf` and
`cargo test --profile release -p refeff-cli external_iterative_scf` now run as
covered compatibility-matrix evidence for restart, external, and
external-restart iterative SCF POT source-output scheduler gates.
`cargo test --profile release -p refeff-cli updates_scf_retry_controls` now
runs as covered compatibility-matrix evidence for FEFF-style `nstarts` SCF
retry-control updates, including the max-start hard stop.
`cargo test --profile release -p refeff-cli atomic_module_assembles_terminal_scf_final_pot_candidate`
now runs as covered compatibility-matrix evidence that only converged and
iteration-limit SCF terminal states materialize final `pot.bin` candidates,
while repeat/missing-source/non-converged states preserve the unavailable
boundary.
`cargo test --profile release -p refeff-cli atomic_module_reaches_finite_nucleus_iterative_pot_scf_repeat_boundary_from_sources`
now runs as covered compatibility-matrix evidence for the bounded finite-nucleus
repeat-exhaustion source-loop boundary.
`cargo test --profile release -p refeff-core pot_scf` now runs as covered
compatibility-matrix evidence for FEFF SCF contour stepping, endpoint
finishing, source-row lifting, and density/coulomb outer-iteration composition.
`cargo test --profile release -p refeff-cli pot_scf` now runs as covered
compatibility-matrix evidence for CLI SCF source loops that build initial
states, advance contours, prepare next iterations, assemble FMS source grids,
and write full-run reference POT outputs from source handoffs.
`cargo test --profile release -p refeff-cli finite_nucleus` now runs as
covered compatibility-matrix evidence for finite-nucleus APOT/POT source
handoffs, no-SCF source outputs, iterative repeat-boundary handling, and the
full-run finite-nucleus source boundary.
`cargo test --profile release -p refeff-cli atomic_module_generates_finite_nucleus_apot_from_geometry_source_handoff_without_pot_bin`
now runs as covered compatibility-matrix evidence that finite-nucleus ATOM
source handoffs generate rendered APOT sections from `pot.inp` plus `geom.dat`
without cached `pot.bin`.
`cargo test --profile release -p refeff-cli atomic_module_generates_finite_nucleus_scf_state_from_pot_input`
now runs as covered compatibility-matrix evidence that finite-nucleus generated
SCF states select the finite nuclear mesh and differ from point-nucleus
generated states in their starting radii, nuclear potential, density, and first
large component.
`cargo test --profile release -p refeff-core atom_nuclear_potential_matches_feff_nucdev_reference`
now runs as covered compatibility-matrix evidence that point and finite nuclear
potentials match FEFF `nucdev` reference behavior.
`cargo test --profile release -p refeff-core atom_scf_state_from_configuration`
now runs as covered compatibility-matrix evidence that the composed atomic SCF
state driver threads finite-nucleus requests through FEFF-style state
construction.
`cargo test --profile release -p refeff-cli ldos_module_matches_production_fms_reference_from_source_handoffs`
now runs as covered compatibility-matrix evidence for production full-FMS LDOS
parity.
`cargo test --profile release -p refeff-cli ldos_module_matches_nonzero_fms_reference_from_source_handoffs`
and
`cargo test --profile release -p refeff-cli ldos_module_matches_ordinary_spin_fms_reference_from_source_handoffs`
now run as covered compatibility-matrix evidence for nonzero and ordinary-spin
FMS `gtrNN.bin`/LDOS/RHOC source-reference parity.
`cargo test --profile release -p refeff-cli ldos_module_roundtrips_hubbard_nio_reference_zip_magnetic_sidecars`
now runs as covered compatibility-matrix evidence for active-Hubbard NiO
LDOS/RHOC plus magnetic sidecar cache contracts.
`cargo test --profile release -p refeff-cli xanes_cu_no_fms_ldos` now runs as
covered compatibility-matrix evidence that full-run supported-stage scheduling
generates no-FMS LDOS/RHOC final tables from source handoffs and repairs stale
same-shape caches.
`cargo test --profile release -p refeff-cli active_hubbard_ldos` now runs as
covered compatibility-matrix evidence for full-run active-Hubbard LDOS
positive, repair, stale-grid/layout, malformed-sidecar, and
`gtr`/`gtr_m`/`gtr_off` source-contract gates.
`cargo test --profile release -p refeff-cli active_hubbard_cache` now runs as
covered compatibility-matrix evidence for direct-module active-Hubbard
ordinary, magnetic, and off-diagonal trace-source contracts, including matching
fallback source bundles for a nonzero cached potential and conflict/omission
rejections.
`cargo test --profile release -p refeff-core ldos_hubbard_magnetic_ff2rho_tables_match_feff_step2_order`
now runs as covered compatibility-matrix evidence for FEFF
`LDOS/ff2rho_h_step2.f90` magnetic `lmdosNN.dat`/`rhocmNN.dat` table assembly.
`cargo test --profile release -p refeff-core ldos_ff2rho_tables_match_feff_non_full_potential_reference`
now runs as covered compatibility-matrix evidence for FEFF
`LDOS/ff2rho.f90` non-full-potential final-table density formulas.
`cargo test --profile release -p refeff-core ldos_fmsdos_trace_matches_feff_non_full_potential_loop`
now runs as covered compatibility-matrix evidence for FEFF
`LDOS/fmsdos.f90` non-full-potential packed-`gg` trace projection.
`cargo test --profile release -p refeff-cli ldos_module_generates_` now runs as
covered compatibility-matrix evidence for direct LDOS source-generation breadth,
including kmesh, `gtr`, no-FMS, wavefunction/radial, zero-cluster FMS,
missing-pair, spin-pair, and module-log handoffs.
`cargo test --profile release -p refeff-cli ldos_module_recovers_` now runs as
covered compatibility-matrix evidence for direct LDOS repair breadth, including
malformed kmesh/log/output caches, paired LDOS/RHOC recovery, spin RHOC
recovery, and no-FMS active-Hubbard ordinary/magnetic sidecars.
Spin-Hubbard source generation and full-potential parity remain open LDOS
follow-up work.
`cargo test --profile release -p refeff-linalg complex32_general_eigenvalues`
and `cargo check --profile release -p refeff-core` now run as covered
compatibility-matrix evidence that BAND KKR/freeprop eigenvalue solves have a
pure-Rust `faer` CGEES-style adapter available to the release build.
`cargo test --profile release -p refeff-cli xsph_module_matches_broader_source_generated_reference_when_present`
now runs as covered compatibility-matrix evidence for multi-fixture XSPH source
parity across generated `phase.bin`/`xsect.dat` outputs, including legacy
FEFF `xsph.inp` handoff parsing, legacy 8- and 10-column `phase.bin` parsing,
old no-`config.dat` GeCl4 source parity from `pot.bin`, and the zip-backed
`XANES/BN`, `XES/BN`, `XES/GeCl_4`, and `NRIXS/GeCl_4` source fixtures.
`cargo test --profile release -p refeff-cli xsph_reference_phase_and_xsect_from_source_handoffs`
now runs as covered compatibility-matrix evidence that the full-run scheduler
carries those reference-backed source phase/xsect fixtures through completed
`xsph` reports.
`cargo test --profile release -p refeff-core xsph_phase_` now runs as covered
compatibility-matrix evidence for FEFF XSPH phase setup, skip, plasmon-pole,
radial-output, mesh, self-energy-summary, and reference-tail core primitives.
`cargo test --profile release -p refeff-cli positive_izstd` now runs as
covered compatibility-matrix evidence that positive-`izstd` source handoffs
ignore PMBSE controls like FEFF while still producing completed XSPH outputs at
module, scheduler, and full-run boundaries.
`cargo test --profile release -p refeff-cli global_multipole_xsph` now runs as
covered compatibility-matrix evidence that full-run scheduling carries global
multipole controls through completed source-backed XSPH output.
`cargo test --profile release -p refeff-cli two_spin_filtered_xsph` now runs as
covered compatibility-matrix evidence that scheduler and full-run paths carry
two-spin filtered XSPH source handoffs through completed outputs.
`cargo test --profile release -p refeff-cli full_run_scheduler_generates_remaining_ldos_xsph_reference_phase_and_xsect_from_source_handoffs`
now runs as covered compatibility-matrix evidence that LDOS FMS and ordinary
spin-FMS source handoffs generate scheduler `phase.bin`/`xsect.dat` outputs
matching FEFF references.
`cargo test --profile release -p refeff-cli full_run_scheduler_runs_nrixs_gecl4_xsph_source_handoff`
and
`cargo test --profile release -p refeff-cli nrixs_xsectjas`
now run as covered compatibility-matrix evidence for NRIXS/JAS source
generation of `phase.bin`, `xsect.dat`, `xsecl.dat`, `xsecl2.dat`, and
`xsecl.bin`.
`cargo test --profile release -p refeff-cli xsph_module_generates_` now runs
as covered compatibility-matrix evidence for broad XSPH CLI source-generation
branches, including empty-cell phase, Hubbard phase handoff, negative and
positive `izstd`, FPRIME, E2/L2LP controls, MPSE, AXAFS, NRIXS, phase-text,
and two-spin branch outputs.
`cargo test --profile release -p refeff-cli xsph_module_writes_tdlda_xsedge`
now runs as covered compatibility-matrix evidence for occupied, file-basis, and
generated-basis TDLDA/PMBSE `xsedge.dat` source generation.
`cargo test --profile release -p refeff-cli tdlda_xsedge_from_pmbse_source_handoffs`
now runs as covered compatibility-matrix evidence for the same TDLDA/PMBSE
branches at the full-run scheduler boundary, including stale `xsedge.dat`
repair for the file-basis and generated-basis projector paths.
`cargo test --profile release -p refeff-core xsph_tdlda_` now runs as covered
compatibility-matrix evidence for FEFF TDLDA/`xsectd` core formulas including
`getmat`, energy-row setup, `getchi0`, `ridxmu`, `kkchi`, channel weighting,
broadening, and final `xsedge.dat` row assembly.
`full_run_scheduler_generates_debye_dm_xanes_cu_xsph_reference_phase_and_xsect_from_source_handoffs`
now carries one of those broader fixtures through full-run orchestration:
`DEBYE/DM/XANES/Cu` starts from `xsph.inp`, `global.inp`, `pot.bin`, and
`config.dat`, omits cached `phase.bin`/`xsect.dat`, and compares generated
phase, cross-section, and `emesh` sidecars to the FEFF reference.
`full_run_scheduler_generates_elnes_cu_xsph_reference_phase_and_xsect_from_source_handoffs`
does the same for the `ELNES/Cu` source fixture, including its looser
reference-backed `xsect.dat` tolerance envelope from the module release gate.
`full_run_scheduler_generates_exafs_cu_scf_xsph_reference_phase_and_xsect_from_source_handoffs`
adds full-run coverage for the SCF-derived `EXAFS/Cu_SCF` XSPH source bundle,
again starting without cached `phase.bin`/`xsect.dat` and checking the
generated phase, cross-section, and `emesh` sidecars against FEFF.
`full_run_scheduler_generates_danes_cu_xsph_reference_phase_and_xsect_from_source_handoffs`
adds scheduler coverage for the ordinary `DANES/Cu` `ispec = 3` source
fixture, using only the upstream `xsph.inp`/`global.inp`/`pot.bin`/`config.dat`
plus `wscrn.dat` screened-potential handoff and comparing the generated phase,
cross-section, and `emesh` sidecars against FEFF.
`full_run_scheduler_generates_xes_cu_xsph_reference_phase_and_xsect_from_source_handoffs`
promotes the zip-backed `XES/Cu` `ispec = 2` reference into full-run scheduler
coverage, starting from `xsph.inp`, `global.inp`, `pot.bin`, `config.dat`, and
`wscrn.dat`, omitting cached phase/xsect/emesh outputs, and checking generated
`phase.bin`, `xsect.dat`, `emesh.dat`, `emesh.bin`, `axafs.dat`, and `mpse.dat`
sidecars at the completed `xsph` boundary.
`cargo test --profile release -p refeff-cli band_cr2gec_generated_bandstructure_matches_reference_when_present`
is now a covered compatibility-matrix row for source-generated Cr2GeC
`bandstructure.dat` parity.
`cargo test --profile release -p refeff-cli full_run_scheduler_generates_cr2gec_reference_bandstructure_from_source_handoffs`
now runs as covered compatibility-matrix evidence that full-run orchestration
carries that Cr2GeC source bundle through `bandstructure.dat` FEFF reference
parity; the measured focused release run completed in 63.14s.
`cargo test --profile release -p refeff-cli one_spin_rel_bandstructure` now
runs as covered compatibility-matrix evidence for completed one-spin
relativistic BAND module and scheduler source generation.
`cargo test --profile release -p refeff-cli full_run_scheduler_generates_freeprop_bandstructure_from_source_handoffs`
now runs as covered compatibility-matrix evidence for standalone full-run
`freeprop` BAND source generation; the measured focused release run completed
in 21.33s.
`cargo test --profile release -p refeff-cli band_module_generates_two_spin_non_degenerate`
and
`cargo test --profile release -p refeff-cli full_run_scheduler_generates_two_spin_non_degenerate`
now run as covered compatibility-matrix evidence for direct-module and scheduler
non-degenerate two-spin ordinary and `freeprop` BAND source generation.
`cargo test --profile release -p refeff-cli full_run_scheduler_generates_two_spin_degenerate_bandstructure_from_source_handoffs`
now runs as covered compatibility-matrix evidence for degenerate two-spin BAND
source generation; the measured focused release run completed in 9.20s.
`cargo test --profile release -p refeff-cli band_module_generates_` now runs as
covered compatibility-matrix evidence for the direct BAND module
source-generation sweep, including ordinary, `freeprop`, one-spin relativistic,
two-spin degenerate, two-spin non-degenerate, and kmesh/pre-solver handoffs.
`cargo test --profile release -p refeff-cli full_run_scheduler_regenerates_stale_one_spin_rel`
now runs as covered compatibility-matrix evidence for one-spin relativistic
ordinary and `freeprop` stale `bandstructure.dat` repair from source.
`cargo test --profile release -p refeff-cli full_run_scheduler_regenerates_stale_two_spin`
now runs as covered compatibility-matrix evidence for non-degenerate two-spin
ordinary and `freeprop` stale `bandstructure.dat` repair from source. The
`cargo test --profile release -p refeff-core band_structure_factor_from_kspace`
release gate now covers non-relativistic and relativistic BAND KSPACE
structure-factor grid assembly in FEFF loop order.
`cargo test --profile release -p refeff-core band_kkr_band_energies_from_kspace`
release-gates KKR source-grid eigenvalue counting through final FEFF
`bandstructure.dat` row identification for non-relativistic, relativistic, and
phase-composed ordinary paths.
`cargo test --profile release -p refeff-core band_free_propagation_band_energies_from_kspace`
release-gates raw-`G` `freeprop` source-grid eigenvalue counting through final
FEFF `bandstructure.dat` row identification for non-relativistic and
relativistic paths. The broader BAND row remains open for deeper
generated-output branch/reference broadening beyond these scheduler dispatch,
repair, and core row-generation gates.
`full_run_scheduler_generates_fprime_gecl4_xsph_reference_phase_and_xsect_from_source_handoffs`
adds scheduler coverage for the non-Cu `FPRIME/GeCl4` source fixture,
including FEFF's pure-imaginary `xsect.dat` convention and no-`mpse.dat`
completion path.
`full_run_scheduler_generates_ldos_spin_no_fms_xsph_reference_phase_and_xsect_from_source_handoffs`
and
`full_run_scheduler_generates_remaining_ldos_xsph_reference_phase_and_xsect_from_source_handoffs`
now carry all three broader LDOS-derived Cu source-release fixtures through the
same full-run scheduler boundary, generating `phase.bin`, `xsect.dat`,
`mpse.dat`, `emesh.dat`, `emesh.bin`, and `log2.dat` from
`xsph.inp`/`global.inp`/`pot.bin`/`config.dat` plus `wscrn.dat`.
`crpa_module_generates_reference_zip_from_source_without_phase_or_gg_cache`
now runs as normal default coverage for source-generated CRPA reference parity.
`sfconv_module_generates_specfunct_cache_without_reusable_cache` now runs as
normal default coverage, proving the source-generated SO2CONV
`specfunct.dat` path without a reusable cache.
`sfconv_module_does_not_claim_malformed_target_source_handoff` and the matching
full-run scheduler regression now keep malformed SO2CONV target spectra out of
SFCONV supported-stage accounting while direct execution remains strict.
`sfconv_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_sfconv_input` now keep malformed
`sfconv.inp` out of both SFCONV and SELF supported-stage accounting while direct
execution remains strict.
`self_module_regenerates_stale_exc_dat_from_loss_source` and the matching
full-run regression now cover readable stale SELF `exc.dat` repair from
`xsph.inp`/`loss.dat` many-pole source handoffs.
`full_run_scheduler_generates_mpse_cu_self_reference_exc_from_loss_source_handoff`
now promotes the same branch to a scheduler-level MPSE/Cu reference gate:
`rdinp` creates the enabled SELF handoff, the test supplies only the real
`xsph.inp` and `loss.dat` source tables, and the generated `exc.dat` is checked
against `REFERENCE/exc.dat` without copying a cached final output.
`xsph_module_does_not_claim_malformed_input_during_discovery`,
`self_module_does_not_claim_malformed_xsph_source_handoff`, and the matching
full-run scheduler regression now keep malformed `xsph.inp` source inputs out
of XSPH/SELF supported-stage accounting while direct execution remains strict.
`self_module_does_not_claim_cached_output_with_malformed_xsph_source_handoff`,
`self_module_does_not_claim_cached_output_with_malformed_loss_source_handoff`,
and
`full_run_scheduler_does_not_report_cached_self_when_xsph_source_handoff_is_malformed`
now keep readable `exc.dat` caches from masking malformed declared SELF source
handoffs.
`fms_module_does_not_claim_malformed_input_during_discovery` and the matching
full-run GENFMT scheduler regression now keep malformed `fms.inp` inputs out
of FMS supported-stage accounting and block downstream source-generated GENFMT
completion until direct FMS execution reports the parser error.
`fms_module_does_not_claim_cached_gg_with_malformed_phase_source_handoff` and
`full_run_scheduler_does_not_report_cached_fms_when_phase_source_handoff_is_malformed`
now keep readable FMS GG caches from masking malformed declared
`phase.bin`/`geom.dat`/`global.inp` source bundles during supported-stage
discovery.
`dmdw_module_regenerates_stale_cached_output_from_dym_handoff` and the matching
full-run scheduler regression now cover readable stale `dmdw.out` repair from
`.dym` handoffs.
`full_run_scheduler_generates_debye_dm_exafs_cu_dmdw_reference_from_dym_source`
now runs the DEBYE/DM/EXAFS/Cu `dmdw.inp` plus `feff.dym` source handoff,
omits cached `dmdw.out`, and compares the generated report to the FEFF
reference at printed-report precision.
`dmdw_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_dmdw_input` now keep malformed
`dmdw.inp` out of cached-output and source-handoff supported-stage accounting
while direct DMDW execution continues to report the parser error.
`path_module_regenerates_stale_paths_dat_from_source_handoffs` and the matching
full-run regression now cover readable stale `paths.dat` repair from
`phase.bin`/`geom.dat`/`global.inp` handoffs.
`path_module_does_not_claim_cached_output_with_malformed_phase_source_handoff`
and the matching scheduler regression now keep readable `paths.dat` caches from
masking malformed declared pathfinder source handoffs.
`path_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_paths_input` now keep malformed
`paths.inp` out of supported-stage accounting while direct PATH execution
continues to report the parser error.
`rhorrp_module_regenerates_stale_cached_core_output_from_source` and the
matching full-run regression now cover readable stale core-density repair from
`pot.bin`/`geom.dat` handoffs.
Full-run supported-stage scheduling now runs RHORRP after ATOMIC/config
preparation and before POT refresh or XSPH/FMS handoff work, so a declared
`DENSITY` request is satisfied from its active cache/source bundle before
unrelated downstream handoff failures stop orchestration.
XSPH source-handoff discovery now runs the same phase generator used by direct
execution before advertising phase-only or complete base-output support. Known
FEFF `xcpot` negative-radicand stops decline scheduler discovery, while direct
XSPH execution still reports the exchange failure; the RHORRP/POT refresh
regression pins that split.
`rhorrp_module_does_not_claim_cached_core_outputs_with_malformed_source`,
`rhorrp_module_does_not_claim_cached_non_core_outputs_with_malformed_source`,
and the matching scheduler regression now keep readable RHORRP density caches
from masking malformed declared core/table density source handoffs.
`fms_module_recovers_paired_malformed_gg_caches_from_source_handoffs` and the
matching full-run regression now cover malformed primary `gg.bin`/`gg.dat`
repair from `phase.bin`/`geom.dat`/`global.inp` source handoffs.
`fms_module_regenerates_stale_readable_gg_dat_from_source_handoffs` now covers
readable stale primary GG repair from those same source handoffs before MKGTR
consumes the Green-function matrices.
`genfmt_module_regenerates_stale_readable_base_cache_from_source_handoffs` and
`genfmt_module_regenerates_missing_nstar_from_source_handoffs_with_readable_base_cache`
plus `genfmt_module_regenerates_stale_nstar_from_source_handoffs_with_readable_base_cache`
and `genfmt_module_regenerates_stale_feffl_from_decomposed_jas_source_handoffs`
now cover readable stale GENFMT base-cache repair and source-owned optional
sidecar regeneration, including readable stale `nstar.dat` and decomposed JAS
`feffl.bin`, from `global.inp`/`phase.bin`/`paths.dat` handoffs. The full-run
scheduler now has the same readable-stale `nstar.dat` and `feffl.bin` gates
before reporting completed `genfmt`.
`genfmt_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_genfmt_input` now keep malformed
`genfmt.inp` out of supported-stage accounting while direct GENFMT execution
continues to report the parser error.
`genfmt_module_does_not_claim_cached_output_with_malformed_phase_source_handoff`
and
`full_run_scheduler_does_not_report_cached_genfmt_when_phase_source_handoff_is_malformed`
now keep readable `feff.bin`/`list.dat` caches from masking malformed declared
`phase.bin` source handoffs.
`ff2x_module_regenerates_stale_readable_exafs_outputs_from_source_handoffs`
now covers readable stale ordinary EXAFS `chi.dat`/`xmu.dat` repair from
`xsect.dat`/`feff.bin`/`list.dat` handoffs before cached-output acceptance.
`ff2x_module_regenerates_stale_sig3_cum_dat_from_source_handoffs` and
`full_run_scheduler_regenerates_stale_ff2x_cum_from_sig3_source_handoffs` now
cover readable stale source-owned `cum.dat` thermal-expansion diagnostics from
the same FF2X path-damping handoffs before completed `ff2x` reporting.
`full_run_scheduler_generates_exafs_cu_ff2x_reference_spectra_from_source_handoffs`
now carries the EXAFS/Cu source handoffs through scheduler-level `ff2x`
completion and compares generated `chi.dat`/`xmu.dat` spectra against the FEFF
reference.
`ff2x_module_regenerates_stale_readable_xanes_output_from_source_handoffs` and
`ff2x_module_regenerates_stale_readable_nrixs_xmul_from_source_handoffs` extend
that check to source-generated XANES `xmu.dat` and decomposed NRIXS `xmul.dat`
caches, so readable non-EXAFS spectra cannot mask current source handoffs.
`ff2x_module_does_not_claim_malformed_cache_without_source_handoffs` and the
matching full-run scheduler regression now keep malformed final-spectrum files
from being advertised as completed FF2X output unless complete source handoffs
can regenerate them.
`ff2x_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_ff2x_input` now keep malformed
`ff2x.inp` out of supported-stage accounting while direct FF2X execution
continues to report the parser error.
Malformed declared FF2X source handoffs such as `xsect.dat` now follow the same
scheduler-discovery rule: the supported-stage predicate declines completion
while explicit FF2X execution still reports the source parser failure. Readable
cached final spectra can no longer mask that malformed declared source state.
`fullspectrum_module_does_not_claim_malformed_eps_cache` and the matching
full-run scheduler regression now apply the same completion-predicate discipline
to FULLSPECTRUM: a standalone malformed `eps.dat` is not advertised as a
runnable cached optical-table stage.
`fullspectrum_module_does_not_claim_cached_output_with_malformed_drude_sidecar`,
`fullspectrum_module_does_not_claim_cached_output_with_malformed_osc_str_sidecar`,
`fullspectrum_module_does_not_claim_cached_output_with_malformed_pot_sumrules_source`,
and
`full_run_scheduler_does_not_report_malformed_fullspectrum_pot_source` extend
that rule to optional FULLSPECTRUM sidecars and sum-rule source state: readable
`eps.dat` can no longer mask malformed `drude.dat`, `osc_str.dat`, or `pot.bin`
handoffs during supported-stage discovery.
`opcons_module_does_not_claim_malformed_table_inputs` and the matching
full-run scheduler regression now require parseable `opcons*.dat` source tables
before OPCONS is advertised as a supported cached optical-loss stage.
`opcons_module_does_not_claim_malformed_pot_source_during_discovery` and the
matching scheduler regression now also keep malformed declared component-source
state such as `pot.bin` out of OPCONS completion reports while direct execution
keeps the parser error.
`full_run_scheduler_generates_mpse_cu_opcons_reference_loss_from_source_tables`
now promotes that path to a scheduler-level FEFF reference gate: the
`MPSE/Cu_OPCONS` zip supplies `opcons.inp`, `pot.bin`, and `opconsCu.dat`,
the scheduler generates `loss.dat` without copying the cached final output, and
the generated optical-loss table is compared to `REFERENCE/loss.dat`.
Malformed module inputs follow the same discovery/explicit-run boundary:
`fullspectrum_module_does_not_claim_malformed_input_during_discovery`,
`full_run_scheduler_does_not_report_malformed_fullspectrum_input`,
`opcons_module_does_not_claim_malformed_input_during_discovery`, and
`full_run_scheduler_does_not_report_malformed_opcons_input` now keep malformed
`fullspectrum.inp`/`opcons.inp` from being advertised as supported stages while
direct module execution still reports the parser error.
The COMPTON supported-stage predicate now also rejects malformed standalone
`jzzp.dat`/`rhozzp.dat` caches and readable `jzzp.dat` tables whose grid does
not match `compton.inp`; the RHORRP source-backed recovery path remains
advertised when those handoffs are complete.
`compton_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_compton_input` now keep malformed
`compton.inp` out of supported-stage accounting while direct COMPTON execution
continues to report the parser error.
`compton_module_recovers_malformed_jzzp_from_rhorrp_handoffs`,
`compton_module_recovers_malformed_rhozzp_from_rhorrp_handoffs`,
`compton_module_regenerates_stale_readable_jzzp_from_rhorrp_handoffs`, and
`compton_module_regenerates_stale_readable_rhozzp_from_rhorrp_handoffs` now
cover malformed and readable stale COMPTON `jzzp.dat`/`rhozzp.dat` repair from
complete RHORRP density callback handoffs. The
`full_run_scheduler_regenerates_stale_compton_outputs_from_rhorrp_handoffs`
regression now covers the same stale-cache repair through the full-run
supported-stage scheduler after RDINP prepares `compton.inp`.
`compton_module_does_not_claim_cached_jzzp_with_malformed_rhorrp_source_handoff`
and the matching scheduler regression now keep readable compatible COMPTON
caches from masking malformed declared RHORRP callback source bundles.
`screen_module_regenerates_stale_wscrn_from_vtot_and_apot_before_pot_vtot_recovery`
and the matching full-run regression now cover readable stale SCREEN
`wscrn.dat` repair from complete `vtot.dat`/`apot.bin` handoffs before a
stale table can drive optional `vtot.dat` regeneration.
`screen_module_regenerates_stale_wscrn_from_source_handoffs`,
`screen_module_does_not_claim_cached_output_with_malformed_phase_source_handoff`,
and
`full_run_scheduler_does_not_report_cached_screen_when_phase_source_handoff_is_malformed`
now extend that cache/source rule to complete SCREEN response source bundles:
readable `wscrn.dat` no longer masks stale or malformed
`pot.bin`/`config.dat` plus FMS source state during supported-stage discovery.
SCREEN and CRPA source discovery now also decline malformed declared
`screen.inp` handoffs instead of aborting full-run supported-stage discovery;
explicit SCREEN/CRPA execution still reports the parser error.
`crpa_module_does_not_claim_cached_output_with_malformed_screen_source_handoff`
and
`full_run_scheduler_does_not_report_cached_crpa_when_screen_source_handoff_is_malformed`
now keep readable `crpa.dat` caches from masking that malformed declared SCREEN
source state during supported-stage discovery.
`crpa_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_crpa_input` now keep malformed
`crpa.inp` out of cached-output, `crpa-wscrn`, and source-handoff
supported-stage accounting while direct CRPA execution continues to report the
parser error.
`ldos_module_does_not_claim_cached_tables_with_malformed_wavefunction_source_handoff`
and
`full_run_scheduler_does_not_report_cached_ldos_when_wavefunction_source_handoff_is_malformed`
now extend the same rule to readable LDOS table caches: declared wavefunction
source bundles must be readable before cached `ldosNN.dat`/`rhocNN.dat` output
can satisfy supported-stage discovery.
`pot_module_recover_malformed_pot_bin_from_no_scf_source_handoffs`,
`pot_module_regenerates_stale_pot_bin_from_no_scf_source_handoffs`,
`pot_module_preserves_cached_output_when_no_scf_source_selector_is_unsupported`,
`pot_module_recovers_inconsistent_pot_bin_before_apot_sidecar_from_source_handoffs`,
and the matching full-run regressions now cover malformed and readable stale
final `pot.bin`/`apot.bin` repair from complete no-SCF `pot.inp`/`geom.dat`
source handoffs, plus cache preservation when a hand-written no-SCF source
selector is outside the supported branch set. The full-run supported-stage
scheduler now pins that unsupported-selector cache-preservation branch directly,
reporting completed cached `pot` output instead of `pot-input` or
`pot-scf-source`. A matching no-cache module regression now also keeps that
unsupported selector out of all POT source-discovery predicates when
`pot.inp`/`geom.dat` are present without a final POT cache.
`pot_module_regenerates_stale_external_scf_pot_from_mtdp_handoff` and the
matching full-run regression now also cover readable stale terminal SCF
`pot.bin`/`apot.bin` repair for the `EXTPOT` MTDP/`sort.aip` source route, so
external-potential final caches cannot mask the active source handoff.
Malformed declared external-potential `sort.aip`/MTDP source handoffs now follow
the same rule: supported-stage discovery declines cached POT completion while
direct POT execution reports the external-source parser or validation error.
POT source capability predicates now also decline malformed `geom.dat` handoffs
instead of downgrading them to `pot-input`; the required runner still reports
the `geom.dat` parser/geometry validation error. Readable cached final POT
outputs no longer mask malformed declared `geom.dat` source handoffs during
supported-stage discovery. Custom-configuration `config.inp` handoffs now
follow the same discovery rule for ATOMIC and POT, so malformed custom
configuration source cannot be hidden by readable `apot.bin` or
`pot.bin`/`apot.bin` final caches.
`eels_module_regenerates_stale_cached_output_from_opconskk_sources` and the
matching full-run regression now cover readable stale `eels.dat` repair from
typed `opconsKK*.dat`/`xmu*.dat` handoffs.
`full_run_scheduler_generates_elnes_cu_eels_reference_from_xmu_source_handoffs`
now pins the same path against the generated `ELNES/Cu` FEFF reference bundle:
the scheduler starts with only `eels.inp` plus `xmu.dat` through `xmu09.dat`,
generates `eels.dat`, and compares the tensor spectrum to the reference table
instead of relying on a cached final output.
`eels_module_does_not_claim_malformed_opconskk_source_handoff` and the matching
full-run scheduler regression now require EELS source spectra to parse before
advertising EELS or EELS-MDFF source-backed completion.
Readable `eels.dat` caches are also declined when matching typed EELS source
handoff files are present but malformed, so final-output caches cannot mask bad
`xmu*.dat` or `opconsKK*.dat` sources.
`eels_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_eels_input` now keep malformed
`eels.inp` out of both cached-output and source-handoff supported-stage
accounting while direct EELS execution continues to report the parser error.
`eelsmdff_module_regenerates_stale_cached_output_from_xmu_sources` and the
matching full-run regression now cover readable stale `mdff.dat` repair from
typed `xmu*.dat` handoffs.
`eelsmdff_module_does_not_claim_cached_output_with_malformed_xmu_source_handoff`
and the matching scheduler regression now keep readable `mdff.dat` caches from
masking malformed typed EELS source spectra.
`eelsmdff_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_eelsmdff_input` now keep malformed
`mdff.inp` out of cached-output and source-handoff supported-stage accounting
while direct EELS-MDFF execution continues to report the parser error.
EELS-MDFF discovery now also checks for `mdff.inp` before parsing optional
`global.inp`, so unrelated malformed global state no longer aborts MDFF
supported-stage discovery when the MDFF module input is absent.
When `mdff.inp` is present, malformed optional `global.inp` now makes
EELS-MDFF cached-output and source-handoff discovery decline the stage instead
of surfacing the parser error; direct EELS-MDFF execution still reports the
global parser failure.
KSPACE/LDOS reciprocal `kmesh.dat` discovery now declines malformed declared
`reciprocal.inp` handoffs instead of advertising `kmesh` or `ldos-kmesh`;
explicit module execution still reports the reciprocal parser error.
DMDW source-handoff predicates now parse ordinary `.dym` sources and type-2
coupling tables before advertising source-backed completion; the matching
module and full-run scheduler regressions pin malformed `.dym` files as
unsupported source handoffs rather than completed DMDW work. DMDW cached-output
discovery now also parses `dmdw.out`, malformed final caches regenerate from a
valid `.dym` source handoff, and readable final caches no longer mask malformed
declared `.dym` sources.
FMS `idwopt=5` source-handoff readiness now also parses the referenced `.dym`
file from `dmdw.inp` before advertising source-backed completion, so malformed
FMS DMDW inputs cannot satisfy the supported-stage predicate by filename alone;
module and full-run scheduler regressions now pin that behavior.
BAND cached-output and source-handoff discovery now decline malformed
`band.inp` while direct BAND execution remains strict; the matching
`band_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_band_input` regressions pin
that scheduler boundary.
RHORRP supported-output discovery now declines malformed `density.inp` while
direct RHORRP execution remains strict; the matching
`rhorrp_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_rhorrp_input` regressions pin
that scheduler boundary. Readable RHORRP density caches also validate complete
declared core/table source handoffs first; malformed source bundles are now
declined during discovery and fail through explicit execution instead of being
hidden by cache replay.
LDOS cached-table, source-output, and `ldos-kmesh` discovery now decline
malformed `ldos.inp` while direct LDOS execution remains strict; the matching
`ldos_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_ldos_input` regressions pin that
scheduler boundary.
RIXS cached-output and solver-handoff discovery now decline malformed
`rixs.inp` while direct RIXS execution remains strict; the matching
`rixs_module_does_not_claim_malformed_input_during_discovery` and
`full_run_scheduler_does_not_report_malformed_rixs_input` regressions pin that
scheduler boundary.
RIXS cached-output discovery now also validates declared solver source
handoffs before accepting readable final spectra: malformed `phase.bin` source
state cannot be masked by a cached `herfd.dat` or `rixsET.dat`, while explicit
edge-specific handoffs still override malformed shared fallback files.
ATOMIC cached-output, source-apot, and config-handoff discovery plus POT
cached-output, source, SCF, and input-handoff discovery now decline malformed
`pot.inp` while direct ATOMIC/POT execution remains strict; the matching
`atomic_module_does_not_claim_malformed_input_during_discovery`,
`pot_module_does_not_claim_malformed_input_during_discovery`, and
`full_run_scheduler_does_not_report_malformed_pot_input_for_atomic_or_pot`
regressions pin that shared scheduler boundary.
Cached-output predicates for POT, XSPH, BAND, SCREEN, CRPA, LDOS, DMDW, FMS,
EELS, EELS-MDFF, PATHS, GENFMT, FF2X, SFCONV/SELF, FULLSPECTRUM, COMPTON,
OPCONS, RHORRP, and RIXS now decline orphan final artifacts when the corresponding
module input file is absent, so partial files such as `pot.bin`/`apot.bin`,
`bandstructure.dat`, `phase.bin`/`xsect.dat`, `wscrn.dat`, `crpa.dat`,
`ldosNN.dat`, `dmdw.out`, `gtrNN.bin`, `eels.dat`, `mdff.dat`, `paths.dat`,
`feff.bin`/`list.dat`, `xmu.dat`, `exc.dat`, `eps.dat`, `jzzp.dat`,
`rhozzp.dat`, `opconsCu.dat`, `density.dat`, or `rixsET.dat` do not make full-run
orchestration claim a completed stage.
At the current release gate this scheduler boundary is part of the completed
source-backed port: strict `xtask port-status` reports no unported module
gates, no guarded production branches, and no ignored parity checks, and the
workspace release tests pass. Broader SCF and branch/reference coverage
expansion remain post-completion parity work, not current release blockers.
The first implementation rule is to protect source-handoff predicates while
adding numerical state. A module should recover malformed sidecars only when
Rust actually regenerates or validates source-backed state for that run. A
valid pre-existing sidecar is not enough to mask a malformed final module log.

## Implementation Order

1. Stabilize shared source handoffs.

   Complete and harden handoffs that multiple modules consume:
   `pot.inp`, `config.dat`, `pot.bin`, `apot.bin`, `phase.bin`, `xsect.dat`,
   `rl.dat`, `gg.bin`, `wscrn.dat`, `vtot.dat`, `kmesh.dat`, and module logs.

   Acceptance:
   - malformed stale caches do not suppress source-backed recovery
   - compatible source handoffs repair missing or stale sidecars
   - standalone malformed caches remain strict failures
   - standalone `pot.inp` validation runs in `refeff run` without marking POT
     complete or creating POT-owned output files
   - iterative POT inputs validate the source-built initial SCF state without
     writing a final `pot.bin`, `apot.bin`, `log1.dat`, or `potNN.dat`
   - `START_FROM_FILE` POT inputs treat an existing `pot.bin` as restart
     source state, not as a completed final cache
   - `EXTERNAL_POT` inputs consume FEFF's active MTDP plus `sort.aip` source
     handoff before the final source-completion boundary

2. Broaden XSPH parity after the explicit unported fallback is retired.

   XSPH owns the phase/cross-section state consumed by FMS, BAND, SCREEN, CRPA,
   LDOS, and RIXS. The explicit XSPH unported fallback is retired: missing
   or incomplete `phase.bin` inputs now report a source-requirement error
   unless complete caches or supported `pot.bin`/`config.dat` handoffs can
   satisfy the stage. Remaining XSPH work is branch parity, not a
   module-level unported gate.
   The next XSPH subgoal is to broaden the nonstandard `izstd <= 0`,
   ordinary positive-`izstd` screened-dipole, and mixed positive-`izstd`
   dipole+E2 cross-section paths, then close remaining phase-shift branches
   and add reference-backed parity before widening support to the remaining
   `TDLDA/xsectd.f90` TDLDA/PMBSE output branches.
   Positive-`izstd` inputs that also carry PMBSE controls now mirror FEFF's
   `xsphsub.f90` reset by ignoring those PMBSE controls and using the ordinary
   source-backed `xsect` path. Ordinary DANES (`ispec = 3`) now uses the same
   source-backed `XSPH/xsect.f90` route as EXAFS/XANES/XES when TDLDA is
   disabled, and FPRIME (`ispec = 4`) now uses that same source path with
   FEFF's pure-imaginary `xsect.dat` convention. Positive-`izstd` M1 is
   intentionally guarded because FEFF `XSPH/radint.f90` stops for M1 in the
   nonrelativistic `ifl < 0` branch.
   `full_run_scheduler_generates_global_multipole_xsph_from_source_handoffs`
   and `full_run_scheduler_generates_two_spin_filtered_xsph_from_source_handoffs`
   now pin the global-control and two-spin filtered branches at the
   supported-module scheduler boundary, requiring completed `xsph` reports
   rather than downstream unported-module errors.
   The positive-`izstd` PMBSE-reset branch is now a scheduler/full-run
   regression gate:
   `full_run_scheduler_generates_positive_izstd_xsph_while_ignoring_pmbse_from_source_handoffs`
   requires a completed supported-module `xsph` report from source handoffs, and
   `full_run_ignores_pmbse_for_positive_izstd_xsph_source_handoff`
   reaches completed `xsph` output from source handoffs after the screened
   `phiscf` response path keeps large second-pole rows in double precision,
   conditions the `cchik` solve, and falls back to the unity field only when
   the solved screened field numerically collapses.
   TDLDA/PMBSE inputs now generate the FEFF `TDLDA/meshlda.f90` energy mesh
   (`ik0 = 0`), source-backed `phase.bin`, matching `emesh.dat`/`emesh.bin`
   sidecars, and the FEFF `TDLDA/xsectd.f90` `xsedge.dat` output without
   fabricating an ordinary `xsect.dat` table. The `xsectd` bridge
   source-generates `xsedge.dat` for
   occupied projectors (`ibasis = 0`), FEFF's calculated generated-basis
   projector subset (`ibasis != 0 && ibasis != 1`), and FEFF's file-read
   `ibasis = 1` projectors when the hard-coded `Vila/Orbs/mg.3p.dat` and
   `Vila/Orbs/mg.4p.dat` source files are present. Missing file-basis orbitals
   remain a non-claiming partial handoff; covered TDLDA/PMBSE runs now satisfy
   the XSPH required-stage contract through `phase.bin` plus `xsedge.dat`.
   `full_run_scheduler_generates_tdlda_xsedge_from_pmbse_source_handoffs`
   now pins the RDINP-driven PMBSE source bundle at the supported-module
   scheduler boundary, requiring a completed `xsph` report with `phase.bin`,
   `emesh` sidecars, and the generated unsplit `xsedge.dat` while keeping
   ordinary `xsect.dat` absent.
   `full_run_scheduler_generates_file_basis_tdlda_xsedge_from_pmbse_source_handoffs`
   and
   `full_run_scheduler_generates_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs`
   now pin the same direct scheduler boundary for the `ibasis = 1` file-read
   projector and `ibasis = 2` generated-basis projector branches.
   `full_run_generates_file_basis_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement`
   now carries the file-basis `ibasis = 1` TDLDA/PMBSE branch through RDINP and
   full-run supported-stage scheduling, including the hard-coded `Vila/Orbs`
   source orbitals and the generated four-row unsplit `xsedge.dat` boundary.
   The matching scheduler-negative gate now keeps that same file-basis branch
   as phase/emesh-only progress when the `Vila/Orbs` projector files are
   absent, so incomplete file projectors cannot be advertised as completed
   XSPH or produce `xsedge.dat`.
   `full_run_generates_generated_basis_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement`
   now covers the calculated generated-basis `ibasis = 2` branch at the same
   full-run scheduler boundary, proving it emits `phase.bin`, `emesh` sidecars,
   and a four-row unsplit `xsedge.dat` without claiming ordinary `xsect.dat`.
   Active `xsectd` selectors bypass ordinary base-output completion, so stale
   cached `xsect.dat` files no longer complete TDLDA/PMBSE required stages when
   `xsedge.dat` is absent, and the runner now bypasses ordinary `xsect.dat`
   validation entirely when the active TDLDA/PMBSE source bundle can generate
   `xsedge.dat`. Malformed ordinary cross-section caches and readable stale
   ordinary cross-section caches beside the source handoff therefore no longer
   block `xsectd` generation or get counted as completed XSPH output. Cached
   `xsedge.dat` acceptance now also checks the
   source-inferable TDLDA row count, split-column shape, and PMBSE energy grid
   before treating a readable table as branch-complete; stale shape or energy
   mismatches fall through to source regeneration when the PMBSE handoff bundle
   is complete. Same-grid stale numeric `xsedge.dat` regressions now pin the
   module runner and full-run scheduler to regenerate the source-backed
   `xsectd` table instead of preserving readable spectra. Direct module
   regressions now cover the occupied-orbital `ibasis = 0`, file-read
   `ibasis = 1`, and calculated generated-basis `ibasis = 2` projector
   branches.
   `full_run_scheduler_regenerates_stale_file_basis_tdlda_xsedge_from_pmbse_source_handoffs`
   and
   `full_run_scheduler_regenerates_stale_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs`
   now extend that direct scheduler stale-repair boundary to the file-read
   `ibasis = 1` and calculated generated-basis `ibasis = 2` branches. Declared
   but malformed PMBSE source bundles now also prevent cached `xsedge.dat`
   acceptance instead of being treated as absent optional source state.
   Cached `xsect.dat` base handoffs are validated against the active
   `phase.bin` dimensions and complex energy grid; generated-phase runs
   preserve only matching cached cross-section tables when the normal-potential
   source xsect path is not applicable. When complete source handoffs can
   generate the current normal `xsect.dat` branch, same-shape cached cross
   sections are also compared against that source result and regenerated if
   stale, including branch changes from `global.inp` angular controls.
   Requested AXAFS print sidecars are now
   compared against the `phase.bin`/`xsect.dat`-derived Rust table before a
   cached base stage is advertised; readable stale `axafs.dat` files regenerate
   from those handoffs while the supported-stage predicate still recognizes the
   repairable stage. Readable `phase.bin` caches are now compared against the
   typed source phase handoff too, so stale phase shifts, phase mesh metadata,
   reference energies, or potential labels regenerate from `pot.bin` and
   `config.dat` before the full-run XSPH stage is considered satisfied.
   Readable stale transition moments stored in `phase.bin` now also trigger the
   source `xsect.dat` route, which refreshes both the rendered cross-section
   rows and the dependent transition-moment block.
   Active-Hubbard `aphase_hubbard.bin` sidecars now follow that same source
   comparison rule against the generated `v_hubbard.bin` handoff, so readable
   stale phase sidecars are regenerated before cached XSPH completion is
   advertised.
   Requested `rl.dat` radial sidecars now apply the same
   rule against the normal-potential Rust phase handoff, so readable stale
   radial tables cannot mask regenerable source output while cached
   `phase.bin`/`xsect.dat` files are preserved. Readable stale `mpse.dat`
   sidecars now follow the same rule against the `phase.bin`/`pot.bin`-derived
   Rust MPSE table, so cached
   many-pole self-energy output cannot mask the active source handoff when it is
   regenerable. Cached NRIXS `xsecl.dat`/`xsecl2.dat` text sidecars must
   match the active `phase.bin` energy mesh, including the shifted text energy
   columns and stored row totals that match the printed channel columns, before
   they are preserved, and both text sidecars must use the same channel count
   and shared header scalars.
   Cached `xsecl.bin` final-state and transition dimensions must match the
   active `phase.bin` contract before it is preserved.
   The supported-stage predicate performs the same in-memory
   compatibility checks before reporting XSPH
   complete. NRIXS selections now require the complete `xsectjas` sidecar set
   (`xsecl.dat`, `xsecl2.dat`, and `xsecl.bin`) before a cached XSPH stage is
   advertised as full-run complete. The full-run supported-stage scheduler now
   covers that incomplete-sidecar boundary, stale sidecar energy grids, stale
   text sidecar row totals, primary and secondary malformed text sidecars,
   mismatched text sidecar channel layouts or shared headers, stale binary
   sidecar header dimensions, and malformed binary sidecars, refusing to report
   completed `xsph` from readable `phase.bin`/`xsect.dat` alone or from
   same-shape stale `xsectjas` sidecars. Direct XSPH required-run regressions
   now pin the same primary and secondary stale text-energy-grid and row-sum,
   primary and secondary malformed text, text-layout, shared-header,
   `xsecl.bin` transition-contract and final-state-contract failures, and
   malformed binary sidecars before module completion.

   Acceptance:
   - `has_supported_xsph_output` no longer depends on final-output caches for
     supported branches
   - `refeff run` reaches later modules with source-generated phase state
   - reference parity covers generated EXAFS/XANES/DANES/XES/FPRIME fixture
     paths plus the broader Debye, ELNES, SCF, and LDOS Cu source release gate
   - XSPH source-reference fixtures carry `global.inp` through phase/xsect and
     pre-phase mesh gates, including the NRIXS mesh-capacity path and the
     FEFF `ispec = 5` NRIXS/RHORRP `mk_rhorrp_grid` and user `grid.inp`
     branches, plus the JAS/NRIXS constant-energy `phmeshjas` route and the
     NRIXS/GeCl_4 `pot.bin`/`config.dat` source-generated `phase.bin` reference
     gate. Full-run supported-module orchestration now also covers the
     XANES/Cu screened-core-hole `xsph` scheduler report with source-generated
     `phase.bin`/`xsect.dat` row parity, a DEBYE/DM/XANES/Cu source-reference
     scheduler report with generated `phase.bin`, `xsect.dat`, `emesh.dat`,
     and `emesh.bin` parity, an ELNES/Cu scheduler report with the same source
     inputs and ELNES-specific `xsect.dat` tolerances, an EXAFS/Cu_SCF
     scheduler report for SCF-derived source handoffs, a DANES/Cu scheduler
     report for the ordinary `ispec = 3` source branch, a zip-backed XES/Cu
     scheduler report for the screened-core-hole `ispec = 2` branch, a
     FPRIME/GeCl4 scheduler report for the pure-imaginary `xsect.dat` branch,
     and the GeCl4 NRIXS/JAS fixture through completed `xsph` reporting with
     generated `phase.bin`, `xsect.dat`, `xsecl.dat`, `xsecl2.dat`,
     `xsecl.bin`, and `emesh` sidecars.

3. Finish ATOMIC and POT as the upstream potential pipeline.

   ATOMIC must produce the atomic handoffs from `pot.inp` and optional
   `config.inp`; POT must run the SCF loop and produce self-consistent
   potential handoffs. POT source validation now runs through the direct
   runner before the remaining SCF gate, so invalid `pot.inp` handoffs do not
   get masked by the generic unported-solver error. POT can also recover a
   missing or unreadable `apot.bin` sidecar from complete typed ATOM source
   handoffs (`pot.inp`, `pot.bin`, and `geom.dat`) before consuming the
   existing self-consistent `pot.bin` state through `wpot`; this removes a
   cached free-atom APOT dependency without claiming the POT SCF loop is
   complete.
   ATOMIC now also generates the full `WriteAtomicPots` `apot.bin` stream from
   `pot.inp` plus RDINP geometry alone, using generated SCF state columns and
   the source APOT overlap path; `pot.bin` remains an optional compatibility
   cross-check when it is already present. Missing geometry remains gated, while
   `HIGHZ`/finite-nucleus source inputs now thread FEFF's negative `nuc`
   request through the Rust ATOM SCF state driver. A direct ATOM module gate now
   covers the same finite-nucleus `pot.inp`/`geom.dat` source APOT stream
   without a cached `pot.bin`.
   POT can now also generate a typed no-SCF `pot.bin` for the `nscmt=0`
   source subset from `pot.inp` plus
   `geom.dat`, including finite-nucleus source inputs, using generated ATOM SCF
   states including normal core-hole/final-state rows, overlap density,
   FEFF's `iscfxc` ground-state XC selector (`vBH`, `PZ`, `PDW`, `KSDT`),
   separate `vvalgs` construction for high `EXCHANGE` branches, `sidx`, direct
   `istval`/`fermi` for single-potential inputs, and
   `movrlp`/`ovp2mt` muffin-tin projection for multi-potential interstitial
   state before routing through the existing APOT sidecar and `wpot` writer.
   No-SCF `START_FROM_FILE` inputs now use the existing `pot.bin` only as the
   FEFF `importpot` source for `vtot`, `vint`, `edens`, `rhoint`, and `xmu`,
   then write the generated final `pot.bin` and sidecars through the same route.
   No-SCF `EXTERNAL_POT` inputs now read FEFF's active
   `GeCl4.04.dft.mtdp` plus `sort.aip` handoff, overlay external
   muffin-tin radii/indices, `vtot`, `edens`, `vint`, and the HOMO/LUMO-derived
   Fermi level, and retain the generated `rhoint` for the external density
   tails; when combined, `START_FROM_FILE` imports still apply last to match
   FEFF's POT ordering. The no-SCF source path now also has reference-zip
   SF6 molecular smoke coverage in both the standalone POT module and the
   full-run supported-stage scheduler in addition to the existing Cu source
   reference, proving the multi-potential geometry path writes `pot.bin`,
   `apot.bin`, `potNN.dat`, and `log1.dat` without final-output caches. No-SCF POT
   generation now also writes the `apot.bin` sidecar from the same generated
   ATOM state set used for `pot.bin` instead of recomputing those states in the
   sidecar path, and the capability predicate only validates the source
   handoff. This lets the heavy XMCD/Gd L1 no-SCF source fixture complete the
   full POT output route under a short focused bound. Readable no-SCF final
   caches are now compared against the generated source payload and refreshed
   before `wpot` rendering when stale, so old `pot.bin`/`apot.bin` files cannot
   mask the Rust source driver. The cached-output predicate now performs the
   same render-normalized source comparison before advertising no-SCF POT as a
   completed cached stage, so readable stale `pot.bin` or `apot.bin` files are
   reported as source-repairable rather than cache-complete. Unsupported
   hand-written no-SCF source selectors such as `iscfxc=0` are no longer
   advertised during those predicate checks, so valid final caches remain
   usable when the source branch itself is outside the supported set.
   Iterative SCF inputs with complete `pot.inp`/`geom.dat` source handoffs now
   build the in-memory initial SCF `pot.bin` state, apply the Rust-backed
   `istprm` interstitial/FERMI setup into a `PotScfState` snapshot, and validate
   that it feeds the ported FEFF `broydn` plus `coulom` density/coulomb update
   adapter before the contour/scattering loop. Supported single-potential Be
   SCF handoffs now continue through the explicit source loop to a validated
   final `pot.bin` candidate; remaining POT work is broader generated-row
   parity, retry/exhaustion conversion, and unsupported branch coverage.
   The full-run supported-stage scheduler now also pins the RDINP regular
   core-hole Be path as completed POT output: the source loop writes terminal
   `pot.bin`/`apot.bin`/`pot00.dat` files and a POT-owned `log1.dat` before
   exposing the downstream XSPH source boundary.
   Readable SCF final `pot.bin`/`apot.bin` caches are now render-normalized
   against that terminal source payload before POT is advertised as
   cache-complete, and stale final pairs route through the SCF source-output
   handoff rather than the APOT-only sidecar path. The
   initial snapshot now also carries the Rust-backed `POT/grids.f90` SCMT
   `emg`/floor-step schedule and a strict all-potential FOVRG source-grid
   handoff packed as `(energy, potential, l, radial)` with wave numbers and
   `phamp` phase tables. The FOVRG retained photoelectron length now extends
   when needed to cover the muffin-tin match point plus the six-row inward
   integration history, so large-radius POT SCF rows no longer stop at the
   10-bohr retained-length cap. Compatible FOVRG rows now also feed a POT
   `fmsie` source-grid bridge: Rust derives the nonrelativistic
   `sqrt(2*(em-eref))` FMS wave number, solves the spin-free
   zero-Debye-Waller FMS cluster from `geom.dat`, leaves `gtr` zero for
   one-atom clusters per FEFF's `inclus.gt.1` guard, projects all-potential
   `gtr(l,iph)` traces, and records unavailable reasons when compact radial
   rows cannot be built. The generated radial/FMS grids now
   also feed the Rust `pot_scf_contour_source_rows` bridge in the POT preflight,
   validating the handoff into `xrhole`, `xrhoce`, `yrhole`, and `yrhoce`
   source rows before the live SCF loop is claimed. The CLI now also follows
   FEFF's POT SCMT contour sizing (`negx=80`, `nflrx=17`) and adaptive contour
   energy walk when generating source rows: it solves one FOVRG/FMS/rholie row
   for the current SCMT energy, appends that row to the prefix, and reruns the
   contour driver until a bracket/terminal status or the finite dynamic
   source-row guard is reached; unlike the previous preflight, this guard is no
   longer tied to FEFF's static `emg(1:neg)` table. The current Be iterative
   fixture now reaches a final-pot status from those generated rows instead of
   stopping on stale static-grid energies or an adaptive-source boundary;
   completing POT still requires closing generated-row parity for broader
   reference inputs and the remaining unsupported input branches. The core
   density layer now also has a POT-facing `rholie` post-radial-solver bridge
   that preserves
   complex `xrhole`, `xrhoce`, `yrhole`, and accumulated `yrhoce` work arrays
   on the FEFF 0.05 radial grid before they feed the existing `ff2g` valence
   integration path. A one-energy POT density subdriver now composes that
   bridge with `ff2g`; the live contour scheduler and final SCF convergence
   loop are now Rust-backed by the later SCF source-driver path. The FEFF
   `scmt` Fermi end-cap correction after contour bracketing is also ported:
   Rust refines the endpoint interpolation fraction, returns `xmunew`, and
   applies the matching final corrections to `xnmues` and `rhoval`. The
   adjoining `scmt` contour-search state transition is now Rust-backed too:
   it consumes FEFF's prebuilt `emg` grid, starts horizontal Fermi search,
   moves up/down between floors, and reports the lowest-floor bracket that
   feeds the endpoint correction. The per-energy all-potential `scmt` loop
   around `ff2g` is also a core helper now: Rust resets `xntot`/`fl`/`fr`,
   folds each potential's supplied `rholie`/FMS work arrays through `ff2g`,
   and returns the updated `xrhoce`, `xrhocp`, `yrhoce`, `yrhocp`, `rhoval`,
   and `xnmues` state for the next contour point. A source-row `scmt`
   contour-loop driver now composes those pieces: it copies
   `xrhoce`/`yrhoce` to previous-point state, consumes supplied radial/FMS
   work arrays in FEFF loop order, tracks `xndif`, delegates the up/down
   floor search, and applies the endpoint correction once the lowest-floor
   bracket is reached. The source-row adapter in front of that loop is now
   Rust-backed too: it lifts solved radial channels and FMS `gtr` tables over
   all supplied contour rows/potentials into the `xrhole`, `xrhoce`,
   `yrhole`, and `yrhoce` tables consumed by the contour driver. The
   next SCF iteration handoff is Rust-backed as well: completed contour
   brackets run through FEFF's occupation-count repeat check, the existing
   `broydn`/`coulom` adapter, and the `edens = edens - edenvl + rhoval`
   update with inactive tails zero-filled. The outer `potsub` convergence
   transition is ported too: Rust applies `nscmt_min`, `tolmu`, `tolq`,
   `tolsum`, and `tolqp`, either restores pre-`scmt` density/potential state
   for convergence or iteration-limit exit, or copies mixed `rhoval` into
   `edenvl` for the next `istprm` pass. A state-advance wrapper now carries
   `xmu`, `qnrm`, `qold`, `xnmues`, `edens`, `edenvl`, `vclap`, and Broyden
   workspace across supplied SCMT iterations without hiding the still-missing
   live SCF orchestration step. The first-call `istprm` muffin-tin radius setup
   is now Rust-backed too: it computes `rmt`, `inrm`, `lnear`,
   nearest-neighbor bookkeeping, explicit-`OVERLAP` `inters mod 6`
   normalization, and `folpx` reductions before the existing `movrlp`/`ovp2mt`
   helpers consume that state. The second `istprm` block is now composed in
   Rust as well: `sidx` tail adjustment, `iscfxc` ground-state XC selection,
   `vtot`/`vvalgs` construction, FEFF `volint`/`rnrmav`, `movrlp`/`ovp2mt`
   projection, the `vint >= xmu` fixed-potential retry, and final `fermi`
   state all run through a core helper. The CLI now feeds generated contour
   rows into that state-advance wrapper for the initial SCF preflight when the
   radial and FMS source grids are available, and prepares a following
   non-converged iteration by copying the post-`scmt` `edens`/`edenvl`/`vclap`
   state through the Rust `istprm` helper while preserving FEFF's SCMT Fermi
   level, then attempting the next radial/FMS/contour source-row bundle from
   that prepared state with the same success-or-reason handoff used for the
   initial pass. When those source rows are available, the prepared bundle is
   also advanced through the `iscmt=2`/`first_scmt_call=false` SCMT wrapper. The
   CLI now wraps those source-backed advances in an explicit SCF loop driver,
   continuing across prepared iteration bundles until convergence, iteration
   limit, or a missing-source boundary. Terminal convergence or iteration-limit
   states now materialize a validated in-memory final `pot.bin` candidate by
   overlaying the SCF state (`xmu`, `qnrm`, `xnmues`, `edens`, `edenvl`,
   `vclap`) onto the latest `istprm` POT snapshot, and a release-profile gate
   rejects repeat/missing-source/non-converged statuses from that final-output
   path. The POT FOVRG flat-potential
   bridge now switches to a decaying/growing Hankel basis for large imaginary
   `ck*r` rows and treats singular muffin-tin `phamp` rows as recoverable
   interior matches; the Be iterative fixture no longer falls back to all-zero
   phase amplitudes. The generated POT state now applies a
   source-backed `corval` boundary correction from ATOM orbital energies,
   `xnval`, `CORVAL`, and the projected interstitial potential instead of feeding
   RDINP's raw `ecv` default into SCMT; core-to-valence reassignments now also
   update the generated `xnvmu`, per-orbital `xnval`, and `edenvl` density tables
   using FEFF's `corval.f90` degeneracy rules. The SCF source context now also
  runs the FEFF-style `corval` LDOS peak scan through a request-mask source
  handoff that solves only the suspicious `(l, potential)` channels and keeps
  the embedded `xrhoce` rows needed for first-peak detection, then regenerates
  the initial POT state with those channel peak energies before reassignment.
  Adaptive SCF source-row generation now prepares a reusable POT FOVRG
  source-grid plan once per `scmt` attempt and reuses it for deterministic
  prefix rows plus dynamic horizontal-search extensions, and its per-potential
  `rholie` channel solves run in deterministic parallel batches before the
  packed source-grid arrays are filled. Full-run scheduler coverage now also
  pins the LDOS spin Cu true-SCF final-state-screening fixture as completed
  `pot` output, matching the standalone module source route. The shared FOVRG
  solver now also skips
  building the derivative-backed C3 potential vector when FEFF's `ic3` scale is
  zero, preserving the existing zero-C3 path while avoiding unused work in the
  POT/RHORRP s-wave channels; POT source-grid and CORVAL LDOS handoffs now
  precompute nonzero C3 vectors once per potential/angular channel and reuse
  them across contour energies and regular/irregular solves. The POT FOVRG
  adapter now also
   trims trailing valence-only zero-component `config.dat` orbitals before the
   `rholie` channel solve, which lets the LDOS spin Cu source fixture complete
   a bounded one-iteration SCF POT run without cached `pot.bin`/`apot.bin`.
   The Rust SCMT wrapper now also mirrors FEFF's first `scmt` call behavior by
   tolerating bad occupation counts on the initial call and padding the returned
   `xnmues` table to the persisted `xnvmu` channel shape, which advances the Be
   fixture through the first density update. The source-backed SCF state now also
   initializes `qnrm`/the Norman-charge reference from FEFF's zero history state
   rather than from the static generated `pot.bin` atomic charge, so the first
   Broyden neutrality correction no longer over-mixes the valence density and
   the next `istprm` projection keeps a positive interstitial density. The
   source-loop wrapper now also treats `RepeatRequired` like FEFF's bounded
   `nstarts` retry path by carrying the `corval`-adjusted `ecv` into the next
   source context, skipping directly to the final start when `ecv` changes by
   less than 0.05 Ha, and reducing `ca1=max(ca1/5,0.01)` only on FEFF's
   reduced-mixing branch. Finite-nucleus iterative SCF now uses a wider
   adaptive source-row guard, so the Be `HIGHZ` smoke fixture reaches the
   floor-1 SCMT bracket before landing on the later bounded-repeat boundary
   instead of stopping at `NeedsMoreSourcePoints`; full-run orchestration now
   carries the same RDINP `HIGHZ` source bundle to the `pot-scf-source`
   repeat boundary. The release gate now also pins that finite-nucleus
   repeat-exhaustion boundary as a non-final source-loop outcome after bounded
   FEFF-style start attempts. The screened-core-hole NiO
   reference gate now runs through two source-backed SCMT passes before writing
   `pot.bin`/`apot.bin`, so multi-potential core-hole coverage is no longer
   limited to the one-iteration terminal-output path. The single-potential
   regular core-hole Be smoke fixture now reaches a bounded two-pass
   `ReachedIterationLimit` output with renderable `pot.bin`/`apot.bin`
   candidates after the `ecv` eV-to-Hartree correction keeps the synthetic
   core/valence boundary out of the repeat-retry path.
   The
   CLI also derives the SCMT
   density/Coulomb radial bound from FEFF's `rholie` `nr05` Norman-radius
   formula instead of reusing the wider `istprm` density-tail index, while the
   FOVRG handoff records separate `ilast` solve lengths and `nr05` density
   prefixes so `dfovrg` keeps its inward-history rows. The POT runner now attempts
   that terminal candidate before the loop-only handoff; when available it
   writes final `pot.bin`, regenerates the `apot.bin` sidecar through the
   existing ATOM APOT source path, and continues through `wpot`/`log1.dat`. The
   EXAFS/Cu reference `pot.inp`/`geom.dat` source-only fixture now reaches that
   terminal path and writes `pot.bin`, `apot.bin`, `pot00.dat`, `pot01.dat`, and
   `log1.dat` without cached POT outputs. The public `run_for_input` path and
   the full-run supported-module scheduler now try that source-backed
   final-output route directly, so supported SCF runs no longer need a full solver
   preflight before executing the same source driver. `START_FROM_FILE` SCF
   inputs now import the FEFF `importpot` subset from an existing `pot.bin`
   (`vtot`, `vint`, `edens`, `rhoint`, and `xmu`) after the normal `istprm`
   setup, and the POT runner no longer advertises that restart file as a final
   cache or APOT sidecar source unless the source-backed generation path writes
   a new final `pot.bin`. `EXTERNAL_POT` SCF preparation now performs the same
   MTDP/`sort.aip` overlay before any `START_FROM_FILE` restart import, so the
   Rust path preserves FEFF's source ordering while keeping the generated
   interstitial density for external tails. Higher `EXCHANGE` branch inputs now
   preserve the separate valence potential through the source-backed `pot.bin`
   writer, with an `EXCHANGE 5` iterative SCF output gate and `EXCHANGE 6`
   no-SCF module/full-run gates covering that path. The direct module
   `EXCHANGE 6` no-SCF gate now also regenerates stale readable
   `pot.bin`/`apot.bin` final files from unchanged `pot.inp` plus `geom.dat`
   source handoffs.
   Full-run orchestration now promotes terminal SCF source-driver outcomes to
   completed `pot` scheduler reports once `pot.bin`/`apot.bin` are renderable,
   and it rebuilds stale or incomplete SCF `pot.bin`/`apot.bin` final pairs from
   `pot.inp` plus `geom.dat` before making that completed-stage claim, while
   non-terminal repeat-boundary outcomes remain explicit `pot-scf-source`
   reports. The same scheduler route now has a XANES/BN positive-`totvol` gate
   that writes the three-potential POT output set from source handoffs under
   the bounded one-iteration cap, and the high-`EXCHANGE` RDINP-generated
   source bundle now reaches the completed `pot` report before exposing the
   downstream XSPH source boundary.
   `full_run_regenerates_stale_high_exchange_scf_pot_from_rdinp_sources_before_xsph_error`
   now also verifies that the same high-`EXCHANGE` full-run route regenerates
   stale readable `pot.bin`/`apot.bin` from `pot.inp` plus `geom.dat`, keeps
   the separate valence-potential branch, and avoids a loop-only
   `pot-scf-source` report before the expected downstream XSPH source
   boundary. The standalone POT module now has the matching stale
   high-`EXCHANGE` final-cache regression for direct source handoffs.
   Compatible `START_FROM_FILE` SCF runs now use the same full-run completion
   rule: the pre-existing restart `pot.bin` is not advertised as a final cache,
   but once the source driver writes a new renderable `pot.bin`/`apot.bin`,
   `pot00.dat`, and POT `log1.dat`, the scheduler reports completed `pot`
   output instead of a loop-only `pot-scf-source` handoff.
   The iterative `EXTERNAL_POT` scheduler branches now have matching full-run
   coverage: compatible MTDP/`sort.aip` source state, both standalone and with
   a compatible `START_FROM_FILE` restart `pot.bin`, promotes to completed
   `pot` output after the source driver writes new renderable POT files. The
   standalone external-MTDP route now also regenerates readable stale
   `pot.bin`/`apot.bin` final files from the unchanged MTDP/`sort.aip` source
   handoff before accepting completed `pot` output, while incomplete external
   restart state remains a non-final `pot-scf-source` boundary.
   Successful source-backed SCF runs now also carry the final `apot.bin` sidecar
   out of the same atomic state columns used for the initial POT setup, so the
   POT runner writes final `pot.bin`/`apot.bin` together instead of recomputing
   high-Z ATOM SCF states during sidecar preparation. A bounded RIXS/Pt source
   probe now reaches the complete output route in roughly half the previous
   debug-build wall time while still writing the expected POT output set.
   Adaptive POT SCF source-row advancement now accumulates the generated
   FOVRG/FMS grids alongside contour rows and returns that accumulated state at
   the terminal or guarded boundary, avoiding a second full source-grid rebuild
   after the contour search has already paid for each row. The same adaptive
   wrapper now batches FEFF's deterministic prefix rows (`nflrx` rows for the
   first `scmt` call, `neg` rows for repeat calls) before switching back to
   one-row dynamic horizontal-search extension.
   The direct `pot` module runner also uses that same single-pass SCF output
   writer now; it no longer runs the full source-backed SCF driver once as a
   capability predicate and again to write the final POT/APOT pair. Module and
   full-run orchestration now use a single SCF source-driver outcome for both
   final-output and loop-validated non-final handoffs, so a repeat-boundary run
   does not immediately execute the same source loop again before reporting the
   validated boundary. A bounded
   XANES/GeCl4 true-SCF reference-derived gate now covers the final-output
   route for a multi-potential molecular reference by lowering only `nscmt` to
   FEFF's iteration-limit path. Bounded HUBBARD/NiO true-SCF gates now cover
   the same final-output route for the screened core-hole, multi-potential
   oxide reference with two source-backed SCMT passes, and the standalone module
   plus full-run scheduler gates compare generated geometry and electron-density
   rows against the archived FEFF reference. Optional bounded FEFF parity gates
   now scan `reference-work/tmp/feff-pot-nio-bounded.*/pot.bin`, with
   `REFEFF_NIO_BOUNDED_FEFF_POT_BIN` as an override, and compare the same
   two-iteration NiO run against a local FEFF POT artifact. They confirm full
   `pot.bin` row parity, including carried `edenvl` valence-density rows.
   The no-SCF EXAFS/YBCO reference now reaches the complete `pot` output route
   from source handoffs after fixing the `pot.bin` `iorb` parser to accept full
   `potential_count * iorb_slot_count` payloads for five-potential states; a
   focused POT module reference gate now covers the generated `pot.bin`,
   `apot.bin`, five `potNN.dat` outputs, and `log1.dat` route, and a full-run
   scheduler gate now carries the same YBCO no-SCF source handoffs through the
   completed `pot` report outside the standalone module wrapper. These run
   alongside the fast codec regression and YBCO APOT source regression for that
   boundary.
   Source-generated no-SCF `pot.bin` keeps the FEFF reference behavior of
   deriving `xnatph` from `geom.dat` atom counts, while initial SCF `pot.bin`
   preserves RDINP's `xnatph` values from `pot.inp`; the XANES/BN
   positive-`totvol` gate now covers the true-SCF volume path at the POT
   module boundary, writing `pot.bin`, `apot.bin`, three `potNN.dat` outputs,
   and `log1.dat` from source handoffs under a bounded one-iteration run.
   ATOM/POT source preparation now translates `WARNION` custom `config.inp`
   rows into an effective atomic-solver ionicity derived from the compacted
   electron count, clearing the previous electron-count mismatch for ionized
   custom Cu configurations before the POT SCF source-completion boundary.
   POT APOT sidecar preparation now uses a strict ATOM source writer, so
   malformed source handoffs report the APOT validation error before the broad
   POT source/cache requirement.
   SCF `istprm` preparation now routes the CLI through the Rust first-call
   muffin-tin radius setup, persists the resulting `rmt` values into generated
   POT state, and converts positive `totvol` values from FEFF input units into
   Bohr^3 before overlap-volume/projection calculations. Generated `pot.bin`
   now preserves that converted total volume in the FEFF scalar block while the
   reduced interstitial volume remains internal to muffin-tin projection. A
   true-SCF XANES/BN positive-`totvol` regression now covers that scalar plus
   muffin-tin radius, Norman-radius, overlap, and volume boundaries without
   running the full reference convergence loop.
  The GeCl4 true-SCF plus SF6, YBCO, XMCD/MnF2, and XMCD/Gd L1 no-SCF POT reference gates
  now also compare generated `pot.bin` radius, overlap, and density rows against
  the FEFF reference at both the standalone module and full-run scheduler
  boundaries instead of only checking topology and finite output. The HUBBARD/NiO
  screened-core-hole module and full-run scheduler gates now pin the
  corresponding geometry and electron-density row parity against the archived
  full FEFF reference, and optional bounded FEFF gates prove the matching
  `edenvl` valence-density rows when Rust and FEFF both run with `nscmt=2`. The
  BN module and scheduler gates now pin `electron_density[27]` as an explicit
  bounded-run diagnostic: the source run is capped at one SCF iteration, while
  the archived FEFF reference converges in seven iterations, and that density
  anchor remains outside the full-reference row-parity tolerance. Matching
  bounded FEFF gates now scan
  `reference-work/tmp/feff-pot-bn-positive-totvol-bounded.*/pot.bin`, with
  `REFEFF_BN_POSITIVE_TOTVOL_BOUNDED_FEFF_POT_BIN` as an override, and prove
  full `pot.bin` row parity for the same one-iteration FEFF run, including the
  carried `edenvl` valence-density rows. The Rust `corval` path now converts
  RDINP's eV `ecv` input to Hartree before comparing with orbital energies, so
  deep core levels no longer enter the LDOS peak request mask. The remaining
  POT gap is broadening
  numerical row parity to other FEFF references, plus converting
  retry/exhaustion cases into final source outputs where FEFF can finish.
   The ATOM core now has a source-backed
   `atomic_total_energy_from_radials` assembly helper that composes the ported
   `etotal.f90` algebra with the ported `fdrirk.f90` radial-integral driver and
   owns FEFF's previous-first-factor sentinel state. The CLI now feeds that
   helper from real `apot.bin`/atomic-orbital source tables, including the
   FEFF `NOHOLE`-dependent total-energy column selection, and regenerates the
   EXAFS/Cu, XANES/Cu transition, and NRIXS/GeCl_4 `fpf0.dat` references
   without `fort.16`. The core also now has a composed
   `atomic_dirac_bound_orbital` driver for one `soldir.f90` call, built from
   the ported `intdir`, matching, node-search, energy-correction, and
   normalization pieces. That driver now feeds `atomic_initial_orbitals`, the
   Rust `wfirdf.f90` initializer for radial mesh, nuclear/Thomas-Fermi
   starting potential, origin powers/scales, and starting orbital tables.
   Rust also now composes the positive-`niter` `scfdat.f90` scheduler from
   those tables: FEFF's active-orbital selection loop, optional active
   `lagdat` refresh through the ported radial-integral driver, `potrdf`,
   `vlda`, the method-1 `soldir` call, convergence mixing, `dsordf`
   normalization, final total/valence density recomputation, FEFF's returned
   `srho/r**2` density tables, and `vcoul = potslw(srho) - Z/r`. The
   `atomic_scf_state_from_configuration` driver now composes the production
   ATOM state chain from a compacted `getorb` configuration through `inmuat`,
   `wfirdf`, Coulomb angular coefficients, and positive-`niter` `scfdat`,
   giving CLI code one source-backed numerical entry point for each atomic
   state. The `refeff-io` source adapter now assembles one or more converged
   SCF states into the FEFF `apot.bin` section subset consumed by current ATOM
   source handoffs: merged `norb`, `rho`, `rhoval`, `vcoul`, `xnval`, `eorb`,
   and `kappa` tables, plus FEFF-ordered per-state
   `dgc`/`dpc`/`adgc`/`adpc` matrices. Borrowed-state helpers now populate
   that section subset directly from core `AtomicScfState` values. The ATOM
   CLI now has a staged `generated_atomic_scf_apot_bin` helper that derives
   state-column configurations from `pot.inp`/`config.inp`, runs the core SCF
   driver, and emits that SCF section subset. The core driver now follows
   FEFF `scfdat`'s `xnvalp` table selection for Coulomb angular coefficients,
   so pure Dirac-Fock source states use zero valence coefficients while
   preserving actual `xnval` for output. The CLI source layer also derives the
   FEFF section-21 `iorb(-5:4,0:nph+1)` matrix from compacted `getorb`
   configurations, including K-edge final-state screening projections, and
   now derives section-5 core-hole columns (`dgc0`, `dpc0`, `drho`,
   `dvcoul`) from the same generated SCF states. The section-5 path follows
   FEFF `apot.f90` for zero no-hole density, `NOHOLE=1` initial-orbital
   density, and transition-state initial/final core-density differences. The
   staged CLI layer also derives APOT static source arrays from typed
   `pot.inp`, `geom.dat`, and `pot.bin` handoffs: `iz`, `iatph`, `novr`,
   `rnrm`, `iphat`, Bohr-scaled `rat(3,nat)`, and manual overlap shell
   matrices `iphovr`, `nnovr`, and Bohr-scaled `rovr`. The same source layer
   now derives the FEFF `ovrlp` APOT arrays from generated SCF states and
   static geometry/overlap inputs: `edens`, `edenvl`, `vclap`, replacement
   Norman radii, and the current spin-unpolarized `dmag/edens` table for
   explicit `OVERLAP` shells and geometry-neighbor mode. It also derives
   `xnvmu` from compacted `getorb` configurations using FEFF `scfdat`'s
   kappa-to-angular-channel accumulation for `l=0..3`, derives `s02` from
   generated initial/final absorber SCF states by rebuilding FEFF's
   relaxed-overlap matrix before calling the Rust `s02at` port, and derives
   `erelax`/`emu` from generated absorber total energies, the frozen
   initial-state core orbital energy, and FEFF's `vcoul(1,0)-vclap(1,0)`
   overlap shift. The CLI now stages those source arrays into the higher-level
   `apot_atomic_pots_sections` adapter, producing the complete
   `WriteAtomicPots`-ordered section stream, including scalar, core-hole,
   geometry, overlap, overlapped-density, `iorb`, and per-state orbital
   sections. The source APOT overlap path now also normalizes isolated
   no-overlap generated density columns within a tight charge tolerance when
   FEFF `frnrm` would otherwise miss `Z` by a tiny truncation amount. The ATOM
   CLI gate now uses the full stream for complete `pot.inp`/`geom.dat`/`pot.bin`
   source handoffs, generating or replacing `apot.bin` before the existing
   `config.dat`, optional `fpf0.dat`, and shared `log1.dat` handling. Incomplete
   handoffs still stop at the explicit ATOM boundary until the remaining POT
   source path can provide the upstream `pot.bin` arrays.

   Acceptance:
   - `apot.bin` can be generated from source, not only re-rendered
   - `pot.bin` can be generated from source, not only consumed by `wpot`
   - `log1.dat` ownership is deterministic between ATOMIC and POT

4. Wire BAND from existing core KKR/KSPACE kernels.

   The core BAND numerical helpers already cover many pieces, and the CLI now
   validates source-backed phase search, K-path, and `STRVECGEN` lattice-list
   setup from `phase.bin` plus `reciprocal.inp`, including `QJLTAB`, real Gaunt
   triples, `CIPWL`, STRSET site layout, basis transforms, relativistic
   `NRREL`/`IRREL`/`SRREL` component tables, and spin-orbit tables, reduced
   `STRCC` energies, solver state-ket/FMS atom basis,
   per-energy lattice T-matrix assembly, on-demand non-rel KSPACE point-input
   assembly, the borrowed relativistic `IREL >= 2` point-input boundary, and
   guarded relativistic KKR/`freeprop` grids and final row adapters for
   compatible one-spin handoffs.
   For compatible one-spin ordinary/`freeprop` handoffs,
   spin-degenerate multi-spin handoffs, and non-degenerate spin-resolved
   multi-spin handoffs using FEFF's final-spin scalar `fmsband` wave-number
   semantics, Rust can now iterate the full search-energy/k-path grid, run the
   appropriate KKR or raw-`G` eigenvalue counting path through the pure-Rust
   `faer` general-complex eigenvalue adapter, use full-order two-spin `IREL >=
   2` KSPACE `G` blocks for multi-spin handoffs, and assemble
   `bandstructure.dat` through the production CLI/full-run path.
   Full-run orchestration now reports those completed source outputs as `band`,
   while validation-only pre-solver/kmesh paths remain `band-handoff`. The
   module runner now also compares any valid cached `bandstructure.dat`
   summary header metadata, k-point shape, per-row band counts, and band
   eigenvalue rows with the supported source handoff output, so readable stale
   final caches regenerate from source instead of masking the Rust BAND driver.
   Full-run scheduling now pins same-shape stale eigenvalue regeneration
   directly and also pins the `freeprop` final-row shape path: a readable
   `bandstructure.dat` whose per-row band counts no longer match the source
   handoff is regenerated as a completed `band` stage rather than downgraded to
   a validation-only `band-handoff`.
   `full_run_scheduler_regenerates_stale_two_spin_bandstructure_from_source_handoffs`
   and
   `full_run_scheduler_regenerates_stale_two_spin_freeprop_bandstructure_from_source_handoffs`
   now extend that direct scheduler stale-cache boundary to non-degenerate
   two-spin ordinary and `freeprop` source bundles. Matching direct BAND module
   regressions now generate and regenerate two-spin ordinary and `freeprop`
   `bandstructure.dat` caches from the same source handoffs.
   BAND source setup now
   consumes optional `fms.inp` `lmaxph(0:nph)` cutoffs, matching FEFF's
   `reafms -> kprep` source path where KSPACE `maxl`/`msize` come from FMS
   active angular cutoffs rather than the larger raw phase-write range. The
   optional Cr2GeC reference handoff now pins the resulting 128-state active
   matrix order, and the local generated Cr2GeC BAND reference gate runs the
   Rust source path through `bandstructure.dat` and compares it with FEFF's
   generated file, including summary header metadata. Full-run supported-module
   orchestration now carries the same generated Cr2GeC `band.inp`/
   `reciprocal.inp`/`fms.inp`/`global.inp`/`phase.bin` source bundle through a
   completed `band` report and compares the scheduler-produced
   `bandstructure.dat` with the FEFF generated output. Malformed BAND
   `fms.inp` `lmaxph` handoffs now decline both the direct pre-solver report
   and the full-run `band`/`band-handoff` scheduler reports instead of
   advertising source-backed completion from the remaining source files.
   Full-run orchestration also carries the KSPACE/Graphite `reciprocal.inp`
   handoff through the dedicated validation-only `kmesh` scheduler report,
   comparing generated `kmesh.dat` against FEFF's archived reference without
   advertising completed `bandstructure.dat` output. BAND source setup now
   consumes optional `global.inp` `ispin` and feeds the one-spin spin-orbit
   T-matrix branch instead of hard-coding the default selector.
   Full-run scheduler discovery now also treats malformed declared `global.inp`
   spin-selector handoffs as invalid source state, so those files cannot be
   ignored as optional or reported as completed/validation-only BAND stages.
   BAND source-handoff discovery now also declines malformed declared
   `phase.bin` bundles instead of reporting `band` or `band-handoff`; explicit
   BAND execution still surfaces the phase parser/setup error, and readable
   cached `bandstructure.dat`/`kmesh.dat` output no longer masks a malformed
   declared `phase.bin` source.
   Malformed declared `reciprocal.inp` handoffs now follow the same direct-module
   and full-run scheduler rule, so a bad reciprocal source file cannot advertise
   a completed `band` stage or validation-only `band-handoff`; readable cached
   `bandstructure.dat`/`kmesh.dat` output no longer masks that malformed
   declared source.
   Malformed declared `global.inp` spin-selector and `fms.inp` `lmaxph`
   handoffs now follow that same cached-output rule.
   The explicit BAND unported fallback is now
   retired: missing or incomplete source state reports a normal source
   requirement after deterministic pre-solver validation, while complete
   supported source bundles write `bandstructure.dat`. One-spin relativistic
   KKR/`freeprop` production dispatch now uses nonzero `global.inp` `ispin`
   source selectors and is covered by module and full-run scheduler
   regressions. Full-run scheduling now also covers non-degenerate two-spin
   ordinary and `freeprop` source bundles, so spin-resolved multi-spin dispatch
   reaches completed `bandstructure.dat` output instead of a validation-only
   handoff. Direct BAND module stale-cache regressions now also cover one-spin
   relativistic ordinary and `freeprop` source bundles, and matching full-run
   scheduler regressions now pin readable-stale one-spin relativistic repair
   from `global.inp` `ispin = 1` source handoffs. Remaining BAND work is broader
   branch coverage for generated `bandstructure.dat`, not a module-level
   unported gate.

   Acceptance:
   - `bandstructure.dat` is written from source-backed eigenvalue rows
   - `kmesh.dat` remains a sidecar, not the completion signal
   - malformed final `bandstructure.dat` with valid sources is regenerated
   - same-shape stale final `bandstructure.dat` band values with valid sources
     are regenerated

5. Wire SCREEN and CRPA response assembly.

   Existing adapters can emit `wscrn.dat` and `crpa.dat` once response kernels,
   response slices, radial densities, and core-hole components are available.
   CRPA now reuses the SCREEN source response components to assemble selected
   `den_CRPA`, integrated `totden_CRPA`, occupied response slices, and paired
   `crpa.dat`/`wscrn.dat` output when the full source bundle is present. The
   default source-reference gate now extracts the CRPA reference zip without
   cached `crpa.dat`, `wscrn.dat`, `phase.bin`, or `gg.bin`; the generated
   Hubbard summary is within `1e-5`, and the CRPA-relevant `wscrn.dat`
   radius/screened-potential columns are within `1e-5` of FEFF reference.
   When that complete source bundle is present, readable cached `crpa.dat` and
   `wscrn.dat` are now render-normalized against the generated source payload
   before CRPA is advertised as cache-complete, so stale CRPA output routes back
   through the source response assembly.
   Direct CRPA execution and full-run supported-stage scheduling now recognize
   complete source bundles as `crpa` output, and full-run scheduling now carries
   the CRPA reference zip through source-generated `crpa.dat`/`wscrn.dat` row
   parity without cached `phase.bin`/`gg.bin`. The pre-solver `wscrn.dat` recovery
   path remains for incomplete source state, so recoverable
   `vtot.dat`/`apot.bin` sidecars are still written before the source
   requirement when no cached `crpa.dat` or complete source bundle exists.
   The core SCREEN response path now has the contour response-slice assembly
   that adds FMS cluster corrections to each atomic angular-channel slice, sums
   angular channels, and emits the full `chi0re(:,:,ie)` cube before the
   contour integral. The `refeff-io` SCREEN handoff layer now derives the FMS
   `gtrl(energy,l)` trace table from `phase.bin` absorber phase shifts and
   `gg.bin` scattering sections, with SCREEN's `lmaxp1^2` matrix-order guard.
   Direct SCREEN runs validate complete phase/FMS trace bundles through that
   adapter before the remaining source requirement. They also validate the
   `screen.inp`/`pot.bin` potential-kernel handoff, including radial bounds,
   RPA/TDLDA local-field kernels, Coulomb response kernels, and bound core
   components. The `refeff-io` response-assembly handoff now consumes
   regular/irregular radial solution cubes, FMS traces, and that
   potential/kernel state, then emits `wscrn.dat` data through the Rust
   response-slice, contour-integration, and core-hole solve kernels. The core
   SCREEN radial channel helper now applies FEFF `xfnorm`, Wronskian irregular
   scaling, and exact free-particle tail replacement to raw regular/irregular
   `dfovrg` outputs, then lifts that channel assembly over the
   `(energy, radial, l)` cube layout needed by response assembly. Exact-tail
   generation now evaluates the ported FEFF Bessel/Neumann/Hankel helpers at
   `ck*r` for each tail row, and the one-channel SCREEN wrapper now runs the
   prepared regular FOVRG pass, computes and injects the irregular muffin-tin
   boundary condition, runs the irregular FOVRG pass, and feeds both raw
   solutions through the same exact-tail assembly. A contour-grid wrapper now
   loops those prepared FOVRG channel inputs in FEFF `(energy,l)` order and
   packs response-ready `(energy, radial, l)` radial cubes; its matched variant
   recovers `phamp` phase shifts and amplitudes from the regular FOVRG pass
   instead of requiring an external phase-amplitude table. The `refeff-io`
   SCREEN radial handoff now prepares absorber FOVRG solver grids from
   `pot.bin`/`config.dat` plus the SCREEN energy/reference state, runs that
   matched cube helper, and returns response-ready radial cubes and recovered
   `phamp` phase tables. The direct SCREEN module path now composes the source
   FMS, potential-kernel, FOVRG radial, and inline SCREEN/FMS handoffs and
   writes `wscrn.dat` plus `vtot.dat` without cached screened-potential output.
   The inline FMS source-grid bridge now solves the spin-free SCREEN trace
   table from recovered `getph`/`phamp` phase shifts without `gg.bin`, with the
   cached `gg.bin` adapter retained as a fallback. Non-absorber phase-grid
   entries now use a phase-only FOVRG handoff that avoids unused irregular
   radial solves. The typed `pot.bin` and `config.dat` readers now normalize
   older FEFF text/PAD reference caches that store 30/29 orbital records into
   the current FEFF10 41/40-slot internal shapes, including legacy eight-slot
   `iorb` records, so archived Cu source handoffs reach the SCREEN numerical
   path instead of failing at parser shape checks. The first Cu source-runtime
   fixes are also in place: single-precision complex LU now factors and solves
   through flat buffers, and FMS `g0` assembly validates table shapes outside
   the hot state-pair loop while preserving FEFF Fortran-order output. The
   SCREEN inline FMS bridge now packs only the absorber scattering block it
   consumes, and the `g0` fast path caches the triangular normalization/weight
   table used by `xgllm` outside the state-pair loop. The source SCREEN driver
   now follows `prep.f90` by building the contour from `screen.inp` and `xmu`
   instead of reusing the XSPH `phase.bin` mesh, and the FOVRG handoff uses the
   prepared absorber `eref(1)` reference potential for every contour point. The
   default
   `screen_module_matches_no_cache_inline_fms_generated_reference_when_present`
   parity gate now runs six Cu-family source bundles (`DANES/Cu`, `ELNES/Cu`,
   three LDOS Cu fixtures, and `XANES/Cu`) without cached `wscrn.dat`,
   `vtot.dat`, or `gg.bin`; the generated `wscrn.dat` screened-potential column
   is within `4.7e-6` max absolute difference of the FEFF references, and the
   refreshed bare core-hole column matches the `apot.bin` handoff. The default
   `screen_module_matches_graphite_reference_zip_without_phase_or_gg_cache`
   parity gate adds a non-Cu KSPACE/Graphite archive fixture, exercising
   legacy FMS inputs that omit `save_gg_slice`/`do_fms`, lower
   `fms.inp lmaxph` than `screen.inp maxl`, and no cached `phase.bin`/`gg.bin`;
   generated `wscrn.dat` and `vtot.dat` match the FEFF reference within
   `1e-4`. Full-run supported-module orchestration now also covers the
   XANES/Cu inline-FMS source bundle as a completed `screen` report, verifying
   scheduler-generated `wscrn.dat` and `vtot.dat` radial rows against the FEFF
   reference without cached `gg.bin`. SCREEN now reports a normal
   source-requirement error for missing or incomplete inputs instead of an
   explicit unported fallback.

   Acceptance:
   - SCREEN writes `wscrn.dat`/`vtot.dat` without cached `wscrn.dat`
   - CRPA writes `crpa.dat` and `wscrn.dat` without cached `crpa.dat`
   - shared `wscrn.dat` recovery remains compatible with XSPH/RIXS

6. Wire LDOS density tables.

   Existing `ff2rho`/`ff2rho_h` adapters can render final tables once the FMS
   trace and embedded/scattering LDOS arrays are available.
   `gtrNN.bin` traces can now be projected into LDOS `cchi(l,ie)` layout for a
   selected potential, so the remaining trace work is generation/assembly rather
   than binary orientation. The non-full-potential `fmsdos` trace primitive is
   also ported in core: packed FMS `gg` diagonals are summed over magnetic
   channels and phase-normalized for each `(l, potential)` row, with an
   energy-grid adapter that emits `(energy, potential, angular)` values aligned
   with the `gtrNN.bin` handoff, and the IO codec can now package that grid
   with the FEFF header metadata. The LDOS runner now writes non-spin
   `gtrNN.bin` files from supported source FMS handoffs before the final-table
   source requirement is evaluated. Core now also ports the
   post-radial-solver `LDOS/rhol.f90` density integrals for `xrhole` and
   `xrhoce`, including an energy-grid adapter that emits the `(angular,
   energy)` work arrays consumed by `ff2rho`; an LDOS-facing exact-tail adapter
   now also covers the shared Bessel/Neumann continuation for rows `jri:ilast`.
   A `rhol` radial assembly adapter now normalizes raw regular/irregular
   `dfovrg` outputs, applies the Wronskian irregular replacement, and overwrites
   the exact tail without RHORRP's smoothing branch. A one-channel `rhol`
   wrapper now invokes the shared FOVRG `dfovrg` driver for the regular and
   irregular passes, then feeds the result through the LDOS radial assembly. A
   non-spin source-backed `rhol` table driver now loops prepared per-channel
   FOVRG inputs in `(energy, l)` order, evaluates `xrhole`/`xrhoce`, and feeds
   the Rust `ff2rho` adapter for final table payloads. LDOS can now also
   consume shared source-backed RHORRP wavefunction tables directly: selecting
   one potential from `(energy, l, radial, iph)` `prel`/`pnel`/`qrel`/`qnel`
   arrays, evaluating the LDOS density integrals, and feeding the same
   `ff2rho` adapter. The CLI runner now wires those bridges to source handoff
   files: when compatible non-spin `pot.bin`/`config.dat`/`phase.bin`/
   `pot.inp`/`fms.inp` radial handoffs and matching `gtrNN.bin` traces are
   present, it writes final `ldosNN.dat`/`rhocNN.dat` tables before the source
   requirement. The no-FMS source path now needs only the radial handoff files
   (`pot.bin`/`config.dat`/`phase.bin`/`pot.inp`), prepares the shared radial
   source once, drives the LDOS-specific `rhol` FOVRG table driver on the
   `ldos.inp` mesh for the missing available potentials without requiring
   `gtrNN.bin`, supplies a zero scattering trace, and produces matching
   `ldosNN.dat`/`rhocNN.dat` densities. It has a production EXAFS/Cu
   source-generation smoke gate for the real 101-point mesh, and generated
   tables now preserve FEFF header metadata from `pot.bin`/`fms.inp`/`geom.dat`
   (`xmu`, `qnrm`, `xnmues`, `inclus`, and broadening) under the source
   reference gates. It has generated XANES/Cu and NRIXS/GeCl4 source-handoff
   parity fixtures using production no-FMS LDOS cards, an ordinary-spin
   XANES/Cu no-FMS source parity fixture that preserves FEFF's regular
   four-column output shape, a short NRIXS/GeCl4 no-FMS source parity fixture
   that exercises FEFF-valid valence-only `config.dat` orbitals, and a
   regression that bounds generation to the source potential count. Complete
   no-FMS radial source handoffs now also check readable
   `ldosNN.dat`/`rhocNN.dat` pairs against the source-generated energy and
   density grids, so same-shape stale final tables regenerate from source
   instead of masking the Rust `rhol` driver. Full-run orchestration now also
   has a XANES/Cu no-FMS source parity gate that compares scheduler-generated
   `ldosNN.dat`/`rhocNN.dat` energy and density grids against the FEFF
   reference tables before advertising a completed `ldos` stage.
   `full_run_scheduler_regenerates_stale_xanes_cu_no_fms_ldos_tables_from_source_handoffs`
   now exercises that same production no-FMS source bundle after readable
   `ldos00.dat`/`rhoc00.dat` values have gone stale, requiring source-backed
   regeneration before the completed `ldos` report is accepted. Full-run
   orchestration now advertises complete no-FMS radial source handoffs and
   supported FMS source-grid handoffs as a completed `ldos` supported stage
   before the source requirement. Those source-handoff discovery probes now
   decline malformed declared `pot.bin`/`config.dat`/`phase.bin`/`pot.inp`
   bundles, while explicit LDOS execution still reports the underlying typed
   reader/parser error. Those no-FMS radial profiles now match the FEFF golden tables closely after using
   FEFF's `csomm2` row count, direct LDOS-card
   energies, and `rhol`
   normalization. The absorber FMS source path now also covers FEFF's
   real-space zero-cluster branch: compatible RHORRP wavefunction handoffs
   regenerate zero `gtrNN.bin` files on the LDOS-card grid, and an EXAFS/Cu
   reference regression checks that the resulting `ldosNN.dat` and
   `rhocNN.dat` tables match the FEFF golden no-scattering tables. Full-FMS
   table assembly now matches FEFF `ff2rho` by preferring each potential's
   matching `gtrNN.bin` file and selecting that potential trace from the file.
   The FMS source-grid CLI gate now also exercises non-default `minv` solver
   selectors (`1`, `2`, `3`, and FEFF's fallback path) through source-generated
   `gg.dat`/`gg.bin`, `fms.bin`, and `gtr.dat` outputs. Orphan cached FMS
   artifacts such as `gtrNN.bin` no longer make full-run orchestration claim
   the FMS stage when `fms.inp` is absent, with full-run scheduler coverage
   pinning that non-claiming boundary.
   Active-Hubbard FMS source generation now also covers `save_gg_slice` by
   back-transforming the full full-potential LU matrix and writing source-backed
   `gg_slice.bin`/`gg_diag.bin` sidecars; the release gate checks that the saved
   absorber blocks reproduce the generated `gg.dat` matrix.
   The `gtrNN.bin` codec now roundtrips FEFF's default single-precision complex
   payload, and the source-grid FMS path writes FEFF-shaped per-potential
   `gtrNN.bin` data with only the central potential column populated in each
   file. The phase-grid `gtrNN.bin` handoff writer now also compares readable
   caches against generated source output before preserving them, so stale
   phase-grid traces are regenerated when the LDOS card mesh matches
   `phase.bin`. A generated short XANES/Cu nonzero full-cluster FMS reference
   now checks `gtr00.bin`/`gtr01.bin` and the resulting
   `ldosNN.dat`/`rhocNN.dat` tables. A generated 101-point XANES/Cu
   production full-FMS release gate now checks the same `gtrNN.bin` and
   final-table parity as normal default
   coverage. Complete FMS wavefunction source handoffs now also compare
   readable final `ldosNN.dat`/`rhocNN.dat` pairs against the source-generated
   tables and rewrite stale same-shape caches before they can mask the
   source-grid `gtrNN.bin` and `rhol` driver. LDOS cache discovery now uses
   that same source-rendered comparison before accepting readable FMS final
   tables, so stale `ldosNN.dat`/`rhocNN.dat` pairs do not satisfy the cache
   predicate when source handoffs can render the expected output. Nonmagnetic
   ordinary-spin FMS now reuses the same source-backed FMS grid after verifying
   zero `xsph.inp`
   `spinph` values,
   preserving FEFF's regular four-column LDOS/RHOC output shape, with a short
   XANES/Cu reference parity gate. The explicit LDOS unported fallback is now
   retired: missing or incomplete table/radial/FMS source state reports a
   normal source-requirement error, while complete supported source bundles
   write final `ldosNN.dat`/`rhocNN.dat` tables. The Hubbard LDOS trace
   sidecars now have typed Rust codecs for spin-resolved `gtrNN.bin`,
   magnetic-diagonal `gtr_mNN.bin`, and off-diagonal `gtr_offNN.bin` payloads,
   with NiO reference-zip byte roundtrips pinning FEFF's header and implied-DO
   order. The paired magnetic-orbital text sidecars, `lmdosNN.dat` and
   `rhocmNN.dat`, now also have a typed variable-`lx` parser/renderer with NiO
   reference coverage. The CLI LDOS cache path now also has an active-Hubbard
   NiO reference-zip gate that preserves all three potentials'
   `ldosNN.dat`/`rhocNN.dat` plus `lmdosNN.dat`/`rhocmNN.dat` sidecars,
   including FEFF's legacy wrapped `hubbard.inp`, six-field `ldos.inp`, and
   truncated six-column spin LDOS/RHOC text shapes. Active-Hubbard LDOS cache
   completion now requires paired ordinary `ldosNN.dat`/`rhocNN.dat` tables
   plus paired `lmdosNN.dat`/`rhocmNN.dat` sidecars for every cached potential,
   so a partial ordinary cache no longer masks the remaining spin-Hubbard
   source-generation boundary. The ordinary pair must share the same energy
   grid and density-column layout, and the magnetic sidecars must also match
   that ordinary grid and each other's magnetic `lx`/density layout, so stale
   active-Hubbard ordinary or magnetic LDOS/RHOC sidecars no longer complete
   the cached stage. Non-Hubbard LDOS runs still ignore stray magnetic sidecar
   files at both direct-module and full-run scheduler boundaries, so malformed
   `lmdosNN.dat` or `rhocmNN.dat` files cannot turn an ordinary LDOS cache into
   an active-Hubbard requirement. When a valid `gtr_mNN.bin` magnetic trace
   source is present, the cache gate also requires the sidecars' energy count
   and magnetic layout to match that source contract. Malformed ordinary
   `ldosNN.dat`/`rhocNN.dat` pair members are treated as incomplete
   active-Hubbard caches, letting the existing no-FMS regeneration path repair
   them from the valid counterpart before the magnetic sidecar contract is
   evaluated. Full-run supported-stage discovery now mirrors that direct-run
   repair rule, so a malformed ordinary
   `ldosNN.dat` beside a valid `rhocNN.dat` and magnetic sidecars is reported
   as a completed `ldos` stage only after the Rust runner regenerates the
   ordinary half and `logdos.dat`. Standalone no-FMS spin-resolved
   `ldosNN.dat`/`rhocNN.dat` cache pairs now have the same scheduler-level
   repair coverage in both directions, while full-run ordinary-spin source
   handoffs continue to prefer FEFF's regular four-column source-generated
   tables when RDINP has produced the radial handoff bundle. When a valid
   spin-resolved Hubbard
   `gtrNN.bin` trace source is present, the ordinary
   `ldosNN.dat`/`rhocNN.dat` pair must also match that source's energy count and
   spin-density column layout. When both `gtrNN.bin` and `gtr_mNN.bin` source
   traces are valid, their energy count and angular layout must also agree
   before the active-Hubbard cache is accepted. Valid `gtr_offNN.bin`
   off-diagonal source traces are also checked against the magnetic sidecars and
   any ordinary/magnetic trace contracts for matching energy count and angular
   layout before cached active-Hubbard completion is accepted. Readable Hubbard
   trace sources that do not contain the cached potential index are now treated
   as incompatible contracts rather than absent optional sources, so truncated
   per-potential trace bundles cannot bless stale final tables. The full-run
   scheduler now also has active-Hubbard coverage for this boundary: an
   `ldos00` cache with stale ordinary `ldos00.dat` or `rhoc00.dat` energy
   grids or a stale ordinary density-column layout is not reported complete,
   an `ldos01` cache plus complete magnetic sidecars is not reported complete
   when readable `gtr01.bin`, `gtr_m01.bin`, or `gtr_off01.bin` omits
   potential 1,
   and an `ldos00` active-Hubbard cache is likewise rejected when `gtr00.bin`
   advertises an ordinary layout that conflicts with `ldos00.dat`/`rhoc00.dat`,
   when `gtr00.bin` and `gtr_m00.bin` advertise incompatible angular layouts,
   when `gtr_m00.bin` advertises a conflicting magnetic layout, or when
   `gtr_off00.bin` advertises a stale off-diagonal energy/angular layout. The
   matching scheduler boundary now also rejects stale or malformed
   active-Hubbard magnetic text sidecars directly, including shifted
   `lmdos00.dat` and `rhocm00.dat` energy grids plus a stale `rhocm00.dat`
   magnetic layout or malformed `lmdos00.dat`/`rhocm00.dat` text. The matching
   positive direct-module and scheduler gates now accept a complete `ldos00`
   active-Hubbard cache when ordinary `gtr00.bin`, magnetic `gtr_m00.bin`, and
   off-diagonal `gtr_off00.bin` source contracts all agree with the ordinary
   and magnetic final tables, and the direct-module gate now also accepts a
   nonzero active-Hubbard cached potential when fallback
   `gtr00.bin`/`gtr_m00.bin`/`gtr_off00.bin` source bundles contain that
   potential and agree with the tables. Both paths re-render the `logdos.dat`
   wrapper through the supported `ldos` report.
   Rust now also
   ports the
   FEFF `LDOS/ff2rho_h_step2.f90` magnetic table assembly that writes
   `rhocmNN.dat` from embedded `xmrhoce`
   and `lmdosNN.dat` from `xmrhoce/(2*l+1) + imag(gtr_m*xmrhole)`, with an IO
   adapter that builds renderable `lmdos`/`rhocm` payloads from those source
   work arrays. The `gtr_mNN.bin` codec now also selects one potential into the
   adapter's `(l, magnetic, spin, energy)` trace layout, and the adapter has a
   release-profile compatibility-matrix gate. No-FMS active-Hubbard LDOS now
   uses that same adapter with zero scattering to repair one-sided
   `lmdosNN.dat`/`rhocmNN.dat` magnetic sidecars from the paired magnetic table.
   Spin-resolved Hubbard source generation and full-potential LDOS branches
   remain parity follow-up work. The
   phase-grid-only source `gtrNN.bin`
   path is still guarded so it only runs when the LDOS card mesh matches
   `phase.bin`, while the RHORRP-backed path
   regenerates `gtrNN.bin` from LDOS-card-grid RHORRP wave numbers and phase
   shifts before final table assembly, so stale phase-grid FMS traces no
   longer satisfy a mismatched LDOS mesh.

   Acceptance:
   - non-spin and spin-resolved `ldosNN.dat`/`rhocNN.dat` are source-generated
   - no-FMS recovery remains a supported narrow path
   - full FMS LDOS stops relying on cached final tables

7. RIXS source-backed gate is retired.

   RIXS already writes standard and satellite outputs from complete source
   bundles through module and full-run orchestration. Full-run complete source
   bundles are reported as completed `rixs` stages, while incomplete bundles
   remain validation-only `rixs-handoff` stages. `ReadSigma` complete bundles
   can now use an in-memory XSPH-generated MPSE table when `mpse.dat` is absent,
   and cached-`mpse.dat` ReadSigma now has an MPSE/Cu reference-zip gate that
   checks FEFF's table through the production CLI self-energy handoff path.
   When both a readable cache and compatible XSPH MPSE source handoff are
   present, RIXS compares the energy/self-energy columns and uses the source
   table if the cache is stale. The direct RIXS module path also uses the
   generated XSPH MPSE table when a malformed `mpse.dat` cache is paired with a
   complete compatible source handoff. Full-run scheduler coverage now pins that
   same MPSE source-preference branch by comparing source-only,
   stale-cache-plus-source, malformed-cache-plus-source, and stale-cache-only
   ReadSigma runs without scheduling a standalone XSPH repair stage. Shared
   `wscrn.dat` screened-core handoffs now follow the same
   source-backed repair rule when `vtot.dat`/`apot.bin` can regenerate the
   shared table and explicit edge handoffs are absent.
   A valid but partial RIXS cache no longer masks a complete source bundle:
   when regular or requested satellite final outputs are incomplete, the
   source-backed bundle opportunistically fills the full map/line set, while
   malformed or incomplete source handoffs still fall back to an otherwise
   usable partial cache. The satellite variant is pinned by
   `rixs_module_writes_satellite_source_outputs_when_partial_cache_exists`, so
   an existing regular HERFD cache cannot prevent requested MBConv satellite
   outputs from being generated from source.
   Malformed declared RIXS solver handoffs such as a bad `global.inp` now
   decline supported-stage discovery instead of reporting `rixs` or
   `rixs-handoff`; explicit RIXS runs still surface the parser/setup error.
   Full-run scheduler coverage now also pins explicit edge handoff precedence:
   malformed shared `phase.bin`, `rl.dat`, `wscrn.dat`, `gg.bin`, and
   `xsect.dat` sidecars do not block complete edge-specific source bundles
   from reporting a completed `rixs` stage and writing final spectra.
   The explicit unported fallback is retired: missing or incomplete source
   state now reports a normal source-requirement error, and complete source
   bundles write standard and satellite final spectra. FF2X now also has a
   source-backed GeCl4 NRIXS `xmul.dat` generation gate that applies corrected
   `xscorr` diagonal channel backgrounds and tolerates unused `xsecl.bin`
   transition channels above `ldecmx`. Remaining RIXS work is parity broadening
   for MPSE branches and additional NRIXS branch coverage, not a module-level
   unported gate.

   The lower-level SELF many-pole branch now includes source-backed BPR
   `UseBP = .TRUE.` support: Rust ports of FEFF `bpr1`, `bpr2`, and `bpr3`
   feed the `Sigma1`/`CSigZ` dispatcher, with FEFF-backed direct-integrand and
   full many-pole reference tests. The typed `xcpot` MPSE self-energy path now
   forwards an explicit `UseBP` selector into that dispatcher, while FEFF's
   ordinary XSPH/EXCH MPSE route remains at its source hard-coded non-BPR
   default. Remaining MPSE production work is broadening generated reference
   coverage where FEFF actually requests BPR.

   Acceptance:
   - complete source bundles always write final RIXS outputs, even when a
     valid partial cache is present
   - incomplete bundles continue to fail loudly
   - `SkipCalc` cache-derived transforms remain supported but do not mask
     missing source solver state

## Test Cadence

Do not run the full workspace test suite for every slice.

For ordinary implementation slices:

- `cargo fmt --all --check`
- one or more focused `cargo test -p <crate> <filter> -- --nocapture`
- `cargo run -p xtask -- port-status --detail` when a gate or source-handoff
  status changes; the detail view also mirrors the current branch
  compatibility blockers so module support cannot be mistaken for full FEFF10
  parity
- `cargo run --profile release -p xtask -- port-status --detail --json-out target/port-status.json`
  when saving the module inventory, source-handoff markers, guarded branches,
  ignored parity checks, and next module-level actions for CI artifacts or
  review notes
- `cargo run -p xtask -- compatibility-matrix --open-only --detail` when
  auditing the remaining branch-level blockers after adding evidence rows; the
  detail view prints each open row's `next:` implementation or parity task and
  `verify:` release-profile closure gate, plus local fixture status for
  reference-backed covered rows and open-row blocker prerequisites
- `cargo run -p xtask -- compatibility-matrix --module <name> --open-only --detail`
  when auditing a targeted module's remaining branch-level blockers
- `cargo run --profile release -p xtask -- compatibility-matrix --row <id> --fail-on-open`
  when proving that one compatibility row from the open-row list has closed
  without waiting for every other row in the same module
- `cargo run --profile release -p xtask -- compatibility-matrix --fail-on-missing-fixtures`
  when proving covered reference-backed rows and open-row blocker prerequisites
  have their required local fixture artifact groups and will not silently skip
  their reference checks
- `cargo run --profile release -p xtask -- compatibility-matrix --open-only --detail --json-out target/compatibility-matrix.json`
  when saving the selected matrix rows, open row IDs, per-row fixture counts,
  and fixture-audit results for CI artifacts or review notes
- `cargo run --profile release -p xtask -- release-readiness --detail --open-only`
  when checking the final pure-Rust FEFF release claim; it composes strict
  module support with strict branch-level compatibility and fixture presence,
  and fails until every open row closes. Use `--module <name>` or `--row <id>`
  on the same command for targeted readiness audits. Add `--json-out <path>`
  for a combined readiness summary with each selected open blocker's fixture
  prerequisite count and missing fixture groups, and add
  `--port-json-out <path>` plus `--compatibility-json-out <path>` when the
  module-status and compatibility artifacts should also be retained from a
  failing release-readiness run
- `cargo run -p xtask -- port-status --fail-on-guarded-branches --fail-on-ignored-parity`
  when promoting guarded branch or parity checks into default coverage
- `git diff --check -- <touched files>`

Run broader tests when:

- a shared parser/renderer changes
- `refeff run` orchestration changes
- a module gate is removed
- a release branch is being prepared

Release-gate verification should include:

- `cargo test --workspace`
- generated-reference tests for affected fixtures
- `cargo run --profile release -p xtask -- release-readiness --detail --open-only --json-out target/release-readiness.json --port-json-out target/release-readiness-port-status.json --compatibility-json-out target/release-readiness-compatibility.json`

## Follow-Up Coverage Backlog

These items are not current unported module gates or guarded release blockers.
Use this order for post-completion parity-broadening slices unless a blocker
forces a different path:

1. ATOMIC/POT: broaden source-backed potential generation across remaining
   SCF convergence, retry/exhaustion, exchange, and finite-nucleus parity
   branches.
2. XSPH: broaden the remaining phase-shift branches beyond the source-backed
   `XANES/BN` zip fixture. Legacy GeCl4/XMCD `phase.bin` headers now parse,
   old no-`config.dat` GeCl4 source parity now runs from `pot.bin`, and the XES
   BN/GeCl4 plus NRIXS/GeCl4 zip references, MnF2 XMCD source-generated
   `ltot` capacity/`xsect.dat` parity, and Gd L1 XMCD fine-radial-grid
   capacity/`xsect.dat` parity are covered; NRIXS/MgB2 numerical parity, XMCD
   phase numeric parity, and reference parity for the source-backed
   `TDLDA/xsectd.f90` TDLDA/PMBSE cross-section driver remain open.
3. BAND: broaden generated-output parity for source-backed ordinary,
   `freeprop`, and relativistic `bandstructure.dat` branches.
4. LDOS: broaden source-generated final tables into spin-Hubbard/full-potential
   parity and keep full-FMS reference coverage on an optimized release-gate
   cadence.

Keep each follow-up slice small enough that its tests can run in focused form.
