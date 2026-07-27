# FEFF10 Porting Plan

This document tracks the active goal for `refeff`: complete the FEFF10 to Rust
port while keeping the local `feff10/` checkout as a reference only. The
reference source and generated reference work products must not be added to the
Git repository.

## Acceptance Criteria

A module is considered ported only when all of the following are true:

- The Rust implementation performs the module's production calculation without
  relying on cached FEFF output.
- Input and output files preserve FEFF-compatible names, ordering, formatting,
  units, and default behavior.
- Reference-backed tests compare Rust output against FEFF10 output generated
  from the local reference tree, with documented floating-point tolerances.
- Unit tests cover reusable numerical kernels, validation paths, and error
  handling.
- Public APIs and module entry points have doc comments explaining the FEFF
  routine or stage being ported, expected inputs and outputs, and compatibility
  limits.
- Production code uses safe Rust, returns typed errors or `anyhow::Result` at
  application boundaries, and does not use `unsafe`, `panic!`, `unwrap`, or
  `expect`.
- The pre-commit hook passes: formatting, whitespace checks, workspace checks,
  tests, documentation, and clippy with warnings denied.
- The exact workspace lint gate is `cargo clippy --workspace --all-targets
  --all-features --locked -- -D warnings`.
- Benchmarks cover the module's expensive kernels or end-to-end workflow before
  performance tuning is claimed complete.

## Implementation Constraints

- Target Rust version: 1.95.
- Use `ndarray` as the primary array representation.
- Use `faer` only for pure-Rust linear algebra paths where it is a clear fit.
- Keep implementation structure close to FEFF module boundaries, but split long
  files into focused submodules as they grow.
- Prefer simple, explicit functional-style transformations when they improve
  clarity and optimization opportunities. Do not add abstractions that obscure
  the FEFF algorithm.

## Current Port Status

`cargo run -p xtask -- port-status --detail` reports module support separately
from branch-level release readiness. The module inventory, source-handoff
coverage, guarded-branch audit, ignored release-gate audit, and current branch
compatibility blockers are currently visible from the same detail view.
Use `cargo run -p xtask -- port-status --fail-on-unported --fail-on-guarded-branches --fail-on-ignored-parity`
as the final module-support, guarded-branch, and ignored-gate audit; it should
pass before the Rust port is treated as release-complete. Add
`--json-out <path>` to keep the module inventory and module-level blocker state
as a machine-readable release artifact.
Use `cargo run -p xtask -- compatibility-matrix --detail` for the stricter
branch-level FEFF10 audit. Add `--open-only` to focus the table on remaining
blocking rows, add `--module <name>` for a targeted module audit, add
`--row <id>` for a single compatibility-row closure check, and use
`--fail-on-open` as the release gate for the selected rows before declaring the
pure-Rust workflow complete rather than merely module-supported. Add
`--fail-on-missing-fixtures` to also fail when reference-backed covered rows
or open-row blocker prerequisites are missing required local fixture artifact
groups. Detail mode prints `next:` and `verify:` lines for each open row and
fixture status for reference-backed covered rows and blocker prerequisites, so
the release gate remains an actionable implementation checklist. Add
`--json-out <path>` to persist the selected rows, displayed row state, open row
IDs, per-row fixture counts, and missing fixture groups before the strict gate
exits.
Use `cargo run --profile release -p xtask -- release-readiness --detail --open-only`
as the single final readiness gate. It composes strict `port-status` with
`compatibility-matrix --fail-on-open --fail-on-missing-fixtures`, so a clean
module inventory cannot be mistaken for complete FEFF10 branch parity while
compatibility rows remain open or required local reference fixtures are
missing. The readiness gate accepts the same `--module` and `--row` filters for
targeted audits. Add `--json-out <path>` for a combined readiness summary, and
add `--port-json-out <path>` plus `--compatibility-json-out <path>` when the
module-status and compatibility artifacts should also be retained from a
failing run. The combined readiness summary includes each selected open
blocker's fixture prerequisite count and missing fixture groups.
`cargo test --profile release -p refeff-engine full_run_completes_minimal_cu_smoke_input`
now pins the release full-run smoke path through `phase.bin`, `xsect.dat`,
`chi.dat`, and `xmu.dat`.

`cargo run --profile release -p xtask -- scope-audit --detail` currently
audits 22 FEFF production executables, 3 Rust extensions, 110 card-token IDs,
44 stock workflows, and the 138-case HIGHZ range. Separately, the module-status
inventory has 22 entries: 21 scheduler workflow stages with source handoffs
plus `dym2feffinp`, which consumes a `.dym` file directly and therefore does
not have a scheduler source handoff.

There are no ignored release gates in the current inventory. The SCREEN inline
source-FMS gate, SCREEN Graphite source gate, LDOS production full-FMS gate,
XSPH broader source parity gate, CRPA source-reference gate, and SO2CONV
spectral-function generation gate now run as default coverage. Those passing
gates are coverage evidence for the current fixtures, not a claim that every
FEFF branch is covered.
The XSPH broader source parity and scheduler reference phase/xsect fixtures are
also release-profile compatibility evidence through
`cargo test --profile release -p refeff-engine xsph_module_matches_broader_source_generated_reference_when_present`
and
`cargo test --profile release -p refeff-engine xsph_reference_phase_and_xsect_from_source_handoffs`.
The broader source gate now also accepts legacy FEFF `xsph.inp` handoff shapes
and includes the zip-backed `XANES/BN` and old no-`config.dat` `XANES/GeCl_4`
source fixtures, plus the `XES/BN`, old no-`config.dat` `XES/GeCl_4`,
`NRIXS/GeCl_4`, MnF2 XMCD `ltot` capacity/`xsect.dat` parity, and Gd L1 XMCD
fine-radial-grid capacity/`xsect.dat` parity source fixtures. Legacy 8- and
10-column `phase.bin` headers now parse, and normal XSPH can derive
conservative orbital tables from `pot.bin` when older archives omit
`config.dat`; current pinned NRIXS/MgB2 and XMCD phase/xsect outputs are now
covered, with other-material cases left as non-blocking fixture broadening.
XSPH NRIXS/JAS `xsectjas` production now validates readable
`xsecl.dat`/`xsecl2.dat`/`xsecl.bin` caches and has a q-resolved,
source-backed one-spin normal-potential writer that produces `xsect.dat`, the
`xsecl*` sidecars, and matching `phase.bin` transition moments. Broader spin
normalization and additional branch parity remain follow-up coverage work, not
a current guarded release blocker.

The detailed implementation plan lives in
[`docs/FEFF_RUST_PORT_PLAN.md`](FEFF_RUST_PORT_PLAN.md). It records the active
module inventory, source-backed acceptance gates, follow-up parity backlog, and
focused-test cadence.
Full-run scheduler predicates now also decline orphan final artifacts when the
matching module input is absent for POT, XSPH, BAND, SCREEN, CRPA, LDOS, DMDW,
FMS, EELS, EELS-MDFF, PATHS, GENFMT, FF2X, SFCONV/SELF, FULLSPECTRUM, and
COMPTON, OPCONS, RHORRP, and RIXS, so partial caches no longer claim completed
stages by themselves.

Recent ATOM core progress now covers the composed `soldir.f90` Dirac orbital
driver, the `wfirdf.f90` initial-orbital/radial-potential initializer, and one
source-backed positive-`niter` `scfdat.f90` scheduler
(`lagdat -> potrdf -> vlda -> soldir -> cofcon -> dsordf`, plus final density
recomputation and returned `srho/r**2`/`vcoul` tables). The ordinary `atomic`
module gate is now source-backed for `pot.inp` plus `geom.dat`, including the
finite-nucleus APOT stream. Release-profile gates now pin finite nuclear mesh
selection through direct APOT source generation, starting radii, nuclear
potential, density, and component differences, FEFF `nucdev` point/finite
nuclear-potential behavior, and the composed atomic SCF state driver's
finite-nucleus request path. Full-range validation covers the nuclear data and
kernel contract for Z=1 through Z=138; production/reference checks compare
Z=4, 29, 79, and 92 and preserve the typed upstream Z=119 failure. The 139-row
configuration table, including the Z+1 sentinel, is validated separately. The
release gate makes no production completion claim for the pinned report's
Z=118 failure or for Z=138. Additional full-SCF generated references remain
non-blocking parity broadening.

The strict compatibility matrix is closed at 98/98 rows. POT
retry/exhaustion and final output, XSPH TDLDA/PMBSE,
BAND ordinary/freeprop/spin/relativistic output, and spin-Hubbard
full-potential LDOS generation all have focused release-profile gates. Further
fixture broadening is non-blocking coverage work rather than an unported or
compatibility-gated production path. This inventory status does not replace an
actual run of the required-fixture workspace suite and strict readiness
command.

The completion audit also closes the production branches that were missed by
the earlier lexical inventory:

- `opcons` can generate missing elemental `opcons*.dat` inputs from FEFF's
  bundled `epsdb` rows for Z=1 through Z=99.
- broadened Hedin-Lundqvist XSPH exchange loads an external 1050-row
  `bphl.dat` table and applies the FEFF `rhlbp` interpolation.
- polarized `MULTIPOLES=3` composes E1+E2+M1 with E1 counted once; PMBSE
  nonlocal selectors consume `pot.ch` or `yoshi.dat`/`wscrn.dat`, and
  two-spin TDLDA merges both spin results.
- FULLSPECTRUM writes FEFF's final fake `xmu.dat`; when `CONTROL(6)=0`, it
  keeps source-spectrum output but neither rewrites nor advertises the optical
  post-processing files.
- unavailable DEBYE selectors warn and normalize to `idwopt=2`, while DMDW
  type-2 `E_k_opt` converts only selector 1 and passes other selector values
  through unchanged.
- iterative POT terminal/retry states materialize final `pot.bin`/`apot.bin`,
  and the generated XANES/Cu reference includes a nested RHORRP fixture with
  density text/binary files plus `gg_slice.bin`/`gg_diag.bin`.
- the canonical fresh `XANES/BN` workflow now completes at approximately
  `1–2e-5` relative L2 parity. Its closure required POT independent-center
  FMS rows in slot zero with saved SCMT retry state, a frozen
  `sqrt(rhoint)` plasmon value, raw-Hartree `emu` with distinct `ixc0` and
  `ixc` XSPH roles, and FMS reversed-axis rotations computed from the original
  vectors.
- parity prefers a canonical generated file such as `xmu.dat` over a legacy
  `referencexmu.dat` alias and excludes nested compatibility subcases from the
  parent workflow comparison.

Recent source-status cleanup gives DMDW, EELS, EELS-MDFF, and SFCONV/SELF
canonical `has_supported_*_source_handoff` predicates around their existing
source-backed generators. The project status metric now recognizes source
handoffs for all 21 modules instead of counting those modules as cache-only
despite their typed source paths. DMDW now also compares readable `dmdw.out`
caches against regenerable `.dym` source handoffs and rewrites stale output
through the same Rust generator, with `refeff run` scheduler coverage for that
repair. Malformed `dmdw.out` caches are also regenerated from valid `.dym`
source handoffs, while readable `dmdw.out` caches no longer mask malformed
declared `.dym` sources.
The scheduler also has a DEBYE/DM/EXAFS/Cu DMDW reference gate that starts
from `dmdw.inp` plus `feff.dym`, omits cached `dmdw.out`, and checks the
generated report against the FEFF reference at printed-report precision.
DMDW source-handoff detection now parses ordinary `.dym` sources and validates
type-2 phonon coupling tables before advertising source-backed completion, so
malformed source files fall through to the required-stage parser error instead
of being reported as completed work. Malformed `dmdw.inp` is declined during
cached-output and source-handoff discovery, while direct DMDW execution still
reports the parser error.
Type-2 `E_k_opt` now follows FEFF's selector contract exactly: selector 1
converts the electron energy to the characteristic-energy scale, while every
other selector value leaves it unchanged. The production `dym2feffinp`
executable parses FEFF's option spellings and writes reparsable centered
`feff.inp` and `.dym` outputs through the typed IO converter.
FMS `idwopt=5` source-handoff detection now parses the `.dym` file referenced
by `dmdw.inp` before reporting source-backed completion, keeping malformed
DMDW damping inputs out of completed-stage accounting; the matching module and
scheduler regressions cover malformed FMS DMDW handoffs.
EELS source-handoff detection now parses the requested `xmu*.dat` or
`opconsKK*.dat` source spectra before advertising EELS or EELS-MDFF as
source-backed supported stages, so malformed source files fall through to the
normal required-stage parser error instead of being reported as completed work.
Readable `eels.dat` caches no longer mask malformed typed EELS source spectra
when those handoff files are present.
The full-run scheduler also has a generated `ELNES/Cu` reference gate that
starts from `eels.inp` plus `xmu.dat` through `xmu09.dat`, omits cached
`eels.dat`, and compares the generated tensor spectrum to the FEFF reference.
Malformed `eels.inp` is likewise declined by both cached-output and source-handoff
discovery, while direct EELS execution remains strict.
SFCONV SO2CONV source detection now applies the same boundary to requested
target spectra: malformed `xmu.dat`/`chi.dat`/path targets are not advertised
as supported work during scheduler discovery, while direct SFCONV execution
still reports the target parser error. Malformed `sfconv.inp` is declined by
SFCONV and SELF discovery, while direct SFCONV/SELF execution remains strict.
SELF now follows the same source-backed recovery rule for excitation poles:
malformed or readable-but-stale `exc.dat` caches are regenerated from supported
`xsph.inp`/`loss.dat` many-pole handoffs, while malformed standalone caches are
not advertised as supported. Full `refeff run` covers stale `exc.dat` repair
before later required stages stop the run. A scheduler-level MPSE/Cu reference
gate now lets `rdinp` create the enabled SELF handoff, supplies only the real
`xsph.inp` and `loss.dat` source tables, omits cached `exc.dat`, and compares
the generated excitation-pole table to `REFERENCE/exc.dat`. Malformed
`xsph.inp` source inputs are now declined by XSPH and SELF supported-stage
discovery, and readable `exc.dat` caches no longer mask malformed declared
`xsph.inp` or `loss.dat` SELF source handoffs. Direct XSPH or SELF execution
still reports the underlying parser/source error.

COMPTON now applies the same cache/source boundary to its RHORRP-backed
diagnostics: malformed or readable-but-stale `jzzp.dat` and `rhozzp.dat`
caches fall through to the Rust RHORRP density callback only when the complete
callback handoff is present, while standalone readable caches continue to be
preserved. The full-run supported-stage scheduler now also covers stale
COMPTON cache regeneration after RDINP has prepared `compton.inp` and the
RHORRP callback handoff is installed. The supported-stage predicate now parses
standalone `jzzp.dat`/`rhozzp.dat` caches before advertising COMPTON and also
checks `jzzp.dat` grid compatibility with `compton.inp` when no RHORRP source
fallback is available. Readable COMPTON caches now also validate complete
declared RHORRP callback bundles first, so malformed `phase.bin`/density-source
handoffs cannot be hidden by a compatible `jzzp.dat` or `rhozzp.dat` cache.
Malformed `compton.inp` is declined during COMPTON supported-stage discovery,
while direct COMPTON execution remains strict.

Recent PATH orchestration progress lets malformed, unreadable, or readable but
stale `paths.dat` caches fall through to the Rust pathfinder source route when
compatible `phase.bin`/`geom.dat`/`global.inp` handoffs are present, while
still refusing to advertise malformed standalone caches. Full `refeff run` now
has regression coverage for recovering `paths.dat` in the supported-stage pass
before later required stages stop the run. Malformed `paths.inp` is also
declined during supported-stage discovery, leaving explicit PATH execution to
report the parser error. Readable `paths.dat` caches no longer mask malformed
declared `phase.bin`/`geom.dat`/`global.inp` pathfinder source handoffs.

Recent RHORRP orchestration progress applies the same source-backed recovery
rule to requested density grids: malformed, unreadable, or readable but stale
core-density outputs are regenerated from valid `pot.bin`/`geom.dat` handoffs,
while malformed standalone density caches still fail instead of being
advertised as supported. Full `refeff run` now covers this recovery before
POT refreshes or later XSPH/FMS handoff stages can preempt the RHORRP density
request. Malformed `density.inp` is also declined during supported-stage
discovery, leaving explicit RHORRP execution to report the parser error.
Readable RHORRP density caches now also validate complete declared core/table
source handoffs before being accepted, so malformed `geom.dat` or `phase.bin`
source state cannot be hidden by cache replay.
XSPH supported-stage discovery now also uses the real source phase generator
before advertising a phase-only or complete base-output handoff. FEFF-style
`xcpot` negative-radicand stops, such as the RHORRP/POT refresh case that fails
while evaluating `corrected_momentum`, decline scheduler discovery while
explicit XSPH phase execution remains strict and reports the exchange error.

Recent CRPA orchestration progress reuses the SCREEN source handoff for both the
optional screened-potential sidecar and the source response assembly: missing or
malformed `wscrn.dat` is regenerated from valid `vtot.dat`/`apot.bin` handoffs
when cached `crpa.dat` is present, and malformed `logscrn.dat` is regenerated
only for that source-backed sidecar repair. When full SCREEN source state is
available, CRPA now consumes the same potential, FOVRG radial, phase, and FMS
components to assemble `den_CRPA`, `totden_CRPA`, occupied response slices, and
paired `crpa.dat`/`wscrn.dat` output. The default source-reference gate
`crpa_module_generates_reference_zip_from_source_without_phase_or_gg_cache`
extracts the CRPA reference zip without cached `crpa.dat`, `wscrn.dat`,
`phase.bin`, or `gg.bin`; the generated Hubbard summary is within `1e-5`, and
the CRPA-relevant `wscrn.dat` radius/screened-potential columns are within
`1e-5` of the FEFF reference. When that complete source bundle is present,
readable cached `crpa.dat` and `wscrn.dat` are now render-normalized against the
generated source payload before CRPA is advertised as cache-complete, so stale
CRPA output regenerates through the source response assembly instead of
shadowing it. Full-run supported-module orchestration now also runs the same
CRPA zip fixture through the scheduler, reporting completed `crpa` output and
comparing the generated Hubbard summary plus CRPA `wscrn.dat` screened-potential
rows against the FEFF reference without cached `phase.bin`/`gg.bin`. The same
`vtot.dat`/`apot.bin` pair is still
scheduled as a validation-only `crpa-wscrn` pre-solver handoff when `crpa.dat`
is absent and the full source bundle is incomplete. Full `refeff run` now
advertises complete source bundles as a completed `crpa` stage, while preserving
the validation-only `crpa-wscrn` repair path for incomplete source state before
later required stages stop the run. Shared `apot.bin` recovery
sidecars no longer make ATOM advertise a cached stage unless `pot.inp` is also
present. Malformed standalone CRPA sidecars and module logs remain strict;
malformed final `crpa.dat` caches with recoverable `vtot.dat`/`apot.bin` state
fall through to the same source-backed `crpa-wscrn` handoff before stopping at
the source requirement when no complete source assembly exists. Malformed
declared `screen.inp` source handoffs are now treated as unsupported during
CRPA scheduler discovery, while explicit CRPA execution still reports the
parser error. Readable `crpa.dat` caches now follow the same rule, so cached
CRPA output cannot mask a malformed declared `screen.inp` handoff during
supported-stage discovery. Malformed `crpa.inp` is likewise declined during
cached-output, `crpa-wscrn`, and source-handoff discovery, while direct CRPA
execution remains strict.

Recent KSPACE/LDOS orchestration progress includes a shared `refeff-io`
`kmesh.dat` handoff generator backed by the Rust ports of FEFF
`KSPACE/kmesh.f90` mesh division and reduction helpers. BAND and
reciprocal-space LDOS runs now regenerate no-symmetry FEFF `kmesh.dat`
sidecars from `reciprocal.inp` before their remaining branch-parity
boundaries, while existing valid cached `kmesh.dat` files continue to
be validated and re-rendered and absent optional `kmesh.dat` files no longer
block non-reciprocal cached or handoff routes. Malformed or unreadable
`kmesh.dat` sidecars are now regenerated from the same no-symmetry reciprocal
handoff when available.
Full `refeff run` orchestration also schedules that source-backed `kmesh.dat`
handoff directly from `reciprocal.inp` before earlier source requirements stop
the run for both `band-handoff` and `ldos-kmesh` supported stages. Malformed
declared `reciprocal.inp` handoffs are now treated as unsupported during
KSPACE/LDOS scheduler discovery, while explicit module execution still reports
the reciprocal parser error. Readable LDOS `ldosNN.dat`/`rhocNN.dat` caches
now also validate declared wavefunction source handoffs before advertising
cache completion, so malformed `pot.bin`/`config.dat`/`phase.bin`/`pot.inp`
source bundles cannot mask source-backed LDOS repair or regeneration paths.
`refeff-io` also exposes an explicit-operation
symmetry-reduction handoff for callers that have parsed FEFF-compatible
crystallographic symmetry matrices. The BAND final-output adapter now also
assembles FEFF-compatible `bandstructure.dat` rows directly from solved
k-point eigenvalue rows, matching the `BAND/bandtot.f90` row layout and summary
headers; a typed setup/result adapter now combines sampled K-paths, clipped
search meshes, FEFF phase-energy counts, and Hartree BAND rows into the final
`bandstructure.dat` payload. The `bandtot.f90` K-path sampling setup is also
ported: Rust now
distributes the requested `nkp` across high-symmetry segments with FEFF's
integer arithmetic, emits the sampled `bk` points, and preserves the scalar
`KP` path distances and duplicate segment junctions used by the solver. The
`bandtot.f90` energy-search setup and band-identification tail are now ported
as well: Rust clips the requested energy window to the XSPH phase-shift range,
recomputes FEFF's `nep`/`estep` search mesh, and converts positive-eigenvalue
count increases into variable-length band-energy rows. The BAND-facing
`phase.bin` handoff and `bandtot.f90` reference-energy/phase-shift
interpolation loop are now Rust-backed too, including shared signed-`l` table
normalization and FEFF-compatible interpolation onto the BAND search mesh. The
uncached BAND CLI path now consumes this `phase.bin` handoff when present,
builds the high-symmetry K-path setup from `reciprocal.inp`, and preserves or
generates the no-symmetry `kmesh.dat` sidecar before the source-output
requirement. When both phase and reciprocal handoffs are present, Rust
validates them as one combined pre-solver setup and also derives the FEFF
`STRVECGEN` lattice state: `ALAT`-scaled direct basis, inverse `BGX/BGY/BGZ`
basis, q-pair groups, direct `R1/R2/R3` lists with `SMAX`/`INDR`, reciprocal
`G1/G2/G3` lists, Ewald defaults, reduced search-energy bounds, `QJLTAB`, real
Gaunt triples, `CIPWL`, non-relativistic `IND0Q`/`NKMQ` site layout, basis
transforms, relativistic `NRREL`/`IRREL`/`SRREL` component tables, spin-orbit
tables, the solver state-ket/FMS atom basis, and the `STRCC` reduced-energy
schedule `ERYD/(2*pi/ALAT)^2` from interpolated `E-eref`. Rust can now build
one-energy `STRCC` Ewald tables on demand,
assemble a borrowed non-relativistic `STRBBDD -> STRSET -> structurefactor`
point input from the combined handoff, assemble the borrowed relativistic
`IREL >= 2` point input through the same source handoff boundary, build
guarded relativistic KKR/`freeprop` grids and final `bandstructure.dat` row
adapters for compatible one-spin handoffs, and build the per-energy
`fmsband.f90` lattice T-matrix grid from the same source phase state. For
compatible ordinary non-relativistic handoffs, Rust can now iterate the full
search-energy/k-path grid, compose source-backed KSPACE structure factors with
those T-matrices, run KKR eigenvalue counting, identify final band rows, and
assemble FEFF-compatible `bandstructure.dat` through the production CLI path.
Compatible `freeprop` non-relativistic handoffs now use the same KSPACE grid
and the ported raw-`G` diagonalization branch to assemble `bandstructure.dat`
through that production CLI path as well. Spin-degenerate multi-spin source
handoffs now use the full two-spin `IREL >= 2` KSPACE `G` grid before the
existing multi-spin T-matrix/eigenvalue solve. Non-degenerate spin-resolved
multi-spin handoffs use FEFF's final-spin scalar `fmsband` wave-number
semantics to build the same full-order source-backed KSPACE grid before KKR or
`freeprop` solve.
Full `refeff run` orchestration now promotes compatible source handoffs that
generate `bandstructure.dat` to the completed `band` supported stage, while
validation-only reciprocal/pre-solver paths stay reported as `band-handoff` and
continue to report generated `kmesh.dat` through that BAND pre-solver stage
instead of relying on the later standalone kmesh fallback.
Malformed final `bandstructure.dat` caches no longer suppress that source
handoff validation when compatible pre-solver inputs are present. Valid but
stale final caches are also compared against the supported source summary
header metadata, k-point shape, per-row band counts, and band eigenvalue rows,
then regenerated from the Rust source driver when they no longer match, while
standalone malformed final caches still fail validation. Full-run scheduling now
pins same-shape stale eigenvalue regeneration directly and also pins the
`freeprop` final-row shape path: a readable
`bandstructure.dat` whose per-row band counts no longer match the source
handoff is regenerated as a completed `band` stage rather than downgraded to a
validation-only `band-handoff`.
The
`fmsband.f90` lattice T-matrix expansion and `kkrband.f90` `G - T^-1`
work-matrix setup are also source-backed in Rust, with the T-matrix expansion
now assembled across every interpolated BAND search energy and composed with
completed FEFF-basis `G(energy,kpoint)` grids through final band rows, along
with the `structurefactor.f90` tail that
converts completed SPRKKR `tauk` blocks and search grids into FEFF-basis `G`
blocks, plus KSPACE-backed `STRBBDD -> STRSET -> G` grid assembly and
ordinary KSPACE-plus-phase solve composition through final rows, plus one-point
and `(energy,kpoint)` KKR solve composition. The
`kkrband.f90` `freeprop` branch is also ported through the same KSPACE-backed
one-point and `(energy,kpoint)` orchestration, preserving FEFF's raw-`G`
diagonalization path when the lattice T-matrix is intentionally skipped. The
`fmsband.f90` `Gfms*p` general complex eigenvalue extraction through the
pure-Rust `faer` adapter, real-part sort, single-solve and `(energy,kpoint)`
KKR eigenvalue-grid orchestration, and
`bandtot.f90` positive-eigenvalue count table through final band-energy rows
are now Rust-backed too. The
`KSPACE/strbbdd.f90` reciprocal/direct lattice-sum kernel is also ported,
including the `strharpol.f90` real-harmonic polynomial generator, reciprocal
cutoff denominator, missing q-pair phase, direct-list indirection, and `D300`
correction. The `strset.f90` non-relativistic Gaunt contraction into FEFF
`TAUKINV`, including `CIPWL` phase ratios, first-pair `-i*p` diagonals, and
equivalent q-pair block copying, is now Rust-backed and composed with
`STRBBDD` for one-k-point source-backed structure-constant assembly.
`strvecgen.f90` q-pair
grouping now produces the shared `QQP` offsets and equivalent site-pair lists
for those kernels in Rust, and the direct-lattice side now emits FEFF's sorted
`R1/R2/R3` vector list plus adjusted `SMAX`/`INDR` direct-term indirection. The
reciprocal side now emits the sorted `G1/G2/G3` vector list with FEFF's
half-cell shifted reduced-energy probe, and `straa.f90` now builds the
`EXPGNQ` reciprocal pair phase table with FEFF's `D1TERM1` and Ewald Gaussian
prefactor plus the base direct-lattice `QQMLRS` table and `GGJLRS`
continued-fraction radial table. `strcc.f90` fixed-`ETA` energy products are
also Rust-backed now, including `IILERS`, current-energy `QQMLRS`, `D1TERM3`,
and `D300`; FEFF `change_eta.f90` retry policy now rebuilds the Rust Ewald
tables by increasing `ETA` by 1.4 up to the hard maximum of 3.0. The
`strset.f90` relativistic `SRREL` transform and equivalent q-pair copying are
now Rust-backed and composed with `STRBBDD` as well. Standalone BAND module
runs now surface the same source-backed pre-solver `phase.bin`/reciprocal
handoff validation used by full-run orchestration before the KKR solver
boundary. The `freeprop` empty-lattice pre-solver branch now treats
`reciprocal.inp` as the only source handoff it needs, so stale `phase.bin`
sidecars no longer block K-path generation or make phase-only state look like a
supported BAND handoff. The cached BAND path now also regenerates malformed
`logband.dat` when `kmesh.dat` is repaired from a valid `reciprocal.inp`
handoff, or when validated `phase.bin`/reciprocal pre-solver state supplies the
same deterministic module-wrapper boundary; pure cached malformed module logs
remain validation failures. The validation-only `band-handoff` stage now also
repairs an existing malformed `logband.dat` without creating a new log for clean
pre-solver handoffs. BAND source setup now also reads optional `fms.inp`
`lmaxph(0:nph)` cutoffs, matching FEFF's `reafms -> kprep` path where KSPACE
`maxl`/`msize` come from FMS active angular cutoffs rather than the larger raw
phase-write range; the optional Cr2GeC reference handoff pins the resulting
128-state active matrix order, and the local generated Cr2GeC BAND reference
gate now runs the Rust source path through `bandstructure.dat` and compares the
result with FEFF's generated file, including summary header metadata. Full-run
supported-module orchestration now carries the same generated Cr2GeC
`band.inp`/`reciprocal.inp`/`fms.inp`/`global.inp`/`phase.bin` source bundle
through a completed `band` scheduler report and compares the produced
`bandstructure.dat` with FEFF's generated output. Malformed BAND `fms.inp`
`lmaxph` handoffs now decline both the direct pre-solver report and the
full-run `band`/`band-handoff` scheduler reports instead of advertising a
source-backed completion from the remaining source files. Full-run
orchestration now also carries the KSPACE/Graphite `reciprocal.inp` handoff
through the dedicated validation-only `kmesh` scheduler report, comparing
generated `kmesh.dat` against FEFF's archived reference without advertising completed
`bandstructure.dat` output. BAND source setup now also reads optional
`global.inp` and carries FEFF's `ispin` selector into the one-spin spin-orbit T-matrix
handoff, so polarized source bundles no longer collapse to the default
non-rel source path. One-spin relativistic KKR/`freeprop` production dispatch
now uses that nonzero `ispin` selector and is covered by module and full-run
scheduler regressions. Full-run scheduling now also covers non-degenerate
two-spin ordinary and `freeprop` source bundles, so spin-resolved multi-spin
dispatch reaches completed `bandstructure.dat` output instead of a
validation-only handoff.
The release-profile direct module sweep `cargo test --profile release -p refeff-engine band_module_generates_`
now covers ordinary, `freeprop`, one-spin relativistic, two-spin degenerate,
two-spin non-degenerate, and kmesh/pre-solver source handoffs.
`full_run_scheduler_regenerates_stale_two_spin_bandstructure_from_source_handoffs`
and
`full_run_scheduler_regenerates_stale_two_spin_freeprop_bandstructure_from_source_handoffs`
now pin the same readable-stale `bandstructure.dat` repair boundary for those
non-degenerate two-spin ordinary and `freeprop` source bundles, with matching
direct BAND module regressions for generated and stale two-spin ordinary/freeprop and
one-spin relativistic ordinary/freeprop final caches. Release-profile full-run
scheduler regressions now also pin readable-stale one-spin relativistic
ordinary/freeprop repair from the same `global.inp` `ispin = 1` source
handoffs. The remaining
core BAND release gates now also cover non-relativistic and relativistic
KSPACE structure-factor grid assembly, KKR source-grid final-row identification,
and raw-`G` `freeprop` source-grid final-row identification through
`cargo test --profile release -p refeff-core band_structure_factor_from_kspace`,
`cargo test --profile release -p refeff-core band_kkr_band_energies_from_kspace`,
and
`cargo test --profile release -p refeff-core band_free_propagation_band_energies_from_kspace`.
The remaining BAND work is broader branch coverage for generated
`bandstructure.dat`, not module-level source dispatch. The explicit BAND
unported fallback is retired: missing or incomplete source state now
reports a normal source-requirement error after pre-solver validation, while
complete supported source bundles write `bandstructure.dat`.
BAND source-handoff discovery now also declines malformed declared `phase.bin`
bundles instead of reporting `band` or `band-handoff`; explicit BAND execution
still surfaces the phase parser/setup error, and readable cached
`bandstructure.dat`/`kmesh.dat` output no longer masks a malformed declared
`phase.bin` source.
Malformed declared `reciprocal.inp` handoffs now follow the same direct-module
and full-run scheduler rule, so a bad reciprocal source file cannot advertise a
completed `band` stage or validation-only `band-handoff`; readable cached
`bandstructure.dat`/`kmesh.dat` output no longer masks that malformed declared
source.
Malformed declared `global.inp` spin-selector handoffs now follow the same
scheduler-discovery rule, so a bad optional global file cannot be treated as
absent and cannot advertise a completed or validation-only BAND source stage,
including through readable cached final output. Malformed declared `fms.inp`
`lmaxph` handoffs now follow that same cached-output rule.
BAND supported-stage discovery now also declines malformed declared `band.inp`
files instead of reporting cached or source-backed BAND completion; explicit
BAND execution still surfaces the control-input parser error.

Recent EELS-MDFF progress removes the standalone MDFF module gate: `mdff.dat`
can now be generated from the same typed EELS source spectra used by the EELS
module. The Rust path covers FEFF `EELSMDFF/mdff_eels.f90` manual
`q_input=1` reduction, global q-vector manual handoffs, and the hardcoded
two-position `q_input=2` automatic q-grid branch backed by the ported
`mdff_qmesh.f90` helper, with task 1/2/3 differences reflected in the generated
module log. EELS and EELS-MDFF orchestration now also treats malformed
`eels.dat`/`mdff.dat` caches as recoverable when the matching typed source
spectra are present, while malformed standalone caches still fail validation.
EELS-MDFF discovery now also checks for `mdff.inp` before parsing optional
`global.inp`, so unrelated malformed global state no longer aborts MDFF
supported-stage discovery when the MDFF module input is absent. Malformed
`global.inp` with a declared `mdff.inp`, and malformed `mdff.inp`, are likewise
declined during cached-output and source-handoff discovery, while direct
EELS-MDFF execution reports the parser error.
Readable `eels.dat` and `mdff.dat` caches are also compared against
regenerable typed EELS source spectra and rewritten when stale, so a parseable
final-output cache no longer hides newer `xmu*.dat` or `opconsKK*.dat`
handoffs. EELS-MDFF now also declines readable `mdff.dat` caches when matching
typed EELS source spectra are present but malformed. The scheduler-level
`ELNES/Cu` reference gate now exercises that EELS generation path without a
cached `eels.dat`, proving the full-run source handoff can reproduce the
generated FEFF tensor spectrum.

Recent LDOS progress includes the `LDOS/ff2rho.f90` non-full-potential and
`LDOS/ff2rho_h.f90` spin-resolved output handoffs: completed `em`, `xrhoce`,
`xrhole`, and `cchi` work arrays can now produce typed `ldosNN.dat` and
`rhocNN.dat` payloads, with `rhoc` preserving embedded density and `ldos`
applying `imag(cchi*xrhole)` when `msapp != 1`. The spin path preserves FEFF's
spin-major column order, four orbital channels for spin up followed by the same
four for spin down. The CLI now composes the non-scattering no-FMS branch from
existing `rhocNN.dat` handoffs, filling missing per-index `ldosNN.dat` files
through the same typed `ff2rho` adapter with scattering disabled instead of
treating the embedded-density sidecar as a complete LDOS cache. That no-FMS
regeneration now also covers spin-resolved `rhocNN.dat` tables by routing
through the `ff2rho_h` adapter and requiring the table shape to agree with
`ldos.inp` `ispin`; the reverse no-FMS cache recovery now also fills missing
`rhocNN.dat` files from cached `ldosNN.dat` by preserving the shared
energy/density table and dropping LDOS-only header metadata. The same no-FMS
counterpart recovery now treats malformed or unreadable paired sidecars as
recoverable when the valid counterpart can be parsed and converted through the
Rust `ff2rho`/`ff2rho_h` adapters, and full-run orchestration has regression
coverage for repairing malformed no-FMS final tables from complete radial
source handoffs before stale counterpart caches can mask the source result.
Module-level coverage still pins valid-counterpart repair for malformed
`ldosNN.dat` and `rhocNN.dat` without requiring an unrelated `kmesh.dat`.
Malformed `logdos.dat` wrappers are
also regenerated for those source-backed table repairs and reciprocal k-mesh
handoffs, while pure cached-output runs still reject malformed module logs. The
validation-only `ldos-kmesh` handoff now repairs an existing malformed
`logdos.dat` without creating a new log for clean pre-solver mesh handoffs.
When complete no-FMS radial source handoffs are present, readable
`ldosNN.dat`/`rhocNN.dat` pairs are now compared against the source-generated
energy and density grids and rewritten if a same-shape stale pair would
otherwise mask the Rust `rhol` driver.
Malformed `ldos.inp` is declined during LDOS supported-stage discovery,
including cached-table, source-output, and `ldos-kmesh` predicates, while
explicit LDOS execution still reports the parser error.
Stale malformed final `ldosNN.dat` caches no longer suppress the reciprocal
`kmesh.dat` source handoff; that path now validates or writes the mesh before
the LDOS source requirement. The standalone `refeff module ldos` path now
also uses the reciprocal `kmesh.dat` source handoff directly instead of entering
the source-output boundary after writing the mesh. The `gtrNN.bin` codec now
also projects one potential's FMS trace into the `(l, energy)` `cchi` layout
consumed by the Rust `ff2rho` adapter. The core LDOS helpers also include the
non-full-potential `LDOS/fmsdos.f90` trace projection from packed FMS `gg`
matrices and phase shifts, including the FEFF magnetic-channel sum and
`exp(2*i*phase)/(2*l+1)` normalization. That projection now also has an
energy-grid adapter that emits `(energy, potential, angular)` values in the
same orientation as `gtrNN.bin`, and the IO codec can package that source grid
with FEFF header metadata. The LDOS runner now uses those pieces to write
non-spin `gtrNN.bin` source handoffs from supported `phase.bin`/`geom.dat`/
`global.inp`/`fms.inp` inputs before the final-table source requirement.
Release-profile compatibility rows now separately gate the non-full-potential
`ff2rho` final-table formulas and this `fmsdos` packed-`gg` trace projection,
and release-profile direct LDOS sweeps now gate broad source generation plus
repair/recovery behavior, so those FEFF loops and workflow branches are tracked
independently from the subsequently completed spin-Hubbard/full-potential
final-table source-generation path.
The core LDOS helpers now also cover the post-radial-solver
`LDOS/rhol.f90` density integrals that turn normalized regular and irregular
radial components into `xrhole` and `xrhoce` values, with an energy-grid
adapter that emits the same `(angular, energy)` work-array orientation consumed
by the Rust `ff2rho` table writer. The LDOS-facing `rhol` exact radial-tail
adapter now also reuses the shared FEFF Bessel/Neumann continuation for rows
`jri:ilast`, preserving the regular and irregular component formulas used
before those density integrals. LDOS now also has a `rhol` radial assembly
adapter for raw regular/irregular `dfovrg` outputs: it applies FEFF's
`xfnorm`, Wronskian irregular replacement, and exact-tail overwrite while
omitting RHORRP's origin smoothing branch. A one-channel LDOS `rhol` wrapper
now also invokes the shared FOVRG `dfovrg` driver for the regular and irregular
passes, performs the muffin-tin match and irregular boundary setup, and returns
the LDOS radial assembly for one `(energy, l, potential)` channel. A non-spin
source-backed `rhol` table driver now loops prepared per-channel FOVRG inputs
in `(energy, l)` order, evaluates `xrhole`/`xrhoce`, and feeds the Rust
`ff2rho` adapter for final `ldosNN.dat`/`rhocNN.dat` payloads. LDOS can now
also consume shared source-backed RHORRP wavefunction tables directly:
selecting one potential from `(energy, l, radial, iph)` `prel`/`pnel`/`qrel`/
`qnel` arrays, evaluating the LDOS density integrals, and feeding the same
`ff2rho` adapter. The CLI runner now wires those bridges to source handoff
files: when compatible non-spin `pot.bin`/`config.dat`/`phase.bin`/`pot.inp`/
`fms.inp` radial handoffs and matching `gtrNN.bin` traces are present, it
writes final `ldosNN.dat`/`rhocNN.dat` tables before the source requirement. The
no-FMS source path now needs only the radial handoff files
(`pot.bin`/`config.dat`/`phase.bin`/`pot.inp`), prepares the shared radial
source once, drives the LDOS-specific `rhol` FOVRG table driver on the
`ldos.inp` mesh for the missing
available potentials without requiring `gtrNN.bin`, supplies a zero scattering
trace, and produces matching `ldosNN.dat`/`rhocNN.dat` densities. It has a
production EXAFS/Cu source-generation smoke gate for the real 101-point mesh,
and generated tables preserve FEFF header metadata from
`pot.bin`/`fms.inp`/`geom.dat` (`xmu`, `qnrm`, `xnmues`, `inclus`, and
broadening) under the source reference gates,
generated XANES/Cu and NRIXS/GeCl4 source-handoff parity fixtures using
production no-FMS LDOS cards, an ordinary-spin XANES/Cu no-FMS source parity
fixture that preserves FEFF's regular four-column output shape, a short
NRIXS/GeCl4 no-FMS source parity fixture that exercises FEFF-valid
valence-only `config.dat` orbitals, and a regression that bounds generation to
the source potential count. Full-run orchestration now also has a XANES/Cu
no-FMS source parity gate that compares scheduler-generated
`ldosNN.dat`/`rhocNN.dat` energy and density grids against the FEFF reference
tables before advertising a completed `ldos` stage.
`full_run_scheduler_regenerates_stale_xanes_cu_no_fms_ldos_tables_from_source_handoffs`
now reuses that production no-FMS source bundle to prove readable same-shape
`ldos00.dat`/`rhoc00.dat` caches are regenerated from source before the
completed scheduler report is accepted. Full-run orchestration now
advertises complete no-FMS radial source handoffs and supported FMS source-grid
handoffs as a completed `ldos` supported stage before the source requirement. Those no-FMS
radial source-handoff probes now decline malformed declared
`pot.bin`/`config.dat`/`phase.bin`/`pot.inp` bundles during scheduler
discovery, while explicit LDOS execution still reports the underlying typed
reader/parser error. Those no-FMS
radial profiles now match the FEFF
golden tables closely after using FEFF's `csomm2` row count,
direct LDOS-card energies, and `rhol` normalization. The absorber FMS source
path now also covers FEFF's real-space zero-cluster branch: compatible RHORRP
wavefunction handoffs regenerate zero `gtrNN.bin` files on the LDOS-card grid,
and an EXAFS/Cu reference regression checks that the resulting `ldosNN.dat` and
`rhocNN.dat` tables match the FEFF golden no-scattering tables. Full-FMS table
assembly now matches FEFF `ff2rho` by preferring each potential's matching
`gtrNN.bin` file and selecting that potential trace from the file.
The `gtrNN.bin` codec now roundtrips FEFF's default single-precision complex
payload, and the source-grid FMS path writes FEFF-shaped per-potential
`gtrNN.bin` data with only the central potential column populated in each file.
The phase-grid `gtrNN.bin` handoff writer now also compares readable caches
against generated source output before preserving them, so stale phase-grid
traces are regenerated when the LDOS card mesh matches `phase.bin`.
A generated short XANES/Cu nonzero full-cluster FMS reference now checks
`gtr00.bin`/`gtr01.bin` and the resulting `ldosNN.dat`/`rhocNN.dat` tables.
A generated 101-point XANES/Cu production full-FMS release gate now checks the
same `gtrNN.bin` and final-table parity as normal default coverage.
Complete FMS wavefunction source handoffs now also compare readable final
`ldosNN.dat`/`rhocNN.dat` pairs against the source-generated tables and rewrite
stale same-shape caches before they can mask the source-grid `gtrNN.bin` and
`rhol` driver. LDOS cache discovery now uses that same source-rendered
comparison before accepting readable FMS final tables, so stale
`ldosNN.dat`/`rhocNN.dat` pairs do not satisfy the cache predicate when source
handoffs can render the expected output. Nonmagnetic ordinary-spin FMS now
reuses the same source-backed
FMS grid after verifying zero `xsph.inp` `spinph` values, preserving FEFF's
regular four-column LDOS/RHOC output shape, with a short XANES/Cu reference
parity gate. The explicit LDOS unported fallback is now retired: missing or
incomplete table/radial/FMS source state reports a normal source-requirement
error, while complete supported source bundles write final
`ldosNN.dat`/`rhocNN.dat` tables. The Hubbard LDOS trace sidecars now have
typed Rust codecs for spin-resolved `gtrNN.bin`, magnetic-diagonal
`gtr_mNN.bin`, and off-diagonal `gtr_offNN.bin` payloads, including
byte-for-byte NiO reference roundtrips, so
the spin-Hubbard source generator and `ff2rho_h` orchestration can build on
typed binary handoffs instead of opaque parsing. The paired magnetic-orbital
text sidecars, `lmdosNN.dat` and `rhocmNN.dat`, now also parse/render through a
typed variable-`lx` table model with NiO reference coverage. The CLI LDOS cache
path now also has an active-Hubbard NiO reference-zip gate that preserves all
three potentials' `ldosNN.dat`/`rhocNN.dat` plus `lmdosNN.dat`/`rhocmNN.dat`
sidecars, including FEFF's legacy wrapped `hubbard.inp`, six-field `ldos.inp`,
and truncated six-column spin LDOS/RHOC text shapes. Active-Hubbard LDOS cache
completion now requires paired ordinary `ldosNN.dat`/`rhocNN.dat` tables plus
paired `lmdosNN.dat`/`rhocmNN.dat` sidecars for every cached potential, so a
partial ordinary cache no longer masks the spin-Hubbard source-generation
boundary. The ordinary pair must share the same energy grid
and density-column layout, and the paired magnetic sidecars must also share that
ordinary energy grid and each other's magnetic `lx`/density layout, so stale
ordinary or magnetic LDOS/RHOC sidecars cannot satisfy an active-Hubbard cached
stage. Non-Hubbard LDOS runs still ignore stray magnetic sidecar files at both
direct-module and full-run scheduler boundaries, so malformed `lmdosNN.dat` or
`rhocmNN.dat` files cannot turn an ordinary LDOS cache into an active-Hubbard
requirement. When a valid `gtr_mNN.bin` magnetic trace source is present, the
cache gate also requires the sidecars' energy count and magnetic layout to match
that source contract. Malformed ordinary `ldosNN.dat`/`rhocNN.dat` pair members
are treated as incomplete active-Hubbard caches, letting the existing no-FMS
regeneration path repair them from the valid counterpart before the magnetic
sidecar contract is evaluated. The full-run scheduler now mirrors that repair
rule, so a recoverable active-Hubbard ordinary pair is advertised as a completed
`ldos` stage only after the Rust runner regenerates the malformed half and
re-renders `logdos.dat`. Standalone no-FMS spin-resolved
`ldosNN.dat`/`rhocNN.dat` cache pairs now have the same scheduler-level repair
coverage in both directions, while
full-run ordinary-spin source handoffs continue to prefer FEFF's regular
four-column source-generated tables when RDINP has produced the radial handoff
bundle. When a valid spin-resolved Hubbard `gtrNN.bin` trace source is
present, the ordinary `ldosNN.dat`/`rhocNN.dat` pair must also match that
source's energy count and spin-density column layout. When both `gtrNN.bin` and
`gtr_mNN.bin` source traces are valid, their energy count and angular layout
must also agree before the active-Hubbard cache is accepted. Valid
`gtr_offNN.bin` off-diagonal source traces are also checked against the
magnetic sidecars and any ordinary/magnetic trace contracts for matching energy
count and angular layout before cached active-Hubbard completion is accepted.
Readable Hubbard trace sources that do not contain the cached potential index
are now treated as incompatible contracts rather than as absent optional
sources, so truncated per-potential trace bundles cannot bless stale final
tables. The direct LDOS module also accepts nonzero active-Hubbard cached
potentials through fallback `gtr00.bin`/`gtr_m00.bin`/`gtr_off00.bin` source
bundles when those bundles include the cached potential and agree with the
ordinary and magnetic tables. The full-run scheduler now also covers this boundary: an `ldos00`
cache with stale ordinary `ldos00.dat` or `rhoc00.dat` energy grids is not
reported complete, a stale ordinary density-column layout is not reported
complete, an `ldos01` cache with complete magnetic sidecars is not reported as
complete when readable `gtr01.bin`, `gtr_m01.bin`, or `gtr_off01.bin` omits
potential 1, and an `ldos00` active-Hubbard cache is
likewise rejected when `gtr00.bin`
advertises an ordinary layout that conflicts with `ldos00.dat`/`rhoc00.dat`,
when `gtr00.bin` and `gtr_m00.bin` advertise incompatible angular layouts, when
`gtr_m00.bin` advertises a conflicting magnetic layout, or when `gtr_off00.bin`
advertises a stale off-diagonal energy/angular layout. Full-run discovery now
also rejects stale or malformed active-Hubbard magnetic text sidecars directly,
including shifted `lmdos00.dat` and `rhocm00.dat` energy grids plus a stale
`rhocm00.dat` magnetic layout or malformed `lmdos00.dat`/`rhocm00.dat` text.
The matching positive direct-module and
scheduler gates now accept a complete `ldos00` active-Hubbard cache when
ordinary `gtr00.bin`, magnetic `gtr_m00.bin`, and off-diagonal
`gtr_off00.bin` source contracts all agree with the ordinary and magnetic final
tables, and re-render the `logdos.dat` wrapper through the supported `ldos`
report.
Rust now also ports
the FEFF `LDOS/ff2rho_h_step2.f90` magnetic table assembly that writes
`rhocmNN.dat` from embedded `xmrhoce` and `lmdosNN.dat` from
`xmrhoce/(2*l+1) + imag(gtr_m*xmrhole)`, with an IO adapter that builds
renderable `lmdos`/`rhocm` payloads from those source work arrays. The
`gtr_mNN.bin` codec now also selects one potential into that adapter's
`(l, magnetic, spin, energy)` trace layout, with release-profile coverage for
the magnetic `ff2rho_h_step2` table adapter. No-FMS active-Hubbard LDOS now uses
the same zero-scattering adapter path to repair one-sided
`lmdosNN.dat`/`rhocmNN.dat` magnetic sidecars from the paired magnetic table.
Nonzero and ordinary-spin FMS source-reference gates now compare generated
`gtrNN.bin`, LDOS, and RHOC tables against FEFF. Spin-resolved Hubbard source
generation and broader full-potential LDOS branches remain parity follow-up
work after the direct `ldos_module_generates_` and `ldos_module_recovers_`
release sweeps. The
phase-grid-only source `gtrNN.bin` path is still guarded so it only runs when
the LDOS card mesh matches `phase.bin`, while
the RHORRP-backed path regenerates `gtrNN.bin` from LDOS-card-grid RHORRP wave
numbers and phase shifts before final table assembly, so stale phase-grid FMS
traces no longer satisfy a mismatched LDOS mesh. Spin-Hubbard and
full-potential branches remain beyond that FMS-grid work.

Recent CRPA progress includes the solved `CRPA/chi_crpa.f90` Hubbard-summary
tail and the source response assembly ahead of it: SCREEN source components now
feed a typed `refeff-io` CRPA handoff that derives `ck(ie)`, selected-channel
`den_CRPA`, integrated `totden_CRPA`, occupied `chi0re(:,:,ie)` slices, and the
shared symmetric `chi0r` contour response. The final adapter normalizes and
projects that density, builds the bare Coulomb potential, solves for `wscrn`,
and emits paired `crpa.dat` plus CRPA `wscrn.dat`, using the final
`vch(i)=wscrn(i)*den_CRPA(i,ie)` sidecar column written by FEFF rather than the
earlier bare Coulomb vector. The source-generated CRPA reference zip gate now
runs as default coverage without cached `phase.bin`/`gg.bin`, and the
full-run supported-stage scheduler now reports complete CRPA source bundles as
`crpa` output while keeping `crpa-wscrn` for incomplete source-sidecar repair.

Recent XSPH kernel progress includes a Rust port of `XSPH/radint.f90` for the
reduced radial matrix-element branch (`ifl = 1` and `ifl = -1`) and central-atom
cross-section double-integral branches (`ifl = 2`, `3`, and `4`), plus the
JAS/NRIXS constant-step phase mesh from `XSPH/phmeshjas.f90`, JAS Bessel helper
from `XSPH/besjnjas.f90`, photon Bessel table setup from `XSPH/xsect.f90`, the
`xsect.f90` initial core-hole normalization check, `bcoef` ordinary
`kiind`/`jind`/`lind` transition-index setup and traced XSPH diagonal/cross-term
weight extraction, per-energy `p2`/`ck`/`omega`/`ilast` setup, `mult`/`kx`/`ks`/`kdif`
transition-loop planning with `kiind`/`lind` and `l2lp` filters,
screened-dipole `ww`/`wse` setup, `phiscf` workspace constants, local
exchange-field `fxc` setup, `phiscf` occupied-DOS/two-pole/dipole contribution
traversal, pole-energy/photon-correction and below-edge broadening setup,
`phiscf` radial-solver `ck`/`jrip`/`iwkb` setup, irregular Hankel seed
coefficients, regular/irregular `wfirdc` source contribution generation,
Wronskian normalization and outside-region field continuation, `lipman`
`K*chi0` response assembly, FEFF `aa` contribution scaling/imaginary-pole rule,
FEFF-style `cchik` response accumulation, owned radial-contribution handoff
from matched fields into `lipman`, `chiklu` screened-field LU
solve/interpolation, the multi-contribution screened field solve chain, and
the `wfirdc`-backed contribution collector that feeds that `fscf` solve,
plus the exported FOVRG C3 `vm` potential builder needed to construct
source-backed `phiscf` `wfirdc` inputs from the CLI and the CLI-side
positive-`izstd` per-pole `wfirdc` input assembly/collector invocation boundary,
`fscf` real/imaginary radial-pass weighting and negative-`ifl` `xk0`/`ww` scaling,
regular-solution `xfnorm` normalization, irregular muffin-tin boundary
initialization, and `N = iR - H exp(i*ph0)` post-`dfovrg` transform, plus the
positive-omega `xsnorm`/`xsec`/`rkk` output-normalization block, the direct
transition `rkk`/`phx` storage and unnormalized `xsnorm`/`xsec` accumulation
block with traced-`bcoef` weight handoff and explicit `rkk`/`phx` row-workspace
update, the diagonal central-atom `radint(ifl=2)` `xsec` accumulation block
with traced-`bcoef` weight handoff, the composed ordinary transition-row handoff
that applies those two updates in FEFF order, the standard-potential ordinary
row and spin-aware energy-row handoff that fold real/imaginary
`fscf`-weighted `radint(-1)`, `radint(-2)`, and same-`l` retry
`radint(-3/-4)` passes through the same traced-`bcoef` update and
positive-omega normalization path, the
spin-polarized XMCD cross-term `aa`/`bb`/`cc` accumulation block
with traced-`bcoef` off-diagonal weight handoff and explicit
`rkk1`/`phold`/`xrcold`/`xncold` retry-state reuse for `radint` branches `3`
and `4`, plus state-backed handoff into the bcoef-weighted cross-term
accumulation,
the `xsphsub.f90` final `xsect.dat` spin merge and `nq == 1` two-spin `rkk`
normalization handoff, plus the IO adapter that converts completed per-spin
XSPH rows into renderable `xsect.dat` data and matching post-merge transition
moments, a typed NRIXS adapter that converts completed `xsectjas` channel and
atomic final-state rows into renderable `xsecl.dat`, `xsecl2.dat`, and
`xsecl.bin` payloads with computed row sums, CLI-side q-resolved JAS radial
workspace and row-assembly helpers, and a one-spin normal-potential source
solver loop that writes those sidecars and updates phase `rkk` transition
moments in smoke coverage,
CLI-side validation that cached
NRIXS text sidecars match the active `phase.bin` energy mesh before they are
preserved, including the shifted `xsecl.dat`/`xsecl2.dat` energy columns and
stored row totals that match the printed channel columns, and that both text
sidecars use the same channel count and shared header scalars,
and that cached `xsecl.bin` final-state and transition dimensions match the
active `phase.bin` contract before it is preserved,
required-stage acceptance that demands the complete `xsectjas` sidecar set for
NRIXS selections, and full-run scheduler coverage that refuses to advertise
cached XSPH completion from readable `phase.bin`/`xsect.dat` alone when that
sidecar set is incomplete, when those sidecar energy rows or row totals are
stale, when primary or secondary text sidecars are malformed, when the text
sidecar channel layouts or shared headers disagree, or when the binary sidecar
header belongs to a stale phase contract or the binary sidecar is malformed,
with matching direct XSPH module regressions for primary and secondary
malformed text, primary and secondary stale text energy grids and row sums,
mismatched text contracts, and stale binary transition and final-state contracts
plus malformed binary sidecars, a typed
FF2X-side NRIXS adapter
that converts completed angular-decomposition rows into renderable `xmul.dat`
payloads with xsect-backed photon-energy/momentum grid conversion and the total
single-electron response column computed from the channel backgrounds, a
decomposed FF2X/JAS path-sum primitive that combines
`feffl.bin` channel amplitudes/phases on the FEFF momentum grid and adds
file-backed `fmsl.bin` decomposed FMS traces over the full XSPH grid, with the
GeCl4 NRIXS reference checking `fmsl.bin` against readable `gtrl.dat` rows and
pinning `xmul.dat` photon/k grid plus total `S^0(q,w)` row totals against
summed channel backgrounds, plus a source-backed FF2X GeCl4 gate that
regenerates `xmul.dat` from `ff2x.inp`/`global.inp`, `feff.bin`, `feffl.bin`,
`fmsl.bin`, `xsecl.bin`, and `xsect.dat` while applying FEFF's corrected
`xscorr` diagonal channel backgrounds and ignoring unused `xsecl.bin`
transition channels above `ldecmx`, and the CLI source-backed `xsect.dat` path
for normal-mesh/user-grid ordinary EXAFS/XANES/XES/DANES,
single-spin and ordinary plus M1/E2 `ispin = +/-1` two-spin, nonstandard
normal-potential absorber inputs for the `izstd <= 0` source branch,
the embedded central-density `xrhoce` and projected-density `xrhopr` radial
integration blocks used by ratio output, and the `iorb`/`kdif` branch selector
that chooses when those density ratios are evaluated, plus the `xirf`/`xirf1`
`fscf` magnitude combiner and spin-polarized `iold` cross-term retry planner,
JAS
orthogonality correction helper, overlap quadrature helper, reduced
radial-integral branch, and
central-atom double radial-integral branch from
`XSPH/radjas.f90`, plus phase angular-cutoff planning, the empty-cell
phase-matching branch from `XSPH/phase.f90`, the muffin-tin `imt`/`jri` radial
index setup, the per-energy momentum and wave-number setup branch that decides
whether an energy continues into phase matching, the per-angular-channel
`ll`/`ikap`/`ilp` setup loop, and the post-`phamp` small-phase cutoff/zeroing
branch, plus the final `eref(ne1)` tail copy for auxiliary energies, the
`phase.f90` `mpse.dat` self-energy summary values, MPSE plasmon-pole scaling,
and `PrintRl` header and radial-output normalization branch, and the
`phase_h.f90` Hubbard `Vnlm` potential-shift, `aph` assignment, and
reference-tail setup. Normal-potential phase support now also has source-backed
unreferenced `fixvar`/`fixdsx` grid preparation for the `xcpot` boundary and a
regular FOVRG-to-`phamp` channel primitive that returns both the matched phase
and the regular radial solution used by `PrintRl`. CLI pre-phase XSPH
orchestration now regenerates missing or unreadable `emesh.dat` and `emesh.bin`
from `pot.bin` before the source requirement boundary for the default,
user `grid.inp`,
finite-temperature `phmesh2T`, XES, FPRIME, FEFF `ispec = 5`
NRIXS/RHORRP `mk_rhorrp_grid`, `ispec = 5` user `grid.inp`, and
JAS/NRIXS constant-energy `phmeshjas` phase meshes, with global/RDINP NRIXS
`l2lp = 30` inputs selecting the FEFF NRIXS default-mesh capacity instead of
the ordinary `l2lp` transition filter.
Standalone
`refeff module xsph` runs now use the same complete-output, phase-only,
phase-text, and phase-mesh handoff ordering as full-run orchestration before
reporting a source requirement for incomplete phase state. The
uncached CLI path now also builds `phase.bin` directly for empty-cell-only
`pot.bin` inputs by composing the ported phase-energy setup, signed-`l` channel
plan, empty-cell `phamp` branch, cutoff handling, and `eref` tail finalization,
and it builds a conservative source-backed normal-potential `phase.bin` when
`pot.bin` and `config.dat` supply the prepared potential, density,
bound-orbital, and occupation handoffs, including the `loss.dat`/`MkExc` MPSE
pole handoff for `iPl > 0` and `ixc = 0`, and now covers two-spin phase-only
handoffs plus ordinary dipole and M1/E2 `ispin = +/-1` two-spin `xsect.dat`
spin-merge handoffs, including the unfiltered XMCD `ic3 = 1` cross-term retry.
It also writes the `PrintRl` `rl.dat`
radial-function sidecar for the generated absorber normal-potential phase
rows; the same handoff now restores a missing `rl.dat` when cached
`phase.bin` and `xsect.dat` are preserved. It also replaces malformed or
readable-but-stale `rl.dat` sidecars from that source handoff without rewriting
the preserved phase/cross-section caches, and malformed `log2.dat` wrappers are
regenerated when that source-backed `rl.dat` sidecar is written. For the
same single-spin and ordinary plus M1/E2 `ispin = +/-1` two-spin nonstandard
absorber branch on normal meshes and explicit user meshes, the uncached CLI
path now also drives the Rust
regular/irregular cross-section channels, ordinary `bcoef` accumulation, FEFF
edge-position photon scaling, ordinary `l2lp = -1, 0, 1` transition filtering,
the same MPSE pole handoff into dynamic `xcpot`, and spin merge to write
`xsect.dat` plus matching `phase.bin` transition moments and `mpse.dat`
self-energy sidecar rows. The standard-row radial accumulation now covers the
`fscf`-weighted same-`l` retry branches, and ordinary positive `izstd`
screened-dipole production now assembles occupied-state `wfirdc` rows, solves
the source-backed `phiscf` screened field, and feeds the resulting `fscf` into
the production `xsect.dat` row path. Mixed positive `izstd` dipole+E2 rows now
use per-transition screened fields so dipole transitions consume the screened
`fscf` while E2 transitions keep FEFF's unity field, and positive `izstd`
inputs that also carry PMBSE controls now mirror FEFF `xsphsub.f90` by ignoring
those PMBSE controls and using the ordinary source-backed `xsect` path. The
module-level `xsph.inp` path and the full-run scheduler path now cover that
PMBSE reset, including RDINP-shaped advanced controls. The full-run gate
`full_run_scheduler_generates_positive_izstd_xsph_while_ignoring_pmbse_from_source_handoffs`
now requires a completed supported-module `xsph` report from source handoffs,
and
`full_run_ignores_pmbse_for_positive_izstd_xsph_source_handoff`
now reaches completed `xsph` output after the positive-`izstd` `phiscf`
response accumulation and `cchik` solve keep large second-pole rows in double
precision, condition the solve, and fall back to the unity field only when the
screened solution numerically collapses.
Direct supported-module scheduler gates now also cover the global E2
multipole branch and the two-spin filtered source branch via
`full_run_scheduler_generates_global_multipole_xsph_from_source_handoffs`,
and `full_run_scheduler_generates_two_spin_filtered_xsph_from_source_handoffs`,
so those paths must report completed `xsph` output rather than relying only on
later full-run failure boundaries.
Release-profile scheduler gates now also cover the remaining LDOS
FMS/spin-FMS XSPH reference handoffs and the NRIXS/JAS source path that writes
`phase.bin`, `xsect.dat`, and the `xsecl*` sidecars from source handoffs.
Release-profile core gates now also pin FEFF XSPH phase setup, skip,
plasmon-pole, radial-output, mesh, self-energy-summary, and reference-tail
primitives through `cargo test --profile release -p refeff-core xsph_phase_`.
Positive `izstd` M1 remains intentionally guarded because FEFF
`XSPH/radint.f90` stops for M1 in the nonrelativistic `ifl < 0` branch. The real
`TDLDA/xsectd.f90` TDLDA/PMBSE driver now emits FEFF `TDLDA/meshlda.f90`-style
`ik0 = 0` source-backed `phase.bin`, matching `emesh.dat`/`emesh.bin`
sidecars, and source-generated `xsedge.dat` for the covered occupied,
calculated generated-basis, and hard-coded `Vila/Orbs` file-basis projector
paths. Covered TDLDA/PMBSE runs satisfy the XSPH required-stage contract via
`phase.bin` plus `xsedge.dat`; the Rust port does not fabricate the ordinary
`xsect.dat` table that FEFF `xsectd` does not source-populate. Broader TDLDA
projector coverage now also reaches full-run scheduling under release profile,
and release-profile
core gates pin FEFF `getmat`, energy-row setup, `getchi0`, `ridxmu`, `kkchi`,
channel weighting, broadening, and final `xsedge.dat` row assembly through
`cargo test --profile release -p refeff-core xsph_tdlda_`. The RDINP-driven PMBSE
source bundle is now pinned directly at the supported-module scheduler
boundary. Nonlocal core-hole selectors build their PMBSE source potential from
`pot.ch` or `yoshi.dat`/`wscrn.dat`, and the two-spin driver executes both
spin channels and merges their `xsedge.dat` response:
`full_run_scheduler_generates_tdlda_xsedge_from_pmbse_source_handoffs` requires
a completed `xsph` report with `phase.bin`, `emesh` sidecars, and the generated
unsplit `xsedge.dat` while keeping ordinary `xsect.dat` absent.
`full_run_scheduler_generates_file_basis_tdlda_xsedge_from_pmbse_source_handoffs`
and
`full_run_scheduler_generates_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs`
now pin the same completed scheduler report for the file-read `ibasis = 1` and
calculated generated-basis `ibasis = 2` projector branches.
`full_run_generates_file_basis_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement`
drives RDINP through the hard-coded `Vila/Orbs` source orbitals and checks the
generated four-row unsplit `xsedge.dat` while keeping ordinary `xsect.dat`
absent, and
`full_run_generates_generated_basis_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement`
does the same for the calculated generated-basis `ibasis = 2` branch.
Active `xsectd` selectors now also bypass the ordinary base-output completion
predicate, so a stale cached `xsect.dat` cannot mask a missing TDLDA
`xsedge.dat` required-stage output, and the runner now bypasses ordinary
`xsect.dat` validation entirely when the active TDLDA/PMBSE source bundle can
generate `xsedge.dat`. Malformed ordinary cross-section caches and readable
stale ordinary cross-section caches beside the source handoff therefore no
longer block `xsectd` generation or get counted as completed XSPH output. When
PMBSE source handoffs can infer the
active `xsedge.dat` row count and split-column shape, readable cached
`xsedge.dat` files must match that shape and the source PMBSE energy grid before
they are treated as cached completion; stale shape or energy mismatches fall
through to source regeneration when the PMBSE handoff bundle is complete.
Same-grid stale numeric `xsedge.dat` spectra are now covered by module and
full-run regressions that force source-backed `xsectd` regeneration. Direct
module regressions now cover the occupied-orbital `ibasis = 0`, file-read
`ibasis = 1`, and calculated generated-basis `ibasis = 2` projector branches;
the full-run scheduler gates
`full_run_scheduler_regenerates_stale_file_basis_tdlda_xsedge_from_pmbse_source_handoffs`
and
`full_run_scheduler_regenerates_stale_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs`
pin the same stale-table repair boundary for the file-read `ibasis = 1` and
calculated generated-basis `ibasis = 2` branches. Declared but malformed PMBSE
source bundles now also prevent cached `xsedge.dat` acceptance instead of being
treated as absent optional source state.
Full-run scheduling now also pins the incomplete file-read `ibasis = 1`
boundary: PMBSE runs without the required `Vila/Orbs` projector files may
report phase/emesh progress, but they do not report completed XSPH or write
`xsedge.dat`.
That source-backed cross-section path now also resolves optional `global.inp`
`le2`/`l2lp` handoffs, so rdinp-style FEFF inputs can drive the M1/E2
higher-multipole selector and global transition-direction filter instead of
falling back to dipole-only `xsph.inp` controls. The same adapter now passes
ordinary `global.inp` angular `bcoef` controls through to the source-backed
cross-section row builder, including `ipol`, `ispin`, `angks`, and the complex
polarization tensor, and the conservative two-spin `ispin = +/-1` path now
fills both FEFF spin rows for filtered dipole `l2lp = +/-1`, unfiltered
ordinary dipole `l2lp = 0`, and M1/E2 `le2` handoffs before the final
`xsect.dat` merge. The XSPH source-backed reference gates now copy and require
the FEFF `global.inp` handoff for phase/xsect and pre-phase mesh fixtures, so
EXAFS/XANES/DANES/FPRIME/XES and NRIXS mesh parity no longer silently falls
back to default angular controls. Full-run supported-module orchestration now
also regenerates the XANES/Cu screened-core-hole `phase.bin` and `xsect.dat`
from `xsph.inp`/`global.inp`/`pot.bin`/`config.dat`/`wscrn.dat` handoffs and
compares the scheduler-written phase shifts plus energy/background/cross-section
rows against the FEFF reference before reporting completed `xsph`. A dedicated
NRIXS/GeCl_4 source gate now copies `config.dat` with
`xsph.inp`/`global.inp`/`pot.bin`, regenerates `phase.bin`, `xsect.dat`,
`xsecl.dat`, `xsecl2.dat`, `xsecl.bin`, and `emesh` sidecars, and compares the
generated NRIXS phase and phase-derived mesh sidecars against the FEFF
reference. Full-run supported-module orchestration carries that same fixture
through the completed `xsph` scheduler report.
The normal-potential XSPH source gate now also verifies that every occupied
orbital implied by `config.dat` has matching `pot.bin` bound radial components
and origin coefficients before entering the FOVRG boundary, so incomplete
handoffs stay behind the source requirement instead of failing mid-channel.
The same gate parses optional `hubbard.inp` controls and now allows FEFF's
`mldos_hubb = 2` Hubbard phase branch when a compatible `v_hubbard.bin` handoff
is present; the source path applies the `Vnlm`-driven `phase_h` shifted
potentials, fills the `aph` workspace, and writes `aphase_hubbard.bin`.
Active Hubbard inputs without `v_hubbard.bin` stay behind the source
requirement, and ordinary `mldos_hubb = 1` inputs continue through the existing
normal-potential source path. The `refeff-io` layer now has typed
Fortran-unformatted codecs for `v_hubbard.bin`, `aphase_hubbard.bin`, and
`transformation_hubbard.bin`, including `lx` inference from record length, and
the XSPH cached-output path validates and re-renders cached
`aphase_hubbard.bin` when present. Active-Hubbard cached base
`phase.bin`/`xsect.dat` stages now also regenerate missing or malformed
`aphase_hubbard.bin` from the supported `pot.bin`/`config.dat`/`v_hubbard.bin`
source handoff. Readable `aphase_hubbard.bin` caches are compared against that
same generated source sidecar and regenerated when stale, while active-Hubbard
base caches without either a valid sidecar or that source handoff stay behind
the source requirement. The FMS
cached-output path now also
validates and re-renders cached `transformation_hubbard.bin` when
`hubbard.inp` and `phase.bin` provide the required dimensions, and repairs an
unreadable `gg.bin`/`gg.dat` companion from the other readable side before the
typed FMS roundtrip runs. When neither primary GG cache is readable but the
FMS source handoffs are complete, malformed `gg.bin`/`gg.dat` now fall through
to source-grid generation instead of blocking on the stale cache. Readable
`gg.bin`/`gg.dat` caches are also render-normalized against the generated FMS
source-grid output when those source handoffs are complete, so stale GG
matrices regenerate before MKGTR consumes them. The FMS
source-grid CLI gate now also covers the non-default `minv` solver selectors
(`1`, `2`, `3`, and FEFF's fallback path) through source-generated
`gg.dat`/`gg.bin`, `fms.bin`, and `gtr.dat` outputs. Orphan cached FMS
artifacts such as `gtrNN.bin` no longer make full-run orchestration claim the
FMS stage when `fms.inp` is absent, with full-run scheduler coverage pinning
that non-claiming boundary. Readable primary GG caches now also validate
declared `phase.bin`/`geom.dat`/`global.inp` source bundles before FMS
cache-completion discovery, so malformed FMS source state cannot be hidden by
cached `gg.bin`/`gg.dat` output.
Active-Hubbard FMS source generation now also honors `save_gg_slice` by
requesting the full full-potential LU scattering matrix, applying the Hubbard
back-transform to that full matrix, and writing source-backed `gg_slice.bin`
and `gg_diag.bin` sidecars. The release-profile FMS gate checks that the saved
absorber blocks reproduce the generated `gg.dat` matrix.
For cached base `phase.bin`/`xsect.dat` stages, missing optional `mpse.dat`
generation now remains opportunistic: complete compatible `phase.bin`/`pot.bin`
handoffs still produce the sidecar, while unsuitable cached potential state no
longer blocks the supported XSPH stage or downstream full-run orchestration.
Malformed or readable-but-stale `mpse.dat` sidecars now use that same
source-backed generator when the typed `phase.bin`/`pot.bin` handoff can
rebuild the MPSE table; if the handoff cannot regenerate MPSE, malformed
sidecars remain strict validation failures.
Full `refeff run` orchestration now recognizes that complete
`pot.bin`/`config.dat` handoff as a supported XSPH stage before cached
`phase.bin`/`xsect.dat` exist, so source-backed XSPH runs are reported in the
supported-stage summary instead of being skipped until the required-module pass.
The same source-backed base path now recovers malformed `phase.bin` or
`xsect.dat` base caches when the normal-potential handoffs can regenerate both
typed outputs, and it regenerates malformed `log2.dat` wrappers for those
source-backed base runs, while malformed standalone base caches and cached-only
module logs still fail validation. Requested AXAFS print sidecars now follow the
same source-backed rule: malformed or readable-but-stale `axafs.dat` is
regenerated from valid `phase.bin`/`xsect.dat` handoffs when `PRINT` asks for
AXAFS output, and the supported-stage detector still accepts that repairable
base stage; unrequested or ungeneratable AXAFS caches remain strict;
active Hubbard inputs join that stage only when `v_hubbard.bin` is available,
and FMS source generation now consumes complete active-Hubbard
`aphase_hubbard.bin` plus `transformation_hubbard.bin` handoffs through the
ported `fms_h` magnetic T-matrix, selected T-matrix transform, full-potential
LU, and back-transform path. GENFMT source generation now follows the ordinary
FEFF `rdxsph`-backed path for active-Hubbard inputs as well, matching the
reference GENFMT branch, and cached-output GENFMT runs now fall back to that
Rust source assembly when base `feff.bin`/`list.dat` outputs are unreadable or
readable-but-stale against complete `global.inp`/`phase.bin`/`paths.dat`
handoffs. Source-owned optional outputs such as `nstar.dat` and decomposed
JAS `feffl.bin` are regenerated from the same handoffs when a readable base
cache would otherwise mask the missing or readable stale sidecar. Malformed
declared `global.inp`/`phase.bin`/`paths.dat` handoffs now also block readable
`feff.bin`/`list.dat` caches from being advertised as completed GENFMT output.
Malformed `genfmt.inp` is not advertised during supported-stage discovery,
while direct GENFMT execution still reports the parser error. Full-run source
generation for GENFMT and FF2X now waits when an active FMS stage is not yet
satisfiable from cached outputs or Rust handoffs, preserving the active-Hubbard
`transformation_hubbard.bin` gate before downstream spectra are synthesized.
Malformed `fms.inp` inputs are now treated
as unsatisfied FMS state during discovery, so downstream GENFMT source handoffs
are not reported complete until direct FMS execution can surface the parser
error. FF2X source generation likewise runs the
ordinary FEFF final-spectrum assembly from active-Hubbard
`xsect.dat`/`feff.bin`/`list.dat` handoffs. FF2X cached-output runs now also
fall back to that Rust source assembly when a stale final-spectrum cache such
as `chi.dat` is unreadable but the complete source handoffs are present.
Ordinary EXAFS `chi.dat`/`xmu.dat` caches are now also render-normalized
against the source-generated pair, so readable stale EXAFS final spectra
regenerate before the cached-output path accepts them. Source-owned FF2X
thermal-expansion diagnostics such as `cum.dat` are likewise regenerated from
the path-damping `feff.bin`/`list.dat` handoffs when readable stale sidecars
would otherwise be accepted. Full-run
supported-module orchestration now also carries the EXAFS/Cu `ff2x.inp` plus
`xsect.dat`/`feff.bin`/`list.dat` handoffs through a completed `ff2x` report and
compares scheduler-generated `chi.dat`/`xmu.dat` spectra against the FEFF
reference. The same cached-output runner now checks source-backed non-EXAFS
spectra by regenerating the active source handoff in an isolated scratch workdir
before preserving readable `xmu.dat`/`xmul.dat` caches, so stale XANES and
decomposed NRIXS final spectra route back through the Rust FF2X source
assembler. The
supported-stage predicate now validates final-spectrum cache readability before
advertising FF2X completion; malformed `xmu.dat`/`chi.dat`-style files without
complete source handoffs fall through to the normal source requirement instead
of being reported as a completed cached stage. It also treats malformed
declared `xsect.dat` source handoffs as unsupported during scheduler discovery,
leaving the explicit FF2X runner to report the parser error. Readable cached
final spectra no longer mask those malformed declared source handoffs during
supported-stage discovery. Malformed
`ff2x.inp` is likewise declined during supported-stage discovery, while direct
FF2X execution remains strict. The
FULLSPECTRUM supported-stage predicate now applies the same discipline to
`eps.dat`: the scheduler only advertises cached optical-table generation when
`fullspectrum.inp` enables the module and `eps.dat` parses successfully, so a
standalone malformed dielectric cache no longer appears as a runnable stage. The
same predicate now declines malformed `fullspectrum.inp` during scheduler
discovery, leaving direct FULLSPECTRUM execution to report the parser error.
FULLSPECTRUM discovery now also validates optional source/sidecar state that the
runner consumes (`drude.dat`, `osc_str.dat`, `hamaker.dat`,
`logfullspectrum.dat`, and sum-rule `pot.bin`) before advertising completion,
so a readable `eps.dat` cannot mask malformed declared optical inputs.
FULLSPECTRUM can now also build `eps.dat` without a seeded dielectric cache:
it parses the appended `rdop` option cards, discovers explicit or automatic
`edges/<component>/<edge>` sources, assembles contiguous FPRIME backgrounds
and optional FMS/path fine structure, derives missing component density from
the edge `fms_im/pot.bin`, and adds requested valence and Drude response before
writing oscillator-strength, sum-rule, and optical tables. Source discovery
takes precedence over readable restart caches, while cache-only execution
remains available for FEFF-compatible restarts. The source assembler also
writes FEFF's final fake `xmu.dat`. `CONTROL(6)=0` preserves source-spectrum
output but skips optical post-processing, leaves pre-existing optical files
untouched, and prevents the scheduler from advertising them as new work. The
OPCONS predicate now validates the required `opcons*.dat` source tables before
advertising optical-loss generation, so malformed component tables fall through
to the normal required-stage parser error instead of being reported as completed
supported work. Malformed `opcons.inp` is likewise not advertised during
complete-stage discovery, and malformed declared component sources such as
`pot.bin` now decline scheduler completion instead of aborting discovery; direct
OPCONS execution remains strict. A
scheduler-level `MPSE/Cu_OPCONS` reference gate now starts from `opcons.inp`,
`pot.bin`, and `opconsCu.dat`, omits cached `loss.dat`, and compares the
generated optical-loss table to `REFERENCE/loss.dat`. When an elemental
`opcons*.dat` table is absent, OPCONS can generate it from the bundled FEFF
`epsdb` source for Z=1 through Z=99 before assembling `loss.dat`. The
full-run gate now covers ordinary source handoffs, the ATOMIC-generated `config.dat`
handoff composed into XSPH in the
same supported-stage pass, and the XES screened-core-hole variant that consumes
`wscrn.dat`.
Full-run supported-stage ordering now recovers SCREEN `wscrn.dat` from
`vtot.dat`/`apot.bin` before XSPH source-handoff detection, so recoverable
screened-core-hole XES inputs can generate source-backed `phase.bin` and
`xsect.dat` in the supported-stage pass instead of waiting for the later
required-module retry.
Standalone `refeff module xsph` source generation now uses the same SCREEN
recovery bridge for screened-core-hole inputs: a missing or malformed
`wscrn.dat` no longer blocks source-backed `phase.bin`/`xsect.dat` generation
when valid typed `vtot.dat` and `apot.bin` handoffs can rebuild it, while inputs
without either a usable `wscrn.dat` or that recovery pair still remain behind
the XSPH source requirement.
Partial XSPH phase caches are also scheduled through additive supported
handoffs in full runs, generating `PRINT 2` `phaseNN.dat`/`phminNN.dat`
sidecars plus missing or unreadable `emesh.dat` and `emesh.bin` from
`phase.bin` before the remaining cross-section/phase source requirement stops
the required-module pass. Malformed partial `phase.bin` caches no longer block the
normal-potential source phase handoff: when `pot.bin` and `config.dat` can
regenerate the phase but the full base-output contract is not otherwise
satisfied, full-run orchestration now reports the same `xsph-phase` handoff
instead of waiting for the required-stage retry. The mesh-only branch follows
the same rule now: stale malformed `phase.bin` no longer blocks the initial
`emesh.dat`/`emesh.bin` handoff when `pot.bin` can produce the phase-energy
mesh before the source requirement boundary.
The required full-run XSPH stage now rechecks that the complete
`phase.bin`/`xsect.dat` contract is satisfied after these partial handoffs, so
phase-only compatibility output is preserved without letting later required
stages treat XSPH as complete.
Empty-cell phase-cache recovery now uses the same phase-mesh support gate as
the source generator, so unsupported spectroscopy selectors are not advertised
as supported XSPH stages merely because `xsect.dat` is cached. Cached
`xsect.dat` base handoffs are now also validated against the active
`phase.bin` energy count, main horizontal count, and Fermi index; when
`phase.bin` is regenerated and the normal-potential source `xsect.dat` path is
not applicable, a cached cross-section is preserved and counted only when its
complex energy grid matches the active phase mesh, while mismatched base pairs
stay behind the XSPH boundary instead of being advertised as complete. When the
current normal-potential source handoff can generate
`xsect.dat`, same-shape cached cross sections are now compared against that
source result and regenerated if stale, so branch-changing `global.inp` angular
controls cannot be masked by an older cache. Readable `phase.bin` caches are
also compared against the typed source phase handoff before they are accepted;
stale phase shifts, phase mesh metadata, reference energies, or potential
labels regenerate from `pot.bin`/`config.dat`, and the existing `xsect.dat`
source path refreshes the dependent transition moments. Readable stale
transition moments stored in `phase.bin` also force that source `xsect.dat`
route, so a matching rendered cross-section cannot hide stale downstream
transition data. The supported-stage detector now proves that same
generated-phase, cached-cross-section, and source-result compatibility in
memory before reporting XSPH as a complete base-output stage.
The default XANES path now preserves FEFF's 120-row horizontal phase mesh plus
vertical contour tail and is covered against the Cu reference for mesh shape,
screened-core-hole potential handling from `wscrn.dat`, strict raw phase shifts,
and mixed absolute/relative `xsect.dat` rows. The XES/Cu screened-core-hole
handoff is also covered from the reference zip, including source-backed
`phase.bin`, `xsect.dat`, AXAFS, MPSE, phase-mesh, and `log2.dat` outputs. The
FPRIME/GeCl4 source handoff is pinned against the generated reference for
`phase.bin`, `xsect.dat`, `emesh.dat`, and `emesh.bin`; the generated
`xsect.dat` path now follows FEFF's pure-imaginary FPRIME convention
(`xsec = i*xsnorm`) and does not require MPSE. The EXAFS and XANES Cu gates
remain strict for raw phase shifts; the XES/Cu phase gate uses `1e-4` while
preserving the same cross-section row tolerances. The default
`xsph_module_matches_broader_source_generated_reference_when_present` parity
gate now broadens no-cache source parity across Debye EXAFS/XANES Cu,
`DANES/Cu`, `ELNES/Cu`, `EXAFS/Cu_SCF`, zip-backed `XANES/BN`, and three LDOS
Cu source fixtures, all generating `phase.bin`, `xsect.dat`, `mpse.dat`,
phase-mesh sidecars, and `log2.dat` from `pot.bin` plus the angular-control
handoffs.
Full-run supported-module orchestration now also carries the
`DEBYE/DM/XANES/Cu` source fixture through the completed `xsph` scheduler
report, starting from `xsph.inp`, `global.inp`, `pot.bin`, and `config.dat`
without cached `phase.bin`/`xsect.dat`, and compares the generated phase,
cross-section, and `emesh` sidecars against the FEFF reference.
The scheduler-level `DANES/Cu` XSPH gate now follows that same source-only
pattern plus its screened-potential `wscrn.dat` handoff for the ordinary
`ispec = 3` branch and checks generated `phase.bin`, `xsect.dat`, `emesh.dat`,
and `emesh.bin` against FEFF output.
The zip-backed `XES/Cu` XSPH reference is now promoted to the full-run
scheduler as well: it starts from `xsph.inp`, `global.inp`, `pot.bin`,
`config.dat`, and `wscrn.dat`, omits cached phase/xsect/emesh outputs, and
checks generated `phase.bin`, `xsect.dat`, `emesh.dat`, `emesh.bin`,
`axafs.dat`, and `mpse.dat` before reporting completed `xsph`.
The scheduler-level `ELNES/Cu` XSPH gate now follows the same source-only
pattern and uses the ELNES-specific reference tolerance envelope for
`xsect.dat` while still checking generated `phase.bin`, `emesh.dat`, and
`emesh.bin` against FEFF output.
The `EXAFS/Cu_SCF` XSPH source bundle now has the same scheduler-level
reference gate, extending full-run phase/xsect coverage to SCF-derived
potential handoffs rather than only ordinary Cu source fixtures.
The scheduler-level `FPRIME/GeCl4` XSPH gate now covers the non-Cu
pure-imaginary `xsect.dat` branch from source handoffs and verifies that the
completed `xsph` report does not require `mpse.dat`.
The scheduler now also carries all three LDOS-derived Cu XSPH source-release
fixtures (`LDOS/XANES_Cu_fms`, `LDOS/XANES_Cu_spin_fms_short`, and
`LDOS/XANES_Cu_spin_no_fms`) through completed `xsph` reporting, so the broader
module-gate bundles are covered in full-run orchestration with generated phase,
cross-section, MPSE, and phase-mesh sidecars.

Recent ATOMIC progress includes a shared `refeff-io` refresh adapter for the
`apot.bin` section-5 core-hole Coulomb payload: `dvcoul` is regenerated from the
persisted `drho` density column with the ported FEFF four-point Coulomb
transform, while no-hole runs continue to require zero core-hole density and
regenerate a zero core-hole Coulomb column. The ATOM CLI path uses that adapter
and also regenerates `config.dat` from `pot.inp`/`config.inp` before the
remaining `apot.bin` solver boundary, matching FEFF `DumpConfig2` occupation
output for covered references. Full `refeff run` orchestration and standalone
`refeff module atomic` runs now also run that source-backed `config.dat` handoff
directly from RDINP-generated `pot.inp`/optional `config.inp`, even before a
compatible `pot.bin` exists. Pre-existing `config.dat` files whose potential
metadata matches `pot.inp` are re-rendered through the same typed writer; stale,
malformed, or unreadable `config.dat` sidecars are repaired for cached
`apot.bin` runs and pre-solver `atomic-config` handoffs without treating the
missing APOT source path as complete. Malformed `apot.bin` caches no
longer suppress that source-backed `config.dat` handoff, while standalone
malformed atomic-potential caches still fail validation when the required ATOM
stage reaches the `apot.bin` boundary. Downstream XSPH/POT-style source routes
still require compatible `pot.bin` potential data before using the generated
configuration with potential-dependent handoffs. The
typed `fpf0.dat` form-factor handoff is now regenerated from `apot.bin` and
ATOM input metadata when the sidecar is missing, unreadable, or
structurally stale against the source absorber/oscillator/form-factor shape.
Full-run orchestration now has explicit reference-backed coverage that a
malformed stale `fpf0.dat` is repaired from that source handoff before the
POT source/cache requirement.
The same source-backed ATOM sidecar path now regenerates malformed `log1.dat`
wrappers when `config.dat`/`fpf0.dat` are repaired from typed inputs and no
usable cached POT stage is sharing that wrapper; malformed or incomplete POT
caches no longer suppress that ATOM-side repair, while purely cached malformed
`log1.dat` files remain strict validation failures for ATOM-only paths.
Config-only handoffs now also repair an existing malformed `log1.dat` while
still avoiding new `log1.dat` creation for clean pre-solver `config.dat`
validation. Full atomic-potential generation is now source-backed for ordinary
`pot.inp` plus `geom.dat` handoffs; missing geometry and finite-nucleus inputs
remain explicit source gaps.
Malformed `pot.inp` is declined during ATOMIC cached-output, source-apot, and
config-handoff discovery, leaving explicit ATOMIC execution to report the
control-input parser error.
The ATOM core now also composes the ported `etotal.f90` total-energy algebra
with the ported `fdrirk.f90` radial-integral driver in
`atomic_total_energy_from_radials`, including FEFF's previous-first-factor
sentinel state. This provides the file-level ATOM driver with a single
source-backed total-energy entry point for the `fpf0.dat` refresh path, but
does not by itself complete `apot.bin` generation.
The ATOM core also now exposes `atomic_dirac_bound_orbital`, a composed
`soldir.f90` driver over the ported `intdir`, matching, node-search,
energy-correction, and normalization helpers. This gives the upcoming
`scfdat` port one Rust call for each bound-orbital solve while the production
`apot.bin` writer remains gated.
That driver is now used by `atomic_initial_orbitals`, a Rust port of
`wfirdf.f90` that builds the ATOM logarithmic radial mesh, point/finite nuclear
potential, Thomas-Fermi starting potential, origin powers/scales, and starting
`cg/cp/bg/bp/en/nmax` orbital tables for `scfdat`.
Those tables now feed `atomic_self_consistent_orbitals`, which ports FEFF's
positive-`niter` `scfdat.f90` active-orbital scheduler, optional active
`lagdat` refresh, convergence selection, and final total/valence density
recomputation. The returned state also carries FEFF's final `srho/r**2`,
`srhovl/r**2`, and `vcoul = potslw(srho) - Z/r` tables for the upcoming
`apot.bin` section assembly. The `atomic_scf_state_from_configuration` driver
now composes the production ATOM state chain from a compacted `getorb`
configuration through `inmuat`, `wfirdf`, Coulomb angular coefficients, and
positive-`niter` `scfdat`, giving CLI code one source-backed numerical entry
point for each atomic state. The `refeff-io` `apot_atomic_scf_sections`
adapter now translates one or more converged SCF states into the FEFF section
numbers consumed by current ATOM source handoffs: merged `norb`, `rho`,
`rhoval`, `vcoul`, `xnval`, `eorb`, and `kappa` tables, plus FEFF-ordered
per-state `dgc`/`dpc`/`adgc`/`adpc` orbital matrices. Its borrowed-state
helpers can now populate those sections directly from core `AtomicScfState`
values, so callers do not need to rebuild IO array views by hand. The
ATOM CLI now has a staged `generated_atomic_scf_apot_bin` helper that derives
FEFF state-column configurations from `pot.inp`/`config.inp`, runs the core
SCF driver, and emits the source-backed SCF `apot.bin` section subset. The
driver now mirrors FEFF `scfdat`'s `xnvalp` selection for Coulomb angular
coefficients (`idfock=1` uses zero valence; `idfock=2` uses total
occupations; separated branches use actual valence occupations), which keeps
multi-orbital and K-edge core-hole source states on the fast convergence path.
The CLI source layer also derives the FEFF section-21 `iorb(-5:4,0:nph+1)`
matrix from the same compacted `getorb` configurations, including final-state
screening-orbital projections, and now derives section-5 core-hole columns
(`dgc0`, `dpc0`, `drho`, `dvcoul`) from the generated SCF states. The
section-5 branch follows FEFF `apot.f90`: no-hole runs keep zero
`drho`/`dvcoul`, `NOHOLE=1` uses the initial core orbital density, and
transition-state runs use the initial/final absorber core-density difference.
The same staged CLI layer now derives APOT static source arrays from typed
`pot.inp` and `geom.dat` handoffs, with optional `pot.bin` cross-checks when
that self-consistent state is already present: `iz`, `iatph`, `novr`, `rnrm`,
`iphat`, Bohr-scaled `rat(3,nat)`, and manual overlap shell matrices
`iphovr`, `nnovr`, and Bohr-scaled `rovr`, while checking that available
handoffs agree on potential counts and atomic numbers. It also derives the
FEFF `ovrlp` APOT arrays from generated SCF state columns and those static
geometry/overlap inputs: source-backed `edens`, `edenvl`, `vclap`, Norman
radii, and the current spin-unpolarized `dmag/edens` table, preserving both
explicit `OVERLAP` shells and geometry-neighbor mode plus FEFF's `edenvl`
neighbor-density convention. The APOT source layer also derives `xnvmu` from
the same compacted `getorb` configurations as `xnval` and `iorb`, using
FEFF `scfdat`'s kappa-to-angular-channel accumulation for `l=0..3`. It now
derives the `s02` scalar from generated initial/final absorber SCF states by
reconstructing FEFF's relaxed-overlap matrix and feeding the existing Rust
`s02at` port, and derives `erelax`/`emu` from the generated absorber total
energies, frozen initial-state core orbital energy, and FEFF's
`vcoul(1,0)-vclap(1,0)` overlap shift.
The CLI now has a staged full-APOT assembly helper that feeds those source
arrays into the higher-level `apot_atomic_pots_sections` adapter, producing the
complete `WriteAtomicPots`-ordered section stream: scalar, core-hole, geometry,
overlap, overlapped-density, `iorb`, and per-state orbital sections. The APOT
overlap source path now also handles isolated no-overlap generated densities
whose FEFF `frnrm` charge integral lands just below `Z`, normalizing only that
isolated density column within a tight charge tolerance before the Norman-radius
search. The ATOM CLI gate now uses that full stream for complete typed source
handoffs: when `pot.inp` and `geom.dat` are present, `refeff module atomic`
and full-run orchestration generate or replace `apot.bin`, then continue
through the existing `config.dat`, optional `fpf0.dat`, and shared `log1.dat`
handling. If `pot.bin` is present, the source path still validates it against
the typed inputs before using the older pot-backed static-array route. Missing
geometry remains behind the explicit ATOM gate. Finite-nucleus inputs now reach
the source-backed ATOM APOT stream from `pot.inp` plus `geom.dat` without a
cached `pot.bin`, and that direct APOT source-generation path is release-gated.
The finite-nucleus source state also has release-profile core gates for FEFF
`nucdev` point/finite nuclear-potential behavior and the composed atomic SCF
state's finite-nucleus request path; broader finite-nucleus reference parity
remains follow-up.
The ATOM CLI now has an `apot.bin`-backed total-energy candidate path that
extracts orbital counts, occupations, bound-state energies, kappa values, and
large/small radial components from typed ATOM sections before calling that
shared core helper. The `fpf0.dat` refresh now uses that source-backed total
energy for EXAFS/Cu, XANES/Cu transition, and NRIXS/GeCl_4 `NOHOLE >= 0`
reference paths without `fort.16`, including FEFF's Breit-only large/small
radial mode, the `idfock=1` zero-valence `xnvalp` total-energy pass, and the
FEFF column choice that uses the final absorber column for normal core-hole
runs but absorber column zero for `NOHOLE >= 0`. Full `apot.bin` generation is
now source-backed for ordinary and finite-nucleus geometry handoffs; remaining
ATOM-gate work is broader finite-nucleus/generated-reference parity.
Recent exchange-kernel progress also includes the `EXCH/xcpot.f90` MPSE
Wigner-Seitz density-grid setup used by the many-pole self-energy path and the
MPSE enable/pole-count setup, MPSE delta-self-energy table shaping around
`CSigZ`, MPSE row-delta selection, local density/momentum scale setup, the
nested `sigma` dispatcher, Fermi-level self-energy cache setup, Dyson
self-energy correction, self-energy delta application, the composed dynamic
`xcpot` potential update for non-MPSE, prepared-MPSE, and the current non-BPR
computed `CSigZ` MPSE production path, plus the early ground-state/static
potential branch and final
`eref = v(jri1)` reference potential shift.
Recent self-energy progress includes the FEFF `SELF/csigz.f90` paths used by
MPSE: FEFF-backed Rust ports of `Sigma1`, `dSigma`, and `CSigZ` many-pole
accumulation with Hartree-Fock exchange and `ZTot` renormalization. The
broadened-pole BPR `UseBP = .TRUE.` branch is now source-backed in
`refeff-core`, including direct `bpr1`/`bpr2`/`bpr3` integrand fixtures and a
full `Sigma1`/`CSigZ` reference gate. The typed `xcpot` MPSE self-energy input
now forwards an explicit `UseBP` selector into that `CSigZ` branch while keeping
the ordinary XSPH/EXCH production path at FEFF's hard-coded non-BPR default.
Recent POT orchestration progress wires the existing `pot.bin`/`apot.bin`
handoff bridge into full FEFF runs as the `pot` stage instead of only running
the lower-level `wpot` compatibility writer. The cached POT stage now also
preserves or regenerates the FEFF `log1.dat` SCF-potential module wrapper
through a shared `refeff-io` log adapter, including missing or unreadable
POT-owned wrappers, so full cached runs do not leave the earlier ATOM
`log1.dat` stand-in in place after cached self-consistent potential state is
consumed. The POT runner can now also rebuild a missing or unreadable
`apot.bin` sidecar from complete typed ATOM source handoffs (`pot.inp`,
`pot.bin`, and `geom.dat`) before consuming the existing self-consistent
`pot.bin` state through `wpot`; this keeps standalone `refeff module pot` from
requiring cached free-atom APOT output when the source bundle is complete.
The POT runner can now also generate a typed no-SCF `pot.bin` for the
default `EXCHANGE 0` and `EXCHANGE 2`, `nscmt=0` source subset from
`pot.inp` plus `geom.dat`: it
reuses the ATOM SCF state generator, including normal core-hole/final-state
mode rows, the ported `ovrlp`/Norman-radius density path, static
Perdew-Zunger exchange, `sidx`, and for multi-potential inputs the ported
`movrlp`/`ovp2mt` muffin-tin projection to estimate the interstitial
potential/density before feeding the existing APOT sidecar and `wpot` writer.
Malformed final `pot.bin`/`apot.bin` caches no longer suppress that source
route when the no-SCF handoff is complete; readable stale no-SCF final caches
are now compared against the generated source payload and refreshed before
`wpot` renders `potNN.dat`. The cached-output predicate uses the same
render-normalized source comparison before advertising no-SCF POT as
cache-complete, so readable stale `pot.bin` or `apot.bin` files are scheduled
as source-repairable instead of being classified as valid cached output;
standalone malformed caches still fail through the cached POT runner.
Full-run orchestration now also pins stale-cache preference for terminal
iterative SCF POT source runs: stale readable `pot.bin`/`apot.bin` outputs are
rebuilt from `pot.inp` plus `geom.dat` before the stage is reported as
completed `pot`, rather than being preserved as cached output or downgraded to
`pot-scf-source`. The direct POT runner and full-run scheduler now also pin
that rule for compatible `EXTPOT` MTDP/`sort.aip` source handoffs, regenerating
readable stale terminal SCF `pot.bin`/`apot.bin` final files before accepting
completed POT output.
Malformed external `sort.aip`/MTDP source handoffs are also declined during
cached POT discovery, so readable final caches no longer mask a broken active
`EXTPOT` source bundle.
Malformed `pot.inp` is also declined during POT cached-output, source, SCF,
and input-handoff discovery, so full-run supported-stage scanning does not
report `pot`, `pot-input`, or `pot-scf-source` from an unparsable control
file; explicit POT execution still reports the parser error.
Malformed declared `geom.dat` source handoffs are also declined during cached
POT discovery, so readable final `pot.bin`/`apot.bin` outputs no longer mask a
broken active geometry source.
Malformed custom `config.inp` source handoffs now follow the same
discovery/explicit-run split for ATOMIC and POT: supported-stage discovery
declines them, while direct execution reports the configuration parser error,
and readable final caches no longer mask the broken custom configuration input.
Unsupported no-SCF source selectors such as a hand-written `iscfxc=0` are now
treated as non-advertised source branches for those predicate checks, so a
valid standalone final cache is preserved instead of failing during stale-source
comparison. Full-run supported-module orchestration now covers the same
unsupported-selector cache-preservation branch, reporting the preserved final
cache as completed `pot` output rather than a source handoff. The matching
no-cache module regression now also keeps that unsupported selector out of all
POT source-discovery predicates when `pot.inp`/`geom.dat` are present but no
final POT cache exists.
The EXAFS/YBCO no-SCF reference is now pinned at the POT module boundary as a
five-potential source-output gate: `pot.inp` plus `geom.dat` writes
`pot.bin`, `apot.bin`, `pot00.dat` through `pot04.dat`, and `log1.dat`, with
the generated potential identities and multiplicities checked against FEFF's
archived `pot.bin`. Full-run supported-module orchestration now carries the
same YBCO no-SCF source handoffs through the completed `pot` scheduler report,
so that five-potential route is pinned outside the standalone module wrapper as
well.
The EXAFS/SF6 no-SCF molecular reference is now pinned the same way at the
full-run scheduler boundary: `pot.inp` plus `geom.dat` source handoffs produce
`pot.bin`, `apot.bin`, both `potNN.dat` files, and `log1.dat` as a completed
`pot` report without final-output caches.
Generated no-SCF `pot.bin` state keeps FEFF reference parity by deriving
`xnatph` from `geom.dat` atom counts, while generated initial-SCF `pot.bin`
state preserves FEFF/RDINP `pot.inp` multiplicities so positive-`totvol`
true-SCF fixtures keep a positive overlap-corrected interstitial volume.
Single-potential inputs still use the direct `istval`/`fermi` shell average.
Iterative SCF inputs with complete `pot.inp`/`geom.dat` source handoffs now
also build the in-memory initial SCF `pot.bin` state, apply the Rust-backed
`istprm` interstitial/FERMI setup into a `PotScfState` snapshot, and validate
that it feeds the ported FEFF `broydn` plus `coulom` density/coulomb update
adapter before the source-row contour/scattering loop; this preflight does not
write a final `pot.bin`, `apot.bin`, `log1.dat`, or `potNN.dat`, and it does not
mark POT complete. The initial snapshot now also carries the Rust-backed
`POT/grids.f90` SCMT `emg`/floor-step schedule and a strict all-potential
FOVRG source-grid handoff packed as `(energy, potential, l, radial)` with wave
numbers and `phamp` phase tables. The FOVRG retained photoelectron length now
extends when needed to cover the muffin-tin match point plus the six-row inward
integration history, so large-radius POT SCF rows no longer stop at the
10-bohr retained-length cap. Compatible FOVRG rows now also feed a POT `fmsie`
source-grid bridge: Rust derives the nonrelativistic `sqrt(2*(em-eref))` FMS
wave number, solves the spin-free zero-Debye-Waller FMS cluster from `geom.dat`,
projects all-potential `gtr(l,iph)` traces, and records unavailable reasons
when compact radial rows cannot be built. The generated radial/FMS grids now
also feed the Rust `pot_scf_contour_source_rows` bridge in the POT preflight,
validating the handoff into `xrhole`, `xrhoce`, `yrhole`, and `yrhoce` source
rows before the live SCF loop is claimed. The CLI now also follows FEFF's POT
SCMT contour sizing (`negx=80`, `nflrx=17`) and adaptive contour energy walk
when generating source rows: it solves one FOVRG/FMS/rholie row for the
current SCMT energy, appends that row to the prefix, and reruns the contour
driver until a bracket/terminal status or the finite dynamic source-row guard
is reached; unlike the previous preflight, this guard is no longer tied to
FEFF's static `emg(1:neg)` table. The current Be iterative fixture now reaches
a final-pot status from those generated rows instead of stopping on stale
static-grid energies or an adaptive-source boundary. Generated
positive-`totvol` POT state now preserves FEFF's converted
total-volume scalar in `pot.bin` while keeping the reduced interstitial volume
inside the overlap/projection math. The GeCl4 true-SCF plus SF6, YBCO,
XMCD/MnF2, and XMCD/Gd L1 no-SCF POT reference gates now compare generated
`pot.bin` radius, overlap, and density rows against FEFF output at both the
module and full-run scheduler boundaries. The HUBBARD/NiO screened-core-hole
true-SCF gates now also cover the standalone module and full-run scheduler
final-output routes and compare generated radius, overlap, and electron-density
rows against the archived full FEFF reference. Optional bounded FEFF gates now
scan `reference-work/tmp/feff-pot-nio-bounded.*/pot.bin`, with
`REFEFF_NIO_BOUNDED_FEFF_POT_BIN` as an override, and compare the same
two-iteration NiO run at the module and full-run scheduler boundaries. Those
gates confirm full `pot.bin` row parity, including carried `edenvl`
valence-density rows. Bounded XANES/BN positive-`totvol` tests still pin
`electron_density[27]`, carried `edenvl`, and the one-iteration POT output set
as focused diagnostics. The canonical acceptance path now uses the stock
fresh XANES/BN controls and completes through `xmu.dat` at approximately
`1–2e-5` relative L2 parity. That closure keeps independent-center FMS data in
slot zero, carries saved SCMT state across retry starts, and freezes FEFF's
initial `sqrt(rhoint)` plasmon value. The Rust `corval` path keeps RDINP's
`ecv` as an eV module-input field and converts it to Hartree before comparing
with orbital energies, preventing deep core levels from entering the LDOS peak
request mask.
The core density layer now also
exposes a POT-facing `rholie` post-radial-solver bridge that preserves complex
`xrhole`, `xrhoce`, `yrhole`,
and accumulated `yrhoce` work arrays on FEFF's 0.05 radial grid before the
existing `ff2g` valence integration adapter. A one-energy POT density
subdriver now composes that bridge with `ff2g`, leaving incomplete live
iterative-loop and unsupported exchange-selector cases behind the POT
source/cache requirement. The FEFF `scmt` Fermi
end-cap correction after contour bracketing is also Rust-backed now: the core
density layer refines the endpoint interpolation fraction, returns `xmunew`,
and applies the matching final corrections to `xnmues` and `rhoval`. The
adjoining `scmt` contour-search transition is Rust-backed too: it consumes
FEFF's prebuilt `emg` grid, starts horizontal Fermi search, moves up/down
between floors, and reports the lowest-floor bracket that feeds the endpoint
correction. The per-energy all-potential `scmt` loop around `ff2g` is also a
core helper now: Rust resets `xntot`/`fl`/`fr`, folds each potential's supplied
`rholie`/FMS work arrays through `ff2g`, and returns the updated `xrhoce`,
`xrhocp`, `yrhoce`, `yrhocp`, `rhoval`, and `xnmues` state for the next
contour point. A source-row `scmt` contour-loop driver now composes those
pieces: it copies `xrhoce`/`yrhoce` to previous-point state, consumes supplied
radial/FMS work arrays in FEFF loop order, tracks `xndif`, delegates the
up/down floor search, and applies the endpoint correction once the
lowest-floor bracket is reached. The source-row adapter in front of that loop
is now Rust-backed too: it lifts solved radial channels and FMS `gtr` tables
over all supplied contour rows/potentials into the `xrhole`, `xrhoce`,
`yrhole`, and `yrhoce` tables consumed by the contour driver. The remaining
SCF iteration handoff is Rust-backed as well: completed contour brackets run
through FEFF's occupation-count repeat check, the existing `broydn`/`coulom`
adapter, and the `edens = edens - edenvl + rhoval` update with inactive tails
zero-filled. The outer `potsub` convergence transition is ported too: Rust
applies `nscmt_min`, `tolmu`, `tolq`, `tolsum`, and `tolqp`, either restores
pre-`scmt` density/potential state for convergence or iteration-limit exit,
or copies mixed `rhoval` into `edenvl` for the next `istprm` pass. A
state-advance wrapper now carries `xmu`, `qnrm`, `qold`, `xnmues`, `edens`,
`edenvl`, `vclap`, and Broyden workspace across supplied SCMT iterations
for the explicit source-backed SCF loop driver. The first-call
`istprm` muffin-tin radius setup is now Rust-backed too: it computes `rmt`,
`inrm`, `lnear`, nearest-neighbor bookkeeping, explicit-`OVERLAP`
`inters mod 6` normalization, and `folpx` reductions before the existing
`movrlp`/`ovp2mt` helpers consume that state. The second `istprm` block is now
composed in Rust as well: `sidx` tail adjustment, `iscfxc` ground-state XC
selection, `vtot`/`vvalgs` construction, FEFF `volint`/`rnrmav`,
`movrlp`/`ovp2mt` projection, the `vint >= xmu` fixed-potential retry, and
final `fermi` state all run through a core helper. The CLI now feeds generated
contour rows into that state-advance wrapper for the initial SCF preflight when
the radial and FMS source grids are available, and prepares a following
non-converged iteration by copying the post-`scmt` `edens`/`edenvl`/`vclap`
state through the Rust `istprm` helper while preserving FEFF's SCMT Fermi
level, then attempting the next radial/FMS/contour source-row bundle from that
prepared state with the same success-or-reason handoff used for the initial
pass. When those source rows are available, the prepared bundle is also advanced
through the `iscmt=2`/`first_scmt_call=false` SCMT wrapper. The CLI now wraps
those source-backed advances in an explicit SCF loop driver, continuing across
prepared iteration bundles until convergence, iteration limit, or a
missing-source boundary. Terminal convergence or iteration-limit states now
materialize a validated in-memory final `pot.bin` candidate by overlaying the
SCF state (`xmu`, `qnrm`, `xnmues`, `edens`, `edenvl`, `vclap`) onto the latest
`istprm` POT snapshot, with release-profile coverage that repeat,
missing-source, and non-converged statuses cannot enter that final-output path.
The POT runner now attempts that terminal candidate
before the loop-only handoff; when available it writes final `pot.bin`,
regenerates the `apot.bin` sidecar through the existing ATOM APOT source path,
and continues through `wpot`/`log1.dat`. Readable SCF final
`pot.bin`/`apot.bin` caches are now render-normalized against that generated
terminal source payload before POT is advertised as cache-complete, so stale
or incomplete final pairs route back through the SCF source writer instead of
the APOT-only sidecar handoff. Stock iterative SCF tests now cover clean-cache
successful and iteration-limit terminal output plus retry-state completion;
additional convergence fixtures are non-blocking numerical broadening. SF6,
YBCO, MnF2 XMCD, and Gd L1 no-SCF reference parity is release-covered at both
module and full-run scheduler boundaries. Adaptive source-row advancement now accumulates
the generated FOVRG/FMS grids alongside the contour rows and returns that state
at terminal or guarded boundaries, avoiding a second full source-grid rebuild
after the contour search has already generated each row. It also batches FEFF's
deterministic prefix rows (`nflrx` rows for the first `scmt` call, `neg` rows
for repeat calls) before falling back to one-row dynamic horizontal-search
extension. The initial SCF `corval` LDOS peak scan now uses a narrow
request-mask handoff that solves only suspicious `(l, potential)` channels and
returns the embedded `xrhoce` rows needed by the first-peak detector, instead of
building full SCF contour rows solely for the core/valence boundary correction.
The SCF source-loop retry wrapper now follows FEFF's bounded `nstarts` path by
carrying the `corval`-adjusted `ecv` into retries, jumping to the final start
when the adjustment changes by less than 0.05 Ha, and reducing
`ca1=max(ca1/5,0.01)` on the same reduced-mixing branch as `potsub.f90`; the
release gate now also checks that the max-start attempt cannot advance again.
The core POT SCF release gate now also pins FEFF contour stepping, endpoint
finishing, source-row lifting, and density/coulomb outer-iteration composition
through `cargo test --profile release -p refeff-core pot_scf`.
The matching CLI release gate `cargo test --profile release -p refeff-engine pot_scf`
now pins SCF source-loop initial-state construction, contour advance, next
iteration preparation, FMS source-grid assembly, and full-run reference POT
output from source handoffs.
Adaptive SCF now also prepares a reusable POT FOVRG source-grid plan once per
`scmt` attempt and reuses it for the deterministic prefix plus dynamic
horizontal-search rows. The adaptive source-row guard now also covers the
finite-nucleus Be `HIGHZ` smoke contour far enough to reach a floor-1 SCMT
bracket before the remaining bounded repeat boundary, and full `refeff run`
orchestration now verifies that RDINP carries `HIGHZ` into the
`pot-scf-source` repeat-boundary handoff. The finite-nucleus direct-module
release gate now also pins that repeat-exhaustion boundary as a non-final
source-loop outcome after bounded FEFF-style start attempts. The screened-core-hole
NiO reference gate now runs two source-backed SCMT passes before writing final
`pot.bin`/`apot.bin` outputs, and the module plus full-run scheduler gates
compare the generated geometry and electron-density rows to the FEFF reference,
so that multi-potential core-hole path is covered beyond the earlier
terminal-output case. A local ignored bounded FEFF `nscmt=2` POT artifact under
`reference-work/tmp/feff-pot-nio-bounded.*` confirms that the generated
`edenvl` valence-density rows also match FEFF for the same bounded run through
optional module and full-run scheduler gates; `REFEFF_NIO_BOUNDED_FEFF_POT_BIN`
can point at an alternate artifact. The archived NiO reference remains a longer
10-iteration converged run and is only used for bounded geometry and
electron-density checks. The single-potential regular core-hole Be smoke
fixture now reaches a bounded two-pass `ReachedIterationLimit` output with
renderable `pot.bin`/`apot.bin` candidates after the `ecv` eV-to-Hartree
correction keeps the synthetic core/valence boundary out of the repeat-retry
path. The POT FOVRG adapter now solves independent per-potential `(energy, l)`
`rholie` channels in parallel and then fills the source-grid arrays in
deterministic order. Full-run scheduler coverage now also pins the LDOS spin Cu
true-SCF final-state-screening fixture as completed `pot` output, so that
two-potential source route writes renderable `pot.bin`/`apot.bin`, `potNN.dat`,
and `log1.dat` outside the standalone module wrapper. The
shared FOVRG solver now also
short-circuits C3 potential-vector construction when FEFF's `ic3` scale is
zero, keeping the disabled-C3 numerical path unchanged while avoiding unused
derivative work for POT/RHORRP s-wave channels. POT source-grid and CORVAL LDOS
handoffs now also precompute nonzero C3 vectors once per potential/angular
channel and reuse them across contour energies and regular/irregular solves.
Standalone `refeff module pot` and full `refeff run` orchestration now also
validate the typed RDINP-generated `pot.inp` source handoff before the final
source/cache boundary, reporting `pot-input` or the consolidated
`pot-scf-source` SCF source-driver outcome in full-run summaries when the
driver stops at a non-terminal repeat boundary. Terminal SCF source-driver
outcomes are now promoted to completed `pot` scheduler reports after
`pot.bin`/`apot.bin` become renderable. The XANES/BN positive-`totvol`
fixture has both focused bounded POT diagnostics and the canonical fresh
full-run parity gate; the latter is the completion evidence and compares the
stock Rust and pinned-FEFF `xmu.dat` outputs.
`full_run_regenerates_stale_high_exchange_scf_pot_from_rdinp_sources_before_xsph_error`
now also pins the full-run route for an RDINP high-`EXCHANGE` SCF source bundle
after both readable `pot.bin` and `apot.bin` have gone stale, preserving the
separate valence-potential branch before the expected downstream XSPH source
boundary instead of falling back to a loop-only `pot-scf-source` report. The
direct POT module runner now has the matching stale high-`EXCHANGE` final-cache
regression for `EXCHANGE 5` iterative SCF source handoffs, plus direct
`EXCHANGE 6` no-SCF generation and stale `pot.bin`/`apot.bin` regeneration from
`pot.inp` plus `geom.dat`. The
direct POT runner now applies the same source validation before it reports the
source/cache requirement and only writes `log1.dat` when the source driver
reaches terminal POT output.
POT source readiness also parses `geom.dat` before scheduler accounting, so a
malformed geometry handoff is not reported as completed `pot`, `pot-input`, or
`pot-scf-source` work while the explicit required runner still surfaces the
geometry parser/validation error.
Full-run orchestration now also covers the RDINP high-`EXCHANGE` iterative SCF
branch, preserving the generated valence/total-potential distinction through
the completed `pot` scheduler report before the later XSPH source boundary is
reported.
Compatible iterative `EXTPOT` full-run branches are now covered too: a
compatible MTDP/`sort.aip` handoff, both standalone and paired with a compatible
`RESTART` `pot.bin`, advances through the source driver to completed `pot`
output. The standalone external-MTDP route also regenerates readable stale
`pot.bin`/`apot.bin` final files from the unchanged MTDP/`sort.aip` source
handoff before accepting the completed `pot` report, while the incompatible
external-restart fixture still stops at the explicit non-final `pot-scf-source`
boundary.
Full runs now report the POT source/cache requirement when upstream atomic
caches exist but self-consistent potential caches or complete source handoffs do
not.
Recent SCREEN progress includes source-backed refresh paths around cached
screened-core-hole output: a shared `refeff-io` adapter regenerates `vtot.dat`
from `wscrn.dat` plus the absorber total-potential column in `pot.bin`, now
including missing, stale, or unreadable `vtot.dat` sidecars. The cached CLI
path can also recover a missing `wscrn.dat` from `vtot.dat` plus `apot.bin` by
reusing
the `vtot.dat` screened-potential column and regenerating the bare core-hole
column `v_ch(r)` from `apot.bin` section-5 `dgc0`/`dpc0` with the ported FEFF
bare-core-hole potential helper. The same recovery now rewrites malformed or
unreadable `wscrn.dat` files when those typed `vtot.dat` and `apot.bin`
handoffs are valid. Readable `wscrn.dat` caches are also compared against the
same recovery handoff when `vtot.dat`/`apot.bin` are complete, so stale screened
potential columns cannot be accepted just because `pot.bin` could regenerate a
matching `vtot.dat` sidecar from the stale table. Full `refeff run`
orchestration now reports that
`vtot.dat`/`apot.bin` recovery as the validation-only `screen-wscrn` handoff
when a usable cached `wscrn.dat` stage is absent, without writing
`logscreen.dat`; required-stage orchestration now uses that wrapper to
distinguish completed cached SCREEN output from the pre-solver handoff.
Source-backed SCREEN writes from the explicit cached-output path still
regenerate malformed `logscreen.dat` wrappers, while cached-only malformed logs
remain strict validation failures. Malformed declared `screen.inp` source
handoffs are now unsupported during SCREEN scheduler discovery, while explicit
SCREEN execution still reports the parser error. Full `refeff run`
orchestration has
regression coverage for that recoverable SCREEN handoff, including the
cross-stage route where recovered `wscrn.dat` feeds source-backed XES XSPH
generation before the remaining required-stage solver boundary.
Readable SCREEN `wscrn.dat` caches are now also compared against complete
source-response bundles before cached-stage acceptance, so stale screened
potential rows regenerate from `pot.bin`/`config.dat` plus FMS source state
instead of masking the Rust SCREEN driver. Malformed declared source bundles
such as a bad `phase.bin` now decline SCREEN supported-stage discovery while
the explicit runner still reports the source parser/setup error.
The solved `SCREEN/screensub.f90` tail can now also form `v_ch` from
`dgc0`/`dpc0`, solve `(I-K*Im(chi0))*wscrn=v_ch`, and produce a FEFF-compatible
`wscrn.dat` handoff for already-assembled response kernels and
susceptibilities. The shared `refeff-core` contour helper now integrates
per-energy upper-triangle response slices into symmetric `chi0r`, and the
`refeff-io` `wscrn.dat` adapter accepts those slices directly. The core SCREEN
response path now also composes each FEFF atomic upper-triangle angular-channel
slice with the FMS cluster correction, then sums angular channels into one
per-energy `chi0re` slice while preserving the stored upper-triangle convention
used before contour integration. That per-energy assembler now lifts over the
complex-energy contour to produce the full `chi0re(:,:,ie)` response-slice cube
consumed by the existing contour integral and `wscrn.dat` adapters. The
`refeff-io` SCREEN handoff layer can now also derive the FMS
`gtrl(energy,l)` trace table from `phase.bin` absorber phase shifts and
`gg.bin` scattering sections, including the FEFF positive signed-`l` phase-slot
selection and the squared magnetic-substate matrix-order guard needed by
SCREEN. Direct SCREEN module execution now validates complete `phase.bin` plus
`gg.bin` FMS trace bundles through that adapter before the remaining SCREEN
source requirement, so malformed FMS source state is no longer hidden by the
generic SCREEN fallback. The same direct path now validates the `screen.inp`/`pot.bin`
potential-kernel handoff, including radial bounds, RPA/TDLDA local-field
kernels, Coulomb response kernels, and bound core components. A new typed
response-assembly handoff consumes regular/irregular radial solution cubes,
those FMS traces, and the potential/kernel handoff, then runs the Rust
per-energy response-slice assembly, contour integration, and screened-core-hole
linear solve to produce `wscrn.dat` data. The core SCREEN radial layer now also
assembles one angular channel from raw regular/irregular `dfovrg` output by
applying FEFF `xfnorm`, Wronskian irregular scaling, and optional exact
free-particle tail replacement, and lifts that channel helper over the
production `(energy, radial, l)` layout to build response-ready radial cubes.
Exact-tail generation now evaluates the ported FEFF Bessel/Neumann/Hankel
helpers at `ck*r` for each tail row before feeding the same channel/cube
assembly. A one-channel SCREEN wrapper now also runs the prepared regular
FOVRG pass, computes and injects the irregular muffin-tin boundary condition,
runs the irregular FOVRG pass, and feeds both raw solutions through that
exact-tail assembly. A contour-grid wrapper now loops those prepared FOVRG
channel inputs in FEFF `(energy,l)` order and packs response-ready
`(energy, radial, l)` radial cubes. Its matched variant recovers `phamp`
phase shifts and amplitudes from the regular FOVRG pass instead of requiring
an external phase-amplitude table. The `refeff-io` SCREEN radial handoff now
prepares absorber FOVRG solver grids from `pot.bin`/`config.dat` plus the
SCREEN energy/reference state, runs that matched cube helper, and returns
response-ready radial cubes and recovered `phamp` phase tables. Direct SCREEN
module execution now composes those radial cubes with the source FMS,
potential-kernel, and inline SCREEN/FMS handoffs and writes
`wscrn.dat`/`vtot.dat` without cached screened-potential output. The inline
source-grid FMS path now derives SCREEN `getph`/`phamp` phase shifts across
potentials, solves the spin-free FMS trace table without `gg.bin`, and keeps
the cached `gg.bin` adapter as a fallback. The non-absorber phase-grid route
now uses a phase-only FOVRG handoff that stops after the regular solve and
`phamp` recovery instead of running unused irregular radial cubes. The typed
`pot.bin` and `config.dat` readers now also normalize older FEFF text/PAD
reference caches that store 30/29 orbital records into the current FEFF10
41/40-slot internal shapes, including legacy eight-slot `iorb` records, so
archived Cu source handoffs reach the SCREEN numerical path instead of failing
at parser shape checks. The Cu source-runtime path now also carries the first
FMS hot-path fixes: single-precision complex LU uses flat factor/solve buffers,
and FMS `g0` assembly validates table shapes outside the hot state-pair loop
while preserving FEFF Fortran-order output. The SCREEN inline FMS bridge now
packs only the absorber scattering block it consumes, and the `g0` fast path
caches the triangular normalization/weight table used by `xgllm` outside the
state-pair loop. The source SCREEN driver now follows `prep.f90` by building
the contour from `screen.inp` and `xmu` instead of reusing the XSPH
`phase.bin` mesh, and the FOVRG handoff uses the prepared absorber `eref(1)`
reference potential for every contour point. The default
`screen_module_matches_no_cache_inline_fms_generated_reference_when_present`
parity gate now runs six Cu-family source bundles (`DANES/Cu`, `ELNES/Cu`,
three LDOS Cu fixtures, and `XANES/Cu`) without cached `wscrn.dat`,
`vtot.dat`, or `gg.bin`; the generated `wscrn.dat` screened-potential column is
within `4.7e-6` max absolute difference of the FEFF references, and the
refreshed bare core-hole column matches the `apot.bin` handoff. The default
`screen_module_matches_graphite_reference_zip_without_phase_or_gg_cache` parity
gate adds a non-Cu KSPACE/Graphite archive fixture, exercising legacy
FMS inputs that omit `save_gg_slice`/`do_fms`, lower `fms.inp lmaxph` than
`screen.inp maxl`, and no cached `phase.bin`/`gg.bin`; generated `wscrn.dat`
and `vtot.dat` match the FEFF reference within `1e-4`. SCREEN now reports a
normal source-requirement error for missing or incomplete inputs instead of an
explicit unported fallback. Full-run supported-module orchestration now also
carries the XANES/Cu inline-FMS SCREEN source bundle through the completed
`screen` scheduler report without cached `gg.bin`, and compares the generated
`wscrn.dat`/`vtot.dat` radial rows against the FEFF reference before accepting
that stage.
Recent RIXS handoff progress includes the FEFF `RIXS/rdxsphrxs.f90`
`phase.bin` reader normalization: Rust now derives the per-energy
`lmax(ie, iph)` active angular limits from the positive-`l` phase shifts and
extracts the legacy first-eight `rkk` transition channels from the first
momentum-transfer block. The uncached RIXS CLI path now consumes that
`phase.bin` handoff when present before the remaining scattering-source requirement,
uses `global.inp` with that phase handoff to assemble the source-backed
`setkap`/`bcoef` transition labels and diagonal B-matrix setup,
selects the incident/final signed-`l` transition phase-shift tables,
and validates/assembles any available `rl_1.dat`/`rl_2.dat` radial-function
handoffs, falling back to shared `rl.dat` where an edge-specific radial handoff
is absent, through the typed XSPH `rl.dat` codec and the core RIXS read-loop
adapter. It also validates optional `wscrn_1.dat`/`wscrn_2.dat` screened-core
handoffs, falling back to shared `wscrn.dat` for the incident screened core
where `wscrn_1.dat` is absent, through the typed SCREEN codec and RIXS `DeltaV`
setup. When `global.inp`, phase, both radial handoffs, screened-core handoffs,
and `xsect.dat` are all present, the pre-solver path now also runs the core
RIXS radial `DeltaV` overlap kernel to build the source-backed radial overlap
intermediate. When the matching Green-function handoff is also available, it
feeds the core initial `TLb` amplitude assembly, wave-number setup,
incident-amplitude convolution, and raw cross-section assembly, then retains
those intermediates in the typed solver handoff bundle. The same path now
normalizes `edges.dat` poles when `ReadPoles` is enabled, falls back to FEFF's
single zero-split/unit-amplitude pole when `ReadPoles` is disabled, uses the
zero self-energy grid unless `ReadSigma` requests `mpse.dat`, assembles the
post-raw standard RIXS spectra, and converts them into renderable standard
map/line output payloads. Complete source bundles now write `rixsET.dat`,
`herfd.dat`, `xasEI.dat`, `xasEF.dat`, `rixsEE.dat`, and `logrixs.dat` through
direct module execution, `run_for_input`, and full `refeff run`
orchestration. The same standard-output source path is now pinned for complete
shared handoff bundles using `phase.bin`, `rl.dat`, `wscrn.dat`, `gg.bin`, and
`xsect.dat` without edge-specific duplicates. Full-run orchestration now
reports those complete source bundles as a completed `rixs` supported stage,
while partial source bundles continue to report as validation-only
`rixs-handoff` stages. When MBConv is enabled and `XES/xmu.dat` is present,
the same source-backed path also writes
`rixsET-sat.dat`, `herfd-sat.dat`, `xasEI-sat.dat`, `xasEF-sat.dat`, and
`rixsEE-sat.dat`.
`ReadSigma` source bundles are covered through typed `mpse.dat` self-energy
preparation before output writing, and can now consume the same MPSE table
generated in memory from compatible XSPH `xsph.inp`/`phase.bin`/`pot.bin`
source state when no cached `mpse.dat` exists. The cached-`mpse.dat` ReadSigma
path now also has an MPSE/Cu reference-zip gate that extracts FEFF's
`mpse.dat`, prepares the RIXS self-energy grid through the production CLI
handoff builder, and checks it against the core interpolation path before
writing final spectra. When a compatible XSPH MPSE source handoff is also
present, readable cached `mpse.dat` is checked against that generated table and
stale cached self-energy columns no longer override the active source handoff.
The direct RIXS module path also uses the generated XSPH MPSE table when a
malformed `mpse.dat` cache is paired with a complete compatible source handoff.
The matching full-run scheduler regression now compares source-only,
stale-cache-plus-source, malformed-cache-plus-source, and stale-cache-only
ReadSigma runs, so scheduler ordering cannot hide the MPSE source preference
behind a standalone XSPH cache repair.
Incomplete source bundles still stop at the remaining solver boundary. Optional
final-edge `xsect_2.dat` cross-section handoffs fall back to shared `xsect.dat`
where the edge-specific file is absent and are checked against matching
phase/radial energy counts before reaching the solver boundary. Optional
`gg_1.bin`/`gg_2.bin` Green-function handoffs now use the
typed FMS `gg.bin` sectioned matrix codec, fall back to shared `gg.bin` when
edge-specific Green matrices are absent, and validate square, consistent
section matrices against the available phase energy and angular limits before
the solver boundary. The RIXS handoff builder now also recovers a missing or
readable-but-stale shared `wscrn.dat` from `vtot.dat`/`apot.bin` before
validating screened-core inputs only when the shared incident screened-core
handoff is actually needed; explicit `wscrn_1.dat`/`wscrn_2.dat` edge handoffs
remain strict and are no longer
blocked by unrelated malformed shared `vtot.dat` state. The supported-stage
detector still treats a recoverable `vtot.dat`/`apot.bin` pair as a RIXS
handoff even before `wscrn.dat` exists, so the `DeltaV` route does not depend on
a separate SCREEN module pass, but malformed or otherwise unrecoverable
`vtot.dat`/`apot.bin` pairs are no longer advertised as supported RIXS handoffs.
Explicit `phase_1.bin`/`phase_2.bin` edge handoffs now also take precedence
over shared `phase.bin`, so a stale malformed shared phase sidecar no longer
blocks RIXS handoff validation when both edge-specific phase files are usable.
Full-run scheduler coverage now pins that explicit-edge precedence for the
related shared `rl.dat`, `wscrn.dat`, `gg.bin`, and `xsect.dat` handoffs too:
stale shared sidecars are ignored when the matching edge-specific inputs are
complete, while edge-specific handoffs remain strict.
Full `refeff run` orchestration and standalone
`refeff module rixs` runs also schedule incomplete source handoffs as a
`rixs-handoff` supported stage so RIXS handoff compatibility is reported before
the source requirement is evaluated. Malformed final spectrum caches no longer
suppress this source-handoff validation path when the required handoffs are
present; standalone malformed caches without handoffs still fail validation.
Malformed declared solver handoffs such as a bad `global.inp` now decline
supported-stage discovery instead of reporting `rixs` or `rixs-handoff`; the
explicit RIXS runner still surfaces the parser/setup error.
Malformed `rixs.inp` is also declined during RIXS cached-output and
solver-handoff discovery, leaving explicit RIXS execution to report the
control-input parser error.
When cached RIXS map handoffs regenerate missing or malformed line/final
outputs, the same path now regenerates malformed `logrixs.dat` wrappers; pure
cached-output directories with only a malformed module log still fail
validation.
The FEFF `RIXS/rixs.f90` final spectrum-output block is
also covered for already-computed `xsect_tmp` tables, including per-edge
radial-grid and screened-core `DeltaV` setup, radial `DeltaV` overlap
integration, `setkap`/`bcoef` transition-matrix setup, `rl.dat`
radial-function table assembly, signed-`l` transition phase-shift selection,
initial `TLb` amplitude assembly, direct final-transition term, wave-number
preparation, incident-amplitude convolution, raw cross-section assembly,
`mpse.dat` self-energy grid preparation,
incident-energy broadening, final-energy broadening, per-edge broadening
pipeline, multi-edge summation, post-raw standard spectrum pipeline, MBConv
satellite convolution, `edges.dat` pole normalization, `xasEI`, `xasEF`, HERFD
diagonal extraction, `rixsET` row ordering, and `rixsEE`
constant incident/emission grid interpolation.
The cached RIXS CLI path now also derives missing diagonal line spectra from
FEFF map caches: `herfd.dat` from `rixsET.dat`, `herfd-sat.dat` from
`rixsET-sat.dat`, and the intermediate `xas0.dat`/`xas1.dat` diagnostics from
`rixs0.dat`/`rixs1.dat`, while preserving valid existing FEFF line files and
regenerating malformed derived line files from the matching map cache.
When `edges.dat` is available, the regular cached and `SkipCalc` paths reuse
the same Rust final-spectrum transform to derive missing or malformed standard
and satellite `xasEI`, `xasEF`, and `rixsEE` outputs from cached `rixsET.dat`
and `rixsET-sat.dat`, again without overwriting valid existing FEFF side
outputs. Full `refeff run` orchestration now has regression coverage for the
regular cached `edges.dat` final-output route and malformed HERFD recovery from
`rixsET.dat`.
The cached `SkipCalc` path now uses a shared `refeff-io` handoff adapter for
`edges.dat` normalization, HERFD diagonal extraction, cached `rixsET.dat` to
`xasEI`/`xasEF`/`rixsEE` assembly, cached `rixsET-sat.dat` satellite-output
recovery, and optional MBConv satellite output assembly. That adapter calls the
same core `ReadPoles`, final-spectrum, and satellite-convolution helpers used
by the numerical port tests, and produces typed `rixsET`, `herfd`, `xasEI`,
`xasEF`, and `rixsEE` payloads for both standard and MBConv satellite outputs.
RDINP now carries optional RIXS switch arguments through to `rixs.inp`, so
FEFF-style `RIXS` cards can enable `SkipCalc`/`ReadPoles` and full `refeff run`
can exercise the Rust cached-map postprocessor without hand-written module
input; the same full-run path now covers MBConv satellite generation from a
valid XES `xmu.dat` source and emits the satellite RIXS map and line outputs
through the Rust `SkipCalc` postprocessor.
`SkipCalc` runs now also roundtrip any RIXS side outputs that were not
regenerated by the available source handoffs, so malformed stale final-output
files do not bypass validation when `edges.dat` or satellite source data is
absent; the supported-stage predicate mirrors that validation so unrecoverable
malformed SkipCalc side outputs are not advertised as cached RIXS stages. The
same predicate now checks an existing `XES/xmu.dat` satellite source when MBConv
`SkipCalc` would consume it, so malformed optional XES state no longer causes
full-run orchestration to advertise RIXS as a supported cached stage before the
explicit RIXS run reaches the malformed satellite source.
The explicit RIXS unported fallback is now retired: missing or incomplete
source state reports a normal source-requirement error, while complete source
bundles write standard and satellite final spectra. The partial-cache source
fill path is now also pinned for requested MBConv satellite spectra, so a valid
regular HERFD cache cannot mask source-generated `rixsET-sat.dat`,
`herfd-sat.dat`, `xasEI-sat.dat`, `xasEF-sat.dat`, or `rixsEE-sat.dat`.
RIXS cached-output discovery now validates declared solver handoff files before
accepting readable final spectra; malformed `phase.bin` source state keeps
cached `herfd.dat`/`rixsET.dat` output from being advertised as a completed
stage, while explicit edge-specific handoffs still take precedence over
malformed shared fallback files.
Remaining RIXS work is parity broadening for MPSE branches and additional
NRIXS branch coverage, not a module-level unported gate.
The explicit XSPH unported fallback is now retired as well: missing or
incomplete phase state reports the XSPH source requirement, while complete
caches or supported `pot.bin`/`config.dat` source handoffs produce the phase,
cross-section, and sidecar outputs covered by current reference gates.

## Recommended Order

1. Run the pinned scope audit, required-fixture release workspace suite, strict
   release-readiness command, and stock-workflow parity loop.
2. Broaden reference fixtures and tighten tolerances without reopening a
   production branch unless the new evidence exposes a real mismatch.
3. Add module-level benchmarks and optimize allocation, iteration, and linear
   algebra hotspots only after reference parity remains established.
