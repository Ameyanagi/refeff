use super::*;

/// Angular-momentum limit shared by every synthesized realistic cluster.
const REALISTIC_GLOBAL_LMAX: usize = 3;
/// Angular-momentum limit as the raw FEFF `lipotx` integer type.
const REALISTIC_LMAX_RAW: i32 = 3;
/// Single spin channel, matching a non-spin-polarized XANES/EXAFS run.
const REALISTIC_SPIN_CHANNELS: usize = 1;

/// Energy-independent geometry and per-run tables for one synthesized
/// cluster, built once and shared across every energy point in a bench.
struct RealisticCluster {
    atoms: Vec<FmsAtom>,
    max_potential: usize,
    raw_potential_lmax: Vec<i32>,
    direct_cutoff: f32,
    xnlm: Array2<f64>,
    rotations: Array6<Complex32>,
    mean_square_displacements: Array2<f32>,
    spin_orbit: refeff_core::angular::SpinOrbitCouplingTables,
    calculated_l: Vec<bool>,
}

impl RealisticCluster {
    fn build(atoms: Vec<FmsAtom>, max_potential: usize) -> Option<Self> {
        let extent = cluster_extent(&atoms);
        // Generous enough that no pair in the synthesized cluster is
        // dropped, so the LU solve exercises the full dense FMS matrix.
        let direct_cutoff = 2.0 * extent + 1.0;
        let xnlm = legendre_normalization_table(REALISTIC_GLOBAL_LMAX).ok()?;
        let geometry =
            fms_yprep_geometry(REALISTIC_GLOBAL_LMAX, REALISTIC_GLOBAL_LMAX, &atoms).ok()?;
        let mean_square_displacements =
            Array2::from_elem((atoms.len(), atoms.len()).f(), 0.003_f32);
        let spin_orbit = spin_orbit_coupling_tables(REALISTIC_GLOBAL_LMAX).ok()?;
        let calculated_l = vec![true; REALISTIC_GLOBAL_LMAX + 1];
        let raw_potential_lmax = vec![REALISTIC_LMAX_RAW; max_potential + 1];
        Some(Self {
            atoms,
            max_potential,
            raw_potential_lmax,
            direct_cutoff,
            xnlm,
            rotations: geometry.rotations,
            mean_square_displacements,
            spin_orbit,
            calculated_l,
        })
    }

    fn energy_input<'a>(
        &'a self,
        wave_numbers: &'a [Complex32],
        phase_shifts: &'a Array3<Complex32>,
    ) -> FmsRealSpaceEnergyInput<'a> {
        FmsRealSpaceEnergyInput {
            lfms: 1,
            minv: 0,
            spin_channels: REALISTIC_SPIN_CHANNELS,
            spin_selector: 0,
            atoms: &self.atoms,
            max_potential: self.max_potential,
            global_lmax: REALISTIC_GLOBAL_LMAX,
            raw_potential_lmax: &self.raw_potential_lmax,
            state_capacity: None,
            wave_numbers,
            phase_shifts: phase_shifts.view(),
            spin_orbit: &self.spin_orbit,
            direct_cutoff: self.direct_cutoff,
            mean_square_displacements: self.mean_square_displacements.view(),
            xnlm: self.xnlm.view(),
            rotations: self.rotations.view(),
            calculated_l: &self.calculated_l,
            convergence_tolerance: 1.0e-5,
            zero_tolerance: 0.0,
            full_scattering_matrix_requested: false,
        }
    }
}

/// One-shot re-implementation of the per-energy assembly steps that
/// `fms_real_space_energy` runs internally (pair tables, free propagator,
/// t-matrix, and the compact `I - G0 T` system matrix), stopping just short
/// of the LU factorization/solve so the assembly cost can be measured on its
/// own using only the plain per-step FMS functions.
fn realistic_energy_assembly(
    cluster: &RealisticCluster,
    states: &[StateKet],
    energy_index: usize,
) -> Option<Array2<Complex32>> {
    let wave_numbers = [realistic_wave_number(energy_index)];
    let phase_shifts = realistic_phase_shifts(cluster.max_potential + 1, energy_index);
    let pair_tables =
        fms_spin_pair_tables(REALISTIC_GLOBAL_LMAX, &wave_numbers, &cluster.atoms).ok()?;
    let free_propagator = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
        states,
        atoms: &cluster.atoms,
        direct_cutoff: cluster.direct_cutoff,
        rho: pair_tables.rho.view(),
        wave_numbers: &wave_numbers,
        mean_square_displacements: cluster.mean_square_displacements.view(),
        xclm: pair_tables.polynomials.view(),
        xnlm: cluster.xnlm.view(),
        rotations: cluster.rotations.view(),
    })
    .ok()?;
    let t_matrix = fms_t_matrix_table(FmsTMatrixTableInput {
        states,
        atoms: &cluster.atoms,
        spin_channels: REALISTIC_SPIN_CHANNELS,
        spin_selector: 0,
        phase_shifts: phase_shifts.view(),
        spin_orbit: &cluster.spin_orbit,
    })
    .ok()?;
    fms_iterative_system_matrix(FmsIterativeSystemInput {
        states,
        spin_channels: REALISTIC_SPIN_CHANNELS,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: 0.0,
    })
    .ok()
}

/// Plausible photoelectron wave number (a.u.) at one of eight sweep points:
/// increasing real part (higher kinetic energy) with a small, slowly growing
/// imaginary part representing inelastic damping.
fn realistic_wave_number(energy_index: usize) -> Complex32 {
    let step = energy_index as f32;
    Complex32::new(0.4 + 0.35 * step, 0.03 + 0.01 * step)
}

/// Plausible complex phase shifts `xphase(spin, l, potential)`: amplitude
/// falls off with angular momentum and grows slightly with potential index
/// (heavier/more-scattering species), with a small absorptive part that
/// grows a little across the energy sweep.
fn realistic_phase_shifts(potential_count: usize, energy_index: usize) -> Array3<Complex32> {
    let angular_len = 2 * REALISTIC_GLOBAL_LMAX + 1;
    let mut table = Array3::zeros((REALISTIC_SPIN_CHANNELS, angular_len, potential_count).f());
    let energy_scale = 1.0 - 0.05 * energy_index as f32;
    for potential in 0..potential_count {
        let potential_scale = 1.0 + 0.15 * potential as f32;
        for l in 0..=REALISTIC_GLOBAL_LMAX {
            let index = l + REALISTIC_GLOBAL_LMAX;
            let falloff = (-0.6 * l as f32).exp();
            let magnitude = 1.5 * potential_scale * energy_scale * falloff;
            let absorption = 0.08 * (-0.4 * l as f32).exp() * (1.0 + 0.1 * energy_index as f32);
            table[(0, index, potential)] = Complex32::new(magnitude, absorption);
        }
    }
    table
}

fn cluster_extent(atoms: &[FmsAtom]) -> f32 {
    atoms.iter().fold(0.0_f32, |farthest, atom| {
        let [x, y, z] = atom.position;
        farthest.max((x * x + y * y + z * z).sqrt())
    })
}

/// Every point of the conventional 4-atom FCC basis within `cell_range`
/// cubic cells of the origin, in Angstrom.
fn fcc_lattice_points(cell_range: i32, lattice_constant: f32) -> Vec<[f32; 3]> {
    let basis = [
        [0.0_f32, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut points = Vec::new();
    for i in -cell_range..=cell_range {
        for j in -cell_range..=cell_range {
            for k in -cell_range..=cell_range {
                for offset in basis {
                    points.push([
                        (i as f32 + offset[0]) * lattice_constant,
                        (j as f32 + offset[1]) * lattice_constant,
                        (k as f32 + offset[2]) * lattice_constant,
                    ]);
                }
            }
        }
    }
    points
}

/// A single-species FCC cluster (e.g. a Pt-like metal) of exactly
/// `target_count` atoms, closest-to-farthest from a central absorber at the
/// origin. Non-central atoms all share one FEFF potential index.
fn fcc_cluster(target_count: usize, lattice_constant: f32) -> Vec<FmsAtom> {
    let mut ranked: Vec<(f32, [f32; 3])> = fcc_lattice_points(4, lattice_constant)
        .into_iter()
        .map(|position| {
            let [x, y, z] = position;
            ((x * x + y * y + z * z).sqrt(), position)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked
        .into_iter()
        .take(target_count)
        .enumerate()
        .map(|(index, (_, position))| FmsAtom {
            position,
            potential: if index == 0 { 0 } else { 1 },
        })
        .collect()
}

/// A rocksalt-structure cluster (e.g. an MgO-like ionic compound) of
/// exactly `target_count` atoms: a cation FCC sublattice and an anion FCC
/// sublattice offset by half a lattice constant along x. The central
/// absorber (potential 0) sits on the cation sublattice; other cation-site
/// atoms share potential 1 and anion-site atoms share potential 2.
fn rocksalt_cluster(target_count: usize, lattice_constant: f32) -> Vec<FmsAtom> {
    let mut ranked: Vec<(f32, [f32; 3], bool)> = fcc_lattice_points(4, lattice_constant)
        .into_iter()
        .map(|position| {
            let [x, y, z] = position;
            ((x * x + y * y + z * z).sqrt(), position, true)
        })
        .collect();
    ranked.extend(
        fcc_lattice_points(4, lattice_constant)
            .into_iter()
            .map(|position| {
                let anion_position = [
                    position[0] + 0.5 * lattice_constant,
                    position[1],
                    position[2],
                ];
                let [x, y, z] = anion_position;
                ((x * x + y * y + z * z).sqrt(), anion_position, false)
            }),
    );
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked
        .into_iter()
        .take(target_count)
        .enumerate()
        .map(|(index, (_, position, is_cation_site))| {
            let potential = if index == 0 {
                0
            } else if is_cation_site {
                1
            } else {
                2
            };
            FmsAtom {
                position,
                potential,
            }
        })
        .collect()
}

pub(super) fn bench_fms_realistic(c: &mut Criterion) {
    // Pt-like FCC lattice constant (Angstrom); FCC coordination shells give
    // exactly 87 atoms (center + 6 shells) at lmax=3 -> 16 states/atom ->
    // matrix order 1392.
    let Some(fcc) = RealisticCluster::build(fcc_cluster(87, 3.92), 1) else {
        return;
    };
    // MgO-like rocksalt lattice constant (Angstrom); truncating the merged
    // cation/anion shells to 177 atoms gives matrix order 2832.
    let Some(rocksalt) = RealisticCluster::build(rocksalt_cluster(177, 4.21), 2) else {
        return;
    };

    let Ok(fcc_setup) = fms_driver_setup(FmsDriverSetupInput {
        lfms: 1,
        spin_channels: REALISTIC_SPIN_CHANNELS,
        atoms: &fcc.atoms,
        max_potential: fcc.max_potential,
        global_lmax: REALISTIC_GLOBAL_LMAX,
        raw_potential_lmax: &fcc.raw_potential_lmax,
        state_capacity: None,
    }) else {
        return;
    };
    let Ok(rocksalt_setup) = fms_driver_setup(FmsDriverSetupInput {
        lfms: 1,
        spin_channels: REALISTIC_SPIN_CHANNELS,
        atoms: &rocksalt.atoms,
        max_potential: rocksalt.max_potential,
        global_lmax: REALISTIC_GLOBAL_LMAX,
        raw_potential_lmax: &rocksalt.raw_potential_lmax,
        state_capacity: None,
    }) else {
        return;
    };

    let mut group = c.benchmark_group("fms_realistic");
    group.sample_size(10);

    // (a) One `fms_real_space_energy` LU solve (assembly + direct solve).
    group.bench_function("fcc_atoms87_lmax3_lu_solve", |b| {
        b.iter(|| {
            let wave_numbers = [realistic_wave_number(0)];
            let phase_shifts = realistic_phase_shifts(fcc.max_potential + 1, 0);
            black_box(fms_real_space_energy(
                fcc.energy_input(black_box(&wave_numbers), black_box(&phase_shifts)),
            ))
        });
    });
    group.bench_function("rocksalt_atoms177_lmax3_lu_solve", |b| {
        b.iter(|| {
            let wave_numbers = [realistic_wave_number(0)];
            let phase_shifts = realistic_phase_shifts(rocksalt.max_potential + 1, 0);
            black_box(fms_real_space_energy(
                rocksalt.energy_input(black_box(&wave_numbers), black_box(&phase_shifts)),
            ))
        });
    });

    // (b) System-matrix assembly alone: pair tables, free propagator,
    // t-matrix, and the compact `I - G0 T` system matrix, without the LU
    // factorization/solve.
    group.bench_function("fcc_atoms87_lmax3_assembly", |b| {
        b.iter(|| {
            black_box(realistic_energy_assembly(
                black_box(&fcc),
                &fcc_setup.state_kets.states,
                0,
            ))
        });
    });
    group.bench_function("rocksalt_atoms177_lmax3_assembly", |b| {
        b.iter(|| {
            black_box(realistic_energy_assembly(
                black_box(&rocksalt),
                &rocksalt_setup.state_kets.states,
                0,
            ))
        });
    });

    // (c) An 8-energy sweep of full `fms_real_space_energy` LU solves, as a
    // stand-in for one FMS module's per-energy loop.
    group.bench_function("fcc_atoms87_lmax3_energy_sweep8", |b| {
        b.iter(|| {
            for energy_index in 0..8_usize {
                let wave_numbers = [realistic_wave_number(energy_index)];
                let phase_shifts = realistic_phase_shifts(fcc.max_potential + 1, energy_index);
                let _ = black_box(fms_real_space_energy(
                    fcc.energy_input(&wave_numbers, &phase_shifts),
                ));
            }
        });
    });
    group.bench_function("rocksalt_atoms177_lmax3_energy_sweep8", |b| {
        b.iter(|| {
            for energy_index in 0..8_usize {
                let wave_numbers = [realistic_wave_number(energy_index)];
                let phase_shifts = realistic_phase_shifts(rocksalt.max_potential + 1, energy_index);
                let _ = black_box(fms_real_space_energy(
                    rocksalt.energy_input(&wave_numbers, &phase_shifts),
                ));
            }
        });
    });

    group.finish();
}
