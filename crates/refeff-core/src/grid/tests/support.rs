use super::*;

pub(super) fn assert_spinor_value(
    spinor: &DiracSpinorGrid,
    index_1based: usize,
    expected_large: Real,
    expected_small: Real,
) {
    let index = index_1based - 1;
    assert_close(spinor.large_component[index], expected_large);
    assert_close(spinor.small_component[index], expected_small);
}

pub(super) fn assert_orbital_value(
    spinor: &DiracSpinorOrbitalsGrid,
    index_1based: usize,
    orbital_1based: usize,
    expected_large: Real,
    expected_small: Real,
) {
    let radial = index_1based - 1;
    let orbital = orbital_1based - 1;
    assert_close(spinor.large_components[(radial, orbital)], expected_large);
    assert_close(spinor.small_components[(radial, orbital)], expected_small);
}

pub(super) fn assert_potential_value(
    grid: &PotentialGrid,
    index_1based: usize,
    expected_radius: Real,
    expected_potential: Real,
    expected_density: Real,
    expected_magnetization: Real,
) {
    let index = index_1based - 1;
    assert_close(grid.radii[index], expected_radius);
    assert_close(grid.total_potential[index], expected_potential);
    assert_close(grid.charge_density[index], expected_density);
    assert_close(grid.magnetization[index], expected_magnetization);
}

pub(super) fn assert_atomic_quantity_value(
    grid: &AtomicQuantitiesGrid,
    index_1based: usize,
    expected: [Real; 6],
) {
    let index = index_1based - 1;
    assert_scfdat_fix_close(grid.coulomb_potential[index], expected[0]);
    assert_scfdat_fix_close(grid.charge_density[index], expected[1]);
    assert_scfdat_fix_close(grid.magnetization[index], expected[2]);
    assert_scfdat_fix_close(grid.valence_density[index], expected[3]);
    assert_scfdat_fix_close(grid.initial_large_component[index], expected[4]);
    assert_scfdat_fix_close(grid.initial_small_component[index], expected[5]);
}

pub(super) fn assert_atomic_orbital_quantity_value(
    grid: &AtomicQuantitiesGrid,
    index_1based: usize,
    expected: [Real; 6],
) {
    let index = index_1based - 1;
    assert_scfdat_fix_close(grid.large_components[(index, 0)], expected[0]);
    assert_scfdat_fix_close(grid.small_components[(index, 0)], expected[1]);
    assert_scfdat_fix_close(grid.large_components[(index, 2)], expected[2]);
    assert_scfdat_fix_close(grid.small_components[(index, 2)], expected[3]);
    assert_scfdat_fix_close(grid.large_components[(index, 40)], expected[4]);
    assert_scfdat_fix_close(grid.small_components[(index, 40)], expected[5]);
}

pub(super) fn assert_scfdat_fix_close(actual: Real, expected: Real) {
    assert_close_with_tolerance(actual, expected, 5.0e-7_f64.max(expected.abs() * 5.0e-7));
}

pub(super) fn assert_energy(
    grid: &ScmtEnergyGrid,
    index_1based: usize,
    expected_real: Real,
    expected_imaginary: Real,
) {
    let value = grid.energies[index_1based - 1];
    assert_close(value.re, expected_real);
    assert_close(value.im, expected_imaginary);
}

pub(super) fn assert_step(grid: &ScmtEnergyGrid, index_1based: usize, expected: Real) {
    assert_close(grid.steps[index_1based - 1], expected);
}

pub(super) fn assert_overlap_value(
    overlap: &LoucksSphericalOverlap,
    base: &Array1<Real>,
    index_1based: usize,
    expected_total: Real,
    expected_contribution: Real,
) {
    let index = index_1based - 1;
    const SUMAX_ORACLE_TOLERANCE: Real = 5.0e-9;

    assert_close_with_tolerance(
        overlap.accumulated[index],
        expected_total,
        SUMAX_ORACLE_TOLERANCE,
    );
    assert_close_with_tolerance(
        overlap.accumulated[index] - base[index],
        expected_contribution,
        SUMAX_ORACLE_TOLERANCE,
    );
}

pub(super) fn assert_interstitial_values(
    values: InterstitialShellValues,
    expected_potential: Real,
    expected_density: Real,
    expected_volume: Real,
) {
    const ISTVAL_ORACLE_TOLERANCE: Real = 5.0e-10;

    assert_close_with_tolerance(
        values.interstitial_potential,
        expected_potential,
        ISTVAL_ORACLE_TOLERANCE,
    );
    assert_close_with_tolerance(
        values.interstitial_density,
        expected_density,
        ISTVAL_ORACLE_TOLERANCE,
    );
    assert_close_with_tolerance(
        values.shell_volume,
        expected_volume,
        1.0e-15_f64.max(expected_volume.abs() * 5.0e-7),
    );
}

pub(super) fn assert_fermi_level(
    value: FermiLevel,
    expected_chemical_potential: Real,
    expected_density_parameter: Real,
    expected_fermi_momentum: Real,
) {
    assert_close(value.chemical_potential, expected_chemical_potential);
    assert_close(value.density_parameter, expected_density_parameter);
    assert_close(value.fermi_momentum, expected_fermi_momentum);
}

pub(super) fn run_sample_potential_grid(
    jump_mode: i32,
    potential_jump: Real,
) -> Result<PotentialGrid, GridError> {
    let (density, potential, magnetization) = sample_potential_sources();
    fix_potential_grid(PotentialGridInput {
        muffin_tin_radius: (-8.8 + 60.4 * 0.05_f64).exp(),
        electron_density: density.view(),
        total_potential: potential.view(),
        magnetization: magnetization.view(),
        interstitial_potential: -0.75,
        interstitial_density: 0.28,
        original_delta: 0.05,
        new_delta: 0.025,
        jump_mode,
        potential_jump,
        output_len: 180,
    })
}

pub(super) fn sample_potential_sources() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let source_len = 251;
    let density = (1..=source_len)
        .map(|index| {
            let i = index as Real;
            0.4 + 0.002 * i + 0.03 * (0.04 * i).sin()
        })
        .collect::<Array1<_>>();
    let potential = (1..=source_len)
        .map(|index| {
            let i = index as Real;
            -2.0 + 0.015 * i + 0.05 * (0.03 * i).cos()
        })
        .collect::<Array1<_>>();
    let magnetization = (1..=source_len)
        .map(|index| {
            let i = index as Real;
            0.01 * (0.08 * i).sin() - 0.0001 * i
        })
        .collect::<Array1<_>>();
    (density, potential, magnetization)
}

#[derive(Debug, Clone)]
pub(super) struct AtomicQuantitiesSample {
    radii: Array1<Real>,
    coulomb_potential: Array1<Real>,
    charge_density: Array1<Real>,
    magnetization: Array1<Real>,
    valence_density: Array1<Real>,
    initial_large_component: Array1<Real>,
    initial_small_component: Array1<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
}

impl AtomicQuantitiesSample {
    pub(super) fn input(&self) -> AtomicQuantitiesGridInput<'_> {
        AtomicQuantitiesGridInput {
            source_radii: self.radii.view(),
            coulomb_potential: self.coulomb_potential.view(),
            charge_density: self.charge_density.view(),
            magnetization: self.magnetization.view(),
            valence_density: self.valence_density.view(),
            initial_large_component: self.initial_large_component.view(),
            initial_small_component: self.initial_small_component.view(),
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            output_len: 251,
        }
    }
}

pub(super) fn sample_atomic_quantities() -> AtomicQuantitiesSample {
    let source_len = 251;
    let radii = (1..=source_len)
        .map(|index| {
            (-8.85 + 0.051 * (index - 1) as Real + 1.0e-4 * (0.37 * index as Real).cos()).exp()
        })
        .collect::<Array1<_>>();
    let coulomb_potential = (1..=source_len)
        .map(|index| 0.2 + 0.01 * index as Real + (0.03 * index as Real).sin())
        .collect::<Array1<_>>();
    let charge_density = (1..=source_len)
        .map(|index| 0.1 * index as Real + 0.25 * (0.02 * index as Real).cos())
        .collect::<Array1<_>>();
    let magnetization = (1..=source_len)
        .map(|index| -0.04 * index as Real + 0.1 * (0.05 * index as Real).sin())
        .collect::<Array1<_>>();
    let valence_density = (1..=source_len)
        .map(|index| 0.05 * (index as Real).sqrt() + 0.002 * (index % 5) as Real)
        .collect::<Array1<_>>();
    let initial_large_component = (1..=source_len)
        .map(|index| 0.003 * index as Real + 1.0e-5 * (index * index) as Real)
        .collect::<Array1<_>>();
    let initial_small_component = (1..=source_len)
        .map(|index| -0.002 * index as Real + 2.0e-6 * (index * index) as Real)
        .collect::<Array1<_>>();
    let large_components = Array2::from_shape_fn((source_len, 41).f(), |(row, column)| {
        let i = (row + 1) as Real;
        let j = (column + 1) as Real;
        0.001 * i * j + 0.02 * (0.01 * (i + j)).sin()
    });
    let small_components = Array2::from_shape_fn((source_len, 41).f(), |(row, column)| {
        let i = (row + 1) as Real;
        let j = (column + 1) as Real;
        -0.0007 * i * j + 0.015 * (0.012 * (i + 2.0 * j)).cos()
    });
    AtomicQuantitiesSample {
        radii,
        coulomb_potential,
        charge_density,
        magnetization,
        valence_density,
        initial_large_component,
        initial_small_component,
        large_components,
        small_components,
    }
}

#[derive(Debug, Clone)]
pub(super) struct MovrlpSample {
    atom_potentials: Array1<usize>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    potential_multiplicities: Array1<Real>,
    neighbors0: [MuffinTinOverlapNeighbor; 1],
    neighbors1: [MuffinTinOverlapNeighbor; 1],
    norman_indices: Array1<usize>,
    muffin_tin_indices: Array1<usize>,
    muffin_tin_radii: Array1<Real>,
    norman_radii: Array1<Real>,
    near_neighbor_flags: Array1<bool>,
}

impl MovrlpSample {
    pub(super) fn explicit_overlaps(&self) -> [&[MuffinTinOverlapNeighbor]; 2] {
        [&self.neighbors0, &self.neighbors1]
    }

    pub(super) fn input<'a>(
        &'a self,
        explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
    ) -> MuffinTinOverlapMatrixInput<'a> {
        MuffinTinOverlapMatrixInput {
            highest_potential_index: 1,
            atom_potentials: self.atom_potentials.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            potential_multiplicities: self.potential_multiplicities.view(),
            explicit_overlaps,
            muffin_tin_indices: self.muffin_tin_indices.view(),
            muffin_tin_radii: self.muffin_tin_radii.view(),
            norman_radii: self.norman_radii.view(),
            near_neighbor_flags: self.near_neighbor_flags.view(),
            interstitial_selector: 0,
            interstitial_volume: 12.5,
        }
    }

    pub(super) fn projection_input<'a>(
        &'a self,
        values: ArrayView2<'a, Real>,
        radii: ArrayView1<'a, Real>,
        overlap_matrix: &'a MuffinTinOverlapMatrix,
        mode: MuffinTinOverlapProjectionMode,
        interstitial_value: Real,
    ) -> MuffinTinOverlapProjectionInput<'a> {
        MuffinTinOverlapProjectionInput {
            highest_potential_index: 1,
            values,
            radii,
            potential_multiplicities: self.potential_multiplicities.view(),
            norman_indices: self.norman_indices.view(),
            muffin_tin_indices: self.muffin_tin_indices.view(),
            muffin_tin_radii: self.muffin_tin_radii.view(),
            norman_radii: self.norman_radii.view(),
            near_neighbor_flags: self.near_neighbor_flags.view(),
            overlap_matrix,
            interstitial_selector: 0,
            interstitial_value,
            mode,
        }
    }
}

pub(super) fn sample_movrlp_state() -> MovrlpSample {
    let atom_potentials = Array1::from_vec(vec![0, 1]);
    let atom_positions = Array2::<Real>::zeros((2, 3));
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let potential_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let neighbors0 = [MuffinTinOverlapNeighbor {
        source_potential: 1,
        multiplicity: 2,
        distance: 0.030,
    }];
    let neighbors1 = [MuffinTinOverlapNeighbor {
        source_potential: 0,
        multiplicity: 1,
        distance: 0.031,
    }];
    let norman_indices = Array1::from_vec(vec![90, 92]);
    let muffin_tin_indices = Array1::from_vec(vec![95, 100]);
    let muffin_tin_radii = Array1::from_vec(vec![0.020, 0.024]);
    let norman_radii = Array1::from_vec(vec![0.015, 0.018]);
    let near_neighbor_flags = Array1::from_vec(vec![false, false]);
    MovrlpSample {
        atom_potentials,
        atom_positions,
        representative_atoms,
        potential_multiplicities,
        neighbors0,
        neighbors1,
        norman_indices,
        muffin_tin_indices,
        muffin_tin_radii,
        norman_radii,
        near_neighbor_flags,
    }
}

pub(super) fn sample_ovp2mt_values(radii: ArrayView1<'_, Real>) -> Array2<Real> {
    Array2::from_shape_fn((251, 2), |(radial, potential)| {
        let index = (radial + 1) as Real;
        0.1 * (potential + 1) as Real
            + 0.001 * index
            + 0.00001 * index * index
            + 0.02 * radii[radial]
    })
}

pub(super) fn sample_sumax_grids() -> (Array1<Real>, Array1<Real>) {
    let len = 250;
    let source = (1..=len)
        .map(|index| {
            let i = index as Real;
            0.2 + 0.004 * i + 0.03 * (0.035 * i).sin()
        })
        .collect::<Array1<_>>();
    let base = (1..=len)
        .map(|index| {
            let i = index as Real;
            0.01 * (0.027 * i).cos()
        })
        .collect::<Array1<_>>();
    (source, base)
}

pub(super) fn sample_istval_grids() -> (Array1<Real>, Array1<Real>) {
    let len = 1251;
    let potential = (1..=len)
        .map(|index| {
            let i = index as Real;
            -1.5 + 0.002 * i + 0.04 * (0.017 * i).cos()
        })
        .collect::<Array1<_>>();
    let density = (1..=len)
        .map(|index| {
            let i = index as Real;
            0.5 + 0.003 * i + 0.02 * (0.023 * i).sin()
        })
        .collect::<Array1<_>>();
    (potential, density)
}

pub(super) fn sample_frnrm_oxygen_density() -> Array1<Real> {
    (1..=FRNRM_DENSITY_POINTS)
        .map(|index| {
            let radius = feff_legacy_loucks_radius(index);
            50.0 * (-1.2 * radius).exp() + 0.1 * (-0.05 * radius).exp()
        })
        .collect::<Array1<_>>()
}

pub(super) fn sample_frnrm_iron_density() -> Array1<Real> {
    (1..=FRNRM_DENSITY_POINTS)
        .map(|index| {
            let radius = feff_legacy_loucks_radius(index);
            220.0 * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
        })
        .collect::<Array1<_>>()
}

pub(super) fn sample_frnrm_gold_density() -> Array1<Real> {
    (1..=FRNRM_DENSITY_POINTS)
        .map(|index| {
            let radius = feff_legacy_loucks_radius(index);
            950.0 * (-0.55 * radius).exp() / (1.0 + 0.08 * radius * radius)
        })
        .collect::<Array1<_>>()
}

pub(super) fn sample_sidx_keep_density() -> Array1<Real> {
    (1..=250)
        .map(|index| {
            let i = index as Real;
            0.08 + 0.0004 * i + 0.002 * (0.05 * i).sin()
        })
        .collect::<Array1<_>>()
}

pub(super) fn sample_sidx_cutoff_density() -> Array1<Real> {
    (1..=250)
        .map(|index| {
            if index <= 92 {
                0.04 + 0.0002 * index as Real
            } else {
                1.0e-6
            }
        })
        .collect::<Array1<_>>()
}

pub(super) fn assert_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
    assert_close_with_tolerance(actual, expected, tolerance);
}

pub(super) fn assert_close_with_tolerance(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

pub(super) fn assert_complex32_close(actual: Complex32, expected: Complex32) {
    assert_close_with_tolerance(actual.re as Real, expected.re as Real, 5.0e-6);
    assert_close_with_tolerance(actual.im as Real, expected.im as Real, 5.0e-6);
}
