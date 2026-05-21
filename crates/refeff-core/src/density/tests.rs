use super::*;
use ndarray::{Array1, Array2};

#[test]
fn valence_density_update_matches_feff_ff2g_first_energy_reference() -> Result<(), DensityError> {
    let sample = sample_ff2g_state();

    let result = update_valence_density(ValenceDensityUpdateInput {
        scattering_trace: sample.scattering_trace.view(),
        potential_index: 1,
        energy_index: 1,
        last_radial_index: 5,
        scattering_ldos: sample.scattering_ldos.view(),
        embedded_ldos: sample.embedded_ldos.view(),
        previous_ldos: sample.previous_ldos.view(),
        scattering_density: sample.scattering_density.view(),
        embedded_density: sample.embedded_density.view(),
        previous_density: sample.previous_density.view(),
        valence_density: sample.valence_density.view(),
        occupancy_by_l: sample.occupancy_by_l.view(),
        current_energy: Complex::new(0.72, 0.11),
        previous_energy: Complex::new(0.61, -0.04),
        potential_multiplicity: 2.5,
        current_floor: 1,
        previous_floor: 0,
        left_sum: Complex::new(0.2, -0.1),
        right_sum: Complex::new(-0.3, 0.25),
        total_electron_count: 1.25,
        include_high_l: false,
    })?;

    assert_complex_close(
        result.embedded_ldos[(0, 1)],
        Complex::new(0.451_099_999_919_533_76, -0.215_299_999_862_909_35),
    );
    assert_complex_close(
        result.embedded_ldos[(2, 1)],
        Complex::new(0.539_700_002_484_023_6, -0.211_100_000_292_062_77),
    );
    assert_complex_close(
        result.embedded_ldos[(3, 1)],
        Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
    );
    assert_complex_close(result.previous_ldos[(2, 1)], result.embedded_ldos[(2, 1)]);
    assert_complex_close(
        result.embedded_density[0],
        Complex::new(6.406_000_025_570_392e-2, -1.129_999_976_605_176_9e-2),
    );
    assert_complex_close(
        result.embedded_density[4],
        Complex::new(2.775_000_003_725_290_3e-1, -9.609_999_980_777_503e-2),
    );
    assert_complex_close(result.previous_density[4], result.embedded_density[4]);
    assert_close(result.valence_density[0], 1.263_880_007_192_492_5e-2);
    assert_close(result.valence_density[3], 4.145_320_007_205_01e-2);
    assert_close(result.valence_density[4], 5.105_800_007_209_182e-2);
    assert_close(result.occupancy_by_l[0], -4.127_799_997_627_735e-2);
    assert_close(result.occupancy_by_l[2], -3.265_999_865_531_922_5e-3);
    assert_close(result.occupancy_by_l[3], 1.5e-2);
    assert_close(result.total_electron_count, 1.082_550_000_302_493_7);
    assert_complex_close(
        result.left_sum,
        Complex::new(7.618_000_007_234_514, -3.296_999_999_880_791),
    );
    assert_complex_close(
        result.right_sum,
        Complex::new(7.118_000_007_234_514, -2.946_999_999_880_791_4),
    );
    Ok(())
}

#[test]
fn valence_density_update_matches_feff_ff2g_high_l_reference() -> Result<(), DensityError> {
    let sample = sample_ff2g_state();

    let result = update_valence_density(ValenceDensityUpdateInput {
        scattering_trace: sample.scattering_trace.view(),
        potential_index: 1,
        energy_index: 2,
        last_radial_index: 4,
        scattering_ldos: sample.scattering_ldos.view(),
        embedded_ldos: sample.embedded_ldos.view(),
        previous_ldos: sample.previous_ldos.view(),
        scattering_density: sample.scattering_density.view(),
        embedded_density: sample.embedded_density.view(),
        previous_density: sample.previous_density.view(),
        valence_density: sample.valence_density.view(),
        occupancy_by_l: sample.occupancy_by_l.view(),
        current_energy: Complex::new(0.91, -0.08),
        previous_energy: Complex::new(0.77, 0.05),
        potential_multiplicity: 1.75,
        current_floor: 0,
        previous_floor: 1,
        left_sum: Complex::new(-0.15, 0.09),
        right_sum: Complex::new(0.05, -0.12),
        total_electron_count: -0.2,
        include_high_l: true,
    })?;

    assert_complex_close(
        result.embedded_ldos[(3, 1)],
        Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
    );
    assert_complex_close(result.previous_ldos[(2, 1)], Complex::new(-4.0e-2, 4.5e-2));
    assert_complex_close(
        result.embedded_density[0],
        Complex::new(8.203_999_945_521_354e-2, -1.959_999_881_684_781e-3),
    );
    assert_complex_close(result.embedded_density[4], Complex::new(2.5e-1, -1.0e-1));
    assert_complex_close(result.previous_density[4], Complex::new(-1.5e-1, 2.0e-1));
    assert_close(result.valence_density[0], 5.560_400_087_386_373e-3);
    assert_close(result.valence_density[3], 2.428_160_011_395_812e-2);
    assert_close(result.valence_density[4], 5.0e-2);
    assert_close(result.occupancy_by_l[0], -1.041_849_999_703_466_9e-1);
    assert_close(result.occupancy_by_l[2], -9.221_500_036_381_186e-2);
    assert_close(result.occupancy_by_l[3], -8.732_799_936_085_94e-2);
    assert_close(result.total_electron_count, -8.677_334_992_048_331e-1);
    assert_complex_close(result.left_sum, Complex::new(-8.85e-1, 8.6e-1));
    assert_complex_close(
        result.right_sum,
        Complex::new(7.313_899_995_405_228, -3.091_499_992_907_047_5),
    );
    Ok(())
}

#[test]
fn valence_density_update_rejects_invalid_inputs() {
    let sample = sample_ff2g_state();
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            energy_index: 0,
            ..sample.input()
        }),
        Err(DensityError::InvalidIndex {
            name: "energy_index",
            index: 0,
        })
    );
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            last_radial_index: 0,
            ..sample.input()
        }),
        Err(DensityError::InvalidIndex {
            name: "last_radial_index",
            index: 0,
        })
    );

    let short_ldos = Array1::<Complex>::zeros(2);
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            scattering_ldos: short_ldos.view(),
            ..sample.input()
        }),
        Err(DensityError::LengthMismatch {
            left_name: "scattering_trace",
            left_len: 4,
            right_name: "scattering_ldos",
            right_len: 2,
        })
    );

    let short_density = Array1::<Complex>::zeros(3);
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            embedded_density: short_density.view(),
            ..sample.input()
        }),
        Err(DensityError::LengthTooShort {
            name: "embedded_density",
            required: 5,
            actual: 3,
        })
    );

    let small_matrix = Array2::<Complex>::zeros((2, 2));
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            embedded_ldos: small_matrix.view(),
            ..sample.input()
        }),
        Err(DensityError::ShapeTooSmall {
            name: "embedded_ldos",
            rows: 2,
            columns: 2,
            required_rows: 4,
            required_columns: 2,
        })
    );

    let mut bad_trace = sample.scattering_trace.clone();
    bad_trace[1] = Complex32::new(f32::NAN, 0.0);
    assert!(matches!(
        update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: bad_trace.view(),
            ..sample.input()
        }),
        Err(DensityError::NonFiniteComplexValue {
            name: "scattering_trace",
            index: 1,
            ..
        })
    ));
}

#[test]
fn broyden_density_mix_matches_feff_broydn_reference() -> Result<(), DensityError> {
    let sample = sample_broydn_state();
    let references = broydn_references();
    let mut workspace = BroydenWorkspace::zeros(4, 2);
    let mut norman_charges = sample.norman_charges.clone();

    for reference in references {
        let input_density = sample.valence_density_for_iteration(reference.iteration);
        let result = mix_broyden_density(sample.input(
            reference.iteration,
            norman_charges.view(),
            input_density.view(),
            &workspace,
        ))?;

        for potential in 0..=1 {
            assert_broydn_grid_values(
                &result.valence_density,
                potential,
                sample.last_indices[potential],
                reference.valence_density[potential],
            );
            assert_close(
                result.charge_deltas[potential],
                reference.charge_deltas[potential],
            );
            assert_close(
                result.norman_charges[potential],
                reference.norman_charges[potential],
            );
        }
        assert_close(
            result.workspace.norms[reference.iteration - 1],
            reference.norm,
        );
        for (column, expected) in reference.coefficients.into_iter().enumerate() {
            assert_close(
                result.workspace.coefficients[(reference.iteration - 1, column)],
                expected,
            );
        }
        assert_close(
            result.workspace.previous_density[(0, 0)],
            reference.previous_density_1_0,
        );

        workspace = result.workspace;
        norman_charges = result.norman_charges;
    }

    Ok(())
}

#[test]
fn broyden_density_mix_rejects_invalid_inputs() {
    let sample = sample_broydn_state();
    let workspace = BroydenWorkspace::zeros(4, 2);
    let input_density = sample.valence_density_for_iteration(1);

    assert_eq!(
        mix_broyden_density(BroydenMixInput {
            iteration: 0,
            ..sample.input(
                1,
                sample.norman_charges.view(),
                input_density.view(),
                &workspace,
            )
        }),
        Err(DensityError::InvalidIndex {
            name: "iteration",
            index: 0,
        })
    );

    let bad_last_indices = Array1::from_vec(vec![190, 0]);
    assert_eq!(
        mix_broyden_density(BroydenMixInput {
            last_indices: bad_last_indices.view(),
            ..sample.input(
                1,
                sample.norman_charges.view(),
                input_density.view(),
                &workspace,
            )
        }),
        Err(DensityError::InvalidIndex {
            name: "last_indices",
            index: 0,
        })
    );

    let zero_occupancy = Array2::<Real>::zeros((3, 2));
    assert_eq!(
        mix_broyden_density(BroydenMixInput {
            valence_occupancy: zero_occupancy.view(),
            ..sample.input(
                1,
                sample.norman_charges.view(),
                input_density.view(),
                &workspace,
            )
        }),
        Err(DensityError::ZeroScalar {
            name: "broyden_total_fermi_count",
            value: 0.0,
        })
    );

    let short_workspace = BroydenWorkspace::zeros(1, 2);
    let second_density = sample.valence_density_for_iteration(2);
    assert_eq!(
        mix_broyden_density(sample.input(
            2,
            sample.norman_charges.view(),
            second_density.view(),
            &short_workspace,
        )),
        Err(DensityError::ShapeTooSmall {
            name: "workspace.coefficients",
            rows: 1,
            columns: 1,
            required_rows: 2,
            required_columns: 2,
        })
    );
}

#[test]
fn coulomb_update_matches_feff_coulom_norman_reference() -> Result<(), DensityError> {
    let sample = sample_coulom_state();
    let result = update_coulomb_potential(CoulombPotentialUpdateInput {
        mode: CoulombUpdateMode::Norman,
        ..sample.input()
    })?;

    assert_coulom_values(
        &result.coulomb_potential,
        0,
        [
            -1.775_572_357_598_355,
            -1.771_572_355_686_939_8,
            -1.523_562_494_397_465,
            -1.201_342_285_954_037_5,
            0.0,
            0.0,
        ],
    );
    assert_coulom_values(
        &result.coulomb_potential,
        1,
        [
            -1.995_771_090_609_550_5,
            -1.991_771_087_961_443_9,
            -1.743_757_425_966_650_6,
            -1.460_139_524_464_028_5,
            0.0,
            0.0,
        ],
    );
    Ok(())
}

#[test]
fn coulomb_update_matches_feff_coulom_long_range_reference() -> Result<(), DensityError> {
    let sample = sample_coulom_state();
    let result = update_coulomb_potential(CoulombPotentialUpdateInput {
        mode: CoulombUpdateMode::LongRange,
        ..sample.input()
    })?;

    assert_coulom_values(
        &result.coulomb_potential,
        0,
        [
            -1.593_233_968_914_050_2,
            -1.589_233_967_002_635,
            -1.341_224_105_713_160_2,
            -1.019_003_897_269_732_6,
            0.0,
            0.0,
        ],
    );
    assert_coulom_values(
        &result.coulomb_potential,
        1,
        [
            -2.000_638_675_823_475_3,
            -1.996_638_673_175_368_7,
            -1.748_625_011_180_575_5,
            -1.465_007_109_677_953_3,
            0.0,
            0.0,
        ],
    );
    Ok(())
}

#[test]
fn coulomb_update_rejects_invalid_inputs() {
    let sample = sample_coulom_state();
    let short = Array1::<usize>::zeros(1);
    assert_eq!(
        update_coulomb_potential(CoulombPotentialUpdateInput {
            last_indices: short.view(),
            ..sample.input()
        }),
        Err(DensityError::LengthTooShort {
            name: "last_indices",
            required: 2,
            actual: 1,
        })
    );

    let bad_last = Array1::from_vec(vec![140, 252]);
    assert_eq!(
        update_coulomb_potential(CoulombPotentialUpdateInput {
            last_indices: bad_last.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidIndex {
            name: "last_indices",
            index: 252,
        })
    );

    let bad_atoms = Array1::from_vec(vec![0, 2, 1]);
    assert_eq!(
        update_coulomb_potential(CoulombPotentialUpdateInput {
            atom_potentials: bad_atoms.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidPotentialIndex {
            name: "atom_potentials",
            index: 2,
            available: 2,
        })
    );
}

#[test]
fn potential_overlap_matches_feff_ovrlp_explicit_reference() -> Result<(), DensityError> {
    let sample = sample_ovrlp_state();

    let result = overlap_potential_density(PotentialOverlapInput {
        potential_index: 1,
        explicit_overlaps: &[
            PotentialOverlapNeighbor {
                source_potential: 0,
                multiplicity: 2.0,
                distance: 1.6,
            },
            PotentialOverlapNeighbor {
                source_potential: 2,
                multiplicity: 1.0,
                distance: 2.4,
            },
        ],
        ..sample.input()
    })?;

    assert_overlap_grid_values(
        &result.electron_density,
        [
            1.147_581_324_726_077_3e2,
            1.159_343_448_785_123_5e2,
            1.182_847_850_423_736_6e2,
            7.780_182_969_116_142e1,
        ],
    );
    assert_overlap_grid_values(
        &result.valence_density,
        [
            9.268_692_173_721_425e1,
            9.345_625_940_677_17e1,
            9.499_826_182_402_593e1,
            6.839_302_990_519_607e1,
        ],
    );
    assert_overlap_grid_values(
        &result.coulomb_potential,
        [
            -6.116_080_385_415_219,
            -6.053_852_541_970_13,
            -5.705_837_372_088_443,
            -5.416_140_791_110_99,
        ],
    );
    assert_overlap_grid_values(
        &result.spin_density_ratio,
        [
            2.204_636_783_021_805e-4,
            2.803_310_790_607_974_4e-4,
            4.649_794_982_532_802e-4,
            1.015_400_284_461_108_2e-3,
        ],
    );
    assert_close(result.norman_radius.radius, 6.257_226_100_235_719e-1);
    Ok(())
}

#[test]
fn potential_overlap_matches_feff_ovrlp_geometry_reference() -> Result<(), DensityError> {
    let sample = sample_ovrlp_state();

    let result = overlap_potential_density(PotentialOverlapInput {
        potential_index: 0,
        explicit_overlaps: &[],
        ..sample.input()
    })?;

    assert_overlap_grid_values(
        &result.electron_density,
        [
            8.079_917_195_503_039e1,
            8.198_343_896_737_67e1,
            8.480_691_764_162_866e1,
            5.743_104_082_268_246e1,
        ],
    );
    assert_overlap_grid_values(
        &result.valence_density,
        [
            6.503_424_582_159_577e1,
            6.580_881_910_411_395e1,
            6.765_853_259_870_691e1,
            4.938_868_124_806_859e1,
        ],
    );
    assert_overlap_grid_values(
        &result.coulomb_potential,
        [
            -4.792_432_321_760_901,
            -4.716_935_029_071_595,
            -4.417_627_169_440_431,
            -4.116_381_332_024_653,
        ],
    );
    assert_overlap_grid_values(
        &result.spin_density_ratio,
        [
            2.512_401_984_923_580_4e-4,
            3.354_335_991_070_458_7e-4,
            5.895_745_463_982_858e-4,
            1.288_501_809_125_729_8e-3,
        ],
    );
    assert_close(result.norman_radius.radius, 6.302_380_902_894_656e-1);
    Ok(())
}

#[test]
fn potential_overlap_rejects_invalid_inputs() {
    let sample = sample_ovrlp_state();
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            potential_index: 8,
            ..sample.input()
        }),
        Err(DensityError::LengthTooShort {
            name: "atomic_numbers",
            required: 9,
            actual: 3,
        })
    );

    let bad_positions = Array2::<Real>::zeros((4, 2));
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            atom_positions: bad_positions.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidPositionShape {
            rows: 4,
            columns: 2,
        })
    );

    let bad_potentials = Array1::from_vec(vec![0, 4, 2, 1]);
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            atom_potentials: bad_potentials.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidPotentialIndex {
            name: "atom_potentials",
            index: 4,
            available: 3,
        })
    );

    let bad_overlap = [PotentialOverlapNeighbor {
        source_potential: 0,
        multiplicity: 1.0,
        distance: 0.0,
    }];
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            explicit_overlaps: &bad_overlap,
            ..sample.input()
        }),
        Err(DensityError::NonPositiveScalar {
            name: "explicit_overlaps.distance",
            value: 0.0,
        })
    );
}

#[derive(Debug, Clone)]
struct Ff2gSample {
    scattering_trace: Array1<Complex32>,
    scattering_ldos: Array1<Complex>,
    embedded_ldos: Array2<Complex>,
    previous_ldos: Array2<Complex>,
    scattering_density: Array2<Complex>,
    embedded_density: Array1<Complex>,
    previous_density: Array1<Complex>,
    valence_density: Array1<Real>,
    occupancy_by_l: Array1<Real>,
}

#[derive(Debug, Clone)]
struct CoulomSample {
    last_indices: Array1<usize>,
    valence_density: Array2<Real>,
    overlapped_valence_density: Array2<Real>,
    overlapped_density: Array2<Real>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    atom_potentials: Array1<usize>,
    norman_radii: Array1<Real>,
    charge_deltas: Array1<Real>,
    atomic_numbers: Array1<usize>,
    coulomb_potential: Array2<Real>,
}

#[derive(Debug, Clone)]
struct BroydenSample {
    last_indices: Array1<usize>,
    potential_multiplicities: Array1<Real>,
    norman_radii: Array1<Real>,
    norman_charges: Array1<Real>,
    valence_occupancy: Array2<Real>,
    overlapped_valence_density: Array2<Real>,
}

#[derive(Debug, Clone, Copy)]
struct BroydenReference {
    iteration: usize,
    valence_density: [[Real; 4]; 2],
    charge_deltas: [Real; 2],
    norman_charges: [Real; 2],
    norm: Real,
    coefficients: [Real; 3],
    previous_density_1_0: Real,
}

#[derive(Debug, Clone)]
struct OvrlpSample {
    atom_potentials: Array1<usize>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    atomic_numbers: Array1<usize>,
    electron_density: Array2<Real>,
    spin_density: Array2<Real>,
    valence_density: Array2<Real>,
    coulomb_potential: Array2<Real>,
}

impl Ff2gSample {
    fn input(&self) -> ValenceDensityUpdateInput<'_> {
        ValenceDensityUpdateInput {
            scattering_trace: self.scattering_trace.view(),
            potential_index: 1,
            energy_index: 1,
            last_radial_index: 5,
            scattering_ldos: self.scattering_ldos.view(),
            embedded_ldos: self.embedded_ldos.view(),
            previous_ldos: self.previous_ldos.view(),
            scattering_density: self.scattering_density.view(),
            embedded_density: self.embedded_density.view(),
            previous_density: self.previous_density.view(),
            valence_density: self.valence_density.view(),
            occupancy_by_l: self.occupancy_by_l.view(),
            current_energy: Complex::new(0.72, 0.11),
            previous_energy: Complex::new(0.61, -0.04),
            potential_multiplicity: 2.5,
            current_floor: 1,
            previous_floor: 0,
            left_sum: Complex::new(0.2, -0.1),
            right_sum: Complex::new(-0.3, 0.25),
            total_electron_count: 1.25,
            include_high_l: false,
        }
    }
}

impl CoulomSample {
    fn input(&self) -> CoulombPotentialUpdateInput<'_> {
        CoulombPotentialUpdateInput {
            mode: CoulombUpdateMode::Norman,
            highest_potential_index: 1,
            last_indices: self.last_indices.view(),
            valence_density: self.valence_density.view(),
            overlapped_valence_density: self.overlapped_valence_density.view(),
            overlapped_density: self.overlapped_density.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            atom_potentials: self.atom_potentials.view(),
            norman_radii: self.norman_radii.view(),
            charge_deltas: self.charge_deltas.view(),
            atomic_numbers: self.atomic_numbers.view(),
            coulomb_potential: self.coulomb_potential.view(),
        }
    }
}

impl BroydenSample {
    fn input<'a>(
        &'a self,
        iteration: usize,
        norman_charges: ndarray::ArrayView1<'a, Real>,
        valence_density: ndarray::ArrayView2<'a, Real>,
        workspace: &'a BroydenWorkspace,
    ) -> BroydenMixInput<'a> {
        BroydenMixInput {
            iteration,
            accelerator: 0.35,
            highest_potential_index: 1,
            valence_occupancy: self.valence_occupancy.view(),
            last_indices: self.last_indices.view(),
            potential_multiplicities: self.potential_multiplicities.view(),
            norman_radii: self.norman_radii.view(),
            norman_charges,
            overlapped_valence_density: self.overlapped_valence_density.view(),
            valence_density,
            workspace,
        }
    }

    fn valence_density_for_iteration(&self, iteration: usize) -> Array2<Real> {
        Array2::from_shape_fn((OVRLP_DENSITY_POINTS, 2), |(radial, potential)| {
            let radius = (-8.8 + 0.05 * radial as Real).exp();
            self.overlapped_valence_density[(radial, potential)]
                * (0.97 + 0.018 * iteration as Real + 0.004 * potential as Real)
                + (0.015 * iteration as Real + 0.003 * potential as Real) * (-0.35 * radius).exp()
        })
    }
}

impl OvrlpSample {
    fn input(&self) -> PotentialOverlapInput<'_> {
        PotentialOverlapInput {
            potential_index: 1,
            atom_potentials: self.atom_potentials.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            atomic_numbers: self.atomic_numbers.view(),
            explicit_overlaps: &[],
            electron_density: self.electron_density.view(),
            spin_density: self.spin_density.view(),
            valence_density: self.valence_density.view(),
            coulomb_potential: self.coulomb_potential.view(),
        }
    }
}

fn sample_broydn_state() -> BroydenSample {
    let last_indices = Array1::from_vec(vec![190, 196]);
    let potential_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let norman_radii = Array1::from_vec(vec![0.72, 0.88]);
    let norman_charges = Array1::from_vec(vec![1.40, 2.10]);
    let mut valence_occupancy = Array2::<Real>::zeros((3, 2));
    valence_occupancy[(0, 0)] = 1.10;
    valence_occupancy[(1, 0)] = 0.60;
    valence_occupancy[(0, 1)] = 1.45;
    valence_occupancy[(1, 1)] = 0.80;
    valence_occupancy[(2, 1)] = 0.30;

    let overlapped_valence_density =
        Array2::from_shape_fn((OVRLP_DENSITY_POINTS, 2), |(radial, potential)| {
            let radius = (-8.8 + 0.05 * radial as Real).exp();
            (45.0 + 8.0 * potential as Real) * (-0.92 * radius).exp() / (1.0 + 0.10 * radius)
        });

    BroydenSample {
        last_indices,
        potential_multiplicities,
        norman_radii,
        norman_charges,
        valence_occupancy,
        overlapped_valence_density,
    }
}

fn broydn_references() -> [BroydenReference; 3] {
    [
        BroydenReference {
            iteration: 1,
            valence_density: [
                [
                    6.850_151_802_587_897e8,
                    6.198_224_678_030_062e8,
                    1.385_302_559_470_316e7,
                    -2.110_332_566_524_774e1,
                ],
                [
                    8.100_660_694_916_214e8,
                    7.329_722_972_649_122e8,
                    1.638_192_383_754_785_5e7,
                    -1.286_749_342_987_842_8e1,
                ],
            ],
            charge_deltas: [-2.099_260_615_232_956_3, 1.049_630_307_616_478_6],
            norman_charges: [-6.992_606_152_329_563e-1, 3.149_630_307_616_478_7],
            norm: 0.0,
            coefficients: [0.0, 0.0, 0.0],
            previous_density_1_0: 4.499_308_188_880_038e1,
        },
        BroydenReference {
            iteration: 2,
            valence_density: [
                [
                    7.521_952_443_079_169e8,
                    6.806_090_282_421_587e8,
                    1.521_161_506_215_363_7e7,
                    -2.378_323_890_391_962_8e1,
                ],
                [
                    8.889_720_867_627_683e8,
                    8.043_688_549_825_951e8,
                    1.797_764_579_810_173_4e7,
                    -1.449_719_473_393_818_3e1,
                ],
            ],
            charge_deltas: [-1.764_945_825_152_590_7e-1, 8.824_729_125_762_865e-2],
            norman_charges: [-8.757_551_977_482_154e-1, 3.237_877_598_874_107_3],
            norm: 7.657_998_793_600_876,
            coefficients: [0.0, -4.286_904_563_383_834_5, 0.0],
            previous_density_1_0: 4.499_308_188_880_038e1,
        },
        BroydenReference {
            iteration: 3,
            valence_density: [
                [
                    7.521_952_443_079_171e8,
                    6.806_090_282_421_585e8,
                    1.521_161_506_215_365_4e7,
                    -2.378_323_890_391_962_8e1,
                ],
                [
                    8.889_720_867_627_683e8,
                    8.043_688_549_825_957e8,
                    1.797_764_579_810_172_7e7,
                    -1.449_719_473_393_818_5e1,
                ],
            ],
            charge_deltas: [1.776_356_839_400_250_5e-15, -1.776_356_839_400_250_5e-15],
            norman_charges: [-8.757_551_977_482_136e-1, 3.237_877_598_874_105_5],
            norm: 7.657_998_793_600_846_5,
            coefficients: [0.0, -3.286_904_563_383_833_6, -3.286_904_563_383_776_3],
            previous_density_1_0: 4.499_308_188_880_038e1,
        },
    ]
}

fn sample_ff2g_state() -> Ff2gSample {
    let l_count = 4;
    let potential_count = 3;
    let radial_count = 251;
    let scattering_trace = (0..l_count)
        .map(|angular| {
            let l = angular as Real;
            Complex32::new(
                ((0.05_f32 as Real) * l + 0.11_f32 as Real) as f32,
                ((-0.03_f32 as Real) * l + 0.07_f32 as Real) as f32,
            )
        })
        .collect::<Array1<_>>();
    let scattering_ldos = (0..l_count)
        .map(|angular| {
            let l = angular as Real;
            Complex::new(0.2 + 0.04 * l, -0.13 + 0.02 * l)
        })
        .collect::<Array1<_>>();
    let mut embedded_ldos = Array2::<Complex>::zeros((l_count, potential_count));
    let mut previous_ldos = Array2::<Complex>::zeros((l_count, potential_count));
    for angular in 0..l_count {
        let l = angular as Real;
        for potential in 0..potential_count {
            let p = potential as Real;
            embedded_ldos[(angular, potential)] =
                Complex::new(0.4 + 0.03 * l + 0.02 * p, -0.2 + 0.01 * l - 0.015 * p);
            previous_ldos[(angular, potential)] =
                Complex::new(-0.1 + 0.025 * l + 0.01 * p, 0.08 - 0.02 * l + 0.005 * p);
        }
    }
    let embedded_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as Real;
            Complex::new(0.05 * r, -0.02 * r)
        })
        .collect::<Array1<_>>();
    let previous_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as Real;
            Complex::new(-0.03 * r, 0.04 * r)
        })
        .collect::<Array1<_>>();
    let valence_density = (1..=radial_count)
        .map(|radial| 0.01 * radial as Real)
        .collect::<Array1<_>>();
    let mut scattering_density = Array2::<Complex>::zeros((radial_count, l_count));
    for radial in 0..radial_count {
        let r = (radial + 1) as Real;
        for angular in 0..l_count {
            let l = angular as Real;
            scattering_density[(radial, angular)] =
                Complex::new(0.006 * r + 0.02 * l, -0.004 * r + 0.015 * l);
        }
    }
    let occupancy_by_l = (0..l_count)
        .map(|angular| -0.03 + 0.015 * angular as Real)
        .collect::<Array1<_>>();

    Ff2gSample {
        scattering_trace,
        scattering_ldos,
        embedded_ldos,
        previous_ldos,
        scattering_density,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
    }
}

fn sample_coulom_state() -> CoulomSample {
    let last_indices = Array1::from_vec(vec![140, 132]);
    let atom_potentials = Array1::from_vec(vec![0, 1, 1]);
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let norman_radii = Array1::from_vec(vec![0.65, 0.82]);
    let charge_deltas = Array1::from_vec(vec![0.15, -0.07]);
    let atomic_numbers = Array1::from_vec(vec![8, 14]);
    let mut atom_positions = Array2::<Real>::zeros((3, 3));
    for (atom, position) in [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.1, 0.0]]
        .into_iter()
        .enumerate()
    {
        for axis in 0..3 {
            atom_positions[(atom, axis)] = position[axis];
        }
    }

    let mut valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    let mut overlapped_valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    let mut overlapped_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    let mut coulomb_potential = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    for potential in 0..=1 {
        let p = potential as Real;
        for index in 1..=OVRLP_DENSITY_POINTS {
            let radius = (-8.8 + 0.05 * (index - 1) as Real).exp();
            let density = (80.0 + 15.0 * p) * (-0.85 * radius).exp() / (1.0 + 0.12 * radius);
            overlapped_density[(index - 1, potential)] = density;
            overlapped_valence_density[(index - 1, potential)] = (0.42 + 0.03 * p) * density;
            valence_density[(index - 1, potential)] = (0.36 + 0.02 * p) * density;
            coulomb_potential[(index - 1, potential)] = -1.7 - 0.25 * p + 0.004 * index as Real;
        }
    }

    CoulomSample {
        last_indices,
        valence_density,
        overlapped_valence_density,
        overlapped_density,
        atom_positions,
        representative_atoms,
        atom_potentials,
        norman_radii,
        charge_deltas,
        atomic_numbers,
        coulomb_potential,
    }
}

fn sample_ovrlp_state() -> OvrlpSample {
    let atom_potentials = Array1::from_vec(vec![0, 1, 2, 1]);
    let mut atom_positions = Array2::<Real>::zeros((4, 3));
    for (atom, position) in [
        [0.0, 0.0, 0.0],
        [1.35, 0.2, -0.15],
        [3.10, -0.4, 0.25],
        [13.5, 0.0, 0.0],
    ]
    .into_iter()
    .enumerate()
    {
        for axis in 0..3 {
            atom_positions[(atom, axis)] = position[axis];
        }
    }
    let representative_atoms = Array1::from_vec(vec![0, 1, 2]);
    let atomic_numbers = Array1::from_vec(vec![6, 8, 14]);
    let mut electron_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    let mut spin_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    let mut valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    let mut coulomb_potential = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    for potential in 0..4 {
        let p = potential as Real;
        for index in 1..=OVRLP_DENSITY_POINTS {
            let i = index as Real;
            let radius = legacy_loucks_radius(index);
            let density = (45.0 + 18.0 * p) * (-(1.0 + 0.08 * p) * radius).exp() + 0.05 * (i + p);
            electron_density[(index - 1, potential)] = density;
            valence_density[(index - 1, potential)] = 0.65 * density + 0.01 * p + 0.0002 * i;
            coulomb_potential[(index - 1, potential)] =
                -2.0 - 0.12 * p + 0.004 * i + 0.03 * (0.05 * i + p).cos();
            spin_density[(index - 1, potential)] = 0.02 + 0.0003 * i + 0.005 * p;
        }
    }

    OvrlpSample {
        atom_potentials,
        atom_positions,
        representative_atoms,
        atomic_numbers,
        electron_density,
        spin_density,
        valence_density,
        coulomb_potential,
    }
}

fn legacy_loucks_radius(index_1based: usize) -> Real {
    ((0.05_f32 as Real) * (index_1based as Real - 1.0) - 8.8_f32 as Real).exp()
}

fn assert_coulom_values(values: &Array2<Real>, potential: usize, expected: [Real; 6]) {
    let indices = [
        1,
        2,
        64,
        if potential == 0 { 140 } else { 132 },
        if potential == 0 { 141 } else { 133 },
        251,
    ];
    for (index, expected_value) in indices.into_iter().zip(expected) {
        assert_close(values[(index - 1, potential)], expected_value);
    }
}

fn assert_broydn_grid_values(
    values: &Array2<Real>,
    potential: usize,
    last_index: usize,
    expected: [Real; 4],
) {
    for (index, expected_value) in [1, 2, 40, last_index].into_iter().zip(expected) {
        assert_close(values[(index - 1, potential)], expected_value);
    }
}

fn assert_overlap_grid_values(values: &Array1<Real>, expected: [Real; 4]) {
    const OVRLP_ORACLE_TOLERANCE: Real = 5.0e-7;

    for (index, expected_value) in [1, 25, 100, 180].into_iter().zip(expected) {
        assert!(
            (values[index - 1] - expected_value).abs() <= OVRLP_ORACLE_TOLERANCE,
            "{} != {}",
            values[index - 1],
            expected_value
        );
    }
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

fn assert_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-8_f64.max(expected.abs() * 1.0e-12);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}
