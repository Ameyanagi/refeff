use super::*;

pub(crate) fn crpa_dat_bench_data() -> CrpaDatData {
    CrpaDatData {
        header_lines: vec!["U, n, U_Bare".to_string()],
        hubbard_u: 0.197879035252010,
        occupation: 1.0,
        bare_u: 0.694283422651496,
    }
}

pub(crate) fn loss_dat_bench_data() -> LossDatData {
    let point_count = 8192;
    LossDatData {
        header_lines: vec!["# E(eV)    Loss".to_string()],
        energy_ev: Array1::from_shape_fn(point_count, |index| {
            0.01 + 50_000.0 * index as f64 / (point_count - 1) as f64
        }),
        loss: Array1::from_shape_fn(point_count, |index| {
            let energy = 0.01 + 50_000.0 * index as f64 / (point_count - 1) as f64;
            2.0e-6 * (-energy / 25_000.0).exp() + 5.0e-5 / (1.0 + (energy / 25.0).powi(2))
        }),
    }
}

pub(crate) fn osc_str_dat_bench_data() -> OscStrDatData {
    let edges = ["K", "L1", "L2", "L3"];
    OscStrDatData {
        header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
        rows: (0..256)
            .map(|index| OscStrRow {
                component: if index % 2 == 0 {
                    "Cu".to_string()
                } else {
                    "O".to_string()
                },
                edge: edges[index % edges.len()].to_string(),
                core_hole_index: (index % edges.len() + 1) as i32,
                effective_electron_count: 0.5 + 0.01 * index as f64,
            })
            .collect(),
    }
}

pub(crate) fn fullspectrum_edge_assembly_bench_data(
    effective_electron_count: f64,
) -> FullSpectrumEdgeAssembly {
    FullSpectrumEdgeAssembly {
        scattering_factor: Array1::from_elem(2, Complex64::new(0.0, 0.0)),
        background: Array1::from_elem(2, Complex64::new(0.0, 0.0)),
        effective_electron_count,
        zero_energy_fprime: 0.0,
        overlap_points: 1,
    }
}

pub(crate) fn sumrules_dat_bench_data() -> SumRulesDatData {
    let point_count = 8192;
    SumRulesDatData {
        header_lines: Vec::new(),
        energy_ev: Array1::from_shape_fn(point_count, |index| {
            10.0 + 50_000.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon2_effective_electrons: Array1::from_shape_fn(point_count, |index| {
            0.0001 * index as f64 + 0.05 * (index as f64 * 0.001).sin().abs()
        }),
        absorption_effective_electrons: Array1::from_shape_fn(point_count, |index| {
            0.05 * index as f64 + 0.1 * (index as f64 * 0.002).cos().abs()
        }),
        loss_effective_electrons: Array1::from_shape_fn(point_count, |index| {
            0.00008 * index as f64 + 0.02 * (index as f64 * 0.003).sin().abs()
        }),
        absorption_refractive_sum: Array1::from_shape_fn(point_count, |index| {
            0.001 * index as f64 + 0.005 * (index as f64 * 0.004).cos()
        }),
        refractive_index_sum_ratio: Array1::from_shape_fn(point_count, |index| {
            0.8 + 0.2 * (index as f64 * 0.005).sin().abs()
        }),
        log_loss_moment_ratio: Array1::from_shape_fn(point_count, |index| {
            -2.0 + 0.0005 * index as f64
        }),
    }
}

pub(crate) fn drude_dat_bench_data() -> DrudeDatData {
    let point_count = 8192;
    DrudeDatData {
        gamma_ev: 0.658,
        plasma_frequency_ev: 26.417_175_795_207_253,
        omega: Array1::from_shape_fn(point_count, |index| {
            0.01 + 10.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon: Array1::from_shape_fn(point_count, |index| {
            let omega = 0.01 + 10.0 * index as f64 / (point_count - 1) as f64;
            Complex64::new(-1.0 / (1.0 + omega * omega), 0.2 / (omega + 0.1))
        }),
    }
}

pub(crate) fn hamaker_dat_bench_data() -> HamakerDatData {
    let point_count = 8192;
    HamakerDatData {
        header_lines: Vec::new(),
        omega: Array1::from_shape_fn(point_count, |index| {
            0.01 + 10.0 * index as f64 / (point_count - 1) as f64
        }),
        imaginary_axis_epsilon: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.1 + 0.02 * phase.sin(), 0.0)
        }),
    }
}

pub(crate) fn mpse_dat_bench_data() -> MpseDatData {
    let point_count = 1024;
    MpseDatData {
        header_lines: vec!["# E-EFermi Re[Sigma] Im[Sigma] Re[Z] Im[Z]".to_string()],
        energy_ev: Array1::from_shape_fn(point_count, |index| 0.05 + 0.15 * index as f64),
        self_energy: Array1::from_shape_fn(point_count, |index| {
            let energy = 0.05 + 0.15 * index as f64;
            Complex64::new(0.02 * energy.sqrt(), -0.01 * (1.0 + energy).ln())
        }),
        renormalization: Some(Array1::from_shape_fn(point_count, |index| {
            let scale = 1.0 + index as f64 / point_count as f64;
            Complex64::new(1.0 - 0.05 / scale, -0.02 / scale)
        })),
        renormalization_magnitude: None,
        renormalization_phase: None,
        inelastic_mean_free_path: None,
    }
}

pub(crate) fn rixs_map_bench_data() -> RixsMapData {
    let block_count = 64;
    let rows_per_block = 64;
    let point_count = block_count * rows_per_block;
    RixsMapData {
        header_lines: Vec::new(),
        block_lengths: vec![rows_per_block; block_count],
        first_energy_ev: Array1::from_shape_fn(point_count, |index| {
            11_540.0 + (index % rows_per_block) as f64
        }),
        second_energy_ev: Array1::from_shape_fn(point_count, |index| {
            -15.0 + (index / rows_per_block) as f64 * 0.5
        }),
        channels: Array2::from_shape_fn((point_count, 4), |(row, channel)| {
            let local = (row % rows_per_block) as f64;
            let block = (row / rows_per_block) as f64;
            1.0e-6 * (channel + 1) as f64 * (1.0 + 0.01 * local) * (1.0 + 0.005 * block)
        }),
    }
}

pub(crate) fn rixs_line_bench_data() -> RixsLineData {
    let point_count = 512;
    RixsLineData {
        header_lines: Vec::new(),
        energy_ev: Array1::from_shape_fn(point_count, |index| 11_540.0 + index as f64),
        channels: Array2::from_shape_fn((point_count, 4), |(row, channel)| {
            1.0e-5 * (channel + 1) as f64 * (1.0 + 0.01 * row as f64).ln()
        }),
    }
}
