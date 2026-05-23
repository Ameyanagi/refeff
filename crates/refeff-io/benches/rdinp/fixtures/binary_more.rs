use super::*;

pub(crate) fn fms_bin_bench_data() -> FmsBinData {
    let energy_count = 256;
    let spectrum_count = 4;
    FmsBinData {
        cluster_radius_angstrom: 6.25,
        energy_count,
        main_energy_count: 192,
        auxiliary_energy_count: 16,
        highest_potential_index: 5,
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        declared_spectrum_count: Some(spectrum_count),
        spectra: Array2::from_shape_fn((spectrum_count, energy_count), |(spectrum, energy)| {
            Complex64::new(
                0.001 * (energy + 1) as f64 + spectrum as f64 * 0.01,
                -0.0005 * (energy + 1) as f64 - spectrum as f64 * 0.005,
            )
        }),
    }
}

pub(crate) fn gtr_bin_bench_data() -> GtrBinData {
    let energy_count = 256;
    let potential_count = 4;
    let angular_channel_count = 4;
    GtrBinData {
        point_count_declared: energy_count,
        horizontal_count: 192,
        danes_extension_count: 0,
        highest_potential_index: potential_count - 1,
        fms_mode: 2,
        values: Array3::from_shape_fn(
            (energy_count, potential_count, angular_channel_count),
            |(energy, potential, angular)| {
                Complex64::new(
                    0.001 * (energy + 1) as f64 + 0.01 * potential as f64 + 0.02 * angular as f64,
                    -0.0005 * (energy + 1) as f64
                        - 0.005 * potential as f64
                        - 0.01 * angular as f64,
                )
            },
        ),
    }
}

pub(crate) fn fmsl_bin_bench_data() -> FmslBinData {
    let energy_count = 256;
    let max_decomposition_channel = 4;
    let channel_count = max_decomposition_channel + 1;
    FmslBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        max_decomposition_channel,
        traces: Array3::from_shape_fn(
            (energy_count, channel_count, channel_count),
            |(energy, lg2, lg1)| {
                Complex64::new(
                    0.001 * (energy + 1) as f64 + 0.01 * lg2 as f64 + 0.02 * lg1 as f64,
                    -0.0005 * (energy + 1) as f64 - 0.005 * lg2 as f64 - 0.01 * lg1 as f64,
                )
            },
        ),
    }
}

pub(crate) fn xsecl_dat_bench_data() -> XseclDatData {
    let energy_count = 192;
    let channel_count = 11;
    let channel_cross_sections =
        Array2::from_shape_fn((energy_count, channel_count), |(energy, channel)| {
            let scale = (energy + 1) as f64;
            Complex64::new(
                1.0e-4 * scale / (channel + 1) as f64,
                -8.0e-5 * scale / (channel + 2) as f64,
            )
        });
    let channel_sum = Array1::from_shape_fn(energy_count, |energy| {
        channel_cross_sections.row(energy).iter().copied().sum()
    });
    XseclDatData {
        header: XseclDatHeader {
            real_energy_count: 157,
            fermi_index: 11,
            edge: -0.196_469_493_817_166_7,
            emu: 408.320_206_199_998_44,
            core_hole_width: 8.394_938_649_968_564e-2,
        },
        energy: Array1::from_shape_fn(energy_count, |energy| 408.083_58 + 0.003_5 * energy as f64),
        channel_cross_sections,
        channel_sum,
    }
}

pub(crate) fn xsecl_bin_bench_data() -> XseclBinData {
    let energy_count = 256;
    let final_state_count = 12;
    XseclBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        initial_state_j: 1,
        transitions: (0..8)
            .map(|index| XseclBinTransition {
                final_state_kappa: if index % 2 == 0 {
                    -((index / 2) + 1)
                } else {
                    (index / 2) + 1
                },
                decomposition_channel: index % 4,
                total_angular_momentum_channel: index % 5,
                orbital_angular_momentum: index % 4,
            })
            .collect(),
        atom_cross_sections: Array2::from_shape_fn(
            (energy_count, final_state_count),
            |(energy, final_state)| {
                Complex64::new(
                    0.002 * (energy + 1) as f64 + 0.01 * final_state as f64,
                    -0.001 * (energy + 1) as f64 - 0.005 * final_state as f64,
                )
            },
        ),
        raw_atom_cross_section_pad: None,
    }
}

pub(crate) fn feffl_bin_bench_data() -> FefflBinData {
    let path_count = 64;
    let energy_count = 128;
    let max_decomposition_channel = 2;
    let channel_count = max_decomposition_channel + 1;
    FefflBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        max_decomposition_channel,
        amplitudes: Array4::from_shape_fn(
            (path_count, channel_count, channel_count, energy_count),
            |(path, lg2, lg1, energy)| {
                0.01 * (path + 1) as f64
                    + 0.001 * lg2 as f64
                    + 0.002 * lg1 as f64
                    + 0.0001 * energy as f64
            },
        ),
        phases: Array4::from_shape_fn(
            (path_count, channel_count, channel_count, energy_count),
            |(path, lg2, lg1, energy)| {
                -0.005 * (path + 1) as f64
                    - 0.0005 * lg2 as f64
                    - 0.001 * lg1 as f64
                    - 0.00005 * energy as f64
            },
        ),
    }
}
