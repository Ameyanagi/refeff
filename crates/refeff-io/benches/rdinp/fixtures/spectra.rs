use super::*;

pub(crate) fn xsect_dat_bench_data() -> XsectDatData {
    let energy_count = 256;
    XsectDatData {
        titles: vec!["Cu crystal".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            edge_energy: 9.1,
            chemical_potential: -0.4,
        },
        core_hole_width_ev: 1.23,
        main_energy_count: 192,
        fermi_index: 24,
        energy_grid_ev: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.25 * energy as f64, 0.01 * energy as f64)
        }),
        normalized_background: Array1::from_shape_fn(energy_count, |energy| {
            1.0 + 0.002 * energy as f64
        }),
        cross_section: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + 0.001 * energy as f64, -0.1 - 0.0005 * energy as f64)
        }),
    }
}

pub(crate) fn xmu_dat_bench_data() -> XmuDatData {
    let point_count = 512;
    XmuDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0.0"
                .to_string(),
            "#     0/   0 paths used".to_string(),
            "#  xsedge+ 50, used to normalize mu           1.2667E-04".to_string(),
            "#  -----------------------------------------------------------------------"
                .to_string(),
            "#  omega    e    k    mu    mu0     chi     @#".to_string(),
        ],
        normalization: Some(1.2667e-4),
        photon_energy_ev: Array1::from_shape_fn(point_count, |index| 8979.0 + 0.5 * index as f64),
        relative_energy_ev: Array1::from_shape_fn(point_count, |index| -40.0 + 0.5 * index as f64),
        wave_number: Array1::from_shape_fn(point_count, |index| -3.0 + 0.02 * index as f64),
        mu: Array1::from_shape_fn(point_count, |index| 0.01 + 0.0001 * index as f64),
        mu0: Array1::from_shape_fn(point_count, |index| 0.009 + 0.00008 * index as f64),
        chi: Array1::from_shape_fn(point_count, |index| 0.001 + 0.00002 * index as f64),
    }
}

pub(crate) fn opcons_dat_bench_data() -> OpconsDatData {
    let point_count = 4096;
    OpconsDatData {
        header_lines: vec![
            "# Cu K".to_string(),
            "#   omega (eV)      epsilon_1       epsilon_2       n               kappa           mu (cm^(-1))    R               epsinv".to_string(),
        ],
        energy_ev: Array1::from_shape_fn(point_count, |index| {
            10.0 + 50_000.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon_minus_one: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.2 + 0.03 * phase.sin(), 0.1 + 0.02 * phase.cos())
        }),
        refractive_index_minus_one: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.05 + 0.005 * phase.cos(), 0.02 + 0.004 * phase.sin())
        }),
        absorption_coefficient: Array1::from_shape_fn(point_count, |index| {
            1000.0 + 5.0 * index as f64
        }),
        reflectivity: Array1::from_shape_fn(point_count, |index| 0.01 + 0.000001 * index as f64),
        loss: Array1::from_shape_fn(point_count, |index| 0.02 + 0.000002 * index as f64),
    }
}

pub(crate) fn eps_dat_bench_data() -> EpsDatData {
    let point_count = 4096;
    EpsDatData {
        header_lines: vec!["# FULLSPECTRUM eps.dat".to_string()],
        omega: Array1::from_shape_fn(point_count, |index| {
            0.01 + 10.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.2 + 0.03 * phase.sin(), 0.1 + 0.02 * phase.cos())
        }),
        background_epsilon: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.15 + 0.02 * phase.cos(), 0.08 + 0.015 * phase.sin())
        }),
        sigma: Array1::from_shape_fn(point_count, |index| 0.001 + 0.000001 * index as f64),
    }
}

pub(crate) fn xmul_dat_bench_data() -> XmulDatData {
    let point_count = 512;
    let max_decomposition_channel = 2;
    let channel_count = max_decomposition_channel + 1;
    XmulDatData {
        header_lines: vec![
            "#  Decomposition of S(q,w) for a single electron".to_string(),
            "#  omega    k   S^0(qw)  S_{l=0,...,ldecmx}^0(qw)       chi^q_{l=0,..ldecmx,l^*=0,...,ldecmx}".to_string(),
            "# and ldecmx=     2".to_string(),
        ],
        max_decomposition_channel,
        photon_energy_ev: Array1::from_shape_fn(point_count, |index| {
            11_100.0 + 0.571 * index as f64
        }),
        wave_number: Array1::from_shape_fn(point_count, |index| -1.3 + 0.05 * index as f64),
        total_single_electron: Array1::from_shape_fn(point_count, |index| {
            2.0e-6 * (1.0 + 0.01 * index as f64)
        }),
        channel_background: Array2::from_shape_fn((point_count, channel_count), |(row, channel)| {
            1.0e-7 * (channel + 1) as f64 * (1.0 + 0.005 * row as f64)
        }),
        normalized_fine_structure: Array3::from_shape_fn(
            (point_count, channel_count, channel_count),
            |(row, l_star, channel)| {
                0.05 * (channel + 1) as f64 / (l_star + 1) as f64
                    * (1.0 + 0.001 * row as f64)
            },
        ),
    }
}

pub(crate) fn chi_dat_bench_data() -> ChiDatData {
    let point_count = 512;
    ChiDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0.0"
                .to_string(),
            "#     0/   0 paths used".to_string(),
            "#  -----------------------------------------------------------------------"
                .to_string(),
            "#       k          chi          mag           phase @#".to_string(),
        ],
        wave_number: Array1::from_shape_fn(point_count, |index| 0.05 * index as f64),
        chi: Array1::from_shape_fn(point_count, |index| {
            (0.04 * index as f64).sin() * (-0.001 * index as f64).exp()
        }),
        magnitude: Array1::from_shape_fn(point_count, |index| {
            0.25 * (-0.0005 * index as f64).exp()
        }),
        phase: Array1::from_shape_fn(point_count, |index| -2.7 + 0.01 * index as f64),
        phase_minus_2kr: None,
        ckp_real: None,
        ckp_imag: None,
    }
}

pub(crate) fn eels_dat_bench_data() -> EelsDatData {
    let point_count = 512;
    EelsDatData {
        header_lines: vec![
            "# Orientation sensitive EELS calculation - beam energy =   300.keV".to_string(),
            "# Units are a_0^2 / eV.  Multiply by 28.00 10^-18  to get cm^-2 / eV.".to_string(),
            format!(
                "#  Energy       total         atomic-bg     fine-struct   {}",
                EELS_TENSOR_LABELS.join("            ")
            ),
        ],
        energy_loss_ev: Array1::from_shape_fn(point_count, |index| 8979.0 + 0.25 * index as f64),
        total: Array1::from_shape_fn(point_count, |index| 1.0e-12 + 1.0e-15 * index as f64),
        atomic_background: Array1::from_shape_fn(point_count, |index| {
            1.2e-12 + 0.8e-15 * index as f64
        }),
        fine_structure: Array1::from_shape_fn(point_count, |index| {
            -0.2e-12 + 0.2e-15 * index as f64
        }),
        tensor: Some(Array2::from_shape_fn(
            (point_count, EELS_TENSOR_LABELS.len()),
            |(row, column)| 1.0e-14 * (column + 1) as f64 + 1.0e-18 * row as f64,
        )),
    }
}

pub(crate) fn danes_dat_bench_data() -> DanesDatData {
    let point_count = 512;
    DanesDatData {
        header_lines: vec!["# E  matsub. sommerf. anomal. tale, total, differ.".to_string()],
        energy_ev: Array1::from_shape_fn(point_count, |index| -100.0 + 0.5 * index as f64),
        matsubara: Array1::from_shape_fn(point_count, |_| 0.0),
        sommerfeld: Array1::from_shape_fn(point_count, |index| 1.0e-4 * index as f64),
        anomalous: Array1::from_shape_fn(point_count, |index| 8.0 + (0.01 * index as f64).sin()),
        tail: Array1::from_shape_fn(point_count, |index| 4.0 + 0.001 * index as f64),
        total: Array1::from_shape_fn(point_count, |index| 4.5 + 0.0015 * index as f64),
        difference: Array1::from_shape_fn(point_count, |index| -5.0 + 0.002 * index as f64),
    }
}

pub(crate) fn ldos_dat_bench_data() -> LdosDatData {
    let point_count = 512;
    LdosDatData {
        header_lines: vec![
            "#  Fermi level (eV): -14.683".to_string(),
            "#  Charge transfer :   0.711".to_string(),
            "#    Electron counts for each orbital momentum:".to_string(),
            "#       0      1.428".to_string(),
            "#       1      1.637".to_string(),
            "#       2     10.223".to_string(),
            "#       3      0.000".to_string(),
            "#  Number of atoms in cluster:   0".to_string(),
            "#  Lorentzian broadening with HWHH     0.0100 eV".to_string(),
            "# -----------------------------------------------------------------------".to_string(),
            "#      e        sDOS           pDOS          dDOS          fDOS    @#".to_string(),
        ],
        fermi_level_ev: Some(-14.683),
        charge_transfer: Some(0.711),
        electron_counts: vec![
            LdosElectronCount {
                angular_momentum: 0,
                count: 1.428,
            },
            LdosElectronCount {
                angular_momentum: 1,
                count: 1.637,
            },
            LdosElectronCount {
                angular_momentum: 2,
                count: 10.223,
            },
            LdosElectronCount {
                angular_momentum: 3,
                count: 0.0,
            },
        ],
        atom_count: Some(0),
        lorentzian_hwhh_ev: Some(0.0100),
        energy_ev: Array1::from_shape_fn(point_count, |index| -30.0 + 0.45 * index as f64),
        density: Array2::from_shape_fn((point_count, 4), |(row, column)| {
            1.0e-4 * (column + 1) as f64 * (1.0 + 0.01 * row as f64)
        }),
    }
}

pub(crate) fn compton_dat_bench_data() -> ComptonDatData {
    let point_count = 1000;
    ComptonDatData {
        header_lines: vec![
            " # Compton profile, J(pq)".to_string(),
            " # ns:            32".to_string(),
            " # nphi:          32".to_string(),
            " # nz:            32".to_string(),
            " # nzp:          120".to_string(),
            " # zpmax:   10.0000000000000".to_string(),
            " # temperature (eV):  0.0000000E+00".to_string(),
            " #----------------------------".to_string(),
            " # pq               J".to_string(),
        ],
        ns: Some(32),
        nphi: Some(32),
        nz: Some(32),
        nzp: Some(120),
        zpmax: Some(10.0),
        temperature_ev: Some(0.0),
        momentum: Array1::from_shape_fn(point_count, |index| 5.0 * index as f64 / 999.0),
        profile: Array1::from_shape_fn(point_count, |index| {
            let momentum = 5.0 * index as f64 / 999.0;
            2.75 * (-0.6 * momentum).exp() + 0.02 * (2.0 * momentum).cos()
        }),
    }
}
