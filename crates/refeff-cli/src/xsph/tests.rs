use super::{has_cached_xsph_output, run_in_dir};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use refeff_io::{
    AxafsDatData, EmeshBinData, EmeshDatData, ModuleLogData, MpseDatData, PhaseBinData,
    PhaseBinPotential, PhaseBinScalars, XseclBinData, XseclBinTransition, XseclDatData,
    XseclDatHeader, XsectDatData, XsectDatScalars, XsphAdvanced, XsphControl, XsphGrid, XsphInput,
    read_axafs_dat, read_emesh_bin, read_emesh_dat, read_module_log_dat, read_mpse_dat,
    read_phase_bin, read_xsecl_bin, read_xsecl_dat, read_xsecl2_dat, read_xsect_dat,
    write_axafs_dat, write_emesh_bin, write_emesh_dat, write_module_log_dat, write_mpse_dat,
    write_phase_bin, write_xsecl_bin, write_xsecl_dat, write_xsecl2_dat, write_xsect_dat,
    xsph_input_string,
};
use std::path::{Path, PathBuf};

#[test]
fn xsph_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_xsph_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_rejects_generation_until_solver_is_ported() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled XSPH should require the numerical solver")?;

    assert!(
        error
            .to_string()
            .contains("XSPH phase-shift generation requires the unported XSPH numerical solver")
    );
    Ok(())
}

#[test]
fn xsph_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let axafs_path = temp.path().join("axafs.dat");
    let xsecl_path = temp.path().join("xsecl.dat");
    let xsecl2_path = temp.path().join("xsecl2.dat");
    let xsecl_bin_path = temp.path().join("xsecl.bin");
    let mpse_path = temp.path().join("mpse.dat");
    let emesh_path = temp.path().join("emesh.dat");
    let emesh_bin_path = temp.path().join("emesh.bin");
    let log2_path = temp.path().join("log2.dat");

    write_phase_bin(&phase_path, &sample_phase_bin())?;
    write_xsect_dat(&xsect_path, &sample_xsect_dat())?;
    write_axafs_dat(&axafs_path, &sample_axafs_dat())?;
    write_xsecl_dat(&xsecl_path, &sample_xsecl_dat())?;
    write_xsecl2_dat(&xsecl2_path, &sample_xsecl_dat())?;
    write_xsecl_bin(&xsecl_bin_path, &sample_xsecl_bin())?;
    write_mpse_dat(&mpse_path, &sample_mpse_dat())?;
    write_emesh_dat(&emesh_path, &sample_emesh_dat())?;
    write_emesh_bin(&emesh_bin_path, &sample_emesh_bin())?;
    write_module_log_dat(&log2_path, &sample_module_log())?;
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let expected_axafs = read_axafs_dat(&axafs_path)?;
    let expected_xsecl = read_xsecl_dat(&xsecl_path)?;
    let expected_xsecl2 = read_xsecl2_dat(&xsecl2_path)?;
    let expected_xsecl_bin = read_xsecl_bin(
        &xsecl_bin_path,
        expected_phase.pad_width,
        expected_phase.energy_count,
    )?;
    let expected_mpse = read_mpse_dat(&mpse_path)?;
    let expected_emesh = read_emesh_dat(&emesh_path)?;
    let expected_emesh_bin = read_emesh_bin(&emesh_bin_path)?;
    let expected_log = read_module_log_dat(&log2_path)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 10);
    assert!(has_cached_xsph_output(temp.path())?);
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_eq!(read_xsect_dat(&xsect_path)?, expected_xsect);
    assert_eq!(read_axafs_dat(&axafs_path)?, expected_axafs);
    assert_eq!(read_xsecl_dat(&xsecl_path)?, expected_xsecl);
    assert_eq!(read_xsecl2_dat(&xsecl2_path)?, expected_xsecl2);
    assert_eq!(
        read_xsecl_bin(
            &xsecl_bin_path,
            expected_phase.pad_width,
            expected_phase.energy_count
        )?,
        expected_xsecl_bin
    );
    assert_eq!(read_mpse_dat(&mpse_path)?, expected_mpse);
    assert_eq!(read_emesh_dat(&emesh_path)?, expected_emesh);
    assert_eq!(read_emesh_bin(&emesh_bin_path)?, expected_emesh_bin);
    assert_eq!(read_module_log_dat(&log2_path)?, expected_log);
    Ok(())
}

#[test]
fn xsph_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_xsph_dir()? else {
        eprintln!("skipping XSPH reference test; generated EXAFS/Cu reference not found");
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "phase.bin", "xsect.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    for name in [
        "xsecl.dat",
        "xsecl2.dat",
        "xsecl.bin",
        "axafs.dat",
        "mpse.dat",
        "emesh.dat",
        "emesh.bin",
        "log2.dat",
    ] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }

    let expected_phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    let expected_axafs = optional_axafs_dat(temp.path().join("axafs.dat"))?;
    let expected_xsecl = optional_xsecl_dat(temp.path().join("xsecl.dat"))?;
    let expected_xsecl2 = optional_xsecl2_dat(temp.path().join("xsecl2.dat"))?;
    let expected_xsecl_bin = optional_xsecl_bin(
        temp.path().join("xsecl.bin"),
        expected_phase.pad_width,
        expected_phase.energy_count,
    )?;
    let expected_mpse = optional_mpse_dat(temp.path().join("mpse.dat"))?;
    let expected_emesh = optional_emesh_dat(temp.path().join("emesh.dat"))?;
    let expected_emesh_bin = optional_emesh_bin(temp.path().join("emesh.bin"))?;
    let expected_log = optional_module_log(temp.path().join("log2.dat"))?;

    let count = run_in_dir(temp.path())?;

    let optional_count = [
        expected_xsecl.as_ref().map(|_| 1_usize),
        expected_xsecl2.as_ref().map(|_| 1_usize),
        expected_xsecl_bin.as_ref().map(|_| 1_usize),
        expected_axafs.as_ref().map(|_| 1_usize),
        expected_mpse.as_ref().map(|_| 1_usize),
        expected_emesh.as_ref().map(|_| 1_usize),
        expected_emesh_bin.as_ref().map(|_| 1_usize),
        expected_log.as_ref().map(|_| 1_usize),
    ]
    .into_iter()
    .flatten()
    .sum::<usize>();
    assert_eq!(count, 2 + optional_count);
    assert_eq!(
        read_phase_bin(temp.path().join("phase.bin"))?,
        expected_phase
    );
    assert_eq!(
        read_xsect_dat(temp.path().join("xsect.dat"))?,
        expected_xsect
    );
    if let Some(expected) = expected_axafs {
        assert_eq!(read_axafs_dat(temp.path().join("axafs.dat"))?, expected);
    }
    if let Some(expected) = expected_xsecl {
        assert_eq!(read_xsecl_dat(temp.path().join("xsecl.dat"))?, expected);
    }
    if let Some(expected) = expected_xsecl2 {
        assert_eq!(read_xsecl2_dat(temp.path().join("xsecl2.dat"))?, expected);
    }
    if let Some(expected) = expected_xsecl_bin {
        assert_eq!(
            read_xsecl_bin(
                temp.path().join("xsecl.bin"),
                expected_phase.pad_width,
                expected_phase.energy_count
            )?,
            expected
        );
    }
    if let Some(expected) = expected_mpse {
        assert_eq!(read_mpse_dat(temp.path().join("mpse.dat"))?, expected);
    }
    if let Some(expected) = expected_emesh {
        assert_eq!(read_emesh_dat(temp.path().join("emesh.dat"))?, expected);
    }
    if let Some(expected) = expected_emesh_bin {
        assert_eq!(read_emesh_bin(temp.path().join("emesh.bin"))?, expected);
    }
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log2.dat"))?, expected);
    }
    Ok(())
}

fn write_xsph_input(work_dir: &Path, mphase: i32) -> Result<()> {
    let input = XsphInput {
        control: XsphControl {
            mphase,
            ipr2: 0,
            ixc: 0,
            ixc0: 0,
            ispec: 0,
            lreal: 0,
            lfms2: 0,
            nph: 1,
            l2lp: 0,
            i_plsmn: 0,
            n_poles: 100,
            i_gamma_ch: 0,
            i_grid: 0,
            i_core_state: -1,
            iscfxc: 11,
        },
        vr0: 0.0,
        vi0: 0.0,
        lmaxph: vec![1, 1],
        pot_labels: vec!["Cu".to_string(), "O".to_string()],
        grid: XsphGrid {
            rgrd: 0.05,
            rfms2: 0.0,
            gamach: 1.0,
            xkstep: 0.05,
            xkmax: 10.0,
            vixan: 0.0,
            eps0: 0.0,
            egap: 0.0,
        },
        spinph: vec![0.0, 0.0],
        advanced: XsphAdvanced {
            izstd: 0,
            ifxc: 0,
            ipmbse: 0,
            itdlda: 0,
            nonlocal: 0,
            ibasis: 0,
        },
        electronic_temperature: 0.0,
        chsh_type: 0,
        decomposition_channels: -1,
        lopt: false,
        print_rl: false,
    };
    std::fs::write(work_dir.join("xsph.inp"), xsph_input_string(&input)?)?;
    Ok(())
}

fn sample_phase_bin() -> PhaseBinData {
    let spin_count = 1;
    let energy_count = 2;
    let transition_count = 2;
    let q_count = 1;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 2,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: 4,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: -0.35,
            edge_energy: 9.8,
        },
        energy_grid: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + energy as f64, 0.01 * energy as f64)
        }),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
            Complex64::new(-1.0 + 0.2 * energy as f64, 0.0)
        }),
        potentials: vec![
            sample_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
            sample_potential(1, 8, "O", energy_count, spin_count, 0.2),
        ],
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

fn sample_potential(
    lmax: usize,
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
    scale: f64,
) -> PhaseBinPotential {
    let l_count = 2 * lmax + 1;
    PhaseBinPotential {
        lmax,
        atomic_number,
        label: label.to_string(),
        phase_shifts: Array3::from_shape_fn(
            (energy_count, l_count, spin_count),
            |(energy, l_slot, spin)| {
                Complex64::new(
                    scale + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        ),
    }
}

fn sample_xsect_dat() -> XsectDatData {
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
        main_energy_count: 2,
        fermi_index: 1,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(1.25, 0.01),
            Complex64::new(1.5, 0.02),
        ]),
        normalized_background: Array1::from_vec(vec![2.0, 2.5]),
        cross_section: Array1::from_vec(vec![Complex64::new(3.0, -0.4), Complex64::new(3.5, -0.5)]),
    }
}

fn sample_axafs_dat() -> AxafsDatData {
    AxafsDatData {
        header_lines: vec![
            " # File contains AXAFS. See manual for details.".to_string(),
            " #--------------------------------------------------------------".to_string(),
            " #  e, e(wrt edge), k, mu_at=(1+chi_at)*mu0_at, mu0_at, chi_at @#".to_string(),
        ],
        energy_ev: Array1::from_vec(vec![8979.0, 8980.5]),
        edge_relative_energy_ev: Array1::from_vec(vec![0.0, 1.5]),
        wave_number_inverse_angstrom: Array1::from_vec(vec![0.0, 0.627]),
        atomic_absorption: Array1::from_vec(vec![1.234_56, 1.061_11]),
        atomic_background: Array1::from_vec(vec![1.0, 1.111_11]),
        chi_atomic: Array1::from_vec(vec![0.234_56, -0.045]),
    }
}

fn sample_xsecl_dat() -> XseclDatData {
    XseclDatData {
        header: XseclDatHeader {
            real_energy_count: 2,
            fermi_index: 1,
            edge: -0.25,
            emu: 408.0,
            core_hole_width: 0.083_949_386_5,
        },
        energy: Array1::from_vec(vec![408.083_58, 408.118_59]),
        channel_cross_sections: Array2::from_shape_fn((2, 2), |(energy, channel)| {
            let real = match (energy, channel) {
                (0, 0) => -0.000_094_722_801,
                (0, 1) => 0.000_058_529_371,
                (1, 0) => -0.000_042_446_685,
                (1, 1) => -0.000_117_763_55,
                _ => 0.0,
            };
            let imag = match (energy, channel) {
                (0, 0) => 0.000_115_562_54,
                (0, 1) => -0.000_120_865_91,
                (1, 0) => 0.000_105_705_03,
                (1, 1) => -0.000_144_091_45,
                _ => 0.0,
            };
            Complex64::new(real, imag)
        }),
        channel_sum: Array1::from_vec(vec![
            Complex64::new(-0.000_036_126_732, -0.000_005_278_514_8),
            Complex64::new(-0.000_160_211_14, -0.000_038_440_289),
        ]),
    }
}

fn sample_xsecl_bin() -> XseclBinData {
    XseclBinData {
        pad_width: 8,
        initial_state_j: 1,
        transitions: vec![
            XseclBinTransition {
                final_state_kappa: -1,
                decomposition_channel: 0,
                total_angular_momentum_channel: 0,
                orbital_angular_momentum: 0,
            },
            XseclBinTransition {
                final_state_kappa: 2,
                decomposition_channel: 1,
                total_angular_momentum_channel: 1,
                orbital_angular_momentum: 1,
            },
        ],
        atom_cross_sections: Array2::from_shape_fn((2, 4), |(energy, final_state)| {
            Complex64::new(
                0.1 * (energy + 1) as f64 + 0.01 * final_state as f64,
                -0.05 * (energy + 1) as f64 - 0.005 * final_state as f64,
            )
        }),
        raw_atom_cross_section_pad: None,
    }
}

fn sample_mpse_dat() -> MpseDatData {
    MpseDatData {
        header_lines: vec!["# XSPH MPSE self-energy sidecar".to_string()],
        energy_ev: Array1::from_vec(vec![0.038_099_840_30, 0.152_399_361_2]),
        self_energy: Array1::from_vec(vec![
            Complex64::new(0.001_436_696_198, -0.000_007_842_984_015),
            Complex64::new(0.005_774_807_411, -0.000_124_742_315_9),
        ]),
        renormalization: Some(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ])),
        renormalization_magnitude: Some(Array1::from_vec(vec![1.0, 1.0])),
        renormalization_phase: Some(Array1::from_vec(vec![0.0, 0.0])),
        inelastic_mean_free_path: Some(Array1::from_vec(vec![48_578.245_52, 6_108.567_091])),
    }
}

fn sample_emesh_dat() -> EmeshDatData {
    EmeshDatData {
        edge_hartree: 333.333,
        bohr_angstrom: 0.529_177_249,
        edge_ev: 9_071.2,
        spectrum: 0,
        fermi_index: 1,
        indices: Array1::from_vec(vec![1, 2, 3]),
        energy_ev: Array1::from_vec(vec![0.0, 1.5, 3.0]),
        wave_number_inverse_angstrom: Array1::from_vec(vec![0.0, 0.627, 0.887]),
    }
}

fn sample_emesh_bin() -> EmeshBinData {
    EmeshBinData {
        point_count_declared: 3,
        horizontal_count: 2,
        danes_extension_count: 1,
        energy_hartree: Array1::from_vec(vec![
            Complex64::new(-0.25, 0.01),
            Complex64::new(0.0, 0.02),
            Complex64::new(0.5, 0.03),
        ]),
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating potentials and phases ...".to_string(),
            "Done with module: potentials and phases.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn optional_xsecl_dat(path: impl AsRef<Path>) -> Result<Option<XseclDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_xsecl_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_axafs_dat(path: impl AsRef<Path>) -> Result<Option<AxafsDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_axafs_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_xsecl2_dat(path: impl AsRef<Path>) -> Result<Option<XseclDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_xsecl2_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_xsecl_bin(
    path: impl AsRef<Path>,
    pad_width: usize,
    energy_count: usize,
) -> Result<Option<XseclBinData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_xsecl_bin(path, pad_width, energy_count)?))
    } else {
        Ok(None)
    }
}

fn optional_mpse_dat(path: impl AsRef<Path>) -> Result<Option<MpseDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_mpse_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_emesh_dat(path: impl AsRef<Path>) -> Result<Option<EmeshDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_emesh_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_emesh_bin(path: impl AsRef<Path>) -> Result<Option<EmeshBinData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_emesh_bin(path)?))
    } else {
        Ok(None)
    }
}

fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_module_log_dat(path)?))
    } else {
        Ok(None)
    }
}

fn reference_xsph_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    let required = ["xsph.inp", "phase.bin", "xsect.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}
