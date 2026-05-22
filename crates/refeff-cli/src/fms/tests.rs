use super::{has_cached_fms_output, run_in_dir};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4, Axis, ShapeBuilder};
use num_complex::Complex64;
use refeff_core::{
    MkgtrGreenTraceInput, TransitionBMatrixInput, core_hole_quantum_numbers, mkgtr_green_trace,
    transition_b_matrix,
};
use refeff_io::{
    CfAverage, FmsBinData, FmsCluster, FmsControl, FmsDebye, FmsInput, FmslBinData, GgDatData,
    GgDatSection, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GtrBinData, GtrDatData,
    GtrlDatData, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars, fms_input_string,
    global_input_string, parse_gtrl_dat, read_fms_bin, read_fmsl_bin, read_gg_bin, read_gg_dat,
    read_gtr_bin, read_gtr_dat, read_gtrl_dat, read_module_log_dat, write_fms_bin, write_fmsl_bin,
    write_gg_bin, write_gg_dat, write_gtr_bin, write_gtr_dat, write_gtrl_dat, write_module_log_dat,
    write_phase_bin,
};
use std::path::{Path, PathBuf};

#[test]
fn fms_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 0, -1)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_fms_output(temp.path())?);
    Ok(())
}

#[test]
fn fms_module_rejects_generation_until_solver_is_ported() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled FMS should require the numerical solver")?;

    assert!(
        error
            .to_string()
            .contains("FMS Green's-function generation requires the unported FMS numerical solver")
    );
    Ok(())
}

#[test]
fn fms_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, 2)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;
    write_fmsl_bin(temp.path().join("fmsl.bin"), &sample_fmsl_bin())?;
    write_gg_bin(temp.path().join("gg.bin"), &sample_gg_dat())?;
    write_gg_dat(temp.path().join("gg.dat"), &sample_gg_dat())?;
    write_gtr_dat(temp.path().join("gtr.dat"), &sample_gtr_dat())?;
    write_gtr_bin(temp.path().join("gtr00.bin"), &sample_gtr_bin())?;
    write_gtrl_dat(temp.path().join("gtrl.dat"), &sample_gtrl_dat()?)?;
    write_module_log_dat(temp.path().join("log3.dat"), &sample_module_log())?;

    let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let expected_fmsl = read_fmsl_bin(
        temp.path().join("fmsl.bin"),
        expected_fms.pad_width,
        expected_fms.energy_count,
        2,
    )?;
    let expected_gg_bin = read_gg_bin(temp.path().join("gg.bin"))?;
    let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
    let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
    let expected_gtr_bin = read_gtr_bin(temp.path().join("gtr00.bin"))?;
    let expected_gtrl = read_gtrl_dat(temp.path().join("gtrl.dat"))?;
    let expected_log = read_module_log_dat(temp.path().join("log3.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 8);
    assert!(has_cached_fms_output(temp.path())?);
    assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
    assert_eq!(
        read_fmsl_bin(
            temp.path().join("fmsl.bin"),
            expected_fms.pad_width,
            expected_fms.energy_count,
            2,
        )?,
        expected_fmsl
    );
    assert_eq!(read_gg_bin(temp.path().join("gg.bin"))?, expected_gg_bin);
    assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
    assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
    assert_eq!(
        read_gtr_bin(temp.path().join("gtr00.bin"))?,
        expected_gtr_bin
    );
    assert_eq!(read_gtrl_dat(temp.path().join("gtrl.dat"))?, expected_gtrl);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn fms_module_generates_mkgtr_outputs_from_cached_gg() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input_with_lmax(temp.path(), 1, -1, &[1])?;
    let global = sample_global_input();
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    let gg = sample_mkgtr_gg();
    write_gg_bin(temp.path().join("gg.bin"), &gg)?;

    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_eq!(fms.declared_spectrum_count, Some(0));
    assert_eq!(fms.energy_count, phase.energy_count);
    assert_eq!(fms.main_energy_count, phase.main_energy_count);
    assert_eq!(fms.highest_potential_index, phase.potential_count() - 1);
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 1.0e-8);
    assert_eq!(gtr.energy, phase.energy_grid);
    assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);
    Ok(())
}

#[test]
fn fms_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_fms_dir()? else {
        eprintln!("skipping FMS reference test; generated EXAFS/Cu reference not found");
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
    for name in required {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    for name in ["gg.bin", "gtrl.dat", "fmsl.bin", "log3.dat"] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    copy_gtr_bin_references(&reference_dir, temp.path())?;

    let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
    let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
    let expected_log = optional_module_log(temp.path().join("log3.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert!(count >= required.len() - 1);
    assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
    assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
    assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log3.dat"))?, expected);
    }
    Ok(())
}

fn write_fms_input(work_dir: &Path, mfms: i32, decomposition_channels: i32) -> Result<()> {
    write_fms_input_with_lmax(work_dir, mfms, decomposition_channels, &[2, 2])
}

fn write_fms_input_with_lmax(
    work_dir: &Path,
    mfms: i32,
    decomposition_channels: i32,
    lmaxph: &[i32],
) -> Result<()> {
    let input = FmsInput {
        control: FmsControl {
            mfms,
            idwopt: 0,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: -1.0,
            rdirec: -1.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 190.0,
            thetad: 315.0,
            sig2g: 0.0,
        },
        lmaxph: lmaxph.to_vec(),
        decomposition_channels,
        save_gg_slice: false,
        do_fms: 0,
    };
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
    Ok(())
}

fn sample_global_input() -> GlobalInput {
    GlobalInput {
        cfaverage: CfAverage {
            nabs: 1,
            iphabs: 0,
            rclabs: 0.0,
        },
        control: GlobalControl {
            ipol: 0,
            ispin: 0,
            le2: 0,
            elpty: 0.0,
            angks: 0.0,
            l2lp: 0,
            do_nrixs: 0,
            ldecmx: 0,
            lj: 0,
        },
        evec: [0.0, 0.0, 1.0],
        xivec: [1.0, 0.0, 0.0],
        spvec: [0.0, 0.0, 1.0],
        polarization_tensor: [
            [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0],
        ],
        norms: GlobalNorms {
            evnorm: 1.0,
            xivnorm: 1.0,
            spvnorm: 1.0,
        },
        q_control: GlobalQControl {
            nq: 0,
            imdff: 0,
            qaverage: false,
            mixdff: false,
        },
        q_vectors: Vec::new(),
        mdff: None,
    }
}

fn sample_phase_bin() -> PhaseBinData {
    let energy_count = 2;
    let spin_count = 1;
    let transition_count = 8;
    let energy_grid = Array1::from_vec(vec![Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)]);
    let reference_energy = Array2::from_shape_fn((energy_count, spin_count), |(energy, spin)| {
        Complex64::new(0.01 * (energy + 1) as f64, -0.02 * spin as f64)
    });
    let phase_shifts =
        Array3::from_shape_fn((energy_count, 3, spin_count), |(energy, angular, spin)| {
            Complex64::new(
                0.1 * (energy + 1) as f64 + 0.01 * angular as f64,
                -0.005 * spin as f64,
            )
        });
    let mut transition_moments =
        Array4::<Complex64>::zeros((energy_count, 1, transition_count, spin_count).f());
    for energy in 0..energy_count {
        for transition in 0..transition_count {
            transition_moments[(energy, 0, transition, 0)] = Complex64::new(
                0.25 + 0.1 * energy as f64 + 0.03 * transition as f64,
                -0.02 * transition as f64,
            );
        }
    }

    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: energy_count,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: transition_count,
        transition_count,
        q_count: 1,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: 0.0,
            edge_energy: 8_979.0,
        },
        energy_grid,
        reference_energy,
        potentials: vec![PhaseBinPotential {
            lmax: 1,
            atomic_number: 29,
            label: "Cu".to_string(),
            phase_shifts,
        }],
        transition_moments,
        raw_pads: None,
    }
}

fn sample_mkgtr_gg() -> GgDatData {
    GgDatData {
        sections: (0..2)
            .map(|energy| GgDatSection {
                section_number: energy + 1,
                values: Array2::from_shape_fn((4, 4), |(row, column)| {
                    let base =
                        0.15 + 0.2 * energy as f64 + 0.03 * row as f64 + 0.01 * column as f64;
                    Complex64::new(base, -0.5 * base)
                }),
                raw_prefix_lines: None,
            })
            .collect(),
    }
}

fn expected_mkgtr_trace(
    global: &GlobalInput,
    phase: &PhaseBinData,
    gg: &GgDatData,
    lmax: usize,
) -> Result<Array2<Complex64>> {
    let core_hole = core_hole_quantum_numbers(phase.ihole)?;
    let transition_matrix = transition_b_matrix(TransitionBMatrixInput {
        lmax,
        initial_kappa: core_hole.kappa,
        polarization: global.control.ipol,
        polarization_tensor: super::polarization_tensor(global),
        multipole: global.control.le2,
        trace_orbital: false,
        spin: global.control.ispin,
        spin_channels: phase.spin_count,
        spin_vector_angle: global.control.angks,
    })?;
    let green_functions = super::green_functions_from_gg(gg, phase.energy_count)?;
    let transition_moments = phase.transition_moments.index_axis(Axis(1), 0);
    Ok(mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels: 1,
        green_functions: green_functions.view(),
        transition_matrices: &[transition_matrix],
        transition_moments,
    })?
    .traces)
}

fn assert_complex_table_close(
    actual: ndarray::ArrayView2<'_, Complex64>,
    expected: ndarray::ArrayView2<'_, Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= tolerance,
            "complex table mismatch at {index}: actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_complex_vec_close(
    actual: ndarray::ArrayView1<'_, Complex64>,
    expected: ndarray::ArrayView1<'_, Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= tolerance,
            "complex vector mismatch at {index}: actual={actual:?} expected={expected:?}"
        );
    }
}

fn sample_fms_bin() -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 2,
        main_energy_count: 1,
        auxiliary_energy_count: 0,
        highest_potential_index: 1,
        pad_width: 8,
        declared_spectrum_count: Some(2),
        spectra: Array2::from_shape_fn((2, 2), |(spectrum, energy)| {
            Complex64::new(
                0.25 * (energy + 1) as f64 + spectrum as f64,
                -0.05 * (energy + 1) as f64 - spectrum as f64,
            )
        }),
    }
}

fn sample_fmsl_bin() -> FmslBinData {
    FmslBinData {
        pad_width: 8,
        max_decomposition_channel: 2,
        traces: Array3::from_shape_fn((2, 3, 3), |(energy, lg2, lg1)| {
            Complex64::new(
                energy as f64 + 0.1 * lg2 as f64 + 0.01 * lg1 as f64,
                -(energy as f64) - 0.2 * lg2 as f64 - 0.02 * lg1 as f64,
            )
        }),
    }
}

fn sample_gg_dat() -> GgDatData {
    GgDatData {
        sections: vec![
            GgDatSection {
                section_number: 1,
                values: Array2::from_shape_fn((2, 2), |(row, column)| {
                    let value = 1.0 + row as f64 + 2.0 * column as f64;
                    Complex64::new(value, -0.5 * value)
                }),
                raw_prefix_lines: None,
            },
            GgDatSection {
                section_number: 2,
                values: Array2::from_shape_fn((1, 2), |(_, column)| {
                    let value = 5.0 + column as f64;
                    Complex64::new(value, -value - 0.5)
                }),
                raw_prefix_lines: None,
            },
        ],
    }
}

fn sample_gtr_dat() -> GtrDatData {
    GtrDatData {
        energy: Array1::from_vec(vec![
            Complex64::new(-0.138_801, 0.031_773),
            Complex64::new(-0.137_401, 0.031_773),
            Complex64::new(55.866_911, 0.031_773),
        ]),
        trace: Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.624_106, 1.081_113),
        ]),
    }
}

fn sample_gtr_bin() -> GtrBinData {
    GtrBinData {
        point_count_declared: 2,
        horizontal_count: 1,
        danes_extension_count: 0,
        highest_potential_index: 1,
        fms_mode: 2,
        values: Array3::from_shape_fn((2, 2, 2), |(energy, potential, angular)| {
            let value = energy as f64 + 0.1 * potential as f64 + 0.01 * angular as f64;
            Complex64::new(value, -value)
        }),
    }
}

fn sample_gtrl_dat() -> Result<GtrlDatData> {
    Ok(parse_gtrl_dat(
        r#"    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01
    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01
"#,
    )?)
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "FMS calculation of full Green's function ...".to_string(),
            "Done with module: FMS.".to_string(),
            "MKGTR: Tracing over Green's function ...".to_string(),
            "Done with module: MKGTR.".to_string(),
        ],
        line_terminators: vec![
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
        ],
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

fn copy_gtr_bin_references(source_dir: &Path, target_dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(source_dir)
        .with_context(|| format!("failed to read {}", source_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", source_dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if super::is_gtr_bin_name(name) {
            std::fs::copy(entry.path(), target_dir.join(name))?;
        }
    }
    Ok(())
}

fn reference_fms_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}
