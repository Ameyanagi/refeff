use super::run_in_dir;
use anyhow::{Context, Result};
use ndarray::{Array4, arr1, arr2};
use refeff_io::{
    DmdwA2fInfoData, DmdwOutData, DmdwOutHeader, DmdwOutSection, DmdwOutSubject,
    DmdwOutTemperature, DymCoordinates, DymData, DymType2Metadata, DymUniqueAtom, read_dmdw_a2_dat,
    read_dmdw_a2f_info, read_dmdw_akw_dat, read_dmdw_egrid_info, read_dmdw_out,
    read_dmdw_self_energy_dat, read_dmdw_spectral_info, write_dmdw_a2f_info, write_dmdw_out,
    write_dym,
};
use std::path::{Path, PathBuf};

#[test]
fn dmdw_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_disabled_dmdw_input(temp.path())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!temp.path().join("dmdw.out").exists());
    Ok(())
}

#[test]
fn dmdw_module_rejects_invalid_feff_run_type() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_unsupported_dmdw_input(temp.path())?;

    let error = run_in_dir(temp.path())
        .err()
        .context("unsupported DMDW should reject the invalid FEFF run type")?;

    assert!(
        error
            .to_string()
            .contains("DMDW run type 6 is not supported by FEFF DMDW")
    );
    Ok(())
}

#[test]
fn dmdw_module_roundtrips_cached_type2_self_energy_marker() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type2_dmdw_input(temp.path())?;
    write_dmdw_out(temp.path().join("dmdw.out"), &sample_type2_dmdw_out())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 0);
    assert!(output.mass_enhancement_header);
    assert!(output.sections.is_empty());
    Ok(())
}

#[test]
fn dmdw_module_generates_type2_coupling_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type2_dmdw_input(temp.path())?;
    write_type2_coupling_inputs(temp.path())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;
    let sidecar = read_dmdw_a2_dat(temp.path().join("dmdw_A2.dat"))?;

    assert_eq!(count, 0);
    assert!(output.mass_enhancement_header);
    assert!(output.sections.is_empty());
    assert_eq!(sidecar.point_count(), 3);
    assert_eq!(sidecar.energy_hartree, arr1(&[0.001, 0.002, 0.004]));
    assert_eq!(sidecar.matrix_element, arr1(&[0.05, 0.05, 0.05]));
    Ok(())
}

#[test]
fn dmdw_module_generates_type2_a2f_info_from_type2_dym() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type2_dmdw_input(temp.path())?;
    write_type2_coupling_inputs(temp.path())?;
    write_dym(temp.path().join("feff.dym"), &sample_type2_dym())?;

    let count = run_in_dir(temp.path())?;
    let a2f_info = read_dmdw_a2f_info(temp.path().join("dmdw_a2f.info"))?;
    let spectral = read_dmdw_spectral_info(temp.path().join("dmdw_spectral.info"))?;
    let akw = read_dmdw_akw_dat(temp.path().join("dmdw_Akw.dat"))?;

    assert_eq!(count, 0);
    assert_eq!(a2f_info.calculation_type, 2);
    assert_eq!(a2f_info.displacement_option, 0);
    assert_eq!(a2f_info.lanczos_order, 1);
    assert_eq!(a2f_info.pole_count(), 1);
    assert!(a2f_info.normalization.is_finite());
    assert!(a2f_info.normalization > 0.0);
    assert!(a2f_info.pole_energy_ev[0].is_finite());
    assert!(a2f_info.pole_energy_ev[0] > 0.0);
    assert!(a2f_info.pole_weight[0].is_finite());
    assert!(a2f_info.characteristic_energy_ev > 0.0);
    assert!(spectral.gamma.is_finite());
    assert_eq!(akw.point_count(), 10_001);
    Ok(())
}

#[test]
fn dmdw_module_generates_type2_self_energy_sidecars_from_a2f_info() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type2_dmdw_input(temp.path())?;
    write_type2_coupling_inputs(temp.path())?;
    write_type2_a2f_info(temp.path())?;

    let count = run_in_dir(temp.path())?;
    let egrid = read_dmdw_egrid_info(temp.path().join("dmdw_Egrid.info"))?;
    let real = read_dmdw_self_energy_dat(temp.path().join("dmdw_reSE_a2F.dat"))?;
    let imaginary = read_dmdw_self_energy_dat(temp.path().join("dmdw_imSE_a2F.dat"))?;
    let spectral = read_dmdw_spectral_info(temp.path().join("dmdw_spectral.info"))?;
    let akw = read_dmdw_akw_dat(temp.path().join("dmdw_Akw.dat"))?;

    assert_eq!(count, 0);
    assert_eq!(real.point_count(), 10_001);
    assert_eq!(imaginary.point_count(), 10_001);
    assert_eq!(akw.point_count(), 10_001);
    assert_close(egrid.low_energy_mev, -200.0);
    assert_close(egrid.high_energy_mev, 200.0);
    assert_close(egrid.step_mev, 0.04);
    assert_close(egrid.characteristic_energy_mev, 20.0);
    assert_close(egrid.electron_energy_mev, 0.0);
    assert_close(egrid.selected_energy_mev, 0.0);
    assert_close(real.energy_ev[0], -0.2);
    assert_close(real.energy_ev[10_000], 0.2);
    assert_eq!(real.energy_ev, imaginary.energy_ev);
    assert!(real.value_ev.iter().all(|value| value.is_finite()));
    assert!(imaginary.value_ev.iter().all(|value| value.is_finite()));
    assert!(spectral.gamma >= 0.005);
    assert!(spectral.effective_electron_energy.is_finite());
    assert!(spectral.total_cumulant_derivative.re.is_finite());
    assert!(spectral.total_cumulant_derivative.im.is_finite());
    assert!(spectral.quasiparticle_weight.re.is_finite());
    assert!(spectral.quasiparticle_weight.im.is_finite());
    assert_close(akw.energy_mev[0], -200.0);
    assert_close(akw.energy_mev[10_000], 200.0);
    assert!(
        akw.normalization
            .is_some_and(|value| value.is_finite() && value > 0.0)
    );
    assert!(akw.magnitude.iter().all(|value| value.is_finite()));
    assert!(akw.phase.iter().all(|value| value.is_finite()));
    assert!(akw.real.iter().all(|value| value.is_finite()));
    assert!(akw.imaginary.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn dmdw_module_generates_type0_path_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_dmdw_input(temp.path())?;
    write_dym(temp.path().join("feff.dym"), &sample_dym())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 1);
    assert_eq!(output.section_count(), 1);
    let section = &output.sections[0];
    assert_eq!(section.subject, DmdwOutSubject::PathIndices(vec![1, 2]));
    assert_eq!(section.pdos_poles.len(), 1);
    assert!(section.einstein.is_some());
    assert_eq!(section.moments.len(), 5);
    assert!(section.reduced_mass_amu.is_some_and(|value| value > 0.0));
    assert!(
        section
            .path_length_angstrom
            .is_some_and(|value| value > 0.0)
    );
    assert!(
        section
            .sigma2_1e_minus_3_angstrom2
            .is_some_and(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn dmdw_module_generates_type0_multi_temperature_path_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_multi_temperature_dmdw_input(temp.path())?;
    write_dym(temp.path().join("feff.dym"), &sample_dym())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 1);
    assert!(matches!(
        output.header.as_ref().map(|header| &header.temperature),
        Some(DmdwOutTemperature::ListedBelow)
    ));
    let section = &output.sections[0];
    assert_eq!(section.subject, DmdwOutSubject::PathIndices(vec![1, 2]));
    assert!(section.sigma2_1e_minus_3_angstrom2.is_none());
    assert_eq!(section.sigma2_by_temperature.len(), 3);
    let temperatures = section
        .sigma2_by_temperature
        .iter()
        .map(|row| row.temperature_kelvin)
        .collect::<Vec<_>>();
    assert_eq!(temperatures, vec![100.0, 300.0, 500.0]);
    assert!(
        section
            .sigma2_by_temperature
            .iter()
            .all(|row| row.value.is_finite())
    );
    Ok(())
}

#[test]
fn dmdw_module_generates_type3_u2_atom_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type3_dmdw_input(temp.path())?;
    write_dym(temp.path().join("feff.dym"), &sample_dym())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 3);
    assert_eq!(output.section_count(), 3);
    let mut directions = Vec::new();
    for section in &output.sections {
        let DmdwOutSubject::AtomIndex { indices, direction } = &section.subject else {
            anyhow::bail!("unexpected DMDW subject {:?}", section.subject);
        };
        assert_eq!(indices, &vec![1]);
        let Some(direction) = direction else {
            anyhow::bail!("missing DMDW perturbation direction");
        };
        directions.push(direction.clone());
    }
    assert_eq!(directions, vec!["x", "y", "z"]);
    assert!(output.sections.iter().all(|section| {
        section
            .u2_1e_minus_3_angstrom2
            .is_some_and(|value| value.is_finite())
    }));
    Ok(())
}

#[test]
fn dmdw_module_generates_type1_vfe_atom_and_total_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type1_dmdw_input(temp.path())?;
    write_dym(temp.path().join("feff.dym"), &sample_dym())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 10);
    assert_eq!(output.section_count(), 10);
    for section in output.sections.iter().take(9) {
        let DmdwOutSubject::AtomIndex { direction, .. } = &section.subject else {
            anyhow::bail!("unexpected DMDW subject {:?}", section.subject);
        };
        if direction.is_none() {
            anyhow::bail!("missing DMDW VFE perturbation direction");
        }
        assert!(
            section
                .vibrational_free_energy_ev
                .is_some_and(|value| value.is_finite())
        );
    }

    let Some(total) = output.sections.last() else {
        anyhow::bail!("missing DMDW total VFE section");
    };
    assert_eq!(total.subject, DmdwOutSubject::TotalVfe);
    assert!(
        total
            .vibrational_free_energy_ev
            .is_some_and(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn dmdw_module_generates_type4_ir_diagnostics() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type4_dmdw_input(temp.path())?;
    write_dym(temp.path().join("feff-ir.dym"), &sample_ir_dym())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 1);
    assert_eq!(output.section_count(), 1);
    let section = &output.sections[0];
    assert_eq!(section.subject, DmdwOutSubject::PathIndices(vec![1, 2]));
    assert_eq!(section.pdos_poles.len(), 1);
    assert!(section.einstein.is_some());
    assert_eq!(section.moments.len(), 5);
    assert!(section.sigma2_1e_minus_3_angstrom2.is_none());
    assert!(section.u2_1e_minus_3_angstrom2.is_none());
    assert!(section.vibrational_free_energy_ev.is_none());
    Ok(())
}

#[test]
fn dmdw_module_generates_type5_projected_dos_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_type5_dmdw_input(temp.path())?;
    write_dym(temp.path().join("feff.dym"), &sample_dym())?;

    let count = run_in_dir(temp.path())?;
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;

    assert_eq!(count, 10);
    assert_eq!(output.section_count(), 10);
    for section in output.sections.iter().take(9) {
        let DmdwOutSubject::AtomIndex { direction, .. } = &section.subject else {
            anyhow::bail!("unexpected DMDW subject {:?}", section.subject);
        };
        if direction.is_none() {
            anyhow::bail!("missing DMDW projected-DOS perturbation direction");
        }
        assert!(section.projected_dos_component_computed);
        assert_eq!(section.pdos_poles.len(), 1);
    }

    let Some(total) = output.sections.last() else {
        anyhow::bail!("missing DMDW total PDOS section");
    };
    assert_eq!(total.subject, DmdwOutSubject::TotalPdos);
    assert!(total.projected_dos_component_computed);
    assert!(!total.pdos_poles.is_empty());
    let total_weight = total.pdos_poles.iter().map(|pole| pole.weight).sum::<f64>();
    assert!((total_weight - 1.0).abs() < 1.0e-6);

    let sidecar = temp.path().join("dmdw_pdos.poles.tot.dat");
    assert!(sidecar.is_file());
    let sidecar_text = std::fs::read_to_string(sidecar)?;
    assert!(sidecar_text.lines().any(|line| line.starts_with('#')));
    assert!(!temp.path().join("dmdw_pdos.poles.001.x.dat").exists());
    Ok(())
}

#[test]
fn dmdw_module_roundtrips_cached_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_dmdw_input(temp.path())?;
    let expected = sample_dmdw_out();
    write_dmdw_out(temp.path().join("dmdw.out"), &expected)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(read_dmdw_out(temp.path().join("dmdw.out"))?, expected);
    Ok(())
}

#[test]
fn dmdw_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_dmdw_dir()? else {
        eprintln!("skipping DMDW reference test; generated DEBYE/DM/EXAFS/Cu reference not found");
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    std::fs::copy(reference_dir.join("dmdw.inp"), temp.path().join("dmdw.inp"))?;
    std::fs::copy(reference_dir.join("dmdw.out"), temp.path().join("dmdw.out"))?;
    let expected = read_dmdw_out(temp.path().join("dmdw.out"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, expected.section_count());
    assert_eq!(read_dmdw_out(temp.path().join("dmdw.out"))?, expected);
    Ok(())
}

fn write_disabled_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(work_dir.join("dmdw.inp"), "-999\n")?;
    Ok(())
}

fn write_enabled_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   0\n",
            "feff.dym\n",
            "   1\n",
            "   2   1   2          10.00\n",
        ),
    )?;
    Ok(())
}

fn write_multi_temperature_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   3    500.000    100.000\n",
            "   0\n",
            "feff.dym\n",
            "   1\n",
            "   2   1   2          10.00\n",
        ),
    )?;
    Ok(())
}

fn write_unsupported_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   6\n",
            "feff.dym\n",
            "   1\n",
            "   2   1   2          10.00\n",
        ),
    )?;
    Ok(())
}

fn write_type2_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   2\n",
            "   0\n",
            "   0      0.000\n",
            "feff.dym\n",
            "phonon.pds\n",
            "coupling.a2f\n",
        ),
    )?;
    Ok(())
}

fn write_type2_coupling_inputs(work_dir: &Path) -> Result<()> {
    std::fs::write(work_dir.join("phonon.pds"), SAMPLE_TYPE2_PDS)?;
    std::fs::write(work_dir.join("coupling.a2f"), SAMPLE_TYPE2_A2F)?;
    Ok(())
}

fn write_type2_a2f_info(work_dir: &Path) -> Result<()> {
    let data = DmdwA2fInfoData {
        calculation_type: 2,
        displacement_option: 0,
        lanczos_order: 2,
        lanczos_frequency_thz: arr1(&[2.0, 5.0]),
        lanczos_weight: arr1(&[0.4, 0.6]),
        normalization: 1.0,
        pole_energy_ev: arr1(&[0.012, 0.024]),
        pole_weight: arr1(&[0.35, 0.65]),
        mass_enhancement: 75.833_333_333_333_33,
        characteristic_energy_ev: 0.020,
    };
    write_dmdw_a2f_info(work_dir.join("dmdw_a2f.info"), &data)?;
    Ok(())
}

fn write_type1_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   1\n",
            "feff.dym\n",
            "   0\n",
        ),
    )?;
    Ok(())
}

fn write_type3_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   3\n",
            "feff.dym\n",
            "   1\n",
            "   1   1              10.00\n",
        ),
    )?;
    Ok(())
}

fn write_type4_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   4\n",
            "feff-ir.dym\n",
            "   1\n",
            "   2   1   2          10.00\n",
        ),
    )?;
    Ok(())
}

fn write_type5_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   5\n",
            "feff.dym\n",
            "   0\n",
        ),
    )?;
    Ok(())
}

fn sample_dym() -> DymData {
    let atomic_numbers = arr1(&[29, 29, 29]);
    let atomic_masses = arr1(&[63.546, 63.546, 63.546]);
    let coordinates =
        DymCoordinates::Cartesian(arr2(&[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.7, 0.0]]));
    let mut force_constants = Array4::zeros((3, 3, 3, 3));
    for atom in 0..3 {
        for component in 0..3 {
            force_constants[(atom, atom, component, component)] =
                0.02 + 0.003 * atom as f64 + 0.001 * component as f64;
        }
    }
    DymData {
        dym_type: 1,
        atomic_numbers,
        atomic_masses,
        coordinates,
        force_constants,
        type2_metadata: None,
        dipole_derivatives: None,
    }
}

fn sample_ir_dym() -> DymData {
    let mut data = sample_dym();
    data.dym_type = 3;
    let mut dipoles = ndarray::Array3::zeros((3, 3, 3));
    for atom in 0..3 {
        for displacement_component in 0..3 {
            for dipole_component in 0..3 {
                dipoles[(atom, displacement_component, dipole_component)] = 0.1
                    + atom as f64
                    + 0.2 * displacement_component as f64
                    + 0.3 * dipole_component as f64;
            }
        }
    }
    data.dipole_derivatives = Some(dipoles);
    data
}

fn sample_type2_dym() -> DymData {
    let mut data = sample_dym();
    data.dym_type = 2;
    for component in 0..3 {
        data.force_constants[(0, 1, component, component)] = -0.004;
        data.force_constants[(1, 0, component, component)] = -0.004;
        data.force_constants[(1, 2, component, component)] = -0.003;
        data.force_constants[(2, 1, component, component)] = -0.003;
    }
    data.type2_metadata = Some(DymType2Metadata {
        cell_atom_count: 1,
        unique_atoms: vec![DymUniqueAtom {
            atom_type: 29,
            center_atom_indices: arr1(&[0_usize]),
            weights: arr1(&[1.0]),
            coordinates: arr2(&[[0.0, 0.0, 0.0]]),
        }],
    });
    data
}

fn sample_dmdw_out() -> DmdwOutData {
    let mut section = DmdwOutSection::new(DmdwOutSubject::PathIndices(vec![1, 2]));
    section.reduced_mass_amu = Some(31.773);
    section.path_length_angstrom = Some(2.5323);
    section.sigma2_1e_minus_3_angstrom2 = Some(11.8576);

    DmdwOutData {
        header: Some(DmdwOutHeader {
            lanczos_recursion_order: 2,
            temperature: DmdwOutTemperature::Single(450.0),
            dynamical_matrix_file: "feff.dym".to_string(),
        }),
        mass_enhancement_header: false,
        sections: vec![section],
    }
}

fn sample_type2_dmdw_out() -> DmdwOutData {
    DmdwOutData {
        header: Some(DmdwOutHeader {
            lanczos_recursion_order: 6,
            temperature: DmdwOutTemperature::Single(450.0),
            dynamical_matrix_file: "feff.dym".to_string(),
        }),
        mass_enhancement_header: true,
        sections: Vec::new(),
    }
}

const SAMPLE_TYPE2_PDS: &str = concat!(
    "# header 1\n",
    "# header 2\n",
    "# header 3\n",
    "# header 4\n",
    "# header 5\n",
    "# header 6\n",
    "# header 7\n",
    "# header 8\n",
    "# header 9\n",
    "# header 10\n",
    " 1.0D-03 1.0D+01\n",
    " 2.0D-03 2.0D+01\n",
    " 4.0D-03 3.0D+01\n",
);

const SAMPLE_TYPE2_A2F: &str = concat!(
    "# header 1\n",
    "# header 2\n",
    "# header 3\n",
    "# header 4\n",
    "# header 5\n",
    "# header 6\n",
    "# header 7\n",
    "# header 8\n",
    "# header 9\n",
    "# header 10\n",
    " 1.0D-03 5.0D-01\n",
    " 2.0D-03 1.0D+00\n",
    " 4.0D-03 1.5D+00\n",
);

fn assert_close(actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

fn reference_dmdw_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/DEBYE/DM/EXAFS/Cu");
    let required = ["dmdw.inp", "dmdw.out"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}
