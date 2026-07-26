use super::*;

pub(super) fn write_type2_coupling_sidecar(
    work_dir: &Path,
    calculation: &DmdwCalculation,
) -> Result<()> {
    let options = calculation
        .self_energy_options
        .as_ref()
        .context("DMDW run type 2 requires self-energy options")?;
    let pds_path = work_dir.join(&options.pds_file);
    let a2f_path = work_dir.join(&options.a2f_file);
    let pds = read_dmdw_coupling_table(&pds_path)
        .with_context(|| format!("failed to read {}", pds_path.display()))?;
    let a2f = read_dmdw_coupling_table(&a2f_path)
        .with_context(|| format!("failed to read {}", a2f_path.display()))?;
    let coupling = dmdw_phonon_coupling_from_tables(&pds, &a2f)
        .context("failed to compute DMDW type 2 phonon coupling")?;
    let sidecar = dmdw_a2_dat_from_coupling(&coupling)?;
    let path = work_dir.join("dmdw_A2.dat");
    write_dmdw_a2_dat(&path, &sidecar)
        .with_context(|| format!("failed to write {}", path.display()))?;
    write_type2_a2f_info_sidecar(work_dir, calculation, &coupling)?;
    write_type2_self_energy_sidecars(work_dir, calculation)
}

fn write_type2_a2f_info_sidecar(
    work_dir: &Path,
    calculation: &DmdwCalculation,
    coupling: &refeff_core::DmdwPhononCoupling,
) -> Result<()> {
    let dym_path = work_dir.join(&calculation.dym_file);
    if !dym_path.is_file() {
        return Ok(());
    }

    let options = calculation
        .self_energy_options
        .as_ref()
        .context("DMDW run type 2 requires self-energy options")?;
    let pole_count = usize::try_from(calculation.order).context("invalid DMDW Lanczos order")?;
    let dym =
        read_dym(&dym_path).with_context(|| format!("failed to read {}", dym_path.display()))?;
    let metadata = dym
        .type2_metadata
        .as_ref()
        .context("DMDW run type 2 .dym file requires type 2 unique-atom metadata")?;
    let unique_atom_groups = metadata
        .unique_atoms
        .iter()
        .map(|atom| DmdwType2AtomGroup {
            center_atom_indices: atom.center_atom_indices.iter().copied().collect(),
        })
        .collect::<Vec<_>>();
    let diagnostic = dmdw_type2_pole_weighted_a2f(
        dym.force_constants.view(),
        dym.atomic_masses.view(),
        &unique_atom_groups,
        options.displacement_option,
        pole_count,
        coupling,
    )
    .context("failed to compute DMDW type 2 pole-weighted a2f diagnostic")?;
    let data = dmdw_a2f_info_from_pole_weighted(
        calculation.calculation_type,
        options.displacement_option,
        pole_count,
        &diagnostic,
    )
    .context("failed to build DMDW type 2 a2f diagnostic")?;
    let a2f_info_path = work_dir.join("dmdw_a2f.info");
    write_dmdw_a2f_info(&a2f_info_path, &data)
        .with_context(|| format!("failed to write {}", a2f_info_path.display()))
}

fn write_type2_self_energy_sidecars(work_dir: &Path, calculation: &DmdwCalculation) -> Result<()> {
    let a2f_info_path = work_dir.join("dmdw_a2f.info");
    if !a2f_info_path.is_file() {
        return Ok(());
    }

    let options = calculation
        .self_energy_options
        .as_ref()
        .context("DMDW run type 2 requires self-energy options")?;
    let temperatures = dmdw_temperatures(calculation)?;
    let temperature = temperatures[0];
    let a2f_info = read_dmdw_a2f_info(&a2f_info_path)
        .with_context(|| format!("failed to read {}", a2f_info_path.display()))?;
    let diagnostic = dmdw_pole_weighted_from_info(&a2f_info);
    let w0 = diagnostic.characteristic_energy_ev;
    if !w0.is_finite() || w0 <= 0.0 {
        bail!("DMDW type 2 characteristic energy w0 must be positive and finite");
    }

    let electron_energy_w0 =
        dmdw_type2_electron_energy_w0(options.energy_option, options.electron_energy, w0)?;
    let window = dmdw_type2_energy_window(electron_energy_w0, w0)?;
    let grid =
        dmdw_self_energy_grid_from_a2f_poles(temperature, window.energy_ev.view(), &diagnostic)
            .context("failed to compute DMDW type 2 self-energy grid")?;
    let tables = dmdw_type2_self_energy_tables(window.w.view(), &grid)?;
    let spectral = dmdw_type2_spectral_info(DmdwType2SpectralInput {
        temperature,
        electron_energy_w0,
        characteristic_energy_ev: w0,
        selected_index: window.selected_index,
        w: window.w.view(),
        re_se_ev: tables.re_se_ev.view(),
        im_se_ev: tables.im_se_ev.view(),
        diagnostic: &diagnostic,
    })?;
    let akw = dmdw_type2_akw_dat(DmdwType2AkwInput {
        temperature,
        electron_energy_w0,
        characteristic_energy_ev: w0,
        w: window.w.view(),
        diagnostic: &diagnostic,
    })?;

    let egrid_path = work_dir.join("dmdw_Egrid.info");
    write_dmdw_egrid_info(&egrid_path, &window.info)
        .with_context(|| format!("failed to write {}", egrid_path.display()))?;
    let re_path = work_dir.join("dmdw_reSE_a2F.dat");
    write_dmdw_self_energy_dat(&re_path, &tables.real_table)
        .with_context(|| format!("failed to write {}", re_path.display()))?;
    let im_path = work_dir.join("dmdw_imSE_a2F.dat");
    write_dmdw_self_energy_dat(&im_path, &tables.imaginary_table)
        .with_context(|| format!("failed to write {}", im_path.display()))?;
    let spectral_path = work_dir.join("dmdw_spectral.info");
    write_dmdw_spectral_info(&spectral_path, &spectral)
        .with_context(|| format!("failed to write {}", spectral_path.display()))?;
    let akw_path = work_dir.join("dmdw_Akw.dat");
    write_dmdw_akw_dat(&akw_path, &akw)
        .with_context(|| format!("failed to write {}", akw_path.display()))
}

#[derive(Debug)]
struct DmdwType2EnergyWindow {
    w: Array1<f64>,
    energy_ev: Array1<f64>,
    selected_index: usize,
    info: DmdwEnergyGridInfo,
}

#[derive(Debug)]
struct DmdwType2SelfEnergyTables {
    real_table: DmdwSelfEnergyDatData,
    imaginary_table: DmdwSelfEnergyDatData,
    re_se_ev: Array1<f64>,
    im_se_ev: Array1<f64>,
}

#[derive(Debug, Clone, Copy)]
struct DmdwType2SpectralInput<'a> {
    temperature: f64,
    electron_energy_w0: f64,
    characteristic_energy_ev: f64,
    selected_index: usize,
    w: ndarray::ArrayView1<'a, f64>,
    re_se_ev: ndarray::ArrayView1<'a, f64>,
    im_se_ev: ndarray::ArrayView1<'a, f64>,
    diagnostic: &'a DmdwPoleWeightedA2f,
}

#[derive(Debug, Clone, Copy)]
struct DmdwType2AkwInput<'a> {
    temperature: f64,
    electron_energy_w0: f64,
    characteristic_energy_ev: f64,
    w: ndarray::ArrayView1<'a, f64>,
    diagnostic: &'a DmdwPoleWeightedA2f,
}

fn dmdw_pole_weighted_from_info(data: &DmdwA2fInfoData) -> DmdwPoleWeightedA2f {
    DmdwPoleWeightedA2f {
        lanczos_frequency_thz: data.lanczos_frequency_thz.clone(),
        lanczos_weight: data.lanczos_weight.clone(),
        normalization: data.normalization,
        pole_energy_ev: data.pole_energy_ev.clone(),
        pole_weight: data.pole_weight.clone(),
        mass_enhancement: data.mass_enhancement,
        characteristic_energy_ev: data.characteristic_energy_ev,
    }
}

fn dmdw_type2_electron_energy_w0(
    energy_option: i32,
    electron_energy: f64,
    characteristic_energy_ev: f64,
) -> Result<f64> {
    if !electron_energy.is_finite() {
        bail!("DMDW type 2 electron energy must be finite");
    }
    if energy_option == 1 {
        Ok(electron_energy * 1.0e-3 / characteristic_energy_ev)
    } else {
        Ok(electron_energy)
    }
}

fn dmdw_type2_energy_window(
    electron_energy_w0: f64,
    characteristic_energy_ev: f64,
) -> Result<DmdwType2EnergyWindow> {
    let low = electron_energy_w0 - DMDW_TYPE2_SELF_ENERGY_WINDOW_W0;
    let high = electron_energy_w0 + DMDW_TYPE2_SELF_ENERGY_WINDOW_W0;
    let step = (high - low) / (DMDW_TYPE2_SELF_ENERGY_POINTS as f64 - 1.0);
    if !step.is_finite() || step <= 0.0 {
        bail!("DMDW type 2 self-energy grid step must be positive and finite");
    }

    let low_index = (low / step).round() as i32;
    let high_index = (high / step).round() as i32;
    let w = (low_index..=high_index)
        .map(|index| f64::from(index) * step)
        .collect::<Array1<_>>();
    let energy_ev = w.mapv(|value| value * characteristic_energy_ev);
    let selected_index = dmdw_type2_selected_energy_index(w.view(), electron_energy_w0)?;
    let info = DmdwEnergyGridInfo {
        low_energy_mev: low * characteristic_energy_ev * 1.0e3,
        high_energy_mev: high * characteristic_energy_ev * 1.0e3,
        step_mev: step * characteristic_energy_ev * 1.0e3,
        characteristic_energy_mev: characteristic_energy_ev * 1.0e3,
        electron_energy_mev: electron_energy_w0 * characteristic_energy_ev * 1.0e3,
        selected_energy_mev: w[selected_index] * characteristic_energy_ev * 1.0e3,
    };
    Ok(DmdwType2EnergyWindow {
        w,
        energy_ev,
        selected_index,
        info,
    })
}

fn dmdw_type2_selected_energy_index(
    w: ndarray::ArrayView1<'_, f64>,
    electron_energy_w0: f64,
) -> Result<usize> {
    for index in 0..w.len().saturating_sub(1) {
        if w[index] <= electron_energy_w0 && electron_energy_w0 <= w[index + 1] {
            return Ok(
                if electron_energy_w0 - w[index] < w[index + 1] - electron_energy_w0 {
                    index
                } else {
                    index + 1
                },
            );
        }
    }
    bail!("DMDW type 2 electron energy was not covered by the self-energy grid")
}

fn dmdw_type2_self_energy_tables(
    w: ndarray::ArrayView1<'_, f64>,
    grid: &refeff_core::DmdwSelfEnergyGrid,
) -> Result<DmdwType2SelfEnergyTables> {
    if w.len() != grid.point_count() {
        bail!("DMDW type 2 self-energy grid shape mismatch");
    }
    let re_se = grid.self_energy.mapv(|value| value.re);
    let im_se = grid
        .energy_ev
        .iter()
        .zip(grid.self_energy.iter())
        .map(|(&energy, value)| -dmdw_fortran_sign(energy) * value.im.abs())
        .collect::<Array1<_>>();
    let real_table = DmdwSelfEnergyDatData {
        header_lines: vec!["#  Real part of the Self-energy".to_string()],
        energy_ev: grid.energy_ev.clone(),
        value_ev: re_se.clone(),
    };
    let imaginary_table = DmdwSelfEnergyDatData {
        header_lines: vec!["#  Imaginary part of the Self-energy".to_string()],
        energy_ev: grid.energy_ev.clone(),
        value_ev: im_se.clone(),
    };
    Ok(DmdwType2SelfEnergyTables {
        real_table,
        imaginary_table,
        re_se_ev: re_se,
        im_se_ev: im_se,
    })
}

fn dmdw_type2_spectral_info(input: DmdwType2SpectralInput<'_>) -> Result<DmdwSpectralInfoData> {
    let DmdwType2SpectralInput {
        temperature,
        electron_energy_w0,
        characteristic_energy_ev,
        selected_index,
        w,
        re_se_ev,
        im_se_ev,
        diagnostic,
    } = input;
    if selected_index == 0 || selected_index + 1 >= w.len() {
        bail!("DMDW type 2 selected electron energy must have neighboring grid points");
    }
    let zero_self_energy = dmdw_self_energy_from_a2f_poles(
        temperature,
        Complex::new(0.0, 0.0),
        diagnostic.pole_energy_ev.view(),
        diagnostic.pole_weight.view(),
    )
    .context("failed to compute DMDW type 2 zero-energy self energy")?;
    let electron_self_energy = dmdw_self_energy_from_a2f_poles(
        temperature,
        Complex::new(electron_energy_w0 * characteristic_energy_ev, 0.0),
        diagnostic.pole_energy_ev.view(),
        diagnostic.pole_weight.view(),
    )
    .context("failed to compute DMDW type 2 electron self energy")?;
    let gamma = (zero_self_energy.im.abs() / characteristic_energy_ev).max(0.005);
    let effective_electron_energy =
        electron_energy_w0 - electron_self_energy.re / characteristic_energy_ev;
    let left = Complex::new(
        re_se_ev[selected_index - 1] / characteristic_energy_ev,
        im_se_ev[selected_index - 1] / characteristic_energy_ev,
    );
    let right = Complex::new(
        re_se_ev[selected_index + 1] / characteristic_energy_ev,
        im_se_ev[selected_index + 1] / characteristic_energy_ev,
    );
    let total_cumulant_derivative =
        -(right - left) / (w[selected_index + 1] - w[selected_index - 1]);
    let quasiparticle_weight = (-total_cumulant_derivative).exp();
    Ok(DmdwSpectralInfoData {
        gamma,
        effective_electron_energy,
        total_cumulant_derivative,
        quasiparticle_weight,
    })
}

fn dmdw_type2_akw_dat(input: DmdwType2AkwInput<'_>) -> Result<DmdwAkwDatData> {
    let DmdwType2AkwInput {
        temperature,
        electron_energy_w0,
        characteristic_energy_ev,
        w,
        diagnostic,
    } = input;
    let spectral = dmdw_spectral_function_from_a2f_poles(
        temperature,
        w,
        electron_energy_w0,
        characteristic_energy_ev,
        diagnostic,
        DMDW_TYPE2_SPECTRAL_TIME_HALF_WIDTH,
        DMDW_TYPE2_SPECTRAL_TIME_POINTS,
    )
    .context("failed to compute DMDW type 2 spectral function")?;
    let output_scale = characteristic_energy_ev * 1.0e3;
    if !output_scale.is_finite() || output_scale <= 0.0 {
        bail!("DMDW type 2 spectral output scale must be positive and finite");
    }

    let energy_mev = spectral
        .energy_w0
        .iter()
        .map(|&energy| energy * output_scale)
        .collect::<Array1<_>>();
    let real = spectral
        .spectral_function
        .iter()
        .map(|value| value.re / output_scale)
        .collect::<Array1<_>>();
    let imaginary = spectral
        .spectral_function
        .iter()
        .map(|value| value.im / output_scale)
        .collect::<Array1<_>>();
    let magnitude = real
        .iter()
        .zip(imaginary.iter())
        .map(|(&x, &y)| (x * x + y * y).sqrt())
        .collect::<Array1<_>>();
    let phase = real
        .iter()
        .zip(imaginary.iter())
        .map(|(&x, &y)| y.atan2(x))
        .collect::<Array1<_>>();

    Ok(DmdwAkwDatData {
        normalization: Some(spectral.normalization),
        energy_mev,
        magnitude,
        phase,
        real,
        imaginary,
    })
}

fn dmdw_fortran_sign(value: f64) -> f64 {
    if value < 0.0 { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::dmdw_type2_electron_energy_w0;

    #[test]
    fn type2_electron_energy_option_one_converts_mev_to_w0() {
        let energy_w0 = dmdw_type2_electron_energy_w0(1, 50.0, 0.020).unwrap();

        assert!((energy_w0 - 2.5).abs() <= f64::EPSILON);
    }

    #[test]
    fn type2_electron_energy_options_other_than_one_leave_energy_unchanged() {
        for energy_option in [-7, 0, 2, 42] {
            let energy_w0 = dmdw_type2_electron_energy_w0(energy_option, -3.25, 0.020).unwrap();

            assert_eq!(energy_w0, -3.25, "energy option {energy_option}");
        }
    }
}
