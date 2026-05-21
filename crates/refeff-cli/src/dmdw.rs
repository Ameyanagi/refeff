use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, ArrayView2};
use refeff_core::{
    Complex, DMDW_ANGSTROM_TO_BOHR, DmdwLanczosCoefficients, DmdwLanczosPoleSpectrum,
    DmdwPathDescriptor, DmdwPoleWeightedA2f, DmdwType2AtomGroup,
    dmdw_debye_waller_factors_from_poles, dmdw_expand_path_descriptor,
    dmdw_expand_path_descriptors, dmdw_ir_dipole_seed_vector, dmdw_lanczos_coefficients,
    dmdw_lanczos_pole_spectrum, dmdw_mass_weighted_dynamical_matrix,
    dmdw_moment_summaries_from_poles, dmdw_path_motion, dmdw_project_seed_vector,
    dmdw_rigid_body_projection_modes, dmdw_self_energy_from_a2f_poles,
    dmdw_self_energy_grid_from_a2f_poles, dmdw_single_pole_einstein_summary,
    dmdw_spectral_function_from_a2f_poles, dmdw_type2_pole_weighted_a2f,
    dmdw_vibrational_free_energy_from_poles,
};
use refeff_io::{
    DmdwA2fInfoData, DmdwAkwDatData, DmdwCalculation, DmdwEnergyGridInfo, DmdwInput, DmdwOutData,
    DmdwOutEinstein, DmdwOutHeader, DmdwOutMoment, DmdwOutPole, DmdwOutSection, DmdwOutSubject,
    DmdwOutTemperature, DmdwOutTemperatureValue, DmdwPdosOptions, DmdwSelfEnergyDatData,
    DmdwSpectralInfoData, DymData, dmdw_a2_dat_from_coupling, dmdw_a2f_info_from_pole_weighted,
    dmdw_phonon_coupling_from_tables, read_dmdw_a2f_info, read_dmdw_coupling_table, read_dmdw_out,
    read_dym, write_dmdw_a2_dat, write_dmdw_a2f_info, write_dmdw_akw_dat, write_dmdw_egrid_info,
    write_dmdw_out, write_dmdw_self_energy_dat, write_dmdw_spectral_info,
};

use crate::work_dir_for_input;

const DMDW_J_PER_MOL_TO_EV: f64 = 96_485.310_0;
const DMDW_PDOS_DEGENERATE_THRESHOLD_THZ: f64 = 0.000_1;
const DMDW_TYPE2_SELF_ENERGY_POINTS: usize = 10_001;
const DMDW_TYPE2_SELF_ENERGY_WINDOW_W0: f64 = 10.0;
const DMDW_TYPE2_SPECTRAL_TIME_HALF_WIDTH: f64 = 2000.0;
#[cfg(not(test))]
const DMDW_TYPE2_SPECTRAL_TIME_POINTS: usize = 40_001;
#[cfg(test)]
const DMDW_TYPE2_SPECTRAL_TIME_POINTS: usize = 101;

/// Run the supported FEFF DMDW cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF DMDW run can be satisfied from an existing `dmdw.out` cache.
pub(crate) fn has_cached_dmdw_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("dmdw.out").is_file() {
        return Ok(false);
    }
    Ok(matches!(read_input(work_dir)?, DmdwInput::Enabled(_)))
}

/// Run FEFF DMDW from a cache or from supported standalone DMDW branches.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let DmdwInput::Enabled(calculation) = input else {
        return Ok(0);
    };

    let output_path = work_dir.join("dmdw.out");
    if output_path.is_file() {
        let data = read_dmdw_out(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))?;
        let section_count = data.section_count();
        write_cached_output(&output_path, &data)?;
        return Ok(section_count);
    }

    let data = generate_dmdw_output(work_dir, &calculation)?;
    let section_count = data.section_count();
    write_cached_output(&output_path, &data)?;
    write_generated_sidecars(work_dir, &calculation, &data)?;
    Ok(section_count)
}

fn generate_dmdw_output(work_dir: &Path, calculation: &DmdwCalculation) -> Result<DmdwOutData> {
    if !matches!(calculation.calculation_type, 0..=5) {
        bail!(
            "DMDW run type {} is not supported by FEFF DMDW; allowed values are 0 through 5",
            calculation.calculation_type
        );
    }
    if calculation.order <= 0 {
        bail!(
            "DMDW Lanczos recursion order must be positive, got {}",
            calculation.order
        );
    }
    let pole_count = usize::try_from(calculation.order).context("invalid DMDW Lanczos order")?;
    let temperatures = dmdw_temperatures(calculation)?;

    if calculation.calculation_type == 2 {
        return Ok(DmdwOutData {
            header: Some(DmdwOutHeader {
                lanczos_recursion_order: pole_count,
                temperature: dmdw_temperature_header(temperatures.view()),
                dynamical_matrix_file: calculation.dym_file.clone(),
            }),
            mass_enhancement_header: true,
            sections: Vec::new(),
        });
    }

    let dym_path = work_dir.join(&calculation.dym_file);
    let dym =
        read_dym(&dym_path).with_context(|| format!("failed to read {}", dym_path.display()))?;
    let sections = match calculation.calculation_type {
        0 => generate_type0_sections(&dym, calculation, pole_count, temperatures.view())?,
        1 => generate_type1_sections(&dym, calculation, pole_count, temperatures.view())?,
        3 => generate_type3_sections(&dym, calculation, pole_count, temperatures.view())?,
        4 => generate_type4_sections(&dym, calculation, pole_count)?,
        5 => generate_type5_sections(&dym, calculation, pole_count)?,
        branch => {
            bail!(
                "DMDW run type {branch} is not supported by FEFF DMDW; allowed values are 0 through 5"
            )
        }
    };
    if sections.is_empty() {
        bail!(
            "DMDW run type {} generated no output sections from the requested descriptors",
            calculation.calculation_type
        );
    }

    Ok(DmdwOutData {
        header: Some(DmdwOutHeader {
            lanczos_recursion_order: pole_count,
            temperature: dmdw_temperature_header(temperatures.view()),
            dynamical_matrix_file: calculation.dym_file.clone(),
        }),
        mass_enhancement_header: false,
        sections,
    })
}

fn dmdw_temperatures(calculation: &DmdwCalculation) -> Result<Array1<f64>> {
    if calculation.temperature_flag <= 0 {
        bail!(
            "DMDW temperature count must be positive, got {}",
            calculation.temperature_flag
        );
    }
    if !calculation.temperature.is_finite() {
        bail!("DMDW temperature must be finite");
    }

    let temperature_count =
        usize::try_from(calculation.temperature_flag).context("invalid DMDW temperature count")?;
    if temperature_count == 1 {
        return Ok(Array1::from_vec(vec![calculation.temperature]));
    }

    let temperature_max = calculation
        .temperature_max
        .context("DMDW multi-temperature run requires an upper temperature")?;
    if !temperature_max.is_finite() {
        bail!("DMDW upper temperature must be finite");
    }

    let (start, end) = if temperature_max < calculation.temperature {
        (temperature_max, calculation.temperature)
    } else {
        (calculation.temperature, temperature_max)
    };
    Ok(Array1::linspace(start, end, temperature_count))
}

fn dmdw_temperature_header(temperatures: ndarray::ArrayView1<'_, f64>) -> DmdwOutTemperature {
    if temperatures.len() == 1 {
        DmdwOutTemperature::Single(temperatures[0])
    } else {
        DmdwOutTemperature::ListedBelow
    }
}

fn dmdw_descriptors(calculation: &DmdwCalculation) -> Vec<DmdwPathDescriptor> {
    calculation
        .paths
        .iter()
        .map(|path| DmdwPathDescriptor {
            selectors: std::iter::once(path.absorber_selector)
                .chain(path.potentials.iter().copied())
                .collect(),
            max_effective_length: path.max_distance * DMDW_ANGSTROM_TO_BOHR,
        })
        .collect()
}

fn generate_type0_sections(
    dym: &DymData,
    calculation: &DmdwCalculation,
    pole_count: usize,
    temperatures: ndarray::ArrayView1<'_, f64>,
) -> Result<Vec<DmdwOutSection>> {
    let positions = dym.coordinates.cartesian_positions();
    let matrix =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())?;
    let rigid_modes = dmdw_rigid_body_projection_modes(positions.view(), dym.atomic_masses.view())?;
    let descriptors = dmdw_descriptors(calculation);
    let expanded_paths = dmdw_expand_path_descriptors(positions.view(), &descriptors)?;
    let mut sections = Vec::new();

    for path in expanded_paths.iter().filter(|path| path.atoms.len() > 1) {
        let motion = dmdw_path_motion(positions.view(), dym.atomic_masses.view(), &path.atoms)?;
        let seed = dmdw_project_seed_vector(
            motion.initial_vector.view(),
            rigid_modes.projection_modes.view(),
        )?;
        let coefficients =
            dmdw_lanczos_coefficients(matrix.matrix.view(), seed.view(), pole_count)?;
        let spectrum = dmdw_lanczos_pole_spectrum(
            pole_count,
            coefficients.alpha.view(),
            coefficients.beta.view(),
        )?;
        let sigma2 = dmdw_debye_waller_factors_from_poles(
            temperatures,
            motion.reduced_mass,
            spectrum.angular_frequencies.view(),
            spectrum.weights.view(),
        )?;

        let mut section = DmdwOutSection::new(DmdwOutSubject::PathIndices(
            path.atoms.iter().map(|atom| atom + 1).collect(),
        ));
        populate_lanczos_diagnostics(&mut section, &coefficients, &spectrum, motion.reduced_mass)?;
        section.reduced_mass_amu = Some(motion.reduced_mass);
        section.path_length_angstrom =
            Some(dmdw_path_length(positions.view(), &path.atoms)? / DMDW_ANGSTROM_TO_BOHR);
        if temperatures.len() == 1 {
            section.sigma2_1e_minus_3_angstrom2 = sigma2.get(0).map(|value| value * 1000.0);
        } else {
            section.sigma2_by_temperature = temperatures
                .iter()
                .zip(sigma2.iter())
                .map(|(&temperature_kelvin, &value)| DmdwOutTemperatureValue {
                    temperature_kelvin,
                    value: value * 1000.0,
                })
                .collect();
        }
        sections.push(section);
    }

    Ok(sections)
}

fn generate_type3_sections(
    dym: &DymData,
    calculation: &DmdwCalculation,
    pole_count: usize,
    temperatures: ndarray::ArrayView1<'_, f64>,
) -> Result<Vec<DmdwOutSection>> {
    let positions = dym.coordinates.cartesian_positions();
    let matrix =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())?;
    let rigid_modes = dmdw_rigid_body_projection_modes(positions.view(), dym.atomic_masses.view())?;
    let descriptors = dmdw_descriptors(calculation);
    let expanded_paths = dmdw_expand_path_descriptors(positions.view(), &descriptors)?;
    let mut sections = Vec::new();

    for path in expanded_paths.iter().filter(|path| path.atoms.len() == 1) {
        let atom = path.atoms[0];
        let reduced_mass = dym.atomic_masses[atom];
        for perturbation in 0..3 {
            let seed = dmdw_single_atom_seed(
                positions.nrows(),
                atom,
                perturbation,
                rigid_modes.projection_modes.view(),
            )?;
            let coefficients =
                dmdw_lanczos_coefficients(matrix.matrix.view(), seed.view(), pole_count)?;
            let spectrum = dmdw_lanczos_pole_spectrum(
                pole_count,
                coefficients.alpha.view(),
                coefficients.beta.view(),
            )?;
            let u2 = dmdw_debye_waller_factors_from_poles(
                temperatures,
                reduced_mass,
                spectrum.angular_frequencies.view(),
                spectrum.weights.view(),
            )?;

            let mut section = DmdwOutSection::new(DmdwOutSubject::AtomIndex {
                indices: vec![atom + 1],
                direction: Some(dmdw_perturbation_label(perturbation).to_string()),
            });
            populate_lanczos_diagnostics(&mut section, &coefficients, &spectrum, reduced_mass)?;
            if temperatures.len() == 1 {
                section.u2_1e_minus_3_angstrom2 = u2.get(0).map(|value| value * 1000.0);
            } else {
                section.u2_by_temperature = temperatures
                    .iter()
                    .zip(u2.iter())
                    .map(|(&temperature_kelvin, &value)| DmdwOutTemperatureValue {
                        temperature_kelvin,
                        value: value * 1000.0,
                    })
                    .collect();
            }
            sections.push(section);
        }
    }

    Ok(sections)
}

fn generate_type1_sections(
    dym: &DymData,
    calculation: &DmdwCalculation,
    pole_count: usize,
    temperatures: ndarray::ArrayView1<'_, f64>,
) -> Result<Vec<DmdwOutSection>> {
    let positions = dym.coordinates.cartesian_positions();
    let matrix =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())?;
    let rigid_modes = dmdw_rigid_body_projection_modes(positions.view(), dym.atomic_masses.view())?;
    let atom_paths = dmdw_single_atom_paths(positions.view(), calculation)?;
    let mut sections = Vec::new();
    let mut total_free_energy_j_per_mol = Array1::<f64>::zeros(temperatures.len());

    for atom in atom_paths {
        let reduced_mass = dym.atomic_masses[atom];
        for perturbation in 0..3 {
            let seed = dmdw_single_atom_seed(
                positions.nrows(),
                atom,
                perturbation,
                rigid_modes.projection_modes.view(),
            )?;
            let coefficients =
                dmdw_lanczos_coefficients(matrix.matrix.view(), seed.view(), pole_count)?;
            let spectrum = dmdw_lanczos_pole_spectrum(
                pole_count,
                coefficients.alpha.view(),
                coefficients.beta.view(),
            )?;
            let free_energy_j_per_mol = dmdw_vibrational_free_energy_from_poles(
                temperatures,
                spectrum.angular_frequencies.view(),
                spectrum.weights.view(),
            )?;
            for (sum, value) in total_free_energy_j_per_mol
                .iter_mut()
                .zip(free_energy_j_per_mol.iter())
            {
                *sum += *value;
            }

            let mut section = DmdwOutSection::new(DmdwOutSubject::AtomIndex {
                indices: vec![atom + 1],
                direction: Some(dmdw_perturbation_label(perturbation).to_string()),
            });
            populate_lanczos_diagnostics(&mut section, &coefficients, &spectrum, reduced_mass)?;
            set_vfe_result(&mut section, temperatures, free_energy_j_per_mol.view());
            sections.push(section);
        }
    }

    if !sections.is_empty() {
        let mut total = DmdwOutSection::new(DmdwOutSubject::TotalVfe);
        set_vfe_result(&mut total, temperatures, total_free_energy_j_per_mol.view());
        sections.push(total);
    }

    Ok(sections)
}

fn generate_type4_sections(
    dym: &DymData,
    calculation: &DmdwCalculation,
    pole_count: usize,
) -> Result<Vec<DmdwOutSection>> {
    let positions = dym.coordinates.cartesian_positions();
    let matrix =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())?;
    let path = dmdw_first_ir_path(positions.view(), calculation)?;
    let motion = dmdw_path_motion(positions.view(), dym.atomic_masses.view(), &path.atoms)?;
    let dipole_derivatives = dym
        .dipole_derivatives
        .as_ref()
        .context("DMDW run type 4 requires a type 3 .dym file with dipole derivatives")?;
    let seed = dmdw_ir_dipole_seed_vector(dym.atomic_masses.view(), dipole_derivatives.view())?;
    let coefficients = dmdw_lanczos_coefficients(matrix.matrix.view(), seed.view(), pole_count)?;
    let spectrum = dmdw_lanczos_pole_spectrum(
        pole_count,
        coefficients.alpha.view(),
        coefficients.beta.view(),
    )?;

    let mut section = DmdwOutSection::new(DmdwOutSubject::PathIndices(
        path.atoms.iter().map(|atom| atom + 1).collect(),
    ));
    populate_lanczos_diagnostics(&mut section, &coefficients, &spectrum, motion.reduced_mass)?;
    Ok(vec![section])
}

fn generate_type5_sections(
    dym: &DymData,
    calculation: &DmdwCalculation,
    pole_count: usize,
) -> Result<Vec<DmdwOutSection>> {
    let positions = dym.coordinates.cartesian_positions();
    let matrix =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())?;
    let rigid_modes = dmdw_rigid_body_projection_modes(positions.view(), dym.atomic_masses.view())?;
    let atom_paths = dmdw_single_atom_paths(positions.view(), calculation)?;
    let mut sections = Vec::new();
    let mut reduced_masses = Vec::new();
    let mut single_pole_frequencies = Vec::new();

    for atom in atom_paths {
        let reduced_mass = dym.atomic_masses[atom];
        for perturbation in 0..3 {
            let seed = dmdw_single_atom_seed(
                positions.nrows(),
                atom,
                perturbation,
                rigid_modes.projection_modes.view(),
            )?;
            let coefficients =
                dmdw_lanczos_coefficients(matrix.matrix.view(), seed.view(), pole_count)?;
            let spectrum = dmdw_lanczos_pole_spectrum(
                pole_count,
                coefficients.alpha.view(),
                coefficients.beta.view(),
            )?;

            let mut section = DmdwOutSection::new(DmdwOutSubject::AtomIndex {
                indices: vec![atom + 1],
                direction: Some(dmdw_perturbation_label(perturbation).to_string()),
            });
            populate_lanczos_diagnostics(&mut section, &coefficients, &spectrum, reduced_mass)?;
            section.projected_dos_component_computed = true;
            reduced_masses.push(reduced_mass);
            single_pole_frequencies.push(coefficients.single_pole_frequency);
            sections.push(section);
        }
    }

    if !sections.is_empty() {
        sections.push(total_pdos_section(
            &sections,
            &reduced_masses,
            &single_pole_frequencies,
        )?);
    }

    Ok(sections)
}

fn dmdw_first_ir_path(
    atom_positions: ArrayView2<'_, f64>,
    calculation: &DmdwCalculation,
) -> Result<refeff_core::DmdwExpandedPath> {
    let Some(descriptor) = dmdw_descriptors(calculation).into_iter().next() else {
        bail!("DMDW run type 4 requires at least one path descriptor");
    };
    let Some(path) = dmdw_expand_path_descriptor(atom_positions, &descriptor)?
        .into_iter()
        .next()
    else {
        bail!("DMDW run type 4 first path descriptor did not expand to any paths");
    };
    Ok(path)
}

fn total_pdos_section(
    partial_sections: &[DmdwOutSection],
    reduced_masses: &[f64],
    single_pole_frequencies: &[f64],
) -> Result<DmdwOutSection> {
    if partial_sections.is_empty() {
        bail!("DMDW projected DOS needs at least one component");
    }
    if partial_sections.len() != reduced_masses.len()
        || partial_sections.len() != single_pole_frequencies.len()
    {
        bail!("DMDW projected DOS component metadata is inconsistent");
    }

    let mut poles = partial_sections
        .iter()
        .flat_map(|section| section.pdos_poles.iter().cloned())
        .collect::<Vec<_>>();
    normalize_and_merge_pdos_poles(&mut poles)?;

    let average_reduced_mass =
        reduced_masses.iter().copied().sum::<f64>() / reduced_masses.len() as f64;
    let average_single_pole_frequency =
        single_pole_frequencies.iter().copied().sum::<f64>() / single_pole_frequencies.len() as f64;
    let einstein =
        dmdw_single_pole_einstein_summary(average_single_pole_frequency, average_reduced_mass)?;
    let frequencies = Array1::from_vec(poles.iter().map(|pole| pole.frequency_thz).collect());
    let weights = Array1::from_vec(poles.iter().map(|pole| pole.weight).collect());
    let moments =
        dmdw_moment_summaries_from_poles(average_reduced_mass, frequencies.view(), weights.view())?;

    let mut section = DmdwOutSection::new(DmdwOutSubject::TotalPdos);
    section.pdos_poles = poles;
    section.einstein = Some(DmdwOutEinstein {
        frequency_thz: einstein.frequency_thz,
        temperature_kelvin: einstein.temperature_kelvin,
        effective_force_constant_n_per_m: einstein.effective_force_constant_n_per_m,
    });
    section.moments = moments
        .into_iter()
        .map(|moment| DmdwOutMoment {
            order: moment.order,
            moment_thz_power_n: moment.moment_thz_power_n,
            frequency_thz: moment.frequency_thz,
            temperature_kelvin: moment.temperature_kelvin,
            effective_force_constant_n_per_m: moment.effective_force_constant_n_per_m,
        })
        .collect();
    section.projected_dos_component_computed = true;
    Ok(section)
}

fn normalize_and_merge_pdos_poles(poles: &mut Vec<DmdwOutPole>) -> Result<()> {
    if poles.is_empty() {
        bail!("DMDW projected DOS pole table must not be empty");
    }
    let weight_sum = poles.iter().map(|pole| pole.weight).sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        bail!("DMDW projected DOS pole weights must have a positive finite sum");
    }
    for pole in poles.iter_mut() {
        pole.weight /= weight_sum;
    }
    poles.sort_by(|left, right| left.frequency_thz.total_cmp(&right.frequency_thz));

    let mut merged = Vec::with_capacity(poles.len());
    for pole in poles.drain(..) {
        let Some(last) = merged.last_mut() else {
            merged.push(pole);
            continue;
        };
        if (last.frequency_thz - pole.frequency_thz).abs() > DMDW_PDOS_DEGENERATE_THRESHOLD_THZ {
            merged.push(pole);
            continue;
        }
        let combined_weight = last.weight + pole.weight;
        if !combined_weight.is_finite() || combined_weight <= 0.0 {
            bail!("DMDW projected DOS merged pole has a non-positive weight");
        }
        last.frequency_thz =
            (last.frequency_thz * last.weight + pole.frequency_thz * pole.weight) / combined_weight;
        last.weight = combined_weight;
    }
    *poles = merged;
    Ok(())
}

fn dmdw_single_atom_paths(
    atom_positions: ArrayView2<'_, f64>,
    calculation: &DmdwCalculation,
) -> Result<Vec<usize>> {
    if calculation.paths.is_empty() {
        return Ok((0..atom_positions.nrows()).collect());
    }

    let descriptors = dmdw_descriptors(calculation);
    let paths = dmdw_expand_path_descriptors(atom_positions, &descriptors)?
        .into_iter()
        .filter(|path| path.atoms.len() == 1)
        .map(|path| path.atoms[0])
        .collect();
    Ok(paths)
}

fn set_vfe_result(
    section: &mut DmdwOutSection,
    temperatures: ndarray::ArrayView1<'_, f64>,
    free_energy_j_per_mol: ndarray::ArrayView1<'_, f64>,
) {
    if temperatures.len() == 1 {
        section.vibrational_free_energy_ev = free_energy_j_per_mol
            .get(0)
            .map(|value| dmdw_vfe_ev(*value));
    } else {
        section.vibrational_free_energy_by_temperature = temperatures
            .iter()
            .zip(free_energy_j_per_mol.iter())
            .map(|(&temperature_kelvin, &value)| DmdwOutTemperatureValue {
                temperature_kelvin,
                value: dmdw_vfe_ev(value),
            })
            .collect();
    }
}

fn dmdw_vfe_ev(value_j_per_mol: f64) -> f64 {
    value_j_per_mol / DMDW_J_PER_MOL_TO_EV
}

fn populate_lanczos_diagnostics(
    section: &mut DmdwOutSection,
    coefficients: &DmdwLanczosCoefficients,
    spectrum: &DmdwLanczosPoleSpectrum,
    reduced_mass: f64,
) -> Result<()> {
    let einstein =
        dmdw_single_pole_einstein_summary(coefficients.single_pole_frequency, reduced_mass)?;
    let moments = dmdw_moment_summaries_from_poles(
        reduced_mass,
        spectrum.frequencies.view(),
        spectrum.weights.view(),
    )?;

    section.pdos_poles = spectrum
        .frequencies
        .iter()
        .zip(spectrum.weights.iter())
        .map(|(&frequency_thz, &weight)| DmdwOutPole {
            frequency_thz,
            weight,
        })
        .collect();
    section.einstein = Some(DmdwOutEinstein {
        frequency_thz: einstein.frequency_thz,
        temperature_kelvin: einstein.temperature_kelvin,
        effective_force_constant_n_per_m: einstein.effective_force_constant_n_per_m,
    });
    section.moments = moments
        .into_iter()
        .map(|moment| DmdwOutMoment {
            order: moment.order,
            moment_thz_power_n: moment.moment_thz_power_n,
            frequency_thz: moment.frequency_thz,
            temperature_kelvin: moment.temperature_kelvin,
            effective_force_constant_n_per_m: moment.effective_force_constant_n_per_m,
        })
        .collect();
    Ok(())
}

fn dmdw_single_atom_seed(
    atom_count: usize,
    atom: usize,
    perturbation: usize,
    projection_modes: ArrayView2<'_, f64>,
) -> Result<Array1<f64>> {
    if atom >= atom_count {
        bail!("DMDW atom index {atom} is outside 0..{atom_count}");
    }
    if perturbation >= 3 {
        bail!("DMDW perturbation index {perturbation} is outside 0..3");
    }
    let mut seed = Array1::<f64>::zeros(atom_count * 3);
    seed[perturbation * atom_count + atom] = 1.0;
    Ok(dmdw_project_seed_vector(seed.view(), projection_modes)?)
}

fn dmdw_perturbation_label(perturbation: usize) -> &'static str {
    match perturbation {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => "?",
    }
}

fn dmdw_path_length(atom_positions: ArrayView2<'_, f64>, path_atoms: &[usize]) -> Result<f64> {
    if atom_positions.ncols() != 3 {
        bail!(
            "DMDW atom positions must have exactly 3 columns, got {}x{}",
            atom_positions.nrows(),
            atom_positions.ncols()
        );
    }
    if path_atoms.len() < 2 {
        return Ok(0.0);
    }

    path_atoms
        .windows(2)
        .map(|pair| {
            let first = pair[0];
            let second = pair[1];
            if first >= atom_positions.nrows() || second >= atom_positions.nrows() {
                bail!(
                    "DMDW path atom pair {first}-{second} is outside 0..{}",
                    atom_positions.nrows()
                );
            }
            Ok((0..3)
                .map(|component| {
                    let delta =
                        atom_positions[(first, component)] - atom_positions[(second, component)];
                    delta * delta
                })
                .sum::<f64>()
                .sqrt())
        })
        .try_fold(0.0, |sum, length: Result<f64>| {
            let length = length?;
            if !length.is_finite() {
                bail!("DMDW path length became non-finite");
            }
            Ok(sum + length)
        })
}

fn write_generated_sidecars(
    work_dir: &Path,
    calculation: &DmdwCalculation,
    data: &DmdwOutData,
) -> Result<()> {
    if calculation.calculation_type == 2 {
        return write_type2_coupling_sidecar(work_dir, calculation);
    }
    if calculation.calculation_type != 5 {
        return Ok(());
    }

    let options = calculation.pdos_options.clone().unwrap_or_default();
    let selected_sections = data
        .sections
        .iter()
        .filter(|section| {
            matches!(section.subject, DmdwOutSubject::TotalPdos)
                || (options.write_partial
                    && matches!(section.subject, DmdwOutSubject::AtomIndex { .. }))
        })
        .collect::<Vec<_>>();

    for section in selected_sections {
        if matches!(options.format, 0 | 10) {
            write_pdos_poles_sidecar(work_dir, section)?;
        }
        if matches!(options.format, 1 | 10) {
            write_pdos_rect_sidecar(work_dir, section, options.drop_left_edges)?;
        }
        if matches!(options.format, 2 | 10) {
            write_pdos_gaussian_sidecar(work_dir, section, &options)?;
        }
    }
    Ok(())
}

fn write_type2_coupling_sidecar(work_dir: &Path, calculation: &DmdwCalculation) -> Result<()> {
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
    match energy_option {
        0 => Ok(electron_energy),
        1 => Ok(electron_energy * 1.0e-3 / characteristic_energy_ev),
        option => bail!("DMDW type 2 electron-energy option {option} is not supported"),
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

fn write_pdos_poles_sidecar(work_dir: &Path, section: &DmdwOutSection) -> Result<()> {
    let mut out = pdos_pole_comments(section)?;
    let poles = &section.pdos_poles;
    let Some(first) = poles.first() else {
        bail!("DMDW projected DOS sidecar needs at least one pole");
    };
    if first.frequency_thz > 0.0 {
        push_pdos_pair(&mut out, 0.0, 0.0)?;
    } else {
        let spacing = pdos_first_edge_spacing(poles)?;
        push_pdos_pair(&mut out, first.frequency_thz - spacing / 2.0, 0.0)?;
    }

    for pole in poles {
        push_pdos_pair(&mut out, pole.frequency_thz, 0.0)?;
        push_pdos_pair(&mut out, pole.frequency_thz, pole.weight)?;
        push_pdos_pair(&mut out, pole.frequency_thz, 0.0)?;
    }

    let Some(last) = poles.last() else {
        bail!("DMDW projected DOS sidecar needs at least one pole");
    };
    push_pdos_pair(
        &mut out,
        last.frequency_thz + pdos_last_edge_spacing(poles)? / 2.0,
        0.0,
    )?;
    write_pdos_sidecar(work_dir, "poles", section, None, out)
}

fn write_pdos_rect_sidecar(
    work_dir: &Path,
    section: &DmdwOutSection,
    drop_left_edges: bool,
) -> Result<()> {
    let poles = &section.pdos_poles;
    if poles.is_empty() {
        bail!("DMDW rectangular projected DOS sidecar needs at least one pole");
    }

    let mut out = pdos_pole_comments(section)?;
    let edges = pdos_rect_edges(poles)?;
    for window in edges.windows(2) {
        let left = window[0];
        let right = window[1];
        let width = right - left;
        if !width.is_finite() || width <= 0.0 {
            bail!("DMDW rectangular projected DOS bin has non-positive width");
        }
        let weight = poles
            .iter()
            .filter(|pole| left < pole.frequency_thz && pole.frequency_thz < right)
            .map(|pole| pole.weight / width)
            .sum::<f64>();
        if drop_left_edges {
            push_pdos_pair(&mut out, left, 0.0)?;
        }
        push_pdos_pair(&mut out, left, weight)?;
        push_pdos_pair(&mut out, right, weight)?;
        if drop_left_edges {
            push_pdos_pair(&mut out, right, 0.0)?;
        }
    }

    write_pdos_sidecar(
        work_dir,
        "rect",
        section,
        drop_left_edges.then_some("wdl"),
        out,
    )
}

fn write_pdos_gaussian_sidecar(
    work_dir: &Path,
    section: &DmdwOutSection,
    options: &DmdwPdosOptions,
) -> Result<()> {
    let poles = &section.pdos_poles;
    if poles.is_empty() {
        bail!("DMDW Gaussian projected DOS sidecar needs at least one pole");
    }
    if !options.gaussian_broadening_thz.is_finite() || options.gaussian_broadening_thz <= 0.0 {
        bail!("DMDW Gaussian projected DOS broadening must be positive and finite");
    }
    if !options.gaussian_resolution_thz.is_finite() || options.gaussian_resolution_thz <= 0.0 {
        bail!("DMDW Gaussian projected DOS resolution must be positive and finite");
    }

    let mut out = pdos_pole_comments(section)?;
    let broadening = options.gaussian_broadening_thz;
    let resolution = options.gaussian_resolution_thz;
    let min_frequency = poles
        .iter()
        .map(|pole| pole.frequency_thz)
        .fold(f64::INFINITY, f64::min);
    let max_frequency = poles
        .iter()
        .map(|pole| pole.frequency_thz)
        .fold(f64::NEG_INFINITY, f64::max);
    let start = 0.0_f64.min(min_frequency - 6.0 * broadening);
    let end = max_frequency + 6.0 * broadening;
    let point_count = ((end - start) / resolution).floor() as usize + 2;
    if point_count > 2_000_000 {
        bail!("DMDW Gaussian projected DOS grid would write {point_count} points");
    }

    let ln2 = std::f64::consts::LN_2;
    let sqrt_pi_over_ln2 = (std::f64::consts::PI / ln2).sqrt();
    let beta_arg = ln2 / (broadening / 2.0).powi(2);
    let height_arg = 1.0 / ((broadening / 2.0) * sqrt_pi_over_ln2);
    for index in 0..point_count {
        let frequency = start + resolution * index as f64;
        let weight = poles
            .iter()
            .map(|pole| {
                height_arg
                    * pole.weight
                    * (-(beta_arg) * (frequency - pole.frequency_thz).powi(2)).exp()
            })
            .sum::<f64>();
        push_pdos_pair(&mut out, frequency, weight)?;
    }

    write_pdos_sidecar(work_dir, "gaussian", section, None, out)
}

fn pdos_pole_comments(section: &DmdwOutSection) -> Result<String> {
    use std::fmt::Write as _;

    let mut out = String::new();
    for pole in &section.pdos_poles {
        writeln!(out, "#{:12.6}{:12.6}", pole.frequency_thz, pole.weight)?;
    }
    out.push('\n');
    Ok(out)
}

fn pdos_rect_edges(poles: &[DmdwOutPole]) -> Result<Vec<f64>> {
    let Some(first) = poles.first() else {
        bail!("DMDW rectangular projected DOS needs at least one pole");
    };
    if poles.len() == 1 {
        let spacing = pdos_edge_spacing(poles)?;
        return Ok(vec![
            (0.0_f64).min(first.frequency_thz - spacing),
            first.frequency_thz + spacing,
        ]);
    }

    let mut edges = Vec::new();
    if first.frequency_thz > 0.0 {
        edges.push(0.0);
        edges.push(first.frequency_thz / 2.0);
    } else {
        let second = poles[1].frequency_thz;
        edges.push(2.0 * first.frequency_thz - second);
        edges.push(first.frequency_thz - (second - first.frequency_thz) / 2.0);
    }
    for pair in poles.windows(2) {
        let left = pair[0].frequency_thz;
        let right = pair[1].frequency_thz;
        if left < 0.0 && right > 0.0 {
            edges.push(left / 2.0);
            edges.push(right / 2.0);
        } else {
            edges.push((left + right) / 2.0);
        }
    }
    let last = poles[poles.len() - 1].frequency_thz;
    let previous_edge = *edges
        .last()
        .ok_or_else(|| anyhow::anyhow!("DMDW rectangular PDOS edge list is empty"))?;
    edges.push(2.0 * last - previous_edge);
    edges.push(3.0 * last - 2.0 * previous_edge);
    Ok(edges)
}

fn pdos_first_edge_spacing(poles: &[DmdwOutPole]) -> Result<f64> {
    match poles {
        [] => bail!("DMDW projected DOS pole table must not be empty"),
        [_] => Ok(1.0),
        [first, second, ..] => pdos_spacing(first.frequency_thz, second.frequency_thz),
    }
}

fn pdos_last_edge_spacing(poles: &[DmdwOutPole]) -> Result<f64> {
    match poles {
        [] => bail!("DMDW projected DOS pole table must not be empty"),
        [_] => Ok(1.0),
        [.., previous, last] => pdos_spacing(previous.frequency_thz, last.frequency_thz),
    }
}

fn pdos_edge_spacing(poles: &[DmdwOutPole]) -> Result<f64> {
    pdos_last_edge_spacing(poles)
}

fn pdos_spacing(left: f64, right: f64) -> Result<f64> {
    let spacing = right - left;
    if !spacing.is_finite() || spacing == 0.0 {
        Ok(1.0)
    } else {
        Ok(spacing.abs())
    }
}

fn push_pdos_pair(out: &mut String, frequency: f64, weight: f64) -> Result<()> {
    use std::fmt::Write as _;

    writeln!(out, "{frequency:12.6}{weight:12.6}")?;
    Ok(())
}

fn write_pdos_sidecar(
    work_dir: &Path,
    format: &str,
    section: &DmdwOutSection,
    suffix: Option<&str>,
    text: String,
) -> Result<()> {
    let mut label = format!("dmdw_pdos.{}.{}", format, pdos_section_label(section)?);
    if let Some(suffix) = suffix {
        label.push('.');
        label.push_str(suffix);
    }
    label.push_str(".dat");
    let path = work_dir.join(label);
    std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
}

fn pdos_section_label(section: &DmdwOutSection) -> Result<String> {
    match &section.subject {
        DmdwOutSubject::TotalPdos => Ok("tot".to_string()),
        DmdwOutSubject::AtomIndex { indices, direction } if indices.len() == 1 => {
            let direction = direction.as_deref().unwrap_or("?");
            Ok(format!("{:03}.{direction}", indices[0]))
        }
        DmdwOutSubject::PathIndices(indices) => Ok(indices
            .iter()
            .map(|index| format!("{index:03}"))
            .collect::<Vec<_>>()
            .join(".")),
        subject => bail!("DMDW projected DOS sidecar does not support subject {subject:?}"),
    }
}

fn read_input(work_dir: &Path) -> Result<DmdwInput> {
    let input_path = work_dir.join("dmdw.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    DmdwInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &DmdwOutData) -> Result<()> {
    write_dmdw_out(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests;
