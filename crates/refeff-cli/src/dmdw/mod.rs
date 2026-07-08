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
    dmdw_out_string, dmdw_phonon_coupling_from_tables, read_dmdw_a2f_info,
    read_dmdw_coupling_table, read_dmdw_out, read_dym, write_dmdw_a2_dat, write_dmdw_a2f_info,
    write_dmdw_akw_dat, write_dmdw_egrid_info, write_dmdw_out, write_dmdw_self_energy_dat,
    write_dmdw_spectral_info,
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

mod pdos;
mod type2;

/// Run the supported FEFF DMDW cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF DMDW run can be satisfied from an existing `dmdw.out` cache.
pub(crate) fn has_cached_dmdw_output(work_dir: &Path) -> Result<bool> {
    let output_path = work_dir.join("dmdw.out");
    if !work_dir.join("dmdw.inp").is_file() || !output_path.is_file() {
        return Ok(false);
    }
    let Ok(DmdwInput::Enabled(calculation)) = read_input(work_dir) else {
        return Ok(false);
    };
    if read_dmdw_out(&output_path).is_err() {
        return Ok(false);
    }
    if validate_declared_dmdw_source_handoffs(work_dir, &calculation).is_err() {
        return Ok(false);
    }
    Ok(true)
}

/// Whether FEFF DMDW can generate outputs from supported source handoffs.
pub(crate) fn has_supported_dmdw_source_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("dmdw.inp").is_file() {
        return Ok(false);
    }
    let Ok(DmdwInput::Enabled(calculation)) = read_input(work_dir) else {
        return Ok(false);
    };
    Ok(dmdw_source_handoff_present_for_calculation(
        work_dir,
        &calculation,
    ))
}

/// Run FEFF DMDW from a cache or from supported standalone DMDW branches.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let DmdwInput::Enabled(calculation) = input else {
        return Ok(0);
    };

    let output_path = work_dir.join("dmdw.out");
    if output_path.is_file() {
        match read_dmdw_out(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))
        {
            Ok(data) => {
                if let Some(generated) =
                    generate_dmdw_if_stale_against_source(work_dir, &calculation, &data)?
                {
                    let section_count = generated.section_count();
                    write_cached_output(&output_path, &generated)?;
                    write_generated_sidecars(work_dir, &calculation, &generated)?;
                    return Ok(section_count);
                }
                let section_count = data.section_count();
                write_cached_output(&output_path, &data)?;
                return Ok(section_count);
            }
            Err(error) => {
                if !dmdw_source_handoff_present_for_calculation(work_dir, &calculation) {
                    return Err(error);
                }
            }
        }
    }

    let data = generate_dmdw_output(work_dir, &calculation)?;
    let section_count = data.section_count();
    write_cached_output(&output_path, &data)?;
    write_generated_sidecars(work_dir, &calculation, &data)?;
    Ok(section_count)
}

fn generate_dmdw_if_stale_against_source(
    work_dir: &Path,
    calculation: &DmdwCalculation,
    cached: &DmdwOutData,
) -> Result<Option<DmdwOutData>> {
    validate_declared_dmdw_source_handoffs(work_dir, calculation)?;
    if !dmdw_source_handoff_present_for_calculation(work_dir, calculation) {
        return Ok(None);
    }
    let generated = match generate_dmdw_output(work_dir, calculation) {
        Ok(generated) => generated,
        Err(_) => return Ok(None),
    };
    if dmdw_out_string(cached)? == dmdw_out_string(&generated)? {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn dmdw_source_handoff_present_for_calculation(
    work_dir: &Path,
    calculation: &DmdwCalculation,
) -> bool {
    if !matches!(calculation.calculation_type, 0..=5) || calculation.order <= 0 {
        return false;
    }
    if calculation.calculation_type == 2 {
        return type2_coupling_source_handoff_is_parseable(work_dir, calculation);
    }
    dym_source_handoff_is_parseable(work_dir, &calculation.dym_file)
}

fn validate_declared_dmdw_source_handoffs(
    work_dir: &Path,
    calculation: &DmdwCalculation,
) -> Result<()> {
    validate_present_dym_source_handoff(work_dir, calculation)?;
    if calculation.calculation_type == 2 {
        validate_declared_type2_coupling_source_handoffs(work_dir, calculation)?;
    }
    Ok(())
}

fn validate_present_dym_source_handoff(
    work_dir: &Path,
    calculation: &DmdwCalculation,
) -> Result<()> {
    let path = work_dir.join(&calculation.dym_file);
    if path.is_file() {
        read_dym(&path).with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(())
}

fn validate_declared_type2_coupling_source_handoffs(
    work_dir: &Path,
    calculation: &DmdwCalculation,
) -> Result<()> {
    let Some(options) = calculation.self_energy_options.as_ref() else {
        return Ok(());
    };
    let pds_path = work_dir.join(&options.pds_file);
    let a2f_path = work_dir.join(&options.a2f_file);
    let pds = if pds_path.is_file() {
        Some(
            read_dmdw_coupling_table(&pds_path)
                .with_context(|| format!("failed to read {}", pds_path.display()))?,
        )
    } else {
        None
    };
    let a2f = if a2f_path.is_file() {
        Some(
            read_dmdw_coupling_table(&a2f_path)
                .with_context(|| format!("failed to read {}", a2f_path.display()))?,
        )
    } else {
        None
    };
    if let (Some(pds), Some(a2f)) = (pds.as_ref(), a2f.as_ref()) {
        dmdw_phonon_coupling_from_tables(pds, a2f)
            .context("failed to validate DMDW type 2 phonon coupling source handoff")?;
    }
    Ok(())
}

fn dym_source_handoff_is_parseable(work_dir: &Path, file_name: &str) -> bool {
    let path = work_dir.join(file_name);
    path.is_file() && read_dym(&path).is_ok()
}

fn type2_coupling_source_handoff_is_parseable(
    work_dir: &Path,
    calculation: &DmdwCalculation,
) -> bool {
    let Some(options) = calculation.self_energy_options.as_ref() else {
        return false;
    };
    let pds_path = work_dir.join(&options.pds_file);
    let a2f_path = work_dir.join(&options.a2f_file);
    if !pds_path.is_file() || !a2f_path.is_file() {
        return false;
    }
    let Ok(pds) = read_dmdw_coupling_table(&pds_path) else {
        return false;
    };
    let Ok(a2f) = read_dmdw_coupling_table(&a2f_path) else {
        return false;
    };
    dmdw_phonon_coupling_from_tables(&pds, &a2f).is_ok()
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
        return type2::write_type2_coupling_sidecar(work_dir, calculation);
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
            pdos::write_pdos_poles_sidecar(work_dir, section)?;
        }
        if matches!(options.format, 1 | 10) {
            pdos::write_pdos_rect_sidecar(work_dir, section, options.drop_left_edges)?;
        }
        if matches!(options.format, 2 | 10) {
            pdos::write_pdos_gaussian_sidecar(work_dir, section, &options)?;
        }
    }
    Ok(())
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
