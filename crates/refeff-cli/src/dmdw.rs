use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, ArrayView2};
use refeff_core::{
    DMDW_ANGSTROM_TO_BOHR, DmdwLanczosCoefficients, DmdwLanczosPoleSpectrum, DmdwPathDescriptor,
    dmdw_debye_waller_factors_from_poles, dmdw_expand_path_descriptors, dmdw_lanczos_coefficients,
    dmdw_lanczos_pole_spectrum, dmdw_mass_weighted_dynamical_matrix,
    dmdw_moment_summaries_from_poles, dmdw_path_motion, dmdw_project_seed_vector,
    dmdw_rigid_body_projection_modes, dmdw_single_pole_einstein_summary,
    dmdw_vibrational_free_energy_from_poles,
};
use refeff_io::{
    DmdwCalculation, DmdwInput, DmdwOutData, DmdwOutEinstein, DmdwOutHeader, DmdwOutMoment,
    DmdwOutPole, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature, DmdwOutTemperatureValue,
    DmdwPdosOptions, DymData, read_dmdw_out, read_dym, write_dmdw_out,
};

use crate::work_dir_for_input;

const DMDW_J_PER_MOL_TO_EV: f64 = 96_485.310_0;
const DMDW_PDOS_DEGENERATE_THRESHOLD_THZ: f64 = 0.000_1;

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
    if !matches!(calculation.calculation_type, 0 | 1 | 3 | 5) {
        bail!(
            "DMDW run type {} generation requires an unported DMDW solver branch",
            calculation.calculation_type
        );
    }
    if calculation.order <= 0 {
        bail!(
            "DMDW Lanczos recursion order must be positive, got {}",
            calculation.order
        );
    }

    let dym_path = work_dir.join(&calculation.dym_file);
    let dym =
        read_dym(&dym_path).with_context(|| format!("failed to read {}", dym_path.display()))?;
    let pole_count = usize::try_from(calculation.order).context("invalid DMDW Lanczos order")?;
    let temperatures = dmdw_temperatures(calculation)?;
    let sections = match calculation.calculation_type {
        0 => generate_type0_sections(&dym, calculation, pole_count, temperatures.view())?,
        1 => generate_type1_sections(&dym, calculation, pole_count, temperatures.view())?,
        3 => generate_type3_sections(&dym, calculation, pole_count, temperatures.view())?,
        5 => generate_type5_sections(&dym, calculation, pole_count)?,
        branch => {
            bail!("DMDW run type {branch} generation requires an unported DMDW solver branch")
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
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use ndarray::{Array4, arr1, arr2};
    use refeff_io::{
        DmdwOutData, DmdwOutHeader, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature,
        DymCoordinates, DymData, read_dmdw_out, write_dmdw_out, write_dym,
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
    fn dmdw_module_rejects_unsupported_generation_until_solver_branch_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_unsupported_dmdw_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("unsupported DMDW should require an unported solver branch")?;

        assert!(
            error
                .to_string()
                .contains("DMDW run type 2 generation requires an unported DMDW solver branch")
        );
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
            eprintln!(
                "skipping DMDW reference test; generated DEBYE/DM/EXAFS/Cu reference not found"
            );
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
                "   2\n",
                "feff.dym\n",
                "   1\n",
                "   1   1              10.00\n",
            ),
        )?;
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
        }
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
            sections: vec![section],
        }
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
}
