use super::*;

/// Port FEFF DMDW run-type 2 `phonon_coupling` table preparation.
///
/// FEFF skips the first ten lines of the PDS and `a2f` files, then reads two
/// columns from each row. The PDS value is the projected phonon density of
/// states, the `a2f` value is the Eliashberg coupling, and this helper returns
/// the FEFF `a2` matrix-element table (`eli / phdos`) plus the normalization
/// accumulated as `phdos * (w - wprev) * 27.211396`.
pub fn dmdw_phonon_coupling(
    pds_energy_hartree: ArrayView1<'_, Real>,
    phonon_dos: ArrayView1<'_, Real>,
    a2f_energy_hartree: ArrayView1<'_, Real>,
    eliashberg: ArrayView1<'_, Real>,
) -> Result<DmdwPhononCoupling, DebyeError> {
    validate_dmdw_coupling_tables(
        pds_energy_hartree,
        phonon_dos,
        a2f_energy_hartree,
        eliashberg,
    )?;

    let point_count = pds_energy_hartree.len();
    let mut energy_hartree = Vec::with_capacity(point_count);
    let mut energy_ev = Vec::with_capacity(point_count);
    let mut matrix_element = Vec::with_capacity(point_count);
    let mut previous_energy = 0.0;
    let mut normalization = 0.0;

    for row in 0..point_count {
        let energy = a2f_energy_hartree[row];
        let density = phonon_dos[row];
        let coupling = eliashberg[row];
        normalization += density * (energy - previous_energy) * DMDW_COUPLING_NORM_HARTREE_EV;
        previous_energy = energy;

        let energy_ev_value = energy * DMDW_COUPLING_ENERGY_HARTREE_EV;
        let matrix_value = coupling / density;
        ensure_finite_output("DMDW coupling energy eV", energy_ev_value)?;
        ensure_finite_output("DMDW coupling matrix element", matrix_value)?;
        ensure_finite_output("DMDW coupling normalization", normalization)?;

        energy_hartree.push(energy);
        energy_ev.push(energy_ev_value);
        matrix_element.push(matrix_value);
    }
    ensure_positive("DMDW coupling normalization", normalization)?;

    Ok(DmdwPhononCoupling {
        energy_hartree: Array1::from_vec(energy_hartree),
        energy_ev: Array1::from_vec(energy_ev),
        eliashberg: eliashberg.to_owned(),
        matrix_element: Array1::from_vec(matrix_element),
        normalization,
    })
}

/// Port FEFF DMDW run-type 2 pole-weight `a2f` preparation.
///
/// This mirrors the `m_dmdw.f90` loop that writes `dmdw_a2f.info`: Lanczos
/// poles are compared against the coupling grid using FEFF's diagnostic
/// `w_pole / 6.28` THz conversion, each matched pole receives
/// `a2(2,inull) * wil * norm`, and imaginary poles keep zero weight.
pub fn dmdw_pole_weighted_a2f(
    spectrum: &DmdwLanczosPoleSpectrum,
    coupling: &DmdwPhononCoupling,
) -> Result<DmdwPoleWeightedA2f, DebyeError> {
    validate_dmdw_frequency_weight_poles(
        spectrum.angular_frequencies.view(),
        spectrum.weights.view(),
    )?;
    validate_dmdw_phonon_coupling_result(coupling)?;

    let pole_count = spectrum.angular_frequencies.len();
    let mut pole_energy_ev = vec![0.0; pole_count];
    let mut pole_weight = vec![0.0; pole_count];
    let mut matched = 0;

    for row in 0..coupling.point_count() {
        if matched >= pole_count {
            break;
        }
        let pole_frequency_thz = spectrum.angular_frequencies[matched] / DMDW_A2F_DIAGNOSTIC_TWO_PI;
        let coupling_frequency_thz = coupling.energy_hartree[row] * DMDW_COUPLING_NORM_HARTREE_EV
            / DMDW_A2F_PLANCK_EV_PS
            * 1000.0;
        ensure_finite_output("DMDW a2f diagnostic pole frequency", pole_frequency_thz)?;
        ensure_finite_output(
            "DMDW a2f diagnostic coupling frequency",
            coupling_frequency_thz,
        )?;

        if pole_frequency_thz - coupling_frequency_thz < 0.0 {
            let energy = spectrum.angular_frequencies[matched] * DMDW_A2F_POLE_ANGULAR_TO_EV;
            let weight = if spectrum.angular_frequencies[matched] > 0.0 {
                coupling.matrix_element[row] * spectrum.weights[matched] * coupling.normalization
            } else {
                0.0
            };
            ensure_finite_output("DMDW a2f pole energy", energy)?;
            ensure_finite_output("DMDW a2f pole weight", weight)?;
            pole_energy_ev[matched] = energy;
            pole_weight[matched] = weight;
            matched += 1;
        }
    }

    if matched != pole_count {
        return Err(DebyeError::UnmatchedDmdwA2fPole {
            pole_index: matched,
            frequency_thz: spectrum.angular_frequencies[matched] / DMDW_A2F_DIAGNOSTIC_TWO_PI,
            coupling_points: coupling.point_count(),
        });
    }

    let mut weighted_energy = 0.0;
    let mut total_weight = 0.0;
    let mut mass_enhancement = 0.0;
    for (&energy, &weight) in pole_energy_ev.iter().zip(pole_weight.iter()) {
        weighted_energy += energy * weight;
        total_weight += weight;
        mass_enhancement += 2.0 * weight / energy;
        ensure_finite_output("DMDW a2f weighted energy", weighted_energy)?;
        ensure_finite_output("DMDW a2f total weight", total_weight)?;
        ensure_finite_output("DMDW a2f mass enhancement", mass_enhancement)?;
    }
    ensure_positive("DMDW a2f total pole weight", total_weight)?;
    let characteristic_energy_ev = weighted_energy / total_weight;
    ensure_positive("DMDW a2f characteristic energy", characteristic_energy_ev)?;

    Ok(DmdwPoleWeightedA2f {
        lanczos_frequency_thz: spectrum
            .angular_frequencies
            .mapv(|frequency| frequency / DMDW_A2F_DIAGNOSTIC_TWO_PI),
        lanczos_weight: spectrum.weights.clone(),
        normalization: coupling.normalization,
        pole_energy_ev: Array1::from_vec(pole_energy_ev),
        pole_weight: Array1::from_vec(pole_weight),
        mass_enhancement,
        characteristic_energy_ev,
    })
}

/// Port FEFF DMDW run-type 2 unique-atom displacement `a2f` preparation.
///
/// This builds FEFF's mass-weighted dynamical matrix, creates one unit
/// displacement seed per requested type-2 central atom, runs Lanczos, stitches
/// the per-group pole weights using FEFF's normalization rules, and then maps
/// those poles through [`dmdw_pole_weighted_a2f`].
pub fn dmdw_type2_pole_weighted_a2f(
    force_blocks: ArrayView4<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    unique_atom_groups: &[DmdwType2AtomGroup],
    displacement_option: i32,
    pole_count: usize,
    coupling: &DmdwPhononCoupling,
) -> Result<DmdwPoleWeightedA2f, DebyeError> {
    validate_dmdw_type2_a2f_inputs(
        atom_masses.len(),
        unique_atom_groups,
        displacement_option,
        pole_count,
    )?;
    let matrix = dmdw_mass_weighted_dynamical_matrix(force_blocks, atom_masses)?;
    let group_count = unique_atom_groups.len();
    let mut pole_weights_by_group = vec![0.0; group_count * pole_count];
    let mut pole_frequencies_by_group = vec![0.0; group_count * pole_count];
    let normalization = if displacement_option == 0 {
        dmdw_type2_accumulate_all_displacements(
            matrix.matrix.view(),
            atom_masses,
            unique_atom_groups,
            pole_count,
            &mut pole_weights_by_group,
            &mut pole_frequencies_by_group,
        )?
    } else {
        dmdw_type2_accumulate_selected_displacement(
            matrix.matrix.view(),
            atom_masses,
            unique_atom_groups,
            displacement_option as usize - 1,
            pole_count,
            &mut pole_weights_by_group,
            &mut pole_frequencies_by_group,
        )?
    };
    ensure_positive("DMDW type 2 Lanczos weight normalization", normalization)?;
    for weight in &mut pole_weights_by_group {
        *weight /= normalization;
    }

    let group_scale = 1.0 / group_count as Real;
    let angular_frequencies = (0..pole_count)
        .map(|pole| {
            (0..group_count)
                .map(|group| pole_frequencies_by_group[group * pole_count + pole])
                .sum::<Real>()
                * group_scale
        })
        .collect::<Array1<_>>();
    let weights = (0..pole_count)
        .map(|pole| {
            (0..group_count)
                .map(|group| pole_weights_by_group[group * pole_count + pole])
                .sum::<Real>()
                * group_scale
        })
        .collect::<Array1<_>>();
    let spectrum = DmdwLanczosPoleSpectrum {
        expected_poles: pole_count,
        squared_angular_frequencies: angular_frequencies.mapv(|value| value * value),
        angular_frequencies: angular_frequencies.clone(),
        frequencies: angular_frequencies.mapv(|value| value / std::f64::consts::TAU),
        weights,
        imaginary_warnings: Vec::new(),
    };
    dmdw_pole_weighted_a2f(&spectrum, coupling)
}

fn dmdw_type2_accumulate_all_displacements(
    matrix: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    unique_atom_groups: &[DmdwType2AtomGroup],
    pole_count: usize,
    pole_weights_by_group: &mut [Real],
    pole_frequencies_by_group: &mut [Real],
) -> Result<Real, DebyeError> {
    let atom_count = atom_masses.len();
    let mut normalization = 0.0;
    for (group_index, unique_atom) in unique_atom_groups.iter().enumerate() {
        let offset = group_index * pole_count;
        for &atom in &unique_atom.center_atom_indices {
            for component in 0..3 {
                let spectrum = dmdw_type2_lanczos_for_unit_displacement(
                    matrix, atom_count, atom, component, pole_count,
                )?;
                for pole in 0..pole_count {
                    pole_weights_by_group[offset + pole] +=
                        spectrum.weights[pole] * atom_masses[atom];
                    pole_frequencies_by_group[offset + pole] +=
                        spectrum.angular_frequencies[pole] / 3.0;
                }
            }
            normalization += (0..pole_count)
                .map(|pole| pole_weights_by_group[offset + pole])
                .sum::<Real>();
        }
    }
    Ok(normalization)
}

fn dmdw_type2_accumulate_selected_displacement(
    matrix: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    unique_atom_groups: &[DmdwType2AtomGroup],
    component: usize,
    pole_count: usize,
    pole_weights_by_group: &mut [Real],
    pole_frequencies_by_group: &mut [Real],
) -> Result<Real, DebyeError> {
    let atom_count = atom_masses.len();
    let mut normalization = 0.0;
    for (group_index, unique_atom) in unique_atom_groups.iter().enumerate() {
        let offset = group_index * pole_count;
        for &atom in &unique_atom.center_atom_indices {
            let spectrum = dmdw_type2_lanczos_for_unit_displacement(
                matrix, atom_count, atom, component, pole_count,
            )?;
            normalization = 0.0;
            for pole in 0..pole_count {
                pole_weights_by_group[offset + pole] = spectrum.weights[pole] * atom_masses[atom];
                pole_frequencies_by_group[offset + pole] = spectrum.angular_frequencies[pole];
                normalization += pole_weights_by_group[offset + pole];
            }
        }
    }
    Ok(normalization)
}

fn dmdw_type2_lanczos_for_unit_displacement(
    matrix: ArrayView2<'_, Real>,
    atom_count: usize,
    atom: usize,
    component: usize,
    pole_count: usize,
) -> Result<DmdwLanczosPoleSpectrum, DebyeError> {
    let mut seed = Array1::<Real>::zeros(atom_count * 3);
    seed[component * atom_count + atom] = 1.0;
    let coefficients = dmdw_lanczos_coefficients(matrix, seed.view(), pole_count)?;
    dmdw_lanczos_pole_spectrum(
        pole_count,
        coefficients.alpha.view(),
        coefficients.beta.view(),
    )
}
