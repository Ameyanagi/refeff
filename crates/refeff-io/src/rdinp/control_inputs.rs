use std::fmt::Write as _;

use super::*;

/// Render FEFF-compatible `global.inp` content from an [`FeffDocument`].
pub fn global_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_global_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `reciprocal.inp` content for real-space inputs.
#[must_use]
pub fn reciprocal_inp_string() -> String {
    "ispace\n   1\n".to_string()
}

/// Render FEFF-compatible `crpa.inp` content with current CRPA controls.
#[must_use]
pub fn crpa_inp_string(document: &FeffDocument) -> String {
    let crpa = document.crpa;
    format!(
        " do_CRPA{:12}\n rcut{:21.16}     \n l_crpa{:12}\n",
        i32::from(crpa.enabled),
        crpa.rcut,
        crpa.l
    )
}

/// Render FEFF-compatible `config.inp` content from `CONFIG card` payload rows.
pub fn config_inp_string(document: &FeffDocument) -> Result<String> {
    config_inp_lines_string(&document.config_records)
}

/// Render FEFF-compatible `fullspectrum.inp` content with current defaults.
#[must_use]
pub fn fullspectrum_inp_string() -> String {
    " mFullSpectrum\n           0\n".to_string()
}

/// Render FEFF-compatible `fullspectrum.inp` content from `FULLSPECTRUM`.
pub fn fullspectrum_inp_string_for_document(document: &FeffDocument) -> Result<String> {
    fullspectrum_input_string(&document.full_spectrum_input)
}

/// Render FEFF-compatible default `eels.inp` content.
pub fn eels_inp_string(document: &FeffDocument) -> Result<String> {
    let eels = document.eels;
    let beam_direction = if eels.enabled {
        eels.beam_direction
    } else {
        document
            .nrixs
            .as_ref()
            .map(|nrixs| nrixs.qvec)
            .filter(|vector| vector_norm(*vector) > 0.0)
            .unwrap_or(document.incidence_vector)
    };
    let mut out = String::new();
    writeln!(out, "calculate ELNES?")?;
    write_i4_list(&mut out, [i32::from(eels.enabled)])?;
    writeln!(out, "average? relativistic? cross-terms? Which input?")?;
    write_i4_list(
        &mut out,
        [
            eels.average,
            eels.relativistic,
            eels.cross_terms,
            eels.input,
            eels.spectrum_column,
        ],
    )?;
    writeln!(out, "polarizations to be used ; min step max")?;
    write_i4_list(
        &mut out,
        [
            eels.polarization_min,
            eels.polarization_step,
            eels.polarization_max,
        ],
    )?;
    writeln!(out, "beam energy in eV")?;
    writeln!(out, "{:13.5}", eels.beam_energy)?;
    writeln!(out, "beam direction in arbitrary units")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        beam_direction[0], beam_direction[1], beam_direction[2]
    )?;
    writeln!(out, "collection and convergence semiangle in rad")?;
    writeln!(
        out,
        "{:13.5}{:13.5}",
        eels.collection_angle, eels.convergence_angle
    )?;
    writeln!(out, "qmesh - radial and angular grid size")?;
    write_i4_list(&mut out, [eels.qmesh_radial, eels.qmesh_angular])?;
    writeln!(out, "detector positions - two angles in rad")?;
    writeln!(out, "{:13.5}{:13.5}", eels.detector[0], eels.detector[1])?;
    writeln!(out, "calculate magic angle if magic=1")?;
    write_i4_list(&mut out, [eels.magic])?;
    writeln!(out, "energy for magic angle - eV above threshold")?;
    writeln!(out, "{:13.5}", eels.magic_energy)?;
    Ok(out)
}

/// Render FEFF-compatible default `mdff.inp` content for `MDFF 3`.
pub fn mdff_inp_string() -> Result<String> {
    crate::mdff_input::mdff_input_string(&crate::mdff_input::MdffInput {
        task: 1,
        q_input: 2,
    })
}

/// Render FEFF-compatible `compton.inp` content from the COMPTON card family.
pub fn compton_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_compton_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible default `band.inp` content.
#[must_use]
pub fn band_inp_string() -> String {
    concat!(
        "mband : calculate bands if = 1\n",
        "   0\n",
        "emin, emax, estep : energy mesh\n",
        "      0.00000      0.00000      0.00000\n",
        "nkp : # points in k-path\n",
        "   0\n",
        "ikpath : type of k-path\n",
        "  -1\n",
        "freeprop :  empty lattice if = T\n",
        " F\n",
    )
    .to_string()
}

fn write_compton_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let compton = &document.compton;
    let run_compton = i32::from(compton.do_compton || compton.do_rhozzp);

    writeln!(out, "run compton module?")?;
    writeln!(out, "{run_compton:12}")?;
    writeln!(out, "pqmax, npq")?;
    writeln!(out, "{:13.8}{:16}", compton.pqmax, compton.npq)?;
    writeln!(out, "ns, nphi, nz, nzp")?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}",
        compton.ns, compton.nphi, compton.nz, compton.nzp
    )?;
    writeln!(out, "smax, phimax, zmax, zpmax")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        0.0,
        std::f64::consts::TAU,
        0.0,
        compton.zpmax
    )?;
    writeln!(out, "jpq? rhozzp? force_recalc_jzzp?")?;
    writeln!(
        out,
        " {} {} {}",
        fortran_bool(compton.do_compton),
        fortran_bool(compton.do_rhozzp),
        fortran_bool(compton.force_jzzp)
    )?;
    writeln!(out, "window_type (0=Step, 1=Hann), window_cutoff")?;
    writeln!(out, "           1   0.00000000    ")?;
    writeln!(out, "temperature (in eV)")?;
    writeln!(out, "{:13.5}", 0.0)?;
    writeln!(out, "set_chemical_potential? chemical_potential(eV)")?;
    writeln!(out, " F   0.00000000    ")?;
    writeln!(out, "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?")?;
    writeln!(out, " F F F F F")?;
    writeln!(out, "qhat_x qhat_y qhat_z")?;
    writeln!(
        out,
        "   0.0000000000000000        0.0000000000000000        1.0000000000000000     "
    )?;
    Ok(())
}

/// Render FEFF-compatible `hubbard.inp` content.
#[must_use]
pub fn hubbard_inp_string(document: &FeffDocument) -> String {
    let hubbard = document.hubbard;
    let j_field = if hubbard.j == 0.0 {
        format!("{:26.16}", hubbard.j)
    } else {
        format!("{:26.17}", hubbard.j)
    };
    format!(
        "i_hubbard mldos_hubb U_hubbard J_hubbard fermi_shift l_hubbard\n{:12}{:12}{:21.16}{j_field}{:26.16}{:17}\n",
        hubbard.i_hubbard, hubbard.mldos_hubb, hubbard.u, hubbard.fermi_shift, hubbard.l
    )
}

/// Render FEFF-compatible `opcons.inp` content.
pub fn opcons_inp_string(document: &FeffDocument) -> Result<String> {
    opcons_input_string(&document.opcons_input)
}

/// Render FEFF-compatible default `screen.inp` content.
#[must_use]
pub fn screen_inp_string() -> String {
    concat!(
        " ner          40\n",
        " nei          20\n",
        " maxl           4\n",
        " irrh           1\n",
        " iend           0\n",
        " lfxc           0\n",
        " emin  -40.000000000000000     \n",
        " emax   0.0000000000000000     \n",
        " eimax   2.0000000000000000     \n",
        " ermin   1.0000000000000000E-003\n",
        " rfms   4.0000000000000000     \n",
        " nrptx0         251\n",
        " icore          -1\n",
    )
    .to_string()
}

/// Render FEFF-compatible `screen.inp` content from parsed `SCREEN` cards.
pub fn screen_inp_string_for_document(document: &FeffDocument) -> Result<String> {
    screen_input_string(&document.screen_input)
}

/// Render FEFF-compatible `density.inp` content from a `DENSITY` block.
pub fn density_inp_string(document: &FeffDocument) -> Result<String> {
    if document.density_records.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write density.inp without DENSITY payload rows".to_string(),
        });
    }

    let mut out = String::new();
    for record in &document.density_records {
        writeln!(out, "{record}")?;
    }
    Ok(out)
}

/// Render FEFF-compatible `grid.inp` content from an `EGRID` block.
pub fn grid_inp_string(document: &FeffDocument) -> Result<String> {
    if document.egrid_records.is_empty()
        && !document.active_cards.iter().any(|card| card == "EGRID")
    {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write grid.inp without EGRID payload rows".to_string(),
        });
    }

    let mut out = String::new();
    for record in &document.egrid_records {
        writeln!(out, " {record} ")?;
    }
    Ok(out)
}

fn write_global_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let ipol = if document.eels.enabled || document.nrixs.is_some() {
        1
    } else {
        document.ipol
    };
    let nrixs = document.nrixs.as_ref();
    let xivec = if let Some(nrixs) = nrixs {
        normalize_vector(rotate_into_reference_frame(nrixs.qvec, nrixs.qvec))
    } else if document.eels.enabled {
        document.eels.beam_direction
    } else if vector_norm(document.incidence_vector) > 0.0 {
        normalize_vector(document.incidence_vector)
    } else if document.spin != 0 {
        normalize_vector(document.spin_vector)
    } else {
        document.polarization_vector
    };
    let evec = if nrixs.is_some() {
        [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ]
    } else {
        [0.0; 3]
    };
    let evnorm = if let Some(nrixs) = nrixs {
        [0.0, nrixs.qnorm, 0.0]
    } else if document.eels.enabled {
        [0.0, 1.0, 0.0]
    } else if vector_norm(document.incidence_vector) > 0.0 {
        [0.0, vector_norm(document.incidence_vector), 0.0]
    } else if document.spin != 0 {
        [0.0, 0.0, vector_norm(document.spin_vector)]
    } else {
        [0.0; 3]
    };
    let polarization_tensor = global_polarization_tensor(document, nrixs.is_some())?;
    let le2 = if let Some(nrixs) = nrixs {
        nrixs.lj
    } else if !document.eels.enabled
        && document.ipol == 1
        && vector_norm(document.incidence_vector) == 0.0
    {
        0
    } else {
        document.le2
    };

    writeln!(out, " nabs, iphabs - CFAVERAGE data")?;
    writeln!(
        out,
        "{:8}{:8}{:13.5}",
        document.cfaverage.nabs, document.cfaverage.iphabs, document.cfaverage.rclabs
    )?;
    writeln!(
        out,
        " ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj"
    )?;
    writeln!(
        out,
        "{:5}{:5}{:5}{:12.4}{:12.4}{:5}{:5}{:5}{:5}",
        ipol,
        document.spin,
        le2,
        nrixs
            .map(|nrixs| if nrixs.qaverage { -nrixs.nq } else { nrixs.nq } as f64)
            .unwrap_or(document.ellipticity),
        0.0,
        nrixs.map(|_| 30).unwrap_or(document.l2lp),
        i32::from(nrixs.is_some()),
        nrixs.map(|nrixs| nrixs.ldecmx).unwrap_or(-1),
        nrixs.map(|nrixs| nrixs.lj).unwrap_or(-1)
    )?;
    writeln!(out, "evec\t\t  xivec \t   spvec")?;
    for idx in 0..3 {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}",
            evec[idx], xivec[idx], document.spin_vector[idx]
        )?;
    }
    writeln!(out, " polarization tensor ")?;
    for row in polarization_tensor {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
            row[0], row[1], row[2], row[3], row[4], row[5]
        )?;
    }
    writeln!(out, "evnorm, xivnorm, spvnorm - only used for nrixs")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        evnorm[0], evnorm[1], evnorm[2]
    )?;
    writeln!(out, "nq,    imdff,   qaverage,   mixdff")?;
    let mixdff = global_mixdff(document);
    writeln!(
        out,
        "{:12}{:12} {} {}",
        nrixs.map(|nrixs| nrixs.nq).unwrap_or(0),
        document.mdff.imdff,
        fortran_bool(nrixs.map(|nrixs| nrixs.qaverage).unwrap_or(true)),
        fortran_bool(mixdff)
    )?;
    writeln!(
        out,
        " q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi"
    )?;
    if let Some(nrixs) = nrixs {
        for q_vector in &nrixs.q_vectors {
            let qtrig = nrixs_qtrig(q_vector.vector, q_vector.norm);
            writeln!(
                out,
                "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
                q_vector.vector[0],
                q_vector.vector[1],
                q_vector.vector[2],
                q_vector.norm,
                q_vector.weight[0],
                q_vector.weight[1],
                qtrig[0],
                qtrig[1],
                qtrig[2],
                qtrig[3]
            )?;
        }
    }
    if mixdff {
        let Some(nrixs) = nrixs else {
            return Err(IoError::Parse {
                path: document.source.clone(),
                line: 0,
                message: "MDFF mixdff requires NRIXS".to_string(),
            });
        };
        writeln!(out, "    qqmdff,   cos<q,q'>")?;
        write!(out, "{:22.16}", document.mdff.qqmdff)?;
        for value in global_mdff_cosines(nrixs) {
            write!(out, "{value:22.16}")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

fn global_mixdff(document: &FeffDocument) -> bool {
    document.nrixs.is_some() && matches!(document.mdff.imdff, 1 | 2)
}

fn global_mdff_cosines(nrixs: &crate::model::Nrixs) -> Vec<f64> {
    nrixs
        .q_vectors
        .iter()
        .flat_map(|left| {
            nrixs.q_vectors.iter().map(move |right| {
                let norm_product = left.norm * right.norm;
                let normalized_dot = if norm_product > 0.0 {
                    left.vector
                        .iter()
                        .zip(right.vector)
                        .map(|(left, right)| *left * right)
                        .sum::<f64>()
                        / norm_product
                } else {
                    0.0
                };
                (std::f64::consts::PI / 180.0 * normalized_dot).cos()
            })
        })
        .collect()
}

fn global_polarization_tensor(document: &FeffDocument, has_nrixs: bool) -> Result<[[f64; 6]; 3]> {
    if has_nrixs {
        return Ok([
            [0.5, -0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        ]);
    }
    if document.ipol == 2 {
        return Ok([
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
    }
    if document.eels.enabled {
        return Ok([[0.0; 6]; 3]);
    }
    if document.ipol == 1 {
        return linear_polarization_tensor(
            document.polarization_vector,
            document.incidence_vector,
            document.ellipticity,
        );
    }
    Ok(averaged_polarization_tensor())
}

fn averaged_polarization_tensor() -> [[f64; 6]; 3] {
    [
        [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0],
    ]
}

fn linear_polarization_tensor(
    polarization: [f64; 3],
    incidence: [f64; 3],
    ellipticity: f64,
) -> Result<[[f64; 6]; 3]> {
    let incidence_norm = vector_norm(incidence);
    let rotated_polarization = if incidence_norm > 0.0 {
        rotate_into_reference_frame(polarization, incidence)
    } else {
        polarization
    };
    let rotated_incidence = if incidence_norm > 0.0 {
        normalize_vector(rotate_into_reference_frame(incidence, incidence))
    } else {
        [0.0; 3]
    };

    let mut evec = normalize_checked_polarization(rotated_polarization)?;
    let mut effective_ellipticity = ellipticity;
    if incidence_norm > 0.0 {
        let dot = dot_product(evec, rotated_incidence);
        if dot.abs() > 0.9 {
            return Err(IoError::InvalidPolarizationGeometry { dot });
        }
        if dot.abs() > 0.00001 {
            evec = normalize_checked_polarization([
                evec[0] - dot * rotated_incidence[0],
                evec[1] - dot * rotated_incidence[1],
                evec[2] - dot * rotated_incidence[2],
            ])?;
        }
    } else {
        effective_ellipticity = 0.0;
    }

    let e2 = cross_product(rotated_incidence, evec);
    let positive = [
        Complex64::new(evec[0], effective_ellipticity * e2[0]),
        Complex64::new(evec[1], effective_ellipticity * e2[1]),
        Complex64::new(evec[2], effective_ellipticity * e2[2]),
    ];
    let negative = [
        Complex64::new(evec[0], -effective_ellipticity * e2[0]),
        Complex64::new(evec[1], -effective_ellipticity * e2[1]),
        Complex64::new(evec[2], -effective_ellipticity * e2[2]),
    ];
    let eps = spherical_components(positive);
    let epc = spherical_components(negative);
    let scale = 1.0 / (1.0 + effective_ellipticity * effective_ellipticity) / 2.0;
    let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    for row_magnetic in -1..=1 {
        for column_magnetic in -1..=1 {
            let sign = if column_magnetic % 2 == 0 { 1.0 } else { -1.0 };
            let row = tensor_index(row_magnetic);
            let column = tensor_index(column_magnetic);
            tensor[row][column] = sign
                * (epc[column] * eps[tensor_index(-row_magnetic)]
                    + eps[column] * epc[tensor_index(-row_magnetic)])
                * scale;
        }
    }
    Ok(polarization_rows(tensor))
}

fn normalize_checked_polarization(vector: [f64; 3]) -> Result<[f64; 3]> {
    let norm = vector_norm(vector);
    if norm <= 0.000001 {
        return Err(IoError::InvalidPolarizationVector { norm });
    }
    Ok([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

fn spherical_components(vector: [Complex64; 3]) -> [Complex64; 3] {
    let root_half = 1.0 / 2.0_f64.sqrt();
    let imaginary = Complex64::new(0.0, 1.0);
    [
        (vector[0] - imaginary * vector[1]) * root_half,
        vector[2],
        -(vector[0] + imaginary * vector[1]) * root_half,
    ]
}

fn polarization_rows(tensor: [[Complex64; 3]; 3]) -> [[f64; 6]; 3] {
    let mut rows = [[0.0; 6]; 3];
    for row in 0..3 {
        for column in 0..3 {
            rows[row][2 * column] = tensor[row][column].re;
            rows[row][2 * column + 1] = tensor[row][column].im;
        }
    }
    rows
}

fn tensor_index(magnetic: i32) -> usize {
    match magnetic {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 0,
    }
}

fn dot_product(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross_product(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
