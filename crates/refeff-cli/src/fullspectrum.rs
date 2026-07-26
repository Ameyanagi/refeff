use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use num_complex::Complex64;
use refeff_core::{
    FEFF_BOHR_ANGSTROM, FEFF_FULLSPECTRUM_EDGE_TRANSITION_SIZE,
    FEFF_FULLSPECTRUM_FINE_STRUCTURE_HIGH_K, FEFF_FULLSPECTRUM_FINE_STRUCTURE_LOW_K,
    FEFF_FULLSPECTRUM_GRID_CAPACITY, FEFF_FULLSPECTRUM_XK_STEP, FEFF_HARTREE_EV,
    FullSpectrumBackgroundInput, FullSpectrumDefaultGridEdge, FullSpectrumEdgeAssemblyInput,
    FullSpectrumEdgeGridInput, FullSpectrumFineStructureInput, FullSpectrumKramersKronigInput,
    FullSpectrumScatteringDielectricInput, edge_index, full_spectrum_assemble_edge,
    full_spectrum_background_from_fprime, full_spectrum_default_energy_grid,
    full_spectrum_edge_energy_grid, full_spectrum_elam_edge_energies,
    full_spectrum_fine_structure_from_segments, full_spectrum_kramers_kronig,
    full_spectrum_scattering_to_dielectric, standard_edge_label,
};
use refeff_io::{
    DrudeDatData, EpsDatData, FullSpectrumAutomaticFineStructure,
    FullSpectrumBackgroundSegmentData, FullSpectrumComponentEdgeSource,
    FullSpectrumFineStructureSegmentData, FullSpectrumInput, FullSpectrumOptions, HamakerDatData,
    ModuleLogData, OpconsDatData, OscStrDatData, OscStrRow, XmuDatData, drude_dat_from_grid,
    fullspectrum_background_segment_from_fprime_xmu_dat,
    fullspectrum_imaginary_fine_structure_segment_from_xmu_dat,
    fullspectrum_number_density_from_pot_bin,
    fullspectrum_real_fine_structure_segment_from_xmu_dat,
    opcons_dat_from_fullspectrum_epsilon_minus_one, read_drude_dat, read_eps_dat, read_hamaker_dat,
    read_module_log_dat, read_osc_str_dat, read_pot_bin, read_xmu_dat, sumrules_dat_from_opcons,
    valence_epsilon2_from_xmu_dat, write_drude_dat, write_eps_dat, write_hamaker_dat,
    write_module_log_dat, write_opcons_dat, write_osc_str_dat, write_sumrules_dat, write_xmu_dat,
};

use crate::work_dir_for_input;

const OMEGA_MATCH_TOLERANCE: f64 = 1.0e-10;

/// Run cached FEFF `FULLSPECTRUM` optical-table generation beside an input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether FEFF FULLSPECTRUM has cached dielectric data ready for optical tables.
pub(crate) fn has_cached_optical_inputs(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("fullspectrum.inp");
    if !input_path.is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.m_full_spectrum <= 0 {
        return Ok(false);
    }
    if has_complete_source_layout(work_dir) {
        return Ok(true);
    }
    let Ok(optical_outputs_enabled) = read_optical_outputs_enabled(work_dir) else {
        return Ok(false);
    };
    if !optical_outputs_enabled {
        return Ok(false);
    }
    let eps_path = work_dir.join("eps.dat");
    if !eps_path.is_file() {
        return Ok(false);
    }
    let Ok(eps) = read_eps_dat(&eps_path) else {
        return Ok(false);
    };
    Ok(validate_cached_optical_inputs(work_dir, &eps).is_ok())
}

/// Generate or read `eps.dat`, then write the FULLSPECTRUM optical tables.
///
/// When FEFF edge-source directories and option cards are present, this
/// assembles `eps.dat` from FPRIME, FMS, and path-expansion `xmu.dat` files.
/// A pre-existing `eps.dat` remains a supported restart/cache handoff.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if input.m_full_spectrum <= 0 {
        return Ok(0);
    }
    let optical_outputs_enabled = read_optical_outputs_enabled(work_dir)?;

    let prepared = read_or_generate_eps(work_dir)?;
    let eps = prepared.eps;
    if !optical_outputs_enabled {
        return Ok(0);
    }

    let sumrules_number_density = match prepared.source_number_density {
        Some(number_density) => Some(number_density),
        None => read_optional_sumrules_number_density(work_dir)?,
    };
    let drude = if prepared.generated_from_sources {
        if let Some(drude) = &prepared.drude {
            write_drude_cache(&work_dir.join("drude.dat"), drude)?;
        }
        prepared.drude
    } else {
        read_optional_drude_cache(work_dir, &eps.omega)?
    };
    write_optional_sidecar_caches(work_dir)?;
    let total_epsilon =
        add_optional_drude_epsilon(&eps.epsilon, drude.as_ref().map(|data| &data.epsilon));
    let direct = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM optical constants from eps.dat".to_string()],
        &eps,
        &total_epsilon,
    )?;
    write_opcons(work_dir, "opcons.dat", &direct)?;

    let kk_bound = kramers_kronig_epsilon_minus_one(&eps.omega, &eps.epsilon)
        .context("failed to compute FULLSPECTRUM Kramers-Kronig table")?;
    let kk = add_optional_drude_epsilon(&kk_bound, drude.as_ref().map(|data| &data.epsilon));
    let kk_table = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM Kramers-Kronig optical constants from eps.dat".to_string()],
        &eps,
        &kk,
    )?;
    write_opcons(work_dir, "opconsKK.dat", &kk_table)?;
    write_optional_sumrules(work_dir, &kk_table, sumrules_number_density)?;

    let background_kk_bound = kramers_kronig_epsilon_minus_one(&eps.omega, &eps.background_epsilon)
        .context("failed to compute FULLSPECTRUM background Kramers-Kronig table")?;
    let background_kk = add_optional_drude_epsilon(
        &background_kk_bound,
        drude.as_ref().map(|data| &data.epsilon),
    );
    let background_table = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM atomic-background optical constants from eps.dat".to_string()],
        &eps,
        &background_kk,
    )?;
    write_opcons(work_dir, "opcons0.dat", &background_table)?;

    Ok(eps.point_count())
}

#[derive(Debug)]
struct FullSpectrumSourceComponent {
    name: String,
    atomic_number: i32,
    number_density: Option<f64>,
    edges: Vec<FullSpectrumSourceEdge>,
}

#[derive(Debug)]
struct FullSpectrumSourceEdge {
    label: String,
    fine_structure: bool,
    path: PathBuf,
}

struct PreparedFullSpectrum {
    eps: EpsDatData,
    drude: Option<DrudeDatData>,
    source_number_density: Option<f64>,
    generated_from_sources: bool,
}

fn read_or_generate_eps(work_dir: &Path) -> Result<PreparedFullSpectrum> {
    if work_dir.join("edges").is_dir() {
        return generate_eps_from_sources(work_dir);
    }

    let eps_path = work_dir.join("eps.dat");
    let eps = read_eps_dat(&eps_path)
        .with_context(|| format!("failed to read {}", eps_path.display()))?;
    Ok(PreparedFullSpectrum {
        eps,
        drude: None,
        source_number_density: None,
        generated_from_sources: false,
    })
}

fn has_complete_source_layout(work_dir: &Path) -> bool {
    let Ok((options, components)) = read_source_layout(work_dir) else {
        return false;
    };
    if components.is_empty() {
        return false;
    }

    for component in &components {
        let has_density = component.number_density.is_some()
            || component
                .edges
                .first()
                .is_some_and(|edge| edge.path.join("fms_im").join("pot.bin").is_file());
        if !has_density || component.edges.is_empty() {
            return false;
        }
        for edge in &component.edges {
            if read_background_segments(&edge.path).is_err() {
                return false;
            }
            if edge.fine_structure
                && (read_real_fine_structure_segment(&edge.path, "fms_re").is_err()
                    || read_real_fine_structure_segment(&edge.path, "path_re").is_err()
                    || read_imaginary_fine_structure_segment(&edge.path, "fms_im").is_err()
                    || read_imaginary_fine_structure_segment(&edge.path, "path_im").is_err())
            {
                return false;
            }
        }
        if options.valence
            && !work_dir
                .join("edges")
                .join(&component.name)
                .join("valence")
                .join("xmu.val")
                .is_file()
        {
            return false;
        }
        if options.valence {
            let path = work_dir
                .join("edges")
                .join(&component.name)
                .join("valence")
                .join("xmu.val");
            let Ok(valence) = read_xmu_dat(&path) else {
                return false;
            };
            if valence.normalization.is_none() {
                return false;
            }
        }
        if component.number_density.is_none() && source_component_number_density(component).is_err()
        {
            return false;
        }
    }
    true
}

fn read_source_layout(
    work_dir: &Path,
) -> Result<(FullSpectrumOptions, Vec<FullSpectrumSourceComponent>)> {
    let options = read_options(work_dir)?;

    let mut components = Vec::with_capacity(options.components.len());
    for component in &options.components {
        let component_path = work_dir.join("edges").join(&component.name);
        let edges = match &component.edge_source {
            FullSpectrumComponentEdgeSource::Explicit(edges) => edges
                .iter()
                .map(|edge| FullSpectrumSourceEdge {
                    label: edge.label.clone(),
                    fine_structure: edge.fine_structure,
                    path: component_path.join(&edge.label),
                })
                .collect(),
            FullSpectrumComponentEdgeSource::Automatic { fine_structure } => {
                discover_automatic_edges(&component_path, fine_structure)?
            }
        };
        components.push(FullSpectrumSourceComponent {
            name: component.name.clone(),
            atomic_number: component.atomic_number,
            number_density: component.number_density_bohr3,
            edges,
        });
    }
    Ok((options, components))
}

fn read_options(work_dir: &Path) -> Result<FullSpectrumOptions> {
    let input_path = work_dir.join("fullspectrum.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    FullSpectrumOptions::parse_str(&input_path, &input_text).with_context(|| {
        format!(
            "failed to parse FULLSPECTRUM options in {}",
            input_path.display()
        )
    })
}

fn read_optical_outputs_enabled(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("fullspectrum.inp");
    let mut input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    // Cached eps.dat restarts can carry only the small mFullSpectrum handoff,
    // while the standalone options parser quite properly requires a component
    // for source assembly. A synthetic ignored component lets the same parser
    // validate CONTROL without weakening source-layout validation.
    input_text.push_str("\nCOMPONENT Rfx 1\n");
    let options = FullSpectrumOptions::parse_str(&input_path, &input_text).with_context(|| {
        format!(
            "failed to parse FULLSPECTRUM CONTROL in {}",
            input_path.display()
        )
    })?;
    Ok(options.control[5] == 1)
}

fn discover_automatic_edges(
    component_path: &Path,
    fine_structure: &FullSpectrumAutomaticFineStructure,
) -> Result<Vec<FullSpectrumSourceEdge>> {
    if !component_path.is_dir() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(component_path)
        .with_context(|| format!("failed to read {}", component_path.display()))?;
    let mut edges = Vec::new();
    for entry in entries {
        let entry =
            entry.with_context(|| format!("failed to inspect {}", component_path.display()))?;
        if !entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?
            .is_dir()
        {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(label) = standard_edge_label(&name) else {
            continue;
        };
        let fine = match fine_structure {
            FullSpectrumAutomaticFineStructure::None => false,
            FullSpectrumAutomaticFineStructure::All => true,
            FullSpectrumAutomaticFineStructure::Listed(labels) => {
                labels.iter().any(|candidate| candidate == label)
            }
        };
        edges.push(FullSpectrumSourceEdge {
            label: label.to_string(),
            fine_structure: fine,
            path: entry.path(),
        });
    }
    edges.sort_by_key(|edge| edge_index(&edge.label).unwrap_or(i32::MAX));
    Ok(edges)
}

fn generate_eps_from_sources(work_dir: &Path) -> Result<PreparedFullSpectrum> {
    let (options, mut components) = read_source_layout(work_dir)?;
    bail_if_incomplete_source_layout(work_dir, &options, &components)?;
    let omega = source_energy_grid(&options, &components)?;
    let mut epsilon = Array1::<Complex64>::zeros(omega.len());
    let mut background_epsilon = Array1::<Complex64>::zeros(omega.len());
    let mut sigma = Array1::<f64>::zeros(omega.len());
    let mut oscillator_rows = Vec::new();
    let mut component_densities = Vec::with_capacity(components.len());

    for component in &mut components {
        let number_density = source_component_number_density(component)?;
        component_densities.push(number_density);
        for edge in &component.edges {
            let background_segments = read_background_segments(&edge.path)?;
            let background_inputs = background_segments
                .iter()
                .map(FullSpectrumBackgroundSegmentData::as_core_input)
                .collect::<Vec<_>>();
            let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
                omega: omega.view(),
                segments: &background_inputs,
            })
            .with_context(|| {
                format!(
                    "failed to assemble FPRIME background for {} {}",
                    component.name, edge.label
                )
            })?;

            let (scattering_factor, background_scattering_factor, effective_electron_count) =
                if edge.fine_structure {
                    let fine = read_fine_structure(&edge.path, omega.view())?;
                    let assembled = full_spectrum_assemble_edge(FullSpectrumEdgeAssemblyInput {
                        omega: omega.view(),
                        background: &background,
                        fine_structure: &fine,
                        transition_size: FEFF_FULLSPECTRUM_EDGE_TRANSITION_SIZE,
                    })
                    .with_context(|| {
                        format!(
                            "failed to assemble FULLSPECTRUM edge {} {}",
                            component.name, edge.label
                        )
                    })?;
                    (
                        assembled.scattering_factor,
                        assembled.background,
                        assembled.effective_electron_count,
                    )
                } else {
                    let shift = Complex64::new(background.zero_energy_fprime, 0.0);
                    let background_only = background
                        .scattering_factor
                        .mapv(|value| value.conj() - shift);
                    (
                        background_only.clone(),
                        background_only,
                        background.effective_electron_count,
                    )
                };

            let dielectric =
                full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
                    number_density,
                    omega: omega.view(),
                    scattering_factor: scattering_factor.view(),
                    background_scattering_factor: background_scattering_factor.view(),
                })
                .with_context(|| {
                    format!(
                        "failed to convert FULLSPECTRUM edge {} {} to dielectric response",
                        component.name, edge.label
                    )
                })?;
            epsilon += &dielectric.epsilon_minus_one;
            background_epsilon += &dielectric.background_epsilon_minus_one;
            sigma += &dielectric.sigma;
            oscillator_rows.push(OscStrRow {
                component: component.name.clone(),
                edge: edge.label.clone(),
                core_hole_index: edge_index(&edge.label).unwrap_or(0),
                effective_electron_count,
            });
        }

        if options.valence {
            let valence_path = work_dir
                .join("edges")
                .join(&component.name)
                .join("valence")
                .join("xmu.val");
            let valence = read_xmu_dat(&valence_path)
                .with_context(|| format!("failed to read {}", valence_path.display()))?;
            let valence_epsilon2 =
                valence_epsilon2_from_xmu_dat(number_density, omega.view(), &valence)
                    .with_context(|| {
                        format!(
                            "failed to add FULLSPECTRUM valence response for {}",
                            component.name
                        )
                    })?;
            for (value, addition) in epsilon.iter_mut().zip(valence_epsilon2.iter()) {
                value.im += addition;
            }
        }
    }

    let eps = trim_source_eps(EpsDatData {
        header_lines: vec![
            "# refeff FULLSPECTRUM dielectric response assembled from FEFF edge sources"
                .to_string(),
        ],
        omega,
        epsilon,
        background_epsilon,
        sigma,
    })?;
    let eps_path = work_dir.join("eps.dat");
    write_eps_dat(&eps_path, &eps)
        .with_context(|| format!("failed to write {}", eps_path.display()))?;

    let osc_str = OscStrDatData {
        header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
        rows: oscillator_rows,
    };
    write_osc_str_cache(&work_dir.join("osc_str.dat"), &osc_str)?;
    write_fullspectrum_xmu_cache(&work_dir.join("xmu.dat"), &eps, &osc_str)?;

    let drude = if options.control[5] == 1 {
        options
            .drude
            .map(|settings| {
                let density_per_site = settings.electron_density.unwrap_or(0.0);
                drude_dat_from_grid(
                    eps.omega.view(),
                    settings.lifetime_seconds,
                    density_per_site * component_densities[0],
                )
                .context("failed to generate FULLSPECTRUM Drude response")
            })
            .transpose()?
    } else {
        None
    };

    let source_number_density = component_densities.iter().copied().reduce(f64::min);
    Ok(PreparedFullSpectrum {
        eps,
        drude,
        source_number_density,
        generated_from_sources: true,
    })
}

/// Write the final "fake" `xmu.dat` emitted by FEFF FULLSPECTRUM.
///
/// `fullspectrum.f90` reports one comment row per assembled edge, then writes
/// the trimmed FULLSPECTRUM grid in ordinary xmu units: both energy columns in
/// eV, wave number in inverse Angstrom, and the accumulated cross section in
/// both absorption columns. The typed codec supplies the implied zero `chi`
/// column so the result remains a valid six-column FEFF `xmu.dat`.
fn write_fullspectrum_xmu_cache(
    path: &Path,
    eps: &EpsDatData,
    oscillator_strengths: &OscStrDatData,
) -> Result<()> {
    let mut header_lines = vec!["# component  edge  n_eff".to_string(), " ".to_string()];
    header_lines.extend(oscillator_strengths.rows.iter().map(|row| {
        let component_source = format!("{:<3}", row.component);
        let component = format!("{component_source:>11}");
        let edge_source = format!("{:<2}", row.edge);
        let edge = format!("{edge_source:>6}");
        format!("# {component}{edge}{:>8.3}", row.effective_electron_count)
    }));
    header_lines.extend([
        "#     0/   0 paths used".to_string(),
        "#  xsedge+ 50, used to normalize mu           1.0000E+00".to_string(),
        "# ".to_string(),
        "# ".to_string(),
    ]);

    let photon_energy_ev = eps.omega.mapv(|omega| omega * FEFF_HARTREE_EV);
    let wave_number = eps
        .omega
        .mapv(|omega| (2.0 * omega).sqrt() / FEFF_BOHR_ANGSTROM);
    let data = XmuDatData {
        header_lines,
        normalization: Some(1.0),
        relative_energy_ev: photon_energy_ev.clone(),
        photon_energy_ev,
        wave_number,
        mu: eps.sigma.clone(),
        mu0: eps.sigma.clone(),
        chi: Array1::zeros(eps.point_count()),
    };
    write_xmu_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

fn bail_if_incomplete_source_layout(
    work_dir: &Path,
    options: &FullSpectrumOptions,
    components: &[FullSpectrumSourceComponent],
) -> Result<()> {
    if components.is_empty() {
        bail!("FULLSPECTRUM source generation requires at least one component");
    }
    for component in components {
        if component.edges.is_empty() {
            bail!(
                "FULLSPECTRUM component {} has no selected edge source directories",
                component.name
            );
        }
        if component.number_density.is_none()
            && !component.edges[0]
                .path
                .join("fms_im")
                .join("pot.bin")
                .is_file()
        {
            bail!(
                "FULLSPECTRUM component {} needs an explicit number density or {}",
                component.name,
                component.edges[0]
                    .path
                    .join("fms_im")
                    .join("pot.bin")
                    .display()
            );
        }
        for edge in &component.edges {
            let fprime = edge.path.join("fprime1").join("xmu.dat");
            if !fprime.is_file() {
                bail!("missing FULLSPECTRUM FPRIME source {}", fprime.display());
            }
            if edge.fine_structure {
                for branch in ["fms_re", "path_re", "fms_im", "path_im"] {
                    let path = edge.path.join(branch).join("xmu.dat");
                    if !path.is_file() {
                        bail!(
                            "missing FULLSPECTRUM fine-structure source {}",
                            path.display()
                        );
                    }
                }
            }
        }
        if options.valence {
            let path = work_dir
                .join("edges")
                .join(&component.name)
                .join("valence")
                .join("xmu.val");
            if !path.is_file() {
                bail!("missing FULLSPECTRUM valence source {}", path.display());
            }
        }
    }
    Ok(())
}

fn source_energy_grid(
    options: &FullSpectrumOptions,
    components: &[FullSpectrumSourceComponent],
) -> Result<Array1<f64>> {
    let (min_energy, max_energy) = match (
        options.energy_grid.min_hartree,
        options.energy_grid.max_hartree,
    ) {
        (Some(min), Some(max)) => (min, max),
        _ => {
            let edges = components
                .iter()
                .flat_map(|component| {
                    component.edges.iter().filter_map(|edge| {
                        edge_index(&edge.label).map(|hole_index| FullSpectrumDefaultGridEdge {
                            atomic_number: component.atomic_number,
                            hole_index,
                            fine_structure: edge.fine_structure,
                        })
                    })
                })
                .collect::<Vec<_>>();
            let defaults = full_spectrum_default_energy_grid(&edges)
                .context("failed to infer FULLSPECTRUM energy-grid bounds from selected edges")?;
            (defaults.min_energy, defaults.max_energy)
        }
    };
    let atomic_numbers = components
        .iter()
        .map(|component| component.atomic_number)
        .collect::<Vec<_>>();
    let edge_energies = full_spectrum_elam_edge_energies(&atomic_numbers)
        .context("failed to build FULLSPECTRUM Elam edge table")?;
    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        max_points: FEFF_FULLSPECTRUM_GRID_CAPACITY,
        min_energy,
        max_energy,
        wave_number_step: FEFF_FULLSPECTRUM_XK_STEP,
        edge_energies: edge_energies.view(),
    })
    .context("failed to generate FULLSPECTRUM edge-aware energy grid")?;
    if grid.clipped {
        bail!(
            "FULLSPECTRUM energy grid exhausted its {}-point capacity before reaching {max_energy} Hartree",
            FEFF_FULLSPECTRUM_GRID_CAPACITY
        );
    }
    Ok(grid.energy)
}

fn source_component_number_density(component: &FullSpectrumSourceComponent) -> Result<f64> {
    if let Some(number_density) = component.number_density {
        return Ok(number_density);
    }
    let path = component.edges[0].path.join("fms_im").join("pot.bin");
    let pot = read_pot_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    fullspectrum_number_density_from_pot_bin(component.atomic_number as usize, &pot).with_context(
        || {
            format!(
                "failed to estimate FULLSPECTRUM number density for component {}",
                component.name
            )
        },
    )
}

fn read_background_segments(edge_path: &Path) -> Result<Vec<FullSpectrumBackgroundSegmentData>> {
    let mut segments = Vec::new();
    for index in 1.. {
        let path = edge_path.join(format!("fprime{index}")).join("xmu.dat");
        if !path.is_file() {
            break;
        }
        let xmu =
            read_xmu_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        segments.push(
            fullspectrum_background_segment_from_fprime_xmu_dat(&xmu)
                .with_context(|| format!("failed to adapt {}", path.display()))?,
        );
    }
    if segments.is_empty() {
        bail!(
            "FULLSPECTRUM edge source {} has no contiguous fprimeN/xmu.dat segments",
            edge_path.display()
        );
    }
    Ok(segments)
}

fn read_fine_structure(
    edge_path: &Path,
    omega: ndarray::ArrayView1<'_, f64>,
) -> Result<refeff_core::FullSpectrumFineStructure> {
    let real_fms = read_real_fine_structure_segment(edge_path, "fms_re")?;
    let real_path = read_real_fine_structure_segment(edge_path, "path_re")?;
    let imaginary_fms = read_imaginary_fine_structure_segment(edge_path, "fms_im")?;
    let imaginary_path = read_imaginary_fine_structure_segment(edge_path, "path_im")?;
    full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
        omega,
        real_fms: real_fms.as_core_input(),
        real_path: real_path.as_core_input(),
        imaginary_fms: imaginary_fms.as_core_input(),
        imaginary_path: imaginary_path.as_core_input(),
        low_wave_number: FEFF_FULLSPECTRUM_FINE_STRUCTURE_LOW_K,
        high_wave_number: FEFF_FULLSPECTRUM_FINE_STRUCTURE_HIGH_K,
    })
    .context("failed to interpolate FULLSPECTRUM FMS/path fine structure")
}

fn read_real_fine_structure_segment(
    edge_path: &Path,
    branch: &str,
) -> Result<FullSpectrumFineStructureSegmentData> {
    let path = edge_path.join(branch).join("xmu.dat");
    let xmu = read_xmu_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    fullspectrum_real_fine_structure_segment_from_xmu_dat(&xmu)
        .with_context(|| format!("failed to adapt {}", path.display()))
}

fn read_imaginary_fine_structure_segment(
    edge_path: &Path,
    branch: &str,
) -> Result<FullSpectrumFineStructureSegmentData> {
    let path = edge_path.join(branch).join("xmu.dat");
    let xmu = read_xmu_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    fullspectrum_imaginary_fine_structure_segment_from_xmu_dat(&xmu)
        .with_context(|| format!("failed to adapt {}", path.display()))
}

fn trim_source_eps(data: EpsDatData) -> Result<EpsDatData> {
    if data.point_count() < 3 {
        bail!(
            "FULLSPECTRUM source energy grid has {} point(s), expected at least 3",
            data.point_count()
        );
    }
    let take = data.point_count() - 2;
    Ok(EpsDatData {
        header_lines: data.header_lines,
        omega: data.omega.iter().copied().skip(1).take(take).collect(),
        epsilon: data.epsilon.iter().copied().skip(1).take(take).collect(),
        background_epsilon: data
            .background_epsilon
            .iter()
            .copied()
            .skip(1)
            .take(take)
            .collect(),
        sigma: data.sigma.iter().copied().skip(1).take(take).collect(),
    })
}

fn write_optional_sumrules(
    work_dir: &Path,
    opcons_kk: &OpconsDatData,
    number_density: Option<f64>,
) -> Result<()> {
    let Some(number_density) = number_density else {
        return Ok(());
    };
    let sumrules = sumrules_dat_from_opcons(number_density, opcons_kk)
        .context("failed to compute FULLSPECTRUM sum rules")?;
    let path = work_dir.join("sumrules.dat");
    write_sumrules_dat(&path, &sumrules)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn read_optional_sumrules_number_density(work_dir: &Path) -> Result<Option<f64>> {
    let path = work_dir.join("pot.bin");
    if !path.is_file() {
        return Ok(None);
    }
    let pot = read_pot_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut atomic_numbers = Vec::new();
    for atomic_number in pot.atomic_numbers.iter().copied() {
        if !atomic_numbers.contains(&atomic_number) {
            atomic_numbers.push(atomic_number);
        }
    }
    let density = atomic_numbers
        .into_iter()
        .map(|atomic_number| fullspectrum_number_density_from_pot_bin(atomic_number, &pot))
        .collect::<refeff_io::Result<Vec<_>>>()
        .context("failed to estimate FULLSPECTRUM sum-rule number densities")?
        .into_iter()
        .filter(|density| *density > 0.0)
        .reduce(f64::min)
        .context("pot.bin did not provide a positive FULLSPECTRUM number density")?;
    Ok(Some(density))
}

fn read_optional_drude_cache(work_dir: &Path, omega: &Array1<f64>) -> Result<Option<DrudeDatData>> {
    let Some(drude) = read_optional_drude_input(work_dir, omega)? else {
        return Ok(None);
    };
    write_drude_cache(&work_dir.join("drude.dat"), &drude)?;
    Ok(Some(drude))
}

fn validate_cached_optical_inputs(work_dir: &Path, eps: &EpsDatData) -> Result<()> {
    read_optional_drude_input(work_dir, &eps.omega)?;
    validate_optional_sidecar_inputs(work_dir)?;
    read_optional_sumrules_number_density(work_dir)?;
    Ok(())
}

fn read_optional_drude_input(work_dir: &Path, omega: &Array1<f64>) -> Result<Option<DrudeDatData>> {
    let path = work_dir.join("drude.dat");
    if !path.is_file() {
        return Ok(None);
    }
    let drude =
        read_drude_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    validate_drude_grid(&drude, omega)?;
    Ok(Some(drude))
}

fn write_optional_sidecar_caches(work_dir: &Path) -> Result<()> {
    write_optional_osc_str_cache(&work_dir.join("osc_str.dat"))?;
    write_optional_hamaker_cache(&work_dir.join("hamaker.dat"))?;
    write_optional_module_log(&work_dir.join("logfullspectrum.dat"))?;
    Ok(())
}

fn validate_optional_sidecar_inputs(work_dir: &Path) -> Result<()> {
    validate_optional_osc_str_input(&work_dir.join("osc_str.dat"))?;
    validate_optional_hamaker_input(&work_dir.join("hamaker.dat"))?;
    validate_optional_module_log_input(&work_dir.join("logfullspectrum.dat"))?;
    Ok(())
}

fn write_optional_osc_str_cache(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_osc_str_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_osc_str_cache(path, &data)
}

fn validate_optional_osc_str_input(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    read_osc_str_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(())
}

fn write_optional_hamaker_cache(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_hamaker_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_hamaker_cache(path, &data)
}

fn validate_optional_hamaker_input(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    read_hamaker_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(())
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
}

fn validate_optional_module_log_input(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(())
}

fn validate_drude_grid(drude: &DrudeDatData, omega: &Array1<f64>) -> Result<()> {
    if drude.point_count() != omega.len() {
        bail!(
            "drude.dat has {} row(s) but eps.dat has {} row(s)",
            drude.point_count(),
            omega.len()
        );
    }
    for (index, (&drude_omega, &eps_omega)) in drude.omega.iter().zip(omega.iter()).enumerate() {
        let scale = drude_omega.abs().max(eps_omega.abs()).max(1.0);
        if (drude_omega - eps_omega).abs() > OMEGA_MATCH_TOLERANCE * scale {
            bail!(
                "drude.dat omega row {} ({drude_omega}) does not match eps.dat omega ({eps_omega})",
                index + 1
            );
        }
    }
    Ok(())
}

fn add_optional_drude_epsilon(
    bound: &Array1<Complex64>,
    drude: Option<&Array1<Complex64>>,
) -> Array1<Complex64> {
    drude.map_or_else(
        || bound.clone(),
        |drude| {
            Array1::from_iter(
                bound
                    .iter()
                    .zip(drude.iter())
                    .map(|(&bound, &free)| bound + free),
            )
        },
    )
}

fn read_input(work_dir: &Path) -> Result<FullSpectrumInput> {
    let input_path = work_dir.join("fullspectrum.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    FullSpectrumInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn opcons_from_eps(
    header_lines: Vec<String>,
    eps: &EpsDatData,
    epsilon_minus_one: &Array1<Complex64>,
) -> Result<OpconsDatData> {
    opcons_dat_from_fullspectrum_epsilon_minus_one(
        header_lines,
        eps.omega.view(),
        epsilon_minus_one.view(),
    )
    .context("failed to compute FULLSPECTRUM optical constants")
}

fn kramers_kronig_epsilon_minus_one(
    omega: &Array1<f64>,
    epsilon_minus_one: &Array1<Complex64>,
) -> Result<Array1<Complex64>> {
    let epsilon2 = epsilon_minus_one.mapv(|value| value.im);
    let epsilon1 = full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
        omega: omega.view(),
        epsilon2: epsilon2.view(),
    })?;
    Ok(Array1::from_iter(epsilon1.iter().zip(epsilon2.iter()).map(
        |(&real, &imaginary)| Complex64::new(real, imaginary),
    )))
}

fn write_opcons(work_dir: &Path, file_name: &str, data: &OpconsDatData) -> Result<()> {
    let path = work_dir.join(file_name);
    write_opcons_dat(&path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_drude_cache(path: &Path, data: &DrudeDatData) -> Result<()> {
    write_drude_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_osc_str_cache(path: &Path, data: &OscStrDatData) -> Result<()> {
    write_osc_str_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_hamaker_cache(path: &Path, data: &HamakerDatData) -> Result<()> {
    write_hamaker_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests;
