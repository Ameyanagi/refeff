//! Typed extraction of common FEFF input cards.
//!
//! This layer intentionally starts with stable structural cards and grows as
//! each FEFF module is ported. Unknown or module-specific cards remain
//! available in [`crate::FeffInput`] so no information is lost.

use std::path::{Component, Path, PathBuf};

use crate::cif::{
    CifCluster, CifEquivalence, expand_cif_cluster_with_equivalence,
    expand_cif_structure_with_equivalence, read_cif,
};
use crate::control_input::{
    BandEnergyMesh, BandInput, DensityInput, FullSpectrumInput, OpconsInput, ReciprocalCell,
    ReciprocalInput, ReciprocalKMesh,
};
use crate::dym::parse_dym;
use crate::error::{IoError, Result};
use crate::global_input::CfAverage;
use crate::grid_input::parse_grid_inp;
use crate::input::{FeffInput, FeffLine, LineKind};
use crate::screen_input::ScreenInput;
use crate::sfconv_input::{SfconvControl, SfconvInput, SfconvSpectrum, SfconvWindow};
use crate::spring_input::parse_spring_inp;
use crate::xsph_input::XsphAdvanced;

mod cards;
mod controls;
mod parse;
mod spectrum;
mod structure;
mod types;
mod validation;

use cards::*;
use controls::*;
use parse::*;
use spectrum::*;
use structure::*;
use validation::*;

pub use types::*;

impl FeffDocument {
    /// Extract the currently supported typed card subset from parsed input.
    pub fn from_input(input: &FeffInput) -> Result<Self> {
        let active_cards = parse_active_cards(input);
        let input_cards = parse_input_cards(input);
        validate_feff_consistency(input, &active_cards)?;
        let titles = parse_titles(input)?;
        let edge = parse_edge(input)?;
        let (hole, hole_s02) = parse_hole(input)?;
        let s02 = parse_scalar_card(input, "S02")?.or(hole_s02);
        let corrections = parse_corrections(input)?;
        let chsh_type = parse_chsh_type(input)?;
        let xsph_handoff = parse_xsph_handoff(input)?;
        let xsph_advanced = parse_xsph_advanced(input)?;
        let (mut cfaverage, cfaverage_requested) = parse_cfaverage(input)?;
        let corval_emin = parse_corval_emin(input)?;
        let control = parse_i32_6(input, "CONTROL")?;
        let print = parse_i32_6(input, "PRINT")?;
        let mut scf = parse_scf(input)?;
        let exchange = parse_exchange(input)?;
        let exafs = parse_exafs(input)?;
        let ispec = parse_ispec(input);
        let ipol = parse_ipol(input);
        let (le2, l2lp) = parse_multipole(input)?;
        let (ellipticity, incidence_vector) = parse_ellipticity(input)?;
        let polarization_vector = parse_polarization_vector(input)?;
        let (spin, spin_vector) = parse_spin(input)?;
        let spectrum_grid = parse_spectrum_grid(input, exchange.as_ref(), ispec)?;
        let reciprocal = parse_reciprocal_space(input);
        let cif_equivalence = parse_cif_equivalence(input)?;
        let coordinate_mode = parse_coordinate_mode(input)?;
        let band_input = parse_band_input(input)?;
        let full_spectrum_input = parse_full_spectrum_input(&active_cards);
        let screen_input = parse_screen_input(input)?;
        let i_grid = i32::from(card_by_feff_name(input, "EGRID").is_some());
        let egrid_records = parse_egrid_records(input)?;
        let density_records = parse_density_records(input)?;
        let (electronic_temperature, iscfxc) = parse_temp(input)?;
        let rgrid = parse_scalar_card(input, "RGRID")?.unwrap_or(0.05);
        let (critcw, critpw) = parse_criteria(input)?;
        let (pcritk, pcrith) = parse_pcriteria(input)?;
        let lreal = parse_lreal(input);
        let iorder = parse_iorder(input)?;
        let nstar = active_cards.iter().any(|card| card == "NSTAR") && ipol == 1;
        let (i_plsmn, n_poles) = parse_mpse(input)?;
        let opcons = active_cards.iter().any(|card| card == "OPCONS");
        let many_body_convolution = active_cards.iter().any(|card| card == "MBCONV");
        let fine_structure_damping = parse_fine_structure_damping(input)?;
        parse_strfac(input)?;
        let unfreezef = card_by_feff_name(input, "UNFREEZEF").is_some();
        let external_pot = active_cards.iter().any(|card| card == "EXTPOT");
        let restart_from_pot_bin = active_cards.iter().any(|card| card == "RESTART");
        let config_type = parse_config_type(input)?;
        let config_records = parse_config_records(input)?;
        let warn_ion = active_cards.iter().any(|card| card == "WARN");
        let finite_nucleus = active_cards.iter().any(|card| card == "HIGHZ");
        let scf_thermal = parse_scf_thermal(input)?;
        let scf_ramp = parse_scf_ramp(input)?;
        let scf_tolerances = parse_scf_tolerances(input)?;
        let nohole = parse_nohole(input)?;
        let jump_removal = active_cards.iter().any(|card| card == "JUMPRM");
        let absolute = card_by_feff_name(input, "ABSOLUTE").is_some();
        let mut fms = parse_fms(input)?;
        let crpa = parse_crpa(input)?;
        let compton = parse_compton(input)?;
        let hubbard = parse_hubbard(input)?;
        let eels = parse_eels(input)?;
        let rixs = parse_rixs(input)?;
        let mut nrixs = parse_nrixs(input)?;
        let mdff = parse_mdff(input, &mut nrixs, &eels)?;
        let nohole = if compton.do_compton || compton.do_rhozzp {
            0
        } else {
            nohole
        };
        let debye = parse_debye(input)?;
        let spring_input_text = parse_spring_input_text(input, debye.as_ref())?;
        let dym_input = parse_dym_input(input, debye.as_ref())?;
        let mut rpath = parse_rpath(input)?;
        let mut overlap_shells = parse_overlap_shells(input)?;
        let mut single_scattering_paths = parse_single_scattering_paths(input)?;
        if !single_scattering_paths.is_empty() && card_by_feff_name(input, "OVERLAP").is_none() {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "SS cards require an OVERLAP card".to_string(),
            });
        }
        if !single_scattering_paths.is_empty() && overlap_shells.is_empty() {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "SS cards require OVERLAP shell rows".to_string(),
            });
        }
        let cif_cluster_radius = cif_cluster_radius(scf.as_ref(), fms.as_ref(), rpath);
        let nleg = parse_nleg(input)?;
        let path_symmetry = parse_path_symmetry(input)?;
        let no_geom = active_cards.iter().any(|card| card == "NOGEOM");
        let r_multiplier = parse_scalar_card(input, "RMULT")?.unwrap_or(1.0);
        if r_multiplier != 1.0 {
            if let Some(scf) = &mut scf {
                scf.radius *= r_multiplier;
            }
            if let Some(fms) = &mut fms {
                fms.radius *= r_multiplier;
            }
            if let Some(rpath) = &mut rpath {
                *rpath *= r_multiplier;
            }
            for shell in &mut overlap_shells {
                shell.distance *= r_multiplier;
            }
            for path in &mut single_scattering_paths {
                path.distance *= r_multiplier;
            }
        }
        let dims = parse_dims(input)?;
        let ldos = parse_ldos(input)?;
        let interstitial = parse_interstitial(input)?;
        let afolp = parse_afolp(input)?;
        let overlap_factors = parse_overlap_factors(input)?;
        let ionizations = parse_ionizations(input)?;
        let mut potentials = parse_potentials(input)?;
        let input_atoms = parse_atoms(input)?;
        let mut atoms = input_atoms.clone();
        let cif_cluster = parse_cif_cluster(
            input,
            cif_cluster_radius,
            potentials.is_empty() || atoms.is_empty(),
            cif_equivalence,
        )?;
        if potentials.is_empty() {
            potentials = cif_cluster
                .as_ref()
                .map(cif_cluster_potentials)
                .transpose()?
                .unwrap_or_default();
        }
        if atoms.is_empty() {
            atoms = cif_cluster
                .as_ref()
                .map(cif_cluster_atoms)
                .unwrap_or_default();
        } else if let Some(lattice_atoms) =
            parse_lattice_cluster_atoms(input, &input_atoms, cif_cluster_radius, reciprocal)?
        {
            atoms = lattice_atoms;
        }
        let atoms: Vec<Atom> = atoms
            .into_iter()
            .map(|mut atom| {
                atom.x *= r_multiplier;
                atom.y *= r_multiplier;
                atom.z *= r_multiplier;
                atom.distance = atom.distance.map(|distance| distance * r_multiplier);
                atom
            })
            .collect();
        if !overlap_shells.is_empty() && !atoms.is_empty() {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "cannot use ATOMS and OVERLAP in the same input".to_string(),
            });
        }
        if overlap_shells
            .iter()
            .any(|shell| shell.potential_index == 0)
        {
            cfaverage = default_cfaverage();
        } else {
            cfaverage = effective_cfaverage(cfaverage, cfaverage_requested, &atoms);
        }
        ensure_cfaverage_absorber_potential(input, &mut potentials, cfaverage)?;
        let reciprocal_input =
            parse_reciprocal_input(input, nohole, &input_atoms, reciprocal, cif_equivalence)?;
        let opcons_input = parse_opcons_input(input, opcons, &potentials)?;
        let sfconv_input = parse_sfconv_input(input, ispec, ipol, spin, print)?;
        let sfconv = sfconv_input.control.msfconv != 0;

        Ok(Self {
            source: input.source.clone(),
            active_cards,
            input_cards,
            titles,
            edge,
            hole,
            s02,
            corrections,
            chsh_type,
            xsph_handoff,
            xsph_advanced,
            cfaverage,
            corval_emin,
            control,
            print,
            scf,
            exchange,
            exafs,
            spectrum_grid,
            reciprocal,
            cif_equivalence,
            coordinate_mode,
            reciprocal_input,
            band_input,
            full_spectrum_input,
            screen_input,
            i_grid,
            egrid_records,
            density_records,
            electronic_temperature,
            iscfxc,
            rgrid,
            critcw,
            critpw,
            pcritk,
            pcrith,
            lreal,
            iorder,
            nstar,
            i_plsmn,
            n_poles,
            opcons,
            opcons_input,
            sfconv,
            sfconv_input,
            many_body_convolution,
            fine_structure_damping,
            unfreezef,
            external_pot,
            restart_from_pot_bin,
            config_type,
            config_records,
            warn_ion,
            finite_nucleus,
            scf_thermal,
            scf_ramp,
            scf_tolerances,
            nohole,
            jump_removal,
            ispec,
            ipol,
            le2,
            l2lp,
            ellipticity,
            polarization_vector,
            incidence_vector,
            spin,
            spin_vector,
            absolute,
            fms,
            crpa,
            compton,
            hubbard,
            eels,
            rixs,
            nrixs,
            mdff,
            debye,
            spring_input_text,
            dym_input,
            rpath,
            nleg,
            path_symmetry,
            no_geom,
            r_multiplier,
            dims,
            ldos,
            interstitial,
            afolp,
            overlap_factors,
            ionizations,
            overlap_shells,
            single_scattering_paths,
            potentials,
            atoms,
        })
    }
}

fn parse_edge(input: &FeffInput) -> Result<Option<Edge>> {
    let Some(line) = card_by_feff_name(input, "EDGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let label = args.first().map_or("K", String::as_str);
    Ok(Some(Edge {
        label: label.to_ascii_uppercase(),
    }))
}

fn parse_hole(input: &FeffInput) -> Result<(Option<i32>, Option<f64>)> {
    let Some(line) = card_by_feff_name(input, "HOLE") else {
        return Ok((None, None));
    };
    let args = card_args(line)?;
    let Some(ihole) = args.first() else {
        return Err(parse_error(line, "HOLE requires ihole"));
    };
    let ihole = parse_i32(line, ihole)?;
    if ihole <= 0 {
        return Err(parse_error(line, "HOLE ihole must be positive"));
    }
    Ok((Some(ihole), parse_optional_f64(line, args.get(1))?))
}

fn parse_scalar_card(input: &FeffInput, keyword: &str) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(
            line,
            format!("{keyword} requires a numeric value"),
        ));
    };
    Ok(Some(parse_f64(line, value)?))
}

fn parse_optional_i32_card_by_feff_name(
    input: &FeffInput,
    canonical: &str,
    _label: &str,
) -> Result<Option<i32>> {
    let Some(line) = card_by_feff_name(input, canonical) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    parse_optional_i32(line, args.first())
}

fn parse_optional_f64_card_by_feff_name(
    input: &FeffInput,
    canonical: &str,
    label: &str,
) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, canonical) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = parse_optional_f64(line, args.first())? else {
        return Ok(None);
    };
    if !value.is_finite() {
        return Err(parse_error(line, format!("{label} value must be finite")));
    }
    Ok(Some(value))
}

fn parse_single_scattering_paths(input: &FeffInput) -> Result<Vec<SingleScatteringPath>> {
    let mut paths = Vec::new();
    for line in input.cards() {
        if let LineKind::Card { keyword, .. } = &line.kind
            && keyword == "SS"
        {
            let args = card_args(line)?;
            if args.len() < 4 {
                return Err(parse_error(
                    line,
                    "SS requires index, ipot, degeneracy, and rss",
                ));
            }
            let degeneracy = parse_f64(line, &args[2])?;
            let distance = parse_f64(line, &args[3])?;
            if !degeneracy.is_finite() || !distance.is_finite() {
                return Err(parse_error(
                    line,
                    "SS degeneracy and distance must be finite",
                ));
            }
            paths.push(SingleScatteringPath {
                index: parse_i32(line, &args[0])?,
                potential_index: parse_i32(line, &args[1])?,
                degeneracy,
                distance,
            });
        }
    }
    Ok(paths)
}

fn parse_overlap_shells(input: &FeffInput) -> Result<Vec<OverlapShell>> {
    let mut shells = Vec::new();
    let mut current_potential_index = None;
    for line in &input.lines {
        match &line.kind {
            LineKind::Card { keyword, args, .. } if keyword == "OVERLAP" => {
                let Some(value) = args.first() else {
                    return Err(parse_error(line, "OVERLAP requires a potential index"));
                };
                current_potential_index = Some(parse_i32(line, value)?);
            }
            LineKind::SectionData { section, fields } if section == "OVERLAP" => {
                let Some(potential_index) = current_potential_index else {
                    return Err(parse_error(line, "OVERLAP row without an OVERLAP card"));
                };
                if fields.len() < 3 {
                    return Err(parse_error(
                        line,
                        "OVERLAP rows require iphovr, nnovr, and rovr",
                    ));
                }
                let distance = parse_f64(line, &fields[2])?;
                if !distance.is_finite() {
                    return Err(parse_error(line, "OVERLAP distance must be finite"));
                }
                shells.push(OverlapShell {
                    potential_index,
                    neighbor_potential_index: parse_i32(line, &fields[0])?,
                    count: parse_i32(line, &fields[1])?,
                    distance,
                });
            }
            LineKind::Card { .. } | LineKind::SectionData { .. } => {}
        }
    }
    Ok(shells)
}

fn parse_corrections(input: &FeffInput) -> Result<[f64; 2]> {
    let Some(line) = card_by_feff_name(input, "CORRECTIONS") else {
        return Ok([0.0, 0.0]);
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(
            line,
            "CORRECTIONS requires real and imaginary shifts",
        ));
    }
    Ok([parse_f64(line, &args[0])?, parse_f64(line, &args[1])?])
}

fn parse_chsh_type(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "CHSHIFT") else {
        return Ok(0);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "CHSHIFT requires ChSh_Type"));
    };
    parse_i32(line, value)
}

fn parse_xsph_handoff(input: &FeffInput) -> Result<XsphHandoffControls> {
    Ok(XsphHandoffControls {
        core_hole_broadening: parse_optional_i32_card_by_feff_name(
            input,
            "CHBROADENING",
            "CHBROADENING",
        )?
        .unwrap_or(0),
        core_state: parse_optional_i32_card_by_feff_name(input, "ICOR", "ICORE")?.unwrap_or(-1),
        eps0: parse_optional_f64_card_by_feff_name(input, "EPS0", "EPS0")?.unwrap_or(0.0),
        egap: parse_optional_f64_card_by_feff_name(input, "EGAP", "EGAP")?.unwrap_or(0.0),
        core_hole_width: parse_optional_f64_card_by_feff_name(input, "CHWIDTH", "CHWIDTH")?,
        set_edge: card_by_feff_name(input, "SETE").is_some(),
        print_radial_wavefunctions: card_by_feff_name(input, "RLPR").is_some(),
    })
}

fn parse_xsph_advanced(input: &FeffInput) -> Result<XsphAdvanced> {
    let mut advanced = XsphAdvanced {
        izstd: 0,
        ifxc: 0,
        ipmbse: 0,
        itdlda: 0,
        nonlocal: 0,
        ibasis: 0,
    };

    for line in input.cards() {
        let LineKind::Card { keyword, args, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("TDLDA") => {
                advanced.izstd = 1;
                if let Some(value) = args.first() {
                    advanced.ifxc = parse_i32(line, value)?;
                }
            }
            Some("PMBSE") => {
                advanced.itdlda = 2;
                if let Some(value) = args.first() {
                    advanced.ipmbse = parse_i32(line, value)?;
                }
                if let Some(value) = args.get(1) {
                    advanced.nonlocal = parse_i32(line, value)?;
                }
                if let Some(value) = args.get(2)
                    && advanced.izstd == 0
                {
                    advanced.ifxc = parse_i32(line, value)?;
                }
                if let Some(value) = args.get(3) {
                    advanced.ibasis = parse_i32(line, value)?;
                }
            }
            _ => {}
        }
    }

    Ok(advanced)
}

fn parse_cfaverage(input: &FeffInput) -> Result<(CfAverage, bool)> {
    let mut cfaverage = default_cfaverage();
    let mut found = false;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        if feff_card_token(keyword).map(|(_, display)| display) != Some("CFAVERAGE") {
            continue;
        }

        found = true;
        let args = card_args(line)?;
        if args.len() < 3 {
            return Err(parse_error(line, "CFAVERAGE requires iphabs nabs rclabs"));
        }
        let mut rclabs = parse_f64(line, &args[2])?;
        if !rclabs.is_finite() {
            return Err(parse_error(line, "CFAVERAGE rclabs must be finite"));
        }
        if rclabs < 0.5 {
            rclabs = default_cfaverage().rclabs;
        }
        cfaverage = CfAverage {
            iphabs: parse_i32(line, &args[0])?,
            nabs: parse_i32(line, &args[1])?,
            rclabs,
        };
    }

    Ok((cfaverage, found))
}

fn default_cfaverage() -> CfAverage {
    CfAverage {
        nabs: 1,
        iphabs: 0,
        rclabs: 100000.0,
    }
}

fn effective_cfaverage(
    mut cfaverage: CfAverage,
    cfaverage_requested: bool,
    atoms: &[Atom],
) -> CfAverage {
    if !cfaverage_requested {
        return cfaverage;
    }

    let absorber_count = atoms
        .iter()
        .filter(|atom| atom.ipot == cfaverage.iphabs || atom.ipot == 0)
        .count() as i32;
    if absorber_count > 0 && (cfaverage.nabs <= 0 || cfaverage.nabs > absorber_count) {
        cfaverage.nabs = absorber_count;
    }
    cfaverage
}

fn ensure_cfaverage_absorber_potential(
    input: &FeffInput,
    potentials: &mut Vec<Potential>,
    cfaverage: CfAverage,
) -> Result<()> {
    if cfaverage.iphabs <= 0 || potentials.iter().any(|potential| potential.ipot == 0) {
        return Ok(());
    }

    let Some(mut absorber) = potentials
        .iter()
        .find(|potential| potential.ipot == cfaverage.iphabs)
        .cloned()
    else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!(
                "CFAVERAGE absorber potential {} is missing",
                cfaverage.iphabs
            ),
        });
    };
    absorber.ipot = 0;
    potentials.push(absorber);
    potentials.sort_by_key(|potential| potential.ipot);
    Ok(())
}

fn parse_opcons_input(
    input: &FeffInput,
    run_opcons: bool,
    potentials: &[Potential],
) -> Result<OpconsInput> {
    let mut number_densities = vec![-1.0; opcons_density_count(potentials)];
    let mut print_eps = false;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("NUMD") => {
                let args = card_args(line)?;
                if args.is_empty() {
                    if let Some(density) = number_densities.first_mut() {
                        *density = 0.0;
                    }
                    continue;
                }
                if args.len() < 2 {
                    return Err(parse_error(line, "NUMDENS requires ipot and numdens"));
                }
                let ipot = parse_i32(line, &args[0])?;
                if ipot < 0 {
                    return Err(parse_error(line, "NUMDENS ipot must be non-negative"));
                }
                let density = parse_f64(line, &args[1])?;
                if !density.is_finite() {
                    return Err(parse_error(line, "NUMDENS value must be finite"));
                }
                let index = usize::try_from(ipot)
                    .map_err(|_| parse_error(line, "NUMDENS ipot is out of range"))?;
                if index >= number_densities.len() {
                    number_densities.resize(index + 1, -1.0);
                }
                number_densities[index] = density;
            }
            Some("PREP") => print_eps = true,
            _ => {}
        }
    }

    Ok(OpconsInput {
        run_opcons,
        print_eps,
        number_densities,
    })
}

fn opcons_density_count(potentials: &[Potential]) -> usize {
    let nph = potentials
        .iter()
        .map(|potential| potential.ipot)
        .max()
        .unwrap_or(1)
        .max(1);
    usize::try_from(nph).map_or(2, |nph| nph + 1)
}

fn parse_sfconv_input(
    input: &FeffInput,
    ispec: i32,
    ipol: i32,
    spin: i32,
    print: Option<[i32; 6]>,
) -> Result<SfconvInput> {
    let mut sfconv = SfconvInput {
        control: SfconvControl {
            msfconv: 0,
            ipse: 0,
            ipsk: 0,
        },
        window: SfconvWindow {
            wsigk: 0.0,
            cen: 0.0,
        },
        spectrum: SfconvSpectrum {
            ispec: output_ispec_for_handoff(ispec, ipol, spin),
            ipr6: print.and_then(|values| values.get(5).copied()).unwrap_or(0),
        },
        cfname: "NULL".to_string(),
    };

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("SFCONV") => sfconv.control.msfconv = 1,
            Some("SELF") => sfconv.control.ipse = 1,
            Some("SFSE") => {
                let args = card_args(line)?;
                let Some(wsigk) = args.first() else {
                    return Err(parse_error(line, "SFSE requires wsigk"));
                };
                sfconv.control.ipsk = 1;
                sfconv.window.wsigk = parse_f64(line, wsigk)?;
                if !sfconv.window.wsigk.is_finite() {
                    return Err(parse_error(line, "SFSE wsigk must be finite"));
                }
            }
            Some("RCONV") => {
                let args = card_args(line)?;
                if args.len() < 2 {
                    return Err(parse_error(line, "RCONV requires cen and cfname"));
                }
                sfconv.window.cen = parse_f64(line, &args[0])?;
                if !sfconv.window.cen.is_finite() {
                    return Err(parse_error(line, "RCONV cen must be finite"));
                }
                sfconv.cfname = fortran_fixed_string(&args[1], 12);
            }
            _ => {}
        }
    }

    Ok(sfconv)
}

fn output_ispec_for_handoff(ispec: i32, ipol: i32, spin: i32) -> i32 {
    if ipol == 2 && spin != 0 { -1 } else { ispec }
}

fn fortran_fixed_string(value: &str, width: usize) -> String {
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > width {
            break;
        }
        end = next;
    }
    value[..end].to_string()
}

fn parse_corval_emin(input: &FeffInput) -> Result<f64> {
    let Some(line) = card_by_feff_name(input, "CORVAL") else {
        return Ok(-70.0);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(-70.0);
    };
    let value = parse_f64(line, value)?;
    if !value.is_finite() {
        return Err(parse_error(line, "CORVAL emin must be finite"));
    }
    Ok(value)
}

fn parse_scf_thermal(input: &FeffInput) -> Result<ScfThermal> {
    let Some(line) = card_by_feff_name(input, "SCFTH") else {
        return Ok(default_scf_thermal());
    };
    let args = card_args(line)?;
    let iscfth = parse_optional_i32(line, args.first())?.unwrap_or(0);
    let emaxscf = parse_optional_f64(line, args.get(1))?.unwrap_or(5.0);
    let xntol = parse_optional_f64(line, args.get(4))?.unwrap_or(1.0e-4);
    if !emaxscf.is_finite() || !xntol.is_finite() {
        return Err(parse_error(line, "SCFTH values must be finite"));
    }
    Ok(ScfThermal {
        iscfth,
        emaxscf,
        negrid: parse_optional_i32(line, args.get(2))?.unwrap_or(400),
        nmu: parse_optional_i32(line, args.get(3))?.unwrap_or(100),
        xntol,
    })
}

fn default_scf_thermal() -> ScfThermal {
    ScfThermal {
        iscfth: 2,
        xntol: 1.0e-4,
        nmu: 100,
        negrid: 400,
        emaxscf: 5.0,
    }
}

fn parse_scf_ramp(input: &FeffInput) -> Result<ScfRamp> {
    let Some(line) = card_by_feff_name(input, "SCFR") else {
        return Ok(ScfRamp {
            enabled: false,
            rfms_start: 0.0,
            nramp: 1,
        });
    };
    let args = card_args(line)?;
    let rfms_start = parse_optional_f64(line, args.first())?.unwrap_or(0.0);
    if !rfms_start.is_finite() {
        return Err(parse_error(line, "SCFR rfms_start must be finite"));
    }
    Ok(ScfRamp {
        enabled: true,
        rfms_start,
        nramp: parse_optional_i32(line, args.get(1))?.unwrap_or(1),
    })
}

fn parse_scf_tolerances(input: &FeffInput) -> Result<ScfTolerances> {
    let default = ScfTolerances {
        tolmu: 0.001,
        tolq: 0.001,
        tolqp: 0.0002,
    };
    let Some(line) = card_by_feff_name(input, "TOLS") else {
        return Ok(default);
    };
    let args = card_args(line)?;
    let Some(first) = args.first() else {
        return Err(parse_error(line, "TOLS requires a tolerance value"));
    };
    let first = parse_f64(line, first)?;
    if !first.is_finite() {
        return Err(parse_error(line, "TOLS values must be finite"));
    }
    if first < 0.0 {
        return Ok(ScfTolerances {
            tolmu: default.tolmu * first,
            tolq: default.tolq * first,
            tolqp: default.tolqp * first,
        });
    }
    let tolq = parse_optional_f64(line, args.get(1))?.unwrap_or(default.tolq);
    let tolqp = parse_optional_f64(line, args.get(2))?.unwrap_or(default.tolqp);
    if !tolq.is_finite() || !tolqp.is_finite() {
        return Err(parse_error(line, "TOLS values must be finite"));
    }
    Ok(ScfTolerances {
        tolmu: default.tolmu,
        tolq,
        tolqp,
    })
}

fn parse_fine_structure_damping(input: &FeffInput) -> Result<FineStructureDamping> {
    let mut damping = FineStructureDamping {
        alphat: 0.0,
        thetae: 0.0,
        sig2g: 0.0,
        sig_gk: 0.0,
    };

    if let Some(line) = card_by_feff_name(input, "SIG2") {
        let args = card_args(line)?;
        let Some(sig2g) = args.first() else {
            return Err(parse_error(line, "SIG2 requires sig2g"));
        };
        damping.sig2g = parse_f64(line, sig2g)?;
        if !damping.sig2g.is_finite() {
            return Err(parse_error(line, "SIG2 sig2g must be finite"));
        }
    }

    if let Some(line) = card_by_feff_name(input, "SIG3") {
        let args = card_args(line)?;
        let Some(alphat) = args.first() else {
            return Err(parse_error(line, "SIG3 requires alphat"));
        };
        damping.alphat = parse_f64(line, alphat)?;
        damping.thetae = parse_optional_f64(line, args.get(1))?.unwrap_or(0.0);
        if !damping.alphat.is_finite() || !damping.thetae.is_finite() {
            return Err(parse_error(line, "SIG3 values must be finite"));
        }
    }

    if let Some(line) = card_by_feff_name(input, "SIGGK") {
        let args = card_args(line)?;
        if let Some(sig_gk) = args.first() {
            damping.sig_gk = parse_f64(line, sig_gk)?;
            if !damping.sig_gk.is_finite() {
                return Err(parse_error(line, "SIGGK sig_gk must be finite"));
            }
        }
    }

    Ok(damping)
}

fn parse_config_type(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "CONFIGURATION") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    Ok(match args.first().map(|arg| arg.to_ascii_lowercase()) {
        Some(kind) if kind == "file" || kind == "card" => 2,
        Some(kind) if kind == "feff7" => 7,
        _ => 1,
    })
}

fn parse_config_records(input: &FeffInput) -> Result<Vec<String>> {
    let mut records = Vec::new();
    let mut index = 0_usize;
    while let Some(line) = input.lines.get(index) {
        if let LineKind::Card { keyword, args, .. } = &line.kind
            && keyword == "CONFIG"
            && args
                .first()
                .is_some_and(|arg| arg.eq_ignore_ascii_case("card"))
        {
            let Some(count_token) = args.get(1) else {
                return Err(parse_error(
                    line,
                    "CONFIG card requires a payload line count",
                ));
            };
            let count = parse_i32(line, count_token)?;
            if count < 0 {
                return Err(parse_error(
                    line,
                    "CONFIG card line count must be non-negative",
                ));
            }
            let count = usize::try_from(count)
                .map_err(|_| parse_error(line, "CONFIG card line count is out of range"))?;
            for offset in 1..=count {
                let Some(payload) = input.lines.get(index + offset) else {
                    return Err(parse_error(
                        line,
                        "CONFIG card payload is shorter than declared",
                    ));
                };
                match &payload.kind {
                    LineKind::SectionData { section, .. } if section == "CONFIG" => {
                        records.push(payload.raw.clone());
                    }
                    LineKind::SectionData { .. } | LineKind::Card { .. } => {
                        return Err(parse_error(
                            payload,
                            "CONFIG card payload ended before declared line count",
                        ));
                    }
                }
            }
            index += count;
        }
        index += 1;
    }
    Ok(records)
}

fn parse_egrid_records(input: &FeffInput) -> Result<Vec<String>> {
    let mut records = Vec::new();
    let mut index = 0_usize;
    while let Some(line) = input.lines.get(index) {
        if let LineKind::Card { keyword, args, .. } = &line.kind
            && keyword == "EGRID"
            && args.is_empty()
        {
            let mut block = Vec::new();
            let mut offset = 1_usize;
            while let Some(payload) = input.lines.get(index + offset) {
                match &payload.kind {
                    LineKind::SectionData { section, fields } if section == "EGRID" => {
                        block.push(fields.join(" "));
                    }
                    LineKind::SectionData { .. } | LineKind::Card { .. } => break,
                }
                offset += 1;
            }

            if block.is_empty() {
                index += offset;
                continue;
            }

            let text = block
                .iter()
                .map(|record| format!(" {record} \n"))
                .collect::<String>();
            parse_grid_inp(&text)?;
            records.extend(block);
            index += offset.saturating_sub(1);
        }
        index += 1;
    }
    Ok(records)
}

fn parse_density_records(input: &FeffInput) -> Result<Vec<String>> {
    let records = input
        .section_rows("DENSITY")
        .map(|line| line.raw.clone())
        .collect::<Vec<_>>();
    if records.is_empty() {
        return Ok(records);
    }

    let text = records
        .iter()
        .map(|record| format!("{record}\n"))
        .collect::<String>();
    let density_path = input
        .source
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("density.inp");
    DensityInput::parse_str(density_path, &text)?;
    Ok(records)
}

fn parse_band_input(input: &FeffInput) -> Result<BandInput> {
    let Some(line) = card_by_feff_name(input, "BAND") else {
        return Ok(default_band_input());
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "BANDSTRUCTURE requires emin emax estep ikpath",
        ));
    }
    Ok(BandInput {
        mband: 1,
        energy_mesh: BandEnergyMesh {
            emin: parse_f64(line, &args[0])?,
            emax: parse_f64(line, &args[1])?,
            estep: parse_f64(line, &args[2])?,
        },
        nkp: parse_optional_i32(line, args.get(4))?.unwrap_or(0),
        ikpath: parse_i32(line, &args[3])?,
        freeprop: match args.get(5) {
            Some(value) => parse_logical(line, value)?,
            None => false,
        },
    })
}

fn default_band_input() -> BandInput {
    BandInput {
        mband: 0,
        energy_mesh: BandEnergyMesh {
            emin: 0.0,
            emax: 0.0,
            estep: 0.0,
        },
        nkp: 0,
        ikpath: -1,
        freeprop: false,
    }
}

fn parse_full_spectrum_input(active_cards: &[String]) -> FullSpectrumInput {
    FullSpectrumInput {
        m_full_spectrum: i32::from(active_cards.iter().any(|card| card == "FULLSPECTRUM")),
    }
}

fn parse_screen_input(input: &FeffInput) -> Result<ScreenInput> {
    let mut screen = ScreenInput::default();

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        if feff_card_token(keyword).map(|(_, display)| display) != Some("SCREEN") {
            continue;
        }

        let args = card_args(line)?;
        if args.len() < 2 {
            return Err(parse_error(line, "SCREEN requires keyword and value"));
        }

        let key = screen_key_prefix(&args[0]);
        let value = parse_f64(line, &args[1])?;
        if !value.is_finite() {
            return Err(parse_error(line, "SCREEN value must be finite"));
        }

        match key.as_str() {
            "ner" => screen.ner = parse_screen_i32(line, "ner", value)?,
            "nei" => screen.nei = parse_screen_i32(line, "nei", value)?,
            "max" => screen.maxl = parse_screen_i32(line, "maxl", value)?,
            "irr" => screen.irrh = parse_screen_i32(line, "irrh", value)?,
            "ien" => screen.iend = parse_screen_i32(line, "iend", value)?,
            "lfx" => screen.lfxc = parse_screen_i32(line, "lfxc", value)?,
            "emi" => screen.emin = value,
            "ema" => screen.emax = value,
            "eim" => screen.eimax = value,
            "erm" => screen.ermin = value,
            "rfm" => screen.rfms = value,
            "nrp" => screen.nrptx0 = parse_screen_i32(line, "nrptx0", value)?,
            "ico" => screen.icore = parse_screen_i32(line, "icore", value)?,
            _ => {
                return Err(parse_error(
                    line,
                    format!("unrecognized SCREEN keyword {:?}", args[0]),
                ));
            }
        }
    }

    Ok(screen)
}

fn screen_key_prefix(key: &str) -> String {
    key.chars().take(3).flat_map(char::to_lowercase).collect()
}

fn parse_screen_i32(line: &FeffLine, field: &str, value: f64) -> Result<i32> {
    let rounded = value.round();
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(parse_error(
            line,
            format!("SCREEN {field} value is out of range"),
        ));
    }
    Ok(rounded as i32)
}

fn parse_i32_6(input: &FeffInput, keyword: &str) -> Result<Option<[i32; 6]>> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    if args.len() == 4 {
        let shared = parse_i32(line, &args[0])?;
        return Ok(Some([
            shared,
            shared,
            shared,
            parse_i32(line, &args[1])?,
            parse_i32(line, &args[2])?,
            parse_i32(line, &args[3])?,
        ]));
    }
    let mut values = [0_i32; 6];
    for (slot, arg) in values.iter_mut().zip(args.iter()) {
        *slot = parse_i32(line, arg)?;
    }
    Ok(Some(values))
}

fn parse_debye(input: &FeffInput) -> Result<Option<Debye>> {
    let Some(line) = card_by_feff_name(input, "DEBYE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(temperature) = args.first() else {
        return Err(parse_error(line, "DEBYE requires a temperature"));
    };
    let Some(debye_temperature) = args.get(1) else {
        return Err(parse_error(line, "DEBYE requires a Debye temperature"));
    };

    let idwopt = parse_optional_i32(line, args.get(2))?.unwrap_or(0);
    let dym_file = (idwopt == 5).then(|| {
        args.get(3)
            .map(|value| strip_card_delimiters(value).to_string())
            .unwrap_or_else(|| "feff.dym".to_string())
    });

    Ok(Some(Debye {
        temperature: parse_f64(line, temperature)?,
        debye_temperature: parse_f64(line, debye_temperature)?,
        idwopt,
        dym_file,
        dmdw_order: parse_optional_i32(line, args.get(4))?.unwrap_or(2),
        dmdw_type: parse_optional_i32(line, args.get(5))?.unwrap_or(0),
        dmdw_route: parse_optional_i32(line, args.get(6))?.unwrap_or(0),
    }))
}

fn parse_spring_input_text(input: &FeffInput, debye: Option<&Debye>) -> Result<Option<String>> {
    if !debye.is_some_and(|debye| matches!(debye.idwopt, 1 | 2)) {
        return Ok(None);
    }

    let path = input
        .source
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("spring.inp");
    let text = std::fs::read_to_string(&path).map_err(|source| IoError::io(&path, source))?;
    parse_spring_inp(&text)?;
    Ok(Some(text))
}

fn parse_dym_input(input: &FeffInput, debye: Option<&Debye>) -> Result<Option<AuxiliaryTextFile>> {
    let Some(dym_file) = debye
        .filter(|debye| debye.idwopt == 5)
        .and_then(|debye| debye.dym_file.as_deref())
    else {
        return Ok(None);
    };

    let output_name = relative_auxiliary_output_name(dym_file)?;
    let path = resolve_auxiliary_path(input, dym_file);
    let text = std::fs::read_to_string(&path).map_err(|source| IoError::io(&path, source))?;
    parse_dym(&text)?;
    let Some(output_name) = output_name else {
        return Ok(None);
    };
    Ok(Some(AuxiliaryTextFile { output_name, text }))
}

fn resolve_auxiliary_path(input: &FeffInput, name: &str) -> PathBuf {
    let path = PathBuf::from(name);
    if path.is_absolute() {
        path
    } else {
        input
            .source
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn relative_auxiliary_output_name(name: &str) -> Result<Option<String>> {
    let path = Path::new(name);
    if path.is_absolute() {
        return Ok(None);
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(IoError::Parse {
                    path: path.to_path_buf(),
                    line: 0,
                    message: "DMDW auxiliary output path must stay within the output directory"
                        .to_string(),
                });
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        return Err(IoError::Parse {
            path: path.to_path_buf(),
            line: 0,
            message: "DMDW auxiliary output path is empty".to_string(),
        });
    }

    Ok(Some(normalized.to_string_lossy().into_owned()))
}

fn parse_rpath(input: &FeffInput) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, "RPATH") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "RPATH requires rmax"));
    };
    Ok(Some(parse_f64(line, radius)?))
}

fn parse_nleg(input: &FeffInput) -> Result<Option<i32>> {
    let Some(line) = card_by_feff_name(input, "NLEG") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(Some(7));
    };
    Ok(Some(parse_i32(line, value)?))
}

fn parse_path_symmetry(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "SYMMETRY") else {
        return Ok(-1);
    };
    let args = card_args(line)?;
    let ica = parse_optional_i32(line, args.first())?.unwrap_or(-1);
    Ok(if (1..=7).contains(&ica) { ica } else { -1 })
}

fn parse_dims(input: &FeffInput) -> Result<Option<DimensionLimits>> {
    let Some(line) = card_by_feff_name(input, "DIMS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let nclusx = parse_optional_i32(line, args.first())?.unwrap_or(0);
    let lx = parse_optional_i32(line, args.get(1))?.unwrap_or(0);

    Ok(Some(DimensionLimits { nclusx, lx }))
}

fn parse_ldos(input: &FeffInput) -> Result<Option<Ldos>> {
    let Some(line) = card_by_feff_name(input, "LDOS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(emin) = args.first() else {
        return Err(parse_error(line, "LDOS requires emin"));
    };
    let Some(emax) = args.get(1) else {
        return Err(parse_error(line, "LDOS requires emax"));
    };
    let Some(eimag) = args.get(2) else {
        return Err(parse_error(line, "LDOS requires eimag"));
    };

    Ok(Some(Ldos {
        emin: parse_f64(line, emin)?,
        emax: parse_f64(line, emax)?,
        eimag: parse_f64(line, eimag)?,
        neldos: parse_optional_i32(line, args.get(3))?.unwrap_or(101),
        ldostype: parse_optional_i32(line, args.get(4))?.unwrap_or(0),
    }))
}

fn parse_interstitial(input: &FeffInput) -> Result<Option<Interstitial>> {
    let Some(line) = card_by_feff_name(input, "INTERSTITIAL") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    Ok(Some(Interstitial {
        mode: parse_optional_i32(line, args.first())?.unwrap_or(0),
        volume_scale: parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
    }))
}

fn parse_afolp(input: &FeffInput) -> Result<f64> {
    let Some(line) = card_by_feff_name(input, "AFOLP") else {
        return Ok(1.15);
    };
    let args = card_args(line)?;
    parse_optional_f64(line, args.first()).map(|value| value.unwrap_or(1.15))
}

fn parse_overlap_factors(input: &FeffInput) -> Result<Vec<OverlapFactor>> {
    let mut factors = Vec::new();
    for line in input.cards() {
        if let LineKind::Card { keyword, .. } = &line.kind
            && feff_card_token(keyword).map(|(_, display)| display) == Some("FOLP")
        {
            let args = card_args(line)?;
            if args.len() < 2 {
                return Err(parse_error(line, "FOLP requires ipot and folp"));
            }
            let factor = parse_f64(line, &args[1])?;
            if !factor.is_finite() {
                return Err(parse_error(line, "FOLP factor must be finite"));
            }
            factors.push(OverlapFactor {
                potential_index: parse_i32(line, &args[0])?,
                factor,
            });
        }
    }
    Ok(factors)
}

fn parse_ionizations(input: &FeffInput) -> Result<Vec<Ionization>> {
    let mut ionizations = Vec::new();
    for line in input.cards() {
        if let LineKind::Card { keyword, .. } = &line.kind
            && feff_card_token(keyword).map(|(_, display)| display) == Some("ION")
        {
            let args = card_args(line)?;
            if args.len() < 2 {
                return Err(parse_error(line, "ION requires ipot and ionization"));
            }
            let value = parse_f64(line, &args[1])?;
            if !value.is_finite() {
                return Err(parse_error(line, "ION value must be finite"));
            }
            ionizations.push(Ionization {
                potential_index: parse_i32(line, &args[0])?,
                value,
            });
        }
    }
    Ok(ionizations)
}

fn parse_potentials(input: &FeffInput) -> Result<Vec<Potential>> {
    input
        .section_rows("POTENTIALS")
        .map(|line| {
            let fields = section_fields_before_star(line)?;
            if fields.len() < 2 {
                return Err(parse_error(line, "POTENTIALS rows require ipot and Z"));
            }
            let z = parse_i32(line, fields[1])?;

            Ok(Potential {
                ipot: parse_i32(line, fields[0])?,
                z: Some(z),
                z_token: fields[1].clone(),
                tag: fields.get(2).map(|value| (*value).clone()),
                lmax1: parse_optional_i32(line, fields.get(3).copied())?,
                lmax2: parse_optional_i32(line, fields.get(4).copied())?,
                xnatph: parse_optional_f64(line, fields.get(5).copied())?,
                spinph: parse_optional_f64(line, fields.get(6).copied())?,
            })
        })
        .collect()
}

fn parse_atoms(input: &FeffInput) -> Result<Vec<Atom>> {
    input
        .section_rows("ATOMS")
        .map(|line| {
            let fields = section_fields_before_star(line)?;
            if fields.len() < 4 {
                return Err(parse_error(line, "ATOMS rows require x y z ipot"));
            }

            Ok(Atom {
                x: parse_f64(line, fields[0])?,
                y: parse_f64(line, fields[1])?,
                z: parse_f64(line, fields[2])?,
                ipot: parse_i32(line, fields[3])?,
                tag: fields.get(4).map(|value| (*value).clone()),
                distance: fields
                    .iter()
                    .skip(5)
                    .find_map(|value| parse_f64(line, value).ok()),
                index: fields
                    .iter()
                    .skip(5)
                    .find_map(|value| value.parse::<usize>().ok()),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
