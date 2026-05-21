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
mod types;

use cards::*;

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

fn validate_feff_consistency(input: &FeffInput, active_cards: &[String]) -> Result<()> {
    let spectroscopy_cards = [
        "XANES", "EXAFS", "XES", "DANES", "FPRIME", "ELNES", "EXELFS",
    ];
    let active_spectroscopy = spectroscopy_cards
        .iter()
        .filter(|card| active_card(active_cards, card))
        .count();
    if active_spectroscopy > 1
        && let Some(card) = spectroscopy_cards
            .iter()
            .find(|card| active_card(active_cards, card))
    {
        return Err(parse_error(
            required_card_line(input, card)?,
            "ERROR more than one type of spectroscopy selected",
        ));
    }

    if active_card(active_cards, "NRIXS") {
        let nrixs_line = required_card_line(input, "NRIXS")?;
        let xanes_or_exafs = ["XANES", "EXAFS"]
            .iter()
            .filter(|card| active_card(active_cards, card))
            .count();
        if xanes_or_exafs != 1 {
            return Err(parse_error(
                nrixs_line,
                "NRIXS must be combined with XANES or EXAFS",
            ));
        }
        if let Some(card) = ["FPRIME", "XES", "DANES", "ELNES", "EXELFS"]
            .iter()
            .find(|card| active_card(active_cards, card))
        {
            return Err(parse_error(
                required_card_line(input, card)?,
                "NRIXS combined with incompatible spectroscopy card",
            ));
        }
        if active_card(active_cards, "MULT") {
            return Err(parse_error(
                required_card_line(input, "MULT")?,
                "you cannot combine NRIXS and MULTIPOLE",
            ));
        }
        if let Some(card) = [
            "ELLIPTICITY",
            "POLARIZATION",
            "NSTAR",
            "SPIN",
            "CFAVERAGE",
            "XMCD",
            "RPHASES",
            "TDLDA",
            "PMBSE",
            "HUBBARD",
        ]
        .iter()
        .find(|card| active_card(active_cards, card))
        {
            return Err(parse_error(
                required_card_line(input, card)?,
                "card is explicitly forbidden for NRIXS",
            ));
        }
    } else if active_card(active_cards, "LJMAX") || active_card(active_cards, "LDECMX") {
        let line = card_by_feff_name(input, "LJMAX")
            .or_else(|| card_by_feff_name(input, "LDECMX"))
            .ok_or_else(|| IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "LDEC/LJMAX card not found".to_string(),
            })?;
        return Err(parse_error(
            line,
            "LDEC and LJMAX cards only allowed with NRIXS",
        ));
    }

    if active_card(active_cards, "RECIPROCAL") {
        let reciprocal_line = required_card_line(input, "RECIPROCAL")?;
        if !(active_card(active_cards, "KMESH") && active_card(active_cards, "TARGET")) {
            return Err(parse_error(
                reciprocal_line,
                "KMESH and TARGET are required for RECIPROCAL card",
            ));
        }

        let structure_source_count = ["LATTICE", "CIF"]
            .iter()
            .filter(|card| active_card(active_cards, card))
            .count();
        if structure_source_count != 1 {
            return Err(parse_error(
                reciprocal_line,
                "use either LATTICE or CIF with RECIPROCAL card",
            ));
        }
    }

    if active_card(active_cards, "CGRID")
        && !(active_card(active_cards, "COMPTON") || active_card(active_cards, "RHOZZP"))
    {
        return Err(parse_error(
            required_card_line(input, "CGRID")?,
            "Cannot use CGRID without COMPTON or RHOZZP.  Exiting.",
        ));
    }

    if active_card(active_cards, "HUBBARD") && active_card(active_cards, "RECIPROCAL") {
        return Err(parse_error(
            required_card_line(input, "HUBBARD")?,
            "Cannot use RECIPROCAL with HUBBARD.",
        ));
    }

    Ok(())
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

fn parse_reciprocal_space(input: &FeffInput) -> bool {
    input.cards().fold(false, |reciprocal, line| {
        let LineKind::Card { keyword, .. } = &line.kind else {
            return reciprocal;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("REAL") => false,
            Some("RECIPROCAL") => true,
            _ => reciprocal,
        }
    })
}

fn parse_cif_equivalence(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "EQUIVALENCE") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    let Some(selector) = args.first() else {
        return Err(parse_error(line, "EQUIVALENCE requires a selector"));
    };
    let selector = parse_i32(line, selector)?;
    match selector {
        1 | 2 | 4 => Ok(selector),
        3 => Err(parse_error(
            line,
            "EQUIVALENCE 3 is not implemented by FEFF10",
        )),
        _ => Err(parse_error(line, "EQUIVALENCE must be 1, 2, 3, or 4")),
    }
}

fn cif_equivalence_mode(selector: i32) -> CifEquivalence {
    match selector {
        2 => CifEquivalence::AtomicNumber,
        4 => CifEquivalence::AutomaticLimit,
        _ => CifEquivalence::Crystallographic,
    }
}

fn parse_coordinate_mode(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "COORDINATES") else {
        return Ok(3);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "COORDINATES requires a selector"));
    };
    let mode = parse_i32(line, value)?;
    if (1..=6).contains(&mode) {
        Ok(mode)
    } else {
        Err(parse_error(line, "COORDINATES must be between 1 and 6"))
    }
}

fn parse_reciprocal_input(
    input: &FeffInput,
    nohole: i32,
    atoms: &[Atom],
    reciprocal: bool,
    cif_equivalence: i32,
) -> Result<Option<ReciprocalInput>> {
    if !reciprocal {
        return Ok(None);
    };
    let Some(reciprocal_line) = card_by_feff_name(input, "RECIPROCAL") else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "RECIPROCAL mode requires a RECIPROCAL card".to_string(),
        });
    };
    let k_mesh = parse_k_mesh(input)?;
    let absorber = parse_required_i32_card(input, "TARGET")?;
    let stretch = parse_strfac(input)?;

    let Some(lattice) = parse_lattice_block(input)? else {
        if let Some(cif_line) = card_by_feff_name(input, "CIF") {
            let cif_path = parse_cif_path(input, cif_line)?;
            let cif = read_cif(&cif_path)?;
            if absorber <= 0 {
                return Err(parse_error(
                    cif_line,
                    "TARGET must be positive for CIF input",
                ));
            }
            let target = usize::try_from(absorber)
                .map_err(|_| parse_error(cif_line, "TARGET is out of range for CIF input"))?;
            let structure = expand_cif_structure_with_equivalence(
                &cif,
                target,
                cif_equivalence_mode(cif_equivalence),
            )?;
            return Ok(Some(ReciprocalInput {
                ispace: 0,
                cell: Some(ReciprocalCell {
                    lattice_vectors: structure.lattice_vectors,
                    volume_scale: -1.0,
                    imaginary_energy: 0.0,
                    core_hole_strength: 1.0,
                    lattice_name: structure.lattice_name,
                    space_group_hm: structure.space_group_hm,
                    space_group: structure.space_group,
                    atom_count: structure.positions.len(),
                    absorber: i32::try_from(structure.absorber).map_err(|_| {
                        parse_error(cif_line, "expanded CIF absorber index is out of range")
                    })?,
                    core_hole: i32::from(nohole != 0),
                    k_mesh,
                    positions: structure.positions,
                    potentials: structure.potentials,
                    labels: structure.labels,
                    stretch,
                }),
            }));
        }
        return Err(parse_error(
            reciprocal_line,
            "RECIPROCAL requires LATTICE or CIF",
        ));
    };
    if atoms.is_empty() {
        return Err(parse_error(
            reciprocal_line,
            "RECIPROCAL with LATTICE requires ATOMS rows",
        ));
    }

    let space_group = parse_sgroup(input)?;
    let coordinate_mode = parse_coordinate_mode(input)?;
    let atoms = convert_lattice_atoms(input, &lattice, atoms, coordinate_mode)?;
    let positions = atoms.iter().map(|atom| [atom.x, atom.y, atom.z]).collect();
    let potentials = atoms.iter().map(|atom| atom.ipot).collect();

    Ok(Some(ReciprocalInput {
        ispace: 0,
        cell: Some(ReciprocalCell {
            lattice_vectors: lattice.vectors,
            volume_scale: -1.0,
            imaginary_energy: 0.0,
            core_hole_strength: 1.0,
            lattice_name: lattice.name,
            space_group_hm: "\0".repeat(8),
            space_group,
            atom_count: atoms.len(),
            absorber,
            core_hole: i32::from(nohole != 0),
            k_mesh,
            positions,
            potentials,
            labels: Vec::new(),
            stretch,
        }),
    }))
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

fn parse_cif_path(input: &FeffInput, line: &FeffLine) -> Result<PathBuf> {
    let args = card_args(line)?;
    let Some(path) = args.first() else {
        return Err(parse_error(line, "CIF requires a file path"));
    };
    let path = strip_card_delimiters(path);
    let path = PathBuf::from(path);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(input
            .source
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path))
    }
}

fn strip_card_delimiters(value: &str) -> &str {
    let pairs = [
        ('"', '"'),
        ('\'', '\''),
        ('{', '}'),
        ('(', ')'),
        ('<', '>'),
        ('[', ']'),
    ];
    pairs
        .iter()
        .find_map(|(open, close)| {
            (value.starts_with(*open) && value.ends_with(*close) && value.len() >= 2)
                .then_some(&value[1..value.len() - 1])
        })
        .unwrap_or(value)
}

fn parse_cif_cluster(
    input: &FeffInput,
    radius: f64,
    needed: bool,
    cif_equivalence: i32,
) -> Result<Option<CifCluster>> {
    if !needed {
        return Ok(None);
    }
    let Some(cif_line) = card_by_feff_name(input, "CIF") else {
        return Ok(None);
    };
    let cif_path = parse_cif_path(input, cif_line)?;
    let cif = read_cif(&cif_path)?;
    let target = parse_cif_target(input, cif_line)?;
    expand_cif_cluster_with_equivalence(&cif, target, radius, cif_equivalence_mode(cif_equivalence))
        .map(Some)
}

fn cif_cluster_radius(scf: Option<&Scf>, fms: Option<&Fms>, rpath: Option<f64>) -> f64 {
    [scf.map(|scf| scf.radius), fms.map(|fms| fms.radius), rpath]
        .into_iter()
        .flatten()
        .fold(0.0, f64::max)
}

fn cif_cluster_potentials(cluster: &CifCluster) -> Result<Vec<Potential>> {
    cluster
        .potentials
        .iter()
        .map(|potential| {
            let xnatph = if potential.absorber {
                Some(0.01)
            } else {
                Some(potential.multiplicity as f64)
            };
            Ok(Potential {
                ipot: potential.ipot,
                z: Some(potential.atomic_number),
                z_token: potential.atomic_number.to_string(),
                tag: Some(potential.label.clone()),
                lmax1: None,
                lmax2: None,
                xnatph,
                spinph: None,
            })
        })
        .collect()
}

fn cif_cluster_atoms(cluster: &CifCluster) -> Vec<Atom> {
    cluster
        .atoms
        .iter()
        .map(|atom| Atom {
            x: atom.x,
            y: atom.y,
            z: atom.z,
            ipot: atom.potential,
            tag: None,
            distance: None,
            index: None,
        })
        .collect()
}

fn parse_cif_target(input: &FeffInput, cif_line: &FeffLine) -> Result<usize> {
    let Some(target_line) = card_by_feff_name(input, "TARGET") else {
        return Ok(1);
    };
    let args = card_args(target_line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(target_line, "TARGET requires a value"));
    };
    let target = parse_i32(target_line, value)?;
    if target <= 0 {
        return Err(parse_error(
            cif_line,
            "TARGET must be positive for CIF input",
        ));
    }
    usize::try_from(target)
        .map_err(|_| parse_error(cif_line, "TARGET is out of range for CIF input"))
}

struct LatticeBlock {
    name: String,
    vectors: [[f64; 3]; 3],
}

#[derive(Debug, Clone, Copy)]
struct PeriodicAtom {
    x: f64,
    y: f64,
    z: f64,
    ipot: i32,
    distance: f64,
}

fn parse_lattice_cluster_atoms(
    input: &FeffInput,
    atoms: &[Atom],
    radius: f64,
    reciprocal: bool,
) -> Result<Option<Vec<Atom>>> {
    if !reciprocal || card_by_feff_name(input, "CIF").is_some() {
        return Ok(None);
    }
    let Some(lattice) = parse_lattice_block(input)? else {
        return Ok(None);
    };
    if atoms.is_empty() {
        return Ok(None);
    }
    let target = parse_required_i32_card(input, "TARGET")?;
    if target <= 0 {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "TARGET must be positive for LATTICE input".to_string(),
        });
    }
    let target = usize::try_from(target - 1).map_err(|_| IoError::Parse {
        path: input.source.clone(),
        line: 0,
        message: "TARGET is out of range for LATTICE input".to_string(),
    })?;
    if target >= atoms.len() {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!(
                "TARGET {} is outside the ATOMS row range 1..={}",
                target + 1,
                atoms.len()
            ),
        });
    }
    let coordinate_mode = parse_coordinate_mode(input)?;
    let atoms = convert_lattice_atoms(input, &lattice, atoms, coordinate_mode)?;
    Ok(Some(expand_lattice_cluster(
        &lattice, &atoms, target, radius,
    )))
}

fn convert_lattice_atoms(
    input: &FeffInput,
    lattice: &LatticeBlock,
    atoms: &[Atom],
    coordinate_mode: i32,
) -> Result<Vec<Atom>> {
    let lengths = lattice_vector_lengths(input, lattice)?;
    atoms
        .iter()
        .map(|atom| convert_lattice_atom(input, lattice, lengths, atom, coordinate_mode))
        .collect()
}

fn convert_lattice_atom(
    input: &FeffInput,
    lattice: &LatticeBlock,
    lengths: [f64; 3],
    atom: &Atom,
    coordinate_mode: i32,
) -> Result<Atom> {
    let [a1_len, a2_len, a3_len] = lengths;
    let position = match coordinate_mode {
        1 => [atom.x / a1_len, atom.y / a1_len, atom.z / a1_len],
        2 => [atom.x, atom.y * a2_len / a1_len, atom.z * a3_len / a1_len],
        3 => [atom.x, atom.y, atom.z],
        4 => scale_vector(
            fractional_to_cartesian([atom.x, atom.y, atom.z], lattice.vectors),
            1.0 / a1_len,
        ),
        5 => {
            let fractional = [atom.x, atom.y * a1_len / a2_len, atom.z * a1_len / a3_len];
            scale_vector(
                fractional_to_cartesian(fractional, lattice.vectors),
                1.0 / a1_len,
            )
        }
        6 => {
            let fractional = [atom.x / a1_len, atom.y / a2_len, atom.z / a3_len];
            scale_vector(
                fractional_to_cartesian(fractional, lattice.vectors),
                1.0 / a1_len,
            )
        }
        _ => {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "COORDINATES must be between 1 and 6".to_string(),
            });
        }
    };
    Ok(Atom {
        x: position[0],
        y: position[1],
        z: position[2],
        ipot: atom.ipot,
        tag: atom.tag.clone(),
        distance: atom.distance,
        index: atom.index,
    })
}

fn lattice_vector_lengths(input: &FeffInput, lattice: &LatticeBlock) -> Result<[f64; 3]> {
    let lengths = lattice.vectors.map(lattice_vector_length);
    if lengths
        .iter()
        .all(|length| length.is_finite() && *length > 0.0)
    {
        Ok(lengths)
    } else {
        Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "LATTICE vectors must have positive finite lengths".to_string(),
        })
    }
}

fn expand_lattice_cluster(
    lattice: &LatticeBlock,
    atoms: &[Atom],
    target: usize,
    radius: f64,
) -> Vec<Atom> {
    let [a1, a2, a3] = lattice.vectors;
    let ratomslist = 8.0_f64.max(1.33 * radius.max(0.0));
    let i1 = lattice_repeat_count(ratomslist, a1);
    let i2 = lattice_repeat_count(ratomslist, a2);
    let i3 = lattice_repeat_count(ratomslist, a3);
    let shifts = lattice_centering_shifts(&lattice.name);
    let lattice_scale = lattice_vector_length(a1);
    let absorber = lattice_atom_position(&atoms[target], lattice_scale);

    let mut expanded = Vec::new();
    let mut absorber_index = 0_usize;
    for j1 in -i1..=i1 {
        for j2 in -i2..=i2 {
            for j3 in -i3..=i3 {
                let translation = lattice_translation(j1, j2, j3, a1, a2, a3);
                for (index, atom) in atoms.iter().enumerate() {
                    let position =
                        add_vectors(lattice_atom_position(atom, lattice_scale), translation);
                    let mut ipot = atom.ipot;
                    if j1 == 0 && j2 == 0 && j3 == 0 && index == target {
                        ipot = 0;
                        absorber_index = expanded.len();
                    }
                    expanded.push(periodic_atom(position, ipot, absorber));

                    for shift in &shifts {
                        let shifted =
                            add_vectors(position, fractional_to_cartesian(*shift, [a1, a2, a3]));
                        expanded.push(periodic_atom(shifted, atom.ipot, absorber));
                    }
                }
            }
        }
    }

    feff_sort_periodic_atoms(&mut expanded, absorber_index);
    let cutoff = (lattice_vector_length(a1) * f64::from(i1))
        .min(lattice_vector_length(a2) * f64::from(i1))
        .min(lattice_vector_length(a3) * f64::from(i1));
    let keep = expanded
        .iter()
        .position(|atom| atom.distance > cutoff)
        .unwrap_or(expanded.len());
    expanded.truncate(keep);

    expanded
        .into_iter()
        .map(|atom| Atom {
            x: atom.x,
            y: atom.y,
            z: atom.z,
            ipot: atom.ipot,
            tag: None,
            distance: None,
            index: None,
        })
        .collect()
}

fn periodic_atom(position: [f64; 3], ipot: i32, absorber: [f64; 3]) -> PeriodicAtom {
    PeriodicAtom {
        x: position[0],
        y: position[1],
        z: position[2],
        ipot,
        distance: lattice_distance(position, absorber),
    }
}

fn lattice_atom_position(atom: &Atom, scale: f64) -> [f64; 3] {
    [atom.x * scale, atom.y * scale, atom.z * scale]
}

fn feff_sort_periodic_atoms(atoms: &mut [PeriodicAtom], mut absorber_index: usize) {
    for i in 0..atoms.len() {
        let mut min_index = i;
        let mut min_distance = atoms[i].distance;
        for (j, atom) in atoms.iter().enumerate().skip(i) {
            if atom.distance < min_distance {
                min_index = j;
                min_distance = atom.distance;
            }
        }
        atoms.swap(i, min_index);
        if i == absorber_index {
            absorber_index = min_index;
        }
        if min_index == absorber_index {
            absorber_index = i;
        }
    }
}

fn lattice_repeat_count(radius: f64, vector: [f64; 3]) -> i32 {
    (radius / lattice_vector_length(vector)).trunc() as i32 + 1
}

fn lattice_centering_shifts(lattice_name: &str) -> Vec<[f64; 3]> {
    match lattice_name {
        "F" => vec![[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        "CXY" => vec![[0.5, 0.5, 0.0]],
        "CXZ" => vec![[0.5, 0.0, 0.5]],
        "CYZ" => vec![[0.0, 0.5, 0.5]],
        "B" | "I" => vec![[0.5, 0.5, 0.5]],
        _ => Vec::new(),
    }
}

fn fractional_to_cartesian(position: [f64; 3], lattice_vectors: [[f64; 3]; 3]) -> [f64; 3] {
    [
        position[0].mul_add(
            lattice_vectors[0][0],
            position[1].mul_add(lattice_vectors[1][0], position[2] * lattice_vectors[2][0]),
        ),
        position[0].mul_add(
            lattice_vectors[0][1],
            position[1].mul_add(lattice_vectors[1][1], position[2] * lattice_vectors[2][1]),
        ),
        position[0].mul_add(
            lattice_vectors[0][2],
            position[1].mul_add(lattice_vectors[1][2], position[2] * lattice_vectors[2][2]),
        ),
    ]
}

fn lattice_translation(
    j1: i32,
    j2: i32,
    j3: i32,
    a1: [f64; 3],
    a2: [f64; 3],
    a3: [f64; 3],
) -> [f64; 3] {
    add_vectors(
        add_vectors(
            scale_vector(a1, f64::from(j1)),
            scale_vector(a2, f64::from(j2)),
        ),
        scale_vector(a3, f64::from(j3)),
    )
}

fn lattice_distance(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lattice_vector_length([lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]])
}

fn lattice_vector_length(vector: [f64; 3]) -> f64 {
    vector[0]
        .mul_add(
            vector[0],
            vector[1].mul_add(vector[1], vector[2] * vector[2]),
        )
        .sqrt()
}

fn add_vectors(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn scale_vector(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn parse_lattice_block(input: &FeffInput) -> Result<Option<LatticeBlock>> {
    let Some(line) = card_by_feff_name(input, "LATTICE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(name) = args.first() else {
        return Err(parse_error(line, "LATTICE requires a lattice type"));
    };
    let scale = parse_optional_f64(line, args.get(1))?.unwrap_or(1.0);
    let rows = input.section_rows("LATTICE").collect::<Vec<_>>();
    if rows.len() < 3 {
        return Err(parse_error(line, "LATTICE requires three vector rows"));
    }

    let mut vectors = [[0.0; 3]; 3];
    for (idx, row) in rows.iter().take(3).enumerate() {
        let fields = section_fields(row)?;
        if fields.len() < 3 {
            return Err(parse_error(row, "LATTICE vector rows require x y z"));
        }
        vectors[idx] = [
            parse_f64(row, &fields[0])? * scale,
            parse_f64(row, &fields[1])? * scale,
            parse_f64(row, &fields[2])? * scale,
        ];
    }

    Ok(Some(LatticeBlock {
        name: name.clone(),
        vectors,
    }))
}

fn parse_k_mesh(input: &FeffInput) -> Result<ReciprocalKMesh> {
    let Some(line) = card_by_feff_name(input, "KMESH") else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "RECIPROCAL requires KMESH".to_string(),
        });
    };
    let args = card_args(line)?;
    let Some(x) = args.first() else {
        return Err(parse_error(line, "KMESH requires at least one value"));
    };
    let x = parse_i32(line, x)?;
    let y = parse_optional_i32(line, args.get(1))?.unwrap_or(0);
    let z = parse_optional_i32(line, args.get(2))?.unwrap_or(0);
    let product = x * y * z;
    Ok(ReciprocalKMesh {
        total: if product == 0 { x } else { product },
        x,
        y,
        z,
        kind: parse_optional_i32(line, args.get(3))?.unwrap_or(1),
        use_symmetry: parse_optional_i32(line, args.get(4))?.unwrap_or(0) != 0,
    })
}

fn parse_required_i32_card(input: &FeffInput, keyword: &str) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!("RECIPROCAL requires {keyword}"),
        });
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, format!("{keyword} requires a value")));
    };
    parse_i32(line, value)
}

fn parse_strfac(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = card_by_feff_name(input, "STRFAC") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "STRFAC requires three values"));
    }
    Ok([
        parse_f64(line, &args[0])?,
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
    ])
}

fn parse_sgroup(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "SGROUP") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(1);
    };
    parse_i32(line, value)
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

fn parse_scf(input: &FeffInput) -> Result<Option<Scf>> {
    let Some(line) = card_by_feff_name(input, "SCF") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "SCF requires a radius"));
    };
    let mut iterations = parse_optional_i32(line, args.get(2))?.unwrap_or(100);
    if iterations <= 0 || iterations > 100 {
        iterations = 100;
    }
    let mut ca = parse_optional_f64(line, args.get(3))?.unwrap_or(0.2);
    if ca < 0.0 {
        ca = 0.0;
    }
    let mut nmix = parse_optional_i32(line, args.get(4))?.unwrap_or(1);
    if nmix <= 0 {
        nmix = 1;
    } else if nmix > 30 {
        nmix = 30;
    }
    let mut ecv = parse_optional_f64(line, args.get(5))?.unwrap_or(-40.0);
    if ecv >= 0.0 {
        ecv = -40.0;
    }

    Ok(Some(Scf {
        radius: parse_f64(line, radius)?,
        lfms: parse_optional_i32(line, args.get(1))?.unwrap_or(0).min(1),
        iterations,
        ca,
        nmix,
        ecv,
        icoul: parse_optional_i32(line, args.get(6))?.unwrap_or(0),
    }))
}

fn parse_exchange(input: &FeffInput) -> Result<Option<Exchange>> {
    let Some(line) = card_by_feff_name(input, "EXCHANGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "EXCHANGE requires ixc, vr0, and vi0"));
    }

    Ok(Some(Exchange {
        ixc: parse_i32(line, &args[0])?,
        vr0: parse_f64(line, &args[1])?,
        vi0: parse_f64(line, &args[2])?,
        ixc0: parse_optional_i32(line, args.get(3))?,
    }))
}

fn parse_exafs(input: &FeffInput) -> Result<Option<Exafs>> {
    let Some(line) = card_by_feff_name(input, "EXAFS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(xkmax) = args.first() else {
        return Err(parse_error(line, "EXAFS requires an xkmax value"));
    };

    Ok(Some(Exafs {
        xkmax: parse_f64(line, xkmax)?,
    }))
}

fn parse_spectrum_grid(
    input: &FeffInput,
    exchange: Option<&Exchange>,
    ispec: i32,
) -> Result<SpectrumGrid> {
    let mut grid = SpectrumGrid {
        ixc0: exchange
            .and_then(|exchange| exchange.ixc0)
            .filter(|ixc0| *ixc0 >= 0)
            .unwrap_or_else(|| if (1..=4).contains(&ispec) { 2 } else { 0 }),
        ..SpectrumGrid::default()
    };

    if let Some((name, line)) = ["XANES", "DANES", "ELNES"]
        .into_iter()
        .find_map(|name| card_by_feff_name(input, name).map(|line| (name, line)))
    {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            grid.xkstep = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep < 0.01 {
            grid.xkstep = 0.01;
        }
        if matches!(name, "XANES" | "ELNES") {
            if grid.xkstep > 2.0 {
                grid.xkstep = 0.5;
            }
            if grid.xkmax.abs() < 2.0 {
                grid.xkmax = 2.0;
            }
            if grid.xkmax.abs() > 200.0 {
                grid.xkmax = 200.0;
            }
        } else if grid.xkmax < 2.0 {
            grid.xkmax = 2.0;
        }
    } else if let Some(line) = card_by_feff_name(input, "XES") {
        let args = card_args(line)?;
        grid.xkstep = 0.01;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            grid.xkstep = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep <= grid.xkmax {
            grid.xkstep = 0.01;
        }
        if grid.xkmax >= 0.0 {
            grid.xkmax = -40.0;
        }
    } else if let Some(line) = card_by_feff_name(input, "FPRIME") {
        let args = card_args(line)?;
        if args.len() < 2 {
            return Err(parse_error(line, "FPRIME requires emin and emax"));
        }
        grid.xkmax = parse_f64(line, &args[0])?;
        grid.xkstep = parse_f64(line, &args[1])?;
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep < grid.xkmax {
            grid.xkstep = grid.xkmax;
        }
    } else if let Some(line) =
        card_by_feff_name(input, "EXAFS").or_else(|| card_by_feff_name(input, "EXELFS"))
    {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
    }

    Ok(grid)
}

fn parse_temp(input: &FeffInput) -> Result<(f64, i32)> {
    let mut temperature = 0.0;
    let mut iscfxc = 11;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("TEMP") => {
                let args = card_args(line)?;
                if let Some(value) = args.first() {
                    temperature = parse_f64(line, value)?;
                    if !temperature.is_finite() {
                        return Err(parse_error(line, "TEMP value must be finite"));
                    }
                }
                if let Some(value) = args.get(1) {
                    iscfxc = parse_i32(line, value)?;
                }
            }
            Some("SCXC") => {
                let args = card_args(line)?;
                let Some(value) = args.first() else {
                    return Err(parse_error(line, "SCXC requires iscfxc"));
                };
                iscfxc = parse_i32(line, value)?;
                validate_scxc(line, iscfxc)?;
            }
            _ => {}
        }
    }

    Ok((temperature, iscfxc))
}

fn validate_scxc(line: &FeffLine, iscfxc: i32) -> Result<()> {
    if matches!(iscfxc, 11 | 12 | 21 | 22) {
        Ok(())
    } else {
        Err(parse_error(
            line,
            "SCXC iscfxc must be one of 11, 12, 21, or 22",
        ))
    }
}

fn parse_criteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = card_by_feff_name(input, "CRITERIA") else {
        return Ok((4.0, 2.5));
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "CRITERIA requires critcw and critpw"));
    }
    Ok((parse_f64(line, &args[0])?, parse_f64(line, &args[1])?))
}

fn parse_pcriteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = card_by_feff_name(input, "PCRITERIA") else {
        return Ok((0.0, 0.0));
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "PCRITERIA requires pcritk and pcrith"));
    }
    Ok((parse_f64(line, &args[0])?, parse_f64(line, &args[1])?))
}

fn parse_lreal(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "RPHASES").is_some() {
        2
    } else {
        i32::from(card_by_feff_name(input, "RSIGMA").is_some())
    }
}

fn parse_iorder(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "IORD") else {
        return Ok(2);
    };
    let args = card_args(line)?;
    parse_optional_i32(line, args.first()).map(|value| value.unwrap_or(0))
}

fn parse_mpse(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = card_by_feff_name(input, "MPSE") else {
        return Ok((0, 100));
    };
    let args = card_args(line)?;
    let mut i_plsmn = parse_optional_i32(line, args.first())?.unwrap_or(1);
    if i_plsmn == 4 {
        i_plsmn = 1;
    }
    let n_poles = parse_optional_i32(line, args.get(1))?.unwrap_or(100);
    Ok((i_plsmn, n_poles))
}

fn parse_ispec(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "COMPTON").is_some() || card_by_feff_name(input, "DENS").is_some() {
        5
    } else if card_by_feff_name(input, "FPRIME").is_some() {
        4
    } else if card_by_feff_name(input, "DANES").is_some() {
        3
    } else if card_by_feff_name(input, "XES").is_some() {
        2
    } else if card_by_feff_name(input, "XANES").is_some()
        || card_by_feff_name(input, "ELNES").is_some()
        || card_by_feff_name(input, "NRIXS").is_some()
    {
        1
    } else {
        0
    }
}

fn parse_ipol(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "XMCD").is_some() {
        2
    } else if card_by_feff_name(input, "POLARIZATION").is_some() {
        1
    } else {
        0
    }
}

fn parse_multipole(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = card_by_feff_name(input, "MULT") else {
        return Ok((0, 0));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_i32(line, args.first())?.unwrap_or(0),
        parse_optional_i32(line, args.get(1))?.unwrap_or(0),
    ))
}

fn parse_polarization_vector(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = card_by_feff_name(input, "POLARIZATION") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "POLARIZATION requires x, y, and z"));
    }
    Ok([
        parse_f64(line, &args[0])?,
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
    ])
}

fn parse_ellipticity(input: &FeffInput) -> Result<(f64, [f64; 3])> {
    let Some(line) = card_by_feff_name(input, "ELLIPTICITY") else {
        return Ok((0.0, [0.0; 3]));
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "ELLIPTICITY requires ellipticity and incident direction",
        ));
    }
    Ok((
        parse_f64(line, &args[0])?,
        [
            parse_f64(line, &args[1])?,
            parse_f64(line, &args[2])?,
            parse_f64(line, &args[3])?,
        ],
    ))
}

fn parse_spin(input: &FeffInput) -> Result<(i32, [f64; 3])> {
    let Some(line) = card_by_feff_name(input, "SPIN") else {
        return Ok((0, [0.0; 3]));
    };
    let args = card_args(line)?;
    let spin = parse_optional_i32(line, args.first())?.unwrap_or(0);
    let default_vector = if spin == 0 { [0.0; 3] } else { [0.0, 0.0, 1.0] };
    Ok((
        spin,
        [
            parse_optional_f64(line, args.get(1))?.unwrap_or(default_vector[0]),
            parse_optional_f64(line, args.get(2))?.unwrap_or(default_vector[1]),
            parse_optional_f64(line, args.get(3))?.unwrap_or(default_vector[2]),
        ],
    ))
}

fn parse_nohole(input: &FeffInput) -> Result<i32> {
    if let (Some(corehole), Some(_)) = (
        card_by_feff_name(input, "COREHOLE"),
        card_by_feff_name(input, "NOHOLE"),
    ) {
        return Err(parse_error(
            corehole,
            "NOHOLE and COREHOLE cards are mutually exclusive",
        ));
    }

    if let Some(line) = card_by_feff_name(input, "COREHOLE") {
        let args = card_args(line)?;
        let Some(mode) = args.first() else {
            return Ok(-1);
        };
        return match mode.to_ascii_uppercase().as_str() {
            "NONE" => Ok(0),
            "RPA" => Ok(2),
            "FSR" | "REGULAR" => Ok(-1),
            _ => Err(parse_error(
                line,
                "COREHOLE must be NONE, RPA, FSR, or REGULAR",
            )),
        };
    }

    if let Some(line) = card_by_feff_name(input, "NOHOLE") {
        let args = card_args(line)?;
        return parse_optional_i32(line, args.first()).map(|value| value.unwrap_or(0));
    }

    Ok(-1)
}

fn parse_fms(input: &FeffInput) -> Result<Option<Fms>> {
    let Some(line) = card_by_feff_name(input, "FMS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "FMS requires a radius"));
    };

    let radius = parse_f64(line, radius)?;
    let lfms = parse_optional_i32(line, args.get(1))?.unwrap_or(0).min(1);
    let rdirec = parse_optional_f64(line, args.get(5))?.unwrap_or(2.0 * radius);

    Ok(Some(Fms {
        radius,
        lfms,
        minv: parse_optional_i32(line, args.get(2))?.unwrap_or(0),
        toler1: parse_optional_f64(line, args.get(3))?.unwrap_or(0.001),
        toler2: parse_optional_f64(line, args.get(4))?.unwrap_or(0.001),
        rdirec: if rdirec < 0.0 || rdirec > 2.0 * radius {
            2.0 * radius
        } else {
            rdirec
        },
    }))
}

fn parse_crpa(input: &FeffInput) -> Result<Crpa> {
    let Some(line) = card_by_feff_name(input, "CRPA") else {
        return Ok(Crpa::default());
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "CRPA requires l and rcut values"));
    }
    Ok(Crpa {
        enabled: true,
        l: parse_i32(line, &args[0])?,
        rcut: parse_f64(line, &args[1])?,
    })
}

fn parse_compton(input: &FeffInput) -> Result<Compton> {
    let mut compton = Compton::default();

    if let Some(line) = card_by_feff_name(input, "COMPTON") {
        let args = card_args(line)?;
        compton.do_compton = true;
        if let Some(value) = args.first() {
            compton.pqmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            compton.npq = parse_i32(line, value)?;
        }
        if parse_optional_i32(line, args.get(2))?.unwrap_or(0) > 0 {
            compton.force_jzzp = true;
        }
    }

    compton.do_rhozzp = card_by_feff_name(input, "RHOZZP").is_some();

    if let Some(line) = card_by_feff_name(input, "CGRID") {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            compton.zpmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            compton.ns = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(2) {
            compton.nphi = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(3) {
            compton.nz = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(4) {
            compton.nzp = parse_i32(line, value)?;
        }
    }

    Ok(compton)
}

fn parse_hubbard(input: &FeffInput) -> Result<Hubbard> {
    let Some(line) = card_by_feff_name(input, "HUBBARD") else {
        return Ok(Hubbard::default());
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "HUBBARD requires U, J, fermi_shift, and l values",
        ));
    }
    Ok(Hubbard {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: parse_f64(line, &args[0])?,
        j: parse_f64(line, &args[1])?,
        fermi_shift: parse_f64(line, &args[2])?,
        l: parse_i32(line, &args[3])?,
    })
}

fn parse_eels(input: &FeffInput) -> Result<Eels> {
    let magic_energy = parse_magic_energy(input)?;
    let (section, section_line) = if let Some(line) = card_by_feff_name(input, "ELNES") {
        ("ELNES", line)
    } else if let Some(line) = card_by_feff_name(input, "EXELFS") {
        ("EXELFS", line)
    } else {
        return Ok(Eels::default());
    };

    let rows = input.section_rows(section).collect::<Vec<_>>();
    let mut eels = Eels {
        enabled: true,
        ..Eels::default()
    };

    let line = required_eels_row(
        section_line,
        &rows,
        0,
        format!("{section} requires beam-energy row"),
    )?;
    let fields = section_fields_before_star(line)?;
    let Some(beam_energy) = fields.first() else {
        return Err(parse_error(
            line,
            format!("{section} beam row requires beam energy"),
        ));
    };
    eels.beam_energy = parse_f64(line, beam_energy)? * 1000.0;
    if let Some(value) = fields.get(1) {
        eels.average = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(2) {
        eels.cross_terms = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(3) {
        eels.relativistic = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(4) {
        eels.input = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(5) {
        eels.spectrum_column = parse_i32(line, value)?;
    }

    let mut row_index = 1;
    if eels.average != 1 {
        let line = required_eels_row(
            section_line,
            &rows,
            row_index,
            format!("{section} requires beam-direction row"),
        )?;
        let fields = section_fields_before_star(line)?;
        if fields.len() < 3 {
            return Err(parse_error(
                line,
                format!("{section} beam-direction row requires x, y, and z"),
            ));
        }
        let mut vector = [
            parse_f64(line, fields[0])?,
            parse_f64(line, fields[1])?,
            parse_f64(line, fields[2])?,
        ];
        let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        eels.beam_direction = vector;
        row_index += 1;
    }

    let line = required_eels_row(
        section_line,
        &rows,
        row_index,
        format!("{section} requires collection-angle row"),
    )?;
    let fields = section_fields_before_star(line)?;
    if fields.len() < 2 {
        return Err(parse_error(
            line,
            format!("{section} collection row requires collection and convergence angles"),
        ));
    }
    eels.collection_angle = parse_f64(line, fields[0])? / 1000.0;
    eels.convergence_angle = parse_f64(line, fields[1])? / 1000.0;
    row_index += 1;

    let line = required_eels_row(
        section_line,
        &rows,
        row_index,
        format!("{section} requires q-mesh row"),
    )?;
    let fields = section_fields_before_star(line)?;
    if fields.len() < 2 {
        return Err(parse_error(
            line,
            format!("{section} q-mesh row requires radial and angular values"),
        ));
    }
    eels.qmesh_radial = parse_i32(line, fields[0])?;
    eels.qmesh_angular = parse_i32(line, fields[1])?;
    row_index += 1;

    let line = required_eels_row(
        section_line,
        &rows,
        row_index,
        format!("{section} requires detector row"),
    )?;
    let fields = section_fields_before_star(line)?;
    if fields.len() < 2 {
        return Err(parse_error(
            line,
            format!("{section} detector row requires detector angles"),
        ));
    }
    eels.detector = [
        parse_f64(line, fields[0])? / 1000.0,
        parse_f64(line, fields[1])? / 1000.0,
    ];

    if let Some(magic_energy) = magic_energy {
        eels.magic = 1;
        eels.magic_energy = magic_energy;
    }

    if eels.average == 1 {
        eels.polarization_min = 10;
        eels.polarization_step = 1;
        eels.polarization_max = 10;
    } else {
        eels.polarization_min = 1;
        eels.polarization_step = if eels.cross_terms == 1 { 1 } else { 4 };
        eels.polarization_max = 9;
    }

    Ok(eels)
}

fn required_eels_row<'a>(
    section_line: &FeffLine,
    rows: &'a [&'a FeffLine],
    index: usize,
    message: String,
) -> Result<&'a FeffLine> {
    rows.get(index)
        .copied()
        .ok_or_else(|| parse_error(section_line, message))
}

fn parse_magic_energy(input: &FeffInput) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, "MAGIC") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "MAGIC requires emagic"));
    };
    Ok(Some(parse_f64(line, value)?))
}

fn parse_nrixs(input: &FeffInput) -> Result<Option<Nrixs>> {
    let Some(line) = card_by_feff_name(input, "NRIXS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(raw_nq) = args.first() else {
        return Err(parse_error(line, "NRIXS requires nq"));
    };
    let raw_nq = parse_i32(line, raw_nq)?;
    if raw_nq == i32::MIN {
        return Err(parse_error(line, "NRIXS q-vector count is too negative"));
    }
    let qaverage = raw_nq < 0;
    let nq = raw_nq.abs().max(1);
    let q_count = nq as usize;
    let mut q_vectors = Vec::with_capacity(q_count);
    q_vectors.push(parse_nrixs_card_vector(line, args, qaverage, nq > 1)?);
    for row in input.section_rows("NRIXS").take(q_count.saturating_sub(1)) {
        q_vectors.push(parse_nrixs_section_vector(row, qaverage)?);
    }
    if q_vectors.len() != q_count {
        return Err(parse_error(
            line,
            format!(
                "NRIXS nq={nq} requires {} q-vector rows after the card",
                q_count.saturating_sub(1)
            ),
        ));
    }
    let first = q_vectors[0];
    let ldecmx = match parse_scalar_card(input, "LDEC")? {
        Some(value) => value,
        None => parse_scalar_card(input, "LDECMX")?.unwrap_or(-1.0),
    };
    Ok(Some(Nrixs {
        nq,
        qaverage,
        qvec: first.vector,
        qnorm: first.norm,
        q_vectors,
        ldecmx: ldecmx as i32,
        lj: parse_scalar_card(input, "LJMAX")?.unwrap_or(0.0) as i32,
    }))
}

fn parse_nrixs_card_vector(
    line: &FeffLine,
    args: &[String],
    qaverage: bool,
    require_weight: bool,
) -> Result<NrixsQVector> {
    if qaverage {
        let Some(qz) = args.get(1) else {
            return Err(parse_error(line, "NRIXS q-average card requires nq and q"));
        };
        let qz = parse_nrixs_qaverage_magnitude(line, qz)?;
        let weight = if require_weight {
            parse_nrixs_weight(line, args.get(2), args.get(3), true)?
        } else {
            [1.0, 0.0]
        };
        return Ok(NrixsQVector {
            vector: [0.0, 0.0, qz],
            norm: qz,
            weight,
        });
    }

    if args.len() < 4 {
        return Err(parse_error(line, "NRIXS card requires nq qx qy qz"));
    }
    let vector = [
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
        parse_f64(line, &args[3])?,
    ];
    let weight = if require_weight {
        parse_nrixs_weight(line, args.get(4), args.get(5), true)?
    } else {
        [1.0, 0.0]
    };
    Ok(NrixsQVector {
        vector,
        norm: vector[0].hypot(vector[1]).hypot(vector[2]),
        weight,
    })
}

fn parse_nrixs_section_vector(line: &FeffLine, qaverage: bool) -> Result<NrixsQVector> {
    let fields = section_fields_before_star(line)?;
    if qaverage {
        if fields.len() < 2 {
            return Err(parse_error(
                line,
                "NRIXS q-average row requires q and weight",
            ));
        }
        let qz = parse_nrixs_qaverage_magnitude(line, fields[0])?;
        let weight =
            parse_nrixs_weight(line, fields.get(1).copied(), fields.get(2).copied(), true)?;
        return Ok(NrixsQVector {
            vector: [0.0, 0.0, qz],
            norm: qz,
            weight,
        });
    }

    if fields.len() < 4 {
        return Err(parse_error(
            line,
            "NRIXS q-vector row requires qx qy qz and weight",
        ));
    }
    let vector = [
        parse_f64(line, fields[0])?,
        parse_f64(line, fields[1])?,
        parse_f64(line, fields[2])?,
    ];
    let weight = parse_nrixs_weight(line, fields.get(3).copied(), fields.get(4).copied(), true)?;
    Ok(NrixsQVector {
        vector,
        norm: vector[0].hypot(vector[1]).hypot(vector[2]),
        weight,
    })
}

fn parse_nrixs_qaverage_magnitude(line: &FeffLine, value: &str) -> Result<f64> {
    let qz = parse_f64(line, value)?;
    if qz <= 0.0 {
        return Err(parse_error(
            line,
            "NRIXS q-average magnitude must be positive",
        ));
    }
    Ok(qz)
}

fn parse_nrixs_weight(
    line: &FeffLine,
    real: Option<&String>,
    imaginary: Option<&String>,
    require_real: bool,
) -> Result<[f64; 2]> {
    if require_real {
        let Some(real) = real else {
            return Err(parse_error(line, "NRIXS q weight is required when nq > 1"));
        };
        return Ok([
            parse_f64(line, real)?,
            parse_optional_f64(line, imaginary)?.unwrap_or(0.0),
        ]);
    }
    Ok([
        parse_optional_f64(line, real)?.unwrap_or(1.0),
        parse_optional_f64(line, imaginary)?.unwrap_or(0.0),
    ])
}

fn parse_mdff(input: &FeffInput, nrixs: &mut Option<Nrixs>, eels: &Eels) -> Result<Mdff> {
    let Some(line) = card_by_feff_name(input, "MDFF") else {
        return Ok(default_mdff());
    };
    let args = card_args(line)?;
    let imdff = parse_optional_i32(line, args.first())?.unwrap_or(1);
    let mut mdff = Mdff {
        imdff,
        qqmdff: -1.0,
        cosmdff_angle: 0.0,
    };

    if imdff == 2 {
        match args.len() {
            1 => {}
            3 => {
                mdff.qqmdff = parse_f64(line, &args[1])?;
                mdff.cosmdff_angle = parse_f64(line, &args[2])?;
            }
            _ => {
                return Err(parse_error(
                    line,
                    "MDFF 2 requires either no q-prime parameters or qqmdff cosmdff",
                ));
            }
        }
    }

    match imdff {
        value if value <= 0 => Ok(default_mdff()),
        1 => {
            if nrixs.is_some() {
                Ok(mdff)
            } else {
                Err(parse_error(line, "MDFF 1 requires NRIXS"))
            }
        }
        2 => {
            let Some(nrixs) = nrixs.as_mut() else {
                return Err(parse_error(line, "MDFF 2 requires NRIXS"));
            };
            if nrixs.nq != 2 {
                return Err(parse_error(line, "MDFF 2 requires NRIXS nq=2"));
            }
            if mdff.qqmdff >= 0.0 {
                apply_generated_mdff_qprime(line, &mdff, nrixs)?;
            }
            Ok(mdff)
        }
        3 => {
            if eels.enabled {
                Ok(mdff)
            } else {
                Err(parse_error(line, "MDFF 3 requires ELNES or EXELFS"))
            }
        }
        _ => Err(parse_error(line, "MDFF selector must be 0, 1, 2, or 3")),
    }
}

fn apply_generated_mdff_qprime(line: &FeffLine, mdff: &Mdff, nrixs: &mut Nrixs) -> Result<()> {
    if nrixs.q_vectors.len() < 2 {
        return Err(parse_error(line, "MDFF 2 requires two NRIXS q-vector rows"));
    }
    let first = nrixs.q_vectors[0];
    if first.norm == 0.0 {
        return Err(parse_error(
            line,
            "MDFF 2 generated q-prime requires a nonzero first q-vector",
        ));
    }
    let scale = mdff.qqmdff / first.norm;
    let angle = mdff.cosmdff_angle.to_radians();
    let (sin_angle, cos_angle) = angle.sin_cos();
    nrixs.q_vectors[1].vector = [
        first.vector[0] * scale,
        scale * (first.vector[1] * cos_angle + first.vector[2] * sin_angle),
        scale * (-first.vector[1] * sin_angle + first.vector[2] * cos_angle),
    ];
    Ok(())
}

fn default_mdff() -> Mdff {
    Mdff {
        imdff: 0,
        qqmdff: -1.0,
        cosmdff_angle: 0.0,
    }
}

fn parse_rixs(input: &FeffInput) -> Result<Rixs> {
    let mut rixs = Rixs::default();

    if let Some(line) = card_by_feff_name(input, "EDGE") {
        let args = card_args(line)?;
        if let Some(edge) = args.first() {
            rixs.edges.clear();
            rixs.edges.push(edge.to_ascii_uppercase());
            for edge in args.iter().skip(1) {
                let edge = edge.to_ascii_uppercase();
                let is_valence = edge == "VAL";
                rixs.edges.push(edge);
                if is_valence {
                    rixs.mbconv = true;
                    break;
                }
            }
        }
    }

    if let Some(line) = card_by_feff_name(input, "RIXS") {
        let args = card_args(line)?;
        rixs.run = true;
        rixs.gamma_exp[0] = parse_optional_f64(line, args.first())?;
        rixs.gamma_exp[1] = parse_optional_f64(line, args.get(1))?;
        rixs.xmu = parse_optional_f64(line, args.get(2))?;
    }

    Ok(rixs)
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

fn card_args(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::Card { args, .. } => Ok(args),
        LineKind::SectionData { .. } => Err(parse_error(line, "expected card line")),
    }
}

fn section_fields(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::SectionData { fields, .. } => Ok(fields),
        LineKind::Card { .. } => Err(parse_error(line, "expected section data line")),
    }
}

fn section_fields_before_star(line: &FeffLine) -> Result<Vec<&String>> {
    Ok(section_fields(line)?
        .iter()
        .take_while(|field| field.as_str() != "*")
        .collect())
}

fn parse_i32(line: &FeffLine, value: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .map_err(|_| parse_error(line, format!("invalid integer {value:?}")))
}

fn parse_optional_i32(line: &FeffLine, value: Option<&String>) -> Result<Option<i32>> {
    value.map(|value| parse_i32(line, value)).transpose()
}

fn parse_logical(line: &FeffLine, value: &str) -> Result<bool> {
    let normalized = value.trim_matches('.').to_ascii_uppercase();
    match normalized.as_str() {
        "T" | "TRUE" | "1" => Ok(true),
        "F" | "FALSE" | "0" => Ok(false),
        _ => Err(parse_error(line, format!("invalid logical {value:?}"))),
    }
}

fn parse_f64(line: &FeffLine, value: &str) -> Result<f64> {
    value
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error(line, format!("invalid float {value:?}")))
}

fn parse_optional_f64(line: &FeffLine, value: Option<&String>) -> Result<Option<f64>> {
    value.map(|value| parse_f64(line, value)).transpose()
}

fn parse_error(line: &FeffLine, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: line.location.path.clone(),
        line: line.location.line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;
