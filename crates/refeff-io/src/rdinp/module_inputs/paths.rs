//! PATH, GENFMT, and FF2X writers for FEFF `rdinp` handoff data.

use crate::model::{FeffDocument, SingleScatteringPath};
use crate::{IoError, Result};
use refeff_core::{normalize_vector, vector_norm};

use super::super::{
    control_flag, debye_values, fixed_a6, fortran_bool, output_ispec, path_flag, path_ms_flag,
    path_rmax, potential_label, print_flag, validate_single_scattering_path, write_i4_list,
};

/// Render FEFF-compatible `paths.inp` content from an [`FeffDocument`].
pub fn paths_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_paths_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `paths.dat` content for explicit `SS` cards.
///
/// FEFF handles `SS` cards inside `rdinp` instead of running the pathfinder:
/// each card becomes a two-leg path with the requested scatterer followed by
/// the absorber at the origin.
pub fn single_scattering_paths_dat_string(document: &FeffDocument) -> Result<String> {
    if document.single_scattering_paths.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write paths.dat without SS cards".to_string(),
        });
    }

    let mut out = String::new();
    write_single_scattering_paths_dat(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `genfmt.inp` content from an [`FeffDocument`].
pub fn genfmt_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_genfmt_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `ff2x.inp` content from an [`FeffDocument`].
pub fn ff2x_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_ff2x_inp(document, &mut out)?;
    Ok(out)
}

fn write_paths_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let rfms2 = document.fms.as_ref().map(|fms| fms.radius).unwrap_or(-1.0);
    let rmax = path_rmax(document);

    writeln!(out, "mpath, ms, nncrit, nlegxx, ipr4")?;
    write_i4_list(
        out,
        [
            path_flag(document),
            path_ms_flag(document),
            0,
            document.nleg.unwrap_or(7),
            print_flag(document, 3, 0),
        ],
    )?;
    writeln!(out, "critpw, pcritk, pcrith,  rmax, rfms2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        document.critpw, document.pcritk, document.pcrith, rmax, rfms2
    )?;
    writeln!(out, "ica")?;
    let ica = document
        .nrixs
        .as_ref()
        .map(|nrixs| if nrixs.qaverage { 5 } else { 7 })
        .unwrap_or(document.path_symmetry);
    write_i4_list(out, [ica])?;
    Ok(())
}

fn write_single_scattering_paths_dat(
    document: &FeffDocument,
    out: &mut impl std::fmt::Write,
) -> Result<()> {
    for title in &document.titles {
        writeln!(out, " {title}")?;
    }
    writeln!(
        out,
        " Single scattering paths from ss lines cards in feff input"
    )?;
    writeln!(out, " {}", "-".repeat(71))?;

    let rmax = path_rmax(document);
    for path in document
        .single_scattering_paths
        .iter()
        .filter(|path| rmax <= 0.0 || path.distance <= rmax)
    {
        write_single_scattering_path(document, path, out)?;
    }
    Ok(())
}

fn write_single_scattering_path(
    document: &FeffDocument,
    path: &SingleScatteringPath,
    out: &mut impl std::fmt::Write,
) -> Result<()> {
    validate_single_scattering_path(document, path)?;

    writeln!(
        out,
        "{:4}{:4}{:8.3}  index,nleg,degeneracy,r={:8.4}",
        path.index, 2, path.degeneracy, path.distance
    )?;
    writeln!(out, " single scattering")?;

    let scatterer_label = fixed_a6(potential_label(document, path.potential_index)?);
    writeln!(
        out,
        "{:12.6}{:12.6}{:12.6}{:4} '{}'",
        path.distance, 0.0, 0.0, path.potential_index, scatterer_label
    )?;

    let absorber_label = fixed_a6(potential_label(document, 0)?);
    writeln!(
        out,
        "{:12.6}{:12.6}{:12.6}{:4} '{}'  x,y,z,ipot",
        0.0, 0.0, 0.0, 0, absorber_label
    )?;
    Ok(())
}

fn write_genfmt_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "mfeff, ipr5, iorder, critcw, wnstar")?;
    writeln!(
        out,
        "{:4}{:4}{:8}{:13.5}{:>5}",
        control_flag(document, 4, 1),
        print_flag(document, 4, 0),
        document.iorder,
        document.critcw,
        fortran_bool(document.nstar)
    )?;
    writeln!(out, " the number of decomposi")?;
    writeln!(
        out,
        "{:5}",
        document
            .nrixs
            .as_ref()
            .map(|nrixs| nrixs.ldecmx)
            .unwrap_or(-1)
    )?;
    Ok(())
}

fn write_ff2x_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let (tk, thetad, idwopt) = debye_values(document);
    let absolu = i32::from(document.absolute || document.eels.enabled);
    let momentum_transfer = if document.eels.enabled {
        document.eels.beam_direction
    } else if let Some(nrixs) = document.nrixs.as_ref() {
        nrixs.qvec
    } else if vector_norm(document.incidence_vector) > 0.0 {
        normalize_vector(document.incidence_vector)
    } else {
        [0.0; 3]
    };

    writeln!(out, "mchi, ispec, idwopt, ipr6, mbconv, absolu, iGammaCH")?;
    write_i4_list(
        out,
        [
            control_flag(document, 5, 1),
            output_ispec(document),
            idwopt,
            print_flag(document, 5, 0),
            i32::from(document.many_body_convolution),
            absolu,
            document.xsph_handoff.core_hole_broadening,
        ],
    )?;
    writeln!(out, "vrcorr, vicorr, s02, critcw")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        document.corrections[0],
        document.corrections[1],
        document.s02.unwrap_or(1.0),
        document.critcw
    )?;
    writeln!(out, "tk, thetad, alphat, thetae, sig2g, sig_gk")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        tk,
        thetad,
        document.fine_structure_damping.alphat,
        document.fine_structure_damping.thetae,
        document.fine_structure_damping.sig2g,
        document.fine_structure_damping.sig_gk
    )?;
    writeln!(out, "momentum transfer")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        momentum_transfer[0], momentum_transfer[1], momentum_transfer[2]
    )?;
    writeln!(out, " the number of decomposi")?;
    writeln!(
        out,
        "{:5}",
        document
            .nrixs
            .as_ref()
            .map(|nrixs| nrixs.ldecmx)
            .unwrap_or(-1)
    )?;
    writeln!(out, "electronic temperature")?;
    writeln!(out, "{:13.5}", document.electronic_temperature)?;
    Ok(())
}
