//! Shared `rdinp` formatting and FEFF handoff helpers.

use crate::model::{Atom, FeffDocument, Potential, SingleScatteringPath};
use crate::{IoError, Result};
use refeff_core::{atomic_symbol, core_hole_width_ev, edge_index};

pub(super) fn write_i4_list(
    out: &mut impl std::fmt::Write,
    values: impl IntoIterator<Item = i32>,
) -> Result<()> {
    for value in values {
        write!(out, "{value:4}")?;
    }
    writeln!(out)?;
    Ok(())
}

pub(super) fn fortran_bool(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

pub(super) fn control_flag(document: &FeffDocument, index: usize, default: i32) -> i32 {
    document
        .control
        .and_then(|control| control.get(index).copied())
        .unwrap_or(default)
}

pub(super) fn should_write_single_scattering_paths(document: &FeffDocument) -> bool {
    !document.single_scattering_paths.is_empty() && control_flag(document, 3, 1) != 0
}

pub(super) fn fms_flag(document: &FeffDocument) -> i32 {
    if uses_overlap_geometry(document) {
        0
    } else {
        control_flag(document, 2, 1)
    }
}

pub(super) fn do_fms_flag(document: &FeffDocument) -> i32 {
    if uses_overlap_geometry(document) {
        0
    } else {
        i32::from(document.fms.is_some())
    }
}

pub(super) fn path_flag(document: &FeffDocument) -> i32 {
    if uses_overlap_geometry(document) {
        0
    } else {
        control_flag(document, 3, 1)
    }
}

pub(super) fn path_ms_flag(document: &FeffDocument) -> i32 {
    i32::from(!uses_overlap_geometry(document))
}

fn uses_overlap_geometry(document: &FeffDocument) -> bool {
    !document.overlap_shells.is_empty() && document.atoms.is_empty()
}

pub(super) fn automatic_folp_flag(document: &FeffDocument) -> i32 {
    if document.overlap_factors.is_empty() {
        0
    } else {
        -1
    }
}

pub(super) fn potential_overlap_factor(document: &FeffDocument, ipot: i32) -> f64 {
    document
        .overlap_factors
        .iter()
        .rev()
        .find(|factor| factor.potential_index == ipot)
        .map(|factor| factor.factor)
        .unwrap_or_else(|| {
            if document.overlap_factors.is_empty() {
                document.afolp
            } else {
                1.0
            }
        })
}

pub(super) fn potential_ionization(document: &FeffDocument, ipot: i32) -> f64 {
    document
        .ionizations
        .iter()
        .rev()
        .find(|ionization| ionization.potential_index == ipot)
        .map(|ionization| ionization.value)
        .unwrap_or(0.0)
}

pub(super) fn print_flag(document: &FeffDocument, index: usize, default: i32) -> i32 {
    document
        .print
        .and_then(|print| print.get(index).copied())
        .unwrap_or(default)
}

pub(super) fn debye_values(document: &FeffDocument) -> (f64, f64, i32) {
    document
        .debye
        .as_ref()
        .map(|debye| (debye.temperature, debye.debye_temperature, debye.idwopt))
        .unwrap_or((0.0, 0.0, -1))
}

pub(super) fn output_ispec(document: &FeffDocument) -> i32 {
    if document.ipol == 2 && document.spin != 0 {
        -1
    } else {
        document.ispec
    }
}

pub(super) fn dimensions_values(document: &FeffDocument) -> Result<(usize, i32, i32, i32)> {
    const NCLUSX_HARD_LIMIT: usize = 3000;
    const LX_HARD_LIMIT: i32 = 20;
    const NPHX_HARD_LIMIT: i32 = 31;

    let mut nclusx = dimension_cluster_size(document)?;
    if let Some(limit) = document.dims.map(|dims| dims.nclusx) {
        if limit > 0 {
            nclusx = nclusx.min(limit as usize);
        } else {
            nclusx = nclusx.min(NCLUSX_HARD_LIMIT);
        }
    } else {
        nclusx = nclusx.min(NCLUSX_HARD_LIMIT);
    }

    let mut lx = document
        .potentials
        .iter()
        .map(|potential| lmaxsc(document, potential).max(lmaxph(potential)))
        .max()
        .unwrap_or(0);
    if let Some(limit) = document.dims.map(|dims| dims.lx) {
        if limit >= 0 {
            lx = lx.min(limit);
        } else {
            lx = lx.min(LX_HARD_LIMIT);
        }
    } else {
        lx = lx.min(LX_HARD_LIMIT);
    }

    let nphu = nph(document)?.clamp(1, NPHX_HARD_LIMIT);
    let nspu = if document.hubbard.i_hubbard >= 2
        || document.spin != 0
        || document
            .potentials
            .iter()
            .any(|potential| potential.spinph.unwrap_or(0.0) != 0.0)
    {
        2
    } else {
        1
    };
    Ok((nclusx, lx, nphu, nspu))
}

fn dimension_cluster_size(document: &FeffDocument) -> Result<usize> {
    if document.atoms.is_empty() {
        return Ok(0);
    }

    let absorber = &document.atoms[absorber_index(document)?];
    let ratmin = nearest_nonabsorber_distance(document).unwrap_or(0.0);
    let rfms1 = document
        .scf
        .as_ref()
        .map(|scf| radius_or_disabled(scf.radius, ratmin))
        .unwrap_or(-1.0);
    let rfms2 = document
        .fms
        .as_ref()
        .map(|fms| radius_or_disabled(fms.radius, ratmin))
        .unwrap_or(-1.0);
    let rmax = document.rpath.unwrap_or(-1.0);
    let rdims = rfms1.max(rfms2).max(rmax);

    if rdims <= 0.0 {
        return Ok(document.atoms.len());
    }

    Ok(document
        .atoms
        .iter()
        .filter(|atom| distance_from(absorber, atom) <= rdims)
        .count())
}

fn radius_or_disabled(radius: f64, nearest_neighbor: f64) -> f64 {
    if radius < nearest_neighbor {
        -1.0
    } else {
        radius
    }
}

pub(super) fn nph(document: &FeffDocument) -> Result<i32> {
    document
        .potentials
        .iter()
        .map(|potential| potential.ipot)
        .max()
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "POTENTIALS is empty".to_string(),
        })
}

pub(super) fn potential_for_ipot(document: &FeffDocument, ipot: i32) -> Result<&Potential> {
    document
        .potentials
        .iter()
        .find(|potential| potential.ipot == ipot)
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("missing potential {ipot}"),
        })
}

pub(super) fn potential_label(document: &FeffDocument, ipot: i32) -> Result<&str> {
    Ok(potential_for_ipot(document, ipot)?
        .tag
        .as_deref()
        .unwrap_or(""))
}

pub(super) fn overlap_shell_count(document: &FeffDocument, ipot: i32) -> usize {
    document
        .overlap_shells
        .iter()
        .filter(|shell| shell.potential_index == ipot)
        .count()
}

pub(super) fn write_overlap_shells(
    document: &FeffDocument,
    out: &mut impl std::fmt::Write,
    nph: i32,
) -> Result<()> {
    for ipot in 0..=nph {
        for shell in document
            .overlap_shells
            .iter()
            .filter(|shell| shell.potential_index == ipot)
        {
            writeln!(
                out,
                "{:5}{:5}{:13.5}",
                shell.neighbor_potential_index, shell.count, shell.distance
            )?;
        }
    }
    Ok(())
}

pub(super) fn validate_single_scattering_path(
    document: &FeffDocument,
    path: &SingleScatteringPath,
) -> Result<()> {
    if path.index < 0 {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("SS path index must be non-negative: {}", path.index),
        });
    }
    if !path.degeneracy.is_finite() || !path.distance.is_finite() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "SS degeneracy and distance must be finite".to_string(),
        });
    }
    if path.distance < 0.0 {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("SS distance must be non-negative: {}", path.distance),
        });
    }
    Ok(())
}

pub(super) fn absorber_potential(document: &FeffDocument) -> Result<&Potential> {
    potential_for_ipot(document, 0)
}

pub(super) fn absorber_label(document: &FeffDocument) -> Result<String> {
    let potential = absorber_potential(document)?;
    if let Some(tag) = potential
        .tag
        .as_deref()
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
    {
        return Ok(tag.to_string());
    }

    if let Some(z) = potential.z {
        let atomic_number = usize::try_from(z).map_err(|_| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("invalid absorbing atomic number {z}"),
        })?;
        return atomic_symbol(atomic_number)
            .map(str::to_string)
            .map_err(|err| IoError::Parse {
                path: document.source.clone(),
                line: 0,
                message: err.to_string(),
            });
    }

    let label = potential.z_token.trim();
    if label.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "absorbing potential has no label".to_string(),
        });
    }
    Ok(label.to_string())
}

pub(super) fn absorber_index(document: &FeffDocument) -> Result<usize> {
    if document.atoms.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "ATOMS is empty".to_string(),
        });
    }

    Ok(document
        .atoms
        .iter()
        .position(|atom| atom.ipot == 0)
        .unwrap_or(0))
}

fn edge_hole(label: &str) -> Result<i32> {
    edge_index(label).ok_or_else(|| IoError::Parse {
        path: Default::default(),
        line: 0,
        message: format!("unknown EDGE {label:?}"),
    })
}

pub(super) fn document_ihole(document: &FeffDocument) -> Result<i32> {
    if let Some(hole) = document.hole {
        return Ok(hole);
    }
    document
        .edge
        .as_ref()
        .map(|edge| edge_hole(&edge.label))
        .transpose()
        .map(|hole| hole.unwrap_or(1))
}

pub(super) fn fixed_title(title: &str) -> String {
    let mut out: String = title.chars().take(80).collect();
    while out.len() < 80 {
        out.push(' ');
    }
    out
}

pub(super) fn fixed_a6(value: &str) -> String {
    let mut out: String = value.chars().take(6).collect();
    while out.len() < 6 {
        out.push(' ');
    }
    out
}

pub(super) fn lmaxsc(document: &FeffDocument, potential: &Potential) -> i32 {
    let raw = raw_lmaxsc(potential);
    if !document.unfreezef && raw > 2 {
        2
    } else {
        raw
    }
}

pub(super) fn raw_lmaxsc(potential: &Potential) -> i32 {
    let default = match potential.z {
        Some(z) if z < 6 => 1,
        Some(z) if z < 55 => 2,
        Some(_) => 3,
        None => 0,
    };

    potential
        .lmax1
        .filter(|requested| *requested >= 0)
        .unwrap_or(default)
}

pub(super) fn lmaxph(potential: &Potential) -> i32 {
    let default = match potential.z {
        Some(z) if z < 6 => 2,
        Some(_) => 3,
        None => 0,
    };
    potential
        .lmax2
        .filter(|requested| *requested >= 0)
        .unwrap_or(default)
}

pub(super) fn xnatph(document: &FeffDocument, potential: &Potential) -> f64 {
    if document
        .potentials
        .iter()
        .any(|potential| potential.xnatph.is_some())
    {
        if potential.ipot == 0 {
            return 1.0;
        }

        let absorber_count = document
            .potentials
            .iter()
            .find(|potential| potential.ipot == 0)
            .and_then(|potential| potential.xnatph)
            .filter(|value| *value > 0.01)
            .unwrap_or(0.01);
        let raw = potential.xnatph.unwrap_or(0.0);
        let adjusted = if raw <= 0.0 { 1.0 } else { raw };
        return adjusted / absorber_count;
    }

    if let Some(xnatph) = potential.xnatph {
        return xnatph;
    }
    let count = document
        .atoms
        .iter()
        .filter(|atom| atom.ipot == potential.ipot)
        .count();
    if document.cfaverage.iphabs > 0 && potential.ipot == document.cfaverage.iphabs {
        return count.saturating_sub(1) as f64;
    }
    if count == 0 { 1.0 } else { count as f64 }
}

pub(super) fn interstitial_volume(document: &FeffDocument) -> f64 {
    let Some(interstitial) = document.interstitial else {
        return 0.0;
    };
    let Some(ratmin) = nearest_nonabsorber_distance(document) else {
        return 0.0;
    };
    let xnat = document
        .potentials
        .iter()
        .map(|potential| xnatph(document, potential))
        .sum::<f64>();
    interstitial.volume_scale * ratmin.powi(3) * xnat
}

pub(super) fn path_rmax(document: &FeffDocument) -> f64 {
    if let Some(rpath) = document.rpath {
        return rpath;
    }
    if document.ispec > 0 {
        return -1.0;
    }
    let Some(ratmin) = nearest_nonabsorber_distance(document) else {
        return -1.0;
    };
    let Some(absorber) = document.atoms.first() else {
        return -1.0;
    };
    let ratmax = document
        .atoms
        .iter()
        .map(|atom| distance_from(absorber, atom))
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    (2.2 * ratmin).min(1.01 * ratmax)
}

pub(super) fn spinph(document: &FeffDocument, ipot: i32) -> f64 {
    if document.hubbard.i_hubbard >= 2 {
        return 0.0;
    }
    potential_for_ipot(document, ipot)
        .ok()
        .and_then(|potential| {
            potential.spinph.or_else(|| {
                (document.spin != 0)
                    .then(|| potential.z.and_then(default_atomic_spinph))
                    .flatten()
            })
        })
        .unwrap_or(0.0)
}

pub(super) fn default_atomic_spinph(z: i32) -> Option<f64> {
    match z {
        64 => Some(7.0),
        _ => None,
    }
}

pub(super) fn nearest_nonabsorber_distance(document: &FeffDocument) -> Option<f64> {
    let origin = document.atoms.first()?;
    document
        .atoms
        .iter()
        .skip(1)
        .map(|atom| distance_from(origin, atom))
        .filter(|distance| *distance > 0.0)
        .min_by(f64::total_cmp)
}

pub(super) fn max_interatomic_distance(document: &FeffDocument) -> f64 {
    let mut max_distance = 0.0_f64;
    for (idx, atom) in document.atoms.iter().enumerate() {
        for other in document.atoms.iter().skip(idx + 1) {
            max_distance = max_distance.max(distance_from(atom, other));
        }
    }
    max_distance
}

fn core_hole_width_for_document(document: &FeffDocument, z: i32, ihole: i32) -> Result<f64> {
    core_hole_width_ev(z, ihole).map_err(|err| IoError::Parse {
        path: document.source.clone(),
        line: 0,
        message: err.to_string(),
    })
}

pub(super) fn core_hole_width_for_handoff(
    document: &FeffDocument,
    z: i32,
    ihole: i32,
) -> Result<f64> {
    let default_width = core_hole_width_for_document(document, z, ihole)?;
    Ok(match document.xsph_handoff.core_hole_width {
        Some(width) if width > 0.0 => width,
        Some(width) => default_width.min(width.abs()),
        None => default_width,
    })
}

pub(super) fn distance_from(origin: &Atom, atom: &Atom) -> f64 {
    let dx = atom.x - origin.x;
    let dy = atom.y - origin.y;
    let dz = atom.z - origin.z;
    dx.hypot(dy).hypot(dz)
}
