use super::*;

pub(super) fn write_pdos_poles_sidecar(work_dir: &Path, section: &DmdwOutSection) -> Result<()> {
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

pub(super) fn write_pdos_rect_sidecar(
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

pub(super) fn write_pdos_gaussian_sidecar(
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
