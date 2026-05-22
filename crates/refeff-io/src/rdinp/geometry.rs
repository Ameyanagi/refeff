//! Geometry ordering helpers for FEFF `rdinp` structural handoff files.

use crate::Result;
use crate::model::FeffDocument;
use refeff_core::rotate_into_reference_frame;

use super::helpers::absorber_index;

#[derive(Debug, Clone)]
pub(super) struct GeometryRow {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) z: f64,
    pub(super) ipot: i32,
    pub(super) distance2: f64,
}

pub(super) fn geometry_rows(document: &FeffDocument) -> Result<Vec<GeometryRow>> {
    let absorber_index = absorber_index(document)?;
    let absorber = &document.atoms[absorber_index];
    let mut rows = Vec::with_capacity(document.atoms.len());
    rows.push(GeometryRow {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        ipot: 0,
        distance2: 0.0,
    });

    for (input_index, atom) in document.atoms.iter().enumerate() {
        if input_index == absorber_index {
            continue;
        }
        let x = atom.x - absorber.x;
        let y = atom.y - absorber.y;
        let z = atom.z - absorber.z;
        let distance2 = feff_geometry_distance2(x, y, z);
        if distance2 > 0.0 {
            rows.push(GeometryRow {
                x,
                y,
                z,
                ipot: atom.ipot,
                distance2,
            });
        }
    }

    feff_selection_sort_by_distance(&mut rows);
    if let Some(nrixs) = document.nrixs.as_ref() {
        for row in &mut rows {
            let [x, y, z] = rotate_into_reference_frame([row.x, row.y, row.z], nrixs.qvec);
            row.x = x;
            row.y = y;
            row.z = z;
        }
    }
    Ok(rows)
}

fn feff_selection_sort_by_distance(rows: &mut [GeometryRow]) {
    // FEFF's ffsort uses a strict less-than selection sort. That matters for
    // equal-distance shells because swaps can move tied atoms differently than
    // a stable distance sort.
    let mut i = 0;
    while i + 1 < rows.len() {
        let mut min_idx = i;
        let mut min_distance = rows[i].distance2;
        let mut j = i + 1;
        while j < rows.len() {
            if rows[j].distance2 < min_distance {
                min_distance = rows[j].distance2;
                min_idx = j;
            }
            j += 1;
        }
        if min_idx != i {
            rows.swap(i, min_idx);
        }
        i += 1;
    }
}

fn feff_geometry_distance2(x: f64, y: f64, z: f64) -> f64 {
    // The reference FEFF10 binaries are optimized, so equal-shell ordering
    // follows the fused arithmetic generated for the ffsort distance sum.
    z.mul_add(z, x.mul_add(x, y * y))
}

pub(super) fn geometry_model_atoms(document: &FeffDocument, rows: &[GeometryRow]) -> Vec<usize> {
    let max_potential = document
        .potentials
        .iter()
        .map(|potential| potential.ipot.max(0) as usize)
        .max()
        .unwrap_or_else(|| {
            rows.iter()
                .map(|row| row.ipot.max(0) as usize)
                .max()
                .unwrap_or(0)
        });
    let mut iatph = vec![0_usize; max_potential + 1];
    if !iatph.is_empty() {
        iatph[0] = 1;
    }
    for (iph, model_atom) in iatph.iter_mut().enumerate().take(max_potential + 1).skip(1) {
        if let Some((idx, _)) = rows
            .iter()
            .enumerate()
            .find(|(_, row)| row.ipot == iph as i32)
        {
            *model_atom = idx + 1;
        }
    }
    while iatph.len() > 1 && iatph.last() == Some(&0) {
        iatph.pop();
    }
    iatph
}
