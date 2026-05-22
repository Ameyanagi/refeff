use super::*;

/// Port of FEFF `XSPH/getholeorb0.f90`.
///
/// The surrounding FEFF routine first calls `getorb` to locate `iholep`, then
/// passes `dgc(:, iholep, 0)` and `dpc(:, iholep, 0)` through this interpolation
/// step. This pure helper accepts those selected components directly, finds the
/// last source sample above FEFF's `1e-11` tail cutoff, interpolates with
/// FEFF-compatible cubic `terp`, and zero-fills the output tail after `jnew`.
pub fn xsph_initial_hole_orbital(
    input: XsphHoleOrbitalInput<'_>,
) -> Result<XsphHoleOrbital, XsphError> {
    validate_finite_real("original_step", input.original_step)?;
    validate_finite_real("new_step", input.new_step)?;
    if input.large_component.len() != input.small_component.len() {
        return Err(XsphError::HoleOrbitalLengthMismatch {
            large_len: input.large_component.len(),
            small_len: input.small_component.len(),
        });
    }
    if input.output_count > input.output_capacity {
        return Err(XsphError::InvalidHoleOrbitalOutputCount {
            output_count: input.output_count,
            output_capacity: input.output_capacity,
        });
    }

    for (&large, &small) in input
        .large_component
        .iter()
        .zip(input.small_component.iter())
    {
        validate_finite_real("large_component", large)?;
        validate_finite_real("small_component", small)?;
    }

    let last_nonzero = input
        .large_component
        .iter()
        .zip(input.small_component.iter())
        .rposition(|(&large, &small)| {
            large.abs() >= XSPH_HOLE_ORBITAL_TAIL_CUTOFF
                || small.abs() >= XSPH_HOLE_ORBITAL_TAIL_CUTOFF
        })
        .ok_or(XsphError::EmptyHoleOrbital)?;
    let source_count = last_nonzero
        .saturating_add(2)
        .min(input.large_component.len());
    let source_x: Vec<_> = (0..source_count)
        .map(|index| -XSPH_HOLE_ORBITAL_X0 + index as Real * input.original_step)
        .collect();
    let large_source: Vec<_> = input
        .large_component
        .iter()
        .take(source_count)
        .copied()
        .collect();
    let small_source: Vec<_> = input
        .small_component
        .iter()
        .take(source_count)
        .copied()
        .collect();

    let mut large_component = Array1::<Real>::zeros(input.output_capacity);
    let mut small_component = Array1::<Real>::zeros(input.output_capacity);
    for index in 0..input.output_count {
        let x = -XSPH_HOLE_ORBITAL_X0 + index as Real * input.new_step;
        large_component[index] = terp(&source_x, &large_source, 3, x)?.value;
        small_component[index] = terp(&source_x, &small_source, 3, x)?.value;
    }

    Ok(XsphHoleOrbital {
        large_component,
        small_component,
        active_count: input.output_count,
        source_count,
    })
}
