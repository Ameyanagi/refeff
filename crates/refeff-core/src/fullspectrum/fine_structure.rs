//! FULLSPECTRUM FMS/path fine-structure interpolation.

use ndarray::{Array1, ArrayView1};

use crate::interpolation::{LintCache, lint_with_cache};
use crate::{Complex, Real};

use super::constants::FEFF_HARTREE_EV;
use super::types::*;
use super::validation::{validate_finite_value, validate_positive, validate_segment_len};

/// Port of `FULLSPECTRUM/rdst.f90`: combine FMS and path fine structure.
///
/// The four input segments correspond to FEFF's `fms_re`, `path_re`,
/// `fms_im`, and `path_im` `xmu.dat` files after parsing and unit conversion.
/// Values are interpolated onto `omega`; in the transition interval selected
/// from the FMS wave-number grid, FEFF mixes FMS and path values with
/// `sin(theta)^2`/`cos(theta)^2` weights.
pub fn full_spectrum_fine_structure_from_segments(
    input: FullSpectrumFineStructureInput<'_>,
) -> Result<FullSpectrumFineStructure, FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "omega" });
    }
    validate_positive("low_wave_number", input.low_wave_number)?;
    validate_positive("high_wave_number", input.high_wave_number)?;
    if input.high_wave_number <= input.low_wave_number {
        return Err(FullSpectrumError::InvalidEnergyRange {
            name: "fine_structure_wave_number",
            min: input.low_wave_number,
            max: input.high_wave_number,
        });
    }
    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("fine_structure omega", row, value)?;
        if row > 0 && value < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: value,
            });
        }
    }

    let mut real_fms = prepare_fine_structure_segment(0, input.real_fms)?;
    let mut real_path = prepare_fine_structure_segment(1, input.real_path)?;
    let mut imaginary_fms = prepare_fine_structure_segment(2, input.imaginary_fms)?;
    let mut imaginary_path = prepare_fine_structure_segment(3, input.imaginary_path)?;

    let real_transition = fine_structure_transition_interval(
        "real_fms",
        &real_fms,
        input.low_wave_number,
        input.high_wave_number,
    )?;
    let imaginary_transition = fine_structure_transition_interval(
        "imaginary_fms",
        &imaginary_fms,
        input.low_wave_number,
        input.high_wave_number,
    )?;

    let mut real_part = Array1::<Real>::zeros(input.omega.len());
    let mut imaginary_part = Array1::<Real>::zeros(input.omega.len());
    let mut real_background = Array1::<Real>::zeros(input.omega.len());
    let mut imaginary_background = Array1::<Real>::zeros(input.omega.len());

    interpolate_fine_structure_fms(
        &mut real_fms,
        input.omega,
        real_transition[1],
        FineStructureFmsBounds {
            include_low: true,
            include_high: true,
        },
        false,
        &mut real_part,
        &mut real_background,
    )?;
    interpolate_fine_structure_path(
        &mut real_path,
        input.omega,
        real_transition,
        false,
        &mut real_part,
        &mut real_background,
    )?;
    interpolate_fine_structure_fms(
        &mut imaginary_fms,
        input.omega,
        imaginary_transition[1],
        FineStructureFmsBounds {
            include_low: false,
            include_high: false,
        },
        true,
        &mut imaginary_part,
        &mut imaginary_background,
    )?;
    interpolate_fine_structure_path(
        &mut imaginary_path,
        input.omega,
        imaginary_transition,
        true,
        &mut imaginary_part,
        &mut imaginary_background,
    )?;

    let scattering_factor = Array1::from_shape_fn(input.omega.len(), |row| {
        Complex::new(real_part[row], imaginary_part[row])
    });
    let background = Array1::from_shape_fn(input.omega.len(), |row| {
        Complex::new(real_background[row], imaginary_background[row])
    });

    Ok(FullSpectrumFineStructure {
        scattering_factor,
        background,
        real_energy_interval: [real_fms.low_energy, real_path.high_energy],
        imaginary_energy_interval: [imaginary_fms.low_energy, imaginary_path.high_energy],
        real_transition_interval: real_transition,
        imaginary_transition_interval: imaginary_transition,
    })
}

#[derive(Debug, Clone)]
struct PreparedFineStructureSegment {
    energy_hartree: Vec<Real>,
    wave_number: Vec<Real>,
    scattering_factor: Vec<Real>,
    background: Vec<Real>,
    low_energy: Real,
    high_energy: Real,
    cache: LintCache,
}

#[derive(Debug, Clone, Copy)]
struct FineStructureFmsBounds {
    include_low: bool,
    include_high: bool,
}

fn prepare_fine_structure_segment(
    segment_index: usize,
    input: FullSpectrumFineStructureSegmentInput<'_>,
) -> Result<PreparedFineStructureSegment, FullSpectrumError> {
    let len = input.photon_energy_ev.len();
    if len < 2 {
        return Err(FullSpectrumError::SegmentTooShort {
            name: "fine_structure",
            segment: segment_index,
            len,
        });
    }
    validate_segment_len(
        "wave_number_inverse_angstrom",
        segment_index,
        input.wave_number_inverse_angstrom.len(),
        len,
    )?;
    validate_segment_len(
        "scattering_factor",
        segment_index,
        input.scattering_factor.len(),
        len,
    )?;
    validate_segment_len("background", segment_index, input.background.len(), len)?;

    let energy_ev = input.photon_energy_ev.to_vec();
    let wave_number = input.wave_number_inverse_angstrom.to_vec();
    let scattering_factor = input.scattering_factor.to_vec();
    let background = input.background.to_vec();
    for row in 0..len {
        validate_finite_value("fine_structure photon_energy_ev", row, energy_ev[row])?;
        validate_finite_value("fine_structure wave_number", row, wave_number[row])?;
        validate_finite_value(
            "fine_structure scattering_factor",
            row,
            scattering_factor[row],
        )?;
        validate_finite_value("fine_structure background", row, background[row])?;
        if row > 0 && energy_ev[row] <= energy_ev[row - 1] {
            return Err(FullSpectrumError::SegmentNonIncreasingEnergy {
                segment: segment_index,
                row,
                previous: energy_ev[row - 1],
                current: energy_ev[row],
            });
        }
    }

    let energy_hartree = energy_ev
        .into_iter()
        .map(|energy| energy / FEFF_HARTREE_EV)
        .collect::<Vec<_>>();
    let low_energy = energy_hartree[0];
    let high_energy = energy_hartree[len - 1];

    Ok(PreparedFineStructureSegment {
        energy_hartree,
        wave_number,
        scattering_factor,
        background,
        low_energy,
        high_energy,
        cache: LintCache::new(),
    })
}

fn fine_structure_transition_interval(
    name: &'static str,
    segment: &PreparedFineStructureSegment,
    low_wave_number: Real,
    high_wave_number: Real,
) -> Result<[Real; 2], FullSpectrumError> {
    let low = transition_energy_at_wave_number(name, segment, low_wave_number)?;
    let high = transition_energy_at_wave_number(name, segment, high_wave_number)?;
    if high <= low {
        return Err(FullSpectrumError::InvalidEnergyRange {
            name: "fine_structure_transition",
            min: low,
            max: high,
        });
    }
    Ok([low, high])
}

fn transition_energy_at_wave_number(
    name: &'static str,
    segment: &PreparedFineStructureSegment,
    threshold: Real,
) -> Result<Real, FullSpectrumError> {
    segment
        .energy_hartree
        .iter()
        .copied()
        .zip(segment.wave_number.iter().copied())
        .filter_map(|(energy, wave_number)| (wave_number <= threshold).then_some(energy))
        .next_back()
        .ok_or(FullSpectrumError::MissingTransitionThreshold { name, threshold })
}

fn interpolate_fine_structure_fms(
    segment: &mut PreparedFineStructureSegment,
    omega: ArrayView1<'_, Real>,
    high_transition: Real,
    bounds: FineStructureFmsBounds,
    clamp_nonnegative: bool,
    scattering_factor_out: &mut Array1<Real>,
    background_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    segment.cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if within_fms_interval(energy, segment.low_energy, high_transition, bounds) {
            scattering_factor_out[row] = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.scattering_factor,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
            background_out[row] = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.background,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
        }
    }
    Ok(())
}

fn interpolate_fine_structure_path(
    segment: &mut PreparedFineStructureSegment,
    omega: ArrayView1<'_, Real>,
    transition: [Real; 2],
    clamp_nonnegative: bool,
    scattering_factor_out: &mut Array1<Real>,
    background_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    segment.cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if energy >= transition[0] && energy <= segment.high_energy {
            let (path_weight, fms_weight) = fine_structure_transition_weights(energy, transition);
            let scattering_factor = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.scattering_factor,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
            scattering_factor_out[row] =
                scattering_factor * path_weight + scattering_factor_out[row] * fms_weight;
            let background = maybe_clamp_fine_structure(
                lint_with_cache(
                    &segment.energy_hartree,
                    &segment.background,
                    energy,
                    &mut segment.cache,
                )
                .map_err(|source| FullSpectrumError::Interpolation { source })?,
                clamp_nonnegative,
            );
            background_out[row] = background * path_weight + background_out[row] * fms_weight;
        }
    }
    Ok(())
}

fn within_fms_interval(
    energy: Real,
    low_energy: Real,
    high_transition: Real,
    bounds: FineStructureFmsBounds,
) -> bool {
    let above_low = if bounds.include_low {
        energy >= low_energy
    } else {
        energy > low_energy
    };
    let below_high = if bounds.include_high {
        energy <= high_transition
    } else {
        energy < high_transition
    };
    above_low && below_high
}

fn fine_structure_transition_weights(energy: Real, transition: [Real; 2]) -> (Real, Real) {
    if energy >= transition[0] && energy <= transition[1] {
        let fraction = (transition[1] - energy) / (transition[1] - transition[0]);
        let theta = fraction * std::f64::consts::FRAC_PI_2;
        (theta.cos().powi(2), theta.sin().powi(2))
    } else {
        (1.0, 0.0)
    }
}

fn maybe_clamp_fine_structure(value: Real, clamp_nonnegative: bool) -> Real {
    if clamp_nonnegative {
        value.max(0.0)
    } else {
        value
    }
}
