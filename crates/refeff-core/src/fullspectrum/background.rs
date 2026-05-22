//! FULLSPECTRUM FPRIME background interpolation.

use ndarray::{Array1, ArrayView1};

use crate::interpolation::{LintCache, lint_with_cache};
use crate::{Complex, Real};

use super::constants::{
    FEFF_FULLSPECTRUM_BACKGROUND_SUM_MAX, FEFF_FULLSPECTRUM_BACKGROUND_SUM_MIN,
    FEFF_FULLSPECTRUM_GRID_CAPACITY, FEFF_HARTREE_EV,
};
use super::grids::full_spectrum_linear_energy_grid;
use super::sum_rules::full_spectrum_effective_electron_count;
use super::types::*;
use super::validation::{validate_finite_value, validate_segment_len};

/// Port of `FULLSPECTRUM/rdbkg.f90`: assemble FPRIME background factors.
///
/// FEFF reads `fprime1/xmu.dat`, `fprime2/xmu.dat`, and following numbered
/// files, then processes them in reverse so lower-numbered files overwrite
/// overlapping higher-numbered intervals. This helper accepts those parsed
/// segments directly in that file-priority order and returns the complex
/// background scattering factor on the caller's Hartree grid.
pub fn full_spectrum_background_from_fprime(
    input: FullSpectrumBackgroundInput<'_>,
) -> Result<FullSpectrumBackground, FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "omega" });
    }
    if input.segments.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "background_segments",
        });
    }
    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("background omega", row, value)?;
        if row > 0 && value < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: value,
            });
        }
    }

    let mut prepared = input
        .segments
        .iter()
        .enumerate()
        .map(|(segment, source)| prepare_background_segment(segment, *source))
        .collect::<Result<Vec<_>, _>>()?;

    let mut f_prime = Array1::<Real>::zeros(input.omega.len());
    let mut f_double_prime = Array1::<Real>::zeros(input.omega.len());
    let sum_grid = full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
        point_count: FEFF_FULLSPECTRUM_GRID_CAPACITY,
        min_energy: FEFF_FULLSPECTRUM_BACKGROUND_SUM_MIN,
        max_energy: FEFF_FULLSPECTRUM_BACKGROUND_SUM_MAX,
    })?;
    let mut sum_f_double_prime = Array1::<Real>::zeros(sum_grid.len());

    let mut zero_energy_fprime = 0.0;
    let mut zero_energy = Real::INFINITY;
    for segment in prepared.iter_mut().rev() {
        if segment.low_energy < zero_energy {
            zero_energy = segment.low_energy;
            zero_energy_fprime = segment.f_prime[0];
        }
        interpolate_background_segment(
            segment,
            input.omega,
            zero_energy_fprime,
            &mut f_prime,
            &mut f_double_prime,
        )?;
        interpolate_background_sum_segment(segment, sum_grid.view(), &mut sum_f_double_prime)?;
    }

    for (row, value) in sum_f_double_prime.iter_mut().enumerate() {
        *value /= sum_grid[row].powi(2);
    }
    let effective_electron_count = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
        number_density: 1.0 / (4.0 * std::f64::consts::PI),
        epsilon2: sum_f_double_prime.view(),
        omega: sum_grid.view(),
        active_len: sum_grid.len(),
    })?;

    let scattering_factor = Array1::from_shape_fn(input.omega.len(), |row| {
        if input.omega[row] <= zero_energy {
            Complex::new(zero_energy_fprime, 0.0)
        } else {
            Complex::new(f_prime[row], f_double_prime[row])
        }
    });

    Ok(FullSpectrumBackground {
        scattering_factor,
        effective_electron_count,
        zero_energy_fprime,
    })
}

#[derive(Debug, Clone)]
struct PreparedBackgroundSegment {
    energy_hartree: Vec<Real>,
    f_prime: Vec<Real>,
    f_double_prime: Vec<Real>,
    low_energy: Real,
    high_energy: Real,
    output_cache: LintCache,
    sum_cache: LintCache,
}

fn prepare_background_segment(
    segment_index: usize,
    input: FullSpectrumBackgroundSegmentInput<'_>,
) -> Result<PreparedBackgroundSegment, FullSpectrumError> {
    let len = input.photon_energy_ev.len();
    if len < 2 {
        return Err(FullSpectrumError::SegmentTooShort {
            name: "background",
            segment: segment_index,
            len,
        });
    }
    validate_segment_len("f_prime", segment_index, input.f_prime.len(), len)?;
    validate_segment_len(
        "f_double_prime",
        segment_index,
        input.f_double_prime.len(),
        len,
    )?;

    let mut energy_ev = input.photon_energy_ev.to_vec();
    let mut f_prime = input.f_prime.to_vec();
    let mut f_double_prime = input.f_double_prime.to_vec();
    for row in 0..len {
        validate_finite_value("background photon_energy_ev", row, energy_ev[row])?;
        validate_finite_value("background f_prime", row, f_prime[row])?;
        validate_finite_value("background f_double_prime", row, f_double_prime[row])?;
        if row > 0 && energy_ev[row] <= energy_ev[row - 1] {
            return Err(FullSpectrumError::SegmentNonIncreasingEnergy {
                segment: segment_index,
                row,
                previous: energy_ev[row - 1],
                current: energy_ev[row],
            });
        }
    }

    if energy_ev[0] <= 0.0 {
        let shifted_energy = 0.01;
        let mut cache = LintCache::new();
        f_prime[0] = lint_with_cache(&energy_ev, &f_prime, shifted_energy, &mut cache)
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
        cache.reset();
        f_double_prime[0] =
            lint_with_cache(&energy_ev, &f_double_prime, shifted_energy, &mut cache)
                .map_err(|source| FullSpectrumError::Interpolation { source })?;
        energy_ev[0] = shifted_energy;
        if energy_ev[1] <= energy_ev[0] {
            return Err(FullSpectrumError::SegmentNonIncreasingEnergy {
                segment: segment_index,
                row: 1,
                previous: energy_ev[0],
                current: energy_ev[1],
            });
        }
    }

    let energy_hartree = energy_ev
        .into_iter()
        .map(|energy| energy / FEFF_HARTREE_EV)
        .collect::<Vec<_>>();
    let low_energy = energy_hartree[0];
    let high_energy = energy_hartree[len - 1];

    Ok(PreparedBackgroundSegment {
        energy_hartree,
        f_prime,
        f_double_prime,
        low_energy,
        high_energy,
        output_cache: LintCache::new(),
        sum_cache: LintCache::new(),
    })
}

fn interpolate_background_segment(
    segment: &mut PreparedBackgroundSegment,
    omega: ArrayView1<'_, Real>,
    zero_energy_fprime: Real,
    f_prime_out: &mut Array1<Real>,
    f_double_prime_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    let scratch = Vec::from_iter(
        segment
            .energy_hartree
            .iter()
            .zip(segment.f_prime.iter())
            .map(|(energy, f_prime)| {
                if *energy != 0.0 {
                    (*f_prime - zero_energy_fprime) / energy.powi(2)
                } else {
                    0.0
                }
            }),
    );

    segment.output_cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if energy > segment.low_energy && energy < segment.high_energy {
            let interpolated = lint_with_cache(
                &segment.energy_hartree,
                &scratch,
                energy,
                &mut segment.output_cache,
            )
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
            f_prime_out[row] = interpolated * energy.powi(2) + zero_energy_fprime;
        }
    }

    segment.output_cache.reset();
    for (row, energy) in omega.iter().copied().enumerate() {
        if energy > segment.low_energy && energy < segment.high_energy {
            let interpolated = lint_with_cache(
                &segment.energy_hartree,
                &segment.f_double_prime,
                energy,
                &mut segment.output_cache,
            )
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
            f_double_prime_out[row] = interpolated.max(0.0);
        }
    }
    Ok(())
}

fn interpolate_background_sum_segment(
    segment: &mut PreparedBackgroundSegment,
    sum_grid: ArrayView1<'_, Real>,
    f_double_prime_out: &mut Array1<Real>,
) -> Result<(), FullSpectrumError> {
    segment.sum_cache.reset();
    for (row, energy) in sum_grid.iter().copied().enumerate() {
        if energy > segment.low_energy && energy < segment.high_energy {
            let interpolated = lint_with_cache(
                &segment.energy_hartree,
                &segment.f_double_prime,
                energy,
                &mut segment.sum_cache,
            )
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
            f_double_prime_out[row] = interpolated.max(0.0);
        }
    }
    Ok(())
}
