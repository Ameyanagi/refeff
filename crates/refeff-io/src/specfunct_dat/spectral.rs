use ndarray::{Array2, ArrayView3};
use refeff_core::{
    Real, SfconvMomentumSpectralInterpolation, SfconvMomentumSpectralInterpolationInput,
    sfconv_interpolate_momentum_spectral_function,
};

use crate::error::{IoError, Result};

use super::types::{
    SfconvSpecfunctCompatibilityInput, SfconvSpecfunctData, SfconvSpecfunctSpectralRowsInput,
};
use super::validation::{
    validate_compatibility_input, validate_finite_scalar, validate_specfunct_dat,
    validate_specfunct_spectral_rows_input,
};

pub fn sfconv_specfunct_data_from_spectral_rows(
    input: SfconvSpecfunctSpectralRowsInput<'_>,
) -> Result<SfconvSpecfunctData> {
    validate_specfunct_spectral_rows_input(input)?;
    let data = SfconvSpecfunctData {
        wigner_seitz_radius: input.wigner_seitz_radius,
        core_hole_lifetime: input.core_hole_lifetime,
        asymmetric_phase: input.asymmetric_phase,
        satellite_type: input.satellite_type,
        low_q_mode: input.low_q_mode,
        pole_count: input.pole_count,
        pole_energy: input.pole_energy.to_owned(),
        pole_broadening: input.pole_broadening.to_owned(),
        pole_weight: input.pole_weight.to_owned(),
        spectral_info: input.spectral_info.to_owned(),
        weights: input.weights.to_owned(),
        extrinsic_quasiparticle: spectral_function_row(input.spectral_function, 0),
        extrinsic_satellite: spectral_function_row(input.spectral_function, 1),
        interference_quasiparticle: spectral_function_row(input.spectral_function, 2),
        interference_satellite: spectral_function_row(input.spectral_function, 3),
        intrinsic_satellite: spectral_function_row(input.spectral_function, 4),
        clipped_extrinsic_satellite: spectral_function_row(input.spectral_function, 7),
        energy_grid: input.energy_grid.to_owned(),
    };
    validate_specfunct_dat(&data)?;
    Ok(data)
}

/// Return whether parsed cache data matches the current SO2CONV inputs.
///
/// This mirrors the reuse checks in `SFCONV/so2conv.f90`: material scalars,
/// integer selectors, active pole rows, and the momentum grid must match. FEFF
/// compares the momentum grid after converting both values to default `REAL`,
/// so this function compares those entries as `f32`.
pub fn sfconv_specfunct_matches_so2conv_inputs(
    data: &SfconvSpecfunctData,
    input: SfconvSpecfunctCompatibilityInput<'_>,
) -> Result<bool> {
    validate_specfunct_dat(data)?;
    validate_compatibility_input(input)?;

    if data.wigner_seitz_radius != input.wigner_seitz_radius
        || data.core_hole_lifetime != input.core_hole_lifetime
        || data.asymmetric_phase != input.asymmetric_phase
        || data.low_q_mode != input.low_q_mode
        || data.satellite_type != input.satellite_type
        || data.pole_count != input.pole_count
        || data.momentum_count() != input.momentum_grid.len()
    {
        return Ok(false);
    }

    let active_poles_match = (0..data.pole_count).all(|index| {
        data.pole_energy[index] == input.pole_energy[index]
            && data.pole_broadening[index] == input.pole_broadening[index]
            && data.pole_weight[index] == input.pole_weight[index]
    });
    if !active_poles_match {
        return Ok(false);
    }

    let momentum_matches = (0..data.momentum_count()).all(|index| {
        (data.spectral_info[[index, 0]] as f32) == (input.momentum_grid[index] as f32)
    });
    Ok(momentum_matches)
}

/// Build a validated core interpolation view over a `specfunct.dat` cache.
///
/// This maps the FEFF cache layout to `refeff-core`'s momentum spectral
/// interpolation input without copying the cached arrays. The first `sfinfo`
/// column is FEFF `pgrid`; columns 4 through 8 are `se`, `ce`, `width`, `z1`,
/// and `z1i`.
pub fn sfconv_specfunct_momentum_interpolation_input(
    data: &SfconvSpecfunctData,
    photoelectron_momentum: Real,
) -> Result<SfconvMomentumSpectralInterpolationInput<'_>> {
    validate_specfunct_dat(data)?;
    validate_finite_scalar(photoelectron_momentum, "photoelectron momentum")?;

    Ok(SfconvMomentumSpectralInterpolationInput {
        photoelectron_momentum,
        momentum_grid: data.spectral_info.column(0),
        energy_grid: data.energy_grid.view(),
        extrinsic_quasiparticle: data.extrinsic_quasiparticle.view(),
        extrinsic_satellite: data.extrinsic_satellite.view(),
        interference_quasiparticle: data.interference_quasiparticle.view(),
        interference_satellite: data.interference_satellite.view(),
        intrinsic_satellite: data.intrinsic_satellite.view(),
        clipped_extrinsic_satellite: data.clipped_extrinsic_satellite.view(),
        weights: data.weights.view(),
        self_energy_real: data.spectral_info.column(3),
        energy_correction: data.spectral_info.column(4),
        width: data.spectral_info.column(5),
        renormalization_real: data.spectral_info.column(6),
        renormalization_imag: data.spectral_info.column(7),
    })
}

/// Interpolate one cached `specfunct.dat` spectral row to a photoelectron momentum.
///
/// This is the typed handoff from FEFF's binary SO2CONV cache to the core
/// numerical interpolation kernel used by the future full driver.
pub fn sfconv_specfunct_interpolate_momentum(
    data: &SfconvSpecfunctData,
    photoelectron_momentum: Real,
) -> Result<SfconvMomentumSpectralInterpolation> {
    let input = sfconv_specfunct_momentum_interpolation_input(data, photoelectron_momentum)?;
    sfconv_interpolate_momentum_spectral_function(input)
        .map_err(|source| IoError::SpecfunctDatInterpolation { source })
}
fn spectral_function_row(values: ArrayView3<'_, f64>, source_row: usize) -> Array2<f64> {
    let (momentum_count, _, spectral_point_count) = values.dim();
    Array2::from_shape_fn(
        (momentum_count, spectral_point_count),
        |(momentum, point)| values[[momentum, source_row, point]],
    )
}
