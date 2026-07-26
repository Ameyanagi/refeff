use super::*;
use crate::constants::HARTREE_EV;
use crate::interpolation::locate_below;

/// Evaluate FEFF `EXCH/rhlbp.f90` with a caller-provided `bphl.dat` table.
///
/// FEFF does not distribute `bphl.dat`; licensed users obtain it separately.
/// The table therefore remains an explicit input instead of being replaced by
/// generated or approximate values.
pub fn broadened_hedin_lundqvist_self_energy(
    table: &BroadenedHedinLundqvistTable,
    radius: Real,
    momentum: Real,
) -> Result<XcpotSigma, ExchangeError> {
    ensure_positive("rs", radius)?;
    ensure_positive("xk", momentum)?;

    let fermi_momentum = FEFF_FA / radius;
    let fermi_energy = fermi_momentum.powi(2) / 2.0;
    let plasma_over_fermi = (3.0 / radius.powi(3)).sqrt() / fermi_energy;
    let normalized_momentum = momentum / fermi_momentum;
    let reduced_energy = (normalized_momentum.powi(2) - 1.0) / radius.sqrt();

    let real =
        source_compatible_terp2d(table, &table.real, radius, reduced_energy)? / radius / HARTREE_EV;
    let mut imaginary = source_compatible_terp2d(table, &table.imaginary, radius, reduced_energy)?
        / radius
        / HARTREE_EV;

    let quinn =
        quinn_imaginary_self_energy(normalized_momentum, radius, plasma_over_fermi, fermi_energy)?;
    if imaginary >= quinn {
        imaginary = quinn;
    }

    ensure_finite("rhlbp real", real)?;
    ensure_finite("rhlbp imaginary", imaginary)?;
    Ok(XcpotSigma { real, imaginary })
}

/// Port FEFF's local `terp2d` exactly, including its historical `z2 == z1`
/// assignment. Consequently the reduced-energy coordinate selects a table
/// column but is not interpolated between adjacent columns. This behavior is
/// also retained by the source-faithful FEFF85 C++ port.
fn source_compatible_terp2d(
    table: &BroadenedHedinLundqvistTable,
    values: &[Real],
    radius: Real,
    reduced_energy: Real,
) -> Result<Real, ExchangeError> {
    let radius_index = locate_below(radius, &table.radius_mesh).clamp(1, BPHL_RADIUS_COUNT - 1) - 1;
    let energy_index = locate_below(reduced_energy, &table.reduced_energy_mesh)
        .clamp(1, BPHL_REDUCED_ENERGY_COUNT - 1)
        - 1;

    let radius_left = table.radius_mesh[radius_index];
    let radius_right = table.radius_mesh[radius_index + 1];
    let denominator = radius_right - radius_left;
    if denominator == 0.0 {
        return Err(ExchangeError::ZeroDenominator {
            name: "bphl.dat radius interval",
        });
    }
    let radius_fraction = (radius - radius_left) / denominator;
    let value_left = values[table.flat_index(radius_index, energy_index)];
    let value_right = values[table.flat_index(radius_index + 1, energy_index)];
    Ok(value_left + radius_fraction * (value_right - value_left))
}
