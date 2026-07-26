//! LDOS final table assembly helpers.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, Array3, Array4, Array5, Axis, Slice};
use num_complex::Complex32;
use refeff_linalg::{SymmetricTriangle, real64_symmetric_eigen};

use crate::fovrg::{FovrgDiracSolverInput, fovrg_dirac_solver};
use crate::interpolation::{terp, terpc};
use crate::quadrature::{csomm2, trap};
use crate::rhorrp::{
    RhorrpExactRadialTailInput, RhorrpIrregularInitialConditionInput,
    RhorrpIrregularSolutionTransformInput, RhorrpIrregularWronskianScaleInput,
    RhorrpMuffinTinMatchInput, RhorrpRegularSolutionScaleInput, rhorrp_exact_radial_tail,
    rhorrp_irregular_initial_condition, rhorrp_irregular_solution_transform,
    rhorrp_irregular_wronskian_scale, rhorrp_muffin_tin_match, rhorrp_regular_solution_scale,
};
use crate::{Complex, FEFF_HARTREE_EV, Real};

use super::support::{
    ensure_len, ensure_length_match, ensure_shape, ensure_shape3, validate_complex_scalar,
    validate_complex_values, validate_positive_real_scalar, validate_positive_real_values,
    validate_real_scalar, widen_complex32,
};
use super::{
    DensityError, LdosFf2rhoInput, LdosFf2rhoTables, LdosFmsdosTraceGridInput,
    LdosFmsdosTraceInput, LdosHubbardMagneticFf2rhoInput, LdosHubbardMagneticFf2rhoTables,
    LdosHubbardStep1, LdosHubbardStep1Input, LdosRholChannel, LdosRholChannelInput,
    LdosRholDensity, LdosRholDensityGrid, LdosRholDensityGridInput, LdosRholDensityInput,
    LdosRholExactRadialTail, LdosRholExactRadialTailInput, LdosRholRadialAssembly,
    LdosRholRadialAssemblyInput, LdosRholTableDriver, LdosRholTableDriverInput,
    LdosRholWavefunctionTables, LdosRholWavefunctionTablesInput, LdosSpinFf2rhoInput,
    LdosSpinFf2rhoTables, PotRholieDensity, PotRholieDensityGrid, PotRholieDensityGridInput,
    PotRholieDensityInput, PotScfContourSourceRows, PotScfContourSourceRowsInput,
    PotScfEnergyDensity, PotScfEnergyDensityInput, ValenceDensityUpdateInput,
    update_valence_density,
};

const LDOS_FINE_STRUCTURE_ALPHA: Real = 1.0 / 137.035_989_56;
const LDOS_HUBBARD_OCCUPATION_POINTS: usize = 600;

/// Port the non-full-potential table update in FEFF `LDOS/ff2rho.f90`.
///
/// FEFF first writes the embedded-atom density `xrhoce` to `rhocNN.dat`, then
/// updates the same work array for `ldosNN.dat` when `msapp.ne.1`:
/// `xrhoce(l,ie) += imag(cchi(l,ie) * xrhole(l,ie))`.
pub fn ldos_ff2rho_tables(input: LdosFf2rhoInput<'_>) -> Result<LdosFf2rhoTables, DensityError> {
    validate_ldos_ff2rho_input(input)?;

    let energy_count = input.energy_grid_hartree.len();
    let mut energy_ev = Array1::<Real>::zeros(energy_count);
    let mut ldos_density = Array2::<Real>::zeros((energy_count, input.angular_count));
    let mut rhoc_density = Array2::<Real>::zeros((energy_count, input.angular_count));

    for energy_index in 0..energy_count {
        let energy = input.energy_grid_hartree[energy_index];
        validate_complex_scalar("ldos_energy", energy)?;
        energy_ev[energy_index] = energy.re * FEFF_HARTREE_EV;
        validate_real_scalar("ldos_energy_ev", energy_ev[energy_index])?;

        for angular in 0..input.angular_count {
            let embedded = input.embedded_ldos[(angular, energy_index)];
            validate_real_scalar("ldos_embedded_density", embedded)?;
            rhoc_density[(energy_index, angular)] = embedded;

            let mut density = embedded;
            if input.apply_scattering {
                let trace = input.scattering_trace[(angular, energy_index)];
                let scattering = input.scattering_ldos[(angular, energy_index)];
                validate_complex_scalar("ldos_scattering_trace", trace)?;
                validate_complex_scalar("ldos_scattering_density", scattering)?;
                let correction = (trace * scattering).im;
                validate_real_scalar("ldos_scattering_correction", correction)?;
                density += correction;
            }
            validate_real_scalar("ldos_density", density)?;
            ldos_density[(energy_index, angular)] = density;
        }
    }

    Ok(LdosFf2rhoTables {
        energy_ev,
        ldos_density,
        rhoc_density,
    })
}

/// Port the spin-resolved LDOS table update in FEFF `LDOS/ff2rho_h.f90`.
///
/// FEFF writes `rhocNN.dat` from embedded `xrhoce(l,is,ie)` values in
/// spin-major order and writes `ldosNN.dat` after the optional
/// `imag(cchi(l,is,ie) * xrhole(l,is,ie))` scattering correction.
pub fn ldos_spin_ff2rho_tables(
    input: LdosSpinFf2rhoInput<'_>,
) -> Result<LdosSpinFf2rhoTables, DensityError> {
    validate_ldos_spin_ff2rho_input(input)?;

    let energy_count = input.energy_grid_hartree.len();
    let column_count = input.angular_count * 2;
    let mut energy_ev = Array1::<Real>::zeros(energy_count);
    let mut ldos_density = Array2::<Real>::zeros((energy_count, column_count));
    let mut rhoc_density = Array2::<Real>::zeros((energy_count, column_count));

    for energy_index in 0..energy_count {
        let energy = input.energy_grid_hartree[energy_index];
        validate_complex_scalar("ldos_energy", energy)?;
        energy_ev[energy_index] = energy.re * FEFF_HARTREE_EV;
        validate_real_scalar("ldos_energy_ev", energy_ev[energy_index])?;

        for spin in 0..2 {
            for angular in 0..input.angular_count {
                let column = spin * input.angular_count + angular;
                let embedded = input.embedded_ldos[(angular, spin, energy_index)];
                validate_real_scalar("ldos_embedded_density", embedded)?;
                rhoc_density[(energy_index, column)] = embedded;

                let mut density = embedded;
                if input.apply_scattering {
                    let trace = input.scattering_trace[(angular, spin, energy_index)];
                    let scattering = input.scattering_ldos[(angular, spin, energy_index)];
                    validate_complex_scalar("ldos_scattering_trace", trace)?;
                    validate_complex_scalar("ldos_scattering_density", scattering)?;
                    let correction = (trace * scattering).im;
                    validate_real_scalar("ldos_scattering_correction", correction)?;
                    density += correction;
                }
                validate_real_scalar("ldos_density", density)?;
                ldos_density[(energy_index, column)] = density;
            }
        }
    }

    Ok(LdosSpinFf2rhoTables {
        energy_ev,
        ldos_density,
        rhoc_density,
    })
}

/// Port the density-matrix, eigensystem, and Hubbard-potential pass in
/// FEFF `LDOS/ff2rho_h_step1.f90`.
///
/// FEFF first forms diagonal and off-diagonal magnetic densities from the
/// ordinary radial `xrhoce`/`xrhole` arrays and the first-pass FMS traces. It
/// cubic-interpolates those densities to a 600-point occupation grid ending
/// just below `xmu-fermi_shift`, diagonalizes the active potential-1 Hubbard
/// block, and constructs `Vnlm`, `TFrm`, and `TFrmInv` for the second pass.
pub fn ldos_hubbard_step1(
    input: LdosHubbardStep1Input<'_>,
) -> Result<LdosHubbardStep1, DensityError> {
    validate_ldos_hubbard_step1_input(input)?;

    let energy_count = input.energy_grid_hartree.len();
    let magnetic_count = input.angular_count * input.angular_count;
    let hubbard_order = 2 * input.hubbard_l + 1;
    let off_diagonal_count = (input.hubbard_l + 1) * (input.hubbard_l + 1);
    let mut embedded_magnetic_ldos =
        Array4::<Real>::zeros((input.angular_count, magnetic_count, 2, energy_count));
    let mut off_diagonal_density = Array5::<Real>::zeros((
        input.angular_count,
        off_diagonal_count,
        off_diagonal_count,
        2,
        energy_count,
    ));

    for angular in 0..input.angular_count {
        let magnetic_start = angular * angular;
        let magnetic_end = (angular + 1) * (angular + 1);
        let degeneracy = (2 * angular + 1) as Real;
        for spin in 0..2 {
            for energy_index in 0..energy_count {
                let embedded = input.embedded_ldos[(angular, spin, energy_index)];
                let scattering = input.scattering_ldos[(angular, spin, energy_index)];
                for magnetic in magnetic_start..magnetic_end {
                    let trace =
                        input.magnetic_scattering_trace[(angular, magnetic, spin, energy_index)];
                    embedded_magnetic_ldos[(angular, magnetic, spin, energy_index)] =
                        embedded / degeneracy + (trace * scattering).im;
                }
                if angular == input.hubbard_l {
                    for row in magnetic_start..magnetic_end {
                        for column in magnetic_start..magnetic_end {
                            let mut density = if row == column {
                                embedded / degeneracy
                            } else {
                                0.0
                            };
                            density += (input.off_diagonal_scattering_trace
                                [(angular, row, column, spin, energy_index)]
                                * scattering)
                                .im;
                            validate_real_scalar("ldos_hubbard_off_diagonal_density", density)?;
                            off_diagonal_density[(angular, row, column, spin, energy_index)] =
                                density;
                        }
                    }
                }
            }
        }
    }

    let energies = input
        .energy_grid_hartree
        .iter()
        .map(|energy| energy.re)
        .collect::<Vec<_>>();
    let occupation_limit =
        input.chemical_potential_hartree - input.fermi_shift_ev / FEFF_HARTREE_EV;
    let mut occupations = Array3::<Real>::zeros((input.angular_count, magnetic_count, 2));
    for angular in 0..input.angular_count {
        let magnetic_start = angular * angular;
        let magnetic_end = (angular + 1) * (angular + 1);
        for spin in 0..2 {
            for magnetic in magnetic_start..magnetic_end {
                let density = (0..energy_count)
                    .map(|energy| embedded_magnetic_ldos[(angular, magnetic, spin, energy)])
                    .collect::<Vec<_>>();
                occupations[(angular, magnetic, spin)] =
                    ldos_hubbard_occupation(&energies, &density, occupation_limit)?;
            }
        }
    }

    let hubbard_start = input.hubbard_l * input.hubbard_l;
    let hubbard_end = (input.hubbard_l + 1) * (input.hubbard_l + 1);
    let mut transform =
        Array4::<Complex>::zeros((2, input.angular_count, hubbard_order, hubbard_order));
    let mut inverse_transform =
        Array4::<Complex>::zeros((2, input.angular_count, hubbard_order, hubbard_order));
    for spin in 0..2 {
        for angular in 0..input.angular_count {
            for row in 0..hubbard_order {
                transform[(spin, angular, row, row)] = Complex::new(1.0, 0.0);
                inverse_transform[(spin, angular, row, row)] = Complex::new(1.0, 0.0);
            }
        }
    }

    if input.potential_index == 1 {
        for spin in 0..2 {
            let mut occupation_matrix = Array2::<Real>::zeros((hubbard_order, hubbard_order));
            for row in 0..hubbard_order {
                for column in 0..hubbard_order {
                    // FEFF explicitly copies the lower triangle into the upper
                    // triangle before calling SSYEV with UPLO='U'.
                    let source_row = row.max(column);
                    let source_column = row.min(column);
                    occupation_matrix[(row, column)] = ldos_hubbard_occupation(
                        &energies,
                        &(0..energy_count)
                            .map(|energy| {
                                off_diagonal_density[(
                                    input.hubbard_l,
                                    hubbard_start + source_row,
                                    hubbard_start + source_column,
                                    spin,
                                    energy,
                                )]
                            })
                            .collect::<Vec<_>>(),
                        occupation_limit,
                    )?;
                }
            }
            let eigensystem =
                real64_symmetric_eigen(occupation_matrix.view(), SymmetricTriangle::Upper)?;
            for row in 0..hubbard_order {
                occupations[(input.hubbard_l, hubbard_start + row, spin)] =
                    eigensystem.eigenvalues()[row];
                for column in 0..hubbard_order {
                    let eigenvector = eigensystem.eigenvectors()[(column, row)];
                    transform[(spin, input.hubbard_l, row, column)] =
                        Complex::new(eigenvector, 0.0);
                    inverse_transform[(spin, input.hubbard_l, column, row)] =
                        Complex::new(eigenvector, 0.0);
                }
            }
        }
    }

    let mut hubbard_potential = Array3::<Real>::zeros((2, input.angular_count, magnetic_count));
    if input.potential_index <= 1 {
        let mut occupation_sum = 0.0;
        for spin in 0..2 {
            for magnetic in hubbard_start..hubbard_end {
                occupation_sum += occupations[(input.hubbard_l, magnetic, spin)];
            }
        }
        let average_occupation = occupation_sum / (4 * input.hubbard_l + 2) as Real;
        let u = input.hubbard_u_ev / FEFF_HARTREE_EV;
        let u_minus_j = (input.hubbard_u_ev - input.hubbard_j_ev) / FEFF_HARTREE_EV;
        for spin in 0..2 {
            let opposite_spin = 1 - spin;
            for magnetic in hubbard_start..hubbard_end {
                let opposite = (hubbard_start..hubbard_end)
                    .map(|other| {
                        u * (occupations[(input.hubbard_l, other, opposite_spin)]
                            - average_occupation)
                    })
                    .sum::<Real>();
                let same = (hubbard_start..hubbard_end)
                    .filter(|&other| other != magnetic)
                    .map(|other| {
                        u_minus_j
                            * (occupations[(input.hubbard_l, other, spin)] - average_occupation)
                    })
                    .sum::<Real>();
                hubbard_potential[(spin, input.hubbard_l, magnetic)] = opposite + same;
            }
        }
    }

    Ok(LdosHubbardStep1 {
        embedded_magnetic_ldos,
        occupations,
        hubbard_potential,
        transform,
        inverse_transform,
    })
}

/// Port the magnetic-orbital table update in FEFF `LDOS/ff2rho_h_step2.f90`.
///
/// FEFF writes `rhocmNN.dat` from embedded `xmrhoce(l,im,is,ie)` values, then
/// updates the same work array for `lmdosNN.dat`:
/// `xmrhoce/(2*l+1) + imag(gtr_m * xmrhole)`.
pub fn ldos_hubbard_magnetic_ff2rho_tables(
    input: LdosHubbardMagneticFf2rhoInput<'_>,
) -> Result<LdosHubbardMagneticFf2rhoTables, DensityError> {
    validate_ldos_hubbard_magnetic_ff2rho_input(input)?;

    let energy_count = input.energy_grid_hartree.len();
    let magnetic_count =
        input
            .angular_count
            .checked_mul(input.angular_count)
            .ok_or(DensityError::InvalidIndex {
                name: "angular_count",
                index: input.angular_count,
            })?;
    let column_count = magnetic_count
        .checked_mul(2)
        .ok_or(DensityError::InvalidIndex {
            name: "angular_count",
            index: input.angular_count,
        })?;
    let mut energy_ev = Array1::<Real>::zeros(energy_count);
    let mut lmdos_density = Array2::<Real>::zeros((energy_count, column_count));
    let mut rhocm_density = Array2::<Real>::zeros((energy_count, column_count));

    for energy_index in 0..energy_count {
        let energy = input.energy_grid_hartree[energy_index];
        validate_complex_scalar("ldos_energy", energy)?;
        energy_ev[energy_index] = energy.re * FEFF_HARTREE_EV;
        validate_real_scalar("ldos_energy_ev", energy_ev[energy_index])?;

        for spin in 0..2 {
            for angular in 0..input.angular_count {
                let angular_start =
                    angular
                        .checked_mul(angular)
                        .ok_or(DensityError::InvalidIndex {
                            name: "angular_count",
                            index: input.angular_count,
                        })?;
                let angular_end =
                    (angular + 1)
                        .checked_mul(angular + 1)
                        .ok_or(DensityError::InvalidIndex {
                            name: "angular_count",
                            index: input.angular_count,
                        })?;
                let degeneracy = (2 * angular + 1) as Real;
                for magnetic in angular_start..angular_end {
                    let column = spin * magnetic_count + magnetic;
                    let embedded =
                        input.embedded_magnetic_ldos[(angular, magnetic, spin, energy_index)];
                    validate_real_scalar("ldos_magnetic_embedded_density", embedded)?;
                    rhocm_density[(energy_index, column)] = embedded;

                    let trace =
                        input.magnetic_scattering_trace[(angular, magnetic, spin, energy_index)];
                    let scattering =
                        input.scattering_magnetic_ldos[(angular, magnetic, spin, energy_index)];
                    validate_complex_scalar("ldos_magnetic_scattering_trace", trace)?;
                    validate_complex_scalar("ldos_magnetic_scattering_density", scattering)?;
                    let density = embedded / degeneracy + (trace * scattering).im;
                    validate_real_scalar("ldos_magnetic_density", density)?;
                    lmdos_density[(energy_index, column)] = density;
                }
            }
        }
    }

    Ok(LdosHubbardMagneticFf2rhoTables {
        energy_ev,
        lmdos_density,
        rhocm_density,
    })
}

/// Port the non-full-potential trace projection in FEFF `LDOS/fmsdos.f90`.
///
/// For each potential and angular channel, FEFF sums the diagonal packed FMS
/// `gg` entries over magnetic channels and applies
/// `exp(2*i*xphase(l,iph))/(2*l+1)`.
pub fn ldos_fmsdos_trace(input: LdosFmsdosTraceInput<'_>) -> Result<Array2<Complex>, DensityError> {
    validate_ldos_fmsdos_trace_input(input)?;

    let potential_count = input.scattering_matrices.shape()[2];
    let phase_offset = input.phase_shifts.shape()[1] / 2;
    let mut trace = Array2::<Complex>::zeros((input.angular_count, potential_count));
    let imaginary = Complex::new(0.0, 1.0);

    for potential in 0..potential_count {
        for angular in 0..input.angular_count {
            let channel_start = angular
                .checked_mul(angular)
                .ok_or(DensityError::InvalidIndex {
                    name: "angular_count",
                    index: input.angular_count,
                })?;
            let magnetic_count = angular
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or(DensityError::InvalidIndex {
                    name: "angular_count",
                    index: input.angular_count,
                })?;
            let mut diagonal_sum = Complex::new(0.0, 0.0);
            for magnetic in 0..magnetic_count {
                let channel = channel_start + magnetic;
                let value = input.scattering_matrices[(channel, channel, potential)];
                validate_complex32_scalar("ldos_fmsdos_gg", value)?;
                diagonal_sum += widen_complex32(value);
            }
            let phase = input.phase_shifts[(input.spin_index, phase_offset + angular, potential)];
            validate_complex32_scalar("ldos_fmsdos_phase", phase)?;
            let normalization = (2 * angular + 1) as Real;
            let value =
                diagonal_sum * (imaginary * 2.0 * widen_complex32(phase)).exp() / normalization;
            validate_complex_scalar("ldos_fmsdos_trace", value)?;
            trace[(angular, potential)] = value;
        }
    }

    Ok(trace)
}

/// Project a full energy grid of FEFF `fmsdos` traces into `gtrNN.bin` order.
///
/// The one-energy primitive follows the FEFF work-array order `(l, iph)`. This
/// adapter returns `(energy, potential, angular)`, which is the binary handoff
/// layout consumed by `gtrNN.bin`.
pub fn ldos_fmsdos_trace_grid(
    input: LdosFmsdosTraceGridInput<'_>,
) -> Result<Array3<Complex>, DensityError> {
    validate_ldos_fmsdos_trace_grid_input(input)?;

    let energy_count = input.scattering_matrices.shape()[0];
    let potential_count = input.scattering_matrices.shape()[3];
    let mut traces = Array3::<Complex>::zeros((energy_count, potential_count, input.angular_count));

    for energy_index in 0..energy_count {
        let trace = ldos_fmsdos_trace(LdosFmsdosTraceInput {
            scattering_matrices: input.scattering_matrices.index_axis(Axis(0), energy_index),
            phase_shifts: input.phase_shifts.index_axis(Axis(0), energy_index),
            spin_index: input.spin_index,
            angular_count: input.angular_count,
        })?;

        for potential in 0..potential_count {
            for angular in 0..input.angular_count {
                traces[(energy_index, potential, angular)] = trace[(angular, potential)];
            }
        }
    }

    Ok(traces)
}

/// Port the post-solver LDOS density integrals in FEFF `LDOS/rhol.f90`.
///
/// This helper starts at the point where `rhol` already has normalized regular
/// and irregular radial solutions. It evaluates the `csomm2` integrals that
/// populate FEFF `xrhole(l,ie)` and `xrhoce(l,ie)`.
pub fn ldos_rhol_density(input: LdosRholDensityInput<'_>) -> Result<LdosRholDensity, DensityError> {
    validate_ldos_rhol_density_input(input)?;

    let radii = input.radii.to_vec();
    let regular_large = input.regular_large.to_vec();
    let regular_small = input.regular_small.to_vec();
    let irregular_large = input.irregular_large.to_vec();
    let irregular_small = input.irregular_small.to_vec();
    let integration_len = ldos_rhol_csomm2_integration_len(&radii, input.norman_radius)?;

    let small_component_factor = ldos_relativistic_small_component_factor(input.wave_number)?;
    let density_scale = ((2 * input.angular_momentum + 1) as Real)
        / (Complex::new(1.0, 0.0) + small_component_factor * small_component_factor)
        / PI
        * input.wave_number
        * (4.0 / FEFF_HARTREE_EV);
    validate_complex_scalar("ldos_rhol_density_scale", density_scale)?;

    let scattering_integrand = regular_large
        .iter()
        .zip(regular_small.iter())
        .map(|(&large, &small)| large * large + small * small)
        .collect::<Vec<_>>();
    validate_complex_values(
        "ldos_rhol_scattering_integrand",
        scattering_integrand.iter().copied(),
    )?;
    let near_origin_power = (2 * input.angular_momentum + 2) as Real;
    let scattering_integral = csomm2(
        &radii[..integration_len],
        &scattering_integrand[..integration_len],
        input.radial_step,
        near_origin_power,
        input.norman_radius,
    )?;
    let scattering_ldos = scattering_integral * density_scale;
    validate_complex_scalar("ldos_rhol_scattering_ldos", scattering_ldos)?;

    let imaginary = Complex::new(0.0, 1.0);
    let embedded_integrand = irregular_large
        .iter()
        .zip(regular_large.iter())
        .zip(irregular_small.iter().zip(regular_small.iter()))
        .map(
            |((&irregular_large, &regular_large), (&irregular_small, &regular_small))| {
                irregular_large * regular_large - imaginary * regular_large * regular_large
                    + irregular_small * regular_small
                    - imaginary * regular_small * regular_small
            },
        )
        .collect::<Vec<_>>();
    validate_complex_values(
        "ldos_rhol_embedded_integrand",
        embedded_integrand.iter().copied(),
    )?;
    let embedded_integral = csomm2(
        &radii[..integration_len],
        &embedded_integrand[..integration_len],
        input.radial_step,
        1.0,
        input.norman_radius,
    )?;
    let embedded_ldos = -(embedded_integral * density_scale).im;
    validate_real_scalar("ldos_rhol_embedded_ldos", embedded_ldos)?;

    Ok(LdosRholDensity {
        scattering_ldos,
        embedded_ldos,
        density_scale,
    })
}

/// Port the post-solver density work-array assembly in FEFF `POT/rholie.f90`.
///
/// This helper starts after the regular and irregular radial solutions have
/// been normalized. Unlike LDOS `rhol`, POT keeps complex `xrhoce` values and
/// interpolates `yrhole`/`yrhoce` onto the 0.05 radial grid consumed by `ff2g`.
pub fn pot_rholie_density(
    input: PotRholieDensityInput<'_>,
) -> Result<PotRholieDensity, DensityError> {
    validate_pot_rholie_density_input(input)?;

    let source_radii = input.source_radii.to_vec();
    let regular_large = input.regular_large.to_vec();
    let regular_small = input.regular_small.to_vec();
    let irregular_large = input.irregular_large.to_vec();
    let irregular_small = input.irregular_small.to_vec();
    let output_radii = input.output_radii.to_vec();
    let integration_len = ldos_rhol_csomm2_integration_len(&source_radii, input.norman_radius)?;

    let small_component_factor = ldos_relativistic_small_component_factor(input.wave_number)?;
    let density_scale = ((2 * input.angular_momentum + 1) as Real)
        / (Complex::new(1.0, 0.0) + small_component_factor * small_component_factor)
        / PI
        * input.wave_number
        * 2.0;
    validate_complex_scalar("pot_rholie_density_scale", density_scale)?;

    let scattering_integrand = regular_large
        .iter()
        .zip(regular_small.iter())
        .map(|(&large, &small)| large * large + small * small)
        .collect::<Vec<_>>();
    validate_complex_values(
        "pot_rholie_scattering_integrand",
        scattering_integrand.iter().copied(),
    )?;

    let near_origin_power = (2 * input.angular_momentum + 2) as Real;
    let scattering_integral = csomm2(
        &source_radii[..integration_len],
        &scattering_integrand[..integration_len],
        input.radial_step,
        near_origin_power,
        input.norman_radius,
    )?;
    let scattering_ldos = scattering_integral * density_scale;
    validate_complex_scalar("pot_rholie_scattering_ldos", scattering_ldos)?;

    let mut scattering_density = Array1::<Complex>::zeros(output_radii.len());
    for (row, &radius) in output_radii.iter().enumerate() {
        scattering_density[row] =
            terpc(&source_radii, &scattering_integrand, 3, radius)?.value * density_scale;
    }
    validate_complex_values(
        "pot_rholie_scattering_density",
        scattering_density.iter().copied(),
    )?;

    let imaginary = Complex::new(0.0, 1.0);
    let embedded_integrand = irregular_large
        .iter()
        .zip(regular_large.iter())
        .zip(irregular_small.iter().zip(regular_small.iter()))
        .map(
            |((&irregular_large, &regular_large), (&irregular_small, &regular_small))| {
                irregular_large * regular_large - imaginary * regular_large * regular_large
                    + irregular_small * regular_small
                    - imaginary * regular_small * regular_small
            },
        )
        .collect::<Vec<_>>();
    validate_complex_values(
        "pot_rholie_embedded_integrand",
        embedded_integrand.iter().copied(),
    )?;

    let embedded_integral = csomm2(
        &source_radii[..integration_len],
        &embedded_integrand[..integration_len],
        input.radial_step,
        1.0,
        input.norman_radius,
    )?;
    let embedded_ldos = -(embedded_integral * density_scale);
    validate_complex_scalar("pot_rholie_embedded_ldos", embedded_ldos)?;

    let mut embedded_density = Array1::<Complex>::zeros(output_radii.len());
    for (row, &radius) in output_radii.iter().enumerate() {
        embedded_density[row] =
            -(terpc(&source_radii, &embedded_integrand, 3, radius)?.value * density_scale);
    }
    validate_complex_values(
        "pot_rholie_embedded_density",
        embedded_density.iter().copied(),
    )?;

    Ok(PotRholieDensity {
        scattering_ldos,
        embedded_ldos,
        scattering_density,
        embedded_density,
        density_scale,
    })
}

/// Assemble FEFF `POT/rholie.f90` work arrays for all angular channels at one energy.
pub fn pot_rholie_density_grid(
    input: PotRholieDensityGridInput<'_>,
) -> Result<PotRholieDensityGrid, DensityError> {
    validate_pot_rholie_density_grid_input(input)?;

    let radial_count = input.output_radii.len();
    let mut scattering_ldos = Array1::<Complex>::zeros(input.angular_count);
    let mut embedded_ldos = Array1::<Complex>::zeros(input.angular_count);
    let mut scattering_density = Array2::<Complex>::zeros((radial_count, input.angular_count));
    let mut embedded_density = Array1::<Complex>::zeros(radial_count);
    let mut density_scale = Array1::<Complex>::zeros(input.angular_count);

    for angular in 0..input.angular_count {
        let density = pot_rholie_density(PotRholieDensityInput {
            source_radii: input.source_radii,
            output_radii: input.output_radii,
            regular_large: input.regular_large.index_axis(Axis(0), angular),
            regular_small: input.regular_small.index_axis(Axis(0), angular),
            irregular_large: input.irregular_large.index_axis(Axis(0), angular),
            irregular_small: input.irregular_small.index_axis(Axis(0), angular),
            radial_step: input.radial_step,
            norman_radius: input.norman_radius,
            wave_number: input.wave_number,
            angular_momentum: angular,
        })?;

        scattering_ldos[angular] = density.scattering_ldos;
        embedded_ldos[angular] = density.embedded_ldos;
        density_scale[angular] = density.density_scale;
        for radial in 0..radial_count {
            scattering_density[(radial, angular)] = density.scattering_density[radial];
            embedded_density[radial] += density.embedded_density[radial];
        }
    }

    validate_complex_values(
        "pot_rholie_grid_scattering_ldos",
        scattering_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_grid_embedded_ldos",
        embedded_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_grid_scattering_density",
        scattering_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_grid_embedded_density",
        embedded_density.iter().copied(),
    )?;

    Ok(PotRholieDensityGrid {
        scattering_ldos,
        embedded_ldos,
        scattering_density,
        embedded_density,
        density_scale,
    })
}

/// Build FEFF `POT/scmt.f90` source rows from solved radial channels and FMS traces.
///
/// This is the multi-energy, multi-potential lift of `rholie`: it preserves
/// the source-row ordering used by `scmt`, assembles `xrhole`, `xrhoce`,
/// `yrhole`, and `yrhoce`, and carries the supplied FMS `gtr` table forward
/// for the contour-loop driver.
pub fn pot_scf_contour_source_rows(
    input: PotScfContourSourceRowsInput<'_>,
) -> Result<PotScfContourSourceRows, DensityError> {
    validate_pot_scf_contour_source_rows_input(input)?;

    let point_count = input.source_energies.len();
    let potential_count = input.highest_potential_index + 1;
    let radial_count = input.output_radii.len();
    let angular_count = input.angular_count;
    let mut scattering_ldos =
        Array3::<Complex>::zeros((point_count, angular_count, potential_count));
    let mut embedded_ldos_source =
        Array3::<Complex>::zeros((point_count, angular_count, potential_count));
    let mut scattering_density =
        Array4::<Complex>::zeros((point_count, radial_count, angular_count, potential_count));
    let mut embedded_density_source =
        Array3::<Complex>::zeros((point_count, radial_count, potential_count));
    let mut density_scale = Array3::<Complex>::zeros((point_count, angular_count, potential_count));

    for point in 0..point_count {
        for potential in 0..potential_count {
            let regular_large_point = input.regular_large.index_axis(Axis(0), point);
            let regular_small_point = input.regular_small.index_axis(Axis(0), point);
            let irregular_large_point = input.irregular_large.index_axis(Axis(0), point);
            let irregular_small_point = input.irregular_small.index_axis(Axis(0), point);
            let grid = pot_rholie_density_grid(PotRholieDensityGridInput {
                source_radii: input.source_radii,
                output_radii: input.output_radii,
                regular_large: regular_large_point.index_axis(Axis(0), potential),
                regular_small: regular_small_point.index_axis(Axis(0), potential),
                irregular_large: irregular_large_point.index_axis(Axis(0), potential),
                irregular_small: irregular_small_point.index_axis(Axis(0), potential),
                radial_step: input.radial_step,
                norman_radius: input.norman_radii[potential],
                wave_number: input.wave_numbers[(point, potential)],
                angular_count,
            })?;

            scattering_ldos
                .index_axis_mut(Axis(0), point)
                .index_axis_mut(Axis(1), potential)
                .assign(&grid.scattering_ldos);
            embedded_ldos_source
                .index_axis_mut(Axis(0), point)
                .index_axis_mut(Axis(1), potential)
                .assign(&grid.embedded_ldos);
            scattering_density
                .index_axis_mut(Axis(0), point)
                .index_axis_mut(Axis(2), potential)
                .assign(&grid.scattering_density);
            embedded_density_source
                .index_axis_mut(Axis(0), point)
                .index_axis_mut(Axis(1), potential)
                .assign(&grid.embedded_density);
            density_scale
                .index_axis_mut(Axis(0), point)
                .index_axis_mut(Axis(1), potential)
                .assign(&grid.density_scale);
        }
    }

    validate_complex_values(
        "pot_scf_contour_source_rows_scattering_ldos",
        scattering_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_source_rows_embedded_ldos",
        embedded_ldos_source.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_source_rows_scattering_density",
        scattering_density.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_source_rows_embedded_density",
        embedded_density_source.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_contour_source_rows_density_scale",
        density_scale.iter().copied(),
    )?;

    Ok(PotScfContourSourceRows {
        source_energies: input.source_energies.to_owned(),
        scattering_trace: input.scattering_trace.to_owned(),
        scattering_ldos,
        embedded_ldos_source,
        scattering_density,
        embedded_density_source,
        density_scale,
    })
}

/// Compose one FEFF `POT/scmt.f90` energy/potential density update.
///
/// This source-backed subdriver preserves the FEFF stage boundary: `rholie`
/// builds complex radial density work arrays from solved radial channels, then
/// `ff2g` folds in the FMS trace and updates contour-integrated valence state.
pub fn pot_scf_energy_density(
    input: PotScfEnergyDensityInput<'_>,
) -> Result<PotScfEnergyDensity, DensityError> {
    let rholie = pot_rholie_density_grid(PotRholieDensityGridInput {
        source_radii: input.source_radii,
        output_radii: input.output_radii,
        regular_large: input.regular_large,
        regular_small: input.regular_small,
        irregular_large: input.irregular_large,
        irregular_small: input.irregular_small,
        radial_step: input.radial_step,
        norman_radius: input.norman_radius,
        wave_number: input.wave_number,
        angular_count: input.angular_count,
    })?;

    let valence = update_valence_density(ValenceDensityUpdateInput {
        scattering_trace: input.scattering_trace,
        potential_index: input.potential_index,
        energy_index: input.energy_index,
        last_radial_index: input.last_radial_index,
        scattering_ldos: rholie.scattering_ldos.view(),
        embedded_ldos: input.embedded_ldos,
        previous_ldos: input.previous_ldos,
        scattering_density: rholie.scattering_density.view(),
        embedded_density: rholie.embedded_density.view(),
        previous_density: input.previous_density,
        valence_density: input.valence_density,
        occupancy_by_l: input.occupancy_by_l,
        current_energy: input.current_energy,
        previous_energy: input.previous_energy,
        potential_multiplicity: input.potential_multiplicity,
        current_floor: input.current_floor,
        previous_floor: input.previous_floor,
        left_sum: input.left_sum,
        right_sum: input.right_sum,
        total_electron_count: input.total_electron_count,
        include_high_l: input.include_high_l,
    })?;

    Ok(PotScfEnergyDensity { rholie, valence })
}

/// Port the exact free-particle tail loop in FEFF `LDOS/rhol.f90`.
///
/// FEFF overwrites rows `jri:ilast` with Bessel/Neumann combinations after the
/// regular and irregular radial solver passes. This LDOS-facing adapter reuses
/// the shared RHORRP implementation because the tail formula is identical.
pub fn ldos_rhol_exact_radial_tail(
    input: LdosRholExactRadialTailInput<'_>,
) -> Result<LdosRholExactRadialTail, DensityError> {
    let radii = input.radii.to_vec();
    let tail = rhorrp_exact_radial_tail(RhorrpExactRadialTailInput {
        radii: &radii,
        start_index_1based: input.start_index_1based,
        angular_momentum: input.angular_momentum,
        phase_shift: input.phase_shift,
        wave_number: input.wave_number,
    })?;

    Ok(LdosRholExactRadialTail {
        start_index_1based: tail.start_index_1based,
        regular_large: tail.regular_large_components,
        regular_small: tail.regular_small_components,
        irregular_large: tail.irregular_large_components,
        irregular_small: tail.irregular_small_components,
    })
}

/// Assemble normalized FEFF `rhol` radial components from raw `dfovrg` outputs.
///
/// This ports the vector operations between the two `dfovrg` calls and the
/// density integrals in `LDOS/rhol.f90`: regular-solution normalization,
/// irregular Wronskian scaling, irregular row replacement, and exact-tail
/// overwrite. Unlike RHORRP, LDOS does not apply the `fix_irreg` origin
/// smoothing branch.
pub fn ldos_rhol_assemble_radial_components(
    input: LdosRholRadialAssemblyInput<'_>,
) -> Result<LdosRholRadialAssembly, DensityError> {
    validate_ldos_rhol_radial_assembly_input(input)?;

    let regular_solution_scale = rhorrp_regular_solution_scale(RhorrpRegularSolutionScaleInput {
        phase_amplitude: input.phase_amplitude,
    })?
    .scale;
    let mut regular_large = input
        .raw_regular_large
        .mapv(|value| value * regular_solution_scale);
    let mut regular_small = input
        .raw_regular_small
        .mapv(|value| value * regular_solution_scale);
    validate_complex_values("ldos_rhol_regular_large", regular_large.iter().copied())?;
    validate_complex_values("ldos_rhol_regular_small", regular_small.iter().copied())?;

    let match_index = input.match_index_1based - 1;
    let wronskian = rhorrp_irregular_wronskian_scale(RhorrpIrregularWronskianScaleInput {
        phase_shift: input.phase_shift,
        wave_number: input.wave_number,
        regular_large_at_match: regular_large[match_index],
        regular_small_at_match: regular_small[match_index],
        irregular_large_at_match: input.raw_irregular_large[match_index],
        irregular_small_at_match: input.raw_irregular_small[match_index],
    })?;

    let mut irregular_large = Array1::<Complex>::zeros(input.radii.len());
    let mut irregular_small = Array1::<Complex>::zeros(input.radii.len());
    for row in 0..input.radii.len() {
        let transformed =
            rhorrp_irregular_solution_transform(RhorrpIrregularSolutionTransformInput {
                phase_factor: wronskian.phase_factor,
                reciprocal_wave_scale: wronskian.reciprocal_wave_scale,
                regular_large_component: regular_large[row],
                regular_small_component: regular_small[row],
                irregular_large_component: input.raw_irregular_large[row],
                irregular_small_component: input.raw_irregular_small[row],
            })?;
        irregular_large[row] = transformed.large_component;
        irregular_small[row] = transformed.small_component;
    }

    let exact_tail = ldos_rhol_exact_radial_tail(LdosRholExactRadialTailInput {
        radii: input.radii,
        start_index_1based: input.exact_tail_start_index_1based,
        angular_momentum: input.angular_momentum,
        phase_shift: input.phase_shift,
        wave_number: input.wave_number,
    })?;
    let tail_start = exact_tail.start_index_1based - 1;
    for offset in 0..exact_tail.row_count() {
        let row = tail_start + offset;
        regular_large[row] = exact_tail.regular_large[offset];
        regular_small[row] = exact_tail.regular_small[offset];
        irregular_large[row] = exact_tail.irregular_large[offset];
        irregular_small[row] = exact_tail.irregular_small[offset];
    }

    Ok(LdosRholRadialAssembly {
        regular_solution_scale,
        irregular_phase_factor: wronskian.phase_factor,
        irregular_reciprocal_wave_scale: wronskian.reciprocal_wave_scale,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
    })
}

/// Solve one FEFF `LDOS/rhol.f90` radial channel from source FOVRG inputs.
///
/// This composes the regular `dfovrg` pass, muffin-tin `phamp` match,
/// irregular-boundary setup, irregular `dfovrg` pass, and LDOS radial assembly
/// for one `(energy, l, potential)` channel.
pub fn ldos_rhol_channel(input: LdosRholChannelInput<'_>) -> Result<LdosRholChannel, DensityError> {
    validate_complex_scalar("ldos_rhol_channel_wave_number", input.wave_number)?;

    let zero = Complex::new(0.0, 0.0);
    let regular_input = FovrgDiracSolverInput {
        irregular: false,
        muffin_tin_large_component: zero,
        muffin_tin_small_component: zero,
        ..input.solver
    };
    let regular_solution = fovrg_dirac_solver(regular_input)?;
    let muffin_tin_match = rhorrp_muffin_tin_match(RhorrpMuffinTinMatchInput {
        muffin_tin_radius: input.solver.muffin_tin_radius,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: input.solver.target_kappa,
    })?;
    let irregular_initial =
        rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
            muffin_tin_radius: input.solver.muffin_tin_radius,
            phase_shift: muffin_tin_match.phase_shift,
            wave_number: input.wave_number,
            bessel_j_l: muffin_tin_match.bessel_j_l,
            neumann_l: muffin_tin_match.neumann_l,
            bessel_j_l_plus_1: muffin_tin_match.bessel_j_l_plus_1,
            neumann_l_plus_1: muffin_tin_match.neumann_l_plus_1,
        })?;

    let irregular_input = FovrgDiracSolverInput {
        irregular: true,
        muffin_tin_large_component: irregular_initial.large_component,
        muffin_tin_small_component: irregular_initial.small_component,
        ..input.solver
    };
    let irregular_solution = fovrg_dirac_solver(irregular_input)?;

    ensure_length_match(
        "regular_solution.large_component",
        regular_solution.large_component.len(),
        "regular_solution.active_len",
        regular_solution.active_len,
    )?;
    ensure_length_match(
        "regular_solution.small_component",
        regular_solution.small_component.len(),
        "regular_solution.active_len",
        regular_solution.active_len,
    )?;
    ensure_length_match(
        "irregular_solution.large_component",
        irregular_solution.large_component.len(),
        "regular_solution.active_len",
        regular_solution.active_len,
    )?;
    ensure_length_match(
        "irregular_solution.small_component",
        irregular_solution.small_component.len(),
        "regular_solution.active_len",
        regular_solution.active_len,
    )?;
    ensure_len(
        "ldos_rhol_channel_radii",
        input.solver.radii.len(),
        regular_solution.active_len,
    )?;

    let radial_count = regular_solution
        .active_len
        .min(input.solver.target_last_index + 1);
    ensure_len("ldos_rhol_channel_radii", radial_count, 4)?;
    let active_radii = Array1::from_iter(input.solver.radii.iter().take(radial_count).copied());
    let match_index_1based =
        input
            .solver
            .radial_match_index
            .checked_add(1)
            .ok_or(DensityError::InvalidIndex {
                name: "ldos_rhol_channel_radial_match_index",
                index: input.solver.radial_match_index,
            })?;
    let radial_components = ldos_rhol_assemble_radial_components(LdosRholRadialAssemblyInput {
        radii: active_radii.view(),
        raw_regular_large: regular_solution
            .large_component
            .slice_axis(Axis(0), Slice::from(..radial_count)),
        raw_regular_small: regular_solution
            .small_component
            .slice_axis(Axis(0), Slice::from(..radial_count)),
        raw_irregular_large: irregular_solution
            .large_component
            .slice_axis(Axis(0), Slice::from(..radial_count)),
        raw_irregular_small: irregular_solution
            .small_component
            .slice_axis(Axis(0), Slice::from(..radial_count)),
        phase_shift: muffin_tin_match.phase_shift,
        phase_amplitude: muffin_tin_match.phase_amplitude,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        match_index_1based,
        exact_tail_start_index_1based: match_index_1based,
    })?;

    Ok(LdosRholChannel {
        phase_shift: muffin_tin_match.phase_shift,
        phase_amplitude: muffin_tin_match.phase_amplitude,
        irregular_initial_large: irregular_initial.large_component,
        irregular_initial_small: irregular_initial.small_component,
        radial_components,
        regular_active_len: regular_solution.active_len,
        irregular_active_len: irregular_solution.active_len,
        regular_iteration_count: regular_solution.iteration_count,
        irregular_iteration_count: irregular_solution.iteration_count,
        difficult_iterations: regular_solution.difficult_iterations
            + irregular_solution.difficult_iterations,
    })
}

/// Assemble FEFF `xrhole(l,ie)` and `xrhoce(l,ie)` for a full LDOS energy grid.
///
/// The input radial solutions are shaped `(energy, angular, radial)`, matching
/// the natural driver loop in `rhol.f90`. The returned arrays are shaped
/// `(angular, energy)`, matching FEFF's final `ff2rho` work arrays.
pub fn ldos_rhol_density_grid(
    input: LdosRholDensityGridInput<'_>,
) -> Result<LdosRholDensityGrid, DensityError> {
    validate_ldos_rhol_density_grid_input(input)?;

    let energy_count = input.wave_numbers.len();
    let mut scattering_ldos = Array2::<Complex>::zeros((input.angular_count, energy_count));
    let mut embedded_ldos = Array2::<Real>::zeros((input.angular_count, energy_count));
    let mut density_scale = Array2::<Complex>::zeros((input.angular_count, energy_count));

    for energy_index in 0..energy_count {
        for angular in 0..input.angular_count {
            let regular_large_energy = input.regular_large.index_axis(Axis(0), energy_index);
            let regular_small_energy = input.regular_small.index_axis(Axis(0), energy_index);
            let irregular_large_energy = input.irregular_large.index_axis(Axis(0), energy_index);
            let irregular_small_energy = input.irregular_small.index_axis(Axis(0), energy_index);
            let density = ldos_rhol_density(LdosRholDensityInput {
                radii: input.radii,
                regular_large: regular_large_energy.index_axis(Axis(0), angular),
                regular_small: regular_small_energy.index_axis(Axis(0), angular),
                irregular_large: irregular_large_energy.index_axis(Axis(0), angular),
                irregular_small: irregular_small_energy.index_axis(Axis(0), angular),
                radial_step: input.radial_step,
                norman_radius: input.norman_radius,
                wave_number: input.wave_numbers[energy_index],
                angular_momentum: angular,
            })?;
            scattering_ldos[(angular, energy_index)] = density.scattering_ldos;
            embedded_ldos[(angular, energy_index)] = density.embedded_ldos;
            density_scale[(angular, energy_index)] = density.density_scale;
        }
    }

    Ok(LdosRholDensityGrid {
        scattering_ldos,
        embedded_ldos,
        density_scale,
    })
}

/// Solve source-backed FEFF `rhol` channels and emit non-spin `ff2rho` tables.
///
/// This is the production-shaped bridge between prepared potential/radial
/// source inputs and FEFF `ldosNN.dat`/`rhocNN.dat` payloads for the
/// non-full-potential, non-spin branch. Solver inputs are traversed in FEFF
/// order `(energy, l)`, converted to `xrhole(l,ie)`/`xrhoce(l,ie)`, then passed
/// through the Rust `ff2rho` table adapter.
pub fn ldos_rhol_table_driver(
    input: LdosRholTableDriverInput<'_>,
) -> Result<LdosRholTableDriver, DensityError> {
    validate_ldos_rhol_table_driver_input(input)?;

    let energy_count = input.energy_grid_hartree.len();
    let channel_count = input.solvers.len();
    let mut channels = Vec::with_capacity(channel_count);
    for energy_index in 0..energy_count {
        for angular in 0..input.angular_count {
            let solver_index = ldos_rhol_solver_index(energy_index, angular, input.angular_count)?;
            channels.push(ldos_rhol_channel(LdosRholChannelInput {
                solver: input.solvers[solver_index],
                angular_momentum: angular,
                wave_number: input.wave_numbers[energy_index],
            })?);
        }
    }

    let radial_count = channels
        .first()
        .map(|channel| channel.radial_components.row_count())
        .ok_or(DensityError::LengthTooShort {
            name: "ldos_rhol_table_solvers",
            required: 1,
            actual: 0,
        })?;
    let radii = ldos_rhol_shared_radii(input.solvers, radial_count)?;

    let mut regular_large =
        Array3::<Complex>::zeros((energy_count, input.angular_count, radial_count));
    let mut regular_small =
        Array3::<Complex>::zeros((energy_count, input.angular_count, radial_count));
    let mut irregular_large =
        Array3::<Complex>::zeros((energy_count, input.angular_count, radial_count));
    let mut irregular_small =
        Array3::<Complex>::zeros((energy_count, input.angular_count, radial_count));
    let mut phase_shifts = Array2::<Complex>::zeros((input.angular_count, energy_count));
    let mut phase_amplitudes = Array2::<Complex>::zeros((input.angular_count, energy_count));
    let mut regular_iteration_counts = Array2::<usize>::zeros((input.angular_count, energy_count));
    let mut irregular_iteration_counts =
        Array2::<usize>::zeros((input.angular_count, energy_count));
    let mut difficult_iterations = Array2::<usize>::zeros((input.angular_count, energy_count));

    for energy_index in 0..energy_count {
        for angular in 0..input.angular_count {
            let solver_index = ldos_rhol_solver_index(energy_index, angular, input.angular_count)?;
            let channel = &channels[solver_index];
            ensure_length_match(
                "ldos_rhol_channel_rows",
                channel.radial_components.row_count(),
                "ldos_rhol_table_radial_count",
                radial_count,
            )?;

            phase_shifts[(angular, energy_index)] = channel.phase_shift;
            phase_amplitudes[(angular, energy_index)] = channel.phase_amplitude;
            regular_iteration_counts[(angular, energy_index)] = channel.regular_iteration_count;
            irregular_iteration_counts[(angular, energy_index)] = channel.irregular_iteration_count;
            difficult_iterations[(angular, energy_index)] = channel.difficult_iterations;
            for row in 0..radial_count {
                regular_large[(energy_index, angular, row)] =
                    channel.radial_components.regular_large[row];
                regular_small[(energy_index, angular, row)] =
                    channel.radial_components.regular_small[row];
                irregular_large[(energy_index, angular, row)] =
                    channel.radial_components.irregular_large[row];
                irregular_small[(energy_index, angular, row)] =
                    channel.radial_components.irregular_small[row];
            }
        }
    }

    let density_grid = ldos_rhol_density_grid(LdosRholDensityGridInput {
        radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        wave_numbers: input.wave_numbers,
        radial_step: input.radial_step,
        norman_radius: input.norman_radius,
        angular_count: input.angular_count,
    })?;
    let tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: input.energy_grid_hartree,
        embedded_ldos: density_grid.embedded_ldos.view(),
        scattering_ldos: density_grid.scattering_ldos.view(),
        scattering_trace: input.scattering_trace,
        angular_count: input.angular_count,
        apply_scattering: input.apply_scattering,
    })?;

    Ok(LdosRholTableDriver {
        tables,
        density_grid,
        phase_shifts,
        phase_amplitudes,
        regular_iteration_counts,
        irregular_iteration_counts,
        difficult_iterations,
    })
}

/// Assemble one potential's non-spin LDOS tables from RHORRP wavefunction grids.
///
/// This is the source-assembly bridge used after the shared `pot.bin` /
/// `config.dat` / `phase.bin` radial setup has produced FEFF-compatible
/// `prel`, `pnel`, `qrel`, and `qnel` tables. It selects one `iph` block,
/// evaluates LDOS `rhol` density integrals, and feeds the Rust `ff2rho`
/// adapter.
pub fn ldos_rhol_wavefunction_tables(
    input: LdosRholWavefunctionTablesInput<'_>,
) -> Result<LdosRholWavefunctionTables, DensityError> {
    validate_ldos_rhol_wavefunction_tables_input(input)?;

    let potential = input.potential_index;
    let wave_numbers = input
        .wavefunctions
        .wave_numbers
        .index_axis(Axis(1), potential)
        .to_owned();
    let regular_large = input
        .wavefunctions
        .regular_large
        .index_axis(Axis(3), potential);
    let regular_small = input
        .wavefunctions
        .regular_small
        .index_axis(Axis(3), potential);
    let irregular_large = input
        .wavefunctions
        .irregular_large
        .index_axis(Axis(3), potential);
    let irregular_small = input
        .wavefunctions
        .irregular_small
        .index_axis(Axis(3), potential);

    let density_grid = ldos_rhol_density_grid(LdosRholDensityGridInput {
        radii: input.radii,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        wave_numbers: wave_numbers.view(),
        radial_step: input.radial_step,
        norman_radius: input.norman_radius,
        angular_count: input.angular_count,
    })?;
    let tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: input.energy_grid_hartree,
        embedded_ldos: density_grid.embedded_ldos.view(),
        scattering_ldos: density_grid.scattering_ldos.view(),
        scattering_trace: input.scattering_trace,
        angular_count: input.angular_count,
        apply_scattering: input.apply_scattering,
    })?;

    Ok(LdosRholWavefunctionTables {
        tables,
        density_grid,
        wave_numbers,
    })
}

fn validate_ldos_ff2rho_input(input: LdosFf2rhoInput<'_>) -> Result<(), DensityError> {
    ensure_len("energy_grid_hartree", input.energy_grid_hartree.len(), 1)?;
    ensure_len("angular_count", input.angular_count, 1)?;
    let energy_count = input.energy_grid_hartree.len();

    ensure_shape(
        "embedded_ldos",
        input.embedded_ldos.shape(),
        input.angular_count,
        energy_count,
    )?;
    validate_complex_values(
        "energy_grid_hartree",
        input.energy_grid_hartree.iter().copied(),
    )?;

    if input.apply_scattering {
        ensure_shape(
            "scattering_ldos",
            input.scattering_ldos.shape(),
            input.angular_count,
            energy_count,
        )?;
        ensure_shape(
            "scattering_trace",
            input.scattering_trace.shape(),
            input.angular_count,
            energy_count,
        )?;
    }

    Ok(())
}

fn validate_ldos_rhol_wavefunction_tables_input(
    input: LdosRholWavefunctionTablesInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("rhol_energy", input.energy_grid_hartree.len(), 1)?;
    ensure_len("angular_count", input.angular_count, 1)?;
    let wavefunctions = input.wavefunctions;
    let energy_count = wavefunctions.energy_count();
    let angular_count = wavefunctions.angular_momentum_count();
    let radial_count = wavefunctions.radial_count();
    let potential_count = wavefunctions.potential_count();
    ensure_len("rhol_wavefunctions_energy", energy_count, 1)?;
    ensure_len(
        "rhol_wavefunctions_angular",
        angular_count,
        input.angular_count,
    )?;
    ensure_len("rhol_wavefunctions_radial", radial_count, 4)?;
    ensure_length_match(
        "energy_grid_hartree",
        input.energy_grid_hartree.len(),
        "rhol_wavefunctions_energy",
        energy_count,
    )?;
    if input.potential_index >= potential_count {
        return Err(DensityError::InvalidPotentialIndex {
            name: "ldos_rhol_wavefunction_tables_potential",
            index: input.potential_index,
            available: potential_count,
        });
    }
    ensure_len("radii", input.radii.len(), radial_count)?;
    validate_positive_real_values("radii", input.radii)?;
    validate_complex_values(
        "energy_grid_hartree",
        input.energy_grid_hartree.iter().copied(),
    )?;
    validate_positive_real_scalar("radial_step", input.radial_step)?;
    validate_positive_real_scalar("norman_radius", input.norman_radius)?;
    let (wave_energy, wave_potential) = wavefunctions.wave_numbers.dim();
    ensure_shape(
        "wave_numbers",
        &[wave_energy, wave_potential],
        energy_count,
        potential_count,
    )?;
    if input.apply_scattering {
        ensure_shape(
            "scattering_trace",
            input.scattering_trace.shape(),
            input.angular_count,
            energy_count,
        )?;
    }
    Ok(())
}

fn validate_ldos_rhol_table_driver_input(
    input: LdosRholTableDriverInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("rhol_energy", input.energy_grid_hartree.len(), 1)?;
    ensure_len("angular_count", input.angular_count, 1)?;
    ensure_length_match(
        "energy_grid_hartree",
        input.energy_grid_hartree.len(),
        "wave_numbers",
        input.wave_numbers.len(),
    )?;
    let solver_count = input
        .energy_grid_hartree
        .len()
        .checked_mul(input.angular_count)
        .ok_or(DensityError::InvalidIndex {
            name: "rhol_energy*angular_count",
            index: input.angular_count,
        })?;
    ensure_length_match(
        "ldos_rhol_table_solvers",
        input.solvers.len(),
        "rhol_energy*angular_count",
        solver_count,
    )?;
    validate_complex_values(
        "energy_grid_hartree",
        input.energy_grid_hartree.iter().copied(),
    )?;
    validate_complex_values("wave_numbers", input.wave_numbers.iter().copied())?;
    validate_positive_real_scalar("radial_step", input.radial_step)?;
    validate_positive_real_scalar("norman_radius", input.norman_radius)?;
    if input.apply_scattering {
        ensure_shape(
            "scattering_trace",
            input.scattering_trace.shape(),
            input.angular_count,
            input.energy_grid_hartree.len(),
        )?;
    }
    Ok(())
}

fn ldos_rhol_solver_index(
    energy_index: usize,
    angular: usize,
    angular_count: usize,
) -> Result<usize, DensityError> {
    energy_index
        .checked_mul(angular_count)
        .and_then(|offset| offset.checked_add(angular))
        .ok_or(DensityError::InvalidIndex {
            name: "ldos_rhol_solver_index",
            index: energy_index,
        })
}

fn ldos_rhol_shared_radii(
    solvers: &[FovrgDiracSolverInput<'_>],
    radial_count: usize,
) -> Result<Array1<Real>, DensityError> {
    let first = solvers.first().ok_or(DensityError::LengthTooShort {
        name: "ldos_rhol_table_solvers",
        required: 1,
        actual: 0,
    })?;
    ensure_len(
        "ldos_rhol_table_solver_radii",
        first.radii.len(),
        radial_count,
    )?;
    let radii = Array1::from_iter(first.radii.iter().take(radial_count).copied());
    validate_positive_real_values("ldos_rhol_table_radii", radii.view())?;

    for (solver_index, solver) in solvers.iter().enumerate().skip(1) {
        ensure_len(
            "ldos_rhol_table_solver_radii",
            solver.radii.len(),
            radial_count,
        )?;
        for row in 0..radial_count {
            let actual = solver.radii[row];
            validate_positive_real_scalar("ldos_rhol_table_solver_radius", actual)?;
            let expected = radii[row];
            if actual != expected {
                return Err(DensityError::ValueMismatch {
                    name: "ldos_rhol_table_solver_radii",
                    index: solver_index * radial_count + row,
                    expected,
                    actual,
                });
            }
        }
    }

    Ok(radii)
}

fn validate_ldos_fmsdos_trace_input(input: LdosFmsdosTraceInput<'_>) -> Result<(), DensityError> {
    ensure_len("angular_count", input.angular_count, 1)?;
    let channel_count =
        input
            .angular_count
            .checked_mul(input.angular_count)
            .ok_or(DensityError::InvalidIndex {
                name: "angular_count",
                index: input.angular_count,
            })?;
    ensure_shape3(
        "scattering_matrices",
        input.scattering_matrices.shape(),
        channel_count,
        channel_count,
        1,
    )?;
    let phase_l_count = input.phase_shifts.shape()[1];
    ensure_len("phase_shifts.signed_l", phase_l_count, 1)?;
    let phase_offset = phase_l_count / 2;
    ensure_shape3(
        "phase_shifts",
        input.phase_shifts.shape(),
        input.spin_index + 1,
        phase_offset + input.angular_count,
        input.scattering_matrices.shape()[2],
    )?;
    Ok(())
}

fn validate_ldos_fmsdos_trace_grid_input(
    input: LdosFmsdosTraceGridInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("fmsdos_energy", input.scattering_matrices.shape()[0], 1)?;
    ensure_length_match(
        "scattering_matrices.energy",
        input.scattering_matrices.shape()[0],
        "phase_shifts.energy",
        input.phase_shifts.shape()[0],
    )?;
    Ok(())
}

fn validate_ldos_rhol_density_input(input: LdosRholDensityInput<'_>) -> Result<(), DensityError> {
    ensure_len("radii", input.radii.len(), 4)?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "regular_large",
        input.regular_large.len(),
    )?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "regular_small",
        input.regular_small.len(),
    )?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "irregular_large",
        input.irregular_large.len(),
    )?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "irregular_small",
        input.irregular_small.len(),
    )?;
    validate_positive_real_values("radii", input.radii)?;
    validate_positive_real_scalar("radial_step", input.radial_step)?;
    validate_positive_real_scalar("norman_radius", input.norman_radius)?;
    validate_complex_values("regular_large", input.regular_large.iter().copied())?;
    validate_complex_values("regular_small", input.regular_small.iter().copied())?;
    validate_complex_values("irregular_large", input.irregular_large.iter().copied())?;
    validate_complex_values("irregular_small", input.irregular_small.iter().copied())?;
    validate_complex_scalar("wave_number", input.wave_number)?;
    Ok(())
}

fn validate_ldos_rhol_density_grid_input(
    input: LdosRholDensityGridInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("radii", input.radii.len(), 4)?;
    ensure_len("rhol_energy", input.wave_numbers.len(), 1)?;
    ensure_len("angular_count", input.angular_count, 1)?;
    validate_positive_real_values("radii", input.radii)?;
    validate_positive_real_scalar("radial_step", input.radial_step)?;
    validate_positive_real_scalar("norman_radius", input.norman_radius)?;
    validate_complex_values("wave_numbers", input.wave_numbers.iter().copied())?;

    let energy_count = input.wave_numbers.len();
    let radial_count = input.radii.len();
    ensure_shape3(
        "regular_large",
        input.regular_large.shape(),
        energy_count,
        input.angular_count,
        radial_count,
    )?;
    ensure_shape3(
        "regular_small",
        input.regular_small.shape(),
        energy_count,
        input.angular_count,
        radial_count,
    )?;
    ensure_shape3(
        "irregular_large",
        input.irregular_large.shape(),
        energy_count,
        input.angular_count,
        radial_count,
    )?;
    ensure_shape3(
        "irregular_small",
        input.irregular_small.shape(),
        energy_count,
        input.angular_count,
        radial_count,
    )?;
    validate_complex_values("regular_large", input.regular_large.iter().copied())?;
    validate_complex_values("regular_small", input.regular_small.iter().copied())?;
    validate_complex_values("irregular_large", input.irregular_large.iter().copied())?;
    validate_complex_values("irregular_small", input.irregular_small.iter().copied())?;
    Ok(())
}

fn validate_pot_rholie_density_input(input: PotRholieDensityInput<'_>) -> Result<(), DensityError> {
    ensure_len("pot_rholie_source_radii", input.source_radii.len(), 4)?;
    ensure_len("pot_rholie_output_radii", input.output_radii.len(), 1)?;
    ensure_length_match(
        "pot_rholie_source_radii",
        input.source_radii.len(),
        "pot_rholie_regular_large",
        input.regular_large.len(),
    )?;
    ensure_length_match(
        "pot_rholie_source_radii",
        input.source_radii.len(),
        "pot_rholie_regular_small",
        input.regular_small.len(),
    )?;
    ensure_length_match(
        "pot_rholie_source_radii",
        input.source_radii.len(),
        "pot_rholie_irregular_large",
        input.irregular_large.len(),
    )?;
    ensure_length_match(
        "pot_rholie_source_radii",
        input.source_radii.len(),
        "pot_rholie_irregular_small",
        input.irregular_small.len(),
    )?;
    validate_positive_real_values("pot_rholie_source_radii", input.source_radii)?;
    validate_positive_real_values("pot_rholie_output_radii", input.output_radii)?;
    validate_positive_real_scalar("pot_rholie_radial_step", input.radial_step)?;
    validate_positive_real_scalar("pot_rholie_norman_radius", input.norman_radius)?;
    validate_complex_values(
        "pot_rholie_regular_large",
        input.regular_large.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_regular_small",
        input.regular_small.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_irregular_large",
        input.irregular_large.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_irregular_small",
        input.irregular_small.iter().copied(),
    )?;
    validate_complex_scalar("pot_rholie_wave_number", input.wave_number)?;
    Ok(())
}

fn validate_pot_rholie_density_grid_input(
    input: PotRholieDensityGridInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("pot_rholie_source_radii", input.source_radii.len(), 4)?;
    ensure_len("pot_rholie_output_radii", input.output_radii.len(), 1)?;
    ensure_len("pot_rholie_angular_count", input.angular_count, 1)?;
    validate_positive_real_values("pot_rholie_source_radii", input.source_radii)?;
    validate_positive_real_values("pot_rholie_output_radii", input.output_radii)?;
    validate_positive_real_scalar("pot_rholie_radial_step", input.radial_step)?;
    validate_positive_real_scalar("pot_rholie_norman_radius", input.norman_radius)?;
    validate_complex_scalar("pot_rholie_wave_number", input.wave_number)?;

    let source_len = input.source_radii.len();
    ensure_shape(
        "pot_rholie_regular_large",
        input.regular_large.shape(),
        input.angular_count,
        source_len,
    )?;
    ensure_shape(
        "pot_rholie_regular_small",
        input.regular_small.shape(),
        input.angular_count,
        source_len,
    )?;
    ensure_shape(
        "pot_rholie_irregular_large",
        input.irregular_large.shape(),
        input.angular_count,
        source_len,
    )?;
    ensure_shape(
        "pot_rholie_irregular_small",
        input.irregular_small.shape(),
        input.angular_count,
        source_len,
    )?;
    validate_complex_values(
        "pot_rholie_regular_large",
        input.regular_large.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_regular_small",
        input.regular_small.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_irregular_large",
        input.irregular_large.iter().copied(),
    )?;
    validate_complex_values(
        "pot_rholie_irregular_small",
        input.irregular_small.iter().copied(),
    )?;
    Ok(())
}

fn validate_pot_scf_contour_source_rows_input(
    input: PotScfContourSourceRowsInput<'_>,
) -> Result<(), DensityError> {
    let point_count = input.source_energies.len();
    ensure_len("pot_scf_source_row_energies", point_count, 1)?;
    validate_complex_values(
        "pot_scf_source_row_energies",
        input.source_energies.iter().copied(),
    )?;
    ensure_len("pot_scf_source_row_angular_count", input.angular_count, 1)?;
    ensure_len(
        "pot_scf_source_row_source_radii",
        input.source_radii.len(),
        4,
    )?;
    ensure_len(
        "pot_scf_source_row_output_radii",
        input.output_radii.len(),
        1,
    )?;
    validate_positive_real_values("pot_scf_source_row_source_radii", input.source_radii)?;
    validate_positive_real_values("pot_scf_source_row_output_radii", input.output_radii)?;
    validate_positive_real_scalar("pot_scf_source_row_radial_step", input.radial_step)?;

    let potential_count =
        input
            .highest_potential_index
            .checked_add(1)
            .ok_or(DensityError::InvalidIndex {
                name: "pot_scf_source_row_highest_potential_index",
                index: input.highest_potential_index,
            })?;
    ensure_len(
        "pot_scf_source_row_norman_radii",
        input.norman_radii.len(),
        potential_count,
    )?;
    validate_positive_real_values("pot_scf_source_row_norman_radii", input.norman_radii)?;
    ensure_shape(
        "pot_scf_source_row_wave_numbers",
        input.wave_numbers.shape(),
        point_count,
        potential_count,
    )?;
    validate_complex_values(
        "pot_scf_source_row_wave_numbers",
        input.wave_numbers.iter().copied(),
    )?;
    ensure_shape3(
        "pot_scf_source_row_scattering_trace",
        input.scattering_trace.shape(),
        point_count,
        input.angular_count,
        potential_count,
    )?;
    validate_complex32_values(
        "pot_scf_source_row_scattering_trace",
        input.scattering_trace.iter().copied(),
    )?;

    validate_pot_scf_source_wavefunction_shape(
        "pot_scf_source_row_regular_large",
        input.regular_large.shape(),
        point_count,
        potential_count,
        input.angular_count,
        input.source_radii.len(),
    )?;
    validate_pot_scf_source_wavefunction_shape(
        "pot_scf_source_row_regular_small",
        input.regular_small.shape(),
        point_count,
        potential_count,
        input.angular_count,
        input.source_radii.len(),
    )?;
    validate_pot_scf_source_wavefunction_shape(
        "pot_scf_source_row_irregular_large",
        input.irregular_large.shape(),
        point_count,
        potential_count,
        input.angular_count,
        input.source_radii.len(),
    )?;
    validate_pot_scf_source_wavefunction_shape(
        "pot_scf_source_row_irregular_small",
        input.irregular_small.shape(),
        point_count,
        potential_count,
        input.angular_count,
        input.source_radii.len(),
    )?;
    validate_complex_values(
        "pot_scf_source_row_regular_large",
        input.regular_large.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_source_row_regular_small",
        input.regular_small.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_source_row_irregular_large",
        input.irregular_large.iter().copied(),
    )?;
    validate_complex_values(
        "pot_scf_source_row_irregular_small",
        input.irregular_small.iter().copied(),
    )
}

fn validate_pot_scf_source_wavefunction_shape(
    name: &'static str,
    shape: &[usize],
    required_points: usize,
    required_potentials: usize,
    required_angular: usize,
    required_radial: usize,
) -> Result<(), DensityError> {
    ensure_len(name, shape[0], required_points)?;
    ensure_len(name, shape[1], required_potentials)?;
    ensure_len(name, shape[2], required_angular)?;
    ensure_len(name, shape[3], required_radial)
}

fn validate_complex32_values<I>(name: &'static str, values: I) -> Result<(), DensityError>
where
    I: IntoIterator<Item = Complex32>,
{
    for (index, value) in values.into_iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re as Real,
                imaginary: value.im as Real,
            });
        }
    }
    Ok(())
}

fn validate_ldos_rhol_radial_assembly_input(
    input: LdosRholRadialAssemblyInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("radii", input.radii.len(), 1)?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "raw_regular_large",
        input.raw_regular_large.len(),
    )?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "raw_regular_small",
        input.raw_regular_small.len(),
    )?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "raw_irregular_large",
        input.raw_irregular_large.len(),
    )?;
    ensure_length_match(
        "radii",
        input.radii.len(),
        "raw_irregular_small",
        input.raw_irregular_small.len(),
    )?;
    if input.match_index_1based == 0 || input.match_index_1based > input.radii.len() {
        return Err(DensityError::InvalidIndex {
            name: "ldos_rhol_match_index_1based",
            index: input.match_index_1based,
        });
    }
    validate_positive_real_values("radii", input.radii)?;
    validate_complex_values("raw_regular_large", input.raw_regular_large.iter().copied())?;
    validate_complex_values("raw_regular_small", input.raw_regular_small.iter().copied())?;
    validate_complex_values(
        "raw_irregular_large",
        input.raw_irregular_large.iter().copied(),
    )?;
    validate_complex_values(
        "raw_irregular_small",
        input.raw_irregular_small.iter().copied(),
    )?;
    validate_complex_scalar("phase_shift", input.phase_shift)?;
    validate_complex_scalar("phase_amplitude", input.phase_amplitude)?;
    validate_complex_scalar("wave_number", input.wave_number)?;
    Ok(())
}

fn ldos_relativistic_small_component_factor(wave_number: Complex) -> Result<Complex, DensityError> {
    let alpha_wave = wave_number * LDOS_FINE_STRUCTURE_ALPHA;
    let factor = -alpha_wave
        / (Complex::new(1.0, 0.0) + (Complex::new(1.0, 0.0) + alpha_wave * alpha_wave).sqrt());
    validate_complex_scalar("ldos_rhol_small_component_factor", factor)?;
    Ok(factor)
}

fn ldos_rhol_csomm2_integration_len(
    radii: &[Real],
    norman_radius: Real,
) -> Result<usize, DensityError> {
    let norman_below = radii
        .iter()
        .rposition(|&radius| radius <= norman_radius)
        .ok_or(DensityError::LengthTooShort {
            name: "ldos_rhol_density_rnrm_prefix",
            required: 1,
            actual: 0,
        })?;
    let integration_len = norman_below
        .checked_add(3)
        .ok_or(DensityError::InvalidIndex {
            name: "ldos_rhol_density_rnrm_prefix",
            index: norman_below,
        })?;
    if integration_len > radii.len() {
        return Err(DensityError::LengthTooShort {
            name: "ldos_rhol_density_rnrm_prefix",
            required: integration_len,
            actual: radii.len(),
        });
    }
    ensure_len("ldos_rhol_density_rnrm_prefix", integration_len, 4)?;
    Ok(integration_len)
}

fn validate_complex32_scalar(name: &'static str, value: Complex32) -> Result<(), DensityError> {
    validate_complex_scalar(name, widen_complex32(value))
}

fn validate_ldos_spin_ff2rho_input(input: LdosSpinFf2rhoInput<'_>) -> Result<(), DensityError> {
    ensure_len("energy_grid_hartree", input.energy_grid_hartree.len(), 1)?;
    ensure_len("angular_count", input.angular_count, 1)?;
    let energy_count = input.energy_grid_hartree.len();

    ensure_shape3(
        "embedded_ldos",
        input.embedded_ldos.shape(),
        input.angular_count,
        2,
        energy_count,
    )?;
    validate_complex_values(
        "energy_grid_hartree",
        input.energy_grid_hartree.iter().copied(),
    )?;

    if input.apply_scattering {
        ensure_shape3(
            "scattering_ldos",
            input.scattering_ldos.shape(),
            input.angular_count,
            2,
            energy_count,
        )?;
        ensure_shape3(
            "scattering_trace",
            input.scattering_trace.shape(),
            input.angular_count,
            2,
            energy_count,
        )?;
    }

    Ok(())
}

fn ldos_hubbard_occupation(
    energies_hartree: &[Real],
    density: &[Real],
    occupation_limit_hartree: Real,
) -> Result<Real, DensityError> {
    let step =
        (occupation_limit_hartree - energies_hartree[0]) / LDOS_HUBBARD_OCCUPATION_POINTS as Real;
    let integration_energy = (0..LDOS_HUBBARD_OCCUPATION_POINTS)
        .map(|index| energies_hartree[0] + index as Real * step)
        .collect::<Vec<_>>();
    let mut interpolated_density = Vec::with_capacity(LDOS_HUBBARD_OCCUPATION_POINTS);
    for &energy in &integration_energy {
        interpolated_density.push(terp(energies_hartree, density, 3, energy)?.value);
    }
    Ok(trap(&integration_energy, &interpolated_density)? * FEFF_HARTREE_EV / 2.0)
}

fn validate_ldos_hubbard_step1_input(input: LdosHubbardStep1Input<'_>) -> Result<(), DensityError> {
    ensure_len(
        "hubbard_step1_energy_grid",
        input.energy_grid_hartree.len(),
        4,
    )?;
    ensure_len("hubbard_step1_angular_count", input.angular_count, 1)?;
    if input.hubbard_l >= input.angular_count {
        return Err(DensityError::InvalidIndex {
            name: "hubbard_step1_l_hubbard",
            index: input.hubbard_l,
        });
    }
    let energy_count = input.energy_grid_hartree.len();
    let magnetic_count =
        input
            .angular_count
            .checked_mul(input.angular_count)
            .ok_or(DensityError::InvalidIndex {
                name: "hubbard_step1_angular_count",
                index: input.angular_count,
            })?;
    ensure_shape3(
        "hubbard_step1_embedded_ldos",
        input.embedded_ldos.shape(),
        input.angular_count,
        2,
        energy_count,
    )?;
    ensure_shape3(
        "hubbard_step1_scattering_ldos",
        input.scattering_ldos.shape(),
        input.angular_count,
        2,
        energy_count,
    )?;
    ensure_len(
        "hubbard_step1_magnetic_scattering_trace angular",
        input.magnetic_scattering_trace.shape()[0],
        input.angular_count,
    )?;
    ensure_len(
        "hubbard_step1_magnetic_scattering_trace magnetic",
        input.magnetic_scattering_trace.shape()[1],
        magnetic_count,
    )?;
    ensure_len(
        "hubbard_step1_magnetic_scattering_trace spin",
        input.magnetic_scattering_trace.shape()[2],
        2,
    )?;
    ensure_len(
        "hubbard_step1_magnetic_scattering_trace energy",
        input.magnetic_scattering_trace.shape()[3],
        energy_count,
    )?;
    ensure_len(
        "hubbard_step1_off_diagonal_scattering_trace angular",
        input.off_diagonal_scattering_trace.shape()[0],
        input.angular_count,
    )?;
    let off_diagonal_count = (input.hubbard_l + 1) * (input.hubbard_l + 1);
    ensure_len(
        "hubbard_step1_off_diagonal_scattering_trace rows",
        input.off_diagonal_scattering_trace.shape()[1],
        off_diagonal_count,
    )?;
    ensure_len(
        "hubbard_step1_off_diagonal_scattering_trace columns",
        input.off_diagonal_scattering_trace.shape()[2],
        off_diagonal_count,
    )?;
    ensure_len(
        "hubbard_step1_off_diagonal_scattering_trace spin",
        input.off_diagonal_scattering_trace.shape()[3],
        2,
    )?;
    ensure_len(
        "hubbard_step1_off_diagonal_scattering_trace energy",
        input.off_diagonal_scattering_trace.shape()[4],
        energy_count,
    )?;

    validate_complex_values(
        "hubbard_step1_energy_grid",
        input.energy_grid_hartree.iter().copied(),
    )?;
    for &value in input.embedded_ldos {
        validate_real_scalar("hubbard_step1_embedded_ldos", value)?;
    }
    validate_complex_values(
        "hubbard_step1_scattering_ldos",
        input.scattering_ldos.iter().copied(),
    )?;
    validate_complex_values(
        "hubbard_step1_magnetic_scattering_trace",
        input.magnetic_scattering_trace.iter().copied(),
    )?;
    validate_complex_values(
        "hubbard_step1_off_diagonal_scattering_trace",
        input.off_diagonal_scattering_trace.iter().copied(),
    )?;
    for (name, value) in [
        (
            "hubbard_step1_chemical_potential",
            input.chemical_potential_hartree,
        ),
        ("hubbard_step1_fermi_shift", input.fermi_shift_ev),
        ("hubbard_step1_u", input.hubbard_u_ev),
        ("hubbard_step1_j", input.hubbard_j_ev),
    ] {
        validate_real_scalar(name, value)?;
    }
    Ok(())
}

fn validate_ldos_hubbard_magnetic_ff2rho_input(
    input: LdosHubbardMagneticFf2rhoInput<'_>,
) -> Result<(), DensityError> {
    ensure_len("energy_grid_hartree", input.energy_grid_hartree.len(), 1)?;
    ensure_len("angular_count", input.angular_count, 1)?;
    let energy_count = input.energy_grid_hartree.len();
    let magnetic_count =
        input
            .angular_count
            .checked_mul(input.angular_count)
            .ok_or(DensityError::InvalidIndex {
                name: "angular_count",
                index: input.angular_count,
            })?;

    ensure_len(
        "embedded_magnetic_ldos angular",
        input.embedded_magnetic_ldos.shape()[0],
        input.angular_count,
    )?;
    ensure_len(
        "embedded_magnetic_ldos magnetic",
        input.embedded_magnetic_ldos.shape()[1],
        magnetic_count,
    )?;
    ensure_len(
        "embedded_magnetic_ldos spin",
        input.embedded_magnetic_ldos.shape()[2],
        2,
    )?;
    ensure_len(
        "embedded_magnetic_ldos energy",
        input.embedded_magnetic_ldos.shape()[3],
        energy_count,
    )?;
    ensure_len(
        "scattering_magnetic_ldos angular",
        input.scattering_magnetic_ldos.shape()[0],
        input.angular_count,
    )?;
    ensure_len(
        "scattering_magnetic_ldos magnetic",
        input.scattering_magnetic_ldos.shape()[1],
        magnetic_count,
    )?;
    ensure_len(
        "scattering_magnetic_ldos spin",
        input.scattering_magnetic_ldos.shape()[2],
        2,
    )?;
    ensure_len(
        "scattering_magnetic_ldos energy",
        input.scattering_magnetic_ldos.shape()[3],
        energy_count,
    )?;
    ensure_len(
        "magnetic_scattering_trace angular",
        input.magnetic_scattering_trace.shape()[0],
        input.angular_count,
    )?;
    ensure_len(
        "magnetic_scattering_trace magnetic",
        input.magnetic_scattering_trace.shape()[1],
        magnetic_count,
    )?;
    ensure_len(
        "magnetic_scattering_trace spin",
        input.magnetic_scattering_trace.shape()[2],
        2,
    )?;
    ensure_len(
        "magnetic_scattering_trace energy",
        input.magnetic_scattering_trace.shape()[3],
        energy_count,
    )?;
    validate_complex_values(
        "energy_grid_hartree",
        input.energy_grid_hartree.iter().copied(),
    )?;

    Ok(())
}
