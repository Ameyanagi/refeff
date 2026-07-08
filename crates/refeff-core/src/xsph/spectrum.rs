//! FEFF XSPH NRIXS transition weights and spectrum updates.

use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, Axis, ShapeBuilder,
};
use num_complex::Complex32;
use refeff_linalg::{complex32_lu_factor, complex32_lu_solve_vector};

use crate::{
    Complex, FovrgYkZkExchangeInput, Real, conv, fovrg_yk_zk_exchange, legendre_polynomials_into,
    somm2, terp, wigner_3j,
};

use super::{
    XsphError, XsphLgSpectrumUpdateInput, XsphLjSpectrumUpdateInput, XsphSpectrumUpdateMode,
    XsphTdldaAngularKernel, XsphTdldaAngularKernelInput, XsphTdldaBroadenedChannelSpectra,
    XsphTdldaChannelBroadeningInput, XsphTdldaChannelMultipliers, XsphTdldaChannelMultipliersInput,
    XsphTdldaChannelSpectra, XsphTdldaChannelSpectraInput, XsphTdldaConditionedResponse,
    XsphTdldaCoulombFields, XsphTdldaCoulombFieldsInput, XsphTdldaDirectKernel,
    XsphTdldaDirectKernelInput, XsphTdldaEnergyRows, XsphTdldaEnergyRowsInput,
    XsphTdldaKramersKronigInput, XsphTdldaKramersKronigResponse, XsphTdldaNonlocalExchangeInput,
    XsphTdldaProjectedKernel, XsphTdldaProjectedKernelInput, XsphTdldaProjectorOrthogonalization,
    XsphTdldaProjectorOrthogonalizationInput, XsphTdldaRadialKernel, XsphTdldaRadialKernelInput,
    XsphTdldaRawResponse, XsphTdldaRawResponseInput, XsphTdldaResponseConditioningInput,
    XsphTdldaRowWaveNumbers, XsphTdldaRowWaveNumbersInput, XsphTdldaScreenedDipole,
    XsphTdldaScreenedDipoleInput, XsphTdldaWeightedResponse, XsphTdldaWeightedResponseInput,
    XsphTdldaXmuChannelInput, XsphTdldaXsedgeRows, XsphTdldaXsedgeRowsInput, XsphXsectSpinMerge,
    XsphXsectSpinMergeInput, doubled_j_from_kappa, usize_to_i32, validate_active_len,
    validate_cwig3j_doubled_argument, validate_cwig3j_integer_argument, validate_finite_complex,
    validate_finite_real, validate_indexed_angular_momentum,
};

const TDLDA_KK_FAKE_GRID_COUNT: usize = 2000;
const TDLDA_KK_RIGHT_PADDING_EV: Real = 2.0;
const TDLDA_RIDXMU_RANGE_TOLERANCE: Real = 1.0e-12;
const TDLDA_XSECTD_OMEGA_FLOOR_EV: Real = 0.1;
const TDLDA_XSECTD_MIN_ACTIVE_ENERGY: Real = -10.0;

/// Port of FEFF `XSPH/bcoefjas.f90`.
///
/// Builds the two spin-component NRIXS transition weights `hbmat(0:1, 1:indmax)`
/// for a single doubled initial magnetic quantum number. The returned array is
/// Fortran-order with shape `(2, active_len)`, matching FEFF's spin-first
/// storage.
#[allow(clippy::too_many_arguments)]
pub fn xsph_nrixs_transition_weights(
    initial_kappa: i32,
    initial_mj2: i32,
    lmax: usize,
    jmax: i32,
    ljmax: i32,
    lgind: ArrayView1<'_, i32>,
    ljind: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<Array2<Real>, XsphError> {
    if initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len("lgind", lgind.len(), active_len)?;
    validate_active_len("ljind", ljind.len(), active_len)?;
    if jmax < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "jmax",
            index: 0,
            value: jmax,
        });
    }
    validate_cwig3j_doubled_argument("jmax", jmax, jmax)?;

    let lmax_i32 = usize_to_i32("lmax", lmax)?;
    let doubled_lmax = lmax_i32.checked_mul(2).ok_or(XsphError::SizeOutOfRange {
        name: "lmax",
        value: lmax,
    })?;
    let abs_ljmax = ljmax.checked_abs().ok_or(XsphError::IntegerOutOfRange {
        name: "ljmax",
        value: ljmax,
    })?;
    validate_cwig3j_integer_argument("ljmax", abs_ljmax)?;
    let jinit = doubled_j_from_kappa("initial_kappa", initial_kappa)?;
    validate_cwig3j_doubled_argument("initial_kappa", initial_kappa, jinit)?;
    let abs_initial_mj2 = initial_mj2
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_mj2",
            value: initial_mj2,
        })?;
    let initial_parity = if initial_kappa > 0 { -1 } else { 1 };

    let mut final_j2 = Vec::new();
    for lj in 0..=abs_ljmax {
        let lower = (2 * lj - jinit).abs().max(1);
        let upper = (2 * lj + jinit).min(jmax);
        let mut jfin = lower;
        while jfin <= upper {
            let final_parity = if (jinit + jfin + 2 * lj).rem_euclid(4) == 0 {
                -initial_parity
            } else {
                initial_parity
            };
            let final_l2 = if final_parity > 0 { jfin - 1 } else { jfin + 1 };
            if final_l2 <= doubled_lmax {
                final_j2.push(jfin);
            }
            jfin += 2;
        }
    }
    if final_j2.len() < active_len {
        return Err(XsphError::InsufficientGeneratedStates {
            required: active_len,
            generated: final_j2.len(),
        });
    }

    let mut weights = Array2::<Real>::zeros((2, active_len).f());
    for index in 0..active_len {
        let jfin = final_j2[index];
        let lj = validate_indexed_angular_momentum("ljind", index, ljind[index])?;
        let lg = validate_indexed_angular_momentum("lgind", index, lgind[index])?
            .checked_mul(2)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "lgind",
                value: lgind[index],
            })?;
        validate_cwig3j_doubled_argument("jfin", jfin, jfin)?;
        validate_cwig3j_integer_argument("ljind", lj)?;
        validate_cwig3j_doubled_argument("lgind", lgind[index], lg)?;

        let mut simple_3j = if abs_initial_mj2 <= jfin {
            wigner_3j(jinit, 2 * lj, jfin, -initial_mj2, 0, 2)?
        } else {
            0.0
        };
        if (i64::from(initial_mj2) + 1).rem_euclid(4) != 0 {
            simple_3j = -simple_3j;
        }

        for spin_index in 0..=1 {
            let mut ls_to_j = 0.0;
            if abs_initial_mj2 <= jfin && abs_initial_mj2 - 1 <= doubled_lmax {
                let spin_mj2 = 2 * usize_to_i32("spin_index", spin_index)? - 1;
                let magnetic_l2 =
                    initial_mj2
                        .checked_sub(spin_mj2)
                        .ok_or(XsphError::IntegerOutOfRange {
                            name: "initial_mj2",
                            value: initial_mj2,
                        })?;
                ls_to_j = wigner_3j(lg, 1, jfin, magnetic_l2, spin_mj2, 2)?;
                if (i64::from(lg) - 1 + i64::from(initial_mj2)).rem_euclid(4) != 0 {
                    ls_to_j = -ls_to_j;
                }
                ls_to_j *= (f64::from(jfin) + 1.0).sqrt();
            }
            weights[(spin_index, index)] = ls_to_j * simple_3j;
        }
    }

    Ok(weights)
}

/// Port of FEFF `XSPH/xsphsub.f90` final `xsect.dat` spin merge.
///
/// For non-XMCD output FEFF writes the first spin channel directly. For
/// `abs(ispin) == 1` two-spin output it writes the average `xsnorm`, the summed
/// `xsec`, and, for ordinary `nq == 1` output, rescales the first and last spin
/// `rkk` rows by `sqrt(2*xsnorm_spin/(xsnorm_down+xsnorm_up))`.
pub fn xsph_xsect_spin_merge(
    input: XsphXsectSpinMergeInput<'_>,
) -> Result<XsphXsectSpinMerge, XsphError> {
    let spin_count = validate_xsect_spin_merge_input(&input)?;
    let last_spin_index = spin_count - 1;

    let mut reduced_matrix_elements =
        Array3::<Complex>::zeros((input.q_count, input.transition_count, spin_count).f());
    for iq in 0..input.q_count {
        for transition in 0..input.transition_count {
            for spin in 0..spin_count {
                let value = input.reduced_matrix_elements[(iq, transition, spin)];
                validate_finite_complex("xsect_spin_merge_rkk", transition, value)?;
                reduced_matrix_elements[(iq, transition, spin)] = value;
            }
        }
    }

    if !input.spin_polarized {
        return Ok(XsphXsectSpinMerge {
            spectrum_norm: input.spectrum_norms[0],
            cross_section: input.cross_sections[0],
            spin_scales: None,
            reduced_matrix_elements,
        });
    }

    let spectrum_norm_sum = input.spectrum_norms[0] + input.spectrum_norms[last_spin_index];
    validate_finite_real("xsect_spin_merge_norm_sum", spectrum_norm_sum)?;
    let spectrum_norm = spectrum_norm_sum / 2.0;
    validate_finite_real("xsect_spin_merge_norm", spectrum_norm)?;
    let cross_section = input.cross_sections[0] + input.cross_sections[last_spin_index];
    validate_finite_complex("xsect_spin_merge_cross_section", 0, cross_section)?;

    let spin_scales = if input.q_count == 1 {
        if spectrum_norm_sum <= 0.0 {
            return Err(XsphError::InvalidPositiveScalar {
                name: "xsect_spin_merge_norm_sum",
                value: spectrum_norm_sum,
            });
        }
        let first_scale = (2.0 * input.spectrum_norms[0] / spectrum_norm_sum).sqrt();
        let last_scale = (2.0 * input.spectrum_norms[last_spin_index] / spectrum_norm_sum).sqrt();
        validate_finite_real("xsect_spin_merge_first_scale", first_scale)?;
        validate_finite_real("xsect_spin_merge_last_scale", last_scale)?;
        for transition in 0..input.transition_count {
            reduced_matrix_elements[(0, transition, 0)] *= first_scale;
            reduced_matrix_elements[(0, transition, last_spin_index)] *= last_scale;
        }
        Some([first_scale, last_scale])
    } else {
        None
    };

    Ok(XsphXsectSpinMerge {
        spectrum_norm,
        cross_section,
        spin_scales,
        reduced_matrix_elements,
    })
}

/// Port of FEFF `TDLDA/dmscf.f90` screened dipole solve.
///
/// FEFF forms `Mscf = 1 - K*chi0` for each energy row, casts that matrix and
/// the real dipole vector to legacy single-complex LAPACK inputs, then solves
/// `Mscf * dipscf = dipmat`. Keeping the single-complex solve here preserves
/// the numerical behavior of the original `cgetrf`/`cgetrs` call site.
pub fn xsph_tdlda_screened_dipoles(
    input: XsphTdldaScreenedDipoleInput<'_>,
) -> Result<XsphTdldaScreenedDipole, XsphError> {
    validate_tdlda_screened_dipole_input(&input)?;

    let mut screened_dipoles = Array2::<Complex>::zeros((input.energy_count, input.matrix_size));
    for energy in 0..input.energy_count {
        let mut system = Array2::<Complex32>::zeros((input.matrix_size, input.matrix_size));
        for row in 0..input.matrix_size {
            for column in 0..input.matrix_size {
                let mut value = if row == column {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                for contracted in 0..input.matrix_size {
                    value -= input.kernel[(energy, row, contracted)]
                        * input.response[(energy, contracted, column)];
                }
                validate_finite_complex(
                    "tdlda_dmscf_system",
                    row * input.matrix_size + column,
                    value,
                )?;
                system[(row, column)] = complex_to_complex32(value);
            }
        }

        let rhs = Array1::from_shape_fn(input.matrix_size, |row| {
            Complex32::new(input.dipole_matrix[(energy, row)] as f32, 0.0)
        });
        let lu = complex32_lu_factor(system.view())?;
        let solved = complex32_lu_solve_vector(&lu, rhs.view())?;
        for row in 0..input.matrix_size {
            screened_dipoles[(energy, row)] = complex32_to_complex(solved[row]);
        }
    }

    Ok(XsphTdldaScreenedDipole { screened_dipoles })
}

/// Port of FEFF `TDLDA/xsectd.f90` per-energy setup before `getchi0`.
///
/// For each `emr(ie)`, FEFF evaluates the `l3` and split `l2` relativistic
/// wave numbers from the energy-dependent `xcpot` reference, floors the photon
/// energy at `0.1 eV`, computes the PMBSE/TDLDA separation function, and skips
/// only rows below `-10 Hartree` before entering `getchi0`.
pub fn xsph_tdlda_energy_rows(
    input: XsphTdldaEnergyRowsInput<'_>,
) -> Result<XsphTdldaEnergyRows, XsphError> {
    validate_tdlda_energy_rows_input(&input)?;

    let separation_function =
        xsph_tdlda_separation_function(input.ipmbse, input.energy_hartree, input.energy_count)?;
    let omega_floor = TDLDA_XSECTD_OMEGA_FLOOR_EV / super::XSPH_HARTREE_EV;
    let mut photon_energy = Array1::<Real>::zeros(input.energy_count);
    let mut plus_wave_number = Array1::<Real>::zeros(input.energy_count);
    let mut minus_wave_number = Array1::<Real>::zeros(input.energy_count);
    let mut active_rows = Array1::<bool>::from_elem(input.energy_count, false);

    for energy in 0..input.energy_count {
        let row_energy = input.energy_hartree[energy];
        let reference = input.reference_energy[energy];
        let plus_momentum_squared = Complex::new(row_energy, 0.0) - reference;
        let minus_momentum_squared =
            Complex::new(row_energy - input.spin_orbit_split, 0.0) - reference;
        let plus_wave = tdlda_relativistic_wave_number(plus_momentum_squared)?;
        let minus_wave = tdlda_relativistic_wave_number(minus_momentum_squared)?;
        plus_wave_number[energy] = plus_wave.re;
        minus_wave_number[energy] = minus_wave.re;
        validate_finite_real("tdlda_energy_plus_wave_number", plus_wave_number[energy])?;
        validate_finite_real("tdlda_energy_minus_wave_number", minus_wave_number[energy])?;

        let omega = (row_energy - input.edge_energy + input.chemical_potential).max(omega_floor);
        validate_finite_real("tdlda_energy_omega", omega)?;
        photon_energy[energy] = omega;
        active_rows[energy] = row_energy >= TDLDA_XSECTD_MIN_ACTIVE_ENERGY;
    }

    Ok(XsphTdldaEnergyRows {
        photon_energy,
        plus_wave_number,
        minus_wave_number,
        separation_function,
        active_rows,
    })
}

/// Port of FEFF `TDLDA/getchi0.f90` per-channel wave-number setup.
///
/// For each matrix row, FEFF shifts the current energy by `refsh(im)` before
/// evaluating `ck`. Unlike `xsectd`'s `ckl3/ckl2` setup, this path uses
/// `dble(eref)` and therefore ignores any imaginary self-energy component.
pub fn xsph_tdlda_row_wave_numbers(
    input: XsphTdldaRowWaveNumbersInput<'_>,
) -> Result<XsphTdldaRowWaveNumbers, XsphError> {
    validate_tdlda_row_wave_numbers_input(&input)?;

    let mut momentum_squared = Array1::<Real>::zeros(input.matrix_size);
    let mut row_wave_numbers = Array1::<Real>::zeros(input.matrix_size);
    let mut positive_momentum_rows = Array1::<bool>::from_elem(input.matrix_size, false);
    for row in 0..input.matrix_size {
        let row_momentum_squared =
            input.energy_hartree - input.reference_energy.re + input.reference_shifts[row];
        validate_finite_real(
            "tdlda_row_wave_number_momentum_squared",
            row_momentum_squared,
        )?;
        let wave_number =
            tdlda_relativistic_wave_number(Complex::new(row_momentum_squared, 0.0))?.re;
        validate_finite_real("tdlda_row_wave_number", wave_number)?;
        momentum_squared[row] = row_momentum_squared;
        row_wave_numbers[row] = wave_number;
        positive_momentum_rows[row] = row_momentum_squared > 0.0;
    }

    Ok(XsphTdldaRowWaveNumbers {
        momentum_squared,
        row_wave_numbers,
        positive_momentum_rows,
    })
}

/// Port of FEFF `TDLDA/getchi0.f90` raw overlap response assembly.
///
/// This is the non-radial finalization performed after `getchi0` computes the
/// row overlap integral `ovrl(im)` and dipoles. FEFF fills the diagonal and
/// same-projector-family off-diagonals of the local `chi0im` matrix, then zeros
/// both dipole arrays for rows below `edge - refsh(im)`.
pub fn xsph_tdlda_raw_response(
    input: XsphTdldaRawResponseInput<'_>,
) -> Result<XsphTdldaRawResponse, XsphError> {
    let (plus_stride, minus_stride, plus_block_size) = validate_tdlda_raw_response_input(&input)?;

    let mut raw_imaginary_response = Array2::<Real>::zeros((input.matrix_size, input.matrix_size));
    let mut localized_dipoles = input.localized_dipoles.to_owned();
    let mut full_dipoles = input.full_dipoles.to_owned();
    let mut occupied_rows = Array1::<bool>::from_elem(input.matrix_size, false);

    for row in 0..input.matrix_size {
        let occupied = input.energy_hartree >= input.edge_energy - input.reference_shifts[row];
        occupied_rows[row] = occupied;
        if !occupied {
            localized_dipoles[row] = 0.0;
            full_dipoles[row] = 0.0;
            continue;
        }

        let row_scale = -2.0 * input.row_wave_numbers[row] * input.overlaps[row];
        let diagonal = row_scale * input.overlaps[row];
        validate_finite_real("tdlda_raw_response", diagonal)?;
        raw_imaginary_response[(row, row)] = diagonal;

        if input.plus_basis_count > 1 {
            for basis_delta in 1..input.plus_basis_count {
                let step =
                    basis_delta
                        .checked_mul(plus_stride)
                        .ok_or(XsphError::SizeOutOfRange {
                            name: "tdlda_raw_response_plus_stride",
                            value: basis_delta,
                        })?;
                if row + 1 > step {
                    let column = row - step;
                    let value = row_scale * input.overlaps[column];
                    validate_finite_real("tdlda_raw_response", value)?;
                    raw_imaginary_response[(row, column)] = value;
                    raw_imaginary_response[(column, row)] = value;
                }
            }
        }

        if input.minus_basis_count > 1 {
            for basis_delta in 1..input.minus_basis_count {
                let step =
                    basis_delta
                        .checked_mul(minus_stride)
                        .ok_or(XsphError::SizeOutOfRange {
                            name: "tdlda_raw_response_minus_stride",
                            value: basis_delta,
                        })?;
                let minimum_row =
                    plus_block_size
                        .checked_add(step)
                        .ok_or(XsphError::SizeOutOfRange {
                            name: "tdlda_raw_response_minus_stride",
                            value: basis_delta,
                        })?;
                if row + 1 > minimum_row {
                    let column = row - step;
                    let value = row_scale * input.overlaps[column];
                    validate_finite_real("tdlda_raw_response", value)?;
                    raw_imaginary_response[(row, column)] = value;
                    raw_imaginary_response[(column, row)] = value;
                }
            }
        }
    }

    Ok(XsphTdldaRawResponse {
        raw_imaginary_response,
        localized_dipoles,
        full_dipoles,
        occupied_rows,
    })
}

/// Port of FEFF `TDLDA/getchi0.f90` projected-kernel row folding.
///
/// After the radial `xkmatp` contributions are accumulated, FEFF keeps only the
/// representative first `lin + 1` block, folds the first `lin - 1` block into
/// the row range immediately after it, and zeros all extra projector rows.
pub fn xsph_tdlda_projected_kernel(
    input: XsphTdldaProjectedKernelInput<'_>,
) -> Result<XsphTdldaProjectedKernel, XsphError> {
    let (plus_stride, minus_stride, plus_block_size) =
        validate_tdlda_projected_kernel_input(&input)?;
    let mut projected_kernel = input.projected_kernel.to_owned();

    for row in 0..input.matrix_size {
        if input.plus_basis_count > 0 {
            if row >= plus_stride && row < plus_block_size {
                for column in 0..input.matrix_size {
                    projected_kernel[(row, column)] = Complex::new(0.0, 0.0);
                }
            }

            let row_after_plus = row + 1;
            if row_after_plus > plus_block_size {
                let folded_row_1based = row_after_plus - plus_block_size;
                if folded_row_1based <= minus_stride {
                    let target = plus_stride + folded_row_1based - 1;
                    for column in 0..input.matrix_size {
                        projected_kernel[(target, column)] = projected_kernel[(row, column)];
                    }
                }
                for column in 0..input.matrix_size {
                    projected_kernel[(row, column)] = Complex::new(0.0, 0.0);
                }
            }
        } else if row + 1 > minus_stride {
            for column in 0..input.matrix_size {
                projected_kernel[(row, column)] = Complex::new(0.0, 0.0);
            }
        }
    }

    Ok(XsphTdldaProjectedKernel { projected_kernel })
}

/// Port of FEFF `TDLDA/getchi0.f90` direct core-hole potential kernel terms.
///
/// This covers the `vcx = vch * (1 - sfun)` contribution that fills the
/// diagonal `xkmat`, representative projected `xkmatp`, and same-projector
/// off-diagonal `xkmat` entries before the Coulomb/xc angular kernels are
/// accumulated.
pub fn xsph_tdlda_direct_kernel(
    input: XsphTdldaDirectKernelInput<'_>,
) -> Result<XsphTdldaDirectKernel, XsphError> {
    let (plus_stride, minus_stride, plus_block_size) = validate_tdlda_direct_kernel_input(&input)?;
    let mut kernel = Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));
    let mut projected_kernel = Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));
    let direct_scale = 1.0 - input.separation_function;
    validate_finite_real("tdlda_direct_kernel_scale", direct_scale)?;

    for row in 0..input.matrix_size {
        if input.momentum_squared[row] <= 0.0 {
            continue;
        }

        let diagonal = tdlda_direct_kernel_integral(
            input.radii,
            input.core_hole_potential,
            direct_scale,
            input.active_len,
            |radial| {
                input.localized_large[(radial, row)] * input.localized_large[(radial, row)]
                    + input.localized_small[(radial, row)] * input.localized_small[(radial, row)]
            },
        )?;
        kernel[(row, row)] = Complex::new(diagonal, 0.0);

        let projected = tdlda_direct_kernel_integral(
            input.radii,
            input.core_hole_potential,
            direct_scale,
            input.active_len,
            |radial| {
                input.localized_large[(radial, row)] * input.full_large[(radial, row)]
                    + input.localized_small[(radial, row)] * input.full_small[(radial, row)]
            },
        )?;
        let projected_row = tdlda_projected_kernel_representative_row(
            row,
            plus_stride,
            minus_stride,
            plus_block_size,
        );
        projected_kernel[(projected_row, row)] = Complex::new(projected, 0.0);

        if input.plus_basis_count > 1 {
            for basis_delta in 1..input.plus_basis_count {
                let step =
                    basis_delta
                        .checked_mul(plus_stride)
                        .ok_or(XsphError::SizeOutOfRange {
                            name: "tdlda_direct_kernel_plus_stride",
                            value: basis_delta,
                        })?;
                if row + 1 > step {
                    let column = row - step;
                    let value = tdlda_direct_kernel_integral(
                        input.radii,
                        input.core_hole_potential,
                        direct_scale,
                        input.active_len,
                        |radial| {
                            input.localized_large[(radial, row)]
                                * input.localized_large[(radial, column)]
                                + input.localized_small[(radial, row)]
                                    * input.localized_small[(radial, column)]
                        },
                    )?;
                    kernel[(row, column)] = Complex::new(value, 0.0);
                    kernel[(column, row)] = Complex::new(value, 0.0);
                }
            }
        }

        if input.minus_basis_count > 1 {
            for basis_delta in 1..input.minus_basis_count {
                let step =
                    basis_delta
                        .checked_mul(minus_stride)
                        .ok_or(XsphError::SizeOutOfRange {
                            name: "tdlda_direct_kernel_minus_stride",
                            value: basis_delta,
                        })?;
                if row < step {
                    continue;
                }
                let column_1based = row + 1 - step;
                let occupied =
                    input.energy_hartree >= input.edge_energy - input.reference_shifts[row];
                if column_1based > plus_block_size && occupied {
                    let column = column_1based - 1;
                    let value = tdlda_direct_kernel_integral(
                        input.radii,
                        input.core_hole_potential,
                        direct_scale,
                        input.active_len,
                        |radial| {
                            input.localized_large[(radial, row)]
                                * input.localized_large[(radial, column)]
                                + input.localized_small[(radial, row)]
                                    * input.localized_small[(radial, column)]
                        },
                    )?;
                    kernel[(row, column)] = Complex::new(value, 0.0);
                    kernel[(column, row)] = Complex::new(value, 0.0);
                }
            }
        }
    }

    Ok(XsphTdldaDirectKernel {
        kernel,
        projected_kernel,
    })
}

/// Port of FEFF `TDLDA/yzktd.f90` for the `j = 0` TDLDA source-field branch.
///
/// FEFF forms a source `f = cg_core * pf + cp_core * qf`, builds the origin
/// development coefficients from the corresponding polynomial products, and
/// delegates the Coulomb transform to `FOVRG/yzktec.f90`. The returned `fields`
/// matrix is the `ykgr` input consumed by [`xsph_tdlda_radial_kernel_integrals`].
pub fn xsph_tdlda_coulomb_fields(
    input: XsphTdldaCoulombFieldsInput<'_>,
) -> Result<XsphTdldaCoulombFields, XsphError> {
    validate_tdlda_coulomb_fields_input(&input)?;

    let mut fields = Array2::<Complex>::zeros((input.active_len, input.matrix_size));
    let mut computed_lengths = Array1::<usize>::zeros(input.matrix_size);
    let mut origin_constants = Array1::<Complex>::zeros(input.matrix_size);

    for row in 0..input.matrix_size {
        let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
            large_component: input.core_large.index_axis(Axis(1), row),
            small_component: input.core_small.index_axis(Axis(1), row),
            large_coefficients: input.core_large_coefficients.index_axis(Axis(1), row),
            small_coefficients: input.core_small_coefficients.index_axis(Axis(1), row),
            partner_large_component: input.target_large.index_axis(Axis(1), row),
            partner_small_component: input.target_small.index_axis(Axis(1), row),
            partner_large_coefficients: input.target_large_coefficients.index_axis(Axis(1), row),
            partner_small_coefficients: input.target_small_coefficients.index_axis(Axis(1), row),
            radii: input.radii,
            orbital_power: input.core_powers[row],
            partner_power: input.target_powers[row],
            step: input.step,
            angular_momentum: input.multipole,
            coefficient_count: input.coefficient_count,
            orbital_len: input.core_lengths[row],
            source_len: input.source_len,
            active_len: input.active_len,
        })?;

        computed_lengths[row] = transform.computed_len;
        origin_constants[row] = transform.origin_constant;
        for radial in 0..input.active_len {
            fields[(radial, row)] = transform.yk[radial];
        }
    }

    Ok(XsphTdldaCoulombFields {
        fields,
        computed_lengths,
        origin_constants,
    })
}

/// Port of FEFF `TDLDA/getchi0.f90` PMBSE nonlocal exchange radial integrals.
///
/// In the `ifxc = 5` branch FEFF calls `yzktd` with `j = ncore(imi)` and
/// `nu = 2`, so the Coulomb source is the product of the two bound core
/// orbitals for the `(imi, imj)` row/column pair. The resulting `ykgrex` field
/// is integrated against localized/full continuum products and later
/// subtracted by [`xsph_tdlda_angular_kernel`].
pub fn xsph_tdlda_nonlocal_exchange_integrals(
    input: XsphTdldaNonlocalExchangeInput<'_>,
) -> Result<XsphTdldaRadialKernel, XsphError> {
    validate_tdlda_nonlocal_exchange_input(&input)?;

    let mut radial_integrals = Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));
    let mut projected_radial_integrals =
        Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));

    for column in 0..input.matrix_size {
        if !input.positive_momentum_rows[column] {
            continue;
        }
        for row in 0..input.matrix_size {
            if !input.positive_momentum_rows[row]
                || input.initial_kappas[row] == input.initial_kappas[column]
            {
                continue;
            }

            let partner_large = input
                .core_large
                .index_axis(Axis(1), row)
                .mapv(|value| Complex::new(value, 0.0));
            let partner_small = input
                .core_small
                .index_axis(Axis(1), row)
                .mapv(|value| Complex::new(value, 0.0));
            let partner_large_coefficients = input
                .core_large_coefficients
                .index_axis(Axis(1), row)
                .mapv(|value| Complex::new(value, 0.0));
            let partner_small_coefficients = input
                .core_small_coefficients
                .index_axis(Axis(1), row)
                .mapv(|value| Complex::new(value, 0.0));
            let orbital_len = input.core_lengths[column].min(input.core_lengths[row]);

            let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                large_component: input.core_large.index_axis(Axis(1), column),
                small_component: input.core_small.index_axis(Axis(1), column),
                large_coefficients: input.core_large_coefficients.index_axis(Axis(1), column),
                small_coefficients: input.core_small_coefficients.index_axis(Axis(1), column),
                partner_large_component: partner_large.view(),
                partner_small_component: partner_small.view(),
                partner_large_coefficients: partner_large_coefficients.view(),
                partner_small_coefficients: partner_small_coefficients.view(),
                radii: input.radii,
                orbital_power: input.core_powers[column],
                partner_power: input.core_powers[row],
                step: input.step,
                angular_momentum: input.multipole,
                coefficient_count: input.coefficient_count,
                orbital_len,
                source_len: input.source_len,
                active_len: input.active_len,
            })?;

            radial_integrals[(row, column)] =
                tdlda_nonlocal_exchange_integral(&input, transform.yk.view(), row, column, false)?;
            projected_radial_integrals[(row, column)] =
                tdlda_nonlocal_exchange_integral(&input, transform.yk.view(), row, column, true)?;
        }
    }

    Ok(XsphTdldaRadialKernel {
        radial_integrals,
        projected_radial_integrals,
    })
}

/// Port of FEFF `TDLDA/getwf.f90` projector cleanup after a basis orbital is chosen.
///
/// FEFF orthogonalizes `pat/qat` against previously stored projectors using
/// `somm2` inside the Norman sphere, then normalizes the cleaned projector with
/// the same `somm2` norm before storing it in `dgcnp/dpcnp`.
pub fn xsph_tdlda_projector_orthogonalization(
    input: XsphTdldaProjectorOrthogonalizationInput<'_>,
) -> Result<XsphTdldaProjectorOrthogonalization, XsphError> {
    validate_tdlda_projector_orthogonalization_input(&input)?;

    let output_len = input.candidate_large.len();
    let previous_count = input.previous_large.ncols();
    let active_radii = input
        .radii
        .iter()
        .take(input.active_len)
        .copied()
        .collect::<Vec<_>>();
    let near_origin_power = 2.0 * input.final_l as Real + 2.0;
    let mut large = input.candidate_large.to_owned();
    let mut small = input.candidate_small.to_owned();
    let mut overlaps = Array1::<Real>::zeros(previous_count);

    for previous in 0..previous_count {
        let samples = (0..input.active_len)
            .map(|radial| {
                large[radial] * input.previous_large[(radial, previous)]
                    + small[radial] * input.previous_small[(radial, previous)]
            })
            .collect::<Vec<_>>();
        let overlap = somm2(
            &active_radii,
            &samples,
            input.log_step,
            near_origin_power,
            input.norman_radius,
            0,
        )?;
        validate_finite_real("tdlda_projector_overlap", overlap)?;
        overlaps[previous] = overlap;
        for radial in 0..output_len {
            large[radial] -= overlap * input.previous_large[(radial, previous)];
            small[radial] -= overlap * input.previous_small[(radial, previous)];
            validate_finite_real("tdlda_projector_large", large[radial])?;
            validate_finite_real("tdlda_projector_small", small[radial])?;
        }
    }

    let norm_samples = (0..input.active_len)
        .map(|radial| large[radial].powi(2) + small[radial].powi(2))
        .collect::<Vec<_>>();
    let norm_integral = somm2(
        &active_radii,
        &norm_samples,
        input.log_step,
        near_origin_power,
        input.norman_radius,
        0,
    )?;
    if !norm_integral.is_finite() || norm_integral <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "tdlda_projector_norm",
            value: norm_integral,
        });
    }
    let norm_sqrt = norm_integral.sqrt();
    validate_finite_real("tdlda_projector_norm_sqrt", norm_sqrt)?;
    for radial in 0..output_len {
        large[radial] /= norm_sqrt;
        small[radial] /= norm_sqrt;
        validate_finite_real("tdlda_projector_large", large[radial])?;
        validate_finite_real("tdlda_projector_small", small[radial])?;
    }

    Ok(XsphTdldaProjectorOrthogonalization {
        large,
        small,
        overlaps,
        norm_integral,
        norm_sqrt,
    })
}

/// Port of FEFF `TDLDA/getchi0.f90` Coulomb/xc radial kernel integrals.
///
/// This forms the `rabcd` and `rabcdp` matrices from source radial products
/// and the `Y(r)` fields produced by FEFF `yzktd`. Angular Wigner prefactors are
/// applied later by [`xsph_tdlda_angular_kernel`].
pub fn xsph_tdlda_radial_kernel_integrals(
    input: XsphTdldaRadialKernelInput<'_>,
) -> Result<XsphTdldaRadialKernel, XsphError> {
    validate_tdlda_radial_kernel_input(&input)?;

    let mut radial_integrals = Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));
    let mut projected_radial_integrals =
        Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));

    for column in 0..input.matrix_size {
        if !input.positive_momentum_rows[column] {
            continue;
        }
        for row in 0..input.matrix_size {
            if !input.positive_momentum_rows[row] {
                continue;
            }

            radial_integrals[(row, column)] =
                tdlda_radial_kernel_integral(&input, row, column, false)?;
            projected_radial_integrals[(row, column)] =
                tdlda_radial_kernel_integral(&input, row, column, true)?;
        }
    }

    Ok(XsphTdldaRadialKernel {
        radial_integrals,
        projected_radial_integrals,
    })
}

/// Port of FEFF `TDLDA/getchi0.f90` angular Coulomb/xc kernel accumulation.
///
/// The radial integrations that produce `rabcd`/`rabcdp` are source-backed
/// inputs here. This routine applies FEFF's Wigner-3j angular prefactors for
/// the ordinary `nu = 1` kernel and, when supplied, subtracts the dominant
/// PMBSE nonlocal exchange `nu = 2` term for rows with different initial kappa.
pub fn xsph_tdlda_angular_kernel(
    input: XsphTdldaAngularKernelInput<'_>,
) -> Result<XsphTdldaAngularKernel, XsphError> {
    validate_tdlda_angular_kernel_input(&input)?;

    let mut kernel = Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));
    let mut projected_kernel = Array2::<Complex>::zeros((input.matrix_size, input.matrix_size));
    let mut prefactors = Array2::<Real>::zeros((input.matrix_size, input.matrix_size));
    let mut nonlocal_prefactors = Array2::<Real>::zeros((input.matrix_size, input.matrix_size));

    for row in 0..input.matrix_size {
        if !input.positive_momentum_rows[row] {
            continue;
        }
        for column in 0..input.matrix_size {
            if !input.positive_momentum_rows[column] {
                continue;
            }

            if angular_selection_allows(
                input.final_m2[row],
                input.initial_m2[column],
                input.initial_m2[row],
                input.final_m2[column],
            ) {
                let prefactor = tdlda_angular_kernel_prefactor(TdldaAngularPrefactorInput {
                    final_j2_left: input.final_j2[row],
                    final_m2_left: input.final_m2[row],
                    initial_j2_left: input.initial_j2[row],
                    initial_m2_left: input.initial_m2[row],
                    final_j2_right: input.final_j2[column],
                    final_m2_right: input.final_m2[column],
                    initial_j2_right: input.initial_j2[column],
                    initial_m2_right: input.initial_m2[column],
                    multipole: 1,
                })?;
                kernel[(row, column)] += input.radial_integrals[(row, column)] * prefactor;
                projected_kernel[(row, column)] +=
                    input.projected_radial_integrals[(row, column)] * prefactor;
                prefactors[(row, column)] = prefactor;
            }

            if let (Some(nonlocal), Some(nonlocal_projected)) = (
                input.nonlocal_radial_integrals,
                input.nonlocal_projected_radial_integrals,
            ) {
                if input.initial_kappas[row] == input.initial_kappas[column] {
                    continue;
                }
                if !angular_selection_allows(
                    input.final_m2[row],
                    input.initial_m2[column],
                    input.final_m2[column],
                    input.initial_m2[row],
                ) {
                    continue;
                }
                let prefactor = tdlda_angular_kernel_prefactor(TdldaAngularPrefactorInput {
                    final_j2_left: input.final_j2[row],
                    final_m2_left: input.final_m2[row],
                    initial_j2_left: input.final_j2[column],
                    initial_m2_left: input.final_m2[column],
                    final_j2_right: input.initial_j2[row],
                    final_m2_right: input.initial_m2[row],
                    initial_j2_right: input.initial_j2[column],
                    initial_m2_right: input.initial_m2[column],
                    multipole: 2,
                })?;
                kernel[(row, column)] -= nonlocal[(row, column)] * prefactor;
                projected_kernel[(row, column)] -= nonlocal_projected[(row, column)] * prefactor;
                nonlocal_prefactors[(row, column)] = prefactor;
            }
        }
    }

    Ok(XsphTdldaAngularKernel {
        kernel,
        projected_kernel,
        prefactors,
        nonlocal_prefactors,
    })
}

/// Port of FEFF `TDLDA/xsectd.f90` PMBSE/TDLDA separation function.
///
/// FEFF uses a smooth cubic ramp from 0 at 100 eV to 1 at 150 eV above the
/// edge, then overrides it for PMBSE modes: `ipmbse = 1` or `3` forces the
/// combined branch (`1`), while `ipmbse = 2` forces PMBSE only (`0`).
pub fn xsph_tdlda_separation_function(
    ipmbse: i32,
    energies_hartree: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<Array1<Real>, XsphError> {
    validate_active_len("tdlda_sfun_energy", energies_hartree.len(), active_len)?;

    let e1 = 100.0 / super::XSPH_HARTREE_EV;
    let e2 = 150.0 / super::XSPH_HARTREE_EV;
    let midpoint = (e1 + e2) / 2.0;
    let half_width = (e2 - midpoint).abs().max(0.05 / super::XSPH_HARTREE_EV);
    let mut values = Array1::<Real>::zeros(active_len);
    for index in 0..active_len {
        let energy = energies_hartree[index];
        validate_finite_real("tdlda_sfun_energy", energy)?;
        let xx = (energy - midpoint) / half_width;
        values[index] = if xx < -1.0 {
            0.0
        } else if xx >= 1.0 {
            1.0
        } else {
            0.25 * (2.0 + 3.0 * xx - xx.powi(3))
        };
        if ipmbse == 1 || ipmbse == 3 {
            values[index] = 1.0;
        } else if ipmbse == 2 {
            values[index] = 0.0;
        }
    }
    Ok(values)
}

/// Port of FEFF `TDLDA/kkchi.f90` Kramers-Kronig response transform.
///
/// FEFF evaluates the principal-value integral on a 2000-point constant-step
/// fake grid around each requested energy and linearly interpolates the
/// imaginary response onto that grid. This preserves the same grid construction,
/// lower-edge cutoff, and second-pole correction used before `xsectd` builds
/// the complex `chi0` matrix.
pub fn xsph_tdlda_kramers_kronig_response(
    input: XsphTdldaKramersKronigInput<'_>,
) -> Result<XsphTdldaKramersKronigResponse, XsphError> {
    validate_tdlda_kk_input(&input)?;
    let energies = input
        .energy_hartree
        .iter()
        .take(input.energy_count)
        .copied()
        .collect::<Vec<_>>();
    let mut real_response =
        Array3::<Real>::zeros((input.energy_count, input.matrix_size, input.matrix_size));

    for target_energy in 0..input.energy_count {
        let fake_grid = tdlda_kk_fake_grid(&energies, target_energy)?;
        let target = energies[target_energy];
        for row in 0..input.matrix_size {
            let cutoff = input.edge_energy - input.reference_shifts[row];
            validate_finite_real("tdlda_kk_cutoff", cutoff)?;
            for column in 0..input.matrix_size {
                let fake_values = tdlda_kk_interpolate_fake_values(
                    &energies,
                    input.imaginary_response,
                    input.energy_count,
                    row,
                    column,
                    &fake_grid,
                )?;
                let value = tdlda_kk_integral(
                    target,
                    input.chemical_potential,
                    cutoff,
                    &fake_grid,
                    &fake_values,
                )?;
                real_response[(target_energy, row, column)] = value;
            }
        }
    }

    Ok(XsphTdldaKramersKronigResponse { real_response })
}

/// Port of FEFF `TDLDA/xsectd.f90` response broadening and complex assembly.
///
/// `xsectd` first broadens each raw `chi0im(:,im,imp)` row with FEFF `conv`
/// using `gammab(im)`, then calls `kkchi` and assembles
/// `chi0 = chi0r + i*chi0im` before the screened dipole solve.
pub fn xsph_tdlda_condition_response(
    input: XsphTdldaResponseConditioningInput<'_>,
) -> Result<XsphTdldaConditionedResponse, XsphError> {
    validate_tdlda_response_conditioning_input(&input)?;
    let energies = input
        .energy_hartree
        .iter()
        .take(input.energy_count)
        .copied()
        .collect::<Vec<_>>();
    let mut broadened_imaginary_response =
        Array3::<Real>::zeros((input.energy_count, input.matrix_size, input.matrix_size));

    for row in 0..input.matrix_size {
        for column in 0..input.matrix_size {
            let values = (0..input.energy_count)
                .map(|energy| Complex::new(input.imaginary_response[(energy, row, column)], 0.0))
                .collect::<Vec<_>>();
            let broadened = conv(&energies, &values, input.row_broadenings[row])?;
            for energy in 0..input.energy_count {
                broadened_imaginary_response[(energy, row, column)] = broadened[energy].re;
            }
        }
    }

    let real = xsph_tdlda_kramers_kronig_response(XsphTdldaKramersKronigInput {
        energy_count: input.energy_count,
        matrix_size: input.matrix_size,
        energy_hartree: input.energy_hartree,
        chemical_potential: input.chemical_potential,
        edge_energy: input.edge_energy,
        reference_shifts: input.reference_shifts,
        imaginary_response: broadened_imaginary_response.view(),
    })?;
    let mut response =
        Array3::<Complex>::zeros((input.energy_count, input.matrix_size, input.matrix_size));
    for energy in 0..input.energy_count {
        for row in 0..input.matrix_size {
            for column in 0..input.matrix_size {
                response[(energy, row, column)] = Complex::new(
                    real.real_response[(energy, row, column)],
                    broadened_imaginary_response[(energy, row, column)],
                );
            }
        }
    }

    Ok(XsphTdldaConditionedResponse {
        broadened_imaginary_response,
        real_response: real.real_response,
        response,
    })
}

/// Port of FEFF `TDLDA/ridxmu.f90` channel multiplier interpolation.
///
/// FEFF reads PMBSE-generated `xmu.dat` channel files, converts each row to
/// `1 + chi/mu0`, merges the two `l -> l + 1` edge grids when spin-orbit
/// splitting is present, then interpolates the `chil3/chil2/chil5/chil4`
/// multipliers onto that output grid. Channel order in the returned matrix is
/// the order consumed by [`xsph_tdlda_xsedge_rows`]: `l3`, `l2`, `l5`, `l4`.
pub fn xsph_tdlda_channel_multipliers(
    input: XsphTdldaChannelMultipliersInput<'_>,
) -> Result<XsphTdldaChannelMultipliers, XsphError> {
    validate_active_len(
        "tdlda_ridxmu_energy_capacity",
        input.energy_capacity,
        input.energy_capacity,
    )?;
    if input.initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }

    let dominant_plus = tdlda_xmu_channel("tdlda_ridxmu_odd_plus", input.dominant_plus)?;
    let split_required = input.initial_kappa < -1;
    let (energy_hartree, spin_orbit_split, split_plus, dominant_minus, split_minus) =
        if split_required {
            let split_plus = tdlda_xmu_channel(
                "tdlda_ridxmu_even_plus",
                input.split_plus.ok_or(XsphError::MissingTdldaChannel {
                    name: "tdlda_ridxmu_even_plus",
                })?,
            )?;
            let dominant_minus = tdlda_xmu_channel(
                "tdlda_ridxmu_odd_minus",
                input.dominant_minus.ok_or(XsphError::MissingTdldaChannel {
                    name: "tdlda_ridxmu_odd_minus",
                })?,
            )?;
            let split_minus = tdlda_xmu_channel(
                "tdlda_ridxmu_even_minus",
                input.split_minus.ok_or(XsphError::MissingTdldaChannel {
                    name: "tdlda_ridxmu_even_minus",
                })?,
            )?;
            let spin_orbit_split =
                (split_plus.edge_photon_ev - dominant_plus.edge_photon_ev) / super::XSPH_HARTREE_EV;
            validate_finite_real("tdlda_ridxmu_spin_orbit_split", spin_orbit_split)?;
            let split_plus_energy = split_plus
                .energy_hartree
                .iter()
                .map(|energy| energy + spin_orbit_split)
                .collect::<Vec<_>>();
            let energy_hartree = tdlda_ridxmu_merged_energy_grid(
                &dominant_plus.energy_hartree,
                &split_plus_energy,
                split_plus.edge_photon_ev,
                input.energy_capacity,
            )?;
            (
                energy_hartree,
                spin_orbit_split,
                Some((split_plus, split_plus_energy)),
                Some(dominant_minus),
                Some(split_minus),
            )
        } else {
            (
                dominant_plus
                    .energy_hartree
                    .iter()
                    .take(input.energy_capacity)
                    .copied()
                    .collect::<Vec<_>>(),
                0.0,
                None,
                None,
                None,
            )
        };

    let mut channel_multipliers = Array2::<Real>::from_elem((energy_hartree.len(), 4), 1.0);
    tdlda_ridxmu_interpolate_channel(
        &energy_hartree,
        &dominant_plus.energy_hartree,
        &dominant_plus.multipliers,
        channel_multipliers.column_mut(0),
    )?;

    if let (Some((split_plus, split_plus_energy)), Some(dominant_minus), Some(split_minus)) = (
        split_plus.as_ref(),
        dominant_minus.as_ref(),
        split_minus.as_ref(),
    ) {
        tdlda_ridxmu_interpolate_channel(
            &energy_hartree,
            split_plus_energy,
            &split_plus.multipliers,
            channel_multipliers.column_mut(1),
        )?;
        tdlda_ridxmu_interpolate_channel(
            &energy_hartree,
            &dominant_minus.energy_hartree,
            &dominant_minus.multipliers,
            channel_multipliers.column_mut(2),
        )?;
        tdlda_ridxmu_interpolate_channel(
            &energy_hartree,
            split_plus_energy,
            &split_minus.multipliers,
            channel_multipliers.column_mut(3),
        )?;
    }

    tdlda_ridxmu_pad_channel_prefix(&mut channel_multipliers, 0, &[2])?;
    if split_required {
        tdlda_ridxmu_pad_channel_prefix(&mut channel_multipliers, 1, &[3])?;
    }

    Ok(XsphTdldaChannelMultipliers {
        energy_hartree: Array1::from_vec(energy_hartree),
        spin_orbit_split,
        channel_multipliers,
    })
}

/// Port of FEFF `TDLDA/xsectd.f90` PMBSE weighting of raw `getchi0` response rows.
///
/// `getchi0` returns the local raw `chi(im,imp)` matrix for one energy row.
/// `xsectd` stores it in `chi0im(ie,im,imp)` after multiplying each row by
/// the PMBSE fine-structure channel selected from `kinitm(im)` and
/// `kfinm(im)`: `chil3`, `chil2`, `chil5`, or `chil4`.
pub fn xsph_tdlda_weight_response(
    input: XsphTdldaWeightedResponseInput<'_>,
) -> Result<XsphTdldaWeightedResponse, XsphError> {
    validate_tdlda_weighted_response_input(&input)?;

    let mut row_channels = Array1::<usize>::zeros(input.matrix_size);
    for row in 0..input.matrix_size {
        row_channels[row] =
            tdlda_weighted_response_channel(input.initial_kappas[row], input.final_kappas[row])?;
    }

    let mut imaginary_response =
        Array3::<Real>::zeros((input.energy_count, input.matrix_size, input.matrix_size));
    for energy in 0..input.energy_count {
        for row in 0..input.matrix_size {
            let channel = row_channels[row];
            let multiplier = input.channel_multipliers[(energy, channel)];
            validate_finite_real("tdlda_weighted_response_multiplier", multiplier)?;
            for column in 0..input.matrix_size {
                let value = input.raw_imaginary_response[(energy, row, column)] * multiplier;
                validate_finite_real("tdlda_weighted_response", value)?;
                imaginary_response[(energy, row, column)] = value;
            }
        }
    }

    Ok(XsphTdldaWeightedResponse {
        imaginary_response,
        row_channels,
    })
}

/// Port of FEFF `TDLDA/xsectd.f90` channel-spectrum accumulation.
///
/// This is the post-`dmscf` loop that fills `xsnorml3/l2/l5/l4` and
/// `xsscfl3/l2/l5/l4`. The returned channel arrays are already multiplied by
/// FEFF's relativistic `prefacl3`/`prefacl2` factors; inactive channels for
/// `nch = 1` or `2` remain zero.
pub fn xsph_tdlda_channel_spectra(
    input: XsphTdldaChannelSpectraInput<'_>,
) -> Result<XsphTdldaChannelSpectra, XsphError> {
    let active_channels = validate_tdlda_channel_spectra_input(&input)?;

    let mut single_particle_channels = Array2::<Real>::zeros((input.energy_count, 4));
    let mut screened_channels = Array2::<Real>::zeros((input.energy_count, 4));
    let mut plus_prefactors = Array1::<Real>::zeros(input.energy_count);
    let mut minus_prefactors = Array1::<Real>::zeros(input.energy_count);

    for energy in 0..input.energy_count {
        let prefactor = tdlda_channel_prefactor(input.photon_energy[energy])?;
        let plus_prefactor = -2.0 * input.plus_wave_number[energy] * prefactor;
        let minus_prefactor = -2.0 * input.minus_wave_number[energy] * prefactor;
        validate_finite_real("tdlda_channel_plus_prefactor", plus_prefactor)?;
        validate_finite_real("tdlda_channel_minus_prefactor", minus_prefactor)?;
        plus_prefactors[energy] = plus_prefactor;
        minus_prefactors[energy] = minus_prefactor;

        for row in 0..input.matrix_size {
            let channel =
                tdlda_channel_index(input.initial_kappas[row], row, input.primary_channel_count);
            if !active_channels.contains(&channel) {
                continue;
            }

            let dipole = input.dipole_matrix[(energy, row)];
            let mut screened_amplitude = Complex::new(dipole, 0.0);
            for projected in 0..input.matrix_size {
                for response_column in 0..input.matrix_size {
                    screened_amplitude += input.projected_kernel[(energy, row, projected)]
                        * input.response[(energy, projected, response_column)]
                        * input.screened_dipoles[(energy, response_column)];
                }
            }
            validate_finite_complex(
                "tdlda_channel_screened_amplitude",
                energy * input.matrix_size + row,
                screened_amplitude,
            )?;

            single_particle_channels[(energy, channel)] += dipole * dipole;
            screened_channels[(energy, channel)] += screened_amplitude.norm_sqr();
        }

        for &channel in active_channels {
            let scale = if matches!(channel, 0 | 2) {
                plus_prefactor
            } else {
                minus_prefactor
            };
            single_particle_channels[(energy, channel)] *= scale;
            screened_channels[(energy, channel)] *= scale;
            validate_finite_real(
                "tdlda_channel_single_particle",
                single_particle_channels[(energy, channel)],
            )?;
            validate_finite_real(
                "tdlda_channel_screened",
                screened_channels[(energy, channel)],
            )?;
        }
    }

    Ok(XsphTdldaChannelSpectra {
        single_particle_channels,
        screened_channels,
        plus_prefactors,
        minus_prefactors,
    })
}

/// Port of FEFF `TDLDA/xsectd.f90` channel broadening before `xsedge.dat`.
///
/// FEFF first zeros each spin-orbit channel below its onset (`edge` for
/// `l3/l5`, `edge + deltaso` for `l2/l4`), applies `conv` with the
/// corresponding core-hole broadening (`gaml3` or `gaml2`), then copies the
/// real part back into the channel work arrays.
pub fn xsph_tdlda_broaden_channel_spectra(
    input: XsphTdldaChannelBroadeningInput<'_>,
) -> Result<XsphTdldaBroadenedChannelSpectra, XsphError> {
    let active_channels = validate_tdlda_channel_broadening_input(&input)?;
    let energies = input
        .energy_hartree
        .iter()
        .take(input.energy_count)
        .copied()
        .collect::<Vec<_>>();

    let mut single_particle_channels = Array2::<Real>::zeros((input.energy_count, 4));
    let mut screened_channels = Array2::<Real>::zeros((input.energy_count, 4));

    for &channel in active_channels {
        let threshold =
            tdlda_channel_broadening_threshold(channel, input.edge_energy, input.spin_orbit_split)?;
        let width =
            tdlda_channel_broadening_width(channel, input.plus_broadening, input.minus_broadening);

        let single = tdlda_threshold_channel(
            &energies,
            input.single_particle_channels,
            input.energy_count,
            channel,
            threshold,
        )?;
        let screened = tdlda_threshold_channel(
            &energies,
            input.screened_channels,
            input.energy_count,
            channel,
            threshold,
        )?;
        let broadened_single = conv(&energies, &single, width)?;
        let broadened_screened = conv(&energies, &screened, width)?;

        for energy in 0..input.energy_count {
            single_particle_channels[(energy, channel)] = broadened_single[energy].re;
            screened_channels[(energy, channel)] = broadened_screened[energy].re;
        }
    }

    Ok(XsphTdldaBroadenedChannelSpectra {
        single_particle_channels,
        screened_channels,
    })
}

/// Port of FEFF `TDLDA/xsectd.f90` final `xsedge.dat` row assembly.
///
/// The upstream driver applies the channel multipliers after broadening, then
/// writes totals for the active spin-orbit channels. Channel order follows the
/// FEFF work arrays: `l3`, `l2`, `l5`, `l4`.
pub fn xsph_tdlda_xsedge_rows(
    input: XsphTdldaXsedgeRowsInput<'_>,
) -> Result<XsphTdldaXsedgeRows, XsphError> {
    let active_channels = validate_tdlda_xsedge_input(&input)?;

    let mut energy_ev = Array1::<Real>::zeros(input.energy_count);
    let mut total_single_particle = Array1::<Real>::zeros(input.energy_count);
    let mut total_screened = Array1::<Real>::zeros(input.energy_count);
    let mut plus_branch_single_particle = Array1::<Real>::zeros(input.energy_count);
    let mut minus_branch_single_particle = Array1::<Real>::zeros(input.energy_count);
    let mut plus_branch_screened = Array1::<Real>::zeros(input.energy_count);
    let mut minus_branch_screened = Array1::<Real>::zeros(input.energy_count);

    for energy in 0..input.energy_count {
        let output_energy = input.energy_hartree[energy] * super::XSPH_HARTREE_EV;
        validate_finite_real("tdlda_xsedge_energy_ev", output_energy)?;
        energy_ev[energy] = output_energy;

        for &channel in active_channels {
            let multiplier = input.channel_multipliers[(energy, channel)];
            let single = input.single_particle_channels[(energy, channel)] * multiplier;
            let screened = input.screened_channels[(energy, channel)] * multiplier;
            validate_finite_real("tdlda_xsedge_single_particle", single)?;
            validate_finite_real("tdlda_xsedge_screened", screened)?;

            total_single_particle[energy] += single;
            total_screened[energy] += screened;
            if matches!(channel, 0 | 2) {
                plus_branch_single_particle[energy] += single;
                plus_branch_screened[energy] += screened;
            } else {
                minus_branch_single_particle[energy] += single;
                minus_branch_screened[energy] += screened;
            }
        }
    }

    Ok(XsphTdldaXsedgeRows {
        energy_ev,
        total_single_particle,
        total_screened,
        plus_branch_single_particle,
        minus_branch_single_particle,
        plus_branch_screened,
        minus_branch_screened,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct TdldaXmuChannel {
    energy_hartree: Vec<Real>,
    multipliers: Vec<Real>,
    edge_photon_ev: Real,
}

fn tdlda_xmu_channel(
    name: &'static str,
    input: XsphTdldaXmuChannelInput<'_>,
) -> Result<TdldaXmuChannel, XsphError> {
    validate_active_len(name, input.photon_energy_ev.len(), input.point_count)?;
    validate_active_len(name, input.relative_energy_ev.len(), input.point_count)?;
    validate_active_len(name, input.wave_number.len(), input.point_count)?;
    validate_active_len(name, input.background.len(), input.point_count)?;
    validate_active_len(name, input.fine_structure.len(), input.point_count)?;
    if input.point_count < 4 {
        return Err(XsphError::SizeOutOfRange {
            name,
            value: input.point_count,
        });
    }

    let mut energy_hartree = Vec::<Real>::with_capacity(input.point_count);
    let mut multipliers = Vec::<Real>::with_capacity(input.point_count);
    let mut edge_photon_ev = None;
    for row in 0..input.point_count {
        let photon_energy = input.photon_energy_ev[row];
        let relative_energy = input.relative_energy_ev[row] / super::XSPH_HARTREE_EV;
        let wave_number = input.wave_number[row];
        let background = input.background[row];
        let fine_structure = input.fine_structure[row];
        validate_finite_real("tdlda_ridxmu_photon_energy", photon_energy)?;
        validate_finite_real("tdlda_ridxmu_relative_energy", relative_energy)?;
        validate_finite_real("tdlda_ridxmu_wave_number", wave_number)?;
        validate_finite_real("tdlda_ridxmu_background", background)?;
        validate_finite_real("tdlda_ridxmu_fine_structure", fine_structure)?;
        if background <= 0.0 {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_ridxmu_background",
                value: background,
            });
        }
        if row > 0 {
            let step = relative_energy - energy_hartree[row - 1];
            if !step.is_finite() || step <= 0.0 {
                return Err(XsphError::InvalidPositiveScalar {
                    name: "tdlda_ridxmu_energy_step",
                    value: step,
                });
            }
        }
        if wave_number > -0.01 && wave_number < 0.01 {
            edge_photon_ev = Some(photon_energy);
        }
        energy_hartree.push(relative_energy);
        multipliers.push(fine_structure / background + 1.0);
    }

    let edge_photon_ev = edge_photon_ev.ok_or(XsphError::MissingTdldaEdge { name })?;
    Ok(TdldaXmuChannel {
        energy_hartree,
        multipliers,
        edge_photon_ev,
    })
}

fn tdlda_ridxmu_merged_energy_grid(
    dominant_plus: &[Real],
    split_plus: &[Real],
    split_plus_edge_photon_ev: Real,
    capacity: usize,
) -> Result<Vec<Real>, XsphError> {
    let mut grid = Vec::with_capacity(capacity.min(dominant_plus.len() + split_plus.len()));
    let mut ix = None;
    for (index, &energy) in dominant_plus.iter().enumerate() {
        if energy < split_plus[0] && grid.len() < capacity {
            if index > grid.len() {
                return Err(XsphError::LengthTooShort {
                    name: "tdlda_ridxmu_dominant_plus_grid",
                    required: index + 1,
                    actual: grid.len(),
                });
            }
            grid.push(energy);
            ix = Some(index);
        }
    }
    let Some(mut dominant_index) = ix else {
        return Err(XsphError::EmptyIndexSet);
    };
    let mut split_index = 0_usize;

    loop {
        if dominant_index + 1 >= dominant_plus.len() {
            break;
        }
        validate_finite_real("tdlda_ridxmu_split_plus_edge", split_plus_edge_photon_ev)?;
        if split_plus[split_index] >= split_plus_edge_photon_ev {
            break;
        }
        if split_index + 1 >= split_plus.len() {
            break;
        }

        let dominant_step = dominant_plus[dominant_index + 1] - dominant_plus[dominant_index];
        let split_step = split_plus[split_index + 1] - split_plus[split_index];
        if split_step <= dominant_step {
            break;
        }
        if grid.len() >= capacity {
            return Ok(grid);
        }
        dominant_index += 1;
        grid.push(dominant_plus[dominant_index]);
        if dominant_index + 1 < dominant_plus.len()
            && dominant_plus[dominant_index + 1] > split_plus[split_index + 1]
        {
            split_index += 1;
        }
    }

    let end = split_plus
        .len()
        .min(capacity.saturating_sub(grid.len()) + split_index);
    for &energy in &split_plus[split_index..end] {
        if grid.len() >= capacity {
            break;
        }
        grid.push(energy);
    }
    if grid.is_empty() {
        return Err(XsphError::EmptyIndexSet);
    }
    Ok(grid)
}

fn tdlda_ridxmu_interpolate_channel(
    output_energy: &[Real],
    source_energy: &[Real],
    source_multiplier: &[Real],
    mut output: ArrayViewMut1<'_, Real>,
) -> Result<(), XsphError> {
    for (row, &energy) in output_energy.iter().enumerate() {
        let lower = source_energy[0];
        let upper = source_energy[source_energy.len() - 1];
        if energy >= lower - TDLDA_RIDXMU_RANGE_TOLERANCE
            && energy <= upper + TDLDA_RIDXMU_RANGE_TOLERANCE
        {
            let interpolation_energy = energy.clamp(lower, upper);
            output[row] = terp(source_energy, source_multiplier, 3, interpolation_energy)?.value;
            validate_finite_real("tdlda_ridxmu_multiplier", output[row])?;
        }
    }
    Ok(())
}

fn tdlda_ridxmu_pad_channel_prefix(
    channels: &mut Array2<Real>,
    source_channel: usize,
    companion_channels: &[usize],
) -> Result<(), XsphError> {
    let Some(first_nonunit) =
        (0..channels.nrows()).find(|&row| channels[(row, source_channel)] != 1.0)
    else {
        return Ok(());
    };
    let source_value = channels[(first_nonunit, source_channel)];
    let companion_values = companion_channels
        .iter()
        .map(|&channel| (channel, channels[(first_nonunit, channel)]))
        .collect::<Vec<_>>();
    for row in 0..first_nonunit {
        channels[(row, source_channel)] = source_value;
        for &(channel, value) in &companion_values {
            channels[(row, channel)] = value;
        }
    }
    Ok(())
}

fn validate_tdlda_row_wave_numbers_input(
    input: &XsphTdldaRowWaveNumbersInput<'_>,
) -> Result<(), XsphError> {
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_row_wave_number_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    validate_active_len(
        "tdlda_row_wave_number_refsh",
        input.reference_shifts.len(),
        input.matrix_size,
    )?;
    validate_finite_real("tdlda_row_wave_number_energy", input.energy_hartree)?;
    validate_finite_complex("tdlda_row_wave_number_reference", 0, input.reference_energy)?;
    for row in 0..input.matrix_size {
        validate_finite_real("tdlda_row_wave_number_refsh", input.reference_shifts[row])?;
    }
    Ok(())
}

fn validate_tdlda_raw_response_input(
    input: &XsphTdldaRawResponseInput<'_>,
) -> Result<(usize, usize, usize), XsphError> {
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_raw_response_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    if input.plus_basis_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_raw_response_plus_basis",
            required: 1,
            actual: 0,
        });
    }
    if input.initial_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "tdlda_raw_response_initial_l",
            index: 0,
            value: input.initial_l,
        });
    }
    if input.initial_l == 0 && input.minus_basis_count > 0 {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_raw_response_minus_basis",
            value: input.minus_basis_count,
        });
    }

    let plus_stride =
        tdlda_raw_response_stride("tdlda_raw_response_plus_stride", input.initial_l, 1)?;
    let minus_stride = if input.initial_l == 0 {
        0
    } else {
        tdlda_raw_response_stride("tdlda_raw_response_minus_stride", input.initial_l, -1)?
    };
    let plus_block_size =
        input
            .plus_basis_count
            .checked_mul(plus_stride)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_raw_response_plus_block",
                value: input.plus_basis_count,
            })?;
    let minus_block_size =
        input
            .minus_basis_count
            .checked_mul(minus_stride)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_raw_response_minus_block",
                value: input.minus_basis_count,
            })?;
    let expected_matrix_size =
        plus_block_size
            .checked_add(minus_block_size)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_raw_response_matrix_size",
                value: input.matrix_size,
            })?;
    if input.matrix_size < expected_matrix_size {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_raw_response_matrix_size",
            required: expected_matrix_size,
            actual: input.matrix_size,
        });
    }
    if input.matrix_size != expected_matrix_size {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_raw_response_matrix_size",
            value: input.matrix_size,
        });
    }

    validate_active_len(
        "tdlda_raw_response_refsh",
        input.reference_shifts.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_raw_response_wave_number",
        input.row_wave_numbers.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_raw_response_overlap",
        input.overlaps.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_raw_response_localized_dipole",
        input.localized_dipoles.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_raw_response_full_dipole",
        input.full_dipoles.len(),
        input.matrix_size,
    )?;
    validate_finite_real("tdlda_raw_response_energy", input.energy_hartree)?;
    validate_finite_real("tdlda_raw_response_edge", input.edge_energy)?;
    for row in 0..input.matrix_size {
        validate_finite_real("tdlda_raw_response_refsh", input.reference_shifts[row])?;
        validate_finite_real(
            "tdlda_raw_response_wave_number",
            input.row_wave_numbers[row],
        )?;
        validate_finite_real("tdlda_raw_response_overlap", input.overlaps[row])?;
        validate_finite_real(
            "tdlda_raw_response_localized_dipole",
            input.localized_dipoles[row],
        )?;
        validate_finite_real("tdlda_raw_response_full_dipole", input.full_dipoles[row])?;
    }

    Ok((plus_stride, minus_stride, plus_block_size))
}

fn validate_tdlda_projected_kernel_input(
    input: &XsphTdldaProjectedKernelInput<'_>,
) -> Result<(usize, usize, usize), XsphError> {
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_projected_kernel_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    if input.plus_basis_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_projected_kernel_plus_basis",
            required: 1,
            actual: 0,
        });
    }
    if input.initial_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "tdlda_projected_kernel_initial_l",
            index: 0,
            value: input.initial_l,
        });
    }
    if input.initial_l == 0 && input.minus_basis_count > 0 {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_projected_kernel_minus_basis",
            value: input.minus_basis_count,
        });
    }

    let plus_stride =
        tdlda_raw_response_stride("tdlda_projected_kernel_plus_stride", input.initial_l, 1)?;
    let minus_stride = if input.initial_l == 0 {
        0
    } else {
        tdlda_raw_response_stride("tdlda_projected_kernel_minus_stride", input.initial_l, -1)?
    };
    let plus_block_size =
        input
            .plus_basis_count
            .checked_mul(plus_stride)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_projected_kernel_plus_block",
                value: input.plus_basis_count,
            })?;
    let minus_block_size =
        input
            .minus_basis_count
            .checked_mul(minus_stride)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_projected_kernel_minus_block",
                value: input.minus_basis_count,
            })?;
    let expected_matrix_size =
        plus_block_size
            .checked_add(minus_block_size)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_projected_kernel_matrix_size",
                value: input.matrix_size,
            })?;
    if input.matrix_size < expected_matrix_size {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_projected_kernel_matrix_size",
            required: expected_matrix_size,
            actual: input.matrix_size,
        });
    }
    if input.matrix_size != expected_matrix_size {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_projected_kernel_matrix_size",
            value: input.matrix_size,
        });
    }

    validate_active_len(
        "tdlda_projected_kernel_rows",
        input.projected_kernel.nrows(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_projected_kernel_cols",
        input.projected_kernel.ncols(),
        input.matrix_size,
    )?;
    for row in 0..input.matrix_size {
        for column in 0..input.matrix_size {
            validate_finite_complex(
                "tdlda_projected_kernel",
                row * input.matrix_size + column,
                input.projected_kernel[(row, column)],
            )?;
        }
    }

    Ok((plus_stride, minus_stride, plus_block_size))
}

fn validate_tdlda_direct_kernel_input(
    input: &XsphTdldaDirectKernelInput<'_>,
) -> Result<(usize, usize, usize), XsphError> {
    if input.active_len < 2 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_direct_kernel_active_len",
            required: 2,
            actual: input.active_len,
        });
    }
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_direct_kernel_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    if input.plus_basis_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_direct_kernel_plus_basis",
            required: 1,
            actual: 0,
        });
    }
    if input.initial_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "tdlda_direct_kernel_initial_l",
            index: 0,
            value: input.initial_l,
        });
    }
    if input.initial_l == 0 && input.minus_basis_count > 0 {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_direct_kernel_minus_basis",
            value: input.minus_basis_count,
        });
    }

    let plus_stride =
        tdlda_raw_response_stride("tdlda_direct_kernel_plus_stride", input.initial_l, 1)?;
    let minus_stride = if input.initial_l == 0 {
        0
    } else {
        tdlda_raw_response_stride("tdlda_direct_kernel_minus_stride", input.initial_l, -1)?
    };
    let plus_block_size =
        input
            .plus_basis_count
            .checked_mul(plus_stride)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_direct_kernel_plus_block",
                value: input.plus_basis_count,
            })?;
    let minus_block_size =
        input
            .minus_basis_count
            .checked_mul(minus_stride)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_direct_kernel_minus_block",
                value: input.minus_basis_count,
            })?;
    let expected_matrix_size =
        plus_block_size
            .checked_add(minus_block_size)
            .ok_or(XsphError::SizeOutOfRange {
                name: "tdlda_direct_kernel_matrix_size",
                value: input.matrix_size,
            })?;
    if input.matrix_size < expected_matrix_size {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_direct_kernel_matrix_size",
            required: expected_matrix_size,
            actual: input.matrix_size,
        });
    }
    if input.matrix_size != expected_matrix_size {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_direct_kernel_matrix_size",
            value: input.matrix_size,
        });
    }

    validate_active_len(
        "tdlda_direct_kernel_refsh",
        input.reference_shifts.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_direct_kernel_momentum",
        input.momentum_squared.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_direct_kernel_radii",
        input.radii.len(),
        input.active_len,
    )?;
    validate_active_len(
        "tdlda_direct_kernel_vch",
        input.core_hole_potential.len(),
        input.active_len,
    )?;
    validate_tdlda_direct_kernel_matrix_shape(
        "tdlda_direct_kernel_localized_large",
        input.localized_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_direct_kernel_matrix_shape(
        "tdlda_direct_kernel_localized_small",
        input.localized_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_direct_kernel_matrix_shape(
        "tdlda_direct_kernel_full_large",
        input.full_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_direct_kernel_matrix_shape(
        "tdlda_direct_kernel_full_small",
        input.full_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_finite_real("tdlda_direct_kernel_energy", input.energy_hartree)?;
    validate_finite_real("tdlda_direct_kernel_edge", input.edge_energy)?;
    validate_finite_real("tdlda_direct_kernel_sfun", input.separation_function)?;

    for radial in 0..input.active_len {
        validate_finite_real("tdlda_direct_kernel_radius", input.radii[radial])?;
        if input.radii[radial] <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "tdlda_direct_kernel_radius",
                value: input.radii[radial],
            });
        }
        if radial > 0 && input.radii[radial] <= input.radii[radial - 1] {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_direct_kernel_radius_step",
                value: input.radii[radial] - input.radii[radial - 1],
            });
        }
        validate_finite_real("tdlda_direct_kernel_vch", input.core_hole_potential[radial])?;
        for row in 0..input.matrix_size {
            validate_finite_real(
                "tdlda_direct_kernel_localized_large",
                input.localized_large[(radial, row)],
            )?;
            validate_finite_real(
                "tdlda_direct_kernel_localized_small",
                input.localized_small[(radial, row)],
            )?;
            validate_finite_real(
                "tdlda_direct_kernel_full_large",
                input.full_large[(radial, row)],
            )?;
            validate_finite_real(
                "tdlda_direct_kernel_full_small",
                input.full_small[(radial, row)],
            )?;
        }
    }
    for row in 0..input.matrix_size {
        validate_finite_real("tdlda_direct_kernel_refsh", input.reference_shifts[row])?;
        validate_finite_real("tdlda_direct_kernel_momentum", input.momentum_squared[row])?;
    }

    Ok((plus_stride, minus_stride, plus_block_size))
}

fn validate_tdlda_direct_kernel_matrix_shape(
    name: &'static str,
    values: ArrayView2<'_, Real>,
    active_len: usize,
    matrix_size: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, values.nrows(), active_len)?;
    validate_active_len(name, values.ncols(), matrix_size)
}

fn tdlda_projected_kernel_representative_row(
    row: usize,
    plus_stride: usize,
    minus_stride: usize,
    plus_block_size: usize,
) -> usize {
    let row_1based = row + 1;
    if row_1based <= plus_block_size {
        let remainder = row_1based % plus_stride;
        if remainder == 0 {
            plus_stride - 1
        } else {
            remainder - 1
        }
    } else {
        let remainder = (row_1based - plus_block_size) % minus_stride;
        if remainder == 0 {
            plus_stride + minus_stride - 1
        } else {
            plus_stride + remainder - 1
        }
    }
}

fn tdlda_direct_kernel_integral(
    radii: ArrayView1<'_, Real>,
    core_hole_potential: ArrayView1<'_, Real>,
    direct_scale: Real,
    active_len: usize,
    mut radial_factor: impl FnMut(usize) -> Real,
) -> Result<Real, XsphError> {
    let mut integral = 0.0;
    let mut previous = direct_scale * core_hole_potential[0] * radial_factor(0);
    validate_finite_real("tdlda_direct_kernel_integrand", previous)?;
    for radial in 1..active_len {
        let current = direct_scale * core_hole_potential[radial] * radial_factor(radial);
        validate_finite_real("tdlda_direct_kernel_integrand", current)?;
        integral += (current + previous) * (radii[radial] - radii[radial - 1]) / 2.0;
        previous = current;
    }
    validate_finite_real("tdlda_direct_kernel_integral", integral)?;
    Ok(integral)
}

fn validate_tdlda_coulomb_fields_input(
    input: &XsphTdldaCoulombFieldsInput<'_>,
) -> Result<(), XsphError> {
    if input.active_len < 2 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_coulomb_field_active_len",
            required: 2,
            actual: input.active_len,
        });
    }
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_coulomb_field_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    if input.source_len == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_coulomb_field_source_len",
            required: 1,
            actual: 0,
        });
    }
    if input.coefficient_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_coulomb_field_coefficient_count",
            required: 1,
            actual: 0,
        });
    }
    validate_active_len(
        "tdlda_coulomb_field_radii",
        input.radii.len(),
        input.active_len,
    )?;
    validate_active_len(
        "tdlda_coulomb_field_core_power",
        input.core_powers.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_coulomb_field_core_length",
        input.core_lengths.len(),
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_coulomb_field_core_large",
        input.core_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_coulomb_field_core_small",
        input.core_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_coulomb_field_core_large_coefficients",
        input.core_large_coefficients,
        input.coefficient_count,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_coulomb_field_core_small_coefficients",
        input.core_small_coefficients,
        input.coefficient_count,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_complex_matrix_shape(
        "tdlda_coulomb_field_target_large",
        input.target_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_complex_matrix_shape(
        "tdlda_coulomb_field_target_small",
        input.target_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_complex_matrix_shape(
        "tdlda_coulomb_field_target_large_coefficients",
        input.target_large_coefficients,
        input.coefficient_count,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_complex_matrix_shape(
        "tdlda_coulomb_field_target_small_coefficients",
        input.target_small_coefficients,
        input.coefficient_count,
        input.matrix_size,
    )?;
    validate_finite_real("tdlda_coulomb_field_step", input.step)?;
    validate_active_len(
        "tdlda_coulomb_field_target_power",
        input.target_powers.len(),
        input.matrix_size,
    )?;

    for radial in 0..input.active_len {
        validate_finite_real("tdlda_coulomb_field_radius", input.radii[radial])?;
        if input.radii[radial] <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "tdlda_coulomb_field_radius",
                value: input.radii[radial],
            });
        }
        if radial > 0 && input.radii[radial] <= input.radii[radial - 1] {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_coulomb_field_radius_step",
                value: input.radii[radial] - input.radii[radial - 1],
            });
        }
    }
    for row in 0..input.matrix_size {
        validate_finite_real("tdlda_coulomb_field_core_power", input.core_powers[row])?;
        validate_finite_real("tdlda_coulomb_field_target_power", input.target_powers[row])?;
        if input.core_lengths[row] == 0 {
            return Err(XsphError::LengthTooShort {
                name: "tdlda_coulomb_field_core_length",
                required: 1,
                actual: 0,
            });
        }
    }

    Ok(())
}

fn validate_tdlda_coulomb_field_real_matrix_shape(
    name: &'static str,
    values: ArrayView2<'_, Real>,
    rows: usize,
    columns: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, values.nrows(), rows)?;
    validate_active_len(name, values.ncols(), columns)?;
    for row in 0..rows {
        for column in 0..columns {
            validate_finite_real(name, values[(row, column)])?;
        }
    }
    Ok(())
}

fn validate_tdlda_coulomb_field_complex_matrix_shape(
    name: &'static str,
    values: ArrayView2<'_, Complex>,
    rows: usize,
    columns: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, values.nrows(), rows)?;
    validate_active_len(name, values.ncols(), columns)?;
    for row in 0..rows {
        for column in 0..columns {
            validate_finite_complex(name, row * columns + column, values[(row, column)])?;
        }
    }
    Ok(())
}

fn validate_tdlda_nonlocal_exchange_input(
    input: &XsphTdldaNonlocalExchangeInput<'_>,
) -> Result<(), XsphError> {
    if input.active_len < 2 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_nonlocal_exchange_active_len",
            required: 2,
            actual: input.active_len,
        });
    }
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_nonlocal_exchange_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    if input.source_len == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_nonlocal_exchange_source_len",
            required: 1,
            actual: 0,
        });
    }
    if input.coefficient_count == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_nonlocal_exchange_coefficient_count",
            required: 1,
            actual: 0,
        });
    }
    validate_active_len(
        "tdlda_nonlocal_exchange_positive_momentum",
        input.positive_momentum_rows.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_nonlocal_exchange_initial_kappa",
        input.initial_kappas.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_nonlocal_exchange_radii",
        input.radii.len(),
        input.active_len,
    )?;
    validate_active_len(
        "tdlda_nonlocal_exchange_core_power",
        input.core_powers.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_nonlocal_exchange_core_length",
        input.core_lengths.len(),
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_nonlocal_exchange_core_large",
        input.core_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_nonlocal_exchange_core_small",
        input.core_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_nonlocal_exchange_core_large_coefficients",
        input.core_large_coefficients,
        input.coefficient_count,
        input.matrix_size,
    )?;
    validate_tdlda_coulomb_field_real_matrix_shape(
        "tdlda_nonlocal_exchange_core_small_coefficients",
        input.core_small_coefficients,
        input.coefficient_count,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_nonlocal_exchange_localized_large",
        input.localized_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_nonlocal_exchange_localized_small",
        input.localized_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_nonlocal_exchange_full_large",
        input.full_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_nonlocal_exchange_full_small",
        input.full_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_finite_real("tdlda_nonlocal_exchange_step", input.step)?;
    validate_finite_real("tdlda_nonlocal_exchange_direct_scale", input.direct_scale)?;

    for radial in 0..input.active_len {
        validate_finite_real("tdlda_nonlocal_exchange_radius", input.radii[radial])?;
        if input.radii[radial] <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "tdlda_nonlocal_exchange_radius",
                value: input.radii[radial],
            });
        }
        if radial > 0 && input.radii[radial] <= input.radii[radial - 1] {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_nonlocal_exchange_radius_step",
                value: input.radii[radial] - input.radii[radial - 1],
            });
        }
    }
    for row in 0..input.matrix_size {
        validate_finite_real("tdlda_nonlocal_exchange_core_power", input.core_powers[row])?;
        if input.core_lengths[row] == 0 {
            return Err(XsphError::LengthTooShort {
                name: "tdlda_nonlocal_exchange_core_length",
                required: 1,
                actual: 0,
            });
        }
    }

    Ok(())
}

fn validate_tdlda_projector_orthogonalization_input(
    input: &XsphTdldaProjectorOrthogonalizationInput<'_>,
) -> Result<(), XsphError> {
    if input.active_len < 4 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_projector_active_len",
            required: 4,
            actual: input.active_len,
        });
    }
    let output_len = input.candidate_large.len();
    validate_active_len("tdlda_projector_output_len", output_len, input.active_len)?;
    validate_active_len(
        "tdlda_projector_candidate_small",
        input.candidate_small.len(),
        output_len,
    )?;
    validate_active_len("tdlda_projector_radii", input.radii.len(), input.active_len)?;
    validate_active_len(
        "tdlda_projector_previous_large_rows",
        input.previous_large.nrows(),
        output_len,
    )?;
    validate_active_len(
        "tdlda_projector_previous_small_rows",
        input.previous_small.nrows(),
        output_len,
    )?;
    if input.previous_small.ncols() != input.previous_large.ncols() {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_projector_previous_small_columns",
            required: input.previous_large.ncols(),
            actual: input.previous_small.ncols(),
        });
    }
    for radial in 0..output_len {
        for previous in 0..input.previous_large.ncols() {
            validate_finite_real(
                "tdlda_projector_previous_large",
                input.previous_large[(radial, previous)],
            )?;
            validate_finite_real(
                "tdlda_projector_previous_small",
                input.previous_small[(radial, previous)],
            )?;
        }
    }
    validate_finite_real("tdlda_projector_log_step", input.log_step)?;
    if input.log_step <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "tdlda_projector_log_step",
            value: input.log_step,
        });
    }
    validate_finite_real("tdlda_projector_norman_radius", input.norman_radius)?;
    if input.norman_radius <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "tdlda_projector_norman_radius",
            value: input.norman_radius,
        });
    }

    for radial in 0..input.active_len {
        validate_finite_real("tdlda_projector_radius", input.radii[radial])?;
        if input.radii[radial] <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "tdlda_projector_radius",
                value: input.radii[radial],
            });
        }
        if radial > 0 && input.radii[radial] <= input.radii[radial - 1] {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_projector_radius_step",
                value: input.radii[radial] - input.radii[radial - 1],
            });
        }
    }
    for radial in 0..output_len {
        validate_finite_real(
            "tdlda_projector_candidate_large",
            input.candidate_large[radial],
        )?;
        validate_finite_real(
            "tdlda_projector_candidate_small",
            input.candidate_small[radial],
        )?;
    }

    Ok(())
}

fn validate_tdlda_radial_kernel_input(
    input: &XsphTdldaRadialKernelInput<'_>,
) -> Result<(), XsphError> {
    if input.active_len < 2 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_radial_kernel_active_len",
            required: 2,
            actual: input.active_len,
        });
    }
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_radial_kernel_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    validate_active_len(
        "tdlda_radial_kernel_positive_momentum",
        input.positive_momentum_rows.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_radial_kernel_initial_kappa",
        input.initial_kappas.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_radial_kernel_radii",
        input.radii.len(),
        input.active_len,
    )?;
    validate_active_len(
        "tdlda_radial_kernel_fxc0",
        input.exchange_correlation_same_edge.len(),
        input.active_len,
    )?;
    validate_active_len(
        "tdlda_radial_kernel_fxc",
        input.exchange_correlation_real.len(),
        input.active_len,
    )?;
    validate_active_len(
        "tdlda_radial_kernel_fxcim",
        input.exchange_correlation_imaginary.len(),
        input.active_len,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_response_large",
        input.response_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_response_small",
        input.response_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_localized_large",
        input.localized_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_localized_small",
        input.localized_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_full_large",
        input.full_large,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_full_small",
        input.full_small,
        input.active_len,
        input.matrix_size,
    )?;
    validate_tdlda_radial_kernel_matrix_shape(
        "tdlda_radial_kernel_ykgr",
        input.coulomb_fields,
        input.active_len,
        input.matrix_size,
    )?;
    validate_finite_real("tdlda_radial_kernel_direct_scale", input.direct_scale)?;

    for radial in 0..input.active_len {
        validate_finite_real("tdlda_radial_kernel_radius", input.radii[radial])?;
        if input.radii[radial] <= 0.0 {
            return Err(XsphError::InvalidPositiveRadius {
                name: "tdlda_radial_kernel_radius",
                value: input.radii[radial],
            });
        }
        if radial > 0 && input.radii[radial] <= input.radii[radial - 1] {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_radial_kernel_radius_step",
                value: input.radii[radial] - input.radii[radial - 1],
            });
        }
        validate_finite_real(
            "tdlda_radial_kernel_fxc0",
            input.exchange_correlation_same_edge[radial],
        )?;
        validate_finite_real(
            "tdlda_radial_kernel_fxc",
            input.exchange_correlation_real[radial],
        )?;
        validate_finite_real(
            "tdlda_radial_kernel_fxcim",
            input.exchange_correlation_imaginary[radial],
        )?;
    }
    for row in 0..input.matrix_size {
        if input.initial_kappas[row] == 0 {
            return Err(XsphError::ZeroKappa);
        }
    }

    Ok(())
}

fn validate_tdlda_radial_kernel_matrix_shape(
    name: &'static str,
    values: ArrayView2<'_, Complex>,
    active_len: usize,
    matrix_size: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, values.nrows(), active_len)?;
    validate_active_len(name, values.ncols(), matrix_size)?;
    for radial in 0..active_len {
        for row in 0..matrix_size {
            validate_finite_complex(name, radial * matrix_size + row, values[(radial, row)])?;
        }
    }
    Ok(())
}

fn tdlda_radial_kernel_integral(
    input: &XsphTdldaRadialKernelInput<'_>,
    row: usize,
    column: usize,
    projected: bool,
) -> Result<Complex, XsphError> {
    let mut integral = Complex::new(0.0, 0.0);
    let mut previous = tdlda_radial_kernel_integrand(input, 0, row, column, projected)?;
    for radial in 1..input.active_len {
        let current = tdlda_radial_kernel_integrand(input, radial, row, column, projected)?;
        integral += (current + previous) * (input.radii[radial] - input.radii[radial - 1]) / 2.0;
        previous = current;
    }
    validate_finite_complex(
        "tdlda_radial_kernel_integral",
        row * input.matrix_size + column,
        integral,
    )?;
    Ok(integral)
}

fn tdlda_radial_kernel_integrand(
    input: &XsphTdldaRadialKernelInput<'_>,
    radial: usize,
    row: usize,
    column: usize,
    projected: bool,
) -> Result<Complex, XsphError> {
    let row_product = if projected {
        tdlda_radial_component_product(
            input.response_large[(radial, row)],
            input.response_small[(radial, row)],
            input.full_large[(radial, row)],
            input.full_small[(radial, row)],
        )
    } else {
        tdlda_radial_component_product(
            input.response_large[(radial, row)],
            input.response_small[(radial, row)],
            input.localized_large[(radial, row)],
            input.localized_small[(radial, row)],
        )
    };
    let column_product = tdlda_radial_component_product(
        input.response_large[(radial, column)],
        input.response_small[(radial, column)],
        input.localized_large[(radial, column)],
        input.localized_small[(radial, column)],
    );
    let coulomb = row_product * input.coulomb_fields[(radial, column)].re / input.radii[radial]
        * input.direct_scale;
    let fxc = tdlda_radial_exchange_kernel(input, radial, row, column);
    let value = Complex::new(coulomb, 0.0) + fxc * row_product * column_product;
    validate_finite_complex(
        "tdlda_radial_kernel_integrand",
        radial * input.matrix_size * input.matrix_size + row * input.matrix_size + column,
        value,
    )?;
    Ok(value)
}

fn tdlda_radial_component_product(
    response_large: Complex,
    response_small: Complex,
    target_large: Complex,
    target_small: Complex,
) -> Real {
    (response_large * target_large + response_small * target_small).re
}

fn tdlda_radial_exchange_kernel(
    input: &XsphTdldaRadialKernelInput<'_>,
    radial: usize,
    row: usize,
    column: usize,
) -> Complex {
    if input.initial_kappas[row] == input.initial_kappas[column]
        && input.exchange_correlation_selector != 2
    {
        Complex::new(input.exchange_correlation_same_edge[radial], 0.0)
    } else if input.initial_kappas[row] > 0 || input.exchange_correlation_selector == 2 {
        Complex::new(
            input.exchange_correlation_real[radial],
            input.exchange_correlation_imaginary[radial],
        )
    } else {
        Complex::new(
            input.exchange_correlation_real[radial],
            -input.exchange_correlation_imaginary[radial],
        )
    }
}

fn tdlda_nonlocal_exchange_integral(
    input: &XsphTdldaNonlocalExchangeInput<'_>,
    field: ArrayView1<'_, Complex>,
    row: usize,
    column: usize,
    projected: bool,
) -> Result<Complex, XsphError> {
    validate_active_len(
        "tdlda_nonlocal_exchange_field",
        field.len(),
        input.active_len,
    )?;
    let mut integral = Complex::new(0.0, 0.0);
    let mut previous = tdlda_nonlocal_exchange_integrand(input, field, 0, row, column, projected)?;
    for radial in 1..input.active_len {
        let current =
            tdlda_nonlocal_exchange_integrand(input, field, radial, row, column, projected)?;
        integral += (current + previous) * (input.radii[radial] - input.radii[radial - 1]) / 2.0;
        previous = current;
    }
    validate_finite_complex(
        "tdlda_nonlocal_exchange_integral",
        row * input.matrix_size + column,
        integral,
    )?;
    Ok(integral)
}

fn tdlda_nonlocal_exchange_integrand(
    input: &XsphTdldaNonlocalExchangeInput<'_>,
    field: ArrayView1<'_, Complex>,
    radial: usize,
    row: usize,
    column: usize,
    projected: bool,
) -> Result<Complex, XsphError> {
    let product = if projected {
        tdlda_nonlocal_component_product(
            input.localized_large[(radial, column)],
            input.localized_small[(radial, column)],
            input.full_large[(radial, row)],
            input.full_small[(radial, row)],
        )
    } else {
        tdlda_nonlocal_component_product(
            input.localized_large[(radial, column)],
            input.localized_small[(radial, column)],
            input.localized_large[(radial, row)],
            input.localized_small[(radial, row)],
        )
    };
    let value = Complex::new(
        product * field[radial].re / input.radii[radial] * input.direct_scale,
        0.0,
    );
    validate_finite_complex(
        "tdlda_nonlocal_exchange_integrand",
        radial * input.matrix_size * input.matrix_size + row * input.matrix_size + column,
        value,
    )?;
    Ok(value)
}

fn tdlda_nonlocal_component_product(
    left_large: Complex,
    left_small: Complex,
    right_large: Complex,
    right_small: Complex,
) -> Real {
    (left_large * right_large + left_small * right_small).re
}

fn validate_tdlda_angular_kernel_input(
    input: &XsphTdldaAngularKernelInput<'_>,
) -> Result<(), XsphError> {
    if input.matrix_size == 0 {
        return Err(XsphError::LengthTooShort {
            name: "tdlda_angular_kernel_matrix_size",
            required: 1,
            actual: 0,
        });
    }
    validate_active_len(
        "tdlda_angular_kernel_initial_j2",
        input.initial_j2.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_angular_kernel_initial_m2",
        input.initial_m2.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_angular_kernel_initial_kappa",
        input.initial_kappas.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_angular_kernel_final_j2",
        input.final_j2.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_angular_kernel_final_m2",
        input.final_m2.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_angular_kernel_positive_momentum",
        input.positive_momentum_rows.len(),
        input.matrix_size,
    )?;
    validate_tdlda_angular_kernel_matrix_shape(
        "tdlda_angular_kernel_radial",
        input.radial_integrals,
        input.matrix_size,
    )?;
    validate_tdlda_angular_kernel_matrix_shape(
        "tdlda_angular_kernel_projected_radial",
        input.projected_radial_integrals,
        input.matrix_size,
    )?;

    match (
        input.nonlocal_radial_integrals,
        input.nonlocal_projected_radial_integrals,
    ) {
        (Some(nonlocal), Some(nonlocal_projected)) => {
            validate_tdlda_angular_kernel_matrix_shape(
                "tdlda_angular_kernel_nonlocal_radial",
                nonlocal,
                input.matrix_size,
            )?;
            validate_tdlda_angular_kernel_matrix_shape(
                "tdlda_angular_kernel_nonlocal_projected_radial",
                nonlocal_projected,
                input.matrix_size,
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(XsphError::LengthTooShort {
                name: "tdlda_angular_kernel_nonlocal_pair",
                required: 2,
                actual: 1,
            });
        }
    }

    for row in 0..input.matrix_size {
        validate_tdlda_angular_j2(
            "tdlda_angular_kernel_initial_j2",
            row,
            input.initial_j2[row],
        )?;
        validate_tdlda_angular_j2("tdlda_angular_kernel_final_j2", row, input.final_j2[row])?;
        validate_cwig3j_doubled_argument(
            "tdlda_angular_kernel_initial_j2",
            input.initial_j2[row],
            input.initial_j2[row],
        )?;
        validate_cwig3j_doubled_argument(
            "tdlda_angular_kernel_final_j2",
            input.final_j2[row],
            input.final_j2[row],
        )?;
        if input.initial_kappas[row] == 0 {
            return Err(XsphError::ZeroKappa);
        }
    }

    Ok(())
}

fn validate_tdlda_angular_j2(
    name: &'static str,
    index: usize,
    value: i32,
) -> Result<(), XsphError> {
    if value < 0 {
        return Err(XsphError::NegativeAngularMomentum { name, index, value });
    }
    Ok(())
}

fn validate_tdlda_angular_kernel_matrix_shape(
    name: &'static str,
    values: ArrayView2<'_, Complex>,
    matrix_size: usize,
) -> Result<(), XsphError> {
    validate_active_len(name, values.nrows(), matrix_size)?;
    validate_active_len(name, values.ncols(), matrix_size)?;
    for row in 0..matrix_size {
        for column in 0..matrix_size {
            validate_finite_complex(name, row * matrix_size + column, values[(row, column)])?;
        }
    }
    Ok(())
}

fn angular_selection_allows(ma2: i32, mb2: i32, mc2: i32, md2: i32) -> bool {
    i64::from(ma2) + i64::from(mb2) == i64::from(mc2) + i64::from(md2)
}

#[derive(Debug, Clone, Copy)]
struct TdldaAngularPrefactorInput {
    final_j2_left: i32,
    final_m2_left: i32,
    initial_j2_left: i32,
    initial_m2_left: i32,
    final_j2_right: i32,
    final_m2_right: i32,
    initial_j2_right: i32,
    initial_m2_right: i32,
    multipole: i32,
}

fn tdlda_angular_kernel_prefactor(input: TdldaAngularPrefactorInput) -> Result<Real, XsphError> {
    validate_cwig3j_integer_argument("tdlda_angular_kernel_multipole", input.multipole)?;
    let multipole2 = input
        .multipole
        .checked_mul(2)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "tdlda_angular_kernel_multipole",
            value: input.multipole,
        })?;
    validate_cwig3j_doubled_argument(
        "tdlda_angular_kernel_multipole",
        input.multipole,
        multipole2,
    )?;

    let left_angular = wigner_3j(
        input.final_j2_left,
        multipole2,
        input.initial_j2_left,
        1,
        0,
        2,
    )? * wigner_3j(
        input.final_j2_left,
        multipole2,
        input.initial_j2_left,
        -input.final_m2_left,
        input.final_m2_left - input.initial_m2_left,
        2,
    )?;
    let right_angular = wigner_3j(
        input.final_j2_right,
        multipole2,
        input.initial_j2_right,
        1,
        0,
        2,
    )? * wigner_3j(
        input.final_j2_right,
        multipole2,
        input.initial_j2_right,
        -input.final_m2_right,
        input.final_m2_right - input.initial_m2_right,
        2,
    )?;
    let phase = tdlda_angular_phase(input.final_m2_left, "tdlda_angular_kernel_final_m2_left")?
        * tdlda_angular_phase(input.final_m2_right, "tdlda_angular_kernel_final_m2_right")?;
    let degeneracy = (input.final_j2_left + 1) as Real
        * (input.final_j2_right + 1) as Real
        * (input.initial_j2_left + 1) as Real
        * (input.initial_j2_right + 1) as Real;
    let prefactor = phase * left_angular * right_angular * degeneracy.sqrt();
    validate_finite_real("tdlda_angular_kernel_prefactor", prefactor)?;
    Ok(prefactor)
}

fn tdlda_angular_phase(m2: i32, name: &'static str) -> Result<Real, XsphError> {
    let numerator = i64::from(m2) + 1;
    if numerator.rem_euclid(2) != 0 {
        return Err(XsphError::IntegerOutOfRange { name, value: m2 });
    }
    let exponent = numerator / 2;
    Ok(if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    })
}

fn tdlda_raw_response_stride(
    name: &'static str,
    initial_l: i32,
    offset: i32,
) -> Result<usize, XsphError> {
    let angular_slots = initial_l
        .checked_mul(2)
        .and_then(|value| value.checked_add(offset))
        .ok_or(XsphError::IntegerOutOfRange {
            name,
            value: initial_l,
        })?;
    if angular_slots <= 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name,
            index: 0,
            value: angular_slots,
        });
    }
    let stride_i32 = angular_slots
        .checked_mul(3)
        .ok_or(XsphError::IntegerOutOfRange {
            name,
            value: angular_slots,
        })?;
    usize::try_from(stride_i32).map_err(|_| XsphError::IntegerOutOfRange {
        name,
        value: stride_i32,
    })
}

fn validate_tdlda_energy_rows_input(input: &XsphTdldaEnergyRowsInput<'_>) -> Result<(), XsphError> {
    validate_active_len(
        "tdlda_energy_rows_energy",
        input.energy_hartree.len(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_energy_rows_reference",
        input.reference_energy.len(),
        input.energy_count,
    )?;
    validate_finite_real("tdlda_energy_rows_edge", input.edge_energy)?;
    validate_finite_real(
        "tdlda_energy_rows_chemical_potential",
        input.chemical_potential,
    )?;
    validate_finite_real("tdlda_energy_rows_spin_orbit_split", input.spin_orbit_split)?;
    for energy in 0..input.energy_count {
        validate_finite_real("tdlda_energy_rows_energy", input.energy_hartree[energy])?;
        validate_finite_complex(
            "tdlda_energy_rows_reference",
            energy,
            input.reference_energy[energy],
        )?;
    }
    Ok(())
}

fn tdlda_relativistic_wave_number(momentum_squared: Complex) -> Result<Complex, XsphError> {
    validate_finite_complex("tdlda_energy_momentum_squared", 0, momentum_squared)?;
    let alpha_scaled = momentum_squared * super::XSPH_FINE_STRUCTURE_ALPHA;
    let wave_number = (2.0 * momentum_squared + alpha_scaled * alpha_scaled).sqrt();
    validate_finite_complex("tdlda_energy_wave_number", 0, wave_number)?;
    Ok(wave_number)
}

fn validate_tdlda_weighted_response_input(
    input: &XsphTdldaWeightedResponseInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len(
        "tdlda_weighted_response_energy",
        input.raw_imaginary_response.shape()[0],
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_weighted_response_rows",
        input.raw_imaginary_response.shape()[1],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_weighted_response_cols",
        input.raw_imaginary_response.shape()[2],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_weighted_response_kinitm",
        input.initial_kappas.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_weighted_response_kfinm",
        input.final_kappas.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_weighted_response_multiplier_energy",
        input.channel_multipliers.nrows(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_weighted_response_multiplier_channels",
        input.channel_multipliers.ncols(),
        4,
    )?;

    for row in 0..input.matrix_size {
        tdlda_weighted_response_channel(input.initial_kappas[row], input.final_kappas[row])?;
    }
    for energy in 0..input.energy_count {
        for channel in 0..4 {
            validate_finite_real(
                "tdlda_weighted_response_multiplier",
                input.channel_multipliers[(energy, channel)],
            )?;
        }
        for row in 0..input.matrix_size {
            for column in 0..input.matrix_size {
                validate_finite_real(
                    "tdlda_weighted_response_raw",
                    input.raw_imaginary_response[(energy, row, column)],
                )?;
            }
        }
    }

    Ok(())
}

fn tdlda_weighted_response_channel(
    initial_kappa: i32,
    final_kappa: i32,
) -> Result<usize, XsphError> {
    if initial_kappa == 0 || final_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    let initial_shifted = initial_kappa
        .checked_add(1)
        .and_then(i32::checked_abs)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "tdlda_weighted_response_kinitm",
            value: initial_kappa,
        })?;
    let final_shifted = final_kappa
        .checked_add(1)
        .and_then(i32::checked_abs)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "tdlda_weighted_response_kfinm",
            value: final_kappa,
        })?;
    Ok(match (initial_kappa > 0, final_shifted > initial_shifted) {
        (true, true) => 1,
        (true, false) => 3,
        (false, true) => 0,
        (false, false) => 2,
    })
}

fn validate_tdlda_response_conditioning_input(
    input: &XsphTdldaResponseConditioningInput<'_>,
) -> Result<(), XsphError> {
    validate_tdlda_kk_common(
        input.energy_count,
        input.matrix_size,
        input.energy_hartree,
        input.chemical_potential,
        input.edge_energy,
        input.reference_shifts,
        input.imaginary_response,
    )?;
    validate_active_len(
        "tdlda_response_conditioning_broadening",
        input.row_broadenings.len(),
        input.matrix_size,
    )?;
    for row in 0..input.matrix_size {
        validate_tdlda_channel_broadening_width(
            "tdlda_response_conditioning_broadening",
            input.row_broadenings[row],
        )?;
    }
    Ok(())
}

fn validate_tdlda_kk_input(input: &XsphTdldaKramersKronigInput<'_>) -> Result<(), XsphError> {
    validate_tdlda_kk_common(
        input.energy_count,
        input.matrix_size,
        input.energy_hartree,
        input.chemical_potential,
        input.edge_energy,
        input.reference_shifts,
        input.imaginary_response,
    )
}

fn validate_tdlda_kk_common(
    energy_count: usize,
    matrix_size: usize,
    energy_hartree: ArrayView1<'_, Real>,
    chemical_potential: Real,
    edge_energy: Real,
    reference_shifts: ArrayView1<'_, Real>,
    imaginary_response: ArrayView3<'_, Real>,
) -> Result<(), XsphError> {
    validate_active_len("tdlda_kk_energy", energy_hartree.len(), energy_count)?;
    validate_active_len("tdlda_kk_matrix", matrix_size, matrix_size)?;
    validate_active_len(
        "tdlda_kk_reference_shifts",
        reference_shifts.len(),
        matrix_size,
    )?;
    validate_active_len(
        "tdlda_kk_imaginary_response_energy",
        imaginary_response.shape()[0],
        energy_count,
    )?;
    validate_active_len(
        "tdlda_kk_imaginary_response_rows",
        imaginary_response.shape()[1],
        matrix_size,
    )?;
    validate_active_len(
        "tdlda_kk_imaginary_response_cols",
        imaginary_response.shape()[2],
        matrix_size,
    )?;
    if energy_count < 2 {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_kk_energy_count",
            value: energy_count,
        });
    }
    validate_finite_real("tdlda_kk_chemical_potential", chemical_potential)?;
    validate_finite_real("tdlda_kk_edge", edge_energy)?;

    for energy in 0..energy_count {
        validate_finite_real("tdlda_kk_energy", energy_hartree[energy])?;
        if energy > 0 {
            let step = energy_hartree[energy] - energy_hartree[energy - 1];
            if !step.is_finite() || step <= 0.0 {
                return Err(XsphError::InvalidPositiveScalar {
                    name: "tdlda_kk_energy_step",
                    value: step,
                });
            }
        }
        for row in 0..matrix_size {
            for column in 0..matrix_size {
                validate_finite_real(
                    "tdlda_kk_imaginary_response",
                    imaginary_response[(energy, row, column)],
                )?;
            }
        }
    }
    for row in 0..matrix_size {
        validate_finite_real("tdlda_kk_reference_shift", reference_shifts[row])?;
    }

    Ok(())
}

fn tdlda_kk_fake_grid(energies: &[Real], target_index: usize) -> Result<Vec<Real>, XsphError> {
    let left = energies[0];
    let right = energies[energies.len() - 1] + TDLDA_KK_RIGHT_PADDING_EV / super::XSPH_HARTREE_EV;
    let step = (right - left) / (TDLDA_KK_FAKE_GRID_COUNT as Real - 1.0);
    if !step.is_finite() || step <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "tdlda_kk_fake_grid_step",
            value: step,
        });
    }
    let target = energies[target_index];
    let interval = ((target - left) / step).trunc();
    let delta = step / 2.0 - ((target - left) - step * interval);
    validate_finite_real("tdlda_kk_fake_grid_delta", delta)?;
    let mut first = left - delta;
    if delta > 0.0 {
        first += step;
    }
    validate_finite_real("tdlda_kk_fake_grid_first", first)?;

    let mut grid = Vec::with_capacity(TDLDA_KK_FAKE_GRID_COUNT);
    for index in 0..TDLDA_KK_FAKE_GRID_COUNT {
        let value = first + step * index as Real;
        validate_finite_real("tdlda_kk_fake_grid_energy", value)?;
        grid.push(value);
    }
    Ok(grid)
}

fn tdlda_kk_interpolate_fake_values(
    energies: &[Real],
    imaginary_response: ArrayView3<'_, Real>,
    energy_count: usize,
    row: usize,
    column: usize,
    fake_grid: &[Real],
) -> Result<Vec<Real>, XsphError> {
    let mut values = Vec::with_capacity(fake_grid.len());
    let mut interval = 0_usize;
    for &energy in fake_grid {
        while interval + 1 < energy_count - 1 {
            let del1 = energy - energies[interval];
            let del2 = energy - energies[interval + 1];
            if del1 == 0.0 || del1 * del2 < 0.0 {
                break;
            }
            interval += 1;
        }
        let value = if energy == energies[interval] {
            imaginary_response[(interval, row, column)]
        } else if interval + 1 < energy_count {
            let left = energies[interval];
            let right = energies[interval + 1];
            let left_value = imaginary_response[(interval, row, column)];
            let right_value = imaginary_response[(interval + 1, row, column)];
            (left_value * (right - energy) + right_value * (energy - left)) / (right - left)
        } else {
            imaginary_response[(energy_count - 1, row, column)]
        };
        validate_finite_real("tdlda_kk_fake_imaginary_response", value)?;
        values.push(value);
    }
    Ok(values)
}

fn tdlda_kk_integral(
    target: Real,
    chemical_potential: Real,
    cutoff: Real,
    fake_grid: &[Real],
    fake_values: &[Real],
) -> Result<Real, XsphError> {
    let mut integral = 0.0;
    for index in 0..fake_grid.len() - 1 {
        let e1 = fake_grid[index];
        let e2 = fake_grid[index + 1];
        if e1 < cutoff {
            continue;
        }

        let mut panel = if e2 > target && e1 < target {
            let a1 = tdlda_kk_second_pole_scaled_value(
                fake_values[index] * (e2 - target),
                target,
                chemical_potential,
                e2,
            )?;
            let a2 = tdlda_kk_second_pole_scaled_value(
                fake_values[index + 1] * (target - e1),
                target,
                chemical_potential,
                e1,
            )?;
            let interpolated = (a1 + a2) / (e2 - e1);
            let ratio = (target - e1) / (e2 - target);
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(XsphError::InvalidPositiveScalar {
                    name: "tdlda_kk_log_argument",
                    value: ratio,
                });
            }
            -interpolated * ratio.ln() + (fake_values[index + 1] - fake_values[index])
        } else {
            let a1 = tdlda_kk_second_pole_scaled_value(
                fake_values[index + 1] / (e2 - target),
                target,
                chemical_potential,
                e2,
            )?;
            let a2 = tdlda_kk_second_pole_scaled_value(
                fake_values[index] / (e1 - target),
                target,
                chemical_potential,
                e1,
            )?;
            0.5 * (e2 - e1) * (a2 + a1)
        };
        panel /= std::f64::consts::PI;
        validate_finite_real("tdlda_kk_panel", panel)?;
        integral += panel;
    }
    validate_finite_real("tdlda_kk_real_response", integral)?;
    Ok(integral)
}

fn tdlda_kk_second_pole_scaled_value(
    value: Real,
    target: Real,
    chemical_potential: Real,
    energy: Real,
) -> Result<Real, XsphError> {
    let denominator = 2.0 * chemical_potential + target + energy;
    if !denominator.is_finite() || denominator == 0.0 {
        return Err(XsphError::ZeroComplexResult {
            name: "tdlda_kk_second_pole_denominator",
        });
    }
    let scaled = value * 2.0 * (chemical_potential + energy) / denominator;
    validate_finite_real("tdlda_kk_second_pole", scaled)?;
    Ok(scaled)
}

fn validate_tdlda_channel_spectra_input<'a>(
    input: &XsphTdldaChannelSpectraInput<'a>,
) -> Result<&'static [usize], XsphError> {
    validate_active_len(
        "tdlda_channel_spectra_energy",
        input.energy_count,
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_matrix",
        input.matrix_size,
        input.matrix_size,
    )?;
    if input.primary_channel_count == 0 || input.primary_channel_count > input.matrix_size {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_channel_spectra_primary_channel_count",
            value: input.primary_channel_count,
        });
    }
    let active_channels = tdlda_active_channel_indices(input.channel_count)?;
    validate_active_len(
        "tdlda_channel_spectra_omega",
        input.photon_energy.len(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_plus_wave_number",
        input.plus_wave_number.len(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_minus_wave_number",
        input.minus_wave_number.len(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_kinitm",
        input.initial_kappas.len(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_dipole_energy",
        input.dipole_matrix.nrows(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_dipole_cols",
        input.dipole_matrix.ncols(),
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_response_energy",
        input.response.shape()[0],
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_response_rows",
        input.response.shape()[1],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_response_cols",
        input.response.shape()[2],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_kernel_energy",
        input.projected_kernel.shape()[0],
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_kernel_rows",
        input.projected_kernel.shape()[1],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_kernel_cols",
        input.projected_kernel.shape()[2],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_dipscf_energy",
        input.screened_dipoles.nrows(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_channel_spectra_dipscf_cols",
        input.screened_dipoles.ncols(),
        input.matrix_size,
    )?;

    for energy in 0..input.energy_count {
        if !input.photon_energy[energy].is_finite() || input.photon_energy[energy] <= 0.0 {
            return Err(XsphError::InvalidPositiveScalar {
                name: "tdlda_channel_spectra_omega",
                value: input.photon_energy[energy],
            });
        }
        validate_finite_real(
            "tdlda_channel_spectra_plus_wave_number",
            input.plus_wave_number[energy],
        )?;
        validate_finite_real(
            "tdlda_channel_spectra_minus_wave_number",
            input.minus_wave_number[energy],
        )?;
        for row in 0..input.matrix_size {
            validate_finite_real(
                "tdlda_channel_spectra_dipole",
                input.dipole_matrix[(energy, row)],
            )?;
            validate_finite_complex(
                "tdlda_channel_spectra_dipscf",
                energy * input.matrix_size + row,
                input.screened_dipoles[(energy, row)],
            )?;
            for column in 0..input.matrix_size {
                validate_finite_complex(
                    "tdlda_channel_spectra_kernel",
                    energy * input.matrix_size * input.matrix_size
                        + row * input.matrix_size
                        + column,
                    input.projected_kernel[(energy, row, column)],
                )?;
                validate_finite_complex(
                    "tdlda_channel_spectra_response",
                    energy * input.matrix_size * input.matrix_size
                        + row * input.matrix_size
                        + column,
                    input.response[(energy, row, column)],
                )?;
            }
        }
    }

    Ok(active_channels)
}

fn tdlda_active_channel_indices(channel_count: usize) -> Result<&'static [usize], XsphError> {
    match channel_count {
        1 => Ok(&[0][..]),
        2 => Ok(&[0, 1][..]),
        4 => Ok(&[0, 1, 2, 3][..]),
        value => Err(XsphError::SizeOutOfRange {
            name: "tdlda_channel_count",
            value,
        }),
    }
}

fn tdlda_channel_prefactor(photon_energy: Real) -> Result<Real, XsphError> {
    if !photon_energy.is_finite() || photon_energy <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar {
            name: "tdlda_channel_spectra_omega",
            value: photon_energy,
        });
    }
    let prefactor = -4.0 * std::f64::consts::PI / super::XSPH_FINE_STRUCTURE_ALPHA / photon_energy
        * super::XSPH_BOHR_ANGSTROM.powi(2)
        * 100.0;
    validate_finite_real("tdlda_channel_prefactor", prefactor)?;
    Ok(prefactor)
}

fn tdlda_channel_index(initial_kappa: i32, row: usize, primary_channel_count: usize) -> usize {
    match (initial_kappa < 0, row < primary_channel_count) {
        (true, true) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (false, false) => 3,
    }
}

fn validate_tdlda_channel_broadening_input<'a>(
    input: &XsphTdldaChannelBroadeningInput<'a>,
) -> Result<&'static [usize], XsphError> {
    validate_active_len(
        "tdlda_channel_broadening_energy",
        input.energy_hartree.len(),
        input.energy_count,
    )?;
    let active_channels = tdlda_active_channel_indices(input.channel_count)?;
    for (name, rows, columns) in [
        (
            "tdlda_channel_broadening_single_particle_channels",
            input.single_particle_channels.nrows(),
            input.single_particle_channels.ncols(),
        ),
        (
            "tdlda_channel_broadening_screened_channels",
            input.screened_channels.nrows(),
            input.screened_channels.ncols(),
        ),
    ] {
        validate_active_len(name, rows, input.energy_count)?;
        validate_active_len(name, columns, 4)?;
    }

    validate_finite_real("tdlda_channel_broadening_edge", input.edge_energy)?;
    validate_finite_real(
        "tdlda_channel_broadening_spin_orbit_split",
        input.spin_orbit_split,
    )?;
    validate_tdlda_channel_broadening_width(
        "tdlda_channel_broadening_plus_width",
        input.plus_broadening,
    )?;
    validate_tdlda_channel_broadening_width(
        "tdlda_channel_broadening_minus_width",
        input.minus_broadening,
    )?;

    for energy in 0..input.energy_count {
        validate_finite_real(
            "tdlda_channel_broadening_energy",
            input.energy_hartree[energy],
        )?;
        for &channel in active_channels {
            validate_finite_real(
                "tdlda_channel_broadening_single_particle_channel",
                input.single_particle_channels[(energy, channel)],
            )?;
            validate_finite_real(
                "tdlda_channel_broadening_screened_channel",
                input.screened_channels[(energy, channel)],
            )?;
        }
    }

    Ok(active_channels)
}

fn validate_tdlda_channel_broadening_width(
    name: &'static str,
    width: Real,
) -> Result<(), XsphError> {
    if !width.is_finite() || width <= 0.0 {
        return Err(XsphError::InvalidPositiveScalar { name, value: width });
    }
    Ok(())
}

fn tdlda_channel_broadening_threshold(
    channel: usize,
    edge_energy: Real,
    spin_orbit_split: Real,
) -> Result<Real, XsphError> {
    let threshold = if matches!(channel, 0 | 2) {
        edge_energy
    } else {
        edge_energy + spin_orbit_split
    };
    validate_finite_real("tdlda_channel_broadening_threshold", threshold)?;
    Ok(threshold)
}

fn tdlda_channel_broadening_width(
    channel: usize,
    plus_broadening: Real,
    minus_broadening: Real,
) -> Real {
    if matches!(channel, 0 | 2) {
        plus_broadening
    } else {
        minus_broadening
    }
}

fn tdlda_threshold_channel(
    energies: &[Real],
    channels: ArrayView2<'_, Real>,
    energy_count: usize,
    channel: usize,
    threshold: Real,
) -> Result<Vec<Complex>, XsphError> {
    let mut spectrum = Vec::with_capacity(energy_count);
    for energy in 0..energy_count {
        let value = if energies[energy] < threshold {
            0.0
        } else {
            channels[(energy, channel)]
        };
        validate_finite_real("tdlda_channel_broadening_thresholded_channel", value)?;
        spectrum.push(Complex::new(value, 0.0));
    }
    Ok(spectrum)
}

fn validate_tdlda_xsedge_input<'a>(
    input: &XsphTdldaXsedgeRowsInput<'a>,
) -> Result<&'static [usize], XsphError> {
    validate_active_len(
        "tdlda_xsedge_energy",
        input.energy_hartree.len(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_xsedge_single_particle_energy",
        input.single_particle_channels.nrows(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_xsedge_screened_energy",
        input.screened_channels.nrows(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_xsedge_multiplier_energy",
        input.channel_multipliers.nrows(),
        input.energy_count,
    )?;
    let active_channels = tdlda_active_channel_indices(input.channel_count).map_err(|_| {
        XsphError::SizeOutOfRange {
            name: "tdlda_xsedge_channel_count",
            value: input.channel_count,
        }
    })?;
    for (name, columns) in [
        (
            "tdlda_xsedge_single_particle_channels",
            input.single_particle_channels.ncols(),
        ),
        (
            "tdlda_xsedge_screened_channels",
            input.screened_channels.ncols(),
        ),
        (
            "tdlda_xsedge_channel_multipliers",
            input.channel_multipliers.ncols(),
        ),
    ] {
        validate_active_len(name, columns, 4)?;
    }

    for energy in 0..input.energy_count {
        validate_finite_real("tdlda_xsedge_energy", input.energy_hartree[energy])?;
        for &channel in active_channels {
            validate_finite_real(
                "tdlda_xsedge_single_particle_channel",
                input.single_particle_channels[(energy, channel)],
            )?;
            validate_finite_real(
                "tdlda_xsedge_screened_channel",
                input.screened_channels[(energy, channel)],
            )?;
            validate_finite_real(
                "tdlda_xsedge_channel_multiplier",
                input.channel_multipliers[(energy, channel)],
            )?;
        }
    }

    Ok(active_channels)
}

fn validate_tdlda_screened_dipole_input(
    input: &XsphTdldaScreenedDipoleInput<'_>,
) -> Result<(), XsphError> {
    validate_active_len(
        "tdlda_dmscf_energy_count",
        input.energy_count,
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_dmscf_matrix_size",
        input.matrix_size,
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_dmscf_response_energy",
        input.response.shape()[0],
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_dmscf_kernel_energy",
        input.kernel.shape()[0],
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_dmscf_dipole_energy",
        input.dipole_matrix.nrows(),
        input.energy_count,
    )?;
    validate_active_len(
        "tdlda_dmscf_response_rows",
        input.response.shape()[1],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_dmscf_response_cols",
        input.response.shape()[2],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_dmscf_kernel_rows",
        input.kernel.shape()[1],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_dmscf_kernel_cols",
        input.kernel.shape()[2],
        input.matrix_size,
    )?;
    validate_active_len(
        "tdlda_dmscf_dipole_cols",
        input.dipole_matrix.ncols(),
        input.matrix_size,
    )?;

    for energy in 0..input.energy_count {
        for row in 0..input.matrix_size {
            validate_finite_real("tdlda_dmscf_dipole", input.dipole_matrix[(energy, row)])?;
            for column in 0..input.matrix_size {
                validate_finite_complex(
                    "tdlda_dmscf_response",
                    energy * input.matrix_size * input.matrix_size
                        + row * input.matrix_size
                        + column,
                    input.response[(energy, row, column)],
                )?;
                validate_finite_complex(
                    "tdlda_dmscf_kernel",
                    energy * input.matrix_size * input.matrix_size
                        + row * input.matrix_size
                        + column,
                    input.kernel[(energy, row, column)],
                )?;
            }
        }
    }
    Ok(())
}

fn complex_to_complex32(value: Complex) -> Complex32 {
    Complex32::new(value.re as f32, value.im as f32)
}

fn complex32_to_complex(value: Complex32) -> Complex {
    Complex::new(f64::from(value.re), f64::from(value.im))
}

/// Port of FEFF `XSPH/specupdlg.f90`.
///
/// Updates the NRIXS angular-decomposition spectrum buckets `xseclg(0:ljmax)`
/// in place for one shared calculation and spin component. The transition
/// weights use compact magnetic columns: `mjinit = -jinit, -jinit+2, ..., jinit`
/// maps to `(mjinit + jinit) / 2`.
pub fn xsph_update_nrixs_lg_spectrum(
    input: XsphLgSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
) -> Result<(), XsphError> {
    if input.calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex {
            calculation_index: input.calculation_index,
        });
    }
    if input.spin_index > 1 {
        return Err(XsphError::InvalidSpinIndex {
            spin_index: input.spin_index,
        });
    }
    if input.initial_j2 < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "initial_j2",
            index: 0,
            value: input.initial_j2,
        });
    }
    validate_cwig3j_doubled_argument("initial_j2", input.initial_j2, input.initial_j2)?;
    validate_active_len("index_map", input.index_map.len(), input.active_len)?;
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    validate_active_len("final_lj", input.final_lj.len(), input.active_len)?;
    let channel_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "radial_integrals",
        input.radial_integrals.len(),
        channel_count,
    )?;
    validate_active_len("spectrum", spectrum.len(), channel_count)?;

    let magnetic_count = usize::try_from(input.initial_j2)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?;
    let required_weights = [2, input.active_len, magnetic_count];
    let weight_shape = input.transition_weights.shape();
    let actual_weights = [weight_shape[0], weight_shape[1], weight_shape[2]];
    if actual_weights
        .iter()
        .zip(required_weights.iter())
        .any(|(actual, required)| actual < required)
    {
        return Err(XsphError::ShapeTooSmall {
            name: "transition_weights",
            required: required_weights,
            actual: actual_weights,
        });
    }

    let q_count = input.q_weights.len();
    validate_q_inputs(input.q_weights, input.q_cosines, q_count)?;
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_count = channel_count;
    let mut legendre_by_pair = vec![0.0; q_count * q_count * legendre_count];
    for iq in 0..q_count {
        for iqq in 0..q_count {
            let cosine = input.q_cosines[(iq, iqq)];
            validate_finite_real("q_cosines", cosine)?;
            let offset = (iq * q_count + iqq) * legendre_count;
            legendre_polynomials_into(
                cosine,
                &mut legendre_by_pair[offset..offset + legendre_count],
            );
        }
    }

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_indexed_angular_momentum("final_lj", index, input.final_lj[index])?;
        let final_lj = usize::try_from(final_lj).map_err(|_| XsphError::IntegerOutOfRange {
            name: "final_lj",
            value: input.final_lj[index],
        })?;
        if final_lj > input.ljmax {
            return Err(XsphError::AngularMomentumOutOfRange {
                angular_momentum: final_lj,
                ljmax: input.ljmax,
            });
        }
        let orbital_l =
            validate_indexed_angular_momentum("orbital_l", index, input.orbital_l[index])?;
        let orbital_l = usize::try_from(orbital_l).map_err(|_| XsphError::IntegerOutOfRange {
            name: "orbital_l",
            value: input.orbital_l[index],
        })?;
        if orbital_l > input.ljmax {
            continue;
        }

        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;
        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * legendre_count + final_lj];
            let radial = input.radial_integrals[final_lj];
            let amplitude = match input.mode {
                XsphSpectrumUpdateMode::Regular => {
                    -Complex::new(0.0, 1.0) * radial * radial * legendre
                }
                XsphSpectrumUpdateMode::Irregular => radial * legendre,
            };
            spectrum[orbital_l] -= amplitude * trace * q_weights[iq] * q_weights[iqq];
        }
    }

    Ok(())
}

/// Port of FEFF `XSPH/specupd.f90`.
///
/// Updates the NRIXS spectrum buckets `xsec(0:ljmax)` in place for one shared
/// calculation and spin component, accumulating regular-branch normalization in
/// `spectrum_norm`. FEFF assigns the complex q-weight product into a real
/// accumulator; this port preserves that behavior by using the real part.
pub fn xsph_update_nrixs_lj_spectrum(
    input: XsphLjSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
    spectrum_norm: &mut Real,
) -> Result<(), XsphError> {
    validate_finite_real("spectrum_norm", *spectrum_norm)?;
    let channel_count = validate_lj_spectrum_update_input(input)?;
    validate_active_len("spectrum", spectrum.len(), channel_count)?;
    let q_count = input.q_weights.len();
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_by_pair = xsph_legendre_by_q_pair(input.q_cosines, q_count, channel_count)?;

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_lj_update_channel(input, index)?;
        let radial = input.radial_integrals[final_lj];
        validate_finite_complex("radial_integrals", final_lj, radial)?;
        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;

        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * channel_count + final_lj];
            let amplitude = xsph_spectrum_amplitude(radial, legendre, input.mode);
            let q_product = q_weights[iq] * q_weights[iqq];
            spectrum[final_lj] -= amplitude * trace * q_product;
            if input.mode == XsphSpectrumUpdateMode::Regular {
                *spectrum_norm += xsph_regular_norm_increment(radial, final_lj, q_product);
            }
        }
    }

    Ok(())
}

/// Port of FEFF `XSPH/specupdatom.f90`.
///
/// Updates per-final-state NRIXS spectrum slots `xsec(1:kfinmax)` in place for
/// one shared calculation and spin component, accumulating the same
/// regular-branch normalization used by [`xsph_update_nrixs_lj_spectrum`].
pub fn xsph_update_nrixs_atom_spectrum(
    input: XsphLjSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
    spectrum_norm: &mut Real,
) -> Result<(), XsphError> {
    validate_finite_real("spectrum_norm", *spectrum_norm)?;
    let channel_count = validate_lj_spectrum_update_input(input)?;
    validate_active_len("spectrum", spectrum.len(), input.active_len)?;
    let q_count = input.q_weights.len();
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_by_pair = xsph_legendre_by_q_pair(input.q_cosines, q_count, channel_count)?;

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_lj_update_channel(input, index)?;
        let radial = input.radial_integrals[final_lj];
        validate_finite_complex("radial_integrals", final_lj, radial)?;
        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;

        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * channel_count + final_lj];
            let amplitude = xsph_spectrum_amplitude(radial, legendre, input.mode);
            let q_product = q_weights[iq] * q_weights[iqq];
            spectrum[index] -= amplitude * trace * q_product;
            if input.mode == XsphSpectrumUpdateMode::Regular {
                *spectrum_norm += xsph_regular_norm_increment(radial, final_lj, q_product);
            }
        }
    }

    Ok(())
}

fn validate_q_inputs(
    q_weights: ArrayView1<'_, Complex>,
    q_cosines: ArrayView2<'_, Real>,
    q_count: usize,
) -> Result<(), XsphError> {
    let shape = q_cosines.shape();
    let actual = [shape[0], shape[1]];
    let required = [q_count, q_count];
    if actual[0] < required[0] || actual[1] < required[1] {
        return Err(XsphError::MatrixTooSmall {
            name: "q_cosines",
            required,
            actual,
        });
    }
    for (index, &weight) in q_weights.iter().enumerate() {
        validate_finite_complex("q_weights", index, weight)?;
    }
    Ok(())
}

fn validate_xsect_spin_merge_input(
    input: &XsphXsectSpinMergeInput<'_>,
) -> Result<usize, XsphError> {
    validate_active_len("xsect_q_count", input.q_count, 1)?;
    validate_active_len("xsect_transition_count", input.transition_count, 1)?;

    let rkk_shape = input.reduced_matrix_elements.shape();
    let spin_count = rkk_shape[2];
    let required_rkk = [input.q_count, input.transition_count, 1];
    let actual_rkk = [rkk_shape[0], rkk_shape[1], rkk_shape[2]];
    if actual_rkk
        .iter()
        .zip(required_rkk.iter())
        .any(|(actual, required)| actual < required)
    {
        return Err(XsphError::ShapeTooSmall {
            name: "xsect_spin_merge_rkk",
            required: required_rkk,
            actual: actual_rkk,
        });
    }

    let required_spin_count = if input.spin_polarized { spin_count } else { 1 };
    validate_active_len(
        "xsect_spin_merge_norms",
        input.spectrum_norms.len(),
        required_spin_count,
    )?;
    validate_active_len(
        "xsect_spin_merge_cross_sections",
        input.cross_sections.len(),
        required_spin_count,
    )?;
    if input.spin_polarized && spin_count < 2 {
        return Err(XsphError::LengthTooShort {
            name: "xsect_spin_merge_spin_channels",
            required: 2,
            actual: spin_count,
        });
    }

    for spin in 0..required_spin_count {
        validate_finite_real("xsect_spin_merge_norms", input.spectrum_norms[spin])?;
        validate_finite_complex(
            "xsect_spin_merge_cross_sections",
            spin,
            input.cross_sections[spin],
        )?;
    }

    Ok(spin_count)
}

fn validate_lj_spectrum_update_input(
    input: XsphLjSpectrumUpdateInput<'_>,
) -> Result<usize, XsphError> {
    if input.calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex {
            calculation_index: input.calculation_index,
        });
    }
    if input.spin_index > 1 {
        return Err(XsphError::InvalidSpinIndex {
            spin_index: input.spin_index,
        });
    }
    if input.initial_j2 < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "initial_j2",
            index: 0,
            value: input.initial_j2,
        });
    }
    validate_cwig3j_doubled_argument("initial_j2", input.initial_j2, input.initial_j2)?;
    validate_active_len("index_map", input.index_map.len(), input.active_len)?;
    validate_active_len("final_lj", input.final_lj.len(), input.active_len)?;

    let channel_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "radial_integrals",
        input.radial_integrals.len(),
        channel_count,
    )?;

    let magnetic_count = usize::try_from(input.initial_j2)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?;
    let required_weights = [2, input.active_len, magnetic_count];
    let weight_shape = input.transition_weights.shape();
    let actual_weights = [weight_shape[0], weight_shape[1], weight_shape[2]];
    if actual_weights
        .iter()
        .zip(required_weights.iter())
        .any(|(actual, required)| actual < required)
    {
        return Err(XsphError::ShapeTooSmall {
            name: "transition_weights",
            required: required_weights,
            actual: actual_weights,
        });
    }

    validate_q_inputs(input.q_weights, input.q_cosines, input.q_weights.len())?;
    Ok(channel_count)
}

fn validate_lj_update_channel(
    input: XsphLjSpectrumUpdateInput<'_>,
    index: usize,
) -> Result<usize, XsphError> {
    let final_lj = validate_indexed_angular_momentum("final_lj", index, input.final_lj[index])?;
    let final_lj = usize::try_from(final_lj).map_err(|_| XsphError::IntegerOutOfRange {
        name: "final_lj",
        value: input.final_lj[index],
    })?;
    if final_lj > input.ljmax {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: final_lj,
            ljmax: input.ljmax,
        });
    }
    Ok(final_lj)
}

fn xsph_legendre_by_q_pair(
    q_cosines: ArrayView2<'_, Real>,
    q_count: usize,
    legendre_count: usize,
) -> Result<Vec<Real>, XsphError> {
    let mut legendre_by_pair = vec![0.0; q_count * q_count * legendre_count];
    for iq in 0..q_count {
        for iqq in 0..q_count {
            let cosine = q_cosines[(iq, iqq)];
            validate_finite_real("q_cosines", cosine)?;
            let offset = (iq * q_count + iqq) * legendre_count;
            legendre_polynomials_into(
                cosine,
                &mut legendre_by_pair[offset..offset + legendre_count],
            );
        }
    }
    Ok(legendre_by_pair)
}

fn xsph_spectrum_amplitude(
    radial: Complex,
    legendre: Real,
    mode: XsphSpectrumUpdateMode,
) -> Complex {
    match mode {
        XsphSpectrumUpdateMode::Regular => -Complex::new(0.0, 1.0) * radial * radial * legendre,
        XsphSpectrumUpdateMode::Irregular => radial * legendre,
    }
}

fn xsph_regular_norm_increment(radial: Complex, final_lj: usize, q_product: Complex) -> Real {
    let denominator = (2 * final_lj + 1) as Real;
    radial.norm_sqr() / denominator * q_product.re
}

fn xsph_effective_q_weights(
    q_weights: ArrayView1<'_, Complex>,
    mix_dff: bool,
) -> Result<Vec<Complex>, XsphError> {
    q_weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| {
            let effective = if mix_dff { weight } else { weight.sqrt() };
            validate_finite_complex("effective_q_weight", index, effective)?;
            Ok(effective)
        })
        .collect()
}

fn xsph_q_pairs(
    mix_dff: bool,
    mdff_mode: i32,
    q_count: usize,
) -> Result<Vec<(usize, usize)>, XsphError> {
    if !mix_dff {
        return Ok((0..q_count).map(|index| (index, index)).collect());
    }
    match mdff_mode {
        1 => Ok((0..q_count)
            .flat_map(|iq| (0..q_count).map(move |iqq| (iq, iqq)))
            .collect()),
        2 if q_count >= 2 => Ok(vec![(0, 1)]),
        2 => Err(XsphError::MatrixTooSmall {
            name: "q_weights",
            required: [2, 1],
            actual: [q_count, 1],
        }),
        _ => Err(XsphError::InvalidMdffMode { mdff_mode }),
    }
}

fn xsph_transition_trace(
    transition_weights: ArrayView3<'_, Real>,
    spin_index: usize,
    state_index: usize,
    initial_j2: i32,
) -> Result<Real, XsphError> {
    let mut trace = 0.0;
    let mut magnetic_j2 = -initial_j2;
    while magnetic_j2 <= initial_j2 {
        let magnetic_index = usize::try_from((magnetic_j2 + initial_j2) / 2).map_err(|_| {
            XsphError::IntegerOutOfRange {
                name: "initial_j2",
                value: initial_j2,
            }
        })?;
        let value = transition_weights[(spin_index, state_index, magnetic_index)];
        validate_finite_real("transition_weights", value)?;
        trace += value * value;
        magnetic_j2 += 2;
    }
    Ok(trace)
}
