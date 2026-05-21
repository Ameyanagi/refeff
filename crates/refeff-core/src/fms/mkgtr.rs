//! MKGTR Green's-function trace folding for FMS outputs.

use ndarray::{Array2, ArrayView3, ShapeBuilder};
use num_complex::Complex32;

use crate::{Complex, Real, angular::TransitionBMatrix};

use super::{FmsError, ensure_axis_len, ensure_spin_channels};

/// Inputs for FEFF `MKGTR/getgtr.f90` Green's-function trace folding.
#[derive(Debug, Clone)]
pub struct MkgtrGreenTraceInput<'a> {
    /// Active spin channels used by `getgtr` after FEFF's `ispin` selection.
    pub active_spin_channels: usize,
    /// `gg(energy, channel1, channel2)` FMS Green's-function matrices for
    /// absorber potential `iph=0`.
    pub green_functions: ArrayView3<'a, Complex32>,
    /// Transition B matrices for the spectra selected by `ipmin:ipstep:ipmax`.
    pub transition_matrices: &'a [TransitionBMatrix],
    /// FEFF transition moments `rkk(energy, transition, spin)`.
    pub transition_moments: ArrayView3<'a, Complex>,
}

/// FEFF MKGTR folded FMS trace spectra.
#[derive(Debug, Clone, PartialEq)]
pub struct MkgtrGreenTraceResult {
    /// `gtr(spectrum, energy)` values ready for `fms.bin` or `gtr.dat`.
    pub traces: Array2<Complex>,
}

/// Fold FEFF FMS Green's-function matrices into MKGTR trace spectra.
///
/// This ports the non-NRIXS `Form gtr` loop in `MKGTR/getgtr.f90`. The input
/// Green's functions are the absorber-potential `gg` matrices for each energy,
/// while `transition_matrices` corresponds to the per-spectrum `bmat` blocks
/// built by FEFF `bcoef`.
pub fn mkgtr_green_trace(
    input: MkgtrGreenTraceInput<'_>,
) -> Result<MkgtrGreenTraceResult, FmsError> {
    ensure_spin_channels(input.active_spin_channels)?;
    let shape = input.green_functions.shape();
    if shape[0] == 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "energy",
            index: 0,
        });
    }
    if shape[1] == 0 || shape[1] != shape[2] {
        return Err(FmsError::TableIndexOutOfRange {
            table: "gg",
            axis: "shape",
            index: shape[1],
        });
    }
    if input.transition_matrices.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "bmat",
            axis: "spectrum",
            index: 0,
        });
    }
    ensure_axis_len(
        "rkk",
        "energy",
        input.transition_moments.shape()[0],
        shape[0] - 1,
    )?;
    ensure_axis_len("rkk", "transition", input.transition_moments.shape()[1], 7)?;
    if input.transition_moments.shape()[2] < input.active_spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "rkk",
            expected: input.active_spin_channels,
            actual: input.transition_moments.shape()[2],
        });
    }

    for (spectrum, matrix) in input.transition_matrices.iter().enumerate() {
        validate_mkgtr_transition_matrix(spectrum, matrix)?;
        validate_mkgtr_green_channels(
            input.green_functions.shape()[1],
            input.active_spin_channels,
            matrix,
        )?;
    }

    let mut traces = Array2::zeros((input.transition_matrices.len(), shape[0]).f());
    for (spectrum, matrix) in input.transition_matrices.iter().enumerate() {
        for energy in 0..shape[0] {
            traces[(spectrum, energy)] = mkgtr_green_trace_energy(&input, matrix, energy)?;
        }
    }
    Ok(MkgtrGreenTraceResult { traces })
}

fn mkgtr_green_trace_energy(
    input: &MkgtrGreenTraceInput<'_>,
    transition_matrix: &TransitionBMatrix,
    energy: usize,
) -> Result<Complex, FmsError> {
    let mut trace = Complex::new(0.0, 0.0);
    for transition1 in 0..8 {
        let angular1 = transition_matrix.orbital_momenta[transition1];
        if angular1 < 0 {
            continue;
        }
        let angular1 = usize::try_from(angular1).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: 0,
            lx: transition_matrix.l_offset,
        })?;
        for spin1 in 0..input.active_spin_channels {
            let rkk1 = input.transition_moments[(energy, transition1, spin1)];
            validate_finite_complex_value(
                "rkk",
                flat_index3(input.transition_moments.shape(), energy, transition1, spin1),
                rkk1,
            )?;
            for transition2 in 0..8 {
                let angular2 = transition_matrix.orbital_momenta[transition2];
                if angular2 < 0 {
                    continue;
                }
                let angular2 =
                    usize::try_from(angular2).map_err(|_| FmsError::InvalidAngularLimit {
                        name: "lnd",
                        value: 0,
                        lx: transition_matrix.l_offset,
                    })?;
                for spin2 in 0..input.active_spin_channels {
                    let rkk2 = input.transition_moments[(energy, transition2, spin2)];
                    validate_finite_complex_value(
                        "rkk",
                        flat_index3(input.transition_moments.shape(), energy, transition2, spin2),
                        rkk2,
                    )?;
                    for magnetic1 in signed_magnetic_range(angular1)? {
                        let row = mkgtr_channel_index(
                            input.active_spin_channels,
                            angular1,
                            magnetic1,
                            spin1,
                        )?;
                        for magnetic2 in signed_magnetic_range(angular2)? {
                            let column = mkgtr_channel_index(
                                input.active_spin_channels,
                                angular2,
                                magnetic2,
                                spin2,
                            )?;
                            let green = input.green_functions[(energy, row, column)];
                            validate_finite_complex32_value(
                                "gg",
                                flat_index3(input.green_functions.shape(), energy, row, column),
                                green,
                            )?;
                            let bmat = transition_matrix
                                .value(
                                    magnetic2 as isize,
                                    spin2,
                                    transition2 + 1,
                                    magnetic1 as isize,
                                    spin1,
                                    transition1 + 1,
                                )
                                .ok_or(FmsError::TableIndexOutOfRange {
                                    table: "bmat",
                                    axis: "magnetic",
                                    index: transition_matrix.l_offset,
                                })?;
                            validate_finite_complex_value(
                                "bmat",
                                flat_index6(
                                    transition_matrix.matrix.shape(),
                                    [
                                        signed_to_shifted_magnetic(
                                            magnetic2,
                                            transition_matrix.l_offset,
                                        )?,
                                        spin2,
                                        transition2,
                                        signed_to_shifted_magnetic(
                                            magnetic1,
                                            transition_matrix.l_offset,
                                        )?,
                                        spin1,
                                        transition1,
                                    ],
                                ),
                                bmat,
                            )?;
                            trace += widen_complex32(green) * bmat * rkk1 * rkk2;
                        }
                    }
                }
            }
        }
    }
    validate_finite_complex_value("gtr", energy, trace)?;
    Ok(trace)
}

fn validate_mkgtr_transition_matrix(
    _spectrum: usize,
    matrix: &TransitionBMatrix,
) -> Result<(), FmsError> {
    let shape = matrix.matrix.shape();
    ensure_axis_len("bmat", "ml2", shape[0], matrix.l_offset)?;
    ensure_axis_len("bmat", "ms2", shape[1], 1)?;
    ensure_axis_len("bmat", "transition2", shape[2], 7)?;
    ensure_axis_len("bmat", "ml1", shape[3], matrix.l_offset)?;
    ensure_axis_len("bmat", "ms1", shape[4], 1)?;
    ensure_axis_len("bmat", "transition1", shape[5], 7)?;

    for angular in matrix.orbital_momenta {
        if angular < 0 {
            continue;
        }
        let angular = usize::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: 0,
            lx: matrix.l_offset,
        })?;
        if matrix.l_offset < angular {
            return Err(FmsError::TableIndexOutOfRange {
                table: "bmat",
                axis: "magnetic",
                index: angular,
            });
        }
        let high = matrix
            .l_offset
            .checked_add(angular)
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lnd",
                value: angular,
                lx: matrix.l_offset,
            })?;
        ensure_axis_len("bmat", "ml2", shape[0], high)?;
        ensure_axis_len("bmat", "ml1", shape[3], high)?;
    }
    Ok(())
}

fn validate_mkgtr_green_channels(
    channel_count: usize,
    spin_channels: usize,
    matrix: &TransitionBMatrix,
) -> Result<(), FmsError> {
    for angular in matrix.orbital_momenta {
        if angular < 0 {
            continue;
        }
        let angular = usize::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: 0,
            lx: matrix.l_offset,
        })?;
        let magnetic = i32::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
            name: "lnd",
            value: angular,
            lx: matrix.l_offset,
        })?;
        let channel = mkgtr_channel_index(spin_channels, angular, magnetic, spin_channels - 1)?;
        ensure_axis_len("gg", "channel", channel_count, channel)?;
    }
    Ok(())
}

fn mkgtr_channel_index(
    spin_channels: usize,
    angular: usize,
    magnetic: i32,
    spin: usize,
) -> Result<usize, FmsError> {
    let angular_isize = isize::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lnd",
        value: angular,
        lx: angular,
    })?;
    let magnetic_isize = magnetic as isize;
    let orbital = angular_isize
        .checked_mul(angular_isize)
        .and_then(|value| value.checked_add(angular_isize))
        .and_then(|value| value.checked_add(magnetic_isize))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lnd",
            value: angular,
            lx: angular,
        })?;
    let orbital = usize::try_from(orbital).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lnd",
        value: angular,
        lx: angular,
    })?;
    orbital
        .checked_mul(spin_channels)
        .and_then(|value| value.checked_add(spin))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lnd",
            value: angular,
            lx: angular,
        })
}

fn signed_magnetic_range(angular: usize) -> Result<std::ops::RangeInclusive<i32>, FmsError> {
    let angular = i32::try_from(angular).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lnd",
        value: angular,
        lx: angular,
    })?;
    Ok(-angular..=angular)
}

fn signed_to_shifted_magnetic(magnetic: i32, offset: usize) -> Result<usize, FmsError> {
    let offset_i32 = i32::try_from(offset).map_err(|_| FmsError::InvalidAngularLimit {
        name: "bmat",
        value: offset,
        lx: offset,
    })?;
    let shifted = magnetic
        .checked_add(offset_i32)
        .ok_or(FmsError::InvalidAngularLimit {
            name: "bmat",
            value: offset,
            lx: offset,
        })?;
    usize::try_from(shifted).map_err(|_| FmsError::TableIndexOutOfRange {
        table: "bmat",
        axis: "magnetic",
        index: 0,
    })
}

fn validate_finite_complex32_value(
    table: &'static str,
    index: usize,
    value: Complex32,
) -> Result<(), FmsError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FmsError::NonFiniteComplexValue { table, index })
    }
}

fn validate_finite_complex_value(
    table: &'static str,
    index: usize,
    value: Complex,
) -> Result<(), FmsError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FmsError::NonFiniteComplexValue { table, index })
    }
}

fn widen_complex32(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}

fn flat_index3(shape: &[usize], axis0: usize, axis1: usize, axis2: usize) -> usize {
    let dim1 = match shape.get(1) {
        Some(value) => *value,
        None => 0,
    };
    let dim2 = match shape.get(2) {
        Some(value) => *value,
        None => 0,
    };
    axis0
        .saturating_mul(dim1)
        .saturating_add(axis1)
        .saturating_mul(dim2)
        .saturating_add(axis2)
}

fn flat_index6(shape: &[usize], axes: [usize; 6]) -> usize {
    axes.into_iter()
        .enumerate()
        .fold(0usize, |index, (axis, value)| {
            let dimension = match shape.get(axis) {
                Some(value) => *value,
                None => 0,
            };
            index.saturating_mul(dimension).saturating_add(value)
        })
}
