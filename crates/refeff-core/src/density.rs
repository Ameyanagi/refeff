//! FEFF density and LDOS accumulation helpers.
//!
//! This module ports compact numerical routines that update radial valence
//! densities and angular-momentum-resolved density of states after scattering
//! terms have been computed.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex32;
use thiserror::Error;

use crate::{Complex, Real};

/// Inputs for FEFF `POT/ff2g.f90` valence-density accumulation.
#[derive(Debug, Clone, Copy)]
pub struct ValenceDensityUpdateInput<'a> {
    /// Single-precision FMS scattering trace `gtr(0:lx)`.
    pub scattering_trace: ArrayView1<'a, Complex32>,
    /// Zero-based potential column corresponding to FEFF `iph`.
    pub potential_index: usize,
    /// One-based energy index `ie`; `ie == 1` initializes previous-energy work arrays.
    pub energy_index: usize,
    /// One-based last radial point to update, FEFF `ilast`.
    pub last_radial_index: usize,
    /// Scattering contribution to angular-momentum LDOS, `xrhole(0:lx)`.
    pub scattering_ldos: ArrayView1<'a, Complex>,
    /// Embedded-atom LDOS table, indexed as `(l, potential)`, FEFF `xrhoce`.
    pub embedded_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy LDOS table, indexed as `(l, potential)`, FEFF `xrhocp`.
    pub previous_ldos: ArrayView2<'a, Complex>,
    /// Scattering radial density table, indexed as `(radial, l)`, FEFF `yrhole`.
    pub scattering_density: ArrayView2<'a, Complex>,
    /// Embedded radial density for the current potential, FEFF `yrhoce`.
    pub embedded_density: ArrayView1<'a, Complex>,
    /// Previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: ArrayView1<'a, Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: ArrayView1<'a, Real>,
    /// Energy-integrated electron count per angular momentum, FEFF `xnmues`.
    pub occupancy_by_l: ArrayView1<'a, Real>,
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Number of atoms with this potential type, FEFF `xnatph`.
    pub potential_multiplicity: Real,
    /// Current contour-floor flag `iflr`.
    pub current_floor: i32,
    /// Previous contour-floor flag `iflrp`.
    pub previous_floor: i32,
    /// Running left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Running right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Running total valence electron count `xntot`.
    pub total_electron_count: Real,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// Updated FEFF `ff2g` density and LDOS state.
#[derive(Debug, Clone, PartialEq)]
pub struct ValenceDensityUpdate {
    /// Updated embedded-atom LDOS table, FEFF `xrhoce`.
    pub embedded_ldos: Array2<Complex>,
    /// Updated previous-energy LDOS table, FEFF `xrhocp`.
    pub previous_ldos: Array2<Complex>,
    /// Updated embedded radial density, FEFF `yrhoce`.
    pub embedded_density: Array1<Complex>,
    /// Updated previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: Array1<Complex>,
    /// Updated energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: Array1<Real>,
    /// Updated angular-momentum electron counts, FEFF `xnmues`.
    pub occupancy_by_l: Array1<Real>,
    /// Updated left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Updated right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Updated total valence electron count `xntot`.
    pub total_electron_count: Real,
}

/// Error returned by density accumulation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum DensityError {
    /// FEFF indices that select a point count or energy point are 1-based.
    #[error("{name} must be 1-based and positive, got {index}")]
    InvalidIndex { name: &'static str, index: usize },
    /// A vector must contain enough values for the FEFF loop bounds.
    #[error("{name} length {actual} is shorter than required length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// A pair of angular-momentum vectors must have matching lengths.
    #[error("{left_name} length {left_len} does not match {right_name} length {right_len}")]
    LengthMismatch {
        left_name: &'static str,
        left_len: usize,
        right_name: &'static str,
        right_len: usize,
    },
    /// A matrix must have enough rows and columns for the FEFF loop bounds.
    #[error(
        "{name} shape ({rows},{columns}) is smaller than required ({required_rows},{required_columns})"
    )]
    ShapeTooSmall {
        name: &'static str,
        rows: usize,
        columns: usize,
        required_rows: usize,
        required_columns: usize,
    },
    /// A real scalar must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// A complex scalar must have finite components.
    #[error("{name} must be finite, got ({real},{imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// A real vector entry must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// A complex vector or matrix entry must have finite components.
    #[error("{name}[{index}] must be finite, got ({real},{imaginary})")]
    NonFiniteComplexValue {
        name: &'static str,
        index: usize,
        real: Real,
        imaginary: Real,
    },
}

/// Accumulate valence LDOS and radial density from FEFF scattering terms.
///
/// This ports `POT/ff2g.f90`. The routine first folds the single-precision
/// scattering trace into the embedded LDOS, then integrates the current and
/// previous energy endpoints. For `energy_index == 1`, FEFF initializes the
/// previous-energy work arrays from the current values; for later energies it
/// preserves the caller-provided previous state.
pub fn update_valence_density(
    input: ValenceDensityUpdateInput<'_>,
) -> Result<ValenceDensityUpdate, DensityError> {
    validate_valence_density_input(input)?;

    let l_count = input.scattering_trace.len();
    let radial_count = input.last_radial_index;
    let potential = input.potential_index;
    let mut embedded_ldos = input.embedded_ldos.to_owned();
    let mut previous_ldos = input.previous_ldos.to_owned();
    let mut embedded_density = input.embedded_density.to_owned();
    let mut previous_density = input.previous_density.to_owned();
    let mut valence_density = input.valence_density.to_owned();
    let mut occupancy_by_l = input.occupancy_by_l.to_owned();
    let mut left_sum = input.left_sum;
    let mut right_sum = input.right_sum;
    let mut total_electron_count = input.total_electron_count;

    for angular in 0..l_count {
        embedded_ldos[(angular, potential)] +=
            widen_complex32(input.scattering_trace[angular]) * input.scattering_ldos[angular];
        if input.energy_index == 1 {
            previous_ldos[(angular, potential)] = embedded_ldos[(angular, potential)];
        }
    }

    let mut left_step = input.current_energy - input.previous_energy;
    let mut right_step = left_step;
    if input.current_floor == 1 {
        right_step -= Complex::new(0.0, 2.0 * input.current_energy.im);
    }
    if input.previous_floor == 1 {
        left_step += Complex::new(0.0, 2.0 * input.previous_energy.im);
    }

    for angular in 0..l_count {
        if includes_angular_channel(angular, input.include_high_l) {
            left_sum += previous_ldos[(angular, potential)] * (2.0 * input.potential_multiplicity);
            right_sum += embedded_ldos[(angular, potential)] * (2.0 * input.potential_multiplicity);
            occupancy_by_l[angular] += (embedded_ldos[(angular, potential)] * right_step
                + previous_ldos[(angular, potential)] * left_step)
                .im;
            total_electron_count += occupancy_by_l[angular] * input.potential_multiplicity;
        }
    }

    for angular in 0..l_count {
        if includes_angular_channel(angular, input.include_high_l) {
            let trace = widen_complex32(input.scattering_trace[angular]);
            for radial in 0..radial_count {
                embedded_density[radial] += trace * input.scattering_density[(radial, angular)];
                if input.energy_index == 1 {
                    previous_density[radial] = embedded_density[radial];
                }
            }
        }
    }

    for radial in 0..radial_count {
        valence_density[radial] +=
            (embedded_density[radial] * right_step + previous_density[radial] * left_step).im;
    }

    Ok(ValenceDensityUpdate {
        embedded_ldos,
        previous_ldos,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
        left_sum,
        right_sum,
        total_electron_count,
    })
}

fn validate_valence_density_input(
    input: ValenceDensityUpdateInput<'_>,
) -> Result<(), DensityError> {
    if input.energy_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "energy_index",
            index: input.energy_index,
        });
    }
    if input.last_radial_index == 0 {
        return Err(DensityError::InvalidIndex {
            name: "last_radial_index",
            index: input.last_radial_index,
        });
    }

    let l_count = input.scattering_trace.len();
    ensure_length_match(
        "scattering_trace",
        l_count,
        "scattering_ldos",
        input.scattering_ldos.len(),
    )?;
    ensure_len("occupancy_by_l", input.occupancy_by_l.len(), l_count)?;
    ensure_len(
        "embedded_density",
        input.embedded_density.len(),
        input.last_radial_index,
    )?;
    ensure_len(
        "previous_density",
        input.previous_density.len(),
        input.last_radial_index,
    )?;
    ensure_len(
        "valence_density",
        input.valence_density.len(),
        input.last_radial_index,
    )?;
    ensure_shape(
        "embedded_ldos",
        input.embedded_ldos.shape(),
        l_count,
        input.potential_index + 1,
    )?;
    ensure_shape(
        "previous_ldos",
        input.previous_ldos.shape(),
        l_count,
        input.potential_index + 1,
    )?;
    ensure_shape(
        "scattering_density",
        input.scattering_density.shape(),
        input.last_radial_index,
        l_count,
    )?;

    validate_complex32_values("scattering_trace", input.scattering_trace)?;
    validate_complex_values("scattering_ldos", input.scattering_ldos.iter().copied())?;
    validate_complex_values("embedded_ldos", input.embedded_ldos.iter().copied())?;
    validate_complex_values("previous_ldos", input.previous_ldos.iter().copied())?;
    validate_complex_values(
        "scattering_density",
        input.scattering_density.iter().copied(),
    )?;
    validate_complex_values("embedded_density", input.embedded_density.iter().copied())?;
    validate_complex_values("previous_density", input.previous_density.iter().copied())?;
    validate_real_values("valence_density", input.valence_density)?;
    validate_real_values("occupancy_by_l", input.occupancy_by_l)?;
    validate_complex_scalar("current_energy", input.current_energy)?;
    validate_complex_scalar("previous_energy", input.previous_energy)?;
    validate_complex_scalar("left_sum", input.left_sum)?;
    validate_complex_scalar("right_sum", input.right_sum)?;
    validate_real_scalar("potential_multiplicity", input.potential_multiplicity)?;
    validate_real_scalar("total_electron_count", input.total_electron_count)?;

    Ok(())
}

fn includes_angular_channel(angular: usize, include_high_l: bool) -> bool {
    angular <= 2 || include_high_l
}

fn widen_complex32(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}

fn ensure_len(name: &'static str, actual: usize, required: usize) -> Result<(), DensityError> {
    if actual >= required {
        Ok(())
    } else {
        Err(DensityError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}

fn ensure_length_match(
    left_name: &'static str,
    left_len: usize,
    right_name: &'static str,
    right_len: usize,
) -> Result<(), DensityError> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(DensityError::LengthMismatch {
            left_name,
            left_len,
            right_name,
            right_len,
        })
    }
}

fn ensure_shape(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
) -> Result<(), DensityError> {
    let rows = shape[0];
    let columns = shape[1];
    if rows >= required_rows && columns >= required_columns {
        Ok(())
    } else {
        Err(DensityError::ShapeTooSmall {
            name,
            rows,
            columns,
            required_rows,
            required_columns,
        })
    }
}

fn validate_real_scalar(name: &'static str, value: Real) -> Result<(), DensityError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DensityError::NonFiniteScalar { name, value })
    }
}

fn validate_complex_scalar(name: &'static str, value: Complex) -> Result<(), DensityError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(DensityError::NonFiniteComplex {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_real_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DensityError::NonFiniteValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_complex32_values(
    name: &'static str,
    values: ArrayView1<'_, Complex32>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
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

fn validate_complex_values(
    name: &'static str,
    values: impl Iterator<Item = Complex>,
) -> Result<(), DensityError> {
    for (index, value) in values.enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re,
                imaginary: value.im,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn valence_density_update_matches_feff_ff2g_first_energy_reference() -> Result<(), DensityError>
    {
        let sample = sample_ff2g_state();

        let result = update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: sample.scattering_trace.view(),
            potential_index: 1,
            energy_index: 1,
            last_radial_index: 5,
            scattering_ldos: sample.scattering_ldos.view(),
            embedded_ldos: sample.embedded_ldos.view(),
            previous_ldos: sample.previous_ldos.view(),
            scattering_density: sample.scattering_density.view(),
            embedded_density: sample.embedded_density.view(),
            previous_density: sample.previous_density.view(),
            valence_density: sample.valence_density.view(),
            occupancy_by_l: sample.occupancy_by_l.view(),
            current_energy: Complex::new(0.72, 0.11),
            previous_energy: Complex::new(0.61, -0.04),
            potential_multiplicity: 2.5,
            current_floor: 1,
            previous_floor: 0,
            left_sum: Complex::new(0.2, -0.1),
            right_sum: Complex::new(-0.3, 0.25),
            total_electron_count: 1.25,
            include_high_l: false,
        })?;

        assert_complex_close(
            result.embedded_ldos[(0, 1)],
            Complex::new(0.451_099_999_919_533_76, -0.215_299_999_862_909_35),
        );
        assert_complex_close(
            result.embedded_ldos[(2, 1)],
            Complex::new(0.539_700_002_484_023_6, -0.211_100_000_292_062_77),
        );
        assert_complex_close(
            result.embedded_ldos[(3, 1)],
            Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
        );
        assert_complex_close(result.previous_ldos[(2, 1)], result.embedded_ldos[(2, 1)]);
        assert_complex_close(
            result.embedded_density[0],
            Complex::new(6.406_000_025_570_392e-2, -1.129_999_976_605_176_9e-2),
        );
        assert_complex_close(
            result.embedded_density[4],
            Complex::new(2.775_000_003_725_290_3e-1, -9.609_999_980_777_503e-2),
        );
        assert_complex_close(result.previous_density[4], result.embedded_density[4]);
        assert_close(result.valence_density[0], 1.263_880_007_192_492_5e-2);
        assert_close(result.valence_density[3], 4.145_320_007_205_01e-2);
        assert_close(result.valence_density[4], 5.105_800_007_209_182e-2);
        assert_close(result.occupancy_by_l[0], -4.127_799_997_627_735e-2);
        assert_close(result.occupancy_by_l[2], -3.265_999_865_531_922_5e-3);
        assert_close(result.occupancy_by_l[3], 1.5e-2);
        assert_close(result.total_electron_count, 1.082_550_000_302_493_7);
        assert_complex_close(
            result.left_sum,
            Complex::new(7.618_000_007_234_514, -3.296_999_999_880_791),
        );
        assert_complex_close(
            result.right_sum,
            Complex::new(7.118_000_007_234_514, -2.946_999_999_880_791_4),
        );
        Ok(())
    }

    #[test]
    fn valence_density_update_matches_feff_ff2g_high_l_reference() -> Result<(), DensityError> {
        let sample = sample_ff2g_state();

        let result = update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: sample.scattering_trace.view(),
            potential_index: 1,
            energy_index: 2,
            last_radial_index: 4,
            scattering_ldos: sample.scattering_ldos.view(),
            embedded_ldos: sample.embedded_ldos.view(),
            previous_ldos: sample.previous_ldos.view(),
            scattering_density: sample.scattering_density.view(),
            embedded_density: sample.embedded_density.view(),
            previous_density: sample.previous_density.view(),
            valence_density: sample.valence_density.view(),
            occupancy_by_l: sample.occupancy_by_l.view(),
            current_energy: Complex::new(0.91, -0.08),
            previous_energy: Complex::new(0.77, 0.05),
            potential_multiplicity: 1.75,
            current_floor: 0,
            previous_floor: 1,
            left_sum: Complex::new(-0.15, 0.09),
            right_sum: Complex::new(0.05, -0.12),
            total_electron_count: -0.2,
            include_high_l: true,
        })?;

        assert_complex_close(
            result.embedded_ldos[(3, 1)],
            Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
        );
        assert_complex_close(result.previous_ldos[(2, 1)], Complex::new(-4.0e-2, 4.5e-2));
        assert_complex_close(
            result.embedded_density[0],
            Complex::new(8.203_999_945_521_354e-2, -1.959_999_881_684_781e-3),
        );
        assert_complex_close(result.embedded_density[4], Complex::new(2.5e-1, -1.0e-1));
        assert_complex_close(result.previous_density[4], Complex::new(-1.5e-1, 2.0e-1));
        assert_close(result.valence_density[0], 5.560_400_087_386_373e-3);
        assert_close(result.valence_density[3], 2.428_160_011_395_812e-2);
        assert_close(result.valence_density[4], 5.0e-2);
        assert_close(result.occupancy_by_l[0], -1.041_849_999_703_466_9e-1);
        assert_close(result.occupancy_by_l[2], -9.221_500_036_381_186e-2);
        assert_close(result.occupancy_by_l[3], -8.732_799_936_085_94e-2);
        assert_close(result.total_electron_count, -8.677_334_992_048_331e-1);
        assert_complex_close(result.left_sum, Complex::new(-8.85e-1, 8.6e-1));
        assert_complex_close(
            result.right_sum,
            Complex::new(7.313_899_995_405_228, -3.091_499_992_907_047_5),
        );
        Ok(())
    }

    #[test]
    fn valence_density_update_rejects_invalid_inputs() {
        let sample = sample_ff2g_state();
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                energy_index: 0,
                ..sample.input()
            }),
            Err(DensityError::InvalidIndex {
                name: "energy_index",
                index: 0,
            })
        );
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                last_radial_index: 0,
                ..sample.input()
            }),
            Err(DensityError::InvalidIndex {
                name: "last_radial_index",
                index: 0,
            })
        );

        let short_ldos = Array1::<Complex>::zeros(2);
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                scattering_ldos: short_ldos.view(),
                ..sample.input()
            }),
            Err(DensityError::LengthMismatch {
                left_name: "scattering_trace",
                left_len: 4,
                right_name: "scattering_ldos",
                right_len: 2,
            })
        );

        let short_density = Array1::<Complex>::zeros(3);
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                embedded_density: short_density.view(),
                ..sample.input()
            }),
            Err(DensityError::LengthTooShort {
                name: "embedded_density",
                required: 5,
                actual: 3,
            })
        );

        let small_matrix = Array2::<Complex>::zeros((2, 2));
        assert_eq!(
            update_valence_density(ValenceDensityUpdateInput {
                embedded_ldos: small_matrix.view(),
                ..sample.input()
            }),
            Err(DensityError::ShapeTooSmall {
                name: "embedded_ldos",
                rows: 2,
                columns: 2,
                required_rows: 4,
                required_columns: 2,
            })
        );

        let mut bad_trace = sample.scattering_trace.clone();
        bad_trace[1] = Complex32::new(f32::NAN, 0.0);
        assert!(matches!(
            update_valence_density(ValenceDensityUpdateInput {
                scattering_trace: bad_trace.view(),
                ..sample.input()
            }),
            Err(DensityError::NonFiniteComplexValue {
                name: "scattering_trace",
                index: 1,
                ..
            })
        ));
    }

    #[derive(Debug, Clone)]
    struct Ff2gSample {
        scattering_trace: Array1<Complex32>,
        scattering_ldos: Array1<Complex>,
        embedded_ldos: Array2<Complex>,
        previous_ldos: Array2<Complex>,
        scattering_density: Array2<Complex>,
        embedded_density: Array1<Complex>,
        previous_density: Array1<Complex>,
        valence_density: Array1<Real>,
        occupancy_by_l: Array1<Real>,
    }

    impl Ff2gSample {
        fn input(&self) -> ValenceDensityUpdateInput<'_> {
            ValenceDensityUpdateInput {
                scattering_trace: self.scattering_trace.view(),
                potential_index: 1,
                energy_index: 1,
                last_radial_index: 5,
                scattering_ldos: self.scattering_ldos.view(),
                embedded_ldos: self.embedded_ldos.view(),
                previous_ldos: self.previous_ldos.view(),
                scattering_density: self.scattering_density.view(),
                embedded_density: self.embedded_density.view(),
                previous_density: self.previous_density.view(),
                valence_density: self.valence_density.view(),
                occupancy_by_l: self.occupancy_by_l.view(),
                current_energy: Complex::new(0.72, 0.11),
                previous_energy: Complex::new(0.61, -0.04),
                potential_multiplicity: 2.5,
                current_floor: 1,
                previous_floor: 0,
                left_sum: Complex::new(0.2, -0.1),
                right_sum: Complex::new(-0.3, 0.25),
                total_electron_count: 1.25,
                include_high_l: false,
            }
        }
    }

    fn sample_ff2g_state() -> Ff2gSample {
        let l_count = 4;
        let potential_count = 3;
        let radial_count = 251;
        let scattering_trace = (0..l_count)
            .map(|angular| {
                let l = angular as Real;
                Complex32::new(
                    ((0.05_f32 as Real) * l + 0.11_f32 as Real) as f32,
                    ((-0.03_f32 as Real) * l + 0.07_f32 as Real) as f32,
                )
            })
            .collect::<Array1<_>>();
        let scattering_ldos = (0..l_count)
            .map(|angular| {
                let l = angular as Real;
                Complex::new(0.2 + 0.04 * l, -0.13 + 0.02 * l)
            })
            .collect::<Array1<_>>();
        let mut embedded_ldos = Array2::<Complex>::zeros((l_count, potential_count));
        let mut previous_ldos = Array2::<Complex>::zeros((l_count, potential_count));
        for angular in 0..l_count {
            let l = angular as Real;
            for potential in 0..potential_count {
                let p = potential as Real;
                embedded_ldos[(angular, potential)] =
                    Complex::new(0.4 + 0.03 * l + 0.02 * p, -0.2 + 0.01 * l - 0.015 * p);
                previous_ldos[(angular, potential)] =
                    Complex::new(-0.1 + 0.025 * l + 0.01 * p, 0.08 - 0.02 * l + 0.005 * p);
            }
        }
        let embedded_density = (1..=radial_count)
            .map(|radial| {
                let r = radial as Real;
                Complex::new(0.05 * r, -0.02 * r)
            })
            .collect::<Array1<_>>();
        let previous_density = (1..=radial_count)
            .map(|radial| {
                let r = radial as Real;
                Complex::new(-0.03 * r, 0.04 * r)
            })
            .collect::<Array1<_>>();
        let valence_density = (1..=radial_count)
            .map(|radial| 0.01 * radial as Real)
            .collect::<Array1<_>>();
        let mut scattering_density = Array2::<Complex>::zeros((radial_count, l_count));
        for radial in 0..radial_count {
            let r = (radial + 1) as Real;
            for angular in 0..l_count {
                let l = angular as Real;
                scattering_density[(radial, angular)] =
                    Complex::new(0.006 * r + 0.02 * l, -0.004 * r + 0.015 * l);
            }
        }
        let occupancy_by_l = (0..l_count)
            .map(|angular| -0.03 + 0.015 * angular as Real)
            .collect::<Array1<_>>();

        Ff2gSample {
            scattering_trace,
            scattering_ldos,
            embedded_ldos,
            previous_ldos,
            scattering_density,
            embedded_density,
            previous_density,
            valence_density,
            occupancy_by_l,
        }
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-8_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}
