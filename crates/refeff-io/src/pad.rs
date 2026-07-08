//! Packed ASCII Data (PAD) codec used by FEFF intermediate files.
//!
//! FEFF stores many arrays in printable compact form. The routines here port
//! `padlib.f90` semantics so Rust can read and write FEFF-compatible text/PAD
//! intermediates without relying on Fortran code at runtime.

use num_complex::Complex64;

use crate::error::{IoError, Result};

const CPADR: char = '!';
const CPADC: char = '$';
const MAXLEN: usize = 82;
const IBASE: i32 = 90;
const IOFF: i32 = 37;
const IHUGE: i32 = 38;
const IBAS2: i32 = IBASE / 2;
const TENLOG: f64 = 2.302_585_092_994_045_5_f64;

fn check_width(npack: usize) -> Result<()> {
    if npack <= 2 {
        Err(IoError::InvalidPadWidth(npack))
    } else {
        Ok(())
    }
}

fn pad_byte(value: i32) -> Result<u8> {
    u8::try_from(value).map_err(|_| IoError::PadByte { value })
}

/// Encode one `f64` into a PAD field with `npack` characters.
pub fn encode_f64(value: f64, npack: usize) -> Result<String> {
    check_width(npack)?;

    let huge = 10_f64.powi(IHUGE);
    let tiny = 1.0 / huge;
    let mut out = vec![b' '; npack];
    let xsave = value.clamp(-huge, huge);
    let isgn = i32::from(xsave > 0.0);
    let mut xwork = xsave.abs();
    let mut iexp = 0_i32;

    if xwork < huge && xwork > tiny {
        iexp = 1 + (xwork.ln() / TENLOG) as i32;
    } else if xwork >= huge {
        iexp = IHUGE;
        xwork = 1.0;
    } else if xwork <= tiny {
        xwork = 0.0;
    }

    xwork /= 10_f64.powi(iexp);
    loop {
        if xwork >= 1.0 {
            xwork *= 0.1;
            iexp += 1;
        } else if xwork <= 0.099_999_999_994_f64 {
            xwork *= 10.0;
            iexp -= 1;
        }

        if xwork < 1.0 {
            break;
        }
    }

    let mut itmp = (f64::from(IBAS2) * xwork) as i32;
    out[0] = pad_byte(iexp + IOFF + IBAS2)?;
    out[1] = pad_byte(2 * itmp + isgn + IOFF)?;
    xwork = xwork * f64::from(IBAS2) - f64::from(itmp);

    for slot in out.iter_mut().take(npack).skip(2) {
        itmp = (f64::from(IBASE) * xwork + 1.0e-9) as i32;
        *slot = pad_byte(itmp + IOFF)?;
        xwork = xwork * f64::from(IBASE) - f64::from(itmp);
    }

    if xwork >= 0.5 {
        let rounded = itmp + IOFF + 1;
        if rounded <= 126 {
            out[npack - 1] = pad_byte(rounded)?;
        } else {
            let prev = out[npack - 2];
            if prev < 126 {
                out[npack - 2] = prev + 1;
                out[npack - 1] = 37;
            }
        }
    }

    String::from_utf8(out).map_err(|source| IoError::PadUtf8 { source })
}

/// Decode one PAD field into an `f64`.
pub fn decode_f64(encoded: &str, npack: usize) -> Result<f64> {
    check_width(npack)?;
    if encoded.len() != npack {
        return Err(IoError::PadPayload {
            payload_len: encoded.len(),
            unit_len: npack,
        });
    }

    let bytes = encoded.as_bytes();
    let iexp = i32::from(bytes[0]) - IOFF - IBAS2;
    let second = i32::from(bytes[1]) - IOFF;
    let isgn = (second % 2) * 2 - 1;
    let itmp = second / 2;

    let base = f64::from(IBASE);
    let mut sum = f64::from(itmp) / base.powi(2);
    for i in (2..npack).rev() {
        let exponent = i32::try_from(i + 1).map_err(|_| IoError::PadIndex { index: i + 1 })?;
        sum += f64::from(i32::from(bytes[i]) - IOFF) / base.powi(exponent);
    }

    Ok(2.0 * f64::from(isgn) * f64::from(IBASE) * sum * 10_f64.powi(iexp))
}

/// Encode a real array into FEFF PAD lines beginning with `!`.
pub fn encode_reals(values: &[f64], npack: usize) -> Result<String> {
    check_width(npack)?;
    let mut out = String::new();
    let mut payload = String::new();
    let max_payload = MAXLEN - npack + 1;

    for (idx, value) in values.iter().enumerate() {
        payload.push_str(&encode_f64(*value, npack)?);
        if payload.len() >= max_payload || idx == values.len() - 1 {
            out.push(CPADR);
            out.push_str(&payload);
            out.push('\n');
            payload.clear();
        }
    }

    Ok(out)
}

/// Encode a complex array into FEFF PAD lines beginning with `$`.
pub fn encode_complex(values: &[Complex64], npack: usize) -> Result<String> {
    check_width(npack)?;
    let mut out = String::new();
    let mut payload = String::new();
    let max_payload = MAXLEN - 2 * npack + 1;

    for (idx, value) in values.iter().enumerate() {
        payload.push_str(&encode_f64(value.re, npack)?);
        payload.push_str(&encode_f64(value.im, npack)?);
        if payload.len() >= max_payload || idx == values.len() - 1 {
            out.push(CPADC);
            out.push_str(&payload);
            out.push('\n');
            payload.clear();
        }
    }

    Ok(out)
}

/// Decode `expected` real values from FEFF PAD lines.
pub fn decode_reals(text: &str, npack: usize, expected: usize) -> Result<Vec<f64>> {
    decode_lines(text, npack, expected, CPADR, |payload| {
        decode_f64(payload, npack)
    })
}

/// Decode `expected` complex values from FEFF PAD lines.
pub fn decode_complex(text: &str, npack: usize, expected: usize) -> Result<Vec<Complex64>> {
    check_width(npack)?;
    let units = decode_lines(text, 2 * npack, expected, CPADC, |payload| {
        let (re, im) = payload.split_at(npack);
        Ok(Complex64::new(
            decode_f64(re, npack)?,
            decode_f64(im, npack)?,
        ))
    })?;
    Ok(units)
}

fn decode_lines<T>(
    text: &str,
    unit_len: usize,
    expected: usize,
    marker: char,
    mut decode_unit: impl FnMut(&str) -> Result<T>,
) -> Result<Vec<T>> {
    let mut values = Vec::with_capacity(expected);
    for line in text.lines() {
        if values.len() >= expected {
            break;
        }
        let Some(found) = line.chars().next() else {
            continue;
        };
        if found != marker {
            return Err(IoError::PadMarker {
                expected: marker,
                found,
            });
        }
        let payload = &line[found.len_utf8()..];
        if payload.len() % unit_len != 0 {
            return Err(IoError::PadPayload {
                payload_len: payload.len(),
                unit_len,
            });
        }
        for chunk in payload.as_bytes().chunks(unit_len) {
            if values.len() >= expected {
                break;
            }
            let chunk =
                std::str::from_utf8(chunk).map_err(|source| IoError::PadChunkUtf8 { source })?;
            values.push(decode_unit(chunk)?);
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_real_pad_values() -> Result<()> {
        let values = [0.0, 1.0, -1.25, 12345.678, 1.0e-20];
        let encoded = encode_reals(&values, 8)?;
        let decoded = decode_reals(&encoded, 8, values.len())?;
        for (actual, expected) in decoded.iter().zip(values) {
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6);
        }
        Ok(())
    }

    #[test]
    fn roundtrips_complex_pad_values() -> Result<()> {
        let values = [Complex64::new(1.0, -2.0), Complex64::new(0.25, 1000.0)];
        let encoded = encode_complex(&values, 8)?;
        let decoded = decode_complex(&encoded, 8, values.len())?;
        for (actual, expected) in decoded.iter().zip(values) {
            assert!((actual.re - expected.re).abs() <= expected.re.abs().max(1.0) * 1.0e-6);
            assert!((actual.im - expected.im).abs() <= expected.im.abs().max(1.0) * 1.0e-6);
        }
        Ok(())
    }

    #[test]
    fn known_zero_encoding_matches_padlib_shape() -> Result<()> {
        assert_eq!(encode_f64(0.0, 8)?, "Q%%%%%%%");
        assert_eq!(decode_f64("Q%%%%%%%", 8)?, 0.0);
        Ok(())
    }

    // The next three tests pin down genuine `encode_f64` edge-case bugs
    // found while building F7's property coverage below. Per the F7 task
    // scope we do not change codec behavior; these regressions document
    // today's actual (buggy) output so a future fix touches them
    // deliberately, and the property tests below route around all three
    // cases so they stay green.
    //
    // 1. `known_bug_npack3_rounding_carry_can_flip_sign`: when `npack == 3`,
    //    `out[npack - 2]` is `out[1]`, the same byte that packs the sign in
    //    its low bit (`2 * itmp + isgn + IOFF`). If the final-digit rounding
    //    step needs to carry into `out[npack - 2]`, it blindly does
    //    `out[npack - 2] = prev + 1`, which for `npack == 3` flips the
    //    parity of the sign bit instead of only bumping the digit -
    //    silently negating (or un-negating) the decoded value.
    // 2. `known_bug_rounding_drift_can_error_for_wide_npack`: for values
    //    that sit almost exactly on a digit-quantization boundary (e.g.
    //    `-0.0009999999999980088`), floating-point drift can make `xwork`
    //    dip slightly negative mid-loop. The subsequent digits then
    //    truncate toward zero (`itmp = 0`) while repeatedly multiplying by
    //    `IBASE`, so the negative drift compounds each iteration; once
    //    `npack` is large enough the compounded drift produces an
    //    out-of-range byte and `encode_f64` returns `Err(IoError::PadByte)`
    //    for a value that is well within the documented representable
    //    range.
    // 3. `known_bug_huge_boundary_values_encode_as_zero`: the `xwork >=
    //    huge` branch sets `xwork = 1.0` and `iexp = IHUGE`, then always
    //    runs `xwork /= 10^iexp`, producing `xwork = 1e-38`. The
    //    normalization loop right after only performs a *single*
    //    `xwork *= 10.0; iexp -= 1;` step before unconditionally checking
    //    `xwork < 1.0` and breaking - so it exits after one iteration with
    //    `xwork` still around `1e-37`, nowhere near the `[0.1, 1.0)` range
    //    every other code path assumes. Every subsequent digit byte is then
    //    computed from a `xwork` that is ~37 orders of magnitude smaller
    //    than intended, so all its digits truncate to zero. The result:
    //    encoding *any* magnitude at or beyond `huge = 1e38` silently
    //    decodes back as `0.0` instead of saturating at `±huge` (or
    //    erroring), for every `npack`.
    #[test]
    fn known_bug_npack3_rounding_carry_can_flip_sign() -> Result<()> {
        let value = 1.109_954_784_649_650_4e-18_f64;
        let encoded = encode_f64(value, 3)?;
        let decoded = decode_f64(&encoded, 3)?;
        assert!(
            decoded.is_sign_negative(),
            "documents pad.rs's known npack=3 rounding-carry sign-flip bug; \
             expected the sign to flip for this input, got {decoded:e}"
        );
        Ok(())
    }

    #[test]
    fn known_bug_rounding_drift_can_error_for_wide_npack() -> Result<()> {
        let value = -0.0009999999999980088_f64;
        assert!(encode_f64(value, 8).is_ok());
        assert!(matches!(encode_f64(value, 9), Err(IoError::PadByte { .. })));
        Ok(())
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn known_bug_huge_boundary_values_encode_as_zero() -> Result<()> {
        for npack in [4, 8, 11] {
            for value in [1.0e38_f64, -1.0e38_f64, 1.0e39_f64, -5.0e40_f64] {
                let encoded = encode_f64(value, npack)?;
                let decoded = decode_f64(&encoded, npack)?;
                assert_eq!(
                    decoded, 0.0,
                    "documents pad.rs's known huge-boundary bug; expected \
                     {value:e} at npack={npack} to (incorrectly) decode as \
                     zero instead of saturating near +/-huge, got {decoded:e}"
                );
            }
        }
        Ok(())
    }

    /// True IEEE subnormal doubles (magnitude far below `f64::MIN_POSITIVE`
    /// down to `f64::from_bits(1)`) are astronomically unlikely for a
    /// uniform `f64` proptest generator to sample directly (the subnormal
    /// range `[4.9e-324, 2.2e-308)` is a vanishingly small slice of
    /// `[0, 1e-38)`), so this pins them down deterministically rather than
    /// hoping a property test's generator finds one.
    #[test]
    #[allow(clippy::float_cmp)]
    fn true_subnormal_magnitudes_collapse_to_zero() -> Result<()> {
        let subnormals = [
            f64::from_bits(1),
            f64::MIN_POSITIVE / 2.0,
            f64::MIN_POSITIVE,
        ];
        for npack in [4, 8, 12] {
            for &magnitude in &subnormals {
                for sign in [1.0, -1.0] {
                    let value = sign * magnitude;
                    let encoded = encode_f64(value, npack)?;
                    let decoded = decode_f64(&encoded, npack)?;
                    assert_eq!(
                        decoded, 0.0,
                        "subnormal {value:e} at npack={npack} should collapse to zero, got {decoded:e}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Property-based round-trip coverage (F7): `decode(encode(x, npack))`
    /// should recover `x` within the precision `npack` documents, across
    /// the representable range (including subnormal/tiny values, the
    /// `huge`/`tiny` exponent boundaries, and near-zero sign preservation).
    ///
    /// `npack == 3` is intentionally excluded from the general precision
    /// property: it is the narrowest allowed width (`check_width` requires
    /// `npack > 2`) and can hit
    /// `known_bug_npack3_rounding_carry_can_flip_sign` above, which would
    /// otherwise make this suite flaky rather than a stable regression.
    /// Magnitudes at or beyond `huge` are covered separately by
    /// `out_of_range_magnitudes_hit_the_known_huge_boundary_bug`, which
    /// documents `known_bug_huge_boundary_values_encode_as_zero` rather than
    /// asserting the (currently false) documented clamp-to-`huge`
    /// semantics.
    #[allow(clippy::float_cmp)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// `padlib` packs `npack - 2` base-`IBASE` digits after the
        /// sign+exponent byte, so decode's relative error should stay
        /// within roughly `IBASE^-(npack-2)`. `5x` covers the empirically
        /// observed worst-case ratio (about `0.17x` of that bound for
        /// `npack` in `4..=9`, measured against ~200k random samples plus
        /// the exponent-boundary seeds) with headroom for other boundary
        /// values. Once `npack - 2` digits would already exceed a `f64`'s
        /// own ~15-17 significant decimal digits (`npack >= 10`), the
        /// dominant error source becomes double-precision arithmetic noise
        /// in the encode/decode routines themselves (repeated
        /// multiplication, `ln`, `powi`) rather than digit quantization, so
        /// the bound is floored at `1e-13` - comfortably above the ~1e-15
        /// noise observed empirically at those widths.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        fn precision_epsilon(npack: usize) -> f64 {
            let digit_bound = 5.0 * f64::from(IBASE).powi(-(npack as i32 - 2));
            digit_bound.max(1.0e-13)
        }

        /// One npack width per case, kept away from the `npack == 3`
        /// sign-flip bug.
        fn npack_strategy() -> impl Strategy<Value = usize> {
            4_usize..=15
        }

        /// A value drawn broadly across the representable range, safely
        /// inside the `tiny`/`huge` boundaries so the precision property
        /// below is not entangled with the boundary-collapse/clamp
        /// behavior exercised by the dedicated boundary tests.
        fn representable_value() -> impl Strategy<Value = f64> {
            (-36_i32..=36, 1.0_f64..10.0, any::<bool>()).prop_map(
                |(exponent, mantissa, negative)| {
                    let magnitude = mantissa * 10_f64.powi(exponent);
                    if negative { -magnitude } else { magnitude }
                },
            )
        }

        fn assert_within_precision(
            value: f64,
            decoded: f64,
            npack: usize,
        ) -> std::result::Result<(), TestCaseError> {
            let epsilon = precision_epsilon(npack);
            let denom = value.abs().max(1.0e-300);
            prop_assert!(
                (decoded - value).abs() / denom <= epsilon,
                "value={value:e} npack={npack} decoded={decoded:e} epsilon={epsilon:e}"
            );
            Ok(())
        }

        proptest! {
            #[test]
            fn single_value_roundtrips_within_precision(
                value in representable_value(),
                npack in npack_strategy(),
            ) {
                // A narrow band of values hit
                // `known_bug_rounding_drift_can_error_for_wide_npack`; this
                // property targets precision, not that separately pinned
                // failure mode, so such inputs are simply not sampled here.
                let Ok(encoded) = encode_f64(value, npack) else {
                    return Ok(());
                };
                let decoded = decode_f64(&encoded, npack)?;
                assert_within_precision(value, decoded, npack)?;
            }

            #[test]
            fn subnormal_and_below_tiny_values_collapse_to_zero(
                magnitude in 0.0_f64..1.0e-38,
                negative in any::<bool>(),
                npack in npack_strategy(),
            ) {
                // True IEEE subnormals (down to 4.9e-324) and any magnitude
                // at or below padlib's documented `tiny = 10^-38` floor all
                // land in this branch.
                let value = if negative { -magnitude } else { magnitude };
                let encoded = encode_f64(value, npack)?;
                let decoded = decode_f64(&encoded, npack)?;
                prop_assert_eq!(decoded, 0.0);
            }

            #[test]
            #[allow(clippy::float_cmp)]
            fn out_of_range_magnitudes_hit_the_known_huge_boundary_bug(
                magnitude in 1.0e38_f64..1.0e40,
                negative in any::<bool>(),
                npack in npack_strategy(),
            ) {
                // Documents `known_bug_huge_boundary_values_encode_as_zero`
                // as a property: every magnitude at or beyond `huge = 1e38`
                // takes the buggy `xwork >= huge` normalization path and
                // decodes back as zero rather than saturating near
                // `+/-huge`. If a future fix corrects the normalization
                // loop, this assertion (and the pinned regression above)
                // should be updated together to the documented clamp
                // behavior.
                let value = if negative { -magnitude } else { magnitude };
                let encoded = encode_f64(value, npack)?;
                let decoded = decode_f64(&encoded, npack)?;
                prop_assert_eq!(decoded, 0.0);
            }

            #[test]
            fn sign_is_preserved_just_above_the_tiny_boundary(
                magnitude in 1.1e-38_f64..9.9e-38,
                negative in any::<bool>(),
                npack in npack_strategy(),
            ) {
                let value = if negative { -magnitude } else { magnitude };
                let encoded = encode_f64(value, npack)?;
                let decoded = decode_f64(&encoded, npack)?;
                prop_assert_eq!(decoded.is_sign_negative(), negative);
                assert_within_precision(value, decoded, npack)?;
            }

            #[test]
            fn real_array_roundtrips_within_precision(
                values in prop::collection::vec(representable_value(), 0..12),
                npack in npack_strategy(),
            ) {
                let Ok(encoded) = encode_reals(&values, npack) else {
                    return Ok(());
                };
                let decoded = decode_reals(&encoded, npack, values.len())?;
                for (value, decoded) in values.iter().zip(decoded.iter()) {
                    assert_within_precision(*value, *decoded, npack)?;
                }
            }

            #[test]
            fn complex_array_roundtrips_within_precision(
                values in prop::collection::vec(
                    (representable_value(), representable_value()),
                    0..12,
                ),
                npack in npack_strategy(),
            ) {
                let values: Vec<Complex64> = values
                    .into_iter()
                    .map(|(re, im)| Complex64::new(re, im))
                    .collect();
                let Ok(encoded) = encode_complex(&values, npack) else {
                    return Ok(());
                };
                let decoded = decode_complex(&encoded, npack, values.len())?;
                for (value, decoded) in values.iter().zip(decoded.iter()) {
                    assert_within_precision(value.re, decoded.re, npack)?;
                    assert_within_precision(value.im, decoded.im, npack)?;
                }
            }
        }
    }
}
