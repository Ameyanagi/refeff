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
    out[0] = u8::try_from(iexp + IOFF + IBAS2).expect("PAD exponent byte");
    out[1] = u8::try_from(2 * itmp + isgn + IOFF).expect("PAD sign byte");
    xwork = xwork * f64::from(IBAS2) - f64::from(itmp);

    for slot in out.iter_mut().take(npack).skip(2) {
        itmp = (f64::from(IBASE) * xwork + 1.0e-9) as i32;
        *slot = u8::try_from(itmp + IOFF).expect("PAD mantissa byte");
        xwork = xwork * f64::from(IBASE) - f64::from(itmp);
    }

    if xwork >= 0.5 {
        let rounded = itmp + IOFF + 1;
        if rounded <= 126 {
            out[npack - 1] = u8::try_from(rounded).expect("PAD rounded byte");
        } else {
            let prev = out[npack - 2];
            if prev < 126 {
                out[npack - 2] = prev + 1;
                out[npack - 1] = 37;
            }
        }
    }

    Ok(String::from_utf8(out).expect("PAD is printable ASCII"))
}

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
        let exponent = i32::try_from(i + 1).expect("PAD exponent index");
        sum += f64::from(i32::from(bytes[i]) - IOFF) / base.powi(exponent);
    }

    Ok(2.0 * f64::from(isgn) * f64::from(IBASE) * sum * 10_f64.powi(iexp))
}

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

pub fn decode_reals(text: &str, npack: usize, expected: usize) -> Result<Vec<f64>> {
    decode_lines(text, npack, expected, CPADR, |payload| {
        decode_f64(payload, npack)
    })
}

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
            let chunk = std::str::from_utf8(chunk).expect("PAD is ASCII");
            values.push(decode_unit(chunk)?);
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_real_pad_values() {
        let values = [0.0, 1.0, -1.25, 12345.678, 1.0e-20];
        let encoded = encode_reals(&values, 8).expect("encode");
        let decoded = decode_reals(&encoded, 8, values.len()).expect("decode");
        for (actual, expected) in decoded.iter().zip(values) {
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6);
        }
    }

    #[test]
    fn roundtrips_complex_pad_values() {
        let values = [Complex64::new(1.0, -2.0), Complex64::new(0.25, 1000.0)];
        let encoded = encode_complex(&values, 8).expect("encode");
        let decoded = decode_complex(&encoded, 8, values.len()).expect("decode");
        for (actual, expected) in decoded.iter().zip(values) {
            assert!((actual.re - expected.re).abs() <= expected.re.abs().max(1.0) * 1.0e-6);
            assert!((actual.im - expected.im).abs() <= expected.im.abs().max(1.0) * 1.0e-6);
        }
    }

    #[test]
    fn known_zero_encoding_matches_padlib_shape() {
        assert_eq!(encode_f64(0.0, 8).expect("encode"), "Q%%%%%%%");
        assert_eq!(decode_f64("Q%%%%%%%", 8).expect("decode"), 0.0);
    }
}
