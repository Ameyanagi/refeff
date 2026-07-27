//! Shared numeric tolerance policy for parity/regression assertions (F6).
//!
//! Historically every parity test picked its own magic tolerance float and
//! `assert_close` reported only the first failing pair with no context. This
//! module gives magic numbers a name (`Tol::XMU`, `Tol::PHASE_SHIFT`, ...) and
//! provides scalar/array comparators that report max-abs, max-rel, RMS and
//! the offending index on failure.
//!
//! `Tol` combines a relative tolerance (`rel * |expected|`) and an absolute
//! floor (`abs`); the effective threshold is `abs.max(rel * |expected|)`.
//! When `rel == abs == t` this is exactly `t * expected.abs().max(1.0)`,
//! which is the formula the legacy [`assert_close`] used everywhere, so
//! named profiles built that way reproduce old call sites bit-for-bit.

/// A named relative+absolute tolerance floor for a scalar comparison.
#[derive(Debug, Clone, Copy)]
pub(in crate::tests) struct Tol {
    pub(in crate::tests) rel: f64,
    pub(in crate::tests) abs: f64,
}

impl Tol {
    /// Values that should round-trip through a bin/text codec essentially
    /// exactly (write-then-read-back with no lossy re-derivation).
    pub(in crate::tests) const EXACT_ECHO: Tol = Tol {
        rel: 1.0e-12,
        abs: 1.0e-12,
    };

    /// Typical spectral quantity comparison against a FEFF10 reference
    /// (xmu/eels/band spectra); matches the `5.0e-5` constant repeated
    /// across `eels.rs`, `eelsmdff.rs`, `ldos.rs`, `band.rs`.
    // Part of the named-tolerance vocabulary (F6); not yet referenced by a
    // migrated call site.
    #[allow(dead_code)]
    pub(in crate::tests) const XMU: Tol = Tol {
        rel: 5.0e-5,
        abs: 5.0e-5,
    };

    /// Phase-shift-like quantities compared against FEFF10 output, tighter
    /// than [`Tol::XMU`] since these are direct solver outputs rather than
    /// post-processed spectra.
    pub(in crate::tests) const PHASE_SHIFT: Tol = Tol {
        rel: 1.0e-10,
        abs: 1.0e-10,
    };

    /// OPCONS/loss energy-grid comparison against a FEFF10 reference.
    pub(in crate::tests) const REFERENCE_ENERGY: Tol = Tol {
        rel: 2.0e-6,
        abs: 2.0e-6,
    };

    /// OPCONS loss-function value comparison against a FEFF10 reference.
    pub(in crate::tests) const REFERENCE_LOSS: Tol = Tol {
        rel: 2.0e-5,
        abs: 2.0e-5,
    };

    /// Round-trip precision floor for the PAD fixed-width codec (see F7);
    /// PAD is lossy, so exact-echo tolerance is inappropriate for it.
    // Part of the named-tolerance vocabulary (F6); not yet referenced by a
    // migrated call site.
    #[allow(dead_code)]
    pub(in crate::tests) const PAD_ROUNDTRIP: Tol = Tol {
        rel: 1.0e-6,
        abs: 1.0e-10,
    };

    /// The effective absolute threshold for a comparison against `expected`.
    pub(in crate::tests) fn threshold(&self, expected: f64) -> f64 {
        self.abs.max(self.rel * expected.abs())
    }

    /// Assert `actual` is within this tolerance of `expected`.
    pub(in crate::tests) fn assert(&self, actual: f64, expected: f64) {
        let threshold = self.threshold(expected);
        let diff = (actual - expected).abs();
        assert!(
            diff <= threshold,
            "{actual} != {expected} (|diff|={diff:.6e} > threshold={threshold:.6e}, rel={:.3e}, abs={:.3e})",
            self.rel,
            self.abs
        );
    }

    /// Assert every element of `actual` is within this tolerance of the
    /// corresponding element of `expected`. On failure the panic message
    /// reports max-abs, max-rel, RMS and the first offending index rather
    /// than only the first failing pair.
    // Part of the named-tolerance vocabulary (F6); not yet referenced by a
    // migrated call site.
    #[allow(dead_code)]
    pub(in crate::tests) fn assert_slice(&self, actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "array length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        );

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut first_offender: Option<usize> = None;

        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let rel = if e.abs() > 0.0 { diff / e.abs() } else { diff };
            max_abs = max_abs.max(diff);
            max_rel = max_rel.max(rel);
            sum_sq += diff * diff;
            if first_offender.is_none() && diff > self.threshold(e) {
                first_offender = Some(index);
            }
        }

        let Some(index) = first_offender else {
            return;
        };
        let rms = (sum_sq / actual.len().max(1) as f64).sqrt();
        panic!(
            "array comparison failed at index {index} (actual={} expected={}): max_abs={max_abs:.6e} max_rel={max_rel:.6e} rms={rms:.6e}",
            actual[index], expected[index]
        );
    }
}

/// Legacy scalar comparator: `|actual - expected| <= tolerance *
/// expected.abs().max(1.0)`. Delegates to [`Tol`] so all comparisons share
/// one implementation; kept so the ~250+ existing call sites across the
/// crate don't have to churn (see F6 in TODO.md).
// Kept as the migration target for module-local assert_close helpers (F6).
#[allow(dead_code)]
pub(in crate::tests) fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    Tol {
        rel: tolerance,
        abs: tolerance,
    }
    .assert(actual, expected);
}
