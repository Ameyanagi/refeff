//! `require_fixture!` (F3): make fixture-gated test skips visible and
//! enforceable instead of a silent `return Ok(())`.
//!
//! Call as `require_fixture!("reason fixture X was not found")` in place of
//! an optional-fixture early return. Under
//! `REFEFF_REQUIRE_FIXTURES=1` (the CI parity job) it panics instead of
//! skipping; otherwise it appends `<test name>: <reason>` to a ledger file
//! under `target/` so the skip count can be surfaced (e.g. "N parity tests
//! skipped") rather than silently disabling the test forever.

use super::*;
use std::io::Write;
use std::sync::Mutex;

static LEDGER_LOCK: Mutex<()> = Mutex::new(());

/// `true` when the CI parity job has asked for missing fixtures to be a
/// hard failure rather than a skip.
pub(crate) fn fixtures_required() -> bool {
    std::env::var_os("REFEFF_REQUIRE_FIXTURES").as_deref() == Some(std::ffi::OsStr::new("1"))
}

/// The ledger file that skipped fixture-gated tests are recorded to.
pub(crate) fn fixture_skip_ledger_path() -> Option<PathBuf> {
    let root = workspace_root()?;
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));
    Some(target_dir.join("fixture-skips.log"))
}

/// Append `<test name>: <reason>` to the fixture-skip ledger. Best-effort:
/// a ledger write failure must never fail the (already-skipping) test.
pub(crate) fn record_fixture_skip(reason: &str) {
    let test_name = std::thread::current()
        .name()
        .unwrap_or("<unknown test>")
        .to_string();

    let Some(path) = fixture_skip_ledger_path() else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }

    let _guard = LEDGER_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(file, "{test_name}: {reason}");
    }
}

/// Skip the current test for a missing fixture, recording it to the skip
/// ledger — unless `REFEFF_REQUIRE_FIXTURES=1`, in which case it panics.
/// Expands to a `return Ok(());` (or a panic) so it must be invoked from a
/// function returning `anyhow::Result<()>`, typically inside a `let ... else`
/// fixture-lookup block.
#[macro_export]
macro_rules! record_missing_fixture {
    ($($message:tt)+) => {{
        let reason = format!($($message)+);
        if $crate::tests::fixtures::fixtures_required() {
            panic!("required fixture missing: {reason}");
        }
        eprintln!("skipping: {reason}");
        $crate::tests::fixtures::record_fixture_skip(&reason);
    }};
}

#[macro_export]
macro_rules! require_fixture {
    ($($message:tt)+) => {{
        $crate::record_missing_fixture!($($message)+);
        return Ok(());
    }};
}

pub(crate) use require_fixture;
