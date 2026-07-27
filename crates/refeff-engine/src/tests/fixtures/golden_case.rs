//! `GoldenCase` API (F5): a single lookup for a golden fixture tree under
//! `reference-work/golden/<category>/<name>`, replacing 38 bespoke
//! `reference_*`/`stock_*` helper functions and the `unzip` subprocess used
//! to pull individual entries out of `REFERENCE.zip`.

use super::*;

/// A located golden fixture directory, e.g. `reference-work/golden/XANES/BN`.
pub(in crate::tests) struct GoldenCase {
    dir: PathBuf,
}

impl GoldenCase {
    /// Locate `reference-work/golden/<rel>` relative to the workspace root.
    /// Returns `None` (not an error) when the directory does not exist, so
    /// fixture-gated tests can skip cleanly.
    pub(in crate::tests) fn locate(rel: &str) -> Option<GoldenCase> {
        let root = workspace_root()?;
        let dir = root.join("reference-work/golden").join(rel);
        dir.is_dir().then_some(GoldenCase { dir })
    }

    /// The golden case's directory on disk.
    pub(in crate::tests) fn path(&self) -> &Path {
        &self.dir
    }

    /// `REFERENCE.zip` inside this case's directory, if present.
    pub(in crate::tests) fn zip(&self) -> Option<PathBuf> {
        let zip_path = self.dir.join("REFERENCE.zip");
        zip_path.is_file().then_some(zip_path)
    }

    /// `true` when every one of `names` exists as a file directly under this
    /// case's directory.
    pub(in crate::tests) fn require_files(&self, names: &[&str]) -> bool {
        names.iter().all(|name| self.dir.join(name).is_file())
    }

    /// Read one entry out of this case's `REFERENCE.zip` using the
    /// pure-Rust `zip` crate (no `unzip` subprocess).
    // Part of the GoldenCase API (F5); call sites currently go through
    // `zip()` + `unzip_reference_entry` directly.
    #[allow(dead_code)]
    pub(in crate::tests) fn zip_entry(&self, name: &str) -> Result<Vec<u8>> {
        let zip_path = self
            .zip()
            .with_context(|| format!("REFERENCE.zip not found under {}", self.dir.display()))?;
        unzip_reference_entry(&zip_path, name)
    }
}

/// The workspace root, found relative to this crate's manifest directory.
pub(in crate::tests) fn workspace_root() -> Option<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

/// Read `entry` out of the zip archive at `zip_path` using the pure-Rust
/// `zip` crate. Replaces the previous `unzip -p` subprocess call, which was
/// slow and unavailable on Windows CI runners.
pub(in crate::tests) fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
    let file = std::fs::File::open(zip_path)
        .with_context(|| format!("failed to open {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read zip archive {}", zip_path.display()))?;
    let mut zip_file = archive
        .by_name(entry)
        .with_context(|| format!("failed to find {entry} in {}", zip_path.display()))?;
    let mut contents = Vec::new();
    std::io::Read::read_to_end(&mut zip_file, &mut contents)
        .with_context(|| format!("failed to read {entry} from {}", zip_path.display()))?;
    Ok(contents)
}

/// Find the most recently created (by lexicographic sort of directory name,
/// which matches the timestamp-suffixed naming these generated tmp dirs
/// use) subdirectory of `reference-work/tmp` whose name starts with
/// `prefix` and which contains every file in `required`.
pub(in crate::tests) fn latest_generated_tmp_dir(
    prefix: &str,
    required: &[&str],
) -> Result<Option<PathBuf>> {
    let Some(root) = workspace_root() else {
        return Ok(None);
    };
    let tmp_dir = root.join("reference-work/tmp");
    if !tmp_dir.is_dir() {
        return Ok(None);
    }

    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(&tmp_dir)
        .with_context(|| format!("failed to read {}", tmp_dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", tmp_dir.display()))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with(prefix) && required.iter().all(|file| path.join(file).is_file()) {
            candidates.push(path);
        }
    }
    candidates.sort();
    Ok(candidates.pop())
}
