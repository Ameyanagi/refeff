//! Provenance manifests for golden fixture directories (F2).
//!
//! Golden trees under `reference-work/golden/<category>/<name>` and the
//! `REFERENCE.zip` fixtures they carry record nothing about which FEFF10
//! commit, compiler, or compiler flags produced them, even though compiler
//! choice measurably changes FEFF numerics. `xtask generate-golden` writes a
//! `manifest.json` alongside every case it (re)generates recording that
//! provenance plus a SHA-256 checksum of every file in the case directory;
//! `xtask compatibility-matrix` reads those manifests back to warn about
//! fixtures with no manifest and to detect fixtures that are stale relative
//! to the current `feff10/` checkout.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Name of the manifest file written into (and read from) a golden case
/// directory.
pub(crate) const MANIFEST_FILE_NAME: &str = "manifest.json";

/// Which reference-FEFF10 compiler (if any) produced a golden fixture tree,
/// and with what flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompilerInfo {
    pub(crate) name: String,
    pub(crate) flags: String,
}

impl CompilerInfo {
    /// Used when `generate-golden --no-build` skips building the reference
    /// FEFF10 binary, so the compiler that actually produced it is unknown.
    pub(crate) fn unknown() -> Self {
        Self {
            name: "unknown".to_string(),
            flags: "n/a (reference binary not built by this invocation)".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FileChecksum {
    /// Path relative to the golden case directory, `/`-separated.
    pub(crate) path: String,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GoldenManifest {
    /// `git -C feff10 rev-parse HEAD` at generation time, or `None` when
    /// `feff10/` was not a git checkout.
    pub(crate) feff10_rev: Option<String>,
    pub(crate) compiler: String,
    pub(crate) compiler_flags: String,
    pub(crate) hostname: String,
    pub(crate) os: String,
    /// Seconds since the Unix epoch (kept dependency-free rather than
    /// pulling in a date/time formatting crate).
    pub(crate) generated_unix_time: u64,
    pub(crate) xtask_version: String,
    /// SHA-256 of every file in the case directory (excluding
    /// `manifest.json` itself), sorted by path for stable diffs.
    pub(crate) files: Vec<FileChecksum>,
}

/// Write `manifest.json` inside `case_dir`, hashing every other file
/// currently present there.
pub(crate) fn write_manifest(
    case_dir: &Path,
    feff10_rev: Option<&str>,
    compiler: &CompilerInfo,
) -> Result<()> {
    let files = checksum_directory(case_dir)?;
    let manifest = GoldenManifest {
        feff10_rev: feff10_rev.map(str::to_string),
        compiler: compiler.name.clone(),
        compiler_flags: compiler.flags.clone(),
        hostname: current_hostname(),
        os: std::env::consts::OS.to_string(),
        generated_unix_time: unix_now(),
        xtask_version: env!("CARGO_PKG_VERSION").to_string(),
        files,
    };
    let json = serde_json::to_string_pretty(&manifest)
        .context("failed to serialize golden fixture manifest.json")?;
    std::fs::write(case_dir.join(MANIFEST_FILE_NAME), json).with_context(|| {
        format!(
            "failed to write {} in {}",
            MANIFEST_FILE_NAME,
            case_dir.display()
        )
    })
}

/// Read and parse `manifest.json` from `case_dir`.
pub(crate) fn read_manifest(case_dir: &Path) -> Result<GoldenManifest> {
    let path = case_dir.join(MANIFEST_FILE_NAME);
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))
}

/// `true` when `case_dir/manifest.json` exists.
pub(crate) fn has_manifest(case_dir: &Path) -> bool {
    case_dir.join(MANIFEST_FILE_NAME).is_file()
}

/// Current `HEAD` commit of the FEFF10 reference checkout at `ref_dir`
/// (`git -C <ref_dir> rev-parse HEAD`), or `None` when `ref_dir` does not
/// exist or is not a git checkout.
pub(crate) fn feff10_git_rev(ref_dir: &Path) -> Option<String> {
    if !ref_dir.is_dir() {
        return None;
    }
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(ref_dir)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|rev| rev.trim().to_string())
        .filter(|rev| !rev.is_empty())
}

fn checksum_directory(dir: &Path) -> Result<Vec<FileChecksum>> {
    let mut files = Vec::new();
    collect_checksums(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

fn collect_checksums(root: &Path, dir: &Path, out: &mut Vec<FileChecksum>) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_checksums(root, &path, out)?;
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) == Some(MANIFEST_FILE_NAME) {
            continue;
        }
        let bytes =
            std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        out.push(FileChecksum {
            path: rel,
            sha256: sha256_hex(&bytes),
        });
    }
    Ok(())
}

fn current_hostname() -> String {
    if let Ok(name) = std::env::var("HOSTNAME")
        && !name.is_empty()
    {
        return name;
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

// ---- pure-Rust SHA-256 -----------------------------------------------------
//
// `sha2` is not a dependency of `xtask` (only pulled in transitively by
// `pest_meta`, not reachable from here without editing `Cargo.toml`), so this
// is a small textbook implementation (FIPS 180-4) rather than an external
// crate, per F2's "or skip checksums" fallback.

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Hex-encoded SHA-256 digest of `data`.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    sha256(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_INITIAL;

    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.chunks_exact(64) {
        sha256_process_block(&mut state, chunk);
    }

    let mut digest = [0_u8; 32];
    for (word_index, word) in state.iter().enumerate() {
        digest[word_index * 4..word_index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn sha256_process_block(state: &mut [u32; 8], chunk: &[u8]) {
    let mut schedule = [0_u32; 64];
    for (word_index, word) in schedule.iter_mut().enumerate().take(16) {
        let offset = word_index * 4;
        *word = u32::from_be_bytes([
            chunk[offset],
            chunk[offset + 1],
            chunk[offset + 2],
            chunk[offset + 3],
        ]);
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[index])
            .wrapping_add(schedule[index]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_test_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn write_manifest_then_read_manifest_round_trips() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-manifest-test-{}-{}",
            std::process::id(),
            unix_now()
        ));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("feff.inp"), b"TITLE test\nEND\n")?;
        std::fs::write(root.join("xmu.dat"), b"# xmu\n1.0 2.0\n")?;

        assert!(!has_manifest(&root));
        write_manifest(
            &root,
            Some("deadbeef"),
            &CompilerInfo {
                name: "gfortran".to_string(),
                flags: "-O3".to_string(),
            },
        )?;
        assert!(has_manifest(&root));

        let manifest = read_manifest(&root)?;
        assert_eq!(manifest.feff10_rev.as_deref(), Some("deadbeef"));
        assert_eq!(manifest.compiler, "gfortran");
        assert_eq!(manifest.compiler_flags, "-O3");
        assert_eq!(manifest.files.len(), 2);
        assert!(manifest.files.iter().any(|file| file.path == "feff.inp"));
        assert!(manifest.files.iter().any(|file| file.path == "xmu.dat"));
        assert!(
            manifest
                .files
                .iter()
                .all(|file| file.sha256.len() == 64 && file.sha256 != MANIFEST_FILE_NAME)
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn feff10_git_rev_is_none_for_missing_directory() {
        let missing = std::env::temp_dir().join("refeff-xtask-manifest-missing-feff10-checkout");
        assert_eq!(feff10_git_rev(&missing), None);
    }
}
