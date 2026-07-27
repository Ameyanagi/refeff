//! Validation and provenance for the current-source native FEFF RIXS oracle.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result};
use refeff_io::{RixsMapData, read_rixs_map};
use serde::{Deserialize, Serialize};

pub(crate) const PROVENANCE_FILE_NAME: &str = ".rixs-native-provenance.json";
pub(crate) const MAP_FILE_NAME: &str = "rixsET.dat";
pub(crate) const LEGACY_MAP_FILE_NAME: &str = "referencerixsET.dat";
pub(crate) const PROVENANCE_SCHEMA_VERSION: u8 = 1;
pub(crate) const MAP_ORDER: usize = 193;
pub(crate) const MAP_POINT_COUNT: usize = MAP_ORDER * MAP_ORDER;
pub(crate) const CURRENT_SOURCE_POLE_INDEX: usize = 24;
pub(crate) const EXPECTED_GG_SECTION_COUNT: usize = 205;
pub(crate) const MAX_GG_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_MAP_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_PROVENANCE_BYTES: u64 = 1024 * 1024;
pub(crate) const GG_DESCRIPTOR: &[u8] = b"#DF# This section written in txt.\n";
pub(crate) const GG_NORMALIZATION_OPERATION: &str = "for every staged native gg.bin section, replace only the '#DF# This section written in <four uninitialized descriptor bytes>.' record and its optional single pre-#H# continuation with '#DF# This section written in txt.\\n'; copy every other byte unchanged";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SanitizedGg {
    pub(crate) bytes: Vec<u8>,
    pub(crate) descriptor_records: usize,
    pub(crate) continuation_lines_removed: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRixsProvenance {
    pub(crate) schema_version: u8,
    pub(crate) generator: String,
    pub(crate) native_commit: String,
    pub(crate) normalization_operation: String,
    pub(crate) edges: Vec<NativeRixsEdgeProvenance>,
    pub(crate) val_zero_screen: NativeRixsZeroScreenProvenance,
    pub(crate) solver: NativeRixsSolverProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRixsEdgeProvenance {
    pub(crate) edge: String,
    pub(crate) derived_input_sha256: String,
    pub(crate) original_gg_sha256: String,
    pub(crate) sanitized_gg_sha256: String,
    pub(crate) descriptor_records: usize,
    pub(crate) continuation_lines_removed: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRixsZeroScreenProvenance {
    pub(crate) derivation: String,
    pub(crate) row_count: usize,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRixsSolverProvenance {
    pub(crate) executable: String,
    pub(crate) output_sha256: String,
    pub(crate) map_order: usize,
    pub(crate) point_count: usize,
    pub(crate) peak_row: usize,
    pub(crate) peak_first_energy_ev: f64,
    pub(crate) peak_second_energy_ev: f64,
    pub(crate) peak_intensity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CurrentSourceMapValidation {
    pub(crate) peak_row: usize,
    pub(crate) peak_first_energy_ev: f64,
    pub(crate) peak_second_energy_ev: f64,
    pub(crate) peak_intensity: f64,
}

pub(crate) fn read_bounded_regular_file(
    path: &Path,
    byte_limit: u64,
    description: &str,
) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {description} {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{description} {} must be a regular non-symlink file",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= byte_limit,
        "{description} {} is {} bytes, exceeding the {byte_limit}-byte limit",
        path.display(),
        metadata.len()
    );
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read {description} {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() as u64 == metadata.len(),
        "{description} {} changed length while it was being read",
        path.display()
    );
    Ok(bytes)
}

/// Canonicalize only FEFF's malformed `#DF#` records.
///
/// The current `FMS/fmstot.f90` call leaves the four-byte file-type argument
/// uninitialized. Those bytes are confined to the descriptor record, but may
/// contain a newline and leave one short continuation immediately before the
/// following `#H#` record. Every non-descriptor byte is copied verbatim.
pub(crate) fn sanitize_native_gg_descriptors(input: &[u8]) -> Result<SanitizedGg> {
    anyhow::ensure!(
        input.len() as u64 <= MAX_GG_BYTES,
        "native gg.bin is {} bytes, exceeding the {}-byte limit",
        input.len(),
        MAX_GG_BYTES
    );

    let mut output = Vec::with_capacity(input.len());
    let mut cursor = 0usize;
    let mut descriptor_records = 0usize;
    let mut continuation_lines_removed = 0usize;

    while cursor < input.len() {
        let line_end = next_line_end(input, cursor);
        let line = &input[cursor..line_end];
        if !line.starts_with(b"#DF#") {
            output.extend_from_slice(line);
            cursor = line_end;
            continue;
        }

        anyhow::ensure!(
            line.starts_with(b"#DF# This section written in "),
            "native gg.bin contains an unexpected #DF# descriptor"
        );
        descriptor_records += 1;
        anyhow::ensure!(
            descriptor_records <= 4096,
            "native gg.bin contains more than 4096 #DF# records"
        );
        output.extend_from_slice(GG_DESCRIPTOR);
        cursor = line_end;

        anyhow::ensure!(
            cursor < input.len(),
            "native gg.bin ends immediately after a #DF# descriptor"
        );
        let following_end = next_line_end(input, cursor);
        let following = &input[cursor..following_end];
        if following.starts_with(b"#H#") {
            continue;
        }

        let header_end = next_line_end(input, following_end);
        let header = input
            .get(following_end..header_end)
            .context("native gg.bin descriptor continuation has no following header")?;
        anyhow::ensure!(
            header.starts_with(b"#H#"),
            "native gg.bin has data outside the bounded #DF# continuation position"
        );
        anyhow::ensure!(
            following.len() <= 8,
            "native gg.bin #DF# continuation is unexpectedly long"
        );
        continuation_lines_removed += 1;
        cursor = following_end;
    }

    anyhow::ensure!(
        descriptor_records == EXPECTED_GG_SECTION_COUNT,
        "native gg.bin contains {descriptor_records} #DF# records, expected {EXPECTED_GG_SECTION_COUNT}"
    );
    Ok(SanitizedGg {
        bytes: output,
        descriptor_records,
        continuation_lines_removed,
    })
}

fn next_line_end(bytes: &[u8], start: usize) -> usize {
    bytes[start..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(bytes.len(), |offset| start + offset + 1)
}

pub(crate) fn validate_current_source_map_file(path: &Path) -> Result<CurrentSourceMapValidation> {
    let _bytes = read_bounded_regular_file(path, MAX_MAP_BYTES, "native RIXS map")?;
    let map = read_rixs_map(path)
        .with_context(|| format!("failed to parse native RIXS map {}", path.display()))?;
    validate_current_source_map(&map)
}

pub(crate) fn validate_current_source_map(map: &RixsMapData) -> Result<CurrentSourceMapValidation> {
    anyhow::ensure!(
        map.point_count() == MAP_POINT_COUNT,
        "native RIXS map has {} numeric rows, expected {MAP_ORDER}x{MAP_ORDER}={MAP_POINT_COUNT}",
        map.point_count()
    );
    anyhow::ensure!(
        map.block_lengths.len() == MAP_ORDER
            && map
                .block_lengths
                .iter()
                .all(|block_length| *block_length == MAP_ORDER),
        "native RIXS map blocks are not exactly {MAP_ORDER} blocks of {MAP_ORDER} rows"
    );
    anyhow::ensure!(
        map.channel_count() > 0,
        "native RIXS map has no intensity channels"
    );
    anyhow::ensure!(
        map.first_energy_ev.iter().all(|value| value.is_finite())
            && map.second_energy_ev.iter().all(|value| value.is_finite()),
        "native RIXS map contains a non-finite energy"
    );
    anyhow::ensure!(
        map.channels.iter().all(|value| value.is_finite()),
        "native RIXS map contains a non-finite intensity"
    );
    anyhow::ensure!(
        map.channels.iter().all(|value| *value >= 0.0),
        "native RIXS map contains a negative intensity"
    );

    let (peak_row, peak_intensity) = map
        .channels
        .column(0)
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .context("native RIXS map has no primary-channel rows")?;
    let expected_peak_row = CURRENT_SOURCE_POLE_INDEX * MAP_ORDER + CURRENT_SOURCE_POLE_INDEX;
    anyhow::ensure!(
        peak_row == expected_peak_row,
        "native RIXS primary pole is at row {peak_row}, expected current-source row {expected_peak_row}"
    );
    let peak_first_energy_ev = map.first_energy_ev[peak_row];
    let peak_second_energy_ev = map.second_energy_ev[peak_row];
    anyhow::ensure!(
        peak_first_energy_ev > 11_560.0
            && peak_first_energy_ev < 11_565.0
            && peak_second_energy_ev.abs() <= 1.0e-6
            && peak_intensity > 0.0,
        "native RIXS peak does not match the shifted current-source pole sentinel"
    );

    Ok(CurrentSourceMapValidation {
        peak_row,
        peak_first_energy_ev,
        peak_second_energy_ev,
        peak_intensity,
    })
}

pub(crate) fn validate_published_reference(case_dir: &Path) -> Result<()> {
    let map_path = case_dir.join(MAP_FILE_NAME);
    let validation = validate_current_source_map_file(&map_path)?;
    let map_bytes = read_bounded_regular_file(&map_path, MAX_MAP_BYTES, "native RIXS map")?;
    let marker_path = case_dir.join(PROVENANCE_FILE_NAME);
    let marker_bytes =
        read_bounded_regular_file(&marker_path, MAX_PROVENANCE_BYTES, "RIXS provenance")?;
    let provenance: NativeRixsProvenance = serde_json::from_slice(&marker_bytes)
        .with_context(|| format!("failed to parse {}", marker_path.display()))?;

    anyhow::ensure!(
        provenance.schema_version == PROVENANCE_SCHEMA_VERSION,
        "unsupported RIXS provenance schema {}",
        provenance.schema_version
    );
    anyhow::ensure!(
        provenance.generator == "xtask generate-golden exact RIXS native-current-source oracle",
        "unexpected RIXS provenance generator"
    );
    anyhow::ensure!(
        is_lowercase_hex(&provenance.native_commit, 40),
        "RIXS provenance native commit is not a lowercase 40-digit Git hash"
    );
    anyhow::ensure!(
        provenance.normalization_operation == GG_NORMALIZATION_OPERATION,
        "RIXS provenance records an unexpected GG normalization"
    );
    anyhow::ensure!(
        provenance.edges.len() == 2,
        "RIXS provenance must contain exactly two edge records"
    );
    let mut edge_labels = BTreeSet::new();
    for edge in &provenance.edges {
        anyhow::ensure!(
            matches!(edge.edge.as_str(), "L3" | "VAL") && edge_labels.insert(edge.edge.as_str()),
            "RIXS provenance has an unexpected or duplicate edge label"
        );
        anyhow::ensure!(
            is_lowercase_hex(&edge.derived_input_sha256, 64)
                && is_lowercase_hex(&edge.original_gg_sha256, 64)
                && is_lowercase_hex(&edge.sanitized_gg_sha256, 64)
                && edge.original_gg_sha256 != edge.sanitized_gg_sha256,
            "RIXS provenance contains an invalid GG/input digest"
        );
        anyhow::ensure!(
            edge.descriptor_records == EXPECTED_GG_SECTION_COUNT,
            "RIXS provenance records an unexpected GG descriptor count"
        );
    }
    anyhow::ensure!(
        edge_labels == BTreeSet::from(["L3", "VAL"]),
        "RIXS provenance does not contain the L3 and VAL edge pair"
    );
    anyhow::ensure!(
        provenance.val_zero_screen.row_count > 0
            && is_lowercase_hex(&provenance.val_zero_screen.sha256, 64)
            && !provenance.val_zero_screen.derivation.trim().is_empty(),
        "RIXS provenance contains invalid VAL zero-screen metadata"
    );
    anyhow::ensure!(
        provenance.solver.executable == "bin/Seq/rixs"
            && provenance.solver.output_sha256 == crate::manifest::sha256_hex(&map_bytes)
            && provenance.solver.map_order == MAP_ORDER
            && provenance.solver.point_count == MAP_POINT_COUNT
            && provenance.solver.peak_row == validation.peak_row
            && provenance.solver.peak_first_energy_ev == validation.peak_first_energy_ev
            && provenance.solver.peak_second_energy_ev == validation.peak_second_energy_ev
            && provenance.solver.peak_intensity == validation.peak_intensity,
        "RIXS provenance does not match the validated canonical map"
    );
    Ok(())
}

fn is_lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::{
        CURRENT_SOURCE_POLE_INDEX, EXPECTED_GG_SECTION_COUNT, GG_DESCRIPTOR, MAP_ORDER,
        sanitize_native_gg_descriptors, validate_current_source_map,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2};
    use refeff_io::RixsMapData;

    #[test]
    fn gg_normalization_changes_only_bounded_descriptor_records() -> Result<()> {
        let mut input = Vec::new();
        for section in 0..EXPECTED_GG_SECTION_COUNT {
            input.extend_from_slice(format!("#SN#   Section:    {}\n", section + 1).as_bytes());
            input.extend_from_slice(b"#DF# This section written in  [\x93\n.\n");
            input.extend_from_slice(b"#H#\n#DT# 2D complex array with sizes   1   1\n");
            input.extend_from_slice(format!("    0.{section:010}E+00\n").as_bytes());
        }

        let normalized = sanitize_native_gg_descriptors(&input)?;

        assert_eq!(normalized.descriptor_records, EXPECTED_GG_SECTION_COUNT);
        assert_eq!(
            normalized.continuation_lines_removed,
            EXPECTED_GG_SECTION_COUNT
        );
        assert_eq!(
            normalized
                .bytes
                .windows(GG_DESCRIPTOR.len())
                .filter(|window| *window == GG_DESCRIPTOR)
                .count(),
            EXPECTED_GG_SECTION_COUNT
        );
        for section in 0..EXPECTED_GG_SECTION_COUNT {
            let numeric = format!("    0.{section:010}E+00\n");
            assert!(
                normalized
                    .bytes
                    .windows(numeric.len())
                    .any(|window| window == numeric.as_bytes()),
                "numeric matrix row {section} changed"
            );
        }
        Ok(())
    }

    #[test]
    fn gg_normalization_rejects_unbounded_or_data_like_continuations() {
        let mut input = Vec::new();
        for section in 0..EXPECTED_GG_SECTION_COUNT {
            input.extend_from_slice(format!("#SN# {section}\n").as_bytes());
            input.extend_from_slice(b"#DF# This section written in  bad\n");
            if section == 0 {
                input.extend_from_slice(b"    0.1234567890E+00\n#H#\n");
            } else {
                input.extend_from_slice(b"#H#\n");
            }
        }
        let error = sanitize_native_gg_descriptors(&input)
            .expect_err("a numeric-sized continuation must fail closed");
        assert!(error.to_string().contains("unexpectedly long"));
    }

    #[test]
    fn current_source_map_validation_rejects_stale_shift_and_nonphysical_values() -> Result<()> {
        let mut map = current_source_map();
        let expected_peak = CURRENT_SOURCE_POLE_INDEX * MAP_ORDER + CURRENT_SOURCE_POLE_INDEX;
        let validated = validate_current_source_map(&map)?;
        assert_eq!(validated.peak_row, expected_peak);

        map.channels[(expected_peak, 0)] = 0.0;
        let stale_peak = 26 * MAP_ORDER + 26;
        map.channels[(stale_peak, 0)] = 1.0;
        let error = validate_current_source_map(&map)
            .expect_err("the legacy unshifted pole must fail current-source validation");
        assert!(error.to_string().contains("current-source row"));

        let mut map = current_source_map();
        map.channels[(0, 0)] = -1.0;
        let error = validate_current_source_map(&map)
            .expect_err("negative RIXS intensity must fail closed");
        assert!(error.to_string().contains("negative intensity"));

        let mut map = current_source_map();
        map.first_energy_ev[0] = f64::NAN;
        let error =
            validate_current_source_map(&map).expect_err("non-finite axes must fail closed");
        assert!(error.to_string().contains("non-finite energy"));
        Ok(())
    }

    fn current_source_map() -> RixsMapData {
        let point_count = MAP_ORDER * MAP_ORDER;
        let expected_peak = CURRENT_SOURCE_POLE_INDEX * MAP_ORDER + CURRENT_SOURCE_POLE_INDEX;
        let mut channels = Array2::zeros((point_count, 1));
        channels[(expected_peak, 0)] = 0.55;
        RixsMapData {
            header_lines: Vec::new(),
            block_lengths: vec![MAP_ORDER; MAP_ORDER],
            first_energy_ev: Array1::from_shape_fn(point_count, |row| {
                11_547.5 + (row % MAP_ORDER) as f64 * 0.625
            }),
            second_energy_ev: Array1::from_shape_fn(point_count, |row| {
                -15.0 + (row / MAP_ORDER) as f64 * 0.625
            }),
            channels,
        }
    }

    #[test]
    fn malformed_gg_without_all_sections_fails_closed() -> Result<()> {
        let error = sanitize_native_gg_descriptors(
            b"#SN# 1\n#DF# This section written in  bad.\n#H#\n0.0\n",
        )
        .err()
        .context("short native GG should fail")?;
        assert!(error.to_string().contains("expected 205"));
        Ok(())
    }
}
