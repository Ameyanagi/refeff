//! Shared FEFF file-format identity and codec contracts.
//!
//! FEFF uses the `.bin` suffix for both formatted PAD text (`pot.bin`,
//! `phase.bin`, `feff.bin`) and byte-oriented payloads (`gg.dat`, `gg.bin`).
//! The registry makes that distinction explicit for inspection and parity
//! tools.

use std::path::Path;

use crate::error::{IoError, Result};

/// Stable identifier for a FEFF-compatible file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FileFormat {
    /// Root FEFF input.
    FeffInput,
    /// Per-potential ATOM diagnostic table.
    AtomDat,
    /// Atomic-potential formatted handoff.
    ApotBin,
    /// Potential-state PAD text.
    PotBin,
    /// Phase-shift PAD text.
    PhaseBin,
    /// Path-amplitude PAD text.
    FeffBin,
    /// FMS metadata PAD text.
    FmsBin,
    /// Unformatted Green's-function matrix.
    GgBin,
    /// XSPH unformatted complex energy mesh.
    EmeshBin,
    /// FF2X unformatted configuration-average accumulator.
    ChiaBin,
    /// Per-potential Green's trace.
    GtrBin,
    /// Magnetic per-potential Green's trace.
    HubbardGtrMBin,
    /// Off-diagonal Hubbard Green's trace.
    HubbardGtrOffBin,
    /// Hubbard on-site potential matrix.
    HubbardVBin,
    /// Hubbard phase-shift matrix.
    HubbardAphaseBin,
    /// Hubbard basis transformation matrices.
    HubbardTransformationBin,
    /// RHORRP pair-block Green's-function slice.
    RhorrpGgSliceBin,
    /// RHORRP diagonal Green's-function matrices.
    RhorrpGgDiagBin,
    /// RHORRP density-grid output.
    RhorrpDensityBin,
    /// NRIXS path-decomposition PAD text.
    FefflBin,
    /// NRIXS transition cross-section PAD text.
    XseclBin,
    /// Absorption spectrum.
    XmuDat,
    /// EXAFS spectrum or per-path contribution.
    ChiDat,
    /// Cross-section table.
    XsectDat,
    /// TDLDA/PMBSE edge table.
    XsedgeDat,
    /// Scattering path list.
    PathsDat,
    /// GENFMT path list.
    ListDat,
    /// BAND result table.
    BandstructureDat,
    /// LDOS table family.
    LdosDat,
    /// Charge-density table family.
    RhocDat,
    /// Magnetic LDOS table family.
    LmdosDat,
    /// Magnetic charge-density table family.
    RhocmDat,
    /// EELS spectrum.
    EelsDat,
    /// EELS mixed dynamic form factor.
    MdffDat,
    /// RIXS map or HERFD spectrum.
    RixsDat,
    /// Compton profile.
    ComptonDat,
    /// Constrained-RPA summary.
    CrpaDat,
    /// Optical loss function.
    LossDat,
    /// Full-spectrum optical constants.
    OpconsDat,
    /// Dynamical-matrix Debye-Waller report.
    DmdwOut,
}

/// Physical representation used by a FEFF format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// Human-readable text containing ordinary numeric fields.
    Text,
    /// Formatted text containing FEFF PAD-encoded arrays.
    PadText,
    /// Compiler-independent unformatted bytes with a typed decoder.
    Binary,
}

/// Numeric comparison envelope for a decoded format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericTolerance {
    /// Relative tolerance.
    pub relative: f64,
    /// Absolute floor near zero.
    pub absolute: f64,
}

/// Static metadata for a registered FEFF format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormatDescriptor {
    /// Stable format identifier.
    pub format: FileFormat,
    /// FEFF module that normally produces the file.
    pub producer: &'static str,
    /// On-disk representation.
    pub representation: Representation,
    /// Default semantic-comparison tolerance.
    pub tolerance: NumericTolerance,
}

const STRICT_TEXT: NumericTolerance = NumericTolerance {
    relative: 1.0e-6,
    absolute: 1.0e-12,
};
const PHASE_TEXT: NumericTolerance = NumericTolerance {
    relative: 5.0e-5,
    absolute: 5.0e-8,
};
const SPECTRUM_TEXT: NumericTolerance = NumericTolerance {
    relative: 5.0e-5,
    absolute: 5.0e-8,
};

/// Codec implemented by typed FEFF input/output models.
pub trait FeffCodec: Sized {
    /// Registered format represented by this type.
    const FORMAT: FileFormat;

    /// Decode a complete file payload.
    fn decode(path: &Path, bytes: &[u8]) -> Result<Self>;

    /// Encode a complete canonical file payload.
    fn encode(&self) -> Result<Vec<u8>>;
}

/// Identify a known FEFF format from its basename.
#[must_use]
pub fn identify_format(path: impl AsRef<Path>) -> Option<FormatDescriptor> {
    let name = path.as_ref().file_name()?.to_str()?;
    let descriptor = match name {
        "feff.inp" => descriptor(FileFormat::FeffInput, "rdinp", Representation::Text),
        "apot.bin" => descriptor(FileFormat::ApotBin, "atomic", Representation::PadText),
        "pot.bin" => descriptor(FileFormat::PotBin, "pot", Representation::PadText),
        "phase.bin" => descriptor(FileFormat::PhaseBin, "xsph", Representation::PadText),
        "feff.bin" => descriptor(FileFormat::FeffBin, "genfmt", Representation::PadText),
        "feffl.bin" => descriptor(FileFormat::FefflBin, "genfmt", Representation::PadText),
        "fms.bin" | "fmsl.bin" => descriptor(FileFormat::FmsBin, "mkgtr", Representation::PadText),
        "gg.dat" | "gg.bin" => descriptor(FileFormat::GgBin, "fms", Representation::Binary),
        "gg_slice.bin" => descriptor(
            FileFormat::RhorrpGgSliceBin,
            "rhorrp",
            Representation::Binary,
        ),
        "gg_diag.bin" => descriptor(
            FileFormat::RhorrpGgDiagBin,
            "rhorrp",
            Representation::Binary,
        ),
        "emesh.bin" => descriptor(FileFormat::EmeshBin, "xsph", Representation::Binary),
        "chia.bin" => descriptor(FileFormat::ChiaBin, "ff2x", Representation::Binary),
        "density.bin" | "valence.bin" => descriptor(
            FileFormat::RhorrpDensityBin,
            "rhorrp",
            Representation::Binary,
        ),
        "v_hubbard.bin" => descriptor(FileFormat::HubbardVBin, "pot", Representation::Binary),
        "aphase_hubbard.bin" => {
            descriptor(FileFormat::HubbardAphaseBin, "xsph", Representation::Binary)
        }
        "transformation_hubbard.bin" => descriptor(
            FileFormat::HubbardTransformationBin,
            "fms",
            Representation::Binary,
        ),
        "xsecl.bin" => descriptor(FileFormat::XseclBin, "xsph", Representation::PadText),
        "xmu.dat" | "xmu1.dat" | "xmu2.dat" => {
            descriptor(FileFormat::XmuDat, "ff2x", Representation::Text)
        }
        "chi.dat" => descriptor(FileFormat::ChiDat, "ff2x", Representation::Text),
        "xsect.dat" => descriptor(FileFormat::XsectDat, "xsph", Representation::Text),
        "xsedge.dat" => descriptor(FileFormat::XsedgeDat, "xsph", Representation::Text),
        "paths.dat" => descriptor(FileFormat::PathsDat, "path", Representation::Text),
        "list.dat" => descriptor(FileFormat::ListDat, "genfmt", Representation::Text),
        "bandstructure.dat" => {
            descriptor(FileFormat::BandstructureDat, "band", Representation::Text)
        }
        "eels.dat" => descriptor(FileFormat::EelsDat, "eels", Representation::Text),
        "mdff.dat" => descriptor(FileFormat::MdffDat, "eelsmdff", Representation::Text),
        "rixsET.dat" | "herfd.dat" | "herfd-sat.dat" => {
            descriptor(FileFormat::RixsDat, "rixs", Representation::Text)
        }
        "compton.dat" => descriptor(FileFormat::ComptonDat, "compton", Representation::Text),
        "crpa.dat" => descriptor(FileFormat::CrpaDat, "crpa", Representation::Text),
        "loss.dat" => descriptor(FileFormat::LossDat, "opconsat", Representation::Text),
        "opcons.dat" => descriptor(FileFormat::OpconsDat, "fullspectrum", Representation::Text),
        "dmdw.out" => descriptor(FileFormat::DmdwOut, "dmdw", Representation::Text),
        _ if indexed_name(name, "chip", ".dat") => {
            descriptor(FileFormat::ChiDat, "ff2x", Representation::Text)
        }
        _ if indexed_name(name, "atom", ".dat") => {
            descriptor(FileFormat::AtomDat, "atomic", Representation::Text)
        }
        _ if indexed_name(name, "feff", ".bin") => {
            descriptor(FileFormat::FeffBin, "genfmt", Representation::PadText)
        }
        _ if indexed_name(name, "phase_", ".bin") => {
            descriptor(FileFormat::PhaseBin, "xsph", Representation::PadText)
        }
        _ if indexed_name(name, "gg_", ".bin") => {
            descriptor(FileFormat::GgBin, "fms", Representation::Binary)
        }
        _ if indexed_name(name, "gtr_m", ".bin") => {
            descriptor(FileFormat::HubbardGtrMBin, "mkgtr", Representation::Binary)
        }
        _ if indexed_name(name, "gtr_off", ".bin") => descriptor(
            FileFormat::HubbardGtrOffBin,
            "mkgtr",
            Representation::Binary,
        ),
        _ if indexed_name(name, "gtr", ".bin") => {
            descriptor(FileFormat::GtrBin, "mkgtr", Representation::Binary)
        }
        _ if indexed_name(name, "ldos", ".dat") => {
            descriptor(FileFormat::LdosDat, "ldos", Representation::Text)
        }
        _ if indexed_name(name, "rhoc", ".dat") => {
            descriptor(FileFormat::RhocDat, "ldos", Representation::Text)
        }
        _ if indexed_name(name, "lmdos", ".dat") => {
            descriptor(FileFormat::LmdosDat, "ldos", Representation::Text)
        }
        _ if indexed_name(name, "rhocm", ".dat") => {
            descriptor(FileFormat::RhocmDat, "ldos", Representation::Text)
        }
        _ => return None,
    };
    Some(descriptor)
}

const fn descriptor(
    format: FileFormat,
    producer: &'static str,
    representation: Representation,
) -> FormatDescriptor {
    FormatDescriptor {
        format,
        producer,
        representation,
        tolerance: match format {
            FileFormat::PhaseBin | FileFormat::XsectDat => PHASE_TEXT,
            FileFormat::XmuDat | FileFormat::ChiDat => SPECTRUM_TEXT,
            _ => STRICT_TEXT,
        },
    }
}

fn indexed_name(name: &str, prefix: &str, suffix: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
}

fn text<'a>(path: &Path, bytes: &'a [u8]) -> Result<&'a str> {
    std::str::from_utf8(bytes).map_err(|error| IoError::Parse {
        path: path.to_path_buf(),
        line: 0,
        message: format!("file is not valid UTF-8: {error}"),
    })
}

macro_rules! text_codec {
    ($type:ty, $format:expr, $parse:path, $render:path) => {
        impl FeffCodec for $type {
            const FORMAT: FileFormat = $format;

            fn decode(path: &Path, bytes: &[u8]) -> Result<Self> {
                $parse(text(path, bytes)?)
            }

            fn encode(&self) -> Result<Vec<u8>> {
                $render(self).map(String::into_bytes)
            }
        }
    };
}

macro_rules! binary_codec {
    ($type:ty, $format:expr, $parse:path, $render:path) => {
        impl FeffCodec for $type {
            const FORMAT: FileFormat = $format;

            fn decode(_path: &Path, bytes: &[u8]) -> Result<Self> {
                $parse(bytes)
            }

            fn encode(&self) -> Result<Vec<u8>> {
                $render(self)
            }
        }
    };
}

text_codec!(
    crate::XmuDatData,
    FileFormat::XmuDat,
    crate::parse_xmu_dat,
    crate::xmu_dat_string
);
text_codec!(
    crate::ChiDatData,
    FileFormat::ChiDat,
    crate::parse_chi_dat,
    crate::chi_dat_string
);
text_codec!(
    crate::XsectDatData,
    FileFormat::XsectDat,
    crate::parse_xsect_dat,
    crate::xsect_dat_string
);
text_codec!(
    crate::PotBinData,
    FileFormat::PotBin,
    crate::parse_pot_bin,
    crate::pot_bin_string
);
text_codec!(
    crate::PhaseBinData,
    FileFormat::PhaseBin,
    crate::parse_phase_bin,
    crate::phase_bin_string
);
text_codec!(
    crate::FeffBinData,
    FileFormat::FeffBin,
    crate::parse_feff_bin,
    crate::feff_bin_string
);
text_codec!(
    crate::FmsBinData,
    FileFormat::FmsBin,
    crate::parse_fms_bin,
    crate::fms_bin_string
);
binary_codec!(
    crate::EmeshBinData,
    FileFormat::EmeshBin,
    crate::parse_emesh_bin,
    crate::emesh_bin_bytes
);
binary_codec!(
    crate::ChiaBinData,
    FileFormat::ChiaBin,
    crate::parse_chia_bin,
    crate::chia_bin_bytes
);
binary_codec!(
    crate::GtrBinData,
    FileFormat::GtrBin,
    crate::parse_gtr_bin,
    crate::gtr_bin_bytes
);
binary_codec!(
    crate::GgDatData,
    FileFormat::GgBin,
    crate::parse_gg_bin_bytes,
    crate::gg_bin_bytes
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinguishes_formatted_and_unformatted_bin_files() {
        assert_eq!(
            identify_format("pot.bin").map(|format| format.representation),
            Some(Representation::PadText)
        );
        assert_eq!(
            identify_format("gg.bin").map(|format| format.representation),
            Some(Representation::Binary)
        );
        assert_eq!(
            identify_format("gtr03.bin").map(|format| format.format),
            Some(FileFormat::GtrBin)
        );
        assert_eq!(
            identify_format("emesh.bin").map(|format| format.format),
            Some(FileFormat::EmeshBin)
        );
        assert_eq!(
            identify_format("gtr_m03.bin").map(|format| format.format),
            Some(FileFormat::HubbardGtrMBin)
        );
        assert_eq!(
            identify_format("gtr_off03.bin").map(|format| format.format),
            Some(FileFormat::HubbardGtrOffBin)
        );
        assert_eq!(
            identify_format("xsecl.bin").map(|format| format.representation),
            Some(Representation::PadText)
        );
        assert_eq!(
            identify_format("feff09.bin").map(|format| format.format),
            Some(FileFormat::FeffBin)
        );
    }

    #[test]
    fn self_describing_binary_codecs_roundtrip() -> Result<()> {
        let emesh = crate::EmeshBinData {
            point_count_declared: 1,
            horizontal_count: 1,
            danes_extension_count: 0,
            energy_hartree: ndarray::arr1(&[num_complex::Complex64::new(1.0, 0.5)]),
        };
        let encoded = <crate::EmeshBinData as FeffCodec>::encode(&emesh)?;
        let decoded = <crate::EmeshBinData as FeffCodec>::decode(Path::new("emesh.bin"), &encoded)?;
        assert_eq!(decoded, emesh);
        Ok(())
    }
}
