use ndarray::Array1;
use refeff_core::SfconvSo2convMaterialInput;

use crate::chi_dat::ChiDatData;
use crate::xmu_dat::XmuDatData;

/// FEFF marker written at the top of files already processed by `SO2CONV`.
pub const SFCONV_SO2CONV_CONVOLUTED_MARKER: &str = "# Convoluted with A(omega).";

/// Parsed contents of a FEFF `sfconv.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvInput {
    /// Spectral-function convolution switches.
    pub control: SfconvControl,
    /// Width and center controls.
    pub window: SfconvWindow,
    /// Spectrum type and print flag.
    pub spectrum: SfconvSpectrum,
    /// Convolution filename, or `NULL`.
    pub cfname: String,
}

/// First control line of `sfconv.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfconvControl {
    pub msfconv: i32,
    pub ipse: i32,
    pub ipsk: i32,
}

/// Width and center controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvWindow {
    pub wsigk: f64,
    pub cen: f64,
}

/// Spectrum type and print controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfconvSpectrum {
    pub ispec: i32,
    pub ipr6: i32,
}

/// Kind of FEFF spectrum file selected by `SFCONV/so2conv.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfconvSo2convTargetKind {
    /// `chi.dat` or `chipNNNN.dat` EXAFS-like spectrum columns.
    Chi,
    /// `xmu.dat` XANES spectrum columns.
    Xmu,
    /// `feffNNNN.dat` path file.
    FeffPath,
}

/// One concrete file that FEFF `SO2CONV` would attempt to convolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SfconvSo2convTarget {
    /// FEFF file name as selected by the legacy `so2conv.f90` dispatch logic.
    pub file_name: String,
    /// Spectrum layout expected for the file.
    pub kind: SfconvSo2convTargetKind,
}

/// Header scan result for a FEFF spectrum file consumed by `SO2CONV`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSo2convHeader {
    /// Material constants read from fixed-width FEFF header fields.
    pub material: SfconvSo2convMaterialInput,
    /// Whether FEFF's previous-convolution marker was found before the table.
    pub already_convoluted: bool,
}

/// Parsed contents of one selected `SO2CONV` input file.
#[derive(Debug, Clone, PartialEq)]
pub enum SfconvSo2convTargetData {
    /// `xmu.dat` XANES-style table.
    Xmu {
        /// Header material and previous-convolution status.
        header: SfconvSo2convHeader,
        /// Parsed six-column FEFF `xmu.dat` data.
        data: XmuDatData,
    },
    /// `chi.dat` or `chipNNNN.dat` EXAFS-style table.
    Chi {
        /// Header material and previous-convolution status.
        header: SfconvSo2convHeader,
        /// Parsed FEFF `chi.dat`/`chipNNNN.dat` data.
        data: ChiDatData,
    },
    /// `feffNNNN.dat` path table consumed by `SO2CONV`.
    FeffPath {
        /// Header material and previous-convolution status.
        header: SfconvSo2convHeader,
        /// Parsed seven-column path data plus `reff` metadata.
        data: SfconvSo2convFeffPathData,
    },
}

/// Plain-text `feffNNNN.dat` path data consumed by `SO2CONV`.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSo2convFeffPathData {
    /// Header and metadata lines before the numeric path table.
    pub header_lines: Vec<String>,
    /// Number of legs from the `reff` metadata line.
    pub leg_count: usize,
    /// Path degeneracy from the `reff` metadata line.
    pub degeneracy: f64,
    /// Effective scattering half-path length in Angstrom from the file.
    pub effective_half_path_length_angstrom: f64,
    /// `feffNNNN.dat` path momentum grid in inverse Angstrom, FEFF `xk2`.
    pub wave_number_inverse_angstrom: Array1<f64>,
    /// Central atom phase shift, FEFF `caph2`.
    pub central_phase: Array1<f64>,
    /// Effective scattering amplitude, FEFF `xmfeff2`.
    pub effective_amplitude: Array1<f64>,
    /// Effective scattering phase, FEFF `phfeff2`.
    pub effective_phase: Array1<f64>,
    /// Reduction factor, FEFF `redfac2`.
    pub reduction_factor: Array1<f64>,
    /// Mean free path in Angstrom, FEFF `xlam2`.
    pub mean_free_path_angstrom: Array1<f64>,
    /// Real part of the complex momentum in inverse Angstrom, FEFF `realck2`.
    pub real_momentum_inverse_angstrom: Array1<f64>,
}

impl SfconvSo2convTargetData {
    /// Header scan result for this target data.
    #[must_use]
    pub fn header(&self) -> SfconvSo2convHeader {
        match self {
            Self::Xmu { header, .. } | Self::Chi { header, .. } | Self::FeffPath { header, .. } => {
                *header
            }
        }
    }

    /// Number of numeric spectrum or path rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        match self {
            Self::Xmu { data, .. } => data.point_count(),
            Self::Chi { data, .. } => data.point_count(),
            Self::FeffPath { data, .. } => data.point_count(),
        }
    }
}

impl SfconvSo2convFeffPathData {
    /// Number of path data rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.wave_number_inverse_angstrom.len()
    }
}
