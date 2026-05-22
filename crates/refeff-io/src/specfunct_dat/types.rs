use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use refeff_core::{Real, SfconvSo2convXanesPreparation};

use crate::chi_dat::ChiDatData;
use crate::sfconv_input::{SfconvSo2convFeffPathData, SfconvSo2convTargetData};
use crate::xmu_dat::XmuDatData;

pub const SPECFUNCT_DAT_INFO_COLUMNS: usize = 8;
/// Parsed FEFF `specfunct.dat` SO2CONV spectral-function cache.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpecfunctData {
    /// Interstitial Wigner-Seitz radius, FEFF `rs`.
    pub wigner_seitz_radius: f64,
    /// Core-hole lifetime broadening in Hartree, FEFF `gammach`.
    pub core_hole_lifetime: f64,
    /// Asymmetric quasiparticle-phase selector, FEFF `iasym`.
    pub asymmetric_phase: i32,
    /// Satellite approximation selector, FEFF `isattype`.
    pub satellite_type: i32,
    /// Low-q self-energy selector, FEFF `lowq`.
    pub low_q_mode: i32,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies for the full FEFF `nplmax` slot capacity, FEFF `plengy`.
    pub pole_energy: Array1<f64>,
    /// Pole broadenings for the full FEFF `nplmax` slot capacity, FEFF `plbrd`.
    pub pole_broadening: Array1<f64>,
    /// Pole weights for the full FEFF `nplmax` slot capacity, FEFF `plwt`.
    pub pole_weight: Array1<f64>,
    /// Momentum-row metadata table, FEFF `sfinfo(nqpts,8)`.
    pub spectral_info: Array2<f64>,
    /// Eight spectral weights for each momentum row, FEFF `wgts(nqpts,8)`.
    pub weights: Array2<f64>,
    /// Extrinsic quasiparticle table, FEFF `emsf(nqpts,nsfpts)`.
    pub extrinsic_quasiparticle: Array2<f64>,
    /// Extrinsic satellite table, FEFF `essf(nqpts,nsfpts)`.
    pub extrinsic_satellite: Array2<f64>,
    /// Interference quasiparticle table, FEFF `xmsf(nqpts,nsfpts)`.
    pub interference_quasiparticle: Array2<f64>,
    /// Interference satellite table, FEFF `xssf(nqpts,nsfpts)`.
    pub interference_satellite: Array2<f64>,
    /// Intrinsic satellite table, FEFF `xissf(nqpts,nsfpts)`.
    pub intrinsic_satellite: Array2<f64>,
    /// Clipped extrinsic satellite table, FEFF `escsf(nqpts,nsfpts)`.
    pub clipped_extrinsic_satellite: Array2<f64>,
    /// Spectral-function energy table, FEFF `engrid(nqpts,nsfpts)`.
    pub energy_grid: Array2<f64>,
}

impl SfconvSpecfunctData {
    /// Number of pole slots serialized in each FEFF pole record.
    #[must_use]
    pub fn pole_capacity(&self) -> usize {
        self.pole_energy.len()
    }

    /// Number of SO2CONV momentum rows, FEFF `nqpts`.
    #[must_use]
    pub fn momentum_count(&self) -> usize {
        self.spectral_info.nrows()
    }

    /// Number of spectral-function energy rows, FEFF `nsfpts`.
    #[must_use]
    pub fn spectral_point_count(&self) -> usize {
        self.energy_grid.ncols()
    }
}

/// Current SO2CONV inputs used to decide whether a cache can be reused.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctCompatibilityInput<'a> {
    /// Current interstitial Wigner-Seitz radius, FEFF `rs`.
    pub wigner_seitz_radius: f64,
    /// Current core-hole lifetime broadening in Hartree, FEFF `gammach`.
    pub core_hole_lifetime: f64,
    /// Current asymmetric quasiparticle-phase selector, FEFF `iasym`.
    pub asymmetric_phase: i32,
    /// Current satellite approximation selector, FEFF `isattype`.
    pub satellite_type: i32,
    /// Current low-q self-energy selector, FEFF `lowq`.
    pub low_q_mode: i32,
    /// Number of active current poles, FEFF `npl`.
    pub pole_count: usize,
    /// Current pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, f64>,
    /// Current pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, f64>,
    /// Current pole weights, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, f64>,
    /// Current minimal SO2CONV momentum grid, FEFF `pgrid`.
    pub momentum_grid: ArrayView1<'a, f64>,
}

/// Inputs for convolving EXAFS rows with a `specfunct.dat` cache.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctExafsRowsInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Signal energy grid, FEFF `epts2`.
    pub signal_energy: ArrayView1<'a, Real>,
    /// Real EXAFS channel, FEFF `chir`.
    pub real_signal: ArrayView1<'a, Real>,
    /// Imaginary EXAFS channel, FEFF `chii`.
    pub imaginary_signal: ArrayView1<'a, Real>,
    /// Original EXAFS magnitude, FEFF `xmag`.
    pub original_magnitude: ArrayView1<'a, Real>,
    /// Original EXAFS phase, FEFF `phase`.
    pub original_phase: ArrayView1<'a, Real>,
    /// Original phase with `2 k R` removed, FEFF `phm2kr`.
    pub phase_minus_2kr: ArrayView1<'a, Real>,
    /// Photoelectron momentum for each active signal row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Number of target rows to convolve.
    pub active_len: usize,
    /// EXAFS convolution chemical potential, FEFF `cmu`.
    pub chemical_potential: Real,
    /// Apply FEFF's available-energy cutoff, FEFF `icut`.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for convolving prepared XANES rows with a `specfunct.dat` cache.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctXanesRowsInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Prepared padded XANES arrays from the core SO2CONV signal-preparation step.
    pub prepared: &'a SfconvSo2convXanesPreparation,
    /// Photoelectron momentum for each active signal row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Number of target rows to convolve.
    pub active_len: usize,
    /// XANES convolution chemical potential, FEFF `cmu + vint`.
    pub chemical_potential: Real,
    /// Apply FEFF's available-energy cutoff, FEFF `icut`.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for applying a cached `specfunct.dat` convolution to one `chi.dat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctChiDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Source `chi.dat` or `chipNNNN.dat` rows before many-body convolution.
    pub source: &'a ChiDatData,
    /// Material constants scanned from the source FEFF spectrum header.
    pub material: refeff_core::SfconvSo2convMaterialInput,
    /// Corrected photoelectron momentum for each source row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's padded EXAFS work arrays, FEFF `npts2`.
    pub work_len: usize,
}

/// Inputs for applying a cached `specfunct.dat` convolution to one `feffNNNN.dat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctFeffPathDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Source `feffNNNN.dat` path rows before many-body convolution.
    pub source: &'a SfconvSo2convFeffPathData,
    /// Material constants scanned from the source FEFF spectrum header.
    pub material: refeff_core::SfconvSo2convMaterialInput,
    /// Corrected photoelectron momentum for each dense uniform path row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's dense uniform path work arrays, FEFF `npts2`.
    pub work_len: usize,
}

/// Inputs for dispatching cached `specfunct.dat` convolution by target type.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctTargetDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Parsed target selected by FEFF `SO2CONV`.
    pub source: &'a SfconvSo2convTargetData,
    /// Corrected photoelectron momentum on the target's active grid, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's padded work arrays, FEFF `npts2`.
    pub work_len: usize,
}

/// Inputs for applying a cached `specfunct.dat` convolution to one `xmu.dat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctXmuDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Source `xmu.dat` rows before many-body convolution.
    pub source: &'a XmuDatData,
    /// Material constants scanned from the source FEFF spectrum header.
    pub material: refeff_core::SfconvSo2convMaterialInput,
    /// Corrected photoelectron momentum for each source row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's padded XANES work arrays, FEFF `npts2`.
    pub work_len: usize,
}

/// Inputs for assembling a FEFF `specfunct.dat` cache from finalized spectral rows.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctSpectralRowsInput<'a> {
    /// Interstitial Wigner-Seitz radius, FEFF `rs`.
    pub wigner_seitz_radius: f64,
    /// Core-hole lifetime broadening in Hartree, FEFF `gammach`.
    pub core_hole_lifetime: f64,
    /// Asymmetric quasiparticle-phase selector, FEFF `iasym`.
    pub asymmetric_phase: i32,
    /// Satellite approximation selector, FEFF `isattype`.
    pub satellite_type: i32,
    /// Low-q self-energy selector, FEFF `lowq`.
    pub low_q_mode: i32,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies for the full FEFF `nplmax` slot capacity, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, f64>,
    /// Pole broadenings for the full FEFF `nplmax` slot capacity, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, f64>,
    /// Pole weights for the full FEFF `nplmax` slot capacity, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, f64>,
    /// Momentum-row metadata table, FEFF `sfinfo(nqpts,8)`.
    pub spectral_info: ArrayView2<'a, f64>,
    /// Eight spectral weights for each momentum row, FEFF `wgts(nqpts,8)`.
    pub weights: ArrayView2<'a, f64>,
    /// Finalized FEFF `mkspectf` rows, shaped as `(nqpts,8,nsfpts)`.
    pub spectral_function: ArrayView3<'a, f64>,
    /// Spectral-function energy table, FEFF `engrid(nqpts,nsfpts)`.
    pub energy_grid: ArrayView2<'a, f64>,
}
