//! FEFF SFCONV numerical helpers.
//!
//! These kernels support spectral-function convolution. The full SFCONV driver
//! also depends on spectrum file orchestration, so this module keeps the
//! reusable numerical transforms independent and directly testable.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Real, RealVec, RootError, real_polynomial_roots};

const SFCONV_GRATER_MAX_REGIONS: usize = 1_500;
const SFCONV_GRATER_MAX_SINGULARITIES: usize = 20;
/// Legacy Hartree/eV conversion used inside FEFF `SFCONV/so2conv.f90`.
pub const SFCONV_SO2CONV_HARTREE_EV: Real = 27.21160;
/// Legacy Bohr/Angstrom conversion used inside FEFF `SFCONV/so2conv.f90`.
pub const SFCONV_SO2CONV_BOHR_ANGSTROM: Real = 0.529_177_06;
/// Number of energy rows in FEFF `SFCONV/mkspectf.f90` spectral functions.
pub const SFCONV_MKSPECTF_GRID_LEN: usize = 112;
/// Number of FEFF `SFCONV/so2conv.f90` minimal momentum-grid rows.
pub const SFCONV_SO2CONV_MOMENTUM_GRID_LEN: usize = 66;
const SFCONV_GRATER_DX: [Real; 3] = [
    0.112_701_66_f32 as Real,
    0.5_f32 as Real,
    0.887_298_35_f32 as Real,
];
const SFCONV_GRATER_WT: [Real; 3] = [
    0.277_777_8_f32 as Real,
    0.444_444_45_f32 as Real,
    0.277_777_8_f32 as Real,
];
const SFCONV_GRATER_WT9: [Real; 9] = [
    0.061_693_88_f32 as Real,
    0.108_384_23_f32 as Real,
    0.039_846_36_f32 as Real,
    0.175_209_03_f32 as Real,
    0.229_732_99_f32 as Real,
    0.175_209_03_f32 as Real,
    0.039_846_36_f32 as Real,
    0.108_384_23_f32 as Real,
    0.061_693_88_f32 as Real,
];

/// Inputs for FEFF `SFCONV/mkrmu.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvKramersKronigInput<'a> {
    /// Imaginary part of the spectrum-dependent function, FEFF `xmu`.
    pub imaginary: ArrayView1<'a, Real>,
    /// Reference imaginary part to subtract before the transform, FEFF `xmu0`.
    pub reference_imaginary: ArrayView1<'a, Real>,
    /// Energy grid, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Number of active rows, FEFF `npts`.
    pub active_len: usize,
}

/// Inputs for FEFF `SFCONV/sfconvsub.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvConvolutionInput<'a> {
    /// Photoelectron energy neglecting collective excitations, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Chemical potential / edge position, FEFF `mu`.
    pub chemical_potential: Real,
    /// Core-hole lifetime width, FEFF `gammach`.
    pub core_hole_lifetime: Real,
    /// Signal energy grid, FEFF `wpts2`.
    pub signal_energy: ArrayView1<'a, Real>,
    /// Signal values on `signal_energy`, FEFF `xchi`.
    pub signal: ArrayView1<'a, Real>,
    /// Spectral-function energy grid, FEFF `wpts1`.
    pub spectral_energy: ArrayView1<'a, Real>,
    /// Spectral function values, FEFF `spectf`.
    pub spectral_function: ArrayView1<'a, Real>,
    /// FEFF eight-slot spectral weights array.
    pub weights: ArrayView1<'a, Real>,
    /// Include quasiparticle phase as an asymmetric `1 / omega` term.
    pub asymmetric_phase: bool,
    /// Apply FEFF's available-energy cutoff.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for FEFF `SFCONV/interpsf.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralInterpolationInput<'a> {
    /// Minimal spectral-function energy grid, FEFF `epts`.
    pub energy: ArrayView1<'a, Real>,
    /// Eight-row spectral-function table, FEFF `spectf(row, point)`.
    pub spectral_function: ArrayView2<'a, Real>,
    /// Number of rows in the uniform output grid, FEFF `npts`.
    pub output_len: usize,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` momentum-grid spectral interpolation.
#[derive(Debug, Clone, Copy)]
pub struct SfconvMomentumSpectralInterpolationInput<'a> {
    /// Photoelectron momentum for the current signal row, FEFF `pk(jj)`.
    pub photoelectron_momentum: Real,
    /// Minimal SO2CONV momentum grid, FEFF `pgrid`.
    pub momentum_grid: ArrayView1<'a, Real>,
    /// Spectral energy tables on `momentum_grid`, FEFF `engrid`.
    pub energy_grid: ArrayView2<'a, Real>,
    /// Extrinsic quasiparticle row tables, FEFF `emsf`.
    pub extrinsic_quasiparticle: ArrayView2<'a, Real>,
    /// Extrinsic satellite row tables, FEFF `essf`.
    pub extrinsic_satellite: ArrayView2<'a, Real>,
    /// Interference quasiparticle row tables, FEFF `xmsf`.
    pub interference_quasiparticle: ArrayView2<'a, Real>,
    /// Interference satellite row tables, FEFF `xssf`.
    pub interference_satellite: ArrayView2<'a, Real>,
    /// Intrinsic satellite row tables, FEFF `xissf`.
    pub intrinsic_satellite: ArrayView2<'a, Real>,
    /// Clipped extrinsic satellite row tables, FEFF `escsf`.
    pub clipped_extrinsic_satellite: ArrayView2<'a, Real>,
    /// Eight FEFF spectral weights on each momentum row, FEFF `wgts`.
    pub weights: ArrayView2<'a, Real>,
    /// Real self-energy table on `momentum_grid`, FEFF `sfinfo(:,4)`.
    pub self_energy_real: ArrayView1<'a, Real>,
    /// Energy correction table on `momentum_grid`, FEFF `sfinfo(:,5)`.
    pub energy_correction: ArrayView1<'a, Real>,
    /// Width table on `momentum_grid`, FEFF `sfinfo(:,6)`.
    pub width: ArrayView1<'a, Real>,
    /// Real renormalization table on `momentum_grid`, FEFF `sfinfo(:,7)`.
    pub renormalization_real: ArrayView1<'a, Real>,
    /// Imaginary renormalization table on `momentum_grid`, FEFF `sfinfo(:,8)`.
    pub renormalization_imag: ArrayView1<'a, Real>,
}

/// Selected FEFF SFCONV pole parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvPole {
    /// Pole energy, FEFF `ompl`.
    pub energy: Real,
    /// Pole weight, FEFF `wt`.
    pub weight: Real,
    /// Pole broadening, FEFF `brd`.
    pub broadening: Real,
}

/// Electron-gas parameters produced by FEFF `SFCONV/ppset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvPlasmaParameters {
    /// Fermi momentum, FEFF `qf`.
    pub fermi_momentum: Real,
    /// Fermi energy, FEFF `ef`.
    pub fermi_energy: Real,
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// FEFF output-header values consumed by `SFCONV/so2conv.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSo2convMaterialInput {
    /// FEFF `Gam_ch` header value in eV.
    pub core_hole_width_ev: Real,
    /// Interstitial Wigner-Seitz radius, FEFF `Rs_int`.
    pub wigner_seitz_radius: Real,
    /// Interstitial potential header value in eV, FEFF `Vint`.
    pub interstitial_potential_ev: Real,
    /// Chemical-potential header value in eV, FEFF `Mu`.
    pub chemical_potential_ev: Real,
    /// Fermi wave number header value in inverse Angstrom, FEFF `kf`.
    pub fermi_wave_number_inv_angstrom: Real,
}

/// FEFF `SO2CONV` material constants after legacy unit conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSo2convMaterialParameters {
    /// Core-hole lifetime broadening in Hartree, FEFF `gammach`.
    pub core_hole_lifetime: Real,
    /// Interstitial potential in Hartree, FEFF `vint`.
    pub interstitial_potential: Real,
    /// Chemical potential offset from `Vint` in Hartree, FEFF `cmu`.
    pub chemical_potential_offset: Real,
    /// Header Fermi wave number in atomic units, FEFF `ckf`.
    pub fermi_wave_number: Real,
    /// Free-electron-gas Fermi momentum from `Rs_int`, FEFF `qf`.
    pub fermi_momentum: Real,
    /// Free-electron-gas Fermi energy, FEFF `ef`.
    pub fermi_energy: Real,
    /// Electron concentration, FEFF `conc`.
    pub electron_concentration: Real,
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Pole-dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Initial photoelectron energy assigned before pole loops, FEFF `ekp`.
    pub initial_photoelectron_energy: Real,
    /// Initial photoelectron momentum assigned before pole loops, FEFF `qpk`.
    pub initial_photoelectron_momentum: Real,
    /// FEFF global SO2CONV relative accuracy, `acc`.
    pub accuracy: Real,
}

/// Limiting momentum values produced by FEFF `SFCONV/qlimits.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvQLimits {
    /// Number of active limiting values, FEFF `nq`.
    pub count: usize,
    /// First limiting value, FEFF `q1`.
    pub q1: Real,
    /// Second limiting value, FEFF `q2`.
    pub q2: Real,
    /// Third limiting value, FEFF `q3`.
    pub q3: Real,
}

/// Result from FEFF `SFCONV/grater.f90` adaptive quadrature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvAdaptiveIntegral {
    /// Accumulated real integral value.
    pub value: Real,
    /// FEFF `error`: accumulated absolute difference between local estimates.
    pub estimated_error: Real,
    /// FEFF `numcal`: number of integrand evaluations.
    pub evaluations: usize,
    /// FEFF `maxns`: maximum number of active regions on the stack.
    pub max_regions: usize,
}

/// Shared pole/plasma context for FEFF `SFCONV/mksat.f90` helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatelliteContext {
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Pole energy, FEFF `ompl`.
    pub pole_energy: Real,
    /// Pole dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Bare photoelectron kinetic energy, FEFF `ek`.
    pub photoelectron_energy: Real,
    /// Global relative accuracy parameter, FEFF `acc`.
    pub accuracy: Real,
}

/// Shared electron-gas context for FEFF `SFCONV/senergies.f90` beta helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSelfEnergyContext {
    /// Fermi energy, FEFF `ef`.
    pub fermi_energy: Real,
    /// Fermi momentum, FEFF `qf`.
    pub fermi_momentum: Real,
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Active pole energy, FEFF `ompl`.
    pub pole_energy: Real,
    /// Photoelectron quasiparticle energy, FEFF `ekp`.
    pub quasiparticle_energy: Real,
    /// Photoelectron momentum, FEFF `pk`.
    pub photoelectron_momentum: Real,
    /// Global relative accuracy parameter, FEFF `acc`.
    pub accuracy: Real,
    /// Pole broadening, FEFF `brd`.
    pub pole_broadening: Real,
    /// Pole dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Include below-Fermi contributions, FEFF common block `belowqf`.
    pub include_below_fermi: bool,
}

/// FEFF `SFCONV/mksat.f90` self-energy state from common block `energies`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatelliteSelfEnergy {
    /// Real part of the on-shell self energy, FEFF `se`.
    pub on_shell_real: Real,
    /// Quasiparticle broadening, FEFF `width`.
    pub width: Real,
    /// Real part of the renormalization constant, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub renormalization_imag: Real,
    /// Real part of the self energy at the current energy, FEFF `se2`.
    pub off_shell_real: Real,
    /// Imaginary part of the self energy at the current energy, FEFF `xise`.
    pub off_shell_imag: Real,
}

/// Result from an integrated FEFF `SFCONV/mksat.f90` satellite helper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatelliteIntegral {
    /// Accumulated satellite value.
    pub value: Real,
    /// Sum of FEFF `grater` local error estimates.
    pub estimated_error: Real,
    /// Total integrand evaluations across FEFF `grater` calls.
    pub evaluations: usize,
    /// Maximum active FEFF `grater` stack size across calls.
    pub max_regions: usize,
}

/// Magnitude and phase produced by FEFF `SFCONV/sfconvsub.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvConvolution {
    /// Magnitude of the convoluted signal, FEFF `cchi`.
    pub amplitude: Real,
    /// Phase of the convoluted signal, FEFF `phase`.
    pub phase: Real,
}

/// Uniform spectral function produced by FEFF `SFCONV/interpsf.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpectralInterpolation {
    /// Uniform energy grid, FEFF `wpts`.
    pub energy: RealVec,
    /// Interpolated spectral function, FEFF `cspec`.
    pub spectral_function: RealVec,
}

/// FEFF spectral-function rows interpolated to one photoelectron momentum.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvMomentumSpectralInterpolation {
    /// Interpolated spectral energy grid, FEFF `epts`.
    pub energy: RealVec,
    /// Eight-row spectral-function table, FEFF `spectf(row, point)`.
    pub spectral_function: Array2<Real>,
    /// Interpolated eight-slot spectral weights, FEFF `weights`.
    pub weights: RealVec,
    /// Real self-energy, FEFF `se`.
    pub self_energy_real: Real,
    /// Energy correction, FEFF `ce`.
    pub energy_correction: Real,
    /// Spectral width, FEFF `width`.
    pub width: Real,
    /// Real renormalization value, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary renormalization value, FEFF `z1i`.
    pub renormalization_imag: Real,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` photoelectron momentum refinement.
#[derive(Debug, Clone, Copy)]
pub struct SfconvPhotoelectronMomentumInput<'a> {
    /// FEFF wavenumber grid `xk`, in atomic units.
    pub momentum: ArrayView1<'a, Real>,
    /// FEFF chemical potential offset `cmu`.
    pub chemical_potential: Real,
    /// Fermi momentum, FEFF `qf`.
    pub fermi_momentum: Real,
    /// Self-consistent Fermi level, FEFF `fmu`.
    pub fermi_level: Real,
    /// Self energy at the Fermi level, FEFF `sef0`.
    pub fermi_self_energy: Real,
    /// Zeroth-order self-energy samples, FEFF `seg`.
    pub self_energy: ArrayView1<'a, Real>,
}

/// FEFF `SO2CONV` momentum arrays used for spectral-function interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvPhotoelectronMomentum {
    /// Photoelectron kinetic energy, FEFF `ekpg`.
    pub kinetic_energy: RealVec,
    /// Zeroth-order photoelectron momentum estimate, FEFF `xpkg`.
    pub zero_order_momentum: RealVec,
    /// Momentum-derivative renormalization factor, FEFF `zkk`.
    pub renormalization: RealVec,
    /// Corrected photoelectron momentum, FEFF `pk`.
    pub photoelectron_momentum: RealVec,
}

/// Inputs for one SO2CONV weighted pole self-energy sample.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSo2convSelfEnergySampleInput<'a> {
    /// FEFF material and electron-gas constants used by `SO2CONV`.
    pub material: SfconvSo2convMaterialParameters,
    /// Energy argument passed to FEFF `renergies` or `brsigma`; `SO2CONV` uses zero.
    pub energy: Real,
    /// Photoelectron quasiparticle energy, FEFF `ekp`.
    pub quasiparticle_energy: Real,
    /// Photoelectron momentum for this sample, FEFF `qpk` or `pk`.
    pub photoelectron_momentum: Real,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
    /// Pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, Real>,
    /// Include below-Fermi terms, FEFF common block `lowq`.
    pub include_below_fermi: bool,
}

/// Inputs for the SO2CONV self-energy samples used before momentum refinement.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSo2convSelfEnergyGridInput<'a> {
    /// FEFF input momentum grid `xk`, already converted to atomic units.
    pub momentum: ArrayView1<'a, Real>,
    /// Chemical-potential offset from the interstitial potential, FEFF `cmu`.
    pub chemical_potential: Real,
    /// Self-consistent Fermi level, FEFF `fmu`.
    pub fermi_level: Real,
    /// FEFF material and electron-gas constants used by `SO2CONV`.
    pub material: SfconvSo2convMaterialParameters,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
    /// Pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, Real>,
    /// Include below-Fermi terms, FEFF common block `lowq`.
    pub include_below_fermi: bool,
}

/// SO2CONV self-energy samples that feed photoelectron-momentum refinement.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSo2convSelfEnergyGrid {
    /// Photoelectron kinetic energy neglecting collective excitations, FEFF `ekpg`.
    pub kinetic_energy: RealVec,
    /// Zeroth-order photoelectron momentum estimate, FEFF `xpkg`.
    pub zero_order_momentum: RealVec,
    /// Real self-energy samples at the zeroth-order momenta, FEFF `seg`.
    pub self_energy: RealVec,
    /// Real self-energy at the Fermi momentum, FEFF `sef0`.
    pub fermi_self_energy: Real,
}

/// FEFF `brsigma` log/atan integrand family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfconvBroadenedSelfEnergyBranch {
    /// FEFF `fqlog*1`/`fqatn*1`: particle denominator on both momentum sides.
    ParticlePair,
    /// FEFF `fqlog*2`/`fqatn*2`: particle denominator against the Fermi level.
    ParticleFermi,
    /// FEFF `fqlog*3`/`fqatn*3`: below-Fermi denominator against the Fermi level.
    HoleFermi,
    /// FEFF `fqlog*4`/`fqatn*4`: below-Fermi denominator on both momentum sides.
    HolePair,
}

/// Inputs for one FEFF `brsigma` log/atan integrand evaluation.
#[derive(Debug, Clone, Copy)]
pub struct SfconvBroadenedSelfEnergyIntegrandInput {
    /// Integration momentum `q`.
    pub momentum: Real,
    /// Self-energy frequency argument `w`; FEFF combines it with `ekp` as `wp`.
    pub energy: Real,
    /// Active-pole electron-gas context.
    pub context: SfconvSelfEnergyContext,
}

/// Four broadened `brsigma` integrands for one branch and momentum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvBroadenedSelfEnergyIntegrands {
    /// FEFF `fqlogrN`.
    pub log_real: Real,
    /// FEFF `fqlogiN`.
    pub log_imag: Real,
    /// FEFF `fqatnrN`.
    pub atan_real: Real,
    /// FEFF `fqatniN`.
    pub atan_imag: Real,
}

/// Four FEFF `dbrsigma` derivative integrands for one branch and momentum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvBroadenedSelfEnergyDerivativeIntegrands {
    /// FEFF `dqlogrN`.
    pub log_real: Real,
    /// FEFF `dqlogiN`.
    pub log_imag: Real,
    /// FEFF `dqatnrN`.
    pub atan_real: Real,
    /// FEFF `dqatniN`.
    pub atan_imag: Real,
}

/// Broad Lorentzian-pole self-energy returned by FEFF `brsigma`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvBroadenedSelfEnergy {
    /// Real part of the broadened self energy, FEFF `rbeta`.
    pub real: Real,
    /// Imaginary part of the broadened self energy, FEFF `xibeta`.
    pub imaginary: Real,
    /// Accumulated absolute quadrature error estimate for [`Self::real`].
    pub real_estimated_error: Real,
    /// Accumulated absolute quadrature error estimate for [`Self::imaginary`].
    pub imaginary_estimated_error: Real,
    /// Total number of FEFF `grater` integrand evaluations.
    pub evaluations: usize,
    /// Largest FEFF `grater` active-region stack seen in any component.
    pub max_regions: usize,
}

/// Energy derivative of the broad Lorentzian-pole self-energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvBroadenedSelfEnergyDerivative {
    /// Real part of FEFF `dbrsigma`, `drbeta`.
    pub real: Real,
    /// Imaginary part of FEFF `dbrsigma`, `dibeta`.
    pub imaginary: Real,
    /// Accumulated absolute quadrature error estimate for [`Self::real`].
    pub real_estimated_error: Real,
    /// Accumulated absolute quadrature error estimate for [`Self::imaginary`].
    pub imaginary_estimated_error: Real,
    /// Total number of FEFF `grater` integrand evaluations.
    pub evaluations: usize,
    /// Largest FEFF `grater` active-region stack seen in any component.
    pub max_regions: usize,
}

/// FEFF `mkspectf` complex renormalization factor from self-energy slopes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvRenormalization {
    /// Real part of the renormalization constant, FEFF `z1`.
    pub real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub imaginary: Real,
    /// Magnitude of the renormalization constant, FEFF `z1m`.
    pub magnitude: Real,
}

/// Inputs for FEFF `SFCONV/mkspectf.f90` pole-reduction factor.
#[derive(Debug, Clone, Copy)]
pub struct SfconvExponentialReductionInput<'a> {
    /// Plasma frequency scale, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `SFCONV/mkspectf.f90` quasiparticle pole refinement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvQuasiparticlePoleInput {
    /// Photoelectron quasiparticle energy before final pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// On-shell broadening before renormalization, FEFF `width`.
    pub width: Real,
    /// Complex self-energy renormalization, FEFF `z1`, `z1i`, and `z1m`.
    pub renormalization: SfconvRenormalization,
}

/// Refined FEFF `mkspectf` quasiparticle pole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvQuasiparticlePole {
    /// Refined quasiparticle pole energy, FEFF `qpengy`.
    pub energy: Real,
    /// Refined quasiparticle pole width, FEFF `qpwidth`.
    pub width: Real,
}

/// FEFF `SFCONV/mkspectf.f90` spectral energy mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpectralEnergyGrid {
    /// Spectral-function center energies, FEFF `wpts`.
    pub energy: RealVec,
    /// Cell boundaries used for finite-element widths, FEFF `wlim(0:npts)`.
    pub boundaries: RealVec,
}

/// Inputs for the FEFF `SFCONV/mkspectf.f90` quasiparticle peak bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvQuasiparticlePeakInput {
    /// Center energy of the spectral-function cell, FEFF `wpts(i)`.
    pub center_energy: Real,
    /// Lower finite-element boundary, FEFF `wlim(i-1)`.
    pub lower_boundary: Real,
    /// Upper finite-element boundary, FEFF `wlim(i)`.
    pub upper_boundary: Real,
    /// Photoelectron quasiparticle energy before pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Refined quasiparticle pole energy, FEFF `qpengy`.
    pub quasiparticle_energy: Real,
    /// Refined quasiparticle pole width, FEFF `qpwidth`.
    pub quasiparticle_width: Real,
    /// Plasma frequency scale, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Real part of the renormalization constant, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub renormalization_imag: Real,
}

/// Inputs for FEFF `SFCONV/mkspectf.f90` quasiparticle rows.
#[derive(Debug, Clone, Copy)]
pub struct SfconvQuasiparticleTableInput<'a> {
    /// Spectral-function center energies, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Finite-element cell boundaries, FEFF `wlim(0:npts)`.
    pub boundaries: ArrayView1<'a, Real>,
    /// Photoelectron quasiparticle energy before pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Refined quasiparticle pole energy, FEFF `qpengy`.
    pub quasiparticle_energy: Real,
    /// On-shell broadening before renormalization, FEFF `width`.
    pub endpoint_width: Real,
    /// Refined quasiparticle pole width, FEFF `qpwidth`.
    pub quasiparticle_width: Real,
    /// Plasma frequency scale, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Real part of the renormalization constant, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub renormalization_imag: Real,
    /// Magnitude of the renormalization constant, FEFF `zm`.
    pub renormalization_magnitude: Real,
    /// Interference quasiparticle amplitude after reduction, FEFF `ak`.
    pub interference_amplitude: Real,
    /// Exponential reduction factor, FEFF `expa`.
    pub exponential_reduction: Real,
}

/// Inputs for FEFF `mkspectf` quasiparticle-interference amplitude `ak`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvQuasiparticleInterferenceInput<'a> {
    /// Refined quasiparticle energy passed to FEFF `xmkak`, normally `ekp`.
    pub quasiparticle_energy: Real,
    /// Highest relative spectral energy, FEFF `wmax`.
    pub upper_energy: Real,
    /// Bare photoelectron kinetic energy, FEFF `ek`.
    pub bare_photoelectron_energy: Real,
    /// Plasma frequency scale, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Pole dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Global relative accuracy parameter, FEFF `acc`.
    pub accuracy: Real,
    /// FEFF ad-hoc interference reduction factor, FEFF `xreduc`.
    pub interference_reduction: Real,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
}

/// Weighted FEFF `mkspectf` quasiparticle-interference amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvQuasiparticleInterference {
    /// FEFF `ak` after the `xreduc` and pole-weight factors.
    pub amplitude: Real,
    /// Accumulated quadrature error estimate after the same weights.
    pub estimated_error: Real,
    /// Total FEFF `grater` integrand evaluations across active poles.
    pub evaluations: usize,
    /// Largest FEFF `grater` active-region stack seen in any pole.
    pub max_regions: usize,
}

/// FEFF `mkspectf` quasiparticle and interference rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvQuasiparticleTable {
    /// Extrinsic quasiparticle row, FEFF `spectf(1,:)`.
    pub main_peak: RealVec,
    /// Interference quasiparticle row, FEFF `spectf(3,:)`.
    pub interference_peak: RealVec,
    /// Endpoint-corrected integral accumulated as FEFF `wtemain`.
    pub integrated_main_weight: Real,
    /// Endpoint-corrected integral accumulated as FEFF `wtxmain`.
    pub integrated_interference_weight: Real,
}

/// Inputs for FEFF `SFCONV/mkspectf.f90` satellite row assembly.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSatelliteTableInput<'a> {
    /// Extrinsic quasiparticle row, FEFF `spectf(1,:)`.
    pub main_peak: ArrayView1<'a, Real>,
    /// Interference quasiparticle row, FEFF `spectf(3,:)`.
    pub quasiparticle_interference: ArrayView1<'a, Real>,
    /// Extrinsic satellite row before endpoint averaging, FEFF `esat`.
    pub extrinsic_satellite: ArrayView1<'a, Real>,
    /// Interference satellite row, FEFF `xsat`.
    pub interference_satellite: ArrayView1<'a, Real>,
    /// Intrinsic satellite row, FEFF `xisat`.
    pub intrinsic_satellite: ArrayView1<'a, Real>,
    /// Finite-element cell boundaries, FEFF `wlim(0:npts)`.
    pub boundaries: ArrayView1<'a, Real>,
    /// FEFF one-based lower quasiparticle column, normally `iqpl`.
    pub quasiparticle_lower_column_1based: usize,
    /// FEFF one-based upper quasiparticle column, normally `iqph`.
    pub quasiparticle_upper_column_1based: usize,
    /// Include FEFF `isattype.eq.3` quasiparticle interference in row 6.
    pub include_full_broadening_quasiparticle: bool,
    /// Exponential reduction factor, FEFF `expa`.
    pub exponential_reduction: Real,
}

/// Inputs for the FEFF `mkspectf` pole loop that builds `xsat` and `xisat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSatellitePoleContributionsInput<'a> {
    /// Relative satellite energy, FEFF `w-ekp`.
    pub energy: Real,
    /// Uniform mesh spacing used to choose local broadenings, FEFF `dw`.
    pub uniform_width: Real,
    /// Quasiparticle width added for FEFF `isattype.eq.3`, FEFF `width`.
    pub quasiparticle_width: Real,
    /// Plasma frequency scale, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Bare photoelectron kinetic energy, FEFF `ek`.
    pub bare_photoelectron_energy: Real,
    /// Pole dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Global relative accuracy parameter, FEFF `acc`.
    pub accuracy: Real,
    /// FEFF ad-hoc interference reduction factor, FEFF `xreduc`.
    pub interference_reduction: Real,
    /// Add quasiparticle width to satellite broadenings, FEFF `isattype.eq.3`.
    pub include_full_broadening: bool,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
    /// Pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, Real>,
}

/// Weighted FEFF `mkspectf` pole-loop satellite contributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatellitePoleContributions {
    /// Interference satellite value, FEFF `xsat`.
    pub interference_satellite: Real,
    /// Intrinsic satellite value, FEFF `xisat`.
    pub intrinsic_satellite: Real,
    /// Weighted quadrature error for the interference contribution.
    pub interference_estimated_error: Real,
    /// Weighted quadrature error for the intrinsic contribution.
    pub intrinsic_estimated_error: Real,
    /// Total FEFF `grater` integrand evaluations across active poles.
    pub evaluations: usize,
    /// Largest FEFF `grater` active-region stack seen in any pole.
    pub max_regions: usize,
}

/// FEFF `mkspectf` extrinsic satellite approximation selected by `isattype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfconvExtrinsicSatelliteMode {
    /// FEFF `isattype.eq.1`: full broadening with the quasiparticle peak removed.
    BroadenedMinusMain,
    /// FEFF `isattype.eq.2`: local derivative expansion near the quasiparticle.
    DerivativeExpansion,
    /// FEFF `isattype.eq.3`: full broadening including quasiparticle structure.
    FullBroadening,
    /// FEFF default branch: de-broadened extrinsic satellite, `xmkesat`.
    Debroadened,
}

/// Inputs for FEFF `mkspectf` extrinsic satellite branch selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvExtrinsicSatelliteInput {
    /// Relative satellite energy, FEFF `w-ekp`.
    pub energy: Real,
    /// Extrinsic quasiparticle peak for this cell, FEFF `emain`.
    pub main_peak: Real,
    /// Imaginary self-energy derivative, FEFF `xaa`.
    pub imaginary_derivative: Real,
    /// FEFF `isattype` branch to use.
    pub mode: SfconvExtrinsicSatelliteMode,
    /// Active pole/plasma context used by FEFF `xmkesat`.
    pub context: SfconvSatelliteContext,
    /// On/off-shell self-energy state used by the extrinsic satellite formulas.
    pub self_energy: SfconvSatelliteSelfEnergy,
}

/// Inputs for one FEFF `mkspectf` spectral-function cell.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralCellInput<'a> {
    /// Center energy of the spectral-function cell, FEFF `wpts(i)`.
    pub center_energy: Real,
    /// Lower finite-element boundary, FEFF `wlim(i-1)`.
    pub lower_boundary: Real,
    /// Upper finite-element boundary, FEFF `wlim(i)`.
    pub upper_boundary: Real,
    /// Photoelectron quasiparticle energy before pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Refined quasiparticle pole energy, FEFF `qpengy`.
    pub quasiparticle_energy: Real,
    /// Refined quasiparticle pole width, FEFF `qpwidth`.
    pub quasiparticle_width: Real,
    /// Interference quasiparticle amplitude after reduction, FEFF `ak`.
    pub interference_amplitude: Real,
    /// FEFF `isattype` branch used to form `esat`.
    pub extrinsic_mode: SfconvExtrinsicSatelliteMode,
    /// Imaginary self-energy derivative used by `isattype.eq.2`, FEFF `xaa`.
    pub imaginary_derivative: Real,
    /// Uniform mesh spacing used to choose local broadenings, FEFF `dw`.
    pub uniform_width: Real,
    /// FEFF ad-hoc interference reduction factor, FEFF `xreduc`.
    pub interference_reduction: Real,
    /// Active pole/plasma context for FEFF satellite helpers.
    pub context: SfconvSatelliteContext,
    /// On/off-shell self-energy state for FEFF satellite helpers.
    pub self_energy: SfconvSatelliteSelfEnergy,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
    /// Pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, Real>,
}

/// FEFF `mkspectf` rows for one spectral-function cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSpectralCell {
    /// Extrinsic quasiparticle row, FEFF `spectf(1,i)`.
    pub main_peak: Real,
    /// Extrinsic satellite row, FEFF `spectf(2,i)`.
    pub extrinsic_satellite: Real,
    /// Interference quasiparticle row, FEFF `spectf(3,i)`.
    pub quasiparticle_interference: Real,
    /// Interference satellite row, FEFF `spectf(4,i)`.
    pub interference_satellite: Real,
    /// Intrinsic satellite row, FEFF `spectf(5,i)`.
    pub intrinsic_satellite: Real,
    /// Combined satellite row, FEFF `spectf(6,i)`.
    pub combined_satellite: Real,
    /// Weighted quadrature error for the interference satellite contribution.
    pub interference_estimated_error: Real,
    /// Weighted quadrature error for the intrinsic satellite contribution.
    pub intrinsic_estimated_error: Real,
    /// Total FEFF `grater` integrand evaluations across active poles.
    pub evaluations: usize,
    /// Largest FEFF `grater` active-region stack seen in any pole.
    pub max_regions: usize,
}

/// Inputs for the FEFF `mkspectf` spectral-function cell loop.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralTableInput<'a> {
    /// Spectral-function center energies, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Finite-element cell boundaries, FEFF `wlim(0:npts)`.
    pub boundaries: ArrayView1<'a, Real>,
    /// Photoelectron quasiparticle energy before pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Refined quasiparticle pole energy, FEFF `qpengy`.
    pub quasiparticle_energy: Real,
    /// Refined quasiparticle pole width, FEFF `qpwidth`.
    pub quasiparticle_width: Real,
    /// Interference quasiparticle amplitude after reduction, FEFF `ak`.
    pub interference_amplitude: Real,
    /// FEFF `isattype` branch used to form `esat`.
    pub extrinsic_mode: SfconvExtrinsicSatelliteMode,
    /// Imaginary self-energy derivative used by `isattype.eq.2`, FEFF `xaa`.
    pub imaginary_derivative: Real,
    /// Uniform mesh spacing used to choose local broadenings, FEFF `dw`.
    pub uniform_width: Real,
    /// FEFF ad-hoc interference reduction factor, FEFF `xreduc`.
    pub interference_reduction: Real,
    /// Exponential quasiparticle-reduction factor, FEFF `expa`.
    pub exponential_reduction: Real,
    /// Active pole/plasma context for FEFF satellite helpers.
    pub context: SfconvSatelliteContext,
    /// On-shell self-energy state. Off-shell fields are replaced per cell.
    pub self_energy: SfconvSatelliteSelfEnergy,
    /// Per-cell real off-shell self energy, FEFF `sefr`.
    pub off_shell_real: ArrayView1<'a, Real>,
    /// Per-cell positive imaginary off-shell self energy, FEFF `sefi`.
    pub off_shell_imag: ArrayView1<'a, Real>,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, Real>,
    /// Pole weights normalized from oscillator strengths, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, Real>,
    /// Pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, Real>,
    /// FEFF one-based lower quasiparticle column, normally `iqpl`.
    pub quasiparticle_lower_column_1based: usize,
    /// FEFF one-based upper quasiparticle column, normally `iqph`.
    pub quasiparticle_upper_column_1based: usize,
}

/// FEFF `mkspectf` rows and raw accumulators from the spectral cell loop.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpectralTable {
    /// Eight-row spectral-function table with FEFF rows 1 through 6 filled.
    pub spectral_function: Array2<Real>,
    /// Endpoint-corrected integral accumulated as FEFF `wtemain`.
    pub integrated_main_weight: Real,
    /// Endpoint-corrected integral accumulated as FEFF `wtxmain`.
    pub integrated_quasiparticle_interference_weight: Real,
    /// Raw extrinsic satellite integral accumulated as FEFF `wtesat`.
    pub integrated_extrinsic_weight: Real,
    /// Raw interference satellite integral accumulated as FEFF `wtxsat`.
    pub integrated_interference_weight: Real,
    /// Raw intrinsic satellite integral accumulated as FEFF `wtisat`.
    pub integrated_intrinsic_weight: Real,
    /// Weighted quadrature error for the interference satellite contribution.
    pub interference_estimated_error: Real,
    /// Weighted quadrature error for the intrinsic satellite contribution.
    pub intrinsic_estimated_error: Real,
    /// Total FEFF `grater` integrand evaluations across active poles.
    pub evaluations: usize,
    /// Largest FEFF `grater` active-region stack seen in any pole.
    pub max_regions: usize,
}

/// FEFF `mkspectf` satellite rows and raw satellite weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSatelliteTable {
    /// Eight-row spectral-function table with FEFF rows 1 through 6 filled.
    pub spectral_function: Array2<Real>,
    /// Raw extrinsic satellite integral accumulated as FEFF `wtesat`.
    pub integrated_extrinsic_weight: Real,
    /// Raw interference satellite integral accumulated as FEFF `wtxsat`.
    pub integrated_interference_weight: Real,
    /// Raw intrinsic satellite integral accumulated as FEFF `wtisat`.
    pub integrated_intrinsic_weight: Real,
}

/// Inputs for FEFF `SFCONV/mkspectf.f90` extrinsic-satellite splitting.
#[derive(Debug, Clone, Copy)]
pub struct SfconvExtrinsicSatelliteSplitInput<'a> {
    /// Eight-row spectral-function table, FEFF `spectf(row, point)`.
    pub spectral_function: ArrayView2<'a, Real>,
    /// Spectral-function center energies, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Finite-element cell boundaries, FEFF `wlim(0:npts)`.
    pub boundaries: ArrayView1<'a, Real>,
    /// Photoelectron quasiparticle energy before pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Value of FEFF `beta(0.d0)` used by the split-trigger branch.
    pub beta_zero: Real,
}

/// FEFF `mkspectf` extrinsic-satellite split into rows 7 and 8.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvExtrinsicSatelliteSplit {
    /// Spectral-function table with FEFF rows 7 and 8 replaced.
    pub spectral_function: Array2<Real>,
    /// Zero-based column where FEFF switches from row 7 to row 8.
    pub switch_column: usize,
    /// FEFF `wpts(iswitch) + ekp`.
    pub switch_energy: Real,
    /// Whether FEFF selected the first-derivative trigger over curvature.
    pub derivative_triggered: bool,
}

/// Inputs for FEFF `SFCONV/mkspectf.f90` satellite clipping correction.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSatelliteCorrectionInput<'a> {
    /// Eight-row spectral-function table, FEFF `spectf(row, point)`.
    pub spectral_function: ArrayView2<'a, Real>,
    /// Finite-element cell boundaries, FEFF `wlim(0:npts)`.
    pub boundaries: ArrayView1<'a, Real>,
    /// Uniform mesh width `dw` used by FEFF for clipped component weights.
    pub uniform_width: Real,
    /// Exponential reduction factor, FEFF `expa`.
    pub exponential_reduction: Real,
}

/// Corrected FEFF `mkspectf` satellite table and final satellite weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSatelliteCorrection {
    /// Corrected eight-row spectral-function table.
    pub spectral_function: Array2<Real>,
    /// FEFF weights 4 through 8: extrinsic, interference, intrinsic,
    /// clipped satellite, and clipped main-region weights.
    pub weights: RealVec,
    /// Integrated combined satellite weight before clipping, FEFF `satwt`.
    pub uncorrected_satellite_weight: Real,
    /// Integrated negative satellite weight removed, FEFF `swtcorr`.
    pub clipped_negative_weight: Real,
    /// Renormalization factor applied to preserved positive satellite weight.
    pub correction_factor: Real,
}

/// Inputs for final FEFF `SFCONV/mkspectf.f90` spectral-table postprocessing.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralFinalizationInput<'a> {
    /// Eight-row spectral-function table after rows 1 through 6 are assembled.
    pub spectral_function: ArrayView2<'a, Real>,
    /// Spectral-function center energies, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Finite-element cell boundaries, FEFF `wlim(0:npts)`.
    pub boundaries: ArrayView1<'a, Real>,
    /// Photoelectron quasiparticle energy before pole refinement, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Value of FEFF `beta(0.d0)` used by the split-trigger branch.
    pub beta_zero: Real,
    /// Uniform mesh width `dw` used by FEFF for clipped component weights.
    pub uniform_width: Real,
    /// Real part of the renormalization constant, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub renormalization_imag: Real,
    /// Magnitude of the renormalization constant, FEFF `zm`.
    pub renormalization_magnitude: Real,
    /// Interference quasiparticle amplitude accumulated as FEFF `ak`.
    pub interference_amplitude: Real,
    /// FEFF ad-hoc interference reduction factor, FEFF `xreduc`.
    pub interference_reduction: Real,
    /// Exponential quasiparticle-reduction factor, FEFF `expa`.
    pub exponential_reduction: Real,
}

/// Final FEFF `mkspectf` spectral table, weights, and postprocessing metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpectralFinalization {
    /// Corrected eight-row spectral-function table.
    pub spectral_function: Array2<Real>,
    /// Final FEFF `weights(1:8)` vector.
    pub weights: RealVec,
    /// Zero-based column where FEFF switches from row 7 to row 8.
    pub switch_column: usize,
    /// FEFF `wpts(iswitch) + ekp`.
    pub switch_energy: Real,
    /// Whether FEFF selected the first-derivative trigger over curvature.
    pub derivative_triggered: bool,
    /// Integrated combined satellite weight before clipping, FEFF `satwt`.
    pub uncorrected_satellite_weight: Real,
    /// Integrated negative satellite weight removed, FEFF `swtcorr`.
    pub clipped_negative_weight: Real,
    /// Renormalization factor applied to preserved positive satellite weight.
    pub correction_factor: Real,
}

/// Inputs for the final FEFF `SFCONV/mkspectf.f90` eight-slot weight vector.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralWeightsInput<'a> {
    /// Real part of the renormalization constant, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub renormalization_imag: Real,
    /// Magnitude of the renormalization constant, FEFF `zm`.
    pub renormalization_magnitude: Real,
    /// Interference quasiparticle amplitude accumulated as FEFF `ak`.
    pub interference_amplitude: Real,
    /// FEFF ad-hoc interference reduction factor, FEFF `xreduc`.
    pub interference_reduction: Real,
    /// Exponential reduction factor, FEFF `expa`.
    pub exponential_reduction: Real,
    /// FEFF weights 4 through 8 from `sfconv_correct_satellite_weights`.
    pub satellite_weights: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` path-column interpolation.
#[derive(Debug, Clone, Copy)]
pub struct SfconvFeffPathInterpolationInput<'a> {
    /// Uniform SO2CONV momentum grid, FEFF `xk`.
    pub source_momentum: ArrayView1<'a, Real>,
    /// Coarse `feffNNNN.dat` path momentum grid, FEFF `xk2`.
    pub path_momentum: ArrayView1<'a, Real>,
    /// Central atom phase shifts on `path_momentum`, FEFF `caph2`.
    pub central_phase: ArrayView1<'a, Real>,
    /// Effective scattering amplitude on `path_momentum`, FEFF `xmfeff2`.
    pub effective_amplitude: ArrayView1<'a, Real>,
    /// Effective scattering phase on `path_momentum`, FEFF `phfeff2`.
    pub effective_phase: ArrayView1<'a, Real>,
    /// Reduction factors on `path_momentum`, FEFF `redfac2`.
    pub reduction_factor: ArrayView1<'a, Real>,
    /// Mean free paths on `path_momentum`, FEFF `xlam2`.
    pub mean_free_path: ArrayView1<'a, Real>,
}

/// FEFF `feffNNNN.dat` path columns interpolated onto a SO2CONV grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvFeffPathInterpolation {
    /// Central atom phase shifts on the source grid, FEFF `caph`.
    pub central_phase: RealVec,
    /// Effective scattering amplitude on the source grid, FEFF `xmfeff`.
    pub effective_amplitude: RealVec,
    /// Effective scattering phase on the source grid, FEFF `phfeff`.
    pub effective_phase: RealVec,
    /// Reduction factors on the source grid, FEFF `redfac`.
    pub reduction_factor: RealVec,
    /// Mean free paths on the source grid, FEFF `xlam`.
    pub mean_free_path: RealVec,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` raw path-signal construction.
#[derive(Debug, Clone, Copy)]
pub struct SfconvFeffPathSignalInput<'a> {
    /// Uniform SO2CONV momentum grid, FEFF `xk`.
    pub momentum: ArrayView1<'a, Real>,
    /// Central atom phase shifts on `momentum`, FEFF `caph`.
    pub central_phase: ArrayView1<'a, Real>,
    /// Effective scattering amplitude on `momentum`, FEFF `xmfeff`.
    pub effective_amplitude: ArrayView1<'a, Real>,
    /// Effective scattering phase on `momentum`, FEFF `phfeff`.
    pub effective_phase: ArrayView1<'a, Real>,
    /// Reduction factors on `momentum`, FEFF `redfac`.
    pub reduction_factor: ArrayView1<'a, Real>,
    /// Mean free paths on `momentum`, FEFF `xlam`.
    pub mean_free_path: ArrayView1<'a, Real>,
    /// Path degeneracy, FEFF `deg`.
    pub degeneracy: Real,
    /// Scattering half-path length, FEFF `Rnn`.
    pub half_path_length: Real,
}

/// Raw FEFF path signal before many-body SFCONV convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvFeffPathSignal {
    /// Path magnitude, FEFF `xmag`.
    pub magnitude: RealVec,
    /// Phase with the dominant `2kr` oscillation removed, FEFF `phm2kr`.
    pub phase_minus_2kr: RealVec,
    /// Full path phase, FEFF `phase`.
    pub phase: RealVec,
    /// Real EXAFS contribution, FEFF `chir`.
    pub real: RealVec,
    /// Imaginary EXAFS contribution, FEFF `chii`.
    pub imaginary: RealVec,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` EXAFS post-convolution row assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvExafsConvolutionInput {
    /// Convolved real-channel magnitude, FEFF `xchir`.
    pub real_convolution_amplitude: Real,
    /// Convolved real-channel phase, FEFF `phchir`.
    pub real_convolution_phase: Real,
    /// Convolved imaginary-channel magnitude, FEFF `xchii`.
    pub imaginary_convolution_amplitude: Real,
    /// Convolved imaginary-channel phase, FEFF `phchii`.
    pub imaginary_convolution_phase: Real,
    /// Original EXAFS magnitude before many-body convolution, FEFF `xmag(jj)`.
    pub original_magnitude: Real,
    /// Original EXAFS phase before many-body convolution, FEFF `phase(jj)`.
    pub original_phase: Real,
    /// Original phase with `2 k R` removed, FEFF `phm2kr(jj)`.
    pub phase_minus_2kr: Real,
    /// Previous raw many-body phase used for FEFF jump removal, `phshftold`.
    pub previous_phase: Real,
    /// FEFF integer phase-jump counter, `npi`.
    pub phase_jump_count: i32,
}

/// FEFF `SO2CONV` EXAFS row after spectral-function convolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvExafsConvolution {
    /// Real many-body EXAFS signal, FEFF `chirr`.
    pub real: Real,
    /// Imaginary many-body EXAFS signal written as the second output column.
    pub imaginary: Real,
    /// Many-body EXAFS magnitude, FEFF `sqrt(chirr**2 + chiii**2)`.
    pub magnitude: Real,
    /// Unwrapped many-body phase written to `chi.dat`/`chipNNNN.dat`.
    pub output_phase: Real,
    /// FEFF output phase correction `output_phase + phm2kr - phase`.
    pub output_phase_minus_original: Real,
    /// Many-body amplitude reduction, FEFF `s02list(jj)`.
    pub amplitude_reduction: Real,
    /// Many-body phase shift, FEFF `phlist(jj)`.
    pub phase_shift: Real,
    /// Updated raw phase state, FEFF `phshftold`.
    pub previous_phase: Real,
    /// Updated FEFF integer phase-jump counter, `npi`.
    pub phase_jump_count: i32,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` XANES post-convolution row assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvXanesConvolutionInput {
    /// Whether FEFF used the asymmetric quasiparticle phase branch, `iasym`.
    pub asymmetric_phase: bool,
    /// Convolved absorption from the asymmetric branch, FEFF `xmu2`.
    pub absorption_convolution: Real,
    /// Convolved embedded atom background, FEFF `xmu02`.
    pub embedded_background: Real,
    /// Convolved imaginary fine-structure component, FEFF `ximu2`.
    pub fine_structure_imaginary_amplitude: Real,
    /// Phase for `ximu2`, FEFF `phmu`.
    pub fine_structure_imaginary_phase: Real,
    /// Convolved real fine-structure component, FEFF `rmu2`.
    pub fine_structure_real_amplitude: Real,
    /// Phase for `rmu2`, FEFF `phrmu`.
    pub fine_structure_real_phase: Real,
}

/// FEFF `SO2CONV` XANES absorption row after spectral-function convolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvXanesConvolution {
    /// Many-body absorption value written as FEFF `xmu2`.
    pub absorption: Real,
    /// Embedded atom background written as FEFF `xmu02`.
    pub embedded_background: Real,
    /// Fine structure written by FEFF10 as `xmu2 - xmu02`.
    pub fine_structure: Real,
}

/// Inputs for FEFF `SO2CONV` EXAFS/`feffNNNN.dat` energy-grid padding.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSo2convExafsEnergyPaddingInput<'a> {
    /// Active energy grid before padding, FEFF `epts2(1:j)`.
    pub energy: ArrayView1<'a, Real>,
    /// Number of rows read from the FEFF file, FEFF `j`.
    pub active_len: usize,
    /// Full convolution work-array length, FEFF `npts2`.
    pub output_len: usize,
}

/// Inputs for FEFF `SO2CONV` EXAFS channel preparation.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSo2convExafsPreparationInput<'a> {
    /// FEFF wavenumber grid `xk`, in atomic units.
    pub momentum: ArrayView1<'a, Real>,
    /// EXAFS magnitude from `chi.dat`/`chipNNNN.dat`, FEFF `xmag`.
    pub magnitude: ArrayView1<'a, Real>,
    /// EXAFS phase from `chi.dat`/`chipNNNN.dat`, FEFF `phase`.
    pub phase: ArrayView1<'a, Real>,
    /// Optional path phase column, FEFF `phm2kr`; absent tables use zero.
    pub phase_minus_2kr: Option<ArrayView1<'a, Real>>,
    /// Chemical-potential offset, FEFF `cmu`.
    pub chemical_potential: Real,
    /// Number of rows read from the FEFF file, FEFF `j`.
    pub active_len: usize,
    /// Full convolution work-array length, FEFF `npts2`.
    pub output_len: usize,
}

/// FEFF `SO2CONV` EXAFS arrays prepared for spectral-function convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSo2convExafsPreparation {
    /// Padded signal-energy grid, FEFF `epts2`.
    pub signal_energy: RealVec,
    /// Real EXAFS channel, FEFF `chir`.
    pub real_signal: RealVec,
    /// Imaginary EXAFS channel, FEFF `chii`.
    pub imaginary_signal: RealVec,
    /// Original EXAFS magnitude, FEFF `xmag`.
    pub original_magnitude: RealVec,
    /// Original EXAFS phase, FEFF `phase`.
    pub original_phase: RealVec,
    /// Original phase with `2 k R` removed, FEFF `phm2kr`.
    pub phase_minus_2kr: RealVec,
}

/// Inputs for FEFF `SO2CONV` XANES signal padding and phase preparation.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSo2convXanesPreparationInput<'a> {
    /// Incident-energy output column, FEFF `e1`.
    pub incident_energy: ArrayView1<'a, Real>,
    /// Excitation energy relative to the edge, FEFF `epts2`.
    pub excitation_energy: ArrayView1<'a, Real>,
    /// Absorption signal, FEFF `xmu`.
    pub absorption: ArrayView1<'a, Real>,
    /// Embedded-atom background, FEFF `xmu0`.
    pub embedded_background: ArrayView1<'a, Real>,
    /// Number of rows read from `xmu.dat`, FEFF `j`.
    pub active_len: usize,
    /// Full convolution work-array length, FEFF `npts2`.
    pub output_len: usize,
}

/// FEFF `SO2CONV` XANES arrays prepared for spectral-function convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSo2convXanesPreparation {
    /// Padded incident-energy output column, FEFF `e1`.
    pub incident_energy: RealVec,
    /// Padded excitation grid, FEFF `epts2`.
    pub excitation_energy: RealVec,
    /// Padded absorption signal, FEFF `xmu`.
    pub absorption: RealVec,
    /// Padded embedded-atom background, FEFF `xmu0`.
    pub embedded_background: RealVec,
    /// Imaginary fine-structure component, FEFF `ximu = xmu - xmu0`.
    pub imaginary_fine_structure: RealVec,
    /// Kramers-Kronig real fine-structure component, FEFF `rmu`.
    pub real_fine_structure: RealVec,
}

/// Inputs for FEFF `SFCONV/so2conv.f90` path-grid averaging.
#[derive(Debug, Clone, Copy)]
pub struct SfconvPathAverageInput<'a> {
    /// Uniform source momentum grid, FEFF `xk`.
    pub source_momentum: ArrayView1<'a, Real>,
    /// Many-body amplitude reductions on `source_momentum`, FEFF `s02list`.
    pub amplitude_reduction: ArrayView1<'a, Real>,
    /// Many-body phase shifts on `source_momentum`, FEFF `phlist`.
    pub phase_shift: ArrayView1<'a, Real>,
    /// Previous coarse FEFF path momentum, FEFF `xk2(jj-1)`.
    pub previous_momentum: Real,
    /// Current coarse FEFF path momentum, FEFF `xk2(jj)`.
    pub center_momentum: Real,
    /// Next coarse FEFF path momentum, FEFF `xk2(jj+1)`.
    pub next_momentum: Real,
    /// Uniform source momentum spacing, FEFF `dk`.
    pub momentum_step: Real,
}

/// Averaged `SO2CONV` amplitude and phase for one coarse path row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvPathAverage {
    /// FEFF `s02sum/xnorm`, used to scale `redfac2(jj)`.
    pub amplitude_reduction: Real,
    /// FEFF `dphsum/xnorm`, used to shift `caph2(jj)`.
    pub phase_shift: Real,
    /// FEFF triangular finite-element normalization, `xnorm`.
    pub normalization: Real,
}

/// Error returned by SFCONV helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SfconvError {
    /// FEFF `mkrmu` smooths rows 20 and 21, so shorter inputs are unsupported.
    #[error("SFCONV {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    /// Active rows must fit in each input array.
    #[error("SFCONV active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Two related arrays must have the same length.
    #[error("SFCONV {left} length {left_len} does not match {right} length {right_len}")]
    LengthMismatch {
        left: &'static str,
        left_len: usize,
        right: &'static str,
        right_len: usize,
    },
    /// Fixed-size FEFF helper arrays must have the expected number of slots.
    #[error("SFCONV {field} count {actual} does not match expected count {expected}")]
    CountMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Scalar values must be finite.
    #[error("SFCONV {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Scalar values that appear in denominators must be positive.
    #[error("SFCONV {field} must be positive, got {value}")]
    NonPositiveScalar { field: &'static str, value: Real },
    /// Array values must be finite.
    #[error("SFCONV {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// The energy grid must be strictly increasing to avoid FEFF's pole division.
    #[error("SFCONV {field} row {row} must increase, got {current} after {previous}")]
    NonIncreasingEnergy {
        field: &'static str,
        row: usize,
        previous: Real,
        current: Real,
    },
    /// FEFF one-based pole selectors must fit the input arrays.
    #[error("SFCONV {field} index {index} is outside 1..={len}")]
    IndexOutOfRange {
        field: &'static str,
        index: usize,
        len: usize,
    },
    /// The asymmetric branch divides by the real quasiparticle weight.
    #[error("SFCONV asymmetric phase requires a nonzero real quasiparticle weight")]
    ZeroAsymmetricWeight,
    /// The asymmetric branch needs a nonzero plasma-frequency scale.
    #[error("SFCONV asymmetric phase requires a nonzero plasma frequency")]
    ZeroPlasmaFrequency,
    /// FEFF normalizes by the total spectral weight.
    #[error("SFCONV normalization weight must be finite and nonzero, got {value}")]
    InvalidNormalization { value: Real },
    /// The transformed value must be finite.
    #[error("SFCONV transformed row {row} must be finite, got {value}")]
    NonFiniteResult { row: usize, value: Real },
    /// A square-root radicand must stay non-negative.
    #[error("SFCONV {field} radicand must be non-negative, got {value}")]
    NegativeRadicand { field: &'static str, value: Real },
    /// FEFF formula denominator is singular for this input.
    #[error("SFCONV denominator {field} is zero")]
    ZeroDenominator { field: &'static str },
    /// Cubic root solving failed while finding FEFF pole limits.
    #[error("SFCONV pole-limit root solve failed: {source}")]
    RootSolve { source: RootError },
    /// Integration tolerances must be strictly positive.
    #[error("SFCONV tolerance {field} must be positive, got {value}")]
    NonPositiveTolerance { field: &'static str, value: Real },
    /// FEFF integration bounds must form a finite increasing interval.
    #[error("SFCONV integration interval must increase: lower={lower}, upper={upper}")]
    InvalidIntegrationInterval { lower: Real, upper: Real },
    /// FEFF `grater` stores at most 20 explicit split points.
    #[error("SFCONV integration received {count} split points; maximum is {max}")]
    TooManySingularities { count: usize, max: usize },
    /// Explicit split points must be finite, ordered, and inside the interval.
    #[error("invalid SFCONV split point {index}: {value}")]
    InvalidSingularity { index: usize, value: Real },
    /// FEFF `grater` exhausted its fixed region stack.
    #[error("SFCONV adaptive integration exceeded {max_regions} active regions")]
    TooManyIntegrationRegions { max_regions: usize },
    /// FEFF did not encounter the requested trigger in the supplied grid.
    #[error("SFCONV did not find mkspectf {field} trigger")]
    MissingTrigger { field: &'static str },
    /// The FEFF integer phase-jump counter overflowed.
    #[error("SFCONV phase-jump counter {value} cannot be adjusted by {delta}")]
    PhaseJumpOverflow { value: i32, delta: i32 },
}

/// Port of `SFCONV/mkrmu.f90`: discrete Kramers-Kronig transform.
///
/// FEFF integrates `(xmu - xmu0) / (w_i - w_j)` with endpoint/centered energy
/// widths, divides by `pi`, then averages rows 20 and 21 to smooth the legacy
/// phase handoff. The returned array contains exactly `active_len` rows.
pub fn sfconv_kramers_kronig_real_part(
    input: SfconvKramersKronigInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_count_at_least("active_len", input.active_len, 21)?;
    validate_active_len("imaginary", input.active_len, input.imaginary.len())?;
    validate_active_len(
        "reference_imaginary",
        input.active_len,
        input.reference_imaginary.len(),
    )?;
    validate_active_len("energy", input.active_len, input.energy.len())?;

    for row in 0..input.active_len {
        validate_finite_value("imaginary", row, input.imaginary[row])?;
        validate_finite_value("reference_imaginary", row, input.reference_imaginary[row])?;
        validate_finite_value("energy", row, input.energy[row])?;
        if row > 0 && input.energy[row] <= input.energy[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field: "energy",
                row,
                previous: input.energy[row - 1],
                current: input.energy[row],
            });
        }
    }

    let mut real_part = Array1::<Real>::zeros(input.active_len);
    for target in 0..input.active_len {
        let mut sum = 0.0;
        for source in 0..input.active_len {
            if source == target {
                continue;
            }
            let width = integration_width(input.energy, input.active_len, source);
            let numerator = input.imaginary[source] - input.reference_imaginary[source];
            sum += width * numerator / (input.energy[source] - input.energy[target]);
        }
        let value = sum / std::f64::consts::PI;
        if !value.is_finite() {
            return Err(SfconvError::NonFiniteResult { row: target, value });
        }
        real_part[target] = value;
    }

    let smoothed = 0.5 * (real_part[19] + real_part[20]);
    real_part[19] = smoothed;
    real_part[20] = smoothed;

    Ok(real_part)
}

/// Port of `SFCONV/plset.f90`: select one epsilon-inverse pole.
///
/// `pole_index_1based` follows FEFF's one-based `ipl` convention. The input
/// arrays correspond to `plengy`, `plwt`, and `plbrd`, and must have matching
/// lengths.
pub fn sfconv_select_pole(
    pole_index_1based: usize,
    energy: ArrayView1<'_, Real>,
    weight: ArrayView1<'_, Real>,
    broadening: ArrayView1<'_, Real>,
) -> Result<SfconvPole, SfconvError> {
    validate_count_at_least("poles", energy.len(), 1)?;
    validate_matching_lengths("energy", energy.len(), "weight", weight.len())?;
    validate_matching_lengths("energy", energy.len(), "broadening", broadening.len())?;
    validate_finite_array("energy", energy)?;
    validate_finite_array("weight", weight)?;
    validate_finite_array("broadening", broadening)?;

    if pole_index_1based == 0 || pole_index_1based > energy.len() {
        return Err(SfconvError::IndexOutOfRange {
            field: "pole",
            index: pole_index_1based,
            len: energy.len(),
        });
    }
    let index = pole_index_1based - 1;
    Ok(SfconvPole {
        energy: energy[index],
        weight: weight[index],
        broadening: broadening[index],
    })
}

/// Port of `SFCONV/ppset`: electron-gas parameters for a Wigner-Seitz radius.
pub fn sfconv_plasma_parameters(
    wigner_seitz_radius: Real,
) -> Result<SfconvPlasmaParameters, SfconvError> {
    validate_positive_scalar("wigner_seitz_radius", wigner_seitz_radius)?;

    let pi = std::f64::consts::PI;
    let fermi_momentum = (9.0 * pi / 4.0).powf(1.0 / 3.0) / wigner_seitz_radius;
    let fermi_energy = fermi_momentum * fermi_momentum / 2.0;
    let concentration = 3.0 / (4.0 * pi * wigner_seitz_radius.powi(3));
    let plasma_frequency = (4.0 * pi * concentration).sqrt();
    Ok(SfconvPlasmaParameters {
        fermi_momentum,
        fermi_energy,
        plasma_frequency,
    })
}

/// Port of the `SO2CONV` material-constant setup from FEFF output headers.
///
/// FEFF stores `Gam_ch`, `Vint`, `Mu`, and `kf` in spectrum-file headers and
/// converts them using legacy local constants in `so2conv.f90`. This helper
/// preserves those constants and returns the electron-gas quantities that feed
/// pole loading, threshold selection, momentum refinement, and convolution.
pub fn sfconv_so2conv_material_parameters(
    input: SfconvSo2convMaterialInput,
) -> Result<SfconvSo2convMaterialParameters, SfconvError> {
    validate_so2conv_material_input(input)?;

    let core_hole_lifetime = finite_result(
        "so2conv core_hole_lifetime",
        (input.core_hole_width_ev / 2.0) / SFCONV_SO2CONV_HARTREE_EV,
    )?;
    let interstitial_potential = finite_result(
        "so2conv interstitial_potential",
        input.interstitial_potential_ev / SFCONV_SO2CONV_HARTREE_EV,
    )?;
    let chemical_potential_offset = finite_result(
        "so2conv chemical_potential_offset",
        (input.chemical_potential_ev - input.interstitial_potential_ev) / SFCONV_SO2CONV_HARTREE_EV,
    )?;
    let fermi_wave_number = finite_result(
        "so2conv fermi_wave_number",
        input.fermi_wave_number_inv_angstrom * SFCONV_SO2CONV_BOHR_ANGSTROM,
    )?;
    let pi = std::f64::consts::PI;
    let fermi_momentum = finite_result(
        "so2conv fermi_momentum",
        (9.0 * pi / 4.0).powf(1.0 / 3.0) / input.wigner_seitz_radius,
    )?;
    let fermi_energy = finite_result("so2conv fermi_energy", fermi_momentum.powi(2) / 2.0)?;
    let electron_concentration = finite_result(
        "so2conv electron_concentration",
        3.0 / (4.0 * pi * input.wigner_seitz_radius.powi(3)),
    )?;
    let plasma_frequency = checked_sqrt(
        "so2conv plasma_frequency",
        4.0 * pi * electron_concentration,
    )?;
    let dispersion_parameter =
        finite_result("so2conv dispersion_parameter", 2.0 * fermi_energy / 3.0)?;

    Ok(SfconvSo2convMaterialParameters {
        core_hole_lifetime,
        interstitial_potential,
        chemical_potential_offset,
        fermi_wave_number,
        fermi_momentum,
        fermi_energy,
        electron_concentration,
        plasma_frequency,
        dispersion_parameter,
        initial_photoelectron_energy: fermi_energy,
        initial_photoelectron_momentum: fermi_momentum,
        accuracy: 1.0e-4,
    })
}

/// Port of `SFCONV/ppole.f90` `wdisp`: pole dispersion relation.
pub fn sfconv_pole_dispersion(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    pole_dispersion_value(momentum, pole_energy, dispersion_parameter)
}

/// Port of `SFCONV/ppole.f90` `dwdq`: first dispersion derivative.
pub fn sfconv_pole_dispersion_derivative(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    let dispersion = pole_dispersion_value(momentum, pole_energy, dispersion_parameter)?;
    Ok((momentum.powi(3) + 2.0 * dispersion_parameter * momentum) / (2.0 * dispersion))
}

/// Port of `SFCONV/ppole.f90` `d2wdq2`: second dispersion derivative.
pub fn sfconv_pole_dispersion_second_derivative(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    let dispersion = pole_dispersion_value(momentum, pole_energy, dispersion_parameter)?;
    let derivative =
        (momentum.powi(3) + 2.0 * dispersion_parameter * momentum) / (2.0 * dispersion);
    let numerator = (3.0 * momentum.powi(2) + 2.0 * dispersion_parameter) * dispersion
        - (momentum.powi(3) + 2.0 * dispersion_parameter * momentum) * derivative;
    Ok(numerator / (2.0 * dispersion.powi(2)))
}

/// Port of `SFCONV/ppole.f90` `qdisp`: inverse pole dispersion relation.
pub fn sfconv_inverse_pole_dispersion(
    energy: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("energy", energy)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;

    let discriminant = dispersion_parameter.powi(2) + energy.powi(2) - pole_energy.powi(2);
    if discriminant >= 0.0 {
        let radicand = -2.0 * dispersion_parameter + 2.0 * discriminant.sqrt();
        if radicand >= 0.0 {
            return Ok(radicand.sqrt());
        }
    }
    Ok(0.0)
}

/// Port of `SFCONV/ppole.f90` `vpp2`: squared pole-coupling potential.
pub fn sfconv_coupling_potential_squared(
    momentum: Real,
    plasma_frequency: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum.abs())?;
    validate_positive_scalar("plasma_frequency", plasma_frequency)?;
    let dispersion = sfconv_pole_dispersion(momentum, pole_energy, dispersion_parameter)?;
    Ok(2.0 * std::f64::consts::PI * plasma_frequency.powi(2) / (momentum.powi(2) * dispersion))
}

/// Port of `SFCONV/qlimits.f90`: momentum limits for pole-loss inequalities.
pub fn sfconv_q_limits(
    energy: Real,
    photoelectron_momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
    upper_limit: Real,
) -> Result<SfconvQLimits, SfconvError> {
    validate_finite_scalar("energy", energy)?;
    validate_positive_scalar("photoelectron_momentum", photoelectron_momentum)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;
    validate_positive_scalar("upper_limit", upper_limit)?;

    sfconv_q_limits_with_upper(
        energy,
        photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )
}

fn sfconv_q_limits_with_upper(
    energy: Real,
    photoelectron_momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
    upper_limit: Real,
) -> Result<SfconvQLimits, SfconvError> {
    let a = photoelectron_momentum;
    let b = energy + dispersion_parameter - 3.0 * photoelectron_momentum.powi(2) / 2.0;
    let c = photoelectron_momentum.powi(3) - 2.0 * energy * photoelectron_momentum;
    let d = pole_energy.powi(2) - energy.powi(2) + energy * photoelectron_momentum.powi(2)
        - photoelectron_momentum.powi(4) / 4.0;
    let roots =
        real_polynomial_roots([a, b, c, d]).map_err(|source| SfconvError::RootSolve { source })?;
    let values = roots.into_inner();

    if roots.real_root_count() == 3 {
        let root0 = values[0].re;
        let root1 = values[1].re;
        let root2 = values[2].re;
        let dev0 = (pole_dispersion_value(root0, pole_energy, dispersion_parameter)?
            + (root0 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let dev1 = (pole_dispersion_value(root1, pole_energy, dispersion_parameter)?
            + (root1 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let dev2 = (pole_dispersion_value(root2, pole_energy, dispersion_parameter)?
            + (root2 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let (q1, q2, q3) = if dev0 > dev1 && dev0 > dev2 {
            (
                root1.abs().min(root2.abs()),
                root1.abs().max(root2.abs()),
                root0.abs(),
            )
        } else if dev1 > dev2 {
            (
                root0.abs().min(root2.abs()),
                root0.abs().max(root2.abs()),
                root1.abs(),
            )
        } else {
            (
                root0.abs().min(root1.abs()),
                root0.abs().max(root1.abs()),
                root2.abs(),
            )
        };
        Ok(SfconvQLimits {
            count: 3,
            q1: q1.min(upper_limit),
            q2: q2.min(upper_limit),
            q3,
        })
    } else {
        let imag0 = values[0].im.abs();
        let imag1 = values[1].im.abs();
        let imag2 = values[2].im.abs();
        let q3 = if imag0 < imag1 && imag0 < imag2 {
            values[0].re.abs()
        } else if imag1 < imag2 {
            values[1].re.abs()
        } else {
            values[2].re.abs()
        };
        Ok(SfconvQLimits {
            count: 1,
            q1: 0.0,
            q2: 0.0,
            q3,
        })
    }
}

/// Port of `SFCONV/ppole.f90` `qthresh`: plasmon-loss onset momentum.
pub fn sfconv_plasmon_threshold_momentum(
    pole_energy: Real,
    dispersion_parameter: Real,
    fermi_energy: Real,
    fermi_momentum: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;
    validate_positive_scalar("fermi_energy", fermi_energy)?;
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;

    let roots = real_polynomial_roots([
        1.0,
        -3.0 * dispersion_parameter,
        3.0 * dispersion_parameter.powi(2) - 27.0 * pole_energy.powi(2) / 4.0,
        -dispersion_parameter.powi(3),
    ])
    .map_err(|source| SfconvError::RootSolve { source })?;
    let qthresh1 = if roots.real_root_count() == 1 {
        let sorted = roots_sorted_by_imag_descending(roots.into_inner());
        sorted[1].re
    } else {
        roots
            .roots()
            .iter()
            .map(|root| root.re)
            .fold(f64::NEG_INFINITY, Real::max)
    };
    let qthresh1 = if qthresh1 > 0.0 { qthresh1.sqrt() } else { 0.0 };

    let b = 1.5 * fermi_momentum + dispersion_parameter / fermi_momentum;
    let c = fermi_momentum.powi(2) + 2.0 * dispersion_parameter;
    let d = fermi_momentum.powi(3) / 4.0
        + dispersion_parameter * fermi_momentum
        + pole_energy.powi(2) / fermi_momentum;
    let roots_a = real_polynomial_roots([1.0, b, c, d])
        .map_err(|source| SfconvError::RootSolve { source })?;
    let values_a = roots_a.into_inner();
    let q01 = if roots_a.real_root_count() == 1 {
        roots_sorted_by_imag_descending(values_a)[1].re
    } else {
        let selected = select_threshold_root(values_a, |root| {
            let xfact = threshold_factor(dispersion_parameter, pole_energy, root)?;
            Ok(root - fermi_momentum - checked_sqrt("qthresh test", 2.0 * xfact)?)
        })?;
        selected.re
    };

    let roots_b = real_polynomial_roots([1.0, -b, c, -d])
        .map_err(|source| SfconvError::RootSolve { source })?;
    let values_b = roots_b.into_inner();
    let q02 = if roots_b.real_root_count() == 1 {
        roots_sorted_by_imag_descending(values_b)[1].re
    } else {
        // FEFF selects the index using the second cubic, but returns from the
        // first root array. Preserve that historical behavior.
        let index = select_threshold_root_index(values_b, |root| {
            let xfact = threshold_factor(dispersion_parameter, pole_energy, root)?;
            Ok(root + fermi_momentum - checked_sqrt("qthresh test", 2.0 * xfact)?)
        })?;
        values_a[index].re
    };

    let qthresh2 = q01.abs().min(q02.abs());
    let upper_limit = 1000.0 * fermi_momentum;
    let energy1 = qthresh1.powi(2) / 2.0;
    let limits_a = sfconv_q_limits(
        energy1,
        qthresh1,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let _q0a =
        sfconv_inverse_pole_dispersion(energy1 - fermi_energy, pole_energy, dispersion_parameter)?;

    let energy2 = qthresh2.powi(2) / 2.0;
    let limits_b = sfconv_q_limits(
        energy2,
        qthresh2,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let q0b =
        sfconv_inverse_pole_dispersion(energy2 - fermi_energy, pole_energy, dispersion_parameter)?;

    if limits_a.count == 0 || (limits_a.q1 - limits_a.q2).abs() < (limits_b.q1 - q0b).abs() {
        Ok(qthresh1)
    } else {
        Ok(qthresh2)
    }
}

/// Port of the FEFF `SO2CONV` minimal momentum grid construction.
///
/// FEFF tabulates spectral functions on 66 momentum rows. The first section
/// bridges from the Fermi momentum `qf` to the plasmon threshold `pthresh`;
/// subsequent sections extend that grid to `300 * pthresh`.
pub fn sfconv_so2conv_momentum_grid(
    fermi_momentum: Real,
    threshold_momentum: Real,
) -> Result<RealVec, SfconvError> {
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;
    validate_positive_scalar("threshold_momentum", threshold_momentum)?;
    if threshold_momentum <= fermi_momentum {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: fermi_momentum,
            upper: threshold_momentum,
        });
    }

    let mut grid = Array1::<Real>::zeros(SFCONV_SO2CONV_MOMENTUM_GRID_LEN);

    let first_step = (threshold_momentum - fermi_momentum) / 10.0;
    for (index, value) in grid.iter_mut().take(10).enumerate() {
        *value = fermi_momentum + (index as Real + 1.0) * first_step;
    }

    let second_step = 0.25 * threshold_momentum / 30.0;
    let second_anchor = grid[9];
    for offset in 1..=30 {
        grid[9 + offset] = second_anchor + offset as Real * second_step;
    }

    let third_step = 0.75 * threshold_momentum / 10.0;
    let third_anchor = grid[39];
    for offset in 1..=10 {
        grid[39 + offset] = third_anchor + offset as Real * third_step;
    }

    let fourth_step = 2.0 * threshold_momentum / 10.0;
    let fourth_anchor = grid[49];
    for offset in 1..=10 {
        grid[49 + offset] = fourth_anchor + offset as Real * fourth_step;
    }

    for (index, multiplier) in [5.0, 7.0, 10.0, 30.0, 100.0, 300.0].into_iter().enumerate() {
        grid[60 + index] = multiplier * threshold_momentum;
    }

    validate_strictly_increasing("so2conv_momentum_grid", grid.view())?;
    Ok(grid)
}

/// Port of `SFCONV/senergies.f90` `exchange`.
///
/// Computes the Hartree-Fock exchange potential for a free electron gas at
/// photoelectron momentum `momentum`.
pub fn sfconv_free_electron_exchange(
    momentum: Real,
    fermi_momentum: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;

    let value = if momentum == fermi_momentum {
        -fermi_momentum / std::f64::consts::PI
    } else {
        let ratio = (momentum + fermi_momentum) / (momentum - fermi_momentum);
        validate_nonzero_denominator("exchange logarithm", ratio)?;
        -(fermi_momentum
            + ((fermi_momentum.powi(2) - momentum.powi(2)) / (2.0 * momentum)) * ratio.abs().ln())
            / std::f64::consts::PI
    };
    finite_result("free electron exchange", value)
}

/// Port of `SFCONV/senergies.f90` `beta`.
///
/// FEFF uses this extrinsic beta function as the analytic imaginary
/// self-energy contribution for the active pole.
pub fn sfconv_extrinsic_beta(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let pole_energy = context.pole_energy;
    let dispersion_parameter = context.dispersion_parameter;
    let fermi_limited_energy =
        (energy + context.quasiparticle_energy - context.fermi_energy).max(pole_energy);
    let qh =
        sfconv_inverse_pole_dispersion(fermi_limited_energy, pole_energy, dispersion_parameter)?;
    let q0 = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - energy - context.quasiparticle_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        energy + context.quasiparticle_energy,
        context.photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        qh,
    )?;

    let above_fermi = if limits.count == 3 {
        let q1 = checked_sqrt(
            "beta q1",
            limits.q1.powi(2) + context.accuracy * pole_energy,
        )?;
        let q2 = checked_sqrt(
            "beta q2",
            limits.q2.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq1 = sfconv_pole_dispersion(q1, pole_energy, dispersion_parameter)?;
        let wq2 = sfconv_pole_dispersion(q2, pole_energy, dispersion_parameter)?;
        beta_prefactor(context)
            * beta_log_argument(q2, wq2, q1, wq1, pole_energy, dispersion_parameter)?.ln()
    } else {
        0.0
    };

    let below_fermi = if limits.q3 < q0 && context.include_below_fermi {
        let q0 = checked_sqrt("beta q0", q0.powi(2) + context.accuracy * pole_energy)?;
        let q3 = checked_sqrt(
            "beta q3",
            limits.q3.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq0 = sfconv_pole_dispersion(q0, pole_energy, dispersion_parameter)?;
        let wq3 = sfconv_pole_dispersion(q3, pole_energy, dispersion_parameter)?;
        beta_prefactor(context)
            * beta_log_argument(q0, wq0, q3, wq3, pole_energy, dispersion_parameter)?.ln()
    } else {
        0.0
    };

    finite_result("extrinsic beta", above_fermi - below_fermi)
}

/// Port of `SFCONV/senergies.f90` `xienergies`.
pub fn sfconv_imaginary_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    finite_result(
        "imaginary self energy",
        -std::f64::consts::PI * sfconv_extrinsic_beta(energy, context)?,
    )
}

/// Port of `SFCONV/senergies.f90` `renergies`.
///
/// Returns the real part of the photoelectron self energy for the active pole,
/// with FEFF `grater` diagnostics accumulated across the piecewise momentum
/// integrals.
pub fn sfconv_real_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let qmax = 100.0 * checked_sqrt("self-energy qmax", context.pole_energy)?
        + context.photoelectron_momentum
        + context.fermi_momentum;
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut total = SfconvAdaptiveIntegral {
        value: 0.0,
        estimated_error: 0.0,
        evaluations: 0,
        max_regions: 0,
    };

    if context.photoelectron_momentum > context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            context.photoelectron_momentum - context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum - context.fermi_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_middle(momentum, energy, context),
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            context.fermi_momentum - context.photoelectron_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_middle(momentum, energy, context),
        )?;
        if context.include_below_fermi {
            add_real_self_energy_range(
                &mut total,
                0.0,
                context.fermi_momentum - context.photoelectron_momentum,
                absolute_tolerance,
                relative_tolerance,
                |momentum| sfconv_real_self_energy_integrand_lower(momentum, energy, context),
            )?;
        }
    } else {
        add_real_self_energy_range(
            &mut total,
            2.0 * context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            2.0 * context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_middle(momentum, energy, context),
        )?;
    }

    let scale = -context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    total.value = finite_result("real self energy", total.value * scale)?;
    total.estimated_error *= scale.abs();
    Ok(total)
}

/// Port of `SFCONV/senergies.f90` `drenergies`.
///
/// Returns the energy derivative of the real part of the photoelectron self
/// energy for the active pole, with accumulated FEFF `grater` diagnostics.
pub fn sfconv_real_self_energy_derivative(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)?;

    let qmax = 100.0 * checked_sqrt("self-energy derivative qmax", context.pole_energy)?;
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut total = SfconvAdaptiveIntegral {
        value: 0.0,
        estimated_error: 0.0,
        evaluations: 0,
        max_regions: 0,
    };

    if context.photoelectron_momentum > context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            context.photoelectron_momentum - context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum - context.fermi_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_middle(momentum, energy, context)
            },
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            context.fermi_momentum - context.photoelectron_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_middle(momentum, energy, context)
            },
        )?;
        if context.include_below_fermi {
            add_real_self_energy_range(
                &mut total,
                0.0,
                context.fermi_momentum - context.photoelectron_momentum,
                absolute_tolerance,
                relative_tolerance,
                |momentum| {
                    sfconv_real_self_energy_derivative_integrand_lower(momentum, energy, context)
                },
            )?;
        }
    } else {
        add_real_self_energy_range(
            &mut total,
            2.0 * context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            2.0 * context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_middle(momentum, energy, context)
            },
        )?;
    }

    let scale = context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    total.value = finite_result("real self energy derivative", total.value * scale)?;
    total.estimated_error *= scale.abs();
    Ok(total)
}

/// Port of `SFCONV/senergies.f90` `dienergies`.
pub fn sfconv_imaginary_self_energy_derivative(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let pole_energy = context.pole_energy;
    let dispersion_parameter = context.dispersion_parameter;
    let shifted_energy = energy + context.quasiparticle_energy;
    let qh = sfconv_inverse_pole_dispersion(
        (shifted_energy - context.fermi_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let mut q0 = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - shifted_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let upper_limit = 1.0e6 * context.fermi_momentum;
    let limits = sfconv_q_limits_with_upper(
        shifted_energy,
        context.photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let mut q1 = limits.q1;
    let mut q2 = limits.q2;
    let mut q3 = limits.q3;

    let (dqhdw, dq0dw) = self_energy_fermi_limit_derivatives(shifted_energy, qh, q0, context)?;
    let dq1dw = self_energy_upper_limit_derivative(&mut q1, qh, dqhdw, shifted_energy, context)?;
    let dq2dw = self_energy_upper_limit_derivative(&mut q2, qh, dqhdw, shifted_energy, context)?;
    let dq3dw = self_energy_lower_limit_derivative(&mut q3, q0, dq0dw, shifted_energy, context)?;

    let mut derivative = 0.0;
    let prefactor =
        context.plasma_frequency.powi(2) / (4.0 * context.photoelectron_momentum * pole_energy);

    if limits.count == 3 {
        q1 = checked_sqrt(
            "imaginary derivative q1",
            q1.powi(2) + context.accuracy * pole_energy,
        )?;
        q2 = checked_sqrt(
            "imaginary derivative q2",
            q2.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq1 = sfconv_pole_dispersion(q1, pole_energy, dispersion_parameter)?;
        let wq2 = sfconv_pole_dispersion(q2, pole_energy, dispersion_parameter)?;
        derivative += prefactor
            * dq1dw
            * self_energy_imaginary_derivative_factor(q1, wq1, pole_energy, dispersion_parameter)?;
        derivative -= prefactor
            * dq2dw
            * self_energy_imaginary_derivative_factor(q2, wq2, pole_energy, dispersion_parameter)?;
    }

    if q3 < q0 && context.include_below_fermi {
        q0 = checked_sqrt(
            "imaginary derivative q0",
            q0.powi(2) + context.accuracy * pole_energy,
        )?;
        q3 = checked_sqrt(
            "imaginary derivative q3",
            q3.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq0 = sfconv_pole_dispersion(q0, pole_energy, dispersion_parameter)?;
        let wq3 = sfconv_pole_dispersion(q3, pole_energy, dispersion_parameter)?;
        derivative += prefactor
            * dq0dw
            * self_energy_imaginary_derivative_factor(q0, wq0, pole_energy, dispersion_parameter)?;
        derivative -= prefactor
            * dq3dw
            * self_energy_imaginary_derivative_factor(q3, wq3, pole_energy, dispersion_parameter)?;
    }

    finite_result("imaginary self energy derivative", derivative)
}

/// Port of FEFF `SFCONV/mkspectf.f90` renormalization from self-energy slopes.
///
/// FEFF forms `xrz = 1 - d(Re Sigma)/dE`, `xiz = -d(Im Sigma)/dE`, and
/// returns the reciprocal complex factor used for the quasiparticle peak and
/// satellite amplitudes.
pub fn sfconv_self_energy_renormalization(
    real_derivative: Real,
    imaginary_derivative: Real,
) -> Result<SfconvRenormalization, SfconvError> {
    validate_finite_scalar("self-energy real derivative", real_derivative)?;
    validate_finite_scalar("self-energy imaginary derivative", imaginary_derivative)?;

    let real_inverse = 1.0 - real_derivative;
    let imaginary_inverse = -imaginary_derivative;
    let denominator = real_inverse.powi(2) + imaginary_inverse.powi(2);
    validate_nonzero_denominator("self-energy renormalization", denominator)?;

    let real = finite_result("renormalization real", real_inverse / denominator)?;
    let imaginary = finite_result(
        "renormalization imaginary",
        -imaginary_inverse / denominator,
    )?;
    let magnitude = checked_hypot("renormalization magnitude", real, imaginary)?;
    Ok(SfconvRenormalization {
        real,
        imaginary,
        magnitude,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` exponential pole-reduction factor.
///
/// FEFF accumulates `xa += 3*wt*(omp/ompl)**2/(8*sqrt(2*ompl))` over the
/// active epsilon-inverse poles and returns `exp(-xa)`.
pub fn sfconv_exponential_reduction(
    input: SfconvExponentialReductionInput<'_>,
) -> Result<Real, SfconvError> {
    validate_exponential_reduction_input(input)?;

    let exponent = (0..input.pole_count).try_fold(0.0, |total, index| {
        let pole_energy = input.pole_energy[index];
        let pole_weight = input.pole_weight[index];
        let denominator = 8.0 * checked_sqrt("exponential reduction pole", 2.0 * pole_energy)?;
        validate_nonzero_denominator("exponential reduction pole", denominator)?;
        finite_result(
            "exponential reduction exponent",
            total
                + 3.0 * pole_weight * (input.plasma_frequency / pole_energy).powi(2) / denominator,
        )
    })?;
    finite_result("exponential reduction", (-exponent).exp())
}

/// Port of FEFF `SFCONV/mkspectf.f90` quasiparticle pole refinement.
///
/// FEFF computes `qpengy = ekp + width*z1i` and `qpwidth = width*z1`
/// after the final on-shell self-energy derivative pass. The returned pole
/// feeds the finite-element quasiparticle peak rows.
pub fn sfconv_quasiparticle_pole(
    input: SfconvQuasiparticlePoleInput,
) -> Result<SfconvQuasiparticlePole, SfconvError> {
    validate_quasiparticle_pole_input(input)?;

    let energy = finite_result(
        "quasiparticle energy",
        input.photoelectron_energy + input.width * input.renormalization.imaginary,
    )?;
    let width = finite_result(
        "quasiparticle width",
        input.width * input.renormalization.real,
    )?;
    validate_positive_scalar("quasiparticle width", width)?;
    Ok(SfconvQuasiparticlePole { energy, width })
}

/// Port of the `SFCONV/mkspectf.f90` fixed spectral-function energy mesh.
///
/// FEFF uses 112 nonuniform offsets around the quasiparticle peak and a
/// companion `wlim(0:npts)` boundary array to integrate each cell. The mesh is
/// scaled by the plasma frequency, `omp`.
pub fn sfconv_spectral_energy_grid(
    plasma_frequency: Real,
) -> Result<SfconvSpectralEnergyGrid, SfconvError> {
    validate_positive_scalar("plasma_frequency", plasma_frequency)?;

    let mut energy = Array1::<Real>::zeros(SFCONV_MKSPECTF_GRID_LEN);
    let dw = plasma_frequency / 30.0;
    let iqph = 54;
    let iqpl = 53;

    energy[feff_index(iqph)] = dw * 1.0e-2;
    energy[feff_index(iqpl)] = -dw * 1.0e-2;
    energy[feff_index(iqph + 1)] = dw * 2.0e-2;
    energy[feff_index(iqpl - 1)] = -dw * 2.0e-2;
    for i in 1..=30 {
        let offset = i as Real;
        energy[feff_index(i + 1 + iqph)] = offset * dw;
        energy[feff_index(iqpl - 1 - i)] = -offset * dw;
    }
    for i in 1..=3 {
        let offset = i as Real;
        energy[feff_index(i + 31 + iqph)] = energy[feff_index(31 + iqph)] + offset * dw;
        energy[feff_index(iqpl - 31 - i)] = energy[feff_index(iqpl - 31)] - offset * dw;
    }
    for i in 1..=3 {
        let offset = i as Real;
        energy[feff_index(i + 34 + iqph)] = energy[feff_index(34 + iqph)] + 2.0 * offset * dw;
        energy[feff_index(iqpl - 34 - i)] = energy[feff_index(iqpl - 33)] - 2.0 * offset * dw;
    }
    for i in 1..=3 {
        let offset = i as Real;
        energy[feff_index(i + 37 + iqph)] = energy[feff_index(37 + iqph)] + 4.0 * offset * dw;
        energy[feff_index(iqpl - 37 - i)] = energy[feff_index(iqpl - 36)] - 4.0 * offset * dw;
    }
    for i in 1..=12 {
        let offset = i as Real;
        energy[feff_index(i + 40 + iqph)] = energy[feff_index(40 + iqph)] + 10.0 * offset * dw;
        energy[feff_index(iqpl - 40 - i)] = energy[feff_index(iqpl - 39)] - 10.0 * offset * dw;
    }
    for i in 1..=6 {
        let offset = i as Real;
        energy[feff_index(i + 52 + iqph)] = energy[feff_index(52 + iqph)] + 30.0 * offset * dw;
    }

    let mut boundaries = Array1::<Real>::zeros(SFCONV_MKSPECTF_GRID_LEN + 1);
    for index in 1..SFCONV_MKSPECTF_GRID_LEN {
        boundaries[index] = 0.5 * (energy[index - 1] + energy[index]);
    }
    boundaries[0] = 2.0 * energy[0] - energy[1];
    boundaries[SFCONV_MKSPECTF_GRID_LEN] =
        2.0 * energy[SFCONV_MKSPECTF_GRID_LEN - 1] - energy[SFCONV_MKSPECTF_GRID_LEN - 2];

    validate_finite_array("spectral energy grid", energy.view())?;
    validate_finite_array("spectral energy boundaries", boundaries.view())?;
    Ok(SfconvSpectralEnergyGrid { energy, boundaries })
}

/// Port of the `SFCONV/mkspectf.f90` extrinsic quasiparticle peak cell.
///
/// FEFF stores the quasiparticle peak as a finite-element average over
/// `wlim(i-1)..wlim(i)`. The real renormalization contributes the integrated
/// Lorentzian term, while the imaginary renormalization contributes FEFF's
/// logarithmic asymmetric term with the same Gaussian damping used in
/// `mkspectf`.
pub fn sfconv_quasiparticle_main_peak(
    input: SfconvQuasiparticlePeakInput,
) -> Result<Real, SfconvError> {
    validate_quasiparticle_peak_input(input)?;

    let bin_width = input.upper_boundary - input.lower_boundary;
    let upper_delta =
        input.upper_boundary - input.quasiparticle_energy + input.photoelectron_energy;
    let lower_delta =
        input.lower_boundary - input.quasiparticle_energy + input.photoelectron_energy;
    let pi = std::f64::consts::PI;
    let atan_term = input.renormalization_real
        * ((upper_delta / input.quasiparticle_width).atan()
            - (lower_delta / input.quasiparticle_width).atan())
        / (pi * bin_width);

    let upper_norm = input.quasiparticle_width.powi(2) + upper_delta.powi(2);
    let lower_norm = input.quasiparticle_width.powi(2) + lower_delta.powi(2);
    validate_positive_scalar("quasiparticle peak lower norm", lower_norm)?;
    let log_argument = upper_norm / lower_norm;
    validate_positive_scalar("quasiparticle peak logarithm", log_argument)?;

    let center_delta =
        input.center_energy + input.photoelectron_energy - input.quasiparticle_energy;
    let gaussian = (-(center_delta / (2.0 * input.plasma_frequency)).powi(2)).exp();
    let log_term =
        input.renormalization_imag * log_argument.ln() * gaussian / (2.0 * pi * bin_width);

    finite_result("quasiparticle main peak", atan_term - log_term)
}

/// Port of the `SFCONV/mkspectf.f90` quasiparticle row assembly.
///
/// FEFF fills `spectf(1,:)` with finite-element quasiparticle peak averages
/// and `spectf(3,:)` with the proportional interference term. It also carries
/// endpoint-corrected integrals for both rows; those accumulators are returned
/// for tests and future full-driver assembly.
pub fn sfconv_quasiparticle_table(
    input: SfconvQuasiparticleTableInput<'_>,
) -> Result<SfconvQuasiparticleTable, SfconvError> {
    validate_quasiparticle_table_input(input)?;

    let pi = std::f64::consts::PI;
    let endpoint_main = ((input.boundaries[0] / input.endpoint_width).atan() + pi / 2.0) / pi
        + (pi / 2.0 - (input.boundaries[input.boundaries.len() - 1] / input.endpoint_width).atan())
            / pi;
    let mut integrated_interference = 2.0
        * endpoint_main
        * input.renormalization_magnitude
        * input.renormalization_real
        * input.interference_amplitude;
    let mut integrated_main =
        endpoint_main * input.renormalization_real * input.exponential_reduction;

    let mut main_peak = Array1::<Real>::zeros(input.energy.len());
    let mut interference_peak = Array1::<Real>::zeros(input.energy.len());
    for column in 0..input.energy.len() {
        let peak = sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
            center_energy: input.energy[column],
            lower_boundary: input.boundaries[column],
            upper_boundary: input.boundaries[column + 1],
            photoelectron_energy: input.photoelectron_energy,
            quasiparticle_energy: input.quasiparticle_energy,
            quasiparticle_width: input.quasiparticle_width,
            plasma_frequency: input.plasma_frequency,
            renormalization_real: input.renormalization_real,
            renormalization_imag: input.renormalization_imag,
        })?;
        let interference =
            2.0 * input.renormalization_magnitude * input.interference_amplitude * peak;
        let width = input.boundaries[column + 1] - input.boundaries[column];

        main_peak[column] = peak;
        interference_peak[column] = interference;
        integrated_main += peak * input.exponential_reduction * width;
        integrated_interference += interference * input.exponential_reduction * width;
    }

    validate_finite_array("quasiparticle main row", main_peak.view())?;
    validate_finite_array("quasiparticle interference row", interference_peak.view())?;
    finite_result("quasiparticle integrated main weight", integrated_main)?;
    finite_result(
        "quasiparticle integrated interference weight",
        integrated_interference,
    )?;
    Ok(SfconvQuasiparticleTable {
        main_peak,
        interference_peak,
        integrated_main_weight: integrated_main,
        integrated_interference_weight: integrated_interference,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` quasiparticle-interference `ak` loop.
///
/// FEFF calls `xmkak(ekp)` once per active pole, multiplies by the empirical
/// `xreduc` factor and the pole weight, and accumulates the result into `ak`.
/// This helper preserves that accumulation and returns the combined integration
/// diagnostics from the underlying `xmkak` integrations.
pub fn sfconv_quasiparticle_interference_amplitude(
    input: SfconvQuasiparticleInterferenceInput<'_>,
) -> Result<SfconvQuasiparticleInterference, SfconvError> {
    validate_quasiparticle_interference_input(input)?;

    let mut amplitude = 0.0;
    let mut estimated_error = 0.0;
    let mut evaluations = 0;
    let mut max_regions = 0;

    for pole_index in 0..input.pole_count {
        let pole_weight = input.pole_weight[pole_index];
        let context = SfconvSatelliteContext {
            plasma_frequency: input.plasma_frequency,
            pole_energy: input.pole_energy[pole_index],
            dispersion_parameter: input.dispersion_parameter,
            photoelectron_energy: input.bare_photoelectron_energy,
            accuracy: input.accuracy,
        };
        let integral = sfconv_interference_quasiparticle(
            input.quasiparticle_energy,
            input.upper_energy,
            context,
        )?;
        let scale = input.interference_reduction * pole_weight;
        amplitude = finite_result(
            "quasiparticle interference amplitude",
            amplitude + integral.value * scale,
        )?;
        estimated_error = finite_result(
            "quasiparticle interference error",
            estimated_error + integral.estimated_error * scale.abs(),
        )?;
        evaluations += integral.evaluations;
        max_regions = max_regions.max(integral.max_regions);
    }

    Ok(SfconvQuasiparticleInterference {
        amplitude,
        estimated_error,
        evaluations,
        max_regions,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` satellite pole contribution loop.
///
/// FEFF chooses pole-local broadenings from `max(5*dw, brd)` for `xmkxsat` and
/// `max(2*dw, brd)` for `xmkisat`, optionally adds the quasiparticle width for
/// `isattype.eq.3`, then accumulates `xsat` and `xisat` using the active pole
/// weights. This helper preserves that loop around the already ported
/// `xmkxsat` and `xmkisat` integrators.
pub fn sfconv_satellite_pole_contributions(
    input: SfconvSatellitePoleContributionsInput<'_>,
) -> Result<SfconvSatellitePoleContributions, SfconvError> {
    validate_satellite_pole_contributions_input(input)?;

    let mut interference_satellite = 0.0;
    let mut intrinsic_satellite = 0.0;
    let mut interference_estimated_error = 0.0;
    let mut intrinsic_estimated_error = 0.0;
    let mut evaluations = 0;
    let mut max_regions = 0;

    for pole_index in 0..input.pole_count {
        let pole_weight = input.pole_weight[pole_index];
        let pole_broadening = input.pole_broadening[pole_index];
        let width_offset = if input.include_full_broadening {
            input.quasiparticle_width
        } else {
            0.0
        };
        let interference_width = finite_result(
            "interference satellite width",
            (5.0 * input.uniform_width).max(pole_broadening) + width_offset,
        )?;
        let intrinsic_width = finite_result(
            "intrinsic satellite width",
            (2.0 * input.uniform_width).max(pole_broadening) + width_offset,
        )?;
        validate_positive_scalar("interference satellite width", interference_width)?;
        validate_positive_scalar("intrinsic satellite width", intrinsic_width)?;

        let context = SfconvSatelliteContext {
            plasma_frequency: input.plasma_frequency,
            pole_energy: input.pole_energy[pole_index],
            dispersion_parameter: input.dispersion_parameter,
            photoelectron_energy: input.bare_photoelectron_energy,
            accuracy: input.accuracy,
        };
        let interference =
            sfconv_interference_satellite(input.energy, interference_width, context)?;
        let intrinsic = sfconv_intrinsic_satellite(input.energy, intrinsic_width, context)?;

        let interference_scale = input.interference_reduction * pole_weight;
        interference_satellite = finite_result(
            "interference satellite contribution",
            interference_satellite + interference.value * interference_scale,
        )?;
        intrinsic_satellite = finite_result(
            "intrinsic satellite contribution",
            intrinsic_satellite + intrinsic.value * pole_weight,
        )?;
        interference_estimated_error = finite_result(
            "interference satellite error",
            interference_estimated_error + interference.estimated_error * interference_scale.abs(),
        )?;
        intrinsic_estimated_error = finite_result(
            "intrinsic satellite error",
            intrinsic_estimated_error + intrinsic.estimated_error * pole_weight.abs(),
        )?;
        evaluations += interference.evaluations + intrinsic.evaluations;
        max_regions = max_regions
            .max(interference.max_regions)
            .max(intrinsic.max_regions);
    }

    Ok(SfconvSatellitePoleContributions {
        interference_satellite,
        intrinsic_satellite,
        interference_estimated_error,
        intrinsic_estimated_error,
        evaluations,
        max_regions,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` extrinsic satellite `isattype` branch.
///
/// FEFF selects one of four approximations for `esat`: the full-broadening
/// branch with `emain` removed, a local derivative expansion, the full
/// broadening branch, or the default de-broadened `xmkesat` expression.
pub fn sfconv_extrinsic_satellite(
    input: SfconvExtrinsicSatelliteInput,
) -> Result<Real, SfconvError> {
    validate_extrinsic_satellite_input(input)?;

    match input.mode {
        SfconvExtrinsicSatelliteMode::BroadenedMinusMain => finite_result(
            "extrinsic satellite",
            sfconv_extrinsic_satellite_broadened(input.energy, input.self_energy)?
                - input.main_peak,
        ),
        SfconvExtrinsicSatelliteMode::DerivativeExpansion => {
            validate_nonzero_denominator("derivative extrinsic satellite energy", input.energy)?;
            finite_result(
                "derivative extrinsic satellite",
                (input.self_energy.off_shell_imag
                    - input.self_energy.width
                    - input.energy * input.imaginary_derivative)
                    / (std::f64::consts::PI * input.energy.powi(2)),
            )
        }
        SfconvExtrinsicSatelliteMode::FullBroadening => {
            sfconv_extrinsic_satellite_broadened(input.energy, input.self_energy)
        }
        SfconvExtrinsicSatelliteMode::Debroadened => {
            sfconv_extrinsic_satellite_debroadened(input.energy, input.context, input.self_energy)
        }
    }
}

/// Port of one iteration of FEFF `SFCONV/mkspectf.f90` spectral row assembly.
///
/// This helper computes `emain`, `xmain`, `esat`, `xsat`, `xisat`, and the
/// combined row for one energy cell. Later table-level helpers still handle the
/// endpoint average, satellite split, clipping, and final weights.
pub fn sfconv_spectral_cell(
    input: SfconvSpectralCellInput<'_>,
) -> Result<SfconvSpectralCell, SfconvError> {
    validate_spectral_cell_input(input)?;

    let main_peak = sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
        center_energy: input.center_energy,
        lower_boundary: input.lower_boundary,
        upper_boundary: input.upper_boundary,
        photoelectron_energy: input.photoelectron_energy,
        quasiparticle_energy: input.quasiparticle_energy,
        quasiparticle_width: input.quasiparticle_width,
        plasma_frequency: input.context.plasma_frequency,
        renormalization_real: input.self_energy.renormalization_real,
        renormalization_imag: input.self_energy.renormalization_imag,
    })?;
    let renormalization_magnitude = checked_hypot(
        "spectral cell renormalization",
        input.self_energy.renormalization_real,
        input.self_energy.renormalization_imag,
    )?;
    let quasiparticle_interference = finite_result(
        "spectral cell quasiparticle interference",
        2.0 * renormalization_magnitude * input.interference_amplitude * main_peak,
    )?;
    let extrinsic_satellite = sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
        energy: input.center_energy,
        main_peak,
        imaginary_derivative: input.imaginary_derivative,
        mode: input.extrinsic_mode,
        context: input.context,
        self_energy: input.self_energy,
    })?;
    let satellite = sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
        energy: input.center_energy,
        uniform_width: input.uniform_width,
        quasiparticle_width: input.self_energy.width,
        plasma_frequency: input.context.plasma_frequency,
        bare_photoelectron_energy: input.context.photoelectron_energy,
        dispersion_parameter: input.context.dispersion_parameter,
        accuracy: input.context.accuracy,
        interference_reduction: input.interference_reduction,
        include_full_broadening: matches!(
            input.extrinsic_mode,
            SfconvExtrinsicSatelliteMode::FullBroadening
        ),
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
    })?;
    let mut combined_satellite = finite_result(
        "spectral cell combined satellite",
        extrinsic_satellite + satellite.intrinsic_satellite
            - 2.0 * satellite.interference_satellite,
    )?;
    if matches!(
        input.extrinsic_mode,
        SfconvExtrinsicSatelliteMode::FullBroadening
    ) {
        combined_satellite = finite_result(
            "spectral cell combined satellite",
            combined_satellite + quasiparticle_interference,
        )?;
    }

    Ok(SfconvSpectralCell {
        main_peak,
        extrinsic_satellite,
        quasiparticle_interference,
        interference_satellite: satellite.interference_satellite,
        intrinsic_satellite: satellite.intrinsic_satellite,
        combined_satellite,
        interference_estimated_error: satellite.interference_estimated_error,
        intrinsic_estimated_error: satellite.intrinsic_estimated_error,
        evaluations: satellite.evaluations,
        max_regions: satellite.max_regions,
    })
}

/// Port of the FEFF `SFCONV/mkspectf.f90` spectral-function cell loop.
///
/// This assembles rows 1 through 6 from the per-cell helper, preserves FEFF's
/// endpoint-corrected quasiparticle accumulators, and applies the legacy
/// average of the two quasiparticle-adjacent extrinsic-satellite cells. Later
/// helpers still split the extrinsic satellite, clip negative satellite weight,
/// and write the final eight-slot weight vector.
pub fn sfconv_spectral_table(
    input: SfconvSpectralTableInput<'_>,
) -> Result<SfconvSpectralTable, SfconvError> {
    validate_spectral_table_input(input)?;

    let columns = input.energy.len();
    let renormalization_magnitude = checked_hypot(
        "spectral table renormalization",
        input.self_energy.renormalization_real,
        input.self_energy.renormalization_imag,
    )?;
    let pi = std::f64::consts::PI;
    let endpoint_main = ((input.boundaries[0] / input.self_energy.width).atan() + pi / 2.0) / pi
        + (pi / 2.0
            - (input.boundaries[input.boundaries.len() - 1] / input.self_energy.width).atan())
            / pi;
    let mut integrated_quasiparticle_interference = finite_result(
        "spectral table integrated quasiparticle interference weight",
        2.0 * endpoint_main
            * renormalization_magnitude
            * input.self_energy.renormalization_real
            * input.interference_amplitude,
    )?;
    let mut integrated_main = finite_result(
        "spectral table integrated main weight",
        endpoint_main * input.self_energy.renormalization_real * input.exponential_reduction,
    )?;
    let mut integrated_extrinsic = 0.0;
    let mut integrated_interference = 0.0;
    let mut integrated_intrinsic = 0.0;
    let mut interference_estimated_error = 0.0;
    let mut intrinsic_estimated_error = 0.0;
    let mut evaluations = 0;
    let mut max_regions = 0;
    let mut spectral_function = Array2::<Real>::zeros((8, columns));

    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        let self_energy = SfconvSatelliteSelfEnergy {
            off_shell_real: input.off_shell_real[column],
            off_shell_imag: input.off_shell_imag[column],
            ..input.self_energy
        };
        let cell = sfconv_spectral_cell(SfconvSpectralCellInput {
            center_energy: input.energy[column],
            lower_boundary: input.boundaries[column],
            upper_boundary: input.boundaries[column + 1],
            photoelectron_energy: input.photoelectron_energy,
            quasiparticle_energy: input.quasiparticle_energy,
            quasiparticle_width: input.quasiparticle_width,
            interference_amplitude: input.interference_amplitude,
            extrinsic_mode: input.extrinsic_mode,
            imaginary_derivative: input.imaginary_derivative,
            uniform_width: input.uniform_width,
            interference_reduction: input.interference_reduction,
            context: input.context,
            self_energy,
            pole_count: input.pole_count,
            pole_energy: input.pole_energy,
            pole_weight: input.pole_weight,
            pole_broadening: input.pole_broadening,
        })?;

        integrated_main = finite_result(
            "spectral table integrated main weight",
            integrated_main + cell.main_peak * input.exponential_reduction * width,
        )?;
        integrated_quasiparticle_interference = finite_result(
            "spectral table integrated quasiparticle interference weight",
            integrated_quasiparticle_interference
                + cell.quasiparticle_interference * input.exponential_reduction * width,
        )?;
        integrated_extrinsic = finite_result(
            "spectral table integrated extrinsic weight",
            integrated_extrinsic + cell.extrinsic_satellite * input.exponential_reduction * width,
        )?;
        integrated_interference = finite_result(
            "spectral table integrated interference weight",
            integrated_interference
                + cell.interference_satellite * input.exponential_reduction * width,
        )?;
        integrated_intrinsic = finite_result(
            "spectral table integrated intrinsic weight",
            integrated_intrinsic + cell.intrinsic_satellite * input.exponential_reduction * width,
        )?;
        interference_estimated_error = finite_result(
            "spectral table interference satellite error",
            interference_estimated_error + cell.interference_estimated_error,
        )?;
        intrinsic_estimated_error = finite_result(
            "spectral table intrinsic satellite error",
            intrinsic_estimated_error + cell.intrinsic_estimated_error,
        )?;
        evaluations += cell.evaluations;
        max_regions = max_regions.max(cell.max_regions);

        spectral_function[(0, column)] = cell.main_peak;
        spectral_function[(1, column)] = cell.extrinsic_satellite;
        spectral_function[(2, column)] = cell.quasiparticle_interference;
        spectral_function[(3, column)] = cell.interference_satellite;
        spectral_function[(4, column)] = cell.intrinsic_satellite;
        spectral_function[(5, column)] = cell.combined_satellite;
    }

    let lower_column = feff_index(input.quasiparticle_lower_column_1based);
    let upper_column = feff_index(input.quasiparticle_upper_column_1based);
    let averaged_extrinsic =
        0.5 * (spectral_function[(1, lower_column)] + spectral_function[(1, upper_column)]);
    spectral_function[(1, lower_column)] = averaged_extrinsic;
    spectral_function[(1, upper_column)] = averaged_extrinsic;

    validate_finite_spectral_rows(spectral_function.view())?;
    Ok(SfconvSpectralTable {
        spectral_function,
        integrated_main_weight: integrated_main,
        integrated_quasiparticle_interference_weight: integrated_quasiparticle_interference,
        integrated_extrinsic_weight: integrated_extrinsic,
        integrated_interference_weight: integrated_interference,
        integrated_intrinsic_weight: integrated_intrinsic,
        interference_estimated_error,
        intrinsic_estimated_error,
        evaluations,
        max_regions,
    })
}

/// Port of the `SFCONV/mkspectf.f90` satellite row assembly.
///
/// FEFF fills rows 2, 4, and 5 from the extrinsic, interference, and intrinsic
/// satellite estimates, forms row 6 as their combined satellite contribution,
/// and accumulates the raw satellite weights before later splitting and
/// clipping. The extrinsic satellite is then averaged across the two
/// quasiparticle-adjacent cells, preserving FEFF's order of operations.
pub fn sfconv_satellite_table(
    input: SfconvSatelliteTableInput<'_>,
) -> Result<SfconvSatelliteTable, SfconvError> {
    validate_satellite_table_input(input)?;

    let columns = input.extrinsic_satellite.len();
    let mut spectral_function = Array2::<Real>::zeros((8, columns));
    let mut integrated_extrinsic = 0.0;
    let mut integrated_interference = 0.0;
    let mut integrated_intrinsic = 0.0;

    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        let extrinsic = input.extrinsic_satellite[column];
        let interference = input.interference_satellite[column];
        let intrinsic = input.intrinsic_satellite[column];
        let quasiparticle_interference = input.quasiparticle_interference[column];
        let mut combined = extrinsic + intrinsic - 2.0 * interference;
        if input.include_full_broadening_quasiparticle {
            combined += quasiparticle_interference;
        }

        integrated_extrinsic += extrinsic * width * input.exponential_reduction;
        integrated_interference += interference * width * input.exponential_reduction;
        integrated_intrinsic += intrinsic * width * input.exponential_reduction;

        spectral_function[(0, column)] = input.main_peak[column];
        spectral_function[(1, column)] = extrinsic;
        spectral_function[(2, column)] = quasiparticle_interference;
        spectral_function[(3, column)] = interference;
        spectral_function[(4, column)] = intrinsic;
        spectral_function[(5, column)] = combined;
    }

    let lower_column = feff_index(input.quasiparticle_lower_column_1based);
    let upper_column = feff_index(input.quasiparticle_upper_column_1based);
    let averaged_extrinsic =
        0.5 * (spectral_function[(1, lower_column)] + spectral_function[(1, upper_column)]);
    spectral_function[(1, lower_column)] = averaged_extrinsic;
    spectral_function[(1, upper_column)] = averaged_extrinsic;

    validate_finite_array("satellite table main row", spectral_function.row(0))?;
    validate_finite_array("satellite table extrinsic row", spectral_function.row(1))?;
    validate_finite_array(
        "satellite table quasiparticle row",
        spectral_function.row(2),
    )?;
    validate_finite_array("satellite table interference row", spectral_function.row(3))?;
    validate_finite_array("satellite table intrinsic row", spectral_function.row(4))?;
    validate_finite_array("satellite table combined row", spectral_function.row(5))?;
    finite_result(
        "satellite integrated extrinsic weight",
        integrated_extrinsic,
    )?;
    finite_result(
        "satellite integrated interference weight",
        integrated_interference,
    )?;
    finite_result(
        "satellite integrated intrinsic weight",
        integrated_intrinsic,
    )?;
    Ok(SfconvSatelliteTable {
        spectral_function,
        integrated_extrinsic_weight: integrated_extrinsic,
        integrated_interference_weight: integrated_interference,
        integrated_intrinsic_weight: integrated_intrinsic,
    })
}

/// Port of the `SFCONV/mkspectf.f90` extrinsic-satellite split.
///
/// FEFF scans the extrinsic satellite row from high to low energy, finds the
/// first derivative or curvature trigger after the satellite begins rising,
/// then copies `spectf(2)` into row 7 below that switch and row 8 at and above
/// it. The legacy code currently sets the smoothing width to zero, so this
/// helper preserves the resulting sharp split.
pub fn sfconv_split_extrinsic_satellite(
    input: SfconvExtrinsicSatelliteSplitInput<'_>,
) -> Result<SfconvExtrinsicSatelliteSplit, SfconvError> {
    validate_extrinsic_satellite_split_input(input)?;

    let columns = input.spectral_function.ncols();
    let mut derivative_switch = None;
    let mut curvature_switch = None;
    let mut satellite_started = false;

    for ii_1based in 2..columns {
        let column = columns - ii_1based;
        let satellite = input.spectral_function[(1, column)];
        let slope = (satellite - input.spectral_function[(1, column - 1)])
            / (input.energy[column] - input.energy[column - 1]);
        let high_slope = (input.spectral_function[(1, column + 1)] - satellite)
            / (input.energy[column + 1] - input.energy[column]);
        let curvature =
            (high_slope - slope) / (input.boundaries[column + 1] - input.boundaries[column]);
        let absolute_energy = input.energy[column] + input.photoelectron_energy;

        if slope > 0.0 && satellite > 0.0 {
            satellite_started = true;
        }

        let derivative_allowed = input.beta_zero > 0.0 || absolute_energy > 0.0;
        if slope < 0.0 && satellite_started && derivative_allowed && derivative_switch.is_none() {
            derivative_switch = Some((column, absolute_energy));
        }
        if curvature > 0.0 && satellite_started && curvature_switch.is_none() {
            curvature_switch = Some((column, absolute_energy));
        }
    }

    let (switch_column, switch_energy, derivative_triggered) =
        if let Some((column, energy)) = derivative_switch {
            (column, energy, true)
        } else if let Some((column, energy)) = curvature_switch {
            (column, energy, false)
        } else {
            return Err(SfconvError::MissingTrigger {
                field: "extrinsic satellite split",
            });
        };

    let mut spectral_function = input.spectral_function.to_owned();
    for column in 0..columns {
        spectral_function[(6, column)] = 0.0;
        spectral_function[(7, column)] = 0.0;
        if column >= switch_column {
            spectral_function[(7, column)] = spectral_function[(1, column)];
        } else {
            spectral_function[(6, column)] = spectral_function[(1, column)];
        }
    }

    validate_finite_array("extrinsic split row 7", spectral_function.row(6))?;
    validate_finite_array("extrinsic split row 8", spectral_function.row(7))?;
    finite_result("extrinsic split switch energy", switch_energy)?;
    Ok(SfconvExtrinsicSatelliteSplit {
        spectral_function,
        switch_column,
        switch_energy,
        derivative_triggered,
    })
}

/// Port of the final `SFCONV/mkspectf.f90` satellite clipping correction.
///
/// FEFF first forms the combined satellite row
/// `spectf(6)=spectf(2)-2*spectf(4)+spectf(5)`. Negative combined satellite
/// cells are clipped to zero, the surviving positive part is renormalized to
/// preserve the original integral, and the interference row is recomputed so
/// downstream interpolation sees the corrected combined satellite. The returned
/// weights are FEFF `weights(4:8)`.
pub fn sfconv_correct_satellite_weights(
    input: SfconvSatelliteCorrectionInput<'_>,
) -> Result<SfconvSatelliteCorrection, SfconvError> {
    validate_satellite_correction_input(input)?;

    let columns = input.spectral_function.ncols();
    let mut corrected = input.spectral_function.to_owned();
    for column in 0..columns {
        corrected[(5, column)] =
            corrected[(1, column)] - 2.0 * corrected[(3, column)] + corrected[(4, column)];
    }

    let mut clipped_negative_weight = 0.0;
    let mut uncorrected_satellite_weight = 0.0;
    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        let combined = corrected[(5, column)];
        uncorrected_satellite_weight += combined * width;
        if combined < 0.0 {
            clipped_negative_weight += combined * width;
            corrected[(5, column)] = 0.0;
            corrected[(3, column)] = 0.5 * (corrected[(1, column)] + corrected[(4, column)]);
        }
    }

    let correction_denominator = uncorrected_satellite_weight - clipped_negative_weight;
    validate_nonzero_denominator("satellite correction", correction_denominator)?;
    let correction_factor = (uncorrected_satellite_weight / correction_denominator).max(0.0);

    let mut weights = Array1::<Real>::zeros(5);
    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        corrected[(3, column)] = 0.5
            * (corrected[(1, column)] + corrected[(4, column)]
                - corrected[(5, column)] * correction_factor);
        weights[0] += corrected[(1, column)] * width * input.exponential_reduction;
        weights[1] += corrected[(3, column)] * width * input.exponential_reduction;
        weights[2] += corrected[(4, column)] * width * input.exponential_reduction;
        weights[3] += corrected[(7, column)] * input.exponential_reduction * input.uniform_width;
        weights[4] += corrected[(6, column)] * input.exponential_reduction * input.uniform_width;
    }

    validate_finite_array("satellite correction weights", weights.view())?;
    validate_finite_spectral_rows(corrected.view())?;
    finite_result("uncorrected satellite weight", uncorrected_satellite_weight)?;
    finite_result("clipped satellite weight", clipped_negative_weight)?;
    finite_result("satellite correction factor", correction_factor)?;
    Ok(SfconvSatelliteCorrection {
        spectral_function: corrected,
        weights,
        uncorrected_satellite_weight,
        clipped_negative_weight,
        correction_factor,
    })
}

/// Port of the final FEFF `SFCONV/mkspectf.f90` postprocessing sequence.
///
/// FEFF first splits the extrinsic satellite into satellite-like and
/// quasiparticle-like rows, then clips negative combined satellite weight, and
/// finally writes the eight `weights` values. This helper chains the already
/// ported stages without changing their order or formulas.
pub fn sfconv_finalize_spectral_table(
    input: SfconvSpectralFinalizationInput<'_>,
) -> Result<SfconvSpectralFinalization, SfconvError> {
    validate_spectral_finalization_input(input)?;

    let split = sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
        spectral_function: input.spectral_function,
        energy: input.energy,
        boundaries: input.boundaries,
        photoelectron_energy: input.photoelectron_energy,
        beta_zero: input.beta_zero,
    })?;
    let correction = sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
        spectral_function: split.spectral_function.view(),
        boundaries: input.boundaries,
        uniform_width: input.uniform_width,
        exponential_reduction: input.exponential_reduction,
    })?;
    let weights = sfconv_spectral_weights(SfconvSpectralWeightsInput {
        renormalization_real: input.renormalization_real,
        renormalization_imag: input.renormalization_imag,
        renormalization_magnitude: input.renormalization_magnitude,
        interference_amplitude: input.interference_amplitude,
        interference_reduction: input.interference_reduction,
        exponential_reduction: input.exponential_reduction,
        satellite_weights: correction.weights.view(),
    })?;

    Ok(SfconvSpectralFinalization {
        spectral_function: correction.spectral_function,
        weights,
        switch_column: split.switch_column,
        switch_energy: split.switch_energy,
        derivative_triggered: split.derivative_triggered,
        uncorrected_satellite_weight: correction.uncorrected_satellite_weight,
        clipped_negative_weight: correction.clipped_negative_weight,
        correction_factor: correction.correction_factor,
    })
}

/// Port of the final FEFF `SFCONV/mkspectf.f90` `weights(1:8)` assignment.
///
/// FEFF does not use the endpoint-corrected quasiparticle accumulators for the
/// final array. It writes the first three slots directly from the
/// renormalization constants and interference amplitude, then copies the five
/// corrected satellite weights into slots 4 through 8.
pub fn sfconv_spectral_weights(
    input: SfconvSpectralWeightsInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_spectral_weights_input(input)?;

    let mut weights = Array1::<Real>::zeros(8);
    weights[0] = input.renormalization_real * input.exponential_reduction;
    weights[1] = input.renormalization_imag * input.exponential_reduction;
    weights[2] = 2.0
        * input.renormalization_real
        * input.renormalization_magnitude
        * input.interference_amplitude
        * input.interference_reduction
        * input.exponential_reduction;
    for (index, weight) in input.satellite_weights.iter().copied().enumerate() {
        weights[index + 3] = weight;
    }

    validate_finite_array("spectral weights", weights.view())?;
    Ok(weights)
}

/// Port of the `SO2CONV` `feffNNNN.dat` interpolation loop.
///
/// FEFF first interpolates `caph2`, `xmfeff2`, `phfeff2`, `redfac2`, and
/// `xlam2` from a coarse path grid onto the uniform SO2CONV momentum grid. Rows
/// outside the coarse path range remain zero, while a source point exactly at
/// the final coarse momentum receives the final path row.
pub fn sfconv_interpolate_feff_path(
    input: SfconvFeffPathInterpolationInput<'_>,
) -> Result<SfconvFeffPathInterpolation, SfconvError> {
    validate_feff_path_interpolation_input(input)?;

    let mut output = SfconvFeffPathInterpolation {
        central_phase: Array1::<Real>::zeros(input.source_momentum.len()),
        effective_amplitude: Array1::<Real>::zeros(input.source_momentum.len()),
        effective_phase: Array1::<Real>::zeros(input.source_momentum.len()),
        reduction_factor: Array1::<Real>::zeros(input.source_momentum.len()),
        mean_free_path: Array1::<Real>::zeros(input.source_momentum.len()),
    };

    let last_path_row = input.path_momentum.len() - 1;
    for (source_row, &momentum) in input.source_momentum.iter().enumerate() {
        let mut matched_segment = None;
        for segment in 0..last_path_row {
            if momentum >= input.path_momentum[segment]
                && momentum < input.path_momentum[segment + 1]
            {
                matched_segment = Some(segment);
                break;
            }
        }

        if let Some(segment) = matched_segment {
            set_feff_path_interpolated_row(&mut output, source_row, input, segment)?;
        } else if momentum == input.path_momentum[last_path_row] {
            set_feff_path_exact_row(&mut output, source_row, input, last_path_row);
        }
    }

    validate_finite_array("interpolated central_phase", output.central_phase.view())?;
    validate_finite_array(
        "interpolated effective_amplitude",
        output.effective_amplitude.view(),
    )?;
    validate_finite_array(
        "interpolated effective_phase",
        output.effective_phase.view(),
    )?;
    validate_finite_array(
        "interpolated reduction_factor",
        output.reduction_factor.view(),
    )?;
    validate_finite_array("interpolated mean_free_path", output.mean_free_path.view())?;
    Ok(output)
}

/// Port of the `SO2CONV` raw EXAFS signal loop for `feffNNNN.dat` rows.
///
/// FEFF builds the unconvoluted complex path signal from interpolated path
/// columns before applying the spectral-function convolution. The first
/// magnitude row is linearly extrapolated from rows two and three, matching the
/// historical `xmag(1)` fixup that avoids the singular `k = 0` row.
pub fn sfconv_feff_path_signal(
    input: SfconvFeffPathSignalInput<'_>,
) -> Result<SfconvFeffPathSignal, SfconvError> {
    validate_feff_path_signal_input(input)?;

    let len = input.momentum.len();
    let mut output = SfconvFeffPathSignal {
        magnitude: Array1::<Real>::zeros(len),
        phase_minus_2kr: Array1::<Real>::zeros(len),
        phase: Array1::<Real>::zeros(len),
        real: Array1::<Real>::zeros(len),
        imaginary: Array1::<Real>::zeros(len),
    };

    for row in 0..len {
        output.phase_minus_2kr[row] = input.effective_phase[row] + input.central_phase[row];
        output.phase[row] =
            output.phase_minus_2kr[row] + 2.0 * input.momentum[row] * input.half_path_length;
    }

    for row in 1..len {
        output.magnitude[row] = feff_path_signal_magnitude(input, row)?;
        output.real[row] = output.magnitude[row] * output.phase[row].cos();
        output.imaginary[row] = output.magnitude[row] * output.phase[row].sin();
    }

    let extrapolation_denominator = input.momentum[2] - input.momentum[1];
    validate_nonzero_denominator(
        "feff path signal first-row extrapolation",
        extrapolation_denominator,
    )?;
    output.magnitude[0] = output.magnitude[1]
        + (input.momentum[0] - input.momentum[1]) * (output.magnitude[2] - output.magnitude[1])
            / extrapolation_denominator;
    output.real[0] = output.magnitude[0] * output.phase[0].cos();
    output.imaginary[0] = output.magnitude[0] * output.phase[0].sin();

    validate_finite_array("path signal magnitude", output.magnitude.view())?;
    validate_finite_array("path signal phase_minus_2kr", output.phase_minus_2kr.view())?;
    validate_finite_array("path signal phase", output.phase.view())?;
    validate_finite_array("path signal real", output.real.view())?;
    validate_finite_array("path signal imaginary", output.imaginary.view())?;
    Ok(output)
}

/// Port of the `SO2CONV` EXAFS post-convolution row calculation.
///
/// FEFF convolves the real and imaginary EXAFS channels separately, combines
/// their magnitudes/phases into a complex many-body signal, removes `2 pi`
/// phase jumps with the legacy `npi` state, and stores the amplitude/phase
/// correction arrays later averaged back onto `feffNNNN.dat` path grids.
pub fn sfconv_exafs_convolution(
    input: SfconvExafsConvolutionInput,
) -> Result<SfconvExafsConvolution, SfconvError> {
    validate_exafs_convolution_input(input)?;

    let real = input.real_convolution_amplitude * input.real_convolution_phase.cos()
        - input.imaginary_convolution_amplitude * input.imaginary_convolution_phase.sin();
    let imaginary = input.imaginary_convolution_amplitude * input.imaginary_convolution_phase.cos()
        + input.real_convolution_amplitude * input.real_convolution_phase.sin();
    let magnitude = checked_hypot("exafs convolution magnitude", real, imaginary)?;
    let raw_phase = finite_result("exafs convolution phase", imaginary.atan2(real))?;
    let phase_jump_count =
        so2conv_update_phase_jump_count(input.phase_jump_count, raw_phase, input.previous_phase)?;
    let output_phase = finite_result(
        "exafs output phase",
        raw_phase - std::f64::consts::PI * Real::from(phase_jump_count),
    )?;

    Ok(SfconvExafsConvolution {
        real: finite_result("exafs convolution real", real)?,
        imaginary: finite_result("exafs convolution imaginary", imaginary)?,
        magnitude,
        output_phase,
        output_phase_minus_original: finite_result(
            "exafs output phase correction",
            output_phase + input.phase_minus_2kr - input.original_phase,
        )?,
        amplitude_reduction: finite_result(
            "exafs amplitude reduction",
            magnitude / input.original_magnitude,
        )?,
        phase_shift: finite_result("exafs phase shift", output_phase - input.original_phase)?,
        previous_phase: raw_phase,
        phase_jump_count,
    })
}

/// Port of the `SO2CONV` XANES post-convolution row calculation.
///
/// FEFF either uses a real-valued asymmetric convolution result directly as
/// `xmu2`, or recombines real and imaginary fine-structure convolution channels
/// as `ximu2*cos(phmu) + rmu2*sin(phrmu) + xmu02`. FEFF10 writes the
/// unnormalized fine structure `xmu2 - xmu02`.
pub fn sfconv_xanes_convolution(
    input: SfconvXanesConvolutionInput,
) -> Result<SfconvXanesConvolution, SfconvError> {
    validate_xanes_convolution_input(input)?;

    let background = input.embedded_background;
    let absorption = if input.asymmetric_phase {
        input.absorption_convolution
    } else {
        input.fine_structure_imaginary_amplitude * input.fine_structure_imaginary_phase.cos()
            + input.fine_structure_real_amplitude * input.fine_structure_real_phase.sin()
            + background
    };

    let absorption = finite_result("xanes absorption", absorption)?;
    Ok(SfconvXanesConvolution {
        absorption,
        embedded_background: background,
        fine_structure: finite_result("xanes fine structure", absorption - background)?,
    })
}

/// Port of the `SO2CONV` EXAFS energy-grid padding loop.
///
/// FEFF extends `epts2` from the last two active rows through the full
/// convolution work-array length so endpoint interpolation in `sfconvsub` has a
/// flat continuation beyond the rows read from `chi.dat`, `chipNNNN.dat`, or
/// `feffNNNN.dat`.
pub fn sfconv_so2conv_pad_exafs_energy_grid(
    input: SfconvSo2convExafsEnergyPaddingInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_so2conv_exafs_energy_padding_input(input)?;

    let mut energy = Array1::<Real>::zeros(input.output_len);
    for row in 0..input.active_len {
        energy[row] = input.energy[row];
    }

    let step = energy[input.active_len - 1] - energy[input.active_len - 2];
    for row in input.active_len..input.output_len {
        energy[row] = finite_result("so2conv padded exafs energy", energy[row - 1] + step)?;
    }

    validate_finite_array("so2conv padded exafs energy", energy.view())?;
    Ok(energy)
}

/// Port of the `SO2CONV` EXAFS channel preparation loops.
///
/// FEFF converts the input `xk` grid to `epts2`, decomposes the magnitude and
/// phase columns into real and imaginary EXAFS channels, leaves padded signal
/// rows at zero, and then extends only the energy grid to the full convolution
/// work-array length.
pub fn sfconv_so2conv_prepare_exafs_signal(
    input: SfconvSo2convExafsPreparationInput<'_>,
) -> Result<SfconvSo2convExafsPreparation, SfconvError> {
    validate_so2conv_exafs_preparation_input(input)?;

    let mut signal_energy = Array1::<Real>::zeros(input.output_len);
    let mut real_signal = Array1::<Real>::zeros(input.output_len);
    let mut imaginary_signal = Array1::<Real>::zeros(input.output_len);
    let mut original_magnitude = Array1::<Real>::zeros(input.output_len);
    let mut original_phase = Array1::<Real>::zeros(input.output_len);
    let mut phase_minus_2kr = Array1::<Real>::zeros(input.output_len);

    for row in 0..input.active_len {
        let momentum = input.momentum[row];
        let energy = if momentum >= 0.0 {
            momentum.powi(2) / 2.0 + input.chemical_potential
        } else {
            -momentum.powi(2) / 2.0 + input.chemical_potential
        };
        signal_energy[row] = finite_result("so2conv exafs energy", energy)?;
        original_magnitude[row] = input.magnitude[row];
        original_phase[row] = input.phase[row];
        phase_minus_2kr[row] = input.phase_minus_2kr.map_or(0.0, |values| values[row]);
        real_signal[row] = finite_result(
            "so2conv exafs real signal",
            input.magnitude[row] * input.phase[row].cos(),
        )?;
        imaginary_signal[row] = finite_result(
            "so2conv exafs imaginary signal",
            input.magnitude[row] * input.phase[row].sin(),
        )?;
    }

    signal_energy = sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
        energy: signal_energy.view(),
        active_len: input.active_len,
        output_len: input.output_len,
    })?;

    validate_finite_array("so2conv exafs real signal", real_signal.view())?;
    validate_finite_array("so2conv exafs imaginary signal", imaginary_signal.view())?;
    validate_finite_array(
        "so2conv exafs original magnitude",
        original_magnitude.view(),
    )?;
    validate_finite_array("so2conv exafs original phase", original_phase.view())?;
    validate_finite_array("so2conv exafs phase minus 2kr", phase_minus_2kr.view())?;

    Ok(SfconvSo2convExafsPreparation {
        signal_energy,
        real_signal,
        imaginary_signal,
        original_magnitude,
        original_phase,
        phase_minus_2kr,
    })
}

/// Port of the `SO2CONV` XANES signal preparation loop.
///
/// FEFF pads `xmu.dat` by overwriting rows `j..npts2` with a flat
/// embedded-atom background, then computes `rmu` with `mkrmu` and `ximu` as the
/// residual `xmu - xmu0`. The one-based FEFF row `j` maps to
/// `active_len - 1`, so the last active row is intentionally replaced.
pub fn sfconv_so2conv_prepare_xanes_signal(
    input: SfconvSo2convXanesPreparationInput<'_>,
) -> Result<SfconvSo2convXanesPreparation, SfconvError> {
    validate_so2conv_xanes_preparation_input(input)?;

    let mut incident_energy = Array1::<Real>::zeros(input.output_len);
    let mut excitation_energy = Array1::<Real>::zeros(input.output_len);
    let mut absorption = Array1::<Real>::zeros(input.output_len);
    let mut embedded_background = Array1::<Real>::zeros(input.output_len);

    for row in 0..input.active_len {
        incident_energy[row] = input.incident_energy[row];
        excitation_energy[row] = input.excitation_energy[row];
        absorption[row] = input.absorption[row];
        embedded_background[row] = input.embedded_background[row];
    }

    let step = excitation_energy[input.active_len - 1] - excitation_energy[input.active_len - 2];
    let tail_background = embedded_background[input.active_len - 1];
    for row in (input.active_len - 1)..input.output_len {
        incident_energy[row] = finite_result(
            "so2conv padded xanes incident energy",
            incident_energy[row - 1] + step,
        )?;
        excitation_energy[row] = finite_result(
            "so2conv padded xanes excitation energy",
            excitation_energy[row - 1] + step,
        )?;
        embedded_background[row] = tail_background;
        absorption[row] = tail_background;
    }

    let real_fine_structure = sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
        imaginary: absorption.view(),
        reference_imaginary: embedded_background.view(),
        energy: excitation_energy.view(),
        active_len: input.output_len,
    })?;
    let mut imaginary_fine_structure = Array1::<Real>::zeros(input.output_len);
    for row in 0..input.output_len {
        imaginary_fine_structure[row] = finite_result(
            "so2conv xanes imaginary fine structure",
            absorption[row] - embedded_background[row],
        )?;
    }

    validate_finite_array(
        "so2conv padded xanes incident energy",
        incident_energy.view(),
    )?;
    validate_finite_array(
        "so2conv padded xanes excitation energy",
        excitation_energy.view(),
    )?;
    validate_finite_array("so2conv padded xanes absorption", absorption.view())?;
    validate_finite_array(
        "so2conv padded xanes embedded_background",
        embedded_background.view(),
    )?;
    validate_finite_array(
        "so2conv xanes imaginary fine structure",
        imaginary_fine_structure.view(),
    )?;

    Ok(SfconvSo2convXanesPreparation {
        incident_energy,
        excitation_energy,
        absorption,
        embedded_background,
        imaginary_fine_structure,
        real_fine_structure,
    })
}

/// Port of the `SO2CONV` triangular average for one FEFF path row.
///
/// FEFF computes `s02list` and `phlist` on a dense uniform momentum grid, then
/// averages nearby dense rows back onto the coarser `feffNNNN.dat` path grid
/// with a triangular finite-element weight. This helper returns the two
/// averaged values before the caller applies them to `redfac2` and `caph2`.
pub fn sfconv_path_average(
    input: SfconvPathAverageInput<'_>,
) -> Result<SfconvPathAverage, SfconvError> {
    validate_path_average_input(input)?;

    let mut amplitude_sum = 0.0;
    let mut phase_sum = 0.0;
    let mut normalization = 0.0;

    for ((&momentum, &amplitude), &phase) in input
        .source_momentum
        .iter()
        .zip(input.amplitude_reduction.iter())
        .zip(input.phase_shift.iter())
    {
        let weight = if momentum == input.center_momentum {
            1.0
        } else if momentum > input.previous_momentum
            && momentum <= input.center_momentum
            && input.previous_momentum != input.center_momentum
        {
            (momentum - input.previous_momentum) / (input.center_momentum - input.previous_momentum)
        } else if momentum > input.center_momentum
            && momentum < input.next_momentum
            && input.next_momentum != input.center_momentum
        {
            (input.next_momentum - momentum) / (input.next_momentum - input.center_momentum)
        } else {
            0.0
        };

        amplitude_sum += amplitude * weight * input.momentum_step;
        phase_sum += phase * weight * input.momentum_step;
        normalization += weight * input.momentum_step;
    }

    validate_nonzero_denominator("path average normalization", normalization)?;
    Ok(SfconvPathAverage {
        amplitude_reduction: finite_result(
            "path average amplitude",
            amplitude_sum / normalization,
        )?,
        phase_shift: finite_result("path average phase", phase_sum / normalization)?,
        normalization: finite_result("path average normalization", normalization)?,
    })
}

/// Port of `SFCONV/senergies.f90` `rseint1`.
pub fn sfconv_real_self_energy_integrand_upper(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let regularization = (context.accuracy * context.pole_energy).powi(2);
    let numerator = ((context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy
        + dispersion)
        .powi(2)
        + regularization;
    let denominator = ((context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy
        + dispersion)
        .powi(2)
        + regularization;
    real_self_energy_log_integrand(
        "real self-energy upper integrand",
        momentum,
        context,
        dispersion,
        numerator,
        denominator,
    )
}

/// Port of `SFCONV/senergies.f90` `rseint2`.
pub fn sfconv_real_self_energy_integrand_middle(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let regularization = (context.accuracy * context.pole_energy).powi(2);
    let mut ratio = 1.0;
    if context.include_below_fermi {
        let below_numerator =
            (context.fermi_energy - shifted_energy - dispersion).powi(2) + regularization;
        let below_denominator = ((context.photoelectron_momentum - momentum).powi(2) / 2.0
            - shifted_energy
            - dispersion)
            .powi(2)
            + regularization;
        validate_nonzero_denominator(
            "middle real self-energy below denominator",
            below_denominator,
        )?;
        ratio *= below_numerator / below_denominator;
    }
    let numerator = ((context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy
        + dispersion)
        .powi(2)
        + regularization;
    let denominator = (context.fermi_energy - shifted_energy + dispersion).powi(2) + regularization;
    ratio *= numerator;
    ratio /= denominator;
    real_self_energy_log_integrand_with_ratio(
        "real self-energy middle integrand",
        momentum,
        context,
        dispersion,
        ratio,
    )
}

/// Port of `SFCONV/senergies.f90` `rseint3`.
pub fn sfconv_real_self_energy_integrand_lower(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let regularization = (context.accuracy * context.pole_energy).powi(2);
    let numerator =
        ((context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy - dispersion)
            .powi(2)
            + regularization;
    let denominator =
        ((context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy - dispersion)
            .powi(2)
            + regularization;
    real_self_energy_log_integrand(
        "real self-energy lower integrand",
        momentum,
        context,
        dispersion,
        numerator,
        denominator,
    )
}

/// Port of `SFCONV/senergies.f90` `drseint1`.
pub fn sfconv_real_self_energy_derivative_integrand_upper(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_derivative_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let upper =
        (context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let lower =
        (context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let term = derivative_lorentz_term(upper, context.pole_broadening)?
        - derivative_lorentz_term(lower, context.pole_broadening)?;
    real_self_energy_derivative_integrand(
        "real self-energy derivative upper integrand",
        momentum,
        context,
        dispersion,
        term,
    )
}

/// Port of `SFCONV/senergies.f90` `drseint2`.
pub fn sfconv_real_self_energy_derivative_integrand_middle(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_derivative_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let mut term = 0.0;
    if context.include_below_fermi {
        let below_fermi = context.fermi_energy - shifted_energy - dispersion;
        let below_photoelectron =
            (context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy - dispersion;
        term += derivative_lorentz_term(below_fermi, context.pole_broadening)?;
        term -= derivative_lorentz_term(below_photoelectron, context.pole_broadening)?;
    }
    let upper_photoelectron =
        (context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let upper_fermi = context.fermi_energy - shifted_energy + dispersion;
    term += derivative_lorentz_term(upper_photoelectron, context.pole_broadening)?;
    term -= derivative_lorentz_term(upper_fermi, context.pole_broadening)?;
    real_self_energy_derivative_integrand(
        "real self-energy derivative middle integrand",
        momentum,
        context,
        dispersion,
        term,
    )
}

/// Port of `SFCONV/senergies.f90` `drseint3`.
pub fn sfconv_real_self_energy_derivative_integrand_lower(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_derivative_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let upper =
        (context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    let lower =
        (context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    let lower_denominator = checked_sqrt(
        "real self-energy derivative lower denominator",
        lower.powi(2) + context.pole_broadening.powi(2),
    )?;
    validate_nonzero_denominator(
        "real self-energy derivative lower denominator",
        lower_denominator,
    )?;
    let term = derivative_lorentz_term(upper, context.pole_broadening)? - lower / lower_denominator;
    real_self_energy_derivative_integrand(
        "real self-energy derivative lower integrand",
        momentum,
        context,
        dispersion,
        term,
    )
}

/// Port of `SFCONV/senergies.f90` `findsing`.
pub fn sfconv_find_singularities(
    lower: Real,
    upper: Real,
    candidates: ArrayView1<'_, Real>,
) -> Result<RealVec, SfconvError> {
    validate_finite_scalar("singularity lower bound", lower)?;
    validate_finite_scalar("singularity upper bound", upper)?;
    let mut singularities = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, &candidate)| {
            if !candidate.is_finite() {
                return Some(Err(SfconvError::NonFiniteValue {
                    field: "singularity candidate",
                    row: index,
                    value: candidate,
                }));
            }
            let in_forward_interval = candidate > lower && candidate < upper;
            let in_reverse_interval = candidate < lower && candidate > upper;
            (in_forward_interval || in_reverse_interval).then_some(Ok(candidate))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if singularities.len() > SFCONV_GRATER_MAX_SINGULARITIES {
        return Err(SfconvError::TooManySingularities {
            count: singularities.len(),
            max: SFCONV_GRATER_MAX_SINGULARITIES,
        });
    }
    singularities.sort_by(|left, right| left.total_cmp(right));
    Ok(Array1::from_vec(singularities))
}

/// Port of `SFCONV/grater.f90`: adaptive real quadrature with split points.
///
/// `singularities` are FEFF `xsing`: ordered real split points inserted
/// between `lower` and `upper` before the adaptive stack starts. The returned
/// diagnostics mirror FEFF `error`, `numcal`, and `maxns`.
pub fn sfconv_grater_integrate(
    mut integrand: impl FnMut(Real) -> Result<Real, SfconvError>,
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_grater_input(
        lower,
        upper,
        absolute_tolerance,
        relative_tolerance,
        singularities,
    )?;

    let mut xleft = vec![0.0; SFCONV_GRATER_MAX_REGIONS];
    let mut fval = vec![[0.0; 3]; SFCONV_GRATER_MAX_REGIONS];
    let mut nstack = singularities.len() + 1;
    let mut max_regions = nstack;
    let mut estimated_error = 0.0;
    let mut value_total = 0.0;

    xleft[0] = lower;
    xleft[singularities.len() + 1] = upper;
    for (index, &singularity) in singularities.iter().enumerate() {
        xleft[index + 1] = singularity;
    }

    for region in 0..nstack {
        let delta = xleft[region + 1] - xleft[region];
        for point in 0..3 {
            fval[region][point] = eval_grater_integrand(
                &mut integrand,
                xleft[region] + delta * SFCONV_GRATER_DX[point],
                region * 3 + point,
            )?;
        }
    }
    let mut evaluations = nstack * 3;
    let total_interval = upper - lower;

    loop {
        if nstack + 3 >= SFCONV_GRATER_MAX_REGIONS {
            return Err(SfconvError::TooManyIntegrationRegions {
                max_regions: SFCONV_GRATER_MAX_REGIONS,
            });
        }

        let region = nstack - 1;
        let delta = xleft[region + 1] - xleft[region];
        xleft[region + 3] = xleft[region + 1];
        xleft[region + 1] = xleft[region] + delta * SFCONV_GRATER_DX[0] * 2.0;
        xleft[region + 2] = xleft[region + 3] - delta * SFCONV_GRATER_DX[0] * 2.0;
        fval[region + 2][1] = fval[region][2];
        fval[region + 1][1] = fval[region][1];
        fval[region][1] = fval[region][0];

        let mut weight_index = 0;
        let mut high_order = 0.0;
        let mut low_order = 0.0;
        for current_region in region..=region + 2 {
            let sub_delta = xleft[current_region + 1] - xleft[current_region];
            fval[current_region][0] = eval_grater_integrand(
                &mut integrand,
                xleft[current_region] + SFCONV_GRATER_DX[0] * sub_delta,
                evaluations,
            )?;
            evaluations += 1;
            fval[current_region][2] = eval_grater_integrand(
                &mut integrand,
                xleft[current_region] + SFCONV_GRATER_DX[2] * sub_delta,
                evaluations,
            )?;
            evaluations += 1;
            for point in 0..3 {
                high_order += SFCONV_GRATER_WT9[weight_index] * fval[current_region][point] * delta;
                low_order += fval[current_region][point] * SFCONV_GRATER_WT[point] * sub_delta;
                weight_index += 1;
            }
        }

        let difference = (high_order - low_order).abs();
        let fraction = delta / total_interval;
        let at_singularity = fraction <= 1.0e-8;
        if difference <= absolute_tolerance * fraction
            || difference <= relative_tolerance * high_order.abs()
            || (at_singularity && (fraction <= 1.0e-15 || difference <= absolute_tolerance * 0.1))
        {
            value_total += high_order;
            estimated_error += difference.abs();
            nstack -= 1;
            if nstack == 0 {
                return Ok(SfconvAdaptiveIntegral {
                    value: value_total,
                    estimated_error,
                    evaluations,
                    max_regions,
                });
            }
        } else {
            nstack += 2;
            max_regions = max_regions.max(nstack);
        }
    }
}

/// Port of `SFCONV/mksat.f90` `xmkesat`.
///
/// This is the extrinsic satellite with the quasiparticle pole subtracted and
/// quasiparticle broadening removed.
pub fn sfconv_extrinsic_satellite_debroadened(
    energy: Real,
    context: SfconvSatelliteContext,
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_satellite_context(context)?;
    validate_satellite_self_energy(self_energy)?;
    validate_nonzero_denominator("satellite energy", energy)?;

    let renormalization_magnitude = checked_hypot(
        "satellite renormalization",
        self_energy.renormalization_real,
        self_energy.renormalization_imag,
    )?;
    validate_nonzero_denominator("satellite renormalization", renormalization_magnitude)?;

    let width_difference = self_energy.width - self_energy.off_shell_imag;
    let energy_difference = energy + self_energy.on_shell_real - self_energy.off_shell_real;
    let denominator = energy_difference.powi(2) + width_difference.powi(2);
    validate_nonzero_denominator("extrinsic satellite", denominator)?;

    let total = -width_difference / denominator;
    let main = -self_energy.renormalization_imag
        / (energy * std::f64::consts::PI * renormalization_magnitude)
        * (-(energy / (2.0 * context.plasma_frequency)).powi(2)).exp();
    finite_result(
        "extrinsic satellite",
        total / (std::f64::consts::PI * renormalization_magnitude) - main,
    )
}

/// Port of `SFCONV/mksat.f90` `xmkgwext`.
///
/// This is the full-broadening extrinsic satellite including quasiparticle
/// contributions.
pub fn sfconv_extrinsic_satellite_broadened(
    energy: Real,
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_satellite_self_energy(self_energy)?;
    let energy_difference = energy + self_energy.on_shell_real - self_energy.off_shell_real;
    let denominator =
        std::f64::consts::PI * (energy_difference.powi(2) + self_energy.off_shell_imag.powi(2));
    validate_nonzero_denominator("broadened extrinsic satellite", denominator)?;
    finite_result(
        "broadened extrinsic satellite",
        self_energy.off_shell_imag / denominator,
    )
}

/// Port of `SFCONV/mksat.f90` `xintxsat`.
pub fn sfconv_interference_satellite_integrand(
    momentum: Real,
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let tolerance = 0.2 * context.plasma_frequency;
    let energy_delta = context.photoelectron_energy - energy;
    let lorentzian =
        width / (std::f64::consts::PI * ((energy - dispersion).powi(2) + width.powi(2)));

    let factor = if energy_delta >= 0.0 {
        let wave_number = checked_sqrt("interference wave number", 2.0 * energy_delta)?;
        validate_nonzero_denominator("interference wave number", wave_number)?;
        let numerator = (dispersion - momentum.powi(2) / 2.0 + wave_number * momentum).powi(2)
            + tolerance.powi(2);
        let denominator = (dispersion - momentum.powi(2) / 2.0 - wave_number * momentum).powi(2)
            + tolerance.powi(2);
        validate_nonzero_denominator("interference logarithm", denominator)?;
        (numerator / denominator).ln() / 2.0 / wave_number
    } else {
        let wave_number = checked_sqrt("interference evanescent wave number", -2.0 * energy_delta)?;
        validate_nonzero_denominator("interference evanescent wave number", wave_number)?;
        let denominator = dispersion - momentum.powi(2) / 2.0;
        validate_nonzero_denominator("interference arctangent", denominator)?;
        (wave_number * momentum / denominator).atan() / wave_number
    };

    finite_result(
        "interference satellite integrand",
        momentum * coupling * lorentzian * factor / dispersion,
    )
}

/// Port of `SFCONV/mksat.f90` `xintisat`.
pub fn sfconv_intrinsic_satellite_integrand(
    momentum: Real,
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let lorentzian =
        width / (((energy - dispersion).powi(2) + width.powi(2)) * std::f64::consts::PI);
    finite_result(
        "intrinsic satellite integrand",
        momentum.powi(2) * coupling * lorentzian / dispersion.powi(2),
    )
}

/// Port of `SFCONV/mksat.f90` `xmkxsat`.
pub fn sfconv_interference_satellite(
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;
    let q2 = checked_sqrt(
        "interference satellite q2",
        (2.0 * (energy - context.pole_energy)).max(width),
    )?;
    validate_nonzero_denominator("interference satellite q2", q2)?;
    let qwidth = 10.0 * width / q2;
    let qmin = 0.0_f64.max(q2 - qwidth);
    let qmax = q2 + qwidth;
    let first = integrate_mksat_range(qmin, q2, context, |momentum, context| {
        sfconv_interference_satellite_integrand(momentum, energy, width, context)
    })?;
    let second = integrate_mksat_range(q2, qmax, context, |momentum, context| {
        sfconv_interference_satellite_integrand(momentum, energy, width, context)
    })?;
    combine_satellite_integrals(first, second, (2.0 * std::f64::consts::PI).powi(2))
}

/// Port of `SFCONV/mksat.f90` `xmkisat`.
pub fn sfconv_intrinsic_satellite(
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;
    let q2 = if energy - context.pole_energy > width {
        checked_sqrt(
            "intrinsic satellite q2",
            2.0 * (energy - context.pole_energy),
        )?
    } else {
        checked_sqrt("intrinsic satellite q2", 2.0 * width)?
    };
    validate_nonzero_denominator("intrinsic satellite q2", q2)?;
    let qwidth = 10.0 * q2.min(width / q2);
    let qmax = q2 + qwidth;
    let first = integrate_mksat_range(0.0, q2, context, |momentum, context| {
        sfconv_intrinsic_satellite_integrand(momentum, energy, width, context)
    })?;
    let second = integrate_mksat_range(q2, qmax, context, |momentum, context| {
        sfconv_intrinsic_satellite_integrand(momentum, energy, width, context)
    })?;
    combine_satellite_integrals(first, second, 2.0 * std::f64::consts::PI.powi(2))
}

/// Port of `SFCONV/mksat.f90` `xintak`.
pub fn sfconv_interference_quasiparticle_integrand(
    momentum: Real,
    photoelectron_momentum: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_positive_scalar("photoelectron_momentum", photoelectron_momentum)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let epsilon = 0.1_f64;
    let numerator = (dispersion + momentum.powi(2) / 2.0 + photoelectron_momentum * momentum)
        .powi(2)
        + (context.pole_energy * epsilon).powi(2);
    let denominator = (dispersion + momentum.powi(2) / 2.0 - photoelectron_momentum * momentum)
        .powi(2)
        + (context.pole_energy * epsilon).powi(2);
    validate_nonzero_denominator("quasiparticle logarithm", denominator)?;
    let log_factor = (numerator / denominator).ln() / 2.0;
    finite_result(
        "interference quasiparticle integrand",
        momentum * coupling * log_factor
            / (dispersion * photoelectron_momentum * 4.0 * std::f64::consts::PI.powi(2)),
    )
}

/// Port of `SFCONV/mksat.f90` `xmkak`.
pub fn sfconv_interference_quasiparticle(
    energy: Real,
    upper_energy: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_finite_scalar("satellite upper energy", upper_energy)?;
    validate_satellite_context(context)?;
    if energy <= 0.0 {
        return Ok(SfconvSatelliteIntegral {
            value: 0.0,
            estimated_error: 0.0,
            evaluations: 0,
            max_regions: 0,
        });
    }
    let absolute_tolerance =
        checked_sqrt("quasiparticle tolerance", context.plasma_frequency)? * context.accuracy;
    let upper_momentum = checked_sqrt("quasiparticle upper momentum", 2.0 * upper_energy)?;
    let photoelectron_momentum = checked_sqrt(
        "quasiparticle photoelectron momentum",
        2.0 * context.photoelectron_energy,
    )?;
    validate_nonzero_denominator(
        "quasiparticle photoelectron momentum",
        photoelectron_momentum,
    )?;
    let integral = sfconv_grater_integrate(
        |momentum| {
            sfconv_interference_quasiparticle_integrand(momentum, photoelectron_momentum, context)
        },
        absolute_tolerance,
        upper_momentum,
        absolute_tolerance,
        context.accuracy,
        &[],
    )?;
    Ok(SfconvSatelliteIntegral {
        value: integral.value,
        estimated_error: integral.estimated_error,
        evaluations: integral.evaluations,
        max_regions: integral.max_regions,
    })
}

/// Port of the SO2CONV spectral-function interpolation over momentum.
///
/// FEFF caches spectral functions on the 66-row `pgrid` and, for each signal
/// row, interpolates those cached tables to the current photoelectron momentum
/// `pk(jj)`. Values at or above the final momentum copy the final cached row.
/// Values below the first momentum copy the first spectral rows and weights,
/// but preserve FEFF's historical endpoint quirk of taking `epts` from the
/// final cached momentum row.
pub fn sfconv_interpolate_momentum_spectral_function(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
) -> Result<SfconvMomentumSpectralInterpolation, SfconvError> {
    validate_momentum_spectral_interpolation_input(input)?;

    let columns = input.energy_grid.ncols();
    let mut output = SfconvMomentumSpectralInterpolation {
        energy: Array1::<Real>::zeros(columns),
        spectral_function: Array2::<Real>::zeros((8, columns)),
        weights: Array1::<Real>::zeros(8),
        self_energy_real: 0.0,
        energy_correction: 0.0,
        width: 0.0,
        renormalization_real: 0.0,
        renormalization_imag: 0.0,
    };

    let last = input.momentum_grid.len() - 1;
    if input.photoelectron_momentum >= input.momentum_grid[last] {
        set_momentum_spectral_exact_row(&mut output, input, last, last);
    } else if input.photoelectron_momentum < input.momentum_grid[0] {
        set_momentum_spectral_exact_row(&mut output, input, last, 0);
    } else {
        let segment = find_momentum_spectral_segment(input)?;
        set_momentum_spectral_interpolated_row(&mut output, input, segment)?;
    }

    validate_finite_array("momentum spectral energy", output.energy.view())?;
    validate_finite_array("momentum spectral weights", output.weights.view())?;
    validate_finite_matrix(
        "momentum spectral function",
        output.spectral_function.view(),
    )?;
    finite_result(
        "momentum spectral self_energy_real",
        output.self_energy_real,
    )?;
    finite_result(
        "momentum spectral energy_correction",
        output.energy_correction,
    )?;
    finite_result("momentum spectral width", output.width)?;
    finite_result(
        "momentum spectral renormalization_real",
        output.renormalization_real,
    )?;
    finite_result(
        "momentum spectral renormalization_imag",
        output.renormalization_imag,
    )?;
    Ok(output)
}

/// Port of the `SO2CONV` photoelectron momentum refinement.
///
/// FEFF first maps the input `xk` grid to `ekpg`, builds a zeroth-order
/// momentum estimate `xpkg`, estimates `zkk` from a finite difference of the
/// supplied self-energy samples, then applies the self-energy correction to
/// produce the momentum `pk` used for spectral-function interpolation.
pub fn sfconv_so2conv_photoelectron_momentum(
    input: SfconvPhotoelectronMomentumInput<'_>,
) -> Result<SfconvPhotoelectronMomentum, SfconvError> {
    validate_photoelectron_momentum_input(input)?;

    let len = input.momentum.len();
    let mut kinetic_energy = Array1::<Real>::zeros(len);
    let mut zero_order_momentum = Array1::<Real>::zeros(len);
    let mut renormalization = Array1::<Real>::zeros(len);
    let mut photoelectron_momentum = Array1::<Real>::zeros(len);

    for row in 0..len {
        let momentum = input.momentum[row];
        let energy = if momentum >= 0.0 {
            momentum.powi(2) / 2.0 + input.chemical_potential
        } else {
            -momentum.powi(2) / 2.0 + input.chemical_potential
        };
        kinetic_energy[row] = finite_result("photoelectron kinetic energy", energy)?;
        if energy >= 0.0 {
            zero_order_momentum[row] = checked_sqrt(
                "photoelectron zero-order momentum",
                input.fermi_momentum.powi(2) + 2.0 * (energy - input.fermi_level),
            )?;
        }
    }

    for row in 0..len {
        if kinetic_energy[row] < 0.0 {
            continue;
        }

        let (lower_row, upper_row) = if row == 0 {
            (0, 1)
        } else if row + 1 == len {
            (row - 1, row)
        } else {
            (row - 1, row + 1)
        };
        let self_energy_delta = input.self_energy[upper_row] - input.self_energy[lower_row];
        let kinetic_delta = zero_order_momentum[upper_row].powi(2) / 2.0
            - zero_order_momentum[lower_row].powi(2) / 2.0;
        validate_nonzero_denominator("photoelectron momentum finite difference", kinetic_delta)?;

        let denominator = 1.0 + self_energy_delta / kinetic_delta;
        validate_nonzero_denominator("photoelectron momentum renormalization", denominator)?;
        renormalization[row] =
            finite_result("photoelectron momentum renormalization", 1.0 / denominator)?;

        photoelectron_momentum[row] = checked_sqrt(
            "photoelectron momentum",
            zero_order_momentum[row].powi(2)
                - 2.0 * renormalization[row] * (input.self_energy[row] - input.fermi_self_energy),
        )?;
    }

    validate_finite_array("photoelectron kinetic energy", kinetic_energy.view())?;
    validate_finite_array(
        "photoelectron zero-order momentum",
        zero_order_momentum.view(),
    )?;
    validate_finite_array("photoelectron renormalization", renormalization.view())?;
    validate_finite_array("photoelectron momentum", photoelectron_momentum.view())?;
    Ok(SfconvPhotoelectronMomentum {
        kinetic_energy,
        zero_order_momentum,
        renormalization,
        photoelectron_momentum,
    })
}

/// Compute one SO2CONV unbroadened weighted-pole self-energy sample.
///
/// This is the FEFF `brpole = .false.` branch: each active pole contributes
/// `plwt * renergies(energy)`, and the free-electron exchange term is added at
/// the requested photoelectron momentum.
pub fn sfconv_so2conv_unbroadened_self_energy_sample(
    input: SfconvSo2convSelfEnergySampleInput<'_>,
) -> Result<Real, SfconvError> {
    validate_so2conv_self_energy_sample_input(input)?;

    let pole_sum = (1..=input.pole_count).try_fold(0.0, |accumulator, pole_index| {
        let pole = sfconv_select_pole(
            pole_index,
            input.pole_energy,
            input.pole_weight,
            input.pole_broadening,
        )?;
        let context = SfconvSelfEnergyContext {
            fermi_energy: input.material.fermi_energy,
            fermi_momentum: input.material.fermi_momentum,
            plasma_frequency: input.material.plasma_frequency,
            pole_energy: pole.energy,
            quasiparticle_energy: input.quasiparticle_energy,
            photoelectron_momentum: input.photoelectron_momentum,
            accuracy: input.material.accuracy,
            pole_broadening: pole.broadening,
            dispersion_parameter: input.material.dispersion_parameter,
            include_below_fermi: input.include_below_fermi,
        };
        let self_energy = sfconv_real_self_energy(input.energy, context)?.value;
        finite_result(
            "so2conv weighted self energy",
            accumulator + pole.weight * self_energy,
        )
    })?;
    let exchange =
        sfconv_free_electron_exchange(input.photoelectron_momentum, input.material.fermi_momentum)?;
    finite_result("so2conv unbroadened self energy", pole_sum + exchange)
}

/// Build SO2CONV unbroadened self-energy samples for momentum refinement.
///
/// FEFF first maps each input `xk` row to `ekpg`, estimates the zeroth-order
/// photoelectron momentum `xpkg`, evaluates the real self energy `seg` at that
/// momentum, and then calls the momentum-refinement formula. This helper
/// performs the `ekpg`/`xpkg`/`seg` part for the unbroadened `renergies`
/// branch and returns `sef0` for the existing
/// [`sfconv_so2conv_photoelectron_momentum`] helper.
pub fn sfconv_so2conv_unbroadened_self_energy_grid(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
) -> Result<SfconvSo2convSelfEnergyGrid, SfconvError> {
    build_so2conv_self_energy_grid(input, sfconv_so2conv_unbroadened_self_energy_sample)
}

/// Compute one SO2CONV broadened weighted-pole self-energy sample.
///
/// This is the FEFF default `brpole = .true.` branch: each active pole
/// contributes `plwt * brsigma(energy).real`, and the free-electron exchange
/// term is added at the requested photoelectron momentum.
pub fn sfconv_so2conv_broadened_self_energy_sample(
    input: SfconvSo2convSelfEnergySampleInput<'_>,
) -> Result<Real, SfconvError> {
    validate_so2conv_self_energy_sample_input(input)?;

    let pole_sum = (1..=input.pole_count).try_fold(0.0, |accumulator, pole_index| {
        let pole = sfconv_select_pole(
            pole_index,
            input.pole_energy,
            input.pole_weight,
            input.pole_broadening,
        )?;
        let context = SfconvSelfEnergyContext {
            fermi_energy: input.material.fermi_energy,
            fermi_momentum: input.material.fermi_momentum,
            plasma_frequency: input.material.plasma_frequency,
            pole_energy: pole.energy,
            quasiparticle_energy: input.quasiparticle_energy,
            photoelectron_momentum: input.photoelectron_momentum,
            accuracy: input.material.accuracy,
            pole_broadening: pole.broadening,
            dispersion_parameter: input.material.dispersion_parameter,
            include_below_fermi: input.include_below_fermi,
        };
        let self_energy = sfconv_broadened_self_energy(input.energy, context)?.real;
        finite_result(
            "so2conv broadened weighted self energy",
            accumulator + pole.weight * self_energy,
        )
    })?;
    let exchange =
        sfconv_free_electron_exchange(input.photoelectron_momentum, input.material.fermi_momentum)?;
    finite_result("so2conv broadened self energy", pole_sum + exchange)
}

/// Build SO2CONV broadened self-energy samples for momentum refinement.
///
/// This mirrors the FEFF default `brpole = .true.` setup in `so2conv.f90`,
/// using [`sfconv_broadened_self_energy`] for each active pole before the
/// existing photoelectron-momentum refinement step.
pub fn sfconv_so2conv_broadened_self_energy_grid(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
) -> Result<SfconvSo2convSelfEnergyGrid, SfconvError> {
    build_so2conv_self_energy_grid(input, sfconv_so2conv_broadened_self_energy_sample)
}

fn build_so2conv_self_energy_grid(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
    sample: impl Fn(SfconvSo2convSelfEnergySampleInput<'_>) -> Result<Real, SfconvError>,
) -> Result<SfconvSo2convSelfEnergyGrid, SfconvError> {
    validate_so2conv_self_energy_grid_input(input)?;

    let fermi_self_energy = sample(SfconvSo2convSelfEnergySampleInput {
        material: input.material,
        energy: 0.0,
        quasiparticle_energy: input.material.fermi_energy,
        photoelectron_momentum: input.material.fermi_momentum,
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
        include_below_fermi: input.include_below_fermi,
    })?;

    let len = input.momentum.len();
    let mut kinetic_energy = Array1::<Real>::zeros(len);
    let mut zero_order_momentum = Array1::<Real>::zeros(len);
    let mut self_energy = Array1::<Real>::zeros(len);

    for row in 0..len {
        let momentum = input.momentum[row];
        let energy = if momentum >= 0.0 {
            momentum.powi(2) / 2.0 + input.chemical_potential
        } else {
            -momentum.powi(2) / 2.0 + input.chemical_potential
        };
        kinetic_energy[row] = finite_result("so2conv self-energy kinetic energy", energy)?;
        if energy >= 0.0 {
            let row_momentum = checked_sqrt(
                "so2conv self-energy zero-order momentum",
                input.material.fermi_momentum.powi(2) + 2.0 * (energy - input.fermi_level),
            )?;
            zero_order_momentum[row] = row_momentum;
            self_energy[row] = sample(SfconvSo2convSelfEnergySampleInput {
                material: input.material,
                energy: 0.0,
                quasiparticle_energy: energy,
                photoelectron_momentum: row_momentum,
                pole_count: input.pole_count,
                pole_energy: input.pole_energy,
                pole_weight: input.pole_weight,
                pole_broadening: input.pole_broadening,
                include_below_fermi: input.include_below_fermi,
            })?;
        }
    }

    validate_finite_array("so2conv self-energy kinetic energy", kinetic_energy.view())?;
    validate_finite_array(
        "so2conv self-energy zero-order momentum",
        zero_order_momentum.view(),
    )?;
    validate_finite_array("so2conv self-energy", self_energy.view())?;
    Ok(SfconvSo2convSelfEnergyGrid {
        kinetic_energy,
        zero_order_momentum,
        self_energy,
        fermi_self_energy,
    })
}

/// Evaluate the FEFF `brsigma` broadened self-energy integrand family.
///
/// The returned values correspond to the Fortran functions `fqlogrN`,
/// `fqlogiN`, `fqatnrN`, and `fqatniN` for the selected branch `N`. This helper
/// does not perform the `grater` interval integration or final `brsigma`
/// scaling; it keeps the branch formulas directly testable before the full
/// broadened self-energy driver is assembled.
pub fn sfconv_broadened_self_energy_integrands(
    branch: SfconvBroadenedSelfEnergyBranch,
    input: SfconvBroadenedSelfEnergyIntegrandInput,
) -> Result<SfconvBroadenedSelfEnergyIntegrands, SfconvError> {
    validate_broadened_self_energy_integrand_input(input)?;

    let context = input.context;
    let shifted_energy = finite_result(
        "broadened self-energy shifted energy",
        input.energy + context.quasiparticle_energy,
    )?;
    let dispersion = sfconv_pole_dispersion(
        input.momentum,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let pole_denominator = dispersion.powi(2) + context.pole_broadening.powi(2);
    validate_nonzero_denominator("broadened self-energy pole denominator", pole_denominator)?;
    let log_ratio = broadened_self_energy_log_ratio(
        branch,
        input.momentum,
        shifted_energy,
        context,
        dispersion,
    )?;
    let atan_delta = broadened_self_energy_atan_delta(
        branch,
        input.momentum,
        shifted_energy,
        context,
        dispersion,
    );
    let log_norm = checked_sqrt(
        "broadened self-energy log normalization",
        input.momentum.powi(2) + context.pole_energy * context.accuracy,
    )?;
    let atan_norm = checked_sqrt(
        "broadened self-energy atan normalization",
        input.momentum.powi(2) + context.plasma_frequency * context.accuracy,
    )?;
    validate_nonzero_denominator("broadened self-energy log normalization", log_norm)?;
    validate_nonzero_denominator("broadened self-energy atan normalization", atan_norm)?;

    let log_value = log_ratio.ln();
    let pole_real = dispersion / pole_denominator;
    let pole_imag = context.pole_broadening / pole_denominator;
    Ok(SfconvBroadenedSelfEnergyIntegrands {
        log_real: finite_result(
            "broadened log real integrand",
            pole_real * log_value / log_norm,
        )?,
        log_imag: finite_result(
            "broadened log imag integrand",
            pole_imag * log_value / log_norm,
        )?,
        atan_real: finite_result(
            "broadened atan real integrand",
            pole_imag * atan_delta / atan_norm,
        )?,
        atan_imag: finite_result(
            "broadened atan imag integrand",
            pole_real * atan_delta / atan_norm,
        )?,
    })
}

/// Evaluate the FEFF `dbrsigma` broadened self-energy derivative integrands.
///
/// The returned values correspond to the Fortran functions `dqlogrN`,
/// `dqlogiN`, `dqatnrN`, and `dqatniN` for the selected branch `N`.
pub fn sfconv_broadened_self_energy_derivative_integrands(
    branch: SfconvBroadenedSelfEnergyBranch,
    input: SfconvBroadenedSelfEnergyIntegrandInput,
) -> Result<SfconvBroadenedSelfEnergyDerivativeIntegrands, SfconvError> {
    validate_broadened_self_energy_integrand_input(input)?;

    let context = input.context;
    let shifted_energy = finite_result(
        "broadened self-energy derivative shifted energy",
        input.energy + context.quasiparticle_energy,
    )?;
    let dispersion = sfconv_pole_dispersion(
        input.momentum,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let pole_denominator = dispersion.powi(2) + context.pole_broadening.powi(2);
    validate_nonzero_denominator(
        "broadened self-energy derivative pole denominator",
        pole_denominator,
    )?;
    let (left, right) = broadened_self_energy_response_arguments(
        branch,
        input.momentum,
        shifted_energy,
        context,
        dispersion,
    );
    let left_denominator = finite_result(
        "broadened self-energy derivative left denominator",
        left.powi(2) + context.pole_broadening.powi(2),
    )?;
    let right_denominator = finite_result(
        "broadened self-energy derivative right denominator",
        right.powi(2) + context.pole_broadening.powi(2),
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative left denominator",
        left_denominator,
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative right denominator",
        right_denominator,
    )?;
    let log_derivative = finite_result(
        "broadened self-energy log derivative",
        left / left_denominator - right / right_denominator,
    )?;
    let atan_derivative = finite_result(
        "broadened self-energy atan derivative",
        context.pole_broadening / left_denominator - context.pole_broadening / right_denominator,
    )?;
    let log_norm = checked_sqrt(
        "broadened self-energy derivative log normalization",
        input.momentum.powi(2) + context.pole_energy * context.accuracy,
    )?;
    let atan_norm = checked_sqrt(
        "broadened self-energy derivative atan normalization",
        input.momentum.powi(2) + context.plasma_frequency * context.accuracy,
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative log normalization",
        log_norm,
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative atan normalization",
        atan_norm,
    )?;

    let pole_real = dispersion / pole_denominator;
    let pole_imag = context.pole_broadening / pole_denominator;
    Ok(SfconvBroadenedSelfEnergyDerivativeIntegrands {
        log_real: finite_result(
            "broadened log real derivative integrand",
            pole_real * log_derivative / log_norm,
        )?,
        log_imag: finite_result(
            "broadened log imag derivative integrand",
            pole_imag * log_derivative / log_norm,
        )?,
        atan_real: finite_result(
            "broadened atan real derivative integrand",
            pole_imag * atan_derivative / atan_norm,
        )?,
        atan_imag: finite_result(
            "broadened atan imag derivative integrand",
            pole_real * atan_derivative / atan_norm,
        )?,
    })
}

/// Port of `SFCONV/senergies.f90` `brsigma`.
///
/// This integrates the broadened log and arctangent branch kernels over FEFF's
/// piecewise momentum intervals, applies the `omp**2/(pi*pk)` scaling, and
/// rotates the complex self energy by FEFF's Lorentzian pole factor
/// `1 - i * brd / ompl`.
pub fn sfconv_broadened_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvBroadenedSelfEnergy, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)?;

    let shifted_energy = finite_result(
        "broadened self-energy shifted energy",
        energy + context.quasiparticle_energy,
    )?;
    let qmax = 100.0 * checked_sqrt("broadened self-energy qmax", context.pole_energy)?
        + context.photoelectron_momentum
        + context.fermi_momentum;
    let high_limit = context.photoelectron_momentum + context.fermi_momentum;
    let low_limit = (context.photoelectron_momentum - context.fermi_momentum).abs();
    let high_singularity = sfconv_inverse_pole_dispersion(
        (shifted_energy - context.fermi_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let low_singularity = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - shifted_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        shifted_energy,
        context.photoelectron_momentum,
        context.pole_energy,
        context.dispersion_parameter,
        qmax,
    )?;
    let singularity_candidates = Array1::from_vec(vec![
        low_singularity,
        limits.q1,
        limits.q2,
        limits.q3,
        high_singularity,
    ]);
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut sums = BroadenedSelfEnergyAccumulator::default();
    let range_input = |branch, lower, upper| BroadenedSelfEnergyRangeInput {
        branch,
        lower,
        upper,
        energy,
        context,
        singularity_candidates: singularity_candidates.view(),
        absolute_tolerance,
        relative_tolerance,
    };

    integrate_broadened_self_energy_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            high_limit,
            qmax,
        ),
    )?;
    integrate_broadened_self_energy_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            low_limit,
            high_limit,
        ),
    )?;
    if context.include_below_fermi {
        integrate_broadened_self_energy_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::HoleFermi,
                low_limit,
                high_limit,
            ),
        )?;
    }

    if context.photoelectron_momentum > context.fermi_momentum {
        integrate_broadened_self_energy_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                0.0,
                low_limit,
            ),
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum && context.include_below_fermi
    {
        integrate_broadened_self_energy_range(
            &mut sums,
            range_input(SfconvBroadenedSelfEnergyBranch::HolePair, 0.0, low_limit),
        )?;
    }

    let log_scale = context.plasma_frequency.powi(2)
        / (4.0 * std::f64::consts::PI * context.photoelectron_momentum);
    let atan_scale = context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    let unrotated_real = finite_result(
        "broadened self-energy real",
        sums.log_real * log_scale + sums.atan_real * atan_scale,
    )?;
    let unrotated_imaginary = finite_result(
        "broadened self-energy imaginary",
        sums.log_imag * log_scale - sums.atan_imag * atan_scale,
    )?;
    let unrotated_real_error =
        sums.log_real_error * log_scale.abs() + sums.atan_real_error * atan_scale.abs();
    let unrotated_imaginary_error =
        sums.log_imag_error * log_scale.abs() + sums.atan_imag_error * atan_scale.abs();
    let pole_rotation = context.pole_broadening / context.pole_energy;

    Ok(SfconvBroadenedSelfEnergy {
        real: finite_result(
            "broadened self-energy rotated real",
            unrotated_real + unrotated_imaginary * pole_rotation,
        )?,
        imaginary: finite_result(
            "broadened self-energy rotated imaginary",
            unrotated_imaginary - unrotated_real * pole_rotation,
        )?,
        real_estimated_error: unrotated_real_error
            + unrotated_imaginary_error * pole_rotation.abs(),
        imaginary_estimated_error: unrotated_imaginary_error
            + unrotated_real_error * pole_rotation.abs(),
        evaluations: sums.evaluations,
        max_regions: sums.max_regions,
    })
}

/// Port of `SFCONV/senergies.f90` `dbrsigma`.
///
/// This evaluates the energy derivative of [`sfconv_broadened_self_energy`]
/// using FEFF's derivative log and arctangent kernels over the same piecewise
/// momentum intervals.
pub fn sfconv_broadened_self_energy_derivative(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvBroadenedSelfEnergyDerivative, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)?;

    let shifted_energy = finite_result(
        "broadened self-energy derivative shifted energy",
        energy + context.quasiparticle_energy,
    )?;
    let qmax = 100.0 * checked_sqrt("broadened self-energy derivative qmax", context.pole_energy)?
        + context.photoelectron_momentum
        + context.fermi_momentum;
    let high_limit = context.photoelectron_momentum + context.fermi_momentum;
    let low_limit = (context.photoelectron_momentum - context.fermi_momentum).abs();
    let high_singularity = sfconv_inverse_pole_dispersion(
        (shifted_energy - context.fermi_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let low_singularity = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - shifted_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        shifted_energy,
        context.photoelectron_momentum,
        context.pole_energy,
        context.dispersion_parameter,
        qmax,
    )?;
    let singularity_candidates = Array1::from_vec(vec![
        low_singularity,
        limits.q1,
        limits.q2,
        limits.q3,
        high_singularity,
    ]);
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut sums = BroadenedSelfEnergyAccumulator::default();
    let range_input = |branch, lower, upper| BroadenedSelfEnergyRangeInput {
        branch,
        lower,
        upper,
        energy,
        context,
        singularity_candidates: singularity_candidates.view(),
        absolute_tolerance,
        relative_tolerance,
    };

    integrate_broadened_self_energy_derivative_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            high_limit,
            qmax,
        ),
    )?;
    integrate_broadened_self_energy_derivative_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            low_limit,
            high_limit,
        ),
    )?;
    if context.include_below_fermi {
        integrate_broadened_self_energy_derivative_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::HoleFermi,
                low_limit,
                high_limit,
            ),
        )?;
    }

    if context.photoelectron_momentum > context.fermi_momentum {
        integrate_broadened_self_energy_derivative_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                0.0,
                low_limit,
            ),
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum && context.include_below_fermi
    {
        integrate_broadened_self_energy_derivative_range(
            &mut sums,
            range_input(SfconvBroadenedSelfEnergyBranch::HolePair, 0.0, low_limit),
        )?;
    }

    let scale = context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    let unrotated_real = finite_result(
        "broadened self-energy derivative real",
        (sums.log_real + sums.atan_real) * scale,
    )?;
    let unrotated_imaginary = finite_result(
        "broadened self-energy derivative imaginary",
        (sums.log_imag - sums.atan_imag) * scale,
    )?;
    let unrotated_real_error = (sums.log_real_error + sums.atan_real_error) * scale.abs();
    let unrotated_imaginary_error = (sums.log_imag_error + sums.atan_imag_error) * scale.abs();
    let pole_rotation = context.pole_broadening / context.pole_energy;

    Ok(SfconvBroadenedSelfEnergyDerivative {
        real: finite_result(
            "broadened self-energy derivative rotated real",
            unrotated_real + unrotated_imaginary * pole_rotation,
        )?,
        imaginary: finite_result(
            "broadened self-energy derivative rotated imaginary",
            unrotated_imaginary - unrotated_real * pole_rotation,
        )?,
        real_estimated_error: unrotated_real_error
            + unrotated_imaginary_error * pole_rotation.abs(),
        imaginary_estimated_error: unrotated_imaginary_error
            + unrotated_real_error * pole_rotation.abs(),
        evaluations: sums.evaluations,
        max_regions: sums.max_regions,
    })
}

/// Port of `SFCONV/interpsf.f90`: interpolate spectral function to a uniform grid.
///
/// FEFF builds the scalar spectral function from rows 2, 5, and 4 of
/// `spectf` as `spectf(2,j) + spectf(5,j) - 2*spectf(4,j)`, then linearly
/// interpolates that combination from the minimal input grid to `output_len`
/// uniformly spaced points spanning the same energy range.
pub fn sfconv_interpolate_spectral_function(
    input: SfconvSpectralInterpolationInput<'_>,
) -> Result<SfconvSpectralInterpolation, SfconvError> {
    validate_count_at_least("output_len", input.output_len, 2)?;
    validate_count_at_least("energy", input.energy.len(), 2)?;
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_matching_lengths(
        "energy",
        input.energy.len(),
        "spectral_function columns",
        input.spectral_function.ncols(),
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_spectral_rows(input.spectral_function)?;

    let last_input = input.energy.len() - 1;
    let first_energy = input.energy[0];
    let last_energy = input.energy[last_input];
    let step = (last_energy - first_energy) / (input.output_len as Real - 1.0);
    let mut energy = Array1::<Real>::zeros(input.output_len);
    let mut spectral_function = Array1::<Real>::zeros(input.output_len);

    energy[0] = first_energy;
    spectral_function[0] = combined_spectral_function(input.spectral_function, 0);
    energy[input.output_len - 1] = last_energy;
    spectral_function[input.output_len - 1] =
        combined_spectral_function(input.spectral_function, last_input);

    let mut lower = 0usize;
    for output in 1..(input.output_len - 1) {
        let output_energy = first_energy + step * output as Real;
        energy[output] = output_energy;

        while lower + 1 < last_input && output_energy >= input.energy[lower + 1] {
            lower += 1;
        }

        let upper = lower + 1;
        if !(input.energy[lower]..input.energy[upper]).contains(&output_energy) {
            return Err(SfconvError::NonFiniteResult {
                row: output,
                value: output_energy,
            });
        }
        let low = combined_spectral_function(input.spectral_function, lower);
        let high = combined_spectral_function(input.spectral_function, upper);
        let fraction =
            (output_energy - input.energy[lower]) / (input.energy[upper] - input.energy[lower]);
        spectral_function[output] = low + (high - low) * fraction;
    }

    Ok(SfconvSpectralInterpolation {
        energy,
        spectral_function,
    })
}

/// Port of `SFCONV/sfconvsub.f90`: spectral-function convolution.
///
/// The kernel integrates a signal over the spectral function, optionally
/// applying FEFF's available-energy cutoff and asymmetric quasiparticle phase
/// branch. Diagnostic file emission from the Fortran routine is intentionally
/// kept out of this pure numerical helper.
pub fn sfconv_convolve(
    input: SfconvConvolutionInput<'_>,
) -> Result<SfconvConvolution, SfconvError> {
    validate_convolution_input(input)?;

    let weights = input.weights;
    let pi = std::f64::consts::PI;
    let mut real_convolution = 0.0;
    let mut imag_convolution = 0.0;
    let quasiparticle_magnitude = if input.asymmetric_phase {
        weights[0]
    } else {
        weights[0].hypot(weights[1])
    };
    let quasiparticle_phase = if weights[0] != 0.0 && !input.asymmetric_phase {
        (weights[1] / weights[0]).atan()
    } else {
        0.0
    };
    let quasiparticle_reduction = if !input.cutoff {
        1.0
    } else if input.photoelectron_energy - input.chemical_potential != 0.0 {
        input
            .core_hole_lifetime
            .atan2(input.chemical_potential - input.photoelectron_energy)
            / pi
    } else {
        0.5
    };
    let quasiparticle_weight = quasiparticle_reduction * (quasiparticle_magnitude + weights[2]);
    let mut normalization = quasiparticle_weight;

    let mut cutoff_spectral_function = Array1::<Real>::zeros(input.spectral_function.len());
    for row in 0..input.spectral_function.len() {
        let width = integration_width(input.spectral_energy, input.spectral_function.len(), row);
        let excitation_energy = input.spectral_energy[row];
        let available_energy = input.photoelectron_energy - excitation_energy;
        let cutoff_weight = cutoff_weight(
            input.cutoff,
            available_energy,
            input.chemical_potential,
            input.core_hole_lifetime,
        );

        let mut value = if !input.cutoff {
            input.spectral_function[row]
        } else if excitation_energy >= 0.0 {
            input.spectral_function[row] * cutoff_weight
        } else {
            (input.spectral_function[row] * cutoff_weight).max(0.0)
        };
        if input.asymmetric_phase {
            let half_width = 0.5 * width;
            let smoothing = 3.0 * width;
            let log_ratio = (((excitation_energy + half_width).powi(2) + smoothing.powi(2))
                / ((excitation_energy - half_width).powi(2) + smoothing.powi(2)))
            .ln();
            value -= quasiparticle_reduction
                * (weights[1] / (pi * quasiparticle_magnitude * width))
                * log_ratio
                * (-(excitation_energy / (2.0 * input.plasma_frequency)).powi(2)).exp()
                / 2.0;
        }
        cutoff_spectral_function[row] = value;
        normalization += value * width;
    }
    if !normalization.is_finite() || normalization == 0.0 {
        return Err(SfconvError::InvalidNormalization {
            value: normalization,
        });
    }

    for row in 0..input.spectral_function.len() {
        let width = integration_width(input.spectral_energy, input.spectral_function.len(), row);
        let excitation_energy = input.spectral_energy[row];
        let available_energy = input.photoelectron_energy - excitation_energy;
        let signal = interpolated_signal(input, available_energy)?;
        if row > 0 && row + 1 < input.spectral_function.len() {
            let left_midpoint = 0.5 * (excitation_energy + input.spectral_energy[row - 1]);
            let right_midpoint = 0.5 * (excitation_energy + input.spectral_energy[row + 1]);
            if left_midpoint < 0.0 && right_midpoint >= 0.0 {
                real_convolution += quasiparticle_weight * signal;
            }
        }
        real_convolution += cutoff_spectral_function[row] * width * signal;
    }

    let stored_real = real_convolution;
    real_convolution =
        stored_real * quasiparticle_phase.cos() - imag_convolution * quasiparticle_phase.sin();
    imag_convolution =
        imag_convolution * quasiparticle_phase.cos() + stored_real * quasiparticle_phase.sin();
    real_convolution /= normalization;
    imag_convolution /= normalization;

    let amplitude = real_convolution.hypot(imag_convolution);
    let phase = imag_convolution.atan2(real_convolution);
    if !amplitude.is_finite() {
        return Err(SfconvError::NonFiniteResult {
            row: 0,
            value: amplitude,
        });
    }
    if !phase.is_finite() {
        return Err(SfconvError::NonFiniteResult {
            row: 1,
            value: phase,
        });
    }

    Ok(SfconvConvolution { amplitude, phase })
}

fn validate_convolution_input(input: SfconvConvolutionInput<'_>) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_finite_scalar("core_hole_lifetime", input.core_hole_lifetime)?;
    validate_finite_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_count_at_least("spectral_function", input.spectral_function.len(), 2)?;
    validate_count_at_least("signal", input.signal.len(), 2)?;
    validate_matching_lengths(
        "spectral_energy",
        input.spectral_energy.len(),
        "spectral_function",
        input.spectral_function.len(),
    )?;
    validate_matching_lengths(
        "signal_energy",
        input.signal_energy.len(),
        "signal",
        input.signal.len(),
    )?;
    validate_count_exact("weights", input.weights.len(), 8)?;
    validate_finite_array("spectral_energy", input.spectral_energy)?;
    validate_finite_array("spectral_function", input.spectral_function)?;
    validate_finite_array("signal_energy", input.signal_energy)?;
    validate_finite_array("signal", input.signal)?;
    validate_finite_array("weights", input.weights)?;
    validate_strictly_increasing("spectral_energy", input.spectral_energy)?;
    validate_strictly_increasing("signal_energy", input.signal_energy)?;
    if input.asymmetric_phase && input.weights[0] == 0.0 {
        return Err(SfconvError::ZeroAsymmetricWeight);
    }
    if input.asymmetric_phase && input.plasma_frequency == 0.0 {
        return Err(SfconvError::ZeroPlasmaFrequency);
    }
    Ok(())
}

fn validate_grater_input(
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<(), SfconvError> {
    validate_finite_scalar("grater lower", lower)?;
    validate_finite_scalar("grater upper", upper)?;
    if upper <= lower {
        return Err(SfconvError::InvalidIntegrationInterval { lower, upper });
    }
    validate_positive_tolerance("abr", absolute_tolerance)?;
    validate_positive_tolerance("rlr", relative_tolerance)?;
    if singularities.len() > SFCONV_GRATER_MAX_SINGULARITIES {
        return Err(SfconvError::TooManySingularities {
            count: singularities.len(),
            max: SFCONV_GRATER_MAX_SINGULARITIES,
        });
    }

    let mut previous = lower;
    for (index, &singularity) in singularities.iter().enumerate() {
        if !singularity.is_finite()
            || singularity <= lower
            || singularity >= upper
            || singularity <= previous
        {
            return Err(SfconvError::InvalidSingularity {
                index,
                value: singularity,
            });
        }
        previous = singularity;
    }
    Ok(())
}

fn validate_positive_tolerance(field: &'static str, value: Real) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfconvError::NonPositiveTolerance { field, value })
    }
}

fn eval_grater_integrand(
    integrand: &mut impl FnMut(Real) -> Result<Real, SfconvError>,
    argument: Real,
    row: usize,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("grater argument", argument)?;
    let value = integrand(argument)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteValue {
            field: "grater integrand",
            row,
            value,
        })
    }
}

fn validate_satellite_context(context: SfconvSatelliteContext) -> Result<(), SfconvError> {
    validate_positive_scalar("plasma_frequency", context.plasma_frequency)?;
    validate_positive_scalar("pole_energy", context.pole_energy)?;
    validate_finite_scalar("dispersion_parameter", context.dispersion_parameter)?;
    validate_positive_scalar("photoelectron_energy", context.photoelectron_energy)?;
    validate_positive_tolerance("accuracy", context.accuracy)
}

fn validate_so2conv_material_input(input: SfconvSo2convMaterialInput) -> Result<(), SfconvError> {
    validate_finite_scalar("core_hole_width_ev", input.core_hole_width_ev)?;
    validate_positive_scalar("wigner_seitz_radius", input.wigner_seitz_radius)?;
    validate_finite_scalar("interstitial_potential_ev", input.interstitial_potential_ev)?;
    validate_finite_scalar("chemical_potential_ev", input.chemical_potential_ev)?;
    validate_finite_scalar(
        "fermi_wave_number_inv_angstrom",
        input.fermi_wave_number_inv_angstrom,
    )
}

fn validate_so2conv_material_parameters(
    parameters: SfconvSo2convMaterialParameters,
) -> Result<(), SfconvError> {
    validate_finite_scalar("core_hole_lifetime", parameters.core_hole_lifetime)?;
    validate_finite_scalar("interstitial_potential", parameters.interstitial_potential)?;
    validate_finite_scalar(
        "chemical_potential_offset",
        parameters.chemical_potential_offset,
    )?;
    validate_finite_scalar("fermi_wave_number", parameters.fermi_wave_number)?;
    validate_positive_scalar("fermi_momentum", parameters.fermi_momentum)?;
    validate_positive_scalar("fermi_energy", parameters.fermi_energy)?;
    validate_positive_scalar("electron_concentration", parameters.electron_concentration)?;
    validate_positive_scalar("plasma_frequency", parameters.plasma_frequency)?;
    validate_finite_scalar("dispersion_parameter", parameters.dispersion_parameter)?;
    validate_finite_scalar(
        "initial_photoelectron_energy",
        parameters.initial_photoelectron_energy,
    )?;
    validate_positive_scalar(
        "initial_photoelectron_momentum",
        parameters.initial_photoelectron_momentum,
    )?;
    validate_positive_tolerance("accuracy", parameters.accuracy)
}

fn validate_self_energy_context(context: SfconvSelfEnergyContext) -> Result<(), SfconvError> {
    validate_positive_scalar("fermi_energy", context.fermi_energy)?;
    validate_positive_scalar("fermi_momentum", context.fermi_momentum)?;
    validate_positive_scalar("plasma_frequency", context.plasma_frequency)?;
    validate_positive_scalar("pole_energy", context.pole_energy)?;
    validate_finite_scalar("quasiparticle_energy", context.quasiparticle_energy)?;
    validate_positive_scalar("photoelectron_momentum", context.photoelectron_momentum)?;
    validate_positive_tolerance("accuracy", context.accuracy)?;
    validate_finite_scalar("pole_broadening", context.pole_broadening)?;
    validate_finite_scalar("dispersion_parameter", context.dispersion_parameter)
}

fn validate_broadened_self_energy_integrand_input(
    input: SfconvBroadenedSelfEnergyIntegrandInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("broadened self-energy momentum", input.momentum)?;
    if input.momentum < 0.0 {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: input.momentum,
            upper: 0.0,
        });
    }
    validate_finite_scalar("self-energy energy", input.energy)?;
    validate_self_energy_derivative_context(input.context)
}

fn validate_so2conv_self_energy_sample_input(
    input: SfconvSo2convSelfEnergySampleInput<'_>,
) -> Result<(), SfconvError> {
    validate_so2conv_material_parameters(input.material)?;
    validate_finite_scalar("self-energy energy", input.energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("photoelectron_momentum", input.photoelectron_momentum)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    validate_active_finite_array("pole_broadening", input.pole_broadening, input.pole_count)
}

fn validate_so2conv_self_energy_grid_input(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("momentum", input.momentum.len(), 1)?;
    validate_finite_array("momentum", input.momentum)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_finite_scalar("fermi_level", input.fermi_level)?;
    validate_so2conv_self_energy_sample_input(SfconvSo2convSelfEnergySampleInput {
        material: input.material,
        energy: 0.0,
        quasiparticle_energy: input.material.fermi_energy,
        photoelectron_momentum: input.material.fermi_momentum,
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
        include_below_fermi: input.include_below_fermi,
    })
}

fn validate_self_energy_derivative_context(
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_self_energy_context(context)?;
    validate_positive_scalar("pole_broadening", context.pole_broadening)
}

fn validate_real_self_energy_integrand_inputs(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)
}

fn validate_real_self_energy_derivative_integrand_inputs(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)
}

fn validate_satellite_self_energy(
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<(), SfconvError> {
    validate_finite_scalar("on_shell_real", self_energy.on_shell_real)?;
    validate_finite_scalar("satellite width", self_energy.width)?;
    validate_finite_scalar("renormalization_real", self_energy.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", self_energy.renormalization_imag)?;
    validate_finite_scalar("off_shell_real", self_energy.off_shell_real)?;
    validate_finite_scalar("off_shell_imag", self_energy.off_shell_imag)
}

fn validate_momentum_spectral_interpolation_input(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_momentum", input.photoelectron_momentum)?;
    let rows = input.momentum_grid.len();
    validate_count_at_least("momentum_grid", rows, 2)?;
    validate_finite_array("momentum_grid", input.momentum_grid)?;
    validate_strictly_increasing("momentum_grid", input.momentum_grid)?;

    let columns = input.energy_grid.ncols();
    validate_count_at_least("spectral columns", columns, 1)?;
    validate_matrix_shape("energy_grid", input.energy_grid, rows, columns)?;
    validate_matrix_shape(
        "extrinsic_quasiparticle",
        input.extrinsic_quasiparticle,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "extrinsic_satellite",
        input.extrinsic_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "interference_quasiparticle",
        input.interference_quasiparticle,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "interference_satellite",
        input.interference_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "intrinsic_satellite",
        input.intrinsic_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "clipped_extrinsic_satellite",
        input.clipped_extrinsic_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape("weights", input.weights, rows, 8)?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "self_energy_real",
        input.self_energy_real.len(),
    )?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "energy_correction",
        input.energy_correction.len(),
    )?;
    validate_matching_lengths("momentum_grid", rows, "width", input.width.len())?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "renormalization_real",
        input.renormalization_real.len(),
    )?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "renormalization_imag",
        input.renormalization_imag.len(),
    )?;

    validate_finite_matrix("energy_grid", input.energy_grid)?;
    validate_finite_matrix("extrinsic_quasiparticle", input.extrinsic_quasiparticle)?;
    validate_finite_matrix("extrinsic_satellite", input.extrinsic_satellite)?;
    validate_finite_matrix(
        "interference_quasiparticle",
        input.interference_quasiparticle,
    )?;
    validate_finite_matrix("interference_satellite", input.interference_satellite)?;
    validate_finite_matrix("intrinsic_satellite", input.intrinsic_satellite)?;
    validate_finite_matrix(
        "clipped_extrinsic_satellite",
        input.clipped_extrinsic_satellite,
    )?;
    validate_finite_matrix("weights", input.weights)?;
    validate_finite_array("self_energy_real", input.self_energy_real)?;
    validate_finite_array("energy_correction", input.energy_correction)?;
    validate_finite_array("width", input.width)?;
    validate_finite_array("renormalization_real", input.renormalization_real)?;
    validate_finite_array("renormalization_imag", input.renormalization_imag)
}

fn validate_photoelectron_momentum_input(
    input: SfconvPhotoelectronMomentumInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("momentum", input.momentum.len(), 2)?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "self_energy",
        input.self_energy.len(),
    )?;
    validate_finite_array("momentum", input.momentum)?;
    validate_finite_array("self_energy", input.self_energy)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_positive_scalar("fermi_momentum", input.fermi_momentum)?;
    validate_finite_scalar("fermi_level", input.fermi_level)?;
    validate_finite_scalar("fermi_self_energy", input.fermi_self_energy)
}

fn validate_quasiparticle_peak_input(
    input: SfconvQuasiparticlePeakInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("center_energy", input.center_energy)?;
    validate_finite_scalar("lower_boundary", input.lower_boundary)?;
    validate_finite_scalar("upper_boundary", input.upper_boundary)?;
    if input.upper_boundary <= input.lower_boundary {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: input.lower_boundary,
            upper: input.upper_boundary,
        });
    }
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)
}

fn validate_exponential_reduction_input(
    input: SfconvExponentialReductionInput<'_>,
) -> Result<(), SfconvError> {
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    Ok(())
}

fn validate_quasiparticle_pole_input(
    input: SfconvQuasiparticlePoleInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_positive_scalar("width", input.width)?;
    validate_finite_scalar("renormalization_real", input.renormalization.real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization.imaginary)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization.magnitude)
}

fn validate_quasiparticle_table_input(
    input: SfconvQuasiparticleTableInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("energy", input.energy.len(), 1)?;
    validate_matching_lengths(
        "boundaries",
        input.boundaries.len(),
        "energy plus endpoints",
        input.energy.len() + 1,
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("endpoint_width", input.endpoint_width)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization_magnitude)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)
}

fn validate_quasiparticle_interference_input(
    input: SfconvQuasiparticleInterferenceInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_finite_scalar("upper_energy", input.upper_energy)?;
    validate_positive_scalar("bare_photoelectron_energy", input.bare_photoelectron_energy)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_finite_scalar("dispersion_parameter", input.dispersion_parameter)?;
    validate_positive_tolerance("accuracy", input.accuracy)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    Ok(())
}

fn validate_satellite_pole_contributions_input(
    input: SfconvSatellitePoleContributionsInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("satellite_energy", input.energy)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_positive_scalar("bare_photoelectron_energy", input.bare_photoelectron_energy)?;
    validate_finite_scalar("dispersion_parameter", input.dispersion_parameter)?;
    validate_positive_tolerance("accuracy", input.accuracy)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    validate_active_finite_array("pole_broadening", input.pole_broadening, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    Ok(())
}

fn validate_extrinsic_satellite_input(
    input: SfconvExtrinsicSatelliteInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("satellite energy", input.energy)?;
    validate_finite_scalar("main_peak", input.main_peak)?;
    validate_finite_scalar("imaginary_derivative", input.imaginary_derivative)?;
    validate_satellite_context(input.context)?;
    validate_satellite_self_energy(input.self_energy)
}

fn validate_spectral_cell_input(input: SfconvSpectralCellInput<'_>) -> Result<(), SfconvError> {
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("imaginary_derivative", input.imaginary_derivative)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_satellite_context(input.context)?;
    validate_satellite_self_energy(input.self_energy)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )
}

fn validate_spectral_table_input(input: SfconvSpectralTableInput<'_>) -> Result<(), SfconvError> {
    let columns = input.energy.len();
    validate_count_at_least("energy", columns, 1)?;
    validate_count_exact("boundaries", input.boundaries.len(), columns + 1)?;
    validate_matching_lengths(
        "off_shell_real",
        input.off_shell_real.len(),
        "energy",
        columns,
    )?;
    validate_matching_lengths(
        "off_shell_imag",
        input.off_shell_imag.len(),
        "energy",
        columns,
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_finite_array("off_shell_real", input.off_shell_real)?;
    validate_finite_array("off_shell_imag", input.off_shell_imag)?;
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("imaginary_derivative", input.imaginary_derivative)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_satellite_context(input.context)?;
    validate_satellite_self_energy(input.self_energy)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    validate_active_finite_array("pole_broadening", input.pole_broadening, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    validate_feff_column_index(
        "quasiparticle_lower_column",
        input.quasiparticle_lower_column_1based,
        columns,
    )?;
    validate_feff_column_index(
        "quasiparticle_upper_column",
        input.quasiparticle_upper_column_1based,
        columns,
    )
}

fn validate_satellite_table_input(input: SfconvSatelliteTableInput<'_>) -> Result<(), SfconvError> {
    let columns = input.extrinsic_satellite.len();
    validate_count_at_least("satellite columns", columns, 1)?;
    validate_matching_lengths(
        "main_peak",
        input.main_peak.len(),
        "satellite columns",
        columns,
    )?;
    validate_matching_lengths(
        "quasiparticle_interference",
        input.quasiparticle_interference.len(),
        "satellite columns",
        columns,
    )?;
    validate_matching_lengths(
        "interference_satellite",
        input.interference_satellite.len(),
        "satellite columns",
        columns,
    )?;
    validate_matching_lengths(
        "intrinsic_satellite",
        input.intrinsic_satellite.len(),
        "satellite columns",
        columns,
    )?;
    validate_count_exact("boundaries", input.boundaries.len(), columns + 1)?;
    validate_finite_array("main_peak", input.main_peak)?;
    validate_finite_array(
        "quasiparticle_interference",
        input.quasiparticle_interference,
    )?;
    validate_finite_array("extrinsic_satellite", input.extrinsic_satellite)?;
    validate_finite_array("interference_satellite", input.interference_satellite)?;
    validate_finite_array("intrinsic_satellite", input.intrinsic_satellite)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_feff_column_index(
        "quasiparticle_lower_column",
        input.quasiparticle_lower_column_1based,
        columns,
    )?;
    validate_feff_column_index(
        "quasiparticle_upper_column",
        input.quasiparticle_upper_column_1based,
        columns,
    )
}

fn validate_extrinsic_satellite_split_input(
    input: SfconvExtrinsicSatelliteSplitInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_count_at_least(
        "spectral_function columns",
        input.spectral_function.ncols(),
        3,
    )?;
    validate_matching_lengths(
        "energy",
        input.energy.len(),
        "spectral_function columns",
        input.spectral_function.ncols(),
    )?;
    validate_count_exact(
        "boundaries",
        input.boundaries.len(),
        input.spectral_function.ncols() + 1,
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("beta_zero", input.beta_zero)?;
    validate_finite_array("extrinsic satellite", input.spectral_function.row(1))?;
    validate_finite_array("intrinsic satellite", input.spectral_function.row(4))
}

fn validate_satellite_correction_input(
    input: SfconvSatelliteCorrectionInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_count_at_least(
        "spectral_function columns",
        input.spectral_function.ncols(),
        1,
    )?;
    validate_count_exact(
        "boundaries",
        input.boundaries.len(),
        input.spectral_function.ncols() + 1,
    )?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_finite_mkspectf_satellite_rows(input.spectral_function)
}

fn validate_spectral_finalization_input(
    input: SfconvSpectralFinalizationInput<'_>,
) -> Result<(), SfconvError> {
    validate_extrinsic_satellite_split_input(SfconvExtrinsicSatelliteSplitInput {
        spectral_function: input.spectral_function,
        energy: input.energy,
        boundaries: input.boundaries,
        photoelectron_energy: input.photoelectron_energy,
        beta_zero: input.beta_zero,
    })?;
    validate_satellite_correction_input(SfconvSatelliteCorrectionInput {
        spectral_function: input.spectral_function,
        boundaries: input.boundaries,
        uniform_width: input.uniform_width,
        exponential_reduction: input.exponential_reduction,
    })?;
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization_magnitude)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)
}

fn validate_spectral_weights_input(
    input: SfconvSpectralWeightsInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization_magnitude)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_count_exact("satellite_weights", input.satellite_weights.len(), 5)?;
    validate_finite_array("satellite_weights", input.satellite_weights)
}

fn validate_feff_path_interpolation_input(
    input: SfconvFeffPathInterpolationInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("source_momentum", input.source_momentum.len(), 1)?;
    validate_count_at_least("path_momentum", input.path_momentum.len(), 2)?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "central_phase",
        input.central_phase.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "effective_amplitude",
        input.effective_amplitude.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "effective_phase",
        input.effective_phase.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "reduction_factor",
        input.reduction_factor.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "mean_free_path",
        input.mean_free_path.len(),
    )?;
    validate_finite_array("source_momentum", input.source_momentum)?;
    validate_strictly_increasing("source_momentum", input.source_momentum)?;
    validate_finite_array("path_momentum", input.path_momentum)?;
    validate_strictly_increasing("path_momentum", input.path_momentum)?;
    validate_finite_array("central_phase", input.central_phase)?;
    validate_finite_array("effective_amplitude", input.effective_amplitude)?;
    validate_finite_array("effective_phase", input.effective_phase)?;
    validate_finite_array("reduction_factor", input.reduction_factor)?;
    validate_finite_array("mean_free_path", input.mean_free_path)
}

fn validate_feff_path_signal_input(
    input: SfconvFeffPathSignalInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("momentum", input.momentum.len(), 3)?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "central_phase",
        input.central_phase.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "effective_amplitude",
        input.effective_amplitude.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "effective_phase",
        input.effective_phase.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "reduction_factor",
        input.reduction_factor.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "mean_free_path",
        input.mean_free_path.len(),
    )?;
    validate_finite_array("momentum", input.momentum)?;
    validate_strictly_increasing("momentum", input.momentum)?;
    validate_finite_array("central_phase", input.central_phase)?;
    validate_finite_array("effective_amplitude", input.effective_amplitude)?;
    validate_finite_array("effective_phase", input.effective_phase)?;
    validate_finite_array("reduction_factor", input.reduction_factor)?;
    validate_finite_array("mean_free_path", input.mean_free_path)?;
    validate_positive_scalar("degeneracy", input.degeneracy)?;
    validate_positive_scalar("half_path_length", input.half_path_length)
}

fn validate_exafs_convolution_input(input: SfconvExafsConvolutionInput) -> Result<(), SfconvError> {
    validate_finite_scalar(
        "real_convolution_amplitude",
        input.real_convolution_amplitude,
    )?;
    validate_finite_scalar("real_convolution_phase", input.real_convolution_phase)?;
    validate_finite_scalar(
        "imaginary_convolution_amplitude",
        input.imaginary_convolution_amplitude,
    )?;
    validate_finite_scalar(
        "imaginary_convolution_phase",
        input.imaginary_convolution_phase,
    )?;
    validate_positive_scalar("original_magnitude", input.original_magnitude)?;
    validate_finite_scalar("original_phase", input.original_phase)?;
    validate_finite_scalar("phase_minus_2kr", input.phase_minus_2kr)?;
    validate_finite_scalar("previous_phase", input.previous_phase)
}

fn validate_xanes_convolution_input(input: SfconvXanesConvolutionInput) -> Result<(), SfconvError> {
    validate_finite_scalar("embedded_background", input.embedded_background)?;
    if input.asymmetric_phase {
        validate_finite_scalar("absorption_convolution", input.absorption_convolution)
    } else {
        validate_finite_scalar(
            "fine_structure_imaginary_amplitude",
            input.fine_structure_imaginary_amplitude,
        )?;
        validate_finite_scalar(
            "fine_structure_imaginary_phase",
            input.fine_structure_imaginary_phase,
        )?;
        validate_finite_scalar(
            "fine_structure_real_amplitude",
            input.fine_structure_real_amplitude,
        )?;
        validate_finite_scalar("fine_structure_real_phase", input.fine_structure_real_phase)
    }
}

fn validate_so2conv_exafs_energy_padding_input(
    input: SfconvSo2convExafsEnergyPaddingInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_active_len("energy", input.active_len, input.energy.len())?;
    validate_active_len("output_len", input.active_len, input.output_len)?;
    validate_active_finite_array("energy", input.energy, input.active_len)?;
    validate_active_strictly_increasing("energy", input.energy, input.active_len)
}

fn validate_so2conv_exafs_preparation_input(
    input: SfconvSo2convExafsPreparationInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_active_len("momentum", input.active_len, input.momentum.len())?;
    validate_active_len("magnitude", input.active_len, input.magnitude.len())?;
    validate_active_len("phase", input.active_len, input.phase.len())?;
    if let Some(phase_minus_2kr) = input.phase_minus_2kr {
        validate_active_len("phase_minus_2kr", input.active_len, phase_minus_2kr.len())?;
        validate_active_finite_array("phase_minus_2kr", phase_minus_2kr, input.active_len)?;
    }
    validate_active_len("output_len", input.active_len, input.output_len)?;
    validate_active_finite_array("momentum", input.momentum, input.active_len)?;
    validate_active_finite_array("magnitude", input.magnitude, input.active_len)?;
    validate_active_finite_array("phase", input.phase, input.active_len)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    for row in 0..input.active_len {
        validate_positive_scalar("magnitude", input.magnitude[row])?;
    }
    Ok(())
}

fn validate_so2conv_xanes_preparation_input(
    input: SfconvSo2convXanesPreparationInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("output_len", input.output_len, 21)?;
    validate_active_len(
        "incident_energy",
        input.active_len,
        input.incident_energy.len(),
    )?;
    validate_active_len(
        "excitation_energy",
        input.active_len,
        input.excitation_energy.len(),
    )?;
    validate_active_len("absorption", input.active_len, input.absorption.len())?;
    validate_active_len(
        "embedded_background",
        input.active_len,
        input.embedded_background.len(),
    )?;
    validate_active_len("output_len", input.active_len, input.output_len)?;
    validate_active_finite_array("incident_energy", input.incident_energy, input.active_len)?;
    validate_active_finite_array(
        "excitation_energy",
        input.excitation_energy,
        input.active_len,
    )?;
    validate_active_finite_array("absorption", input.absorption, input.active_len)?;
    validate_active_finite_array(
        "embedded_background",
        input.embedded_background,
        input.active_len,
    )?;
    validate_active_strictly_increasing(
        "excitation_energy",
        input.excitation_energy,
        input.active_len,
    )
}

fn validate_path_average_input(input: SfconvPathAverageInput<'_>) -> Result<(), SfconvError> {
    validate_count_at_least("source_momentum", input.source_momentum.len(), 1)?;
    validate_matching_lengths(
        "source_momentum",
        input.source_momentum.len(),
        "amplitude_reduction",
        input.amplitude_reduction.len(),
    )?;
    validate_matching_lengths(
        "source_momentum",
        input.source_momentum.len(),
        "phase_shift",
        input.phase_shift.len(),
    )?;
    validate_finite_array("source_momentum", input.source_momentum)?;
    validate_strictly_increasing("source_momentum", input.source_momentum)?;
    validate_finite_array("amplitude_reduction", input.amplitude_reduction)?;
    validate_finite_array("phase_shift", input.phase_shift)?;
    validate_finite_scalar("previous_momentum", input.previous_momentum)?;
    validate_finite_scalar("center_momentum", input.center_momentum)?;
    validate_finite_scalar("next_momentum", input.next_momentum)?;
    if input.previous_momentum > input.center_momentum
        || input.center_momentum > input.next_momentum
    {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: input.previous_momentum,
            upper: input.next_momentum,
        });
    }
    validate_positive_scalar("momentum_step", input.momentum_step)
}

fn set_feff_path_interpolated_row(
    output: &mut SfconvFeffPathInterpolation,
    source_row: usize,
    input: SfconvFeffPathInterpolationInput<'_>,
    lower_row: usize,
) -> Result<(), SfconvError> {
    let upper_row = lower_row + 1;
    let lower_momentum = input.path_momentum[lower_row];
    let upper_momentum = input.path_momentum[upper_row];
    let denominator = upper_momentum - lower_momentum;
    validate_nonzero_denominator("feff path interpolation interval", denominator)?;
    let fraction = (input.source_momentum[source_row] - lower_momentum) / denominator;

    output.central_phase[source_row] = linear_blend(
        input.central_phase[lower_row],
        input.central_phase[upper_row],
        fraction,
    );
    output.effective_amplitude[source_row] = linear_blend(
        input.effective_amplitude[lower_row],
        input.effective_amplitude[upper_row],
        fraction,
    );
    output.effective_phase[source_row] = linear_blend(
        input.effective_phase[lower_row],
        input.effective_phase[upper_row],
        fraction,
    );
    output.reduction_factor[source_row] = linear_blend(
        input.reduction_factor[lower_row],
        input.reduction_factor[upper_row],
        fraction,
    );
    output.mean_free_path[source_row] = linear_blend(
        input.mean_free_path[lower_row],
        input.mean_free_path[upper_row],
        fraction,
    );
    Ok(())
}

fn set_feff_path_exact_row(
    output: &mut SfconvFeffPathInterpolation,
    source_row: usize,
    input: SfconvFeffPathInterpolationInput<'_>,
    path_row: usize,
) {
    output.central_phase[source_row] = input.central_phase[path_row];
    output.effective_amplitude[source_row] = input.effective_amplitude[path_row];
    output.effective_phase[source_row] = input.effective_phase[path_row];
    output.reduction_factor[source_row] = input.reduction_factor[path_row];
    output.mean_free_path[source_row] = input.mean_free_path[path_row];
}

fn find_momentum_spectral_segment(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
) -> Result<usize, SfconvError> {
    for segment in 0..(input.momentum_grid.len() - 1) {
        if input.photoelectron_momentum >= input.momentum_grid[segment]
            && input.photoelectron_momentum < input.momentum_grid[segment + 1]
        {
            return Ok(segment);
        }
    }
    Err(SfconvError::MissingTrigger {
        field: "momentum spectral interval",
    })
}

fn set_momentum_spectral_interpolated_row(
    output: &mut SfconvMomentumSpectralInterpolation,
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    lower_row: usize,
) -> Result<(), SfconvError> {
    let upper_row = lower_row + 1;
    let denominator = input.momentum_grid[upper_row] - input.momentum_grid[lower_row];
    validate_nonzero_denominator("momentum spectral interval", denominator)?;
    let fraction = (input.photoelectron_momentum - input.momentum_grid[lower_row]) / denominator;

    for column in 0..input.energy_grid.ncols() {
        output.energy[column] = linear_blend(
            input.energy_grid[(lower_row, column)],
            input.energy_grid[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(0, column)] = linear_blend(
            input.extrinsic_quasiparticle[(lower_row, column)],
            input.extrinsic_quasiparticle[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(1, column)] = linear_blend(
            input.extrinsic_satellite[(lower_row, column)],
            input.extrinsic_satellite[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(2, column)] = linear_blend(
            input.interference_quasiparticle[(lower_row, column)],
            input.interference_quasiparticle[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(3, column)] = linear_blend(
            input.interference_satellite[(lower_row, column)],
            input.interference_satellite[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(4, column)] = linear_blend(
            input.intrinsic_satellite[(lower_row, column)],
            input.intrinsic_satellite[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(5, column)] = linear_blend(
            combined_momentum_satellite(input, lower_row, column),
            combined_momentum_satellite(input, upper_row, column),
            fraction,
        );
        output.spectral_function[(6, column)] = linear_blend(
            clipped_momentum_satellite(input, lower_row, column),
            clipped_momentum_satellite(input, upper_row, column),
            fraction,
        );
        output.spectral_function[(7, column)] = linear_blend(
            input.clipped_extrinsic_satellite[(lower_row, column)],
            input.clipped_extrinsic_satellite[(upper_row, column)],
            fraction,
        );
    }

    for slot in 0..8 {
        output.weights[slot] = linear_blend(
            input.weights[(lower_row, slot)],
            input.weights[(upper_row, slot)],
            fraction,
        );
    }
    output.self_energy_real = linear_blend(
        input.self_energy_real[lower_row],
        input.self_energy_real[upper_row],
        fraction,
    );
    output.energy_correction = linear_blend(
        input.energy_correction[lower_row],
        input.energy_correction[upper_row],
        fraction,
    );
    output.width = linear_blend(input.width[lower_row], input.width[upper_row], fraction);
    output.renormalization_real = linear_blend(
        input.renormalization_real[lower_row],
        input.renormalization_real[upper_row],
        fraction,
    );
    output.renormalization_imag = linear_blend(
        input.renormalization_imag[lower_row],
        input.renormalization_imag[upper_row],
        fraction,
    );
    Ok(())
}

fn set_momentum_spectral_exact_row(
    output: &mut SfconvMomentumSpectralInterpolation,
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    energy_row: usize,
    data_row: usize,
) {
    for column in 0..input.energy_grid.ncols() {
        output.energy[column] = input.energy_grid[(energy_row, column)];
        output.spectral_function[(0, column)] = input.extrinsic_quasiparticle[(data_row, column)];
        output.spectral_function[(1, column)] = input.extrinsic_satellite[(data_row, column)];
        output.spectral_function[(2, column)] =
            input.interference_quasiparticle[(data_row, column)];
        output.spectral_function[(3, column)] = input.interference_satellite[(data_row, column)];
        output.spectral_function[(4, column)] = input.intrinsic_satellite[(data_row, column)];
        output.spectral_function[(5, column)] =
            combined_momentum_satellite(input, data_row, column);
        output.spectral_function[(6, column)] = clipped_momentum_satellite(input, data_row, column);
        output.spectral_function[(7, column)] =
            input.clipped_extrinsic_satellite[(data_row, column)];
    }
    for slot in 0..8 {
        output.weights[slot] = input.weights[(data_row, slot)];
    }
    output.self_energy_real = input.self_energy_real[data_row];
    output.energy_correction = input.energy_correction[data_row];
    output.width = input.width[data_row];
    output.renormalization_real = input.renormalization_real[data_row];
    output.renormalization_imag = input.renormalization_imag[data_row];
}

fn combined_momentum_satellite(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    row: usize,
    column: usize,
) -> Real {
    input.extrinsic_satellite[(row, column)] + input.intrinsic_satellite[(row, column)]
        - 2.0 * input.interference_satellite[(row, column)]
}

fn clipped_momentum_satellite(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    row: usize,
    column: usize,
) -> Real {
    input.extrinsic_satellite[(row, column)] - input.clipped_extrinsic_satellite[(row, column)]
}

fn linear_blend(lower: Real, upper: Real, fraction: Real) -> Real {
    lower + (upper - lower) * fraction
}

fn feff_path_signal_magnitude(
    input: SfconvFeffPathSignalInput<'_>,
    row: usize,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("path signal momentum", input.momentum[row])?;
    let path_factor =
        input.degeneracy * input.effective_amplitude[row] * input.reduction_factor[row];
    if path_factor == 0.0 {
        return Ok(0.0);
    }

    validate_positive_scalar("mean_free_path", input.mean_free_path[row])?;
    let attenuation = (-2.0 * input.half_path_length / input.mean_free_path[row]).exp();
    let denominator = input.momentum[row] * input.half_path_length.powi(2);
    validate_nonzero_denominator("feff path signal magnitude", denominator)?;
    finite_result(
        "feff path signal magnitude",
        path_factor * attenuation / denominator,
    )
}

fn checked_hypot(field: &'static str, left: Real, right: Real) -> Result<Real, SfconvError> {
    validate_finite_scalar(field, left)?;
    validate_finite_scalar(field, right)?;
    let value = left.hypot(right);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn so2conv_update_phase_jump_count(
    phase_jump_count: i32,
    phase: Real,
    previous_phase: Real,
) -> Result<i32, SfconvError> {
    let delta = if phase - previous_phase > 5.0 {
        2
    } else if phase - previous_phase < -5.0 {
        -2
    } else {
        0
    };
    phase_jump_count
        .checked_add(delta)
        .ok_or(SfconvError::PhaseJumpOverflow {
            value: phase_jump_count,
            delta,
        })
}

fn validate_nonzero_denominator(field: &'static str, value: Real) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value == 0.0 {
        Err(SfconvError::ZeroDenominator { field })
    } else {
        Ok(())
    }
}

fn finite_result(field: &'static str, value: Real) -> Result<Real, SfconvError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn add_real_self_energy_range(
    total: &mut SfconvAdaptiveIntegral,
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    integrand: impl FnMut(Real) -> Result<Real, SfconvError>,
) -> Result<(), SfconvError> {
    let current = sfconv_grater_integrate(
        integrand,
        lower,
        upper,
        absolute_tolerance,
        relative_tolerance,
        &[],
    )?;
    total.value += current.value;
    total.estimated_error += current.estimated_error;
    total.evaluations += current.evaluations;
    total.max_regions = total.max_regions.max(current.max_regions);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct BroadenedSelfEnergyRangeInput<'a> {
    branch: SfconvBroadenedSelfEnergyBranch,
    lower: Real,
    upper: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
    singularity_candidates: ArrayView1<'a, Real>,
    absolute_tolerance: Real,
    relative_tolerance: Real,
}

#[derive(Debug, Clone, Copy, Default)]
struct BroadenedSelfEnergyAccumulator {
    log_real: Real,
    log_imag: Real,
    atan_real: Real,
    atan_imag: Real,
    log_real_error: Real,
    log_imag_error: Real,
    atan_real_error: Real,
    atan_imag_error: Real,
    evaluations: usize,
    max_regions: usize,
}

impl BroadenedSelfEnergyAccumulator {
    fn add(&mut self, range: BroadenedSelfEnergyRange) {
        self.log_real += range.log_real.value;
        self.log_imag += range.log_imag.value;
        self.atan_real += range.atan_real.value;
        self.atan_imag += range.atan_imag.value;
        self.log_real_error += range.log_real.estimated_error;
        self.log_imag_error += range.log_imag.estimated_error;
        self.atan_real_error += range.atan_real.estimated_error;
        self.atan_imag_error += range.atan_imag.estimated_error;
        self.evaluations += range.log_real.evaluations
            + range.log_imag.evaluations
            + range.atan_real.evaluations
            + range.atan_imag.evaluations;
        self.max_regions = self
            .max_regions
            .max(range.log_real.max_regions)
            .max(range.log_imag.max_regions)
            .max(range.atan_real.max_regions)
            .max(range.atan_imag.max_regions);
    }
}

#[derive(Debug, Clone, Copy)]
struct BroadenedSelfEnergyRange {
    log_real: SfconvAdaptiveIntegral,
    log_imag: SfconvAdaptiveIntegral,
    atan_real: SfconvAdaptiveIntegral,
    atan_imag: SfconvAdaptiveIntegral,
}

fn integrate_broadened_self_energy_range(
    total: &mut BroadenedSelfEnergyAccumulator,
    input: BroadenedSelfEnergyRangeInput<'_>,
) -> Result<(), SfconvError> {
    if input.lower == input.upper {
        return Ok(());
    }
    let singularities =
        sfconv_find_singularities(input.lower, input.upper, input.singularity_candidates)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
    let range = BroadenedSelfEnergyRange {
        log_real: integrate_broadened_self_energy_component(input, &singularities, |integrands| {
            integrands.log_real
        })?,
        log_imag: integrate_broadened_self_energy_component(input, &singularities, |integrands| {
            integrands.log_imag
        })?,
        atan_real: integrate_broadened_self_energy_component(
            input,
            &singularities,
            |integrands| integrands.atan_real,
        )?,
        atan_imag: integrate_broadened_self_energy_component(
            input,
            &singularities,
            |integrands| integrands.atan_imag,
        )?,
    };
    total.add(range);
    Ok(())
}

fn integrate_broadened_self_energy_component(
    input: BroadenedSelfEnergyRangeInput<'_>,
    singularities: &[Real],
    select: impl Fn(SfconvBroadenedSelfEnergyIntegrands) -> Real,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| {
            let integrands = sfconv_broadened_self_energy_integrands(
                input.branch,
                SfconvBroadenedSelfEnergyIntegrandInput {
                    momentum,
                    energy: input.energy,
                    context: input.context,
                },
            )?;
            finite_result("broadened self-energy component", select(integrands))
        },
        input.lower,
        input.upper,
        input.absolute_tolerance,
        input.relative_tolerance,
        singularities,
    )
}

fn integrate_broadened_self_energy_derivative_range(
    total: &mut BroadenedSelfEnergyAccumulator,
    input: BroadenedSelfEnergyRangeInput<'_>,
) -> Result<(), SfconvError> {
    if input.lower == input.upper {
        return Ok(());
    }
    let singularities =
        sfconv_find_singularities(input.lower, input.upper, input.singularity_candidates)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
    let range = BroadenedSelfEnergyRange {
        log_real: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.log_real,
        )?,
        log_imag: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.log_imag,
        )?,
        atan_real: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.atan_real,
        )?,
        atan_imag: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.atan_imag,
        )?,
    };
    total.add(range);
    Ok(())
}

fn integrate_broadened_self_energy_derivative_component(
    input: BroadenedSelfEnergyRangeInput<'_>,
    singularities: &[Real],
    select: impl Fn(SfconvBroadenedSelfEnergyDerivativeIntegrands) -> Real,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| {
            let integrands = sfconv_broadened_self_energy_derivative_integrands(
                input.branch,
                SfconvBroadenedSelfEnergyIntegrandInput {
                    momentum,
                    energy: input.energy,
                    context: input.context,
                },
            )?;
            finite_result(
                "broadened self-energy derivative component",
                select(integrands),
            )
        },
        input.lower,
        input.upper,
        input.absolute_tolerance,
        input.relative_tolerance,
        singularities,
    )
}

fn beta_prefactor(context: SfconvSelfEnergyContext) -> Real {
    context.plasma_frequency.powi(2)
        / (4.0 * std::f64::consts::PI * context.photoelectron_momentum * context.pole_energy)
}

fn beta_log_argument(
    numerator_momentum: Real,
    numerator_dispersion: Real,
    denominator_momentum: Real,
    denominator_dispersion: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    let numerator_denominator = pole_energy
        + numerator_dispersion
        + dispersion_parameter * numerator_momentum.powi(2) / (2.0 * pole_energy);
    let denominator_denominator = pole_energy
        + denominator_dispersion
        + dispersion_parameter * denominator_momentum.powi(2) / (2.0 * pole_energy);
    validate_nonzero_denominator("beta numerator", numerator_denominator)?;
    validate_nonzero_denominator("beta denominator", denominator_momentum)?;
    validate_nonzero_denominator("beta denominator", denominator_denominator)?;
    let argument = numerator_momentum.powi(2) / numerator_denominator * denominator_denominator
        / denominator_momentum.powi(2);
    validate_positive_scalar("beta logarithm", argument)?;
    Ok(argument)
}

fn real_self_energy_log_integrand(
    field: &'static str,
    momentum: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
    numerator: Real,
    denominator: Real,
) -> Result<Real, SfconvError> {
    validate_nonzero_denominator(field, denominator)?;
    real_self_energy_log_integrand_with_ratio(
        field,
        momentum,
        context,
        dispersion,
        numerator / denominator,
    )
}

fn real_self_energy_log_integrand_with_ratio(
    field: &'static str,
    momentum: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
    ratio: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar(field, ratio)?;
    validate_nonzero_denominator(field, dispersion)?;
    let denominator = dispersion
        * checked_sqrt(
            field,
            momentum.powi(2) + context.pole_energy * context.accuracy,
        )?;
    validate_nonzero_denominator(field, denominator)?;
    finite_result(field, ratio.ln() / (2.0 * denominator))
}

fn derivative_lorentz_term(value: Real, broadening: Real) -> Result<Real, SfconvError> {
    let denominator = value.powi(2) + broadening.powi(2);
    validate_nonzero_denominator("self-energy derivative denominator", denominator)?;
    finite_result("self-energy derivative term", value / denominator)
}

fn real_self_energy_derivative_integrand(
    field: &'static str,
    momentum: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
    term: Real,
) -> Result<Real, SfconvError> {
    validate_nonzero_denominator(field, dispersion)?;
    let denominator = dispersion
        * checked_sqrt(
            field,
            momentum.powi(2) + context.pole_energy * context.accuracy,
        )?;
    validate_nonzero_denominator(field, denominator)?;
    finite_result(field, term / denominator)
}

fn self_energy_fermi_limit_derivatives(
    shifted_energy: Real,
    qh: Real,
    q0: Real,
    context: SfconvSelfEnergyContext,
) -> Result<(Real, Real), SfconvError> {
    let upper_gap = shifted_energy - context.fermi_energy;
    let lower_gap = context.fermi_energy - shifted_energy;
    if upper_gap > context.pole_energy {
        let denominator = qh
            * checked_sqrt(
                "imaginary derivative high limit",
                context.dispersion_parameter.powi(2) + upper_gap.powi(2)
                    - context.pole_energy.powi(2),
            )?;
        validate_nonzero_denominator("imaginary derivative high limit", denominator)?;
        Ok((upper_gap / denominator, 0.0))
    } else if lower_gap > context.pole_energy {
        let denominator = q0
            * checked_sqrt(
                "imaginary derivative low limit",
                context.dispersion_parameter.powi(2) + lower_gap.powi(2)
                    - context.pole_energy.powi(2),
            )?;
        validate_nonzero_denominator("imaginary derivative low limit", denominator)?;
        Ok((0.0, -lower_gap / denominator))
    } else {
        Ok((0.0, 0.0))
    }
}

fn self_energy_upper_limit_derivative(
    momentum: &mut Real,
    fermi_limit: Real,
    fermi_limit_derivative: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    let dispersion =
        sfconv_pole_dispersion(*momentum, context.pole_energy, context.dispersion_parameter)?;
    let plus_test =
        (context.photoelectron_momentum + *momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let minus_test =
        (context.photoelectron_momentum - *momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    if *momentum >= fermi_limit {
        *momentum = fermi_limit;
        Ok(fermi_limit_derivative)
    } else if plus_test.abs() < minus_test.abs() {
        self_energy_q_limit_derivative(*momentum, dispersion, context, 1.0, 1.0)
    } else {
        self_energy_q_limit_derivative(*momentum, dispersion, context, -1.0, 1.0)
    }
}

fn self_energy_lower_limit_derivative(
    momentum: &mut Real,
    fermi_limit: Real,
    fermi_limit_derivative: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    let dispersion =
        sfconv_pole_dispersion(*momentum, context.pole_energy, context.dispersion_parameter)?;
    let plus_test =
        (context.photoelectron_momentum + *momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    let minus_test =
        (context.photoelectron_momentum - *momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    if *momentum >= fermi_limit {
        *momentum = fermi_limit;
        Ok(fermi_limit_derivative)
    } else if plus_test.abs() < minus_test.abs() {
        self_energy_q_limit_derivative(*momentum, dispersion, context, 1.0, -1.0)
    } else {
        self_energy_q_limit_derivative(*momentum, dispersion, context, -1.0, -1.0)
    }
}

fn self_energy_q_limit_derivative(
    momentum: Real,
    dispersion: Real,
    context: SfconvSelfEnergyContext,
    momentum_sign: Real,
    dispersion_sign: Real,
) -> Result<Real, SfconvError> {
    let denominator = (momentum + momentum_sign * context.photoelectron_momentum) * dispersion
        + dispersion_sign * (context.dispersion_parameter * momentum + momentum.powi(3) / 2.0);
    validate_nonzero_denominator("imaginary derivative momentum limit", denominator)?;
    finite_result(
        "imaginary derivative momentum limit",
        dispersion / denominator,
    )
}

fn self_energy_imaginary_derivative_factor(
    momentum: Real,
    dispersion: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_nonzero_denominator("imaginary derivative momentum", momentum)?;
    validate_nonzero_denominator("imaginary derivative dispersion", dispersion)?;
    validate_nonzero_denominator("imaginary derivative pole", pole_energy)?;
    let denominator =
        pole_energy + dispersion + dispersion_parameter * momentum.powi(2) / (2.0 * pole_energy);
    validate_nonzero_denominator("imaginary derivative factor", denominator)?;
    let slope = dispersion_parameter * momentum * (1.0 / dispersion + 1.0 / pole_energy)
        + momentum.powi(3) / (2.0 * dispersion);
    finite_result(
        "imaginary derivative factor",
        2.0 / momentum - slope / denominator,
    )
}

fn broadened_self_energy_log_ratio(
    branch: SfconvBroadenedSelfEnergyBranch,
    momentum: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
) -> Result<Real, SfconvError> {
    let minus_energy = (context.photoelectron_momentum - momentum).powi(2) / 2.0;
    let plus_energy = (context.photoelectron_momentum + momentum).powi(2) / 2.0;
    let broadening = context.pole_broadening;
    let (numerator_arg, denominator_arg) = match branch {
        SfconvBroadenedSelfEnergyBranch::ParticlePair => (
            minus_energy - shifted_energy + dispersion,
            plus_energy - shifted_energy + dispersion,
        ),
        SfconvBroadenedSelfEnergyBranch::ParticleFermi => (
            context.fermi_energy - shifted_energy + dispersion,
            plus_energy - shifted_energy + dispersion,
        ),
        SfconvBroadenedSelfEnergyBranch::HoleFermi => (
            minus_energy - shifted_energy - dispersion,
            context.fermi_energy - shifted_energy - dispersion,
        ),
        SfconvBroadenedSelfEnergyBranch::HolePair => (
            minus_energy - shifted_energy - dispersion,
            plus_energy - shifted_energy - dispersion,
        ),
    };
    let numerator = finite_result(
        "broadened self-energy log numerator",
        numerator_arg.powi(2) + broadening.powi(2),
    )?;
    let denominator = finite_result(
        "broadened self-energy log denominator",
        denominator_arg.powi(2) + broadening.powi(2),
    )?;
    validate_nonzero_denominator("broadened self-energy log denominator", denominator)?;
    let ratio = numerator / denominator;
    validate_positive_scalar("broadened self-energy log ratio", ratio)?;
    Ok(ratio)
}

fn broadened_self_energy_atan_delta(
    branch: SfconvBroadenedSelfEnergyBranch,
    momentum: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
) -> Real {
    let broadening = context.pole_broadening;
    let (left, right) = broadened_self_energy_response_arguments(
        branch,
        momentum,
        shifted_energy,
        context,
        dispersion,
    );
    (left / broadening).atan() - (right / broadening).atan()
}

fn broadened_self_energy_response_arguments(
    branch: SfconvBroadenedSelfEnergyBranch,
    momentum: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
) -> (Real, Real) {
    let minus_energy = (context.photoelectron_momentum - momentum).powi(2) / 2.0;
    let plus_energy = (context.photoelectron_momentum + momentum).powi(2) / 2.0;
    match branch {
        SfconvBroadenedSelfEnergyBranch::ParticlePair => (
            shifted_energy - dispersion - minus_energy,
            shifted_energy - dispersion - plus_energy,
        ),
        SfconvBroadenedSelfEnergyBranch::ParticleFermi => (
            shifted_energy - dispersion - context.fermi_energy,
            shifted_energy - dispersion - plus_energy,
        ),
        SfconvBroadenedSelfEnergyBranch::HoleFermi => (
            shifted_energy + dispersion - minus_energy,
            shifted_energy + dispersion - context.fermi_energy,
        ),
        SfconvBroadenedSelfEnergyBranch::HolePair => (
            shifted_energy + dispersion - minus_energy,
            shifted_energy + dispersion - plus_energy,
        ),
    }
}

fn integrate_mksat_range(
    lower: Real,
    upper: Real,
    context: SfconvSatelliteContext,
    mut integrand: impl FnMut(Real, SfconvSatelliteContext) -> Result<Real, SfconvError>,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| integrand(momentum, context),
        lower,
        upper,
        context.plasma_frequency * context.accuracy,
        context.accuracy,
        &[],
    )
}

fn combine_satellite_integrals(
    first: SfconvAdaptiveIntegral,
    second: SfconvAdaptiveIntegral,
    normalization: Real,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_nonzero_denominator("satellite normalization", normalization)?;
    let value = finite_result(
        "satellite integral",
        (first.value + second.value) / normalization,
    )?;
    Ok(SfconvSatelliteIntegral {
        value,
        estimated_error: (first.estimated_error + second.estimated_error) / normalization.abs(),
        evaluations: first.evaluations + second.evaluations,
        max_regions: first.max_regions.max(second.max_regions),
    })
}

fn validate_dispersion_inputs(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)
}

fn pole_dispersion_value(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    let radicand =
        pole_energy.powi(2) + dispersion_parameter * momentum.powi(2) + momentum.powi(4) / 4.0;
    checked_sqrt("pole_dispersion", radicand)
}

fn checked_sqrt(field: &'static str, value: Real) -> Result<Real, SfconvError> {
    if !value.is_finite() {
        return Err(SfconvError::NonFiniteScalar { field, value });
    }
    if value < 0.0 {
        return Err(SfconvError::NegativeRadicand { field, value });
    }
    Ok(value.sqrt())
}

fn threshold_factor(
    dispersion_parameter: Real,
    pole_energy: Real,
    root: Real,
) -> Result<Real, SfconvError> {
    let radicand =
        dispersion_parameter.powi(2) + (root.powi(2) / 2.0).powi(2) - pole_energy.powi(2);
    Ok(checked_sqrt("qthresh factor", radicand)? - dispersion_parameter)
}

fn roots_sorted_by_imag_descending(mut roots: [crate::Complex; 3]) -> [crate::Complex; 3] {
    loop {
        let mut swaps = 0;
        for index in 0..2 {
            if roots[index].im < roots[index + 1].im {
                roots.swap(index, index + 1);
                swaps += 1;
            }
        }
        if swaps == 0 {
            return roots;
        }
    }
}

const fn feff_index(index_1based: usize) -> usize {
    index_1based - 1
}

fn select_threshold_root<F>(
    roots: [crate::Complex; 3],
    score: F,
) -> Result<crate::Complex, SfconvError>
where
    F: FnMut(Real) -> Result<Real, SfconvError>,
{
    let index = select_threshold_root_index(roots, score)?;
    Ok(roots[index])
}

fn select_threshold_root_index<F>(
    roots: [crate::Complex; 3],
    mut score: F,
) -> Result<usize, SfconvError>
where
    F: FnMut(Real) -> Result<Real, SfconvError>,
{
    let test0 = score(roots[0].re)?;
    let test1 = score(roots[1].re)?;
    let test2 = score(roots[2].re)?;
    if test0 < test1 && test0 < test2 {
        Ok(0)
    } else if test1 < test2 {
        Ok(1)
    } else {
        Ok(2)
    }
}

fn cutoff_weight(
    cutoff: bool,
    available_energy: Real,
    chemical_potential: Real,
    gamma: Real,
) -> Real {
    if !cutoff {
        1.0
    } else if available_energy - chemical_potential != 0.0 {
        gamma.atan2(chemical_potential - available_energy) / std::f64::consts::PI
    } else {
        0.5
    }
}

fn interpolated_signal(
    input: SfconvConvolutionInput<'_>,
    available_energy: Real,
) -> Result<Real, SfconvError> {
    let last = input.signal.len() - 1;
    if available_energy > input.signal_energy[last] {
        return Ok(input.signal[last]);
    }
    if available_energy <= input.signal_energy[0] {
        let amplitude = input.signal[0];
        let delta = input.chemical_potential - input.signal_energy[0];
        let lambda = delta.powi(2)
            / (std::f64::consts::PI
                * amplitude.abs()
                * (delta.powi(2) + input.core_hole_lifetime.powi(2)));
        let signal = amplitude * (lambda * (available_energy - input.signal_energy[0])).exp();
        if signal.is_finite() {
            return Ok(signal);
        }
        return Err(SfconvError::NonFiniteResult {
            row: 2,
            value: signal,
        });
    }

    for row in 0..last {
        if available_energy > input.signal_energy[row]
            && available_energy <= input.signal_energy[row + 1]
        {
            let fraction = (available_energy - input.signal_energy[row])
                / (input.signal_energy[row + 1] - input.signal_energy[row]);
            return Ok(input.signal[row] + (input.signal[row + 1] - input.signal[row]) * fraction);
        }
    }

    Err(SfconvError::NonFiniteResult {
        row: 3,
        value: available_energy,
    })
}

fn integration_width(energy: ArrayView1<'_, Real>, active_len: usize, row: usize) -> Real {
    if row == 0 {
        energy[1] - energy[0]
    } else if row + 1 == active_len {
        energy[active_len - 1] - energy[active_len - 2]
    } else {
        0.5 * (energy[row + 1] - energy[row - 1])
    }
}

fn combined_spectral_function(spectral_function: ArrayView2<'_, Real>, column: usize) -> Real {
    spectral_function[(1, column)] + spectral_function[(4, column)]
        - 2.0 * spectral_function[(3, column)]
}

fn validate_finite_matrix(
    field: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    let columns = values.ncols();
    for row in 0..values.nrows() {
        for column in 0..columns {
            validate_finite_value(field, row * columns + column, values[(row, column)])?;
        }
    }
    Ok(())
}

fn validate_finite_spectral_rows(
    spectral_function: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    for &row in &[1, 3, 4] {
        for column in 0..spectral_function.ncols() {
            validate_finite_value(
                "spectral_function",
                column,
                spectral_function[(row, column)],
            )?;
        }
    }
    Ok(())
}

fn validate_finite_mkspectf_satellite_rows(
    spectral_function: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    let columns = spectral_function.ncols();
    for &row in &[1, 3, 4, 6, 7] {
        for column in 0..columns {
            validate_finite_value(
                "spectral_function",
                row * columns + column,
                spectral_function[(row, column)],
            )?;
        }
    }
    Ok(())
}

fn validate_matching_lengths(
    left: &'static str,
    left_len: usize,
    right: &'static str,
    right_len: usize,
) -> Result<(), SfconvError> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(SfconvError::LengthMismatch {
            left,
            left_len,
            right,
            right_len,
        })
    }
}

fn validate_matrix_shape(
    field: &'static str,
    matrix: ArrayView2<'_, Real>,
    rows: usize,
    columns: usize,
) -> Result<(), SfconvError> {
    validate_count_exact(field, matrix.nrows(), rows)?;
    validate_count_exact(field, matrix.ncols(), columns)
}

fn validate_count_exact(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SfconvError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SfconvError::CountMismatch {
            field,
            actual,
            expected,
        })
    }
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), SfconvError> {
    if actual < minimum {
        Err(SfconvError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), SfconvError> {
    if active_len > len {
        Err(SfconvError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn validate_positive_scalar(field: &'static str, value: Real) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfconvError::NonPositiveScalar { field, value })
    }
}

fn validate_feff_column_index(
    field: &'static str,
    index_1based: usize,
    len: usize,
) -> Result<(), SfconvError> {
    if index_1based == 0 || index_1based > len {
        Err(SfconvError::IndexOutOfRange {
            field,
            index: index_1based,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_array(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for (row, value) in values.iter().copied().enumerate() {
        validate_finite_value(field, row, value)?;
    }
    Ok(())
}

fn validate_active_finite_array(
    field: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), SfconvError> {
    for row in 0..active_len {
        validate_finite_value(field, row, values[row])?;
    }
    Ok(())
}

fn validate_finite_value(field: &'static str, row: usize, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteValue { field, row, value })
    }
}

fn validate_strictly_increasing(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for row in 1..values.len() {
        if values[row] <= values[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field,
                row,
                previous: values[row - 1],
                current: values[row],
            });
        }
    }
    Ok(())
}

fn validate_active_strictly_increasing(
    field: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), SfconvError> {
    for row in 1..active_len {
        if values[row] <= values[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field,
                row,
                previous: values[row - 1],
                current: values[row],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
