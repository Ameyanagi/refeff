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
mod tests {
    use ndarray::{Array1, Array2, ShapeBuilder, array};

    use crate::Real;

    use super::{
        SFCONV_MKSPECTF_GRID_LEN, SFCONV_SO2CONV_MOMENTUM_GRID_LEN, SfconvAdaptiveIntegral,
        SfconvBroadenedSelfEnergyBranch, SfconvBroadenedSelfEnergyDerivativeIntegrands,
        SfconvBroadenedSelfEnergyIntegrandInput, SfconvBroadenedSelfEnergyIntegrands,
        SfconvConvolutionInput, SfconvError, SfconvExafsConvolutionInput,
        SfconvExponentialReductionInput, SfconvExtrinsicSatelliteInput,
        SfconvExtrinsicSatelliteMode, SfconvExtrinsicSatelliteSplitInput,
        SfconvFeffPathInterpolationInput, SfconvFeffPathSignalInput, SfconvKramersKronigInput,
        SfconvMomentumSpectralInterpolation, SfconvMomentumSpectralInterpolationInput,
        SfconvPathAverageInput, SfconvPhotoelectronMomentumInput, SfconvPole, SfconvQLimits,
        SfconvQuasiparticleInterferenceInput, SfconvQuasiparticlePeakInput,
        SfconvQuasiparticlePoleInput, SfconvQuasiparticleTableInput, SfconvRenormalization,
        SfconvSatelliteContext, SfconvSatelliteCorrectionInput,
        SfconvSatellitePoleContributionsInput, SfconvSatelliteSelfEnergy,
        SfconvSatelliteTableInput, SfconvSelfEnergyContext, SfconvSo2convExafsEnergyPaddingInput,
        SfconvSo2convExafsPreparationInput, SfconvSo2convMaterialInput,
        SfconvSo2convMaterialParameters, SfconvSo2convSelfEnergyGridInput,
        SfconvSo2convSelfEnergySampleInput, SfconvSo2convXanesPreparationInput,
        SfconvSpectralCellInput, SfconvSpectralEnergyGrid, SfconvSpectralInterpolationInput,
        SfconvSpectralWeightsInput, SfconvXanesConvolutionInput, sfconv_broadened_self_energy,
        sfconv_broadened_self_energy_derivative,
        sfconv_broadened_self_energy_derivative_integrands,
        sfconv_broadened_self_energy_integrands, sfconv_convolve, sfconv_correct_satellite_weights,
        sfconv_coupling_potential_squared, sfconv_exafs_convolution, sfconv_exponential_reduction,
        sfconv_extrinsic_beta, sfconv_extrinsic_satellite, sfconv_extrinsic_satellite_broadened,
        sfconv_extrinsic_satellite_debroadened, sfconv_feff_path_signal, sfconv_find_singularities,
        sfconv_free_electron_exchange, sfconv_grater_integrate, sfconv_imaginary_self_energy,
        sfconv_imaginary_self_energy_derivative, sfconv_interference_quasiparticle,
        sfconv_interference_quasiparticle_integrand, sfconv_interference_satellite,
        sfconv_interference_satellite_integrand, sfconv_interpolate_feff_path,
        sfconv_interpolate_momentum_spectral_function, sfconv_interpolate_spectral_function,
        sfconv_intrinsic_satellite, sfconv_intrinsic_satellite_integrand,
        sfconv_inverse_pole_dispersion, sfconv_kramers_kronig_real_part, sfconv_path_average,
        sfconv_plasma_parameters, sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion,
        sfconv_pole_dispersion_derivative, sfconv_pole_dispersion_second_derivative,
        sfconv_q_limits, sfconv_quasiparticle_interference_amplitude,
        sfconv_quasiparticle_main_peak, sfconv_quasiparticle_pole, sfconv_quasiparticle_table,
        sfconv_real_self_energy, sfconv_real_self_energy_derivative,
        sfconv_real_self_energy_derivative_integrand_lower,
        sfconv_real_self_energy_derivative_integrand_middle,
        sfconv_real_self_energy_derivative_integrand_upper,
        sfconv_real_self_energy_integrand_lower, sfconv_real_self_energy_integrand_middle,
        sfconv_real_self_energy_integrand_upper, sfconv_satellite_pole_contributions,
        sfconv_satellite_table, sfconv_select_pole, sfconv_self_energy_renormalization,
        sfconv_so2conv_broadened_self_energy_grid, sfconv_so2conv_broadened_self_energy_sample,
        sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
        sfconv_so2conv_pad_exafs_energy_grid, sfconv_so2conv_photoelectron_momentum,
        sfconv_so2conv_prepare_exafs_signal, sfconv_so2conv_prepare_xanes_signal,
        sfconv_so2conv_unbroadened_self_energy_grid, sfconv_so2conv_unbroadened_self_energy_sample,
        sfconv_spectral_cell, sfconv_spectral_energy_grid, sfconv_spectral_weights,
        sfconv_split_extrinsic_satellite, sfconv_xanes_convolution,
    };

    #[test]
    fn kramers_kronig_real_part_matches_feff_mkrmu_reference() -> Result<(), SfconvError> {
        let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(25);

        let real_part = sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: energy.view(),
            active_len: 25,
        })?;

        let expected = [
            0.653_321_127_749_770_8,
            0.750_003_058_275_569_8,
            0.770_088_761_144_957_1,
            0.744_953_602_096_770_5,
            0.685_875_097_053_667_7,
            0.599_956_814_602_449_9,
            0.492_993_575_338_788_3,
            0.370_329_818_936_448_6,
            0.237_144_234_118_930_07,
            0.098_519_596_973_469_21,
            -0.040_581_567_325_286_456,
            -0.175_385_521_001_154_32,
            -0.301_395_336_623_902_3,
            -0.414_483_981_972_534_94,
            -0.510_982_552_336_513_5,
            -0.587_755_578_520_523_2,
            -0.642_255_441_484_044_2,
            -0.672_546_008_587_787_2,
            -0.677_279_884_911_601_4,
            -0.631_242_351_812_862_9,
            -0.631_242_351_812_862_9,
            -0.530_174_264_181_443_8,
            -0.422_544_809_832_420_15,
            -0.273_383_187_221_121_7,
            -0.036_668_636_491_773_95,
        ];
        for (actual, expected) in real_part.iter().zip(expected) {
            assert_close(*actual, expected, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn kramers_kronig_real_part_rejects_invalid_inputs() {
        let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(21);

        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 20,
            }),
            Err(SfconvError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 22,
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "imaginary",
                ..
            })
        ));

        let mut bad_imaginary = imaginary.clone();
        bad_imaginary[3] = f64::NAN;
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: bad_imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 21,
            }),
            Err(SfconvError::NonFiniteValue {
                field: "imaginary",
                row: 3,
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[5] = bad_energy[4];
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: bad_energy.view(),
                active_len: 21,
            }),
            Err(SfconvError::NonIncreasingEnergy { row: 5, .. })
        ));
    }

    #[test]
    fn selects_pole_parameters_matches_feff_plset_reference() -> Result<(), SfconvError> {
        let (energy, weight, broadening) = plset_reference_inputs();

        assert_pole_close(
            sfconv_select_pole(3, energy.view(), weight.view(), broadening.view())?,
            SfconvPole {
                energy: 0.495,
                weight: 0.46,
                broadening: 0.048,
            },
        );
        assert_pole_close(
            sfconv_select_pole(5, energy.view(), weight.view(), broadening.view())?,
            SfconvPole {
                energy: 0.975,
                weight: 0.600_000_000_000_000_1,
                broadening: 0.1,
            },
        );
        Ok(())
    }

    #[test]
    fn selects_pole_parameters_rejects_invalid_inputs() {
        let (energy, weight, broadening) = plset_reference_inputs();

        assert!(matches!(
            sfconv_select_pole(0, energy.view(), weight.view(), broadening.view()),
            Err(SfconvError::IndexOutOfRange {
                field: "pole",
                index: 0,
                len: 5,
            })
        ));
        assert!(matches!(
            sfconv_select_pole(6, energy.view(), weight.view(), broadening.view()),
            Err(SfconvError::IndexOutOfRange {
                field: "pole",
                index: 6,
                len: 5,
            })
        ));

        let short_weight = Array1::from_iter(weight.iter().copied().take(4));
        assert!(matches!(
            sfconv_select_pole(1, energy.view(), short_weight.view(), broadening.view()),
            Err(SfconvError::LengthMismatch {
                left: "energy",
                right: "weight",
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[2] = f64::NAN;
        assert!(matches!(
            sfconv_select_pole(3, bad_energy.view(), weight.view(), broadening.view()),
            Err(SfconvError::NonFiniteValue {
                field: "energy",
                row: 2,
                ..
            })
        ));
    }

    #[test]
    fn plasma_parameters_match_feff_ppset_reference() -> Result<(), SfconvError> {
        let first = sfconv_plasma_parameters(2.35)?;
        assert_close(first.fermi_momentum, 0.816_663_103_267_026_7, 1.0e-15);
        assert_close(first.fermi_energy, 0.333_469_312_118_865_2, 1.0e-15);
        assert_close(first.plasma_frequency, 0.480_793_772_651_942_2, 1.0e-15);

        let second = sfconv_plasma_parameters(0.95)?;
        assert_close(second.fermi_momentum, 2.020_166_623_871_066, 1.0e-15);
        assert_close(second.fermi_energy, 2.040_536_594_101_310_7, 1.0e-15);
        assert_close(second.plasma_frequency, 1.870_575_403_449_765_5, 1.0e-15);
        Ok(())
    }

    #[test]
    fn plasma_parameters_reject_invalid_radius() {
        assert_eq!(
            sfconv_plasma_parameters(0.0),
            Err(SfconvError::NonPositiveScalar {
                field: "wigner_seitz_radius",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_plasma_parameters(f64::NAN),
            Err(SfconvError::NonFiniteScalar {
                field: "wigner_seitz_radius",
                ..
            })
        ));
    }

    #[test]
    fn so2conv_material_parameters_match_feff_reference() -> Result<(), SfconvError> {
        assert_so2conv_material_close(
            sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
                core_hole_width_ev: 1.729,
                wigner_seitz_radius: 2.05,
                interstitial_potential_ev: 12.34,
                chemical_potential_ev: 18.76,
                fermi_wave_number_inv_angstrom: 1.23,
            })?,
            SfconvSo2convMaterialParameters {
                core_hole_lifetime: 0.031_769_539_461_112_17,
                interstitial_potential: 0.453_483_073_395_169_7,
                chemical_potential_offset: 0.235_928_795_072_689_63,
                fermi_wave_number: 0.650_887_783_8,
                fermi_momentum: 0.936_174_776_915_860,
                fermi_energy: 0.438_211_606_466_730_13,
                electron_concentration: 0.027_710_847_450_018_78,
                plasma_frequency: 0.590_105_735_521_106_2,
                dispersion_parameter: 0.292_141_070_977_820_1,
                initial_photoelectron_energy: 0.438_211_606_466_730_13,
                initial_photoelectron_momentum: 0.936_174_776_915_860,
                accuracy: 1.0e-4,
            },
            1.0e-15,
        );

        assert_so2conv_material_close(
            sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
                core_hole_width_ev: 5.533,
                wigner_seitz_radius: 1.42,
                interstitial_potential_ev: -3.25,
                chemical_potential_ev: 0.80,
                fermi_wave_number_inv_angstrom: 0.78,
            })?,
            SfconvSo2convMaterialParameters {
                core_hole_lifetime: 0.101_666_201_178_909,
                interstitial_potential: -0.119_434_358_876_361_54,
                chemical_potential_offset: 0.148_833_585_676_696_7,
                fermi_wave_number: 0.412_758_106_8,
                fermi_momentum: 1.351_519_924_420_783_8,
                fermi_energy: 0.913_303_053_053_180_6,
                electron_concentration: 0.083_377_017_833_289_21,
                plasma_frequency: 1.023_594_893_897_554_8,
                dispersion_parameter: 0.608_868_702_035_453_7,
                initial_photoelectron_energy: 0.913_303_053_053_180_6,
                initial_photoelectron_momentum: 1.351_519_924_420_783_8,
                accuracy: 1.0e-4,
            },
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn so2conv_material_parameters_reject_invalid_inputs() {
        let valid = SfconvSo2convMaterialInput {
            core_hole_width_ev: 1.729,
            wigner_seitz_radius: 2.05,
            interstitial_potential_ev: 12.34,
            chemical_potential_ev: 18.76,
            fermi_wave_number_inv_angstrom: 1.23,
        };

        assert!(matches!(
            sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
                core_hole_width_ev: f64::NAN,
                ..valid
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "core_hole_width_ev",
                ..
            })
        ));
        assert_eq!(
            sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
                wigner_seitz_radius: 0.0,
                ..valid
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "wigner_seitz_radius",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
                fermi_wave_number_inv_angstrom: f64::NAN,
                ..valid
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "fermi_wave_number_inv_angstrom",
                ..
            })
        ));
    }

    #[test]
    fn pole_dispersion_helpers_match_feff_ppole_reference() -> Result<(), SfconvError> {
        let pole_energy = 0.47;
        let dispersion_parameter = 0.28;
        let plasma_frequency = 0.62;

        assert_close(
            sfconv_pole_dispersion(0.35, pole_energy, dispersion_parameter)?,
            0.508_872_835_293_848_2,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_derivative(0.35, pole_energy, dispersion_parameter)?,
            0.234_709_915_161_871_29,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_second_derivative(0.35, pole_energy, dispersion_parameter)?,
            0.803_071_469_689_919_9,
            1.0e-15,
        );
        assert_close(
            sfconv_inverse_pole_dispersion(0.80, pole_energy, dispersion_parameter)?,
            0.922_319_683_172_048_9,
            1.0e-15,
        );
        assert_close(
            sfconv_coupling_potential_squared(
                0.35,
                plasma_frequency,
                pole_energy,
                dispersion_parameter,
            )?,
            38.745_198_544_546_376,
            1.0e-14,
        );

        assert_close(
            sfconv_pole_dispersion(1.70, pole_energy, dispersion_parameter)?,
            1.765_821_338_641_030_2,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_derivative(1.70, pole_energy, dispersion_parameter)?,
            1.660_700_284_807_318_7,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_second_derivative(1.70, pole_energy, dispersion_parameter)?,
            1.051_677_496_133_378_6,
            1.0e-15,
        );
        assert_close(
            sfconv_inverse_pole_dispersion(0.30, pole_energy, dispersion_parameter)?,
            0.0,
            0.0,
        );
        assert_close(
            sfconv_coupling_potential_squared(
                1.70,
                plasma_frequency,
                pole_energy,
                dispersion_parameter,
            )?,
            0.473_280_535_773_200_1,
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn pole_dispersion_helpers_reject_invalid_inputs() {
        assert!(matches!(
            sfconv_pole_dispersion(f64::NAN, 0.47, 0.28),
            Err(SfconvError::NonFiniteScalar {
                field: "momentum",
                ..
            })
        ));
        assert_eq!(
            sfconv_pole_dispersion(0.35, 0.0, 0.28),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_energy",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_coupling_potential_squared(0.0, 0.62, 0.47, 0.28),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_pole_dispersion(1.0, 0.47, -10.0),
            Err(SfconvError::NegativeRadicand {
                field: "pole_dispersion",
                ..
            })
        ));
    }

    #[test]
    fn q_limits_match_feff_qlimits_reference() -> Result<(), SfconvError> {
        assert_q_limits_close(
            sfconv_q_limits(1.15, 1.05, 0.47, 0.28, 12.0)?,
            SfconvQLimits {
                count: 3,
                q1: 0.112_905_963_336_969_05,
                q2: 1.252_615_998_981_518,
                q3: 0.926_614_797_549_310_8,
            },
            1.0e-14,
        );
        assert_q_limits_close(
            sfconv_q_limits(0.55, 0.92, 0.47, 0.28, 3.0)?,
            SfconvQLimits {
                count: 1,
                q1: 0.0,
                q2: 0.0,
                q3: 0.590_402_885_211_133_4,
            },
            1.0e-14,
        );
        assert_q_limits_close(
            sfconv_q_limits(2.40, 0.60, 0.47, 0.28, 0.75)?,
            SfconvQLimits {
                count: 3,
                q1: 0.75,
                q2: 0.75,
                q3: 4.179_832_657_474_71,
            },
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn q_limits_reject_invalid_inputs() {
        assert!(matches!(
            sfconv_q_limits(1.15, f64::NAN, 0.47, 0.28, 12.0),
            Err(SfconvError::NonFiniteScalar {
                field: "photoelectron_momentum",
                ..
            })
        ));
        assert_eq!(
            sfconv_q_limits(1.15, 0.0, 0.47, 0.28, 12.0),
            Err(SfconvError::NonPositiveScalar {
                field: "photoelectron_momentum",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_q_limits(1.15, 1.05, 0.47, 0.28, 0.0),
            Err(SfconvError::NonPositiveScalar {
                field: "upper_limit",
                value: 0.0,
            })
        );
    }

    #[test]
    fn plasmon_threshold_momentum_matches_feff_qthresh_reference() -> Result<(), SfconvError> {
        assert_close(
            sfconv_plasmon_threshold_momentum(0.47, 0.28, 0.42, 0.88)?,
            0.972_154_268_542_323_2,
            1.0e-14,
        );
        assert_close(
            sfconv_plasmon_threshold_momentum(0.75, 0.31, 0.55, 1.05)?,
            1.230_338_193_805_480_7,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn plasmon_threshold_momentum_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_plasmon_threshold_momentum(0.0, 0.28, 0.42, 0.88),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_energy",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_plasmon_threshold_momentum(0.47, 0.28, 0.0, 0.88),
            Err(SfconvError::NonPositiveScalar {
                field: "fermi_energy",
                value: 0.0,
            })
        );
    }

    #[test]
    fn so2conv_momentum_grid_matches_feff_reference() -> Result<(), SfconvError> {
        let grid = sfconv_so2conv_momentum_grid(0.816_663_103_267_026_7, 1.733_25)?;
        assert_eq!(grid.len(), SFCONV_SO2CONV_MOMENTUM_GRID_LEN);

        let expected = [
            (0, 0.908_321_792_940_324),
            (4, 1.274_956_551_633_513_3),
            (9, 1.733_25),
            (10, 1.747_693_75),
            (39, 2.166_562_5),
            (40, 2.296_556_25),
            (49, 3.466_5),
            (50, 3.813_15),
            (59, 6.933),
            (60, 8.666_25),
            (61, 12.132_75),
            (62, 17.332_5),
            (63, 51.997_5),
            (64, 173.325),
            (65, 519.975),
        ];
        for (index, expected) in expected {
            assert_close(grid[index], expected, 1.0e-15);
        }
        assert_close(grid.sum(), 937.896_733_964_701_5, 1.0e-15);
        Ok(())
    }

    #[test]
    fn so2conv_momentum_grid_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_so2conv_momentum_grid(0.0, 1.73),
            Err(SfconvError::NonPositiveScalar {
                field: "fermi_momentum",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_so2conv_momentum_grid(0.82, 0.0),
            Err(SfconvError::NonPositiveScalar {
                field: "threshold_momentum",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_so2conv_momentum_grid(0.82, 0.82),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: 0.82,
                upper: 0.82,
            })
        );
        assert!(matches!(
            sfconv_so2conv_momentum_grid(f64::NAN, 1.73),
            Err(SfconvError::NonFiniteScalar {
                field: "fermi_momentum",
                ..
            })
        ));
    }

    #[test]
    fn so2conv_momentum_spectral_interpolation_matches_feff_reference() -> Result<(), SfconvError> {
        let inputs = so2conv_momentum_spectral_inputs();

        let below = sfconv_interpolate_momentum_spectral_function(
            so2conv_momentum_spectral_input(&inputs, 0.25),
        )?;
        assert_momentum_spectral_close(
            &below,
            &[0.41, 0.42, 0.43, 0.44],
            &[
                [1.11, 1.12, 1.13, 1.14],
                [2.22, 2.24, 2.26, 2.28],
                [3.33, 3.36, 3.39, 3.42],
                [0.444, 0.448, 0.452, 0.456],
                [0.555, 0.560, 0.565, 0.570],
                [1.887, 1.904, 1.921, 1.938],
                [1.554, 1.568, 1.582, 1.596],
                [0.666, 0.672, 0.678, 0.684],
            ],
            &[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
            &[41.0, 51.0, 61.0, 71.0, 81.0],
        );

        let interior = sfconv_interpolate_momentum_spectral_function(
            so2conv_momentum_spectral_input(&inputs, 0.75),
        )?;
        assert_momentum_spectral_close(
            &interior,
            &[0.16, 0.17, 0.18, 0.19],
            &[
                [1.16, 1.17, 1.18, 1.19],
                [2.32, 2.34, 2.36, 2.38],
                [3.48, 3.51, 3.54, 3.57],
                [0.464, 0.468, 0.472, 0.476],
                [0.580, 0.585, 0.590, 0.595],
                [1.972, 1.989, 2.006, 2.023],
                [1.624, 1.638, 1.652, 1.666],
                [0.696, 0.702, 0.708, 0.714],
            ],
            &[0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23],
            &[41.5, 51.5, 61.5, 71.5, 81.5],
        );

        let exact = sfconv_interpolate_momentum_spectral_function(
            so2conv_momentum_spectral_input(&inputs, 2.0),
        )?;
        assert_momentum_spectral_close(
            &exact,
            &[0.31, 0.32, 0.33, 0.34],
            &[
                [1.31, 1.32, 1.33, 1.34],
                [2.62, 2.64, 2.66, 2.68],
                [3.93, 3.96, 3.99, 4.02],
                [0.524, 0.528, 0.532, 0.536],
                [0.655, 0.660, 0.665, 0.670],
                [2.227, 2.244, 2.261, 2.278],
                [1.834, 1.848, 1.862, 1.876],
                [0.786, 0.792, 0.798, 0.804],
            ],
            &[0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
            &[43.0, 53.0, 63.0, 73.0, 83.0],
        );

        let above = sfconv_interpolate_momentum_spectral_function(
            so2conv_momentum_spectral_input(&inputs, 4.5),
        )?;
        assert_momentum_spectral_close(
            &above,
            &[0.41, 0.42, 0.43, 0.44],
            &[
                [1.41, 1.42, 1.43, 1.44],
                [2.82, 2.84, 2.86, 2.88],
                [4.23, 4.26, 4.29, 4.32],
                [0.564, 0.568, 0.572, 0.576],
                [0.705, 0.710, 0.715, 0.720],
                [2.397, 2.414, 2.431, 2.448],
                [1.974, 1.988, 2.002, 2.016],
                [0.846, 0.852, 0.858, 0.864],
            ],
            &[0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
            &[44.0, 54.0, 64.0, 74.0, 84.0],
        );
        Ok(())
    }

    #[test]
    fn so2conv_momentum_spectral_interpolation_rejects_invalid_inputs() {
        let inputs = so2conv_momentum_spectral_inputs();
        let input = so2conv_momentum_spectral_input(&inputs, 0.75);

        assert_eq!(
            sfconv_interpolate_momentum_spectral_function(
                SfconvMomentumSpectralInterpolationInput {
                    momentum_grid: array![0.50].view(),
                    energy_grid: array![[0.11, 0.12, 0.13, 0.14]].view(),
                    extrinsic_quasiparticle: array![[1.11, 1.12, 1.13, 1.14]].view(),
                    extrinsic_satellite: array![[2.22, 2.24, 2.26, 2.28]].view(),
                    interference_quasiparticle: array![[3.33, 3.36, 3.39, 3.42]].view(),
                    interference_satellite: array![[0.444, 0.448, 0.452, 0.456]].view(),
                    intrinsic_satellite: array![[0.555, 0.560, 0.565, 0.570]].view(),
                    clipped_extrinsic_satellite: array![[0.666, 0.672, 0.678, 0.684]].view(),
                    weights: array![[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]].view(),
                    self_energy_real: array![41.0].view(),
                    energy_correction: array![51.0].view(),
                    width: array![61.0].view(),
                    renormalization_real: array![71.0].view(),
                    renormalization_imag: array![81.0].view(),
                    ..input
                },
            ),
            Err(SfconvError::CountTooSmall {
                name: "momentum_grid",
                actual: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            sfconv_interpolate_momentum_spectral_function(
                SfconvMomentumSpectralInterpolationInput {
                    energy_grid: array![[0.11, 0.12, 0.13, 0.14]].view(),
                    ..input
                },
            ),
            Err(SfconvError::CountMismatch {
                field: "energy_grid",
                actual: 1,
                expected: 4,
            })
        );
        assert_eq!(
            sfconv_interpolate_momentum_spectral_function(
                SfconvMomentumSpectralInterpolationInput {
                    weights: array![
                        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                        [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
                        [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37],
                        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47],
                    ]
                    .view(),
                    ..input
                },
            ),
            Err(SfconvError::CountMismatch {
                field: "weights",
                actual: 7,
                expected: 8,
            })
        );
        assert_eq!(
            sfconv_interpolate_momentum_spectral_function(
                SfconvMomentumSpectralInterpolationInput {
                    self_energy_real: array![41.0, 42.0].view(),
                    ..input
                },
            ),
            Err(SfconvError::LengthMismatch {
                left: "momentum_grid",
                left_len: 4,
                right: "self_energy_real",
                right_len: 2,
            })
        );
        assert_eq!(
            sfconv_interpolate_momentum_spectral_function(
                SfconvMomentumSpectralInterpolationInput {
                    momentum_grid: array![0.50, 1.00, 0.75, 4.00].view(),
                    ..input
                },
            ),
            Err(SfconvError::NonIncreasingEnergy {
                field: "momentum_grid",
                row: 2,
                previous: 1.00,
                current: 0.75,
            })
        );
        assert!(matches!(
            sfconv_interpolate_momentum_spectral_function(
                SfconvMomentumSpectralInterpolationInput {
                    intrinsic_satellite: array![
                        [0.555, 0.560, 0.565, 0.570],
                        [0.605, f64::NAN, 0.615, 0.620],
                        [0.655, 0.660, 0.665, 0.670],
                        [0.705, 0.710, 0.715, 0.720],
                    ]
                    .view(),
                    ..input
                },
            ),
            Err(SfconvError::NonFiniteValue {
                field: "intrinsic_satellite",
                row: 5,
                ..
            })
        ));
    }

    #[test]
    fn so2conv_photoelectron_momentum_matches_feff_reference() -> Result<(), SfconvError> {
        let (momentum, self_energy) = so2conv_photoelectron_momentum_inputs();

        let output = sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            momentum: momentum.view(),
            chemical_potential: 0.47,
            fermi_momentum: 0.92,
            fermi_level: 0.36,
            fermi_self_energy: 0.115,
            self_energy: self_energy.view(),
        })?;

        assert_real_slice_close(
            &output.kinetic_energy,
            &[
                0.47,
                0.531_25,
                0.389_999_999_999_999_96,
                0.806_199_999_999_999_9,
                1.075_000_000_000_000_2,
                1.521_25,
            ],
            1.0e-15,
        );
        assert_real_slice_close(
            &output.zero_order_momentum,
            &[
                1.032_666_451_474_047,
                1.090_366_910_723_174_8,
                0.952_050_418_832_952_4,
                1.318_635_658_550_154_6,
                1.508_774_337_003_384,
                1.780_140_443_897_615_6,
            ],
            1.0e-15,
        );
        assert_real_slice_close(
            &output.renormalization,
            &[
                0.803_278_688_524_59,
                1.600_000_000_000_000_5,
                0.859_353_023_909_986,
                0.907_284_768_211_920_6,
                0.877_308_140_604_871,
                0.881_481_481_481_481_3,
            ],
            1.0e-15,
        );
        assert_real_slice_close(
            &output.photoelectron_momentum,
            &[
                1.051_933_426_803_345_6,
                1.104_943_437_466_371,
                0.947_526_500_822_483_8,
                1.294_329_968_062_690_5,
                1.464_514_861_279_758_5,
                1.711_987_149_484_481_4,
            ],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn so2conv_photoelectron_momentum_rejects_invalid_inputs() {
        let (momentum, self_energy) = so2conv_photoelectron_momentum_inputs();
        let input = SfconvPhotoelectronMomentumInput {
            momentum: momentum.view(),
            chemical_potential: 0.47,
            fermi_momentum: 0.92,
            fermi_level: 0.36,
            fermi_self_energy: 0.115,
            self_energy: self_energy.view(),
        };

        assert_eq!(
            sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
                momentum: array![0.0].view(),
                self_energy: array![0.09].view(),
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "momentum",
                actual: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
                self_energy: array![0.09, 0.105].view(),
                ..input
            }),
            Err(SfconvError::LengthMismatch {
                left: "momentum",
                left_len: 6,
                right: "self_energy",
                right_len: 2,
            })
        );
        assert_eq!(
            sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
                fermi_momentum: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "fermi_momentum",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
                momentum: array![0.0, f64::NAN, 0.35, 0.82, 1.10, 1.45].view(),
                ..input
            }),
            Err(SfconvError::NonFiniteValue {
                field: "momentum",
                row: 1,
                ..
            })
        ));
        assert_eq!(
            sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
                momentum: array![0.0, 0.0].view(),
                self_energy: array![0.09, 0.105].view(),
                ..input
            }),
            Err(SfconvError::ZeroDenominator {
                field: "photoelectron momentum finite difference",
            })
        );
        assert!(matches!(
            sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
                self_energy: array![0.09, 0.105, 4.00, 0.150, 0.190, 0.250].view(),
                ..input
            }),
            Err(SfconvError::NegativeRadicand {
                field: "photoelectron momentum",
                ..
            })
        ));
    }

    #[test]
    fn so2conv_unbroadened_self_energy_sample_matches_weighted_poles() -> Result<(), SfconvError> {
        let material = so2conv_self_energy_material();
        let pole_energy = array![0.35, 0.57];
        let pole_weight = array![0.30, 0.70];
        let pole_broadening = array![0.01, 0.02];
        let input = SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: 0.85,
            photoelectron_momentum: 1.15,
            pole_count: 2,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        };

        let actual = sfconv_so2conv_unbroadened_self_energy_sample(input)?;
        let expected_poles =
            pole_weight
                .iter()
                .enumerate()
                .try_fold(0.0, |accumulator, (index, &weight)| {
                    let context = SfconvSelfEnergyContext {
                        fermi_energy: material.fermi_energy,
                        fermi_momentum: material.fermi_momentum,
                        plasma_frequency: material.plasma_frequency,
                        pole_energy: pole_energy[index],
                        quasiparticle_energy: input.quasiparticle_energy,
                        photoelectron_momentum: input.photoelectron_momentum,
                        accuracy: material.accuracy,
                        pole_broadening: pole_broadening[index],
                        dispersion_parameter: material.dispersion_parameter,
                        include_below_fermi: input.include_below_fermi,
                    };
                    let value = sfconv_real_self_energy(input.energy, context)?.value;
                    Ok::<_, SfconvError>(accumulator + weight * value)
                })?;
        let expected = expected_poles
            + sfconv_free_electron_exchange(input.photoelectron_momentum, material.fermi_momentum)?;
        assert_close(actual, expected, 1.0e-12);
        Ok(())
    }

    #[test]
    fn so2conv_unbroadened_self_energy_grid_builds_momentum_inputs() -> Result<(), SfconvError> {
        let material = so2conv_self_energy_material();
        let pole_energy = array![0.42];
        let pole_weight = array![1.0];
        let pole_broadening = array![0.02];
        let momentum = array![0.25, 0.50];
        let input = SfconvSo2convSelfEnergyGridInput {
            momentum: momentum.view(),
            chemical_potential: 0.80,
            fermi_level: 0.45,
            material,
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        };

        let grid = sfconv_so2conv_unbroadened_self_energy_grid(input)?;
        assert_real_slice_close(&grid.kinetic_energy, &[0.831_25, 0.925], 1.0e-15);
        assert_real_slice_close(
            &grid.zero_order_momentum,
            &[
                (material.fermi_momentum.powi(2) + 2.0 * (0.831_25 - input.fermi_level)).sqrt(),
                (material.fermi_momentum.powi(2) + 2.0 * (0.925 - input.fermi_level)).sqrt(),
            ],
            1.0e-15,
        );

        let expected_fermi =
            sfconv_so2conv_unbroadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                material,
                energy: 0.0,
                quasiparticle_energy: material.fermi_energy,
                photoelectron_momentum: material.fermi_momentum,
                pole_count: input.pole_count,
                pole_energy: input.pole_energy,
                pole_weight: input.pole_weight,
                pole_broadening: input.pole_broadening,
                include_below_fermi: input.include_below_fermi,
            })?;
        assert_close(grid.fermi_self_energy, expected_fermi, 1.0e-12);

        for row in 0..momentum.len() {
            let expected = sfconv_so2conv_unbroadened_self_energy_sample(
                SfconvSo2convSelfEnergySampleInput {
                    material,
                    energy: 0.0,
                    quasiparticle_energy: grid.kinetic_energy[row],
                    photoelectron_momentum: grid.zero_order_momentum[row],
                    pole_count: input.pole_count,
                    pole_energy: input.pole_energy,
                    pole_weight: input.pole_weight,
                    pole_broadening: input.pole_broadening,
                    include_below_fermi: input.include_below_fermi,
                },
            )?;
            assert_close(grid.self_energy[row], expected, 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn so2conv_broadened_self_energy_sample_matches_weighted_poles() -> Result<(), SfconvError> {
        let material = so2conv_self_energy_material();
        let pole_energy = array![0.35, 0.57];
        let pole_weight = array![0.30, 0.70];
        let pole_broadening = array![0.01, 0.02];
        let input = SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: 0.85,
            photoelectron_momentum: 1.15,
            pole_count: 2,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        };

        let actual = sfconv_so2conv_broadened_self_energy_sample(input)?;
        let expected_poles =
            pole_weight
                .iter()
                .enumerate()
                .try_fold(0.0, |accumulator, (index, &weight)| {
                    let context = SfconvSelfEnergyContext {
                        fermi_energy: material.fermi_energy,
                        fermi_momentum: material.fermi_momentum,
                        plasma_frequency: material.plasma_frequency,
                        pole_energy: pole_energy[index],
                        quasiparticle_energy: input.quasiparticle_energy,
                        photoelectron_momentum: input.photoelectron_momentum,
                        accuracy: material.accuracy,
                        pole_broadening: pole_broadening[index],
                        dispersion_parameter: material.dispersion_parameter,
                        include_below_fermi: input.include_below_fermi,
                    };
                    let value = sfconv_broadened_self_energy(input.energy, context)?.real;
                    Ok::<_, SfconvError>(accumulator + weight * value)
                })?;
        let expected = expected_poles
            + sfconv_free_electron_exchange(input.photoelectron_momentum, material.fermi_momentum)?;
        assert_close(actual, expected, 1.0e-12);
        Ok(())
    }

    #[test]
    fn so2conv_broadened_self_energy_grid_builds_momentum_inputs() -> Result<(), SfconvError> {
        let material = so2conv_self_energy_material();
        let pole_energy = array![0.42];
        let pole_weight = array![1.0];
        let pole_broadening = array![0.02];
        let momentum = array![0.25, 0.50];
        let input = SfconvSo2convSelfEnergyGridInput {
            momentum: momentum.view(),
            chemical_potential: 0.80,
            fermi_level: 0.45,
            material,
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        };

        let grid = sfconv_so2conv_broadened_self_energy_grid(input)?;
        assert_real_slice_close(&grid.kinetic_energy, &[0.831_25, 0.925], 1.0e-15);
        assert_real_slice_close(
            &grid.zero_order_momentum,
            &[
                (material.fermi_momentum.powi(2) + 2.0 * (0.831_25 - input.fermi_level)).sqrt(),
                (material.fermi_momentum.powi(2) + 2.0 * (0.925 - input.fermi_level)).sqrt(),
            ],
            1.0e-15,
        );

        let expected_fermi =
            sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                material,
                energy: 0.0,
                quasiparticle_energy: material.fermi_energy,
                photoelectron_momentum: material.fermi_momentum,
                pole_count: input.pole_count,
                pole_energy: input.pole_energy,
                pole_weight: input.pole_weight,
                pole_broadening: input.pole_broadening,
                include_below_fermi: input.include_below_fermi,
            })?;
        assert_close(grid.fermi_self_energy, expected_fermi, 1.0e-12);

        for row in 0..momentum.len() {
            let expected =
                sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                    material,
                    energy: 0.0,
                    quasiparticle_energy: grid.kinetic_energy[row],
                    photoelectron_momentum: grid.zero_order_momentum[row],
                    pole_count: input.pole_count,
                    pole_energy: input.pole_energy,
                    pole_weight: input.pole_weight,
                    pole_broadening: input.pole_broadening,
                    include_below_fermi: input.include_below_fermi,
                })?;
            assert_close(grid.self_energy[row], expected, 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn so2conv_unbroadened_self_energy_rejects_invalid_inputs() {
        let material = so2conv_self_energy_material();
        let pole_energy = array![0.42];
        let pole_weight = array![1.0];
        let pole_broadening = array![0.02];

        assert_eq!(
            sfconv_so2conv_unbroadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                material,
                energy: 0.0,
                quasiparticle_energy: 0.85,
                photoelectron_momentum: 1.15,
                pole_count: 0,
                pole_energy: pole_energy.view(),
                pole_weight: pole_weight.view(),
                pole_broadening: pole_broadening.view(),
                include_below_fermi: false,
            }),
            Err(SfconvError::CountTooSmall {
                name: "pole_count",
                actual: 0,
                minimum: 1,
            })
        );
        assert_eq!(
            sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                material,
                energy: 0.0,
                quasiparticle_energy: 0.85,
                photoelectron_momentum: 1.15,
                pole_count: 0,
                pole_energy: pole_energy.view(),
                pole_weight: pole_weight.view(),
                pole_broadening: pole_broadening.view(),
                include_below_fermi: false,
            }),
            Err(SfconvError::CountTooSmall {
                name: "pole_count",
                actual: 0,
                minimum: 1,
            })
        );
        assert_eq!(
            sfconv_so2conv_unbroadened_self_energy_grid(SfconvSo2convSelfEnergyGridInput {
                momentum: array![0.25].view(),
                chemical_potential: 0.80,
                fermi_level: 0.45,
                material,
                pole_count: 2,
                pole_energy: pole_energy.view(),
                pole_weight: pole_weight.view(),
                pole_broadening: pole_broadening.view(),
                include_below_fermi: false,
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "pole_energy",
                active_len: 2,
                len: 1,
            })
        );
        assert_eq!(
            sfconv_so2conv_broadened_self_energy_grid(SfconvSo2convSelfEnergyGridInput {
                momentum: array![0.25].view(),
                chemical_potential: 0.80,
                fermi_level: 0.45,
                material,
                pole_count: 2,
                pole_energy: pole_energy.view(),
                pole_weight: pole_weight.view(),
                pole_broadening: pole_broadening.view(),
                include_below_fermi: false,
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "pole_energy",
                active_len: 2,
                len: 1,
            })
        );
    }

    #[test]
    fn brsigma_broadened_integrands_match_feff_formulas() -> Result<(), SfconvError> {
        let input = SfconvBroadenedSelfEnergyIntegrandInput {
            momentum: 0.73,
            energy: 0.21,
            context: senergies_reference_context(false),
        };
        let expected = [
            (
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                SfconvBroadenedSelfEnergyIntegrands {
                    log_real: -7.011_705_793_259_941,
                    log_imag: -0.369_504_267_018_922_97,
                    atan_real: 0.325_185_453_673_107_86,
                    atan_imag: 6.170_712_852_111_19,
                },
            ),
            (
                SfconvBroadenedSelfEnergyBranch::ParticleFermi,
                SfconvBroadenedSelfEnergyIntegrands {
                    log_real: -13.797_429_315_272_487,
                    log_imag: -0.727_099_675_343_745_2,
                    atan_real: 0.070_287_942_851_676_6,
                    atan_imag: 1.333_782_638_196_666_4,
                },
            ),
            (
                SfconvBroadenedSelfEnergyBranch::HoleFermi,
                SfconvBroadenedSelfEnergyIntegrands {
                    log_real: 0.953_851_591_976_764_5,
                    log_imag: 0.050_266_260_982_741_846,
                    atan_real: 0.000_611_333_609_869_711_8,
                    atan_imag: 0.011_600_654_705_615_22,
                },
            ),
            (
                SfconvBroadenedSelfEnergyBranch::HolePair,
                SfconvBroadenedSelfEnergyIntegrands {
                    log_real: 7.129_828_634_229_525,
                    log_imag: 0.375_729_127_995_351,
                    atan_real: 0.324_874_938_869_935_85,
                    atan_imag: 6.164_820_529_238_007,
                },
            ),
        ];

        for (branch, expected_integrands) in expected {
            let actual = sfconv_broadened_self_energy_integrands(branch, input)?;
            assert_close(actual.log_real, expected_integrands.log_real, 1.0e-13);
            assert_close(actual.log_imag, expected_integrands.log_imag, 1.0e-13);
            assert_close(actual.atan_real, expected_integrands.atan_real, 1.0e-13);
            assert_close(actual.atan_imag, expected_integrands.atan_imag, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn dbrsigma_broadened_derivative_integrands_match_feff_formulas() -> Result<(), SfconvError> {
        let input = SfconvBroadenedSelfEnergyIntegrandInput {
            momentum: 0.73,
            energy: 0.21,
            context: senergies_reference_context(false),
        };
        let expected = [
            (
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                SfconvBroadenedSelfEnergyDerivativeIntegrands {
                    log_real: 8.237_536_803_919_268,
                    log_imag: 0.434_103_353_523_399_8,
                    atan_real: 0.042_642_154_194_309_74,
                    atan_imag: 0.809_176_689_659_211_2,
                },
            ),
            (
                SfconvBroadenedSelfEnergyBranch::ParticleFermi,
                SfconvBroadenedSelfEnergyDerivativeIntegrands {
                    log_real: -27.330_804_124_143_9,
                    log_imag: -1.440_284_153_769_992_4,
                    atan_real: 1.193_324_638_256_228,
                    atan_imag: 22.644_505_154_990_576,
                },
            ),
            (
                SfconvBroadenedSelfEnergyBranch::HoleFermi,
                SfconvBroadenedSelfEnergyDerivativeIntegrands {
                    log_real: -0.331_054_683_886_457,
                    log_imag: -0.017_445_985_601_711,
                    atan_real: -0.000_853_016_831_865_627_2,
                    atan_imag: -0.016_186_830_831_466_846,
                },
            ),
            (
                SfconvBroadenedSelfEnergyBranch::HolePair,
                SfconvBroadenedSelfEnergyDerivativeIntegrands {
                    log_real: 8.400_819_219_599_3,
                    log_imag: 0.442_708_042_753_362_3,
                    atan_real: -0.044_853_381_319_184_02,
                    atan_imag: -0.851_136_892_627_315_1,
                },
            ),
        ];

        for (branch, expected_integrands) in expected {
            let actual = sfconv_broadened_self_energy_derivative_integrands(branch, input)?;
            assert_close(actual.log_real, expected_integrands.log_real, 1.0e-13);
            assert_close(actual.log_imag, expected_integrands.log_imag, 1.0e-13);
            assert_close(actual.atan_real, expected_integrands.atan_real, 1.0e-13);
            assert_close(actual.atan_imag, expected_integrands.atan_imag, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn brsigma_broadened_integrands_reject_invalid_inputs() {
        let input = SfconvBroadenedSelfEnergyIntegrandInput {
            momentum: -0.10,
            energy: 0.21,
            context: senergies_reference_context(false),
        };
        assert_eq!(
            sfconv_broadened_self_energy_integrands(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                input,
            ),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: -0.10,
                upper: 0.0,
            })
        );
        assert_eq!(
            sfconv_broadened_self_energy_derivative_integrands(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                input,
            ),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: -0.10,
                upper: 0.0,
            })
        );

        let zero_broadening = SfconvBroadenedSelfEnergyIntegrandInput {
            momentum: 0.73,
            energy: 0.21,
            context: SfconvSelfEnergyContext {
                pole_broadening: 0.0,
                ..senergies_reference_context(false)
            },
        };
        assert_eq!(
            sfconv_broadened_self_energy_integrands(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                zero_broadening,
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_broadening",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_broadened_self_energy_derivative_integrands(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                zero_broadening,
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_broadening",
                value: 0.0,
            })
        );
    }

    #[test]
    fn brsigma_broadened_self_energy_matches_feff_reference() -> Result<(), SfconvError> {
        let cases = [
            (
                0.36,
                senergies_reference_context(false),
                -0.518_548_796_704_916_7,
                -0.820_845_165_208_279_3,
            ),
            (
                -0.20,
                senergies_reference_context(true),
                -0.276_438_440_404_569,
                -0.012_356_840_692_487_325,
            ),
            (
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 0.82,
                    ..senergies_reference_context(false)
                },
                -0.090_781_303_269_171_75,
                -0.280_887_927_239_661_94,
            ),
            (
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 0.82,
                    include_below_fermi: true,
                    ..senergies_reference_context(false)
                },
                0.008_365_301_760_209_81,
                -0.284_132_323_784_229_7,
            ),
            (
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 1.0,
                    ..senergies_reference_context(false)
                },
                0.013_728_093_655_548_983,
                -0.412_629_377_510_605_5,
            ),
        ];

        for (energy, context, expected_real, expected_imaginary) in cases {
            let actual = sfconv_broadened_self_energy(energy, context)?;
            assert_close(actual.real, expected_real, 1.0e-12);
            assert_close(actual.imaginary, expected_imaginary, 1.0e-12);
            assert!(actual.real_estimated_error >= 0.0);
            assert!(actual.real_estimated_error < 1.0e-6);
            assert!(actual.imaginary_estimated_error >= 0.0);
            assert!(actual.imaginary_estimated_error < 1.0e-6);
            assert!(actual.evaluations > 0);
            assert!(actual.max_regions > 0);
        }
        Ok(())
    }

    #[test]
    fn brsigma_broadened_self_energy_rejects_invalid_inputs() {
        let context = senergies_reference_context(false);
        assert!(matches!(
            sfconv_broadened_self_energy(f64::NAN, context),
            Err(SfconvError::NonFiniteScalar {
                field: "self-energy energy",
                ..
            })
        ));
        assert_eq!(
            sfconv_broadened_self_energy(
                0.36,
                SfconvSelfEnergyContext {
                    pole_broadening: 0.0,
                    ..context
                },
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_broadening",
                value: 0.0,
            })
        );
    }

    #[test]
    fn dbrsigma_broadened_self_energy_derivative_matches_feff_reference() -> Result<(), SfconvError>
    {
        let cases = [
            (
                0.36,
                senergies_reference_context(false),
                2.953_632_555_240_584,
                -4.153_776_392_437_791,
            ),
            (
                -0.20,
                senergies_reference_context(true),
                -0.453_145_835_952_415_03,
                -0.046_313_231_462_640_74,
            ),
            (
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 0.82,
                    ..senergies_reference_context(false)
                },
                0.533_248_980_604_782_1,
                0.196_090_288_785_958_72,
            ),
            (
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 0.82,
                    include_below_fermi: true,
                    ..senergies_reference_context(false)
                },
                0.467_087_536_743_928_44,
                0.199_768_325_815_296_63,
            ),
            (
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 1.0,
                    ..senergies_reference_context(false)
                },
                0.462_197_179_911_945_2,
                0.647_423_140_545_274,
            ),
        ];

        for (energy, context, expected_real, expected_imaginary) in cases {
            let actual = sfconv_broadened_self_energy_derivative(energy, context)?;
            assert_close(actual.real, expected_real, 1.0e-12);
            assert_close(actual.imaginary, expected_imaginary, 1.0e-12);
            assert!(actual.real_estimated_error >= 0.0);
            assert!(actual.real_estimated_error < 1.0e-6);
            assert!(actual.imaginary_estimated_error >= 0.0);
            assert!(actual.imaginary_estimated_error < 1.0e-6);
            assert!(actual.evaluations > 0);
            assert!(actual.max_regions > 0);
        }
        Ok(())
    }

    #[test]
    fn dbrsigma_broadened_self_energy_derivative_rejects_invalid_inputs() {
        let context = senergies_reference_context(false);
        assert!(matches!(
            sfconv_broadened_self_energy_derivative(f64::NAN, context),
            Err(SfconvError::NonFiniteScalar {
                field: "self-energy energy",
                ..
            })
        ));
        assert_eq!(
            sfconv_broadened_self_energy_derivative(
                0.36,
                SfconvSelfEnergyContext {
                    pole_broadening: 0.0,
                    ..context
                },
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_broadening",
                value: 0.0,
            })
        );
    }

    #[test]
    fn so2conv_signal_preparation_matches_feff_reference() -> Result<(), SfconvError> {
        let exafs_energy = array![0.10, 0.22, 0.37, 0.55];
        let padded_exafs =
            sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
                energy: exafs_energy.view(),
                active_len: 4,
                output_len: 7,
            })?;
        assert_real_slice_close(
            &padded_exafs,
            &[0.10, 0.22, 0.37, 0.55, 0.73, 0.91, 1.09],
            1.0e-14,
        );
        let exafs_momentum = array![0.0, 0.1, 0.2, 0.3];
        let exafs_magnitude = array![1.0, 2.0, 3.0, 4.0];
        let exafs_phase = array![
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
        ];
        let exafs_phase_minus_2kr = array![0.1, 0.2, 0.3, 0.4];
        let prepared_exafs =
            sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
                momentum: exafs_momentum.view(),
                magnitude: exafs_magnitude.view(),
                phase: exafs_phase.view(),
                phase_minus_2kr: Some(exafs_phase_minus_2kr.view()),
                chemical_potential: 0.5,
                active_len: 4,
                output_len: 6,
            })?;
        assert_real_slice_close(
            &prepared_exafs.signal_energy,
            &[0.5, 0.505, 0.52, 0.545, 0.57, 0.595],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared_exafs.real_signal,
            &[1.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared_exafs.imaginary_signal,
            &[0.0, 2.0, 0.0, -4.0, 0.0, 0.0],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared_exafs.original_magnitude,
            &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared_exafs.original_phase,
            &[
                0.0,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
                -std::f64::consts::FRAC_PI_2,
                0.0,
                0.0,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared_exafs.phase_minus_2kr,
            &[0.1, 0.2, 0.3, 0.4, 0.0, 0.0],
            1.0e-14,
        );

        let prepared_exafs_default_phase =
            sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
                phase_minus_2kr: None,
                ..SfconvSo2convExafsPreparationInput {
                    momentum: exafs_momentum.view(),
                    magnitude: exafs_magnitude.view(),
                    phase: exafs_phase.view(),
                    phase_minus_2kr: Some(exafs_phase_minus_2kr.view()),
                    chemical_potential: 0.5,
                    active_len: 4,
                    output_len: 6,
                }
            })?;
        assert_real_slice_close(
            &prepared_exafs_default_phase.phase_minus_2kr,
            &[0.0; 6],
            1.0e-14,
        );

        let (incident_energy, excitation_energy, absorption, embedded_background) =
            so2conv_xanes_preparation_inputs();
        let prepared = sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
            incident_energy: incident_energy.view(),
            excitation_energy: excitation_energy.view(),
            absorption: absorption.view(),
            embedded_background: embedded_background.view(),
            active_len: 22,
            output_len: 25,
        })?;

        assert_real_slice_close(
            &prepared.incident_energy,
            &[
                0.202, 0.334, 0.460, 0.592, 0.724, 0.850, 0.982, 1.114, 1.240, 1.372, 1.504, 1.630,
                1.762, 1.894, 2.020, 2.152, 2.284, 2.410, 2.542, 2.674, 2.800, 2.911, 3.022, 3.133,
                3.244,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared.excitation_energy,
            &[
                -0.399, -0.288, -0.177, -0.070, 0.041, 0.152, 0.263, 0.370, 0.481, 0.592, 0.703,
                0.810, 0.921, 1.032, 1.143, 1.250, 1.361, 1.472, 1.583, 1.690, 1.801, 1.912, 2.023,
                2.134, 2.245,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared.absorption,
            &[
                1.013_002_345_457_738,
                1.040_241_406_421_492,
                1.066_864_797_635_351,
                1.088_831_359_977_982,
                1.108_791_350_567_574,
                1.123_338_851_323_157,
                1.135_831_399_724_224,
                1.143_574_970_312_228,
                1.150_575_738_690_336,
                1.154_663_226_497_332,
                1.160_192_154_165_129,
                1.165_132_358_117_229,
                1.173_757_266_916_467,
                1.183_741_568_249_87,
                1.198_877_822_449_645,
                1.216_219_972_021_848,
                1.238_859_132_580_952,
                1.263_133_973_109_753,
                1.291_474_695_964_75,
                1.319_676_423_887_3,
                1.349_794_997_826_861,
                1.315,
                1.315,
                1.315,
                1.315,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared.embedded_background,
            &[
                1.0008, 1.015, 1.0308, 1.045, 1.0608, 1.075, 1.0908, 1.105, 1.1208, 1.135, 1.1508,
                1.165, 1.1808, 1.195, 1.2108, 1.225, 1.2408, 1.255, 1.2708, 1.285, 1.3008, 1.315,
                1.315, 1.315, 1.315,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared.imaginary_fine_structure,
            &[
                0.012_202_345_457_738,
                0.025_241_406_421_492,
                0.036_064_797_635_351,
                0.043_831_359_977_982,
                0.047_991_350_567_574,
                0.048_338_851_323_157,
                0.045_031_399_724_224,
                0.038_574_970_312_228,
                0.029_775_738_690_336,
                0.019_663_226_497_332,
                0.009_392_154_165_129,
                0.000_132_358_117_229,
                -0.007_042_733_083_533,
                -0.011_258_431_750_130,
                -0.011_922_177_550_355,
                -0.008_780_027_978_152,
                -0.001_940_867_419_048,
                0.008_133_973_109_753,
                0.020_674_695_964_750,
                0.034_676_423_887_300,
                0.048_994_997_826_861,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &prepared.real_fine_structure,
            &[
                0.032_463_374_088_541,
                0.031_708_281_403_956,
                0.027_054_881_272_691,
                0.017_990_328_415_378,
                0.008_527_437_386_775,
                -0.002_125_087_125_751,
                -0.011_497_227_273_338,
                -0.020_683_431_261_378,
                -0.025_978_917_059_008,
                -0.029_016_022_387_064,
                -0.028_834_004_298_412,
                -0.025_910_106_145_618,
                -0.020_120_578_606_356,
                -0.012_652_748_322_213,
                -0.004_600_832_388_766,
                0.003_191_694_845_944,
                0.009_092_681_421_030,
                0.012_096_380_083_534,
                0.010_920_250_201_848,
                -0.009_338_141_883_948,
                -0.009_338_141_883_948,
                -0.029_208_468_871_716,
                -0.018_711_184_393_096,
                -0.014_581_157_747_772,
                -0.012_254_476_090_090,
            ],
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn so2conv_signal_preparation_rejects_invalid_inputs() {
        let exafs_energy = array![0.10, 0.22, 0.37, 0.55];
        assert_eq!(
            sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
                energy: exafs_energy.view(),
                active_len: 1,
                output_len: 7,
            }),
            Err(SfconvError::CountTooSmall {
                name: "active_len",
                actual: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
                energy: array![0.10, 0.22, 0.20].view(),
                active_len: 3,
                output_len: 5,
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "energy",
                row: 2,
                previous: 0.22,
                current: 0.20,
            })
        );
        let exafs_momentum = array![0.0, 0.1, 0.2, 0.3];
        let exafs_magnitude = array![1.0, 2.0, 3.0, 4.0];
        let exafs_phase = array![0.0, 0.1, 0.2, 0.3];
        let exafs_input = SfconvSo2convExafsPreparationInput {
            momentum: exafs_momentum.view(),
            magnitude: exafs_magnitude.view(),
            phase: exafs_phase.view(),
            phase_minus_2kr: None,
            chemical_potential: 0.5,
            active_len: 4,
            output_len: 6,
        };
        assert_eq!(
            sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
                active_len: 1,
                ..exafs_input
            }),
            Err(SfconvError::CountTooSmall {
                name: "active_len",
                actual: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
                momentum: array![0.3, 0.2, 0.1, 0.0].view(),
                ..exafs_input
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "energy",
                row: 1,
                previous: 0.545,
                current: 0.52,
            })
        );
        assert_eq!(
            sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
                magnitude: array![1.0, 0.0, 3.0, 4.0].view(),
                ..exafs_input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "magnitude",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
                phase: array![0.0, f64::NAN, 0.2, 0.3].view(),
                ..exafs_input
            }),
            Err(SfconvError::NonFiniteValue {
                field: "phase",
                row: 1,
                ..
            })
        ));

        let (incident_energy, excitation_energy, absorption, embedded_background) =
            so2conv_xanes_preparation_inputs();
        let input = SfconvSo2convXanesPreparationInput {
            incident_energy: incident_energy.view(),
            excitation_energy: excitation_energy.view(),
            absorption: absorption.view(),
            embedded_background: embedded_background.view(),
            active_len: 22,
            output_len: 25,
        };
        assert_eq!(
            sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
                output_len: 20,
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "output_len",
                actual: 20,
                minimum: 21,
            })
        );
        assert_eq!(
            sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
                output_len: 21,
                ..input
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "output_len",
                active_len: 22,
                len: 21,
            })
        );
        assert!(matches!(
            sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
                absorption: array![1.0, f64::NAN, 1.1, 1.2].view(),
                active_len: 4,
                output_len: 25,
                ..input
            }),
            Err(SfconvError::NonFiniteValue {
                field: "absorption",
                row: 1,
                ..
            })
        ));
        assert_eq!(
            sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
                excitation_energy: array![0.0, 0.2, 0.1, 0.4].view(),
                active_len: 4,
                output_len: 25,
                ..input
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "excitation_energy",
                row: 2,
                previous: 0.2,
                current: 0.1,
            })
        );
    }

    #[test]
    fn so2conv_feff_path_interpolation_matches_feff_reference() -> Result<(), SfconvError> {
        let inputs = so2conv_feff_path_interpolation_inputs();

        let interpolated = sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
            source_momentum: inputs.source_momentum.view(),
            path_momentum: inputs.path_momentum.view(),
            central_phase: inputs.central_phase.view(),
            effective_amplitude: inputs.effective_amplitude.view(),
            effective_phase: inputs.effective_phase.view(),
            reduction_factor: inputs.reduction_factor.view(),
            mean_free_path: inputs.mean_free_path.view(),
        })?;

        assert_real_slice_close(
            &interpolated.central_phase,
            &[0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &interpolated.effective_amplitude,
            &[0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &interpolated.effective_phase,
            &[0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &interpolated.reduction_factor,
            &[0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &interpolated.mean_free_path,
            &[0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn so2conv_feff_path_interpolation_rejects_invalid_inputs() {
        let inputs = so2conv_feff_path_interpolation_inputs();
        let input = SfconvFeffPathInterpolationInput {
            source_momentum: inputs.source_momentum.view(),
            path_momentum: inputs.path_momentum.view(),
            central_phase: inputs.central_phase.view(),
            effective_amplitude: inputs.effective_amplitude.view(),
            effective_phase: inputs.effective_phase.view(),
            reduction_factor: inputs.reduction_factor.view(),
            mean_free_path: inputs.mean_free_path.view(),
        };

        assert_eq!(
            sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
                path_momentum: array![0.25].view(),
                central_phase: array![0.10].view(),
                effective_amplitude: array![1.00].view(),
                effective_phase: array![0.50].view(),
                reduction_factor: array![0.80].view(),
                mean_free_path: array![6.00].view(),
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "path_momentum",
                actual: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
                central_phase: array![0.10, 0.20].view(),
                ..input
            }),
            Err(SfconvError::LengthMismatch {
                left: "path_momentum",
                left_len: 4,
                right: "central_phase",
                right_len: 2,
            })
        );
        assert_eq!(
            sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
                source_momentum: array![0.0, 0.50, 0.25].view(),
                ..input
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "source_momentum",
                row: 2,
                previous: 0.50,
                current: 0.25,
            })
        );
        assert_eq!(
            sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
                path_momentum: array![0.25, 0.75, 0.70, 1.75].view(),
                ..input
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "path_momentum",
                row: 2,
                previous: 0.75,
                current: 0.70,
            })
        );
        assert!(matches!(
            sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
                effective_phase: array![0.50, f64::NAN, 0.60, 1.00].view(),
                ..input
            }),
            Err(SfconvError::NonFiniteValue {
                field: "effective_phase",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn so2conv_feff_path_signal_matches_feff_reference() -> Result<(), SfconvError> {
        let inputs = so2conv_feff_path_interpolation_inputs();
        let signal = sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            momentum: inputs.source_momentum.view(),
            central_phase: inputs.interpolated_central_phase.view(),
            effective_amplitude: inputs.interpolated_effective_amplitude.view(),
            effective_phase: inputs.interpolated_effective_phase.view(),
            reduction_factor: inputs.interpolated_reduction_factor.view(),
            mean_free_path: inputs.interpolated_mean_free_path.view(),
            degeneracy: 4.0,
            half_path_length: 3.25,
        })?;

        assert_real_slice_close(
            &signal.magnitude,
            &[
                0.536_124_841_919_397_1,
                0.410_164_018_117_519_6,
                0.284_203_194_315_642_06,
                0.251_379_063_300_987_75,
                0.174_109_626_719_572_4,
                0.125_698_646_320_718_7,
                0.153_357_484_762_483_76,
                0.179_719_087_666_981_03,
                0.0,
            ],
            1.0e-15,
        );
        assert_real_slice_close(
            &signal.phase_minus_2kr,
            &[0.0, 0.60, 0.75, 0.90, 0.80, 0.70, 1.00, 1.30, 0.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &signal.phase,
            &[0.0, 2.225, 4.0, 5.775, 7.30, 8.825, 10.75, 12.675, 13.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &signal.real,
            &[
                0.536_124_841_919_397_1,
                -0.249_596_094_763_011_48,
                -0.185_767_604_993_480_97,
                0.219_612_030_110_783_8,
                0.091_595_160_176_783_6,
                -0.103_759_326_185_720_1,
                -0.037_283_262_995_958_43,
                0.178_659_756_510_624_74,
                0.0,
            ],
            1.0e-15,
        );
        assert_real_slice_close(
            &signal.imaginary,
            &[
                0.0,
                0.325_478_587_986_003_7,
                -0.215_085_686_632_561_95,
                -0.122_319_212_295_952_02,
                0.148_069_202_566_293_94,
                0.070_951_757_669_183,
                -0.148_756_433_249_287_2,
                0.019_484_400_822_614_257,
                0.0,
            ],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn so2conv_feff_path_signal_rejects_invalid_inputs() {
        let inputs = so2conv_feff_path_interpolation_inputs();
        let input = SfconvFeffPathSignalInput {
            momentum: inputs.source_momentum.view(),
            central_phase: inputs.interpolated_central_phase.view(),
            effective_amplitude: inputs.interpolated_effective_amplitude.view(),
            effective_phase: inputs.interpolated_effective_phase.view(),
            reduction_factor: inputs.interpolated_reduction_factor.view(),
            mean_free_path: inputs.interpolated_mean_free_path.view(),
            degeneracy: 4.0,
            half_path_length: 3.25,
        };

        assert_eq!(
            sfconv_feff_path_signal(SfconvFeffPathSignalInput {
                momentum: array![0.0, 0.25].view(),
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "momentum",
                actual: 2,
                minimum: 3,
            })
        );
        assert_eq!(
            sfconv_feff_path_signal(SfconvFeffPathSignalInput {
                central_phase: array![0.0, 0.10].view(),
                ..input
            }),
            Err(SfconvError::LengthMismatch {
                left: "momentum",
                left_len: 9,
                right: "central_phase",
                right_len: 2,
            })
        );
        assert_eq!(
            sfconv_feff_path_signal(SfconvFeffPathSignalInput {
                momentum: array![0.0, 0.50, 0.25, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0].view(),
                ..input
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "momentum",
                row: 2,
                previous: 0.50,
                current: 0.25,
            })
        );
        assert!(matches!(
            sfconv_feff_path_signal(SfconvFeffPathSignalInput {
                effective_amplitude: array![0.0, f64::NAN, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0]
                    .view(),
                ..input
            }),
            Err(SfconvError::NonFiniteValue {
                field: "effective_amplitude",
                row: 1,
                ..
            })
        ));
        assert_eq!(
            sfconv_feff_path_signal(SfconvFeffPathSignalInput {
                half_path_length: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "half_path_length",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_feff_path_signal(SfconvFeffPathSignalInput {
                mean_free_path: array![0.0, 0.0, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0].view(),
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "mean_free_path",
                value: 0.0,
            })
        );
    }

    #[test]
    fn so2conv_exafs_convolution_matches_feff_reference() -> Result<(), SfconvError> {
        let real_channel = [
            1.960_133_155_682_483_3,
            -1.493_739_884_954_432_7,
            -1.494_388_190_129_498_7,
            -1.942_505_586_276_729,
            -1.979_984_993_200_890_8,
        ];
        let imaginary_channel = [
            0.397_338_661_590_122_43,
            0.137_168_698_409_705_4,
            -0.137_577_673_742_690_1,
            -0.478_498_658_427_964_87,
            0.282_240_016_119_734_4,
        ];
        let original_magnitude = [2.4, 1.8, 1.7, 2.3, 2.6];
        let original_phase = [0.10, 0.20, 0.25, 0.30, 0.35];
        let phase_minus_2kr = [0.01, 0.02, 0.03, 0.04, 0.05];
        let expected = [
            (
                0,
                1.960_133_155_682_483_3,
                0.397_338_661_590_122_43,
                2.000_000_000_000_000_0,
                0.2,
                0.110_000_000_000_000_01,
                0.833_333_333_333_333_4,
                0.1,
                0.2,
            ),
            (
                0,
                -1.493_739_884_954_432_8,
                0.137_168_698_409_705_4,
                1.500_024_698_372_361_7,
                3.050_020_434_612_271,
                2.870_020_434_612_271,
                0.833_347_054_651_312_1,
                2.850_020_434_612_271,
                3.050_020_434_612_271,
            ),
            (
                -2,
                -1.494_388_190_129_498_6,
                -0.137_577_673_742_690_1,
                1.500_707_726_078_255_8,
                3.233_396_748_497_55,
                3.013_396_748_497_55,
                0.882_769_250_634_268_1,
                2.983_396_748_497_55,
                -3.049_788_558_682_036,
            ),
            (
                -2,
                -1.942_505_586_276_729,
                -0.478_498_658_427_964_85,
                2.000_572_147_870_119,
                3.383_114_837_790_301_5,
                3.123_114_837_790_301_7,
                0.869_813_977_334_834_3,
                3.083_114_837_790_301_7,
                -2.900_070_469_389_284_7,
            ),
            (
                0,
                -1.979_984_993_200_890_8,
                0.282_240_016_119_734_4,
                1.999_999_999_999_999_8,
                3.000_000_000_000_000_0,
                2.699_999_999_999_999_7,
                0.769_230_769_230_769_2,
                2.65,
                3.000_000_000_000_000_0,
            ),
        ];

        let mut previous_phase = 0.0;
        let mut phase_jump_count = 0;
        for row in 0..real_channel.len() {
            let actual = sfconv_exafs_convolution(SfconvExafsConvolutionInput {
                real_convolution_amplitude: real_channel[row],
                real_convolution_phase: 0.0,
                imaginary_convolution_amplitude: imaginary_channel[row],
                imaginary_convolution_phase: 0.0,
                original_magnitude: original_magnitude[row],
                original_phase: original_phase[row],
                phase_minus_2kr: phase_minus_2kr[row],
                previous_phase,
                phase_jump_count,
            })?;
            let expected_row = expected[row];

            assert_eq!(actual.phase_jump_count, expected_row.0);
            assert_close(actual.real, expected_row.1, 1.0e-15);
            assert_close(actual.imaginary, expected_row.2, 1.0e-15);
            assert_close(actual.magnitude, expected_row.3, 1.0e-15);
            assert_close(actual.output_phase, expected_row.4, 1.0e-15);
            assert_close(actual.output_phase_minus_original, expected_row.5, 1.0e-15);
            assert_close(actual.amplitude_reduction, expected_row.6, 1.0e-15);
            assert_close(actual.phase_shift, expected_row.7, 1.0e-15);
            assert_close(actual.previous_phase, expected_row.8, 1.0e-15);

            previous_phase = actual.previous_phase;
            phase_jump_count = actual.phase_jump_count;
        }

        Ok(())
    }

    #[test]
    fn so2conv_exafs_convolution_rejects_invalid_inputs() {
        let input = SfconvExafsConvolutionInput {
            real_convolution_amplitude: 1.0,
            real_convolution_phase: 0.0,
            imaginary_convolution_amplitude: 0.2,
            imaginary_convolution_phase: 0.0,
            original_magnitude: 2.0,
            original_phase: 0.1,
            phase_minus_2kr: 0.05,
            previous_phase: 0.0,
            phase_jump_count: 0,
        };

        assert_eq!(
            sfconv_exafs_convolution(SfconvExafsConvolutionInput {
                original_magnitude: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "original_magnitude",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_exafs_convolution(SfconvExafsConvolutionInput {
                real_convolution_phase: f64::NAN,
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "real_convolution_phase",
                ..
            })
        ));
        assert_eq!(
            sfconv_exafs_convolution(SfconvExafsConvolutionInput {
                real_convolution_amplitude: -1.0,
                imaginary_convolution_amplitude: 0.0,
                previous_phase: -3.0,
                phase_jump_count: i32::MAX,
                ..input
            }),
            Err(SfconvError::PhaseJumpOverflow {
                value: i32::MAX,
                delta: 2,
            })
        );
    }

    #[test]
    fn so2conv_xanes_convolution_matches_feff_reference() -> Result<(), SfconvError> {
        let inputs = [
            SfconvXanesConvolutionInput {
                asymmetric_phase: false,
                absorption_convolution: f64::NAN,
                embedded_background: 3.40,
                fine_structure_imaginary_amplitude: 1.80,
                fine_structure_imaginary_phase: 0.20,
                fine_structure_real_amplitude: 0.70,
                fine_structure_real_phase: 0.90,
            },
            SfconvXanesConvolutionInput {
                asymmetric_phase: false,
                absorption_convolution: f64::NAN,
                embedded_background: 2.10,
                fine_structure_imaginary_amplitude: -0.55,
                fine_structure_imaginary_phase: 2.40,
                fine_structure_real_amplitude: 1.25,
                fine_structure_real_phase: -0.35,
            },
            SfconvXanesConvolutionInput {
                asymmetric_phase: true,
                absorption_convolution: 5.25,
                embedded_background: 4.90,
                fine_structure_imaginary_amplitude: f64::NAN,
                fine_structure_imaginary_phase: f64::NAN,
                fine_structure_real_amplitude: f64::NAN,
                fine_structure_real_phase: f64::NAN,
            },
            SfconvXanesConvolutionInput {
                asymmetric_phase: true,
                absorption_convolution: -0.75,
                embedded_background: -1.10,
                fine_structure_imaginary_amplitude: f64::NAN,
                fine_structure_imaginary_phase: f64::NAN,
                fine_structure_real_amplitude: f64::NAN,
                fine_structure_real_phase: f64::NAN,
            },
        ];
        let expected = [
            (5.712_448_676_853_473, 3.40, 2.312_448_676_853_473),
            (2.076_944_284_228_370_7, 2.10, -0.023_055_715_771_629_348),
            (5.25, 4.90, 0.349_999_999_999_999_64),
            (-0.75, -1.10, 0.350_000_000_000_000_1),
        ];

        for (input, expected_row) in inputs.into_iter().zip(expected) {
            let actual = sfconv_xanes_convolution(input)?;
            assert_close(actual.absorption, expected_row.0, 1.0e-14);
            assert_close(actual.embedded_background, expected_row.1, 1.0e-14);
            assert_close(actual.fine_structure, expected_row.2, 1.0e-14);
        }

        Ok(())
    }

    #[test]
    fn so2conv_xanes_convolution_rejects_invalid_inputs() {
        let input = SfconvXanesConvolutionInput {
            asymmetric_phase: false,
            absorption_convolution: 0.0,
            embedded_background: 3.40,
            fine_structure_imaginary_amplitude: 1.80,
            fine_structure_imaginary_phase: 0.20,
            fine_structure_real_amplitude: 0.70,
            fine_structure_real_phase: 0.90,
        };

        assert!(matches!(
            sfconv_xanes_convolution(SfconvXanesConvolutionInput {
                embedded_background: f64::NAN,
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "embedded_background",
                ..
            })
        ));
        assert!(matches!(
            sfconv_xanes_convolution(SfconvXanesConvolutionInput {
                fine_structure_real_phase: f64::NAN,
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "fine_structure_real_phase",
                ..
            })
        ));
        assert!(matches!(
            sfconv_xanes_convolution(SfconvXanesConvolutionInput {
                asymmetric_phase: true,
                absorption_convolution: f64::NAN,
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "absorption_convolution",
                ..
            })
        ));
    }

    #[test]
    fn senergies_beta_helpers_match_feff_reference() -> Result<(), SfconvError> {
        let lowq0_context = senergies_reference_context(false);
        assert_close(
            sfconv_free_electron_exchange(1.0, lowq0_context.fermi_momentum)?,
            -std::f64::consts::FRAC_1_PI,
            1.0e-15,
        );
        assert_close(
            sfconv_free_electron_exchange(1.35, lowq0_context.fermi_momentum)?,
            -0.133_662_411_513_184_28,
            1.0e-15,
        );
        assert_close(
            sfconv_extrinsic_beta(0.36, lowq0_context)?,
            0.287_008_463_933_952_74,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_beta(0.95, lowq0_context)?,
            0.099_242_494_271_372_31,
            1.0e-14,
        );
        assert_close(
            sfconv_imaginary_self_energy(0.36, lowq0_context)?,
            -0.901_663_681_812_997,
            1.0e-14,
        );

        let lowq1_context = senergies_reference_context(true);
        assert_close(sfconv_extrinsic_beta(-0.20, lowq1_context)?, 0.0, 0.0);
        assert_close(
            sfconv_extrinsic_beta(0.36, lowq1_context)?,
            0.287_008_463_933_952_74,
            1.0e-14,
        );
        assert_close(
            sfconv_imaginary_self_energy(-0.20, lowq1_context)?,
            0.0,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn senergies_real_self_energy_matches_feff_reference() -> Result<(), SfconvError> {
        let pkgt_lowq0_context = senergies_reference_context(false);
        assert_close(
            sfconv_real_self_energy_integrand_upper(0.55, 0.36, pkgt_lowq0_context)?,
            2.874_639_111_469_788_7,
            1.0e-14,
        );
        assert_close(
            sfconv_real_self_energy_integrand_middle(0.55, 0.36, pkgt_lowq0_context)?,
            5.222_817_359_927_24,
            1.0e-14,
        );
        assert_close(
            sfconv_real_self_energy_integrand_lower(0.55, 0.36, pkgt_lowq0_context)?,
            -8.010_746_486_392_092,
            1.0e-14,
        );
        let real_pkgt = sfconv_real_self_energy(0.36, pkgt_lowq0_context)?;
        assert_close(real_pkgt.value, -0.707_783_970_737_988_9, 1.0e-12);
        assert!(real_pkgt.evaluations > 0);
        assert!(real_pkgt.max_regions > 0);
        assert_close(
            sfconv_real_self_energy(0.95, pkgt_lowq0_context)?.value,
            0.196_748_431_942_598_25,
            1.0e-12,
        );

        let pkgt_lowq1_context = senergies_reference_context(true);
        assert_close(
            sfconv_real_self_energy(-0.20, pkgt_lowq1_context)?.value,
            -0.277_039_230_882_649,
            1.0e-12,
        );

        let pklt_lowq0_context = SfconvSelfEnergyContext {
            photoelectron_momentum: 0.82,
            ..senergies_reference_context(false)
        };
        assert_close(
            sfconv_real_self_energy_integrand_upper(0.55, 0.36, pklt_lowq0_context)?,
            -3.190_158_193_028_965_5,
            1.0e-14,
        );
        assert_close(
            sfconv_real_self_energy_integrand_middle(0.55, 0.36, pklt_lowq0_context)?,
            0.649_162_805_914_428_2,
            1.0e-14,
        );
        assert_close(
            sfconv_real_self_energy_integrand_lower(0.55, 0.36, pklt_lowq0_context)?,
            -2.194_055_192_971_564_6,
            1.0e-14,
        );
        assert_close(
            sfconv_real_self_energy(0.36, pklt_lowq0_context)?.value,
            -0.077_377_126_607_744_2,
            1.0e-12,
        );

        let pklt_lowq1_context = SfconvSelfEnergyContext {
            include_below_fermi: true,
            ..pklt_lowq0_context
        };
        assert_close(
            sfconv_real_self_energy_integrand_middle(0.55, 0.36, pklt_lowq1_context)?,
            -0.291_337_926_232_215_6,
            1.0e-14,
        );
        assert_close(
            sfconv_real_self_energy(0.36, pklt_lowq1_context)?.value,
            0.021_796_867_569_840_478,
            1.0e-12,
        );

        let pkeq_context = SfconvSelfEnergyContext {
            photoelectron_momentum: 1.0,
            ..senergies_reference_context(false)
        };
        assert_close(
            sfconv_real_self_energy(0.36, pkeq_context)?.value,
            0.043_101_938_251_358_85,
            1.0e-12,
        );
        Ok(())
    }

    #[test]
    fn senergies_self_energy_derivatives_match_feff_reference() -> Result<(), SfconvError> {
        let pkgt_lowq0_context = senergies_reference_context(false);
        assert_close(
            sfconv_real_self_energy_derivative_integrand_upper(0.55, 0.36, pkgt_lowq0_context)?,
            10.732_547_867_812_46,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pkgt_lowq0_context)?,
            18.720_823_012_355_86,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative_integrand_lower(0.55, 0.36, pkgt_lowq0_context)?,
            -20.398_432_856_236_53,
            1.0e-13,
        );
        let real_derivative = sfconv_real_self_energy_derivative(0.36, pkgt_lowq0_context)?;
        assert_close(real_derivative.value, 2.961_445_535_932_464, 1.0e-12);
        assert!(real_derivative.evaluations > 0);
        assert!(real_derivative.max_regions > 0);
        assert_close(
            sfconv_real_self_energy_derivative(0.95, pkgt_lowq0_context)?.value,
            -0.034_316_545_918_129_96,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(0.36, pkgt_lowq0_context)?,
            -6.610_090_947_687_186,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(0.95, pkgt_lowq0_context)?,
            0.400_030_527_250_079_1,
            1.0e-12,
        );

        let pkgt_lowq1_context = senergies_reference_context(true);
        assert_close(
            sfconv_real_self_energy_derivative(-0.20, pkgt_lowq1_context)?.value,
            -0.452_613_488_967_939_7,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(-0.20, pkgt_lowq1_context)?,
            0.0,
            0.0,
        );
        assert_close(
            sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pkgt_lowq1_context)?,
            18.394_386_251_356_508,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative(0.36, pkgt_lowq1_context)?.value,
            2.951_013_422_721_360_7,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(0.36, pkgt_lowq1_context)?,
            -6.610_090_947_687_186,
            1.0e-12,
        );

        let pklt_lowq0_context = SfconvSelfEnergyContext {
            photoelectron_momentum: 0.82,
            ..senergies_reference_context(false)
        };
        assert_close(
            sfconv_real_self_energy_derivative_integrand_upper(0.55, 0.36, pklt_lowq0_context)?,
            17.650_634_174_439_2,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pklt_lowq0_context)?,
            28.479_960_013_795_644,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative_integrand_lower(0.55, 0.36, pklt_lowq0_context)?,
            -0.329_585_793_363_548_93,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative(0.36, pklt_lowq0_context)?.value,
            0.540_035_967_831_518_6,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(0.36, pklt_lowq0_context)?,
            0.295_448_827_556_208_1,
            1.0e-12,
        );

        let pklt_lowq1_context = SfconvSelfEnergyContext {
            include_below_fermi: true,
            ..pklt_lowq0_context
        };
        assert_close(
            sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pklt_lowq1_context)?,
            27.874_936_402_523_584,
            1.0e-13,
        );
        assert_close(
            sfconv_real_self_energy_derivative(0.36, pklt_lowq1_context)?.value,
            0.664_935_465_444_516_1,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(0.36, pklt_lowq1_context)?,
            0.295_448_827_556_208_1,
            1.0e-12,
        );

        let pkeq_context = SfconvSelfEnergyContext {
            photoelectron_momentum: 1.0,
            ..senergies_reference_context(false)
        };
        assert_close(
            sfconv_real_self_energy_derivative(0.36, pkeq_context)?.value,
            0.468_906_060_872_854_14,
            1.0e-12,
        );
        assert_close(
            sfconv_imaginary_self_energy_derivative(0.36, pkeq_context)?,
            0.750_887_782_735_307_7,
            1.0e-12,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_energy_grid_matches_feff_reference() -> Result<(), SfconvError> {
        let grid = sfconv_spectral_energy_grid(0.62)?;
        assert_eq!(grid.energy.len(), SFCONV_MKSPECTF_GRID_LEN);
        assert_eq!(grid.boundaries.len(), SFCONV_MKSPECTF_GRID_LEN + 1);

        let expected_energy = [
            (0, -3.389_333_333_333_333),
            (12, -0.992),
            (21, -0.62),
            (51, -0.000_413_333_333_333_333_3),
            (52, -0.000_206_666_666_666_666_66),
            (53, 0.000_206_666_666_666_666_66),
            (54, 0.000_413_333_333_333_333_3),
            (84, 0.62),
            (93, 1.053_999_999_999_999_8),
            (105, 3.534),
            (111, 7.253_999_999_999_6),
        ];
        for (index, expected) in expected_energy {
            assert_close(grid.energy[index], expected, 1.0e-12);
        }

        let expected_boundaries = [
            (0, -3.595_999_999_999_999),
            (1, -3.286),
            (52, -0.000_31),
            (53, 0.0),
            (54, 0.000_31),
            (111, 6.944),
            (112, 7.873_999_999_999_999),
        ];
        for (index, expected) in expected_boundaries {
            assert_close(grid.boundaries[index], expected, 1.0e-12);
        }
        assert_close(grid.boundaries[1] - grid.boundaries[0], 0.31, 1.0e-14);
        assert_close(grid.boundaries[53] - grid.boundaries[52], 0.000_31, 1.0e-16);
        assert_close(grid.boundaries[112] - grid.boundaries[111], 0.93, 1.0e-14);
        Ok(())
    }

    #[test]
    fn mkspectf_energy_grid_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_spectral_energy_grid(0.0),
            Err(SfconvError::NonPositiveScalar {
                field: "plasma_frequency",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_spectral_energy_grid(f64::NAN),
            Err(SfconvError::NonFiniteScalar {
                field: "plasma_frequency",
                ..
            })
        ));
    }

    #[test]
    fn mkspectf_self_energy_renormalization_matches_feff_formula() -> Result<(), SfconvError> {
        let renormalization = sfconv_self_energy_renormalization(0.18, 0.06)?;

        assert_close(renormalization.real, 1.213_017_751_479_289_7, 1.0e-15);
        assert_close(renormalization.imaginary, 0.088_757_396_449_704_12, 1.0e-15);
        assert_close(renormalization.magnitude, 1.216_260_638_526_299_5, 1.0e-15);
        Ok(())
    }

    #[test]
    fn mkspectf_self_energy_renormalization_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_self_energy_renormalization(1.0, 0.0),
            Err(SfconvError::ZeroDenominator {
                field: "self-energy renormalization",
            })
        );
        assert!(matches!(
            sfconv_self_energy_renormalization(f64::NAN, 0.0),
            Err(SfconvError::NonFiniteScalar {
                field: "self-energy real derivative",
                ..
            })
        ));
    }

    #[test]
    fn mkspectf_exponential_reduction_matches_feff_formula() -> Result<(), SfconvError> {
        let pole_energy = array![0.5, 0.9, 1.4, 9.0];
        let pole_weight = array![0.42, 0.36, 0.22, 0.99];

        let reduction = sfconv_exponential_reduction(SfconvExponentialReductionInput {
            plasma_frequency: 0.62,
            pole_count: 3,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
        })?;

        assert_close(reduction, 0.741_119_102_598_755_9, 1.0e-15);
        Ok(())
    }

    #[test]
    fn mkspectf_exponential_reduction_rejects_invalid_inputs() {
        let pole_energy = array![0.5, 0.9, 1.4];
        let pole_weight = array![0.42, 0.36, 0.22];
        let input = SfconvExponentialReductionInput {
            plasma_frequency: 0.62,
            pole_count: 3,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
        };

        assert_eq!(
            sfconv_exponential_reduction(SfconvExponentialReductionInput {
                pole_count: 0,
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "pole_count",
                actual: 0,
                minimum: 1,
            })
        );
        assert_eq!(
            sfconv_exponential_reduction(SfconvExponentialReductionInput {
                pole_count: 4,
                ..input
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "pole_energy",
                active_len: 4,
                len: 3,
            })
        );
        assert_eq!(
            sfconv_exponential_reduction(SfconvExponentialReductionInput {
                pole_energy: array![0.5, 0.0, 1.4].view(),
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_energy",
                value: 0.0,
            })
        );
    }

    #[test]
    fn mkspectf_quasiparticle_pole_matches_feff_formula() -> Result<(), SfconvError> {
        let pole = sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
            photoelectron_energy: 0.944,
            width: 0.073,
            renormalization: SfconvRenormalization {
                real: 0.82,
                imaginary: 0.06,
                magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
            },
        })?;

        assert_close(pole.energy, 0.948_38, 1.0e-15);
        assert_close(pole.width, 0.059_86, 1.0e-15);
        Ok(())
    }

    #[test]
    fn mkspectf_quasiparticle_pole_rejects_invalid_inputs() {
        let input = SfconvQuasiparticlePoleInput {
            photoelectron_energy: 0.944,
            width: 0.073,
            renormalization: SfconvRenormalization {
                real: 0.82,
                imaginary: 0.06,
                magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
            },
        };

        assert_eq!(
            sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
                width: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "width",
                value: 0.0,
            })
        );
        let negative_width = sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
            renormalization: SfconvRenormalization {
                real: -0.82,
                ..input.renormalization
            },
            ..input
        });
        assert!(matches!(
            negative_width,
            Err(SfconvError::NonPositiveScalar {
                field: "quasiparticle width",
                value,
            }) if (value + 0.059_86).abs() <= 1.0e-15
        ));
        assert!(matches!(
            sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
                renormalization: SfconvRenormalization {
                    imaginary: f64::NAN,
                    ..input.renormalization
                },
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "renormalization_imag",
                ..
            })
        ));
    }

    #[test]
    fn mkspectf_quasiparticle_interference_matches_feff_loop() -> Result<(), SfconvError> {
        let pole_energy = array![0.47, 0.91];
        let pole_weight = array![0.35, 0.65];

        let interference =
            sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
                quasiparticle_energy: 0.35,
                upper_energy: 2.40,
                bare_photoelectron_energy: 0.85,
                plasma_frequency: 0.62,
                dispersion_parameter: 0.28,
                accuracy: 1.0e-4,
                interference_reduction: 0.43,
                pole_count: 1,
                pole_energy: pole_energy.view(),
                pole_weight: pole_weight.view(),
            })?;

        assert_close(interference.amplitude, 0.132_771_156_149_889_24, 1.0e-13);
        assert!(interference.estimated_error >= 0.0);
        assert!(interference.evaluations > 0);
        assert!(interference.max_regions > 0);
        Ok(())
    }

    #[test]
    fn mkspectf_quasiparticle_interference_rejects_invalid_inputs() {
        let pole_energy = array![0.47, 0.91];
        let pole_weight = array![0.35, 0.65];
        let input = SfconvQuasiparticleInterferenceInput {
            quasiparticle_energy: 0.35,
            upper_energy: 2.40,
            bare_photoelectron_energy: 0.85,
            plasma_frequency: 0.62,
            dispersion_parameter: 0.28,
            accuracy: 1.0e-4,
            interference_reduction: 0.43,
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
        };

        assert_eq!(
            sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
                pole_count: 0,
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "pole_count",
                actual: 0,
                minimum: 1,
            })
        );
        assert_eq!(
            sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
                bare_photoelectron_energy: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "bare_photoelectron_energy",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
                pole_count: 3,
                ..input
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "pole_energy",
                active_len: 3,
                len: 2,
            })
        );
    }

    #[test]
    fn mkspectf_quasiparticle_peak_matches_feff_reference() -> Result<(), SfconvError> {
        let grid = sfconv_spectral_energy_grid(0.62)?;
        let base = mkspectf_quasiparticle_peak_input(&grid, 53);

        let expected = [
            (1, 1.447_562_484_485_791_4e-3),
            (53, 3.978_159_860_663_877_3),
            (54, 3.979_528_363_928_183),
            (85, 2.074_480_177_474_116_4e-2),
            (112, 3.135_403_407_459_253_6e-4),
        ];
        for (index, expected_peak) in expected {
            let input = mkspectf_quasiparticle_peak_input(&grid, index);
            assert_close(
                sfconv_quasiparticle_main_peak(input)?,
                expected_peak,
                1.0e-13,
            );
        }
        assert_close(
            sfconv_quasiparticle_main_peak(base)?,
            3.978_159_860_663_877_3,
            1.0e-13,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_quasiparticle_peak_rejects_invalid_inputs() {
        let input = SfconvQuasiparticlePeakInput {
            center_energy: 0.0,
            lower_boundary: -0.1,
            upper_boundary: 0.1,
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.9348,
            quasiparticle_width: 0.0656,
            plasma_frequency: 0.62,
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
        };

        assert_eq!(
            sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
                upper_boundary: -0.1,
                ..input
            }),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: -0.1,
                upper: -0.1,
            })
        );
        assert_eq!(
            sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
                quasiparticle_width: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "quasiparticle_width",
                value: 0.0,
            })
        );
    }

    #[test]
    fn mkspectf_quasiparticle_table_matches_feff_reference() -> Result<(), SfconvError> {
        let (energy, boundaries) = mkspectf_quasiparticle_table_grid();

        let table = sfconv_quasiparticle_table(SfconvQuasiparticleTableInput {
            energy: energy.view(),
            boundaries: boundaries.view(),
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.944,
            endpoint_width: 0.073,
            quasiparticle_width: 0.073 * 0.82,
            plasma_frequency: 0.62,
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
            renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
            interference_amplitude: 0.135,
            exponential_reduction: 0.74,
        })?;

        assert_close(table.integrated_main_weight, 0.611_144_694_397_008, 1.0e-14);
        assert_close(
            table.integrated_interference_weight,
            0.139_028_009_901_435_63,
            1.0e-14,
        );
        assert_real_slice_close(
            &table.main_peak,
            &[
                0.144_118_631_068_914_32,
                0.796_854_020_052_775_2,
                3.306_037_878_829_96,
                2.944_827_731_705_054,
                0.351_606_691_790_681_77,
                0.027_414_131_538_569_52,
            ],
            1.0e-14,
        );
        assert_real_slice_close(
            &table.interference_peak,
            &[
                0.031_993_167_546_517_99,
                0.176_895_131_355_183_62,
                0.733_913_602_898_189_5,
                0.653_727_879_020_868,
                0.078_053_834_660_399_79,
                0.006_085_714_920_760_973,
            ],
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_quasiparticle_table_rejects_invalid_inputs() {
        let (energy, boundaries) = mkspectf_quasiparticle_table_grid();
        let input = SfconvQuasiparticleTableInput {
            energy: energy.view(),
            boundaries: boundaries.view(),
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.944,
            endpoint_width: 0.073,
            quasiparticle_width: 0.073 * 0.82,
            plasma_frequency: 0.62,
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
            renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
            interference_amplitude: 0.135,
            exponential_reduction: 0.74,
        };

        assert_eq!(
            sfconv_quasiparticle_table(SfconvQuasiparticleTableInput {
                boundaries: array![-0.55, -0.25, -0.05].view(),
                ..input
            }),
            Err(SfconvError::LengthMismatch {
                left: "boundaries",
                left_len: 3,
                right: "energy plus endpoints",
                right_len: 7,
            })
        );
        assert_eq!(
            sfconv_quasiparticle_table(SfconvQuasiparticleTableInput {
                endpoint_width: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "endpoint_width",
                value: 0.0,
            })
        );
    }

    #[test]
    fn mkspectf_satellite_pole_contributions_match_feff_loop() -> Result<(), SfconvError> {
        let pole_energy = array![0.47, 0.91];
        let pole_weight = array![0.35, 0.65];
        let pole_broadening = array![0.045, 0.060];

        let contributions =
            sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
                energy: 0.75,
                uniform_width: 0.009,
                quasiparticle_width: 0.02,
                plasma_frequency: 0.62,
                bare_photoelectron_energy: 0.85,
                dispersion_parameter: 0.28,
                accuracy: 1.0e-4,
                interference_reduction: 0.43,
                include_full_broadening: false,
                pole_count: 1,
                pole_energy: pole_energy.view(),
                pole_weight: pole_weight.view(),
                pole_broadening: pole_broadening.view(),
            })?;

        assert_close(
            contributions.interference_satellite,
            0.111_714_271_709_832_78,
            1.0e-12,
        );
        assert_close(
            contributions.intrinsic_satellite,
            0.173_898_309_184_430_17,
            1.0e-12,
        );
        assert!(contributions.interference_estimated_error >= 0.0);
        assert!(contributions.intrinsic_estimated_error >= 0.0);
        assert!(contributions.evaluations > 0);
        assert!(contributions.max_regions > 0);
        Ok(())
    }

    #[test]
    fn mkspectf_satellite_pole_contributions_rejects_invalid_inputs() {
        let pole_energy = array![0.47, 0.91];
        let pole_weight = array![0.35, 0.65];
        let pole_broadening = array![0.045, 0.060];
        let short_broadening = array![0.045];
        let input = SfconvSatellitePoleContributionsInput {
            energy: 0.75,
            uniform_width: 0.009,
            quasiparticle_width: 0.02,
            plasma_frequency: 0.62,
            bare_photoelectron_energy: 0.85,
            dispersion_parameter: 0.28,
            accuracy: 1.0e-4,
            interference_reduction: 0.43,
            include_full_broadening: false,
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
        };

        assert_eq!(
            sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
                pole_count: 0,
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "pole_count",
                actual: 0,
                minimum: 1,
            })
        );
        assert_eq!(
            sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
                uniform_width: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "uniform_width",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
                pole_count: 2,
                pole_broadening: short_broadening.view(),
                ..input
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "pole_broadening",
                active_len: 2,
                len: 1,
            })
        );
    }

    #[test]
    fn mkspectf_extrinsic_satellite_modes_match_feff_branches() -> Result<(), SfconvError> {
        let input = SfconvExtrinsicSatelliteInput {
            energy: 0.36,
            main_peak: 0.0123,
            imaginary_derivative: -0.015,
            mode: SfconvExtrinsicSatelliteMode::Debroadened,
            context: mksat_reference_context(),
            self_energy: mksat_reference_self_energy(),
        };

        assert_close(
            sfconv_extrinsic_satellite(input)?,
            -0.044_294_665_346_589_21,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
                mode: SfconvExtrinsicSatelliteMode::FullBroadening,
                ..input
            })?,
            0.039_176_601_376_466_56,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
                mode: SfconvExtrinsicSatelliteMode::BroadenedMinusMain,
                ..input
            })?,
            0.026_876_601_376_466_56,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
                mode: SfconvExtrinsicSatelliteMode::DerivativeExpansion,
                ..input
            })?,
            -0.121_822_302_119_722_35,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_extrinsic_satellite_rejects_invalid_inputs() {
        let input = SfconvExtrinsicSatelliteInput {
            energy: 0.36,
            main_peak: 0.0123,
            imaginary_derivative: -0.015,
            mode: SfconvExtrinsicSatelliteMode::Debroadened,
            context: mksat_reference_context(),
            self_energy: mksat_reference_self_energy(),
        };

        assert!(matches!(
            sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
                main_peak: f64::NAN,
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "main_peak",
                ..
            })
        ));
        assert_eq!(
            sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
                energy: 0.0,
                mode: SfconvExtrinsicSatelliteMode::DerivativeExpansion,
                ..input
            }),
            Err(SfconvError::ZeroDenominator {
                field: "derivative extrinsic satellite energy",
            })
        );
    }

    #[test]
    fn mkspectf_spectral_cell_matches_feff_loop() -> Result<(), SfconvError> {
        let pole_energy = array![0.47, 0.91];
        let pole_weight = array![0.35, 0.65];
        let pole_broadening = array![0.045, 0.060];

        let cell = sfconv_spectral_cell(SfconvSpectralCellInput {
            center_energy: 0.75,
            lower_boundary: 0.70,
            upper_boundary: 0.80,
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.944,
            quasiparticle_width: 0.073 * 0.82,
            interference_amplitude: 0.135,
            extrinsic_mode: SfconvExtrinsicSatelliteMode::Debroadened,
            imaginary_derivative: -0.015,
            uniform_width: 0.009,
            interference_reduction: 0.43,
            context: mksat_reference_context(),
            self_energy: mksat_reference_self_energy(),
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
        })?;

        assert_close(cell.main_peak, 0.010_633_354_619_341_801, 1.0e-14);
        assert_close(
            cell.quasiparticle_interference,
            0.002_360_518_507_530_576,
            1.0e-14,
        );
        assert_close(
            cell.extrinsic_satellite,
            -0.008_565_813_402_423_753,
            1.0e-14,
        );
        assert_close(
            cell.interference_satellite,
            0.111_714_271_709_832_78,
            1.0e-12,
        );
        assert_close(cell.intrinsic_satellite, 0.173_898_309_184_430_17, 1.0e-12);
        assert_close(cell.combined_satellite, -0.058_096_047_637_659_13, 1.0e-12);
        assert!(cell.evaluations > 0);
        assert!(cell.max_regions > 0);
        Ok(())
    }

    #[test]
    fn mkspectf_spectral_cell_adds_quasiparticle_for_full_broadening() -> Result<(), SfconvError> {
        let pole_energy = array![0.47];
        let pole_weight = array![1.0];
        let pole_broadening = array![0.045];

        let cell = sfconv_spectral_cell(SfconvSpectralCellInput {
            center_energy: 0.75,
            lower_boundary: 0.70,
            upper_boundary: 0.80,
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.944,
            quasiparticle_width: 0.073 * 0.82,
            interference_amplitude: 0.135,
            extrinsic_mode: SfconvExtrinsicSatelliteMode::FullBroadening,
            imaginary_derivative: -0.015,
            uniform_width: 0.009,
            interference_reduction: 0.43,
            context: mksat_reference_context(),
            self_energy: mksat_reference_self_energy(),
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
        })?;

        assert_close(
            cell.combined_satellite,
            cell.extrinsic_satellite + cell.intrinsic_satellite - 2.0 * cell.interference_satellite
                + cell.quasiparticle_interference,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_spectral_cell_rejects_invalid_inputs() {
        let pole_energy = array![0.47];
        let pole_weight = array![1.0];
        let pole_broadening = array![0.045];
        let input = SfconvSpectralCellInput {
            center_energy: 0.75,
            lower_boundary: 0.70,
            upper_boundary: 0.80,
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.944,
            quasiparticle_width: 0.073 * 0.82,
            interference_amplitude: 0.135,
            extrinsic_mode: SfconvExtrinsicSatelliteMode::Debroadened,
            imaginary_derivative: -0.015,
            uniform_width: 0.009,
            interference_reduction: 0.43,
            context: mksat_reference_context(),
            self_energy: mksat_reference_self_energy(),
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
        };

        assert!(matches!(
            sfconv_spectral_cell(SfconvSpectralCellInput {
                interference_amplitude: f64::NAN,
                ..input
            }),
            Err(SfconvError::NonFiniteScalar {
                field: "interference_amplitude",
                ..
            })
        ));
        assert_eq!(
            sfconv_spectral_cell(SfconvSpectralCellInput {
                pole_count: 0,
                ..input
            }),
            Err(SfconvError::CountTooSmall {
                name: "pole_count",
                actual: 0,
                minimum: 1,
            })
        );
    }

    #[test]
    fn mkspectf_satellite_table_matches_feff_reference() -> Result<(), SfconvError> {
        let inputs = mkspectf_satellite_table_inputs();

        let table = sfconv_satellite_table(SfconvSatelliteTableInput {
            main_peak: inputs.main_peak.view(),
            quasiparticle_interference: inputs.quasiparticle_interference.view(),
            extrinsic_satellite: inputs.extrinsic.view(),
            interference_satellite: inputs.interference.view(),
            intrinsic_satellite: inputs.intrinsic.view(),
            boundaries: inputs.boundaries.view(),
            quasiparticle_lower_column_1based: 3,
            quasiparticle_upper_column_1based: 4,
            include_full_broadening_quasiparticle: true,
            exponential_reduction: 0.74,
        })?;

        assert_close(
            table.integrated_extrinsic_weight,
            0.081_844_000_000_000_01,
            1.0e-15,
        );
        assert_close(table.integrated_interference_weight, 0.022_610_7, 1.0e-15);
        assert_close(
            table.integrated_intrinsic_weight,
            0.036_378_400_000_000_005,
            1.0e-15,
        );
        assert_real_slice_close(
            &table.spectral_function.row(1).to_owned(),
            &[0.04, 0.09, 0.08, 0.08, 0.13, 0.07],
            1.0e-15,
        );
        assert_real_slice_close(
            &table.spectral_function.row(3).to_owned(),
            &[0.01, 0.025, 0.006, 0.055, 0.04, 0.015],
            1.0e-15,
        );
        assert_real_slice_close(
            &table.spectral_function.row(4).to_owned(),
            &[0.02, 0.035, 0.012, 0.08, 0.065, 0.025],
            1.0e-15,
        );
        assert_real_slice_close(
            &table.spectral_function.row(5).to_owned(),
            &[
                0.071_993_167_546_518,
                0.251_895_131_355_183_6,
                0.713_913_602_898_189_5,
                0.803_727_879_020_868_1,
                0.193_053_834_660_399_8,
                0.071_085_714_920_760_98,
            ],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_satellite_table_rejects_invalid_inputs() {
        let inputs = mkspectf_satellite_table_inputs();
        let input = SfconvSatelliteTableInput {
            main_peak: inputs.main_peak.view(),
            quasiparticle_interference: inputs.quasiparticle_interference.view(),
            extrinsic_satellite: inputs.extrinsic.view(),
            interference_satellite: inputs.interference.view(),
            intrinsic_satellite: inputs.intrinsic.view(),
            boundaries: inputs.boundaries.view(),
            quasiparticle_lower_column_1based: 3,
            quasiparticle_upper_column_1based: 4,
            include_full_broadening_quasiparticle: true,
            exponential_reduction: 0.74,
        };

        assert_eq!(
            sfconv_satellite_table(SfconvSatelliteTableInput {
                main_peak: array![0.1, 0.2].view(),
                ..input
            }),
            Err(SfconvError::LengthMismatch {
                left: "main_peak",
                left_len: 2,
                right: "satellite columns",
                right_len: 6,
            })
        );
        assert_eq!(
            sfconv_satellite_table(SfconvSatelliteTableInput {
                quasiparticle_lower_column_1based: 0,
                ..input
            }),
            Err(SfconvError::IndexOutOfRange {
                field: "quasiparticle_lower_column",
                index: 0,
                len: 6,
            })
        );
        assert_eq!(
            sfconv_satellite_table(SfconvSatelliteTableInput {
                exponential_reduction: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "exponential_reduction",
                value: 0.0,
            })
        );
    }

    #[test]
    fn mkspectf_extrinsic_split_matches_feff_reference() -> Result<(), SfconvError> {
        let (spectral_function, energy, boundaries) = mkspectf_extrinsic_split_inputs();

        let split = sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
            spectral_function: spectral_function.view(),
            energy: energy.view(),
            boundaries: boundaries.view(),
            photoelectron_energy: 0.05,
            beta_zero: 1.0,
        })?;

        assert_eq!(split.switch_column, 5);
        assert!(split.derivative_triggered);
        assert_close(split.switch_energy, 0.35, 1.0e-15);
        assert_real_slice_close(
            &split.spectral_function.row(6).to_owned(),
            &[0.10, 0.18, 0.35, 0.30, 0.22, 0.0, 0.0, 0.0],
            1.0e-15,
        );
        assert_real_slice_close(
            &split.spectral_function.row(7).to_owned(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.25, 0.20],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_extrinsic_split_rejects_invalid_inputs() {
        let (spectral_function, energy, boundaries) = mkspectf_extrinsic_split_inputs();
        assert_eq!(
            sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
                spectral_function: Array2::<Real>::zeros((7, energy.len()).f()).view(),
                energy: energy.view(),
                boundaries: boundaries.view(),
                photoelectron_energy: 0.05,
                beta_zero: 1.0,
            }),
            Err(SfconvError::CountMismatch {
                field: "spectral_function rows",
                actual: 7,
                expected: 8,
            })
        );
        assert_eq!(
            sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
                spectral_function: spectral_function.view(),
                energy: array![-0.6, -0.3, -0.4, 0.0, 0.1, 0.3, 0.6, 1.0].view(),
                boundaries: boundaries.view(),
                photoelectron_energy: 0.05,
                beta_zero: 1.0,
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "energy",
                row: 2,
                previous: -0.3,
                current: -0.4,
            })
        );

        let mut flat = spectral_function.clone();
        flat.row_mut(1).fill(0.1);
        assert_eq!(
            sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
                spectral_function: flat.view(),
                energy: energy.view(),
                boundaries: boundaries.view(),
                photoelectron_energy: 0.05,
                beta_zero: 1.0,
            }),
            Err(SfconvError::MissingTrigger {
                field: "extrinsic satellite split",
            })
        );
    }

    #[test]
    fn mkspectf_satellite_correction_matches_feff_reference() -> Result<(), SfconvError> {
        let (spectral_function, boundaries) = mkspectf_satellite_correction_inputs();

        let correction = sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
            spectral_function: spectral_function.view(),
            boundaries: boundaries.view(),
            uniform_width: 0.2,
            exponential_reduction: 0.73,
        })?;

        assert_close(correction.uncorrected_satellite_weight, 0.267, 1.0e-15);
        assert_close(
            correction.clipped_negative_weight,
            -0.053_999_999_999_999_99,
            1.0e-15,
        );
        assert_close(
            correction.correction_factor,
            0.831_775_700_934_579_4,
            1.0e-15,
        );
        assert_real_slice_close(
            &correction.weights,
            &[
                0.259_15,
                0.119_355_000_000_000_03,
                0.174_470_000_000_000_01,
                0.036_5,
                0.054_02,
            ],
            1.0e-14,
        );

        let expected_rows = [
            (0, 0.121_028_037_383_177_58, 0.25),
            (1, 0.11, 0.0),
            (2, 0.088_411_214_953_271_03, 0.1),
            (3, 0.265, 0.0),
            (4, 0.090_373_831_775_700_99, 0.48),
            (5, 0.048_504_672_897_196_26, 0.220_000_000_000_000_03),
        ];
        for (column, expected_interference, expected_combined) in expected_rows {
            assert_close(
                correction.spectral_function[(3, column)],
                expected_interference,
                1.0e-14,
            );
            assert_close(
                correction.spectral_function[(5, column)],
                expected_combined,
                1.0e-14,
            );
        }
        Ok(())
    }

    #[test]
    fn mkspectf_satellite_correction_rejects_invalid_inputs() {
        let (spectral_function, boundaries) = mkspectf_satellite_correction_inputs();
        assert_eq!(
            sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
                spectral_function: Array2::<Real>::zeros((7, spectral_function.ncols()).f()).view(),
                boundaries: boundaries.view(),
                uniform_width: 0.2,
                exponential_reduction: 0.73,
            }),
            Err(SfconvError::CountMismatch {
                field: "spectral_function rows",
                actual: 7,
                expected: 8,
            })
        );
        assert_eq!(
            sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
                spectral_function: spectral_function.view(),
                boundaries: array![0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6].view(),
                uniform_width: 0.2,
                exponential_reduction: 0.73,
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "boundaries",
                row: 2,
                previous: 0.2,
                current: 0.1,
            })
        );
        assert_eq!(
            sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
                spectral_function: Array2::<Real>::zeros((8, 2).f()).view(),
                boundaries: array![0.0, 0.2, 0.4].view(),
                uniform_width: 0.2,
                exponential_reduction: 0.73,
            }),
            Err(SfconvError::ZeroDenominator {
                field: "satellite correction",
            })
        );
    }

    #[test]
    fn mkspectf_spectral_weights_match_feff_reference() -> Result<(), SfconvError> {
        let satellite_weights = array![0.259_15, 0.119_355, 0.174_47, 0.036_5, 0.054_02];

        let weights = sfconv_spectral_weights(SfconvSpectralWeightsInput {
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
            renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
            interference_amplitude: 0.135,
            interference_reduction: 0.43,
            exponential_reduction: 0.74,
            satellite_weights: satellite_weights.view(),
        })?;

        assert_real_slice_close(
            &weights,
            &[
                0.606_8,
                0.044_4,
                0.057_923_012_361_364_55,
                0.259_15,
                0.119_355,
                0.174_47,
                0.036_5,
                0.054_02,
            ],
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn mkspectf_spectral_weights_rejects_invalid_inputs() {
        let satellite_weights = array![0.259_15, 0.119_355, 0.174_47, 0.036_5, 0.054_02];
        let input = SfconvSpectralWeightsInput {
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
            renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
            interference_amplitude: 0.135,
            interference_reduction: 0.43,
            exponential_reduction: 0.74,
            satellite_weights: satellite_weights.view(),
        };

        assert_eq!(
            sfconv_spectral_weights(SfconvSpectralWeightsInput {
                satellite_weights: array![0.1, 0.2].view(),
                ..input
            }),
            Err(SfconvError::CountMismatch {
                field: "satellite_weights",
                actual: 2,
                expected: 5,
            })
        );
        assert_eq!(
            sfconv_spectral_weights(SfconvSpectralWeightsInput {
                renormalization_magnitude: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "renormalization_magnitude",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_spectral_weights(SfconvSpectralWeightsInput {
                exponential_reduction: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "exponential_reduction",
                value: 0.0,
            })
        );
    }

    #[test]
    fn so2conv_path_average_matches_feff_reference() -> Result<(), SfconvError> {
        let (source_momentum, amplitude_reduction, phase_shift) = so2conv_path_average_inputs();

        let no_exact = sfconv_path_average(SfconvPathAverageInput {
            source_momentum: source_momentum.view(),
            amplitude_reduction: amplitude_reduction.view(),
            phase_shift: phase_shift.view(),
            previous_momentum: 1.00,
            center_momentum: 1.60,
            next_momentum: 2.30,
            momentum_step: 0.05,
        })?;
        assert_close(
            no_exact.amplitude_reduction,
            0.888_169_014_084_507_1,
            1.0e-15,
        );
        assert_close(no_exact.phase_shift, 0.136_384_976_525_821_6, 1.0e-15);
        assert_close(no_exact.normalization, 0.126_785_714_285_714_28, 1.0e-15);

        let exact = sfconv_path_average(SfconvPathAverageInput {
            source_momentum: source_momentum.view(),
            amplitude_reduction: amplitude_reduction.view(),
            phase_shift: phase_shift.view(),
            previous_momentum: 1.00,
            center_momentum: 1.50,
            next_momentum: 2.00,
            momentum_step: 0.05,
        })?;
        assert_close(exact.amplitude_reduction, 0.897_5, 1.0e-15);
        assert_close(exact.phase_shift, 0.152_5, 1.0e-15);
        assert_close(exact.normalization, 0.1, 1.0e-15);
        Ok(())
    }

    #[test]
    fn so2conv_path_average_rejects_invalid_inputs() {
        let (source_momentum, amplitude_reduction, phase_shift) = so2conv_path_average_inputs();
        let input = SfconvPathAverageInput {
            source_momentum: source_momentum.view(),
            amplitude_reduction: amplitude_reduction.view(),
            phase_shift: phase_shift.view(),
            previous_momentum: 1.00,
            center_momentum: 1.60,
            next_momentum: 2.30,
            momentum_step: 0.05,
        };

        assert_eq!(
            sfconv_path_average(SfconvPathAverageInput {
                amplitude_reduction: array![0.1].view(),
                ..input
            }),
            Err(SfconvError::LengthMismatch {
                left: "source_momentum",
                left_len: 7,
                right: "amplitude_reduction",
                right_len: 1,
            })
        );
        assert_eq!(
            sfconv_path_average(SfconvPathAverageInput {
                source_momentum: array![0.75, 1.00, 0.90, 1.50, 1.75, 2.00, 2.25].view(),
                ..input
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "source_momentum",
                row: 2,
                previous: 1.00,
                current: 0.90,
            })
        );
        assert_eq!(
            sfconv_path_average(SfconvPathAverageInput {
                previous_momentum: 2.00,
                center_momentum: 1.50,
                next_momentum: 2.30,
                ..input
            }),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: 2.00,
                upper: 2.30,
            })
        );
        assert_eq!(
            sfconv_path_average(SfconvPathAverageInput {
                previous_momentum: 3.00,
                center_momentum: 3.20,
                next_momentum: 3.40,
                ..input
            }),
            Err(SfconvError::ZeroDenominator {
                field: "path average normalization",
            })
        );
        assert_eq!(
            sfconv_path_average(SfconvPathAverageInput {
                momentum_step: 0.0,
                ..input
            }),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum_step",
                value: 0.0,
            })
        );
    }

    #[test]
    fn finds_senergies_split_points_like_feff() -> Result<(), SfconvError> {
        let candidates = array![0.90, 0.20, 1.40, 0.70, -0.10];

        let forward = sfconv_find_singularities(0.15, 1.00, candidates.view())?;
        assert_real_slice_close(&forward, &[0.20, 0.70, 0.90], 0.0);

        let reverse = sfconv_find_singularities(1.00, 0.15, candidates.view())?;
        assert_real_slice_close(&reverse, &[0.20, 0.70, 0.90], 0.0);

        let empty = sfconv_find_singularities(0.15, 0.15, candidates.view())?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[test]
    fn senergies_helpers_reject_invalid_inputs() {
        let context = senergies_reference_context(false);
        assert_eq!(
            sfconv_free_electron_exchange(0.0, 1.0),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_extrinsic_beta(
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 0.0,
                    ..context
                },
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "photoelectron_momentum",
                ..
            })
        ));
        assert!(matches!(
            sfconv_real_self_energy_derivative(
                0.36,
                SfconvSelfEnergyContext {
                    pole_broadening: 0.0,
                    ..context
                },
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_broadening",
                ..
            })
        ));
        assert!(matches!(
            sfconv_find_singularities(0.0, 1.0, array![0.2, f64::NAN].view()),
            Err(SfconvError::NonFiniteValue {
                field: "singularity candidate",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn grater_integrate_matches_feff_reference() -> Result<(), SfconvError> {
        assert_integral_close(
            sfconv_grater_integrate(
                |x| Ok(x.powi(4) - 2.0 * x + 1.0),
                -0.25,
                1.75,
                1.0e-6,
                1.0e-6,
                &[],
            )?,
            SfconvAdaptiveIntegral {
                value: 2.282_812_623_992_166_7,
                estimated_error: 1.651_258_862_978_011_2e-8,
                evaluations: 9,
                max_regions: 1,
            },
            1.0e-14,
        );

        assert_integral_close(
            sfconv_grater_integrate(
                |x| Ok((5.0 * x).sin() / (1.0 + x * x)),
                0.0,
                4.0,
                1.0e-6,
                1.0e-6,
                &[],
            )?,
            SfconvAdaptiveIntegral {
                value: 0.214_866_405_696_591,
                estimated_error: 2.960_202_197_766_978_5e-7,
                evaluations: 135,
                max_regions: 6,
            },
            1.0e-13,
        );

        assert_integral_close(
            sfconv_grater_integrate(
                |x| Ok((x - 0.3).abs() + 0.25 * (x - 0.8).abs()),
                -1.0,
                2.0,
                1.0e-6,
                1.0e-6,
                &[0.3, 0.8],
            )?,
            SfconvAdaptiveIntegral {
                value: 2.874_999_978_367_709_4,
                estimated_error: 1.071_163_531_207_730_6e-7,
                evaluations: 27,
                max_regions: 3,
            },
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn grater_integrate_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_grater_integrate(Ok, 1.0, 1.0, 1.0e-6, 1.0e-6, &[]),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: 1.0,
                upper: 1.0,
            })
        );
        assert_eq!(
            sfconv_grater_integrate(Ok, 0.0, 1.0, 0.0, 1.0e-6, &[]),
            Err(SfconvError::NonPositiveTolerance {
                field: "abr",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_grater_integrate(Ok, 0.0, 1.0, 1.0e-6, 1.0e-6, &[0.5, 0.4]),
            Err(SfconvError::InvalidSingularity {
                index: 1,
                value: 0.4,
            })
        );
        assert!(matches!(
            sfconv_grater_integrate(|_| Ok(f64::NAN), 0.0, 1.0, 1.0e-6, 1.0e-6, &[]),
            Err(SfconvError::NonFiniteValue {
                field: "grater integrand",
                ..
            })
        ));
    }

    #[test]
    fn mksat_helpers_match_feff_reference() -> Result<(), SfconvError> {
        let context = mksat_reference_context();
        let self_energy = mksat_reference_self_energy();

        assert_close(
            sfconv_extrinsic_satellite_debroadened(0.36, context, self_energy)?,
            -0.044_294_665_346_589_21,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_satellite_broadened(0.36, self_energy)?,
            0.039_176_601_376_466_56,
            1.0e-14,
        );
        assert_close(
            sfconv_interference_satellite_integrand(0.55, 0.32, 0.045, context)?,
            4.656_810_436_207_971,
            1.0e-13,
        );
        assert_close(
            sfconv_intrinsic_satellite_integrand(0.55, 0.32, 0.045, context)?,
            2.780_182_754_299_514_3,
            1.0e-13,
        );
        assert_close(
            sfconv_interference_satellite_integrand(0.55, 0.95, 0.045, context)?,
            1.568_981_693_763_851_9,
            1.0e-13,
        );

        let interference = sfconv_interference_satellite(0.75, 0.045, context)?;
        assert_close(interference.value, 0.742_287_519_666_663_1, 1.0e-12);
        assert!(interference.evaluations > 0);
        assert!(interference.max_regions > 0);

        let intrinsic = sfconv_intrinsic_satellite(0.75, 0.045, context)?;
        assert_close(intrinsic.value, 0.496_852_311_955_514_77, 1.0e-12);
        assert!(intrinsic.evaluations > 0);
        assert!(intrinsic.max_regions > 0);

        let quasiparticle = sfconv_interference_quasiparticle(0.35, 2.40, context)?;
        assert_close(quasiparticle.value, 0.882_200_373_088_965_2, 1.0e-12);
        assert!(quasiparticle.evaluations > 0);
        assert!(quasiparticle.max_regions > 0);

        assert_close(
            sfconv_interference_quasiparticle(-0.01, 2.40, context)?.value,
            0.0,
            0.0,
        );
        assert_close(
            sfconv_interference_quasiparticle_integrand(0.55, (2.0_f64 * 0.85).sqrt(), context)?,
            0.886_179_631_715_177_2,
            1.0e-13,
        );
        Ok(())
    }

    #[test]
    fn mksat_helpers_reject_invalid_inputs() {
        let context = mksat_reference_context();
        let self_energy = mksat_reference_self_energy();
        assert_eq!(
            sfconv_extrinsic_satellite_debroadened(0.0, context, self_energy),
            Err(SfconvError::ZeroDenominator {
                field: "satellite energy",
            })
        );
        assert!(matches!(
            sfconv_interference_satellite_integrand(0.0, 0.32, 0.045, context),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum",
                ..
            })
        ));
        assert!(matches!(
            sfconv_intrinsic_satellite(0.75, 0.0, context),
            Err(SfconvError::NonPositiveScalar {
                field: "satellite width",
                ..
            })
        ));
        assert!(matches!(
            sfconv_interference_quasiparticle(0.35, -1.0, context),
            Err(SfconvError::NegativeRadicand { .. })
        ));
    }

    #[test]
    fn interpolates_spectral_function_matches_feff_interpsf_reference() -> Result<(), SfconvError> {
        let (energy, spectral_function) = interpsf_reference_inputs();
        let interpolation =
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            })?;

        let expected_energy = [
            -2.0,
            -1.727_590_833_333_333_4,
            -1.455_181_666_666_666_8,
            -1.182_772_5,
            -0.910_363_333_333_333_4,
            -0.637_954_166_666_666_8,
            -0.365_545,
            -0.093_135_833_333_333_42,
            0.179_273_333_333_333_17,
            0.451_682_5,
            0.724_091_666_666_666_4,
            0.996_500_833_333_333_2,
            1.268_91,
        ];
        let expected_spectral_function = [
            -0.03,
            -0.035_578_048_005_086_65,
            -0.040_441_264_512_519_18,
            -0.044_809_714_285_714_24,
            -0.048_809_091_974_223_85,
            -0.052_519_432_577_500_3,
            -0.055_996_334_265_299_72,
            -0.059_278_128_963_028_02,
            -0.062_395_108_746_383_016,
            -0.065_369_121_964_238_19,
            -0.068_218_832_777_920_12,
            -0.070_958_429_921_906_93,
            -0.073_599_999_999_999_89,
        ];

        assert_real_slice_close(&interpolation.energy, &expected_energy, 1.0e-15);
        assert_real_slice_close(
            &interpolation.spectral_function,
            &expected_spectral_function,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn interpolates_spectral_function_rejects_invalid_inputs() {
        let (energy, spectral_function) = interpsf_reference_inputs();

        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 1,
            }),
            Err(SfconvError::CountTooSmall {
                name: "output_len",
                ..
            })
        ));

        let short_rows =
            Array2::from_shape_fn((7, spectral_function.ncols()).f(), |(row, column)| {
                spectral_function[(row, column)]
            });
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: short_rows.view(),
                output_len: 13,
            }),
            Err(SfconvError::CountMismatch {
                field: "spectral_function rows",
                actual: 7,
                expected: 8,
            })
        ));

        let short_energy = Array1::from_iter(energy.iter().copied().take(100));
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: short_energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            }),
            Err(SfconvError::LengthMismatch {
                left: "energy",
                right: "spectral_function columns",
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[10] = bad_energy[9];
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: bad_energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            }),
            Err(SfconvError::NonIncreasingEnergy { row: 10, .. })
        ));
    }

    #[test]
    fn convolve_matches_feff_sfconvsub_reference() -> Result<(), SfconvError> {
        let reference = sfconvsub_reference_inputs();

        let cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
            photoelectron_energy: 1.35,
            chemical_potential: 0.15,
            core_hole_lifetime: 0.08,
            signal_energy: reference.signal_energy.view(),
            signal: reference.signal.view(),
            spectral_energy: reference.spectral_energy.view(),
            spectral_function: reference.spectral_function.view(),
            weights: reference.weights.view(),
            asymmetric_phase: false,
            cutoff: true,
            plasma_frequency: 0.55,
        })?;
        assert_close(cutoff_phase.amplitude, 0.404_768_834_000_475_8, 1.0e-14);
        assert_close(cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

        let no_cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
            cutoff: false,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        })?;
        assert_close(no_cutoff_phase.amplitude, 0.405_036_447_280_840_4, 1.0e-14);
        assert_close(no_cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

        let asym_cutoff = sfconv_convolve(SfconvConvolutionInput {
            asymmetric_phase: true,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        })?;
        assert_close(asym_cutoff.amplitude, 0.394_308_834_584_619_57, 1.0e-14);
        assert_close(asym_cutoff.phase, 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn convolve_rejects_invalid_inputs() {
        let reference = sfconvsub_reference_inputs();

        let short_signal = array![0.62, 0.82, 0.74, 0.48, 0.22];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                signal: short_signal.view(),
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::LengthMismatch {
                left: "signal_energy",
                ..
            })
        ));

        let bad_spectral_energy = array![-0.18, -0.04, 0.0, 0.0, 0.31, 0.55, 0.82];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                spectral_energy: bad_spectral_energy.view(),
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "spectral_energy",
                row: 3,
                ..
            })
        ));

        let zero_asym_weight = array![0.0, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                weights: zero_asym_weight.view(),
                asymmetric_phase: true,
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::ZeroAsymmetricWeight)
        ));
    }

    fn mkrmu_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let indices = (1..=count).map(|index| index as Real);
        let imaginary = Array1::from_iter(
            indices
                .clone()
                .map(|index| (0.17 * index).sin() + 0.01 * index),
        );
        let reference_imaginary =
            Array1::from_iter(indices.clone().map(|index| 0.2 * (0.11 * index).cos()));
        let energy = Array1::from_iter((0..count).map(|index| {
            let index = index as Real;
            0.05 * index + 0.002 * index * index
        }));
        (imaginary, reference_imaginary, energy)
    }

    fn plset_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let energy = Array1::from_shape_fn(5, |index| {
            let i = index as Real + 1.0;
            0.12 * i + 0.015 * i * i
        });
        let weight = Array1::from_shape_fn(5, |index| {
            let i = index as Real + 1.0;
            0.25 + 0.07 * i
        });
        let broadening = Array1::from_shape_fn(5, |index| {
            let i = index as Real + 1.0;
            0.01 * i + 0.002 * i * i
        });
        (energy, weight, broadening)
    }

    fn interpsf_reference_inputs() -> (Array1<Real>, Array2<Real>) {
        let count = 110usize;
        let energy = Array1::from_shape_fn(count, |index| {
            let i = index as Real;
            -2.0 + 0.018 * i + 0.000_11 * i * i
        });
        let spectral_function = Array2::from_shape_fn((8, count).f(), |(row, column)| {
            let fortran_row = row as Real + 1.0;
            let i = column as Real;
            0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
        });
        (energy, spectral_function)
    }

    struct SfconvSubReference {
        spectral_energy: Array1<Real>,
        spectral_function: Array1<Real>,
        signal_energy: Array1<Real>,
        signal: Array1<Real>,
        weights: Array1<Real>,
    }

    fn sfconvsub_reference_inputs() -> SfconvSubReference {
        SfconvSubReference {
            spectral_energy: array![-0.18, -0.04, 0.0, 0.12, 0.31, 0.55, 0.82],
            spectral_function: array![0.05, 0.18, 0.30, 0.23, 0.14, 0.07, 0.02],
            signal_energy: array![0.40, 0.72, 0.95, 1.22, 1.58, 1.95],
            signal: array![0.62, 0.82, 0.74, 0.48, 0.22, 0.12],
            weights: array![0.72, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    fn sfconv_reference_input<'a>(
        signal_energy: ndarray::ArrayView1<'a, Real>,
        signal: ndarray::ArrayView1<'a, Real>,
        spectral_energy: ndarray::ArrayView1<'a, Real>,
        spectral_function: ndarray::ArrayView1<'a, Real>,
        weights: ndarray::ArrayView1<'a, Real>,
    ) -> SfconvConvolutionInput<'a> {
        SfconvConvolutionInput {
            photoelectron_energy: 1.35,
            chemical_potential: 0.15,
            core_hole_lifetime: 0.08,
            signal_energy,
            signal,
            spectral_energy,
            spectral_function,
            weights,
            asymmetric_phase: false,
            cutoff: true,
            plasma_frequency: 0.55,
        }
    }

    fn mkspectf_quasiparticle_peak_input(
        grid: &SfconvSpectralEnergyGrid,
        index_1based: usize,
    ) -> SfconvQuasiparticlePeakInput {
        let index = index_1based - 1;
        SfconvQuasiparticlePeakInput {
            center_energy: grid.energy[index],
            lower_boundary: grid.boundaries[index],
            upper_boundary: grid.boundaries[index + 1],
            photoelectron_energy: 0.93,
            quasiparticle_energy: 0.93 + 0.08 * 0.06,
            quasiparticle_width: 0.08 * 0.82,
            plasma_frequency: 0.62,
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
        }
    }

    fn mkspectf_quasiparticle_table_grid() -> (Array1<Real>, Array1<Real>) {
        let energy = array![-0.40, -0.12, -0.01, 0.02, 0.20, 0.55];
        let boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
        (energy, boundaries)
    }

    struct MkspectfSatelliteTableInputs {
        main_peak: Array1<Real>,
        quasiparticle_interference: Array1<Real>,
        extrinsic: Array1<Real>,
        interference: Array1<Real>,
        intrinsic: Array1<Real>,
        boundaries: Array1<Real>,
    }

    fn mkspectf_satellite_table_inputs() -> MkspectfSatelliteTableInputs {
        let main_peak = array![
            0.144_118_631_068_914_32,
            0.796_854_020_052_775_2,
            3.306_037_878_829_96,
            2.944_827_731_705_054,
            0.351_606_691_790_681_77,
            0.027_414_131_538_569_52,
        ];
        let quasiparticle_interference = array![
            0.031_993_167_546_517_99,
            0.176_895_131_355_183_62,
            0.733_913_602_898_189_5,
            0.653_727_879_020_868,
            0.078_053_834_660_399_79,
            0.006_085_714_920_760_973,
        ];
        let extrinsic = array![0.04, 0.09, -0.02, 0.18, 0.13, 0.07];
        let interference = array![0.01, 0.025, 0.006, 0.055, 0.04, 0.015];
        let intrinsic = array![0.02, 0.035, 0.012, 0.08, 0.065, 0.025];
        let boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
        MkspectfSatelliteTableInputs {
            main_peak,
            quasiparticle_interference,
            extrinsic,
            interference,
            intrinsic,
            boundaries,
        }
    }

    fn mkspectf_extrinsic_split_inputs() -> (Array2<Real>, Array1<Real>, Array1<Real>) {
        let mut spectral_function = Array2::<Real>::zeros((8, 8).f());
        for (row, values) in [
            (1, [0.10, 0.18, 0.35, 0.30, 0.22, 0.15, 0.25, 0.20]),
            (4, [0.02, 0.05, 0.11, 0.16, 0.13, 0.09, 0.12, 0.07]),
            (6, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
            (7, [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        ] {
            for (column, value) in values.into_iter().enumerate() {
                spectral_function[(row, column)] = value;
            }
        }
        let energy = array![-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0];
        let boundaries = array![-0.75, -0.45, -0.20, -0.05, 0.05, 0.20, 0.45, 0.80, 1.20];
        (spectral_function, energy, boundaries)
    }

    fn mkspectf_satellite_correction_inputs() -> (Array2<Real>, Array1<Real>) {
        let mut spectral_function = Array2::<Real>::zeros((8, 6).f());
        for (row, values) in [
            (1, [0.40, 0.18, 0.06, 0.50, 0.28, 0.08]),
            (3, [0.10, 0.16, 0.08, 0.35, 0.05, 0.03]),
            (4, [0.05, 0.04, 0.20, 0.03, 0.30, 0.20]),
            (6, [0.08, 0.05, 0.03, 0.12, 0.07, 0.02]),
            (7, [0.04, 0.02, 0.01, 0.06, 0.09, 0.03]),
        ] {
            for (column, value) in values.into_iter().enumerate() {
                spectral_function[(row, column)] = value;
            }
        }
        let boundaries = array![-0.4, -0.2, 0.0, 0.15, 0.35, 0.7, 1.1];
        (spectral_function, boundaries)
    }

    struct So2convMomentumSpectralInputs {
        momentum_grid: Array1<Real>,
        energy_grid: Array2<Real>,
        extrinsic_quasiparticle: Array2<Real>,
        extrinsic_satellite: Array2<Real>,
        interference_quasiparticle: Array2<Real>,
        interference_satellite: Array2<Real>,
        intrinsic_satellite: Array2<Real>,
        clipped_extrinsic_satellite: Array2<Real>,
        weights: Array2<Real>,
        self_energy_real: Array1<Real>,
        energy_correction: Array1<Real>,
        width: Array1<Real>,
        renormalization_real: Array1<Real>,
        renormalization_imag: Array1<Real>,
    }

    fn so2conv_momentum_spectral_inputs() -> So2convMomentumSpectralInputs {
        So2convMomentumSpectralInputs {
            momentum_grid: array![0.50, 1.00, 2.00, 4.00],
            energy_grid: array![
                [0.11, 0.12, 0.13, 0.14],
                [0.21, 0.22, 0.23, 0.24],
                [0.31, 0.32, 0.33, 0.34],
                [0.41, 0.42, 0.43, 0.44],
            ],
            extrinsic_quasiparticle: array![
                [1.11, 1.12, 1.13, 1.14],
                [1.21, 1.22, 1.23, 1.24],
                [1.31, 1.32, 1.33, 1.34],
                [1.41, 1.42, 1.43, 1.44],
            ],
            extrinsic_satellite: array![
                [2.22, 2.24, 2.26, 2.28],
                [2.42, 2.44, 2.46, 2.48],
                [2.62, 2.64, 2.66, 2.68],
                [2.82, 2.84, 2.86, 2.88],
            ],
            interference_quasiparticle: array![
                [3.33, 3.36, 3.39, 3.42],
                [3.63, 3.66, 3.69, 3.72],
                [3.93, 3.96, 3.99, 4.02],
                [4.23, 4.26, 4.29, 4.32],
            ],
            interference_satellite: array![
                [0.444, 0.448, 0.452, 0.456],
                [0.484, 0.488, 0.492, 0.496],
                [0.524, 0.528, 0.532, 0.536],
                [0.564, 0.568, 0.572, 0.576],
            ],
            intrinsic_satellite: array![
                [0.555, 0.560, 0.565, 0.570],
                [0.605, 0.610, 0.615, 0.620],
                [0.655, 0.660, 0.665, 0.670],
                [0.705, 0.710, 0.715, 0.720],
            ],
            clipped_extrinsic_satellite: array![
                [0.666, 0.672, 0.678, 0.684],
                [0.726, 0.732, 0.738, 0.744],
                [0.786, 0.792, 0.798, 0.804],
                [0.846, 0.852, 0.858, 0.864],
            ],
            weights: array![
                [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
                [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
                [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
                [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
            ],
            self_energy_real: array![41.0, 42.0, 43.0, 44.0],
            energy_correction: array![51.0, 52.0, 53.0, 54.0],
            width: array![61.0, 62.0, 63.0, 64.0],
            renormalization_real: array![71.0, 72.0, 73.0, 74.0],
            renormalization_imag: array![81.0, 82.0, 83.0, 84.0],
        }
    }

    fn so2conv_momentum_spectral_input<'a>(
        inputs: &'a So2convMomentumSpectralInputs,
        photoelectron_momentum: Real,
    ) -> SfconvMomentumSpectralInterpolationInput<'a> {
        SfconvMomentumSpectralInterpolationInput {
            photoelectron_momentum,
            momentum_grid: inputs.momentum_grid.view(),
            energy_grid: inputs.energy_grid.view(),
            extrinsic_quasiparticle: inputs.extrinsic_quasiparticle.view(),
            extrinsic_satellite: inputs.extrinsic_satellite.view(),
            interference_quasiparticle: inputs.interference_quasiparticle.view(),
            interference_satellite: inputs.interference_satellite.view(),
            intrinsic_satellite: inputs.intrinsic_satellite.view(),
            clipped_extrinsic_satellite: inputs.clipped_extrinsic_satellite.view(),
            weights: inputs.weights.view(),
            self_energy_real: inputs.self_energy_real.view(),
            energy_correction: inputs.energy_correction.view(),
            width: inputs.width.view(),
            renormalization_real: inputs.renormalization_real.view(),
            renormalization_imag: inputs.renormalization_imag.view(),
        }
    }

    fn so2conv_photoelectron_momentum_inputs() -> (Array1<Real>, Array1<Real>) {
        let momentum = array![0.0, 0.35, -0.40, 0.82, 1.10, 1.45];
        let self_energy = array![0.090, 0.105, 0.120, 0.150, 0.190, 0.250];
        (momentum, self_energy)
    }

    fn so2conv_self_energy_material() -> SfconvSo2convMaterialParameters {
        SfconvSo2convMaterialParameters {
            core_hole_lifetime: 0.03,
            interstitial_potential: 0.0,
            chemical_potential_offset: 0.20,
            fermi_wave_number: 1.0,
            fermi_momentum: 1.0,
            fermi_energy: 0.50,
            electron_concentration: 0.08,
            plasma_frequency: 0.70,
            dispersion_parameter: 0.33,
            initial_photoelectron_energy: 0.50,
            initial_photoelectron_momentum: 1.0,
            accuracy: 1.0e-4,
        }
    }

    fn so2conv_xanes_preparation_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>, Array1<Real>)
    {
        let count = 22;
        let incident_energy = Array1::from_shape_fn(count, |index| {
            let i = index as Real + 1.0;
            0.2 + 0.13 * (i - 1.0) + 0.002 * ((i as usize) % 3) as Real
        });
        let excitation_energy = Array1::from_shape_fn(count, |index| {
            let i = index as Real + 1.0;
            -0.4 + 0.11 * (i - 1.0) + 0.001 * ((i as usize) % 4) as Real
        });
        let embedded_background = Array1::from_shape_fn(count, |index| {
            let i = index as Real + 1.0;
            1.0 + 0.015 * (i - 1.0) + 0.0008 * ((i as usize) % 2) as Real
        });
        let absorption = Array1::from_shape_fn(count, |index| {
            let i = index as Real + 1.0;
            embedded_background[index] + 0.04 * (0.31 * i).sin() + 0.002 * (i - 1.0)
        });
        (
            incident_energy,
            excitation_energy,
            absorption,
            embedded_background,
        )
    }

    struct So2convFeffPathInterpolationInputs {
        source_momentum: Array1<Real>,
        path_momentum: Array1<Real>,
        central_phase: Array1<Real>,
        effective_amplitude: Array1<Real>,
        effective_phase: Array1<Real>,
        reduction_factor: Array1<Real>,
        mean_free_path: Array1<Real>,
        interpolated_central_phase: Array1<Real>,
        interpolated_effective_amplitude: Array1<Real>,
        interpolated_effective_phase: Array1<Real>,
        interpolated_reduction_factor: Array1<Real>,
        interpolated_mean_free_path: Array1<Real>,
    }

    fn so2conv_feff_path_interpolation_inputs() -> So2convFeffPathInterpolationInputs {
        So2convFeffPathInterpolationInputs {
            source_momentum: array![0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
            path_momentum: array![0.25, 0.75, 1.25, 1.75],
            central_phase: array![0.10, 0.20, 0.10, 0.30],
            effective_amplitude: array![1.00, 1.40, 1.10, 1.80],
            effective_phase: array![0.50, 0.70, 0.60, 1.00],
            reduction_factor: array![0.80, 0.90, 0.85, 0.95],
            mean_free_path: array![6.00, 7.00, 8.00, 9.00],
            interpolated_central_phase: array![0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0],
            interpolated_effective_amplitude: array![
                0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0
            ],
            interpolated_effective_phase: array![
                0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0
            ],
            interpolated_reduction_factor: array![
                0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0
            ],
            interpolated_mean_free_path: array![0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0],
        }
    }

    fn so2conv_path_average_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let source_momentum = array![0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25];
        let amplitude_reduction = array![0.82, 0.84, 0.88, 0.91, 0.89, 0.86, 0.83];
        let phase_shift = array![0.05, 0.08, 0.13, 0.17, 0.14, 0.09, 0.02];
        (source_momentum, amplitude_reduction, phase_shift)
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn assert_real_slice_close(actual: &Array1<Real>, expected: &[Real], tolerance: Real) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_close(actual, expected, tolerance);
        }
    }

    fn assert_so2conv_material_close(
        actual: SfconvSo2convMaterialParameters,
        expected: SfconvSo2convMaterialParameters,
        tolerance: Real,
    ) {
        assert_close(
            actual.core_hole_lifetime,
            expected.core_hole_lifetime,
            tolerance,
        );
        assert_close(
            actual.interstitial_potential,
            expected.interstitial_potential,
            tolerance,
        );
        assert_close(
            actual.chemical_potential_offset,
            expected.chemical_potential_offset,
            tolerance,
        );
        assert_close(
            actual.fermi_wave_number,
            expected.fermi_wave_number,
            tolerance,
        );
        assert_close(actual.fermi_momentum, expected.fermi_momentum, tolerance);
        assert_close(actual.fermi_energy, expected.fermi_energy, tolerance);
        assert_close(
            actual.electron_concentration,
            expected.electron_concentration,
            tolerance,
        );
        assert_close(
            actual.plasma_frequency,
            expected.plasma_frequency,
            tolerance,
        );
        assert_close(
            actual.dispersion_parameter,
            expected.dispersion_parameter,
            tolerance,
        );
        assert_close(
            actual.initial_photoelectron_energy,
            expected.initial_photoelectron_energy,
            tolerance,
        );
        assert_close(
            actual.initial_photoelectron_momentum,
            expected.initial_photoelectron_momentum,
            tolerance,
        );
        assert_close(actual.accuracy, expected.accuracy, tolerance);
    }

    fn assert_momentum_spectral_close(
        actual: &SfconvMomentumSpectralInterpolation,
        expected_energy: &[Real; 4],
        expected_rows: &[[Real; 4]; 8],
        expected_weights: &[Real; 8],
        expected_self_energy: &[Real; 5],
    ) {
        assert_real_slice_close(&actual.energy, expected_energy, 1.0e-15);
        for (row, expected) in expected_rows.iter().enumerate() {
            assert_real_slice_close(
                &actual.spectral_function.row(row).to_owned(),
                expected,
                1.0e-15,
            );
        }
        assert_real_slice_close(&actual.weights, expected_weights, 1.0e-15);
        assert_close(actual.self_energy_real, expected_self_energy[0], 1.0e-15);
        assert_close(actual.energy_correction, expected_self_energy[1], 1.0e-15);
        assert_close(actual.width, expected_self_energy[2], 1.0e-15);
        assert_close(
            actual.renormalization_real,
            expected_self_energy[3],
            1.0e-15,
        );
        assert_close(
            actual.renormalization_imag,
            expected_self_energy[4],
            1.0e-15,
        );
    }

    fn assert_pole_close(actual: SfconvPole, expected: SfconvPole) {
        assert_close(actual.energy, expected.energy, 1.0e-15);
        assert_close(actual.weight, expected.weight, 1.0e-15);
        assert_close(actual.broadening, expected.broadening, 1.0e-15);
    }

    fn assert_q_limits_close(actual: SfconvQLimits, expected: SfconvQLimits, tolerance: Real) {
        assert_eq!(actual.count, expected.count);
        assert_close(actual.q1, expected.q1, tolerance);
        assert_close(actual.q2, expected.q2, tolerance);
        assert_close(actual.q3, expected.q3, tolerance);
    }

    fn assert_integral_close(
        actual: SfconvAdaptiveIntegral,
        expected: SfconvAdaptiveIntegral,
        tolerance: Real,
    ) {
        assert_close(actual.value, expected.value, tolerance);
        assert_close(
            actual.estimated_error,
            expected.estimated_error,
            tolerance.max(1.0e-12),
        );
        assert_eq!(actual.evaluations, expected.evaluations);
        assert_eq!(actual.max_regions, expected.max_regions);
    }

    fn mksat_reference_context() -> SfconvSatelliteContext {
        SfconvSatelliteContext {
            plasma_frequency: 0.62,
            pole_energy: 0.47,
            dispersion_parameter: 0.28,
            photoelectron_energy: 0.85,
            accuracy: 1.0e-4,
        }
    }

    fn mksat_reference_self_energy() -> SfconvSatelliteSelfEnergy {
        SfconvSatelliteSelfEnergy {
            on_shell_real: 0.12,
            width: 0.08,
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
            off_shell_real: 0.03,
            off_shell_imag: 0.025,
        }
    }

    fn senergies_reference_context(include_below_fermi: bool) -> SfconvSelfEnergyContext {
        SfconvSelfEnergyContext {
            fermi_energy: 0.50,
            fermi_momentum: 1.00,
            plasma_frequency: 0.62,
            pole_energy: 0.47,
            quasiparticle_energy: 0.91,
            photoelectron_momentum: (2.0_f64 * 0.85).sqrt(),
            accuracy: 1.0e-4,
            pole_broadening: 0.035,
            dispersion_parameter: 0.28,
            include_below_fermi,
        }
    }
}
