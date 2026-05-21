use ndarray::{Array2, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Real, RealVec, RootError};

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
