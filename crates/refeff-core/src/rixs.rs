//! FEFF RIXS numerical helpers.
//!
//! This module ports the small analytic and interpolation kernels from
//! `RIXS/kkint.f90`, `RIXS/doublelorentz.f90`, `RIXS/blinterp2d.f90`, and the
//! final spectrum assembly block in `RIXS/rixs.f90`. The Rust API uses
//! `ndarray` views for table inputs and reports structured errors instead of
//! terminating the process with Fortran `STOP`.

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use thiserror::Error;

use crate::angular::{AngularError, TransitionBMatrixInput, transition_b_matrix};
use crate::core_hole::{CoreHoleError, core_hole_quantum_numbers};
use crate::quadrature::{QuadratureError, csomm2};
use crate::{Complex, FEFF_HARTREE_EV, Real};

const BL_INTERP_TOLERANCE: Real = 1.0e-5;
const KKINT_PI: Real = 3_141_592_653.0 / 1_000_000_000.0;
const RIXS_DIPOLE_TRANSITION_COUNT: usize = 3;
const RIXS_RAW_CROSS_SECTION_CHANNEL_COUNT: usize = 2;
/// FEFF skips the final-energy convolution when `gam_exp(2) + gam_Edge` is below this value.
pub const FEFF_RIXS_FINAL_BROADENING_SKIP_WIDTH: Real = 0.01 / FEFF_HARTREE_EV;

/// Error returned by FEFF RIXS helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum RixsError {
    /// Scalar inputs must be finite real values.
    #[error("RIXS input {name} must be finite, got {value}")]
    NonFiniteReal { name: &'static str, value: Real },
    /// Complex table entries must have finite real and imaginary parts.
    #[error("RIXS complex input {name} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// Lorentzian widths must be positive and finite.
    #[error("RIXS width {name} must be positive and finite, got {value}")]
    InvalidWidth { name: &'static str, value: Real },
    /// Analytic integration bounds must be finite and increasing.
    #[error("RIXS integration interval must increase: lower={lower}, upper={upper}")]
    InvalidInterval { lower: Real, upper: Real },
    /// Bilinear interpolation needs at least two points on each axis.
    #[error("RIXS interpolation axis {axis} requires at least 2 points, got {len}")]
    InsufficientGrid { axis: &'static str, len: usize },
    /// The interpolation value table must cover the requested x/y grids.
    #[error(
        "RIXS interpolation table has shape ({rows}, {cols}) but needs at least ({required_rows}, {required_cols})"
    )]
    MatrixTooSmall {
        rows: usize,
        cols: usize,
        required_rows: usize,
        required_cols: usize,
    },
    /// FEFF assumes strictly increasing interpolation grids.
    #[error(
        "RIXS interpolation axis {axis} must increase at index {index}: previous={previous}, current={current}"
    )]
    NonIncreasingGrid {
        axis: &'static str,
        index: usize,
        previous: Real,
        current: Real,
    },
    /// FEFF `BLInterp2D` allows only a small tolerance outside the table.
    #[error(
        "RIXS interpolation coordinate {axis}={value} is outside [{min}, {max}] with tolerance {tolerance}"
    )]
    OutOfRange {
        axis: &'static str,
        value: Real,
        min: Real,
        max: Real,
        tolerance: Real,
    },
    /// Duplicate adjacent grid points make the bilinear denominator zero.
    #[error("RIXS interpolation axis {axis} has a zero-width interval at index {index}")]
    ZeroInterval { axis: &'static str, index: usize },
    /// FEFF RIXS output assembly needs at least one spectrum channel.
    #[error("RIXS final spectrum requires at least one channel")]
    EmptyChannelTable,
    /// FEFF RIXS output assembly expects a square energy-by-energy table for each channel.
    #[error(
        "RIXS final spectrum table has shape ({rows}, {columns}, {channels}) but expected ({energy_count}, {energy_count}, channels)"
    )]
    FinalSpectrumShape {
        energy_count: usize,
        rows: usize,
        columns: usize,
        channels: usize,
    },
    /// FEFF Hartree-to-eV conversion must be positive and finite.
    #[error("RIXS Hartree/eV conversion must be positive and finite, got {value}")]
    InvalidHartreeEv { value: Real },
    /// FEFF RIXS edge accumulation needs at least one spectrum channel.
    #[error("RIXS edge contribution table requires at least one channel")]
    EmptyEdgeContributionChannelTable,
    /// FEFF RIXS edge accumulation expects one square energy table per channel and edge.
    #[error(
        "RIXS edge contribution table has shape ({rows}, {columns}, {channels}, {edges}) but expected ({energy_count}, {energy_count}, channels, {edge_count})"
    )]
    EdgeContributionShape {
        energy_count: usize,
        edge_count: usize,
        rows: usize,
        columns: usize,
        channels: usize,
        edges: usize,
    },
    /// FEFF RIXS edge broadening needs at least one final edge.
    #[error("RIXS edge broadening requires at least one edge")]
    EmptyEdgeBroadeningEdgeTable,
    /// FEFF RIXS edge broadening width/amplitude arrays must align.
    #[error(
        "RIXS edge broadening length mismatch: widths={width_count}, amplitudes={amplitude_count}"
    )]
    EdgeBroadeningLengthMismatch {
        width_count: usize,
        amplitude_count: usize,
    },
    /// FEFF RIXS final-energy broadening needs at least one spectrum channel.
    #[error("RIXS final-energy broadening table requires at least one channel")]
    EmptyFinalBroadeningChannelTable,
    /// FEFF RIXS final-energy broadening expects a square energy table for each channel.
    #[error(
        "RIXS final-energy broadening table has shape ({rows}, {columns}, {channels}) but expected ({energy_count}, {energy_count}, channels)"
    )]
    FinalBroadeningShape {
        energy_count: usize,
        rows: usize,
        columns: usize,
        channels: usize,
    },
    /// FEFF RIXS raw cross-section assembly needs at least one transition.
    #[error("RIXS raw cross-section assembly requires at least one transition")]
    EmptyCrossSectionTransitionTable,
    /// FEFF RIXS raw cross-section assembly needs at least one spin channel.
    #[error("RIXS raw cross-section assembly requires at least one spin channel, got {count}")]
    InvalidSpinChannelCount { count: usize },
    /// FEFF RIXS raw cross-section transition metadata must align across tables.
    #[error(
        "RIXS raw cross-section transition mismatch: l values={transition_count}, amplitudes={amplitude_transitions}, phases={phase_transitions}"
    )]
    CrossSectionTransitionMismatch {
        transition_count: usize,
        amplitude_transitions: usize,
        phase_transitions: usize,
    },
    /// FEFF RIXS raw cross-section energy dimensions must align.
    #[error(
        "RIXS raw cross-section energy shape mismatch: amplitudes incident={incident_count}, transfer={transfer_count}, green energies={green_energy}, phase energies={phase_energy}"
    )]
    CrossSectionEnergyShape {
        incident_count: usize,
        transfer_count: usize,
        green_energy: usize,
        phase_energy: usize,
    },
    /// FEFF RIXS raw cross-section angular dimensions must cover all active `l` values.
    #[error(
        "RIXS raw cross-section angular shape needs {required} channel(s), got amplitudes={amplitude_angular}, green=({green_rows}, {green_columns})"
    )]
    CrossSectionAngularShape {
        required: usize,
        amplitude_angular: usize,
        green_rows: usize,
        green_columns: usize,
    },
    /// FEFF RIXS raw cross-section angular momentum is too large for indexing.
    #[error("RIXS raw cross-section angular momentum {value} is too large")]
    InvalidAngularMomentum { value: isize },
    /// FEFF RIXS initial amplitude assembly needs at least one transition.
    #[error("RIXS initial amplitude assembly requires at least one transition")]
    EmptyInitialAmplitudeTransitionTable,
    /// FEFF RIXS initial amplitude transition metadata must align across tables.
    #[error(
        "RIXS initial amplitude transition mismatch: l values={transition_count}, radial overlaps={radial_transitions}, rkk={moment_transitions}, phases={phase_transitions}"
    )]
    InitialAmplitudeTransitionMismatch {
        transition_count: usize,
        radial_transitions: usize,
        moment_transitions: usize,
        phase_transitions: usize,
    },
    /// FEFF RIXS initial amplitude energy dimensions must align.
    #[error(
        "RIXS initial amplitude energy shape mismatch: radial incident={incident_count}, transfer={transfer_count}, rkk={moment_energy}, phases={phase_energy}, green energies={green_energy}, xsnorm={normalization_energy}"
    )]
    InitialAmplitudeEnergyShape {
        incident_count: usize,
        transfer_count: usize,
        moment_energy: usize,
        phase_energy: usize,
        green_energy: usize,
        normalization_energy: usize,
    },
    /// FEFF RIXS initial amplitude angular dimensions must cover all active `l` values.
    #[error(
        "RIXS initial amplitude angular shape needs {required} channel(s), got green=({green_rows}, {green_columns})"
    )]
    InitialAmplitudeAngularShape {
        required: usize,
        green_rows: usize,
        green_columns: usize,
    },
    /// FEFF RIXS initial amplitude assembly needs at least one spin channel.
    #[error("RIXS initial amplitude assembly requires at least one spin channel")]
    EmptyInitialAmplitudeSpinTable,
    /// FEFF RIXS incident-amplitude convolution needs at least one spin channel.
    #[error("RIXS incident-amplitude convolution requires at least one spin channel")]
    EmptyIncidentConvolutionSpinTable,
    /// FEFF RIXS direct final-transition setup needs at least one transition.
    #[error("RIXS direct final-transition setup requires at least one transition")]
    EmptyDirectFinalTransitionTransitionTable,
    /// FEFF RIXS direct final-transition setup needs at least one spin channel.
    #[error("RIXS direct final-transition setup requires at least one spin channel")]
    EmptyDirectFinalTransitionSpinTable,
    /// FEFF RIXS direct final-transition metadata must align across tables.
    #[error(
        "RIXS direct final-transition mismatch: l values={transition_count}, rkk={moment_transitions}, phases={phase_transitions}"
    )]
    DirectFinalTransitionMismatch {
        transition_count: usize,
        moment_transitions: usize,
        phase_transitions: usize,
    },
    /// FEFF RIXS direct final-transition energy dimensions must align.
    #[error(
        "RIXS direct final-transition energy mismatch: relative={energy_count}, rkk={moment_energy}, phases={phase_energy}, xsnorm={normalization_energy}"
    )]
    DirectFinalTransitionEnergyShape {
        energy_count: usize,
        moment_energy: usize,
        phase_energy: usize,
        normalization_energy: usize,
    },
    /// FEFF RIXS incident-amplitude convolution transition metadata must align.
    #[error(
        "RIXS incident-amplitude convolution transition mismatch: l values={transition_count}, amplitudes={amplitude_transitions}, rkk={moment_transitions}, phases={phase_transitions}, bmat={bmat_transitions}"
    )]
    IncidentConvolutionTransitionMismatch {
        transition_count: usize,
        amplitude_transitions: usize,
        moment_transitions: usize,
        phase_transitions: usize,
        bmat_transitions: usize,
    },
    /// FEFF RIXS incident-amplitude convolution energy dimensions must align.
    #[error(
        "RIXS incident-amplitude convolution energy mismatch: amplitudes incident={incident_count}, transfer={transfer_count}, rkk={moment_energy}, phases={phase_energy}, k2={wave_energy}, xsnorm={normalization_energy}"
    )]
    IncidentConvolutionEnergyShape {
        incident_count: usize,
        transfer_count: usize,
        moment_energy: usize,
        phase_energy: usize,
        wave_energy: usize,
        normalization_energy: usize,
    },
    /// FEFF RIXS incident-amplitude convolution angular dimensions must cover all active `l` values.
    #[error(
        "RIXS incident-amplitude convolution angular shape needs {required} channel(s), got amplitudes={amplitude_angular}, bmat={bmat_angular}"
    )]
    IncidentConvolutionAngularShape {
        required: usize,
        amplitude_angular: usize,
        bmat_angular: usize,
    },
    /// FEFF RIXS incident-amplitude convolution spin dimensions must align.
    #[error(
        "RIXS incident-amplitude convolution spin mismatch: rkk={moment_spin}, bmat={bmat_spin}"
    )]
    IncidentConvolutionSpinShape {
        moment_spin: usize,
        bmat_spin: usize,
    },
    /// FEFF RIXS transition-matrix setup accepts only the two FEFF spin axes.
    #[error("RIXS transition-matrix setup expects 1 or 2 spin channels, got {count}")]
    InvalidTransitionSpinChannelCount { count: usize },
    /// FEFF `setkap` failed while preparing the RIXS transition matrix.
    #[error("RIXS transition-matrix core-hole setup failed: {source}")]
    TransitionCoreHoleSetup { source: CoreHoleError },
    /// FEFF `bcoef` failed while preparing the RIXS transition matrix.
    #[error("RIXS transition-matrix B-coefficient setup failed: {source}")]
    TransitionBMatrixSetup { source: AngularError },
    /// FEFF `bcoef` did not provide a requested diagonal transition entry.
    #[error(
        "RIXS transition-matrix diagonal missing for m={magnetic}, spin={spin}, transition={transition}"
    )]
    TransitionBMatrixDiagonalMissing {
        magnetic: isize,
        spin: usize,
        transition: usize,
    },
    /// FEFF RIXS transition phase-shift selection needs at least one energy row.
    #[error("RIXS transition phase-shift table requires at least one energy row")]
    EmptyTransitionPhaseShiftEnergyTable,
    /// FEFF RIXS transition phase-shift selection needs at least one signed-l column.
    #[error("RIXS transition phase-shift table requires at least one signed-l column")]
    EmptyTransitionPhaseShiftAngularTable,
    /// FEFF RIXS transition phase-shift selection needs at least one transition label.
    #[error("RIXS transition phase-shift selection requires at least one transition")]
    EmptyTransitionPhaseShiftTransitionTable,
    /// FEFF RIXS transition phase-shift signed-l lookup is outside the supplied table.
    #[error(
        "RIXS transition phase-shift transition {transition} needs signed-l {signed_l}, outside [{min_signed_l}, {max_signed_l}]"
    )]
    TransitionPhaseShiftAngularOutOfRange {
        transition: usize,
        signed_l: isize,
        min_signed_l: isize,
        max_signed_l: isize,
    },
    /// FEFF `xsnorm` enters a square root and must not be negative.
    #[error("RIXS normalization {index} must be non-negative, got {value}")]
    NegativeNormalization { index: usize, value: Real },
    /// FEFF crossing branch references the point after the threshold interval.
    #[error("RIXS threshold crossing at interval {interval} requires another energy point")]
    IncidentConvolutionThresholdAtLastInterval { interval: usize },
    /// FEFF RIXS incident-energy broadening needs at least one spectrum channel.
    #[error("RIXS incident-energy broadening table requires at least one channel")]
    EmptyIncidentBroadeningChannelTable,
    /// FEFF RIXS incident-energy broadening expects a square energy table and aligned self-energy.
    #[error(
        "RIXS incident-energy broadening shape mismatch: xsect=({rows}, {columns}, {channels}), self-energy={self_energy_count}, expected ({energy_count}, {energy_count}, channels)"
    )]
    IncidentBroadeningShape {
        energy_count: usize,
        rows: usize,
        columns: usize,
        channels: usize,
        self_energy_count: usize,
    },
    /// FEFF RIXS satellite convolution needs at least one spectrum channel.
    #[error("RIXS satellite convolution table requires at least one channel")]
    EmptySatelliteConvolutionChannelTable,
    /// FEFF RIXS satellite convolution expects a square RIXS table.
    #[error(
        "RIXS satellite convolution table has shape ({rows}, {columns}, {channels}) but expected ({energy_count}, {energy_count}, channels)"
    )]
    SatelliteConvolutionShape {
        energy_count: usize,
        rows: usize,
        columns: usize,
        channels: usize,
    },
    /// FEFF RIXS satellite convolution needs at least two XES points.
    #[error("RIXS satellite convolution requires at least two XES points, got {count}")]
    InsufficientSatelliteXesGrid { count: usize },
    /// FEFF RIXS satellite convolution XES energy and intensity arrays must align.
    #[error("RIXS satellite convolution XES length mismatch: energy={energy_count}, mu={mu_count}")]
    SatelliteXesLengthMismatch {
        energy_count: usize,
        mu_count: usize,
    },
    /// FEFF RIXS pole normalization needs the incident pole plus at least one final pole.
    #[error("RIXS pole normalization requires at least two pole rows, got {count}")]
    InsufficientPoleRows { count: usize },
    /// FEFF RIXS pole normalization arrays must align.
    #[error(
        "RIXS pole normalization length mismatch: energy={energy_count}, amplitude={amplitude_count}, width={width_count}"
    )]
    PoleLengthMismatch {
        energy_count: usize,
        amplitude_count: usize,
        width_count: usize,
    },
    /// FEFF RIXS radial overlap assembly needs at least one transition.
    #[error("RIXS radial overlap assembly requires at least one transition")]
    EmptyRadialOverlapTransitionTable,
    /// FEFF RIXS screened-core potential setup needs at least one radial point.
    #[error("RIXS core-hole potential table requires at least one radial point")]
    EmptyCoreHolePotentialTable,
    /// FEFF RIXS screened-core potential tables must have aligned radial dimensions.
    #[error(
        "RIXS core-hole potential length mismatch: initial={initial_count}, final={final_count}"
    )]
    CoreHolePotentialLengthMismatch {
        initial_count: usize,
        final_count: usize,
    },
    /// FEFF RIXS radial setup must keep the active muffin-tin grid inside the radial table.
    #[error(
        "RIXS radial grid active point count {active_point_count} exceeds radial point count {point_count}"
    )]
    RadialGridActivePointCount {
        active_point_count: usize,
        point_count: usize,
    },
    /// FEFF RIXS radial-function record count must match the nested energy/angular read loop.
    #[error(
        "RIXS radial-function record count mismatch: records={record_count}, expected={expected_count}"
    )]
    RadialFunctionRecordCountMismatch {
        record_count: usize,
        expected_count: usize,
    },
    /// FEFF RIXS radial-function table dimensions must be positive.
    #[error("RIXS radial-function {axis} count must be positive, got {count}")]
    EmptyRadialFunctionTable { axis: &'static str, count: usize },
    /// FEFF RIXS radial-function records must match the declared radial table length.
    #[error(
        "RIXS radial-function record {record} has {value_count} radial value(s), expected {radial_count}"
    )]
    RadialFunctionRecordShape {
        record: usize,
        value_count: usize,
        radial_count: usize,
    },
    /// FEFF RIXS radial-function angular labels must fit the allocated `Rl` table.
    #[error(
        "RIXS radial-function record {record} angular label {angular_momentum} is outside 0..{angular_count}"
    )]
    RadialFunctionAngularOutOfRange {
        record: usize,
        angular_momentum: isize,
        angular_count: usize,
    },
    /// FEFF RIXS radial overlap arrays must have aligned radial and energy dimensions.
    #[error(
        "RIXS radial overlap shape mismatch: energies={energy_count}, radii={radial_count}, potential={potential_count}, initial=({initial_radial}, {initial_angular}, {initial_energy}), final=({final_radial}, {final_angular}, {final_energy})"
    )]
    RadialOverlapShape {
        energy_count: usize,
        radial_count: usize,
        potential_count: usize,
        initial_radial: usize,
        initial_angular: usize,
        initial_energy: usize,
        final_radial: usize,
        final_angular: usize,
        final_energy: usize,
    },
    /// FEFF RIXS radial overlap angular tables must cover all active transition labels.
    #[error(
        "RIXS radial overlap angular shape needs {required} l channel(s), got initial={initial_angular}, final={final_angular}"
    )]
    RadialOverlapAngularShape {
        required: usize,
        initial_angular: usize,
        final_angular: usize,
    },
    /// FEFF RIXS radial overlap quadrature failed.
    #[error("RIXS radial overlap quadrature failed: {source}")]
    RadialOverlapQuadrature { source: QuadratureError },
    /// FEFF RIXS many-pole self-energy grid columns must align.
    #[error(
        "RIXS self-energy grid length mismatch: energy={energy_count}, self_energy={self_energy_count}"
    )]
    SelfEnergyGridLengthMismatch {
        energy_count: usize,
        self_energy_count: usize,
    },
}

/// Inputs for the final RIXS spectrum assembly block in FEFF `RIXS/rixs.f90`.
///
/// `cross_section` is indexed as `(energy_transfer, incident_energy, channel)`,
/// matching FEFF `xsect_tmp(iE2, iE1, ind)`. Energies and windows are in
/// Hartree; returned output energies are converted to eV using `hartree_ev`.
#[derive(Debug, Clone, Copy)]
pub struct RixsFinalSpectrumInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF `xsect_tmp(iE2, iE1, ind)` table.
    pub cross_section: ArrayView3<'a, Real>,
    /// FEFF incident-projection integration window `EMin(1):EMax(1)`.
    pub incident_window: (Real, Real),
    /// FEFF final-projection integration window `EMin(2):EMax(2)`.
    pub final_window: (Real, Real),
    /// FEFF `Edge1(1)`, incident-edge offset.
    pub incident_edge: Real,
    /// FEFF `Edge1(2)`, final-edge offset.
    pub final_edge: Real,
    /// Hartree-to-eV conversion constant.
    pub hartree_ev: Real,
}

/// Output arrays produced by FEFF `RIXS/rixs.f90` final spectrum assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsFinalSpectrum {
    /// `xasEI.dat` energy column.
    pub incident_xas_energy_ev: Array1<Real>,
    /// `xasEI.dat` channel table.
    pub incident_xas: Array2<Real>,
    /// `xasEF.dat` energy column.
    pub final_xas_energy_ev: Array1<Real>,
    /// `xasEF.dat` channel table.
    pub final_xas: Array2<Real>,
    /// `herfd.dat` energy column.
    pub herfd_energy_ev: Array1<Real>,
    /// `herfd.dat` channel table.
    pub herfd: Array2<Real>,
    /// `rixsET.dat` first energy column.
    pub rixs_et_first_energy_ev: Array1<Real>,
    /// `rixsET.dat` second energy column.
    pub rixs_et_second_energy_ev: Array1<Real>,
    /// `rixsET.dat` channel table.
    pub rixs_et: Array2<Real>,
    /// `rixsEE.dat` incident-energy column.
    pub rixs_ee_incident_energy_ev: Array1<Real>,
    /// `rixsEE.dat` emission-energy column.
    pub rixs_ee_emission_energy_ev: Array1<Real>,
    /// `rixsEE.dat` channel table.
    pub rixs_ee: Array2<Real>,
}

/// Inputs for the FEFF `RIXS/rixs.f90` multi-edge spectrum summation block.
///
/// `edge_contributions` is indexed as `(energy_transfer, incident_energy,
/// channel, edge)`, matching FEFF `xsect_rxs(iE2, iE1, ind, iEdge)`.
/// `edge_splits` must already be normalized the same way FEFF normalizes
/// `EdgeSplit(1:nEdge)` before the final edge-accumulation loop.
#[derive(Debug, Clone, Copy)]
pub struct RixsEdgeContributionInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF normalized edge shifts `EdgeSplit(1:nEdge)`.
    pub edge_splits: ArrayView1<'a, Real>,
    /// FEFF `xsect_rxs(iE2, iE1, ind, iEdge)` table.
    pub edge_contributions: ArrayView4<'a, Real>,
}

/// Inputs for the FEFF `RIXS/rixs.f90` per-edge broadening loop.
///
/// `raw_cross_section` is the first `xsect_rxs(:,:,:,1)` table before the
/// incident-energy convolution. FEFF duplicates that table across all final
/// edges, applies the same incident broadening to each edge, then applies a
/// final-energy convolution using `gam_exp(2) + gam_Edge(iEdge)` and
/// `EdgeAmp(iEdge)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsEdgeBroadeningInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// Raw FEFF `xsect_rxs(iE2, iE1, ind, 1)` before edge broadening.
    pub raw_cross_section: ArrayView3<'a, Real>,
    /// FEFF complex self-energy `Sigma(1:ne1)`.
    pub self_energy: ArrayView1<'a, Complex>,
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
    /// FEFF core-hole width `gam_ch`.
    pub core_width: Real,
    /// FEFF incident broadening width `gam_exp(1)`.
    pub incident_width: Real,
    /// FEFF final experimental width `gam_exp(2)`.
    pub final_width_base: Real,
    /// FEFF `gam_Edge(1:nEdge)` final-pole widths.
    pub edge_widths: ArrayView1<'a, Real>,
    /// FEFF `EdgeAmp(1:nEdge)` final-pole amplitudes.
    pub edge_amplitudes: ArrayView1<'a, Real>,
}

/// Inputs for the FEFF `RIXS/rixs.f90` post-raw cross-section pipeline.
///
/// This starts from the raw `xsect_rxs(:,:,:,1)` table, applies FEFF's per-edge
/// broadening loop, sums shifted edge contributions, and assembles the standard
/// `xasEI`, `xasEF`, HERFD, `rixsET`, and `rixsEE` spectra. Optional MBConv
/// satellite output is intentionally separate because FEFF rewrites the
/// satellite line-spectrum energy columns differently from the normal outputs.
#[derive(Debug, Clone, Copy)]
pub struct RixsPostRawSpectrumInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// Raw FEFF `xsect_rxs(iE2, iE1, ind, 1)` before edge broadening.
    pub raw_cross_section: ArrayView3<'a, Real>,
    /// FEFF complex self-energy `Sigma(1:ne1)`.
    pub self_energy: ArrayView1<'a, Complex>,
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
    /// FEFF core-hole width `gam_ch`.
    pub core_width: Real,
    /// FEFF incident broadening width `gam_exp(1)`.
    pub incident_width: Real,
    /// FEFF final experimental width `gam_exp(2)`.
    pub final_width_base: Real,
    /// FEFF normalized edge shifts `EdgeSplit(1:nEdge)`.
    pub edge_splits: ArrayView1<'a, Real>,
    /// FEFF `gam_Edge(1:nEdge)` final-pole widths.
    pub edge_widths: ArrayView1<'a, Real>,
    /// FEFF `EdgeAmp(1:nEdge)` final-pole amplitudes.
    pub edge_amplitudes: ArrayView1<'a, Real>,
    /// FEFF incident integration window `(EMin(1), EMax(1))`.
    pub incident_window: (Real, Real),
    /// FEFF final integration window `(EMin(2), EMax(2))`.
    pub final_window: (Real, Real),
    /// FEFF `Edge1(1)` incident-edge offset.
    pub incident_edge: Real,
    /// FEFF `Edge1(2)` final-edge offset.
    pub final_edge: Real,
    /// Hartree-to-eV conversion factor.
    pub hartree_ev: Real,
}

/// FEFF RIXS standard spectra assembled from a raw cross-section table.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsPostRawSpectrum {
    /// FEFF `xsect_rxs(iE2, iE1, ind, iEdge)` after per-edge broadening.
    pub edge_contributions: Array4<Real>,
    /// FEFF `xsect_tmp(iE2, iE1, ind)` after shifted edge summation.
    pub summed_cross_section: Array3<Real>,
    /// FEFF standard non-satellite output spectra.
    pub spectrum: RixsFinalSpectrum,
}

/// Inputs for the FEFF `RIXS/rixs.f90` `MBConv` satellite spectrum block.
///
/// This starts from the standard summed `xsect_tmp(iE2, iE1, ind)` table,
/// applies the FEFF XES satellite convolution, and assembles the
/// `*-sat.dat` spectrum outputs.
#[derive(Debug, Clone, Copy)]
pub struct RixsSatelliteSpectrumInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF `xsect_tmp(iE2, iE1, ind)` before `MBConv`.
    pub cross_section: ArrayView3<'a, Real>,
    /// FEFF `XES/xmu.dat` energy column, in eV.
    pub xes_energy_ev: ArrayView1<'a, Real>,
    /// FEFF `XES/xmu.dat` intensity column.
    pub xes_mu: ArrayView1<'a, Real>,
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
    /// FEFF incident integration window `(EMin(1), EMax(1))`.
    pub incident_window: (Real, Real),
    /// FEFF final integration window `(EMin(2), EMax(2))`.
    pub final_window: (Real, Real),
    /// FEFF `Edge1(1)` incident-edge offset.
    pub incident_edge: Real,
    /// FEFF `Edge1(2)` final-edge offset.
    pub final_edge: Real,
    /// Hartree-to-eV conversion factor.
    pub hartree_ev: Real,
}

/// FEFF RIXS satellite spectra assembled from a standard cross-section table.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsSatelliteSpectrum {
    /// FEFF `xsect_tmp(iE2, iE1, ind)` after `MBConv`.
    pub satellite_cross_section: Array3<Real>,
    /// FEFF `*-sat.dat` output spectra.
    pub spectrum: RixsFinalSpectrum,
}

/// Inputs for FEFF `RIXS/rixs.f90` `edges.dat` pole normalization.
///
/// FEFF reads `edges.dat` as `(EdgeSplit, EdgeAmp, gam_Edge)`, treats the first
/// row as the incident edge/core width, reverses the remaining rows, replaces
/// non-positive final-edge energies with `-xmu`, and shifts the active edge
/// columns exactly as the `ReadPoles` block does.
#[derive(Debug, Clone, Copy)]
pub struct RixsPoleNormalizationInput<'a> {
    /// FEFF `edges.dat` `emu` column.
    pub pole_energies: ArrayView1<'a, Real>,
    /// FEFF `edges.dat` `M_kk` column.
    pub pole_amplitudes: ArrayView1<'a, Real>,
    /// FEFF `edges.dat` `gam` column.
    pub pole_widths: ArrayView1<'a, Real>,
    /// FEFF Fermi level `xmu`, in Hartree.
    pub fermi_level: Real,
}

/// Normalized pole data produced by FEFF `RIXS/rixs.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsPoleNormalization {
    /// FEFF `Edge1(1)` after the `ReadPoles` shift.
    pub incident_edge: Real,
    /// FEFF `Edge1(2)` after the `ReadPoles` shift.
    pub final_edge: Real,
    /// FEFF normalized `EdgeSplit(1:nEdge)` after reversing and shifting.
    pub edge_splits: Array1<Real>,
    /// FEFF `EdgeAmp(1:nEdge)` after dropping the incident row and reversing.
    pub edge_amplitudes: Array1<Real>,
    /// FEFF `gam_Edge(1:nEdge)` after dropping the incident row and reversing.
    pub edge_widths: Array1<Real>,
    /// FEFF `gam_ch`, taken from the first pole width.
    pub core_width: Real,
}

/// FEFF RIXS pole data for the `ReadPoles = F` branch.
///
/// `RIXS/rixs.f90` sets `nEdge = 1`, `EdgeSplit(1) = 0`, and
/// `EdgeAmp(1) = 1` when `edges.dat` is not used. The legacy code leaves the
/// edge offsets implicit; Rust makes them deterministic zeros while retaining
/// `gam_ch` from `rixs.inp` as the incident core-hole width.
pub fn rixs_default_pole_normalization(
    core_width: Real,
) -> Result<RixsPoleNormalization, RixsError> {
    validate_width("core_width", core_width)?;
    Ok(RixsPoleNormalization {
        incident_edge: 0.0,
        final_edge: 0.0,
        edge_splits: Array1::from_vec(vec![0.0]),
        edge_amplitudes: Array1::from_vec(vec![1.0]),
        edge_widths: Array1::from_vec(vec![0.0]),
        core_width,
    })
}

/// Inputs for the FEFF `RIXS/rixs.f90` final-energy convolution of one edge.
///
/// `edge_cross_section` is indexed as `(energy_transfer, incident_energy,
/// channel)`, matching one `xsect_rxs(iE2, iE1, ind, iEdge)` slice. The
/// `final_width` value is FEFF `gam_exp(2) + gam_Edge(iEdge)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsFinalEnergyBroadeningInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// One FEFF `xsect_rxs(iE2, iE1, ind, iEdge)` edge slice.
    pub edge_cross_section: ArrayView3<'a, Real>,
    /// FEFF core-hole width `gam_ch`.
    pub core_width: Real,
    /// FEFF final broadening width `gam_exp(2) + gam_Edge(iEdge)`.
    pub final_width: Real,
    /// FEFF pole amplitude `EdgeAmp(iEdge)`.
    pub edge_amplitude: Real,
}

/// Inputs for the FEFF `RIXS/rixs.f90` incident-energy convolution of one edge.
///
/// `edge_cross_section` is indexed as `(energy_transfer, incident_energy,
/// channel)`, matching one `xsect_rxs(iE2, iE1, ind, iEdge)` slice. The
/// `incident_width` value is FEFF `gam_exp(1)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsIncidentEnergyBroadeningInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// One FEFF `xsect_rxs(iE2, iE1, ind, iEdge)` edge slice.
    pub edge_cross_section: ArrayView3<'a, Real>,
    /// FEFF complex self-energy `Sigma(1:ne1)`.
    pub self_energy: ArrayView1<'a, Complex>,
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
    /// FEFF incident broadening width `gam_exp(1)`.
    pub incident_width: Real,
}

/// Inputs for FEFF `RIXS/rixs.f90` `mpse.dat` self-energy preparation.
///
/// FEFF reads `mpse.dat` energy and complex self-energy columns in eV, converts
/// both to Hartree, and maps the many-pole table onto the RIXS `rem(1:ne1)`
/// grid before incident-energy broadening.
#[derive(Debug, Clone, Copy)]
pub struct RixsSelfEnergyGridInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`, in Hartree.
    pub relative_energies: ArrayView1<'a, Real>,
    /// `mpse.dat` photoelectron energy column, in eV relative to the Fermi level.
    pub mpse_energy_ev: ArrayView1<'a, Real>,
    /// `mpse.dat` complex self-energy column, in eV.
    pub mpse_self_energy_ev: ArrayView1<'a, Complex>,
    /// FEFF Fermi level `xmu`, in Hartree.
    pub fermi_level: Real,
    /// Hartree-to-eV conversion constant.
    pub hartree_ev: Real,
}

/// Inputs for the FEFF `RIXS/rixs.f90` incident/final wave-number setup.
///
/// FEFF computes `k1(iE1)` from the complex incident reference energy
/// `eref(1,nspx)`, while `k2(iE2)` uses only `DBLE(eref_2(1,nspx))` from the
/// final-state reference energy.
#[derive(Debug, Clone, Copy)]
pub struct RixsWaveNumberInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF incident-state reference energy `eref(1,nspx)`.
    pub incident_reference_energy: Complex,
    /// FEFF final-state reference energy `eref_2(1,nspx)`.
    pub final_reference_energy: Complex,
}

/// Incident and final wave numbers produced by FEFF `RIXS/rixs.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsWaveNumbers {
    /// FEFF `k1(1:ne1)` incident-state wave numbers.
    pub incident_wave_numbers: Array1<Complex>,
    /// FEFF `k2(1:ne1)` final-state wave numbers.
    pub final_wave_numbers: Array1<Complex>,
}

/// Inputs for the FEFF `RIXS/rixs.f90` logarithmic radial-grid setup.
///
/// FEFF fills `ri(ir) = exp(-x0_1 + dx1*(ir-1))`, then computes `imt` and
/// `jmt=imt+1` for the active muffin-tin integration range.
#[derive(Debug, Clone, Copy)]
pub struct RixsRadialGridInput {
    /// FEFF radial table length `nrptx`.
    pub point_count: usize,
    /// FEFF logarithmic-grid origin `x0_1`.
    pub log_origin: Real,
    /// FEFF logarithmic-grid step `dx1`.
    pub log_step: Real,
    /// FEFF muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
}

/// Logarithmic radial grid produced by FEFF `RIXS/rixs.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsRadialGrid {
    /// FEFF `ri(1:nrptx)` logarithmic radial grid.
    pub radii: Array1<Real>,
    /// FEFF one-based `imt` value after integer truncation.
    pub muffin_tin_index_fortran: usize,
    /// FEFF active integration count `jmt = imt + 1`.
    pub active_point_count: usize,
}

/// Inputs for FEFF `RIXS/rixs.f90` screened-core potential setup.
///
/// For ordinary final edges FEFF computes `DeltaV = -vch1 + vch2`. For the
/// `VAL` final-edge branch it sets `vch2 = 0`, so `DeltaV = -vch1`.
#[derive(Debug, Clone, Copy)]
pub struct RixsCoreHolePotentialInput<'a> {
    /// FEFF incident-edge screened core-hole potential `vch1`.
    pub incident_screened_core_hole: ArrayView1<'a, Real>,
    /// FEFF final-edge screened core-hole potential `vch2`; `None` is the `VAL` branch.
    pub final_screened_core_hole: Option<ArrayView1<'a, Real>>,
}

/// One radial-function record read from FEFF `rl.dat` by `RIXS/rixs.f90`.
#[derive(Debug, Clone, Copy)]
pub struct RixsRadialFunctionRecord<'a> {
    /// FEFF record energy written into `em(iE)` or `em_2(iE)`.
    pub energy: Complex,
    /// FEFF `llltmp` angular label selecting the `Rl(:, llltmp, iE)` column.
    pub angular_momentum: isize,
    /// FEFF radial function values for one `(energy, angular)` record.
    pub radial_values: ArrayView1<'a, Complex>,
}

/// Inputs for assembling FEFF RIXS radial-function tables from `rl.dat` records.
#[derive(Debug, Clone, Copy)]
pub struct RixsRadialFunctionTableInput<'a> {
    /// Records in the nested FEFF read-loop order: energy outer, angular record inner.
    pub records: &'a [RixsRadialFunctionRecord<'a>],
    /// FEFF active RIXS energy count `ne1`.
    pub energy_count: usize,
    /// Number of non-negative angular labels allocated in Rust.
    pub angular_count: usize,
    /// FEFF radial record length `jri`.
    pub radial_count: usize,
}

/// Radial functions assembled in the layout consumed by RIXS radial overlaps.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsRadialFunctionTable {
    /// FEFF `em(1:ne1)` after the radial-function read loop.
    pub energies: Array1<Complex>,
    /// FEFF `Rl(radial, l, iE)` in Rust axis order `(radial, angular, energy)`.
    pub radial_functions: Array3<Complex>,
}

/// Inputs for the FEFF `RIXS/rixs.f90` `setkap`/`bcoef` transition setup.
///
/// This mirrors the block that calls `setkap(ihole, kinit, linit)` and then
/// `bcoef(...)`. The returned diagonal table is in the flattened angular
/// layout used by RIXS `TLb` arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RixsTransitionMatrixInput {
    /// Maximum orbital momentum `lx`.
    pub lmax: usize,
    /// FEFF core-hole index `ihole`.
    pub hole: i32,
    /// FEFF polarization selector `ipol`.
    pub polarization: i32,
    /// FEFF polarization tensor `ptz(-1:1,-1:1)`, indexed as `[p+1][p'+1]`.
    pub polarization_tensor: [[Complex; 3]; 3],
    /// FEFF multipole selector `le2`.
    pub multipole: i32,
    /// FEFF `ltrace` flag.
    pub trace_orbital: bool,
    /// FEFF spin selector `ispin`.
    pub spin: i32,
    /// FEFF spin-channel loop count `nspx`.
    pub spin_channel_count: usize,
    /// FEFF angle `angks` between x-ray k-vector and spin vector.
    pub spin_vector_angle: Real,
}

/// Transition metadata and diagonal B-matrix entries used by FEFF RIXS.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsTransitionMatrixSetup {
    /// FEFF `kinit` from `setkap`.
    pub initial_kappa: i32,
    /// FEFF `linit` from `setkap`.
    pub initial_angular_momentum: i32,
    /// FEFF `knd(1:8)` transition kappa labels.
    pub transition_kappas: [i32; 8],
    /// FEFF `lnd(1:8)` transition angular momenta.
    pub transition_angular_momenta: [isize; 8],
    /// FEFF diagonal `bmat(m,is,ind,m,is,ind)` in flattened angular layout.
    pub b_matrix_diagonal: Array3<Complex>,
}

/// Inputs for selecting FEFF RIXS per-transition phase shifts.
///
/// FEFF repeatedly indexes phase shifts as `ph(iE, -lnd(ind), 1, 0)`. The
/// input table is indexed as `(energy, signed_l - signed_l_min)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsTransitionPhaseShiftInput<'a> {
    /// FEFF signed-l phase shifts for one spin/potential branch.
    pub phase_shifts: ArrayView2<'a, Complex>,
    /// Signed-l value stored in column zero of `phase_shifts`.
    pub signed_l_min: isize,
    /// FEFF `lnd(ind)` transition angular momentum labels.
    pub transition_angular_momenta: &'a [isize],
}

/// Inputs for the FEFF `RIXS/rixs.f90` `MBConv` satellite convolution.
///
/// `cross_section` is indexed as `(energy_transfer, incident_energy, channel)`,
/// matching FEFF `xsect_tmp(iE2, iE1, ind)`. `xes_energy_ev` and `xes_mu`
/// correspond to the columns FEFF reads from `XES/xmu.dat`; energies are
/// converted internally with `xmu - energy_ev / hartree_ev` and reversed before
/// integration, matching the Fortran block.
#[derive(Debug, Clone, Copy)]
pub struct RixsSatelliteConvolutionInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF `xsect_tmp(iE2, iE1, ind)` table before `MBConv`.
    pub cross_section: ArrayView3<'a, Real>,
    /// FEFF `XES/xmu.dat` energy column, in eV.
    pub xes_energy_ev: ArrayView1<'a, Real>,
    /// FEFF `XES/xmu.dat` spectral weights.
    pub xes_mu: ArrayView1<'a, Real>,
    /// FEFF Fermi level `xmu`, in Hartree.
    pub fermi_level: Real,
    /// Hartree-to-eV conversion constant.
    pub hartree_ev: Real,
}

/// Inputs for FEFF `RIXS/rixs.f90` raw `xsect_rxs` assembly.
///
/// This corresponds to the block that folds `TLb` amplitudes with the final
/// Green-function diagonal and final-core-hole phase shifts before experimental
/// broadening. `transition_amplitudes` is indexed as `(incident_energy,
/// energy_transfer, angular_channel, transition)`, matching FEFF
/// `TLb(iE1, iE2, L, ind)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsRawCrossSectionInput<'a> {
    /// FEFF `TLb(iE1, iE2, L, ind)` transition amplitudes.
    pub transition_amplitudes: ArrayView4<'a, Complex>,
    /// FEFF `gg2(L1, L2, iph, iE2)` for the selected potential `iph`.
    pub final_green: ArrayView3<'a, Complex>,
    /// FEFF `ph_2(iE2, -lnd(ind), 1, 0)`, preselected per transition.
    pub final_phase_shifts: ArrayView2<'a, Complex>,
    /// FEFF `lnd(ind)` transition angular momentum labels.
    pub transition_angular_momenta: &'a [isize],
    /// FEFF `nspx` spin-channel loop count.
    pub spin_channel_count: usize,
}

/// Inputs for FEFF `RIXS/rixs.f90` radial overlap assembly.
///
/// This ports the `rInt = Rl * DeltaV * Rl_2; CALL csomm2(...)` block that
/// computes the real radial factor later consumed by the initial `TLb`
/// assembly. Radial functions are indexed as `(radial_point, l, energy)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsRadialOverlapInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// Active FEFF logarithmic radial grid `ri(1:jmt)`.
    pub radii: ArrayView1<'a, Real>,
    /// FEFF `Rl(1:jmt, l, iE1)` initial-core radial functions.
    pub initial_radial_functions: ArrayView3<'a, Complex>,
    /// FEFF `Rl_2(1:jmt, l, iE2)` final-core radial functions.
    pub final_radial_functions: ArrayView3<'a, Complex>,
    /// FEFF screened-core potential difference `DeltaV(1:jmt)`.
    pub potential_difference: ArrayView1<'a, Real>,
    /// FEFF `lnd(ind)` transition angular momentum labels.
    pub transition_angular_momenta: &'a [isize],
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
    /// FEFF logarithmic radial-grid step `dx1`.
    pub log_step: Real,
    /// FEFF muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
}

/// Inputs for FEFF `RIXS/rixs.f90` initial `TLb` assembly.
///
/// This corresponds to the block after the radial `DeltaV` overlap is computed
/// and before the incident-energy `KKInt` convolution. `radial_overlaps`
/// supplies FEFF `DBLE(totTLb(1))` as `(incident_energy, energy_transfer,
/// transition)`. The returned table is FEFF `TLb(iE1, iE2, L, ind)` indexed as
/// `(incident_energy, energy_transfer, angular_channel, transition)`.
#[derive(Debug, Clone, Copy)]
pub struct RixsInitialAmplitudeInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// Precomputed radial overlaps from `Rl * DeltaV * Rl_2`.
    pub radial_overlaps: ArrayView3<'a, Real>,
    /// FEFF `rkk(iE1, ind, is1)` incident transition moments.
    pub incident_transition_moments: ArrayView3<'a, Complex>,
    /// FEFF `ph(iE1, -lnd(ind), 1, 0)`, preselected per transition.
    pub incident_phase_shifts: ArrayView2<'a, Complex>,
    /// FEFF `gg(L1, L2, iph, iE1)` for the selected potential `iph`.
    pub incident_green: ArrayView3<'a, Complex>,
    /// FEFF `xsnorm(iE1)` normalization table.
    pub normalization: ArrayView1<'a, Real>,
    /// FEFF `lnd(ind)` transition angular momentum labels.
    pub transition_angular_momenta: &'a [isize],
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
}

/// Inputs for the FEFF `RIXS/rixs.f90` direct final-transition term.
///
/// This ports the `ctmp(1) = ABS(rkk_2 * EXP(-i*ph_2)) * SQRT(xsnorm)` branch
/// used inside the incident-energy `TLb` convolution.
#[derive(Debug, Clone, Copy)]
pub struct RixsDirectFinalTransitionInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF `rkk_2(iE2, ind, is1)` transition moments.
    pub final_transition_moments: ArrayView3<'a, Complex>,
    /// FEFF `ph_2(iE2, -lnd(ind), 1, 0)`, preselected per transition.
    pub final_phase_shifts: ArrayView2<'a, Complex>,
    /// FEFF `xsnorm(iE2)` normalization table.
    pub normalization: ArrayView1<'a, Real>,
    /// FEFF `lnd(ind)` transition angular momentum labels.
    pub transition_angular_momenta: &'a [isize],
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
}

/// Inputs for FEFF `RIXS/rixs.f90` incident-energy convolution of `TLb`.
///
/// `transition_amplitudes` is FEFF `TLb(iE1, iE2, L, ind)` before the
/// convolution. The returned table has the same layout after the `KKInt` edge
/// convolution, final transition-moment term, and diagonal `bmat` scaling.
#[derive(Debug, Clone, Copy)]
pub struct RixsIncidentAmplitudeConvolutionInput<'a> {
    /// FEFF relative energy grid `rem(1:ne1)`.
    pub relative_energies: ArrayView1<'a, Real>,
    /// FEFF input `TLb(iE1, iE2, L, ind)` table.
    pub transition_amplitudes: ArrayView4<'a, Complex>,
    /// FEFF `rkk_2(iE2, ind, is1)` transition moments.
    pub final_transition_moments: ArrayView3<'a, Complex>,
    /// FEFF `ph_2(iE2, -lnd(ind), 1, 0)`, preselected per transition.
    pub final_phase_shifts: ArrayView2<'a, Complex>,
    /// FEFF complex `k2(iE2)` values.
    pub final_wave_numbers: ArrayView1<'a, Complex>,
    /// FEFF `xsnorm(iE2)` normalization table.
    pub normalization: ArrayView1<'a, Real>,
    /// Diagonal FEFF `bmat(m1,is1-1,ind,m1,is1-1,ind,ipmin)` values as `(L, ind, is1)`.
    pub b_matrix_diagonal: ArrayView3<'a, Complex>,
    /// FEFF `lnd(ind)` transition angular momentum labels.
    pub transition_angular_momenta: &'a [isize],
    /// FEFF Fermi level `xmu`.
    pub fermi_level: Real,
    /// FEFF core-hole width `gam_ch`.
    pub core_width: Real,
}

#[derive(Debug)]
struct RixsEmissionEnergyMap {
    incident_energy_ev: Array1<Real>,
    emission_energy_ev: Array1<Real>,
    channels: Array2<Real>,
}

/// Port of FEFF `KKInt`.
///
/// This evaluates the analytic integral of `(slope * x' + intercept) /
/// (x' - x + i * width)` from `x0` to `x1`. FEFF uses separate expressions
/// for interior/off-interval points and the two exact endpoint cases; this
/// function preserves those branches.
pub fn kk_integral(
    slope: Complex,
    intercept: Complex,
    x0: Real,
    x1: Real,
    width: Real,
    x: Real,
) -> Result<Complex, RixsError> {
    validate_complex("a", slope)?;
    validate_complex("b", intercept)?;
    validate_finite("x0", x0)?;
    validate_finite("x1", x1)?;
    validate_width("gam", width)?;
    validate_finite("x", x)?;
    if x0 >= x1 {
        return Err(RixsError::InvalidInterval {
            lower: x0,
            upper: x1,
        });
    }

    let i = Complex::new(0.0, 1.0);
    let width_at_x = Complex::new(width, x);
    if x != x0 && x != x1 {
        let left = x - x0;
        let right = x - x1;
        let log_ratio = ((width * width + left * left) / (width * width + right * right)).ln();
        let bracket = Complex::new(
            (width / left).atan() - (width / right).atan() - KKINT_PI,
            0.5 * log_ratio,
        );
        let mut value = slope * (Complex::new(x1 - x0, 0.0) + width_at_x * bracket);
        if x < x0 || x > x1 {
            value += slope * KKINT_PI * width_at_x;
        }
        Ok(value + intercept * (Complex::new(x1 - x, width) / Complex::new(x0 - x, width)).ln())
    } else if x == x0 {
        let span = x1 - x0;
        let bracket = Complex::new(
            (width / (x0 - x1)).atan() + 0.5 * KKINT_PI,
            0.5 * ((width * width + span * span) / (width * width)).ln(),
        );
        Ok(
            slope * (Complex::new(span, 0.0) - Complex::new(width, x0) * bracket)
                + intercept
                    * (Complex::new(width, 0.0) / (Complex::new(width, 0.0) - i * span)).ln(),
        )
    } else {
        let span = x1 - x0;
        let bracket = Complex::new(
            (width / (x0 - x1)).atan() + 0.5 * KKINT_PI,
            0.5 * ((width * width) / (width * width + span * span)).ln(),
        );
        Ok(
            slope * (Complex::new(span, 0.0) - Complex::new(width, x1) * bracket)
                + intercept * ((Complex::new(width, 0.0) - i * span) / width).ln(),
        )
    }
}

/// Port of FEFF `IntDoubleLorentz`.
///
/// `omega = Some(value)` corresponds to FEFF `iinf >= 0`, where the analytic
/// antiderivative is evaluated at a finite upper limit. `omega = None`
/// corresponds to FEFF `iinf < 0`, the simplified infinite-limit branch.
pub fn integrated_double_lorentz(
    rem1: Real,
    rem2: Real,
    core_width: Real,
    width: Real,
    intercept: Real,
    slope: Real,
    omega: Option<Real>,
) -> Result<Real, RixsError> {
    validate_finite("rem1", rem1)?;
    validate_finite("rem2", rem2)?;
    validate_width("gamch", core_width)?;
    validate_width("gam", width)?;
    validate_finite("a", intercept)?;
    validate_finite("b", slope)?;

    let delta = rem1 - rem2;
    let value = if let Some(omega) = omega {
        validate_finite("omega", omega)?;
        let gamch2 = core_width * core_width;
        let gam2 = width * width;
        let delta2 = delta * delta;
        let first = 2.0
            * width
            * (intercept * (gam2 - gamch2 + delta2)
                + slope * (gam2 * rem1 + gamch2 * (rem1 - 2.0 * rem2) + rem1 * delta2))
            * ((omega - rem1) / core_width).atan();
        let second = core_width
            * (2.0
                * (intercept * (-gam2 + gamch2 + delta2)
                    + slope * ((gamch2 + delta2) * rem2 + gam2 * (-2.0 * rem1 + rem2)))
                * ((omega - rem2) / width).atan()
                + width
                    * (2.0 * intercept * (-rem1 + rem2)
                        + slope * (gam2 - gamch2 - rem1 * rem1 + rem2 * rem2))
                    * ((gamch2 + (omega - rem1) * (omega - rem1)).ln()
                        - (gam2 + (omega - rem2) * (omega - rem2)).ln()));
        let denominator = 2.0
            * width
            * core_width
            * ((width - core_width) * (width - core_width) + delta2)
            * ((width + core_width) * (width + core_width) + delta2);
        (first + second) / denominator
    } else {
        intercept * (width + core_width) * std::f64::consts::PI
            / (delta * delta + (width + core_width) * (width + core_width))
            / (2.0 * width * core_width)
    };

    Ok(value * width / std::f64::consts::PI)
}

/// Port of FEFF `BLInterp2D`: bilinear interpolation of a complex 2-D table.
///
/// `x` and `y` are strictly increasing coordinate grids. `values` is indexed as
/// `values[(x_index, y_index)]`, matching FEFF `A(ix, iy)`. Coordinates within
/// `1e-5` outside either endpoint use FEFF's endpoint interval and therefore
/// may extrapolate slightly, including FEFF's sentinel-order behavior above the
/// upper endpoint.
pub fn bilinear_interpolate_complex(
    x: ArrayView1<'_, Real>,
    y: ArrayView1<'_, Real>,
    values: ArrayView2<'_, Complex>,
    x0: Real,
    y0: Real,
) -> Result<Complex, RixsError> {
    validate_bilinear_inputs(x, y, values, x0, y0)?;

    let (x_lower, x_upper) = interpolation_interval(x, x0);
    let (y_lower, y_upper) = interpolation_interval(y, y0);
    let dx = x[x_upper] - x[x_lower];
    let dy = y[y_upper] - y[y_lower];
    if dx == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "x",
            index: x_lower,
        });
    }
    if dy == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "y",
            index: y_lower,
        });
    }

    let lower_lower = matrix_value(values, x_lower, y_lower)?;
    let upper_lower = matrix_value(values, x_upper, y_lower)?;
    let lower_upper = matrix_value(values, x_lower, y_upper)?;
    let upper_upper = matrix_value(values, x_upper, y_upper)?;
    let dxdy = dx * dy;
    Ok((lower_lower * (x[x_upper] - x0) * (y[y_upper] - y0)
        + upper_lower * (x0 - x[x_lower]) * (y[y_upper] - y0)
        + lower_upper * (x[x_upper] - x0) * (y0 - y[y_lower])
        + upper_upper * (x0 - x[x_lower]) * (y0 - y[y_lower]))
        / dxdy)
}

/// Port of FEFF `RIXS/rixs.f90` final spectrum output assembly.
///
/// This builds the data FEFF writes to `xasEI.dat`, `xasEF.dat`, `herfd.dat`,
/// `rixsET.dat`, and `rixsEE.dat` from an already-computed `xsect_tmp` table.
/// It preserves FEFF's row ordering, trapezoid-window rules, diagonal HERFD
/// extraction, and constant incident/emission grid interpolation.
pub fn rixs_final_spectrum(
    input: RixsFinalSpectrumInput<'_>,
) -> Result<RixsFinalSpectrum, RixsError> {
    validate_final_spectrum_input(input)?;
    let energy_count = input.relative_energies.len();
    let channel_count = input.cross_section.dim().2;
    let row_count = energy_count * energy_count;

    let incident_xas = rixs_project_incident_xas(
        input.relative_energies,
        input.cross_section,
        input.incident_window,
    );
    let final_xas = rixs_project_final_xas(
        input.relative_energies,
        input.cross_section,
        input.final_window,
    );
    let incident_xas_energy_ev: Array1<Real> = input
        .relative_energies
        .iter()
        .map(|energy| (*energy + input.incident_edge) * input.hartree_ev)
        .collect();
    let final_xas_energy_ev: Array1<Real> = input
        .relative_energies
        .iter()
        .map(|energy| (*energy + input.incident_edge - input.final_edge) * input.hartree_ev)
        .collect();
    let herfd_energy_ev = incident_xas_energy_ev.clone();
    let herfd = Array2::from_shape_fn((energy_count, channel_count), |(energy, channel)| {
        input.cross_section[(energy, energy, channel)]
    });

    let mut rixs_et_first_energy_ev = Array1::zeros(row_count);
    let mut rixs_et_second_energy_ev = Array1::zeros(row_count);
    let mut rixs_et = Array2::zeros((row_count, channel_count));
    for incident in 0..energy_count {
        for transfer in 0..energy_count {
            let row = incident * energy_count + transfer;
            rixs_et_first_energy_ev[row] =
                (input.relative_energies[transfer] + input.incident_edge) * input.hartree_ev;
            rixs_et_second_energy_ev[row] =
                (input.relative_energies[incident] + input.final_edge) * input.hartree_ev;
            for channel in 0..channel_count {
                rixs_et[(row, channel)] = input.cross_section[(incident, transfer, channel)];
            }
        }
    }

    let emission_map = rixs_emission_energy_map(input, channel_count)?;

    Ok(RixsFinalSpectrum {
        incident_xas_energy_ev,
        incident_xas,
        final_xas_energy_ev,
        final_xas,
        herfd_energy_ev,
        herfd,
        rixs_et_first_energy_ev,
        rixs_et_second_energy_ev,
        rixs_et,
        rixs_ee_incident_energy_ev: emission_map.incident_energy_ev,
        rixs_ee_emission_energy_ev: emission_map.emission_energy_ev,
        rixs_ee: emission_map.channels,
    })
}

/// Port of the FEFF `RIXS/rixs.f90` final multi-edge accumulation loop.
///
/// FEFF shifts each per-edge spectrum by `EdgeSplit(iEdge)`, clamps requests
/// below the first `rem` point to the first row, linearly interpolates or
/// extrapolates other requests with `terpc(..., m=1)`, then sums all edge
/// contributions into `xsect_tmp(iE2, iE1, ind)`.
pub fn rixs_sum_edge_contributions(
    input: RixsEdgeContributionInput<'_>,
) -> Result<Array3<Real>, RixsError> {
    validate_edge_contribution_input(input)?;
    let energy_count = input.relative_energies.len();
    let channel_count = input.edge_contributions.dim().2;
    let edge_count = input.edge_splits.len();
    let mut summed = Array3::zeros((energy_count, energy_count, channel_count));
    if edge_count == 0 {
        return Ok(summed);
    }

    for channel in 0..channel_count {
        for incident in 0..energy_count {
            for edge in 0..edge_count {
                let edge_split = input.edge_splits[edge];
                for transfer in 0..energy_count {
                    let target = input.relative_energies[transfer] - edge_split;
                    let contribution = if target > input.relative_energies[0] {
                        rixs_linear_edge_interpolate(input, incident, channel, edge, target)?
                    } else {
                        input.edge_contributions[(0, incident, channel, edge)]
                    };
                    summed[(transfer, incident, channel)] += contribution;
                }
            }
        }
    }
    Ok(summed)
}

/// Port of the FEFF `RIXS/rixs.f90` per-edge broadening loop.
///
/// FEFF first copies the raw cross-section into every edge slot, then runs the
/// same incident-energy broadening for each edge before applying edge-specific
/// final broadening and amplitudes. The Rust port computes the identical
/// incident-broadened table once and reuses it for each final edge. The result
/// is indexed as `(energy_transfer, incident_energy, channel, edge)`.
pub fn rixs_broaden_edge_contributions(
    input: RixsEdgeBroadeningInput<'_>,
) -> Result<Array4<Real>, RixsError> {
    validate_edge_broadening_input(input)?;
    let energy_count = input.relative_energies.len();
    let channel_count = input.raw_cross_section.dim().2;
    let edge_count = input.edge_widths.len();
    let incident_broadened = rixs_incident_energy_broadening(RixsIncidentEnergyBroadeningInput {
        relative_energies: input.relative_energies,
        edge_cross_section: input.raw_cross_section,
        self_energy: input.self_energy,
        fermi_level: input.fermi_level,
        incident_width: input.incident_width,
    })?;
    let mut edge_contributions =
        Array4::zeros((energy_count, energy_count, channel_count, edge_count));

    for edge in 0..edge_count {
        let final_broadened = rixs_final_energy_broadening(RixsFinalEnergyBroadeningInput {
            relative_energies: input.relative_energies,
            edge_cross_section: incident_broadened.view(),
            core_width: input.core_width,
            final_width: input.final_width_base + input.edge_widths[edge],
            edge_amplitude: input.edge_amplitudes[edge],
        })?;
        for transfer in 0..energy_count {
            for incident in 0..energy_count {
                for channel in 0..channel_count {
                    edge_contributions[(transfer, incident, channel, edge)] =
                        final_broadened[(transfer, incident, channel)];
                }
            }
        }
    }

    Ok(edge_contributions)
}

/// Port of the FEFF `RIXS/rixs.f90` standard post-raw spectrum pipeline.
///
/// This composes the FEFF per-edge broadening loop, shifted multi-edge
/// accumulation loop, and final spectrum-output block for a precomputed raw
/// cross-section table. It is the downstream half of the normal RIXS solver,
/// after raw `xsect_rxs(:,:,:,1)` has been assembled.
pub fn rixs_post_raw_spectrum(
    input: RixsPostRawSpectrumInput<'_>,
) -> Result<RixsPostRawSpectrum, RixsError> {
    let edge_contributions = rixs_broaden_edge_contributions(RixsEdgeBroadeningInput {
        relative_energies: input.relative_energies,
        raw_cross_section: input.raw_cross_section,
        self_energy: input.self_energy,
        fermi_level: input.fermi_level,
        core_width: input.core_width,
        incident_width: input.incident_width,
        final_width_base: input.final_width_base,
        edge_widths: input.edge_widths,
        edge_amplitudes: input.edge_amplitudes,
    })?;
    let summed_cross_section = rixs_sum_edge_contributions(RixsEdgeContributionInput {
        relative_energies: input.relative_energies,
        edge_splits: input.edge_splits,
        edge_contributions: edge_contributions.view(),
    })?;
    let spectrum = rixs_final_spectrum(RixsFinalSpectrumInput {
        relative_energies: input.relative_energies,
        cross_section: summed_cross_section.view(),
        incident_window: input.incident_window,
        final_window: input.final_window,
        incident_edge: input.incident_edge,
        final_edge: input.final_edge,
        hartree_ev: input.hartree_ev,
    })?;

    Ok(RixsPostRawSpectrum {
        edge_contributions,
        summed_cross_section,
        spectrum,
    })
}

/// Port of the FEFF `RIXS/rixs.f90` `MBConv` satellite spectrum block.
///
/// FEFF applies the XES satellite convolution to the already-summed
/// `xsect_tmp` table, then writes the same spectrum products as the standard
/// output block. The satellite `xasEI-sat.dat` and `xasEF-sat.dat` line
/// spectra use `rem(iE)*hart` as their energy columns, so this helper adjusts
/// those columns after reusing [`rixs_final_spectrum`].
pub fn rixs_satellite_spectrum(
    input: RixsSatelliteSpectrumInput<'_>,
) -> Result<RixsSatelliteSpectrum, RixsError> {
    let satellite_cross_section = rixs_satellite_convolution(RixsSatelliteConvolutionInput {
        relative_energies: input.relative_energies,
        cross_section: input.cross_section,
        xes_energy_ev: input.xes_energy_ev,
        xes_mu: input.xes_mu,
        fermi_level: input.fermi_level,
        hartree_ev: input.hartree_ev,
    })?;
    let mut spectrum = rixs_final_spectrum(RixsFinalSpectrumInput {
        relative_energies: input.relative_energies,
        cross_section: satellite_cross_section.view(),
        incident_window: input.incident_window,
        final_window: input.final_window,
        incident_edge: input.incident_edge,
        final_edge: input.final_edge,
        hartree_ev: input.hartree_ev,
    })?;
    let relative_energy_ev = input
        .relative_energies
        .iter()
        .map(|energy| energy * input.hartree_ev)
        .collect::<Array1<_>>();
    spectrum.incident_xas_energy_ev = relative_energy_ev.clone();
    spectrum.final_xas_energy_ev = relative_energy_ev;

    Ok(RixsSatelliteSpectrum {
        satellite_cross_section,
        spectrum,
    })
}

/// Port of the FEFF `RIXS/rixs.f90` `ReadPoles` edge normalization block.
///
/// This preserves FEFF's ordering and shifting semantics, including the
/// one-row case after dropping the incident pole where only `Edge1(1)` is
/// shifted by `xmu`.
pub fn rixs_normalize_poles(
    input: RixsPoleNormalizationInput<'_>,
) -> Result<RixsPoleNormalization, RixsError> {
    validate_pole_normalization_input(input)?;
    let pole_count = input.pole_energies.len();
    let edge_count = pole_count - 1;

    let mut edge_splits = input
        .pole_energies
        .iter()
        .skip(1)
        .copied()
        .rev()
        .collect::<Array1<_>>();
    let edge_widths = input
        .pole_widths
        .iter()
        .skip(1)
        .copied()
        .rev()
        .collect::<Array1<_>>();
    let mut edge_amplitudes = input.pole_amplitudes.iter().copied().collect::<Array1<_>>();
    if edge_amplitudes[pole_count - 1] < 0.0 {
        edge_amplitudes[pole_count - 1] = 1.0;
    }
    let edge_amplitudes = edge_amplitudes.iter().skip(1).copied().rev().collect();

    for split in &mut edge_splits {
        if *split <= 0.0 {
            *split = -input.fermi_level;
        }
    }

    let mut edge1 = [input.pole_energies[0], edge_splits[0]];
    for edge in edge1.iter_mut().take(edge_count.min(2)) {
        *edge -= input.fermi_level;
    }
    for split in &mut edge_splits {
        *split -= edge1[1];
    }

    Ok(RixsPoleNormalization {
        incident_edge: edge1[0],
        final_edge: edge1[1],
        edge_splits,
        edge_amplitudes,
        edge_widths,
        core_width: input.pole_widths[0],
    })
}

/// Port of the FEFF `RIXS/rixs.f90` incident-energy convolution for one edge.
///
/// This is the first broadening pass after raw `xsect_rxs` assembly. FEFF
/// integrates along a diagonal incident-energy path, samples the cross-section
/// with `BLInterp2D`, uses order-1 `terpc` self-energy interpolation for the
/// width, and adds the same arctangent endpoint correction.
pub fn rixs_incident_energy_broadening(
    input: RixsIncidentEnergyBroadeningInput<'_>,
) -> Result<Array3<Real>, RixsError> {
    validate_incident_energy_broadening_input(input)?;
    let energy_count = input.relative_energies.len();
    let channel_count = input.edge_cross_section.dim().2;
    let delta_min = input
        .relative_energies
        .iter()
        .zip(input.relative_energies.iter().skip(1))
        .map(|(&lower, &upper)| upper - lower)
        .fold(Real::INFINITY, Real::min);
    let mut broadened = Array3::zeros((energy_count, energy_count, channel_count));

    for channel in 0..channel_count {
        for incident in 0..energy_count {
            for transfer in 0..energy_count {
                broadened[(transfer, incident, channel)] =
                    rixs_integrated_incident_broadening_point(
                        input, delta_min, incident, transfer, channel,
                    )?;
            }
        }
    }
    Ok(broadened)
}

/// Port of the FEFF `RIXS/rixs.f90` `ReadSigma`/`mpse.dat` preparation block.
///
/// The output is FEFF `Sigma(1:ne1)` after eV-to-Hartree conversion and
/// piecewise-linear `terpc(..., m=1)` interpolation at `rem(iE1) - xmu`.
/// Energies below `xmu` use the first MPSE value; energies above the active
/// MPSE range use the final MPSE value, preserving the Fortran branch tests.
pub fn rixs_prepare_self_energy_grid(
    input: RixsSelfEnergyGridInput<'_>,
) -> Result<Array1<Complex>, RixsError> {
    validate_self_energy_grid_input(input)?;
    let mpse_energy_hartree: Array1<Real> = input
        .mpse_energy_ev
        .iter()
        .map(|energy| energy / input.hartree_ev)
        .collect();
    let mpse_self_energy_hartree: Array1<Complex> = input
        .mpse_self_energy_ev
        .iter()
        .map(|self_energy| self_energy / input.hartree_ev)
        .collect();
    let last_mpse_energy = mpse_energy_hartree[mpse_energy_hartree.len() - 1];
    let first_self_energy = mpse_self_energy_hartree[0];
    let last_self_energy = mpse_self_energy_hartree[mpse_self_energy_hartree.len() - 1];
    let mut prepared = Array1::zeros(input.relative_energies.len());

    for (index, &energy) in input.relative_energies.iter().enumerate() {
        prepared[index] = if energy >= input.fermi_level && energy <= last_mpse_energy {
            rixs_linear_self_energy_interpolate(
                mpse_energy_hartree.view(),
                mpse_self_energy_hartree.view(),
                energy - input.fermi_level,
            )?
        } else if energy < input.fermi_level {
            first_self_energy
        } else {
            last_self_energy
        };
    }

    Ok(prepared)
}

/// Port of the FEFF `RIXS/rixs.f90` `k1`/`k2` wave-number setup.
///
/// The final-state branch intentionally discards the imaginary part of
/// `final_reference_energy`, matching FEFF's `DBLE(eref_2(1,nspx))`.
pub fn rixs_wave_numbers(input: RixsWaveNumberInput<'_>) -> Result<RixsWaveNumbers, RixsError> {
    validate_wave_number_input(input)?;
    let incident_wave_numbers = input
        .relative_energies
        .iter()
        .map(|&energy| (Complex::new(energy, 0.0) - input.incident_reference_energy) * 2.0)
        .map(Complex::sqrt)
        .collect();
    let final_reference = input.final_reference_energy.re;
    let final_wave_numbers = input
        .relative_energies
        .iter()
        .map(|&energy| Complex::new(2.0 * (energy - final_reference), 0.0))
        .map(Complex::sqrt)
        .collect();

    Ok(RixsWaveNumbers {
        incident_wave_numbers,
        final_wave_numbers,
    })
}

/// Port of the FEFF `RIXS/rixs.f90` logarithmic radial-grid setup.
///
/// The returned `muffin_tin_index_fortran` preserves FEFF's one-based `imt`
/// integer truncation, and `active_point_count` is the `jmt` count used for
/// radial integration.
pub fn rixs_radial_grid(input: RixsRadialGridInput) -> Result<RixsRadialGrid, RixsError> {
    validate_radial_grid_input(input)?;
    let radii = Array1::from_iter(
        (0..input.point_count)
            .map(|index| (-input.log_origin + input.log_step * index as Real).exp()),
    );
    let muffin_tin_index_fortran =
        ((input.muffin_tin_radius.ln() + input.log_origin) / input.log_step + 1.0) as usize;
    let active_point_count = muffin_tin_index_fortran + 1;
    if active_point_count > input.point_count {
        return Err(RixsError::RadialGridActivePointCount {
            active_point_count,
            point_count: input.point_count,
        });
    }

    Ok(RixsRadialGrid {
        radii,
        muffin_tin_index_fortran,
        active_point_count,
    })
}

/// Port of the FEFF `RIXS/rixs.f90` screened-core potential difference setup.
///
/// Passing `None` for `final_screened_core_hole` selects the FEFF `VAL` branch,
/// where the final screened core-hole potential is zero.
pub fn rixs_core_hole_potential_difference(
    input: RixsCoreHolePotentialInput<'_>,
) -> Result<Array1<Real>, RixsError> {
    validate_core_hole_potential_input(input)?;
    let potential_difference = match input.final_screened_core_hole {
        Some(final_screened) => input
            .incident_screened_core_hole
            .iter()
            .zip(final_screened.iter())
            .map(|(&incident, &final_value)| -incident + final_value)
            .collect(),
        None => input
            .incident_screened_core_hole
            .iter()
            .map(|&incident| -incident)
            .collect(),
    };
    Ok(potential_difference)
}

/// Port of the FEFF `RIXS/rixs.f90` radial-function `rl.dat` read loop.
///
/// FEFF reads records in nested `(energy, angular-record)` order, writes each
/// record's energy into `em(iE)`, and stores radial values at the signed label
/// carried by the record. Repeated labels therefore overwrite earlier records,
/// matching Fortran array assignment.
pub fn rixs_radial_function_table(
    input: RixsRadialFunctionTableInput<'_>,
) -> Result<RixsRadialFunctionTable, RixsError> {
    validate_radial_function_table_input(input)?;
    let mut energies = Array1::zeros(input.energy_count);
    let mut radial_functions =
        Array3::zeros((input.radial_count, input.angular_count, input.energy_count));

    for (record_index, record) in input.records.iter().enumerate() {
        let energy = record_index / input.angular_count;
        let angular_momentum = usize::try_from(record.angular_momentum).map_err(|_| {
            RixsError::RadialFunctionAngularOutOfRange {
                record: record_index,
                angular_momentum: record.angular_momentum,
                angular_count: input.angular_count,
            }
        })?;
        energies[energy] = record.energy;
        for radial in 0..input.radial_count {
            radial_functions[(radial, angular_momentum, energy)] = record.radial_values[radial];
        }
    }

    Ok(RixsRadialFunctionTable {
        energies,
        radial_functions,
    })
}

/// Port of the FEFF `RIXS/rixs.f90` `setkap`/`bcoef` transition setup.
///
/// The full FEFF B matrix is delegated to the shared `bcoef` port. This helper
/// extracts the diagonal entries that RIXS later uses as
/// `sqrt(ABS(bmat(m,is,ind,m,is,ind)))`.
pub fn rixs_transition_matrix_setup(
    input: RixsTransitionMatrixInput,
) -> Result<RixsTransitionMatrixSetup, RixsError> {
    validate_transition_matrix_input(input)?;
    let core_hole = core_hole_quantum_numbers(input.hole)
        .map_err(|source| RixsError::TransitionCoreHoleSetup { source })?;
    let transition_matrix = transition_b_matrix(TransitionBMatrixInput {
        lmax: input.lmax,
        initial_kappa: core_hole.kappa,
        polarization: input.polarization,
        polarization_tensor: input.polarization_tensor,
        multipole: input.multipole,
        trace_orbital: input.trace_orbital,
        spin: input.spin,
        spin_channels: input.spin_channel_count,
        spin_vector_angle: input.spin_vector_angle,
    })
    .map_err(|source| RixsError::TransitionBMatrixSetup { source })?;

    let angular_count = input
        .lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(RixsError::InvalidAngularMomentum { value: isize::MAX })?;
    let mut transition_angular_momenta = [0_isize; 8];
    for (target, &angular_momentum) in transition_angular_momenta
        .iter_mut()
        .zip(transition_matrix.orbital_momenta.iter())
    {
        *target = angular_momentum as isize;
    }
    let mut b_matrix_diagonal = Array3::zeros((angular_count, 8, input.spin_channel_count));

    for (transition, &angular_momentum) in transition_angular_momenta.iter().enumerate() {
        if angular_momentum < 0 {
            continue;
        }
        let angular_momentum =
            usize::try_from(angular_momentum).map_err(|_| RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        let angular_momentum_isize = isize::try_from(angular_momentum)
            .map_err(|_| RixsError::InvalidAngularMomentum { value: isize::MAX })?;
        for magnetic in -angular_momentum_isize..=angular_momentum_isize {
            let angular_channel = angular_momentum
                .checked_mul(angular_momentum)
                .and_then(|start| {
                    usize::try_from(magnetic + angular_momentum_isize)
                        .ok()
                        .and_then(|offset| start.checked_add(offset))
                })
                .ok_or(RixsError::InvalidAngularMomentum {
                    value: angular_momentum_isize,
                })?;
            for spin in 0..input.spin_channel_count {
                b_matrix_diagonal[(angular_channel, transition, spin)] = transition_matrix
                    .value(
                        magnetic,
                        spin,
                        transition + 1,
                        magnetic,
                        spin,
                        transition + 1,
                    )
                    .ok_or(RixsError::TransitionBMatrixDiagonalMissing {
                        magnetic,
                        spin,
                        transition: transition + 1,
                    })?;
            }
        }
    }

    Ok(RixsTransitionMatrixSetup {
        initial_kappa: core_hole.kappa,
        initial_angular_momentum: core_hole.angular_momentum,
        transition_kappas: transition_matrix.kappa_indices,
        transition_angular_momenta,
        b_matrix_diagonal,
    })
}

/// Port of FEFF RIXS `ph(iE, -lnd(ind), 1, 0)` phase-shift selection.
///
/// Active non-negative transition labels select signed-l `-lnd(ind)`.
/// Negative `lnd` transitions are inactive in the surrounding FEFF loops and
/// remain zero in the returned `(energy, transition)` table.
pub fn rixs_transition_phase_shifts(
    input: RixsTransitionPhaseShiftInput<'_>,
) -> Result<Array2<Complex>, RixsError> {
    validate_transition_phase_shift_input(input)?;
    let (energy_count, _) = input.phase_shifts.dim();
    let transition_count = input.transition_angular_momenta.len();
    let mut selected = Array2::zeros((energy_count, transition_count));

    for (transition, &angular_momentum) in input.transition_angular_momenta.iter().enumerate() {
        if angular_momentum < 0 {
            continue;
        }
        let signed_l = angular_momentum
            .checked_neg()
            .ok_or(RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        let column = (signed_l - input.signed_l_min) as usize;
        for energy in 0..energy_count {
            selected[(energy, transition)] = input.phase_shifts[(energy, column)];
        }
    }

    Ok(selected)
}

/// Port of the FEFF `RIXS/rixs.f90` `MBConv` satellite convolution.
///
/// FEFF transforms the `XES/xmu.dat` energy grid to Hartree offsets, reverses
/// the XES table, then trapezoid-integrates shifted slices of `xsect_tmp`.
/// Requests below the first `rem` point are zero for the upper endpoint; the
/// lower endpoint uses FEFF's `xmu` cutoff before applying order-1 `terpc`
/// interpolation or clamping above the last `rem` point.
pub fn rixs_satellite_convolution(
    input: RixsSatelliteConvolutionInput<'_>,
) -> Result<Array3<Real>, RixsError> {
    validate_satellite_convolution_input(input)?;
    let energy_count = input.relative_energies.len();
    let channel_count = input.cross_section.dim().2;
    let (xes_energy, xes_mu) = rixs_prepared_xes_grid(input)?;
    let mut convolved = Array3::zeros((energy_count, energy_count, channel_count));

    for channel in 0..channel_count {
        for incident in 0..energy_count {
            for transfer in 0..energy_count {
                let mut total = 0.0;
                for step in 1..xes_energy.len() {
                    let current = rixs_satellite_interpolated_value(
                        input,
                        incident,
                        channel,
                        input.relative_energies[transfer] - xes_energy[step],
                        input.relative_energies[0],
                    )?;
                    let previous = rixs_satellite_interpolated_value(
                        input,
                        incident,
                        channel,
                        input.relative_energies[transfer] - xes_energy[step - 1],
                        input.fermi_level,
                    )?;
                    total += 0.5
                        * (current * xes_mu[step] + previous * xes_mu[step - 1])
                        * (xes_energy[step] - xes_energy[step - 1]);
                }
                convolved[(transfer, incident, channel)] = total;
            }
        }
    }
    Ok(convolved)
}

/// Port of the FEFF `RIXS/rixs.f90` final-energy convolution for one edge.
///
/// For ordinary final widths this integrates the piecewise-linear
/// `xsect_rxs(:, iE1, ind, iEdge)` curve against the double-Lorentzian analytic
/// primitive from `RIXS/doublelorentz.f90`, including FEFF's `1.001` interval
/// endpoint scaling and infinite-tail correction. Widths below
/// [`FEFF_RIXS_FINAL_BROADENING_SKIP_WIDTH`] follow FEFF's direct no-convolution
/// branch.
pub fn rixs_final_energy_broadening(
    input: RixsFinalEnergyBroadeningInput<'_>,
) -> Result<Array3<Real>, RixsError> {
    validate_final_energy_broadening_input(input)?;
    let energy_count = input.relative_energies.len();
    let channel_count = input.edge_cross_section.dim().2;
    let mut broadened = Array3::zeros((energy_count, energy_count, channel_count));

    for channel in 0..channel_count {
        for incident in 0..energy_count {
            for transfer in 0..energy_count {
                let value = if input.final_width < FEFF_RIXS_FINAL_BROADENING_SKIP_WIDTH {
                    let delta =
                        input.relative_energies[incident] - input.relative_energies[transfer];
                    input.edge_cross_section[(transfer, incident, channel)]
                        / (delta * delta + input.core_width * input.core_width)
                } else {
                    rixs_integrated_final_broadening_point(input, incident, transfer, channel)?
                };
                broadened[(transfer, incident, channel)] = value * input.edge_amplitude;
            }
        }
    }
    Ok(broadened)
}

/// Port of the FEFF `RIXS/rixs.f90` raw `xsect_rxs` assembly block.
///
/// Active transitions with `ind <= 3` contribute to output channel 0 and later
/// active transitions contribute to output channel 1, matching FEFF's two
/// output channels. Negative `lnd(ind)` transitions are skipped. The output is
/// indexed as `(energy_transfer, incident_energy, channel)`.
pub fn rixs_raw_cross_section(
    input: RixsRawCrossSectionInput<'_>,
) -> Result<Array3<Real>, RixsError> {
    validate_raw_cross_section_input(input)?;
    let (energy_count, _, _, transition_count) = input.transition_amplitudes.dim();
    let spin_multiplier = input.spin_channel_count as Real;
    let mut cross_section = Array3::zeros((
        energy_count,
        energy_count,
        RIXS_RAW_CROSS_SECTION_CHANNEL_COUNT,
    ));

    for incident in 0..energy_count {
        for transfer in 0..energy_count {
            for transition in 0..transition_count {
                let angular_momentum = input.transition_angular_momenta[transition];
                if angular_momentum < 0 {
                    continue;
                }
                let angular_momentum = usize::try_from(angular_momentum).map_err(|_| {
                    RixsError::InvalidAngularMomentum {
                        value: input.transition_angular_momenta[transition],
                    }
                })?;
                let phase_factor = (Complex::new(0.0, 2.0)
                    * input.final_phase_shifts[(transfer, transition)])
                    .exp();
                let output_channel = usize::from(transition >= RIXS_DIPOLE_TRANSITION_COUNT);
                for angular in rixs_angular_channel_range(angular_momentum)? {
                    let amplitude =
                        input.transition_amplitudes[(incident, transfer, angular, transition)];
                    let green = input.final_green[(angular, angular, transfer)];
                    let density_factor = 1.0 + (green * phase_factor).im;
                    cross_section[(transfer, incident, output_channel)] +=
                        amplitude.norm_sqr() * density_factor * spin_multiplier;
                }
            }
        }
    }
    Ok(cross_section)
}

/// Port of the FEFF `RIXS/rixs.f90` radial `DeltaV` overlap block.
///
/// The output is FEFF `DBLE(totTLb(1))` indexed as `(incident_energy,
/// energy_transfer, transition)`. Negative `lnd(ind)` transitions and
/// incident/transfer pairs below `xmu` remain zero, matching the Fortran loop.
pub fn rixs_radial_transition_overlaps(
    input: RixsRadialOverlapInput<'_>,
) -> Result<Array3<Real>, RixsError> {
    validate_radial_overlap_input(input)?;
    let energy_count = input.relative_energies.len();
    let transition_count = input.transition_angular_momenta.len();
    let radial_count = input.radii.len();
    let radii = input.radii.to_vec();
    let mut overlaps = Array3::zeros((energy_count, energy_count, transition_count));

    for transition in 0..transition_count {
        let angular_momentum = input.transition_angular_momenta[transition];
        if angular_momentum < 0 {
            continue;
        }
        let angular_momentum =
            usize::try_from(angular_momentum).map_err(|_| RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        for transfer in 0..energy_count {
            for incident in 0..energy_count {
                if input.relative_energies[incident] < input.fermi_level
                    || input.relative_energies[transfer] < input.fermi_level
                {
                    continue;
                }
                let mut integrand = Vec::with_capacity(radial_count);
                for radial in 0..radial_count {
                    integrand.push(
                        input.initial_radial_functions[(radial, angular_momentum, incident)]
                            * input.potential_difference[radial]
                            * input.final_radial_functions[(radial, angular_momentum, transfer)],
                    );
                }
                let overlap = csomm2(
                    &radii,
                    &integrand,
                    input.log_step,
                    0.0,
                    input.muffin_tin_radius,
                )
                .map_err(|source| RixsError::RadialOverlapQuadrature { source })?;
                overlaps[(incident, transfer, transition)] = overlap.re;
            }
        }
    }

    Ok(overlaps)
}

/// Port of the FEFF `RIXS/rixs.f90` initial `TLb` assembly block.
///
/// Active transitions with non-negative `lnd(ind)` are filled only when both
/// incident and transfer energies are at or above `xmu`. FEFF loops over spin
/// channels but writes into a spinless `TLb` table; this routine preserves that
/// behavior by letting later spin channels overwrite earlier ones.
pub fn rixs_initial_transition_amplitudes(
    input: RixsInitialAmplitudeInput<'_>,
) -> Result<Array4<Complex>, RixsError> {
    validate_initial_amplitude_input(input)?;
    let energy_count = input.relative_energies.len();
    let transition_count = input.transition_angular_momenta.len();
    let spin_count = input.incident_transition_moments.dim().2;
    let angular_count = input.incident_green.dim().0;
    let mut amplitudes =
        Array4::zeros((energy_count, energy_count, angular_count, transition_count));

    for transition in 0..transition_count {
        let angular_momentum = input.transition_angular_momenta[transition];
        if angular_momentum < 0 {
            continue;
        }
        let angular_momentum =
            usize::try_from(angular_momentum).map_err(|_| RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        for spin in 0..spin_count {
            for transfer in 0..energy_count {
                for incident in 0..energy_count {
                    if input.relative_energies[incident] < input.fermi_level
                        || input.relative_energies[transfer] < input.fermi_level
                    {
                        continue;
                    }
                    let moment = input.incident_transition_moments[(incident, transition, spin)];
                    let phase = input.incident_phase_shifts[(incident, transition)];
                    let moment_factor = (moment * (-Complex::new(0.0, 1.0) * phase).exp()).norm();
                    let normalization = input.normalization[incident].sqrt();
                    let radial = input.radial_overlaps[(incident, transfer, transition)];
                    let phase_factor = (Complex::new(0.0, 2.0) * phase).exp();
                    for angular in rixs_angular_channel_range(angular_momentum)? {
                        let green = input.incident_green[(angular, angular, incident)];
                        let density_factor = 1.0 + (green * phase_factor).im;
                        amplitudes[(incident, transfer, angular, transition)] = Complex::new(
                            radial * moment_factor * density_factor * normalization,
                            0.0,
                        );
                    }
                }
            }
        }
    }

    Ok(amplitudes)
}

/// Port of the FEFF `RIXS/rixs.f90` direct final-transition term.
///
/// The output is indexed as `(energy_transfer, transition, spin)`. Energies
/// below `xmu` and inactive negative `lnd(ind)` transitions remain zero,
/// matching the FEFF loop around the `ctmp(1)` assignment.
pub fn rixs_direct_final_transition_amplitudes(
    input: RixsDirectFinalTransitionInput<'_>,
) -> Result<Array3<Complex>, RixsError> {
    validate_direct_final_transition_input(input)?;
    let energy_count = input.relative_energies.len();
    let transition_count = input.transition_angular_momenta.len();
    let spin_count = input.final_transition_moments.dim().2;
    let mut direct = Array3::zeros((energy_count, transition_count, spin_count));

    for transition in 0..transition_count {
        if input.transition_angular_momenta[transition] < 0 {
            continue;
        }
        for spin in 0..spin_count {
            for transfer in 0..energy_count {
                if input.relative_energies[transfer] < input.fermi_level {
                    continue;
                }
                direct[(transfer, transition, spin)] =
                    rixs_direct_final_transition_value(input, transfer, transition, spin);
            }
        }
    }

    Ok(direct)
}

/// Port of the FEFF `RIXS/rixs.f90` incident-energy convolution of `TLb`.
///
/// This applies the `KKInt` convolution over incident energy, adds FEFF's
/// inverse-square high-energy tail model, folds in the direct final transition
/// moment term, and finally applies the diagonal `bmat` factor. The output keeps
/// FEFF's `TLb(iE1, iE2, L, ind)` layout as `(incident, transfer, angular,
/// transition)`.
pub fn rixs_incident_amplitude_convolution(
    input: RixsIncidentAmplitudeConvolutionInput<'_>,
) -> Result<Array4<Complex>, RixsError> {
    validate_incident_amplitude_convolution_input(input)?;
    let (energy_count, _, _, transition_count) = input.transition_amplitudes.dim();
    let spin_count = input.final_transition_moments.dim().2;
    let mut convolved = input.transition_amplitudes.to_owned();
    let direct_terms = rixs_direct_final_transition_amplitudes(RixsDirectFinalTransitionInput {
        relative_energies: input.relative_energies,
        final_transition_moments: input.final_transition_moments,
        final_phase_shifts: input.final_phase_shifts,
        normalization: input.normalization,
        transition_angular_momenta: input.transition_angular_momenta,
        fermi_level: input.fermi_level,
    })?;

    for transition in 0..transition_count {
        let angular_momentum = input.transition_angular_momenta[transition];
        if angular_momentum < 0 {
            continue;
        }
        let angular_momentum =
            usize::try_from(angular_momentum).map_err(|_| RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        for spin in 0..spin_count {
            for angular in rixs_angular_channel_range(angular_momentum)? {
                for transfer in 0..energy_count {
                    let direct = direct_terms[(transfer, transition, spin)];
                    let mut updated = Vec::with_capacity(energy_count);
                    for incident in 0..energy_count {
                        let total = rixs_incident_convolution_point(
                            input,
                            convolved.view(),
                            incident,
                            transfer,
                            angular,
                            transition,
                            direct,
                        )?;
                        updated.push(-total);
                    }
                    for (incident, value) in updated.into_iter().enumerate() {
                        convolved[(incident, transfer, angular, transition)] = value;
                    }
                }
                let scale = input.b_matrix_diagonal[(angular, transition, spin)]
                    .norm()
                    .sqrt();
                for transfer in 0..energy_count {
                    for incident in 0..energy_count {
                        convolved[(incident, transfer, angular, transition)] *= scale;
                    }
                }
            }
        }
    }
    Ok(convolved)
}

fn rixs_project_incident_xas(
    energies: ArrayView1<'_, Real>,
    cross_section: ArrayView3<'_, Real>,
    window: (Real, Real),
) -> Array2<Real> {
    let energy_count = energies.len();
    let channel_count = cross_section.dim().2;
    Array2::from_shape_fn((energy_count, channel_count), |(incident, channel)| {
        rixs_windowed_trapezoid(energies, window, |transfer| {
            cross_section[(transfer, incident, channel)]
        })
    })
}

fn rixs_project_final_xas(
    energies: ArrayView1<'_, Real>,
    cross_section: ArrayView3<'_, Real>,
    window: (Real, Real),
) -> Array2<Real> {
    let energy_count = energies.len();
    let channel_count = cross_section.dim().2;
    Array2::from_shape_fn((energy_count, channel_count), |(incident, channel)| {
        rixs_windowed_trapezoid(energies, window, |transfer| {
            cross_section[(incident, transfer, channel)]
        })
    })
}

fn rixs_windowed_trapezoid(
    energies: ArrayView1<'_, Real>,
    window: (Real, Real),
    value_at: impl Fn(usize) -> Real,
) -> Real {
    let mut integral = 0.0;
    for index in 1..energies.len() {
        if energies[index] > window.1 {
            break;
        }
        if energies[index] > window.0 {
            integral += 0.5
                * (value_at(index) + value_at(index - 1))
                * (energies[index] - energies[index - 1]);
        }
    }
    integral
}

fn rixs_emission_energy_map(
    input: RixsFinalSpectrumInput<'_>,
    channel_count: usize,
) -> Result<RixsEmissionEnergyMap, RixsError> {
    let energy_count = input.relative_energies.len();
    let row_count = energy_count * energy_count;
    let first = input.relative_energies[0];
    let last = input.relative_energies[energy_count - 1];
    let emission_min = first - last;
    let emission_max = last - first;
    let denominator = (energy_count - 1) as Real;
    let mut incident_energy_ev = Array1::zeros(row_count);
    let mut emission_energy_ev = Array1::zeros(row_count);
    let mut channels = Array2::zeros((row_count, channel_count));

    for emission_index in 0..energy_count {
        let emission =
            (emission_max - emission_min) / denominator * emission_index as Real + emission_min;
        for incident_index in 0..energy_count {
            let incident = (last - first) / denominator * incident_index as Real + first;
            let transfer = incident - emission;
            let row = emission_index * energy_count + incident_index;
            incident_energy_ev[row] = (incident + input.incident_edge) * input.hartree_ev;
            emission_energy_ev[row] =
                (emission + input.incident_edge - input.final_edge) * input.hartree_ev;
            if transfer < first || transfer > last {
                continue;
            }
            for channel in 0..channel_count {
                channels[(row, channel)] = bilinear_interpolate_real(
                    input.relative_energies,
                    input.relative_energies,
                    input.cross_section,
                    channel,
                    incident,
                    transfer,
                )?;
            }
        }
    }
    Ok(RixsEmissionEnergyMap {
        incident_energy_ev,
        emission_energy_ev,
        channels,
    })
}

fn rixs_integrated_incident_broadening_point(
    input: RixsIncidentEnergyBroadeningInput<'_>,
    delta_min: Real,
    incident: usize,
    transfer: usize,
    channel: usize,
) -> Result<Real, RixsError> {
    let energy_count = input.relative_energies.len();
    let first_energy = input.relative_energies[0];
    let last_energy = input.relative_energies[energy_count - 1];
    let incident_energy = input.relative_energies[incident];
    let transfer_energy = input.relative_energies[transfer];
    let (upper_bound, lower_bound) = if incident_energy - transfer_energy > 0.0 {
        (
            last_energy - incident_energy + transfer_energy,
            first_energy,
        )
    } else {
        (
            last_energy,
            first_energy - incident_energy + transfer_energy,
        )
    };
    let window_min = lower_bound.min(upper_bound);
    let window_max = lower_bound.max(upper_bound);
    let width = window_max - window_min;
    let incident_delta = if incident > 0 {
        input.relative_energies[incident] - input.relative_energies[incident - 1]
    } else {
        input.relative_energies[incident + 1] - input.relative_energies[incident]
    };
    let transfer_delta = if transfer > 0 {
        input.relative_energies[transfer] - input.relative_energies[transfer - 1]
    } else {
        input.relative_energies[transfer + 1] - input.relative_energies[transfer]
    };
    let delta = incident_delta.min(transfer_delta);
    let nominal_step = ((delta + delta_min) / 4.0).max(delta_min);
    let subdivisions = (width / nominal_step) as usize;
    let pole_center = transfer_energy;
    let mut total = 0.0;
    let mut last_width = input.incident_width;
    let mut last_self_energy = Complex::new(0.0, 0.0);
    let mut endpoint_x = window_min;

    if subdivisions > 0 {
        let step = width / subdivisions as Real;
        let mut previous_value = bilinear_interpolate_real(
            input.relative_energies,
            input.relative_energies,
            input.edge_cross_section,
            channel,
            window_min,
            incident_energy + window_min - transfer_energy,
        )?;

        for interval in 0..subdivisions {
            let lower = window_min + step * interval as Real;
            let upper = (lower + step).clamp(lower_bound, upper_bound);
            let incident_self_energy = if lower >= input.fermi_level {
                rixs_self_energy_order1(input, incident_energy)?
            } else {
                Complex::new(0.0, 0.0)
            };
            let transfer_self_energy = rixs_self_energy_order1(input, transfer_energy)?;
            let combined_self_energy = incident_self_energy + transfer_self_energy;
            let total_width = input.incident_width + combined_self_energy.im.abs() / 2.0;
            validate_width("incident_total_width", total_width)?;
            let current_value = bilinear_interpolate_real(
                input.relative_energies,
                input.relative_energies,
                input.edge_cross_section,
                channel,
                upper,
                incident_energy + upper - transfer_energy,
            )?;
            let slope = (current_value - previous_value) / step;
            let intercept = current_value - slope * upper;
            previous_value = current_value;
            total += rixs_incident_lorentz_linear_segment(
                intercept,
                slope,
                pole_center,
                total_width,
                lower,
                step,
            );
            last_width = total_width;
            last_self_energy = combined_self_energy / 2.0;
            endpoint_x = upper;
        }
    }

    let endpoint_value = bilinear_interpolate_real(
        input.relative_energies,
        input.relative_energies,
        input.edge_cross_section,
        channel,
        endpoint_x,
        incident_energy + window_max - transfer_energy,
    )?;
    total += endpoint_value
        * (0.5
            - ((window_max - transfer_energy + last_self_energy.re) / last_width).atan()
                / std::f64::consts::PI);
    Ok(total)
}

fn rixs_incident_lorentz_linear_segment(
    intercept: Real,
    slope: Real,
    pole_center: Real,
    width: Real,
    lower: Real,
    step: Real,
) -> Real {
    let upper = lower + step;
    let lower_delta = lower - pole_center;
    let upper_delta = upper - pole_center;
    let value_at_center = intercept + slope * pole_center;
    let arctangent_term = (lower_delta / width).atan() - (upper_delta / width).atan();
    let logarithm_term = ((upper_delta * upper_delta + width * width)
        / (lower_delta * lower_delta + width * width))
        .ln();
    (-2.0 * value_at_center * arctangent_term + slope * width * logarithm_term) * 0.5
        / std::f64::consts::PI
}

fn rixs_self_energy_order1(
    input: RixsIncidentEnergyBroadeningInput<'_>,
    energy: Real,
) -> Result<Complex, RixsError> {
    let (lower, upper) = terpc_order1_interval(input.relative_energies, energy);
    let lower_energy = input.relative_energies[lower];
    let upper_energy = input.relative_energies[upper];
    let width = upper_energy - lower_energy;
    if width == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "self_energy",
            index: lower,
        });
    }
    Ok(input.self_energy[lower]
        + (input.self_energy[upper] - input.self_energy[lower]) * (energy - lower_energy) / width)
}

fn rixs_linear_self_energy_interpolate(
    energy: ArrayView1<'_, Real>,
    self_energy: ArrayView1<'_, Complex>,
    target: Real,
) -> Result<Complex, RixsError> {
    let (lower, upper) = terpc_order1_interval(energy, target);
    let lower_energy = energy[lower];
    let upper_energy = energy[upper];
    let width = upper_energy - lower_energy;
    if width == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "mpse_energy",
            index: lower,
        });
    }
    Ok(self_energy[lower]
        + (self_energy[upper] - self_energy[lower]) * (target - lower_energy) / width)
}

fn rixs_prepared_xes_grid(
    input: RixsSatelliteConvolutionInput<'_>,
) -> Result<(Vec<Real>, Vec<Real>), RixsError> {
    let mut energy = Vec::with_capacity(input.xes_energy_ev.len());
    let mut mu = Vec::with_capacity(input.xes_mu.len());
    for index in (0..input.xes_energy_ev.len()).rev() {
        energy.push(input.fermi_level - input.xes_energy_ev[index] / input.hartree_ev);
        mu.push(input.xes_mu[index]);
    }

    for index in 1..energy.len() {
        let previous = energy[index - 1];
        let current = energy[index];
        if current <= previous {
            return Err(RixsError::NonIncreasingGrid {
                axis: "xes_energy",
                index,
                previous,
                current,
            });
        }
    }

    Ok((energy, mu))
}

fn rixs_satellite_interpolated_value(
    input: RixsSatelliteConvolutionInput<'_>,
    incident: usize,
    channel: usize,
    target: Real,
    lower_cutoff: Real,
) -> Result<Real, RixsError> {
    if target < lower_cutoff {
        return Ok(0.0);
    }

    let energy_count = input.relative_energies.len();
    if target > input.relative_energies[energy_count - 1] {
        return Ok(input.cross_section[(energy_count - 1, incident, channel)]);
    }

    let (lower, upper) = terpc_order1_interval(input.relative_energies, target);
    let lower_energy = input.relative_energies[lower];
    let upper_energy = input.relative_energies[upper];
    let width = upper_energy - lower_energy;
    if width == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "satellite_energy",
            index: lower,
        });
    }
    Ok(input.cross_section[(lower, incident, channel)]
        + (input.cross_section[(upper, incident, channel)]
            - input.cross_section[(lower, incident, channel)])
            * (target - lower_energy)
            / width)
}

fn rixs_integrated_final_broadening_point(
    input: RixsFinalEnergyBroadeningInput<'_>,
    incident: usize,
    transfer: usize,
    channel: usize,
) -> Result<Real, RixsError> {
    let energy_count = input.relative_energies.len();
    let incident_energy = input.relative_energies[incident];
    let transfer_energy = input.relative_energies[transfer];
    let mut total = 0.0;
    for interval in 0..(energy_count - 1) {
        let lower_energy = input.relative_energies[interval];
        let upper_energy = input.relative_energies[interval + 1];
        let slope = (input.edge_cross_section[(interval + 1, incident, channel)]
            - input.edge_cross_section[(interval, incident, channel)])
            / (upper_energy - lower_energy);
        let intercept =
            input.edge_cross_section[(interval, incident, channel)] - slope * lower_energy;
        total += integrated_double_lorentz(
            incident_energy,
            transfer_energy,
            input.core_width,
            input.final_width,
            intercept,
            slope,
            Some(upper_energy * 1.001),
        )? - integrated_double_lorentz(
            incident_energy,
            transfer_energy,
            input.core_width,
            input.final_width,
            intercept,
            slope,
            Some(lower_energy * 1.001),
        )?;
    }

    let last_energy = input.relative_energies[energy_count - 1];
    let tail_value = input.edge_cross_section[(energy_count - 1, incident, channel)];
    total += integrated_double_lorentz(
        incident_energy,
        transfer_energy,
        input.core_width,
        input.final_width,
        tail_value,
        0.0,
        None,
    )? - integrated_double_lorentz(
        incident_energy,
        transfer_energy,
        input.core_width,
        input.final_width,
        tail_value,
        0.0,
        Some(last_energy * 1.001),
    )?;
    Ok(total)
}

fn rixs_direct_final_transition_value(
    input: RixsDirectFinalTransitionInput<'_>,
    transfer: usize,
    transition: usize,
    spin: usize,
) -> Complex {
    let moment = input.final_transition_moments[(transfer, transition, spin)];
    let phase = input.final_phase_shifts[(transfer, transition)];
    Complex::new(
        (moment * (-Complex::new(0.0, 1.0) * phase).exp()).norm()
            * input.normalization[transfer].sqrt(),
        0.0,
    )
}

fn rixs_incident_convolution_point(
    input: RixsIncidentAmplitudeConvolutionInput<'_>,
    amplitudes: ArrayView4<'_, Complex>,
    incident: usize,
    transfer: usize,
    angular: usize,
    transition: usize,
    direct: Complex,
) -> Result<Complex, RixsError> {
    let energy_count = input.relative_energies.len();
    let mut total = Complex::new(0.0, 0.0);
    for interval in 0..(energy_count - 1) {
        let upper_energy = input.relative_energies[interval + 1];
        if upper_energy <= input.fermi_level {
            continue;
        }
        let (lower, upper, lower_bound, upper_bound) =
            if input.relative_energies[interval] < input.fermi_level {
                let next = interval + 2;
                if next >= energy_count {
                    return Err(RixsError::IncidentConvolutionThresholdAtLastInterval { interval });
                }
                (
                    interval + 1,
                    next,
                    input.fermi_level,
                    input.relative_energies[interval + 1],
                )
            } else {
                (
                    interval,
                    interval + 1,
                    input.relative_energies[interval],
                    upper_energy,
                )
            };
        let slope = (amplitudes[(upper, transfer, angular, transition)]
            - amplitudes[(lower, transfer, angular, transition)])
            / (input.relative_energies[upper] - input.relative_energies[lower]);
        let intercept = amplitudes[(upper, transfer, angular, transition)]
            - slope * input.relative_energies[upper];
        total += kk_integral(
            slope,
            intercept,
            lower_bound,
            upper_bound,
            input.core_width,
            input.relative_energies[incident] * 1.00001,
        )?;
    }

    let last_energy = input.relative_energies[energy_count - 1];
    let tail_scale =
        amplitudes[(energy_count - 1, transfer, angular, transition)] * last_energy * last_energy;
    let incident_energy = input.relative_energies[incident];
    let i = Complex::new(0.0, 1.0);
    total += tail_scale
        * (-i * input.core_width
            + incident_energy
            + last_energy
                * ((i * input.core_width - incident_energy + last_energy) / last_energy).ln())
        / ((input.core_width + i * incident_energy).powi(2) * last_energy);

    Ok(-total / std::f64::consts::PI
        * (input.final_wave_numbers[transfer] / std::f64::consts::PI).sqrt()
        + direct)
}

fn rixs_angular_channel_range(
    angular_momentum: usize,
) -> Result<std::ops::RangeInclusive<usize>, RixsError> {
    let start = angular_momentum
        .checked_mul(angular_momentum)
        .ok_or(RixsError::InvalidAngularMomentum { value: isize::MAX })?;
    let width = angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(RixsError::InvalidAngularMomentum { value: isize::MAX })?;
    let end = start
        .checked_add(width - 1)
        .ok_or(RixsError::InvalidAngularMomentum { value: isize::MAX })?;
    Ok(start..=end)
}

fn rixs_required_angular_channels(angular_momenta: &[isize]) -> Result<usize, RixsError> {
    let mut required = 0;
    for &angular_momentum in angular_momenta {
        if angular_momentum < 0 {
            continue;
        }
        let angular_momentum =
            usize::try_from(angular_momentum).map_err(|_| RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        let required_for_l = angular_momentum
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .ok_or(RixsError::InvalidAngularMomentum {
                value: isize::try_from(angular_momentum).unwrap_or(isize::MAX),
            })?;
        required = required.max(required_for_l);
    }
    Ok(required)
}

fn rixs_required_radial_angular_channels(angular_momenta: &[isize]) -> Result<usize, RixsError> {
    let mut required = 0;
    for &angular_momentum in angular_momenta {
        if angular_momentum < 0 {
            continue;
        }
        let angular_momentum =
            usize::try_from(angular_momentum).map_err(|_| RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        required = required.max(angular_momentum + 1);
    }
    Ok(required)
}

fn rixs_linear_edge_interpolate(
    input: RixsEdgeContributionInput<'_>,
    incident: usize,
    channel: usize,
    edge: usize,
    target: Real,
) -> Result<Real, RixsError> {
    let (lower, upper) = terpc_order1_interval(input.relative_energies, target);
    let lower_energy = input.relative_energies[lower];
    let upper_energy = input.relative_energies[upper];
    let width = upper_energy - lower_energy;
    if width == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "relative_energies",
            index: lower,
        });
    }
    let lower_value = input.edge_contributions[(lower, incident, channel, edge)];
    let upper_value = input.edge_contributions[(upper, incident, channel, edge)];
    Ok(lower_value + (target - lower_energy) * (upper_value - lower_value) / width)
}

fn terpc_order1_interval(values: ArrayView1<'_, Real>, target: Real) -> (usize, usize) {
    let mut located = 0;
    let mut upper = values.len() + 1;
    while upper - located > 1 {
        let middle = (upper + located) / 2;
        if target < values[middle - 1] {
            upper = middle;
        } else {
            located = middle;
        }
    }
    let lower = located.clamp(1, values.len() - 1) - 1;
    (lower, lower + 1)
}

fn bilinear_interpolate_real(
    x: ArrayView1<'_, Real>,
    y: ArrayView1<'_, Real>,
    values: ArrayView3<'_, Real>,
    channel: usize,
    x0: Real,
    y0: Real,
) -> Result<Real, RixsError> {
    validate_range("x", x0, x[0], x[x.len() - 1])?;
    validate_range("y", y0, y[0], y[y.len() - 1])?;
    let (x_lower, x_upper) = interpolation_interval(x, x0);
    let (y_lower, y_upper) = interpolation_interval(y, y0);
    let dx = x[x_upper] - x[x_lower];
    let dy = y[y_upper] - y[y_lower];
    if dx == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "x",
            index: x_lower,
        });
    }
    if dy == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "y",
            index: y_lower,
        });
    }
    let dxdy = dx * dy;
    Ok(
        (values[(x_lower, y_lower, channel)] * (x[x_upper] - x0) * (y[y_upper] - y0)
            + values[(x_upper, y_lower, channel)] * (x0 - x[x_lower]) * (y[y_upper] - y0)
            + values[(x_lower, y_upper, channel)] * (x[x_upper] - x0) * (y0 - y[y_lower])
            + values[(x_upper, y_upper, channel)] * (x0 - x[x_lower]) * (y0 - y[y_lower]))
            / dxdy,
    )
}

fn validate_bilinear_inputs(
    x: ArrayView1<'_, Real>,
    y: ArrayView1<'_, Real>,
    values: ArrayView2<'_, Complex>,
    x0: Real,
    y0: Real,
) -> Result<(), RixsError> {
    validate_finite("x0", x0)?;
    validate_finite("y0", y0)?;
    validate_grid("x", x)?;
    validate_grid("y", y)?;
    if values.nrows() < x.len() || values.ncols() < y.len() {
        return Err(RixsError::MatrixTooSmall {
            rows: values.nrows(),
            cols: values.ncols(),
            required_rows: x.len(),
            required_cols: y.len(),
        });
    }
    validate_range("x", x0, x[0], x[x.len() - 1])?;
    validate_range("y", y0, y[0], y[y.len() - 1])?;
    for row in 0..x.len() {
        for col in 0..y.len() {
            validate_complex("values", matrix_value(values, row, col)?)?;
        }
    }
    Ok(())
}

fn validate_final_spectrum_input(input: RixsFinalSpectrumInput<'_>) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_finite("incident_window_min", input.incident_window.0)?;
    validate_finite("incident_window_max", input.incident_window.1)?;
    validate_finite("final_window_min", input.final_window.0)?;
    validate_finite("final_window_max", input.final_window.1)?;
    validate_finite("incident_edge", input.incident_edge)?;
    validate_finite("final_edge", input.final_edge)?;
    if !(input.hartree_ev.is_finite() && input.hartree_ev > 0.0) {
        return Err(RixsError::InvalidHartreeEv {
            value: input.hartree_ev,
        });
    }
    let (rows, columns, channels) = input.cross_section.dim();
    let energy_count = input.relative_energies.len();
    if channels == 0 {
        return Err(RixsError::EmptyChannelTable);
    }
    if rows != energy_count || columns != energy_count {
        return Err(RixsError::FinalSpectrumShape {
            energy_count,
            rows,
            columns,
            channels,
        });
    }
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                validate_finite("cross_section", input.cross_section[(row, column, channel)])?;
            }
        }
    }
    Ok(())
}

fn validate_edge_contribution_input(input: RixsEdgeContributionInput<'_>) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    for &split in input.edge_splits {
        validate_finite("edge_split", split)?;
    }

    let energy_count = input.relative_energies.len();
    let edge_count = input.edge_splits.len();
    let (rows, columns, channels, edges) = input.edge_contributions.dim();
    if channels == 0 {
        return Err(RixsError::EmptyEdgeContributionChannelTable);
    }
    if rows != energy_count || columns != energy_count || edges != edge_count {
        return Err(RixsError::EdgeContributionShape {
            energy_count,
            edge_count,
            rows,
            columns,
            channels,
            edges,
        });
    }
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                for edge in 0..edges {
                    validate_finite(
                        "edge_contribution",
                        input.edge_contributions[(row, column, channel, edge)],
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn validate_edge_broadening_input(input: RixsEdgeBroadeningInput<'_>) -> Result<(), RixsError> {
    validate_incident_energy_broadening_input(RixsIncidentEnergyBroadeningInput {
        relative_energies: input.relative_energies,
        edge_cross_section: input.raw_cross_section,
        self_energy: input.self_energy,
        fermi_level: input.fermi_level,
        incident_width: input.incident_width,
    })?;
    validate_width("core_width", input.core_width)?;
    validate_finite("final_width_base", input.final_width_base)?;

    let width_count = input.edge_widths.len();
    let amplitude_count = input.edge_amplitudes.len();
    if width_count == 0 {
        return Err(RixsError::EmptyEdgeBroadeningEdgeTable);
    }
    if width_count != amplitude_count {
        return Err(RixsError::EdgeBroadeningLengthMismatch {
            width_count,
            amplitude_count,
        });
    }
    for edge in 0..width_count {
        validate_finite("edge_width", input.edge_widths[edge])?;
        validate_finite("edge_amplitude", input.edge_amplitudes[edge])?;
    }
    Ok(())
}

fn validate_pole_normalization_input(
    input: RixsPoleNormalizationInput<'_>,
) -> Result<(), RixsError> {
    validate_finite("fermi_level", input.fermi_level)?;
    let energy_count = input.pole_energies.len();
    let amplitude_count = input.pole_amplitudes.len();
    let width_count = input.pole_widths.len();
    if energy_count != amplitude_count || energy_count != width_count {
        return Err(RixsError::PoleLengthMismatch {
            energy_count,
            amplitude_count,
            width_count,
        });
    }
    if energy_count < 2 {
        return Err(RixsError::InsufficientPoleRows {
            count: energy_count,
        });
    }
    for index in 0..energy_count {
        validate_finite("pole_energy", input.pole_energies[index])?;
        validate_finite("pole_amplitude", input.pole_amplitudes[index])?;
        validate_width("pole_width", input.pole_widths[index])?;
    }
    Ok(())
}

fn validate_incident_energy_broadening_input(
    input: RixsIncidentEnergyBroadeningInput<'_>,
) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_width("incident_width", input.incident_width)?;
    validate_finite("fermi_level", input.fermi_level)?;
    let energy_count = input.relative_energies.len();
    let (rows, columns, channels) = input.edge_cross_section.dim();
    let self_energy_count = input.self_energy.len();
    if channels == 0 {
        return Err(RixsError::EmptyIncidentBroadeningChannelTable);
    }
    if rows != energy_count || columns != energy_count || self_energy_count != energy_count {
        return Err(RixsError::IncidentBroadeningShape {
            energy_count,
            rows,
            columns,
            channels,
            self_energy_count,
        });
    }
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                validate_finite(
                    "edge_cross_section",
                    input.edge_cross_section[(row, column, channel)],
                )?;
            }
        }
    }
    for energy in 0..self_energy_count {
        validate_complex("self_energy", input.self_energy[energy])?;
    }
    Ok(())
}

fn validate_self_energy_grid_input(input: RixsSelfEnergyGridInput<'_>) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_finite("fermi_level", input.fermi_level)?;
    if !(input.hartree_ev.is_finite() && input.hartree_ev > 0.0) {
        return Err(RixsError::InvalidHartreeEv {
            value: input.hartree_ev,
        });
    }
    let energy_count = input.mpse_energy_ev.len();
    let self_energy_count = input.mpse_self_energy_ev.len();
    if energy_count != self_energy_count {
        return Err(RixsError::SelfEnergyGridLengthMismatch {
            energy_count,
            self_energy_count,
        });
    }

    let mpse_energy_hartree: Array1<Real> = input
        .mpse_energy_ev
        .iter()
        .map(|energy| energy / input.hartree_ev)
        .collect();
    validate_grid("mpse_energy", mpse_energy_hartree.view())?;
    for row in 0..self_energy_count {
        validate_complex("mpse_self_energy", input.mpse_self_energy_ev[row])?;
    }
    Ok(())
}

fn validate_wave_number_input(input: RixsWaveNumberInput<'_>) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_complex("incident_reference_energy", input.incident_reference_energy)?;
    validate_complex("final_reference_energy", input.final_reference_energy)?;
    Ok(())
}

fn validate_radial_grid_input(input: RixsRadialGridInput) -> Result<(), RixsError> {
    if input.point_count < 2 {
        return Err(RixsError::InsufficientGrid {
            axis: "radial_grid",
            len: input.point_count,
        });
    }
    validate_finite("log_origin", input.log_origin)?;
    validate_width("log_step", input.log_step)?;
    validate_width("muffin_tin_radius", input.muffin_tin_radius)?;
    Ok(())
}

fn validate_core_hole_potential_input(
    input: RixsCoreHolePotentialInput<'_>,
) -> Result<(), RixsError> {
    let initial_count = input.incident_screened_core_hole.len();
    if initial_count == 0 {
        return Err(RixsError::EmptyCoreHolePotentialTable);
    }
    if let Some(final_screened) = input.final_screened_core_hole {
        let final_count = final_screened.len();
        if final_count != initial_count {
            return Err(RixsError::CoreHolePotentialLengthMismatch {
                initial_count,
                final_count,
            });
        }
        for radial in 0..final_count {
            validate_finite("final_screened_core_hole", final_screened[radial])?;
        }
    }
    for radial in 0..initial_count {
        validate_finite(
            "incident_screened_core_hole",
            input.incident_screened_core_hole[radial],
        )?;
    }
    Ok(())
}

fn validate_radial_function_table_input(
    input: RixsRadialFunctionTableInput<'_>,
) -> Result<(), RixsError> {
    if input.energy_count < 1 {
        return Err(RixsError::EmptyRadialFunctionTable {
            axis: "energy",
            count: input.energy_count,
        });
    }
    if input.angular_count < 1 {
        return Err(RixsError::EmptyRadialFunctionTable {
            axis: "angular",
            count: input.angular_count,
        });
    }
    if input.radial_count < 1 {
        return Err(RixsError::EmptyRadialFunctionTable {
            axis: "radial",
            count: input.radial_count,
        });
    }
    let expected_count = input
        .energy_count
        .checked_mul(input.angular_count)
        .ok_or(RixsError::InvalidAngularMomentum { value: isize::MAX })?;
    if input.records.len() != expected_count {
        return Err(RixsError::RadialFunctionRecordCountMismatch {
            record_count: input.records.len(),
            expected_count,
        });
    }
    for (record_index, record) in input.records.iter().enumerate() {
        validate_complex("radial_function_energy", record.energy)?;
        if record.angular_momentum < 0
            || usize::try_from(record.angular_momentum)
                .map_or(true, |angular| angular >= input.angular_count)
        {
            return Err(RixsError::RadialFunctionAngularOutOfRange {
                record: record_index,
                angular_momentum: record.angular_momentum,
                angular_count: input.angular_count,
            });
        }
        let value_count = record.radial_values.len();
        if value_count != input.radial_count {
            return Err(RixsError::RadialFunctionRecordShape {
                record: record_index,
                value_count,
                radial_count: input.radial_count,
            });
        }
        for radial in 0..value_count {
            validate_complex("radial_function_values", record.radial_values[radial])?;
        }
    }
    Ok(())
}

fn validate_transition_matrix_input(input: RixsTransitionMatrixInput) -> Result<(), RixsError> {
    if input.spin_channel_count == 0 || input.spin_channel_count > 2 {
        return Err(RixsError::InvalidTransitionSpinChannelCount {
            count: input.spin_channel_count,
        });
    }
    validate_finite("spin_vector_angle", input.spin_vector_angle)?;
    for row in input.polarization_tensor {
        for value in row {
            validate_complex("polarization_tensor", value)?;
        }
    }
    Ok(())
}

fn validate_transition_phase_shift_input(
    input: RixsTransitionPhaseShiftInput<'_>,
) -> Result<(), RixsError> {
    let (energy_count, signed_l_count) = input.phase_shifts.dim();
    if energy_count == 0 {
        return Err(RixsError::EmptyTransitionPhaseShiftEnergyTable);
    }
    if signed_l_count == 0 {
        return Err(RixsError::EmptyTransitionPhaseShiftAngularTable);
    }
    if input.transition_angular_momenta.is_empty() {
        return Err(RixsError::EmptyTransitionPhaseShiftTransitionTable);
    }
    let max_signed_l = input
        .signed_l_min
        .checked_add(
            isize::try_from(signed_l_count - 1)
                .map_err(|_| RixsError::InvalidAngularMomentum { value: isize::MAX })?,
        )
        .ok_or(RixsError::InvalidAngularMomentum { value: isize::MAX })?;
    for energy in 0..energy_count {
        for signed_l in 0..signed_l_count {
            validate_complex(
                "transition_phase_shifts",
                input.phase_shifts[(energy, signed_l)],
            )?;
        }
    }
    for (transition, &angular_momentum) in input.transition_angular_momenta.iter().enumerate() {
        if angular_momentum < 0 {
            continue;
        }
        let signed_l = angular_momentum
            .checked_neg()
            .ok_or(RixsError::InvalidAngularMomentum {
                value: angular_momentum,
            })?;
        if signed_l < input.signed_l_min || signed_l > max_signed_l {
            return Err(RixsError::TransitionPhaseShiftAngularOutOfRange {
                transition,
                signed_l,
                min_signed_l: input.signed_l_min,
                max_signed_l,
            });
        }
    }
    Ok(())
}

fn validate_satellite_convolution_input(
    input: RixsSatelliteConvolutionInput<'_>,
) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_finite("fermi_level", input.fermi_level)?;
    if !(input.hartree_ev.is_finite() && input.hartree_ev > 0.0) {
        return Err(RixsError::InvalidHartreeEv {
            value: input.hartree_ev,
        });
    }
    let energy_count = input.relative_energies.len();
    let (rows, columns, channels) = input.cross_section.dim();
    if channels == 0 {
        return Err(RixsError::EmptySatelliteConvolutionChannelTable);
    }
    if rows != energy_count || columns != energy_count {
        return Err(RixsError::SatelliteConvolutionShape {
            energy_count,
            rows,
            columns,
            channels,
        });
    }
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                validate_finite("cross_section", input.cross_section[(row, column, channel)])?;
            }
        }
    }

    let xes_energy_count = input.xes_energy_ev.len();
    let xes_mu_count = input.xes_mu.len();
    if xes_energy_count != xes_mu_count {
        return Err(RixsError::SatelliteXesLengthMismatch {
            energy_count: xes_energy_count,
            mu_count: xes_mu_count,
        });
    }
    if xes_energy_count < 2 {
        return Err(RixsError::InsufficientSatelliteXesGrid {
            count: xes_energy_count,
        });
    }
    for index in 0..xes_energy_count {
        validate_finite("xes_energy_ev", input.xes_energy_ev[index])?;
        validate_finite("xes_mu", input.xes_mu[index])?;
    }
    Ok(())
}

fn validate_final_energy_broadening_input(
    input: RixsFinalEnergyBroadeningInput<'_>,
) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_width("core_width", input.core_width)?;
    validate_finite("final_width", input.final_width)?;
    validate_finite("edge_amplitude", input.edge_amplitude)?;
    let energy_count = input.relative_energies.len();
    let (rows, columns, channels) = input.edge_cross_section.dim();
    if channels == 0 {
        return Err(RixsError::EmptyFinalBroadeningChannelTable);
    }
    if rows != energy_count || columns != energy_count {
        return Err(RixsError::FinalBroadeningShape {
            energy_count,
            rows,
            columns,
            channels,
        });
    }
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                validate_finite(
                    "edge_cross_section",
                    input.edge_cross_section[(row, column, channel)],
                )?;
            }
        }
    }
    Ok(())
}

fn validate_raw_cross_section_input(input: RixsRawCrossSectionInput<'_>) -> Result<(), RixsError> {
    if input.spin_channel_count == 0 {
        return Err(RixsError::InvalidSpinChannelCount {
            count: input.spin_channel_count,
        });
    }
    let transition_count = input.transition_angular_momenta.len();
    if transition_count == 0 {
        return Err(RixsError::EmptyCrossSectionTransitionTable);
    }

    let (incident_count, transfer_count, amplitude_angular, amplitude_transitions) =
        input.transition_amplitudes.dim();
    let (green_rows, green_columns, green_energy) = input.final_green.dim();
    let (phase_energy, phase_transitions) = input.final_phase_shifts.dim();
    if amplitude_transitions != transition_count || phase_transitions != transition_count {
        return Err(RixsError::CrossSectionTransitionMismatch {
            transition_count,
            amplitude_transitions,
            phase_transitions,
        });
    }
    if incident_count == 0
        || incident_count != transfer_count
        || green_energy != transfer_count
        || phase_energy != transfer_count
    {
        return Err(RixsError::CrossSectionEnergyShape {
            incident_count,
            transfer_count,
            green_energy,
            phase_energy,
        });
    }

    let required = rixs_required_angular_channels(input.transition_angular_momenta)?;
    if amplitude_angular < required || green_rows < required || green_columns < required {
        return Err(RixsError::CrossSectionAngularShape {
            required,
            amplitude_angular,
            green_rows,
            green_columns,
        });
    }
    for incident in 0..incident_count {
        for transfer in 0..transfer_count {
            for angular in 0..amplitude_angular {
                for transition in 0..amplitude_transitions {
                    validate_complex(
                        "transition_amplitudes",
                        input.transition_amplitudes[(incident, transfer, angular, transition)],
                    )?;
                }
            }
        }
    }
    for row in 0..green_rows {
        for column in 0..green_columns {
            for energy in 0..green_energy {
                validate_complex("final_green", input.final_green[(row, column, energy)])?;
            }
        }
    }
    for energy in 0..phase_energy {
        for transition in 0..phase_transitions {
            validate_complex(
                "final_phase_shifts",
                input.final_phase_shifts[(energy, transition)],
            )?;
        }
    }
    Ok(())
}

fn validate_radial_overlap_input(input: RixsRadialOverlapInput<'_>) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_grid("radii", input.radii)?;
    validate_finite("fermi_level", input.fermi_level)?;
    validate_finite("log_step", input.log_step)?;
    if input.log_step <= 0.0 {
        return Err(RixsError::InvalidWidth {
            name: "log_step",
            value: input.log_step,
        });
    }
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(RixsError::InvalidWidth {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    let transition_count = input.transition_angular_momenta.len();
    if transition_count == 0 {
        return Err(RixsError::EmptyRadialOverlapTransitionTable);
    }

    let energy_count = input.relative_energies.len();
    let radial_count = input.radii.len();
    let potential_count = input.potential_difference.len();
    let (initial_radial, initial_angular, initial_energy) = input.initial_radial_functions.dim();
    let (final_radial, final_angular, final_energy) = input.final_radial_functions.dim();
    if potential_count != radial_count
        || initial_radial != radial_count
        || final_radial != radial_count
        || initial_energy != energy_count
        || final_energy != energy_count
    {
        return Err(RixsError::RadialOverlapShape {
            energy_count,
            radial_count,
            potential_count,
            initial_radial,
            initial_angular,
            initial_energy,
            final_radial,
            final_angular,
            final_energy,
        });
    }

    let required = rixs_required_radial_angular_channels(input.transition_angular_momenta)?;
    if initial_angular < required || final_angular < required {
        return Err(RixsError::RadialOverlapAngularShape {
            required,
            initial_angular,
            final_angular,
        });
    }
    for radial in 0..potential_count {
        validate_finite("potential_difference", input.potential_difference[radial])?;
    }
    for radial in 0..initial_radial {
        for angular in 0..initial_angular {
            for energy in 0..initial_energy {
                validate_complex(
                    "initial_radial_functions",
                    input.initial_radial_functions[(radial, angular, energy)],
                )?;
            }
        }
    }
    for radial in 0..final_radial {
        for angular in 0..final_angular {
            for energy in 0..final_energy {
                validate_complex(
                    "final_radial_functions",
                    input.final_radial_functions[(radial, angular, energy)],
                )?;
            }
        }
    }

    Ok(())
}

fn validate_initial_amplitude_input(input: RixsInitialAmplitudeInput<'_>) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_finite("fermi_level", input.fermi_level)?;
    let transition_count = input.transition_angular_momenta.len();
    if transition_count == 0 {
        return Err(RixsError::EmptyInitialAmplitudeTransitionTable);
    }

    let (incident_count, transfer_count, radial_transitions) = input.radial_overlaps.dim();
    let (moment_energy, moment_transitions, moment_spin) = input.incident_transition_moments.dim();
    let (phase_energy, phase_transitions) = input.incident_phase_shifts.dim();
    let (green_rows, green_columns, green_energy) = input.incident_green.dim();
    let normalization_energy = input.normalization.len();
    if moment_spin == 0 {
        return Err(RixsError::EmptyInitialAmplitudeSpinTable);
    }
    if radial_transitions != transition_count
        || moment_transitions != transition_count
        || phase_transitions != transition_count
    {
        return Err(RixsError::InitialAmplitudeTransitionMismatch {
            transition_count,
            radial_transitions,
            moment_transitions,
            phase_transitions,
        });
    }

    let energy_count = input.relative_energies.len();
    if incident_count != energy_count
        || transfer_count != energy_count
        || moment_energy != energy_count
        || phase_energy != energy_count
        || green_energy != energy_count
        || normalization_energy != energy_count
    {
        return Err(RixsError::InitialAmplitudeEnergyShape {
            incident_count,
            transfer_count,
            moment_energy,
            phase_energy,
            green_energy,
            normalization_energy,
        });
    }

    let required = rixs_required_angular_channels(input.transition_angular_momenta)?;
    if green_rows < required || green_columns < required {
        return Err(RixsError::InitialAmplitudeAngularShape {
            required,
            green_rows,
            green_columns,
        });
    }

    for energy in 0..normalization_energy {
        let value = input.normalization[energy];
        validate_finite("normalization", value)?;
        if value < 0.0 {
            return Err(RixsError::NegativeNormalization {
                index: energy,
                value,
            });
        }
    }
    for incident in 0..incident_count {
        for transfer in 0..transfer_count {
            for transition in 0..radial_transitions {
                validate_finite(
                    "radial_overlaps",
                    input.radial_overlaps[(incident, transfer, transition)],
                )?;
            }
        }
    }
    for incident in 0..moment_energy {
        for transition in 0..moment_transitions {
            for spin in 0..moment_spin {
                validate_complex(
                    "incident_transition_moments",
                    input.incident_transition_moments[(incident, transition, spin)],
                )?;
            }
        }
    }
    for incident in 0..phase_energy {
        for transition in 0..phase_transitions {
            validate_complex(
                "incident_phase_shifts",
                input.incident_phase_shifts[(incident, transition)],
            )?;
        }
    }
    for row in 0..green_rows {
        for column in 0..green_columns {
            for energy in 0..green_energy {
                validate_complex(
                    "incident_green",
                    input.incident_green[(row, column, energy)],
                )?;
            }
        }
    }

    Ok(())
}

fn validate_direct_final_transition_input(
    input: RixsDirectFinalTransitionInput<'_>,
) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_finite("fermi_level", input.fermi_level)?;
    let transition_count = input.transition_angular_momenta.len();
    if transition_count == 0 {
        return Err(RixsError::EmptyDirectFinalTransitionTransitionTable);
    }

    let (moment_energy, moment_transitions, moment_spin) = input.final_transition_moments.dim();
    let (phase_energy, phase_transitions) = input.final_phase_shifts.dim();
    let normalization_energy = input.normalization.len();
    if moment_spin == 0 {
        return Err(RixsError::EmptyDirectFinalTransitionSpinTable);
    }
    if moment_transitions != transition_count || phase_transitions != transition_count {
        return Err(RixsError::DirectFinalTransitionMismatch {
            transition_count,
            moment_transitions,
            phase_transitions,
        });
    }

    let energy_count = input.relative_energies.len();
    if moment_energy != energy_count
        || phase_energy != energy_count
        || normalization_energy != energy_count
    {
        return Err(RixsError::DirectFinalTransitionEnergyShape {
            energy_count,
            moment_energy,
            phase_energy,
            normalization_energy,
        });
    }

    for energy in 0..normalization_energy {
        let value = input.normalization[energy];
        validate_finite("normalization", value)?;
        if value < 0.0 {
            return Err(RixsError::NegativeNormalization {
                index: energy,
                value,
            });
        }
    }
    for transfer in 0..moment_energy {
        for transition in 0..moment_transitions {
            for spin in 0..moment_spin {
                validate_complex(
                    "final_transition_moments",
                    input.final_transition_moments[(transfer, transition, spin)],
                )?;
            }
        }
    }
    for transfer in 0..phase_energy {
        for transition in 0..phase_transitions {
            validate_complex(
                "final_phase_shifts",
                input.final_phase_shifts[(transfer, transition)],
            )?;
        }
    }
    Ok(())
}

fn validate_incident_amplitude_convolution_input(
    input: RixsIncidentAmplitudeConvolutionInput<'_>,
) -> Result<(), RixsError> {
    validate_grid("relative_energies", input.relative_energies)?;
    validate_finite("fermi_level", input.fermi_level)?;
    validate_width("core_width", input.core_width)?;
    let transition_count = input.transition_angular_momenta.len();
    if transition_count == 0 {
        return Err(RixsError::EmptyCrossSectionTransitionTable);
    }

    let (incident_count, transfer_count, amplitude_angular, amplitude_transitions) =
        input.transition_amplitudes.dim();
    let (moment_energy, moment_transitions, moment_spin) = input.final_transition_moments.dim();
    let (phase_energy, phase_transitions) = input.final_phase_shifts.dim();
    let wave_energy = input.final_wave_numbers.len();
    let normalization_energy = input.normalization.len();
    let (bmat_angular, bmat_transitions, bmat_spin) = input.b_matrix_diagonal.dim();
    if moment_spin == 0 || bmat_spin == 0 {
        return Err(RixsError::EmptyIncidentConvolutionSpinTable);
    }
    if amplitude_transitions != transition_count
        || moment_transitions != transition_count
        || phase_transitions != transition_count
        || bmat_transitions != transition_count
    {
        return Err(RixsError::IncidentConvolutionTransitionMismatch {
            transition_count,
            amplitude_transitions,
            moment_transitions,
            phase_transitions,
            bmat_transitions,
        });
    }
    if incident_count == 0
        || incident_count != transfer_count
        || moment_energy != transfer_count
        || phase_energy != transfer_count
        || wave_energy != transfer_count
        || normalization_energy != transfer_count
    {
        return Err(RixsError::IncidentConvolutionEnergyShape {
            incident_count,
            transfer_count,
            moment_energy,
            phase_energy,
            wave_energy,
            normalization_energy,
        });
    }
    if moment_spin != bmat_spin {
        return Err(RixsError::IncidentConvolutionSpinShape {
            moment_spin,
            bmat_spin,
        });
    }

    let required = rixs_required_angular_channels(input.transition_angular_momenta)?;
    if amplitude_angular < required || bmat_angular < required {
        return Err(RixsError::IncidentConvolutionAngularShape {
            required,
            amplitude_angular,
            bmat_angular,
        });
    }
    for energy in 0..normalization_energy {
        let value = input.normalization[energy];
        validate_finite("normalization", value)?;
        if value < 0.0 {
            return Err(RixsError::NegativeNormalization {
                index: energy,
                value,
            });
        }
    }
    for incident in 0..incident_count {
        for transfer in 0..transfer_count {
            for angular in 0..amplitude_angular {
                for transition in 0..amplitude_transitions {
                    validate_complex(
                        "transition_amplitudes",
                        input.transition_amplitudes[(incident, transfer, angular, transition)],
                    )?;
                }
            }
        }
    }
    for transfer in 0..moment_energy {
        for transition in 0..moment_transitions {
            for spin in 0..moment_spin {
                validate_complex(
                    "final_transition_moments",
                    input.final_transition_moments[(transfer, transition, spin)],
                )?;
            }
        }
    }
    for transfer in 0..phase_energy {
        for transition in 0..phase_transitions {
            validate_complex(
                "final_phase_shifts",
                input.final_phase_shifts[(transfer, transition)],
            )?;
        }
    }
    for (index, &wave_number) in input.final_wave_numbers.iter().enumerate() {
        validate_complex("final_wave_numbers", wave_number).map_err(|_| {
            RixsError::NonFiniteComplex {
                name: "final_wave_numbers",
                real: input.final_wave_numbers[index].re,
                imaginary: input.final_wave_numbers[index].im,
            }
        })?;
    }
    for angular in 0..bmat_angular {
        for transition in 0..bmat_transitions {
            for spin in 0..bmat_spin {
                validate_complex(
                    "b_matrix_diagonal",
                    input.b_matrix_diagonal[(angular, transition, spin)],
                )?;
            }
        }
    }
    Ok(())
}

fn validate_grid(axis: &'static str, values: ArrayView1<'_, Real>) -> Result<(), RixsError> {
    if values.len() < 2 {
        return Err(RixsError::InsufficientGrid {
            axis,
            len: values.len(),
        });
    }

    let mut previous = values[0];
    validate_finite(axis, previous)?;
    for (index, &current) in values.iter().enumerate().skip(1) {
        validate_finite(axis, current)?;
        if current <= previous {
            return Err(RixsError::NonIncreasingGrid {
                axis,
                index,
                previous,
                current,
            });
        }
        previous = current;
    }
    Ok(())
}

fn validate_range(axis: &'static str, value: Real, min: Real, max: Real) -> Result<(), RixsError> {
    if value < min - BL_INTERP_TOLERANCE || value > max + BL_INTERP_TOLERANCE {
        Err(RixsError::OutOfRange {
            axis,
            value,
            min,
            max,
            tolerance: BL_INTERP_TOLERANCE,
        })
    } else {
        Ok(())
    }
}

fn interpolation_interval(values: ArrayView1<'_, Real>, target: Real) -> (usize, usize) {
    let (mut lower, mut upper) = values
        .iter()
        .position(|&value| value >= target)
        .map_or((-1, (values.len() * 2) as isize), |index| {
            (index as isize - 1, index as isize)
        });

    if lower < 0 {
        lower = 0;
        upper = 1;
    }
    if upper > values.len() as isize - 1 {
        upper = values.len() as isize - 1;
        lower = values.len() as isize - 2;
    }
    (lower as usize, upper as usize)
}

fn matrix_value(
    values: ArrayView2<'_, Complex>,
    row: usize,
    col: usize,
) -> Result<Complex, RixsError> {
    values
        .get((row, col))
        .copied()
        .ok_or(RixsError::MatrixTooSmall {
            rows: values.nrows(),
            cols: values.ncols(),
            required_rows: row + 1,
            required_cols: col + 1,
        })
}

fn validate_width(name: &'static str, value: Real) -> Result<(), RixsError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(RixsError::InvalidWidth { name, value })
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), RixsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RixsError::NonFiniteReal { name, value })
    }
}

fn validate_complex(name: &'static str, value: Complex) -> Result<(), RixsError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(RixsError::NonFiniteComplex {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ShapeBuilder;

    use super::*;

    fn assert_real_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_complex_close_tol(actual, expected, 1.0e-12);
    }

    fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
        assert!(
            (actual - expected).norm() < tolerance,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn assert_array_close(actual: &[Real], expected: &[Real]) {
        assert_array_close_tol(actual, expected, 1.0e-12);
    }

    fn assert_array_close_tol(actual: &[Real], expected: &[Real], tolerance: Real) {
        assert_eq!(actual.len(), expected.len());
        for (&actual_value, &expected_value) in actual.iter().zip(expected) {
            assert!(
                (actual_value - expected_value).abs() < tolerance,
                "actual={actual_value}, expected={expected_value}, diff={}",
                (actual_value - expected_value).abs()
            );
        }
    }

    fn sample_transition_matrix_input() -> RixsTransitionMatrixInput {
        RixsTransitionMatrixInput {
            lmax: 3,
            hole: 1,
            polarization: 1,
            polarization_tensor: [
                [
                    Complex::new(0.20, -0.05),
                    Complex::new(-0.10, 0.04),
                    Complex::new(0.03, 0.02),
                ],
                [
                    Complex::new(0.11, -0.07),
                    Complex::new(0.50, 0.00),
                    Complex::new(-0.08, 0.09),
                ],
                [
                    Complex::new(0.06, 0.01),
                    Complex::new(0.13, -0.02),
                    Complex::new(0.17, 0.03),
                ],
            ],
            multipole: 2,
            trace_orbital: false,
            spin: 1,
            spin_channel_count: 1,
            spin_vector_angle: 0.3,
        }
    }

    fn interpolation_fixture() -> (
        ndarray::Array1<Real>,
        ndarray::Array1<Real>,
        ndarray::Array2<Complex>,
    ) {
        let x = ndarray::arr1(&[0.0, 1.0, 2.5]);
        let y = ndarray::arr1(&[-1.0, 0.5, 2.0, 4.0]);
        let mut values = ndarray::Array2::zeros((x.len(), y.len()).f());
        for col in 0..y.len() {
            for row in 0..x.len() {
                let fortran_row = row as Real + 1.0;
                let fortran_col = col as Real + 1.0;
                values[(row, col)] = Complex::new(
                    10.0 * fortran_row + fortran_col,
                    -1.5 * fortran_row + 0.25 * fortran_col,
                );
            }
        }
        (x, y, values)
    }

    #[test]
    fn kk_integral_matches_feff_reference() -> Result<(), RixsError> {
        let slope = Complex::new(0.7, -0.2);
        let intercept = Complex::new(1.1, 0.3);
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, 0.4)?,
            Complex::new(2.399_207_722_391_849_5, -4.331_304_425_751_682),
        );
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, -2.5)?,
            Complex::new(1.408_013_813_215_639, 0.155_788_642_294_960_7),
        );
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, -1.0)?,
            Complex::new(-2.912_583_862_845_679, 1.467_861_035_772_940_7),
        );
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, 2.0)?,
            Complex::new(1.068_803_131_267_718_9, -2.067_433_969_813_704_3),
        );
        Ok(())
    }

    #[test]
    fn double_lorentz_matches_feff_reference() -> Result<(), RixsError> {
        assert_real_close(
            integrated_double_lorentz(3.1, 2.7, 0.45, 0.3, 1.2, -0.08, Some(5.0))?,
            1.117_803_997_544_239_5,
        );
        assert_real_close(
            integrated_double_lorentz(1.4, 2.2, 0.25, 0.65, -0.7, 0.18, Some(1.9))?,
            -0.348_748_408_558_602_4,
        );
        assert_real_close(
            integrated_double_lorentz(3.1, 2.7, 0.45, 0.3, 1.2, -0.08, None)?,
            1.384_083_044_982_698_6,
        );
        Ok(())
    }

    #[test]
    fn bilinear_interpolation_matches_feff_reference() -> Result<(), RixsError> {
        let (x, y, values) = interpolation_fixture();
        assert_complex_close(
            bilinear_interpolate_complex(x.view(), y.view(), values.view(), 0.4, 1.1)?,
            Complex::new(16.400_000_000_000_002, -1.5),
        );
        assert_complex_close(
            bilinear_interpolate_complex(
                x.view(),
                y.view(),
                values.view(),
                -0.000_004,
                -1.000_003,
            )?,
            Complex::new(10.999_958, -1.249_994_5),
        );
        assert_complex_close(
            bilinear_interpolate_complex(x.view(), y.view(), values.view(), 2.500_004, 4.000_002)?,
            Complex::new(39.333_374_666_666_68, -4.166_672_333_333_331),
        );
        Ok(())
    }

    #[test]
    fn final_spectrum_matches_feff_rixs_output_block_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.2, 0.0, 0.35, 0.9]);
        let cross_section =
            ndarray::Array3::from_shape_fn((4, 4, 2).f(), |(row, column, channel)| {
                let fortran_row = row as Real + 1.0;
                let fortran_column = column as Real + 1.0;
                if channel == 0 {
                    0.45 + 0.17 * fortran_row
                        + 0.11 * fortran_column
                        + 0.015 * fortran_row * fortran_column
                } else {
                    1.10 - 0.08 * fortran_row
                        + 0.13 * fortran_column
                        + 0.02 * fortran_row * fortran_row
                }
            });

        let spectrum = rixs_final_spectrum(RixsFinalSpectrumInput {
            relative_energies: energies.view(),
            cross_section: cross_section.view(),
            incident_window: (-0.05, 0.80),
            final_window: (0.10, 1.00),
            incident_edge: 1.20,
            final_edge: 0.40,
            hartree_ev: 27.211_396,
        })?;

        assert_array_close(
            spectrum.incident_xas_energy_ev.as_slice().unwrap(),
            &[27.211_396, 32.653_675_2, 42.177_663_8, 57.143_931_6],
        );
        assert_array_close(
            spectrum.incident_xas.as_slice().unwrap(),
            &[
                0.525_375, 0.638, 0.6035, 0.7095, 0.681_625, 0.781, 0.759_75, 0.8525,
            ],
        );
        assert_array_close(
            spectrum.final_xas_energy_ev.as_slice().unwrap(),
            &[16.326_837_6, 21.769_116_8, 31.293_105_4, 46.259_373_2],
        );
        assert_array_close(
            spectrum.final_xas.as_slice().unwrap(),
            &[0.908, 1.3, 1.103, 1.282, 1.298, 1.3, 1.493, 1.354],
        );
        assert_array_close(
            spectrum.herfd.as_slice().unwrap(),
            &[0.745, 1.17, 1.07, 1.28, 1.425, 1.43, 1.81, 1.62],
        );
        assert_array_close(
            spectrum.rixs_et_first_energy_ev.as_slice().unwrap(),
            &[
                27.211_396,
                32.653_675_2,
                42.177_663_8,
                57.143_931_6,
                27.211_396,
                32.653_675_2,
                42.177_663_8,
                57.143_931_6,
                27.211_396,
                32.653_675_2,
                42.177_663_8,
                57.143_931_6,
                27.211_396,
                32.653_675_2,
                42.177_663_8,
                57.143_931_6,
            ],
        );
        assert_array_close(
            spectrum.rixs_et_second_energy_ev.as_slice().unwrap(),
            &[
                5.442_279_2,
                5.442_279_2,
                5.442_279_2,
                5.442_279_2,
                10.884_558_4,
                10.884_558_4,
                10.884_558_4,
                10.884_558_4,
                20.408_547,
                20.408_547,
                20.408_547,
                20.408_547,
                35.374_814_8,
                35.374_814_8,
                35.374_814_8,
                35.374_814_8,
            ],
        );
        assert_array_close(
            spectrum.rixs_et.as_slice().unwrap(),
            &[
                0.745, 1.17, 0.87, 1.3, 0.995, 1.43, 1.12, 1.56, 0.93, 1.15, 1.07, 1.28, 1.21,
                1.41, 1.35, 1.54, 1.115, 1.17, 1.27, 1.3, 1.425, 1.43, 1.58, 1.56, 1.3, 1.23, 1.47,
                1.36, 1.64, 1.49, 1.81, 1.62,
            ],
        );
        assert_array_close(
            spectrum.rixs_ee_incident_energy_ev.as_slice().unwrap(),
            &[
                27.211_396,
                37.188_907_866_666_67,
                47.166_419_733_333_335,
                57.143_931_6,
                27.211_396,
                37.188_907_866_666_67,
                47.166_419_733_333_335,
                57.143_931_6,
                27.211_396,
                37.188_907_866_666_67,
                47.166_419_733_333_335,
                57.143_931_6,
                27.211_396,
                37.188_907_866_666_67,
                47.166_419_733_333_335,
                57.143_931_6,
            ],
        );
        assert_array_close(
            spectrum.rixs_ee_emission_energy_ev.as_slice().unwrap(),
            &[
                -8.163_418_8,
                -8.163_418_8,
                -8.163_418_8,
                -8.163_418_8,
                11.791_604_933_333_33,
                11.791_604_933_333_33,
                11.791_604_933_333_33,
                11.791_604_933_333_33,
                31.746_628_666_666_663,
                31.746_628_666_666_663,
                31.746_628_666_666_663,
                31.746_628_666_666_663,
                51.701_652_4,
                51.701_652_4,
                51.701_652_4,
                51.701_652_4,
            ],
        );
        assert_array_close(
            spectrum.rixs_ee.as_slice().unwrap(),
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.929_523_809_523_809_5,
                1.361_904_761_904_762,
                1.361_428_571_428_571_7,
                1.462_857_142_857_143,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.018_095_238_095_238_4,
                1.159_523_809_523_809_6,
                1.412_857_142_857_143,
                1.381_904_761_904_762,
                2.246_666_666_666_668,
                1.363_333_333_333_332_8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.762_499_999_999_999_7,
                1.060_000_000_000_000_7,
            ],
        );
        Ok(())
    }

    #[test]
    fn edge_contribution_sum_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.30, 0.10, 0.80, 1.70]);
        let edge_splits = ndarray::arr1(&[0.00, 0.25, -0.45]);
        let edge_contributions = ndarray::Array4::from_shape_fn(
            (4, 4, 2, 3).f(),
            |(transfer, incident, channel, edge)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_incident = incident as Real + 1.0;
                let fortran_channel = channel as Real + 1.0;
                let fortran_edge = edge as Real + 1.0;
                0.20 + 0.41 * fortran_transfer + 0.07 * fortran_incident + 0.11 * fortran_channel
                    - 0.05 * fortran_edge
                    + 0.025 * fortran_transfer * fortran_edge
                    + 0.01 * fortran_incident * fortran_channel
            },
        );

        let summed = rixs_sum_edge_contributions(RixsEdgeContributionInput {
            relative_energies: energies.view(),
            edge_splits: edge_splits.view(),
            edge_contributions: edge_contributions.view(),
        })?;

        assert_array_close(
            summed.as_slice().unwrap(),
            &[
                2.769_642_857_142_857_3,
                3.129_642_857_142_857,
                3.009_642_857_142_857_5,
                3.399_642_857_142_857,
                3.249_642_857_142_857_3,
                3.669_642_857_142_856_8,
                3.489_642_857_142_857,
                3.939_642_857_142_857_2,
                3.654_285_714_285_715,
                4.014_285_714_285_714,
                3.894_285_714_285_714_6,
                4.284_285_714_285_714,
                4.134_285_714_285_714,
                4.554_285_714_285_714,
                4.374_285_714_285_715,
                4.824_285_714_285_715,
                5.088_214_285_714_286,
                5.448_214_285_714_286,
                5.328_214_285_714_286,
                5.718_214_285_714_286,
                5.568_214_285_714_285,
                5.988_214_285_714_285,
                5.808_214_285_714_286,
                6.258_214_285_714_286,
                6.504_722_222_222_221,
                6.864_722_222_222_223,
                6.744_722_222_222_223,
                7.134_722_222_222_223,
                6.984_722_222_222_221,
                7.404_722_222_222_222,
                7.224_722_222_222_223,
                7.674_722_222_222_224,
            ],
        );
        Ok(())
    }

    #[test]
    fn edge_broadening_pipeline_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.20, 0.10, 0.55, 1.20]);
        let self_energy = ndarray::Array1::from_iter((1..=4).map(|index| {
            let value = index as Real;
            Complex::new(
                0.012 * value - 0.003 * value * value,
                -0.020 - 0.008 * value + 0.002 * value * value,
            )
        }));
        let raw_cross_section =
            ndarray::Array3::from_shape_fn((4, 4, 2).f(), |(transfer, incident, channel)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_incident = incident as Real + 1.0;
                let fortran_channel = channel as Real + 1.0;
                0.25 + 0.14 * fortran_transfer
                    + 0.08 * fortran_incident
                    + 0.05 * fortran_channel
                    + 0.012 * fortran_transfer * fortran_incident
                    - 0.004 * fortran_transfer * fortran_channel
                    + 0.003 * fortran_incident * fortran_incident
            });
        let edge_widths = ndarray::arr1(&[0.0001, 0.27]);
        let edge_amplitudes = ndarray::arr1(&[0.75, 1.40]);

        let broadened = rixs_broaden_edge_contributions(RixsEdgeBroadeningInput {
            relative_energies: energies.view(),
            raw_cross_section: raw_cross_section.view(),
            self_energy: self_energy.view(),
            fermi_level: 0.0,
            core_width: 0.13,
            incident_width: 0.16,
            final_width_base: 0.0,
            edge_widths: edge_widths.view(),
            edge_amplitudes: edge_amplitudes.view(),
        })?;

        let expected = [
            ((0, 0, 0, 0), 17.698_292_010_491_215),
            ((1, 2, 1, 0), 2.866_227_967_272_558_4),
            ((3, 1, 0, 0), 0.537_777_015_048_798_6),
            ((0, 0, 0, 1), 5.613_223_265_596_719),
            ((2, 3, 1, 1), 9.073_099_003_784_598),
            ((3, 3, 0, 1), 34.043_958_299_568_99),
        ];
        for (index, expected) in expected {
            assert_real_close(broadened[index], expected);
        }
        Ok(())
    }

    #[test]
    fn post_raw_spectrum_pipeline_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.20, 0.10, 0.55, 1.20]);
        let self_energy = ndarray::Array1::from_iter((1..=4).map(|index| {
            let value = index as Real;
            Complex::new(
                0.012 * value - 0.003 * value * value,
                -0.020 - 0.008 * value + 0.002 * value * value,
            )
        }));
        let raw_cross_section =
            ndarray::Array3::from_shape_fn((4, 4, 2).f(), |(transfer, incident, channel)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_incident = incident as Real + 1.0;
                let fortran_channel = channel as Real + 1.0;
                0.25 + 0.14 * fortran_transfer
                    + 0.08 * fortran_incident
                    + 0.05 * fortran_channel
                    + 0.012 * fortran_transfer * fortran_incident
                    - 0.004 * fortran_transfer * fortran_channel
                    + 0.003 * fortran_incident * fortran_incident
            });
        let edge_splits = ndarray::arr1(&[0.0, 0.25]);
        let edge_widths = ndarray::arr1(&[0.0001, 0.27]);
        let edge_amplitudes = ndarray::arr1(&[0.75, 1.40]);

        let assembled = rixs_post_raw_spectrum(RixsPostRawSpectrumInput {
            relative_energies: energies.view(),
            raw_cross_section: raw_cross_section.view(),
            self_energy: self_energy.view(),
            fermi_level: 0.0,
            core_width: 0.13,
            incident_width: 0.16,
            final_width_base: 0.0,
            edge_splits: edge_splits.view(),
            edge_widths: edge_widths.view(),
            edge_amplitudes: edge_amplitudes.view(),
            incident_window: (0.05, 0.95),
            final_window: (0.00, 1.25),
            incident_edge: 0.30,
            final_edge: 0.08,
            hartree_ev: 27.211_396,
        })?;

        let summed_expected = [
            ((0, 0, 0), 23.311_515_276_087_935),
            ((1, 2, 1), 8.889_405_808_949_395),
            ((3, 1, 0), 5.280_783_794_536_031),
            ((2, 3, 1), 7.892_428_058_654_735),
        ];
        for (index, expected) in summed_expected {
            assert_real_close(assembled.summed_cross_section[index], expected);
        }
        let herfd_expected = [
            ((0, 0), 23.311_515_276_087_935),
            ((1, 1), 45.448_641_738_434_56),
            ((3, 0), 81.367_401_613_176_53),
        ];
        for (index, expected) in herfd_expected {
            assert_real_close(assembled.spectrum.herfd[index], expected);
        }
        let rixs_et_expected = [
            ((0, 0), 23.311_515_276_087_935),
            ((2, 1), 5.625_349_899_630_866),
            ((5, 0), 43.400_533_235_684_04),
            ((15, 1), 83.499_477_703_009_86),
        ];
        for (index, expected) in rixs_et_expected {
            assert_real_close(assembled.spectrum.rixs_et[index], expected);
        }
        let incident_xas_expected = [
            ((0, 0), 7.733_963_606_764_034),
            ((1, 1), 22.985_429_717_954_68),
            ((3, 0), 3.095_165_125_358_989_6),
        ];
        for (index, expected) in incident_xas_expected {
            assert_real_close(assembled.spectrum.incident_xas[index], expected);
        }
        Ok(())
    }

    #[test]
    fn pole_normalization_matches_feff_rixs_readpoles_reference() -> Result<(), RixsError> {
        let pole_energies = ndarray::arr1(&[1.30, 0.42, -0.30]);
        let pole_amplitudes = ndarray::arr1(&[0.55, 1.20, -0.40]);
        let pole_widths = ndarray::arr1(&[0.14, 0.05, 0.07]);

        let normalized = rixs_normalize_poles(RixsPoleNormalizationInput {
            pole_energies: pole_energies.view(),
            pole_amplitudes: pole_amplitudes.view(),
            pole_widths: pole_widths.view(),
            fermi_level: 0.18,
        })?;

        assert_real_close(normalized.incident_edge, 1.12);
        assert_real_close(normalized.final_edge, -0.36);
        assert_real_close(normalized.core_width, 0.14);
        assert_array_close(normalized.edge_splits.as_slice().unwrap(), &[0.18, 0.78]);
        assert_array_close(normalized.edge_amplitudes.as_slice().unwrap(), &[1.0, 1.2]);
        assert_array_close(normalized.edge_widths.as_slice().unwrap(), &[0.07, 0.05]);

        let single_final = rixs_normalize_poles(RixsPoleNormalizationInput {
            pole_energies: ndarray::arr1(&[1.30, 0.42]).view(),
            pole_amplitudes: ndarray::arr1(&[0.55, 1.20]).view(),
            pole_widths: ndarray::arr1(&[0.14, 0.05]).view(),
            fermi_level: 0.18,
        })?;
        assert_real_close(single_final.incident_edge, 1.12);
        assert_real_close(single_final.final_edge, 0.42);
        assert_array_close(single_final.edge_splits.as_slice().unwrap(), &[0.0]);
        Ok(())
    }

    #[test]
    fn default_pole_normalization_matches_feff_rixs_no_readpoles_branch() -> Result<(), RixsError> {
        let normalized = rixs_default_pole_normalization(0.14)?;

        assert_real_close(normalized.incident_edge, 0.0);
        assert_real_close(normalized.final_edge, 0.0);
        assert_real_close(normalized.core_width, 0.14);
        assert_array_close(normalized.edge_splits.as_slice().unwrap(), &[0.0]);
        assert_array_close(normalized.edge_amplitudes.as_slice().unwrap(), &[1.0]);
        assert_array_close(normalized.edge_widths.as_slice().unwrap(), &[0.0]);
        Ok(())
    }

    #[test]
    fn incident_energy_broadening_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.25, 0.05, 0.42, 0.95, 1.70]);
        let self_energy = ndarray::Array1::from_iter((1..=5).map(|index| {
            let value = index as Real;
            Complex::new(
                0.015 * value - 0.004 * value * value,
                -0.025 - 0.011 * value + 0.003 * value * value,
            )
        }));
        let edge_cross_section =
            ndarray::Array3::from_shape_fn((5, 5, 2).f(), |(transfer, incident, channel)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_incident = incident as Real + 1.0;
                let fortran_channel = channel as Real + 1.0;
                0.35 + 0.12 * fortran_transfer
                    + 0.07 * fortran_incident
                    + 0.04 * fortran_channel
                    + 0.015 * fortran_transfer * fortran_incident
                    + 0.006 * fortran_incident * fortran_incident
                    - 0.003 * fortran_transfer * fortran_channel
            });

        let broadened = rixs_incident_energy_broadening(RixsIncidentEnergyBroadeningInput {
            relative_energies: energies.view(),
            edge_cross_section: edge_cross_section.view(),
            self_energy: self_energy.view(),
            fermi_level: 0.10,
            incident_width: 0.18,
        })?;

        let expected = [
            ((0, 0, 0), 0.461_944_586_698_008_53),
            ((2, 1, 0), 0.889_382_066_651_542),
            ((4, 0, 0), 0.563_000_000_000_000_1),
            ((0, 4, 0), 0.541),
            ((1, 3, 1), 1.011_617_853_758_565),
            ((3, 2, 1), 1.226_814_399_976_938_3),
            ((4, 4, 1), 1.825_225_597_330_654_8),
        ];
        for (index, value) in expected {
            assert_real_close(broadened[index], value);
        }
        Ok(())
    }

    #[test]
    fn self_energy_grid_matches_feff_rixs_readsigma_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.15, 0.05, 0.22, 0.48, 0.92]);
        let mpse_energy_ev = ndarray::arr1(&[-2.0, 4.0, 12.0, 22.0]);
        let mpse_self_energy_ev = ndarray::arr1(&[
            Complex::new(0.45, -0.80),
            Complex::new(0.70, -1.10),
            Complex::new(1.05, -1.45),
            Complex::new(1.60, -2.10),
        ]);

        let self_energy = rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
            relative_energies: energies.view(),
            mpse_energy_ev: mpse_energy_ev.view(),
            mpse_self_energy_ev: mpse_self_energy_ev.view(),
            fermi_level: 0.12,
            hartree_ev: FEFF_HARTREE_EV,
        })?;

        let expected = [
            Complex::new(0.016_537_189_051_234_27, -0.029_399_447_202_194_257),
            Complex::new(0.016_537_189_051_234_27, -0.029_399_447_202_194_257),
            Complex::new(0.023_766_298_134_796_172, -0.038_074_378_102_468_54),
            Complex::new(0.035_043_387_226_439_98, -0.049_743_110_827_537_11),
            Complex::new(0.058_798_894_404_388_51, -0.077_173_548_905_759_92),
        ];
        for (actual, expected) in self_energy.iter().zip(expected) {
            assert_complex_close(*actual, expected);
        }
        Ok(())
    }

    #[test]
    fn wave_numbers_match_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.12, 0.05, 0.32, 1.10]);

        let wave_numbers = rixs_wave_numbers(RixsWaveNumberInput {
            relative_energies: energies.view(),
            incident_reference_energy: Complex::new(0.08, -0.025),
            final_reference_energy: Complex::new(0.20, 0.17),
        })?;

        let expected_incident = [
            Complex::new(3.945_178_966_110_081_4e-2, 6.336_848_141_682_613e-1),
            Complex::new(9.513_804_906_310_237e-2, 2.627_760_422_480_201_5e-1),
            Complex::new(6.937_568_523_915_804e-1, 3.603_568_010_004_914e-2),
            Complex::new(1.428_392_917_425_385, 1.750_218_703_482_610_3e-2),
        ];
        let expected_final = [
            Complex::new(0.0, 8.0e-1),
            Complex::new(0.0, 5.477_225_575_051_662e-1),
            Complex::new(4.898_979_485_566_356e-1, 0.0),
            Complex::new(1.341_640_786_499_873_8, 0.0),
        ];
        for (actual, expected) in wave_numbers
            .incident_wave_numbers
            .iter()
            .zip(expected_incident)
        {
            assert_complex_close(*actual, expected);
        }
        for (actual, expected) in wave_numbers.final_wave_numbers.iter().zip(expected_final) {
            assert_complex_close(*actual, expected);
        }
        Ok(())
    }

    #[test]
    fn radial_setup_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let grid = rixs_radial_grid(RixsRadialGridInput {
            point_count: 6,
            log_origin: 0.45,
            log_step: 0.17,
            muffin_tin_radius: 1.08,
        })?;
        assert_eq!(grid.muffin_tin_index_fortran, 4);
        assert_eq!(grid.active_point_count, 5);
        assert_array_close(
            grid.radii.as_slice().unwrap(),
            &[
                0.637_628_151_621_773,
                0.755_783_741_455_725,
                0.895_834_135_296_528,
                1.061_836_546_545_36,
                1.258_600_009_929_48,
                1.491_824_697_641_27,
            ],
        );

        let incident_screened = ndarray::Array1::from_iter((1..=6).map(|index| {
            let index = index as Real;
            0.8 + 0.13 * index - 0.01 * index * index
        }));
        let final_screened = ndarray::Array1::from_iter((1..=6).map(|index| {
            let index = index as Real;
            -0.2 + 0.07 * index + 0.005 * index * index
        }));

        let difference = rixs_core_hole_potential_difference(RixsCoreHolePotentialInput {
            incident_screened_core_hole: incident_screened.view(),
            final_screened_core_hole: Some(final_screened.view()),
        })?;
        assert_array_close(
            difference.as_slice().unwrap(),
            &[-1.045, -1.06, -1.045, -1.0, -0.925, -0.82],
        );

        let val_difference = rixs_core_hole_potential_difference(RixsCoreHolePotentialInput {
            incident_screened_core_hole: incident_screened.view(),
            final_screened_core_hole: None,
        })?;
        assert_array_close(
            val_difference.as_slice().unwrap(),
            &[-0.92, -1.02, -1.1, -1.16, -1.2, -1.22],
        );
        Ok(())
    }

    #[test]
    fn radial_function_table_matches_feff_rixs_read_loop_reference() -> Result<(), RixsError> {
        let labels = [0, 2, 1, 1, 0, 2];
        let energies = [
            Complex::new(0.15, 0.01),
            Complex::new(0.16, 0.02),
            Complex::new(0.17, 0.03),
            Complex::new(0.31, -0.01),
            Complex::new(0.32, -0.02),
            Complex::new(0.33, -0.03),
        ];
        let values: Vec<_> = labels
            .iter()
            .enumerate()
            .map(|(record, &angular)| {
                let energy_index = record / 3;
                ndarray::Array1::from_iter((1..=4).map(|radial| {
                    let radial = radial as Real;
                    Complex::new(
                        10.0 * (energy_index as Real + 1.0) + angular as Real + 0.1 * radial,
                        -0.5 * angular as Real + 0.02 * radial,
                    )
                }))
            })
            .collect();
        let records: Vec<_> = labels
            .iter()
            .zip(energies.iter())
            .zip(values.iter())
            .map(
                |((&angular_momentum, &energy), radial_values)| RixsRadialFunctionRecord {
                    energy,
                    angular_momentum: angular_momentum as isize,
                    radial_values: radial_values.view(),
                },
            )
            .collect();

        let table = rixs_radial_function_table(RixsRadialFunctionTableInput {
            records: &records,
            energy_count: 2,
            angular_count: 3,
            radial_count: 4,
        })?;

        assert_complex_close(table.energies[0], Complex::new(0.17, 0.03));
        assert_complex_close(table.energies[1], Complex::new(0.33, -0.03));
        let expected = [
            ((0, 0, 0), Complex::new(10.1, 0.02)),
            ((1, 0, 0), Complex::new(10.2, 0.04)),
            ((3, 0, 0), Complex::new(10.4, 0.08)),
            ((0, 1, 0), Complex::new(11.1, -0.48)),
            ((1, 1, 0), Complex::new(11.2, -0.46)),
            ((3, 1, 0), Complex::new(11.4, -0.42)),
            ((0, 2, 0), Complex::new(12.1, -0.98)),
            ((1, 2, 0), Complex::new(12.2, -0.96)),
            ((3, 2, 0), Complex::new(12.4, -0.92)),
            ((0, 0, 1), Complex::new(20.1, 0.02)),
            ((1, 0, 1), Complex::new(20.2, 0.04)),
            ((3, 0, 1), Complex::new(20.4, 0.08)),
            ((0, 1, 1), Complex::new(21.1, -0.48)),
            ((1, 1, 1), Complex::new(21.2, -0.46)),
            ((3, 1, 1), Complex::new(21.4, -0.42)),
            ((0, 2, 1), Complex::new(22.1, -0.98)),
            ((1, 2, 1), Complex::new(22.2, -0.96)),
            ((3, 2, 1), Complex::new(22.4, -0.92)),
        ];
        for (index, expected) in expected {
            assert_complex_close(table.radial_functions[index], expected);
        }
        Ok(())
    }

    #[test]
    fn transition_matrix_setup_matches_feff_bcoef_reference() -> Result<(), RixsError> {
        let setup = rixs_transition_matrix_setup(sample_transition_matrix_input())?;

        assert_eq!(setup.initial_kappa, -1);
        assert_eq!(setup.initial_angular_momentum, 0);
        assert_eq!(setup.transition_kappas, [-2, 1, 0, 0, 0, -1, 2, -3]);
        assert_eq!(
            setup.transition_angular_momenta,
            [1, 1, -1, -1, -1, 0, 2, 2]
        );
        assert_eq!(setup.b_matrix_diagonal.dim(), (16, 8, 1));
        assert_complex_close(
            setup.b_matrix_diagonal[(2, 0, 0)],
            Complex::new(-0.062_734_417_449_000_05, -0.002_816_883_591_316_175),
        );
        assert_complex_close(
            setup.b_matrix_diagonal[(6, 6, 0)],
            Complex::new(0.028_021_517_059_414_094, 0.001_228_613_664_916_864_3),
        );
        assert_complex_close(
            setup.b_matrix_diagonal[(6, 7, 0)],
            Complex::new(0.032_783_586_221_432_195, 0.001_621_399_287_040_505),
        );
        assert_complex_close(setup.b_matrix_diagonal[(0, 2, 0)], Complex::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn transition_phase_shifts_match_feff_rixs_signed_l_reference() -> Result<(), RixsError> {
        let phase_shifts = ndarray::Array2::from_shape_fn((3, 5).f(), |(energy, signed_slot)| {
            let fortran_energy = energy as Real + 1.0;
            let signed_l = signed_slot as isize - 2;
            Complex::new(
                0.1 * fortran_energy + 0.03 * signed_l as Real,
                -0.2 * fortran_energy + 0.01 * signed_l as Real,
            )
        });
        let transition_angular_momenta = [0, 1, -1, 2];

        let selected = rixs_transition_phase_shifts(RixsTransitionPhaseShiftInput {
            phase_shifts: phase_shifts.view(),
            signed_l_min: -2,
            transition_angular_momenta: &transition_angular_momenta,
        })?;

        let expected = [
            ((0, 0), Complex::new(0.10, -0.20)),
            ((0, 1), Complex::new(0.07, -0.21)),
            ((0, 2), Complex::new(0.0, 0.0)),
            ((0, 3), Complex::new(0.04, -0.22)),
            ((1, 0), Complex::new(0.20, -0.40)),
            ((1, 1), Complex::new(0.17, -0.41)),
            ((1, 2), Complex::new(0.0, 0.0)),
            ((1, 3), Complex::new(0.14, -0.42)),
            ((2, 0), Complex::new(0.30, -0.60)),
            ((2, 1), Complex::new(0.27, -0.61)),
            ((2, 2), Complex::new(0.0, 0.0)),
            ((2, 3), Complex::new(0.24, -0.62)),
        ];
        for (index, expected) in expected {
            assert_complex_close(selected[index], expected);
        }
        Ok(())
    }

    #[test]
    fn direct_final_transition_amplitudes_match_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.10, 0.20, 0.70]);
        let transition_angular_momenta = [0, 1, -1];
        let final_transition_moments =
            ndarray::Array3::from_shape_fn((3, 3, 2).f(), |(energy, transition, spin)| {
                let fortran_energy = energy as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                let fortran_spin = spin as Real + 1.0;
                Complex::new(
                    0.45 + 0.06 * fortran_energy + 0.04 * fortran_transition + 0.03 * fortran_spin,
                    0.08 * fortran_energy - 0.02 * fortran_transition + 0.01 * fortran_spin,
                )
            });
        let final_phase_shifts =
            ndarray::Array2::from_shape_fn((3, 3).f(), |(energy, transition)| {
                let fortran_energy = energy as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                Complex::new(
                    0.04 * fortran_energy - 0.01 * fortran_transition,
                    0.005 * fortran_energy + 0.002 * fortran_transition,
                )
            });
        let normalization = ndarray::arr1(&[0.81, 1.44, 2.25]);

        let amplitudes = rixs_direct_final_transition_amplitudes(RixsDirectFinalTransitionInput {
            relative_energies: energies.view(),
            final_transition_moments: final_transition_moments.view(),
            final_phase_shifts: final_phase_shifts.view(),
            normalization: normalization.view(),
            transition_angular_momenta: &transition_angular_momenta,
            fermi_level: 0.0,
        })?;

        let expected = [
            ((1, 0, 0), 0.798_334_528_101_581_3),
            ((1, 0, 1), 0.836_586_569_583_987_8),
            ((1, 1, 0), 0.842_490_637_468_567),
            ((1, 1, 1), 0.880_648_613_637_335_2),
            ((2, 0, 0), 1.124_175_680_991_883),
            ((2, 0, 1), 1.172_422_693_875_440_7),
            ((2, 1, 0), 1.175_962_949_506_474_3),
            ((2, 1, 1), 1.224_259_475_847_014_7),
        ];
        for (index, expected) in expected {
            assert_complex_close(amplitudes[index], Complex::new(expected, 0.0));
        }
        for transition in 0..3 {
            for spin in 0..2 {
                assert_complex_close(amplitudes[(0, transition, spin)], Complex::new(0.0, 0.0));
            }
        }
        for energy in 0..3 {
            for spin in 0..2 {
                assert_complex_close(amplitudes[(energy, 2, spin)], Complex::new(0.0, 0.0));
            }
        }
        Ok(())
    }

    #[test]
    fn satellite_convolution_matches_feff_rixs_mbconv_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.30, 0.10, 0.65, 1.45]);
        let xes_energy_ev = ndarray::arr1(&[-12.0, -3.0, 5.0, 18.0, 31.0]);
        let xes_mu = ndarray::arr1(&[0.40, 0.85, 1.10, 0.70, 0.25]);
        let cross_section =
            ndarray::Array3::from_shape_fn((4, 4, 2).f(), |(transfer, incident, channel)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_incident = incident as Real + 1.0;
                let fortran_channel = channel as Real + 1.0;
                0.28 + 0.19 * fortran_transfer
                    + 0.11 * fortran_incident
                    + 0.07 * fortran_channel
                    + 0.021 * fortran_transfer * fortran_incident
                    - 0.009 * fortran_transfer * fortran_channel
                    + 0.004 * fortran_incident * fortran_incident
            });

        let convolved = rixs_satellite_convolution(RixsSatelliteConvolutionInput {
            relative_energies: energies.view(),
            cross_section: cross_section.view(),
            xes_energy_ev: xes_energy_ev.view(),
            xes_mu: xes_mu.view(),
            fermi_level: 0.08,
            hartree_ev: 27.211_396,
        })?;

        let expected = [
            ((0, 0, 0), 0.566_874_081_685_175_1),
            ((1, 0, 0), 0.913_113_040_960_111_5),
            ((3, 0, 0), 1.441_665_749_061_858),
            ((0, 3, 0), 0.904_799_062_008_342_4),
            ((2, 1, 1), 1.536_542_588_867_955_4),
            ((3, 2, 1), 1.960_375_503_540_547_2),
            ((3, 3, 1), 2.213_501_121_696_587),
        ];
        for (index, value) in expected {
            assert_real_close(convolved[index], value);
        }
        Ok(())
    }

    #[test]
    fn initial_transition_amplitudes_match_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[0.10, 0.30, 0.65]);
        let transition_angular_momenta = [0, 1, -1];
        let radial_overlaps =
            ndarray::Array3::from_shape_fn((3, 3, 3).f(), |(incident, transfer, transition)| {
                let fortran_incident = incident as Real + 1.0;
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                0.2 * fortran_transition + 0.03 * fortran_incident - 0.015 * fortran_transfer
            });
        let incident_transition_moments =
            ndarray::Array3::from_shape_fn((3, 3, 2).f(), |(incident, transition, spin)| {
                let fortran_incident = incident as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                let fortran_spin = spin as Real + 1.0;
                Complex::new(
                    0.45 + 0.11 * fortran_incident
                        + 0.07 * fortran_transition
                        + 0.13 * fortran_spin,
                    -0.20 + 0.04 * fortran_incident - 0.03 * fortran_transition
                        + 0.05 * fortran_spin,
                )
            });
        let incident_phase_shifts =
            ndarray::Array2::from_shape_fn((3, 3).f(), |(incident, transition)| {
                let fortran_incident = incident as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                Complex::new(
                    0.08 * fortran_incident - 0.03 * fortran_transition,
                    0.015 * fortran_incident + 0.01 * fortran_transition,
                )
            });
        let mut incident_green = ndarray::Array3::zeros((4, 4, 3).f());
        for incident in 0..3 {
            for angular in 0..4 {
                let fortran_incident = incident as Real + 1.0;
                let fortran_angular = angular as Real + 1.0;
                incident_green[(angular, angular, incident)] = Complex::new(
                    0.02 * fortran_angular + 0.01 * fortran_incident,
                    0.05 * fortran_angular - 0.015 * fortran_incident,
                );
            }
        }
        let normalization = ndarray::arr1(&[0.81, 1.44, 2.25]);

        let amplitudes = rixs_initial_transition_amplitudes(RixsInitialAmplitudeInput {
            relative_energies: energies.view(),
            radial_overlaps: radial_overlaps.view(),
            incident_transition_moments: incident_transition_moments.view(),
            incident_phase_shifts: incident_phase_shifts.view(),
            incident_green: incident_green.view(),
            normalization: normalization.view(),
            transition_angular_momenta: &transition_angular_momenta,
            fermi_level: 0.25,
        })?;

        let expected = [
            ((1, 1, 0, 0), 0.295_484_626_454_576_2),
            ((2, 1, 0, 0), 0.467_619_677_002_705_57),
            ((1, 2, 0, 0), 0.276_213_889_946_669),
            ((2, 2, 0, 0), 0.440_641_618_714_088),
            ((1, 1, 1, 1), 0.624_456_928_816_644_7),
            ((2, 1, 1, 1), 0.927_500_168_465_832_4),
            ((1, 2, 1, 1), 0.602_673_547_578_854_8),
            ((2, 2, 1, 1), 0.897_255_597_754_990_1),
            ((1, 1, 2, 1), 0.652_357_574_820_746_5),
            ((2, 1, 2, 1), 0.968_601_795_678_888_2),
            ((1, 2, 2, 1), 0.629_600_915_233_976_2),
            ((2, 2, 2, 1), 0.937_016_954_515_446_3),
            ((1, 1, 3, 1), 0.680_258_220_824_848_3),
            ((2, 1, 3, 1), 1.009_703_422_891_944_1),
            ((1, 2, 3, 1), 0.656_528_282_889_097_8),
            ((2, 2, 3, 1), 0.976_778_311_275_902_5),
        ];
        for (index, value) in expected {
            assert_complex_close(amplitudes[index], Complex::new(value, 0.0));
        }
        assert_complex_close(amplitudes[(0, 2, 0, 0)], Complex::new(0.0, 0.0));
        assert_complex_close(amplitudes[(2, 2, 1, 0)], Complex::new(0.0, 0.0));
        assert_complex_close(amplitudes[(2, 2, 0, 2)], Complex::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn radial_transition_overlaps_match_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.10, 0.20, 0.70]);
        let transition_angular_momenta = [0, 1, -1];
        let radii = ndarray::Array1::from_iter((1..=6).map(|index| {
            let fortran_index = index as Real;
            (-0.80 + 0.20 * (fortran_index - 1.0)).exp()
        }));
        let potential_difference = ndarray::Array1::from_iter((1..=6).map(|index| {
            let fortran_index = index as Real;
            -0.35 + 0.08 * fortran_index - 0.006 * fortran_index * fortran_index
        }));
        let initial_radial_functions =
            ndarray::Array3::from_shape_fn((6, 3, 3).f(), |(radial, angular, energy)| {
                let fortran_radial = radial as Real + 1.0;
                let fortran_angular = angular as Real + 1.0;
                let fortran_energy = energy as Real + 1.0;
                Complex::new(
                    0.10 * fortran_radial + 0.04 * fortran_angular + 0.03 * fortran_energy,
                    -0.06 + 0.012 * fortran_radial - 0.018 * fortran_angular
                        + 0.02 * fortran_energy,
                )
            });
        let final_radial_functions =
            ndarray::Array3::from_shape_fn((6, 3, 3).f(), |(radial, angular, energy)| {
                let fortran_radial = radial as Real + 1.0;
                let fortran_angular = angular as Real + 1.0;
                let fortran_energy = energy as Real + 1.0;
                Complex::new(
                    -0.04 + 0.07 * fortran_radial + 0.02 * fortran_angular - 0.015 * fortran_energy,
                    0.03 * fortran_radial + 0.014 * fortran_angular + 0.011 * fortran_energy,
                )
            });

        let overlaps = rixs_radial_transition_overlaps(RixsRadialOverlapInput {
            relative_energies: energies.view(),
            radii: radii.view(),
            initial_radial_functions: initial_radial_functions.view(),
            final_radial_functions: final_radial_functions.view(),
            potential_difference: potential_difference.view(),
            transition_angular_momenta: &transition_angular_momenta,
            fermi_level: 0.15,
            log_step: 0.20,
            muffin_tin_radius: radii[3] * 0.07_f64.exp(),
        })?;

        let expected = [
            ((1, 1, 0), -2.601_462_949_696_547_7e-4),
            ((2, 1, 0), 7.338_721_533_247_522e-5),
            ((1, 2, 0), 2.918_759_126_864_182_3e-4),
            ((2, 2, 0), 8.006_809_039_911_933e-4),
            ((1, 1, 1), -1.696_439_189_091_96e-3),
            ((2, 1, 1), -1.446_617_430_910_497_3e-3),
            ((1, 2, 1), -1.039_254_092_834_299_1e-3),
            ((2, 2, 1), -6.141_608_536_501_895e-4),
        ];
        for (index, value) in expected {
            assert_real_close(overlaps[index], value);
        }
        assert_real_close(overlaps[(0, 2, 0)], 0.0);
        assert_real_close(overlaps[(2, 2, 2)], 0.0);
        Ok(())
    }

    #[test]
    fn final_energy_broadening_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.25, 0.05, 0.60, 1.40]);
        let edge_cross_section =
            ndarray::Array3::from_shape_fn((4, 4, 2).f(), |(transfer, incident, channel)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_incident = incident as Real + 1.0;
                let fortran_channel = channel as Real + 1.0;
                0.35 + 0.28 * fortran_transfer
                    + 0.09 * fortran_incident
                    + 0.14 * fortran_channel
                    + 0.018 * fortran_incident * fortran_transfer
                    + 0.012 * fortran_channel * fortran_transfer
            });

        let broadened = rixs_final_energy_broadening(RixsFinalEnergyBroadeningInput {
            relative_energies: energies.view(),
            edge_cross_section: edge_cross_section.view(),
            core_width: 0.16,
            final_width: 0.22,
            edge_amplitude: 1.35,
        })?;

        assert_array_close_tol(
            broadened.as_slice().unwrap(),
            &[
                11.189_673_394_727_524,
                12.930_083_358_089_608,
                13.999_713_402_489_379,
                15.811_407_269_379_34,
                5.124_511_240_562_001,
                5.662_282_122_409_239,
                1.903_791_886_188_359_8,
                2.072_478_073_226_352,
                11.201_650_387_821_998,
                12.841_553_104_937_592,
                28.159_952_371_056_917,
                31.665_533_275_017_246,
                11.562_601_603_306_833,
                12.758_954_924_029_79,
                3.158_782_789_179_022,
                3.438_810_305_469_413_5,
                3.714_542_665_133_305_6,
                4.200_925_064_174_225,
                10.210_887_747_774_724,
                11.391_133_041_834_799,
                39.598_354_439_647_146,
                43.489_512_750_889_92,
                8.687_094_133_414_618,
                9.429_222_486_432_655,
                1.262_787_232_522_739_8,
                1.413_525_175_301_683_8,
                2.521_835_694_562_579_3,
                2.794_927_112_076_852,
                7.844_866_095_361_267,
                8.579_432_124_236_986,
                50.590_493_134_969_485,
                54.743_893_094_269_26,
            ],
            1.0e-6,
        );
        Ok(())
    }

    #[test]
    fn raw_cross_section_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let transition_angular_momenta = [0, 1, -1, 1, 2];
        let transition_amplitudes = ndarray::Array4::from_shape_fn(
            (3, 3, 9, 5).f(),
            |(incident, transfer, angular, transition)| {
                let fortran_incident = incident as Real + 1.0;
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_angular = angular as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                Complex::new(
                    0.10 * fortran_incident
                        + 0.05 * fortran_transfer
                        + 0.02 * fortran_angular
                        + 0.03 * fortran_transition,
                    -0.04 * fortran_incident + 0.01 * fortran_transfer + 0.015 * fortran_angular
                        - 0.02 * fortran_transition,
                )
            },
        );
        let final_green =
            ndarray::Array3::from_shape_fn((9, 9, 3).f(), |(row, column, transfer)| {
                let fortran_row = row as Real + 1.0;
                let fortran_column = column as Real + 1.0;
                let fortran_transfer = transfer as Real + 1.0;
                Complex::new(
                    0.02 * fortran_row - 0.015 * fortran_column + 0.01 * fortran_transfer,
                    0.03 * fortran_row + 0.005 * fortran_column - 0.02 * fortran_transfer,
                )
            });
        let final_phase_shifts =
            ndarray::Array2::from_shape_fn((3, 5).f(), |(transfer, transition)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                Complex::new(
                    0.04 * fortran_transfer - 0.01 * fortran_transition,
                    0.005 * fortran_transfer + 0.002 * fortran_transition,
                )
            });

        let cross_section = rixs_raw_cross_section(RixsRawCrossSectionInput {
            transition_amplitudes: transition_amplitudes.view(),
            final_green: final_green.view(),
            final_phase_shifts: final_phase_shifts.view(),
            transition_angular_momenta: &transition_angular_momenta,
            spin_channel_count: 2,
        })?;

        assert_array_close(
            cross_section.as_slice().unwrap(),
            &[
                0.566_222_712_281_630_4,
                3.130_393_696_653_945,
                1.116_944_683_812_450_2,
                4.907_528_282_323_794,
                1.865_741_213_143_577_8,
                7.117_835_710_992_395,
                0.787_332_183_513_828,
                3.858_055_586_200_104_7,
                1.406_328_454_902_034_5,
                5.777_674_611_281,
                2.220_160_508_268_778_5,
                8.123_917_422_368_548,
                1.045_648_422_395_727_2,
                4.660_376_005_249_884,
                1.731_620_088_328_451_6,
                6.719_217_551_325_36,
                2.609_548_402_271_206,
                9.198_547_170_886_702,
            ],
        );
        Ok(())
    }

    #[test]
    fn incident_amplitude_convolution_matches_feff_rixs_reference() -> Result<(), RixsError> {
        let energies = ndarray::arr1(&[-0.20, 0.10, 0.55, 1.20]);
        let transition_angular_momenta = [0, 1, -1, 2];
        let transition_amplitudes = ndarray::Array4::from_shape_fn(
            (4, 4, 9, 4).f(),
            |(incident, transfer, angular, transition)| {
                let fortran_incident = incident as Real + 1.0;
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_angular = angular as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                Complex::new(
                    0.12 * fortran_incident
                        + 0.07 * fortran_transfer
                        + 0.015 * fortran_angular
                        + 0.04 * fortran_transition,
                    -0.03 * fortran_incident + 0.018 * fortran_transfer + 0.01 * fortran_angular
                        - 0.025 * fortran_transition,
                )
            },
        );
        let final_transition_moments =
            ndarray::Array3::from_shape_fn((4, 4, 2).f(), |(transfer, transition, spin)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                let fortran_spin = spin as Real + 1.0;
                Complex::new(
                    0.45 + 0.06 * fortran_transfer
                        + 0.04 * fortran_transition
                        + 0.03 * fortran_spin,
                    0.08 * fortran_transfer - 0.02 * fortran_transition + 0.01 * fortran_spin,
                )
            });
        let final_phase_shifts =
            ndarray::Array2::from_shape_fn((4, 4).f(), |(transfer, transition)| {
                let fortran_transfer = transfer as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                Complex::new(
                    0.03 * fortran_transfer - 0.012 * fortran_transition,
                    0.004 * fortran_transfer + 0.003 * fortran_transition,
                )
            });
        let final_wave_numbers = ndarray::Array1::from_iter((1..=4).map(|index| {
            let value = index as Real;
            Complex::new(0.35 + 0.17 * value, 0.025 * value)
        }));
        let normalization = ndarray::arr1(&[1.10, 0.90, 1.30, 1.60]);
        let b_matrix_diagonal =
            ndarray::Array3::from_shape_fn((9, 4, 2).f(), |(angular, transition, spin)| {
                let fortran_angular = angular as Real + 1.0;
                let fortran_transition = transition as Real + 1.0;
                let fortran_spin = spin as Real + 1.0;
                Complex::new(
                    0.70 + 0.04 * fortran_angular + 0.03 * fortran_transition + 0.02 * fortran_spin,
                    0.03 * fortran_angular - 0.01 * fortran_transition + 0.005 * fortran_spin,
                )
            });

        let convolved =
            rixs_incident_amplitude_convolution(RixsIncidentAmplitudeConvolutionInput {
                relative_energies: energies.view(),
                transition_amplitudes: transition_amplitudes.view(),
                final_transition_moments: final_transition_moments.view(),
                final_phase_shifts: final_phase_shifts.view(),
                final_wave_numbers: final_wave_numbers.view(),
                normalization: normalization.view(),
                b_matrix_diagonal: b_matrix_diagonal.view(),
                transition_angular_momenta: &transition_angular_momenta,
                fermi_level: 0.0,
                core_width: 0.17,
            })?;

        let expected = [
            (
                (0, 0, 0, 0),
                Complex::new(0.001_626_252_939_151_686, -0.042_781_616_777_522_88),
            ),
            (
                (1, 0, 0, 0),
                Complex::new(-0.023_292_512_891_365_653, -0.064_232_207_420_809_94),
            ),
            (
                (2, 1, 2, 1),
                Complex::new(-0.879_579_658_415_556_5, 0.164_494_182_723_412_5),
            ),
            (
                (3, 2, 3, 1),
                Complex::new(-1.011_818_326_199_234_2, 0.396_752_569_383_931_6),
            ),
            ((0, 3, 4, 2), Complex::new(0.595, 0.017)),
            (
                (1, 1, 4, 3),
                Complex::new(-1.125_498_720_014_404, 0.056_433_640_649_343_04),
            ),
            (
                (3, 3, 8, 3),
                Complex::new(-1.581_153_165_129_665_5, 0.743_233_454_382_036_4),
            ),
        ];
        for (index, value) in expected {
            assert_complex_close_tol(convolved[index], value, 1.0e-10);
        }
        Ok(())
    }

    #[test]
    fn rixs_helpers_reject_invalid_inputs() {
        assert!(matches!(
            integrated_double_lorentz(3.1, 2.7, 0.0, 0.3, 1.2, -0.08, Some(5.0)),
            Err(RixsError::InvalidWidth { name: "gamch", .. })
        ));
        assert!(matches!(
            integrated_double_lorentz(3.1, 2.7, 0.45, 0.3, 1.2, -0.08, Some(Real::NAN)),
            Err(RixsError::NonFiniteReal { name: "omega", .. })
        ));
        assert!(matches!(
            kk_integral(
                Complex::new(0.7, -0.2),
                Complex::new(1.1, 0.3),
                2.0,
                -1.0,
                0.25,
                0.4,
            ),
            Err(RixsError::InvalidInterval { .. })
        ));

        let (x, y, values) = interpolation_fixture();
        assert!(matches!(
            bilinear_interpolate_complex(x.view(), y.view(), values.view(), -0.1, 1.0),
            Err(RixsError::OutOfRange { axis: "x", .. })
        ));
        assert!(matches!(
            bilinear_interpolate_complex(
                ndarray::arr1(&[0.0, 0.0]).view(),
                y.view(),
                values.view(),
                0.0,
                1.0,
            ),
            Err(RixsError::NonIncreasingGrid { axis: "x", .. })
        ));
        assert!(matches!(
            bilinear_interpolate_complex(
                x.view(),
                y.view(),
                ndarray::Array2::zeros((2, 2)).view(),
                0.4,
                1.1,
            ),
            Err(RixsError::MatrixTooSmall { .. })
        ));

        let energies = ndarray::arr1(&[0.0, 1.0]);
        let empty = ndarray::Array3::zeros((2, 2, 0));
        assert!(matches!(
            rixs_wave_numbers(RixsWaveNumberInput {
                relative_energies: ndarray::arr1(&[0.0]).view(),
                incident_reference_energy: Complex::new(0.0, 0.0),
                final_reference_energy: Complex::new(0.0, 0.0),
            }),
            Err(RixsError::InsufficientGrid {
                axis: "relative_energies",
                ..
            })
        ));
        assert!(matches!(
            rixs_wave_numbers(RixsWaveNumberInput {
                relative_energies: energies.view(),
                incident_reference_energy: Complex::new(Real::NAN, 0.0),
                final_reference_energy: Complex::new(0.0, 0.0),
            }),
            Err(RixsError::NonFiniteComplex {
                name: "incident_reference_energy",
                ..
            })
        ));
        assert!(matches!(
            rixs_radial_grid(RixsRadialGridInput {
                point_count: 1,
                log_origin: 0.0,
                log_step: 0.1,
                muffin_tin_radius: 1.0,
            }),
            Err(RixsError::InsufficientGrid {
                axis: "radial_grid",
                ..
            })
        ));
        assert!(matches!(
            rixs_radial_grid(RixsRadialGridInput {
                point_count: 2,
                log_origin: 0.0,
                log_step: 0.1,
                muffin_tin_radius: 2.0,
            }),
            Err(RixsError::RadialGridActivePointCount { .. })
        ));
        assert!(matches!(
            rixs_core_hole_potential_difference(RixsCoreHolePotentialInput {
                incident_screened_core_hole: ndarray::arr1(&[]).view(),
                final_screened_core_hole: None,
            }),
            Err(RixsError::EmptyCoreHolePotentialTable)
        ));
        assert!(matches!(
            rixs_core_hole_potential_difference(RixsCoreHolePotentialInput {
                incident_screened_core_hole: energies.view(),
                final_screened_core_hole: Some(ndarray::arr1(&[1.0]).view()),
            }),
            Err(RixsError::CoreHolePotentialLengthMismatch { .. })
        ));
        let radial_values = ndarray::arr1(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
        let radial_records = [RixsRadialFunctionRecord {
            energy: Complex::new(0.1, 0.0),
            angular_momentum: 0,
            radial_values: radial_values.view(),
        }];
        assert!(matches!(
            rixs_radial_function_table(RixsRadialFunctionTableInput {
                records: &radial_records,
                energy_count: 1,
                angular_count: 2,
                radial_count: 2,
            }),
            Err(RixsError::RadialFunctionRecordCountMismatch { .. })
        ));
        assert!(matches!(
            rixs_radial_function_table(RixsRadialFunctionTableInput {
                records: &radial_records,
                energy_count: 1,
                angular_count: 1,
                radial_count: 3,
            }),
            Err(RixsError::RadialFunctionRecordShape { .. })
        ));
        let bad_angular_records = [RixsRadialFunctionRecord {
            energy: Complex::new(0.1, 0.0),
            angular_momentum: 2,
            radial_values: radial_values.view(),
        }];
        assert!(matches!(
            rixs_radial_function_table(RixsRadialFunctionTableInput {
                records: &bad_angular_records,
                energy_count: 1,
                angular_count: 1,
                radial_count: 2,
            }),
            Err(RixsError::RadialFunctionAngularOutOfRange { .. })
        ));
        assert!(matches!(
            rixs_transition_matrix_setup(RixsTransitionMatrixInput {
                spin_channel_count: 3,
                ..sample_transition_matrix_input()
            }),
            Err(RixsError::InvalidTransitionSpinChannelCount { count: 3 })
        ));
        assert!(matches!(
            rixs_transition_matrix_setup(RixsTransitionMatrixInput {
                hole: 41,
                ..sample_transition_matrix_input()
            }),
            Err(RixsError::TransitionCoreHoleSetup { .. })
        ));
        let mut bad_transition_input = sample_transition_matrix_input();
        bad_transition_input.polarization_tensor[1][1] = Complex::new(Real::NAN, 0.0);
        assert!(matches!(
            rixs_transition_matrix_setup(bad_transition_input),
            Err(RixsError::NonFiniteComplex {
                name: "polarization_tensor",
                ..
            })
        ));
        let phase_shift_table = ndarray::Array2::from_elem((2, 2), Complex::new(0.0, 0.0));
        assert!(matches!(
            rixs_transition_phase_shifts(RixsTransitionPhaseShiftInput {
                phase_shifts: ndarray::Array2::zeros((0, 2)).view(),
                signed_l_min: -1,
                transition_angular_momenta: &[0],
            }),
            Err(RixsError::EmptyTransitionPhaseShiftEnergyTable)
        ));
        assert!(matches!(
            rixs_transition_phase_shifts(RixsTransitionPhaseShiftInput {
                phase_shifts: phase_shift_table.view(),
                signed_l_min: 0,
                transition_angular_momenta: &[2],
            }),
            Err(RixsError::TransitionPhaseShiftAngularOutOfRange { .. })
        ));
        assert!(matches!(
            rixs_transition_phase_shifts(RixsTransitionPhaseShiftInput {
                phase_shifts: phase_shift_table.view(),
                signed_l_min: -1,
                transition_angular_momenta: &[],
            }),
            Err(RixsError::EmptyTransitionPhaseShiftTransitionTable)
        ));
        assert!(matches!(
            rixs_final_spectrum(RixsFinalSpectrumInput {
                relative_energies: energies.view(),
                cross_section: empty.view(),
                incident_window: (0.0, 1.0),
                final_window: (0.0, 1.0),
                incident_edge: 0.0,
                final_edge: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::EmptyChannelTable)
        ));

        let edge_splits = ndarray::arr1(&[0.0]);
        assert!(matches!(
            rixs_sum_edge_contributions(RixsEdgeContributionInput {
                relative_energies: energies.view(),
                edge_splits: edge_splits.view(),
                edge_contributions: ndarray::Array4::zeros((2, 2, 0, 1)).view(),
            }),
            Err(RixsError::EmptyEdgeContributionChannelTable)
        ));
        assert!(matches!(
            rixs_sum_edge_contributions(RixsEdgeContributionInput {
                relative_energies: energies.view(),
                edge_splits: edge_splits.view(),
                edge_contributions: ndarray::Array4::zeros((2, 3, 1, 1)).view(),
            }),
            Err(RixsError::EdgeContributionShape { .. })
        ));

        let edge_broadening_cross_section = ndarray::Array3::zeros((2, 2, 1));
        let edge_broadening_self_energy = ndarray::Array1::from_elem(2, Complex::new(0.0, 0.0));
        assert!(matches!(
            rixs_broaden_edge_contributions(RixsEdgeBroadeningInput {
                relative_energies: energies.view(),
                raw_cross_section: edge_broadening_cross_section.view(),
                self_energy: edge_broadening_self_energy.view(),
                fermi_level: 0.0,
                core_width: 0.1,
                incident_width: 0.1,
                final_width_base: 0.0,
                edge_widths: ndarray::arr1(&[]).view(),
                edge_amplitudes: ndarray::arr1(&[]).view(),
            }),
            Err(RixsError::EmptyEdgeBroadeningEdgeTable)
        ));
        assert!(matches!(
            rixs_broaden_edge_contributions(RixsEdgeBroadeningInput {
                relative_energies: energies.view(),
                raw_cross_section: edge_broadening_cross_section.view(),
                self_energy: edge_broadening_self_energy.view(),
                fermi_level: 0.0,
                core_width: 0.1,
                incident_width: 0.1,
                final_width_base: 0.0,
                edge_widths: ndarray::arr1(&[0.2, 0.3]).view(),
                edge_amplitudes: ndarray::arr1(&[1.0]).view(),
            }),
            Err(RixsError::EdgeBroadeningLengthMismatch { .. })
        ));
        assert!(matches!(
            rixs_post_raw_spectrum(RixsPostRawSpectrumInput {
                relative_energies: energies.view(),
                raw_cross_section: edge_broadening_cross_section.view(),
                self_energy: edge_broadening_self_energy.view(),
                fermi_level: 0.0,
                core_width: 0.1,
                incident_width: 0.1,
                final_width_base: 0.0,
                edge_splits: ndarray::arr1(&[]).view(),
                edge_widths: ndarray::arr1(&[0.2]).view(),
                edge_amplitudes: ndarray::arr1(&[1.0]).view(),
                incident_window: (0.0, 1.0),
                final_window: (0.0, 1.0),
                incident_edge: 0.0,
                final_edge: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::EdgeContributionShape { .. })
        ));
        assert!(matches!(
            rixs_post_raw_spectrum(RixsPostRawSpectrumInput {
                relative_energies: energies.view(),
                raw_cross_section: edge_broadening_cross_section.view(),
                self_energy: edge_broadening_self_energy.view(),
                fermi_level: 0.0,
                core_width: 0.1,
                incident_width: 0.1,
                final_width_base: 0.0,
                edge_splits: ndarray::arr1(&[0.0]).view(),
                edge_widths: ndarray::arr1(&[0.2]).view(),
                edge_amplitudes: ndarray::arr1(&[1.0]).view(),
                incident_window: (0.0, 1.0),
                final_window: (0.0, 1.0),
                incident_edge: 0.0,
                final_edge: 0.0,
                hartree_ev: 0.0,
            }),
            Err(RixsError::InvalidHartreeEv { .. })
        ));

        assert!(matches!(
            rixs_normalize_poles(RixsPoleNormalizationInput {
                pole_energies: ndarray::arr1(&[1.0]).view(),
                pole_amplitudes: ndarray::arr1(&[1.0]).view(),
                pole_widths: ndarray::arr1(&[0.1]).view(),
                fermi_level: 0.0,
            }),
            Err(RixsError::InsufficientPoleRows { .. })
        ));
        assert!(matches!(
            rixs_normalize_poles(RixsPoleNormalizationInput {
                pole_energies: ndarray::arr1(&[1.0, 2.0]).view(),
                pole_amplitudes: ndarray::arr1(&[1.0]).view(),
                pole_widths: ndarray::arr1(&[0.1, 0.2]).view(),
                fermi_level: 0.0,
            }),
            Err(RixsError::PoleLengthMismatch { .. })
        ));

        let self_energy = ndarray::Array1::from_elem(2, Complex::new(0.0, 0.0));
        assert!(matches!(
            rixs_incident_energy_broadening(RixsIncidentEnergyBroadeningInput {
                relative_energies: energies.view(),
                edge_cross_section: ndarray::Array3::zeros((2, 2, 0)).view(),
                self_energy: self_energy.view(),
                fermi_level: 0.0,
                incident_width: 0.1,
            }),
            Err(RixsError::EmptyIncidentBroadeningChannelTable)
        ));
        assert!(matches!(
            rixs_incident_energy_broadening(RixsIncidentEnergyBroadeningInput {
                relative_energies: energies.view(),
                edge_cross_section: ndarray::Array3::zeros((2, 3, 1)).view(),
                self_energy: self_energy.view(),
                fermi_level: 0.0,
                incident_width: 0.1,
            }),
            Err(RixsError::IncidentBroadeningShape { .. })
        ));

        let mpse_self_energy = ndarray::Array1::from_elem(2, Complex::new(0.0, 0.0));
        assert!(matches!(
            rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
                relative_energies: energies.view(),
                mpse_energy_ev: ndarray::arr1(&[0.0]).view(),
                mpse_self_energy_ev: mpse_self_energy.view(),
                fermi_level: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::SelfEnergyGridLengthMismatch { .. })
        ));
        assert!(matches!(
            rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
                relative_energies: energies.view(),
                mpse_energy_ev: ndarray::arr1(&[0.0, 0.0]).view(),
                mpse_self_energy_ev: mpse_self_energy.view(),
                fermi_level: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::NonIncreasingGrid {
                axis: "mpse_energy",
                ..
            })
        ));
        assert!(matches!(
            rixs_prepare_self_energy_grid(RixsSelfEnergyGridInput {
                relative_energies: energies.view(),
                mpse_energy_ev: ndarray::arr1(&[0.0, 1.0]).view(),
                mpse_self_energy_ev: mpse_self_energy.view(),
                fermi_level: 0.0,
                hartree_ev: 0.0,
            }),
            Err(RixsError::InvalidHartreeEv { .. })
        ));

        let xes_energy_ev = ndarray::arr1(&[0.0, 1.0]);
        let xes_mu = ndarray::arr1(&[1.0, 0.5]);
        assert!(matches!(
            rixs_satellite_convolution(RixsSatelliteConvolutionInput {
                relative_energies: energies.view(),
                cross_section: ndarray::Array3::zeros((2, 2, 0)).view(),
                xes_energy_ev: xes_energy_ev.view(),
                xes_mu: xes_mu.view(),
                fermi_level: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::EmptySatelliteConvolutionChannelTable)
        ));
        assert!(matches!(
            rixs_satellite_convolution(RixsSatelliteConvolutionInput {
                relative_energies: energies.view(),
                cross_section: ndarray::Array3::zeros((2, 2, 1)).view(),
                xes_energy_ev: ndarray::arr1(&[0.0]).view(),
                xes_mu: ndarray::arr1(&[1.0]).view(),
                fermi_level: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::InsufficientSatelliteXesGrid { .. })
        ));
        assert!(matches!(
            rixs_satellite_convolution(RixsSatelliteConvolutionInput {
                relative_energies: energies.view(),
                cross_section: ndarray::Array3::zeros((2, 2, 1)).view(),
                xes_energy_ev: ndarray::arr1(&[0.0, 1.0]).view(),
                xes_mu: ndarray::arr1(&[1.0]).view(),
                fermi_level: 0.0,
                hartree_ev: 27.211_396,
            }),
            Err(RixsError::SatelliteXesLengthMismatch { .. })
        ));

        assert!(matches!(
            rixs_final_energy_broadening(RixsFinalEnergyBroadeningInput {
                relative_energies: energies.view(),
                edge_cross_section: ndarray::Array3::zeros((2, 2, 0)).view(),
                core_width: 0.1,
                final_width: 0.2,
                edge_amplitude: 1.0,
            }),
            Err(RixsError::EmptyFinalBroadeningChannelTable)
        ));
        assert!(matches!(
            rixs_final_energy_broadening(RixsFinalEnergyBroadeningInput {
                relative_energies: energies.view(),
                edge_cross_section: ndarray::Array3::zeros((2, 3, 1)).view(),
                core_width: 0.1,
                final_width: 0.2,
                edge_amplitude: 1.0,
            }),
            Err(RixsError::FinalBroadeningShape { .. })
        ));
        assert!(matches!(
            rixs_final_energy_broadening(RixsFinalEnergyBroadeningInput {
                relative_energies: energies.view(),
                edge_cross_section: ndarray::Array3::zeros((2, 2, 1)).view(),
                core_width: 0.0,
                final_width: 0.2,
                edge_amplitude: 1.0,
            }),
            Err(RixsError::InvalidWidth {
                name: "core_width",
                ..
            })
        ));

        let amplitudes = ndarray::Array4::zeros((2, 2, 1, 1));
        let green = ndarray::Array3::zeros((1, 1, 2));
        let phases = ndarray::Array2::zeros((2, 1));
        assert!(matches!(
            rixs_raw_cross_section(RixsRawCrossSectionInput {
                transition_amplitudes: amplitudes.view(),
                final_green: green.view(),
                final_phase_shifts: phases.view(),
                transition_angular_momenta: &[0],
                spin_channel_count: 0,
            }),
            Err(RixsError::InvalidSpinChannelCount { .. })
        ));
        assert!(matches!(
            rixs_raw_cross_section(RixsRawCrossSectionInput {
                transition_amplitudes: amplitudes.view(),
                final_green: green.view(),
                final_phase_shifts: phases.view(),
                transition_angular_momenta: &[],
                spin_channel_count: 1,
            }),
            Err(RixsError::EmptyCrossSectionTransitionTable)
        ));
        assert!(matches!(
            rixs_raw_cross_section(RixsRawCrossSectionInput {
                transition_amplitudes: amplitudes.view(),
                final_green: green.view(),
                final_phase_shifts: phases.view(),
                transition_angular_momenta: &[1],
                spin_channel_count: 1,
            }),
            Err(RixsError::CrossSectionAngularShape { .. })
        ));

        let radial_radii = ndarray::arr1(&[1.0, 1.2, 1.4, 1.6]);
        let radial_potential = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
        let radial_functions = ndarray::Array3::zeros((4, 1, 2));
        assert!(matches!(
            rixs_radial_transition_overlaps(RixsRadialOverlapInput {
                relative_energies: energies.view(),
                radii: radial_radii.view(),
                initial_radial_functions: ndarray::Array3::zeros((4, 0, 2)).view(),
                final_radial_functions: ndarray::Array3::zeros((4, 0, 2)).view(),
                potential_difference: radial_potential.view(),
                transition_angular_momenta: &[],
                fermi_level: 0.0,
                log_step: 0.1,
                muffin_tin_radius: 1.5,
            }),
            Err(RixsError::EmptyRadialOverlapTransitionTable)
        ));
        assert!(matches!(
            rixs_radial_transition_overlaps(RixsRadialOverlapInput {
                relative_energies: energies.view(),
                radii: radial_radii.view(),
                initial_radial_functions: radial_functions.view(),
                final_radial_functions: radial_functions.view(),
                potential_difference: ndarray::arr1(&[0.1, 0.2]).view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
                log_step: 0.1,
                muffin_tin_radius: 1.5,
            }),
            Err(RixsError::RadialOverlapShape { .. })
        ));
        assert!(matches!(
            rixs_radial_transition_overlaps(RixsRadialOverlapInput {
                relative_energies: energies.view(),
                radii: radial_radii.view(),
                initial_radial_functions: radial_functions.view(),
                final_radial_functions: radial_functions.view(),
                potential_difference: radial_potential.view(),
                transition_angular_momenta: &[1],
                fermi_level: 0.0,
                log_step: 0.1,
                muffin_tin_radius: 1.5,
            }),
            Err(RixsError::RadialOverlapAngularShape { .. })
        ));

        let initial_radial = ndarray::Array3::zeros((2, 2, 1));
        let initial_moments = ndarray::Array3::zeros((2, 1, 1));
        assert!(matches!(
            rixs_initial_transition_amplitudes(RixsInitialAmplitudeInput {
                relative_energies: energies.view(),
                radial_overlaps: ndarray::Array3::zeros((2, 2, 0)).view(),
                incident_transition_moments: ndarray::Array3::zeros((2, 0, 1)).view(),
                incident_phase_shifts: ndarray::Array2::zeros((2, 0)).view(),
                incident_green: green.view(),
                normalization: ndarray::arr1(&[1.0, 1.0]).view(),
                transition_angular_momenta: &[],
                fermi_level: 0.0,
            }),
            Err(RixsError::EmptyInitialAmplitudeTransitionTable)
        ));
        assert!(matches!(
            rixs_initial_transition_amplitudes(RixsInitialAmplitudeInput {
                relative_energies: energies.view(),
                radial_overlaps: initial_radial.view(),
                incident_transition_moments: ndarray::Array3::zeros((2, 1, 0)).view(),
                incident_phase_shifts: phases.view(),
                incident_green: green.view(),
                normalization: ndarray::arr1(&[1.0, 1.0]).view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
            }),
            Err(RixsError::EmptyInitialAmplitudeSpinTable)
        ));
        assert!(matches!(
            rixs_initial_transition_amplitudes(RixsInitialAmplitudeInput {
                relative_energies: energies.view(),
                radial_overlaps: initial_radial.view(),
                incident_transition_moments: initial_moments.view(),
                incident_phase_shifts: phases.view(),
                incident_green: green.view(),
                normalization: ndarray::arr1(&[1.0, 1.0]).view(),
                transition_angular_momenta: &[1],
                fermi_level: 0.0,
            }),
            Err(RixsError::InitialAmplitudeAngularShape { .. })
        ));
        assert!(matches!(
            rixs_initial_transition_amplitudes(RixsInitialAmplitudeInput {
                relative_energies: energies.view(),
                radial_overlaps: initial_radial.view(),
                incident_transition_moments: initial_moments.view(),
                incident_phase_shifts: phases.view(),
                incident_green: green.view(),
                normalization: ndarray::arr1(&[1.0, -1.0]).view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
            }),
            Err(RixsError::NegativeNormalization { .. })
        ));

        let direct_final_moments = ndarray::Array3::zeros((2, 1, 1));
        let direct_final_phases = ndarray::Array2::zeros((2, 1));
        let direct_final_normalization = ndarray::arr1(&[1.0, 1.0]);
        assert!(matches!(
            rixs_direct_final_transition_amplitudes(RixsDirectFinalTransitionInput {
                relative_energies: energies.view(),
                final_transition_moments: ndarray::Array3::zeros((2, 0, 1)).view(),
                final_phase_shifts: ndarray::Array2::zeros((2, 0)).view(),
                normalization: direct_final_normalization.view(),
                transition_angular_momenta: &[],
                fermi_level: 0.0,
            }),
            Err(RixsError::EmptyDirectFinalTransitionTransitionTable)
        ));
        assert!(matches!(
            rixs_direct_final_transition_amplitudes(RixsDirectFinalTransitionInput {
                relative_energies: energies.view(),
                final_transition_moments: ndarray::Array3::zeros((2, 1, 0)).view(),
                final_phase_shifts: direct_final_phases.view(),
                normalization: direct_final_normalization.view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
            }),
            Err(RixsError::EmptyDirectFinalTransitionSpinTable)
        ));
        assert!(matches!(
            rixs_direct_final_transition_amplitudes(RixsDirectFinalTransitionInput {
                relative_energies: energies.view(),
                final_transition_moments: direct_final_moments.view(),
                final_phase_shifts: direct_final_phases.view(),
                normalization: ndarray::arr1(&[1.0, -1.0]).view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
            }),
            Err(RixsError::NegativeNormalization { .. })
        ));

        let incident_amplitudes = ndarray::Array4::zeros((2, 2, 1, 1));
        let final_moments = ndarray::Array3::zeros((2, 1, 1));
        let phase_shifts = ndarray::Array2::zeros((2, 1));
        let wave_numbers = ndarray::Array1::from_elem(2, Complex::new(1.0, 0.0));
        let normalization = ndarray::arr1(&[1.0, 1.0]);
        let b_matrix = ndarray::Array3::zeros((1, 1, 1));
        assert!(matches!(
            rixs_incident_amplitude_convolution(RixsIncidentAmplitudeConvolutionInput {
                relative_energies: energies.view(),
                transition_amplitudes: incident_amplitudes.view(),
                final_transition_moments: ndarray::Array3::zeros((2, 1, 0)).view(),
                final_phase_shifts: phase_shifts.view(),
                final_wave_numbers: wave_numbers.view(),
                normalization: normalization.view(),
                b_matrix_diagonal: b_matrix.view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
                core_width: 0.1,
            }),
            Err(RixsError::EmptyIncidentConvolutionSpinTable)
        ));
        assert!(matches!(
            rixs_incident_amplitude_convolution(RixsIncidentAmplitudeConvolutionInput {
                relative_energies: energies.view(),
                transition_amplitudes: incident_amplitudes.view(),
                final_transition_moments: final_moments.view(),
                final_phase_shifts: phase_shifts.view(),
                final_wave_numbers: wave_numbers.view(),
                normalization: ndarray::arr1(&[1.0, -1.0]).view(),
                b_matrix_diagonal: b_matrix.view(),
                transition_angular_momenta: &[0],
                fermi_level: 0.0,
                core_width: 0.1,
            }),
            Err(RixsError::NegativeNormalization { .. })
        ));
    }
}
