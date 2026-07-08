use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use refeff_linalg::LinalgError;
use thiserror::Error;

use crate::angular::AngularError;
use crate::bessel::BesselError;
use crate::fovrg::{FovrgDiracSolverInput, FovrgError};
use crate::grid::GridError;
use crate::interpolation::InterpolationError;
use crate::phase::PhaseError;
use crate::{
    Complex, ComplexArray4, ComplexCube, ComplexMat, ComplexVec, Real, RealCube, RealMat, RealVec,
    Vector3,
};

/// Input for FEFF `point_at_index` density-grid traversal.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityGridInput<'a> {
    /// Grid origin in Bohr, FEFF `grid%origin`.
    pub origin: Vector3,
    /// Grid axes in Bohr with FEFF shape `(xyz, dimension)`.
    pub axes: ArrayView2<'a, Real>,
    /// Number of points along each active axis, FEFF `grid%npts`.
    pub points_per_axis: &'a [usize],
}

/// Input for FEFF `calculate_density` backed by `init_wavefunctions` tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityGridFromTablesInput<'a> {
    /// FEFF-order grid traversal setup.
    pub grid: RhorrpDensityGridInput<'a>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered.
    pub fms_atom_count: Option<usize>,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// All-potential wavefunction handoff tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

/// FEFF-order density-grid point table.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpDensityGridPoints {
    /// Points as `(xyz, point)` in Fortran-order storage, matching FEFF
    /// `points(3, totpts)`.
    pub points: RealMat,
}

/// FEFF-order RHORRP density-grid evaluation in Bohr units.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpDensityGridEvaluation {
    /// Points as `(xyz, point)` in Fortran-order storage, matching FEFF
    /// `points(3, totpts)`.
    pub points: RealMat,
    /// Density values in inverse cubic Bohr, matching the FEFF point order.
    pub density_per_bohr3: RealVec,
}

impl RhorrpDensityGridEvaluation {
    /// Number of evaluated grid points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.density_per_bohr3.len()
    }
}

/// One FEFF density-grid work range for a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RhorrpProcessRange {
    /// Zero-based process rank.
    pub process: usize,
    /// FEFF one-based inclusive first point.
    pub start_1based: usize,
    /// FEFF one-based inclusive last point. Empty ranges have `end < start`.
    pub end_1based: usize,
}

impl RhorrpProcessRange {
    /// Number of points in this range.
    #[must_use]
    pub fn len(self) -> usize {
        self.end_1based
            .checked_sub(self.start_1based)
            .map_or(0, |delta| delta + 1)
    }

    /// Whether this range contains no points.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Input for FEFF `nearest_atom`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpNearestAtomInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
}

/// Input for nearest-atom diagnostics over a FEFF-order point table.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpNearestAtomTableInput<'a> {
    /// Cartesian points in Bohr as `(xyz, point)`.
    pub points: ArrayView2<'a, Real>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
}

/// Input for FEFF `init_inclus` FMS-radius atom counts.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpFmsInclusionInput<'a> {
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Zero-based representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: &'a [usize],
    /// FMS inclusion radius in Bohr, FEFF `rfms2` after unit conversion.
    pub fms_radius: Real,
}

/// Result of FEFF `nearest_atom`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpNearestAtom {
    /// Zero-based atom index for Rust callers.
    pub atom_index: usize,
    /// FEFF one-based atom index.
    pub atom_index_1based: usize,
    /// Potential index associated with the selected atom.
    pub potential_index: usize,
    /// Displacement `point - atom_position`.
    pub displacement: Vector3,
    /// Squared distance to the selected atom.
    pub squared_distance: Real,
}

/// Nearest-atom diagnostics for a density-grid point table.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpNearestAtomTable {
    /// Displacement `point - atom_position` in Bohr as `(point, xyz)`.
    pub displacement_bohr: RealMat,
    /// Zero-based atom index for each point.
    pub atom_indices: Vec<usize>,
    /// FEFF one-based atom index for each point.
    pub atom_indices_1based: Vec<usize>,
    /// Potential index associated with each selected atom.
    pub potential_indices: Vec<usize>,
}

impl RhorrpNearestAtomTable {
    /// Number of point rows in this diagnostic table.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.atom_indices.len()
    }
}

/// Input for FEFF `rhoerrp` radial-grid interpolation location.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpRadialInterpolationInput {
    /// Distance from the selected atom center in Bohr.
    pub radius: Real,
    /// FEFF logarithmic-grid offset `x0`.
    pub x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
}

/// FEFF radial interpolation index and fraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpRadialInterpolationLocation {
    /// FEFF one-based lower radial index to pass into `interpwf`.
    pub index_below_1based: isize,
    /// Fractional distance from the lower radial sample.
    pub fraction: Real,
}

/// Input for the FEFF `rhoerrp` per-energy density prefactor.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpEnergyPrefactorInput {
    /// Complex contour energy in Hartree, FEFF `em(ie)`.
    pub energy_hartree: Complex,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
}

/// Input for FEFF `rhoerrp` final energy-density scaling.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpEnergyDensityInput<'a> {
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Accumulated local plus scattering Green's-function values, FEFF `Ge`.
    pub green_function: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Radius `r` from the nearest atom center in Bohr.
    pub radius: Real,
    /// Radius `r'` from the nearest atom center in Bohr.
    pub prime_radius: Real,
}

/// Input for the FEFF `rhoerrp` point-pair energy-density assembly.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPairEnergyDensityInput<'a> {
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Regular large Dirac component near `r`, `prel(:,:,:,iph)`.
    pub first_regular_large: ArrayView3<'a, Complex>,
    /// Irregular large Dirac component near `r`, `pnel(:,:,:,iph)`.
    pub first_irregular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r`, `qrel(:,:,:,iph)`.
    pub first_regular_small: ArrayView3<'a, Complex>,
    /// Irregular small Dirac component near `r`, `qnel(:,:,:,iph)`.
    pub first_irregular_small: ArrayView3<'a, Complex>,
    /// Regular large Dirac component near `r'`, `prel(:,:,:,iphp)`.
    pub second_regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r'`, `qrel(:,:,:,iphp)`.
    pub second_regular_small: ArrayView3<'a, Complex>,
    /// Phase shifts for the first potential as `(energy, l)`, FEFF `ph2`.
    pub first_phase: ArrayView2<'a, Complex>,
    /// Phase shifts for the second potential as `(energy, l)`, FEFF `ph2`.
    pub second_phase: ArrayView2<'a, Complex>,
    /// FMS scattering matrix slice as `(energy, L, L')`; `None` skips scattering.
    pub scattering_matrix: Option<ArrayView3<'a, Complex>>,
    /// Whether `r` and `r'` are nearest to the same atom and need the local term.
    pub same_atom: bool,
    /// Displacement from the first nearest atom to `r`, FEFF `dv`.
    pub first_displacement: Vector3,
    /// Displacement from the second nearest atom to `r'`, FEFF `dvp`.
    pub second_displacement: Vector3,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
}

/// Input for FEFF `rhorrp` after nearest-atom and FMS-matrix selection.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPairDensityInput<'a> {
    /// Point-pair energy-density assembly input, matching FEFF `rhoerrp`.
    pub pair_energy: RhorrpPairEnergyDensityInput<'a>,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

/// Input for FEFF RHORRP same-point energy-density evaluation from handoff tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointEnergyDensityInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Regular large Dirac component, `prel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_large: ArrayView4<'a, Complex>,
    /// Irregular large Dirac component, `pnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_large: ArrayView4<'a, Complex>,
    /// Regular small Dirac component, `qrel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_small: ArrayView4<'a, Complex>,
    /// Irregular small Dirac component, `qnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_small: ArrayView4<'a, Complex>,
    /// Phase shifts as `(energy, l, potential)`, FEFF `ph2`.
    pub phase: ArrayView3<'a, Complex>,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`,
    /// matching `gg_diag.bin` after promotion to `Complex64`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
}

/// Input for FEFF RHORRP same-point density evaluation from handoff tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointDensityInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Regular large Dirac component, `prel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_large: ArrayView4<'a, Complex>,
    /// Irregular large Dirac component, `pnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_large: ArrayView4<'a, Complex>,
    /// Regular small Dirac component, `qrel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_small: ArrayView4<'a, Complex>,
    /// Irregular small Dirac component, `qnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_small: ArrayView4<'a, Complex>,
    /// Phase shifts as `(energy, l, potential)`, FEFF `ph2`.
    pub phase: ArrayView3<'a, Complex>,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`,
    /// matching `gg_diag.bin` after promotion to `Complex64`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

impl<'a> RhorrpPointDensityInput<'a> {
    /// Drop the contour-integration parameters and keep the FEFF `rhoerrp`
    /// same-point energy-density setup.
    pub fn energy_density_input(self) -> RhorrpPointEnergyDensityInput<'a> {
        RhorrpPointEnergyDensityInput {
            point: self.point,
            atom_positions: self.atom_positions,
            atom_potentials: self.atom_potentials,
            fms_atom_count: self.fms_atom_count,
            energies_hartree: self.energies_hartree,
            reference_energy_hartree: self.reference_energy_hartree,
            regular_large: self.regular_large,
            irregular_large: self.irregular_large,
            regular_small: self.regular_small,
            irregular_small: self.irregular_small,
            phase: self.phase,
            diagonal_scattering_matrices: self.diagonal_scattering_matrices,
            radial_x0: self.radial_x0,
            radial_dx: self.radial_dx,
            radial_count: self.radial_count,
        }
    }
}

/// Input for FEFF RHORRP point-pair energy-density evaluation from handoff tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointPairEnergyDensityInput<'a> {
    /// First Cartesian point in Bohr, FEFF `r`.
    pub first_point: Vector3,
    /// Second Cartesian point in Bohr, FEFF `r'`.
    pub second_point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
    /// Return zero when the first point is outside the central atom Voronoi
    /// cell, matching FEFF `restrict_r_voronoi` for COMPTON.
    pub restrict_first_point_to_central_voronoi: bool,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Regular large Dirac component, `prel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_large: ArrayView4<'a, Complex>,
    /// Irregular large Dirac component, `pnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_large: ArrayView4<'a, Complex>,
    /// Regular small Dirac component, `qrel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_small: ArrayView4<'a, Complex>,
    /// Irregular small Dirac component, `qnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_small: ArrayView4<'a, Complex>,
    /// Phase shifts as `(energy, l, potential)`, FEFF `ph2`.
    pub phase: ArrayView3<'a, Complex>,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`,
    /// matching `gg_diag.bin` after promotion to `Complex64`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// Optional central-row FMS matrices as `(energy, atom, L, L')`, matching
    /// FEFF `gg_slice.bin` blocks for pairs whose first atom is atom 1.
    pub central_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
}

/// Input for selecting FEFF `gg_diag`/`gg_slice` scattering data.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpScatteringMatrixSelectionInput<'a> {
    /// Zero-based atom nearest to `r`, corresponding to FEFF `iat - 1`.
    pub first_atom_index: usize,
    /// Zero-based atom nearest to `r'`, corresponding to FEFF `iatp - 1`.
    pub second_atom_index: usize,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// Optional central-row FMS matrices as `(energy, atom, L, L')`.
    pub central_scattering_matrices: Option<ArrayView4<'a, Complex>>,
}

/// Input for FEFF RHORRP point-pair density-matrix evaluation from handoff tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointPairDensityInput<'a> {
    /// First Cartesian point in Bohr, FEFF `r`.
    pub first_point: Vector3,
    /// Second Cartesian point in Bohr, FEFF `r'`.
    pub second_point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
    /// Return zero when the first point is outside the central atom Voronoi
    /// cell, matching FEFF `restrict_r_voronoi` for COMPTON.
    pub restrict_first_point_to_central_voronoi: bool,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Regular large Dirac component, `prel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_large: ArrayView4<'a, Complex>,
    /// Irregular large Dirac component, `pnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_large: ArrayView4<'a, Complex>,
    /// Regular small Dirac component, `qrel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub regular_small: ArrayView4<'a, Complex>,
    /// Irregular small Dirac component, `qnel(:,:,:,iph)`, as
    /// `(energy, l, radial, potential)`.
    pub irregular_small: ArrayView4<'a, Complex>,
    /// Phase shifts as `(energy, l, potential)`, FEFF `ph2`.
    pub phase: ArrayView3<'a, Complex>,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`,
    /// matching `gg_diag.bin` after promotion to `Complex64`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// Optional central-row FMS matrices as `(energy, atom, L, L')`, matching
    /// FEFF `gg_slice.bin` blocks for pairs whose first atom is atom 1.
    pub central_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

/// FEFF `rhoerrp(v, v, rhoe)` input backed by `init_wavefunctions` tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointEnergyDensityFromTablesInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered.
    pub fms_atom_count: Option<usize>,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// All-potential wavefunction handoff tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
}

/// FEFF `rhorrp(v, v, rho)` input backed by `init_wavefunctions` tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointDensityFromTablesInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered.
    pub fms_atom_count: Option<usize>,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// All-potential wavefunction handoff tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

impl<'a> RhorrpPointDensityFromTablesInput<'a> {
    /// Drop the contour-integration parameters and keep the table-backed
    /// FEFF `rhoerrp` same-point setup.
    pub fn energy_density_input(self) -> RhorrpPointEnergyDensityFromTablesInput<'a> {
        RhorrpPointEnergyDensityFromTablesInput {
            point: self.point,
            atom_positions: self.atom_positions,
            atom_potentials: self.atom_potentials,
            fms_atom_count: self.fms_atom_count,
            energies_hartree: self.energies_hartree,
            reference_energy_hartree: self.reference_energy_hartree,
            wavefunctions: self.wavefunctions,
            diagonal_scattering_matrices: self.diagonal_scattering_matrices,
            radial_x0: self.radial_x0,
            radial_dx: self.radial_dx,
        }
    }
}

/// FEFF `rhoerrp(v, vp, rhoe)` input backed by `init_wavefunctions` tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointPairEnergyDensityFromTablesInput<'a> {
    /// First Cartesian point in Bohr, FEFF `r`.
    pub first_point: Vector3,
    /// Second Cartesian point in Bohr, FEFF `r'`.
    pub second_point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered.
    pub fms_atom_count: Option<usize>,
    /// Return zero when the first point is outside the central atom Voronoi cell.
    pub restrict_first_point_to_central_voronoi: bool,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// All-potential wavefunction handoff tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// Optional central-row FMS matrices as `(energy, atom, L, L')`.
    pub central_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
}

/// FEFF `rhorrp(v, vp, rho)` input backed by `init_wavefunctions` tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPointPairDensityFromTablesInput<'a> {
    /// First Cartesian point in Bohr, FEFF `r`.
    pub first_point: Vector3,
    /// Second Cartesian point in Bohr, FEFF `r'`.
    pub second_point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered.
    pub fms_atom_count: Option<usize>,
    /// Return zero when the first point is outside the central atom Voronoi cell.
    pub restrict_first_point_to_central_voronoi: bool,
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// All-potential wavefunction handoff tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional site-diagonal FMS matrices as `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// Optional central-row FMS matrices as `(energy, atom, L, L')`.
    pub central_scattering_matrices: Option<ArrayView4<'a, Complex>>,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

impl<'a> RhorrpPointPairDensityFromTablesInput<'a> {
    /// Drop the contour-integration parameters and keep the table-backed
    /// FEFF `rhoerrp` point-pair setup.
    pub fn energy_density_input(self) -> RhorrpPointPairEnergyDensityFromTablesInput<'a> {
        RhorrpPointPairEnergyDensityFromTablesInput {
            first_point: self.first_point,
            second_point: self.second_point,
            atom_positions: self.atom_positions,
            atom_potentials: self.atom_potentials,
            fms_atom_count: self.fms_atom_count,
            restrict_first_point_to_central_voronoi: self.restrict_first_point_to_central_voronoi,
            energies_hartree: self.energies_hartree,
            reference_energy_hartree: self.reference_energy_hartree,
            wavefunctions: self.wavefunctions,
            diagonal_scattering_matrices: self.diagonal_scattering_matrices,
            central_scattering_matrices: self.central_scattering_matrices,
            radial_x0: self.radial_x0,
            radial_dx: self.radial_dx,
        }
    }
}

impl<'a> RhorrpPointPairDensityInput<'a> {
    /// Drop the contour-integration parameters and keep the FEFF `rhoerrp`
    /// energy-density setup.
    pub fn energy_density_input(self) -> RhorrpPointPairEnergyDensityInput<'a> {
        RhorrpPointPairEnergyDensityInput {
            first_point: self.first_point,
            second_point: self.second_point,
            atom_positions: self.atom_positions,
            atom_potentials: self.atom_potentials,
            fms_atom_count: self.fms_atom_count,
            restrict_first_point_to_central_voronoi: self.restrict_first_point_to_central_voronoi,
            energies_hartree: self.energies_hartree,
            reference_energy_hartree: self.reference_energy_hartree,
            regular_large: self.regular_large,
            irregular_large: self.irregular_large,
            regular_small: self.regular_small,
            irregular_small: self.irregular_small,
            phase: self.phase,
            diagonal_scattering_matrices: self.diagonal_scattering_matrices,
            central_scattering_matrices: self.central_scattering_matrices,
            radial_x0: self.radial_x0,
            radial_dx: self.radial_dx,
            radial_count: self.radial_count,
        }
    }
}

/// Input for the same-site local Green's-function term in FEFF `rhoerrp`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpSameSiteGreenInput<'a> {
    /// Regular large Dirac component `prel` as `(energy, l, radial)`.
    pub regular_large: ArrayView3<'a, Complex>,
    /// Irregular large Dirac component `pnel` as `(energy, l, radial)`.
    pub irregular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component `qrel` as `(energy, l, radial)`.
    pub regular_small: ArrayView3<'a, Complex>,
    /// Irregular small Dirac component `qnel` as `(energy, l, radial)`.
    pub irregular_small: ArrayView3<'a, Complex>,
    /// Radial interpolation location for `r`.
    pub first_location: RhorrpRadialInterpolationLocation,
    /// Radial interpolation location for `r'`.
    pub second_location: RhorrpRadialInterpolationLocation,
    /// Cosine of the angle between same-site displacement vectors.
    pub cosine_between: Real,
}

/// Input for the scattering Green's-function term in FEFF `rhoerrp`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpScatteringGreenInput<'a> {
    /// Regular large Dirac component near `r`, `prel(:,:,:,iph)`.
    pub first_regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r`, `qrel(:,:,:,iph)`.
    pub first_regular_small: ArrayView3<'a, Complex>,
    /// Regular large Dirac component near `r'`, `prel(:,:,:,iphp)`.
    pub second_regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r'`, `qrel(:,:,:,iphp)`.
    pub second_regular_small: ArrayView3<'a, Complex>,
    /// Phase shifts for the first potential as `(energy, l)`, FEFF `ph2`.
    pub first_phase: ArrayView2<'a, Complex>,
    /// Phase shifts for the second potential as `(energy, l)`, FEFF `ph2`.
    pub second_phase: ArrayView2<'a, Complex>,
    /// FMS scattering matrix slice as `(energy, L, L')`.
    pub scattering_matrix: ArrayView3<'a, Complex>,
    /// Radial interpolation location for `r`.
    pub first_location: RhorrpRadialInterpolationLocation,
    /// Radial interpolation location for `r'`.
    pub second_location: RhorrpRadialInterpolationLocation,
    /// Displacement from the first nearest atom to `r`, FEFF `dv`.
    pub first_displacement: Vector3,
    /// Displacement from the second nearest atom to `r'`, FEFF `dvp`.
    pub second_displacement: Vector3,
}

/// Input for FEFF `interpwf` radial wavefunction interpolation.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionInterpolationInput<'a> {
    /// Wavefunction table as `(energy, angular_momentum, radial)`, matching
    /// FEFF `wf(ne, 0:lx, nr)`.
    pub wavefunctions: ArrayView3<'a, Complex>,
    /// FEFF one-based lower radial index `i`. Negative values return zero and
    /// zero selects the FEFF `wf(:,:,i+1) * f` branch.
    pub index_below_1based: isize,
    /// Fractional distance between the lower and upper radial samples.
    pub fraction: Real,
}

/// Input for FEFF `fermi_dist`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpFermiDistributionInput {
    /// Complex energy in Hartree.
    pub energy_hartree: Complex,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

/// Input for FEFF `fix_irreg` irregular-solution smoothing.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpIrregularFixInput<'a> {
    /// Radial grid `ri`.
    pub radii: &'a [Real],
    /// Irregular solution samples `y0`.
    pub values: ArrayView1<'a, Complex>,
}

/// Input for the FEFF `init_wavefunctions` potential reference shift.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPotentialReferenceShiftInput<'a> {
    /// Muffin-tin radius `rmt(iph)` in Bohr.
    pub muffin_tin_radius: Real,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Total potential after `fixvar`, FEFF `vtotph`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Valence potential after `fixvar`, FEFF `vvalph`.
    pub valence_potential: ArrayView1<'a, Real>,
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
}

/// FEFF `init_wavefunctions` potential tables shifted by `eref0`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpPotentialReferenceShift {
    /// FEFF one-based `jri1` index where `eref0` is sampled.
    pub reference_index_1based: usize,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Adjusted total potential `vtotph`.
    pub total_potential: RealVec,
    /// Adjusted valence potential `vvalph`.
    pub valence_potential: RealVec,
}

/// Input for all FEFF `init_wavefunctions` potential reference shifts.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPotentialReferenceShiftsInput<'a> {
    /// Muffin-tin radii `rmt(iph)` in Bohr.
    pub muffin_tin_radii: &'a [Real],
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Total potential after `fixvar`, FEFF `vtotph(row, iph)`.
    pub total_potential: ArrayView2<'a, Real>,
    /// Valence potential after `fixvar`, FEFF `vvalph(row, iph)`.
    pub valence_potential: ArrayView2<'a, Real>,
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
}

/// FEFF `init_wavefunctions` potential tables shifted by potential-local `eref0`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpPotentialReferenceShifts {
    /// FEFF one-based `jri1` index where each `eref0` is sampled.
    pub reference_indices_1based: Vec<usize>,
    /// Reference potential energies in Hartree, FEFF `eref0(iph)`.
    pub reference_energies_hartree: ComplexVec,
    /// Adjusted total potential `vtotph(row, iph)`.
    pub total_potential: RealMat,
    /// Adjusted valence potential `vvalph(row, iph)`.
    pub valence_potential: RealMat,
}

/// Input for the FEFF `init_wavefunctions` grid-preparation sequence.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionGridPreparationInput<'a> {
    /// Muffin-tin radii `rmt(iph)` in Bohr.
    pub muffin_tin_radii: &'a [Real],
    /// Overlapping charge densities `edens(row, iph)` on the source grid.
    pub electron_density: ArrayView2<'a, Real>,
    /// Total potentials `vtot(row, iph)` on the source grid.
    pub total_potential: ArrayView2<'a, Real>,
    /// Valence densities `edenvl(row, iph)` on the source grid.
    pub valence_density: ArrayView2<'a, Real>,
    /// Valence potentials `vvalgs(row, iph)` on the source grid.
    pub valence_potential: ArrayView2<'a, Real>,
    /// Magnetization densities `dmag(row, iph)` on the source grid.
    pub magnetization: ArrayView2<'a, Real>,
    /// Bound large Dirac components `dgc(row, orbital, iph)`.
    pub bound_large_components: ArrayView3<'a, Real>,
    /// Bound small Dirac components `dpc(row, orbital, iph)`.
    pub bound_small_components: ArrayView3<'a, Real>,
    /// Interstitial potential `vint`.
    pub interstitial_potential: Real,
    /// Interstitial charge density `rhoint`.
    pub interstitial_density: Real,
    /// Source-grid logarithmic spacing, FEFF `dxpot`.
    pub original_radial_dx: Real,
    /// Target-grid logarithmic spacing, FEFF `rgrd`.
    pub target_radial_dx: Real,
    /// FEFF jump mode `jumprm`.
    pub jump_mode: i32,
    /// Initial `vjump` value for jump modes that apply an existing jump.
    pub potential_jump: Real,
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
    /// Target radial table length, usually FEFF `nrptx`.
    pub radial_count: usize,
}

/// FEFF `init_wavefunctions` grids after `fixvar`, `fixdsx`, and `eref0`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpWavefunctionGridPreparation {
    /// Target radial grid `ri`.
    pub radii: RealVec,
    /// Target logarithmic radial spacing, FEFF `dx` after the RGRID pass.
    pub radial_dx: Real,
    /// Per-potential `fixvar` potential jump after the total-potential pass.
    pub potential_jumps: RealVec,
    /// FEFF one-based `jri1` index where each `eref0` was sampled.
    pub reference_indices_1based: Vec<usize>,
    /// Reference energies `eref0(iph)`.
    pub reference_energies_hartree: ComplexVec,
    /// Shifted total potentials as complex FOVRG input `vtotc(row, iph)`.
    pub total_potential: ComplexMat,
    /// Shifted valence potentials as complex FOVRG input `vvalc(row, iph)`.
    pub valence_potential: ComplexMat,
    /// Resampled bound large Dirac components `dgcn(row, orbital, iph)`.
    pub bound_large_components: RealCube,
    /// Resampled bound small Dirac components `dpcn(row, orbital, iph)`.
    pub bound_small_components: RealCube,
    /// Per-orbital active lengths from FEFF `fixdsx`, shaped `(orbital, iph)`.
    pub bound_active_lengths: Array2<usize>,
}

impl RhorrpWavefunctionGridPreparation {
    /// Number of potential blocks represented by this preparation result.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.reference_indices_1based.len()
    }

    /// Number of radial rows in each prepared table.
    #[must_use]
    pub fn radial_count(&self) -> usize {
        self.radii.len()
    }

    /// Number of bound-orbital columns in the resampled Dirac tables.
    #[must_use]
    pub fn orbital_count(&self) -> usize {
        self.bound_large_components.dim().1
    }
}

/// Input for one FEFF `init_wavefunctions` potential using prepared grids.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPreparedPotentialWavefunctionsInput<'a> {
    /// Prepared grids from [`RhorrpWavefunctionGridPreparation`].
    pub prepared: &'a RhorrpWavefunctionGridPreparation,
    /// Zero-based potential index, equivalent to FEFF `iph`.
    pub potential_index: usize,
    /// Complex contour energies `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Muffin-tin radius `rmt(iph)` in Bohr.
    pub muffin_tin_radius: Real,
    /// Norman radius `rnrm(iph)` in Bohr.
    pub norman_radius: Real,
    /// Bound-orbital large origin coefficients `adgc`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `adpc`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// Total bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Valence occupations `xnval`; positive rows are skipped by exchange.
    pub valence_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values.
    pub kappa: ArrayView1<'a, i32>,
    /// Atomic number `iz(iph)`.
    pub atomic_number: Real,
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
    /// Number of ordinary angular momentum channels, equivalent to `lmaxph(iph) + 1`.
    pub angular_momentum_count: usize,
    /// Number of explicitly supplied bound orbitals.
    pub bound_orbital_count: usize,
}

/// Input for all FEFF `init_wavefunctions` potentials using prepared grids.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPreparedWavefunctionTablesInput<'a> {
    /// Prepared grids from [`RhorrpWavefunctionGridPreparation`].
    pub prepared: &'a RhorrpWavefunctionGridPreparation,
    /// Complex contour energies `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Muffin-tin radii `rmt(iph)` in Bohr.
    pub muffin_tin_radii: &'a [Real],
    /// Norman radii `rnrm(iph)` in Bohr.
    pub norman_radii: &'a [Real],
    /// Bound-orbital large origin coefficients `adgc` as `(coefficient, orbital, potential)`.
    pub bound_large_coefficients_by_potential: ArrayView3<'a, Real>,
    /// Bound-orbital small origin coefficients `adpc` as `(coefficient, orbital, potential)`.
    pub bound_small_coefficients_by_potential: ArrayView3<'a, Real>,
    /// Total bound-orbital occupations `xnel` as `(orbital, potential)`.
    ///
    /// FEFF recomputes this row inside `inmuac(..., iph)` for each potential
    /// before `dfovrg` appends the photoelectron channel.
    pub electron_counts_by_potential: ArrayView2<'a, Real>,
    /// Valence occupations `xnval` as `(orbital, potential)`.
    ///
    /// FEFF forwards the potential-local row into `dfovrg`, where positive
    /// rows are skipped by exchange.
    pub valence_counts_by_potential: ArrayView2<'a, Real>,
    /// Bound-orbital relativistic kappa values as `(orbital, potential)`.
    ///
    /// FEFF recomputes this row inside `inmuac(..., iph)`, so heterogeneous
    /// potential tables must not share a single compacted `kap` vector.
    pub kappa_by_potential: ArrayView2<'a, i32>,
    /// Atomic numbers `iz(iph)`.
    pub atomic_numbers: &'a [Real],
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
    /// Number of ordinary angular momentum channels, equivalent to `lx + 1`.
    pub angular_momentum_count: usize,
    /// Number of explicitly supplied bound orbitals per potential, FEFF `norb`.
    pub bound_orbital_counts: &'a [usize],
}

/// Input for FEFF `init_wavefunctions` per-energy radial-solver setup.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionSetupInput {
    /// Complex contour energy `em(ie)` in Hartree.
    pub energy_hartree: Complex,
    /// Reference potential energy `eref0` in Hartree.
    pub reference_energy_hartree: Complex,
    /// Muffin-tin radius `rmt(iph)` in Bohr.
    pub muffin_tin_radius: Real,
    /// Norman radius `rnrm(iph)` in Bohr.
    pub norman_radius: Real,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Maximum radial table length `nrptx`.
    pub radial_capacity: usize,
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
}

/// FEFF `init_wavefunctions` values passed to the radial Dirac solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpWavefunctionSetup {
    /// FEFF `ilast`, the one-based last radial point integrated by `dfovrg`.
    pub last_integration_index_1based: usize,
    /// FEFF `ncycle`, `0` for low exchange models and `3` otherwise.
    pub dirac_cycle_count: usize,
    /// FEFF `p2 = em(ie) - eref0`.
    pub kinetic_energy_hartree: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Muffin-tin wave argument `xkmt = rmt(iph) * ck`.
    pub muffin_tin_wave_number: Complex,
}

/// Input for the FEFF `init_wavefunctions` muffin-tin matching sequence.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpMuffinTinMatchInput {
    /// Muffin-tin radius `rmt(iph)` in Bohr.
    pub muffin_tin_radius: Real,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Angular momentum `lll`.
    pub angular_momentum: usize,
    /// Regular large component at the muffin-tin radius, FEFF `pu`.
    pub regular_large_at_muffin_tin: Complex,
    /// Regular small component at the muffin-tin radius, FEFF `qu`.
    pub regular_small_at_muffin_tin: Complex,
    /// Relativistic kappa `ikap`.
    pub kappa: i32,
}

/// FEFF muffin-tin matching values used by RHORRP radial solutions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpMuffinTinMatch {
    /// FEFF `xkmt = rmt(iph) * ck`.
    pub muffin_tin_wave_number: Complex,
    /// Spherical Bessel value `jl` at `xkmt`.
    pub bessel_j_l: Complex,
    /// Spherical Neumann value `nl` at `xkmt`.
    pub neumann_l: Complex,
    /// Next-order spherical Bessel value `jlp1` at `xkmt`.
    pub bessel_j_l_plus_1: Complex,
    /// Next-order spherical Neumann value `nlp1` at `xkmt`.
    pub neumann_l_plus_1: Complex,
    /// Complex phase shift `phx` returned by FEFF `phamp`.
    pub phase_shift: Complex,
    /// FEFF `temp`, the phase amplitude returned by `phamp`.
    pub phase_amplitude: Complex,
    /// FEFF `xfnorm = 1 / temp`.
    pub regular_solution_scale: Complex,
}

/// Input for FEFF regular-solution normalization in `init_wavefunctions`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpRegularSolutionScaleInput {
    /// FEFF `temp`, the phase-amplitude output from `phamp`.
    pub phase_amplitude: Complex,
}

/// FEFF regular radial-solution multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpRegularSolutionScale {
    /// FEFF `xfnorm = 1 / temp`.
    pub scale: Complex,
}

/// Input for RHORRP irregular-solution initial values before `dfovrg`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpIrregularInitialConditionInput {
    /// Muffin-tin radius `rmt(iph)`.
    pub muffin_tin_radius: Real,
    /// Complex phase shift `phx` from `phamp`.
    pub phase_shift: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Spherical Bessel value `jl`.
    pub bessel_j_l: Complex,
    /// Spherical Neumann value `nl`.
    pub neumann_l: Complex,
    /// Next-order spherical Bessel value `jlp1`.
    pub bessel_j_l_plus_1: Complex,
    /// Next-order spherical Neumann value `nlp1`.
    pub neumann_l_plus_1: Complex,
}

/// FEFF irregular-solution initial values passed into `dfovrg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpIrregularInitialCondition {
    /// FEFF input `pu` for the irregular `dfovrg` call.
    pub large_component: Complex,
    /// FEFF input `qu` for the irregular `dfovrg` call.
    pub small_component: Complex,
}

/// Input for RHORRP irregular-solution Wronskian scaling.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpIrregularWronskianScaleInput {
    /// Complex phase shift `phx` from `phamp`.
    pub phase_shift: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Regular large component at FEFF `jri`, `pr(jri)`.
    pub regular_large_at_match: Complex,
    /// Regular small component at FEFF `jri`, `qr(jri)`.
    pub regular_small_at_match: Complex,
    /// Irregular large component at FEFF `jri`, `pn(jri)`.
    pub irregular_large_at_match: Complex,
    /// Irregular small component at FEFF `jri`, `qn(jri)`.
    pub irregular_small_at_match: Complex,
}

/// FEFF Wronskian scale used when replacing the irregular solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpIrregularWronskianScale {
    /// FEFF `temp = exp(i*phx)`.
    pub phase_factor: Complex,
    /// FEFF denominator before reciprocal wave scaling.
    pub denominator: Complex,
    /// FEFF overwritten `qu = 1 / denominator / ck`.
    pub reciprocal_wave_scale: Complex,
}

/// Input for transforming one irregular-solution row after Wronskian scaling.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpIrregularSolutionTransformInput {
    /// FEFF `temp = exp(i*phx)`.
    pub phase_factor: Complex,
    /// FEFF overwritten `qu = 1 / denominator / ck`.
    pub reciprocal_wave_scale: Complex,
    /// Regular large component `pr(i)`.
    pub regular_large_component: Complex,
    /// Regular small component `qr(i)`.
    pub regular_small_component: Complex,
    /// Raw irregular large component `pn(i)` from `dfovrg`.
    pub irregular_large_component: Complex,
    /// Raw irregular small component `qn(i)` from `dfovrg`.
    pub irregular_small_component: Complex,
}

/// RHORRP transformed irregular-solution row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpIrregularSolutionTransform {
    /// FEFF replacement `pn(i) = i*pr(i) - temp*pn(i)*qu`.
    pub large_component: Complex,
    /// FEFF replacement `qn(i) = i*qr(i) - temp*qn(i)*qu`.
    pub small_component: Complex,
}

/// Input for the RHORRP exact radial continuation after the muffin-tin match.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpExactRadialContinuationInput {
    /// Radial point `ri(i)`.
    pub radius: Real,
    /// Complex phase shift `phx` from `phamp`.
    pub phase_shift: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Spherical Bessel value `jl` at `ck * ri(i)`.
    pub bessel_j_l: Complex,
    /// Spherical Neumann value `nl` at `ck * ri(i)`.
    pub neumann_l: Complex,
    /// Next-order spherical Bessel value `jlp1`.
    pub bessel_j_l_plus_1: Complex,
    /// Next-order spherical Neumann value `nlp1`.
    pub neumann_l_plus_1: Complex,
}

/// RHORRP exact radial values used for samples `jri:nr`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpExactRadialContinuation {
    /// FEFF exact continued regular large component `pr(i)`.
    pub regular_large_component: Complex,
    /// FEFF exact continued regular small component `qr(i)`.
    pub regular_small_component: Complex,
    /// FEFF exact continued irregular large component `pn(i)`.
    pub irregular_large_component: Complex,
    /// FEFF exact continued irregular small component `qn(i)`.
    pub irregular_small_component: Complex,
}

/// Input for the RHORRP exact radial tail overwrite after muffin-tin matching.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpExactRadialTailInput<'a> {
    /// Radial grid `ri`.
    pub radii: &'a [Real],
    /// FEFF one-based first row overwritten by the exact tail, usually `jri`.
    pub start_index_1based: usize,
    /// Angular momentum `lll`.
    pub angular_momentum: usize,
    /// Complex phase shift `phx` from `phamp`.
    pub phase_shift: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
}

/// RHORRP exact free-particle tail rows for `start_index_1based..=nr`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpExactRadialTail {
    /// FEFF one-based first row represented by the returned vectors.
    pub start_index_1based: usize,
    /// FEFF exact continued regular large component `pr(i)`.
    pub regular_large_components: ComplexVec,
    /// FEFF exact continued regular small component `qr(i)`.
    pub regular_small_components: ComplexVec,
    /// FEFF exact continued irregular large component `pn(i)`.
    pub irregular_large_components: ComplexVec,
    /// FEFF exact continued irregular small component `qn(i)`.
    pub irregular_small_components: ComplexVec,
}

impl RhorrpExactRadialTail {
    /// Number of radial rows represented by this tail.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.regular_large_components.len()
    }
}

/// Input for assembling RHORRP radial solution rows after both `dfovrg` passes.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpRadialSolutionAssemblyInput<'a> {
    /// Radial grid `ri`.
    pub radii: &'a [Real],
    /// Raw regular large component from the regular `dfovrg` pass, FEFF `pn`.
    pub raw_regular_large: ArrayView1<'a, Complex>,
    /// Raw regular small component from the regular `dfovrg` pass, FEFF `qn`.
    pub raw_regular_small: ArrayView1<'a, Complex>,
    /// Raw irregular large component from the irregular `dfovrg` pass, FEFF `pn`.
    pub raw_irregular_large: ArrayView1<'a, Complex>,
    /// Raw irregular small component from the irregular `dfovrg` pass, FEFF `qn`.
    pub raw_irregular_small: ArrayView1<'a, Complex>,
    /// Complex phase shift `phx` from `phamp`.
    pub phase_shift: Complex,
    /// FEFF `temp`, the phase amplitude returned by `phamp`.
    pub phase_amplitude: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Angular momentum `lll`.
    pub angular_momentum: usize,
    /// FEFF one-based row used for Wronskian matching, usually `jri`.
    pub match_index_1based: usize,
    /// FEFF one-based first row overwritten by the exact tail, usually `jri`.
    pub exact_tail_start_index_1based: usize,
}

/// RHORRP radial component rows ready for `prel/qrel/pnel/qnel`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpRadialSolutionAssembly {
    /// FEFF `xfnorm = 1 / temp`.
    pub regular_solution_scale: Complex,
    /// Wronskian scale used to transform the irregular solution.
    pub irregular_wronskian_scale: RhorrpIrregularWronskianScale,
    /// Regular large component, FEFF `prel`.
    pub regular_large_components: ComplexVec,
    /// Regular small component, FEFF `qrel`.
    pub regular_small_components: ComplexVec,
    /// Irregular large component, FEFF `pnel`.
    pub irregular_large_components: ComplexVec,
    /// Irregular small component, FEFF `qnel`.
    pub irregular_small_components: ComplexVec,
    /// Whether FEFF `fix_irreg` smoothing was applied to the irregular
    /// `l=0` origin rows.
    pub irregular_origin_smoothed: bool,
}

impl RhorrpRadialSolutionAssembly {
    /// Number of radial rows represented by this assembly.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.regular_large_components.len()
    }
}

/// Input for one RHORRP `init_wavefunctions` energy/angular radial channel.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionChannelInput<'a> {
    /// Base FOVRG solver inputs. RHORRP overwrites the `irregular` flag and
    /// muffin-tin initial values for the regular and irregular passes.
    pub solver: FovrgDiracSolverInput<'a>,
    /// Angular momentum `lll`.
    pub angular_momentum: usize,
    /// Relativistic wave number `ck` from [`RhorrpWavefunctionSetup`].
    pub wave_number: Complex,
}

/// One assembled RHORRP radial channel for a single energy, `l`, and potential.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpWavefunctionChannel {
    /// Muffin-tin phase/amplitude values from the regular radial solve.
    pub muffin_tin_match: RhorrpMuffinTinMatch,
    /// Boundary values used for the irregular radial solve.
    pub irregular_initial_condition: RhorrpIrregularInitialCondition,
    /// Final regular/irregular radial component rows.
    pub radial_solutions: RhorrpRadialSolutionAssembly,
    /// Regular FOVRG active radial length.
    pub regular_active_len: usize,
    /// Irregular FOVRG active radial length.
    pub irregular_active_len: usize,
    /// Regular FOVRG nonlocal-exchange iteration count.
    pub regular_iteration_count: usize,
    /// Irregular FOVRG nonlocal-exchange iteration count.
    pub irregular_iteration_count: usize,
    /// Total difficult Milne iterations reported by both FOVRG passes.
    pub difficult_iterations: usize,
}

/// Input for one FEFF `init_wavefunctions` potential block.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPotentialWavefunctionsInput<'a> {
    /// Base FOVRG solver inputs for this potential. The table builder
    /// overwrites the energy, target kappa, integration endpoint, cycle count,
    /// irregular flag, and muffin-tin boundary values for each channel.
    pub solver: FovrgDiracSolverInput<'a>,
    /// Complex contour energies `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Potential-local reference energy `eref0`.
    pub reference_energy_hartree: Complex,
    /// Norman radius `rnrm(iph)` in Bohr.
    pub norman_radius: Real,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Exchange selector `ixc`.
    pub exchange_index: i32,
    /// Number of ordinary angular momentum channels, equivalent to `lx + 1`.
    pub angular_momentum_count: usize,
}

/// One potential's RHORRP wavefunction tables from FEFF `init_wavefunctions`.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpPotentialWavefunctions {
    /// Per-energy setup values passed to the radial solver.
    pub setups: Vec<RhorrpWavefunctionSetup>,
    /// Relativistic wave number `ck` for each energy.
    pub wave_numbers: ComplexVec,
    /// Phase shifts as `(energy, l)`, FEFF `ph2(:,:,iph)`.
    pub phase_shifts: ComplexMat,
    /// Regular large Dirac component as `(energy, l, radial)`, FEFF `prel`.
    pub regular_large: ComplexCube,
    /// Irregular large Dirac component as `(energy, l, radial)`, FEFF `pnel`.
    pub irregular_large: ComplexCube,
    /// Regular small Dirac component as `(energy, l, radial)`, FEFF `qrel`.
    pub regular_small: ComplexCube,
    /// Irregular small Dirac component as `(energy, l, radial)`, FEFF `qnel`.
    pub irregular_small: ComplexCube,
    /// Total regular FOVRG nonlocal-exchange iterations.
    pub regular_iteration_count: usize,
    /// Total irregular FOVRG nonlocal-exchange iterations.
    pub irregular_iteration_count: usize,
    /// Total difficult Milne iterations reported by all channels.
    pub difficult_iterations: usize,
}

impl RhorrpPotentialWavefunctions {
    /// Number of contour energies represented by this potential block.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.phase_shifts.dim().0
    }

    /// Number of ordinary angular momentum channels represented by this block.
    #[must_use]
    pub fn angular_momentum_count(&self) -> usize {
        self.phase_shifts.dim().1
    }

    /// Number of radial rows represented by the component cubes.
    #[must_use]
    pub fn radial_count(&self) -> usize {
        self.regular_large.dim().2
    }
}

/// Input for all FEFF `init_wavefunctions` potential blocks.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionTablesInput<'a> {
    /// Per-potential wavefunction inputs, one for each FEFF `iph`.
    pub potentials: &'a [RhorrpPotentialWavefunctionsInput<'a>],
}

/// RHORRP wavefunction tables for all potentials.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpWavefunctionTables {
    /// Per-potential, per-energy setup values passed to the radial solver.
    pub setups_by_potential: Vec<Vec<RhorrpWavefunctionSetup>>,
    /// Relativistic wave number `ck` as `(energy, potential)`.
    pub wave_numbers: ComplexMat,
    /// Phase shifts as `(energy, l, potential)`, FEFF `ph2`.
    pub phase_shifts: ComplexCube,
    /// Regular large Dirac component as `(energy, l, radial, potential)`, FEFF `prel`.
    pub regular_large: ComplexArray4,
    /// Irregular large Dirac component as `(energy, l, radial, potential)`, FEFF `pnel`.
    pub irregular_large: ComplexArray4,
    /// Regular small Dirac component as `(energy, l, radial, potential)`, FEFF `qrel`.
    pub regular_small: ComplexArray4,
    /// Irregular small Dirac component as `(energy, l, radial, potential)`, FEFF `qnel`.
    pub irregular_small: ComplexArray4,
    /// Total regular FOVRG nonlocal-exchange iterations.
    pub regular_iteration_count: usize,
    /// Total irregular FOVRG nonlocal-exchange iterations.
    pub irregular_iteration_count: usize,
    /// Total difficult Milne iterations reported by all channels.
    pub difficult_iterations: usize,
}

impl RhorrpWavefunctionTables {
    /// Number of contour energies represented by these tables.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.phase_shifts.dim().0
    }

    /// Number of ordinary angular momentum channels represented by these tables.
    #[must_use]
    pub fn angular_momentum_count(&self) -> usize {
        self.phase_shifts.dim().1
    }

    /// Number of radial rows represented by the component arrays.
    #[must_use]
    pub fn radial_count(&self) -> usize {
        self.regular_large.dim().2
    }

    /// Number of FEFF potential blocks represented by these tables.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.phase_shifts.dim().2
    }
}

/// Input for FEFF `atomic_density`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpAtomicDensityInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// FEFF one-based orbital/core-wavefunction column `il`.
    pub orbital_index_1based: usize,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// FEFF radial grid `ripot`.
    pub radii: &'a [Real],
    /// Large Dirac components `dgc` as `(radial, orbital, potential)`.
    pub large_components: ArrayView3<'a, Real>,
    /// Small Dirac components `dpc` as `(radial, orbital, potential)`.
    pub small_components: ArrayView3<'a, Real>,
}

/// Input for FEFF `rhorrp` contour integration after `rhoerrp`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityIntegrationInput<'a> {
    /// FEFF complex energy contour `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Energy-dependent density matrix values `rhoe`.
    pub energy_density: ArrayView1<'a, Complex>,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

/// Error returned by RHORRP support helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum RhorrpError {
    /// FEFF density grids only support line, plane, and volume commands.
    #[error("RHORRP dimension count must be in 1..=3, got {dimensions}")]
    InvalidDimensionCount { dimensions: usize },
    /// Axis tables must have FEFF shape `(3, dimensions)`.
    #[error("RHORRP axes must have shape (3, {expected_columns}), got ({rows}, {columns})")]
    InvalidAxesShape {
        rows: usize,
        columns: usize,
        expected_columns: usize,
    },
    /// Atom coordinate tables must have shape `(atoms, 3)`.
    #[error("RHORRP atom positions must have shape (atoms, 3), got ({rows}, {columns})")]
    InvalidAtomPositionShape { rows: usize, columns: usize },
    /// Point tables must have shape `(3, points)`.
    #[error("RHORRP point table must have shape (3, points), got ({rows}, {columns})")]
    InvalidPointTableShape { rows: usize, columns: usize },
    /// Point counts must allow FEFF's `(npts - 1)` denominator.
    #[error("RHORRP points_per_axis[{axis}] must be at least 2, got {value}")]
    InvalidPointCount { axis: usize, value: usize },
    /// One-based FEFF indices must stay within their axis bounds.
    #[error("RHORRP index[{axis}]={index} is outside 1..={limit}")]
    InvalidGridIndex {
        axis: usize,
        index: usize,
        limit: usize,
    },
    /// Index vector length must match the active dimension count.
    #[error("RHORRP index length {index_len} does not match dimension count {dimensions}")]
    IndexLengthMismatch { index_len: usize, dimensions: usize },
    /// Atom-potential assignments must match atom coordinates.
    #[error("RHORRP atom potential length {potentials} does not match atom count {atoms}")]
    AtomPotentialLengthMismatch { potentials: usize, atoms: usize },
    /// At least one atom must be available for nearest-atom lookup.
    #[error("RHORRP nearest_atom requires at least one atom")]
    NoAtoms,
    /// The FEFF FMS atom limit must be in the atom table.
    #[error("RHORRP fms_atom_count must be in 1..={atoms}, got {fms_atom_count}")]
    InvalidFmsAtomCount { fms_atom_count: usize, atoms: usize },
    /// FEFF representative atom indices must point into the atom table.
    #[error(
        "RHORRP representative atom for potential {potential} is outside 0..{atoms}, got {representative}"
    )]
    InvalidRepresentativeAtom {
        potential: usize,
        representative: usize,
        atoms: usize,
    },
    /// Floating-point inputs must be finite.
    #[error("RHORRP {name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// Density callbacks must produce finite values.
    #[error("RHORRP density callback returned non-finite value at point {point}: {value}")]
    NonFiniteDensityValue { point: usize, value: Real },
    /// Wavefunction interpolation needs a non-empty `(energy, angular, radial)` table.
    #[error("RHORRP wavefunction table has invalid shape ({energy}, {angular}, {radial})")]
    InvalidWavefunctionShape {
        energy: usize,
        angular: usize,
        radial: usize,
    },
    /// Point-density handoff wavefunctions need non-empty potential tables.
    #[error(
        "RHORRP point-density wavefunction table has invalid shape ({energy}, {angular}, {radial}, {potential})"
    )]
    InvalidPointDensityWavefunctionShape {
        energy: usize,
        angular: usize,
        radial: usize,
        potential: usize,
    },
    /// RHORRP wavefunction component arrays must share the same shape.
    #[error(
        "RHORRP {component} wavefunction shape ({actual_energy}, {actual_angular}, {actual_radial}) does not match ({expected_energy}, {expected_angular}, {expected_radial})"
    )]
    WavefunctionComponentShapeMismatch {
        component: &'static str,
        expected_energy: usize,
        expected_angular: usize,
        expected_radial: usize,
        actual_energy: usize,
        actual_angular: usize,
        actual_radial: usize,
    },
    /// RHORRP point-density handoff wavefunction arrays must share the same shape.
    #[error(
        "RHORRP {component} point-density wavefunction shape ({actual_energy}, {actual_angular}, {actual_radial}, {actual_potential}) does not match ({expected_energy}, {expected_angular}, {expected_radial}, {expected_potential})"
    )]
    PointDensityWavefunctionShapeMismatch {
        component: &'static str,
        expected_energy: usize,
        expected_angular: usize,
        expected_radial: usize,
        expected_potential: usize,
        actual_energy: usize,
        actual_angular: usize,
        actual_radial: usize,
        actual_potential: usize,
    },
    /// RHORRP phase tables must align with wavefunction energy and angular axes.
    #[error(
        "RHORRP {component} phase shape ({actual_energy}, {actual_angular}) does not match ({expected_energy}, {expected_angular})"
    )]
    PhaseShapeMismatch {
        component: &'static str,
        expected_energy: usize,
        expected_angular: usize,
        actual_energy: usize,
        actual_angular: usize,
    },
    /// RHORRP point-density phase tables must align with handoff wavefunctions.
    #[error(
        "RHORRP point-density phase shape ({actual_energy}, {actual_angular}, {actual_potential}) does not match ({expected_energy}, {expected_angular}, {expected_potential})"
    )]
    PointDensityPhaseShapeMismatch {
        expected_energy: usize,
        expected_angular: usize,
        expected_potential: usize,
        actual_energy: usize,
        actual_angular: usize,
        actual_potential: usize,
    },
    /// RHORRP scattering matrices use `(energy, L, L')`.
    #[error(
        "RHORRP scattering matrix shape ({actual_energy}, {actual_rows}, {actual_columns}) does not match ({expected_energy}, {expected_states}, {expected_states})"
    )]
    ScatteringMatrixShapeMismatch {
        expected_energy: usize,
        expected_states: usize,
        actual_energy: usize,
        actual_rows: usize,
        actual_columns: usize,
    },
    /// RHORRP point-density diagonal scattering matrices use `(energy, atom, L, L')`.
    #[error(
        "RHORRP point-density diagonal scattering shape ({actual_energy}, {actual_atoms}, {actual_rows}, {actual_columns}) does not match ({expected_energy}, at least {expected_atoms}, {expected_states}, {expected_states})"
    )]
    PointDensityDiagonalScatteringShapeMismatch {
        expected_energy: usize,
        expected_atoms: usize,
        expected_states: usize,
        actual_energy: usize,
        actual_atoms: usize,
        actual_rows: usize,
        actual_columns: usize,
    },
    /// RHORRP point-pair central scattering matrices use `(energy, atom, L, L')`.
    #[error(
        "RHORRP point-pair central scattering shape ({actual_energy}, {actual_atoms}, {actual_rows}, {actual_columns}) does not match ({expected_energy}, at least {expected_atoms}, {expected_states}, {expected_states})"
    )]
    PointPairCentralScatteringShapeMismatch {
        expected_energy: usize,
        expected_atoms: usize,
        expected_states: usize,
        actual_energy: usize,
        actual_atoms: usize,
        actual_rows: usize,
        actual_columns: usize,
    },
    /// FEFF `gg_diag`/`gg_slice` handoff selection requested an unavailable atom block.
    #[error(
        "RHORRP {matrix} scattering matrix atom {atom_index} is outside available atom count {atom_count}"
    )]
    ScatteringMatrixAtomOutOfRange {
        matrix: &'static str,
        atom_index: usize,
        atom_count: usize,
    },
    /// Atom potential indices must point into the point-density handoff tables.
    #[error(
        "RHORRP point-density atom {atom_index_1based} potential {potential} is outside 0..={max_potential}"
    )]
    InvalidPointDensityPotential {
        atom_index_1based: usize,
        potential: usize,
        max_potential: usize,
    },
    /// Final RHORRP energy-density scaling needs one Green's value per energy.
    #[error("RHORRP energy-density length mismatch: energies={energies}, green={green}")]
    EnergyDensityLengthMismatch { energies: usize, green: usize },
    /// Final RHORRP energy-density scaling divides by positive radii.
    #[error("RHORRP {name} must be positive, got {value}")]
    InvalidPositiveRadius { name: &'static str, value: Real },
    /// FEFF radial interpolation needs at least one radial sample.
    #[error("RHORRP radial_count must be positive, got {radial_count}")]
    InvalidRadialCount { radial_count: usize },
    /// FEFF radial interpolation uses a positive logarithmic-grid spacing.
    #[error("RHORRP radial dx must be positive, got {value}")]
    InvalidRadialStep { value: Real },
    /// FEFF radial interpolation receives radii from vector norms.
    #[error("RHORRP radial radius must be non-negative, got {value}")]
    InvalidRadius { value: Real },
    /// FEFF wavefunction interpolation references both `i` and `i+1`.
    #[error("RHORRP wavefunction index {index} cannot interpolate radial count {radial}")]
    InvalidWavefunctionIndex { index: isize, radial: usize },
    /// FEFF exact radial tail rows must start inside the radial grid.
    #[error("RHORRP exact radial tail start {start_index_1based} is outside 1..={radial_count}")]
    ExactRadialTailStartOutOfRange {
        start_index_1based: usize,
        radial_count: usize,
    },
    /// RHORRP radial solution assembly vectors must share the radial length.
    #[error(
        "RHORRP radial solution {component} length {actual} does not match radial count {expected}"
    )]
    RadialSolutionLengthMismatch {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    /// FEFF Wronskian match row must be inside the radial grid.
    #[error("RHORRP radial solution match row {match_index_1based} is outside 1..={radial_count}")]
    RadialSolutionMatchIndexOutOfRange {
        match_index_1based: usize,
        radial_count: usize,
    },
    /// FEFF target-kappa mapping must fit a signed integer.
    #[error(
        "RHORRP angular momentum {angular_momentum} cannot be converted to photoelectron kappa"
    )]
    PhotoelectronKappaOutOfRange { angular_momentum: usize },
    /// All `init_wavefunctions` channels in a potential block must share the radial length.
    #[error(
        "RHORRP wavefunction channel length mismatch at energy {energy}, l {angular}: expected {expected}, got {actual}"
    )]
    WavefunctionChannelLengthMismatch {
        energy: usize,
        angular: usize,
        expected: usize,
        actual: usize,
    },
    /// Validated non-empty wavefunction input must allocate output tables.
    #[error("RHORRP wavefunction tables were not initialized")]
    UninitializedWavefunctionTables,
    /// The full RHORRP wavefunction table builder needs at least one potential.
    #[error("RHORRP wavefunction table potential count must be positive, got {potential_count}")]
    InvalidWavefunctionPotentialCount { potential_count: usize },
    /// All potential blocks must have the same FEFF wavefunction table dimensions.
    #[error(
        "RHORRP potential {potential} wavefunction shape ({actual_energy}, {actual_angular}, {actual_radial}) does not match ({expected_energy}, {expected_angular}, {expected_radial})"
    )]
    WavefunctionPotentialShapeMismatch {
        potential: usize,
        expected_energy: usize,
        expected_angular: usize,
        expected_radial: usize,
        actual_energy: usize,
        actual_angular: usize,
        actual_radial: usize,
    },
    /// `fix_irreg` requires matching radial and value vectors.
    #[error("RHORRP irregular fix length mismatch: radii={radii}, values={values}")]
    IrregularFixLengthMismatch { radii: usize, values: usize },
    /// `fix_irreg` fits points 50..=100 and replaces 1..=100.
    #[error("RHORRP irregular fix requires at least {required} points, got {points}")]
    InsufficientIrregularFixPoints { points: usize, required: usize },
    /// FEFF `init_wavefunctions` potential arrays must have matching radial lengths.
    #[error("RHORRP potential reference shift length mismatch: total={total}, valence={valence}")]
    PotentialReferenceShiftLengthMismatch { total: usize, valence: usize },
    /// All-potential reference shifting requires at least one potential.
    #[error(
        "RHORRP potential reference shift potential count must be positive, got {potential_count}"
    )]
    InvalidPotentialReferencePotentialCount { potential_count: usize },
    /// All-potential reference shift matrices must share `(radial, potential)` dimensions.
    #[error(
        "RHORRP potential reference shift shape mismatch: total=({total_radial}, {total_potentials}), valence=({valence_radial}, {valence_potentials}), muffin_tin_radii={muffin_tin_radii}"
    )]
    PotentialReferenceShiftShapeMismatch {
        total_radial: usize,
        total_potentials: usize,
        valence_radial: usize,
        valence_potentials: usize,
        muffin_tin_radii: usize,
    },
    /// RHORRP wavefunction grid preparation needs at least one potential.
    #[error(
        "RHORRP wavefunction grid preparation potential count must be positive, got {potential_count}"
    )]
    InvalidWavefunctionGridPotentialCount { potential_count: usize },
    /// FEFF `init_wavefunctions` source matrices must share `(radial, potential)` dimensions.
    #[error(
        "RHORRP wavefunction grid {component} shape ({actual_radial}, {actual_potentials}) does not match ({expected_radial}, {expected_potentials})"
    )]
    WavefunctionGridMatrixShapeMismatch {
        component: &'static str,
        expected_radial: usize,
        expected_potentials: usize,
        actual_radial: usize,
        actual_potentials: usize,
    },
    /// FEFF `fixdsx` source spinor arrays must share their 3-D shape.
    #[error(
        "RHORRP wavefunction grid spinor shape mismatch: large=({large_radial}, {large_orbital}, {large_potential}), small=({small_radial}, {small_orbital}, {small_potential})"
    )]
    WavefunctionGridSpinorShapeMismatch {
        large_radial: usize,
        large_orbital: usize,
        large_potential: usize,
        small_radial: usize,
        small_orbital: usize,
        small_potential: usize,
    },
    /// FEFF `fixdsx` source spinors need non-empty radial/orbital/potential axes.
    #[error(
        "RHORRP wavefunction grid spinor table has invalid shape ({radial}, {orbital}, {potential})"
    )]
    InvalidWavefunctionGridSpinorShape {
        radial: usize,
        orbital: usize,
        potential: usize,
    },
    /// A requested prepared-grid potential is outside the available `iph` range.
    #[error("RHORRP prepared wavefunction potential {potential} is outside 0..{potential_count}")]
    PreparedWavefunctionPotentialOutOfRange {
        potential: usize,
        potential_count: usize,
    },
    /// Prepared-grid per-potential metadata must align with the `iph` axis.
    #[error(
        "RHORRP prepared wavefunction {component} length {actual_potentials} does not match potential count {expected_potentials}"
    )]
    PreparedWavefunctionMetadataLengthMismatch {
        component: &'static str,
        expected_potentials: usize,
        actual_potentials: usize,
    },
    /// Prepared `eref0` metadata must recover FEFF `jri` for FOVRG.
    #[error(
        "RHORRP prepared wavefunction potential {potential} reference index {index_1based} cannot form FEFF jri within radial count {radial_count}"
    )]
    PreparedWavefunctionReferenceIndexOutOfRange {
        potential: usize,
        index_1based: usize,
        radial_count: usize,
    },
    /// FEFF `init_wavefunctions` sampled `eref0` outside the available radial table.
    #[error("RHORRP potential reference index {index_1based} is outside 1..={radial_count}")]
    PotentialReferenceIndexOutOfRange {
        index_1based: usize,
        radial_count: usize,
    },
    /// FEFF `init_wavefunctions` computed an unusable `ilast`.
    #[error("RHORRP wavefunction setup ilast {index_1based} is outside 1..={radial_capacity}")]
    WavefunctionSetupIndexOutOfRange {
        index_1based: isize,
        radial_capacity: usize,
    },
    /// A complex denominator that FEFF divides by is zero.
    #[error("RHORRP complex result {name} is zero")]
    ZeroComplexResult { name: &'static str },
    /// FEFF spherical Bessel/Neumann evaluation failed.
    #[error("RHORRP muffin-tin Bessel evaluation failed: {source}")]
    BesselEvaluation {
        #[from]
        source: BesselError,
    },
    /// FEFF `phamp` phase-amplitude evaluation failed.
    #[error("RHORRP muffin-tin phase-amplitude evaluation failed: {source}")]
    PhaseAmplitude {
        #[from]
        source: PhaseError,
    },
    /// FEFF FOVRG radial solver failed while building RHORRP wavefunctions.
    #[error("RHORRP FOVRG radial solver failed: {source}")]
    FovrgRadialSolver {
        #[from]
        source: FovrgError,
    },
    /// FEFF radial-grid resampling failed while preparing RHORRP wavefunctions.
    #[error("RHORRP wavefunction grid resampling failed: {source}")]
    WavefunctionGridResampling {
        #[from]
        source: GridError,
    },
    /// Polynomial fitting failed while smoothing the irregular solution.
    #[error("RHORRP irregular fix polynomial fit failed: {source}")]
    IrregularFixPolynomial {
        #[from]
        source: LinalgError,
    },
    /// Large and small component tables must have identical dimensions.
    #[error(
        "RHORRP atomic density component shape mismatch: large=({large_radial}, {large_orbital}, {large_potential}), small=({small_radial}, {small_orbital}, {small_potential})"
    )]
    AtomicDensityComponentShapeMismatch {
        large_radial: usize,
        large_orbital: usize,
        large_potential: usize,
        small_radial: usize,
        small_orbital: usize,
        small_potential: usize,
    },
    /// Component tables must have non-empty radial, orbital, and potential axes.
    #[error(
        "RHORRP atomic density {table} table has invalid shape ({radial}, {orbital}, {potential})"
    )]
    InvalidAtomicDensityShape {
        table: &'static str,
        radial: usize,
        orbital: usize,
        potential: usize,
    },
    /// The radial grid must match component-table radial length.
    #[error("RHORRP atomic density radial length mismatch: radii={radii}, components={components}")]
    AtomicDensityRadialLengthMismatch { radii: usize, components: usize },
    /// FEFF `terp` with order 2 needs three radial samples.
    #[error("RHORRP atomic density requires at least {required} radial points, got {points}")]
    InsufficientAtomicDensityRadii { points: usize, required: usize },
    /// FEFF orbital/core-wavefunction columns are one-based.
    #[error("RHORRP atomic density orbital index {orbital} is outside 1..={orbital_count}")]
    InvalidAtomicDensityOrbital {
        orbital: usize,
        orbital_count: usize,
    },
    /// Atom potential indices must point into the component-table potential axis.
    #[error(
        "RHORRP atomic density atom {atom_index_1based} potential {potential} is outside 0..={max_potential}"
    )]
    InvalidAtomicDensityPotential {
        atom_index_1based: usize,
        potential: usize,
        max_potential: usize,
    },
    /// FEFF quadratic radial interpolation failed.
    #[error("RHORRP atomic density interpolation failed: {source}")]
    AtomicDensityInterpolation {
        #[from]
        source: InterpolationError,
    },
    /// Energy and density arrays must have identical lengths.
    #[error(
        "RHORRP density integration length mismatch: energies={energies}, densities={densities}"
    )]
    DensityIntegrationLengthMismatch { energies: usize, densities: usize },
    /// FEFF needs at least two points before Matsubara poles and no more than `ne`.
    #[error(
        "RHORRP density integration real_axis_count {real_axis_count} is outside 2..={energy_count}"
    )]
    InvalidDensityIntegrationRealAxisCount {
        real_axis_count: usize,
        energy_count: usize,
    },
    /// The contour must turn from the vertical leg onto the real axis.
    #[error("RHORRP density integration did not find a horizontal contour segment")]
    MissingDensityIntegrationCorner,
    /// Quadratic interpolation on the real-axis contour needs three points.
    #[error(
        "RHORRP density integration requires at least {required} horizontal points, got {points}"
    )]
    InsufficientDensityIntegrationPoints { points: usize, required: usize },
    /// FEFF complex interpolation failed while integrating the density contour.
    #[error("RHORRP density integration interpolation failed: {source}")]
    DensityIntegrationInterpolation { source: InterpolationError },
    /// FEFF density-grid work partitioning needs at least one process.
    #[error("RHORRP process count must be positive")]
    InvalidProcessCount,
    /// Total point count overflowed `usize`.
    #[error("RHORRP density-grid point count overflows usize")]
    PointCountOverflow,
    /// Spherical harmonic evaluation failed while building the scattering term.
    #[error("RHORRP spherical-harmonic evaluation failed: {source}")]
    SphericalHarmonics {
        #[from]
        source: AngularError,
    },
}
