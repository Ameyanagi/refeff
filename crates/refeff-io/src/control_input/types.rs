use std::path::PathBuf;

use ndarray::{Array2, ShapeBuilder};
use refeff_core::RhorrpDensityGridInput;

use crate::{IoError, Result};

use super::common::FEFF_BOHR_ANGSTROM;
use super::parser::ControlParser;

/// Parsed contents of FEFF `band.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandInput {
    /// Band module run flag.
    pub mband: i32,
    /// Energy mesh definition.
    pub energy_mesh: BandEnergyMesh,
    /// Number of k-path points.
    pub nkp: i32,
    /// K-path selector.
    pub ikpath: i32,
    /// Empty-lattice propagation switch.
    pub freeprop: bool,
}

/// Energy mesh row from `band.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandEnergyMesh {
    pub emin: f64,
    pub emax: f64,
    pub estep: f64,
}

/// Parsed contents of FEFF `density.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityInput {
    /// Requested density grids.
    pub grids: Vec<DensityGrid>,
}

/// One density grid request.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityGrid {
    /// Grid dimensionality and command type.
    pub kind: DensityGridKind,
    /// Output filename requested by FEFF input.
    pub filename: String,
    /// Origin in the file's Angstrom coordinate units.
    pub origin: [f64; 3],
    /// Whether the optional `core` flag is present.
    pub core: bool,
    /// Axis rows in the file's Angstrom coordinate units.
    pub axes: Vec<DensityAxis>,
}

/// RHORRP density grid converted to FEFF atomic units.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityGridBohr {
    /// Grid dimensionality and command type.
    pub kind: DensityGridKind,
    /// Output filename requested by FEFF input.
    pub filename: String,
    /// Origin in Bohr, matching `density_inp_read` after unit conversion.
    pub origin: [f64; 3],
    /// Whether the optional `core` flag is present.
    pub core: bool,
    /// Axis vectors in Bohr as `(xyz, dimension)`.
    pub axes: Array2<f64>,
    /// Number of points along each active axis.
    pub points_per_axis: Vec<usize>,
}

/// Density grid command kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityGridKind {
    Line,
    Plane,
    Volume,
}

/// One axis row from a density grid request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityAxis {
    /// Axis vector in the file's Angstrom coordinate units.
    pub vector: [f64; 3],
    /// Number of points along the axis.
    pub points: usize,
}

/// Parsed contents of FEFF `fullspectrum.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullSpectrumInput {
    /// Full-spectrum module run flag.
    pub m_full_spectrum: i32,
}

/// Parsed contents of FEFF `opcons.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpconsInput {
    /// Whether optical constants should run.
    pub run_opcons: bool,
    /// Whether epsilon output should be printed.
    pub print_eps: bool,
    /// Number densities for potential indices.
    pub number_densities: Vec<f64>,
}

/// Parsed contents of FEFF `reciprocal.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalInput {
    /// FEFF space selector: `1` for real space, `0` for reciprocal space.
    pub ispace: i32,
    /// Reciprocal-space cell block, present only when `ispace == 0`.
    pub cell: Option<ReciprocalCell>,
}

/// Reciprocal-space cell block from `reciprocal.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalCell {
    /// Lattice vectors in Angstrom Cartesian coordinates.
    pub lattice_vectors: [[f64; 3]; 3],
    /// Volume scaling factor.
    pub volume_scale: f64,
    /// Imaginary energy broadening.
    pub imaginary_energy: f64,
    /// Core-hole strength selector.
    pub core_hole_strength: f64,
    /// FEFF lattice name.
    pub lattice_name: String,
    /// Hermann-Mauguin space-group label.
    pub space_group_hm: String,
    /// Numeric space-group identifier.
    pub space_group: i32,
    /// Number of atoms in the unit cell.
    pub atom_count: usize,
    /// Absorber position selector.
    pub absorber: i32,
    /// Core-hole selector.
    pub core_hole: i32,
    /// K-point mesh controls.
    pub k_mesh: ReciprocalKMesh,
    /// Unit-cell atom positions.
    pub positions: Vec<[f64; 3]>,
    /// Potential index for each unit-cell atom.
    pub potentials: Vec<i32>,
    /// Atom labels for each unit-cell atom.
    pub labels: Vec<String>,
    /// `streta`, `strgmax`, and `strrmax` controls.
    pub stretch: [f64; 3],
}

/// K-point mesh controls from reciprocal-space input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalKMesh {
    pub total: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub kind: i32,
    pub use_symmetry: bool,
}

impl BandInput {
    /// Parse a FEFF `band.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_band()
    }
}

impl DensityInput {
    /// Parse a FEFF `density.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_density()
    }

    /// Convert all requested density grids from Angstrom to Bohr.
    pub fn to_bohr_grids(&self) -> Result<Vec<DensityGridBohr>> {
        self.grids.iter().map(DensityGrid::to_bohr_grid).collect()
    }
}

impl FullSpectrumInput {
    /// Parse a FEFF `fullspectrum.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_fullspectrum()
    }
}

impl OpconsInput {
    /// Parse a FEFF `opcons.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_opcons()
    }
}

impl ReciprocalInput {
    /// Parse a FEFF `reciprocal.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ControlParser::new(source.into(), text);
        parser.parse_reciprocal()
    }
}

impl DensityGrid {
    /// Convert this density grid from Angstrom to Bohr like FEFF
    /// `density_inp_read`.
    pub fn to_bohr_grid(&self) -> Result<DensityGridBohr> {
        let dimensions = self.kind.dimensions();
        if self.axes.len() != dimensions {
            return Err(IoError::Parse {
                path: "density.inp".into(),
                line: 0,
                message: format!(
                    "density grid {:?} requires {dimensions} axis row(s), got {}",
                    self.kind,
                    self.axes.len()
                ),
            });
        }

        let mut axes = Array2::zeros((3, dimensions).f());
        let mut points_per_axis = Vec::with_capacity(dimensions);
        for (dimension, axis) in self.axes.iter().enumerate() {
            for coordinate in 0..3 {
                axes[(coordinate, dimension)] = axis.vector[coordinate] / FEFF_BOHR_ANGSTROM;
            }
            points_per_axis.push(axis.points);
        }

        Ok(DensityGridBohr {
            kind: self.kind,
            filename: self.filename.clone(),
            origin: [
                self.origin[0] / FEFF_BOHR_ANGSTROM,
                self.origin[1] / FEFF_BOHR_ANGSTROM,
                self.origin[2] / FEFF_BOHR_ANGSTROM,
            ],
            core: self.core,
            axes,
            points_per_axis,
        })
    }
}

impl DensityGridBohr {
    /// Borrow this grid as a core RHORRP traversal input.
    #[must_use]
    pub fn as_rhorrp_input(&self) -> RhorrpDensityGridInput<'_> {
        RhorrpDensityGridInput {
            origin: self.origin,
            axes: self.axes.view(),
            points_per_axis: &self.points_per_axis,
        }
    }
}

impl DensityGridKind {
    pub(super) fn as_command(self) -> &'static str {
        match self {
            DensityGridKind::Line => "line",
            DensityGridKind::Plane => "plane",
            DensityGridKind::Volume => "volume",
        }
    }

    pub(super) fn dimensions(self) -> usize {
        match self {
            DensityGridKind::Line => 1,
            DensityGridKind::Plane => 2,
            DensityGridKind::Volume => 3,
        }
    }
}
