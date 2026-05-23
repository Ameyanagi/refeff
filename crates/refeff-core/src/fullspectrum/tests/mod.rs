use ndarray::{Array1, array};
use num_complex::Complex64;

use crate::{ElamError, Real};

use super::{
    FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE, FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT,
    FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY, FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY, FEFF_HARTREE_EV,
    FullSpectrumBackground, FullSpectrumBackgroundInput, FullSpectrumBackgroundSegmentInput,
    FullSpectrumDefaultGridEdge, FullSpectrumDrudeInput, FullSpectrumEdgeAssemblyInput,
    FullSpectrumEdgeGridInput, FullSpectrumEdgeSelectionInput, FullSpectrumError,
    FullSpectrumFineStructure, FullSpectrumFineStructureInput,
    FullSpectrumFineStructureSegmentInput, FullSpectrumHamakerInput,
    FullSpectrumKramersKronigInput, FullSpectrumLinearGridInput, FullSpectrumNumberDensityInput,
    FullSpectrumOpticalConstantsInput, FullSpectrumQSumInput,
    FullSpectrumScatteringDielectricInput, FullSpectrumSumRulesInput, FullSpectrumValenceInput,
    full_spectrum_assemble_edge, full_spectrum_background_from_fprime,
    full_spectrum_default_energy_grid, full_spectrum_drude_term, full_spectrum_edge_energy_grid,
    full_spectrum_edges_from_occupations, full_spectrum_effective_electron_count,
    full_spectrum_elam_edge_energies, full_spectrum_fine_structure_from_segments,
    full_spectrum_hamaker_transform, full_spectrum_kramers_kronig,
    full_spectrum_linear_energy_grid, full_spectrum_number_density,
    full_spectrum_optical_constants, full_spectrum_scattering_to_dielectric,
    full_spectrum_sum_rules, full_spectrum_valence_epsilon2,
};

mod assembly;
mod background;
mod edges;
mod fine_structure;
mod grids;
mod support;
mod transforms;
mod valence;
