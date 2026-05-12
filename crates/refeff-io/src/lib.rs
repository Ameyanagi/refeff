#![forbid(unsafe_code)]

//! Input/output compatibility support for the FEFF10 Rust port.
//!
//! This crate owns FEFF text parsing, FEFF-style intermediate writers, and
//! file-format codecs such as Packed ASCII Data (PAD). Numerical modules should
//! depend on these typed structures rather than re-parsing FEFF text ad hoc.

pub mod compton_input;
pub mod control_input;
pub mod crpa_input;
pub mod dmdw_input;
pub mod eels_input;
pub mod error;
pub mod feff_bin;
pub mod ff2x_input;
pub mod fms_bin;
pub mod fms_input;
pub mod format;
pub mod genfmt_input;
pub mod global_input;
pub mod hubbard_input;
pub mod input;
pub mod ldos_input;
pub mod list_dat;
pub mod model;
pub mod mtdp;
pub mod pad;
pub mod paths_input;
pub mod phase_bin;
pub mod pot_bin;
pub mod pot_input;
pub mod pot_output;
pub mod rdinp;
pub mod rixs_input;
pub mod screen_input;
pub mod sfconv_input;
pub mod structure_output;
pub mod xsect_dat;
pub mod xsph_input;

pub use compton_input::{
    ComptonChemicalPotential, ComptonDensityOutputs, ComptonGrid, ComptonInput, ComptonLimits,
    ComptonMomentum, ComptonSwitches, ComptonWindow,
};
pub use control_input::{
    BandEnergyMesh, BandInput, DensityAxis, DensityGrid, DensityGridKind, DensityInput,
    FullSpectrumInput, OpconsInput, ReciprocalCell, ReciprocalInput, ReciprocalKMesh,
};
pub use crpa_input::CrpaInput;
pub use dmdw_input::{DmdwCalculation, DmdwInput, DmdwPath};
pub use eels_input::{EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh};
pub use error::{IoError, Result};
pub use feff_bin::{
    FeffBinData, FeffBinPath, FeffBinPotential, feff_bin_string, parse_feff_bin, read_feff_bin,
    write_feff_bin,
};
pub use ff2x_input::{Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput};
pub use fms_bin::{
    FMS_BIN_DEFAULT_PAD_WIDTH, FmsBinData, fms_bin_string, parse_fms_bin, read_fms_bin,
    write_fms_bin,
};
pub use fms_input::{FmsCluster, FmsControl, FmsDebye, FmsInput};
pub use genfmt_input::{GenfmtControl, GenfmtInput};
pub use global_input::{
    CfAverage, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GlobalQVector,
};
pub use hubbard_input::HubbardInput;
pub use input::{FeffInput, FeffLine, LineKind, SourceLocation};
pub use ldos_input::{LdosControl, LdosFms, LdosInput, LdosMesh};
pub use list_dat::{
    ListDatData, ListDatEntry, list_dat_string, parse_list_dat, read_list_dat, write_list_dat,
};
pub use model::{Atom, FeffDocument, Potential};
pub use mtdp::{MtdpData, mtdp_string, parse_mtdp, read_mtdp, write_mtdp};
pub use paths_input::{PathsControl, PathsCriteria, PathsInput};
pub use phase_bin::{
    PhaseBinData, PhaseBinPotential, PhaseBinScalars, parse_phase_bin, phase_bin_string,
    read_phase_bin, write_phase_bin,
};
pub use pot_bin::{
    PotBinData, PotBinScalars, parse_pot_bin, pot_bin_string, read_pot_bin, write_pot_bin,
};
pub use pot_input::{PotControl, PotInput, PotPotential, PotRun, PotScattering};
pub use pot_output::{
    PotentialDatInput, PotentialDatSetInput, pot_dat_string, potential_dat_filename,
    potential_dat_outputs, write_potential_dat,
};
pub use rixs_input::{RixsBroadening, RixsEnergyWindow, RixsInput, RixsSwitches};
pub use screen_input::ScreenInput;
pub use sfconv_input::{SfconvControl, SfconvInput, SfconvSpectrum, SfconvWindow};
pub use structure_output::{AtomsDat, AtomsDatRow, DimensionsDat, GeomDat, GeomDatRow};
pub use xsect_dat::{
    XsectDatData, XsectDatScalars, parse_xsect_dat, read_xsect_dat, write_xsect_dat,
    xsect_dat_string,
};
pub use xsph_input::{XsphAdvanced, XsphControl, XsphGrid, XsphInput};
