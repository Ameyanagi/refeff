//! Typed readers for small FEFF module-control handoff files.
//!
//! These files mostly carry switches or compact lists emitted by `rdinp`:
//! `band.inp`, `density.inp`, `fullspectrum.inp`, `opcons.inp`, and
//! `reciprocal.inp`.

mod common;
mod parser;
mod render;
mod types;

pub use common::FEFF_BOHR_ANGSTROM;
pub use render::{
    band_input_string, density_input_string, fullspectrum_input_string, opcons_input_string,
    reciprocal_input_string,
};
pub use types::{
    BandEnergyMesh, BandInput, DensityAxis, DensityGrid, DensityGridBohr, DensityGridKind,
    DensityInput, FullSpectrumInput, OpconsInput, ReciprocalCell, ReciprocalInput, ReciprocalKMesh,
};

#[cfg(test)]
mod tests;
