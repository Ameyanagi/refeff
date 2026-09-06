//! Read-only requested-stage planning. Cache and scientific preparation stay in execution.
use crate::{FeffDocument, FeffInput, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
#[derive(Debug, Serialize)]
pub struct PlannedStage {
    pub name: &'static str,
    pub prerequisites: Vec<&'static str>,
    pub available: bool,
}
#[derive(Debug, Serialize)]
pub struct ExecutionPlan {
    pub input: PathBuf,
    pub output: PathBuf,
    pub stages: Vec<PlannedStage>,
    pub cache_policy: &'static str,
    pub note: &'static str,
}
/// Validate and describe the requested major stages without generating artifacts.
pub fn plan(input: &Path, output: &Path) -> Result<ExecutionPlan> {
    crate::check(input)?;
    let parsed = FeffInput::parse_file(input)?;
    let doc = FeffDocument::from_input(&parsed)?;
    let control = doc.control.unwrap_or([1; 6]);
    let mut stages = vec![PlannedStage {
        name: "rdinp",
        prerequisites: vec!["feff.inp"],
        available: true,
    }];
    for (enabled, name, prerequisites) in [
        (control[0] != 0, "pot", vec!["pot.inp", "geom.dat"]),
        (control[1] != 0, "xsph", vec!["xsph.inp", "pot.bin"]),
        (
            control[2] != 0 && doc.fms.is_some(),
            "fms",
            vec!["fms.inp", "phase.bin"],
        ),
        (
            control[3] != 0 && doc.ispec != 5,
            "path",
            vec!["paths.inp", "phase.bin", "geom.dat"],
        ),
        (
            control[4] != 0 && doc.ispec != 5,
            "genfmt",
            vec!["genfmt.inp", "phase.bin", "paths.dat"],
        ),
        (
            control[5] != 0 && doc.ispec != 5,
            "ff2x",
            vec!["ff2x.inp", "feff.bin", "xsect.dat"],
        ),
        (doc.eels.enabled, "eels", vec!["eels.inp", "xmu.dat"]),
        (doc.mdff.imdff == 3, "eelsmdff", vec!["mdff.inp", "xmu.dat"]),
        (doc.rixs.run, "rixs", vec!["rixs.inp", "phase.bin"]),
        (
            doc.compton.do_compton,
            "compton",
            vec!["compton.inp", "pot.bin"],
        ),
        (doc.crpa.enabled, "crpa", vec!["crpa.inp", "pot.bin"]),
        (
            doc.band_input.mband != 0,
            "band",
            vec!["band.inp", "phase.bin"],
        ),
        (
            doc.full_spectrum_input.m_full_spectrum != 0,
            "fullspectrum",
            vec!["fullspectrum.inp"],
        ),
        (doc.ldos.is_some(), "ldos", vec!["ldos.inp", "pot.bin"]),
        (doc.opcons, "opconsat", vec!["opcons.inp"]),
        (doc.sfconv, "sfconv", vec!["sfconv.inp", "chi.dat"]),
    ] {
        if enabled {
            stages.push(PlannedStage {
                name,
                available: crate::ModuleName::parse(name)?.disabled_feature().is_none(),
                prerequisites,
            });
        }
    }
    Ok(ExecutionPlan {
        input: input.into(),
        output: output.into(),
        stages,
        cache_policy: "validated at execution; existing files alone do not prove a cache hit",
        note: "Requested major stages; scientific preparation can add atomic, screening, projection, and optional spectroscopy sub-stages.",
    })
}
