use crate::eels_input::EelsInput;
use crate::list_dat::ListDatData;
use crate::{IoError, Result};

use super::types::{SfconvInput, SfconvSo2convTarget, SfconvSo2convTargetKind};

pub fn sfconv_so2conv_targets(
    input: &SfconvInput,
    list: Option<&ListDatData>,
    eels: Option<&EelsInput>,
) -> Result<Vec<SfconvSo2convTarget>> {
    let polarizations = so2conv_polarization_indices(eels)?;
    let mut targets = Vec::new();

    for polarization in polarizations {
        match input.spectrum.ispec.abs() {
            0 => push_so2conv_exafs_targets(&mut targets, input.spectrum.ipr6, list, polarization)?,
            1 => push_so2conv_xanes_targets(&mut targets, polarization)?,
            _ => {
                return Err(IoError::Parse {
                    path: "sfconv.inp".into(),
                    line: 0,
                    message: format!(
                        "SO2CONV ispec must be -1, 0, or 1, got {}",
                        input.spectrum.ispec
                    ),
                });
            }
        }
    }

    Ok(targets)
}

/// Extract `SO2CONV` material header values from a FEFF spectrum header.
///
/// FEFF `SFCONV/so2conv.f90` scans each header line with fixed-width
/// substrings until the dashed table separator is reached. This parser keeps
/// those legacy widths so values from `xmu.dat`, `chi.dat`, `chipNNNN.dat`,
fn push_so2conv_exafs_targets(
    targets: &mut Vec<SfconvSo2convTarget>,
    ipr6: i32,
    list: Option<&ListDatData>,
    polarization: i32,
) -> Result<()> {
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_xmu_file_name(polarization)?,
        kind: SfconvSo2convTargetKind::Xmu,
    });
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_chi_file_name(polarization, false)?,
        kind: SfconvSo2convTargetKind::Chi,
    });

    if ipr6 >= 2 {
        let list = list.ok_or_else(|| IoError::Parse {
            path: "list.dat".into(),
            line: 0,
            message: "SO2CONV target selection requires list.dat when ipr6 >= 2".to_string(),
        })?;
        let kind = if ipr6 >= 3 {
            SfconvSo2convTargetKind::FeffPath
        } else {
            SfconvSo2convTargetKind::Chi
        };
        let prefix = if ipr6 >= 3 { "feff" } else { "chip" };
        targets.extend(list.entries.iter().map(|entry| SfconvSo2convTarget {
            file_name: format!("{prefix}{}.dat", so2conv_path_suffix(entry.path_index)),
            kind,
        }));
    }

    Ok(())
}

fn push_so2conv_xanes_targets(
    targets: &mut Vec<SfconvSo2convTarget>,
    polarization: i32,
) -> Result<()> {
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_xmu_file_name(polarization)?,
        kind: SfconvSo2convTargetKind::Xmu,
    });
    targets.push(SfconvSo2convTarget {
        file_name: so2conv_chi_file_name(polarization, true)?,
        kind: SfconvSo2convTargetKind::Chi,
    });
    Ok(())
}

fn so2conv_polarization_indices(eels: Option<&EelsInput>) -> Result<Vec<i32>> {
    let Some(eels) = eels.filter(|input| input.calculate_elnes) else {
        return Ok(vec![1]);
    };
    let min = eels.polarization.min;
    let step = eels.polarization.step;
    let max = eels.polarization.max;
    if step == 0 {
        return Err(IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: "SO2CONV EELS polarization step must be nonzero".to_string(),
        });
    }
    if (step > 0 && min > max) || (step < 0 && min < max) {
        return Err(IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: format!(
                "SO2CONV EELS polarization range {min}:{step}:{max} does not reach max"
            ),
        });
    }

    let mut values = Vec::new();
    let mut value = min;
    while if step > 0 { value <= max } else { value >= max } {
        values.push(value);
        value = value.checked_add(step).ok_or_else(|| IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: "SO2CONV EELS polarization range overflows i32".to_string(),
        })?;
    }
    Ok(values)
}

fn so2conv_xmu_file_name(polarization: i32) -> Result<String> {
    so2conv_polarized_file_name("xmu", polarization, false)
}

fn so2conv_chi_file_name(polarization: i32, legacy_xanes_width: bool) -> Result<String> {
    so2conv_polarized_file_name("chi", polarization, legacy_xanes_width)
}

fn so2conv_polarized_file_name(
    stem: &'static str,
    polarization: i32,
    legacy_xanes_width: bool,
) -> Result<String> {
    let full = match polarization {
        1 => format!("{stem}.dat"),
        2..=9 => format!("{stem}0{polarization}.dat"),
        10 => format!("{stem}10.dat"),
        _ => {
            return Err(IoError::Parse {
                path: "eels.inp".into(),
                line: 0,
                message: format!("SO2CONV polarization index must be 1..=10, got {polarization}"),
            });
        }
    };

    if legacy_xanes_width && stem == "chi" && full.len() > 7 {
        Ok(full.chars().take(7).collect())
    } else {
        Ok(full)
    }
}

fn so2conv_path_suffix(path_index: usize) -> String {
    let digits = path_index.to_string();
    match digits.len() {
        0 => "0000".to_string(),
        1 => format!("000{digits}"),
        2 => format!("00{digits}"),
        3 => format!("0{digits}"),
        4 => digits,
        _ => digits.chars().take(4).collect(),
    }
}
