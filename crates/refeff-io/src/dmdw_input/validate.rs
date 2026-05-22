use crate::{IoError, Result};

use super::{DmdwCalculation, DmdwPdosOptions, DmdwSelfEnergyOptions};

pub(super) fn validate_dmdw_calculation(calculation: &DmdwCalculation) -> Result<()> {
    if calculation.path_count != calculation.paths.len() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: format!(
                "DMDW path_count is {} but has {} path row(s)",
                calculation.path_count,
                calculation.paths.len()
            ),
        });
    }
    if calculation.temperature_flag <= 0 {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: format!(
                "DMDW temperature count must be positive, got {}",
                calculation.temperature_flag
            ),
        });
    }
    if !calculation.temperature.is_finite() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW temperature must be finite".to_string(),
        });
    }
    if calculation.temperature_flag == 1 && calculation.temperature_max.is_some() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW single-temperature run must not set an upper temperature".to_string(),
        });
    }
    if calculation.temperature_flag > 1 {
        let Some(temperature_max) = calculation.temperature_max else {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "DMDW multi-temperature run requires an upper temperature".to_string(),
            });
        };
        if !temperature_max.is_finite() {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "DMDW upper temperature must be finite".to_string(),
            });
        }
    }
    validate_dmdw_filename("DMDW dynamical-matrix filename", &calculation.dym_file)?;
    if calculation.calculation_type == 2 {
        validate_self_energy_options(calculation.self_energy_options.as_ref())?;
        if calculation.path_count != 0 || !calculation.paths.is_empty() {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "DMDW run type 2 must not include path rows".to_string(),
            });
        }
    } else if calculation.self_energy_options.is_some() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW self-energy options are only valid for calculation type 2".to_string(),
        });
    }
    if calculation.calculation_type == 5 {
        validate_pdos_options(calculation.pdos_options.as_ref())?;
    } else if calculation.pdos_options.is_some() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW PDOS options are only valid for calculation type 5".to_string(),
        });
    }
    for (index, path) in calculation.paths.iter().enumerate() {
        if !path.max_distance.is_finite() {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: format!("DMDW path {index} maximum distance must be finite"),
            });
        }
    }
    Ok(())
}

fn validate_self_energy_options(options: Option<&DmdwSelfEnergyOptions>) -> Result<()> {
    let Some(options) = options else {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW run type 2 requires self-energy options".to_string(),
        });
    };
    if !options.electron_energy.is_finite() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW self-energy electron energy must be finite".to_string(),
        });
    }
    validate_dmdw_filename("DMDW self-energy PDS filename", &options.pds_file)?;
    validate_dmdw_filename("DMDW self-energy a2f filename", &options.a2f_file)?;
    Ok(())
}

fn validate_dmdw_filename(field: &'static str, value: &str) -> Result<()> {
    if value
        .chars()
        .any(|character| matches!(character, '\n' | '\r'))
    {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: format!("{field} must not contain a line terminator"),
        });
    }
    Ok(())
}

fn validate_pdos_options(options: Option<&DmdwPdosOptions>) -> Result<()> {
    let options = options.cloned().unwrap_or_default();
    if !matches!(options.format, 0 | 1 | 2 | 10) {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: format!(
                "DMDW PDOS format {} is not supported by FEFF",
                options.format
            ),
        });
    }
    if !options.gaussian_broadening_thz.is_finite() || options.gaussian_broadening_thz <= 0.0 {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW PDOS Gaussian broadening must be positive and finite".to_string(),
        });
    }
    if !options.gaussian_resolution_thz.is_finite() || options.gaussian_resolution_thz <= 0.0 {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW PDOS Gaussian resolution must be positive and finite".to_string(),
        });
    }
    Ok(())
}
