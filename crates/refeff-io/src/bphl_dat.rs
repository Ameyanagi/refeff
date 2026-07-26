//! FEFF `bphl.dat` broadened-plasmon self-energy table reader.
//!
//! The FEFF distribution contains `EXCH/rhlbp.f90`, but deliberately does not
//! ship the corresponding data. Users of selectors with `index / 10 == 1`
//! provide this author-supplied file at runtime.

use std::path::Path;

use refeff_core::{
    BPHL_RADIUS_COUNT, BPHL_RECORD_COUNT, BPHL_REDUCED_ENERGY_COUNT, BroadenedHedinLundqvistTable,
};

use crate::error::{IoError, Result};

const BPHL_DAT_ROW_WIDTH: usize = 4;
const EXPLICIT_ENERGY_COUNT: usize = BPHL_REDUCED_ENERGY_COUNT - 1;

/// Parse the fixed 21×50 explicit-record layout read by FEFF `rhlbp`.
///
/// The returned core table restores FEFF's implicit zero-valued first
/// reduced-energy column, giving a 21×51 radius-major value grid.
pub fn parse_bphl_dat(text: &str) -> Result<BroadenedHedinLundqvistTable> {
    let mut records = Vec::with_capacity(BPHL_RECORD_COUNT);

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let tokens = raw.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() != BPHL_DAT_ROW_WIDTH {
            return Err(IoError::BphlDatRowWidth {
                line: line_number,
                actual: tokens.len(),
                expected: BPHL_DAT_ROW_WIDTH,
            });
        }
        records.push((
            line_number,
            parse_f64(line_number, "radius", tokens[0])?,
            parse_f64(line_number, "reduced energy", tokens[1])?,
            parse_f64(line_number, "real self energy", tokens[2])?,
            parse_f64(line_number, "imaginary self energy", tokens[3])?,
        ));
    }

    if records.len() != BPHL_RECORD_COUNT {
        return Err(IoError::BphlDatRecordCount {
            actual: records.len(),
            expected: BPHL_RECORD_COUNT,
        });
    }

    let mut radius_mesh = vec![0.0; BPHL_RADIUS_COUNT];
    let mut reduced_energy_mesh = vec![0.0; BPHL_REDUCED_ENERGY_COUNT];
    let value_count = BPHL_RADIUS_COUNT * BPHL_REDUCED_ENERGY_COUNT;
    let mut real = vec![0.0; value_count];
    let mut imaginary = vec![0.0; value_count];

    for (record_index, &(line, radius, reduced_energy, real_value, imaginary_value)) in
        records.iter().enumerate()
    {
        let radius_index = record_index / EXPLICIT_ENERGY_COUNT;
        let energy_index = record_index % EXPLICIT_ENERGY_COUNT + 1;

        if energy_index == 1 {
            radius_mesh[radius_index] = radius;
        } else if radius != radius_mesh[radius_index] {
            return Err(IoError::BphlDatMeshMismatch {
                field: "radius",
                line,
                actual: radius,
                expected: radius_mesh[radius_index],
            });
        }

        if radius_index == 0 {
            reduced_energy_mesh[energy_index] = reduced_energy;
        } else if reduced_energy != reduced_energy_mesh[energy_index] {
            return Err(IoError::BphlDatMeshMismatch {
                field: "reduced energy",
                line,
                actual: reduced_energy,
                expected: reduced_energy_mesh[energy_index],
            });
        }

        let flat_index = radius_index * BPHL_REDUCED_ENERGY_COUNT + energy_index;
        real[flat_index] = real_value;
        imaginary[flat_index] = imaginary_value;
    }

    BroadenedHedinLundqvistTable::new(radius_mesh, reduced_energy_mesh, real, imaginary)
        .map_err(|source| IoError::BphlDatValidation { source })
}

/// Read and parse a FEFF `bphl.dat` file.
pub fn read_bphl_dat(path: impl AsRef<Path>) -> Result<BroadenedHedinLundqvistTable> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_bphl_dat(&text)
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::BphlDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_source_layout_and_restores_implicit_zero_column() -> Result<()> {
        let text = synthetic_bphl_dat();
        let table = parse_bphl_dat(&text)?;

        assert_eq!(table.radius_mesh().len(), BPHL_RADIUS_COUNT);
        assert_eq!(table.reduced_energy_mesh().len(), BPHL_REDUCED_ENERGY_COUNT);
        assert_eq!(table.radius_mesh()[0], 1.0);
        assert_eq!(table.radius_mesh()[20], 21.0);
        assert_eq!(table.reduced_energy_mesh()[0], 0.0);
        assert_eq!(table.reduced_energy_mesh()[1], 1.0);
        assert_eq!(table.reduced_energy_mesh()[50], 50.0);

        for radius_index in 0..BPHL_RADIUS_COUNT {
            let first = radius_index * BPHL_REDUCED_ENERGY_COUNT;
            assert_eq!(table.real_values()[first], 0.0);
            assert_eq!(table.imaginary_values()[first], 0.0);
        }
        assert_eq!(table.real_values()[2 * 51 + 7], 307.0);
        assert_eq!(table.imaginary_values()[2 * 51 + 7], -1037.0);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let text = synthetic_bphl_dat().replacen("1.000000E0", "1.000000D0", 1);
        let table = parse_bphl_dat(&text)?;
        assert_eq!(table.radius_mesh()[0], 1.0);
        Ok(())
    }

    #[test]
    fn rejects_wrong_shape_and_inconsistent_meshes() {
        assert!(matches!(
            parse_bphl_dat("1 2 3\n"),
            Err(IoError::BphlDatRowWidth {
                line: 1,
                actual: 3,
                expected: 4,
            })
        ));
        assert!(matches!(
            parse_bphl_dat("1 2 3 4\n"),
            Err(IoError::BphlDatRecordCount {
                actual: 1,
                expected: BPHL_RECORD_COUNT,
            })
        ));

        let bad_radius =
            synthetic_bphl_dat().replacen("1.000000E0 2.000000E0", "1.500000E0 2.000000E0", 1);
        assert!(matches!(
            parse_bphl_dat(&bad_radius),
            Err(IoError::BphlDatMeshMismatch {
                field: "radius",
                line: 2,
                ..
            })
        ));

        let line_51 = "2.000000E0 1.000000E0 2.010000E2 -1.021000E3";
        let bad_energy = synthetic_bphl_dat().replacen(
            line_51,
            "2.000000E0 1.500000E0 2.010000E2 -1.021000E3",
            1,
        );
        assert!(matches!(
            parse_bphl_dat(&bad_energy),
            Err(IoError::BphlDatMeshMismatch {
                field: "reduced energy",
                line: 51,
                ..
            })
        ));
    }

    fn synthetic_bphl_dat() -> String {
        let mut text = String::new();
        for radius in 1..=BPHL_RADIUS_COUNT {
            for reduced_energy in 1..BPHL_REDUCED_ENERGY_COUNT {
                let radius = radius as f64;
                let reduced_energy = reduced_energy as f64;
                let real = radius * 100.0 + reduced_energy;
                let imaginary = -(1000.0 + radius * 10.0 + reduced_energy);
                text.push_str(&format!(
                    "{radius:.6E} {reduced_energy:.6E} {real:.6E} {imaginary:.6E}\n"
                ));
            }
        }
        text
    }
}
