use crate::error::{IoError, Result};

use super::validate::invalid_dym;
use super::*;

#[test]
fn parses_type1_dym_and_builds_mass_weighted_matrix() -> Result<()> {
    let parsed = parse_dym(TYPE1_DYM)?;
    assert_eq!(parsed.dym_type, 1);
    assert_eq!(parsed.atom_count(), 2);
    assert_eq!(parsed.atomic_numbers.to_vec(), vec![29, 8]);
    let positions = parsed.coordinates.cartesian_positions();
    assert_eq!(positions[[1, 0]], 1.0);
    assert_eq!(parsed.force_constants[[0, 1, 0, 0]], -1.0);

    let matrix = parsed.mass_weighted_dynamical_matrix()?;
    assert_eq!(matrix.shape(), &[6, 6]);
    assert_eq!(matrix[[0, 0]], 2.0 / 64.0);
    assert!(matrix[[0, 1]] < 0.0);
    Ok(())
}

#[test]
fn roundtrips_type1_dym_text() -> Result<()> {
    let parsed = parse_dym(TYPE1_DYM)?;
    let rendered = dym_string(&parsed)?;
    assert!(rendered.contains("    1.00000000    0.00000000    0.00000000"));
    assert!(rendered.contains("  2.000000E+00  0.000000E+00  0.000000E+00"));
    let reparsed = parse_dym(&rendered)?;
    assert_eq!(reparsed.dym_type, parsed.dym_type);
    assert_eq!(reparsed.atomic_numbers, parsed.atomic_numbers);
    assert_eq!(reparsed.atomic_masses, parsed.atomic_masses);
    assert_eq!(reparsed.force_constants, parsed.force_constants);
    Ok(())
}

#[test]
fn renders_extended_coordinate_fields_when_needed() -> Result<()> {
    let mut parsed = parse_dym(TYPE1_DYM)?;
    let DymCoordinates::Cartesian(positions) = &mut parsed.coordinates else {
        return Err(invalid_dym("coordinates", "expected Cartesian coordinates"));
    };
    positions[[1, 0]] = 1.000000001;

    let rendered = dym_string(&parsed)?;
    assert!(rendered.contains("      1.0000000010    0.0000000000    0.0000000000"));
    assert_eq!(parse_dym(&rendered)?, parsed);
    Ok(())
}

#[test]
fn parses_type4_reduced_coordinates_and_cell() -> Result<()> {
    let parsed = parse_dym(TYPE4_DYM)?;
    assert_eq!(parsed.dym_type, 4);
    let DymCoordinates::Reduced { reduced, cell } = &parsed.coordinates else {
        return Err(invalid_dym("coordinates", "expected reduced coordinates"));
    };
    assert_eq!(reduced[[1, 0]], 0.5);
    assert_eq!(cell[[0, 0]], 4.0);
    let cartesian = parsed.coordinates.cartesian_positions();
    assert_eq!(cartesian[[1, 0]], 2.0);
    assert_eq!(cartesian[[1, 1]], 0.0);
    Ok(())
}

#[test]
fn parses_type2_unique_atom_metadata() -> Result<()> {
    let (_, type1_body) = TYPE1_DYM
        .split_once('\n')
        .ok_or_else(|| invalid_dym("type", "test fixture missing type header"))?;
    let type2_text = String::from("    2\n")
        + type1_body
        + "\
    2    2
   29    1
    1  1.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
    8    1
    2  2.000000E+00  1.000000E+00  0.000000E+00  0.000000E+00
";
    let parsed = parse_dym(&type2_text)?;
    assert_eq!(parsed.dym_type, 2);
    let metadata = parsed
        .type2_metadata
        .as_ref()
        .ok_or_else(|| invalid_dym("type 2 metadata", "missing test metadata"))?;
    assert_eq!(metadata.cell_atom_count, 2);
    assert_eq!(metadata.unique_atoms.len(), 2);
    assert_eq!(metadata.unique_atoms[0].atom_type, 29);
    assert_eq!(
        metadata.unique_atoms[0].center_atom_indices.to_vec(),
        vec![0]
    );
    assert_eq!(metadata.unique_atoms[0].weights.to_vec(), vec![1.0]);
    assert_eq!(metadata.unique_atoms[1].atom_type, 8);
    assert_eq!(
        metadata.unique_atoms[1].center_atom_indices.to_vec(),
        vec![1]
    );
    assert_eq!(metadata.unique_atoms[1].coordinates[[0, 0]], 1.0);

    let rendered = dym_string(&parsed)?;
    let reparsed = parse_dym(&rendered)?;
    assert_eq!(reparsed, parsed);
    Ok(())
}

#[test]
fn parses_type3_dipole_derivatives_for_ir_runs() -> Result<()> {
    let parsed = parse_dym(TYPE3_DYM)?;
    assert_eq!(parsed.dym_type, 3);
    let dipoles = parsed
        .dipole_derivatives
        .as_ref()
        .ok_or_else(|| invalid_dym("dipole derivatives", "missing test dipoles"))?;
    assert_eq!(dipoles.shape(), &[2, 3, 3]);
    assert_eq!(dipoles[[0, 1, 1]], 0.5);
    assert_eq!(dipoles[[1, 2, 2]], 1.8);

    let rendered = dym_string(&parsed)?;
    let reparsed = parse_dym(&rendered)?;
    assert_eq!(reparsed, parsed);
    Ok(())
}

#[test]
fn fills_missing_atomic_metadata_like_feff() -> Result<()> {
    let parsed = parse_dym(TYPE1_MISSING_ATOMIC_METADATA_DYM)?;
    assert_eq!(parsed.atomic_numbers.to_vec(), vec![29, 8]);
    assert!((parsed.atomic_masses[1] - 15.999).abs() < 1.0e-6);
    Ok(())
}

#[test]
fn rejects_bad_dym_inputs() -> Result<()> {
    assert!(matches!(
        parse_dym("1\n"),
        Err(IoError::DymMissing {
            field: "atom count"
        })
    ));
    assert!(matches!(
        parse_dym(TYPE1_BAD_PAIR_DYM),
        Err(IoError::InvalidDym {
            field: "force-constant i atom",
            ..
        })
    ));
    let mut bad_mass = parse_dym(TYPE1_DYM)?;
    bad_mass.atomic_masses[0] = 0.0;
    assert!(matches!(
        dym_string(&bad_mass),
        Err(IoError::InvalidDym {
            field: "atomic mass",
            ..
        })
    ));
    Ok(())
}

const TYPE1_DYM: &str = "\
    1
    2
   29
    8
   64.000000
   16.000000
    0.00000000    0.00000000    0.00000000
    1.00000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
";

const TYPE4_DYM: &str = "\
    4
    2
   29
    8
   64.000000
   16.000000
    0.00000000    0.00000000    0.00000000
    0.50000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00

    4.00000000    0.00000000    0.00000000
    0.00000000    5.00000000    0.00000000
    0.00000000    0.00000000    6.00000000
";

const TYPE3_DYM: &str = "\
    3
    2
   29
    8
   64.000000
   16.000000
    0.00000000    0.00000000    0.00000000
    1.00000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00

  1.000000E-01  2.000000E-01  3.000000E-01
  4.000000E-01  5.000000E-01  6.000000E-01
  7.000000E-01  8.000000E-01  9.000000E-01
  1.000000E+00  1.100000E+00  1.200000E+00
  1.300000E+00  1.400000E+00  1.500000E+00
  1.600000E+00  1.700000E+00  1.800000E+00
";

const TYPE1_BAD_PAIR_DYM: &str = "\
    1
    1
   29
   64.000000
    0.00000000    0.00000000    0.00000000
    2    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
";

const TYPE1_MISSING_ATOMIC_METADATA_DYM: &str = "\
    1
    2
    0
    8
   63.546000
    0.000000
    0.00000000    0.00000000    0.00000000
    1.00000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
";
