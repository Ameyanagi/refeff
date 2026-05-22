use super::common::parse_error_value;
use super::*;
use crate::error::Result;

#[test]
fn parses_empty_dmdw_out() -> Result<()> {
    let parsed = parse_dmdw_out("")?;
    assert_eq!(parsed.header, None);
    assert!(!parsed.mass_enhancement_header);
    assert_eq!(parsed.section_count(), 0);
    assert_eq!(dmdw_out_string(&parsed)?, "");
    Ok(())
}

#[test]
fn parses_type2_mass_enhancement_header() -> Result<()> {
    let parsed = parse_dmdw_out(DMDW_OUT_SELF_ENERGY)?;
    let header = parsed
        .header
        .as_ref()
        .ok_or_else(|| parse_error_value(0, "test fixture should contain a dmdw.out header"))?;

    assert_eq!(header.lanczos_recursion_order, 6);
    assert_eq!(header.temperature, DmdwOutTemperature::Single(450.0));
    assert_eq!(header.dynamical_matrix_file, "feff.dym");
    assert!(parsed.mass_enhancement_header);
    assert_eq!(parsed.section_count(), 0);

    let rendered = dmdw_out_string(&parsed)?;
    assert_eq!(rendered, DMDW_OUT_SELF_ENERGY);
    assert_eq!(parse_dmdw_out(&rendered)?, parsed);
    Ok(())
}

#[test]
fn parses_path_dmdw_out() -> Result<()> {
    let parsed = parse_dmdw_out(DMDW_OUT)?;
    let header = parsed
        .header
        .as_ref()
        .ok_or_else(|| parse_error_value(0, "test fixture should contain a dmdw.out header"))?;
    assert_eq!(header.lanczos_recursion_order, 6);
    assert_eq!(header.temperature, DmdwOutTemperature::Single(450.0));
    assert_eq!(header.dynamical_matrix_file, "feff.dym");
    assert_eq!(parsed.section_count(), 1);

    let section = &parsed.sections[0];
    assert_eq!(section.subject, DmdwOutSubject::PathIndices(vec![1, 2]));
    assert_eq!(section.pdos_poles.len(), 6);
    assert_eq!(section.pdos_poles[0].frequency_thz, 2.860);
    assert_eq!(section.pdos_poles[0].weight, 0.039_469_598);
    assert_eq!(
        section.einstein,
        Some(DmdwOutEinstein {
            frequency_thz: 5.784,
            temperature_kelvin: 277.60,
            effective_force_constant_n_per_m: 69.6914,
        })
    );
    assert_eq!(section.moments.len(), 5);
    assert_eq!(section.moments[2].order, 0);
    assert_eq!(section.moments[2].frequency_thz, None);
    assert_eq!(section.reduced_mass_amu, Some(31.773));
    assert_eq!(section.path_length_angstrom, Some(2.5323));
    assert_eq!(section.sigma2_1e_minus_3_angstrom2, Some(11.8576));

    let rendered = dmdw_out_string(&parsed)?;
    assert_eq!(rendered, DMDW_OUT);
    assert_eq!(parse_dmdw_out(&rendered)?, parsed);
    Ok(())
}

#[test]
fn parses_multi_temperature_and_atom_variants() -> Result<()> {
    let parsed = parse_dmdw_out(DMDW_OUT_VARIANTS)?;
    assert_eq!(
        parsed.header.as_ref().map(|header| &header.temperature),
        Some(&DmdwOutTemperature::ListedBelow)
    );
    assert_eq!(parsed.section_count(), 3);

    let path_section = &parsed.sections[0];
    assert_eq!(path_section.sigma2_by_temperature.len(), 2);
    assert_eq!(
        path_section.sigma2_by_temperature[1].temperature_kelvin,
        300.0
    );
    assert_eq!(path_section.sigma2_by_temperature[1].value, 2.5);

    let atom_section = &parsed.sections[1];
    assert_eq!(
        atom_section.subject,
        DmdwOutSubject::AtomIndex {
            indices: vec![3],
            direction: Some("x".to_owned()),
        }
    );
    assert_eq!(atom_section.u2_by_temperature.len(), 2);
    assert!(atom_section.projected_dos_component_computed);

    let total_section = &parsed.sections[2];
    assert_eq!(total_section.subject, DmdwOutSubject::TotalPdos);
    assert_eq!(total_section.vibrational_free_energy_ev, Some(-0.125));

    let rendered = dmdw_out_string(&parsed)?;
    assert_eq!(rendered, DMDW_OUT_VARIANTS);
    assert_eq!(parse_dmdw_out(&rendered)?, parsed);
    Ok(())
}

#[test]
fn parses_total_vfe_section() -> Result<()> {
    let parsed = parse_dmdw_out(DMDW_OUT_TOTAL_VFE)?;

    assert_eq!(parsed.section_count(), 1);
    let section = &parsed.sections[0];
    assert_eq!(section.subject, DmdwOutSubject::TotalVfe);
    assert_eq!(section.vibrational_free_energy_by_temperature.len(), 2);
    assert_eq!(
        section.vibrational_free_energy_by_temperature[1].temperature_kelvin,
        300.0
    );
    assert_eq!(
        section.vibrational_free_energy_by_temperature[1].value,
        -0.125
    );
    assert_eq!(parse_dmdw_out(&dmdw_out_string(&parsed)?)?, parsed);
    Ok(())
}

#[test]
fn allows_total_pdos_to_collect_more_poles_than_lanczos_order() -> Result<()> {
    let parsed = parse_dmdw_out(DMDW_OUT_TOTAL_PDOS_AGGREGATE)?;

    assert_eq!(parsed.section_count(), 1);
    assert_eq!(parsed.sections[0].subject, DmdwOutSubject::TotalPdos);
    assert_eq!(parsed.sections[0].pdos_poles.len(), 3);
    assert_eq!(parse_dmdw_out(&dmdw_out_string(&parsed)?)?, parsed);
    Ok(())
}

#[test]
fn accepts_fortran_d_exponents() -> Result<()> {
    let parsed = parse_dmdw_out(DMDW_OUT.replace("450.00", "4.5D+02").as_str())?;
    assert_eq!(
        parsed.header.as_ref().map(|header| &header.temperature),
        Some(&DmdwOutTemperature::Single(450.0))
    );
    Ok(())
}

#[test]
fn rejects_bad_dmdw_out_inputs() {
    assert!(parse_dmdw_out("# Temperature: 300.00\n").is_err());
    assert!(
        parse_dmdw_out(
            "# Lanczos recursion order: 0\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\n"
        )
        .is_err()
    );
    assert!(
        parse_dmdw_out(
            "# Lanczos recursion order: 1\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\nPath Indices: 0\n"
        )
        .is_err()
    );
    assert!(
        parse_dmdw_out(
            "# Lanczos recursion order: 1\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\nPath Indices: 1\nPDOS Poles:\nFreq. (THz) Weight\nNaN 1\n"
        )
        .is_err()
    );
    assert!(
        parse_dmdw_out(
            "# Lanczos recursion order: 2\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\nPath Indices: 1\nPDOS Poles:\nFreq. (THz) Weight\n1 1\n"
        )
        .is_err()
    );
}

const DMDW_OUT: &str = concat!(
    r#"# Lanczos recursion order:    6
# Temperature:  450.00
# Dynamical matrix file: feff.dym

--------------------------------------------------------------
 Path Indices:    1   2
 PDOS Poles:
     Freq. (THz)    Weight
        2.860       0.039469598
        3.854       0.182890396
        4.940       0.220041663
        6.026       0.159715119
        6.812       0.284980130
        7.306       0.112876736

"#,
    " PDOS Einstein freq (single pole), associated temp and eff. force constant: \n",
    r#" Freq (THz)   Temp (K)   Eff. FC (N/m)
   5.784       277.60      69.6914

 pDOS n Moments, associated Einstein freqs, temps and eff. force constants:
  n     Mom (THz^n)   Freq (THz)     Temp (K)    Eff. FC (N/m)
 -2       0.03881       5.07607       243.60      53.6688
 -1       0.18959       5.27461       253.13      57.9492
  0       0.99997     ---------     --------
  1       5.63317       5.63317       270.34      66.0957
  2      33.45823       5.78431       277.59      69.6899

 Path Red. Mass (AMU):   31.773000
 Path Length (Ang), s^2 (1e-3 Ang^2):  2.5323  11.8576
--------------------------------------------------------------
"#
);

const DMDW_OUT_SELF_ENERGY: &str = concat!(
    "# Lanczos recursion order:    6\n",
    "# Temperature:  450.00\n",
    "# Dynamical matrix file: feff.dym\n",
    "Mass Enchancement Factor  \n",
);

const DMDW_OUT_VARIANTS: &str = r#"# Lanczos recursion order:    2
# Temperature: (See list Below)
# Dynamical matrix file: feff.dym

--------------------------------------------------------------
 Path Indices:    1   2
 PDOS Poles:
     Freq. (THz)    Weight
        2.000       0.500000000
        3.000       0.500000000

 Path Red. Mass (AMU):   31.773000
 Path Length (Ang):  2.5323
 Temp (K)   s^2 (1e-3 Ang^2)
  100.00       1.2500
  300.00       2.5000

--------------------------------------------------------------
 Atom Index:    3
--------- Direction x ---------------------------
 PDOS Poles:
     Freq. (THz)    Weight
        2.000       0.500000000
        3.000       0.500000000

 Temp (K)   u^2 (1e-3 Ang^2)
  100.00       0.500
  300.00       0.750
 Projected DOS component computed.

--------------------------------------------------------------
 Total PDOS results:
 PDOS Poles:
     Freq. (THz)    Weight
        2.000       0.500000000
        3.000       0.500000000

 VFE (eV):       -0.125000
--------------------------------------------------------------
"#;

const DMDW_OUT_TOTAL_VFE: &str = "\
# Lanczos recursion order:    2
# Temperature: (See list Below)
# Dynamical matrix file: feff.dym

--------------------------------------------------------------
 Total VFE for the paths requested:
 Temp (K)        VFE (eV)
  100.00     -0.050000
  300.00     -0.125000
--------------------------------------------------------------
";

const DMDW_OUT_TOTAL_PDOS_AGGREGATE: &str = "\
# Lanczos recursion order:    1
# Temperature:  300.00
# Dynamical matrix file: feff.dym

--------------------------------------------------------------
 Total PDOS results:
 PDOS Poles:
     Freq. (THz)    Weight
        1.000       0.250000000
        2.000       0.500000000
        3.000       0.250000000

 Projected DOS component computed.

--------------------------------------------------------------
";
