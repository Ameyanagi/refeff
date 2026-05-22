use crate::error::Result;

use super::common::invalid_apot_bin;
use super::parse::parse_matrix_shape;
use super::*;

#[test]
fn parses_records_and_matrix_sections() -> Result<()> {
    let data = parse_apot_bin(APOT_BIN)?;
    assert_eq!(data.section_count(), 3);
    assert_eq!(data.matrix_count(), 1);

    let first = &data.sections[0];
    assert_eq!(first.section_number, 1);
    assert_eq!(first.column_labels, ["nph", "nat", "ihole", "s02"]);
    let Some(records) = first.records() else {
        return invalid_apot_bin(0, "first section should contain records");
    };
    assert_eq!(records.row_count(), 1);
    assert_eq!(records.column_count(), 4);
    assert_eq!(records.rows[0][0], ApotBinValue::Int(1));
    assert_eq!(records.rows[0][3], ApotBinValue::Real(0.95));

    let Some(matrix) = data.sections[2].matrix() else {
        return invalid_apot_bin(0, "third section should contain a matrix");
    };
    assert_eq!(matrix.shape(), (2, 3));
    match &matrix.values {
        ApotBinMatrixValues::Real(values) => assert_eq!(values[[1, 2]], 6.0),
        _ => return invalid_apot_bin(0, "third section should contain real matrix values"),
    }
    assert_eq!(data.sections[2].trailing_headers, ["next block"]);

    Ok(())
}

#[test]
fn parses_compact_i4_shape_fields() -> Result<()> {
    assert_eq!(parse_matrix_shape(1, "   34000")?, (3, 4000));
    Ok(())
}

#[test]
fn roundtrips_apot_bin_data() -> Result<()> {
    let data = parse_apot_bin(APOT_BIN)?;
    let rendered = apot_bin_string(&data)?;
    let reparsed = parse_apot_bin(&rendered)?;
    assert_eq!(reparsed, data);
    Ok(())
}

#[test]
fn rejects_bad_apot_bin_data() {
    assert!(parse_apot_bin("").is_err());
    assert!(parse_apot_bin("1 2 3\n").is_err());
    assert!(parse_apot_bin("#SN#   Section:    1\n#DT# Int\n").is_err());
    assert!(
        parse_apot_bin("#SN#   Section:    1\n#DT# 2D double array with sizes    2   2\n1\n")
            .is_err()
    );
    assert!(parse_apot_bin("#SN#   Section:    1\n#DT# Int\nNaN\n").is_err());
}

const APOT_BIN: &str = r#"#SN#   Section:    1
#DF# This section written in TXT .
#H#
#H# The following data types are written in this section.
#DT#  Int Int Int Double
#H# first section
#CL# nph nat ihole s02
     1         79          1     0.9500000000E+00
#SN#   Section:    2
#DF# This section written in TXT .
#H#
#H# The following data types are written in this section.
#DT#  Int Double
    29     0.2838535628D+01
    30     0.2632330371E+01
#SN#   Section:    3
#DF# This section written in TXT .
#H#
#DT# 2D double array with sizes    2   3
#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . .
#H#                                Array(2,i)
#H#                                     .
#H#                                     .
#H#                                     .
#H# matrix
1.0000000000E+00    2.0000000000E+00    3.0000000000E+00
4.0000000000E+00    5.0000000000E+00    6.0000000000E+00
#H# next block
"#;
