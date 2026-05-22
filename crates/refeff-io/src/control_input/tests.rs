use crate::control_input::{
    BandInput, DensityGridKind, DensityInput, FullSpectrumInput, OpconsInput, ReciprocalInput,
    band_input_string, density_input_string, fullspectrum_input_string, opcons_input_string,
    reciprocal_input_string,
};
use crate::{FeffDocument, FeffInput, rdinp};

#[test]
fn parses_generated_band_input() -> crate::Result<()> {
    let band = BandInput::parse_str("band.inp", &rdinp::band_inp_string())?;

    assert_eq!(band.mband, 0);
    assert_eq!(band.energy_mesh.emin, 0.0);
    assert_eq!(band.energy_mesh.emax, 0.0);
    assert_eq!(band.energy_mesh.estep, 0.0);
    assert_eq!(band.nkp, 0);
    assert_eq!(band.ikpath, -1);
    assert!(!band.freeprop);
    Ok(())
}

#[test]
fn renders_band_input_text() -> crate::Result<()> {
    let band = BandInput::parse_str("band.inp", &rdinp::band_inp_string())?;
    let rendered = band_input_string(&band)?;
    let reparsed = BandInput::parse_str("band.inp", &rendered)?;

    assert_eq!(rendered, rdinp::band_inp_string());
    assert_eq!(reparsed, band);
    Ok(())
}

#[test]
fn parses_empty_density_input() -> crate::Result<()> {
    let density = DensityInput::parse_str("density.inp", "")?;

    assert!(density.grids.is_empty());
    Ok(())
}

#[test]
fn parses_density_grid_requests() -> crate::Result<()> {
    let density = DensityInput::parse_str(
        "density.inp",
        concat!(
            "# comment\n",
            "line line.dat 0.0 1.0 2.0 core\n",
            "1.0 0.0 0.0 101\n",
            "plane plane.dat 0.0 0.0 0.0\n",
            "1.0 0.0 0.0 11\n",
            "0.0 1.0 0.0 12\n",
        ),
    )?;
    let line = density.grids.first().ok_or_else(|| crate::IoError::Parse {
        path: "density.inp".into(),
        line: 0,
        message: "expected line grid".to_string(),
    })?;
    let plane = density.grids.get(1).ok_or_else(|| crate::IoError::Parse {
        path: "density.inp".into(),
        line: 0,
        message: "expected plane grid".to_string(),
    })?;
    let line_axis = line.axes.first().ok_or_else(|| crate::IoError::Parse {
        path: "density.inp".into(),
        line: 0,
        message: "expected line axis".to_string(),
    })?;

    assert_eq!(line.kind, DensityGridKind::Line);
    assert_eq!(line.filename, "line.dat");
    assert!(line.core);
    assert_eq!(line.axes.len(), 1);
    assert_eq!(line_axis.points, 101);
    assert_eq!(plane.kind, DensityGridKind::Plane);
    assert_eq!(plane.axes.len(), 2);
    Ok(())
}

#[test]
fn parses_comma_separated_density_grid_requests() -> crate::Result<()> {
    let density = DensityInput::parse_str(
        "density.inp",
        concat!(
            "line,line.dat,0.0,1.0,2.0,core\n",
            "1.0,0.0,0.0,101\n",
            "plane plane.dat 0.0, 1.0 2.0\n",
            "1.0,0.0,0.0,11\n",
            "0.0,1.0,0.0,12\n",
        ),
    )?;

    assert_eq!(density.grids.len(), 2);
    assert_eq!(density.grids[0].kind, DensityGridKind::Line);
    assert_eq!(density.grids[0].filename, "line.dat");
    assert_eq!(density.grids[0].origin, [0.0, 1.0, 2.0]);
    assert!(density.grids[0].core);
    assert_eq!(density.grids[0].axes[0].vector, [1.0, 0.0, 0.0]);
    assert_eq!(density.grids[0].axes[0].points, 101);
    assert_eq!(density.grids[1].kind, DensityGridKind::Plane);
    assert_eq!(density.grids[1].origin, [0.0, 1.0, 2.0]);
    assert_eq!(density.grids[1].axes[1].vector, [0.0, 1.0, 0.0]);
    Ok(())
}

#[test]
fn renders_density_input_text() -> crate::Result<()> {
    let density = DensityInput::parse_str(
        "density.inp",
        concat!(
            "line line.dat 0.0 1.0 2.0 core\n",
            "1.0 0.0 0.0 101\n",
            "volume volume.bin -1.0 0.0 1.0\n",
            "1.0 0.0 0.0 5\n",
            "0.0 1.0 0.0 6\n",
            "0.0 0.0 1.0 7\n",
        ),
    )?;
    let rendered = density_input_string(&density)?;
    let reparsed = DensityInput::parse_str("density.inp", &rendered)?;

    assert_eq!(reparsed, density);
    assert!(rendered.contains("line line.dat 0.000000000000000"));
    assert!(rendered.contains(" core\n"));
    assert!(rendered.contains("volume volume.bin -1.000000000000000"));
    Ok(())
}

#[test]
fn renders_density_filename_with_feff_fixed_width() -> crate::Result<()> {
    let density = crate::DensityInput {
        grids: vec![crate::DensityGrid {
            kind: DensityGridKind::Line,
            filename: "123456789012345678901234567890ABCDE".to_string(),
            origin: [0.0, 0.0, 0.0],
            core: false,
            axes: vec![crate::DensityAxis {
                vector: [1.0, 0.0, 0.0],
                points: 2,
            }],
        }],
    };
    let rendered = density_input_string(&density)?;
    let reparsed = DensityInput::parse_str("density.inp", &rendered)?;

    assert!(rendered.contains("line 123456789012345678901234567890 "));
    assert_eq!(
        reparsed.grids.first().map(|grid| grid.filename.as_str()),
        Some("123456789012345678901234567890")
    );
    Ok(())
}

#[test]
fn parses_density_fixed_fields_like_feff_reference() -> crate::Result<()> {
    let density = DensityInput::parse_str(
        "density.inp",
        concat!(
            "line 123456789012345678901234567890ABCDE 0.0 0.0 0.0 core\n",
            "1.0 0.0 0.0 2\n",
            "line density.dat 0.0 0.0 0.0 CORE\n",
            "1.0 0.0 0.0 2\n",
            "line density.dat 0.0 0.0 0.0 extra core\n",
            "1.0 0.0 0.0 2\n",
        ),
    )?;

    assert_eq!(density.grids[0].filename, "123456789012345678901234567890");
    assert!(density.grids[0].core);
    assert_eq!(density.grids[1].filename, "density.dat");
    assert!(!density.grids[1].core);
    assert_eq!(density.grids[2].filename, "density.dat");
    assert!(!density.grids[2].core);
    Ok(())
}

#[test]
fn converts_density_grid_to_bohr_like_feff_reference() -> crate::Result<()> {
    let density = DensityInput::parse_str(
        "density.inp",
        concat!(
            "plane plane.dat 0.0 1.0 2.0 core\n",
            "1.0 0.0 0.0 11\n",
            "0.0 1.5 0.25 12\n",
        ),
    )?;
    let grids = density.to_bohr_grids()?;
    let grid = grids.first().ok_or_else(|| crate::IoError::Parse {
        path: "density.inp".into(),
        line: 0,
        message: "expected density grid".to_string(),
    })?;
    let input = grid.as_rhorrp_input();

    assert_eq!(grid.kind, DensityGridKind::Plane);
    assert_eq!(grid.filename, "plane.dat");
    assert!(grid.core);
    assert_eq!(grid.points_per_axis, [11, 12]);
    assert_eq!(input.points_per_axis, [11, 12]);
    assert_close(grid.origin[0], 0.0);
    assert_close(grid.origin[1], 1.889_725_988_578_923_3);
    assert_close(grid.origin[2], 3.779_451_977_157_846_5);
    assert_close(input.axes[(0, 0)], 1.889_725_988_578_923_3);
    assert_close(input.axes[(1, 0)], 0.0);
    assert_close(input.axes[(2, 0)], 0.0);
    assert_close(input.axes[(0, 1)], 0.0);
    assert_close(input.axes[(1, 1)], 2.834_588_982_868_385);
    assert_close(input.axes[(2, 1)], 0.472_431_497_144_730_8);
    Ok(())
}

#[test]
fn rejects_density_grid_bohr_axis_count_mismatch() {
    let grid = crate::DensityGrid {
        kind: DensityGridKind::Plane,
        filename: "bad.dat".to_string(),
        origin: [0.0, 0.0, 0.0],
        core: false,
        axes: vec![crate::DensityAxis {
            vector: [1.0, 0.0, 0.0],
            points: 11,
        }],
    };

    assert!(grid.to_bohr_grid().is_err());
}

#[test]
fn rejects_invalid_density_input_rendering() {
    let bad_axis_count = crate::DensityInput {
        grids: vec![crate::DensityGrid {
            kind: DensityGridKind::Plane,
            filename: "bad.dat".to_string(),
            origin: [0.0, 0.0, 0.0],
            core: false,
            axes: vec![crate::DensityAxis {
                vector: [1.0, 0.0, 0.0],
                points: 2,
            }],
        }],
    };
    assert!(density_input_string(&bad_axis_count).is_err());

    let empty_filename = crate::DensityInput {
        grids: vec![crate::DensityGrid {
            kind: DensityGridKind::Line,
            filename: String::new(),
            origin: [0.0, 0.0, 0.0],
            core: false,
            axes: vec![crate::DensityAxis {
                vector: [1.0, 0.0, 0.0],
                points: 2,
            }],
        }],
    };
    assert!(density_input_string(&empty_filename).is_err());

    let spaced_filename = crate::DensityInput {
        grids: vec![crate::DensityGrid {
            kind: DensityGridKind::Line,
            filename: "bad name.dat".to_string(),
            origin: [0.0, 0.0, 0.0],
            core: false,
            axes: vec![crate::DensityAxis {
                vector: [1.0, 0.0, 0.0],
                points: 2,
            }],
        }],
    };
    assert!(density_input_string(&spaced_filename).is_err());

    let nonfinite = crate::DensityInput {
        grids: vec![crate::DensityGrid {
            kind: DensityGridKind::Line,
            filename: "bad.dat".to_string(),
            origin: [f64::NAN, 0.0, 0.0],
            core: false,
            axes: vec![crate::DensityAxis {
                vector: [1.0, 0.0, 0.0],
                points: 2,
            }],
        }],
    };
    assert!(density_input_string(&nonfinite).is_err());
}

#[test]
fn tokenizes_control_fields_like_feff_bwords_reference() {
    assert_eq!(
        super::parser::fields("line,line.dat,0.0,1.0,2.0,core"),
        ["line", "line.dat", "0.0", "1.0", "2.0", "core"]
    );
    assert_eq!(
        super::parser::fields("plane plane.dat 0.0, 1.0 2.0"),
        ["plane", "plane.dat", "0.0", "1.0", "2.0"]
    );
    assert_eq!(super::parser::fields(",leading"), ["", "leading"]);
    assert_eq!(
        super::parser::fields("middle,,blank"),
        ["middle", "", "blank"]
    );
    assert_eq!(super::parser::fields("trailing,"), ["trailing"]);
}

#[test]
fn truncates_fixed_fortran_strings_on_utf8_boundaries() {
    assert_eq!(super::common::fortran_fixed_string("abcdef", 3), "abc");
    assert_eq!(super::common::fortran_fixed_string("aébc", 3), "aé");
}

#[test]
fn parses_generated_fullspectrum_input() -> crate::Result<()> {
    let fullspectrum =
        FullSpectrumInput::parse_str("fullspectrum.inp", &rdinp::fullspectrum_inp_string())?;

    assert_eq!(fullspectrum.m_full_spectrum, 0);
    Ok(())
}

#[test]
fn renders_fullspectrum_input_text() -> crate::Result<()> {
    let fullspectrum =
        FullSpectrumInput::parse_str("fullspectrum.inp", &rdinp::fullspectrum_inp_string())?;
    let rendered = fullspectrum_input_string(&fullspectrum)?;
    let reparsed = FullSpectrumInput::parse_str("fullspectrum.inp", &rendered)?;

    assert_eq!(rendered, rdinp::fullspectrum_inp_string());
    assert_eq!(reparsed, fullspectrum);
    Ok(())
}

#[test]
fn parses_generated_opcons_input() -> crate::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
OPCONS
POTENTIALS
0 29 Cu
1 29 Cu
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let text = rdinp::opcons_inp_string(&document)?;
    let opcons = OpconsInput::parse_str("opcons.inp", &text)?;

    assert!(opcons.run_opcons);
    assert!(!opcons.print_eps);
    assert_eq!(opcons.number_densities, vec![-1.0, -1.0]);
    Ok(())
}

#[test]
fn renders_opcons_input_text() -> crate::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
OPCONS
POTENTIALS
0 29 Cu
1 29 Cu
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let text = rdinp::opcons_inp_string(&document)?;
    let opcons = OpconsInput::parse_str("opcons.inp", &text)?;
    let rendered = opcons_input_string(&opcons)?;
    let reparsed = OpconsInput::parse_str("opcons.inp", &rendered)?;

    assert_eq!(rendered, text);
    assert_eq!(reparsed, opcons);
    Ok(())
}

#[test]
fn rejects_invalid_control_input_rendering() {
    let bad_band = crate::BandInput {
        mband: 0,
        energy_mesh: crate::BandEnergyMesh {
            emin: f64::NAN,
            emax: 0.0,
            estep: 0.0,
        },
        nkp: 0,
        ikpath: -1,
        freeprop: false,
    };
    assert!(band_input_string(&bad_band).is_err());

    let bad_opcons = crate::OpconsInput {
        run_opcons: true,
        print_eps: false,
        number_densities: vec![1.0, f64::INFINITY],
    };
    assert!(opcons_input_string(&bad_opcons).is_err());
}

#[test]
fn parses_generated_reciprocal_input() -> crate::Result<()> {
    let text = rdinp::reciprocal_inp_string();
    let reciprocal = ReciprocalInput::parse_str("reciprocal.inp", &text)?;

    assert_eq!(reciprocal.ispace, 1);
    assert!(reciprocal.cell.is_none());
    assert_eq!(reciprocal_input_string(&reciprocal)?, text);
    Ok(())
}

#[test]
fn parses_reciprocal_cell_block() -> crate::Result<()> {
    let reciprocal = ReciprocalInput::parse_str(
        "reciprocal.inp",
        concat!(
            "ispace\n",
            "   0\n",
            "lattice vectors  (in A, in Carthesian coordinates)\n",
            "      1.00000      0.00000      0.00000\n",
            "      0.00000      1.00000      0.00000\n",
            "      0.00000      0.00000      1.00000\n",
            "Volume scaling factor (A^3); eimag; core hole\n",
            "     -1.00000      0.00000      1.00000\n",
            "lattice type  (P,I,F,R,B,CXY,CYZ,CXZ)\n",
            "P      P1          1\n",
            "#atoms in unit cell ; position absorber ; corehole?\n",
            "   2   1   1\n",
            "# k-points total/x/y/z ; ktype; use symmetry?\n",
            "8 2 2 2 0 T\n",
            "ppos\n",
            "      0.00000      0.00000      0.00000\n",
            "      0.50000      0.50000      0.50000\n",
            "ppot\n",
            "0 1\n",
            "label\n",
            "Cu Zn\n",
            "streta,strgmax,strrmax\n",
            "      0.10000      2.00000      3.00000\n",
        ),
    )?;
    let cell = reciprocal.cell.ok_or_else(|| crate::IoError::Parse {
        path: "reciprocal.inp".into(),
        line: 0,
        message: "expected reciprocal cell".to_string(),
    })?;

    assert_eq!(reciprocal.ispace, 0);
    assert_eq!(cell.atom_count, 2);
    assert_eq!(cell.k_mesh.total, 8);
    assert!(cell.k_mesh.use_symmetry);
    assert_eq!(cell.potentials, vec![0, 1]);
    assert_eq!(cell.labels, vec!["Cu".to_string(), "Zn".to_string()]);
    assert!(
        reciprocal_input_string(&ReciprocalInput {
            ispace: 0,
            cell: Some(cell)
        })?
        .contains("# k-points total/x/y/z ; ktype; use symmetry?\n")
    );
    Ok(())
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-14,
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}
