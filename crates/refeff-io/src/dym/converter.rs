use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, Array4};
use refeff_core::atomic_symbol;

use crate::error::{IoError, Result};
use crate::model::{Atom, Potential};

use super::validate::{invalid_dym, validate_dym};
use super::{DymCoordinates, DymData, DymType2Metadata, DymUniqueAtom, write_dym};

const SHELL_SORT_TOLERANCE_BOHR: f64 = 1.0e-4;
const DYM_BOHR_ANGSTROM: f64 = 1.0 / 1.889_726_663_510_319_2;

/// Spectrum template selected for a generated `feff.inp`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DymSpectrum {
    /// Generate the production converter's default EXAFS controls.
    #[default]
    Exafs,
    /// Generate the production converter's XANES controls.
    Xanes,
}

/// Typed options for FEFF10's `dym2feffinp` conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DymToFeffOptions {
    /// Zero-based absorber index in the source `.dym` atom order.
    pub center_atom_index: usize,
    /// Spectrum control template to include in the generated input.
    pub spectrum: DymSpectrum,
    /// Include the standard FEFF header. Set this to `false` for `--j`.
    pub write_header: bool,
}

impl Default for DymToFeffOptions {
    fn default() -> Self {
        Self {
            center_atom_index: 0,
            spectrum: DymSpectrum::Exafs,
            write_header: true,
        }
    }
}

/// Typed result of FEFF10's `dym2feffinp` conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct DymFeffConversion {
    /// Converter options used to produce this result.
    pub options: DymToFeffOptions,
    /// Source atom indices in the distance-sorted output order.
    pub atom_order: Vec<usize>,
    /// FEFF `POTENTIALS` rows, including the absorber as potential zero.
    pub potentials: Vec<Potential>,
    /// Recentered, distance-sorted FEFF `ATOMS` rows in Angstrom.
    pub atoms: Vec<Atom>,
    /// Recentered and consistently permuted `.dym` data.
    pub adjusted_dym: DymData,
}

/// Convert parsed `.dym` data to FEFF input tables and a matching adjusted matrix.
///
/// This ports `DMDW/m_dmdw.f90::Write_Feffinp` and `Write_dym`. In
/// particular, atom distances use Cartesian Bohr coordinates, the production
/// shell-sort tolerance is retained, and non-absorber potential numbers are
/// assigned in source atom order.
pub fn convert_dym_to_feff(data: &DymData, options: DymToFeffOptions) -> Result<DymFeffConversion> {
    validate_dym(data)?;
    let atom_count = data.atom_count();
    if options.center_atom_index >= atom_count {
        return Err(invalid_dym(
            "center atom",
            format!(
                "1-based center {} is outside the source atom range 1..={atom_count}",
                options.center_atom_index.saturating_add(1)
            ),
        ));
    }

    let cartesian = data.coordinates.cartesian_positions();
    let center = cartesian.row(options.center_atom_index).to_owned();
    let mut distances = Vec::with_capacity(atom_count);
    for row in cartesian.rows() {
        let squared = row
            .iter()
            .zip(center.iter())
            .map(|(coordinate, center_coordinate)| (coordinate - center_coordinate).powi(2))
            .sum::<f64>();
        distances.push(squared.sqrt());
    }
    let (atom_order, sorted_distances) = production_shell_sort(&distances);

    let (atom_types, type_count) =
        assign_atom_types(&data.atomic_numbers, options.center_atom_index)?;
    let symbols = data
        .atomic_numbers
        .iter()
        .map(|&atomic_number| converter_atomic_symbol(atomic_number))
        .collect::<Result<Vec<_>>>()?;
    let potentials = build_potentials(
        data,
        options.center_atom_index,
        &atom_types,
        type_count,
        &symbols,
    );
    let atoms = atom_order
        .iter()
        .zip(sorted_distances.iter())
        .enumerate()
        .map(|(output_index, (&source_index, &distance_bohr))| Atom {
            x: (cartesian[[source_index, 0]] - center[0]) * DYM_BOHR_ANGSTROM,
            y: (cartesian[[source_index, 1]] - center[1]) * DYM_BOHR_ANGSTROM,
            z: (cartesian[[source_index, 2]] - center[2]) * DYM_BOHR_ANGSTROM,
            ipot: atom_types[source_index],
            tag: Some(symbols[source_index].clone()),
            distance: Some(distance_bohr * DYM_BOHR_ANGSTROM),
            index: Some(output_index),
        })
        .collect();
    let adjusted_dym = permute_dym(data, &atom_order)?;

    Ok(DymFeffConversion {
        options,
        atom_order,
        potentials,
        atoms,
        adjusted_dym,
    })
}

/// Render the FEFF input side of a typed `dym2feffinp` conversion.
pub fn dym_feff_inp_string(conversion: &DymFeffConversion) -> Result<String> {
    let mut out = String::new();
    if conversion.options.write_header {
        write_header(&mut out, conversion.options.spectrum)?;
    }

    writeln!(out, "POTENTIALS")?;
    for potential in &conversion.potentials {
        let symbol = potential.tag.as_deref().unwrap_or_default();
        writeln!(
            out,
            "{:5}{:5}   {symbol:<2}",
            potential.ipot,
            potential.z.unwrap_or_default()
        )?;
    }
    writeln!(out)?;

    writeln!(out, "ATOMS")?;
    for atom in &conversion.atoms {
        let symbol = atom.tag.as_deref().unwrap_or_default();
        writeln!(
            out,
            "{:11.5}{:11.5}{:11.5}{:5}   {symbol:<2}{:9.5}{:5}",
            atom.x,
            atom.y,
            atom.z,
            atom.ipot,
            atom.distance.unwrap_or_default(),
            atom.index.unwrap_or_default()
        )?;
    }
    writeln!(out, "END")?;
    Ok(out)
}

/// Write both production `dym2feffinp` output files.
pub fn write_dym_feff_outputs(
    feff_path: impl AsRef<Path>,
    dym_path: impl AsRef<Path>,
    conversion: &DymFeffConversion,
) -> Result<()> {
    let feff_path = feff_path.as_ref();
    std::fs::write(feff_path, dym_feff_inp_string(conversion)?)
        .map_err(|source| IoError::io(feff_path, source))?;
    write_dym(dym_path, &conversion.adjusted_dym)
}

fn converter_atomic_symbol(atomic_number: i32) -> Result<String> {
    let atomic_number = usize::try_from(atomic_number)
        .map_err(|_| invalid_dym("atomic number", "value must be positive"))?;
    if atomic_number > 103 {
        return Err(invalid_dym(
            "atomic number",
            format!(
                "dym2feffinp's FEFF10 periodic table ends at atomic number 103, got {atomic_number}"
            ),
        ));
    }
    atomic_symbol(atomic_number)
        .map(str::trim)
        .map(str::to_string)
        .map_err(|error| invalid_dym("atomic number", error.to_string()))
}

fn assign_atom_types(
    atomic_numbers: &Array1<i32>,
    center_atom_index: usize,
) -> Result<(Vec<i32>, i32)> {
    let mut element_types = vec![0_i32; 104];
    let mut type_count = 0_i32;
    for (atom_index, &atomic_number) in atomic_numbers.iter().enumerate() {
        let atomic_number = usize::try_from(atomic_number)
            .map_err(|_| invalid_dym("atomic number", "value must be positive"))?;
        let element_type = element_types.get_mut(atomic_number).ok_or_else(|| {
            invalid_dym(
                "atomic number",
                format!(
                    "dym2feffinp's FEFF10 periodic table ends at atomic number 103, got {atomic_number}"
                ),
            )
        })?;
        if *element_type == 0 && atom_index != center_atom_index {
            type_count += 1;
            *element_type = type_count;
        }
    }

    let mut atom_types = atomic_numbers
        .iter()
        .map(|&atomic_number| {
            usize::try_from(atomic_number)
                .ok()
                .and_then(|index| element_types.get(index))
                .copied()
                .ok_or_else(|| invalid_dym("atomic number", "value is outside the FEFF10 table"))
        })
        .collect::<Result<Vec<_>>>()?;
    atom_types[center_atom_index] = 0;
    Ok((atom_types, type_count))
}

fn build_potentials(
    data: &DymData,
    center_atom_index: usize,
    atom_types: &[i32],
    type_count: i32,
    symbols: &[String],
) -> Vec<Potential> {
    let mut potentials = Vec::with_capacity(type_count as usize + 1);
    potentials.push(potential(
        0,
        data.atomic_numbers[center_atom_index],
        &symbols[center_atom_index],
    ));
    for potential_index in 1..=type_count {
        if let Some(atom_index) = atom_types
            .iter()
            .position(|&atom_type| atom_type == potential_index)
        {
            potentials.push(potential(
                potential_index,
                data.atomic_numbers[atom_index],
                &symbols[atom_index],
            ));
        }
    }
    potentials
}

fn potential(ipot: i32, atomic_number: i32, symbol: &str) -> Potential {
    Potential {
        ipot,
        z: Some(atomic_number),
        z_token: atomic_number.to_string(),
        tag: Some(symbol.to_string()),
        lmax1: None,
        lmax2: None,
        xnatph: None,
        spinph: None,
    }
}

fn production_shell_sort(unsorted: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let mut data = unsorted.to_vec();
    let mut order = (0..unsorted.len()).collect::<Vec<_>>();
    let mut increment = ((unsorted.len() as f64) / 2.0).round() as usize;

    while increment > 0 {
        for data_index in increment.saturating_sub(1)..data.len() {
            let value = data[data_index];
            let source_index = order[data_index];
            let mut insertion_index = data_index;
            while insertion_index + 1 > increment
                && data[insertion_index - increment] > value + SHELL_SORT_TOLERANCE_BOHR
            {
                data[insertion_index] = data[insertion_index - increment];
                order[insertion_index] = order[insertion_index - increment];
                insertion_index -= increment;
            }
            data[insertion_index] = value;
            order[insertion_index] = source_index;
        }
        increment = ((increment as f64) / 2.2).round() as usize;
    }
    (order, data)
}

fn permute_dym(data: &DymData, atom_order: &[usize]) -> Result<DymData> {
    let atom_count = atom_order.len();
    let atomic_numbers =
        Array1::from_iter(atom_order.iter().map(|&index| data.atomic_numbers[index]));
    let atomic_masses =
        Array1::from_iter(atom_order.iter().map(|&index| data.atomic_masses[index]));
    let coordinates = match &data.coordinates {
        DymCoordinates::Cartesian(positions) => {
            let origin = positions.row(atom_order[0]).to_owned();
            let mut adjusted = Array2::zeros((atom_count, 3));
            for (output_index, &source_index) in atom_order.iter().enumerate() {
                for coordinate in 0..3 {
                    adjusted[[output_index, coordinate]] =
                        positions[[source_index, coordinate]] - origin[coordinate];
                }
            }
            DymCoordinates::Cartesian(adjusted)
        }
        DymCoordinates::Reduced { reduced, cell } => {
            let origin = reduced.row(atom_order[0]).to_owned();
            let mut adjusted = Array2::zeros((atom_count, 3));
            for (output_index, &source_index) in atom_order.iter().enumerate() {
                for coordinate in 0..3 {
                    adjusted[[output_index, coordinate]] =
                        reduced[[source_index, coordinate]] - origin[coordinate];
                }
            }
            DymCoordinates::Reduced {
                reduced: adjusted,
                cell: cell.clone(),
            }
        }
    };

    let mut force_constants = Array4::zeros((atom_count, atom_count, 3, 3));
    for (output_i, &source_i) in atom_order.iter().enumerate() {
        for (output_j, &source_j) in atom_order.iter().enumerate() {
            for row in 0..3 {
                for column in 0..3 {
                    force_constants[[output_i, output_j, row, column]] =
                        data.force_constants[[source_i, source_j, row, column]];
                }
            }
        }
    }

    let type2_metadata = data
        .type2_metadata
        .as_ref()
        .map(|metadata| permute_type2_metadata(metadata, atom_order));
    let dipole_derivatives = data
        .dipole_derivatives
        .as_ref()
        .map(|dipoles| permute_dipoles(dipoles, atom_order));
    let adjusted = DymData {
        dym_type: data.dym_type,
        atomic_numbers,
        atomic_masses,
        coordinates,
        force_constants,
        type2_metadata,
        dipole_derivatives,
    };
    validate_dym(&adjusted)?;
    Ok(adjusted)
}

fn permute_type2_metadata(metadata: &DymType2Metadata, atom_order: &[usize]) -> DymType2Metadata {
    let mut new_indices = vec![0_usize; atom_order.len()];
    for (new_index, &old_index) in atom_order.iter().enumerate() {
        new_indices[old_index] = new_index;
    }
    DymType2Metadata {
        cell_atom_count: metadata.cell_atom_count,
        unique_atoms: metadata
            .unique_atoms
            .iter()
            .map(|unique_atom| DymUniqueAtom {
                atom_type: unique_atom.atom_type,
                center_atom_indices: unique_atom
                    .center_atom_indices
                    .mapv(|old_index| new_indices[old_index]),
                weights: unique_atom.weights.clone(),
                coordinates: unique_atom.coordinates.clone(),
            })
            .collect(),
    }
}

fn permute_dipoles(dipoles: &Array3<f64>, atom_order: &[usize]) -> Array3<f64> {
    let mut adjusted = Array3::zeros((atom_order.len(), 3, 3));
    for (output_index, &source_index) in atom_order.iter().enumerate() {
        for displacement in 0..3 {
            for dipole in 0..3 {
                adjusted[[output_index, displacement, dipole]] =
                    dipoles[[source_index, displacement, dipole]];
            }
        }
    }
    adjusted
}

fn write_header(out: &mut String, spectrum: DymSpectrum) -> std::fmt::Result {
    out.push_str(concat!(
        " * This feff9 input file was generated by dym2feffinp\n",
        "\n",
        " TITLE dymfile name:  Need to fix\n",
        " TITLE absorbing atom:   0\n",
        "\n",
        " EDGE      L3\n",
        " S02       1.0000\n",
        "\n",
        " *              pot    xsph     fms   paths  genfmt  ff2chi\n",
        " CONTROL          1       1       1       1       1       1\n",
        " PRINT            1       0       0       0       0       0\n",
        "\n",
        " *          ixc  [ Vr  Vi ]\n",
        " EXCHANGE     0\n",
        "\n",
        " *            r_scf  [ l_scf   n_scf   ca ]\n",
        " SCF          4.000\n",
        "\n",
        " *             kmax   [ delta_k  delta_e ]\n",
    ));
    match spectrum {
        DymSpectrum::Xanes => out.push_str(concat!(
            " XANES        4.000\n",
            "\n",
            " *            r_fms     l_fms\n",
            " FMS          6.000\n",
        )),
        DymSpectrum::Exafs => out.push_str(concat!(
            " * XANES        4.000\n",
            "\n",
            " *            r_fms     l_fms\n",
            " * FMS          6.000\n",
        )),
    }
    out.push_str(concat!(
        "\n",
        " *             emin    emax   eimag\n",
        " * LDOS       -30.000  20.000   0.100\n",
        "\n",
    ));
    match spectrum {
        DymSpectrum::Xanes => {
            out.push_str(concat!(" RPATH        0.100\n", "*EXAFS       20.000\n",))
        }
        DymSpectrum::Exafs => out.push_str(concat!(
            " RPATH        6.000\n",
            " EXAFS       20.000\n",
            " NLEG             3\n",
        )),
    }
    out.push_str(concat!(
        "\n",
        " *        Temp  Debye_Temp  DW_Opt  dymfile  DMDW_Order  DMDW_Type  DMDW_Route\n",
        " DEBYE    300.0   315.0  5 feff.dym  16  0  1\n",
        "\n",
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array4};

    use super::*;

    #[test]
    fn converts_typed_tables_and_permuted_dym() -> Result<()> {
        let data = sample_dym()?;
        let conversion = convert_dym_to_feff(
            &data,
            DymToFeffOptions {
                center_atom_index: 1,
                spectrum: DymSpectrum::Xanes,
                write_header: true,
            },
        )?;

        assert_eq!(conversion.atom_order, vec![1, 2, 0]);
        assert_eq!(
            conversion
                .potentials
                .iter()
                .map(|potential| (potential.ipot, potential.z))
                .collect::<Vec<_>>(),
            vec![(0, Some(1)), (1, Some(8)), (2, Some(1))]
        );
        assert_eq!(
            conversion
                .atoms
                .iter()
                .map(|atom| atom.ipot)
                .collect::<Vec<_>>(),
            vec![0, 2, 1]
        );
        assert_eq!(
            conversion.adjusted_dym.atomic_numbers.to_vec(),
            vec![1, 1, 8]
        );
        let positions = conversion.adjusted_dym.coordinates.cartesian_positions();
        assert_eq!(positions.column(0).to_vec(), vec![0.0, 0.5, -2.0]);
        assert_eq!(
            conversion.adjusted_dym.force_constants[[0, 1, 0, 0]],
            data.force_constants[[1, 2, 0, 0]]
        );

        let input = dym_feff_inp_string(&conversion)?;
        assert!(input.starts_with(" * This feff9 input file was generated by dym2feffinp\n"));
        assert!(input.contains(" XANES        4.000\n"));
        assert!(input.contains("    0    1   H \n    1    8   O \n    2    1   H \n"));
        assert!(input.contains("    0.26459    0.00000    0.00000    2   H "));
        Ok(())
    }

    #[test]
    fn jfeff_mode_omits_only_the_header() -> Result<()> {
        let conversion = convert_dym_to_feff(
            &sample_dym()?,
            DymToFeffOptions {
                write_header: false,
                ..DymToFeffOptions::default()
            },
        )?;
        let input = dym_feff_inp_string(&conversion)?;
        assert!(input.starts_with("POTENTIALS\n"));
        assert!(input.contains("\nATOMS\n"));
        assert!(input.ends_with("END\n"));
        Ok(())
    }

    #[test]
    fn matches_production_h2o_center_two_xanes_reference() -> Result<()> {
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/src/DMDW/Test/H2O.g03.dym");
        if !source.is_file() {
            eprintln!("skipping local FEFF10 dym2feffinp reference; source checkout not found");
            return Ok(());
        }
        let data = super::super::read_dym(&source)
            .with_context(|| format!("reading {}", source.display()))?;
        let conversion = convert_dym_to_feff(
            &data,
            DymToFeffOptions {
                center_atom_index: 1,
                spectrum: DymSpectrum::Xanes,
                write_header: true,
            },
        )?;
        assert_eq!(
            dym_feff_inp_string(&conversion)?,
            H2O_CENTER_TWO_XANES_REFERENCE
        );
        assert_eq!(conversion.atom_order, vec![1, 0, 2]);
        assert_eq!(
            conversion.adjusted_dym.atomic_numbers.to_vec(),
            vec![1, 8, 1]
        );
        Ok(())
    }

    #[test]
    fn rejects_center_outside_atom_range() -> Result<()> {
        let error = convert_dym_to_feff(
            &sample_dym()?,
            DymToFeffOptions {
                center_atom_index: 3,
                ..DymToFeffOptions::default()
            },
        )
        .expect_err("out-of-range center must fail");
        assert!(matches!(
            error,
            IoError::InvalidDym {
                field: "center atom",
                ..
            }
        ));
        Ok(())
    }

    fn sample_dym() -> Result<DymData> {
        let mut force_constants = Array4::zeros((3, 3, 3, 3));
        for i_atom in 0..3 {
            for j_atom in 0..3 {
                for row in 0..3 {
                    for column in 0..3 {
                        force_constants[[i_atom, j_atom, row, column]] =
                            (1000 * i_atom + 100 * j_atom + 10 * row + column) as f64;
                    }
                }
            }
        }
        let coordinates =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.5, 0.0, 0.0])
                .context("building converter test coordinates")?;
        Ok(DymData {
            dym_type: 1,
            atomic_numbers: Array1::from_vec(vec![8, 1, 1]),
            atomic_masses: Array1::from_vec(vec![16.0, 1.0, 1.0]),
            coordinates: DymCoordinates::Cartesian(coordinates),
            force_constants,
            type2_metadata: None,
            dipole_derivatives: None,
        })
    }

    const H2O_CENTER_TWO_XANES_REFERENCE: &str = concat!(
        " * This feff9 input file was generated by dym2feffinp\n",
        "\n",
        " TITLE dymfile name:  Need to fix\n",
        " TITLE absorbing atom:   0\n",
        "\n",
        " EDGE      L3\n",
        " S02       1.0000\n",
        "\n",
        " *              pot    xsph     fms   paths  genfmt  ff2chi\n",
        " CONTROL          1       1       1       1       1       1\n",
        " PRINT            1       0       0       0       0       0\n",
        "\n",
        " *          ixc  [ Vr  Vi ]\n",
        " EXCHANGE     0\n",
        "\n",
        " *            r_scf  [ l_scf   n_scf   ca ]\n",
        " SCF          4.000\n",
        "\n",
        " *             kmax   [ delta_k  delta_e ]\n",
        " XANES        4.000\n",
        "\n",
        " *            r_fms     l_fms\n",
        " FMS          6.000\n",
        "\n",
        " *             emin    emax   eimag\n",
        " * LDOS       -30.000  20.000   0.100\n",
        "\n",
        " RPATH        0.100\n",
        "*EXAFS       20.000\n",
        "\n",
        " *        Temp  Debye_Temp  DW_Opt  dymfile  DMDW_Order  DMDW_Type  DMDW_Route\n",
        " DEBYE    300.0   315.0  5 feff.dym  16  0  1\n",
        "\n",
        "POTENTIALS\n",
        "    0    1   H \n",
        "    1    8   O \n",
        "    2    1   H \n",
        "\n",
        "ATOMS\n",
        "    0.00000    0.00000    0.00000    0   H   0.00000    0\n",
        "   -0.96141    0.12674   -0.00000    1   O   0.96972    1\n",
        "   -1.08815    1.08815    0.00000    2   H   1.53887    2\n",
        "END\n",
    );
}
