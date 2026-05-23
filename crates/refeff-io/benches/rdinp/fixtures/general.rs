use super::*;

pub(crate) fn bench_input() -> String {
    let local_cu =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples/EXAFS/Cu/feff.inp");
    std::fs::read_to_string(local_cu).unwrap_or_else(|_| FALLBACK_INPUT.to_string())
}

pub(crate) fn cif_bench_text() -> String {
    let mut text = String::from(
        r#"
data_primary
_cell_length_a 4.2000
_cell_length_b 4.4000
_cell_length_c 5.1000
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
_publ_section_comment
;
"#,
    );
    text.push_str(&"x".repeat(9000));
    text.push_str(
        r#"
;
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
"#,
    );
    for index in 0..128 {
        let x = (index % 8) as f64 / 8.0;
        let y = ((index / 8) % 4) as f64 / 4.0;
        let z = (index / 32) as f64 / 4.0;
        text.push_str(&format!("C{index} {x:.6} {y:.6} {z:.6}\n"));
    }
    text.push_str(
        r#"
data_ignored
_cell_length_a 20.0
_cell_length_b 20.0
_cell_length_c 20.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
O1 0.5 0.5 0.5
"#,
    );
    text
}

pub(crate) struct PotOutputBenchState {
    muffin_tin_indices: Vec<usize>,
    norman_indices: Vec<usize>,
    titles: Vec<String>,
    electron_density: Array2<f64>,
    free_density: Array2<f64>,
    overlapped_coulomb: Array2<f64>,
    free_coulomb: Array2<f64>,
    total_potential: Array2<f64>,
}

impl PotOutputBenchState {
    pub(crate) fn new() -> Self {
        let rows = 251;
        let potentials = 6;
        Self {
            muffin_tin_indices: (0..potentials).map(|potential| 12 + potential).collect(),
            norman_indices: (0..potentials)
                .map(|potential| 40 + 2 * potential)
                .collect(),
            titles: vec![
                "Cu crystal".to_string(),
                "Gam_ch=1.000E+00 H-L exch Vi=0.000E+00 Vr=0.000E+00".to_string(),
            ],
            electron_density: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                0.035 * (row + 1) as f64 + 0.125 * potential as f64
            }),
            free_density: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                0.015 * (row + 1) as f64 + 0.25 * potential as f64
            }),
            overlapped_coulomb: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -1.2 * (potential + 1) as f64 - 0.02 * (row + 1) as f64
            }),
            free_coulomb: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
            }),
            total_potential: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -0.45 * (potential + 1) as f64 + 0.003 * (row + 1) as f64
            }),
        }
    }

    pub(crate) fn input(&self) -> PotentialDatSetInput<'_> {
        PotentialDatSetInput {
            highest_potential_index: self.muffin_tin_indices.len() - 1,
            muffin_tin_indices: &self.muffin_tin_indices,
            norman_indices: &self.norman_indices,
            titles: &self.titles,
            electron_density: self.electron_density.view(),
            free_density: self.free_density.view(),
            overlapped_coulomb: self.overlapped_coulomb.view(),
            free_coulomb: self.free_coulomb.view(),
            total_potential: self.total_potential.view(),
        }
    }
}

pub(crate) fn apot_bin_wpot_bench_data(pot: &PotBinData) -> ApotBinData {
    let rows = POT_BIN_RADIAL_POINTS;
    let columns = pot.potential_count() + 1;
    ApotBinData {
        sections: vec![
            apot_bin_wpot_matrix_section(
                8,
                "rho(r,0:nphx+1) - atomic density for each unique potential",
                Array2::from_shape_fn((rows, columns), |(row, potential)| {
                    0.015 * (row + 1) as f64 + 0.25 * potential as f64
                }),
            ),
            apot_bin_wpot_matrix_section(
                11,
                "vcoul(r,nph) - coulomb potential for each unique potential.",
                Array2::from_shape_fn((rows, columns), |(row, potential)| {
                    -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
                }),
            ),
        ],
    }
}

pub(crate) fn apot_bin_wpot_matrix_section(
    section_number: usize,
    header: &str,
    values: Array2<f64>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: vec![header.to_string()],
        header_texts: vec![format!(" {header}")],
        column_labels: vec![],
        column_label_text: None,
        payload: ApotBinPayload::Matrix(ApotBinMatrix {
            value_type: ApotBinType::Double,
            values: ApotBinMatrixValues::Real(values),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    }
}

pub(crate) fn mtdp_bench_data() -> MtdpData {
    let radial_count = 251;
    let atom_count = 12;
    let empty_count = 4;
    MtdpData {
        radial_count,
        atomic_numbers: Array1::from_shape_fn(
            atom_count,
            |atom| if atom % 3 == 0 { 29 } else { 8 },
        ),
        atom_coordinates: Array2::from_shape_fn((atom_count, 3), |(atom, axis)| {
            atom as f64 * 0.25 + axis as f64 * 0.125
        }),
        atom_radii: Array1::from_shape_fn(atom_count, |atom| 0.4 + atom as f64 * 0.01),
        atom_radius_indices: Array1::from_shape_fn(atom_count, |atom| 40 + atom),
        atom_density: Array2::from_shape_fn((radial_count, atom_count), |(radial, atom)| {
            0.001 * (radial + 1) as f64 + 0.0001 * atom as f64
        }),
        atom_potential: Array2::from_shape_fn((radial_count, atom_count), |(radial, atom)| {
            -1.0 - 0.01 * radial as f64 - 0.05 * atom as f64
        }),
        empty_sphere_coordinates: Array2::from_shape_fn((empty_count, 3), |(sphere, axis)| {
            sphere as f64 * 0.5 + axis as f64 * 0.2
        }),
        empty_sphere_radii: Array1::from_shape_fn(empty_count, |sphere| 0.2 + sphere as f64 * 0.02),
        empty_sphere_radius_indices: Array1::from_shape_fn(empty_count, |sphere| 25 + sphere),
        empty_sphere_density: Array2::from_shape_fn(
            (radial_count, empty_count),
            |(radial, sphere)| 0.0005 * (radial + 1) as f64 + 0.0002 * sphere as f64,
        ),
        empty_sphere_potential: Array2::from_shape_fn(
            (radial_count, empty_count),
            |(radial, sphere)| -0.5 - 0.006 * radial as f64 - 0.025 * sphere as f64,
        ),
        interstitial_potential: -0.75,
        homo_energy: -0.12,
        lumo_energy: 0.34,
    }
}
