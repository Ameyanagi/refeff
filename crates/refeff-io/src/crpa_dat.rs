//! FEFF `crpa.dat` constrained random phase approximation output codec.
//!
//! The CRPA module writes a short text table containing the screened Hubbard
//! `U`, occupation `n`, and unscreened bare `U` values.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder};
use num_complex::Complex64;
use refeff_core::{
    ScreenCrpaProjectionWindow, ScreenCrpaResponseSliceInput, ScreenCrpaScreenedHubbardInput,
    ScreenEnergyStateInput, ScreenIntegratedResponseInput, screen_crpa_hubbard_summary,
    screen_crpa_orbital_density, screen_crpa_response_slice, screen_crpa_screened_hubbard_summary,
    screen_energy_integration_delta, screen_energy_state, screen_integrated_response,
};

use crate::crpa_input::CrpaInput;
use crate::error::{IoError, Result};
use crate::format::fortran_list_directed_g15_f64;
use crate::screen_dat::{
    ScreenFmsClusterGreenHandoff, ScreenPotentialKernelHandoff, WscrnDatData, wscrn_dat_string,
};

const CRPA_DAT_DEFAULT_HEADER: &str = "U, n, U_Bare";
const CRPA_DAT_ROW_WIDTH: usize = 3;

/// Parsed FEFF `crpa.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaDatData {
    /// Header and comment lines before the numeric CRPA result row.
    pub header_lines: Vec<String>,
    /// Screened Hubbard interaction `U`.
    pub hubbard_u: f64,
    /// Occupation number `n`.
    pub occupation: f64,
    /// Unscreened bare interaction `U_Bare`.
    pub bare_u: f64,
}

/// Lossless view of parsed FEFF `crpa.dat` output.
///
/// CRPA rows use compiler-dependent list-directed real formatting. The
/// semantic codec remains canonical while this wrapper retains validated
/// source text for exact fixture roundtrips.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaDatLosslessData {
    pub data: CrpaDatData,
    pub original_text: String,
}

/// Inputs for building FEFF `crpa.dat` from completed CRPA radial arrays.
#[derive(Debug, Clone, Copy)]
pub struct CrpaDatFromHubbardSummaryInput<'a> {
    /// Header/comment lines to preserve before the numeric row.
    pub header_lines: &'a [String],
    /// FEFF radial grid `ri`.
    pub radius_bohr: ArrayView1<'a, f64>,
    /// Screened response potential `wscrn`.
    pub screened_potential: ArrayView1<'a, f64>,
    /// Bare core-hole potential before response solve, FEFF `vbare`.
    pub bare_potential: ArrayView1<'a, f64>,
    /// Normalized total CRPA density, FEFF `totden_CRPA`.
    pub total_density: ArrayView1<'a, f64>,
    /// Selected orbital density row, FEFF `den_CRPA(:,ie)`.
    pub orbital_density: ArrayView1<'a, f64>,
    /// Loucks radial-grid step `dx`.
    pub radial_step: f64,
    /// Active radial prefix, FEFF `ilast`.
    pub active_count: usize,
}

/// Inputs for building FEFF `crpa.dat` from CRPA density and response arrays.
#[derive(Debug, Clone, Copy)]
pub struct CrpaDatFromScreenedHubbardInput<'a> {
    /// Header/comment lines to preserve before the numeric row.
    pub header_lines: &'a [String],
    /// FEFF radial grid `ri`.
    pub radius_bohr: ArrayView1<'a, f64>,
    /// Normalization source density, FEFF `totden_CRPA`.
    pub total_density: ArrayView1<'a, f64>,
    /// Selected orbital density row, FEFF `den_CRPA(:,ie)`.
    pub orbital_density: ArrayView1<'a, f64>,
    /// Screen/CRPA Coulomb response kernel, FEFF `Kmat`.
    pub response_kernel: ArrayView2<'a, f64>,
    /// Integrated response function, FEFF `chi0r`.
    pub susceptibility: ArrayView2<'a, Complex64>,
    /// Loucks radial-grid step `dx`.
    pub radial_step: f64,
    /// Active radial prefix, FEFF `ilast`.
    pub active_count: usize,
    /// Norman-radius prefix, FEFF `jnrm`.
    pub norman_count: usize,
    /// Optional CRPA projection window from `rnrm..rnrm*rcutin`.
    pub projection_window: Option<ScreenCrpaProjectionWindow>,
}

/// Inputs for building FEFF CRPA outputs from per-energy response slices.
#[derive(Debug, Clone, Copy)]
pub struct CrpaDatFromScreenedHubbardResponseSlicesInput<'a> {
    /// Header/comment lines to preserve before the numeric row.
    pub header_lines: &'a [String],
    /// FEFF radial grid `ri`.
    pub radius_bohr: ArrayView1<'a, f64>,
    /// Normalization source density, FEFF `totden_CRPA`.
    pub total_density: ArrayView1<'a, f64>,
    /// Selected orbital density row, FEFF `den_CRPA(:,ie)`.
    pub orbital_density: ArrayView1<'a, f64>,
    /// Screen/CRPA Coulomb response kernel, FEFF `Kmat`.
    pub response_kernel: ArrayView2<'a, f64>,
    /// Complex contour energy grid, FEFF `em`.
    pub energies: ArrayView1<'a, Complex64>,
    /// Per-energy upper-triangle response slices, FEFF `chi0re(:,:,ie)`.
    pub response_slices: ArrayView3<'a, Complex64>,
    /// Loucks radial-grid step `dx`.
    pub radial_step: f64,
    /// Active radial prefix, FEFF `ilast`.
    pub active_count: usize,
    /// Norman-radius prefix, FEFF `jnrm`.
    pub norman_count: usize,
    /// Optional CRPA projection window from `rnrm..rnrm*rcutin`.
    pub projection_window: Option<ScreenCrpaProjectionWindow>,
}

/// Inputs for assembling CRPA response/density state from SCREEN source handoffs.
#[derive(Debug, Clone, Copy)]
pub struct CrpaResponseAssemblyHandoffInput<'a> {
    /// Parsed `crpa.inp` controls.
    pub crpa: &'a CrpaInput,
    /// Radial bounds and response kernel from `screen.inp`/`pot.bin`.
    pub potential: &'a ScreenPotentialKernelHandoff,
    /// FMS trace table from `phase.bin` and FMS source/cache state.
    pub fms: &'a ScreenFmsClusterGreenHandoff,
    /// FEFF `eref(ie)` values used to compute `ck(ie)`.
    pub reference_energies_hartree: ArrayView1<'a, Complex64>,
    /// FEFF Fermi level `xmu`, used by CRPA to gate occupied response slices.
    pub fermi_level_hartree: f64,
    /// Regular radial solutions `pr(energy,r,l)`.
    pub regular_solutions: ArrayView3<'a, Complex64>,
    /// Irregular radial solutions `pn(energy,r,l)`.
    pub irregular_solutions: ArrayView3<'a, Complex64>,
    /// Header/comment lines to write before `crpa.dat`.
    pub crpa_header_lines: &'a [String],
    /// Header/comment lines to write before the CRPA `wscrn.dat` sidecar.
    pub wscrn_header_lines: &'a [String],
}

/// FEFF `crpa.dat` plus the final screened-density potential side product.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaDatFromHubbardSummary {
    /// Renderable FEFF `crpa.dat` payload.
    pub crpa: CrpaDatData,
    /// FEFF final `vch(i) = wscrn(i) * den_CRPA(i,ie)` radial vector.
    pub screened_density_potential: Array1<f64>,
}

/// FEFF `crpa.dat` plus radial intermediates from the solved CRPA response.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaDatFromScreenedHubbard {
    /// Renderable FEFF `crpa.dat` payload.
    pub crpa: CrpaDatData,
    /// Density after optional projection and FEFF normalization.
    pub normalized_density: Array1<f64>,
    /// Bare Coulomb potential before response solve, FEFF `vbare`.
    pub bare_potential: Array1<f64>,
    /// Screened response potential after the linear solve, FEFF `wscrn`.
    pub screened_potential: Array1<f64>,
    /// FEFF final `vch(i) = wscrn(i) * den_CRPA(i,ie)` radial vector.
    pub screened_density_potential: Array1<f64>,
}

/// FEFF CRPA text output plus the `wscrn.dat` sidecar.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaDatAndWscrnDatFromScreenedHubbard {
    /// Renderable FEFF `crpa.dat` payload and CRPA radial intermediates.
    pub crpa: CrpaDatFromScreenedHubbard,
    /// Renderable FEFF `wscrn.dat` payload.
    pub wscrn: WscrnDatData,
}

/// CRPA response handoff assembled from source radial, phase, and FMS state.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaResponseAssemblyHandoff {
    /// Complex photoelectron wave numbers as `ck(ie)`.
    pub wave_numbers: Array1<Complex64>,
    /// Selected-channel density rows, FEFF `den_CRPA(:,ie)`, as `(energy,r)`.
    pub orbital_density_slices: Array2<f64>,
    /// Integrated unnormalized selected-channel density, FEFF `totden_CRPA`.
    pub total_density: Array1<f64>,
    /// Final selected density row passed to the CRPA sidecar, FEFF `den_CRPA(:,ie)`.
    pub orbital_density: Array1<f64>,
    /// Per-energy upper-triangle response slices, FEFF `chi0re(:,:,ie)`.
    pub response_slices: Array3<Complex64>,
    /// Integrated symmetric susceptibility, FEFF `chi0r`.
    pub susceptibility: Array2<Complex64>,
    /// FEFF projection window derived from `rnrm` and `crpa.inp` `rcut`.
    pub projection_window: ScreenCrpaProjectionWindow,
    /// FEFF-compatible `crpa.dat` and CRPA `wscrn.dat` outputs.
    pub outputs: CrpaDatAndWscrnDatFromScreenedHubbard,
}

/// Render FEFF-compatible `crpa.dat` text.
pub fn crpa_dat_string(data: &CrpaDatData) -> Result<String> {
    validate_crpa_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    writeln!(
        out,
        "{}{}{}",
        fortran_list_directed_g15_f64(data.hubbard_u),
        fortran_list_directed_g15_f64(data.occupation),
        fortran_list_directed_g15_f64(data.bare_u)
    )?;
    Ok(out)
}

/// Build FEFF `crpa.dat` data from the CRPA Hubbard accumulation loop.
///
/// This ports the final scalar output handoff from `CRPA/chi_crpa.f90` after
/// the response equation has produced `wscrn`. Values are kept in the same
/// Hartree units FEFF writes to `crpa.dat`; the stdout-only eV conversion is not
/// applied here.
pub fn crpa_dat_from_hubbard_summary(
    input: CrpaDatFromHubbardSummaryInput<'_>,
) -> Result<CrpaDatFromHubbardSummary> {
    let summary = screen_crpa_hubbard_summary(
        &active_prefix(input.radius_bohr, input.active_count),
        &active_prefix(input.screened_potential, input.active_count),
        &active_prefix(input.bare_potential, input.active_count),
        &active_prefix(input.total_density, input.active_count),
        &active_prefix(input.orbital_density, input.active_count),
        input.radial_step,
        input.active_count,
    )
    .map_err(|source| invalid_crpa_dat("hubbard_summary", source.to_string()))?;

    let crpa = CrpaDatData {
        header_lines: if input.header_lines.is_empty() {
            vec![CRPA_DAT_DEFAULT_HEADER.to_string()]
        } else {
            input.header_lines.to_vec()
        },
        hubbard_u: summary.hubbard_u,
        occupation: summary.occupation,
        bare_u: summary.bare_u,
    };
    validate_crpa_dat(&crpa)?;

    Ok(CrpaDatFromHubbardSummary {
        crpa,
        screened_density_potential: summary.screened_density_potential,
    })
}

/// Build FEFF `crpa.dat` data from the solved CRPA response tail.
///
/// This starts before [`crpa_dat_from_hubbard_summary`]: it normalizes
/// `totden_CRPA`, builds the bare Coulomb potential, solves the screened
/// response equation, and then performs FEFF's final Hubbard accumulation.
pub fn crpa_dat_from_screened_hubbard(
    input: CrpaDatFromScreenedHubbardInput<'_>,
) -> Result<CrpaDatFromScreenedHubbard> {
    let radii = active_prefix(input.radius_bohr, input.active_count);
    let total_density = active_prefix(input.total_density, input.active_count);
    let orbital_density = active_prefix(input.orbital_density, input.active_count);
    let summary = screen_crpa_screened_hubbard_summary(ScreenCrpaScreenedHubbardInput {
        radii: &radii,
        total_density: &total_density,
        orbital_density: &orbital_density,
        response_kernel: input.response_kernel,
        susceptibility: input.susceptibility,
        dx: input.radial_step,
        active_count: input.active_count,
        norman_count: input.norman_count,
        projection_window: input.projection_window,
    })
    .map_err(|source| invalid_crpa_dat("screened_hubbard", source.to_string()))?;

    let crpa = CrpaDatData {
        header_lines: if input.header_lines.is_empty() {
            vec![CRPA_DAT_DEFAULT_HEADER.to_string()]
        } else {
            input.header_lines.to_vec()
        },
        hubbard_u: summary.hubbard_summary.hubbard_u,
        occupation: summary.hubbard_summary.occupation,
        bare_u: summary.hubbard_summary.bare_u,
    };
    validate_crpa_dat(&crpa)?;

    Ok(CrpaDatFromScreenedHubbard {
        crpa,
        normalized_density: summary.normalized_density,
        bare_potential: summary.bare_potential,
        screened_potential: summary.screened_potential,
        screened_density_potential: summary.hubbard_summary.screened_density_potential,
    })
}

/// Build FEFF `crpa.dat` plus CRPA's `wscrn.dat` sidecar from solved response inputs.
///
/// FEFF `CRPA/crpa.f90` writes `wscrn.dat` as `(r, wscrn, vch)` after the
/// Hubbard accumulation tail. At that point `vch` is the final
/// density-weighted potential `wscrn(i) * den_CRPA(i,ie)`, not the earlier bare
/// Coulomb potential.
pub fn crpa_dat_and_wscrn_dat_from_screened_hubbard(
    input: CrpaDatFromScreenedHubbardInput<'_>,
) -> Result<CrpaDatAndWscrnDatFromScreenedHubbard> {
    let crpa = crpa_dat_from_screened_hubbard(input)?;
    let wscrn = WscrnDatData {
        header_lines: Vec::new(),
        radius_bohr: Array1::from_iter(input.radius_bohr.iter().take(input.active_count).copied()),
        screened_potential: crpa.screened_potential.clone(),
        core_hole_potential: crpa.screened_density_potential.clone(),
    };
    wscrn_dat_string(&wscrn)?;

    Ok(CrpaDatAndWscrnDatFromScreenedHubbard { crpa, wscrn })
}

/// Build FEFF CRPA `crpa.dat` plus `wscrn.dat` from per-energy response slices.
///
/// This applies the shared SCREEN/CRPA contour accumulation to
/// `chi0re(:,:,ie)` before solving the screened Hubbard response and emitting
/// the CRPA `wscrn.dat` sidecar.
pub fn crpa_dat_and_wscrn_dat_from_screened_hubbard_response_slices(
    input: CrpaDatFromScreenedHubbardResponseSlicesInput<'_>,
) -> Result<CrpaDatAndWscrnDatFromScreenedHubbard> {
    let susceptibility = screen_integrated_response(ScreenIntegratedResponseInput {
        energies: input.energies,
        response_slices: input.response_slices,
        active_count: input.active_count,
    })
    .map_err(|source| invalid_crpa_dat("response_slices", source.to_string()))?;

    crpa_dat_and_wscrn_dat_from_screened_hubbard(CrpaDatFromScreenedHubbardInput {
        header_lines: input.header_lines,
        radius_bohr: input.radius_bohr,
        total_density: input.total_density,
        orbital_density: input.orbital_density,
        response_kernel: input.response_kernel,
        susceptibility: susceptibility.view(),
        radial_step: input.radial_step,
        active_count: input.active_count,
        norman_count: input.norman_count,
        projection_window: input.projection_window,
    })
}

/// Assemble FEFF CRPA response/density state and render `crpa.dat` plus `wscrn.dat`.
///
/// This is the source-backed bridge after SCREEN-style radial solutions and FMS
/// traces are available. It ports the CRPA-specific part of `chi_crpa.f90`:
/// derive `ck(ie)`, build selected-channel `den_CRPA`, integrate
/// `totden_CRPA` with FEFF's contour step convention, gate `chi0re` response
/// slices to occupied contour points, then feed the shared screened-Hubbard
/// output adapter.
pub fn crpa_response_assembly_handoff(
    input: CrpaResponseAssemblyHandoffInput<'_>,
) -> Result<CrpaResponseAssemblyHandoff> {
    validate_crpa_response_assembly_input(&input)?;

    let energy_count = input.fms.cluster_greens.nrows();
    let angular_count = input.fms.cluster_greens.ncols();
    let active_count = input.potential.bounds.active_count;
    let crpa_angular_momentum = usize::try_from(input.crpa.l).map_err(|_| {
        invalid_crpa_dat(
            "l_crpa",
            format!(
                "CRPA angular momentum must be non-negative, got {}",
                input.crpa.l
            ),
        )
    })?;
    if crpa_angular_momentum >= angular_count {
        return Err(invalid_crpa_dat(
            "l_crpa",
            format!(
                "CRPA angular momentum {crpa_angular_momentum} requires at least {} angular channels",
                crpa_angular_momentum + 1
            ),
        ));
    }

    let projection_window = crpa_projection_window(input.crpa, input.potential.norman_radius_bohr)?;
    let radii = input.potential.radius_bohr.as_slice().ok_or_else(|| {
        invalid_crpa_dat("response_assembly", "CRPA radial grid is not contiguous")
    })?;

    let mut wave_numbers = Array1::zeros(energy_count);
    for energy_index in 0..energy_count {
        let state = screen_energy_state(ScreenEnergyStateInput {
            energy: input.fms.energies_hartree[energy_index],
            reference_energy: input.reference_energies_hartree[energy_index],
            muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
            exchange_selector: input.potential.exchange_selector,
        })
        .map_err(|source| invalid_crpa_dat("energy_state", source.to_string()))?;
        wave_numbers[energy_index] = state.wave_number;
    }

    let mut orbital_density_slices = Array2::zeros((energy_count, active_count).f());
    let mut total_density = Array1::<f64>::zeros(active_count);
    let mut orbital_density = Array1::<f64>::zeros(active_count);
    let mut response_slices =
        Array3::<Complex64>::zeros((energy_count, active_count, active_count).f());

    for energy_index in 0..energy_count {
        let regular_at_energy = input.regular_solutions.index_axis(Axis(0), energy_index);
        let irregular_at_energy = input.irregular_solutions.index_axis(Axis(0), energy_index);
        let selected_density = screen_crpa_orbital_density(
            regular_at_energy.column(crpa_angular_momentum),
            irregular_at_energy.column(crpa_angular_momentum),
            input.fms.cluster_greens[(energy_index, crpa_angular_momentum)],
            wave_numbers[energy_index],
            crpa_angular_momentum,
            active_count,
        )
        .map_err(|source| invalid_crpa_dat("orbital_density", source.to_string()))?;
        for radial_index in 0..active_count {
            let value = selected_density[radial_index];
            orbital_density_slices[(energy_index, radial_index)] = value;
            orbital_density[radial_index] = value;
        }

        if energy_index > 0 {
            let delta =
                screen_energy_integration_delta(input.fms.energies_hartree.view(), energy_index)
                    .map_err(|source| {
                        invalid_crpa_dat("density_integration", source.to_string())
                    })?;
            for radial_index in 0..active_count {
                total_density[radial_index] += selected_density[radial_index] * delta.re;
            }
        }

        if input.fms.energies_hartree[energy_index].re > input.fermi_level_hartree {
            continue;
        }

        for angular_momentum in 0..angular_count {
            let channel = screen_crpa_response_slice(ScreenCrpaResponseSliceInput {
                radii,
                regular_solution: regular_at_energy.column(angular_momentum),
                irregular_solution: irregular_at_energy.column(angular_momentum),
                cluster_green: input.fms.cluster_greens[(energy_index, angular_momentum)],
                wave_number: wave_numbers[energy_index],
                dx: input.potential.radial_step,
                angular_momentum,
                crpa_angular_momentum,
                projection_window: Some(projection_window),
                active_count,
            })
            .map_err(|source| invalid_crpa_dat("response_slice", source.to_string()))?;
            for row in 0..active_count {
                for column in row..active_count {
                    response_slices[(energy_index, row, column)] += channel[(row, column)];
                }
            }
        }
    }

    let susceptibility = screen_integrated_response(ScreenIntegratedResponseInput {
        energies: input.fms.energies_hartree.view(),
        response_slices: response_slices.view(),
        active_count,
    })
    .map_err(|source| invalid_crpa_dat("response_slices", source.to_string()))?;

    let mut outputs =
        crpa_dat_and_wscrn_dat_from_screened_hubbard(CrpaDatFromScreenedHubbardInput {
            header_lines: input.crpa_header_lines,
            radius_bohr: input.potential.radius_bohr.view(),
            total_density: total_density.view(),
            orbital_density: orbital_density.view(),
            response_kernel: input.potential.response_kernel.view(),
            susceptibility: susceptibility.view(),
            radial_step: input.potential.radial_step,
            active_count,
            norman_count: input.potential.bounds.norman_index_1based,
            projection_window: Some(projection_window),
        })?;
    outputs.wscrn.header_lines = input.wscrn_header_lines.to_vec();
    wscrn_dat_string(&outputs.wscrn)?;

    Ok(CrpaResponseAssemblyHandoff {
        wave_numbers,
        orbital_density_slices,
        total_density,
        orbital_density,
        response_slices,
        susceptibility,
        projection_window,
        outputs,
    })
}

/// Parse FEFF `crpa.dat` text.
pub fn parse_crpa_dat(text: &str) -> Result<CrpaDatData> {
    let mut header_lines = Vec::new();
    let mut row = None;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if row.is_some() {
                return Err(invalid_crpa_dat(
                    "rows",
                    "crpa.dat must contain exactly one numeric row",
                ));
            }
            if tokens.len() != CRPA_DAT_ROW_WIDTH {
                return Err(IoError::CrpaDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: CRPA_DAT_ROW_WIDTH,
                });
            }
            row = Some((
                parse_f64(line_number, "U", tokens[0])?,
                parse_f64(line_number, "n", tokens[1])?,
                parse_f64(line_number, "U_Bare", tokens[2])?,
            ));
        } else {
            header_lines.push(line.to_string());
        }
    }

    let (hubbard_u, occupation, bare_u) = row.ok_or(IoError::CrpaDatMissing { field: "row" })?;
    let data = CrpaDatData {
        header_lines,
        hubbard_u,
        occupation,
        bare_u,
    };
    validate_crpa_dat(&data)?;
    Ok(data)
}

/// Parse and retain exact validated `crpa.dat` source text.
pub fn parse_crpa_dat_lossless(text: &str) -> Result<CrpaDatLosslessData> {
    Ok(CrpaDatLosslessData {
        data: parse_crpa_dat(text)?,
        original_text: text.to_string(),
    })
}

/// Render a lossless `crpa.dat` view.
pub fn crpa_dat_lossless_string(data: &CrpaDatLosslessData) -> Result<String> {
    if parse_crpa_dat(&data.original_text)? == data.data {
        Ok(data.original_text.clone())
    } else {
        crpa_dat_string(&data.data)
    }
}

/// Write FEFF `crpa.dat` text to a file.
pub fn write_crpa_dat(path: impl AsRef<Path>, data: &CrpaDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, crpa_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `crpa.dat` text from a file.
pub fn read_crpa_dat(path: impl AsRef<Path>) -> Result<CrpaDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_crpa_dat(&text)
}

fn active_prefix(values: ArrayView1<'_, f64>, active_count: usize) -> Vec<f64> {
    values.iter().take(active_count).copied().collect()
}

fn crpa_projection_window(
    input: &CrpaInput,
    norman_radius_bohr: f64,
) -> Result<ScreenCrpaProjectionWindow> {
    validate_finite("rcut", input.rcut)?;
    validate_finite("rnrm", norman_radius_bohr)?;
    let outer_radius = norman_radius_bohr * input.rcut;
    validate_finite("projection_outer_radius", outer_radius)?;
    if outer_radius <= norman_radius_bohr {
        return Err(invalid_crpa_dat(
            "rcut",
            format!(
                "CRPA cutoff multiplier must place rcut beyond rnrm: rnrm={norman_radius_bohr}, rcut={outer_radius}"
            ),
        ));
    }
    Ok(ScreenCrpaProjectionWindow {
        inner_radius: norman_radius_bohr,
        outer_radius,
    })
}

fn validate_crpa_response_assembly_input(
    input: &CrpaResponseAssemblyHandoffInput<'_>,
) -> Result<()> {
    let energy_count = input.fms.cluster_greens.nrows();
    let angular_count = input.fms.cluster_greens.ncols();
    let active_count = input.potential.bounds.active_count;
    if energy_count == 0 {
        return Err(invalid_crpa_dat(
            "response_assembly",
            "CRPA FMS handoff has no energy rows",
        ));
    }
    if angular_count == 0 {
        return Err(invalid_crpa_dat(
            "response_assembly",
            "CRPA FMS handoff has no angular columns",
        ));
    }
    if input.fms.energies_hartree.len() < energy_count {
        return Err(invalid_crpa_dat(
            "response_assembly",
            format!(
                "CRPA energy grid has {} rows, expected at least {energy_count}",
                input.fms.energies_hartree.len()
            ),
        ));
    }
    if input.reference_energies_hartree.len() < energy_count {
        return Err(invalid_crpa_dat(
            "response_assembly",
            format!(
                "CRPA reference-energy grid has {} rows, expected at least {energy_count}",
                input.reference_energies_hartree.len()
            ),
        ));
    }
    validate_shape3(
        "regular_solutions",
        input.regular_solutions.dim(),
        energy_count,
        active_count,
        angular_count,
    )?;
    validate_shape3(
        "irregular_solutions",
        input.irregular_solutions.dim(),
        energy_count,
        active_count,
        angular_count,
    )?;
    validate_finite("fermi_level", input.fermi_level_hartree)?;
    Ok(())
}

fn validate_shape3(
    name: &'static str,
    actual: (usize, usize, usize),
    energy_count: usize,
    active_count: usize,
    angular_count: usize,
) -> Result<()> {
    if actual.0 < energy_count || actual.1 < active_count || actual.2 < angular_count {
        return Err(invalid_crpa_dat(
            name,
            format!(
                "{name} shape {:?} is smaller than required ({energy_count}, {active_count}, {angular_count})",
                actual
            ),
        ));
    }
    Ok(())
}

fn validate_crpa_dat(data: &CrpaDatData) -> Result<()> {
    validate_finite("U", data.hubbard_u)?;
    validate_finite("n", data.occupation)?;
    validate_finite("U_Bare", data.bare_u)?;
    Ok(())
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::CrpaDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_crpa_dat(field, "value must be finite"))
    }
}

fn invalid_crpa_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidCrpaDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, array};
    use num_complex::Complex64;
    use refeff_core::{
        ScreenRadialBounds, screen_crpa_orbital_density, screen_energy_integration_delta,
    };

    use crate::screen_dat::{ScreenFmsClusterGreenHandoff, ScreenPotentialKernelHandoff};

    use super::*;

    #[test]
    fn parses_feff_crpa_reference_values() -> Result<()> {
        let data = parse_crpa_dat(CRPA_DAT)?;
        assert_eq!(data.header_lines, vec!["U, n, U_Bare"]);
        assert_eq!(data.hubbard_u, 0.197879035252010);
        assert_eq!(data.occupation, 1.0);
        assert_eq!(data.bare_u, 0.694283422651496);
        Ok(())
    }

    #[test]
    fn roundtrips_crpa_text() -> Result<()> {
        let data = parse_crpa_dat(CRPA_DAT)?;
        let rendered = crpa_dat_string(&data)?;
        assert_eq!(rendered, CRPA_DAT);
        assert_eq!(parse_crpa_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn lossless_crpa_preserves_list_directed_precision() -> Result<()> {
        let text = concat!(
            "U, n, U_Bare\n",
            "  0.19787689193580690        1.0000000000000000       0.69428098115988179     \n",
        );
        let data = parse_crpa_dat_lossless(text)?;
        assert_eq!(crpa_dat_lossless_string(&data)?, text);
        Ok(())
    }

    #[test]
    fn builds_crpa_dat_from_hubbard_summary() -> Result<()> {
        let radii = array![1.0, 2.0, 3.0, 4.0];
        let screened = array![0.5, 1.0, 1.5, 99.0];
        let bare = array![2.0, 1.0, 0.5, 99.0];
        let total_density = array![5.0 / 7.0, 10.0 / 7.0, 15.0 / 7.0, 99.0];
        let orbital_density = array![0.2, 0.3, 0.4, 99.0];

        let handoff = crpa_dat_from_hubbard_summary(CrpaDatFromHubbardSummaryInput {
            header_lines: &[],
            radius_bohr: radii.view(),
            screened_potential: screened.view(),
            bare_potential: bare.view(),
            total_density: total_density.view(),
            orbital_density: orbital_density.view(),
            radial_step: 0.1,
            active_count: 3,
        })?;

        assert_eq!(
            handoff.crpa.header_lines,
            vec![CRPA_DAT_DEFAULT_HEADER.to_string()]
        );
        assert_close(handoff.crpa.hubbard_u, 9.0 / 7.0, 1.0e-14);
        assert_close(handoff.crpa.occupation, 1.0, 1.0e-14);
        assert_close(handoff.crpa.bare_u, 0.75, 1.0e-14);
        assert_close(handoff.screened_density_potential[0], 0.1, 1.0e-14);
        assert_close(handoff.screened_density_potential[1], 0.3, 1.0e-14);
        assert_close(handoff.screened_density_potential[2], 0.6, 1.0e-14);

        let rendered = crpa_dat_string(&handoff.crpa)?;
        let parsed = parse_crpa_dat(&rendered)?;
        assert_eq!(parsed.header_lines, handoff.crpa.header_lines);
        assert_close(parsed.hubbard_u, handoff.crpa.hubbard_u, 1.0e-14);
        assert_close(parsed.occupation, handoff.crpa.occupation, 1.0e-14);
        assert_close(parsed.bare_u, handoff.crpa.bare_u, 1.0e-14);
        Ok(())
    }

    #[test]
    fn builds_crpa_dat_from_screened_hubbard_response() -> Result<()> {
        let radii = array![1.0, 2.0];
        let total_density = array![2.0, 4.0];
        let orbital_density = array![0.1, 0.2];
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)],
            [Complex64::new(3.0, 0.3), Complex64::new(4.0, 0.05)]
        ];

        let handoff = crpa_dat_from_screened_hubbard(CrpaDatFromScreenedHubbardInput {
            header_lines: &[],
            radius_bohr: radii.view(),
            total_density: total_density.view(),
            orbital_density: orbital_density.view(),
            response_kernel: kernel.view(),
            susceptibility: susceptibility.view(),
            radial_step: 0.1,
            active_count: 2,
            norman_count: 2,
            projection_window: None,
        })?;

        assert_eq!(
            handoff.crpa.header_lines,
            vec![CRPA_DAT_DEFAULT_HEADER.to_string()]
        );
        assert_close(handoff.normalized_density[0], 2.0, 1.0e-14);
        assert_close(handoff.normalized_density[1], 4.0, 1.0e-14);
        assert_close(handoff.bare_potential[0], 0.6, 1.0e-14);
        assert_close(handoff.bare_potential[1], 0.5, 1.0e-14);
        assert_close(handoff.screened_potential[0], 578.0 / 323.0, 1.0e-14);
        assert_close(handoff.screened_potential[1], 428.0 / 323.0, 1.0e-14);
        assert_close(handoff.screened_density_potential[0], 57.8 / 323.0, 1.0e-14);
        assert_close(handoff.crpa.hubbard_u, 458.0 / 323.0, 1.0e-14);
        assert_close(handoff.crpa.occupation, 1.0, 1.0e-14);
        assert_close(handoff.crpa.bare_u, 0.52, 1.0e-14);

        let rendered = crpa_dat_string(&handoff.crpa)?;
        let parsed = parse_crpa_dat(&rendered)?;
        assert_eq!(parsed.header_lines, handoff.crpa.header_lines);
        assert_close(parsed.hubbard_u, handoff.crpa.hubbard_u, 1.0e-14);
        assert_close(parsed.occupation, handoff.crpa.occupation, 1.0e-14);
        assert_close(parsed.bare_u, handoff.crpa.bare_u, 1.0e-14);
        Ok(())
    }

    #[test]
    fn builds_crpa_dat_and_wscrn_dat_from_screened_hubbard_response() -> Result<()> {
        let radii = array![1.0, 2.0];
        let total_density = array![2.0, 4.0];
        let orbital_density = array![0.1, 0.2];
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)],
            [Complex64::new(3.0, 0.3), Complex64::new(4.0, 0.05)]
        ];

        let outputs =
            crpa_dat_and_wscrn_dat_from_screened_hubbard(CrpaDatFromScreenedHubbardInput {
                header_lines: &[],
                radius_bohr: radii.view(),
                total_density: total_density.view(),
                orbital_density: orbital_density.view(),
                response_kernel: kernel.view(),
                susceptibility: susceptibility.view(),
                radial_step: 0.1,
                active_count: 2,
                norman_count: 2,
                projection_window: None,
            })?;

        assert_close(outputs.crpa.crpa.hubbard_u, 458.0 / 323.0, 1.0e-14);
        assert_close(outputs.crpa.bare_potential[0], 0.6, 1.0e-14);
        assert_eq!(outputs.wscrn.header_lines, Vec::<String>::new());
        assert_close(outputs.wscrn.radius_bohr[0], 1.0, 1.0e-14);
        assert_close(outputs.wscrn.radius_bohr[1], 2.0, 1.0e-14);
        assert_close(outputs.wscrn.screened_potential[0], 578.0 / 323.0, 1.0e-14);
        assert_close(outputs.wscrn.core_hole_potential[0], 57.8 / 323.0, 1.0e-14);
        assert_close(outputs.wscrn.core_hole_potential[1], 85.6 / 323.0, 1.0e-14);

        let rendered = wscrn_dat_string(&outputs.wscrn)?;
        let parsed = crate::screen_dat::parse_wscrn_dat(&rendered)?;
        assert!(parsed.header_lines.is_empty());
        assert_close(parsed.core_hole_potential[0], 57.8 / 323.0, 1.0e-10);
        Ok(())
    }

    #[test]
    fn builds_crpa_dat_and_wscrn_dat_from_response_slices() -> Result<()> {
        let radii = array![1.0, 2.0];
        let total_density = array![2.0, 4.0];
        let orbital_density = array![0.1, 0.2];
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)],
            [Complex64::new(2.0, 0.2), Complex64::new(4.0, 0.05)]
        ];
        let energies = array![
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0)
        ];
        let response_slices = Array1::from_iter(susceptibility.iter().copied())
            .into_shape_with_order((1, 2, 2))
            .expect("susceptibility shape")
            .broadcast((3, 2, 2))
            .expect("broadcast response slices")
            .mapv(|value| value / 4.0);

        let expected =
            crpa_dat_and_wscrn_dat_from_screened_hubbard(CrpaDatFromScreenedHubbardInput {
                header_lines: &[],
                radius_bohr: radii.view(),
                total_density: total_density.view(),
                orbital_density: orbital_density.view(),
                response_kernel: kernel.view(),
                susceptibility: susceptibility.view(),
                radial_step: 0.1,
                active_count: 2,
                norman_count: 2,
                projection_window: None,
            })?;
        let actual = crpa_dat_and_wscrn_dat_from_screened_hubbard_response_slices(
            CrpaDatFromScreenedHubbardResponseSlicesInput {
                header_lines: &[],
                radius_bohr: radii.view(),
                total_density: total_density.view(),
                orbital_density: orbital_density.view(),
                response_kernel: kernel.view(),
                energies: energies.view(),
                response_slices: response_slices.view(),
                radial_step: 0.1,
                active_count: 2,
                norman_count: 2,
                projection_window: None,
            },
        )?;

        assert_close(
            actual.crpa.crpa.hubbard_u,
            expected.crpa.crpa.hubbard_u,
            1.0e-14,
        );
        assert_close(
            actual.crpa.screened_potential[0],
            expected.crpa.screened_potential[0],
            1.0e-14,
        );
        assert_close(
            actual.wscrn.core_hole_potential[0],
            expected.wscrn.core_hole_potential[0],
            1.0e-14,
        );
        assert_eq!(actual.wscrn.radius_bohr, expected.wscrn.radius_bohr);
        Ok(())
    }

    #[test]
    fn assembles_crpa_response_handoff_from_source_state() -> Result<()> {
        let potential = sample_crpa_potential_handoff();
        let energies = array![
            Complex64::new(0.2, 0.0),
            Complex64::new(0.4, 0.0),
            Complex64::new(0.8, 0.0)
        ];
        let fms = ScreenFmsClusterGreenHandoff {
            energies_hartree: energies.clone(),
            cluster_greens: Array2::from_shape_fn((3, 2), |(energy, angular)| {
                Complex64::new(0.01 * (energy + angular + 1) as f64, 0.002 * angular as f64)
            }),
            potential_index: 0,
            spin_index: 0,
        };
        let regular = Array3::from_shape_fn((3, 2, 2), |(energy, radial, angular)| {
            Complex64::new(
                0.4 + 0.05 * energy as f64 + 0.03 * radial as f64 + 0.02 * angular as f64,
                0.0,
            )
        });
        let irregular = Array3::from_shape_fn((3, 2, 2), |(energy, radial, angular)| {
            Complex64::new(
                0.01 * angular as f64,
                0.08 + 0.01 * energy as f64 + 0.02 * radial as f64,
            )
        });
        let crpa = CrpaInput {
            enabled: true,
            rcut: 3.0,
            l: 1,
        };

        let handoff = crpa_response_assembly_handoff(CrpaResponseAssemblyHandoffInput {
            crpa: &crpa,
            potential: &potential,
            fms: &fms,
            reference_energies_hartree: Array1::zeros(3).view(),
            fermi_level_hartree: 0.6,
            regular_solutions: regular.view(),
            irregular_solutions: irregular.view(),
            crpa_header_lines: &[],
            wscrn_header_lines: &["# crpa wscrn".to_string()],
        })?;

        let density_1 = screen_crpa_orbital_density(
            regular.index_axis(Axis(0), 1).column(1),
            irregular.index_axis(Axis(0), 1).column(1),
            fms.cluster_greens[(1, 1)],
            handoff.wave_numbers[1],
            1,
            2,
        )
        .expect("energy-1 CRPA density");
        let density_2 = screen_crpa_orbital_density(
            regular.index_axis(Axis(0), 2).column(1),
            irregular.index_axis(Axis(0), 2).column(1),
            fms.cluster_greens[(2, 1)],
            handoff.wave_numbers[2],
            1,
            2,
        )
        .expect("energy-2 CRPA density");
        let delta_1 =
            screen_energy_integration_delta(energies.view(), 1).expect("energy-1 integration step");
        let delta_2 =
            screen_energy_integration_delta(energies.view(), 2).expect("energy-2 integration step");

        for radial in 0..2 {
            assert_close(
                handoff.total_density[radial],
                density_1[radial] * delta_1.re + density_2[radial] * delta_2.re,
                1.0e-14,
            );
            assert_close(handoff.orbital_density[radial], density_2[radial], 1.0e-14);
        }
        assert_eq!(
            handoff.projection_window,
            ScreenCrpaProjectionWindow {
                inner_radius: 1.0,
                outer_radius: 3.0,
            }
        );
        assert!(
            handoff
                .response_slices
                .index_axis(Axis(0), 0)
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        assert!(
            handoff
                .response_slices
                .index_axis(Axis(0), 2)
                .iter()
                .all(|value| value.norm() == 0.0)
        );
        assert!(handoff.outputs.crpa.crpa.hubbard_u.is_finite());
        assert!(handoff.outputs.crpa.crpa.bare_u.is_finite());
        assert_eq!(handoff.outputs.wscrn.row_count(), 2);
        assert_eq!(
            handoff.outputs.wscrn.header_lines,
            vec!["# crpa wscrn".to_string()]
        );
        Ok(())
    }

    #[test]
    fn crpa_dat_from_hubbard_summary_rejects_short_inputs() {
        let error = crpa_dat_from_hubbard_summary(CrpaDatFromHubbardSummaryInput {
            header_lines: &[],
            radius_bohr: array![1.0, 2.0].view(),
            screened_potential: array![0.5, 1.0].view(),
            bare_potential: array![2.0, 1.0].view(),
            total_density: array![0.5, 0.5].view(),
            orbital_density: array![0.2].view(),
            radial_step: 0.1,
            active_count: 2,
        })
        .unwrap_err();

        assert!(error.to_string().contains("active radial count 2"));
    }

    #[test]
    fn crpa_dat_from_screened_hubbard_rejects_short_response_inputs() {
        let error = crpa_dat_from_screened_hubbard(CrpaDatFromScreenedHubbardInput {
            header_lines: &[],
            radius_bohr: array![1.0, 2.0].view(),
            total_density: array![2.0, 4.0].view(),
            orbital_density: array![0.1, 0.2].view(),
            response_kernel: array![[2.0]].view(),
            susceptibility: array![[Complex64::new(1.0, 0.1)]].view(),
            radial_step: 0.1,
            active_count: 2,
            norman_count: 2,
            projection_window: None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("screened_hubbard"));
        assert!(error.to_string().contains("kernel"));
    }

    #[test]
    fn rejects_bad_crpa_inputs() {
        assert!(parse_crpa_dat("U, n, U_Bare\n").is_err());
        assert!(parse_crpa_dat("1 2\n").is_err());
        assert!(parse_crpa_dat("1 NaN 2\n").is_err());
        assert!(parse_crpa_dat("1 2 3\n4 5 6\n").is_err());

        let bad = CrpaDatData {
            header_lines: Vec::new(),
            hubbard_u: f64::NAN,
            occupation: 1.0,
            bare_u: 2.0,
        };
        assert!(crpa_dat_string(&bad).is_err());
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }

    fn sample_crpa_potential_handoff() -> ScreenPotentialKernelHandoff {
        ScreenPotentialKernelHandoff {
            radius_bohr: array![1.0, 2.0],
            bounds: ScreenRadialBounds {
                muffin_tin_index_1based: 1,
                muffin_tin_next_index_1based: 2,
                norman_index_1based: 2,
                active_count: 2,
            },
            local_kernel: None,
            response_kernel: array![[0.20, 0.05], [0.05, 0.10]],
            core_large_component: Array1::zeros(2),
            core_small_component: Array1::zeros(2),
            potential_index: 0,
            muffin_tin_radius_bohr: 1.0,
            norman_radius_bohr: 1.0,
            exchange_selector: 0,
            radial_step: 0.1,
        }
    }

    const CRPA_DAT: &str = concat!(
        "U, n, U_Bare\n",
        "  0.197879035252010        1.00000000000000       0.694283422651496     \n",
    );
}
