#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

//! Core numerical types for the FEFF10 Rust port.
//!
//! `ndarray` is the primary storage and view API. Helpers in this crate create
//! arrays with FEFF-friendly Fortran-order layout where the original algorithms
//! and file formats depend on column-major traversal.

use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::Complex64;

pub mod angular;
pub mod atomic;
pub mod band;
pub mod bessel;
pub mod compton;
pub mod configuration;
pub mod configuration_defaults;
pub mod constants;
pub mod convolution;
pub mod core_hole;
pub mod debye;
pub mod density;
pub mod eels;
pub mod elam;
pub mod error;
pub mod exchange;
pub mod fms;
pub mod fovrg;
pub mod fprime;
pub mod fullspectrum;
pub mod genfmt;
pub mod grid;
pub mod interpolation;
pub mod kspace;
pub mod opcons;
pub mod optimization;
pub mod path;
pub mod phase;
pub mod quadrature;
pub mod rhorrp;
pub mod rixs;
pub mod roots;
pub mod screen;
pub mod self_energy;
pub mod sfconv;
pub mod sort;
pub mod special;
pub mod state;
pub mod vector;
pub mod xscorr;
pub mod xsph;
mod xsph_occ_norm;

pub use error::{Error, Result};

pub use angular::{
    AngularError, BasisTransformMatrices, BasisTransformMode, PolarizationTensorMode,
    RelativisticClebschGordanCoefficients, SpinOrbitCouplingTables, TransitionBMatrix,
    TransitionBMatrixInput, basis_transform_matrices, change_basis_representation,
    legendre_normalization, legendre_normalization_table, legendre_polynomials,
    legendre_polynomials_into, mkgtr_clebsch_gordan_coefficients, polarization_tensor,
    relativistic_clebsch_gordan_coefficients, relativistic_state_index_1based, spherical_harmonics,
    spin_orbit_coupling_tables, transition_b_matrix, wigner_3j, wigner_rotation,
};
pub use atomic::{
    AtomMathError, AtomicBreitAngularCoefficients, AtomicConvergenceMix,
    AtomicCoulombCoefficientInput, AtomicDifferentialIntegralInput, AtomicDifferentialIntegralKind,
    AtomicDiracAbnormalExitRecovery, AtomicDiracAbnormalExitRecoveryInput,
    AtomicDiracEnergyCorrection, AtomicDiracEnergyDisagreementCorrection,
    AtomicDiracEnergyDisagreementCorrectionInput, AtomicDiracEnergyDisagreementMatch,
    AtomicDiracEnergyDisagreementMatchInput, AtomicDiracEnergyDisagreementSource,
    AtomicDiracEnergyDisagreementSourceInput, AtomicDiracEnergyStep, AtomicDiracEnergyStepInput,
    AtomicDiracEntryState, AtomicDiracEntryStateInput, AtomicDiracHomogeneousMatch,
    AtomicDiracHomogeneousMatchInput, AtomicDiracHomogeneousPassSetup,
    AtomicDiracHomogeneousPassSetupInput, AtomicDiracHomogeneousSeedInput,
    AtomicDiracInhomogeneousBranch, AtomicDiracInhomogeneousBranchAction,
    AtomicDiracInhomogeneousBranchInput, AtomicDiracInhomogeneousSeedInput, AtomicDiracIntegration,
    AtomicDiracIntegrationInput, AtomicDiracIntegrationMode, AtomicDiracIntegrationSeed,
    AtomicDiracIterationReset, AtomicDiracIterationResetInput, AtomicDiracLargeComponentMatch,
    AtomicDiracLargeComponentMatchInput, AtomicDiracMatchingPointUpdate,
    AtomicDiracMatchingPointUpdateInput, AtomicDiracMethodOneEnergyCorrectionInput,
    AtomicDiracNodeCount, AtomicDiracNodeCountInput, AtomicDiracNodeEnergySearch,
    AtomicDiracNodeEnergySearchInput, AtomicDiracNormalization, AtomicDiracNormalizationInput,
    AtomicDiracRematchAttempt, AtomicDiracRematchAttemptInput, AtomicDiracShootingPassSetup,
    AtomicDiracShootingPassSetupInput, AtomicDiracSolutionNormalization,
    AtomicDiracSolutionNormalizationInput, AtomicDiracSolverSetup, AtomicDiracSolverSetupInput,
    AtomicDiracTwoComponentMatch, AtomicDiracTwoComponentMatchInput, AtomicError, AtomicFormFactor,
    AtomicFormFactorInput, AtomicFormFactorOscillator, AtomicLagrangeParametersInput,
    AtomicLocalDensityExchangeMode, AtomicLocalDensityPotential, AtomicLocalDensityPotentialInput,
    AtomicNuclearPotential, AtomicNuclearPotentialInput, AtomicOrbitalInitialization,
    AtomicOrbitalInitializationInput, AtomicOrbitalPotential, AtomicOrbitalPotentialInput,
    AtomicOverlapAmplitudeReductionInput, AtomicRadialFirstFactor, AtomicRadialFirstFactorView,
    AtomicRadialIntegral, AtomicRadialIntegralInput, AtomicRadialIntegralRequest, AtomicScfState,
    AtomicScfStateInput, AtomicSchmidtIntegralRequest, AtomicSchmidtNormRequest,
    AtomicSchmidtOrthogonalization, AtomicSchmidtOrthogonalizationInput,
    AtomicSchmidtProjectionRequest, AtomicTabulatedMoment, AtomicTabulatedOrbital,
    AtomicTabulatedOverlap, AtomicTabulation, AtomicTabulationInput,
    AtomicTabulationIntegralRequest, AtomicTotalEnergy, AtomicTotalEnergyInput,
    AtomicTotalEnergyRadialInput, AtomicYkZkExchangeInput, AtomicYkZkPreparedSourceInput,
    AtomicYkZkTransform, AtomicYkZkTransformInput, atomic_breit_angular_coefficients,
    atomic_convergence_mix, atomic_coulomb_coefficients, atomic_differential_integral,
    atomic_dirac_abnormal_exit_recovery, atomic_dirac_energy_disagreement_correction,
    atomic_dirac_energy_disagreement_match, atomic_dirac_energy_disagreement_source,
    atomic_dirac_energy_step, atomic_dirac_entry_state, atomic_dirac_homogeneous_match,
    atomic_dirac_homogeneous_pass_setup, atomic_dirac_homogeneous_seed,
    atomic_dirac_inhomogeneous_branch, atomic_dirac_inhomogeneous_seed, atomic_dirac_integration,
    atomic_dirac_iteration_reset, atomic_dirac_large_component_match,
    atomic_dirac_matching_point_update, atomic_dirac_method_one_energy_correction,
    atomic_dirac_node_count, atomic_dirac_node_energy_search, atomic_dirac_normalization,
    atomic_dirac_rematch_attempt, atomic_dirac_shooting_pass_setup,
    atomic_dirac_solution_normalization, atomic_dirac_solver_setup,
    atomic_dirac_two_component_match, atomic_direct_coulomb_coefficient,
    atomic_exchange_coulomb_coefficient, atomic_form_factor, atomic_lagrange_parameters,
    atomic_local_density_potential, atomic_nuclear_potential, atomic_occupation_product,
    atomic_orbital_initialization, atomic_orbital_potential, atomic_overlap_amplitude_reduction,
    atomic_polynomial_product_coefficient, atomic_radial_integral,
    atomic_scf_state_from_configuration, atomic_schmidt_orthogonalization, atomic_symbol,
    atomic_tabulation, atomic_total_energy, atomic_total_energy_from_radials, atomic_weight,
    atomic_yk_zk_exchange, atomic_yk_zk_prepared_source, atomic_yk_zk_transform, nuclear_mass,
    thomas_fermi_density_potential,
};
pub use band::{
    BandEnergiesFromPositiveCounts, BandEnergiesFromPositiveCountsInput, BandEnergySearchMesh,
    BandEnergySearchMeshInput, BandError, BandFreePropagationBandEnergiesFromKspaceNonRelGridInput,
    BandFreePropagationBandEnergiesFromKspaceRelGridInput, BandFreePropagationBandEnergiesInput,
    BandFreePropagationEigenvalueGridInput, BandFreePropagationEigenvaluesFromStructureFactorInput,
    BandFreePropagationFromKspaceNonRelGridInput, BandFreePropagationFromKspaceNonRelInput,
    BandFreePropagationFromKspaceRelGridInput, BandFreePropagationFromKspaceRelInput,
    BandKkrBandEnergies, BandKkrBandEnergiesFromKspaceGrid,
    BandKkrBandEnergiesFromKspaceNonRelGridInput, BandKkrBandEnergiesFromKspacePhaseGrid,
    BandKkrBandEnergiesFromKspacePhaseNonRelGridInput,
    BandKkrBandEnergiesFromKspacePhaseRelGridInput, BandKkrBandEnergiesFromKspaceRelGridInput,
    BandKkrBandEnergiesFromPhaseStructureGrid, BandKkrBandEnergiesFromPhaseStructureGridInput,
    BandKkrBandEnergiesInput, BandKkrEigenvalueGrid, BandKkrEigenvalueGridInput,
    BandKkrEigenvaluesFromStructureFactorInput, BandKkrFromKspace, BandKkrFromKspaceGrid,
    BandKkrFromKspaceNonRelGridInput, BandKkrFromKspaceNonRelInput, BandKkrFromKspaceRelGridInput,
    BandKkrFromKspaceRelInput, BandKkrMatrixInput, BandLatticeTMatrixGridInput,
    BandLatticeTMatrixInput, BandPhaseSearchInterpolation, BandPhaseSearchInterpolationInput,
    BandPositiveCountsFromEigenvaluesInput, BandSortedKkrEigenvaluesInput,
    BandStructureFactorFeffBasisGridInput, BandStructureFactorFeffBasisInput,
    BandStructureFactorFromKspace, BandStructureFactorFromKspaceGrid,
    BandStructureFactorFromKspaceNonRelGridInput, BandStructureFactorFromKspaceNonRelInput,
    BandStructureFactorFromKspaceRelGridInput, BandStructureFactorFromKspaceRelInput,
    band_energies_from_positive_counts, band_energy_search_mesh,
    band_free_propagation_band_energies,
    band_free_propagation_band_energies_from_kspace_non_rel_grid,
    band_free_propagation_band_energies_from_kspace_rel_grid,
    band_free_propagation_eigenvalue_grid, band_free_propagation_eigenvalues_from_structure_factor,
    band_free_propagation_from_kspace_non_rel, band_free_propagation_from_kspace_non_rel_grid,
    band_free_propagation_from_kspace_rel, band_free_propagation_from_kspace_rel_grid,
    band_kkr_band_energies, band_kkr_band_energies_from_kspace_non_rel_grid,
    band_kkr_band_energies_from_kspace_phase_non_rel_grid,
    band_kkr_band_energies_from_kspace_phase_rel_grid, band_kkr_band_energies_from_kspace_rel_grid,
    band_kkr_band_energies_from_phase_structure_grid, band_kkr_eigenvalue_grid,
    band_kkr_eigenvalues_from_structure_factor, band_kkr_from_kspace_non_rel,
    band_kkr_from_kspace_non_rel_grid, band_kkr_from_kspace_rel, band_kkr_from_kspace_rel_grid,
    band_kkr_matrix_from_structure_factor, band_lattice_t_matrix, band_lattice_t_matrix_grid,
    band_phase_search_interpolation, band_positive_counts_from_eigenvalues,
    band_positive_eigenvalue_count, band_sorted_kkr_eigenvalues, band_structure_factor_feff_basis,
    band_structure_factor_feff_basis_grid, band_structure_factor_from_kspace_non_rel,
    band_structure_factor_from_kspace_non_rel_grid, band_structure_factor_from_kspace_rel,
    band_structure_factor_from_kspace_rel_grid,
};
pub use bessel::{
    BesselError, SphericalBessel, SphericalBesselValue, SphericalHankel, besjh, besjn, exjlnl,
    spherical_bessel_j_h, spherical_bessel_j_y,
};
pub use compton::{
    ComptonError, ComptonGrid, ComptonGridInput, ComptonProfileInput, ComptonRhoZzpInput,
    ComptonRhoZzpSlice, ComptonRhorrpDensityInput, ComptonRotationAxisAngle, ComptonWindow,
    compton_build_grid, compton_cross_product, compton_jzzp, compton_jzzp_from_rhorrp,
    compton_profile, compton_profiles, compton_rhozzp_slice, compton_rhozzp_slice_from_rhorrp,
    compton_rotate_vector, compton_rotate_vector_in_place, compton_rotation_axis_angle,
    compton_rotation_matrix,
};
pub use configuration::{
    FEFF_KAPPA_PROJECTION_COUNT, FEFF_ORBITAL_KAPPAS, FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS,
    FEFF_ORBITAL_SLOT_COUNT, OrbitalConfiguration, OrbitalConfigurationError,
    OrbitalConfigurationInput, orbital_configuration,
};
pub use configuration_defaults::{
    FeffConfigurationRecipe, FeffDefaultConfigurationRows, feff_default_configuration_rows,
};
pub use convolution::{
    ConvolutionError, Ff2xAtanCorrectionInput, Ff2xExcitationConvolutionInput, conv,
    conv as lorentz_convolve, conv_in_place, conv1, conv1 as lorentz_convolution_segment,
    ff2x_atan_correction, ff2x_excitation_convolve,
};
pub use core_hole::{
    CoreHoleError, CoreHoleQuantumNumbers, core_hole_quantum_numbers, core_hole_width_ev,
    edge_index, is_edge_label, standard_edge_label,
};
pub use debye::{
    DMDW_ANGSTROM_TO_BOHR, DebyeCorrelation, DebyeError, DmdwDynamicalMatrix, DmdwEinsteinSummary,
    DmdwExpandedPath, DmdwImaginaryPoleSeverity, DmdwImaginaryPoleWarning, DmdwLanczosCoefficients,
    DmdwLanczosPoleSpectrum, DmdwMomentSummary, DmdwPathDescriptor, DmdwPathMotion,
    DmdwPhononCoupling, DmdwPoleWeightedA2f, DmdwRigidBodyModes, DmdwSelfEnergyGrid,
    DmdwSpectralFunctionGrid, DmdwType2AtomGroup, MorseCumulants, SpringAngle,
    SpringDynamicalMatrix, SpringDynamicalMatrixInput, SpringEquationOfMotionInput,
    SpringEquationOfMotionResult, SpringInput, SpringRecursionInput, SpringRecursionResult,
    SpringRecursionState, SpringStretch, ThermalExpansionCumulants, classical_debye_correlation,
    classical_debye_waller_factor, dmdw_center_of_mass, dmdw_debye_waller_factors_from_poles,
    dmdw_expand_path_descriptor, dmdw_expand_path_descriptors, dmdw_inertia_tensor,
    dmdw_ir_dipole_seed_vector, dmdw_lanczos_coefficients, dmdw_lanczos_pole_spectrum,
    dmdw_lanczos_pole_spectrum_with_search, dmdw_lanczos_r_polynomial, dmdw_lanczos_s_polynomial,
    dmdw_lanczos_s_polynomial_derivative, dmdw_mass_weighted_dynamical_matrix,
    dmdw_moment_summaries_from_poles, dmdw_normalize_seed_vector, dmdw_path_motion,
    dmdw_phonon_coupling, dmdw_pole_weighted_a2f, dmdw_project_seed_vector,
    dmdw_rigid_body_projection_modes, dmdw_self_energy_from_a2f_poles,
    dmdw_self_energy_grid_from_a2f_poles, dmdw_single_pole_einstein_summary,
    dmdw_spectral_function_from_a2f_poles, dmdw_type2_pole_weighted_a2f,
    dmdw_vibrational_free_energy_from_poles, equation_of_motion_debye_waller_factor,
    morse_einstein_cumulants, parse_spring_input, quantum_debye_correlation,
    quantum_debye_waller_factor, recursion_debye_waller_factor, spring_dynamical_matrix,
    thermal_expansion_cumulants, update_spring_recursion_state,
};
pub use density::{
    BroydenMix, BroydenMixInput, BroydenWorkspace, CoulombPotentialUpdate,
    CoulombPotentialUpdateInput, CoulombUpdateMode, DensityError, LdosFf2rhoInput,
    LdosFf2rhoTables, LdosFmsdosTraceGridInput, LdosFmsdosTraceInput,
    LdosHubbardMagneticFf2rhoInput, LdosHubbardMagneticFf2rhoTables, LdosHubbardStep1,
    LdosHubbardStep1Input, LdosRholChannel, LdosRholChannelInput, LdosRholDensity,
    LdosRholDensityGrid, LdosRholDensityGridInput, LdosRholDensityInput, LdosRholExactRadialTail,
    LdosRholExactRadialTailInput, LdosRholRadialAssembly, LdosRholRadialAssemblyInput,
    LdosRholTableDriver, LdosRholTableDriverInput, LdosRholWavefunctionTables,
    LdosRholWavefunctionTablesInput, LdosSpinFf2rhoInput, LdosSpinFf2rhoTables, PotRholieDensity,
    PotRholieDensityGrid, PotRholieDensityGridInput, PotRholieDensityInput, PotScfContourRun,
    PotScfContourRunInput, PotScfContourRunStatus, PotScfContourSourceRows,
    PotScfContourSourceRowsInput, PotScfContourStep, PotScfContourStepInput,
    PotScfContourStepStatus, PotScfEnergyDensity, PotScfEnergyDensityInput, PotScfEnergyPoint,
    PotScfEnergyPointInput, PotScfFermiEndpoint, PotScfFermiEndpointInput, PotScfIteration,
    PotScfIterationInput, PotScfIterationStatus, PotScfOuterIteration, PotScfOuterIterationInput,
    PotScfOuterIterationStatus, PotScfState, PotScfStateAdvance, PotScfStateAdvanceInput,
    PotentialOverlap, PotentialOverlapInput, PotentialOverlapNeighbor, ScfDensityStep,
    ScfDensityStepInput, ValenceDensityUpdate, ValenceDensityUpdateInput,
    accumulate_pot_scf_energy_point, advance_pot_scf_state, finish_pot_scf_fermi_endpoint,
    finish_pot_scf_outer_iteration, ldos_ff2rho_tables, ldos_fmsdos_trace, ldos_fmsdos_trace_grid,
    ldos_hubbard_magnetic_ff2rho_tables, ldos_hubbard_step1, ldos_rhol_assemble_radial_components,
    ldos_rhol_channel, ldos_rhol_density, ldos_rhol_density_grid, ldos_rhol_exact_radial_tail,
    ldos_rhol_table_driver, ldos_rhol_wavefunction_tables, ldos_spin_ff2rho_tables,
    mix_broyden_density, overlap_potential_density, pot_rholie_density, pot_rholie_density_grid,
    pot_scf_contour_source_rows, pot_scf_contour_step, pot_scf_energy_density, run_pot_scf_contour,
    run_pot_scf_iteration, update_coulomb_potential, update_scf_density_potential,
    update_valence_density,
};
pub use eels::{
    EelsAngularDependenceInput, EelsAngularDependenceTable, EelsAngularMesh,
    EelsCollectionDependenceInput, EelsCollectionDependenceTable, EelsError, EelsGosInput,
    EelsGosTable, EelsIntegrationMesh, EelsMeshInput, EelsMeshMode, EelsMeshSetup, EelsQMesh,
    EelsQMeshInput, EelsReadSpectrum, EelsReadSpectrumInput, EelsReadSpectrumSource, EelsSpectrum,
    EelsSpectrumInput, FEFF_EELS_ANGULAR_DEPENDENCE_COLUMN_COUNT,
    FEFF_EELS_COLLECTION_DEPENDENCE_COLUMN_COUNT, FEFF_EELS_GOS_Q_COUNT,
    FEFF_EELS_TRANSITION_TENSOR_COMPONENT_COUNT, FEFF_ELECTRON_REST_ENERGY_EV,
    FEFF_H_ON_SQRT_TWO_ME, FEFF_HBARC_ATOMIC, FEFF_HBARC_EV, FEFF_MDFF_AUTOMATIC_THETA_X,
    FEFF_MDFF_AUTOMATIC_THETA_Y, MdffAutomaticQGridInput, MdffManualQGridInput, MdffQGrid,
    MdffSpectrum, MdffSpectrumInput, eels_angular_dependence, eels_angular_mesh,
    eels_collection_angle_dependence, eels_euler_rotation_matrix,
    eels_generalized_oscillator_strength, eels_integration_mesh, eels_mesh_setup,
    eels_product_matrix_vector, eels_qmesh, eels_read_spectrum, eels_spectrum,
    electron_wavelength_atomic_units, mdff_automatic_q_grid, mdff_manual_q_grid, mdff_spectrum,
};
pub use elam::{
    ELAM_EDGE_ATOMIC_NUMBER_MAX, ELAM_EDGE_HOLE_COUNT, ELAM_NEXT_EDGE_SENTINEL_HARTREE,
    ElamEdgeEnergy, ElamError, elam_component_edge_energies_hartree, elam_edge_energy_ev,
    elam_edge_energy_hartree, next_elam_edge_hartree, previous_elam_edge_hartree,
};
pub use exchange::{
    BPHL_RADIUS_COUNT, BPHL_RECORD_COUNT, BPHL_REDUCED_ENERGY_COUNT, BroadenedHedinLundqvistTable,
    ExchangeCorrelation, ExchangeError, HedinLundqvistImaginary, HedinLundqvistSelfEnergy,
    KsdTFreeEnergy, KsdTSpin, XCPOT_MPSE_GRID_POINTS, XcpotFermiCache, XcpotFermiCacheInput,
    XcpotGroundStateBranchInput, XcpotInput, XcpotLocalScales, XcpotLocalScalesInput,
    XcpotManyPoleControl, XcpotManyPoleControlInput, XcpotManyPoleDeltaTable,
    XcpotManyPoleDeltaTableInput, XcpotManyPoleDensityGrid, XcpotManyPoleDensityGridInput,
    XcpotManyPoleRowDeltaInput, XcpotManyPoleSelfEnergyInput, XcpotManyPoleSelfEnergyTableInput,
    XcpotReferenceShift, XcpotReferenceShiftInput, XcpotResult, XcpotSelfEnergyApplication,
    XcpotSelfEnergyApplicationInput, XcpotSelfEnergyCorrection, XcpotSelfEnergyCorrectionInput,
    XcpotSigma, XcpotSigmaInput, broadened_hedin_lundqvist_self_energy,
    dirac_hara_exchange_potential, hedin_lundqvist_ffq, hedin_lundqvist_imaginary_self_energy,
    hedin_lundqvist_self_energy, karasiev_sjostrom_dufty_trickey_free_energy,
    karasiev_sjostrom_dufty_trickey_internal_energy, karasiev_sjostrom_dufty_trickey_vxc,
    perdew_zunger_exchange_correlation, perdew_zunger_vxc, perrot_dharma_wardana_reduced_vxc,
    perrot_dharma_wardana_vxc, quinn_imaginary_self_energy, von_barth_hedin_potential, xcpot,
    xcpot_apply_self_energy_deltas, xcpot_fermi_cache, xcpot_fermi_cache_with_broadened_table,
    xcpot_ground_state_branch, xcpot_local_scales, xcpot_many_pole_control,
    xcpot_many_pole_delta_table, xcpot_many_pole_density_grid, xcpot_many_pole_row_delta,
    xcpot_many_pole_self_energy_delta_table, xcpot_reference_shift, xcpot_self_energy_correction,
    xcpot_self_energy_correction_with_broadened_table, xcpot_sigma,
    xcpot_sigma_with_broadened_table, xcpot_with_broadened_table,
};
pub use fms::{
    FmsAtom, FmsBiCgStabInput, FmsBiCgStabResult, FmsDriverSetup, FmsDriverSetupInput, FmsError,
    FmsFreePropagatorInput, FmsFreePropagatorMatrixInput, FmsFullPotentialLuInput,
    FmsFullPotentialLuResult, FmsGravesMorrisInput, FmsGravesMorrisResult,
    FmsHubbardFullScatteringTransformInput, FmsHubbardScatteringTransformInput,
    FmsHubbardTMatrixInput, FmsHubbardTMatrixTableInput, FmsHubbardTMatrixTransformInput,
    FmsIterativeSystemInput, FmsLuInput, FmsLuResult, FmsPairTables, FmsRealSpaceEnergyInput,
    FmsRealSpaceEnergyResult, FmsRecursionInput, FmsRecursionResult, FmsRotationDirection,
    FmsScatteringInput, FmsScatteringMethod, FmsScatteringMethodSelection, FmsScatteringResult,
    FmsSpinFreePropagatorMatrixInput, FmsSpinPairTables, FmsTMatrixInput, FmsTMatrixTableInput,
    FmsTfqmrInput, FmsTfqmrResult, FmsYprepCluster, FmsYprepClusterInput, FmsYprepGeometry,
    MkgtrGreenTraceInput, MkgtrGreenTraceResult, MkgtrJasGreenTraceInput, MkgtrJasGreenTraceResult,
    MkgtrJasQPairMode, MkgtrJasTransition, fms_bicgstab_scattering, fms_driver_setup,
    fms_free_propagator_element, fms_free_propagator_matrix, fms_full_potential_lu_scattering,
    fms_graves_morris_scattering, fms_hubbard_back_transform_full_scattering,
    fms_hubbard_back_transform_scattering, fms_hubbard_t_matrix_element,
    fms_hubbard_t_matrix_table, fms_hubbard_transform_t_matrix, fms_iterative_system_matrix,
    fms_lu_scattering, fms_pair_tables, fms_real_space_energy, fms_recursion_scattering,
    fms_rotation_matrix, fms_scattering, fms_scattering_method_selection,
    fms_spin_free_propagator_matrix, fms_spin_pair_tables, fms_t_matrix_element,
    fms_t_matrix_table, fms_tfqmr_scattering, fms_yprep_cluster, fms_yprep_geometry,
    mkgtr_green_trace, mkgtr_jas_green_trace, pair_polar_angles, rehr_albers_polynomials,
    rehr_albers_z_axis_propagator, sort_atoms_by_radius, sort_representative_atoms,
};
pub use fovrg::{
    FovrgAngularCoefficientsInput, FovrgC3DerivativeInput, FovrgC3PotentialInput,
    FovrgDiracSolution, FovrgDiracSolverInput, FovrgError, FovrgExchangePotential,
    FovrgExchangePotentialInput, FovrgFlatPotentialInput, FovrgFlatPotentialPropagation,
    FovrgInitialPhotoelectron, FovrgInitialPhotoelectronInput, FovrgInwardSolution,
    FovrgInwardSolutionInput, FovrgNuclearPotential, FovrgNuclearPotentialInput, FovrgOrbitalSetup,
    FovrgOrbitalSetupInput, FovrgOrthogonalization, FovrgOrthogonalizationInput,
    FovrgOutgoingSolution, FovrgOutgoingSolutionInput, FovrgOutwardIntegration,
    FovrgOutwardIntegrationInput, FovrgOverlapIntegralInput, FovrgPotentialDevelopment,
    FovrgPotentialDevelopmentInput, FovrgYkZkExchangeInput, FovrgYkZkTransform,
    FovrgYkZkTransformInput, fovrg_angular_coefficients, fovrg_c3_derivative, fovrg_c3_potential,
    fovrg_complex_real_product_coefficient, fovrg_dirac_solver, fovrg_dirac_solver_c3_potential,
    fovrg_dirac_solver_with_c3_potential, fovrg_exchange_potential, fovrg_flat_potential_propagate,
    fovrg_initial_photoelectron, fovrg_inward_solution, fovrg_nuclear_potential,
    fovrg_orbital_setup, fovrg_outgoing_solution, fovrg_outward_integrate, fovrg_overlap_integral,
    fovrg_potential_development, fovrg_real_product_coefficient, fovrg_schmidt_orthogonalize,
    fovrg_yk_zk_exchange, fovrg_yk_zk_transform,
};
pub use fprime::{
    FprimeContourIntegralInput, FprimeCorrectionInput, FprimeCorrectionOutput,
    FprimeDanesDiagnostics, FprimeError, FprimeLogCase, FprimePositiveAxisIntegralInput,
    fprime_contour_integral, fprime_correction, fprime_correction_with_diagnostics,
    fprime_log_correction, fprime_positive_axis_integral,
};
pub use fullspectrum::{
    FEFF_ALPHA_INV, FEFF_BOHR_ANGSTROM, FEFF_FULLSPECTRUM_BACKGROUND_SUM_MAX,
    FEFF_FULLSPECTRUM_BACKGROUND_SUM_MIN, FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE,
    FEFF_FULLSPECTRUM_DEFAULT_EDGE_HIGH_PADDING_EV, FEFF_FULLSPECTRUM_DEFAULT_EDGE_LOW_PADDING_EV,
    FEFF_FULLSPECTRUM_DEFAULT_EDGE_MIN_EV, FEFF_FULLSPECTRUM_DEFAULT_EDGE_STEP_EV,
    FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT, FEFF_FULLSPECTRUM_EDGE_TRANSITION_SIZE,
    FEFF_FULLSPECTRUM_FINE_STRUCTURE_HIGH_K, FEFF_FULLSPECTRUM_FINE_STRUCTURE_LOW_K,
    FEFF_FULLSPECTRUM_GRID_CAPACITY, FEFF_FULLSPECTRUM_IMAGINARY_EXIT_MULTIPLIER,
    FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY, FEFF_FULLSPECTRUM_XK_STEP, FEFF_HBAR_EV_SECONDS,
    FullSpectrumBackground, FullSpectrumBackgroundInput, FullSpectrumBackgroundSegmentInput,
    FullSpectrumDefaultEnergyGrid, FullSpectrumDefaultGridEdge, FullSpectrumDrudeInput,
    FullSpectrumDrudeTerm, FullSpectrumEdgeAssembly, FullSpectrumEdgeAssemblyInput,
    FullSpectrumEdgeGrid, FullSpectrumEdgeGridInput, FullSpectrumEdgeSelection,
    FullSpectrumEdgeSelectionInput, FullSpectrumError, FullSpectrumFineStructure,
    FullSpectrumFineStructureInput, FullSpectrumFineStructureSegmentInput,
    FullSpectrumHamakerInput, FullSpectrumKramersKronigInput, FullSpectrumLinearGridInput,
    FullSpectrumNumberDensityInput, FullSpectrumOpticalConstants,
    FullSpectrumOpticalConstantsInput, FullSpectrumQSumInput, FullSpectrumScatteringDielectric,
    FullSpectrumScatteringDielectricInput, FullSpectrumSelectedEdge, FullSpectrumSumRules,
    FullSpectrumSumRulesInput, FullSpectrumValenceInput, full_spectrum_assemble_edge,
    full_spectrum_background_from_fprime, full_spectrum_default_energy_grid,
    full_spectrum_drude_term, full_spectrum_edge_energy_grid, full_spectrum_edges_from_occupations,
    full_spectrum_effective_electron_count, full_spectrum_elam_edge_energies,
    full_spectrum_fine_structure_from_segments, full_spectrum_hamaker_transform,
    full_spectrum_kramers_kronig, full_spectrum_linear_energy_grid, full_spectrum_number_density,
    full_spectrum_optical_constants, full_spectrum_scattering_to_dielectric,
    full_spectrum_sum_rules, full_spectrum_valence_epsilon2,
};
pub use genfmt::{
    CurvedWavePolynomialInput, EnergyIndependentMatrixInput, GenfmtCentralPhaseShiftInput,
    GenfmtCentralPhaseShifts, GenfmtChiAmplitudePhase, GenfmtChiAmplitudePhaseInput,
    GenfmtCurvedWaveLegLimit, GenfmtCurvedWaveLegLimits, GenfmtCurvedWaveLegLimitsInput,
    GenfmtCurvedWavePathFactor, GenfmtCurvedWavePathFactorInput, GenfmtCurvedWavePolynomialTables,
    GenfmtCurvedWavePolynomialTablesInput, GenfmtDecomposedChiAmplitudePhase,
    GenfmtDecomposedChiAmplitudePhaseInput, GenfmtDriverSetup, GenfmtDriverSetupInput, GenfmtError,
    GenfmtFeffBinHeader, GenfmtFeffBinHeaderInput, GenfmtFeffBinPotential, GenfmtJasDriverOutput,
    GenfmtJasDriverOutputInput, GenfmtJasDriverSetup, GenfmtJasDriverSetupInput,
    GenfmtJasEffectiveInitialJ, GenfmtJasEffectiveInitialJInput,
    GenfmtJasEnergyGridBranchFromTransitionSetupInput, GenfmtJasLeftRightPathTrace,
    GenfmtJasLeftRightPathTraceInput, GenfmtJasPathEnergyBranchInput, GenfmtJasPathEnergyGrid,
    GenfmtJasPathEnergyGridBranchInput, GenfmtJasPathEnergyGridFinalizationInput,
    GenfmtJasPathEnergyGridFromSetupInput, GenfmtJasPathEnergyGridInput, GenfmtJasPathEnergyPoint,
    GenfmtJasPathEnergyPointInput, GenfmtJasPathEvaluation,
    GenfmtJasPathEvaluationFromDriverSetupInput, GenfmtJasPathEvaluationFromSetupInput,
    GenfmtJasPathEvaluationInput, GenfmtJasPathFinalization, GenfmtJasPathFinalizationInput,
    GenfmtJasPathOutputs, GenfmtJasPathOutputsInput, GenfmtJasPathSequence,
    GenfmtJasPathSequenceFromDriverSetupInput, GenfmtJasPathSequenceFromSetupInput,
    GenfmtJasPathSequenceInput, GenfmtJasPathSetupInput, GenfmtJasPathSignal,
    GenfmtJasPathSignalInput, GenfmtJasPathSignals, GenfmtJasPathSignalsInput, GenfmtJasPathTrace,
    GenfmtJasPathTraceInput, GenfmtJasSphericalPathTrace, GenfmtJasSphericalPathTraceInput,
    GenfmtJasSpinRadialFactorInput, GenfmtJasSpinRadialFactors, GenfmtJasSpinSelection,
    GenfmtJasSpinSelectionInput, GenfmtJasTransitionCount, GenfmtJasTransitionCountInput,
    GenfmtJasTransitionMatrices, GenfmtJasTransitionMatricesInput, GenfmtJasTransitionSetup,
    GenfmtJasTransitionSetupInput, GenfmtLegendreNormalizationInput, GenfmtMomentumGrid,
    GenfmtMomentumGridInput, GenfmtNStarDriverInput, GenfmtNStarInput, GenfmtNStarPathInput,
    GenfmtNStarRow, GenfmtNStarRows, GenfmtNStarRowsInput, GenfmtOrdinaryDriverOutput,
    GenfmtOrdinaryDriverOutputInput, GenfmtOrdinaryPathEnergyGrid,
    GenfmtOrdinaryPathEnergyGridFinalizationInput,
    GenfmtOrdinaryPathEnergyGridFromDriverSetupInput, GenfmtOrdinaryPathEnergyGridFromSetupInput,
    GenfmtOrdinaryPathEnergyGridInput, GenfmtOrdinaryPathEnergyPoint,
    GenfmtOrdinaryPathEnergyPointInput, GenfmtOrdinaryPathEvaluation,
    GenfmtOrdinaryPathEvaluationFromDriverSetupInput, GenfmtOrdinaryPathEvaluationFromSetupInput,
    GenfmtOrdinaryPathEvaluationInput, GenfmtOrdinaryPathFinalization,
    GenfmtOrdinaryPathFinalizationInput, GenfmtOrdinaryPathOutputs, GenfmtOrdinaryPathOutputsInput,
    GenfmtOrdinaryPathSequence, GenfmtOrdinaryPathSequenceFromDriverSetupInput,
    GenfmtOrdinaryPathSequenceFromSetupInput, GenfmtOrdinaryPathSequenceInput,
    GenfmtOrdinaryPathSetupInput, GenfmtOrdinaryPathTrace, GenfmtOrdinaryPathTraceInput,
    GenfmtOrdinarySpinMomentumGrid, GenfmtOrdinarySpinMomentumGridInput,
    GenfmtOrdinaryTransitionMatrices, GenfmtOrdinaryTransitionMatricesInput, GenfmtPathGeometry,
    GenfmtPathGeometryInput, GenfmtPathImportance, GenfmtPathImportanceInput,
    GenfmtPathMatrixProduct, GenfmtPathMatrixProductInput, GenfmtPathMatrixTrace,
    GenfmtPathMatrixTraceInput, GenfmtPathOutputDecision, GenfmtPathOutputDecisionInput,
    GenfmtPathOutputSummary, GenfmtPathRetention, GenfmtPathRetentionInput,
    GenfmtPathRotationTables, GenfmtPathRotationTablesInput, GenfmtPathSetup,
    GenfmtPathSignalContribution, GenfmtPathSignalContributionInput, GenfmtPathSignals,
    GenfmtPathSignalsInput, GenfmtReferenceEnergyMode, GenfmtRetainedPathOutput,
    GenfmtRetainedPathOutputInput, GenfmtScatteringMatrixPlan, GenfmtScatteringMatrixPlanInput,
    GenfmtScatteringMatrixRole, GenfmtScatteringMatrixTask, GenfmtScatteringPathProduct,
    GenfmtScatteringPathProductInput, GenfmtSpinChannelCountInput, GenfmtSpinPhaseShiftInput,
    GenfmtSpinPhaseShifts, GenfmtSpinRadialFactorInput, GenfmtSpinRadialFactors,
    GenfmtSpinReferenceEnergies, GenfmtSpinReferenceEnergyInput, InitialStateRotation,
    InitialStateRotationInput, JasLeftRightAmplitudeInput, JasLeftRightAmplitudeMatrices,
    JasOneSidedTransitionInput, JasOneSidedTransitionMatrices, JasQAngleInput, JasQAngles,
    JasScatteringAmplitudeInput, JasScatteringAmplitudeMatrices, JasSpinTransitionInput,
    JasSpinTransitionMatrix, LambdaIndexInput, LambdaIndexSet, PathRotationAngles,
    PathRotationInput, PolarizedScatteringAmplitudeInput, ScatteringAmplitudeMatrixInput,
    TransitionRotationInput, XStarInput, curved_wave_polynomials,
    energy_independent_transition_matrix, genfmt_central_phase_shifts, genfmt_chi_amplitude_phase,
    genfmt_curved_wave_leg_limits, genfmt_curved_wave_path_factor,
    genfmt_curved_wave_polynomial_tables, genfmt_decomposed_chi_amplitude_phase,
    genfmt_driver_setup, genfmt_feff_bin_header, genfmt_jas_driver_output, genfmt_jas_driver_setup,
    genfmt_jas_effective_initial_j, genfmt_jas_energy_grid_branch_from_transition_setup,
    genfmt_jas_left_right_path_trace, genfmt_jas_path_energy_grid,
    genfmt_jas_path_energy_grid_finalization, genfmt_jas_path_energy_grid_from_setup,
    genfmt_jas_path_energy_point, genfmt_jas_path_evaluation,
    genfmt_jas_path_evaluation_from_driver_setup, genfmt_jas_path_evaluation_from_setup,
    genfmt_jas_path_finalization, genfmt_jas_path_outputs, genfmt_jas_path_sequence,
    genfmt_jas_path_sequence_from_driver_setup, genfmt_jas_path_sequence_from_setup,
    genfmt_jas_path_setup, genfmt_jas_path_signal, genfmt_jas_path_signals, genfmt_jas_path_trace,
    genfmt_jas_spherical_path_trace, genfmt_jas_spin_radial_factors, genfmt_jas_spin_selection,
    genfmt_jas_transition_count, genfmt_jas_transition_matrices, genfmt_jas_transition_setup,
    genfmt_legendre_normalization_table, genfmt_momentum_grid, genfmt_nstar_row, genfmt_nstar_rows,
    genfmt_ordinary_driver_output, genfmt_ordinary_path_energy_grid,
    genfmt_ordinary_path_energy_grid_finalization,
    genfmt_ordinary_path_energy_grid_from_driver_setup,
    genfmt_ordinary_path_energy_grid_from_setup, genfmt_ordinary_path_energy_point,
    genfmt_ordinary_path_evaluation, genfmt_ordinary_path_evaluation_from_driver_setup,
    genfmt_ordinary_path_evaluation_from_setup, genfmt_ordinary_path_finalization,
    genfmt_ordinary_path_outputs, genfmt_ordinary_path_sequence,
    genfmt_ordinary_path_sequence_from_driver_setup, genfmt_ordinary_path_sequence_from_setup,
    genfmt_ordinary_path_setup, genfmt_ordinary_path_trace, genfmt_ordinary_spin_momentum_grid,
    genfmt_ordinary_transition_matrices, genfmt_path_geometry, genfmt_path_importance,
    genfmt_path_matrix_product, genfmt_path_matrix_trace, genfmt_path_output_decision,
    genfmt_path_retention, genfmt_path_rotation_tables, genfmt_path_signal_contribution,
    genfmt_path_signals, genfmt_retained_path_output, genfmt_scattering_matrix_plan,
    genfmt_scattering_path_product, genfmt_spin_channel_count, genfmt_spin_phase_shifts,
    genfmt_spin_radial_factors, genfmt_spin_reference_energies, initial_state_rotation,
    jas_left_right_amplitude_matrices, jas_one_sided_transition_matrices, jas_q_angles,
    jas_scattering_amplitude_matrices, jas_spin_transition_matrix, lambda_indices,
    path_rotation_angles, polarized_scattering_amplitude_matrix, scattering_amplitude_matrix,
    xstar,
};
pub use grid::{
    AtomicQuantitiesGrid, AtomicQuantitiesGridInput, CoulombPotentialSlw, CoulombPotentialSlwInput,
    DiracSpinorGrid, DiracSpinorGridInput, DiracSpinorOrbitalsGrid, DiracSpinorOrbitalsGridInput,
    FEFF_FERMI_MOMENTUM_FACTOR, FEFF_HARTREE_EV, FermiLevel, FermiLevelInput, GridError,
    InterstitialShellValues, InterstitialShellValuesInput, LOUCKS_DELTA, LOUCKS_X_OFFSET,
    LoucksSphericalOverlap, LoucksSphericalOverlapInput, MuffinTinInterstitialParameters,
    MuffinTinInterstitialParametersInput, MuffinTinOverlapMatrix, MuffinTinOverlapMatrixInput,
    MuffinTinOverlapNeighbor, MuffinTinOverlapProjection, MuffinTinOverlapProjectionInput,
    MuffinTinOverlapProjectionMode, MuffinTinRadiusParameters, MuffinTinRadiusParametersInput,
    NormanRadius, NormanRadiusInput, OverlapDensityIndices, OverlapDensityIndicesInput,
    PotentialGrid, PotentialGridInput, ScmtEnergyGrid, ScmtEnergyGridInput, coulomb_potential_slw,
    fix_atomic_quantities_grid, fix_dirac_spinor_grid, fix_dirac_spinor_orbitals_grid,
    fix_potential_grid, interstitial_fermi_level, interstitial_shell_values, loucks_index_below,
    loucks_radius, loucks_x, muffin_tin_interstitial_parameters, muffin_tin_overlap_matrix,
    muffin_tin_radius_parameters, norman_radius_from_density, overlap_density_indices,
    project_muffin_tin_overlap, radial_index_below, radial_radius, radial_x, scmt_energy_grid,
    sphere_overlap_cap_volume, sphere_overlap_lens_volume, sum_loucks_spherical_overlap,
    wave_number_from_hartree,
};
pub use interpolation::{
    Interpolation, InterpolationError, LintCache, interpolation_polynomial_coefficients, lint,
    lint_with_cache, locate_below, polynomial_interpolate, polynomial_interpolate_complex, terp,
    terp1, terpc,
};
pub use kspace::{
    BandKPathMesh, BravaisLattice, KMeshArbitraryMesh, KMeshBravaisBasis, KMeshDivisionReduction,
    KMeshDivisions, KMeshReduction, KMeshTetrahedronRecords, KPath, KSPACE_Q_PAIR_TOLERANCE,
    KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE, KSpaceAngularTables, KSpaceDirectLatticeSetup,
    KSpaceDirectLatticeTerms, KSpaceDirectLatticeTermsInput, KSpaceEnergyDependentTerms,
    KSpaceEnergyDependentTermsInput, KSpaceError, KSpaceEwaldEnergyTables,
    KSpaceEwaldEnergyTablesInput, KSpaceHarmonicPolynomialsInput, KSpaceInitialEwaldTables,
    KSpaceQPairGroups, KSpaceReciprocalLatticeSetup, KSpaceReciprocalPairPhases,
    KSpaceReciprocalPairPhasesInput, KSpaceStrbbddInput, KSpaceStrsetMatrices,
    KSpaceStrsetNonRelFromLatticeSumInput, KSpaceStrsetNonRelInput,
    KSpaceStrsetRelFromLatticeSumInput, KSpaceStrsetRelInput, LdosWeylKMesh, PointGroup,
    ReducedVector, SymmetryCheck, band_k_path_mesh, bravais_lattice, bravais_lattice_index,
    change_cartesian_basis, define_k_path, kmesh_arbitrary_mesh, kmesh_basis_divisions,
    kmesh_bravais_basis, kmesh_tetrahedron_division, kmesh_tetrahedron_records,
    kspace_angular_tables, kspace_direct_lattice_setup, kspace_direct_lattice_terms,
    kspace_energy_dependent_terms, kspace_ewald_energy_tables,
    kspace_ewald_energy_tables_from_initial, kspace_harmonic_polynomials, kspace_q_pair_groups,
    kspace_qjltab, kspace_reciprocal_lattice_setup, kspace_reciprocal_pair_phases,
    kspace_strbbdd_lattice_sum, kspace_strset_non_rel_from_lattice_sum,
    kspace_strset_non_relativistic, kspace_strset_rel_from_lattice_sum, kspace_strset_relativistic,
    ldos_weyl_kmesh, point_group_operations, reciprocal_lattice_vectors, reciprocal_metric,
    redefine_lattice_symmetry_operations, reduce_kmesh_common_divisor,
    reduce_kmesh_irreducible_points, reduce_to_lattice_cell, subtract_lattice_translation,
    symmetry_check, transform_lapw_symmetry_operations,
};
pub use opcons::{CombinedEpsilon, EpsilonTable, OpconsError, combine_epsilon_tables};
pub use optimization::{
    MinimumBracket, OptimizationError, TableMinimum, bracket_table_minimum,
    brent_derivative_minimum, brent_table_minimum,
};
pub use path::{
    PathCanonicalRepresentation, PathCanonicalRepresentationInput, PathCriteriaDecision,
    PathCriteriaDecisionInput, PathDegeneracyCandidate, PathDegeneracyGroup,
    PathDegeneracyGroupsInput, PathDegeneracyProcessedRange, PathDegeneracyRange,
    PathDegeneracyRangeInput, PathDegeneracyRecord, PathDegeneracyReduction,
    PathDegeneracyReductionInput, PathDegeneracyRetention, PathDegeneracyRetentionDecision,
    PathDegeneracyRetentionInput, PathDegeneracyRetentionReference, PathError, PathGeometry,
    PathOutputCriterion, PathOutputCriterionInput, PathOutputImportance, PathOutputImportanceInput,
    PathOutputParameters, PathPhaseCriteriaInput, PathPhaseCriteriaTables, PathStandardCoordinates,
    PathStandardCoordinatesInput, PathfinderPreparation, PathfinderPreparationInput,
    PathfinderRecord, PathfinderReduction, PathfinderReductionInput, PathfinderSearch,
    PathfinderSearchInput, pack_path_indices, path_beta_indices, path_canonical_representation,
    path_criteria_decision, path_degeneracy_groups, path_degeneracy_hash, path_degeneracy_range,
    path_degeneracy_reduction, path_degeneracy_retention, path_geometry, path_heap_bubble_down,
    path_heap_bubble_up, path_heap_criterion, path_output_criterion, path_output_importance,
    path_output_parameters, path_phase_criteria_tables, path_standard_coordinates,
    pathfinder_preparation, pathfinder_reduction, pathfinder_search, unpack_path_indices,
};
pub use phase::{
    ComplexAmplitudePhase, PhaseError, complex_atan, complex_atan2_amplitude_phase,
    muffin_tin_phase_amplitude, remove_phase_jump, remove_phase_jumps, remove_phase_jumps_array,
};
pub use quadrature::{
    GaussLegendreQuadrature, QuadratureError, csomm, csomm2, csommjas, gauss_legendre_quadrature,
    somm, somm2, strap, trap,
};
pub use rhorrp::{
    RhorrpAtomicDensityInput, RhorrpDensityGridEvaluation, RhorrpDensityGridFromTablesInput,
    RhorrpDensityGridInput, RhorrpDensityGridPoints, RhorrpDensityIntegrationInput,
    RhorrpEnergyDensityInput, RhorrpEnergyPrefactorInput, RhorrpError,
    RhorrpExactRadialContinuation, RhorrpExactRadialContinuationInput, RhorrpExactRadialTail,
    RhorrpExactRadialTailInput, RhorrpFermiDistributionInput, RhorrpFmsInclusionInput,
    RhorrpIrregularFixInput, RhorrpIrregularInitialCondition, RhorrpIrregularInitialConditionInput,
    RhorrpIrregularSolutionTransform, RhorrpIrregularSolutionTransformInput,
    RhorrpIrregularWronskianScale, RhorrpIrregularWronskianScaleInput, RhorrpMuffinTinMatch,
    RhorrpMuffinTinMatchInput, RhorrpNearestAtom, RhorrpNearestAtomInput, RhorrpNearestAtomTable,
    RhorrpNearestAtomTableInput, RhorrpPairDensityInput, RhorrpPairEnergyDensityInput,
    RhorrpPointDensityFromTablesInput, RhorrpPointDensityInput,
    RhorrpPointEnergyDensityFromTablesInput, RhorrpPointEnergyDensityInput,
    RhorrpPointPairDensityFromTablesInput, RhorrpPointPairDensityInput,
    RhorrpPointPairEnergyDensityFromTablesInput, RhorrpPointPairEnergyDensityInput,
    RhorrpPotentialReferenceShift, RhorrpPotentialReferenceShiftInput,
    RhorrpPotentialReferenceShifts, RhorrpPotentialReferenceShiftsInput,
    RhorrpPotentialWavefunctions, RhorrpPotentialWavefunctionsInput,
    RhorrpPreparedPotentialWavefunctionsInput, RhorrpPreparedWavefunctionTablesInput,
    RhorrpProcessRange, RhorrpRadialInterpolationInput, RhorrpRadialInterpolationLocation,
    RhorrpRadialSolutionAssembly, RhorrpRadialSolutionAssemblyInput, RhorrpRegularSolutionScale,
    RhorrpRegularSolutionScaleInput, RhorrpSameSiteGreenInput, RhorrpScatteringGreenInput,
    RhorrpScatteringMatrixSelectionInput, RhorrpWavefunctionChannel,
    RhorrpWavefunctionChannelInput, RhorrpWavefunctionGridPreparation,
    RhorrpWavefunctionGridPreparationInput, RhorrpWavefunctionInterpolationInput,
    RhorrpWavefunctionSetup, RhorrpWavefunctionSetupInput, RhorrpWavefunctionTables,
    RhorrpWavefunctionTablesInput, rhorrp_assemble_radial_solutions, rhorrp_atomic_density,
    rhorrp_c3_scale_for_angular_momentum, rhorrp_density_grid_points,
    rhorrp_density_reference_energy_hartree, rhorrp_effective_temperature_hartree,
    rhorrp_energy_prefactor, rhorrp_evaluate_density_grid,
    rhorrp_evaluate_density_grid_from_tables, rhorrp_exact_radial_continuation,
    rhorrp_exact_radial_tail, rhorrp_fermi_distribution, rhorrp_finish_energy_density,
    rhorrp_fix_irregular_origin, rhorrp_fms_inclusion_counts, rhorrp_integrate_density,
    rhorrp_interpolate_wavefunction, rhorrp_irregular_initial_condition,
    rhorrp_irregular_solution_transform, rhorrp_irregular_wronskian_scale, rhorrp_muffin_tin_match,
    rhorrp_nearest_atom, rhorrp_nearest_atom_table, rhorrp_next_index_1based, rhorrp_pair_density,
    rhorrp_pair_energy_density, rhorrp_photoelectron_kappa, rhorrp_point_at_index,
    rhorrp_point_density, rhorrp_point_density_from_tables, rhorrp_point_energy_density,
    rhorrp_point_energy_density_from_tables, rhorrp_point_pair_density,
    rhorrp_point_pair_density_from_tables, rhorrp_point_pair_energy_density,
    rhorrp_point_pair_energy_density_from_tables, rhorrp_potential_reference_shift,
    rhorrp_potential_reference_shifts, rhorrp_potential_wavefunctions,
    rhorrp_prepare_wavefunction_grids, rhorrp_prepared_potential_wavefunctions,
    rhorrp_prepared_wavefunction_tables, rhorrp_process_ranges,
    rhorrp_radial_interpolation_location, rhorrp_regular_solution_scale, rhorrp_same_site_green,
    rhorrp_scattering_green, rhorrp_select_scattering_matrix, rhorrp_wavefunction_channel,
    rhorrp_wavefunction_setup, rhorrp_wavefunction_tables,
};
pub use rixs::{
    FEFF_RIXS_FINAL_BROADENING_SKIP_WIDTH, RixsCoreHolePotentialInput,
    RixsDirectFinalTransitionInput, RixsEdgeBroadeningInput, RixsEdgeContributionInput, RixsError,
    RixsFinalEnergyBroadeningInput, RixsFinalSpectrum, RixsFinalSpectrumInput,
    RixsIncidentAmplitudeConvolutionInput, RixsIncidentEnergyBroadeningInput,
    RixsInitialAmplitudeInput, RixsPoleNormalization, RixsPoleNormalizationInput,
    RixsPostRawSpectrum, RixsPostRawSpectrumInput, RixsRadialFunctionRecord,
    RixsRadialFunctionTable, RixsRadialFunctionTableInput, RixsRadialGrid, RixsRadialGridInput,
    RixsRadialOverlapInput, RixsRawCrossSectionInput, RixsSatelliteConvolutionInput,
    RixsSatelliteSpectrum, RixsSatelliteSpectrumInput, RixsSelfEnergyGridInput,
    RixsTransitionMatrixInput, RixsTransitionMatrixSetup, RixsTransitionPhaseShiftInput,
    RixsWaveNumberInput, RixsWaveNumbers, bilinear_interpolate_complex, integrated_double_lorentz,
    kk_integral, rixs_broaden_edge_contributions, rixs_core_hole_potential_difference,
    rixs_default_pole_normalization, rixs_direct_final_transition_amplitudes,
    rixs_final_energy_broadening, rixs_final_spectrum, rixs_incident_amplitude_convolution,
    rixs_incident_energy_broadening, rixs_initial_transition_amplitudes, rixs_normalize_poles,
    rixs_post_raw_spectrum, rixs_prepare_self_energy_grid, rixs_radial_function_table,
    rixs_radial_grid, rixs_radial_transition_overlaps, rixs_raw_cross_section,
    rixs_satellite_convolution, rixs_satellite_spectrum, rixs_sum_edge_contributions,
    rixs_transition_matrix_setup, rixs_transition_phase_shifts, rixs_wave_numbers,
};
pub use roots::{
    ComplexRoots, RealPolynomialRoots, RootError, cubic_zeros, depressed_quartic_roots,
    quadratic_zeros, real_polynomial_roots,
};
pub use screen::{
    SCREEN_ALPHA_INVERSE, SCREEN_BOHR_ANGSTROM, SCREEN_FINE_STRUCTURE_ALPHA, SCREEN_HARTREE_EV,
    ScreenClusterResponseSliceInput, ScreenClusterResponseSlicesInput, ScreenContourEnergyGrid,
    ScreenContourEnergyGridInput, ScreenCrpaDensityWeights, ScreenCrpaHubbardSummary,
    ScreenCrpaProjectionWindow, ScreenCrpaResponseSliceInput, ScreenCrpaScreenedHubbard,
    ScreenCrpaScreenedHubbardInput, ScreenEnergyState, ScreenEnergyStateInput, ScreenError,
    ScreenExactRadialContinuation, ScreenExactRadialContinuationInput,
    ScreenExactRadialContinuationTail, ScreenExactRadialContinuationTailInput,
    ScreenFmsResponseSliceInput, ScreenFovrgChannelAssembly, ScreenFovrgChannelAssemblyInput,
    ScreenFovrgCubeAssembly, ScreenFovrgCubeAssemblyInput, ScreenFovrgMatchedChannelAssemblyInput,
    ScreenFovrgMatchedCubeAssembly, ScreenFovrgMatchedCubeAssemblyInput, ScreenGetphRadialBounds,
    ScreenGetphRadialBoundsInput, ScreenIntegratedResponseInput, ScreenIrregularInitialCondition,
    ScreenIrregularInitialConditionInput, ScreenIrregularWronskianScale,
    ScreenIrregularWronskianScaleInput, ScreenPhasePotential, ScreenPhasePotentialInput,
    ScreenRadialBounds, ScreenRadialBoundsInput, ScreenRadialChannelAssembly,
    ScreenRadialChannelAssemblyInput, ScreenRadialCubeAssembly, ScreenRadialCubeAssemblyInput,
    ScreenRdgeomAtomicUnits, ScreenRdgeomAtomicUnitsInput, ScreenSolutionNormalization,
    ScreenSolutionNormalizationInput, ScreenSolvedCoreHoleResponse,
    ScreenSolvedCoreHoleResponseInput, screen_atomic_response_slice,
    screen_bare_core_hole_potential, screen_cluster_response_channel_slice,
    screen_cluster_response_slice, screen_cluster_response_slices, screen_contour_energy_grid,
    screen_coulomb_kernel_matrix, screen_crpa_density_weights, screen_crpa_hubbard_summary,
    screen_crpa_orbital_density, screen_crpa_response_slice, screen_crpa_screened_hubbard_summary,
    screen_energy_integration_delta, screen_energy_state, screen_exact_radial_continuation,
    screen_exact_radial_continuation_tail, screen_exponential_energy_grid,
    screen_fms_cluster_green_trace, screen_fms_response_slice, screen_fovrg_channel_assembly,
    screen_fovrg_cube_assembly, screen_fovrg_matched_channel_assembly,
    screen_fovrg_matched_cube_assembly, screen_getph_lmax, screen_getph_radial_bounds,
    screen_integrate_response_step, screen_integrated_response, screen_irregular_initial_condition,
    screen_irregular_wronskian_scale, screen_lda_exchange_correlation_kernel,
    screen_phase_potential_reference_shift, screen_radial_bounds, screen_radial_channel_assembly,
    screen_radial_coulomb_potential, screen_radial_cube_assembly, screen_radial_grid,
    screen_radial_index_1based, screen_rdgeom_atomic_units, screen_response_system_matrix,
    screen_solution_normalization, screen_solve_response_potential,
    screen_solved_core_hole_response, screen_symmetrize_response_upper,
};
pub use self_energy::{
    CgratrIntegral, ExcitationPole, ManyPoleSelfEnergy, ManyPoleSelfEnergyInput, SelfEnergyError,
    SelfEnergyIntegrandInput, SelfEnergySinglePoleInput, SingularityFunction, cgratr,
    find_self_energy_singularities, gamma_q, hartree_fock_exchange, log_i, make_excitation_poles,
    many_pole_self_energy, omega_q, self_energy_bpr1_integrand, self_energy_bpr2_integrand,
    self_energy_bpr3_integrand, self_energy_dr1_integrand, self_energy_dr2_integrand,
    self_energy_dr3_integrand, self_energy_pole_dispersion, self_energy_r1_integrand,
    self_energy_r2_integrand, self_energy_r3_integrand, self_energy_single_pole,
    self_energy_single_pole_derivative,
};
pub use sfconv::{
    SFCONV_MKSPECTF_GRID_LEN, SFCONV_SO2CONV_BOHR_ANGSTROM, SFCONV_SO2CONV_HARTREE_EV,
    SFCONV_SO2CONV_MOMENTUM_GRID_LEN, SfconvAdaptiveIntegral, SfconvBroadenedSelfEnergy,
    SfconvBroadenedSelfEnergyBranch, SfconvBroadenedSelfEnergyDerivative,
    SfconvBroadenedSelfEnergyDerivativeIntegrands, SfconvBroadenedSelfEnergyIntegrandInput,
    SfconvBroadenedSelfEnergyIntegrands, SfconvConvolution, SfconvConvolutionInput, SfconvError,
    SfconvExafsConvolution, SfconvExafsConvolutionInput, SfconvExponentialReductionInput,
    SfconvExtrinsicSatelliteInput, SfconvExtrinsicSatelliteMode, SfconvExtrinsicSatelliteSplit,
    SfconvExtrinsicSatelliteSplitInput, SfconvFeffPathInterpolation,
    SfconvFeffPathInterpolationInput, SfconvFeffPathSignal, SfconvFeffPathSignalInput,
    SfconvKramersKronigInput, SfconvMomentumSpectralInterpolation,
    SfconvMomentumSpectralInterpolationInput, SfconvPathAverage, SfconvPathAverageInput,
    SfconvPhotoelectronMomentum, SfconvPhotoelectronMomentumInput, SfconvPlasmaParameters,
    SfconvPole, SfconvQLimits, SfconvQuasiparticleInterference,
    SfconvQuasiparticleInterferenceInput, SfconvQuasiparticlePeakInput, SfconvQuasiparticlePole,
    SfconvQuasiparticlePoleInput, SfconvQuasiparticleTable, SfconvQuasiparticleTableInput,
    SfconvRenormalization, SfconvSatelliteContext, SfconvSatelliteCorrection,
    SfconvSatelliteCorrectionInput, SfconvSatelliteIntegral, SfconvSatellitePoleContributions,
    SfconvSatellitePoleContributionsInput, SfconvSatelliteSelfEnergy, SfconvSatelliteTable,
    SfconvSatelliteTableInput, SfconvSelfEnergyContext, SfconvSo2convExafsEnergyPaddingInput,
    SfconvSo2convExafsPreparation, SfconvSo2convExafsPreparationInput, SfconvSo2convMaterialInput,
    SfconvSo2convMaterialParameters, SfconvSo2convSelfEnergyGrid, SfconvSo2convSelfEnergyGridInput,
    SfconvSo2convSelfEnergySampleInput, SfconvSo2convSpecfunctGrid, SfconvSo2convSpecfunctInput,
    SfconvSo2convXanesPreparation, SfconvSo2convXanesPreparationInput, SfconvSpectralCell,
    SfconvSpectralCellInput, SfconvSpectralEnergyGrid, SfconvSpectralFinalization,
    SfconvSpectralFinalizationInput, SfconvSpectralInterpolation, SfconvSpectralInterpolationInput,
    SfconvSpectralTable, SfconvSpectralTableInput, SfconvSpectralWeightsInput,
    SfconvXanesConvolution, SfconvXanesConvolutionInput, sfconv_broadened_self_energy,
    sfconv_broadened_self_energy_derivative, sfconv_broadened_self_energy_derivative_integrands,
    sfconv_broadened_self_energy_integrands, sfconv_convolve, sfconv_correct_satellite_weights,
    sfconv_coupling_potential_squared, sfconv_exafs_convolution, sfconv_exponential_reduction,
    sfconv_extrinsic_beta, sfconv_extrinsic_satellite, sfconv_extrinsic_satellite_broadened,
    sfconv_extrinsic_satellite_debroadened, sfconv_feff_path_signal,
    sfconv_finalize_spectral_table, sfconv_find_singularities, sfconv_free_electron_exchange,
    sfconv_grater_integrate, sfconv_imaginary_self_energy, sfconv_imaginary_self_energy_derivative,
    sfconv_interference_quasiparticle, sfconv_interference_quasiparticle_integrand,
    sfconv_interference_satellite, sfconv_interference_satellite_integrand,
    sfconv_interpolate_feff_path, sfconv_interpolate_momentum_spectral_function,
    sfconv_interpolate_spectral_function, sfconv_intrinsic_satellite,
    sfconv_intrinsic_satellite_integrand, sfconv_inverse_pole_dispersion,
    sfconv_kramers_kronig_real_part, sfconv_path_average, sfconv_plasma_parameters,
    sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion, sfconv_pole_dispersion_derivative,
    sfconv_pole_dispersion_second_derivative, sfconv_q_limits,
    sfconv_quasiparticle_interference_amplitude, sfconv_quasiparticle_main_peak,
    sfconv_quasiparticle_pole, sfconv_quasiparticle_table, sfconv_real_self_energy,
    sfconv_real_self_energy_derivative, sfconv_real_self_energy_derivative_integrand_lower,
    sfconv_real_self_energy_derivative_integrand_middle,
    sfconv_real_self_energy_derivative_integrand_upper, sfconv_real_self_energy_integrand_lower,
    sfconv_real_self_energy_integrand_middle, sfconv_real_self_energy_integrand_upper,
    sfconv_satellite_pole_contributions, sfconv_satellite_table, sfconv_select_pole,
    sfconv_self_energy_renormalization, sfconv_so2conv_broadened_self_energy_grid,
    sfconv_so2conv_broadened_self_energy_sample, sfconv_so2conv_material_parameters,
    sfconv_so2conv_momentum_grid, sfconv_so2conv_pad_exafs_energy_grid,
    sfconv_so2conv_photoelectron_momentum, sfconv_so2conv_prepare_exafs_signal,
    sfconv_so2conv_prepare_xanes_signal, sfconv_so2conv_specfunct_grid,
    sfconv_so2conv_unbroadened_self_energy_grid, sfconv_so2conv_unbroadened_self_energy_sample,
    sfconv_spectral_cell, sfconv_spectral_energy_grid, sfconv_spectral_table,
    sfconv_spectral_weights, sfconv_split_extrinsic_satellite, sfconv_xanes_convolution,
};
pub use sort::{
    SortError, qsortd_order_1based, qsorti_compatible_order, qsorti_order_1based, sort_order,
    sort_order_1based, sortid_order_1based, sortii_order_1based, sortir_order_1based,
};
pub use special::{SpecialFunctionError, complex_digamma, x_log_x};
pub use state::{
    StateKet, StateKetError, StateKetSet, construct_state_kets, construct_state_kets_with_limit,
};
pub use vector::{
    HydrogenBondAdjustment, HydrogenBondAdjustmentInput, ReferenceFrameRotation, Vector3,
    VectorError, adjust_hydrogen_bonds, distance_between, normalize_vector, nrixs_qtrig,
    rotate_into_reference_frame, single_precision_distance_between, vector_norm,
};
pub use xscorr::{XscorrError, xscorr_arctangent_step, xscorr_lorentz_kernel};
pub use xsph::{
    XSPH_AXAFS_COLUMN_COUNT, XsphAxafs, XsphAxafsInput, XsphBcoefTransitionIndex,
    XsphBcoefTransitionIndices, XsphBcoefTransitionIndicesInput, XsphCalculationPlan,
    XsphEmptyCellPhase, XsphEmptyCellPhaseInput, XsphError, XsphFprimeEnergyGrid84,
    XsphHoleOrbital, XsphHoleOrbitalInput, XsphHubbardPhaseAssignment,
    XsphHubbardPhaseAssignmentInput, XsphHubbardPhasePotentialInput,
    XsphHubbardPhasePotentialShift, XsphJasOrthogonalityCorrection,
    XsphJasOrthogonalityCorrectionInput, XsphJasOverlap, XsphJasOverlapInput,
    XsphJasPhaseEnergyMesh, XsphJasPhaseEnergyMeshInput, XsphJasRadialCrossIntegral,
    XsphJasRadialCrossIntegralInput, XsphJasRadialIntegral, XsphJasRadialIntegralInput,
    XsphLgSpectrumUpdateInput, XsphLjSpectrumUpdateInput, XsphNrixsTransitionIndex,
    XsphNrixsTransitionIndices, XsphNrixsTransitionIndicesInput, XsphPhaseAngularLimit,
    XsphPhaseAngularLimitInput, XsphPhaseChannel, XsphPhaseChannelPlan, XsphPhaseChannelPlanInput,
    XsphPhaseCutoff, XsphPhaseCutoffInput, XsphPhaseEnergyDecision, XsphPhaseEnergyDynamics,
    XsphPhaseEnergyMesh84, XsphPhaseEnergyMesh84Input, XsphPhaseEnergySetup,
    XsphPhaseEnergySetupInput, XsphPhaseGridPreparation, XsphPhaseGridPreparationInput,
    XsphPhasePlasmonPole, XsphPhasePlasmonPoleSetup, XsphPhasePlasmonPoleSetupInput,
    XsphPhaseReferenceTail, XsphPhaseSelfEnergySummary, XsphPhaseSelfEnergySummaryInput,
    XsphPhaseUserGridInput, XsphPhaseUserGridKind, XsphPhaseUserGridMinimum,
    XsphPhaseUserGridRecord, XsphPhaseUserRegularGrid, XsphRadialCrossIntegral,
    XsphRadialCrossIntegralBranch, XsphRadialCrossIntegralInput, XsphRadialIntegral,
    XsphRadialIntegralInput, XsphRadialIntegralMode, XsphRegularPhase, XsphRegularPhaseChannel,
    XsphRegularPhaseInput, XsphRelativisticMultipoleFactors, XsphSortedEnergyGrid,
    XsphSpectrumUpdateMode, XsphTdldaAngularKernel, XsphTdldaAngularKernelInput,
    XsphTdldaBroadenedChannelSpectra, XsphTdldaChannelBroadeningInput, XsphTdldaChannelMultipliers,
    XsphTdldaChannelMultipliersInput, XsphTdldaChannelSpectra, XsphTdldaChannelSpectraInput,
    XsphTdldaConditionedResponse, XsphTdldaCoulombFields, XsphTdldaCoulombFieldsInput,
    XsphTdldaDirectKernel, XsphTdldaDirectKernelInput, XsphTdldaEnergyRows,
    XsphTdldaEnergyRowsInput, XsphTdldaKramersKronigInput, XsphTdldaKramersKronigResponse,
    XsphTdldaNonlocalExchangeInput, XsphTdldaProjectedKernel, XsphTdldaProjectedKernelInput,
    XsphTdldaProjectorOrthogonalization, XsphTdldaProjectorOrthogonalizationInput,
    XsphTdldaProjectorSelector, XsphTdldaRadialKernel, XsphTdldaRadialKernelInput,
    XsphTdldaRawResponse, XsphTdldaRawResponseInput, XsphTdldaResponseConditioningInput,
    XsphTdldaRowWaveNumbers, XsphTdldaRowWaveNumbersInput, XsphTdldaScreenedDipole,
    XsphTdldaScreenedDipoleInput, XsphTdldaWeightedResponse, XsphTdldaWeightedResponseInput,
    XsphTdldaXmuChannelInput, XsphTdldaXsedgeRows, XsphTdldaXsedgeRowsInput,
    XsphThermalPhaseEnergyMesh, XsphThermalPhaseEnergyMeshInput, XsphTransitionMultipole,
    XsphXanesEnergyGrid84, XsphXesEnergyGrid84, XsphXrayBesselTable, XsphXrayBesselTableInput,
    XsphXsectBcoefCentralCrossSectionInput, XsphXsectBcoefCrossTermAccumulationInput,
    XsphXsectBcoefCrossTermStateAccumulationInput, XsphXsectBcoefDirectTransitionInput,
    XsphXsectBcoefDirectTransitionUpdate, XsphXsectBcoefDirectTransitionUpdateInput,
    XsphXsectBcoefNonstandardChannelRow, XsphXsectBcoefNonstandardChannelRowInput,
    XsphXsectBcoefNonstandardEnergyRow, XsphXsectBcoefNonstandardEnergyRowInput,
    XsphXsectBcoefOrdinaryRow, XsphXsectBcoefOrdinaryRowInput, XsphXsectBcoefStandardChannelRow,
    XsphXsectBcoefStandardChannelRowInput, XsphXsectBcoefStandardEnergyRow,
    XsphXsectBcoefStandardEnergyRowFieldsInput, XsphXsectBcoefStandardEnergyRowInput,
    XsphXsectBcoefStandardTransitionField, XsphXsectBcoefWeights, XsphXsectBcoefWeightsInput,
    XsphXsectCentralCrossSection, XsphXsectCentralCrossSectionInput,
    XsphXsectCrossTermAccumulation, XsphXsectCrossTermAccumulationInput, XsphXsectCrossTermMode,
    XsphXsectCrossTermPlan, XsphXsectCrossTermPlanInput, XsphXsectCrossTermState,
    XsphXsectCrossTermStateReuse, XsphXsectCrossTermStateReuseInput,
    XsphXsectCrossTermStateSaveInput, XsphXsectDensityBranch, XsphXsectDensityBranchInput,
    XsphXsectDirectTransition, XsphXsectDirectTransitionInput, XsphXsectEmbeddedDensity,
    XsphXsectEmbeddedDensityInput, XsphXsectEnergyDecision, XsphXsectEnergySetup,
    XsphXsectEnergySetupInput, XsphXsectFscfComponentPart, XsphXsectFscfIntegral,
    XsphXsectFscfIntegralInput, XsphXsectFscfSelection, XsphXsectFscfWeight, XsphXsectFscfWeights,
    XsphXsectFscfWeightsInput, XsphXsectHoleNormalization, XsphXsectHoleNormalizationInput,
    XsphXsectIrregularChannel, XsphXsectIrregularChannelInput, XsphXsectIrregularInitialCondition,
    XsphXsectIrregularInitialConditionInput, XsphXsectIrregularTransform,
    XsphXsectIrregularTransformInput, XsphXsectOutputNormalization,
    XsphXsectOutputNormalizationInput, XsphXsectPhiscfAccumulatedResponse,
    XsphXsectPhiscfAccumulatedResponseInput, XsphXsectPhiscfAngularChannels,
    XsphXsectPhiscfContributionPlan, XsphXsectPhiscfContributionPlanInput,
    XsphXsectPhiscfContributionPlanRow, XsphXsectPhiscfContributionRule,
    XsphXsectPhiscfContributionRuleInput, XsphXsectPhiscfFieldAssemblyInput, XsphXsectPhiscfFields,
    XsphXsectPhiscfIrregularSeed, XsphXsectPhiscfIrregularSeedInput, XsphXsectPhiscfLinearSolve,
    XsphXsectPhiscfLinearSolveInput, XsphXsectPhiscfLipman, XsphXsectPhiscfLipmanInput,
    XsphXsectPhiscfLocalField, XsphXsectPhiscfLocalFieldInput, XsphXsectPhiscfPoleEnergy,
    XsphXsectPhiscfPoleEnergyInput, XsphXsectPhiscfRadialContribution,
    XsphXsectPhiscfRadialContributionInput, XsphXsectPhiscfRadialSolverSetup,
    XsphXsectPhiscfRadialSolverSetupInput, XsphXsectPhiscfResponseContributionInput,
    XsphXsectPhiscfScreenedContributionsInput, XsphXsectPhiscfScreenedSolution,
    XsphXsectPhiscfScreenedSolutionInput, XsphXsectPhiscfWfirdcContribution,
    XsphXsectPhiscfWfirdcContributionInput, XsphXsectPhiscfWfirdcContributions,
    XsphXsectPhiscfWfirdcContributionsInput, XsphXsectPhiscfWorkspace, XsphXsectProjectedDensity,
    XsphXsectProjectedDensityInput, XsphXsectRadialPass, XsphXsectRadialPassInput,
    XsphXsectRadialPassKind, XsphXsectRegularChannel, XsphXsectRegularChannelInput,
    XsphXsectRegularSolution, XsphXsectRegularSolutionInput, XsphXsectScreenedField,
    XsphXsectScreenedFieldInput, XsphXsectScreenedFieldMode, XsphXsectSpinMerge,
    XsphXsectSpinMergeInput, XsphXsectTransition, XsphXsectTransitionPlan,
    XsphXsectTransitionPlanInput, XsphXsectWeightedRadialCrossIntegral,
    XsphXsectWeightedRadialCrossIntegralInput, XsphXsectWeightedRadialIntegral,
    XsphXsectWeightedRadialIntegralInput, xsph_angular_density_coefficients, xsph_axafs,
    xsph_bcoef_transition_indices, xsph_empty_cell_phase, xsph_even_energy_mesh,
    xsph_exafs_energy_grid_84, xsph_exponential_energy_mesh, xsph_fprime_energy_grid_84,
    xsph_hubbard_phase_assignments, xsph_hubbard_phase_potential_shifts,
    xsph_hubbard_phase_reference_tail, xsph_initial_hole_orbital, xsph_jas_bessel_functions,
    xsph_jas_orthogonality_correction, xsph_jas_overlap, xsph_jas_phase_energy_mesh,
    xsph_jas_radial_cross_integral, xsph_jas_radial_integral, xsph_jas_vertical_energy_mesh,
    xsph_k_energy_mesh, xsph_lj_needed_flags, xsph_longitudinal_multipole_factor,
    xsph_minimize_calculations, xsph_nrixs_transition_indices, xsph_nrixs_transition_weights,
    xsph_occupation_normalization, xsph_phase_angular_limit, xsph_phase_channel_plan,
    xsph_phase_cutoff, xsph_phase_energy_mesh_84, xsph_phase_energy_mesh_user,
    xsph_phase_energy_setup, xsph_phase_grid_preparation, xsph_phase_plasmon_pole_setup,
    xsph_phase_reference_tail, xsph_phase_self_energy_summary, xsph_q_bessel_table,
    xsph_radial_cross_integral, xsph_radial_integral, xsph_regular_phase,
    xsph_regular_phase_channel, xsph_relativistic_multipole_factors, xsph_reverse_energy_grid,
    xsph_sort_energy_grid, xsph_tdlda_angular_kernel, xsph_tdlda_broaden_channel_spectra,
    xsph_tdlda_channel_multipliers, xsph_tdlda_channel_spectra, xsph_tdlda_condition_response,
    xsph_tdlda_coulomb_fields, xsph_tdlda_decode_projector_selector, xsph_tdlda_direct_kernel,
    xsph_tdlda_energy_rows, xsph_tdlda_kramers_kronig_response,
    xsph_tdlda_nonlocal_exchange_integrals, xsph_tdlda_projected_kernel,
    xsph_tdlda_projector_orthogonalization, xsph_tdlda_radial_kernel_integrals,
    xsph_tdlda_raw_response, xsph_tdlda_row_wave_numbers, xsph_tdlda_screened_dipoles,
    xsph_tdlda_separation_function, xsph_tdlda_weight_response, xsph_tdlda_xsedge_rows,
    xsph_thermal_phase_energy_mesh, xsph_update_nrixs_atom_spectrum, xsph_update_nrixs_lg_spectrum,
    xsph_update_nrixs_lj_spectrum, xsph_vertical_energy_mesh_84, xsph_xanes_energy_grid_84,
    xsph_xes_energy_grid_84, xsph_xray_bessel_table, xsph_xsect_bcoef_central_cross_section,
    xsph_xsect_bcoef_cross_term_accumulation, xsph_xsect_bcoef_cross_term_state_accumulation,
    xsph_xsect_bcoef_direct_transition, xsph_xsect_bcoef_direct_transition_update,
    xsph_xsect_bcoef_nonstandard_channel_row, xsph_xsect_bcoef_nonstandard_energy_row,
    xsph_xsect_bcoef_ordinary_row, xsph_xsect_bcoef_standard_channel_row,
    xsph_xsect_bcoef_standard_energy_row,
    xsph_xsect_bcoef_standard_energy_row_with_transition_fields, xsph_xsect_bcoef_weights,
    xsph_xsect_central_cross_section, xsph_xsect_cross_term_accumulation,
    xsph_xsect_cross_term_plan, xsph_xsect_cross_term_state_reuse,
    xsph_xsect_cross_term_state_save, xsph_xsect_density_branch, xsph_xsect_direct_transition,
    xsph_xsect_embedded_density, xsph_xsect_energy_setup, xsph_xsect_fscf_integral,
    xsph_xsect_fscf_weights, xsph_xsect_hole_normalization, xsph_xsect_irregular_channel,
    xsph_xsect_irregular_initial_condition, xsph_xsect_irregular_transform,
    xsph_xsect_output_normalization, xsph_xsect_phiscf_accumulated_response,
    xsph_xsect_phiscf_angular_channels, xsph_xsect_phiscf_contribution_plan,
    xsph_xsect_phiscf_contribution_rule, xsph_xsect_phiscf_field_assembly,
    xsph_xsect_phiscf_irregular_seed, xsph_xsect_phiscf_linear_solve,
    xsph_xsect_phiscf_lipman_response, xsph_xsect_phiscf_local_field,
    xsph_xsect_phiscf_pole_energy, xsph_xsect_phiscf_radial_contribution,
    xsph_xsect_phiscf_radial_solver_setup, xsph_xsect_phiscf_screened_contributions,
    xsph_xsect_phiscf_screened_solution, xsph_xsect_phiscf_wfirdc_contribution,
    xsph_xsect_phiscf_wfirdc_contributions, xsph_xsect_projected_density, xsph_xsect_radial_pass,
    xsph_xsect_regular_channel, xsph_xsect_regular_solution, xsph_xsect_screened_field_setup,
    xsph_xsect_spin_merge, xsph_xsect_transition_plan, xsph_xsect_weighted_radial_cross_integral,
    xsph_xsect_weighted_radial_integral,
};

pub type Real = f64;
pub type Complex = Complex64;

pub type RealVec = Array1<Real>;
pub type RealMat = Array2<Real>;
pub type RealCube = Array3<Real>;
pub type RealArray4 = Array4<Real>;

pub type ComplexVec = Array1<Complex>;
pub type ComplexMat = Array2<Complex>;
pub type ComplexCube = Array3<Complex>;
pub type ComplexArray4 = Array4<Complex>;

pub fn zeros_real_vec(len: usize) -> RealVec {
    Array1::zeros(len)
}

pub fn zeros_complex_vec(len: usize) -> ComplexVec {
    Array1::zeros(len)
}

pub fn zeros_real_mat_fortran(rows: usize, cols: usize) -> RealMat {
    Array2::zeros((rows, cols).f())
}

pub fn zeros_complex_mat_fortran(rows: usize, cols: usize) -> ComplexMat {
    Array2::zeros((rows, cols).f())
}

pub fn zeros_real_cube_fortran(a: usize, b: usize, c: usize) -> RealCube {
    Array3::zeros((a, b, c).f())
}

pub fn zeros_complex_cube_fortran(a: usize, b: usize, c: usize) -> ComplexCube {
    Array3::zeros((a, b, c).f())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeffDimensions {
    pub nph: usize,
    pub ne: usize,
    pub nsp: usize,
    pub lmax: usize,
}

impl FeffDimensions {
    pub fn phase_shape(self) -> (usize, usize, usize, usize) {
        (self.ne, 2 * self.lmax + 1, self.nsp, self.nph + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_fortran_order_matrix_storage() {
        let mat = zeros_real_mat_fortran(3, 2);
        assert_eq!(mat.shape(), &[3, 2]);
        assert_eq!(mat.strides(), &[1, 3]);
    }

    #[test]
    fn computes_phase_shape_with_absorber_potential() {
        let dims = FeffDimensions {
            nph: 2,
            ne: 10,
            nsp: 1,
            lmax: 3,
        };
        assert_eq!(dims.phase_shape(), (10, 7, 1, 3));
    }
}
