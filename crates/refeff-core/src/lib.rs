#![forbid(unsafe_code)]

//! Core numerical types for the FEFF10 Rust port.
//!
//! `ndarray` is the primary storage and view API. Helpers in this crate create
//! arrays with FEFF-friendly Fortran-order layout where the original algorithms
//! and file formats depend on column-major traversal.

use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::Complex64;

pub mod angular;
pub mod atomic;
pub mod bessel;
pub mod compton;
pub mod configuration;
pub mod convolution;
pub mod core_hole;
pub mod debye;
pub mod density;
pub mod eels;
pub mod elam;
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
    AtomMathError, AtomicConvergenceMix, AtomicError, atomic_convergence_mix,
    atomic_direct_coulomb_coefficient, atomic_exchange_coulomb_coefficient,
    atomic_occupation_product, atomic_polynomial_product_coefficient, atomic_symbol, atomic_weight,
    nuclear_mass, thomas_fermi_density_potential,
};
pub use bessel::{
    BesselError, SphericalBessel, SphericalBesselValue, SphericalHankel, besjh, besjn, exjlnl,
    spherical_bessel_j_h, spherical_bessel_j_y,
};
pub use compton::{
    ComptonError, ComptonGrid, ComptonGridInput, ComptonProfileInput, ComptonRhoZzpInput,
    ComptonRhoZzpSlice, ComptonRotationAxisAngle, ComptonWindow, compton_build_grid,
    compton_cross_product, compton_jzzp, compton_profile, compton_profiles, compton_rhozzp_slice,
    compton_rotate_vector, compton_rotate_vector_in_place, compton_rotation_axis_angle,
    compton_rotation_matrix,
};
pub use configuration::{
    FEFF_KAPPA_PROJECTION_COUNT, FEFF_ORBITAL_KAPPAS, FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS,
    FEFF_ORBITAL_SLOT_COUNT, OrbitalConfiguration, OrbitalConfigurationError,
    OrbitalConfigurationInput, orbital_configuration,
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
    DmdwSpectralFunctionGrid, DmdwType2AtomGroup, MorseCumulants, ThermalExpansionCumulants,
    classical_debye_correlation, classical_debye_waller_factor, dmdw_center_of_mass,
    dmdw_debye_waller_factors_from_poles, dmdw_expand_path_descriptor,
    dmdw_expand_path_descriptors, dmdw_inertia_tensor, dmdw_ir_dipole_seed_vector,
    dmdw_lanczos_coefficients, dmdw_lanczos_pole_spectrum, dmdw_lanczos_pole_spectrum_with_search,
    dmdw_lanczos_r_polynomial, dmdw_lanczos_s_polynomial, dmdw_lanczos_s_polynomial_derivative,
    dmdw_mass_weighted_dynamical_matrix, dmdw_moment_summaries_from_poles,
    dmdw_normalize_seed_vector, dmdw_path_motion, dmdw_phonon_coupling, dmdw_pole_weighted_a2f,
    dmdw_project_seed_vector, dmdw_rigid_body_projection_modes, dmdw_self_energy_from_a2f_poles,
    dmdw_self_energy_grid_from_a2f_poles, dmdw_single_pole_einstein_summary,
    dmdw_spectral_function_from_a2f_poles, dmdw_type2_pole_weighted_a2f,
    dmdw_vibrational_free_energy_from_poles, morse_einstein_cumulants, quantum_debye_correlation,
    quantum_debye_waller_factor, thermal_expansion_cumulants,
};
pub use density::{
    BroydenMix, BroydenMixInput, BroydenWorkspace, CoulombPotentialUpdate,
    CoulombPotentialUpdateInput, CoulombUpdateMode, DensityError, PotentialOverlap,
    PotentialOverlapInput, PotentialOverlapNeighbor, ValenceDensityUpdate,
    ValenceDensityUpdateInput, mix_broyden_density, overlap_potential_density,
    update_coulomb_potential, update_valence_density,
};
pub use eels::{
    EelsAngularMesh, EelsError, EelsIntegrationMesh, EelsMeshInput, EelsMeshMode, EelsMeshSetup,
    FEFF_ELECTRON_REST_ENERGY_EV, FEFF_H_ON_SQRT_TWO_ME, eels_angular_mesh,
    eels_euler_rotation_matrix, eels_integration_mesh, eels_mesh_setup,
    electron_wavelength_atomic_units,
};
pub use elam::{
    ELAM_EDGE_ATOMIC_NUMBER_MAX, ELAM_EDGE_HOLE_COUNT, ELAM_NEXT_EDGE_SENTINEL_HARTREE,
    ElamEdgeEnergy, ElamError, elam_component_edge_energies_hartree, elam_edge_energy_ev,
    elam_edge_energy_hartree, next_elam_edge_hartree, previous_elam_edge_hartree,
};
pub use exchange::{
    ExchangeCorrelation, ExchangeError, HedinLundqvistImaginary, HedinLundqvistSelfEnergy,
    KsdTFreeEnergy, KsdTSpin, dirac_hara_exchange_potential, hedin_lundqvist_ffq,
    hedin_lundqvist_imaginary_self_energy, hedin_lundqvist_self_energy,
    karasiev_sjostrom_dufty_trickey_free_energy, karasiev_sjostrom_dufty_trickey_internal_energy,
    karasiev_sjostrom_dufty_trickey_vxc, perdew_zunger_exchange_correlation, perdew_zunger_vxc,
    perrot_dharma_wardana_reduced_vxc, perrot_dharma_wardana_vxc, quinn_imaginary_self_energy,
    von_barth_hedin_potential,
};
pub use fms::{
    FmsAtom, FmsBiCgStabInput, FmsBiCgStabResult, FmsError, FmsFreePropagatorInput,
    FmsFreePropagatorMatrixInput, FmsFullPotentialLuInput, FmsFullPotentialLuResult,
    FmsGravesMorrisInput, FmsGravesMorrisResult, FmsIterativeSystemInput, FmsLuInput, FmsLuResult,
    FmsPairTables, FmsRecursionInput, FmsRecursionResult, FmsRotationDirection, FmsTMatrixInput,
    FmsTMatrixTableInput, FmsTfqmrInput, FmsTfqmrResult, fms_bicgstab_scattering,
    fms_free_propagator_element, fms_free_propagator_matrix, fms_full_potential_lu_scattering,
    fms_graves_morris_scattering, fms_iterative_system_matrix, fms_lu_scattering, fms_pair_tables,
    fms_recursion_scattering, fms_rotation_matrix, fms_t_matrix_element, fms_t_matrix_table,
    fms_tfqmr_scattering, pair_polar_angles, rehr_albers_polynomials,
    rehr_albers_z_axis_propagator, sort_atoms_by_radius, sort_representative_atoms,
};
pub use fovrg::{
    FovrgAngularCoefficientsInput, FovrgC3DerivativeInput, FovrgDiracSolution,
    FovrgDiracSolverInput, FovrgError, FovrgExchangePotential, FovrgExchangePotentialInput,
    FovrgFlatPotentialInput, FovrgFlatPotentialPropagation, FovrgInitialPhotoelectron,
    FovrgInitialPhotoelectronInput, FovrgInwardSolution, FovrgInwardSolutionInput,
    FovrgNuclearPotential, FovrgNuclearPotentialInput, FovrgOrbitalSetup, FovrgOrbitalSetupInput,
    FovrgOrthogonalization, FovrgOrthogonalizationInput, FovrgOutgoingSolution,
    FovrgOutgoingSolutionInput, FovrgOutwardIntegration, FovrgOutwardIntegrationInput,
    FovrgOverlapIntegralInput, FovrgPotentialDevelopment, FovrgPotentialDevelopmentInput,
    FovrgYkZkExchangeInput, FovrgYkZkTransform, FovrgYkZkTransformInput,
    fovrg_angular_coefficients, fovrg_c3_derivative, fovrg_complex_real_product_coefficient,
    fovrg_dirac_solver, fovrg_exchange_potential, fovrg_flat_potential_propagate,
    fovrg_initial_photoelectron, fovrg_inward_solution, fovrg_nuclear_potential,
    fovrg_orbital_setup, fovrg_outgoing_solution, fovrg_outward_integrate, fovrg_overlap_integral,
    fovrg_potential_development, fovrg_real_product_coefficient, fovrg_schmidt_orthogonalize,
    fovrg_yk_zk_exchange, fovrg_yk_zk_transform,
};
pub use fprime::{
    FprimeContourIntegralInput, FprimeError, FprimeLogCase, FprimePositiveAxisIntegralInput,
    fprime_contour_integral, fprime_log_correction, fprime_positive_axis_integral,
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
    CurvedWavePolynomialInput, EnergyIndependentMatrixInput, GenfmtError,
    GenfmtLegendreNormalizationInput, InitialStateRotation, InitialStateRotationInput,
    LambdaIndexInput, LambdaIndexSet, PathRotationAngles, PathRotationInput,
    PolarizedScatteringAmplitudeInput, ScatteringAmplitudeMatrixInput, TransitionRotationInput,
    XStarInput, curved_wave_polynomials, energy_independent_transition_matrix,
    genfmt_legendre_normalization_table, initial_state_rotation, lambda_indices,
    path_rotation_angles, polarized_scattering_amplitude_matrix, scattering_amplitude_matrix,
    xstar,
};
pub use grid::{
    CoulombPotentialSlw, CoulombPotentialSlwInput, DiracSpinorGrid, DiracSpinorGridInput,
    DiracSpinorOrbitalsGrid, DiracSpinorOrbitalsGridInput, FEFF_FERMI_MOMENTUM_FACTOR,
    FEFF_HARTREE_EV, FermiLevel, FermiLevelInput, GridError, InterstitialShellValues,
    InterstitialShellValuesInput, LOUCKS_DELTA, LOUCKS_X_OFFSET, LoucksSphericalOverlap,
    LoucksSphericalOverlapInput, MuffinTinOverlapMatrix, MuffinTinOverlapMatrixInput,
    MuffinTinOverlapNeighbor, MuffinTinOverlapProjection, MuffinTinOverlapProjectionInput,
    MuffinTinOverlapProjectionMode, NormanRadius, NormanRadiusInput, OverlapDensityIndices,
    OverlapDensityIndicesInput, PotentialGrid, PotentialGridInput, ScmtEnergyGrid,
    ScmtEnergyGridInput, coulomb_potential_slw, fix_dirac_spinor_grid,
    fix_dirac_spinor_orbitals_grid, fix_potential_grid, interstitial_fermi_level,
    interstitial_shell_values, loucks_index_below, loucks_radius, loucks_x,
    muffin_tin_overlap_matrix, norman_radius_from_density, overlap_density_indices,
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
    BravaisLattice, KMeshArbitraryMesh, KMeshBravaisBasis, KMeshDivisionReduction, KMeshDivisions,
    KMeshReduction, KMeshTetrahedronRecords, KPath, KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE,
    KSpaceError, PointGroup, ReducedVector, SymmetryCheck, bravais_lattice, bravais_lattice_index,
    change_cartesian_basis, define_k_path, kmesh_arbitrary_mesh, kmesh_basis_divisions,
    kmesh_bravais_basis, kmesh_tetrahedron_division, kmesh_tetrahedron_records,
    point_group_operations, reciprocal_lattice_vectors, reciprocal_metric,
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
    PathCriteriaDecisionInput, PathError, PathGeometry, PathOutputCriterion,
    PathOutputCriterionInput, PathOutputImportance, PathOutputImportanceInput,
    PathOutputParameters, PathPhaseCriteriaInput, PathPhaseCriteriaTables, PathStandardCoordinates,
    PathStandardCoordinatesInput, pack_path_indices, path_beta_indices,
    path_canonical_representation, path_criteria_decision, path_degeneracy_hash, path_geometry,
    path_heap_bubble_down, path_heap_bubble_up, path_heap_criterion, path_output_criterion,
    path_output_importance, path_output_parameters, path_phase_criteria_tables,
    path_standard_coordinates, unpack_path_indices,
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
    RhorrpAtomicDensityInput, RhorrpDensityGridEvaluation, RhorrpDensityGridInput,
    RhorrpDensityGridPoints, RhorrpDensityIntegrationInput, RhorrpEnergyDensityInput,
    RhorrpEnergyPrefactorInput, RhorrpError, RhorrpFermiDistributionInput, RhorrpFmsInclusionInput,
    RhorrpIrregularFixInput, RhorrpNearestAtom, RhorrpNearestAtomInput, RhorrpNearestAtomTable,
    RhorrpNearestAtomTableInput, RhorrpPairDensityInput, RhorrpPairEnergyDensityInput,
    RhorrpProcessRange, RhorrpRadialInterpolationInput, RhorrpRadialInterpolationLocation,
    RhorrpSameSiteGreenInput, RhorrpScatteringGreenInput, RhorrpWavefunctionInterpolationInput,
    rhorrp_atomic_density, rhorrp_density_grid_points, rhorrp_energy_prefactor,
    rhorrp_evaluate_density_grid, rhorrp_fermi_distribution, rhorrp_finish_energy_density,
    rhorrp_fix_irregular_origin, rhorrp_fms_inclusion_counts, rhorrp_integrate_density,
    rhorrp_interpolate_wavefunction, rhorrp_nearest_atom, rhorrp_nearest_atom_table,
    rhorrp_next_index_1based, rhorrp_pair_density, rhorrp_pair_energy_density,
    rhorrp_point_at_index, rhorrp_process_ranges, rhorrp_radial_interpolation_location,
    rhorrp_same_site_green, rhorrp_scattering_green,
};
pub use rixs::{RixsError, bilinear_interpolate_complex, integrated_double_lorentz, kk_integral};
pub use roots::{
    ComplexRoots, RealPolynomialRoots, RootError, cubic_zeros, depressed_quartic_roots,
    quadratic_zeros, real_polynomial_roots,
};
pub use screen::{
    ScreenContourEnergyGrid, ScreenContourEnergyGridInput, ScreenError, screen_contour_energy_grid,
    screen_exponential_energy_grid, screen_lda_exchange_correlation_kernel, screen_radial_grid,
    screen_radial_index_1based,
};
pub use self_energy::{
    CgratrIntegral, ExcitationPole, SelfEnergyError, SelfEnergyIntegrandInput, SingularityFunction,
    cgratr, find_self_energy_singularities, gamma_q, hartree_fock_exchange, log_i,
    make_excitation_poles, omega_q, self_energy_dr1_integrand, self_energy_dr2_integrand,
    self_energy_dr3_integrand, self_energy_pole_dispersion, self_energy_r1_integrand,
    self_energy_r2_integrand, self_energy_r3_integrand,
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
    SfconvSo2convSelfEnergySampleInput, SfconvSo2convXanesPreparation,
    SfconvSo2convXanesPreparationInput, SfconvSpectralCell, SfconvSpectralCellInput,
    SfconvSpectralEnergyGrid, SfconvSpectralInterpolation, SfconvSpectralInterpolationInput,
    SfconvSpectralTable, SfconvSpectralTableInput, SfconvSpectralWeightsInput,
    SfconvXanesConvolution, SfconvXanesConvolutionInput, sfconv_broadened_self_energy,
    sfconv_broadened_self_energy_derivative, sfconv_broadened_self_energy_derivative_integrands,
    sfconv_broadened_self_energy_integrands, sfconv_convolve, sfconv_correct_satellite_weights,
    sfconv_coupling_potential_squared, sfconv_exafs_convolution, sfconv_exponential_reduction,
    sfconv_extrinsic_beta, sfconv_extrinsic_satellite, sfconv_extrinsic_satellite_broadened,
    sfconv_extrinsic_satellite_debroadened, sfconv_feff_path_signal, sfconv_find_singularities,
    sfconv_free_electron_exchange, sfconv_grater_integrate, sfconv_imaginary_self_energy,
    sfconv_imaginary_self_energy_derivative, sfconv_interference_quasiparticle,
    sfconv_interference_quasiparticle_integrand, sfconv_interference_satellite,
    sfconv_interference_satellite_integrand, sfconv_interpolate_feff_path,
    sfconv_interpolate_momentum_spectral_function, sfconv_interpolate_spectral_function,
    sfconv_intrinsic_satellite, sfconv_intrinsic_satellite_integrand,
    sfconv_inverse_pole_dispersion, sfconv_kramers_kronig_real_part, sfconv_path_average,
    sfconv_plasma_parameters, sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion,
    sfconv_pole_dispersion_derivative, sfconv_pole_dispersion_second_derivative, sfconv_q_limits,
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
    sfconv_so2conv_prepare_xanes_signal, sfconv_so2conv_unbroadened_self_energy_grid,
    sfconv_so2conv_unbroadened_self_energy_sample, sfconv_spectral_cell,
    sfconv_spectral_energy_grid, sfconv_spectral_table, sfconv_spectral_weights,
    sfconv_split_extrinsic_satellite, sfconv_xanes_convolution,
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
    XSPH_AXAFS_COLUMN_COUNT, XsphAxafs, XsphAxafsInput, XsphCalculationPlan, XsphError,
    XsphFprimeEnergyGrid84, XsphHoleOrbital, XsphHoleOrbitalInput, XsphLgSpectrumUpdateInput,
    XsphLjSpectrumUpdateInput, XsphPhaseEnergyMesh84, XsphPhaseEnergyMesh84Input,
    XsphPhaseUserGridInput, XsphPhaseUserGridKind, XsphPhaseUserGridMinimum,
    XsphPhaseUserGridRecord, XsphPhaseUserRegularGrid, XsphRelativisticMultipoleFactors,
    XsphSortedEnergyGrid, XsphSpectrumUpdateMode, XsphThermalPhaseEnergyMesh,
    XsphThermalPhaseEnergyMeshInput, XsphXanesEnergyGrid84, XsphXesEnergyGrid84,
    xsph_angular_density_coefficients, xsph_axafs, xsph_even_energy_mesh,
    xsph_exafs_energy_grid_84, xsph_exponential_energy_mesh, xsph_fprime_energy_grid_84,
    xsph_initial_hole_orbital, xsph_k_energy_mesh, xsph_lj_needed_flags,
    xsph_longitudinal_multipole_factor, xsph_minimize_calculations, xsph_nrixs_transition_weights,
    xsph_occupation_normalization, xsph_phase_energy_mesh_84, xsph_phase_energy_mesh_user,
    xsph_q_bessel_table, xsph_relativistic_multipole_factors, xsph_reverse_energy_grid,
    xsph_sort_energy_grid, xsph_thermal_phase_energy_mesh, xsph_update_nrixs_atom_spectrum,
    xsph_update_nrixs_lg_spectrum, xsph_update_nrixs_lj_spectrum, xsph_vertical_energy_mesh_84,
    xsph_xanes_energy_grid_84, xsph_xes_energy_grid_84,
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
