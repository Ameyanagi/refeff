use ndarray::{Array2, Array3, Array4, ArrayView2};

use super::*;

const SPRING_DEFAULT_DISTANCE_TOLERANCE_PERCENT: Real = 2.0;
const SPRING_AMU0: Real = 1.660_54;
const SPRING_RM_SIGMA_FACTOR: Real = 3.1746;
const SPRING_RM_TEMPERATURE_FACTOR: Real = 7.6383;
const SPRING_PATH_MATCH_SCALE: Real = 100.0;
const SPRING_PATH_MATCH_TOLERANCE: Real = 1.0;
const SPRING_EM_MAX_GRID_POINTS: usize = 700;
const SPRING_EM_LOW_FREQUENCY: Real = 0.000_000_1;
const SPRING_EM_FREQUENCY_STEP: Real = 0.01;
const SPRING_EM_TIME_SAMPLES_PER_PERIOD: Real = 15.0;
const SPRING_EM_THERMAL_FACTOR: Real = 187.64;
const SPRING_EM_SIGMA_FACTOR: Real = 0.258_792_6;

#[derive(Debug, Clone, PartialEq)]
pub struct SpringInput {
    pub resolution: Real,
    pub cutoff: Real,
    pub max_frequency: Real,
    pub dos_fit: Real,
    pub print_projected_dos: i32,
    pub stretches: Vec<SpringStretch>,
    pub angles: Vec<SpringAngle>,
}

impl Default for SpringInput {
    fn default() -> Self {
        Self {
            resolution: 0.05,
            cutoff: 3.0,
            max_frequency: 1.0,
            dos_fit: 0.0,
            print_projected_dos: 0,
            stretches: Vec::new(),
            angles: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringStretch {
    pub first_atom: usize,
    pub second_atom: usize,
    pub force_constant: Real,
    pub distance_tolerance_percent: Real,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringAngle {
    pub first_atom: usize,
    pub center_atom: usize,
    pub third_atom: usize,
    pub force_constant: Real,
    pub angle_tolerance_percent: Real,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpringDynamicalMatrix {
    pub atom_positions_angstrom: Array2<Real>,
    pub atomic_numbers: Vec<usize>,
    pub potential_indices: Vec<usize>,
    pub matrix: Array4<Real>,
    pub pair_directions: Array3<Real>,
    pub interaction_radius_angstrom: Real,
    pub characteristic_frequency: Real,
    pub first_shell_coordination: Real,
}

#[derive(Debug, Clone, Copy)]
pub struct SpringDynamicalMatrixInput<'a> {
    pub spring: &'a SpringInput,
    pub atom_positions_angstrom: ArrayView2<'a, Real>,
    pub atomic_numbers: &'a [usize],
    pub potential_indices: &'a [usize],
    pub absorber_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpringRecursionState {
    pub max_sigma2: Real,
    pub pair_sigma2: Array2<Real>,
}

impl SpringRecursionState {
    #[must_use]
    pub fn new(potential_count: usize) -> Self {
        Self {
            max_sigma2: 0.0,
            pair_sigma2: Array2::zeros((potential_count, potential_count)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpringRecursionInput<'a> {
    pub matrix: &'a SpringDynamicalMatrix,
    pub temperature: Real,
    pub path_positions_angstrom: ArrayView2<'a, Real>,
    pub state: Option<&'a SpringRecursionState>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringRecursionResult {
    pub sigma2: Real,
    pub reduced_mass: Real,
    pub einstein_frequency: Real,
    pub two_pole_frequencies: Option<[Real; 2]>,
    pub two_pole_weights: Option<[Real; 2]>,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SpringEquationOfMotionInput<'a> {
    pub matrix: &'a SpringDynamicalMatrix,
    pub spring: &'a SpringInput,
    pub temperature: Real,
    pub path_positions_angstrom: ArrayView2<'a, Real>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringEquationOfMotionResult {
    pub sigma2: Real,
    pub reduced_mass: Real,
    pub density_normalization: Real,
    pub normalization_check_percent: Real,
    pub moment_frequency: Real,
    pub capped: bool,
}

pub fn parse_spring_input(text: &str) -> Result<SpringInput, DebyeError> {
    let mut input = SpringInput::default();
    let mut mode = SpringParseMode::Cards;

    for raw_line in text.lines() {
        let line = spring_data_line(raw_line);
        if line.is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        let keyword = fields[0].to_ascii_uppercase();
        let token = spring_keyword(&keyword);

        match mode {
            SpringParseMode::Cards => match token {
                Some(SpringKeyword::Stretches) => mode = SpringParseMode::Stretches,
                Some(SpringKeyword::Angles) => mode = SpringParseMode::Angles,
                Some(SpringKeyword::Vdos) => parse_spring_vdos(&mut input, &fields)?,
                Some(SpringKeyword::Prdos) => {
                    input.print_projected_dos = fields
                        .get(1)
                        .map_or(Ok(1), |value| parse_spring_i32(value))?;
                }
                Some(SpringKeyword::End) => break,
                None => {
                    return Err(DebyeError::InvalidSpringInput {
                        reason: "unknown card",
                    });
                }
            },
            SpringParseMode::Stretches => {
                if let Some(keyword) = token {
                    mode = SpringParseMode::Cards;
                    match keyword {
                        SpringKeyword::Stretches => mode = SpringParseMode::Stretches,
                        SpringKeyword::Angles => mode = SpringParseMode::Angles,
                        SpringKeyword::Vdos => parse_spring_vdos(&mut input, &fields)?,
                        SpringKeyword::Prdos => {
                            input.print_projected_dos = fields
                                .get(1)
                                .map_or(Ok(1), |value| parse_spring_i32(value))?;
                        }
                        SpringKeyword::End => break,
                    }
                } else {
                    input.stretches.push(parse_spring_stretch(&fields)?);
                }
            }
            SpringParseMode::Angles => {
                if let Some(keyword) = token {
                    mode = SpringParseMode::Cards;
                    match keyword {
                        SpringKeyword::Stretches => mode = SpringParseMode::Stretches,
                        SpringKeyword::Angles => mode = SpringParseMode::Angles,
                        SpringKeyword::Vdos => parse_spring_vdos(&mut input, &fields)?,
                        SpringKeyword::Prdos => {
                            input.print_projected_dos = fields
                                .get(1)
                                .map_or(Ok(1), |value| parse_spring_i32(value))?;
                        }
                        SpringKeyword::End => break,
                    }
                } else {
                    input.angles.push(parse_spring_angle(&fields)?);
                }
            }
        }
    }

    validate_spring_input(&input)?;
    Ok(input)
}

pub fn spring_dynamical_matrix(
    input: SpringDynamicalMatrixInput<'_>,
) -> Result<SpringDynamicalMatrix, DebyeError> {
    validate_spring_matrix_input(input)?;

    let atom_count = input.atom_positions_angstrom.nrows();
    let mut stretch_constants = Array2::<Real>::zeros((atom_count, atom_count));
    let mut distance_tolerances = Array2::<Real>::from_elem(
        (atom_count, atom_count),
        SPRING_DEFAULT_DISTANCE_TOLERANCE_PERCENT / 100.0,
    );
    let mut min_stretch = Real::INFINITY;
    let mut min_stretch_pair = None;

    for stretch in &input.spring.stretches {
        validate_spring_atom_index(stretch.first_atom, atom_count)?;
        validate_spring_atom_index(stretch.second_atom, atom_count)?;
        stretch_constants[(stretch.first_atom, stretch.second_atom)] = stretch.force_constant;
        let tolerance = stretch.distance_tolerance_percent.abs() / 100.0;
        distance_tolerances[(stretch.first_atom, stretch.second_atom)] = tolerance;
        distance_tolerances[(stretch.second_atom, stretch.first_atom)] = tolerance;
        if stretch.force_constant < min_stretch {
            min_stretch = stretch.force_constant;
            min_stretch_pair = Some((stretch.first_atom, stretch.second_atom));
        }
    }

    expand_stretches(input, &mut stretch_constants, &mut distance_tolerances)?;

    let expanded_angles = expand_angles(input, &distance_tolerances)?;
    let mut pair_directions = Array3::<Real>::zeros((3, atom_count, atom_count));
    let mut stretch_matrix = Array4::<Real>::zeros((3, 3, atom_count, atom_count));
    let mut angle_matrix = Array4::<Real>::zeros((3, 3, atom_count, atom_count));
    let mut matrix = Array4::<Real>::zeros((3, 3, atom_count, atom_count));
    let interaction_radius = spring_interaction_radius(
        input.atom_positions_angstrom,
        &stretch_constants,
        &distance_tolerances,
    )?;

    fill_pair_directions(input.atom_positions_angstrom, &mut pair_directions)?;
    build_angle_matrix(
        input.atom_positions_angstrom,
        &expanded_angles,
        interaction_radius,
        &mut angle_matrix,
    )?;
    build_stretch_matrix(&stretch_constants, &pair_directions, &mut stretch_matrix);
    build_mass_weighted_matrix(
        &stretch_matrix,
        &angle_matrix,
        input.atomic_numbers,
        &mut matrix,
    )?;
    let characteristic_frequency = spring_characteristic_frequency(
        input,
        &stretch_constants,
        &pair_directions,
        &matrix,
        &distance_tolerances,
        min_stretch,
        min_stretch_pair,
    )?;
    let first_shell_coordination = spring_first_shell_coordination(
        input.atom_positions_angstrom,
        input.absorber_index,
        &distance_tolerances,
    )?;

    Ok(SpringDynamicalMatrix {
        atom_positions_angstrom: input.atom_positions_angstrom.to_owned(),
        atomic_numbers: input.atomic_numbers.to_vec(),
        potential_indices: input.potential_indices.to_vec(),
        matrix,
        pair_directions,
        interaction_radius_angstrom: interaction_radius,
        characteristic_frequency,
        first_shell_coordination,
    })
}

pub fn equation_of_motion_debye_waller_factor(
    input: SpringEquationOfMotionInput<'_>,
) -> Result<SpringEquationOfMotionResult, DebyeError> {
    ensure_nonnegative("tk", input.temperature)?;
    validate_spring_path(input.path_positions_angstrom)?;
    ensure_positive(
        "spring EM first-shell coordination",
        input.matrix.first_shell_coordination,
    )?;
    let setup = spring_path_setup(input.matrix, input.path_positions_angstrom)?;
    let w0 = input.matrix.characteristic_frequency;
    if w0 <= 0.0 || !w0.is_finite() {
        return Err(DebyeError::NonPositiveSpringFrequency { value: w0 });
    }

    let frequency_scale = input.matrix.first_shell_coordination.sqrt();
    let time_step =
        2.0 * std::f64::consts::PI / frequency_scale / SPRING_EM_TIME_SAMPLES_PER_PERIOD;
    let cutoff =
        2.0 * (2.0 * input.spring.cutoff).sqrt() / input.spring.resolution / frequency_scale;
    let time_steps = (cutoff / time_step) as usize;
    if time_steps == 0 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "spring EM time grid is empty",
        });
    }
    let damping_lambda = input.spring.cutoff / cutoff.powi(2);
    let max_frequency = input.spring.max_frequency * frequency_scale;
    let mut frequency_step = SPRING_EM_FREQUENCY_STEP;
    let mut frequency_points =
        ((max_frequency - SPRING_EM_LOW_FREQUENCY) / frequency_step + 1.0) as usize;
    if frequency_points > SPRING_EM_MAX_GRID_POINTS {
        frequency_points = SPRING_EM_MAX_GRID_POINTS;
        frequency_step = (max_frequency - SPRING_EM_LOW_FREQUENCY)
            / (frequency_points.saturating_sub(1) as Real);
    }
    if frequency_points < 3 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "spring EM frequency grid is too small",
        });
    }
    let fit_points = ((input.spring.dos_fit * frequency_points as Real) / 20.0) as usize;
    let frequencies = (0..frequency_points)
        .map(|index| SPRING_EM_LOW_FREQUENCY + index as Real * frequency_step)
        .collect::<Vec<_>>();
    let mut density = vec![0.0; frequency_points];
    let mut displacement = setup.initial_vector.clone();
    let mut previous = displacement.clone();
    let mut force = Array2::<Real>::zeros((3, input.matrix.atom_positions_angstrom.nrows()));
    let nonzero_pairs = spring_nonzero_upper_pairs(input.matrix);
    let time_step2 = time_step * time_step;
    let mut time = time_step / 2.0;

    for step in 1..=time_steps {
        let gaussian = (-damping_lambda * time * time).exp();
        let mut overlap = 0.0;
        for atom in 0..input.matrix.atom_positions_angstrom.nrows() {
            for axis in 0..3 {
                overlap += displacement[(axis, atom)] * setup.initial_vector[(axis, atom)];
            }
        }
        overlap *= gaussian;
        for index in 0..frequency_points {
            density[index] += overlap * (frequencies[index] * time).cos() * time_step;
        }
        if step != time_steps {
            force.fill(0.0);
            let inverse_w0_squared = 1.0 / w0.powi(2);
            for &(left, right) in &nonzero_pairs {
                for axis1 in 0..3 {
                    for axis2 in 0..3 {
                        force[(axis1, left)] -= input.matrix.matrix[(axis1, axis2, left, right)]
                            * displacement[(axis2, right)]
                            * inverse_w0_squared;
                        if left != right {
                            force[(axis1, right)] -= input.matrix.matrix
                                [(axis1, axis2, right, left)]
                                * displacement[(axis2, left)]
                                * inverse_w0_squared;
                        }
                    }
                }
            }
            for atom in 0..input.matrix.atom_positions_angstrom.nrows() {
                for axis in 0..3 {
                    let next = 2.0 * displacement[(axis, atom)] - previous[(axis, atom)]
                        + time_step2 * force[(axis, atom)];
                    previous[(axis, atom)] = displacement[(axis, atom)];
                    displacement[(axis, atom)] = next;
                }
            }
        }
        time += time_step;
    }

    spring_fit_low_frequency_density(&mut density, &frequencies, fit_points);

    let last = frequency_points - 1;
    density[last] = 0.0;
    if density[0] < 0.0 {
        density[0] = 0.0;
    }
    let mut integral = (density[0] + density[last]) * frequency_step / 2.0;
    for value in density.iter_mut().take(last).skip(1) {
        if *value < 0.0 {
            *value = 0.0;
        }
        integral += *value * frequency_step;
    }
    ensure_positive("spring EM density integral", integral)?;
    let normalization = 1.0 / integral;
    let check =
        ((2.0 / std::f64::consts::PI - normalization) / (2.0 / std::f64::consts::PI)).abs() * 100.0;
    let coefficient = normalization * 0.5 * SPRING_EM_SIGMA_FACTOR / setup.reduced_mass / w0;
    let reduced_temperature = input.temperature / SPRING_EM_THERMAL_FACTOR / w0;
    let mut sigma2 = 0.0;
    for index in 1..last {
        let coth = if reduced_temperature == 0.0 {
            1.0
        } else {
            1.0 / (frequencies[index] / (2.0 * reduced_temperature)).tanh()
        };
        sigma2 += coefficient * density[index] * coth * frequency_step / frequencies[index];
    }
    ensure_finite_output("spring EM sigma2", sigma2)?;
    let capped = sigma2 > 1.0;
    if capped {
        sigma2 = 1.0;
    }
    Ok(SpringEquationOfMotionResult {
        sigma2,
        reduced_mass: setup.reduced_mass,
        density_normalization: normalization,
        normalization_check_percent: check,
        moment_frequency: setup.moment_frequency,
        capped,
    })
}

pub(crate) fn spring_fit_low_frequency_density(
    density: &mut [Real],
    frequencies: &[Real],
    fit_points: usize,
) {
    let fit_points = fit_points.min(density.len()).min(frequencies.len());
    if fit_points == 0 {
        return;
    }
    // FEFF's nfit is a 1-based Fortran index: gr(nfit) provides the
    // coefficient used to replace gr(1:nfit). Convert that sample to its
    // zero-based Rust index while preserving the same replaced span.
    let fit_index = fit_points - 1;
    let fit = density[fit_index] / frequencies[fit_index].powi(4);
    for index in 0..fit_points {
        density[index] = fit * frequencies[index].powi(4);
    }
}

pub fn recursion_debye_waller_factor(
    input: SpringRecursionInput<'_>,
) -> Result<SpringRecursionResult, DebyeError> {
    ensure_nonnegative("tk", input.temperature)?;
    validate_spring_path(input.path_positions_angstrom)?;

    let matrix = input.matrix;
    let setup = spring_path_setup(matrix, input.path_positions_angstrom)?;

    let w0 = matrix.characteristic_frequency;
    if w0 <= 0.0 || !w0.is_finite() {
        return Err(DebyeError::NonPositiveSpringFrequency { value: w0 });
    }
    let wnorm = 100.0 * w0 / (SPRING_AMU0 * 10.0).sqrt();
    let moment0 = (setup.moment_frequency / wnorm).powi(2);
    let einstein_frequency = setup.moment_frequency;
    if einstein_frequency < 1.0 {
        let sigma2 = spring_recursion_fallback_sigma2(input.state, matrix, setup.nconv[1])?;
        return Ok(SpringRecursionResult {
            sigma2,
            reduced_mass: setup.reduced_mass,
            einstein_frequency,
            two_pole_frequencies: None,
            two_pole_weights: None,
            fallback_used: true,
        });
    }

    let atom_count = matrix.atom_positions_angstrom.nrows();
    let mut q1 = Array2::<Real>::zeros((3, atom_count));
    for &atom in &setup.neighborhood {
        for axis in 0..3 {
            let mut q1i = 0.0;
            for &source in &setup.path_unique {
                for source_axis in 0..3 {
                    q1i += matrix.matrix[(axis, source_axis, atom, source)]
                        * setup.initial_vector[(source_axis, source)]
                        / w0
                        / w0;
                }
            }
            q1[(axis, atom)] = q1i - moment0 * setup.initial_vector[(axis, atom)];
        }
    }

    let mut b0 = 0.0;
    for atom in 0..atom_count {
        for axis in 0..3 {
            b0 += q1[(axis, atom)].powi(2);
        }
    }
    ensure_positive("spring RM b0", b0)?;

    let mut moment1 = 0.0;
    for &atom in &setup.neighborhood {
        for axis in 0..3 {
            let mut q2 = 0.0;
            for &source in &setup.neighborhood {
                for source_axis in 0..3 {
                    q2 += matrix.matrix[(axis, source_axis, atom, source)]
                        * q1[(source_axis, source)]
                        / w0
                        / w0;
                }
            }
            moment1 += q1[(axis, atom)] * q2;
        }
    }

    let a0 = moment0 * wnorm.powi(2);
    let a1 = (moment1 / b0) * wnorm.powi(2);
    let b0_scaled = b0 * wnorm.powi(4);
    let discriminant = (a0 + a1).powi(2) - 4.0 * (a0 * a1 - b0_scaled);
    if discriminant < 0.0 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "negative RM discriminant",
        });
    }
    let root = discriminant.sqrt();
    let x1 = (a0 + a1 + root) / 2.0;
    let x2 = (a0 + a1 - root) / 2.0;
    let denominator = x1 - x2;
    if denominator == 0.0 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "degenerate RM poles",
        });
    }
    let weight2 = (a1 - x2) / denominator;
    let weight1 = (x1 - a1) / denominator;
    let w1 = x1.sqrt();
    let w2 = x2.sqrt();
    ensure_positive("spring RM w1", w1)?;
    ensure_positive("spring RM w2", w2)?;

    let s1 = spring_recursion_sigma_component(setup.reduced_mass, w1, input.temperature);
    let s2 = spring_recursion_sigma_component(setup.reduced_mass, w2, input.temperature);
    let sigma2 = weight1 * s1 + weight2 * s2;
    ensure_finite_output("spring RM sigma2", sigma2)?;

    Ok(SpringRecursionResult {
        sigma2,
        reduced_mass: setup.reduced_mass,
        einstein_frequency,
        two_pole_frequencies: Some([w1, w2]),
        two_pole_weights: Some([weight1, weight2]),
        fallback_used: false,
    })
}

pub fn update_spring_recursion_state(
    state: &mut SpringRecursionState,
    matrix: &SpringDynamicalMatrix,
    path_positions_angstrom: ArrayView2<'_, Real>,
    sigma2: Real,
) -> Result<(), DebyeError> {
    let path_atoms = rotated_spring_path_atoms(
        matrix.atom_positions_angstrom.view(),
        path_positions_angstrom,
    )?;
    if path_atoms.len() < 2 {
        return Ok(());
    }
    if sigma2 > state.max_sigma2 {
        state.max_sigma2 = sigma2;
    }
    let first = matrix.potential_indices[path_atoms[0]];
    let second = matrix.potential_indices[path_atoms[1]];
    if first >= state.pair_sigma2.nrows() || second >= state.pair_sigma2.ncols() {
        return Err(DebyeError::InvalidSpringInput {
            reason: "spring recursion state has too few potentials",
        });
    }
    if sigma2 > state.pair_sigma2[(first, second)] {
        state.pair_sigma2[(first, second)] = sigma2;
        state.pair_sigma2[(second, first)] = sigma2;
    }
    Ok(())
}

struct SpringPathSetup {
    nconv: Vec<usize>,
    path_unique: Vec<usize>,
    neighborhood: Vec<usize>,
    reduced_mass: Real,
    initial_vector: Array2<Real>,
    moment_frequency: Real,
}

fn spring_path_setup(
    matrix: &SpringDynamicalMatrix,
    path_positions_angstrom: ArrayView2<'_, Real>,
) -> Result<SpringPathSetup, DebyeError> {
    validate_spring_path(path_positions_angstrom)?;
    let atom_count = matrix.atom_positions_angstrom.nrows();
    let nleg = path_positions_angstrom.nrows() - 1;
    let path_atoms = rotated_spring_path_atoms(
        matrix.atom_positions_angstrom.view(),
        path_positions_angstrom,
    )?;
    let mut nconv = vec![0_usize; nleg + 1];
    nconv[1..=nleg].copy_from_slice(&path_atoms);
    nconv[0] = nconv[nleg];

    let mut path_unique = Vec::new();
    let mut neighborhood = Vec::new();
    if atom_count > 0 {
        neighborhood.push(0);
    }
    let mut inverse_mu = 0.0;
    for il in 1..=nleg {
        let atom = nconv[il];
        if !path_unique.contains(&atom) {
            path_unique.push(atom);
        }
        for candidate in 0..atom_count {
            if distance_rows(matrix.atom_positions_angstrom.view(), candidate, atom)
                <= matrix.interaction_radius_angstrom
                && !neighborhood.contains(&candidate)
            {
                neighborhood.push(candidate);
            }
        }

        let previous = nconv[il - 1];
        let next = if il == nleg { nconv[1] } else { nconv[il + 1] };
        let mass = atomic_weight(matrix.atomic_numbers[atom])?;
        for axis in 0..3 {
            let director = matrix.pair_directions[(axis, atom, previous)]
                + matrix.pair_directions[(axis, atom, next)];
            inverse_mu += 0.25 * director.powi(2) / mass;
        }
    }
    if inverse_mu == 0.0 {
        return Err(DebyeError::ZeroSpringReducedMassDenominator);
    }

    let mut reduced_mass = 1.0 / inverse_mu;
    let mut initial_vector = Array2::<Real>::zeros((3, atom_count));
    for _ in 0..10 {
        initial_vector.fill(0.0);
        for il in 1..=nleg {
            let atom = nconv[il];
            let previous = if il == 1 { nconv[nleg] } else { nconv[il - 1] };
            let next = if il == nleg { nconv[1] } else { nconv[il + 1] };
            let mass = atomic_weight(matrix.atomic_numbers[atom])?;
            let scale = (reduced_mass / mass).sqrt() / 2.0;
            for axis in 0..3 {
                initial_vector[(axis, atom)] += scale
                    * (matrix.pair_directions[(axis, previous, atom)]
                        - matrix.pair_directions[(axis, atom, next)]);
            }
        }
        let mut q0q0 = 0.0;
        for &atom in &path_unique {
            for axis in 0..3 {
                q0q0 += initial_vector[(axis, atom)].powi(2);
            }
        }
        let rounded = (q0q0 * 1000.0).round() / 1000.0;
        if (rounded - 1.0).abs() <= 5.0e-4 {
            break;
        }
        if q0q0 == 0.0 {
            return Err(DebyeError::ZeroDmdwSeedNorm);
        }
        reduced_mass /= q0q0;
    }

    let w0 = matrix.characteristic_frequency;
    if w0 <= 0.0 || !w0.is_finite() {
        return Err(DebyeError::NonPositiveSpringFrequency { value: w0 });
    }
    let mut moment0 = 0.0;
    for &left in &path_unique {
        for &right in &path_unique {
            for left_axis in 0..3 {
                for right_axis in 0..3 {
                    moment0 += initial_vector[(left_axis, left)]
                        * matrix.matrix[(left_axis, right_axis, left, right)]
                        * initial_vector[(right_axis, right)]
                        / w0
                        / w0;
                }
            }
        }
    }
    let wnorm = 100.0 * w0 / (SPRING_AMU0 * 10.0).sqrt();
    let moment_frequency = wnorm * moment0.sqrt();

    Ok(SpringPathSetup {
        nconv,
        path_unique,
        neighborhood,
        reduced_mass,
        initial_vector,
        moment_frequency,
    })
}

fn spring_nonzero_upper_pairs(matrix: &SpringDynamicalMatrix) -> Vec<(usize, usize)> {
    let atom_count = matrix.atom_positions_angstrom.nrows();
    let mut pairs = Vec::new();
    for left in 0..atom_count {
        for right in left..atom_count {
            let mut sum = 0.0;
            for axis1 in 0..3 {
                for axis2 in 0..3 {
                    sum += matrix.matrix[(axis1, axis2, left, right)].abs();
                }
            }
            if sum != 0.0 {
                pairs.push((left, right));
            }
        }
    }
    pairs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpringParseMode {
    Cards,
    Stretches,
    Angles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpringKeyword {
    Stretches,
    Angles,
    Vdos,
    Prdos,
    End,
}

fn spring_data_line(line: &str) -> &str {
    let line = line.split_once('*').map_or(line, |(prefix, _)| prefix);
    let line = line.split_once('#').map_or(line, |(prefix, _)| prefix);
    let line = line.split_once('!').map_or(line, |(prefix, _)| prefix);
    line.trim()
}

fn spring_keyword(word: &str) -> Option<SpringKeyword> {
    match word {
        "STRE" | "STRETCH" | "STRETCHES" => Some(SpringKeyword::Stretches),
        "ANGL" | "ANGLE" | "ANGLES" => Some(SpringKeyword::Angles),
        "VDOS" => Some(SpringKeyword::Vdos),
        "PRDOS" | "PRINT" => Some(SpringKeyword::Prdos),
        "END" => Some(SpringKeyword::End),
        _ => None,
    }
}

fn parse_spring_vdos(input: &mut SpringInput, fields: &[&str]) -> Result<(), DebyeError> {
    if fields.len() < 4 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "VDOS requires resolution, wmax, and dosfit",
        });
    }
    input.resolution = parse_spring_real(fields[1])?;
    input.max_frequency = parse_spring_real(fields[2])?;
    input.dos_fit = parse_spring_real(fields[3])?;
    if let Some(cutoff) = fields.get(4) {
        input.cutoff = parse_spring_real(cutoff)?;
    }
    Ok(())
}

fn parse_spring_stretch(fields: &[&str]) -> Result<SpringStretch, DebyeError> {
    if fields.len() < 4 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "STRETCH row requires four fields",
        });
    }
    Ok(SpringStretch {
        first_atom: parse_spring_usize(fields[0])?,
        second_atom: parse_spring_usize(fields[1])?,
        force_constant: parse_spring_real(fields[2])?,
        distance_tolerance_percent: parse_spring_real(fields[3])?,
    })
}

fn parse_spring_angle(fields: &[&str]) -> Result<SpringAngle, DebyeError> {
    if fields.len() < 5 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "ANGLE row requires five fields",
        });
    }
    Ok(SpringAngle {
        first_atom: parse_spring_usize(fields[0])?,
        center_atom: parse_spring_usize(fields[1])?,
        third_atom: parse_spring_usize(fields[2])?,
        force_constant: parse_spring_real(fields[3])?,
        angle_tolerance_percent: parse_spring_real(fields[4])?,
    })
}

fn parse_spring_i32(text: &str) -> Result<i32, DebyeError> {
    text.parse::<i32>()
        .map_err(|_| DebyeError::InvalidSpringInput {
            reason: "failed to parse integer",
        })
}

fn parse_spring_usize(text: &str) -> Result<usize, DebyeError> {
    text.parse::<usize>()
        .map_err(|_| DebyeError::InvalidSpringInput {
            reason: "failed to parse atom index",
        })
}

fn parse_spring_real(text: &str) -> Result<Real, DebyeError> {
    text.parse::<Real>()
        .map_err(|_| DebyeError::InvalidSpringInput {
            reason: "failed to parse real",
        })
}

fn validate_spring_input(input: &SpringInput) -> Result<(), DebyeError> {
    ensure_positive("spring resolution", input.resolution)?;
    ensure_nonnegative("spring cutoff", input.cutoff)?;
    ensure_positive("spring max frequency", input.max_frequency)?;
    ensure_nonnegative("spring dosfit", input.dos_fit)?;
    for stretch in &input.stretches {
        ensure_nonnegative("spring stretch force constant", stretch.force_constant)?;
        ensure_finite(
            "spring stretch distance tolerance",
            stretch.distance_tolerance_percent,
        )?;
    }
    for angle in &input.angles {
        ensure_nonnegative("spring angle force constant", angle.force_constant)?;
        ensure_finite("spring angle tolerance", angle.angle_tolerance_percent)?;
    }
    if input.stretches.is_empty() {
        return Err(DebyeError::InvalidSpringInput {
            reason: "at least one stretch is required",
        });
    }
    Ok(())
}

fn validate_spring_matrix_input(input: SpringDynamicalMatrixInput<'_>) -> Result<(), DebyeError> {
    let (rows, columns) = input.atom_positions_angstrom.dim();
    if rows == 0 || columns != 3 {
        return Err(DebyeError::InvalidDmdwAtomShape { rows, columns });
    }
    if input.atomic_numbers.len() != rows {
        return Err(DebyeError::InvalidDmdwMassCount {
            positions: rows,
            masses: input.atomic_numbers.len(),
        });
    }
    if input.potential_indices.len() != rows {
        return Err(DebyeError::InvalidSpringInput {
            reason: "potential indices must align with atoms",
        });
    }
    validate_spring_atom_index(input.absorber_index, rows)?;
    for value in input.atom_positions_angstrom.iter().copied() {
        ensure_finite("spring atom position", value)?;
    }
    for &atomic_number in input.atomic_numbers {
        atomic_weight(atomic_number)?;
    }
    for stretch in &input.spring.stretches {
        validate_spring_atom_index(stretch.first_atom, rows)?;
        validate_spring_atom_index(stretch.second_atom, rows)?;
    }
    for angle in &input.spring.angles {
        validate_spring_atom_index(angle.first_atom, rows)?;
        validate_spring_atom_index(angle.center_atom, rows)?;
        validate_spring_atom_index(angle.third_atom, rows)?;
    }
    Ok(())
}

fn validate_spring_atom_index(index: usize, atom_count: usize) -> Result<(), DebyeError> {
    if index < atom_count {
        Ok(())
    } else {
        Err(DebyeError::InvalidSpringAtomIndex { index, atom_count })
    }
}

fn expand_stretches(
    input: SpringDynamicalMatrixInput<'_>,
    stretch_constants: &mut Array2<Real>,
    distance_tolerances: &mut Array2<Real>,
) -> Result<(), DebyeError> {
    let atom_count = input.atom_positions_angstrom.nrows();
    for stretch in &input.spring.stretches {
        let first = stretch.first_atom;
        let second = stretch.second_atom;
        let reference = distance_rows(input.atom_positions_angstrom, first, second);
        let tolerance = distance_tolerances[(first, second)];
        let first_z = input.atomic_numbers[first];
        let second_z = input.atomic_numbers[second];
        for left in 0..atom_count.saturating_sub(1) {
            for right in (left + 1)..atom_count {
                let candidate = distance_rows(input.atom_positions_angstrom, left, right);
                if candidate == 0.0 {
                    continue;
                }
                let relative = (reference / candidate - 1.0).abs();
                if relative > tolerance {
                    continue;
                }
                let left_z = input.atomic_numbers[left];
                let right_z = input.atomic_numbers[right];
                if !((first_z == left_z && second_z == right_z)
                    || (first_z == right_z && second_z == left_z))
                {
                    continue;
                }
                stretch_constants[(left, right)] = stretch.force_constant;
                stretch_constants[(right, left)] = stretch.force_constant;
            }
        }
        stretch_constants[(second, first)] = stretch.force_constant;
    }
    Ok(())
}

fn expand_angles(
    input: SpringDynamicalMatrixInput<'_>,
    distance_tolerances: &Array2<Real>,
) -> Result<Vec<SpringAngle>, DebyeError> {
    let atom_count = input.atom_positions_angstrom.nrows();
    let mut angles = input.spring.angles.clone();
    let originals = angles.clone();
    for angle in originals {
        let first = angle.first_atom;
        let center = angle.center_atom;
        let third = angle.third_atom;
        let first_center = distance_rows(input.atom_positions_angstrom, first, center);
        let third_center = distance_rows(input.atom_positions_angstrom, third, center);
        let reference_cos = feff_cosine(input.atom_positions_angstrom, first, center, third)?;
        let reference_angle = reference_cos.acos();
        if reference_angle == 0.0 {
            continue;
        }
        let first_z = input.atomic_numbers[first];
        let center_z = input.atomic_numbers[center];
        let third_z = input.atomic_numbers[third];
        let distance_tol_first = distance_tolerances[(first, center)];
        let distance_tol_third = distance_tolerances[(third, center)];
        let angle_tolerance = angle.angle_tolerance_percent.abs() / 100.0;
        for left in 0..atom_count {
            for middle in 0..atom_count {
                if left == middle {
                    continue;
                }
                let left_middle = distance_rows(input.atom_positions_angstrom, left, middle);
                for right in (left + 1)..atom_count {
                    if right == middle {
                        continue;
                    }
                    let right_middle = distance_rows(input.atom_positions_angstrom, right, middle);
                    let direct = ((left_middle / first_center - 1.0).abs() <= distance_tol_first)
                        && ((right_middle / third_center - 1.0).abs() <= distance_tol_third);
                    let reversed = ((right_middle / first_center - 1.0).abs()
                        <= distance_tol_first)
                        && ((left_middle / third_center - 1.0).abs() <= distance_tol_third);
                    if !direct && !reversed {
                        continue;
                    }
                    let left_z = input.atomic_numbers[left];
                    let middle_z = input.atomic_numbers[middle];
                    let right_z = input.atomic_numbers[right];
                    if !((left_z == first_z && middle_z == center_z && right_z == third_z)
                        || (right_z == first_z && middle_z == center_z && left_z == third_z))
                    {
                        continue;
                    }
                    let candidate_angle =
                        feff_cosine(input.atom_positions_angstrom, left, middle, right)?.acos();
                    let relative_angle = (candidate_angle / reference_angle - 1.0).abs();
                    if relative_angle >= angle_tolerance {
                        continue;
                    }
                    if angles.iter().any(|existing| {
                        (existing.first_atom == left
                            && existing.center_atom == middle
                            && existing.third_atom == right)
                            || (existing.first_atom == right
                                && existing.center_atom == middle
                                && existing.third_atom == left)
                    }) {
                        continue;
                    }
                    angles.push(SpringAngle {
                        first_atom: left,
                        center_atom: middle,
                        third_atom: right,
                        force_constant: angle.force_constant,
                        angle_tolerance_percent: angle.angle_tolerance_percent,
                    });
                }
            }
        }
    }
    Ok(angles)
}

fn spring_interaction_radius(
    positions: ArrayView2<'_, Real>,
    stretch_constants: &Array2<Real>,
    distance_tolerances: &Array2<Real>,
) -> Result<Real, DebyeError> {
    let atom_count = positions.nrows();
    let mut shell_radii = vec![Vec::<Real>::new(); atom_count];
    let mut radius = 0.0;
    for atom in 0..atom_count {
        for other in 0..atom_count {
            if atom == other {
                continue;
            }
            let distance = distance_rows(positions, atom, other);
            let tolerance = distance_tolerances[(atom, other)];
            let exists = shell_radii[atom]
                .iter()
                .any(|&shell| shell != 0.0 && ((distance - shell) / shell).abs() <= tolerance);
            if !exists {
                if stretch_constants[(atom, other)] != 0.0 && distance > radius {
                    radius = distance;
                }
                shell_radii[atom].push(distance);
            }
        }
        shell_radii[atom].sort_by(|left, right| left.total_cmp(right));
    }
    Ok(radius)
}

fn fill_pair_directions(
    positions: ArrayView2<'_, Real>,
    pair_directions: &mut Array3<Real>,
) -> Result<(), DebyeError> {
    let atom_count = positions.nrows();
    for left in 0..atom_count.saturating_sub(1) {
        for right in (left + 1)..atom_count {
            let dx = positions[(right, 0)] - positions[(left, 0)];
            let dy = positions[(right, 1)] - positions[(left, 1)];
            let dz = positions[(right, 2)] - positions[(left, 2)];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance == 0.0 {
                return Err(DebyeError::ZeroLengthDmdwAtomPair {
                    first: left,
                    second: right,
                });
            }
            for (axis, value) in [dx, dy, dz].into_iter().enumerate() {
                pair_directions[(axis, left, right)] = value / distance;
                pair_directions[(axis, right, left)] = -value / distance;
            }
        }
    }
    Ok(())
}

fn build_angle_matrix(
    positions: ArrayView2<'_, Real>,
    angles: &[SpringAngle],
    interaction_radius: Real,
    angle_matrix: &mut Array4<Real>,
) -> Result<(), DebyeError> {
    for angle in angles {
        let first = angle.first_atom;
        let center = angle.center_atom;
        let third = angle.third_atom;
        if first == center || center == third || angle.force_constant == 0.0 {
            continue;
        }
        if distance_rows(positions, first, center) > interaction_radius
            || distance_rows(positions, third, center) > interaction_radius
        {
            continue;
        }
        let (si, sj, sk) = spring_angle_coefficients(positions, first, center, third)?;
        for axis1 in 0..3 {
            for axis2 in 0..3 {
                let force = angle.force_constant;
                angle_matrix[(axis1, axis2, first, center)] += force * si[axis1] * sj[axis2];
                angle_matrix[(axis1, axis2, center, third)] += force * sj[axis1] * sk[axis2];
                angle_matrix[(axis1, axis2, first, third)] += force * si[axis1] * sk[axis2];
                angle_matrix[(axis1, axis2, first, first)] += force * si[axis1] * si[axis2];
                angle_matrix[(axis1, axis2, center, center)] += force * sj[axis1] * sj[axis2];
                angle_matrix[(axis1, axis2, third, third)] += force * sk[axis1] * sk[axis2];
                angle_matrix[(axis2, axis1, center, first)] =
                    angle_matrix[(axis1, axis2, first, center)];
                angle_matrix[(axis2, axis1, third, center)] =
                    angle_matrix[(axis1, axis2, center, third)];
                angle_matrix[(axis2, axis1, third, first)] =
                    angle_matrix[(axis1, axis2, first, third)];
            }
        }
    }
    Ok(())
}

fn build_stretch_matrix(
    stretch_constants: &Array2<Real>,
    pair_directions: &Array3<Real>,
    stretch_matrix: &mut Array4<Real>,
) {
    let atom_count = stretch_constants.nrows();
    for left in 0..atom_count {
        for right in left..atom_count {
            for axis1 in 0..3 {
                let directional =
                    stretch_constants[(left, right)] * pair_directions[(axis1, left, right)];
                for axis2 in 0..3 {
                    let mut diagonal = 0.0;
                    if left == right {
                        for atom in 0..atom_count {
                            if stretch_constants[(atom, right)] != 0.0 {
                                diagonal += stretch_constants[(atom, right)]
                                    * pair_directions[(axis1, atom, right)]
                                    * pair_directions[(axis2, atom, right)];
                            }
                        }
                    }
                    stretch_matrix[(axis1, axis2, left, right)] =
                        diagonal - directional * pair_directions[(axis2, left, right)];
                    stretch_matrix[(axis2, axis1, right, left)] =
                        stretch_matrix[(axis1, axis2, left, right)];
                }
            }
        }
    }
}

fn build_mass_weighted_matrix(
    stretch_matrix: &Array4<Real>,
    angle_matrix: &Array4<Real>,
    atomic_numbers: &[usize],
    matrix: &mut Array4<Real>,
) -> Result<(), DebyeError> {
    let atom_count = atomic_numbers.len();
    let masses = atomic_numbers
        .iter()
        .map(|&z| atomic_weight(z))
        .collect::<Result<Vec<_>, _>>()?;
    for left in 0..atom_count {
        let left_mass = masses[left].sqrt();
        for right in 0..atom_count {
            let right_mass = masses[right].sqrt();
            for axis1 in 0..3 {
                for axis2 in 0..3 {
                    matrix[(axis1, axis2, left, right)] = (angle_matrix
                        [(axis1, axis2, left, right)]
                        + stretch_matrix[(axis1, axis2, left, right)])
                        / left_mass
                        / right_mass;
                }
            }
        }
    }
    Ok(())
}

fn spring_characteristic_frequency(
    input: SpringDynamicalMatrixInput<'_>,
    stretch_constants: &Array2<Real>,
    pair_directions: &Array3<Real>,
    matrix: &Array4<Real>,
    distance_tolerances: &Array2<Real>,
    min_stretch: Real,
    min_stretch_pair: Option<(usize, usize)>,
) -> Result<Real, DebyeError> {
    let absorber = input.absorber_index;
    let first_shell = first_shell_radius(input.atom_positions_angstrom, absorber)?;
    let mut first_neighbor = None;
    for atom in 0..input.atom_positions_angstrom.nrows() {
        if atom == absorber {
            continue;
        }
        let distance = distance_rows(input.atom_positions_angstrom, absorber, atom);
        let distance_tolerance = distance_tolerances[(absorber, atom)];
        if first_shell != 0.0
            && (distance / first_shell - 1.0).abs() <= distance_tolerance
            && stretch_constants[(absorber, atom)] != 0.0
        {
            first_neighbor = Some(atom);
            break;
        }
    }
    let Some(first_neighbor) = first_neighbor else {
        return spring_characteristic_frequency_from_min_stretch(
            input,
            min_stretch,
            min_stretch_pair,
        );
    };

    let absorber_mass = atomic_weight(input.atomic_numbers[absorber])?;
    let neighbor_mass = atomic_weight(input.atomic_numbers[first_neighbor])?;
    let reduced_mass = 1.0 / (1.0 / absorber_mass + 1.0 / neighbor_mass);
    let mut moment = 0.0;
    for ii in 0..2 {
        for jj in 0..2 {
            let left = if ii == 0 { absorber } else { first_neighbor };
            let right = if jj == 0 { absorber } else { first_neighbor };
            let sign = if (ii + jj) % 2 == 0 { 1.0 } else { -1.0 };
            let factor = sign * reduced_mass;
            let mass_factor = (1.0
                / atomic_weight(input.atomic_numbers[left])?
                / atomic_weight(input.atomic_numbers[right])?)
            .sqrt();
            for axis1 in 0..3 {
                for axis2 in 0..3 {
                    moment += factor
                        * mass_factor
                        * pair_directions[(axis1, absorber, first_neighbor)]
                        * matrix[(axis1, axis2, left, right)]
                        * pair_directions[(axis2, absorber, first_neighbor)];
                }
            }
        }
    }
    if moment > 0.0 {
        return Ok(moment.sqrt());
    }
    spring_characteristic_frequency_from_min_stretch(input, min_stretch, min_stretch_pair)
}

fn spring_characteristic_frequency_from_min_stretch(
    input: SpringDynamicalMatrixInput<'_>,
    min_stretch: Real,
    min_stretch_pair: Option<(usize, usize)>,
) -> Result<Real, DebyeError> {
    let Some((first, second)) = min_stretch_pair else {
        return Err(DebyeError::InvalidSpringInput {
            reason: "no spring stretch pair",
        });
    };
    let first_mass = atomic_weight(input.atomic_numbers[first])?;
    let second_mass = atomic_weight(input.atomic_numbers[second])?;
    let reduced_mass = 1.0 / (1.0 / first_mass + 1.0 / second_mass);
    if min_stretch <= 0.0 {
        return Err(DebyeError::NonPositiveSpringFrequency { value: min_stretch });
    }
    Ok((min_stretch / reduced_mass).sqrt())
}

fn first_shell_radius(
    positions: ArrayView2<'_, Real>,
    absorber: usize,
) -> Result<Real, DebyeError> {
    let mut first_shell = Real::INFINITY;
    for atom in 0..positions.nrows() {
        if atom == absorber {
            continue;
        }
        let distance = distance_rows(positions, absorber, atom);
        if distance < first_shell {
            first_shell = distance;
        }
    }
    if first_shell.is_finite() {
        Ok(first_shell)
    } else {
        Err(DebyeError::InvalidSpringInput {
            reason: "absorber has no neighbor",
        })
    }
}

fn spring_first_shell_coordination(
    positions: ArrayView2<'_, Real>,
    absorber: usize,
    distance_tolerances: &Array2<Real>,
) -> Result<Real, DebyeError> {
    let first_shell = first_shell_radius(positions, absorber)?;
    let mut count = 0.0;
    for atom in 0..positions.nrows() {
        if atom == absorber {
            continue;
        }
        let distance = distance_rows(positions, absorber, atom);
        if first_shell != 0.0
            && (distance / first_shell - 1.0).abs() <= distance_tolerances[(absorber, atom)]
        {
            count += 1.0;
        }
    }
    ensure_positive("spring first-shell coordination", count)?;
    Ok(count)
}

fn validate_spring_path(path: ArrayView2<'_, Real>) -> Result<(), DebyeError> {
    let (rows, columns) = path.dim();
    if rows < 2 || columns != 3 {
        return Err(DebyeError::InvalidPathShape { rows, columns });
    }
    for value in path.iter().copied() {
        ensure_finite("spring path position", value)?;
    }
    Ok(())
}

fn rotated_spring_path_atoms(
    atoms: ArrayView2<'_, Real>,
    path: ArrayView2<'_, Real>,
) -> Result<Vec<usize>, DebyeError> {
    validate_spring_path(path)?;
    let nleg = path.nrows() - 1;
    let mut rotated = Array2::<Real>::zeros((nleg, 3));
    for axis in 0..3 {
        rotated[(0, axis)] = path[(nleg, axis)];
    }
    for leg in 1..nleg {
        for axis in 0..3 {
            rotated[(leg, axis)] = path[(leg, axis)];
        }
    }

    let mut matched = Vec::with_capacity(nleg);
    for leg in 0..nleg {
        let atom = match_spring_atom(atoms, rotated.row(leg), leg + 1)?;
        matched.push(atom);
    }
    Ok(matched)
}

fn match_spring_atom(
    atoms: ArrayView2<'_, Real>,
    position: ndarray::ArrayView1<'_, Real>,
    leg: usize,
) -> Result<usize, DebyeError> {
    for atom in 0..atoms.nrows() {
        let matches = (0..3).all(|axis| {
            let path_value = (SPRING_PATH_MATCH_SCALE * position[axis]).round();
            let atom_value = (SPRING_PATH_MATCH_SCALE * atoms[(atom, axis)]).round();
            (path_value - atom_value).abs() <= SPRING_PATH_MATCH_TOLERANCE
        });
        if matches {
            return Ok(atom);
        }
    }
    Err(DebyeError::UnmatchedSpringPathAtom { leg })
}

fn spring_recursion_fallback_sigma2(
    state: Option<&SpringRecursionState>,
    matrix: &SpringDynamicalMatrix,
    atom: usize,
) -> Result<Real, DebyeError> {
    let Some(state) = state else {
        return Ok(0.0);
    };
    let potential = matrix.potential_indices[atom];
    if potential >= state.pair_sigma2.nrows() {
        return Err(DebyeError::InvalidSpringInput {
            reason: "spring recursion state has too few potentials",
        });
    }
    let sigma = state.pair_sigma2[(potential, potential)];
    if sigma < 1.0e-6 {
        Ok(state.max_sigma2)
    } else {
        Ok(sigma)
    }
}

fn spring_recursion_sigma_component(
    reduced_mass: Real,
    frequency: Real,
    temperature: Real,
) -> Real {
    let thermal_argument = if temperature == 0.0 {
        Real::INFINITY
    } else {
        frequency * SPRING_RM_TEMPERATURE_FACTOR / 2.0 / temperature
    };
    SPRING_RM_SIGMA_FACTOR / (reduced_mass * frequency * thermal_argument.tanh())
}

type SpringAngleCoefficients = ([Real; 3], [Real; 3], [Real; 3]);

fn spring_angle_coefficients(
    positions: ArrayView2<'_, Real>,
    first: usize,
    center: usize,
    third: usize,
) -> Result<SpringAngleCoefficients, DebyeError> {
    let mut rji = [0.0; 3];
    let mut rjk = [0.0; 3];
    let mut dji = 0.0;
    let mut djk = 0.0;
    for axis in 0..3 {
        rji[axis] = positions[(first, axis)] - positions[(center, axis)];
        rjk[axis] = positions[(third, axis)] - positions[(center, axis)];
        dji += rji[axis].powi(2);
        djk += rjk[axis].powi(2);
    }
    dji = dji.sqrt();
    djk = djk.sqrt();
    if dji == 0.0 || djk == 0.0 {
        return Err(DebyeError::ZeroLengthDmdwAtomPair {
            first,
            second: center,
        });
    }
    let mut eji = [0.0; 3];
    let mut ejk = [0.0; 3];
    let mut dotj = 0.0;
    for axis in 0..3 {
        eji[axis] = rji[axis] / dji;
        ejk[axis] = rjk[axis] / djk;
        dotj += eji[axis] * ejk[axis];
    }
    let cross = cross(eji, ejk);
    let sinj = vector_norm(cross);
    if sinj == 0.0 {
        return Err(DebyeError::InvalidSpringInput {
            reason: "linear spring angle",
        });
    }
    let mut si = [0.0; 3];
    let mut sj = [0.0; 3];
    let mut sk = [0.0; 3];
    for axis in 0..3 {
        si[axis] = (dotj * eji[axis] - ejk[axis]) / dji / sinj;
        sk[axis] = (dotj * ejk[axis] - eji[axis]) / djk / sinj;
        sj[axis] =
            ((dji - djk * dotj) * eji[axis] + (djk - dji * dotj) * ejk[axis]) / dji / djk / sinj;
    }
    Ok((si, sj, sk))
}

fn feff_cosine(
    positions: ArrayView2<'_, Real>,
    first: usize,
    center: usize,
    third: usize,
) -> Result<Real, DebyeError> {
    let mut vv1 = 0.0;
    let mut vv2 = 0.0;
    let mut scalar = 0.0;
    for axis in 0..3 {
        vv1 += (positions[(first, axis)] - positions[(center, axis)]).powi(2);
        vv2 += (positions[(third, axis)] - positions[(center, axis)]).powi(2);
        scalar += (positions[(first, axis)] - positions[(center, axis)])
            * (positions[(third, axis)] - positions[(center, axis)]);
    }
    if vv1 == 0.0 || vv2 == 0.0 {
        return Err(DebyeError::ZeroLengthDmdwAtomPair {
            first,
            second: center,
        });
    }
    Ok(scalar / vv1.sqrt() / vv2.sqrt())
}

fn distance_rows(positions: ArrayView2<'_, Real>, first: usize, second: usize) -> Real {
    (0..3)
        .map(|axis| (positions[(first, axis)] - positions[(second, axis)]).powi(2))
        .sum::<Real>()
        .sqrt()
}
