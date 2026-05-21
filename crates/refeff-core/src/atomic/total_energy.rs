use super::*;

/// Port of FEFF `ATOM/etotal.f90`, excluding the radial integral solver.
///
/// The supplied callback receives FEFF-style `fdrirk(i,j,l,m,k)` requests and
/// must return the corresponding radial integral. This function performs the
/// FEFF accumulation of direct Coulomb, exchange Coulomb, magnetic Breit,
/// retarded Breit, and one-electron energy terms.
pub fn atomic_total_energy<F>(
    input: AtomicTotalEnergyInput<'_>,
    radial_integral: F,
) -> Result<AtomicTotalEnergy, AtomMathError>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_total_energy_input(&input)?;
    AtomicTotalEnergyContext {
        input,
        radial_integral,
    }
    .calculate()
}

pub(super) struct AtomicTotalEnergyContext<'a, F> {
    input: AtomicTotalEnergyInput<'a>,
    radial_integral: F,
}

impl<F> AtomicTotalEnergyContext<'_, F>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicTotalEnergy, AtomMathError> {
        let direct_coulomb = self.direct_coulomb_energy()?;
        let exchange_coulomb = self.exchange_coulomb_energy()?;
        let (magnetic_breit, retarded_breit) = self.breit_energies()?;
        let orbital_energy = self
            .input
            .orbital_energies
            .iter()
            .zip(self.input.occupations)
            .map(|(&energy, &occupation)| energy * occupation)
            .sum::<Real>();
        let total =
            -(direct_coulomb + exchange_coulomb) + magnetic_breit + retarded_breit + orbital_energy;

        Ok(AtomicTotalEnergy {
            total,
            direct_coulomb,
            exchange_coulomb,
            magnetic_breit,
            retarded_breit,
        })
    }

    fn direct_coulomb_energy(&mut self) -> Result<Real, AtomMathError> {
        let mut energy = 0.0;
        for left in 0..self.orbital_count() {
            let left_l = self.abs_kappa(left)? - 1;
            for right in 0..=left {
                let symmetry_weight = if right == left { 2.0 } else { 1.0 };
                let right_l = self.abs_kappa(right)? - 1;
                let max_rank = 2 * left_l.min(right_l);
                let mut rank = 0;
                while rank <= max_rank {
                    let radial = self.radial(left + 1, left + 1, right + 1, right + 1, rank)?;
                    energy +=
                        radial * self.direct_coefficient(left, right, rank)? / symmetry_weight;
                    rank += 2;
                }
            }
        }
        Ok(energy)
    }

    fn exchange_coulomb_energy(&mut self) -> Result<Real, AtomMathError> {
        let mut energy = 0.0;
        for left in 1..self.orbital_count() {
            let valence_weight = if self.input.valence_occupations[left] > 0.0 {
                0.5
            } else {
                1.0
            };
            for right in 0..left {
                if self.input.valence_occupations[right] > 0.0 {
                    continue;
                }
                let left_abs = self.abs_kappa(left)?;
                let right_abs = self.abs_kappa(right)?;
                let mut rank = left_abs.abs_diff(right_abs);
                if self.kappa(left).signum() != self.kappa(right).signum() {
                    rank += 1;
                }
                let max_rank = left_abs + right_abs - 1;
                while rank <= max_rank {
                    let radial = self.radial(left + 1, right + 1, left + 1, right + 1, rank)?;
                    energy -=
                        radial * self.exchange_coefficient(left, right, rank)? * valence_weight;
                    rank += 2;
                }
            }
        }
        Ok(energy)
    }

    fn breit_energies(&mut self) -> Result<(Real, Real), AtomMathError> {
        let mut magnetic = 0.0;
        let mut retarded = 0.0;
        for right in 0..self.orbital_count() {
            let right_j2 = self.j2(right)?;
            for left in 0..=right {
                let left_j2 = self.j2(left)?;
                let max_rank = left_j2.min(right_j2);
                let mut rank = 1;
                while rank <= max_rank {
                    let radial = self.radial(right + 1, right + 1, left + 1, left + 1, rank)?;
                    if left == right {
                        let coefficients = atomic_breit_angular_coefficients(
                            self.kappa(right),
                            self.kappa(right),
                            rank,
                        )?;
                        let occupation = atomic_occupation_product(
                            self.input.occupations,
                            self.input.kappas,
                            right,
                            right,
                        )?;
                        magnetic +=
                            coefficients.magnetic.iter().sum::<Real>() * radial * occupation / 2.0;
                    }
                    rank += 2;
                }
            }
        }

        for right in 1..self.orbital_count() {
            let right_branch = self.exchange_breit_branch(right)?;
            for left in 0..right {
                let left_branch = self.exchange_breit_branch(left)?;
                let occupation = atomic_occupation_product(
                    self.input.occupations,
                    self.input.kappas,
                    right,
                    left,
                )?;
                let mut rank = left_branch.minimum_rank(right_branch)?;
                let max_rank = left_branch.maximum_rank(right_branch)?;
                let parity_sum = i32::try_from(rank)
                    .map_err(|_| AtomMathError::BreitRankOutOfRange { rank })?
                    .checked_add(left_branch.angular_l)
                    .and_then(|value| value.checked_add(right_branch.angular_l))
                    .ok_or(AtomMathError::BreitBranchOutOfRange)?;
                if parity_sum % 2 == 0 {
                    rank += 1;
                }
                let kappa_sum = self.abs_kappa(right)? + self.abs_kappa(left)?;
                while rank <= max_rank {
                    let coefficients = atomic_breit_angular_coefficients(
                        self.kappa(right),
                        self.kappa(left),
                        rank,
                    )?;
                    let radials = self.exchange_breit_radials(left, right, rank, kappa_sum)?;
                    magnetic += coefficients
                        .magnetic
                        .iter()
                        .zip(radials)
                        .map(|(&coefficient, radial)| coefficient * radial * occupation)
                        .sum::<Real>();
                    retarded += coefficients
                        .retarded
                        .iter()
                        .zip(radials)
                        .map(|(&coefficient, radial)| coefficient * radial * occupation)
                        .sum::<Real>();
                    rank += 2;
                }
            }
        }

        Ok((magnetic, retarded))
    }

    fn exchange_breit_radials(
        &mut self,
        left: usize,
        right: usize,
        rank: usize,
        kappa_sum: usize,
    ) -> Result<[Real; 3], AtomMathError> {
        let mut radials = [0.0; 3];
        if !(kappa_sum <= rank && self.kappa(left) < 0 && self.kappa(right) > 0) {
            radials[0] = self.radial(left + 1, right + 1, left + 1, right + 1, rank)?;
            radials[1] = self.radial(0, 0, right + 1, left + 1, rank)?;
        }
        if !(kappa_sum <= rank && self.kappa(left) > 0 && self.kappa(right) < 0) {
            radials[2] = self.radial(right + 1, left + 1, right + 1, left + 1, rank)?;
            if radials[1] == 0.0 {
                radials[1] = self.radial(0, 0, left + 1, right + 1, rank)?;
            }
        }
        Ok(radials)
    }

    fn radial(
        &mut self,
        first_left: usize,
        first_right: usize,
        second_left: usize,
        second_right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let request = AtomicRadialIntegralRequest {
            first_left,
            first_right,
            second_left,
            second_right,
            rank,
        };
        let value = (self.radial_integral)(request)?;
        validate_finite_scalar("radial_integral", value)?;
        Ok(value)
    }

    fn direct_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let channel = self.coefficient_channel(rank)?;
        if left <= right {
            Ok(self.input.coulomb_coefficients[(left, right, channel)])
        } else {
            Ok(self.input.coulomb_coefficients[(right, left, channel)])
        }
    }

    fn exchange_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let channel = self.coefficient_channel(rank)?;
        if left < right {
            Ok(self.input.coulomb_coefficients[(right, left, channel)])
        } else if left > right {
            Ok(self.input.coulomb_coefficients[(left, right, channel)])
        } else {
            Ok(0.0)
        }
    }

    fn coefficient_channel(&self, rank: usize) -> Result<usize, AtomMathError> {
        let channel = rank / 2;
        let channels = self.input.coulomb_coefficients.shape()[2];
        if channel >= channels {
            Err(AtomMathError::CoefficientChannelOutOfRange {
                rank,
                channel,
                channels,
            })
        } else {
            Ok(channel)
        }
    }

    fn exchange_breit_branch(&self, orbital: usize) -> Result<BreitExchangeBranch, AtomMathError> {
        let mut angular_l = abs_kappa_i32(self.kappa(orbital))?;
        let mut sign_shift = -1;
        if self.kappa(orbital) < 0 {
            sign_shift = 1;
            angular_l -= 1;
        }
        Ok(BreitExchangeBranch {
            angular_l,
            sign_shift,
        })
    }

    fn orbital_count(&self) -> usize {
        self.input.kappas.len()
    }

    fn kappa(&self, orbital: usize) -> i32 {
        self.input.kappas[orbital]
    }

    fn abs_kappa(&self, orbital: usize) -> Result<usize, AtomMathError> {
        usize::try_from(abs_kappa_i32(self.kappa(orbital))?).map_err(|_| {
            AtomMathError::InvalidKappa {
                kappa: self.kappa(orbital),
            }
        })
    }

    fn j2(&self, orbital: usize) -> Result<usize, AtomMathError> {
        usize::try_from(doubled_j_from_kappa(self.kappa(orbital))?).map_err(|_| {
            AtomMathError::InvalidKappa {
                kappa: self.kappa(orbital),
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BreitExchangeBranch {
    angular_l: i32,
    sign_shift: i32,
}

impl BreitExchangeBranch {
    fn minimum_rank(self, other: Self) -> Result<usize, AtomMathError> {
        let first = other
            .angular_l
            .checked_add(other.sign_shift)
            .and_then(|value| value.checked_sub(self.angular_l))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?
            .unsigned_abs();
        let second = self
            .angular_l
            .checked_add(self.sign_shift)
            .and_then(|value| value.checked_sub(other.angular_l))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?
            .unsigned_abs();
        usize::try_from(first.min(second)).map_err(|_| AtomMathError::BreitBranchOutOfRange)
    }

    fn maximum_rank(self, other: Self) -> Result<usize, AtomMathError> {
        let first = self
            .angular_l
            .checked_add(other.angular_l)
            .and_then(|value| value.checked_add(other.sign_shift))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?;
        let second = self
            .angular_l
            .checked_add(other.angular_l)
            .and_then(|value| value.checked_add(self.sign_shift))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?;
        usize::try_from(first.max(second)).map_err(|_| AtomMathError::BreitBranchOutOfRange)
    }
}
