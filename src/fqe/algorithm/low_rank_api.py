#   Copyright 2020 Google LLC

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""API for Low Rank Trotter steps.

There are a variety of tunable parameters for the low rank
Trotter steps--L, M, epsilon, lambda.

This set of routines interfaces with OpenFermion to provide
data containers and transformers from molecular integrals to
low-rank trotter step data structures.
"""
from typing import Optional

import numpy as np
from scipy.linalg import expm

from openfermion import MolecularData
from openfermion import (
    low_rank_two_body_decomposition,
    prepare_one_body_squared_evolution,
)


class LowRankTrotter:
    """Holds data for low rank Trotter step generation and analysis specific
    to running analysis through the FQE.  Determination of basis transforms
    and matrices relies heavily on OpenFermion low-rank routines.
    """

    def __init__(
            self,
            molecule: Optional[MolecularData] = None,
            oei: Optional[np.ndarray] = None,
            tei: Optional[np.ndarray] = None,
            integral_cutoff: Optional[float] = 1.0e-8,
            basis_cutoff: Optional[float] = 1.0e-8,
            spin_basis: Optional[bool] = False,
    ):
        self.molecule = molecule
        self.oei = oei
        self.tei = tei
        self.icut = integral_cutoff
        self.lmax = None
        self.mcut = basis_cutoff
        self.mmax = None
        self.spin_basis = spin_basis

        # if only molecule is provided get the spatial-MO matrices
        if molecule is not None and oei is None and tei is None:
            self.oei, self.tei = molecule.get_integrals()

    def first_factorization(self, threshold: Optional[float] = None):
        r"""Factorize :math:`V = 1/2 \sum_{ijkl, st}V_{ijkl} is^ jt^ kt ls` by
        transforming to chemist notation.

        Args:
            threshold: threshold for factorization.

        Returns:
            Tuple of (eigenvalues of factors, one-body ops in factors, one
                      body correction).
        """
        if threshold is None:
            threshold = self.icut

        # convert physics notation integrals into chemist notation
        # and determine the first low-rank factorization
        if self.spin_basis:
            (
                eigenvalues,
                one_body_squares,
                one_body_correction,
                _,
            ) = low_rank_two_body_decomposition(
                self.tei,
                truncation_threshold=threshold,
                final_rank=self.lmax,
                spin_basis=self.spin_basis,
            )
        else:
            (
                eigenvalues,
                one_body_squares,
                one_body_correction,
                _,
            ) = low_rank_two_body_decomposition(
                0.5 * self.tei,  # type: ignore
                truncation_threshold=threshold,
                final_rank=self.lmax,
                spin_basis=self.spin_basis,
            )
        return eigenvalues, one_body_squares, one_body_correction

    def second_factorization(
            self,
            eigenvalues: np.ndarray,
            one_body_squares: np.ndarray,
            threshold: Optional[float] = None,
    ):
        r"""
        Get Givens angles and DiagonalHamiltonian to simulate squared one-body.

        The goal here will be to prepare to simulate evolution under
        :math:`(\sum_{pq} h_{pq} a^\dagger_p a_q)^2` by decomposing as
        :math:`R e^{-i \sum_{pq} V_{pq} n_p n_q} R^\dagger' where
        :math:`R` is a basis transformation matrix.

        Args:
            eigenvalues: eigenvalues of 2nd quantized op
            one_body_squares: one-body-ops to square
            threshold: cutoff threshold.
                TODO: WARNING: THIS IS NOT USED CURRENTLY

        Returns:
            Tuple(List[np.ndarray], List[np.ndarray]) scaled-rho-rho spatial
            matrix and list of spatial basis transformations
        """
        if threshold is None:
            # TODO: update OpenFermion to take cutoff
            threshold = self.mcut
        scaled_density_density_matrices = []
        basis_change_matrices = []
        for j in range(len(eigenvalues)):
            # Testing out constructing density-density
            (
                sdensity_density_matrix,
                sbasis_change_matrix,
            ) = prepare_one_body_squared_evolution(
                one_body_squares[j][::2, ::2], spin_basis=False)
            scaled_density_density_matrices.append(
                np.real(eigenvalues[j]) * sdensity_density_matrix)
            basis_change_matrices.append(sbasis_change_matrix)

        return scaled_density_density_matrices, basis_change_matrices

    def get_l_and_m(self, first_factor_cutoff, second_factor_cutoff):
        """
        Determine the L rank and M rank for an integral matrix

        Args:
            first_factor_cutoff: First factorization cumulative eigenvalue
                cutoff.
            second_factor_cutoff: Second factorization cumulative error cutoff.

        Returns:
            Return L and list of lists with M values for each L.
        """
        (
            _,
            one_body_squares,
            _,
        ) = self.first_factorization(first_factor_cutoff)

        m_factors = []
        for l in range(one_body_squares.shape[0]):
            w, _ = np.linalg.eigh(one_body_squares[l][::2, ::2])
            # Determine upper-bound on truncation errors that would occur
            # if we dropped the eigenvalues lower than some cumulative error
            cumulative_error_sum = np.cumsum(sorted(np.abs(w))[::-1])
            truncation_errors = cumulative_error_sum[-1] - cumulative_error_sum
            max_rank = 1 + np.argmax(truncation_errors <= second_factor_cutoff)
            m_factors.append(max_rank)
        return one_body_squares.shape[0], m_factors

    def prepare_trotter_sequence(self, delta_t: float):
        """Build the Trotter sequence for FQE Evolution.

        Args:
            delta_t: Time to evolve.

        Returns:
            Tuple(List[np.ndarray], List[np.ndarray]) both sets of lists are
            n x n matrices. The first list is the list of basis change
            operators and the second list is the density-density matrix-spatial
            format.
        """
        (
            eigenvalues,
            one_body_squares,
            one_body_correction,
        ) = self.first_factorization(self.icut)
        (
            scaled_density_density_matrices,
            basis_change_matrices,
        ) = self.second_factorization(eigenvalues, one_body_squares, self.mcut)

        trotter_basis_change = [
            basis_change_matrices[0] @ expm(
                -1j * delta_t * (self.oei + one_body_correction[::2, ::2]))
        ]
        time_scaled_rho_rho_matrices = []

        for ii in range(len(basis_change_matrices) - 1):
            trotter_basis_change.append(basis_change_matrices[ii + 1]
                                        @ basis_change_matrices[ii].conj().T)
            time_scaled_rho_rho_matrices.append(
                delta_t * scaled_density_density_matrices[ii])
        # get the last element
        time_scaled_rho_rho_matrices.append(
            delta_t * scaled_density_density_matrices[-1].astype(np.complex128))
        trotter_basis_change.append(basis_change_matrices[ii + 1].conj().T)

        return trotter_basis_change, time_scaled_rho_rho_matrices
