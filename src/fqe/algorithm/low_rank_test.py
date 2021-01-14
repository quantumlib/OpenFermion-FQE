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
"""Unit tests for low_rank.py."""

from itertools import product
import copy
import numpy as np
import pytest
from scipy.linalg import expm

import openfermion as of
from openfermion import givens_decomposition_square
from openfermion.testing.testing_utils import (
    random_quadratic_hamiltonian,
    random_unitary_matrix,
    random_hermitian_matrix,
)

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.algorithm.low_rank import (
    evolve_fqe_givens,
    evolve_fqe_diagaonal_coulomb,
    double_factor_trotter_evolution,
)


def evolve_wf_givens(wfn: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Utility for testing evolution of a full 2^{n} wavefunction.

    Args:
        wfn: 2^{n} x 1 vector.
        u: (n//2 x n//2) unitary matrix.

    Returns:
        New evolved 2^{n} x 1 vector.
    """
    rotations, diagonal = givens_decomposition_square(u.copy())
    n_qubits = u.shape[0] * 2
    # Iterate through each layer and time evolve by the appropriate
    # fermion operators
    for layer in rotations:
        for givens in layer:
            i, j, theta, phi = givens
            op = of.FermionOperator(((2 * j, 1), (2 * j, 0)), coefficient=-phi)
            op += of.FermionOperator(((2 * j + 1, 1), (2 * j + 1, 0)),
                                     coefficient=-phi)
            wfn = (
                expm(-1j * of.get_sparse_operator(op, n_qubits=n_qubits)) @ wfn)

            op = of.FermionOperator(
                ((2 * i, 1),
                 (2 * j, 0)), coefficient=-1j * theta) + of.FermionOperator(
                     ((2 * j, 1), (2 * i, 0)), coefficient=1j * theta)
            op += of.FermionOperator(
                ((2 * i + 1, 1),
                 (2 * j + 1, 0)), coefficient=-1j * theta) + of.FermionOperator(
                     ((2 * j + 1, 1), (2 * i + 1, 0)), coefficient=1j * theta)
            wfn = (
                expm(-1j * of.get_sparse_operator(op, n_qubits=n_qubits)) @ wfn)

    # evolve the last diagonal phases
    for idx, final_phase in enumerate(diagonal):
        if not np.isclose(final_phase, 1.0):
            op = of.FermionOperator(((2 * idx, 1), (2 * idx, 0)),
                                    -np.angle(final_phase))
            op += of.FermionOperator(((2 * idx + 1, 1), (2 * idx + 1, 0)),
                                     -np.angle(final_phase))
            wfn = (
                expm(-1j * of.get_sparse_operator(op, n_qubits=n_qubits)) @ wfn)

    return wfn


def evolve_wf_diagonal_coulomb(wf: np.ndarray, vij_mat: np.ndarray,
                               time=1) -> np.ndarray:
    r"""Utility for testing evolution of a full 2^{n} wavefunction via

    :math:`exp{-i time * \sum_{i,j, sigma, tau}v_{i, j}n_{i\sigma}n_{j\tau}}.`

    Args:
        wf: 2^{n} x 1 vector
        vij_mat: List[(n//2 x n//2)] matrices

    Returns:
        New evolved 2^{n} x 1 vector
    """
    norbs = int(np.log2(wf.shape[0]) / 2)
    diagonal_coulomb = of.FermionOperator()
    for i, j in product(range(norbs), repeat=2):
        for sigma, tau in product(range(2), repeat=2):
            diagonal_coulomb += of.FermionOperator(
                (
                    (2 * i + sigma, 1),
                    (2 * i + sigma, 0),
                    (2 * j + tau, 1),
                    (2 * j + tau, 0),
                ),
                coefficient=vij_mat[i, j],
            )
    bigU = expm(-1j * time *
                of.get_sparse_operator(diagonal_coulomb, n_qubits=2 * norbs))
    return bigU @ wf


def double_factor_trotter_wf_evolution(initial_wfn: np.ndarray,
                                       basis_change_unitaries, vij_mats,
                                       deltat) -> np.ndarray:
    r"""Doubled Factorized Trotter Evolution.

    This is for testing the FQE evolution. Same input except the initial
    wavefunction should be the full 2^{2 * norbs) space column vector.

    Args:
        initial_wfn: Initial wavefunction to evolve.
        basis_change_unitaries: List L + 1 unitaries. The first
            unitary is U1 :math:`e^{-iTdt}` where T is the one-electron
            component of the evolution.  he remaining unitaries are
            :math:`U_{i}U_{i-1}^{\dagger}.` All unitaries are expressed with
            respect to the number of spatial basis functions.
        vij_mats: list matrices of rho-rho interactions where
            i, j indices of the matrix index the :math:`n_{i} n_{j}` integral.
            Evolution is performed with respect to :math:`n_{i\sigma} n_{j\tau}`
            where sigma and tau are up or down electron spins--a total of 4
            Hamiltonian terms per i, j term.
        deltat: evolution time prefactor for all v_ij Hamiltonians.

    Returns:
        The final wavefunction from a single Trotter evolution.
    """
    intermediate_wfn = evolve_wf_givens(initial_wfn, basis_change_unitaries[0])
    for step in range(1, len(basis_change_unitaries)):
        intermediate_wfn = evolve_wf_diagonal_coulomb(intermediate_wfn,
                                                      vij_mats[step - 1],
                                                      deltat)
        intermediate_wfn = evolve_wf_givens(intermediate_wfn,
                                            basis_change_unitaries[step])
    return intermediate_wfn


def test_fqe_givens():
    """Test Givens Rotation evolution for correctness."""
    # set up
    norbs = 4
    n_elec = norbs
    sz = 0
    n_qubits = 2 * norbs
    time = 0.126
    fqe_wfn = fqe.Wavefunction([[n_elec, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    ikappa = random_quadratic_hamiltonian(
        norbs,
        conserves_particle_number=True,
        real=False,
        expand_spin=False,
        seed=2,
    )
    fqe_ham = RestrictedHamiltonian((ikappa.n_body_tensors[1, 0],))
    u = expm(-1j * ikappa.n_body_tensors[1, 0] * time)

    # time-evolve
    final_fqe_wfn = fqe_wfn.time_evolve(time, fqe_ham)
    spin_ham = np.kron(ikappa.n_body_tensors[1, 0], np.eye(2))
    assert of.is_hermitian(spin_ham)
    ikappa_spin = of.InteractionOperator(
        constant=0,
        one_body_tensor=spin_ham,
        two_body_tensor=np.zeros((n_qubits, n_qubits, n_qubits, n_qubits)),
    )
    bigU = expm(-1j * of.get_sparse_operator(ikappa_spin).toarray() * time)
    initial_wf = fqe.to_cirq(fqe_wfn).reshape((-1, 1))
    final_wf = bigU @ initial_wf
    final_wfn_test = fqe.from_cirq(final_wf.flatten(), 1.0e-12)

    assert np.allclose(final_fqe_wfn.rdm("i^ j"), final_wfn_test.rdm("i^ j"))
    assert np.allclose(final_fqe_wfn.rdm("i^ j^ k l"),
                       final_wfn_test.rdm("i^ j^ k l"))

    final_wfn_test2 = fqe.from_cirq(
        evolve_wf_givens(initial_wf.copy(), u.copy()).flatten(), 1.0e-12)
    givens_fqe_wfn = evolve_fqe_givens(fqe_wfn, u.copy())
    assert np.allclose(givens_fqe_wfn.rdm("i^ j"), final_wfn_test2.rdm("i^ j"))
    assert np.allclose(givens_fqe_wfn.rdm("i^ j^ k l"),
                       final_wfn_test2.rdm("i^ j^ k l"))


def test_charge_charge_evolution():
    norbs = 4
    n_elec = norbs
    sz = 0
    time = 0.126
    fqe_wfn = fqe.Wavefunction([[n_elec, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    initial_fqe_wfn = copy.deepcopy(fqe_wfn)
    initial_wf = fqe.to_cirq(fqe_wfn).reshape((-1, 1))

    # time-evolve

    vij = np.random.random((norbs, norbs))
    vij = vij + vij.T
    final_fqe_wfn = evolve_fqe_diagaonal_coulomb(initial_fqe_wfn, vij, time)
    test_wfn = evolve_wf_diagonal_coulomb(wf=initial_wf, vij_mat=vij, time=time)
    test_wfn = fqe.from_cirq(test_wfn.flatten(), 1.0e-12)

    assert np.allclose(final_fqe_wfn.rdm("i^ j"), test_wfn.rdm("i^ j"))
    assert np.allclose(final_fqe_wfn.rdm("i^ j^ k l"),
                       test_wfn.rdm("i^ j^ k l"))


def test_double_factorization_trotter():
    norbs = 4
    n_elec = norbs
    sz = 0
    time = 0.126
    fqe_wfn = fqe.Wavefunction([[n_elec, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    fqe_wfn.print_wfn()
    initial_wf = fqe.to_cirq_ncr(fqe_wfn).reshape((-1, 1))

    basis_change_unitaries = []
    for ii in range(2):
        basis_change_unitaries.append(
            random_unitary_matrix(4, real=False, seed=ii))
    vij_mats = []
    for ii in range(1):
        vij_mats.append(random_hermitian_matrix(norbs, real=True, seed=ii))

    with pytest.raises(ValueError):
        _ = double_factor_trotter_evolution(fqe_wfn,
                                            basis_change_unitaries + [0],
                                            vij_mats=vij_mats,
                                            deltat=0)

    final_wfn = double_factor_trotter_evolution(
        initial_wfn=fqe_wfn,
        basis_change_unitaries=basis_change_unitaries,
        vij_mats=vij_mats,
        deltat=time,
    )

    intermediate_wfn = evolve_wf_givens(initial_wf.copy(),
                                        basis_change_unitaries[0])
    for step in range(1, len(basis_change_unitaries)):
        intermediate_wfn = evolve_wf_diagonal_coulomb(intermediate_wfn,
                                                      vij_mats[step - 1], time)
        intermediate_wfn = evolve_wf_givens(intermediate_wfn,
                                            basis_change_unitaries[step])

    test_final_wfn = fqe.from_cirq(intermediate_wfn.flatten(), 1.0e-12)
    assert np.allclose(final_wfn.rdm("i^ j"), test_final_wfn.rdm("i^ j"))
    assert np.allclose(final_wfn.rdm("i^ j^ k l"),
                       test_final_wfn.rdm("i^ j^ k l"))

    test_final_wfn = double_factor_trotter_wf_evolution(initial_wf.copy(),
                                                        basis_change_unitaries,
                                                        vij_mats, time)
    test_final_wfn = fqe.from_cirq(test_final_wfn.flatten(), 1.0e-12)
    assert np.allclose(final_wfn.rdm("i^ j"), test_final_wfn.rdm("i^ j"))
    assert np.allclose(final_wfn.rdm("i^ j^ k l"),
                       test_final_wfn.rdm("i^ j^ k l"))
