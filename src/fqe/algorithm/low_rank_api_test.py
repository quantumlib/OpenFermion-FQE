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
"""Unit tests for Low Rank Trotter Steps."""

from itertools import product

import numpy as np
from scipy.linalg import expm

import openfermion as of
from openfermion.config import EQ_TOLERANCE
from fqe.algorithm.low_rank_api import LowRankTrotter
from fqe.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata,)


def test_initialization():
    empty = LowRankTrotter()
    obj_attributes = ["molecule", "oei", "tei", "icut", "lmax", "mcut", "mmax"]
    for oa in obj_attributes:
        assert hasattr(empty, oa)

    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    molecule_load = LowRankTrotter(molecule=molecule)
    assert np.allclose(molecule_load.oei, oei)
    assert np.allclose(molecule_load.tei, tei)
    assert np.isclose(molecule_load.icut, 1.0e-8)
    assert np.isclose(molecule_load.mcut, 1.0e-8)


def test_first_factorization():
    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    lrt_obj = LowRankTrotter(oei=oei, tei=tei)
    (
        eigenvalues,
        one_body_squares,
        one_body_correction,
    ) = lrt_obj.first_factorization()

    # get true op
    n_qubits = molecule.n_qubits
    fermion_tei = of.FermionOperator()
    test_tei_tensor = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for p, q, r, s in product(range(oei.shape[0]), repeat=4):
        if np.abs(tei[p, q, r, s]) < EQ_TOLERANCE:
            coefficient = 0.0
        else:
            coefficient = tei[p, q, r, s] / 2.0

        for sigma, tau in product(range(2), repeat=2):
            if 2 * p + sigma == 2 * q + tau or 2 * r + tau == 2 * s + sigma:
                continue
            term = (
                (2 * p + sigma, 1),
                (2 * q + tau, 1),
                (2 * r + tau, 0),
                (2 * s + sigma, 0),
            )
            fermion_tei += of.FermionOperator(term, coefficient=coefficient)
            test_tei_tensor[2 * p + sigma, 2 * q + tau, 2 * r + tau, 2 * s +
                            sigma] = coefficient

    mol_ham = of.InteractionOperator(
        one_body_tensor=np.zeros((n_qubits, n_qubits)),
        two_body_tensor=test_tei_tensor,
        constant=0,
    )

    # check induced norm on operator
    checked_op = of.FermionOperator()
    for (
            p,
            q,
    ) in product(range(n_qubits), repeat=2):
        term = ((p, 1), (q, 0))
        coefficient = one_body_correction[p, q]
        checked_op += of.FermionOperator(term, coefficient)

    # Build back two-body component.
    for l in range(one_body_squares.shape[0]):
        one_body_operator = of.FermionOperator()
        for p, q in product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = one_body_squares[l, p, q]
            one_body_operator += of.FermionOperator(term, coefficient)
        checked_op += eigenvalues[l] * (one_body_operator**2)

    true_fop = of.normal_ordered(of.get_fermion_operator(mol_ham))
    difference = of.normal_ordered(checked_op - true_fop)
    assert np.isclose(0, difference.induced_norm())


def test_second_factorization():
    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    lrt_obj = LowRankTrotter(oei=oei, tei=tei)
    eigenvalues, one_body_squares, _ = lrt_obj.first_factorization()

    s_rho_rho, basis_changes = lrt_obj.second_factorization(
        eigenvalues, one_body_squares)
    true_basis_changes = []
    for l in range(one_body_squares.shape[0]):
        w, v = np.linalg.eigh(one_body_squares[l][::2, ::2])
        true_basis_changes.append(v.conj().T)
        assert np.allclose(v.conj().T, basis_changes[l])

        assert np.allclose(s_rho_rho[l], eigenvalues[l] * np.outer(w, w))


def test_trotter_prep():
    times = [0.1, 0.2, 0.3]
    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    lrt_obj = LowRankTrotter(oei=oei, tei=tei)
    (
        eigenvalues,
        one_body_squares,
        one_body_correction,
    ) = lrt_obj.first_factorization()
    (
        scaled_density_density_matrices,
        basis_change_matrices,
    ) = lrt_obj.second_factorization(eigenvalues, one_body_squares)

    for tt in times:
        # get values from function to test
        test_tbasis, test_srr = lrt_obj.prepare_trotter_sequence(tt)

        # compute true values
        trotter_basis_change = [
            basis_change_matrices[0] @ expm(
                -1j * tt * (lrt_obj.oei + one_body_correction[::2, ::2]))
        ]
        time_scaled_rho_rho_matrices = []
        for ii in range(len(basis_change_matrices) - 1):
            trotter_basis_change.append(basis_change_matrices[ii + 1]
                                        @ basis_change_matrices[ii].conj().T)
            time_scaled_rho_rho_matrices.append(
                tt * scaled_density_density_matrices[ii])
        time_scaled_rho_rho_matrices.append(tt *
                                            scaled_density_density_matrices[-1])
        trotter_basis_change.append(basis_change_matrices[ii + 1].conj().T)

        assert len(trotter_basis_change) == len(test_tbasis)
        assert len(time_scaled_rho_rho_matrices) == len(test_srr)
        # check against true values
        for t1, t2 in zip(trotter_basis_change, test_tbasis):
            assert np.allclose(t1, t2)
            assert np.allclose(t1.conj().T @ t1, np.eye(t1.shape[0]))

        for t1, t2 in zip(test_srr, time_scaled_rho_rho_matrices):
            assert np.allclose(t1, t2)
            assert of.is_hermitian(t1)


def test_get_l_m():
    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    lrt_obj = LowRankTrotter(oei=oei, tei=tei)
    num_l, m_list = lrt_obj.get_l_and_m(1.0e-8, 1.0e-8)
    assert isinstance(num_l, int)
    assert isinstance(m_list, list)
    assert len(m_list) == num_l
