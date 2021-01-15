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
"""Tests if the BC calculation is correct."""

from itertools import product

import pytest

import numpy as np
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

import fqe
from fqe.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata, build_h4square_moleculardata)

from fqe.algorithm.brillouin_calculator import (
    get_tpdm_grad_fqe_parallel, get_tpdm_grad_fqe,
    get_acse_residual_fqe_parallel, get_acse_residual_fqe, get_fermion_op,
    two_rdo_commutator, two_rdo_commutator_symm, two_rdo_commutator_antisymm,
    one_rdo_commutator_symm)

try:
    from joblib import Parallel, delayed

    PARALLELIZABLE = True
except ImportError:
    PARALLELIZABLE = False


def get_acse_residual(wf, hammat, nso):
    """Testing utilities"""
    adag = [
        of.get_sparse_operator(of.FermionOperator(((x, 1))), n_qubits=nso)
        for x in range(nso)
    ]
    alow = [op.conj().T for op in adag]
    acse_residual = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
    for p, q, r, s in product(range(nso), repeat=4):
        rdo = adag[p] @ adag[q] @ alow[r] @ alow[s]
        acse_residual[p, q, r, s] = wf.conj().T @ (rdo @ hammat -
                                                   hammat @ rdo) @ wf
    return acse_residual


def get_one_body_acse_residual(wf, hammat, nso):
    """Testing utilities"""
    adag = [
        of.get_sparse_operator(of.FermionOperator(((x, 1))), n_qubits=nso)
        for x in range(nso)
    ]
    alow = [op.conj().T for op in adag]
    acse_residual = np.zeros((nso, nso), dtype=np.complex128)
    for p, q in product(range(nso), repeat=2):
        rdo = adag[p] @ alow[q]
        acse_residual[p, q] = wf.conj().T @ (rdo @ hammat - hammat @ rdo) @ wf
    return acse_residual


def get_tpdm_grad(wf, acse_res, nso):
    """Testing utilities"""
    adag = [
        of.get_sparse_operator(of.FermionOperator(((x, 1))), n_qubits=nso)
        for x in range(nso)
    ]
    alow = [op.conj().T for op in adag]
    acse_residual = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
    for p, q, r, s in product(range(nso), repeat=4):
        rdo = adag[p] @ adag[q] @ alow[r] @ alow[s]
        acse_residual[p, q, r, s] = wf.conj().T @ (rdo @ acse_res -
                                                   acse_res @ rdo) @ wf
    return acse_residual


def get_particle_projected(wf, n_target):
    """Testing Utility"""
    dim = int(np.log2(wf.shape[0]))
    for ii in range(2**dim):
        ket = [int(x) for x in np.binary_repr(ii, width=dim)]
        if sum(ket) != n_target:
            wf[ii, 0] = 0
    wf /= np.linalg.norm(wf)
    return wf


def get_random_wf(n_qubits):
    wf = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
    wf /= np.linalg.norm(wf)
    assert np.isclose(wf.conj().T @ wf, 1)
    return wf.reshape((-1, 1))


def test_get_acse_residual_fqe():
    molecule = build_h4square_moleculardata()
    sdim = molecule.n_orbitals
    molehammat = of.get_sparse_operator(molecule.get_molecular_hamiltonian())
    oei, tei = molecule.get_integrals()
    elec_ham = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (oei, np.einsum('ijlk', -0.5 * tei)))
    fqe_wf = fqe.Wavefunction([[sdim, 0, sdim]])
    fqe_wf.set_wfn('random')
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).reshape((-1, 1))
    true_acse_residual = get_acse_residual(cirq_wf, molehammat, 2 * sdim)
    test_acse_residual = get_acse_residual_fqe(fqe_wf, elec_ham, sdim)
    assert np.allclose(true_acse_residual, test_acse_residual)

    if PARALLELIZABLE:
        test_acse_residual = get_acse_residual_fqe_parallel(
            fqe_wf, elec_ham, sdim)
        assert np.allclose(true_acse_residual, test_acse_residual)


@pytest.mark.skip(reason="slow test")
def test_get_acse_residual_fqe_lih():
    molecule = build_lih_moleculardata()
    sdim = molecule.n_orbitals
    molehammat = of.get_sparse_operator(molecule.get_molecular_hamiltonian())
    oei, tei = molecule.get_integrals()
    elec_ham = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (oei, np.einsum('ijlk', -0.5 * tei)))
    fqe_wf = fqe.Wavefunction([[sdim, 0, sdim]])
    fqe_wf.set_wfn('random')
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).reshape((-1, 1))
    true_acse_residual = get_acse_residual(cirq_wf, molehammat, 2 * sdim)
    test_acse_residual = get_acse_residual_fqe(fqe_wf, elec_ham, sdim)
    assert np.allclose(true_acse_residual, test_acse_residual)

    if PARALLELIZABLE:
        test_acse_residual = get_acse_residual_fqe_parallel(
            fqe_wf, elec_ham, sdim)
        assert np.allclose(true_acse_residual, test_acse_residual)


def test_get_tpdm_grad_residual_fqe():
    molecule = build_h4square_moleculardata()
    sdim = molecule.n_orbitals
    molehammat = of.get_sparse_operator(molecule.get_molecular_hamiltonian())
    fqe_wf = fqe.Wavefunction([[sdim, 0, sdim]])
    fqe_wf.set_wfn('random')
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).reshape((-1, 1))
    true_acse_residual = get_acse_residual(cirq_wf, molehammat, 2 * sdim)
    acse_res_op = get_fermion_op(true_acse_residual)
    acse_res_op_mat = of.get_sparse_operator(acse_res_op, n_qubits=2 * sdim)
    true_tpdm_grad = get_tpdm_grad(cirq_wf, acse_res_op_mat, 2 * sdim)
    test_tpdm_grad = get_tpdm_grad_fqe(fqe_wf, true_acse_residual, sdim)
    assert np.allclose(true_tpdm_grad, test_tpdm_grad)

    if PARALLELIZABLE:
        test_tpdm_grad = get_tpdm_grad_fqe_parallel(fqe_wf, true_acse_residual,
                                                    sdim)
        assert np.allclose(true_tpdm_grad, test_tpdm_grad)


@pytest.mark.skip(reason='slow-test')
def test_get_tpdm_grad_residual_fqe_lih():
    molecule = build_lih_moleculardata()
    sdim = molecule.n_orbitals
    molehammat = of.get_sparse_operator(molecule.get_molecular_hamiltonian())
    fqe_wf = fqe.Wavefunction([[sdim, 0, sdim]])
    fqe_wf.set_wfn('random')
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).reshape((-1, 1))
    true_acse_residual = get_acse_residual(cirq_wf, molehammat, 2 * sdim)
    acse_res_op = get_fermion_op(true_acse_residual)
    acse_res_op_mat = of.get_sparse_operator(acse_res_op, n_qubits=2 * sdim)
    true_tpdm_grad = get_tpdm_grad(cirq_wf, acse_res_op_mat, 2 * sdim)
    test_tpdm_grad = get_tpdm_grad_fqe(fqe_wf, true_acse_residual, sdim)
    assert np.allclose(true_tpdm_grad, test_tpdm_grad)

    if PARALLELIZABLE:
        test_tpdm_grad = get_tpdm_grad_fqe_parallel(fqe_wf, true_acse_residual,
                                                    sdim)
        assert np.allclose(true_tpdm_grad, test_tpdm_grad)


def test_to_cirq_fast():
    for norbs in range(4, 6):
        for nn in range(norbs // 2, norbs):
            fqe_wf = fqe.get_number_conserving_wavefunction(nn, norbs)
            fqe_wf.set_wfn('random')
            fqe_wf.normalize()
            cirq_wf = fqe.to_cirq_ncr(fqe_wf).reshape((-1, 1))
            true_cirq_wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
            assert np.isclose(abs(cirq_wf.conj().T @ true_cirq_wf), 1)


def test_get_acse_residual_rdm():
    # Set up
    np.random.seed(10)

    # get molecule and reduced Ham
    molecule = build_h4square_moleculardata()
    sdim = molecule.n_orbitals
    molehammat = of.get_sparse_operator(molecule.get_molecular_hamiltonian())
    oei, tei = molecule.get_integrals()
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.make_reduced_hamiltonian(molecular_hamiltonian,
                                              molecule.n_electrons)

    # Initialize a random wavefunction
    nalpha, nbeta = molecule.n_electrons // 2, molecule.n_electrons // 2
    sz = nalpha - nbeta
    fqe_wf = fqe.Wavefunction([[nalpha + nbeta, sz, sdim]])
    fqe_wf.set_wfn('random')
    # make the wavefunction real
    coeffs = fqe_wf.get_coeff((nalpha + nbeta, sz)).real
    fqe_wf.set_wfn(strategy='from_data',
                   raw_data={(nalpha + nbeta, sz): coeffs})
    fqe_wf.normalize()
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).reshape((-1, 1))

    # check that the Reduced Hamiltonian is antisymmetric
    for p, q, r, s in product(range(2 * sdim), repeat=4):
        assert np.isclose(reduced_ham.two_body_tensor[p, q, r, s],
                          -reduced_ham.two_body_tensor[q, p, r, s])
        assert np.isclose(reduced_ham.two_body_tensor[p, q, r, s],
                          -reduced_ham.two_body_tensor[p, q, s, r])
        assert np.isclose(reduced_ham.two_body_tensor[p, q, r, s],
                          reduced_ham.two_body_tensor[q, p, s, r])

    # get the fqe_data object and produce RDMs
    fqe_data = fqe_wf.sector((nalpha + nbeta, sz))
    d3 = fqe_data.get_three_pdm()
    _, tpdm = fqe_data.get_openfermion_rdms()

    # compare acse residual expressions from RDMs
    test_acse_residual = two_rdo_commutator(reduced_ham.two_body_tensor, tpdm,
                                            d3)
    test2_acse_residual = two_rdo_commutator_symm(reduced_ham.two_body_tensor,
                                                  tpdm, d3)
    true_acse_residual = get_acse_residual(cirq_wf, molehammat, 2 * sdim)
    assert np.allclose(true_acse_residual, test_acse_residual)
    assert np.allclose(true_acse_residual, test2_acse_residual)

    acse_res_op = get_fermion_op(true_acse_residual)
    acse_res_op_mat = of.get_sparse_operator(acse_res_op, n_qubits=2 * sdim)
    true_tpdm_grad = get_tpdm_grad(cirq_wf, acse_res_op_mat, 2 * sdim)
    test_tpdm_grad = two_rdo_commutator_antisymm(true_acse_residual, tpdm, d3)
    assert np.allclose(test_tpdm_grad, true_tpdm_grad)

    test_one_acse_residual = one_rdo_commutator_symm(
        reduced_ham.two_body_tensor, tpdm)
    true_one_acse_residual = get_one_body_acse_residual(cirq_wf, molehammat,
                                                        2 * sdim)
    assert np.allclose(true_one_acse_residual, test_one_acse_residual)
