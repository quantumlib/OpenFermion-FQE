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
""" algorithm/davidson unit tests
"""
import pytest
import fqe
import openfermion as of
import numpy as np
import copy
from tests.unittest_data.build_lih_data import build_lih_data
from fqe.algorithm import davidson


def test_davidson():
    eref = -8.877719570384043
    norb = 6
    nalpha = 2
    nbeta = 2
    sz = nalpha - nbeta
    nele = nalpha + nbeta
    h1e, h2e, lih_ground = build_lih_data("energy")
    elec_hamil = fqe.restricted_hamiltonian.RestrictedHamiltonian((h1e, h2e))
    wfn = fqe.Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy="from_data",
                raw_data={(nele, nalpha - nbeta): lih_ground})
    graph = wfn.sector((4, 0)).get_fcigraph()
    ecalc = wfn.expectationValue(elec_hamil)
    assert abs(eref - ecalc) < 1e-10

    # Generate Guess Vecs for Davidson-Liu
    guess_vec1_coeffs = np.zeros((graph.lena(), graph.lenb()),
                                 dtype=np.complex128)
    guess_vec2_coeffs = np.zeros((graph.lena(), graph.lenb()),
                                 dtype=np.complex128)
    alpha_hf = fqe.util.init_bitstring_groundstate(2)
    beta_hf = fqe.util.init_bitstring_groundstate(2)
    guess_vec1_coeffs[graph.index_alpha(alpha_hf),
                      graph.index_beta(beta_hf)] = 1.0
    guess_vec2_coeffs[graph.index_alpha(alpha_hf << 1),
                      graph.index_beta(beta_hf << 1)] = 1.0

    guess_wfn1 = copy.deepcopy(wfn)
    guess_wfn2 = copy.deepcopy(wfn)
    guess_wfn1.set_wfn(
        strategy="from_data",
        raw_data={(nele, nalpha - nbeta): guess_vec1_coeffs},
    )
    guess_wfn2.set_wfn(
        strategy="from_data",
        raw_data={(nele, nalpha - nbeta): guess_vec2_coeffs},
    )
    guess_vecs = [guess_wfn1, guess_wfn2]
    dl_w, dl_v = davidson.davidsonliu_fqe(elec_hamil, 1, guess_vecs, nele=nele,
                                          sz=sz, norb=norb, verbose=True)
    assert abs(dl_w - ecalc) < 1e-10

    # dummy geometry
    geometry = [["Li", [0, 0, 0], ["H", [0, 0, 1.4]]]]
    charge = 0
    multiplicity = 1
    molecule = of.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        charge=charge,
        multiplicity=multiplicity,
    )
    molecule.one_body_integrals = h1e
    molecule.two_body_integrals = np.einsum("ijlk", -2 * h2e)
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    molecular_hamiltonian.constant = 0
    ham_fop = of.get_fermion_operator(molecular_hamiltonian)
    ham_mat = of.get_sparse_operator(of.jordan_wigner(ham_fop)).toarray()

    cirq_ci = fqe.to_cirq(wfn)
    cirq_ci = cirq_ci.reshape((2**12, 1))
    assert np.isclose(cirq_ci.conj().T @ ham_mat @ cirq_ci, ecalc)

    hf_idx = int("111100000000", 2)
    hf_idx2 = int("111001000000", 2)
    hf_vec = np.zeros((2**12, 1))
    hf_vec2 = np.zeros((2**12, 1))
    hf_vec[hf_idx, 0] = 1.0
    hf_vec2[hf_idx2, 0] = 1.0

    # scale diagonal so vacuum has non-zero energy
    ww, vv = davidson.davidsonliu(ham_mat + np.eye(ham_mat.shape[0]), 1,
                                  guess_vecs=[hf_vec, hf_vec2], verbose=True)
    assert abs(ww - 1 - ecalc) < 1e-10

    api_w, api_v = davidson.davidson_diagonalization(hamiltonian=elec_hamil,
                                                     n_alpha=nalpha,
                                                     n_beta=nbeta)
    assert abs(api_w - dl_w) < 1e-10


def test_davidson_error():
    h1e, h2e, lih_ground = build_lih_data("energy")
    with pytest.raises(ValueError):
        davidson.davidsonliu(h1e, 0)
    with pytest.raises(ValueError):
        davidson.davidsonliu(h1e, h1e.shape[0] // 2 + 1)


@pytest.mark.skip(reason='broken')
def test_davidson_no_init():
    # Does not handle well when the initial guesses live in separate
    # unconnected spaces by the Hamiltonian. It does not detect that It has
    # exhausted the search in the particle-spaces of the initial guesses.

    eref = -8.877719570384043
    norb = 6
    nalpha = 2
    nbeta = 2
    sz = nalpha - nbeta
    nele = nalpha + nbeta
    h1e, h2e, lih_ground = build_lih_data("energy")
    # dummy geometry
    geometry = [["Li", [0, 0, 0], ["H", [0, 0, 1.4]]]]
    charge = 0
    multiplicity = 1
    molecule = of.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        charge=charge,
        multiplicity=multiplicity,
    )
    molecule.one_body_integrals = h1e
    molecule.two_body_integrals = np.einsum("ijlk", -2 * h2e)
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    molecular_hamiltonian.constant = 0
    ham_fop = of.get_fermion_operator(molecular_hamiltonian)
    ham_mat = of.get_sparse_operator(of.jordan_wigner(ham_fop)).toarray()

    # scale diagonal so vacuum has non-zero energy
    ww, vv = davidson.davidsonliu(ham_mat + np.eye(ham_mat.shape[0]), 1,
                                  verbose=True)
    assert abs(ww - 1 - ecalc) < 1e-10


if __name__ == "__main__":
    test_davidson()
