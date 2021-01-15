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
"""Unit tests for BrillouinCondition."""

import numpy as np

import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

import fqe
from fqe.algorithm.brillouin_condition_solver import BrillouinCondition
from fqe.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata,)


def test_solver():
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    sz = 0
    oei, tei = molecule.get_integrals()
    elec_hamil = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (oei, np.einsum("ijlk", -0.5 * tei)))
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(molecular_hamiltonian,
                                                   molecule.n_electrons)

    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    graph = fqe_wf.sector((n_electrons, sz)).get_fcigraph()
    fci_coeffs = np.zeros((graph.lena(), graph.lenb()), dtype=np.complex128)
    fci_coeffs[0, 0] = 1.0
    fqe_wf.set_wfn(strategy="from_data",
                   raw_data={(n_electrons, sz): fci_coeffs})

    bcsolve = BrillouinCondition(molecule, run_parallel=False)
    assert bcsolve.reduced_ham == reduced_ham
    for pp, qq in zip(elec_hamil.tensors(), bcsolve.elec_hamil.tensors()):
        assert np.allclose(pp, qq)

    bcsolve.iter_max = 1
    bcsolve.bc_solve(fqe_wf)
    assert np.allclose(bcsolve.acse_energy,
                       [-8.957417182801088, -8.969256797233033])


def test_solver_via_rdms():
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    sz = 0
    oei, tei = molecule.get_integrals()
    elec_hamil = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (oei, np.einsum("ijlk", -0.5 * tei)))
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(molecular_hamiltonian,
                                                   molecule.n_electrons)

    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    graph = fqe_wf.sector((n_electrons, sz)).get_fcigraph()
    fci_coeffs = np.zeros((graph.lena(), graph.lenb()), dtype=np.complex128)
    fci_coeffs[0, 0] = 1.0
    fqe_wf.set_wfn(strategy="from_data",
                   raw_data={(n_electrons, sz): fci_coeffs})

    bcsolve = BrillouinCondition(molecule, run_parallel=False)
    assert bcsolve.reduced_ham == reduced_ham
    for pp, qq in zip(elec_hamil.tensors(), bcsolve.elec_hamil.tensors()):
        assert np.allclose(pp, qq)

    bcsolve.iter_max = 1
    bcsolve.bc_solve_rdms(fqe_wf)
    assert np.allclose(bcsolve.acse_energy,
                       [-8.957417182801088, -8.969256797233033])
