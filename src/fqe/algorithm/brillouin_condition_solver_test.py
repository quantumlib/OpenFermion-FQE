import pytest
import numpy as np
from fqe.algorithm.brillouin_condition_solver import BrillouinCondition

import openfermion as of

import fqe
from fqe.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata)


@pytest.mark.skip(reason='slow-test')
def test_solver():
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    sz = 0
    oei, tei = molecule.get_integrals()
    elec_hamil = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (oei, np.einsum('ijlk', -0.5 * tei)))
    moleham = molecule.get_molecular_hamiltonian()
    reduced_ham = of.chem.make_reduced_hamiltonian(moleham,
                                                   molecule.n_electrons)

    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    graph = fqe_wf.sector((n_electrons, sz)).get_fcigraph()
    fci_coeffs = np.zeros((graph.lena(), graph.lenb()), dtype=np.complex128)
    fci_coeffs[0, 0] = 1.
    fqe_wf.set_wfn(strategy='from_data',
                   raw_data={(n_electrons, sz): fci_coeffs})

    bcsolve = BrillouinCondition(molecule, run_parallel=False)
    assert bcsolve.reduced_ham == reduced_ham
    for pp, qq in zip(elec_hamil.tensors(), bcsolve.elec_hamil.tensors()):
        assert np.allclose(pp, qq)

    bcsolve.iter_max = 1
    bcsolve.bc_solve(fqe_wf)
    assert np.allclose(bcsolve.acse_energy, [-8.957417182801088,
                                             -8.969256797233033])
