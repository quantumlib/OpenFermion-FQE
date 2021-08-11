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
""" Test VBC
"""
import copy
from itertools import product
import numpy as np
import openfermion as of
import pytest

from fqe import get_number_conserving_wavefunction, vdot, Wavefunction, \
                get_restricted_hamiltonian
from fqe.algorithm.low_rank import evolve_fqe_givens_unrestricted
from fqe.algorithm.vbc import VBC
from tests.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata, build_h4square_moleculardata)
from fqe.algorithm.brillouin_calculator import two_rdo_commutator_symm
from fqe.fqe_decorators import build_hamiltonian
from tests.unittest_data import build_wfn
from fqe.algorithm.algorithm_util import valdemaro_reconstruction


@pytest.fixture
def vbc_fixture():
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    oei, tei = molecule.get_integrals()
    nalpha = molecule.n_electrons // 2
    nbeta = nalpha
    sz = nalpha - nbeta

    fqe_wf = Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')

    adapt = VBC(oei, tei, nalpha, nbeta, iter_max=1, verbose=False)

    return fqe_wf, adapt


def test_vbc(vbc_fixture):
    """ Test vbc with default parameters.
    """
    fqe_wf, adapt = vbc_fixture
    adapt.vbc(fqe_wf)
    assert np.isclose(adapt.energies[0], -8.957417182801091)
    assert np.isclose(adapt.energies[-1], -8.97304439380826)


def test_vbc_valdemaro(vbc_fixture):
    """ Test valdemaro reconstruction.
    """
    fqe_wf, adapt = vbc_fixture
    adapt.vbc(fqe_wf, v_reconstruct=True)
    assert np.isclose(adapt.energies[0], -8.957417182801091)
    assert np.isclose(adapt.energies[-1], -8.97304439380826)


def test_vbc_wrong_decomposition_raises(vbc_fixture):
    """ Test if vbc raises an exception on incorrect decomposition method
    """
    fqe_wf, adapt = vbc_fixture
    with pytest.raises(ValueError):
        adapt.vbc(fqe_wf, v_reconstruct=True, generator_decomp='test')


@pytest.mark.skip(reason="does not work, tries to pass SumOfSquaresOperator"
                  "as a FermionOperator in optimize_param() ")
def test_vbc_svd(vbc_fixture):
    """ Test VBC with takagi generator decomposition.
    """
    fqe_wf, adapt = vbc_fixture
    adapt.vbc(fqe_wf, v_reconstruct=True, generator_decomp='svd')
    assert np.isclose(adapt.energies[0], -8.957417182801091)
    assert np.isclose(adapt.energies[-1], -8.97304439380826)


@pytest.mark.skip(reason="does not work, tries to pass SumOfSquaresOperator"
                  "as a FermionOperator in optimize_param() ")
def test_vbc_takagi(vbc_fixture):
    """ Test VBC with takagi generator decomposition.
    """
    fqe_wf, adapt = vbc_fixture
    adapt.vbc(fqe_wf, v_reconstruct=True, generator_decomp='takagi')
    assert np.isclose(adapt.energies[0], -8.957417182801091)
    assert np.isclose(adapt.energies[-1], -8.97304439380826)


def test_vbc_opt_var(vbc_fixture):
    """ Test vbc variable optimisation.
        Warning:
        This test produces energies identical to those with num_opt_var=0,
        so it probably does not check the logic properly yet.
    """
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    oei, tei = molecule.get_integrals()
    nalpha = molecule.n_electrons // 2
    nbeta = nalpha
    sz = nalpha - nbeta

    fqe_wf = Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')

    adapt = VBC(oei, tei, nalpha, nbeta, iter_max=2, verbose=False)
    adapt.vbc(fqe_wf, v_reconstruct=True, num_opt_var=1)
    assert np.isclose(adapt.energies[0], -8.957417182801091)
    assert np.isclose(adapt.energies[1], -8.97304439380826)
    assert np.isclose(adapt.energies[2], -8.97482142285195)


def test_vbc_takagi_decomps():
    """ Test Takagi decomposition
    """
    molecule = build_h4square_moleculardata()
    oei, tei = molecule.get_integrals()
    nele = 4
    nalpha = 2
    nbeta = 2
    sz = 0
    norbs = oei.shape[0]
    nso = 2 * norbs
    fqe_wf = Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.normalize()
    _, tpdm = fqe_wf.sector((nele, sz)).get_openfermion_rdms()
    d3 = fqe_wf.sector((nele, sz)).get_three_pdm()

    adapt = VBC(oei, tei, nalpha, nbeta, iter_max=50)
    acse_residual = two_rdo_commutator_symm(adapt.reduced_ham.two_body_tensor,
                                            tpdm, d3)
    for p, q, r, s in product(range(nso), repeat=4):
        if p == q or r == s:
            continue
        assert np.isclose(acse_residual[p, q, r, s],
                          -acse_residual[s, r, q, p].conj())

    sos_op = adapt.get_takagi_tensor_decomp(acse_residual, None)

    # reconstruct tensor from sop
    test_tensor = np.zeros_like(acse_residual)
    for v, cc in zip(sos_op.basis_rotation, sos_op.charge_charge):
        vc = v.conj()
        test_tensor += np.einsum('pi,si,ij,qj,rj->pqrs', v, vc, -1j * cc, v, vc)
    assert np.allclose(test_tensor, acse_residual)


def test_vbc_svd_decomps():
    molecule = build_h4square_moleculardata()
    oei, tei = molecule.get_integrals()
    nele = 4
    nalpha = 2
    nbeta = 2
    sz = 0
    norbs = oei.shape[0]
    nso = 2 * norbs
    fqe_wf = Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.normalize()
    _, tpdm = fqe_wf.sector((nele, sz)).get_openfermion_rdms()
    d3 = fqe_wf.sector((nele, sz)).get_three_pdm()

    adapt = VBC(oei, tei, nalpha, nbeta, iter_max=50)
    acse_residual = two_rdo_commutator_symm(adapt.reduced_ham.two_body_tensor,
                                            tpdm, d3)
    new_residual = np.zeros_like(acse_residual)
    for p, q, r, s in product(range(nso), repeat=4):
        new_residual[p, q, r, s] = (acse_residual[p, q, r, s] -
                                    acse_residual[s, r, q, p]) / 2

    sos_op = adapt.get_svd_tensor_decomp(new_residual, None)

    # reconstruct tensor from sop
    test_tensor = np.zeros_like(new_residual)
    for v, cc in zip(sos_op.basis_rotation, sos_op.charge_charge):
        vc = v.conj()
        test_tensor += np.einsum('pi,si,ij,qj,rj->pqrs', v, vc, -1j * cc, v, vc)
    assert np.allclose(test_tensor, new_residual)


def test_vbc_time_evolve():
    molecule = build_h4square_moleculardata()
    oei, tei = molecule.get_integrals()
    nele = molecule.n_electrons
    nalpha = nele // 2
    nbeta = nele // 2
    sz = 0
    norbs = oei.shape[0]
    nso = 2 * norbs
    fqe_wf = Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.normalize()
    nfqe_wf = get_number_conserving_wavefunction(nele, norbs)
    nfqe_wf.sector((nele, sz)).coeff = fqe_wf.sector((nele, sz)).coeff
    _, tpdm = nfqe_wf.sector((nele, sz)).get_openfermion_rdms()
    d3 = nfqe_wf.sector((nele, sz)).get_three_pdm()

    adapt = VBC(oei, tei, nalpha, nbeta, iter_max=50)
    acse_residual = two_rdo_commutator_symm(adapt.reduced_ham.two_body_tensor,
                                            tpdm, d3)
    sos_op = adapt.get_takagi_tensor_decomp(acse_residual, None)

    test_wf = copy.deepcopy(nfqe_wf)
    test_wf = sos_op.time_evolve(test_wf)

    true_wf = copy.deepcopy(nfqe_wf)
    for v, cc in zip(sos_op.basis_rotation, sos_op.charge_charge):
        vc = v.conj()
        new_tensor = np.einsum('pi,si,ij,qj,rj->pqrs', v, vc, -1j * cc, v, vc)
        if np.isclose(np.linalg.norm(new_tensor), 0):
            continue
        fop = of.FermionOperator()
        for p, q, r, s in product(range(nso), repeat=4):
            op = ((p, 1), (s, 0), (q, 1), (r, 0))
            fop += of.FermionOperator(op, coefficient=new_tensor[p, q, r, s])
        fqe_op = build_hamiltonian(1j * fop, conserve_number=True)
        true_wf = true_wf.time_evolve(1, fqe_op)
    true_wf = evolve_fqe_givens_unrestricted(true_wf, sos_op.one_body_rotation)

    assert np.isclose(abs(vdot(true_wf, test_wf))**2, 1)
