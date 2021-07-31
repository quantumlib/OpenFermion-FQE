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
"""Tests for time evolution of various hamiltonians
"""
#accessing protected members is convenient for testing
#pylint: disable=protected-access

import copy
import numpy
import pytest
from numpy import linalg
from scipy import special

from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated

import fqe
from fqe.hamiltonians import gso_hamiltonian
from fqe.hamiltonians import sso_hamiltonian, sparse_hamiltonian
from fqe.hamiltonians import general_hamiltonian, restricted_hamiltonian
from fqe.hamiltonians import hamiltonian_utils
from fqe.fqe_data import FqeData
from fqe.wavefunction import Wavefunction
from fqe import fqe_decorators

from tests.unittest_data import build_hamiltonian

def setup_function():
    """setting up the test
    """
    # === numpy control options ===
    numpy.set_printoptions(floatmode='fixed',
                            precision=6,
                            linewidth=200,
                            suppress=True)
    numpy.random.seed(seed=409)

def test_diagonal_evolution():
    """Test time evolution of diagonal Hamiltonians
    """
    norb = 4
    nalpha = 3
    nbeta = 1
    time = 0.001
    nele = nalpha + nbeta
    lena = int(special.binom(norb, nalpha))
    lenb = int(special.binom(norb, nbeta))
    cidim = lena * lenb

    h1e = numpy.zeros((norb, norb), dtype=numpy.complex128)
    for i in range(norb):
        h1e[i, i] += (i + 1) * 2.0

    h_wrap = tuple([h1e])
    # === Reference Wavefunction ===

    hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

    for i in range(lena):
        for j in range(lenb):
            ket = FqeData(nalpha, nbeta, norb)
            ket.coeff[i, j] = 1.0
            ket_h = ket.apply(h_wrap)
            hci[:, :, i, j] = ket_h.coeff

    hci = numpy.reshape(hci, (cidim, cidim))
    fci_eig, fci_vec = linalg.eigh(hci)
    test_state = numpy.random.rand(cidim).astype(numpy.complex128)

    proj_coeff = fci_vec.conj().T @ test_state
    fci_rep = fci_vec @ proj_coeff
    assert numpy.abs(numpy.linalg.norm(fci_rep - test_state)) < 1.e-8

    phase = numpy.multiply(numpy.exp(-1.j * time * fci_eig), proj_coeff)
    reference = fci_vec @ phase

    test_state = numpy.reshape(test_state, (lena, lenb))
    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): test_state})

    hamil = fqe.get_diagonal_hamiltonian(h1e.diagonal())
    initial_energy = wfn.expectationValue(hamil)

    evol_wfn = fqe.time_evolve(wfn, time, hamil)
    computed = numpy.reshape(evol_wfn._civec[(nele, nalpha - nbeta)].coeff,
                              (cidim))
    assert numpy.abs(linalg.norm(reference - computed)) < 1.e-8

    final_energy = evol_wfn.expectationValue(hamil)
    assert numpy.abs(final_energy - initial_energy) < 1.e-7

    tay_wfn = wfn.apply_generated_unitary(time, 'taylor', hamil)
    computed = numpy.reshape(tay_wfn._civec[(nele, nalpha - nbeta)].coeff,
                              (cidim))
    assert numpy.abs(linalg.norm(reference - computed)) < 1.e-8

    tay_ene = tay_wfn.expectationValue(hamil)
    assert numpy.abs(tay_ene - initial_energy) < 1.e-7

    fqe.time_evolve(wfn, time, hamil, True)
    computed = numpy.reshape(wfn._civec[(nele, nalpha - nbeta)].coeff,
                              (cidim))
    assert numpy.abs(linalg.norm(reference - computed)) < 1.e-8

def test_quadratic_both_conserved():
    """Test time evolution with a Hamiltonian that conserves both spin and number
    """
    norb = 4
    h1e = numpy.zeros((norb, norb), dtype=numpy.complex128)
    for i in range(norb):
        for j in range(norb):
            h1e[i, j] += (i + j) * 0.02
        h1e[i, i] += i * 2.0

    hamil = fqe.get_restricted_hamiltonian((h1e,))

    nalpha = 2
    nbeta = 2
    lena = int(special.binom(norb, nalpha))
    lenb = int(special.binom(norb, nbeta))
    hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)
    for i in range(lena):
        for j in range(lenb):
            ket = FqeData(nalpha, nbeta, norb)
            ket.coeff[i, j] = 1.0
            ket_h = ket.apply((h1e,))
            hci[:, :, i, j] = ket_h.coeff

    fci_eig, fci_vec = linalg.eigh(
        numpy.reshape(hci, (lena * lenb, lena * lenb)))

    wfn = fqe.Wavefunction([[nalpha + nbeta, nalpha - nbeta, norb]])

    wfn._civec[(nalpha + nbeta,
                nalpha - nbeta)].coeff.flat[:] = fci_vec[:, 0]

    initial_energy = wfn.expectationValue(hamil)
    assert numpy.abs(initial_energy - fci_eig[0]) < 1.0e-8

    time = 1.0
    evolved = wfn.time_evolve(time, hamil)

    final_energy = evolved.expectationValue(hamil)
    assert numpy.abs(initial_energy - final_energy) < 1.0e-8

def test_quadratic_number_conserved_spin_broken():
    """Quadratic number conserving and spin broken evolution
    """
    time = 0.001
    norb = 4
    nele = 4
    maxb = min(norb, nele)
    minb = nele - maxb

    h1e = numpy.zeros((2 * norb, 2 * norb), dtype=numpy.complex128)

    mata = numpy.zeros((norb, norb), dtype=numpy.complex128)
    matb = numpy.zeros((norb, norb), dtype=numpy.complex128)
    for i in range(norb):
        for j in range(norb):
            mata[i, j] += (i + j) * 0.02
            matb[i, j] += (i - j) * 0.002
        mata[i, i] += i * 2.0
    h1e[:norb, :norb] = mata
    h1e[norb:, norb:] = mata.conj()
    h1e[:norb, norb:] = matb
    h1e[norb:, :norb] = -matb.conj()

    hamil = gso_hamiltonian.GSOHamiltonian(tuple([h1e]))

    ndim = int(special.binom(2 * norb, nele))
    hci = numpy.zeros((ndim, ndim), dtype=numpy.complex128)

    for i in range(ndim):
        wfn = fqe.get_number_conserving_wavefunction(nele, norb)
        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = wfn.get_coeff((nele, nele - 2 * nbeta))
            size = coeff.size
            if cnt <= i < cnt + size:
                coeff.flat[i - cnt] = 1.0
            cnt += size

        result = fqe.apply(hamil, wfn)

        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = result.get_coeff((nele, nele - 2 * nbeta))
            for j in range(coeff.size):
                hci[cnt + j, i] = coeff.flat[j]
            cnt += coeff.size

    assert numpy.std(hci - hci.T.conj()) < 1.0e-8

    fci_eig, fci_vec = linalg.eigh(hci)
    initial_energy = fci_eig[0]

    wfn = fqe.get_number_conserving_wavefunction(4, norb)

    cnt = 0
    for nbeta in range(minb, maxb + 1):
        coeff = wfn.get_coeff((nele, nele - 2 * nbeta))
        for i in range(coeff.size):
            coeff.flat[i] = fci_vec[i + cnt, 0]
        cnt += coeff.size

    tr_op = fqe.get_time_reversal_operator()
    assert numpy.abs(tr_op.contract(wfn, wfn) - 1.0) < 1.0e-8

    evolved = wfn.time_evolve(time, hamil)

    final_energy = evolved.expectationValue(hamil)

    err = numpy.abs(final_energy - initial_energy)
    assert err < 1.0e-8

    eig, _ = numpy.linalg.eigh(h1e)
    polywfn = wfn.apply_generated_unitary(time,
                                          'chebyshev',
                                          hamil,
                                          accuracy=1.0e-8,
                                          spec_lim=(eig[0], eig[-1]))

    for key in polywfn.sectors():
        diff = linalg.norm(polywfn._civec[key].coeff -
                            evolved._civec[key].coeff)
        assert diff < 1.0e-8

    poly_energy = polywfn.expectationValue(hamil)

    err = numpy.abs(poly_energy - final_energy)
    assert err < 1.0e-8

def test_evolve_diagonal_coulomb():
    """Diagonal Coulomb evolution
    """
    norb = 4
    nalpha = 3
    nbeta = 2
    nele = nalpha + nbeta
    m_s = nalpha - nbeta
    time = 0.001

    lena = int(special.binom(norb, nalpha))
    lenb = int(special.binom(norb, nbeta))
    cidim = lena * lenb

    h1e = numpy.zeros((norb, norb), dtype=numpy.complex128)
    vijkl = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)

    for i in range(norb):
        for j in range(norb):
            vijkl[i, j, i, j] += 4 * (i % norb + 1) * (j % norb + 1) * 0.21

    h_wrap = tuple([h1e, vijkl])

    hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)
    for i in range(lena):
        for j in range(lenb):
            ket = FqeData(nalpha, nbeta, norb)
            ket.coeff[i, j] = 1.0
            ket_h = ket.apply(h_wrap)
            hci[:, :, i, j] = ket_h.coeff

    hci = numpy.reshape(hci, (cidim, cidim))

    fci_eig, fci_vec = linalg.eigh(hci)

    test_state = numpy.random.rand(cidim).astype(numpy.complex128)

    proj_coeff = fci_vec.conj().T @ test_state
    fci_rep = fci_vec @ proj_coeff

    err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
    assert err < 1.e-8

    # Time evolve fci_vec
    phase = numpy.zeros(cidim, dtype=numpy.complex128)

    for i in range(cidim):
        phase[i] = numpy.exp(-1.j * time * fci_eig[i]) * proj_coeff[i]

    reference = fci_vec @ phase

    test_wfn = Wavefunction([[nele, m_s, norb]])
    test_wfn._civec[(nele,
                      m_s)].coeff = numpy.reshape(test_state, (lena, lenb))

    hamil = fqe.get_diagonalcoulomb_hamiltonian(vijkl)
    coul_evol = test_wfn.time_evolve(time, hamil)

    coeff = coul_evol._civec[(nele, m_s)].coeff

    err = numpy.std(reference / reference.flat[0] -
                    coeff.flatten() / coeff.flat[0])
    assert err < 1.e-8

    test_wfn.time_evolve(time, hamil, True)
    coeff = test_wfn._civec[(nele, m_s)].coeff

    err = numpy.std(reference / reference.flat[0] -
                    coeff.flatten() / coeff.flat[0])
    assert err < 1.e-8

def test_diagonal_spin():
    """Evolution of diagonal hamiltonian with different spin components
    """
    norb = 4
    nalpha = 2
    nbeta = 2
    time = 0.001
    nele = nalpha + nbeta
    lena = int(special.binom(norb, nalpha))
    lenb = int(special.binom(norb, nbeta))
    cidim = lena * lenb

    h1e = numpy.zeros((2 * norb, 2 * norb), dtype=numpy.complex128)

    for i in range(2 * norb):
        for j in range(2 * norb):
            h1e[i, j] += (i + j) * 0.02
        h1e[i, i] += i * 2.0
    h1e[:norb, norb:] = 0.0
    h1e[norb:, :norb] = 0.0

    h_wrap = tuple([h1e])
    # === Reference Wavefunction ===

    hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

    for i in range(lena):
        for j in range(lenb):
            ket = FqeData(nalpha, nbeta, norb)
            ket.coeff[i, j] = 1.0
            ket_h = ket.apply(h_wrap)
            hci[:, :, i, j] = ket_h.coeff

    hci = numpy.reshape(hci, (cidim, cidim))
    fci_eig, fci_vec = linalg.eigh(hci)

    test_state = numpy.random.rand(cidim).astype(numpy.complex128)

    proj_coeff = fci_vec.conj().T @ test_state
    fci_rep = fci_vec @ proj_coeff
    err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
    assert err < 1.e-8

    phase = numpy.multiply(numpy.exp(-1.j * time * fci_eig), proj_coeff)

    reference = fci_vec @ phase

    hamil = fqe.get_sso_hamiltonian(h_wrap)

    wfn = Wavefunction([[4, 0, norb]])
    test_state = numpy.reshape(test_state, (lena, lenb))
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): test_state})

    evolwfn = wfn.time_evolve(time, hamil)

    computed = numpy.reshape(evolwfn._civec[(nele, nalpha - nbeta)].coeff,
                              (cidim))

    numpy.testing.assert_almost_equal(reference, computed)

class NBodyTestParams:
    def __init__(self, norb, nalpha, nbeta, time):
        self.norb = norb
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.nele = nalpha + nbeta
        self.time = 0.001

@pytest.fixture
def params():
    return NBodyTestParams(norb=3, nalpha=2, nbeta=2, time=0.001)

def test_1body(params):
    """
    1-body operator time evolution test
    """
    ops = FermionOperator('0^ 1', 2.2 - 0.1j) + FermionOperator(
        '1^ 0', 2.2 + 0.1j)
    sham = fqe.get_sparse_hamiltonian(ops, conserve_spin=False)

    h1e = hamiltonian_utils.nbody_matrix(ops, params.norb)

    wfn = fqe.get_number_conserving_wavefunction(params.nele, params.norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    result = wfn._evolve_individual_nbody(params.time, sham)

    hamil = general_hamiltonian.General(tuple([h1e]))
    nbody_evol = wfn.apply_generated_unitary(params.time,
                                              'taylor',
                                              hamil,
                                              accuracy=1.0e-8)

    result.ax_plus_y(-1.0, nbody_evol)
    assert result.norm() < 1.e-8

    wfn = wfn._evolve_individual_nbody(params.time, sham, True)
    wfn.ax_plus_y(-1.0, nbody_evol)
    assert wfn.norm() < 1.e-8

def test_2body(params):
    """
    2-body operator time evolution test
    """
    ops = FermionOperator('1^ 3^ 1 2', 2.0 - 2.j) + FermionOperator(
        '2^ 1^ 3 1', 2.0 + 2.j)

    sham = fqe.get_sparse_hamiltonian(ops)

    h2e = hamiltonian_utils.nbody_matrix(ops, params.norb)
    h2e = hamiltonian_utils.antisymm_two_body(h2e)

    wfn = fqe.get_number_conserving_wavefunction(params.nele, params.norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    result = wfn._evolve_individual_nbody(params.time, sham)

    h1e = numpy.zeros((2 * params.norb, 2 * params.norb),
                       dtype=numpy.complex128)
    hamil = general_hamiltonian.General(tuple([h1e, h2e]))
    nbody_evol = wfn.apply_generated_unitary(params.time,
                                              'taylor',
                                              hamil,
                                              accuracy=1.0e-8)

    result.ax_plus_y(-1.0, nbody_evol)
    assert result.norm() < 1.e-8

def test_3body(params):
    """
    3-body operator time evolution test
    """
    ops = FermionOperator('2^ 1^ 0^ 2 0 1', 1.0 - 1.j) + FermionOperator(
          '1^ 0^ 2^ 0 1 2', 1.0 + 1.j)
    sham = fqe.get_sparse_hamiltonian(ops)

    h3e = hamiltonian_utils.nbody_matrix(ops, params.norb)
    h3e = hamiltonian_utils.antisymm_three_body(h3e)

    wfn = fqe.get_number_conserving_wavefunction(params.nele, params.norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    result = wfn._evolve_individual_nbody(params.time, sham)

    h1e = numpy.zeros((2 * params.norb, 2 * params.norb),
                       dtype=numpy.complex128)
    h2e = numpy.zeros((2 * params.norb, 2 * params.norb,
                       2 * params.norb, 2 * params.norb),
                       dtype=numpy.complex128)
    hamil = general_hamiltonian.General(tuple([h1e, h2e, h3e]))
    nbody_evol = wfn.apply_generated_unitary(params.time,
                                              'taylor',
                                              hamil,
                                              expansion=20)
    result.ax_plus_y(-1.0, nbody_evol)
    assert result.norm() < 1.e-8

def test_4body(params):
    """
    4-body operator time evolution test
    """
    ops = FermionOperator('0^ 1^ 3^ 4^ 2 1 3 4', 1.0 + 0.j) + FermionOperator(
      '4^ 3^ 1^ 2^ 4 3 1 0', 1.0 + 0.j)
    sham = fqe.get_sparse_hamiltonian(ops)

    h4e = hamiltonian_utils.nbody_matrix(ops, params.norb)
    h4e = hamiltonian_utils.antisymm_four_body(h4e)

    wfn = fqe.get_number_conserving_wavefunction(params.nele, params.norb)
    wfn.set_wfn(strategy='ones')
    wfn.normalize()

    result = wfn._evolve_individual_nbody(params.time, sham)

    h1e = numpy.zeros((2 * params.norb, 2 * params.norb),
                       dtype=numpy.complex128)
    h2e = numpy.zeros((2 * params.norb, 2 * params.norb,
                       2 * params.norb, 2 * params.norb),
                       dtype=numpy.complex128)
    h3e = numpy.zeros((2 * params.norb, 2 * params.norb,
                       2 * params.norb, 2 * params.norb,
                       2 * params.norb, 2 * params.norb),
                       dtype=numpy.complex128)

    hamil = general_hamiltonian.General(tuple([h1e, h2e, h3e, h4e]))
    nbody_evol = wfn.apply_generated_unitary(params.time,
                                              'taylor',
                                              hamil,
                                              expansion=20)
    result.print_wfn()
    nbody_evol.print_wfn()
    result.ax_plus_y(-1.0, nbody_evol)
    assert result.norm() < 1.e-8

def test_restricted():
    """Evolution of a Hamiltonian with identical alpha and beta interactions
    """
    norb = 3
    nalpha = 2
    nbeta = 2
    nele = nalpha + nbeta
    time = 0.001
    lena = int(special.binom(norb, nalpha))
    lenb = int(special.binom(norb, nbeta))

    cidim = lena * lenb

    h1e, h2e, h3e, h4e = build_hamiltonian.build_restricted(norb, full=True)

    h_wrap = tuple([
        h1e[:norb, :norb], h2e[:norb, :norb, :norb, :norb],
        h3e[:norb, :norb, :norb, :norb, :norb, :norb],
        h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    ])

    hamil = restricted_hamiltonian.RestrictedHamiltonian(h_wrap)

    hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

    for i in range(lena):
        for j in range(lenb):
            ket = FqeData(nalpha, nbeta, norb)
            ket.coeff[i, j] = 1.0
            ket_h = ket.apply((h1e, h2e, h3e, h4e))
            hci[:, :, i, j] = ket_h.coeff[:, :]

    hci = numpy.reshape(hci, (cidim, cidim))
    fci_eig, fci_vec = linalg.eigh(hci)

    test = numpy.reshape(fci_vec[:, 0], (lena, lenb))

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn._civec[(nele, nalpha - nbeta)].coeff = test

    energy = fqe.expectationValue(wfn, hamil)

    assert round(abs(energy-fci_eig[0]), 7) == 0

    test_state = numpy.random.rand(cidim).astype(numpy.complex128)

    proj_coeff = fci_vec.conj().T @ test_state
    fci_rep = fci_vec @ proj_coeff
    err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
    assert err < 1.e-8

    phase = numpy.multiply(numpy.exp(-1.j * time * fci_eig), proj_coeff)

    reference = fci_vec @ phase

    test_state = numpy.reshape(test_state, (lena, lenb))

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): test_state})

    evol_wfn = wfn.time_evolve(time, hamil)

    computed = numpy.reshape(evol_wfn._civec[(nele, nalpha - nbeta)].coeff,
                              (cidim))

    err = numpy.abs(linalg.norm(reference - computed))
    assert err < 1.e-8

def test_gso_four_body():
    """Evolution of four body hamiltonian with different alpha and beta
    spin interactions
    """
    norb = 2
    nele = 2
    time = 0.001
    maxb = min(norb, nele)
    minb = nele - maxb
    ndim = int(special.binom(norb * 2, nele))

    h1e, h2e, h3e, h4e = build_hamiltonian.build_gso(norb)

    h_wrap = tuple([h1e, h2e, h3e, h4e])
    hamil = fqe.get_gso_hamiltonian(h_wrap)

    hci = numpy.zeros((ndim, ndim), dtype=numpy.complex128)

    def unpack(inp: numpy.ndarray) -> 'Wavefunction':
        out = fqe.get_number_conserving_wavefunction(nele, norb)
        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = out.get_coeff((nele, nele - 2 * nbeta))
            for i in range(coeff.size):
                coeff.flat[i] = inp[i + cnt]
            cnt += coeff.size
        return out

    def pack(inp: 'Wavefunction') -> numpy.ndarray:
        out = numpy.zeros((ndim,), dtype=numpy.complex128)
        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = inp.get_coeff((nele, nele - 2 * nbeta))
            for j in range(coeff.size):
                out[cnt + j] = coeff.flat[j]
            cnt += coeff.size

        return out

    for i in range(ndim):
        wfn = fqe.get_number_conserving_wavefunction(nele, norb)
        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = wfn.get_coeff((nele, nele - 2 * nbeta))
            size = coeff.size
            if cnt <= i < cnt + size:
                coeff.flat[i - cnt] = 1.0
            cnt += size

        test_axpy = 1.0
        result = wfn._apply_array(h_wrap, test_axpy)
        result.ax_plus_y(-test_axpy, wfn)
        hci[:, i] = pack(result)

    fci_eig, fci_vec = linalg.eigh(hci)

    state = 0

    wfn = unpack(fci_vec[:, state])

    energy = wfn.expectationValue(hamil)

    assert round(abs(energy-fci_eig[0]), 7) == 0

    rdms = wfn._compute_rdm(4)
    rdms1 = wfn._compute_rdm(1)
    rdms1exp = wfn.expectationValue('i^ j')
    rdms1val = wfn.expectationValue('0^ 0')
    rdms2 = wfn._compute_rdm(2)
    rdms3 = wfn._compute_rdm(3)
    energy2 = numpy.inner(rdms[0].flatten(), h1e.flatten()) \
              + numpy.inner(rdms[1].flatten(), h2e.flatten()) \
              + numpy.inner(rdms[2].flatten(), h3e.flatten()) \
              + numpy.inner(rdms[3].flatten(), h4e.flatten())

    assert round(abs(energy-energy2), 7) == 0
    assert numpy.allclose(rdms[0], rdms1[0])
    assert numpy.allclose(rdms[0], rdms1exp)
    assert numpy.abs(rdms1val - rdms[0].flat[0]) < 1.0e-8
    assert numpy.allclose(rdms[0], rdms2[0])
    assert numpy.allclose(rdms[1], rdms2[1])
    assert numpy.allclose(rdms[2], rdms3[2])

    test_state = numpy.random.rand(ndim).astype(numpy.complex128)

    proj_coeff = fci_vec.conj().T @ test_state
    fci_rep = fci_vec @ proj_coeff
    err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
    assert err < 1.e-8

    phase = numpy.multiply(numpy.exp(-1.j * time * fci_eig), proj_coeff)
    reference = fci_vec @ phase
    ref_wfn = unpack(reference)

    wfn = unpack(test_state)
    evol_wfn = wfn.time_evolve(time, hamil)

    err = (evol_wfn - ref_wfn).norm()
    assert err < 1.e-8

    rdms = wfn._compute_rdm(4, evol_wfn)
    rdms1 = wfn._compute_rdm(1, evol_wfn)
    rdms1exp = wfn.expectationValue('i^ j', evol_wfn)
    rdms1val = wfn.expectationValue('0^ 0', evol_wfn)
    rdms1val2 = wfn.rdm('0^ 0', evol_wfn)
    rdms2 = wfn._compute_rdm(2, evol_wfn)
    rdms3 = wfn._compute_rdm(3, evol_wfn)
    assert numpy.allclose(rdms[0], rdms1[0])
    assert numpy.allclose(rdms[0], rdms1exp)
    assert numpy.abs(rdms1val - rdms[0].flat[0]) < 1.0e-8
    assert numpy.abs(rdms1val2 - rdms[0].flat[0]) < 1.0e-8
    assert numpy.allclose(rdms[0], rdms2[0])
    assert numpy.allclose(rdms[1], rdms2[1])
    assert numpy.allclose(rdms[2], rdms3[2])

def test_sso_four_body():
    """Evolution of four body hamiltonian with different alpha and beta
    spin interactions
    """
    norb = 3
    nalpha = 2
    nbeta = 2
    nele = nalpha + nbeta
    time = 0.001
    lena = int(special.binom(norb, nalpha))
    lenb = int(special.binom(norb, nbeta))

    cidim = lena * lenb
    h1e, h2e, h3e, h4e = build_hamiltonian.build_sso(norb)

    h_wrap = tuple([h1e, h2e, h3e, h4e])
    hamil = sso_hamiltonian.SSOHamiltonian(h_wrap)

    hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

    for i in range(lena):
        for j in range(lenb):
            ket = FqeData(nalpha, nbeta, norb)
            ket.coeff[i, j] = 1.0
            ket_h = ket.apply(h_wrap)
            hci[:, :, i, j] = ket_h.coeff[:, :]

    hci = numpy.reshape(hci, (cidim, cidim))
    fci_eig, fci_vec = linalg.eigh(hci)

    test = numpy.reshape(fci_vec[:, 0], (lena, lenb))

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn._civec[(nele, nalpha - nbeta)].coeff = test

    energy = wfn.expectationValue(hamil)

    assert round(abs(energy-fci_eig[0]), 7) == 0

    test_state = numpy.random.rand(cidim).astype(numpy.complex128)

    proj_coeff = fci_vec.conj().T @ test_state
    fci_rep = fci_vec @ proj_coeff
    err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
    assert err < 1.e-8

    phase = numpy.multiply(numpy.exp(-1.j * time * fci_eig), proj_coeff)
    reference = fci_vec @ phase

    test_state = numpy.reshape(test_state, (lena, lenb))

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): test_state})

    evol_wfn = wfn.time_evolve(time, hamil)

    computed = numpy.reshape(evol_wfn._civec[(nele, nalpha - nbeta)].coeff,
                              (cidim))

    err = numpy.abs(linalg.norm(reference - computed))
    assert err < 1.e-8

def test_chebyshev():
    """Evolution using chebyshev polynomial expansion
    """
    norb = 2
    nalpha = 1
    nbeta = 1
    nele = nalpha + nbeta
    time = 0.01

    h1e = numpy.zeros((norb * 2, norb * 2), dtype=numpy.complex128)

    for i in range(2 * norb):
        for j in range(2 * norb):
            h1e[i, j] += (i + j) * 0.02
        h1e[i, i] += i * 2.0

    eig, _ = linalg.eigh(h1e)
    hamil = fqe.get_general_hamiltonian((h1e,))

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='random')

    chebyshev = wfn.apply_generated_unitary(time,
                                            'chebyshev',
                                            hamil,
                                            spec_lim=(eig[0], eig[-1]))

    taylor = wfn.apply_generated_unitary(time, 'taylor', hamil)

    err = (chebyshev - taylor).norm()
    assert err < 1.e-8

def test_few_nbody_evolution():
    """Check the evolution for very sparse hamiltonians
    """
    norb = 4
    nalpha = 3
    nbeta = 0
    nele = nalpha + nbeta
    time = 0.01

    ops = FermionOperator('1^ 2^ 0 5', 2.0 - 2.j)
    ops += FermionOperator('5^ 0^ 2 1', 2.0 + 2.j)
    ops2 = FermionOperator('3^ 2^ 0 5', 2.0 - 2.j)
    ops2 += FermionOperator('5^ 0^ 2 3', 2.0 + 2.j)
    sham = sparse_hamiltonian.SparseHamiltonian(ops + ops2)

    wfn = fqe.get_number_conserving_wavefunction(nele, norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    result = wfn.apply_generated_unitary(time, 'taylor', sham)

    h2e = hamiltonian_utils.nbody_matrix(ops, norb)
    h2e += hamiltonian_utils.nbody_matrix(ops2, norb)
    h1e = numpy.zeros((norb * 2, norb * 2), dtype=h2e.dtype)
    hamiltest = general_hamiltonian.General(tuple([h1e, h2e]))
    nbody_evol = wfn.apply_generated_unitary(time, 'taylor', hamiltest)

    result.ax_plus_y(-1.0, nbody_evol)
    assert result.norm() < 1.e-8

def test_time_evolve_broken_symm():
    """Compare spin and number conserving
    """
    norb = 4
    time = 0.05
    wfn_spin = fqe.get_spin_conserving_wavefunction(-2, norb)
    wfn_number = fqe.get_number_conserving_wavefunction(2, norb)

    work = build_hamiltonian.number_nonconserving_fop(2, norb)
    h_noncon = fqe.get_hamiltonian_from_openfermion(work,
                                                    norb=norb,
                                                    conserve_number=False)
    h_con = copy.deepcopy(h_noncon)
    h_con._conserve_number = True

    test = numpy.ones((4, 4))
    wfn_spin.set_wfn(strategy='from_data', raw_data={(4, -2): test})
    wfn_spin.normalize()
    wfn_number.set_wfn(strategy='from_data',
                        raw_data={(2, 0): numpy.flip(test, 1)})
    wfn_number.normalize()
    spin_evolved = wfn_spin.time_evolve(time, h_noncon)
    number_evolved = wfn_number.time_evolve(time, h_con)
    ref = spin_evolved._copy_beta_inversion()
    hamil = general_hamiltonian.General(h_con.tensors(), e_0=h_noncon.e_0())
    unitary_evolved = wfn_number.apply_generated_unitary(
        time, 'taylor', hamil)
    for key in number_evolved.sectors():
        assert numpy.allclose(number_evolved._civec[key].coeff,
                            ref._civec[key].coeff)
        assert numpy.allclose(number_evolved._civec[key].coeff,
                            unitary_evolved._civec[key].coeff)

    assert round(abs(spin_evolved.rdm('2^ 1^')-(-0.004985346234592781 - 0.0049853462345928745j)), 7) == 0

def test_broken_number_nbody():
    """Compare spin and number conserving
    """
    norb = 4
    time = 0.001
    wfn_spin = fqe.get_spin_conserving_wavefunction(3, norb)

    work = FermionOperator('0^ 1^ 2 3 4^ 6', 3.0 - 1.3j)
    work += hermitian_conjugated(work)
    h_noncon = fqe_decorators.build_hamiltonian(work,
                                                norb=norb,
                                                conserve_number=False)

    gen = fqe_decorators.normal_ordered(
        fqe_decorators.transform_to_spin_broken(work))
    matrix = fqe_decorators.fermionops_tomatrix(gen, norb)
    wrap = tuple([
        numpy.zeros((2 * norb, 2 * norb), dtype=numpy.complex128),
        numpy.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb),
                    dtype=numpy.complex128), matrix
    ])

    hamil = general_hamiltonian.General(wrap)
    hamil._conserve_number = False

    wfn_spin.set_wfn(strategy='random')
    nbody_evolved = wfn_spin.time_evolve(time, h_noncon)
    unitary_evolved = wfn_spin.apply_generated_unitary(
        time, 'taylor', hamil)
    for key in nbody_evolved.sectors():
        assert numpy.allclose(nbody_evolved._civec[key].coeff,
                            unitary_evolved._civec[key].coeff)
