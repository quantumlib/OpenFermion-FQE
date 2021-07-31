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
"""Wavefunction class unit tests
"""
# pylint: disable=protected-access

import os
import sys
import copy
import numpy
import pytest

from io import StringIO

from scipy.special import binom

from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated

import fqe
from fqe.wavefunction import Wavefunction
from fqe import get_spin_conserving_wavefunction
from fqe import get_number_conserving_wavefunction
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import sso_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import diagonal_hamiltonian
from fqe.hamiltonians import diagonal_coulomb
from fqe.hamiltonians import hamiltonian_utils
from fqe import get_restricted_hamiltonian
from fqe import NumberOperator
from fqe.util import vdot

from tests.unittest_data import build_wfn, build_hamiltonian
from tests.unittest_data.build_lih_data import build_lih_data

from tests.unittest_data.wavefunction.generate_data import \
    all_cases, spin_broken_cases
from tests.unittest_data.wavefunction_data import loader
from tests.comparisons import FqeDataSet_isclose, Wavefunction_isclose, \
    FqeData_isclose


def test_init_exceptions():
    """Check that wavefunction throws the proper errors when an incorrect
    initialization is passed.
    """
    with pytest.raises(TypeError):
        Wavefunction(broken=['spin', 'number'])
    with pytest.raises(ValueError):
        Wavefunction(param=[[0, 0, 2], [0, 0, 4]])
    with pytest.raises(ValueError):
        Wavefunction(param=[[2, 1, 4]])
    with pytest.raises(ValueError):
        Wavefunction(param=[[2, 4, 4]])
    with pytest.raises(ValueError):
        Wavefunction(param=[[-1, 0, 4]])


def test_general_exceptions():
    """Test general method exceptions
    """
    test1 = Wavefunction(param=[[2, 0, 4]])
    test2 = Wavefunction(param=[[4, -4, 8]])
    test1.set_wfn(strategy='ones')
    test2.set_wfn(strategy='ones')
    with pytest.raises(ValueError):
        test1.ax_plus_y(1.0, test2)
    with pytest.raises(ValueError):
        test1.__add__(test2)
    with pytest.raises(ValueError):
        test1.__sub__(test2)
    with pytest.raises(ValueError):
        test1.set_wfn(strategy='from_data')


def test_general_functions():
    """Test general wavefunction members
    """
    test = Wavefunction(param=[[2, 0, 4]])
    test.set_wfn(strategy='ones')
    assert 1. + 0.j == test[(4, 8)]
    test[(4, 8)] = 3.14 + 0.00159j
    assert 3.14 + 0.00159j == test[(4, 8)]
    assert 3.14 + 0.00159j == test.max_element()
    assert test.conserve_spin()
    test1 = Wavefunction(param=[[2, 0, 4]])
    test2 = Wavefunction(param=[[2, 0, 4]])
    test1.set_wfn(strategy='ones')
    test2.set_wfn(strategy='ones')
    work = test1 + test2
    ref = 2.0 * numpy.ones((4, 4), dtype=numpy.complex128)
    assert numpy.allclose(ref, work._civec[(2, 0)].coeff)
    work = test1 - test2
    ref = numpy.zeros((4, 4), dtype=numpy.complex128)
    assert numpy.allclose(ref, work._civec[(2, 0)].coeff)


def test_empty_copy():
    """Test empty_copy function
    """
    test = Wavefunction(param=[[2, 0, 4]])
    test.set_wfn(strategy='ones')
    test1 = test.empty_copy()
    assert test1._norb == test._norb
    assert len(test1._civec) == len(test._civec)
    assert (2, 0) in test1._civec.keys()
    assert not numpy.any(test1._civec[(2, 0)].coeff)


def test_apply_number():
    norb = 4
    diag = numpy.random.rand(norb * 2)
    test = numpy.random.rand(norb, norb).astype(numpy.complex128)
    diag2 = copy.deepcopy(diag)
    e_0 = 0
    for i in range(norb):
        e_0 += diag[i + norb]
        diag2[i + norb] = -diag[i + norb]
    hamil = diagonal_hamiltonian.Diagonal(diag2, e_0=e_0)
    hamil._conserve_number = False
    wfn = Wavefunction([[4, 2, norb]], broken=['number'])
    wfn.set_wfn(strategy='from_data', raw_data={(4, 2): test})
    out1 = wfn.apply(hamil)

    hamil = diagonal_hamiltonian.Diagonal(diag)
    wfn = Wavefunction([[4, 2, norb]])
    wfn.set_wfn(strategy='from_data', raw_data={(4, 2): test})
    out2 = wfn.apply(hamil)

    assert numpy.allclose(out1._civec[(4, 2)].coeff,
                          out2._civec[(4, 2)].coeff)


def test_apply_type_error():
    data = numpy.zeros((2, 2), dtype=numpy.complex128)
    wfn = Wavefunction([[2, 0, 2]], broken=['spin'])
    hamil = general_hamiltonian.General((data,))
    hamil._conserve_number = False
    with pytest.raises(TypeError):
        wfn.apply(hamil)
    with pytest.raises(TypeError):
        wfn.time_evolve(0.1, hamil)

    wfn = Wavefunction([[2, 0, 2]], broken=['number'])
    hamil = general_hamiltonian.General((data,))
    with pytest.raises(TypeError):
        wfn.apply(hamil)
    with pytest.raises(TypeError):
        wfn.time_evolve(0.1, hamil)

    wfn = Wavefunction([[2, 0, 2]])
    data2 = numpy.zeros((2, 2, 2, 2), dtype=numpy.complex128)
    hamil = get_restricted_hamiltonian((data, data2))
    with pytest.raises(ValueError):
        wfn.time_evolve(0.1, hamil, True)


def test_apply_individual_nbody_error():
    fop = FermionOperator('1^ 0')
    fop += FermionOperator('2^ 0')
    fop += FermionOperator('2^ 1')
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    wfn = Wavefunction([[2, 0, 2]], broken=['spin'])
    with pytest.raises(ValueError):
        wfn._apply_individual_nbody(hamil)
    with pytest.raises(ValueError):
        wfn._evolve_individual_nbody(0.1, hamil)

    fop = FermionOperator('1^ 0')
    fop += FermionOperator('2^ 0')
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    with pytest.raises(ValueError):
        wfn._evolve_individual_nbody(0.1, hamil)

    fop = FermionOperator('1^ 0', 1.0)
    fop += FermionOperator('0^ 1', 0.9)
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    with pytest.raises(ValueError):
        wfn._evolve_individual_nbody(0.1, hamil)

    fop = FermionOperator('1^ 0^')
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    with pytest.raises(ValueError):
        wfn._apply_individual_nbody(hamil)
    with pytest.raises(ValueError):
        wfn._evolve_individual_nbody(0.1, hamil)

    with pytest.raises(TypeError):
        wfn._evolve_individual_nbody(0.1, 1)


def test_apply_diagonal():
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')

    data = numpy.random.rand(2)
    hamil = diagonal_hamiltonian.Diagonal(data)
    out1 = wfn._apply_diagonal(hamil)

    fac = 0.5
    hamil = diagonal_hamiltonian.Diagonal(data, e_0=fac)
    out2 = wfn._apply_diagonal(hamil)
    out2.ax_plus_y(-fac, wfn)
    assert (out1 - out2).norm() < 1.0e-8


def test_apply_diagonal_coulomb():
    norb = 4
    wfn = Wavefunction([[norb, 0, norb]])
    wfn.set_wfn(strategy='random')

    rng = numpy.random.default_rng(680429)
    vij = 8*rng.uniform(0, 1, size=(norb, norb))
    h2 = numpy.zeros((norb, norb, norb, norb))
    h1 = numpy.zeros((norb, norb))
    for i in range(norb):
        for j in range(norb):
            h2[i, j, i, j] = -vij[i, j]

    e0 = rng.uniform(-100, 0)

    hamil1 = diagonal_coulomb.DiagonalCoulomb(h2, e_0=e0)
    hamil2 = get_restricted_hamiltonian((h1, h2), e_0=e0)
    out1 = wfn.apply(hamil1)
    out2 = wfn.apply(hamil2)

    assert (out1 - out2).norm() < 1.0e-8


def test_apply_nbody():
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')

    fac = 3.14
    fop = FermionOperator('1^ 1', fac)
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    out1 = wfn._apply_few_nbody(hamil)

    fop = FermionOperator('1 1^', fac)
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    out2 = wfn._apply_few_nbody(hamil)
    out2.scale(-1.0)
    out2.ax_plus_y(fac, wfn)
    assert (out1 - out2).norm() < 1.0e-8

    # Check if apply does the same as wfn._apply_few_nbody
    fop = FermionOperator('1^ 1', fac)
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    assert (out1 - wfn.apply(hamil)).norm() < 1e-13

def test_apply_few_body():
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')

    fop1 = FermionOperator('1^ 1', 1.0)
    hamil1 = sparse_hamiltonian.SparseHamiltonian(fop1)
    fop2 = FermionOperator('0^ 0', 0.3)
    hamil2 = sparse_hamiltonian.SparseHamiltonian(fop2)
    hamil12 = sparse_hamiltonian.SparseHamiltonian(fop1 + fop2)

    ref = wfn.apply(hamil1) + wfn.apply(hamil2)
    out = wfn._apply_few_nbody(hamil12)
    assert (ref - out).norm() < 1e-13

def test_apply_empty_nbody():
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')

    fop = FermionOperator()
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    out1 = wfn._apply_few_nbody(hamil)
    assert (wfn - out1).norm() < 1e-13

def test_nbody_evolve():
    """Check that 'individual_nbody' is consistent with a general 1-electron op
    """
    norb = 4
    nele = 4
    time = 0.1
    ops = FermionOperator('0^ 1', 1 - 0.2j) + FermionOperator(
        '1^ 0', 1 + 0.2j)
    sham = fqe.get_sparse_hamiltonian(ops, conserve_spin=False)

    h1e = hamiltonian_utils.nbody_matrix(ops, norb)

    wfn = fqe.get_number_conserving_wavefunction(nele, norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    out = wfn._evolve_individual_nbody(time, sham)

    hamil = general_hamiltonian.General(tuple([h1e]))
    ref = wfn.apply_generated_unitary(time,
                                      'taylor',
                                      hamil,
                                      accuracy=1.0e-9)

    assert (ref - out).norm() < 1.e-8

def test_quadratic_evolve():
    norb = 4
    nalpha = 2
    nbeta = 2
    time = 0.1
    h1e = numpy.zeros((2*norb, 2*norb))
    h2e = numpy.zeros((2*norb, 2*norb, 2*norb, 2*norb))

    rng = numpy.random.default_rng(826283)
    h1a = rng.uniform(-1, 1, size=(norb, norb))
    h1a += h1a.transpose()
    h1b = rng.uniform(-1, 1, size=(norb, norb))
    h1b += h1b.transpose()
    h1e[:norb, :norb] = h1a
    h1e[norb:, norb:] = h1b

    hamil1 = sso_hamiltonian.SSOHamiltonian((h1e,))
    hamil2 = sso_hamiltonian.SSOHamiltonian((h1e, h2e))

    wfn = Wavefunction([[nalpha + nbeta, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    ref = wfn.time_evolve(time, hamil2)
    out = wfn.time_evolve(time, hamil1)

    assert (ref - out).norm() < 1.e-8

    wfn.time_evolve(time, hamil1, inplace=True)
    assert (ref - wfn).norm() < 1.e-8

def test_rdm():
    """Check that the rdms will properly return the energy
    """
    wfn = Wavefunction(param=[[4, 0, 3]])
    work, energy = build_wfn.restricted_wfn_energy()
    wfn.set_wfn(strategy='from_data', raw_data={(4, 0): work})
    rdm1 = wfn.rdm('i^ j')
    rdm2 = wfn.rdm('i^ j^ k l')
    rdm3 = wfn.rdm('i^ j^ k^ l m n')
    rdm4 = wfn.rdm('i^ j^ k^ l^ m n o p')
    h1e, h2e, h3e, h4e = build_hamiltonian.build_restricted(3, full=False)
    expval = 0. + 0.j
    axes = [0, 1]
    expval += numpy.tensordot(h1e, rdm1, axes=(axes, axes))
    axes = [0, 1, 2, 3]
    expval += numpy.tensordot(h2e, rdm2, axes=(axes, axes))
    axes = [0, 1, 2, 3, 4, 5]
    expval += numpy.tensordot(h3e, rdm3, axes=(axes, axes))
    axes = [0, 1, 2, 3, 4, 5, 6, 7]
    expval += numpy.tensordot(h4e, rdm4, axes=(axes, axes))
    assert round(abs(expval-energy), 13) == 0


def test_expectation_value_type_error():
    wfn = Wavefunction([[4, 0, 4]])
    with pytest.raises(TypeError):
        wfn.expectationValue(1)


def test_save_read():
    """Check that the wavefunction can be properly archived and
    retrieved
    """
    numpy.random.seed(seed=409)
    wfn = get_number_conserving_wavefunction(3, 3)
    wfn.set_wfn(strategy='random')
    wfn.save('test_save_read')
    read_wfn = Wavefunction()
    read_wfn.read('test_save_read')
    for key in read_wfn.sectors():
        assert FqeData_isclose(read_wfn._civec[key], wfn._civec[key])
    assert read_wfn._symmetry_map == wfn._symmetry_map
    assert read_wfn._conserved == wfn._conserved
    assert read_wfn._conserve_spin == wfn._conserve_spin
    assert read_wfn._conserve_number == wfn._conserve_number
    assert read_wfn._norb == wfn._norb

    os.remove('test_save_read')

    wfn = get_spin_conserving_wavefunction(2, 6)
    wfn.set_wfn(strategy='random')
    wfn.save('test_save_read')
    read_wfn = Wavefunction()
    read_wfn.read('test_save_read')
    for key in read_wfn.sectors():
        assert numpy.allclose(read_wfn._civec[key].coeff,
                              wfn._civec[key].coeff)
    assert read_wfn._symmetry_map == wfn._symmetry_map
    assert read_wfn._conserved == wfn._conserved
    assert read_wfn._conserve_spin == wfn._conserve_spin
    assert read_wfn._conserve_number == wfn._conserve_number
    assert read_wfn._norb == wfn._norb

    os.remove('test_save_read')


def test_wavefunction_print():
    """Check printing routine for the wavefunction.
    """
    numpy.random.seed(seed=409)
    wfn = get_number_conserving_wavefunction(3, 3)
    sector_alpha_dim, sector_beta_dim = wfn.sector((3, -3)).coeff.shape
    coeffs = numpy.arange(1,
                          sector_alpha_dim * sector_beta_dim + 1).reshape(
                              (sector_alpha_dim, sector_beta_dim))
    wfn.sector((3, -3)).coeff = coeffs

    sector_alpha_dim, sector_beta_dim = wfn.sector((3, -1)).coeff.shape
    coeffs = numpy.arange(1,
                          sector_alpha_dim * sector_beta_dim + 1).reshape(
                              (sector_alpha_dim, sector_beta_dim))
    wfn.sector((3, -1)).coeff = coeffs

    sector_alpha_dim, sector_beta_dim = wfn.sector((3, 1)).coeff.shape
    coeffs = numpy.arange(1,
                          sector_alpha_dim * sector_beta_dim + 1).reshape(
                              (sector_alpha_dim, sector_beta_dim))
    wfn.sector((3, 1)).coeff = coeffs

    sector_alpha_dim, sector_beta_dim = wfn.sector((3, 3)).coeff.shape
    coeffs = numpy.arange(1,
                          sector_alpha_dim * sector_beta_dim + 1).reshape(
                              (sector_alpha_dim, sector_beta_dim))
    wfn.sector((3, 3)).coeff = coeffs

    ref_string = 'Sector N = 3 : S_z = -3\n' + \
        "a'000'b'111' 1\n" + \
        "Sector N = 3 : S_z = -1\n" + \
        "a'001'b'011' 1\n" + \
        "a'001'b'101' 2\n" + \
        "a'001'b'110' 3\n" + \
        "a'010'b'011' 4\n" + \
        "a'010'b'101' 5\n" + \
        "a'010'b'110' 6\n" + \
        "a'100'b'011' 7\n" + \
        "a'100'b'101' 8\n" + \
        "a'100'b'110' 9\n" + \
        "Sector N = 3 : S_z = 1\n" + \
        "a'011'b'001' 1\n" + \
        "a'011'b'010' 2\n" + \
        "a'011'b'100' 3\n" + \
        "a'101'b'001' 4\n" + \
        "a'101'b'010' 5\n" + \
        "a'101'b'100' 6\n" + \
        "a'110'b'001' 7\n" + \
        "a'110'b'010' 8\n" + \
        "a'110'b'100' 9\n" + \
        "Sector N = 3 : S_z = 3\n" + \
        "a'111'b'000' 1\n"
    save_stdout = sys.stdout
    sys.stdout = chkprint = StringIO()
    wfn.print_wfn()
    sys.stdout = save_stdout
    outstring = chkprint.getvalue()
    assert outstring == ref_string

    wfn.print_wfn(fmt='occ')
    ref_string = "Sector N = 3 : S_z = -3\n" + \
        "bbb 1\n" + \
        "Sector N = 3 : S_z = -1\n" + \
        ".b2 1\n" + \
        "b.2 2\n" + \
        "bba 3\n" + \
        ".2b 4\n" + \
        "bab 5\n" + \
        "b2. 6\n" + \
        "abb 7\n" + \
        "2.b 8\n" + \
        "2b. 9\n" + \
        "Sector N = 3 : S_z = 1\n" + \
        ".a2 1\n" + \
        ".2a 2\n" + \
        "baa 3\n" + \
        "a.2 4\n" + \
        "aba 5\n" + \
        "2.a 6\n" + \
        "aab 7\n" + \
        "a2. 8\n" + \
        "2a. 9\n" + \
        "Sector N = 3 : S_z = 3\n" + \
        "aaa 1\n"
    save_stdout = sys.stdout
    sys.stdout = chkprint = StringIO()
    wfn.print_wfn(fmt='occ')
    sys.stdout = save_stdout
    outstring = chkprint.getvalue()
    assert outstring == ref_string


def test_hartree_fock_init():
    h1e, h2e, _ = build_lih_data('energy')
    elec_hamil = get_restricted_hamiltonian((h1e, h2e))
    norb = 6
    nalpha = 2
    nbeta = 2
    wfn = Wavefunction([[nalpha + nbeta, nalpha - nbeta, norb]])
    wfn.print_wfn()
    wfn.set_wfn(strategy='hartree-fock')
    wfn.print_wfn()
    assert round(abs(wfn.expectationValue(elec_hamil) -
                     (-8.857341498221992)), 13) == 0
    hf_wf = numpy.zeros((int(binom(norb, 2)), int(binom(norb, 2))))
    hf_wf[0, 0] = 1.
    assert numpy.allclose(wfn.get_coeff((4, 0)), hf_wf)

    wfn = Wavefunction([[nalpha + nbeta, nalpha - nbeta, norb],
                        [nalpha + nbeta, 2, norb]])
    with pytest.raises(ValueError):
        wfn.set_wfn(strategy='hartree-fock')


def test_set_wfn_random_with_multiple_sectors_is_normalized():
    wfn = Wavefunction([[2, 0, 4], [2, -2, 4]], broken=None)
    wfn.set_wfn(strategy="random")
    assert round(abs(wfn.norm()-1.0), 7) == 0


def test_iadd():
    """Checks __iadd__"""
    test1 = Wavefunction(param=[[4, 0, 4]])
    test2 = Wavefunction(param=[[4, 0, 4]])
    test1.set_wfn(strategy='ones')
    test2.set_wfn(strategy='ones')

    test1 += test2
    test2.scale(2.0)

    assert (test1 - test2).norm() < 1e-13

    test3 = Wavefunction(param=[[4, 2, 4]])
    with pytest.raises(ValueError):
        test1 += test3


@pytest.mark.parametrize("param", all_cases)
def test_norb(param):
    """Checks norb"""
    test1 = Wavefunction(param=param)
    assert set([test1.norb()]) == set(norbs for _, _, norbs in param)


@pytest.mark.parametrize("param", spin_broken_cases)
def test_number_sectors(param):
    """Checks _number_sectors
    """
    test = loader(param, 'wfn')
    reference = loader(param, 'number_sectors')

    if len(param) > 1:
        # Make param not spin complete, _number_sectors should fail
        param = tuple(x for x in param[1:])
        test_fail = Wavefunction(param=param)
        with pytest.raises(ValueError):
            test_fail._number_sectors()

    numbersectors = test._number_sectors()
    assert numbersectors.keys() == reference.keys()
    for k in numbersectors:
        assert FqeDataSet_isclose(numbersectors[k], reference[k])


@pytest.mark.parametrize("param,kind", [(c, k) for c in all_cases
                                        for k in ['apply_array',
                                                  'apply_sparse',
                                                  'apply_diagonal',
                                                  'apply_quadratic',
                                                  'apply_dc']])
def test_apply(param, kind):
    """Checks _apply_array through the apply API
    """
    test = loader(param, 'wfn')
    reference_data = loader(param, kind)
    hamil = reference_data['hamil']
    if reference_data['wfn_out'] is None:
        # Out was not generated due to a thrown ValueError
        with pytest.raises(ValueError):
            out = test.apply(hamil)
    else:
        out = test.apply(hamil)
        assert Wavefunction_isclose(out, reference_data['wfn_out'])


@pytest.mark.parametrize("param,kind", [(c, k) for c in all_cases
                                        for k in ['apply_array',
                                                  'apply_sparse',
                                                  'apply_diagonal',
                                                  'apply_quadratic',
                                                  'apply_dc']])
def test_evolve(param, kind):
    """Checks time_evolve through the evolve API
    """
    test = loader(param, 'wfn')
    reference_data = loader(param, kind)
    hamil = reference_data['hamil']
    if reference_data['wfn_evolve'] is None:
        # Out was not generated due to a thrown ValueError
        with pytest.raises(ValueError):
            out = test.time_evolve(0.1, hamil)
    else:
        out = test.time_evolve(0.1, hamil)
        assert Wavefunction_isclose(out, reference_data['wfn_evolve'])


def test_chebyshev():
    """apply_generated_unitary with chebyshev polynomial expansion
    """
    norb = 4
    nalpha = 2
    nbeta = 1
    nele = nalpha + nbeta
    time = 0.1

    h1e = build_hamiltonian.build_H1(norb, full=True, asymmetric=True)

    eig, _ = numpy.linalg.eigh(h1e)
    hamil = fqe.get_general_hamiltonian((h1e,))

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='random')

    chebyshev = wfn.apply_generated_unitary(time,
                                            'chebyshev',
                                            hamil,
                                            accuracy=1e-9,
                                            spec_lim=(eig[0], eig[-1]))

    taylor = wfn.apply_generated_unitary(time, 'taylor', hamil, accuracy=1e-9)

    err = (chebyshev - taylor).norm()
    assert err < 1.e-8

def test_number_broken_rdm():
    wfn_alpha = Wavefunction([[1, 1, 2]])
    wfn_beta = Wavefunction([[1, -1, 2]])
    wfn_alpha.set_wfn(strategy='random')
    wfn_beta.set_wfn(strategy='random')

    wfn_tot = Wavefunction([[1, 1, 2], [1, -1, 2]], broken=["spin"])
    wfn_tot.set_wfn(strategy='from_data',
                    raw_data={(1, 1): wfn_alpha._civec[(1, 1)].coeff,
                              (1, -1): wfn_beta._civec[(1, -1)].coeff})

    rdm_a = wfn_alpha._compute_rdm(4)
    rdm_b = wfn_beta._compute_rdm(4)
    rdm_full1 = wfn_tot._compute_rdm(1)
    assert numpy.allclose(rdm_full1[0][:2, :2], rdm_a[0])
    assert numpy.allclose(rdm_full1[0][2:, 2:], rdm_b[0])

    rdm_full2 = wfn_tot._compute_rdm(2)
    assert numpy.allclose(rdm_full2[1][:2, :2, :2, :2], rdm_a[1])
    assert numpy.allclose(rdm_full2[1][2:, 2:, 2:, 2:], rdm_b[1])

    assert numpy.allclose(rdm_full2[0][:2, 2:], rdm_full1[0][:2, 2:])
    assert numpy.allclose(rdm_full2[0][2:, :2], rdm_full1[0][2:, :2])

    rdm_full3 = wfn_tot._compute_rdm(3)
    assert numpy.allclose(rdm_full3[2][:2, :2, :2, :2, :2, :2], rdm_a[2])
    assert numpy.allclose(rdm_full3[2][2:, 2:, 2:, 2:, 2:, 2:], rdm_b[2])

    rdm_full4 = wfn_tot._compute_rdm(4)
    assert numpy.allclose(
        rdm_full4[3][:2, :2, :2, :2, :2, :2, :2, :2], rdm_a[3])
    assert numpy.allclose(
        rdm_full4[3][2:, 2:, 2:, 2:, 2:, 2:, 2:, 2:], rdm_b[3])

def test_rdm_expectationvalue():
    norb = 4
    nalpha = 2
    nbeta = 1
    nele = nalpha + nbeta
    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='random')
    rdm1full = wfn.expectationValue('i^ j')
    rdm = wfn._compute_rdm(1)
    rdm1 = wfn.rdm('i^ j')

    assert numpy.allclose(rdm[0], rdm1full)
    assert numpy.allclose(rdm[0], rdm1)

    rdm100 = wfn.rdm('0^ 0')
    rdm100 += wfn.rdm('1^ 1')
    rdm1exp = wfn.expectationValue('0^ 0')
    rdm1exp += wfn.expectationValue('1^ 1')
    assert abs(rdm100 - rdm1exp) < 1e-13
    assert abs(rdm100 - rdm1[0, 0]) < 1e-13

    H00 = numpy.zeros((norb, norb))
    H00[0, 0] = 1.0
    H00op = get_restricted_hamiltonian((H00,))
    H00exp = wfn.expectationValue(H00op)
    assert abs(rdm100 - H00exp) < 1e-13

    Nop = NumberOperator()
    N = wfn.expectationValue(Nop)
    assert abs(N - numpy.trace(rdm1)) < 1e-13

    brawfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    brawfn.set_wfn(strategy='random')
    trdm01 = wfn.rdm('0^ 2', brawfn=brawfn)
    texp01 = wfn.expectationValue('0^ 2', brawfn=brawfn)
    assert abs(trdm01 - texp01) < 1e-13

    NtimesOvlp = wfn.expectationValue(Nop, brawfn=brawfn)
    ref = N*vdot(brawfn, wfn)
    assert abs(ref - NtimesOvlp) < 1e-13

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

def test_broken_number_3body():
    norb = 4
    time = 0.001
    wfn_spin = fqe.get_spin_conserving_wavefunction(3, norb)

    work = FermionOperator('0^ 1^ 2 3 4^ 6', 3.0 - 1.3j)
    work += hermitian_conjugated(work)
    h_noncon = fqe.build_hamiltonian(work,
                                     norb=norb,
                                     conserve_number=False)

    gen = fqe.fqe_decorators.normal_ordered(
        fqe.fqe_decorators.transform_to_spin_broken(work))
    matrix = fqe.fqe_decorators.fermionops_tomatrix(gen, norb)
    h1 = numpy.zeros((2*norb, 2*norb))
    h2 = numpy.zeros((2*norb, 2*norb, 2*norb, 2*norb))

    hamil = general_hamiltonian.General((h1, h2, matrix))
    hamil._conserve_number = False

    wfn_spin.set_wfn(strategy='random')
    nbody_evolved = wfn_spin.time_evolve(time, h_noncon)
    unitary_evolved = wfn_spin.apply_generated_unitary(
        time, 'taylor', hamil)
    for key in nbody_evolved.sectors():
        assert numpy.allclose(nbody_evolved._civec[key].coeff,
                            unitary_evolved._civec[key].coeff)
