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
""" fci_graph unit tests
"""
#pylint: disable=protected-access

import numpy
from numpy import linalg
import pytest

from openfermion import FermionOperator
from openfermion import bravyi_kitaev_code

from fqe import wavefunction
from fqe.hamiltonians import *
from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)
from fqe.fqe_decorators import build_hamiltonian
from tests.unittest_data import build_hamiltonian as test_hamiltonian
from fqe.util import tensors_equal

import fqe


def test_fqe_control_dot_vdot():
    """Find the dot product of two wavefunctions.
    """
    wfn1 = fqe.get_number_conserving_wavefunction(4, 8)
    wfn1.set_wfn(strategy='ones')
    wfn1.normalize()
    assert round(abs(fqe.vdot(wfn1, wfn1) - 1. + .0j), 7) == 0
    assert round(abs(fqe.dot(wfn1, wfn1) - 1. + .0j), 7) == 0
    wfn1.set_wfn(strategy='random')
    wfn1.normalize()
    assert round(abs(fqe.vdot(wfn1, wfn1) - 1. + .0j), 7) == 0


def test_Wavefunction():
    """Test free function that construct a Wavefunction object
    """
    wfn1 = fqe.Wavefunction([[2, 0, 2]])
    wfn2 = wavefunction.Wavefunction([[2, 0, 2]])
    for key, sector in wfn1._civec.items():
        assert key in wfn2._civec
        assert sector.coeff.shape == wfn2._civec[key].coeff.shape


def test_initialize_new_wavefunction():
    """Test initialization of the new wavefunction
    """
    nele = 3
    m_s = -1
    norb = 4
    wfn = fqe.get_wavefunction(nele, m_s, norb)
    assert isinstance(wfn, wavefunction.Wavefunction)


def test_initialize_new_wavefunctions_multi():
    """Test initialization of the new wavefunction with multiple parameters
    """
    multiple = [[4, 0, 4], [4, 2, 4], [3, -3, 4], [1, 1, 4]]
    wfns = fqe.get_wavefunction_multiple(multiple)
    for wfn in wfns:
        assert isinstance(wfn, wavefunction.Wavefunction)


def test_time_evolve():
    """Test time_evolve of a wavefunction
    """
    wfn1 = fqe.Wavefunction([[2, 0, 2]])
    wfn1.set_wfn('ones')
    wfn2 = wavefunction.Wavefunction([[2, 0, 2]])
    wfn2.set_wfn('ones')
    op = FermionOperator('1^ 3') + FermionOperator('3^ 1')
    time = 1.2
    wfn1 = fqe.time_evolve(wfn1, time, op, True)
    wfn2 = wfn2.time_evolve(time, op, True)
    for key, sector in wfn1._civec.items():
        assert key in wfn2._civec
        assert sector.coeff.shape == wfn2._civec[key].coeff.shape
        assert numpy.allclose(sector.coeff, wfn2._civec[key].coeff)


def test_apply():
    """Test time_evolve of a wavefunction
    """
    wfn1 = fqe.Wavefunction([[2, 0, 2]])
    wfn1.set_wfn('ones')
    wfn2 = wavefunction.Wavefunction([[2, 0, 2]])
    wfn2.set_wfn('ones')
    op = FermionOperator('1^ 3') + FermionOperator('3^ 1')
    wfn1 = fqe.apply(op, wfn1)
    wfn2 = wfn2.apply(op)
    for key, sector in wfn1._civec.items():
        assert key in wfn2._civec
        assert sector.coeff.shape == wfn2._civec[key].coeff.shape
        assert numpy.allclose(sector.coeff, wfn2._civec[key].coeff)


def test_expectationValue():
    """Test time_evolve of a wavefunction
    """
    wfn1 = fqe.Wavefunction([[2, 0, 2]])
    wfn1.set_wfn('ones')
    wfn2 = wavefunction.Wavefunction([[2, 0, 2]])
    wfn2.set_wfn('ones')
    op = sparse_hamiltonian.SparseHamiltonian(FermionOperator('1^ 3'))
    ex1 = fqe.expectationValue(wfn1, op)
    ex2 = wfn2.expectationValue(op)
    assert numpy.isclose(ex1, ex2)


def test_apply_generated_unitary():
    """Test applying generated unitary transformation
    """
    norb = 4
    nele = 3
    time = 0.001
    ops = FermionOperator('1^ 3^ 5 0', 2.0 - 2.j) + FermionOperator(
        '0^ 5^ 3 1', 2.0 + 2.j)

    wfn = fqe.get_number_conserving_wavefunction(nele, norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()

    reference = fqe.apply_generated_unitary(wfn, time, 'taylor', ops)

    h1e = numpy.zeros((2 * norb, 2 * norb), dtype=numpy.complex128)
    h2e = hamiltonian_utils.nbody_matrix(ops, norb)
    h2e = hamiltonian_utils.antisymm_two_body(h2e)
    hamil = general_hamiltonian.General(tuple([h1e, h2e]))
    compute = wfn.apply_generated_unitary(time, 'taylor', hamil)

    for key in wfn.sectors():
        diff = reference._civec[key].coeff - compute._civec[key].coeff
        err = linalg.norm(diff)
        assert err < 1.e-8


def test_cirq_interop(c_or_python):
    """Check the transition from a line qubit and back.
    """
    fqe.settings.use_accelerated_code = c_or_python
    work = numpy.random.rand(16).astype(numpy.complex128)
    work[0] = 0.0 + 0.0j
    work[15] = 0.0 + 0.0j
    norm = numpy.sqrt(numpy.vdot(work, work))
    numpy.divide(work, norm, out=work)
    wfn = fqe.from_cirq(work, thresh=1.0e-7)
    sec = [(1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -1), (3, 1)]
    assert set(sec) == set(wfn._civec.keys()) 
    test = fqe.to_cirq(wfn)
    assert numpy.allclose(test, work)

    # check with Bravyi-Kitaev
    bc = bravyi_kitaev_code(4)
    wfn = fqe.from_cirq(work, thresh=1.0e-7, binarycode=bc)
    test = fqe.to_cirq(wfn, binarycode=bc)
    assert numpy.allclose(test, work)


def test_get_spin_conserving_wavefunction():
    """ Test get_spin_conserving_wavefunction
    """
    norb = 4
    s_z = 2
    wfn_spin = fqe.get_spin_conserving_wavefunction(s_z, norb)

    assert 's_z' in wfn_spin._conserved.keys()
    assert wfn_spin._conserved['s_z'] == 2
    assert wfn_spin.conserve_spin()

    ref_sectors = {(2, 2), (4, 2), (6, 2)}
    assert ref_sectors == set(wfn_spin.sectors())

    s_z = -2

    wfn_spin = fqe.get_spin_conserving_wavefunction(s_z, norb)

    assert 's_z' in wfn_spin._conserved.keys()
    assert wfn_spin._conserved['s_z'] == -2
    assert wfn_spin.conserve_spin()

    ref_sectors = {(2, -2), (4, -2), (6, -2)}
    assert ref_sectors == set(wfn_spin.sectors())


def test_get_number_conserving_wavefunction():
    """ Test get_number_conserving_wavefunction
    """
    norb = 4
    nel = 2
    wfn_spin = fqe.get_number_conserving_wavefunction(nel, norb)

    assert 'n' in wfn_spin._conserved.keys()
    assert wfn_spin._conserved['n'] == 2
    assert wfn_spin.conserve_number()

    ref_sectors = {(2, 2), (2, 0), (2, -2)}
    assert ref_sectors == set(wfn_spin.sectors())


def test_operator_constructors():
    """ Creation of FQE-operators
    """
    assert isinstance(fqe.get_s2_operator(), S2Operator)
    assert isinstance(fqe.get_sz_operator(), SzOperator)
    assert isinstance(fqe.get_time_reversal_operator(), TimeReversalOp)
    assert isinstance(fqe.get_number_operator(), NumberOperator)


def test_get_hamiltonian_from_openfermion_raises():
    """ Check the type check of get_hamiltonian_from_openfermion()
    """
    with pytest.raises(AssertionError):
        fqe.get_hamiltonian_from_openfermion([])


def test_get_hamiltonian_from_openfermion():
    """ Check get_hamiltonian_from_openfermion()
    """
    norb = 4
    ops = test_hamiltonian.number_nonconserving_fop(2, norb=norb)
    test = fqe.get_hamiltonian_from_openfermion(ops, norb=norb, \
                                          conserve_number=False)
    test2 = build_hamiltonian(ops, norb=norb, conserve_number=False)
    assert test == test2


def test_get_diagonal_hamiltonian():
    """ Check whether get_diagonal_hamiltonian returns the same value as its
        underlying function is supposed to return.
    """
    diag = numpy.zeros((5,), dtype=numpy.complex128)
    e_0 = -4.2
    test = diagonal_hamiltonian.Diagonal(diag, e_0)
    test2 = fqe.get_diagonal_hamiltonian(diag, e_0)

    assert test == test2


def test_get_diagonal_coulomb():
    """ Check whether get_diagonal_coulomb returns the same value as its
        underlying function is supposed to return.
    """
    diag = numpy.zeros((5, 5), dtype=numpy.complex128)
    e_0 = -4.2
    test = diagonal_coulomb.DiagonalCoulomb(diag, e_0)
    test2 = fqe.get_diagonalcoulomb_hamiltonian(diag, e_0)

    assert test == test2


@pytest.mark.parametrize("hamiltonian, get_function", \
              [ (sso_hamiltonian.SSOHamiltonian, fqe.get_sso_hamiltonian),
                (gso_hamiltonian.GSOHamiltonian, fqe.get_gso_hamiltonian),
                (general_hamiltonian.General, fqe.get_general_hamiltonian),
                (restricted_hamiltonian.RestrictedHamiltonian, \
                 fqe.get_restricted_hamiltonian)
                ])
def test_get_hamiltonians(hamiltonian, get_function):
    """ Check whether other Hamiltonian getters return the same value
    as their underlying functions are supposed to return.
    """
    h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
    e_0 = -4.2
    test = hamiltonian((h1e,))
    test2 = get_function((h1e,))

    assert test == test2


def test_get_sparse_hamiltonian():
    oper = FermionOperator('0 0^')
    test = sparse_hamiltonian.SparseHamiltonian(oper)
    test2 = fqe.get_sparse_hamiltonian(oper)

    assert test == test2
