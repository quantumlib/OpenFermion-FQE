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
"""Unittest for the fqe decorators
"""

import copy
from itertools import product

import numpy
import pytest
import scipy

from openfermion.transforms import normal_ordered
from openfermion import (FermionOperator, hermitian_conjugated,
                         get_sparse_operator)
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import restricted_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import sso_hamiltonian
from fqe.hamiltonians import gso_hamiltonian
from fqe.hamiltonians import diagonal_coulomb
from fqe.hamiltonians import diagonal_hamiltonian

from fqe.fqe_decorators import split_openfermion_tensor
from fqe.fqe_decorators import build_hamiltonian
from fqe.fqe_decorators import transform_to_spin_broken
from fqe.fqe_decorators import fermionops_tomatrix
from fqe.fqe_decorators import process_rank2_matrix
from fqe.fqe_decorators import check_diagonal_coulomb
from fqe import get_spin_conserving_wavefunction

from fqe.wavefunction import Wavefunction

from fqe import to_cirq, from_cirq

def test_basic_split():
    """Test spliting the fermion operators for a simple case.
    """
    test_ops = FermionOperator('1 1^', 1.0)
    test_ops = normal_ordered(test_ops)
    terms, _ = split_openfermion_tensor(test_ops)
    assert FermionOperator('1^ 1', -1.0) == terms[2]

def test_split_rank2468():
    """Split up to rank four operators
    """
    ops = {}
    ops[2] = FermionOperator('10^ 1', 1.0)
    ops[4] = FermionOperator('3^ 7^ 8 7', 1.0)
    ops[6] = FermionOperator('7^ 6^ 2^ 2 1 9', 1.0)
    ops[8] = FermionOperator('0^ 3^ 2^ 11^ 12 3 9 4', 1.0)
    full_string = ops[2] + ops[4] + ops[6] + ops[8]
    terms, _ = split_openfermion_tensor(full_string)
    for rank in range(2, 9, 2):
        assert ops[rank] == terms[rank]

def test_odd_rank_error():
    """Check that odd rank operators are not processed
    """
    ops = []
    ops.append(FermionOperator('10^ 1 2', 1.0))
    ops.append(FermionOperator('5^ 3^ 7^ 8 7', 1.0))
    for rank in range(2):
        with pytest.raises(ValueError):
            split_openfermion_tensor(ops[rank])

def test_fermionops_tomatrix():
    """Check if fermionops_tomatrix raises required exceptions
    """
    norb = 4
    ops = FermionOperator('9^ 1')
    with pytest.raises(ValueError):
        fermionops_tomatrix(ops, norb)
    ops = FermionOperator('10^ 4')
    with pytest.raises(ValueError):
        fermionops_tomatrix(ops, norb)
    ops = FermionOperator('3^ 1 4')
    with pytest.raises(ValueError):
        fermionops_tomatrix(ops, norb)
    ops = FermionOperator('3 1')
    with pytest.raises(ValueError):
        fermionops_tomatrix(ops, norb)
    ops = FermionOperator('3^ 1^')
    with pytest.raises(ValueError):
        fermionops_tomatrix(ops, norb)

def test_process_rank2_matrix():
    numpy.random.seed(seed=409)
    raw = numpy.random.rand(8, 8) + 1.j * numpy.random.rand(8, 8)
    with pytest.raises(ValueError):
        process_rank2_matrix(raw, 0)

def test_check_diagonal_coulomb():
    mat = numpy.random.rand(4, 4, 4, 4)
    assert not check_diagonal_coulomb(mat)

def test_transform_to_spin_broken():
    """Check the conversion between number and spin broken
    representations
    """
    in_ops = FermionOperator('5^ 7', 1.0)
    in_ops += FermionOperator('0^ 2^ 1 3', 2.0)
    in_ops += FermionOperator('5^ 6 1 7', 3.0)

    ref_ops = FermionOperator('7^ 5', -1.0)
    ref_ops += FermionOperator('3^ 2^ 1^ 0^', -2.0)
    ref_ops += FermionOperator('7^ 1^ 6 5', 3.0)
    test = normal_ordered(transform_to_spin_broken(in_ops))
    assert ref_ops == test

def test_fail_empty_hamiltonian():
    """Check that all cases of hamiltonian objects are built
    """
    with pytest.raises(TypeError):
        build_hamiltonian(0)

def test_general_hamiltonian():
    ops = FermionOperator('1^ 4^ 0 3', 1.0) \
          + FermionOperator('0^ 5^ 4^ 1 7 6', 1.2) \
          + FermionOperator('1^ 6', -0.3)
    ops += hermitian_conjugated(ops)
    ham = build_hamiltonian(ops, norb=5)
    assert isinstance(ham, general_hamiltonian.General)
    assert ham._tensor[2].shape[0] == 10
    assert ham._tensor[2][5, 3] == -0.3
    assert ham._tensor[2][3, 5] == -0.3
    ham._tensor[2][5, 3] = 0.0
    ham._tensor[2][3, 5] = 0.0
    assert not numpy.any(ham._tensor[2])
    assert ham._tensor[4][5, 2, 6, 0] == -0.5
    assert ham._tensor[4][6, 0, 5, 2] == -0.5

def test_sparse_hamiltonian():
    ops = FermionOperator('5^ 1^ 3^ 2 0 1', 1.0 - 1.j) \
          + FermionOperator('1^ 0^ 2^ 3 1 5', 1.0 + 1.j)
    ham = build_hamiltonian(ops)
    assert isinstance(ham, sparse_hamiltonian.SparseHamiltonian)
    sparse = [((-1+1j), [(1, 0), (0, 0)], [(2, 1), (1, 1), (0, 1), (0, 0)]),
              ((-1-1j), [(1, 1), (0, 1)], [(0, 1), (2, 0), (1, 0), (0, 0)])]
    for i, data in enumerate(ham._operators):
        assert abs(data[0] - sparse[i][0]) < 1.0e-8
        assert data[1:2] == sparse[i][1:2]

def test_diagonal_hamiltonian():
    ops = FermionOperator('1^ 1', 1.0) \
          + FermionOperator('2^ 2', 2.0) \
          + FermionOperator('3^ 3', 3.0) \
          + FermionOperator('4^ 4', 4.0)
    ham = build_hamiltonian(ops)
    assert isinstance(ham, diagonal_hamiltonian.Diagonal)
    assert (ham.diag_values() == numpy.array([0.0, 2.0, 4.0, 1.0, 3.0, 0.0],
                                             dtype=numpy.complex128)).all() 

def test_gso_hamiltonian():
    ops = FermionOperator()
    for i in range(4):
        for j in range(4):
            opstr = str(i) + '^ ' + str(j)
            coeff = complex((i + 1) * (j + 1) * 0.1)
            ops += FermionOperator(opstr, coeff)
    ham = build_hamiltonian(ops)
    assert isinstance(ham, gso_hamiltonian.GSOHamiltonian)
    ref = numpy.array([[0.1+0.j, 0.3+0.j, 0.2+0.j, 0.4+0.j],
                       [0.3+0.j, 0.9+0.j, 0.6+0.j, 1.2+0.j],
                       [0.2+0.j, 0.6+0.j, 0.4+0.j, 0.8+0.j],
                       [0.4+0.j, 1.2+0.j, 0.8+0.j, 1.6+0.j]], dtype=numpy.complex128)
    assert numpy.allclose(ham._tensor[2], ref)

def test_restricted_hamiltonian():
    ops = FermionOperator()
    for i in range(0, 3, 2):
        for j in range(0, 3, 2):
            coeff = complex((i + 1) * (j + 1) * 0.1)
            opstr = str(i) + '^ ' + str(j)
            ops += FermionOperator(opstr, coeff)
            opstr = str(i + 1) + '^ ' + str(j + 1)
            ops += FermionOperator(opstr, coeff)
    ham = build_hamiltonian(ops)
    assert isinstance(ham, restricted_hamiltonian.RestrictedHamiltonian)
    ref = numpy.array([[0.1+0.j, 0.3+0.j],
                       [0.3+0.j, 0.9+0.j]], dtype=numpy.complex128)
    assert numpy.allclose(ham._tensor[2], ref)

def test_sso_hamiltonian():
    ops = FermionOperator()
    for i in range(0, 3, 2):
        for j in range(0, 3, 2):
            coeff = complex((i + 1) * (j + 1) * 0.1)
            opstr = str(i) + '^ ' + str(j)
            ops += FermionOperator(opstr, coeff)
            opstr = str(i + 1) + '^ ' + str(j + 1)
            coeff *= 1.5
            ops += FermionOperator(opstr, coeff)
    ham = build_hamiltonian(ops)
    assert isinstance(ham, sso_hamiltonian.SSOHamiltonian)
    ref = numpy.array([[0.1 +0.j, 0.3 +0.j, 0.  +0.j, 0.  +0.j],
                       [0.3 +0.j, 0.9 +0.j, 0.  +0.j, 0.  +0.j],
                       [0.  +0.j, 0.  +0.j, 0.15+0.j, 0.45+0.j],
                       [0.  +0.j, 0.  +0.j, 0.45+0.j, 1.35+0.j]], dtype=numpy.complex128)
    assert numpy.allclose(ham._tensor[2], ref)

def test_diagonal_coulomb_hamiltonian():
    ops = FermionOperator()
    norb = 4
    ref = numpy.zeros((norb, norb), dtype=numpy.complex128)
    index = [0, 2, 1, 3]
    for i in range(norb):
        for j in range(norb):
            opstring = str(i) + '^ ' + str(j) + '^ ' + str(
                i) + ' ' + str(j)
            value = 0.001 * (index[i] + 1) * (index[j] + 1)
            ops += FermionOperator(opstring, value)
            ref[index[i], index[j]] -= value * 0.5
            ref[index[j], index[i]] -= value * 0.5
    numpy.fill_diagonal(ref, 0.0)
    ham = build_hamiltonian(ops)
    assert isinstance(ham, diagonal_coulomb.DiagonalCoulomb)
    assert numpy.allclose(ham._tensor[2], ref)

def test_build_hamiltonian_number_broken():
    """Check build_hamiltonian in the number-broken case 
    """
    in_ops = FermionOperator('5^ 7', 1.0)
    in_ops += FermionOperator('0^ 2^ 1^ 3^', 2.0)
    in_ops += FermionOperator('5^ 6 1 7', 3.0)

    ref_ops = FermionOperator('7^ 5', -1.0)
    ref_ops += FermionOperator('0^ 2^ 1 3', 2.0)
    ref_ops += FermionOperator('7^ 1^ 6 5', 3.0)

    in_ops += hermitian_conjugated(in_ops) 
    ham1 = build_hamiltonian(in_ops, conserve_number=False) 

    ref_ops += hermitian_conjugated(ref_ops) 
    ham2 = build_hamiltonian(ref_ops, conserve_number=True)
    for key in ham1._tensor.keys():
        assert key in ham2._tensor.keys()
        assert numpy.allclose(ham1._tensor[key], ham2._tensor[key])
    assert not ham1.conserve_number()
    assert ham2.conserve_number()

def test_evolve_spinful_fermionop():
    """
    Make sure the spin-orbital reordering is working by comparing
    time evolution
    """
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')
    wfn.normalize()
    cirq_wf = to_cirq(wfn).reshape((-1, 1))

    op_to_apply = FermionOperator()
    for p, q, r, s in product(range(2), repeat=4):
        op = FermionOperator(
            ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0)),
            coefficient=numpy.random.randn())
        op_to_apply += op + hermitian_conjugated(op)

    opmat = get_sparse_operator(op_to_apply, n_qubits=4).toarray()
    dt = 0.765
    new_state_cirq = scipy.linalg.expm(-1j * dt * opmat) @ cirq_wf
    new_state_wfn = from_cirq(new_state_cirq.flatten(), thresh=1.0E-12)
    test_state = wfn.time_evolve(dt, op_to_apply)
    numpy.testing.assert_almost_equal(test_state.get_coeff((2, 0)),
                                      new_state_wfn.get_coeff((2, 0)))

def test_apply_spinful_fermionop():
    """
    Make sure the spin-orbital reordering is working by comparing
    apply operation
    """
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')
    wfn.normalize()
    cirq_wf = to_cirq(wfn).reshape((-1, 1))

    op_to_apply = FermionOperator()
    test_state = copy.deepcopy(wfn)
    test_state.set_wfn('zero')
    for p, q, r, s in product(range(2), repeat=4):
        op = FermionOperator(
            ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0)),
            coefficient=numpy.random.randn())
        op_to_apply += op + hermitian_conjugated(op)
        test_state += wfn.apply(op + hermitian_conjugated(op))

    opmat = get_sparse_operator(op_to_apply, n_qubits=4).toarray()
    new_state_cirq = opmat @ cirq_wf

    # this part is because we need to pass a normalized wavefunction
    norm_constant = new_state_cirq.conj().T @ new_state_cirq
    new_state_cirq /= numpy.sqrt(norm_constant)
    new_state_wfn = from_cirq(new_state_cirq.flatten(), thresh=1.0E-12)
    new_state_wfn.scale(numpy.sqrt(norm_constant))

    assert numpy.allclose(test_state.get_coeff((2, 0)),
                          new_state_wfn.get_coeff((2, 0)))

def test_rdm_fermionop():
    numpy.random.seed(seed=409)
    wfn = Wavefunction([[2, 0, 2]])
    wfn.set_wfn(strategy='random')
    wfn.normalize()
    rdm1 = wfn.rdm('i^ j')
    ref = numpy.array([[ 0.64858462+0.j        , -0.39651606-0.67039465j],
                       [-0.39651606+0.67039465j,  1.35141538+0.j        ]])
    numpy.testing.assert_almost_equal(rdm1, ref)


def test_rdm_fermionop_broken():
    numpy.random.seed(seed=409)
    wfn = get_spin_conserving_wavefunction(s_z=0, norb=2)
    wfn.set_wfn(strategy='random')
    wfn.normalize()
    rdm1 = wfn.rdm('i^ j')
    ref = numpy.array([[ 0.68460237+0.j        ,  0.1763499 +0.14853599j,
                         0.12581489+0.07549915j, -0.02583937-0.03371479j],
                       [ 0.1763499 -0.14853599j,  0.43412586+0.j        ,
                        -0.19017178-0.0074684j ,  0.02792504-0.12677385j],
                       [ 0.12581489-0.07549915j, -0.19017178+0.0074684j ,
                         0.60527581+0.j        , -0.08557504+0.18551942j],
                       [-0.02583937+0.03371479j,  0.02792504+0.12677385j,
                        -0.08557504-0.18551942j,  0.27599596+0.j        ]])
    numpy.testing.assert_almost_equal(rdm1, ref)

    assert abs(wfn.rdm('0^ 1^') \
        - (0.12581488681522182+0.07549915168581758j)) < 1.0e-8 
