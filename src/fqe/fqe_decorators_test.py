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
import unittest

import numpy

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

from fqe.wavefunction import Wavefunction

from fqe import to_cirq, from_cirq


class TestFqedecorators(unittest.TestCase):
    """Fqe decorators test class
    """

    def test_basic_split(self):
        """Test spliting the fermion operators for a simple case.
        """
        test_ops = FermionOperator('1 1^', 1.0)
        test_ops = normal_ordered(test_ops)
        terms, _ = split_openfermion_tensor(test_ops)
        self.assertEqual(FermionOperator('1^ 1', -1.0), terms[2])

    def test_split_rank2468(self):
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
            with self.subTest(rank=rank):
                self.assertEqual(ops[rank], terms[rank])

    def test_odd_rank_error(self):
        """Check that odd rank operators are not processed
        """
        ops = []
        ops.append(FermionOperator('10^ 1 2', 1.0))
        ops.append(FermionOperator('5^ 3^ 7^ 8 7', 1.0))
        for rank in range(2):
            with self.subTest(rank=(2 * rank + 1)):
                self.assertRaises(ValueError, split_openfermion_tensor,
                                  ops[rank])

    def test_fermoinops_tomatrix(self):
        norb = 4
        ops = FermionOperator('9^ 1')
        self.assertRaises(ValueError, fermionops_tomatrix, ops, norb)
        ops = FermionOperator('10^ 4')
        self.assertRaises(ValueError, fermionops_tomatrix, ops, norb)
        ops = FermionOperator('3^ 1 4')
        self.assertRaises(ValueError, fermionops_tomatrix, ops, norb)
        ops = FermionOperator('3 1')
        self.assertRaises(ValueError, fermionops_tomatrix, ops, norb)
        ops = FermionOperator('3^ 1^')
        self.assertRaises(ValueError, fermionops_tomatrix, ops, norb)

    def test_process_rank2_matrix(self):
        numpy.random.seed(seed=409)
        raw = numpy.random.rand(8, 8) + 1.j * numpy.random.rand(8, 8)
        self.assertRaises(ValueError, process_rank2_matrix, raw, 0)

    def test_check_diagonal_coulomb(self):
        mat = numpy.random.rand(4, 4, 4, 4)
        self.assertTrue(not check_diagonal_coulomb(mat))

    def test_transform_to_spin_broken(self):
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
        self.assertEqual(ref_ops, test)

    def test_build_hamiltonian_paths(self):
        """Check that all cases of hamiltonian objects are built
        """
        self.assertRaises(TypeError, build_hamiltonian, 0)
        with self.subTest(name='general'):
            ops = FermionOperator('1^ 4^ 0 3', 1.0) \
                  + FermionOperator('0^ 5^ 3 2^ 4^ 1 7 6', 1.2) \
                  + FermionOperator('1^ 6', -0.3)
            ops += hermitian_conjugated(ops)
            self.assertIsInstance(build_hamiltonian(ops),
                                  general_hamiltonian.General)

        with self.subTest(name='sparse'):
            ops = FermionOperator('5^ 1^ 3^ 2 0 1', 1.0 - 1.j) \
                  + FermionOperator('1^ 0^ 2^ 3 1 5', 1.0 + 1.j)
            self.assertIsInstance(build_hamiltonian(ops),
                                  sparse_hamiltonian.SparseHamiltonian)

        with self.subTest(name='diagonal'):
            ops = FermionOperator('1^ 1', 1.0) \
                  + FermionOperator('2^ 2', 2.0) \
                  + FermionOperator('3^ 3', 3.0) \
                  + FermionOperator('4^ 4', 4.0)
            self.assertIsInstance(build_hamiltonian(ops),
                                  diagonal_hamiltonian.Diagonal)

        with self.subTest(name='gso'):
            ops = FermionOperator()
            for i in range(4):
                for j in range(4):
                    opstr = str(i) + '^ ' + str(j)
                    coeff = complex((i + 1) * (j + 1) * 0.1)
                    ops += FermionOperator(opstr, coeff)
            self.assertIsInstance(build_hamiltonian(ops),
                                  gso_hamiltonian.GSOHamiltonian)

        with self.subTest(name='restricted'):
            ops = FermionOperator()
            for i in range(0, 3, 2):
                for j in range(0, 3, 2):
                    coeff = complex((i + 1) * (j + 1) * 0.1)
                    opstr = str(i) + '^ ' + str(j)
                    ops += FermionOperator(opstr, coeff)
                    opstr = str(i + 1) + '^ ' + str(j + 1)
                    ops += FermionOperator(opstr, coeff)
            self.assertIsInstance(build_hamiltonian(ops),
                                  restricted_hamiltonian.RestrictedHamiltonian)

        with self.subTest(name='sso'):
            ops = FermionOperator()
            for i in range(0, 3, 2):
                for j in range(0, 3, 2):
                    coeff = complex((i + 1) * (j + 1) * 0.1)
                    opstr = str(i) + '^ ' + str(j)
                    ops += FermionOperator(opstr, coeff)
                    opstr = str(i + 1) + '^ ' + str(j + 1)
                    coeff *= 1.5
                    ops += FermionOperator(opstr, coeff)
            self.assertIsInstance(build_hamiltonian(ops),
                                  sso_hamiltonian.SSOHamiltonian)

        with self.subTest(name='diagonal_coulomb'):
            ops = FermionOperator()
            for i in range(4):
                for j in range(4):
                    opstring = str(i) + '^ ' + str(j) + '^ ' + str(
                        i) + ' ' + str(j)
                    ops += FermionOperator(opstring, 0.001 * (i + 1) * (j + 1))
            self.assertIsInstance(build_hamiltonian(ops),
                                  diagonal_coulomb.DiagonalCoulomb)

    def test_evolve_spinful_fermionop(self):
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
        self.assertTrue(
            numpy.allclose(test_state.get_coeff((2, 0)),
                           new_state_wfn.get_coeff((2, 0))))

    def test_apply_spinful_fermionop(self):
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

        self.assertTrue(
            numpy.allclose(test_state.get_coeff((2, 0)),
                           new_state_wfn.get_coeff((2, 0))))
