#   Copyright 2019 Quantum Simulation Technologies Inc.
#
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

import numpy

from openfermion.utils import normal_ordered
from openfermion import FermionOperator
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import restricted_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian


import unittest

from fqe.fqe_decorators import *


class TestFqedecorators(unittest.TestCase):


    def test_basic_split(self):
        """
        """
        test_ops = FermionOperator('1^ 1', 1.0)
        terms = split_openfermion_tensor(test_ops)
        self.assertEqual(test_ops, terms[2])


    def test_split_rank2468(self):
        """
        """
        ops = {}
        ops[2] = FermionOperator('10^ 1', 1.0)
        ops[4] = FermionOperator('3^ 7^ 8 7', 1.0)
        ops[6] = FermionOperator('7^ 6^ 2^ 2 1 9', 1.0)
        ops[8] = FermionOperator('0^ 3^ 2^ 11^ 12 3 9 4', 1.0)
        full_string = ops[2] + ops[4] + ops[6] + ops[8]
        terms = split_openfermion_tensor(full_string)
        for rank in range(2, 9, 2):
            with self.subTest(rank=rank):
                self.assertEqual(ops[rank], terms[rank])


    def test_odd_rank_error(self):
        """
        """
        ops = []
        ops.append(FermionOperator('10^ 1 2', 1.0))
        ops.append(FermionOperator('5^ 3^ 7^ 8 7', 1.0))
        for rank in range(2):
            with self.subTest(rank=(2*rank + 1)):
                self.assertRaises(ValueError, split_openfermion_tensor, ops[rank])


    def test_build_rank2_matrix(self):
        # === numpy control options ===
        numpy.set_printoptions(floatmode='fixed', precision=6, linewidth=200, suppress=True)
        numpy.random.seed(seed=409)

        norb = 2
        raw = numpy.random.rand(8, 8) + 1.j*numpy.random.rand(8, 8)
        hermitian = raw + raw.conj().T

        opstr = FermionOperator()

        for i in range(norb):
            for j in range(norb):
                opstr += FermionOperator(str(2*i) + '^ ' + str(2*j), hermitian[i, j])
                opstr += FermionOperator(str(2*i) + '^ ' + str(2*j + 1), hermitian[i, j + norb])
                opstr += FermionOperator(str(2*i + 1) + '^ ' + str(2*j), hermitian[i + norb, j])
                opstr += FermionOperator(str(2*i + 1) + '^ ' + str(2*j + 1), hermitian[i + norb, j + norb])

        for i in range(norb):
            for j in range(norb):
                opstr += FermionOperator(str(2*i) + '^ ' + str(2*j) + '^ ', hermitian[i, j + 2*norb])
                opstr += FermionOperator(str(2*i) + '^ ' + str(2*j + 1) + '^ ', hermitian[i, j + 3*norb])
                opstr += FermionOperator(str(2*i + 1) + '^ ' + str(2*j) + '^ ', hermitian[i + norb, j + 2*norb])
                opstr += FermionOperator(str(2*i + 1) + '^ ' + str(2*j + 1) + '^ ', hermitian[i + norb, j + 3*norb])

        for i in range(norb):
            for j in range(norb):
                opstr += FermionOperator(str(2*i) + ' ' + str(2*j), hermitian[i + 2*norb, j])
                opstr += FermionOperator(str(2*i) + ' ' + str(2*j + 1), hermitian[i + 2*norb, j + norb])
                opstr += FermionOperator(str(2*i + 1) + ' ' + str(2*j), hermitian[i + 3*norb, j])
                opstr += FermionOperator(str(2*i + 1) + ' ' + str(2*j + 1), hermitian[i + 3*norb, j + norb])

        for i in range(norb):
            for j in range(norb):
                opstr += FermionOperator(str(2*i) + ' ' + str(2*j) + '^ ', hermitian[i + 2*norb, j + 2*norb])
                opstr += FermionOperator(str(2*i) + ' ' + str(2*j + 1) + '^ ', hermitian[i + 2*norb, j + 3*norb])
                opstr += FermionOperator(str(2*i + 1) + ' ' + str(2*j) + '^ ', hermitian[i + 3*norb, j + 2*norb])
                opstr += FermionOperator(str(2*i + 1) + ' ' + str(2*j + 1) + '^ ', hermitian[i + 3*norb, j + 3*norb])

        test_hamil = fermion_op_to_rank2(opstr)
        self.assertTrue(numpy.allclose(test_hamil, hermitian))


    def test_transform_to_number_broken(self):
        """
        """
        in_ops = FermionOperator('5^ 7', 1.0)
        in_ops += FermionOperator('0^ 2^ 1 3', 2.0)
        in_ops += FermionOperator('5^ 6 1 7', 3.0)

        ref_ops = FermionOperator('7^ 5', -1.0)
        ref_ops += FermionOperator('3^ 2^ 1^ 0^', -2.0)
        ref_ops += FermionOperator('7^ 1^ 6 5', 3.0)
        test = normal_ordered(transform_to_number_broken(in_ops))
        self.assertEqual(ref_ops, test)


    def test_build_hamiltonian_paths(self):
        """
        """
        with self.subTest(name='general'):
            ops = FermionOperator('1^ 4^ 0 3', 1.0) \
                  + FermionOperator('0^ 5^ 3 2^ 4^ 1 7 6', 1.2) \
                  + FermionOperator('1^ 6', -0.3)
            self.assertIsInstance(build_hamiltonian(ops), general_hamiltonian.General)

        with self.subTest(name='sparse'):
            ops = FermionOperator('5^ 1^ 3^ 2 0 1', 1.0 - 1.j) \
                  + FermionOperator('1^ 0^ 2^ 3 1 5', 1.0 + 1.j)
            self.assertIsInstance(build_hamiltonian(ops), sparse_hamiltonian.SparseHamiltonian)

        with self.subTest(name='diagonal'):
            ops = FermionOperator('1^ 1', 1.0) \
                  + FermionOperator('2^ 2', 2.0) \
                  + FermionOperator('3^ 3', 3.0) \
                  + FermionOperator('4^ 4', 4.0) 
            self.assertIsInstance(build_hamiltonian(ops), diagonal_hamiltonian.Diagonal)

        with self.subTest(name='gso'):
            ops = FermionOperator()
            for i in range(4):
                for j in range(4):
                    opstr = str(i) + '^ ' + str(j)
                    coeff = complex((i + 1)*(j + 1)*0.1)
                    ops += FermionOperator(opstr, coeff)
            self.assertIsInstance(build_hamiltonian(ops), gso_hamiltonian.GSOHamiltonian)

        with self.subTest(name='restricted'):
            ops = FermionOperator()
            for i in range(0, 3, 2):
                for j in range(0, 3, 2):
                    coeff = complex((i + 1)*(j + 1)*0.1)
                    opstr = str(i) + '^ ' + str(j)
                    ops += FermionOperator(opstr, coeff)
                    opstr = str(i + 1) + '^ ' + str(j + 1)
                    ops += FermionOperator(opstr, coeff)
            self.assertIsInstance(build_hamiltonian(ops), restricted_hamiltonian.Restricted)

        with self.subTest(name='sso'):
            ops = FermionOperator()
            for i in range(0, 3, 2):
                for j in range(0, 3, 2):
                    coeff = complex((i + 1)*(j + 1)*0.1)
                    opstr = str(i) + '^ ' + str(j)
                    ops += FermionOperator(opstr, coeff)
                    opstr = str(i + 1) + '^ ' + str(j + 1)
                    coeff *= 1.5
                    ops += FermionOperator(opstr, coeff)
            self.assertIsInstance(build_hamiltonian(ops), sso_hamiltonian.SSOHamiltonian)



if __name__ == "__main__":
    unittest.main()
