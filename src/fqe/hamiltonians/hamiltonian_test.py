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
"""Base class for Hamiltonians in the fqe
"""
#the Test class is just to make sure the Hamiltonian class is tested
#pylint: disable=useless-super-delegation

import unittest

from typing import Tuple

import numpy
from openfermion import FermionOperator

from fqe.hamiltonians import hamiltonian
from fqe.hamiltonians import diagonal_coulomb
from fqe.hamiltonians import diagonal_hamiltonian
from fqe.hamiltonians import gso_hamiltonian
from fqe.hamiltonians import sso_hamiltonian
from fqe.hamiltonians import restricted_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import general_hamiltonian

class TestHamiltonian(unittest.TestCase):
    """Test class for the base Hamiltonian
    """


    def test_general(self):
        """The base Hamiltonian initializes some common vaiables
        """
        class Test(hamiltonian.Hamiltonian):
            """A testing dummy class
            """

            def __init__(self):
                super().__init__()


            def dim(self) -> int:
                return super().dim()


            def calc_diag_transform(self) -> numpy.ndarray:
                return super().calc_diag_transform()


            def rank(self) -> int:
                return super().rank()


            def iht(self, time: float) -> Tuple[numpy.ndarray, ...]:
                return super().iht(time)


            def tensors(self) -> Tuple[numpy.ndarray, ...]:
                return super().tensors()


            def diag_values(self) -> numpy.ndarray:
                return super().diag_values()


            def transform(self, trans: numpy.ndarray) -> numpy.ndarray:
                return super().transform(trans)


        test = Test()
        self.assertEqual(test.dim(), 0)
        self.assertEqual(test.rank(), 0)
        self.assertEqual(test.e_0(), 0. + 0.j)
        self.assertEqual(test.tensors(), tuple())
        self.assertEqual(test.iht(0.0), tuple())
        self.assertEqual(test.diag_values().shape, (0, ))
        self.assertEqual(test.calc_diag_transform().shape, (0, ))
        self.assertEqual(test.transform(numpy.empty(0)).shape, (0, ))
        self.assertFalse(test.quadratic())
        self.assertFalse(test.diagonal())
        self.assertFalse(test.diagonal_coulomb())
        self.assertTrue(test.conserve_number())


    def test_diagonal_coulomb(self):
        """Test some of the functions in DiagonalCoulomb
        """
        diag = numpy.zeros((5, 5), dtype=numpy.complex128)
        test = diagonal_coulomb.DiagonalCoulomb(diag)
        self.assertEqual(test.dim(), 5)
        self.assertEqual(test.rank(), 4)


    def test_diagonal(self):
        """Test some of the functions in Diagonal
        """
        bad_diag = numpy.zeros((5, 5), dtype=numpy.complex128)
        self.assertRaises(ValueError, diagonal_hamiltonian.Diagonal, bad_diag)
        diag = numpy.zeros((5, ), dtype=numpy.complex128)
        test = diagonal_hamiltonian.Diagonal(diag)
        self.assertEqual(test.dim(), 5)
        self.assertEqual(test.rank(), 2)


    def test_gso(self):
        """Test some of the functions in GSOHamiltonian
        """
        h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
        test = gso_hamiltonian.GSOHamiltonian((h1e, ))
        self.assertEqual(test.dim(), 5)
        self.assertEqual(test.rank(), 2)
        self.assertTrue(numpy.allclose(h1e, test.tensor(2)))
        self.assertRaises(TypeError, gso_hamiltonian.GSOHamiltonian, "test")


    def test_sso(self):
        """Test some of the functions in SSOHamiltonian
        """
        h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
        test = sso_hamiltonian.SSOHamiltonian((h1e, ))
        self.assertEqual(test.dim(), 5)
        self.assertEqual(test.rank(), 2)
        self.assertTrue(numpy.allclose(h1e, test.tensor(2)))
        self.assertRaises(TypeError, sso_hamiltonian.SSOHamiltonian, "test")


    def test_restricted(self):
        """Test some of the functions in SSOHamiltonian
        """
        h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
        test = restricted_hamiltonian.Restricted((h1e, ))
        self.assertEqual(test.dim(), 5)
        self.assertEqual(test.rank(), 2)
        self.assertTrue(numpy.allclose(h1e, test.tensor(2)))
        self.assertRaises(TypeError,
                          restricted_hamiltonian.Restricted,
                          "test")


    def test_general_hamiltonian(self):
        """Test some of the functions in General
        """
        h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
        h1e += h1e.T.conj()
        test = general_hamiltonian.General((h1e, ))
        self.assertEqual(test.dim(), 5)
        self.assertEqual(test.rank(), 2)
        self.assertTrue(numpy.allclose(h1e, test.tensor(2)))

        trans = test.calc_diag_transform()
        h1e = trans.T.conj() @ h1e @ trans
        self.assertTrue(numpy.allclose(h1e, test.transform(trans)))
        for i in range(h1e.shape[0]):
            h1e[i, i] = 0.0
        self.assertTrue(numpy.std(h1e) < 1.0e-8)
        self.assertRaises(TypeError, general_hamiltonian.General, "test")


    def test_sparse(self):
        """Test some of the functions in SparseHamiltonian
        """
        oper = FermionOperator('0 0^')
        oper += FermionOperator('1 1^')
        test = sparse_hamiltonian.SparseHamiltonian(4, oper)
        self.assertEqual(test.dim(), 4)
        self.assertEqual(test.rank(), 2)
        test = sparse_hamiltonian.SparseHamiltonian(4, oper, False)
        self.assertEqual(test.dim(), 8)
        self.assertTrue(not test.is_individual())
