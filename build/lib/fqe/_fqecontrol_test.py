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

""" fci_graph unit tests
"""

import unittest

import numpy

from openfermion import FermionOperator
from openfermion.transforms import get_quadratic_hamiltonian as quad_of

from fqe.util import rand_wfn
import fqe

class FqeControlTest(unittest.TestCase):
    """For Fqe structures
    """


    def test_fqe_control_dot_vdot(self):
        """There should not be more particles than orbitals
        """
        wfn1 = fqe.get_spin_nonconserving_wavefunction(1)
        wfn1.set_wfn(strategy='lowest')
        wfn2 = fqe.get_wavefunction_multiple([[1, 1, 2]])
        wfn2[0].set_wfn(strategy='lowest')
        self.assertAlmostEqual(fqe.vdot(wfn1, wfn2[0]), 1. + .0j)
        self.assertAlmostEqual(fqe.dot(wfn1, wfn2[0]), 1. + .0j)
        wfn3 = fqe.get_wavefunction_multiple([[0, 0, 1], [1, 1, 1]])
        self.assertAlmostEqual(fqe.vdot(wfn3[0], wfn3[1]), 0. + .0j)
        self.assertAlmostEqual(fqe.dot(wfn3[0], wfn3[1]), 0. + .0j)


    def test_fqecontrol_init(self):
        """Set wavefunction data from input rather than an internal method
        """
        wfn = fqe.get_spin_nonconserving_wavefunction(2)
        data = {}
        data[(2, 0)] = .5*numpy.ones((16, 10), dtype=numpy.complex64)
        wfn.set_wfn(strategy='from_data', raw_data=data)
        self.assertTrue(numpy.allclose(wfn.get_coeff((2, 0)), data[(2, 0)]))


    def test_fqecontrol_interop(self):
        """Set wavefunction data from input rather than an internal method
        """
        test = rand_wfn(4).astype(numpy.complex64)
        wfn = fqe.from_cirq(test, thresh=0.00000001)
        ref = fqe.to_cirq(wfn)
        self.assertTrue(numpy.allclose(ref, test))


    def test_hamiltonian_conversion(self):
        """Set wavefunction data from input rather than an internal method
        """
        ops1 = FermionOperator('0^ 0', 2.) + \
              FermionOperator('1^ 0', 0. + .75j) + \
              FermionOperator('0^ 1', 0. - .75j) + \
              FermionOperator('1^ 1', 2.)
        ops2 = FermionOperator('0^ 0^ 0 0', 2.) + \
              FermionOperator('1^ 1^ 0 0', 1.) + \
              FermionOperator('1^ 0^ 1 0', -.5) + \
              FermionOperator('0^ 0^ 1 1', 1.) + \
              FermionOperator('0^ 1^ 0 1', -.5) + \
              FermionOperator('1^ 1^ 1 1', 2.)
        ops = ops1 + ops2
        fqeham = fqe.get_hamiltonian_from_ops(ops, 0.0, 0.0)
        ofham = fqe.hamiltonian_to_openfermion(fqeham)
        self.assertTrue(numpy.allclose(fqeham.h1e,
                                       ofham.n_body_tensors[(1, 0)]))
        self.assertTrue(numpy.allclose(fqeham.g2e,
                                       ofham.n_body_tensors[(1, 1, 0, 0)]))
        qham = fqe.get_quadratic_hamiltonian(ops1, 0.0)
        qofham = fqe.hamiltonian_to_openfermion(qham)
        self.assertTrue(numpy.allclose(qham.h1e,
                                       qofham.n_body_tensors[(1, 0)]))


    def test_hamiltonian_openfermion_utils(self):
        """Check import from OpenFermion
        """
        ops = (FermionOperator('1^ 1', 3.) + FermionOperator('1^ 2', 3. + 4.j)
               + FermionOperator('2^ 1', 3. - 4.j)
               + FermionOperator('3^ 4', 2. + 5.j)
               + FermionOperator('4^ 3', 2. - 5.j))
        ofquad = quad_of(ops)
        ham = fqe.get_hamiltonian_from_openfermion(ofquad)
        ofham = fqe.hamiltonian_to_openfermion(ham)
        self.assertTrue(numpy.allclose(ham.h1e,
                                       ofham.n_body_tensors[(1, 0)]))


    def test_hamiltonian_two_body(self):
        """Make sure that the Hamiltonian can be initialized
        """
        symmh = [[[1, 2], 1.0, False]]
        symmg = [[[1, 2, 3, 4], 1.0, False]]
        ham = fqe.get_two_body_hamiltonian(0.0,
                                           numpy.zeros((2, 2),
                                                       dtype=numpy.complex64),
                                           numpy.ones((2, 2, 2, 2),
                                                      dtype=numpy.complex64),
                                           0.0, symmh, symmg)
        self.assertEqual(ham.identity(), 'General')
        ham.mu_c = 1.0
        self.assertEqual(ham.mu_c, 1.0)
        self.assertTrue(numpy.allclose(ham.h1e,
                                       numpy.zeros((2, 2),
                                                   dtype=numpy.complex64)))
        ham.mu_c = -1.0
        self.assertEqual(ham.mu_c, -1.0)
        self.assertTrue(numpy.allclose(ham.h_mu,
                                       numpy.identity(2,
                                                      dtype=numpy.complex64)))
        self.assertTrue(numpy.allclose(ham.g2e,
                                       numpy.ones((2, 2, 2, 2),
                                                  dtype=numpy.complex64)))


    def test_apply(self):
        """Check the wrappers for unitary wavefunction applications
        """
        ops = FermionOperator('1^ 0^', 1.0)
        test = fqe.get_wavefunction(2, 0, 2)
        newwfn = fqe.apply(ops, test)
        newwfn.set_wfn(strategy='ones')
        lena = newwfn.lena
        lenb = newwfn.lenb
        val = numpy.ones((1), dtype=numpy.complex64)
        self.assertEqual(1, lena[(4, 0)])
        self.assertEqual(1, lenb[(4, 0)])
        self.assertEqual(val, newwfn.get_coeff((4, 0)))


    def test_apply_generated_unitary(self):
        """Check the wrappers for unitary wavefunction applications
        """
        tay_wfn = fqe.get_wavefunction(2, 0, 2)
        tay_wfn.set_wfn(strategy='ones')
        tay_wfn.normalize()
        ops = FermionOperator('2^ 0', .2 + .3j) + \
              FermionOperator('0^ 2', .2 - .3j)
        test_tay = fqe.apply_generated_unitary(ops, tay_wfn, 'taylor')
        che_wfn = fqe.Wavefunction([[2, 0, 2]])
        che_wfn.set_wfn(strategy='ones')
        che_wfn.normalize()
        ops = FermionOperator('2^ 0', .2 + .3j) + \
              FermionOperator('0^ 2', .2 - .3j)
        test_che = fqe.apply_generated_unitary(ops, che_wfn, 'chebyshev')
        self.assertTrue(numpy.allclose(test_tay.get_coeff((2, 0), vec=0),
                                       test_che.get_coeff((2, 0), vec=0)))
