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

from fqe.util import rand_wfn
import fqe

class FqeControlTest(unittest.TestCase):
    """For Fqe structures
    """


    def test_fqe_control_dot_vdot(self):
        """Find the dot product of two wavefunctions.
        """
        wfn1 = fqe.get_spin_nonconserving_wavefunction(4, 8)
        wfn1.set_wfn(strategy='ones')
        wfn1.normalize()
        self.assertAlmostEqual(fqe.vdot(wfn1, wfn1), 1. + .0j)
        self.assertAlmostEqual(fqe.dot(wfn1, wfn1), 1. + .0j)
        wfn1.set_wfn(strategy='random')
        wfn1.normalize()
        self.assertAlmostEqual(fqe.vdot(wfn1, wfn1), 1. + .0j)


    @unittest.SkipTest
    def test_fqecontrol_interop(self):
        """Set wavefunction data from input rather than an internal method
        """
        test = rand_wfn(4, 4).astype(numpy.complex128)
        test = numpy.reshape(test, (16))
        wfn = fqe.from_cirq(test, thresh=0.00000001)
        ref = fqe.to_cirq(wfn)
        self.assertTrue(numpy.allclose(ref, test))


    @unittest.SkipTest
    def test_hamiltonian_two_body(self):
        """Make sure that the Hamiltonian can be initialized
        """
        symmh = [[[1, 2], 1.0, False]]
        symmg = [[[1, 2, 3, 4], 1.0, False]]
        ham = fqe.get_two_body_hamiltonian(0.0,
                                           numpy.zeros((2, 2),
                                                       dtype=numpy.complex128),
                                           numpy.ones((2, 2, 2, 2),
                                                      dtype=numpy.complex128),
                                           0.0, symmh, symmg)
        self.assertEqual(ham.identity(), 'General')
        ham.mu_c = 1.0
        self.assertEqual(ham.mu_c, 1.0)
        self.assertTrue(numpy.allclose(ham.h1e,
                                       numpy.zeros((2, 2),
                                                   dtype=numpy.complex128)))
        ham.mu_c = -1.0
        self.assertEqual(ham.mu_c, -1.0)
        self.assertTrue(numpy.allclose(ham.h_mu,
                                       numpy.identity(2,
                                                      dtype=numpy.complex128)))
        self.assertTrue(numpy.allclose(ham.g2e,
                                       numpy.ones((2, 2, 2, 2),
                                                  dtype=numpy.complex128)))


if __name__ == '__main__':
    unittest.main()
