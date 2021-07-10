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

import unittest

import numpy
from numpy import linalg

from openfermion import FermionOperator

from fqe import wavefunction
from fqe.hamiltonians import general_hamiltonian, hamiltonian_utils
from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)

import fqe


class FqeControlTest(unittest.TestCase):
    """For Fqe structures
    """

    def test_fqe_control_dot_vdot(self):
        """Find the dot product of two wavefunctions.
        """
        wfn1 = fqe.get_number_conserving_wavefunction(4, 8)
        wfn1.set_wfn(strategy='ones')
        wfn1.normalize()
        self.assertAlmostEqual(fqe.vdot(wfn1, wfn1), 1. + .0j)
        self.assertAlmostEqual(fqe.dot(wfn1, wfn1), 1. + .0j)
        wfn1.set_wfn(strategy='random')
        wfn1.normalize()
        self.assertAlmostEqual(fqe.vdot(wfn1, wfn1), 1. + .0j)

    def test_initialize_new_wavefunctions(self):
        """APply the generated unitary transformation from the fqe namespace
        """
        nele = 3
        m_s = -1
        norb = 4
        wfn = fqe.get_wavefunction(nele, m_s, norb)
        self.assertIsInstance(wfn, wavefunction.Wavefunction)
        multiple = [[4, 0, 4], [4, 2, 4], [3, -3, 4], [1, 1, 4]]
        wfns = fqe.get_wavefunction_multiple(multiple)
        for wfn in wfns:
            with self.subTest():
                self.assertIsInstance(wfn, wavefunction.Wavefunction)

    def test_apply_generated_unitary(self):
        """APply the generated unitary transformation from the fqe namespace
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
            with self.subTest(key=key):
                diff = reference._civec[key].coeff - compute._civec[key].coeff
                err = linalg.norm(diff)
                self.assertTrue(err < 1.e-8)

    def test_cirq_interop(self):
        """Check the transition from a line quibit and back.
        """
        work = numpy.random.rand(16).astype(numpy.complex128)
        norm = numpy.sqrt(numpy.vdot(work, work))
        numpy.divide(work, norm, out=work)
        wfn = fqe.from_cirq(work, thresh=1.0e-7)
        test = fqe.to_cirq(wfn)
        self.assertTrue(numpy.allclose(test, work))

    def test_operator_constructors(self):
        """Creation of FQE-operators
        """
        self.assertIsInstance(fqe.get_s2_operator(), S2Operator)
        self.assertIsInstance(fqe.get_sz_operator(), SzOperator)
        self.assertIsInstance(fqe.get_time_reversal_operator(), TimeReversalOp)
        self.assertIsInstance(fqe.get_number_operator(), NumberOperator)
