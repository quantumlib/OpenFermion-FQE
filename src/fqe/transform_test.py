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

"""Test cases for transform
"""

import unittest

import random

from openfermion import count_qubits, FermionOperator
from openfermion.transforms import jordan_wigner
from cirq import LineQubit, Simulator
import numpy

from fqe import transform

from fqe.wavefunction import Wavefunction
from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe.cirq_utils import qubit_ops_to_circuit
from fqe.openfermion_utils import ladder_op
from fqe.util import qubit_config_sector


class TransformTest(unittest.TestCase):
    """Unit tests
    """


    def test_cirq_to_fqe_error(self):
        """Wrap together all the routines to build a wavefuntion that can be
        read by the fqe by simply passing a cirq wavefunction.
        """
        cirq_wfn = numpy.ones((2, 1), dtype=numpy.complex64)
        self.assertRaises(ValueError, transform.cirq_to_fqe_single, cirq_wfn,
                          20, 1, None)

    @unittest.SkipTest
    def test_cirq_to_fqe_single(self):
        """Wrap together all the routines to build a wavefuntion that can be
        read by the fqe by simply passing a cirq wavefunction.
        """
        cof = numpy.array([0.3901112 - 0.1543j, 0.01213 + 0.79120j],
                          dtype=numpy.complex64)
        cof /= numpy.sqrt(numpy.vdot(cof, cof))
        wfn_ops = cof[0]*(ladder_op(0, 1)*ladder_op(1, 1)*ladder_op(2, 1))
        wfn_ops += cof[1]*(ladder_op(0, 1)*ladder_op(2, 1)*ladder_op(3, 1))
        qpu = LineQubit.range(count_qubits(wfn_ops))
        cirq_wfn = qubit_wavefunction_from_vacuum(wfn_ops, qpu)
        fqe_wfn = transform.cirq_to_fqe_single(cirq_wfn, 3, 1, None)
        fqe_jw = jordan_wigner(fqe_wfn)
        test_key = list(wfn_ops.terms.keys())
        for keyval in list(fqe_jw.terms.keys()):
            self.assertTrue(keyval in test_key)

        for i in fqe_jw.terms:
            self.assertAlmostEqual(fqe_jw.terms[i], wfn_ops.terms[i])


    @unittest.SkipTest
    def test_from_cirq(self):
        """Perform the transformation from a cirq wavefunction for all particle
        number sectors.  Once this is done, apply an operator to the FQE
        wavefunction and the same operator to the cirq wavefunction.
        Then generate the new FQE wavefunction from the moditfied cirq wavefunction
        compare the results.
        """
        nqubits = 4
        state = numpy.zeros((2**nqubits), dtype=numpy.complex64)
        wfn = Wavefunction([[0, 0, 2], [1, 1, 2], [1, -1, 2], [2, 0, 2]])

        def random_cplx():
            """A small helper function to generate random complex numbers.
            """
            return -1. - 1.j + 2.*(random.random()*1. + random.random()*1.j)

        configs = [[0, 0], [1, 1], [1, -1], [2, 0]]
        for key in configs:
            basis = qubit_config_sector(nqubits, key[0], key[1])
            for vec in basis:
                state += vec*random_cplx()

        norm = numpy.vdot(state, state)
        state /= numpy.sqrt(norm)
        transform.from_cirq(wfn, state)
        testop = FermionOperator('1', 1.0)
        newwfn = wfn.apply(testop)
        newwfn.normalize(vec=[0])

        qubits = LineQubit.range(nqubits)
        ladder = jordan_wigner(testop)
        qpu = Simulator()
        test = numpy.zeros((2**nqubits), dtype=numpy.complex64)
        for ops in ladder.terms:
            circuit = qubit_ops_to_circuit(ops, qubits)
            work_state = state.copy()
            result = qpu.simulate(circuit, qubit_order=qubits,
                                  initial_state=work_state)
            test += result.final_state*ladder.terms[ops]
        norm = numpy.vdot(test, test)
        test /= numpy.sqrt(norm)
        wfn = Wavefunction([[0, 0, 2], [1, 1, 2]])
        transform.from_cirq(wfn, test)
        wfn.normalize(vec=[0])

        self.assertTrue(sorted(list(newwfn.configs)) == sorted(list(wfn.configs)))
        for key in wfn.configs:
            self.assertTrue(numpy.allclose(newwfn.get_coeff(key),
                                           wfn.get_coeff(key)))


if __name__ == '__main__':
    unittest.main()
