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
"""Test cases for transform
"""

import unittest

from openfermion import count_qubits
from openfermion.transforms import jordan_wigner
from cirq import LineQubit
import numpy

from fqe import transform

from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe.openfermion_utils import ladder_op


class TransformTest(unittest.TestCase):
    """Unit tests
    """

    def test_cirq_to_fqe_error(self):
        """Wrap together all the routines to build a wavefuntion that can be
        read by the fqe by simply passing a cirq wavefunction.
        """
        cirq_wfn = numpy.ones((2, 1), dtype=numpy.complex128)
        self.assertRaises(ValueError, transform.cirq_to_fqe_single, cirq_wfn,
                          20, 1, None)

    def test_cirq_to_fqe_single(self):
        """Wrap together all the routines to build a wavefuntion that can be
        read by the fqe by simply passing a cirq wavefunction.
        """
        cof = numpy.array([0.3901112 - 0.1543j, 0.01213 + 0.79120j],
                          dtype=numpy.complex128)
        cof /= numpy.sqrt(numpy.vdot(cof, cof))
        wfn_ops = cof[0] * (ladder_op(0, 1) * ladder_op(1, 1) * ladder_op(2, 1))
        wfn_ops += cof[1] * (ladder_op(0, 1) * ladder_op(2, 1) *
                             ladder_op(3, 1))
        qpu = LineQubit.range(count_qubits(wfn_ops))
        cirq_wfn = qubit_wavefunction_from_vacuum(wfn_ops, qpu)
        fqe_wfn = transform.cirq_to_fqe_single(cirq_wfn, 3, 1, None)
        fqe_jw = jordan_wigner(fqe_wfn)
        test_key = list(wfn_ops.terms.keys())
        for keyval in list(fqe_jw.terms.keys()):
            self.assertTrue(keyval in test_key)

        for i in fqe_jw.terms:
            self.assertAlmostEqual(fqe_jw.terms[i], wfn_ops.terms[i])
