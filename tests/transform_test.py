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

import numpy
import pytest

from openfermion import count_qubits
from openfermion.transforms import jordan_wigner
from cirq import LineQubit

from fqe import transform

from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe.openfermion_utils import ladder_op
import fqe.settings


def test_cirq_to_fqe_error():
    """Check if cirq_to_fqe_single raises an error
    """
    cirq_wfn = numpy.ones((2, 1), dtype=numpy.complex128)
    with pytest.raises(ValueError):
        transform.cirq_to_fqe_single(cirq_wfn, 20, 1, None)


def test_cirq_to_fqe_single():
    """Wrap together all the routines to build a wavefuntion that can be
    read by the fqe by simply passing a cirq wavefunction.
    """
    cof = numpy.array([0.3901112 - 0.1543j, 0.01213 + 0.79120j],
                      dtype=numpy.complex128)
    cof /= numpy.sqrt(numpy.vdot(cof, cof))
    wfn_ops = cof[0] * (ladder_op(0, 1) * ladder_op(1, 1) * ladder_op(2, 1))
    wfn_ops += cof[1] * (ladder_op(0, 1) * ladder_op(2, 1) * ladder_op(3, 1))
    qpu = LineQubit.range(count_qubits(wfn_ops))
    cirq_wfn = qubit_wavefunction_from_vacuum(wfn_ops, qpu)
    fqe_wfn = transform.cirq_to_fqe_single(cirq_wfn, 3, 1, None)
    fqe_jw = jordan_wigner(fqe_wfn)
    test_key = list(wfn_ops.terms.keys())
    for keyval in list(fqe_jw.terms.keys()):
        assert keyval in test_key

    for i in fqe_jw.terms:
        assert round(abs(fqe_jw.terms[i] - wfn_ops.terms[i]), 7) == 0


def test_from_cirq(c_or_python):
    """Check the transition from a line qubit and back.
       WARNING This is a duplicate of half of the test_cirq_interop() test
       in _fqe_control_test.py, but it is required for testing from_cirq.
    """
    fqe.settings.use_accelerated_code = c_or_python
    work = numpy.random.rand(16).astype(numpy.complex128)
    norm = numpy.sqrt(numpy.vdot(work, work))
    numpy.divide(work, norm, out=work)
    wfn = fqe.from_cirq(work, thresh=1.0e-7)
    test = fqe.to_cirq(wfn)
    assert numpy.allclose(test, work)
