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
"""Unit tests for the GeneralHamiltonian class."""

import pytest

import numpy

from fqe.hamiltonians import general_hamiltonian


def test_general_hamiltonian():
    """Test some of the functions in General."""
    h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
    h1e += h1e.T.conj()
    test = general_hamiltonian.General((h1e,))
    assert test.dim() == 5
    assert test.rank() == 2
    assert numpy.allclose(h1e, test.tensor(2))
    assert test.quadratic()

    tensors = test.tensors()
    assert numpy.allclose(h1e, tensors[0])

    time = 3.1
    iht = test.iht(time)
    assert numpy.allclose(h1e * (-1j) * time, iht[0])

    trans = test.calc_diag_transform()
    h1e = trans.T.conj() @ h1e @ trans
    assert numpy.allclose(h1e, test.transform(trans))
    for i in range(h1e.shape[0]):
        h1e[i, i] = 0.0
    assert numpy.std(h1e) < 1.0e-8
    with pytest.raises(TypeError):
        general_hamiltonian.General("test")
    with pytest.raises(ValueError):
        general_hamiltonian.General((numpy.zeros((2, 2, 2)),))


def test_equality():
    """ Test the equality operator """
    h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
    e_0 = -4.2
    test = general_hamiltonian.General((h1e,))
    test2 = general_hamiltonian.General((h1e,))
    assert test == test2
    assert not (test == 1)

    h1e2 = numpy.random.rand(5, 5).astype(numpy.complex128)

    test2 = general_hamiltonian.General((h1e2,))
    assert test != test2
