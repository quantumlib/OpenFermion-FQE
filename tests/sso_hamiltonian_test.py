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
"""Unit tests for the SSOHamiltonian class."""

import pytest

import numpy

from fqe.hamiltonians import sso_hamiltonian


def test_sso():
    """Test some of the functions in SSOHamiltonian."""
    norb = 4
    h1e = numpy.random.rand(norb, norb).astype(numpy.complex128)
    h1e[:2, 2:] = 0.0
    h1e[2:, :2] = 0.0
    h1e += h1e.T.conj()
    test = sso_hamiltonian.SSOHamiltonian((h1e,))
    assert test.dim() == norb
    assert test.rank() == 2
    assert numpy.allclose(h1e, test.tensor(2))
    assert test.quadratic()
    with pytest.raises(TypeError):
        sso_hamiltonian.SSOHamiltonian("test")
    with pytest.raises(ValueError):
        sso_hamiltonian.SSOHamiltonian((numpy.zeros((2, 2, 2)),))

    tensors = test.tensors()
    assert numpy.allclose(tensors[0], h1e)

    trans = test.calc_diag_transform()
    diag = trans.conj().T @ h1e @ trans
    diag2 = test.transform(trans)
    assert numpy.allclose(diag, diag2)
    numpy.fill_diagonal(diag, 0.0)
    assert numpy.allclose(diag, 0.0)

    time = 3.4
    iht = test.iht(time)
    assert numpy.allclose(iht, h1e * (-1j * time))

    h2e = numpy.random.rand(norb, norb, norb, norb).astype(numpy.complex128)
    test2 = sso_hamiltonian.SSOHamiltonian((h1e, h2e))
    assert not test2.quadratic()


def test_equality():
    """ Test the equality operator """
    h1e = numpy.random.rand(5, 5).astype(numpy.complex128)
    e_0 = -4.2
    test = sso_hamiltonian.SSOHamiltonian((h1e,))
    test2 = sso_hamiltonian.SSOHamiltonian((h1e,))
    assert test == test2
    assert not (test == 1)

    h1e2 = numpy.random.rand(5, 5).astype(numpy.complex128)

    test2 = sso_hamiltonian.SSOHamiltonian((h1e2,))
    assert test != test2
