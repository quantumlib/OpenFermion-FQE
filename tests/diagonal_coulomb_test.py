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
"""Unit tests for the DiagonalCoulombHamiltonian class."""

import numpy

from fqe.hamiltonians import diagonal_coulomb


def test_diagonal_coulomb():
    """Test some of the functions in DiagonalCoulomb."""
    norb = 5
    diag = numpy.zeros((norb, norb), dtype=numpy.complex128)
    test = diagonal_coulomb.DiagonalCoulomb(diag)
    assert test.dim() == norb
    assert test.rank() == 4
    assert test.diagonal_coulomb()

    time = 1.8
    iht = test.iht(time)
    assert numpy.allclose(iht[0], diag * (-1j) * time)

    diag2 = numpy.empty((norb, norb, norb, norb), dtype=numpy.complex128)
    for i in range(norb):
        for j in range(norb):
            diag2[i, j, i, j] = 1.0
    test2 = diagonal_coulomb.DiagonalCoulomb(diag2)
    assert test.dim() == norb

    assert numpy.allclose(test2._tensor[1], 1.0)


def test_equality():
    """ Test the equality operator """
    diag = numpy.zeros((5, 5), dtype=numpy.complex128)
    e_0 = -4.2
    test = diagonal_coulomb.DiagonalCoulomb(diag, e_0)
    test2 = diagonal_coulomb.DiagonalCoulomb(diag, e_0)
    assert test == test2
    assert not (test == 1)

    diag2 = numpy.zeros((5, 5), dtype=numpy.complex128)
    diag2[0, 1] = 1.0
    test2 = diagonal_coulomb.DiagonalCoulomb(diag2, e_0)
    assert test != test2
