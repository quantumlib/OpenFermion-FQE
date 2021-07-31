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
"""Unit tests for Hamiltonian constructor routines."""

import pytest
import numpy

import openfermion as of
from fqe.hamiltonians.hamiltonian_utils import antisymm_two_body 
from fqe.hamiltonians.hamiltonian_utils import antisymm_three_body
from fqe.hamiltonians.hamiltonian_utils import antisymm_four_body
from fqe.hamiltonians.hamiltonian_utils import nbody_matrix
from fqe.hamiltonians.hamiltonian_utils import gather_nbody_spin_sectors

def test_antisymm_two_body():
    numpy.random.seed(seed=409)
    norb = 4
    two = numpy.random.rand(norb, norb, norb, norb) \
         + 1.j * numpy.random.rand(norb, norb, norb, norb)
    two = antisymm_two_body(two)
    assert numpy.isclose(two[1, 2, 3, 2], -two[2, 1, 3, 2])
    assert numpy.isclose(two[1, 2, 3, 2],  two[2, 1, 2, 3])
    assert numpy.isclose(two[1, 2, 3, 2], -two[1, 2, 2, 3])

def test_antisymm_three_body():
    numpy.random.seed(seed=409)
    norb = 4
    three = numpy.random.rand(norb, norb, norb, norb, norb, norb) \
         + 1.j * numpy.random.rand(norb, norb, norb, norb, norb, norb)
    three = antisymm_three_body(three)
    assert numpy.isclose(three[0, 1, 2, 3, 0, 2], -three[0, 2, 1, 3, 0, 2])
    assert numpy.isclose(three[0, 1, 2, 3, 0, 2],  three[0, 2, 1, 3, 2, 0])
    assert numpy.isclose(three[0, 1, 2, 3, 0, 2], -three[0, 2, 1, 2, 3, 0])

def test_antisymm_four_body():
    numpy.random.seed(seed=409)
    norb = 4
    four = numpy.random.rand(norb, norb, norb, norb, norb, norb, norb, norb) \
         + 1.j * numpy.random.rand(norb, norb, norb, norb, norb, norb, norb, norb)
    four = antisymm_four_body(four)
    assert numpy.isclose(four[0, 1, 2, 3, 0, 2, 3, 1], -four[0, 1, 2, 3, 1, 2, 3, 0])

def test_nbody_spin_sectors():
    op = of.FermionOperator(((3, 1), (4, 1), (2, 0), (1, 0)),
                            coefficient=1.0 + 0.5j)
    (
        coefficient,
        parity,
        alpha_sub_ops,
        beta_sub_ops,
    ) = gather_nbody_spin_sectors(op)
    assert numpy.isclose(coefficient.real, 1.0)
    assert numpy.isclose(coefficient.imag, 0.5)
    assert numpy.isclose(parity, 1)
    assert tuple(map(tuple, alpha_sub_ops)) == ((4, 1), (2, 0))
    assert tuple(map(tuple, beta_sub_ops)) == ((3, 1), (1, 0))

def test_nbody_matrix():
    term = of.FermionOperator('1^ 0') + of.FermionOperator('0^ 2', 1.2j)
    mat = nbody_matrix(term, norb=3)
    ref = numpy.array([[0.+0.j,  0.+1.2j, 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]])
    assert numpy.allclose(mat, ref)

    with pytest.raises(ValueError):
        nbody_matrix(term + of.FermionOperator('0^ 1^ 3 2'), norb=3)


    empty = nbody_matrix(of.FermionOperator(), norb=3)
    assert numpy.array_equal(empty, numpy.empty(0))
