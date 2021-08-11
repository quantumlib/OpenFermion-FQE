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
"""Unit tests for the SparseHamiltonian class."""

import pytest

from openfermion import FermionOperator

from fqe.hamiltonians import sparse_hamiltonian


def test_sparse():
    """Test some of the functions in SparseHamiltonian."""
    oper = FermionOperator('0 0^')
    oper += FermionOperator('1 1^')
    test = sparse_hamiltonian.SparseHamiltonian(oper)
    assert test.rank() == 2
    test = sparse_hamiltonian.SparseHamiltonian(oper, False)
    assert not test.is_individual()

    terms = test.terms()
    assert terms == [(-1.0, [(0, 1), (0, 0)], []), (-1.0, [], [(0, 1), (0, 0)])]

    ham = test.terms_hamiltonian()
    assert len(ham) == 2
    assert isinstance(ham[0], sparse_hamiltonian.SparseHamiltonian)
    assert ham[1].terms()[0] == (-1.0, [], [(0, 1), (0, 0)])


def test_sparse_from_string():
    test = sparse_hamiltonian.SparseHamiltonian('1^ 0')
    assert test.rank() == 2
    assert test.is_individual()
    assert test.nterms() == 1

    time = 3.1
    iht = test.iht(time)
    terms = test.terms()[0]
    iterms = iht.terms()[0]
    assert terms[1:] == iterms[1:]
    assert abs(terms[0] * (-1j) * time - iterms[0]) < 1.0e-8


def test_dim_error():
    """Test if SparseHamiltonian raises an error if dim() is accessed
    """
    oper = FermionOperator('0 0^')
    test = sparse_hamiltonian.SparseHamiltonian(oper)
    with pytest.raises(NotImplementedError):
        d = test.dim()


def test_equality():
    """ Test the equality operator """
    oper = FermionOperator('0 0^')
    test = sparse_hamiltonian.SparseHamiltonian(oper)
    test2 = sparse_hamiltonian.SparseHamiltonian(oper)

    assert test == test2
    assert not (test == 1)

    oper2 = FermionOperator('1 1^')
    test2 = sparse_hamiltonian.SparseHamiltonian(oper2)

    assert test != test2
