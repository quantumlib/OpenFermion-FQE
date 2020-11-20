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
