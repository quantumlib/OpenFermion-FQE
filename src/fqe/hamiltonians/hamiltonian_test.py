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
"""Unit tests for the base Hamiltonian class."""

import pytest
from typing import Tuple

import numpy as np

from openfermion import FermionOperator

from fqe.hamiltonians import hamiltonian
from fqe.hamiltonians import diagonal_coulomb
from fqe.hamiltonians import diagonal_hamiltonian
from fqe.hamiltonians import gso_hamiltonian
from fqe.hamiltonians import sso_hamiltonian
from fqe.hamiltonians import restricted_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import general_hamiltonian


def test_general():
    """The base Hamiltonian initializes some common variables."""
    class Test(hamiltonian.Hamiltonian):
        """A testing dummy class."""

        def __init__(self):
            super().__init__()

        def dim(self) -> int:
            return super().dim()

        def calc_diag_transform(self) -> np.ndarray:
            return super().calc_diag_transform()

        def rank(self) -> int:
            return super().rank()

        def iht(self, time: float) -> Tuple[np.ndarray, ...]:
            return super().iht(time)

        def tensors(self) -> Tuple[np.ndarray, ...]:
            return super().tensors()

        def diag_values(self) -> np.ndarray:
            return super().diag_values()

        def transform(self, trans: np.ndarray) -> np.ndarray:
            return super().transform(trans)

    test = Test()
    assert test.dim() == 0
    assert test.rank() == 0
    assert np.isclose(test.e_0(), 0.0 + 0.0j)
    assert test.tensors() == tuple()
    assert test.iht(0.0) == tuple()
    assert test.diag_values().shape == (0,)
    assert test.calc_diag_transform().shape == (0,)
    assert test.transform(np.empty(0)).shape == (0,)
    assert not test.quadratic()
    assert not test.diagonal()
    assert not test.diagonal_coulomb()
    assert test.conserve_number()


def test_diagonal_coulomb():
    """Test some of the functions in DiagonalCoulomb."""
    diag = np.zeros((5, 5), dtype=np.complex128)
    test = diagonal_coulomb.DiagonalCoulomb(diag)
    assert test.dim() == 5
    assert test.rank() == 4


def test_diagonal():
    """Test some of the functions in Diagonal."""
    bad_diag = np.zeros((5, 5), dtype=np.complex128)
    with pytest.raises(ValueError):
        diagonal_hamiltonian.Diagonal(bad_diag)
    diag = np.zeros((5, ), dtype=np.complex128)
    test = diagonal_hamiltonian.Diagonal(diag)
    assert test.dim() == 5
    assert test.rank() == 2


def test_gso():
    """Test some of the functions in GSOHamiltonian."""
    h1e = np.random.rand(5, 5).astype(np.complex128)
    test = gso_hamiltonian.GSOHamiltonian((h1e,))
    assert test.dim() == 5
    assert test.rank() == 2
    assert np.allclose(h1e, test.tensor(2))
    with pytest.raises(TypeError):
        gso_hamiltonian.GSOHamiltonian("test")


def test_sso():
    """Test some of the functions in SSOHamiltonian."""
    h1e = np.random.rand(5, 5).astype(np.complex128)
    test = sso_hamiltonian.SSOHamiltonian((h1e,))
    assert test.dim() == 5
    assert test.rank() == 2
    assert np.allclose(h1e, test.tensor(2))
    with pytest.raises(TypeError):
        sso_hamiltonian.SSOHamiltonian("test")


def test_restricted():
    """Test some of the functions in SSOHamiltonian."""
    h1e = np.random.rand(5, 5).astype(np.complex128)
    test = restricted_hamiltonian.RestrictedHamiltonian((h1e,))
    assert test.dim() == 5
    assert test.rank() == 2
    assert np.allclose(h1e, test.tensor(2))
    with pytest.raises(TypeError):
        restricted_hamiltonian.RestrictedHamiltonian("test")


def test_general_hamiltonian():
    """Test some of the functions in General."""
    h1e = np.random.rand(5, 5).astype(np.complex128)
    h1e += h1e.T.conj()
    test = general_hamiltonian.General((h1e, ))
    assert test.dim() == 5
    assert test.rank() == 2
    assert np.allclose(h1e, test.tensor(2))

    trans = test.calc_diag_transform()
    h1e = trans.T.conj() @ h1e @ trans
    assert np.allclose(h1e, test.transform(trans))
    for i in range(h1e.shape[0]):
        h1e[i, i] = 0.0
    assert np.std(h1e) < 1.0e-8
    with pytest.raises(TypeError):
        general_hamiltonian.General("test")


def test_sparse():
    """Test some of the functions in SparseHamiltonian."""
    oper = FermionOperator('0 0^')
    oper += FermionOperator('1 1^')
    test = sparse_hamiltonian.SparseHamiltonian(oper)
    assert test.rank() == 2
    test = sparse_hamiltonian.SparseHamiltonian(oper, False)
    assert not test.is_individual()
