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

from typing import Tuple

import numpy as np

from fqe.hamiltonians import hamiltonian


def test_base_hamiltonian():
    """Tests the base Hamiltonian class by subclassing it."""

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
