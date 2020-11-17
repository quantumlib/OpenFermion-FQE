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

import numpy as np

from fqe.hamiltonians import diagonal_coulomb


def test_diagonal_coulomb():
    """Test some of the functions in DiagonalCoulomb."""
    diag = np.zeros((5, 5), dtype=np.complex128)
    test = diagonal_coulomb.DiagonalCoulomb(diag)
    assert test.dim() == 5
    assert test.rank() == 4
