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

import numpy as np

from fqe.hamiltonians import general_hamiltonian


def test_general_hamiltonian():
    """Test some of the functions in General."""
    h1e = np.random.rand(5, 5).astype(np.complex128)
    h1e += h1e.T.conj()
    test = general_hamiltonian.General((h1e,))
    assert test.dim() == 5
    assert test.rank() == 2
    assert np.allclose(h1e, test.tensor(2))
    assert test.quadratic()

    trans = test.calc_diag_transform()
    h1e = trans.T.conj() @ h1e @ trans
    assert np.allclose(h1e, test.transform(trans))
    for i in range(h1e.shape[0]):
        h1e[i, i] = 0.0
    assert np.std(h1e) < 1.0e-8
    with pytest.raises(TypeError):
        general_hamiltonian.General("test")
