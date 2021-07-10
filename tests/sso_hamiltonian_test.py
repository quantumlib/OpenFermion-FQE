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

import numpy as np

from fqe.hamiltonians import sso_hamiltonian


def test_sso():
    """Test some of the functions in SSOHamiltonian."""
    h1e = np.random.rand(5, 5).astype(np.complex128)
    test = sso_hamiltonian.SSOHamiltonian((h1e,))
    assert test.dim() == 5
    assert test.rank() == 2
    assert np.allclose(h1e, test.tensor(2))
    with pytest.raises(TypeError):
        sso_hamiltonian.SSOHamiltonian("test")
