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

import numpy as np

import openfermion as of
from fqe.hamiltonians.hamiltonian_utils import gather_nbody_spin_sectors


def test_nbody_spin_sectors():
    op = of.FermionOperator(((3, 1), (4, 1), (2, 0), (0, 0)),
                            coefficient=1.0 + 0.5j)
    # op += of.hermitian_conjugated(op)
    (
        coefficient,
        parity,
        alpha_sub_ops,
        beta_sub_ops,
    ) = gather_nbody_spin_sectors(op)
    assert np.isclose(coefficient.real, 1.0)
    assert np.isclose(coefficient.imag, 0.5)
    assert np.isclose(parity, -1)
    assert tuple(map(tuple, alpha_sub_ops)) == ((4, 1), (2, 0), (0, 0))
    assert tuple(map(tuple, beta_sub_ops)) == ((3, 1),)


# TODO: Add tests for antisymm_two_body, antisymm_three_body,
#  antisymm_four_body, and nbody_matrix. (All defined in hamiltonian_utils.py.
