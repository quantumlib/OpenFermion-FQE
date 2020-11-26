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
"""Unit tests for _fqe_control.py."""

# pylint: disable=protected-access

import cmath

import numpy as np

from openfermion import FermionOperator

import fqe
from fqe.hamiltonians import general_hamiltonian, hamiltonian_utils
from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)


def test_fqe_control_dot_vdot():
    """Tests the dot product of two wavefunctions."""
    wfn1 = fqe.get_number_conserving_wavefunction(4, 8)
    wfn1.set_wfn(strategy="ones")
    wfn1.normalize()

    assert cmath.isclose(fqe.vdot(wfn1, wfn1), 1.0 + 0.0j)

    wfn1.set_wfn(strategy="random")
    wfn1.normalize()
    assert cmath.isclose(fqe.vdot(wfn1, wfn1), 1.0 + 0.0j)


def test_initialize_new_wavefunctions():
    """Tests getting wavefunctions."""
    nele = 3
    m_s = -1
    norb = 4
    wfn = fqe.get_wavefunction(nele, m_s, norb)
    assert isinstance(wfn, fqe.wavefunction.Wavefunction)

    multiple = [[4, 0, 4], [4, 2, 4], [3, -3, 4], [1, 1, 4]]
    wfns = fqe.get_wavefunction_multiple(multiple)
    for wfn in wfns:
        assert isinstance(wfn, fqe.wavefunction.Wavefunction)


def test_apply_generated_unitary():
    """Tests applying generated unitary transformations."""
    norb = 4
    nele = 3
    time = 0.001
    ops = FermionOperator("1^ 3^ 5 0", 2.0 - 2.0j) + FermionOperator(
        "0^ 5^ 3 1", 2.0 + 2.0j
    )

    wfn = fqe.get_number_conserving_wavefunction(nele, norb)
    wfn.set_wfn(strategy="random")
    wfn.normalize()

    reference = fqe.apply_generated_unitary(wfn, time, "taylor", ops)

    h1e = np.zeros((2 * norb, 2 * norb), dtype=np.complex128)
    h2e = hamiltonian_utils.nbody_matrix(ops, norb)
    h2e = hamiltonian_utils.antisymm_two_body(h2e)
    hamil = general_hamiltonian.General(tuple([h1e, h2e]))
    compute = wfn.apply_generated_unitary(time, "taylor", hamil)

    for key in wfn.sectors():
        diff = reference._civec[key].coeff - compute._civec[key].coeff
        err = np.linalg.norm(diff)
        assert err < 1.0e-8


def test_cirq_interop():
    """Tests converting wavefunctions."""
    state = np.random.rand(16).astype(np.complex128)
    norm = np.sqrt(np.vdot(state, state))
    np.divide(state, norm, out=state)

    wfn = fqe.from_cirq(state, thresh=1.0e-7)
    converted_state = fqe.to_cirq(wfn)
    assert np.allclose(converted_state, state)


def test_operator_constructors():
    """Tests creating FQE operators."""
    assert isinstance(fqe.get_s2_operator(), S2Operator)
    assert isinstance(fqe.get_sz_operator(), SzOperator)
    assert isinstance(fqe.get_time_reversal_operator(), TimeReversalOp)
    assert isinstance(fqe.get_number_operator(), NumberOperator)

# TODO: Add more tests.
