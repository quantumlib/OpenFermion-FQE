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
"""cirq_utils unit tests.
"""

import cirq
import numpy
import pytest

from fqe import cirq_utils
from openfermion import QubitOperator


def test_pauli_x_error():
    """Confirm that gate is mutating the output
    """
    qpu = cirq.Simulator()
    qubit = cirq.LineQubit.range(1)
    eigenstate = numpy.ones(2, dtype=numpy.complex64)
    eigenstate[1] = 0.0 - 1.0j
    eigenstate /= numpy.sqrt(numpy.vdot(eigenstate, eigenstate))
    gates = [cirq_utils.qubit_op_to_gate('X', qubit[0])]
    circuit = cirq.Circuit([cirq.Moment(gates)])
    result = qpu.simulate(circuit, qubit_order=qubit, initial_state=eigenstate)
    assert result.final_state_vector[0] != eigenstate[0]
    assert result.final_state_vector[1] != eigenstate[1]


def test_pauli_x():
    """Confirm that the Pauli X gate is being properly created
    """
    qpu = cirq.Simulator()
    qubit = cirq.LineQubit.range(1)
    eigenstate = numpy.ones(2, dtype=numpy.complex64)
    eigenstate /= numpy.sqrt(numpy.vdot(eigenstate, eigenstate))
    gates = [cirq_utils.qubit_op_to_gate('X', qubit[0])]
    circuit = cirq.Circuit([cirq.Moment(gates)])
    result = qpu.simulate(circuit, qubit_order=qubit, initial_state=eigenstate)
    assert list(result.final_state_vector) == list(eigenstate)


def test_pauli_y():
    """Confirm that the Pauli Y gate is being properly created
    """
    qpu = cirq.Simulator()
    qubit = cirq.LineQubit.range(1)
    eigenstate = numpy.ones(2, dtype=numpy.complex64)
    eigenstate[1] = 0.0 + 1.0j
    eigenstate /= numpy.sqrt(numpy.vdot(eigenstate, eigenstate))
    gates = [cirq_utils.qubit_op_to_gate('Y', qubit[0])]
    circuit = cirq.Circuit([cirq.Moment(gates)])
    result = qpu.simulate(circuit, qubit_order=qubit, initial_state=eigenstate)
    assert list(result.final_state_vector) == list(eigenstate)


def test_pauli_z():
    """Confirm that the Pauli Z gate is being properly created
    """
    qpu = cirq.Simulator()
    qubit = cirq.LineQubit.range(1)
    eigenstate = numpy.ones(2, dtype=numpy.complex64)
    eigenstate[1] = 0.0 + 0.0j
    eigenstate /= numpy.sqrt(numpy.vdot(eigenstate, eigenstate))
    _gates = [cirq_utils.qubit_op_to_gate('Z', qubit[0])]
    circuit = cirq.Circuit([cirq.Moment(_gates)])
    result = qpu.simulate(circuit, qubit_order=qubit, initial_state=eigenstate)
    assert list(result.final_state_vector) == list(eigenstate)


def test_build_ops_error():
    """Circuits with incorrect arguments should raise and error.
    """
    qubit = cirq.LineQubit.range(1)
    with pytest.raises(ValueError):
        cirq_utils.qubit_op_to_gate('W', qubit[0])


def test_build_circuit_product():
    """Qubit operations which are products of operators should be compiled
    into a single circuit.
    """
    qpu = cirq.Simulator(dtype=numpy.complex128)
    qubits = cirq.LineQubit.range(4)
    ops = QubitOperator('', 1.0)
    for i in range(4):
        ops *= QubitOperator('X' + str(i), 1.0)
    for j in ops.terms:
        circuit = cirq_utils.qubit_ops_to_circuit(j, qubits)
    init_state = numpy.zeros(2**4, dtype=numpy.complex128)
    init_state[0] = 1.0 + 0.0j
    result = qpu.simulate(circuit, qubit_order=qubits, initial_state=init_state)
    final_state = numpy.zeros(2**4, dtype=numpy.complex128)
    final_state[-1] = 1.0 + 0.0j
    assert list(result.final_state_vector) == list(final_state)


def test_single_mode_projection():
    """Find the coeffcient of a wavefunction generated from a single qubit.
    """
    n_qubits = 1
    qubits = cirq.LineQubit.range(n_qubits)
    ops = QubitOperator('X0', 1.0)
    init_state = numpy.zeros(2**n_qubits, dtype=numpy.complex128)
    init_state[1] = 1.0 + 0.0j
    cof = numpy.zeros(n_qubits, dtype=numpy.complex128)
    cirq_utils.qubit_projection(ops, qubits, init_state, cof)
    assert cof[0] == 1.0 + 0.0j


def test_x_y_z_mode_projection():
    """Find the projection of a wavefunction generated from a linear
    combination of qubits.

    python2 and python3 iterate through Qubit operators differently.
    Consequently, the coeffcients must be set based on that iteration at
    test time.
    """
    n_qubits = 2
    qubits = cirq.LineQubit.range(n_qubits)
    test_wfn = numpy.array(
        [0.92377985 + 0.j, 0. - 0.20947377j, 0.32054904 + 0.j, 0. + 0.j],
        dtype=numpy.complex128)
    assert round(abs(numpy.vdot(test_wfn, test_wfn) - 1.0 + 0.0j), 6) == 0
    ops = QubitOperator('X0', 1.0) + QubitOperator('Y1', 1.0) \
        + QubitOperator('Z0', 1.0)

    test_cof = numpy.zeros((3, 1), dtype=numpy.complex128)
    for indx, cluster in enumerate(ops.terms):
        if cluster[0][1] == 'X':
            test_cof[indx] = 0.32054904
        if cluster[0][1] == 'Y':
            test_cof[indx] = -0.20947377
        if cluster[0][1] == 'Z':
            test_cof[indx] = 0.92377985

    cof = numpy.zeros((3, 1), dtype=numpy.complex128)
    cirq_utils.qubit_projection(ops, qubits, test_wfn, cof)
    assert numpy.allclose(cof, test_cof)


def test_qubit_wavefunction_from_vacuum():
    """Build a wavefunction given a group of qubit operations.
    """
    test_val = 1.0 + 2.0 + 3.0 + 5.0 + 7.0 + 11.0 + 13.0 + 0.j
    n_qubits = 1
    qubits = cirq.LineQubit.range(n_qubits)
    ops = QubitOperator('X0', 1.0) + QubitOperator('X0', 2.0) \
        + QubitOperator('X0', 3.0) + QubitOperator('X0', 5.0) \
        + QubitOperator('X0', 7.0) + QubitOperator('X0', 11.0) \
        + QubitOperator('X0', 13.0)
    state = cirq_utils.qubit_wavefunction_from_vacuum(ops, qubits)
    assert state[1] == test_val
