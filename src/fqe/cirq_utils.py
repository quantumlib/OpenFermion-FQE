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
"""Utilities which specifically require import from Cirq
"""
#Type checking splits the imports
#pylint: disable=ungrouped-imports

from typing import List, TYPE_CHECKING

import numpy

from cirq import X, Y, Z, Moment, Circuit, Simulator, LineQubit
from openfermion import QubitOperator

from fqe.util import init_qubit_vacuum

if TYPE_CHECKING:
    from cirq.ops.pauli_string import SingleQubitPauliStringGateOperation


def qubit_ops_to_circuit(ops: 'QubitOperator',
                         qpu: List[LineQubit]) -> 'Circuit':
    """Generate a circuit that can be run on a Cirq simulator from the ops
    passed

    Args:
        ops (Qubit Operator) - a product of operations to compile into a
            circuit
        qpu (Qid) - the quantum processing unit that this circuit will be run
            on

    Returns:
        (circuit) - returns a circuit to run in cirq
    """
    gates = []
    for operation in ops:
        gate_type = operation[1]
        qubit = qpu[operation[0]]
        gates.append(qubit_op_to_gate(gate_type, qubit))
    moment = [Moment(gates)]
    return Circuit(moment)


def qubit_op_to_gate(operation: 'QubitOperator',
                     qubit) -> 'SingleQubitPauliStringGateOperation':
    """Convert a qubit operation into a gate operations that can be digested
    by a Cirq simulator.

    Args:
        operation (QubitOperator)
        qubit (Qid) - a qubit on which the Pauli matrices will act.

    Returns:
        (gate) - a gate that can be executed on the qubit passed
    """
    if operation == 'X':
        return X.on(qubit)
    if operation == 'Y':
        return Y.on(qubit)
    if operation == 'Z':
        return Z.on(qubit)
    raise ValueError('No gate identified in qubit_op_to_gate')


def qubit_projection(ops: QubitOperator, qubits: List[LineQubit],
                     state: numpy.ndarray, coeff: numpy.ndarray) -> None:
    """Find the projection of each set of qubit operators on a
    wavefunction.

    Args:
        ops (qubit gates) - A sum of qubit operations which represent the \
            full ci wavefunction in the qubit basis.

        qubits (Qid) - The qubits of a quantum computer.

        state (numpy.array(dtype=numpy.complex64)) - a cirq wavefunction that \
            is being projected

        coeff (numpy.array(dtype=numpy.complex64)) - a coefficient array that \
            will store the result of the projection.
    """
    qpu = Simulator(dtype=numpy.complex128)
    for indx, cluster in enumerate(ops.terms):
        circuit = qubit_ops_to_circuit(cluster, qubits)
        work_state = state.copy()
        result = qpu.simulate(circuit,
                              qubit_order=qubits,
                              initial_state=work_state)
        coeff[indx] = result.final_state_vector[0]


def qubit_wavefunction_from_vacuum(ops: 'QubitOperator',
                                   qubits: List['LineQubit']) -> numpy.ndarray:
    """Generate a cirq wavefunction from the vacuum given qubit operators and a
    set of qubits that this wavefunction will be represented on

    Args:
        ops (QubitOperators) - a sum of qubit operations with coefficients \
            scheduling the creation of the qubit wavefunction

        qubits (Qid) - the qubits of the quantum computer

    Returns:
        final_state (numpy.array(dtype=numpy.complex64)) - the fully projected \
            wavefunction in the cirq representation
    """
    nqubits = len(qubits)
    vacuum = init_qubit_vacuum(nqubits)
    final_state = numpy.zeros(2**nqubits, dtype=numpy.complex128)
    qpu = Simulator(dtype=numpy.complex128)
    for term in ops.terms:
        circuit = qubit_ops_to_circuit(term, qubits)
        state = vacuum.copy()
        result = qpu.simulate(circuit, qubit_order=qubits, initial_state=state)
        final_state += result.final_state_vector * ops.terms[term]
    return final_state
