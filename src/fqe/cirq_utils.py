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
"""Utilities for interfacing with Cirq."""

# Type checking splits the imports
# pylint: disable=ungrouped-imports

from typing import List, Union, TYPE_CHECKING

import numpy as np

from cirq import X, Y, Z, Moment, Circuit, Simulator, Qid
from openfermion import QubitOperator

from fqe.util import init_qubit_vacuum

if TYPE_CHECKING:
    from cirq.ops.pauli_string import SingleQubitPauliStringGateOperation


def qubit_ops_to_circuit(
    ops: "QubitOperator", qubits: List[Qid]
) -> "Circuit":
    """Returns a Cirq circuit from the input ops and qubits.

    Args:
        ops: A product of operations to compile into a circuit.
        qubits: List of qubits the operators should act on.

    Returns:
        A Cirq circuit representation of the input ops.
    """
    gates = []
    for operation in ops:
        gate_type = operation[1]
        qubit = qubits[operation[0]]
        gates.append(qubit_op_to_gate(gate_type, qubit))
    moment = [Moment(gates)]
    return Circuit(moment)


def qubit_op_to_gate(
    operation: Union[str, "QubitOperator"], qubit: Qid
) -> "SingleQubitPauliStringGateOperation":
    """Converts a qubit operation into a gate operations that can be digested
    by a Cirq simulator.

    Args:
        operation: Operation represented as a QubitOperator.
            Either "X", "Y", or "Z".
        qubit: A qubit on which the Pauli matrices will act.

    Returns:
        A Pauli gate operation acting on the qubit.
    """
    if operation == "X":
        return X.on(qubit)
    if operation == "Y":
        return Y.on(qubit)
    if operation == "Z":
        return Z.on(qubit)
    raise ValueError("Invalid operation. Expected 'X', 'Y', or 'Z'.")


def qubit_projection(
    ops: QubitOperator,
    qubits: List[Qid],
    state: np.ndarray,
    coeff: np.ndarray,
) -> None:
    """Finds the projection of each set of qubit operators on a
    wavefunction, modifying the input coeff array.

    Args:
        ops: Sum of qubit operations which represent the full CI wavefunction
            in the qubit basis.
        qubits: Qubits of a quantum computer.
        state: Cirq wavefunction that is being projected.
        coeff: Coefficient array that will store the result of the projection.
    """
    qpu = Simulator(dtype=np.complex128)
    for indx, cluster in enumerate(ops.terms):
        circuit = qubit_ops_to_circuit(cluster, qubits)
        work_state = state.copy()
        result = qpu.simulate(
            circuit, qubit_order=qubits, initial_state=work_state
        )
        coeff[indx] = result.final_state_vector[0]


def qubit_wavefunction_from_vacuum(
    ops: "QubitOperator", qubits: List[Qid]
) -> np.ndarray:
    """Generate a Cirq wavefunction from the vacuum given qubit operators and a
    set of qubits that this wavefunction will be represented on.

    Args:
        ops: Sum of qubit operations with coefficients scheduling the creation
            of the qubit wavefunction
        qubits: Qubits of a quantum computer.

    Returns:
        The fully projected wavefunction.
    """
    nqubits = len(qubits)
    vacuum = init_qubit_vacuum(nqubits)
    final_state = np.zeros(2 ** nqubits, dtype=np.complex128)
    qpu = Simulator(dtype=np.complex128)
    for term in ops.terms:
        circuit = qubit_ops_to_circuit(term, qubits)
        state = vacuum.copy()
        result = qpu.simulate(circuit, qubit_order=qubits, initial_state=state)
        final_state += result.final_state_vector * ops.terms[term]
    return final_state
