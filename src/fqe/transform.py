#   Copyright 2019 Quantum Simulation Technologies Inc.
#
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

"""Transformations between the various paradigms availible to OpenFermion
that provide interoperability.
"""
from openfermion import FermionOperator

import numpy

from cirq import LineQubit

from fqe.cirq_utils import qubit_projection
from fqe.openfermion_utils import convert_qubit_wfn_to_fqe_syntax
from fqe.openfermion_utils import fci_qubit_representation
from fqe.openfermion_utils import update_operator_coeff
from fqe.openfermion_utils import fermion_opstring_to_bitstring


def cirq_to_fqe_single(cirq_wfn, nele, m_s, qubin=None):
    """Given a wavefunction from cirq, create a FermionOperator string which
    will create the same state in the basis of Fermionic modes such that

    |Psi> = (qubit operators)|0 0 ...> = (Fermion Operators)|-> |phi> =
    = sum_{i}C_{i}ops_{i}|->

    where the c_{i} are the projection of the wavefunction onto a FCI space.

    Args:
        cirq-wfn (numpy.array(ndim=1, numpy.dtype=complex64)) - coeffcients in
            the qubit basis.
    """
    if nele == 0:
        return FermionOperator('', cirq_wfn[0]*1.)
    if qubin:
        nqubits = qubin
    else:
        nqubits = int(numpy.log2(cirq_wfn.size))

    if nele > nqubits:
        raise ValueError('particle number > number of orbitals')

    norb = nqubits // 2

    jw_ops = fci_qubit_representation(norb, nele, m_s)

    qubits = LineQubit.range(nqubits)
    proj_coeff = numpy.zeros(len(jw_ops.terms), dtype=numpy.complex64)
    qubit_projection(jw_ops, qubits, cirq_wfn, proj_coeff)
    proj_coeff /= (2.**nele)

    update_operator_coeff(jw_ops, proj_coeff)
    return convert_qubit_wfn_to_fqe_syntax(jw_ops)


def from_cirq(wfn, state):
    """For each availble FqeData structure, find the projection onto the cirq
    wavefunction and set the coefficients to the proper value.

    Args:
        wfn (wavefunction.Wavefunction) - an Fqe Wavefunction to fill from the
            cirq wavefunction
        state (numpy.array(numpy.dtype=complex64)) - a cirq state to convert
            into an Fqe wavefunction

    Returns:
        nothing - mutates the wfn in place
    """
    nqubits = int(numpy.log2(state.size))
    for key in wfn.configs:
        usevec = state.copy()
        fqe_wfn = cirq_to_fqe_single(usevec, key[0], key[1], nqubits)
        wfndata = fermion_opstring_to_bitstring(fqe_wfn)
        for val in wfndata:
            wfn.set_ele(val[0], val[1], val[2])
