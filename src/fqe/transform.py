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
"""Transformations between the various paradigms availible to OpenFermion
that provide interoperability.
"""

from typing import TYPE_CHECKING

from openfermion import FermionOperator

import numpy

from cirq import LineQubit

from fqe.cirq_utils import qubit_projection
from fqe.openfermion_utils import convert_qubit_wfn_to_fqe_syntax
from fqe.openfermion_utils import fci_qubit_representation
from fqe.openfermion_utils import update_operator_coeff
from fqe.openfermion_utils import fermion_opstring_to_bitstring

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


def cirq_to_fqe_single(cirq_wfn: numpy.ndarray, nele: int, m_s: int,
                       qubin: int) -> FermionOperator:
    """Given a wavefunction from cirq, create a FermionOperator string which
    will create the same state in the basis of Fermionic modes such that

    .. math::

        |\\Psi\\rangle &= \\mathrm{(qubit\\ operators)}|0 0\\cdots\\rangle
        = \\mathrm{(Fermion\\ Operators)}|\\mathrm{vac}\\rangle \\\\
        |\\Psi\\rangle &= \\sum_iC_i \\mathrm{ops}_{i}|\\mathrm{vac}>

    where the c_{i} are the projection of the wavefunction onto a FCI space.

    Args:
        cirq_wfn (numpy.array(ndim=1, numpy.dtype=complex64)) - coeffcients in \
            the qubit basis.

        nele (int) - the number of electrons

        m_s (int) - the s_z spin angular momentum

        qubiin (LineQUibit) - LineQubits to process the representation

    Returns:
        FermionOperator
    """
    if nele == 0:
        return FermionOperator('', cirq_wfn[0] * 1.)

    if qubin:
        nqubits = qubin
    else:
        nqubits = int(numpy.log2(cirq_wfn.size))

    if nele > nqubits:
        raise ValueError('particle number > number of orbitals')

    norb = nqubits // 2

    jw_ops = fci_qubit_representation(norb, nele, m_s)

    qubits = LineQubit.range(nqubits)
    proj_coeff = numpy.zeros(len(jw_ops.terms), dtype=numpy.complex128)
    qubit_projection(jw_ops, qubits, cirq_wfn, proj_coeff)
    proj_coeff /= (2.**nele)

    update_operator_coeff(jw_ops, proj_coeff)
    return convert_qubit_wfn_to_fqe_syntax(jw_ops)


def from_cirq(wfn: 'Wavefunction', state: numpy.ndarray) -> None:
    """For each availble FqeData structure, find the projection onto the cirq
    wavefunction and set the coefficients to the proper value.

    Args:
        wfn (wavefunction.Wavefunction) - an Fqe Wavefunction to fill from the \
            cirq wavefunction

        state (numpy.array(numpy.dtype=complex64)) - a cirq state to convert \
            into an Fqe wavefunction

    Returns:
        nothing - mutates the wfn in place
    """
    nqubits = int(numpy.log2(state.size))
    for key in wfn.sectors():
        usevec = state.copy()
        fqe_wfn = cirq_to_fqe_single(usevec, key[0], key[1], nqubits)
        wfndata = fermion_opstring_to_bitstring(fqe_wfn)
        for val in wfndata:
            wfn[(val[0], val[1])] = val[2]
