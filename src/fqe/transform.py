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

from typing import TYPE_CHECKING, Optional

import numpy
from openfermion import FermionOperator, BinaryCode
from cirq import LineQubit

import fqe.settings
from fqe.cirq_utils import qubit_projection
from fqe.openfermion_utils import convert_qubit_wfn_to_fqe_syntax
from fqe.openfermion_utils import fci_qubit_representation
from fqe.openfermion_utils import update_operator_coeff
from fqe.openfermion_utils import fermion_opstring_to_bitstring
from fqe.lib.fqe_data import _from_cirq

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


def cirq_to_fqe_single(cirq_wfn: numpy.ndarray, nele: int, m_s: int,
                       qubin: int) -> FermionOperator:
    """Given a wavefunction from cirq, create a FermionOperator string which
    will create the same state in the basis of Fermionic modes such that

    .. math::

        |\\Psi\\rangle &= \\mathrm{(qubit\\ operators)}|0 0\\cdots\\rangle
        = \\mathrm{(Fermion\\ Operators)}|\\mathrm{vac}\\rangle \\\\
        |\\Psi\\rangle &= \\sum_iC_i \\mathrm{ops}_{i}|\\mathrm{vac}\\rangle

    where the :math:`c_{i}` are the projection of the wavefunction onto a FCI space.

    Args:
        cirq_wfn (numpy.array(ndim=1, numpy.dtype=complex128)): coeffcients in \
            the qubit basis.

        nele (int): the number of electrons

        m_s (int): the s_z spin angular momentum

        qubiin (LineQubit): LineQubits to process the representation.

    Returns:
        (FermionOperator)
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


def from_cirq_old(wfn: 'Wavefunction', state: numpy.ndarray) -> None:
    """For each availble FqeData structure, find the projection onto the cirq
    wavefunction and set the coefficients to the proper value.

    This is the old Python implementation of `from_cirq`. It is advised to use the
    function `from_cirq` instead, which may select an accelerated C codepath
    if available.

    Args:
        wfn (wavefunction.Wavefunction): an FQE Wavefunction to fill from the \
            cirq wavefunction

        state (numpy.array(numpy.dtype=complex128)): a cirq state to convert \
            into an FQE wavefunction

    Returns:
        nothing: mutates the wfn in place
    """
    nqubits = int(numpy.log2(state.size))
    for key in wfn.sectors():
        usevec = state.copy()
        fqe_wfn = cirq_to_fqe_single(usevec, key[0], key[1], nqubits)
        wfndata = fermion_opstring_to_bitstring(fqe_wfn)
        for val in wfndata:
            wfn[(val[0], val[1])] = val[2]


def from_cirq(wfn: 'Wavefunction',
              state: numpy.ndarray,
              binarycode: Optional['BinaryCode'] = None) -> None:
    """For each availble FqeData structure, find the projection onto the cirq
    wavefunction and set the coefficients to the proper value.

    Cirq coefficients that have zero projection onto `wfn` are ignored. It is
    up to the user's own descretion to provide a correct wavefunction.

    If in doubt, the user can use `fqe.from_cirq`, which initializes the
    Wavefunction.

    Args:
        wfn (wavefunction.Wavefunction): an FQE Wavefunction to fill from the \
            cirq wavefunction

        state (numpy.array(numpy.dtype=complex128)): a cirq state to convert \
            into an FQE wavefunction

        binarycode (Optional[openfermion.ops.BinaryCode]): binary code to \
            encode the fermions to the qbit bosons. If None given,
            Jordan-Wigner transform is assumed.

    Returns:
        nothing - mutates the wfn in place
    """
    if fqe.settings.use_accelerated_code:
        for key in wfn.sectors():
            csector = wfn._civec[(key[0], key[1])]
            _from_cirq(csector, state, binarycode)
    else:
        from_cirq_old(wfn, state)
