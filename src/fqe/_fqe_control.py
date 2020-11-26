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
"""FQE control is a wrapper to allow for convenient or more readable access
to the emulator."""

# Ungrouped imports are for type hinting
# pylint: disable=ungrouped-imports
# pylint: disable=too-many-arguments

from typing import List, Optional, TYPE_CHECKING, Union, Tuple

import cirq
import numpy as np

from openfermion.transforms.opconversions import jordan_wigner
from openfermion.ops import FermionOperator
from openfermion import normal_ordered

from fqe.util import qubit_particle_number_index_spin
from fqe import util
from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe import transform
from fqe import wavefunction
from fqe.openfermion_utils import fqe_to_fermion_operator
from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)
from fqe.fqe_decorators import build_hamiltonian
from fqe.hamiltonians import diagonal_coulomb
from fqe.hamiltonians import diagonal_hamiltonian
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import gso_hamiltonian
from fqe.hamiltonians import restricted_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import sso_hamiltonian

if TYPE_CHECKING:
    from fqe.hamiltonians import hamiltonian


def apply_generated_unitary(
    wfn: "wavefunction.Wavefunction",
    time: float,
    algo: str,
    hamil: Union["hamiltonian.Hamiltonian", "FermionOperator"],
    accuracy: float = 0.0,
    expansion: int = 30,
    spec_lim: Optional[List[float]] = None,
) -> "wavefunction.Wavefunction":
    """Apply the algebraic operators to the wavefunction with a specfiic
    algorithm and to the requested accuracy.

    Args:
        wfn: The wavefunction to evolve.
        time: Evolution time.
        algo: String dictating the method to use.
        hamil: Hamiltonian which generates the unitary to evolve by.
        accuracy: Desired accuracy to evolve the system to.
        expansion: Maximum number of terms in the polynomial expansion.
        spec_lim: Spectral range of the Hamiltonian. The the length of the list
            should be 2, if provided.

    Returns:
        The evolved wavefunction.
    """
    return wfn.apply_generated_unitary(
        time,
        algo,
        hamil,
        accuracy=accuracy,
        expansion=expansion,
        spec_lim=spec_lim,
    )


def get_spin_conserving_wavefunction(
    s_z: int, norb: int
) -> "wavefunction.Wavefunction":
    """Returns a wavefunction which has s_z conserved.

    Args:
        s_z: The value of :math:`S_z`
        norb: The number of orbitals in the system

    Returns:
        Wavefunction initialized to zero.
    """
    param = []
    if s_z >= 0:
        max_ele = norb + 1
        min_ele = s_z
    if s_z < 0:
        max_ele = norb + s_z + 1
        min_ele = 0

    for nalpha in range(min_ele, max_ele):
        param.append([2 * nalpha - s_z, s_z, norb])

    return wavefunction.Wavefunction(param, broken=["number"])


def get_number_conserving_wavefunction(
    nele: int, norb: int
) -> "wavefunction.Wavefunction":
    """Returns a number conserving wavefunction.

    Args:
        nele: Number of electrons in the system.
        norb: Number of orbitals.

    Returns:
        Wavefunction object meeting the criteria of the input arguments.
    """
    param = []
    maxb = min(norb, nele)
    minb = nele - maxb
    for nbeta in range(minb, maxb + 1):
        m_s = nele - nbeta * 2
        param.append([nele, m_s, norb])
    return wavefunction.Wavefunction(param, broken=["spin"])


# TODO: Function names should be lowercase.
def Wavefunction(
    param: List[List[int]], broken: Optional[Union[List[str], str]] = None
) -> "wavefunction.Wavefunction":
    """Initialize a wavefunction through the fqe namespace.

    Args:
        param: Parameters for the sectors.
        broken: Symmetry to be broken.

    Returns:
        Wavefunction object meeting the criteria of the input arguments.
    """
    return wavefunction.Wavefunction(param, broken=broken)


def get_wavefunction(
    nele: int, m_s: int, norb: int
) -> "wavefunction.Wavefunction":
    """Build a wavefunction with definite particle number and spin.

    Args:
        nele (int) - the number of electrons in the system

        m_s (int) - the s_z spin projection of the system

        norb (int) - the number of spatial orbtials to used

    Returns:
        (wavefunction.Wavefunction) - a wavefunction object meeting the \
            criteria laid out in the calling argument
    """
    arg = [[nele, m_s, norb]]
    return wavefunction.Wavefunction(param=arg)


def time_evolve(
    wfn: "wavefunction.Wavefunction",
    time: float,
    hamil: Union["hamiltonian.Hamiltonian", "FermionOperator"],
    inplace: bool = False,
) -> "wavefunction.Wavefunction":
    """Time-evolve a wavefunction with the specified Hamiltonian.

    Args:
        wfn: Wavefunction to be time-evolved.
        time: Time for propagation.
        hamil: Hamiltonian to be used for time evolution.
        inplace: If True, returns None and modifies input `wfn` in place.

    Returns:
        The evolved wavefunction if `inplace` is False, else None.
    """
    return wfn.time_evolve(time, hamil, inplace)


def get_wavefunction_multiple(
    params: List[List[int]],
) -> List["wavefunction.Wavefunction"]:
    """Returns a list of wavefunction objects, one for each element of params,
    satisfying the criteria defined by params.

    Args:
        params: List of parameters used to initialize wavefunctions.
            The arguments in the parameters are:
                nele (int) - the number of electrons in the system;
                m_s (int) - the s_z spin projection of the system;
                norb (int) - the number of spatial orbitals to use.

    Returns:
        List of wavefunction objects.
    """
    state = []
    for val in params:
        state.append(wavefunction.Wavefunction(param=[val]))
    return state


# TODO: Consider renaming. Numpy arrays are more general than just Cirq.
def to_cirq(wfn: "wavefunction.Wavefunction") -> np.ndarray:
    """Returns a Cirq-compatible wavefunction based on the information stored
    in the FQE wavefunction.

    Args:
        wfn: An FQE wavefunction object.

    Returns:
        A Cirq wavefunction that can be used in a simulator object.
    """
    nqubit = wfn.norb() * 2
    ops = jordan_wigner(fqe_to_fermion_operator(wfn))
    qid = cirq.LineQubit.range(nqubit)
    return qubit_wavefunction_from_vacuum(ops, qid)


def to_cirq_ncr(wfn: "wavefunction.Wavefunction") -> np.ndarray:
    """TODO: Add docstring."""
    nqubit = wfn.norb() * 2
    ops = normal_ordered(fqe_to_fermion_operator(wfn))
    wf_array = np.zeros(2 ** nqubit, dtype=np.complex128)
    for term, coeff in ops.terms.items():
        occ_idx = sum([2 ** (nqubit - oo[0] - 1) for oo in term])
        wf_array[occ_idx] = coeff
    return wf_array


def from_cirq(
    state: np.ndarray, thresh: float
) -> "wavefunction.Wavefunction":
    """Returns an FQE wavefunction object equivalent to the input state.

    Args:
        state: Wavefunction expressed as a np.ndarray.

        thresh: Sets the limit at which an amplitude should be considered zero
            and not make a contribution to the FQE wavefunction.

    Returns:
        FQE wavefunction object equivalent to the input state.
    """
    param = []
    nqubits = int(np.log2(state.size))
    norb = nqubits // 2
    for pnum in range(nqubits + 1):
        occ = qubit_particle_number_index_spin(nqubits, pnum)
        for orb in occ:
            if np.absolute(state[orb[0]]) > thresh:
                param.append([pnum, orb[1], norb])
    param = set(tuple(p) for p in param)
    param = [list(x) for x in param]
    wfn = wavefunction.Wavefunction(param)
    transform.from_cirq(wfn, state)
    return wfn


def apply(
    ops: Union["hamiltonian.Hamiltonian", "FermionOperator"],
    wfn: "wavefunction.Wavefunction",
) -> "wavefunction.Wavefunction":
    """Returns a new wavefunction after applying the fermionic operators to the
    wavefunction.

    Args:
        ops: A FermionOperator to apply to the wavefunction.
        wfn: An FQE wavefunction to mutate.

    Returns:
        A new wavefunction generated from the application of the fermion
            operators to the input wavefunction.
    """
    return wfn.apply(ops)


# TODO: Arg names could be more descriptive.
#  Even just bra_wfn instead of brawfn.
def expectation_value(
    wfn: "wavefunction.Wavefunction",
    ops: Union["hamiltonian.Hamiltonian", "FermionOperator"],
    brawfn: Optional["wavefunction.Wavefunction"] = None,
) -> complex:
    """Returns the expectation value for the operator and wavefunction.

    Args:
        wfn (wavefunction.Wavefunction) - an FQE wavefunction on the ket side
        ops: FermionOperator to apply to the wavefunction.
        brawfn: An FQE wavefunction on the bra side. If not specified, it is
         assumed that the bra and ket wavefunctions are the same.

    Returns:
        Expectation value.
    """
    return wfn.expectationValue(ops, brawfn)


def get_s2_operator() -> "s2_op.S2Operator":
    """Returns an S^2 operator."""
    return S2Operator()


def get_sz_operator() -> "sz_op.SzOperator":
    """Returns an S_z operator."""
    return SzOperator()


def get_time_reversal_operator() -> "tr_op.TimeReversalOp":
    """Returns a time reversal operator."""
    return TimeReversalOp()


def get_number_operator() -> "number_op.NumberOperator":
    """Returns the particle number operator."""
    return NumberOperator()


def dot(
    wfn1: "wavefunction.Wavefunction", wfn2: "wavefunction.Wavefunction"
) -> complex:
    """Returns the inner product of two wavefunctions using conjugation on
    the elements of wfn1.

    Args:
        wfn1: Wavefunction corresponding to the conjugate row vector.
        wfn2: wavefunction corresponding to the column vector.

    Returns:
        Inner (dot) product of the two wavefunctions.
    """
    return util.dot(wfn1, wfn2)


def vdot(
    wfn1: "wavefunction.Wavefunction", wfn2: "wavefunction.Wavefunction"
) -> complex:
    """Returns the inner product of two wavefunctions using conjugation on
    the elements of wfn1.

    Args:
        wfn1: Wavefunction corresponding to the conjugate row vector.
        wfn2: wavefunction corresponding to the column vector.

    Returns:
        Inner (dot) product of the two wavefunctions.
    """
    return util.vdot(wfn1, wfn2)


def get_hamiltonian_from_openfermion(
    ops: "FermionOperator",
    norb: int = 0,
    conserve_number: bool = True,
    e_0: complex = 0.0 + 0.0j,
) -> "hamiltonian.Hamiltonian":
    """Returns the equivalent FQE Hamiltonian from the given Hamiltonian
    represented as OpenFermion FermionOperators.

    Args:
        ops: String of FermionOperators representing the Hamiltonian.
        norb: Number of spatial orbitals in the Hamiltonian.
        conserve_number: Flag to indicate if the Hamiltonian will be
            applied to a number conserving wavefunction.
        e_0: Scalar potential of the hamiltonian.

    Returns:
        An FQE Hamiltonian equivalent to the input OpenFermion Hamiltonian
        defined by `ops`.
    """
    assert isinstance(ops, FermionOperator)

    return build_hamiltonian(
        ops, norb=norb, conserve_number=conserve_number, e_0=e_0
    )


def get_diagonalcoulomb_hamiltonian(
    h2e: "np.ndarray", e_0: complex = 0.0 + 0.0j
) -> "diagonal_coulomb.DiagonalCoulomb":
    """Returns a diagonal coulomb Hamiltonian.

    Args:
        h2e: TODO: Add description.
        e_0: Scalar part of the Hamiltonian.
    """
    return diagonal_coulomb.DiagonalCoulomb(h2e, e_0=e_0)


def get_diagonal_hamiltonian(
    hdiag: "np.ndarray", e_0: complex = 0.0 + 0.0j
) -> "diagonal_hamiltonian.Diagonal":
    """Returns a diagonal Hamiltonian.

    Args:
        hdiag: Diagonal elements.
        e_0: Scalar part of the Hamiltonian.
    """
    return diagonal_hamiltonian.Diagonal(hdiag, e_0=e_0)


def get_general_hamiltonian(
    tensors: Tuple[np.ndarray, ...], e_0: complex = 0.0 + 0.0j
) -> "general_hamiltonian.General":
    """Returns a general Hamiltonian.

    Args:
        tensors: Tensors for the Hamiltonian elements.
        e_0: Scalar part of the Hamiltonian.
    """
    return general_hamiltonian.General(tensors, e_0=e_0)


def get_gso_hamiltonian(
    tensors: Tuple[np.ndarray, ...], e_0: complex = 0.0 + 0.0j
) -> "gso_hamiltonian.GSOHamiltonian":
    """Returns a generalized spin orbital Hamiltonian.

    Args:
        tensors: Tensors for the Hamiltonian elements.
        e_0: Scalar part of the Hamiltonian.
    """
    return gso_hamiltonian.GSOHamiltonian(tensors, e_0=e_0)


def get_restricted_hamiltonian(
    tensors: Tuple[np.ndarray, ...], e_0: complex = 0.0 + 0.0j
) -> "restricted_hamiltonian.Restricted":
    """Returns spin conserving spin restricted hamiltonian

    Args:
        tensors: Tensors for the Hamiltonian elements.
        e_0: Scalar part of the Hamiltonian.
    """
    return restricted_hamiltonian.RestrictedHamiltonian(tensors, e_0=e_0)


def get_sparse_hamiltonian(
    operators: Union["FermionOperator", str],
    conserve_spin: bool = True,
    e_0: complex = 0.0 + 0.0j,
) -> "sparse_hamiltonian.SparseHamiltonian":
    """Returns a sparse Hamiltonian.

    Args:
        operators: FermionOperator to be used to initialize the sparse
            Hamiltonian.
        conserve_spin: Whether the Hamiltonian conserves Sz.
        e_0: Scalar part of the Hamiltonian.
    """
    return sparse_hamiltonian.SparseHamiltonian(
        operators, conserve_spin=conserve_spin, e_0=e_0
    )


def get_sso_hamiltonian(
    tensors: Tuple[np.ndarray, ...], e_0: complex = 0.0 + 0.0j
) -> "sso_hamiltonian.SSOHamiltonian":
    """Returns a Spin-conserving Spin Orbital Hamiltonian.

    Args:
        tensors: Tensors for the Hamiltonian elements.
        e_0: Scalar part of the Hamiltonian.
    """
    return sso_hamiltonian.SSOHamiltonian(tensors, e_0=e_0)
