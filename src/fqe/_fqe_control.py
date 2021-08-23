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
""" FQE control is a wrapper to allow for convenient or more readable access
to the emulator.
"""
#ungrouped imports are for type hinting
#pylint: disable=ungrouped-imports

from typing import List, Optional, TYPE_CHECKING, Union, Tuple

import cirq
import numpy

from openfermion.transforms.opconversions import jordan_wigner
from openfermion.ops import FermionOperator, BinaryCode
from openfermion import normal_ordered, up_index, down_index

import fqe.settings
from fqe.util import qubit_particle_number_index_spin
from fqe import util
from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe.lib.cirq_utils import _detect_cirq_sectors
from fqe import transform
from fqe import wavefunction
from fqe.bitstring import integer_index

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
from fqe.lib.fqe_data import _to_cirq

if TYPE_CHECKING:
    from fqe.hamiltonians import hamiltonian
    from fqe.wavefunction import Wavefunction


def apply_generated_unitary(
        wfn: 'Wavefunction',
        time: float,
        algo: str,
        hamil: Union['hamiltonian.Hamiltonian', 'FermionOperator'],
        accuracy: float = 0.0,
        expansion: int = 30,
        spec_lim: Optional[List[float]] = None) -> 'Wavefunction':
    """Apply the algebraic operators to the wavefunction with a specfiic
    algorithm and to the requested accuracy.

    Args:
        wfn (fqe.wavefunction): the wavefunction to evolve

        time (float): time for propagation

        algo (string): a string dictating the method to use

        hamil (hamiltonian.Hamiltonian or FermionOperator): a hermetian operator \
            to apply to the wavefunction

        accuracy (float): a desired accuracy to evolve the system to

        expansion (int): the max order of the polynomial expansion

        spec_lim (Optional[List[float]]): range of eigenvalues to be given when \
            the Chebyshev expansion is used

    Returns:
        (Wavefunction): the evolved wavefunction
    """
    return wfn.apply_generated_unitary(time,
                                       algo,
                                       hamil,
                                       accuracy=accuracy,
                                       expansion=expansion,
                                       spec_lim=spec_lim)


def get_spin_conserving_wavefunction(s_z: int, norb: int) -> 'Wavefunction':
    """Return a wavefunction which has s_z conserved

    Args:
        s_z (int): the value of :math:`S_z`

        norb (int): the number of orbitals in the system

    Returns:
        (Wavefunction): wave function initialized to zero
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

    return wavefunction.Wavefunction(param, broken=['number'])


def get_number_conserving_wavefunction(nele: int, norb: int) -> 'Wavefunction':
    """Build a wavefunction

    Args:
        nele (int): the number of electrons in the system

        norb (int): the number of orbitals

    Returns:
        (Wavefunction): a wavefunction object meeting the \
            criteria laid out in the calling argument
    """
    param = []
    maxb = min(norb, nele)
    minb = nele - maxb
    for nbeta in range(minb, maxb + 1):
        m_s = nele - nbeta * 2
        param.append([nele, m_s, norb])
    return wavefunction.Wavefunction(param, broken=['spin'])


def Wavefunction(  # type: ignore # pylint: disable=function-redefined
        param: List[List[int]],
        broken: Optional[Union[List[str], str]] = None) -> 'Wavefunction':
    """Initialize a wavefunction through the fqe namespace

    Args:
        param (List[List[int]]): parameters for the sectors

        broken (Union[List[str], str]): symmetry to be broken

    Returns:
        (Wavefunction): a wavefunction object meeting the \
            criteria laid out in the calling argument
    """
    return wavefunction.Wavefunction(param, broken=broken)


def get_wavefunction(nele: int, m_s: int, norb: int) -> 'Wavefunction':
    """Build a wavefunction with definite particle number and spin.

    Args:
        nele (int): the number of electrons in the system

        m_s (int): the s_z spin projection of the system

        norb (int): the number of spatial orbtials to used

    Returns:
        (Wavefunction): a wavefunction object meeting the \
            criteria laid out in the calling argument
    """
    arg = [[nele, m_s, norb]]
    return wavefunction.Wavefunction(param=arg)


def time_evolve(wfn: 'Wavefunction',
                time: float,
                hamil: Union['hamiltonian.Hamiltonian', 'FermionOperator'],
                inplace: bool = False) -> 'Wavefunction':
    """Time-evolve a wavefunction with the specified Hamiltonian.

    Args:
        wfn (Wavefunction): Wave function to be time-evolved

        time (float): time for propagation

        hamil (Hamiltonian or FermionOperator): Hamiltonian to be used for time evolution

        inplace (bool): specify if the result is stored in-place

    Returns:
        (Wavefunction): a wavefunction object after time evolution
    """
    return wfn.time_evolve(time, hamil, inplace)


def get_wavefunction_multiple(param: List[List[int]]) -> List['Wavefunction']:
    """Generate many different wavefunctions.

    Args:
        param (list[list[nele, m_s, norb]]): a list of parameters used to \
            initialize wavefunctions.  The arguments in the parameters are

                nele (int): the number of electrons in the system;

                m_s (int): the s_z spin projection of the system;

                norb (int): the number of spatial orbtials to used

    Returns:
        list[(Wavefunction)]: a list of wavefunction objects
    """
    state = []
    for val in param:
        state.append(wavefunction.Wavefunction(param=[val]))
    return state


def to_cirq_old(wfn: 'Wavefunction') -> numpy.ndarray:
    """Interoperability between cirq and the openfermion-fqe.  This takes an
    FQE wavefunction and returns a cirq compatible wavefunction based on the
    information stored within.

    Args:
        wfn (Wavefunction): a openfermion-fqe wavefunction object

    Returns:
        numpy.array(dtype=numpy.complex128): a cirq wavefunction that can be \
            used in a simulator object.
    """
    nqubit = wfn.norb() * 2
    ops = jordan_wigner(fqe_to_fermion_operator(wfn))
    qid = cirq.LineQubit.range(nqubit)
    return qubit_wavefunction_from_vacuum(ops, qid)


def to_cirq(wfn: 'Wavefunction',
            binarycode: Optional['BinaryCode'] = None) -> numpy.ndarray:
    """Interoperability between cirq and the openfermion-fqe.  This takes an
    FQE wavefunction and returns a cirq compatible wavefunction based on the
    information stored within.

    Args:
        wfn (Wavefunction): a openfermion-fqe wavefunction object

        binarycode (Optional[openfermion.ops.BinaryCode]): binary code to \
            encode the fermions to the qbit bosons. If None given, \
            Jordan-Wigner transform is assumed.

    Returns:
        numpy.array(dtype=numpy.complex128): a cirq wavefunction that can be \
            used in a simulator object.
    """

    if fqe.settings.use_accelerated_code:
        nqubits = wfn.norb() * 2
        wf = numpy.zeros(2**nqubits, dtype=numpy.complex128)

        for key in wfn.sectors():
            csector = wfn._civec[(key[0], key[1])]
            _to_cirq(csector, wf, binarycode)
        return wf
    else:
        return to_cirq_old(wfn)


def from_cirq(state: numpy.ndarray,
              thresh: float,
              binarycode: Optional['BinaryCode'] = None) -> 'Wavefunction':
    """Interoperability between cirq and the openfermion-fqe.  This takes a
    cirq wavefunction and creates an FQE wavefunction object initialized with
    the correct data.

    Args:
        state (numpy.array(dtype=numpy.complex128)): a cirq wavefunction

        thresh (double): set the limit at which a cirq element should be \
            considered zero and not make a contribution to the FQE wavefunction

        binarycode (Optional[openfermion.ops.BinaryCode]): binary code to \
            encode the fermions to the qbit bosons. If None given, \
            Jordan-Wigner transform is assumed.

    Returns:
        openfermion-fqe.Wavefunction
    """
    # first detect non-zero sectors
    state = state.astype(numpy.complex128)
    nqubit = int(numpy.log2(state.size))
    norb = nqubit // 2

    nlena = 2**norb
    nlenb = 2**norb

    # occupations of all possible alpha and beta strings
    aoccs = [[up_index(x) for x in integer_index(astr)] for astr in range(nlena)
            ]
    boccs = [
        [down_index(x) for x in integer_index(bstr)] for bstr in range(nlenb)
    ]

    # Since cirq starts counting from the leftmost bit in a bitstring
    pow_of_two = 2**(nqubit - numpy.arange(nqubit, dtype=numpy.int64) - 1)
    if binarycode is None:
        # cirq index for each alpha or beta string
        cirq_aid = numpy.array([pow_of_two[aocc].sum() for aocc in aoccs])
        cirq_bid = numpy.array([pow_of_two[bocc].sum() for bocc in boccs])
    else:

        def occ_to_cirq_ids(occs):
            cirq_ids = numpy.zeros(len(aoccs), dtype=numpy.int64)
            for ii, occ in enumerate(occs):
                of_state = numpy.zeros(nqubit, dtype=int)
                of_state[occ] = 1
                # Encode the occupation state to the qbit spin state
                cirq_state = numpy.mod(binarycode.encoder.dot(of_state), 2)
                cirq_ids[ii] = numpy.dot(pow_of_two, cirq_state)
            return cirq_ids

        # cirq index for each alpha or beta string
        cirq_aid = occ_to_cirq_ids(aoccs)
        cirq_bid = occ_to_cirq_ids(boccs)

    # Number of alpha or beta electrons for each alpha or beta string
    anumb = numpy.array([len(x) for x in aoccs], dtype=numpy.int32)
    bnumb = numpy.array([len(x) for x in boccs], dtype=numpy.int32)
    param = numpy.zeros((2 * norb + 1, 2 * norb + 1), dtype=numpy.int32)

    if fqe.settings.use_accelerated_code:
        _detect_cirq_sectors(state, thresh, param, norb, nlena, nlenb, cirq_aid,
                             cirq_bid, anumb, bnumb)
    else:
        for aid in range(nlena):
            c_aid = cirq_aid[aid]
            anum = anumb[aid]
            for bid in range(nlenb):
                c_id = c_aid ^ cirq_bid[bid]
                if abs(state[c_id]) < thresh:
                    continue
                bnum = bnumb[bid]
                param[anum + bnum, anum - bnum + norb] = 1

    sectors = [
        [pnum, sz - norb, norb] for pnum, sz in zip(*numpy.nonzero(param))
    ]

    wfn = wavefunction.Wavefunction(sectors)
    transform.from_cirq(wfn, state, binarycode)
    return wfn


def apply(ops: Union['hamiltonian.Hamiltonian', 'FermionOperator'],
          wfn: 'Wavefunction') -> 'Wavefunction':
    """Create a new wavefunction by applying the fermionic operators to the
    wavefunction.

    Args:
        ops (FermionOperator or Hamiltonian): a Fermion Operator string to apply \
            to the wavefunction

        wfn (Wavefunction): an FQE wavefunction to mutate

    Returns:
        openfermion-fqe.Wavefunction: a new wavefunction generated from the \
            application of the fermion operators to the wavefunction
    """
    return wfn.apply(ops)


def expectationValue(wfn: 'Wavefunction',
                     ops: Union['hamiltonian.Hamiltonian', 'FermionOperator'],
                     brawfn: Optional['Wavefunction'] = None) -> complex:
    """Return the expectation value for the passed operator and wavefunction

    Args:
        wfn (Wavefunction): an FQE wavefunction on the ket side

        ops (FermionOperator or Hamiltonian): a Fermion Operator string to apply \
            to the wavefunction

        brawfn (Wavefunction): an FQE wavefunction on the bra side \
            if not specified, it is assumed that the bra nad ket wave functions \
            are the same

    Returns:
        (complex): expectation value
    """
    return wfn.expectationValue(ops, brawfn)


def get_s2_operator() -> 'S2Operator':
    """Return an S^2 operator.

    Returns:
        S2Operator
    """
    return S2Operator()


def get_sz_operator() -> 'SzOperator':
    """Return an S_z operator.

    Returns:
        SzOperator
    """
    return SzOperator()


def get_time_reversal_operator() -> 'TimeReversalOp':
    """Return a time reversal operator

    Returns:
        TimeReversalOp
    """
    return TimeReversalOp()


def get_number_operator() -> 'NumberOperator':
    """Return the particle number operator

    Returns:
        NumberOperator
    """
    return NumberOperator()


def dot(wfn1: 'Wavefunction', wfn2: 'Wavefunction') -> complex:
    """Calculate the inner product of two wavefunctions using conjugation on
    the elements of wfn1.

    Args:
        wfn1 (Wavefunction): wavefunction corresponding to the \
            conjugate row vector

        wfn2 (Wavefunction): wavefunction corresponding to the \
            coumn vector

    Returns:
        (complex): scalar as result of the dot product
    """
    return util.dot(wfn1, wfn2)


def vdot(wfn1: 'Wavefunction', wfn2: 'Wavefunction') -> complex:
    """Calculate the inner product of two wavefunctions using conjugation on
    the elements of wfn1.

    Args:
        wfn1 (Wavefunction): wavefunction corresponding to the \
            conjugate row vector

        wfn2 (Wavefunction): wavefunction corresponding to the \
            coumn vector

    Returns:
        (complex): scalar as result of the dot product
    """
    return util.vdot(wfn1, wfn2)


def get_hamiltonian_from_openfermion(ops: 'FermionOperator',
                                     norb: int = 0,
                                     conserve_number: bool = True,
                                     e_0: complex = 0. + 0.j
                                    ) -> 'hamiltonian.Hamiltonian':
    """Given an OpenFermion Hamiltonian return the fqe hamiltonian.

    Args:
        ops (openfermion.FermionOperator): a string of FermionOperators \
            representing the Hamiltonian.

        norb (int): the number of spatial orbitals in the Hamiltonian

        conserve_number (bool): a flag to indicate if the Hamiltonian will be \
            applied to a number_conserving wavefunction.

        e_0 (complex): the scalar potential of the hamiltonian


    Returns:
        (fqe.hamiltonians.hamiltonian): Initialized Hamiltonian object
    """
    assert isinstance(ops, FermionOperator)

    return build_hamiltonian(ops,
                             norb=norb,
                             conserve_number=conserve_number,
                             e_0=e_0)


def get_diagonalcoulomb_hamiltonian(h2e: 'numpy.ndarray',
                                    e_0: complex = 0. + 0.j
                                   ) -> 'diagonal_coulomb.DiagonalCoulomb':
    """Initialize a diagonal coulomb hamiltonian

    Args:
        h2e (Nparray): two-body part of the diagonal Coulomb Hamiltonian

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (diagonal_coulomb.DiagonalCoulomb): Initialized diagonal-Coulomb Hamiltonian
    """
    return diagonal_coulomb.DiagonalCoulomb(h2e, e_0=e_0)


def get_diagonal_hamiltonian(hdiag: 'numpy.ndarray', e_0: complex = 0. + 0.j
                            ) -> 'diagonal_hamiltonian.Diagonal':
    """Initialize a diagonal hamiltonian

    Args:
        hdiag (numpy.ndarray): diagonal elements

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (diagonal_hamiltonian.Diagonal): Initialized diagonal Hamiltonian
    """
    return diagonal_hamiltonian.Diagonal(hdiag, e_0=e_0)


def get_general_hamiltonian(tensors: Tuple[numpy.ndarray, ...],
                            e_0: complex = 0. + 0.j
                           ) -> 'general_hamiltonian.General':
    """Initialize the most general hamiltonian class.

    Args:
        tensors (Tuple[numpy.ndarray, ...]): tensors for the Hamiltonian elements

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (general_hamiltonian.General): Initialized general Hamiltonian
    """
    return general_hamiltonian.General(tensors, e_0=e_0)


def get_gso_hamiltonian(tensors: Tuple[numpy.ndarray, ...],
                        e_0: complex = 0. + 0.j
                       ) -> 'gso_hamiltonian.GSOHamiltonian':
    """Initialize the generalized spin orbital hamiltonian

    Args:
        tensors (Tuple[numpy.ndarray, ...]): tensors for the Hamiltonian elements

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (gso_hamiltonian.GSOHamiltonian): Initialized general spin-orbital
            Hamiltonian
    """
    return gso_hamiltonian.GSOHamiltonian(tensors, e_0=e_0)


def get_restricted_hamiltonian(
        tensors: Tuple[numpy.ndarray, ...], e_0: complex = 0. + 0.j
) -> 'restricted_hamiltonian.RestrictedHamiltonian':
    """Initialize spin conserving spin restricted hamiltonian

    Args:
        tensors (Tuple[numpy.ndarray, ...]): tensors for the Hamiltonian elements. \
            Note that the tensor should be in a spin-free form; therefore, \
            the size of each dimension is the number of spatial orbitals.

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (restricted_hamiltonian.RestrictedHamiltonian): Initialized restricted Hamiltonian
    """
    return restricted_hamiltonian.RestrictedHamiltonian(tensors, e_0=e_0)


def get_sparse_hamiltonian(operators: Union['FermionOperator', str],
                           conserve_spin: bool = True,
                           e_0: complex = 0. + 0.j
                          ) -> 'sparse_hamiltonian.SparseHamiltonian':
    """Initialize the sparse hamiltonaian

    Args:
        operators (Union[FermionOperator, str]): the FermionOperator to be used to \
            initialize the sparse Hamiltonian

        conserve_spin (bool): whether the Hamiltonian conserves Sz

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (sparse_hamiltonian.SparseHamiltonian): Initialized sparse Hamiltonian
    """
    return sparse_hamiltonian.SparseHamiltonian(operators,
                                                conserve_spin=conserve_spin,
                                                e_0=e_0)


def get_sso_hamiltonian(tensors: Tuple[numpy.ndarray, ...],
                        e_0: complex = 0. + 0.j
                       ) -> 'sso_hamiltonian.SSOHamiltonian':
    """Initialize the Spin-conserving Spin Orbital Hamiltonian

    Args:
        tensors (Tuple[numpy.ndarray, ...]): tensors for the Hamiltonian elements

        e_0 (complex): scalar part of the Hamiltonian

    Returns:
        (sso_hamiltonian.SSOHamiltonian): Initialized spin-conserving,
            spin-orbital Hamiltonian
    """
    return sso_hamiltonian.SSOHamiltonian(tensors, e_0=e_0)
