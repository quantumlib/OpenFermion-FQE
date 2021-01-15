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
"""Utilities and decorators for converting external types into the fqe
intrinsics
"""
#there are two places where access to protected members improves code quality
#pylint: disable=protected-access

from typing import Any, Dict, Tuple, Union, Optional, List
from functools import wraps

import copy

import numpy

from openfermion import FermionOperator
from openfermion.utils import is_hermitian
from openfermion import normal_ordered

from fqe.hamiltonians import hamiltonian
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import diagonal_hamiltonian
from fqe.hamiltonians import diagonal_coulomb
from fqe.hamiltonians import gso_hamiltonian
from fqe.hamiltonians import restricted_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import sso_hamiltonian
from fqe.openfermion_utils import largest_operator_index
from fqe.util import validate_tuple, reverse_bubble_list
from fqe.fqe_ops import fqe_ops_utils


def build_hamiltonian(ops: Union[FermionOperator, hamiltonian.Hamiltonian],
                      norb: int = 0,
                      conserve_number: bool = True,
                      e_0: complex = 0. + 0.j) -> 'hamiltonian.Hamiltonian':
    """Build a Hamiltonian object for the fqe

    Args:
        ops (FermionOperator, hamiltonian.Hamiltonian) - input operator as \
            FermionOperator.  If a Hamiltonian is passed as an argument, \
            this function returns as is.

        norb (int) - the number of orbitals in the system

        conserve_number (bool) - whether the operator conserves the number

        e_0 (complex) - the scalar part of the operator

    Returns:
        (hamiltonian.Hamiltonian) - General Hamiltonian that is created from ops
    """
    if isinstance(ops, hamiltonian.Hamiltonian):
        return ops

    if isinstance(ops, tuple):
        validate_tuple(ops)

        return general_hamiltonian.General(ops, e_0=e_0)

    if not isinstance(ops, FermionOperator):
        raise TypeError('Expected FermionOperator' \
                        ' but received {}.'.format(type(ops)))

    assert is_hermitian(ops)

    out: Any
    if len(ops.terms) <= 2:
        out = sparse_hamiltonian.SparseHamiltonian(ops, e_0=e_0)

    else:
        if not conserve_number:
            ops = transform_to_spin_broken(ops)

        ops = normal_ordered(ops)

        ops_rank, e_0 = split_openfermion_tensor(ops)  # type: ignore

        if norb == 0:
            for term in ops_rank.values():
                ablk, bblk = largest_operator_index(term)
                norb = max(norb, ablk // 2 + 1, bblk // 2 + 1)
        else:
            norb = norb

        ops_mat = {}
        maxrank = 0
        for rank, term in ops_rank.items():
            index = rank // 2 - 1
            ops_mat[index] = fermionops_tomatrix(term, norb)
            maxrank = max(index, maxrank)

        if len(ops_mat) == 1 and (0 in ops_mat):
            out = process_rank2_matrix(ops_mat[0], norb=norb, e_0=e_0)
        elif len(ops_mat) == 1 and \
            (1 in ops_mat) and \
            check_diagonal_coulomb(ops_mat[1]):
            out = diagonal_coulomb.DiagonalCoulomb(ops_mat[1], e_0=e_0)

        else:
            dtypes = [xx.dtype for xx in ops_mat.values()]
            dtypes = numpy.unique(dtypes)
            if len(dtypes) != 1:
                raise TypeError(
                    "Non-unique coefficient types for input operator")

            for i in range(maxrank + 1):
                if i not in ops_mat:
                    mat_dim = tuple([2 * norb for _ in range((i + 1) * 2)])
                    ops_mat[i] = numpy.zeros(mat_dim, dtype=dtypes[0])

            ops_mat2 = []
            for i in range(maxrank + 1):
                ops_mat2.append(ops_mat[i])

            out = general_hamiltonian.General(tuple(ops_mat2), e_0=e_0)

    out._conserve_number = conserve_number
    return out


def transform_to_spin_broken(ops: 'FermionOperator') -> 'FermionOperator':
    """Convert a Fermion Operator string from number broken to spin broken
    operators.

    Args:
        ops (FermionOperator) - input FermionOperator

    Returns:
        (FermionOperator) - transformed FermionOperator to spin broken indexing
    """
    newstr = FermionOperator()
    for term in ops.terms:
        opstr = ''
        for element in term:
            if element[0] % 2:
                if element[1]:
                    opstr += str(element[0]) + ' '
                else:
                    opstr += str(element[0]) + '^ '
            else:
                if element[1]:
                    opstr += str(element[0]) + '^ '
                else:
                    opstr += str(element[0]) + ' '
        newstr += FermionOperator(opstr, ops.terms[term])
    return newstr


def split_openfermion_tensor(ops: 'FermionOperator'
                            ) -> Tuple[Dict[int, 'FermionOperator'], complex]:
    """Given a string of openfermion operators, split them according to their
    rank.

    Args:
        ops (FermionOperator) - a string of Fermion Operators

    Returns:
        split dict[int] = FermionOperator - a list of Fermion Operators sorted
            according to their degree
    """
    e_0 = 0. + 0.j

    split: Dict[int, 'FermionOperator'] = {}

    for term in ops:
        rank = term.many_body_order()

        if rank % 2:
            raise ValueError('Odd rank term not accepted')

        if rank == 0:
            e_0 += term.terms[()]

        else:
            if rank not in split:
                split[rank] = term
            else:
                split[rank] += term

    return split, e_0


def fermionops_tomatrix(ops: 'FermionOperator', norb: int) -> numpy.ndarray:
    """Convert FermionOperators to matrix

    Args:
        ops (FermionOperator) - input FermionOperator from OpenFermion

        norb (int) - the number of orbitals in the system

    Returns:
        (numpy.ndarray) - resulting matrix
    """
    ablk, bblk = largest_operator_index(ops)

    if norb <= ablk // 2:
        raise ValueError('Highest alpha index exceeds the norb of orbitals')
    if norb <= bblk // 2:
        raise ValueError('Highest beta index exceeds the norb of orbitals')

    rank = ops.many_body_order()

    if rank % 2:
        raise ValueError('Odd rank operator not supported')

    tensor_dim = [norb * 2 for _ in range(rank)]
    index_mask = [0 for _ in range(rank)]

    index_dict_dagger = [[0, 0] for _ in range(rank // 2)]
    index_dict_nondagger = [[0, 0] for _ in range(rank // 2)]

    tensor = numpy.zeros(tensor_dim, dtype=numpy.complex128)

    for term in ops.terms:

        for i in range(rank):
            index = term[i][0]

            if i < rank // 2:
                if not term[i][1]:
                    raise ValueError('Found annihilation operator where' \
                                     'creation is expected')
            elif term[i][1]:
                raise ValueError('Found creattion operator where' \
                                 'annihilation is expected')

            spin = index % 2

            if spin == 1:
                ind = (index - 1) // 2 + norb
            else:
                ind = index // 2

            if i < rank // 2:
                index_dict_dagger[i][0] = spin
                index_dict_dagger[i][1] = ind
            else:
                index_dict_nondagger[i - rank // 2][0] = spin
                index_dict_nondagger[i - rank // 2][1] = ind

        parity = reverse_bubble_list(index_dict_dagger)
        parity += reverse_bubble_list(index_dict_nondagger)

        for i in range(rank):
            if i < rank // 2:
                index_mask[i] = index_dict_dagger[i][1]
            else:
                index_mask[i] = index_dict_nondagger[i - rank // 2][1]

        tensor[tuple(index_mask)] += (-1)**parity * ops.terms[term]

    return tensor


def process_rank2_matrix(mat: numpy.ndarray, norb: int,
                         e_0: complex = 0. + 0.j) -> 'hamiltonian.Hamiltonian':
    """Look at the structure of the (1, 0) component of the one body matrix and
    determine the symmetries.

    Args:
        mat (numpy.ndarray) - input matrix to be processed

        norb (int) - the number of orbitals in the system

        e_0 (copmlex) - scalar part of the Hamiltonian

    Returns:
        (Hamiltonian) - resulting Hamiltonian
    """
    if not numpy.allclose(mat, mat.conj().T):
        raise ValueError('Input matrix is not Hermitian')

    diagonal = True

    for i in range(1, max(norb, mat.shape[0])):
        for j in range(0, i):
            if mat[i, j] != 0. + 0.j:
                diagonal = False
                break

    if diagonal:
        return diagonal_hamiltonian.Diagonal(mat.diagonal(), e_0=e_0)

    if mat[norb:2 * norb, :norb].any():
        return gso_hamiltonian.GSOHamiltonian(tuple([mat]), e_0=e_0)

    if numpy.allclose(mat[:norb, :norb], mat[norb:, norb:]):
        return restricted_hamiltonian.RestrictedHamiltonian(tuple([mat]),
                                                            e_0=e_0)

    spin_mat = numpy.zeros((norb, 2 * norb), dtype=mat.dtype)
    spin_mat[:, :norb] = mat[:norb, :norb]
    spin_mat[:, norb:2 * norb] = mat[norb:, norb:]
    return sso_hamiltonian.SSOHamiltonian(tuple([spin_mat]), e_0=e_0)


def check_diagonal_coulomb(mat: numpy.ndarray) -> bool:
    """Look at the structure of the two body matrix and determine
    if it is diagonal coulomb

    Args:
        mat (numpy.ndarray) - input two-body Hamiltonian elements

    Returns:
        (bool) - whether mat is diagonal Coulomb
    """
    dim = mat.shape[0]
    assert mat.shape == (dim, dim, dim, dim)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    if i == k and j == l:
                        pass
                    elif mat[i, j, k, l] != 0. + 0.j:
                        return False
    return True


def wrap_rdm(rdm):
    """Decorator to convert arguments to the fqe internal classes
    """

    @wraps(rdm)
    def symmetry_process(self, string, brawfn=None):
        if self.conserve_spin() and not self.conserve_number():
            wfn = self._copy_beta_inversion()
        else:
            wfn = copy.deepcopy(self)

        if any(char.isdigit() for char in string):
            if self.conserve_spin() and not self.conserve_number():
                string = fqe_ops_utils.switch_broken_symmetry(string)

        return rdm(wfn, string, brawfn=brawfn)

    return symmetry_process


def wrap_apply(apply):
    """Decorator to convert arguments to the fqe internal classes
    """

    @wraps(apply)
    def convert(self, ops: Union['FermionOperator', 'hamiltonian.Hamiltonian']):
        """ Converts an FermionOperator to hamiltonian.Hamiltonian

        Args:
            ops (FermionOperator or Hamiltonian) - input operator
        """
        hamil = build_hamiltonian(ops, conserve_number=self.conserve_number())
        return apply(self, hamil)

    return convert


def wrap_time_evolve(time_evolve):
    """Decorator to convert arguments to the fqe internal classes
    """

    @wraps(time_evolve)
    def convert(self,
                time: float,
                ops: Union['FermionOperator', 'hamiltonian.Hamiltonian'],
                inplace: bool = False):
        """ Converts an FermionOperator to hamiltonian.Hamiltonian

        Args:
            time (float) - time to be propagated

            ops (FermionOperator or Hamiltonian) - input operator
        """
        hamil = build_hamiltonian(ops, conserve_number=self.conserve_number())
        return time_evolve(self, time, hamil, inplace)

    return convert


def wrap_apply_generated_unitary(apply_generated_unitary):
    """Decorator to convert arguments to the fqe internal classes
    """

    @wraps(apply_generated_unitary)
    def convert(self,
                time: float,
                algo: str,
                ops: Union['FermionOperator', 'hamiltonian.Hamiltonian'],
                accuracy: float = 0.0,
                expansion: int = 30,
                spec_lim: Optional[List[float]] = None):
        """Perform the exponentiation of fermionic algebras to the
        wavefunction according the method and accuracy.

        Args:
            time (float) - the final time value to evolve to

            algo (string) - polynomial expansion algorithm to be used

            hamil (Hamiltonian) - the Hamiltonian used to generate the unitary

            accuracy (double) - the accuracy to which the system should be evolved

            expansion (int) - the maximum number of terms in the polynomial expansion

            spec_lim (List[float]) - spectral range of the Hamiltonian, the length of \
                the list should be 2. Optional.

        Returns:
            newwfn (Wavefunction) - a new intialized wavefunction object
        """
        hamil = build_hamiltonian(ops, conserve_number=self.conserve_number())
        return apply_generated_unitary(self,
                                       time,
                                       algo,
                                       hamil,
                                       accuracy=accuracy,
                                       expansion=expansion,
                                       spec_lim=spec_lim)

    return convert
