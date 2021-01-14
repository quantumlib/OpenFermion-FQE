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
"""Contains definitions for several Hamiltonian constructor routines."""

# pylint does not like zeros_like initializer
# pylint: disable=unsupported-assignment-operation
# pylint: disable=invalid-sequence-index

# more descriptive index names for iterators are not necessary
# pylint: disable=invalid-name

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from fqe.util import paritysort_list, reverse_bubble_list

Newop = Tuple[complex, int, List[Tuple[int, int]], List[Tuple[int, int]]]

if TYPE_CHECKING:
    from openfermion import FermionOperator


def antisymm_two_body(h2e: np.ndarray) -> np.ndarray:
    """Given a two body matrix, perform antisymmeterization on the elements.

    Args:
        h2e: Input 2-body tensor.

    Returns:
        Output 2-body tensor.
    """
    tmp = np.zeros_like(h2e)
    dim = h2e.shape[0]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    tmp[i, j, k, m] = (h2e[i, j, k, m] - h2e[j, i, k, m] -
                                       h2e[i, j, m, k] + h2e[j, i, m, k]) * 0.25
    return tmp


def antisymm_three_body(h3e: np.ndarray) -> np.ndarray:
    """Given a three body matrix, perform antisymmeterization on the elements.

    Args:
        h3e: Input 3-body tensor.

    Returns:
        Output 3-body tensor.
    """
    tmp = np.zeros_like(h3e)
    dim = h3e.shape[0]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                tmp[i, j, k, :, :, :] = (
                    h3e[i, j, k, :, :, :] - h3e[j, i, k, :, :, :] -
                    h3e[i, k, j, :, :, :] - h3e[k, j, i, :, :, :] +
                    h3e[k, i, j, :, :, :] + h3e[j, k, i, :, :, :]) / 6.0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                h3e[:, :, :, i, j, k] = (
                    tmp[:, :, :, i, j, k] - tmp[:, :, :, j, i, k] -
                    tmp[:, :, :, i, k, j] - tmp[:, :, :, k, j, i] +
                    tmp[:, :, :, k, i, j] + tmp[:, :, :, j, k, i]) / 6.0
    return h3e


def antisymm_four_body(h4e: np.ndarray) -> np.ndarray:
    """Given a four body matrix, perform antisymmeterization on the elements.

    Args:
        h4e: Input 4-body tensor.

    Returns:
        Output 4-body tensor.
    """
    tmp = np.zeros_like(h4e)
    dim = h4e.shape[0]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    tmp[i, j, k, m, :, :, :, :] = (
                        (h4e[i, j, k, m, :, :, :, :] -
                         h4e[i, j, m, k, :, :, :, :] +
                         h4e[i, m, j, k, :, :, :, :] -
                         h4e[i, m, k, j, :, :, :, :] -
                         h4e[i, k, j, m, :, :, :, :] +
                         h4e[i, k, m, j, :, :, :, :]) -
                        (h4e[j, i, k, m, :, :, :, :] -
                         h4e[j, i, m, k, :, :, :, :] +
                         h4e[j, m, i, k, :, :, :, :] -
                         h4e[j, m, k, i, :, :, :, :] -
                         h4e[j, k, i, m, :, :, :, :] +
                         h4e[j, k, m, i, :, :, :, :]) -
                        (h4e[k, j, i, m, :, :, :, :] -
                         h4e[k, j, m, i, :, :, :, :] +
                         h4e[k, m, j, i, :, :, :, :] -
                         h4e[k, m, i, j, :, :, :, :] -
                         h4e[k, i, j, m, :, :, :, :] +
                         h4e[k, i, m, j, :, :, :, :]) -
                        (h4e[m, j, k, i, :, :, :, :] -
                         h4e[m, j, i, k, :, :, :, :] +
                         h4e[m, i, j, k, :, :, :, :] -
                         h4e[m, i, k, j, :, :, :, :] -
                         h4e[m, k, j, i, :, :, :, :] +
                         h4e[m, k, i, j, :, :, :, :])) / 24.0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    h4e[:, :, :, :, i, j, k, m] = (
                        (tmp[:, :, :, :, i, j, k, m] -
                         tmp[:, :, :, :, i, j, m, k] +
                         tmp[:, :, :, :, i, m, j, k] -
                         tmp[:, :, :, :, i, m, k, j] -
                         tmp[:, :, :, :, i, k, j, m] +
                         tmp[:, :, :, :, i, k, m, j]) -
                        (tmp[:, :, :, :, j, i, k, m] -
                         tmp[:, :, :, :, j, i, m, k] +
                         tmp[:, :, :, :, j, m, i, k] -
                         tmp[:, :, :, :, j, m, k, i] -
                         tmp[:, :, :, :, j, k, i, m] +
                         tmp[:, :, :, :, j, k, m, i]) -
                        (tmp[:, :, :, :, k, j, i, m] -
                         tmp[:, :, :, :, k, j, m, i] +
                         tmp[:, :, :, :, k, m, j, i] -
                         tmp[:, :, :, :, k, m, i, j] -
                         tmp[:, :, :, :, k, i, j, m] +
                         tmp[:, :, :, :, k, i, m, j]) -
                        (tmp[:, :, :, :, m, j, k, i] -
                         tmp[:, :, :, :, m, j, i, k] +
                         tmp[:, :, :, :, m, i, j, k] -
                         tmp[:, :, :, :, m, i, k, j] -
                         tmp[:, :, :, :, m, k, j, i] +
                         tmp[:, :, :, :, m, k, i, j])) / 24.0
    return h4e


def gather_nbody_spin_sectors(operators: 'FermionOperator') -> Newop:
    """Given an nbody FermionOperator string, split it into alpha
    and beta spin sectors. This routine assumes that there are equal
    numbers of creation and annihilation operators and that they are
    passed in {creation}{annihilation} order.

    Args:
        operators: Operators in the FermionOperator format.
    """
    # Get the indices of the elements
    nalpha = 0
    for opstr in operators.terms:
        indexes = list(opstr)
        nswaps, indexes = paritysort_list(indexes)
        nda = 0
        ndb = 0
        nua = 0
        nub = 0
        for i in indexes:
            if i[0] % 2 == 0 and i[1] == 1:
                nda += 1
            elif i[0] % 2 == 0 and i[1] == 0:
                nua += 1
            elif i[0] % 2 == 1 and i[1] == 1:
                ndb += 1
            elif i[0] % 2 == 1 and i[1] == 0:
                nub += 1

        assert nda + ndb + nua + nub == len(indexes)
        nalpha = nda + nua

        coeff = operators.terms[opstr]

        ablock = indexes[:nalpha]
        nswaps += reverse_bubble_list(ablock[:nda])
        nswaps += reverse_bubble_list(ablock[nda:])
        bblock = indexes[nalpha:]
        nswaps += reverse_bubble_list(bblock[:ndb])
        nswaps += reverse_bubble_list(bblock[ndb:])

    return coeff, (-1)**nswaps, indexes[:nalpha], indexes[nalpha:]


def nbody_matrix(ops: 'FermionOperator', norb: int) -> np.ndarray:
    """Parse the creation and annihilation operators and return a sparse matrix
    with the elements of the matrix filled with the convention

    .. code-block::

            i^j^k^    o p q
            1 2 3 ... 1 2 3 ...

    Args:
        ops: Matrix in the FermionOperator format.
        norb: The number of orbitals in the system.
    """
    orb_ptr = norb
    orbdim = 2 * norb

    for prod in ops.terms:
        number_operators = len(prod)
        mat_dim = [orbdim for _ in range(number_operators)]
        # TODO: Should `nbodymat` be defined outside the loop?
        nbodymat = np.zeros(mat_dim, dtype=np.complex128)
        mat_ele = [((ele[0] - 1) // 2) + orb_ptr if ele[0] % 2 else ele[0] // 2
                   for ele in prod]
        ele = tuple(mat_ele)
        con = tuple(reversed(mat_ele))
        sval = complex(ops.terms[prod])
        nbodymat[ele] += sval
        nbodymat[con] += sval.conjugate()

    return nbodymat
