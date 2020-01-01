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
"""Hamiltonian constructor routines
"""

from typing import Tuple

import numpy

from openfermion import FermionOperator
from openfermion.utils import is_hermitian

from fqe.openfermion_utils import generate_one_particle_matrix
from fqe.util import paritysort_list, reverse_bubble_list


def antisymm_two_body(h2e):
    """Given a two body matrix perform antisymmeterization on the elements.
    """
    tmp = numpy.zeros_like(h2e)
    dim = h2e.shape[0]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    tmp[i,j,k,l] = (h2e[i, j, k, l] - h2e[j, i, k, l] \
                                    - h2e[i, j, l, k] + h2e[j, i, l, k]) * 0.25
    return tmp


def antisymm_three_body(h3e):
    """Given a two body matrix perform antisymmeterization on the elements.
    """
    tmp = numpy.zeros_like(h3e)
    dim = h3e.shape[0]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                tmp[i, j, k, :, :, :] = (h3e[i, j, k, :, :, :] \
                                         - h3e[j, i, k, :, :, :] \
                                         - h3e[i, k, j, :, :, :] \
                                         - h3e[k, j, i, :, :, :] \
                                         + h3e[k, i, j, :, :, :] \
                                         + h3e[j, k, i, :, :, :])/6.0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                h3e[:, :, :, i, j, k] = (tmp[:, :, :, i, j, k] \
                                         - tmp[:, :, :, j, i, k] \
                                         - tmp[:, :, :, i, k, j] \
                                         - tmp[:, :, :, k, j, i] \
                                         + tmp[:, :, :, k, i, j] \
                                         + tmp[:, :, :, j, k, i])/6.0
    return h3e


def antisymm_four_body(h4e):
    """Given a two body matrix perform antisymmeterization on the elements.
    """
    tmp = numpy.zeros_like(h4e)
    dim = h4e.shape[0]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    tmp[i, j, k, l, :, :, :, :] = ((h4e[i,j,k,l,:,:,:,:] \
                                                    - h4e[i,j,l,k,:,:,:,:] \
                                                    + h4e[i,l,j,k,:,:,:,:] \
                                                    - h4e[i,l,k,j,:,:,:,:] \
                                                    - h4e[i,k,j,l,:,:,:,:] \
                                                    + h4e[i,k,l,j,:,:,:,:]) \
                                                   - (h4e[j,i,k,l,:,:,:,:] \
                                                      - h4e[j,i,l,k,:,:,:,:] \
                                                      + h4e[j,l,i,k,:,:,:,:] \
                                                      - h4e[j,l,k,i,:,:,:,:] \
                                                      - h4e[j,k,i,l,:,:,:,:] \
                                                      + h4e[j,k,l,i,:,:,:,:])\
                                                   - (h4e[k,j,i,l,:,:,:,:] \
                                                      - h4e[k,j,l,i,:,:,:,:] \
                                                      + h4e[k,l,j,i,:,:,:,:] \
                                                      - h4e[k,l,i,j,:,:,:,:] \
                                                      - h4e[k,i,j,l,:,:,:,:] \
                                                      + h4e[k,i,l,j,:,:,:,:])\
                                                   - (h4e[l,j,k,i,:,:,:,:] \
                                                      - h4e[l,j,i,k,:,:,:,:] \
                                                      + h4e[l,i,j,k,:,:,:,:] \
                                                      - h4e[l,i,k,j,:,:,:,:] \
                                                      - h4e[l,k,j,i,:,:,:,:] \
                                                      + h4e[l,k,i,j,:,:,:,:])
                                                  )/24.0
    h4e = tmp
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    tmp[:, :, :, :, i, j, k, l] = ((h4e[:,:,:,:,i,j,k,l] \
                                                    - h4e[:,:,:,:,i,j,l,k] \
                                                    + h4e[:,:,:,:,i,l,j,k] \
                                                    - h4e[:,:,:,:,i,l,k,j] \
                                                    - h4e[:,:,:,:,i,k,j,l] \
                                                    + h4e[:,:,:,:,i,k,l,j]) \
                                                   - (h4e[:,:,:,:,j,i,k,l] \
                                                      - h4e[:,:,:,:,j,i,l,k] \
                                                      + h4e[:,:,:,:,j,l,i,k] \
                                                      - h4e[:,:,:,:,j,l,k,i] \
                                                      - h4e[:,:,:,:,j,k,i,l] \
                                                      + h4e[:,:,:,:,j,k,l,i]) \
                                                   - (h4e[:,:,:,:,k,j,i,l] \
                                                      - h4e[:,:,:,:,k,j,l,i] \
                                                      + h4e[:,:,:,:,k,l,j,i] \
                                                      - h4e[:,:,:,:,k,l,i,j] \
                                                      - h4e[:,:,:,:,k,i,j,l] \
                                                      + h4e[:,:,:,:,k,i,l,j]) \
                                                   - (h4e[:,:,:,:,l,j,k,i] \
                                                      - h4e[:,:,:,:,l,j,i,k] \
                                                      + h4e[:,:,:,:,l,i,j,k] \
                                                      - h4e[:,:,:,:,l,i,k,j] \
                                                      - h4e[:,:,:,:,l,k,j,i] \
                                                      + h4e[:,:,:,:,l,k,i,j])
                                                   )/24.0
    return h4e
    


def nbody_process(operator):
    """Perform checks on an individual nbody operator to determine the proper
    course of action.
    """
    if len(operator.terms) == 0:
        return True

    elif len(operator.terms) == 2:
        if not is_hermitian(operator):
            raise ValueError('n-body operator is not Hermitian')

    elif len(operator.terms) > 2:
        raise ValueError('More than one n-body term passed')
   
    term = list(operator.terms)[0]

    if len(term) % 2:
        raise ValueError('Odd number of operators')

    nbody = len(term) // 2

    creation_list = []
    annihilation_list = []

    for opi in range(nbody):

        if not term[opi][1]:
            raise ValueError('Annihilation found where creation should be')
        cre_ind = term[opi][0]

        if term[opi + nbody][1]:
            raise ValueError('Creation found where annhilation should be')
        ani_ind = term[opi + nbody][1]

        if cre_ind in creation_list:
            return True
        creation_list.append(term[opi][1])

        if ani_ind in annihilation_list:
            return True
        annihilation_list.append(term[opi + nbody][1])

    return False


def gather_nbody_spin_sectors(operators):
    """Given an nbody fermionoperator string, split it into alpha
    and beta spin sectors.  This routine assumes that there are equal
    numbers of creation and annihilation operators and that they are
    passed in {creation}{annihilation} order.
    """
    # Get the indices of the elements
    nalpha = 0
    for opstr in operators.terms:
        indexes = [op for op in opstr]
        nswaps, indexes = paritysort_list(indexes)
        nda = 0
        ndb = 0
        nua = 0
        nub = 0
        for i in indexes:
            if   i[0] % 2 == 0 and i[1] == 1: nda += 1
            elif i[0] % 2 == 0 and i[1] == 0: nua += 1
            elif i[0] % 2 == 1 and i[1] == 1: ndb += 1
            elif i[0] % 2 == 1 and i[1] == 0: nub += 1

        assert nda+ndb+nua+nub == len(indexes)
        nalpha = nda + nua

        coeff = operators.terms[opstr]

        ablock = indexes[:nalpha]
        nswaps += reverse_bubble_list(ablock[:nda])
        nswaps += reverse_bubble_list(ablock[nda:])
        bblock = indexes[nalpha:]
        nswaps += reverse_bubble_list(bblock[:ndb])
        nswaps += reverse_bubble_list(bblock[ndb:])

    return coeff, (-1)**nswaps, indexes[:nalpha], indexes[nalpha:]


def one_body_construct(onebody, norb, precision = 1.e-7):
    """Parse an object passed into the routine and return numpy objects along
    with symmetry or hamiltonian information.

    """
    h1a = numpy.empty(0)
    h1b = numpy.empty(0)
    h1bc = numpy.empty(0)

    if isinstance(onebody, FermionOperator):
        h1a, h1b, h1bc = generate_one_particle_matrix(split[2])

    elif isinstance(onebody, tuple) or isinstance(onebody, list):
        # If a tuple/list is passed assume we enforce the order ha, hb, hb_conjugated
        if len(onebody) > 0:
            h1a = onebody[0]

        if len(onebody) > 1:
            h1b = onebody[1]

        if len(onebody) > 2:
            h1bc = onebody[2]

    elif isinstance(onebody, dict):
        try:
            h1a = onebody[(1, 0)]
        except KeyError:
            pass

        try:
            h1b = onebody[(1, 1)]
        except KeyError:
            pass

        try:
            h1bc = onebody[(0, 0)]
        except KeyError:
            pass

    elif isinstance(onebody, numpy.ndarray):
        h1a = onebody


def generate_one_particle_matrix(ops: 'FermionOperator'
                                ) -> Tuple[numpy.ndarray]:
    """Convert a string of FermionOperators into a matrix.  If the dimension
    is not passed we will search the string to find the largest value.

    Args:
        ops (FermionOperator) - a string of FermionOperators

    Returns:
        Tuple(numpy.array(dtype=numpy.complex64),
              numpy.array(dtype=numpy.complex64),
              numpy.array(dtype=numpy.complex64))
    """
    ablk, bblk = largest_operator_index(ops)
    dim = max(2*ablk, 2*bblk)
    ablk = dim //2
    h1a = numpy.zeros((dim, dim), dtype=numpy.complex64)
    h1b = numpy.zeros((dim, dim), dtype=numpy.complex64)
    h1b_conj = numpy.zeros((dim, dim), dtype=numpy.complex64)
    for term in ops.terms:

        left, right = term[0][0], term[1][0]

        if left % 2:
            ind = (left - 1)//2 + ablk
        else:
            ind = left // 2

        if right % 2:
            jnd = (right - 1)//2 + ablk
        else:
            jnd = right // 2

        if term[0][1] and term[1][1]:
            h1b[ind, jnd] += ops.terms[term]
        elif term[0][1] and not term[1][1]:
            h1a[ind, jnd] += ops.terms[term]
        else:
            h1b_conj[ind, jnd] += ops.terms[term]

    return h1a, 2.0*h1b, 2.0*h1b_conj


def generate_two_particle_matrix(ops: 'FermionOperator') -> numpy.ndarray:
    """Convert a string of FermionOperators into a matrix.  If the operators
    in the term do not fit the action form of (1, 1, 0, 0) then permute them
    to git the form

    Args:
        ops (FermionOperator) - a string of FermionOperators

    Returns:
        numpy.array(dtype=numpy.complex64)
    """
    ablk, bblk = largest_operator_index(ops)
    dim = max(2*ablk, 2*bblk)
    ablk = dim // 2
    g2e = numpy.zeros((dim, dim, dim, dim), dtype=numpy.complex64)
    for term in ops.terms:

        iact, jact, kact, lact = term[0][1], term[1][1], term[2][1], term[3][1]

        if iact + jact + kact + lact != 2:
            raise ValueError('Unsupported four index tensor')

        first, second, third, fourth = term[0][0], term[1][0], term[2][0], term[3][0] 

        nper = 0
        if not iact:
            if not jact:
                jact, kact = kact, jact
                second, third = third, second
                nper += 1
            iact, jact = jact, iact
            first, second = second, first
            nper += 1

        if not jact:
            if not kact:
                kact, lact = lact, kact
                third, fourth = fourth, third
                nper += 1
            jact, kact = kact, jact
            second, third = third, second
            nper += 1

        if first % 2:
            ind = (first - 1)//2 + ablk
        else:
            ind = first // 2

        if second % 2:
            jnd = (second - 1)//2 + ablk
        else:
            jnd = second // 2

        if third % 2:
            knd = (third - 1)//2 + ablk
        else:
            knd = third // 2

        if fourth % 2:
            lnd = (fourth - 1)//2 + ablk
        else:
            lnd = fourth // 2

        g2e[ind, jnd, knd, lnd] = ((-1.0)**nper)*ops.terms[term]

    return g2e


def nbody_matrix(ops, norb, spinsum=False):
    """Parse the creation and annihilation operators and return a sparse matrix
    with the elements of the matrix filled with the convention
            i^j^k^    o p q
            1 2 3 ... 1 2 3 ...
    """
    if spinsum:
        orb_ptr = 0
        orbdim = norb
    else:
        orb_ptr = norb
        orbdim = 2*norb

    for prod in ops.terms:
        number_operators = len(prod)
        mat_dim = [orbdim for _ in range(number_operators)]
        nbodymat = numpy.zeros(mat_dim, dtype=numpy.complex128)
        mat_ele = [((ele[0] - 1) // 2) + orb_ptr if ele[0] % 2 else ele[0] // 2 for ele in prod]
        ele = tuple(mat_ele)
        con = tuple(reversed(mat_ele))
        sval = complex(ops.terms[prod])
        nbodymat[ele] += sval
        nbodymat[con] += sval.conjugate()

    return nbodymat


def process_1_0_matrix(mat, norb):
    """Look at the spin blocks of the (1, 0) matrix and return the symmetry
    information.
    """
    if not numpy.allclose(mat, mat.conj().T):
        diff = abs(mat - mat.conj().T)
        i, j = numpy.unravel_index(diff.argmax(), diff.shape)
        print('Element {} {} outside tolerance'.format(i, j))
        print('{} != {} '.format(mat[i, j], mat[j, i].conj()))
        raise ValueError

    spin_conserve = True
    restricted = False

    if 2*norb == mat.shape[0]:
        if numpy.allclose(h1a[:norb, :norb], h1a[norb:, norb:]):
            restricted = True

        if h1a[norb:, :norb].any():
            spin_conserve = False

        else: # check for diagonal matrix
            diagonal = True
            for i in range(1, 2*norb):
                for j in range(1, i):
                    if numpy.abs(h1a[j, i]) > 1.e-8:
                        diagonal = False
                        break

    if norb == mat.shape[0]:
        restricted = True


if __name__ == '__main__':
#    indexes =[
#        [7, 1],
#        [6, 0]
#        ]
#    print(indexes)
#    nswap = paritysort(indexes)
#    print(nswap)
#    print(indexes)
#    print('\n')
#
#    indexes =[
#        [6, 1],
#        [3, 1],
#        [4, 0],
#        [1, 0]
#        ]
#
#    print(indexes)
#    nswap = paritysort(indexes)
#    print(nswap)
#    print(indexes)
#    print('\n')
#
#    indexes =[
#        [7, 1],
#        [6, 1],
#        [2, 1],
#        [1, 1],
#        [2, 0],
#        [6, 0],
#        [10, 0],
#        [3, 0],
#        ]
#    print(indexes)
#    nswap = paritysort(indexes)
#    print(nswap)
#    print(indexes)

#    ops = FermionOperator('2^ 6^ 3^ 5^ 7 1 6 10', 1.0)
#    print(ops)
#    out = gather_nbody_spin_sectors(ops)
#    print(out)
    ops = FermionOperator('0^ 0', 1.0)
    print(ops)
    out = gather_nbody_spin_sectors(ops)
    print(out)
