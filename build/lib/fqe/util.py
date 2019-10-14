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

""" General Utilities
"""

from operator import itemgetter

import random

from scipy.special import binom
import numpy

from fqe.bitstring import lexicographic_bitstring_generator
from fqe.bitstring import check_conserved_bits
from fqe.string_addressing import count_bits


def alpha_beta_electrons(nele, m_s):
    """Given the total number of electrons and the z-spin quantum number, return
    the number of alpha and beta electrons in the system.

    Args:
        nele (int) - number of electrons
        m_s (int) - spin angular momentum on the z-axis

    Return:
        number of alpha electrons (int), number of beta electrons (int)
    """
    if nele < 0:
        raise ValueError('Cannot have negative electrons')
    if nele < abs(m_s):
        raise ValueError('Spin projection exceeds physical limits')
    nalpha = int(nele + m_s) // 2
    nbeta = nele - nalpha
    return nalpha, nbeta


def bubblesort(arr):
    """Bubble Sort algorithm to arrange a list so that the lowest value is
    stored in 0 and the highest value is stored in len(arr)-1.  It is included
    here in order to access the swap count

    Args:
        arr (list) - object to be sorted

    Returns:
        arr (list) - sorted
        swap_count (int) - number of permutations to achieve the sort
    """
    larr = len(arr)
    swap_count = 0
    for i in range(larr):
        swapped = False
        for j in range(0, larr-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
                swap_count += 1
        if not swapped:
            break
    return swap_count


def configuration_key_union(*argv):
    """Given a list of configuration keys, build a list which is the union of
    all configuration keys in the list

    Args:
        *args (list) - any number of configuration key lists to be joined

    Returns:
        list[(int, int)] - a list of unique configuration keys found among all
            the passed arguments
    """
    keyunion = set()
    for keys in argv:
        keyunion.update(keys)
    return list(keyunion)


def configuration_key_intersection(*argv):
    """Return the intersection of the passed configuration key lists.

    Args:
        *args (list) - any number of configuration key lists to be joined

    Returns:
        list [(int, int)] - a list of configuration keys found in every
            configuration passed.
    """
    keyinter = argv[0]
    ref = []
    for config in argv[1:]:
        for key in config:
            if key in keyinter:
                ref.append(key)
        keyinter = ref
    return keyinter


def init_bitstring_groundstate(occ_num):
    """Occupy the n lowest orbitals of a state in the bitstring representation

    Args:
        occ_num (integer) - number of orbitals to occupy

    Returns:
        (integer) - bitstring representation of the ground state
    """
    return (1<<occ_num)-1


def init_qubit_vacuum(nqubits):
    """Build the ground state wavefunction for an nqubit system.

    Args:
        nqubits (integer) - The number of qubits in the qpu

    Returns:
        numpy.array(dtype=numpy.complex64)
    """
    _gs = numpy.zeros(2**nqubits, dtype=numpy.complex64)
    _gs[0] = 1.0 + 0.0j
    return _gs


def ltlt_index_generator(dim):
    """Generate index sets into a lower triangle, lower triangle matrix

    Args:
        dim (int) - the dimension of the array

    Returns:
        int, int, int, int - unique pointers into the compressed matrix
    """
    lim = dim
    for i in range(lim):
        for j in range(i+1):
            for k in range(i+1):
                if k == i:
                    _ull = j+1
                else:
                    _ull = k+1
                for lst in range(_ull):
                    yield i, j, k, lst


def invert_bitstring_with_mask(string, masklen):
    """Invert a bitstring with a mask.

    Args:
        string (bitstring) - the bitstring to invert
        masklen (int) - the value to mask the inverted bitstring to

    Returns:
        (bitstring) - a bitstring inverted up to the masking length
    """
    mask = (1 << masklen) - 1
    return ~string & mask


def paritysort(arr):
    """Move all even numbers to the left and all odd numbers to the right

    Args:
        arr list[int] - a list of integers to be sorted

    Returns:
        arr [list] - mutated in place
        swap_count (int) - number of exchanges needed to complete the sorting
    """
    larr = len(arr)
    _parr = [[i % 2, i] for i in arr]

    swap_count = 0
    for i in range(larr):
        swapped = False
        for j in range(0, larr-i-1):
            if _parr[j][0] > _parr[j+1][0]:
                _parr[j], _parr[j+1] = _parr[j+1], _parr[j]
                swapped = True
                swap_count += 1
        if not swapped:
            break

    for indx, val in enumerate(_parr):
        arr[indx] = val[1]

    return swap_count


def qubit_particle_number_sector(nqubits, pnum):
    """Generate the basis vectors into the qubit basis representing all states
    which have a definite particle number.

    Args:
        nqubits (int) - the number of qubits in the qpu
        pnum (int) - the number of particles to build vectors into

    Returns:
        list[numpy.array(dtype=nmpy.complex64)]
    """
    occ = numpy.array([0, 1], dtype=numpy.int)
    uno = numpy.array([1, 0], dtype=numpy.int)
    seed = init_bitstring_groundstate(pnum)
    pn_set = lexicographic_bitstring_generator(seed, nqubits)
    vectors = []
    for orbocc in pn_set:
        if orbocc & 1:
            vec = occ
        else:
            vec = uno
        orbocc = orbocc>>1
        for _ in range(nqubits-1):
            if orbocc & 1:
                vec = numpy.kron(vec, occ)
            else:
                vec = numpy.kron(vec, uno)
            orbocc = orbocc>>1
        vectors.append(vec)
    return vectors


def qubit_config_sector(nqubits, pnum, m_s):
    """Generate the basis vectors into the qubit basis representing all states
    which have a definite particle number and spin.

    Args:
        nqubits (int) - the number of qubits in the qpu
        pnum (int) - the number of particles to build vectors into
        m_s (int) - the s_z spin quantum number

    Returns:
        list[numpy.array(dtype=nmpy.complex64)]
    """
    occ = numpy.array([0, 1], dtype=numpy.int)
    uno = numpy.array([1, 0], dtype=numpy.int)
    seed = init_bitstring_groundstate(pnum)
    achk = 0
    bchk = 0
    pn_set = []

    for num in range(nqubits):
        if num % 2:
            bchk += 2**num
        else:
            achk += 2**num

    initpn = lexicographic_bitstring_generator(seed, nqubits)
    for occu in initpn:
        if (count_bits(occu & achk) - count_bits(occu & bchk)) == m_s:
            pn_set.append(occu)

    vectors = []
    for orbocc in pn_set:
        if orbocc & 1:
            vec = occ
        else:
            vec = uno
        orbocc = orbocc>>1
        for _ in range(nqubits-1):
            if orbocc & 1:
                vec = numpy.kron(vec, occ)
            else:
                vec = numpy.kron(vec, uno)
            orbocc = orbocc>>1
        vectors.append(vec)
    return vectors


def qubit_particle_number_index(nqubits, pnum):
    """Generate indexes corresponding to the coefficient that is associated
    with a specific particle number

    Args:
        nqubits (int) - the number of qubits to act upon
        pnum (int) - the number of particles to view

    Returns:
        list[int] - integers indicating where in the qubit wavefunction the
            basis state corresponds to particle number
    """
    seed = init_bitstring_groundstate(pnum)
    indexes = []
    pn_set = lexicographic_bitstring_generator(seed, nqubits)
    for orbocc in pn_set:
        if orbocc & 1:
            index = 1
        else:
            index = 0
        orbocc = orbocc>>1
        veclen = 2
        for _ in range(nqubits-1):
            if orbocc & 1:
                index = index + veclen
            orbocc = orbocc >> 1
            veclen *= 2
        indexes.append(index)
    return indexes


def qubit_particle_number_index_spin(nqubits, pnum):
    """Generate indexes corresponding to the coefficient that is associated
    with a specific particle number and spin

    Args:
        nqubits (int) - the number of qubits to act upon
        pnum (int) - the number of particles to view

    Returns:
        list[(int int)] - tuples of integers indicating where in the qubit
            wavefunction the basis state corresponds to particle number and
            return the corresponding spin
    """
    seed = init_bitstring_groundstate(pnum)
    indexes = []
    pn_set = lexicographic_bitstring_generator(seed, nqubits)
    for orbocc in pn_set:
        totspn = 0
        curspn = 1
        if orbocc & 1:
            index = 1
            totspn += curspn
        else:
            index = 0
        orbocc = orbocc>>1
        veclen = 2
        for _ in range(nqubits-1):
            curspn *= -1
            if orbocc & 1:
                index = index + veclen
                totspn += curspn
            orbocc = orbocc >> 1
            veclen *= 2
        indexes.append((index, totspn))
    return indexes


def rand_wfn(dim, sparse='none'):
    """Utility for generating random normalized wavefunctions.

    Args:
        dim (int) - length of the wavefunction
        sparse (string) - a string indiicating some approximate measure of how
            often zeros should show up in the wavefunction

    Returns:
        numpy.array(dim=dim, dtype=numpy.complex64)
    """
    wfn = numpy.random.randn(dim) + numpy.random.randn(dim)*1.j
    if sparse == 'low':
        nzero = max(int(dim*(0.3 - numpy.random.ranf()*0.15)), 1)
        zind = random.randint(0, nzero)
        wfn[zind] = 0. + .0j

    if sparse == 'med':
        nzero = max(int(dim*(0.65 - numpy.random.ranf()*0.25)), 1)
        zind = random.randint(0, nzero)
        wfn[zind] = 0. + .0j

    if sparse == 'high':
        nzero = max(int(dim*(0.85 - numpy.random.ranf()*0.15)) - 1, dim//2)
        zind = random.randint(0, nzero)
        wfn[zind] = 0. + .0j

    norm = numpy.sqrt(numpy.vdot(wfn, wfn))
    wfn /= norm
    return wfn


def sort_configuration_keys(configs):
    """Return a standard sorting of configuration keys in a wavefunction for
    comparison.  The configurations are sorted first by the number of particles
    and then by the spin quantum number.

    Args:
        wfn list[(dict.keys)] - a dictionary of keys

    Returns:
        list with the sorted keys
    """
    return sorted(configs, key=itemgetter(0, 1))


def validate_config(nalpha, nbeta, norb):
    """ Check that the parameters passed are valid to build a configuration

    Args:
        nalpha (int) - the number of alpha electrons
        nbeta (int) - the number of beta electrons
        norb (int) - the number of spatial orbitals

    Returns:
        nothing - only raises errors if necessary
    """
    if nalpha < 0:
        raise ValueError("Cannot have negative number of alpha electrons")

    if nbeta < 0:
        raise ValueError("Cannot have negative number of beta electrons")

    if norb < 0:
        raise ValueError("Cannot have negative number of orbitals")

    if norb < nalpha or norb < nbeta:
        raise ValueError("Insufficient number of orbitals")


def weyl_paldus(nele, m_s, norb):
    """Weyl-Paldus formula for calculating the number of CSF necessary for a
    configuration.

    Args:
        m_s (int) - the 2*Sz spin quantum number
        norb (int) - the number of spatial orbital
        nele (int) - the number of electrons

    Returns:
        ncsf (int) - the number of configuration state functions that describe
            the CI space
    """
    ams = abs(m_s)
    ncsf = binom(norb + 1, (nele - ams)//2)
    ncsf *= binom(norb + 1, norb - (nele + ams)//2)
    ncsf *= (ams + 1)
    ncsf /= (norb + 1)
    return int(ncsf)


def zero_transform(string0, unocc, occ, norb):
    """Given a bitstring, determine if it satisfies the occupation and
    nonoccupation conditions necessary to be non zero when a product of creation
    and annihilation operators are applied.

    Args:
        string0 (bitstring) - the occupation representation being acted upon
        unocc (bitstring) - orbitals which should be unoccupied in string0
        occ (bitstring) - orbitals which should be occupied in string0
        norb (int) - the number of spatial orbitals for masking the bitstrings

    Returns:
        (bool) - true if the transformation is non zero, false if the
            transformation is zero
    """
    if check_conserved_bits(string0, occ):
        if check_conserved_bits(invert_bitstring_with_mask(string0, norb),
                                unocc):
            return False

    return True
