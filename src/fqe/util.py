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
""" General Utilities
"""

from operator import itemgetter
from typing import Any, Dict, KeysView, List, Set, Tuple, TYPE_CHECKING

import numpy

from fqe.bitstring import lexicographic_bitstring_generator
from fqe.bitstring import count_bits

if TYPE_CHECKING:
    from fqe import wavefunction


def alpha_beta_electrons(nele: int, m_s: int) -> Tuple[int, int]:
    """Given the total number of electrons and the z-spin quantum number, return
    the number of alpha and beta electrons in the system.

    Args:
        nele (int): number of electrons

        m_s (int): spin angular momentum on the z-axis

    Return:
        Tuple[int, int]: number of alpha and beta electrons
    """
    if nele < 0:
        raise ValueError('Cannot have negative electrons')
    if nele < abs(m_s):
        raise ValueError('Spin quantum number exceeds physical limits')
    if (nele + m_s) % 2 != 0:
        raise ValueError('Parity of spin quantum number and number of '
                         'electrons is incompatible')
    nalpha = int(nele + m_s) // 2
    nbeta = nele - nalpha
    return nalpha, nbeta


def reverse_bubble_list(arr: List[List[Any]]) -> int:
    """Bubble Sort algorithm to arrange a list so that the highest value is
    stored in 0 and the lowest value is stored in len(arr)-1. It is included
    here in order to access the swap count.

    Args:
        arr (list): List of objects to be sorted. The objects are sorted with \
            respect to the value of object[0]. The data are sorted inplace.

    Returns:
        (int): number of permutations to achieve the sort
    """
    larr = len(arr)
    swap_count = 0
    for i in range(larr):
        swapped = False
        for j in range(0, larr - i - 1):
            if arr[j][0] < arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                swap_count += 1

        if not swapped:
            break

    return swap_count


def bubblesort(arr: List[Any]) -> int:
    """Bubble Sort algorithm to arrange a list so that the lowest value is
    stored in 0 and the highest value is stored in len(arr)-1.  It is included
    here in order to access the swap count.

    Args:
        arr (list): object to be sorted in place.

    Returns:
        (int): number of permutations to achieve the sort
    """
    larr = len(arr)
    swap_count = 0
    for i in range(larr):
        swapped = False
        for j in range(0, larr - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                swap_count += 1
        if not swapped:
            break
    return swap_count


def configuration_key_union(*argv: KeysView[Tuple[int, int]]
                           ) -> List[Tuple[int, int]]:
    """Given a list of configuration keys, build a list which is the union of
    all configuration keys in the list

    Args:
        *args (list[(int, int)]): any number of configuration key lists to be \
            joined

    Returns:
        list[(int, int)]: a list of unique configuration keys found among all \
            the passed arguments
    """
    keyunion: Set[Tuple[int, int]] = set()
    for configs in argv:
        keyunion.update(configs)
    return list(keyunion)


def configuration_key_intersection(*argv: List[Tuple[int, int]]
                                  ) -> List[Tuple[int, int]]:
    """Return the intersection of the passed configuration key lists.

    Args:
        *args (list[(int, int)]): any number of configuration key lists to be \
            intersected

    Returns:
        list[(int, int)]: a list of configuration keys found in every \
            configuration passed.
    """
    ref = set(argv[0])
    for config in argv[1:]:
        ref = ref.intersection(config)
    return list(ref)


def init_bitstring_groundstate(occ_num: int) -> int:
    """Occupy the n lowest orbitals of a state in the bitstring representation

    Args:
        occ_num (integer): number of orbitals to occupy

    Returns:
        (integer): bitstring representation of the ground state
    """
    return (1 << occ_num) - 1


def init_qubit_vacuum(nqubits: int) -> numpy.ndarray:
    """Build the ground state wavefunction for an nqubit system.

    Args:
        nqubits (integer): The number of qubits in the qpu

    Returns:
        numpy.array(dtype=numpy.complex128): ground state in the qubit \
            representation
    """
    ground_state = numpy.zeros(2**nqubits, dtype=numpy.complex128)
    ground_state[0] = 1.0 + 0.0j
    return ground_state


def paritysort_int(arr: List[int]) -> Tuple[int, List[int]]:
    """Move all even numbers to the left and all odd numbers to the right

    Args:
        arr (list[int]): a list of integers to be sorted in place.

    Returns:
        (int): number of exchanges needed to complete the sorting
    """
    larr = len(arr)
    parr = [[i % 2, i] for i in arr]

    swap_count = 0
    for i in range(larr):
        swapped = False
        for j in range(0, larr - i - 1):
            if parr[j][0] > parr[j + 1][0]:
                parr[j], parr[j + 1] = parr[j + 1], parr[j]
                swapped = True
                swap_count += 1
        if not swapped:
            break

    for indx, val in enumerate(parr):
        arr[indx] = val[1]

    return swap_count, arr


def paritysort_list(arr):
    """Move all even numbers to the left and all odd numbers to the right

    Args:
        arr (list[int]): a list of integers to be sorted in place

    Returns:
        (int): number of exchanges needed to complete the sorting
    """
    larr = len(arr)
    parr = [[i[0] % 2, i] for i in arr]

    swap_count = 0
    for i in range(larr):
        swapped = False
        for j in range(0, larr - i - 1):
            if parr[j][0] > parr[j + 1][0]:
                parr[j], parr[j + 1] = parr[j + 1], parr[j]
                swapped = True
                swap_count += 1
        if not swapped:
            break

    for indx, val in enumerate(parr):
        arr[indx] = list(val[1])

    return swap_count, arr


def qubit_particle_number_sector(nqubits: int,
                                 pnum: int) -> List[numpy.ndarray]:
    """Generate the basis vectors into the qubit basis representing all states
    which have a definite particle number.

    Args:
        nqubits (int): the number of qubits in the qpu

        pnum (int): the number of particles to build vectors into

    Returns:
        list[numpy.array(dtype=numpy.complex128)]
    """
    occ = numpy.array([0, 1], dtype=numpy.int32)
    uno = numpy.array([1, 0], dtype=numpy.int32)
    pn_set = lexicographic_bitstring_generator(pnum, nqubits)
    vectors = []
    for iorbocc in pn_set:
        orbocc = int(iorbocc)
        if orbocc & 1:
            vec = occ
        else:
            vec = uno
        orbocc = orbocc >> 1
        for _ in range(nqubits - 1):
            if orbocc & 1:
                vec = numpy.kron(vec, occ)
            else:
                vec = numpy.kron(vec, uno)
            orbocc = orbocc >> 1
        vectors.append(vec)
    return vectors


def qubit_config_sector(nqubits: int, pnum: int,
                        m_s: int) -> List[numpy.ndarray]:
    """Generate the basis vectors into the qubit basis representing all states
    which have a definite particle number and spin.

    Args:
        nqubits (int): the number of qubits in the qpu

        pnum (int): the number of particles to build vectors into

        m_s (int): the s_z spin quantum number

    Returns:
        list[numpy.array(dtype=numpy.complex128)]
    """
    occ = numpy.array([0, 1], dtype=numpy.int32)
    uno = numpy.array([1, 0], dtype=numpy.int32)
    achk = 0
    bchk = 0
    pn_set = []

    for num in range(nqubits):
        if num % 2:
            bchk += 2**num
        else:
            achk += 2**num

    initpn = lexicographic_bitstring_generator(pnum, nqubits)
    for ioccu in initpn:
        occu = int(ioccu)
        if (count_bits(occu & achk) - count_bits(occu & bchk)) == m_s:
            pn_set.append(occu)

    vectors = []
    for orbocc in pn_set:
        if orbocc & 1:
            vec = occ
        else:
            vec = uno
        orbocc = orbocc >> 1
        for _ in range(nqubits - 1):
            if orbocc & 1:
                vec = numpy.kron(vec, occ)
            else:
                vec = numpy.kron(vec, uno)
            orbocc = orbocc >> 1
        vectors.append(vec)
    return vectors


def qubit_particle_number_index(nqubits: int, pnum: int) -> List[int]:
    """Generate indexes corresponding to the coefficient that is associated
    with a specific particle number

    Args:
        nqubits (int): the number of qubits to act upon

        pnum (int): the number of particles to view

    Returns:
        list[int]: integers indicating where in the qubit wavefunction the
            basis state corresponds to particle number
    """
    indexes = []
    pn_set = lexicographic_bitstring_generator(pnum, nqubits)
    for occ in pn_set:
        orbocc = int(occ)
        if orbocc & 1:
            index = 1
        else:
            index = 0
        orbocc = orbocc >> 1
        veclen = 2
        for _ in range(nqubits - 1):
            if orbocc & 1:
                index = index + veclen
            orbocc = orbocc >> 1
            veclen *= 2
        indexes.append(index)
    return indexes


def qubit_particle_number_index_spin(nqubits: int,
                                     pnum: int) -> List[Tuple[int, int]]:
    """Generate indexes corresponding to the coefficient that is associated
    with a specific particle number and spin

    Args:
        nqubits (int): the number of qubits to act upon

        pnum (int): the number of particles to view

    Returns:
        list[(int int)]: tuples of integers indicating where in the qubit
            wavefunction the basis state corresponds to particle number and
            return the corresponding spin
    """
    indexes = []
    pn_set = lexicographic_bitstring_generator(pnum, nqubits)
    for occ in pn_set:
        orbocc = int(occ)
        totspn = 0
        curspn = -1
        if orbocc & 1:
            index = 1
            totspn += curspn
        else:
            index = 0
        orbocc = orbocc >> 1
        veclen = 2
        for _ in range(nqubits - 1):
            curspn *= -1
            if orbocc & 1:
                index = index + veclen
                totspn += curspn
            orbocc = orbocc >> 1
            veclen *= 2
        indexes.append((index, totspn))
    return indexes


def rand_wfn(adim: int, bdim: int) -> numpy.ndarray:
    """Utility for generating random normalized wavefunctions.

    Args:
        adim (int): length of the alpha string

        bdim (int): length of the beta string

    Returns:
        numpy.ndarray(shape=(adim, bdim), dtype=numpy.complex128)
    """
    wfn = numpy.random.randn(adim, bdim).astype(numpy.complex128) + \
          numpy.random.randn(adim, bdim).astype(numpy.complex128)*1.j

    return wfn


def map_broken_symmetry(s_z: int,
                        norb: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Create a map between spin broken and number broken wavefunctions.
    Args:
        s_z (int): quantum number s_z

        norb (int): number of orbitals

    Returns:
        Dict[Tuple[int, int], Tuple[int, int]]
    """
    spin_to_number = {}
    nele = norb + s_z

    maxb = min(norb, nele)
    minb = nele - maxb

    for nbeta in range(minb, maxb + 1):
        nb_beta = norb - nbeta
        nalpha = nele - nbeta
        spin_to_number[(nalpha, nb_beta)] = (nele - nbeta, nbeta)

    return spin_to_number


def sort_configuration_keys(configs: KeysView[Tuple[int, int]]
                           ) -> List[Tuple[int, int]]:
    """Return a standard sorting of configuration keys in a wavefunction for
    comparison.  The configurations are sorted first by the number of particles
    and then by the spin quantum number.

    Args:
        configs (KeysView[Tuple[int, int]]): a dictionary of keys

    Returns:
        List[Tuple[int, int]]: list with the sorted keys
    """
    return sorted(configs, key=itemgetter(0, 1))


def validate_config(nalpha: int, nbeta: int, norb: int) -> None:
    """ Check that the parameters passed are valid to build a configuration

    Args:
        nalpha (int): the number of alpha electrons

        nbeta (int): the number of beta electrons

        norb (int): the number of spatial orbitals

    Returns:
        nothing: only raises errors if necessary
    """
    if nalpha < 0:
        raise ValueError("Cannot have negative number of alpha electrons")

    if nbeta < 0:
        raise ValueError("Cannot have negative number of beta electrons")

    if norb < 0:
        raise ValueError("Cannot have negative number of orbitals")

    if norb < nalpha or norb < nbeta:
        raise ValueError("Insufficient number of orbitals")


def validate_tuple(matrices: Tuple[numpy.ndarray, ...]) -> None:
    """Validate that the tuple passed in is valid for initializing a general
    Hamiltonian

    Args:
        matrices(tuple[Nparray,...]): arrays to be used to initialize a \
            Hamiltonian
    """
    assert isinstance(matrices, tuple)
    for rank, term in enumerate(matrices):
        assert isinstance(term, numpy.ndarray)
        assert 2 * (rank + 1) == term.ndim


def dot(wfn1: 'wavefunction.Wavefunction',
        wfn2: 'wavefunction.Wavefunction') -> complex:
    """Calculate the dot product of two wavefunctions.  Note that this does
    not use the conjugate.  See vdot for the similar conjugate functionality.

    Args:
        wfn1 (wavefunction.Wavefunction): wavefunction corresponding to the
            row vector

        wfn2 (wavefunction.Wavefunction): wavefunction corresponding to the
            coumn vector

    Returns:
        (complex): scalar as result of the dot product
    """
    brakeys = wfn1.sectors()
    ketkeys = wfn2.sectors()
    keylist = [config for config in brakeys if config in ketkeys]
    ipval = .0 + .0j
    for sector in keylist:
        ipval += numpy.dot(
            wfn1.get_coeff(sector).flat,
            wfn2.get_coeff(sector).flat)
    return ipval


def vdot(wfn1: 'wavefunction.Wavefunction',
         wfn2: 'wavefunction.Wavefunction') -> complex:
    """Calculate the inner product of two wavefunctions using conjugation on
    the elements of wfn1.

    Args:
        wfn1 (wavefunction.Wavefunction): wavefunction corresponding to the
            conjugate row vector

        wfn2 (wavefunction.Wavefunction): wavefunction corresponding to the
            coumn vector

    Returns:
        (complex): scalar as result of the dot product
    """
    brakeys = wfn1.sectors()
    ketkeys = wfn2.sectors()
    keylist = [config for config in brakeys if config in ketkeys]
    ipval = .0 + .0j
    for config in keylist:
        ipval += numpy.vdot(
            wfn1.get_coeff(config).flat,
            wfn2.get_coeff(config).flat)
    return ipval


def tensors_equal(tensor1: Dict[int, numpy.ndarray], \
                    tensor2: Dict[int, numpy.ndarray]) -> bool:
    """ Comparison function for two tensors (dictionaries as used in
    subclasses of Hamiltonian). Used in implementing equality operators
    in Hamiltonian subclasses.

    Note: it compares using numpy.all and not numpy.isclose, so that even small
    numerical deviations may lead to inequality.

    Args:
        tensor1 (Dict[int, numpy.ndarray]): first tensor

        tensor2 (Dict[int, numpy.ndarray]): second tensor

    Returns:
        bool: True if all keys and corresponding values of tensor1 match to that
        of tensor2, false otherwise.
    """
    if tensor1.keys() != tensor2.keys():
        return False
    for (k, v), (k2, v2) in zip(tensor1.items(), tensor2.items()):
        if k != k2 or not numpy.all(v == v2):
            return False
    return True
