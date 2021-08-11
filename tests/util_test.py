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
""" util unit tests
"""
import numpy
import random
import pytest
from typing import List, Any

from fqe import util
from fqe.wavefunction import Wavefunction


def permBetween(orig: List[Any], perm: List[Any]) -> int:
    """Checks the parity of the permutation between orig and perm
    """
    perm = list(perm)

    swaps = 0
    for ii in range(len(orig) - 1):
        p0 = orig[ii]
        p1 = perm[ii]
        if p0 != p1:
            sii = perm[ii:].index(p0) + ii  # Find position in perm
            perm[ii], perm[sii] = p0, p1  # Swap in perm
            swaps += 1
    return swaps % 2


def test_alpha_beta_electrons():
    """Check to make sure that the correct number of alpha and beta
    electrons are parsed from the number and multiplicity
    """
    assert (1, 1) == util.alpha_beta_electrons(2, 0)
    assert (4, 1) == util.alpha_beta_electrons(5, 3)
    assert (0, 5) == util.alpha_beta_electrons(5, -5)


def test_alpha_beta_error():
    """Check to make sure that alpha_beta_electrons() throws the right errors
    """
    with pytest.raises(ValueError):
        util.alpha_beta_electrons(-1, 0)
    with pytest.raises(ValueError):
        util.alpha_beta_electrons(2, 4)
    with pytest.raises(ValueError):
        util.alpha_beta_electrons(4, 1)


def test_bubblesort_order():
    """Check that bubble sort works.
    """
    length = 5
    test_list = [(length - 1 - i) for i in range(length)]
    ordered_list = [i for i in range(length)]
    util.bubblesort(test_list)
    assert ordered_list == test_list


def test_bubblesort_permutation_count():
    """ Make sure that we are counting the correct number of permutations
    to sort the list
    """
    length = 2
    test_list = [(length - 1 - i) for i in range(length)]
    assert 1 == util.bubblesort(test_list)
    length = 3
    test_list = [(length - 1 - i) for i in range(length)]
    assert 3 == util.bubblesort(test_list)
    test_list = [2, 0, 1]
    assert 2 == util.bubblesort(test_list)


def test_reverse_bubblesort_permutation_count():
    """ Make sure that we are counting the correct number of permutations
    to sort the list
    """
    test_list = [[0, 0], [1, 0]]
    assert 1 == util.reverse_bubble_list(test_list)
    test_list = [[0, 0], [1, 0], [2, 0]]
    assert 3 == util.reverse_bubble_list(test_list)
    test_list = [[0, 0], [2, 0], [1, 0]]
    assert 2 == util.reverse_bubble_list(test_list)


def test_configuration_key_union_empty():
    """The union of no configuration keys should be an empty list
    """
    assert [] == util.configuration_key_union()


def test_configuration_key_union():
    """The union of no configuration keys should be an empty list
    """
    configs0 = [(2, 0), (3, 1)]
    configs1 = [(2, 0), (5, 1), (6, -2)]
    testset = set([(2, 0), (3, 1), (5, 1), (6, -2)])
    assert testset == set(util.configuration_key_union(configs0, configs1))


def test_configuration_key_union_many():
    """The union of many different keys should be all of them
    """
    configs0 = [(2, 0)]
    configs1 = [(5, 1)]
    configs2 = [(6, -2)]
    configs3 = [(3, -3)]
    refset = set([(2, 0), (5, 1), (6, -2), (3, -3)])
    testset = set(
        util.configuration_key_union(configs0, configs1, configs2, configs3))
    assert testset == refset


def test_configuration_key_intersection_none():
    """If there are no keys in common the intersection should be zero
    """
    assert [] == util.configuration_key_intersection([(2, 0)], [(2, 2)])


def test_configuration_key_intersection():
    """Check that the intersection returns the intersection
    """
    configs0 = [(10, 0), (3, 1), (5, -1)]
    configs1 = [(2, 0), (3, 1), (3, -1)]
    configs2 = [(10, 0), (3, 1), (3, -1)]
    assert [
        (3, 1)
    ] == util.configuration_key_intersection(configs0, configs1, configs2)


def test_bitstring_groundstate():
    """The ground state bitstring has the n lowest bits flipped
    """
    assert 15 == util.init_bitstring_groundstate(4)


def test_qubit_particle_number_sector():
    """Find the vectors which are the basis for a particular particle
    number.
    """
    zero = 0
    one = 1
    ref = [
        numpy.array([zero, zero, one, zero], dtype=numpy.int32),
        numpy.array([zero, one, zero, zero], dtype=numpy.int32)
    ]
    test = util.qubit_particle_number_sector(2, 1)
    for i, j in zip(test, ref):
        assert i.all() == j.all()


def test_qubit_particle_number_index_spin():
    """Find the indexes which point to the correct coefficients in a qubit
    particle number sector and return the total spin.
    """
    ref = [(3, 0), (5, -2), (6, 0), (9, 0), (10, 2), (12, 0)]
    test = util.qubit_particle_number_index_spin(4, 2)
    assert ref == test


def test_qubit_config_sector():
    """Find the basis vectors for a particular particle number and spin
    configuration
    """
    zero = 0
    one = 1
    lowstate = [
        zero, zero, one, zero, zero, zero, zero, zero, zero, zero, zero, zero,
        zero, zero, zero, zero
    ]
    highstate = [
        zero, zero, zero, zero, zero, zero, zero, zero, one, zero, zero, zero,
        zero, zero, zero, zero
    ]
    ref = [
        numpy.array(lowstate, dtype=numpy.int32),
        numpy.array(highstate, dtype=numpy.int32)
    ]
    test = util.qubit_config_sector(4, 1, 1)
    for i, j in zip(test, ref):
        assert i.all() == j.all()
    ref = [numpy.array([zero, zero, zero, one], dtype=numpy.int32)]
    test = util.qubit_config_sector(2, 2, 0)
    for i, j in zip(test, ref):
        assert i.all() == j.all()


def test_qubit_particle_number_index():
    """Find the indexes which point to the correct coefficients in a qubit
    particle number sector and return the total spin.
    """
    ref = [1, 2, 4, 8]
    test = util.qubit_particle_number_index(4, 1)
    assert ref == test


def test_qubit_vacuum():
    """The qubit vacuum is the first vector in the qubit basis.
    """
    _gs = numpy.array([1. + .0j, 0. + .0j, 0. + .0j, 0. + .0j],
                      dtype=numpy.complex64)
    assert list(_gs) == list(util.init_qubit_vacuum(2))


def test_sort_config_keys():
    """Keys are sorted by particle number and then by m_s
    """
    ref = [(0, 0), (1, -1), (3, -3), (3, 1), (5, -2), (5, 1)]
    keys = [(5, 1), (5, -2), (0, 0), (1, -1), (3, -3), (3, 1)]
    test = util.sort_configuration_keys(keys)
    assert test == ref


def test_validate_config():
    """Make sure that the configuration validation routine identifies
    problematic values
    """
    with pytest.raises(ValueError):
        util.validate_config(0, 0, -1)
    with pytest.raises(ValueError):
        util.validate_config(3, 0, 2)
    with pytest.raises(ValueError):
        util.validate_config(0, 3, 2)
    with pytest.raises(ValueError):
        util.validate_config(-1, 1, 2)
    with pytest.raises(ValueError):
        util.validate_config(1, -1, 2)
    assert util.validate_config(0, 0, 0) is None
    assert util.validate_config(0, 0, 1) is None


def test_parity_sort_list():
    """Sort a list of lists according to the parity of the index in the 0th
    element.
    """
    test = [[x, -x, {'Unused': True}] for x in range(19)]
    random.shuffle(test)  # randomly shuffled array
    test_copy = list(test)
    test_even = [x for x in test if x[0] % 2 == 0]
    test_odd = [x for x in test if x[0] % 2 == 1]

    nswap, _ = util.paritysort_list(test)
    assert test_even + test_odd == test  # First even elements, then odds
    assert permBetween(test, test_copy) == nswap % 2


def test_parity_sort_int():
    """Sort a list of ints according to the parity of the element.
    """
    test = list(range(19))
    random.shuffle(test)  # randomly shuffled array
    test_copy = list(test)
    test_even = [x for x in test if x % 2 == 0]
    test_odd = [x for x in test if x % 2 == 1]

    nswap, _ = util.paritysort_int(test)
    assert test_even + test_odd == test  # First even elements, then odds
    assert permBetween(test, test_copy) == nswap % 2


def test_rand_wfn():
    """Check rand_wfn
    """
    adim = 10
    bdim = 9
    test = util.rand_wfn(adim, bdim)
    assert test.shape == (adim, bdim)
    assert test.dtype == numpy.complex128


def test_validate_tuple():
    """Check validate_tuple. assert is evaluated in the function.
    """
    param = (numpy.zeros((2, 2)), numpy.zeros((2, 2, 2, 2)))
    util.validate_tuple(param)


def test_dot():
    numpy.random.seed(seed=409)
    wfn1 = Wavefunction([[2, 0, 2]])
    wfn1.set_wfn(strategy='random')

    wfn2 = Wavefunction([[2, 0, 2]])
    wfn2.set_wfn(strategy='random')
    assert abs(util.dot(wfn1, wfn2) - (-0.1872999545144855+0.21646742443751746j)) \
        < 1.0e-8

    wfn3 = Wavefunction([[2, 2, 2]])
    wfn3.set_wfn(strategy='random')
    assert util.dot(wfn1, wfn3) == 0.0


def test_vdot():
    numpy.random.seed(seed=409)
    wfn1 = Wavefunction([[2, 0, 2]])
    wfn1.set_wfn(strategy='random')

    wfn2 = Wavefunction([[2, 0, 2]])
    wfn2.set_wfn(strategy='random')
    assert abs(util.vdot(wfn1, wfn2) - (-0.04163626246314951-0.43391345135564796j)) \
        < 1.0e-8

    wfn3 = Wavefunction([[2, 2, 2]])
    wfn3.set_wfn(strategy='random')
    assert util.vdot(wfn1, wfn3) == 0.0


@pytest.mark.parametrize("sz,norb", [[1, 10], [0, 5], [-4, 7], [7, 7]])
def test_map_broken_symmetry(sz, norb):
    """Checks map_broken_symmetry
    """
    mapping = util.map_broken_symmetry(sz, norb)
    assert set(k[0] - k[1] for k in mapping) == set([sz])
    assert len(mapping) == norb - abs(sz) + 1
    assert set(v[0] + v[1] for v in mapping.values()) == set([norb + sz])


def test_tensors_equal():
    """ Test tensors_equal comparison function
    """
    tensor1 = { '0' : numpy.zeros((2,2), dtype=numpy.complex128), \
                '1' : numpy.zeros((2,2), dtype=numpy.complex128) }

    tensor2 = { '0' : numpy.zeros((2,2), dtype=numpy.complex128), \
                '1' : numpy.ones((2,2), dtype=numpy.complex128) }

    tensor3 = {'0': numpy.zeros((2, 2), dtype=numpy.complex128)}
    assert util.tensors_equal(tensor1, tensor1)
    assert not util.tensors_equal(tensor1, tensor2)
    assert not util.tensors_equal(tensor1, tensor3)
