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
""" fci_graph unit tests
"""

import numpy
import pytest
from scipy import special

import fqe
from fqe import fci_graph
from fqe.lib.fci_graph import _c_map_to_deexc_alpha_icol
from fqe.settings import CodePath

from tests.unittest_data.fci_graph_data import loader
from tests.comparisons import compare_Spinmap

cases = [(4, 3, 8), (4, 4, 6), (0, 3, 7), (2, 0, 6)]


def test_fci_graph(c_or_python):
    """Check the basic initializers and getter functions.
    """
    fqe.settings.use_accelerated_code = c_or_python

    refdata = [
        15, 23, 39, 71, 135, 27, 43, 75, 139, 51, 83, 147, 99, 163, 195, 29, 45,
        77, 141, 53, 85, 149, 101, 165, 197, 57, 89, 153, 105, 169, 201, 113,
        177, 209, 225, 30, 46, 78, 142, 54, 86, 150, 102, 166, 198, 58, 90, 154,
        106, 170, 202, 114, 178, 210, 226, 60, 92, 156, 108, 172, 204, 116, 180,
        212, 228, 120, 184, 216, 232, 240
    ]
    reflist = numpy.array(refdata, dtype=numpy.uint64)

    refdict = {
        15: 0,
        23: 1,
        27: 5,
        29: 15,
        30: 35,
        39: 2,
        43: 6,
        45: 16,
        46: 36,
        51: 9,
        53: 19,
        54: 39,
        57: 25,
        58: 45,
        60: 55,
        71: 3,
        75: 7,
        77: 17,
        78: 37,
        83: 10,
        85: 20,
        86: 40,
        89: 26,
        90: 46,
        92: 56,
        99: 12,
        101: 22,
        102: 42,
        105: 28,
        106: 48,
        108: 58,
        113: 31,
        114: 51,
        116: 61,
        120: 65,
        135: 4,
        139: 8,
        141: 18,
        142: 38,
        147: 11,
        149: 21,
        150: 41,
        153: 27,
        154: 47,
        156: 57,
        163: 13,
        165: 23,
        166: 43,
        169: 29,
        170: 49,
        172: 59,
        177: 32,
        178: 52,
        180: 62,
        184: 66,
        195: 14,
        197: 24,
        198: 44,
        201: 30,
        202: 50,
        204: 60,
        209: 33,
        210: 53,
        212: 63,
        216: 67,
        225: 34,
        226: 54,
        228: 64,
        232: 68,
        240: 69
    }
    norb = 8
    nalpha = 4
    nbeta = 0
    lena = int(special.binom(norb, nalpha))
    max_bitstring = (1 << norb) - (1 << (norb - nalpha))
    testgraph = fci_graph.FciGraph(nalpha, nbeta, norb)
    assert testgraph._build_string_address(nalpha, norb, [0, 1, 2, 3]) == 0
    assert testgraph._build_string_address(nalpha, norb, [1, 2, 3, 7]) == 38

    test_list, test_dict = testgraph._build_strings(nalpha, lena)
    assert numpy.array_equal(test_list, reflist)
    assert test_dict == refdict
    assert testgraph.string_beta(0) == 0
    assert testgraph.string_alpha(lena - 1) == max_bitstring
    assert testgraph.index_beta(0) == 0
    assert testgraph.index_alpha(max_bitstring) == lena - 1
    assert testgraph.lena() == lena
    assert testgraph.lenb() == 1
    assert testgraph.nalpha() == nalpha
    assert testgraph.nbeta() == nbeta
    assert testgraph.norb() == norb
    assert testgraph.string_alpha(lena - 1) == max_bitstring

    assert numpy.array_equal(testgraph.string_alpha_all(), reflist)
    assert numpy.array_equal(testgraph.string_beta_all(),
                             numpy.array([0], dtype=numpy.uint64))

    assert testgraph.index_alpha_all() == refdict
    assert testgraph.index_beta_all() == {0: 0}


def test_fci_graph_maps(c_or_python):
    """Check graph mapping functions
    """
    fqe.settings.use_accelerated_code = c_or_python

    ref_alpha_map = {
        (0, 0): [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
        (0, 1): [(3, 1, 1), (4, 2, 1)],
        (0, 2): [(3, 0, -1), (5, 2, 1)],
        (0, 3): [(4, 0, -1), (5, 1, -1)],
        (1, 0): [(1, 3, 1), (2, 4, 1)],
        (1, 1): [(0, 0, 1), (3, 3, 1), (4, 4, 1)],
        (1, 2): [(1, 0, 1), (5, 4, 1)],
        (1, 3): [(2, 0, 1), (5, 3, -1)],
        (2, 0): [(0, 3, -1), (2, 5, 1)],
        (2, 1): [(0, 1, 1), (4, 5, 1)],
        (2, 2): [(1, 1, 1), (3, 3, 1), (5, 5, 1)],
        (2, 3): [(2, 1, 1), (4, 3, 1)],
        (3, 0): [(0, 4, -1), (1, 5, -1)],
        (3, 1): [(0, 2, 1), (3, 5, -1)],
        (3, 2): [(1, 2, 1), (3, 4, 1)],
        (3, 3): [(2, 2, 1), (4, 4, 1), (5, 5, 1)]
    }
    ref_beta_map = {
        (0, 0): [(0, 0, 1)],
        (0, 1): [(1, 0, 1)],
        (0, 2): [(2, 0, 1)],
        (0, 3): [(3, 0, 1)],
        (1, 0): [(0, 1, 1)],
        (1, 1): [(1, 1, 1)],
        (1, 2): [(2, 1, 1)],
        (1, 3): [(3, 1, 1)],
        (2, 0): [(0, 2, 1)],
        (2, 1): [(1, 2, 1)],
        (2, 2): [(2, 2, 1)],
        (2, 3): [(3, 2, 1)],
        (3, 0): [(0, 3, 1)],
        (3, 1): [(1, 3, 1)],
        (3, 2): [(2, 3, 1)],
        (3, 3): [(3, 3, 1)]
    }
    alist = numpy.array([3, 5, 9, 6, 10, 12], dtype=numpy.uint64)
    blist = numpy.array([1, 2, 4, 8], dtype=numpy.uint64)
    aind = {3: 0, 5: 1, 6: 3, 9: 2, 10: 4, 12: 5}
    bind = {1: 0, 2: 1, 4: 2, 8: 3}
    norb = 4
    nalpha = 2
    nbeta = 1
    testgraph = fci_graph.FciGraph(nalpha, nbeta, norb)
    alpha_map = testgraph._build_mapping(alist, nalpha, aind)
    beta_map = testgraph._build_mapping(blist, nbeta, bind)

    assert alpha_map.keys() == ref_alpha_map.keys()
    for ak in alpha_map:
        numpy.testing.assert_equal(alpha_map[ak], ref_alpha_map[ak])

    assert beta_map.keys() == ref_beta_map.keys()
    for ak in alpha_map:
        numpy.testing.assert_equal(alpha_map[ak], ref_alpha_map[ak])

    dummy_map = ({(1, 1): (0, 1, 2)}, {(-1, -1), (0, 1, 2)})
    testgraph.insert_mapping(1, -1, dummy_map)
    assert testgraph.find_mapping(1, -1) == dummy_map


def test_alpha_beta_transpose(norb=4, nalpha=3, nbeta=2):
    """Check alpha_beta_transpose
    """
    original = fci_graph.FciGraph(nalpha, nbeta, norb)
    transposed = original.alpha_beta_transpose()

    assert original is not transposed
    assert original._nalpha == transposed._nbeta
    assert original._nbeta == transposed._nalpha
    assert original._lena == transposed._lenb
    assert original._lenb == transposed._lena

    assert original._astr is not transposed._bstr  # not same object
    assert numpy.array_equal(original._astr, transposed._bstr)  # but equiv
    assert original._bstr is not transposed._astr  # not same object
    assert numpy.array_equal(original._bstr, transposed._astr)  # but equiv

    assert original._aind is not transposed._bind  # not same object
    assert original._aind == transposed._bind  # but equiv
    assert original._bind is not transposed._aind  # not same object
    assert original._bind == transposed._aind  # but equiv

    assert original._alpha_map is not transposed._beta_map  # not same object
    compare_Spinmap(original._alpha_map, transposed._beta_map)
    assert original._beta_map is not transposed._alpha_map  # not same object
    compare_Spinmap(transposed._beta_map, original._alpha_map)

    assert original._dexca is not transposed._dexcb  # not same object
    assert numpy.array_equal(original._dexca, transposed._dexcb)  # but equiv
    assert original._dexcb is not transposed._dexca  # not same object
    assert numpy.array_equal(original._dexcb, transposed._dexca)  # but equiv


def test_map(alpha_or_beta, norb=4, nalpha=3, nbeta=2):
    """Check alpha_map or beta_map
    """
    graph = fci_graph.FciGraph(nalpha, nbeta, norb)
    if alpha_or_beta == "alpha":
        get_map = graph.alpha_map
        map_object = graph._alpha_map
    elif alpha_or_beta == "beta":
        get_map = graph.beta_map
        map_object = graph._beta_map
    else:
        raise ValueError(f'Unknown value {alpha_or_beta}')

    assert get_map(1, 2) is map_object[(1, 2)]
    assert get_map(2, 0) is map_object[(2, 0)]

    with pytest.raises(KeyError):
        get_map(-1, 2)

    with pytest.raises(KeyError):
        get_map(0, 4)


def test_init_logic():
    """Checks the logic of the __init__ of FciGraph
    """
    with pytest.raises(ValueError):
        fci_graph.FciGraph(-1, 10, 10)
    with pytest.raises(ValueError):
        fci_graph.FciGraph(11, 1, 10)
    with pytest.raises(ValueError):
        fci_graph.FciGraph(1, -1, 10)
    with pytest.raises(ValueError):
        fci_graph.FciGraph(1, 11, 10)
    with pytest.raises(ValueError):
        fci_graph.FciGraph(1, 1, -1)


@pytest.mark.parametrize("nalpha,nbeta,norb", cases)
def test_make_mapping_each(alpha_or_beta, c_or_python, nalpha, nbeta, norb):
    """Check make_mapping_each wrt reference data
    """
    fqe.settings.use_accelerated_code = c_or_python
    # graph = loader(nalpha, nbeta, norb, 'graph')
    graph = fci_graph.FciGraph(nalpha, nbeta, norb)
    reference = loader(nalpha, nbeta, norb, 'make_mapping_each')

    alpha = {"alpha": True, "beta": False}[alpha_or_beta]
    length = {"alpha": graph.lena(), "beta": graph.lenb()}[alpha_or_beta]

    for (c_alpha, dag, undag), refval in reference.items():
        if c_alpha == alpha:
            result = numpy.zeros((length, 3), dtype=numpy.uint64)
            count = graph.make_mapping_each(result, alpha, dag, undag)
            assert numpy.array_equal(result[:count, :], refval)


@pytest.mark.parametrize("nalpha,nbeta,norb", cases)
def test_map_to_deexc_alpha_icol(c_or_python, norb, nalpha, nbeta):
    """Check _map_to_deexc_alpha_icol
    """
    fqe.settings.use_accelerated_code = c_or_python
    # graph = loader(nalpha, nbeta, norb, 'graph')
    graph = fci_graph.FciGraph(nalpha, nbeta, norb)
    rindex, rexc, rdiag = loader(nalpha, nbeta, norb, 'map_to_deexc_alpha_icol')

    index, exc, diag = graph._map_to_deexc_alpha_icol()
    assert numpy.array_equal(rindex, index)
    assert numpy.array_equal(rexc, exc)
    assert numpy.array_equal(rdiag, diag)

    if c_or_python == CodePath.PYTHON:
        return

    tmp = numpy.zeros((norb, norb))
    test = {(i, i): numpy.zeros((1, 1)) for i in range(norb)}
    with pytest.raises(ValueError):
        _c_map_to_deexc_alpha_icol(tmp, tmp, tmp, tmp, norb, test)


@pytest.mark.parametrize("nalpha,nbeta,norb", cases)
def test_get_block_mappings(norb, nalpha, nbeta):
    """Check _get_block_mappings
    """
    # graph = loader(nalpha, nbeta, norb, 'graph')
    graph = fci_graph.FciGraph(nalpha, nbeta, norb)
    rmappings_set = loader(nalpha, nbeta, norb, 'get_block_mappings')

    for (ms, jo), rmappings in rmappings_set.items():
        mappings = graph._get_block_mappings(max_states=ms, jorb=jo)

        # Check if the ranges (cmap[0] and cmap[1]) loops over all states
        # Just an extra check
        assert set((x for cmap in mappings for x in cmap[0])) == \
            set(range(graph.lena()))
        assert set((x for cmap in mappings for x in cmap[1])) == \
            set(range(graph.lenb()))

        for rmap, cmap in zip(rmappings, mappings):
            assert rmap[0] == cmap[0]
            assert rmap[1] == cmap[1]
            assert numpy.array_equal(rmap[2], cmap[2])
            assert numpy.array_equal(rmap[3], cmap[3])
