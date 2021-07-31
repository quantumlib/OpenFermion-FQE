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
"""Unittests for fci_graph_set
"""

import pytest
import numpy

import fqe
from fqe.fci_graph import FciGraph
from fqe.fci_graph_set import FciGraphSet
from tests.unittest_data.fci_graph_set_loader import loader

refs = [(4, 3, 8), (4, 4, 6)]

def test_init_from_params():
    """The fci graph set should also be initializable from a prebuilt set
    of parameters.
    """
    params = [[6, 6, 6], [6, 4, 6], [6, 2, 6], [6, 0, 6], [6, -2, 6],
              [6, -4, 6], [6, -6, 6]]
    test = FciGraphSet(6, 6, params)
    assert isinstance(test, FciGraphSet)
    params = [[3, 3, 6], [3, 1, 6], [3, -1, 6], [2, 0, 6]]
    test = FciGraphSet(0, 1, params)
    assert isinstance(test, FciGraphSet)
    assert test._linked == set([((2, 1), (3, 0)), ((1, 2), (2, 1))])

def test_append():
    """This function append a FciGraph object to the set. It automatically
    calls a function to link the new FciGraph to the existing ones.
    """
    params = [[3, 3, 6], [3, 1, 6], [3, -1, 6], [2, 0, 6]]
    test = FciGraphSet(0, 1, params)
    graph = FciGraph(2, 0, 6)
    test.append(graph)
    assert test._linked == set([((2, 1), (3, 0)), ((1, 2), (2, 1)), ((1, 1), (2, 0))])

    betamap = graph._fci_map[(-1, 1)][1]
    for i in range(6):
        assert numpy.array_equal(betamap[(i,)], numpy.array([[0, i, 1]], dtype=numpy.int32))

@pytest.mark.parametrize("nalpha,nbeta,norb", refs)
def test_vs_ref(c_or_python, nalpha, nbeta, norb):
    """Test vs reference data for maps to N - 2 systems"""
    fqe.settings.use_accelerated_code = c_or_python
    assert nalpha >= 2
    assert nbeta >= 2
    graphset = FciGraphSet(2, 2)
    graphset.append(FciGraph(nalpha, nbeta, norb))
    graphset.append(FciGraph(nalpha - 1, nbeta - 1, norb))
    graphset.append(FciGraph(nalpha - 2, nbeta, norb))
    graphset.append(FciGraph(nalpha, nbeta - 2, norb))

    # load reference fci_graph
    graphref = loader(2, nalpha, nbeta, norb)

    def _dcompare(x, y):
        for key in x.keys():
            a = x[key]
            b = y[key]
            assert (numpy.asarray(a) == numpy.asarray(b)).all()

    for key in graphref._dataset.keys():
        ref = graphref._dataset[key]
        out = graphset._dataset[key]
        for midx in ref._fci_map.keys():
            x1, x2 = ref._fci_map[midx]
            y1, y2 = out._fci_map[midx]
            _dcompare(x1, y1)
            _dcompare(x2, y2)
