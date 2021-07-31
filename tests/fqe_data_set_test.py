#   Copyright 2020 Google LLC

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http:gc/www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Unittesting for the fqe_data_set module
"""
import copy
import numpy

import fqe
from fqe import fqe_data
from fqe import fqe_data_set
from tests.unittest_data.build_hamiltonian import build_H1, build_H2,\
                                                  build_H3, build_H4
from tests.unittest_data.fqe_data_set_loader import FqeDataSetLoader
import pytest

def test_init_sectors():
    data1 = fqe_data.FqeData(2, 0, 2)
    data2 = fqe_data.FqeData(2, 2, 2)
    val = 1.2 + 3.1j
    data1.coeff[:, :] = val
    data2.coeff[:, :] = val
    test = {(1, 1): data1, (2, 0): data2}
    dataset = fqe_data_set.FqeDataSet(0, 0, test)

    sectors = dataset.sectors()
    assert set(sectors.keys()) == set([(2, 0), (1, 1)])
    for sec in sectors.values():
        assert numpy.allclose(sec.coeff, val)

def test_empty_copy():
    data1 = fqe_data.FqeData(2, 0, 2)
    data2 = fqe_data.FqeData(2, 2, 2)
    val = 1.2 + 3.1j
    data1.coeff[:, :] = val
    data2.coeff[:, :] = val
    test = {(1, 1): data1, (2, 0): data2}
    dataset = fqe_data_set.FqeDataSet(0, 0, test)

    dataset2 = dataset.empty_copy()

    sectors = dataset2.sectors()
    assert set(sectors.keys()) == set([(2, 0), (1, 1)])
    for sec in sectors.values():
        assert numpy.allclose(sec.coeff, 0.0)

def test_ax_plus_y_scale_fill():
    data1 = fqe_data.FqeData(2, 0, 2)
    data2 = fqe_data.FqeData(2, 2, 2)
    data1.coeff[:, :] = 1.0
    data2.coeff[:, :] = 1.0
    testa = {(1, 1): data1, (2, 0): data2}
    dataseta = fqe_data_set.FqeDataSet(0, 0, testa)

    data1 = fqe_data.FqeData(2, 0, 2)
    data2 = fqe_data.FqeData(2, 2, 2)
    data1.coeff[:, :] = 2.3
    data2.coeff[:, :] = 2.3
    testb = {(1, 1): data1, (2, 0): data2}
    datasetb = fqe_data_set.FqeDataSet(0, 0, testb)

    coeff = 1.2 + 0.5j
    dataseta.ax_plus_y(coeff, datasetb)
    for sec in dataseta._data.values():
        assert numpy.allclose(sec.coeff, 1.0 + coeff*2.3)
    for sec in datasetb._data.values():
        assert numpy.allclose(sec.coeff, 2.3)

    dataseta.scale(coeff)
    for sec in dataseta._data.values():
        assert numpy.allclose(sec.coeff, coeff * (1.0 + coeff*2.3))

    dataseta.fill(coeff)
    for sec in dataseta._data.values():
        assert numpy.allclose(sec.coeff, coeff)

def test_apply_error():
    data = fqe_data.FqeData(0, 0, 0)
    test1 = {(0, 0): data}
    test2 = {(0, 1): data}
    set1 = fqe_data_set.FqeDataSet(0, 0, test1)
    set2 = fqe_data_set.FqeDataSet(0, 0, test2)
    with pytest.raises(ValueError):
        set1.ax_plus_y(1.0, set2)

    arr = numpy.empty(0)
    with pytest.raises(ValueError):
        set1.apply((arr, arr, arr, arr, arr))

def test_apply1_inplace():
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    h1 = build_H1(norb, full=True)
    out = test.apply((h1,))
    test.apply_inplace((h1,))
    tsectors = test.sectors()
    osectors = out.sectors()
    for x in tsectors.keys():
        tsec = tsectors[x]
        osec = osectors[x]
        assert numpy.allclose(tsec.coeff, osec.coeff)

def test_apply1(c_or_python):
    """Test applications of 1-particle operator"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    h1 = build_H1(norb, full=True)
    ref = loader.get_href('1')
    out = test.apply((h1,))
    rsectors = ref.sectors()
    osectors = out.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)

def test_apply12(c_or_python):
    """Test applications of 1,2-particle operators"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    h1 = build_H1(norb, full=True)
    h2 = build_H2(norb, full=True)
    ref = loader.get_href('12')
    out = test.apply((h1, h2))
    rsectors = ref.sectors()
    osectors = out.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)

def test_apply123(c_or_python):
    """Test applications of 1,2,3-particle operators"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    h1 = build_H1(norb, full=True)
    h2 = build_H2(norb, full=True)
    h3 = build_H3(norb, full=True)
    ref = loader.get_href('123')
    out = test.apply((h1, h2, h3))
    rsectors = ref.sectors()
    osectors = out.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)

def test_apply1234(c_or_python):
    """Test applications of 1,2,3,4-particle operators"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    h1 = build_H1(norb, full=True)
    h2 = build_H2(norb, full=True)
    h3 = build_H3(norb, full=True)
    h4 = build_H4(norb, full=True)
    ref = loader.get_href('1234')
    out = test.apply((h1, h2, h3, h4))
    rsectors = ref.sectors()
    osectors = out.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)

def test_apply1_onecolumn(c_or_python):
    """Test applications of 1-particle operator column-by-column"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    h1 = build_H1(norb, full=True)
    ref = loader.get_href('1')
    h10 = numpy.zeros((2*norb, 2*norb))
    h10[:, 0] = h1[:, 0]
    out1 = test.apply((h10,))
    for i in range(1, 2*norb):
        h1x = numpy.zeros(h10.shape)
        h1x[:, i] = h1[:, i]
        out1.ax_plus_y(1, test.apply((h1x,)))
    rsectors = ref.sectors()
    osectors = out1.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)

def test_rdm1(c_or_python):
    """Test computation of 1-RDM"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    od1 = test.rdm1()
    rd1 = loader.get_rdm(1)
    assert numpy.allclose(rd1, od1)

def test_rdm12(c_or_python):
    """Test computation of 1- and 2-RDM"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    od1, od2 = test.rdm12()
    rd1 = loader.get_rdm(1)
    rd2 = loader.get_rdm(2)
    assert numpy.allclose(rd1, od1)
    assert numpy.allclose(rd2, od2)

def test_rdm123(c_or_python):
    """Test computation of 1-, 2- and 3-RDM"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    od1, od2, od3 = test.rdm123()
    rd1 = loader.get_rdm(1)
    rd2 = loader.get_rdm(2)
    rd3 = loader.get_rdm(3)
    assert numpy.allclose(rd1, od1)
    assert numpy.allclose(rd2, od2)
    assert numpy.allclose(rd3, od3)

def test_rdm1234(c_or_python):
    """Test computation of 1-, 2-, 3- and 4-RDM"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    od1, od2, od3, od4 = test.rdm1234()
    rd1 = loader.get_rdm(1)
    rd2 = loader.get_rdm(2)
    rd3 = loader.get_rdm(3)
    rd4 = loader.get_rdm(4)
    assert numpy.allclose(rd1, od1)
    assert numpy.allclose(rd2, od2)
    assert numpy.allclose(rd3, od3)
    assert numpy.allclose(rd4, od4)

def test_indv_nbody(c_or_python):
    """Test application of an individual N-body operator"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    daga = [0]
    undaga = [1]
    dagb = [0]
    undagb = [0]
    ref = loader.get_indv_ref(daga, undaga, dagb, undagb)
    out = test.apply_individual_nbody(
        complex(1), daga, undaga, dagb, undagb)
    rsectors = ref.sectors()
    osectors = out.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)

def test_evolve_indv_nbody(c_or_python):
    """Test application of an individual N-body operator"""
    fqe.settings.use_accelerated_code = c_or_python
    norb = 2
    nelec = 2

    loader = FqeDataSetLoader(nelec, norb)
    test = loader.get_fqe_data_set()
    daga = [0]
    undaga = [1]
    dagb = [0]
    undagb = [0]
    ref = loader.get_ievo_ref(daga, undaga, dagb, undagb)
    out = test.evolve_individual_nbody(
        0.1, complex(1), daga, undaga, dagb, undagb)
    rsectors = ref.sectors()
    osectors = out.sectors()
    for x in rsectors.keys():
        rsec = rsectors[x]
        osec = osectors[x]
        assert numpy.allclose(rsec.coeff, osec.coeff)
