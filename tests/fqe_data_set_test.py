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
"""Unittesting for the fqe_data module
"""

import unittest

import numpy

from fqe import fqe_data
from fqe import fqe_data_set


class FqeDataSetTest(unittest.TestCase):
    """Unit tests for FqeDataSet
    """
    def test_init_sectors(self):
        data1 = fqe_data.FqeData(2, 0, 2)
        data2 = fqe_data.FqeData(2, 2, 2)
        val = 1.2 + 3.1j
        data1.coeff[:, :] = val
        data2.coeff[:, :] = val
        test = {(1, 1): data1, (2, 0): data2}
        dataset = fqe_data_set.FqeDataSet(0, 0, test)

        sectors = dataset.sectors()
        self.assertTrue(set(sectors.keys()) == set([(2, 0), (1, 1)]))
        for sec in sectors.values():
            self.assertTrue(numpy.allclose(sec.coeff, val))

    def test_empty_copy(self):
        data1 = fqe_data.FqeData(2, 0, 2)
        data2 = fqe_data.FqeData(2, 2, 2)
        val = 1.2 + 3.1j
        data1.coeff[:, :] = val
        data2.coeff[:, :] = val
        test = {(1, 1): data1, (2, 0): data2}
        dataset = fqe_data_set.FqeDataSet(0, 0, test)

        dataset2 = dataset.empty_copy()

        sectors = dataset2.sectors()
        self.assertTrue(set(sectors.keys()) == set([(2, 0), (1, 1)]))
        for sec in sectors.values():
            self.assertTrue(numpy.allclose(sec.coeff, 0.0))

    def test_ax_plus_y_scale_fill(self):
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
            self.assertTrue(numpy.allclose(sec.coeff, 1.0 + coeff*2.3))
        for sec in datasetb._data.values():
            self.assertTrue(numpy.allclose(sec.coeff, 2.3))

        dataseta.scale(coeff)
        for sec in dataseta._data.values():
            self.assertTrue(numpy.allclose(sec.coeff, coeff * (1.0 + coeff*2.3)))

        dataseta.fill(coeff)
        for sec in dataseta._data.values():
            self.assertTrue(numpy.allclose(sec.coeff, coeff))

    def test_apply_error(self):
        data = fqe_data.FqeData(0, 0, 0)
        test1 = {(0, 0): data}
        test2 = {(0, 1): data}
        set1 = fqe_data_set.FqeDataSet(0, 0, test1)
        set2 = fqe_data_set.FqeDataSet(0, 0, test2)
        self.assertRaises(ValueError, set1.ax_plus_y, 1.0, set2)

        arr = numpy.empty(0)
        self.assertRaises(ValueError, set1.apply, (arr, arr, arr, arr, arr))

