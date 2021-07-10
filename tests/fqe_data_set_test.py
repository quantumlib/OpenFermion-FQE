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

    def test_apply_error(self):
        data = fqe_data.FqeData(0, 0, 0)
        test1 = {(0, 0): data}
        test2 = {(0, 1): data}
        set1 = fqe_data_set.FqeDataSet(0, 0, test1)
        set2 = fqe_data_set.FqeDataSet(0, 0, test2)
        self.assertRaises(ValueError, set1.ax_plus_y, 1.0, set2)

        arr = numpy.empty(0)
        self.assertRaises(ValueError, set1.apply, (arr, arr, arr, arr, arr))
