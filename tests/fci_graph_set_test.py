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

import unittest

from fqe.fci_graph_set import FciGraphSet


class TestFciGraphSet(unittest.TestCase):
    """fci graph set contains linkers between sectors in the fci system.
    """

    def test_init_from_params(self):
        """The fci graph set should also be initializable from a prebuilt set
        of parameters.
        """
        params = [[6, 6, 6], [6, 4, 6], [6, 2, 6], [6, 0, 6], [6, -2, 6],
                  [6, -4, 6], [6, -6, 6]]
        test = FciGraphSet(6, 6, params)
        self.assertIsInstance(test, FciGraphSet)
        params = [[3, 3, 6], [3, 1, 6], [3, -1, 6], [2, 0, 6]]
        test = FciGraphSet(3, 3, params)
        self.assertIsInstance(test, FciGraphSet)
