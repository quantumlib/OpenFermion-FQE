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

""" fci_graph unit tests
"""

import unittest

from fqe import fci_graph

class FciGraphTest(unittest.TestCase):
    """ Unit tests
    """


    def test_fci_graph_validate_nparticles(self):
        """There should not be more particles than orbitals
        """
        self.assertRaises(ValueError, fci_graph.FciGraph, 4, 4, 1)
        self.assertRaises(ValueError, fci_graph.FciGraph, 4, -4, 1)


    def test_fci_graph_validate_existence(self):
        """Negative numbers of particles in not useful
        """
        self.assertRaises(ValueError, fci_graph.FciGraph, -8, 0, 1)


    def test_fci_graph_index_range_alpha(self):
        """Accessing elements outside of our space is not allowed
        """
        testgraph = fci_graph.FciGraph(1, 0, 4)
        self.assertRaises(IndexError, testgraph.get_alpha, 25)


    def test_fci_graph_index_range_beta(self):
        """No determinants are indexed with negative values.
        """
        testgraph = fci_graph.FciGraph(0, 1, 4)
        self.assertRaises(IndexError, testgraph.get_beta, -25)


    def test_fci_graph_check_value_min(self):
        """The bitstring corresponding to the occupation of the lowest n
        orbitals and the first element accessed should be sum_{n} 2**n.
        """
        testgraph = fci_graph.FciGraph(4, 4, 8)
        self.assertEqual(15, testgraph.get_alpha(0))


    def test_fci_graph_check_value_max(self):
        """The bitstring corresponding to the occupation of the highest n
        orbitals will be in bitwise operations

            (1<<norbs) - (1<<(norbs - nele)) - 2

        """
        testgraph = fci_graph.FciGraph(4, 4, 8)
        self.assertEqual(240, testgraph.get_beta(69))


    def test_fci_graph_vacuum(self):
        """The vacuum should just be a coefficient
        """
        testgraph = fci_graph.FciGraph(0, 0, 8)
        self.assertEqual(0, testgraph.get_alpha(0))
        self.assertEqual(0, testgraph.get_beta(0))
