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

import unittest

from scipy import special
from fqe import fci_graph


class FciGraphTest(unittest.TestCase):
    """ Unit tests
    """

    def test_fci_graph(self):
        """Check the basic initializers and getter functions.
        """
        reflist = [15, 23, 39, 71, 135, 27, 43, 75, 139, 51, 83, 147, 99, \
                    163, 195, 29, 45, 77, 141, 53, 85, 149, 101, 165, 197, \
                    57, 89, 153, 105, 169, 201, 113, 177, 209, 225, 30, 46,\
                    78, 142, 54, 86, 150, 102, 166, 198, 58, 90, 154, 106, \
                    170, 202, 114, 178, 210, 226, 60, 92, 156, 108, 172, \
                    204, 116, 180, 212, 228, 120, 184, 216, 232, 240]
        refdict = {15: 0, 23: 1, 27: 5, 29: 15, 30: 35, 39: 2, 43: 6, 45: 16, \
                    46: 36, 51: 9, 53: 19, 54: 39, 57: 25, 58: 45, 60: 55, \
                    71: 3, 75: 7, 77: 17, 78: 37, 83: 10, 85: 20, 86: 40, \
                    89: 26, 90: 46, 92: 56, 99: 12, 101: 22, 102: 42, 105: 28, \
                    106: 48, 108: 58, 113: 31, 114: 51, 116: 61, 120: 65, \
                    135: 4, 139: 8, 141: 18, 142: 38, 147: 11, 149: 21, \
                    150: 41, 153: 27, 154: 47, 156: 57, 163: 13, 165: 23, \
                    166: 43, 169: 29, 170: 49, 172: 59, 177: 32, 178: 52, \
                    180: 62, 184: 66, 195: 14, 197: 24, 198: 44, 201: 30, \
                    202: 50, 204: 60, 209: 33, 210: 53, 212: 63, 216: 67, \
                    225: 34, 226: 54, 228: 64, 232: 68, 240: 69}
        norb = 8
        nalpha = 4
        nbeta = 0
        lena = int(special.binom(norb, nalpha))
        max_bitstring = (1 << norb) - (1 << (norb - nalpha))
        testgraph = fci_graph.FciGraph(nalpha, nbeta, norb)
        self.assertEqual(
            testgraph._build_string_address(nalpha, norb, [0, 1, 2, 3]), 0)
        self.assertEqual(
            testgraph._build_string_address(nalpha, norb, [1, 2, 3, 7]), 38)
        test_list, test_dict = testgraph._build_strings(nalpha, lena)
        self.assertListEqual(test_list, reflist)
        self.assertDictEqual(test_dict, refdict)
        self.assertEqual(testgraph.string_beta(0), 0)
        self.assertEqual(testgraph.string_alpha(lena - 1), max_bitstring)
        self.assertEqual(testgraph.index_beta(0), 0)
        self.assertEqual(testgraph.index_alpha(max_bitstring), lena - 1)
        self.assertEqual(testgraph.lena(), lena)
        self.assertEqual(testgraph.lenb(), 1)
        self.assertEqual(testgraph.nalpha(), nalpha)
        self.assertEqual(testgraph.nbeta(), nbeta)
        self.assertEqual(testgraph.norb(), norb)
        self.assertEqual(testgraph.string_alpha(lena - 1), max_bitstring)
        self.assertListEqual(testgraph.string_alpha_all(), reflist)
        self.assertListEqual(testgraph.string_beta_all(), [0])
        self.assertDictEqual(testgraph.index_alpha_all(), refdict)
        self.assertDictEqual(testgraph.index_beta_all(), {0: 0})

    def test_fci_graph_maps(self):
        """Check graph mapping functions
        """
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
        alist = [3, 5, 9, 6, 10, 12]
        blist = [1, 2, 4, 8]
        aind = {3: 0, 5: 1, 6: 3, 9: 2, 10: 4, 12: 5}
        bind = {1: 0, 2: 1, 4: 2, 8: 3}
        norb = 4
        nalpha = 2
        nbeta = 1
        testgraph = fci_graph.FciGraph(nalpha, nbeta, norb)
        alpha_map = testgraph._build_mapping(alist, aind)
        self.assertDictEqual(alpha_map, ref_alpha_map)
        beta_map = testgraph._build_mapping(blist, bind)
        self.assertDictEqual(beta_map, ref_beta_map)
        dummy_map = ({(1, 1): (0, 1, 2)}, {(-1, -1), (0, 1, 2)})
        testgraph.insert_mapping(1, -1, dummy_map)
        self.assertEqual(testgraph.find_mapping(1, -1), dummy_map)
