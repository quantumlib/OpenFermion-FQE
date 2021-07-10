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
"""Unit tests for fqe utils."""

import unittest

from fqe.fqe_ops import fqe_ops_utils


class TestFqeUtils(unittest.TestCase):
    """Test class for fqe utils."""

    def test_validate_rdm_string(self):
        """Check that the string passed in to rdm access is valid."""
        bad = "1d 2"
        self.assertRaises(TypeError, fqe_ops_utils.validate_rdm_string, bad)
        bad = "td s"
        self.assertRaises(TypeError, fqe_ops_utils.validate_rdm_string, bad)
        bad = "t^ s^"
        self.assertRaises(ValueError, fqe_ops_utils.validate_rdm_string, bad)
        rdm1 = "1^ 2"
        rdm2 = "7^ 8 23 1^"
        rdm3 = "0^ 1 2^ 3 4^ 5"
        rdm4 = "10^ 9^ 8^ 7^ 6 5 4 3"
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm1), "element")
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm2), "element")
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm3), "element")
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm4), "element")
        rdm1 = "i^ j"
        rdm2 = "k f^ l t^"
        rdm3 = "x^ q w^ k u^ m"
        rdm4 = "v^ b^ n^ m^ d f g h"
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm1), "tensor")
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm2), "tensor")
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm3), "tensor")
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm4), "tensor")

    def test_switch_broken(self):
        """Test that the strings properly switch between number and spin broken
        representation.
        """
        unchanged = "0^ 2 10^ 4 6^ 18"
        self.assertEqual(unchanged,
                         fqe_ops_utils.switch_broken_symmetry(unchanged))
        numberbroken = "0^ 1^ 4^ 6 5^ 7"
        self.assertEqual(
            "0^ 1 4^ 6 5 7^",
            fqe_ops_utils.switch_broken_symmetry(numberbroken),
        )
        spinbroken = "0^ 6 7^ 2 10^ 5"
        self.assertEqual("0^ 6 7 2 10^ 5^",
                         fqe_ops_utils.switch_broken_symmetry(spinbroken))
        self.assertEqual("t^ s", fqe_ops_utils.switch_broken_symmetry("t^ s"))
