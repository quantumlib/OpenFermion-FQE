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
"""Test cases for fqe operators."""

import unittest

from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)
from fqe import wavefunction


class TestFqeOps(unittest.TestCase):
    """Test case class."""

    def test_ops_general(self):
        """Check general properties of the fqe_ops."""
        s_2 = S2Operator()
        self.assertEqual(s_2.representation(), "s_2")
        self.assertEqual(s_2.rank(), 2)

        s_z = SzOperator()
        self.assertEqual(s_z.representation(), "s_z")
        self.assertEqual(s_z.rank(), 2)

        t_r = TimeReversalOp()
        self.assertEqual(t_r.representation(), "T")
        self.assertEqual(t_r.rank(), 2)

        num = NumberOperator()
        self.assertEqual(num.representation(), "N")
        self.assertEqual(num.rank(), 2)

        wfn = wavefunction.Wavefunction([[4, 2, 4], [4, 0, 4]])
        self.assertRaises(ValueError, t_r.contract, wfn, wfn)

        wfn = wavefunction.Wavefunction([[4, -2, 4], [4, 0, 4]])
        self.assertRaises(ValueError, t_r.contract, wfn, wfn)
