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
"""Tests for FqeOperator."""

import unittest
from fqe.fqe_ops import fqe_operator
from fqe import wavefunction


class TestFqeOperator(unittest.TestCase):
    """Test class for the base FqeOperator"""

    def test_operator(self):
        """Testing base FqeOperator using a dummy class"""

        # The Test class is just to make sure the Hamiltonian class is tested.
        # pylint: disable=useless-super-delegation
        class Test(fqe_operator.FqeOperator):
            """A testing dummy class."""

            def contract(
                    self,
                    brastate: "wavefunction.Wavefunction",
                    ketstate: "wavefunction.Wavefunction",
            ) -> complex:
                return super().contract(brastate, ketstate)

            def representation(self) -> str:
                return super().representation()

            def rank(self) -> int:
                return super().rank()

        test = Test()
        wfn = wavefunction.Wavefunction([[1, 0, 1]])
        self.assertAlmostEqual(0.0 + 0.0j, test.contract(wfn, wfn))
        self.assertEqual("fqe-operator", test.representation())
        self.assertEqual(0, test.rank())
