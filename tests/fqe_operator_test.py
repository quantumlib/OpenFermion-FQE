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

from fqe.fqe_ops import fqe_operator
from fqe import wavefunction

def test_operator():
    """Testing abstract FqeOperator class using a dummy class"""
    # pylint: disable=useless-super-delegation
    class TestFQEOperator(fqe_operator.FqeOperator):
        """
        This class is just to make sure the abstract FqeOperator class is tested.
        """
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

    test = TestFQEOperator()
    wfn = wavefunction.Wavefunction([[1, 1, 1]])
    assert round(abs(0.0 + 0.0j-test.contract(wfn, wfn)), 7) == 0
    assert "fqe-operator" == test.representation()
    assert 0 == test.rank()
