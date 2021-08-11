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

import pytest
import numpy

import fqe
from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)
from fqe import wavefunction


def test_ops_general():
    """Check general properties of the fqe_ops."""
    s_2 = S2Operator()
    assert s_2.representation() == "s_2"
    assert s_2.rank() == 2

    s_z = SzOperator()
    assert s_z.representation() == "s_z"
    assert s_z.rank() == 2

    t_r = TimeReversalOp()
    assert t_r.representation() == "T"
    assert t_r.rank() == 2

    num = NumberOperator()
    assert num.representation() == "N"
    assert num.rank() == 2

    wfn = wavefunction.Wavefunction([[4, 2, 4], [4, 0, 4]])
    with pytest.raises(ValueError):
        t_r.contract(wfn, wfn)

    wfn = wavefunction.Wavefunction([[4, -2, 4], [4, 0, 4]])
    with pytest.raises(ValueError):
        t_r.contract(wfn, wfn)


def test_number_operator():
    num = NumberOperator()
    wfn = wavefunction.Wavefunction([[4, 2, 4], [4, 0, 4]])
    wfn.set_wfn('random')
    wfn.normalize()
    num_expec = num.contract(wfn, wfn)
    assert numpy.isclose(num_expec, 4)


def test_S2Operator():
    s2 = S2Operator()
    wfn = wavefunction.Wavefunction([[2, 2, 4]])
    wfn[3, 0] = 1.0
    s2_expec = s2.contract(wfn, wfn)
    assert numpy.isclose(s2_expec, 2)


def test_SzOperator():
    sz = SzOperator()
    wfn = wavefunction.Wavefunction([[2, 2, 4]])
    wfn.set_wfn('random')
    wfn.normalize()
    sz_expec = sz.contract(wfn, wfn)
    assert numpy.isclose(sz_expec, 1)


def test_TimeReversalOp():
    tr = TimeReversalOp()
    wfn = fqe.get_number_conserving_wavefunction(nele=3, norb=4)
    wfn[7, 0] = 1.0 + 1.0j
    wfn[0, 7] = 1.0 - 1.0j
    wfn.normalize()
    wfn.print_wfn()
    tr_expec = tr.contract(wfn, wfn)
    assert numpy.isclose(tr_expec, 0)

    wfn1 = fqe.get_number_conserving_wavefunction(nele=2, norb=4)
    wfn1[1, 1] = 1.0
    wfn1.normalize()
    wfn1.print_wfn()
    tr_expec = tr.contract(wfn1, wfn1)
    assert numpy.isclose(tr_expec, 1)
