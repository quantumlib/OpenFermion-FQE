#   Copyright 2021 Google LLC

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
"""Tests for Linalg."""

import pytest
import numpy
from fqe.lib.linalg import _zimatadd, _transpose

def test_zimatadd():
    """Testing zimatadd"""
    data = numpy.random.rand(3, 2) + 1.j * numpy.random.rand(3, 2)
    out = numpy.random.rand(2, 3) + 1.j * numpy.random.rand(2, 3)
    factor = 1.2 + 2.3j
    ref = out + data.T * factor
    _zimatadd(out, data, factor)
    assert numpy.allclose(out, ref) 

    with pytest.raises(ValueError):
        out2 = numpy.random.rand(2, 2, 3)
        _zimatadd(out2, data, factor)

def test_transpose():
    """Testing zimatadd"""
    data = numpy.random.rand(3, 2) + 1.j * numpy.random.rand(3, 2)
    out = numpy.random.rand(2, 3) + 1.j * numpy.random.rand(2, 3)
    _transpose(out, data)
    assert numpy.allclose(out, data.T) 

    with pytest.raises(ValueError):
        out2 = numpy.random.rand(2, 2, 3)
        _transpose(out2, data)

