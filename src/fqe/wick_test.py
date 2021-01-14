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
"""unittest for wick's index utlities
"""

# It is unclear why pylint dislikes zeros_like calls from numpy
#pylint: disable=unsupported-assignment-operation
# Getting all rdms at once is more convenient accessing the protected member
#pylint: disable=protected-access
#pylint: disable=invalid-sequence-index

import unittest

import numpy

from fqe import wick
from fqe.wavefunction import Wavefunction


class TestWick(unittest.TestCase):
    """Wick's test class
    """

    def test_wick(self):
        """Check that wick performs the proper restructuring of the
        density matrix given a string of indexes.
        """
        norb = 4
        nele = 4
        s_z = 0
        wfn = Wavefunction([[nele, s_z, norb]])
        numpy.random.seed(seed=1)
        wfn.set_wfn(strategy='random')
        wfn.normalize()
        rdms = wfn._compute_rdm(4)
        out1 = wick.wick('k j^', list(rdms), True)
        two = numpy.eye(norb, dtype=out1.dtype) * 2.0
        self.assertRaises(ValueError, wick.wick, 'k0 j', list(rdms))
        self.assertTrue(numpy.allclose(two - out1.T, rdms[0]))

        self.assertRaises(ValueError, wick.wick, 'k^ l i^ j', list(rdms), True)
        out2 = wick.wick('k l i^ j^', list(rdms), True)

        h_1 = numpy.zeros_like(out1)
        for i in range(norb):
            h_1[:, :] += out2[:, i, :, i] / (norb * 2 - nele - 1)
        self.assertAlmostEqual(numpy.std(out1 + h_1), 0.)

        out2a = wick.wick('k l^ i^ j', list(rdms), True)
        self.assertAlmostEqual(out2a[2, 3, 0, 1], -rdms[1][0, 3, 2, 1])

        out3 = wick.wick('k l m i^ j^ n^', list(rdms), True)
        h_2 = numpy.zeros_like(out2)
        for i in range(norb):
            h_2[:, :, :, :] += out3[:, i, :, :, i, :] / (norb * 2 - nele - 2)
        self.assertAlmostEqual(numpy.std(out2 - h_2), 0.)

        out4 = wick.wick('k l m x i^ j^ n^ y^', list(rdms), True)
        h_3 = numpy.zeros_like(out3)
        for i in range(norb):
            h_3[:, :, :, :, :, :] += out4[:, i, :, :, :, i, :, :] / (norb * 2 -
                                                                     nele - 3)
        self.assertAlmostEqual(numpy.std(out3 + h_3), 0.)
