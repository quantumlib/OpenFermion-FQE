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

"""Unittesting for the fqedata module
"""

import unittest

import numpy

from fqe import fqedata


class FqeDataTest(unittest.TestCase):
    """Unit tests
    """


    def test_fqedata_negative_particle_error(self):
        """Negative particle number is not meaningful
        """
        self.assertRaises(ValueError, fqedata.FqeData, -1, 0, 8)
        self.assertRaises(ValueError, fqedata.FqeData, 0, -1, 8)


    def test_fqedata_negative_orbital_error(self):
        """Negative orbital number is not meaningful
        """
        self.assertRaises(ValueError, fqedata.FqeData, 1, 2, -2)


    def test_fqedata_orbital_consistency(self):
        """The number of orbitals should be sufficient to accommodate all the
        particles
        """
        self.assertRaises(ValueError, fqedata.FqeData, 4, 4, 2)


    def test_fqedata_init(self):
        """Check that we initialize the private values
        """
        test = fqedata.FqeData(2, 4, 10)
        self.assertEqual(test.n_electrons, 6)
        self.assertEqual(test.m_s, -2)
        self.assertEqual(test.nalpha, 2)
        self.assertEqual(test.nbeta, 4)
        self.assertEqual(test.lena, 45)
        self.assertEqual(test.lenb, 210)
        self.assertEqual(test.ci_space_length, 45*210)


    def test_fqedata_initialization(self):
        """Make sure that the fci string graph is created
        """
        test = fqedata.FqeData(1, 0, 1)
        self.assertEqual(1, test.str_alpha(0))


    def test_fqedata_civec_lim(self):
        """Check that we cannot access values beyond the limit of the ci space
        """
        test = fqedata.FqeData(1, 1, 2)
        self.assertRaises(IndexError, test.cc_i, 1, 1, 4)
        self.assertRaises(IndexError, test.ci_i, 5, 4)
        self.assertRaises(IndexError, test.cc_s, 2, 2, 4)


    def test_fqedata_scale(self):
        """Scale the entire vector
        """
        test = fqedata.FqeData(1, 1, 2)
        test.scale(0. + .0j)
        ref = numpy.zeros((4, 3), dtype=numpy.complex64)
        self.assertTrue(numpy.allclose(test.coeff, ref))


    def test_fqedata_generator(self):
        """Access each element of any given vector
        """
        test = fqedata.FqeData(1, 1, 2)
        gtest = test.insequence_generator(0)
        testx = next(gtest)
        self.assertListEqual([.0 + .0j, 1, 1], testx)
        testx = next(gtest)
        self.assertListEqual([.0 + .0j, 1, 2], testx)
        testx = next(gtest)
        self.assertListEqual([.0 + .0j, 2, 1], testx)
        testx = next(gtest)
        self.assertListEqual([.0 + .0j, 2, 2], testx)


    def test_fqedata_set_add_element_and_retrieve(self):
        """Set elements and retrieve them one by one
        """
        test = fqedata.FqeData(1, 1, 2)
        valtest = numpy.array([3.14 + .00159j, 1.61 + .00803j, 2.71 + .00828j],
                              dtype=numpy.complex64)
        test.set_element(2, 2, 0, 3.14 + .00159j)
        self.assertEqual(test.cc_s(2, 2, 0), valtest[0])
        test.set_element(2, 2, 1, 1.61 + .00803j)
        self.assertEqual(test.ci_i(3, 1), valtest[1])
        test.set_element(2, 2, 2, 2.71 + .00828j)
        self.assertEqual(test.cc_i(1, 1, 2), valtest[2])
        test.add_element(2, 2, 2, 2.71 + .00828j)
        self.assertEqual(test.cc_i(1, 1, 2), 2.*valtest[2])


    def test_fqedata_init_vec(self):
        """Set vectors in the fqedata set using different strategies
        """
        test = fqedata.FqeData(1, 1, 2)
        ref = numpy.ones((test.coeff.shape), dtype=numpy.complex64)
        ref[:, 1].fill(0. + .0j)
        ref[1:, 2].fill(0. + .0j)
        test.set_wfn(vrange=[0], strategy='ones')
        test.set_wfn(vrange=[1], strategy='zero')
        test.set_wfn(vrange=[2], strategy='lowest')
        self.assertTrue(numpy.allclose(test.coeff, ref))
        test = fqedata.FqeData(1, 1, 2)
        self.assertIsNone(test.set_wfn(vrange=[0], strategy='random'))
        self.assertIsNone(test.set_wfn(strategy='random'))
        test.set_wfn(vrange=[0], strategy='lowest')
        ref = numpy.zeros((test.coeff.shape), dtype=numpy.complex64)
        ref[0, 0] = 1. + .0j
        self.assertTrue(numpy.allclose(test.coeff[:, 0], ref[:, 0]))
        test.set_wfn(strategy='lowest')
        ref = numpy.zeros((test.coeff.shape), dtype=numpy.complex64)
        ref[0, :] = 1. + .0j
        self.assertTrue(numpy.allclose(test.coeff, ref))


    def test_fqedata_set_wfn_data(self):
        """Set vectors in the fqedata set from a data block
        """
        test = fqedata.FqeData(1, 1, 2)
        ref = numpy.array((numpy.random.rand((4, 3)) +
                           numpy.random.rand((4, 3))*1.j),
                          dtype=numpy.complex64)
        test.set_wfn(strategy='from_data', raw_data=ref)
        self.assertTrue(numpy.allclose(test.coeff, ref))


    def test_fqedata_conj(self):
        """The fqedata can be conjugated in place
        """
        test = fqedata.FqeData(1, 1, 2)
        ref = numpy.array((numpy.random.rand((4, 3)) +
                           numpy.random.rand((4, 3))*1.j),
                          dtype=numpy.complex64)
        test.set_wfn(strategy='from_data', raw_data=ref)
        test.conj
        self.assertTrue(numpy.allclose(test.coeff, numpy.conj(ref)))


    def test_fqedata_normalize(self):
        """ Our vectors should be normalizable
        """
        test = fqedata.FqeData(1, 1, 2)
        test.set_wfn(strategy='ones')
        test.normalize()
        ref = numpy.ones((3, 3), dtype=numpy.complex64)
        test = numpy.dot(test.coeff.conj().T, test.coeff)
        self.assertTrue(numpy.allclose(ref, test))


    def test_fqedata_normalize_error(self):
        """Normalizing a zero wavefunction should be an error
        """
        test = fqedata.FqeData(1, 1, 2)
        test.set_wfn(strategy='zero')
        self.assertRaises(FloatingPointError, test.normalize)


    def test_fqedata_initialize_errors(self):
        """There are many ways to not initialize a wavefunction
        """
        bad0 = numpy.ones((5, 3), dtype=numpy.complex64)
        bad1 = numpy.ones((4, 6), dtype=numpy.complex64)
        good0 = numpy.random.rand(4, 2) + numpy.random.rand(4, 2)*1.j
        good1 = numpy.random.rand(4, 3) + numpy.random.rand(4, 3)*1.j
        test = fqedata.FqeData(1, 1, 2)
        self.assertRaises(ValueError, test.set_wfn)
        self.assertRaises(ValueError, test.set_wfn, strategy='from_data')
        self.assertRaises(ValueError, test.set_wfn, strategy='ones', raw_data=1)
        self.assertRaises(ValueError, test.set_wfn, strategy='onse')
        self.assertRaises(ValueError, test.set_wfn, strategy='from_data',
                          raw_data=bad0)
        self.assertRaises(ValueError, test.set_wfn, strategy='from_data',
                          raw_data=bad1)
        self.assertRaises(ValueError, test.set_wfn, vrange=[-3],
                          strategy='from_data', raw_data=good0)
        self.assertRaises(ValueError, test.set_wfn, vrange=[7],
                          strategy='from_data', raw_data=good0)
        self.assertRaises(ValueError, test.set_wfn, vrange=[0, 0, 0, 0, 0, 0],
                          strategy='from_data', raw_data=good0)
        self.assertIsNone(test.set_wfn(vrange=[0, 2], strategy='from_data',
                                       raw_data=good0))
        self.assertIsNone(test.set_wfn(strategy='from_data', raw_data=good1))


    def test_fqedata_vacuum(self):
        """Make sure that the vacuum exists
        """
        test = fqedata.FqeData(0, 0, 2)
        test.set_wfn(strategy='ones')
        test.normalize()
        ref = numpy.ones((1, 1), dtype=numpy.complex64)
        norm = numpy.dot(test.coeff.conj().T, test.coeff)
        self.assertTrue(numpy.allclose(ref, norm))
        self.assertEqual(test.n_electrons, 0)
        self.assertEqual(test.m_s, 0)
        self.assertEqual(test.nalpha, 0)
        self.assertEqual(test.nbeta, 0)
        self.assertEqual(test.lena, 1)
        self.assertEqual(test.lenb, 1)
        self.assertEqual(test.ci_space_length, 1)
        self.assertEqual(test.ci_configuration_dimension, 1)
