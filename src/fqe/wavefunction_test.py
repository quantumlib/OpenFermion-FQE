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

"""Wavefunction class unit tests
"""

import sys

from io import StringIO

import unittest

from openfermion import FermionOperator

import numpy
from numpy import linalg
from scipy.special import binom

from fqe.wavefunction import Wavefunction
from fqe.fqe_data import FqeData


class WavefunctionTest(unittest.TestCase):
    """Unit tests
    """


    def test_wavefunction(self):
        """
        """

#    def test_wavefunction_negative_orbital_error(self):
#        """Negative orbital number is not meaningful
#        """
#        self.assertRaises(ValueError, wavefunction.Wavefunction, [[2, 0, -1]])
#
#
#    def test_wavefunction_orbital_consistency(self):
#        """The number of orbitals should be suffcient to accomodate all the
#        particles
#        """
#        self.assertRaises(ValueError, wavefunction.Wavefunction, [[10, 2, 4]])
#
#
#    def test_wavefunction_particle_conserving(self):
#        """Particle number conserving wavefunctions will check the value of
#        n as its contructed
#        """
#        self.assertRaises(ValueError, wavefunction.Wavefunction,
#                          [[2, 0, 6], [4, 0, 6]], conserveparticlenumber=True)
#
#
#    def test_wavefunction_spin_conserving(self):
#        """Spin conserving wavefunctions will check the value of
#        n as its constructed
#        """
#        self.assertRaises(ValueError, wavefunction.Wavefunction,
#                          [[2, 0, 6], [4, 2, 6]], conservespin=True)
#
#
#    def test_wavefunction_generator(self):
#        """Ensure that we recover information on the wavefunction.
#        """
#        wfn = wavefunction.Wavefunction([[5, 1, 4], [5, 3, 4], [5, -3, 4]])
#        for config in wfn.generator():
#            self.assertEqual(5, config.n_electrons)
#
#
#    def test_wavefunction_max_ele(self):
#        """Ensure that we recover information on the wavefunction.
#        """
#        wfn = wavefunction.Wavefunction([[1, 1, 1]])
#        wfn.set_wfn(strategy='lowest')
#        self.assertAlmostEqual(1. + .0j, wfn.max_element())
#
#
#    def test_wavefunction_data(self):
#        """Ensure that we recover information on the wavefunction.
#        """
#        wfn = wavefunction.Wavefunction([[2, 0, 4], [1, 1, 4], [2, -2, 4]])
#        test = wfn.lena
#        self.assertListEqual([4, 4, 1],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        test = wfn.lenb
#        self.assertListEqual([4, 1, 6],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        test = wfn.gs_a
#        self.assertListEqual([1, 1, 0],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        test = wfn.gs_b
#        self.assertListEqual([1, 0, 3],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        test = wfn.nalpha
#        self.assertListEqual([1, 1, 0],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        test = wfn.nbeta
#        self.assertListEqual([1, 0, 2],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        test = wfn.cidim
#        self.assertListEqual([16, 4, 6],
#                             [test[(2, 0)], test[(1, 1)], test[(2, -2)]])
#        self.assertEqual(wfn.norb, 4)
#
#    def test_set_wfn_with_data(self):
#        """Set wavefunction data from input rather than an internal method
#        """
#        wfn = wavefunction.Wavefunction([[2, 0, 4]])
#        data = {}
#        data[(2, 0)] = .5*numpy.ones((16, 10), dtype=numpy.complex64)
#        wfn.set_wfn(strategy='from_data', raw_data=data)
#        self.assertTrue(numpy.allclose(wfn.get_coeff((2, 0)), data[(2, 0)]))
#
#
#    def test_add_sector_behavior(self):
#        """If an unphysical config would be added then ignore it.  If a config
#        that exists would be added let the user know and ignore it.  The
#        exception is if data is passed the behavior is undefined.
#        """
#        wfn = wavefunction.Wavefunction([[2, 0, 4]])
#        config = wfn.configs
#        wfn.add_sector(-1, -1, 10)
#        self.assertTrue(config == wfn.configs)
#        wfn.add_sector(2, 0, 4)
#        self.assertTrue(config == wfn.configs)
#        self.assertRaises(ValueError, wfn.add_sector, 2, 0, 4, 0)
#
#
#    def test__add__equal_config_wavefunctions(self):
#        """Two wavefunctions with the same configuration should just add
#        together their configurations
#        """
#        wfn1 = wavefunction.Wavefunction([[2, 2, 4]])
#        wfn1.set_wfn(strategy='ones')
#        wfn2 = wavefunction.Wavefunction([[2, 2, 4]])
#        wfn2.set_wfn(strategy='ones')
#        wfn3 = wfn1 + wfn2
#        self.assertEqual(4, wfn3.norb)
#        nalpha = wfn3.nalpha
#        self.assertEqual(2, nalpha[(2, 2)])
#        nbeta = wfn3.nbeta
#        self.assertEqual(0, nbeta[(2, 2)])
#        lena = wfn3.lena
#        lenb = wfn3.lenb
#        cidimlen = lena[(2, 2)]*lenb[(2, 2)]
#        ref = (2. + .0j)*numpy.ones((cidimlen, 1), dtype=numpy.complex64)
#        self.assertTrue(numpy.allclose(wfn3.get_coeff((2, 2)), ref))
#
#
#    def test__add__different_config_wavefunctions(self):
#        """Two wavefunctions with the same configuration should just add
#        together their configurations
#        """
#        wfn1 = wavefunction.Wavefunction([[2, 2, 4]])
#        wfn1.set_wfn(strategy='ones')
#        wfn2 = wavefunction.Wavefunction([[2, 0, 4]])
#        wfn2.set_wfn(strategy='ones')
#        wfn3 = wfn1 + wfn2
#        cidim = wfn3.cidim
#        ref2_2 = numpy.ones((cidim[(2, 2)]), dtype=numpy.complex64)
#        ref2_0 = numpy.ones((cidim[(2, 0)]), dtype=numpy.complex64)
#        self.assertTrue(numpy.allclose(wfn3.get_coeff((2, 2)), ref2_2))
#        self.assertTrue(numpy.allclose(wfn3.get_coeff((2, 0)), ref2_0))
#
#
#    def test__sub__equal_config_wavefunctions(self):
#        """Two wavefunctions with the same configuration should just add
#        together their configurations
#        """
#        wfn1 = wavefunction.Wavefunction([[2, 2, 4]])
#        wfn1.set_wfn(strategy='ones')
#        wfn2 = wavefunction.Wavefunction([[2, 2, 4]])
#        wfn2.set_wfn(strategy='ones')
#        wfn3 = wfn1 - wfn2
#        self.assertEqual(4, wfn3.norb)
#        nalpha = wfn3.nalpha
#        self.assertEqual(2, nalpha[(2, 2)])
#        nbeta = wfn3.nbeta
#        self.assertEqual(0, nbeta[(2, 2)])
#        lena = wfn3.lena
#        lenb = wfn3.lenb
#        cidimlen = lena[(2, 2)]*lenb[(2, 2)]
#        ref = numpy.zeros((cidimlen, 1), dtype=numpy.complex64)
#        self.assertTrue(numpy.allclose(wfn3.get_coeff((2, 2)), ref))
#
#
#    def test__sub__different_config_wavefunctions(self):
#        """Two wavefunctions with the same configuration should just add
#        together their configurations
#        """
#        wfn1 = wavefunction.Wavefunction([[2, 2, 4]])
#        wfn1.set_wfn(strategy='ones')
#        wfn2 = wavefunction.Wavefunction([[2, 0, 4]])
#        wfn2.set_wfn(strategy='ones')
#        wfn3 = wfn1 - wfn2
#        cidim = wfn3.cidim
#        ref2_2 = numpy.ones((cidim[(2, 2)]), dtype=numpy.complex64)
#        ref2_0 = -numpy.ones((cidim[(2, 0)]), dtype=numpy.complex64)
#        self.assertTrue(numpy.allclose(wfn3.get_coeff((2, 2)), ref2_2))
#        self.assertTrue(numpy.allclose(wfn3.get_coeff((2, 0)), ref2_0))
#
#
#    def test__mul__wavefunction(self):
#        """Check that * is properly overloaded for scalars
#        """
#        wfn = wavefunction.Wavefunction([[2, 0, 4]])
#        wfn.set_wfn(strategy='ones')
#        wfn*2.0
#        cidim = wfn.cidim
#        ref = 2.0*numpy.ones((cidim[(2, 0)]), dtype=numpy.complex64)
#        self.assertTrue(numpy.allclose(wfn.get_coeff((2, 0)), ref))
#
#    def test_wavefunction_config_access_error(self):
#        """Check that access of the wavefunction values catches errors.
#        """
#        test = wavefunction.Wavefunction([[2, 0, 2]])
#        test.set_wfn(strategy='random')
#        self.assertIsNone(test.set_ele(567, 346, 0. + .0j))
#        self.assertAlmostEqual(test.get_ele(567, 346), 0. + .0j)
#        self.assertIsNone(test.add_ele(567, 346, 0. + .0j))
#
#
#    def test_wavefunction_config_access(self):
#        """Direct access to the underlying configurations can be useful.
#        """
#        test = wavefunction.Wavefunction([[4, 2, 4]])
#        test.set_wfn(strategy='ones')
#        test.set_ele(7, 4, .333 + .54j)
#        lena = test.lena
#        lenb = test.lenb
#        cilen = lena[(4, 2)]*lenb[(4, 2)]
#        ref = numpy.ones((cilen, 1), dtype=numpy.complex64)
#        ref[2, 0] = .333 + .54j
#        self.assertTrue(numpy.allclose(test.get_coeff((4, 2)), ref))
#        test.add_ele(7, 4, .333 + .54j)
#        ref[2, 0] *= 2. + 0.j
#        self.assertTrue(numpy.allclose(test.get_coeff((4, 2)), ref))
#        self.assertAlmostEqual(test.get_ele(7, 4,), 2.*(.333 + .54j))
#
#
#    def test_wavefunction_apply_orbital_limit(self):
#        """The apply method will fail if you add into an orbital that does not
#        exist
#        """
#        ops = FermionOperator('8^ ', 1.0)
#        test = wavefunction.Wavefunction([[2, 2, 3]])
#        self.assertRaises(ValueError, test.apply, ops)
#
#
#    def test_wavefunction_apply_create(self):
#        """Generate the fully occupied state
#        """
#        ops = FermionOperator('1^ 0^', 1.0)
#        test = wavefunction.Wavefunction([[2, 0, 2]])
#        newwfn = test.apply(ops)
#        newwfn.set_wfn(strategy='ones')
#        lena = newwfn.lena
#        lenb = newwfn.lenb
#        val = numpy.ones((1), dtype=numpy.complex64)
#        self.assertEqual(1, lena[(4, 0)])
#        self.assertEqual(1, lenb[(4, 0)])
#        self.assertEqual(val, newwfn.get_coeff((4, 0)))
#
#
#    def test_wavefunction_apply_annihilate(self):
#        """Remove a beta electron
#        """
#        ops = FermionOperator('1', 1.0)
#        test = wavefunction.Wavefunction([[2, 0, 2]])
#        newwfn = test.apply(ops)
#        newwfn.set_wfn(strategy='ones')
#        ref = numpy.ones(2, dtype=numpy.complex64)
#        self.assertTrue(numpy.allclose(newwfn.get_coeff((1, 1)), ref))
#        ops = FermionOperator('0', 1.0) + FermionOperator('2', 1.0)
#        vacuum = newwfn.apply(ops)
#        self.assertEqual(vacuum.get_coeff((0, 0)), 2. + .0j)
#
#
#    def test_wavefunction_apply_conserve(self):
#        """Make sure that apply doesn't break requested conservation
#        """
#        ops = FermionOperator('4^', 1.0)
#        test = wavefunction.Wavefunction([[2, 0, 4]], conserveparticlenumber=True)
#        self.assertRaises(ValueError, test.apply, ops)
#        test = wavefunction.Wavefunction([[2, -1, 4]], conservespin=True)
#        self.assertRaises(ValueError, test.apply, ops)
#
#
#    def test_wavefunction_apply_unitary_error(self):
#        """Ensure that an error is raised if the input is incorrect.
#        """
#        test = wavefunction.Wavefunction([[2, 0, 4]])
#        ops = FermionOperator('2^ 0', .2 - .3j) + \
#              FermionOperator('0^ 2', .2 - .3j)
#        self.assertRaises(ValueError, test.apply_generated_unitary, ops,
#                          'taylor')
#        ops = FermionOperator('2^ 0', .2 + .3j) + \
#              FermionOperator('0^ 2', .2 - .3j)
#        self.assertRaises(ValueError, test.apply_generated_unitary, ops,
#                          'bestest')
#
#
#    def test_wavefunction_apply_unitary(self):
#        """The Taylor series expansion of an exponential and cheybshev
#        expansion
#        """
#        tay_wfn = wavefunction.Wavefunction([[2, 0, 2]])
#        tay_wfn.set_wfn(strategy='ones')
#        tay_wfn.normalize()
#        ops = FermionOperator('2^ 0', .2 + .3j) + \
#              FermionOperator('0^ 2', .2 - .3j)
#        test_tay = tay_wfn.apply_generated_unitary(ops, 'taylor')
#        che_wfn = wavefunction.Wavefunction([[2, 0, 2]])
#        che_wfn.set_wfn(strategy='ones')
#        che_wfn.normalize()
#        ops = FermionOperator('2^ 0', .2 + .3j) + \
#              FermionOperator('0^ 2', .2 - .3j)
#        test_che = che_wfn.apply_generated_unitary(ops, 'chebyshev')
#        self.assertTrue(numpy.allclose(test_tay.get_coeff((2, 0)),
#                                       test_che.get_coeff((2, 0))))


#    def test_wavefunction_print(self):
#        """Check printing routine for the wavefunction.
#        """
#        refstr = "Configurationnelectrons:2m_s:0Vector:0a'01'b'01':(1+0j)a'01'b'10':(1+0j)a'10'b'01':(1+0j)a'10'b'10':(1+0j)"
#        refstr1 = "Vector:1a'01'b'01':(1+0j)a'01'b'10':(1+0j)a'10'b'01':(1+0j)a'10'b'10':(1+0j)"
#        refocc = "Configurationnelectrons:2m_s:0Vector:0.2:(1+0j)ba:(1+0j)ab:(1+0j)2.:(1+0j)"
#        wfn = wavefunction.Wavefunction([[2, 0, 2]])
#        wfn.set_wfn(strategy='ones')
#        old_stdout = sys.stdout
#        sys.stdout = chkprint = StringIO()
#        wfn.print_wfn()
#        sys.stdout = old_stdout
#        outstring = chkprint.getvalue()
#        test = ''.join(outstring.split(None))
#
#        old_stdout = sys.stdout
#        sys.stdout = chkprint = StringIO()
#        wfn.print_wfn()
#        sys.stdout = old_stdout
#        outstring = chkprint.getvalue()
#        test = ''.join(outstring.split(None))
#
#        old_stdout = sys.stdout
#        sys.stdout = chkprint = StringIO()
#        wfn.print_wfn()
#        sys.stdout = old_stdout
#        outstring = chkprint.getvalue()
#        test = ''.join(outstring.split(None))
#        self.assertEqual(test, refstr)
#
#        old_stdout = sys.stdout
#        sys.stdout = chkprint = StringIO()
#        wfn.print_wfn()
#        sys.stdout = old_stdout
#        outstring = chkprint.getvalue()
#        test = ''.join(outstring.split(None))
#        self.assertEqual(test, refstr + refstr1)
#
#        old_stdout = sys.stdout
#        sys.stdout = chkprint = StringIO()
#        wfn.print_wfn(fmt='occ')
#        sys.stdout = old_stdout
#        outstring = chkprint.getvalue()
#        test = ''.join(outstring.split(None))
#        self.assertEqual(test, refocc)

if __name__ == '__main__':
    unittest.main()
