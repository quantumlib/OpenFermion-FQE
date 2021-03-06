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
"""Wavefunction class unit tests
"""
#pylint: disable=protected-access

import os
import sys
import copy

from io import StringIO

import unittest

import numpy
from scipy.special import binom

from openfermion import FermionOperator

from fqe.wavefunction import Wavefunction
from fqe import get_spin_conserving_wavefunction
from fqe import get_number_conserving_wavefunction
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import diagonal_hamiltonian
from fqe import get_restricted_hamiltonian

from fqe.unittest_data import build_wfn, build_hamiltonian
from fqe.unittest_data.build_lih_data import build_lih_data


class WavefunctionTest(unittest.TestCase):
    """Unit tests
    """

    def test_init_exceptions(self):
        """Check that wavefunction throws the proper errors when an incorrect
        initialization is passed.
        """
        self.assertRaises(TypeError, Wavefunction, broken=['spin', 'number'])
        self.assertRaises(ValueError,
                          Wavefunction,
                          param=[[0, 0, 2], [0, 0, 4]])

    def test_general_exceptions(self):
        """Test general method exceptions
        """
        test1 = Wavefunction(param=[[2, 0, 4]])
        test2 = Wavefunction(param=[[4, -4, 8]])
        test1.set_wfn(strategy='ones')
        test2.set_wfn(strategy='ones')
        self.assertRaises(ValueError, test1.ax_plus_y, 1.0, test2)
        self.assertRaises(ValueError, test1.__add__, test2)
        self.assertRaises(ValueError, test1.__sub__, test2)
        self.assertRaises(ValueError, test1.set_wfn, strategy='from_data')

    def test_general_functions(self):
        """Test general wavefunction members
        """
        test = Wavefunction(param=[[2, 0, 4]])
        test.set_wfn(strategy='ones')
        self.assertEqual(1. + 0.j, test[(4, 8)])
        test[(4, 8)] = 3.14 + 0.00159j
        self.assertEqual(3.14 + 0.00159j, test[(4, 8)])
        self.assertEqual(3.14 + 0.00159j, test.max_element())
        self.assertTrue(test.conserve_spin())
        test1 = Wavefunction(param=[[2, 0, 4]])
        test2 = Wavefunction(param=[[2, 0, 4]])
        test1.set_wfn(strategy='ones')
        test2.set_wfn(strategy='ones')
        work = test1 + test2
        ref = 2.0 * numpy.ones((4, 4), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(ref, work._civec[(2, 0)].coeff))
        work = test1 - test2
        ref = numpy.zeros((4, 4), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(ref, work._civec[(2, 0)].coeff))

    def test_apply_number(self):
        norb = 4
        test = numpy.random.rand(norb, norb)
        diag = numpy.random.rand(norb * 2)
        diag2 = copy.deepcopy(diag)
        e_0 = 0
        for i in range(norb):
            e_0 += diag[i + norb]
            diag2[i + norb] = -diag[i + norb]
        hamil = diagonal_hamiltonian.Diagonal(diag2, e_0=e_0)
        hamil._conserve_number = False
        wfn = Wavefunction([[4, 2, norb]], broken=['number'])
        wfn.set_wfn(strategy='from_data', raw_data={(4, 2): test})
        out1 = wfn.apply(hamil)

        hamil = diagonal_hamiltonian.Diagonal(diag)
        wfn = Wavefunction([[4, 2, norb]])
        wfn.set_wfn(strategy='from_data', raw_data={(4, 2): test})
        out2 = wfn.apply(hamil)

        self.assertTrue(
            numpy.allclose(out1._civec[(4, 2)].coeff,
                           out2._civec[(4, 2)].coeff))

    def test_apply_type_error(self):
        data = numpy.zeros((2, 2), dtype=numpy.complex128)
        wfn = Wavefunction([[2, 0, 2]], broken=['spin'])
        hamil = general_hamiltonian.General((data,))
        hamil._conserve_number = False
        self.assertRaises(TypeError, wfn.apply, hamil)
        self.assertRaises(TypeError, wfn.time_evolve, 0.1, hamil)

        wfn = Wavefunction([[2, 0, 2]], broken=['number'])
        hamil = general_hamiltonian.General((data,))
        self.assertRaises(TypeError, wfn.apply, hamil)
        self.assertRaises(TypeError, wfn.time_evolve, 0.1, hamil)

        wfn = Wavefunction([[2, 0, 2]])
        hamil = get_restricted_hamiltonian((data,))
        self.assertRaises(ValueError, wfn.time_evolve, 0.1, hamil, True)

    def test_apply_individual_nbody_error(self):
        fop = FermionOperator('1^ 0')
        fop += FermionOperator('2^ 0')
        fop += FermionOperator('2^ 1')
        hamil = sparse_hamiltonian.SparseHamiltonian(fop)
        wfn = Wavefunction([[2, 0, 2]], broken=['spin'])
        self.assertRaises(ValueError, wfn._apply_individual_nbody, hamil)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        fop = FermionOperator('1^ 0')
        fop += FermionOperator('2^ 0')
        hamil = sparse_hamiltonian.SparseHamiltonian(fop)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        fop = FermionOperator('1^ 0', 1.0)
        fop += FermionOperator('0^ 1', 0.9)
        hamil = sparse_hamiltonian.SparseHamiltonian(fop)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        fop = FermionOperator('1^ 0^')
        hamil = sparse_hamiltonian.SparseHamiltonian(fop)
        self.assertRaises(ValueError, wfn._apply_individual_nbody, hamil)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        self.assertRaises(TypeError, wfn._evolve_individual_nbody, 0.1, 1)

    def test_apply_diagonal(self):
        wfn = Wavefunction([[2, 0, 2]])
        wfn.set_wfn(strategy='random')

        data = numpy.random.rand(2)
        hamil = diagonal_hamiltonian.Diagonal(data)
        out1 = wfn._apply_diagonal(hamil)

        fac = 0.5
        hamil = diagonal_hamiltonian.Diagonal(data, e_0=fac)
        out2 = wfn._apply_diagonal(hamil)
        out2.ax_plus_y(-fac, wfn)
        self.assertTrue((out1 - out2).norm() < 1.0e-8)

    def test_apply_nbody(self):
        wfn = Wavefunction([[2, 0, 2]])
        wfn.set_wfn(strategy='random')

        fac = 3.14
        fop = FermionOperator('1^ 1', fac)
        hamil = sparse_hamiltonian.SparseHamiltonian(fop)
        out1 = wfn._apply_few_nbody(hamil)

        fop = FermionOperator('1 1^', fac)
        hamil = sparse_hamiltonian.SparseHamiltonian(fop)
        out2 = wfn._apply_few_nbody(hamil)
        out2.scale(-1.0)
        out2.ax_plus_y(fac, wfn)
        self.assertTrue((out1 - out2).norm() < 1.0e-8)

    def test_rdm(self):
        """Check that the rdms will properly return the energy
        """
        wfn = Wavefunction(param=[[4, 0, 3]])
        work, energy = build_wfn.restricted_wfn_energy()
        wfn.set_wfn(strategy='from_data', raw_data={(4, 0): work})
        rdm1 = wfn.rdm('i^ j')
        rdm2 = wfn.rdm('i^ j^ k l')
        rdm3 = wfn.rdm('i^ j^ k^ l m n')
        rdm4 = wfn.rdm('i^ j^ k^ l^ m n o p')
        h1e, h2e, h3e, h4e = build_hamiltonian.build_restricted(3, full=False)
        expval = 0. + 0.j
        axes = [0, 1]
        expval += numpy.tensordot(h1e, rdm1, axes=(axes, axes))
        axes = [0, 1, 2, 3]
        expval += numpy.tensordot(h2e, rdm2, axes=(axes, axes))
        axes = [0, 1, 2, 3, 4, 5]
        expval += numpy.tensordot(h3e, rdm3, axes=(axes, axes))
        axes = [0, 1, 2, 3, 4, 5, 6, 7]
        expval += numpy.tensordot(h4e, rdm4, axes=(axes, axes))
        self.assertAlmostEqual(expval, energy)

    def test_expectation_value_type_error(self):
        wfn = Wavefunction([[4, 0, 4]])
        self.assertRaises(TypeError, wfn.expectationValue, 1)

    def test_save_read(self):
        """Check that the wavefunction can be properly archived and
        retieved
        """
        numpy.random.seed(seed=409)
        wfn = get_number_conserving_wavefunction(3, 3)
        wfn.set_wfn(strategy='random')
        wfn.save('test_save_read')
        read_wfn = Wavefunction()
        read_wfn.read('test_save_read')
        for key in read_wfn.sectors():
            self.assertTrue(
                numpy.allclose(read_wfn._civec[key].coeff,
                               wfn._civec[key].coeff))
        self.assertEqual(read_wfn._symmetry_map, wfn._symmetry_map)
        self.assertEqual(read_wfn._conserved, wfn._conserved)
        self.assertEqual(read_wfn._conserve_spin, wfn._conserve_spin)
        self.assertEqual(read_wfn._conserve_number, wfn._conserve_number)
        self.assertEqual(read_wfn._norb, wfn._norb)

        os.remove('test_save_read')

        wfn = get_spin_conserving_wavefunction(2, 6)
        wfn.set_wfn(strategy='random')
        wfn.save('test_save_read')
        read_wfn = Wavefunction()
        read_wfn.read('test_save_read')
        for key in read_wfn.sectors():
            self.assertTrue(
                numpy.allclose(read_wfn._civec[key].coeff,
                               wfn._civec[key].coeff))
        self.assertEqual(read_wfn._symmetry_map, wfn._symmetry_map)
        self.assertEqual(read_wfn._conserved, wfn._conserved)
        self.assertEqual(read_wfn._conserve_spin, wfn._conserve_spin)
        self.assertEqual(read_wfn._conserve_number, wfn._conserve_number)
        self.assertEqual(read_wfn._norb, wfn._norb)

        os.remove('test_save_read')

    def test_wavefunction_print(self):
        """Check printing routine for the wavefunction.
        """
        numpy.random.seed(seed=409)
        wfn = get_number_conserving_wavefunction(3, 3)
        sector_alpha_dim, sector_beta_dim = wfn.sector((3, -3)).coeff.shape
        coeffs = numpy.arange(1,
                              sector_alpha_dim * sector_beta_dim + 1).reshape(
                                  (sector_alpha_dim, sector_beta_dim))
        wfn.sector((3, -3)).coeff = coeffs

        sector_alpha_dim, sector_beta_dim = wfn.sector((3, -1)).coeff.shape
        coeffs = numpy.arange(1,
                              sector_alpha_dim * sector_beta_dim + 1).reshape(
                                  (sector_alpha_dim, sector_beta_dim))
        wfn.sector((3, -1)).coeff = coeffs

        sector_alpha_dim, sector_beta_dim = wfn.sector((3, 1)).coeff.shape
        coeffs = numpy.arange(1,
                              sector_alpha_dim * sector_beta_dim + 1).reshape(
                                  (sector_alpha_dim, sector_beta_dim))
        wfn.sector((3, 1)).coeff = coeffs

        sector_alpha_dim, sector_beta_dim = wfn.sector((3, 3)).coeff.shape
        coeffs = numpy.arange(1,
                              sector_alpha_dim * sector_beta_dim + 1).reshape(
                                  (sector_alpha_dim, sector_beta_dim))
        wfn.sector((3, 3)).coeff = coeffs

        ref_string = 'Sector N = 3 : S_z = -3\n' + \
                     "a'000'b'111' 1\n" + \
                     "Sector N = 3 : S_z = -1\n" + \
                     "a'001'b'011' 1\n" + \
                     "a'001'b'101' 2\n" + \
                     "a'001'b'110' 3\n" + \
                     "a'010'b'011' 4\n" + \
                     "a'010'b'101' 5\n" + \
                     "a'010'b'110' 6\n" + \
                     "a'100'b'011' 7\n" + \
                     "a'100'b'101' 8\n" + \
                     "a'100'b'110' 9\n" + \
                     "Sector N = 3 : S_z = 1\n" + \
                     "a'011'b'001' 1\n" + \
                     "a'011'b'010' 2\n" + \
                     "a'011'b'100' 3\n" + \
                     "a'101'b'001' 4\n" + \
                     "a'101'b'010' 5\n" + \
                     "a'101'b'100' 6\n" + \
                     "a'110'b'001' 7\n" + \
                     "a'110'b'010' 8\n" + \
                     "a'110'b'100' 9\n" + \
                     "Sector N = 3 : S_z = 3\n" + \
                     "a'111'b'000' 1\n"
        save_stdout = sys.stdout
        sys.stdout = chkprint = StringIO()
        wfn.print_wfn()
        sys.stdout = save_stdout
        outstring = chkprint.getvalue()
        self.assertEqual(outstring, ref_string)

        wfn.print_wfn(fmt='occ')
        ref_string = "Sector N = 3 : S_z = -3\n" + \
                     "bbb 1\n" + \
                     "Sector N = 3 : S_z = -1\n" + \
                     ".b2 1\n" + \
                     "b.2 2\n" + \
                     "bba 3\n" + \
                     ".2b 4\n" + \
                     "bab 5\n" + \
                     "b2. 6\n" + \
                     "abb 7\n" + \
                     "2.b 8\n" + \
                     "2b. 9\n" + \
                     "Sector N = 3 : S_z = 1\n" + \
                     ".a2 1\n" + \
                     ".2a 2\n" + \
                     "baa 3\n" + \
                     "a.2 4\n" + \
                     "aba 5\n" + \
                     "2.a 6\n" + \
                     "aab 7\n" + \
                     "a2. 8\n" + \
                     "2a. 9\n" + \
                     "Sector N = 3 : S_z = 3\n" + \
                     "aaa 1\n"
        save_stdout = sys.stdout
        sys.stdout = chkprint = StringIO()
        wfn.print_wfn(fmt='occ')
        sys.stdout = save_stdout
        outstring = chkprint.getvalue()
        self.assertEqual(outstring, ref_string)

    def test_hartree_fock_init(self):
        h1e, h2e, _ = build_lih_data('energy')
        elec_hamil = get_restricted_hamiltonian((h1e, h2e))
        norb = 6
        nalpha = 2
        nbeta = 2
        wfn = Wavefunction([[nalpha + nbeta, nalpha - nbeta, norb]])
        wfn.print_wfn()
        wfn.set_wfn(strategy='hartree-fock')
        wfn.print_wfn()
        self.assertEqual(wfn.expectationValue(elec_hamil), -8.857341498221992)
        hf_wf = numpy.zeros((int(binom(norb, 2)), int(binom(norb, 2))))
        hf_wf[0, 0] = 1.
        self.assertTrue(numpy.allclose(wfn.get_coeff((4, 0)), hf_wf))

        wfn = Wavefunction([[nalpha + nbeta, nalpha - nbeta, norb],
                            [nalpha + nbeta, 2, norb]])
        self.assertRaises(ValueError, wfn.set_wfn, strategy='hartree-fock')
