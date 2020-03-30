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

from openfermion import FermionOperator

from fqe.wavefunction import Wavefunction
from fqe import get_spin_conserving_wavefunction
from fqe import get_number_conserving_wavefunction
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import sparse_hamiltonian
from fqe.hamiltonians import diagonal_hamiltonian

from fqe.unittest_data import build_wfn, build_hamiltonian


class WavefunctionTest(unittest.TestCase):
    """Unit tests
    """


    def test_init_exceptions(self):
        """Check that wavefunction throws the proper errors when an incorrect
        initialization is passed.
        """
        self.assertRaises(TypeError, Wavefunction, broken=['spin', 'number'])
        self.assertRaises(ValueError, Wavefunction, param=[[0, 0, 2], [0, 0, 4]])


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
        ref = 2.0*numpy.ones((4, 4), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(ref, work._civec[(2, 0)].coeff))
        work = test1 - test2
        ref = numpy.zeros((4, 4), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(ref, work._civec[(2, 0)].coeff))


    def test_apply_number(self):
        norb = 4
        test = numpy.random.rand(norb, norb)
        diag = numpy.random.rand(norb*2)
        diag2 = copy.deepcopy(diag)
        e_0 = 0
        for i in range(norb):
            e_0 += diag[i + norb]
            diag2[i + norb] = - diag[i + norb]
        hamil = diagonal_hamiltonian.Diagonal(diag2, e_0=e_0)
        hamil._conserve_number = False
        wfn = Wavefunction([[4, 2, norb]], broken=['number'])
        wfn.set_wfn(strategy='from_data', raw_data={(4, 2): test})
        out1 = wfn.apply(hamil)

        hamil = diagonal_hamiltonian.Diagonal(diag)
        wfn = Wavefunction([[4, 2, norb]])
        wfn.set_wfn(strategy='from_data', raw_data={(4, 2): test})
        out2 = wfn.apply(hamil)

        self.assertTrue(numpy.allclose(out1._civec[(4, 2)].coeff,
                                       out2._civec[(4, 2)].coeff))


    def test_apply_type_error(self):
        data = numpy.zeros((2, 2), dtype=numpy.complex128)
        wfn = Wavefunction([[2, 0, 2]], broken=['spin'])
        hamil = general_hamiltonian.General((data, ))
        hamil._conserve_number = False
        self.assertRaises(TypeError, wfn.apply, hamil)
        self.assertRaises(TypeError, wfn.time_evolve, 0.1, hamil)

        wfn = Wavefunction([[2, 0, 2]], broken=['number'])
        hamil = general_hamiltonian.General((data, ))
        self.assertRaises(TypeError, wfn.apply, hamil)
        self.assertRaises(TypeError, wfn.time_evolve, 0.1, hamil)


    def test_apply_individual_nbody_error(self):
        fop = FermionOperator('1^ 0')
        fop += FermionOperator('2^ 0')
        fop += FermionOperator('2^ 1')
        hamil = sparse_hamiltonian.SparseHamiltonian(2, fop)
        wfn = Wavefunction([[2, 0, 2]], broken=['spin'])
        self.assertRaises(ValueError, wfn._apply_individual_nbody, hamil)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        fop = FermionOperator('1^ 0')
        fop += FermionOperator('2^ 0')
        hamil = sparse_hamiltonian.SparseHamiltonian(2, fop)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        fop = FermionOperator('1^ 0', 1.0)
        fop += FermionOperator('0^ 1', 0.9)
        hamil = sparse_hamiltonian.SparseHamiltonian(2, fop)
        self.assertRaises(ValueError, wfn._evolve_individual_nbody, 0.1, hamil)

        fop = FermionOperator('1^ 0^')
        hamil = sparse_hamiltonian.SparseHamiltonian(2, fop)
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
        hamil = sparse_hamiltonian.SparseHamiltonian(2, fop)
        out1 = wfn._apply_few_nbody(hamil)

        fop = FermionOperator('1 1^', fac)
        hamil = sparse_hamiltonian.SparseHamiltonian(2, fop)
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


    def test_expectatoin_value_type_error(self):
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
            self.assertTrue(numpy.allclose(read_wfn._civec[key].coeff, wfn._civec[key].coeff))
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
            self.assertTrue(numpy.allclose(read_wfn._civec[key].coeff, wfn._civec[key].coeff))
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
        wfn.set_wfn(strategy='random')
        ref_string = 'Sector N = 3 : S_z = -3\n' + \
        'a\'000\'b\'111\' (-0.965716165103582-0.2596002474144259j)\n' + \
        'Sector N = 3 : S_z = -1\n' + \
        'a\'001\'b\'011\' (-0.36353371799047474-0.20659882336491125j)\n' + \
        'a\'001\'b\'101\' (-0.2595794430985754-0.21559318752883358j)\n' + \
        'a\'001\'b\'110\' (0.2245720691863791+0.04852785090765824j)\n' + \
        'a\'010\'b\'011\' (-0.3343371194649537-0.18162164502182865j)\n' + \
        'a\'010\'b\'101\' (-0.12276329653757072+0.2598595053848905j)\n' + \
        'a\'010\'b\'110\' (-0.07129059307188947-0.09903086482985644j)\n' + \
        'a\'100\'b\'011\' (-0.21689451722319453-0.37961322467516906j)\n' + \
        'a\'100\'b\'101\' (-0.08335325115398513-0.3320831963824638j)\n' + \
        'a\'100\'b\'110\' (0.3223504737186291-0.06300341495552426j)\n' + \
        'Sector N = 3 : S_z = 1\n' + \
        'a\'011\'b\'001\' (-0.21405181650984867+0.291191428014912j)\n' + \
        'a\'011\'b\'010\' (-0.27528122914339537+0.17928779581227006j)\n' + \
        'a\'011\'b\'100\' (-0.03830344247705324-0.1018560909069887j)\n' + \
        'a\'101\'b\'001\' (-0.45862002455262096+0.15967403671706776j)\n' + \
        'a\'101\'b\'010\' (-0.38283591522104243+0.11908329862006684j)\n' + \
        'a\'101\'b\'100\' (0.4116282600794628+0.010105890130903789j)\n' + \
        'a\'110\'b\'001\' (0.1076905656249564-0.00752210752071855j)\n' + \
        'a\'110\'b\'010\' (0.11663872769596699-0.22956164504983004j)\n' + \
        'a\'110\'b\'100\' (-0.16087960736695867+0.2822626579924094j)\n' + \
        'Sector N = 3 : S_z = 3\n' + \
        'a\'111\'b\'000\' (0.803446320104347-0.5953772003619078j)\n'
        save_stdout = sys.stdout
        sys.stdout = chkprint = StringIO()
        wfn.print_wfn()
        sys.stdout = save_stdout
        outstring = chkprint.getvalue()
        self.assertEqual(outstring, ref_string)

        wfn.set_wfn(strategy='random')
        ref_string = 'Sector N = 3 : S_z = -3\n' + \
        'bbb (-0.8113698523271319-0.5845331151736812j)\n' + \
        'Sector N = 3 : S_z = -1\n' + \
        '.b2 (0.27928557602787163-0.1474291874811043j)\n' + \
        'b.2 (-0.1665913776204976-0.3617026012726579j)\n' + \
        'bba (-0.34237638530199677-0.3478680323908946j)\n' + \
        '.2b (-0.06261445720131753+0.06768497529405092j)\n' + \
        'bab (-0.38139927374414034+0.1861924936737463j)\n' + \
        'b2. (0.12088212990158276+0.017989309605196964j)\n' + \
        'abb (0.21003022341703897+0.1796342715165676j)\n' + \
        '2.b (0.025719719969361773-0.26643597861625606j)\n' + \
        '2b. (-0.3300411848918476+0.2071714738026307j)\n' + \
        'Sector N = 3 : S_z = 1\n' + \
        '.a2 (0.04353310004001345-0.28962822210805944j)\n' + \
        '.2a (-0.24253795700144656+0.4082951994423171j)\n' + \
        'baa (0.0416027021668677+0.14568595964440914j)\n' + \
        'a.2 (-0.4200764867734443-0.3368997907424329j)\n' + \
        'aba (0.011316109760530853+0.14538028430182576j)\n' + \
        '2.a (0.10466970722751164-0.3036806837765713j)\n' + \
        'aab (-0.4332181974443824-0.06627315601193698j)\n' + \
        'a2. (-0.10718975397926216+0.2170023330304916j)\n' + \
        '2a. (0.0013687148060600703+0.026018656390685173j)\n' + \
        'Sector N = 3 : S_z = 3\n' + \
        'aaa (-0.6533937058927956-0.757018272632622j)\n'
        save_stdout = sys.stdout
        sys.stdout = chkprint = StringIO()
        wfn.print_wfn(fmt='occ')
        sys.stdout = save_stdout
        outstring = chkprint.getvalue()
        self.assertEqual(outstring, ref_string)
