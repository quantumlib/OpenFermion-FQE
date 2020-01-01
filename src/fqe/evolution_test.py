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

import unittest
import random
import copy
import math
import fqe
from openfermion import FermionOperator
from fqe.hamiltonians import diagonal_hamiltonian, gso_hamiltonian
from fqe.hamiltonians import sso_hamiltonian, sparse_hamiltonian
from fqe.hamiltonians import general_hamiltonian, restricted_hamiltonian
from fqe.hamiltonians import hamiltonian_utils, diagonal_coulomb
from fqe.fqe_data import FqeData
from fqe.wavefunction import Wavefunction

import numpy
from numpy import linalg
from scipy import special


class FqeControlTest(unittest.TestCase):


    def setUp(self):
        # === numpy control options ===
        numpy.set_printoptions(floatmode='fixed', precision=6, linewidth=200, suppress=True)
        numpy.random.seed(seed=409)


#    @unittest.SkipTest
    def test_diagonal_evolution(self):
        """Test time evolution of diagonal Hamiltonians
        """
        norb = 8
        nalpha = 5
        nbeta = 1
        time = 0.001
        nele = nalpha + nbeta
        lena = int(special.binom(norb, nalpha))
        lenb = int(special.binom(norb, nbeta))
        cidim = lena*lenb
        
        h1e = numpy.zeros((norb, norb), dtype=numpy.complex128)
        for i in range(norb):
            h1e[i, i] += (i + 1) * 2.0
       
        h_wrap = tuple([h1e])
        # === Reference Wavefunction ===
 
        hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)
 
        for i in range(lena):
            for j in range(lenb):
                ket = FqeData(nalpha, nbeta, norb)
                ket.coeff[i, j] = 1.0
                ket_h = ket.apply(h_wrap)
                hci[:, :, i, j] = ket_h.coeff

        hci = numpy.reshape(hci, (cidim, cidim))
        fci_eig, fci_vec = linalg.eigh(hci)
        test_state = numpy.random.rand(cidim).astype(numpy.complex128)
 
        proj_coeff = fci_vec.conj().T @ test_state
        fci_rep = fci_vec @ proj_coeff
        self.assertTrue(numpy.abs(numpy.linalg.norm(fci_rep - test_state)) < 1.e-8)

        phase = numpy.multiply(numpy.exp(-1.j*time*fci_eig), proj_coeff)
        reference = fci_vec @ phase
    
        test_state = numpy.reshape(test_state, (lena, lenb))
        wfn = Wavefunction([[nele, nalpha-nbeta, norb]])
        wfn.set_wfn(strategy='from_data', raw_data={(nele, nalpha-nbeta): test_state})

        hamil = diagonal_hamiltonian.Diagonal(h1e.diagonal())
        initial_energy = wfn.expectationValue(hamil)
    
        evol_wfn = wfn.time_evolve(time, hamil)
        computed = numpy.reshape(evol_wfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))
        self.assertTrue(numpy.abs(linalg.norm(reference - computed)) < 1.e-8)
    
        final_energy = evol_wfn.expectationValue(hamil)
        self.assertTrue(numpy.abs(final_energy - initial_energy) < 1.e-7)

        tay_wfn = wfn.apply_generated_unitary(time, 'taylor', hamil)
        computed = numpy.reshape(tay_wfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))
        self.assertTrue(numpy.abs(linalg.norm(reference - computed)) < 1.e-8)

        tay_ene = tay_wfn.expectationValue(hamil)
        self.assertTrue(numpy.abs(tay_ene - initial_energy) < 1.e-7)


#    @unittest.SkipTest
    def test_quadratic_number_conserved_spin_broken(self):  
        """
        """      
        time = 0.001
        norb = 4
        
        h1e = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
        
        for i in range(2*norb):
            for j in range(2*norb):
                h1e[i, j] += (i+j) * 0.02
            h1e[i, i] += i * 2.0
        
        hamil = gso_hamiltonian.GSOHamiltonian(tuple([h1e]))

        wfn = fqe.get_spin_nonconserving_wavefunction(4, norb)
        wfn.set_wfn(strategy='random')
    
        initial_energy = wfn.expectationValue(hamil)

        evovlwfn = wfn.time_evolve(time, hamil)
       
        final_energy = evovlwfn.expectationValue(hamil)
    
        err = numpy.abs(final_energy - initial_energy)
        self.assertTrue(err < 1.0e-7)
        
        polywfn = wfn.apply_generated_unitary(time, 'taylor', hamil, expansion=20)
        
        for key in polywfn.sectors():
            with self.subTest(sector=key):
                diff = linalg.norm(polywfn._civec[key].coeff - evovlwfn._civec[key].coeff)
                self.assertTrue(diff < 1.0e-8)
    
        poly_energy = polywfn.expectationValue(hamil)
    
        err = numpy.abs(poly_energy - final_energy)
        self.assertTrue(err < 1.0e-7)
    
    
#    @unittest.SkipTest
    def test_diagonal_coulomb(self):
        """
        """
        norb = 5
        nalpha = 3
        nbeta = 2
        nele = nalpha + nbeta
        m_s = nalpha - nbeta
        time = 0.001
    
        lena = int(special.binom(norb, nalpha))
        lenb = int(special.binom(norb, nbeta))
        cidim = lena*lenb
    
        h1e = numpy.zeros((norb, norb), dtype=numpy.complex128)
        vijkl = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
    
        for i in range(norb):
            for j in range(norb):
                vijkl[i, j, i, j] += 4*(i%norb+1)*(j%norb+1)*0.21
    
    
        h_wrap = tuple([h1e, vijkl])
    
        hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)
        for i in range(lena):
            for j in range(lenb):
                ket = FqeData(nalpha, nbeta, norb)
                ket.coeff[i, j] = 1.0
                ket_h = ket.apply(h_wrap)
                hci[:, :, i, j] = ket_h.coeff
        
        hci = numpy.reshape(hci, (cidim, cidim))
        
        fci_eig, fci_vec = linalg.eigh(hci)

        test_state = numpy.random.rand(cidim).astype(numpy.complex128)
    
        proj_coeff = fci_vec.conj().T @ test_state
        fci_rep = fci_vec @ proj_coeff
    
        err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
        self.assertTrue(err < 1.e-8)
        
        # Time evolve fci_vec
        phase = numpy.zeros(cidim, dtype=numpy.complex128)
        
        for i in range(cidim):
            phase[i] = numpy.exp(-1.j*time*fci_eig[i])*proj_coeff[i]
        
        reference = fci_vec @ phase
        
        test_wfn = Wavefunction([[nele, m_s, norb]])
        test_wfn._civec[(nele, m_s)].coeff = numpy.reshape(test_state, (lena, lenb))

        hamil = diagonal_coulomb.DiagonalCoulomb(vijkl)
        coul_evol = test_wfn.time_evolve(time, hamil)
        
        coeff = coul_evol._civec[(nele, m_s)].coeff
        
        err = numpy.std(reference / reference.flat[0] - coeff.flatten() / coeff.flat[0])
        self.assertTrue(err < 1.e-8)


#    @unittest.SkipTest
    def test_diagonal_spin(self):
        """
        """
        norb = 4
        nalpha = 2
        nbeta = 2
        time = 0.001
        nele = nalpha + nbeta
        lena = int(special.binom(norb, nalpha))
        lenb = int(special.binom(norb, nbeta))
        cidim = lena*lenb

        h1e = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
        
        for i in range(2*norb):
            for j in range(2*norb):
                h1e[i, j] += (i+j) * 0.02
            h1e[i, i] += i * 2.0
        h1e[:norb, norb:] = 0.0
        h1e[norb:, :norb] = 0.0

        h_wrap = tuple([h1e])
        # === Reference Wavefunction ===
 
        hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)
 
        for i in range(lena):
            for j in range(lenb):
                ket = FqeData(nalpha, nbeta, norb)
                ket.coeff[i, j] = 1.0
                ket_h = ket.apply(h_wrap)
                hci[:, :, i, j] = ket_h.coeff

        hci = numpy.reshape(hci, (cidim, cidim))
        fci_eig, fci_vec = linalg.eigh(hci)
 
        test_state = numpy.random.rand(cidim).astype(numpy.complex128)
 
        proj_coeff = fci_vec.conj().T @ test_state
        fci_rep = fci_vec @ proj_coeff
        err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
        self.assertTrue(err < 1.e-8)
 
        phase = numpy.multiply(numpy.exp(-1.j*time*fci_eig), proj_coeff)
 
        reference = fci_vec @ phase

        hamil = sso_hamiltonian.SSOHamiltonian(h_wrap)

        wfn = Wavefunction([[4, 0, norb]])
        test_state = numpy.reshape(test_state, (lena, lenb))
        wfn.set_wfn(strategy='from_data', raw_data={(nele, nalpha-nbeta): test_state})

        evolwfn = wfn.time_evolve(time, hamil)

        computed = numpy.reshape(evolwfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))

        err = numpy.abs(linalg.norm(reference - computed))
        self.assertTrue(err < 1.e-8)


#    @unittest.SkipTest
    def test_nbody(self):
        """
        """
        norb = 4
        nalpha = 3
        nbeta = 0
        nele = nalpha + nbeta
        m_s = nalpha - nbeta
        time = 0.001

        with self.subTest(nbody='one body'):
            ops = FermionOperator('0^ 1', 2.2 - 0.1j) + FermionOperator('1^ 0', 2.2 + 0.1j) 
            sham = sparse_hamiltonian.SparseHamiltonian(norb,
                                                        ops,
                                                        conserve_spin=False,
                                                        conserve_number=True)
            
            h1e = hamiltonian_utils.nbody_matrix(ops, norb)
            
            wfn = fqe.get_spin_nonconserving_wavefunction(nele, norb)
            wfn.set_wfn(strategy='random')
            wfn.normalize()
        
            result = wfn.evolve_individual_nbody(time, sham)
            
            hamil = general_hamiltonian.General(tuple([h1e]))
            nbody_evol = wfn.apply_generated_unitary(time, 'taylor', hamil)
        
            result.ax_plus_y(-1.0, nbody_evol)
            self.assertTrue(result.norm() < 1.e-8)

        
        with self.subTest(nbody='two body'):
            ops = FermionOperator('1^ 3^ 5 0', 2.0 - 2.j) + FermionOperator('0^ 5^ 3 1',  2.0 + 2.j)
            sham = sparse_hamiltonian.SparseHamiltonian(norb, ops, False)
        
            h2e = hamiltonian_utils.nbody_matrix(ops, norb)
       
            h2e = hamiltonian_utils.antisymm_two_body(h2e)
        
            wfn = fqe.get_spin_nonconserving_wavefunction(nele, norb)
            wfn.set_wfn(strategy='random')
            wfn.normalize()
        
            result = wfn.evolve_individual_nbody(time, sham)
            
            h1e = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
            hamil = general_hamiltonian.General(tuple([h1e, h2e]))
            nbody_evol = wfn.apply_generated_unitary(time, 'taylor', hamil)
        
            result.ax_plus_y(-1.0, nbody_evol)
            self.assertTrue(result.norm() < 1.e-8)


        with self.subTest(nbody='three body'):
            ops = FermionOperator('5^ 1^ 3^ 2 0 1', 1.0 - 1.j) + FermionOperator('1^ 0^ 2^ 3 1 5', 1.0 + 1.j)
            sham = sparse_hamiltonian.SparseHamiltonian(norb, ops, False)
    
            h3e = hamiltonian_utils.nbody_matrix(ops, norb)
    
            h3e = hamiltonian_utils.antisymm_three_body(h3e)
    
            wfn = fqe.get_spin_nonconserving_wavefunction(nele, norb)
            wfn.set_wfn(strategy='random')
            wfn.normalize()

            result = wfn.evolve_individual_nbody(time, sham)

            h1e = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
            h2e = numpy.zeros((2*norb, 2*norb, 2*norb, 2*norb), dtype=numpy.complex128)
            hamil = general_hamiltonian.General(tuple([h1e, h2e, h3e]))
            nbody_evol = wfn.apply_generated_unitary(time, 'taylor', hamil, expansion=20)
            result.ax_plus_y(-1.0, nbody_evol)
            self.assertTrue(result.norm() < 1.e-8)


        with self.subTest(nbody='four body'):
            ops = FermionOperator('0^ 2^ 3^ 4^ 1 0 4 2', 1.0 + 0.j) + FermionOperator('2^ 4^ 0^ 1^ 4 3 2 0', 1.0+ 0.j)
            sham = sparse_hamiltonian.SparseHamiltonian(norb, ops, False)
        
            h4e = hamiltonian_utils.nbody_matrix(ops, norb)
    
            h4e = hamiltonian_utils.antisymm_four_body(h4e)
        
            wfn = fqe.get_spin_nonconserving_wavefunction(nele, norb)
            wfn.set_wfn(strategy='random')
            wfn.normalize()
            
            result = wfn.evolve_individual_nbody(time, sham)
            
            h1e = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
            h2e = numpy.zeros((2*norb, 2*norb, 2*norb, 2*norb), dtype=numpy.complex128)
            h3e = numpy.zeros((2*norb, 2*norb, 2*norb, 2*norb, 2*norb, 2*norb), dtype=numpy.complex128)
            hamil = general_hamiltonian.General(tuple([h1e, h2e, h3e, h4e]))
            nbody_evol = wfn.apply_generated_unitary(time, 'taylor', hamil, expansion=20)
            result.ax_plus_y(-1.0, nbody_evol)
            self.assertTrue(result.norm() < 1.e-8)


#    @unittest.SkipTest
    def test_restricted(self):
        """
        """
        norb = 4
        nalpha = 2
        nbeta = 2
        nele = nalpha + nbeta
        time = 0.001
        lena = int(special.binom(norb, nalpha))
        lenb = int(special.binom(norb, nbeta))

        cidim = lena*lenb

        h1e = numpy.zeros((norb*2, norb*2), dtype=numpy.complex128)
        h2e = numpy.zeros((norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
        h3e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
        h4e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)

        for i in range(norb):
            for j in range(norb):
                h1e[i, j] += (i+j) * 0.02
                for k in range(norb):
                    for l in range(norb):
                        h2e[i, j, k, l] += (i+k)*(j+l)*0.02
                        for m in range(norb):
                            for n in range(norb):
                                h3e[i, j, k, l, m, n] += (i+l)*(j+m)*(k+n)*0.002
                                for o in range(norb):
                                    for p in range(norb):
                                        h4e[i, j, k, l, m, n, o, p] += (i+m)*(j+n)*(k+o)*(l+p)*0.001
            h1e[i, i] += i * 2.0

        h_wrap = tuple([h1e[:norb, :norb],
                        h2e[:norb, :norb, :norb, :norb],
                        h3e[:norb, :norb, :norb, :norb, :norb, :norb],
                        h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]])

        hamil = restricted_hamiltonian.Restricted(h_wrap)
        
        h1e[norb:, norb:] = h1e[:norb, :norb]

        h2e[norb:, norb:, norb:, norb:] = h2e[:norb, :norb, :norb, :norb]
        h2e[:norb, norb:, :norb, norb:] = h2e[:norb, :norb, :norb, :norb]
        h2e[norb:, :norb, norb:, :norb] = h2e[:norb, :norb, :norb, :norb]
        
        h3e[:norb, :norb, :norb, :norb, :norb, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, norb:, :norb, norb:, norb:, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, norb:, :norb, :norb, norb:, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, :norb, :norb, norb:, :norb, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, :norb, norb:, :norb, :norb, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, norb:, norb:, norb:, norb:, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, norb:, norb:, :norb, norb:, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, :norb, norb:, norb:, :norb, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]

        h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, :norb, :norb, norb:, norb:, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, :norb, :norb, :norb, norb:, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, :norb, :norb, norb:, :norb, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, norb:, :norb, :norb, :norb, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, norb:, :norb, norb:, norb:, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, norb:, :norb, :norb, norb:, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, norb:, :norb, norb:, :norb, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, :norb, norb:, :norb, :norb, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, :norb, norb:, norb:, norb:, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, :norb, norb:, :norb, norb:, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, :norb, norb:, norb:, :norb, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, norb:, norb:, :norb, :norb, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, norb:, norb:, norb:, norb:, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, norb:, norb:, :norb, norb:, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, norb:, norb:, norb:, :norb, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

        hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

        for i in range(lena):
            for j in range(lenb):
                ket = FqeData(nalpha, nbeta, norb)
                ket.coeff[i, j] = 1.0
                ket_h = ket.apply((h1e, h2e, h3e, h4e))
                hci[:, :, i, j] = ket_h.coeff[:, :]

        hci = numpy.reshape(hci, (cidim, cidim))
        fci_eig, fci_vec = linalg.eigh(hci)

        test = numpy.reshape(fci_vec[:, 0], (lena, lenb))

        wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
        wfn._civec[(nele, nalpha - nbeta)].coeff = test

        energy = wfn.expectationValue(hamil)

        self.assertAlmostEqual(energy, fci_eig[0])

        test_state = numpy.random.rand(cidim).astype(numpy.complex128)
 
        proj_coeff = fci_vec.conj().T @ test_state
        fci_rep = fci_vec @ proj_coeff
        err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
        self.assertTrue(err < 1.e-8)
 
        phase = numpy.multiply(numpy.exp(-1.j*time*fci_eig), proj_coeff)
 
        reference = fci_vec @ phase

        test_state = numpy.reshape(test_state, (lena, lenb))

        wfn = Wavefunction([[nele, nalpha-nbeta, norb]])
        wfn.set_wfn(strategy='from_data', raw_data={(nele, nalpha - nbeta): test_state})

        evol_wfn = wfn.time_evolve(time, hamil)

        computed = numpy.reshape(evol_wfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))

        err = numpy.abs(linalg.norm(reference - computed))
        self.assertTrue(err < 1.e-8)


#    @unittest.SkipTest
    def test_sso_four_body(self):
        """
        """
        norb = 4
        nalpha = 2
        nbeta = 2
        nele = nalpha + nbeta
        time = 0.001
        lena = int(special.binom(norb, nalpha))
        lenb = int(special.binom(norb, nbeta))

        cidim = lena*lenb

        h1e = numpy.zeros((norb*2, norb*2), dtype=numpy.complex128)
        h2e = numpy.zeros((norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
        h3e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
        h4e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)

        for i in range(norb):
            for j in range(norb):
                h1e[i, j] += (i+j) * 0.02
                for k in range(norb):
                    for l in range(norb):
                        h2e[i, j, k, l] += (i+k)*(j+l)*0.02
                        for m in range(norb):
                            for n in range(norb):
                                h3e[i, j, k, l, m, n] += (i+l)*(j+m)*(k+n)*0.002
                                for o in range(norb):
                                    for p in range(norb):
                                        h4e[i, j, k, l, m, n, o, p] += (i+m)*(j+n)*(k+o)*(l+p)*0.001
            h1e[i, i] += i * 2.0
        
        h1e[norb:, norb:] = 2.0*h1e[:norb, :norb]

        h2e[:norb, norb:, :norb, norb:] = 2.0*h2e[:norb, :norb, :norb, :norb]
        h2e[norb:, :norb, norb:, :norb] = 2.0*h2e[:norb, :norb, :norb, :norb]

        h2e[norb:, norb:, norb:, norb:] = 4.0*h2e[:norb, :norb, :norb, :norb]

        h3e[:norb, norb:, :norb, :norb, norb:, :norb] = 2.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, :norb, :norb, norb:, :norb, :norb] = 2.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, :norb, norb:, :norb, :norb, norb:] = 2.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        
        h3e[norb:, norb:, :norb, norb:, norb:, :norb] = 4.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, norb:, norb:, :norb, norb:, norb:] = 4.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, :norb, norb:, norb:, :norb, norb:] = 4.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]

        h3e[norb:, norb:, norb:, norb:, norb:, norb:] = 6.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]

        h4e[:norb, norb:, :norb, :norb, :norb, norb:, :norb, :norb] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, :norb, :norb, norb:, :norb, :norb, :norb] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, norb:, :norb, :norb, :norb, norb:, :norb] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, :norb, norb:, :norb, :norb, :norb, norb:] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        
        h4e[norb:, norb:, :norb, :norb, norb:, norb:, :norb, :norb] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, norb:, :norb, :norb, norb:, norb:, :norb] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, norb:, :norb, norb:, :norb, norb:, :norb] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, :norb, norb:, :norb, norb:, :norb, norb:] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, :norb, norb:, norb:, :norb, :norb, norb:] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, norb:, norb:, :norb, :norb, norb:, norb:] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

        h4e[norb:, norb:, norb:, :norb, norb:, norb:, norb:, :norb] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, :norb, norb:, norb:, norb:, :norb, norb:] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, norb:, norb:, :norb, norb:, norb:, norb:] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, norb:, norb:, norb:, :norb, norb:, norb:] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

        h4e[norb:, norb:, norb:, norb:, norb:, norb:, norb:, norb:] = 8.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

        h_wrap = tuple([h1e, h2e, h3e, h4e])
        hamil = sso_hamiltonian.SSOHamiltonian(h_wrap)

        hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

        for i in range(lena):
            for j in range(lenb):
                ket = FqeData(nalpha, nbeta, norb)
                ket.coeff[i, j] = 1.0
                ket_h = ket.apply(h_wrap)
                hci[:, :, i, j] = ket_h.coeff[:, :]

        hci = numpy.reshape(hci, (cidim, cidim))
        fci_eig, fci_vec = linalg.eigh(hci)

        test = numpy.reshape(fci_vec[:, 0], (lena, lenb))

        wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
        wfn._civec[(nele, nalpha - nbeta)].coeff = test

        energy = wfn.expectationValue(hamil)

        self.assertAlmostEqual(energy, fci_eig[0])

        test_state = numpy.random.rand(cidim).astype(numpy.complex128)
 
        proj_coeff = fci_vec.conj().T @ test_state
        fci_rep = fci_vec @ proj_coeff
        err = numpy.abs(numpy.linalg.norm(fci_rep - test_state))
        self.assertTrue(err < 1.e-8)
 
        phase = numpy.multiply(numpy.exp(-1.j*time*fci_eig), proj_coeff)
        reference = fci_vec @ phase

        test_state = numpy.reshape(test_state, (lena, lenb))

        wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
        wfn.set_wfn(strategy='from_data', raw_data={(nele, nalpha - nbeta): test_state})

        evol_wfn = wfn.time_evolve(time, hamil)

        computed = numpy.reshape(evol_wfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))

        err = numpy.abs(linalg.norm(reference - computed))
        self.assertTrue(err < 1.e-8)


#    @unittest.SkipTest
    def test_chebyshev(self):
        """
        """
        norb = 2
        nalpha = 1
        nbeta = 1
        nele = nalpha + nbeta
        time = 0.001
        lena = int(special.binom(norb, nalpha))
        lenb = int(special.binom(norb, nbeta))

        cidim = lena*lenb

        h1e = numpy.zeros((norb*2, norb*2), dtype=numpy.complex128)

        for i in range(2*norb):
            for j in range(2*norb):
                h1e[i, j] += (i+j) * 0.02
            h1e[i, i] += i * 2.0

        h_wrap = tuple([h1e])
        hamil = general_hamiltonian.General(h_wrap)

        hci = numpy.zeros((lena, lenb, lena, lenb), dtype=numpy.complex128)

        for i in range(lena):
            for j in range(lenb):
                ket = FqeData(nalpha, nbeta, norb)
                ket.coeff[i, j] = 1.0
                ket_h = ket.apply(h_wrap)
                hci[:, :, i, j] = ket_h.coeff[:, :]

        hci = numpy.reshape(hci, (cidim, cidim))
        fci_eig, fci_vec = linalg.eigh(hci)
        spec_lim = [fci_eig[0], fci_eig[-1]]

        test = numpy.reshape(fci_vec[:, 0], (lena, lenb))

        wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
        wfn._civec[(nele, nalpha - nbeta)].coeff = test

        evol_wfn = wfn.apply_generated_unitary(time, 'chebyshev', hamil, spec_lim=spec_lim)
        chebyshev = numpy.reshape(evol_wfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))

        evol_wfn = wfn.apply_generated_unitary(time, 'taylor', hamil)
        taylor = numpy.reshape(evol_wfn._civec[(nele, nalpha-nbeta)].coeff, (cidim))

        err = numpy.abs(linalg.norm(taylor - chebyshev))
        self.assertTrue(err < 1.e-8)


#    @unittest.SkipTest
    def test_few_nbody_evolution(self):
        """Check the evolution for very sparse hamiltonians
        """
        norb = 4
        nalpha = 3
        nbeta = 0
        nele = nalpha + nbeta
        m_s = nalpha - nbeta
        time = 0.01
        
        ops = FermionOperator('1^ 2^ 0 5', 2.0 - 2.j) + FermionOperator('5^ 0^ 2 1',  2.0 + 2.j)
        ops2 = FermionOperator('3^ 2^ 0 5', 2.0 - 2.j) + FermionOperator('5^ 0^ 2 3',  2.0 + 2.j)
        sham = sparse_hamiltonian.SparseHamiltonian(norb, ops + ops2, False)
        
        wfn = fqe.get_spin_nonconserving_wavefunction(nele, norb)
        wfn.set_wfn(strategy='random')
        wfn.normalize()
        
        result = wfn.apply_generated_unitary(time, 'taylor', sham)
        
        h2e = hamiltonian_utils.nbody_matrix(ops, norb)
        h2e += hamiltonian_utils.nbody_matrix(ops2, norb)
        h1e = numpy.zeros((norb*2, norb*2), dtype = h2e.dtype)
        hamiltest = general_hamiltonian.General(tuple([h1e, h2e]))
        nbody_evol = wfn.apply_generated_unitary(time, 'taylor', hamiltest)
        
        result.ax_plus_y(-1.0, nbody_evol)
        self.assertTrue(result.norm() < 1.e-8)

        


if __name__ == '__main__':
    unittest.main()
