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

""" FQE control is a wrapper to allow for convenient or more readable access
to the emulator.
"""

from openfermion import FermionOperator, PolynomialTensor
from openfermion.transforms import jordan_wigner
from openfermion.transforms import get_quadratic_hamiltonian as quad_of
from openfermion.ops import QuadraticHamiltonian, DiagonalCoulombHamiltonian

import cirq
import numpy
from fqe.util import sort_configuration_keys
from fqe.util import qubit_particle_number_index_spin
from fqe.util import rand_wfn
from fqe.util import qubit_particle_number_sector
from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe import transform
from fqe.wavefunction import Wavefunction
from fqe.openfermion_utils import split_openfermion_tensor
from fqe.openfermion_utils import generate_one_particle_matrix
from fqe.openfermion_utils import generate_two_particle_matrix
from fqe.openfermion_utils import fqe_to_fermion_operator
from fqe.hamiltonians import general_hamiltonian
from fqe.hamiltonians import quadratic_hamiltonian


def apply_generated_unitary(ops, wfn, algo, accuracy=1.e-7):
    """APply the algebraic operators to the wavefunction with a specfiic
    algorithm and to the requested accuracy.

    Args:
        ops (FermionOperators) - a hermetian operator to apply to the
            wavefunction
        wfn (fqe.wavefunction) - the wavefunction to evolve
        algo (string) - a string dictating the method to use
        accuracy (double) - a desired accuracy to evolve the system to

    Retuns:
        wfn (fqe.wavefunction) - the evolved wavefunction
    """
    return wfn.apply_generated_unitary(ops, algo, accuracy)


def get_spin_nonconserving_wavefunction(nele):
    """Build a wavefunction with definite particle number and spin.

    Args:
        nele (int) - the number of electrons in the system

    Returns:
        (wavefunction.Wavefunction) - a wavefunction object meeting the
            criteria laid out in the calling argument
    """
    norb = 2*nele
    if nele % 2:
        m_s = 1
    else:
        m_s = 0
    return Wavefunction(param=[[nele, m_s, norb]])


def get_wavefunction(nele, m_s, norb):
    """Build a wavefunction with definite particle number and spin.

    Args:
        nele (int) - the number of electrons in the system
        m_s (int) - the s_z spin projection of the system
        norb (int) - the number of spatial orbtials to used

    Returns:
        (wavefunction.Wavefunction) - a wavefunction object meeting the
            criteria laid out in the calling argument
    """
    arg = [[nele, m_s, norb]]
    return Wavefunction(param=arg)


def get_wavefunction_multiple(param):
    """Generate many different wavefunctions.

    Args:
        param (list[list[nele, m_s, norb]]) - a list of parameters used to
            initialize wavefunctions.  The arguments in the parameters are

                nele (int) - the number of electrons in the system
                m_s (int) - the s_z spin projection of the system
                norb (int) - the number of spatial orbtials to used

    Returns:
        list[(wavefunction.Wavefunction)] - a list of wavefunction objects
    """
    state = []
    for val in param:
        state.append(Wavefunction(param=[val]))
    return state


def to_cirq(wfn):
    """Interoperability between cirq and the openfermion-fqe.  This takes an
    FQE wavefunction and returns a cirq compatible wavefunction based on the
    information stored within.

    Args:
        wfn (wavefunction.Wavefunction) - a openfermion-fqe wavefunction object

    Returns:
        numpy.array(dtype=numpy.complex64) - a cirq wavefunction that can be
            used in a simulator object.
    """
    nqubit = wfn.norb*2
    ops = jordan_wigner(fqe_to_fermion_operator(wfn))
    qid = cirq.LineQubit.range(nqubit)
    return qubit_wavefunction_from_vacuum(ops, qid)


def from_cirq(state, thresh):
    """Interoperability between cirq and the openfermion-fqe.  This takes a
    cirq wavefunction and creates an FQE wavefunction object initialized with
    the correct data.

    Args:
        state (numpy.array(dtype=numpy.complex64)) - a cirq wavefunction
        thresh (double) - set the limit at which a cirq element should be
            considered zero and not make a contribution to the FQE wavefunction

    Returns:
        openfermion-fqe.Wavefunction
    """
    param = []
    nqubits = int(numpy.log2(state.size))
    norb = nqubits//2
    for pnum in range(nqubits + 1):
        occ = qubit_particle_number_index_spin(nqubits, pnum)
        for orb in occ:
            if numpy.absolute(state[orb[0]]) > thresh:
                param.append([pnum, orb[1], norb])
    wfn = Wavefunction(param)
    transform.from_cirq(wfn, state)
    return wfn


def apply(ops, wfn):
    """Create a new wavefunction by applying the fermionic operators to the
    wavefunction.

    Args:
        ops (FermionOperator) - a Fermion Operator string to apply to the
            wavefunction
        wfn (wavefunction.Wavefunction) - an FQE wavefunction to mutate

    Returns:
        openfermion-fqe.Wavefunction - a new wavefunction generated from the
            application of the fermion operators to the wavefunction
    """
    return wfn.apply(ops)


def dot(wfn1, wfn2):
    """Calculate the dot product of two wavefunctions.  Note that this does
    not use the conjugate.  See vdot for the similar conjugate functionality.
    
    Args:
        wfn1 (wavefunction.Wavefunction) - wavefunction corresponding to the
            row vector
        wfn2 (wavefunction.Wavefunction) - wavefunction corresponding to the
            coumn vector

    Returns:
        (complex) - scalar as result of the dot product
    """
    brakeys = wfn1.configs
    ketkeys = wfn2.configs
    keylist = [config for config in brakeys if config in ketkeys]
    ipval = .0 + .0j
    if len(keylist) == 0:
        return ipval
    for config in keylist:
        ipval += numpy.dot(wfn1.get_coeff(config, vec=[0]).T, 
                           wfn2.get_coeff(config, vec=[0]))
    return ipval


def vdot(wfn1, wfn2):
    """Calculate the inner product of two wavefunctions using conjugation on
    the elements of wfn1.
    
    Args:
        wfn1 (wavefunction.Wavefunction) - wavefunction corresponding to the
            conjugate row vector
        wfn2 (wavefunction.Wavefunction) - wavefunction corresponding to the
            coumn vector

    Returns:
        (complex) - scalar as result of the dot product
    """
    brakeys = wfn1.configs
    ketkeys = wfn2.configs
    keylist = [config for config in brakeys if config in ketkeys]
    ipval = .0 + .0j
    if len(keylist) == 0:
        return ipval
    for config in keylist:
        ipval += numpy.vdot(wfn1.get_coeff(config, vec=[0]),
                            wfn2.get_coeff(config, vec=[0]))
    return ipval


def get_quadratic_hamiltonian(ops, chem):
    split = split_openfermion_tensor(ops)
    h1e = generate_one_particle_matrix(split[2])
    symmh = [[[1, 2], 1.0, False]]
    return quadratic_hamiltonian.Quadratic(0.0, h1e, chem, symmh)


def get_two_body_hamiltonian(pot, h1e, g2e, chem, symmh, symmg):
    """
    """
    return general_hamiltonian.General(pot, h1e, g2e, chem, symmh, symmg)


def get_hamiltonian_from_openfermion(hamiltonian):
    """Wrapper to parse Openfermion Hamiltonians and put them into the FQE

    Args:
        hamiltonian (openfermion.Hamiltonian)

    Returns:
        FQE-Hamiltonian
    """
    hamiltonian_type = hamiltonian.__class__.__name__

    if hamiltonian_type == 'QuadraticHamiltonian':
        h1e = hamiltonian.n_body_tensors[(1, 0)]
        chem = hamiltonian.chemical_potential
        h1e += chem*numpy.identity(h1e.shape[0], dtype=numpy.complex64)
        symmh = [
            [[1, 2], 1.0, False],
            [[2, 1], 1.0, True]
            ]

        return quadratic_hamiltonian.Quadratic(0.0, h1e, chem, symmh)


def get_hamiltonian_from_ops(ops, pot, chem):
    """Given a string of OpenFermion operators, generate a Hamiltonian for the
    FQE.

    Args:
        ops (OpenFermion) - a string of OpenFermion operators
        pot (double) - a constant potential to add
        chem (double) - a value for a chemical poential

    Returns:
        (fqe.hamiltonian.general_hamiltonian) - no symmetry
    """
    split = split_openfermion_tensor(ops)
    h1e = generate_one_particle_matrix(split[2])
    g2e = generate_two_particle_matrix(split[4])
    symmh = [[[1, 2], 1.0, False]]
    symmg = [[[1, 2, 3, 4], 1.0, False]]
    return general_hamiltonian.General(pot, h1e, g2e, chem, symmh, symmg)


def hamiltonian_to_openfermion(hamiltonian):
    """Return a polynomial tensor for Openfermion by parsing the Hamiltonian
    elements into dict.
    """
    tensors = {}
    hamiltonian_type = hamiltonian.__class__.__name__

    if hamiltonian_type == 'General':
        tensors[(1, 0)] = hamiltonian.h
        tensors[(1, 1, 0, 0)] = hamiltonian.g

    if hamiltonian_type == 'Quadratic':
        tensors[(1, 0)] = hamiltonian.h

    return PolynomialTensor(tensors)
