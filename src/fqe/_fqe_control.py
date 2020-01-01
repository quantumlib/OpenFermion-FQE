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

from typing import Any, Dict, List, Tuple, Type, TYPE_CHECKING, Union

from openfermion import PolynomialTensor
from openfermion.transforms import jordan_wigner

import importlib.abc
import cirq
import numpy
from fqe.util import qubit_particle_number_index_spin
from fqe.cirq_utils import qubit_wavefunction_from_vacuum
from fqe import transform
from fqe.wavefunction import Wavefunction
from fqe.openfermion_utils import split_openfermion_tensor
from fqe.openfermion_utils import generate_one_particle_matrix
from fqe.openfermion_utils import generate_two_particle_matrix
from fqe.openfermion_utils import fqe_to_fermion_operator
from fqe.hamiltonians import general_hamiltonian

if TYPE_CHECKING:
    from openfermion import FermionOperator
    from fqe.hamiltonians import hamiltonian


def apply_generated_unitary(ops: 'FermionOperator',
                            wfn: 'Wavefunction',
                            algo: str,
                            accuracy: float = 1.e-7) -> 'Wavefunction':
    """APply the algebraic operators to the wavefunction with a specfiic
    algorithm and to the requested accuracy.

    Args:
        ops (FermionOperator) - a hermetian operator to apply to the
            wavefunction
        wfn (fqe.wavefunction) - the wavefunction to evolve
        algo (string) - a string dictating the method to use
        accuracy (double) - a desired accuracy to evolve the system to

    Retuns:
        wfn (fqe.wavefunction) - the evolved wavefunction
    """
    return wfn.apply_generated_unitary(ops, algo, accuracy)


def get_spin_nonconserving_wavefunction(nele: int, norb: int) -> 'Wavefunction':
    """Build a wavefunction with definite particle number and spin.

    Args:
        nele (int) - the number of electrons in the system
        norb (int) - the number of orbitals

    Returns:
        (wavefunction.Wavefunction) - a wavefunction object meeting the
            criteria laid out in the calling argument
    """
    
    param = []
    maxb = min(norb, nele)
    minb = nele - maxb
    for nbeta in range(minb, maxb+1):
        m_s = nele - nbeta*2
        param.append([nele, m_s, norb])
    return Wavefunction(param, broken=['spin'])


def get_wavefunction(nele: int, m_s: int, norb: int) -> 'Wavefunction':
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


def get_wavefunction_multiple(param: List[List[int]]) -> List['Wavefunction']:
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


def to_cirq(wfn: 'Wavefunction') -> numpy.ndarray:
    """Interoperability between cirq and the openfermion-fqe.  This takes an
    FQE wavefunction and returns a cirq compatible wavefunction based on the
    information stored within.

    Args:
        wfn (wavefunction.Wavefunction) - a openfermion-fqe wavefunction object

    Returns:
        numpy.array(dtype=numpy.complex128) - a cirq wavefunction that can be
            used in a simulator object.
    """
    nqubit = wfn.norb()*2
    ops = jordan_wigner(fqe_to_fermion_operator(wfn))
    qid = cirq.LineQubit.range(nqubit)
    return qubit_wavefunction_from_vacuum(ops, qid)


def from_cirq(state: numpy.ndarray, thresh: float) -> 'Wavefunction':
    """Interoperability between cirq and the openfermion-fqe.  This takes a
    cirq wavefunction and creates an FQE wavefunction object initialized with
    the correct data.

    Args:
        state (numpy.array(dtype=numpy.complex128)) - a cirq wavefunction
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


def apply(ops: 'FermionOperator', wfn: 'Wavefunction') -> 'Wavefunction':
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


def dot(wfn1: 'Wavefunction', wfn2: 'Wavefunction') -> complex:
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
    brakeys = wfn1.sectors()
    ketkeys = wfn2.sectors()
    keylist = [config for config in brakeys if config in ketkeys]
    ipval = .0 + .0j
    if not keylist:
        return ipval
    for config in keylist:
        ipval += numpy.dot(wfn1.get_coeff(config).flatten(),
                           wfn2.get_coeff(config).flatten())
    return ipval


def vdot(wfn1: 'Wavefunction', wfn2: 'Wavefunction') -> complex:
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
    brakeys = wfn1.sectors()
    ketkeys = wfn2.sectors()
    keylist = [config for config in brakeys if config in ketkeys]
    ipval = .0 + .0j
    if not keylist:
        return ipval
    for config in keylist:
        ipval += numpy.vdot(wfn1.get_coeff(config).flatten(),
                            wfn2.get_coeff(config).flatten())
    return ipval


def get_two_body_hamiltonian(pot: Union[complex, float],
                             h1e: numpy.ndarray,
                             g2e: numpy.ndarray,
                             chem: float,
                             symmh: List[Any],
                             symmg: List[Any]) -> 'general_hamiltonian.General':
    """Interface from the fqe to generate a two body hamiltonian.

        Args:
            pot (complex) - a complex scalar
            h1e (numpy.array(dim=2, dtype=complex128)) - matrix elements for
                single particle states
            g2e (numpy.array(dim=4, dtype=complex128)) - matrix elements for
                two particle states
            chem (double) - a value for the chemical potential
            symmh (list[list[int], double, bool]) - symmetry permutations for
                the one body matrix elements
            symmg (list[list[int], double, bool]) - symmetry permutations for
                the two body matrix elements

    Returns:
        General - an fqe General Hamiltonian Object
    """
    return general_hamiltonian.General(pot, h1e, g2e, chem, symmh, symmg)


def get_hamiltonian_from_ops(ops: 'FermionOperator') -> 'hamiltonian.Hamiltonian':
#                             pot: Union[complex, float],
#                             chem: float) -> 'general_hamiltonian.General':
    """Given a string of OpenFermion operators, generate a Hamiltonian for the
    FQE.

    Args:
        ops (FermionOpertor) - a string of OpenFermion operators
        pot (complex) - a constant potential to add
        chem (double) - a value for a chemical poential

    Returns:
        (fqe.hamiltonian.general_hamiltonian) - no symmetry
    """
    quadratic_hamiltonian = False
    quartic_hamiltonian = False

    a_spin_conserve = False
    b_spin_conserve = False
    diagonal_coulomb = False

    a_particle_conserve = False
    b_particle_conserve = False

    split = split_openfermion_tensor(ops)

    

    h1a, h1b, h1b_conj = generate_one_particle_matrix(split[2])

    dimension = h1a.shape[0]
    block_index = dimension // 2
    g2e = generate_two_particle_matrix(split[4])

    if (h1a.any() or h1b.any()) and g2e.any():
        symmh = [[[1, 2], 1.0, False]]
        symmg = [[[1, 2, 3, 4], 1.0, False]]
        return general_hamiltonian.General(pot, h1a, h1b, g2e, chem, symmh, symmg)

    if h1b.any() or h1b_conj.any():
        quadratic_hamiltonian = True
        if not numpy.allclose(h1b, h1b_conj.conj()):
            diff = abs(h1b - h1b_conj.conj())
            i, j = numpy.unravel_index(diff.argmax(), diff.shape)
            print('Elements {} {} outside tolerance'.format(i, j))
            print('{} != {} '.format(h1b[i, j], h1b_conj[i, j].conj()))
            raise ValueError

        if h1b[:block_index, :block_index].any():
            b_spin_conserve = False
        elif h1b[block_index:, block_index:].any():
            b_spin_conserve = False
        else:
            b_spin_conserve = True

    else:
        b_particle_conserve = True

    if h1a.any():
        quadratic_hamiltonian = True
        if h1a[block_index:, :block_index].any():
            a_spin_conserve = False

        elif h1a[:block_index, block_index:].any():
            a_spin_conserve = False

        else:
            a_spin_conserve = True

    if g2e.any():
        quartic_hamiltonian = False
        for index in range(dimension):
            for jndex in range(dimension):
                for kndex in range(dimension):
                    for lndex in range(dimension):
                        if index == kndex and jndex == lndex:
                            continue
                        if g[index, jndex, kndex, lndex]:
                            diagonal_coulomb = False
                            break

        if g2e == numpy.tranpose(g2e, axes=[2, 3, 0, 1]):
            diagonal_coulomb = True

    if quartic_hamiltonian and quadratic_hamiltonian:
        symmh = [[[1, 2], 1.0, False]]
        symmg = [[[1, 2, 3, 4], 1.0, False]]
        return general_hamiltonian.General(pot, h1a, h1b, h1b_conj, g2e, chem, symmh, symmg)

    if quartic_hamiltonian:
        if diagonal_coulomb:
            return diagonal_coulomb_hamiltonian.DiagonalCoulomb()
        return two_body_hamiltonian.TwoBody()

    if quadratic_hamiltonian:

        spin_conserve = a_spin_conserve and b_spin_conserve

        if spin_conserve and particle_conserve:
            return restrcted_hamiltonian.Restricted()
        if spin_conserve and particle_conserve:
            return gso_hamiltonian.Gso()
        if spin_conserve and particle_conserve:
            return bcs_hamiltonian.Bcs()

    symmh = [[[1, 2], 1.0, False]]
    symmg = [[[1, 2, 3, 4], 1.0, False]]
    return general_hamiltonian.General(pot, h1a, h1b, h1b_conj, g2e, chem, symmh, symmg)
