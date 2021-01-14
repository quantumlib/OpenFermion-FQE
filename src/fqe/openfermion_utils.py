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
"""Utilities which specifically require import from OpernFermion
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from openfermion import (FermionOperator, up_index, down_index, QubitOperator,
                         MolecularData, SymbolicOperator)
from openfermion.transforms import jordan_wigner, reverse_jordan_wigner

import numpy

from fqe.bitstring import gbit_index, integer_index, count_bits
from fqe.bitstring import lexicographic_bitstring_generator
from fqe.util import alpha_beta_electrons, bubblesort
from fqe.util import init_bitstring_groundstate, paritysort_int
from fqe.hamiltonians.hamiltonian import Hamiltonian
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


def ascending_index_order(ops: 'FermionOperator',
                          coeff: complex,
                          order: str = '') -> 'FermionOperator':
    """Permute a product of creation FermionOperators so that the index goes
    from the lowest to the highest inside of each spin sector and multiply by
    the correct parity.

    Args:
        ops (FermionOperator) - A product of FermionOperators

        coeff (complex64) - The coefficient in front of the operator strings

        order (string) - specify if the operators should be returned in \
            numerical order or not

    Returns:
        (FermionOperator) - A product of FermionOperators with the correct \
            sign
    """

    ilist = []

    if order == 'inorder':
        for term in ops:
            ilist.append(term[0])
        nperm = bubblesort(ilist)
        ops = FermionOperator('', 1.0)
        for i in ilist:
            ops *= FermionOperator(str(i) + '^', 1.0)
    else:
        for term in ops:
            ilist.append(term[0])
        nperm, _ = paritysort_int(ilist)
        alpl = []
        betl = []
        for i in ilist:
            if i % 2 == 0:
                alpl.append(i)
            else:
                betl.append(i)
        nperm += bubblesort(alpl)
        nperm += bubblesort(betl)
        ops = FermionOperator('', 1.0)
        for i in alpl:
            ops *= FermionOperator(str(i) + '^', 1.0)
        for i in betl:
            ops *= FermionOperator(str(i) + '^', 1.0)
    return coeff * ops * (-1.0 + 0.0j)**nperm


def bit_to_fermion_creation(string: int, spin: Optional[str] = None
                           ) -> Union[None, 'FermionOperator']:
    """Convert an occupation bitstring representation for a single spin case
    into openfermion operators

    Args:
        string (bitstring) - a string occupation representation

        spin (string) - a flag to indicate if the indexes of the occupation \
            should be converted to alpha, beta or spinless

    Returns:
        ops (Openfermion operators) - a FermionOperator string
    """
    ops = None
    if string != 0:

        def _index_parser(spin):
            if spin is None:

                def _spinless_index(i):
                    return i

                return _spinless_index

            if spin.lower() == 'a' or spin.lower() == 'up':
                return up_index
            if spin.lower() == 'b' or spin.lower() == 'down':
                return down_index
            raise ValueError("Unidentified spin option passed to"
                             "bit_to_fermion_creation")

        spin_index = _index_parser(spin)
        gbit = gbit_index(string)
        ops = FermionOperator(str(spin_index(next(gbit))) + '^', 1. + .0j)
        for i in gbit:
            ops *= FermionOperator(str(spin_index(i)) + '^', 1. + .0j)
    return ops


def fqe_to_fermion_operator(wfn: 'Wavefunction') -> 'FermionOperator':
    """Given an FQE wavefunction, convert it into strings of openfermion
    operators with appropriate coefficients for weights.

    Args:
        wfn (fqe.wavefunction)

    Returns:
        ops (FermionOperator)
    """
    ops = FermionOperator()
    for key in wfn.sectors():
        sector = wfn.sector(key)
        for alpha, beta, coeffcient in sector.generator():
            ops += coeffcient * determinant_to_ops(alpha, beta)
            if numpy.isclose(coeffcient, 0):
                continue
    return ops


def convert_qubit_wfn_to_fqe_syntax(ops: 'QubitOperator') -> 'FermionOperator':
    """This takes a qubit wavefunction in the form of a string of qubit
    operators and returns a string of FermionOperators with the proper
    formatting for easy digestion by FQE

    Args:
        ops (QubitOperator) - a string of qubit operators

    Returns:
        out (FermionOperator) - a string of fermion operators
    """
    ferm_str = reverse_jordan_wigner(ops)
    out = FermionOperator()
    for term in ferm_str.terms:
        out += ascending_index_order(term, ferm_str.terms[term])
    return out


def determinant_to_ops(a_str: int, b_str: int,
                       inorder: bool = False) -> 'FermionOperator':
    """Given the alpha and beta occupation strings return, a fermion operator
    which would produce that state when acting on the vacuum.

    Args:
        a_str (bitstring) - spatial orbital occupation by the alpha electrons

        b_str (bitstring) - spatial orbital occupation by the beta electrons

        inorder (bool) - flag to control creation and annihilation order

    Returns:
        (FermionOperator)
    """
    if a_str + b_str == 0:
        return FermionOperator('', 1.0)

    if inorder:
        occ = 0
        # Get occupation list from beta. Convert to spin down.
        for i in integer_index(b_str):
            occ += 2**down_index(i)
        for i in integer_index(a_str):
            occ += 2**up_index(i)
        # TODO: Consider forcing bit_to_fermion_creation to always return a
        #  FermionOperator instead of Optional[FermionOperator].
        return bit_to_fermion_creation(occ)  # type: ignore

    a_up: FermionOperator = bit_to_fermion_creation(a_str, 'up')  # type: ignore
    b_down: FermionOperator = bit_to_fermion_creation(b_str,
                                                      'down')  # type: ignore

    if a_str == 0:
        return b_down

    if b_str == 0:
        return a_up

    return a_up * b_down


def fci_fermion_operator_representation(norb: int, nele: int,
                                        m_s: int) -> 'FermionOperator':
    """Generate the Full Configuration interaction wavefunction in the
    openfermion FermionOperator representation with coefficients of 1.0.

    Args:
        norb (int) - number of spatial orbitals

        nele (int) - number of electrons

        m_s (int) - s_z spin quantum number

    Returns:
        FermionOperator
    """
    nalpha, nbeta = alpha_beta_electrons(nele, m_s)
    gsstr = init_bitstring_groundstate(nalpha)
    alphadets = lexicographic_bitstring_generator(gsstr, norb)
    gsstr = init_bitstring_groundstate(nbeta)
    betadets = lexicographic_bitstring_generator(gsstr, norb)
    ops = FermionOperator('', 1.0)
    for bstr in betadets:
        for astr in alphadets:
            ops += determinant_to_ops(astr, bstr)
    return ops - FermionOperator('', 1.0)


def fci_qubit_representation(norb: int, nele: int, m_s: int) -> 'QubitOperator':
    """Create the qubit representation of Full CI according to the parameters
    passed

    Args:
        norb (int) - number of spatial orbitals

        nele (int) - number of electrons

        m_s (int) - spin projection onto sz

    Returns:
        ops (QubitOperator)
    """
    return jordan_wigner(fci_fermion_operator_representation(norb, nele, m_s))


def fermion_operator_to_bitstring(term: 'FermionOperator') -> Tuple[int, int]:
    """Convert an openfermion FermionOperator object into a bitstring
    representation.

    Args:
        cluster (FermionOperator) - A product of FermionOperators with a \
            coefficient

    Returns:
        (bitstring, bitstring) - a pair of bitstrings representing the \
            alpha and beta occupation in the orbitals.
    """
    upstring = 0
    downstring = 0
    for ops in term:
        if ops[0] % 2 == 0:
            upstring += 2**(ops[0] // 2)
        else:
            downstring += 2**((ops[0] - 1) // 2)
    return upstring, downstring


def fermion_opstring_to_bitstring(ops) -> List[List[Any]]:
    """Convert an openfermion FermionOperator object into the corresponding
    bitstring representation with the appropriate coefficient.

    Args:
        ops (FermionOperator) - A product of FermionOperators with a coefficient

    Returns:
        list(list(int, int, coeff)) - a pair of integers representing a bitstring \
            of alpha and beta occupation in the orbitals.
    """
    raw_data = []
    for term in ops.terms:
        rval = fermion_operator_to_bitstring(term)
        raw_data.append([rval[0], rval[1], ops.terms[term]])
    return raw_data


def largest_operator_index(ops: 'FermionOperator') -> Tuple[int, int]:
    """Search through a FermionOperator string and return the largest even
    value and the largest odd value to find the total dimension of the matrix
    and the size of the spin blocks

    Args:
        ops (OpenFermion operators) - a string of operators comprising a \
            Hamiltonian

    Returns:
        int, int - the largest even integer and the largest odd integer of the \
            indexes passed to the subroutine.
    """
    maxodd = -1
    maxeven = -1
    for term in ops.terms:
        for ele in term:
            if ele[0] % 2:
                maxodd = max(ele[0], maxodd)
            else:
                maxeven = max(ele[0], maxeven)

    return maxeven, maxodd


def ladder_op(ind: int, step: int):
    """Local function to generate ladder operators in Qubit form given an index
    and action.

    Args:
        ind (int) - index of the orbital occupation to create

        step (int) - an integer indicating whether you should create or \
        annhilate the orbital
    """
    xstr = 'X' + str(ind)
    ystr = 'Y' + str(ind)
    op_ = QubitOperator(xstr, 1. +.0j) + \
            ((-1)**step)*QubitOperator(ystr, 0. + 1.j)
    zstr = QubitOperator('', 1. + .0j)
    for i in range(ind):
        zstr *= QubitOperator('Z' + str(i), 1. + .0j)
    return zstr * op_ / 2.0


def mutate_config(stra: int, strb: int,
                  term: Tuple[Tuple[int, int]]) -> Tuple[int, int, int]:
    """Apply creation and annihilation operators to a configuration in the
    bitstring representation and return the new configuration and the parity.

    Args:
        stra (bitstring) - the alpha string to be manipulated

        stra (bitstring) - the beta string to be manipulated

        term (Operators) - the operations to apply

    Returns:
        tuple(bitstring, bitstring, int) - the mutated alpha and beta \
            configuration and the parity to go with it.
    """
    newa = stra
    newb = strb
    parity = 1

    for ops in reversed(term):
        if ops[0] % 2:
            abits = count_bits(newa)
            indx = (ops[0] - 1) // 2
            if ops[1] == 1:
                if newb & 2**indx:
                    #                    return -1, -1, 0
                    return stra, strb, 0

                newb += 2**indx
            else:
                if not newb & 2**indx:
                    #                    return -1, -1, 0
                    return stra, strb, 0

                newb -= 2**indx

            lowbit = (1 << (indx)) - 1
            parity *= (-1)**(count_bits(lowbit & newb) + abits)

        else:
            indx = ops[0] // 2
            if ops[1] == 1:
                if newa & 2**indx:
                    #                    return -1, -1, 0
                    return stra, strb, 0

                newa += 2**indx
            else:
                if not newa & 2**indx:
                    #                    return -1, -1, 0
                    return stra, strb, 0

                newa -= 2**indx

            lowbit = (1 << (indx)) - 1
            parity *= (-1)**count_bits(lowbit & newa)

    return newa, newb, parity


def split_openfermion_tensor(ops: 'FermionOperator'
                            ) -> Dict[int, 'FermionOperator']:
    """Given a string of openfermion operators, split them according to their
    rank.

    Args:
        ops (FermionOperator) - a string of Fermion Operators

    Returns:
        split dict[int] = FermionOperator - a list of Fermion Operators sorted \
            according to their degree
    """
    split: Dict[int, 'FermionOperator'] = {}
    for term in ops:
        degree = term.many_body_order()
        if degree not in split:
            split[degree] = term
        else:
            split[degree] += term

    return split


def update_operator_coeff(operators: 'SymbolicOperator',
                          coeff: numpy.ndarray) -> None:
    """Given a string of Symbolic operators, set each prefactor equal to the
    value in coeff.  Note that the order in which SymbolicOperators in
    OpenFermion are printed out is not necessarily the order in which they are
    iterated over.

    Args:
        operators (FermionOperator) - A linear combination of operators.

        coeff (numpy.array(ndim=1,dtype=numpy.complex64)

    Returns:
        nothing - mutates coeff in place
    """
    for ops, val in zip(operators.terms, coeff):
        operators.terms[ops] = 1.0 * val


def molecular_data_to_restricted_fqe_op(molecule: MolecularData
                                       ) -> RestrictedHamiltonian:
    """
    Convert an OpenFermion MolecularData object to a FQE Hamiltonian

    FQE Hamiltonians are provide spatial orbitals

    Args:
        molecule: MolecularData object to convert.
    """
    return integrals_to_fqe_restricted(molecule.one_body_integrals,
                                       molecule.two_body_integrals)


def integrals_to_fqe_restricted(h1e, h2e) -> RestrictedHamiltonian:
    """
    Convert integrals in physics ordering to a RestrictedHamiltonian

    Args:
        h1e: one-electron spin-free integrals
        h2e: two-electron spin-free integrals <12|21>
    """
    return RestrictedHamiltonian((h1e, numpy.einsum('ijlk', -0.5 * h2e)))
