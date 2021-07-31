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
"""Unit tests for openfermion_utils.
"""

import numpy
import pytest

from openfermion import FermionOperator, MolecularData
from openfermion.transforms import jordan_wigner

from fqe import openfermion_utils
from fqe import wavefunction
from tests.unittest_data.build_lih_data import build_lih_data


def test_ascending_index_order_inorder():
    """Transformations in openfermion return FermionOperators from highest
    index to the lowest index.  Occationally we would like to reverse this.
    """
    ops = FermionOperator('5^ 4^ 3^ 2^ 1^ 0^', 1.0)
    test = FermionOperator('0^ 1^ 2^ 3^ 4^ 5^', -1.0)
    for key in ops.terms:
        f_ops = openfermion_utils.ascending_index_order(key,
                                                        ops.terms[key],
                                                        order='inorder')
    assert f_ops == test

def test_ascending_index_order():
    """Transformations in openfermion return FermionOperators from highest
    index to the lowest index.  Occationally we would like to reverse this.
    """
    ops = FermionOperator('5^ 4^ 3^ 2^ 1^ 0^', 1.0)
    test = FermionOperator('0^ 2^ 4^ 1^ 3^ 5^', 1.0)
    for key in ops.terms:
        f_ops = openfermion_utils.ascending_index_order(key, ops.terms[key])
    assert f_ops == test

def test_bit_to_fermion_creation_alpha():
    """Spin index conversetion between the fqe and opernfermion is that
    alpha/up fermions are indicated by index*2. bit 1 is index 0 for up
    spin.
    """
    assert openfermion_utils.bit_to_fermion_creation(1, 'up') == \
                      FermionOperator('0^', 1.0)

def test_bit_to_fermion_creation_beta():
    """Spin index conversetion between the fqe and opernfermion is that
    beta/down fermions are indicated by index*2+1. bit 1 is index 1 for
    down spin.
    """
    assert openfermion_utils.bit_to_fermion_creation(1, 'down') == \
                      FermionOperator('1^', 1.0)

def test_bit_to_fermion_creation_spinless():
    """Spinless particles are converted directly into their occupation index.
    """
    assert openfermion_utils.bit_to_fermion_creation(1) == \
                      FermionOperator('0^', 1.0)

def test_bit_to_fermion_creation_null():
    """Empty bitstrings will return None.
    """
    assert openfermion_utils.bit_to_fermion_creation(0) is None

def test_bit_to_fermion_creation_error():
    """Bad options for indexes should raise an error
    """
    with pytest.raises(ValueError):
        openfermion_utils.bit_to_fermion_creation(1,
                      spin='around')

def test_bit_to_fermion_creation_order():
    """Return the representation of a bitstring as FermionOperators acting
    on the vacuum.
    """
    bit = (1 << 6) + 31
    assert openfermion_utils.bit_to_fermion_creation(bit) == \
                      FermionOperator('0^ 1^ 2^ 3^ 4^ 6^', 1.0)

def test_fqe_to_fermion_operator():
    """Convert the fqe representation to openfermion operators.

    Note: Due to the unique OpenFermion data structure and the conversion
    of the data internally, test requires an iterative stucture rather than
    assertDictEqual
    """

    def _calc_coeff(val):
        return round(val / numpy.sqrt(30.0), 7) + .0j

    coeff = [
        _calc_coeff(1.0),
        _calc_coeff(2.0),
        _calc_coeff(3.0),
        _calc_coeff(4.0)
    ]

    data = numpy.array([[coeff[0]], [coeff[1]], [coeff[2]], [coeff[3]]],
                        dtype=numpy.complex128)

    test = FermionOperator('0^ 1^', coeff[0])
    test += FermionOperator('0^ 3^', coeff[1])
    test += FermionOperator('2^ 1^', coeff[2])
    test += FermionOperator('2^ 3^', coeff[3])
    wfn = wavefunction.Wavefunction([[2, 0, 2]])
    data = numpy.reshape(data, (2, 2))
    passed_data = {(2, 0): data}
    wfn.set_wfn(strategy='from_data', raw_data=passed_data)
    ops = openfermion_utils.fqe_to_fermion_operator(wfn)
    assert list(ops.terms.keys()) == list(test.terms.keys())
    for term in ops.terms:
        assert round(abs(ops.terms[term]-test.terms[term]), 7) == 0

def test_fqe_to_fermion_operator_sparse():
    """Convert the fqe representation to openfermion operators.
    """
    def _calc_coeff(val):
        return round(val / numpy.sqrt(26.0), 7) + .0j

    coeff = [
        _calc_coeff(1.0),
        _calc_coeff(0.0),
        _calc_coeff(3.0),
        _calc_coeff(4.0)
    ]

    data = numpy.array([[coeff[0]], [coeff[1]], [coeff[2]], [coeff[3]]],
                        dtype=numpy.complex128)

    test = FermionOperator('0^ 1^', coeff[0])
    test += FermionOperator('2^ 1^', coeff[2])
    test += FermionOperator('2^ 3^', coeff[3])
    wfn = wavefunction.Wavefunction([[2, 0, 2]])
    data = numpy.reshape(data, (2, 2))
    passed_data = {(2, 0): data}
    wfn.set_wfn(strategy='from_data', raw_data=passed_data)
    ops = openfermion_utils.fqe_to_fermion_operator(wfn)
    assert list(ops.terms.keys()) == list(test.terms.keys())
    for term in ops.terms:
        assert round(abs(ops.terms[term]-test.terms[term]), 7) == 0

def test_convert_qubit_wfn_to_fqe_syntax():
    """Reexpress a qubit wavefunction in terms of Fermion Operators
    """
    coeff = [0.166595741722 + .0j, 0.986025283063 + .0j]
    test = FermionOperator('0^ 2^ 1^', coeff[0])
    test += FermionOperator('2^ 4^ 3^', -coeff[1])

    laddero = openfermion_utils.ladder_op

    q_ops = coeff[0] * laddero(0, 1) * laddero(2, 1) * laddero(1, 1)
    q_ops += coeff[1] * laddero(2, 1) * laddero(3, 1) * laddero(4, 1)
    ops = openfermion_utils.convert_qubit_wfn_to_fqe_syntax(q_ops)
    # Test Keys
    assert list(ops.terms.keys()) == list(test.terms.keys())
    # Test Values
    for i in ops.terms:
        assert round(abs(ops.terms[i]-test.terms[i]), 7) == 0

def test_determinant_to_ops_null():
    """A determinant with no occupation should return None type.
    """
    assert openfermion_utils.determinant_to_ops(0, 0) == \
                      FermionOperator('', 1.0)

def test_determinant_to_ops_alpha():
    """A singly occupied determinant should just have a single creation
    operator in it.
    """
    assert openfermion_utils.determinant_to_ops(1, 0) == \
                      FermionOperator('0^', 1.0)

def test_determinant_to_ops_beta():
    """A singly occupied determinant should just have a single creation
    operator in it.
    """
    assert openfermion_utils.determinant_to_ops(0, 1) == \
                      FermionOperator('1^', 1.0)

def test_determinant_to_ops():
    """Convert a determinant to openfermion ops
    """
    test_ops = FermionOperator('0^ 2^ 4^ 1^ 3^', 1.0)
    assert openfermion_utils.determinant_to_ops(7, 3) == test_ops

def test_determinant_to_ops_inorder():
    """If spin case is not necessary to distinguish in the representation
    return the orbitals based on increasing index.
    """
    test_ops = FermionOperator('0^ 1^ 2^ 3^ 4^', 1.0)
    assert openfermion_utils.determinant_to_ops(7, 3, inorder=True) == test_ops

def test_fci_fermion_operator_representation():
    """If we don't already have a wavefunction object to build from we can
    just specify the parameters and build the FermionOperator string.
    """
    test = FermionOperator('0^ 2^ 1^', 1.0) + \
        FermionOperator('0^ 4^ 1^', 1.0) + \
        FermionOperator('2^ 4^ 1^', 1.0) + \
        FermionOperator('0^ 2^ 3^', 1.0) + \
        FermionOperator('0^ 4^ 3^', 1.0) + \
        FermionOperator('2^ 4^ 3^', 1.0) + \
        FermionOperator('0^ 2^ 5^', 1.0) + \
        FermionOperator('0^ 4^ 5^', 1.0) + \
        FermionOperator('2^ 4^ 5^', 1.0)

    ops = openfermion_utils.fci_fermion_operator_representation(3, 3, 1)
    assert ops == test

def test_fci_qubit_representation():
    """If we don't already have a wavefunction object to build from we can
    just specify the parameters and build the FermionOperator string.
    """
    test = jordan_wigner(FermionOperator('0^ 2^ 1^', 1.0) + \
        FermionOperator('0^ 4^ 1^', 1.0) + \
        FermionOperator('2^ 4^ 1^', 1.0) + \
        FermionOperator('0^ 2^ 3^', 1.0) + \
        FermionOperator('0^ 4^ 3^', 1.0) + \
        FermionOperator('2^ 4^ 3^', 1.0) + \
        FermionOperator('0^ 2^ 5^', 1.0) + \
        FermionOperator('0^ 4^ 5^', 1.0) + \
        FermionOperator('2^ 4^ 5^', 1.0))

    ops = openfermion_utils.fci_qubit_representation(3, 3, 1)
    assert ops == test

def test_fermion_operator_to_bitstring():
    """Interoperability between OpenFermion and the FQE necessitates that
    we can convert between the representations.
    """
    ops = FermionOperator('0^ 1^ 2^ 5^ 8^ 9^ ', 1. + .0j)
    testalpha = 1 + 2 + 16
    testbeta = 1 + 4 + 16
    test = [testalpha, testbeta]
    for term in ops.terms:
        result = openfermion_utils.fermion_operator_to_bitstring(term)
        assert test == list(result)

def test_fermion_opstring_to_bitstring():
    """Interoperability between OpenFermion and the FQE necessitates that
    we can convert between the representations.
    """
    ops = FermionOperator('0^ 1^', 1. + .0j) + \
          FermionOperator('2^ 5^', 2. + .0j) + \
          FermionOperator('8^ 9^', 3. + .0j)
    test = [[1, 1, 1. + .0j], [2, 4, 2. + .0j], [16, 16, 3. + .0j]]
    result = openfermion_utils.fermion_opstring_to_bitstring(ops)
    for val in result:
        assert val in test

def test_largest_operator_index():
    """Find a largest operator index from the input operator.
    """
    ops = FermionOperator('0^ 1^', 1. + .0j) + \
          FermionOperator('2^ 5^', 2. + .0j) + \
          FermionOperator('8^ 9^', 3. + .0j)
    maxe, maxo = openfermion_utils.largest_operator_index(ops)
    assert maxe == 8 and maxo == 9 

def test_ladder_op():
    """Make sure that our qubit ladder operators are the same as the jw
    transformation
    """
    create = jordan_wigner(FermionOperator('10^', 1. + .0j))
    annihilate = jordan_wigner(FermionOperator('10', 1. + .0j))
    assert create == openfermion_utils.ladder_op(10, 1)
    assert annihilate == openfermion_utils.ladder_op(10, 0)

def test_update_operator_coeff():
    """Update the coefficients of an operator string
    """
    coeff = numpy.ones(6, dtype=numpy.complex64)
    test = FermionOperator('1^', 1. + .0j)
    ops = FermionOperator('1^', 1. + .0j)
    for i in range(2, 7):
        ops += FermionOperator(str(i) + '^', i * (1. + .0j))
        test += FermionOperator(str(i) + '^', 1. + .0j)

    openfermion_utils.update_operator_coeff(ops, coeff)
    assert ops == test

def test_split_openfermion_tensor():
    """Split a FermionOperator string into terms based on the length
    of the terms.
    """
    ops0 = FermionOperator('', 1.0)
    ops1 = FermionOperator('1^', 1.0)
    ops2 = FermionOperator('2 1', 1.0)
    ops3a = FermionOperator('2^ 1^ 7', 1.0)
    ops3b = FermionOperator('5^ 0 6', 1.0)
    ops3 = ops3a + ops3b
    total = ops0 + ops1 + ops2 + ops3a + ops3b
    split = openfermion_utils.split_openfermion_tensor(total)
    assert ops0 == split[0]
    assert ops1 == split[1]
    assert ops2 == split[2]
    assert ops3 == split[3]

def test_moleculardata_to_restricted_hamiltonian():
    """
    Convert an OpenFermion MolecularData object to a
    fqe.RestrictedHamiltonian
    """
    h1e, h2e, _ = build_lih_data('energy')
    # dummy geometry
    geometry = [['Li', [0, 0, 0], ['H', [0, 0, 1.4]]]]
    charge = 0
    multiplicity = 1
    molecule = MolecularData(geometry=geometry,
                              basis='sto-3g',
                              charge=charge,
                              multiplicity=multiplicity)
    molecule.one_body_integrals = h1e
    molecule.two_body_integrals = numpy.einsum('ijlk', -2 * h2e)
    molecule.nuclear_repulsion = 0

    restricted_ham = openfermion_utils.molecular_data_to_restricted_fqe_op(
        molecule=molecule)
    h1e_test, h2e_test = restricted_ham.tensors()
    assert numpy.allclose(h1e_test, h1e)
    assert numpy.allclose(h2e_test, h2e)
