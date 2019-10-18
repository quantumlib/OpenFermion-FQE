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

""" Fermionic Quantum Emulator data class for holding wavefunction data.
"""

from typing import Any, List, Generator

import numpy

from fqe.bitstring import integer_index
from fqe import fci_graph
from fqe.string_addressing import build_string_address
from fqe.util import rand_wfn, weyl_paldus, init_bitstring_groundstate


class FqeData(fci_graph.FciGraph):
    """The data structure of the CI space is useful for a single
    CI space with particle and spin quantum numbers.  Ultimately a combination
    of CI spaces will be necessary to perform calculations using the FQE.  The
    FQE Data structure compartmentalizes that structure.
    """


    def __init__(self, nalpha: int, nbeta: int, norb: int) -> None:
        """The FqeData structure holds the wavefunction for a particular
        configuration and provides an interace for accessing the data through
        the fcigraph functionality.

        Args:
            nalpha (int) - the number of alpha electrons
            nbeta (int) - the number of beta electrons
            norb (int) - the number of spatial orbitals

        Members:
            _nele (int) - total number of electrons
            _m_s (int) - value for s_z
            _ga_a (bitstring) - store the ground state of the alpha string
            _ga_b (bitstring) - store the ground state of the beta string
            _cidim (int) - store the dimension of the ci vector
            coeff (numpy.array) - the wavefunction
        """
        super(FqeData, self).__init__(nalpha, nbeta, norb)
        self._nele = nalpha + nbeta
        self._m_s = nalpha - nbeta
        self._gs_a = init_bitstring_groundstate(self._nalpha)
        self._gs_b = init_bitstring_groundstate(self._nbeta)
        self._cidim = weyl_paldus(self._nele, self._m_s, norb)
        self.coeff = numpy.zeros((self._lena*self._lenb, self._cidim),
                                 dtype=numpy.complex64)


    @property
    def conj(self) -> None:
        """Conjugate the coefficients
        """
        numpy.conjugate(self.coeff, self.coeff)


    @property
    def n_electrons(self) -> int:
        """Particle number getter
        """
        return self._nele


    @property
    def m_s(self) -> int:
        """Spin projection along the z axis
        """
        return self._m_s


    @property
    def nalpha(self) -> int:
        """Number of alpha electrons
        """
        return self._nalpha


    @property
    def nbeta(self) -> int:
        """Number of beta electrons
        """
        return self._nbeta


    @property
    def lena(self) -> int:
        """Length of the alpha configuration space
        """
        return self._lena


    @property
    def lenb(self) -> int:
        """Length of the beta configuration space
        """
        return self._lenb


    @property
    def ci_space_length(self) -> int:
        """Return the length ci space
        """
        return self._lena*self._lenb


    @property
    def ci_configuration_dimension(self) -> int:
        """Return the number of states possible given this configuration
        """
        return self._cidim


    def str_alpha(self, address: int) -> int:
        """Return the bitstring stored at the address for the alpha spin case
        """
        return self.get_alpha(address)


    def str_beta(self, address: int) -> int:
        """Return the bitstring stored at the address for the beta spin case
        """
        return self.get_beta(address)


    def _index_to_address(self, i_a: int, i_b: int) -> int:
        """The CI vector is accessed as a rectangular array C(i_a, i_b)
        but it is stored as C(len_alpha*len_beta) in row-major order consistent
        with C and the numpy default for reshaping.

        Args:
            i_a (int) - index into the CI vector
            i_b (int) - index into the CI vector
        """
        return i_a*self._lenb + i_b


    def cc_i(self, i_a: int, i_b: int, ivec: int) -> complex:
        """Given the address for the alpha string and the beta string
        return the element of the CIvector at C( i_a, i_b), ivec

        Args:
            i_a (int) - alpha string index representation
            i_b (int) - beta string index representation
            ivec (int) - index for the configuration to access

        Returns:
            (complex) - return a scalar value of wavefunction consistent with
                the addressing passed.
        """
        return self.coeff[self._index_to_address(i_a, i_b), ivec]


    def _address_from_strings(self, a_str: int, b_str: int) -> int:
        """Passing in the occupation representation of the system is often more
        convenient than the index associated with that string.

        Args:
            a_str (bitstring) - an occupation representation for alpha
                electrons
            b_str (bitstring) - an occupation representation for beta
                electrons

        Return:
            (int) - the index into the CI space consistent with the occupation
                strings
        """
        occa = integer_index(a_str)
        occb = integer_index(b_str)
        i_a = (build_string_address(self._nalpha, self._norb, occa))
        i_b = (build_string_address(self._nbeta, self._norb, occb))
        return self._index_to_address(i_a, i_b)


    def _string_from_address(self, address: int) -> List[int]:
        """Find out which strings are connected to the address index.

        Args:
            address (int) - a pointer into the fci space

        Returns:
            list[(bitstring)] - the alpha string and beta string that correspond
                to the index into the Ci space
        """
        adr_a = address // self._lenb
        return [self.str_alpha(adr_a), self.str_beta(address -
                                                     adr_a*self._lenb)]


    def ci_i(self, address: int, ivec: int) -> complex:
        """Returns the value of the coefficient corresponding to the
            determinant accessed by a specific address and configuration
            index.

        Args:
            address (int) - index into the ci space
            ivec (int) - index for the ci configuration to access

        Returns:
            (complex) - return a scalar value of wavefunction consistent with
                the addressing passed.
        """
        return self.coeff[address, ivec]


    def cc_s(self, astr: int, bstr: int, ivec: int) -> complex:
        """Returns the value of the coefficient corresponding to the
        determinant given by the string representation and a configuration
        index.

        Args:
            astr (bitstring) - integer representation of the alpha bitstring
            bstr (bitstring) - integer representation of the beta bitstring
            ivec (int) - index for the ci configuration to access

        Returns:
            (complex) - return a scalar value of wavefunction consistent with
                the addressing passed.
        """
        return self.coeff[self._address_from_strings(astr, bstr), ivec]


    def add_element(self, astr: int, bstr: int, ivec: int, value: complex) -> None:
        """Add a value to an existing element in the wavefunction

        Args:
            astr (bitstring) - integer representation of the alpha bitstring
            bstr (bitstring) - integer representation of the beta bitstring
            ivec (int) - index for the ci configuration to access
            value (complex) - value to set the coefficient to

        Returns:
            nothing - Modifies the wavefunction in place
        """
        self.coeff[self._address_from_strings(astr, bstr), ivec] += value


    def set_element(self, astr: int, bstr: int, ivec: int,
                    value: complex) -> None:
        """Override a value stored in the wavefunction

        Args:
            astr (bitstring) - integer representation of the alpha bitstring
            bstr (bitstring) - integer representation of the beta bitstring
            ivec (int) - index for the ci configuration to access
            value (complex) - value to set the coefficient to

        Returns:
            nothing - Modifies the wavefunction in place
        """
        self.coeff[self._address_from_strings(astr, bstr), ivec] = value


    def scale(self, sval: complex, ivec: int = None):
        """ Scale the wavefunction by the value sval

        Args:
            sval (complex) - value to scale by
            ivec (int) - optionally provide an index to point at a specfic
                vector to scale

        Returns:
            nothing - Modifies the wavefunction in place
        """
        if ivec:
            self.coeff[:, ivec] *= sval
            return
        self.coeff *= sval


    def insequence_generator(self,
                             vector: int = 0
                            ) -> Generator[List[Any], None, None]:
        """Iterate through the addresses of the wavefunction and return each
        determinant identifier and its coefficient.

        Args:
            vector (int) - an index indicating which state should be returned
                from the generator

        Returns:
            list[complex, bitstring, bitstring]
        """
        addr = 0
        ivec = vector
        while addr < self._lena*self._lenb:
            addra, addrb = self._string_from_address(addr)
            yield [self.coeff[addr, ivec], addra, addrb]
            addr += 1


    def normalize(self, vec: List[int] = None) -> None:
        """For each vector in the ci space, scale it such that the Frobenius
        inner product equals 1.

        Args:
            vec (list[int]) - a list of integers indicating which vectors
                should be normalized.

        Returns:
            none - modified in place
        """
        if vec is None:
            vec = list(range(self._cidim))

        for ivec in vec:
            norm = numpy.linalg.norm(self.coeff[:, ivec])

            # numpy errstate treats (1 / 0) and (0 / 0) as two different exceptions
            # which is why there are two errstate arguments but we will treat them
            # as one

            with numpy.errstate(divide='raise', invalid='raise'):
                try:
                    sval = 1.0 / norm
                    self.scale(sval, ivec)
                except FloatingPointError:
                    print('Dividing by zero in normalize. norm = {}'.format(norm))
                    print('config identity {}'.format((self._nele, self._m_s)))
                    print('Vector {}'.format(ivec))
                    raise


    def set_wfn(self, vrange: List[int] = None, strategy: str = None,
                raw_data: numpy.ndarray = None) -> None:
        """Set the values of the fqedata wavefunction based on a strategy

        Args:
            vrange (list(int)) - a list of vectors to set with this strategy.
                If no vectors are given every vector is set
            strategy (string) - the procedure to follow to set the coeffs
            raw_data (numpy.array(dim(self._lena*self._lenb, :),
                                  dtype=numpy.complex64)) - the values to use
                if setting from data.  If vrange is supplied, the first column
                in data will correspond to the first index in vrange

        Returns:
            nothing - modifies the wavefunction in place
        """

        strategy_args = [
            'ones',
            'zero',
            'lowest',
            'random',
            'from_data'
        ]

        if strategy is None and raw_data is None:
            raise ValueError('No strategy and no data passed.'
                             ' Cannot initialize')
        if strategy == 'from_data' and raw_data is None:
            raise ValueError('No data passed to initialize from')
        if raw_data is not None and strategy not in ['from_data', None]:
            raise ValueError('Inconsistent strategy for set_vec passed with'
                             'data')
        if strategy not in strategy_args:
            raise ValueError('Unknown Argument passed to set_vec')

        if strategy == 'from_data':
            chkdim = raw_data.shape
            if chkdim[0] != self.ci_space_length:
                print('Local {} != passed {}'.format(self.ci_space_length,
                                                     chkdim[0]))
                raise ValueError('Dim 0 in passed data is wrong')
            if vrange:
                minvec = min(vrange)
                if minvec < 0:
                    raise ValueError('Vector index {}'
                                     ' is meaningless'.format(minvec))
                maxvec = max(vrange)
                if maxvec >= self._cidim:
                    raise ValueError('Vector index {} greater than ci'
                                     ' space'.format(maxvec))
                if len(vrange) > self._cidim:
                    raise ValueError('Too many vectors passed to set_vec. We'
                                     ' will not guess the correct behavior')
            else:
                if chkdim[1] != self._cidim:
                    raise ValueError('Dim 1 in passed data is wrong')


        if vrange:
            for i in vrange:
                raw_ind = 0
                if strategy == 'ones':
                    self.coeff[:, i].fill(1. + .0j)
                elif strategy == 'zero':
                    self.coeff[:, i].fill(.0 + .0j)
                elif strategy == 'lowest':
                    self.coeff[:, i].fill(.0 + .0j)
                    self.coeff[0, i] = 1. + .0j
                elif strategy == 'random':
                    self.coeff[:, i] = rand_wfn(self.ci_space_length)
                elif strategy == 'from_data':
                    self.coeff[:, i] = numpy.copy(raw_data[:, raw_ind])
                    raw_ind += 1
        else:
            if strategy == 'ones':
                self.coeff.fill(1. + .0j)
            elif strategy == 'zero':
                self.coeff.fill(0. + .0j)
            elif strategy == 'lowest':
                self.coeff.fill(0. + .0j)
                self.coeff[0, :] = 1. + .0j
            elif strategy == 'random':
                for i in range(self._cidim):
                    self.coeff[:, i] = rand_wfn(self.ci_space_length)
            elif strategy == 'from_data':
                self.coeff = numpy.copy(raw_data)
