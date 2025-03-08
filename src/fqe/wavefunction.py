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
"""Wavefunction class for accessing and manipulation of the state of interest
"""
#zeros_like is incorrectly flagged by pylint
#pylint: disable=unsupported-assignment-operation
#there are many instances where access to protected members/methods simplify the
#code structure. They are not exposed to the users, and therefore, harmless
#pylint: disable=protected-access

import copy
import os
import math
from typing import (Any, Callable, cast, Dict, KeysView, List, Optional, Tuple,
                    Union)

import pickle
import numpy
from scipy import linalg
from scipy.special import jv

from fqe.fqe_decorators import wrap_apply, wrap_apply_generated_unitary
from fqe.fqe_decorators import wrap_time_evolve, wrap_rdm
from fqe.fqe_data import FqeData
from fqe.fqe_data_set import FqeDataSet
from fqe.util import alpha_beta_electrons
from fqe.util import map_broken_symmetry
from fqe.util import sort_configuration_keys
from fqe.util import vdot
from fqe.hamiltonians import hamiltonian, sparse_hamiltonian, \
                             diagonal_hamiltonian, diagonal_coulomb, \
                             restricted_hamiltonian
from fqe.bitstring import count_bits
from fqe.fqe_ops import fqe_operator, fqe_ops_utils
from fqe.wick import wick


class Wavefunction:
    """Wavefunction is the central object for manipulaion in the
    OpenFermion-FQE.
    """

    def __init__(self,
                 param: Optional[List[List[int]]] = None,
                 broken: Optional[Union[List[str], str]] = None) -> None:
        """
        Args:
            param (list[list[n, ms, norb]]): the constructor accepts a list of \
              parameter lists.  The parameter lists are comprised of

              p[0] (integer) - number of particles;

              p[1] (integer) - z component of spin angular momentum;

              p[2] (integer) - number of spatial orbitals

            broken (str): pass in the symmetries that should be preserved by \
                the wavefunction.

        Member Variables:
            _conserve_spin (bool): When this flag is true, the wavefunction \
                will maintain a constant m_s

            _conserve_number (bool): When this flag is true, the wavefunction \
                will maintain a constant nele

            _civec (dict[(int, int)] -> FqeData): This is a dictionary for \
                FqeData objects.  The key is a tuple defined by the number of \
                electrons and the spin projection of the system.
        """
        self._symmetry_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._conserved: Dict[str, int] = {}

        self._conserve_spin: bool = False
        if broken is None or 'spin' not in broken:
            self._conserve_spin = True

        self._conserve_number: bool = False
        if broken is None or 'number' not in broken:
            self._conserve_number = True

        self._norb: int = 0
        self._civec: Dict[Tuple[int, int], 'FqeData'] = {}

        if not self._conserve_spin and not self._conserve_number:
            raise TypeError('Number and spin non-conserving waveunfction is' \
                            ' the complete Fock space.')

        if param:
            user_input_norbs = set([x[2] for x in param])
            if len(user_input_norbs) != 1:
                raise ValueError('Number of orbitals is not consistent')

            self._norb = list(user_input_norbs)[0]
            for i in param:
                nalpha, nbeta = alpha_beta_electrons(i[0], i[1])
                self._civec[(i[0], i[1])] = FqeData(nalpha, nbeta, self._norb)

            if self._conserve_number:
                self._conserved['n'] = param[0][0]

            if self._conserve_spin:
                self._conserved['s_z'] = param[0][1]

                if not self._conserve_number:
                    self._symmetry_map = map_broken_symmetry(
                        param[0][1], param[0][2])

    def __add__(self, other: 'Wavefunction') -> 'Wavefunction':
        """Intrinsic addition function to combine two wavefunctions.  This acts \
        to iterate through the wavefunctions, combine coefficients of \
        configurations they have in common and add configurations that are \
        unique to each one.  The values are all combined into a new \
        wavefunction object

        Args:
            other (wavefunction.Wavefunction): the second wavefunction to \
                add with the local wavefunction

        Returns:
            wfn (wavefunction.Wavefunction): a new wavefunction with the \
                values set by adding together values
        """
        out = copy.deepcopy(self)
        out.ax_plus_y(1.0, other)
        return out

    def __iadd__(self, wfn: 'Wavefunction') -> 'Wavefunction':
        """Same is __add___ but performed in-place

        Args:
            wfn (wavefunction.Wavefunction): the second wavefunction to \
                add with the local wavefunction

        Returns:
            Wavefunction: self
        """
        self.ax_plus_y(1.0, wfn)
        return self

    def __sub__(self, other: 'Wavefunction') -> 'Wavefunction':
        """Intrinsic subtraction function to combine two wavefunctions.  This
        acts to iterate through the wavefunctions, combine coefficients of
        configurations they have in common and include configurations that are
        unique to each one. The values are all combined into a new
        wavefunction object

        Args:
            other (wavefunction.Wavefunction): the second wavefunction that \
                will be subtracted from the first wavefunction

        Returns:
            wfn (wavefunction.Wavefunction): a new wavefunction with the
                values set by subtracting the wfn from the first
        """
        out = copy.deepcopy(self)
        out.ax_plus_y(-1.0, other)
        return out

    def __getitem__(self, key: Tuple[int, int]) -> complex:
        """Element read access to the wave function.
        Args:
            key (Tuple[int, int]): a pair of strings for alpha and beta

        Returns:
            (complex): the value of the wave function
        """
        astr, bstr = key[0], key[1]
        sector = (count_bits(astr) + count_bits(bstr),
                  count_bits(astr) - count_bits(bstr))
        return self._civec[sector][key]

    def __setitem__(self, key: Tuple[int, int], value: complex) -> None:
        """Element write access to the wave function.
        Args:
            key (Tuple[int, int]): a pair of strings for alpha and beta

            value (complex): the value to be set to the wave function
        """
        astr, bstr = key[0], key[1]
        sector = (count_bits(astr) + count_bits(bstr),
                  count_bits(astr) - count_bits(bstr))
        self._civec[sector][key] = value

    def empty_copy(self) -> 'Wavefunction':
        """create a copy of self with zero coefficients

        Returns:
            Wavefunction: a new object with zero coefficients
        """
        out = Wavefunction()
        out._norb = self._norb
        out._conserved = self._conserved
        out._symmetry_map = self._symmetry_map
        for key, civec in self._civec.items():
            out._civec[key] = civec.empty_copy()
        return out

    def _copy_beta_inversion(self) -> 'Wavefunction':
        """Return a copy of the wavefunction with the beta particle and hole
        inverted.

        Returns:
            Wavefunction: wavefuction with beta particle/hole conjugation
        """
        norb = self._norb
        m_s = self._conserved['s_z']

        nele = norb + m_s
        param = []
        maxb = min(norb, nele)
        minb = nele - maxb
        param = [
            [nele, nele - nbeta * 2, norb] for nbeta in range(minb, maxb + 1)
        ]

        inverted = Wavefunction(param, broken=['spin'])

        data = {}
        for key, sector in self._civec.items():
            work = ((key[0] + key[1]) // 2, (key[0] - key[1]) // 2)
            nkey = self._symmetry_map[work]
            data[(nkey[0] + nkey[1],
                  nkey[0] - nkey[1])] = sector.beta_inversion()

        inverted.set_wfn(strategy='from_data', raw_data=data)

        return inverted

    def ax_plus_y(self, sval: complex, wfn: 'Wavefunction') -> None:
        """Perform scale and add of the wavefunction. The result will be stored
        in self.

        Args:
            sval (complex): a factor to be multiplied to wfn

            wfn (Wavefunction): a wavefunction to be added to self
        """
        if self._civec.keys() != wfn._civec.keys():
            raise ValueError('inconsistent sectors in Wavefunction.ax_plus_y')

        for sector in self._civec:
            self._civec[sector].ax_plus_y(sval, wfn._civec[sector])

    def sector(self, key: Tuple[int, int]) -> 'FqeData':
        """Return a specific sector of the wavefunction using a key.

        Args:
            key (Tuple[int, int]): key for ci vector
        Returns:
            FqeData: corresponding sector as an FqeData object
        """
        return self._civec[key]

    def sectors(self) -> KeysView[Tuple[int, int]]:
        """
        Return:
            KeysView[Tuple[int, int]]: a list of the configuration keys \
                in the wavefunction
        """
        return self._civec.keys()

    def conserve_number(self) -> bool:
        """
        Returns:
            (bool): True if this wave function conserves the number symmetry
        """
        return self._conserve_number

    def conserve_spin(self) -> bool:
        """
        Returns:
            (bool): True if this wave function conserves the spin (Sz) \
                symmetry. Otherwise False
        """
        return self._conserve_spin

    def norb(self) -> int:
        """
        Returns:
            (int): the number of orbitals
        """
        return self._norb

    def norm(self) -> float:
        """Calculate the norm of the wavefuntion

        Returns:
            (float): the norm
        """
        normall = 0.0
        for sector in self._civec.values():
            normall += sector.norm()**2
        normall = math.sqrt(normall)
        return normall

    def normalize(self) -> None:
        """Generte the wavefunction norm and then scale each element by that
        value.
        """
        self.scale(1.0 / self.norm())

    @wrap_apply
    def apply(self, hamil: 'hamiltonian.Hamiltonian') -> 'Wavefunction':
        """ Returns a wavefunction subject to application of the Hamiltonian
        (or more generally, the operator).

        Args:
            hamil (Hamiltonian): Hamiltonian to be applied

        Returns:
            (Wavefunction): resulting wave function array
        """
        if not self._conserve_number or not hamil.conserve_number():
            if self._conserve_number:
                raise TypeError('Number non-conserving hamiltonian passed to'
                                ' number conserving wavefunction')
            if hamil.conserve_number():
                raise TypeError('Number conserving hamiltonian passed to'
                                ' number non-conserving wavefunction')

        if isinstance(hamil, sparse_hamiltonian.SparseHamiltonian):

            transformed = self._apply_few_nbody(hamil)

        else:
            if self._conserve_spin and not self._conserve_number:
                out = self._copy_beta_inversion()
            else:
                out = self

            if isinstance(hamil, diagonal_hamiltonian.Diagonal):
                transformed = out._apply_diagonal(hamil)
            elif isinstance(hamil, diagonal_coulomb.DiagonalCoulomb):
                transformed = out._apply_diagonal_coulomb(hamil)
            else:
                if isinstance(hamil,
                              restricted_hamiltonian.RestrictedHamiltonian):
                    expected = self._norb
                else:
                    expected = self._norb * 2
                if hamil.dim() != expected:
                    raise ValueError('Hamiltonian has incorrect size:' \
                                     + ' expected {}'.format(expected) \
                                     + ' provided {}'.format(hamil.dim()))

                transformed = out._apply_array(hamil.tensors(), hamil.e_0())

            if self._conserve_spin and not self._conserve_number:
                transformed = transformed._copy_beta_restore(
                    self._conserved['s_z'], self._norb, self._symmetry_map)

        return transformed

    def _apply_array(self, array: Tuple[numpy.ndarray, ...],
                     e_0: complex) -> 'Wavefunction':
        """Return a wavefunction subject to application of the numpy array as

        .. math::
            h[i, j]a_i^+ a_j|Psi>
            h[i, j, k, l]a_i^+(rho) a_j^+(eta) a_k(rho) a_l(eta)|Psi>

        Arg:
            array (numpy.array): numpy array

            e_0 (complex): scalar part of the Hamiltonian

        Returns:
            newwfn (Wavefunction): a new initialized wavefunction object

        """
        if self._conserve_spin:
            assert array[0].shape[0] == self._norb or array[0].shape[
                0] == self._norb * 2
            out = copy.deepcopy(self)
            for _, sector in out._civec.items():
                sector.apply_inplace(array)
        else:
            assert array[0].shape[0] == self._norb * 2
            out = copy.deepcopy(self)
            nsectors = out._number_sectors()
            for _, nsector in nsectors.items():
                nsector.apply_inplace(array)

        if numpy.abs(e_0) > 1.e-15:
            out.ax_plus_y(e_0, self)

        return out

    def _apply_diagonal(self, hamil: 'diagonal_hamiltonian.Diagonal'
                       ) -> 'Wavefunction':
        """Applies the diagonal operator to the wavefunction

        Args:
            hamil (Diagonal): diagonal Hamiltonian to be applied

        Returns:
            (Wavefunction): resulting wave function
        """
        out = copy.deepcopy(self)

        for _, sector in out._civec.items():
            sector.apply_diagonal_inplace(hamil.diag_values())

        if numpy.abs(hamil.e_0()) > 1.e-15:
            out.ax_plus_y(hamil.e_0(), self)

        return out

    def _apply_diagonal_coulomb(self, hamil: 'diagonal_coulomb.DiagonalCoulomb'
                               ) -> 'Wavefunction':
        """Applies the diagonal coulomb operator to the wavefunction

        Args:
            hamil (DiagonalCoulomb): diagonal coulomb Hamiltonian to be applied

        Returns:
            (Wavefunction): resulting wave function
        """
        out = copy.deepcopy(self)

        for _, sector in out._civec.items():
            diag, array = hamil._tensor[1], hamil._tensor[2]
            sector.apply_diagonal_coulomb(diag, array, inplace=True)

        if numpy.abs(hamil.e_0()) > 1.e-15:
            out.ax_plus_y(hamil.e_0(), self)

        return out

    def _number_sectors(self) -> Dict[int, FqeDataSet]:
        """An internal utility function that groups FqeData into a set of
        FqeDataSet that corresponds to the same number of electrons.  It checks
        spin completeness and raises an exception if the wave function space is
        not spin complete

        Returns:
            Dict[int, FqeDataSet]: stores FqeDataSet for each number of
                particles. Keys are the number of particles.
        """
        norb = self.norb()
        numbers = set(key[0] for key in self._civec)
        numbersectors = {}
        for nele in numbers:
            # generate all possible sz
            maxalpha = min(norb, nele)
            minalpha = nele - maxalpha
            sectors = {}
            sp_compl = set(((nele, 2 * nalpha - nele)
                            for nalpha in range(minalpha, maxalpha + 1)))

            if set(self._civec.keys()).intersection(sp_compl) != sp_compl:
                raise ValueError('Wave function space is not spin complete.')

            for nalpha in range(minalpha, maxalpha + 1):
                nbeta = nele - nalpha
                sectors[(nalpha, nbeta)] = self._civec[(nele, nalpha - nbeta)]
            dataset = FqeDataSet(nele, norb, sectors)
            numbersectors[nele] = dataset
        return numbersectors

    def _copy_beta_restore(self, s_z: int, norb: int,
                           map_symm: Dict[Tuple[int, int], Tuple[int, int]]
                          ) -> 'Wavefunction':
        """Return a copy of the wavefunction with beta restored back to number
        breaking/spin conserving.

        Args:
            s_z (int): the value of Sz

            norb (int): the number of orbitals in the system

            map_symm (Dict[Tuple[int,int], Tuple[int,int]]): dictionary that maps \
                between number-conserved and spin-conserved wave function sectors

        Returns:
            Wavefunction: restored wavefunction
        """
        max_alpha = min(norb, norb + s_z)
        min_alpha = max(s_z, 0)

        param = [[2 * nalpha - s_z, s_z, norb]
                 for nalpha in range(min_alpha, max_alpha + 1)]

        restored = Wavefunction(param, broken=['number'])

        data = {}
        for key in param:
            work = ((key[0] + key[1]) // 2, (key[0] - key[1]) // 2)
            nkey = map_symm[work]
            other = (nkey[0] + nkey[1], nkey[0] - nkey[1])
            data[(key[0], key[1])] = self._civec[other].beta_inversion()

        restored.set_wfn(strategy='from_data', raw_data=data)

        return restored

    @wrap_apply_generated_unitary
    def apply_generated_unitary(self,
                                time: float,
                                algo: str,
                                hamil: 'hamiltonian.Hamiltonian',
                                accuracy: float = 1.0E-15,
                                expansion: int = 30,
                                spec_lim: Optional[List[float]] = None
                               ) -> 'Wavefunction':
        """Perform the exponentiation of fermionic algebras to the
        wavefunction according the method and accuracy.

        Args:
            time (float): the final time value to evolve to

            algo (string): polynomial expansion algorithm to be used

            hamil (Hamiltonian): the Hamiltonian used to generate the unitary

            accuracy (float): the accuracy to which the system should be evolved

            expansion (int): the maximum number of terms in the polynomial expansion.

            spec_lim (List[float]): spectral range of the Hamiltonian, the length of \
                the list should be 2. Optional.

        Returns:
            newwfn (Wavefunction): a new initialized wavefunction object
        """

        assert isinstance(hamil, hamiltonian.Hamiltonian)
        if not isinstance(expansion, int):
            raise TypeError(
                "expansion must be an int. You provided {}".format(expansion))

        algo_avail = ['taylor', 'chebyshev']

        assert algo in algo_avail

        if not isinstance(hamil, sparse_hamiltonian.SparseHamiltonian) \
            and self._conserve_spin and not self._conserve_number:
            base = self._copy_beta_inversion()
        else:
            base = self

        max_expansion = expansion

        if algo == 'taylor':
            ham_arrays = hamil.iht(time)
            time_evol = copy.deepcopy(base)
            work = copy.deepcopy(base)
            coeff = 1.0
            for order in range(1, max_expansion):
                work = work.apply(ham_arrays)
                coeff /= order
                time_evol.ax_plus_y(coeff, work)
                if work.norm() * numpy.abs(coeff) < accuracy:
                    break
            else:
                raise RuntimeError("maximum taylor expansion limit reached")

        elif algo == 'chebyshev':

            assert spec_lim, 'Spectral range was not provided.' + \
                                 ' Provide upper and lower limits.'

            wprime = 0.9875
            ascale = (spec_lim[1] - spec_lim[0]) / (2.0 * wprime)
            eshift = -(spec_lim[0] + ascale * wprime)

            time_evol = copy.deepcopy(base)
            time_evol.scale(jv(0, ascale * time))
            minus = copy.deepcopy(base)

            current = minus.apply(hamil)
            current.ax_plus_y(eshift, minus)
            current.scale(1.0 / ascale)
            time_evol.ax_plus_y(2.0 * jv(1, ascale * time) * (-1.j), current)

            for order in range(2, max_expansion):
                minus.scale(-1.0)
                minus.ax_plus_y(2.0 / ascale, current.apply(hamil))
                minus.ax_plus_y(2.0 * eshift / ascale, current)
                current, minus = minus, current

                coeff = 2.0 * jv(order, ascale * time) * (-1.j)**order
                time_evol.ax_plus_y(coeff, current)

                if current.norm() * numpy.abs(coeff) < accuracy:
                    break
            else:
                raise RuntimeError("maximum chebyshev expansion limit reached")

            time_evol.scale(numpy.exp(eshift * time * 1.j))

        if numpy.abs(hamil.e_0() * time) > 1.e-15:
            time_evol.scale(numpy.exp(-1.j * time * hamil.e_0()))

        if self._conserve_spin and not self._conserve_number:
            time_evol = time_evol._copy_beta_restore(self._conserved['s_z'],
                                                     self._norb,
                                                     self._symmetry_map)
        return time_evol

    def get_coeff(self, key: Tuple[int, int]) -> numpy.ndarray:
        """Retrieve a vector from a configuration in the wavefunction

        key indicates wavefunction sector by [num_alpha, num_beta]

        Args:
            key (Tuple[int, int]): a key identifying the configuration to access

        Returns:
            numpy.array(dtype=numpy.complex128): coeff that corresponds to key
        """
        return self._civec[key].coeff

    def max_element(self) -> complex:
        """
        Return:
            (complex): the largest magnitude value in the wavefunction
        """
        maxval = 0.0
        for config in self._civec.values():
            configmax = max(config.coeff.max(),
                            config.coeff.min(),
                            key=numpy.abs)
            maxval = max(configmax, maxval)

        return maxval

    def print_wfn(self, threshold: float = 0.001, fmt: str = 'str') -> None:
        """Print occupations and coefficients to the screen.

        Args:
            threshold (float): only print CI vector values such that \
              :math:`|c|` > threshold.

            fmt (string): formats print according to argument
        """

        def _print_format(fmt: str) -> Callable[[int, int], str]:
            """ Select the function which will perform formatted printing
            """
            if fmt == 'occ':

                def _occupation_format(iastring: int, ibstring: int):
                    """occ - Prints a string indicating occupation state of each spatial
                    orbital.  A doubly occupied orbital will be indicated with "2",
                    a singly occupied orbital will get "a" or "b" depending on the
                    spin state.  An empy orbital will be ".".
                    """
                    astring = int(iastring)
                    bstring = int(ibstring)
                    occstr = [
                        '.' for _ in range(
                            max(astring.bit_length(), bstring.bit_length()))
                    ]
                    docc = astring & bstring

                    def build_occ_value(bstr: int, char: str,
                                        occstr: List[str]):
                        """Fill a list with a character corresponding to the
                        location of '1' bits in bitstring

                        Args:
                            bstr (int): a bitstring to examine

                            char (str): a string to put into the list

                            occstr (list): a list to store the value
                                corresponding to flipped bits
                        """
                        ind = 1
                        while bstr:
                            if bstr & 1:
                                occstr[-ind] = char
                            ind += 1
                            bstr = bstr >> 1

                    build_occ_value(astring, 'a', occstr)
                    build_occ_value(bstring, 'b', occstr)
                    build_occ_value(docc, '2', occstr)

                    return ''.join(occstr).rjust(self._norb, '.')

                return _occupation_format

            def _string_format(iastring: int, ibstring: int) -> str:
                """Prints a binary string indicating which orbital creation
                operators are acting on the vacuum with a 1. The position in
                the string indicates the index of the orbital. The beta string
                is shown acting on the vacuum first.

                Args:
                    iastring (int): alpha string

                    ibstring (int): alpha string

                Returns:
                    str: representation of binary string
                """
                astring = int(iastring)
                bstring = int(ibstring)
                fmt_a = bin(1 << self._norb | astring)
                fmt_b = bin(1 << self._norb | bstring)
                return "a'{}'b'{}'".format(fmt_a[3:], fmt_b[3:])

            return _string_format

        print_format = _print_format(fmt)

        config_in_order = sort_configuration_keys(self.sectors())
        for key in config_in_order:
            self._civec[key].print_sector(pformat=print_format,
                                          threshold=threshold)

    def read(self, filename: str, path: str = os.getcwd()) -> None:
        """Initialize a wavefunction from a binary file.

        Args:
            filename (str): the name of the file to write the wavefunction to.

            path (str): the path to save the file.  If no path is given then \
              it is saved in the current working directory.
        """
        with open(os.path.join(path, filename), 'r+b') as wfnfile:
            wfn_data = pickle.load(wfnfile)

        self._symmetry_map = wfn_data[0]
        self._conserved = wfn_data[1]
        self._conserve_spin = wfn_data[2]
        self._conserve_number = wfn_data[3]
        self._norb = wfn_data[4]

        for sector in wfn_data[5:]:
            self._civec[(sector[0][0], sector[0][1])] = sector[1]

    def save(self, filename: str, path: str = os.getcwd()) -> None:
        """Save the wavefunction into path/filename.

        Args:
            filename (str): the name of the file to write the wavefunction to.

            path (str): the path to save the file.  If no path is given, then \
              it is saved in the current working directory.
        """
        wfn_data = [
            self._symmetry_map, self._conserved, self._conserve_spin,
            self._conserve_number, self._norb
        ]

        for key in self._civec:
            wfn_data.append([key, self._civec[key]])

        with open(os.path.join(path, filename), 'w+b') as wfnfile:
            pickle.dump(wfn_data, wfnfile)

    def scale(self, sval: complex) -> None:
        """ Scale each configuration space by the value sval

        Args:
            sval (complex): value to scale by
        """

        sval = complex(sval)  # type: ignore

        for sector in self._civec.values():
            sector.scale(sval)

    def set_wfn(self,
                strategy: str = 'ones',
                raw_data: Optional[Dict[Tuple[int, int], numpy.ndarray]] = None
               ) -> None:
        """Set the values of the wavefunction inplace based on a strategy.

        Args:
            strategy (string): The procedure to follow to set the wavefunction \
                coefficients. One of 'random', 'hartree-fock', 'ones', 'zeros', \
                or 'from_data'. If 'from_data', raw_data must be provided.

            raw_data (Dict[Tuple[int, int], numpy.ndarray]): The values to use \
                if setting from data. Optional.
        """
        if strategy == 'from_data' and not raw_data:
            raise ValueError('No data provided for set_wfn')

        if strategy == 'from_data':
            for key, data in raw_data.items():  # type: ignore
                self._civec[key].set_wfn(strategy='from_data', raw_data=data)
        elif strategy == 'hartree-fock':
            # make sure we only have 1 sector
            if len(self.sectors()) != 1:
                raise ValueError(("Hartree-Fock wf initialization only works "
                                  "with single sector wavefunctions"))
            for sector in self._civec.values():
                sector.set_wfn(strategy=strategy)
        else:
            for sector in self._civec.values():
                sector.set_wfn(strategy=strategy)

        if strategy == 'random':
            self.normalize()

    def transform(
            self,
            rotation: numpy.ndarray,
            low: Optional[numpy.ndarray] = None,
            upp: Optional[numpy.ndarray] = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, 'Wavefunction']:
        """Transform the wavefunction using the orbtial rotation matrix and
        return the new wavefunction and the permutation matrix for the unitary
        transformation. This is an internal code, so performs minimal checking

        Args:
            rotation (numpy.ndarray): MO rotation matrix, which is unitary

            low (numpy.ndarray): L in the LU decomposition (optional)

            upp (numpy.ndarray): U in the LU decomposition (optional)

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray, 'Wavefunction'): \
                permutation, L, U, and transformed wavefunction
        """
        norb = self._norb
        external = low is not None
        assert external == (upp is not None)
        if external:
            assert numpy.allclose(rotation, low @ upp)  # type: ignore

        def ludecomp(rotmat: numpy.ndarray
                    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            """Returns permutation, lower triangular, and upper triangular
            matrices from the LU decomposition. Note that the rotation matrix
            is Hermitian conjugated.

            Args:
                rotmat (numpy.ndarray): MO rotation matrix, which is unitary

                (numpy.ndarray, numpy.ndarray, numpy.ndarray): permutation, \
                    L, and U from the LU decomposition.
            """
            tmat = rotmat.transpose().conjugate()
            return linalg.lu(tmat)

        def transpose_matrix(low: numpy.ndarray, upp: numpy.ndarray
                            ) -> Tuple[numpy.ndarray, numpy.ndarray]:
            """Returns Hermitian conjugate of the L and U factors. The diagonal
            elements are in the upper triagular matrix. This transposition seeks to
            compensate the transposition in ludecomp above.

            Args:
                low (numpy.ndarray): L in the LU decomposition

                upp (numpy.ndarray): U in the LU decomposition

            Returns:
                (numpy.ndarray, numpy.ndarray): L and U after transposition \
                    where L and U are lower- and upper-triangular, respectively
            """
            ndim = low.shape[0]
            assert low.shape[1] == ndim and upp.shape == (ndim, ndim)
            lowt = copy.deepcopy(low)
            uppt = copy.deepcopy(upp)
            for irow in range(ndim):
                for icol in range(irow + 1, ndim):
                    uppt[irow, icol] /= uppt[irow, irow]
                lowt[irow, irow], uppt[irow, irow] = uppt[irow, irow], lowt[
                    irow, irow]
                for icol in range(irow):
                    lowt[irow, icol] *= lowt[icol, icol]
            return uppt.T.conj(), lowt.T.conj()

        def process_matrix(low: numpy.ndarray,
                           upp: numpy.ndarray) -> numpy.ndarray:
            """Returns an operator using which the wavefuction will be transformed.

            Args:
                low (numpy.ndarray): L in the LU decomposition

                upp (numpy.ndarray): U in the LU decomposition

            Returns:
                (numpy.ndarray): matrix elements of the transformation operator T
            """
            ndim = low.shape[0]
            assert low.shape[1] == ndim and upp.shape == (ndim, ndim)
            unitmat = numpy.identity(ndim)
            output = linalg.solve_triangular(upp, unitmat)

            for icol in range(ndim):
                for irow in range(icol + 1, ndim):
                    output[irow, icol] -= low[irow, icol]
                output[icol, icol] -= 1.0
            return output

        current = self
        perm = None

        if not self._conserve_spin:
            if not external:
                perm, low, upp = ludecomp(rotation)
                lowt, uppt = transpose_matrix(low, upp)
            else:
                lowt, uppt = low, upp
            output = process_matrix(lowt, uppt)
            for icol in range(norb * 2):
                work = numpy.zeros_like(rotation)
                work[:, icol] = output[:, icol]
                onefwfn = current.apply((work,))
                current.ax_plus_y(1.0, onefwfn)
        else:
            if rotation.shape[0] == norb:
                if not external:
                    perm, low, upp = ludecomp(rotation)
                    lowt, uppt = transpose_matrix(low, upp)
                else:
                    lowt, uppt = low, upp
                output = process_matrix(lowt, uppt)
                for _, civec in current._civec.items():
                    civec.apply_columns_recursive_inplace(output, output)
            elif rotation.shape[0] == norb * 2:
                assert numpy.std(rotation[:norb, norb:]) \
                       + numpy.std(rotation[norb:, :norb]) < 1.0e-8
                if not external:
                    perm1, low1, upp1 = ludecomp(rotation[:norb, :norb])
                    perm2, low2, upp2 = ludecomp(rotation[norb:, norb:])
                    lowt1, uppt1 = transpose_matrix(low1, upp1)
                    lowt2, uppt2 = transpose_matrix(low2, upp2)
                else:
                    lowt1 = low[:norb, :norb]  # type: ignore
                    lowt2 = low[norb:, norb:]  # type: ignore
                    uppt1 = upp[:norb, :norb]  # type: ignore
                    uppt2 = upp[norb:, norb:]  # type: ignore
                output1 = process_matrix(lowt1, uppt1)
                output2 = process_matrix(lowt2, uppt2)
                for _, civec in current._civec.items():
                    civec.apply_columns_recursive_inplace(output1, output2)
                if not external:
                    perm = numpy.zeros_like(rotation)
                    perm[:norb, :norb] = perm1[:, :]
                    perm[norb:, norb:] = perm2[:, :]
                    upp = numpy.zeros_like(rotation)
                    upp[:norb, :norb] = upp1[:, :]
                    upp[norb:, norb:] = upp2[:, :]
                    low = numpy.zeros_like(rotation)
                    low[:norb, :norb] = low1[:, :]
                    low[norb:, norb:] = low2[:, :]

        return perm, low, upp, current

    @wrap_time_evolve
    def time_evolve(
            self,
            time: float,
            hamil: Union['fqe_operator.FqeOperator', 'hamiltonian.Hamiltonian'],
            inplace: bool = False) -> 'Wavefunction':
        """Perform time evolution of the wavefunction given Fermion Operators
        either as raw operations or wrapped up in a Hamiltonian. The evolved
        wavefunction is always returned (so the return value should be set), but
        with `inplace=True` the input wavefunction may be partially or fully
        overwritten for the sake of efficiency.

        Args:
            time (float): the duration by which to evolve the operators

            hamil (Hamiltoninan or FermionOperator): Hamiltonian to be used for \
                time evolution

            inplace (bool): whether inplace memory will be used for efficiency

        Returns:
            (Wavefunction): a wavefunction object that has been time evolved
        """
        assert isinstance(hamil, hamiltonian.Hamiltonian)

        if not self._conserve_number or not hamil.conserve_number():
            if self._conserve_number:
                raise TypeError('Number non-conserving hamiltonian passed to'
                                ' number conserving wavefunction')
            if hamil.conserve_number():
                raise TypeError('Number conserving hamiltonian passed to'
                                ' number non-conserving wavefunction')

        if isinstance(
                hamil,
                sparse_hamiltonian.SparseHamiltonian) and hamil.is_individual():

            final_wfn = self._evolve_individual_nbody(time, hamil, inplace)

        else:
            is_diag = ((hamil.quadratic() and hamil.diagonal()) or
                       hamil.diagonal_coulomb())
            if inplace and (not is_diag and not hamil.quadratic()):
                raise ValueError("Inplace is not implemented for this case")

            if self._conserve_spin and not self._conserve_number:
                work_wfn = self._copy_beta_inversion()
            elif inplace:
                work_wfn = self
            else:
                work_wfn = copy.deepcopy(self)

            if hamil.quadratic():
                if hamil.diagonal():

                    ihtdiag = -1.j * time * hamil.diag_values()
                    final_wfn = work_wfn._evolve_diagonal(ihtdiag, inplace)

                else:
                    transformation = hamil.calc_diag_transform()

                    permu, low, upp, work_wfn = work_wfn.transform(
                        transformation)

                    ci_trans = transformation @ permu

                    h1e = hamil.transform(ci_trans)

                    ihtdiag = -1.j * time * h1e.diagonal()
                    evolved_wfn = work_wfn._evolve_diagonal(ihtdiag,
                                                            inplace=True)

                    _, _, _, final_wfn = evolved_wfn.transform(
                        ci_trans.T.conj(), low, upp)

            elif hamil.diagonal_coulomb():

                diag, vij = hamil.iht(time)

                final_wfn = work_wfn._evolve_diagonal_coulomb_inplace(diag, vij)

            else:

                final_wfn = work_wfn.apply_generated_unitary(
                    time, 'taylor', hamil)

            if self._conserve_spin and not self._conserve_number:
                final_wfn = final_wfn._copy_beta_restore(
                    self._conserved['s_z'], self._norb, self._symmetry_map)

        if numpy.abs(hamil.e_0()) > 1.0e-15:
            final_wfn.scale(numpy.exp(-1.j * time * hamil.e_0()))

        return final_wfn

    def _evolve_diagonal(self, ithdiag: numpy.ndarray,
                         inplace: bool = False) -> 'Wavefunction':
        """Evolve a diagonal Hamiltonian on the wavefunction

        Args:
            ithdiag (numpy.ndarray): preprocessed diagonal array

            inplace (bool): whether the result will be stored in-place

        Returns:
            (Wavefunction): resulting wave function. If in-place is True,
                self is returned.
        """
        if inplace:
            wfn = self
        else:
            wfn = copy.deepcopy(self)

        for key, sector in self._civec.items():
            wfn._civec[key].coeff = sector.evolve_diagonal(ithdiag, inplace)

        return wfn

    def _evolve_diagonal_coulomb_inplace(self, diag: numpy.ndarray,
                                         vij: numpy.ndarray) -> 'Wavefunction':
        """Evolve a diagonal coulomb Hamiltonian on the wavefunction and store the
        result inplace.  (Not in-place version is no longer used and thus has been
        removed).

        Args:
            diag (numpy.ndarray): 1-body part of the Hamiltonian

            vij (numpy.ndarray): 2-body part of the Hamiltonian

        Returns:
            (Wavefunction): wavefunction after 1 time step of the time evolution.
                since this function is in-place, what is returned is always self.
        """

        for key, sector in self._civec.items():
            self._civec[key].coeff = sector.evolve_diagonal_coulomb(
                diag, vij, inplace=True)
        return self

    def expectationValue(
            self,
            ops: Union['fqe_operator.FqeOperator', 'hamiltonian.Hamiltonian'],
            brawfn: 'Wavefunction' = None) -> Union[complex, numpy.ndarray]:
        """Calculates expectation values given operators

        Args:
            ops (FqeOperator or Hamiltonian): operator for which the expectation value is \
                computed

            brawfn (Wavefunction): bra-side wave function for transition quantity (optional)

        Returns:
            (complex or numpy.ndarray): resulting expectation value or RDM
        """
        if isinstance(ops, fqe_operator.FqeOperator):
            if brawfn:
                return ops.contract(brawfn, self)
            return ops.contract(self, self)

        if isinstance(ops, str):
            if any(char.isdigit() for char in ops):
                ops = sparse_hamiltonian.SparseHamiltonian(ops)
            else:
                return self.rdm(ops, brawfn=brawfn)

        if not isinstance(ops, hamiltonian.Hamiltonian):
            raise TypeError('Expected an Fqe Hamiltonian or Operator' \
                            ' but recieved {}'.format(type(ops)))
        workwfn = self.apply(ops)

        if brawfn:
            return vdot(brawfn, workwfn)
        return vdot(self, workwfn)

    def _apply_individual_nbody(self,
                                hamil: 'sparse_hamiltonian.SparseHamiltonian',
                                base: 'Wavefunction' = None) -> 'Wavefunction':
        """
        Applies an individual n-body operator to the wave function self.

        Args:
            hamil (SparseHamiltonian): Sparse Hamiltonian to be applied to the wavefunction

            base (Wavefunction): the result will be accumulated to base

        Returns:
            (Wavefunction): resulting wavefunciton. When base is provided, base is returned
        """
        assert isinstance(hamil, sparse_hamiltonian.SparseHamiltonian)

        if hamil.nterms() > 1:
            raise ValueError(
                'Indivisual n-body code is called with multiple terms')

        [(coeff, alpha, beta)] = hamil.terms()
        daga = []
        dagb = []
        undaga = []
        undagb = []
        for oper in alpha:
            assert oper[0] < self._norb
            if oper[1] == 1:
                daga.append(oper[0])
            else:
                undaga.append(oper[0])
        for oper in beta:
            assert oper[0] < self._norb
            if oper[1] == 1:
                dagb.append(oper[0])
            else:
                undagb.append(oper[0])

        if len(daga) + len(dagb) != len(undaga) + len(undagb):
            raise ValueError('Number non-conserving operators specified')

        if base is None:
            out = self.empty_copy()
        else:
            out = base
        if len(daga) == len(undaga) and len(dagb) == len(undagb):
            for key in self._civec.keys():
                out._civec[key].apply_individual_nbody_accumulate(
                    coeff, self._civec[key], daga, undaga, dagb, undagb)
        else:
            ssectors = self._number_sectors()
            nsectors = out._number_sectors()
            for skey, nsector in nsectors.items():
                nsector.apply_individual_nbody_accumulate(
                    coeff, ssectors[skey], daga, undaga, dagb, undagb)
        return out

    def _evolve_individual_nbody(self,
                                 time: float,
                                 hamil: 'sparse_hamiltonian.SparseHamiltonian',
                                 inplace: bool = False) -> 'Wavefunction':
        """Apply up to 4-body individual operator.

        This routine assumes the Hamiltonian is normal ordered.

        Args:
            time (float): time for evolution

            hamil (SparseHamiltonian): Sparse Hamiltonian using which \
                the wavefunction is evolved

            inplace (bool): whether inplace memory will be used for \
                efficiency.

        Returns:
            (Wavefunction): resulting wavefunction.
        """
        if not isinstance(hamil, sparse_hamiltonian.SparseHamiltonian):
            raise TypeError('Expected a Hamiltonian Object but received' \
                            ' {}'.format(hamil))

        if hamil.nterms() > 2:
            raise ValueError(
                'Individual n-body code is called with multiple terms')

        # check if everything is paired
        if hamil.nterms() == 2:
            [(coeff0, alpha0, beta0), (coeff1, alpha1, beta1)] = hamil.terms()
            check = True
            for aop in alpha0:
                check &= (aop[0], aop[1] ^ 1) in alpha1
            for bop in beta0:
                check &= (bop[0], bop[1] ^ 1) in beta1

            if self._conserve_number:
                if not check:
                    raise ValueError(
                        'Operators in _evolve_individual_nbody is not Hermitian'
                    )
        elif hamil.nterms() == 1:
            [
                (coeff0, alpha0, beta0),
            ] = hamil.terms()
            check = True
            for aop in alpha0:
                check &= (aop[0], aop[1] ^ 1) in alpha0
            for bop in beta0:
                check &= (bop[0], bop[1] ^ 1) in beta0

            if self._conserve_number:
                if not check:
                    raise ValueError(
                        'Operators in _evolve_individual_nbody is not Hermitian'
                    )
            coeff0 *= 0.5
        else:
            assert hamil.nterms() == 0
            return self if inplace else copy.deepcopy(self)

        daga = []
        dagb = []
        undaga = []
        undagb = []

        for oper in alpha0:
            assert oper[0] < self._norb
            if oper[1] == 1:
                daga.append(oper[0])
            else:
                undaga.append(oper[0])

        for oper in beta0:
            assert oper[0] < self._norb
            if oper[1] == 1:
                dagb.append(oper[0])
            else:
                undagb.append(oper[0])

        if hamil.nterms() == 2:
            parity = (-1)**(len(alpha0) * len(beta0) + len(daga) * (len(daga)-1)//2 \
                            + len(dagb) * (len(dagb) - 1) // 2 \
                            + len(undaga) * (len(undaga) - 1) // 2 \
                            + len(undagb) * (len(undagb) - 1) // 2)
            if not numpy.abs(coeff0 - numpy.conj(coeff1) * parity) < 1.0e-8:
                raise ValueError(
                    'Coefficients in _evolve_individual_nbody is not Hermitian')


        if daga == undaga and dagb == undagb:
            out = self if inplace else copy.deepcopy(self)
            for _, sector in out._civec.items():
                sector.evolve_inplace_individual_nbody_trivial(
                    time, coeff0, daga, dagb)
        else:
            out = Wavefunction()
            out._norb = self._norb
            out._conserved = self._conserved
            out._symmetry_map = self._symmetry_map
            if len(daga) == len(undaga) and len(dagb) == len(undagb):
                for label, isec in self._civec.items():
                    osec = isec.evolve_individual_nbody_nontrivial(time, coeff0, daga, \
                                                                   undaga, dagb, undagb)
                    out._civec[label] = osec
            else:
                nsectors = self._number_sectors()
                for _, nsector in nsectors.items():
                    osector = nsector.evolve_individual_nbody(time, coeff0, daga, \
                                                              undaga, dagb, undagb)
                    for (nalpha, nbeta), osec in osector.sectors().items():
                        out._civec[(nalpha + nbeta, nalpha - nbeta)] = osec
        return out

    def _apply_few_nbody(self, hamil: 'sparse_hamiltonian.SparseHamiltonian'
                        ) -> 'Wavefunction':
        """ Applies SparseHamiltonian by looping over all of the operators.
        Useful when the operator is extremely sparse

        Args:
            hamil (SparseHamiltonian): Sparse Hamiltonian to be applied to the wavefunction

        Returns:
            (Wavefunction): resulting wavefunciton
        """
        out = None
        for oper in hamil.terms_hamiltonian():
            if out is None:
                out = self._apply_individual_nbody(oper)
            else:
                out = self._apply_individual_nbody(oper, base=out)

        if out is None:
            out = copy.deepcopy(self)

        if numpy.abs(hamil.e_0()) > 1.e-15:
            out.ax_plus_y(hamil.e_0(), self)

        return out

    @wrap_rdm
    def rdm(self, string: str, brawfn: Optional['Wavefunction'] = None
           ) -> Union[complex, numpy.ndarray]:
        """ Returns rank-1 RDM. The operator order is specified by string.
        Note that, if the entire RDM is requested for N-broken wave function,
        this returns a packed format.

        Args:
            string (str): character strings that specify the quantity to be computed

            brawfn (Wavefunction): bra-side wave function for transition RDM (optional)

        Returns:
            Union[complex, numpy.ndarray]: Resulting RDM in numpy.ndarray or \
                an RDM element in complex
        """
        rank = len(string.split()) // 2
        if any(char.isdigit() for char in string):
            result = self.apply(sparse_hamiltonian.SparseHamiltonian(string))
            if brawfn is None:
                return vdot(self, result)
            return vdot(brawfn, result)

        fqe_ops_utils.validate_rdm_string(string)
        rdm = list(self._compute_rdm(rank, brawfn))
        return wick(string, rdm, self._conserve_spin)

    def _compute_rdm(self, rank: int, brawfn: 'Wavefunction' = None
                    ) -> Tuple[numpy.ndarray, ...]:
        """Internal function that computes RDM up to rank = rank

        Args:
            rank (int): the rank up to which RDMs are computed

            brawfn (Wavefunction): bra wavefunction for transition RDMs (optional)

        Returns:
            Tuple[numpy.ndarray, ...]: tuple of RDMs
        """
        assert rank > 0
        assert rank < 5

        out: List[Any] = [None, None, None, None]
        tmp: List[Any] = [None, None, None, None]

        if self._conserve_spin:
            for key, sector in self._civec.items():
                assert brawfn is None or key in brawfn.sectors()
                bra = None if brawfn is None else brawfn._civec[key]
                if rank == 1:
                    (tmp[0],) = sector.rdm1(bra)
                elif rank == 2:
                    (tmp[0], tmp[1]) = sector.rdm12(bra)
                elif rank == 3:
                    (tmp[0], tmp[1], tmp[2]) = sector.rdm123(bra)
                elif rank == 4:
                    (tmp[0], tmp[1], tmp[2], tmp[3]) = sector.rdm1234(bra)
                for i in range(4):
                    if tmp[i] is not None:
                        out[i] = tmp[i] if out[i] is None else out[i] + tmp[i]
            out2: List[numpy.ndarray] = []
            for i in range(4):
                if out[i] is not None:
                    out2.append(out[i])
            return tuple(out2)

        numbersectors = self._number_sectors()
        for nkey, dataset in numbersectors.items():
            assert brawfn is None or nkey in brawfn._number_sectors().keys()
            nbra = None if brawfn is None else brawfn._number_sectors()[nkey]
            if rank == 1:
                (tmp[0],) = dataset.rdm1(nbra)
            elif rank == 2:
                (tmp[0], tmp[1]) = dataset.rdm12(nbra)
            elif rank == 3:
                (tmp[0], tmp[1], tmp[2]) = dataset.rdm123(nbra)
            elif rank == 4:
                (tmp[0], tmp[1], tmp[2], tmp[3]) = dataset.rdm1234(nbra)
            for i in range(4):
                if tmp[i] is not None:
                    out[i] = tmp[i] if out[i] is None else out[i] + tmp[i]
        out2 = []
        for i in range(4):
            if out[i] is not None:
                out2.append(out[i])
        return tuple(out2)
