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

"""Wavefunction class for accessing and manipulation of the state of interest
"""

import copy
import os
import math
from typing import (Callable,
                    Dict,
                    Generator,
                    KeysView,
                    List,
                    Optional,
                    Tuple,
                    Type,
                    Union)

from scipy.special import factorial, jv
from openfermion import FermionOperator
from openfermion.utils import is_hermitian, hermitian_conjugated

import numpy
from numpy import linalg
import scipy
from scipy import special

import fqe
from fqe.fqe_decorators import wrap_apply, wrap_apply_generated_unitary
from fqe.fqe_decorators import wrap_time_evolve
from fqe.fqe_data import FqeData
from fqe.fqe_data_set import FqeDataSet
from fqe.fci_graph import FciGraph
from fqe.fci_graph_set import FciGraphSet
from fqe.util import alpha_beta_electrons, init_bitstring_groundstate
from fqe.util import sort_configuration_keys
from fqe.util import configuration_key_union
from fqe.openfermion_utils import new_wfn_from_ops, mutate_config
from fqe.openfermion_utils import mutate_config, classify_symmetries
from fqe.openfermion_utils import split_openfermion_tensor
from fqe.openfermion_utils import generate_one_particle_matrix
from fqe.openfermion_utils import generate_two_particle_matrix
from fqe.hamiltonians import hamiltonian, sparse_hamiltonian
from fqe.hamiltonians import gso_hamiltonian, sso_hamiltonian
from fqe.hamiltonians import general_hamiltonian
from fqe.bitstring import integer_index, count_bits
from fqe.fqe_ops import fqe_operator, fqe_ops_utils
from fqe.wick import wick
#from fqe._fqe_control import get_hamiltonian_from_ops


#TODO after unittesting is implemented, change the keys of _civec such that the key is (nalpha, nbeta).
# this is the only place where (N, Sz) key is used, which is terrible


class Wavefunction:
    """Wavefunction is the central object for manipulaion in the
    OpenFermion-FQE.
    """


    def __init__(self, param: Optional[List[List[int]]] = None,
                 broken: Optional[Union[List[str], str]] = None) -> None:
        """
        Args:
            param (list[list[n, ms, norb]]) - the constructor accepts a list of
              parameter lists.  The parameter lists are comprised of

              p[0] (integer) - number of particles
              p[1] (integer) - z component of spin angular momentum
              p[2] (integer) - number of spatial orbitals

            broken (str) - pass in the symmetries that should be preserved by
                the wavefunction.

        Member Variables:
            _conserve_spin (Bool) - When this flag is true, the wavefunction
                will maintain a constant m_s
            _conserve_number (Bool) - When this flag is true, the wavefunction
                will maintain a constant nele
            _civec (dict[(int, int)] -> FqeData) - This is a dictionary for
                FqeData objects.  The key is a tuple defined by the number of
                electrons and the spin projection of the system.
        """

        self._conserve_spin: bool = True if (broken is None or not 'spin' in broken) else False
        self._conserve_number: bool = True if (broken is None or not 'number' in broken) else False
        self._norb: int = 0
        self._civec: Dict[Tuple[int, int], 'FqeData'] = {}

        if not self._conserve_spin and not self._conserve_number:
            raise TypeError('Number and spin non-conserving waveunfction is' \
                            ' the complete Fock space.')

        if param:
            for i in param:
                self._norb = i[2]
                nalpha, nbeta = alpha_beta_electrons(i[0], i[1])
                self._civec[(i[0], i[1])] = FqeData(nalpha, nbeta, self._norb)

            for i in param:
                if i[2] != self._norb:
                    raise ValueError('Number of orbitals is not consistent')



    def __add__(self, other: 'Wavefunction') -> 'Wavefunction':
        """Intrinsic addition function to combine two wavefunctions.  This acts
        to iterate through the wavefunctions, combine coefficients of
        configurations they have in common and add configurations that are
        unique to each one.  The values are all combined into a new
        wavefunction object

        Args:
            other (wavefunction.Wavefunction) - the second wavefunction to
                add with the local wavefunction

        Returns:
            wfn (wavefunction.Wavefunction) - a new wavefunction with the
                values set by adding together values
        """
        out = copy.deepcopy(self)
        out.ax_plus_y(1.0, other)
        return out


    def __iadd__(self, wfn: 'Wavefunction') -> 'Wavefunction':
        self.ax_plus_y(1.0, wfn)
        return self


    def __mul__(self, sval: complex) -> None:
        """Multiply will scale the whole wavefunction by the value passed.

        Args:
            sval (complex) - a number to multiply the wavefunction by

        Returns:
            nothing -mutates in place
        """
        self.scale(sval)


    def __sub__(self, other: 'Wavefunction') -> 'Wavefunction':
        """Intrinsic subtraction function to combine two wavefunctions.  This
        acts to iterate through the wavefunctions, combine coefficients of
        configurations they have in common and include configurations that are
        unique to each one. The values are all combined into a new
        wavefunction object

        Args:
            wfn (wavefunction.Wavefunction) - the second wavefunction that will
                be subtracted from the first wavefunction

        Returns:
            wfn (wavefunction.Wavefunction) - a new wavefunction with the
                values set by subtracting the wfn from the first
        """
        out = copy.deepcopy(self)
        out.ax_plus_y(-1.0, other)
        return out


    def __getitem__(self, key):
        """
        """
        astr, bstr = key[0], key[1]
        sector = (count_bits(astr) + count_bits(bstr),
                  count_bits(astr) - count_bits(bstr))
        inda = self._civec[sector].index_alpha(astr)
        indb = self._civec[sector].index_alpha(astr)
        return self._civec[sector].coeff[inda, indb]


    def __setitem__(self, key, value):
        """
        """
        astr, bstr = key[0], key[1]
        sector = (count_bits(astr) + count_bits(bstr),
                  count_bits(astr) - count_bits(bstr))
        inda = self._civec[sector].index_alpha(astr)
        indb = self._civec[sector].index_alpha(astr)
        self._civec[sector].coeff[inda, indb] = value


    def ax_plus_y(self, sval: complex, wfn: 'Wavefunction') -> None:
        if self._civec.keys() != wfn._civec.keys():
            raise ValueError('inconsistent sectors in Wavefunction.ax_plus_y')

        for sector in self._civec.keys():
            self._civec[sector].ax_plus_y(sval, wfn._civec[sector])


    def sectors(self) -> KeysView[Tuple[int, int]]:
        """Return a list of the configuration keys in the wavefunction
        """
        return self._civec.keys()


    def conserve_number(self) -> bool:
        """
        """
        return self._conserve_number


    def conserve_spin(self) -> bool:
        """
        """
        return self._conserve_spin


    def norb(self) -> int:
        """Return the number of orbitals
        """
        return self._norb


    def norm(self) -> float:
        """Calculate the norm of the wavefuntion
        """
        normall = 0.0
        for sector in self._civec.values():
            normall += sector.norm() ** 2
        normall = math.sqrt(normall)
        return normall


    def normalize(self) -> None:
        """Generte the wavefunction norm and then scale each element by that
        value.
        """
        self.scale(1.0/self.norm())


    def number_sectors(self) -> Dict[int, FqeDataSet]:
        """An internal utility function that groups FqeData into
        a set of FqeDataSet that corresponds to the same number of electrons.
        It checks spin completeness and raises an exception if the wave function space
        is not spin complete
        """
        numbers = set()
        norb = self.norb()
        for key in self._civec:
            numbers.add(key[0])
        numbersectors = {}
        for nele in numbers:
            # generate all possible sz
            maxalpha = min(norb, nele)
            minalpha = nele - maxalpha
            check = True
            sectors = {}
            for nalpha in range(minalpha, maxalpha+1):
                nbeta = nele - nalpha
                check &= (nele, nalpha-nbeta) in self._civec.keys()
            if not check:
                raise Exception('Wave function does not have all of the spin sectors')
            for nalpha in range(minalpha, maxalpha+1):
                nbeta = nele - nalpha
                sectors[(nalpha,nbeta)] = self._civec[(nele, nalpha-nbeta)]
            dataset = FqeDataSet(nele, norb, sectors)
            numbersectors[nele] = dataset
        return numbersectors


    @wrap_apply
    def apply(self, hamil: 'Hamiltonian') -> 'Wavefunction':
        """
        """
        if not isinstance(hamil, hamiltonian.Hamiltonian):
            raise TypeError('Expected a Hamiltonian Object but received' \
                            ' []'.format(type(hamil)))

        work_wfn = copy.deepcopy(self)

        if isinstance(hamil, sparse_hamiltonian.SparseHamiltonian):

            return work_wfn.apply_few_nbody(hamil)

        else:

            return work_wfn.apply_array(hamil.tensors())


    def apply_array(self, array: Tuple[numpy.ndarray]) -> 'Wavefunction':
        """Return a wavefunction subject to application of the numpy array as

            h[i, j]a_i^+ a_j|Psi>
            h[i, j, k, l]a_i^+(rho) a_j^+(eta) a_k(rho) a_l(eta)|Psi>

        Arg:
            array (numpy.array) - numpy array

        Returns:
            newwfn (Wavvefunction) - a new intialized wavefunction object

        """

        if self._conserve_spin:
            assert array[0].shape[0] == self._norb or array[0].shape[0] == self._norb * 2
            out = copy.deepcopy(self)
            for _, sector in out._civec.items():
                sector.apply_inplace(array)
            return out
        else:
            assert array[0].shape[0] == self._norb * 2
            out = copy.deepcopy(self)
            nsectors = out.number_sectors()
            for _, sector in nsectors.items():
                sector.apply_inplace(array)
            return out


    def compute_rdm(self, rank: int, brawfn: 'Wavefunction' = None):
        """compute RDM up to rank = rank
        """
        if rank < 1 or rank > 4:
            raise ValueError('invalid rank in Wavefunction.compute_rdm')

        if self._conserve_spin:
            out = [None, None, None, None]
            tmp = [None, None, None, None]
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
            out2 = []
            for i in range(4):
                if out[i] is not None:
                    out2.append(out[i])
            return tuple(out2)
        else:
            numbersectors = self.number_sectors()
            brasectors = None if brawfn is None else brawfn.number_sectors()
            out = [None, None, None, None]
            tmp = [None, None, None, None]
            for key, dataset in numbersectors.items():
                assert brasectors is None or key in brasectors.keys()
                bra = None if brawfn is None else brasectors[key]
                if rank == 1:
                    (tmp[0],) = dataset.rdm1(bra)
                elif rank == 2:
                    (tmp[0], tmp[1]) = dataset.rdm12(bra)
                elif rank == 3:
                    (tmp[0], tmp[1], tmp[2]) = dataset.rdm123(bra)
                elif rank == 4:
                    (tmp[0], tmp[1], tmp[2], tmp[3]) = dataset.rdm1234(bra)
                for i in range(4):
                    if tmp[i] is not None:
                        out[i] = tmp[i] if out[i] is None else out[i] + tmp[i]
            out2 = []
            for i in range(4):
                if out[i] is not None:
                    out2.append(out[i])
            return tuple(out2)


    @wrap_apply_generated_unitary
    def apply_generated_unitary(self,
                                time: float,
                                algo: str,
                                hamil,
                                accuracy: Optional[float] = None,
                                expansion: int = 30,
                                spec_lim = None,
                                safety = 0.025) -> 'Wavefunction':
        """Perform the exponentiation of fermionic algebras to the
        wavefunction according the method and accuracy.

        Args:
            time_final (float) - the final time value to evolve to
            algo (string) - which algorithm should we use
            accuracy (double) - the accuracy to which the system should be evovled

        Returns:
            newwfn (Wavefunction) - a new intialized wavefunction object
        """

        if not isinstance(hamil, hamiltonian.Hamiltonian):
            raise TypeError('Expected a Hamiltonian Object but received' \
                            ' []'.format(type(hamil)))


        algo_avail = [
            'taylor',
            'chebyshev'
            ]

        if algo not in algo_avail:
            raise ValueError('{} is not availible'.format(algo))

        max_expansion = min(30, expansion)

        if isinstance(hamil, sparse_hamiltonian.SparseHamiltonian):

            hamil_time = hamil.generated_unitary(time)

            if algo == 'taylor':

                if accuracy:
                    time_evol = copy.deepcopy(self)
                    work = copy.deepcopy(self)

                    for order in range(1, max_expansion):
                        work = work.apply(hamil_time)
                        delta = copy.deepcopy(work)
                        delta.scale(1.0 / factorial(order))
                        if delta.max_element() < accuracy:
                            time_evol += delta
                            return time_evol

                        time_evol += delta

                    print('Accuracy not achieved'\
                          ' after {} terms'.format(max_expansion))

                    return time_evol

                else:
                    time_evol = copy.deepcopy(self)
                    work = copy.deepcopy(self)

                    for order in range(1, max_expansion):
                        work = work.apply(hamil_time)
                        time_evol.ax_plus_y(1.0 / factorial(order), work)

                    return time_evol


            if algo == 'chebyshev':
                if accuracy is None:
                    accuracy = 1.e-10

                e_o = self.expectationValue(hamil)
                if spec_lim:
                    spec_min = 2.0*spec_lim[0]
                    spec_max = 2.0*spec_lim[1]
                else:
                    raise ValueError('Spectral width can\'t be estimated.' \
                                     ' Provide upper and lower limits.')

                t_minus1 = copy.deepcopy(self)
                t_n = t_minus1.apply(hamil_time)
                t_x2 = t_n.apply(hamil_time)
                t_x2.scale(2.0)
                t_plus1 = t_x2 - t_minus1
                time_evol = copy.deepcopy(self)
                time_evol.scale(jv(0, 1.j))

                for order in range(1, max_expansion):
                    chebyshev_con = copy.deepcopy(t_n)
                    scale_fac = 2.0*(1.j**order)*jv(order, -1.j)
                    chebyshev_con.scale(scale_fac)
                    if numpy.abs(fqe.vdot(chebyshev_con, self)) < accuracy:
                        time_evol += chebyshev_con
                        return time_evol

                    time_evol += chebyshev_con
                    t_minus1 = copy.deepcopy(t_n)
                    t_n = copy.deepcopy(t_plus1)
                    t_x2 = t_n.apply(hamil_time)
                    t_x2.scale(2.0)
                    t_plus1 = t_x2 - t_minus1

                return time_evol

        else:

            ham_arrays = hamil.iht(time, full=True)

            if algo == 'taylor':

                if accuracy:
                    time_evol = copy.deepcopy(self)
                    work = copy.deepcopy(self)

                    for order in range(1, max_expansion):
                        work = work.apply_array(ham_arrays)
                        delta = copy.deepcopy(work)
                        delta.scale(1.0 / factorial(order))
                        if delta.max_element() < accuracy:
                            time_evol += delta
                            return time_evol

                        time_evol += delta

                    print('Accuracy not achieved'\
                          ' after {} terms'.format(max_expansion))

                    return time_evol

                else:
                    time_evol = copy.deepcopy(self)
                    work = copy.deepcopy(self)

                    for order in range(1, max_expansion):
                        work = work.apply_array(ham_arrays)
                        time_evol.ax_plus_y(1.0 / factorial(order), work)

                    return time_evol


            if algo == 'chebyshev':
                if accuracy is None:
                    accuracy = 1.e-10

                e_o = self.expectationValue(hamil)
                if spec_lim:
                    spec_min = 2.0*spec_lim[0]
                    spec_max = 2.0*spec_lim[1]
                else:
                    raise ValueError('Spectral width can\'t be estimated.' \
                                     ' Provide upper and lower limits.')

                t_minus1 = copy.deepcopy(self)
                t_n = t_minus1.apply_array(ham_arrays)
                t_x2 = t_n.apply_array(ham_arrays)
                t_x2.scale(2.0)
                t_plus1 = t_x2 - t_minus1
                time_evol = copy.deepcopy(self)
                time_evol.scale(jv(0, 1.j))

                for order in range(1, max_expansion):
                    chebyshev_con = copy.deepcopy(t_n)
                    scale_fac = 2.0*(1.j**order)*jv(order, -1.j)
                    chebyshev_con.scale(scale_fac)
                    if numpy.abs(fqe.vdot(chebyshev_con, self)) < accuracy:
                        time_evol += chebyshev_con
                        return time_evol

                    time_evol += chebyshev_con
                    t_minus1 = copy.deepcopy(t_n)
                    t_n = copy.deepcopy(t_plus1)
                    t_x2 = t_n.apply_array(ham_arrays)
                    t_x2.scale(2.0)
                    t_plus1 = t_x2 - t_minus1

                return time_evol


    def get_coeff(self, key: Tuple[int, int]) -> numpy.ndarray:
        """Retrieve a vector from a configuration in the wavefunction

        Args:
            key (int, int) - a key identifying the configuration to access
            vec (int) - an integer indicating which state should be returned

        Returns:
            numpy.array(dtype=numpy.complex128)
        """
        return self._civec[key].coeff


    def max_element(self) -> complex:
        """Return the largest magnitude value in the wavefunction
        """
        maxval = 0.0
        for config in self._civec.values():
            configmax = max(config.coeff.max(), config.coeff.min(), key=abs)
            maxval = max(configmax, maxval)

        return maxval


    def print_wfn(self, threshold: float = 0.001, fmt: str = 'str') -> None:
        """Print occupations and coefficients to the screen.

        Args:
            threshhold (float) - only print CI vector values such that
              |c| > threshold.
            fmt (string) - formats print according to argument
            states (int of list[int]) - an index or indexes indicating which
                states to print.
        """
        def _print_format(fmt: str) -> Callable[[int, int], str]:
            """ Select the function which will perform formatted printing
            """
            if fmt == 'occ':
                def _occupation_format(astring: int, bstring: int):
                    """occ - Prints a string indicating occupation state of each spatial
                    orbital.  A doubly occupied orbital will be indicated with "2",
                    a singly occupied orbital will get "a" or "b" depending on the
                    spin state.  An empy orbital will be ".".
                    """
                    occstr = ['.' for _ in range(max(astring.bit_length(),
                                                     bstring.bit_length()))]
                    docc = astring & bstring

                    def build_occ_value(bstr: int, char: str,
                                        occstr: List[str]):
                        """Fill a list with a character corresponding to the
                        location of '1' bits in bitstring

                        Args:
                            bstr (int) - a bitstring to examine
                            char (str) - a string to put into the list
                            occstr (list) - a list to store the value
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

            def _string_format(astring: int, bstring: int) -> str:
                """ string - Prints a binary string representing indicating
                which orbital creation operators are acting on the vacuum with
                a 1.  The position in the string indicates the index of the
                orbital.  The beta string is shown acting on the vacuum first.
                """
                fmt_a = bin(1 << self._norb | astring)
                fmt_b = bin(1 << self._norb | bstring)
                return "a'{}'b'{}'".format(fmt_a[3:], fmt_b[3:])

            return _string_format


        print_format = _print_format(fmt)

        config_in_order = sort_configuration_keys(self.sectors())
        for key in config_in_order:
            sector = self._civec[key]
            sector.print_sector(print_format=print_format, threshold=threshold)


    def read(self, filename: str, path: str = os.getcwd()) -> None:
        """Initialize a wavefunction from a binary file.

        Args:
            filename (str) - the name of the file to write the wavefunction to
            path (str) - the path to save the file.  If no path is given then
                it is saved in the current working directory.
        """
        with open(path + '/' + filename, 'r+b') as wfnfile:
            wfn_data = numpy.load(wfnfile, allow_pickle=True)

        # TODO how about _conserve_spin?
        norb = wfn_data[0]
        for sector in wfn_data[1:]:
            self._civec[(sector[0][0], sector[0][1])] = sector[1]


    def save(self, filename: str, path: str = os.getcwd()) -> None:
        """Save the wavefunction into path/filename.

        Args:
            filename (str) - the name of the file to write the wavefunction to
            path (str) - the path to save the file.  If no path is given then
                it is saved in the current working directory.
        """
        wfn_data = [self._norb]

        for key in self._civec:
            wfn_data.append([key, self._civec[key].coeff])

        with open(path + '/' + filename, 'w+b') as wfnfile:
            numpy.save(wfnfile, wfn_data, allow_pickle=True)


    def scale(self, sval: complex) -> None:
        """ Scale each configuration space by the value sval

        Args:
            sval (complex) - value to scale by
        """

        sval = complex(sval)

        for sector in self._civec.values():
            sector.scale(sval)


    def set_wfn(self, strategy: str = 'ones',
                raw_data: Optional[Dict[Tuple[int, int], numpy.ndarray]] = None) -> None:
        """Set the values of the ciwfn based on an argument or data to
        initalize from.

        Args:
            strategy (string) - an option controlling how the values are set
            raw_data (numpy.array(dtype=numpy.complex128)) - data to inject into
                the configuration
        """
        if strategy == 'from_data' and not raw_data:
            raise ValueError('No data provided for set_wfn')

        if strategy == 'from_data':
            for key, data in raw_data.items():
                self._civec[key].coeff = data
        else:
            for sector in self._civec.values():
                sector.set_wfn(strategy=strategy)


    def transform(self, rotation) -> Tuple[numpy.ndarray, 'WaveFunction']:
        """Transform the wavefunction using the orbtial rotation matrix and
        return the new wavefunction and the permutation matrix for the unitary
        transformation. This is an internal code, so performs minimal checking
        """
        norb = self._norb

        def process_matrix(rotmat: numpy.ndarray) -> numpy.ndarray:
            ndim = rotmat.shape[0]
            assert rotmat.shape[1] == ndim
            tmat = rotmat.transpose().conjugate()
            (p, l, u) = scipy.linalg.lu(tmat)
            for irow in range(ndim):
                for icol in range(irow+1, ndim):
                    u[irow, icol] /= u[irow, irow]
                l[irow, irow], u[irow, irow] = u[irow, irow], l[irow, irow]
                for icol in range(irow):
                    l[irow, icol] *= l[icol, icol]

            lt = l.transpose().conjugate()
            ut = u.transpose().conjugate()

            unitmat = numpy.identity(ndim)
            output = scipy.linalg.solve_triangular(lt, unitmat)

            for icol in range(ndim):
                for irow in range(icol+1, ndim):
                    output[irow, icol] -= ut[irow, icol]
                output[icol, icol] -= 1.0
            return (p, output)

        current = copy.deepcopy(self)

        if not self._conserve_spin:
            (p, output) = process_matrix(rotation)
            for icol in range(norb*2):
                work = numpy.zeros_like(rotation)
                work[:,icol] = output[:,icol]
                onefwfn = current.apply_array((work,))
                current.ax_plus_y(1.0, onefwfn)
        else:
            if rotation.shape[0] == norb:
                (p, output) = process_matrix(rotation)
                for icol in range(norb):
                    work = numpy.zeros_like(rotation)
                    work[:,icol] = output[:,icol]
                    onefwfn = current.apply_array((work,))
                    twofwfn = onefwfn.apply_array((work,))
                    current.ax_plus_y(1.0-0.5*work[icol,icol], onefwfn)
                    current.ax_plus_y(0.5, twofwfn)
            elif rotation.shape[0] == norb*2:
                assert numpy.std(rotation[:norb,norb:]) + numpy.std(rotation[norb:,:norb]) < 1.0e-8
                (p1, output1) = process_matrix(rotation[:norb, :norb])
                (p2, output2) = process_matrix(rotation[norb:, norb:])
                for icol in range(norb):
                    work = numpy.zeros_like(rotation)
                    work[:norb,icol] = output1[:,icol]
                    onefwfn = current.apply_array((work,))
                    current.ax_plus_y(1.0, onefwfn)
                for icol in range(norb):
                    work = numpy.zeros_like(rotation)
                    work[norb:,icol+norb] = output2[:,icol]
                    onefwfn = current.apply_array((work,))
                    current.ax_plus_y(1.0, onefwfn)
                p = numpy.zeros_like(rotation)
                p[:norb,:norb] = p1[:,:]
                p[norb:,norb:] = p2[:,:]

        return p, current


    @wrap_time_evolve
    def time_evolve(self, time: float, hamil) -> 'Wavefunction':
        """Perform time evolution of the wavefunction given Fermion Operators
        either as raw operations or wrapped up in a Hamiltonian.

        Args:
            ops (FermionOperators) - FermionOperators which are to be time evolved.
            time (float) - the duration by which to evolve the operators

        Returns:
            Wavefunction - a wavefunction object that has been time evolved.
        """
        if not isinstance(hamil, hamiltonian.Hamiltonian):
            raise TypeError('Expected Hamiltonian Object but received {}'.format(type(hamil)))

        if not self._conserve_number or not hamil.conserve_number:
            if self._conserve_number:
                raise TypeError('Number non-conserving hamiltonian passed to' \
                                ' number conserving wavefunction')
            if hamil.conserve_number:
                raise TypeError('Number conserving hamiltonian passed to' \
                                ' number non-conserving wavefunction')

        work_wfn = copy.deepcopy(self)


        if hamil.quadratic():
            if hamil.diagonal():

                ihtdiag = hamil.iht(time, full=False)
                final_wfn = work_wfn.evolve_diagonal(ihtdiag[0])

            else:
                transformation = hamil.calc_diag_transform()

                permu, work_wfn = work_wfn.transform(transformation)
                ci_trans = transformation @ permu

                h1e = hamil.transform(ci_trans)

                ihtdiag = -1.j*time*h1e.diagonal()
                evolved_wfn = work_wfn.evolve_diagonal(ihtdiag)

                permutation, final_wfn = evolved_wfn.transform(ci_trans.conj().T)

                if not numpy.allclose(permutation, numpy.eye(hamil.dim())):
                    raise ValueError('Tranformation to original CI basis' \
                                     'is performing permutation.')
        elif hamil.diagonal_coulomb():

            diag, vij = hamil.iht(time)

            final_wfn = work_wfn.evolve_diagonal_coulomb(diag, vij)

        elif isinstance(hamil, sparse_hamiltonian.SparseHamiltonian):

            final_wfn = work_wfn.evolve_individual_nbody(time, hamil)

        else:

            final_wfn = work_wfn.apply_generated_unitary(time, 'taylor', hamil)

        return final_wfn


    def evolve_diagonal(self, ithdiag) -> None:
        """Evolve a diagonal Hamiltonian on the wavefunction
        """
        wfn = copy.deepcopy(self)

        for key, sector in self._civec.items():
            wfn._civec[key].coeff = sector.apply_diagonal_unitary_array(ithdiag)

        return wfn


    def evolve_diagonal_coulomb(self, diag, vij) -> None:
        """Evolve a diagonal coulomb Hamiltonian on the wavefunction
        """

        wfn = copy.deepcopy(self)
        for key, sector in self._civec.items():
            wfn._civec[key].coeff = sector.diagonal_coulomb(diag, vij)

        return wfn


    def expectationValue(self, ops, brawfn = None) -> complex:
        """
        """
        if isinstance(ops, fqe_operator.FqeOperator):
            if brawfn:
                return ops.contract(brawfn, self)
            return ops.contract(self, self)


        if not isinstance(ops, hamiltonian.Hamiltonian):
            raise TypeError('Expected an Fqe Hamiltonian or Operator' \
                            ' but recieved {}'.format(type(ops)))


        if isinstance(ops, gso_hamiltonian.GSOHamiltonian) or \
            isinstance(ops, sso_hamiltonian.SSOHamiltonian) or \
            isinstance(ops, general_hamiltonian.General):

            work = copy.deepcopy(self)
            workwfn = work.apply(ops)

            if brawfn:
                return fqe.vdot(brawfn, workwfn)
            else:
                return fqe.vdot(self, workwfn)

        expval = 0. + 0.j
        rdm_lib = self.compute_rdm(ops.rank() // 2, brawfn)

        for rank in range(2, ops.rank() + 1, 2):
            axes = list(range(rank))
            expval += numpy.tensordot(ops.tensor(rank), rdm_lib[(rank - 1) // 2], axes=(axes, axes))
        return expval


    def apply_individual_nbody(self, op: 'SparseHamiltonian') -> 'Wavefunction':
        """
        Applies an individual n-body operator to the wave function self.
        """
        if op.nterms() > 1:
            raise Exception('Indivisual n-body code is called with multiple terms')
        [(coeff, alpha, beta)] = op.terms()
        daga = []
        dagb = []
        undaga = []
        undagb = []
        for oper in alpha:
            assert oper[0] < self._norb
            if oper[1] == 1: daga.append(oper[0])
            else:          undaga.append(oper[0])
        for oper in beta:
            assert oper[0] < self._norb
            if oper[1] == 1: dagb.append(oper[0])
            else:          undagb.append(oper[0])

        if len(daga)+len(dagb) != len(undaga)+len(undagb):
            raise Exception('Number non-conserving operators specified')

        out = copy.deepcopy(self)
        if len(daga) == len(undaga) and len(dagb) == len(undagb):
            for key, sector in self._civec.items():
                out._civec[key] = sector.apply_individual_nbody(coeff, daga, undaga, dagb, undagb)
        else:
            nsectors = out.number_sectors()
            for _, sector in nsectors.items():
                sector.apply_inplace_individual_nbody(coeff, daga, undaga, dagb, undagb)
        return out


    def evolve_individual_nbody(self, time: float, hamil: 'SparseHamiltonian') -> 'Wavefunction':
        """Apply up to 4-body individual operator
        """

        if not isinstance(hamil, sparse_hamiltonian.SparseHamiltonian):
            raise TypeError('Expected a Hamiltonian Object but received' \
                            ' []'.format(hamil))

        if hamil.nterms() > 2:
            raise Exception('Indivisual n-body code is called with multiple terms')

        # check if everything is paired
        if hamil.nterms() == 2:
            [(coeff0, alpha0, beta0), (coeff1, alpha1, beta1)] = hamil.terms()
            check = True
            for aop in alpha0:
                check &= [aop[0], aop[1] ^ 1] in alpha1
            for bop in beta0:
                check &= [bop[0], bop[1] ^ 1] in beta1
            if not check:
                raise Exception('Operators in evolve_individual_nbody is not Hermitian')
        else:
            [(coeff0, alpha0, beta0), ] = hamil.terms()
            check = True
            for aop in alpha0:
                check &= [aop[0], aop[1] ^ 1] in alpha0
            for bop in beta0:
                check &= [bop[0], bop[1] ^ 1] in beta0
            if not check:
                raise Exception('Operators in evolve_individual_nbody is not Hermitian')
            coeff0 *= 0.5

        daga = []
        dagb = []
        undaga = []
        undagb = []
        for oper in alpha0:
            assert oper[0] < self._norb
            if oper[1] == 1: daga.append(oper[0])
            else:          undaga.append(oper[0])
        for oper in beta0:
            assert oper[0] < self._norb
            if oper[1] == 1: dagb.append(oper[0])
            else:          undagb.append(oper[0])

        if hamil.nterms() == 2:
            parity = (-1)**(len(alpha0)*len(beta0) + len(daga)*(len(daga)-1)//2 + len(dagb)*(len(dagb)-1)//2 \
                            + len(undaga)*(len(undaga)-1)//2 + len(undagb)*(len(undagb)-1)//2)
            if not numpy.abs(coeff0 - numpy.conj(coeff1)*parity) < 1.0e-8:
                raise Exception('Coefficients in evolve_individual_nbody is not Hermitian')

        out = copy.deepcopy(self)
        if daga == undaga and dagb == undagb:
            for _, sector in out._civec.items():
                sector.evolve_inplace_individual_nbody_trivial(time, coeff0, daga, dagb)
        else:
            if len(daga) == len(undaga) and len(dagb) == len(undagb):
                for _, sector in out._civec.items():
                    sector.evolve_inplace_individual_nbody_nontrivial(time, coeff0, daga, undaga, dagb, undagb)
            else:
                nsectors = out.number_sectors()
                for _, sector in nsectors.items():
                    sector.evolve_inplace_individual_nbody(time, coeff0, daga, undaga, dagb, undagb)
        return out


    def evovle_few_nbody(self, op: 'SparseHamiltonian') -> 'Wavefunction':
        """ Applies SparseHamiltonian by looping over all of the operators.
        Useful when the operator is extremely sparse
        """
        out = copy.deepcopy(self)
        out.set_wfn(strategy="zero")
        for oper in op.terms_hamiltonian():
            out += self.apply_individual_nbody(oper)
        return out


    def apply_few_nbody(self, op: 'SparseHamiltonian') -> 'Wavefunction':
        """ Applies SparseHamiltonian by looping over all of the operators.
        Useful when the operator is extremely sparse
        """
        out = copy.deepcopy(self)
        out.set_wfn(strategy="zero")
        for oper in op.terms_hamiltonian():
            out += self.apply_individual_nbody(oper)
        return out


    def rdm1(self, string):
        target = 1
        work = fqe_ops_utils.validate_rdm_string(string, target)

        if work == 'element':
            result = self.apply(sparse_hamiltonian.SparseHamiltonian(string))
            return vdot(self, result)
        elif work == 'tensor':
            rdm = self.compute_rdm(target)
            return wick(string, [rdm], self._conserve_spin)
        else:
            raise ValueError(work)


    def rdm2(self, string):
        target = 2
        work = fqe_ops_utils.validate_rdm_string(string, target)

        if work == 'element':
            result = self.apply(sparse_hamiltonian.SparseHamiltonian(string))
            return vdot(self, result)
        elif work == 'tensor':
            rdm = self.compute_rdm(target)
            return wick(string, [rdm], self._conserve_spin)
        else:
            raise ValueError(work)


    def rdm3(self, string):
        target = 3
        work = fqe_ops_utils.validate_rdm_string(string, target)

        if work == 'element':
            result = self.apply(sparse_hamiltonian.SparseHamiltonian(string))
            return vdot(self, result)
        elif work == 'tensor':
            rdm = self.compute_rdm(target)
            return wick(string, [rdm], self._conserve_spin)
        else:
            raise ValueError(work)


    def rdm4(self, string):
        target = 4
        work = fqe_ops_utils.validate_rdm_string(string, target)

        if work == 'element':
            result = self.apply(sparse_hamiltonian.SparseHamiltonian(string))
            return vdot(self, result)
        elif work == 'tensor':
            rdm = self.compute_rdm(target)
            return wick(string, [rdm], self._conserve_spin)
        else:
            raise ValueError(work)
