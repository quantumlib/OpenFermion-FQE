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

from scipy.special import factorial, jv
from openfermion import FermionOperator
from openfermion.utils import is_hermitian

import numpy

from fqe.fqedata import FqeData
from fqe.util import alpha_beta_electrons, init_bitstring_groundstate
from fqe.util import sort_configuration_keys
from fqe.util import configuration_key_union
from fqe.openfermion_utils import new_wfn_from_ops
from fqe.openfermion_utils import mutate_config
from fqe.string_addressing import count_bits


class Wavefunction(object):
    """Wavefunction is the central object for manipulaion in the
    OpenFermion-FQE.
    """


    def __init__(self, param=None, conservespin=False,
                 conserveparticlenumber=False):
        """
        Args:
            param (list[list[n, ms, norb]]) - the constructor accepts a list of
              parameter lists.  The parameter lists are comprised of

              p[0] (integer) - number of particles
              p[1] (integer) - z component of spin angular momentum
              p[2] (integer) - number of spatial orbitals

            convervespin (Bool) - Set to true If the wavefunction should only
                have configs with the same m_s
            converveparticlenumber (Bool) - Set to true if the wavefunction
                should only have configs with the same number of particles

        Member Variables:
            Properties of the wavefunction are stored as dicts using the config
                key as the accessor and the corresponding FqeData value as the
                value.  These are not initialzed until they are called.
                _lena = None
                _lenb = None
                _gs_a = None
                _gs_b = None
                _nalpha = None
                _nbeta = None
                _cidim = None
            _spinconserve (Bool) - When this flag is true, the wavefunction
                will maintain a constant m_s
            _numberconserve (Bool) - When this flag is true, the wavefunction
                will maintain a constant nele
            _ncis (int) - the number of FqeData objects stored in this
                wavefunction.
            _civec (dict[(int, int)] -> FqeData) - This is a dictionary for
                FqeData objects.  The key is a tuple defined by the number of
                electrons and the spin projection of the system.
        """

        self._lena = None
        self._lenb = None
        self._gs_a = None
        self._gs_b = None
        self._nalpha = None
        self._nbeta = None
        self._cidim = None
        self._spinconserve = conservespin
        self._numberconserve = conserveparticlenumber
        self._norb = 0
        self._ncis = 0
        self._civec = {}
        if param:
            self._norb = param[0][2]
            if self._numberconserve:
                testnumber = param[0][0]
                for state in param:
                    if testnumber != state[0]:
                        raise ValueError('Inconsistent particle number passed'
                                         ' into particle number preserving '
                                         ' wavefunction.')

            if self._spinconserve:
                testnumber = param[0][1]
                for state in param:
                    if testnumber != state[1]:
                        raise ValueError('Inconsistent spin eigenfunction number'
                                         ' passed into particle number preserving'
                                         ' wavefunction.')


            for i in param:
                self.add_config(i[0], i[1], i[2])


    def __add__(self, wfn):
        """Intrinsic addition function to combine two wavefunctions.  This acts
        to iterate through the wavefunctions, combine coefficients of
        configurations they have in common and add configurations that are
        unique to each one.  The values are all combined into a new
        wavefunction object

        Args:
            wfn (wavefunction.Wavefunction) - the second wavefunction to
                add with the local wavefunction

        Returns:
            wfn (wavefunction.Wavefunction) - a new wavefunction with the
                values set by adding together values
        """
        local_keys = self._civec.keys()
        wfn_keys = wfn.configs
        param = configuration_key_union(local_keys, wfn_keys)

        newwfn = Wavefunction()
        for config in param:
            if config in local_keys and config in wfn_keys:
                data = self.get_coeff(config) + wfn.get_coeff(config)
            elif config in local_keys:
                data = self.get_coeff(config)
            elif config in wfn_keys:
                data = wfn.get_coeff(config)
            newwfn.add_config(config[0], config[1], self._norb, data)

        return newwfn


    def __sub__(self, wfn):
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
        local_keys = self._civec.keys()
        wfn_keys = wfn.configs
        param = configuration_key_union(local_keys, wfn_keys)

        newwfn = Wavefunction()
        for config in param:
            if config in local_keys and config in wfn_keys:
                data = self.get_coeff(config) - wfn.get_coeff(config)
            elif config in local_keys:
                data = self.get_coeff(config)
            elif config in wfn_keys:
                data = -wfn.get_coeff(config)
            newwfn.add_config(config[0], config[1], self._norb, data)

        return newwfn


    def __mul__(self, sval):
        """Multiply will scale the whole wavefunction by the value passed

        Args:
            sval (complex) - a number to multiply the wavefunction by

        Returns:
            nothing -mutates in place
        """
        self.scale(sval)


    def add_config(self, nele, m_s, norb, data=None):
        """Add a FqeData configuration to the wavefunction.  Discard it if it is
        unphysical.

        Args:
            nele (int) - the number of electrons in the configuration to add
            m_s (int) - the s_z quantum number of the configuration to add
            norb (int) - the number of orbitals in the configuration to add
            data (numpy.array) - an array containing data to initialize the new
                configuration with

        Returns:
            nothing - configuration is added in place
        """

        try:
            nalpha, nbeta = alpha_beta_electrons(nele, m_s)
        except ValueError:
            return

        key = (nele, m_s)
        if key in self._civec:
            if data is not None:
                raise ValueError('Configuration already exists in the '
                                 'wavefunction. It is unclear if data passed '
                                 'into the wavefunction should replace what is'
                                 ' currently set')
        else:
            self._ncis += 1
            self._civec[key] = FqeData(nalpha, nbeta, norb)
            self._norb = norb
            if data is not None:
                self._civec[key].set_wfn(strategy='from_data', raw_data=data)
            else:
                self._civec[key].set_wfn(strategy='zero')


    @property
    def configs(self):
        """Return a list of the configuration keys in the wavefunction
        """
        return self._civec.keys()


    @property
    def lena(self):
        """Return a dict of the {(nele, m_s) : lena] for each configuration
        """
        if not self._lena:
            self._lena = {}
            for key, config in self._civec.items():
                self._lena[key] = config.lena
        return self._lena


    @property
    def lenb(self):
        """Return a dict of the {(nele, m_s) : lenb] for each configuration
        """
        if not self._lenb:
            self._lenb = {}
            for key, config in self._civec.items():
                self._lenb[key] = config.lenb
        return self._lenb


    @property
    def gs_a(self):
        """Return a dict of the {(nele, m_s) : gs_a] for each configuration
        """
        if not self._gs_a:
            self._gs_a = {}
            for key, config in self._civec.items():
                self._gs_a[key] = init_bitstring_groundstate(config.nalpha)
        return self._gs_a


    @property
    def gs_b(self):
        """Return a dict of the {(nele, m_s) : gs_b] for each configuration
        """
        if not self._gs_b:
            self._gs_b = {}
            for key, config in self._civec.items():
                self._gs_b[key] = init_bitstring_groundstate(config.nbeta)
        return self._gs_b


    @property
    def nalpha(self):
        """Return a dict of the {(nele, m_s) : nalpha] for each configuration
        """
        if not self._nalpha:
            self._nalpha = {}
            for key, config in self._civec.items():
                self._nalpha[key] = config.nalpha
        return self._nalpha


    @property
    def nbeta(self):
        """Return a dict of the {(nele, m_s) : nbeta] for each configuration
        """
        if not self._nbeta:
            self._nbeta = {}
            for key, config in self._civec.items():
                self._nbeta[key] = config.nbeta
        return self._nbeta


    @property
    def norb(self):
        """Return the number of orbitals
        """
        return self._norb


    @property
    def cidim(self):
        """Return a dict of the {(nele, m_s) : cidim] for each configuration
        """
        if not self._cidim:
            self._cidim = {}
            for key, config in self._civec.items():
                self._cidim[key] = config.ci_space_length
        return self._cidim


    def get_coeff(self, key, vec=None):
        """Retrieve a vector from a configuration in the wavefunction

        Args:
            key (int, int) - a key identifying the configuration to access
            vec (int) - an integer indicating which state should be returned

        Returns:
            numpy.array(dtype=numpy.complex64)
        """
        if vec:
            return self._civec[key].coeff[:, vec]
        return self._civec[key].coeff


    def apply(self, ops):
        """Return a wavefunction subject to the creation and annhilation
        operations passed into apply.

        Arg:
            ops (FermionOperator) - Fermion operators to apply to the
                wavefunction

        Returns:
            newwfn (Wavvefunction) - a new intialized wavefunction object
        """
        newkh, partchg, spinchg = new_wfn_from_ops(ops, self.configs,
                                                   self._norb)
        if self._numberconserve and partchg:
            raise ValueError('This wavefunction is set to conserve particle'
                             ' number but apply breaks particle number')
        if self._spinconserve and spinchg:
            raise ValueError('This wavefunction is set to conserve spin'
                             ' number but apply breaks spin number')

        newwfn = Wavefunction(newkh)

        for term in ops.terms:
            for config in self._civec.values():
                gen_config = config.insequence_generator(0)
                for conf_info in gen_config:
                    newa, newb, parity = mutate_config(conf_info[1],
                                                       conf_info[2], term)
                    if newa != -1 and newb != -1 and parity != 0:
                        newval = conf_info[0]*ops.terms[term]*parity
                        newwfn.add_ele(newa, newb, newval)
        return newwfn


    def apply_generated_unitary(self, ops, algo, accuracy=1.e-7):
        """Perform the exponentialtion of fermionic algebras to the
        wavefunction according the method and accuracy.

        Args:
            ops (FermionOperator) - a FermionOperator string to apply
            algo (string) - which algorithm should we use
            time_final (float) - the final time value to evolve to
            accuracy (double) - the accuracy to which the system should be
                propagated

        Returns:
            newwfn (Wavvefunction) - a new intialized wavefunction object
        """
        algo_avail = [
            'taylor',
            'chebyshev'
        ]

        if not is_hermitian(ops):
            raise ValueError('Non-hermitian operator passed to'
                             ' apply_generated_unitary')

        if algo not in algo_avail:
            raise ValueError('{} is not availible'.format(algo))

        expand = 8
        ops *= -1.j

        if algo == 'taylor':
            apply_ops = FermionOperator('', 1.)
            for poly in range(1, expand):
                apply_ops *= ops
                wfn = self.apply(apply_ops)
                wfn.scale(1.0/factorial(poly))
                if poly == 1:
                    newwfn = wfn
                else:
                    newwfn = oldwfn + wfn
                oldwfn = newwfn

        if algo == 'chebyshev':
            t_n_m1 = FermionOperator('', 1.)
            t_n = ops
            t_n_p1 = 2*ops*t_n - t_n_m1
            for poly in range(1, expand):
                wfn = self.apply(t_n)
                scale_fac = 2.0*(1.j**poly)*jv(poly, -1.j)
                wfn.scale(scale_fac)
                if poly == 1:
                    newwfn = wfn
                else:
                    newwfn = oldwfn + wfn
                oldwfn = newwfn
                t_n_m1 = copy.copy(t_n)
                t_n = copy.copy(t_n_p1)
                t_n_p1 = 2*ops*t_n - t_n_m1
            self.scale(jv(0, 1.j))

        wfnout = self.__add__(newwfn)
        wfnout.normalize()
        return wfnout


    def max_element(self):
        """Return the largest magnitude value in the wavefunction
        """
        maxval = 0.0
        for config in self._civec.values():
            configmax = max(config.coeff.max(), config.coeff.min(), key=abs)
            maxval = max(configmax, maxval)

        return maxval


    def add_ele(self, astr, bstr, val, vec=0):
        """Add a value to an element of a configuration given a key and string
        representation.

        Args:
            key (tuple(int, int)) - the pair of (particle number, ms)
            astr (bitstring) - bitsrting of the alpha configuration
            bstr (bitstring) - bitsrting of the beta configuration
            val (complex double) - value to set
        """
        key = (count_bits(astr) + count_bits(bstr),
               count_bits(astr) - count_bits(bstr))
        try:
            self._civec[key].add_element(astr, bstr, vec, val)
        except KeyError:
            print('The a{} b{} configuration belongs to key {} which is not in'
                  ' this wavefunction'.format(astr, bstr, key))


    def set_ele(self, astr, bstr, val, vec=0):
        """Set an element of a configuration given a key and string
        representation.

        Args:
            key (tuple(int, int)) - the pair of (particle number, ms)
            astr (bitstring) - bitsrting of the alpha configuration
            bstr (bitstring) - bitsrting of the beta configuration
            val (complex double) - value to set
        """
        key = (count_bits(astr) + count_bits(bstr),
               count_bits(astr) - count_bits(bstr))
        try:
            self._civec[key].set_element(astr, bstr, vec, val)
        except KeyError:
            print('The a{} b{} configuration belongs to key {} which is not in'
                  ' this wavefunction'.format(astr, bstr, key))


    def get_ele(self, astr, bstr, vec=0):
        """Return a specific element indexed by it's bitstring accessor and
        the key to the configuration.

        Args:
            astr (bitstring) - bitsrting of the alpha configuration
            bstr (bitstring) - bitsrting of the beta configuration
            vec (int) - element to access

        Returns:
            (complex)
        """
        key = (count_bits(astr) + count_bits(bstr),
               count_bits(astr) - count_bits(bstr))
        try:
            return self._civec[key].cc_s(astr, bstr, vec)
        except KeyError:
            print('The a{} b{} configuration belongs to key {} which is not in'
                  ' this wavefunction'.format(astr, bstr, key))
            return 0. + .0j


    def set_wfn(self, vrange=None, strategy=None, raw_data=None):
        """Set the values of the ciwfn based on an argument or data to
        initalize from.

        Args:
            vrange (list[int]) - integers indicating which state of the
                wavefunction should be set
            strategy (string) - an option controlling how the values are set
            raw_data (numpy.arrau(dtype=numpy.complex64)) - data to inject into
                the configuration
        """
        if strategy == 'from_data':
            for key, data in raw_data.items():
                self._civec[key].set_wfn(vrange=vrange, strategy=strategy,
                                         raw_data=data)
        else:
            for key, config in self._civec.items():
                config.set_wfn(vrange=vrange, strategy=strategy, raw_data=None)


    def scale(self, sval):
        """ Scale each configuration space by the value sval

        Args:
            sval (complex) - value to scale by
        """
        for config in self._civec.values():
            config.scale(sval)


    def normalize(self, vec=None):
        """Iterate through each vector of each data structure such that for a
        vector |i> the Frobenius inner product is 1.  Then scale everything by
        the number of configurations
        """
        for config in self._civec.values():
            config.normalize(vec=vec)


    def generator(self):
        """Return each configuration in the wavefunction for convenient
        manipulation
        """
        for config in self._civec.values():
            yield config


    def print_wfn(self, threshold=0.001, fmt='str', states=None):
        """Print occupations and coefficients to the screen.

        Args:
            threshhold (float) - only print CI vector values such that
              |c| > threshold.
            fmt (string) - formats print according to argument
            states (int of list[int]) - an index or indexes indicating which
                states to print.
        """
        def _print_format(fmt):
            """ Select the function which will perform formatted printing
            """

            if fmt == 'occ':
                def _occupation_format(astring, bstring):
                    """occ - Prints a string indicating occupation state of each spatial
                    orbital.  A doubly occupied orbital will be indicated with "2",
                    a singly occupied orbital will get "a" or "b" depending on the
                    spin state.  An empy orbital will be ".".
                    """
                    occstr = ['.' for _ in range(max(astring.bit_length(),
                                                     bstring.bit_length()))]
                    docc = astring & bstring

                    def build_occ_value(bstr, char, occstr):
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

            def _string_format(astring, bstring):
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
        if states is None:
            vrange = [0]
        elif isinstance(states, int):
            vrange = [i for i in range(states)]
        else:
            vrange = states

        config_in_order = sort_configuration_keys(self.configs)
        for key in config_in_order:
            config = self._civec[key]
            title_print = False
            for ivec in vrange:
                g_con = config.insequence_generator(ivec)
                vec_print = False
                for base in g_con:
                    if numpy.absolute(base[0]) > threshold:
                        if not title_print:
                            title_print = True
                            print('\nConfiguration nelectrons: {} m_s: {}'
                                  .format(key[0], key[1]))
                        if not vec_print:
                            vec_print = True
                            print("Vector : {}".format(ivec))
                        conadr = print_format(base[1], base[2])
                        print(" {} : {}".format(conadr, base[0]))
