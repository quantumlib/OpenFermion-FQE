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
"""Implements the time-reversal operator
"""

import copy
from typing import TYPE_CHECKING

import numpy

from fqe.util import vdot
from fqe.fqe_ops import fqe_operator
from fqe.util import alpha_beta_electrons

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


class TimeReversalOp(fqe_operator.FqeOperator):
    """time-reversal operator as a specialization of FqeOperator.
    The program assumes the Kramers-paired storage for the wave function.
    """


    def contract(self,
                 brastate: 'Wavefunction',
                 ketstate: 'Wavefunction') -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate (Wavefunction) - wave function on the bra side

            ketstate (Wavefunction) - wave function on the ket side
        """
        out = copy.deepcopy(ketstate)
        for (nele, nab), sector in out._civec.items():
            nalpha, nbeta = alpha_beta_electrons(nele, nab)
            if nalpha < nbeta:
                if not (nele, nbeta-nalpha) in out._civec.keys():
                    raise ValueError('The wave function space is not closed under time reversal')
                sector2 = out._civec[(nele, nbeta-nalpha)]
                tmp = numpy.copy(sector.coeff)
                phase = (-1)**(nbeta*(nalpha+1))
                phase2 = (-1)**(nalpha*(nbeta+1))
                sector.coeff = sector2.coeff.T.conj() * phase2
                sector2.coeff = tmp.T.conj() * phase
            elif nalpha > nbeta: 
                if not (nele, nbeta-nalpha) in out._civec.keys():
                    raise ValueError('The wave function space is not closed under time reversal')
            elif nalpha == nbeta:
                sector.coeff = sector.coeff.T.conj()
        return vdot(brastate, out)


    def representation(self):
        """Return the representation of the operator
        """
        return 'T'


    def rank(self):
        """The rank of the operator
        """
        return 2
