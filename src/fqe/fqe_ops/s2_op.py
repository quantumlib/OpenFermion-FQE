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
"""Implements the S^2 operator
"""

import copy

from typing import TYPE_CHECKING

from fqe.util import vdot
from fqe.fqe_ops import fqe_operator

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


class S2Operator(fqe_operator.FqeOperator):
    """:math:`S^2` operator as a specialization of FqeOperator
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
        for _, sector in out._civec.items():
            sector.apply_inplace_s2()
        return vdot(brastate, out)


    def representation(self):
        """Return the representation of the operator
        """
        return 's_2'


    def rank(self):
        """The rank of the operator
        """
        return 2
