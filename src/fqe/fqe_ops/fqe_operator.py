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
"""Base class for fqe operators
"""

from typing import Optional, TYPE_CHECKING

from abc import ABC, abstractmethod, abstractproperty

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction

class FqeOperator(ABC):
    """FqeOperator Base class
    """


    @abstractmethod
    def contract(self,
                 brastate: 'Wavefunction',
                 ketstate: Optional['Wavefunction']) -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate (Wavefunction) - wave function on the bra side

            ketstate (Wavefunction) - wave function on the ket side
        """
        return 0. + 0.j


    @abstractmethod
    def representation(self) -> str:
        """Return the representation of the operator
        """
        return 'fqe-operator'


    @abstractmethod
    def rank(self) -> int:
        """ Return the rank of this operator
        """
        return 0