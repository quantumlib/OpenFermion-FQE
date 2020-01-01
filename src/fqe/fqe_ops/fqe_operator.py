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
from typing import Optional

from abc import ABC, abstractmethod, abstractproperty


class FqeOperator(ABC):
    """FqeOperator Baseclass
    """


    @abstractmethod
    def contract(self,
                 brastate: 'wavefunction.Wavefunction',
                 ketstate: Optional['wavefunction.Wavefunction']) -> None:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.
        """
        pass


    @abstractproperty
    def representation(self):
        """Return the representation of the operator
        """
        return 'fqe-operator'


    @abstractmethod
    def rank(self) -> int:
        """
        """
        pass
