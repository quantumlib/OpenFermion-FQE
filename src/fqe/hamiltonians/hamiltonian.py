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

""" Hamiltonian class for the OpenFermion-FQE.
"""

from abc import ABCMeta, abstractmethod

import numpy

class Hamiltonian(metaclass=ABCMeta):
    """ Abstract class to mediate the functions of Hamiltonian with the
    emulator.  Since the structure of the Hamiltonian may contain symmeties
    which can greatly speed up operations that act up on the object, defining
    unique classes for each case can be a key towards making the code more
    efficient.
    """

    # Variable Annotation
    h1e: numpy.ndarray
    g2e: numpy.ndarray


    @abstractmethod
    def __init__(self) -> None:
        """The abstract class cannot instantiate
        """
        raise TypeError


    @abstractmethod
    def identity(self) -> str:
        """The proper name of the Hamiltonian
        Returns a string with the name of the hamiltonian or general otherwise
        """
        pass
