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
"""Utility functions for algorithms """

import numpy
from openfermion.linalg import wedge


def valdemaro_reconstruction(tpdm, n_electrons):
    """
    Build a 3-RDM by cumulant expansion and setting 3rd cumulant to zero

    d3 approx = D ^ D ^ D + 3 (2C) ^ D

    tpdm has normalization (n choose 2) where n is the number of electrons

    Args:
        tpdm (np.ndarray): four-tensor representing the two-RDM
        n_electrons (int): number of electrons in the system
    Returns:
        six-tensor reprsenting the three-RDM
    """
    opdm = (2 / (n_electrons - 1)) * numpy.einsum('ijjk', tpdm)
    unconnected_tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
    unconnected_d3 = wedge(opdm, unconnected_tpdm, (1, 1), (2, 2))
    return 3 * wedge(tpdm, opdm, (2, 2), (1, 1)) - 2 * unconnected_d3
