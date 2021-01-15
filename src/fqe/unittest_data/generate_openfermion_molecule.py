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
"""Helper functions for getting molecular data for unit tests."""

import os

from openfermion import MolecularData
import fqe.unittest_data as fud


def build_lih_moleculardata() -> MolecularData:
    """Returns LiH molecular data."""
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.45))]
    basis = "sto-3g"
    multiplicity = 1
    filename = os.path.join(
        fud.__file__.replace("__init__.py", ""),
        "H1-Li1_sto-3g_singlet_1.45.hdf5",
    )
    molecule = MolecularData(geometry, basis, multiplicity, filename=filename)
    molecule.load()
    return molecule


def build_h4square_moleculardata() -> MolecularData:
    """Returns H4 molecular data."""
    geometry = [
        ("H", [0.5, 0.5, 0]),
        ("H", [0.5, -0.5, 0]),
        ("H", [-0.5, 0.5, 0]),
        ("H", [-0.5, -0.5, 0]),
    ]
    basis = "sto-3g"
    multiplicity = 1
    filename = os.path.join(fud.__file__.replace("__init__.py", ""),
                            "H4_sto-3g_singlet.hdf5")
    molecule = MolecularData(geometry, basis, multiplicity, filename=filename)
    molecule.load()
    return molecule
