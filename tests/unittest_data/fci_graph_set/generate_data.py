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
""" fci_graph unit tests: generate reference data
"""
import os
import numpy
import pickle
from fqe.fci_graph import FciGraph
from fqe.fci_graph_set import FciGraphSet

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def generate_data(np, nalpha, nbeta, norb):
    """Generate FciGraphSet for the nelec - np space starting with
    nalpha, nbeta."""
    graphset = FciGraphSet(np, np)
    graphset.append(FciGraph(nalpha, nbeta, norb))
    if np == 1:
        graphset.append(FciGraph(nalpha - 1, nbeta, norb))
        graphset.append(FciGraph(nalpha, nbeta - 1, norb))
    elif np == 2:
        graphset.append(FciGraph(nalpha - 1, nbeta - 1, norb))
        graphset.append(FciGraph(nalpha - 2, nbeta, norb))
        graphset.append(FciGraph(nalpha, nbeta - 2, norb))
    else:
        raise Exception("This is only implemented for np < 3")

    return graphset


def regenerate_reference_data():
    """
    Regenerates the reference data
    """
    cases = [(2, 4, 3, 8), (2, 4, 4, 6)]
    for np, nalpha, nbeta, norb in cases:
        filename = "set{:02d}{:02d}{:02d}{:02d}.pickle".format(
            np, nalpha, nbeta, norb)
        with open(os.path.join(datadir, filename), 'wb') as f:
            pickle.dump(generate_data(np, nalpha, nbeta, norb), f)


if __name__ == "__main__":
    regenerate_reference_data()
