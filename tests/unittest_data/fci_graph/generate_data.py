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

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def generate_data(nalpha, nbeta, norb):
    data = {}
    graph = FciGraph(nalpha, nbeta, norb)
    data['graph'] = graph
    operator_list = [[[0, 2, 3], [1, 0, 5]], [[1], [1]], [[], [4]],
                     [[0, 2, 3], [4, 5]]]
    data['make_mapping_each'] = \
        generate_data_make_mapping_each(graph, operator_list)
    data['map_to_deexc_alpha_icol'] = \
        generate_data_map_to_deexc_alpha_icol(graph)
    data['get_block_mappings'] = \
        generate_data_get_block_mappings(graph)

    return data


def generate_data_make_mapping_each(graph, operator_list):
    data = {}
    for alpha in [True, False]:
        length = graph.lena() if alpha else graph.lenb()
        for dag, undag in operator_list:
            result = numpy.zeros((length, 3), dtype=numpy.uint64)
            count = graph.make_mapping_each(result, alpha, dag, undag)
            data[(alpha, tuple(dag), tuple(undag))] = result[:count, :]
    return data


def generate_data_map_to_deexc_alpha_icol(graph):
    return graph._map_to_deexc_alpha_icol()


def generate_data_get_block_mappings(graph):
    max_states = [1, 2, 100]
    jorb = list(range(graph.norb())) + [None]
    data = {}
    for ms in max_states:
        for jo in jorb:
            data[(ms, jo)] = graph._get_block_mappings(max_states=ms, jorb=jo)
    return data


def regenerate_reference_data():
    """
    Regenerates the reference data
    """
    cases = [(4, 3, 8), (4, 4, 6), (0, 3, 7), (2, 0, 6)]
    for nalpha, nbeta, norb in cases:
        filename = f'{nalpha:02d}{nbeta:02d}{norb:02d}.pickle'
        with open(os.path.join(datadir, filename), 'wb') as f:
            pickle.dump(generate_data(nalpha, nbeta, norb), f)


if __name__ == "__main__":
    regenerate_reference_data()
