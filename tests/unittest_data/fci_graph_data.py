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
""" fci_graph unit tests: load reference data
"""
import os
import numpy
import pickle
from fqe.fci_graph import FciGraph

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fci_graph')


def loader(nalpha, nbeta, norb, key):
    filename = f'{nalpha:02d}{nbeta:02d}{norb:02d}.pickle'
    with open(os.path.join(datadir, filename), 'rb') as f:
        return pickle.load(f)[key]
