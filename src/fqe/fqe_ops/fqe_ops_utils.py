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

import re


def validate_rdm_string(ops, target):
    """Check that a string for rdms are valid
    """

    qftops = ops.split()
    nops = len(qftops)

    if nops // 2 != target:
        return 'Incorrect number of operators parsed from {}'.format(ops)

    if nops % 2:
        return 'Odd number of operators not supported'

    if any(char.isdigit() for char in ops):
        
        creation = re.compile(r'^[0-9]+\^$')
        annihilation = re.compile(r'^[0-9]+$')

        ncre = 0
        nani = 0

        for opr in qftops:
            if creation.match(opr):
                ncre += 1
            elif annihilation.match(opr):
                nani += 1
            else:
                raise TypeError('Unsupported behvior for {}'.format(ops))

        if nani != ncre:
            raise ValueError('Unequal creation and annihilation operators')

        return 'element'

    creation = re.compile(r'^[a-z]\^$')
    annihilation = re.compile(r'^[a-z]$')

    ncre = 0
    nani = 0

    for opr in qftops:
        if creation.match(opr):
            ncre += 1
        elif annihilation.match(opr):
            nani += 1
        else:
            raise TypeError('Unsupported behvior for {}'.format(ops))

    if nani != ncre:
        raise ValueError('Unequal creation and annihilation operators')

    return 'tensor'
