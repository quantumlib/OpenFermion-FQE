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
"""Utility functions for FQE operators."""

import re


def validate_rdm_string(ops: str) -> str:
    """Check that a string for rdms are valid.

    Args:
        ops: String expression to be computed.

    Returns
        Either 'element' or 'tensor'.
    """

    qftops = ops.split()
    nops = len(qftops)

    assert (nops % 2) == 0

    if any(char.isdigit() for char in ops):

        creation = re.compile(r"^[0-9]+\^$")
        annihilation = re.compile(r"^[0-9]+$")

        ncre = 0
        nani = 0

        for opr in qftops:
            if creation.match(opr):
                ncre += 1
            elif annihilation.match(opr):
                nani += 1
            else:
                raise TypeError("Unsupported behavior for {}".format(ops))

        assert nani == ncre

        return "element"

    creation = re.compile(r"^[a-z]\^$")
    annihilation = re.compile(r"^[a-z]$")

    ncre = 0
    nani = 0

    for opr in qftops:
        if creation.match(opr):
            ncre += 1
        elif annihilation.match(opr):
            nani += 1
        else:
            raise TypeError("Unsupported behvior for {}.".format(ops))

    if nani != ncre:
        raise ValueError("Unequal creation and annihilation operators.")

    return "tensor"


def switch_broken_symmetry(string: str) -> str:
    """Convert the string passed in to the desired symmetry.

    Args:
        string: Input string in the original expression.

    Returns:
        Output string in the converted format.
    """
    new = ""
    if any(char.isdigit() for char in string):

        work = string.split()
        creation = re.compile(r"^[0-9]+\^$")
        annihilation = re.compile(r"^[0-9]+$")

        for opr in work:
            if creation.match(opr):
                if int(opr[:-1]) % 2:
                    val = opr[:-1]
                else:
                    val = opr
            elif annihilation.match(opr):
                if int(opr) % 2:
                    val = opr + "^"
                else:
                    val = opr
            new += val + " "
    else:
        new = string

    return new.rstrip()
