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

import numpy
import fqe
import copy
from typing import List, Optional, Tuple


def wick(target: str, data: List[numpy.ndarray], spinfree: bool = True):
    """
    Original and target are written in a similar notation to OpenFermion operators.
    For example,
        target = 'i^ l k j^' 
    When spinfree is set to True, we assume that the indices are ordered 123.../123...
    The data has to be a tuple of 1-, 2-, and n-particle RDMs when target
    correspond to n-body strings
    """

    def process_string(inp: str) -> List[Tuple[str,bool,int]]:
        """ input is the string. Returns a list of indices described by index label, dagger (or not), and spin numbers 
        """
        out = []
        used = []
        rawinp = inp.split()
        if len(rawinp) % 2 == 1:
            raise ValueError('unrecognized input in wick (odd number of operators found)')

        nrank = len(rawinp) // 2
        for index in range(nrank*2):
            iop = rawinp[index]
            if not (len(iop) == 1 or (len(iop) == 2 and iop[1] == "^")):
                raise ValueError('unrecognized input in wick')
            if iop[0] in used:
                raise ValueError('unrecognized input in wick (duplicated labels)')

            dagger = len(iop) == 2 and iop[1] == "^"
            ispin = 0 if not spinfree else index % nrank 
            out.append((iop[0], dagger, ispin)) 
            used.append(iop[0])
        return out

    targ = process_string(target)
    assert len(targ) % 2 == 0
    rank = len(targ)//2

    if len(data) < rank:
        raise Exception("Problems in the input RDMs. Requested: " + str(len(targ)//2) \
                         + " Provided: " + str(len(data))) 

    # deltas = List[Tuple[str,str]]
    # operators = List[Tuple[str,bool,int]]
    # we process and store in List[Tuple[List[Tuple[str,str]], List[Tuple[str,bool,int], float]

    def process_one(current: List[Tuple[List[Tuple[int,int]], List[Tuple[str,bool,int]], float]]) -> List[Tuple[List[Tuple[int,int]], List[Tuple[str,bool,int]], float]]:
        out = []
        processedany = False 
        for (delta, ops, factor) in current:
            processed = False
            alldaggered = True
            for i in range(len(ops)):
                if ops[i][1] and not alldaggered: 
                    # swap
                    newdelta = copy.deepcopy(delta)
                    newops = copy.deepcopy(ops)
                    newops[i-1], newops[i] = newops[i], newops[i-1]
                    out.append((newdelta, newops, factor*(-1)))

                    # add deltas
                    newdelta = copy.deepcopy(delta)
                    newdelta.append((ops[i-1][0], ops[i][0]))
                    newops = copy.deepcopy(ops)
                    newops.remove(ops[i-1])
                    newops.remove(ops[i])
                    newfactor = copy.deepcopy(factor)
                    if spinfree:
                        sourcespin = ops[i][2]
                        targetspin = ops[i-1][2] 
                        if sourcespin == targetspin:
                            newfactor *= 2.0
                        else:
                            for n in range(len(newops)):
                                if newops[n][2] == sourcespin:
                                    newops[n] = (newops[n][0], newops[n][1], targetspin)
                    out.append((newdelta, newops, newfactor)) 
                    processed = True
                    break
                elif not ops[i][1]:
                    alldaggered = False
            if not processed:
                out.append((delta, ops, factor))
            else:
                processedany = True
        return out, processedany
 
    current = [([], targ, 1.0)]
    processed = True
    while processed:
        current, processed = process_one(current)

    if spinfree:
        # sort all of the terms based on spin and account for the parity
        for i in range(len(current)):
            cops = copy.deepcopy(current[i][1])
            factor = copy.deepcopy(current[i][2])
            assert len(cops) % 2 == 0
            nterms = len(cops) // 2
            processed = True
            while processed:
                processed = False
                for j in range(1, nterms):
                    if cops[j-1][2] > cops[j][2]: 
                        cops[j-1], cops[j] = cops[j], cops[j-1]
                        factor *= -1.0
                        processed = True
                    if cops[j-1+nterms][2] > cops[j+nterms][2]: 
                        cops[j-1+nterms], cops[j+nterms] = cops[j+nterms], cops[j-1+nterms]
                        factor *= -1.0
                        processed = True
            current[i] = (current[i][0], cops, factor)

    # now construct a copy that adds the RDMs
    out = numpy.zeros_like(data[rank-1])

    indices = {}
    for i in range(len(targ)):
        indices[targ[i][0]] = i
    for term in current:
        assert len(term[1]) % 2 == 0
        irank = len(term[1])//2
        sources = []
        for it in term[1]:
            sources.append(indices[it[0]])
        delta = []
        for it in term[0]:
            delta.append((indices[it[0]], indices[it[1]]))
        wickfill(out, data[irank-1] if irank > 0 else None, sources, term[2], delta)

    return out


def wickfill(target: numpy.ndarray, source: numpy.ndarray, indices: List[int], factor: float, delta: List[Tuple[int,int]]):
    """
    This function is an internal utility that fills in custom RDMs using particle RDMs. The result of Wick's theorem is passed as
    lists (indices and delta) and a factor associated with it. The results are stored in target.
    """
    norb = target.shape[0]
    srank = len(source.shape)//2 if source is not None else 0
    trank = len(target.shape)//2
    assert srank*2 == len(indices)
    if srank == 0 and trank == 1:
        assert len(delta) == 1
        for i in range(norb):
            target[i, i] += factor
    elif srank == 1 and trank == 1:
        assert len(delta) == 0
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                target[i, j] += factor * source[m[indices[0]], m[indices[1]]]
    elif srank == 0 and trank == 2:
        assert len(delta) == 2
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        if m[delta[0][0]] == m[delta[0][1]] and m[delta[1][0]] == m[delta[1][1]]:
                            target[i, j, k, l] += factor
    elif srank == 1 and trank == 2:
        assert len(delta) == 1
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        if m[delta[0][0]] == m[delta[0][1]]: 
                            target[i, j, k, l] += factor * source[m[indices[0]], m[indices[1]]]
    elif srank == 2 and trank == 2:
        assert len(delta) == 0
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        target[i, j, k, l] += factor * source[m[indices[0]], m[indices[1]], m[indices[2]], m[indices[3]]]
    elif srank == 0 and trank == 3:
        assert len(delta) == 3
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                if m[delta[0][0]] == m[delta[0][1]] and m[delta[1][0]] == m[delta[1][1]] and m[delta[2][0]] == m[delta[2][1]]:
                                    target[i, j, k, l, o, p] += factor
    elif srank == 1 and trank == 3:
        assert len(delta) == 2
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                if m[delta[0][0]] == m[delta[0][1]] and m[delta[1][0]] == m[delta[1][1]]:
                                    target[i, j, k, l, o, p] += factor * source[m[indices[0]], m[indices[1]]]
    elif srank == 2 and trank == 3:
        assert len(delta) == 1
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                if m[delta[0][0]] == m[delta[0][1]]:
                                    target[i, j, k, l, o, p] += factor * source[m[indices[0]], m[indices[1]], m[indices[2]], m[indices[3]]]
    elif srank == 3 and trank == 3:
        assert len(delta) == 0
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                target[i, j, k, l, o, p] += factor * source[m[indices[0]], m[indices[1]], m[indices[2]], m[indices[3]], m[indices[4]], m[indices[5]]]
    elif srank == 0 and trank == 4:
        assert len(delta) == 4
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                for q in range(norb):
                                    m[6] = q
                                    for r in range(norb):
                                        m[7] = r
                                        if m[delta[0][0]] == m[delta[0][1]] and m[delta[1][0]] == m[delta[1][1]] and m[delta[2][0]] == m[delta[2][1]] and m[delta[3][0]] == m[delta[3][1]]:
                                            target[i, j, k, l, o, p, q, r] += factor
    elif srank == 1 and trank == 4:
        assert len(delta) == 3
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                for q in range(norb):
                                    m[6] = q
                                    for r in range(norb):
                                        m[7] = r
                                        if m[delta[0][0]] == m[delta[0][1]] and m[delta[1][0]] == m[delta[1][1]] and m[delta[2][0]] == m[delta[2][1]]:
                                            target[i, j, k, l, o, p, q, r] += factor * source[m[indices[0]], m[indices[1]]]
    elif srank == 2 and trank == 4:
        assert len(delta) == 2
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                for q in range(norb):
                                    m[6] = q
                                    for r in range(norb):
                                        m[7] = r
                                        if m[delta[0][0]] == m[delta[0][1]] and m[delta[1][0]] == m[delta[1][1]]:
                                            target[i, j, k, l, o, p, q, r] += factor * source[m[indices[0]], m[indices[1]], m[indices[2]], m[indices[3]]]
    elif srank == 3 and trank == 4:
        assert len(delta) == 1
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                for q in range(norb):
                                    m[6] = q
                                    for r in range(norb):
                                        m[7] = r
                                        if m[delta[0][0]] == m[delta[0][1]]:
                                            target[i, j, k, l, o, p, q, r] += factor * source[m[indices[0]], m[indices[1]], m[indices[2]], m[indices[3]], m[indices[4]], m[indices[5]]]
    elif srank == 4 and trank == 4:
        assert len(delta) == 0
        m = {}
        for i in range(norb):
            m[0] = i
            for j in range(norb):
                m[1] = j
                for k in range(norb):
                    m[2] = k
                    for l in range(norb):
                        m[3] = l
                        for o in range(norb):
                            m[4] = o
                            for p in range(norb):
                                m[5] = p
                                for q in range(norb):
                                    m[6] = q
                                    for r in range(norb):
                                        m[7] = r
                                        target[i, j, k, l, o, p, q, r] += factor * source[m[indices[0]], m[indices[1]], m[indices[2]], m[indices[3]],
                                                                                          m[indices[4]], m[indices[5]], m[indices[6]], m[indices[7]]]
