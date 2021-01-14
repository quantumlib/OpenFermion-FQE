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
"""TODO: Add docstring."""

# pylint: disable=line-too-long

import numpy as np


def restricted_wfn_energy():
    """TODO: Add docstring."""
    wfn = np.asarray(
        [
            [
                0.9926601546004487 + 0.0j, 0.0039419143188713 + 0.0j,
                -0.0595640503923702 + 0.0j
            ],
            [
                0.0039419143188715 + 0.0j, 0.0329080041876743 + 0.0j,
                -0.0507255530900899 + 0.0j
            ],
            [
                -0.0595640503923703 + 0.0j, -0.0507255530900902 + 0.0j,
                0.0356354834557056 + 0.0j
            ],
        ],
        dtype=np.complex128,
    )
    return wfn, 4.011381385952467
