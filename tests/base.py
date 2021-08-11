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
""" Fixtures required for setting up all unit tests
"""

import unittest
import fqe.settings


class AcceleratedCodeValuesIterator:
    """ Class that permits iteration over all possible values of
        use_accelerated_code, depending on compile/runtime options.
        Required for testing both Python and C code.
        For now it returns both True and False, while setting
        use_accelerated_code to the appropriate value, but should
        check whether the package was configured with/without
        accelerated code in the future.
    """

    def __init__(self, init=False):
        self.value = int(init)

    def __iter__(self):
        fqe.settings.use_accelerated_code = bool(self.value)
        return self

    def __next__(self):
        if self.value <= 1:
            tmp = bool(self.value)
            fqe.settings.use_accelerated_code = tmp
            self.value += 1
            return tmp
        #else:
        raise StopIteration


class FQETestCase(unittest.TestCase):
    """ Base class for FQE test cases.
        Allows for easy performing of test for both accelerated
        and unaccelerated codes by iterating over
        self_accelerated_code_values
    """

    def setUp(self):
        self.accelerated_code_values = AcceleratedCodeValuesIterator()
