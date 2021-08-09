#   Copyright 2020 Google LLC

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http:gc/www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import fqe.settings
from fqe.settings import CodePath


def pytest_addoption(parser):
    """
    Add additional options for running C or Python tests only.
    """
    parser.addoption("--c-only", action="store_true", help="Run C tests only")
    parser.addoption("--python-only",
                     action="store_true",
                     help="Run Python tests only")


def pytest_generate_tests(metafunc):
    """
    Parametrize tests for C or Python depending on available code paths and
    command line options
    """

    if metafunc.config.getoption("--c-only") and metafunc.config.getoption(
            "--python-only"):
        raise RuntimeError("Error: --c-only and --python-only options " \
          "are mutually exclusive.")

    if "c_or_python" in metafunc.fixturenames:
        test_code_paths = list(fqe.settings.available_code_paths)

        if metafunc.config.getoption("--c-only"):
            test_code_paths.remove(CodePath.PYTHON)
        if metafunc.config.getoption("--python-only"):
            test_code_paths.remove(CodePath.C)
        metafunc.parametrize("c_or_python", test_code_paths)

    if "alpha_or_beta" in metafunc.fixturenames:
        metafunc.parametrize("alpha_or_beta", ["alpha", "beta"])
