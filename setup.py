#   Copyright 2020 Google

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

"""
Fermionic Quantum Emulator setup script.
"""

import io
import re
import os
import numpy as np

from setuptools import setup, find_packages, Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize


def version_number(path: str) -> str:
    """Get the FQE version number from the src directory
    """
    exp = r'__version__[ ]*=[ ]*["|\']([\d]+\.[\d]+\.[\d]+[\.dev[\d]*]?)["|\']'
    version_re = re.compile(exp)

    with open(path, 'r') as fqe_version:
        version = version_re.search(fqe_version.read()).group(1)

    return version


def main() -> None:
    """
    Perform the necessary tasks to install the Fermionic Quantum Emulator
    """
    version_path = './src/fqe/_version.py'

    __version__ = version_number(version_path)

    if __version__ is None:
        raise ValueError('Version information not found in ' + version_path)

    long_description = ('OpenFermion-FQE\n' +
                        '===============\n')
    stream = io.open('README.md', encoding='utf-8')
    stream.readline()
    long_description += stream.read()

    requirements_buffer = open('requirements.txt').readlines()
    requirements = [r.strip() for r in requirements_buffer]

    # C code extension
    config_vars = get_config_vars()
    config_vars["EXT_SUFFIX"] = '.' + config_vars["EXT_SUFFIX"].split('.')[-1]
    libdir = os.path.join("src", "fqe", "lib")
    cfiles = [
        "macros.c",
        "mylapack.c",
        "fci_graph.c",
        "fqe_data.c",
    ]
    srcs = [os.path.join(libdir, cf) for cf in cfiles]
    compile_flags = ["-O3", "-fopenmp", "-march=native", "-shared", "-fPIC"]
    link_flags = ["-lgomp"]
    libraries = []
    extensions = [Extension("fqe.lib.libfqe", srcs,
                            extra_compile_args=compile_flags,
                            extra_link_args=link_flags,
                            include_dirs=[libdir],
                            library_dirs=[libdir],
                            libraries=libraries,
                            language='c')]

    cythonfiles = [ "_linalg.pyx", ]
    srcs = [os.path.join(libdir, cf) for cf in cythonfiles]
    extensions.append(Extension("fqe.lib.linalg",
                                srcs,
                                language='c'))

    cythonfiles = [ "_fqe_data.pyx", ]
    srcs = [os.path.join(libdir, cf) for cf in cythonfiles]
    extensions.append(Extension("fqe.lib.fqe_data",
                                srcs,
                                language='c'))

    setup(
        name='fqe',
        version=__version__,
        author='The OpenFermion FQE Developers',
        author_email='help@openfermion.org',
        url='http://www.openfermion.org',
        description='OpenFermion Fermionic Quantum Emulator',
        ext_modules=cythonize(extensions,
                              compiler_directives={'language_level' : "3"}),
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=requirements,
        license='Apache 2',
        packages=find_packages(where='src'),
        package_dir={'': 'src'}
        )


main()
