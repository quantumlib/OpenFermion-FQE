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
import sys
from typing import List

from distutils.sysconfig import get_config_vars
from distutils.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


class CustomBuildOptions(build_ext):
    """
    Build_ext subclass that handles custom per-compiler build options.
    Overrides build_extensions function which detects the compiler and architecture
    and adapts the compiling and linking flags accordingly.
    """

    def initialize_options(self):
        # default compile options for various compilers and architectures.
        # Supported compilers should be listed in the dictionary below with keys
        # in the format 'compiler-architecture'
        # If the compiler is not present, default configuration is used
        # On Apple clang, march=native and OpenMP are disabled
        self.compile_flags = { \
          'gcc-linux': ["-O3", "-fopenmp", "-march=native", "-shared", "-fPIC"], \
          'clang-linux': ["-O3", "-fopenmp", "-march=native", "-fPIC"], \
          'icc-linux': ["-O3", "-xHost", "-fPIC", "-qopenmp" ], \
          'clang-darwin': ["-O3", "-fPIC"], \
          'msvc-win32': ['/openmp', '/Ox', '/favor:INTEL64', '/Og'] \
          }
        self.link_flags = { 'gcc-linux': ["-fopenmp"], \
                            'clang-linux': ["-lomp"], \
                            'clang-darwin': [""], \
                            'msvc-win32': [""]
                            }
        super().initialize_options()

    def __not_supported_message(self, compiler: str) -> str:
        error_message = ( "The default compiler-architecture combination %s is not supported. "
                        "Supported compiler-architecture combinations: %s" ) % \
                        (compiler, " ".join(self.compile_flags.keys()))
        return error_message

    def __add_compile_flags(self, compile_flags: List[str],
                            link_flags: List[str]):
        for ext in self.extensions:
            ext.extra_compile_args = compile_flags
            ext.extra_link_args = link_flags

    def build_extensions(self):

        compiler = "unknown"
        compiler_type = self.compiler.compiler_type
        if compiler_type == "unix":
            # Check the CC environment variable. We need to check for a substring match,
            # since the actual executable name may have additional characters
            # "e.g. gcc-4, mpi-gcc or clang-11"
            env = os.getenv("CC")
            if env:
                supported_compilers = ["gcc", "clang", "icc"]
                for comp in supported_compilers:
                    if comp in env:
                        compiler = comp
            else:
                # try to select a reasonable default based on whether we
                # have Linux or Mac OS
                # This might be incorrect if custom gcc is involved on Mac
                compiler = "clang" if sys.platform == "darwin" else "gcc"

        elif compiler_type == "msvc":
            compiler = "msvc"

        compiler_arch_key = "%s-%s" % (compiler, sys.platform)

        if compiler_arch_key in self.compile_flags.keys():
            compile_flags = self.compile_flags[compiler_arch_key]
        else:
            # fall back to gcc if no specific configuration is found
            compile_flags = self.compile_flags["gcc-linux"]

        if compiler_arch_key in self.link_flags.keys():
            link_flags = self.link_flags[compiler_arch_key]
        else:
            link_flags = self.link_flags["gcc-linux"]

        # CFLAGS and LDFLAGS from the environment should be appended
        # to the current flags
        compile_flags_env = os.getenv("CFLAGS")
        link_flags_env = os.getenv("LDFLAGS")
        if compile_flags_env is not None:
            compile_flags += list(compile_flags_env.split())
        if link_flags_env is not None:
            link_flags += list(link_flags_env.split())

        self.__add_compile_flags(compile_flags, link_flags)
        super().build_extensions()


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
        "cirq_utils.c",
        "wick.c",
        "bitstring.c",
        "binom.c",
    ]
    srcs = [os.path.join(libdir, cf) for cf in cfiles]
    libraries = []
    extensions = [
        Extension("fqe.lib.libfqe",
                  srcs,
                  include_dirs=[libdir],
                  library_dirs=[libdir],
                  libraries=libraries,
                  language='c')
    ]

    cythonfiles = ["_fqe_data.pyx"]
    srcs = [os.path.join(libdir, cf) for cf in cythonfiles]
    extensions.append(Extension("fqe.lib.fqe_data", srcs, language='c'))

    setup(name='fqe',
          version=__version__,
          author='The OpenFermion FQE Developers',
          author_email='help@openfermion.org',
          url='http://www.openfermion.org',
          description='OpenFermion Fermionic Quantum Emulator',
          ext_modules=cythonize(extensions,
                                compiler_directives={'language_level': "3"}),
          long_description=long_description,
          long_description_content_type="text/markdown",
          install_requires=requirements,
          license='Apache 2',
          packages=find_packages(where='src'),
          package_dir={'': 'src'},
          cmdclass={'build_ext': CustomBuildOptions})


if __name__ == "__main__":
    main()
