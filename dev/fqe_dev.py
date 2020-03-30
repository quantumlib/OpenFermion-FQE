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

from collections import deque
import os
import subprocess
import sys
from typing import Deque, List


def generate_file_index(sources: List[str]) -> None:
    """Generate a file index by searching through source directories for .py
    files.

    Args:
        sources (list(str)) - a list of directories to begin searching for 
            source files.

    Returns:
        None - There is no return value but it will write a file containing the
            fqe file index.
    """

    py_source: List[str] = []
    source_directories: List[str] = []
    source_contents: Deque[str] = deque()

    for path in sources:
        if not os.path.isdir(path):
            print('{} is not a directory'.format(path))
            continue
        source_directories.append(path)

    for directory in source_directories:
        curr_path = directory
        source_contents = deque(os.listdir(curr_path))
        while source_contents:
            local_path = source_contents.pop()
            path = curr_path + local_path
            if os.path.isdir(path):
                nested_contents = os.listdir(path)
                for name in nested_contents:
                    source_contents.append(local_path + '/' + name)
            elif os.path.isfile(path):
                if path[-3:] == '.py':
                    py_source.append(path)

    py_source = sorted(py_source)
    with open('./fqe_file_index.txt', 'w') as fileindex:
        for path in py_source:
            fileindex.write(path + '\n')


def run_mypy():
    """Run mypy on the files in the fqe_file_index and return the output to
    fqe_mypy.out.
    """
    mypy_path = ''
    cmd = ['which', 'mypy']
    pipe = subprocess.PIPE
    data = subprocess.run(cmd, stdout=pipe, stderr=pipe, check=True)
    mypy_path = data.stdout.decode().rstrip()

    cmd = [mypy_path, '--config=.mypy/mypy.ini', '@fqe_file_index.txt']
    pipe = subprocess.PIPE
    try:
        data = subprocess.run(cmd, stdout=pipe, stderr=pipe, check=True)
        print('mypy raised no errors.')
    except subprocess.CalledProcessError as err:
        if err.returncode == 1:
            print('Dumping mypy errors to mypy.out')
            mypyerr = err.stdout.decode()
            with open('mypy.out', 'w') as mypyout:
                mypyout.write(mypyerr)
        else:
            print('-----error-----')
            print(str(err.returncode))
            print(str(err.cmd))
            print(str(err.output))
            print(str(err.stdout))
            print(str(err.stderr))


def run_pylint():
    """Run mypy on the files in the fqe_file_index and return the output to
    fqe_mypy.out.
    """
    mypy_path = ''
    cmd = ['which', 'pylint3']
    pipe = subprocess.PIPE
    data = subprocess.run(cmd, stdout=pipe, stderr=pipe, check=True)
    pylint_path = data.stdout.decode().rstrip()

    with open('fqe_file_index.txt', 'r') as sourcefiles:
        for path in sourcefiles:
            cmd = [pylint_path, path.rstrip()]
            pipe = subprocess.PIPE
            try:
                data = subprocess.run(cmd, stdout=pipe, stderr=pipe, check=True)
                print('{} had no pylint errors or warnings'.format(path))
            except subprocess.CalledProcessError as err:
                if err.returncode != 1:
                    pylinterr = err.stdout.decode()
                    with open('pylint.out', 'a') as pylintout:
                        pylintout.write(pylinterr)
                        pylintout.write('\n\n')
                else:
                    print('-----error-----')
                    print(str(err.returncode))
                    print(str(err.cmd))
                    print(str(err.output))
                    print(str(err.stdout))
                    print(str(err.stderr))


def main():
    """A collection of devlopment tools for working in FQE
    """
    if len(sys.argv) != 2:
        print('Expected a single argument to fqe_dev.')
        return

    option = sys.argv[1]

    if option == 'mypy':
        generate_file_index(['../src/fqe/'])
        run_mypy()
    elif option == 'pylint':
        generate_file_index(['../src/fqe/'])
        run_pylint()

    return


if __name__ == '__main__':
    main()
