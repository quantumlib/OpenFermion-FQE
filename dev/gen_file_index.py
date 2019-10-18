from collections import deque
import os


def main():
    source_dir = '../src/fqe'
    py_source = []
    source_contents = deque(os.listdir(source_dir))
    curr_path = source_dir
    while source_contents:
        local_path = source_contents.pop()
        path = source_dir + '/' + local_path
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


main()
