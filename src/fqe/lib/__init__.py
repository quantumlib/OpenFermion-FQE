import ctypes
import os

libname = 'libfqe.so'
parentdir = os.path.dirname(os.path.realpath(__file__))
lib_fqe = ctypes.cdll.LoadLibrary(os.path.join(parentdir, libname))


class c_double_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    @property
    def value(self):
        return self.real + 1j * self.imag
