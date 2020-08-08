'''
    blanczos: block lanczos for gf(2)

    Copyright (c) 2020, Sebastian Wouters
    All rights reserved.

    This file is part of blanczos, licensed under the BSD 3-Clause License.
    A copy of the License can be found in the file LICENSE in the root
    folder of this project.
'''

import numpy
import ctypes
libblanczos = ctypes.cdll.LoadLibrary('../build/libblanczos.so')

libblanczos.blanczos.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                                 ctypes.c_ulonglong,
                                 ctypes.c_uint,
                                 ctypes.c_uint,
                                 numpy.ctypeslib.ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS")]
libblanczos.blanczos.restype = ctypes.c_uint

def blanczos(B, N, Nrow, Ncol):
    result = numpy.zeros([Ncol], dtype=ctypes.c_ulonglong)
    Nsol = libblanczos.blanczos(B, N, Nrow, Ncol, result)
    return (Nsol, result)

