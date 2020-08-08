'''
    blanczos: block lanczos for gf(2)

    Copyright (c) 2020, Sebastian Wouters
    All rights reserved.

    This file is part of blanczos, licensed under the BSD 3-Clause License.
    A copy of the License can be found in the file LICENSE in the root
    folder of this project.
'''

import numpy
import blanczos
import ctypes
import random

Nrow = 4096
Ncol = Nrow + 64
density = 0.01

indices = []
for col in range(Ncol):
    for row in range(Nrow):
        if (random.random() < density):
            indices.append(row)
            indices.append(col)
B = numpy.array(indices, dtype=ctypes.c_uint)

print("Constructed B")

Nsol, result = blanczos.blanczos(B, len(B)/2, Nrow, Ncol)

