from __future__ import annotations


__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np

import pyopencl as cl
import pytools.obj_array as obj_array
from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts

from grudge.array_context import (
    PytestPytatoPyOpenCLArrayContextFactory,
)
from grudge.dof_desc import as_dofdesc


pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPytatoPyOpenCLArrayContextFactory])

import logging

import mesh_data
import pytest

import grudge.op as op
from grudge.discretization import make_discretization_collection


logger = logging.getLogger(__name__)
from meshmode import _acf  # noqa: F401


def test_pocl7_issue(actx_factory: ArrayContextFactory):
    actx = actx_factory()

    # Doesn't hang with dim == 1
    dim = 2

    import meshmode.mesh.generation as mgen

    a = [0, 0, 0]
    b = [1, 1, 1]
    mesh = mgen.generate_regular_rect_mesh(
            a=a[:dim], b=b[:dim],
            nelements_per_axis=(3,)*dim)
    assert mesh.dim == dim

    dcoll = make_discretization_collection(actx, mesh, order=1)

    from grudge.geometry.metrics import inverse_metric_derivative
    derivs = []
    for _ in range(dim):
        derivs.append(inverse_metric_derivative(
            actx, dcoll,
            # Hang occurs regardless of what axes are used
            0, 0,
            dd=as_dofdesc("vol"),
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting))
    derivs = actx.freeze(obj_array.new_1d(derivs))
    # Doing this instead doesn't lead to hang
    # derivs = obj_array.new_1d([
    #     actx.freeze(derivs[0]),
    #     actx.freeze(derivs[1])])

    x = actx.from_numpy(np.zeros((5,), dtype=np.float64))


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
