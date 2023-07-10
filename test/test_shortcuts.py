__copyright__ = "Copyright (C) 2023 University of Illinois Board of Trustees"

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

import pytest

from pytato import DataWrapper
from meshmode.dof_array import DOFArray
from grudge.array_context import PytatoPyOpenCLArrayContext
from grudge.shortcuts import (
    euler_step,
    rk4_step,
    compiled_euler_step,
    compiled_lsrk45_step)

from grudge.array_context import (
    PytestPyOpenCLArrayContextFactory,
    PytestPytatoPyOpenCLArrayContextFactory
)
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory,
         PytestPytatoPyOpenCLArrayContextFactory])

import logging

logger = logging.getLogger(__name__)


# {{{ time integrators

@pytest.mark.parametrize(("integrator", "base_dt", "compiled", "expected_order"), [
    # TODO: Set base_dt appropriately for these based on the choice of RHS
    (euler_step, 1e-3, False, 1),
    (rk4_step, 1e-3, False, 4),
    (compiled_euler_step, 1e-3, True, 1),
    (compiled_lsrk45_step, 1e-3, True, 4),
    ])
def test_time_integrators(
        actx_factory, integrator, base_dt, compiled, expected_order):
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    # TODO: Come up with an RHS function with a known exact solution

    def rhs(t, y):
        # return ???
        return y + 0

    def y_exact(t):
        # y_np = np.array(???, dtype=np.float64)
        y_np = np.array(1, dtype=np.float64)
        return DOFArray(actx, data=(actx.from_numpy(y_np),))

    rhs_compiled = actx.compile(rhs)

    # Make sure the inputs and outputs have been evaluated
    # FIXME: It might be better to create an array context that forbids non-compiled
    # execution

    if compiled and isinstance(actx, PytatoPyOpenCLArrayContext):
        def rhs_compiled_with_input_check(t, y):
            assert isinstance(y[0], DataWrapper)
            return rhs_compiled(t, y)

        y_next = integrator(
            actx, y_exact(0), 0, base_dt, rhs_compiled_with_input_check)

        assert isinstance(y_next[0], DataWrapper)

    # Check the accuracy

    t_final = 10*base_dt

    for n in [1, 2, 4, 8]:
        dt = base_dt/n

        t = 0
        y = y_exact(0)

        # TODO: Integrate in time from t=0 to t=t_final

        rel_err = actx.to_numpy(
            (y[0] - y_exact(t_final)[0])
            / y_exact(t_final)[0])[()]
        eoc_rec.add_data_point(dt, rel_err)

    print("Error:")
    print(eoc_rec)
    assert (eoc_rec.order_estimate() >= expected_order - 0.5
                or eoc_rec.max_error() < 1e-11)

# }}}


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
