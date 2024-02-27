"""
.. autoclass:: PyOpenCLArrayContext
.. autoclass:: PytatoPyOpenCLArrayContext
.. autoclass:: MPIBasedArrayContext
.. autoclass:: MPIPyOpenCLArrayContext
.. autoclass:: MPINumpyArrayContext
.. class:: MPIPytatoArrayContext
.. autofunction:: get_reasonable_array_context_class
"""

__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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

# {{{ imports

from typing import (
        TYPE_CHECKING, Mapping, Tuple, Any, Callable, Optional, Type,
        FrozenSet)
from dataclasses import dataclass
from pytools.tag import Tag
from meshmode.array_context import (
        PyOpenCLArrayContext as _PyOpenCLArrayContextBase,
        PytatoPyOpenCLArrayContext as _PytatoPyOpenCLArrayContextBase)
from warnings import warn

import logging
logger = logging.getLogger(__name__)

try:
    # FIXME: temporary workaround while SingleGridWorkBalancingPytatoArrayContext
    # is not available in meshmode's main branch
    # (it currently needs
    # https://github.com/kaushikcfd/meshmode/tree/pytato-array-context-transforms)
    from meshmode.array_context import SingleGridWorkBalancingPytatoArrayContext

    try:
        # Crude check if we have the correct loopy branch
        # (https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms)
        from loopy.codegen.result import get_idis_for_kernel  # noqa
    except ImportError:
        # warn("Your loopy and meshmode branches are mismatched. "
        #      "Please make sure that you have the "
        #      "https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms "  # noqa
        #      "branch of loopy.")
        _HAVE_SINGLE_GRID_WORK_BALANCING = False
    else:
        _HAVE_SINGLE_GRID_WORK_BALANCING = True

except ImportError:
    _HAVE_SINGLE_GRID_WORK_BALANCING = False

try:
    # FIXME: temporary workaround while FusionContractorArrayContext
    # is not available in meshmode's main branch
    from meshmode.array_context import FusionContractorArrayContext

    try:
        # Crude check if we have the correct loopy branch
        # (https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms)
        from loopy.transform.loop_fusion import get_kennedy_unweighted_fusion_candidates  # noqa
    except ImportError:
        warn("Your loopy and meshmode branches are mismatched. "
             "Please make sure that you have the "
             "https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms "  # noqa
             "branch of loopy.")
        _HAVE_FUSION_ACTX = False
    else:
        _HAVE_FUSION_ACTX = True

except ImportError:
    _HAVE_FUSION_ACTX = False


from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        _PytestPytatoPyOpenCLArrayContextFactory,
        register_pytest_array_context_factory)
from arraycontext import ArrayContext
from arraycontext.container import ArrayContainer
from arraycontext.impl.pytato.compile import LazilyPyOpenCLCompilingFunctionCaller

if TYPE_CHECKING:
    import pytato as pt
    from pytato.partition import PartId
    from pytato import DistributedGraphPartition
    import pyopencl
    import pyopencl.tools
    from mpi4py import MPI


class PyOpenCLArrayContext(_PyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`. Extends it
    to understand :mod:`grudge`-specific transform metadata. (Of which there isn't
    any, for now.)
    """
    def __init__(self, queue: "pyopencl.CommandQueue",
            allocator: Optional["pyopencl.tools.AllocatorBase"] = None,
            wait_event_queue_length: Optional[int] = None,
            force_device_scalars: bool = False) -> None:

        if allocator is None:
            warn("No memory allocator specified, please pass one. "
                 "(Preferably a pyopencl.tools.MemoryPool in order "
                 "to reduce device allocations)")

        super().__init__(queue, allocator,
                         wait_event_queue_length, force_device_scalars)

# }}}


# {{{ pytato

class PytatoPyOpenCLArrayContext(_PytatoPyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PytatoPyOpenCLArrayContext`.
    Extends it to understand :mod:`grudge`-specific transform metadata. (Of
    which there isn't any, for now.)
    """
    def __init__(self, queue, allocator=None,
            *,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
             = None) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        if allocator is None:
            warn("No memory allocator specified, please pass one. "
                 "(Preferably a pyopencl.tools.MemoryPool in order "
                 "to reduce device allocations)")
        super().__init__(queue, allocator,
                compile_trace_callback=compile_trace_callback)

# }}}


class MPIBasedArrayContext:
    mpi_communicator: "MPI.Comm"


# {{{ distributed + pytato

def _to_identifier(s: str) -> str:
    return "".join(ch for ch in s if ch.isidentifier())


@dataclass(frozen=True)
class _DistributedPartProgramID:
    f: Callable[..., Any]
    part_id: Any

    def __str__(self):
        name = getattr(self.f, "__name__", "<anonymous>")
        if not name.isidentifier():
            name = _to_identifier(name)

        part = _to_identifier(str(self.part_id))
        if part:
            return f"{name}_part{part}"
        else:
            return name


class _DistributedLazilyPyOpenCLCompilingFunctionCaller(
        LazilyPyOpenCLCompilingFunctionCaller):
    def _dag_to_compiled_func(self, dict_of_named_arrays,
            input_id_to_name_in_program, output_id_to_name_in_program,
            output_template):

        import pytato as pt

        from pytools import ProcessLogger

        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank

        MPI.COMM_WORLD.barrier()
        print(f"{rank}: start concatenate")

        from pytato.transform import CopyMapper
        from pytato.analysis import get_num_nodes

        nnodes_before_copy = get_num_nodes(dict_of_named_arrays)

        dict_of_named_arrays = CopyMapper(err_on_collision=False)(dict_of_named_arrays)

        nnodes_after_copy = get_num_nodes(dict_of_named_arrays)
        print(f"_dag_to_compiled_func, {rank}: {nnodes_before_copy=}, {nnodes_after_copy=} (1)")

        nnodes_before_precompute = get_num_nodes(dict_of_named_arrays)

        with ProcessLogger(logger, "precompute_subexpressions"):
            dict_of_named_arrays = pt.precompute_subexpressions(
                dict_of_named_arrays, self.actx.freeze_thaw)

        nnodes_after_precompute = get_num_nodes(dict_of_named_arrays)
        print(f"_dag_to_compiled_func, {rank}: {nnodes_before_precompute=}, {nnodes_after_precompute=}")

        # FIXME
        from pytato.transform.calls import CallsiteCollector
        all_call_sites = CallsiteCollector(stack=())(dict_of_named_arrays)
        # function_signatures = set()
        # for call_site in all_call_sites:
        #     function = call_site.call.function
        #     function_signatures.add((
        #         next(iter(function.tags_of_type(pt.tags.FunctionIdentifier))),
        #         function.parameters))
        call_ids = set()
        for call_site in all_call_sites:
            call_ids |= call_site.call.tags_of_type(pt.tags.CallIdentifier)

        print(f"_dag_to_compiled_func: {len(all_call_sites)=}, {len(call_ids)=}")

        from pytato.array import forbid_einsums

        forbid_einsums(True)

        with ProcessLogger(logger, "concatenate_calls"):
            from pytato.analysis import get_node_counts, collect_nodes_of_type
            if rank == 3:
                node_counts_before = get_node_counts(dict_of_named_arrays)
                existing_einsums = collect_nodes_of_type(dict_of_named_arrays, pt.Einsum)
                existing_einsums2 = collect_nodes_of_type(dict_of_named_arrays, pt.Einsum)
                assert existing_einsums2 == existing_einsums
                # ncrs_before = collect_nodes_of_type(dict_of_named_arrays, pt.function.NamedCallResult)
            for cid in call_ids:
                dict_of_named_arrays = pt.concatenate_calls(
                    dict_of_named_arrays, lambda x: cid in x.call.tags)
            dict_of_named_arrays = CopyMapper(err_on_collision=True)(dict_of_named_arrays)
            if rank == 3:
                node_counts_after = get_node_counts(dict_of_named_arrays)
                for node_type in node_counts_after:
                    count_before = node_counts_before.get(node_type, 0)
                    count_after = node_counts_after[node_type]
                    print(f"{node_type}: {count_before}, {count_after}")
                after_einsums = collect_nodes_of_type(dict_of_named_arrays, pt.Einsum)
                new_einsums = after_einsums - existing_einsums
                # assert len(after_einsums & existing_einsums) > 0
                if new_einsums:
                    # print(f"Found added einsums:")
                    # for ensm in new_einsums:
                        # print(f"{ensm.non_equality_tags}")
                        # print("")
                    tb_to_existing_einsums = {}
                    for ensm in existing_einsums:
                        tb = str(next(iter(ensm.non_equality_tags)).traceback)
                        tb_to_existing_einsums.setdefault(tb, []).append(ensm)
                    tb_to_new_einsums = {}
                    for ensm in new_einsums:
                        tb = str(next(iter(ensm.non_equality_tags)).traceback)
                        tb_to_new_einsums.setdefault(tb, []).append(ensm)
                    for tb, new_ensms in tb_to_new_einsums.items():
                        print("Found new einsum created at:")
                        print(f"{tb=}")
                        print(f"{sorted([(id(e), hash(e)) for e in new_ensms])=}")
                        existing_ensms = tb_to_existing_einsums.get(tb, [])
                        if existing_ensms:
                            print(f"{sorted([(id(e), hash(e)) for e in existing_ensms])=}")
                            print("")
                # ncrs_after = collect_nodes_of_type(dict_of_named_arrays, pt.function.NamedCallResult)
                # ncrs_intersect = ncrs_before & ncrs_after
                # print(f"{[ncr.name for ncr in ncrs_before]=}")
                # print(f"{[ncr.name for ncr in ncrs_after]=}")
                # if ncrs_intersect:
                #     print(f"Found ncr intersection:")
                #     for ncr in ncrs_intersect:
                #         print(f"{ncr.non_equality_tags}")
                #         print("")

        MPI.COMM_WORLD.barrier()
        print(f"{rank}: end concatenate")

        forbid_einsums(False)

        # dict_of_named_arrays = CopyMapper(err_on_collision=True)(dict_of_named_arrays)

        from pytato.analysis import get_num_nodes
        nnodes_before_inline = get_num_nodes(dict_of_named_arrays)
        print(f"{rank}: {nnodes_before_inline=}")

        MPI.COMM_WORLD.barrier()
        print(f"{rank}: start inline")

        with ProcessLogger(logger, "inline_calls"):
            dict_of_named_arrays = pt.tag_all_calls_to_be_inlined(dict_of_named_arrays)
            dict_of_named_arrays = pt.inline_calls(dict_of_named_arrays)

        MPI.COMM_WORLD.barrier()
        print(f"{rank}: end inline")

        # dict_of_named_arrays = CopyMapper(err_on_collision=False)(dict_of_named_arrays)

        # from mpi4py import MPI
        # rank = MPI.COMM_WORLD.rank

        MPI.COMM_WORLD.barrier()

        nnodes_after_inline = get_num_nodes(dict_of_named_arrays)
        print(f"{rank}: {nnodes_after_inline=}")

        MPI.COMM_WORLD.barrier()

        self.actx._compile_trace_callback(self.f, "pre_deduplicate_data_wrappers",
                dict_of_named_arrays)

        with ProcessLogger(logger, "deduplicate_data_wrappers[pre-partition]"):
            dict_of_named_arrays = pt.transform.deduplicate_data_wrappers(
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "post_deduplicate_data_wrappers",
                dict_of_named_arrays)

        dict_of_named_arrays = CopyMapper(err_on_collision=True)(dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "pre_materialize",
                dict_of_named_arrays)

        with ProcessLogger(logger, "materialize_with_mpms[pre-partition]"):
            dict_of_named_arrays = pt.transform.materialize_with_mpms(
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "post_materialize",
                dict_of_named_arrays)

        dict_of_named_arrays = CopyMapper(err_on_collision=True)(dict_of_named_arrays)

        # FIXME: Remove the import failure handling once this is in upstream grudge
        self.actx._compile_trace_callback(self.f, "pre_infer_axes_tags",
                dict_of_named_arrays)

        with ProcessLogger(logger,
                           "transform_dag.infer_axes_tags[pre-partition]"):
            from meshmode.transform_metadata import DiscretizationEntityAxisTag
            dict_of_named_arrays = pt.unify_axes_tags(
                dict_of_named_arrays,
                tag_t=DiscretizationEntityAxisTag,
            )

        self.actx._compile_trace_callback(self.f, "post_infer_axes_tags",
                dict_of_named_arrays)

        dict_of_named_arrays = CopyMapper(err_on_collision=True)(dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "pre_find_distributed_partition",
                dict_of_named_arrays)

#        # https://github.com/inducer/pytato/pull/393 changes the function signature
#        try:
#            # pylint: disable=too-many-function-args
#            distributed_partition = pt.find_distributed_partition(
#                # pylint-ignore-reason:
#                # '_BasePytatoArrayContext' has no
#                # 'mpi_communicator' member
#                # pylint: disable=no-member
#                self.actx.mpi_communicator, dict_of_named_arrays)
#        except TypeError as e:
#            if "find_distributed_partition() takes 1 positional" in str(e):
#                distributed_partition = pt.find_distributed_partition(
#                    dict_of_named_arrays)
#            else:
#                raise
        distributed_partition = pt.find_distributed_partition(
            # pylint-ignore-reason:
            # '_BasePytatoArrayContext' has no
            # 'mpi_communicator' member
            # pylint: disable=no-member
            self.actx.mpi_communicator, dict_of_named_arrays)

        if __debug__:
            # pylint-ignore-reason:
            # '_BasePytatoArrayContext' has no 'mpi_communicator' member
            pt.verify_distributed_partition(
                self.actx.mpi_communicator,  # pylint: disable=no-member
                distributed_partition)

        self.actx._compile_trace_callback(self.f, "post_find_distributed_partition",
                distributed_partition)

        # {{{ turn symbolic tags into globally agreed-upon integers

        self.actx._compile_trace_callback(self.f, "pre_number_distributed_tags",
                distributed_partition)

        from pytato import number_distributed_tags
        prev_mpi_base_tag = self.actx.mpi_base_tag

        # type-ignore-reason: 'PytatoPyOpenCLArrayContext' has no 'mpi_communicator'
        # pylint: disable=no-member
        distributed_partition, _new_mpi_base_tag = number_distributed_tags(
                self.actx.mpi_communicator,
                distributed_partition,
                base_tag=prev_mpi_base_tag)

        assert prev_mpi_base_tag == self.actx.mpi_base_tag
        # FIXME: Updating stuff inside the array context from here is *cough*
        # not super pretty.
        self.actx.mpi_base_tag = _new_mpi_base_tag

        self.actx._compile_trace_callback(self.f, "post_number_distributed_tags",
                distributed_partition)

        # }}}

        part_id_to_prg = {}
        name_in_program_to_tags = {}
        name_in_program_to_axes = {}

        from pytato import make_dict_of_named_arrays
        for part in distributed_partition.parts.values():
            d = make_dict_of_named_arrays(
                        {var_name: distributed_partition.name_to_output[var_name]
                            for var_name in part.output_names
                         })
            (
                part_id_to_prg[part.pid],
                part_prg_name_to_tags,
                part_prg_name_to_axes
            ) = self._dag_to_transformed_pytato_prg(
                    d, prg_id=_DistributedPartProgramID(self.f, part.pid))

            assert not (set(name_in_program_to_tags.keys())
                        & set(part_prg_name_to_tags.keys()))
            assert not (set(name_in_program_to_axes.keys())
                        & set(part_prg_name_to_axes.keys()))
            name_in_program_to_tags.update(part_prg_name_to_tags)
            name_in_program_to_axes.update(part_prg_name_to_axes)

        from immutabledict import immutabledict
        return _DistributedCompiledFunction(
                actx=self.actx,
                distributed_partition=distributed_partition,
                part_id_to_prg=part_id_to_prg,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                name_in_program_to_tags=immutabledict(name_in_program_to_tags),
                name_in_program_to_axes=immutabledict(name_in_program_to_axes),
                output_template=output_template)


@dataclass(frozen=True)
class _DistributedCompiledFunction:
    """
    A callable which captures the :class:`pytato.target.BoundProgram`  resulting
    from calling :attr:`~LazilyPyOpenCLCompilingFunctionCaller.f` with a given
    set of input types, and generating :mod:`loopy` IR from it.

    .. attribute:: pytato_program

    .. attribute:: input_id_to_name_in_program

        A mapping from input id to the placeholder name in
        :attr:`CompiledFunction.pytato_program`. Input id is represented as the
        position of :attr:`~LazilyPyOpenCLCompilingFunctionCaller.f`'s argument
        augmented with the leaf array's key if the argument is an array
        container.

    .. attribute:: output_id_to_name_in_program

        A mapping from output id to the name of
        :class:`pytato.array.NamedArray` in
        :attr:`CompiledFunction.pytato_program`. Output id is represented by
        the key of a leaf array in the array container
        :attr:`CompiledFunction.output_template`.

    .. attribute:: output_template

       An instance of :class:`arraycontext.ArrayContainer` that is the return
       type of the callable.
    """

    actx: "MPISingleGridWorkBalancingPytatoArrayContext"
    distributed_partition: "DistributedGraphPartition"
    part_id_to_prg: "Mapping[PartId, pt.target.BoundProgram]"
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    name_in_program_to_tags: Mapping[str, FrozenSet[Tag]]
    name_in_program_to_axes: Mapping[str, Tuple["pt.Axis", ...]]
    output_template: ArrayContainer

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        """
        :arg arg_id_to_arg: Mapping from input id to the passed argument. See
            :attr:`CompiledFunction.input_id_to_name_in_program` for input id's
            representation.
        """

        from arraycontext.impl.pytato.compile import _args_to_device_buffers
        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array
        from arraycontext.impl.pytato.utils import get_cl_axes_from_pt_axes
        input_args_for_prg = _args_to_device_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        from pytato import execute_distributed_partition
        out_dict = execute_distributed_partition(
                self.distributed_partition, self.part_id_to_prg,
                self.actx.queue, self.actx.mpi_communicator,
                allocator=self.actx.allocator,
                input_args=input_args_for_prg)

        def to_output_template(keys, _):
            ary_name_in_prg = self.output_id_to_name_in_program[keys]
            return self.actx.thaw(to_tagged_cl_array(
                out_dict[ary_name_in_prg],
                axes=get_cl_axes_from_pt_axes(
                    self.name_in_program_to_axes[ary_name_in_prg]),
                tags=self.name_in_program_to_tags[ary_name_in_prg]))

        from arraycontext.container.traversal import rec_keyed_map_array_container
        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)


class MPIPytatoArrayContextBase(MPIBasedArrayContext):
    def __init__(
            self, mpi_communicator, queue, *, mpi_base_tag, allocator=None,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
            = None) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        if allocator is None:
            warn("No memory allocator specified, please pass one. "
                 "(Preferably a pyopencl.tools.MemoryPool in order "
                 "to reduce device allocations)")

        super().__init__(queue, allocator,
                compile_trace_callback=compile_trace_callback)

        self.mpi_communicator = mpi_communicator
        self.mpi_base_tag = mpi_base_tag

    # FIXME: implement distributed-aware freeze

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return _DistributedLazilyPyOpenCLCompilingFunctionCaller(self, f)

    def clone(self):
        # type-ignore-reason: 'DistributedLazyArrayContext' has no 'queue' member
        # pylint: disable=no-member
        return type(self)(self.mpi_communicator, self.queue,
                mpi_base_tag=self.mpi_base_tag,
                allocator=self.allocator)

# }}}


# {{{ distributed + pyopencl

class MPIPyOpenCLArrayContext(PyOpenCLArrayContext, MPIBasedArrayContext):
    """An array context for using distributed computation with :mod:`pyopencl`
    eager evaluation.

    .. autofunction:: __init__
    """

    def __init__(self,
            mpi_communicator,
            queue: "pyopencl.CommandQueue",
            *, allocator: Optional["pyopencl.tools.AllocatorBase"] = None,
            wait_event_queue_length: Optional[int] = None,
            force_device_scalars: bool = False) -> None:
        """
        See :class:`arraycontext.impl.pyopencl.PyOpenCLArrayContext` for most
        arguments.
        """
        super().__init__(queue, allocator=allocator,
                wait_event_queue_length=wait_event_queue_length,
                force_device_scalars=force_device_scalars)

        self.mpi_communicator = mpi_communicator

    def clone(self):
        # type-ignore-reason: 'DistributedLazyArrayContext' has no 'queue' member
        # pylint: disable=no-member
        return type(self)(self.mpi_communicator, self.queue,
                allocator=self.allocator,
                wait_event_queue_length=self._wait_event_queue_length,
                force_device_scalars=self._force_device_scalars)

# }}}


# {{{ distributed + pytato array context subclasses

class MPIBasePytatoPyOpenCLArrayContext(
        MPIPytatoArrayContextBase, PytatoPyOpenCLArrayContext):
    """
    .. autofunction:: __init__
    """
    pass


if _HAVE_SINGLE_GRID_WORK_BALANCING:
    class MPISingleGridWorkBalancingPytatoArrayContext(
            MPIPytatoArrayContextBase, SingleGridWorkBalancingPytatoArrayContext):
        """
        .. autofunction:: __init__
        """

    MPIPytatoArrayContext = MPISingleGridWorkBalancingPytatoArrayContext
else:
    MPIPytatoArrayContext = MPIBasePytatoPyOpenCLArrayContext


if _HAVE_FUSION_ACTX:
    class MPIFusionContractorArrayContext(
            MPIPytatoArrayContextBase, FusionContractorArrayContext):
        """
        .. autofunction:: __init__
        """

    MPIPytatoArrayContext = MPIFusionContractorArrayContext
else:
    MPIPytatoArrayContext = MPIBasePytatoPyOpenCLArrayContext

# }}}


# {{{ pytest actx factory

class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext


class PytestPytatoPyOpenCLArrayContextFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    actx_class = PytatoPyOpenCLArrayContext


# deprecated
class PytestPyOpenCLArrayContextFactoryWithHostScalars(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext
    force_device_scalars = False


register_pytest_array_context_factory("grudge.pyopencl",
        PytestPyOpenCLArrayContextFactory)
register_pytest_array_context_factory("grudge.pytato-pyopencl",
        PytestPytatoPyOpenCLArrayContextFactory)

# }}}


# {{{ actx selection


def _get_single_grid_pytato_actx_class(distributed: bool) -> Type[ArrayContext]:
    if not _HAVE_SINGLE_GRID_WORK_BALANCING:
        warn("No device-parallel actx available, execution will be slow. "
             "Please make sure you have the right branches for loopy "
             "(https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms) "  # noqa
             "and meshmode "
             "(https://github.com/kaushikcfd/meshmode/tree/pytato-array-context-transforms).")  # noqa
    # lazy, non-distributed
    if not distributed:
        if _HAVE_SINGLE_GRID_WORK_BALANCING:
            return SingleGridWorkBalancingPytatoArrayContext
        else:
            return PytatoPyOpenCLArrayContext
    else:
        # distributed+lazy:
        if _HAVE_SINGLE_GRID_WORK_BALANCING:
            return MPISingleGridWorkBalancingPytatoArrayContext
        else:
            return MPIBasePytatoPyOpenCLArrayContext


def get_reasonable_array_context_class(
        lazy: bool = True, distributed: bool = True,
        fusion: Optional[bool] = None,
        ) -> Type[ArrayContext]:
    """Returns a reasonable :class:`PyOpenCLArrayContext` currently
    supported given the constraints of *lazy* and *distributed*."""
    if fusion is None:
        fusion = lazy

    if lazy:
        if fusion:
            if not _HAVE_FUSION_ACTX:
                warn("No device-parallel actx available, execution will be slow. "
                     "Please make sure you have the right branches for loopy "
                     "(https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms) "  # noqa
                     "and meshmode "
                     "(https://github.com/kaushikcfd/meshmode/tree/pytato-array-context-transforms).")  # noqa
            # lazy+fusion, non-distributed

            if _HAVE_FUSION_ACTX:
                if distributed:
                    actx_class = MPIFusionContractorArrayContext
                else:
                    actx_class = FusionContractorArrayContext
            else:
                actx_class = _get_single_grid_pytato_actx_class(distributed)
        else:
            actx_class = _get_single_grid_pytato_actx_class(distributed)
    else:
        if fusion:
            raise ValueError("No eager actx's support op-fusion.")
        if distributed:
            actx_class = MPIPyOpenCLArrayContext
        else:
            actx_class = PyOpenCLArrayContext

    logger.info("get_reasonable_array_context_class: %s lazy=%r distributed=%r "
                "device-parallel=%r",
                actx_class.__name__, lazy, distributed,
                # eager is always device-parallel:
                (_HAVE_SINGLE_GRID_WORK_BALANCING or _HAVE_FUSION_ACTX or not lazy))
    return actx_class

# }}}


# {{{ distributed + numpy
try:
    from arraycontext import NumpyArrayContext

    class MPINumpyArrayContext(NumpyArrayContext, MPIBasedArrayContext):
        """An array context for using distributed computation with :mod:`numpy`
        eager evaluation.
        .. autofunction:: __init__
        """

        def __init__(self, mpi_communicator) -> None:
            super().__init__()
            self.mpi_communicator = mpi_communicator

        def clone(self):
            return type(self)(self.mpi_communicator)

except ImportError:
    print("Failed to import numpy array context.")
    pass
# }}}


# {{{ Tensor product array contexts

# {{{ Relevant tags

class OutputIsTensorProductDOFArrayOrdered(Tag):
    """Signify that the strides will not be of order "C" or "F". See
    :class:`grudge.array_context.TensorProductArrayContext` for more details.
    """
    pass

# }}}


# {{{ Eager TP array contexts

class TensorProductArrayContext(_PyOpenCLArrayContextBase):
    """Specialized array context for use with tensor product elements.

    The strides for the arrays containing tensor product element data are of the
    form (slow, fastest, faster, fast). These strides are not "C" or "F" order.
    Hence, this specialized array context takes care of specifying the
    particular strides required.
    """

    def transform_loopy_program(self, t_unit):
        knl = t_unit.default_entrypoint
        if knl.tags_of_type(OutputIsTensorProductDOFArrayOrdered):
            new_args = []
            for arg in knl.args:
                if arg.is_output:
                    arg = arg.copy(dim_tags=(
                        f"N{len(arg.shape)-1},"
                        + ",".join(f"N{i}"
                                   for i in range(len(arg.shape)-1))
                        ))

                new_args.append(arg)

            knl = knl.copy(args=new_args)
            t_unit = t_unit.with_kernel(knl)

        return super().transform_loopy_program(t_unit)


class TensorProductMPIPyOpenCLArrayContext(MPIPyOpenCLArrayContext,
                                         TensorProductArrayContext):
    pass

# }}}


# {{{ Lazy tensor product array contexts

class PytatoTensorProductArrayContext(PytatoPyOpenCLArrayContext):
    def transform_dag(self, dag):
        return super().transform_dag(dag)

    def transform_loopy_program(self, t_unit):
        knl = t_unit.default_entrypoint

        # {{{ adjust strides according to tensor product structure
        if knl.tags_of_type(OutputIsTensorProductDOFArrayOrdered):
            new_args = []
            for arg in knl.args:
                if arg.is_output:
                    arg = arg.copy(dim_tags=(
                        f"N{len(arg.shape)-1},"
                        + ",".join(f"N{i}"
                                   for i in range(len(arg.shape)-1))
                        ))

                new_args.append(arg)

            knl = knl.copy(args=new_args)
        # }}}

        return super().transform_loopy_program(t_unit)

# }}}


# {{{ TP fusion actx

from meshmode.array_context import FusionContractorArrayContext


class TensorProductFusionContractorArrayContext(FusionContractorArrayContext):

    def transform_loopy_program(self, t_unit):
        knl = t_unit.default_entrypoint
        if knl.tags_of_type(OutputIsTensorProductDOFArrayOrdered):
            new_args = []
            for arg in knl.args:
                if arg.is_output:
                    arg = arg.copy(dim_tags=(
                        f"N{len(arg.shape)-1},"
                        + ",".join(f"N{i}"
                                   for i in range(len(arg.shape)-1))
                        ))

                new_args.append(arg)

            knl = knl.copy(args=new_args)
            t_unit = t_unit.with_kernel(knl)

        return super().transform_loopy_program(t_unit)


class TensorProductMPIFusionContractorArrayContext(
        MPIPytatoArrayContext, TensorProductFusionContractorArrayContext):
    pass

# }}}


# }}}


# vim: foldmethod=marker
